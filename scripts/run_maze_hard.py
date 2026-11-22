"""Maze-hard training and evaluation for the CF family (packet WP5).

Trains CFNet (CF-WallAware by default, always cross_kernel=True) on the
Maze-hard train split -- a seeded 900/100 train/val split of the train csv
(cf.data.maze_hard.make_maze_hard_splits, seed 0) -- with cf.train.train_model,
the shared resumable training loop: per-epoch checkpoints under
<out>/checkpoints tagged with run_id and the run fingerprint (an interrupted
run resumed with the same command continues from the last completed epoch,
weights / optimizer / per-step scheduler / shuffle RNG all restored), an
optional wall-clock budget per seed, and best-val-strict model selection
(ties by val IoU). Evaluation covers the full Maze-hard test csv (1000
mazes), computed with cf.metrics.evaluate_mask / summarize.

Metrics (per maze, means over the test set unless stated; all defined by
cf.metrics.evaluate_mask, restated here per the reporting rules):

- exact (board match): the predicted cells equal the reference solution
  cell-for-cell. This is WONN's board metric. Maze-hard grids admit several
  equally short routes, so a correct alternative route scores 0 here.
- strict (valid path): the predicted cells form one wall-free simple
  start-to-goal path of the reference length - any valid shortest path.
- cell IoU: |predicted cells AND reference cells| / union of the two; how
  much of the lit region overlaps the reference route.
- path recall: the fraction of reference route cells that are predicted
  (covers_path is the binary variant: recall 1.0 with zero wall hits).
- wall hits: the number of predicted cells that sit on a wall. For
  CF-WallAware this is 0 by construction (the readout masks wall cells
  at -20), so it is reported as a sanity check, not a discriminator.

Test metrics are reported at the training horizon T_train (default 24) and
at T*: the horizon selected per seed on the validation split as argmax val
strict over the candidates {16, 24, 32, 48} (ties to the smaller T), the
same rule scripts/select_T_val.py applies to Week-2 runs.

Every number carries its budget, per the run rules: seeds planned vs
trained, epochs planned vs done per seed (a --max-minutes-per-seed budget
can stop a seed mid-run; a epoch count below plan is reported, never
hidden), and the wall-clock minutes spent (GPU minutes: 0.0 on the CPU-only
device this packet runs on; the field reports actual cuda minutes when the
run happens on GPU).

Outputs, all relative paths under --out (default results/cf_mazehard):
results.json (full metrics, config, budgets, per-seed records) and
summary.md (human summary), rewritten atomically after every seed so an
interrupted sweep keeps the finished seeds.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cf.train as train_mod                                # noqa: E402
from cf.data.maze_hard import (CONFIG, MazeHardDataset,      # noqa: E402
                              load_maze_hard, make_maze_hard_splits)
from cf.models.cf import CFNet, MODEL_SPECS                 # noqa: E402
from cf.models.rcnn import n_params                         # noqa: E402
from run_week2 import (atomic_write_json, effective_n_params,  # noqa: E402
                      git_commit, git_dirty)
from select_T_val import pick_T_star                        # noqa: E402

N_TRAIN = 900
N_VAL = 100
SPLIT_SEED = 0
T_STAR_CANDIDATES = [16, 24, 32, 48]
METRIC_FIELDS = ("exact", "strict", "iou", "recall", "covers_path", "wall_hits")
OUT_DEFAULT = "results/cf_mazehard"
METRICS_NAME = "results.json"
SUMMARY_NAME = "summary.md"


# ------------------------------------------------------------- helpers

def make_model(name, hid):
    """CF variant `name` with hid channels and cross_kernel=True (WP5)."""
    if name not in MODEL_SPECS:
        raise ValueError("unknown model %r (allowed: %s)"
                         % (name, " ".join(sorted(MODEL_SPECS))))
    return CFNet(hid=hid, cross_kernel=True, **MODEL_SPECS[name])


def model_kw(name, hid):
    """Keyword configuration recorded alongside every run."""
    return dict(hid=hid, cross_kernel=True, **MODEL_SPECS[name])


def load_data(n_train=N_TRAIN, n_val=N_VAL, split_seed=SPLIT_SEED):
    """Seeded 900/100 train/val split of the maze-hard train csv; test = full csv."""
    rows_train = load_maze_hard("train")
    train_idx, val_idx = make_maze_hard_splits(n_train, n_val=n_val,
                                               seed=split_seed,
                                               n_available=len(rows_train))
    d_train = MazeHardDataset([rows_train[i] for i in train_idx])
    d_val = MazeHardDataset([rows_train[i] for i in val_idx])
    d_test = MazeHardDataset(load_maze_hard("test"))
    return dict(train=d_train, val=d_val, test=d_test)


def config_of(args):
    return dict(model=args.model, seeds=list(args.seeds), epochs=int(args.epochs),
                T_train=int(args.T_train), hid=int(args.hid), out=args.out,
                max_minutes_per_seed=args.max_minutes_per_seed,
                T_candidates=[int(t) for t in args.T_candidates],
                n_train=N_TRAIN, n_val=N_VAL, split_seed=SPLIT_SEED)


def run_seed(model_name, seed, cfg, data, out_dir, verbose=True):
    """Train one (model, seed) and evaluate it; resumable via its checkpoint.

    Returns the run record: epochs done vs planned, stopped_early flag, raw
    and effective parameter counts, the val sweep over the T candidates,
    T* (argmax val strict, ties to the smaller T), and the full test sweeps
    (every candidate plus T_train) with all cf.metrics summary fields.
    """
    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "states").mkdir(parents=True, exist_ok=True)
    tag = "%s_seed%d" % (model_name, seed)
    train_mod.HYPERPARAMS["epochs"] = int(cfg["epochs"])
    t_train = int(cfg["T_train"])
    T_cands = sorted(set(int(t) for t in cfg["T_candidates"]))
    raw_model = make_model(model_name, cfg["hid"])
    n_raw = n_params(raw_model)
    # effective parameter count: entries with nonzero gradient on one real
    # training batch (dead weights, e.g. the discarded split head of
    # CF-NoFrag, are excluded)
    bs = train_mod.HYPERPARAMS["batch_size"]
    xb = data["train"].x[:bs].to(train_mod.DEVICE)
    yb = data["train"].y[:bs].to(train_mod.DEVICE)
    n_eff = effective_n_params(make_model(model_name, cfg["hid"]), xb, yb, t_train)
    density = float(data["train"].y.mean().item())
    pos_weight = (1.0 - density) / density
    if verbose:
        print("[%s] %d params (%d effective), T_train=%d, epochs=%d%s"
              % (tag, n_raw, n_eff, t_train, cfg["epochs"],
                 ", max %.1f min" % cfg["max_minutes_per_seed"]
                 if cfg["max_minutes_per_seed"] is not None else ""))
    model, hist = train_mod.train_model(
        lambda: make_model(model_name, cfg["hid"]), T=t_train, seed=seed,
        tag=model_name, train_ds=data["train"], val_ds=data["val"],
        pos_weight=pos_weight, results_dir=out_dir / "states",
        verbose=verbose, assert_strict_equals_exact=False,
        max_minutes=cfg["max_minutes_per_seed"],
        checkpoint_path=ckpt_dir / ("%s.ckpt.pt" % tag), run_id=tag)
    epochs_done = (int(hist.get("epochs_done"))
                   if cfg["max_minutes_per_seed"] is not None
                   and hist.get("epochs_done") is not None
                   else len(hist["train_loss"]))
    # validation horizon selection: argmax val strict, ties to the smaller T
    val_sweep = train_mod.sweep_model(model, data["val"], T_cands,
                                      assert_strict_equals_exact=False)
    val_strict = {int(T): s["strict"] for T, s in val_sweep.items()}
    T_star = int(pick_T_star(val_strict))
    test_sweep = train_mod.sweep_model(model, data["test"],
                                       sorted(set(T_cands) | {t_train}),
                                       assert_strict_equals_exact=False)
    return dict(
        model=model_name, seed=int(seed), status="done", tag=tag,
        model_kw=model_kw(model_name, cfg["hid"]),
        params_raw=int(n_raw), params_effective=int(n_eff),
        epochs_done=epochs_done, epochs_planned=int(cfg["epochs"]),
        stopped_early=bool(hist.get("stopped_early", False)),
        best_epoch=hist.get("best_epoch"), T_train=t_train,
        pos_weight=float(pos_weight),
        train_seconds=float(hist.get("seconds", hist.get("train_time_s"))),
        T_star=T_star,
        val_strict_by_T={str(T): val_strict[T] for T in sorted(val_strict)},
        val_sweeps={str(T): val_sweep[T] for T in sorted(val_sweep)},
        T_train_metrics=test_sweep[t_train],
        T_star_metrics=test_sweep[T_star],
        test_sweeps={str(T): test_sweep[T] for T in sorted(test_sweep)},
    )


def _mean_std(vals):
    if not vals:
        return None, 0.0
    return (float(statistics.mean(vals)),
            float(statistics.stdev(vals)) if len(vals) > 1 else 0.0)


def aggregate_per_model(runs, models):
    """Per-model mean / sample std over seeds; preliminary flag when < 3 seeds."""
    out = {}
    for name in models:
        recs = sorted((r for r in runs.values()
                       if r.get("model") == name and r.get("status") == "done"),
                      key=lambda r: r["seed"])
        agg = dict(n_seeds=len(recs), seeds=[r["seed"] for r in recs],
                   preliminary=len(recs) < 3,
                   epochs_done=[r["epochs_done"] for r in recs],
                   T_star=[r["T_star"] for r in recs],
                   train_seconds=[r["train_seconds"] for r in recs])
        for metric in METRIC_FIELDS:
            for label, src in (("T_train", "T_train_metrics"),
                               ("T_star", "T_star_metrics")):
                mean, std = _mean_std([r[src][metric] for r in recs])
                agg["%s_at_%s" % (metric, label)] = mean
                agg["%s_at_%s_std" % (metric, label)] = std
        out[name] = agg
    return out


# ------------------------------------------------------------------ outputs

def _fmt(x, digits=4):
    if x is None:
        return "-"
    return ("%.*f" % (digits, x)).rstrip("0").rstrip(".")


def build_payload(cfg, dataset_sizes, runs, per_model, seconds_total):
    device = train_mod.DEVICE
    minutes = seconds_total / 60.0
    gpu_minutes = minutes if device.type == "cuda" else 0.0
    planned_seeds = cfg["seeds"]
    done_seeds = sorted(r["seed"] for r in runs.values()
                        if r.get("status") == "done")
    return dict(
        experiment=("CF family (packet WP5) on Maze-hard: CFNet "
                    "(cross_kernel=True) trained on %d mazes, %s evaluated "
                    "on the %d-maze test set"
                    % (dataset_sizes["train"], cfg["model"],
                       dataset_sizes["test"])),
        dataset=dict(hf_id=CONFIG["hf_id"], size=CONFIG["size"],
                     n_train=dataset_sizes["train"], n_val=dataset_sizes["val"],
                     n_test=dataset_sizes["test"],
                     splits=dict(train=N_TRAIN, val=N_VAL, seed=SPLIT_SEED,
                                 test="full test csv",
                                 note=("seeded permutation of the train csv; "
                                       "no dihedral augmentation anywhere "
                                       "(keeps the resumable checkpoint "
                                       "trajectory exact)")),
                     verified_by="cf.data.maze_hard validation, reported by check_maze_hard runs"),
        model=dict(cls="cf.models.cf.CFNet", model=cfg["model"],
                   params_note=("params_raw / params_effective per run: raw "
                                "counts all weights, effective counts the "
                                "entries with nonzero gradient on one "
                                "training batch"),
                   variants={cfg["model"]: model_kw(cfg["model"], cfg["hid"])}),
        training=dict(epochs_planned=int(cfg["epochs"]),
                      T_train=int(cfg["T_train"]),
                      max_minutes_per_seed=cfg["max_minutes_per_seed"],
                      model_selection="best val strict (ties: val IoU)",
                      hyperparams=dict(train_mod.HYPERPARAMS),
                      resume=("per-seed checkpoints under <out>/checkpoints "
                              "(run_id + run-fingerprint tagged): an "
                              "interrupted run re-invoked with the same "
                              "command resumes from the last completed epoch")),
        T_star=dict(candidates=[int(t) for t in cfg["T_candidates"]],
                    rule="argmax val strict over candidates, ties to the smaller T",
                    per_seed=True, selected_from="the 100-maze validation split only"),
        accuracy_definition=dict(
            board_match=("predicted cells equal the reference solution "
                         "cell-for-cell (cf.metrics 'exact', the metric of "
                         "the WONN reference code)"),
            valid_path=("predicted cells form one wall-free simple "
                        "start-to-goal path of the reference length "
                        "(cf.metrics 'strict'); on Maze-hard this is NOT "
                        "equal to board match, grids admit several equally "
                        "short routes"),
            cell_iou="|predicted AND reference| / |predicted OR reference| (cf.metrics iou)",
            path_recall="fraction of reference route cells predicted (cf.metrics recall)",
            wall_hits=("number of predicted cells on a wall; 0 for "
                       "CF-WallAware by construction (wall readout masked "
                       "at -20)"),
        ),
        config=cfg,
        budget=dict(seeds_planned=list(planned_seeds), seeds_done=done_seeds,
                    preliminary=(len(done_seeds) < 3),
                    preliminary_note=("fewer than 3 seeds finished: all "
                                      "numbers on this page are preliminary"
                                      if len(done_seeds) < 3 else None),
                    device=str(device),
                    total_seconds=seconds_total,
                    wall_minutes=minutes,
                    gpu_minutes=gpu_minutes,
                    gpu_minutes_note=("cuda minutes actually used by training "
                                      "and evaluation" if device.type == "cuda"
                                      else "0.0: CPU-only run (the packet's "
                                           "run rules allow CPU only); the "
                                           "compute budget spent is the wall "
                                           "time")),
        runs={tag: dict(rec) for tag, rec in runs.items()},
        per_model=per_model,
        wall_time=dict(total_seconds=seconds_total),
        torch_version=torch.__version__,
        git_commit=git_commit(),
        git_dirty=git_dirty(cfg.get("out")),
    )


def render_summary(payload):
    cfg = payload["config"]
    runs = payload["runs"]
    per = payload["per_model"]
    bud = payload["budget"]
    lines = []
    lines.append("# CF family on the hard mazes (packet WP5): results")
    lines.append("")
    lines.append("## What was run")
    lines.append("")
    lines.append("CFNet (%s, cross_kernel=True) trained with "
                 "cf.train.train_model on %d Maze-hard train mazes "
                 "(%d/%d seeded split, seed %d, no augmentation), "
                 "T_train = %d, %d epochs planned, batch %d, Adam lr %g, "
                 "%d hidden channels; %s per seed. Evaluation: the full "
                 "%d-maze Maze-hard test csv with cf.metrics. T* = argmax "
                 "val strict over %s (ties to the smaller T), selected per "
                 "seed on the validation split only."
                 % (cfg["model"], payload["dataset"]["n_train"], N_TRAIN,
                    N_VAL, SPLIT_SEED, cfg["T_train"], cfg["epochs"],
                    payload["training"]["hyperparams"]["batch_size"],
                    payload["training"]["hyperparams"]["lr"], cfg["hid"],
                    ("max %.1f wall minutes" % cfg["max_minutes_per_seed"]
                     if cfg["max_minutes_per_seed"] is not None
                     else "no per-seed wall-clock cap"),
                    payload["dataset"]["n_test"], cfg["T_candidates"]))
    lines.append("")
    lines.append("Metric definitions (fuller statements in results.json and "
                 "the script docstring): **board match** = predicted cells "
                 "equal the reference solution cell-for-cell (cf.metrics "
                 "exact, the WONN reference metric); **valid path** = the "
                 "predicted cells form any wall-free simple start-to-goal "
                 "path of the reference length (cf.metrics strict - NOT "
                 "equal to board match here, Maze-hard grids admit several "
                 "equally short routes); **cell IoU** = overlap of the "
                 "predicted cells with the reference route over their "
                 "union; **path recall** = fraction of reference route "
                 "cells predicted; **wall hits** = predicted cells sitting "
                 "on a wall (0 for CF-WallAware by construction).")
    lines.append("")
    lines.append("## Test set (%d mazes), mean over seeds"
                 % payload["dataset"]["n_test"])
    lines.append("")
    lines.append("| model | seeds | params raw/effective | epochs done/planned | "
                 "T* per seed | board @T_train / @T* | valid path @T_train / @T* | "
                 "IoU @T* | recall @T* | wall hits @T* |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    first = next(iter(runs.values())) if runs else None
    for name, agg in per.items():
        n_par = ("%s/%s" % (_fmt(first["params_raw"]), _fmt(first["params_effective"]))
                 if first else "-")
        eps = ("%s/%d" % ("/".join(str(e) for e in agg["epochs_done"]),
                          cfg["epochs"]) if agg["epochs_done"] else "-")
        lines.append("| %s | %d | %s | %s | %s | %s / %s | %s / %s | %s | %s | %s |"
                     % (name, agg["n_seeds"], n_par, eps,
                        ", ".join(str(t) for t in agg["T_star"]),
                        _fmt(agg["exact_at_T_train"]), _fmt(agg["exact_at_T_star"]),
                        _fmt(agg["strict_at_T_train"]), _fmt(agg["strict_at_T_star"]),
                        _fmt(agg["iou_at_T_star"], 3),
                        _fmt(agg["recall_at_T_star"], 3),
                        _fmt(agg["wall_hits_at_T_star"], 2)))
        if agg.get("preliminary"):
            lines.append("")
            lines.append("Preliminary: fewer than 3 seeds finished for this "
                         "model - the numbers above carry no seed spread.")
    lines.append("")
    lines.append("## Per-seed detail (test set)")
    lines.append("")
    lines.append("| model | seed | epochs done | stopped early | best epoch | T* | "
                 "board @T_train | valid @T_train | board @T* | valid @T* | train s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rec in sorted(runs.values(), key=lambda r: (r["model"], r["seed"])):
        m1, ms = rec["T_train_metrics"], rec["T_star_metrics"]
        lines.append("| %s | %d | %d/%d | %s | %s | %d | %s | %s | %s | %s | %s |"
                     % (rec["model"], rec["seed"], rec["epochs_done"],
                        rec["epochs_planned"], rec["stopped_early"],
                        rec["best_epoch"] if rec["best_epoch"] is not None else "-",
                        rec["T_star"], _fmt(m1["exact"]), _fmt(m1["strict"]),
                        _fmt(ms["exact"]), _fmt(ms["strict"]),
                        _fmt(rec["train_seconds"], 1)))
    lines.append("")
    lines.append("## Budget and provenance")
    lines.append("")
    lines.append("Seeds: %d planned (%s), %d trained. Epochs: %d planned; "
                 "any shortfall is a wall-clock budget stop, reported per "
                 "seed above, never hidden. Total wall time: %s minutes "
                 "(%.1f s). GPU minutes: %s - %s. Resumable per-seed "
                 "checkpoints sit under <out>/checkpoints, so an "
                 "interrupted sweep re-invoked with the same command "
                 "continues from the last completed epoch. Everything runs "
                 "on device '%s'."
                 % (len(cfg["seeds"]), ", ".join(str(s) for s in cfg["seeds"]),
                    len(bud["seeds_done"]), cfg["epochs"],
                    _fmt(bud["wall_minutes"], 2), bud["total_seconds"],
                    _fmt(bud["gpu_minutes"], 2), bud["gpu_minutes_note"],
                    bud["device"]))
    lines.append("")
    return "\n".join(lines)


def write_outputs(out_dir, payload):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rj = atomic_write_json(out_dir / METRICS_NAME, payload)
    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as fh:
        fh.write(render_summary(payload))
    return rj, out_dir / SUMMARY_NAME


# --------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="CF-WallAware",
                    choices=sorted(MODEL_SPECS),
                    help="CF variant, always cross_kernel=True (default: CF-WallAware)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="training seeds in execution order (default: 0 1 2)")
    ap.add_argument("--epochs", type=int, default=40,
                    help="planned training epochs (default: 40)")
    ap.add_argument("--T-train", type=int, default=24,
                    help="training horizon T (default: 24)")
    ap.add_argument("--hid", type=int, default=16,
                    help="hidden channels (default: 16)")
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help="output directory (default: %s)" % OUT_DEFAULT)
    ap.add_argument("--max-minutes-per-seed", type=float, default=None,
                    help="wall-clock budget per seed; a seed stopped by it "
                         "reports epochs_done < epochs and stopped_early=True")
    ap.add_argument("--T-candidates", nargs="+", type=int,
                    default=list(T_STAR_CANDIDATES),
                    help="T* candidates (default: %s)"
                         % " ".join(str(t) for t in T_STAR_CANDIDATES))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = config_of(args)
    t_start = time.perf_counter()
    print("loading maze-hard splits (%d/%d seeded split, test = full csv)"
          % (N_TRAIN, N_VAL))
    data = load_data(cfg["n_train"], cfg["n_val"], cfg["split_seed"])
    dataset_sizes = {k: len(v) for k, v in data.items()}
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    for seed in cfg["seeds"]:
        runs["%s_seed%d" % (cfg["model"], seed)] = run_seed(
            cfg["model"], seed, cfg, data, out_dir)
        payload = build_payload(cfg, dataset_sizes, runs,
                                aggregate_per_model(runs, [cfg["model"]]),
                                time.perf_counter() - t_start)
        write_outputs(out_dir, payload)     # finished seeds survive an interrupt
    payload = build_payload(cfg, dataset_sizes, runs,
                            aggregate_per_model(runs, [cfg["model"]]),
                            time.perf_counter() - t_start)
    rj, sm = write_outputs(out_dir, payload)
    print("wrote %s and %s" % (rj, sm))
    print(render_summary(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
