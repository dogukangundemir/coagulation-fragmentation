"""Validation-horizon selection on saved Week-2 runs.

For every finished (model, seed) run in <results>/week2_metrics.json, reloads
the best-val state <results>/<Model>_seed<k>.pt into a freshly built model
(same factory as scripts/run_week2.py), sweeps the inference horizon on the VAL
split over the candidate T values, and picks T* = argmax val strict (ties go
to the smaller T).  Reports test strict at the training horizon and at T*,
plus probe-7/9 strict at T*.  T-sweep entries already stored in the JSON are
reused; horizons missing from it are recomputed with cf.train.sweep_model.
Writes <results>/valT_selection.json atomically and prints a summary table.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cf.train as train_mod
import run_week2

DEFAULT_T = [4, 8, 12, 16, 24]
OUT_NAME = "valT_selection.json"


# ---- helpers (unit-tested in tests/test_select_T_val.py) ----

def pick_T_star(val_strict):
    """Horizon with the highest val strict; ties go to the smaller T."""
    best_T, best_v = None, -1.0
    for T in sorted(val_strict):
        if val_strict[T] > best_v:
            best_T, best_v = T, val_strict[T]
    return best_T


def stored_sweeps(run):
    """{split: {T: summary}} from a run record (JSON string keys rebuilt as int)."""
    out = {}
    for split, sw in (run.get("sweeps") or {}).items():
        try:
            out[split] = {int(k): v for k, v in sw.items()}
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def metrics_at_T(run, split, T, T_train, model=None, data=None):
    """Summary at horizon T: reuse the run's stored sweeps / recorded test
    result; otherwise recompute with sweep_model (needs model and data)."""
    stored = stored_sweeps(run).get(split, {})
    if T in stored:
        return stored[T]
    if split == "test" and T == T_train and run.get("test") is not None:
        return run["test"]
    if model is None or data is None:
        raise KeyError("no stored %s sweep at T=%s and no model/data to compute it"
                       % (split, T))
    return train_mod.sweep_model(model, data[split], [T])[T]


def load_model_state(results_dir, run):
    """Fresh model of the run's variant with the best-val state loaded in."""
    path = Path(results_dir) / ("%s_seed%d.pt" % (run["model"], run["seed"]))
    if not path.exists():
        sys.exit("missing best-state checkpoint %s; re-run scripts/run_week2.py" % path)
    model = run_week2.make_model(run["model"])
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    return model


def done_runs(runs):
    """Finished (model, seed, run) records, model order as in the JSON."""
    names = []
    for run in runs.values():
        if run.get("status") == "done" and run.get("model") not in names:
            names.append(run["model"])
    out = []
    for name in names:
        seeds = sorted(r["seed"] for r in runs.values()
                       if r.get("status") == "done" and r.get("model") == name)
        for seed in seeds:
            run = runs["%s_seed%d" % (name, seed)]
            out.append((name, seed, run))
    return out


def select_for_run(results_dir, run, Ts, data, T_train, verbose=True):
    """One run's selection record: val sweep, T*, test/probe metrics."""
    model = load_model_state(results_dir, run)
    val_sweep = train_mod.sweep_model(model, data["val"], Ts)
    val_strict = {int(T): s["strict"] for T, s in val_sweep.items()}
    T_star = pick_T_star(val_strict)
    rec = dict(
        model=run["model"], seed=run["seed"], T_star=T_star,
        val_strict_by_T={str(T): val_strict[T] for T in sorted(val_strict)},
        val_strict_at_T_star=val_strict[T_star],
        test_strict_at_T_train=metrics_at_T(run, "test", T_train, T_train)["strict"],
        test_strict_at_T_star=metrics_at_T(run, "test", T_star, T_train,
                                           model=model, data=data)["strict"],
        probe7_strict_at_T_star=metrics_at_T(run, "probe7", T_star, T_train,
                                             model=model, data=data)["strict"],
        probe9_strict_at_T_star=metrics_at_T(run, "probe9", T_star, T_train,
                                             model=model, data=data)["strict"])
    if verbose:
        print("[%s_seed%d] T*=%d val strict=%.4f | test@%d %.4f | test@T* %.4f"
              % (rec["model"], rec["seed"], T_star, rec["val_strict_at_T_star"],
                 T_train, rec["test_strict_at_T_train"], rec["test_strict_at_T_star"]))
    return rec


def aggregate_per_model(selections, models):
    """Per-model mean and sample std of test strict@T* over seeds."""
    per_model = {}
    for name in models:
        vals = [r["test_strict_at_T_star"] for r in selections if r["model"] == name]
        per_model[name] = dict(
            n_seeds=len(vals),
            mean_test_strict_at_T_star=statistics.mean(vals) if vals else None,
            std_test_strict_at_T_star=statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return per_model


def format_table(selections, per_model, models, T_train):
    """The printed report: per-run rows, then per-model mean +- sample std."""
    lines = ["%-14s %5s %4s %14s %15s %15s"
             % ("model", "seed", "T*", "val strict@T*",
                "test strict@%d" % T_train, "test strict@T*")]
    lines.append("-" * len(lines[0]))
    for rec in selections:
        lines.append("%-14s %5d %4d %14.4f %15.4f %15.4f"
                     % (rec["model"], rec["seed"], rec["T_star"],
                        rec["val_strict_at_T_star"], rec["test_strict_at_T_train"],
                        rec["test_strict_at_T_star"]))
    lines.append("")
    for name in models:
        agg = per_model[name]
        if agg["n_seeds"] == 0:
            lines.append("%-14s no finished runs" % name)
        else:
            lines.append("%-14s test strict@T* over %d seed(s): %.4f +- %.4f"
                         % (name, agg["n_seeds"], agg["mean_test_strict_at_T_star"],
                            agg["std_test_strict_at_T_star"]))
    return "\n".join(lines)


def write_selection(path, payload):
    """Atomically write the selection JSON (no partial file on crash)."""
    return run_week2.atomic_write_json(path, payload)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="folder with week2_metrics.json and the *_seed<k>.pt states")
    ap.add_argument("--T", nargs="+", type=int, default=list(DEFAULT_T),
                    help="candidate horizons (default: %s)" % " ".join(map(str, DEFAULT_T)))
    ap.add_argument("--models", nargs="+", default=None,
                    help="models to include (default: all finished runs in the JSON)")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    results = Path(args.results)
    path = run_week2.metrics_path(results)
    if not path.exists():
        sys.exit("no %s in %s" % (run_week2.METRICS_NAME, results))
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    cfg = payload["config"]
    T_train = cfg["T_train"]
    runs = [(name, seed, run) for name, seed, run in done_runs(payload["runs"])
            if args.models is None or name in args.models]
    order = [name for name, _, _ in runs]
    models = (args.models if args.models is not None
              else sorted(set(order), key=order.index))
    print("loading data (same pipeline as run_week2.py) ...")
    data = run_week2.load_data(cfg)
    selections = []
    for name, seed, run in runs:
        selections.append(select_for_run(results, run, args.T, data, T_train))
    per_model = aggregate_per_model(selections, models)
    out = dict(source=str(path), T_candidates=list(args.T), T_train=T_train,
               selection_rule="T* = argmax val strict over candidates (ties: smaller T)",
               selections=selections, per_model=per_model)
    out_path = write_selection(results / OUT_NAME, out)
    print("\n%s" % format_table(selections, per_model, models, T_train))
    print("\nselection written to %s" % out_path)


if __name__ == "__main__":
    main()
