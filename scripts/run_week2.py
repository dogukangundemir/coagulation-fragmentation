"""Week-2 comparison runner: RecurrentCNN vs CF variants on the curriculum.

Trains every (model, seed) pair with cf.train.train_model under a shared
wall-clock budget, then records test metrics, T-sweeps (test + probe 7/9) and
settling curves into --out/week2_metrics.json.  The JSON is rewritten
atomically after every run, so an interrupted sweep can be resumed: runs whose
tag already has a finished result are skipped.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cf.train as train_mod
from cf.data.curriculum import (build_datasets, load_curriculum, make_splits,
                                train_pos_weight)
from cf.models.cf import CFNet, MODEL_SPECS
from cf.models.rcnn import RecurrentCNN, n_params

ALLOWED_MODELS = ["RecurrentCNN", "CF-Full", "CF-NoFrag", "CF-NoBound", "CF-WallAware"]
METRICS_NAME = "week2_metrics.json"


# ---- data / config ----

FAST = dict(n_train=200, n_val=100, n_test=100, n_probe=100, epochs=2,
            seeds=[0], sweep_T=[1, 4, 8])


def config_of(args):
    """Effective config after --fast overrides (as stored in the JSON)."""
    if not args.fast:
        return dict(models=args.models, seeds=args.seeds, T_train=args.T_train,
                    sweep_T=args.sweep_T, budget_minutes=args.budget_minutes,
                    out=args.out, fast=False)
    return dict(models=args.models, seeds=FAST["seeds"], T_train=args.T_train,
                sweep_T=FAST["sweep_T"], budget_minutes=args.budget_minutes,
                out=args.out, fast=True, n_train=FAST["n_train"],
                n_val=FAST["n_val"], n_test=FAST["n_test"], n_probe=FAST["n_probe"])


def load_data(cfg):
    """Week-1 pipeline: level-5 train/val/test, probe levels 7 and 9."""
    mazes, sols, levels = load_curriculum()
    kw = dict(n_train=cfg.get("n_train", 4000), n_val=cfg.get("n_val", 1000),
              n_test=cfg.get("n_test", 1000), n_probe=cfg.get("n_probe", 1000))
    splits, *_ = make_splits(mazes, sols, levels, **kw)
    ds = build_datasets(splits, mazes, sols)
    data = dict(train=ds["train"], val=ds["val"], test=ds["test"],
                probe7=ds["probe"][7], probe9=ds["probe"][9])
    return data


def new_payload(cfg, data):
    sizes = {k: len(v) for k, v in data.items()}
    sizes["pos_weight"] = train_pos_weight(data["train"])
    return dict(config=cfg, hyperparams=dict(train_mod.HYPERPARAMS),
                device=str(train_mod.DEVICE),
                device_name=(torch.cuda.get_device_name(0)
                             if train_mod.DEVICE.type == "cuda" else None),
                torch_version=torch.__version__, git_commit=git_commit(),
                git_dirty=git_dirty(cfg.get("out")),
                dataset_sizes=sizes, runs={})


def metrics_path(out):
    return Path(out) / METRICS_NAME


# ---- helpers (unit-tested in tests/test_run_week2.py and
# ---- tests/test_effective_params.py) ---

def split_budget(remaining_minutes, remaining_runs):
    """Per-run wall-clock slice: remaining budget shared over remaining runs."""
    if remaining_runs <= 0:
        return 0.0
    return remaining_minutes / remaining_runs


def budget_allows(remaining_minutes, min_minutes=1.0):
    """Never start a new training run with less than `min_minutes` left."""
    return remaining_minutes >= min_minutes


def run_plan(models, seeds):
    """[(model_name, seed, tag)] in execution order."""
    return [(name, seed, "%s_seed%d" % (name, seed))
            for name in models for seed in seeds]


def run_done(payload, tag):
    """Resume support: True if the JSON already holds a finished result."""
    run = payload.get("runs", {}).get(tag)
    return bool(run) and run.get("status") == "done"


RESUME_CFG_KEYS = ["models", "seeds", "T_train", "sweep_T", "fast"]


def config_mismatch(saved_cfg, cfg):
    """Keys in which an existing JSON's config differs from the current one."""
    if not isinstance(saved_cfg, dict):
        return list(RESUME_CFG_KEYS)
    return [k for k in RESUME_CFG_KEYS if saved_cfg.get(k) != cfg.get(k)]


def make_model(name):
    if name == "RecurrentCNN":
        return RecurrentCNN(hid=22)
    if name not in MODEL_SPECS:
        raise ValueError("unknown model %r (allowed: %s)" % (name, ALLOWED_MODELS))
    return CFNet(hid=16, **MODEL_SPECS[name])


def effective_n_params(model, x, y, T, pos_weight=None):
    """Element-wise parameter count that training actually reaches.

    Runs one training-style forward+backward (BCEWithLogits, the
    cf.train.HYPERPARAMS loss) on the batch (x, y) and counts the
    individual parameter ENTRIES with a nonzero gradient: raw n_params also
    counts dead weights that never receive gradient (e.g. the split-flux
    output head of CF-NoFrag is computed but immediately discarded with
    fragmentation=False).  The model is first moved to the batch's device
    (training batches live on cf.train.DEVICE, freshly built models on
    CPU).  Returns the count; the model is left on that device with zeroed
    gradients.
    """
    model.to(x.device)
    model.zero_grad(set_to_none=True)
    if pos_weight is not None:
        pos_weight = torch.tensor(float(pos_weight), device=x.device)
    loss = F.binary_cross_entropy_with_logits(model(x, T=T), y,
                                              pos_weight=pos_weight)
    loss.backward()
    n_eff = sum(int(p.grad.ne(0).sum()) for p in model.parameters()
                if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return n_eff


def atomic_write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(str(tmp), str(path))
    return path


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _status_path(line):
    """Path part of one `git status --porcelain` line (quotes stripped)."""
    path = line[3:]
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    return path


def git_dirty(out_dir=None):
    """True if the worktree differs from HEAD; untracked files under
    out_dir (run outputs) are ignored. None if git is unavailable."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, check=True)
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    ignore_prefix = None
    if out_dir is not None:
        try:
            ignore_prefix = Path(out_dir).resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass                            # out_dir outside the repo: nothing to ignore
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], _status_path(line)
        if code == "??":
            path = path.split(" -> ")[-1]
            if ignore_prefix is not None and (path == ignore_prefix
                                              or path.startswith(ignore_prefix + "/")):
                continue
            return True
        return True
    return False


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=ALLOWED_MODELS[:3],
                    choices=ALLOWED_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--T-train", type=int, default=8)
    ap.add_argument("--sweep-T", nargs="+", type=int, default=[1, 2, 4, 8, 12, 16, 24, 32])
    ap.add_argument("--budget-minutes", type=float, default=45)
    ap.add_argument("--out", default="week2_results")
    ap.add_argument("--fast", action="store_true",
                    help="smoke mode: 200/100/100 mazes, 2 epochs, seed 0, T sweep 1 4 8")
    return ap.parse_args(argv)


# ---- run loop ----

def execute_run(model_name, seed, tag, data, cfg, out_dir, max_minutes, verbose=True):
    """Train one (model, seed), then evaluate / sweep / settle on the test sets."""
    out_dir = Path(out_dir)
    factory = lambda: make_model(model_name)              # noqa: E731
    n_p = n_params(make_model(model_name))
    # effective n_params: one forward+backward on the first real training batch
    bs = train_mod.HYPERPARAMS["batch_size"]
    train_ds = data["train"]
    n = min(bs, len(train_ds))
    xb = train_ds.x[:n].to(train_mod.DEVICE)
    yb = train_ds.y[:n].to(train_mod.DEVICE)
    n_eff = effective_n_params(make_model(model_name), xb, yb, cfg["T_train"],
                               pos_weight=train_pos_weight(train_ds))
    print("[%s] %d params (%d effective), training (max %.1f min)"
          % (tag, n_p, n_eff, max_minutes if max_minutes is not None else -1))
    model, hist = train_mod.train_model(
        factory, T=cfg["T_train"], seed=seed, tag=model_name,
        train_ds=data["train"], val_ds=data["val"],
        pos_weight=train_pos_weight(data["train"]),
        results_dir=out_dir, verbose=verbose,
        checkpoint_path=out_dir / ("%s.ckpt.pt" % tag), run_id=tag,
        max_minutes=max_minutes)
    test_summary, _, _ = train_mod.evaluate_model(model, data["test"], cfg["T_train"])
    sweeps = {"test": train_mod.sweep_model(model, data["test"], cfg["sweep_T"]),
              "probe7": train_mod.sweep_model(model, data["probe7"], cfg["sweep_T"]),
              "probe9": train_mod.sweep_model(model, data["probe9"], cfg["sweep_T"])}
    settling = train_mod.settling_curve(model, data["test"], max(cfg["sweep_T"]))
    return dict(model=model_name, seed=seed, status="done", n_params=n_p,
                n_params_effective=n_eff,
                epochs_done=hist.get("epochs_done"), best_epoch=hist.get("best_epoch"),
                stopped_early=hist.get("stopped_early", False),
                train_seconds=hist.get("train_time_s"), test=test_summary,
                sweeps=sweeps, settling=settling)


def run_all(payload, data, out_dir, budget_minutes=45.0, verbose=True):
    """Run every planned (model, seed) under the shared budget, saving after each."""
    cfg = payload["config"]
    path = metrics_path(out_dir)
    payload.setdefault("runs", {})
    plan = run_plan(cfg["models"], cfg["seeds"])
    pending = [r for r in plan if not run_done(payload, r[2])]
    t0 = time.monotonic()
    for i, (name, seed, tag) in enumerate(pending):
        elapsed = (time.monotonic() - t0) / 60.0
        remaining_runs = len(pending) - i
        remaining = budget_minutes - elapsed
        if not budget_allows(remaining):
            payload["runs"][tag] = dict(model=name, seed=seed, status="skipped: budget")
            atomic_write_json(path, payload)
            print("[%s] skipped: budget (%.1f min left)" % (tag, remaining))
            continue
        max_minutes = split_budget(remaining, remaining_runs)
        payload["runs"][tag] = execute_run(name, seed, tag, data, cfg, out_dir,
                                           max_minutes, verbose=verbose)
        atomic_write_json(path, payload)
    return payload


def print_table(payload):
    runs = payload["runs"]
    print("\n%-14s %8s  %-15s %7s %8s %s"
          % ("model", "params", "strict (mean+-std)", "IoU", "epochs", "stopped"))
    for name in payload["config"]["models"]:
        done = [r for r in runs.values()
                if r.get("model") == name and r.get("status") == "done"]
        seeds = payload["config"]["seeds"]
        if not done:
            print("%-14s %8s  %-15s" % (name, "-", "no finished runs"))
            continue
        strict = [r["test"]["strict"] for r in done]
        ious = [r["test"]["iou"] for r in done]
        eps = [r["epochs_done"] for r in done]
        std = statistics.stdev(strict) if len(strict) > 1 else 0.0
        stopped = sum(1 for r in done if r["stopped_early"])
        print("%-14s %8d  %7.4f +- %5.4f  %7.4f %8.1f %d/%d"
              % (name, done[0]["n_params"], statistics.mean(strict), std,
                 statistics.mean(ious), statistics.mean(eps), stopped, len(seeds)))


def main():
    args = parse_args()
    cfg = config_of(args)
    out_dir = Path(cfg["out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg["fast"]:
        train_mod.HYPERPARAMS["epochs"] = FAST["epochs"]
    data = load_data(cfg)
    path = metrics_path(out_dir)
    payload = new_payload(cfg, data)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            diffs = config_mismatch(saved.get("config"), cfg)
            if diffs:
                sys.exit("refusing to mix results: %s was written with a "
                         "different config (differing keys: %s);\n"
                         "use a different --out or remove the existing file"
                         % (path, ", ".join(diffs)))
            payload["runs"] = saved.get("runs", {})
        except (OSError, json.JSONDecodeError):
            pass
    run_all(payload, data, out_dir, budget_minutes=cfg["budget_minutes"])
    print("\nmetrics written to %s" % path)
    print_table(payload)


if __name__ == "__main__":
    main()
