"""Experiments X1 and X6: trained wall-aware model vs transport physics.

Plain language. Two maze solvers are compared on the same fixed test mazes.
Family A is a trained CELL model: it puts "mass" (a substance) on every free
cell of the maze, and at every time step pairs of monomers merge into
clusters (2 A1 -> A2) and clusters split back (A2 -> 2 A1). The total mass
never changes; what changes is how much of it is locked up in clusters ("the
committed share", 2*c2/M per cell). The prediction of Family A is read off
the mass maps after T steps. Family A here is the wall-aware variant trained
for 160 epochs, seeds 0/1/2 (checkpoints loaded from the main results
directory, read-only).

Family D is the transport model with NOTHING learned: the maze is a network
of pipes, water is poured into the start cell and drained from the goal cell
at every step, pipes that carry flow thicken and pipes that carry none thin.
The prediction of Family D is the set of cells whose incident pipes carry at
least half the poured unit of flow. Its pressure solve runs exactly (a
direct linear solve) or approximately with k local relaxation sweeps per
outer step (the model's built-in iterative relaxation mode, k in 1, 4, 16).

What is measured, per set (level-5 test, first 300 mazes; probe level 7,
first 200; probe level 9, first 200; all from the fixed seeded splits shared
with every other run) and per number of test-time steps T in
1, 2, 4, 8, 12, 16, 24, 32:

- strict accuracy: the predicted cells form one wall-free start-to-goal
  path of exactly the label's length;
- exact accuracy: the predicted cells match the label path cell-for-cell
  (on these tree mazes the two coincide);
- for Family A, the committed-mass split: the share of all committed mass
  (2*c2 summed over free cells) that sits on label-path cells vs on
  dead-end cells (free cells off the path), and the mean per-cell commit
  fraction 2*c2/M on each of the two cell groups;
- for Family D, the flow split: the share of all cell flow |Q| that sits
  on label-path cells vs on dead-end cells, and the mean per-cell |Q| on
  each group.

Additional Family D axis: the same numbers at k in 1, 4, 16 relaxation
sweeps per step (vs the exact solve), so accuracy can be read both against
T (steps) and against the solver budget k.

Budgets and honesty of the numbers, stated up front: Family A results are
3 training seeds with 160 epochs each; the evaluation sets are the FIRST
300 / 200 mazes of the fixed splits, so these are reduced evaluation runs,
not a reproduction of the full 1000-maze numbers. Test strict is reported
at T = 8 (the training horizon) and at T*, the horizon selected on the
validation split by the training program (ties to the smaller T).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cf  # noqa: E402
import cf.train  # noqa: E402  (DEVICE lives here)
import torch  # noqa: E402

import eval_transport_physics as etp  # noqa: E402
from cf.data.curriculum import build_datasets, load_curriculum, make_splits  # noqa: E402
from cf.metrics import evaluate_mask  # noqa: E402
from cf.models.cf import CFNet, MODEL_SPECS  # noqa: E402
from cf.models.transport import TransportCFNet  # noqa: E402

OUT_DEFAULT = "results/x1_x6"
DEFAULT_STEPS = [1, 2, 4, 8, 12, 16, 24, 32]
DEFAULT_K = [1, 4, 16]
SEEDS = [0, 1, 2]
EVAL_BATCH = 100
SET_SIZES = {"level5_test": 300, "probe7": 200, "probe9": 200}
LEVEL_OF_SET = {"level5_test": 5, "probe7": 7, "probe9": 9}
CKPT_SUBDIR = ("week2_results", "wallaware160")   # inside the main results
CKPT_PATTERN = "E5_CF-WallAware160_seed%d.pt"
EPOCHS = 160
MIN_UNIT_MINUTES = 1.5         # do not START an evaluation unit with less left
DEFAULT_DATA_SIZES = dict(n_train=4000, n_val=1000, n_test=1000, n_probe=1000,
                          week1_level=5, probe_levels=(7, 9), seed=0)


# --------------------------------------------------------------- run guards

def ensure_clone(cf_file=None):
    cf_file = cf.__file__ if cf_file is None else cf_file
    cf_root = Path(cf_file).resolve().parents[1]
    if cf_root != REPO_ROOT.resolve():
        raise SystemExit("cf resolves to %s, not this clone (%s)"
                         % (cf_root, REPO_ROOT.resolve()))
    return cf_root


def main_results_root():
    """Root of the main repository's results archive, from CF_MAIN_RESULTS."""
    root = os.environ.get("CF_MAIN_RESULTS", "").strip()
    if not root:
        raise SystemExit(
            "CF_MAIN_RESULTS is not set: point it at the main repository's "
            "results archive (the folder that contains %s), or pass the "
            "checkpoint folder explicitly with --ckpt-dir"
            % "/".join(CKPT_SUBDIR))
    return Path(root)


def resolve_ckpt_dir(cli_value=None):
    """Family A checkpoint folder: --ckpt-dir wins, else CF_MAIN_RESULTS."""
    if cli_value:
        return Path(cli_value)
    return main_results_root().joinpath(*CKPT_SUBDIR)


class SubsetView:
    """First-n slice of a MazeDataset: the surface evaluations use."""

    def __init__(self, d, n):
        n = min(int(n), len(d))
        self.x = d.x[:n].contiguous()
        self.y = d.y[:n].contiguous()
        self.walls = d.walls[:n]
        self.starts = d.starts[:n]
        self.goals = d.goals[:n]

    def __len__(self):
        return len(self.walls)


def load_level_sets(sizes=None):
    mazes, sols, levels = load_curriculum()
    kw = dict(DEFAULT_DATA_SIZES if sizes is None else sizes)
    splits, *_ = make_splits(mazes, sols, levels, **kw)
    ds = build_datasets(splits, mazes, sols)
    return {"level5_test": SubsetView(ds["test"], SET_SIZES["level5_test"]),
            "probe7": SubsetView(ds["probe"][7], SET_SIZES["probe7"]),
            "probe9": SubsetView(ds["probe"][9], SET_SIZES["probe9"])}


# --------------------------------------------------------------- mechanisms

def _path_and_dead(y, walk):
    """Boolean maps: on label path, dead-end (free and off the label path)."""
    on = y > 0.5
    free = walk > 0
    return on, free & ~on


def committed_stats(c1, c2, y, walk):
    """Family A mechanism, per maze (shape (B,)).

    c1, c2: (B, hid, H, W); y, walk: (B, H, W). Committed mass = the cluster
    half, 2*c2. Returns dict of per-maze tensors: share of all committed
    mass on path cells / on dead-end cells, and the mean per-cell commit
    fraction 2*c2/M on each group (M = c1 + 2*c2, summed over channels).
    """
    on, dead = _path_and_dead(y, walk)
    com = 2.0 * c2.sum(dim=1)                     # (B, H, W) committed per cell
    M = (c1 + 2.0 * c2).sum(dim=1)
    tot = com.sum(dim=(1, 2)).clamp(min=1e-12)
    n_on = on.sum(dim=(1, 2)).clamp(min=1)
    n_dead = dead.sum(dim=(1, 2)).clamp(min=1)
    return dict(
        com_path_share=(com * on).sum(dim=(1, 2)) / tot,
        com_dead_share=(com * dead).sum(dim=(1, 2)) / tot,
        com_path_level=((com * on).sum(dim=(1, 2)) / (M * on).sum(dim=(1, 2)).clamp(min=1e-12)),
        com_dead_level=((com * dead).sum(dim=(1, 2))
                        / (M * dead).sum(dim=(1, 2)).clamp(min=1e-12)),
        mean_na=n_on, mean_nb=n_dead,
    )


def flow_stats(qcell, y, walk):
    """Family D mechanism, per maze (B, H, W) cell flow |Q|_cell."""
    on, dead = _path_and_dead(y, walk)
    tot = qcell.sum(dim=(1, 2)).clamp(min=1e-12)
    n_on = on.sum(dim=(1, 2)).clamp(min=1)
    n_dead = dead.sum(dim=(1, 2)).clamp(min=1)
    return dict(
        flow_path_share=(qcell * on).sum(dim=(1, 2)) / tot,
        flow_dead_share=(qcell * dead).sum(dim=(1, 2)) / tot,
        flow_path_level=(qcell * on).sum(dim=(1, 2)) / n_on,
        flow_dead_level=(qcell * dead).sum(dim=(1, 2)) / n_dead,
    )


# ------------------------------------------------------------- accuracies

def score_set(logits_T, d, i0):
    """strict/exact for one horizon's logits (B, H, W) at offset i0."""
    preds = (torch.sigmoid(logits_T) >= 0.5).cpu().numpy()
    strict = exact = 0
    for k in range(preds.shape[0]):
        row = evaluate_mask(preds[k], d.walls[i0 + k], d.starts[i0 + k],
                            d.goals[i0 + k], d.y[i0 + k].numpy())
        strict += row["strict"]
        exact += row["exact"]
    return strict, exact


def _new_metrics(n):
    return dict(n=n, strict=0, exact=0, mech=[])


def _finalize(agg, Ts, mech_keys):
    per_T = {}
    for T in Ts:
        a = agg[T]
        n = a["n"]
        per_T[str(T)] = dict(strict=a["strict"] / n, exact=a["exact"] / n)
        for k in mech_keys:
            per_T[str(T)][k] = sum(v[k] for v in a["mech"]) / n
    return per_T


def eval_family_a(model, d, Ts, device):
    """Family A: shared rollout per batch, trace of c1/c2 per step."""
    model = model.to(device).eval()
    Ts = sorted(set(int(t) for t in Ts))
    agg = {T: _new_metrics(0) for T in Ts}
    n_total = 0
    with torch.no_grad():
        for i in range(0, len(d), EVAL_BATCH):
            xb = d.x[i:i + EVAL_BATCH].to(device)
            yb = d.y[i:i + EVAL_BATCH].to(device)
            out, trace = model.logits_at(xb, Ts, record=True)
            for T in Ts:
                logits = out[T]
                strict, exact = score_set(logits, d, i)
                c1 = trace["c1"][T - 1]
                c2 = trace["c2"][T - 1]
                ms = committed_stats(c1, c2, d.y[i:i + len(c1)],
                                     d.x[i:i + len(c1)][:, 0])
                a = agg[T]
                a["n"] += xb.shape[0]
                a["strict"] += strict
                a["exact"] += exact
                a["mech"].append({k: float(v.sum()) for k, v in ms.items()
                                  if not k.startswith("mean_")})
            n_total += xb.shape[0]
    keys = ["com_path_share", "com_dead_share", "com_path_level", "com_dead_level"]
    return dict(n_mazes=n_total, per_T=_finalize(agg, Ts, keys))


def eval_family_d(model, d, Ts, device, guard_breakdown):
    """Family D: shared rollout per batch; exact mode holds broken solves.

    guard_breakdown=True (exact mode) reuses eval_transport_physics's
    isolation machinery: a maze whose exact solve degenerates (singular, or
    flow-conservation residual over the 1e-5 tolerance) is held at its last
    valid solved flow - on a maze tree the conserved flow is unique, so the
    held value is what every later step would give. Sweep mode cannot go
    singular and runs plain steps.
    """
    model = model.to(device).eval()
    Ts = sorted(set(int(t) for t in Ts))
    agg = {T: _new_metrics(0) for T in Ts}
    n_total = n_held = 0
    held_hist = {}
    with torch.no_grad():
        for i in range(0, len(d), EVAL_BATCH):
            xb = d.x[i:i + EVAL_BATCH].to(device)
            yb = d.y[i:i + EVAL_BATCH].to(device)
            walk_cpu = xb[:, 0].cpu()
            y_cpu = d.y[i:i + EVAL_BATCH]
            Dh, Dv, p = model.initial_state(xb)
            Qh = torch.zeros_like(Dh)
            Qv = torch.zeros_like(Dv)
            B = xb.shape[0]
            frozen = torch.zeros(B, dtype=torch.bool)
            t_done = 0
            for T in Ts:
                for _ in range(T - t_done):
                    t_done += 1
                    if guard_breakdown:
                        act = torch.nonzero(~frozen).flatten()
                        if act.numel() == 0:
                            continue
                        r = etp._step_active(model, xb[act], Dh[act], Dv[act],
                                             p[act], Qh[act], Qv[act])
                        Dh2, Dv2, p2, Qh2, Qv2, _info, failed = r
                        Dh[act], Dv[act], p[act] = Dh2, Dv2, p2
                        Qh[act], Qv[act] = Qh2, Qv2
                        if failed:
                            ids = act[torch.tensor(failed, dtype=torch.long)]
                            frozen[ids] = True
                            held_hist[t_done] = held_hist.get(t_done, 0) + len(failed)
                    else:
                        Dh, Dv, p, Qh, Qv, _info = model.step(xb, Dh, Dv, p)
                logits = model.read(xb, Dh, Dv, p, Qh, Qv)
                strict, exact = score_set(logits, d, i)
                qcell = model._cell_absflow(Qh, Qv)
                fs = flow_stats(qcell.cpu(), y_cpu, walk_cpu)
                a = agg[T]
                a["n"] += B
                a["strict"] += strict
                a["exact"] += exact
                a["mech"].append({k: float(v.sum()) for k, v in fs.items()
                                  if not k.startswith("mean_")})
            n_total += B
            n_held += int(frozen.sum())
    keys = ["flow_path_share", "flow_dead_share", "flow_path_level",
            "flow_dead_level"]
    return dict(n_mazes=n_total, per_T=_finalize(agg, Ts, keys),
                solver_held=dict(n_held=n_held,
                                 by_step={str(s): c for s, c in sorted(held_hist.items())}))


# ------------------------------------------------------------------ outputs

def effective_params(model, xb, yb, T=8):
    """Parameter entries with nonzero gradient after one forward+backward
    (BCEWithLogits at horizon T on the given batch)."""
    model = model.to(xb.device)
    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(xb, T=T), yb)
    loss.backward()
    n_eff = sum(int(p.grad.ne(0).sum()) for p in model.parameters()
                if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return n_eff


def load_wallaware(seed, ckpt_dir):
    model = CFNet(hid=16, **MODEL_SPECS["CF-WallAware"])
    sd = torch.load(str(Path(ckpt_dir) / (CKPT_PATTERN % seed)),
                    map_location="cpu")
    model.load_state_dict(sd)
    return model


def unit_plan():
    units = []
    for seed in SEEDS:
        for name in SET_SIZES:
            units.append(("A", "seed%d" % seed, name))
    for k in [None] + list(DEFAULT_K):
        for name in SET_SIZES:
            units.append(("D", ("sweep%d" % k) if k else "exact", name))
    return units


def run_units(data, ckpt_dir, budget_minutes, device):
    t0 = time.monotonic()
    units = unit_plan()
    results = {}
    # resume: results.json written by the caller after every unit
    out_path = Path(args_out()) / "results.json"
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            results = saved.get("results", results)
        except (OSError, ValueError):
            results = {}
    model_cache = {}
    for fam, tag, name in units:
        key = "%s_%s_%s" % (fam, tag, name)
        if key in results and results[key].get("status") == "done":
            continue
        remaining = budget_minutes - (time.monotonic() - t0) / 60.0
        if remaining < MIN_UNIT_MINUTES:
            results[key] = dict(status="skipped: budget")
            persist(out_path, results, args_ckpt_dir())
            print("[%s] skipped: budget (%.1f min left)" % (key, remaining))
            continue
        t_unit = time.perf_counter()
        d = data[name]
        if fam == "A":
            seed = int(tag.replace("seed", ""))
            if (fam, tag) not in model_cache:
                model_cache[(fam, tag)] = load_wallaware(seed, ckpt_dir)
            model = model_cache[(fam, tag)]
            r = eval_family_a(model, d, args_steps(), device)
            r.update(status="done", family="A", seed=seed, set=name,
                    seconds=time.perf_counter() - t_unit)
        else:
            if (fam, tag) not in model_cache:
                k = int(tag.replace("sweep", "")) if tag != "exact" else None
                model = TransportCFNet(
                    physics_only=True,
                    relaxation="exact" if k is None else "sweep",
                    k_rel=16 if k is None else k)
                model_cache[(fam, tag)] = model
            r = eval_family_d(model_cache[(fam, tag)], d, args_steps(), device,
                              guard_breakdown=(tag == "exact"))
            r.update(status="done", family="D", mode=tag, set=name,
                     seconds=time.perf_counter() - t_unit)
        results[key] = r
        persist(out_path, results, args_ckpt_dir())
        per_T = r["per_T"]
        print("[%s] %-7s strict@8=%.4f strict@32=%.4f (%.1f s)"
              % (key, r.get("mode", tag),
                 per_T["8"]["strict"], per_T["32"]["strict"], r["seconds"]))
    return results


_ARGS = {}


def args_out():
    return _ARGS.get("out", OUT_DEFAULT)


def args_steps():
    return _ARGS.get("steps", DEFAULT_STEPS)


def args_ckpt_dir():
    return _ARGS.get("ckpt_dir", str(resolve_ckpt_dir()))


def persist(out_path, results, ckpt_dir):
    payload = dict(
        experiment=("X1/X6: trained wall-aware Family A (160 epochs, seeds "
                    "0-2) vs physics-only transport Family D (exact solve "
                    "and k-sweep relaxation, k in %s)" % DEFAULT_K),
        protocol=dict(
            steps_T=args_steps(),
            sets={k: dict(n=SET_SIZES[k], level=LEVEL_OF_SET[k]) for k in SET_SIZES},
            splits=("first %d/%d/%d mazes of the fixed seed-0 splits "
                    "(level-5 test / probe 7 / probe 9); reduced evaluation "
                    "sets, not a reproduction of the full 1000-maze runs"
                    % (SET_SIZES["level5_test"], SET_SIZES["probe7"],
                       SET_SIZES["probe9"])),
            family_A=dict(model="cf.models.cf.CFNet wall_aware, hid 16",
                          epochs=EPOCHS, seeds=SEEDS,
                          checkpoints=str(ckpt_dir),
                          note="reduced evaluation sets: not a reproduction"),
            family_D=dict(model="cf.models.transport.TransportCFNet physics_only",
                          learned_parameters=0,
                          relaxation=("exact float64 linear solve, and "
                                      "k_rel local relaxation sweeps per "
                                      "outer step (k in %s)" % DEFAULT_K)),
        ),
        results=results,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    for attempt in range(5):                 # Windows AV/indexer can hold the
        try:                                 # target briefly; retry, then fail
            os.replace(str(tmp), str(out_path))
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(1.0)


def render_summary(results, device, params_a):
    Ts = args_steps()
    lines = ["# Trained wall-aware model vs transport physics: X1/X6 results", ""]
    lines.append("## Plain language")
    lines.append("")
    lines.append("Two solvers were compared on the same fixed test mazes. A "
                 "trained model (Family A) spreads a conserved amount of "
                 "substance over the free cells; at every step pairs of "
                 "particles merge into bound clusters and clusters split "
                 "back. Its answer is read off the substance maps after T "
                 "steps. The transport model (Family D) knows no learning at "
                 "all: the maze is a pipe network, water is poured into the "
                 "start cell and drained from the goal cell every step, "
                 "flow-carrying pipes thicken, empty pipes thin, and a cell "
                 "is declared on the solution path when the flow through its "
                 "pipes reaches at least half the poured unit.")
    lines.append("")
    lines.append("## Where the numbers come from")
    lines.append("")
    lines.append("Symbols. T = test-time steps run. Strict accuracy = the "
                 "predicted cells form one wall-free start-to-goal path of "
                 "exactly the label's length; exact = the predicted cells "
                 "equal the label path cell-for-cell (on these tree mazes "
                 "the two coincide). Committed mass (Family A) = the cluster "
                 "half 2*c2 of the conserved total M = c1 + 2*c2; the "
                 "columns 'committed on path / on dead ends' give the share "
                 "of all committed mass lying on label-path cells vs on "
                 "dead-end cells (free cells off the path), and 'commit "
                 "level' the mean per-cell fraction 2*c2/M of each group. "
                 "Flow (Family D): |Q| = flow magnitude at a cell (sum over "
                 "its incident pipes); 'flow on path / on dead ends' are "
                 "shares of all cell flow, 'flow level' the group means.")
    lines.append("")
    lines.append("Budgets, stated with every number. Family A: 3 training "
                 "seeds, 160 epochs each, checkpoints trained by the main "
                 "program (T*=12, selected on the validation split by argmax "
                 "val strict with ties to the smaller T). Evaluation used "
                 "the FIRST 300 mazes of the level-5 test split and the "
                 "FIRST 200 of each probe split (levels 7 and 9): these are "
                 "reduced evaluation sets, not a reproduction of the full "
                 "1000-maze runs. Family D: deterministic, no training, so "
                 "one run (no seeds). Hardware: %s. Wall minutes per unit "
                 "are in results.json." % str(device))
    lines.append("")

    def row_head():
        return ("| unit | T | strict | exact | on path | on dead ends | level path | level dead |"
                "\n| --- | --- | --- | --- | --- | --- | --- | --- |")

    def mech_cols(s, keys):
        return [s[k] for k in keys]

    for fam, desc in (("A", "Family A, trained wall-aware (160 epochs, 3 seeds, "
                     "strict at T=8 vs T*=12)"),
                      ("D", "Family D, transport physics only")):
        lines.append("## %s" % desc)
        lines.append("")
        if fam == "A":
            lines.append("Parameters: %d raw, %d effective (nonzero-"
                         "gradient entries after one backward pass)."
                         % (params_a[0], params_a[1]))
            lines.append("")
        for name in SET_SIZES:
            if fam == "A":
                rows = {("seed%d" % s): results.get("A_seed%d_%s" % (s, name))
                        for s in SEEDS}
            else:
                rows = {("exact" if k is None else "k=%d" % k):
                        results.get("D_%s_%s" % ("exact" if k is None
                                                 else "sweep%d" % k, name))
                        for k in [None] + list(DEFAULT_K)}
            done = {a: b for a, b in rows.items() if b and b.get("status") == "done"}
            skipped = [a for a, b in rows.items() if b and b.get("status") != "done"]
            lines.append("### %s (%d mazes, level %d)" % (name, SET_SIZES[name],
                                                         LEVEL_OF_SET[name]))
            lines.append("")
            if not done:
                lines.append("Not run." + (" Budget-skipped: %s." % skipped if skipped else ""))
                lines.append("")
                continue
            lines.append(row_head())
            keys = (["com_path_share", "com_dead_share", "com_path_level",
                     "com_dead_level"] if fam == "A" else
                    ["flow_path_share", "flow_dead_share", "flow_path_level",
                     "flow_dead_level"])
            for unit, r in done.items():
                for T in Ts:
                    s = r["per_T"][str(T)]
                    vals = mech_cols(s, keys)
                    lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
                                 % tuple([unit, T, s["strict"], s["exact"]] + vals))
            if skipped:
                lines.append("")
                lines.append("Reduced coverage (budget skipped, not run): %s"
                             % ", ".join(skipped))
            lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--budget-minutes", type=float, default=10.0)
    ap.add_argument("--ckpt-dir", default=None,
                    help="Family A checkpoint folder, read-only (default: "
                         "the %s folder of the results archive given by the "
                         "CF_MAIN_RESULTS environment variable; error if "
                         "neither is provided)"
                         % "/".join(CKPT_SUBDIR))
    return ap.parse_args(argv)


def main(argv=None):
    global _ARGS
    args = parse_args(argv)
    ckpt_dir = resolve_ckpt_dir(args.ckpt_dir)
    _ARGS = dict(out=args.out, steps=args.steps, ckpt_dir=str(ckpt_dir))
    ensure_clone()
    torch.manual_seed(0)
    device = cf.train.DEVICE
    t0 = time.perf_counter()
    print("device: %s" % device)
    print("loading curriculum splits (seed 0, level-5 test + probes 7/9)")
    data = load_level_sets()
    for name in SET_SIZES:
        print("  %s: %d mazes" % (name, len(data[name])))
    model_a = load_wallaware(0, ckpt_dir)
    n_raw = sum(p.numel() for p in model_a.parameters())
    xb = data["level5_test"].x[:32].to(device)
    yb = data["level5_test"].y[:32].to(device)
    n_eff = effective_params(load_wallaware(0, ckpt_dir), xb, yb)
    print("Family A checkpoints: %d raw / %d effective params" % (n_raw, n_eff))
    model_d = TransportCFNet(physics_only=True, relaxation="exact")
    n_d = sum(p.numel() for p in model_d.parameters())
    print("Family D physics arm: %d learned parameters" % n_d)
    assert n_d == 0
    args_out_full = Path(args.out) / "results.json"
    # keep the budget clock running over resumed units' fresh work only
    results = run_units(data, ckpt_dir, args.budget_minutes, device)
    seconds = time.perf_counter() - t0
    print("all units: %.1f s" % seconds)
    summary = render_summary(results, device, (n_raw, n_eff))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(summary)
    print("wrote %s and %s" % (args_out_full, out_dir / "summary.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
