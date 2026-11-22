"""Family G: random-start voting, evaluation only.

Plain language. A maze solver can be run several times from different
random starting states. If the runs that end in a "calmer" internal state are
the correct ones, then picking the calmest run of the K is a free accuracy
boost at test time (the voting scheme of algorithm Family G). "Calm" is
measured by an energy that each family already carries:

- Family A (the trained cell model, A-WallAware, 160 epochs): the energy of
  one run is the residual coagulation-fragmentation flux summed over all
  free cells and channels at the readout step - how much reaction is still
  running when the answer is read off. Random start: the initial cluster
  share is drawn per cell as c2 = u M/2 with u ~ U(0, 0.5) (the model's own
  stochastic initial_state; mass M conserved).
- Family D (the physics-only pipe model): the energy of one run is either the
  electrical dissipation sum_e R_e Q_e^2 = sum_e Q_e^2 / D_e of the final
  flow, or the leak fraction (share of the flow sitting in dead-end pipes).
  Random start: the initial pipe conductances are drawn per edge (the
  model's own tie-break variant, normalized so the tube material is kept).

Protocol. Level-5 test mazes, first 300 of the fixed seeded split shared with
every other run in this repository. K in 1, 8, 32 random starts per maze, all
drawn once (K=1 and K=8 use the first 1 / 8 of the same 32 draws). For each
K the lowest-energy run is selected per maze and its prediction is scored
with the same strict metric as every other arm (predicted cells form one
wall-free start-to-goal path of exactly the label's length; on these tree
mazes strict == exact). Family A is read at T = 8 (training horizon) and
T* = 12 (the horizon the training program selected on validation). Family D
is read at T = 8 and T = 16 (its flow is exact from step 1 on a tree). The
deterministic starting state of each family is scored alongside as the
no-voting baseline.

Honesty of the numbers, stated up front: this is one training seed
(seed 0, 160 epochs) on a reduced 300-maze evaluation slice - preliminary,
not a reproduction of the full 1000-maze numbers. The physics arm is
deterministic given the draw; every number below carries its budget
(K random starts, 2 readout horizons, 300 mazes, seed 0).

What the energies can and cannot do is reported, not assumed: per maze the
energy of every correct run is compared with the energy of every wrong run
(within the same maze, across all 32 starts) - the fraction of such pairs
where the correct run is calmer is the separation AUC (0.5 = the energy
carries no information about correctness, 1.0 = perfect separation).
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_transport_physics as etp  # noqa: E402
import eval_x1_x6 as ex  # noqa: E402
from cf.data.curriculum import build_datasets, load_curriculum, make_splits  # noqa: E402
from cf.metrics import evaluate_mask  # noqa: E402
from cf.models.transport import TransportCFNet  # noqa: E402

OUT_DEFAULT = "results/voting"
CKPT_PATTERN = "E5_CF-WallAware160_seed%d.pt"
EPOCHS = 160
SEED = 0                    # training seed of the (only) Family A checkpoint
K_MAX = 32
K_LIST = (1, 8, 32)
N_MAZES = 300
TS_A = (8, 12)              # T_train = 8; T* = 12 (training program's validation pick)
TS_D = (8, 16)
D_CHUNK = 100               # family D rollout chunk (matches the physics arm's batch)
MIN_UNIT_MINUTES = 1.5

DATA_KW = dict(n_train=4000, n_val=1000, n_test=1000, n_probe=1000,
               week1_level=5, probe_levels=(7, 9), seed=0)


# --------------------------------------------------------------- run guards

def ensure_clone(cf_file=None):
    cf_file = cf.__file__ if cf_file is None else cf_file
    cf_root = Path(cf_file).resolve().parents[1]
    if cf_root != REPO_ROOT.resolve():
        raise SystemExit("cf resolves to %s, not this clone (%s)"
                         % (cf_root, REPO_ROOT.resolve()))
    return cf_root


def load_test_set(n=N_MAZES):
    """First-n mazes of the fixed level-5 test split."""
    mazes, sols, levels = load_curriculum()
    splits, *_ = make_splits(mazes, sols, levels, **DATA_KW)
    ds = build_datasets(splits, mazes, sols)
    return ex.SubsetView(ds["test"], n)


# --------------------------------------------------------------- scores

def strict_rows(logits, d, maze_idx):
    """Strict (= exact on these tree mazes) correctness per logits row.

    logits: (N, H, W) tensor; maze_idx: length-N sequence of dataset indices
    (the maze each row belongs to). Returns an int array (N,), 1 = solved.
    """
    preds = (torch.sigmoid(logits.detach().float().cpu()) >= 0.5).numpy()
    out = np.zeros(len(preds), dtype=np.int64)
    for i, mi in enumerate(maze_idx):
        row = evaluate_mask(preds[i], d.walls[mi], d.starts[mi],
                            d.goals[mi], d.y[mi].numpy())
        assert row["strict"] == row["exact"], "perfect maze: strict must equal exact"
        out[i] = row["strict"]
    return out


def within_maze_auc(energy, correct):
    """P(energy of a correct start < energy of a wrong start), within mazes.

    energy, correct: (K, B) arrays. Every (correct, wrong) pair of starts of
    the same maze contributes; ties count half. None when no maze has both.
    """
    if isinstance(energy, torch.Tensor):
        energy = energy.numpy()
    if isinstance(correct, torch.Tensor):
        correct = correct.numpy()
    wins = ties = total = 0
    for j in range(energy.shape[1]):
        e, c = energy[:, j], correct[:, j]
        ec, ew = e[c == 1], e[c == 0]
        if len(ec) == 0 or len(ew) == 0:
            continue
        wins += int((ec[:, None] < ew[None, :]).sum())
        ties += int((ec[:, None] == ew[None, :]).sum())
        total += len(ec) * len(ew)
    return (wins + 0.5 * ties) / total if total else None


def summarize_vote(energy, correct, K_list=None):
    """Per-K voting summary plus the energy-separation diagnostics.

    energy, correct: (K, B) tensors, one row per random start. For each K the
    per-maze argmin-energy start among the first K is selected. Returns a
    dict with accuracies, the selected energies split by outcome, and the
    separation AUC over all starts.
    """
    if isinstance(energy, torch.Tensor):
        energy = energy.numpy()
    if isinstance(correct, torch.Tensor):
        correct = correct.numpy()
    K, B = energy.shape
    if K_list is None:
        K_list = tuple(k for k in K_LIST if k <= K) + ((K,) if K not in K_LIST else ())
    idx = np.argsort(energy, axis=0, kind="stable")       # argmin = first
    out = dict(sample_accuracy=float(correct.mean()),
              any_correct=float((correct.max(axis=0) > 0).mean()),
              separation_auc=within_maze_auc(energy, correct), per_K={})
    au = np.arange(B)
    for Kk in K_list:
        sel = idx[0, :].copy()
        for j in range(B):                                # lowest among first Kk
            sel[j] = np.argmin(energy[:Kk, j])
        selc = correct[sel, au]
        sele = energy[sel, au]
        ec, ew = sele[selc == 1], sele[selc == 0]
        out["per_K"][str(Kk)] = dict(
            accuracy=float(selc.mean()),
            selected_energy_correct=dict(
                n=int(len(ec)), mean=float(ec.mean()) if len(ec) else None,
                median=float(np.median(ec)) if len(ec) else None),
            selected_energy_wrong=dict(
                n=int(len(ew)), mean=float(ew.mean()) if len(ew) else None,
                median=float(np.median(ew)) if len(ew) else None),
            selected_energy_correct_values=ec.tolist(),
            selected_energy_wrong_values=ew.tolist())
    return out


# --------------------------------------------------------------- family A

def eval_family_a(model, d, Ts=TS_A, K=K_MAX, seed=SEED, device="cpu"):
    """A-WallAware with K random cluster-share starts; shared rollout per T.

    The K starts are drawn in ONE replicated batch (rows 0..B-1 are start 1,
    rows B..2B-1 start 2, ...), so energies reshape cleanly to (K, B). The
    energy read at horizon T is the summed residual flux |m - s| of the last
    executed step, per run.
    """
    model = model.to(device).eval()
    B = len(d)
    xrep = d.x.to(device).repeat(K, 1, 1, 1)
    maze_idx = [i % B for i in range(K * B)]
    gen = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        c1, c2 = model.initial_state(xrep, stochastic=True, generator=gen)
        per_T, t = {}, 0
        for T in Ts:
            for _ in range(T - t):
                t += 1
                c1, c2, m, s, _g = model.step(xrep, c1, c2)
            energy = (m - s).abs().sum(dim=(1, 2, 3)).cpu().view(K, B)
            correct = strict_rows(model.read(xrep, c1, c2), d, maze_idx).reshape(K, B)
            per_T[str(T)] = dict(energy=energy.tolist(), correct=correct.reshape(K, B).tolist(),
                                 vote=summarize_vote(energy, correct))
    return per_T


def eval_family_a_baseline(model, d, Ts=TS_A, device="cpu"):
    """The deterministic start (c2 = 0), scored at the same horizons."""
    model = model.to(device).eval()
    x = d.x.to(device)
    maze_idx = list(range(len(d)))
    with torch.no_grad():
        c1, c2 = model.initial_state(x)
        per_T, t = {}, 0
        for T in Ts:
            for _ in range(T - t):
                t += 1
                c1, c2, _m, _s, _g = model.step(x, c1, c2)
            correct = strict_rows(model.read(x, c1, c2), d, maze_idx)
            per_T[str(T)] = dict(accuracy=int(correct.sum()) / len(d))
    return per_T


# --------------------------------------------------------------- family D

def eval_family_d(model, d, Ts=TS_D, K=K_MAX, seed=SEED):
    """Physics-only transport with K random initial conductances.

    Exact pressure solves, run in row chunks with the shared isolation
    machinery of the physics arm: a maze whose solve degenerates is held at
    its last conservation-valid flow (its exact steady flow on a tree).
    Energies per run: dissipation sum_e Q^2/D and the dead-end leak fraction,
    both read at each horizon. Predictions use the spec's fixed threshold
    readout (|Q|_cell >= I/2).
    """
    model = model.eval()
    B = len(d)
    xrep = d.x.repeat(K, 1, 1, 1)
    yrep = d.y.repeat(K, 1, 1)
    Eh, Ev = model.edge_masks(xrep)
    n = xrep.shape[0]
    maze_idx = [i % B for i in range(n)]
    gen = torch.Generator().manual_seed(2000 + seed)
    with torch.no_grad():
        Dh, Dv, p = model.initial_state(xrep, stochastic=True, generator=gen)
        Qh = torch.zeros_like(Dh)
        Qv = torch.zeros_like(Dv)
        frozen = torch.zeros(n, dtype=torch.bool)
        per_T, t, n_held = {}, 0, 0
        for T in Ts:
            for _ in range(T - t):
                t += 1
                for r0 in range(0, n, D_CHUNK):
                    rows = torch.arange(r0, min(r0 + D_CHUNK, n))
                    rows = rows[~frozen[rows]]
                    if rows.numel() == 0:
                        continue
                    r = etp._step_active(model, xrep[rows], Dh[rows], Dv[rows],
                                         p[rows], Qh[rows], Qv[rows])
                    Dh2, Dv2, p2, Qh2, Qv2, _info, failed = r
                    Dh[rows], Dv[rows], p[rows] = Dh2, Dv2, p2
                    Qh[rows], Qv[rows] = Qh2, Qv2
                    if failed:
                        real = rows[torch.tensor(failed, dtype=torch.long)]
                        frozen[real] = True
                        n_held += len(failed)
            dis_h = torch.where(Dh > 0, Qh * Qh / Dh, torch.zeros_like(Dh))
            dis_v = torch.where(Dv > 0, Qv * Qv / Dv, torch.zeros_like(Dv))
            dis = (dis_h.sum(dim=(1, 2)) + dis_v.sum(dim=(1, 2))).view(K, B)
            leak = etp.dead_end_flow_fractions(Qh, Qv, yrep, Eh, Ev).view(K, B)
            correct = strict_rows(model.read(xrep, Dh, Dv, p, Qh, Qv), d, maze_idx)
            correct = torch.tensor(correct).view(K, B)
            preds = (model.read(xrep, Dh, Dv, p, Qh, Qv) >= 0).cpu()
            disagree = int(sum((preds[j * B:(j + 1) * B] != preds[:B]).any(dim=(1, 2)).sum()
                               for j in range(1, K)))
            per_T[str(T)] = dict(
                vote_dissipation=summarize_vote(dis, correct),
                vote_leak=summarize_vote(leak, correct),
                n_start_disagreement=disagree,
                n_mazes_mixed=int(((correct.sum(0) > 0) & (correct.sum(0) < K)).sum()),
                energy_dissipation=dis.tolist(), energy_leak=leak.tolist(),
                correct=correct.tolist())
    return per_T, dict(n_held=int(n_held))


# --------------------------------------------------------------- outputs

def persist(out_dir, payload):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)


def load_done(out_dir):
    p = out_dir / "results.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def write_summary(out_dir, res):
    """Plain-language tables next to the raw results."""
    lines = ["# Family G random-start voting (evaluation only)", ""]
    lines.append("300 level-5 test mazes (fixed split), one training seed "
                 "(A-WallAware, 160 epochs, seed 0), K random starts per "
                 "maze, lowest-energy start selected per maze. Reduced "
                 "evaluation slice: preliminary, not a reproduction of the "
                 "full 1000-maze numbers.")
    lines += ["", "## Accuracy vs K (strict = exact on these tree mazes)", "",
              "| family | T | K=1 | K=8 | K=32 |", "|---|---|---|---|---|"]
    for fam in ("A", "D"):
        if fam not in res:
            continue
        for T in TS_A if fam == "A" else TS_D:
            unit = res[fam].get("per_T", {}).get(str(T), {})
            v = unit.get("vote_dissipation") or unit.get("vote")
            if not v:
                continue
            accs = [v["per_K"][str(k)]["accuracy"] for k in K_LIST]
            lines.append("| %s | %d | %.3f | %.3f | %.3f |" % (fam, T, *accs))
    if "A" in res:
        for T in TS_A:
            b = res["A"].get("baseline", {}).get(str(T))
            if b:
                lines.append("A deterministic start, T=%d: accuracy %.3f" % (T, b["accuracy"]))
    lines += ["", "## Does the energy separate correct from wrong?", ""]
    for fam in ("A", "D"):
        if fam not in res:
            continue
        for T, unit in res[fam].get("per_T", {}).items():
            for key in ("vote", "vote_dissipation", "vote_leak"):
                v = unit.get(key)
                if not v:
                    continue
                auc = v["separation_auc"]
                lines.append("%s T=%s %s: within-maze separation AUC = %s"
                             % (fam, T, key, "n/a" if auc is None else "%.3f" % auc))
    if "D" in res:
        for T, unit in res["D"].get("per_T", {}).items():
            lines.append("D T=%s: mazes whose starts disagree: %d / 300"
                         % (T, unit["n_start_disagreement"]))
    with open(out_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", default=None,
                    help="Family A checkpoint folder, read-only (default: "
                         "resolved from the CF_MAIN_RESULTS environment "
                         "variable; error if neither is provided)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--budget-minutes", type=float, default=10.0)
    args = ap.parse_args()
    ckpt_dir = ex.resolve_ckpt_dir(args.ckpt_dir)
    t0 = time.monotonic()

    ensure_clone()
    out_dir = REPO_ROOT / args.out
    res = load_done(out_dir).get("results", {})
    payload = dict(
        experiment="Family G random-start voting (evaluation only)",
        budget=dict(family="A-WallAware 160 epochs, seed 0 (preliminary: "
                    "one seed, 300-maze reduced slice)",
                    family_D="physics only, no training",
                    mazes=N_MAZES, K=K_MAX, K_list=list(K_LIST), seed=SEED,
                    budget_minutes=args.budget_minutes),
        hardware=dict(device="cpu", note=("evaluation only; torch "
                    f"{'cuda' if torch.cuda.is_available() else 'cpu'} available")),
        params=dict(A="CFNet hid=16 wall_aware (raw 10,581 / effective per "
                     "eval_x1_x6 effective_params); physics D: 0 learned"),
        seeds_note="1 training seed; fewer than 3 seeds = preliminary")
    def do_unit(key, runner):
        if res.get(key, {}).get("status") == "done":
            return
        remaining = args.budget_minutes - (time.monotonic() - t0) / 60.0
        if remaining < MIN_UNIT_MINUTES:
            res[key] = dict(status="skipped: budget")
            persist(out_dir, {**payload, "results": res})
            return
        t = time.perf_counter()
        res[key] = runner()
        res[key]["status"] = "done"
        res[key]["seconds"] = time.perf_counter() - t
        persist(out_dir, {**payload, "results": res})

    def run_a():
        model = ex.load_wallaware(SEED, ckpt_dir)
        return dict(per_T=eval_family_a(model, d),
                    baseline=eval_family_a_baseline(model, d))

    def run_d():
        model = TransportCFNet(physics_only=True, relaxation="exact")
        per_T, held = eval_family_d(model, d)
        return dict(per_T=per_T, solver_held=held)

    d = load_test_set()

    do_unit("A", run_a)
    do_unit("D", run_d)
    write_summary(out_dir, res)
    print("wrote", out_dir / "results.json")


if __name__ == "__main__":
    main()
