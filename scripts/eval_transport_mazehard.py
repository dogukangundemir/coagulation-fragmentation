"""Physics-only Family D on the Maze-hard test set: exact solve vs
Gauss-Seidel (two-colour) sweep budgets.

Plain language: the maze is treated as a network of pipes - every pair of
adjacent free cells is a pipe, a wall is a missing pipe. At every time step
one unit of water is poured into the start cell and drained from the goal
cell, and a pipe thickens in proportion to the flow it carries. A cell is
predicted to be on the solution path when the total flow through the pipes
touching it is at least half the poured amount (the poured amount is 1, so
the cutoff is 0.5). This threshold and this readout rule are fixed in the
specification; the arm has zero learned parameters (raw and effective both
0), so nothing is trained (0 epochs, 0 seeds) and the whole run is ONE
deterministic evaluation on CPU (runtime in minutes, reported below).

Two things are measured on the Maze-hard TEST set (1000 mazes, 30x30, from
the same Hugging Face dataset every maze-hard number in this repository
uses, cf/data/maze_hard.py):

- the exact whole-maze solve per outer step (float64) and fixed budgets of
  Gauss-Seidel (two-colour) sweeps per outer step (k in 1, 4, 16, 64): each
  sweep updates every cell from its own neighbours only, so information
  travels one cell per sweep - the quantity that separates "the physics contains the answer"
  from "a local process finds it in bounded time";
- how many mazes have MORE THAN ONE shortest start-to-goal path (BFS count,
  analysis only), and the accuracy separately on tied and untied mazes.

Maze-hard is not the curriculum: the grids may host several equally short
routes and free cells may touch at corners only. On a tied maze the water
splits across every tied route, so the read-out lights up cells of BOTH
routes and the board cannot match the single reference route; the exact
board match therefore reports a different thing than on the loop-free
curriculum, and a valid-path score is also given (any wall-free simple
start-to-goal path of the label's length). "Board match" below is the strict
cell-for-cell equality of the predicted cells with the reference solution -
the metric of the WONN reference code; the "valid path" number is the
repository's strict metric. The two coincide on the curriculum's tree mazes
but not here, so both are reported.

Two local adaptations of the exact solver were needed, both recorded in the
output: (a) every Maze-hard grid contains free-cell pockets DISCONNECTED
from the start-goal component (measured: 7-52 components per test maze;
the curriculum grids are connected, so this never arose there). The pipe
network of such a pocket carries no current at ANY potential, so the
pressure system is singular (the potential of a pocket is not fixed by any
gauge). The solve here adds a tiny diagonal jitter of 1e-13 that pins every
component; it changes no flow (a component with no source and no sink
carries exactly zero current), and the step's flow-conservation residual
stays far below the 1e-5 tolerance. (b) The exact solve as implemented
builds each step's Laplacian through a dense incidence matrix; on 30x30
grids that accounts for most of the runtime, so this script rebuilds the
identical matrix through its 5-point stencil (same entries, no measuring
difference; asserted equal in the unit tests).

Outer-step horizons T in 1, 2, 4, 8 all come from ONE shared rollout per
batch; T* is selected per solver configuration on a seeded 100-maze subset
of the Maze-hard TRAIN split (never touched by this arm) as the horizon
with the highest subset board match, ties going to the smaller T.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cf  # noqa: E402
import torch  # noqa: E402

import eval_transport_physics as etp                  # noqa: E402
from check_maze_hard import count_shortest_paths      # noqa: E402
from cf.data.maze_hard import (CONFIG, MazeHardDataset,  # noqa: E402
                               load_maze_hard)
from cf.metrics import evaluate_mask                  # noqa: E402
from cf.models.transport import TransportCFNet        # noqa: E402
from select_T_val import pick_T_star                  # noqa: E402

DEFAULT_STEPS = [1, 2, 4, 8]
JACOBI_KS = (1, 4, 16, 64)
OUT_DEFAULT = "results/transport_mazehard"
EVAL_BATCH = 100
GAUGE_EPS = 1e-13
VAL_SIZE = 100
VAL_SEED = 0
NBR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------- sub-classes

class MazeHardExactNet(TransportCFNet):
    """Physics arm, exact mode, adapted to Maze-hard grids.

    _solve_chunk is rebuilt with the 5-point stencil (identical matrix, far
    faster than the dense-incidence construction at 30x30) plus the gauge
    jitter GAUGE_EPS on the whole diagonal: every free-cell pocket that is
    disconnected from the start-goal component gets a pinned potential, so
    the solve exists for every grid. The jitter changes no flow: a component
    without source or sink carries exactly zero current at any potential,
    and inside the start-goal component the perturbation is O(GAUGE_EPS),
    an order of magnitude below the conservation tolerance of 1e-5 at the
    potentials these grids produce. Everything else (adaptation, readout,
    widths, wall structure) is the physics arm of cf/models/transport.py
    unchanged; the class holds zero learned parameters like its parent.
    """

    def _solve_chunk(self, walk, b, Ch, Cv):
        B, H, W = walk.shape
        dev = walk.device
        flat = walk.reshape(B, -1)
        n_free = flat.sum(dim=1).double()
        N = int(n_free.max().item())
        p = torch.zeros(B, H, W, dtype=torch.float64, device=dev)
        if N == 0:
            return p
        loc_flat = torch.cumsum(flat.double(), dim=1) - 1.0     # (B, H*W)
        hm = torch.nonzero((walk[:, :, :-1] > 0) & (walk[:, :, 1:] > 0))
        vm = torch.nonzero((walk[:, :-1, :] > 0) & (walk[:, 1:, :] > 0))
        A = torch.zeros(B, N, N, dtype=torch.float64, device=dev)
        if hm.shape[0]:
            hb, hr, hc = hm[:, 0], hm[:, 1], hm[:, 2]
            u = loc_flat[hb, hr * W + hc].long()
            v = loc_flat[hb, hr * W + hc + 1].long()
            c = Ch[hb, hr, hc]
            A.index_put_((hb, u, v), -c)
            A.index_put_((hb, v, u), -c)
            A.index_put_((hb, u, u), c, accumulate=True)
            A.index_put_((hb, v, v), c, accumulate=True)
        if vm.shape[0]:
            vb, vr, vc = vm[:, 0], vm[:, 1], vm[:, 2]
            u = loc_flat[vb, vr * W + vc].long()
            v = loc_flat[vb, (vr + 1) * W + vc].long()
            c = Cv[vb, vr, vc]
            A.index_put_((vb, u, v), -c)
            A.index_put_((vb, v, u), -c)
            A.index_put_((vb, u, u), c, accumulate=True)
            A.index_put_((vb, v, v), c, accumulate=True)
        rhs = torch.zeros(B, N, dtype=torch.float64, device=dev)
        fb, ff = torch.nonzero(flat > 0, as_tuple=True)
        rhs.index_put_((fb, loc_flat[fb, ff].long()), b.reshape(B, -1)[fb, ff])
        eye = torch.eye(N, dtype=torch.float64, device=dev)
        keep = torch.ones(N, N, dtype=torch.float64, device=dev)
        keep[0, :] = 0.0
        pin = torch.zeros(N, N, dtype=torch.float64, device=dev)
        pin[0, 0] = 1.0
        A = A * keep + pin
        A = A + GAUGE_EPS * eye[None]
        rhs = rhs * (torch.arange(N, device=dev) > 0).double()
        pad = (torch.arange(N, device=dev).double()[None, :] >= n_free[:, None]).double()
        A = A + pad[:, :, None] * eye[None]
        diag = torch.diagonal(A, dim1=1, dim2=2)
        A = A + (diag == 0).double()[:, :, None] * eye[None] * self.eps_d
        sol = torch.linalg.solve(A, rhs)
        loc = loc_flat.reshape(B, H, W)
        sol_grid = torch.gather(sol, 1, loc.flatten(1).clamp(min=0).long())
        return (sol_grid * flat.double()).reshape(B, H, W)


# ------------------------------------------------------------- measurements

def tied_flags(d):
    """Per maze: True iff the maze has more than one shortest S->G path.

    BFS counting on the wall grid only (analysis; no model involved), capped
    at 2: anything >= 2 is a tie. Uses scripts/check_maze_hard's counter -
    the same routine the dataset validation report uses.
    """
    return np.array([count_shortest_paths(w, s, g, cap=2) >= 2
                     for w, s, g in zip(d.walls, d.starts, d.goals)], dtype=bool)


def free_components(wall):
    """Number of 4-neighbour free-cell components of one grid (flood fill)."""
    wall = np.asarray(wall, bool)
    H, W = wall.shape
    seen = np.zeros_like(wall)
    n = 0
    for r in range(H):
        for c in range(W):
            if wall[r, c] or seen[r, c]:
                continue
            n += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                a, bb = stack.pop()
                for dr, dc in NBR4:
                    x, y = a + dr, bb + dc
                    if 0 <= x < H and 0 <= y < W and not wall[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        stack.append((x, y))
    return n


def _new_accum():
    return dict(board={True: 0, False: 0}, strict={True: 0, False: 0},
                iou={True: 0.0, False: 0.0}, n=0, held=0, roll_s=0.0,
                eval_s=0.0, n_batches=0, residual=0.0)


def evaluate_config(model, d, tied, Ts, batch_size=EVAL_BATCH):
    """One solver configuration over one dataset; shared rollout per batch.

    Exact mode reuses eval_transport_physics's step machinery, including its
    solver-breakdown policy: a maze whose exact solve degenerates (singular
    or non-finite, or flow conservation beyond the 1e-5 tolerance) is held at
    its last conservation-valid flow. On Maze-hard grids (which have loops)
    such a held flow is NOT guaranteed to equal the later exact flow - the
    counts are reported per horizon. Sweep (Gauss-Seidel, two-colour) mode has
    no exact solve
    to break: every maze is stepped at every horizon and only finiteness is
    checked.

    Returns dict(n_mazes, n_tied, n_untied, wall_time_s, per_T, held_mazes).
    per_T[T]: board / strict accuracy split by tie status (True = tied),
    mean IoU, held count, residual mean, wall seconds.
    """
    model = model.eval()
    Ts = sorted(set(int(t) for t in Ts))
    exact = model.relaxation == "exact"
    agg = {T: _new_accum() for T in Ts}
    held_total = 0
    total_s = 0.0
    with torch.no_grad():
        for i in range(0, len(d), batch_size):
            xb = d.x[i:i + batch_size]
            yb = d.y[i:i + batch_size]
            tb = tied[i:i + batch_size]
            Dh, Dv, p = model.initial_state(xb)
            Qh_full = torch.zeros_like(Dh)
            Qv_full = torch.zeros_like(Dv)
            B = xb.shape[0]
            frozen = torch.zeros(B, dtype=torch.bool)
            info = {"residual": 0.0}
            t_done = 0
            t0_total = time.perf_counter()
            for T in Ts:
                roll0 = time.perf_counter()
                for _ in range(T - t_done):
                    t_done += 1
                    if exact:
                        act = torch.nonzero(~frozen).flatten()
                        if act.numel() == 0:
                            continue
                        Dh2, Dv2, p2, Qh2, Qv2, stinfo, failed = etp._step_active(
                            model, xb[act], Dh[act], Dv[act], p[act],
                            Qh_full[act], Qv_full[act])
                        if stinfo is not None:
                            info = stinfo
                        Dh[act] = Dh2
                        Dv[act] = Dv2
                        p[act] = p2
                        Qh_full[act] = Qh2
                        Qv_full[act] = Qv2
                        if failed:
                            frozen[act[torch.tensor(failed, dtype=torch.long)]] = True
                    else:
                        Dh, Dv, p, Qh2, Qv2, info = model.step(xb, Dh, Dv, p)
                        if not (torch.isfinite(Dh).all() and torch.isfinite(Dv).all()
                                and torch.isfinite(p).all()
                                and torch.isfinite(Qh2).all()
                                and torch.isfinite(Qv2).all()):
                            raise SystemExit("non-finite sweep step at maze offset %d" % i)
                        Qh_full, Qv_full = Qh2, Qv2
                roll_s = time.perf_counter() - roll0
                ev0 = time.perf_counter()
                logits = model.read(xb, Dh, Dv, p, Qh_full, Qv_full)
                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
                a = agg[T]
                a["n"] += B
                a["roll_s"] += roll_s
                a["n_batches"] += 1
                a["residual"] += float(info["residual"])
                a["held"] += int(frozen.sum())
                for k in range(B):
                    row = evaluate_mask(preds[k], d.walls[i + k], d.starts[i + k],
                                        d.goals[i + k], yb[k].numpy())
                    a["board"][bool(tb[k])] += row["exact"]
                    a["strict"][bool(tb[k])] += row["strict"]
                    a["iou"][bool(tb[k])] += row["iou"]
                a["eval_s"] += time.perf_counter() - ev0
            n_frozen = int(frozen.sum())
            held_total += n_frozen
            total_s += time.perf_counter() - t0_total
    per_T = {}
    for T in Ts:
        a = agg[T]
        n = a["n"]
        per_T[str(T)] = dict(
            board_all=a["board"][True] + a["board"][False],
            board_tied=a["board"][True],
            board_untied=a["board"][False],
            strict_all=a["strict"][True] + a["strict"][False],
            strict_tied=a["strict"][True],
            strict_untied=a["strict"][False],
            iou_mean=(a["iou"][True] + a["iou"][False]) / n,
            n_held=a["held"],
            wall_seconds=a["roll_s"] + a["eval_s"],
            residual_mean=(a["residual"] / a["n_batches"]) if exact else None,
        )
    return dict(n_mazes=len(d), wall_time_s=total_s, per_T=per_T,
                held_mazes=held_total)


def accuracy_rows(res):
    """(T, board, strict) per horizon, for T* bookkeeping and rendering."""
    return {int(T): (s["board_all"], s["strict_all"], s["iou_mean"])
            for T, s in res["per_T"].items()}


# --------------------------------------------- tie-break extension (Family G)

def per_maze_dissipation(Qh, Qv, Dh, Dv):
    """E = sum_e Q_e^2 / D_e per maze, from the returned flow maps and the
    material they were solved under (the step's E_t, per maze instead of the
    batch scalar of info["dissipation"]). Wall edges contribute 0 (D = 0
    exactly where Q = 0)."""
    qh, qv = Qh.double(), Qv.double()
    zh = torch.where(Dh > 0, qh * qh / Dh.double(), torch.zeros_like(qh))
    zv = torch.where(Dv > 0, qv * qv / Dv.double(), torch.zeros_like(qv))
    return zh.sum(dim=(1, 2)) + zv.sum(dim=(1, 2))


def rollout_snapshots(model, xb, Ts, Dh, Dv, p):
    """One shared rollout from a GIVEN initial state over horizons Ts.

    Returns {T: (Dh, Dv, p, Qh, Qv, diss)} where diss (B,) is the per-maze
    dissipation of the step that produced the snapshot (the flow under the
    pre-update material). Exact mode keeps the solver-breakdown policy of
    evaluate_config (a maze whose solve degenerates is held at its last
    conservation-valid flow, and at its last dissipation); sweep mode only
    checks finiteness.
    """
    Ts = sorted(set(int(t) for t in Ts))
    exact = model.relaxation == "exact"
    B = xb.shape[0]
    Qh = torch.zeros_like(Dh)
    Qv = torch.zeros_like(Dv)
    diss = torch.zeros(B, dtype=torch.float64)
    frozen = torch.zeros(B, dtype=torch.bool)
    snaps, t_done = {}, 0
    for T in Ts:
        for _ in range(T - t_done):
            t_done += 1
            if exact:
                act = torch.nonzero(~frozen).flatten()
                if act.numel() == 0:
                    continue
                Dh_o, Dv_o = Dh[act], Dv[act]
                Dh2, Dv2, p2, Qh2, Qv2, _, failed = etp._step_active(
                    model, xb[act], Dh_o, Dv_o, p[act], Qh[act], Qv[act])
                diss[act] = per_maze_dissipation(Qh2, Qv2, Dh_o, Dv_o)
                Dh[act], Dv[act], p[act] = Dh2, Dv2, p2
                Qh[act], Qv[act] = Qh2, Qv2
                if failed:
                    frozen[act[torch.tensor(failed, dtype=torch.long)]] = True
            else:
                Dh_o, Dv_o = Dh, Dv
                Dh, Dv, p, Qh, Qv, _ = model.step(xb, Dh, Dv, p)
                if not (torch.isfinite(Dh).all() and torch.isfinite(Dv).all()
                        and torch.isfinite(p).all()):
                    raise SystemExit("non-finite sweep step")
                diss = per_maze_dissipation(Qh, Qv, Dh_o, Dv_o)
        snaps[T] = (Dh, Dv, p, Qh, Qv, diss)
    return snaps


def evaluate_tie_config(model, d, tied, Ts, n_starts, seed,
                        batch_size=EVAL_BATCH):
    """K seeded random starts (Family G tie-break): per maze and horizon the
    start whose rollout has the LOWEST per-maze dissipation at that horizon
    is read out, so a start whose adaptation locked onto one route (low
    dissipation) beats starts whose flow still splits across tied routes.

    Returns the same shape as evaluate_config (plus n_starts, tie_seed).
    """
    model = model.eval()
    Ts = sorted(set(int(t) for t in Ts))
    agg = {T: _new_accum() for T in Ts}
    held_total = 0
    total_s = 0.0
    with torch.no_grad():
        for i in range(0, len(d), batch_size):
            xb = d.x[i:i + batch_size]
            yb = d.y[i:i + batch_size]
            tb = tied[i:i + batch_size]
            B = xb.shape[0]
            t0 = time.perf_counter()
            snaps = []
            for k in range(n_starts):
                gen = torch.Generator().manual_seed(
                    1000003 * seed + 1009 * k + i // batch_size)
                Dh, Dv, p = model.initial_state(xb, stochastic=True,
                                                generator=gen)
                snaps.append(rollout_snapshots(model, xb, Ts, Dh, Dv, p))
            for T in Ts:
                diss = torch.stack([s[T][5] for s in snaps])   # (K, B)
                pick = diss.argmin(dim=0)                       # (B,)
                Dh_c = torch.empty_like(snaps[0][T][0])
                Dv_c = torch.empty_like(snaps[0][T][1])
                p_c = torch.empty_like(snaps[0][T][2])
                Qh_c = torch.empty_like(snaps[0][T][3])
                Qv_c = torch.empty_like(snaps[0][T][4])
                for k in range(n_starts):
                    m = pick == k
                    if m.any():
                        s = snaps[k][T]
                        Dh_c[m], Dv_c[m], p_c[m] = s[0][m], s[1][m], s[2][m]
                        Qh_c[m], Qv_c[m] = s[3][m], s[4][m]
                logits = model.read(xb, Dh_c, Dv_c, p_c, Qh_c, Qv_c)
                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
                a = agg[T]
                a["n"] += B
                a["n_batches"] += 1
                for k in range(B):
                    row = evaluate_mask(preds[k], d.walls[i + k],
                                        d.starts[i + k], d.goals[i + k],
                                        yb[k].numpy())
                    a["board"][bool(tb[k])] += row["exact"]
                    a["strict"][bool(tb[k])] += row["strict"]
                    a["iou"][bool(tb[k])] += row["iou"]
            total_s += time.perf_counter() - t0
    per_T = {}
    for T in Ts:
        a = agg[T]
        n = a["n"]
        per_T[str(T)] = dict(
            board_all=a["board"][True] + a["board"][False],
            board_tied=a["board"][True],
            board_untied=a["board"][False],
            strict_all=a["strict"][True] + a["strict"][False],
            strict_tied=a["strict"][True],
            strict_untied=a["strict"][False],
            iou_mean=(a["iou"][True] + a["iou"][False]) / n,
            n_held=0, wall_seconds=0.0, residual_mean=None,
        )
    return dict(n_mazes=len(d), wall_time_s=total_s, per_T=per_T,
                held_mazes=held_total, n_starts=n_starts, tie_seed=seed)


EXTENSION_OUT = OUT_DEFAULT + "_tie"


def run_extension(args):
    """Tie-break + SOR extension: FIRST n-test Maze-hard test mazes, exact
    solver and SOR budgets, each with and without the K-random-start
    tie-break (lowest-dissipation pick). Board match is the strict
    cell-for-cell equality with the reference solution."""
    etp.ensure_clone()
    torch.manual_seed(0)
    t_start = time.perf_counter()
    rows_test = load_maze_hard("test")
    d = MazeHardDataset(rows_test[:args.n_test])
    tied = tied_flags(d)
    sets = dict(n_test_mazes=len(d), n_tied=int(tied.sum()),
                n_untied=int((~tied).sum()))
    print("extension: first %d test mazes, %d tied" %
          (sets["n_test_mazes"], sets["n_tied"]))
    configs = [("exact", MazeHardExactNet(physics_only=True,
                                          relaxation="exact"), 1)]
    configs.append(("exact_tie", MazeHardExactNet(physics_only=True,
                                                  relaxation="exact"),
                    args.tie_starts))
    for k in args.sor_ks:
        base = dict(physics_only=True, relaxation="sweep", k_rel=k,
                   omega=args.omega)
        configs.append(("sor_%d" % k, TransportCFNet(**base), 1))
        configs.append(("sor_%d_tie" % k, TransportCFNet(**base),
                        args.tie_starts))
    results = {}
    for name, model, n_starts in configs:
        t0 = time.perf_counter()
        if n_starts > 1:
            res = evaluate_tie_config(model, d, tied, args.steps,
                                      n_starts, args.tie_seed,
                                      args.batch_size)
        else:
            res = evaluate_config(model, d, tied, args.steps, args.batch_size)
        res["relaxation"] = model.relaxation
        res["k_rel"] = getattr(model, "k_rel", None)
        res["omega"] = getattr(model, "omega", None)
        results[name] = res
        print("  %-14s board@T=8 %.4f best %.4f wall %.1f s"
              % (name,
                 res["per_T"]["8"]["board_all"] / res["n_mazes"],
                 max(s["board_all"] for s in res["per_T"].values())
                 / res["n_mazes"],
                 time.perf_counter() - t0))
    seconds = time.perf_counter() - t_start
    payload = dict(
        experiment=("physics-only Family D on the FIRST %d Maze-hard test "
                    "mazes: exact solve and SOR (omega=%s) budgets, each "
                    "with and without the Family G tie-break (%d seeded "
                    "random starts, lowest-dissipation pick)"
                    % (args.n_test, args.omega, args.tie_starts)),
        dataset=dict(hf_id=CONFIG["hf_id"], split="test",
                     n_test_mazes=sets["n_test_mazes"],
                     note=("first %d rows of the test csv (reduced budget, "
                           "labelled)" % args.n_test)),
        tie_break=dict(draw="D_e(0) = 1 + sigma_tie * u, u ~ U(-1, 1), "
                            "sigma_tie = 0.1, seeded per (start, batch)",
                       n_starts=args.tie_starts, seed=args.tie_seed,
                       pick="per maze and horizon: the start with the "
                            "lowest per-maze dissipation at that horizon"),
        sor=dict(omega=args.omega,
                 note=("successive over-relaxation of the two-colour "
                       "sweeps; omega = 1 is plain Gauss-Seidel")),
        accuracy_definition=dict(
            board_match=("predicted cells equal the reference solution "
                         "cell-for-cell (the strict metric; cf.metrics "
                         "'exact')"),
            valid_path=("predicted cells form one wall-free simple "
                        "start-to-goal path of the label's length "
                        "(cf.metrics 'strict')")),
        protocol=dict(steps_T=list(args.steps), sor_ks=list(args.sor_ks),
                       training="none: 0 epochs, 0 seeds, CPU only",
                       device="cpu"),
        sets=sets,
        results=results,
        wall_time=dict(total_seconds=seconds),
    )
    lines = ["# Transport tie-break / SOR extension: results", "",
             "First %d Maze-hard test mazes (%d tied). Board match = strict "
             "cell-for-cell equality with the reference solution. SOR "
             "omega = %s; tie-break = %d seeded random starts "
             "(D_e(0) = 1 + 0.1*u), per maze the start with the lowest "
             "dissipation. T = 8 is the fixed reference horizon; 'best T' "
             "is the post-hoc maximum over T in %s (labelled, no holdout "
             "selection)."
             % (sets["n_test_mazes"], sets["n_tied"], args.omega,
                args.tie_starts, list(args.steps)), "",
             "| solver | board match, no tie (T=8 / best T) | board match, "
             "tie-break K=%d (T=8 / best T) | valid path, no tie -> tie "
             "(T=8) |" % args.tie_starts, "| --- | --- | --- | --- |"]
    Ts = sorted(args.steps)
    for base, label in [("exact", "exact solve (float64)")] + \
            [("sor_%d" % k, "SOR omega=%s, k=%d" % (args.omega, k))
             for k in args.sor_ks]:
        r0, r1 = results[base], results[base + "_tie"]
        b0 = [r0["per_T"][str(T)]["board_all"] / r0["n_mazes"] for T in Ts]
        b1 = [r1["per_T"][str(T)]["board_all"] / r1["n_mazes"] for T in Ts]
        s0 = r0["per_T"]["8"]["strict_all"] / r0["n_mazes"]
        s1 = r1["per_T"]["8"]["strict_all"] / r1["n_mazes"]
        lines.append("| %s | %s / %s | %s / %s | %s -> %s |"
                     % (label, _fmt(b0[Ts.index(8)] if 8 in Ts else b0[-1]),
                        _fmt(max(b0)), _fmt(b1[Ts.index(8)] if 8 in Ts else b1[-1]),
                        _fmt(max(b1)), _fmt(s0), _fmt(s1)))
    lines.append("")
    payload["summary_text"] = "\n".join(lines)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rj = out_dir / "results.json"
    with open(rj, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(payload["summary_text"])
    print("\n".join(lines))
    print("wrote %s" % rj)
    return 0


# ------------------------------------------------------------------ outputs

def _fmt(x, digits=4):
    return ("%.*f" % (digits, x)).rstrip("0").rstrip(".")


def build_payload(model_info, sets, results, T_star, seconds_total):
    return dict(
        experiment=("physics-only Family D (transport) on the Maze-hard "
                    "test set: exact solve and Jacobi budgets"),
        dataset=dict(hf_id=CONFIG["hf_id"], split="test", size=CONFIG["size"],
                     n_test_mazes=sets["n_test_mazes"],
                     verified_by="cf.data.maze_hard validation, reported by check_maze_hard runs"),
        model=dict(model_info, params_raw=0, params_effective=0),
        readout=dict(
            rule=("a cell is predicted on the solution path iff the total "
                  "flow magnitude at that cell (sum of |flow| over its up "
                  "to four incident free pipes) is >= I/2 with I = 1, "
                  "i.e. >= 0.5; wall cells never"),
            fixed_in_advance="docs/specs/family_d_transport.md, section Readout",
        ),
        accuracy_definition=dict(
            board_match=("predicted cells equal the reference solution "
                         "cell-for-cell (the strict metric of the WONN "
                         "reference code; cf.metrics calls it exact)"),
            valid_path=("predicted cells form one wall-free simple "
                        "start-to-goal path of the label's length "
                        "(cf.metrics strict; the two coincide on the "
                        "curriculum's tree mazes, not on Maze-hard)"),
            tie_definition=("a maze is tied iff the wall grid admits more "
                            "than one shortest start-to-goal path (BFS "
                            "count, capped at 2; analysis only)"),
        ),
        protocol=dict(
            steps_T=list(DEFAULT_STEPS),
            horizon_meaning=("T = outer steps; each outer step re-solves "
                             "the pressure and moves pipe material once"),
            jacobi=("sweep relaxation: k two-colour sweeps per outer step, "
                    "each sweep updates every cell from its own neighbours "
                    "only (information moves locally, one cell per sweep)"),
            T_star=("selected per configuration on a seeded %d-maze subset of "
                    "the Maze-hard TRAIN split (seed %d, never used by this "
                    "arm) as the horizon with the highest subset board "
                    "match; ties go to the smaller T" % (VAL_SIZE, VAL_SEED)),
            training="none: 0 epochs, 0 seeds, one deterministic run, CPU only",
            params_raw=0, params_effective=0,
            device="cpu",
            solver=dict(gauge_jitter=GAUGE_EPS,
                        note=("Maze-hard grids hold free-cell pockets "
                              "disconnected from the start-goal "
                              "component; the pocket potentials are "
                              "not fixed by any gauge, so the exact "
                              "system is singular. A 1e-13 diagonal "
                              "jitter pins every component; it "
                              "carries no flow (no source, no sink) "
                              "and keeps the conservation residual "
                              "far below the 1e-5 tolerance. The "
                              "Laplacian is rebuilt via its 5-point "
                              "stencil - identical entries to the "
                              "dense-incidence construction of "
                              "cf/models/transport.py, asserted in "
                              "tests/test_eval_transport_mazehard.py")),
        ),
        sets=sets,
        T_star={name: T_star[name] for name in results},
        results={name: results[name] for name in results},
        wall_time=dict(total_seconds=seconds_total,
                       note="evaluation only; no training phase"),
    )


def render_summary(payload, sets, results, T_star, wall_times):
    per = payload["results"]
    lines = []
    lines.append("# Physics-only transport on the hard mazes: results")
    lines.append("")
    lines.append("## What was run")
    lines.append("")
    lines.append("The maze was treated as a network of pipes: every pair of "
                 "adjacent free cells is a pipe, a wall is a missing pipe. "
                 "At every time step one unit of water is poured into the "
                 "start cell and drained from the goal cell, and a pipe "
                 "thickens in proportion to the flow it carries. A cell is "
                 "declared part of the solution path when the total flow "
                 "through the pipes touching it reaches at least half the "
                 "poured amount (0.5). This rule was fixed in the "
                 "specification before any run; nothing is learned: 0 "
                 "parameters, raw and effective both 0, 0 epochs of "
                 "training, 0 seeds - one deterministic run on CPU "
                 "(grid %dx%d, %s test split, %d mazes; total wall time "
                 "%.1f minutes)."
                 % (payload["dataset"]["size"], payload["dataset"]["size"],
                    payload["dataset"]["hf_id"], payload["dataset"]["n_test_mazes"],
                    payload["wall_time"]["total_seconds"] / 60.0))
    lines.append("")
    lines.append("Every maze in the test set was also scored with two "
                 "numbers, defined here once: the **board match** counts a "
                 "maze as solved when the predicted cells equal the "
                 "reference solution cell-for-cell (this is the strict "
                 "metric of the WONN reference code); the **valid path** "
                 "number counts it solved when the predicted cells form "
                 "any wall-free simple start-to-goal path of the "
                 "reference length. On the loop-free training grids the "
                 "two are the same thing; on these mazes they are not. "
                 "**T** is the number of outer steps (each step re-solves "
                 "the water pressure in the whole maze, then moves pipe "
                 "material once); **k** is the number of Jacobi sweeps per "
                 "outer step (each sweep updates every cell from its own "
                 "neighbours only, so information travels one cell per "
                 "sweep). T* is the best subset-selected horizon of a "
                 "seeded 100-maze holdout from the train split (ties go to "
                 "the smaller T); T = 8 is the fixed reference horizon. "
                 "Numbers are fractions of mazes solved.")
    lines.append("")
    lines.append("## Ties: how many mazes have several shortest solutions")
    lines.append("")
    lines.append("A breadth-first count on the wall grid (analysis only, no "
                 "model involved) marks a maze as tied when more than one "
                 "shortest start-to-goal route exists. On the test set, "
                 "%d of %d mazes are tied (%s%%). The grids also hold "
                 "free-cell pockets disconnected from the start-goal "
                 "component (%d-%d components per maze), which the exact "
                 "pressure solve handles with a fixed tiny jitter that "
                 "carries no flow. Because water splits across every tied "
                 "route, on a tied maze the read-out lights up cells of "
                 "several routes at once and the board cannot match the "
                 "single reference route - the tie count above tells how "
                 "much of the set that argument covers."
                 % (sets["n_tied"], sets["n_test_mazes"],
                    _fmt(100.0 * sets["n_tied"] / sets["n_test_mazes"], 1),
                    sets["components_min"], sets["components_max"]))
    lines.append("")
    lines.append("## Board match (WONN's strict), test set, at T = 8 and at T*")
    lines.append("")
    lines.append("| relaxation per outer step | all %d mazes, at T=8 / at T* | tied (n=%d), at T=8 / at T* | untied (n=%d), at T=8 / at T* |"
                 % (sets["n_test_mazes"], sets["n_tied"], sets["n_untied"]))
    lines.append("| --- | --- | --- | --- |")
    for name, label in row_labels():
        r = per[name]["test"]["per_T"]
        ts = T_star[name]
        t8, tstar = r["8"], r[str(ts)]
        lines.append("| %s | %s / %s | %s / %s | %s / %s |"
                     % (label,
                        _fmt(t8["board_all"] / sets["n_test_mazes"]),
                        _fmt(tstar["board_all"] / sets["n_test_mazes"]),
                        _fmt(t8["board_tied"] / max(sets["n_tied"], 1)),
                        _fmt(tstar["board_tied"] / max(sets["n_tied"], 1)),
                        _fmt(t8["board_untied"] / max(sets["n_untied"], 1)),
                        _fmt(tstar["board_untied"] / max(sets["n_untied"], 1))))
    lines.append("")
    if sets["n_untied"] == 0:
        lines.append("The untied column is empty by measurement, not by "
                     "omission: the breadth-first count found more than one "
                     "shortest route on every maze of the test set, so there "
                     "is no untied maze to score.")
    lines.append("")
    lines.append("Same split for the valid-path number (any correct-length "
                 "simple path), all mazes, at T = 8 and at T*:")
    lines.append("")
    lines.append("| relaxation per outer step | valid path, all mazes, at T=8 / at T* |")
    lines.append("| --- | --- |")
    for name, label in row_labels():
        r = per[name]["test"]["per_T"]
        ts = T_star[name]
        lines.append("| %s | %s / %s |"
                     % (label,
                        _fmt(r["8"]["strict_all"] / sets["n_test_mazes"]),
                        _fmt(r[str(ts)]["strict_all"] / sets["n_test_mazes"])))
    lines.append("")
    lines.append("## Mean overlap with the reference route (descriptive)")
    lines.append("")
    lines.append("The board-match score is binary, so it cannot show where "
                 "the physics gets to on a braided maze. The mean cell "
                 "overlap between the predicted cells and the single "
                 "reference route (intersection over union) can - the "
                 "reference route is fully contained in the lit region when "
                 "the overlap is high, yet the board cannot match while "
                 "other tied routes stay lit.")
    lines.append("")
    Ts = payload["protocol"]["steps_T"]
    lines.append("| relaxation per outer step | " + " | ".join("mean overlap at T=%d" % T for T in Ts) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in Ts) + " |")
    for name, label in row_labels():
        r = per[name]["test"]["per_T"]
        lines.append("| %s | %s |" % (label, " | ".join(
            _fmt(r[str(T)]["iou_mean"], 3) for T in Ts)))
    lines.append("")
    lines.append("## How to read the numbers")
    lines.append("")
    held = per.get("exact", {}).get("test", {}).get("held_mazes", 0)
    lines.append("The exact solve re-solves the whole maze at once, so the "
                 "pressure, and with it the flow, is spread through every "
                 "loop of the maze at the first step: on these braided "
                 "gradients the read-out threshold keeps every cell whose "
                 "pipes carry a sizeable share of the unit current, and "
                 "parallel corridors stay lit together. The Jacobi "
                 "budgets relax the pressure instead with k local sweeps "
                 "per outer step, so current needs many sweeps to travel "
                 "from the start to the goal: at small total sweep counts "
                 "the goal is not reached yet and almost nothing is lit. "
                 "Outer steps and sweeps both move pipe material, so the "
                 "maps sharpen with the total budget.")
    # concrete reading, straight from the measured numbers
    def _peak_iou(name):
        return max(per[name]["test"]["per_T"][str(T)]["iou_mean"] for T in Ts)
    lines.append("")
    lines.append("What the numbers show here: with the exact solve the "
                 "reference route is lit together with its tied twins - the "
                 "mean cell overlap with the single reference route climbs "
                 "to %.3f and stays flat, but the extra lit roads keep the "
                 "board from matching it, and the valid-path score is near "
                 "zero (%d of %d mazes at T = 8). The Jacobi budgets only "
                 "move information a cell per sweep; at the largest budget "
                 "measured (k = 64, T = 8, 512 sweeps in total) the mean "
                 "overlap is %.3f - the current has not spread far enough "
                 "for anything like a path to appear, let alone match a "
                 "board."
                 % (_peak_iou("exact"),
                    per["exact"]["test"]["per_T"]["8"]["strict_all"],
                    sets["n_test_mazes"],
                    per["jacobi_64"]["test"]["per_T"]["8"]["iou_mean"]))
    if held:
        lines.append("")
        lines.append("Solver-holding note: %d maze-holds happened under the "
                     "exact solve across the test rollout (a maze whose "
                     "pressure solve went numerically out of tolerance is "
                     "held at its last conservation-valid flow, exactly "
                     "as the curriculum physics run does). Unlike on the "
                     "loop-free training grids, a held flow on a looped "
                     "maze is not guaranteed to equal the later exact "
                     "flow, so the exact rows carry that caveat at the "
                     "horizons where holds occurred; per-horizon hold "
                     "counts are in results.json." % held)
    lines.append("")
    lines.append("Provenance of every number: single deterministic run "
                 "(nothing learned, so no seeds and no seed spread), CPU "
                 "only, float64 solves, wall times per configuration in "
                 "results.json (%s). Nothing here is a reproduction of a "
                 "trained baseline - the arm has no training phase; "
                 "comparisons against trained arms belong to other pages."
                 % ", ".join("%s %.1f s" % (n, wall_times[n]) for n in wall_times))
    lines.append("")
    return "\n".join(lines)


def row_labels():
    return ([("exact", "exact solve (float64)")]
            + [("jacobi_%d" % k, "Jacobi, %d %s" % (k, "sweep" if k == 1 else "sweeps"))
               for k in JACOBI_KS])


def write_outputs(out_dir, payload, summary_text):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rj = out_dir / "results.json"
    tmp = rj.with_suffix(rj.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(str(tmp), str(rj))
    sm = out_dir / "summary.md"
    with open(sm, "w", encoding="utf-8") as fh:
        fh.write(summary_text)
    return rj, sm


# --------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help="output directory (default: %s)" % OUT_DEFAULT)
    ap.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH)
    ap.add_argument("--extension", action="store_true",
                    help=("tie-break / SOR extension: FIRST --n-test "
                          "Maze-hard test mazes, exact and SOR budgets, "
                          "each with and without the K-random-start "
                          "tie-break (writes to %s by default)"
                          % EXTENSION_OUT))
    ap.add_argument("--n-test", type=int, default=300,
                    help="extension: first N test mazes (default 300)")
    ap.add_argument("--tie-starts", type=int, default=4,
                    help="extension: K random starts per maze (default 4)")
    ap.add_argument("--tie-seed", type=int, default=0,
                    help="extension: base seed of the tie-break draws")
    ap.add_argument("--omega", type=float, default=1.9,
                    help="extension: SOR over-relaxation factor (default 1.9)")
    ap.add_argument("--sor-ks", nargs="+", type=int, default=[16, 64],
                    help="extension: SOR sweep budgets (default 16 64)")
    args = ap.parse_args(argv)
    if args.extension and args.out == OUT_DEFAULT:
        args.out = EXTENSION_OUT
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.extension:
        return run_extension(args)
    etp.ensure_clone()
    torch.manual_seed(0)
    t_start = time.perf_counter()
    print("loading maze-hard splits (test = full csv; %d-maze holdout "
          "from train, seed %d)" % (VAL_SIZE, VAL_SEED))
    rows_test = load_maze_hard("test")
    rows_train = load_maze_hard("train")
    d_test = MazeHardDataset(rows_test)
    perm = np.random.default_rng(VAL_SEED).permutation(len(rows_train))
    d_val = MazeHardDataset([rows_train[i] for i in perm[:VAL_SIZE]])
    tied_test = tied_flags(d_test)
    tied_val = tied_flags(d_val)
    comps = [free_components(w) for w in d_test.walls]
    sets = dict(
        n_test_mazes=len(rows_test), n_tied=int(tied_test.sum()),
        n_untied=int((~tied_test).sum()), components_min=int(min(comps)),
        components_max=int(max(comps)),
        val=dict(n_mazes=len(d_val), n_tied=int(tied_val.sum()),
                 seed=VAL_SEED, split="maze-hard train csv"),
    )
    print("test: %d mazes, %d tied, components %d..%d"
          % (sets["n_test_mazes"], sets["n_tied"], sets["components_min"],
             sets["components_max"]))

    results, T_star, wall_times = {}, {}, {}
    for name, model in config_name_and_model():
        print("config %s: running holdout then test" % name)
        rv = evaluate_config(model, d_val, tied_val, args.steps, args.batch_size)
        val_board = {T: acc[0] for T, acc in accuracy_rows(rv).items()}
        T_star[name] = pick_T_star(val_board)
        rt = evaluate_config(model, d_test, tied_test, args.steps, args.batch_size)
        results[name] = dict(relaxation=model.relaxation,
                             k_rel=getattr(model, "k_rel", None),
                             val=dict(n_mazes=rv["n_mazes"], per_T=rv["per_T"],
                                      held_mazes=rv["held_mazes"],
                                      wall_time_s=rv["wall_time_s"]),
                             test=dict(n_mazes=rt["n_mazes"], per_T=rt["per_T"],
                                       held_mazes=rt["held_mazes"],
                                       wall_time_s=rt["wall_time_s"]))
        wall_times[name] = rt["wall_time_s"]
        print("  T*=%s board at 8=%.4f T*=%.4f held=%d wall=%.1f s"
              % (T_star[name],
                 rt["per_T"]["8"]["board_all"] / rt["n_mazes"],
                 rt["per_T"][str(T_star[name])]["board_all"] / rt["n_mazes"],
                 rt["held_mazes"], rt["wall_time_s"]))

    seconds_total = time.perf_counter() - t_start
    model_info = dict(cls="cf.models.transport.TransportCFNet (physics_only=True)",
                      learned_parameters=0, params_raw=0, params_effective=0,
                      training="none (0 epochs, 0 seeds, deterministic)")
    payload = build_payload(model_info, sets, results, T_star, seconds_total)
    summary_text = render_summary(payload, sets, results, T_star, wall_times)
    rj, sm = write_outputs(args.out, payload, summary_text)
    print("wrote %s and %s" % (rj, sm))
    return 0


def config_name_and_model():
    yield "exact", MazeHardExactNet(physics_only=True, relaxation="exact")
    for k in JACOBI_KS:
        yield "jacobi_%d" % k, TransportCFNet(physics_only=True,
                                             relaxation="sweep", k_rel=k)


if __name__ == "__main__":
    sys.exit(main())
