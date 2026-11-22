"""Invariant tests for the Family D transport model (cf/models/transport.py).

Plain language: the maze is a network of pipes, material moves between
pipes, and the flow it carries is conserved. These tests check every
invariant listed in docs/specs/family_d_transport.md, for both the learned
arm and the physics-only arm (zero learned parameters) and for both
relaxation modes (exact linear solve, and K_rel two-colour Gauss-Seidel
sweeps), for any weights — trained or random:

1. flow conservation at every non-terminal cell (exact mode: Kirchhoff's
   current law to 1e-5; sweep mode: the potential energy Psi decreases
   monotonically per half-sweep);
2. antisymmetric flows: the divergence recomputed independently from both
   stored edge maps matches b, and a transposed-shift probe cannot hide;
3. tube material conservation: sum(Dh) + sum(Dv) = M_tube over 64 outer
   steps, including the randomized tie-break initialization;
4. non-negative conductances, with exact zeros on wall edges at
   initialization and after every step (walls carry zero flow, walls hold
   p = 0);
5. zero net flow in dead ends at steady state on perfect mazes: |Q_e| = 0
   off the label path and |Q_e| = I on it (exact mode), and the off-path
   transient decays with the sweep count in sweep mode;
6. dissipation monotone under fixed conductances: the steady Kirchhoff flow
   minimizes E(q) = sum q^2 / D over random conserved flows (Thomson, on
   loopy grids), the swept Psi converges to the Kirchhoff minimum, and the
   coupled physics-arm dissipation E_t is non-increasing over outer steps;
7. gradients reach the input and every learned parameter.

The mazes are built in the curriculum format (perfect mazes from the
recursive backtracker used by the other test modules) and converted with the
cf.data.curriculum adapter (parse_maze, maze_to_channels, replay_solution),
so channels and label paths match the dataset layout exactly.
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: cf

from cf.data.curriculum import maze_to_channels, parse_maze, replay_solution
from cf.models.transport import TransportCFNet

from maze_synthetic import make_example, make_maze_grid

NEG_EPS = -1e-7          # float32 round-off tolerance for non-negativity
TOL = 1e-5               # spec conservation / dead-end tolerance (float64 solve)
MONO_TOL = 1e-6          # relative tolerance for energy monotonicity


# ----------------------------------------------------------------------- helpers
def maze_batch(seeds, n=5):
    """Perfect mazes in the curriculum format.

    Returns (x, y): x (B, 3, 2n+1, 2n+1) channels [walkable, start, goal],
    y (B, 2n+1, 2n+1) the label path mask replayed from the provided
    solution (no search), both validated by the curriculum checks.
    """
    xs, ys = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        ascii_maze, solution = make_example(rng, n=n, min_moves=4)
        wall, start, goal = parse_maze(ascii_maze)
        xs.append(maze_to_channels(wall, start, goal))
        mask, end, legal, revisited = replay_solution(wall, start, solution)
        assert legal and end == goal and not revisited
        ys.append(mask)
    return torch.from_numpy(np.stack(xs)).float(), np.stack(ys).astype(bool)


def loopy_channels(seed, n=5, removed=8):
    """Perfect maze with `removed` interior walls knocked out.

    Removing walls keeps every free cell free (the maze stays connected) and
    gives the free-cell graph cycles — the setting where the conserved flow
    is not unique and the Thomson inequality is informative. Returns
    (channels, walk, start, goal)."""
    rng = np.random.default_rng(seed)
    grid = make_maze_grid(rng, n).copy()
    interior = [(r, c) for r in range(1, grid.shape[0] - 1)
               for c in range(1, grid.shape[1] - 1) if grid[r, c]]
    for i in rng.choice(len(interior), size=removed, replace=False):
        grid[interior[i]] = 0
    start, goal = (1, 1), (grid.shape[0] - 2, grid.shape[1] - 2)
    return maze_to_channels(grid, start, goal), grid == 0, start, goal


def free_masks(x):
    """free cell mask (B, H, W) and free-edge masks (B, H, W-1), (B, H-1, W)."""
    free = x[:, 0].numpy() > 0
    eh = free[:, :, :-1] & free[:, :, 1:]
    ev = free[:, :-1, :] & free[:, 1:, :]
    return free, eh, ev


def path_edges(y):
    """Label-path edge masks: an edge is on the path iff both endpoints are
    path cells (on a perfect maze these are exactly the chain edges)."""
    return y[:, :, :-1] & y[:, :, 1:], y[:, :-1, :] & y[:, 1:, :]


def b_field(x, current):
    """Source field from the input channels: +I on start, -I on goal."""
    return current * (x[:, 1] - x[:, 2]).numpy()


def divergence_np(Qh, Qv):
    """div[r,c] = Qh[r,c] - Qh[r,c-1] + Qv[r,c] - Qv[r-1,c], zero padding.

    Recomputed independently of the model so an indexing or sign bug in the
    stored flow maps cannot hide behind the model's own diagnostic."""
    eo = np.pad(Qh, ((0, 0), (0, 0), (0, 1)))
    ei = np.pad(Qh, ((0, 0), (0, 0), (1, 0)))
    so = np.pad(Qv, ((0, 0), (0, 1), (0, 0)))
    si = np.pad(Qv, ((0, 0), (1, 0), (0, 0)))
    return eo - ei + so - si


def random_conserved_flow(walk, start, goal, rng, current=1.0):
    """A random conserved flow q with div(q) = b (spec invariant 6a).

    Construction: fix a spanning tree of the free-cell graph (BFS from the
    start), push the unit current along the tree path from start to goal,
    then add an independent random signed circulation around each non-tree
    edge's fundamental cycle. Returns (qh (H, W-1), qv (H-1, W)), signed
    with the model's convention (positive eastward / southward)."""
    H, W = walk.shape
    par = {start: None}                       # cell -> (parent, delta)
    order = deque([start])
    while order:
        r, c = order.popleft()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < H and 0 <= nxt[1] < W and walk[nxt] \
                    and nxt not in par:
                par[nxt] = ((r, c), (dr, dc))
                order.append(nxt)
    qh = np.zeros((H, W - 1))
    qv = np.zeros((H - 1, W))

    def add(a, b, amount):
        """Flow `amount` from cell a to cell b across their shared edge."""
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:                          # horizontal edge
            qh[r1, min(c1, c2)] += amount if c2 > c1 else -amount
        else:                                 # vertical edge
            qv[min(r1, r2), c1] += amount if r2 > r1 else -amount

    cur = goal                                # unit current start -> goal
    while par[cur] is not None:
        prev, _ = par[cur]
        add(prev, cur, current)
        cur = prev

    tree_edges = {frozenset((cell, par[cell][0])) for cell in par if par[cell]}

    def tree_path(u, v):
        """Edges of the tree path u -> v as (from, to) pairs."""
        ancestors = {}                        # cell -> its ancestor list
        for cell in (u, v):
            chain, cur = [cell], cell
            while par[cur] is not None:
                cur = par[cur][0]
                chain.append(cur)
            ancestors[cell] = chain
        lca = next(c for c in ancestors[u] if c in set(ancestors[v]))
        path, cur = [], u
        while cur != lca:
            path.append((cur, par[cur][0]))
            cur = par[cur][0]
        down, cur = [], v
        while cur != lca:
            down.append((par[cur][0], cur))
            cur = par[cur][0]
        return path + down[::-1]

    for r in range(H):                        # circulations around cycles
        for c in range(W - 1):
            if walk[r, c] and walk[r, c + 1] and (r, c) in par \
                    and frozenset(((r, c), (r, c + 1))) not in tree_edges:
                u, v = (r, c), (r, c + 1)
                amount = rng.uniform(-1.0, 1.0)
                for a, b in tree_path(u, v):
                    add(a, b, amount)
                add(v, u, amount)             # close the fundamental cycle
    for r in range(H - 1):
        for c in range(W):
            if walk[r, c] and walk[r + 1, c] and (r, c) in par \
                    and frozenset(((r, c), (r + 1, c))) not in tree_edges:
                u, v = (r, c), (r + 1, c)
                amount = rng.uniform(-1.0, 1.0)
                for a, b in tree_path(u, v):
                    add(a, b, amount)
                add(v, u, amount)
    return qh, qv


# ------------------------------------------- 1. flow conservation (both arms)
@pytest.mark.parametrize("physics_only", [False, True])
def test_flow_conservation_at_every_cell_exact_mode(physics_only):
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=physics_only)
    x, _ = maze_batch([0, 1, 2, 3])
    _, trace = model.logits_at(x, [2], record=True)
    free, _, _ = free_masks(x)
    b = b_field(x, model.I)
    start = x[:, 1].numpy() > 0
    goal = x[:, 2].numpy() > 0
    for t in range(2):
        # the divergence recomputed from the stored edge maps (invariant 2)
        div = divergence_np(trace["Qh"][t].numpy(), trace["Qv"][t].numpy())
        # spec: |div_u - b_u| < 1e-5 * (1 + |b_u|) at every free cell
        assert np.abs(div - b)[free].max() < TOL * (1.0 + np.abs(b)[free].max())
        # terminals are the only cells with non-zero divergence (+I / -I)
        assert np.abs(div[start] - model.I).max() < TOL
        assert np.abs(div[goal] + model.I).max() < TOL
        assert np.abs(div[free & ~start & ~goal]).max() < TOL
        # the model's own diagnostic agrees
        assert trace["residual"][t] < TOL


def test_divergence_probe_detects_transposed_shifts():
    # Antisymmetry / shift consistency: the recomputed divergence matches b,
    # while swapping either pair of shifts (east<->west or south<->north)
    # breaks the match badly — so a transposed-shift bug cannot hide.
    torch.manual_seed(0)
    model = TransportCFNet()
    x, _ = maze_batch([0, 1])
    _, trace = model.logits_at(x, [1], record=True)
    free, _, _ = free_masks(x)
    b = b_field(x, model.I)
    Qh, Qv = trace["Qh"][0].numpy(), trace["Qv"][0].numpy()
    assert np.abs(divergence_np(Qh, Qv) - b)[free].max() < TOL
    eo = np.pad(Qh, ((0, 0), (0, 0), (0, 1)))
    ei = np.pad(Qh, ((0, 0), (0, 0), (1, 0)))
    so = np.pad(Qv, ((0, 0), (0, 1), (0, 0)))
    si = np.pad(Qv, ((0, 0), (1, 0), (0, 0)))
    wrong_h = (ei - eo + so - si - b)[free].max()
    wrong_v = (eo - ei + si - so - b)[free].max()
    assert max(abs(wrong_h), abs(wrong_v)) > 0.1


@pytest.mark.parametrize("physics_only", [False, True])
def test_sweep_potential_energy_monotone_per_half_sweep(physics_only):
    # Sweep mode: each half-sweep is an exact coordinate-wise minimization of
    # the C-weighted potential energy Psi_C = 1/2 sum Q^2/C - sum b p for its
    # colour, so Psi_C must decrease monotonically per half-sweep
    # (conductances fixed within an outer step; the spec's per-half residual
    # is diagnostic only). For the untrained learned arm L = 1 exactly, so
    # the trained-lengths companion test below covers the general case.
    model = TransportCFNet(physics_only=physics_only, relaxation="sweep",
                           k_rel=16)
    x, _ = maze_batch([9, 10])
    _, trace = model.logits_at(x, [4], record=True)
    for t, halves in enumerate(trace["psi_halves"], start=1):
        assert len(halves) == 2 * model.k_rel
        ps = np.array(halves)
        assert np.all(ps[1:] <= ps[:-1]
                      + MONO_TOL * np.maximum(1.0, np.abs(ps[:-1]))), \
            f"Psi increased at outer step {t}"


def test_sweep_potential_energy_monotone_with_trained_lengths():
    # Invariant 1b holds "for any network weights, trained or random": once
    # the length head leaves its zero-weight init (L != 1), the half-sweep
    # update minimizes the C-weighted energy Psi_C = 1/2 sum Q^2/C - sum b p,
    # not the length-blind 1/2 sum Q^2/D - sum b p. The test perturbs every
    # learned parameter (so L_e spreads well away from 1) and checks Psi_C
    # stays monotone per half-sweep — this exact regime is the planned X6
    # sweep-mode probe around an exact-solve-trained checkpoint.
    torch.manual_seed(3)
    model = TransportCFNet(relaxation="sweep", k_rel=16)
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(0.3 * torch.randn_like(prm))
    x, _ = maze_batch([9, 10])
    _, _, Lh, Lv = model.edge_features(x)
    assert (Lh - 1.0).abs().max().item() > 0.05, "lengths stayed at 1"
    assert (Lv - 1.0).abs().max().item() > 0.05, "lengths stayed at 1"
    _, trace = model.logits_at(x, [4], record=True)
    for t, halves in enumerate(trace["psi_halves"], start=1):
        assert len(halves) == 2 * model.k_rel
        ps = np.array(halves)
        assert np.all(ps[1:] <= ps[:-1]
                      + MONO_TOL * np.maximum(1.0, np.abs(ps[:-1]))), \
            f"Psi increased at outer step {t} with trained lengths"


# ------------------------------------------------- 2. tube material conservation
@pytest.mark.parametrize("physics_only", [False, True])
@pytest.mark.parametrize("stochastic", [False, True])
def test_tube_material_conserved_over_64_steps(physics_only, stochastic):
    # sum(Dh) + sum(Dv) = M_tube (the per-maze free-edge count) before and
    # after every material step; the tie-break draw is normalized by its own
    # sum, so it starts at M_tube too.
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=physics_only)
    x, _ = maze_batch([7, 8])
    Eh, Ev = model.edge_masks(x)
    m_tube = (Eh.sum(dim=(1, 2)) + Ev.sum(dim=(1, 2))).numpy()  # (B,)
    gen = torch.Generator().manual_seed(123)
    state = model.initial_state(x, stochastic=stochastic, generator=gen)
    for t in range(65):  # t = 0 is the initial state, then 64 outer steps
        Dh, Dv = state[0], state[1]
        s = (Dh.sum(dim=(1, 2)) + Dv.sum(dim=(1, 2))).detach().numpy()
        # per-maze relative drift (m_tube > 0 on every maze here; the floor
        # only guards a hypothetical all-wall maze)
        assert (np.abs(s - m_tube) / np.maximum(m_tube, 1.0)).max() < TOL, \
            f"material drifted at t={t}"
        if t == 64:
            break
        state = model.step(x, Dh, Dv, state[2])[:3]


# ------------------------------------------ 3. non-negativity and wall structure
@pytest.mark.parametrize("physics_only", [False, True])
def test_conductances_nonnegative_wall_edges_zero_walls_carry_no_flow(physics_only):
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=physics_only)
    x, _ = maze_batch([1])
    Eh, Ev = model.edge_masks(x)
    wall_h, wall_v = Eh == 0, Ev == 0
    wall = x[0, 0] == 0                        # (H, W) wall mask
    Dh, Dv, p = model.initial_state(x)
    for t in range(65):
        assert Dh.min().item() >= NEG_EPS, f"Dh < 0 at t={t}"
        assert Dv.min().item() >= NEG_EPS, f"Dv < 0 at t={t}"
        # edges into or inside walls are exactly 0, so walls carry no flow
        assert (Dh[wall_h] == 0).all(), f"wall edge carries material at t={t}"
        assert (Dv[wall_v] == 0).all(), f"wall edge carries material at t={t}"
        # wall cells never update and hold p = 0
        assert (p[:, wall] == 0).all(), f"wall potential nonzero at t={t}"
        if t == 64:
            break
        Dh, Dv, p, Qh, Qv, _ = model.step(x, Dh, Dv, p)
        assert (Qh[wall_h] == 0).all() and (Qv[wall_v] == 0).all()


# --------------------------------------------------------------- 4. dead ends
@pytest.mark.parametrize("physics_only", [False, True])
def test_dead_ends_carry_no_flow_at_steady_state_exact_mode(physics_only):
    # On a perfect maze the conserved flow with divergence b is unique, so
    # after one exact solve every off-path free edge carries exactly 0 and
    # every path edge exactly I (conservation alone is the reasoning).
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=physics_only)
    x, y = maze_batch(list(range(6)))
    free, eh, ev = free_masks(x)
    ph, pv = path_edges(y)
    _, trace = model.logits_at(x, [1], record=True)
    Qh, Qv = trace["Qh"][0].numpy(), trace["Qv"][0].numpy()
    # the classification is non-degenerate: every free edge is on or off path
    assert (eh & ph).any() and (eh & ~ph).any()
    assert (ev & pv).any() and (ev & ~pv).any()
    assert np.abs(Qh[eh & ~ph]).max() < TOL, "off-path edge carries flow"
    assert np.abs(Qv[ev & ~pv]).max() < TOL, "off-path edge carries flow"
    off = max(np.abs(Qh[eh & ~ph]).max(), np.abs(Qv[ev & ~pv]).max())
    on = max(np.abs(np.abs(Qh[eh & ph]) - model.I).max(),
             np.abs(np.abs(Qv[ev & pv]) - model.I).max())
    assert off < TOL and on < TOL


def test_dead_end_transient_decays_with_sweep_count():
    # Sweep mode: the same dead-end test passes once K_rel is large enough,
    # and the off-path transient decays with the relaxation budget.
    x, y = maze_batch([0, 1, 2])
    free, eh, ev = free_masks(x)
    ph, pv = path_edges(y)
    off = {}
    for k_rel in (16, 256, 4096):
        model = TransportCFNet(relaxation="sweep", k_rel=k_rel)
        _, trace = model.logits_at(x, [1], record=True)
        Qh, Qv = trace["Qh"][0].numpy(), trace["Qv"][0].numpy()
        off[k_rel] = max(np.abs(Qh[eh & ~ph]).max(), np.abs(Qv[ev & ~pv]).max())
        if k_rel == 4096:  # large enough: same tolerance as the exact solve
            on = max(np.abs(np.abs(Qh[eh & ph]) - model.I).max(),
                     np.abs(np.abs(Qv[ev & pv]) - model.I).max())
            assert off[k_rel] < TOL and on < TOL
    assert off[256] < off[16], "off-path transient did not decay 16 -> 256"
    assert off[4096] < off[256], "off-path transient did not decay 256 -> 4096"


# ---------------------------------------------------- 5. dissipation / Thomson
@pytest.mark.parametrize("seed", [0, 1])
def test_steady_flow_minimizes_dissipation_over_random_conserved_flows(seed):
    # Thomson spot check on loopy grids: with the conductances frozen at the
    # uniform start, E(Q*) = sum Q*^2 / D of the steady flow is <= E(q) for
    # 100 random conserved flows per maze.
    torch.manual_seed(0)
    model = TransportCFNet()
    channels, walk, start, goal = loopy_channels(seed)
    x = torch.from_numpy(channels)[None]
    free, eh, ev = free_masks(x)
    Dh, Dv, p = model.initial_state(x)   # conductances frozen at d_0 = 1
    _, _, _, Qh, Qv, _ = model.step(x, Dh, Dv, p)
    Qh_star, Qv_star = Qh[0].detach().numpy(), Qv[0].detach().numpy()
    e_star = (Qh_star[eh[0]] ** 2).sum() + (Qv_star[ev[0]] ** 2).sum()
    rng = np.random.default_rng(seed + 100)
    for _ in range(100):
        qh, qv = random_conserved_flow(walk, start, goal, rng)
        # the construction is genuinely conserved: div(q) = b
        b = np.zeros(walk.shape)
        b[start], b[goal] = model.I, -model.I
        assert np.abs(divergence_np(qh[None], qv[None])[0] - b)[walk].max() < 1e-9
        e_q = (qh[eh[0]] ** 2).sum() + (qv[ev[0]] ** 2).sum()
        assert e_q >= e_star - 1e-5 * (1.0 + e_star), \
            "random conserved flow beats the Kirchhoff flow (Thomson violated)"


def test_thomson_check_vacuous_on_perfect_mazes():
    # On a perfect maze the free-cell graph is a tree: the conserved flow
    # with divergence b is unique, so every random draw equals the steady
    # flow and the inequality holds with (near-)equality.
    torch.manual_seed(0)
    model = TransportCFNet()
    x, y = maze_batch([5])
    free, eh, ev = free_masks(x)
    walk = free[0]
    start = tuple(np.argwhere(x[0, 1].numpy() > 0)[0])
    goal = tuple(np.argwhere(x[0, 2].numpy() > 0)[0])
    Dh, Dv, p = model.initial_state(x)
    _, _, _, Qh, Qv, _ = model.step(x, Dh, Dv, p)
    e_star = (Qh[0].detach().numpy()[eh[0]] ** 2).sum() \
        + (Qv[0].detach().numpy()[ev[0]] ** 2).sum()
    rng = np.random.default_rng(55)
    for _ in range(20):
        qh, qv = random_conserved_flow(walk, start, goal, rng)
        e_q = (qh[eh[0]] ** 2).sum() + (qv[ev[0]] ** 2).sum()
        assert abs(e_q - e_star) < 1e-5 * (1.0 + e_star)


def test_swept_potentials_reach_the_kirchhoff_minimum():
    # From the zero potential, the swept Psi decreases to its minimum, which
    # is attained at the Kirchhoff potential (the exact solve's Psi).
    torch.manual_seed(0)
    x, _ = maze_batch([1])
    sweep = TransportCFNet(relaxation="sweep", k_rel=4096)
    _, trace = sweep.logits_at(x, [1], record=True)
    psi_final = trace["psi_halves"][0][-1]
    exact = TransportCFNet()
    _, tre = exact.logits_at(x, [1], record=True)
    free, eh, ev = free_masks(x)
    Qh, Qv, p = tre["Qh"][0].numpy(), tre["Qv"][0].numpy(), tre["p"][0].numpy()
    e_star = (Qh[eh] ** 2).sum() + (Qv[ev] ** 2).sum()   # D = d_0 = 1 frozen
    psi_star = 0.5 * e_star - (b_field(x, exact.I) * p).sum()
    scale = abs(psi_star)
    assert psi_final >= psi_star - MONO_TOL * scale  # Psi never below the min
    assert abs(psi_final - psi_star) <= 1e-5 * scale  # ... and it reaches it
    # and the whole half-sweep sequence was monotone on the way there
    ps = np.array(trace["psi_halves"][0])
    assert np.all(ps[1:] <= ps[:-1] + MONO_TOL * np.maximum(1.0, np.abs(ps[:-1])))


def test_coupled_physics_arm_dissipation_non_increasing():
    # Coupled steps on a perfect maze with uniform lengths and the
    # feature-free adaptation: the per-outer-step dissipation E_t of the
    # exact flow is non-increasing as the material coagulates onto the path.
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=True)
    x, _ = maze_batch([2, 3])
    _, trace = model.logits_at(x, [16], record=True)
    e_t = np.array(trace["dissipation"])
    assert np.all(e_t[1:] <= e_t[:-1] * (1 + MONO_TOL) + 1e-9), \
        f"dissipation increased along the rollout: {e_t}"
    assert e_t[-1] < e_t[0], "material never coagulated onto the path"


# ------------------------------------------------------------------ 6. gradients
@pytest.mark.parametrize("relaxation", ["exact", "sweep"])
def test_gradients_reach_input_and_every_parameter(relaxation):
    torch.manual_seed(0)
    model = TransportCFNet(relaxation=relaxation)
    x, _ = maze_batch([4])
    x = x.clone().requires_grad_(True)
    model(x, 8).square().sum().backward()
    assert x.grad is not None, "no grad on input x"
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"
    assert sum(p.numel() for p in model.parameters()) == 199


# ------------------------------------------------------------------ 7. readout
@pytest.mark.parametrize("physics_only", [False, True])
def test_wall_logits_are_exactly_minus_20(physics_only):
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=physics_only)
    x, _ = maze_batch([12])
    logits = model(x, 8)
    assert (logits[:, x[0, 0] == 0] == -20.0).all()


@pytest.mark.parametrize("T", [1, 8])
def test_physics_arm_threshold_readout_marks_exactly_the_label_path(T):
    # Physics arm readout: a cell is on the path iff |Q|_cell >= I/2. On a
    # perfect maze the steady flow carries I on the path and 0 in dead ends,
    # so the emitted +-20 logits match the label mask cell-for-cell
    # (walls emit -20 and are never path cells).
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=True)
    x, y = maze_batch([13, 14, 15])
    pred = (model(x, T) > 0).numpy()
    assert (pred == y).all()


@pytest.mark.parametrize("physics_only", [False, True])
def test_logits_at_matches_forward(physics_only):
    torch.manual_seed(0)
    model = TransportCFNet(physics_only=physics_only)
    x, _ = maze_batch([6])
    outs = model.logits_at(x, [1, 4, 8, 24])
    for T in (1, 4, 8, 24):
        torch.testing.assert_close(outs[T], model(x, T), atol=1e-6, rtol=0)


# ------------------------------------------------------- 8. initialisation
def test_stochastic_tie_break_initialisation():
    # D_e(0) = M_tube * exp(sigma_tie * u_e) / sum exp(sigma_tie * u_e'),
    # u ~ U(-1, 1): positive, exactly M_tube in total, zero on wall edges, and
    # different across generators (it exists to break ties between equally
    # short routes).
    torch.manual_seed(0)
    model = TransportCFNet()
    x, _ = maze_batch([16])
    Eh, Ev = model.edge_masks(x)
    m_tube = Eh.sum(dim=(1, 2)) + Ev.sum(dim=(1, 2))
    inits = [model.initial_state(x, stochastic=True,
                                 generator=torch.Generator().manual_seed(s))
             for s in (123, 456)]
    Dh0, Dv0, _ = model.initial_state(x)
    assert torch.equal(Dh0, Eh) and torch.equal(Dv0, Ev)  # d_0 = 1 on free edges
    for Dh, Dv, _ in inits:
        s = Dh.sum(dim=(1, 2)) + Dv.sum(dim=(1, 2))
        assert ((s - m_tube).abs() / m_tube < TOL).all()
        assert Dh.min().item() >= 0 and Dv.min().item() >= 0
        assert (Dh[Eh == 0] == 0).all() and (Dv[Ev == 0] == 0).all()
    assert (inits[0][0] - inits[1][0]).abs().max().item() > 1e-3, \
        "two generators gave the same initial thickness"
    # the physics arm has zero learned parameters by construction
    assert sum(p.numel() for p in TransportCFNet(physics_only=True).parameters()) == 0


def test_learned_edge_lengths_start_at_exactly_one():
    # The length head starts with zero weights and bias log(e - 1), so every
    # L_e starts at exactly 1.0 (softplus(log(e-1)) = 1) and lengths stay
    # positive, and the adaptation keeps w_q >= 0 and floors at eps_f.
    torch.manual_seed(0)
    model = TransportCFNet()
    x, _ = maze_batch([17])
    uh, uv, Lh, Lv = model.edge_features(x)
    ones_h, ones_v = torch.ones_like(Lh), torch.ones_like(Lv)
    assert torch.equal(Lh, ones_h) and torch.equal(Lv, ones_v)
    assert Lh.min().item() > 0 and Lv.min().item() > 0
    assert torch.nn.functional.softplus(model.raw_wq).item() >= 0
    assert model.eps_f > 0
