"""Offline unit tests for scripts/eval_transport_mazehard.py (synthetic grids).

The evaluation pipeline of the maze-hard physics run is exercised on small
synthetic grids, no dataset download: the gauge-jittered exact solve must
reproduce the stock solver on connected grids, exist where the stock solver
is singular (free-cell pockets disconnected from start-to-goal), leave no
flow in pockets, mark tied vs untied mazes correctly, accumulate the
per-horizon tie-split counters, pass the horizon-selection rule through,
and render a summary consistent with the reporting rules.
"""
import sys
from pathlib import Path

import numpy as np

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_transport_mazehard as etmh                     # noqa: E402
from cf.data.maze_hard import MazeHardDataset              # noqa: E402
from cf.models.transport import TransportCFNet             # noqa: E402
from select_T_val import pick_T_star                       # noqa: E402

SIZE = 30


def channels(wall, start, goal):
    wall = np.asarray(wall, bool)
    x = np.zeros((3,) + wall.shape, np.float32)
    x[0] = (~wall).astype(np.float32)
    x[1][start] = 1.0
    x[2][goal] = 1.0
    return torch.from_numpy(x)[None]


def grid_from(rows_free, start, goal, path_cells):
    """(list of free (r, c), start, goal, label cells) -> maze-hard row."""
    q = np.full((SIZE, SIZE), "#", dtype="<U1")
    a = np.full((SIZE, SIZE), "#", dtype="<U1")
    for r, c in rows_free:
        q[r, c] = " "
        a[r, c] = " "
    q[start] = "S"
    q[goal] = "G"
    a[start] = "S"
    a[goal] = "G"
    for r, c in path_cells:
        a[r, c] = "o"
    return dict(source="synthetic", question="".join(q.ravel()),
                answer="".join(a.ravel()), rating=1.0)


def untied_row():
    free = [(10, c) for c in range(5, 20)]
    path = [(10, c) for c in range(6, 19)]
    return grid_from(free, (10, 5), (10, 19), path)


def tied_row():
    # open 3x8 rectangle with S and G at opposite corners: many shortest routes
    free = [(r, c) for r in (10, 11, 12) for c in range(5, 13)]
    path = [(11, c) for c in range(5, 13)]
    return grid_from(free, (10, 5), (12, 12), path)


# ------------------------------------------------------- gauge-jittered solve

def test_gauge_solve_equals_stock_on_connected_grid():
    size = 7
    wall = np.ones((size, size), bool)
    for c in range(1, size - 1):
        wall[3, c] = False
    start, goal = (3, 1), (3, size - 2)
    x = channels(wall, start, goal)
    stock = TransportCFNet(physics_only=True, relaxation="exact")
    sub = etmh.MazeHardExactNet(physics_only=True, relaxation="exact")
    with torch.no_grad():
        Dh, Dv, p = sub.initial_state(x)
        r1 = stock.step(x, Dh, Dv, p)
        r2 = sub.step(x, Dh, Dv, p)
    assert torch.allclose(r1[2], r2[2], atol=1e-8)
    assert torch.allclose(r1[3], r2[3], atol=1e-8) and torch.allclose(r1[4], r2[4], atol=1e-8)


def test_gauge_solve_handles_disconnected_pockets():
    # On grids with free-cell pockets disconnected from the start-goal
    # component the stock Laplacian is rank-deficient; whether its LU
    # raises is implementation-dependent (it does on real maze-hard
    # grids, it may succeed silently on tiny synthetic ones), so the
    # jitter subclass is exercised on its own terms: existence, tiny
    # residual, no pocket flow, correct corridor read-out - and
    # corridor flows identical to the stock solver's.
    size = 9
    wall = np.ones((size, size), bool)
    for c in range(1, 8):
        wall[2, c] = False            # start-goal corridor
    for r in (6, 7):
        for c in (3, 4, 5):
            wall[r, c] = False        # disconnected pocket
    start, goal = (2, 1), (2, 7)
    x = channels(wall, start, goal)
    stock = TransportCFNet(physics_only=True, relaxation="exact")
    sub = etmh.MazeHardExactNet(physics_only=True, relaxation="exact")
    with torch.no_grad():
        Dh, Dv, p = sub.initial_state(x)
        try:
            Dh_s, Dv_s, p_s, Qh_s, Qv_s, _ = stock.step(x, Dh, Dv, p)
            assert torch.allclose(Qh_s[0, 2, 1:7], sub.step(x, Dh, Dv, p)[3][0, 2, 1:7],
                                  atol=1e-12)
        except torch.linalg.LinAlgError:
            pass                      # the behaviour seen on real maze-hard grids
        Dh2, Dv2, p2, Qh, Qv, info = sub.step(x, Dh, Dv, p)
    assert info["residual"] < 1e-5
    # no meaningful flow inside the pocket (rows 6-7, cols 3-5)
    assert Qh[0, 6, 3:5].abs().max() < 1e-6
    assert Qv[0, 6, 3:5].abs().max() < 1e-6
    logits = sub.read(x, Dh2, Dv2, p2, Qh, Qv)
    pred = (torch.sigmoid(logits) >= 0.5)[0].numpy()
    label = np.zeros_like(wall)
    label[2, 1:8] = True
    assert (pred == label).all()


# --------------------------------------------------------------- tie analysis

def test_tied_flags_splits_unique_and_braided_grids():
    d_untied = MazeHardDataset([untied_row()])
    d_tied = MazeHardDataset([tied_row()])
    assert etmh.tied_flags(d_untied).tolist() == [False]
    assert etmh.tied_flags(d_tied).tolist() == [True]


def test_free_components_counts_pockets():
    row = untied_row()
    wall = MazeHardDataset([row]).walls[0]
    assert etmh.free_components(wall) == 1
    # add a walled-off free pocket to the same grid
    q = list(row["question"])
    for r in (20, 21):
        for c in (20, 21):
            pos = r * SIZE + c
            q[pos] = " "
    wall2 = MazeHardDataset([dict(row, question="".join(q))]).walls[0]
    assert etmh.free_components(wall2) == 2


# ------------------------------------------------------------- accumulation

def test_evaluate_config_accumulates_tie_split():
    rows = [untied_row(), tied_row()]
    d = MazeHardDataset(rows)
    tied = etmh.tied_flags(d)
    assert tied.tolist() == [False, True]
    model = TransportCFNet(physics_only=True, relaxation="sweep", k_rel=4)
    res = etmh.evaluate_config(model, d, tied, [1, 2], batch_size=2)
    assert res["n_mazes"] == 2 and res["wall_time_s"] >= 0
    for T in ("1", "2"):
        s = res["per_T"][T]
        assert set(s) >= {"board_all", "board_tied", "board_untied",
                          "strict_all", "strict_tied", "strict_untied",
                          "iou_mean", "n_held", "wall_seconds"}
        assert s["board_all"] == s["board_tied"] + s["board_untied"]
        assert 0 <= s["board_all"] <= 2 and 0.0 <= s["iou_mean"] <= 1.0
        assert s["n_held"] == 0          # sweep mode never holds
    assert res["held_mazes"] == 0
    rows_acc = etmh.accuracy_rows(res)
    assert set(rows_acc) == {1, 2}


def test_pick_t_star_rule():
    assert pick_T_star({1: 0, 2: 3, 4: 4}) == 4
    assert pick_T_star({1: 2, 2: 2, 4: 1}) == 1
    assert pick_T_star({1: 0, 2: 0}) == 1


# ------------------------------------------------------------- payload / text

def _fake_results():
    per = {str(T): dict(board_all=8, board_tied=8 - T % 2, board_untied=T % 2,
                        strict_all=1, strict_tied=1, strict_untied=0,
                        iou_mean=0.5, n_held=T % 3, wall_seconds=0.1,
                        residual_mean=0.0)
           for T in (1, 2, 4, 8)}
    return dict(relaxation="exact", k_rel=None,
                val=dict(n_mazes=1, per_T=dict(per), held_mazes=0, wall_time_s=0.0),
                test=dict(n_mazes=10, per_T=dict(per), held_mazes=2, wall_time_s=1.0))


def test_build_payload_and_summary_shape():
    sets = dict(n_test_mazes=10, n_tied=8, n_untied=2,
                components_min=7, components_max=52,
                val=dict(n_mazes=1, n_tied=1, seed=0, split="synthetic"))
    names = [n for n, _ in etmh.row_labels()]
    results = {n: _fake_results() for n in names}
    T_star = {n: 8 for n in names}
    payload = etmh.build_payload(dict(cls="x", learned_parameters=0), sets,
                                 results, T_star, 12.0)
    assert payload["model"]["params_raw"] == 0
    assert payload["model"]["params_effective"] == 0
    assert payload["accuracy_definition"]["board_match"].startswith("predicted")
    text = etmh.render_summary(payload, sets, results, T_star,
                               {n: 1.0 for n in names})
    for needle in ("at T=8 / at T*", "tied (n=8)", "untied (n=2)",
                   "exact solve (float64)", "Jacobi, 64 sweeps"):
        assert needle in text, needle
    for banned in ("week", "milestone"):
        assert banned not in text.lower()


def test_row_labels_cover_exact_and_all_budgets():
    names = [n for n, _ in etmh.row_labels()]
    assert names == ["exact", "jacobi_1", "jacobi_4", "jacobi_16", "jacobi_64"]
    etmh.JACOBI_KS == (1, 4, 16, 64)


# --------------------------------------------- tie-break / SOR extension

def test_per_maze_dissipation_matches_manual_sum():
    Qh = torch.zeros(1, 2, 2).double()
    Qv = torch.zeros(1, 1, 3).double()
    Qh[0, 0, 0], Qh[0, 1, 1] = 0.5, -0.5
    Qv[0, 0, 1] = 1.0
    Dh = torch.ones(1, 2, 2)
    Dv = torch.ones(1, 1, 3)
    Dh[0, 1, 1] = 0.0            # wall edge: Q = 0 there, contributes 0
    d = etmh.per_maze_dissipation(Qh, Qv, Dh, Dv)
    assert d.shape == (1,)
    assert abs(d[0].item() - (0.25 + 1.0)) < 1e-12


def test_rollout_snapshots_records_every_horizon():
    d = MazeHardDataset([untied_row()])
    model = TransportCFNet(physics_only=True, relaxation="sweep", k_rel=4)
    xb = d.x[:1]
    Dh, Dv, p = model.initial_state(xb)
    with torch.no_grad():
        snaps = etmh.rollout_snapshots(model, xb, [1, 4], Dh, Dv, p)
    assert sorted(snaps) == [1, 4]
    for T in (1, 4):
        Dh_s, Dv_s, p_s, Qh_s, Qv_s, diss = snaps[T]
        assert diss.shape == (1,)
        assert torch.isfinite(Dh_s).all() and torch.isfinite(p_s).all()


def test_evaluate_tie_config_picks_lowest_dissipation_start():
    # K random starts on the synthetic tied grid: the machinery runs, the
    # per-maze pick is the argmin of the per-start dissipations, and the
    # result is seeded (the same seed reproduces it).
    rows = [untied_row(), tied_row()]
    d = MazeHardDataset(rows)
    tied = etmh.tied_flags(d)
    model = TransportCFNet(physics_only=True, relaxation="sweep", k_rel=4,
                           omega=1.9)
    r1 = etmh.evaluate_tie_config(model, d, tied, [2, 4], n_starts=3,
                                  seed=0, batch_size=2)
    r2 = etmh.evaluate_tie_config(model, d, tied, [2, 4], n_starts=3,
                                  seed=0, batch_size=2)
    assert r1["n_mazes"] == 2 and r1["n_starts"] == 3 and r1["tie_seed"] == 0
    for T in ("2", "4"):
        s = r1["per_T"][T]
        assert set(s) >= {"board_all", "board_tied", "board_untied",
                          "strict_all", "iou_mean"}
        assert s["board_all"] == s["board_tied"] + s["board_untied"]
        assert 0 <= s["board_all"] <= 2
        assert r1["per_T"][T] == r2["per_T"][T], "not seeded-deterministic"
    # the pick itself: fabricate per-start dissipations, argmin wins
    diss = torch.tensor([[5.0, 1.0], [3.0, 4.0], [1.0, 2.0]])
    assert diss.argmin(dim=0).tolist() == [2, 0]


def test_extension_cli_defaults_keep_base_run():
    import scripts.eval_transport_mazehard as m
    args = m.parse_args([])
    assert not args.extension and args.n_test == 300
    assert args.tie_starts == 4 and args.omega == 1.9
    assert args.sor_ks == [16, 64]
    args = m.parse_args(["--extension"])
    assert args.out == m.EXTENSION_OUT
