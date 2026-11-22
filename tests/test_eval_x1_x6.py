"""Helper checks for scripts/eval_x1_x6.py (no dataset loading, no network).

The mechanism helpers and the two family evaluations are exercised on small
synthetic perfect mazes (offline stand-in helpers next to this file), in the
curriculum layout: channels [walkable, start one-hot, goal one-hot], label
path replayed from the provided solution, strict/exact metrics unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_x1_x6 as ex  # noqa: E402
from cf.data.curriculum import maze_to_channels, parse_maze, replay_solution  # noqa: E402
from cf.models.cf import CFNet  # noqa: E402
from cf.models.transport import TransportCFNet  # noqa: E402

from maze_synthetic import make_example  # noqa: E402


class FakeLevelSet:
    """The (x, y, walls, starts, goals) surface the evaluations use."""

    def __init__(self, count=3, n=5, seed=0, min_moves=4):
        rng = np.random.default_rng(seed)
        xs, ys, walls, starts, goals = [], [], [], [], []
        for _ in range(count):
            ascii_maze, solution = make_example(rng, n=n, min_moves=min_moves)
            wall, start, goal = parse_maze(ascii_maze)
            mask, end, legal, revisited = replay_solution(wall, start, solution)
            assert legal and end == goal and not revisited
            xs.append(maze_to_channels(wall, start, goal))
            ys.append(mask.astype(np.float32))
            walls.append(wall)
            starts.append(start)
            goals.append(goal)
        self.x = torch.from_numpy(np.stack(xs))
        self.y = torch.from_numpy(np.stack(ys))
        self.walls = walls
        self.starts = starts
        self.goals = goals

    def __len__(self):
        return len(self.walls)


def test_subset_view_slices_first_n():
    d = FakeLevelSet(count=5)
    v = ex.SubsetView(d, 3)
    assert len(v) == 3 and v.x.shape == (3,) + d.x.shape[1:]
    assert v.walls == d.walls[:3] and v.starts == d.starts[:3]
    v2 = ex.SubsetView(d, 10 ** 9)
    assert len(v2) == len(d)


def test_committed_stats_path_vs_dead_shares():
    # one maze 3x3, all cells free, path = the middle row
    y = torch.tensor([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]])[None]
    walk = torch.ones(1, 3, 3)
    c2 = torch.ones(1, 1, 3, 3)          # committed 2*c2 = 2 everywhere
    c1 = torch.zeros(1, 1, 3, 3)
    ms = ex.committed_stats(c1, c2, y, walk)
    assert torch.allclose(ms["com_path_share"], torch.tensor([1. / 3.]))
    assert torch.allclose(ms["com_dead_share"], torch.tensor([2. / 3.]))
    assert torch.allclose(ms["com_path_level"], torch.tensor([1.0]))
    assert torch.allclose(ms["com_dead_level"], torch.tensor([1.0]))
    # all committed mass on the path: dead share must be 0
    c2b = torch.zeros(1, 1, 3, 3)
    c2b[0, 0, 1, :] = 2.0
    ms2 = ex.committed_stats(c1, c2b, y, walk)
    assert torch.allclose(ms2["com_path_share"], torch.tensor([1.0]))
    assert torch.allclose(ms2["com_dead_share"], torch.tensor([0.0]))


def test_flow_stats_shares_and_levels():
    y = torch.tensor([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]])[None]
    walk = torch.ones(1, 3, 3)
    q = torch.zeros(1, 3, 3)
    q[0, 1, :] = 2.0                     # flow only on the path cells
    q[0, 0, 0] = 1.0                     # one dead cell carries 1
    fs = ex.flow_stats(q, y, walk)
    assert torch.allclose(fs["flow_path_share"], torch.tensor([6. / 7.]))
    assert torch.allclose(fs["flow_dead_share"], torch.tensor([1. / 7.]))
    assert torch.allclose(fs["flow_path_level"], torch.tensor([2.0]))
    assert torch.allclose(fs["flow_dead_level"], torch.tensor([1. / 6.]))


def test_family_a_runs_and_mechanism_bounds():
    d = FakeLevelSet(count=2)
    torch.manual_seed(0)
    model = CFNet(hid=4, wall_aware=True)
    r = ex.eval_family_a(model, d, [1, 2], torch.device("cpu"))
    assert r["n_mazes"] == 2
    for T in ("1", "2"):
        s = r["per_T"][T]
        assert s["strict"] == s["exact"]          # tree mazes: strict = exact
        assert 0.0 <= s["strict"] <= 1.0
        assert abs(s["com_path_share"] + s["com_dead_share"] - 1.0) < 1e-4
        assert 0.0 <= s["com_dead_share"] <= 1.0


def test_family_d_sweep_runs_and_shares_sum_to_one():
    d = FakeLevelSet(count=2)
    model = TransportCFNet(physics_only=True, relaxation="sweep", k_rel=2)
    r = ex.eval_family_d(model, d, [1, 2], torch.device("cpu"),
                         guard_breakdown=False)
    assert r["n_mazes"] == 2 and r["solver_held"]["n_held"] == 0
    for T in ("1", "2"):
        s = r["per_T"][T]
        assert 0.0 <= s["strict"] <= 1.0
        assert abs(s["flow_path_share"] + s["flow_dead_share"] - 1.0) < 1e-3


def test_family_d_exact_guard_holds_broken_solves():
    d = FakeLevelSet(count=2)
    model = TransportCFNet(physics_only=True, relaxation="exact")
    r = ex.eval_family_d(model, d, [1, 2, 4], torch.device("cpu"),
                         guard_breakdown=True)
    assert r["n_mazes"] == 2
    assert r["solver_held"]["n_held"] >= 0
    for T in ("1", "2", "4"):
        assert 0.0 <= r["per_T"][T]["strict"] <= 1.0


def test_unit_plan_keys_unique_and_complete():
    units = ex.unit_plan()
    keys = ["%s_%s_%s" % u for u in units]
    assert len(keys) == len(set(keys)) == 3 * 3 + 4 * 3
    assert ("A", "seed0", "level5_test") in units
    assert ("D", "exact", "probe9") in units
    assert ("D", "sweep16", "probe7") in units
