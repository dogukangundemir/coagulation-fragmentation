"""Helper checks for scripts/eval_voting.py (no dataset loading).

The selection, separation and inertness properties of the voting arm are
exercised on small synthetic perfect mazes (the offline stand-in helpers
next to this file): the lowest-energy start must be the selected one, the
within-maze separation AUC must be 0.5 for a blind and 1.0 for a perfect
energy, and - on maze trees, where the conserved source-to-goal flow is
unique regardless of the initial pipe conductances - all random starts of
the physics arm must produce identical predictions, so voting cannot act.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root: cf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_voting as ev                                     # noqa: E402
from cf.data.curriculum import (maze_to_channels, parse_maze,  # noqa: E402
                                replay_solution)
from cf.models.transport import TransportCFNet                # noqa: E402

from maze_synthetic import make_example                      # noqa: E402


def test_clone_guard():
    ev.ensure_clone()                                  # the real import: passes
    try:
        ev.ensure_clone(str(Path("not") / "this" / "clone" / "cf" / "__init__.py"))
    except SystemExit:
        pass
    else:
        raise AssertionError("a foreign cf path must be refused")


def test_lowest_energy_is_selected():
    e = torch.tensor([[5.0, 1.0], [1.0, 5.0], [3.0, 2.0]])
    c = np.array([[0, 1], [1, 1], [0, 0]])
    s = ev.summarize_vote(e, c, K_list=(1, 3))
    assert s["per_K"]["1"]["accuracy"] == 0.5          # first start only
    assert s["per_K"]["3"]["accuracy"] == 1.0          # lowest energy is correct
    assert s["any_correct"] == 1.0


def test_separation_auc_boundaries():
    blind = ev.within_maze_auc(np.array([[5.0, 1.0], [1.0, 5.0], [3.0, 2.0]]),
                               np.array([[1, 1], [0, 0], [0, 0]]))
    assert abs(blind - 0.5) < 1e-9
    perfect = ev.within_maze_auc(np.array([[0.0, 0.0], [1.0, 1.0]]),
                                 np.array([[1, 1], [0, 0]]))
    assert perfect == 1.0
    uninform = ev.within_maze_auc(np.array([[1.0], [1.0]]),
                                  np.array([[1], [1]]))
    assert uninform is None                            # no wrong start to pair


def small_level_set(count=4, n=3, min_moves=2):
    rng = np.random.default_rng(0)
    xs, ys, walls, starts, goals = [], [], [], [], []
    for _ in range(count):
        ascii_maze, solution = make_example(rng, n=n, min_moves=min_moves)
        wall, start, goal = parse_maze(ascii_maze)
        mask, end, legal, revisited = replay_solution(wall, start, solution)
        assert legal and end == goal and not revisited
        xs.append(maze_to_channels(wall, start, goal))
        ys.append(mask)
        walls.append(wall)
        starts.append(start)
        goals.append(goal)

    class LevelSet:
        x = torch.from_numpy(np.stack(xs)).float()
        y = torch.stack([torch.from_numpy(m).float() for m in ys])

        def __init__(self):
            self.walls, self.starts, self.goals = walls, starts, goals

        def __len__(self):
            return len(self.walls)

    return LevelSet()


def test_family_d_random_starts_are_inert_on_trees():
    """Unique tree flow: every start predicts the same cells."""
    d = small_level_set()
    model = TransportCFNet(physics_only=True, relaxation="exact")
    per_T, held = ev.eval_family_d(model, d, Ts=(2,), K=4)
    u = per_T["2"]
    assert u["n_start_disagreement"] == 0              # identical predictions
    assert u["n_mazes_mixed"] == 0                     # no maze mixes outcomes
    a1 = u["vote_dissipation"]["per_K"]["1"]["accuracy"]
    a4 = u["vote_dissipation"]["per_K"]["4"]["accuracy"]
    assert a4 == a1                                     # voting cannot act
    b1 = u["vote_leak"]["per_K"]["1"]["accuracy"]
    assert u["vote_leak"]["per_K"]["4"]["accuracy"] == b1
    assert ev.load_done  # resume surface exists
