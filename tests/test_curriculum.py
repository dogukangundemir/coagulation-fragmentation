"""Offline tests of the curriculum adapter on synthetic lattice mazes."""
import numpy as np
import pytest

from cf.data import curriculum as cur

from maze_synthetic import make_ascii, make_batch, make_example


@pytest.fixture(scope="module")
def example():
    ex, sol = make_example(np.random.default_rng(0), n=5, min_moves=4)
    return ex, sol


# ---- valid examples are accepted ------------------------------------------------
def test_parse_and_channels_accept_valid(example):
    maze_str, sol = example
    wall, start, goal = cur.parse_maze(maze_str)
    assert start is not None and goal is not None
    assert wall.shape == (11, 11) and wall.dtype == np.int8
    assert wall[start] == 0 and wall[goal] == 0
    x = cur.maze_to_channels(wall, start, goal)
    assert x.shape == (3, 11, 11) and x.dtype == np.float32
    assert x[0][start] == 1 and x[0][wall == 1].sum() == 0  # walkable channel
    assert x[1].sum() == 1 and x[1][start] == 1             # start one-hot
    assert x[2].sum() == 1 and x[2][goal] == 1              # goal one-hot


def test_replay_solution_accepts_valid(example):
    maze_str, sol = example
    wall, start, goal = cur.parse_maze(maze_str)
    mask, end, legal, revisited = cur.replay_solution(wall, start, sol)
    assert legal and end == goal and not revisited
    assert mask[start] and mask[goal]
    assert not (mask & (wall == 1)).any()
    assert int(mask.sum()) == 2 * len(sol) + 1


def test_validate_example_accepts_valid(example):
    maze_str, sol = example
    wall, _, _ = cur.parse_maze(maze_str)
    assert cur.is_perfect_maze(wall)
    assert cur.validate_example(maze_str, sol, 5) == []


def test_curriculum_dataset_builds_from_valid_rows():
    mazes, sols = map(list, zip(*make_batch(seed=1, count=4)))
    ds = cur.MazeDataset([0, 1, 2, 3], mazes, sols)
    assert len(ds) == 4
    x, y = ds[0]
    assert tuple(x.shape) == (3, 11, 11) and tuple(y.shape) == (11, 11)
    assert ds.x.shape == (4, 3, 11, 11) and ds.y.shape == (4, 11, 11)


def test_canonical_key_is_dihedral_invariant(example):
    maze_str, _ = example
    lines = [ln for ln in maze_str.split("\n") if ln.strip()]
    A = np.array([[ord(ch) for ch in ln] for ln in lines], np.uint8)
    variants = []
    for k in range(4):
        R = np.rot90(A, k)
        chars = ["".join(chr(v) for v in row) for row in R]
        variants.append("\n".join(chars))
        chars = ["".join(chr(v) for v in row) for row in R[:, ::-1]]
        variants.append("\n".join(chars))
    keys = {cur.canonical_key(v) for v in variants}
    assert len(keys) == 1


# ---- broken examples are rejected ----------------------------------------------
def _broken(maze_str, sol):
    return cur.validate_example(maze_str, sol, 5)


def test_reject_wrong_shape(example):
    maze_str, sol = example
    lines = [ln for ln in maze_str.split("\n") if ln.strip()]
    assert "shape" in _broken("\n".join(lines[:-1]), sol)
    narrow = [ln[:-1] for ln in lines]
    assert "shape" in _broken("\n".join(narrow), sol)


def test_reject_missing_endpoint(example):
    maze_str, sol = example
    assert "one S / one E" in _broken(maze_str.replace("E", " "), sol)
    assert "one S / one E" in _broken(maze_str.replace("S", " "), sol)


def test_reject_endpoint_off_lattice(example):
    maze_str, sol = example
    wall, start, goal = cur.parse_maze(maze_str)
    # move S onto an even coordinate (row 0 -> wall cell of the border)
    bad = wall.copy()
    sr, sc = start
    bad[sr, sc] = 1
    bad[sr - 1, sc] = 0            # (0, 1): even row, odd col -> off-lattice start
    maze2 = make_ascii(bad, (sr - 1, sc), goal)
    fails = _broken(maze2, sol)
    assert "endpoints on lattice" in fails


def test_reject_border_hole(example):
    maze_str, sol = example
    wall, _, goal = cur.parse_maze(maze_str)
    bad = wall.copy()
    bad[0, 5] = 0                  # hole in the outer border
    maze2 = make_ascii(bad, (1, 1), goal)
    assert "border" in _broken(maze2, sol)


def test_reject_lattice_violation(example):
    maze_str, sol = example
    wall, _, goal = cur.parse_maze(maze_str)
    bad = wall.copy()
    r, c = next((r, c) for r in range(1, 10, 2) for c in range(1, 10, 2)
                if bad[r, c] == 0 and (r, c) not in ((1, 1), goal))
    bad[r, c] = 1                  # wall sitting on a lattice cell
    maze2 = make_ascii(bad, (1, 1), goal)
    assert "lattice" in _broken(maze2, sol)


def test_reject_non_perfect_maze(example):
    maze_str, sol = example
    wall, _, goal = cur.parse_maze(maze_str)
    bad = wall.copy()
    # open an interior wall segment between two free lattice cells -> a cycle
    opened = False
    for r in range(1, 10):
        for c in range(1, 10):
            if bad[r, c] == 1 and r % 2 != c % 2:      # horizontal/vertical segment
                if bad[r - 1, c] == 0 and bad[r + 1, c] == 0:
                    bad[r, c] = 0
                    opened = True
                    break
        if opened:
            break
    assert opened, "test maze unexpectedly has no closed segment to open"
    assert not cur.is_perfect_maze(bad)
    maze2 = make_ascii(bad, (1, 1), goal)
    assert "perfect maze" in _broken(maze2, sol)


def test_reject_illegal_replay(example):
    maze_str, _ = example
    # a move off the grid from S=(1,1) is immediately illegal
    assert "legal replay" in _broken(maze_str, "N" + "E" * 20)
    # a move straight through a wall is illegal
    wall, start, goal = cur.parse_maze(maze_str)
    bad_move = next(m for m in "NSEW"
                   if not (0 <= start[0] + cur.DELTAS[m][0] < 11
                           and 0 <= start[1] + cur.DELTAS[m][1] < 11
                           and wall[start[0] + cur.DELTAS[m][0] // 2,
                                   start[1] + cur.DELTAS[m][1] // 2] == 0
                           and wall[start[0] + cur.DELTAS[m][0],
                                   start[1] + cur.DELTAS[m][1]] == 0))
    assert "legal replay" in _broken(maze_str, bad_move)


def test_reject_solution_not_ending_on_e(example):
    maze_str, sol = example
    wall, start, goal = cur.parse_maze(maze_str)
    mask, end, legal, revisited = cur.replay_solution(wall, start, sol[:-1])
    assert legal and end != goal and not revisited
    assert "ends on E" in _broken(maze_str, sol[:-1])


def test_reject_revisiting_solution(example):
    maze_str, sol = example
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    weird = sol + opposite[sol[-1]]     # reach E, then step back onto a visited cell
    wall, start, goal = cur.parse_maze(maze_str)
    mask, end, legal, revisited = cur.replay_solution(wall, start, weird)
    assert legal and revisited
    assert "simple path" in _broken(maze_str, weird)


def test_curriculum_dataset_rejects_bad_row():
    mazes, sols = map(list, zip(*make_batch(seed=2, count=2)))
    sols[1] = sols[1][:-1]        # truncated solution does not end on E
    with pytest.raises(ValueError):
        cur.MazeDataset([0, 1], mazes, sols)


# ---- importing the module must not require 'datasets' ---------------------------
def test_import_works_without_datasets(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "datasets", None)   # 'import datasets' now fails
    mod = importlib.reload(cur)                         # must not run pip
    assert mod.CONFIG["hf_id"] == cur.CONFIG["hf_id"]
    with pytest.raises(ImportError, match="pip install datasets"):
        mod.load_curriculum()


def test_load_curriculum_raises_on_reloaded_module(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "datasets", None)
    importlib.reload(cur)
    with pytest.raises(ImportError, match="pip install datasets"):
        cur.load_curriculum()                     # rebound reference, still lazy

