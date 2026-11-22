"""Offline tests for the Maze-hard 30x30 dataset code, on synthetic rows.

Synthetic rows are built directly as 900-char question/answer strings: a
serpentine corridor (single simple S->G path, everything else walls) plus
one unlabeled free pocket cell at (2, 1), which lets us manufacture broken
rows and spur predictions without touching the real HF data.
"""
import numpy as np
import pytest
import torch

from cf.data.maze_hard import (MazeHardDataset, make_maze_hard_splits,
                               maze_hard_to_channels, parse_maze_hard,
                               validate_maze_hard, validate_maze_hard_dataset)
from cf.metrics import evaluate_mask, trace_path
from cf.train import evaluate_model

SIZE = 30


def _base_grids():
    """(wall, start, goal, label) for the synthetic serpentine maze."""
    wall = np.ones((SIZE, SIZE), bool)
    start, goal = (1, 1), (27, 1)
    cells = []
    for r in range(1, 28, 2):
        cols = range(1, 29) if r % 4 == 1 else range(28, 0, -1)
        cells.extend((r, c) for c in cols)
        if r < 27:
            cells.append((r + 1, 28 if r % 4 == 1 else 1))
    wall[2, 1] = False                              # free pocket, not on path
    label = np.zeros((SIZE, SIZE), bool)
    for r, c in cells:
        wall[r, c] = False
        label[r, c] = True
    return wall, start, goal, label


def _row_from(wall, start, goal, label, rating=1.0):
    """(wall, start, goal, label) -> dataset row dict (question/answer strings)."""
    q = np.where(wall, "#", " ").astype("<U1")
    a = np.where(wall, "#", " ").astype("<U1")
    for r, c in zip(*np.where(label)):
        a[r, c] = "o"
    q[start], a[start] = "S", "S"
    q[goal], a[goal] = "G", "G"
    return dict(source="synthetic", question="".join(q.ravel()),
                answer="".join(a.ravel()), rating=rating)


def _base_row():
    return _row_from(*_base_grids())


def _idx(r, c):
    return r * SIZE + c


def test_valid_synthetic_row_passes_all_checks():
    assert validate_maze_hard(_base_row()) == []


# name -> (mutation on the base row, expected failed-check list)
# path_on_free cannot be triggered in isolation: any label ('o'/S/G)
# character on a question '#' cell necessarily makes (qc == '#') != (ac == '#')
# true as well, so walls_match always co-fails.  The detour mutation below
# keeps the label a single simple S->G path, so exactly the two wall checks
# fire: walls_match + path_on_free.
def _q_o(row):                                     # no_o_in_question
    row["question"] = row["question"][:_idx(5, 5)] + "o" + row["question"][_idx(5, 5) + 1:]
    return row


def _q_x(row):                                     # charset
    row["question"] = row["question"][:_idx(5, 5)] + "X" + row["question"][_idx(5, 5) + 1:]
    return row


def _detour_through_wall(row):
    # pull the first U-turn (cells (1,27)..(3,27)) through the wall (2,26):
    # the label stays one simple S->G path, but now covers a wall cell
    a = list(row["answer"])
    for cell in ((1, 27), (1, 28), (2, 28), (3, 28), (3, 27)):
        a[_idx(*cell)] = " "
    a[_idx(2, 26)] = "o"
    return dict(row, answer="".join(a))


def _mutations():
    return {
        "length900": lambda r: dict(r, answer=r["answer"][:-1]),
        "charset": _q_x,
        "no_o_in_question": _q_o,
        "one_S_one_G": lambda r: dict(r, question=r["question"].replace("G", " ")),
        "walls_match": lambda r: dict(r, answer=r["answer"][:_idx(29, 15)] + " "
                                      + r["answer"][_idx(29, 15) + 1:]),
        "endpoints_match": lambda r: dict(r, answer=r["answer"][:_idx(1, 1)] + "o"
                                          + r["answer"][_idx(1, 1) + 1:_idx(1, 5)] + "S"
                                          + r["answer"][_idx(1, 5) + 1:]),
        "path_on_free": _detour_through_wall,
        "single_simple_path": lambda r: dict(r, answer=r["answer"][:_idx(2, 1)] + "o"
                                             + r["answer"][_idx(2, 1) + 1:]),
        "rating_present": lambda r: dict(r, rating=None),
    }


def _expected_fails():
    return {
        "length900": ["length900"],
        "charset": ["charset"],
        "no_o_in_question": ["no_o_in_question"],
        "one_S_one_G": ["one_S_one_G"],
        "walls_match": ["walls_match"],
        "endpoints_match": ["endpoints_match"],
        "path_on_free": ["path_on_free", "walls_match"],
        "single_simple_path": ["single_simple_path"],
        "rating_present": ["rating_present"],
    }


@pytest.mark.parametrize("name", list(_mutations()))
def test_broken_row_reports_exactly_its_check(name):
    row = _mutations()[name](_base_row())
    fails = validate_maze_hard(row)
    expected = sorted(_expected_fails()[name])
    assert sorted(fails) == expected, (name, fails)
    assert name in fails                            # the targeted check did fire


def test_parse_and_channels_shapes_and_onehot():
    row = _base_row()
    wall, start, goal, label = parse_maze_hard(row["question"], row["answer"])
    x = maze_hard_to_channels(wall, start, goal)
    assert x.shape == (3, SIZE, SIZE) and x.dtype == np.float32
    assert np.array_equal(x[0] > 0.5, ~wall)        # channel 0 = walkable
    assert int(x[1].sum()) == 1 and x[1][start] == 1.0
    assert int(x[2].sum()) == 1 and x[2][goal] == 1.0
    assert x[0][start] == 1.0 and x[0][goal] == 1.0  # S/G sit on free cells
    assert label[start] and label[goal]             # label includes S and G
    assert label.sum() == 405 and not label[2, 1]    # pocket free but unlabeled
    assert trace_path(label, start, goal) is not None


def _dihedral_variants(walk, label):
    # the exact 8 transforms __getitem__ can apply (rot90 then optional flip)
    out = []
    for k4 in range(4):
        w = torch.rot90(walk, k4, dims=(0, 1))
        l = torch.rot90(label, k4, dims=(0, 1))
        for flip in (False, True):
            out.append((torch.flip(w, dims=(1,)) if flip else w,
                        torch.flip(l, dims=(1,)) if flip else l))
    return out


def test_augment_yields_all_dihedral_variants_with_consistent_xy():
    wall, start, goal, label = _base_grids()
    walk = torch.from_numpy((~wall).astype(np.float32))
    lab = torch.from_numpy(label.astype(np.float32))
    variants = _dihedral_variants(walk, lab)
    assert len({w.numpy().tobytes() for w, _ in variants}) == 8  # all distinct

    ds = MazeHardDataset([_base_row()], augment=True, seed=0)  # seeded draws
    seen = set()
    for _ in range(200):
        x, y = ds[0]
        hit = [i for i, (w, l) in enumerate(variants)
               if torch.equal(x[0], w) and torch.equal(y, l)]
        assert len(hit) == 1                          # x and y transformed together
        seen.add(hit[0])
        s = tuple(int(v) for v in np.argwhere(x[1].numpy() > 0.5)[0])
        g = tuple(int(v) for v in np.argwhere(x[2].numpy() > 0.5)[0])
        path = trace_path((y > 0.5).numpy(), s, g)
        assert path is not None and len(path) == int(y.sum())  # still one path
    assert seen == set(range(8))                      # all 8 variants appeared


def _two_shortest_paths_grids():
    # a single open 2x2 square: two equal-length S->G paths (gt vs pred)
    wall = np.ones((SIZE, SIZE), bool)
    wall[1:3, 1:3] = False
    start, goal = (1, 1), (2, 2)
    label = np.zeros((SIZE, SIZE), bool)
    label[1, 1] = label[1, 2] = label[2, 2] = True    # via (1,2)
    other = np.zeros((SIZE, SIZE), bool)
    other[1, 1] = other[2, 1] = other[2, 2] = True    # via (2,1)
    return wall, start, goal, label, other


def test_evaluate_mask_exact_extra_cell_and_alien_shortest_path():
    wall, start, goal, label = _base_grids()
    y = label.astype(np.float32)
    r = evaluate_mask(label, wall, start, goal, y)
    assert r["strict"] == 1 and r["exact"] == 1 and r["covers_path"] == 1

    spur = label.copy()
    spur[2, 1] = True                               # + one extra free cell
    r = evaluate_mask(spur, wall, start, goal, y)
    assert r["strict"] == 0 and r["exact"] == 0 and r["extra_cells"] == 1

    w2, s2, g2, gt, other = _two_shortest_paths_grids()
    r = evaluate_mask(other, w2, s2, g2, gt.astype(np.float32))
    # the OTHER equal-length shortest path is strict-valid but not exact:
    # this is why evaluate_model's strict==exact assert is off for Maze-hard
    assert r["strict"] == 1 and r["exact"] == 0


class _FixedLogits(torch.nn.Module):
    # returns the same (30, 30) logits for every sample, every T
    def __init__(self, mask):
        super().__init__()
        logits = np.full((SIZE, SIZE), -20.0, np.float32)
        logits[mask] = 20.0
        self.logits = torch.from_numpy(logits)

    def forward(self, x, T=None):
        return self.logits[None].expand(x.shape[0], SIZE, SIZE)


def test_evaluate_model_runs_and_asserts_only_when_strict_ne_exact():
    wall2, s2, g2, gt, other = _two_shortest_paths_grids()
    row = _row_from(wall2, s2, g2, gt, rating=1.0)
    d = MazeHardDataset([row])
    assert d.walls[0].shape == (SIZE, SIZE) and d.starts[0] == s2 and d.goals[0] == g2
    model = _FixedLogits(other)                     # always predicts path B

    v, rows, probs = evaluate_model(model, d, T=1, assert_strict_equals_exact=False)
    assert rows[0]["strict"] == 1 and rows[0]["exact"] == 0
    assert v["strict"] == 1.0 and v["exact"] == 0.0
    assert probs.shape == (1, SIZE, SIZE)

    with pytest.raises(AssertionError):
        evaluate_model(model, d, T=1)               # default: strict == exact


def test_make_maze_hard_splits_sizes_disjoint_deterministic():
    tr, va = make_maze_hard_splits(900, 100, seed=0)
    assert (len(tr), len(va)) == (900, 100)
    assert set(tr).isdisjoint(va)
    assert set(tr) | set(va) == set(range(1000))
    tr2, va2 = make_maze_hard_splits(900, 100, seed=0)
    assert (tr, va) == (tr2, va2)                    # deterministic per seed
    tr1, va1 = make_maze_hard_splits(900, 100, seed=1)
    assert tr1 != tr or va1 != va                    # another seed reshuffles


def _seq(ds, n=64):
    return [ds[0] for _ in range(n)]


def test_augment_same_seed_identical_sequence():
    a = MazeHardDataset([_base_row()], augment=True, seed=123)
    b = MazeHardDataset([_base_row()], augment=True, seed=123)
    for (xa, ya), (xb, yb) in zip(_seq(a), _seq(b)):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_augment_different_seed_different_sequence():
    a = MazeHardDataset([_base_row()], augment=True, seed=123)
    b = MazeHardDataset([_base_row()], augment=True, seed=124)
    assert any(not (torch.equal(xa, xb) and torch.equal(ya, yb))
               for (xa, ya), (xb, yb) in zip(_seq(a), _seq(b)))


def test_augment_seed_none_follows_global_torch_rng():
    # seed=None -> torch's global RNG, which is what DataLoader seeds per
    # worker; re-seeding torch reproduces the same augmented sequence
    torch.manual_seed(0)
    a = MazeHardDataset([_base_row()], augment=True)
    sa = _seq(a)
    torch.manual_seed(0)
    b = MazeHardDataset([_base_row()], augment=True)
    sb = _seq(b)
    for (xa, ya), (xb, yb) in zip(sa, sb):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_augment_false_unaffected_by_seed():
    wall, start, goal, label = _base_grids()
    x0 = torch.from_numpy(maze_hard_to_channels(wall, start, goal))
    y0 = torch.from_numpy(label.astype(np.float32))
    for seed in (None, 0, 7):
        d = MazeHardDataset([_base_row()], augment=False, seed=seed)
        for _ in range(8):
            x, y = d[0]
            assert torch.equal(x, x0) and torch.equal(y, y0)


def test_validate_dataset_counts_malformed_rows_instead_of_raising():
    rows = [_base_row(),
            dict(_base_row(), answer=_base_row()["answer"][:-1]),  # length900
            dict(_base_row(), rating=None)]                       # rating_present
    rep = validate_maze_hard_dataset(rows, [])
    assert rep["failures"]["train"] == {"length900": 1, "rating_present": 1}
    # only the valid row feeds the rating / path-length stats
    assert rep["rating"]["train"]["n"] == 1 and rep["rating"]["train"]["min"] == 1.0
    assert rep["path_length"]["train"]["n"] == 1
    assert rep["path_length"]["train"]["min"] == rep["path_length"]["train"]["max"]


def test_make_maze_hard_splits_n_available_check():
    with pytest.raises(ValueError):
        make_maze_hard_splits(40, 20, n_available=50)
    with pytest.raises(ValueError):
        make_maze_hard_splits(60, n_available=50)
    tr, va = make_maze_hard_splits(40, 10, n_available=50)   # exactly enough
    assert (len(tr), len(va)) == (40, 10)
    tr2, va2 = make_maze_hard_splits(40, 10, n_available=200)  # plenty
    assert (tr2, va2) == (tr, va)                            # same permutation
    make_maze_hard_splits(40, 20)                            # None = no check
