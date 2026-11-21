"""The notebook's metric self-tests, on synthetic curriculum mazes."""
import numpy as np
import pytest

from cf.data import curriculum as cur
from cf.metrics import _metric_test_cases, evaluate_mask, summarize, trace_path

from maze_synthetic import make_batch


@pytest.fixture(scope="module")
def val_ds():
    pairs = make_batch(seed=3, count=16, min_moves=5)
    mazes, sols = map(list, zip(*pairs))
    return cur.MazeDataset(list(range(len(mazes))), mazes, sols)


def _first_case_set(ds):
    k = 0
    while True:
        res = _metric_test_cases(k, ds)
        if res is not None:
            return k, res
        k += 1
        assert k < len(ds), "no synthetic maze produced usable metric cases"


def test_metric_self_test_cases(val_ds, capsys):
    # the notebook Section-4 self-test: only the exact legal path is strict
    k, ((wall_t, s_t, g_t, y_t), cases) = _first_case_set(val_ds)
    assert k < len(val_ds)
    names = [name for name, _, _ in cases]
    assert names == ["ground truth", "empty", "all free cells", "GT + wall cell",
                     "GT + spur", "GT - one cell (gap)", "GT + island"]
    for name, pred, expected in cases:
        r = evaluate_mask(pred, wall_t, s_t, g_t, y_t)
        assert r["strict"] == expected, name
        assert r["strict"] == r["exact"] or name == "ground truth"
    print("metric self-tests passed on synthetic sample k=%d: "
          "only the exact legal path is a strict success" % k)


def test_ground_truth_is_strict_and_exact(val_ds):
    d = val_ds
    k = 0
    while _metric_test_cases(k, d) is None:
        k += 1
    wall, s, g, y = d.walls[k], d.starts[k], d.goals[k], d.y[k].numpy()
    gt = y > 0.5
    r = evaluate_mask(gt, wall, s, g, y)
    assert r["strict"] == 1 and r["exact"] == 1 and r["covers_path"] == 1
    assert r["iou"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["wall_hits"] == 0 and r["extra_cells"] == 0 and r["missed_cells"] == 0


def test_trace_path_rejects_branch_and_gap(val_ds):
    d = val_ds
    k = 0
    while _metric_test_cases(k, d) is None:
        k += 1
    (wall_t, s_t, g_t, y_t), cases = _metric_test_cases(k, d)
    wall, s, g, y = d.walls[k], d.starts[k], d.goals[k], d.y[k].numpy()
    gt = y > 0.5
    assert trace_path(gt, s, g) is not None
    assert len(trace_path(gt, s, g)) == int(gt.sum())
    # a branch: add all free cells -> ambiguous neighbours kill the trace
    assert trace_path(wall == 0, s, g) is None
    # a gap: remove the middle of the traced path
    path = trace_path(gt, s, g)
    gap = gt.copy()
    gap[path[len(path) // 2]] = False
    assert trace_path(gap, s, g) is None


def test_summarize_averages_rows(val_ds):
    d = val_ds
    wall, s, g, y = d.walls[0], d.starts[0], d.goals[0], d.y[0].numpy()
    gt = y > 0.5
    rows = [evaluate_mask(gt, wall, s, g, y), evaluate_mask(np.zeros_like(gt), wall, s, g, y)]
    agg = summarize(rows)
    assert agg["strict"] == 0.5 and agg["exact"] == 0.5
    assert set(agg) == set(rows[0])
