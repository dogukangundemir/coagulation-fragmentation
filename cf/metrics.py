"""Strict path evaluation functions (no search).

Code copied verbatim from CF_Milestone2_Week1_HF.ipynb Section 4; the only
change: _metric_test_cases takes the dataset explicitly instead of reading
the notebook-global `ds["val"]`.

Metric definitions (evaluate_mask):
- exact: cell-wise match to the label path — the prediction must equal the
  reference label mask exactly.  This is WONN's board accuracy and the
  comparable number on Maze-hard.
- strict: any valid wall-free single simple S->G path of the label's length.
  On perfect mazes the two coincide (strict == exact); on non-perfect grids
  where several S->G shortest paths exist (e.g. Maze-hard) strict is more
  lenient: a different, equally short path scores strict = 1 yet exact = 0.
"""
import numpy as np

NBR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pred_neighbours(pred, cell):
    H, W = pred.shape
    out = []
    for dr, dc in NBR4:
        r, c = cell[0] + dr, cell[1] + dc
        if 0 <= r < H and 0 <= c < W and pred[r, c]:
            out.append((r, c))
    return out


def trace_path(pred, start, goal):
    # deterministic trace S -> G through predicted cells; None if ambiguous / broken
    if not pred[start] or not pred[goal] or start == goal:
        return None
    path, prev, cur, limit = [start], None, start, int(pred.sum())
    while cur != goal:
        nxt = [n for n in _pred_neighbours(pred, cur) if n != prev]
        if len(nxt) != 1:
            return None                       # dead end or branch
        prev, cur = cur, nxt[0]
        path.append(cur)
        if len(path) > limit:
            return None                       # cycle guard
    if len(_pred_neighbours(pred, goal)) != 1:
        return None                           # G must be an endpoint
    return path


def evaluate_mask(pred, wall, start, goal, y):
    # pred: bool (H, W); wall: int8 (1 = wall); y: float/bool label mask (H, W) - all NumPy
    pred = np.asarray(pred, bool)
    gt = np.asarray(y) > 0.5
    free = wall == 0
    wall_hits = int((pred & ~free).sum())
    path = trace_path(pred, start, goal)
    single_path = path is not None and len(path) == int(pred.sum())
    strict = wall_hits == 0 and single_path and len(path) == int(gt.sum())
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    return dict(
        strict=int(strict),
        exact=int((pred == gt).all()),
        iou=tp / max(tp + fp + fn, 1),
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        covers_path=int(fn == 0 and wall_hits == 0),
        wall_hits=wall_hits,
        extra_cells=fp,
        missed_cells=fn,
    )


def summarize(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def _metric_test_cases(k, dataset):
    # metric self-test cases built on sample k of `dataset` (the notebook's
    # validation split); returns None if the maze is unsuitable
    d = dataset
    wall, s, g, y = d.walls[k], d.starts[k], d.goals[k], d.y[k].numpy()
    gt = y > 0.5
    free = wall == 0
    path = trace_path(gt, s, g)
    near = [c for c in zip(*np.where(free & ~gt)) if any(gt[n] for n in _pred_neighbours(free, c))]
    far = [c for c in zip(*np.where(free & ~gt))
           if not any(gt[n] for n in _pred_neighbours(free, c))]
    if path is None or len(path) < 5 or not near or not far:
        return None
    gap = gt.copy(); gap[path[len(path) // 2]] = False
    spur = gt.copy(); spur[near[0]] = True
    island = gt.copy(); island[far[0]] = True
    wallhit = gt.copy(); wallhit[np.argwhere(wall == 1)[0][0], np.argwhere(wall == 1)[0][1]] = True
    cases = [("ground truth", gt, 1), ("empty", np.zeros_like(gt), 0), ("all free cells", free, 0),
             ("GT + wall cell", wallhit, 0), ("GT + spur", spur, 0), ("GT - one cell (gap)", gap, 0),
             ("GT + island", island, 0)]
    return (wall, s, g, y), cases
