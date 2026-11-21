"""Curriculum maze dataset (Hugging Face selimaktas/maze-curriculum-dataset).

Code in this module is copied verbatim from CF_Milestone2_Week1_HF.ipynb;
top-level notebook statements are wrapped into functions with no behaviour
change.  The seeded split reproduces the Week-1 notebook's rng call order
EXACTLY: numpy.random.default_rng(0); one permutation of the valid level-5
pool gives train/val/test, then one permutation per probe level (7 then 9).
"""
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

CONFIG = dict(
    hf_id="selimaktas/maze-curriculum-dataset",
    license="MIT",
    week1_level=5,              # grid_size 5 -> ASCII 11x11 (easy curriculum level)
    n_train=4000, n_val=1000, n_test=1000,
    n_probe=1000,               # per harder level, Section 10 only
    seed=0,
)


def load_curriculum(hf_id=None):
    """Load the HF dataset; returns (MAZES, SOLS, LEVELS) cached column lists."""
    try:  # lazy: importing cf.data.curriculum must not require 'datasets'
        import datasets
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load the maze curriculum; "
            "install with: pip install datasets") from exc
    raw = datasets.load_dataset(hf_id or CONFIG["hf_id"])["train"]
    MAZES, SOLS, LEVELS = list(raw["maze"]), list(raw["solution"]), list(raw["grid_size"])
    return MAZES, SOLS, LEVELS


# ---- adapter: ASCII -> tensors, provided solution -> mask (notebook Section 2)
DELTAS = {"N": (-2, 0), "S": (2, 0), "E": (0, 2), "W": (0, -2)}


def parse_maze(maze_str):
    # ASCII maze -> wall grid (int8, 1 = wall), start (r, c), goal (r, c)
    lines = [ln for ln in maze_str.split("\n") if ln.strip()]
    H, W = len(lines), len(lines[0])
    wall = np.zeros((H, W), np.int8)
    start = goal = None
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "#":
                wall[r, c] = 1
            elif ch == "S":
                start = (r, c)
            elif ch == "E":
                goal = (r, c)
    return wall, start, goal


def maze_to_channels(wall, start, goal):
    # wall grid + endpoints -> model input (3, H, W): walkable, start one-hot, goal one-hot
    walk = (wall == 0).astype(np.float32)
    s = np.zeros(wall.shape, np.float32)
    s[start] = 1.0
    g = np.zeros(wall.shape, np.float32)
    g[goal] = 1.0
    return np.stack([walk, s, g])


def replay_solution(wall, start, solution):
    # PROVIDED move string -> path mask by pure rule-following (no search).
    # Returns (mask, end_cell, legal, revisited).
    H, W = wall.shape
    mask = np.zeros((H, W), bool)
    mask[start] = True
    r, c = start
    revisited = False
    for mv in solution:
        if mv not in DELTAS:
            return mask, (r, c), False, revisited
        dr, dc = DELTAS[mv]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            return mask, (r, c), False, revisited
        if wall[r + dr // 2, c + dc // 2] or wall[nr, nc]:
            return mask, (r, c), False, revisited
        revisited |= bool(mask[nr, nc])
        mask[r + dr // 2, c + dc // 2] = True
        mask[nr, nc] = True
        r, c = nr, nc
    return mask, (r, c), True, revisited


class MazeDataset(Dataset):
    # (x, y) pairs for training + per-sample metadata for evaluation / plots
    # (notebook version indexed the module-level MAZES/SOLS columns; they are
    # passed explicitly here — behaviour is unchanged)
    def __init__(self, indices, mazes, sols):
        self.indices = list(indices)
        xs, ys = [], []
        self.walls, self.starts, self.goals, self.solutions = [], [], [], []
        for i in self.indices:
            wall, start, goal = parse_maze(mazes[i])
            mask, end, legal, revisited = replay_solution(wall, start, sols[i])
            if not legal or end != goal or revisited:
                raise ValueError("row %d: provided solution does not replay to E" % i)
            xs.append(maze_to_channels(wall, start, goal))
            ys.append(mask.astype(np.float32))
            self.walls.append(wall)
            self.starts.append(start)
            self.goals.append(goal)
            self.solutions.append(sols[i])
        self.x = torch.from_numpy(np.stack(xs))
        self.y = torch.from_numpy(np.stack(ys))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        return self.x[k], self.y[k]


# ---- validation of all samples (notebook Section 3)
def _root(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def is_perfect_maze(wall):
    # union-find over navigable cells: perfect <=> no cycle and exactly n*n - 1 open passages
    H, W = wall.shape
    n = (H - 1) // 2
    parent = list(range(n * n))
    passages = 0
    for r in range(1, H - 1, 2):
        for c in range(1, W - 1, 2):
            for dr, dc in ((0, 1), (1, 0)):
                if r + 2 * dr < H - 1 and c + 2 * dc < W - 1 and wall[r + dr, c + dc] == 0:
                    a = _root(parent, (r // 2) * n + c // 2)
                    b = _root(parent, ((r + 2 * dr) // 2) * n + (c + 2 * dc) // 2)
                    if a == b:
                        return False          # cycle -> more than one route
                    parent[a] = b
                    passages += 1
    return passages == n * n - 1             # spanning tree -> connected


def canonical_key(maze_str):
    # identical up to the 8 rotations / reflections -> same key
    lines = [ln for ln in maze_str.split("\n") if ln.strip()]
    A = np.frombuffer("".join(lines).encode(), np.uint8).reshape(len(lines), -1)
    views = []
    for k in range(4):
        R = np.rot90(A, k)
        views += [R.tobytes(), R[:, ::-1].tobytes()]
    return min(views)


CHECKS = ["shape", "one S / one E", "endpoints on lattice", "border", "lattice",
          "perfect maze", "legal replay", "ends on E", "simple path"]


def validate_example(maze_str, solution, n):
    lines = [ln for ln in maze_str.split("\n") if ln.strip()]
    if len(lines) != 2 * n + 1 or any(len(ln) != 2 * n + 1 for ln in lines):
        return ["shape"]
    if maze_str.count("S") != 1 or maze_str.count("E") != 1:
        return ["one S / one E"]
    wall, s, g = parse_maze(maze_str)
    fails = []
    if s == g or not all(v % 2 == 1 for v in s + g):
        fails.append("endpoints on lattice")
    if not (wall[0].all() and wall[-1].all() and wall[:, 0].all() and wall[:, -1].all()):
        fails.append("border")
    if wall[1::2, 1::2].any() or not wall[::2, ::2].all():
        fails.append("lattice")
    if not is_perfect_maze(wall):
        fails.append("perfect maze")
    mask, end, legal, revisited = replay_solution(wall, s, solution)
    if not legal:
        fails.append("legal replay")
    elif end != g:
        fails.append("ends on E")
    if revisited:
        fails.append("simple path")
    return fails


def validate_dataset(mazes, sols, levels):
    """Validate every sample exactly as the notebook does.

    Returns (valid, fail_counts, exact_dups, canon_dups): `valid` maps each
    level to the list of row indices passing every check.
    """
    fail_counts = {lv: Counter() for lv in sorted(set(levels))}
    valid = {lv: [] for lv in fail_counts}
    exact_seen, canon_seen = {}, {}
    exact_dups = canon_dups = 0
    for i, (m, sol, lv) in enumerate(zip(mazes, sols, levels)):
        fails = validate_example(m, sol, lv)
        if m in exact_seen:
            exact_dups += 1
            fails.append("duplicate (exact)")
        else:
            exact_seen[m] = i
        key = canonical_key(m)
        if key in canon_seen and "duplicate (exact)" not in fails:
            canon_dups += 1
            fails.append("duplicate (symmetry)")
        canon_seen.setdefault(key, i)
        for f in fails:
            fail_counts[lv][f] += 1
        if not fails:
            valid[lv].append(i)
    return valid, fail_counts, exact_dups, canon_dups


def make_splits(mazes, sols, levels, *, week1_level=5, n_train=4000, n_val=1000,
                n_test=1000, n_probe=1000, probe_levels=(7, 9), seed=0,
                validate=True):
    """Reproduce the Week-1 notebook split EXACTLY (same rng call order).

    Validating (when ``validate=True``) runs the notebook's 10-check +
    duplicate scan over every row.  The rng use order is identical to the
    notebook: one permutation of the valid level-5 pool, then one permutation
    per probe level, in ``(7, 9)`` order.

    Returns (splits, valid, fail_counts, exact_dups, canon_dups) where
    splits is {"train": [...], "val": [...], "test": [...], "probe7": [...],
    "probe9": [...]} of dataset row indices.
    """
    if validate:
        valid, fail_counts, exact_dups, canon_dups = validate_dataset(mazes, sols, levels)
    else:
        fail_counts = {lv: Counter() for lv in sorted(set(levels))}
        valid = {lv: [i for i, g in enumerate(levels) if g == lv] for lv in sorted(set(levels))}
        exact_dups = canon_dups = 0

    # ---- seeded split of the valid level-5 pool
    rng = np.random.default_rng(seed)
    pool = valid[week1_level]
    perm = rng.permutation(len(pool))
    n_tr, n_va, n_te = n_train, n_val, n_test
    pick = [pool[j] for j in perm[: n_tr + n_va + n_te]]
    splits = {"train": pick[:n_tr], "val": pick[n_tr:n_tr + n_va], "test": pick[n_tr + n_va:]}

    # split-leakage safety check (notebook cell 8): no maze, even up to a
    # rotation/reflection, may appear in two of train/val/test.  Pure set
    # arithmetic -- consumes no random numbers, so the rng order is unchanged.
    keys = {k: {canonical_key(mazes[i]) for i in v} for k, v in splits.items()}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) \
        and not (keys["val"] & keys["test"]), "split leakage"

    # harder levels, used only by the Section-10 probe (never for training)
    probe = {lv: [valid[lv][j] for j in rng.permutation(len(valid[lv]))[:n_probe]]
             for lv in probe_levels}
    for lv, idx in probe.items():
        splits["probe%d" % lv] = idx
    return splits, valid, fail_counts, exact_dups, canon_dups


def build_datasets(splits, mazes, sols):
    """MazeDataset per split name, plus the curriculum probe sets."""
    out = {name: MazeDataset(idx, mazes, sols) for name, idx in splits.items()
           if name.startswith(("train", "val", "test"))}
    out["probe"] = {lv: MazeDataset(splits["probe%d" % lv], mazes, sols)
                    for lv in (7, 9) if "probe%d" % lv in splits}
    return out


def train_pos_weight(d):
    """pos_weight for BCE (from train label density over all cells)."""
    density = d.y.mean().item()
    return (1.0 - density) / density
