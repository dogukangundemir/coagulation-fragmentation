"""Hard 30x30 maze dataset (Hugging Face sapientinc/maze-30x30-hard-1k).

Unlike the curriculum mazes, these grids may be NON-PERFECT (several
shortest routes can exist between S and G) and free cells may touch at
corners only.  Consequently a strict cell-level metric must NOT assert
strict == exact on this dataset: a prediction can differ cell-wise from
the reference label while still being a valid strict S->G path.
"""
import csv
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

CONFIG = dict(
    hf_id="sapientinc/maze-30x30-hard-1k",
    size=30,                      # 30x30 -> 900-char row-major strings
    charset="# SGo",
    splits=("train", "test"),
)


def download_maze_hard(split):
    """Download one split csv of the HF dataset; returns the local path."""
    if split not in CONFIG["splits"]:
        raise ValueError("unknown split %r; expected one of %s" % (split, CONFIG["splits"]))
    try:  # lazy: importing cf.data.maze_hard must not require huggingface_hub
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required to download the maze-hard "
            "dataset; install with: pip install huggingface_hub") from exc
    return hf_hub_download(repo_id=CONFIG["hf_id"], filename="%s.csv" % split,
                            repo_type="dataset")


def load_maze_hard(split):
    """Load a split as a list of dicts (source, question, answer, rating)."""
    path = download_maze_hard(split)
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["source", "question", "answer", "rating"]:
            raise ValueError("unexpected header in %s: %r" % (path, header))
        rows = []
        for rec in reader:
            if not rec:
                continue
            source, question, answer, rating = rec
            rows.append(dict(source=source, question=question, answer=answer,
                            rating=float(rating)))
    return rows


def parse_maze_hard(question, answer, size=30):
    """(question, answer) strings -> (wall, start, goal, label).

    wall: (H, W) bool, True = '#';  start/goal: (r, c) tuples;  label:
    (H, W) bool, True on path cells ('o') and on S / G in the answer.
    """
    n = size * size
    if len(question) != n or len(answer) != n:
        raise ValueError("expected %d-char strings, got %d / %d" % (n, len(question), len(answer)))
    wall = np.frombuffer(question.encode("ascii"), dtype=np.uint8).reshape(size, size) == ord("#")
    ans = np.frombuffer(answer.encode("ascii"), dtype=np.uint8).reshape(size, size)
    label = (ans == ord("o")) | (ans == ord("S")) | (ans == ord("G"))
    q = np.frombuffer(question.encode("ascii"), dtype=np.uint8).reshape(size, size)
    start_pos = np.argwhere(q == ord("S"))
    goal_pos = np.argwhere(q == ord("G"))
    if len(start_pos) != 1 or len(goal_pos) != 1:
        raise ValueError("question must contain exactly one S and one G")
    start = tuple(int(v) for v in start_pos[0])
    goal = tuple(int(v) for v in goal_pos[0])
    return wall, start, goal, label


def maze_hard_to_channels(wall, start, goal):
    # same convention as curriculum.maze_to_channels: (3, H, W)
    # = [walkable (not wall), start one-hot, goal one-hot], float32
    walk = (~np.asarray(wall, bool)).astype(np.float32)
    s = np.zeros(wall.shape, np.float32)
    s[start] = 1.0
    g = np.zeros(wall.shape, np.float32)
    g[goal] = 1.0
    return np.stack([walk, s, g])


CHECKS = ["length900", "charset", "no_o_in_question", "one_S_one_G",
          "walls_match", "endpoints_match", "path_on_free",
          "single_simple_path", "rating_present"]


def _label_neighbours(grid, cell):
    H, W = grid.shape
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = cell[0] + dr, cell[1] + dc
        if 0 <= r < H and 0 <= c < W and grid[r, c]:
            out.append((r, c))
    return out


def _trace_simple(label, start, goal):
    # deterministic 4-neighbour walk from S through label cells to G;
    # returns True iff every label cell is used, with no branching
    n_label = int(label.sum())
    if not label[start] or not label[goal] or start == goal:
        return False
    visited = 1
    prev, cur = None, start
    while cur != goal:
        nxt = [n for n in _label_neighbours(label, cur) if n != prev]
        if len(nxt) != 1:
            return False                       # dead end or branch
        prev, cur = cur, nxt[0]
        visited += 1
        if visited > n_label:
            return False                       # cycle guard
    return visited == n_label


def validate_maze_hard(row):
    """Return the list of failed check names for one dataset row."""
    size = CONFIG["size"]
    n = size * size
    q, a = row["question"], row["answer"]
    fails = []

    if len(q) != n or len(a) != n:
        return ["length900"]
    charset = set(CONFIG["charset"])
    if not (set(q) <= charset and set(a) <= charset):
        fails.append("charset")
    if "o" in q:
        fails.append("no_o_in_question")
    if q.count("S") != 1 or q.count("G") != 1 or a.count("S") != 1 or a.count("G") != 1:
        fails.append("one_S_one_G")
    if fails:
        return fails                            # cannot parse a malformed row

    wall, start, goal, label = parse_maze_hard(q, a, size)
    if [i for i, (qc, ac) in enumerate(zip(q, a)) if (qc == "#") != (ac == "#")]:
        fails.append("walls_match")
    if q.index("S") != a.index("S") or q.index("G") != a.index("G"):
        fails.append("endpoints_match")
    if (label & wall).any():
        fails.append("path_on_free")
    if (not _trace_simple(label, start, goal)
            or len(_label_neighbours(label, start)) != 1
            or len(_label_neighbours(label, goal)) != 1):
        fails.append("single_simple_path")
    rating = row.get("rating")
    if rating is None or not isinstance(rating, (int, float)) or rating != rating:
        fails.append("rating_present")
    return fails


def validate_maze_hard_dataset(rows_train, rows_test):
    """Dataset-level validation report.

    Returns a dict with: failures per check per split, duplicate question
    counts (within train, within test, across splits), per-split rating
    min/mean/max, and per-split label path length min/mean/max.
    """
    out = {"failures": {}, "duplicates": {}, "rating": {}, "path_length": {}}
    questions = {}
    for name, rows in (("train", rows_train or []), ("test", rows_test or [])):
        counts = Counter()
        ratings, lengths = [], []
        for row in rows:
            fails = validate_maze_hard(row)
            counts.update(fails)
            if not fails:                      # only parse/count valid rows
                ratings.append(float(row["rating"]))
                lengths.append(int(parse_maze_hard(row["question"],
                                                   row["answer"])[3].sum()))
        out["failures"][name] = {c: counts[c] for c in CHECKS if counts[c]}
        qs = [r["question"] for r in rows]
        questions[name] = qs
        out["duplicates"]["within_%s" % name] = len(qs) - len(set(qs))
        out["rating"][name] = dict(n=len(ratings),
                                   min=min(ratings), mean=float(np.mean(ratings)),
                                   max=max(ratings)) if ratings else {}
        out["path_length"][name] = dict(n=len(lengths),
                                        min=min(lengths), mean=float(np.mean(lengths)),
                                        max=max(lengths)) if lengths else {}
    out["duplicates"]["across_splits"] = len(set(questions["train"]) & set(questions["test"]))
    return out


class MazeHardDataset(Dataset):
    """(x, y) tensors: x float32 (3, 30, 30) via maze_hard_to_channels,
    y float32 (30, 30) path label; augment=True applies one of the 8
    dihedral transforms uniformly at random, consistently on x and y.

    walls/starts/goals/x/y always hold the UN-augmented mazes (augmentation
    applies only through __getitem__): evaluation reads them directly via
    cf.train.predict_probs, which walks the stored wall/start/goal in the
    reference orientation, so augment must stay False for val/test data.

    Augmentation RNG: the transform index is drawn from torch's global RNG
    (torch.randint), which cf.train.train_model and the DataLoader seed
    per worker, so workers never share or repeat one numpy stream.  Pass
    seed=<int> to draw instead from a private torch.Generator seeded with
    it, which makes the augmented sample sequence reproducible.
    """
    def __init__(self, rows, augment=False, seed=None):
        self.rows = list(rows)
        self.augment = bool(augment)
        xs, ys = [], []
        # per-sample metadata consumed by cf.train.evaluate_model /
        # cf.metrics._metric_test_cases (same convention as curriculum)
        self.walls, self.starts, self.goals = [], [], []
        for row in self.rows:
            wall, start, goal, label = parse_maze_hard(row["question"], row["answer"])
            self.walls.append(wall)
            self.starts.append(start)
            self.goals.append(goal)
            xs.append(maze_hard_to_channels(wall, start, goal))
            ys.append(label.astype(np.float32))
        self.x = torch.from_numpy(np.stack(xs)) if xs else torch.zeros((0, 3, 30, 30))
        self.y = torch.from_numpy(np.stack(ys)) if ys else torch.zeros((0, 30, 30))
        self.rng = np.random.default_rng(seed)
        self._aug_gen = None
        if seed is not None:
            self._aug_gen = torch.Generator()
            self._aug_gen.manual_seed(int(seed))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        x, y = self.x[k], self.y[k]
        if self.augment:
            # torch RNG: seeded per DataLoader worker (train_model seeds it
            # globally), or by self._aug_gen when seed= was given
            k8 = int(torch.randint(8, (1,), generator=self._aug_gen))
            k4, flip = k8 % 4, k8 >= 4
            x, y = torch.rot90(x, k4, dims=(1, 2)), torch.rot90(y, k4, dims=(0, 1))
            if flip:
                x, y = torch.flip(x, dims=(2, )), torch.flip(y, dims=(1, ))
        return x, y


def make_maze_hard_splits(n_train, n_val=100, seed=0, n_available=None):
    """Seeded (train_idx, val_idx) into the train.csv rows; test is test.csv.

    Assumes train.csv holds at least n_train + n_val rows (pass n_available
    to check that upfront); the valid pool is permuted once with numpy
    default_rng(seed).
    """
    if n_available is not None and n_train + n_val > n_available:
        raise ValueError("n_train + n_val = %d exceeds n_available = %d"
                         % (n_train + n_val, n_available))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train + n_val)
    return perm[:n_train].tolist(), perm[n_train:n_train + n_val].tolist()
