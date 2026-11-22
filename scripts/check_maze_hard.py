"""Sanity-check the Maze-hard 30x30 dataset and optionally render figures.

Prints a clean table of the per-check failure counts (train/test), duplicate
counts, rating and path-length statistics, and how often `rating` equals the
number of label cells (S+G+o).  With --figures it also writes two PNGs to
--out (default figures/maze_hard).
"""
import argparse
from pathlib import Path

import numpy as np

from cf.data.maze_hard import (CHECKS, load_maze_hard, parse_maze_hard,
                               validate_maze_hard_dataset)


def label_len(row):
    return int(parse_maze_hard(row["question"], row["answer"])[3].sum())


def count_rating_matches(rows):
    return sum(1 for row in rows if float(row["rating"]) == label_len(row))


def print_report(rep, rows_train, rows_test):
    print("Maze-hard (%s): %d train / %d test grids"
          % ("30x30", len(rows_train), len(rows_test)))
    print("\ncheck                  train  test  (failed rows)")
    for chk in CHECKS:
        ft = rep["failures"]["train"].get(chk, 0)
        te = rep["failures"]["test"].get(chk, 0)
        print("%-22s %5d %5d" % (chk, ft, te))
    d = rep["duplicates"]
    print("\nduplicates: within_train=%d within_test=%d across_splits=%d"
          % (d["within_train"], d["within_test"], d["across_splits"]))
    for name in ("train", "test"):
        r = rep["rating"][name]
        p = rep["path_length"][name]
        print("\n%s (n=%d): rating min=%d mean=%.2f max=%d"
              % (name, r["n"], r["min"], r["mean"], r["max"]))
        print("%s (n=%d): path length min=%d mean=%.2f max=%d cells"
              % (name, p["n"], p["min"], p["mean"], p["max"]))
    n_t = count_rating_matches(rows_train)
    n_te = count_rating_matches(rows_test)
    print("\nrating == number of label cells (S+G+o) for %d/%d train, %d/%d test"
          % (n_t, len(rows_train), n_te, len(rows_test)))
    return n_t, n_te


def _grid_rgb(wall, start, goal, label):
    rgb = np.ones(wall.shape + (3,))
    rgb[wall] = (0.0, 0.0, 0.0)          # walls black, free white
    rgb[label] = (0.0, 0.35, 0.1)        # label path dark green
    rgb[start] = (0.9, 0.1, 0.1)         # S red
    rgb[goal] = (0.1, 0.2, 0.9)          # G blue
    return rgb


def _finding(rep, rows_train, rows_test):
    lo = min(rep["path_length"][s]["min"] for s in ("train", "test"))
    hi = max(rep["path_length"][s]["max"] for s in ("train", "test"))
    n_bad = sum(sum(f.values()) for f in rep["failures"].values())
    total = len(rows_train) + len(rows_test)
    ok = "all %d grids valid" % total if n_bad == 0 else "%d check failures" % n_bad
    return "Maze-hard 30x30: %s, path %d-%d cells" % (ok, lo, hi)


def make_figures(rep, rows_train, rows_test, n_t, n_te, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    finding = _finding(rep, rows_train, rows_test)

    picks = [(i, r, "train") for i, r in enumerate(rows_train[:2])] + \
            [(i, r, "test") for i, r in enumerate(rows_test[:2])]
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for ax, (i, row, split) in zip(axes.flat, picks):
        wall, s, g, label = parse_maze_hard(row["question"], row["answer"])
        ax.imshow(_grid_rgb(wall, s, g, label), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%s #%d, path = %d cells" % (split, i, int(label.sum())), fontsize=10)
    fig.suptitle(finding, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "maze_hard_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in (("train", rows_train), ("test", rows_test)):
        ax.hist([label_len(r) for r in rows], bins=30, alpha=0.6, label=name)
    lo = min(rep["path_length"][s]["min"] for s in ("train", "test"))
    hi = max(rep["path_length"][s]["max"] for s in ("train", "test"))
    ax.set_xlabel("path length (cells)")
    ax.set_ylabel("number of mazes")
    ax.set_title("path length range: %d-%d cells (train vs test)" % (lo, hi))
    ax.legend()
    fig.suptitle(finding, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "maze_hard_path_length_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\nfigures written to %s" % out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true", help="write the two PNG figures")
    ap.add_argument("--out", default="figures/maze_hard", help="figure output directory")
    args = ap.parse_args()
    rows_train = load_maze_hard("train")
    rows_test = load_maze_hard("test")
    rep = validate_maze_hard_dataset(rows_train, rows_test)
    n_t, n_te = print_report(rep, rows_train, rows_test)
    if args.figures:
        make_figures(rep, rows_train, rows_test, n_t, n_te, Path(args.out))


if __name__ == "__main__":
    main()
