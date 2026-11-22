"""Sanity-check the Maze-hard 30x30 dataset and optionally render figures.

Prints a clean table of the per-check failure counts (train/test), duplicate
counts, rating and path-length statistics, how often `rating` equals the
number of label cells (S+G+o), and the per-maze count of shortest S->G
paths (BFS count, capped at 3, over the validation-passing rows only).  It always writes the validation table to
--out/validation_report.json and --out/validation_report.md plus a rating
histogram --out/rating_hist.png; with --figures it also renders the two
example PNG figures.

Helper functions (count_shortest_paths, shortest_path_counts, build_report,
report_to_md, write_report_files, make_rating_hist, rating_hist_title) are
unit-tested in tests/test_check_maze_hard.py on synthetic grids.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cf.data.maze_hard import (CHECKS, load_maze_hard, parse_maze_hard,
                               validate_maze_hard, validate_maze_hard_dataset)

NBR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SHORT_PATH_CAP = 3


def label_len(row):
    return int(parse_maze_hard(row["question"], row["answer"])[3].sum())


def count_rating_matches(rows):
    return sum(1 for row in rows if float(row["rating"]) == label_len(row))


# ---- shortest-path counting (unit-tested in tests/test_check_maze_hard.py) ----

def _bfs(wall, src):
    """BFS over free (non-wall) 4-neighbour cells; returns (dist, visit order)."""
    H, W = wall.shape
    dist = np.full((H, W), -1, np.int32)
    dist[src] = 0
    order = [src]
    head = 0
    while head < len(order):
        r, c = order[head]
        head += 1
        for dr, dc in NBR4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and not wall[nr, nc]
                    and dist[nr, nc] < 0):
                dist[nr, nc] = dist[r, c] + 1
                order.append((nr, nc))
    return dist, order


def count_shortest_paths(wall, start, goal, cap=SHORT_PATH_CAP):
    """Number of wall-free 4-neighbour shortest S->G paths, capped at `cap`.

    Two BFS passes (from S and from G) give the shortest-path length L and
    mark the cells that lie on some shortest path (dist_s + dist_g == L);
    a forward DP in increasing dist_s then counts the distinct shortest
    paths, clamped per cell at `cap` so braided mazes cannot blow up the
    count.  Returns 0 if G is unreachable from S, 1 if S == G.
    """
    wall = np.asarray(wall, bool)
    H, W = wall.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W) or wall[start]:
        return 0
    if not (0 <= goal[0] < H and 0 <= goal[1] < W) or wall[goal]:
        return 0
    ds, order = _bfs(wall, start)
    L = int(ds[goal])
    if L < 0:
        return 0
    if L == 0:
        return 1
    dg, _ = _bfs(wall, goal)
    cnt = np.zeros((H, W), np.int64)
    cnt[start] = 1
    for cell in order:
        if cell == start:
            continue
        r, c = cell
        if int(ds[r, c]) + int(dg[r, c]) != L:
            continue                # not on any shortest S->G path
        total = 0
        for dr, dc in NBR4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W
                    and int(ds[nr, nc]) == int(ds[r, c]) - 1):
                total += int(cnt[nr, nc])
                if total >= cap:     # clamp: enough to know it reached the cap
                    break
        cnt[r, c] = min(total, cap)
    return int(cnt[goal])


def valid_rows(rows):
    """Rows that pass every validation check (cf.data.maze_hard)."""
    return [row for row in rows if not validate_maze_hard(row)]


def shortest_path_counts(rows, cap=SHORT_PATH_CAP):
    """Capped shortest-path count per maze, in row order.

    Only rows that passed validation are counted (like the rating and
    path-length statistics); malformed rows cannot be parsed anyway.  The
    summary built from the counts states n, which may be < len(rows).
    """
    return [count_shortest_paths(*parse_maze_hard(row["question"],
                                                  row["answer"])[:3], cap=cap)
            for row in valid_rows(rows)]


def shortest_path_summary(counts, cap=SHORT_PATH_CAP):
    """Distribution of the per-maze counts + how many mazes have >1 path."""
    dist = Counter(int(c) for c in counts)
    return dict(cap=cap, n=len(counts),
                counts=[int(c) for c in counts],
                distribution={str(k): dist[k] for k in sorted(dist)},
                n_multiple=sum(1 for c in counts if c > 1))


# ---- reports ----

def build_report(rep, rows_train, rows_test, n_t, n_te, sp_train, sp_test,
                 cap=SHORT_PATH_CAP):
    """JSON-serializable validation report (rendered to .md by report_to_md)."""
    return dict(
        grid="30x30",
        n_train=len(rows_train), n_test=len(rows_test),
        failures=rep["failures"],
        duplicates=rep["duplicates"],
        rating=rep["rating"],
        path_length=rep["path_length"],
        rating_equals_label_cells=dict(
            train=[n_t, len(rows_train)], test=[n_te, len(rows_test)]),
        shortest_paths=dict(
            method="BFS count of wall-free 4-neighbour shortest S->G paths",
            cap=cap,
            train=shortest_path_summary(sp_train, cap),
            test=shortest_path_summary(sp_test, cap)),
    )


def report_to_md(report):
    """Markdown rendering of the validation report."""
    lines = ["# Maze-hard 30x30 validation report", ""]
    lines.append("Validation table: number of failing rows per check "
                 "(%d train / %d test grids; 0 = valid)."
                 % (report["n_train"], report["n_test"]))
    lines += ["", "| check | train | test |", "|---|---:|---:|"]
    fails = report["failures"]
    for chk in CHECKS:
        lines.append("| %s | %d | %d |"
                     % (chk, fails["train"].get(chk, 0), fails["test"].get(chk, 0)))
    d = report["duplicates"]
    lines += ["", "Duplicates: within_train=%d within_test=%d across_splits=%d"
              % (d["within_train"], d["within_test"], d["across_splits"])]
    for name in ("train", "test"):
        r = report["rating"][name]
        p = report["path_length"][name]
        rl = report["rating_equals_label_cells"][name]
        sp = report["shortest_paths"][name]
        if not r or not p:                 # split without any valid rows
            lines += ["", "## %s (n=0)" % name, "", "- no valid rows"]
            continue
        dist = ", ".join("%d path(s): %d mazes"
                         % (int(k), sp["distribution"][k])
                         for k in sorted(sp["distribution"]))
        lines += ["", "## %s (n=%d)" % (name, r["n"]), "",
                  "- rating: min=%d mean=%.2f max=%d"
                  % (r["min"], r["mean"], r["max"]),
                  "- path length: min=%d mean=%.2f max=%d cells"
                  % (p["min"], p["mean"], p["max"]),
                  "- rating == number of label cells (S+G+o): %d/%d"
                  % (rl[0], rl[1]),
                  "- shortest S->G paths per maze (BFS count, capped at %d): %s"
                  % (sp["cap"], dist),
                  "- mazes with more than one shortest path: %d/%d"
                  % (sp["n_multiple"], sp["n"])]
    lines.append("")
    return "\n".join(lines)


def write_report_files(report, out_dir):
    """Write validation_report.json + validation_report.md; returns out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    (out_dir / "validation_report.md").write_text(report_to_md(report) + "\n",
                                                  encoding="utf-8")
    return out_dir


def print_shortest_path_summary(sp_train, sp_test):
    for name, counts in (("train", sp_train), ("test", sp_test)):
        dist = Counter(int(c) for c in counts)
        dist_s = ", ".join("%d path(s): %d" % (k, dist[k]) for k in sorted(dist))
        print("%-5s (n=%d): %s | more than one shortest path: %d/%d"
              % (name, len(counts), dist_s,
                 sum(1 for c in counts if c > 1), len(counts)))


# ---- figures ----

def rating_hist_title(n):
    """Rating-histogram title: states n, no week/milestone labels."""
    return "Maze-hard 30x30: rating distribution (n=%d mazes)" % n


def make_rating_hist(rows_train, rows_test, out_dir):
    """Rating histogram PNG (axes labelled, n stated in title and legend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in (("train", rows_train), ("test", rows_test)):
        ax.hist([float(r["rating"]) for r in rows], bins=30, alpha=0.6,
                label="%s (n=%d)" % (name, len(rows)))
    ax.set_xlabel("rating")
    ax.set_ylabel("number of mazes")
    ax.set_title(rating_hist_title(len(rows_train) + len(rows_test)))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rating_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    ap.add_argument("--figures", action="store_true",
                    help="also render the two example PNG figures")
    ap.add_argument("--out", default="figures/maze_hard",
                    help="report / figure output directory")
    args = ap.parse_args()
    rows_train = load_maze_hard("train")
    rows_test = load_maze_hard("test")
    rep = validate_maze_hard_dataset(rows_train, rows_test)
    n_t, n_te = print_report(rep, rows_train, rows_test)
    sp_train = shortest_path_counts(rows_train)
    sp_test = shortest_path_counts(rows_test)
    print("\nshortest S->G paths per maze (BFS count, capped at %d; "
          "validation-passing rows only, n per split):" % SHORT_PATH_CAP)
    print_shortest_path_summary(sp_train, sp_test)
    out_dir = write_report_files(
        build_report(rep, rows_train, rows_test, n_t, n_te, sp_train, sp_test),
        Path(args.out))
    make_rating_hist(rows_train, rows_test, out_dir)
    print("\nvalidation report + rating histogram written to %s" % out_dir)
    if args.figures:
        make_figures(rep, rows_train, rows_test, n_t, n_te, Path(args.out))


if __name__ == "__main__":
    main()
