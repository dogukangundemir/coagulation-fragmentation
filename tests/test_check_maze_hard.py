"""Offline unit tests for scripts/check_maze_hard.py helpers (synthetic grids)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_maze_hard as cmh
from cf.data.maze_hard import CHECKS, validate_maze_hard_dataset

SIZE = 30


# ---- count_shortest_paths ----

def test_single_corridor_has_exactly_one_shortest_path():
    wall = np.ones((7, 7), bool)
    wall[1, 1:6] = False                   # corridor S=(1,1) -> G=(1,5)
    assert cmh.count_shortest_paths(wall, (1, 1), (1, 5)) == 1


def test_two_by_two_room_has_two_diagonal_shortest_paths():
    wall = np.ones((5, 5), bool)
    wall[1:3, 1:3] = False                # open 2x2 square
    # S=(1,1), G=(2,2) touch only at a corner: via (1,2) or via (2,1)
    assert cmh.count_shortest_paths(wall, (1, 1), (2, 2)) == 2


def test_open_room_count_is_capped():
    wall = np.ones((7, 7), bool)
    wall[1:6, 1:6] = False                # open 5x5 room, S/G opposite corners
    assert cmh.count_shortest_paths(wall, (1, 1), (5, 5)) == 3   # default cap
    # exact count is C(8,4) = 70: a cap above it does not distort it
    assert cmh.count_shortest_paths(wall, (1, 1), (5, 5), cap=100) == 70
    assert cmh.count_shortest_paths(wall, (1, 1), (5, 5), cap=10) == 10


def test_unreachable_goal_counts_zero():
    wall = np.ones((5, 5), bool)
    wall[1, 1:4] = False
    wall[3, 1:4] = False                  # two corridors, walled apart
    assert cmh.count_shortest_paths(wall, (1, 1), (3, 3)) == 0


def test_start_equals_goal_counts_one():
    wall = np.ones((5, 5), bool)
    wall[1:4, 1:4] = False
    assert cmh.count_shortest_paths(wall, (1, 1), (1, 1)) == 1


def test_walled_or_out_of_bounds_endpoints_do_not_crash():
    wall = np.ones((5, 5), bool)
    assert cmh.count_shortest_paths(wall, (0, 0), (3, 3)) == 0
    assert cmh.count_shortest_paths(wall, (3, 3), (-1, 3)) == 0


# ---- report building / writing (on synthetic 30x30 rows) ----

def _to_str(grid):
    return "".join(np.asarray(grid).ravel())


def _corridor_row(rating=28.0):
    """Valid synthetic row: one S->G corridor of 28 label cells along row 1."""
    wall = np.ones((SIZE, SIZE), bool)
    wall[1, 1:29] = False
    q = np.where(wall, "#", " ").astype("<U1")
    a = q.copy()
    q[1, 1], a[1, 1] = "S", "S"
    q[1, 28], a[1, 28] = "G", "G"
    a[1, 2:28] = "o"
    return dict(source="synthetic", question=_to_str(q), answer=_to_str(a),
                rating=rating)


def _two_path_row():
    """Valid row whose grid also admits a second, equally short S->G path."""
    wall = np.ones((SIZE, SIZE), bool)
    wall[1:4, 1:4] = False                # open 3x3 room
    q = np.where(wall, "#", " ").astype("<U1")
    a = q.copy()
    q[1, 1], a[1, 1] = "S", "S"
    q[2, 2], a[2, 2] = "G", "G"
    a[1, 2] = "o"                         # label path via (1,2); the route
    return dict(source="synthetic",       # via (2,1) is equally short
                question=_to_str(q), answer=_to_str(a), rating=3.0)


def test_shortest_path_counts_follow_row_order():
    rows = [_corridor_row(), _two_path_row()]
    assert cmh.shortest_path_counts(rows) == [1, 2]


def test_shortest_path_counts_skip_rows_that_fail_validation():
    """Shortest-path stats cover only validation-passing rows (like the
    rating / path-length stats); malformed rows would not even parse,
    and the summary states n = number of counted rows."""
    bad = dict(source="synthetic", question="xx", answer="yy", rating=1.0)
    rows = [_corridor_row(), bad, _two_path_row()]
    assert cmh.valid_rows(rows) == [_corridor_row(), _two_path_row()]
    counts = cmh.shortest_path_counts(rows)
    assert counts == [1, 2]
    assert cmh.shortest_path_summary(counts)["n"] == 2


def test_build_report_and_write_files(tmp_path):
    rows_train = [_corridor_row(), _two_path_row()]
    rows_test = [_corridor_row(rating=1.0)]
    rep = validate_maze_hard_dataset(rows_train, rows_test)
    report = cmh.build_report(rep, rows_train, rows_test,
                              cmh.count_rating_matches(rows_train),
                              cmh.count_rating_matches(rows_test),
                              cmh.shortest_path_counts(rows_train),
                              cmh.shortest_path_counts(rows_test))
    assert report["failures"]["train"] == {} and report["failures"]["test"] == {}
    assert report["shortest_paths"]["cap"] == 3
    assert report["shortest_paths"]["train"]["counts"] == [1, 2]
    assert report["shortest_paths"]["train"]["n_multiple"] == 1
    assert report["shortest_paths"]["test"]["n_multiple"] == 0

    out = cmh.write_report_files(report, tmp_path)
    loaded = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    assert loaded == report
    md = (out / "validation_report.md").read_text(encoding="utf-8")
    assert "| check | train | test |" in md
    assert "mazes with more than one shortest path: 1/2" in md
    assert "mazes with more than one shortest path: 0/1" in md


def test_report_md_lists_every_check(tmp_path):
    rows = [_corridor_row()]
    rep = validate_maze_hard_dataset(rows, [])
    report = cmh.build_report(rep, rows, [], cmh.count_rating_matches(rows),
                              0, cmh.shortest_path_counts(rows), [])
    md = cmh.report_to_md(report)
    assert all(("| %s | 0 |" % chk) in md for chk in CHECKS)


def test_rating_hist_title_states_n_without_week_or_milestone():
    title = cmh.rating_hist_title(1000)
    assert "n=1000" in title
    low = title.lower()
    assert "week" not in low and "milestone" not in low


def test_make_rating_hist_writes_png(tmp_path):
    cmh.make_rating_hist([_corridor_row(), _corridor_row(rating=29.0)],
                         [_corridor_row(rating=1.0)], tmp_path)
    png = tmp_path / "rating_hist.png"
    assert png.exists() and png.stat().st_size > 1000
