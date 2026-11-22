"""T* selection, JSON writer and table formatting checks (no dataset loaded)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import select_T_val as sel


def test_pick_t_star_takes_the_maximum():
    T = sel.pick_T_star({4: 0.30, 8: 0.55, 12: 0.51, 24: 0.10})
    assert T == 8


def test_pick_t_star_breaks_ties_toward_the_smaller_t():
    assert sel.pick_T_star({4: 0.5, 8: 0.5, 16: 0.4}) == 4
    assert sel.pick_T_star({8: 0.5, 12: 0.5, 16: 0.5}) == 8
    # the maximum coming later must still win over an earlier equal plateau
    assert sel.pick_T_star({4: 0.4, 8: 0.4, 16: 0.7}) == 16


def test_pick_t_star_single_candidate_and_float_strings():
    assert sel.pick_T_star({16: 0.42}) == 16


def test_stored_sweeps_rebuilds_int_horizon_keys():
    run = {"sweeps": {"test": {"4": {"strict": 0.1}, "8": {"strict": 0.2}},
                      "probe7": {"8": {"strict": 0.05}},
                      "bogus": "not-a-dict"}}
    stored = sel.stored_sweeps(run)
    assert stored["test"] == {4: {"strict": 0.1}, 8: {"strict": 0.2}}
    assert sel.stored_sweeps({}) == {}
    assert sel.stored_sweeps({"sweeps": None}) == {}


def test_metrics_at_t_reuses_stored_sweeps_and_recorded_test():
    run = {"sweeps": {"test": {"12": {"strict": 0.66}}},
           "test": {"strict": 0.80}}
    assert sel.metrics_at_T(run, "test", 12, 8)["strict"] == 0.66   # stored sweep
    assert sel.metrics_at_T(run, "test", 8, 8)["strict"] == 0.80     # recorded test
    with pytest.raises(KeyError):                                    # nothing to reuse
        sel.metrics_at_T(run, "probe9", 16, 8)


def test_metrics_at_t_recomputes_only_when_model_and_data_are_given():
    run = {"sweeps": {}}
    calls = []

    def fake_sweep(model, d, Ts):
        calls.append(Ts)
        return {Ts[0]: {"strict": 0.72}}

    class _M:
        pass

    sel.train_mod.sweep_model = fake_sweep
    try:
        out = sel.metrics_at_T(run, "probe7", 24, 8, model=_M(), data={"probe7": None})
        assert out["strict"] == 0.72 and calls == [[24]]
    finally:
        sel.train_mod.sweep_model = __import__("cf.train", fromlist=["sweep_model"]).sweep_model


def test_done_runs_groups_by_model_then_seed():
    runs = {"CF-NoFrag_seed1": {"model": "CF-NoFrag", "seed": 1, "status": "done"},
            "RecurrentCNN_seed0": {"model": "RecurrentCNN", "seed": 0, "status": "done"},
            "CF-NoFrag_seed0": {"model": "CF-NoFrag", "seed": 0, "status": "done"},
            "CF-Full_seed0": {"model": "CF-Full", "seed": 0, "status": "skipped: budget"}}
    got = [(m, s) for m, s, _ in sel.done_runs(runs)]
    assert got == [("CF-NoFrag", 0), ("CF-NoFrag", 1), ("RecurrentCNN", 0)]


def test_aggregate_per_model_mean_and_sample_std():
    selections = [dict(model="A", test_strict_at_T_star=0.60),
                  dict(model="A", test_strict_at_T_star=0.80),
                  dict(model="B", test_strict_at_T_star=0.50),
                  dict(model="C", test_strict_at_T_star=0.10)]
    agg = sel.aggregate_per_model(selections, ["A", "B", "C"])
    assert agg["A"]["n_seeds"] == 2
    assert agg["A"]["mean_test_strict_at_T_star"] == pytest.approx(0.70)
    assert agg["A"]["std_test_strict_at_T_star"] == pytest.approx(0.2 / (2 ** 0.5))
    assert agg["B"]["std_test_strict_at_T_star"] == 0.0        # single seed -> 0
    assert agg["A"]["mean_test_strict_at_T_star"] > agg["C"]["mean_test_strict_at_T_star"]
    empty = sel.aggregate_per_model([], ["D"])
    assert empty["D"]["n_seeds"] == 0 and empty["D"]["mean_test_strict_at_T_star"] is None


def test_format_table_rows_and_per_model_summary():
    selections = [
        dict(model="CF-Full", seed=0, T_star=12, val_strict_at_T_star=0.5000,
             test_strict_at_T_train=0.6111, test_strict_at_T_star=0.7000),
        dict(model="CF-Full", seed=1, T_star=16, val_strict_at_T_star=0.4800,
             test_strict_at_T_train=0.5222, test_strict_at_T_star=0.6000)]
    per_model = sel.aggregate_per_model(selections, ["CF-Full", "Empty"])
    table = sel.format_table(selections, per_model, ["CF-Full", "Empty"], T_train=8)
    lines = table.splitlines()
    header, rule, row0, row1 = lines[0], lines[1], lines[2], lines[3]
    assert header.split() == ["model", "seed", "T*", "val", "strict@T*",
                             "test", "strict@8", "test", "strict@T*"]
    assert set(rule) == {"-"}
    assert "CF-Full" in row0 and "CF-Full" in row1
    assert row0.split()[1] == "0" and row0.split()[2] == "12"
    assert "0.7000" in row0 and "0.6111" in row0 and "0.5000" in row0
    assert "0.6500 +- 0.0707" in table          # mean +- sample std over seeds
    assert "over 2 seed(s)" in table
    assert "Empty" in table and "no finished runs" in table


def test_write_selection_is_atomic_and_replaces(tmp_path):
    path = tmp_path / "valT_selection.json"
    out = sel.write_selection(path, {"T_train": 8, "selections": []})
    assert out == path
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8")) == {"T_train": 8, "selections": []}
    sel.write_selection(path, {"T_train": 12})
    assert json.loads(path.read_text(encoding="utf-8")) == {"T_train": 12}


def test_parse_args_defaults_and_overrides():
    args = sel.parse_args(["--results", "r"])
    assert args.T == [4, 8, 12, 16, 24] and args.models is None and args.results == "r"
    args = sel.parse_args(["--results", "r", "--T", "8", "16", "--models", "CF-Full"])
    assert args.T == [8, 16] and args.models == ["CF-Full"]
