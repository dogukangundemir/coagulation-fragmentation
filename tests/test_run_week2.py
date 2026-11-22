"""Helper checks for scripts/run_week2.py (no dataset loading)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_week2
from cf.models.cf import CFNet
from cf.models.rcnn import RecurrentCNN


def test_split_budget_spreads_the_remaining_minutes():
    assert run_week2.split_budget(45.0, 3) == pytest.approx(15.0)
    assert run_week2.split_budget(10.0, 4) == pytest.approx(2.5)
    assert run_week2.split_budget(0.5, 1) == pytest.approx(0.5)
    assert run_week2.split_budget(45.0, 0) == 0.0


def test_budget_allows_never_starts_below_one_minute():
    assert run_week2.budget_allows(1.0)
    assert not run_week2.budget_allows(0.99)
    assert not run_week2.budget_allows(0.0)


def test_run_plan_tags_in_execution_order():
    plan = run_week2.run_plan(["CF-Full", "RecurrentCNN"], [0, 2])
    assert [(m, s, t) for m, s, t in plan] == [
        ("CF-Full", 0, "CF-Full_seed0"), ("CF-Full", 2, "CF-Full_seed2"),
        ("RecurrentCNN", 0, "RecurrentCNN_seed0"), ("RecurrentCNN", 2, "RecurrentCNN_seed2")]


def test_run_done_only_true_for_finished_results():
    payload = {"runs": {"A_seed0": {"status": "done"},
                        "B_seed1": {"status": "skipped: budget"}}}
    assert run_week2.run_done(payload, "A_seed0")
    assert not run_week2.run_done(payload, "B_seed1")
    assert not run_week2.run_done(payload, "C_seed2")
    assert not run_week2.run_done({}, "A_seed0")


def test_make_model_builds_the_requested_variants():
    assert isinstance(run_week2.make_model("RecurrentCNN"), RecurrentCNN)
    net = run_week2.make_model("CF-NoFrag")
    assert isinstance(net, CFNet) and not net.fragmentation


def test_make_model_rejects_unknown_names():
    with pytest.raises(ValueError):
        run_week2.make_model("PlainCNN")


def test_atomic_write_json_replaces_atomically(tmp_path):
    path = tmp_path / "week2_metrics.json"
    run_week2.atomic_write_json(path, {"runs": {"a": 1}})
    assert not list(tmp_path.glob("*.tmp"))        # no leftover temp file
    run_week2.atomic_write_json(path, {"runs": {"a": 2}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"runs": {"a": 2}}


def test_metrics_path_under_out_dir():
    assert run_week2.metrics_path("week2_results").name == "week2_metrics.json"
    assert run_week2.metrics_path("week2_results").parent == Path("week2_results")


def test_parse_args_defaults_and_fast_overrides():
    args = run_week2.parse_args([])
    assert args.models == ["RecurrentCNN", "CF-Full", "CF-NoFrag"]
    assert args.seeds == [0, 1, 2] and args.budget_minutes == 45
    cfg = run_week2.config_of(args)
    assert cfg["seeds"] == [0, 1, 2] and cfg["sweep_T"][0] == 1 and not cfg["fast"]

    fast = run_week2.config_of(run_week2.parse_args(["--fast"]))
    assert fast["seeds"] == [0] and fast["sweep_T"] == [1, 4, 8]
    assert (fast["n_train"], fast["n_val"], fast["n_test"], fast["n_probe"]) == \
        (200, 100, 100, 100)


def test_new_payload_metadata_without_dataset():
    class _D:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        y = None

    data = {k: _D(n) for k, n in
            dict(train=200, val=100, test=100, probe7=100, probe9=100).items()}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_week2, "train_pos_weight", lambda d: 3.0)
        mp.setattr(run_week2, "git_commit", lambda: "abc1234")
        mp.setattr(run_week2, "git_dirty", lambda out=None: True)
        payload = run_week2.new_payload(dict(fast=True), data)
    assert payload["dataset_sizes"]["train"] == 200
    assert payload["dataset_sizes"]["pos_weight"] == 3.0
    assert payload["git_commit"] == "abc1234"
    assert payload["git_dirty"] is True
    assert payload["runs"] == {} and payload["device"] in ("cpu", "cuda", "mps")


class _Result:
    def __init__(self, stdout=""):
        self.stdout = stdout


def test_config_mismatch_reports_differing_keys():
    cfg = dict(models=["CF-Full"], seeds=[0], T_train=8, sweep_T=[1, 4, 8], fast=True)
    assert run_week2.config_mismatch(cfg, cfg) == []
    assert run_week2.config_mismatch(dict(cfg, seeds=[1]), cfg) == ["seeds"]
    assert run_week2.config_mismatch(dict(cfg, fast=False), cfg) == ["fast"]
    assert run_week2.config_mismatch(None, cfg) == run_week2.RESUME_CFG_KEYS


def test_git_dirty_ignores_untracked_files_under_out_dir(monkeypatch, tmp_path):
    root = tmp_path / "repo"

    def fake_run(cmd, capture_output=True, text=True, check=True):
        assert cmd[:2] == ["git", "status"] or cmd[:2] == ["git", "rev-parse"]
        if cmd[1] == "status":
            return _Result("?? out/result.json\n?? out/sub/deep.pt\n")
        return _Result(str(root))

    monkeypatch.setattr(run_week2.subprocess, "run", fake_run)
    assert run_week2.git_dirty(root / "out") is False        # only out-untracked
    assert run_week2.git_dirty(root / "out") is False        # same, idempotent


def test_git_dirty_flags_other_changes(monkeypatch, tmp_path):
    root = tmp_path / "repo"

    def fake_run(cmd, capture_output=True, text=True, check=True):
        if cmd[1] == "status":
            return _Result("?? elsewhere.txt\n?? out/result.json\n")
        return _Result(str(root))

    monkeypatch.setattr(run_week2.subprocess, "run", fake_run)
    assert run_week2.git_dirty(root / "out") is True        # untracked outside out

    def fake_tracked(cmd, capture_output=True, text=True, check=True):
        if cmd[1] == "status":
            return _Result(" M cf/train.py\n")
        return _Result(str(root))

    monkeypatch.setattr(run_week2.subprocess, "run", fake_tracked)
    assert run_week2.git_dirty(root / "out") is True        # tracked modification


def test_git_dirty_false_on_clean_tree_and_none_without_git(monkeypatch, tmp_path):
    root = tmp_path / "repo"

    def fake_run(cmd, capture_output=True, text=True, check=True):
        if cmd[1] == "status":
            return _Result("")
        return _Result(str(root))

    monkeypatch.setattr(run_week2.subprocess, "run", fake_run)
    assert run_week2.git_dirty(root / "out") is False

    def failing_run(cmd, capture_output=True, text=True, check=True):
        raise run_week2.subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(run_week2.subprocess, "run", failing_run)
    assert run_week2.git_dirty("out") is None


def test_print_table_uses_four_decimals(capsys):
    payload = dict(config=dict(models=["CF-Full"], seeds=[0]),
                   runs={"CF-Full_seed0": dict(
                       model="CF-Full", status="done", n_params=10,
                       epochs_done=2, stopped_early=False,
                       test=dict(strict=0.12344, iou=0.56789))})
    run_week2.print_table(payload)
    out = capsys.readouterr().out
    # values chosen so 3-decimal printing would produce a different string
    assert "0.1234" in out and "0.5679" in out
