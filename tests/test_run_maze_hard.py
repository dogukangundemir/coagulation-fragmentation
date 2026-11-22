"""Offline unit tests for scripts/run_maze_hard.py (synthetic grids).

The WP5 pipeline is exercised on tiny synthetic Maze-hard-format grids
(30x30, 900-char question/answer strings, no dataset download): the CLI
parses the documented defaults, make_model builds the WP5 configuration
(CF-WallAware, cross_kernel=True), a 1-batch 1-epoch CPU smoke run trains
and evaluates 4 grids end-to-end and returns every cf.metrics summary
field, the resumable seed-specific checkpoint restores / refuses correctly,
and the results.json + summary.md outputs round-trip with the budget
reporting rules (raw/effective params, epochs done/planned, GPU minutes).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cf.train as train_mod                       # noqa: E402
import run_maze_hard as rmh                         # noqa: E402
from cf.data.maze_hard import MazeHardDataset       # noqa: E402

SIZE = 30
METRIC_KEYS = {"strict", "exact", "iou", "precision", "recall",
               "covers_path", "wall_hits", "extra_cells", "missed_cells"}
BASE_HYPERPARAMS = dict(train_mod.HYPERPARAMS)   # pristine, captured at import


@pytest.fixture(autouse=True)
def restore_hyperparams():
    """run_seed sets the planned epochs globally; restore the shared value."""
    yield
    train_mod.HYPERPARAMS.clear()
    train_mod.HYPERPARAMS.update(BASE_HYPERPARAMS)


# ------------------------------------------------ Maze-hard-format grids

def make_row(free, start, goal, path_cells):
    """(free cells, start, goal, reference path) -> maze-hard csv row."""
    q = np.full((SIZE, SIZE), "#", dtype="<U1")
    a = np.full((SIZE, SIZE), "#", dtype="<U1")
    for r, c in free:
        q[r, c] = " "
        a[r, c] = " "
    q[start] = "S"
    q[goal] = "G"
    a[start] = "S"
    a[goal] = "G"
    for r, c in path_cells:
        a[r, c] = "o"
    return dict(source="synthetic", question="".join(q.ravel()),
                answer="".join(a.ravel()), rating=1.0)


def corridor_row(r0):
    """Untied corridor: exactly one shortest route."""
    free = [(r0, c) for c in range(2, 12)]
    path = [(r0, c) for c in range(3, 11)]
    return make_row(free, (r0, 2), (r0, 11), path)


def braid_row(r0):
    """Tied 3x8 room: several equally short routes (maze-hard's own trap)."""
    free = [(r, c) for r in range(r0, r0 + 3) for c in range(2, 10)]
    path = [(r0 + 1, c) for c in range(2, 10)]
    return make_row(free, (r0 + 2, 2), (r0, 9), path)


@pytest.fixture
def tiny_data():
    """4 tiny grids shared by the splits (train/val/test of 4 each)."""
    rows = [corridor_row(2), braid_row(6), corridor_row(12), braid_row(18)]
    return {k: MazeHardDataset(rows) for k in ("train", "val", "test")}


def tiny_cfg(**over):
    cfg = dict(model="CF-WallAware", epochs=1, T_train=1, hid=4,
               T_candidates=[1, 2], max_minutes_per_seed=None,
               seeds=[0], out="unused", n_train=4, n_val=4, split_seed=0)
    cfg.update(over)
    return cfg


# ------------------------------------------------------------------ parsing

def test_parse_args_defaults():
    args = rmh.parse_args([])
    assert args.model == "CF-WallAware"
    assert args.seeds == [0, 1, 2]
    assert args.epochs == 40
    assert args.T_train == 24
    assert args.hid == 16
    assert args.out == rmh.OUT_DEFAULT
    assert args.max_minutes_per_seed is None
    assert args.T_candidates == [16, 24, 32, 48]


def test_parse_args_overrides():
    args = rmh.parse_args(["--model", "CF-Full", "--seeds", "3", "4",
                           "--epochs", "2", "--T-train", "16", "--hid", "8",
                           "--out", "results/tmp_out",
                           "--max-minutes-per-seed", "1.5",
                           "--T-candidates", "8", "16"])
    assert args.model == "CF-Full" and args.seeds == [3, 4]
    assert args.epochs == 2 and args.T_train == 16 and args.hid == 8
    assert args.out == "results/tmp_out"
    assert args.max_minutes_per_seed == 1.5
    assert args.T_candidates == [8, 16]


def test_parse_args_rejects_unknown_model():
    with pytest.raises(SystemExit):
        rmh.parse_args(["--model", "NotACF"])


def test_config_of_maps_args():
    args = rmh.parse_args(["--seeds", "7"])
    cfg = rmh.config_of(args)
    assert cfg["model"] == "CF-WallAware" and cfg["seeds"] == [7]
    assert cfg["n_train"] == rmh.N_TRAIN and cfg["n_val"] == rmh.N_VAL
    assert cfg["split_seed"] == rmh.SPLIT_SEED


# -------------------------------------------------------------- model setup

def test_make_model_is_wp5_configuration():
    m = rmh.make_model("CF-WallAware", hid=8)
    assert m.wall_aware and m.cross_kernel and m.fragmentation and m.bound_flux
    assert not m.cross_kernel_encoder     # only the rate net is cross-masked
    assert m.encoder.in_channels == 3
    m2 = rmh.make_model("CF-NoFrag", hid=8)
    assert m2.cross_kernel and not m2.fragmentation
    with pytest.raises(ValueError):
        rmh.make_model("NopeCF", hid=8)


def test_model_kw_records_cross_kernel():
    kw = rmh.model_kw("CF-WallAware", 21)
    assert kw == dict(hid=21, cross_kernel=True, wall_aware=True)


# ------------------------------------------- 1-batch CPU smoke run (4 grids)

@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """One run_seed call: 4 grids, 1 batch, 1 epoch, tiny horizon, CPU."""
    out_dir = tmp_path_factory.mktemp("wp5smoke")
    rows = [corridor_row(2), braid_row(6), corridor_row(12), braid_row(18)]
    data = {k: MazeHardDataset(rows) for k in ("train", "val", "test")}
    cfg = tiny_cfg()
    rec = rmh.run_seed("CF-WallAware", 0, cfg, data, out_dir, verbose=False)
    return rec, out_dir


def test_smoke_run_trains_one_batch_and_reports_budget(smoke_run):
    rec, out_dir = smoke_run
    assert rec["status"] == "done"
    assert rec["epochs_done"] == 1 and rec["epochs_planned"] == 1
    assert rec["stopped_early"] is False
    assert rec["params_raw"] >= rec["params_effective"] > 0
    assert rec["train_seconds"] >= 0.0
    # the 4-grid loader is a single batch (batch size 32 > 4 grids)
    assert train_mod.HYPERPARAMS["batch_size"] >= 4
    assert (out_dir / "checkpoints" / "CF-WallAware_seed0.ckpt.pt").exists()


def test_smoke_run_metric_fields_present(smoke_run):
    rec, _ = smoke_run
    fields = METRIC_KEYS
    for summary in (rec["T_train_metrics"], rec["T_star_metrics"]):
        assert set(summary) >= fields
    for sweeps in (rec["val_sweeps"], rec["test_sweeps"]):
        for summary in sweeps.values():
            assert set(summary) >= fields
    assert rec["T_star"] in (1, 2)
    assert set(rec["val_strict_by_T"]) == {"1", "2"}
    # wall-aware readout masks walls at -20: never a predicted wall cell
    assert rec["T_star_metrics"]["wall_hits"] == 0.0
    assert 0.0 <= rec["T_star_metrics"]["strict"] <= 1.0


def test_smoke_run_metrics_reproducible_from_checkpoint(smoke_run):
    """Reload the checkpoint's best state and re-run the sweep: identical."""
    rec, out_dir = smoke_run
    rows = [corridor_row(2), braid_row(6), corridor_row(12), braid_row(18)]
    d = MazeHardDataset(rows)
    tag = "CF-WallAware_seed0"
    ckpt = out_dir / "checkpoints" / ("%s.ckpt.pt" % tag)
    saved = torch.load(str(ckpt), map_location="cpu")
    assert saved["run_id"] == tag and saved["epoch"] == 0
    model = rmh.make_model("CF-WallAware", 4)
    model.load_state_dict(saved["best_state"])
    sweep = train_mod.sweep_model(model, d, [1, 2], assert_strict_equals_exact=False)
    for T in (1, 2):
        for k, v in sweep[T].items():
            assert abs(rec["test_sweeps"][str(T)][k] - v) < 1e-9


def test_test_sweep_includes_T_train_and_candidates(smoke_run):
    rec, _ = smoke_run
    assert set(rec["test_sweeps"]) >= {"1", "2"}
    assert rec["T_train_metrics"] == rec["test_sweeps"][str(rec["T_train"])]


# ------------------------------------------------- resumable checkpoints

def test_checkpoint_resumes_and_guards_fingerprint(tiny_data, tmp_path):
    cfg = tiny_cfg()
    rec1 = rmh.run_seed("CF-WallAware", 0, cfg, tiny_data, tmp_path, verbose=False)
    # second call resumes from the finished checkpoint: no duplicate training
    rec2 = rmh.run_seed("CF-WallAware", 0, cfg, tiny_data, tmp_path, verbose=False)
    assert rec2["epochs_done"] == 1
    assert rec1["T_star_metrics"] == rec2["T_star_metrics"]
    # a different T_train (or hid) on the same seed checkpoint is refused:
    # the run fingerprint no longer matches the checkpoint
    with pytest.raises(ValueError, match="fingerprint"):
        rmh.run_seed("CF-WallAware", 0, tiny_cfg(T_train=2),
                     tiny_data, tmp_path, verbose=False)
    with pytest.raises(ValueError, match="fingerprint"):
        rmh.run_seed("CF-WallAware", 0, tiny_cfg(hid=8),
                     tiny_data, tmp_path, verbose=False)


# ------------------------------------------------------------- outputs

def test_outputs_round_trip_and_budget_reporting(smoke_run, tmp_path):
    rec, _ = smoke_run
    cfg = tiny_cfg(out=str(tmp_path))
    data_sizes = dict(train=4, val=4, test=4)
    runs = {"CF-WallAware_seed0": rec}
    payload = rmh.build_payload(cfg, data_sizes, runs,
                                rmh.aggregate_per_model(runs, ["CF-WallAware"]),
                                123.4)
    rj, sm = rmh.write_outputs(tmp_path, payload)
    assert rj.exists() and sm.exists()
    with open(rj, encoding="utf-8") as fh:
        loaded = json.load(fh)                     # every field is JSON-safe
    assert loaded["model"]["variants"]["CF-WallAware"]["cross_kernel"] is True
    assert loaded["training"]["epochs_planned"] == 1
    assert loaded["budget"]["wall_minutes"] > 0.0
    assert loaded["budget"]["gpu_minutes"] == 0.0   # CPU-only run
    assert loaded["budget"]["preliminary"] is True   # 1 seed < 3
    per = loaded["per_model"]["CF-WallAware"]
    assert per["n_seeds"] == 1 and per["preliminary"] is True
    assert set(per) >= {"exact_at_T_train", "strict_at_T_star", "iou_at_T_star",
                       "recall_at_T_star", "wall_hits_at_T_star"}
    text = sm.read_text(encoding="utf-8")
    for needle in ("board match", "valid path", "Preliminary", "GPU minutes",
                   "T* =", "epochs done", "params raw/effective"):
        assert needle in text, needle
