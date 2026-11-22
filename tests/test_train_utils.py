"""Training-utility checks: wall-clock budget, checkpoint resume, T-sweep, settling."""
import math

import pytest
import torch

import cf.train as train_mod
from cf.data.curriculum import MazeDataset
from cf.models.cf import CFNet
from cf.models.rcnn import RecurrentCNN
from cf.train import DEVICE, HYPERPARAMS, evaluate_model, settling_curve, sweep_model, train_model

from maze_synthetic import make_batch

GRID = 9       # lattice n=4 -> ASCII 9x9


def _tiny(seed, count, n=4):
    pairs = make_batch(seed=seed, count=count, n=n)
    mazes = [m for m, _ in pairs]
    sols = [s for _, s in pairs]
    return MazeDataset(range(count), mazes, sols)


def _make():
    return RecurrentCNN(hid=8)


class _FakeClock:
    """Deterministic injectable clock: every call advances by `step` seconds."""

    def __init__(self, step=10.0):
        self.step = step
        self.t = 0.0

    def __call__(self):
        self.t += self.step
        return self.t


def _train_args(tmp_path, train_ds, val_ds, tag):
    return dict(T=4, seed=0, tag=tag, train_ds=train_ds, val_ds=val_ds,
                results_dir=tmp_path, verbose=False, assert_strict_equals_exact=False)


def test_default_train_model_keys_unchanged(tmp_path, monkeypatch):
    monkeypatch.setitem(train_mod.HYPERPARAMS, "epochs", 2)
    model, hist = train_model(_make, **_train_args(tmp_path, _tiny(0, 6), _tiny(1, 4), "k"))
    assert set(hist) == {"train_loss", "val_strict", "val_iou", "val_covers_path",
                         "best_epoch", "train_time_s"}
    assert len(hist["train_loss"]) == 2
    for key in ("val_strict", "val_iou", "val_covers_path"):
        assert len(hist[key]) == 2
    assert isinstance(model, RecurrentCNN)


def test_max_minutes_stops_early_with_usable_model(tmp_path, monkeypatch):
    monkeypatch.setitem(train_mod.HYPERPARAMS, "epochs", 5)
    train_ds, val_ds = _tiny(0, 6), _tiny(1, 4)
    # fake clock: +10 s per call, 3 s budget -> the first epoch is always
    # allowed, the second epoch's check trips the budget deterministically
    model, hist = train_model(_make, max_minutes=0.05, _clock=_FakeClock(),
                             **_train_args(tmp_path, train_ds, val_ds, "b"))
    assert hist["stopped_early"] is True
    assert hist["epochs_done"] >= 1                      # at least the first epoch ran
    assert len(hist["train_loss"]) == hist["epochs_done"]
    assert hist["seconds"] > 0.0
    model.eval()
    with torch.no_grad():
        out = model(val_ds.x[:2].to(DEVICE), T=4)
    assert out.shape == (2, GRID, GRID)                   # model still usable


def test_budget_does_not_compress_cosine_schedule(tmp_path, monkeypatch):
    planned = 4
    monkeypatch.setitem(train_mod.HYPERPARAMS, "epochs", planned)
    train_ds, val_ds = _tiny(0, 6), _tiny(1, 4)
    ck_path = tmp_path / "budget.pt"
    train_model(_make, max_minutes=0.05, checkpoint_path=str(ck_path),
                _clock=_FakeClock(),
                **_train_args(tmp_path, train_ds, val_ds, "s"))
    saved = torch.load(str(ck_path), map_location="cpu")
    k = saved["sched_state"]["last_epoch"]                # steps actually taken
    steps_per_epoch = math.ceil(len(train_ds) / HYPERPARAMS["batch_size"])
    total = planned * steps_per_epoch                     # FULL planned horizon
    assert 1 <= k < total
    expected = HYPERPARAMS["lr"] * 0.5 * (1.0 + math.cos(math.pi * k / total))
    assert expected > 0.0                                 # a compressed schedule would be at 0 here
    assert saved["opt_state"]["param_groups"][0]["lr"] == pytest.approx(expected, rel=1e-9)


def test_checkpoint_resume_continues_epochs(tmp_path, monkeypatch):
    monkeypatch.setitem(train_mod.HYPERPARAMS, "epochs", 2)
    train_ds, val_ds = _tiny(0, 6), _tiny(1, 4)
    ck_path = tmp_path / "resume.pt"
    args = _train_args(tmp_path, train_ds, val_ds, "r")
    _, hist1 = train_model(_make, max_minutes=0.05, checkpoint_path=str(ck_path),
                           _clock=_FakeClock(), **args)
    assert hist1["epochs_done"] == 1                     # stopped after the first epoch
    _, hist2 = train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path), **args)
    assert hist2["epochs_done"] == 2                      # resumed run completed the plan
    for key in ("train_loss", "val_strict", "val_iou", "val_covers_path"):
        assert len(hist2[key]) == 2
        assert len(hist1[key]) == 1
    assert hist2["train_loss"][0] == hist1["train_loss"][0]   # history continued, not restarted
    assert hist2["best_epoch"] in (0, 1)


def test_mid_epoch_budget_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    """The regression test for the mid-epoch checkpoint bug: a budget stop
    mid-epoch must leave the last completed-epoch checkpoint untouched, so a
    resume is bit-identical to an uninterrupted run."""
    monkeypatch.setitem(train_mod.HYPERPARAMS, "epochs", 2)
    train_ds, val_ds = _tiny(0, 80), _tiny(1, 8)          # 3 steps per epoch
    args = _train_args(tmp_path, train_ds, val_ds, "m")
    ck_path = tmp_path / "midep.ckpt.pt"

    # uninterrupted reference: full 2-epoch run, no budget
    ref_model, ref_hist = train_model(_make, **args)
    assert not ref_hist.get("stopped_early", False)

    # interrupted run: fake clock +10 s per call, 45 s budget -> epoch 0
    # completes (elapsed 10/20 s), epoch 1 stops mid-epoch (elapsed > 45 s)
    _, hist_a = train_model(_make, max_minutes=0.75, checkpoint_path=str(ck_path),
                            _clock=_FakeClock(), **args)
    assert hist_a["stopped_early"] is True
    assert hist_a["epochs_done"] == 1
    ck = torch.load(str(ck_path), map_location="cpu")
    assert ck["epoch"] == 0                          # last COMPLETED epoch
    assert ck["gen_state"] is not None               # shuffle RNG saved

    # resume from disk: must re-run exactly the interrupted epoch 1
    model_b, hist_b = train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path),
                                  _clock=_FakeClock(), **args)
    assert not hist_b["stopped_early"]
    assert hist_b["epochs_done"] == 2
    for key in ("train_loss", "val_strict", "val_iou", "val_covers_path"):
        assert [round(v, 12) for v in hist_b[key]] == [round(v, 12) for v in ref_hist[key]]
    dw = max((ref_model.state_dict()[k] - model_b.state_dict()[k]).abs().max().item()
             for k in ref_model.state_dict())
    assert dw < 1e-6


@pytest.mark.parametrize("make_model", [lambda: CFNet(hid=4), lambda: RecurrentCNN(hid=8)])
def test_sweep_model_matches_evaluate_model(make_model):
    torch.manual_seed(0)
    model = make_model().to(DEVICE).eval()
    d = _tiny(2, 4)
    Ts = [2, 5, 8]
    sw = sweep_model(model, d, Ts, assert_strict_equals_exact=False)
    assert sorted(sw.keys()) == sorted(Ts)
    for T in Ts:
        summary, _, _ = evaluate_model(model, d, T, assert_strict_equals_exact=False)
        assert sw[T] == summary


@pytest.mark.parametrize("make_model", [lambda: CFNet(hid=4), lambda: RecurrentCNN(hid=8)])
def test_settling_curve_finite_nonnegative(make_model):
    torch.manual_seed(0)
    model = make_model().to(DEVICE).eval()
    curve = settling_curve(model, _tiny(2, 4), T_max=6, n=4)
    assert len(curve) == 6
    assert all(math.isfinite(v) for v in curve)
    assert all(v >= 0.0 for v in curve)
