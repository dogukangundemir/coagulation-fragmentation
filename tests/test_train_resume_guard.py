"""Resume-guard checks: a checkpoint may only be resumed by the run that wrote it.

Regression: seeds 1 and 2 sharing one checkpoint_path silently loaded seed 0's
finished checkpoint and trained 0 steps. Checkpoints now carry run_id plus a
fingerprint (model class, parameter names+shapes, hyperparameters, T, seed and
the effective pos_weight); a resume with a different run_id or fingerprint
raises instead of continuing silently. The seed, T and pos_weight fields make
the collision above detectable even under the default run_id=None, where the
run_id branch cannot fire.
"""
import pytest
import torch
from pathlib import Path

import cf.train as train_mod
from cf.data.curriculum import MazeDataset
from cf.models.rcnn import RecurrentCNN
from cf.train import DEVICE, train_model

from maze_synthetic import make_batch

GRID = 9


def _tiny(seed, count, n=4):
    pairs = make_batch(seed=seed, count=count, n=n)
    mazes = [m for m, _ in pairs]
    sols = [s for _, s in pairs]
    return MazeDataset(range(count), mazes, sols)


def _make():
    return RecurrentCNN(hid=8)


class _FakeClock:
    """Deterministic injectable clock: every call advances by 10 seconds."""

    def __init__(self, step=10.0):
        self.step = step
        self.t = 0.0

    def __call__(self):
        self.t += self.step
        return self.t


def _args(tmp_path, tag):
    return dict(T=4, seed=0, tag=tag, train_ds=_tiny(0, 6), val_ds=_tiny(1, 4),
                results_dir=tmp_path, verbose=False, assert_strict_equals_exact=False)


def _train_one_epoch(tmp_path, ck_path, tag, run_id):
    """Train with a 3 s budget and a +10 s-per-call fake clock: epoch 0 completes,
    epoch 1 trips the budget at its first step, leaving a checkpoint at epoch 0."""
    _, hist = train_model(_make, max_minutes=0.05, checkpoint_path=str(ck_path),
                          run_id=run_id, _clock=_FakeClock(), **_args(tmp_path, tag))
    assert hist["stopped_early"] is True
    assert hist["epochs_done"] == 1
    assert ck_path.exists()
    return hist


@pytest.fixture(autouse=True)
def _two_epochs(monkeypatch):
    monkeypatch.setitem(train_mod.HYPERPARAMS, "epochs", 2)


def test_resume_with_different_run_id_raises(tmp_path):
    ck_path = tmp_path / "run.pt"
    _train_one_epoch(tmp_path, ck_path, "a", run_id="run-a")
    with pytest.raises(ValueError, match="run.pt.*run-a.*run-b"):
        train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path),
                    run_id="run-b", **_args(tmp_path, "a"))
    # the checkpoint itself is untouched by the refused resume
    ck = torch.load(str(ck_path), map_location="cpu")
    assert ck["run_id"] == "run-a"
    assert ck["epoch"] == 0


def test_resume_with_different_hyperparams_raises(tmp_path, monkeypatch):
    ck_path = tmp_path / "run.pt"
    _train_one_epoch(tmp_path, ck_path, "a", run_id="run-a")
    monkeypatch.setitem(train_mod.HYPERPARAMS, "epochs", 3)   # different plan
    with pytest.raises(ValueError, match="run.pt.*hyperparams.*epochs"):
        train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path),
                    run_id="run-a", **_args(tmp_path, "a"))


def test_resume_with_different_model_raises(tmp_path):
    ck_path = tmp_path / "run.pt"
    _train_one_epoch(tmp_path, ck_path, "a", run_id="run-a")
    with pytest.raises(ValueError, match="run.pt.*fingerprint mismatch.*param_shapes"):
        train_model(lambda: RecurrentCNN(hid=6), max_minutes=10.0,
                    checkpoint_path=str(ck_path), run_id="run-a",
                    **_args(tmp_path, "a"))


def test_resume_with_different_seed_default_run_id_raises(tmp_path):
    """The named regression, under the default run_id=None: two calls differing
    only in seed but sharing one checkpoint_path must not silently resume each
    other's checkpoints (the fingerprint's seed field catches it)."""
    ck_path = tmp_path / "run.pt"
    _train_one_epoch(tmp_path, ck_path, "a", run_id=None)
    args = _args(tmp_path, "a")
    args["seed"] = 1
    with pytest.raises(ValueError, match="run.pt.*fingerprint mismatch.*seed"):
        train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path), **args)


def test_same_run_id_resumes_and_continues(tmp_path):
    ck_path = tmp_path / "run.pt"
    hist1 = _train_one_epoch(tmp_path, ck_path, "a", run_id="run-a")
    model, hist2 = train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path),
                               run_id="run-a", **_args(tmp_path, "a"))
    assert hist2["epochs_done"] == 2
    assert not hist2["stopped_early"]
    assert hist2["train_loss"][0] == hist1["train_loss"][0]   # history continued
    assert len(hist2["train_loss"]) == 2
    model.eval()
    with torch.no_grad():
        out = model(_tiny(1, 4).x[:2].to(DEVICE), T=4)
    assert out.shape == (2, GRID, GRID)


def test_default_run_id_none_roundtrip_resumes(tmp_path):
    """Default arguments: saving and resuming with run_id=None still works."""
    ck_path = tmp_path / "run.pt"
    hist1 = _train_one_epoch(tmp_path, ck_path, "a", run_id=None)
    _, hist2 = train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path),
                           **_args(tmp_path, "a"))
    assert hist2["epochs_done"] == 2
    assert hist2["train_loss"][0] == hist1["train_loss"][0]


def test_default_no_checkpoint_path_runs_full(tmp_path):
    """Default path unchanged: no checkpoint_path -> plain full training."""
    model, hist = train_model(_make, **_args(tmp_path, "d"))
    assert len(hist["train_loss"]) == 2
    assert "stopped_early" not in hist


def test_pre_guard_checkpoint_run_id_none_warns_and_resumes(tmp_path):
    """A legacy checkpoint without run_id/fingerprint tags: warn, but resume
    (backward compatibility) when run_id is None."""
    ck_path = tmp_path / "legacy.pt"
    _train_one_epoch(tmp_path, ck_path, "a", run_id=None)
    ck = torch.load(str(ck_path), map_location="cpu")
    for k in ("run_id", "fingerprint"):
        ck.pop(k)
    torch.save(ck, str(ck_path))                        # strip the guard tags
    with pytest.warns(UserWarning, match="legacy.pt"):
        _, hist2 = train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path),
                              **_args(tmp_path, "a"))
    assert hist2["epochs_done"] == 2


def test_pre_guard_checkpoint_with_run_id_raises(tmp_path):
    """Same legacy checkpoint, but run_id is given: identity cannot be verified."""
    ck_path = tmp_path / "legacy.pt"
    _train_one_epoch(tmp_path, ck_path, "a", run_id=None)
    ck = torch.load(str(ck_path), map_location="cpu")
    for k in ("run_id", "fingerprint"):
        ck.pop(k)
    torch.save(ck, str(ck_path))
    with pytest.raises(ValueError, match="legacy.pt.*predate"):
        train_model(_make, max_minutes=10.0, checkpoint_path=str(ck_path),
                    run_id="run-a", **_args(tmp_path, "a"))


def test_checkpoint_stores_run_id_and_fingerprint(tmp_path):
    ck_path = tmp_path / "run.pt"
    _train_one_epoch(tmp_path, ck_path, "a", run_id="run-x")
    ck = torch.load(str(ck_path), map_location="cpu")
    fp = ck["fingerprint"]
    assert ck["run_id"] == "run-x"
    assert fp["model_class"] == "RecurrentCNN"
    assert fp["param_shapes"] == sorted((n, tuple(p.shape))
                                        for n, p in _make().named_parameters())
    assert fp["T"] == 4
    assert fp["seed"] == 0
    density = _tiny(0, 6).y.mean().item()
    assert fp["pos_weight"] == pytest.approx((1.0 - density) / density)
    assert fp["hyperparams"] == dict(train_mod.HYPERPARAMS)
