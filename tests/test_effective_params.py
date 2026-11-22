"""Unit tests for run_week2.effective_n_params (offline, synthetic batch)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from maze_synthetic import make_batch
import cf.train as train_mod
from cf.data.curriculum import MazeDataset
from cf.models.rcnn import n_params
import run_week2


def _train_batch(seed=0, count=8):
    """A real-format training batch: (x, y) from synthetic curriculum mazes."""
    pairs = make_batch(seed=seed, count=count)
    mazes = [ascii for ascii, _ in pairs]
    sols = [sol for _, sol in pairs]
    ds = MazeDataset(list(range(len(mazes))), mazes, sols)
    return ds.x, ds.y


def test_effective_n_params_expected_counts_for_hid16_variants():
    x, y = _train_batch()
    full = run_week2.make_model("CF-Full")
    nofrag = run_week2.make_model("CF-NoFrag")
    assert n_params(full) == 10177 and n_params(nofrag) == 10177
    assert run_week2.effective_n_params(full, x, y, T=8) == 10177
    # CF-NoFrag computes the split-flux head but discards it: those 2320
    # entries (16 out-channels x (16x3x3 weights + 1 bias)) get zero gradient
    assert run_week2.effective_n_params(nofrag, x, y, T=8) == 7857


def test_effective_n_params_matches_raw_for_recurrent_cnn():
    x, y = _train_batch()
    rcnn = run_week2.make_model("RecurrentCNN")
    assert run_week2.effective_n_params(rcnn, x, y, T=8) == n_params(rcnn)


def test_effective_n_params_accepts_pos_weight_and_leaves_no_grads():
    x, y = _train_batch()
    model = run_week2.make_model("CF-NoFrag")
    eff = run_week2.effective_n_params(model, x, y, T=8, pos_weight=3.0)
    assert eff == 7857
    assert all(p.grad is None for p in model.parameters())


def test_effective_n_params_ignores_dead_parameters():
    class Dead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(4, 2)
            self.dead = torch.nn.Linear(4, 4)     # created but never called

        def forward(self, x, T=None):
            return self.used(x)

    x = torch.randn(3, 4)
    y = torch.rand(3, 2)
    assert run_week2.effective_n_params(Dead(), x, y, T=1) == 4 * 2 + 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_effective_n_params_moves_model_to_the_batch_device_on_cuda():
    """execute_run builds a CPU model but feeds cf.train.DEVICE batches:
    effective_n_params must move the model to the batch's device."""
    assert train_mod.DEVICE.type == "cuda"
    x, y = _train_batch()
    model = run_week2.make_model("CF-NoFrag")
    x, y = x.to(train_mod.DEVICE), y.to(train_mod.DEVICE)
    eff = run_week2.effective_n_params(model, x, y, T=8, pos_weight=3.0)
    assert eff == 7857                     # same count as on CPU (see above)
    assert next(model.parameters()).device == x.device
