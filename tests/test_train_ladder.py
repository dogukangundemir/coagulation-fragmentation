"""Offline checks for scripts/train_ladder.py (synthetic mazes, CPU)."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import cf.train as train_mod
import train_ladder
from cf.data.curriculum import MazeDataset
from cf.models.ladder import LadderCFNet
from maze_synthetic import make_batch


@pytest.fixture(autouse=True)
def _restore_hyperparams():
    """train_ladder.run mutates cf.train.HYPERPARAMS['epochs']; undo that."""
    saved = dict(train_mod.HYPERPARAMS)
    yield
    train_mod.HYPERPARAMS.clear()
    train_mod.HYPERPARAMS.update(saved)


def synthetic_data(n=12, seed=0):
    """Small offline stand-in for the curriculum splits, same Dataset class."""
    rows = make_batch(seed=seed, count=n, n=5)
    mazes = [m for m, _ in rows]
    sols = [s for _, s in rows]
    ds = lambda k: MazeDataset(range(k), mazes, sols)           # noqa: E731
    return dict(train=ds(n), val=ds(n), test=ds(n),
                probe7=ds(n), probe9=ds(n))


def test_budget_label_carries_epochs_and_plan():
    assert train_ladder.budget_label(37, 160) == \
        "preliminary, 1 seed, reduced budget (37/160 epochs)"
    assert train_ladder.budget_label(160) == \
        "preliminary, 1 seed, reduced budget (160/160 epochs)"


def test_make_model_is_the_wall_aware_detailed_balance_ladder():
    model = train_ladder.make_model(hid=4, K=4)
    assert isinstance(model, train_ladder.LadderStable)
    assert isinstance(model, LadderCFNet)
    assert model.wall_aware and model.detailed_balance
    assert model.K == 4 and model.hid == 4


def test_stability_guards_keep_mass_positivity_and_finite_grads():
    """The guards may tame the backward, never the chemistry."""
    rows = make_batch(seed=1, count=2)
    rows = [(m, s) for m, s in [rows[0]]] + [(m, s) for m, s in [rows[1]]]
    ds = MazeDataset(range(2), [m for m, _ in rows], [s for _, s in rows])
    model = train_ladder.make_model(hid=3, K=4)
    x = ds.x
    c = model.initial_state(x)
    M0 = model.mass(c)
    rates = None
    for t in range(1, 33):                    # 32 steps like the inference sweep
        if rates is None or (t - 1) % model.r_s == 0:
            rates = model.refresh_rates(x, c)
        c, J = model.micro_step(c, rates)
        assert (c >= -1e-7).all()
        assert torch.isfinite(c).all() and torch.isfinite(J).all()
    drift = (model.mass(c) - M0).abs().max() / M0.abs().max()
    assert drift < 1e-5
    a, b = model.refresh_rates(x, c)
    assert a.max() <= train_ladder.RATE_MAX + 1e-6
    assert b.max() <= train_ladder.RATE_MAX + 1e-6
    logits = model.read(x, c)
    loss = logits.sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_rate_refresh_marks_match_the_rollout_schedule():
    refresh, phase = train_ladder.rate_refresh_marks(5, r_s=2)
    assert refresh == [1, 3, 5]                # slow phase before odd steps
    assert phase == [0, 0, 1, 1, 2]            # steps 1-2, 3-4, 5 share rates
    refresh, phase = train_ladder.rate_refresh_marks(4, r_s=3)
    assert refresh == [1, 4] and phase == [0, 0, 0, 1]


def test_rate_refresh_marks_agree_with_logits_at_refresh_count():
    """The model must actually refresh the rates at the marked steps."""
    x = torch.randn(2, 3, 11, 11).abs()
    model = train_ladder.make_model(hid=2, K=4)
    calls = []
    orig = model.refresh_rates

    def counting(*a, **kw):
        calls.append(1)
        return orig(*a, **kw)

    model.refresh_rates = counting
    T = 7
    model.logits_at(x, [T], record=True)
    refresh, _ = train_ladder.rate_refresh_marks(T, model.r_s)
    assert len(calls) == len(refresh)


def test_logits_at_agrees_with_forward_at_every_marked_horizon():
    x = torch.randn(2, 3, 11, 11).abs()
    model = train_ladder.make_model(hid=2, K=4).eval()
    with torch.no_grad():
        for T in (1, 3, 7, 8):
            assert torch.allclose(model.logits_at(x, [T])[T],
                                 model.forward(x, T), atol=1e-5)


def test_fast_pipeline_trains_selects_T_and_writes_json(tmp_path, monkeypatch):
    train_mod.DEVICE = torch.device("cpu")
    monkeypatch.setattr(train_ladder, "load_data", lambda cfg: synthetic_data())
    out = tmp_path / "ladder"
    argv = ["--out", str(out), "--fast", "--hid", "2", "--K", "4",
            "--sweep-T", "1", "4", "8", "--V-mazes", "4", "--V-T", "4"]
    payload, path = train_ladder.run(argv)
    assert path.exists()
    p = json.loads(path.read_text(encoding="utf-8"))
    assert p["label"] == "preliminary, 1 seed, reduced budget (2/2 epochs)"
    assert p["params"]["raw"] > 0 and p["params"]["effective"] > 0
    assert p["selection"]["T_star"] in (1, 4, 8)
    assert p["test_strict_at_T_train"] >= 0.0
    # energy trace: one V per outer step, n_f micro values per step, phased
    e = p["energy"]
    assert len(e["V_by_outer_step"]) == 4
    assert len(e["V_by_micro_step"]) == 8                 # T_V * n_f = 4 * 2
    assert e["fixed_rate_phase_by_step"] == [0, 0, 1, 1]
    assert e["rate_refresh_steps"] == [1, 3]
    assert (out / "BD-WallAware_seed0.ckpt.pt").exists()
    # the printed table carries T=8 and T* columns
    assert "T*=" in train_ladder.print_table(p)


def test_eval_only_restores_the_checkpoint_and_rewrites_json(tmp_path, monkeypatch):
    train_mod.DEVICE = torch.device("cpu")
    monkeypatch.setattr(train_ladder, "load_data", lambda cfg: synthetic_data())
    out = tmp_path / "ladder"
    argv = ["--out", str(out), "--fast", "--hid", "2",
            "--sweep-T", "1", "4", "8", "--V-mazes", "4", "--V-T", "4"]
    train_ladder.run(argv)
    payload, path = train_ladder.run(argv + ["--eval-only"])
    p = json.loads(path.read_text(encoding="utf-8"))
    assert p["training"]["epochs_done"] == 2
    assert p["params"]["effective"] == "not run (eval-only)"
    first = json.loads((out / "ladder_metrics.json").read_text(encoding="utf-8"))
    assert first["test_strict_at_T_train"] == p["test_strict_at_T_train"]
