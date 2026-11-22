"""Unit tests for scripts/check_trained_invariants.py (synthetic mazes, CPU).

The invariant check functions are exercised on tiny CF / RCNN models over a
small synthetic maze batch: the PASS path on structurally-invariant models,
the FAIL path via CF-NoBound (positivity) and CF-Full (split flux / split
head gradient), plus record assembly and the CLI end-to-end on fake
checkpoints with the Hugging Face dataset loader stubbed out.
"""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_trained_invariants as cti                                # noqa: E402
import run_week2                                                      # noqa: E402
from cf.models.cf import CFNet, MODEL_SPECS                           # noqa: E402
from cf.models.rcnn import RecurrentCNN                               # noqa: E402
from maze_synthetic import make_maze_grid                             # noqa: E402


def maze_batch(n=4, lattice=3):
    """(n, 3, 2*lattice+1, 2*lattice+1) curriculum-format channel batch."""
    xs = []
    for s in range(n):
        grid = make_maze_grid(np.random.default_rng(s), lattice)
        walk = (grid == 0).astype(np.float32)
        start = np.zeros_like(walk); start[1, 1] = 1.0
        goal = np.zeros_like(walk); goal[-2, -2] = 1.0
        xs.append(np.stack([walk, start, goal]))
    return torch.from_numpy(np.stack(xs))


T = 16          # short rollout keeps the suite fast; the checks are T-generic


# ---- pure check functions --------------------------------------------------

def test_positivity_and_mass_pass_on_cf_full():
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-Full"])
    checks = cti.positivity_and_mass(model, maze_batch(), T)
    assert checks["min_c1"] >= cti.NEG_EPS and checks["min_c2"] >= cti.NEG_EPS
    assert checks["max_relative_mass_drift"] < cti.DRIFT_TOL
    assert checks["positivity_ok"] is True
    assert checks["mass_drift_ok"] is True


def test_positivity_and_mass_detects_negative_concentration():
    # CF-NoBound conserves mass but can drive c1 negative: the check must FAIL
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-NoBound"])
    with torch.no_grad():
        model.rate_net[2].bias[: model.rate_net[2].out_channels // 2].fill_(1.0)
    checks = cti.positivity_and_mass(model, maze_batch(), T)
    assert checks["min_c1"] < cti.NEG_EPS
    assert checks["positivity_ok"] is False
    assert checks["mass_drift_ok"] is True        # mass is still conserved


def test_positivity_and_mass_detects_mass_drift_against_doctored_M0():
    # the CF update cannot drift mass, so the drift bookkeeping is checked by
    # feeding it an M0 that does not match the traced M
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-Full"])
    x = maze_batch()
    with torch.no_grad():
        _, trace = model.logits_at(x, [T], record=True)
        c1, c2 = model.initial_state(x)
    M0 = c1 + 2.0 * c2 + 1e5          # nothing like the traced M: huge drift
    scale = M0.amax(dim=(1, 2, 3)).clamp(min=1.0)
    drifts = [(M_t - M0).abs().amax(dim=(1, 2, 3)).div(scale).max().item()
              for M_t in trace["M"]]
    assert max(drifts) > cti.DRIFT_TOL        # the drift the check must catch


def test_nofrag_split_flux_is_exactly_zero():
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-NoFrag"])
    checks = cti.nofrag_split_flux(model, maze_batch(), T)
    assert checks["split_flux_all_zero"] is True
    assert checks["split_flux_max_abs"] == 0.0


def test_split_flux_check_detects_fragmentation():
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-Full"])
    checks = cti.nofrag_split_flux(model, maze_batch(), T)
    assert checks["split_flux_all_zero"] is False
    assert checks["split_flux_max_abs"] > 0.0


def test_split_head_grad_zero_on_nofrag_nonzero_on_full():
    torch.manual_seed(0)
    nofrag = CFNet(**MODEL_SPECS["CF-NoFrag"])
    x = maze_batch(n=2)
    checks = cti.split_head_grad(nofrag, x, T)
    assert checks["split_head_grad_all_zero"] is True
    assert checks["split_head_grad_max_abs"] == 0.0
    # the backward pass leaves the model clean for the next check
    assert all(p.grad is None for p in nofrag.parameters())

    full = CFNet(**MODEL_SPECS["CF-Full"])
    checks = cti.split_head_grad(full, x, T)
    assert checks["split_head_grad_all_zero"] is False
    assert checks["split_head_grad_max_abs"] > 0.0


def test_collect_run_pass_and_fail_status():
    torch.manual_seed(0)
    x = maze_batch(n=2)
    assert cti.collect_run("CF-NoFrag", CFNet(**MODEL_SPECS["CF-NoFrag"]),
                           x, T)[1] == "PASS"
    assert cti.collect_run("CF-Full", CFNet(**MODEL_SPECS["CF-Full"]),
                           x, T)[1] == "PASS"
    nobound = CFNet(**MODEL_SPECS["CF-NoBound"])
    with torch.no_grad():
        nobound.rate_net[2].bias[: nobound.rate_net[2].out_channels // 2].fill_(1.0)
    checks, status = cti.collect_run("CF-NoBound", nobound, x, T)
    assert status == "FAIL"
    assert checks["positivity_ok"] is False


def test_rcnn_finite_check():
    torch.manual_seed(0)
    model = RecurrentCNN(hid=4)
    x = maze_batch(n=2)
    checks = cti.rcnn_finite(model, x, T)
    assert checks["logits_finite"] is True
    assert cti.collect_run("RecurrentCNN", model, x, T)[1] == "PASS"


# ---- checkpoint loading ---------------------------------------------------

def test_load_checkpoint_builds_model_exactly_as_run_week2(tmp_path):
    torch.manual_seed(0)
    model = run_week2.make_model("CF-NoFrag")
    torch.save(model.state_dict(), str(tmp_path / "CF-NoFrag_seed1.pt"))
    loaded = cti.load_checkpoint(tmp_path, "CF-NoFrag", 1)
    assert type(loaded) is type(model)
    assert loaded.fragmentation is model.fragmentation
    for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                  loaded.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2)


# ---- record assembly -------------------------------------------------------

def test_summarize_counts_statuses():
    records = {"a": {"status": "PASS"}, "b": {"status": "PASS"},
              "c": {"status": "FAIL"}, "d": {"status": "MISSING"}}
    assert cti.summarize(records) == {"n_runs": 4, "n_pass": 2, "n_fail": 1,
                                      "n_missing": 1, "status": "FAIL"}
    assert cti.summarize({"a": {"status": "PASS"}})["status"] == "PASS"
    assert cti.summarize({})["status"] == "FAIL"


def test_check_all_records_missing_checkpoint_without_crashing(tmp_path):
    records = cti.check_all(tmp_path, maze_batch(n=2), ["CF-Full"], [0], T)
    rec = records["CF-Full_seed0"]
    assert rec["status"] == "MISSING" and "CF-Full_seed0.pt" in rec["error"]
    summary = cti.summarize(records)
    assert summary["n_missing"] == 1 and summary["status"] == "FAIL"


def test_check_all_records_truncated_checkpoint_as_missing(tmp_path, capsys):
    # half a saved state dict is not a loadable checkpoint: whatever torch
    # wraps the failure in, the run must be MISSING with the error type
    torch.manual_seed(0)
    model = run_week2.make_model("CF-Full")
    p = tmp_path / "CF-Full_seed0.pt"
    torch.save(model.state_dict(), str(p))
    blob = p.read_bytes()
    p.write_bytes(blob[: len(blob) // 2])
    records = cti.check_all(tmp_path, maze_batch(n=2), ["CF-Full"], [0], T)
    rec = records["CF-Full_seed0"]
    assert rec["status"] == "MISSING"
    assert "CF-Full_seed0.pt" in rec["checkpoint"]
    assert "Error" in rec["error"]           # "<ErrorType>: <message>"
    assert cti.summarize(records)["n_missing"] == 1
    assert "MISSING" in capsys.readouterr().out


def test_check_all_records_corrupt_pickle_checkpoint_as_missing(tmp_path):
    # a .pt whose bytes fail inside the unpickler (UnpicklingError / EOFError
    # from pickle, not a torch RuntimeError) must be MISSING too, so the
    # sweep cannot crash on a truncated legacy checkpoint
    p = tmp_path / "CF-NoFrag_seed0.pt"
    p.write_bytes(pickle.dumps({"a": list(range(50))})[:20])
    records = cti.check_all(tmp_path, maze_batch(n=2), ["CF-NoFrag"], [0], T)
    rec = records["CF-NoFrag_seed0"]
    assert rec["status"] == "MISSING"
    assert "UnpicklingError" in rec["error"]      # the error type is recorded


# ---- CLI end-to-end (HF dataset loader stubbed) ---------------------------

def stub_data(monkeypatch, x):
    monkeypatch.setattr(cti.run_week2, "load_data",
                        lambda cfg: {"test": SimpleNamespace(x=x)})


def test_cli_passes_and_writes_json(tmp_path, monkeypatch, capsys):
    models = ["CF-Full", "CF-NoFrag", "RecurrentCNN"]
    for model_name in models:
        torch.manual_seed(0)
        model = run_week2.make_model(model_name)
        torch.save(model.state_dict(), str(tmp_path / ("%s_seed0.pt" % model_name)))
    stub_data(monkeypatch, maze_batch(n=4))

    ret = cti.main(["--results", str(tmp_path), "--seeds", "0",
                    "--T", "8", "--n-mazes", "4"])
    assert ret == 0
    payload = json.loads((tmp_path / "trained_invariants.json")
                        .read_text(encoding="utf-8"))
    assert payload["summary"] == {"n_runs": 3, "n_pass": 3, "n_fail": 0,
                                  "n_missing": 0, "status": "PASS"}
    assert payload["config"]["T"] == 8 and payload["config"]["n_mazes"] == 4
    for model_name in models:
        rec = payload["runs"]["%s_seed0" % model_name]
        assert rec["status"] == "PASS" and rec["seed"] == 0
    nofrag = payload["runs"]["CF-NoFrag_seed0"]["checks"]
    assert nofrag["split_flux_all_zero"] is True
    assert nofrag["split_head_grad_all_zero"] is True
    full = payload["runs"]["CF-Full_seed0"]["checks"]
    assert "split_flux_max_abs" not in full      # H3 checks are NoFrag-only
    trained = " ".join(capsys.readouterr().out.splitlines())
    assert "invariants written to" in trained


def test_cli_still_exits_zero_when_an_invariant_fails(tmp_path, monkeypatch,
                                                      capsys):
    # CF-NoBound breaks positivity with a large merge bias (bound_flux=False,
    # so m = c1 * sigmoid can exceed c1 / 2 and drive c1 negative); the check
    # must catch it and the CLI must still exit 0, the record being the point
    torch.manual_seed(0)
    nobound = run_week2.make_model("CF-NoBound")
    with torch.no_grad():
        nobound.rate_net[2].bias[: nobound.rate_net[2].out_channels // 2].fill_(1.0)
    torch.save(nobound.state_dict(), str(tmp_path / "CF-NoBound_seed0.pt"))
    stub_data(monkeypatch, maze_batch(n=4))

    ret = cti.main(["--results", str(tmp_path), "--models", "CF-NoBound",
                    "--seeds", "0", "--T", "8", "--n-mazes", "4"])
    assert ret == 0
    payload = json.loads((tmp_path / "trained_invariants.json")
                        .read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "FAIL"
    checks = payload["runs"]["CF-NoBound_seed0"]["checks"]
    assert payload["runs"]["CF-NoBound_seed0"]["status"] == "FAIL"
    assert checks["positivity_ok"] is False and checks["min_c1"] < cti.NEG_EPS
    assert "FAIL" in capsys.readouterr().out
