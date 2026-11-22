"""Guardrail invariant checks on the trained Week-2 checkpoints.

Evidence for the two guardrails the pre-registered hypotheses lean on
(docs/hypotheses.md, unchanged here):

- H2/H4 confound (iv): "mass-drift and positivity invariants must be
  re-checked at T = 64 before the sweep is trusted" - checked here on the
  trained CF-Full (and, descriptively, CF-NoFrag) checkpoints, on the first
  64 level-5 test mazes, the same batch cf/train.py settling_curve uses.
- H3 Metric precondition: "the invariant test that the split flux is
  exactly 0 and the split head receives exactly zero gradient is a
  precondition for the arm" - checked on the trained CF-NoFrag checkpoints.

Loads every checkpoint EXACTLY as scripts/run_week2.py builds the models
(run_week2.make_model) with the best-epoch state dicts saved by
cf/train.py train_model to <results>/<Model>_seed<k>.pt, and the level-5
test split exactly as run_week2.load_data builds it.  All checks run on CPU.

Output: --out-dir (default week2_results)/trained_invariants.json with one
record per run (measurements + PASS/FAIL) and a summary.  The script exits 0
when every checkpoint was checked, whatever the PASS/FAIL outcome; the
record is the evidence, and scripts/h_verdicts.py quotes it in the caveats.
"""
import argparse
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: cf/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/: run_week2

import run_week2                                        # noqa: E402 (path setup above)

OUT_NAME = "trained_invariants.json"
NEG_EPS = -1e-7      # positivity tolerance (tests/test_cf_invariants.py NEG_EPS)
DRIFT_TOL = 1e-5     # relative mass-drift tolerance (tests/test_cf_invariants.py)
DEFAULT_MODELS = ["CF-Full", "CF-NoFrag", "RecurrentCNN"]


def checkpoint_path(results_dir, model, seed):
    """Best-epoch state dict written by cf/train.py train_model for one run tag."""
    return Path(results_dir) / ("%s_seed%d.pt" % (model, seed))


def load_checkpoint(results_dir, model, seed):
    """The model exactly as scripts/run_week2.py builds it, with trained weights.

    run_week2.make_model builds the architecture; the .pt file holds the
    best-val-strict state dict saved at the end of training (cf/train.py).
    Loads on CPU regardless of cf.train.DEVICE.
    """
    path = checkpoint_path(results_dir, model, seed)
    model = run_week2.make_model(model)
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    return model.cpu().eval()


def positivity_and_mass(model, x, T):
    """Positivity (min c1 / min c2 over t = 0..T) and per-maze relative mass drift.

    Mass drift follows tests/test_cf_invariants.py
    test_mass_relative_drift_below_1e_minus_5, computed per maze so each
    maze is judged against its own mass scale: drift of maze i at step t is
    max|M_t - M_0| / max(1, max M_0); the recorded value is the worst over
    mazes and steps.
    """
    with torch.no_grad():
        c1, c2 = model.initial_state(x)
        min_c1, min_c2 = float(c1.min()), float(c2.min())
        M0 = c1 + 2.0 * c2
        scale = M0.amax(dim=tuple(range(1, M0.dim()))).clamp(min=1.0)
        _, trace = model.logits_at(x, [T], record=True)
    max_drift = 0.0
    for c1_t, c2_t, M_t in zip(trace["c1"], trace["c2"], trace["M"]):
        min_c1 = min(min_c1, float(c1_t.min()))
        min_c2 = min(min_c2, float(c2_t.min()))
        drift = ((M_t - M0).abs().amax(dim=tuple(range(1, M0.dim())))
                 / scale).max().item()
        max_drift = max(max_drift, drift)
    return {
        "min_c1": min_c1, "min_c2": min_c2,
        "max_relative_mass_drift": max_drift,
        "positivity_ok": bool(min_c1 >= NEG_EPS and min_c2 >= NEG_EPS),
        "mass_drift_ok": bool(max_drift < DRIFT_TOL),
    }


def nofrag_split_flux(model, x, T):
    """H3 precondition, clause 1: the APPLIED split flux is exactly 0 at every step."""
    with torch.no_grad():
        _, trace = model.logits_at(x, [T], record=True)
    max_abs = max((float(s.abs().max()) for s in trace["s"]), default=0.0)
    all_zero = all(bool((s == 0).all()) for s in trace["s"])
    return {"split_flux_all_zero": all_zero, "split_flux_max_abs": max_abs}


def split_head_grad(model, x, T):
    """H3 precondition, clause 2: zero gradient to the split head (one backward).

    The split head is the second half of rate_net[2]'s output channels
    ([0:hid] = raw_merge, [hid:] = raw_split, cf/models/cf.py fluxes); the
    same backward pass tests/test_cf_invariants.py
    test_nofrag_split_head_gets_zero_grad_merge_head_nonzero uses.
    """
    model.zero_grad(set_to_none=True)
    model(x, T).square().sum().backward()
    last = model.rate_net[2]
    hid = last.out_channels // 2
    w = float(last.weight.grad[hid:].abs().max())
    b = float(last.bias.grad[hid:].abs().max())
    model.zero_grad(set_to_none=True)
    return {"split_head_grad_max_abs": max(w, b),
            "split_head_grad_all_zero": bool(w == 0.0 and b == 0.0)}


def rcnn_finite(model, x, T):
    """RecurrentCNN: no conserved state, so the run-level check is finite output
    after a T-step rollout (descriptive; no CF invariant applies)."""
    with torch.no_grad():
        logits = model(x, T)
    finite = bool(torch.isfinite(logits).all())
    return {"logits_finite": finite,
            "logits_abs_max": float(logits.abs().max())}


def collect_run(model_name, model, x, T):
    """All invariant measurements for one loaded checkpoint + PASS/FAIL.

    CF models (attribute `fluxes`, the cf/train.py settling_curve convention)
    get the positivity / mass-drift re-check; a fragmentation=False model
    additionally gets the H3 precondition checks; the RecurrentCNN gets the
    finite-output check only.  PASS means every boolean check is True.
    """
    checks = {}
    if hasattr(model, "fluxes"):
        checks.update(positivity_and_mass(model, x, T))
        if not getattr(model, "fragmentation", True):
            checks.update(nofrag_split_flux(model, x, T))
            checks.update(split_head_grad(model, x, T))
    else:
        checks.update(rcnn_finite(model, x, T))
    status = "PASS" if all(v for v in checks.values() if isinstance(v, bool)) \
        else "FAIL"
    return checks, status


def summarize(records):
    """Run counts by status and the overall PASS/FAIL of the whole record."""
    statuses = [rec.get("status") for rec in records.values()]
    n_pass = statuses.count("PASS")
    n_fail = statuses.count("FAIL")
    n_missing = statuses.count("MISSING")
    return {"n_runs": len(records), "n_pass": n_pass, "n_fail": n_fail,
            "n_missing": n_missing,
            "status": "PASS" if n_pass == len(records) and n_pass > 0
            else "FAIL"}


def check_all(results_dir, x, models, seeds, T):
    """Invariants for every (model, seed): {tag: run record}.

    A missing or unreadable checkpoint - absent, truncated, or not a valid
    .pt file at all - is recorded with status MISSING (never crashes the
    sweep); the summary reports it, and the overall status turns FAIL.
    OSError is included because a truncated zip archive surfaces as
    OSError on Windows (seek past EOF) before torch can wrap it.
    """
    records = {}
    for model_name in models:
        for seed in seeds:
            tag = "%s_seed%d" % (model_name, seed)
            try:
                model = load_checkpoint(results_dir, model_name, seed)
            except (FileNotFoundError, RuntimeError, KeyError, OSError,
                    zipfile.BadZipFile, pickle.UnpicklingError,
                    EOFError) as exc:
                records[tag] = {"model": model_name, "seed": seed,
                                "checkpoint": str(checkpoint_path(
                                    results_dir, model_name, seed)),
                                "status": "MISSING", "error": "%s: %s"
                                % (type(exc).__name__, exc)}
                print("[%s] MISSING (%s)" % (tag, exc))
                continue
            checks, status = collect_run(model_name, model, x, T)
            records[tag] = {"model": model_name, "seed": seed,
                           "checkpoint": checkpoint_path(
                               results_dir, model_name, seed).name,
                           "status": status, "checks": checks}
            print("[%s] %s" % (tag, format_checks(checks, status)))
    return records


def format_checks(checks, status):
    """One-line summary of a run's measurements (solely for the printed table)."""
    def g(key):
        return "%.3g" % checks[key] if isinstance(checks.get(key), float) else "n/a"

    parts = ["min c1 = %s" % g("min_c1"), "min c2 = %s" % g("min_c2"),
             "max mass drift = %s" % g("max_relative_mass_drift")]
    if "split_flux_max_abs" in checks:
        parts += ["max |s| = %s" % g("split_flux_max_abs"),
                  "max |split grad| = %s" % g("split_head_grad_max_abs")]
    if "logits_abs_max" in checks:
        parts += ["max |logit| = %s" % g("logits_abs_max")]
    return "%s | %s" % (status, ", ".join(parts))


def build_payload(records, models, seeds, T, n_mazes, batch_desc):
    """The JSON-serializable trained_invariants.json payload."""
    tolerances = {"positivity_min_c": NEG_EPS, "relative_mass_drift_max": DRIFT_TOL,
                  "split_flux": "exactly 0 (bitwise)",
                  "split_head_grad": "exactly 0 (bitwise)"}
    return {
        "what": "Guardrail invariant checks on the trained Week-2 checkpoints "
                "(docs/hypotheses.md H2/H4 confound (iv) T = 64 re-check; H3 "
                "Metric precondition)",
        "tolerances": tolerances,
        "config": {"T": T, "n_mazes": n_mazes, "models": list(models),
                   "seeds": list(seeds), "device": "cpu", "batch": batch_desc},
        "runs": records,
        "summary": summarize(records),
    }


def write_payload(out_dir, payload):
    """Atomically write trained_invariants.json (no partial file on crash)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / OUT_NAME
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(str(tmp), str(path))
    return path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(
        Path(__file__).resolve().parents[1] / "week2_results"),
        help="folder with the <Model>_seed<k>.pt checkpoints "
             "(default: week2_results)")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=run_week2.ALLOWED_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--T", type=int, default=64,
                    help="rollout length of the invariant checks (default 64)")
    ap.add_argument("--n-mazes", type=int, default=64,
                    help="first N mazes of the level-5 test split (default 64, "
                         "the settling_curve batch)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = Path(args.results)
    # the level-5 test split exactly as scripts/run_week2.py builds it
    # (same seeded make_splits call order; defaults match the Week-2 run)
    data = run_week2.load_data(dict())
    x = data["test"].x[:args.n_mazes].cpu()
    batch_desc = ("level-5 test split (scripts/run_week2.py load_data), "
                  "first %d mazes; H = %d" % (args.n_mazes, x.shape[-1]))
    records = check_all(results, x, args.models, args.seeds, args.T)
    payload = build_payload(records, args.models, args.seeds, args.T,
                            args.n_mazes, batch_desc)
    path = write_payload(results, payload)
    summary = payload["summary"]
    print("\n%d runs: %d PASS, %d FAIL, %d MISSING -> %s"
          % (summary["n_runs"], summary["n_pass"], summary["n_fail"],
             summary["n_missing"], summary["status"]))
    print("invariants written to %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
