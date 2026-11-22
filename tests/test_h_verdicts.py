"""Decision functions and verdict assembly of scripts/h_verdicts.py.

Every decision function is exercised on synthetic inputs for all of its
outcomes (SUPPORTED / NOT SUPPORTED / INCONCLUSIVE), the budget rule is
tested for every blocking case, and the whole payload assembly is checked
end-to-end on a synthetic week2_metrics payload, including the
missing-data -> INCONCLUSIVE path and the CLI writer.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import h_verdicts as hv


# ---- synthetic data -------------------------------------------------------

SEEDS = [0, 1, 2]

# strict accuracy sweeps engineered so that, with all runs finished:
#   H1 = NOT SUPPORTED, H2 = SUPPORTED, H3 primary = SUPPORTED,
#   H3 secondary = SUPPORTED, H4 = SUPPORTED (both clauses).
SC_A8 = [0.20, 0.25, 0.15]     # CF-Full at T=8
SC_A16 = [0.40, 0.35, 0.30]    # CF-Full at T=16
SC_R8 = [0.90, 0.95, 0.85]     # RecurrentCNN at T=8
SC_N8 = [0.10, 0.10, 0.10]     # CF-NoFrag at T=8
SC_N16 = [0.15, 0.05, 0.02]    # CF-NoFrag at T=16
CF_SETTLE = {0: (0.010, 0.005), 1: (0.012, 0.004), 2: (0.008, 0.003)}
RC_SETTLE = {0: (0.5, 0.9), 1: (0.4, 0.8), 2: (0.6, 1.0)}


def make_run(model, seed, strict_by_T, iou8=None, probe_strict=None,
             settling=None, epochs_done=40, planned=None, stopped_early=False,
             status="done"):
    run = {
        "model": model, "seed": seed, "status": status,
        "epochs_done": epochs_done, "best_epoch": epochs_done - 2,
        "stopped_early": stopped_early,
        "sweeps": {
            "test": {str(T): {"strict": v,
                              "iou": (v + 0.05) if iou8 is None else iou8}
                     for T, v in strict_by_T.items()},
            "probe7": {"8": {"strict": probe_strict or 0.10}},
            "probe9": {"8": {"strict": (probe_strict or 0.10) / 2.0}},
        },
        "test": {"strict": strict_by_T[8], "iou": 0.99},
    }
    if settling is not None:
        run["settling"] = settling
    if planned is not None:
        run["planned_epochs"] = planned
    return run


def settling_curve(step8, step16, length=32):
    curve = [0.0] * length
    curve[7] = step8      # list index t-1 holds step t
    curve[15] = step16
    return curve


def synth_metrics(**overrides):
    runs = {}
    for i, seed in enumerate(SEEDS):
        runs["CF-Full_seed%d" % seed] = make_run(
            "CF-Full", seed,
            {8: SC_A8[i], 16: SC_A16[i], 24: 0.10, 32: 0.05},
            settling=settling_curve(*CF_SETTLE[seed]))
        runs["RecurrentCNN_seed%d" % seed] = make_run(
            "RecurrentCNN", seed,
            {8: SC_R8[i], 16: SC_R8[i] / 10.0, 24: 0.0, 32: 0.0},
            settling=settling_curve(*RC_SETTLE[seed]))
        runs["CF-NoFrag_seed%d" % seed] = make_run(
            "CF-NoFrag", seed, {8: SC_N8[i], 16: SC_N16[i]})
    metrics = {"config": {"seeds": list(SEEDS), "T_train": 8},
               "hyperparams": {"epochs": 40},
               "runs": runs}
    metrics.update(overrides)
    return metrics


def synth_valt():
    return {"selection_rule": "T* = argmax val strict (ties: smaller T)",
            "selections": [
                {"model": m, "seed": s, "T_star": 12 if m == "CF-Full" else 8,
                 "val_strict_at_T_star": 0.5,
                 "test_strict_at_T_star": 0.6 if m == "CF-Full" else 0.9}
                for m in ("CF-Full", "RecurrentCNN") for s in SEEDS]}


def synth_followups():
    runs = {}
    for seed in SEEDS:
        runs["E1_CF-Full_seed%d" % seed] = {
            "exp": "E1_CF-Full", "seed": seed, "status": "done",
            "planned_epochs": 160, "epochs_done": 160, "T_star": 16,
            "test_strict_by_T": {"8": 0.73, "16": 0.93},
            "test_strict_at_T_star": 0.93}
        runs["E2_CF-NoFrag_seed%d" % seed] = {
            "exp": "E2_CF-NoFrag", "seed": seed, "status": "done",
            "planned_epochs": 40, "epochs_done": 40, "T_star": 16,
            "test_strict_by_T": {"8": 0.10, "16": 0.36},
            "test_strict_at_T_star": 0.36}
    return {"config": {"seeds": list(SEEDS)}, "runs": runs}


# ---- global decision rule -------------------------------------------------

def test_sample_sd_uses_ddof_1():
    assert hv.sample_sd([1.0, 2.0, 3.0]) == 1.0
    assert hv.sample_sd([5.0, 5.0, 5.0]) == 0.0
    assert hv.sample_sd([1.0]) == 0.0


def test_directional_rule_supported():
    assert hv.directional_rule([0.5, 0.6, 0.7], [0.4, 0.5, 0.6]) == \
        hv.SUPPORTED
    # equal per-seed differences give sd(d) = 0, which the mean difference
    # exceeds
    assert hv.directional_rule([0.2, 0.2, 0.2], [0.1, 0.1, 0.1]) == \
        hv.SUPPORTED


def test_directional_rule_not_supported():
    # the opposite direction (B > A) meets the same three conditions
    assert hv.directional_rule([0.4, 0.5, 0.6], [0.5, 0.6, 0.7]) == \
        hv.NOT_SUPPORTED
    # H1 shape: claimed arm A (CF-Full) loses on every seed to arm B
    assert hv.directional_rule(SC_A8, SC_R8) == hv.NOT_SUPPORTED


def test_directional_rule_inconclusive_when_a_seed_flips():
    a = [0.50, 0.60, 0.55]
    b = [0.45, 0.50, 0.60]      # seed 2 flips the direction
    assert hv.directional_rule(a, b) == hv.INCONCLUSIVE


def test_directional_rule_inconclusive_when_mean_does_not_exceed_sd():
    # all three d_k positive but mean(d) <= sd(d)
    d = [0.01, 0.30, 0.01]
    a = [x + v for x, v in zip([0.50, 0.50, 0.50], d)]
    b = [0.50, 0.50, 0.50]
    assert statistics.mean(d) <= hv.sample_sd(d)
    assert hv.directional_rule(a, b) == hv.INCONCLUSIVE


def test_directional_rule_inconclusive_on_ties_and_missing():
    assert hv.directional_rule([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == \
        hv.INCONCLUSIVE
    assert hv.directional_rule([0.5, None, 0.7], [0.4, 0.5, 0.6]) == \
        hv.INCONCLUSIVE
    assert hv.directional_rule([0.5, 0.6], [0.4, 0.5, 0.6]) == \
        hv.INCONCLUSIVE   # unequal arm lengths


# ---- H3 secondary gap rule -----------------------------------------------

def test_secondary_gap_rule_supported():
    assert hv.secondary_gap_rule([0.15, 0.15, 0.23]) == hv.SUPPORTED
    assert hv.secondary_gap_rule([0.2, 0.2, 0.2]) == hv.SUPPORTED


def test_secondary_gap_rule_not_supported():
    d = [-0.10, -0.12, -0.08]
    assert all(v < 0 for v in d) and statistics.mean(d) < 0
    assert abs(statistics.mean(d)) > hv.sample_sd(d)
    assert hv.secondary_gap_rule(d) == hv.NOT_SUPPORTED


def test_secondary_gap_rule_inconclusive():
    assert hv.secondary_gap_rule([0.2, 0.1, -0.05]) == hv.INCONCLUSIVE
    d = [0.01, 0.30, 0.01]       # all positive but mean(d) <= sd(d)
    assert hv.secondary_gap_rule(d) == hv.INCONCLUSIVE
    assert hv.secondary_gap_rule([None, 0.2, 0.2]) == hv.INCONCLUSIVE


def test_secondary_gap_rule_agrees_with_global_rule_against_zero_arm():
    for d in ([0.2, 0.2, 0.2], [-0.1, -0.3, -0.2], [0.2, -0.1, 0.0],
              [0.01, 0.30, 0.01], [-0.01, -0.001, -0.02]):
        assert hv.secondary_gap_rule(d) == hv.directional_rule(d, [0.0] * 3)


# ---- H4 clause combination -----------------------------------------------

def test_h4_verdict_all_outcome_combinations():
    S, N, I = hv.SUPPORTED, hv.NOT_SUPPORTED, hv.INCONCLUSIVE
    assert hv.h4_verdict(S, S) == S
    assert hv.h4_verdict(N, S) == N       # opposite of either clause -> N
    assert hv.h4_verdict(S, N) == N
    assert hv.h4_verdict(N, N) == N
    assert hv.h4_verdict(S, I) == I       # expected direction in both? no
    assert hv.h4_verdict(I, S) == I
    assert hv.h4_verdict(I, I) == I
    assert hv.h4_verdict(I, N) == N


# ---- H1 equivalence band -------------------------------------------------

def test_equivalence_band_inside_and_outside():
    assert hv.equivalence_band(0.50, 0.48, 0.05) is True    # |0.02| < sd(d)
    assert hv.equivalence_band(0.50, 0.40, 0.05) is False   # |0.10| >= sd(d)
    assert hv.equivalence_band(None, 0.5, 0.1) is None
    assert hv.equivalence_band(0.5, None, 0.1) is None
    assert hv.equivalence_band(0.5, 0.4, None) is None


# ---- budget rule -----------------------------------------------------------

def test_budget_block_all_arms_finished_is_eligible():
    budget = hv.budget_block(synth_metrics(), ["CF-Full", "RecurrentCNN"], SEEDS)
    assert budget["eligible"] is True
    assert budget["planned_epochs"] == 40
    assert budget["problems"] == []
    for tag in ("CF-Full_seed0", "CF-Full_seed2", "RecurrentCNN_seed1"):
        assert budget["arms"][tag] == {"epochs_done": 40,
                                       "planned_epochs": 40,
                                       "stopped_early": False}


def test_budget_block_blocks_stopped_early_run():
    metrics = synth_metrics()
    metrics["runs"]["CF-Full_seed1"]["stopped_early"] = True
    budget = hv.budget_block(metrics, ["CF-Full"], SEEDS)
    assert budget["eligible"] is False
    assert "CF-Full_seed1" in " ".join(budget["problems"])
    # epochs_done still reported for each arm
    assert budget["arms"]["CF-Full_seed1"]["epochs_done"] == 40


def test_budget_block_blocks_early_stopped_epochs():
    metrics = synth_metrics()
    metrics["runs"]["CF-NoFrag_seed2"]["epochs_done"] = 25
    budget = hv.budget_block(metrics, ["CF-NoFrag"], SEEDS)
    assert budget["eligible"] is False
    assert "CF-NoFrag_seed2" in " ".join(budget["problems"])
    assert budget["arms"]["CF-NoFrag_seed2"]["epochs_done"] == 25


def test_budget_block_names_missing_run_and_skipped_status():
    metrics = synth_metrics()
    del metrics["runs"]["RecurrentCNN_seed2"]
    metrics["runs"]["RecurrentCNN_seed1"]["status"] = "skipped: budget"
    budget = hv.budget_block(metrics, ["RecurrentCNN"], SEEDS)
    assert budget["eligible"] is False
    problems = " ".join(budget["problems"])
    assert "missing run record RecurrentCNN_seed2" in problems
    assert "RecurrentCNN_seed1" in problems


def test_budget_block_without_planned_epochs():
    metrics = synth_metrics(hyperparams={})
    budget = hv.budget_block(metrics, ["CF-Full"], SEEDS)
    assert budget["eligible"] is False
    assert "planned epochs" in budget["problems"][0]


# ---- data extraction ------------------------------------------------------

def test_strict_at_uses_stored_sweep_and_recorded_test_fallback():
    metrics = synth_metrics()
    run = metrics["runs"]["CF-Full_seed0"]
    del run["sweeps"]["test"]["8"]
    # recorded final test summary stands in for the T = T_train sweep entry
    assert hv.strict_at(metrics, "CF-Full", 0, 8) == SC_A8[0]
    assert hv.strict_at(metrics, "CF-Full", 0, 16) == SC_A16[0]
    try:
        hv.strict_at(metrics, "CF-Full", 0, 12)      # horizon never recorded
    except hv.MissingData as exc:
        assert "sweep at T=12" in exc.name
    else:
        raise AssertionError("expected MissingData")


def test_non_dict_sweeps_record_is_named_missing_data():
    # a corrupt `sweeps` field (not a JSON object) must take the same
    # MissingData path as an absent sweep entry, not crash the assembly
    metrics = synth_metrics()
    metrics["runs"]["CF-NoFrag_seed1"]["sweeps"] = ["corrupt"]
    try:
        hv.strict_at(metrics, "CF-NoFrag", 1, 16)
    except hv.MissingData as exc:
        assert "CF-NoFrag_seed1 test sweeps record" in exc.name
    else:
        raise AssertionError("expected MissingData")
    # the descriptive sweep table tolerates the same corruption
    assert hv.sweep_table(metrics, ["CF-NoFrag"], SEEDS)["CF-NoFrag"]["1"] \
        is None
    payload = hv.build_verdicts(metrics, None, None)
    hyp = payload["hypotheses"]["H3"]
    assert hyp["verdict"] == hv.INCONCLUSIVE
    assert hyp["verdict_secondary"] == hv.INCONCLUSIVE    # seed 1 gaps lost
    assert "CF-NoFrag_seed1 test sweeps record" in " ".join(hyp["caveats"])
    assert hyp["numbers"]["primary"]["arms"]["B"]["per_seed"]["1"] is None


def test_settling_at_holds_step_at_index_t_minus_one():
    metrics = synth_metrics()
    assert hv.settling_at(metrics, "CF-Full", 0, 8) == CF_SETTLE[0][0]
    assert hv.settling_at(metrics, "CF-Full", 0, 16) == CF_SETTLE[0][1]
    assert hv.settling_at(metrics, "RecurrentCNN", 1, 16) == RC_SETTLE[1][1]


def test_settling_at_missing_or_short_curve_is_named():
    metrics = synth_metrics()
    metrics["runs"]["RecurrentCNN_seed0"]["settling"] = [0.1] * 10
    del metrics["runs"]["RecurrentCNN_seed1"]["settling"]
    for seed in (0, 1):
        try:
            hv.settling_at(metrics, "RecurrentCNN", seed, 16)
        except hv.MissingData as exc:
            assert "RecurrentCNN_seed%d settling curve at step 16" % seed \
                == exc.name
        else:
            raise AssertionError("expected MissingData")


# ---- full payload assembly ----------------------------------------------

def test_build_verdicts_expected_verdicts_on_synthetic_data():
    payload = hv.build_verdicts(synth_metrics(), synth_valt(),
                                synth_followups())
    hyps = payload["hypotheses"]
    assert list(hyps) == ["H1", "H2", "H3", "H4"]

    assert hyps["H1"]["verdict"] == hv.NOT_SUPPORTED
    # CF-Full < RecurrentCNN on every seed with mean |d| >> sd(d)
    d = [a - b for a, b in zip(SC_A8, SC_R8)]
    assert all(v < 0 for v in d)
    assert hyps["H1"]["numbers"]["mean_paired_difference"] == \
        statistics.mean(d)

    assert hyps["H2"]["verdict"] == hv.SUPPORTED
    assert hyps["H3"]["verdict"] == hv.SUPPORTED
    assert hyps["H3"]["verdict_secondary"] == hv.SUPPORTED
    assert hyps["H4"]["verdict"] == hv.SUPPORTED
    assert hyps["H4"]["clause_verdicts"] == {"a": hv.SUPPORTED,
                                             "b": hv.SUPPORTED}


def test_verdict_payload_structure_has_all_required_fields():
    payload = hv.build_verdicts(synth_metrics(), None, None)
    for hid, hyp in payload["hypotheses"].items():
        assert hyp["id"] == hid
        assert hyp["statement"]
        assert "SUPPORTED only if" in hyp["rule"]        # rule quoted
        assert "Budget rule" in hyp["rule"] or \
            "budget" in hyp                              # budget reported
        assert hyp["caveats"]
        if hid == "H4":  # clause-wise budget: (a) CF-Full arms, (b) RCNN arms
            assert hyp["budget"]["a"]["arms"]
            assert hyp["budget"]["b"]["arms"]
        else:
            assert hyp["budget"]["arms"]                # per-arm epochs
        if hid in ("H1", "H2"):
            numbers = hyp["numbers"]
            assert numbers["n"] == 3
            assert numbers["epochs_done"]["CF-Full_seed0"] == 40
            assert set(numbers["paired_differences"]) == {"0", "1", "2"}
            assert "sample_sd_paired_differences" in numbers
            assert numbers["mean_A"] is not None
    assert payload["rule_budget"].startswith("Budget rule (fixed before")
    assert payload["n_seeds"] == 3
    assert payload["post_hoc_context"]["valT_selection"]["available"] is False


def test_missing_sweep_makes_h3_inconclusive_and_is_named():
    metrics = synth_metrics()
    del metrics["runs"]["CF-NoFrag_seed0"]["sweeps"]["test"]["16"]
    payload = hv.build_verdicts(metrics, None, None)
    hyp = payload["hypotheses"]["H3"]
    assert hyp["verdict"] == hv.INCONCLUSIVE
    assert hyp["verdict_secondary"] == hv.INCONCLUSIVE       # seed 0 gap lost
    joined = " ".join(hyp["caveats"])
    assert "CF-NoFrag_seed0 test sweep at T=16" in joined
    assert hyp["numbers"]["primary"]["arms"]["B"]["per_seed"]["0"] is None


def test_missing_settling_makes_h4_inconclusive_and_is_named():
    metrics = synth_metrics()
    del metrics["runs"]["CF-Full_seed1"]["settling"]
    payload = hv.build_verdicts(metrics, None, None)
    hyp = payload["hypotheses"]["H4"]
    assert hyp["verdict"] == hv.INCONCLUSIVE
    assert hyp["clause_verdicts"]["a"] == hv.INCONCLUSIVE
    assert "CF-Full_seed1 settling curve at step 8" in \
        " ".join(hyp["caveats"])
    assert hyp["numbers"]["clause_a_flux"]["arms"]["A"]["per_seed"]["1"] \
        is None


def test_budget_rule_blocks_verdict_even_when_rule_would_decide():
    metrics = synth_metrics()
    metrics["runs"]["CF-Full_seed1"]["epochs_done"] = 25
    metrics["runs"]["CF-Full_seed1"]["stopped_early"] = True
    payload = hv.build_verdicts(metrics, None, None)
    hyps = payload["hypotheses"]
    # every comparison with a CF-Full arm is budget-blocked
    assert hyps["H1"]["verdict"] == hv.INCONCLUSIVE
    assert hyps["H2"]["verdict"] == hv.INCONCLUSIVE
    assert hyps["H3"]["verdict"] == hv.INCONCLUSIVE
    assert hyps["H3"]["verdict_secondary"] == hv.INCONCLUSIVE
    assert hyps["H4"]["verdict"] == hv.INCONCLUSIVE
    # ... and epochs_done is still reported for each arm
    assert hyps["H2"]["numbers"]["epochs_done"]["CF-Full_seed1"] == 25
    assert "CF-Full_seed1 did not finish its planned epochs" in \
        " ".join(hyps["H2"]["caveats"])


def test_post_hoc_context_never_changes_verdicts():
    without = hv.build_verdicts(synth_metrics(), None, None)
    with_ctx = hv.build_verdicts(synth_metrics(), synth_valt(),
                                 synth_followups())
    for hid in ("H1", "H2", "H3", "H4"):
        a, b = without["hypotheses"][hid], with_ctx["hypotheses"][hid]
        assert a["verdict"] == b["verdict"]
        assert a["numbers"] == b["numbers"]
    ctx = with_ctx["post_hoc_context"]
    assert ctx["valT_selection"]["per_model"]["CF-Full"]["T_star_by_seed"] \
        == {"0": 12, "1": 12, "2": 12}
    assert "E1_CF-Full" in " ".join(
        with_ctx["hypotheses"]["H2"]["caveats"])


def test_h1_equivalence_band_caveat():
    metrics = synth_metrics()
    payload = hv.build_verdicts(metrics, None, None)
    joined = " ".join(payload["hypotheses"]["H1"]["caveats"])
    assert "outside the pre-registered band" in joined
    assert "|mean(CF-Full) - mean(RecurrentCNN)|" in joined


def synth_invariants(nofrag_fail_seed=None, models=("CF-Full", "CF-NoFrag",
                                                    "RecurrentCNN")):
    """trained_invariants.json as scripts/check_trained_invariants.py writes it."""
    runs = {}
    for model in models:
        for seed in SEEDS:
            if model == "RecurrentCNN":
                checks = {"logits_finite": True, "logits_abs_max": 3.2}
            else:
                checks = {"min_c1": -1e-9, "min_c2": 0.0,
                          "max_relative_mass_drift": 2e-7,
                          "positivity_ok": True, "mass_drift_ok": True}
                if model == "CF-NoFrag":
                    checks.update({"split_flux_all_zero": True,
                                   "split_flux_max_abs": 0.0,
                                   "split_head_grad_all_zero": True,
                                   "split_head_grad_max_abs": 0.0})
            status = "PASS"
            if model == "CF-NoFrag" and seed == nofrag_fail_seed:
                status = "FAIL"
                checks["split_flux_all_zero"] = False
                checks["split_flux_max_abs"] = 0.25
            runs["%s_seed%d" % (model, seed)] = {
                "model": model, "seed": seed, "status": status,
                "checkpoint": "%s_seed%d.pt" % (model, seed),
                "checks": checks}
    return {"config": {"T": 64, "n_mazes": 64, "seeds": list(SEEDS)},
            "tolerances": {"positivity_min_c": -1e-7,
                           "relative_mass_drift_max": 1e-5},
            "runs": runs,
            "summary": {"n_runs": len(runs),
                        "n_pass": len(runs) - (1 if nofrag_fail_seed is not None else 0),
                        "n_fail": 1 if nofrag_fail_seed is not None else 0,
                        "n_missing": 0,
                        "status": "PASS" if nofrag_fail_seed is None else "FAIL"}}


# ---- guardrail / precondition evidence (trained_invariants.json) ----------

def test_h2_h4_caveats_state_guardrail_pass_when_record_present():
    payload = hv.build_verdicts(synth_metrics(), None, None,
                                invariants=synth_invariants())
    for hid in ("H2", "H4"):
        joined = " ".join(payload["hypotheses"][hid]["caveats"])
        assert "Guardrail evidence (confound (iv), trained_invariants.json)" \
            in joined
        assert "WAS run" in joined
        assert "PASS for all 3 runs (seeds 0, 1, 2)" in joined
        assert "max relative mass drift" in joined
        assert "min c1" in joined


def test_h3_caveat_states_precondition_pass_when_record_present():
    payload = hv.build_verdicts(synth_metrics(), None, None,
                                invariants=synth_invariants())
    joined = " ".join(payload["hypotheses"]["H3"]["caveats"])
    assert "Precondition evidence (H3 Metric, trained_invariants.json)" in joined
    assert "the split flux is exactly 0 at every step" in joined
    assert "zero gradient (one backward pass)" in joined
    assert "PASS for all 3 runs (seeds 0, 1, 2)" in joined
    assert "max |split flux| = 0" in joined


def test_invariant_caveats_say_not_yet_run_when_record_absent():
    payload = hv.build_verdicts(synth_metrics(), None, None)
    assert payload["trained_invariants"]["available"] is False
    for hid in ("H2", "H3", "H4"):
        joined = " ".join(payload["hypotheses"][hid]["caveats"])
        assert "not yet run" in joined
        assert "unevidenced" in joined or "no record" in joined


def test_invariant_caveats_say_not_yet_run_when_model_record_missing():
    invariants = synth_invariants(models=("RecurrentCNN",))
    payload = hv.build_verdicts(synth_metrics(), None, None,
                                 invariants=invariants)
    for hid in ("H2", "H4"):
        joined = " ".join(payload["hypotheses"][hid]["caveats"])
        assert "not yet run" in joined
        assert "no CF-Full record" in joined
    joined = " ".join(payload["hypotheses"]["H3"]["caveats"])
    assert "no CF-NoFrag record" in joined


def test_invariant_fail_is_reported_and_never_changes_verdicts():
    clean = hv.build_verdicts(synth_metrics(), None, None,
                              invariants=synth_invariants())
    failed = hv.build_verdicts(synth_metrics(), None, None,
                               invariants=synth_invariants(
                                   nofrag_fail_seed=1))
    joined = " ".join(failed["hypotheses"]["H3"]["caveats"])
    assert "FAILED" in joined
    assert "CF-NoFrag seed 1" in joined
    assert "precondition for the arm is NOT met" in joined
    for hid in ("H1", "H2", "H3", "H4"):
        a, b = (clean["hypotheses"][hid], failed["hypotheses"][hid])
        assert a["verdict"] == b["verdict"]       # evidence only, no gate
    assert failed["trained_invariants"]["available"] is True
    assert failed["trained_invariants"]["run_status"]["CF-NoFrag_seed1"] == "FAIL"


def test_t64_guardrail_fail_is_reported_in_h2_and_h4():
    invariants = synth_invariants()
    for seed in SEEDS:                     # all CF-Full runs fail the re-check
        rec = invariants["runs"]["CF-Full_seed%d" % seed]
        rec["status"] = "FAIL"
    payload = hv.build_verdicts(synth_metrics(), None, None,
                                invariants=invariants)
    for hid in ("H2", "H4"):
        joined = " ".join(payload["hypotheses"][hid]["caveats"])
        assert "FAILED" in joined
        assert "guardrail is NOT met" in joined
        assert "CF-Full seed 0" in joined


# ---- H4 clause-wise budget rule -------------------------------------------

def test_h4_clause_a_is_gated_on_cf_full_budget_only():
    # the doc's budget rule gates a comparison on "every arm in that
    # comparison"; clause (a)'s arms are the CF-Full checkpoints only, so a
    # budget-stopped RecurrentCNN must NOT block clause (a)
    metrics = synth_metrics()
    metrics["runs"]["RecurrentCNN_seed1"]["stopped_early"] = True
    payload = hv.build_verdicts(metrics, None, None)
    hyp = payload["hypotheses"]["H4"]
    assert hyp["clause_verdicts"]["a"] == hv.SUPPORTED
    assert hyp["clause_verdicts"]["b"] == hv.INCONCLUSIVE
    assert hyp["verdict"] == hv.INCONCLUSIVE
    joined = " ".join(hyp["caveats"])
    assert "Budget rule (clause b" in joined
    assert "RecurrentCNN_seed1 did not finish its planned epochs" in joined
    assert "clause a" not in joined          # clause (a) untouched


def test_h4_clause_b_is_gated_on_rcnn_budget_only():
    metrics = synth_metrics()
    metrics["runs"]["CF-Full_seed2"]["epochs_done"] = 30
    metrics["runs"]["CF-Full_seed2"]["stopped_early"] = True
    payload = hv.build_verdicts(metrics, None, None)
    hyp = payload["hypotheses"]["H4"]
    assert hyp["clause_verdicts"]["a"] == hv.INCONCLUSIVE
    assert hyp["clause_verdicts"]["b"] == hv.SUPPORTED
    assert hyp["verdict"] == hv.INCONCLUSIVE
    joined = " ".join(hyp["caveats"])
    assert "Budget rule (clause a" in joined
    assert "CF-Full_seed2 did not finish its planned epochs" in joined


def test_h4_budget_field_reports_per_clause_arms():
    payload = hv.build_verdicts(synth_metrics(), None, None)
    budget = payload["hypotheses"]["H4"]["budget"]
    assert set(budget) == {"a", "b"}
    assert set(budget["a"]["arms"]) == \
        {"CF-Full_seed%d" % s for s in SEEDS}
    assert set(budget["b"]["arms"]) == \
        {"RecurrentCNN_seed%d" % s for s in SEEDS}


# ---- CLI writer -----------------------------------------------------------

def test_main_writes_h_verdicts_json(tmp_path, capsys):
    results = tmp_path / "week2_results"
    results.mkdir()
    (results / "week2_metrics.json").write_text(
        json.dumps(synth_metrics()), encoding="utf-8")
    (results / "valT_selection.json").write_text(
        json.dumps(synth_valt()), encoding="utf-8")
    (results / "followups").mkdir()
    (results / "followups" / "results.json").write_text(
        json.dumps(synth_followups()), encoding="utf-8")

    assert hv.main(["--results", str(results)]) == 0

    out = json.loads((results / "h_verdicts.json").read_text(encoding="utf-8"))
    assert out["hypotheses"]["H1"]["verdict"] == hv.NOT_SUPPORTED
    assert out["hypotheses"]["H4"]["verdict"] == hv.SUPPORTED
    printed = capsys.readouterr().out
    assert "H1  NOT SUPPORTED" in printed
    assert "verdicts written to" in printed


def test_main_reads_trained_invariants_json(tmp_path, capsys):
    results = tmp_path / "week2_results"
    results.mkdir()
    (results / "week2_metrics.json").write_text(
        json.dumps(synth_metrics()), encoding="utf-8")
    (results / "trained_invariants.json").write_text(
        json.dumps(synth_invariants()), encoding="utf-8")

    assert hv.main(["--results", str(results)]) == 0

    out = json.loads((results / "h_verdicts.json").read_text(encoding="utf-8"))
    assert out["trained_invariants"]["available"] is True
    assert out["trained_invariants"]["run_status"]["CF-NoFrag_seed0"] == "PASS"
    assert out["source_files"]["trained_invariants"].endswith(
        "trained_invariants.json")
    for hid in ("H2", "H3", "H4"):
        joined = " ".join(out["hypotheses"][hid]["caveats"])
        assert "PASS for all 3 runs" in joined


def test_main_exits_nonzero_without_metrics(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert hv.main(["--results", str(empty)]) == 1
    assert "no" in capsys.readouterr().out
