"""Week-2 hypothesis verdicts for the curriculum-maze experiment.

Implements EXACTLY the hypotheses, arms, metrics, budget rule and decision
rules fixed in docs/hypotheses.md.  Every rule applied here is quoted
verbatim from that file in the module constants below and copied into the
output JSON, so the verdict file carries the rules it was judged by.

Inputs (plain JSON only; no torch import, CPU safe):
  week2_results/week2_metrics.json          - the main run: RecurrentCNN,
                                              CF-Full, CF-NoFrag arms
  week2_results/wallaware/week2_metrics.json - the wall-aware arm of H3
                                              (CF-WallAware, same protocol)
  week2_results/valT_selection.json         - post-hoc val-horizon selection (context only)
  week2_results/followups/results.json     - post-hoc follow-ups (context only)
  week2_results/trained_invariants.json    - optional guardrail / precondition
                                             record written by
                                             scripts/check_trained_invariants.py;
                                             quoted in the H2/H4 caveats
                                             ("not yet run" when absent)

Output:
  week2_results/h_verdicts.json   (per hypothesis: id, statement, rule, numbers
  used - per seed values, paired differences, mean, sample sd, n, epochs_done
  -, verdict, caveats).  Missing data yields INCONCLUSIVE naming what is
  missing.  The four hypothesis verdicts are H1-H4 of docs/hypotheses.md;
  the packet-derived settling hypothesis (H-A4) is preserved there in the
  appendix and carried here as a supplementary verdict.
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT SUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"

OUT_NAME = "h_verdicts.json"
INVARIANTS_NAME = "trained_invariants.json"
WALLAWARE_DIRNAME = "wallaware"
WALLAWARE_METRICS_NAME = "week2_metrics.json"

# ---------------------------------------------------------------------------
# Rules fixed in docs/hypotheses.md before any Week-2 number was inspected.
# All quotes below are verbatim from that file.
# ---------------------------------------------------------------------------

GLOBAL_RULE = (
    'Every comparison below uses n = 3 seeds (0, 1, 2), paired by seed index '
    'because both arms are trained with the same seeds and the same data '
    'split. For a directional claim "A > B" computed from seed-level values '
    'a_0, a_1, a_2 and b_0, b_1, b_2:\n'
    '- SUPPORTED only if mean(A) > mean(B), a_k > b_k for every seed k, and '
    'the mean paired difference mean(A) \u2212 mean(B) exceeds sd(d), where '
    'd_k = a_k \u2212 b_k are the per-seed paired differences and sd(d) is '
    'their sample standard deviation (n = 3, ddof = 1).\n'
    '- NOT SUPPORTED if the opposite direction (B > A) meets the same three '
    'conditions.\n'
    '- INCONCLUSIVE otherwise.'
)

BUDGET_RULE = (
    'Budget rule (fixed before verdicts): every arm of the Week-2 run is '
    'trained under one shared wall-clock budget (scripts/run_week2.py, '
    '--budget-minutes), and a run can be stopped early (epochs_done < '
    'planned, stopped_early). A directional verdict (SUPPORTED / NOT '
    'SUPPORTED) for any comparison below is computed only if every arm in '
    'that comparison finished its planned epochs (epochs_done == planned, '
    'not stopped_early); otherwise the comparison is INCONCLUSIVE, with '
    'epochs_done reported for each arm. An arm that merely received more '
    'training time than its opponent \u2014 or lost for lack of it \u2014 '
    'must not carry the verdict; early-stopped runs are reported '
    'descriptively only.'
)

MULTIPLICITY = (
    'Multiplicity: the rule yields seven directional comparisons in one '
    'family across the hypotheses \u2014 H1, H2 primary, H2 secondary, H3, '
    'H4 \u2014 plus the two clauses of the supplementary settling hypothesis '
    'preserved in the appendix. No alpha correction is applied: this is a '
    'deterministic descriptive decision rule on summary statistics, fixed in '
    'advance, not an inferential test with controlled error rates.'
)

SCOPE_NOTE = (
    'Scope (docs/hypotheses.md): "every curriculum split used this week '
    'consists of perfect mazes, so strict = exact is asserted by '
    'construction and the distinction is moot for all verdicts below."'
)

# H1 (quotes verbatim)
H1_STATEMENT = (
    'On the held-out level-5 test set (1,000 mazes), CF-Full (10,177 '
    'parameters) trained with the shared protocol reaches strict test '
    'accuracy at least as high as the parameter-matched RecurrentCNN at the '
    'training horizon T = 8. Directional claim for the decision rule: '
    'CF-Full > RecurrentCNN; parity within noise is INCONCLUSIVE under the '
    'rule.'
)
H1_RULE = (
    'As the global rule, arms = per-seed strict test accuracy of CF-Full and '
    'RecurrentCNN at T = 8.'
)
H1_EQUIVALENCE = (
    'The wording "comparable at matched budget" is pre-registered as '
    'descriptive only, and permitted only inside an explicit equivalence '
    'band: |mean(CF-Full) \u2212 mean(RecurrentCNN)| < sd(d), the sample sd '
    'of the per-seed paired differences \u2014 the same noise scale the '
    'SUPPORTED threshold uses. Outside that band an INCONCLUSIVE verdict '
    'carries no equivalence reading; even inside it the wording is a '
    'descriptive summary, not an inferential equivalence test (no TOST, no '
    'error-rate claim).'
)
H1_CONFOUNDS = (
    'Known confounds. (i) Ceiling: the supervisor judged the curriculum '
    'dataset too easy (Packet \u00a76.1, p.13: "5x5 is too simple; do '
    '30x30"), so a clean positive difference is hard here; the unseen-grid '
    'probes are the secondary diagnostic. (ii) Matched parameter count does '
    'not mean matched information flow: both models move information at '
    'most 2 cells per step (Packet \u00a712.3, p.28), but their rate/readout '
    'structures differ. (iii) The comparison is CF\'s first run under this '
    'pipeline; disadvantages purely from implementation novelty are not '
    'separated from model quality.'
)
H1_PREREGISTRATION = (
    'Pre-committed: the H1 verdict is computed only from these retrained '
    'Week-2 RecurrentCNN numbers; the Week-1 per-seed numbers (Packet \u00a7'
    '5.1, p.12) are background at most and never feed H1, so the H1 pairing '
    'is within a single training run: CF-Full against RecurrentCNN, both '
    'trained afresh under the same protocol and the same seeds.'
)

# H2 - fragmentation contributes (quotes verbatim)
H2_STATEMENT = (
    'CF-NoFrag (split flux forced to zero, otherwise identical layout) '
    'scores lower strict test accuracy than CF-Full at the training horizon '
    'T = 8, and the gap widens when the frozen models run past the training '
    'horizon (T > 8); the packet\'s own condition is Packet \u00a77.5, '
    'p.17: "If fragmentation matters, CF-NoFrag should lose most when run '
    'past T_train." Primary directional claim for the decision rule: at '
    'T = 8, CF-Full > CF-NoFrag on strict test accuracy. Secondary claim '
    'under the same rule: the gap (CF-Full \u2212 CF-NoFrag) at T = 16 '
    'exceeds the gap at T = 8.'
)
H2_RULE = (
    'Global rule; primary arms = per-seed strict accuracy at T = 8. The '
    'secondary claim is the same global rule applied to the per-seed gap '
    'differences: for each seed k, d_k = (CF-Full_k \u2212 CF-NoFrag_k)'
    '|_{T=16} \u2212 (CF-Full_k \u2212 CF-NoFrag_k)|_{T=8}. The secondary '
    'claim is SUPPORTED if all three d_k are positive, mean(d) > 0, and '
    'mean(d) > sd(d); NOT SUPPORTED if the opposite direction (all d_k '
    'negative, mean(d) < 0, |mean(d)| > sd(d)) meets the same three '
    'conditions; INCONCLUSIVE otherwise.'
)
H2_CONFOUNDS = (
    'Known confounds. (i) Effective-parameter confound, fixed as a caveat '
    'in the packet: CF-NoFrag has 7,857 effective parameters vs CF-Full\'s '
    '10,177 (Packet \u00a712.3, p.28: "CF-NoFrag has fewer effective '
    'parameters (7,857 vs 10,177); mention it whenever CF-Full beats it"), '
    'so even the primary T = 8 advantage of CF-Full could be capacity and '
    'not dynamics; the T-dependence of the gap is the dynamics-specific '
    'signature, and a size-matched NoFrag variant is the clean follow-up. '
    '(ii) The two arms differ by construction in state evolution '
    '(irreversible merging), so the ablation removes a mechanism, not just '
    'a benefit: a bad score for CF-NoFrag past the horizon is consistent '
    'with both hypotheses honestly. (iii) CF-NoFrag is a strict subset of '
    'CF-Full\'s function class only at the dynamics level; the rate '
    'networks see different c2 fields during training, so the trained '
    'models are not clones at T = 8 either.'
)
H2_PRECONDITION = (
    'Precondition for the arm: the invariant test that the split flux is '
    'exactly 0 and the split head receives exactly zero gradient '
    '(docs/hypotheses.md, H2 Metric).'
)

# H3 - wall-aware helps (quotes verbatim)
H3_STATEMENT = (
    'On the held-out level-5 test set at the training horizon T = 8, '
    'CF-WallAware \u2014 a CF-Full that builds the maze\'s wall geometry '
    'into its forward pass instead of learning it from data \u2014 reaches '
    'higher strict test accuracy than CF-Full when both arms are trained '
    'for the full 40 epochs under the identical protocol and the same seeds '
    '(0, 1, 2). Directional claim for the decision rule: CF-WallAware > '
    'CF-Full at T = 8 on strict test accuracy. The frozen-weights T-sweep '
    'and the unseen level-7 / 9 probes are diagnostics only.'
)
H3_ARMS_NOTE = (
    'Arms and sources (docs/hypotheses.md, H3): the CF-WallAware arm lives '
    'in week2_results/' + WALLAWARE_DIRNAME + '/' + WALLAWARE_METRICS_NAME + ' '
    '(produced via week2_results/' + WALLAWARE_DIRNAME + '/run_week2_wrapper.py, '
    'which calls run_week2.main() with --models CF-WallAware and the shared '
    'protocol arguments, changing nothing else); the CF-Full arm is the '
    'main run in week2_results/' + WALLAWARE_METRICS_NAME + '. CF-WallAware = '
    'CF-Full plus wall_aware=True (cf/models/cf.py, MODEL_SPECS); three '
    'changes, all driven by the free-cell channel of the input: (i) initial '
    'mass: the encoder output for the initial c1 field is multiplied by the '
    'free-cell mask, so no mass can start inside a wall; (ii) rate net: the '
    'hidden activations are gated by the free-cell mask, so reaction '
    'fluxes are only driven by free cells; (iii) readout: logits inside '
    'walls are masked to \u221220, so a wall cell can never be predicted as '
    'part of the path. The raw parameter count is identical to CF-Full '
    '(10,177, reported by both run books); the run bookkeeping counts '
    '9,921 effective parameters for CF-WallAware (parameter entries that '
    'receive nonzero gradient, scripts/run_week2.py effective_n_params), '
    'against the packet\'s 10,177 raw and effective for CF-Full (Packet '
    '\u00a712.3, p.28); both counts are reported wherever the comparison '
    'is reported.'
)
H3_RULE = (
    'Global rule, arms = per-seed strict test accuracy of CF-WallAware and '
    'CF-Full at T = 8.'
)
H3_CONFOUNDS = (
    'Known confounds. (i) The comparison changes inductive bias, not '
    'capacity: the wall constraint that CF-Full must learn from data (never '
    'predict a wall cell) is hard-coded in CF-WallAware, and the wall-hit '
    'failure channel is closed by construction (readout mask), so a '
    'strict-accuracy advantage may partly reflect a given constraint rather '
    'than a learned skill; CF-Full\'s wall-hits count measures how much of '
    'the gap that one channel can explain. (ii) Parameters are matched in '
    'raw count but not exactly in effective count (9,921 vs 10,177); both '
    'are reported with every verdict. (iii) The arms were trained in two '
    'invocations of the same runner (the wall-aware arm later, via the '
    'wrapper script, from a later commit of the same repository); the '
    'identical protocol, seeds, data splits, GPU and the budget rule\'s '
    'epochs check are the guard, but a training-environment difference '
    'beyond the recorded protocol cannot be fully excluded. (iv) As '
    'everywhere in this document, n = 3 seeds under the deliberately '
    'conservative paired rule: parity within the seed spread is '
    'INCONCLUSIVE, not evidence of no difference.'
)

# H4 - thinking longer (quotes verbatim)
H4_STATEMENT = (
    'For CF-Full with frozen weights, strict test accuracy at T = 16 (and '
    '24, 32 as secondary points) exceeds strict test accuracy at the '
    'training horizon T_train = 8 on the level-5 test set; the direction is '
    'accuracy that rises when the model runs more iterations than it was '
    'trained with.'
)
H4_RULE = (
    'Global rule applied to A = strict accuracy at T = 16, B = strict '
    'accuracy at T = 8, same 3 checkpoints per model.'
)
H4_CONFOUNDS = (
    'Known confounds. (i) Nothing in training asks the dynamics to settle; '
    'the readout is only ever trained at step 8, so a collapse past the '
    'horizon would be a training-scheme artefact, not evidence against the '
    'mechanism. (ii) The 0.5 threshold on logits is held fixed across T '
    'although the logit scale can drift with steps. (iii) Information speed '
    'is 2 cells per step (Packet \u00a712.3, p.28): on the unseen larger '
    'grids, larger T changes solvability mechanically, not only "quality of '
    'thinking", which is why the level-5 split is primary. (iv) Longer '
    'rollout accumulates float32 error; the mass-drift and positivity '
    'invariants must be re-checked at T = 64 before the sweep is trusted.'
)

# H-A4 - supplementary packet-derived settling hypothesis, preserved in the
# docs/hypotheses.md appendix (quotes verbatim from there).
HA4_STATEMENT = (
    'For CF-Full, the residual flux \u03a3|m \u2212 s| / \u03a3M on test '
    'rollouts falls as the model runs past its training horizon (flux at '
    'step 16 lower than at step 8), i.e. the conserved dynamics approach '
    'the chemical equilibrium where \u00bd c1 \u03c3(merge) = c2 '
    '\u03c3(split); and in the same step range the RecurrentCNN\'s relative '
    'state change is larger at step 16 than at step 8 \u2014 that specific '
    'ordering is what the rule clause tests; a flat curve is INCONCLUSIVE, '
    'not evidence for the clause (the RecurrentCNN\'s state is not a '
    'conserved quantity). Primary clause for the decision rule: CF-Full '
    'residual flux at step 16 < CF-Full residual flux at step 8; contrast '
    'clause: RecurrentCNN relative state change at step 16 > at step 8.'
)
HA4_RULE = (
    'Global rule applied clause-wise: (a) flux at t = 16 vs t = 8 for '
    'CF-Full; (b) state change at t = 16 vs t = 8 for RecurrentCNN. '
    'SUPPORTED only if the expected direction holds in both clauses under '
    'the rule; NOT SUPPORTED if the opposite of either clause meets the '
    'rule; otherwise INCONCLUSIVE.'
)
HA4_CONFOUNDS = (
    'Known confounds. (i) Low flux measures chemical rest, not answer '
    'quality: the state can settle onto a wrong, "all committed" '
    'configuration, so H-A4 is a mechanism claim and must be read jointly '
    'with the accuracy hypotheses \u2014 fragmentation (H2) and thinking '
    'longer (H4) above \u2014 never alone. (ii) The two settling quantities '
    'are different objects with different scales (a dimensionless '
    'normalized flux vs an unbounded state\'s relative change); only the '
    'within-model clause comparison is quantitative, never the '
    'cross-model magnitude. (iii) As with H4, training never asked for '
    'settling, so a flat or rising flux past the horizon is a live '
    'possibility and the verdict is expected to be negative-or-inconclusive '
    'without the later remedies planned in the packet (random T, damping, '
    'settling penalty). (iv) Flux is recorded on the same rollout as the '
    'accuracy sweep, so any bug in the step recording silently corrupts '
    'both thinking-longer (H4) and H-A4; the invariant re-checks at T = 64 '
    'are the guardrail.'
)
HA4_METRIC_NOTE = (
    'H-A4 metric note: both settling quantities are computed on one fixed '
    '64-maze subset of the level-5 test set at each step of a T = 32 '
    'rollout (cf/train.py settling_curve); "the value is 0 if \u2016h_'
    '{t\u22121}\u2016 = 0; list index t\u22121 holds step t".'
)

CF_FULL = "CF-Full"
CF_NOFRAG = "CF-NoFrag"
CF_WALLAWARE = "CF-WallAware"
RCNN = "RecurrentCNN"


# ---------------------------------------------------------------------------
# Decision functions (pure; every rule quoted)
# ---------------------------------------------------------------------------

class MissingData(Exception):
    """Raised when a required number is absent; the verdict becomes
    INCONCLUSIVE and the missing item is named in the caveats."""

    def __init__(self, name):
        self.name = name
        super().__init__("missing data: %s" % name)


def sample_sd(values):
    """sd(d): the sample standard deviation of the per-seed paired
    differences (n = 3, ddof = 1), as fixed by the global rule: "sd(d) is
    their sample standard deviation (n = 3, ddof = 1)"."""
    if values is None or len(values) < 2:
        return 0.0
    return statistics.stdev(values)          # sample std: ddof = 1


def _mean(values):
    if not values or any(v is None for v in values):
        return None
    return statistics.mean(values)


def directional_rule(a, b):
    """Global decision rule, verdict for the directional claim "A > B".

    Quote (docs/hypotheses.md, Global decision rule): "SUPPORTED only if
    mean(A) > mean(B), a_k > b_k for every seed k, and the mean paired
    difference mean(A) - mean(B) exceeds sd(d), where d_k = a_k - b_k are
    the per-seed paired differences and sd(d) is their sample standard
    deviation (n = 3, ddof = 1). NOT SUPPORTED if the opposite direction
    (B > A) meets the same three conditions. INCONCLUSIVE otherwise."

    a, b: per-seed values of arms A and B, paired by seed index. A None in
    either arm marks missing data and the verdict is INCONCLUSIVE.
    """
    if (not a or len(a) != len(b)
            or any(v is None for v in list(a) + list(b))):
        return INCONCLUSIVE
    diffs = [x - y for x, y in zip(a, b)]
    mean_diff = statistics.mean(diffs)
    sd = sample_sd(diffs)
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    supported = (mean_a > mean_b
                and all(x > y for x, y in zip(a, b))
                and mean_diff > sd)
    opposite = (mean_b > mean_a
                and all(y > x for x, y in zip(a, b))
                and -mean_diff > sd)
    if supported:
        return SUPPORTED
    if opposite:
        return NOT_SUPPORTED
    return INCONCLUSIVE


def secondary_gap_rule(d):
    """H2 secondary claim: the gap (CF-Full - CF-NoFrag) at T = 16 exceeds
    the gap at T = 8.

    Quote (docs/hypotheses.md, H2 Decision rule): "The secondary claim is
    SUPPORTED if all three d_k are positive, mean(d) > 0, and mean(d) >
    sd(d); NOT SUPPORTED if the opposite direction (all d_k negative,
    mean(d) < 0, |mean(d)| > sd(d)) meets the same three conditions;
    INCONCLUSIVE otherwise."

    d: per-seed gap differences (CF-Full - CF-NoFrag)|T=16 minus the same
    gap at T=8. None entries mark missing data -> INCONCLUSIVE.
    """
    if not d or any(v is None for v in d):
        return INCONCLUSIVE
    mean_d = statistics.mean(d)
    sd = sample_sd(d)
    if all(v > 0 for v in d) and mean_d > 0 and mean_d > sd:
        return SUPPORTED
    if all(v < 0 for v in d) and mean_d < 0 and -mean_d > sd:
        return NOT_SUPPORTED
    return INCONCLUSIVE


def ha4_verdict(clause_a, clause_b):
    """H-A4 clause-wise combination rule.

    Quote (docs/hypotheses.md, H-A4 Decision rule): "SUPPORTED only if the
    expected direction holds in both clauses under the rule; NOT SUPPORTED
    if the opposite of either clause meets the rule; otherwise INCONCLUSIVE."

    clause_a / clause_b: the verdicts of the two expected-direction clauses
    (a) CF-Full flux at step 16 < at step 8 and (b) RecurrentCNN state
    change at step 16 > at step 8, each judged by the global rule.
    """
    if clause_a == SUPPORTED and clause_b == SUPPORTED:
        return SUPPORTED
    if clause_a == NOT_SUPPORTED or clause_b == NOT_SUPPORTED:
        return NOT_SUPPORTED
    return INCONCLUSIVE


def equivalence_band(mean_a, mean_b, sd_diff):
    """H1 pre-registered descriptive equivalence band.

    Quote (docs/hypotheses.md, H1 Statement): "permitted only inside an
    explicit equivalence band: |mean(CF-Full) - mean(RecurrentCNN)| <
    sd(d), the sample sd of the per-seed paired differences - the same
    noise scale the SUPPORTED threshold uses."

    Returns True inside the band, False outside, None when the numbers are
    missing.
    """
    if mean_a is None or mean_b is None or sd_diff is None:
        return None
    return abs(mean_a - mean_b) < sd_diff


# ---------------------------------------------------------------------------
# Extraction from week2_metrics.json (main run) and the wall-aware run file
# ---------------------------------------------------------------------------

def run_of(metrics, model, seed):
    """Run record tagged '<model>_seed<k>'; a missing/incomplete record is
    named MissingData so the verdict can become INCONCLUSIVE while naming it."""
    tag = "%s_seed%d" % (model, seed)
    run = (metrics.get("runs") or {}).get(tag)
    if not isinstance(run, dict):
        raise MissingData("run record %s" % tag)
    return tag, run


def planned_epochs_of(metrics):
    """Planned epochs of a run = hyperparams['epochs'] (the shared protocol:
    Adam 1e-3, ..., 40 epochs)."""
    hp = metrics.get("hyperparams") or {}
    if "epochs" not in hp:
        raise MissingData("planned epochs (hyperparams.epochs)")
    return hp["epochs"]


def arms_view(metrics, wallaware):
    """Extraction view over both run invocations: the CF-WallAware runs come
    from the wall-aware metrics file, every other arm from the main run
    (docs/hypotheses.md, H3 "Comparison and data needed")."""
    if not isinstance(wallaware, dict):
        return metrics
    runs = dict(metrics.get("runs") or {})
    runs.update(wallaware.get("runs") or {})
    view = dict(metrics)
    view["runs"] = runs
    return view


def _stored_sweep(run, split, tag):
    """{int(T): summary} from a run record, ignoring non-horizon keys.

    A `sweeps` field that is not a JSON object is corrupt data: named
    MissingData so the strict_at path becomes INCONCLUSIVE naming the run."""
    sweeps = run.get("sweeps")
    if not isinstance(sweeps, dict):
        raise MissingData("%s %s sweeps record" % (tag, split))
    sw = sweeps.get(split)
    if not isinstance(sw, dict):
        sw = {}
    out = {}
    for key, val in sw.items():
        try:
            out[int(key)] = val
        except (TypeError, ValueError):
            continue
    return out


def strict_at(metrics, model, seed, T, split="test"):
    """Strict accuracy on `split` at horizon T for one run: the stored sweep
    entry, or the recorded final `test` summary when T == T_train."""
    tag, run = run_of(metrics, model, seed)
    if run.get("status") != "done":
        raise MissingData("%s (status %s)" % (tag, run.get("status")))
    stored = _stored_sweep(run, split, tag)
    if T in stored and isinstance(stored[T], dict) and "strict" in stored[T]:
        return stored[T]["strict"]
    t_train = (metrics.get("config") or {}).get("T_train", 8)
    if split == "test" and T == t_train and isinstance(run.get("test"), dict):
        if "strict" in run["test"]:
            return run["test"]["strict"]
    raise MissingData("%s %s sweep at T=%s" % (tag, split, T))


def settling_at(metrics, model, seed, step):
    """Settling-curve value at `step` for one run; the per-step list is
    recorded so that "list index t-1 holds step t" (docs/hypotheses.md, H-A4
    Metric)."""
    tag, run = run_of(metrics, model, seed)
    curve = run.get("settling")
    if not isinstance(curve, list) or len(curve) < step or curve[step - 1] is None:
        raise MissingData("%s settling curve at step %d" % (tag, step))
    return curve[step - 1]


def collect_per_seed(fetch, seeds):
    """Per-seed values with tolerant collection.

    Returns (values, missing_names); each missing entry yields None and the
    missing item's name so the verdict can be INCONCLUSIVE naming it."""
    values, missing = [], []
    for seed in seeds:
        try:
            values.append(fetch(seed))
        except MissingData as exc:
            values.append(None)
            missing.append(exc.name)
    return values, missing


def budget_block(metrics, arm_models, seeds, metrics_by_model=None):
    """Budget rule gate for one comparison.

    Quote (docs/hypotheses.md, Budget rule): "A directional verdict
    (SUPPORTED / NOT SUPPORTED) for any comparison below is computed only if
    every arm in that comparison finished its planned epochs (epochs_done ==
    planned, not stopped_early); otherwise the comparison is INCONCLUSIVE,
    with epochs_done reported for each arm."

    arm_models: every model whose runs are arms of the comparison; arms are
    all (model, seed) pairs. metrics_by_model: optional {model: metrics dict}
    giving the run invocation record that carries that model's own runs and
    planned epochs - the CF-WallAware arm of H3 lives in a separate metrics
    file under its own protocol record. Returns the per-arm epochs_done
    report and the problems that block a directional verdict.
    """
    by_model = metrics_by_model or {}
    try:
        planned = planned_epochs_of(metrics)
    except MissingData as exc:
        return {"eligible": False, "planned_epochs": None,
                "planned_epochs_by_model": {}, "arms": {},
                "problems": [exc.name]}
    arms, planned_by_model, problem_list = {}, {}, []
    for model in arm_models:
        source = by_model.get(model, metrics)
        try:
            model_planned = planned_epochs_of(source)
        except MissingData:
            model_planned = planned
        planned_by_model[model] = model_planned
        for seed in seeds:
            tag = "%s_seed%d" % (model, seed)
            run = (source.get("runs") or {}).get(tag)
            if not isinstance(run, dict):
                problem_list.append("missing run record %s" % tag)
                arms[tag] = {"epochs_done": None,
                             "planned_epochs": model_planned,
                             "stopped_early": None}
                continue
            finished = (run.get("status") == "done"
                        and run.get("epochs_done") == model_planned
                        and not run.get("stopped_early", False))
            arms[tag] = {"epochs_done": run.get("epochs_done"),
                         "planned_epochs": model_planned,
                         "stopped_early": run.get("stopped_early")}
            if not finished:
                problem_list.append(
                    "%s did not finish its planned epochs (epochs_done=%s, "
                    "planned=%s, stopped_early=%s)"
                    % (tag, run.get("epochs_done"), model_planned,
                       run.get("stopped_early")))
    return {"eligible": not problem_list, "planned_epochs": planned,
            "planned_epochs_by_model": planned_by_model,
            "arms": arms, "problems": problem_list}


def epochs_done_map(metrics, arm_models, seeds, metrics_by_model=None):
    """epochs_done for every arm run of the comparison (reported with the
    numbers, as the budget rule requires); each model's own run invocation
    record supplies its runs when metrics_by_model is given."""
    by_model = metrics_by_model or {}
    out = {}
    for model in arm_models:
        source = by_model.get(model, metrics)
        for seed in seeds:
            tag = "%s_seed%d" % (model, seed)
            run = (source.get("runs") or {}).get(tag)
            out[tag] = run.get("epochs_done") if isinstance(run, dict) else None
    return out


def per_seed(seeds, values):
    return {str(seed): value for seed, value in zip(seeds, values)}


def comparison_numbers(seeds, a, b, label_a, label_b, epochs):
    """Numbers used by one directional comparison: per-seed values of both
    arms, per-seed paired differences d_k, means, mean paired difference,
    sample sd of d (n = 3, ddof = 1), n and epochs_done per arm."""
    diffs = [None if (x is None or y is None) else x - y
             for x, y in zip(a, b)]
    known = [d for d in diffs if d is not None]
    return {
        "n": len(a),
        "arms": {
            "A": {"label": label_a, "per_seed": per_seed(seeds, a)},
            "B": {"label": label_b, "per_seed": per_seed(seeds, b)},
        },
        "paired_differences": per_seed(seeds, diffs),
        "mean_A": _mean(a),
        "mean_B": _mean(b),
        "mean_paired_difference": _mean(diffs) if len(known) == len(a) else None,
        "sample_sd_paired_differences":
            sample_sd(known) if len(known) == len(a) else None,
        "epochs_done": epochs,
    }


def comparison_verdict(a, b, budget):
    """Directional verdict under the budget gate: the rule applies only
    when every arm finished its planned epochs, otherwise INCONCLUSIVE."""
    if not budget["eligible"]:
        return INCONCLUSIVE
    return directional_rule(a, b)


def sweep_table(metrics, models, seeds, split="test"):
    """Per-seed strict accuracy of the full stored sweep (secondary
    evidence reported descriptively; None marks absent entries)."""
    table = {}
    for model in models:
        per_model = {}
        for seed in seeds:
            tag = "%s_seed%d" % (model, seed)
            run = (metrics.get("runs") or {}).get(tag)
            try:
                per_model[str(seed)] = (_stored_sweep(run, split, tag)
                                        if isinstance(run, dict) else None)
            except MissingData:
                per_model[str(seed)] = None
        table[model] = per_model
    return table


def _diag_per_seed(metrics, models, seeds, T, key, split="test"):
    out = {}
    for model in models:
        out[model] = {}
        for seed in seeds:
            tag = "%s_seed%d" % (model, seed)
            run = (metrics.get("runs") or {}).get(tag)
            try:
                stored = (_stored_sweep(run, split, tag)
                          if isinstance(run, dict) else {})
            except MissingData:
                stored = {}
            entry = stored.get(T) or {}
            out[model][str(seed)] = (entry.get(key)
                                      if isinstance(entry, dict) else None)
    return out


def _run_field_per_seed(metrics, models, seeds, key):
    """A per-run record field (e.g. n_params / n_params_effective) for every
    (model, seed) arm; None marks absent entries."""
    out = {}
    for model in models:
        out[model] = {}
        for seed in seeds:
            tag = "%s_seed%d" % (model, seed)
            run = (metrics.get("runs") or {}).get(tag)
            out[model][str(seed)] = (run.get(key)
                                     if isinstance(run, dict) else None)
    return out


# ---------------------------------------------------------------------------
# Post-hoc context (valT_selection.json, followups/results.json) - never
# feeds a verdict; summarized into caveats only.
# ---------------------------------------------------------------------------

def _ms(values):
    """Format a mean +- sample sd over seeds, tolerant of missing data."""
    known = [v for v in values if isinstance(v, (int, float))]
    if not known:
        return "n/a"
    sd = sample_sd(known) if len(known) > 1 else 0.0
    return "%.4f +- %.4f (n = %d)" % (statistics.mean(known), sd, len(known))


def _by_seed(entries, field):
    return {str(e.get("seed")): e.get(field) for e in entries}


def valt_summary(valT):
    if not isinstance(valT, dict):
        return {"available": False,
                "note": "valT_selection.json absent" if valT is None
                else "valT_selection.json not a JSON object"}
    selections = valT.get("selections") or []
    models = []
    for entry in selections:
        if entry.get("model") not in models:
            models.append(entry.get("model"))
    per_model = {}
    for model in models:
        rows = [e for e in selections if e.get("model") == model]
        per_model[model] = {
            "T_star_by_seed": _by_seed(rows, "T_star"),
            "mean_test_strict_at_T_star": _mean(
                [r.get("test_strict_at_T_star") for r in rows]),
        }
    return {"available": True,
            "selection_rule": valT.get("selection_rule"),
            "T_candidates": valT.get("T_candidates"),
            "per_model": per_model}


def _test_strict_by_T(run):
    out = {}
    for key, val in (run.get("test_strict_by_T") or {}).items():
        try:
            out[str(int(key))] = val
        except (TypeError, ValueError):
            continue
    return out


def followups_summary(followups):
    if not isinstance(followups, dict) or not followups:
        return {"available": False, "note": "followups/results.json absent"}
    runs = followups.get("runs") or {}
    experiments = []
    for run in runs.values():
        exp = run.get("exp") if isinstance(run, dict) else None
        if exp not in experiments:
            experiments.append(exp)
    per_exp = {}
    for exp in experiments:
        rows = [r for r in runs.values()
                if isinstance(r, dict) and r.get("exp") == exp]
        all_T = sorted({T for r in rows for T in _test_strict_by_T(r)},
                       key=int)
        per_exp[exp] = {
            "n_seeds": len(rows),
            "epochs_done_by_seed": _by_seed(rows, "epochs_done"),
            "planned_epochs_by_seed": _by_seed(rows, "planned_epochs"),
            "T_star_by_seed": _by_seed(rows, "T_star"),
            "test_strict_by_T_by_seed": {
                str(r.get("seed")): _test_strict_by_T(r) for r in rows},
            "mean_test_strict_by_T": {
                T: _mean([_test_strict_by_T(r).get(T) for r in rows])
                for T in all_T},
        }
    return {"available": True, "experiments": experiments, "per_exp": per_exp}


def _ctx_valt_line(ctx, model, what):
    valt = ctx.get("valT_selection") or {}
    if not valt.get("available"):
        return None
    info = (valt.get("per_model") or {}).get(model)
    if not info:
        return None
    stars = info.get("T_star_by_seed") or {}
    mean_star = info.get("mean_test_strict_at_T_star")
    return ("Post-hoc val-horizon selection (valT_selection.json, never part "
            "of the pre-registered rule): %s %s T* = %s (mean test strict at "
            "T* = %s); %s"
            % (model, what,
               "/".join(str(stars[str(s)]) for s in sorted(stars, key=int)),
               "%.4f" % mean_star if isinstance(mean_star, float) else "n/a",
               ("the verdict above uses the registered fixed T values."
                if ctx.get("_fixed_T_note") else ""))).strip()


def _ctx_exp_T_line(ctx, exp, T):
    fu = ctx.get("followups") or {}
    if not fu.get("available"):
        return None
    info = (fu.get("per_exp") or {}).get(exp)
    if not info:
        return None
    by_seed = info.get("test_strict_by_T_by_seed") or {}
    vals = [by_seed[s].get(str(T)) for s in by_seed]
    return "%s mean test strict at T = %d: %s" % (exp, T, _ms(vals))


# ---------------------------------------------------------------------------
# Guardrail / precondition evidence (week2_results/trained_invariants.json,
# written by scripts/check_trained_invariants.py; CPU-safe plain JSON, like
# the other inputs).  Descriptive only: it never changes a verdict, it says
# whether the doc's own guardrails were actually run and what they found.
# ---------------------------------------------------------------------------

def _inv_model_runs(invariants, model):
    """{seed: run record} of one model from trained_invariants.json."""
    if not isinstance(invariants, dict):
        return {}
    out = {}
    for rec in (invariants.get("runs") or {}).values():
        if (isinstance(rec, dict) and rec.get("model") == model
                and rec.get("seed") is not None):
            out[rec.get("seed")] = rec
    return out


def _fmt_value(value):
    return "n/a" if not isinstance(value, (int, float)) else "%.3g" % value


def _inv_check_values(runs, keys, worst):
    """The worst value of each measurement `key` over the given run records.

    `worst` is min for positivity minima (smallest = worst) and max for the
    drift / flux / gradient maxima, matching how the tolerances read.
    """
    out = {}
    for key in keys:
        vals = [((rec.get("checks") or {}).get(key)) for rec in runs.values()]
        vals = [v for v in vals if isinstance(v, (int, float))]
        out[key] = worst(vals) if vals else None
    return out


def t64_guardrail_line(invariants, model):
    """H4/H-A4 confound (iv): was the T = 64 mass-drift / positivity re-check
    run on the trained `model` checkpoints, and what was the result?

    Returns the caveat sentence stating PASS (with the worst measured
    values), FAIL (with the failing seeds), or not yet run.
    """
    if not isinstance(invariants, dict):
        return ("Guardrail evidence (confound (iv)): not yet run - %s is absent, "
                "so the T = 64 mass-drift and positivity re-check on the trained "
                "%s checkpoints has no record and the guardrail is unevidenced."
                % (INVARIANTS_NAME, model))
    runs = _inv_model_runs(invariants, model)
    if not runs:
        return ("Guardrail evidence (confound (iv)): not yet run - %s carries "
                "no %s record." % (INVARIANTS_NAME, model))
    passing = {s: rec for s, rec in runs.items() if rec.get("status") == "PASS"}
    if len(passing) == len(runs):
        worst = _inv_check_values(runs, ["min_c1", "min_c2"], min)
        worst.update(_inv_check_values(runs, ["max_relative_mass_drift"], max))
        return ("Guardrail evidence (confound (iv), %s): the T = 64 mass-drift "
                "and positivity re-check WAS run on the trained %s checkpoints "
                "(first 64 level-5 test mazes, scripts/check_trained_invariants."
                "py): PASS for all %d runs (seeds %s); worst across seeds: "
                "min c1 = %s, min c2 = %s (tolerance >= %g), max relative mass "
                "drift = %s (tolerance < %g)."
                % (INVARIANTS_NAME, model, len(runs),
                   ", ".join(str(s) for s in sorted(runs)),
                   _fmt_value(worst["min_c1"]), _fmt_value(worst["min_c2"]),
                   -1e-7, _fmt_value(worst["max_relative_mass_drift"]), 1e-5))
    bad = sorted(s for s in runs if s not in passing)
    detail = "; ".join("seed %s: %s" % (s, runs[s].get("status")) for s in bad)
    return ("Guardrail evidence (confound (iv), %s): the T = 64 re-check FAILED "
            "for %s (%s); the guardrail is NOT met, so the sweep must not be "
            "trusted until the failure is investigated."
            % (INVARIANTS_NAME, ", ".join("%s seed %s" % (model, s) for s in bad),
               detail))


def h2_precondition_line(invariants):
    """H2 Metric precondition: was the split-flux / zero-gradient invariant
    test run on the trained CF-NoFrag checkpoints, and what was the result?"""
    if not isinstance(invariants, dict):
        return ("Precondition evidence (H2 Metric): not yet run - %s is absent, "
                "so the invariant test that the split flux is exactly 0 and "
                "the split head receives exactly zero gradient on CF-NoFrag "
                "has no record and the precondition is unevidenced."
                % INVARIANTS_NAME)
    runs = _inv_model_runs(invariants, CF_NOFRAG)
    if not runs:
        return ("Precondition evidence (H2 Metric): not yet run - %s carries no "
                "%s record." % (INVARIANTS_NAME, CF_NOFRAG))
    passing = {s: rec for s, rec in runs.items() if rec.get("status") == "PASS"}
    if len(passing) == len(runs):
        worst = _inv_check_values(runs, ["split_flux_max_abs",
                                         "split_head_grad_max_abs"], max)
        return ("Precondition evidence (H2 Metric, %s): the invariant test WAS "
                "run on the trained %s checkpoints: the split flux is exactly 0 "
                "at every step of the T = 64 rollout and the split head "
                "receives exactly zero gradient (one backward pass) - PASS for "
                "all %d runs (seeds %s); max |split flux| = %s, max |split-head "
                "gradient| = %s."
                % (INVARIANTS_NAME, CF_NOFRAG, len(runs),
                   ", ".join(str(s) for s in sorted(runs)),
                   _fmt_value(worst["split_flux_max_abs"]),
                   _fmt_value(worst["split_head_grad_max_abs"])))
    bad = sorted(s for s in runs if s not in passing)
    detail = "; ".join("seed %s: %s" % (s, runs[s].get("status")) for s in bad)
    return ("Precondition evidence (H2 Metric, %s): the invariant test FAILED "
            "for %s (%s); the precondition for the arm is NOT met (%s)."
            % (INVARIANTS_NAME, ", ".join("%s seed %s" % (CF_NOFRAG, s)
                                          for s in bad), detail, H2_PRECONDITION))


def invariants_overview(invariants):
    """Top-level trained_invariants summary carried by the payload."""
    if not isinstance(invariants, dict):
        return {"available": False,
                "note": "%s absent; the H4/H-A4 T = 64 guardrail and the H2 "
                        "precondition re-checks are not yet run."
                        % INVARIANTS_NAME}
    runs = invariants.get("runs") or {}
    return {"available": True,
            "summary": invariants.get("summary"),
            "tolerances": invariants.get("tolerances"),
            "config": invariants.get("config"),
            "run_status": {tag: (rec.get("status") if isinstance(rec, dict)
                                else None)
                           for tag, rec in runs.items()}}


def wallaware_overview(wallaware):
    """Provenance of the CF-WallAware arm (H3): the wall-aware run invocation
    writes its own metrics file; its protocol record, hardware and commit are
    carried with the verdict file."""
    if not isinstance(wallaware, dict):
        return {"available": False,
                "note": "week2_results/%s/%s absent; the CF-WallAware arm "
                        "of H3 has no record."
                        % (WALLAWARE_DIRNAME, WALLAWARE_METRICS_NAME)}
    runs = wallaware.get("runs") or {}
    return {"available": True,
            "config": wallaware.get("config"),
            "hyperparams": wallaware.get("hyperparams"),
            "device": wallaware.get("device"),
            "device_name": wallaware.get("device_name"),
            "git_commit": wallaware.get("git_commit"),
            "git_dirty": wallaware.get("git_dirty"),
            "n_params_effective_by_seed": {
                str(r.get("seed")): r.get("n_params_effective")
                for r in runs.values() if isinstance(r, dict)},
            "run_status": {tag: (rec.get("status") if isinstance(rec, dict)
                                else None)
                           for tag, rec in runs.items()}}


# ---------------------------------------------------------------------------
# Hypothesis builders
# ---------------------------------------------------------------------------

def _t_train(metrics):
    return (metrics.get("config") or {}).get("T_train", 8)


def hypothesis_h1(metrics, seeds, ctx):
    """H1 - CF-Full vs RecurrentCNN at matched parameters, both at T = 8."""
    t_train = _t_train(metrics)
    arm_models = [CF_FULL, RCNN]
    a, a_missing = collect_per_seed(
        lambda s: strict_at(metrics, CF_FULL, s, t_train), seeds)
    b, b_missing = collect_per_seed(
        lambda s: strict_at(metrics, RCNN, s, t_train), seeds)
    budget = budget_block(metrics, arm_models, seeds)
    numbers = comparison_numbers(
        seeds, a, b,
        "CF-Full strict test accuracy at T = %d" % t_train,
        "RecurrentCNN strict test accuracy at T = %d" % t_train,
        epochs_done_map(metrics, arm_models, seeds))
    numbers["descriptive"] = {
        "iou_at_T_train_per_seed": _diag_per_seed(
            metrics, arm_models, seeds, t_train, "iou"),
        "probe7_strict_at_T_train_per_seed": _diag_per_seed(
            metrics, arm_models, seeds, t_train, "strict", split="probe7"),
        "probe9_strict_at_T_train_per_seed": _diag_per_seed(
            metrics, arm_models, seeds, t_train, "strict", split="probe9"),
    }
    verdict = comparison_verdict(a, b, budget)
    caveats = []
    for problem in budget["problems"]:
        caveats.append(
            "Budget rule: comparison is INCONCLUSIVE because %s; early-stopped "
            "runs are reported descriptively only." % problem)
    for name in a_missing + b_missing:
        caveats.append(
            "Missing data -> INCONCLUSIVE: %s." % name)
    band = equivalence_band(numbers["mean_A"], numbers["mean_B"],
                            numbers["sample_sd_paired_differences"])
    if band is None:
        caveats.append("Equivalence band not computable (missing data). "
                       + H1_EQUIVALENCE)
    elif band:
        caveats.append(
            "Equivalence band: |mean(CF-Full) - mean(RecurrentCNN)| = %.6f < "
            "sd(d) = %.6f, inside the pre-registered band, so the "
            "'comparable at matched budget' wording is permitted as a "
            "descriptive summary only (no TOST, no error-rate claim): "
            % (abs(numbers["mean_A"] - numbers["mean_B"]),
               numbers["sample_sd_paired_differences"]) + H1_EQUIVALENCE)
    else:
        caveats.append(
            "Equivalence band: |mean(CF-Full) - mean(RecurrentCNN)| = %.6f >= "
            "sd(d) = %.6f, outside the pre-registered band: the 'comparable "
            "at matched budget' wording is NOT permitted. %s"
            % (abs(numbers["mean_A"] - numbers["mean_B"]),
               numbers["sample_sd_paired_differences"], H1_EQUIVALENCE))
    caveats.append(H1_PREREGISTRATION)
    caveats.append(H1_CONFOUNDS)
    for line in (_ctx_valt_line(ctx, RCNN, "H1 arm"),
                 _ctx_valt_line(ctx, CF_FULL, "H1 arm")):
        if line:
            caveats.append(line + " Descriptive only.")
    for exp in ("E1_CF-Full", "E3_RecurrentCNN"):
        line = _ctx_exp_T_line(ctx, exp, t_train)
        if line:
            caveats.append("Post-hoc follow-up (descriptive only, never "
                           "feeding H1): %s." % line)
    return {
        "id": "H1",
        "title": "CF-Full vs RecurrentCNN at matched parameters",
        "statement": H1_STATEMENT,
        "rule": GLOBAL_RULE + "\n" + H1_RULE,
        "budget": budget,
        "numbers": numbers,
        "verdict": verdict,
        "caveats": caveats,
    }


def hypothesis_h2(metrics, seeds, ctx):
    """H2 - CF-Full vs CF-NoFrag at T = 8 (primary) and the gap growth
    between T = 16 and T = 8 (secondary)."""
    t_train = _t_train(metrics)
    arm_models = [CF_FULL, CF_NOFRAG]
    cf8, m1 = collect_per_seed(lambda s: strict_at(metrics, CF_FULL, s, t_train), seeds)
    cf16, m2 = collect_per_seed(lambda s: strict_at(metrics, CF_FULL, s, 16), seeds)
    nf8, m3 = collect_per_seed(lambda s: strict_at(metrics, CF_NOFRAG, s, t_train), seeds)
    nf16, m4 = collect_per_seed(lambda s: strict_at(metrics, CF_NOFRAG, s, 16), seeds)
    missing = m1 + m2 + m3 + m4
    budget = budget_block(metrics, arm_models, seeds)
    epochs = epochs_done_map(metrics, arm_models, seeds)

    primary = comparison_numbers(
        seeds, cf8, nf8,
        "CF-Full strict test accuracy at T = %d" % t_train,
        "CF-NoFrag strict test accuracy at T = %d" % t_train,
        epochs)

    gaps8 = [None if (x is None or y is None) else x - y
            for x, y in zip(cf8, nf8)]
    gaps16 = [None if (x is None or y is None) else x - y
              for x, y in zip(cf16, nf16)]
    d = [None if (x is None or y is None) else x - y
         for x, y in zip(gaps16, gaps8)]
    secondary = {
        "n": len(d),
        "gaps_at_T%d" % t_train: per_seed(seeds, gaps8),
        "gaps_at_T16": per_seed(seeds, gaps16),
        "gap_differences_d": per_seed(seeds, d),
        "mean_d": _mean(d),
        "sample_sd_d": sample_sd([v for v in d if v is not None])
        if not any(v is None for v in d) else None,
        "epochs_done": epochs,
    }
    primary_verdict = comparison_verdict(cf8, nf8, budget)
    secondary_verdict = (secondary_gap_rule(d) if budget["eligible"]
                         else INCONCLUSIVE)
    numbers = {"primary": primary, "secondary": secondary,
               "descriptive": {
                   "test_sweep_strict_per_seed":
                       sweep_table(metrics, arm_models, seeds)}}
    caveats = []
    for problem in budget["problems"]:
        caveats.append(
            "Budget rule: comparison is INCONCLUSIVE because %s; early-stopped "
            "runs are reported descriptively only." % problem)
    for name in missing:
        caveats.append("Missing data -> INCONCLUSIVE: %s." % name)
    caveats.append(H2_PRECONDITION)
    caveats.append(h2_precondition_line(ctx.get("invariants")))
    caveats.append(H2_CONFOUNDS)
    caveats.append("The observed c1/c2 trajectories predicted in the packet "
                   "are not recorded in week2_metrics.json, so that part of "
                   "the pre-registered data needed is absent.")
    for exp in ("E2_CF-NoFrag", "E4_CF-NoFrag160"):
        line = _ctx_exp_T_line(ctx, exp, 16)
        if line:
            caveats.append(
                "Post-hoc follow-up (the size-matched NoFrag variant of "
                "confound (i); descriptive only, never feeding H2): %s." % line)
    return {
        "id": "H2",
        "title": "Fragmentation ablation: CF-Full vs CF-NoFrag",
        "statement": H2_STATEMENT,
        "rule": GLOBAL_RULE + "\n" + H2_RULE,
        "budget": budget,
        "numbers": numbers,
        "verdict": primary_verdict,
        "verdict_secondary": secondary_verdict,
        "caveats": caveats,
    }


def hypothesis_h3(metrics, seeds, ctx):
    """H3 - CF-WallAware vs CF-Full at T = 8, both trained the full 40
    planned epochs; the CF-WallAware arm is the wall-aware run invocation
    in week2_results/wallaware/."""
    t_train = _t_train(metrics)
    wallaware = ctx.get("wallaware_metrics")
    view = arms_view(metrics, wallaware)
    arm_models = [CF_WALLAWARE, CF_FULL]
    sources = {CF_WALLAWARE: wallaware} if isinstance(wallaware, dict) else {}
    a, a_missing = collect_per_seed(
        lambda s: strict_at(view, CF_WALLAWARE, s, t_train), seeds)
    b, b_missing = collect_per_seed(
        lambda s: strict_at(view, CF_FULL, s, t_train), seeds)
    budget = budget_block(metrics, arm_models, seeds, metrics_by_model=sources)
    numbers = comparison_numbers(
        seeds, a, b,
        "CF-WallAware strict test accuracy at T = %d" % t_train,
        "CF-Full strict test accuracy at T = %d" % t_train,
        epochs_done_map(metrics, arm_models, seeds, metrics_by_model=sources))
    numbers["descriptive"] = {
        "n_params_per_seed": _run_field_per_seed(
            view, arm_models, seeds, "n_params"),
        "n_params_effective_per_seed": _run_field_per_seed(
            view, arm_models, seeds, "n_params_effective"),
        "wall_hits_at_T_train_per_seed": _diag_per_seed(
            view, arm_models, seeds, t_train, "wall_hits"),
        "iou_at_T_train_per_seed": _diag_per_seed(
            view, arm_models, seeds, t_train, "iou"),
        "test_sweep_strict_per_seed": sweep_table(view, arm_models, seeds),
        "note": "n_params_effective is recorded only in the wall-aware run "
                "books (9,921); the packet counts 10,177 raw and effective "
                "for CF-Full (Packet \u00a712.3, p.28), and the raw count is "
                "10,177 for both arms. IoU and wall-hits are diagnostics "
                "only; wall-hits are 0 for CF-WallAware by construction "
                "(readout mask) and are descriptive, never a verdict input.",
    }
    verdict = comparison_verdict(a, b, budget)
    caveats = []
    for problem in budget["problems"]:
        caveats.append(
            "Budget rule: comparison is INCONCLUSIVE because %s; early-stopped "
            "runs are reported descriptively only." % problem)
    if not isinstance(wallaware, dict):
        caveats.append(
            "Missing data -> INCONCLUSIVE: the CF-WallAware arm "
            "(week2_results/%s/%s) is absent, so the H3 arms cannot be "
            "paired and the comparison is INCONCLUSIVE."
            % (WALLAWARE_DIRNAME, WALLAWARE_METRICS_NAME))
    for name in a_missing + b_missing:
        caveats.append(
            "Missing data -> INCONCLUSIVE: %s." % name)
    caveats.append(_h3_sources_line(metrics, wallaware))
    caveats.append(H3_ARMS_NOTE)
    caveats.append(H3_CONFOUNDS)
    return {
        "id": "H3",
        "title": "Wall-aware arm: CF-WallAware vs CF-Full",
        "statement": H3_STATEMENT,
        "rule": GLOBAL_RULE + "\n" + H3_RULE,
        "budget": budget,
        "numbers": numbers,
        "verdict": verdict,
        "caveats": caveats,
    }


def _h3_sources_line(metrics, wallaware):
    """Sentence stating both arms' run provenance (files, commits, hardware),
    as the doc's H3 'Comparison and data needed' fixes it."""
    line = ("Arms' provenance: CF-Full arm = week2_results/%s (git commit %s, "
            "%s)" % (WALLAWARE_METRICS_NAME, metrics.get("git_commit"),
                     metrics.get("device_name") or metrics.get("device")))
    if isinstance(wallaware, dict):
        line += ("; CF-WallAware arm = week2_results/%s/%s (git commit %s, %s)"
                 % (WALLAWARE_DIRNAME, WALLAWARE_METRICS_NAME,
                    wallaware.get("git_commit"),
                    wallaware.get("device_name") or wallaware.get("device")))
    else:
        line += ("; CF-WallAware arm absent (week2_results/%s/%s)"
                 % (WALLAWARE_DIRNAME, WALLAWARE_METRICS_NAME))
    return line


def hypothesis_h4(metrics, seeds, ctx):
    """H4 - Thinking longer: CF-Full frozen weights, strict test accuracy
    T = 16 vs T = 8."""
    t_train = _t_train(metrics)
    arm_models = [CF_FULL]
    a, a_missing = collect_per_seed(
        lambda s: strict_at(metrics, CF_FULL, s, 16), seeds)
    b, b_missing = collect_per_seed(
        lambda s: strict_at(metrics, CF_FULL, s, t_train), seeds)
    budget = budget_block(metrics, arm_models, seeds)
    numbers = comparison_numbers(
        seeds, a, b,
        "CF-Full strict test accuracy at T = 16",
        "CF-Full strict test accuracy at T = %d" % t_train,
        epochs_done_map(metrics, arm_models, seeds))
    numbers["descriptive"] = {
        "test_sweep_strict_per_seed":
            sweep_table(metrics, [CF_FULL, RCNN], seeds),
        "probe7_sweep_strict_per_seed":
            sweep_table(metrics, [CF_FULL], seeds, split="probe7"),
        "probe9_sweep_strict_per_seed":
            sweep_table(metrics, [CF_FULL], seeds, split="probe9"),
        "note": "full sweep curve for CF-Full and the same sweep for the "
                "RecurrentCNN as the planned contrast (Packet \u00a70 item 8: "
                "overthinking at T > T_train); the unseen-grid sweeps are "
                "secondary evidence",
        "secondary_points_mean_strict": {
            "T=24": _mean([v for v in collect_per_seed(
                lambda s: strict_at(metrics, CF_FULL, s, 24), seeds)[0]]),
            "T=32": _mean([v for v in collect_per_seed(
                lambda s: strict_at(metrics, CF_FULL, s, 32), seeds)[0]]),
        },
    }
    verdict = comparison_verdict(a, b, budget)
    caveats = []
    for problem in budget["problems"]:
        caveats.append(
            "Budget rule: comparison is INCONCLUSIVE because %s; early-stopped "
            "runs are reported descriptively only." % problem)
    for name in a_missing + b_missing:
        caveats.append("Missing data -> INCONCLUSIVE: %s." % name)
    sec = numbers["descriptive"]["secondary_points_mean_strict"]
    if isinstance(sec.get("T=24"), float) and isinstance(sec.get("T=32"), float):
        caveats.append(
            "Secondary points (mean strict over %d seeds): T = 24 = %.4f, "
            "T = 32 = %.4f; the directional verdict covers only the "
            "pre-registered 16-vs-8 comparison."
            % (len(seeds), sec["T=24"], sec["T=32"]))
    caveats.append(H4_CONFOUNDS)
    caveats.append(t64_guardrail_line(ctx.get("invariants"), CF_FULL))
    line = _ctx_valt_line(ctx, CF_FULL, "H4 model")
    if line:
        caveats.append(line + " Descriptive only.")
    fu_line = _ctx_exp_T_line(ctx, "E1_CF-Full", 16)
    fu_line8 = _ctx_exp_T_line(ctx, "E1_CF-Full", t_train)
    if fu_line and fu_line8:
        caveats.append("Post-hoc follow-up (descriptive only, never feeding "
                        "H4): %s; %s." % (fu_line8, fu_line))
    return {
        "id": "H4",
        "title": "Thinking longer: test-time T beyond the training horizon",
        "statement": H4_STATEMENT,
        "rule": GLOBAL_RULE + "\n" + H4_RULE,
        "budget": budget,
        "numbers": numbers,
        "verdict": verdict,
        "caveats": caveats,
    }


def hypothesis_h_a4(metrics, seeds, ctx):
    """H-A4 (supplementary, docs/hypotheses.md appendix) - settling:
    (a) CF-Full residual flux step 16 < step 8;
    (b) RecurrentCNN relative state change step 16 > step 8."""
    flux8, m1 = collect_per_seed(
        lambda s: settling_at(metrics, CF_FULL, s, 8), seeds)
    flux16, m2 = collect_per_seed(
        lambda s: settling_at(metrics, CF_FULL, s, 16), seeds)
    sc8, m3 = collect_per_seed(
        lambda s: settling_at(metrics, RCNN, s, 8), seeds)
    sc16, m4 = collect_per_seed(
        lambda s: settling_at(metrics, RCNN, s, 16), seeds)
    missing = m1 + m2 + m3 + m4
    # Budget rule "every arm in that comparison": clause (a) is a CF-Full-only
    # comparison (its arms are the CF-Full checkpoints), clause (b) a
    # RecurrentCNN-only one, so each clause is gated on its own arms' budget
    # status, not on both models.
    budget = {"a": budget_block(metrics, [CF_FULL], seeds),
              "b": budget_block(metrics, [RCNN], seeds)}
    clause_a_numbers = comparison_numbers(
        seeds, flux8, flux16,
        "CF-Full residual flux at step 8",
        "CF-Full residual flux at step 16",
        epochs_done_map(metrics, [CF_FULL], seeds))
    clause_b_numbers = comparison_numbers(
        seeds, sc16, sc8,
        "RecurrentCNN relative state change at step 16",
        "RecurrentCNN relative state change at step 8",
        epochs_done_map(metrics, [RCNN], seeds))
    clause_a = comparison_verdict(flux8, flux16, budget["a"])
    clause_b = comparison_verdict(sc16, sc8, budget["b"])
    verdict = ha4_verdict(clause_a, clause_b)
    caveats = []
    for clause, label in (("a", "a (CF-Full flux)"), ("b", "b (RecurrentCNN state change)")):
        for problem in budget[clause]["problems"]:
            caveats.append(
                "Budget rule (clause %s): the clause is INCONCLUSIVE because "
                "%s; early-stopped runs are reported descriptively only."
                % (label, problem))
    for name in missing:
        caveats.append("Missing data -> INCONCLUSIVE: %s." % name)
    caveats.append(HA4_METRIC_NOTE)
    caveats.append(HA4_CONFOUNDS)
    caveats.append(t64_guardrail_line(ctx.get("invariants"), CF_FULL))
    return {
        "id": "H-A4",
        "title": "Settling: residual flux after the training horizon "
                 "(supplementary)",
        "statement": HA4_STATEMENT,
        "rule": GLOBAL_RULE + "\n" + HA4_RULE,
        "budget": budget,
        "numbers": {
            "clause_a_flux": clause_a_numbers,
            "clause_b_state_change": clause_b_numbers,
        },
        "clause_verdicts": {"a": clause_a, "b": clause_b},
        "verdict": verdict,
        "caveats": caveats,
    }


def build_verdicts(metrics, valT=None, followups=None, sources=None,
                    invariants=None, wallaware=None):
    """Assemble the full verdicts payload from the input JSONs.

    week2_metrics.json carries the RecurrentCNN / CF-Full / CF-NoFrag arms;
    week2_results/wallaware/week2_metrics.json the CF-WallAware arm of H3;
    valT_selection.json and followups/results.json are post-hoc context only
    and never change a verdict; trained_invariants.json (optional) is
    guardrail / precondition evidence quoted in the H2/H4/H-A4 caveats -
    descriptive, never a verdict.
    """
    seeds = list((metrics.get("config") or {}).get("seeds", [0, 1, 2]))
    ctx = {"valT_selection": valt_summary(valT),
           "followups": followups_summary(followups),
           "invariants": invariants,
           "wallaware_metrics": wallaware,
           "_fixed_T_note": True}
    payload = {
        "rule_global": GLOBAL_RULE,
        "rule_budget": BUDGET_RULE,
        "multiplicity_note": MULTIPLICITY,
        "scope_note": SCOPE_NOTE,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "T_train": _t_train(metrics),
        "planned_epochs": (planned_epochs_of(metrics)
                           if (metrics.get("hyperparams") or {}).get("epochs")
                           is not None else None),
        "source_files": sources or {},
        "trained_invariants": invariants_overview(invariants),
        "wallaware_arm": wallaware_overview(wallaware),
        "post_hoc_context": {
            "note": "valT_selection.json and followups/results.json are "
                    "post-hoc descriptive context; they never feed a verdict.",
            "valT_selection": ctx["valT_selection"],
            "followups": ctx["followups"],
        },
        "hypotheses": {
            "H1": hypothesis_h1(metrics, seeds, ctx),
            "H2": hypothesis_h2(metrics, seeds, ctx),
            "H3": hypothesis_h3(metrics, seeds, ctx),
            "H4": hypothesis_h4(metrics, seeds, ctx),
        },
        "supplementary": {
            "H-A4": hypothesis_h_a4(metrics, seeds, ctx),
        },
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_verdicts(path, payload):
    """Atomically write the verdict JSON (no partial file on crash)."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, path)
    return path


def _load_json(path, optional):
    if optional and not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def format_summary(payload):
    """Printed one-line-per-hypothesis summary: the four verdicts."""
    lines = []
    for hid in ("H1", "H2", "H3", "H4"):
        hyp = payload["hypotheses"][hid]
        extra = ""
        if hid == "H2":
            extra = " | secondary: %s" % hyp["verdict_secondary"]
        lines.append("%-3s %-14s %s%s" % (hid, hyp["verdict"], hyp["title"], extra))
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(
        Path(__file__).resolve().parents[1] / "week2_results"),
        help="folder with week2_metrics.json (default: week2_results)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = Path(args.results)
    metrics_path = results / WALLAWARE_METRICS_NAME
    if not metrics_path.exists():
        print("no %s" % metrics_path)
        return 1
    metrics = _load_json(metrics_path, optional=False)
    wallaware_path = (results / WALLAWARE_DIRNAME / WALLAWARE_METRICS_NAME)
    wallaware = _load_json(wallaware_path, optional=True)
    valt = _load_json(results / "valT_selection.json", optional=True)
    followups = _load_json(results / "followups" / "results.json", optional=True)
    invariants = _load_json(results / INVARIANTS_NAME, optional=True)
    sources = {"week2_metrics": str(metrics_path),
               "wallaware_metrics": str(wallaware_path),
               "valT_selection": str(results / "valT_selection.json"),
               "followups": str(results / "followups" / "results.json"),
               "trained_invariants": str(results / INVARIANTS_NAME)}
    payload = build_verdicts(metrics, valT=valt, followups=followups,
                             sources=sources, invariants=invariants,
                             wallaware=wallaware)
    out_path = write_verdicts(results / OUT_NAME, payload)
    print(format_summary(payload))
    print("\nverdicts written to %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
