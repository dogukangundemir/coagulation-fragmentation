# Family B — the Becker-Döring commitment ladder: confidence that grows one rung at a time

Specification card written before implementation, per section 11.3 of
`docs/cf_algorithm_families.md`. It answers the eight axes of section 1, the
eight family headings (idea, physics, state, update, meaning in a maze,
information flow, guarantees and energy, risks), and then fixes every
implementation detail: tensor shapes, the discrete-time update with all
constants, what is learned and what is fixed, the invariants to test, the
readout, and the planned experiments with their exact metrics. Conventions
follow the existing code: the `forward(x, T)` / `logits_at(Ts, record)`
interface and the variant options of `cf/models/cf.py`, the split builders
of `cf/data/curriculum.py`, and the strict/exact metrics of `cf/metrics.py`.

## Plain-language summary

The existing model gives every cell two states: loose evidence and committed
evidence, with nothing in between. This family gives every cell a small
ladder of states instead — rung 1 is "no idea", the top rung is "certainly on
the path". A cell climbs one rung at a time when it absorbs a loose piece of
evidence from its own supply; it steps down one rung at a time when it sheds
a piece back. The total evidence a cell holds never changes — it only
redistributes across the rungs — and the rates of climbing and stepping down
are set by a small network that looks at the cell's neighbourhood. Because of
how the rates are built, the ladder has a genuine energy: a number that the
cell's chemistry always decreases while the rates are held fixed, the way a
ball rolls downhill and stops. A cell that keeps absorbing climbs past a
"commitment" rung and then no single weak signal can knock it back down —
it takes sustained contrary evidence to dissolve it. The resistance lives one
level up from the chemistry: within one fast phase the energy is convex, so
the ladder always settles to that phase's unique equilibrium, and what moves
slowly is the equilibrium itself, because the rates that set it are refreshed
only every few steps. That is what "confidence with hysteresis" means here,
and it is the piece the two-state model lacks.

## The eight axes

| # | Axis | Choice in Family B |
| --- | --- | --- |
| 1 | Particle | one loose evidence unit (a monomer, a chemistry mass unit) held by a cell |
| 2 | Size space | cluster sizes 1..K with K = 4 — the rungs of the ladder (variants 6, 8) |
| 3 | Coagulation kernel | Becker-Döring: only monomer additions A_1 + A_k → A_{k+1}, rate a_k = μ (mobility), the same for every k |
| 4 | Fragmentation | monomer shedding A_{k+1} → A_k + A_1, rate b_{k+1} = μ Q_k / Q_{k+1} (detailed balance by construction): confidence is taken back one rung at a time |
| 5 | Conservation scope | mass per cell and per channel: M = Σ_k k·c_k fixed from t = 0 (a numerical safeguard plus a condensation threshold, as in Family A) |
| 6 | Transport | only through the rate network: two 3×3 convolutions, at most 2 cells of spatial reach per step; mass never moves between cells |
| 7 | Determinism | deterministic mean-field update; a random-start variant splits the initial mass between the first two rungs |
| 8 | Energy and readout | the free energy V = Σ_k c_k (log(c_k/Q_k) − 1) decreases in continuous time under fixed rates (detailed balance; monitored per micro-step in the discrete update); readout is a 1×1 convolution over the rungs, with the mass fraction above the commitment rung as the interpretable diagnostic |

## Idea

Replace "loose vs committed" by a ladder: confidence grows one unit at a time
by absorbing a monomer and shrinks one unit at a time by shedding one, and
the free energy decides where the ladder settles. The network does not
choose the updates directly; it shapes the energy landscape the chemistry
then rolls down.

## Physics

Becker-Döring kinetics (section 2.4 of the families document): clusters
change size only by gaining or losing one monomer, A_k + A_1 ⇌ A_{k+1}.
Under detailed balance there is a Lyapunov free energy, and in the
untruncated system (cluster sizes unbounded) above a critical monomer
activity the excess mass condenses into large clusters. The implemented
ladder truncates at K = 4 rungs, so it has no phase transition: here
"condensation" only means that the mass distribution shifts towards the top
rung as the learned energy profile flattens towards it. The
critical-activity picture is the K → ∞ limit that motivates the family, not
a property of the trained layer. The two-timescales structure mirrors the
fast/slow phase split of the
oscillator-based networks the project builds on: the energy profile is
recomputed from the neighbourhood only every few steps (slow), and the
chemistry relaxes under frozen rates in between (fast, with the provable
energy).

## State

Per cell and per channel: rung concentrations c_1, …, c_K ≥ 0
(K = 4 rungs by default; hid = 8 independent ladders per cell, each an
independent chemistry like the 16 independent dimers of Family A). The mass
M = Σ_k k·c_k is fixed per cell and channel at t = 0. Interpretation: c_k is
how much of the cell's evidence sits in clusters of confidence level k, and
M is the cell's fixed evidence budget.

## Update

With μ the mobility, Q_k = exp(−g_k) the per-rung equilibrium weights
(gauge: Q_1 = 1, i.e. Q_k = exp(g_1 − g_k)), and a slow rate network
producing g (a per-cell energy profile over the rungs) and μ from
`RateNet([x, c])`:

- net rung flux (positive = upward): J_k = a_k c_1 c_k − b_{k+1} c_{k+1},
  with a_k = μ and b_{k+1} = μ Q_k / Q_{k+1}, for k = 1..K−1;
- rung balances with J_0 = J_K = 0:
  Δc_1 = −2 J_1 − Σ_{k=2}^{K−1} J_k,   Δc_k = J_{k−1} − J_k (2 ≤ k ≤ K).

The factor 2 in Δc_1 appears because rung 1 both donates the monomer and,
for k = 1, is itself the partner (A_1 + A_1 → A_2 consumes two monomers
and the reverse sheds two). The discrete-time step caps all fluxes jointly
so every c_k stays non-negative (the exact scheme is under *Implementation*).

## What it means in a maze

A cell's rung is its confidence level. Cells that keep receiving support
from their neighbourhood become supersaturated and condense into the top
rungs ("on the path"); cells in dead ends stay low and remain mostly at
rung 1. Shedding gives gradual doubt instead of the two-state model's
all-or-nothing split. The hysteresis is a two-timescales effect, not a
metastability of the chemistry: within a fast phase the free energy is
convex on the mass simplex, so the ladder has exactly one equilibrium for
the current rates, and what resists one weak signal is that those rates —
hence the equilibrium — are refreshed only every r_s steps; sustained
contrary evidence shifts the rate profile and dissolves the condensed cell
rung by rung.

**Worked numbers.** Take one ladder with mass M = 1 and the energy profile
g = (0, 1, 2, 3), so the equilibrium weights are Q = (1, 0.368, 0.135,
0.0498). Setting every rung flux J_k = 0 forces
c_{k+1} = (Q_{k+1}/Q_k)·c_1·c_k on every rung, and iterating from rung 1
gives c_k = Q_k·z^k with z = c_1: a one-parameter family (the activity z),
with no free amplitude — the amplitude is fixed once z is chosen. Fixing the
mass M = Σ_k k·c_k = 1 then pins z: solving
z + 2·0.368·z² + 3·0.135·z³ + 4·0.0498·z⁴ = 1 gives z ≈ 0.609, so the
stationary ladder is c ≈ (0.609, 0.136, 0.031, 0.007), mean cluster size
Σ k c_k / Σ c_k = 1 / Σ c_k ≈ 1.28, and the mass fraction above the
commitment rung k_c = 3 is 3·0.031 + 4·0.007 ≈ 0.12 of the cell's evidence.
The free energy of the all-monomer state (c = (1, 0, 0, 0), same gauge) is
V = 1·(log(1/1) − 1) = −1.0; at the stationary ladder c_k/Q_k = z^k, so
V = Σ_k c_k (k·log z − 1) = log z · Σ k c_k − Σ c_k = log(0.609) − 0.782 ≈
−1.28. The chemistry moves downhill on V, exactly as the continuous-time
theorem says: starting all-monomer, the cell condenses part of its evidence
up the rungs and V falls from −1.0 towards −1.28. With a flat profile
(g ≡ 0, the parameter-free reference ladder used in the experiments) the
stationary ladder of mass 1 is the geometric one c ≈ (0.397, 0.158, 0.063,
0.025) (it satisfies c_{k+1} = c_1·c_k), with mean size ≈ 1.56 — the
self-consistent monomer level the truncated ladder settles at when the
network shapes no landscape. Note the direction of the comparison: a profile
that rises with k (g = (0, 1, 2, 3)) penalises the upper rungs and condenses
*less* than the flat ladder (mean size 1.28 vs 1.56), not more — the network
shapes condensation through the profile's slope, and cells commit when the
learned profile flattens or falls towards the top rung.

## How information moves

Only through the rate network, as in Family A: the slow phase is two 3×3
convolutions, so spatial reach is at most 2 cells per step and long-range
reasoning is out of this family's scope by design — it is the per-cell
confidence mechanism, to be composed with Family D's transport for reach.

## Guarantees and energy

All of the following hold for any network weights:

- **Mass conservation.** The rung balances above conserve
  M = Σ_k k·c_k exactly at every step, for any flux magnitudes.
- **Positivity.** The joint cap described under *Implementation* bounds
  every rung's total outflow by a fixed fraction ζ of its stock, so c_k ≥ 0
  always.
- **Detailed balance by construction.** a_k = μ and
  b_{k+1} = μ Q_k / Q_{k+1} satisfy a_k Q_1 Q_k = b_{k+1} Q_{k+1}, so the
  state c_k = Q_k·z^k (any activity z > 0, with c_1 = z) has zero net flux
  on every rung: J_k = μ (c_1 c_k − (Q_k/Q_{k+1}) c_{k+1}) =
  μ Q_k z^{k+1} − μ Q_k z^{k+1} = 0. This is the unique equilibrium family —
  an arbitrary rescaling A·Q_k·z^k does not have zero flux unless A = 1
  (with the gauge Q_1 = 1).
- **Free energy (continuous time).** Under fixed rates (within a fast
  phase) the free energy V = Σ_k c_k (log(c_k/Q_k) − 1), summed over rungs,
  cells and channels, decreases in continuous time — the detailed-balance
  Lyapunov theorem. The capped discrete update inherits positivity and
  mass conservation but not the descent: a forward-Euler micro-step can
  overshoot the convex minimum where a rung is near-empty and the rates are
  stiff (the Hessian ∇²V = diag(1/c_k) blows up as c_k → 0), so V is
  recorded per micro-step in the trace and is empirically non-increasing
  for rate outputs in the trained range; per-micro-step monotonicity is a
  monitored diagnostic, not a guarantee. V is the layer's natural
  convergence diagnostic and the candidate selection energy for stochastic
  voting.

## Risks

1. **K times more state** than Family A per channel — parameter counts are
   given under *Implementation*; the matched-budget arm exists precisely for
   this comparison.
2. **Slow nucleation.** A cell starting all-monomer needs several events to
   reach the commitment rung; if μ is small or the profile g is steep, the
   ladder stalls below it and predictions come out undecided
   (mitigations: warm-up schedules on the profile, or initializing the
   profile flat, which is already the default g = 0 start).
3. **Discrete-time stiffness.** Large raw rates produce large requests; the
   joint cap then throttles all fluxes together (a shedding request can
   slow merging in the same cell). This is the price of the exact
   positivity guarantee; it is studied via the fast-phase step count n_f.
4. **No transport.** Like Family A, reach is 2 cells per step; every claim
   about solving long mazes belongs to the D + B composition, not to B
   alone.

**Role.** The best local mechanism: it gives the chemistry a provable energy
and a meaningful notion of graduated confidence. Compose with Family D
(`docs/specs/family_d_transport.md`), whose local flow feeds the rates.

## Implementation

Planned module: `cf/models/ladder.py`, class `LadderCFNet`, exposing the
CFNet interface used by `cf/train.py`: `forward(x, T)` returning logits
`(B, H, W)`, plus `initial_state`, `step`, `read`, and `logits_at(Ts,
record)`; with `record=True` the trace stores, per outer step, the rung
maps c_k (as a stacked tensor), the mass M, the applied rung fluxes J_k, the
per-step free energy V_t, and the residual flux Σ|J| / Σ M. Variants are
registered as `MODEL_SPECS = {"BD-Full": dict(detailed_balance=True),
"BD-FreeRates": dict(detailed_balance=False)}`, mirroring
`cf/models/cf.py`.

### Tensor shapes on a B x H x W grid

| Name | Shape | Meaning |
| --- | --- | --- |
| x | (B, 3, H, W) | input channels [walkable, start, goal], as produced by `cf.data.curriculum.maze_to_channels` |
| c | (B, hid, K, H, W) | rung concentrations; flattened to (B, hid·K, H, W) with rung-major layout [c_1, …, c_K] per channel for every convolutional call |
| M = Σ_k k·c_k | (B, hid, H, W) | conserved mass per cell and channel |
| J | (B, hid, K−1, H, W) | net upward rung fluxes |
| g | (B, hid, K, H, W) | per-rung energy profile (slow-phase output of the rate network) |
| μ | (B, hid, H, W) | mobility (slow-phase output) |
| logits | (B, H, W) | per-cell path logits; exactly −20.0 in walls when the wall-aware option is on (the convention of `cf/models/cf.py`) |

With the wall-aware option on, the mass is multiplied by x[:, 0:1] at
initialisation and the rate-network hidden activations are gated by it, so
walls hold exactly zero state at every step; with the option off, walls may
hold mass but never contribute at readout. The wall-aware variant is the
default for this family: walls are impassable for evidence, and the ladder
claims no robustness for teaching walls to hold mass.

### Discrete-time update with all constants

Constants (fixed, not learned):

| Constant | Default | Role |
| --- | --- | --- |
| K | 4 | rungs per ladder (1 = loose … K = fully committed); variants 6, 8 |
| hid | 8 | independent ladders per cell (BD-8); hid = 12 gives the matched-budget arm (see below) |
| ζ (zeta) | 0.5 | joint flux cap: every rung's total outflow per micro-step is bounded by ζ of its stock (the analogue of the 1/2 merge bound in `cf/models/cf.py`) |
| ε (epsilon) | 1e-12 | request floor inside the cap |
| r_s | 2 | slow phase: the rates (g, μ) are refreshed every r_s outer steps |
| n_f | 2 | fast phase: capped chemistry micro-steps per outer step under frozen rates |
| T_train | 8 | outer steps at training, matching the shared protocol horizon |

Initialisation (as in `cf/models/cf.py`): M = softplus(encoder(x)), a 3×3
convolution over the input; the wall-aware option multiplies M by
x[:, 0:1]; the mass starts all at the bottom rung: c_1 = M, c_{k>1} = 0.
Random-start variant (the stochastic layer's cheapest level, for
tie-breaking and later voting): draw u ~ U(0, 0.5) per cell and channel and
set c_2 = u·M/2, c_1 = (1 − u)·M, other rungs 0. Rate-network
initialisation: g starts at 0 (every Q_k = 1, the flat reference ladder)
and the μ head starts with bias log(e − 1) ≈ 0.5413 after softplus, so the
mobility starts at μ ≈ 1.

Slow phase (every r_s outer steps): a rate network of two 3×3 convolutions
in the CFNet style — input channel count 3 + hid·K, hidden width hid, two
output heads: g (hid·K channels, the per-rung energy profile) and raw_μ
(hid channels; μ = softplus(raw_μ)). The cross-kernel mask option of
`cf/models/cf.py` applies unchanged (the 4-neighbourhood mask on the rate
net's two 3×3 convolutions). Gauge: Q_1 = 1 and Q_k = exp(g_1 − g_k), so
the weights are positive for any g and the overall scale of g is absorbed
into the mobility.

Fast phase (n_f micro-steps per outer step, rates frozen):

1. Raw requests per cell, channel and rung: merge request
   r_k = μ c_1 c_k and shed request d_k = μ (Q_k / Q_{k+1}) c_{k+1}, both
   for k = 1..K−1.
2. Per-rung total requested outflow:
   O_1 = 2 r_1 + Σ_{k=2}^{K−1} r_k (rung 1 donates both monomers of a
   merge); O_j = r_j + d_{j−1} for 2 ≤ j ≤ K−1 (it can merge up or shed
   down); O_K = d_{K−1} (the top rung can only shed; J_K = 0 by
   truncation).
3. Joint cap: Λ = min(1, ζ · min_j c_j / (O_j + ε)), applied to all fluxes.
4. Applied rung fluxes: J_k = Λ (r_k − d_k), then
   Δc_1 = −2 J_1 − Σ_{k=2}^{K−1} J_k,   Δc_k = J_{k−1} − J_k (2 ≤ k ≤ K).

Positivity is exact under this scheme: rung j loses at most Λ·O_j ≤ ζ·c_j ≤
c_j towards other rungs and only ever receives, so c_j stays ≥ 0; and the
mass change is Σ_k k·Δc_k = 0 identically in J. The free-rates variant
(BD-FreeRates) replaces the (g, μ) heads with direct softplus rate heads
a_k, b_{k+1} (most flexible, no energy guarantee — the ablation arm of
experiment X2); every discrete-time part (cap, balances, conservation,
positivity) is unchanged.

### Start and goal placement

Family B has no source or sink and no mass moving between cells, so nothing
is injected anywhere: the start and goal enter as the one-hot channels of x
(the second and third input channels built by
`cf.data.curriculum.maze_to_channels`), seen by the encoder (which sets the
mass M) and by the rate network (which sets the energy profile and
mobility). No cell receives special treatment in the update; the signalling
role of the start and goal is entirely up to the learned rates, exactly as
in Family A. The transport analogue — mass injected at the start and
drained at the goal — is Family D's card.

### What is learned and what is fixed

| Learned (small, strictly local; slow phase) | Fixed (the chemistry; not parameterized at all) |
| --- | --- |
| the energy profile g over the rungs (hid·K numbers per cell) | the stoichiometry of the Becker-Döring reactions and the rung balances |
| the mobility μ (hid numbers per cell) | the conservation of the mass M = Σ_k k·c_k (exact for any fluxes; M's value comes from the learned encoder — only its conservation is fixed) |
| the readout head and the encoder producing M | the detailed-balance relation b_{k+1} = μ Q_k / Q_{k+1}, hence the free energy |
| (free-rates variant only: the rate heads a_k, b_{k+1}) | the joint flux cap (positivity for any weights), the gauge Q_1 = 1, the truncation J_K = 0, the wall handling |

Parameter counts (K = 4; encoder 3×3 → hid; rate net 3×3 → hid → (K+1)·hid;
readout 1×1 over hid·K → 1):

| Config | hid | Rate net | Parameters (approx.) |
| --- | --- | --- | --- |
| BD-8 | 8 | 35 → 8 → 40 | ≈ 5,700 |
| BD-12 (matched-budget arm) | 12 | 51 → 12 → 60 | ≈ 12,400 |
| BD-16 | 16 | 67 → 16 → 80 | ≈ 21,800 |

The matched-budget reference is Family A's CF-Full at 10,177 parameters;
BD-12 is the closest arm under this layout.

### Invariants to test

Mirroring the style of `tests/test_cf_invariants.py` (float32 tolerances;
NEG_EPS = −1e-7; relative drift tolerance 1e-5), on synthetic mazes and
random grids, at initialisation and after SGD:

1. **Mass conservation.** M = Σ_k k·c_k per cell and channel: relative
   drift < 1e-5 over 64 outer steps, for the detailed-balance arm, the
   free-rates arm and the random-start variant alike.
2. **Positivity with jointly bounded fluxes.** c_k ≥ NEG_EPS at t = 0 and
   after every micro-step, for any rate-network outputs; a probe drives the
   raw heads to extreme values (merge-happy and shed-happy) to confirm that
   the joint cap, not the smallness of the rates, is what protects
   positivity.
3. **Detailed balance by construction.** For random (g, μ) outputs and a
   random activity z > 0 (drawn once per test case), the state
   c_k = Q_k·z^k gives rung fluxes |J_k| < 1e-7·(1 + M) on every rung;
   the mass M = Σ_k k·Q_k·z^k follows from z — it is not a free parameter,
   so z is picked first and the mass is whatever it is. In the free-rates
   arm this property is absent by design and the test reports that absence.
4. **Free energy under fixed rates (monitored diagnostic).** Freeze the
   slow phase (skip rate refreshes for 32 outer steps, running the fast
   phase alone) and record V = Σ_k c_k (log(c_k/Q_k) − 1) over all rungs,
   cells and channels after every micro-step. Hard assertions: from the
   all-monomer start V decreases strictly at least once (beyond the
   tolerance 1e-6·max(1, |V|)) before the state is chemically at rest
   (Σ|J| / Σ M < 1e-5), and V at rest is below V at the start. Per
   micro-step monotonicity is monitored, not asserted: the capped
   forward-Euler step is a Lyapunov descent only in continuous time, and a
   randomized search over stiff rate outputs finds one-micro-step V
   increases of order 10 with mass and positivity intact (near-empty rungs
   make ∇²V = diag(1/c_k) large, so a step can overshoot the convex
   minimum). The test reports the largest per-micro-step increase,
   flagging anything above 1e-6·max(1, |V|) as a diagnostic warning rather
   than a failure; for rate outputs in the trained range it is expected to
   be zero.
5. **Truncation.** J_K = 0 exactly (no flux past the top rung) and no rung
   above K receives mass; checked jointly with the mass identity of item 1.
6. **Wall handling (wall-aware variant).** Rung maps are exactly 0 in wall
   cells at every step, and wall logits are exactly −20.0; the
   stochastic-start variant keeps the exact zeros (the same test as in the
   Family A suite).
7. **Gradient reach.** A backward pass from the summed logits reaches every
   parameter and the input, finite and non-zero (the same probe as in the
   Family A suite, extended to the μ head and every g slice).

### Readout

Primary: logits = Conv1x1(flattened rung stack of hid·K channels → 1) per
cell, masked to exactly −20.0 in walls in the wall-aware variant, and
thresholded at 0.5 like the existing models; evaluation is unchanged
`cf.train.evaluate_model(model, d, T)`. Interpretable diagnostics (recorded
in the trace and reported in figures, not trained on): the mean cluster size
Σ_k k·c_k / Σ_k c_k per cell, and the committed share
Σ_{k ≥ k_c} k·c_k / M with the commitment rung k_c = 3 for K = 4 — the
ladder analogue of Family A's 2·c2/M, plotted against the label path.

### Planned experiments

Shared protocol for every arm: the split builders and probe sets of
`cf/data/curriculum.py` with their defaults (`make_splits` with
week1_level=5, 4000/1000/1000 train/val/test, 1000-maze probes at levels 7
and 9, seed 0; `build_datasets` → `data["test"]` and `data["probe"][7]`,
`data["probe"][9]`); seeds 0, 1, 2; the training hyperparameters of
`cf/train.py` (Adam lr 1e-3, cosine decay to 0 per step, grad-norm clip 1.0,
batch 32, 40 epochs, BCEWithLogits with the train pos_weight, threshold 0.5,
model selection on best validation strict, ties by validation IoU); the
paired-seed decision rule of `docs/hypotheses.md` (SUPPORTED only if the
mean is higher, every seed is higher, and the mean paired difference exceeds
the sd of the per-seed paired differences; NOT SUPPORTED for the opposite
direction; otherwise INCONCLUSIVE). Metric on every arm: per maze the
`strict` and `exact` flags of `cf.metrics.evaluate_mask` (strict = a
wall-free simple S→G path exactly of the label's length using every
predicted cell; exact = cell-wise match to the label mask), dataset means
via `cf.metrics.summarize`. All three curriculum sets are perfect mazes, so
`strict == exact` is asserted as `cf.train.evaluate_model` does. Primary
set: level-5 test (1,000 mazes). Transfer sets: probe7 and probe9 (1,000
each, never trained on). Four further commitments of the shared protocol:
(i) Family A's CF-Full is retrained under this protocol with the same seeds
(0, 1, 2) — its published 160-epoch numbers with validation-selected
horizons (`figures/curriculum_results/README.md`) are background only and
never feed a verdict, and A is also reported at its validation-selected T*
alongside the T = 8 point so no comparison is fixed at a single horizon;
(ii) the wall-clock budget rule of `docs/hypotheses.md` applies to X1 and
X2: a directional verdict is computed only if every arm in the comparison
finished its planned epochs under the shared budget
(`scripts/run_week2.py --budget-minutes`), otherwise INCONCLUSIVE, with
epochs_done and stopped_early reported per arm; (iii) every arm and set is
reported with per-seed values, means, sample sds, and the sd of the
per-seed paired differences alongside the verdict — never the verdict
alone; (iv) the split seed is fixed at 0 (the builders' default), so the
seed variation (0, 1, 2) covers training only, never the split draw, and
all arms see bit-identical splits.

**X1 — Family A vs Family D on the same mazes; the ladder's role.** The
registered experiment X1 compares CF-Full (Family A) with the Family D
transport model. The primary arm pair is A vs D+B (the recommended build);
the attribution arms are the learned D-alone and the B-alone ladder (BD-8
trained on its own), each reported alongside so the flow mechanism and the
ladder mechanism are separately attributable. This card registers the
identical arm list as the Family D card's X1
(`docs/specs/family_d_transport.md`), whose wording is the one of record.
Metric on every arm: strict and exact on the level-5 test, probe7, probe9,
under the shared protocol and the paired-seed rule. Reading of the accuracy
numbers: all three sets are perfect mazes on which the physics arm is at
ceiling by construction (see the D card), so the accuracy comparison is a
sanity and ceiling check and the payload for B is the mechanism contrast.
Mechanism measurement: the path-vs-dead-end contrast of the committed
share Σ_{k≥k_c} k·c_k / M over time (the same measurement applied to A's
2·c2/M), from `logits_at(record=True)` traces. Prediction registered in
advance: the B-alone ladder is not expected to beat the transport physics
on accuracy (its reach is 2 cells per step), but its committed-share
contrast should be sharper and more hysteretic than A's binary committed
share; for B the scientific payload of X1 is the mechanism contrast, not
the score gap.

**X2 — D physics-only vs D learned; the ladder's counterpart.** The
registered experiment X2 asks what learning adds to the physics inside the
transport model. The ladder's counterpart under the identical metric and rule
asks the same question about the local chemistry, with two arms: the
detailed-balance parameterisation (physics-constrained: energy guaranteed,
the network only shapes the landscape) vs free rates (unconstrained: no
energy), plus the parameter-free reference ladder (g ≡ 0, μ = 1, no learned
rates) as the physics-only floor. The reference ladder has zero learned
parameters and no training, so its score is one deterministic value; under
the paired-seed rule it enters as b_k identical across the three seeds,
whence sd(d) = sd(a) and the rule reduces to its three usual conditions.
Metric: strict and exact on the three sets. Mechanism measurements: the
fraction of rollouts where the free-rates arm increases V between
fast-phase micro-steps beyond the float tolerance 1e-6·max(1, |V|) (the
energy monitor B's arm is supposed to keep at zero; V needs equilibrium
weights, which the free-rates arm does not carry as a parameter, so they
are reconstructed from the learned rates by the detailed-balance recursion
Q_1 = 1, Q_{k+1} = a_k·Q_1·Q_k / b_{k+1}, which makes the same V formula
well-defined for both arms), the accuracy gap of each arm over the
g ≡ 0 floor, and rung-occupancy distributions of a path cell vs a dead-end
cell. Prediction registered in advance: free rates train no worse than
detailed balance on the curriculum, but lose the settled-state guarantee —
the free-energy curve is the point.

**X6 — Thinking longer.** Which mechanism improves when run longer? Arms:
frozen weights of every Family B arm, swept over T ∈ {1, 2, 4, 8, 12, 16,
24, 32} outer steps via `logits_at`. The rate refreshes are part of the
forward pass, so they continue every r_s = 2 outer steps for as long as the
rollout runs: past the training horizon T_train = 8 the chemistry keeps
relaxing under continually refreshed rates (frozen weights, live rates),
not under the last trained refresh. Metric: strict and exact on the
level-5 test, probe7 and probe9 at every T, with the training point
marked. Primary contrast (pre-registered, the only point that carries a
verdict): strict accuracy at T = 32 vs T = 8, paired by seed, as in H2 of
`docs/hypotheses.md` — SUPPORTED if the T = 32 point is higher under the
global rule, NOT SUPPORTED if the opposite direction meets the rule, and a
flat curve (neither direction meets the rule) is reported as consistent
with the prediction, descriptive only; every other T value is descriptive.
Settling measurements from the trace: the residual flux Σ|J| / Σ M, the free
energy V_t per step, and the relative rung change ‖c_{t+1} − c_t‖₁ / ‖c_t‖₁.
Multiplicity: this card adds the X1 and X2 primary contrasts and this one
X6 contrast to the six named comparisons the global rule was fixed for; no
alpha correction is applied (the same descriptive-rule stance as
`docs/hypotheses.md`), and no other evaluated point may carry a verdict.
Prediction registered in advance: the fast phase relaxes towards chemical
rest on its own, so unlike Family A — whose accuracy is known to drop past
its training horizon — the ladder's readout of an equilibrated state should
hold or improve past T_train, at the known risk of nucleation stalls
(undecided cells staying at rung 1 show up as flat low-confidence logits
removed by the threshold, not as wrong paths).

### Composition with Family D

In the combined build, the ladder's rate network receives the local flow
quantities of the transport layer as extra input channels — |Q|_cell, p and
D_cell from the Family D card's shape table — so the energy profile is
shaped by physical evidence: path cells carry flow and their ladders
condense; dead-end cells see the flow go dark and shed. The rate net's
input count becomes 6 + hid·K in the combined build (3 input channels +
3 flow channels + hid·K rung channels); the parameter table above counts
B alone, at 3 + hid·K. The transport layer
decides *where* the answer is; the ladder decides *how sure* each cell is
and supplies the provable per-cell energy. The experiments above are defined
on the arms separately so that every combined-model number can be attributed
to one of the two mechanisms.
