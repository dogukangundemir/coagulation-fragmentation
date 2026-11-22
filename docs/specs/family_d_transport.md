# Family D — transport coagulation-fragmentation: a conserved flow through the maze

Specification card written before implementation, per section 11.3 of
`docs/cf_algorithm_families.md`. It answers the eight axes of section 1, the
eight family headings (idea, physics, state, update, meaning in a maze,
information flow, guarantees and energy, risks), and then fixes every
implementation detail: tensor shapes, the discrete-time update with all
constants, source and sink placement, learned versus fixed parts, the
invariants to test, the readout, and the planned experiments with their exact
metrics. Conventions follow the existing code: input layout and the
`forward(x, T)` / `logits_at(Ts, record)` interface of `cf/models/cf.py`, the
split builders of `cf/data/curriculum.py`, and the strict/exact metrics of
`cf/metrics.py`.

## Plain-language summary

Think of the maze as a network of pipes. Every pair of adjacent free cells is
a pipe; walls are missing pipes. Water is poured in at the start cell and
drained at the goal cell, one unit per time step, forever. A pipe that carries
flow thickens; a pipe that carries nothing thins. The total amount of pipe
material never changes — material only moves between pipes. After enough
time, the only pipes left with real thickness are the ones on a route from
start to goal: a dead-end corridor has no drain anywhere inside it, so no
steady flow can enter it, so its pipe thins to nothing. On a maze with no
loops there is exactly one route, so the physics alone marks the solution
path. Nothing has to be trained for this to work on the curriculum mazes;
what training adds is edge costs, tie-breaking between equally good routes,
and robustness of the readout.

In coagulation-fragmentation language: tubes that carry flow gather material
and merge into one channel (coagulation), tubes without flow lose material
and fall apart (fragmentation), material mass is conserved, and the
conservation law itself — everything injected at the start must exit at the
goal — is the reasoning step, not a numerical safeguard. This is the
slime-mould *Physarum* mechanism.

## The eight axes

| # | Axis | Choice in Family D |
| --- | --- | --- |
| 1 | Particle | one unit of "tube material" per corridor edge; a particle's location is an edge of the free-cell grid graph |
| 2 | Size space | a continuous non-negative thickness **D_e** per edge (no per-cell size ladder here; that is Family B) |
| 3 | Coagulation kernel | not pairwise collisions: material is *redistributed towards flow-carrying edges* by the learned, positive, increasing adaptation f-theta of the local flow magnitude |
| 4 | Fragmentation | an edge whose flow decays loses material proportionally; it thins smoothly (with an epsilon floor so it can recover, never shattering hard) |
| 5 | Conservation scope | two exact laws: flow conservation at every non-terminal cell (Kirchhoff) and a fixed total tube material **M_tube** per maze |
| 6 | Transport | mass itself moves between neighbouring cells along edges, driven by the potential field; inner relaxation sweeps spread information many cells per outer step |
| 7 | Determinism | deterministic mean-field update; the tie-break variant draws random initial conductances |
| 8 | Energy and readout | dissipation **E = sum(Q_e^2 / D_e)** is minimized by the steady flow at fixed conductances (Thomson's principle); readout from per-cell flow magnitude, potential and thickness |

## Idea

Let mass actually move between neighbouring free cells, inject one unit per
step at the start and remove one unit per step at the goal. Conservation then
forces everything that leaves the source to arrive at the sink, and the only
cells that can carry it in a perfect maze are the cells of the solution path.
Channels that carry flow thicken and merge (coagulation); channels that carry
none thin out and vanish (fragmentation).

## Physics

Kirchhoff's current law on the free-cell grid graph, plus Physarum-style
adaptation (section 2.6 of the families document). Each edge *e* between
neighbouring free cells *u*, *v* is a resistor with conductance
**C_e = D_e / L_e** (thickness D_e, length L_e; resistance R_e = L_e / D_e),
carrying flow **Q_e = C_e (p_u − p_v)** where **p** is a per-cell potential.
The potentials solve the network equation: net outflow equals **+I** at the
start cell, **−I** at the goal cell, zero everywhere else. The adaptation rule
then moves a fixed material budget towards edges that carry flow.

## State

Per maze: one conductance per free edge, **D_e ≥ 0** (two maps on the grid:
horizontal and vertical edges; edges touching walls are fixed to 0), one
potential **p_u** per cell, and — in the recommended combined build — one
Family-B ladder per cell that reads the local flow. Total tube material
**M_tube = sum of all D_e** is fixed at all times.

## Update

One outer step, *t → t + 1* (all symbols as above; I is the source current,
fixed at I = 1.0; b is the per-cell source field with b_u = +I at the start
cell, −I at the goal cell, 0 elsewhere):

1. **Relax the potentials.** Either solve the network equation exactly
   (default on the curriculum grids: at most 161 free cells), or run K_rel
   two-colour Gauss–Seidel sweeps, each exactly minimizing the potential
   energy of section *Guarantees* for one checkerboard colour at a time:

   p_u ← (b_u + Σ_{v ~ u} C_uv p_v) / (Σ_{v ~ u} C_uv + eps_d·[Σ_{v ~ u} C_uv = 0])

   for every free cell u of one colour (its neighbours are all the other
   colour, so the updates inside a colour are independent); walls stay at
   p = 0. One sweep = red half + black half. The eps_d guard is applied
   only where the conductance sum is zero (isolated cells): a blanket
   eps_d in every denominator would shift the sweep fixed point to
   div_u = b_u − eps_d·p_u — a permanent conservation bias of order
   eps_d·max|p| (≈ 2e-5 at the potentials level 9 reaches, above the 1e-5
   tolerance) — and would break the exact coordinate-minimizer property of
   the half-sweep.
2. **Compute flows.** Q_e = C_e (p_u − p_v), signed, positive eastward or
   southward; antisymmetric by construction (flow out of u into v is minus
   the flow out of v into u, in exact arithmetic and in float32).
3. **Redistribute material (coagulation / fragmentation).** For every free
   edge e, with |Q_e| its flow magnitude and u_e a small learned per-edge
   feature:

   f_e = f_theta(|Q_e|, u_e) = softplus(w_q |Q_e| + w_u · u_e + b_f) + eps_f
   D_e ← M_tube · D_e f_e / (Σ_{e'} D_{e'} f_{e'})

   Edges that lost flow keep a smaller share of the material (fragmentation,
   smooth); edges that keep flow gather it (coagulation into one channel).
   The sum of all D_e is exactly M_tube after the step, and no denominator
   guard is needed: f_e ≥ eps_f > 0 and Σ_e D_e = M_tube > 0 always, so
   Σ_e D_e f_e ≥ eps_f · M_tube > 0. (An earlier draft carried an eps_norm
   guard in this denominator; since the bound above holds always, the guard
   was dropped — with it, the sum would drift below M_tube by
   eps_norm / Σ D f per step, of order 1e-6 over 64 steps at the default
   initialisation, and the "exactly" claim would be false.)
4. **Re-impose the wall structure.** D_e ← D_e on free edges, D_e = 0 on any
   edge with a wall endpoint (exact, at every step and at initialisation).

## What it means in a maze

**The argument in one paragraph.** Take a perfect maze (a tree of corridors),
a unit source at S and a unit sink at G. Cut any corridor edge that is not on
the S–G path: it separates a part of the maze containing neither S nor G.
Inside that part there is no source and no sink, so by flow conservation the
net flow across the cut edge must be zero. The steady flow is therefore
non-zero exactly on the solution path — every path edge carries exactly
|Q_e| = I = 1 in a cell-chain where the path is a series connection — and
zero in every dead end. Conservation is the reasoning, not a safeguard.

**Worked numbers (level 5).** A level-5 curriculum maze is 11×11 cells: 25
lattice cells (odd, odd) plus 24 open passage cells joined by the perfect-maze
check (`passages == n^2 − 1` in `cf/data/curriculum.py`) = 49 free cells, 72
walls, and 48 free edges (each passage cell contributes the two unit edges to
its lattice neighbours). The free-cell grid graph is the subdivided maze tree.
M_tube defaults to 48, so the uniform initial thickness is D_e = 1. Suppose
the solution path visits 19 cells (18 edges). At the uniform start the steady
dissipation (defined under *Guarantees*) is the sum over path edges of
L_e/D_e = 18 × 1 = 18. After the material has fully coagulated onto the
path, each path edge holds 48/18 ≈ 2.67 and the dissipation falls to
18/2.67 = 6.75. For reference: level 7 (15×15) has 97 free cells and 96 free
edges; level 9 (19×19) has 161 free cells and 160 free edges.

**Levels and loops.** On a perfect maze the flow argument alone produces the
answer; the material dynamics only sharpens it (all flow is on the path after
the very first relaxation). Where the coagulation/fragmentation dynamics is
irreplaceable is mazes with loops: flow splits over parallel routes, and the
adaptation concentrates it on the shortest one. Concentration has a proven
counterpart, but for a different rule: the Bonifaci/Mehlhorn/Varma Physarum
theorem covers the update dD_e/dt = |Q_e| − D_e, in which tube material is
created and destroyed. The rule implemented here (update step 3) is the
mass-conserving normalised redistribution D_e ← M_tube · D_e f_e / Σ(D f),
inspired by that model, and no convergence proof of that kind is claimed
for it.

## How information moves

Through the potentials. One relaxation sweep updates every free cell from its
four neighbours, so information crosses the grid diffusively, like the
Green's function of a random walk: each sweep is one diffusion-like
relaxation step whose influence spreads over a distance that grows with the
square root of the number of steps, so the expected reach after r sweeps is
on the order of √r cells. Exact
solves have global reach in one step. The learned parts are strictly local
(3×3 convolutions on edge features). Thinking longer is built in: more sweeps
bring the potentials closer to the exact Kirchhoff solution, so extra
test-time computation refines the flow rather than disturbing it — dead ends
drain their transient flow, and the flow map darkens off the path.

## Guarantees and energy

All of the following hold for any network weights, trained or random:

- **Exact flow conservation.** At the exact solve, net outflow equals b_u at
  every free cell up to the linear-solve tolerance; terminals are the only
  cells with non-zero divergence, and b sums to zero (source = sink).
- **Antisymmetric flows.** Q_e is stored once per edge with a sign
  convention, so Q from u to v equals −Q from v to u exactly by
  construction.
- **Tube material conserved.** Σ_e D_e = M_tube exactly after every
  material step (the update is a normalized redistribution whose
  denominator is bounded below by eps_f·M_tube > 0, so no guard is needed
  and the sum never drifts); D_e ≥ 0 always.
- **Zero flow in dead ends.** On a perfect maze with the exact solve, every
  free edge off the solution path carries exactly Q_e = 0 in the steady
  state and every path edge carries |Q_e| = I.
- **Dissipation energy (Thomson's principle).** Freeze the conductances and
  define the dissipation of any flow field q as
  E(q) = Σ_e q_e² / D_e. Among all *conserved* flows (net outflow b_u at
  every cell) the steady Kirchhoff flow minimizes E. The layer's potential
  energy Ψ(p) = ½ Σ_e Q_e(p)²/C_e − Σ_u b_u p_u (= ½ Σ_e C_e Δp_e² −
  Σ_u b_u p_u, which reduces to the older ½ Σ_e Q_e²/D_e − Σ_u b_u p_u
  form only for unit lengths L_e ≡ 1) decreases monotonically with
  each two-colour sweep (each half-sweep is an exact coordinate-wise
  minimization for its colour) and is minimized exactly at the Kirchhoff
  potential, where the induced flow attains the Thomson minimum. E* = Σ_e
  Q_e²/D_e of the steady flow is the layer's natural energy for unit lengths
  only, L_e ≡ 1 (with trained lengths the corresponding quantity is
  Σ_e Q_e²/C_e): it reports how
  "settled" the flow is and, on trees with uniform lengths, decreases as
  material coagulates onto the path (see the invariant tests).

## Risks

1. **Slow diffusion.** Finite relaxation needs sweeps that scale with the
   square of the corridor length; the exact solve removes this on the
   curriculum sizes but does not scale to very large grids (mitigations:
   multigrid or a few Krylov steps inside the layer later).
2. **Ties.** When several equally short routes exist, flow splits evenly and
   the strict metric (one specific path) can fail; the random-initial-
   conductances variant breaks ties, and several samples give voting.
3. **"This is a classical algorithm."** Answer honestly: report the
   physics-only version (no learned parameters) as a reference arm, and
   measure what learning adds — costs for mazes with loops, robustness of
   the readout, tie-breaking. On the perfect-maze curriculum the pure
   physics arm is expected to be at ceiling; the interesting learning
   effects are measured at finite relaxation budgets and, later, on non-
   perfect grids.

**Role.** The recommended transport mechanism: it gives conservation a task
meaning, makes "thinking longer" natural, and matches the Physarum biology.
Compose it with Family B (`docs/specs/family_b_ladder.md`) per cell reading
the local flow.

## Implementation

Planned module: `cf/models/transport.py`, class `TransportCFNet`, exposing
the CFNet interface used by `cf/train.py`: `forward(x, T)` returning logits
`(B, H, W)`, plus `initial_state`, `step`, `read`, and `logits_at(Ts,
record)`; with `record=True` the trace stores, per outer step, the D maps,
|Q| maps, the dissipation E_t, and the flow-conservation residual
`max_u |div_u − b_u|`.

### Tensor shapes on a B x H x W grid

Curriculum mazes are square, H = W = 2n+1 (levels 5, 7, 9 → 11, 15, 19).

| Name | Shape | Meaning |
| --- | --- | --- |
| x | (B, 3, H, W) | input channels [walkable, start one-hot, goal one-hot], as produced by `cf.data.curriculum.maze_to_channels` |
| walk | (B, H, W) | x[:, 0]: 1 free cell, 0 wall |
| Dh | (B, H, W−1) | horizontal edge conductance: Dh[b, r, c] on the edge between (r, c) and (r, c+1) |
| Dv | (B, H−1, W) | vertical edge conductance: Dv[b, r, c] on the edge between (r, c) and (r+1, c) |
| Lh, Lv | same shapes as Dh, Dv | learned edge lengths (resistance R_e = L_e / D_e) |
| Eh, Ev | same shapes, 0/1 | free-edge masks: Eh = walk[:, :, :-1] * walk[:, :, 1:], Ev = walk[:, :-1] * walk[:, 1:] |
| b | (B, H, W) | source field: +I on the start cell, −I on the goal cell, 0 elsewhere |
| p | (B, H, W) | per-cell potential; fixed to 0 in wall cells |
| Qh, Qv | same shapes as Dh, Dv | signed flows; Qh positive = eastward (from (r, c) to (r, c+1)), Qv positive = southward |
| |Q|_cell | (B, H, W) | Σ of |Q| over the ≤4 incident edges of a cell (padded shifts) |
| D_cell | (B, H, W) | mean conductance of the ≤4 incident edges |
| logits | (B, H, W) | per-cell path logits; exactly −20.0 in walls (the wall-aware convention of `cf/models/cf.py`) |

Edges into or inside walls are fixed to 0 in Dh, Dv at initialisation and
re-multiplied by Eh, Ev after every material step, so they stay exactly 0.
Wall cells never update and hold p = 0. Corollary: no edge ever crosses a
wall, so no mass can leak across one — flow moves through free cells only.

### Discrete-time update with all constants

Constants (fixed, not learned):

| Constant | Default | Role |
| --- | --- | --- |
| I | 1.0 | source current (unit mass per step) |
| T | 8 outer steps at training | rollout length |
| K_rel | 16 sweeps per outer step | relaxation budget in sweep mode |
| eps_d | 1e-6 | denominator guard, applied only where Σ_{v~u} C_uv = 0 (isolated cells). It must not sit in every cell's denominator: that would bias the sweep fixed point by div_u = b_u − eps_d·p_u (≈ 2e-5 at level-9 potentials) and break the half-sweep's exact coordinate-minimizer property. On validated mazes no free cell is isolated (S and G are connected lattice cells, and isolated non-terminals have b_u = 0) |
| eps_f | 1e-3 | floor of the adaptation function: an edge with flow keeps a tiny growth term, so a thinned edge can recover and gradients keep flowing; it also bounds the material-share denominator below by eps_f·M_tube, so no guard is needed there |
| M_tube | number of free edges per maze (48 / 96 / 160 on levels 5 / 7 / 9) | fixed total material; uniform mean thickness 1.0 |
| d_0 | 1.0 | initial thickness on free edges (M_tube / free-edge count) |
| sigma_tie | 0.1 | random-start variant: D_e(0) = M_tube · exp(sigma_tie · u_e) / Σ_{e'} exp(sigma_tie · u_{e'}), u_e ~ U(−1, 1) per free edge — the draw is normalized by its own sum so Σ_e D_e(0) = M_tube exactly (an unnormalized draw would start the material sum off M_tube at t = 0); breaks ties between equally short routes (the random-start level of the stochastic layer) |

Outer step (t → t + 1), in order:

1. **Potentials, exact mode (default on the curriculum grids).** Free-cell
   index (N ≤ 161 unknowns on the curriculum). Weighted graph Laplacian
   A: A_uu = Σ_{e ∋ u} C_e, A_uv = −C_e with C_e = D_e / L_e; solve
   A p = b with the gauge pinned by fixing the potential of the first free
   cell (in row-major order) to 0 (b sums to zero, so the system is
   consistent). Batched `torch.linalg.solve` per outer step, evaluated in
   chunks of 100 mazes, in float64: the pinned Laplacian's condition number
   grows like N² (N up to 161), so a backward-stable float32 solve leaves
   residuals of order eps·‖A‖·‖p‖ ≈ 4e-5 — above the 1e-5 conservation and
   dead-end tolerances on level 9 — while a 161×161 float64 system is
   trivial in cost. The solve, the flows and the divergence are computed
   in float64 and cast to float32 only for the material step and the
   readout.
2. **Potentials, sweep mode (for large grids and the "thinking longer"
   ablation).** K_rel two-colour Gauss–Seidel sweeps from the current p:
   p_u ← (b_u + Σ_{v ~ u} C_uv p_v) / (Σ_{v ~ u} C_uv + eps_d·[Σ_{v ~ u} C_uv = 0])
   applied to all red cells, then all black cells (checkerboard masks over
   the H×W grid; wall cells untouched; eps_d only where the conductance
   sum is zero, as in the update spec above). Each half-sweep is one
   vectorized neighbourhood gather.
3. **Flows.** Qh = (Dh / Lh) ⊙ (p[:, :, :-1] − p[:, :, 1:]);
   Qv = (Dv / Lv) ⊙ (p[:, :-1, :] − p[:, 1:, :]). Divergence at cell (r, c):
   div = Qh[r, c] − Qh[r, c−1] + Qv[r, c] − Qv[r−1, c] (zero padding at the
   borders); conservation means div = b.
4. **Material.** With edge features u_e = mean over the edge's two endpoint
   features h = Conv3x3(3 → 4)(x) (the only learned conv before the
   readout): f_e = softplus(w_q |Q_e| + w_u · u_e + b_f) + eps_f with
   w_q = softplus(raw) ≥ 0 (so f is non-decreasing in the flow), then
   D_e ← M_tube · D_e f_e / Σ D f, applied jointly over both
   edge maps; then D ← D ⊙ (Eh, Ev).
5. **Readout.** logits = Conv1x1([x, |Q|_cell, p, D_cell] → 1), walls filled
   with exactly −20.0.

The physics-only arm (experiment X2) sets L_e ≡ 1, f_e = |Q_e| + eps_f, and
replaces the learned readout with the fixed threshold |Q|_cell ≥ I/2 = 0.5;
it has zero learned parameters. The learned arm initializes the length head
with bias log(e − 1) ≈ 0.5413 after softplus, so every L_e starts at 1.0.

### How the source and sink are placed

From the input channels, not from anything learned: b = I · (x[:, 1] −
x[:, 2]) where x[:, 1] is the start one-hot ('S' in the ASCII maze) and
x[:, 2] the goal one-hot ('E'). I = 1.0 is fixed and never trained. The
start and goal cells are otherwise ordinary cells of the network: they are
exactly the two free cells whose net outflow is allowed to be non-zero
(+I and −I); every other free cell has zero net outflow. The dataset checks
guarantee one start and one goal, both on lattice cells of a connected
perfect maze, so the linear system is always solvable.

### What is learned and what is fixed

| Learned (small, strictly local) | Fixed (the physics) |
| --- | --- |
| edge lengths L_e = softplus(Conv3x3-head(x)) | source current I, the source/sink placement, the whole relaxation and the exact solve |
| adaptation f_theta (w_q, w_u, b_f) | the material normalization with M_tube (exact conservation) |
| the 1×1 readout head | the wall masks (edges into walls exactly 0), positivity and antisymmetry of the flows |
| the initial-thickness option d_0 and the tie noise sigma_tie | the ladder (none), the stoichiometry of the material step |

The conservation laws and the wall structure are not parameterized at all:
they hold for every setting of the weights, in the same way mass conservation
holds for every CFNet weight in `cf/models/cf.py`.

### Invariants to test

Mirroring the style of `tests/test_cf_invariants.py` (float32 tolerances;
NEG_EPS = −1e-7; relative drift tolerance 1e-5; the exact solve and the
flow/divergence computation run in float64 per the implementation above, so
the 1e-5 tolerances below are not float32-limited):

1. **Flow conservation at non-terminal cells.** In exact mode: |div_u − b_u|
   < 1e-5 · (1 + |b_u|) at every free cell, and div = (+I, −I) at the two
   terminals. In sweep mode the potential energy Ψ (defined under
   *Guarantees*) decreases monotonically per half-sweep (each half-sweep is
   an exact coordinate-wise minimization for its colour; tolerance 1e-6
   relative); the residual max_u |div_u − b_u| is logged per half-sweep as
   a diagnostic only — red-black Gauss–Seidel decreases Ψ monotonically,
   not the max-norm residual, which rises on many individual half-sweeps
   (on a random 15×15 grid, 58 of 60 half-sweeps increased it).
2. **Antisymmetric Q.** By construction (one signed map per edge); test the
   consistency of the divergence with both edge maps so an indexing bug
   (transposed shifts) cannot hide.
3. **Tube material conserved.** Σ(Dh) + Σ(Dv) = M_tube before and after every
   material step, relative drift < 1e-5 over 64 outer steps, for any
   weights and both arms.
4. **Non-negative conductances.** Dh, Dv ≥ NEG_EPS at initialisation and
   after every step; on wall-adjacent maps the entries are exactly 0.
5. **Zero net flow in dead ends at steady state.** On validated perfect
   mazes (synthetic grids via the `maze_synthetic` helpers plus curriculum
   samples) in exact mode: |Q_e| < 1e-5 off the label path and
   | |Q_e| − I | < 1e-5 on it, for every free edge. In sweep mode the same
   test passes once K_rel is large enough and quantifies how the off-path
   transient decays with the sweep count.
6. **Dissipation monotone under fixed conductances.** Freeze the material
   step (conductances fixed): (a) with the exact solve, E(Q*) ≤ E(q) for
   100 random conserved flows q per maze on loopy grids (add walls to
   curriculum samples or use synthetic grids with cycles; Thomson spot
   check). On a perfect maze this check is vacuous — the free-cell graph
   is a tree, the conserved flow with divergence b is unique, so every
   random draw equals the steady flow and the inequality holds with
   equality. Construction of q: fix a spanning tree of the free-cell
   graph, push the unit current along the tree path from S to G, then add
   independent circulations — for each non-tree edge (its fundamental
   cycle), a random signed amount of flow around that cycle; (b) from a zero potential, the potential energy
   Ψ decreases monotonically per half-sweep (tolerance 1e-6 relative) and
   converges to its minimum at the Kirchhoff potential; (c) coupled steps on
   a perfect maze with uniform lengths and the feature-free adaptation
   (physics arm): the per-outer-step dissipation E_t of the exact flow is
   non-increasing. Sketch: every path edge carries |Q| = I and every
   off-path edge carries 0 exactly, so the material share on the path,
   m_t, updates as m_{t+1} = M_tube · m_t / (m_t + c_off (M_tube − m_t))
   with c_off = f(0) / f(I) ≥ 0, which is non-decreasing in t, and
   E_t = (2p)² / m_t for a path of 2p unit edges, so E_t falls as m_t
   rises. With learned edge features this is monitored but not claimed as a
   theorem.

### Readout

Primary (learned arm): the 1×1 convolution over [x, |Q|_cell, p, D_cell]
produces per-cell logits, thresholded at 0.5 exactly like the existing
models; evaluation is unchanged `cf.train.evaluate_model(model, d, T)`.
Physics arm: cell is on the path iff |Q|_cell ≥ I/2 (on a perfect maze,
path cells see two incident unit flows, or one at the endpoints, and
off-path cells see flow only during the off-path transient). The physics
arm produces no logits, so it cannot feed `cf.train.evaluate_model`
unchanged (that path applies a sigmoid to logits and thresholds at 0.5);
it therefore emits a logit of +20 where |Q|_cell ≥ I/2 and −20 otherwise
(walls −20), after which the standard evaluation applies unchanged.

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
mean is higher, every seed is higher, and the mean paired difference
exceeds the sd of the per-seed paired differences; NOT SUPPORTED for the
opposite direction; otherwise INCONCLUSIVE). Metric: per maze the `strict`
and `exact` flags of `cf.metrics.evaluate_mask` (prediction = cells above
threshold; strict = a wall-free simple S→G path exactly of the label's
length using every predicted cell; exact = cell-wise match to the label
mask), dataset means via `cf.metrics.summarize`. All three curriculum sets
are perfect mazes, so `strict == exact` is asserted as `cf.train.
evaluate_model` does. Primary set: level-5 test (1,000 mazes). Transfer
sets: probe7 and probe9 (1,000 each, never trained on). Four further
commitments of the shared protocol: (i) Family A's CF-Full is retrained
under this protocol with the same seeds (0, 1, 2) — its published
160-epoch numbers with validation-selected horizons
(`figures/curriculum_results/README.md`) are background only and never feed
a verdict, and A is also reported at its validation-selected T* alongside
the T = 8 point so no comparison is fixed at a single horizon; (ii) the
wall-clock budget rule of `docs/hypotheses.md` applies to X1 and X2: a
directional verdict is computed only if every arm in the comparison
finished its planned epochs under the shared budget
(`scripts/run_week2.py --budget-minutes`), otherwise INCONCLUSIVE, with
epochs_done and stopped_early reported per arm — D's exact solve (a
batched 161×161 solve per outer step, 8 steps per forward) costs far more
wall-clock per epoch than A's convolutions, so epoch-matched is not
compute-matched and wall-clock is reported per arm too; (iii) every arm
and set is reported with per-seed values, means, sample sds, and the sd
of the per-seed paired differences alongside the verdict — never the
verdict alone; (iv) the split seed is fixed at 0 (the builders' default),
so the seed variation (0, 1, 2) covers training only, never the split
draw, and all arms see bit-identical splits.

**X1 — Family A vs Family D on the same mazes** (does giving conservation a
task meaning help?). Primary arm pair: CF-Full (Family A, the existing
`cf/models/cf.py` baseline) vs the recommended D + B build, both at T = 8
under the shared protocol; this registration is the one of record and the
Family B card registers the identical list. Attribution arms reported
alongside: the learned D-alone (TransportCFNet), the B-alone ladder, and a
widened readout/cost-head D arm matched to 10,177 parameters, so the score
gap is never purely about capacity and each mechanism is separately
attributable. The learned D parameter count is in fact tiny — about 200 by
the module list above (edge-feature encoder 112, length heads 74,
adaptation 6, readout 7) — which is exactly why the capacity-matched arm
matters. Metric: strict and exact on the level-5 test, probe7, probe9.
Reading of the accuracy numbers: all three sets are perfect mazes and the
physics argument guarantees the physics-only arm solves them exactly in
exact-solve mode, so the level-5/7/9 accuracy comparison is a sanity and
ceiling check, not the test of the heading question — a SUPPORTED "D > A"
would measure the built-in global linear solve against A's 2-cells-per-step
locality. The payload is the contrast curve below and the finite-K_rel
regime of X2; the sets where the heading question is actually testable
are the loop-maze and Maze-hard grids planned after the curriculum.
Mechanism measurement: the flow contrast
C_t = mean |Q| over label-path edges − mean |Q| over non-path free edges,
per outer step (from the recorded trace), next to A's committed share
2·Σc2/ΣM measured identically on path vs dead-end cells. Prediction
registered in advance: D is expected to dominate on the curriculum, with
the physics argument (zero flow in dead ends) explaining why; the
scientific payload is the contrast curve, not the score gap.

**X2 — D physics-only vs D learned** (what does learning add to the
physics?). Arms: the physics-only D (L ≡ 1, f = |Q| + eps_f, fixed
threshold readout, zero learned parameters, no training) vs the learned D.
The physics-only arm has zero learned parameters and no training, so its
score is one deterministic value; under the paired-seed rule it enters as
b_k identical across the three seeds, whence sd(d) = sd(a) and the rule
reduces to its three usual conditions. Metric: strict and exact on the
three sets, same protocol for the learned arm. Mechanism measurements:
(i) the deviation of the learned lengths from uniform: mean |L_e − 1| and
its correlation with local maze structure — labelled exploratory, so it
cannot be presented post hoc as planned;
(ii) the accuracy-vs-K_rel curve of the physics arm at K_rel ∈
{4, 16, 64, exact} — how far finite relaxation alone carries the task;
(iii) for the learned arm, whether training ever falls below the physics
ceiling (a robustness check of the readout). On the perfect-maze
curriculum both arms are expected to sit at or near ceiling in exact mode;
the informative regime is finite K_rel, and later the non-perfect grids,
where flow splits and the adaptation law must choose.

**X6 — Thinking longer** (which mechanism improves when run longer?). Arms:
frozen weights of the learned D arm, swept over T ∈ {1, 2, 4, 8, 12, 16,
24, 32} outer steps via `logits_at`, and over the relaxation budget K_rel ∈
{4, 16, 64, exact} at T = 1 (thinking longer = more sweeps, since after the
flow settles further outer steps only re-concentrate material on a tree).
The learned arm trains in exact-solve mode (the default on the curriculum
grids), so the marked training point is (T = 8, exact solve) and the
K_rel grid is a sweep-mode probe around an exact-trained checkpoint.
Metric: strict and exact on the three sets at every (T, K_rel) pair, with
the training point marked. Primary contrast (pre-registered, the only
point that carries a verdict): strict accuracy at T = 32 vs T = 8 at the
exact solve, paired by seed, as in H2 of `docs/hypotheses.md` — SUPPORTED
if the T = 32 point is higher under the global rule, NOT SUPPORTED if the
opposite direction meets the rule, and a flat curve (neither direction
meets the rule) is reported as consistent with the prediction,
descriptive only; every other grid point — the remaining T values and all
K_rel pairs — is descriptive. Multiplicity: this card adds the X1 and X2
primary contrasts and this one X6 contrast to the six named comparisons
the global rule was fixed for; no alpha correction is applied (the same
descriptive-rule stance as `docs/hypotheses.md`), and no other evaluated
point may carry a verdict. Settling measurements from the trace: the
flow-conservation residual (sweep mode), the relative flow
change ‖Q_{t+1} − Q_t‖₁ / ‖Q_t‖₁ per outer step, and the dissipation E_t.
Prediction registered in advance: accuracy is flat or improving in T and
K_rel (the relaxation converges to the Kirchhoff solution; no overthinking
mechanism exists in the dynamics once the flow is settled), in contrast to
the known accuracy drop of A past its training horizon.

### Composition with Family B

The recommended build is D + B: each free cell runs a Family-B commitment
ladder whose rate network reads, among its inputs, the local flow magnitude
|Q|_cell, the potential p and the cell's conductance, so the ladder's
condensation is driven by the physical flow (path cells condense, dead-end
cells shed). The split of responsibilities — physics carries the path,
chemistry carries per-cell confidence and its energy — is exactly the
combined card recommended in section 11 of the families document; the
ladder's own card fixes its details.

## Physics-only results

The registered physics-only arm has been run with no learning: uniform
lengths L_e = 1, fixed adaptation f_e = |Q_e| + eps_f, the fixed
threshold readout, 0 learned parameters (raw and effective), no training
(0 epochs, 0 seeds — a single deterministic run, so the numbers are exact
values with no seed spread), CPU only (results.json records device
"cpu"), float64 solve, on the shared seeded splits: level-5 test and the
level-7/9 probes, 1000 mazes each (`results/transport_physics/results.json`).

Test strict accuracy per maze set, at the training horizon T = 8 and at
the validation-selected horizon T* ("at T = 8 / at T*"):

| relaxation per outer step | level 5 test | probe 7 | probe 9 |
| --- | --- | --- | --- |
| exact solve | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Gauss-Seidel (two-colour), 1 sweep | not run | not run | not run |
| Gauss-Seidel (two-colour), 4 sweeps | not run | not run | not run |
| Gauss-Seidel (two-colour), 16 sweeps | not run | not run | not run |

Reading. As pre-registered under X1/X2, on the perfect-maze curriculum
this arm sits at ceiling by construction: the mazes are trees, so with the
exact solve the entire unit current is on the unique solution path at the
first outer step already (Kirchhoff flow conservation on a tree, for any
conductances), and strict = exact = 1.000 on all three sets at every
horizon measured (T in {1, 2, 4, 8, 16, 32, 64}; largest off-path flow
share 8.3e-8). No validation sweep was run and none is needed for the
table: every measured horizon is at 1.000, and the arm's flow is the same
unique conserved solution at every outer step (no horizon dependence), so
the entry at T* is 1.000 whichever horizon the selection rule picks. The horizon sweep is
therefore a sanity and ceiling check, not a test; the informative regime
is the finite-sweep one (accuracy vs local relaxation budget — the
information speed per grid 5/7/9), and no k-sweep Gauss-Seidel
(two-colour) or other finite-sweep
run exists in results.json, so those cells are marked not run (absent
measurement, not a measured zero). Run totals per 1000-maze set:
11.2 s at level 5, 40.6 s at level 7, 176.9 s at level 9 (one exact solve
per outer step dominates; where the fixed adaptation drives a maze's
solve past the conservation tolerance its flow is held at the last
conservation-valid value, which on a tree is the unique conserved flow).
Per-horizon tables and the full protocol: `docs/results/transport_physics.md`.
Figure-by-figure walkthrough of the mechanism (flow sweeps, pipe thickening,
conservation monitors): `figures/transport_mechanism/README.md`.

