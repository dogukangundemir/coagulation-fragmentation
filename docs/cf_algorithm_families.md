# CF Algorithms: Mechanisms, Variants and What They Mean

Nov 22, 2026

"Coagulation-fragmentation" is not one algorithm but a family of mechanisms. They differ in what a particle is, which particles may merge, how clusters break, what is conserved, and how information moves. This document describes seven CF algorithm families in depth: the physics behind each, its update rule, what it means for solving a maze, what it guarantees, and what can go wrong. It then compares them with each other; conventional baselines play no role here.

Short answer: our current model (Family A) is the simplest and weakest form of CF: per-cell chemistry with no mass moving between cells. The most promising next mechanism is transport CF (Family D), where conserved mass flows through free cells from start to goal. On a perfect maze, a conserved flow from S to G can only pass through the solution path, so conservation itself does the reasoning. Combined with a commitment ladder (Family B) per cell and stochastic sampling (Family G) for voting, this gives a CF algorithm whose physics carries the explanation. Families E and F (fragments, global particles) are the route to Sudoku and NLP.

Each family section answers the same eight questions, so they can be compared line by line: idea, physics, state, update, what it means in a maze, how information moves, guarantees and energy, risks.

## 1. What any CF algorithm must specify

Every CF algorithm is fixed by eight choices. Two models that share the name "coagulation-fragmentation" can behave completely differently if they differ on any of them, so these axes are the vocabulary for the rest of the document.

| # | Axis | Question it answers | Range of choices |
| --- | --- | --- | --- |
| 1 | Particle | What is one unit of "stuff"? | evidence unit in a cell; a path fragment; a global hypothesis; a token |
| 2 | Size space | What does a cluster's size mean, and how large can it get? | 2 sizes; sizes 1..K; continuous; a set of cells |
| 3 | Coagulation kernel K(i, j) | Which pairs merge, and how fast? | constant; additive (i + j); multiplicative (i · j); learned from features; local or global |
| 4 | Fragmentation | How do clusters break? | reverse of merging (binary); shedding one monomer at a time; shattering into many pieces |
| 5 | Conservation scope | What total stays fixed? | mass per cell; mass per maze; mass with a source and sink (flow) |
| 6 | Transport | How does information or mass move in space? | only through the rate network; mass flows between neighbouring cells; global (attention) |
| 7 | Determinism | Is the update a mean-field ODE or a random process? | deterministic rates; stochastic events (sampling) |
| 8 | Energy and readout | Is there a quantity that decreases, and how is the answer read? | free energy under detailed balance; residual flux; none · read the size distribution, the committed share or the flow |

Three of these matter most for reasoning:

- The kernel decides what can combine with what. It is where the network's knowledge of the task lives.
- Transport decides how far evidence can travel per step, which sets how many steps a long maze needs.
- Conservation scope decides what the constraint means. Mass per cell is only a numerical safeguard; mass that flows from S to G is itself a statement about paths (Family D).

## 2. Physics toolbox

Six results from the mathematics and physics of coagulation-fragmentation decide which neural CF algorithms are stable, which have an energy, and which can find paths. Each is stated here with what it means for us.

### 2.1 Discrete Smoluchowski equation with fragmentation

With c_k the concentration of clusters of size k, a symmetric coagulation kernel K_ij and a fragmentation rate F_{j → k}:

$$
\frac{dc_k}{dt} = \tfrac12 \sum_{i+j=k} K_{ij} c_i c_j \; - \; c_k \sum_{j} K_{kj} c_j \; + \; \sum_{j>k} F_{j\to k}\, c_j \; - \; c_k \sum_{i<k} F_{k\to i}
$$

Every event preserves the total number of monomers, so M = Σ k c_k is conserved formally. For us: this is the general template; every family below is a special case or an extension of it.

### 2.2 Kernels and gelation: a warning for learned kernels

The kernel's growth with size decides whether the system stays well behaved. For the multiplicative kernel K(x, y) = x · y, the Smoluchowski equation loses mass in finite time: clusters of infinite mass appear, which is called gelation (Fournier 2025). The multiplicative case is also the cluster-size law of Erdős-Rényi random graphs (arXiv 1902.07686). For homogeneous kernels K(i, j) = (ij)^α classical theory predicts gelation for α > 1/2, while subcritical kernels conserve mass for all times (arXiv 2601.18806); constant and additive kernels do not gel.

For us: a learned kernel that grows with cluster size can produce runaway commitment, where all mass rushes into the largest allowed size at once. In a truncated network this appears as mass piling up at the maximum size, an early irreversible "decision". Learned kernels should be bounded (for example a sigmoid times a size-independent or sub-linear factor), or fragmentation must grow with size to balance them.

### 2.3 Fragmentation types

| Type | Event | Reasoning meaning |
| --- | --- | --- |
| Binary (reverse of merging) | A_(i+j) → A_i + A_j | split a hypothesis into two parts that can be revised separately |
| Monomer shedding | A_k → A_(k−1) + A_1 | lower confidence one unit at a time; gradual doubt |
| Shattering | A_k → k · A_1 | discard a hypothesis completely; full reset |

### 2.4 Becker-Döring: only monomers move, and there is a free energy

In the Becker-Döring equations clusters change size only by gaining or losing one monomer: A_k + A_1 ⇌ A_(k+1). With rates a_k (gain) and b_(k+1) (loss), the net flux from size k to k + 1 is J_k = a_k c_1 c_k − b_(k+1) c_(k+1). Ball, Carr and Penrose proved existence and uniqueness and showed that the density is a conserved quantity (Commun. Math. Phys. 1986); their analysis of long-time behaviour rests on a Lyapunov function (Heriot-Watt record). Under detailed balance, with coefficients Q_1 = 1 and Q_i a_i = Q_(i+1) b_(i+1), equilibria form a one-parameter family Q_i z^i (Murray & Pego), and for general coagulation-fragmentation equations with detailed balance the function below is a Lyapunov function:

$$
V(c) = \sum_j c_j \left( \log \frac{c_j}{Q_j} - 1 \right)
$$

It can be read as a free energy that dissipates along solutions (Schlichting). Above a critical density the excess mass forms ever larger clusters: condensation.

For us: a CF algorithm built on Becker-Döring kinetics with detailed balance has a provable energy that decreases at every step, the CF counterpart of AKOrN's and WONN's energy, usable for monitoring convergence and for voting. The critical density gives a natural decision threshold: below it, cells stay undecided; above it, they commit.

### 2.5 Stochastic coagulation: the Marcus-Lushnikov process

The Marcus-Lushnikov process is a finite stochastic system of coalescing particles, each characterised by its mass, where each pair of masses x and y merges at rate K(x, y); under suitable assumptions it converges to the Smoluchowski equation as the number of particles grows (Jacquot). Rates of convergence are known for homogeneous-like kernels (Cepeda & Fournier).

For us: the deterministic CF update we use is the mean-field limit of a random process. Sampling that process instead gives many different trajectories from one input, which is what AKOrN and WONN exploit with energy voting (Family G).

### 2.6 Transport networks that find paths: Physarum

In Nakagaki's experiment a maze was covered with pieces of the slime mould Physarum (pieces reunite when brought close together) and fed at two locations; after a few hours the slime retracted to the shortest path, in every maze tried (Becchetti et al.). Tero, Kobayashi and Nakagaki modelled this as a tube network whose shape adapts to the flow (J. Theor. Biol. 2007): Physarum is an electrical network with time-varying resistors (Bonifaci et al.), with a unit source at one food site and a unit sink at the other (Mehlhorn, talk). For this exact update (dD_e/dt = |Q_e| − D_e, in which tube material is created and destroyed) the Bonifaci/Mehlhorn/Varma theorem proves convergence to the shortest path between the two sources, independently of the network's structure and the initial mass distribution; our Family D uses a mass-conserving normalised redistribution (D ← M D f / Σ D f) inspired by it, for which we claim no such proof.

The model, in its common form (D_e the conductance of edge e, L_e its length, p node potentials):

$$
Q_e = \frac{D_e}{L_e}(p_u - p_v), \qquad \sum_{e \ni u} Q_e = \begin{cases} +1 & u = S \\ -1 & u = G \\ 0 & \text{otherwise} \end{cases}, \qquad \frac{dD_e}{dt} = |Q_e| - D_e
$$

For us: this is coagulation-fragmentation with transport. Tubes that carry flow thicken and merge into one channel (coagulation); tubes without flow thin and vanish (fragmentation). The conservation law (Kirchhoff: inflow = outflow at every cell) is what makes it a path finder. This is the single most important idea in this document; Family D builds on it.

## 3. Family A: local reversible dimerization (our Stage A)

**Idea.** Every cell holds a tiny closed chemical system; loose evidence pairs up into committed evidence and can split back.

**Physics.** The smallest closed instance of Section 2.1: sizes 1 and 2, the reaction 2A1 ⇌ A2, run independently in every cell and channel.

**State.** c1, c2 ≥ 0 per cell and per channel (16 channels = 16 species). M = c1 + 2c2 fixed per cell from t = 0.

**Update.**

$$
m = \tfrac12 c_1 \sigma(r_m), \quad s = c_2 \sigma(r_s), \quad c_1 \leftarrow c_1 - 2m + 2s, \quad c_2 \leftarrow c_2 + m - s, \quad [r_m, r_s] = \mathrm{RateNet}_{3\times3}(x, c_1, c_2)
$$

**What it means in a maze.** Each cell decides, through its committed share 2c2/M, how strongly it believes it lies on the path. Merging means "I am becoming sure"; splitting means "I am taking it back". The decision is made by the rate network looking at the 3x3 neighbourhood; the chemistry itself only stores and bounds the decision.

**How information moves.** Only through the rate network: two 3x3 convolutions, at most 2 cells per step. Mass never moves.

**Guarantees and energy.** Non-negativity and exact per-cell conservation (proved). Because the rates are state-dependent outputs of a network, there is no detailed balance and no Lyapunov function; a cell is at rest when ½ c1 σ(r_m) = c2 σ(r_s).

**Risks.** (1) Conservation is only a safeguard here: it bounds the state but does not encode anything about paths. This is exactly why the supervisor found the physics abstract: in Family A, conservation has no task meaning. (2) Bounded is not convergent: the Milestone-1 sweep collapsed after T_train. (3) With a merge flux linear in c1, "coagulation" is really a conversion, not a collision.

**Role.** The working baseline of the CF families, and the platform on which the invariant tests were built. Keep it as the reference CF, not as the final model.

## 4. Family B: Becker-Döring commitment ladder

**Idea.** Replace "loose vs committed" by a ladder of cluster sizes 1..K per cell: confidence grows one unit at a time by absorbing a monomer and shrinks one unit at a time by shedding one, and a free energy decides where the ladder settles.

**Physics.** Becker-Döring kinetics (Section 2.4): A_k + A_1 ⇌ A_(k+1). Under detailed balance there is a Lyapunov free energy, and above a critical density the excess mass condenses into large clusters.

**State.** Per cell and channel: c_1, …, c_K ≥ 0 (K = 4 to 8), conserved mass M = Σ k c_k.

**Update.** Net flux from size k to k + 1, then the balance of every size:

$$
J_k = a_k\, c_1 c_k - b_{k+1}\, c_{k+1}, \qquad \Delta c_k = J_{k-1} - J_k \;(2 \le k \le K), \qquad \Delta c_1 = -2J_1 - \sum_{k=2}^{K-1} J_k
$$

with J_0 = J_K = 0. The neural network writes the rates. Two ways:

- Free rates: a_k, b_(k+1) = softplus(RateNet(x, c)). Most flexible, no energy guarantee.
- Detailed-balance parameterisation (recommended): the network outputs a per-cell energy profile g_k (one number per size) and one mobility μ. Set Q_k = exp(−g_k), a_k = μ, b_(k+1) = μ Q_k / Q_(k+1). Detailed balance then holds by construction, and while the rates are held fixed the free energy V(c) = Σ c_k (log(c_k/Q_k) − 1) decreases at every step (in continuous time).

**Two timescales, as in WONN.** Recompute the energy profile g from the neighbourhood only every few steps (slow), and let the chemistry relax under fixed rates in between (fast). The fast phase has a provable energy; the slow phase carries information between cells. This mirrors WONN's fast phase updates with a slow frequency state.

**Discrete-time positivity.** Several reactions consume c_1 at once, so the per-step outflows must be bounded jointly: give each consuming reaction a share of c_1 through a softmax and cap every flux at that share (and at c_(k+1) for shedding). Conservation holds for any fluxes, as in Family A.

**What it means in a maze.** Cell size k is a confidence level. Cells that keep receiving support become supersaturated and condense into large clusters ("on the path"); cells in dead ends stay below the critical density and remain monomers. Shedding gives gradual doubt instead of A's all-or-nothing split, and the critical size gives hysteresis: a cell that has condensed does not flip back on one weak signal, but sustained contrary evidence can still dissolve it. Readout: the mass fraction above the critical size, or the mean size.

**How information moves.** As in A, only through the rate network (slow phase).

**Guarantees and energy.** Conservation, positivity with bounded fluxes, and a true Lyapunov function during each fast phase. The free energy is the natural candidate for CF energy voting and for a convergence diagnostic.

**Risks.** K times more state; small steps may be needed for stiff rates; decisions can stall when nucleation is slow (cells stuck below the critical size).

**Role.** The best local mechanism: gives CF a provable energy and a meaningful notion of confidence. Combine with Family D for transport.

## 5. Family C: full Smoluchowski per cell with a learned kernel

**Idea.** Allow any two clusters to merge (i + j → i + j), not only monomer additions, with a collision kernel the network learns; big clusters can split into any two parts.

**Physics.** The general discrete equation of Section 2.1, truncated at a maximum size K, with mass-action (quadratic) collisions.

**State.** Per cell and channel: c_1, …, c_K ≥ 0, M = Σ k c_k.

**Update.** For every pair with i + j ≤ K, a merge flux and the matching split flux:

$$
m_{ij} = K_{ij}(x, c)\, c_i c_j, \qquad s_{ij} = F_{ij}(x, c)\, c_{i+j}, \qquad \Delta c_{i+j} \mathrel{+}= m_{ij} - s_{ij}, \quad \Delta c_i \mathrel{-}= m_{ij} - s_{ij}, \quad \Delta c_j \mathrel{-}= m_{ij} - s_{ij}
$$

(for i = j the factor ½ and the double consumption of c_i are applied). The kernel can be parameterised as a size-shape times a learned gate: K_ij = σ(network) · φ(i, j), with φ one of constant, additive (i + j), multiplicative (i · j) or a learned low-rank form u_iᵀ v_j. Detailed balance can again be built in by setting F_ij = K_ij Q_i Q_j / Q_(i+j), which gives the same free energy V as Family B.

**What it means in a maze.** Two partial hypotheses of medium confidence can combine into one strong hypothesis in a single event, instead of B's one-unit steps. In reasoning terms: corroboration. Two independent pieces of evidence that agree make a large jump in confidence; the kernel decides which pieces are allowed to corroborate each other.

**How information moves.** Still only through the rate network; the chemistry is per cell.

**Guarantees and energy.** Conservation for any fluxes; positivity with jointly bounded outflows; a Lyapunov function under detailed balance and fixed rates.

**Risks.** (1) Gelation: a kernel that grows faster than linearly with size (for example multiplicative) drives runaway growth, which in a truncated system shows up as all mass piling at size K, an early irreversible decision for every cell. (2) Cost grows as K² per cell. (3) Many more rates to learn; less interpretable than B.

**Role.** Mainly a scientific tool rather than the final model: the kernel-shape ablation (constant vs additive vs multiplicative vs learned) is a clean CF-vs-CF experiment that tests whether the physics of the kernel matters for reasoning, and whether gelation-prone kernels overthink.

## 6. Family D: transport CF, a conserved flow through the maze

**Idea.** Let mass actually move between neighbouring free cells, inject it at the start and remove it at the goal. Conservation then forces everything that leaves S to arrive at G, and the only cells that can carry it are the cells of the path. Channels that carry flow thicken and merge (coagulation); channels that carry none thin out and break (fragmentation).

**Physics.** Kirchhoff's conservation law on a network plus Physarum-style adaptation (Section 2.6).

**Why it solves mazes, in one argument.** Take a perfect maze (a tree of corridors), a unit source at S and a unit sink at G. Cut any corridor edge that is not on the S-G path: it separates a part of the maze containing neither S nor G. Inside that part there is no source and no sink, so by conservation the net flow across the cut edge must be zero. The flow is therefore non-zero exactly on the solution path, and zero in every dead end. For mazes with loops, flow splits over parallel routes; Physarum's adaptation rule (conductance grows with flux, decays without it) then concentrates all flow on the shortest route, which is the proven convergence result cited in Section 2.6.

**State.** A conductance per corridor edge between neighbouring free cells, D_e ≥ 0 (two maps on a grid: horizontal and vertical edges; edges into walls fixed to 0), and a potential p_u per cell. Optionally Family-B clusters per cell that read the local flow.

**Update (one step).**

1. Relax potentials a few times with a local, parameter-free rule (Jacobi iteration of Kirchhoff's law), with a source b_S = +1 and sink b_G = −1:

$$
p_u \leftarrow \frac{b_u + \sum_{v \sim u} D_{uv}\, p_v}{\sum_{v \sim u} D_{uv}}, \qquad Q_{uv} = D_{uv}(p_u - p_v)
$$

2. Redistribute a fixed amount of "tube material" towards edges that carry flow, a conserved coagulation step:

$$
D_e \leftarrow M_{\text{tube}} \cdot \frac{D_e\, f_\theta(|Q_e|, x)}{\sum_{e'} D_{e'}\, f_\theta(|Q_{e'}|, x)}
$$

with f_θ a learned, positive, increasing function. Total material M_tube is conserved exactly and every D_e stays non-negative; edges that lose flow lose material (fragmentation of unused channels), edges that keep flow gather it (coagulation into one channel).

3. Readout: the path is where |Q| through a cell is large (a learned 1x1 readout of |Q|, p and D).

**What is learned.** The local edge costs from the input (for example how attractive corridors are), the adaptation function f_θ, and the readout. On simple perfect mazes the physics alone may already solve the task; the learning matters for costs, for mazes with loops, and for Sudoku-like problems where "flow" must be defined by the network.

**What it means.** This is the clearest answer to the supervisor's question "what is the physics of mass conservation and why does it matter?": here conservation is the reasoning. A dead end cannot hold flow, because nothing leaves it; the path is the only way for mass to get from S to G.

**How information moves.** Through the potentials. Jacobi relaxation spreads information like diffusion, one cell per inner iteration; inner iterations have no parameters and are cheap, so many can be run per step.

**Guarantees and energy.** Exact conservation of flow (antisymmetric Q) and of tube material; non-negative conductances; zero net flow in dead ends at steady state on perfect mazes; for fixed conductances, the steady flow minimises the dissipated energy Σ Q_e² / D_e (Thomson's principle for electrical networks), which serves as a natural energy. Thinking longer is built in: more relaxation iterations bring the flow closer to the exact Kirchhoff solution, so extra test-time steps refine rather than disturb the answer.

**Risks.** (1) Diffusive relaxation is slow on long paths: plain Jacobi needs many iterations (mitigate with over-relaxation, a multigrid hierarchy, or a few steps of an exact solver inside the layer). (2) Ties: when several shortest paths exist, flow splits evenly and the strict metric (one path) fails; break ties with stochastic initial conductances (Family G). (3) A reviewer may say "this is a classical algorithm"; answer by reporting the pure-physics version (no learning) as a CF reference, and show what learning adds.

**Role.** The recommended next CF mechanism for mazes: it gives conservation a task meaning, makes "thinking longer" natural, and matches both the Physarum biology and WONN's "fragments merge into one path" picture.

## 7. Family E: fragment CF, path fragments as clusters

**Idea.** A cluster is a connected piece of candidate path. Pieces that touch and agree merge into longer pieces; pieces that run into a dead end or conflict are cut. The answer is the piece that contains both S and G.

**Physics.** Coagulation on a graph: clusters are connected components, and two clusters can merge only where they touch. This is Smoluchowski kinetics constrained by geometry. Its unconstrained cousin, random edges joining components, is exactly the multiplicative-kernel case linked to Erdős-Rényi random graphs (Section 2.2), so fragment CF sits squarely inside the coagulation literature.

**State.** Per free cell: a candidate mass w_u ≥ 0 (how much "path material" sits here) and a soft fragment identity ℓ_u (a vector, or a distribution over a small number of labels). The size of a fragment is the total mass of the cells sharing its identity.

**Update.**

1. Coagulation: for every pair of neighbouring cells, a learned gate g_uv = σ(kernel(h_u, h_v, x)) decides whether their fragments may join; identities mix along open gates, weighted by mass: ℓ_u ← normalise(w_u ℓ_u + Σ_v g_uv w_v ℓ_v). Repeated mixing makes a connected, open piece share one identity.
2. Fragmentation: a learned cut score at each cell (high at branch points and at the tips of dead ends) closes gates, so identities on either side drift apart again.
3. Mass: candidate mass is redistributed within and between fragments with a conserved update (as in Family D's material step), flowing towards fragments that connect to S or G.
4. Readout: cells whose identity matches both the identity at S and the identity at G.

**What it means in a maze.** It is literally WONN's Figure 5 process made explicit: many diffuse candidate fragments, merging when compatible, inconsistent ones suppressed, until one fragment spans S to G. The analogy with algorithms is direct as well: merging components is what union-find does in Kruskal's algorithm, so coagulation here is a soft, learned union-find.

**How information moves.** Along the fragments themselves: identity spreads through open gates, so a fragment "knows" whether it touches S or G once mixing has crossed it. Fast inside straight corridors, but diffusive like Family D unless long-range jumps are added (for example pointer doubling, letting each cell also read its neighbour's neighbour every step).

**Guarantees and energy.** Conservation of candidate mass and non-negativity are easy to keep. Under pure coagulation the number of distinct fragments can only fall, a simple monotone quantity; with fragmentation this is lost. There is no free energy unless the gates are tied to one (possible but not derived here).

**Risks.** Soft identities blur over long mixing; loops in the maze can merge fragments around a cycle; the most complex of the families to implement and to explain.

**Role.** The most faithful "fragments merge into a path" mechanism. Its one-dimensional version is important for NLP: on a sequence, fragments are spans, coagulation of neighbouring spans is bottom-up chunking or parsing, and fragmentation re-splits a wrong grouping.

## 8. Family F: particle-population CF, global hypotheses with an attention kernel

**Idea.** A small population of global particles, each a hypothesis with a feature vector and a mass (its support). Compatible hypotheses merge and pool their support; a hypothesis facing conflicting evidence splits into two variants. The total support is fixed.

**Physics.** Smoluchowski kinetics among a finite population of particles carrying features, as in the Marcus-Lushnikov picture (Section 2.5) but with a kernel that depends on what the particles represent, not only on their size.

**State.** N slots, each with features f_k ∈ R^d and mass w_k ≥ 0, with Σ_k w_k = M fixed. Cells (or tokens) are not particles here; they are the evidence the particles explain.

**Update.**

1. Kernel: K_kl = σ(q(f_k)ᵀ k(f_l) / √d): an attention score between hypotheses.
2. Coagulation: mass moves from k to l at rate K_kl w_k w_l (mass-action), and the receiving features move towards the mass-weighted mean, f_l ← (w_l f_l + Δw f_k) / (w_l + Δw). Outflows from each slot are jointly bounded, so Σ w is conserved and no w goes negative.
3. Fragmentation: a slot with high internal conflict (measured against the evidence it explains) gives part of its mass to an empty or weak slot, with features f_k ± δ, where δ is a learned direction computed from the conflicting evidence.
4. Evidence coupling: every step the slots read the input by cross-attention (slots query cells), so the evidence stays visible.
5. Readout: each cell's answer is a mass-weighted vote of the slots that attend to it.

**What it means.** Here mass conservation has a second clear meaning: the belief budget is fixed. Hypotheses compete for a constant amount of support, like probability mass. Coagulation is consolidation (two hypotheses that say the same thing become one, stronger); fragmentation is considering alternatives (a hypothesis that does not fit splits into variants). It is online agglomerative plus divisive clustering, driven by learned kernels.

**How information moves.** Globally in one step: attention connects every slot with every slot and every cell.

**Guarantees and energy.** Conservation of total support and non-negativity by construction. An energy exists if mass transfers between slots obey detailed balance (for example rates derived from a learned per-slot free energy); not derived here.

**Risks.** (1) Early collapse of all support into one slot, the population analogue of gelation (Section 2.2); counter with size-dependent fragmentation or a bounded kernel. (2) Slots are interchangeable, so training must handle permutation symmetry. (3) For mazes, a whole path is hard to store in one feature vector; the readout must decode it through cross-attention.

**Role.** The natural CF for Sudoku (hypotheses about digit assignments) and the bridge to NLP. Combined with Family E's one-dimensional spans, it gives token-level CF where attention is the learned coagulation kernel.

## 9. Family G: stochastic CF and voting

**Idea.** Run the CF dynamics as a random process instead of its average: individual merge and split events are sampled, so the same maze yields different trajectories, and the best one is selected by an energy.

**Physics.** The Marcus-Lushnikov process (Section 2.5): finite stochastic coagulation whose large-population limit is the deterministic Smoluchowski equation we use now. Fluctuations are physically real for small populations, where one early event can tip a whole cluster.

**State and update.** Any of Families A to F, with one of three levels of randomness:

| Level | What is random | Cost | Differentiable? |
| --- | --- | --- | --- |
| Random start | the initial split of each cell's mass between sizes (for A: c2 = u M/2, c1 = (1 − u) M, u ~ U(0, 0.5)), or random initial conductances in D | none extra per step | yes |
| Noisy rates | multiplicative noise on rates during the rollout | small | yes (reparameterised) |
| Sampled events | discrete event counts per step (tau-leaping: Poisson-distributed numbers of merges and splits, capped by what is available) | larger | no; train on the mean-field version, sample only at test time |

**What it means.** Exploration: different trajectories commit to different hypotheses early, and the energy picks the most consistent one. This is exactly the role random initial oscillators play in AKOrN (energy voting with up to 4,096 samples on Sudoku) and WONN (32 trajectories on Maze-hard, +3.9 points).

**Selection energy per family.** A: summed residual flux Σ|m − s| over time (no proof). B and C: the free energy V (Section 2.4). D: dissipated energy Σ Q²/D, or the fraction of flow that leaks into non-path cells. F: a mixture free energy over slots (to be defined).

**Guarantees.** With integer event counts, conservation and non-negativity hold exactly for every sample, not only on average.

**Risks.** Cost grows with the number of samples K; sampled events are not differentiable; if the energy does not correlate with correctness, voting does not help (AKOrN observed this when the Ω term was removed).

**Role.** A layer on top of any family rather than a family of its own. The cheapest version (random start) should be added to whichever family is built next, both for test-time voting and to break ties between equally short paths in Family D.

## 10. Comparison of the families

The families differ most in two ways: how far mass and information move per step, and how much structure a cluster carries. Our current model sits in the weakest corner of both.

Moving right buys reach (fewer steps for long mazes); moving up buys meaning (clusters that correspond to parts of an answer). D + B combines the first real gain in reach with the first meaningful notion of confidence and a provable energy.

| Criterion | A dimer | B ladder | C full Smol. | D transport | E fragments | F global | G stochastic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| What conservation means | safeguard only | safeguard + condensation threshold | safeguard | flow S → G marks the path | path material | fixed belief budget | exact per sample |
| Coagulation means | loose → committed | confidence +1 | corroboration jump | channels thicken and join | fragments join | hypotheses consolidate | sampled merges |
| Fragmentation means | take back | confidence −1 | split a belief | unused channels thin | cut at dead ends | consider alternatives | sampled splits |
| Information reach per step | 2 cells | 2 cells | 2 cells | many cells (inner relaxation) | along fragments | global | as base family |
| Provable energy | no | yes (detailed balance, fixed rates) | yes (same) | yes (dissipation, fixed conductances) | no | possible | as base family |
| Thinking longer built in | no | partly (relaxes to equilibrium) | partly | yes (relaxation converges) | partly | unclear | via voting |
| Main risk | overthinking | slow nucleation | gelation, K² cost | slow diffusion, ties | blurring, complexity | collapse into one slot | cost, no gradients |
| Implementation effort | done | medium | medium-high | medium | high | high | low (random start) |
| Best suited to | reference | confidence per cell | kernel ablation | mazes | mazes, spans in text | Sudoku, NLP | test-time compute |

## 11. Recommendation and a plan to compare CF algorithms with each other

Build one new main CF algorithm, "Transport-Ladder CF" (D + B + G), and compare it against the other CF families rather than against conventional networks. Order of work:

1. Keep Family A as the reference CF: it already runs and is tested.
2. Build D + B: conserved flow through free cells with Physarum-style redistribution of tube material (D), each cell reading its local flow into a Becker-Döring commitment ladder with a detailed-balance free energy (B). Add random initial conductances (G) for tie-breaking and voting.
3. Use C only as an ablation of kernel shape inside B's ladder.
4. Keep E and F for Sudoku and NLP, after the maze results.

### 11.1 CF-vs-CF experiments and the question each answers

| # | Experiment | Question | Mechanism measurement |
| --- | --- | --- | --- |
| X1 | A vs D on the same mazes | Does giving conservation a task meaning (flow from S to G) help? | mass or flow on the path vs in dead ends, over time |
| X2 | D physics-only vs D learned | What does learning add to the physics? | accuracy, and where the learned costs differ from uniform |
| X3 | Fragmentation on vs off, in A, B and D (in D: unused channels never thin) | Does the ability to undo matter, and when? | accuracy vs T; how many wrong commitments are later reversed |
| X4 | Kernel shape in B/C: constant, additive, multiplicative, learned | Do gelation-prone kernels commit too early and overthink? | mass at the maximum size over time; accuracy vs T |
| X5 | Energy vs correctness (B free energy, D dissipation), and voting with G | Is there a CF energy that ranks answers, as in AKOrN and WONN? | energy histograms of correct vs wrong answers; accuracy vs number of samples |
| X6 | Thinking longer for every family | Which mechanism improves when run longer? | accuracy and settling (residual flux, free energy) vs T, training T marked |

### 11.2 Mechanism figures to produce (CF itself, not scores)

- Size distribution over time for a path cell and a dead-end cell (B): condensation vs dissolution.
- Flow map |Q| over time on one maze (D): flow spreading, then collapsing onto the path; dead ends going dark.
- Tube material over time (D): coagulation into one channel, fragmentation of the rest.
- Number of fragments over time (E, later): many fragments merging into one.
- Free energy and dissipation curves (B, D): monotone decrease under fixed rates.

### 11.3 Before implementing any family

Write its specification card first, answering the eight questions of Section 1 and the eight headings used for each family above. Sections 3 to 9 are those cards in draft form; review them with the supervisor before coding, as the feedback asked.

## Sources

- Ball, Carr & Penrose (1986), The Becker-Döring cluster equations: basic properties and asymptotic behaviour of solutions, Commun. Math. Phys. 104
- Murray & Pego, on the Becker-Döring equations and detailed balance coefficients
- Schlichting, Macroscopic limit of the Becker-Döring equation via gradient flows
- Fournier (2025), On gelation for the Smoluchowski equation
- Coagulation with bilinear kernels and random graphs (arXiv 1902.07686)
- Gelation as loss of dynamical stability (arXiv 2601.18806)
- Cepeda & Fournier, Smoluchowski's equation: rate of convergence of the Marcus-Lushnikov process
- Jacquot, A historical law of large numbers for the Marcus-Lushnikov process
- Tero, Kobayashi & Nakagaki (2007), A mathematical model for adaptive transport network in path finding by true slime mold, J. Theor. Biol. 244
- Bonifaci, Mehlhorn & Varma, Physarum can compute shortest paths
- Becchetti et al., Physarum can compute shortest paths: convergence proofs and complexity bounds
- Mehlhorn, The Physarum Computer (talk)
- AKOrN (Miyato et al., ICLR 2025) and WONN (Dai & Song, NeurIPS 2026): project PDFs; figures quoted from their Sudoku and Maze-hard sections.

The Kirchhoff argument in Section 6 (zero flow in dead ends of a perfect maze) and the neural parameterisations in Sections 3 to 9 are our own derivations and designs, not taken from these sources.
