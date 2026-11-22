# Coagulation–Fragmentation for Looped Reasoning

Can a looped network, trained at a fixed unroll depth, keep **solving mazes better the longer it runs at test time** — with its weights frozen? Our answer is to make the loop iterate a **Smoluchowski coagulation–fragmentation** chemistry $A_i + A_j \rightleftharpoons A_{i+j}$: small local hypotheses merge into clusters and wrong clusters split back into revisable units, an alternative to oscillator synchronization in looped physical networks (AKOrN: Kuramoto oscillators; WONN: Winfree oscillators). On the curriculum maze benchmark below, the trained CF models trail a parameter-matched RecurrentCNN at the training depth T=8 but keep improving well past it ("Thinking Longer", H4 supported), and a physics-only conserved-flow arm (Family D, zero learned parameters) solves the perfect-maze curriculum exactly — conservation alone does the reasoning.

## Quick start

```bash
pip install -e .                                  # Python >= 3.10; torch, numpy, matplotlib, datasets
python -m pytest -q tests/                        # full test suite (CPU only; CF_ALLOW_LOCAL_GPU=0 forces it)
python -m pytest -q milestone1/test_maze_cf.py    # self-contained acceptance checks

python -m scripts.run_week2              # week-2 training driver
python -m scripts.select_T_val           # validation-based test-time T selection
python -m scripts.h_verdicts             # regenerate hypothesis verdicts
python -m scripts.train_ladder           # Family B commitment ladder
python -m scripts.eval_transport_physics       # physics-only Family D on the curriculum
python -m scripts.eval_transport_mazehard      # physics-only Family D on Maze-hard
python -m scripts.eval_voting                 # Family G best-of-K voting

python -m scripts.figures.main_results   # one module per figure folder; each writes figures/<folder>/data.csv
python -m scripts.figures.curriculum_figures
# also: a_vs_d, voting, ladder_energy, baselines_reference, commitment, concept,
#       depth_by_difficulty, maze_paths, mechanism, training_budget, transport_mechanism
```

Notebooks: `notebooks/CF_Looped_Reasoning_Milestone_1.ipynb` (one-shot [Colab](https://colab.research.google.com) notebook: tests, untrained invariant diagnostics to T=64, trajectory heatmaps, a smoke-training run and a frozen-weights Thinking-Longer preview) and `notebooks/CF_Milestone2_Week1_HF.ipynb`.

## The CF model

Stage-A is a reversible dimerization $2A_1 \rightleftharpoons A_2$ performed per cell:

$$m_t = \tfrac{1}{2} c_1^{(t)} \sigma(\text{raw-merge}_t), \qquad s_t = c_2^{(t)} \sigma(\text{raw-split}_t)$$
$$c_1^{(t+1)} = c_1^{(t)} - 2 m_t + 2 s_t, \qquad c_2^{(t+1)} = c_2^{(t)} + m_t - s_t$$

The bounds $2m_t \le c_1^{(t)}$, $s_t \le c_2^{(t)}$ guarantee analytically $c_1, c_2 \ge 0$ and **exact conservation** of $M = c_1 + 2 c_2$ at every step: the reversible stoichiometry conserves $M$ for any fluxes, and the bound is what buys (provable) positivity. Both hold at any unroll depth, so the loop can run longer at test time than in training without the state exploding. Ablation switches: `fragmentation=False`, `bound_flux=False`; the invariants, gradient flow and ablations are tested in `milestone1/test_maze_cf.py` and `tests/` (mechanism figures: [`figures/concept/`](figures/concept/README.md), [`figures/mechanism/`](figures/mechanism/README.md)).

### CF families

"Coagulation–fragmentation" is a family of mechanisms, not one algorithm — they differ in what a particle is, what is conserved and how information moves. Seven families (A–G) are surveyed in depth in [`docs/cf_algorithm_families.md`](docs/cf_algorithm_families.md); this repository implements four:

- **Family A — local reversible dimerization** (the trained cell model): per-cell $2A_1 \rightleftharpoons A_2$ chemistry; the simplest, weakest form — no mass moves between cells. Benchmark results below.
- **Family B — Becker-Döring commitment ladder** ([`docs/specs/family_b_ladder.md`](docs/specs/family_b_ladder.md)): confidence grows one rung at a time under detailed-balance rates, giving a provable per-phase Lyapunov (free) energy. Preliminary run below.
- **Family D — transport CF** ([`docs/specs/family_d_transport.md`](docs/specs/family_d_transport.md)): conserved mass flows through the free cells from start to goal (Physarum-style pipe adaptation); on a perfect maze, conservation itself does the reasoning.
- **Family G — stochastic CF and voting**: random-start sampling over any family, keeping the calmest (lowest-energy) run; also the tie-breaker between equally short routes.

## Results

Setup: level-5 curriculum mazes (grid size 5, rendered 11x11 ASCII), train/val/test 4000/1000/1000, three seeds (0, 1, 2), training depth T_train = 8 (the variable-T RecurrentCNN sampled T in {4..16} per batch). **T\*** is the validation-selected test depth (argmax validation strict, ties to the smaller T; candidates {4, 8, 12, 16, 24} for 40-epoch and {4, 8, 12, 16, 24, 32} for 160-epoch runs). **Strict** accuracy counts a maze as solved only if the predicted support is exactly the BFS shortest path (on perfect mazes strict == exact); **params** are given raw and *effective* (nonzero-gradient entries after one forward+backward). Aggregates are mean ± sample sd over the 3 seeds. Hardware: single NVIDIA GeForce RTX 3070 Ti, PyTorch 2.11.0+cu128; every run trained under a wall-clock cap (45 min for the 40-epoch suite), none stopped early.

### Curriculum: test strict accuracy at T=8 and at T*

**40-epoch runs** (per-run wall time: RecurrentCNN 0.68 ± 0.02 min, variable-T RecurrentCNN 0.54 ± 0.02 min, CF-Full 1.07 ± 0.02 min, CF-NoFrag 0.94 ± 0.01 min, CF-WallAware 0.82 ± 0.05 min):

| Model | epochs | params raw / effective | test strict at T=8 | T\* per seed | test strict at T\* |
|---|---|---|---|---|---|
| RecurrentCNN hid=22, trained T=8 | 40 | 9,989 / 9,989 | 0.936 ± 0.015 | 8 / 8 / 8 | 0.936 ± 0.015 |
| RecurrentCNN hid=22, variable-T training | 40 | 9,989 / 9,989 | 0.843 ± 0.052 | 12 / 12 / 12 | 0.942 ± 0.032 |
| CF-Full hid=16 | 40 | 10,177 / 10,177 | 0.212 ± 0.045 | 12 / 12 / 12 | 0.398 ± 0.064 |
| CF-NoFrag hid=16 | 40 | 10,177 / 7,857 | 0.092 ± 0.024 | 16 / 16 / 16 | 0.266 ± 0.102 |
| CF-WallAware hid=16 | 40 | 10,177 / 9,921 | 0.475 ± 0.048 | 16 / 12 / 12 | 0.684 ± 0.125 |

**160-epoch runs** (fresh trainings, cosine schedule; wall time: CF-Full 5.73 ± 4.82 min, CF-NoFrag hid=18 2.50 ± 0.01 min, CF-WallAware 3.56 ± 0.20 min, RecurrentCNN fixed-T 2.06 ± 0.14 min, variable-T 2.17 ± 0.22 min):

| Model | epochs | params raw / effective | test strict at T=8 | T\* per seed | test strict at T\* |
|---|---|---|---|---|---|
| CF-Full hid=16 | 160 | 10,177 / 10,177 | 0.735 ± 0.025 | 16 / 16 / 12 | 0.949 ± 0.041 |
| CF-NoFrag hid=18 | 160 | 12,745 / 9,811 | 0.655 ± 0.057 | 12 / 12 / 16 | 0.921 ± 0.063 |
| CF-WallAware hid=16 | 160 | 10,177 / 9,921 | 0.903 ± 0.002 | 12 / 12 / 12 | 0.996 ± 0.002 |
| RecurrentCNN hid=22, trained T=8 | 160 | 9,989 / 9,989 | 0.991 ± 0.002 | 8 / 8 / 8 | 0.991 ± 0.002 |
| RecurrentCNN hid=22, variable-T training | 160 | 9,989 / 9,989 | 0.983 ± 0.001 | 8 / 12 / 12 | 0.988 ± 0.006 |

**Matched budgets are inconclusive.** At 160 epochs CF-WallAware (0.996 ± 0.002 at T\*) vs the fixed-T RecurrentCNN (0.991 ± 0.002) is inconclusive — seed values overlap around 0.99 — and CF-Full (0.949 ± 0.041) and CF-NoFrag hid=18 (0.921 ± 0.063) stay below both. Never compare a 160-epoch CF row with a 40-epoch RecurrentCNN row: CF improved mainly because training was four times longer (CF-Full 0.398 ± 0.064 → 0.949 ± 0.041, CF-NoFrag hid=18 0.361 ± 0.073 → 0.921 ± 0.063 at T\*).

**Unseen larger grids** (probe splits, 1000 mazes each, scored at T\*, all rows 40-epoch trainings): level-7 (15x15) — RecurrentCNN 0.432 ± 0.028, CF-Full 0.092 ± 0.064, CF-NoFrag 0.095 ± 0.029, CF-WallAware 0.308 ± 0.188; level-9 (19x19) — RecurrentCNN 0.082 ± 0.036, CF-Full 0.024 ± 0.035, CF-NoFrag 0.035 ± 0.008, CF-WallAware 0.101 ± 0.123. On level 7 the RecurrentCNN mean is highest but the gap to CF-WallAware sits within the seed spread; on level 9 every model is low and the differences sit within the seed spread (both inconclusive).

### Thinking longer (T > T_train, frozen weights)

At T = 32 (four times the CF training depth): CF-Full 0.806 ± 0.249 and CF-NoFrag 0.775 ± 0.214, versus RecurrentCNN variable-T 0.419 ± 0.084 and fixed-T 0.000 ± 0.000. The CF models degrade gracefully far past their training depth; the fixed-horizon baseline collapses off-horizon — the same collapse holds at the fair 160-epoch budget: strict 0.259 ± 0.228 at T = 12, 0.004 ± 0.008 at T = 16, and 0.000 at T = 24 and 32 on all 3 seeds. Settling (32-step unroll of the trained 40-epoch checkpoints): the geometric-mean residual CF flux decays from 0.225 (step 1) to 0.000813 (step 32) for CF-Full and 0.150 → 0.000233 for CF-NoFrag (about 280x and 640x reductions), while RecurrentCNN's relative state change stays order-one (1.015 → 1.144): the chemistry settles toward a fixed point, the recurrent baseline keeps churning. Figures: [`depth_extrapolation.png`](figures/curriculum_results/depth_extrapolation.png), [`settling.png`](figures/curriculum_results/settling.png).

### Hypotheses H1–H4

Pre-registered in [`docs/hypotheses.md`](docs/hypotheses.md); decision rule: 3 seeds paired by seed, a difference within the seed spread is inconclusive; every verdict uses the matched 40-epoch budget.

| Hypothesis | Verdict | Evidence (test strict, 40-epoch arms) |
|---|---|---|
| H1 — CF-Full reaches the RecurrentCNN at T=8 | NOT SUPPORTED | 0.212 ± 0.045 vs 0.936 ± 0.015 at T=8 |
| H2 — Fragmentation helps (CF-Full > CF-NoFrag at T=8; gap widens for T > 8) | SUPPORTED at T=8; T>8 clause INCONCLUSIVE | 0.212 ± 0.045 vs 0.092 ± 0.024: every seed pair positive; at T=16 the paired gap change (mean −0.038) is within the seed spread |
| H3 — Wall-aware helps (CF-WallAware > CF-Full) | SUPPORTED | 0.475 ± 0.048 vs 0.212 ± 0.045 at T=8, every seed pair positive |
| H4 — Thinking longer (accuracy improves for T > T_train) | SUPPORTED | CF-Full, same checkpoints: 0.349 ± 0.039 at T=16 vs 0.212 ± 0.045 at T=8, every seed pair positive |

Supplementary settling hypothesis H-A4 (CF residual flux falls past the training horizon while the RecurrentCNN's relative state change rises) is SUPPORTED on both clauses. The 160-epoch trainings and the validation depth selection are post-hoc descriptive context and never feed these verdicts.

### Maze-hard 30x30

The Maze-hard benchmark (1000 train / 1000 test grids) passes all nine validation checks with 0 failing rows per split and 0 duplicates ([`figures/maze_hard/validation_report.md`](figures/maze_hard/validation_report.md)). Every maze in both splits has several shortest paths (BFS counting capped at 3 returned 3 paths for 1000/1000 train and 1000/1000 test mazes), so Maze-hard comparisons use **exact match on the maze board — WONN's board accuracy** — as the headline metric: a strict "predicted support equals the labelled path" metric would undercount correct solutions whenever the model finds a different shortest path than the recorded one. Ties change the physics too: on a tied maze the conserved flow splits across every tied route, so the flow readout lights up cells of all tied routes and cannot match the single reference route. The physics-only Family D arm (`python -m scripts.eval_transport_mazehard.py`) therefore reports board match, a valid-path score, per-solve Jacobi sweep budgets (k = 1, 4, 16, 64) and accuracy separately on tied vs untied mazes; tie-breaking between equally short routes is the registered role of the stochastic layer — random initial conductances (the `sigma_tie` variant of the Family D spec), with several samples giving Family-G voting.

### CF-family findings

- **Information speed (Family D, physics-only, 0 learned parameters).** With the exact float64 solve, strict == exact accuracy is 1.000 on all three curriculum sets at every horizon measured (T = 1 to 64); the answer lands at the first step and the largest off-path flow share on any single maze is 0.000000083 — conservation alone fixes the flow on a tree ([`docs/results/transport_physics.md`](docs/results/transport_physics.md)). The informative measure left is *information speed*: with a budget of k local Jacobi sweeps per step (each sweep updates every cell from its own neighbours only), best strict accuracy over the swept T on levels 5/7/9 is 0.177 / 0.075 / 0.055 at k = 1, 0.393 / 0.235 / 0.205 at k = 4 and 0.770 / 0.575 / 0.355 at k = 16 — the gap to 1.000 is a concentration-of-flow error that more outer steps cannot repair; local relaxation, not step count, is the bottleneck ([`figures/a_vs_d/`](figures/a_vs_d/README.md)).
- **Voting (Family G, preliminary: 1 training seed, first 300 level-5 test mazes, not a reproduction).** Energy voting lifts the trained CF-WallAware net from 61.7% (K = 1) to 75.3% (K = 32) at T = 8 but recovers only part of the distance to its ceiling of 75.7% (mazes with at least one correct start) and stays 15.3 points below the 90.7% of the same net run once from its deterministic start; at T = 12 voting reaches 75.7% vs 99.3% deterministic. The physics-only Family D energy is perfect from any start: 100.0% at every K and both horizons, 0/300 mazes whose starts disagree ([`figures/voting/`](figures/voting/README.md)).
- **Family B (preliminary: 1 seed, reduced 46/160-epoch budget, CPU).** The trained commitment ladder (BD-WallAware, 21,777 raw params) reaches test strict 0.000 at T = 8 and 0.053 at its validation-selected T\* = 12 (probe-7 strict 0.013, probe-9 strict 0.003). The free energy behaves as the theory predicts within fixed-rate phases: summed V decreases in 16/16 phases and 1023 of 1024 per-maze phase pairs end below their start, with upward steps exactly at rate refreshes (a per-phase Lyapunov function, not a global one); the predicted path/dead-end commitment split has not emerged at this reduced budget ([`figures/ladder_energy/`](figures/ladder_energy/README.md)).

### Baselines (not a reproduction)

Single reduced-budget pipeline checks of the official baseline code — no conclusions about the papers' claims should be drawn from them ([`figures/baselines_reference/`](figures/baselines_reference/README.md)). AKOrN (Sudoku), our run at 40/100 of the official epochs (17.9 min): 99.3% in-distribution board accuracy at T=16 (export cross-check 99.4%), close to the published 99.8 ± 0.1; OOD 12.7% at T=16 (published 16.6%) and 38.6% at T_eval=128 (published full budget 51.7 ± 3.3) — the published OOD gain from longer evaluation shows in direction, below the full-budget number. WONN (Maze-hard), our run at 52/9,000 epochs (12.7 min, seed 137): exact-board accuracy 0.0% vs the published 76.2%, while path F1 is 47.7% (EMA checkpoint; 75.1% non-EMA, n = 1,000 test mazes) — expected at 0.6% of the official epoch budget.

## Figures index

- [`figures/a_vs_d/`](figures/a_vs_d/README.md) — Family A (trained) vs Family D (physics-only) on the same mazes, plus the honest maze-256 failure anatomy.
- [`figures/baselines_reference/`](figures/baselines_reference/README.md) — published AKOrN / WONN numbers vs our reduced-budget pipeline checks (not a reproduction).
- [`figures/commitment/`](figures/commitment/README.md) — per-step path probability on two test mazes (packet F4, part 1).
- [`figures/concept/`](figures/concept/README.md) — one worked Stage-A step (F1) and the pipeline diagram (F2).
- [`figures/curriculum_results/`](figures/curriculum_results/README.md) — budget vs fragmentation (F7 part 2), depth extrapolation, settling; full captions and per-seed values.
- [`figures/depth_by_difficulty/`](figures/depth_by_difficulty/README.md) — unroll depth vs maze difficulty (F8 part 1) and the per-seed keeps-improving tally.
- [`figures/ladder_energy/`](figures/ladder_energy/README.md) — Family B ladder: free energy over the rollout and per-cell size distributions.
- [`figures/main_results/`](figures/main_results/README.md) — curriculum main results, strict at T=8 and at T\* (F7 part 1).
- [`figures/maze_hard/`](figures/maze_hard/validation_report.md) — Maze-hard 30x30 dataset previews and validation report.
- [`figures/maze_paths/`](figures/maze_paths/README.md) — what every trained model outputs: thinking steps, path galleries, coverage, failure anatomy (F4-F6, F9).
- [`figures/mechanism/`](figures/mechanism/README.md) — encoder embedding of the initial monomer concentration (F3) and untrained-model invariants (F5).
- [`figures/milestone1/`](figures/milestone1/README.md) — report figures from the Colab notebook (dataset preview, invariant diagnostics, smoke run, Thinking-Longer preview).
- [`figures/training_budget/`](figures/training_budget/README.md) — 40- vs 160-epoch budget curves per family (supplementary to F7).
- [`figures/transport_mechanism/`](figures/transport_mechanism/README.md) — how the zero-parameter transport solver finds the path: flow sweeps, pipe thickening, conservation monitors.
- [`figures/voting/`](figures/voting/README.md) — Family G best-of-K random-start voting and energy histograms.

## Repository layout

```
coagulation-fragmentation/
├── cf/                          # shared package (imported by the notebooks)
│   ├── data/curriculum.py       # curriculum-maze parsing, validation, splits, dataset
│   ├── data/maze_hard.py        # Maze-hard 30x30 loader, validator, dataset
│   ├── metrics.py               # trace_path, evaluate_mask, summarize
│   ├── models/cf.py             # CF network (+ cross kernel for Maze-hard)
│   ├── models/ladder.py         # Family B Becker-Döring commitment ladder
│   ├── models/rcnn.py           # RecurrentCNN with logits_at
│   ├── models/transport.py      # Family D transport CF (physics-only arm)
│   ├── train.py                 # train/evaluate/sweep/settling helpers
│   └── utils.py
├── docs/
│   ├── assets/milestone1/       # presentation assets (schematics, equation renders, animated GIFs)
│   ├── cf_algorithm_families.md # CF algorithm-family survey (A-G)
│   ├── hypotheses.md            # pre-registered hypotheses and verdicts
│   ├── results/                 # results summaries from the CF-family arms (transport_physics.md)
│   └── specs/                   # pre-registered family specs (family_b_ladder.md, family_d_transport.md)
├── figures/                     # packet figures (F1-F10), a README.md (+ data.csv) per packet-figure folder (see index above)
├── milestone1/                  # self-contained module + acceptance checks (maze_cf.py, test_maze_cf.py)
├── notebooks/                   # Colab notebooks (CF_Looped_Reasoning_Milestone_1.ipynb, CF_Milestone2_Week1_HF.ipynb)
├── scripts/
│   ├── check_maze_hard.py       # Maze-hard validation report generator
│   ├── check_trained_invariants.py
│   ├── eval_*.py                # evaluation arms (transport, mazehard, voting, x1_x6)
│   ├── figures/                 # one module per figure folder (style.py shared)
│   ├── h_verdicts.py            # hypothesis-verdict generator
│   ├── run_week2.py             # week-2 training driver
│   ├── select_T_val.py          # validation-based test-time T selection
│   └── train_ladder.py          # Family B training
├── tests/                       # pytest: invariants, metric self-tests, data checks
└── pyproject.toml
```

## Reproducibility

- All numbers come from `week2_results/*.json` and `figures/*/data.csv` only; each figure folder's README states the budget (epochs, seeds, wall time, hardware) of every number. Run outputs (`week2_results/`, `results/`, checkpoints) stay local and are git-ignored.
- Checkpoint resume is bit-identical on CPU and equal up to cuDNN float noise on GPU.
- No script carries a hard-coded machine path. Trained checkpoints and per-maze evaluation CSVs live in the main repository's results archive, read-only for this clone: the scripts that need them (`scripts/eval_x1_x6.py`, `scripts/eval_voting.py`, `scripts/figures/voting.py`, `scripts/figures/a_vs_d.py`) resolve it from the `CF_MAIN_RESULTS` environment variable — the folder containing `week2_results/wallaware160/` and `figures/maze_paths/` — or take an explicit `--ckpt-dir` / `--maze256-csv` flag, and exit with a message instead of guessing. The baselines figure reads run records from `--baselines-dir` / `CF_BASELINES_DIR`.
