# Coagulation–Fragmentation for Looped Reasoning

Research code for a looped neural computation scheme based on **Smoluchowski
coagulation–fragmentation** kinetics — $A_i + A_j \rightleftharpoons A_{i+j}$ —
studied as an alternative to oscillator synchronization in looped physical
networks (AKOrN: Kuramoto oscillators; WONN: Winfree oscillators).

Core hypothesis: **coagulation** (small local hypotheses merging into clusters)
plus **fragmentation** (wrong clusters splitting back into revisable units)
yields both high accuracy at small parameter budgets and *"Thinking Longer"* —
accuracy keeps improving when the network is unrolled for more steps at test
time than it was trained for, without touching the weights.

## Milestone 1 — maze infrastructure (this repository)

| File | Contents |
|---|---|
| `maze_cf.py` | Braided perfect-maze generator (7x7 / 9x9 / 15x15), BFS shortest-path labels, `MazeDataset`, the Stage-A `CoagulationFragmentationNet` (reversible $2A_1 \rightleftharpoons A_2$ with provable non-negativity and exact mass conservation), ablation switches (`fragmentation=False`, `bound_flux=False`), and a parameter-matched `RecurrentCNN` baseline. |
| `test_maze_cf.py` | `pytest` acceptance checks: BFS label validity; $c_1, c_2 \ge 0$; mass conservation $|M^{(t)}-M^{(0)}| < 10^{-5}$; gradient flow to the input and every parameter; ablation sanity. |
| `CF_Looped_Reasoning_Milestone_1.ipynb` | One-shot Colab notebook (mounts Google Drive): runs the tests, untrained-model invariant diagnostics to $T=64$, trajectory heatmaps of the raw chemistry, a smoke-training run with example predictions, and a frozen-weights "Thinking Longer" preview. |

### The Stage-A update rule

$$m_t = \tfrac{1}{2} c_1^{(t)} \sigma(\text{raw-merge}_t), \qquad
s_t = c_2^{(t)} \sigma(\text{raw-split}_t)$$
$$c_1^{(t+1)} = c_1^{(t)} - 2 m_t + 2 s_t, \qquad
c_2^{(t+1)} = c_2^{(t)} + m_t - s_t$$

The bounds $2m_t \le c_1^{(t)}$, $s_t \le c_2^{(t)}$ guarantee analytically:
$c_1, c_2 \ge 0$ and exact conservation of $M = c_1 + 2 c_2$ (distances matter —
the reversible *stoichiometry* conserves $M$ for any fluxes; the bound is what
buys *positivity*).

## Running

Tests (PyTorch + NumPy required, no GPU needed):

```bash
python -m pytest -q test_maze_cf.py
```

Notebook: open `CF_Looped_Reasoning_Milestone_1.ipynb` in
[Colab](https://colab.research.google.com) (Upload notebook; GPU runtime
recommended), then `Run all` (~5-10 min). Figures and checkpoints are written
to Google Drive under `CF_Looped_Reasoning/`.

## Reproducibility

Checkpoint resume is bit-identical on CPU and equal up to cuDNN float noise
on GPU.

## Results so far

### What the numbers mean (plain language first)

Curriculum maze benchmark: level-5 mazes (grid size 5, rendered as 11x11
ASCII), train/val/test 4000/1000/1000, three seeds (0, 1, 2). A model solves a
maze by running a loop for T steps and predicting, per cell, whether that cell
lies on the shortest path. **T** is the number of loop steps at test time
(weights frozen); every model except the variable-T RecurrentCNN was trained
at **T_train = 8** (the variable-T RecurrentCNN saw a random depth in
{4,...,16} per training batch). **T\*** is the validation-selected test depth:
per seed, the argmax of validation strict accuracy over the candidate depths
(ties broken toward the smaller T), then applied to the test split: candidates
are {4, 8, 12, 16, 24} for the 40-epoch runs and {4, 8, 12, 16, 24, 32} for
the 160-epoch runs. **Strict test accuracy** counts a maze as
solved only if the predicted support is exactly the BFS shortest path (on
perfect mazes strict == exact). **Params** are given raw and *effective*
(effective = entries receiving a nonzero gradient after one
forward+backward pass). All numbers are mean ± sample sd over the 3 seeds;
anything with fewer than 3 seeds is explicitly marked **preliminary**.

Hardware: single NVIDIA GeForce RTX 3070 Ti (CUDA), PyTorch 2.11.0+cu128.
Every run trained under a wall-clock budget cap (45 minutes for the 40-epoch
suite) and no run was stopped early; per-run wall times sit under each table.
All numbers below come from `week2_results/*.json` and `figures/*/data.csv`
only; full captions, caveats and per-seed values live in
[`figures/main_results/README.md`](figures/main_results/README.md),
[`figures/training_budget/README.md`](figures/training_budget/README.md)
and [`figures/curriculum_results/README.md`](figures/curriculum_results/README.md).

### Test strict accuracy at T=8 and at the validation-selected T*

**40-epoch runs** (training depth T = 8 unless stated):

| Model | epochs | params raw / effective | test strict at T=8 | T\* per seed | test strict at T\* |
|---|---|---|---|---|---|
| RecurrentCNN hid=22, trained T=8 | 40 | 9,989 / 9,989 | 0.936 ± 0.015 | 8 / 8 / 8 | 0.936 ± 0.015 |
| RecurrentCNN hid=22, variable-T training | 40 | 9,989 / 9,989 | 0.843 ± 0.052 | 12 / 12 / 12 | 0.942 ± 0.032 |
| CF-Full hid=16 | 40 | 10,177 / 10,177 | 0.212 ± 0.045 | 12 / 12 / 12 | 0.398 ± 0.064 |
| CF-NoFrag hid=16 | 40 | 10,177 / 10,177 | 0.092 ± 0.024 | 16 / 16 / 16 | 0.266 ± 0.102 |
| CF-WallAware hid=16 | 40 | 10,177 / 9,921 | 0.475 ± 0.048 | 16 / 12 / 12 | 0.684 ± 0.125 |

Wall time per run: RecurrentCNN 0.68 ± 0.02 min, variable-T RecurrentCNN
0.54 ± 0.02 min, CF-Full 1.07 ± 0.02 min, CF-NoFrag 0.94 ± 0.01 min,
CF-WallAware 0.82 ± 0.05 min.

**160-epoch runs** (fresh trainings, 160-epoch cosine schedule, training
depth T = 8):

| Model | epochs | params raw / effective | test strict at T=8 | T\* per seed | test strict at T\* |
|---|---|---|---|---|---|
| CF-Full hid=16 | 160 | 10,177 / 10,177 | 0.735 ± 0.025 | 16 / 16 / 12 | 0.949 ± 0.041 |
| CF-NoFrag hid=18 | 160 | 12,745 / 9,811 | 0.655 ± 0.057 | 12 / 12 / 16 | 0.921 ± 0.063 |
| CF-WallAware hid=16 | 160 | 10,177 / 9,921 | 0.903 ± 0.002 | 12 / 12 / 12 | 0.996 ± 0.002 |
| RecurrentCNN hid=22, trained T=8 | 160 | 9,989 / 9,989 | 0.991 ± 0.002 | 8 / 8 / 8 | 0.991 ± 0.002 |
| RecurrentCNN hid=22, variable-T training | 160 | 9,989 / 9,989 | 0.983 ± 0.001 | 8 / 12 / 12 | 0.988 ± 0.006 |

Wall time per run: CF-Full 5.73 ± 4.82 min (one seed took 11.3 min, the other
two ~2.9 min), CF-NoFrag hid=18 2.50 ± 0.01 min, CF-WallAware 3.56 ± 0.20 min,
RecurrentCNN fixed-T 2.06 ± 0.14 min, variable-T 2.17 ± 0.22 min.

On the fixed-T RecurrentCNN the fixed-horizon signature persists off-horizon
at the 160-epoch budget too: strict 0.259 ± 0.228 at T = 12 and
0.004 ± 0.008 at T = 16 (mean over the 3 seeds).

**Budget caveat.** The budgets differ: the CF rows at 160 epochs were trained
four times longer than the 40-epoch rows above, so any CF-at-160 vs
RecurrentCNN-at-40 comparison is a training-budget comparison, not an
architecture comparison. The fair-budget RecurrentCNN 160-epoch runs are
finished and in the table; at the matched budget, CF-WallAware
(0.996 ± 0.002 at T\*) vs the fixed-T RecurrentCNN (0.991 ± 0.002) is
inconclusive (the seed values overlap around 0.99), and the 160-epoch CF-Full
(0.949 ± 0.041) and CF-NoFrag hid=18 (0.921 ± 0.063) stay below both.

**Unseen larger grids** (probe splits, 1000 mazes each, scored at the
validation-selected T\*, all rows 40-epoch trainings): level-7 (15x15) —
RecurrentCNN 0.432 ± 0.028, CF-Full 0.092 ± 0.064, CF-NoFrag 0.095 ± 0.029,
CF-WallAware 0.308 ± 0.188; level-9 (19x19) — RecurrentCNN 0.082 ± 0.036,
CF-Full 0.024 ± 0.035, CF-NoFrag 0.035 ± 0.008, CF-WallAware 0.101 ± 0.123.
On level 7 the RecurrentCNN mean is highest but the differences to
CF-WallAware sit within the seed spread (inconclusive); on level 9 every
model is low and the differences sit within the seed spread (inconclusive).

### Hypotheses and verdicts

The pre-registered hypotheses, their names and the decision rule (3 seeds
paired by seed; a difference within the seed spread is inconclusive) are
defined in [`docs/hypotheses.md`](docs/hypotheses.md); every verdict-bearing
comparison uses the matched 40-epoch budget. Verdicts from the regenerated
`week2_results/h_verdicts.json`:

| Hypothesis | Verdict | Evidence (test strict, 40-epoch arms) |
|---|---|---|
| H1 — CF-Full reaches the RecurrentCNN at T=8 | NOT SUPPORTED | CF-Full 0.212 ± 0.045 vs RecurrentCNN 0.936 ± 0.015 at T=8 |
| H2 — Fragmentation contributes (CF-Full > CF-NoFrag at T=8, gap expected to widen for T > 8) | SUPPORTED at T=8; the T>8 widening clause INCONCLUSIVE | 0.212 ± 0.045 vs 0.092 ± 0.024: every seed pair positive, mean paired difference above its sd; at T=16 the paired gap change (mean −0.038) is within the seed spread |
| H3 — Wall-aware helps (CF-WallAware > CF-Full) | SUPPORTED | 0.475 ± 0.048 vs 0.212 ± 0.045 at T=8, every seed pair positive |
| H4 — Thinking longer (accuracy improves for T > T_train) | SUPPORTED | CF-Full, same checkpoints: 0.349 ± 0.039 at T=16 vs 0.212 ± 0.045 at T=8, every seed pair positive |

The supplementary settling hypothesis H-A4 (residual flux falls past the
training horizon while the RecurrentCNN's relative state change rises) is
SUPPORTED on both clauses. The 160-epoch trainings and the validation depth
selection are post-hoc descriptive context; they never feed these verdicts.

### Figures

- **Training budget vs fragmentation** —
  [`budget_vs_fragmentation.png`](figures/curriculum_results/budget_vs_fragmentation.png).
  Test strict at the validation-selected T\*: CF-Full hid=16 (10,177 effective
  params) improves from 0.398 ± 0.064 (40 epochs) to 0.949 ± 0.041
  (160 epochs); the capacity-matched ablation CF-NoFrag hid=18 (9,811
  effective params) improves from 0.361 ± 0.073 to 0.921 ± 0.063. Longer
  training is the main effect. At the 160-epoch budget the CF-Full vs
  CF-NoFrag difference (0.949 ± 0.041 vs 0.921 ± 0.063) is within the seed
  spread — inconclusive as a fragmentation benefit; at the 40-epoch budget
  fragmentation is supported at T=8 (H2 above). The variable-T RecurrentCNN
  hid=22 reference reaches 0.942 ± 0.032 in 40 epochs; the CF points at 160
  epochs are not a matched-budget comparison (CF 160 epochs vs RecurrentCNN
  40 epochs). The finished fair-budget RecurrentCNN 160-epoch runs sit at
  0.991 ± 0.002 (fixed T = 8) and 0.988 ± 0.006 (variable T) at T\*.
- **Depth extrapolation (test-time T sweep)** —
  [`depth_extrapolation.png`](figures/curriculum_results/depth_extrapolation.png).
  At T = 32 (four times the training depth of the CF models): CF-Full
  0.806 ± 0.249 and CF-NoFrag 0.775 ± 0.214, versus RecurrentCNN
  variable-T 0.419 ± 0.084 and RecurrentCNN fixed at T = 8: 0.000 ± 0.000.
  The CF models degrade gracefully far past their training depth; the
  fixed-horizon recurrent baseline collapses off-horizon. The budgets differ
  here too (CF 160 epochs vs RecurrentCNN 40 epochs); the finished
  fair-budget 160-epoch fixed-T RecurrentCNN shows the same collapse on all
  3 seeds (strict 0.004 ± 0.008 at T = 16 and 0.000 at T = 24 and 32).
- **Settling** —
  [`settling.png`](figures/curriculum_results/settling.png).
  Over a 32-step unroll of the trained 40-epoch checkpoints, the
  geometric-mean residual coagulation–fragmentation flux decays from 0.225
  (step 1) to 0.000813 (step 32) for CF-Full and from 0.150 to 0.000233 for
  CF-NoFrag (about 280x and 640x reductions), while RecurrentCNN's relative
  state change stays order-one (1.015 at step 1 to 1.144 at step 32): the CF
  chemistry settles toward a fixed point, the recurrent baseline keeps
  churning. (These are two different settling measures; only the decay trend
  is comparable across families.)

### Maze-hard 30x30

The Maze-hard benchmark (1000 train / 1000 test grids) passes all nine
validation checks with 0 failing rows per split and 0 duplicates (within
train, within test, across splits); see
[`figures/maze_hard/validation_report.md`](figures/maze_hard/validation_report.md).
Every maze in both splits has several shortest paths (BFS counting capped at
3 returned 3 shortest paths for 1000/1000 train and 1000/1000 test mazes), so
Maze-hard comparisons use **exact match on the maze board — WONN's board
accuracy** — as the headline metric: a strict "predicted support equals the
labelled path" metric would undercount correct solutions whenever the model
finds a different shortest path than the recorded one.

## Repository layout

```
coagulation-fragmentation/
├── cf/                          # shared package (imported by the notebooks)
│   ├── data/curriculum.py       # curriculum-maze parsing, validation, splits, dataset
│   ├── data/maze_hard.py        # Maze-hard 30x30 loader, validator, dataset
│   ├── metrics.py               # trace_path, evaluate_mask, summarize
│   ├── models/cf.py             # CF network (+ cross kernel for Maze-hard)
│   ├── models/rcnn.py           # RecurrentCNN with logits_at
│   ├── train.py                 # train/evaluate/sweep/settling helpers
│   └── utils.py
├── scripts/
│   ├── check_maze_hard.py       # Maze-hard validation report generator
│   ├── run_week2.py             # Week-2 training driver
│   ├── select_T_val.py          # validation-based test-time T selection
│   └── figures/                 # curriculum_figures.py, style.py
├── tests/                       # pytest: invariants, metric self-tests, data checks
├── slides/                      # presentation assets (mechanism schematics, equation renders, animated GIFs)
├── figures/                     # packet-figure tags (F1-F9) named in each folder's README.md
│   ├── concept/                 # F1 worked Stage-A step, F2 pipeline diagram (+ data.csv)
│   ├── mechanism/               # F3 encoder embedding, F5 untrained invariants (+ data.csv)
│   ├── commitment/              # F4 part 1 of 2 (other part: figures/maze_paths/thinking_steps.png) commitment traces on two test mazes (+ data.csv)
│   ├── maze_paths/              # F4 part 2 of 2 (other part: figures/commitment/commitment_maze0.png, commitment_maze248.png) thinking steps, F6 path galleries, F9 coverage + failure anatomy (+ data.csv)
│   ├── main_results/            # F7 part 1 of 2 (other part: figures/curriculum_results/budget_vs_fragmentation.png) curriculum main results (+ data.csv)
│   ├── curriculum_results/      # F7 part 2 of 2 (other part: figures/main_results/main_results.png) budget vs fragmentation, F8 part 2 of 2 (other part: figures/depth_by_difficulty/depth_by_difficulty.png) depth extrapolation, settling (no packet tag) (+ data.csv)
│   ├── depth_by_difficulty/     # F8 part 1 of 2 (other part: figures/curriculum_results/depth_extrapolation.png) unroll depth vs maze difficulty (+ data.csv, depth_by_difficulty_per_seed.png, per_seed_tally.csv)
│   ├── training_budget/         # no packet tag (the packet's budget figure is F7, figures/curriculum_results/budget_vs_fragmentation.png) 40- vs 160-epoch budget curves (+ data.csv)
│   └── maze_hard/               # dataset previews + validation report
├── maze_cf.py                   # Milestone-1 self-contained module
├── test_maze_cf.py              # Milestone-1 pytest acceptance checks
├── CF_Looped_Reasoning_Milestone_1.ipynb
└── CF_Milestone2_Week1_HF.ipynb
```

Run outputs (`week2_results/`, checkpoints) stay local and are git-ignored.

## Course milestones

The project follows the course numbering of the Deep Learning Lab (every
milestone is graded, so each must stand alone as a deliverable):

| Milestone | Date | Required content |
|---|---|---|
| **M1 — Problem + Design** | 11 Nov 2026 (done) | Problem statement, minimal example, definitions: why this architecture, why this data, why not other models. |
| **M2 — Prototype** | due 9 Dec 2026 | Implementation and explanation of the algorithm, a running example that should solve the problem, baseline, sample of hyperparameters, first test. |
| **M3 — Experiments** | 20 Jan 2027 | Main algorithm, results on a mock dataset, ongoing results and plan for the final datasets. |
| Final presentation | after M3 | Full results. |

### Grading

- The course's 10% "notebooks" grade refers to the course's pass/fail
  exercise notebooks, not to this project's notebook.
- Milestone 2 itself is worth 20%.
- The Milestone-2 deliverable is the project notebook (explanation + results)
  plus the presentation.
