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

Curriculum maze benchmark: level-5 mazes (grid size 5, rendered as 11x11
ASCII), train/val/test 4000/1000/1000, three seeds
(0, 1, 2); the CF models and the fixed-T RecurrentCNN were trained at
T_train = 8, the variable-T RecurrentCNN at T ~ U{4,...,16}. The accuracy
figures are test **strict accuracy** (a maze counts as solved only if the
predicted support is exactly the BFS shortest path; on perfect mazes
strict == exact), reported as mean ± sample sd over the 3 seeds; the
settling figures are geometric means of their settling measures.
Full captions, caveats and per-seed values live in
[`figures/curriculum_results/README.md`](figures/curriculum_results/README.md)
and [`figures/curriculum_results/data.csv`](figures/curriculum_results/data.csv).

- **Training budget vs fragmentation** —
  [`budget_vs_fragmentation.png`](figures/curriculum_results/budget_vs_fragmentation.png).
  CF-Full hid=16 (10,177 effective params) improves from 0.398 ± 0.064
  (40 epochs) to 0.949 ± 0.041 (160 epochs); the capacity-matched ablation
  CF-NoFrag hid=18 (9,811 effective params) improves from 0.361 ± 0.073 to
  0.921 ± 0.063, so longer training is the main effect and fragmentation gives
  no detectable final-accuracy benefit at matched capacity. The variable-T
  RecurrentCNN hid=22 reference reaches 0.942 ± 0.032 in 40 epochs.
- **Depth extrapolation (test-time T sweep)** —
  [`depth_extrapolation.png`](figures/curriculum_results/depth_extrapolation.png).
  At T = 32 (four times the training depth of the CF models): CF-Full
  0.806 ± 0.249 and CF-NoFrag 0.775 ± 0.214, versus RecurrentCNN
  variable-T 0.419 ± 0.084 and RecurrentCNN fixed at T = 8: 0.000 ± 0.000.
  The CF models degrade gracefully far past their training depth; the
  fixed-horizon recurrent baseline collapses off-horizon.
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
│   ├── commitment/              # F4 commitment traces on two test mazes (+ data.csv)
│   ├── maze_paths/              # F4 thinking steps, F6 path galleries, F9 coverage + failure anatomy (+ data.csv)
│   ├── main_results/            # F7 curriculum main results (+ data.csv)
│   ├── curriculum_results/      # F7 budget vs fragmentation, F8 depth extrapolation, settling (no packet tag) (+ data.csv)
│   ├── depth_by_difficulty/     # F8 unroll depth vs maze difficulty (+ data.csv)
│   └── maze_hard/               # dataset previews + validation report
├── maze_cf.py                   # Milestone-1 self-contained module
├── test_maze_cf.py              # Milestone-1 pytest acceptance checks
├── CF_Looped_Reasoning_Milestone_1.ipynb
└── CF_Milestone2_Week1_HF.ipynb
```

Run outputs (`week2_results/`, checkpoints) stay local and are git-ignored.

## Course milestones

The project follows the course numbering of the Deep Learning Lab (every
milestone is graded, so each must stand alone as a deliverable; grading:
course notebooks 10%, each milestone 20%, final presentation 30%):

| Milestone | Date | Required content |
|---|---|---|
| **M1 — Problem + Design** | 11 Nov 2026 (done) | Problem statement, minimal example, definitions: why this architecture, why this data, why not other models. |
| **M2 — Prototype** | due 9 Dec 2026 | Implementation and explanation of the algorithm, a running example that should solve the problem, baseline, sample of hyperparameters, first test. |
| **M3 — Experiments** | 20 Jan 2027 | Main algorithm, results on a mock dataset, ongoing results and plan for the final datasets. |
| Final presentation | after M3 | Full results. |
