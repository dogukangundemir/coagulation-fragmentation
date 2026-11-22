# Depth by difficulty: strict accuracy vs test-time horizon T

Packet figure F8, part 1 of 2 (other part: figures/curriculum_results/depth_extrapolation.png). This README documents the figure files in this folder:

- `depth_by_difficulty.png` (F8): mean strict accuracy plus a sample-sd
  band per model and difficulty (documented below).
- `depth_by_difficulty_per_seed.png` (F8 companion, documented in the last
  section): every seed's own curve plus the mean, and the "keeps improving
  vs collapse" tally beyond the training horizon.
- `data.csv`: per-point means, sds and per-seed values for both figures.
- `per_seed_tally.csv`: per-seed strict accuracy at T=8 and T=32 and the
  keeps-improving verdict for every model x difficulty.

## Caption

Strict accuracy (prediction is exactly the BFS shortest path) as a function of
the test-time horizon T, one panel per difficulty: the held-out level-5 test
split (11x11, n=1000), probe level 7 (15x15, n=1000) and probe level 9 (19x19,
n=1000). Four models, all swept at T = 1, 2, 4, 8, 12, 16, 24, 32 with frozen
weights: CF-Full (hid 16, 160 epochs, trained at fixed T=8), CF-NoFrag (hid 18,
160 epochs, trained at fixed T=8), RecurrentCNN (hid 22, 40 epochs, trained at
fixed T=8) and RecurrentCNN (hid 22, 40 epochs, trained at variable T, drawn
from U{4..16} per batch). The vertical dashed line marks the training horizon
T=8 of the fixed-T models (it has its own legend entry and applies to the
fixed-T models only).

## How to read

- x-axis: test-time horizon T (unrolling steps at inference, weights frozen),
  linear in T; the eight swept values T = 1, 2, 4, 8, 12, 16, 24, 32 are
  plotted at their true positions. On a linear axis the T=1 and T=2 positions
  are too close for side-by-side tick labels, so those two labels are
  staggered onto two rows (T=1 above, T=2 below, each under its own tick).
- y-axis: strict accuracy, mean over 3 seeds (seeds 0, 1, 2), running from 0
  to the maximum possible 1; the shaded band
  is the sample standard deviation over the seeds, clipped to [0, 1] (strict
  accuracy is a fraction, so a band cannot extend outside it).
- One colour per model: blue = CF-Full, vermillion = CF-NoFrag, reddish purple
  = RecurrentCNN fixed T, bluish green = RecurrentCNN variable T. Training
  settings are restated in the legend.
- The vertical dashed line at T = 8 is where the fixed-T models were trained;
  what happens to the right of it is "thinking longer" at test time.

## Key takeaway (numbers from data.csv)

- CF models keep improving past their training horizon on every difficulty.
  CF-Full: 0.735 +- 0.025 at T=8 to 0.942 +- 0.034 at T=12 on the level-5
  test; 0.272 +- 0.021 to 0.771 +- 0.152 at T=16 on probe 7; 0.048 +- 0.007
  to 0.606 +- 0.275 at T=24 on probe 9. CF-NoFrag shows the same trend
  (test: 0.655 +- 0.057 at T=8, 0.902 +- 0.095 at T=16).
- The RecurrentCNN trained at fixed T=8 peaks exactly at its training horizon
  and collapses beyond it: 0.936 +- 0.015 at T=8 but 0.000 at T=32 on the
  level-5 test (0.432 +- 0.028 -> 0.000 on probe 7, 0.082 +- 0.036 -> 0.000
  on probe 9).
- Training with variable T buys the RecurrentCNN some robustness (test:
  0.942 +- 0.032 at T=12) but it still degrades at large T (0.419 +- 0.084
  at T=32), and on the hardest panel (probe 9) it reaches only
  0.173 +- 0.061 at T=12 versus CF-Full's 0.606 +- 0.275 at T=24.
- Harder mazes need more test-time depth: CF-Full's best T moves from 12
  (level-5 test) to 16 (probe 7) to 24 (probe 9).

## Data source

- Plotting script: `scripts/figures/depth_by_difficulty.py` (run from the repo
  root; re-runnable, reuses the cached sweeps).
- Sweep cache: `probe_sweeps.json` (results archive at the repo root; the
  plotting script writes and reads it there) — strict accuracy per
  (model, seed, dataset, T), computed on CPU from the saved checkpoints with
  `cf.train.sweep_model` (one rollout per batch, scored with the repo's
  `evaluate_mask`/`summarize` path, threshold 0.5).
- Checkpoints (results archive at the repo root; exact paths in the plotting
  script): `followups/E1_CF-Full_seed{0,1,2}.pt` (160 epochs),
  `followups/E4_CF-NoFrag160_seed{0,1,2}.pt` (160 epochs, hid 18),
  `RecurrentCNN_seed{0,1,2}.pt` (40 epochs, fixed T=8),
  `followups/E3_RecurrentCNN_seed{0,1,2}.pt` (40 epochs, variable T).
- Data: curriculum level-5 test split and probe levels 7/9, built with
  `cf.data.curriculum` — fixed seeded splits from numpy `default_rng(0)`
  (4000/1000/1000 train/val/test from the level-5 pool, 1000 mazes per
  probe level), n=1000 per panel.
- Figure file: `depth_by_difficulty.png` in this folder; per-point values:
  `data.csv` in this folder (mean, sd and per-seed values
  for every model x panel x T).

## Caveats

- n = 3 seeds; the band is the sample sd over seeds, not a confidence
  interval, and sd is large where seeds disagree (e.g. CF-Full at T=32 on
  probe 9: 0.555 +- 0.319 — one seed regresses at large T).
- Bands are clipped to [0, 1]; clipping binds on both sides. The upper bound
  clips where mean + sd > 1, at six points where seeds disagree strongly
  (CF-Full on the level-5 test at T=16, 24 and 32 and on probe 7 at T=24 and
  T=32, CF-NoFrag on the level-5 test at T=24), and the lower bound
  clips where mean - sd < 0 at three points: RecurrentCNN
  (fixed T) at T=12 on probe 7 (0.0027 +- 0.0046) and on probe 9
  (0.0010 +- 0.0017), and RecurrentCNN (variable T) at T=4 on probe 7
  (0.0007 +- 0.0012). The clipped amounts are at most 0.054 (upper,
  CF-Full T=32 on the level-5 test) and 0.002 (lower).
- The two RecurrentCNNs were trained for 40 epochs while the CF models ran
  160 epochs (the available trained checkpoints); the comparison across
  training budgets is therefore not fully matched, and the fixed-T vs
  variable-T RecurrentCNN pair is the clean like-for-like contrast.
- The variable-T RecurrentCNN has no single training horizon; the dashed
  line does not apply to it (its training T ranged over 4..16 per batch), as
  the legend entry states.
- Probe levels 7 and 9 are harder, larger mazes never seen in training;
  absolute accuracies there are expectedly lower and seed spread larger.

---

# Per-seed companion: every seed's curve beyond the training horizon

File: `depth_by_difficulty_per_seed.png` in this folder.

## Caption

Per-seed view of the same sweeps as F8: strict accuracy (prediction is
exactly the BFS shortest path) vs test-time horizon T, one panel per
difficulty (level-5 test, 11x11, n=1000; probe level 7, 15x15, n=1000;
probe level 9, 19x19, n=1000), T = 1, 2, 4, 8, 12, 16, 24, 32 with frozen
weights. For each model, the thin lines are its three seeds (0, 1, 2) and
the bold line with dots is their mean. Four models, training settings in
the legend: CF-Full (hid 16, 160 epochs, trained at fixed T=8), CF-NoFrag
(hid 18, 160 epochs, trained at fixed T=8), RecurrentCNN (hid 22, 40
epochs, trained at fixed T=8) and RecurrentCNN (hid 22, 40 epochs, trained
at variable T, drawn from U{4..16} per batch). The vertical dashed line
marks the training horizon T=8 of the fixed-T models. The two guide lines
under the title state how to read the figure and the tally rule.

## How to read

- x-axis: test-time horizon T, linear, the same eight swept values as F8
  (T=1 and T=2 tick labels staggered onto two rows at their true
  positions); y-axis: strict accuracy from 0 to 1.
- One colour per model (same as F8: blue = CF-Full, vermillion = CF-NoFrag,
  reddish purple = RecurrentCNN fixed T, bluish green = RecurrentCNN
  variable T). Thin lines = individual seeds; bold line with dots = mean
  over the 3 seeds.
- There is no sd band in this figure: the thin per-seed lines show the
  spread directly, so nothing is aggregated without being shown per seed
  and no band needs clipping. (The companion figure F8 draws its sd bands
  clipped to [0, 1], as stated in its section above.)
- Everything right of the dashed line (T=8) is untested depth for models
  trained at fixed T=8.

## Key takeaway (numbers from data.csv and per_seed_tally.csv)

Tally rule, stated on the figure and applied uniformly: a seed "keeps
improving" when its strict accuracy at T=32 is at least its strict
accuracy at the training horizon T_train = 8; otherwise it drops.
"Collapse" is used only for a fall to 0.000.

Seeds that keep improving (strict at T=32 >= strict at T=8), out of
n = 3 seeds per cell:

| Model (training budget) | Level-5 test | Probe 7 | Probe 9 |
|---|---|---|---|
| CF-Full (160 epochs, fixed T=8) | 2/3 | 3/3 | 3/3 |
| CF-NoFrag (160 epochs, fixed T=8) | 2/3 | 3/3 | 3/3 |
| RecurrentCNN fixed T (40 epochs) | 0/3 | 0/3 | 0/3 |
| RecurrentCNN variable T (40 epochs) | 0/3 | 1/3 | 2/3 |

- Every CF seed keeps improving on the probe panels (probe 7: 0.144-0.286
  at T=8 rising to 0.333-0.926 at T=32; probe 9: 0.012-0.054 rising to
  0.197-0.810). On the level-5 test one CF seed in three drops (CF-Full
  seed 2: 0.707 -> 0.520; CF-NoFrag seed 0: 0.697 -> 0.542) while the
  other two keep improving (e.g. CF-Full seed 1: 0.754 -> 0.974). So the
  mean gain past T=8 is not guaranteed for every seed, but a drop never
  removes more than about a quarter of the seed's T=8 accuracy.
- The RecurrentCNN trained at fixed T=8 collapses in every seed on every
  panel: 0 of 9 seed-panel curves keep improving and all nine fall to
  exactly 0.000 at T=32 (e.g. test: 0.940, 0.948, 0.919 at T=8 ->
  0.000, 0.000, 0.000 at T=32).
- Variable-T training does not fix this at T=32 (test: 0/3, 0.831, 0.900,
  0.797 -> 0.358, 0.385, 0.515). Its 3 keeps on probe panels are small
  differences, two of them near zero (probe 9: seed 0 0.026 -> 0.041, seed
  2 0.019 -> 0.029); the third (probe 7, seed 2: 0.233 -> 0.236) is also a
  flat floor rather than a real gain.
- Inconclusive: with n = 3 seeds, CF-Full and CF-NoFrag are not
  distinguishable in robustness past T=8 (same 8/9 tally, and their T=32
  means lie within each other's seed spread, e.g. probe 9 at T=32:
  0.555 +- 0.319 vs 0.504 +- 0.263), so any CF-Full-vs-CF-NoFrag ordering
  is inconclusive.

## Data source

- Plotting script: `scripts/figures/depth_by_difficulty.py` (same run
  produces both figures; re-runnable, reuses the cached sweeps).
- Same sweep cache, checkpoints and data as F8:
  `probe_sweeps.json` (results archive at the repo root; strict accuracy
  per model, seed, dataset and T, computed on CPU from the saved checkpoints
  with `cf.train.sweep_model`, threshold 0.5); `data.csv` in this folder
  (per-seed values in the seed0/seed1/seed2 columns); tally:
  `per_seed_tally.csv` in this folder (per-seed strict accuracy at T=8 and
  T=32 and the verdict, plus the per model-panel counts used in the table
  above).

## Caveats

- n = 3 seeds per model; each tally cell is 3 binary outcomes, so the
  counts are coarse and every per-seed number is a single run.
- "Keeps improving" at T=32 vs T=8 is a coarse two-point rule: a seed can
  rise at T=16 or T=24 and still be counted as dropping if it regresses by
  T=32 (CF-Full seed 2 on the test panel behaves like this), or keep a
  flat floor and be counted as improving (RecurrentCNN variable T on
  probe 9).
- For the RecurrentCNN trained at variable T there is no single training
  horizon; the rule was applied at T=8 (one value inside its training
  range 4..16) for uniformity, as stated on the figure.
- Small-sample honesty: differences between models within the seed spread
  are inconclusive (see the CF-Full vs CF-NoFrag point above); 3 seeds is
  preliminary for any stronger claim.
- Training budgets are not matched across families (CF 160 epochs vs
  RecurrentCNN 40 epochs); the clean like-for-like contrasts are within
  each family (CF-Full vs CF-NoFrag; RecurrentCNN fixed vs variable T).
  The sweeps are inference-only (minutes of CPU per model), run on CPU
  from the saved checkpoints.
