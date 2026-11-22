# Depth by difficulty: strict accuracy vs test-time horizon T

Packet figure: F8. This README documents the figure file
`depth_by_difficulty.png` in this folder.

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
- Sweep cache: `week2_results/probe_sweeps.json` — strict accuracy per
  (model, seed, dataset, T), computed on CPU from the saved checkpoints with
  `cf.train.sweep_model` (one rollout per batch, scored with the repo's
  `evaluate_mask`/`summarize` path, threshold 0.5).
- Checkpoints: `week2_results/followups/E1_CF-Full_seed{0,1,2}.pt` (160
  epochs), `week2_results/followups/E4_CF-NoFrag160_seed{0,1,2}.pt` (160
  epochs, hid 18), `week2_results/RecurrentCNN_seed{0,1,2}.pt` (40 epochs,
  fixed T=8), `week2_results/followups/E3_RecurrentCNN_seed{0,1,2}.pt`
  (40 epochs, variable T).
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
