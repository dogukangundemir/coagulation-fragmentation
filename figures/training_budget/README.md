# Training budget: test strict accuracy vs training epochs (40 vs 160)

`training_budget.png` - test strict accuracy at the validation-selected
depth T* as a function of the training budget (40 vs 160 epochs), one
connected pair per model family, mean ± sample sd over 3 seeds with the
individual seeds as dots.

Packet figure: none (supplementary; the packet's budget-vs-fragmentation
figure is F7 part 2, figures/curriculum_results/budget_vs_fragmentation.png)

## Caption

Test strict accuracy (fraction of the 1,000-maze test set solved with an
exact path match) at each family's validation-selected depth T*,
comparing a 40-epoch with a 160-epoch training budget. Families: CF-Full
(hid 16), CF-NoFrag (hid 18), CF-WallAware (hid 16), RecurrentCNN
(hid 22) trained at fixed T=8, and RecurrentCNN (hid 22) trained at
variable depth. Each point is the mean over 3 seeds (seeds 0, 1, 2);
error bars are the sample sd (n−1); open dots are the individual seeds,
offset horizontally only for readability. All models trained on the same
data (4,000 training mazes, T_train = 8 for the CF and fixed-T models).

## How to read

- x-axis: training budget in epochs (40, 160). y-axis: test strict
  accuracy at T* from 0 to 1; nothing truncated; no shading band is
  drawn, so no band clipping applies.
- One coloured line per family connects its mean at 40 epochs (left)
  with its mean at 160 epochs (right); the vertical error bar at each
  point is the sample sd (n−1) and the open dots are the three seeds.
- Colours: blue = CF-Full, vermillion = CF-NoFrag, orange = CF-WallAware,
  reddish purple = RecurrentCNN fixed T=8, bluish green = RecurrentCNN
  variable T (Okabe-Ito, colourblind-safe).
- Symbols: "strict" = the maze is solved with an exact match of the
  predicted path; T* = per-seed argmax of validation strict accuracy
  over candidate test depths, ties to the smaller T (T* depths per seed
  in `data.csv`).

## Results table (all values from `data.csv` written by the script)

| Family (shown variant) | 40 epochs: strict at T* (mean ± sd) | 160 epochs: strict at T* (mean ± sd) | Gain | Train min/seed, 40 → 160 | Hardware 40 / 160 |
|---|---|---|---|---|---|
| CF-WallAware (hid 16, trained T=8) | 0.684 ± 0.125 | 0.996 ± 0.002 | +0.311 | 0.82 → 3.56 | RTX 3070 Ti / cuda (not recorded) |
| RecurrentCNN fixed T=8 (hid 22) | 0.936 ± 0.015 | 0.991 ± 0.002 | +0.055 | 0.68 → 2.06 | RTX 3070 Ti / RTX 3070 Ti |
| RecurrentCNN variable T (hid 22) | 0.942 ± 0.032 | 0.988 ± 0.006 | +0.047 | 0.54 → 2.17 | cuda (not recorded) / RTX 3070 Ti |
| CF-Full (hid 16, trained T=8) | 0.398 ± 0.064 | 0.949 ± 0.041 | +0.551 | 1.07 → 5.73 | RTX 3070 Ti / cuda (not recorded) |
| CF-NoFrag (hid 18, trained T=8) | 0.361 ± 0.073 | 0.921 ± 0.063 | +0.561 | 3.87 → 2.50 | cuda (not recorded) / cuda (not recorded) |

Every point is n = 3 seeds, so no point is flagged "preliminary", and no
reduced baseline ("not a reproduction") is involved.

## Key takeaway (numbers from `data.csv`)

Four times the epochs lifts every CF family by a large margin: CF-Full
0.398 ± 0.064 → 0.949 ± 0.041 (+0.551), CF-NoFrag hid 18
0.361 ± 0.073 → 0.921 ± 0.063 (+0.561, the largest gain), CF-WallAware
0.684 ± 0.125 → 0.996 ± 0.002 (+0.311). The RecurrentCNN baselines
start high at 40 epochs (fixed T=8: 0.936 ± 0.015; variable T:
0.942 ± 0.032) and gain little in relative terms (fixed +0.055 to
0.991 ± 0.002, variable +0.047 to 0.988 ± 0.006). Within each family
the budget gain is conclusive: the best 40-epoch seed of each CF family
(no higher than 0.459, 0.425, 0.812 respectively) never reaches the
worst 160-epoch seed of the same family (no lower than 0.902, 0.867,
0.993). Across families at 160 epochs, CF-WallAware (0.996 ± 0.002) vs
the RecurrentCNN baselines (0.991, 0.988) is inconclusive (seed values
overlap around 0.99), while every 160-epoch family does clearly beat its
own 40-epoch CF start. The CF-NoFrag pair uses the hid-18 architecture
at both budgets (the only architecture-matched pair); the hid-16
40-epoch CF-NoFrag run (0.266 ± 0.102 at T*) appears in the
main-results figure, and that family's 160-epoch budget effect is
inconclusive between the two 40-epoch hid variants (0.266 vs 0.361 is
within their seed spreads).

## Data source

- `../../week2_results/valT_selection.json` - RecurrentCNN fixed T=8 and
  CF-Full 40-epoch runs: per-seed test strict at T*, per-seed T*.
- `../../week2_results/week2_metrics.json` - 40-epoch runs: raw parameter
  counts, per-seed training seconds, dataset sizes, device name.
- `../../week2_results/followups/results.json` - CF-Full 160 epochs (E1),
  CF-NoFrag hid 18 at 40 (E2) and 160 epochs (E4), RecurrentCNN variable
  T at 40 epochs (E3): per-seed values, minutes, effective-parameter
  protocol.
- `../../week2_results/wallaware/valT_selection.json` and
  `../../week2_results/wallaware/week2_metrics.json` - CF-WallAware
  40 epochs: per-seed values, minutes, device.
- `../../week2_results/wallaware160/results.json` - CF-WallAware
  160 epochs (E5): per-seed values, minutes, raw/effective parameters.
- `../../week2_results/rcnn160/results.json` - RecurrentCNN hid 22,
  160 epochs: fixed T=8 (E6, test strict at T* = 0.991 ± 0.002) and
  variable T (E7, 0.988 ± 0.006).
- `data.csv` - every plotted number: per-seed values, means, sample sds,
  T* depths, per-family training minutes, hardware, source paths.
- Script: `../../scripts/figures/training_budget.py` (re-runnable:
  `python scripts/figures/training_budget.py`).

## Caveats

- n = 3 seeds per family and budget; the spread is the sample sd (n−1
  denominator). Values are fractions in [0, 1]; no shading band is
  drawn, so no band clipping applies. Seed dots are offset horizontally
  only for readability; the measurement is the dot's y-position.
- Points plot accuracy at T*, the per-seed validation-selected depth of
  that same model and seed (argmax validation strict, ties to smaller
  T). T* differs by seed and family (depths in `data.csv`); at 40
  epochs the CF and variable-T T* are all 12 or 16 steps while the
  fixed-T seeds select 8, and the 160-epoch RecurrentCNN
  variable-T seeds select T* of 8, 12, 12 and the fixed-T seeds select
  8.
- Budget means epochs with the matching cosine learning-rate decay over
  that epoch count (recorded for the follow-ups and rcnn160 runs), so 40
  vs 160 compares budget + schedule, not schedule-matched training;
  `wallaware160/results.json` records no schedule, so no decay claim is
  made for the CF-WallAware 160-epoch point.
- The CF-NoFrag family uses hid 18 at both budgets; the hid-16 variant
  exists only at 40 epochs (in the main-results figure), so no
  hid-16 160-epoch no-frag comparison exists (not run).
- The 40-epoch CF-NoFrag hid-18 training minutes average (3.87 min/seed)
  is inflated by one slow recorded seed (seed 0: 10.3 min vs 0.63 min
  for seeds 1 and 2); per-seed times are not in `data.csv`, the
  recorded per-seed values are in the `week2_results/followups/results.json`
  wall_minutes fields.
- Hardware is what each source records: NVIDIA GeForce RTX 3070 Ti
  (CUDA) where a device name is logged, "cuda" with the GPU model
  unrecorded for the follow-ups and CF-WallAware 160-epoch runs. Figures
  in this folder were generated on CPU (matplotlib only); the training
  runs themselves executed on the GPUs noted above.
