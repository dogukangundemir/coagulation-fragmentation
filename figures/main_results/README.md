# Main results (packet figure: F7)

`main_results.png` — test strict accuracy at the validation-selected test
depth T*, one bar per model.

## Caption

Test strict accuracy (fraction of the 1,000-maze test set solved with an
exact path match) at each seed's validation-selected T*, for eight models:
RecurrentCNN trained at fixed T=8 (40 epochs), CF-Full (hid 16, trained at
T=8, 40 and 160 epochs), CF-NoFrag (hid 16, trained at T=8, 40 epochs; hid
18, trained at T=8, 160 epochs), CF-WallAware (hid 16, trained at T=8, 40
and 160 epochs) and RecurrentCNN trained with variable T (40 epochs). Bars
show the mean over 3 seeds, error bars the sample sd (n−1), dots the
individual seeds.

## How to read

- One horizontal bar per model, sorted best (top) to worst (bottom); the
  two-line y-axis label gives the model, its training setting (depth,
  hidden size, epochs) and its parameter count (raw / effective, see
  Caveats).
- Bar length = mean test strict accuracy at T*; the dark line through the
  bar end is the sample sd; open dots are the three individual seeds
  (offset vertically only for readability).
- The number to the right of each bar is "mean ± sample sd" (values in
  `data.csv`).
- x-axis: strict accuracy on the test set, ticks from 0 to 1; the drawn
  axis extends to 1.18 only so the value labels fit — every bar starts at
  0, nothing is truncated.
- Colours group architectures: blue = CF-Full, vermillion = CF-NoFrag,
  orange = CF-WallAware, reddish purple = RecurrentCNN fixed T=8,
  bluish green = RecurrentCNN variable T (Okabe-Ito, colourblind-safe).

## Key takeaway (numbers from `data.csv`)

CF-WallAware at 160 epochs is the best model in the chart: 0.996 ± 0.002
test strict at T* = 12 (all seeds), above every other bar. Architecture
sets where a 40-epoch budget lands; a 160-epoch budget lets the CF models
catch up. At 40 epochs the RecurrentCNN baselines (variable T:
0.942 ± 0.032; fixed T=8: 0.936 ± 0.015) sit far above every 40-epoch CF
model (CF-Full 0.398 ± 0.064, CF-NoFrag 0.266 ± 0.102), and the wall-aware
kernel lifts the 40-epoch CF to 0.684 ± 0.125, above plain CF-Full at the
same budget. Raising the budget to 160 epochs closes the gap: CF-Full
reaches 0.949 ± 0.041 and CF-NoFrag hid 18 0.921 ± 0.063, matching the
RecurrentCNN baselines, while CF-WallAware goes further to 0.996. Comparing
a 40-epoch bar with a 160-epoch bar (e.g. CF-WallAware 0.684 vs 0.996) is
a training-budget difference, not evidence about architecture alone. All
conclusions carry the n = 3 caveat: the sd is over three seeds only, and
small absolute differences (e.g. 0.949 vs 0.936) are within one seed's sd.

## Data source

- `../../week2_results/week2_metrics.json` — 40-epoch runs (RecurrentCNN
  fixed T=8, CF-Full, CF-NoFrag): raw parameter counts, dataset sizes
  (test = 1,000 mazes).
- `../../week2_results/valT_selection.json` — 40-epoch runs: per-seed T*
  and test strict at T*.
- `../../week2_results/wallaware/valT_selection.json` and
  `../../week2_results/wallaware/week2_metrics.json` — CF-WallAware:
  per-seed T*, test strict at T*, raw and effective parameter counts
  (10,177 raw / 9,921 effective).
- `../../week2_results/wallaware160/results.json` — CF-WallAware 160-epoch
  runs: per-seed T*, test strict at T*, raw/effective parameter counts
  (10,177 raw / 9,921 effective) and the effective-parameter protocol.
- `../../week2_results/followups/results.json` — 160-epoch CF-Full,
  160-epoch CF-NoFrag (hid 18) and variable-T RecurrentCNN (40 epochs):
  per-seed T*, test strict at T*, raw/effective parameter counts and the
  effective-parameter protocol.
- `data.csv` — every plotted number (per-seed values, mean, sample sd,
  per-seed T*, parameter counts), written by the script.
- Script: `../../scripts/figures/main_results.py` (re-runnable:
  `python scripts/figures/main_results.py`).

## Caveats

- n = 3 seeds per model; the spread is the sample sd (n−1 denominator),
  shown as error bars only — no shading band is drawn, so no band
  clipping applies; all values lie in [0, 1] by construction.
- T* is selected per seed on the validation set (argmax validation strict
  over candidate depths, ties to smaller T), so the test depth differs
  across models and seeds: T* = 8 for RecurrentCNN fixed T=8; 12 for
  CF-Full 40 ep; 16 for CF-NoFrag 40 ep; 16/12/12 for CF-WallAware 40 ep;
  12 for CF-WallAware 160 ep; 16/16/12 for CF-Full 160 ep; 12/12/16 for
  CF-NoFrag hid 18 160 ep; 12 for RecurrentCNN variable T. All CF models
  and the fixed-T RecurrentCNN trained at depth T=8; the variable-T
  RecurrentCNN trained with variable depth.
- "Effective" parameters count only parameter elements with a nonzero
  gradient after one forward+backward pass on a real training batch
  (protocol in `followups/results.json`). Raw equals effective for every
  model except the CF-WallAware architecture, at both budgets
  (10,177 raw / 9,921 effective, from `wallaware/week2_metrics.json` and
  `wallaware160/results.json`), and CF-NoFrag hid 18 (12,745 raw /
  9,811 effective). The 40-epoch CF-Full and hid-16 CF-NoFrag runs share
  the hid-16 CF architecture whose recorded effective count equals its
  raw count (10,177).
- The test set has 1,000 mazes (`dataset_sizes.test = 1000` in
  `week2_results/week2_metrics.json`,
  `week2_results/wallaware/week2_metrics.json`,
  `week2_results/wallaware160/results.json` and
  `week2_results/followups/results.json`).
- The 40-epoch runs and the 160-epoch CF-Full and CF-NoFrag runs use
  cosine learning-rate decay over their respective epoch counts, so
  "epochs" is a training-budget comparison, not a matched-schedule
  comparison. The `wallaware160/results.json` file records no training
  schedule (only `epochs_done: 160` for all seeds), so no decay claim is
  made for CF-WallAware 160 epochs.
- Strict accuracy requires an exact path match; it is a harsh metric and
  small absolute differences (e.g. 0.949 vs 0.936) are within one seed's
  sd of each other.

Packet figure: F7
