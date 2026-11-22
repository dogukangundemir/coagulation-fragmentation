# Main results: test strict accuracy at the trained depth (T=8) and at the validation-selected depth (T*)

`main_results.png` — for every recorded model, two paired horizontal bars:
test strict accuracy at the depth the model was trained to run at (T=8)
and at the depth a validation sweep picks for it (T*). Models are grouped
by training budget (40 epochs left, 160 epochs right).

Packet figure F7, part 1 of 2 (other part: figures/curriculum_results/budget_vs_fragmentation.png)

## Caption

Test strict accuracy (fraction of the 1,000-maze test set solved with an
exact path match), for eleven model/budget combinations: RecurrentCNN
(hid 22) trained at fixed depth T=8 and with variable depth, CF-Full
(hid 16), CF-NoFrag (hid 16 and hid 18) and CF-WallAware (hid 16), each
trained 40 or 160 epochs, always n = 3 seeds (seeds 0, 1, 2). For each
model the solid bar is the accuracy at the validation-selected test depth
T* and the faded bar the accuracy at the trained depth T = 8; error lines
are the sample sd (n−1), open dots the individual seeds.

## How to read

- Two panels group models by training budget: left = trained 40 epochs,
  right = trained 160 epochs; within a panel the best model at T* is the
  top row.
- Each row shows two paired bars: upper, solid = accuracy evaluated at
  the validation-selected depth T*; lower, faded = accuracy evaluated at
  the trained depth T = 8. If the two bars are equal (both RecurrentCNN
  fixed-T rows), the validation sweep selected T* = T = 8, so the two
  measurements coincide.
- The paired labels give "mean ± sample sd"; the dark line is the sample
  sd (n−1 denominator) and the open dots are the three seeds.
- The two-line row label gives the architecture, hidden size and the
  parameter count (raw / effective; effective = parameter elements with
  a nonzero gradient after one training-style forward+backward on a
  training batch). Counts are the recorded measurements; the CF-NoFrag
  hid-16 effective count is measured by the script with the same
  protocol (no recorded value exists).
- x-axis: strict accuracy in [0, 1]; bars start at 0, nothing is
  truncated; the axis is drawn to 1.18 only to fit the value labels.
- Colours group families: blue = CF-Full, vermillion = CF-NoFrag,
  orange = CF-WallAware, reddish purple = RecurrentCNN fixed T=8,
  bluish green = RecurrentCNN variable T (Okabe-Ito, colourblind-safe).
- Symbols: "strict" = the maze is solved with an exact match of the
  predicted path; T* = per-seed argmax of validation strict accuracy
  over candidate test depths, ties to the smaller T.

## Results table (all values from `data.csv` written by the script)

| Model (hid; trained T=8 unless stated) | Budget | T=8 strict (mean ± sd) | T* strict (mean ± sd) | T* depths | Params raw / effective | Train min/seed | Hardware |
|---|---|---|---|---|---|---|---|
| RecurrentCNN fixed T=8 (hid 22) | 40 epochs | 0.936 ± 0.015 | 0.936 ± 0.015 | 8, 8, 8 | 9,989 / 9,989 | 0.68 | RTX 3070 Ti (cuda) |
| RecurrentCNN variable T (hid 22) | 40 epochs | 0.843 ± 0.052 | 0.942 ± 0.032 | 12, 12, 12 | 9,989 / 9,989 | 0.54 | cuda (model not recorded) |
| CF-WallAware (hid 16) | 40 epochs | 0.475 ± 0.048 | 0.684 ± 0.125 | 16, 12, 12 | 10,177 / 9,921 | 0.82 | RTX 3070 Ti (cuda) |
| CF-Full (hid 16) | 40 epochs | 0.212 ± 0.045 | 0.398 ± 0.064 | 12, 12, 12 | 10,177 / 10,177 | 1.07 | RTX 3070 Ti (cuda) |
| CF-NoFrag (hid 18) | 40 epochs | 0.104 ± 0.029 | 0.361 ± 0.073 | 16, 16, 16 | 12,745 / 9,811 | 3.87 | cuda (model not recorded) |
| CF-NoFrag (hid 16) | 40 epochs | 0.092 ± 0.024 | 0.266 ± 0.102 | 16, 16, 16 | 10,177 / 7,857 | 0.94 | RTX 3070 Ti (cuda) |
| CF-WallAware (hid 16) | 160 epochs | 0.903 ± 0.002 | 0.996 ± 0.002 | 12, 12, 12 | 10,177 / 9,921 | 3.56 | cuda (model not recorded) |
| RecurrentCNN fixed T=8 (hid 22) | 160 epochs | 0.991 ± 0.002 | 0.991 ± 0.002 | 8, 8, 8 | 9,989 / 9,989 | 2.06 | RTX 3070 Ti (cuda) |
| RecurrentCNN variable T (hid 22) | 160 epochs | 0.983 ± 0.001 | 0.988 ± 0.006 | 8, 12, 12 | 9,989 / 9,989 | 2.17 | RTX 3070 Ti (cuda) |
| CF-Full (hid 16) | 160 epochs | 0.735 ± 0.025 | 0.949 ± 0.041 | 16, 16, 12 | 10,177 / 10,177 | 5.73 | cuda (model not recorded) |
| CF-NoFrag (hid 18) | 160 epochs | 0.655 ± 0.057 | 0.921 ± 0.063 | 12, 12, 16 | 12,745 / 9,811 | 2.50 | cuda (model not recorded) |

Every run has n = 3 seeds, so no row is flagged "preliminary". Training
minutes are the mean per-seed wall-clock training time recorded with the
run; "not a reproduction" flags do not apply (no reduced baseline is
plotted). Hardware is what each source records: NVIDIA GeForce RTX
3070 Ti (CUDA) where the JSON logs a device name, "cuda" with the GPU
model unrecorded for the follow-ups and CF-WallAware 160-epoch runs.

## Key takeaway (numbers from `data.csv`)

With a 160-epoch budget every family reaches at least 0.92 strict
accuracy at its validation-selected depth T* (CF-NoFrag hid 18:
0.921 ± 0.063; CF-Full: 0.949 ± 0.041; RecurrentCNN variable T:
0.988 ± 0.006; RecurrentCNN fixed T=8: 0.991 ± 0.002; CF-WallAware:
0.996 ± 0.002), while at the trained depth T = 8 the CF models stay at
or below 0.91 even then (CF-WallAware 0.903 ± 0.002, CF-Full
0.735 ± 0.025, CF-NoFrag 0.655 ± 0.057). The CF models need rollouts
longer than their training depth to peak: their validation-selected
depths are 12-16 steps, never 8. The fixed-T RecurrentCNN is the
exception: T* = T = 8 at both budgets. At 40 epochs only the
RecurrentCNN baselines are near 0.94 (fixed 0.936 ± 0.015, variable
0.942 ± 0.032); the best 40-epoch CF model is CF-WallAware at
0.684 ± 0.125 - and every 40-epoch RecurrentCNN seed
(0.909-0.972) lies above every 40-epoch CF-WallAware seed (0.563-0.812).
Differences within the seed spread remain inconclusive:
at 160 epochs the top three rows at T* (0.988-0.996) share seed values
around 0.99, so no ranking among them stands; at T = 8, RecurrentCNN
fixed T=8 (0.991 ± 0.002) does sit clearly above CF-WallAware
(0.903 ± 0.002) - a 0.088 gap against sds of 0.002.

## Data source

- `../../week2_results/valT_selection.json` - 40-epoch runs
  (RecurrentCNN fixed T=8, CF-Full, CF-NoFrag hid 16): per-seed
  test strict at T = 8 and at T*, and per-seed T*.
- `../../week2_results/week2_metrics.json` - 40-epoch runs: raw parameter
  counts, per-seed training seconds, dataset sizes (test = 1,000 mazes),
  device name.
- `../../week2_results/wallaware/valT_selection.json` and
  `../../week2_results/wallaware/week2_metrics.json` - CF-WallAware
  40 epochs: per-seed values, raw/effective parameters, minutes, device.
- `../../week2_results/wallaware160/results.json` - CF-WallAware 160
  epochs (E5): per-seed T*, test strict at T* and by depth.
- `../../week2_results/followups/results.json` - CF-Full 160 epochs (E1),
  CF-NoFrag hid 18 at 40 (E2) and 160 epochs (E4), RecurrentCNN variable
  T 40 epochs (E3): per-seed values and effective-parameter protocol.
- `../../week2_results/rcnn160/results.json` - RecurrentCNN hid 22,
  160 epochs: fixed T=8 (E6) and variable T (E7), n = 3 seeds each
  (seeds 0, 1, 2), per-seed sweep, device name.
- `data.csv` - every plotted number: per-seed values at T = 8 and T*,
  means, sample sds, T* depths, parameter counts, training minutes,
  hardware, source paths.
- `../../scripts/run_week2.py` (function `effective_n_params`) - the
  effective-parameter protocol; the CF-NoFrag hid-16 effective count
  (7,857) is measured by the figure script itself with this protocol
  (no recorded measurement exists in `week2_results/`).
- Script: `../../scripts/figures/main_results.py` (re-runnable:
  `python scripts/figures/main_results.py`).

## Caveats

- n = 3 seeds per model; the spread is the sample sd (n−1 denominator).
  Values are fractions in [0, 1]; no shading band is drawn, so no band
  clipping applies.
- T* is selected per seed on the validation set (argmax validation
  strict over candidate depths, ties to smaller T), so it can differ
  between seeds of the same model (T* depths are in `data.csv`).
- Every CF model and both fixed-T RecurrentCNN rows were trained at depth
  T = 8; both variable-T RecurrentCNN rows were trained with the rollout
  depth sampled per batch (4-16).
- The 40-epoch and 160-epoch runs use cosine learning-rate decay over
  their respective epoch counts (recorded in the follow-ups and rcnn160
  results), so budget comparisons here are "epochs + schedule", not
  matched-schedule; `wallaware160/results.json` records no schedule, so
  no decay claim is made for CF-WallAware 160 epochs.
- Effective parameters count only parameter elements with a nonzero
  gradient after one forward+backward pass on a real training batch
  (protocol in `followups/results.json`). Raw equals effective only for
  CF-Full and the RecurrentCNN variants; the exceptions are CF-WallAware
  at both budgets (10,177 / 9,921), CF-NoFrag
  hid 18 (12,745 / 9,811) and CF-NoFrag hid 16 (10,177 / 7,857): with
  fragmentation=False the split-flux output head is computed but
  immediately discarded, so its 2,320 entries (hid 16: 16 out-channels x
  (16x3x3 weights + 1 bias)) receive zero gradient. The CF-NoFrag hid-16
  effective count has no recorded measurement in `week2_results/`; the
  figure script measures it on the fly with the same protocol
  (`scripts/run_week2.py:effective_n_params`, one training-style
  forward+backward on a fixed synthetic curriculum batch, CPU, T=8), so
  the CF-Full vs CF-NoFrag hid-16 comparison at 40 epochs is
  10,177-effective vs 7,857-effective, not raw-vs-raw.
- Strict accuracy requires an exact path match, a harsh metric; the
  40-epoch CF-NoFrag hid-18 training minutes average is inflated by one
  slow seed (seed 0: 10.3 min vs 0.63 min for the other two, as
  recorded).
- The GPU model is not recorded in `week2_results/followups/results.json`
  and `week2_results/wallaware160/results.json` (device: cuda); where
  the JSON logs it, it is the NVIDIA GeForce RTX 3070 Ti. Figures in this
  folder were generated on CPU (matplotlib only); the runs themselves
  were executed on the GPUs noted above.
