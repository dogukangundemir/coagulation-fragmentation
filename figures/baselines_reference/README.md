# Baselines reference (packet figure: F10)

`baselines_reference.png` — published numbers of AKOrN (Sudoku) and WONN
(maze-hard) vs our own reduced-budget runs of the official baseline code,
as a two-panel table.

## Caption

Published vs "ours" comparison table for the two official baselines. Panel A
(AKOrN, Sudoku): published full-budget numbers (100-epoch official protocol,
paper Table 19, p. 36) vs our run of the official code at 40/100 epochs,
17.9 min on an RTX 3070 Ti, with full-board accuracy (exact solved boards,
%) at evaluation depth T=16 and T_eval=128. Panel B (WONN, maze-hard):
published full-budget numbers (9,000-epoch official protocol, paper Table 3,
p. 9) vs our run of the official code at 52/9,000 epochs, 12.7 min, seed 137,
with exact-board accuracy and path F1 (%) at K=1, T=24. Each row carries its
budget (epochs, minutes, runs/seeds, hardware) and config; both of our rows
are reduced-budget pipeline checks of the official code — not
reproductions.

## How to read

- One row per number source, published value (grey strip) on top and our
  run (blue strip) below; centred columns hold the metrics in %, the last
  column states the verdict.
- Italic grey cells ("not run", "not stated in table") mean the number was
  never measured by us or not given in the cited paper table; nothing in
  the figure or in `data.csv` is estimated.
- The budget of every row is printed in the first column (epochs run of the
  official total, wall minutes, number of runs, seed, hardware); the "+-"
  values attached to published numbers are the sd reported by the papers
  (seed count not stated in the cited tables).

## Key takeaway (numbers from `data.csv`)

The official AKOrN Sudoku code reproduces its in-distribution result even at
a heavily reduced budget: our 40/100-epoch, 17.9-minute run reaches 99.3 %
ID board accuracy at T=16 (export cross-check 99.4 %), close to the
published 99.8 ± 0.1. Its out-of-distribution behaviour is far from the
paper at this budget: 12.7 % OOD at T=16 (published: 16.6 %); at the longer
evaluation depth T_eval=128 our run reaches 38.6 % OOD (published full
budget: 51.7 ± 3.3) — the published OOD gain from longer evaluation shows
in direction, though our value stays below the paper's full-budget number.
The official WONN maze-hard code is only validated as a pipeline at
52/9,000 epochs (12.7 min): exact-board accuracy 0.0 % vs the published
76.2 %, while path F1 47.7 % (EMA checkpoint; 75.1 % for the non-EMA
model.pth, n = 1,000 test mazes) shows the model is learning; expected at
0.6 % of the official epoch budget. Parameter count matches the paper:
396,293 raw vs "0.396 M" (effective count not measured). Both of our rows
are single runs, are labelled "not a reproduction", and no conclusion about
the papers' claims should be drawn from them.

## Data source

- `data.csv` (next to this README) — every number shown in the figure, one
  row per metric and source, with budget, hardware and provenance.
- Our run records live outside this repo, under a baselines directory
  (re-read by the script on each run):
  - `<baselines-dir>/akorn_tools/out/run_record.json` — our AKOrN run:
    epochs, wall time, GPU caps, eval protocol and all accuracies.
  - `<baselines-dir>/wonn_tools/out/run_record.json` — our WONN run:
    config (L=1, T=24, ch=128, heads 8, gamma 0.1, GroupNorm, bf16 AMP,
    batch 32 vs official 64), eval metrics, trained-model log.
  - `<baselines-dir>/wonn_tools/out/train_log.txt` — raw parameter count
    of our WONN model.
- Script: `scripts/figures/baselines_reference.py` (rerunnable; published
  constants cited to AKOrN Table 19, p. 36 and WONN Table 3, p. 9).

## Regenerating the figure

Pass the directory that holds `akorn_tools/` and `wonn_tools/` (the
official baseline checkouts with their `out/run_record.json` files) on the
command line or through the environment:

```
python scripts/figures/baselines_reference.py --baselines-dir DIR
# or: CF_BASELINES_DIR=DIR python scripts/figures/baselines_reference.py
```

If neither `--baselines-dir` nor `CF_BASELINES_DIR` is set, no run records
are read: the figure and `data.csv` are regenerated from the tracked
`data.csv` next to this README only (the script says so on stdout), and
fields that exist only in the run records (commit hashes, AKOrN export
cross-check) read "not recorded".


## Caveats

- Our runs are reduced-budget pipeline checks of the official code (AKOrN:
  40/100 epochs, 17.9 min; WONN: 52/9,000 epochs, 12.7 min) — "not a
  reproduction" of the papers' settings; the published rows were not run by
  us.
- Single runs, no seed spread: AKOrN seed not recorded, WONN seed 137. The
  published "+-" is the sd reported by the papers with seed count not
  stated in the cited tables; small differences (e.g. 99.3 vs 99.8) are
  inconclusive at n = 1.
- Deviations from the official protocols are recorded in the run records:
  fewer epochs, checkpoint frequency, num_workers=0 (Windows), WONN batch
  32 instead of 64 (GPU memory cap, doubles optimizer steps per epoch),
  WONN evals at reduced depth K=1 (as in the cited table row).
- AKOrN OOD at T_eval=128 was evaluated after this figure was first
  rendered (38.6 % at 40/100 epochs vs the published 51.7 ± 3.3 at 100
  epochs); the script re-reads it from the run record on each rerun.
- WONN board accuracy requires an exact full-grid label match, so path F1
  (47.7 % EMA / 75.1 % non-EMA) can be high while board accuracy is 0.0 %.
- Hardware: our runs on an NVIDIA RTX 3070 Ti, 8 GB (memory capped at 70 %,
  power monitored below 70 % of limit); hardware of the published rows is
  not stated in the cited tables.
