# Baselines reference (packet figure: F10)

`baselines_reference.png` — published numbers of AKOrN (Sudoku) and WONN
(maze-hard) vs our own runs of the official baseline code, as a two-panel
table.

## Caption

Published vs "ours" comparison table for the two official baselines. Panel A
(AKOrN, Sudoku): published full-budget numbers (100-epoch official protocol,
paper Table 19, p. 36) vs our run of the official code at the full
100/100 epochs (41.5 min on an RTX 3070 Ti, single unseeded run),
full-board accuracy (exact solved boards, %), evaluated
with the official eval script at evaluation depth T=16 and T_eval=128;
the plotted accuracies are the re-run official eval whose full stdout the
run record preserves. Panel B (WONN, maze-hard): published full-budget
numbers (9,000-epoch official protocol, paper Table 3, p. 9) vs our run of
the official code at 150/9,000 epochs (33.1 min, seed 137, batch 32 vs
official 64), with exact-board accuracy and path F1 (%) at K=1, T=24. Each
row carries its budget (epochs, minutes, runs/seeds, hardware) and config.
Our AKOrN row reached the full official 100-epoch budget on a single
unseeded seed (verdict "full official budget, 1 seed (preliminary)"); the
WONN row is a reduced-budget pipeline check — "not a reproduction".

## How to read

- One row per number source, published value (grey strip) on top and our
  run (blue strip) below; centred columns hold the metrics in %, the last
  column states the verdict.
- Italic grey cells ("not run", "not stated in table") mean the number was
  never measured by us or not given in the cited paper table; nothing in
  the figure or in `data.csv` is estimated.
- The budget of every row is printed in the first column (epochs run of
  the official total, wall minutes, number of runs, seed, hardware); the
  "+-" values attached to published numbers are the sd reported by the
  papers (seed count not stated in the cited tables).

## Key takeaway (numbers from `data.csv`)

The official AKOrN Sudoku code, trained to its full official budget
(100/100 epochs, 41.5 min in two phases, single unseeded run), reproduces
the published Sudoku numbers on that single seed: ID board accuracy
99.8 % vs 99.8 ± 0.1 at T=16, OOD board accuracy at the long evaluation
depth T_eval=128 51.6 % vs published 51.7 ± 3.3, and ID saturated at
100.0 % at T_eval=128 (paper reference quoted in the run record: 100.0);
at the training depth T=16 the OOD value is 19.7 % (published: 16.6 %).
These are the values of the re-run official eval whose full stdout the
run record preserves, re-read from those logs; a superseded earlier
record revision of the same unseeded checkpoint read 99.9 / 19.4 / 51.7 %
(ID / OOD at T=16, OOD at T_eval=128) — up to 0.4 pp of run-to-run
re-eval drift, not seed spread — and a prediction-grid export re-check
confirms the final checkpoint (ID 100.0 %, 1,000/1,000 grids; OOD
51.6 %, 9,284/18,000 grids at T_eval=128). All AKOrN numbers come from
the final EMA checkpoint with the official eval script; the raw
(non-EMA) model reached 98.8 % ID at T=16, a run-record value — the
surviving training log ends at epoch 40 (final raw 94.8 % / EMA 99.1 %),
so the epoch-100 raw value cannot be re-derived from preserved logs.
Being a single unseeded run, the row stays "preliminary".
The official WONN maze-hard code is only validated as a pipeline at
150/9,000 epochs (33.1 min, 1.7 % of the official budget): exact-board
accuracy 0.1 % vs the published 76.2 %, while path F1 84.0 % (EMA
checkpoint; 93.2 % with board 8.4 % for the non-EMA model.pth, n = 1,000
test mazes) shows fast learning that is still far from convergence — "not
a reproduction". Parameter count matches the paper: 396,293 raw vs
"0.396 M" (effective count not measured).

## Data source

- `data.csv` (next to this README) — every number shown in the figure, one
  row per metric and source, with budget, hardware and provenance.
- Our run records live outside this repo, under a baselines directory
  (re-read by the script on each run):
  - `<baselines-dir>/akorn_tools/out/run_record.json` — our AKOrN run:
    official config and epoch count, wall time, GPU caps, eval protocol,
    the single-seed statement and all accuracies; it now points to its
    full-stdout eval logs and quotes the superseded earlier-revision
    values in its eval notes.
  - `<baselines-dir>/akorn_tools/out/eval_100ep_T16.log`,
    `eval_100ep_T128.log` — the re-run official evals of the final EMA
    checkpoint ("Vote accuracy" lines); the plotted values come from
    these logs and are re-read as a cross-check.
  - `<baselines-dir>/akorn_tools/out/_export_id.log`, `_export_ood.log` —
    exact-board accuracy of the exported prediction-grid re-check of the
    final EMA checkpoint (ID 1,000/1,000 grids, OOD 9,284/18,000 grids).
  - `<baselines-dir>/akorn_tools/out/train_reduced.log` — the surviving
    training log; it ends at epoch 40, which bounds the provenance of the
    raw (non-EMA) run-record value.
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

The script drops AKOrN's "not a reproduction" verdict only when the run
record shows the official config trained to the full official epoch count
(`epochs_run >= official_epochs`); the single seed is always stated. If
neither `--baselines-dir` nor `CF_BASELINES_DIR` is set, no run records
are read: the figure and `data.csv` are regenerated from the tracked
`data.csv` next to this README only (the script says so on stdout). The
re-eval, export-re-check, earlier-revision and surviving-log numbers then
fall back to their own `data.csv` rows, so the footnotes regenerate;
fields that live only in the run records (commit hashes, the AKOrN
paper-reference quote, the export-grid counts) read "not recorded" / "not
stated".

## Caveats

- Our AKOrN run reached the official 100-epoch budget and the official
  eval protocol, but is a single unseeded run and was trained in two
  phases (epochs 1-40, then a minimal resume wrapper continued 41-100
  restoring model, Adam and EMA state; no epoch restarted), with further
  recorded deviations: --checkpoint_every=10, --num_workers=0 (Windows);
  so it stays "preliminary" (fewer than 3 seeds), not a seeded
  reproduction.
- Unseeded re-eval drift (AKOrN): the four plotted accuracies are the
  values of the re-run official eval recorded as the run record's
  full-stdout logs (eval_100ep_T16.log / eval_100ep_T128.log); a
  superseded earlier record revision of the same unseeded checkpoint read
  ID 99.9 / OOD 19.4 at T=16 and OOD 51.7 at T_eval=128 — up to 0.4 pp
  of run-to-run re-eval drift, so single-run differences of that size are
  re-eval noise, not seed spread.
- Export cross-check (AKOrN): the board accuracy recomputed from the
  exported prediction grids of the final EMA checkpoint reads ID 100.0 %
  (1,000/1,000 grids) and OOD 51.6 % (9,284/18,000 grids) at T_eval=128,
  agreeing with the official eval within the same run-to-run variance.
- The raw (non-EMA) ID value of 98.8 % at T=16 survives only in the run
  record: the preserved training log covers epochs 1-40 only (its final
  block reads raw 94.8 % / EMA 99.1 %), the phase-2 (epoch 41-100)
  training log is not preserved, so the epoch-100 raw value cannot be
  re-derived from preserved logs.
- Our WONN run is a reduced-budget pipeline check (150/9,000 epochs,
  1.7 % of the official budget) — "not a reproduction" of the paper's
  setting; single seed 137 and batch 32 instead of the official 64 (GPU
  memory cap, doubles optimizer steps per epoch), official LR schedule
  unchanged.
- Published "+-" values are the sd reported by the papers with seed count
  not stated in the cited tables; our rows are single runs (AKOrN: one
  unseeded run; WONN: seed 137), so small differences (e.g. 19.7 vs 16.6
  OOD at T=16) are inconclusive at n = 1.
- WONN board accuracy requires an exact full-grid label match, so path F1
  (84.0 % EMA / 93.2 % non-EMA) can be high while board accuracy is
  0.1 %; the EMA weights lag the online weights at this early stop.
- Hardware: our runs on an NVIDIA RTX 3070 Ti, 8 GB (AKOrN memory capped
  at 75 % per process, 232 W power cap; WONN memory capped at 70 %, power
  monitored below the cap); hardware of the published rows is not stated
  in the cited tables.
