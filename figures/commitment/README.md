# Commitment over time: per-step predictions (packet figure F4)

All numbers below are recomputed by `scripts/figures/commitment.py` from the stored checkpoints and written to `data.csv` in this folder (one row per model, maze and step t; rows with model = WONN and maze = -1 aggregate all 1000 Maze-hard test mazes, storing the per-t median and IQR of the fragment count, the fraction of mazes board-matching at that t (board_ok) and the fraction with at most 1 fragment at that t (frac_le1)); the README numbers are read back from that CSV.

Parts: 1. CF-Full on level-5 test mazes (`commitment_maze0.png`, `commitment_maze248.png`); 2. WONN on one Maze-hard test maze (`commitment_wonn_mazehard_test0.png`); 3. CF-WallAware on the same Maze-hard test maze (`commitment_cfwallaware_mazehard_test0.png`).

Model (part 1): **CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0**, 10177 parameters, checkpoint `week2_results/followups/E1_CF-Full_seed0.pt`; validation-selected inference depth T\* = 16 (test strict solve rate 0.969 at T\*, from `week2_results/followups/results.json`). Input: walkable mask + start + goal one-hot channels; output: per-cell path logit, sigmoid -> P(path), decision threshold 0.5.

Mazes (part 1, level-5 test split, 1000 mazes, 11x11, same pipeline as `run_week2.load_data`): selection rule = test maze 0 and the test maze with the longest true path (ties by maze index). Mazes (parts 2-3, Maze-hard 30x30, HF sapientinc/maze-30x30-hard-1k test split, 1000 mazes): selection rule = test maze 0 for both models, so both panels show the same maze.

---

## `commitment_maze0.png` - level-5 test maze 0 (true path 7 cells)

**Caption.** Top: the ground truth and the per-cell path probability P(path) after t = 1, 2, 4, 8, 12, 16 recurrence steps of CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0 (one shared rollout, read out at each t; the number under each panel is the wrong-cell count of the thresholded prediction at that t, \"solved\" = 0). Bottom: median and IQR of P(path) over t = 1..16, separately for the true-path cells and the off-path free cells.
**How to read.** Map panels: free cells are coloured by P(path) on the viridis scale (dark = 0, yellow = 1, see colour bar), wall cells are dark grey, the true path is outlined in black, start = circle, goal = star. Curve: blue = median over the 7 true-path cells, vermillion = median over the 42 off-path free cells; the shaded bands are the IQR (25th-75th percentile over the cells of each group, n stated in the legend), clipped to [0, 1] because a probability cannot leave that range; the dashed grey line is the 0.5 decision threshold; dots mark the six steps shown as maps.

**Key takeaway.** On test maze 0 the median P(path) on the 7 true-path cells moves from 0.99 at t = 1 to 1.00 at t = 16 (IQR width 0.006 -> 0.003), while the median on the 42 off-path free cells falls from 0.98 to 0.00 (IQR width 0.119 -> 0.004): the loop commits to the true path and suppresses the alternatives. The thresholded prediction has 33 wrong cells at t = 1 and first fully solved (strict) at t = 9 and stays solved through t = 16.
**Data source.** `data.csv` in this folder (rows with model = CF-Full, maze = 0); checkpoint `week2_results/followups/E1_CF-Full_seed0.pt`; data pipeline `scripts/run_week2.py` (level-5 test split).

**Caveats.** Single maze, single seed (0); median + IQR over cells within this one maze (n = 7 true-path and 42 off-path free cells, no averaging over mazes); wall cells are excluded from both groups (the model can still assign them probability, which counts as wrong cells in the strict metric); t = 9..16 is beyond the training rollout depth T = 8; the IQR is a within-maze spread, not a seed uncertainty.

**Packet figure: F4**

---

## `commitment_maze248.png` - level-5 test maze 248 (true path 39 cells)

**Caption.** Top: the ground truth and the per-cell path probability P(path) after t = 1, 2, 4, 8, 12, 16 recurrence steps of CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0 (one shared rollout, read out at each t; the number under each panel is the wrong-cell count of the thresholded prediction at that t, \"solved\" = 0). Bottom: median and IQR of P(path) over t = 1..16, separately for the true-path cells and the off-path free cells.
**How to read.** Map panels: free cells are coloured by P(path) on the viridis scale (dark = 0, yellow = 1, see colour bar), wall cells are dark grey, the true path is outlined in black, start = circle, goal = star. Curve: blue = median over the 39 true-path cells, vermillion = median over the 10 off-path free cells; the shaded bands are the IQR (25th-75th percentile over the cells of each group, n stated in the legend), clipped to [0, 1] because a probability cannot leave that range; the dashed grey line is the 0.5 decision threshold; dots mark the six steps shown as maps.

**Key takeaway.** On test maze 248 the median P(path) on the 39 true-path cells moves from 0.99 at t = 1 to 0.99 at t = 16 (IQR width 0.009 -> 0.005), while the median on the 10 off-path free cells falls from 0.62 to 0.00 (IQR width 0.974 -> 0.002): the loop commits to the true path and suppresses the alternatives. The thresholded prediction has 5 wrong cells at t = 1 and first fully solved (strict) at t = 3 and stays solved through t = 16.
**Data source.** `data.csv` in this folder (rows with model = CF-Full, maze = 248); checkpoint `week2_results/followups/E1_CF-Full_seed0.pt`; data pipeline `scripts/run_week2.py` (level-5 test split).

**Caveats.** Single maze, single seed (0); median + IQR over cells within this one maze (n = 39 true-path and 10 off-path free cells, no averaging over mazes); wall cells are excluded from both groups (the model can still assign them probability, which counts as wrong cells in the strict metric); t = 9..16 is beyond the training rollout depth T = 8; the IQR is a within-maze spread, not a seed uncertainty.

**Packet figure: F4**

---

## `commitment_wonn_mazehard_test0.png` - WONN on Maze-hard test maze 0 (true path 112 cells)

**Caption.** official code, L=1, T=24, ch=128, heads=8, 150/9000 epochs, seed 137 - reduced budget, not a reproduction rolled out on Maze-hard test maze 0 over its T = 24 steps; the model's own readout head is applied to every intermediate step of the rollout. Top: the predicted class grid after t = 6, 8, 12, 16, 20, 24 steps (the number under each panel = path fragments / cells disagreeing with the reference grid). Middle: per-step counts for this maze. Bottom: the fragment count over ALL 1000 Maze-hard test mazes (median line + IQR band), with this maze on top.
**How to read.** Map panels: blue = cells predicted as path, white = free, dark grey = wall, orange = predicted start, yellow = predicted goal; the black outline is the reference route, circle = start, star = goal. Middle panel: green line with dots (left axis) = number of path fragments (4-connected components of path-predicted cells); blue and vermillion lines (right axis) = cells predicted as path and cells disagreeing with the reference grid. Bottom panel: grey line = per-step median fragment count over the 1000 test mazes, band = IQR over mazes (counts are unbounded, so the band needs no clipping); dashed green = this maze. All series are counts on one Maze-hard maze (n = 1) or over n = 1000 mazes; there is no seed spread (single seed).

**Key takeaway.** On test maze 0 the predicted path starts as scattered fragments - 84 fragments at t = 8 (the peak for this maze) - and merges to 3 fragments by t = 24, while the number of cells predicted as path settles at 118 (reference route: 112 cells) and the disagreements with the reference grid fall from 488 at the peak to 28 at t = 24; the maze is still not board-solved at t = 24 (board match = no). Across all 1000 test mazes the same shape holds: the median fragment count peaks at t = 9 (85 fragments) and falls to 3 at t = 24 (IQR 2..5 at t = 24); 0.163 of the 1000 mazes have at most 1 fragment at t = 24, and the board-match rate recomputed here on CPU is 0.096 (0.084 in the run record's official eval of the same checkpoint on GPU).
**Data source.** `data.csv` in this folder (rows with model = WONN; maze = 0 = this maze, maze = -1 = whole test split, n = 1000 mazes); WONN checkpoint and inputs are external, given at run time via --baselines-dir / CF_BASELINES_DIR: official model code under `WONN/` (read-only), checkpoint `WONN/runs/maze/wonn_reduced_L1T24_ch128_s137_bs32/model.pth` (final raw weights), run record `wonn_tools/out/run_record.json`, data `wonn_tools/data/maze-30x30-hard-1k/test/` (.npy conversion of the HF test csv, re-validated here with cf.data.maze_hard, structural checks).

**Caveats.** Budget: WONN: 150 of 9000 official epochs, seed 137, batch 32 instead of 64, single seed -> preliminary. Reduced-budget run over the official code, not a reproduction; single seed -> preliminary. The steps before t = 6 predict no path cells at all (the oscillator field needs several steps before any cell crosses the path class). The board-match rate differs between this CPU rollout (0.096) and the record's GPU eval (0.084): the initial theta field is random (seeded, seed 137) and at this budget the argmax is not numerically stable across devices. Fragment counts are a helper quantity defined here (4-connected components of path-predicted cells), not a metric of the official eval. Wall cells are never predicted as path here (wall hits = 0 on this maze at every t).

**Packet figure: F4**

---

## `commitment_cfwallaware_mazehard_test0.png` - CF-WallAware on Maze-hard test maze 0 (true path 112 cells)

**Caption.** CF-WallAware maze-hard model: hid 16, cross_kernel=True, wall-aware, 120 epochs (seed 0 of 3), trained at T_train = 24 on the same Maze-hard test maze as the WONN panel: per-cell P(path) after t = 1, 4, 8, 16, 24 steps (one shared rollout, read out at each t; the number under each panel is the wrong-cell count of the thresholded prediction), and median + IQR of P(path) over t = 1..24 for the true-path and off-path free cells.
**How to read.** Same reading as the CF-Full panels above: free cells coloured by P(path) (viridis), true path outlined, bands = IQR clipped to [0, 1]; blue = median over the 112 true-path cells, vermillion = median over the 476 off-path free cells; dashed grey = 0.5 decision threshold. Model settings relevant to the plot are in the caption (hid 16, cross_kernel=True, wall-aware, 120 epochs, T_train = 24, seed 0 of 3).

**Key takeaway.** On test maze 0 the median P(path) on the 112 true-path cells rises from 0.44 at t = 1 to 0.91 at t = 24 and the median on the 476 off-path free cells falls from 0.29 to 0.007, but the suppression does not finish: the thresholded prediction keeps 111 wrong cells at t = 24 (152 at t = 1) and never reaches a strict solve (strict = 0 at every t = 1..24). On the full test split this model scores valid-path 0 and board match 0 at every T (seed 0: IoU 0.548, recall 0.979, extra cells 95.9 at T\* = 16, from the maze-hard results.json cited in `data source`).
**Data source.** `data.csv` in this folder (rows with model = CF-WallAware, maze = 0); checkpoint and sweep records are external, given at run time via --cf-results-root / CF_RESULTS_ROOT: `maze_hard_cf/states/CF-WallAware_seed0.pt` and `maze_hard_cf/results.json` (test-split sweep numbers cited, not recomputed here).

**Caveats.** Single maze, single seed (0 of 3; the other seeds exist for the test-split sweep but this rollout shows seed 0); median + IQR over cells within this one maze. The Maze-hard grids are non-perfect (several equally short routes exist), so board match and valid-path are not the same metric; the wrong-cell count here compares the thresholded mask against the single reference route. Wall cells are masked to -20 by construction, so wall hits are 0 by design (reported as a sanity check, not a discriminator).

**Packet figure: F4**

