# Commitment over time: per-step path probability (packet figure F4)

All numbers below are recomputed by `scripts/figures/commitment.py` from the stored checkpoint and written to `data.csv` in this folder (one row per maze and recurrence step t); the README numbers are read back from that CSV.

Model: **CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0**, 10177 parameters, checkpoint `week2_results/followups/E1_CF-Full_seed0.pt`; validation-selected inference depth T\* = 16 (test strict solve rate 0.969 at T\*, from `week2_results/followups/results.json`). Input: walkable mask + start + goal one-hot channels; output: per-cell path logit, sigmoid -> P(path), decision threshold 0.5.

Mazes (level-5 test split, 1000 mazes, 11x11, same pipeline as `run_week2.load_data`): selection rule = test maze 0 and the test maze with the longest true path (ties by maze index).

---

## `commitment_maze0.png` - test maze 0 (true path 7 cells)

**Caption.** Top: the ground truth and the per-cell path probability P(path) after t = 1, 2, 4, 8, 12, 16 recurrence steps of CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0 (one shared rollout, read out at each t; the number under each panel is the wrong-cell count of the thresholded prediction at that t, \"solved\" = 0). Bottom: median and IQR of P(path) over t = 1..16, separately for the true-path cells and the off-path free cells.
**How to read.** Map panels: free cells are coloured by P(path) on the viridis scale (dark = 0, yellow = 1, see colour bar), wall cells are dark grey, the true path is outlined in black, start = circle, goal = star. Curve: blue = median over the 7 true-path cells, vermillion = median over the 42 off-path free cells; the shaded bands are the IQR (25th-75th percentile over the cells of each group, n stated in the legend), clipped to [0, 1] because a probability cannot leave that range; the dashed grey line is the 0.5 decision threshold; dots mark the six steps shown as maps.

**Key takeaway.** On test maze 0 the median P(path) on the 7 true-path cells moves from 0.99 at t = 1 to 1.00 at t = 16 (IQR width 0.006 -> 0.003), while the median on the 42 off-path free cells falls from 0.98 to 0.00 (IQR width 0.119 -> 0.004): the loop commits to the true path and suppresses the alternatives. The thresholded prediction has 33 wrong cells at t = 1 and first fully solved (strict) at t = 9 and stays solved through t = 16.
**Data source.** `data.csv` in this folder (rows with maze = 0); checkpoint `week2_results/followups/E1_CF-Full_seed0.pt`; data pipeline `scripts/run_week2.py` (level-5 test split).

**Caveats.** Single maze, single seed (0); median + IQR over cells within this one maze (n = 7 true-path and 42 off-path free cells, no averaging over mazes); wall cells are excluded from both groups (the model can still assign them probability, which counts as wrong cells in the strict metric); t = 9..16 is beyond the training rollout depth T = 8; the IQR is a within-maze spread, not a seed uncertainty.

**Packet figure F4, part 1 of 2 (other part: figures/maze_paths/thinking_steps.png)**

---

## `commitment_maze248.png` - test maze 248 (true path 39 cells)

**Caption.** Top: the ground truth and the per-cell path probability P(path) after t = 1, 2, 4, 8, 12, 16 recurrence steps of CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0 (one shared rollout, read out at each t; the number under each panel is the wrong-cell count of the thresholded prediction at that t, \"solved\" = 0). Bottom: median and IQR of P(path) over t = 1..16, separately for the true-path cells and the off-path free cells.
**How to read.** Map panels: free cells are coloured by P(path) on the viridis scale (dark = 0, yellow = 1, see colour bar), wall cells are dark grey, the true path is outlined in black, start = circle, goal = star. Curve: blue = median over the 39 true-path cells, vermillion = median over the 10 off-path free cells; the shaded bands are the IQR (25th-75th percentile over the cells of each group, n stated in the legend), clipped to [0, 1] because a probability cannot leave that range; the dashed grey line is the 0.5 decision threshold; dots mark the six steps shown as maps.

**Key takeaway.** On test maze 248 the median P(path) on the 39 true-path cells moves from 0.99 at t = 1 to 0.99 at t = 16 (IQR width 0.009 -> 0.005), while the median on the 10 off-path free cells falls from 0.62 to 0.00 (IQR width 0.974 -> 0.002): the loop commits to the true path and suppresses the alternatives. The thresholded prediction has 5 wrong cells at t = 1 and first fully solved (strict) at t = 3 and stays solved through t = 16.
**Data source.** `data.csv` in this folder (rows with maze = 248); checkpoint `week2_results/followups/E1_CF-Full_seed0.pt`; data pipeline `scripts/run_week2.py` (level-5 test split).

**Caveats.** Single maze, single seed (0); median + IQR over cells within this one maze (n = 39 true-path and 10 off-path free cells, no averaging over mazes); wall cells are excluded from both groups (the model can still assign them probability, which counts as wrong cells in the strict metric); t = 9..16 is beyond the training rollout depth T = 8; the IQR is a within-maze spread, not a seed uncertainty.

**Packet figure F4, part 1 of 2 (other part: figures/maze_paths/thinking_steps.png)**

