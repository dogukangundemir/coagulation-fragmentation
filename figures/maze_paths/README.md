# Maze-path predictions: what every model actually outputs

All numbers below are recomputed by `scripts/figures/maze_paths.py` from the stored checkpoints and written to `data.csv` (one row per model, seed, inference depth T and maze); the README numbers are read back from that CSV.

Models (state dicts loaded with the repo's own factories; seed 0 shown in all instance-level figures, three seeds aggregated wherever a mean is shown; checkpoints live in the main repository's results archive):

| model | params (raw / effective) | training | check-point |
|---|---|---|---|
| CF-Full (hid 16, 160 epochs, trained at fixed T=8) | 10177 | fixed T=8, 160 epochs | `followups/E1_CF-Full_seed0/1/2.pt` |
| CF-NoFrag (hid 18, fragmentation disabled, 160 epochs, trained at fixed T=8) | 12745 raw / 9811 effective | fixed T=8, 160 epochs | `followups/E4_CF-NoFrag160_seed0/1/2.pt` |
| RecurrentCNN (hid 22, 40 epochs, rollout depth T drawn uniformly from 4..16 during training) | 9989 | T sampled from 4..16 per batch, 40 epochs | `followups/E3_RecurrentCNN_seed0/1/2.pt` |
| RecurrentCNN (hid 22, 40 epochs, trained at fixed T=8) | 9989 | fixed T=8, 40 epochs | `RecurrentCNN_seed0/1/2.pt` |
| CF-Full (hid 16, 40 epochs, trained at fixed T=8) | 10177 | fixed T=8, 40 epochs | `CF-Full_seed0/1/2.pt` |

Params: raw = state-dict element count; effective = parameters that receive gradients (nonzero-gradient rule, `followups/results.json` `param_counts`, one real training batch at T = 8). With fragmentation disabled only the 9811 effective of the 12745 raw CF-NoFrag parameters train, so CF-NoFrag effectively has *fewer* trainable parameters than CF-Full (10177); for every other model raw = effective.

Scoring: a maze counts as **solved (strict)** if the thresholded prediction equals the true-path cell mask exactly (on these perfect mazes strict coincides with exact-match, so a wrong-cell count of zero is equivalent). Every model is evaluated at its validation-selected inference depth T\*; the seed-0 rates at those depths match the stored selection results in the results archive (`followups/results.json` and `valT_selection.json`) to within 0.001 (checked in the script, which exits non-zero on a mismatch). Inputs: walkable mask + start + goal one-hot channels; threshold 0.5 on the sigmoid output.

Split: level-5 test split (1000 mazes, 11x11) - the same pipeline as the repo's training driver (`load_data`).

---

## Path gallery (`path_gallery_1.png`, `path_gallery_2.png`)

Packet figure: F6

**Selection rule for the eight rows** (level-5 test split, seed-0 models, each at its own T\*): 4 mazes drawn with numpy RNG seed 0; 2 mazes where the five models disagree on the solve verdict (ranked by total wrong cells across models, largest first); the 2 mazes with the longest true path. Ties by maze index, no repeats. Chosen mazes: test maze 269 (random pick), test maze 510 (random pick), test maze 635 (random pick), test maze 848 (random pick), test maze 256 (models disagree), test maze 870 (models disagree), test maze 248 (longest true path), test maze 370 (longest true path).

**Caption.** Rows are test mazes; columns are the ground truth and every model's thresholded prediction at its seed-0 T\* (model name and T at inference stated above each column, and the header row is repeated on every maze row so each PNG is readable on its own). Walls are dark, free cells light, the true path is outlined in black, start = circle, goal = star.
**How to read.** Blue = predicted cell on the true path (correct); vermillion = false positive (extra predicted cell); purple = missed cell (on the true path, not predicted); the number below each panel is the wrong-cell count, "solved" for zero.

**Key takeaway.** On the whole test split, all five models solve the same 408 mazes and none solves 0 mazes; 592 mazes get mixed verdicts (these two gallery rows show such disagreement cases). Across the 8 gallery rows each model solves: cf-full-160ep: 6/8, cf-nofrag-160ep: 7/8, rcnn-varT: 4/8, rcnn-fixedT8: 6/8, cf-full-40ep: 3/8.
**Data source.** `data.csv` in this folder (rows with split = test, seed = 0, T = the model's T\*); checkpoints listed in the table above, in the main repository's results archive (`followups/*.pt`, root-level `*.pt`).

**Caveats.** Single seed (0) per panel; the four random picks are a tiny sample; disagreement rows are selected to be hard and are not typical mazes.

---

## Whole-test-split coverage (`coverage_<model>.png`, one PNG per model)

Packet figure: F9

**Caption.** For every model, all 1000 test mazes as a 25x40 grid of tiles, one tile border colour per outcome at the seed-0 validation-selected T\*: bluish green = solved, vermillion = not solved. The maze itself is drawn inside the tile (walls dark, true path tinted).
**How to read.** Tiles are sorted by true path length ascending, left to right, top to bottom; the title carries the seed-0 strict solve rate and the mean +- sample sd over the three seeds (each seed at its own T\*, n = 1000 mazes per seed; sd is the sample sd over 3 seeds).

**Key takeaway.**
- CF-Full (hid 16, 160 epochs, trained at fixed T=8): strict solve rate 0.969 (969/1000, seed 0 at T\* = 16); 0.949 +- 0.041 over 3 seeds at their own T\*
- CF-NoFrag (hid 18, fragmentation disabled, 160 epochs, trained at fixed T=8): strict solve rate 0.867 (867/1000, seed 0 at T\* = 12); 0.921 +- 0.063 over 3 seeds at their own T\*
- RecurrentCNN (hid 22, 40 epochs, rollout depth T drawn uniformly from 4..16 during training): strict solve rate 0.944 (944/1000, seed 0 at T\* = 12); 0.942 +- 0.032 over 3 seeds at their own T\*
- RecurrentCNN (hid 22, 40 epochs, trained at fixed T=8): strict solve rate 0.940 (940/1000, seed 0 at T\* = 8); 0.936 +- 0.015 over 3 seeds at their own T\*
- CF-Full (hid 16, 40 epochs, trained at fixed T=8): strict solve rate 0.459 (459/1000, seed 0 at T\* = 12); 0.398 +- 0.064 over 3 seeds at their own T\*

**Data source.** `data.csv` in this folder (split = test, seed 0 for the tiles, all seeds for the mean).

**Caveats.** Tile colours are binary (strict pass/fail); a "not solved" tile may be one wrong cell or a completely wrong path; the 3-seed mean uses each seed's own T\* (T\* differs between seeds for some models).

---

## Failure anatomy (`failure_anatomy.png`)

Packet figure: F9

**Caption.** Top row: one histogram per model of the number of wrong cells per maze, over the unsolved test-maze evaluations (3 seeds pooled, each seed at its own validation-selected T\*, n = 3000 evaluations per model = 3 seeds x 1000 mazes). Bottom: strict solve rate by true path length in 6 fixed bins (mean over the 3 seeds, error bars = sample sd, each seed at its own T\*).
**How to read.** Histogram x = wrong cells (extra + missed, only unsolved evaluations, solved mazes are NOT in the histogram - their count is in the panel caption), y = number of mazes on a log scale, bars binned in steps of 2 wrong cells; each model keeps its Okabe-Ito colour in both panels. Bottom panel: x = true path length (bin ranges and n = mazes per bin are printed under each bin), y = strict solve rate between 0 and 1; error bars are clipped to [0, 1] because the rate cannot leave that range.

**Key takeaway.** Failure depth differs between models, not just failure frequency:
- CF-Full (hid 16, 160 epochs, trained at fixed T=8): unsolved in 154 of 3000 evaluations; median 15.0 wrong cells per failed maze (max 38); solve rate 0.986 on the shortest bin (3-9 cells) vs 0.500 on the longest (39 cells).
- CF-NoFrag (hid 18, fragmentation disabled, 160 epochs, trained at fixed T=8): unsolved in 236 of 3000 evaluations; median 4.0 wrong cells per failed maze (max 28); solve rate 0.929 on the shortest bin (3-9 cells) vs 0.833 on the longest (39 cells).
- RecurrentCNN (hid 22, 40 epochs, rollout depth T drawn uniformly from 4..16 during training): unsolved in 175 of 3000 evaluations; median 2.0 wrong cells per failed maze (max 21); solve rate 0.981 on the shortest bin (3-9 cells) vs 0.000 on the longest (39 cells).
- RecurrentCNN (hid 22, 40 epochs, trained at fixed T=8): unsolved in 193 of 3000 evaluations; median 2.0 wrong cells per failed maze (max 11); solve rate 0.972 on the shortest bin (3-9 cells) vs 0.333 on the longest (39 cells).
- CF-Full (hid 16, 40 epochs, trained at fixed T=8): unsolved in 1806 of 3000 evaluations; median 5.0 wrong cells per failed maze (max 39); solve rate 0.499 on the shortest bin (3-9 cells) vs 0.000 on the longest (39 cells).

Per-bin strict solve rate (mean over 3 seeds +- sample sd, each seed at its own T\*):

| true path length (cells) | n | CF-Full 160 epochs | CF-NoFrag 160 epochs | RecurrentCNN T sampled 4-16 | RecurrentCNN fixed T=8 | CF-Full 40 epochs |
|---|---|---|
| 3-9 | 398 | 0.986 +- 0.023 | 0.929 +- 0.057 | 0.981 +- 0.015 | 0.972 +- 0.009 | 0.499 +- 0.222 |
| 11-15 | 306 | 0.985 +- 0.019 | 0.938 +- 0.061 | 0.960 +- 0.033 | 0.936 +- 0.009 | 0.431 +- 0.126 |
| 17-21 | 210 | 0.894 +- 0.078 | 0.911 +- 0.077 | 0.929 +- 0.041 | 0.919 +- 0.029 | 0.273 +- 0.201 |
| 23-27 | 61 | 0.825 +- 0.109 | 0.880 +- 0.100 | 0.836 +- 0.071 | 0.836 +- 0.000 | 0.131 +- 0.118 |
| 29-33 | 23 | 0.696 +- 0.157 | 0.783 +- 0.157 | 0.507 +- 0.133 | 0.768 +- 0.140 | 0.087 +- 0.087 |
| 39 | 2 | 0.500 +- 0.000 | 0.833 +- 0.289 | 0.000 +- 0.000 | 0.333 +- 0.289 | 0.000 +- 0.000 |

**Data source.** `data.csv` in this folder: `wrong_cells` rows at each model's T\* (histograms; 3 seeds pooled) and `strict` rows at T\* (binned rates, sd over the 3 seeds). Checkpoints as in the model table above.

**Caveats.** The histogram pools the three seeds, which use different T\* for some models; wrong cells = extra + missed, so a one-cell slip and one-cell duplication both count and a near-perfect prediction can still fail strict scoring; sample sd over only 3 seeds is a rough spread estimate; the longest bin (39 cells) holds only 2 of the 1000 mazes.

---

## Thinking steps (`thinking_steps.png`)

Packet figure F4, part 2 of 2 (other part: figures/commitment/commitment_maze0.png, commitment_maze248.png)

**Caption.** Two test mazes, the thresholded prediction after t = 1, 2, 4, 8, 12, 16, 24, 32 recurrence steps (one shared rollout per model, read out at each t). Maze rows: CF-Full (hid 16, 160 epochs, trained at fixed T = 8) above RecurrentCNN (hid 22, trained at fixed T = 8, 40 epochs); both seed 0. The left column repeats the ground truth; the maze selection rule: the test maze with the most wrong cells for each of the two models at its own T\* (seed 0, no repeats, ties by maze index) - i.e. one worst case per model under that shared rule, and test maze 256 = worst case of CF-Full (hid 16, 160 epochs, trained at fixed T=8), 38 wrong cells; test maze 113 = worst remaining case of RecurrentCNN (hid 22, 40 epochs, trained at fixed T=8), 6 wrong cells: its outright worst maze, test maze 256 (9 wrong cells), was already picked for CF-Full, 160 epochs, so the shared no-repeats rule moves to the next-worst maze; 3 mazes tie at that wrong-cell count, tie broken by maze index.
**How to read.** Same colour code as the path gallery: blue = predicted cell on the true path, vermillion = extra predicted cell, purple = missed true-path cell, black outline = true path, circle = start, star = goal. The number under each panel is the wrong-cell count at that t ("solved" = 0 wrong cells).

**Key takeaway.**

What the two rows show, from the probe rows of data.csv: on test maze 256, CF-Full (160 epochs) is solved at t = 4 and then loses the path (38 of its 39 true-path cells missed at t = 16-32), i.e. a failure of running longer on its worst maze, while RecurrentCNN (fixed T=8) degrades after its training depth (9 wrong cells at t = 8, 40 at t = 32); on test maze 113, CF-Full (160 epochs) is solved from t = 12 on, while RecurrentCNN (fixed T=8) degrades after its training depth (6 wrong cells at t = 8, 15 at t = 32). Both mazes are hand-selected worst cases under the selection rule above, not solve rates.

- Test maze 256, CF-Full, 160 epochs, seed 0: 6 wrong cells at t = 1; fully solved only at t = 4, then off the true path again (38 wrong cells at t = 32) (wrong cells at t = 1, 2, 4, 8, 12, 16, 24, 32: 6, 1, 0, 21, 39, 38, 38, 38).
- Test maze 256, RecurrentCNN, fixed T=8, seed 0: 9 wrong cells at t = 1; never solves it within 32 steps (best: 8 wrong cells at t = 4) (wrong cells at t = 1, 2, 4, 8, 12, 16, 24, 32: 9, 11, 8, 9, 33, 39, 39, 40).
- Test maze 113, CF-Full, 160 epochs, seed 0: 29 wrong cells at t = 1; first fully solved at t = 12 and stays solved through t = 32 (wrong cells at t = 1, 2, 4, 8, 12, 16, 24, 32: 29, 20, 13, 2, 0, 0, 0, 0).
- Test maze 113, RecurrentCNN, fixed T=8, seed 0: 29 wrong cells at t = 1; never solves it within 32 steps (best: 6 wrong cells at t = 8) (wrong cells at t = 1, 2, 4, 8, 12, 16, 24, 32: 29, 31, 24, 6, 6, 14, 15, 15).

**Data source.** `data.csv` in this folder: rows with model = cf-full-160ep / rcnn-fixedT8, seed = 0 and T in 1, 2, 4, 8, 12, 16, 24, 32 for the two mazes above (probe = 1 rows; all other rows have probe = 0, the T\*-level evaluation, and are excluded from every rate recomputed here; checkpoints in the results archive: `followups/E1_CF-Full_seed0.pt`, `RecurrentCNN_seed0.pt`).

**Caveats.** Both models were trained with rollout depth 8, so predictions at t > 8 are beyond the training rollout depth (extrapolation for both, in different ways: the RecurrentCNN never saw a longer loop, CF-Full never saw a deeper coagulation-fragmentation process); the two mazes are worst cases selected to be hard (the second under the shared no-repeats rule), not typical test mazes; single seed (0); wrong cells = extra + missed.

