# Family A vs Family D: trained wall-aware model vs physics-only transport

Plain language. Two maze solvers, same mazes. Family A is a trained cell model: it spreads a conserved amount of substance over the free cells, and at every step pairs of particles merge into clusters (2 A1 -> A2) and clusters split back (A2 -> 2 A1). Its answer is read off the substance maps after T steps. Family D is the transport model with nothing learned: the maze is a pipe network, water is poured into the start cell and drained from the goal cell at every step, pipes that carry flow thicken and empty pipes thin. A cell is declared on the solution path when the flow through its pipes reaches at least half the poured unit. Its pressure solve is exact (a float64 linear solve) or approximate with k local Jacobi-type relaxation sweeps per outer step (k in 1, 4, 16).

All figures are built by `scripts/figures/a_vs_d.py` from `results/x1_x6/results.json` (the X1/X6 evaluation, run with reduced sets, see budgets) and, for the maze-256 failure case, from the main repository's per-maze evaluation CSV (read-only). Re-run from the clone root with `python -m scripts.figures.a_vs_d` (CPU only). Every number quoted below is in `data.csv` / `data_maze256.csv` in this folder.

## Symbols and scoring

- **T** = test-time steps run; **t** = recurrence step in the per-maze figure. **k** = local relaxation sweeps per outer step in Family D's solver.
- **strict accuracy** = the predicted cells form one wall-free start-to-goal path of exactly the label's length; **exact** = the predicted cells equal the label path cell-for-cell (on these tree mazes the two coincide).
- **committed mass** (Family A) = the cluster half 2*c2 of the conserved total M = c1 + 2*c2; *†c2* is the cluster-channel mass. Shares are of all committed mass sitting on label-path cells vs on dead-end cells (free cells off the path); levels are the mean per-cell commit fraction 2*c2/M of each group.
- **flow share** (Family D) = share of all cell flow |Q| (sum of a cell's incident pipe flows) on label-path vs dead-end cells; levels are the group means.
- **IoU** = intersection-over-union of predicted and true path cells; **wrong cells** = extra + missed cells.

- **T\*** = the horizon selected on the validation split by the training program (argmax validation strict, ties to the smaller T): T\* = 12 for the Family A checkpoints (source: main repository results archive: wallaware160/results.json, read-only); Family D has no validation horizon (deterministic, no training).

## Budgets, parameters and hardware

| family | model | parameters (raw / effective) | training | seeds | evaluation |
|---|---|---|---|---|---|
| A | cf wall-aware CFNet, hid 16 | 10177 raw / 9921 effective | 160 epochs, fixed T = 8, T\* = 12 | 0, 1, 2 | first 300 level-5 test / first 200 probe-7 / first 200 probe-9 mazes of the fixed seed-0 splits (*reduced evaluation sets, not a reproduction*) |
| D | transport CFNet, physics-only | 0 / 0 | none (0 learned parameters, exact solve or k = 1/4/16 sweeps per step) | - (deterministic, one run) | same reduced sets as A |

Recomputation (this script, CPU only, 9.2 s total): parameter counts from the seed-0 checkpoint (effective = nonzero-gradient entries after one forward+backward at horizon T = 8); spot re-evaluation of Family A seed 0 on the level-5 test set at T = 8 and T\* = 12 and of Family D k = 1 at T = 8 matches the stored results to within 0.001 (the script exits non-zero otherwise). The stored X1/X6 run itself used cuda (per `results/x1_x6/summary.md`); evaluation wall time recorded there: Family A 1.9 s, Family D 76 s over the reduced sets (per-unit `seconds` are in the results JSON). Training wall time of the checkpoints is not recorded in this clone (not run here - not estimated).

### Strict accuracy at T = 8 (training depth) and at T\*

Family A: mean +- sample sd over 3 seeds on the reduced sets. Family D: deterministic single run (no T\*; the exact solve gives 1.000 at every T, so its row states the flat value).

| unit | level-5 test (n = 300) | probe 7 (n = 200) | probe 9 (n = 200) |
|---|---|---|---|
| A @ T = 8 | 0.9100 +- 0.0033 | 0.4700 +- 0.0180 | 0.1467 +- 0.0144 |
| A @ T\* = 12 | 0.9944 +- 0.0019 | 0.8767 +- 0.0153 | 0.5883 +- 0.0176 |
| D exact @ T = 8 | 1.0000 | 1.0000 | 1.0000 |
| D k=1 @ T = 8 | 0.1767 (best 0.1767 at T = 4) | 0.0750 (best 0.0750 at T = 4) | 0.0550 (best 0.0550 at T = 8) |
| D k=4 @ T = 8 | 0.3567 (best 0.3933 at T = 4) | 0.1950 (best 0.2350 at T = 4) | 0.1900 (best 0.2050 at T = 4) |
| D k=16 @ T = 8 | 0.6767 (best 0.7700 at T = 4) | 0.4200 (best 0.5750 at T = 4) | 0.3350 (best 0.3550 at T = 4) |

Family D exact: 1.0000 at every T on all three sets (constant after the held-flow rule; see the caveat under the accuracy figure). Best-T values for the sweep modes are post-hoc maxima over the eight evaluated step counts, not validation-selected.

---

## `committed_mass_vs_flow.png` - where each family puts its mass / flow

**Caption.** Share of Family A's committed mass (2*c2) and of Family D's cell flow |Q| sitting on label-path cells vs dead-end cells over test-time steps T, on the level-5 test set (n = 300). Family A: wall-aware, 160 epochs, mean over 3 seeds. Family D: k = 1 relaxation sweeps per step; the exact solve is the dashed reference (all flow on the path, dead ends at 0).
**How to read.** Blue = share on label-path cells, orange = share on dead-end cells (free cells off the path; the two shares sum to 1). Left band = mean +- sample sd over 3 seeds, clipped to [0, 1] because a share cannot leave that range.

**Key takeaway.** On the level-5 test set at T\* = 12, Family A solves 99.4% of the mazes (strict, 3-seed mean) while only 0.29 of its committed mass sits on the path (0.71 on dead ends; sample sd over 3 seeds) - the learned model does not solve by concentrating committed mass on the path. Family D's flow does concentrate: with k = 1 sweeps per step the path share rises from 0.71 at T = 1 to 0.91 at T = 32, and the exact solve has share 1.000 on the path at every T. (160 epochs x 3 seeds, reduced sets; the X1/X6 evaluation of all Family A units took 1.9 s on cuda.)
**Data source.** `data.csv` in this folder (family = A rows: metric = committed mass; family = D rows: metric = cell flow; evaluation numbers produced by `scripts/eval_x1_x6.py` from the checkpoints in the main repository, read-only).

**Caveats.** Single grid (level 5) and single sweep budget shown per panel; path shares drift only mildly in T for Family A, so the mechanism contrast is a level contrast, not a growth story; dead-end cells vastly outnumber path cells, so an even mass spread would put a much lower share on the path - the relevant comparison is between the two shares and against the exact-solve reference, not against 0.5. Sample sd over only 3 seeds is a rough spread estimate.

---

## `accuracy_vs_steps.png` - accuracy vs test-time steps, per grid

**Caption.** Strict accuracy vs test-time steps T for Family A (three seed lines; wall-aware, 160 epochs each, trained at fixed T = 8) and Family D (exact solve, and k = 1/4/16 relaxation sweeps per step), one panel per grid: level-5 test (n = 300), probe 7 (n = 200), probe 9 (n = 200). Grey dashed line = training depth of Family A (T = 8).
**How to read.** Each line is one unit: Family A per seed (blue/green shades, markers o/s/^; 3 seeds, each 160 epochs), Family D per solver mode (deterministic single run). y = strict accuracy in [0, 1]. Steps beyond T = 8 are beyond Family A's training depth (thinking longer).

**Key takeaway.** Family A turns test-time steps into accuracy on every grid: 3-seed mean strict 0.9944 at its T\* = 12 on level 5, peaking at 0.9850 (T = 16) on probe 7 and 0.9283 (T = 24) on probe 9 (sample sd over 3 seeds; per-seed lines in the figure - the seeds diverge beyond T = 16 on probe 9: seed-1 drops to 0.780 at T = 32 while seeds 0 and 2 hold 0.985 and 0.980, so the late-T seed spread on the probe grids makes any 'running longer always helps' claim inconclusive there). Family D's relaxation modes plateau far below their exact solve on every grid (best over the swept T: 0.770 / 0.575 / 0.355 for k = 16 on levels 5/7/9 vs 1.000 exact), so its steps do not buy the full solution. (160 epochs x 3 seeds for A; D has 0 learned parameters and one deterministic run; reduced sets.)
**Data source.** `data.csv` (family A rows `strict` at T = 1..32; family D rows `strict` per unit).

**Caveats.** Reduced evaluation sets (first 300/200/200 mazes): not a reproduction of the full 1000-maze runs. The probe grids were not the training grid, so probe performance is size extrapolation for both families; differences within the seed spread are inconclusive (e.g. Family A seeds at T = 8 on level 5: 0.9067 vs 0.9100 vs 0.9133). The exact-solve trace is flat because mazes whose exact solve degenerates are held at their last valid solved flow (174/300, 106/200 and 54/200 mazes on the three sets; on a maze tree the conserved flow is unique, so the held value is the converged one).

---

## `d_accuracy_vs_sweeps.png` - Family D accuracy vs the solver budget

**Caption.** Best strict accuracy over the eight evaluated step counts T (1, 2, 4, 8, 12, 16, 24, 32) against the local relaxation sweeps per step k (1, 4, 16) and the exact float64 solve, for the three grids (n = 300 / 200 / 200 mazes). Dashed tail = towards the exact solve.
**How to read.** Each line = one grid (blue level 5, green probe 7, red probe 9); x = solver budget k per outer step; y = post-hoc best strict accuracy over the swept T; the second line of each x label gives, in order for levels 5/7/9, the step count T at which that best value is attained; the exact-solve slot has 1.000 at every T (all three grid points coincide there).

**Key takeaway.** More sweeps per step buy accuracy on every grid, with diminishing returns and a large remaining gap: level 5: 0.1767 (k=1, T=4), 0.3933 (k=4, T=4), 0.7700 (k=16, T=4), 1.000 (exact); level 7: 0.0750 (k=1, T=4), 0.2350 (k=4, T=4), 0.5750 (k=16, T=4), 1.000 (exact); level 9: 0.0550 (k=1, T=8), 0.2050 (k=4, T=4), 0.3550 (k=16, T=4), 1.000 (exact). Even the largest evaluated budget (k = 16, at its best T) stays 0.230 / 0.425 / 0.645 below the exact solve on levels 5/7/9, i.e. the gap is concentration-of-flow error the outer adaptation cannot repair at these budgets, not a step-horizon problem. (Family D: 0 learned parameters, deterministic single run, reduced sets; total X1/X6 evaluation time for the 12 Family D units was 76 s on cuda.)
**Data source.** `data.csv` (family = D rows). The best-T values are maxima over the eight evaluated step counts of the `strict` column.

**Caveats.** Post-hoc best over T (not validation-selected); the sweep-mode accuracy is non-monotone in T (it peaks at small T and decays as the flow collapses onto the path), so a fixed-T comparison would look worse; only three budgets (1, 4, 16) were evaluated, so the exact shape between them is not measured.

---

## `maze256_failure.png` - the honest failure on one maze

**Caption.** Level-5 test maze 256 (true path 39 cells, dataset row 2341) - the worst case of CF-Full (hid 16, 160 epochs, seed 0, trained at fixed T = 8) on its test split at its seed-0 validation-selected depth T\* = 16, selected by the main repository's rule 'the test maze with the most wrong cells for the model at its own T\*' (38 wrong cells there). Thresholded predictions read out after t = 1, 2, 4, 8, 12, 16, 24, 32 recurrence steps; RecurrentCNN (fixed T = 8, 40 epochs, seed 0) is shown as the trained-baseline context. Grey dashed = training depth of both models; grey dotted = T\* = 16 of CF-Full seed 0.
**How to read.** Left: wrong cells (extra + missed) vs step t, 0 = strict solved; right: IoU of predicted vs true path vs t. Single maze, single seed (0): this is an instance-level failure anatomy, not a solve rate.

**Key takeaway.** CF-Full (160 epochs, seed 0) is fully solved at t = 4 (0 wrong cells, IoU 1.000), then loses the path: 21 wrong cells at t = 8, 39 of its 39 true-path cells missed from t = 12 on (IoU 0.0256 at t = 32) - running longer destroys the solution on this maze, and at its own T\* = 16 the verdict is 'not solved' (38 wrong cells). The RecurrentCNN baseline degrades after its training depth too (best 8 wrong cells at t = 4, 40 at t = 32) but never produces the step-4 solved state. Per-step values (wrong cells at t = 1, 2, 4, 8, 12, 16, 24, 32): CF-Full 6, 1, 0, 21, 39, 38, 38, 38; RecurrentCNN 9, 11, 8, 9, 33, 39, 39, 40.
**Data source.** `data_maze256.csv` in this folder - the rows this script read (models cf-full-160ep and rcnn-fixedT8, seed 0, maze 256, probe = 1) from the MAIN repository's `figures/maze_paths/data.csv` (read-only), produced by its `scripts/figures/maze_paths.py` evaluation (160 epochs x 3 seeds for CF-Full; the maze-256 verdict row at T\* matches the probe row at t = 16: 38 wrong cells, checked by this script).

**Caveats.** Hand-selected worst case under the stated rule, not a typical maze; single seed (0); the stored CSV has no cell maps, so this figure re-plots the recorded per-step error counts and IoU (the original panel figure with the maze drawings lives in the main repository's maze_paths folder); steps t > 8 are beyond the training rollout depth of both models. While Family A's *aggregate* accuracy keeps improving with T (figure 2), this maze shows the improvement is not uniform per instance.

