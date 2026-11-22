# Transport physics with no learning: results

## What was run

The maze was treated as a network of pipes: every pair of adjacent free cells is a pipe, and a wall is a missing pipe. At every time step, one unit of water is poured into the start cell and drained from the goal cell. A cell is declared part of the solution path when the total flow through the pipes touching it reaches at least half the poured amount (the poured amount is 1, so the cutoff is 0.5). This cutoff and this rule were written down in the specification before any run; the arm knows nothing else and has zero learned parameters: every pipe costs the same (length 1), and a pipe thickens in proportion to the flow it carries.

Why this should work on these mazes: the mazes in the curriculum are trees (no loops), so the water has exactly one route from the start to the goal. Water cannot be created or destroyed, so no steady flow can enter a dead-end corridor - there is nowhere for it to leave. Every pipe on the unique route therefore carries the full unit of flow, and every dead-end pipe carries none.

## The finding, stated plainly

With the exact pressure solve, the flow lands on the solution path at the very first outer step - before the network has adapted anything, i.e. already at t = 0. This is not something the model has to earn: on a tree (a maze with no loops), conservation alone fixes the flow. One unit of water enters at the start and leaves at the goal; every pipe on the single start-to-goal route must carry that whole unit, and no water can sit in a dead end because it would have nowhere to go. This is flow conservation on a loop-free network - the same fixed-point statement as Kirchhoff's circuit laws on a tree - and it holds for any pipe thicknesses, so no outer step can change it.

The measured numbers confirm exactly this: strict accuracy 1.000 and exact accuracy 1.000 on all three sets at every horizon measured, from T = 1 upward, and the largest share of flow found off the label path on any single maze is 0.000000083. The consequence for reading this page: the sweep over outer steps T carries no information here - the answer is complete at the first step - so the run is the ceiling and sanity check it was registered to be, not a test of learning.

What would be informative on these mazes is the information speed: replace the whole-maze exact solve with a fixed budget of k local relaxation sweeps (Jacobi sweeps - each pass updates every cell from its own neighbours only, so information travels locally) and measure how many sweeps the grid needs before accuracy reaches the ceiling, per maze size. That is the quantity that separates "the physics contains the answer" from "a local process can find it in bounded time"; on larger grids the sweep requirement should grow, which is where a maze-size story would live. No such run is recorded in results/transport_physics/results.json - the file contains the exact solve only - so every finite-sweep row in the table below is marked not run, meaning precisely that it was not measured.

## Headline table: accuracy vs relaxation budget

Test strict accuracy, per maze set, for the exact solve and for fixed budgets of Jacobi sweeps per outer step. Each cell shows strict accuracy at the training horizon T = 8 and at the validation-selected horizon T* as "at T = 8 / at T*".

| relaxation per outer step | level 5 test (1000 mazes) | level 7 probe (1000 mazes) | level 9 probe (1000 mazes) |
| --- | --- | --- | --- |
| exact solve (float64) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Jacobi, 1 sweep | not run | not run | not run |
| Jacobi, 4 sweeps | not run | not run | not run |
| Jacobi, 16 sweeps | not run | not run | not run |

Notes on the table:
- T* is the validation-selected horizon of the shared selection rule (the horizon with the highest validation strict accuracy; ties go to the smaller T). No validation sweep was run for this arm, and at ceiling it cannot matter: strict accuracy measured 1.000 at every horizon tested (T = 1, 2, 4, 8, 16, 32, 64) on all three sets, and the flow does not change between outer steps on these mazes (the conserved flow on a tree is unique and step-independent), so the test strict accuracy at T* is 1.000 whichever horizon the rule would select.
- Parameters: 0 learned, raw and effective both 0. The arm is a fixed rule; nothing is trained (0 epochs, 0 seeds), and the single run is deterministic, so these are exact values with no seed spread - not estimates, and no seed-count qualifier applies.
- Hardware: CPU only; results.json records device "cpu" for every set. The solve is float64.
- Runtime: one whole-maze exact solve per outer step; totals 11.2 s (level 5), 40.6 s (level 7), 176.9 s (level 9) for all horizons together per 1000-maze set.
- The "not run" cells are absent measurements, not measured zeros: no Jacobi or other finite-sweep relaxation run exists in results.json for these sets.

## Where the numbers come from

The same seeded splits as every other run in this repository were used: the level-5 test set (1000 mazes) and the probe sets at levels 7 and 9 (1000 mazes each, never used for training by any arm). Because nothing is learned, there is a single deterministic run - no seeds or training phases are involved. 'Strict' accuracy counts a maze as solved when the predicted cells form one wall-free path from start to goal of exactly the label's length; 'exact' accuracy counts it solved only when the predicted cells match the label path cell-for-cell. On these tree mazes the two are the same thing. The 'dead-end flow fraction' is, per maze, the share of all flow that runs through pipes not on the label path; the physics says it should be zero.

T below is the number of outer steps run (each step re-solves the water pressure in the whole maze, then moves pipe material towards the pipes that carry flow). All horizons share one rollout per batch, so longer horizons only add the extra steps. 'Held' counts the mazes whose pressure solve had already become numerically impossible at that horizon (see the next section) and whose flow was therefore carried by its last solved value.

## level5_test (level 5, 1000 mazes, 11.2 s)

| T | strict | exact | dead-end flow fraction | held | wall seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0 | 0 | 0.36 |
| 2 | 1 | 1 | 0 | 0 | 0.34 |
| 4 | 1 | 1 | 0 | 0 | 0.64 |
| 8 | 1 | 1 | 0 | 93 | 2.33 |
| 16 | 1 | 1 | 0 | 277 | 4.06 |
| 32 | 1 | 1 | 0 | 671 | 2.04 |
| 64 | 1 | 1 | 0 | 671 | 1.38 |

Pressure solve held for 671 of 1000 mazes (first hold at outer step 6, last at step 22); 329 mazes ran the full rollout at every horizon.

## probe7 (level 7, 1000 mazes, 40.6 s)

| T | strict | exact | dead-end flow fraction | held | wall seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0 | 0 | 1.29 |
| 2 | 1 | 1 | 0 | 0 | 1.26 |
| 4 | 1 | 1 | 0 | 0 | 2.48 |
| 8 | 1 | 1 | 0 | 34 | 7.14 |
| 16 | 1 | 1 | 0 | 244 | 13.98 |
| 32 | 1 | 1 | 0 | 588 | 6.86 |
| 64 | 1 | 1 | 0 | 588 | 7.62 |

Pressure solve held for 588 of 1000 mazes (first hold at outer step 6, last at step 29); 412 mazes ran the full rollout at every horizon.

## probe9 (level 9, 1000 mazes, 176.9 s)

| T | strict | exact | dead-end flow fraction | held | wall seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0 | 0 | 4.27 |
| 2 | 1 | 1 | 0 | 0 | 3.83 |
| 4 | 1 | 1 | 0 | 0 | 7.47 |
| 8 | 1 | 1 | 0 | 21 | 20.96 |
| 16 | 1 | 1 | 0 | 154 | 42.46 |
| 32 | 1 | 1 | 0 | 369 | 36.69 |
| 64 | 1 | 1 | 0 | 369 | 61.25 |

Pressure solve held for 369 of 1000 mazes (first hold at outer step 6, last at step 19); 631 mazes ran the full rollout at every horizon.

## Reading of the numbers

Accuracy is at ceiling on every set and at every horizon T from the very first step: the flow-conservation argument places the answer immediately once the water pressures are solved exactly, and further steps only re-arrange pipe material without changing the flow on a tree. The largest dead-end flow fraction seen on any single maze at any horizon is 0.000000083 - flow that stays off the label path is numerically zero to the displayed precision.

Why some mazes were 'held': the fixed adaptation keeps draining pipes that carry no flow, by a factor of about a thousand per outer step. Eventually the pipe map of a maze became too degenerate for the exact pressure solver - its conservation accuracy crossed the pre-set tolerance of one part in a hundred thousand, and shortly after the solve would turn outright singular - at a maze-dependent step (first holds appear around step 5-6). When that happened, that maze's flow was held at its last fully accurate solved value. On these loop-free mazes that is not an approximation: with the water conserved, the flow between the fixed start and goal is unique on a tree and does not depend on the pipe thicknesses, so the held value IS the value every later step would have produced. The accuracy columns are therefore unchanged past a hold point, exactly as the physics predicts. Wall time is dominated by the per-step pressure solve and grows with maze size; totals: level5_test 11.2 s, probe7 40.6 s, probe9 176.9 s
