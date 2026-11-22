# Family B ladder: free energy and per-cell size distributions

Figures built by `scripts/figures/ladder_energy.py` from the trained
checkpoint and metrics in `results/ladder_train/` (BD-WallAware = Family B
Becker-Döring commitment ladder: wall-aware, detailed-balance rates, K = 4
rungs, hid = 16, seed 0, trained at fixed T = 8, r_s = 2,
n_f = 2). Every number below is read back from `data.csv` in this folder.
Every number carries its training budget: **preliminary, 1 seed, reduced budget (46/160 epochs)** (46/160
epochs done of a cosine schedule planned for 160 epochs, stopped by a
30.0 min wall-clock budget, best epoch 3), trained on CPU
(CPU, reduced budget = not a reproduction of the 160-epoch protocol).

Numbers of the run for context (from `results/ladder_train/ladder_metrics.json`,
also mirrored in `data.csv`): 21777 raw parameters (not run (eval-only) effective),
test strict at T = 8 = 0.0 and at the validation-selected T* = 12 =
0.053, probe-7 strict at T* = 0.013, probe-9 strict at T* = 0.003.
Split: level-5 curriculum test split (1000 mazes); model input = walkable +
start + goal channels; readout = threshold 0.5 on per-cell logits.

---

## `ladder_energy_V_steps.png` - free energy over the rollout

**Caption.** Free energy V of the trained ladder, mean per maze over the
first 64 test mazes of the level-5 split, after every chemistry
micro-step of a 32-outer-step rollout (n_f = 2 micro-steps per outer
step; the rollout runs 4x past the training horizon
T_train = 8). Grey spans mark the fixed-rate phases: the rates
(g, mu) are recomputed only every r_s = 2 outer steps, refreshing
before outer steps 1, 1+r_s, 1+2*r_s, ...; between two refreshes the
chemistry relaxes under frozen rates (a fixed-rate phase) and V is a
Lyapunov function, so V must fall inside each fixed-rate phase while the
refreshes can re-set the landscape.

**How to read.** Blue line = mean per-maze V after each micro step; blue
band = sample sd over 64 mazes (n-1 denominator), n = 64 mazes (V is unbounded below; the sd band is not clipped (no valid range)); grey spans = fixed-rate phases
(rates frozen; alternate shading only marks the segmentation, every phase
is fixed-rate); grey dashed line = slow-phase rate refresh; vermillion dotted
line = trained horizon. V is dimensionless (a sum of c log c terms, c in
evidence units); more negative = chemically more settled.

**Key takeaway.** Within fixed-rate phases the summed (batch) V decreases
in 16/16 of 16 phases, and 1023 of 1024 per-maze
phase pairs end below their start (1023/1024 = 99.9%;
exactly one pair of the 64 mazes x 16 phases does not fall); the visible upward steps happen
exactly at the rate refreshes, where the slow phase re-writes the energy
profile - the energy is a per-phase Lyapunov function, not a global one
(two-timescale design, as in the family card).

**Data source.** `data.csv` (panel `V_trace`, panel `V_phase`); checkpoint
`results/ladder_train/BD-WallAware_seed0.ckpt.pt`; metrics
`results/ladder_train/ladder_metrics.json`; mazes via the seeded level-5
curriculum splits (`cf.data.curriculum`). Script recomputed V from the
checkpoint in float64 and cross-checked the summed trace against
`ladder_metrics.json` (agreement to 1e-3 relative, enforced over the
stored trace's finite prefix of 20 of 64 micro steps and the
matching outer prefix).

**Caveats.** Preliminary, 1 seed, reduced budget (46/160 epochs,
~30.0 min CPU); 64 of 1000 test mazes; V depends on the learned mass
scale, so absolute values are not comparable across training runs; the
discrete capped update contracts the continuous-time Lyapunov theorem, so
single micro-steps need not fall, phases do. The stored float32 V trace overflows to NaN from micro step 21 (the equilibrium weights Q leave the float32 range at these rates, b ~ 1e-31, while the chemistry state stays finite); the plotted V is recomputed in float64, which matches the stored trace to 1e-03 relative over its finite prefix (20 of 64 micro steps).

---

## `ladder_cell_size_distribution.png` - path cell vs dead-end cell

**Caption.** Left: test maze 0 of the level-5 split (true path
7 cells) with the path cells in green, every dead-end cell
(walkable, off-path, exactly one free neighbour) in orange, and the two
analysed cells outlined in black; the dots are the trained model's own
on-path prediction at T* = 12. Middle and right: the size
distribution of the two outlined cells over the rollout - the share of the
cell's clusters on each of the K = 4 rungs (rung 1 = "no idea" to rung
4 = "certainly on the path"), t = 0 being the initial state with all
evidence on rung 1. Bottom: the number-weighted mean rung (mean cluster
size) of the same two cells.

**How to read.** Maze map: green = true path (BFS shortest S-G path, S and
G marked), orange = dead-end cells, pale = other free cells, dark = walls.
Heatmaps: x = outer step (0 = initial state), y = ladder rung k, colour =
share of clusters on that rung, one shared colour scale clipped to [0, 1]
(shares sum to 1 over rungs; colour scale fixed to [0, 1] (clipped by construction, no values outside)); a cell "commits" when its mass moves to the top rungs.
Line panel: green = the path cell, orange = the dead-end cell; grey dashed
= T* = 12 (validation-selected horizon).

**Key takeaway.** By step 32 the path cell (r1,c6) sits at a mean
rung of 1.80 (from 1.00 at t = 0; 7.5% of its clusters on
the top rung 4) and the dead-end cell (r3,c5) at 1.71
(55.4% of its clusters on rung 1) - a separation of
0.09 rungs at step 32: the path cell overtakes the dead-end cell only in the second half of the rollout: up to step 16 the dead-end cell has the higher mean rung at 14 of the 17 recorded steps (the path cell is ahead only at steps 1, 14, a tie at t = 0), and the path cell stays above only from step 17 on - even then both stay near the bottom rungs, and the sharp path/dead-end commitment split the family card predicts (hysteresis) has not emerged yet at this reduced budget. At the model's
selected horizon T* = 12 the mean rung is 1.41 (path) vs
1.60 (dead end); the shown maze is NOT solved strictly at T* (False).

**Data source.** `data.csv` (panels `size_dist`, `mean_size`, `info`);
checkpoint `results/ladder_train/BD-WallAware_seed0.ckpt.pt`; metrics
`results/ladder_train/ladder_metrics.json`; maze 0 from the seeded level-5
test split.

**Caveats.** Preliminary, 1 seed, reduced budget (46/160 epochs,
~30.0 min CPU). Cell selection is deterministic, not hand-picked: the
median path cell of 5 candidates and the median dead-end cell of
8 candidates by final mean rung (median by final mean rung (upper middle of the value-sorted list)); still a single maze
and one cell per kind (n = 1, an instance view - the level-5 aggregate
solve rate is in `data.csv`, see test strict above). Shares are
number-weighted over the hid = 16 channels of the cell; the map's
prediction dots are the seed-0 model at its own T*.
