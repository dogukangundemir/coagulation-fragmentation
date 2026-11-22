# Family G — best-of-K random-start voting

Plain language: run the solver several times from different random starting states and keep the run that ends in the calmest internal state. If calm runs are the correct runs, that is a free accuracy boost at test time.

## Budget of every number in this folder

- Family A (trained cell model, A-WallAware): 160 epochs, 1 training seed (seed 0) — fewer than 3 seeds, so **preliminary**; evaluation on the first 300 level-5 test mazes (reduced slice, **not a reproduction** of the full 1,000-maze numbers), CPU, 2.3 s of evaluation.
- Family D: physics only, no training (0 learned parameters), so no seeds; same 300 mazes, CPU, 84.2 s of evaluation.
- Parameters: A = 10,177 raw / 9,921 effective (nonzero-gradient entries after one forward+backward at T = 8, recomputed from the seed-0 checkpoint); D = 0 learned. Hardware: CPU only.

## accuracy_vs_K.png

**Caption.** Strict accuracy of the lowest-energy pick over K = 1, 8, 32 random starts, at the two readout horizons of the trained net (T = 8, its training depth, and T = 12, the horizon its training program selected on validation) and the two horizons of the physics arm (T = 8, 16), against each family's deterministic start.

**How to read.** Reading guide: solved = the predicted cells are exactly the BFS shortest path (strict; equal to exact on these tree mazes). K = number of random starting states tried per maze; the run with the lowest energy is kept. T = steps the network/physics is run before the answer is read off. Energy (Family A) = summed residual coagulation-fragmentation flux |m − s| at the readout step; energy (Family D) = electrical dissipation of the final flow (a dead-end leak fraction gives identical votes). Error bars and bands are Wilson 95% binomial intervals over the 300 mazes, clipped to [0, 1]; they show maze-sampling noise only — with one training seed they say nothing about seed-to-seed variation.

**Key takeaway (all values from data.csv).** The physics energy is perfect from any start: 100.0 % at every K and both horizons (all 300/300 mazes solved; 0/300 mazes whose starts disagree). Energy voting does lift the trained net — from 61.7 % at K = 1 to 75.3 % at K = 32 (T = 8) — but it recovers only part of the distance to its ceiling of 75.7 % (mazes with at least one correct start) and stays 15.3 points below the 90.7 % of the very same net run once from its deterministic start (T = 12: 75.7 % vs 99.3 %).

**Data source.** `../../results/voting/results.json` (produced by `scripts/eval_voting.py`, evaluation only, CPU) and `../../results/voting/summary.md`; every number is in [`data.csv`](data.csv).

## energy_histograms.png

**Caption.** The energies of all 32 random starts on the 300 mazes (n = 9,600 per panel), split into starts that solved the maze and starts that failed; Family A at T = 8 and T = 12, Family D at T = 8 (dissipation, log scale).

**How to read.** Reading guide: solved = the predicted cells are exactly the BFS shortest path (strict; equal to exact on these tree mazes). K = number of random starting states tried per maze; the run with the lowest energy is kept. T = steps the network/physics is run before the answer is read off. Energy (Family A) = summed residual coagulation-fragmentation flux |m − s| at the readout step; energy (Family D) = electrical dissipation of the final flow (a dead-end leak fraction gives identical votes). The histograms are pooled across mazes; the within-maze separation AUC in each panel title compares correct and wrong starts of the *same* maze only (0.5 = the energy carries no correctness information, 1.0 = perfect separation). Legends give the count and median of each group.

**Key takeaway — does the energy separate correct from wrong?** Stated plainly: **yes, largely, for the trained net — and trivially for the physics arm.** The trained net's energy separates the two almost completely: AUC 0.976 at T = 8 and 0.998 at T = 12, with correct starts calmer on average (mean 53.1 vs 80.3 at T = 8; 50.8 vs 76.4 at T = 12). It is not perfect: at K = 32 the vote solves 226/300 mazes while 227/300 mazes contain at least one correct start, so in a few mazes a wrong start is calmer than every correct one. For the physics arm there is nothing to separate: every one of the 9,600 random starts solves its maze (identical under the leak vote and at T = 16), so its energy trivially 'ranks' the correct run first — the vote cannot fail, but it also cannot be credited as an accuracy win.

**Data source.** `../../results/voting/results.json`; every number is in [`data.csv`](data.csv).

## Caveats

- One training seed (seed 0): **preliminary**. The seed spread of the full curriculum runs is several points at this accuracy level, so differences of a few points here (for example T = 8 vs T = 12) are **inconclusive**.
- Reduced 300-maze slice of the 1,000-maze test split: **not a reproduction** of the full-split numbers.
- The Family A random starts are drawn from the model's own stochastic initial-state rule; the deterministic baseline uses the standard fixed start (c2 = 0) — the same model and weights either way.
- Family D's starts differ only in the initial pipe conductances; its exact solve reaches the same steady flow from every start on a tree maze, which is why all its starts agree (0/300 disagreeing mazes). Its 1.000 is the physics, not voting skill.
- The dissipation energies of Family D span many orders of magnitude, so that panel uses a log axis; the histograms are density-normalized (share of starts per bin).
- Seed-0 checkpoint parameter counts recomputed by `scripts/figures/voting.py` (this figure folder's own verification): 10,177 raw / 9,921 effective. The stored results.json states raw 10,581, which does NOT match this recomputation; the counted value is the one used here (the stored text is a hardcoded note, not a counted value).

