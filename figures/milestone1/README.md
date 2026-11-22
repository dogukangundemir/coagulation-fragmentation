# Figures: milestone1

Milestone-1 report figures from the one-shot Colab notebook
(`notebooks/CF_Looped_Reasoning_Milestone_1.ipynb`); the copies here are the
Drive outputs of the notebook's untrained/smoke-run cells.

- `dataset_preview.png` — random 9 x 9 mazes from the generator
  (`milestone1/maze_cf.py`) with the BFS shortest path overlaid in red.
- `invariant_diagnostics.png` — the physics holds without any training:
  total particle mass M stays exactly conserved (and c1, c2 non-negative)
  over an unrolled T = 64 rollout.
- `trajectory_heatmaps_untrained.png` — the raw chemistry in motion
  (snapshots at t in {1, 2, 4, 8, 16}): monomers c1 merge into dimers c2
  near the path, no training anywhere.
- `smoke_training_loss.png` — loss curve of the short 800-step
  smoke-training run: the pipeline learns.
- `smoke_predictions.png` — after that short run, predicted per-cell path
  probability next to the correct shortest path.
- `thinking_longer_preview.png` — how the metrics move when the frozen
  model is simply run longer at test time ("Thinking Longer").
