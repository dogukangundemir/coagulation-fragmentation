# Coagulation–Fragmentation for Looped Reasoning

Research code for a looped neural computation scheme based on **Smoluchowski
coagulation–fragmentation** kinetics — $A_i + A_j \rightleftharpoons A_{i+j}$ —
studied as an alternative to oscillator synchronization in looped physical
networks (AKOrN: Kuramoto oscillators; WONN: Winfree oscillators).

Core hypothesis: **coagulation** (small local hypotheses merging into clusters)
plus **fragmentation** (wrong clusters splitting back into revisable units)
yields both high accuracy at small parameter budgets and *"Thinking Longer"* —
accuracy keeps improving when the network is unrolled for more steps at test
time than it was trained for, without touching the weights.

## Milestone 1 — maze infrastructure (this repository)

| File | Contents |
|---|---|
| `maze_cf.py` | Braided perfect-maze generator (7x7 / 9x9 / 15x15), BFS shortest-path labels, `MazeDataset`, the Stage-A `CoagulationFragmentationNet` (reversible $2A_1 \rightleftharpoons A_2$ with provable non-negativity and exact mass conservation), ablation switches (`fragmentation=False`, `bound_flux=False`), and a parameter-matched `RecurrentCNN` baseline. |
| `test_maze_cf.py` | `pytest` acceptance checks: BFS label validity; $c_1, c_2 \ge 0$; mass conservation $|M^{(t)}-M^{(0)}| < 10^{-5}$; gradient flow to the input and every parameter; ablation sanity. |
| `CF_Looped_Reasoning_Milestone_1.ipynb` | One-shot Colab notebook (mounts Google Drive): runs the tests, untrained-model invariant diagnostics to $T=64$, trajectory heatmaps of the raw chemistry, a smoke-training run with example predictions, and a frozen-weights "Thinking Longer" preview. |

### The Stage-A update rule

$$m_t = \tfrac{1}{2} c_1^{(t)} \sigma(\text{raw\_merge}_t), \qquad
s_t = c_2^{(t)} \sigma(\text{raw\_split}_t)$$
$$c_1^{(t+1)} = c_1^{(t)} - 2 m_t + 2 s_t, \qquad
c_2^{(t+1)} = c_2^{(t)} + m_t - s_t$$

The bounds $2m_t \le c_1^{(t)}$, $s_t \le c_2^{(t)}$ guarantee analytically:
$c_1, c_2 \ge 0$ and exact conservation of $M = c_1 + 2 c_2$ (distances matter —
the reversible *stoichiometry* conserves $M$ for any fluxes; the bound is what
buys *positivity*).

## Running

Tests (PyTorch + NumPy required, no GPU needed):

```bash
python -m pytest -q test_maze_cf.py
```

Notebook: open `CF_Looped_Reasoning_Milestone_1.ipynb` in
[Colab](https://colab.research.google.com) (Upload notebook; GPU runtime
recommended), then `Run all` (~5-10 min). Figures and checkpoints are written
to Google Drive under `CF_Looped_Reasoning/`.

## Roadmap

- **Milestone 2:** full 4-model benchmark (CF-Full / CF-NoFrag / CF-NoBound /
  RecurrentCNN, matched budgets), test-time horizons up to $T = 32$.
- **Milestone 3:** "Problem + Design" presentation (Nov 11).
- **Milestone 4:** general Smoluchowski `CFLayer` — Maze-hard 30x30 (~0.4M
  params), Sudoku-OOD (~1.6M params), energy voting.
- **Milestone 5:** CF on latent chains (Coconut-style continuous-latent
  reasoning, ProsQA / ProntoQA).
