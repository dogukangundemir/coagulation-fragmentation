"""
maze_cf.py — Coagulation-Fragmentation (CF) prototype for looped reasoning on mazes.

Stage-A reversible reaction:            2 A_1  <=>  A_2
Per-cell, per-channel state (non-negative by construction):

    c1, c2 >= 0                          size-1 'monomers' and size-2 'clusters'
    m_t = 0.5 * c1 * sigmoid(raw_merge_t) merge flux  (bounded by available mass)
    s_t = c2 * sigmoid(raw_split_t)      split flux  (bounded by c2)
    c1 <- c1 - 2 m_t + 2 s_t
    c2 <- c2 +   m_t -   s_t
    =>  M = c1 + 2 c2  is exactly conserved at every step.

Ablation switches:
    fragmentation=False  ->  s_t = 0                    (coagulation only)
    bound_flux=False      ->  m_t = c1 * sigmoid(...)    (unbounded flux: c1 can go negative)

Note: the reversible stoichiometry conserves M = c1 + 2 c2 regardless of the
flux magnitudes; the 1/2 flux bound is what additionally guarantees
non-negativity of both concentrations (positivity / long-rollout stability).

Data: odd-size braided perfect mazes; a cell is positive iff it lies on at
least one shortest start->goal path (computed with two BFS distance maps).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

Neighbors4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ---------------------------------------------------------------------------
# 1. Maze generation and BFS labelling
# ---------------------------------------------------------------------------
def generate_maze(rng: np.random.Generator, size: int, braid: float = 0.08) -> np.ndarray:
    """Random *braided perfect maze*: recursive-backtracker spanning tree, then a
    fraction `braid` of interior walls is removed to create loops.

    Returns a (size, size) int8 grid; 1 = wall, 0 = free/walkable.
    """
    if size % 2 == 0:
        raise ValueError(f"size must be odd, got {size}")
    if size < 5:
        raise ValueError("size must be >= 5")

    n = (size - 1) // 2                      # cells per axis (cell k sits at 2k+1)
    grid = np.ones((size, size), dtype=np.int8)
    visited = np.zeros((n, n), dtype=bool)
    grid[1, 1] = 0
    visited[0, 0] = True
    stack: List[Tuple[int, int]] = [(0, 0)]

    while stack:
        i, j = stack[-1]
        candidates = [
            (i + di, j + dj)
            for di, dj in Neighbors4
            if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]
        ]
        if candidates:
            ni, nj = candidates[int(rng.integers(len(candidates)))]
            grid[2 * ni + 1, 2 * nj + 1] = 0          # carve the new cell
            grid[i + ni + 1, j + nj + 1] = 0          # carve the wall between
            visited[ni, nj] = True
            stack.append((ni, nj))
        else:
            stack.pop()

    # Braiding: open some interior wall *segments* that separate two free cells
    # -> loops / multiple shortest paths. Only opening such segments keeps every
    # free cell connected (no isolated cells can ever be created).
    pick = rng.random(grid.shape) < braid
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] != 1 or not pick[r, c]:
                continue
            horizontal = grid[r, c - 1] == 0 and grid[r, c + 1] == 0
            vertical = grid[r - 1, c] == 0 and grid[r + 1, c] == 0
            if horizontal or vertical:
                grid[r, c] = 0
    return grid


def bfs_distances(grid: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """4-neighbour BFS distances from `start` over free cells; -1 = unreachable."""
    size = grid.shape[0]
    dist = np.full((size, size), -1, dtype=np.int32)
    if grid[start] != 0:
        return dist
    dist[start] = 0
    queue: deque = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in Neighbors4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] == 0 and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def shortest_path_mask(
    grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Boolean mask: True iff the cell is free and lies on at least one shortest
    S->G path, i.e. d_S[cell] + d_G[cell] == d_S[G]."""
    ds = bfs_distances(grid, start)
    dg = bfs_distances(grid, goal)
    if ds[goal] < 0:
        return None
    return (grid == 0) & (ds >= 0) & (ds + dg == ds[goal])


def choose_endpoints(
    rng: np.random.Generator, grid: np.ndarray, min_path_len: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Random start/goal pair whose BFS distance is at least `min_path_len`."""
    free = np.argwhere(grid == 0)
    for _ in range(1000):
        s = free[int(rng.integers(len(free)))]
        g = free[int(rng.integers(len(free)))]
        if np.array_equal(s, g):
            continue
        d = bfs_distances(grid, (int(s[0]), int(s[1])))
        if d[g[0], g[1]] >= min_path_len:
            return (int(s[0]), int(s[1])), (int(g[0]), int(g[1]))
    # Fallback: farthest free cell from a random anchor (maze is always connected).
    s = free[int(rng.integers(len(free)))]
    d = bfs_distances(grid, (int(s[0]), int(s[1])))
    g = np.unravel_index(int(np.argmax(d)), d.shape)
    return (int(s[0]), int(s[1])), (int(g[0]), int(g[1]))


def make_sample(
    rng: np.random.Generator, size: int, braid: float = 0.08, min_path_len: int = 2
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], np.ndarray]:
    grid = generate_maze(rng, size, braid)
    start, goal = choose_endpoints(rng, grid, min_path_len)
    mask = shortest_path_mask(grid, start, goal)
    assert mask is not None, "perfect maze must be connected"
    return grid, start, goal, mask


class MazeDataset(Dataset):
    """Pre-generated maze samples.

    x : (3, size, size) float32 — channels = [walkable, start one-hot, goal one-hot]
    y : (size, size) float32     — 1.0 on every cell of at least one shortest path
    """

    def __init__(self, n_samples: int, size: int, seed: int, braid: float = 0.08,
                 min_path_len: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        if min_path_len is None:
            min_path_len = max(2 * (size - 1) // 3, 2)
        xs, ys = [], []
        for _ in range(n_samples):
            grid, (sr, sc), (gr, gc), mask = make_sample(rng, size, braid, min_path_len)
            walkable = (grid == 0).astype(np.float32)
            start_ch = np.zeros((size, size), dtype=np.float32); start_ch[sr, sc] = 1.0
            goal_ch = np.zeros((size, size), dtype=np.float32);  goal_ch[gr, gc] = 1.0
            xs.append(np.stack([walkable, start_ch, goal_ch]))
            ys.append(mask.astype(np.float32))
        self.x = torch.from_numpy(np.stack(xs))          # (N, 3, H, W)
        self.y = torch.from_numpy(np.stack(ys))          # (N, H, W)
        self.pos_weight = float((1.0 - self.y.mean().item()) / self.y.mean().item())

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def path_exists(pred: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """True iff `goal` is reachable from `start` walking only over predicted cells."""
    if not (pred[start] and pred[goal]):
        return False
    size = pred.shape[0]
    seen = np.zeros_like(pred, dtype=bool)
    seen[start] = True
    queue: deque = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in Neighbors4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and pred[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return False


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
class CoagulationFragmentationNet(nn.Module):
    """Stage-A CF network for the reversible reaction 2 A_1 <=> A_2.

    A weight-shared rate network sees, at every step t = 1..T, the original
    input x (walls / S / G), the current state (c1, c2) and their 3x3
    neighbourhood, and emits raw_merge / raw_split maps. Fluxes are bounded by
    the locally available mass, which analytically guarantees

        c1[t] >= 0,   c2[t] >= 0,   M = c1 + 2 c2 = const   for all t.

    Ablations: fragmentation=False (s_t = 0), bound_flux=False (the 1/2
    flux bound removed -> merge flux is unbounded and c1 can go negative,
    a stability ablation: the stoichiometry still conserves M).
    """

    def __init__(self, in_ch: int = 3, hid: int = 16, *, fragmentation: bool = True,
                 bound_flux: bool = True) -> None:
        super().__init__()
        self.fragmentation = fragmentation
        self.bound_flux = bound_flux
        self.encoder = nn.Conv2d(in_ch, hid, kernel_size=3, padding=1)
        self.rate_net = nn.Sequential(
            nn.Conv2d(in_ch + 2 * hid, hid, kernel_size=3, padding=1),  # 3x3 neighbourhood info
            nn.GELU(),
            nn.Conv2d(hid, 2 * hid, kernel_size=3, padding=1),         # heads: raw_merge | raw_split
        )
        self.readout = nn.Conv2d(2 * hid, 1, kernel_size=1)

    def initial_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c1 = F.softplus(self.encoder(x))     # softplus -> strictly positive monomers
        c2 = torch.zeros_like(c1)
        return c1, c2

    def forward(self, x: torch.Tensor, T: int, record: bool = False):
        c1, c2 = self.initial_state(x)
        trace: Dict[str, List[torch.Tensor]] = {"c1": [], "c2": [], "m": [], "s": [], "prob": []}
        for _ in range(T):
            raw = self.rate_net(torch.cat([x, c1, c2], dim=1))
            raw_merge, raw_split = raw.chunk(2, dim=1)
            m = (0.5 if self.bound_flux else 1.0) * c1 * torch.sigmoid(raw_merge)
            s = c2 * torch.sigmoid(raw_split) if self.fragmentation else torch.zeros_like(c2)
            c1 = c1 - 2.0 * m + 2.0 * s
            c2 = c2 + m - s
            if record:
                trace["c1"].append(c1.detach().cpu())
                trace["c2"].append(c2.detach().cpu())
                trace["m"].append(m.detach().cpu())
                trace["s"].append(s.detach().cpu())
                prob = torch.sigmoid(self.readout(torch.cat([c1, c2], dim=1)).squeeze(1))
                trace["prob"].append(prob.detach().cpu())
        logits = self.readout(torch.cat([c1, c2], dim=1)).squeeze(1)
        return (logits, trace) if record else logits


class RecurrentCNN(nn.Module):
    """Parameter-matched looped-processor baseline.

    Weight-shared residual convolutional cell with per-step input injection
    (the bare CNN of a ConvGRU-style processor). No coagulation/fragmentation
    gate, no mass bookkeeping — same loop template, plain residual state.
    """

    def __init__(self, in_ch: int = 3, hid: int = 22) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(in_ch, hid, kernel_size=3, padding=1)
        self.cell = nn.Sequential(
            nn.Conv2d(in_ch + hid, hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, hid, kernel_size=3, padding=1),
        )
        self.readout = nn.Conv2d(hid, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, T: int, record: bool = False):
        h = self.encoder(x)
        for _ in range(T):
            h = h + self.cell(torch.cat([x, h], dim=1))
        logits = self.readout(h).squeeze(1)
        return (logits, {}) if record else logits


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
