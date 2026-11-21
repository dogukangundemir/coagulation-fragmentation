"""Shared helpers: small synthetic lattice mazes in the curriculum format.

Offline stand-in for the Hugging Face curriculum rows: a perfect maze from a
deterministic recursive backtracker on an n x n lattice, rendered as the
dataset's ASCII format ('#' wall, 'S' start, 'E' goal, space = free), plus a
provided solution string of moves (N/S/E/W, 2 ASCII cells per move).
"""
from collections import deque

import numpy as np


def make_maze_grid(rng, n=5):
    """Perfect maze (spanning tree, no braids) as a (2n+1, 2n+1) wall grid."""
    size = 2 * n + 1
    grid = np.ones((size, size), np.int8)
    visited = np.zeros((n, n), bool)
    visited[0, 0] = True
    grid[1, 1] = 0
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        cand = [(i + di, j + dj) for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]]
        if cand:
            ni, nj = cand[int(rng.integers(len(cand)))]
            grid[2 * ni + 1, 2 * nj + 1] = 0
            grid[i + ni + 1, j + nj + 1] = 0
            visited[ni, nj] = True
            stack.append((ni, nj))
        else:
            stack.pop()
    return grid


DELTAS = {"N": (-2, 0), "S": (2, 0), "E": (0, 2), "W": (0, -2)}


def bfs_solution(grid, start):
    """BFS over lattice moves (2 ASCII cells per move); returns cell -> (prev, move)."""
    parents = {start: (None, None)}
    order = deque([start])
    while order:
        r, c = order.popleft()
        for mv, (dr, dc) in DELTAS.items():
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                    and grid[nr, nc] == 0 and grid[r + dr // 2, c + dc // 2] == 0
                    and (nr, nc) not in parents):
                parents[(nr, nc)] = ((r, c), mv)
                order.append((nr, nc))
    return parents


def moves_to(goal, parents):
    path, cur = [], goal
    while parents[cur][0] is not None:
        prev, mv = parents[cur]
        path.append(mv)
        cur = prev
    return "".join(reversed(path))


def make_ascii(grid, start, goal):
    lines = []
    for r in range(grid.shape[0]):
        row = ""
        for c in range(grid.shape[1]):
            if grid[r, c]:
                row += "#"
            elif (r, c) == start:
                row += "S"
            elif (r, c) == goal:
                row += "E"
            else:
                row += " "
        lines.append(row)
    return "\n".join(lines)


def make_example(rng, n=5, min_moves=4):
    """(ascii, solution) pair in the curriculum format: S=(1,1) to a far E."""
    grid = make_maze_grid(rng, n)
    start = (1, 1)
    parents = bfs_solution(grid, start)
    lattice = [(2 * i + 1, 2 * j + 1) for i in range(n) for j in range(n)]
    goals = [g for g in lattice if g in parents and len(moves_to(g, parents)) >= min_moves]
    goal = goals[int(rng.integers(len(goals)))]
    return make_ascii(grid, start, goal), moves_to(goal, parents)


def make_batch(seed=0, count=8, n=5, min_moves=4):
    """Several (ascii, solution) examples for MazeDataset-style use."""
    rng = np.random.default_rng(seed)
    return [make_example(rng, n=n, min_moves=min_moves) for _ in range(count)]
