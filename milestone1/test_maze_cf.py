"""Milestone-1 acceptance tests for maze_cf.  Run: python -m pytest -q milestone1/test_maze_cf.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # milestone1 dir: maze_cf

import maze_cf as m

SIZES = (7, 9, 15)


# --- Test 1: the BFS label is a genuine, valid shortest path ------------------
def test_bfs_label_is_a_valid_shortest_path():
    rng = np.random.default_rng(0)
    for size in SIZES:
        for _ in range(5):
            grid, start, goal, mask = m.make_sample(
                rng, size, braid=0.08, min_path_len=max(2 * (size - 1) // 3, 2)
            )
            assert mask[start] and mask[goal], "endpoints must lie on the shortest path"
            assert not (mask & (grid == 1)).any(), "labels must only cover free cells"
            assert m.path_exists(mask, start, goal), "label must be end-to-end connected"
            ds = m.bfs_distances(grid, start)
            dg = m.bfs_distances(grid, goal)
            assert np.all(ds[mask] + dg[mask] == ds[goal]), "every label cell on a shortest path"
            assert ds[goal] > 0, "start and goal must be distinct and connected"


# --- Test 2: non-negativity of both concentrations ----------------------------
def test_concentrations_stay_nonnegative():
    torch.manual_seed(0)
    model = m.CoagulationFragmentationNet(hid=8)
    x = m.MazeDataset(n_samples=4, size=9, seed=1).x
    _, trace = model(x, T=16, record=True)
    for step in range(16):
        assert (trace["c1"][step] >= -1e-7).all(), f"c1 < 0 at t={step+1}"
        assert (trace["c2"][step] >= -1e-7).all(), f"c2 < 0 at t={step+1}"


# --- Test 3: exact mass conservation to 1e-5 across all steps -----------------
def test_mass_is_exactly_conserved():
    torch.manual_seed(0)
    model = m.CoagulationFragmentationNet(hid=8)
    x = m.MazeDataset(n_samples=4, size=9, seed=1).x
    c1_0, c2_0 = model.initial_state(x)
    mass_0 = c1_0 + 2.0 * c2_0
    _, trace = model(x, T=16, record=True)
    for step in range(16):
        mass_t = trace["c1"][step] + 2.0 * trace["c2"][step]
        max_drift = (mass_t - mass_0).abs().max().item()
        assert max_drift < 1e-5, f"mass drifted {max_drift:.2e} at t={step+1}"


# --- Test 4: gradients flow to the input and every parameter ------------------
def test_gradients_flow_everywhere():
    torch.manual_seed(0)
    model = m.CoagulationFragmentationNet(hid=8)
    data = m.MazeDataset(n_samples=4, size=9, seed=2)
    x = data.x.clone().requires_grad_(True)
    logits = model(x, T=8)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, data.y)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0, "no gradient reached the input"
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached parameter {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


# --- Ablation sanity: coagulation-only still conserves mass --------------------
def test_no_frag_variant_conserves_mass_and_stays_nonnegative():
    torch.manual_seed(0)
    model = m.CoagulationFragmentationNet(hid=8, fragmentation=False)
    x = m.MazeDataset(n_samples=2, size=9, seed=3).x
    c1_0, c2_0 = model.initial_state(x)
    mass_0 = c1_0 + 2.0 * c2_0
    _, trace = model(x, T=8, record=True)
    for step in range(8):
        assert (trace["c1"][step] >= -1e-7).all()
        assert (trace["c2"][step] >= -1e-7).all()
        assert (trace["c1"][step] + 2.0 * trace["c2"][step] - mass_0).abs().max() < 1e-5
