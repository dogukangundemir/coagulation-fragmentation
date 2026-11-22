"""Invariant tests for the Stage-A coagulation-fragmentation model (packet 7.4).

Covers: non-negativity, exact mass conservation, gradient reach, CF-NoFrag
head ablation, wall-aware masking/leak-tightness, maze_cf equivalence,
stochastic initialisation and the cross kernel.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: maze_cf

from cf.models.cf import CFNet, MODEL_SPECS
import maze_cf

from maze_synthetic import make_maze_grid

NEG_EPS = -1e-7  # float32 round-off tolerance for non-negativity


# --------------------------------------------------------------------------- helpers
def synthetic_x(seed=0, n=5):
    """(1, 3, 2n+1, 2n+1) curriculum channels from a small perfect lattice maze."""
    rng = np.random.default_rng(seed)
    grid = make_maze_grid(rng, n)
    walk = (grid == 0).astype(np.float32)
    start = np.zeros_like(walk)
    start[1, 1] = 1.0
    goal = np.zeros_like(walk)
    goal[-2, -2] = 1.0  # lattice cell (2n-1, 2n-1) is always free
    return torch.from_numpy(np.stack([walk, start, goal]))[None]


def random_grid_x(seed=0, size=30):
    """(1, 3, size, size) random grid: channels [walkable, start, goal]."""
    rng = np.random.default_rng(seed)
    walk = (rng.random((size, size)) > 0.35).astype(np.float32)
    free = np.argwhere(walk == 1.0)
    if len(free) < 2:  # degenerate draw: make it all walkable
        walk[:] = 1.0
        free = np.argwhere(walk == 1.0)
    start = np.zeros_like(walk)
    start[free[0][0], free[0][1]] = 1.0
    goal = np.zeros_like(walk)
    goal[free[-1][0], free[-1][1]] = 1.0
    return torch.from_numpy(np.stack([walk, start, goal]))[None]


def rollout_trace(model, x, T):
    """Trace of one T-step rollout: c1, c2, M, m, s recorded after each step."""
    _, trace = model.logits_at(x, Ts=[T], record=True)
    assert len(trace["c1"]) == T
    return trace


# ------------------------------------------------------------------ 1. non-negativity
NONNEG_MODEL_NAMES = ["CF-Full", "CF-NoFrag", "CF-WallAware"]


@pytest.mark.parametrize("damping", [False, True])
@pytest.mark.parametrize("cross_kernel", [False, True])
@pytest.mark.parametrize("model_name", NONNEG_MODEL_NAMES)
@pytest.mark.parametrize("x_factory", ["maze", "random"])
def test_nonnegativity_at_init_and_every_step(model_name, x_factory,
                                              cross_kernel, damping):
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS[model_name], cross_kernel=cross_kernel,
                  damping=damping)
    x = synthetic_x(seed=1) if x_factory == "maze" else random_grid_x(seed=1)
    c1, c2 = model.initial_state(x)
    assert c1.min().item() >= NEG_EPS  # t = 0, before any step
    assert c2.min().item() >= NEG_EPS
    trace = rollout_trace(model, x, 64)
    for t, (c1_t, c2_t) in enumerate(zip(trace["c1"], trace["c2"]), start=1):
        assert c1_t.min().item() >= NEG_EPS, f"c1 < 0 at t={t}"
        assert c2_t.min().item() >= NEG_EPS, f"c2 < 0 at t={t}"


def test_nonnegativity_after_50_sgd_steps():
    torch.manual_seed(0)
    model = CFNet()  # CF-Full
    xs = torch.cat([synthetic_x(seed=s) for s in range(4)])
    ys = (torch.rand(xs.shape[0], xs.shape[2], xs.shape[3]) > 0.5).float()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(50):  # BCE on random labels
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xs, 8), ys)
        loss.backward()
        opt.step()
    c1, c2 = model.initial_state(xs)
    assert c1.min().item() >= NEG_EPS
    assert c2.min().item() >= NEG_EPS
    trace = rollout_trace(model, xs, 64)
    for t, (c1_t, c2_t) in enumerate(zip(trace["c1"], trace["c2"]), start=1):
        assert c1_t.min().item() >= NEG_EPS, f"c1 < 0 at t={t} after SGD"
        assert c2_t.min().item() >= NEG_EPS, f"c2 < 0 at t={t} after SGD"


# ------------------------------------------------------------------------ 2. mass
@pytest.mark.parametrize("model_name", list(MODEL_SPECS))
def test_mass_relative_drift_below_1e_minus_5(model_name):
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS[model_name])
    x = random_grid_x(seed=2)
    c1, c2 = model.initial_state(x)
    M0 = (c1 + 2.0 * c2).detach()
    scale = max(1.0, M0.max().item())
    for t, M_t in enumerate(rollout_trace(model, x, 64)["M"], start=1):
        drift = (M_t - M0).abs().max().item() / scale
        assert drift < 1e-5, f"mass drifted by {drift} at t={t}"


def test_nobound_still_conerves_mass_but_can_go_negative():
    # Without the 1/2 bound the stoichiometry still conserves M exactly, but
    # nothing keeps c1 >= 0: a merge-happy rate net drives c1 negative.  This
    # is exactly why `bound_flux` matters for positivity / long-rollout stability.
    torch.manual_seed(0)
    model = CFNet(bound_flux=False)  # CF-NoBound
    with torch.no_grad():  # push sigmoid(raw_merge) ~ 0.73: merge dominates
        model.rate_net[2].bias[: model.rate_net[2].out_channels // 2].fill_(1.0)
    x = random_grid_x(seed=2)
    c1, c2 = model.initial_state(x)
    M0 = (c1 + 2.0 * c2).detach()
    scale = max(1.0, M0.max().item())
    trace = rollout_trace(model, x, 64)
    for M_t in trace["M"]:
        assert (M_t - M0).abs().max().item() / scale < 1e-5
    assert torch.stack(trace["c1"]).min().item() < 0.0, \
        "CF-NoBound should be able to drive c1 negative (bound is what prevents it)"


# ------------------------------------------------------------------ 3. gradients
GRAD_CONFIGS = [
    pytest.param(dict(damping=True), id="full-damping"),
    pytest.param(dict(wall_aware=True, damping=True), id="wallaware-damping"),
    pytest.param(dict(cross_kernel=True, damping=True), id="crosskernel-damping"),
]


@pytest.mark.parametrize("kwargs", GRAD_CONFIGS)
def test_gradients_reach_input_and_every_parameter(kwargs):
    torch.manual_seed(0)
    model = CFNet(**kwargs)  # includes damp_logit when damping=True
    x = synthetic_x(seed=3).requires_grad_(True)
    logits = model(x, 8)
    logits.square().sum().backward()
    assert x.grad is not None, "no grad on input x"
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"


# ------------------------------------------------------ 3b. damping trace
def test_damped_trace_records_applied_fluxes_and_tiny_residual():
    # Very negative damp_logit: g ~ sigmoid(-20) ~ 2e-9. The trace must record
    # the APPLIED fluxes (g*m, g*s) and g itself, so sum|c2_{t+1} - c2_t| must
    # equal the recorded applied-flux difference, and residual_flux be tiny.
    torch.manual_seed(0)
    model = CFNet(damping=True)
    with torch.no_grad():
        model.damp_logit.fill_(-20.0)
    x = random_grid_x(seed=9)
    T = 8
    _, trace = model.logits_at(x, Ts=[T], record=True)
    g = torch.sigmoid(model.damp_logit).item()
    assert len(trace["g"]) == T
    assert all(abs(g_rec - g) <= 1e-12 for g_rec in trace["g"])
    assert max(trace["residual_flux"]) < 1e-8
    _, c2_prev = model.initial_state(x)
    c2_prev = c2_prev.detach()
    for t in range(T):
        state_change = (trace["c2"][t] - c2_prev).abs().sum().item()
        flux_change = (trace["m"][t] - trace["s"][t]).abs().sum().item()
        assert abs(state_change - flux_change) <= 1e-5 * flux_change, \
            f"c2 change does not match applied flux at t={t + 1}"
        c2_prev = trace["c2"][t]


# --------------------------------------------------------------- 4. CF-NoFrag head
def test_nofrag_split_is_exactly_zero_and_c2_monotone():
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-NoFrag"])
    x = synthetic_x(seed=4)
    c1_0, c2_0 = model.initial_state(x)
    trace = rollout_trace(model, x, 16)
    for t, s_t in enumerate(trace["s"], start=1):
        assert (s_t == 0).all(), f"split flux s != 0 at t={t}"
    c2_prev = c2_0.detach()
    for t, c2_t in enumerate(trace["c2"], start=1):
        assert (c2_t >= c2_prev).all(), f"c2 decreased at t={t}"
        c2_prev = c2_t


def test_nofrag_split_head_gets_zero_grad_merge_head_nonzero():
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-NoFrag"])
    x = synthetic_x(seed=4).requires_grad_(True)
    model(x, 8).square().sum().backward()
    last = model.rate_net[2]
    hid = last.out_channels // 2  # heads: [0:hid]=raw_merge, [hid:]=raw_split
    assert (last.weight.grad[hid:] == 0).all(), "split-half weight grad != 0"
    assert (last.bias.grad[hid:] == 0).all(), "split-half bias grad != 0"
    assert last.weight.grad[:hid].abs().sum() > 0, "merge-half weight grad is 0"
    assert last.bias.grad[:hid].abs().sum() > 0, "merge-half bias grad is 0"


# --------------------------------------------------------------- 5. CF-WallAware
def test_wallaware_state_exactly_zero_on_walls_every_step():
    torch.manual_seed(0)
    model = CFNet(**MODEL_SPECS["CF-WallAware"])
    for x in (synthetic_x(seed=5), random_grid_x(seed=5)):
        wall = x[0, 0] == 0  # (H, W) boolean wall mask
        c1, c2 = model.initial_state(x)
        assert (c1[:, :, wall] == 0).all()
        assert (c2[:, :, wall] == 0).all()
        trace = rollout_trace(model, x, 16)
        for t, (c1_t, c2_t) in enumerate(zip(trace["c1"], trace["c2"]), start=1):
            assert (c1_t[:, :, wall] == 0).all(), f"c1 != 0 on wall at t={t}"
            assert (c2_t[:, :, wall] == 0).all(), f"c2 != 0 on wall at t={t}"


def test_wallaware_wall_logits_are_exactly_minus_20():
    torch.manual_seed(0)
    model = CFNet(wall_aware=True)
    x = synthetic_x(seed=10)
    wall = x[0, 0] == 0
    logits = model(x, 8)
    assert (logits[:, wall] == -20.0).all(), "wall logits are not exactly -20"


def test_wallaware_all_wall_grid_residual_flux_is_zero():
    # All-wall input: total mass is zero, residual_flux must be 0.0 (guarded),
    # not a ZeroDivisionError or NaN.
    torch.manual_seed(0)
    model = CFNet(wall_aware=True)
    walk = np.zeros((9, 9), np.float32)
    x = torch.from_numpy(np.stack([walk, walk, walk]))[None]
    logits = model(x, 4)
    assert (logits == -20.0).all()
    _, trace = model.logits_at(x, Ts=[4], record=True)
    assert all(r == 0.0 for r in trace["residual_flux"])


def _split_room_x(start_pos, size=9):
    """Free room split by a one-cell full-height vertical wall at column 4."""
    walk = np.ones((size, size), np.float32)
    walk[:, size // 2] = 0.0
    start = np.zeros_like(walk)
    start[start_pos] = 1.0
    goal = np.zeros_like(walk)
    goal[size // 2, size - 3] = 1.0
    return torch.from_numpy(np.stack([walk, start, goal]))[None]


def test_wallaware_no_leak_across_wall_plain_model_leaks():
    size = 9
    xa, xb = _split_room_x((2, 2)), _split_room_x((size - 3, 2))
    right = slice(size // 2 + 1, None)  # logits columns strictly right of wall
    torch.manual_seed(7)
    walled = CFNet(wall_aware=True, cross_kernel=True).eval()
    with torch.no_grad():
        leak_walled = (walled(xa, 16) - walled(xb, 16))[:, :, right]
    assert leak_walled.abs().max().item() < 1e-7, "start leaked across the wall"
    torch.manual_seed(7)
    plain = CFNet().eval()  # same seed: validates the probe is sensitive
    with torch.no_grad():
        leak_plain = (plain(xa, 16) - plain(xb, 16))[:, :, right]
    assert leak_plain.abs().max().item() > 1e-6, \
        "plain CFNet should leak across the wall (test sensitivity check)"


# --------------------------------------------------------------- 6. equivalence
def test_param_counts():
    assert sum(p.numel() for p in CFNet().parameters()) == 10177
    assert sum(p.numel() for p in CFNet(damping=True).parameters()) == 10178


def test_matches_maze_cf_reference_with_shared_weights():
    torch.manual_seed(0)
    legacy = maze_cf.CoagulationFragmentationNet(hid=16)
    model = CFNet()  # same module names -> state_dict transfers
    model.load_state_dict(legacy.state_dict())
    model.eval()
    x = synthetic_x(seed=6)
    with torch.no_grad():
        ref = legacy(x, 8)
        got = model(x, 8)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=0)


@pytest.mark.parametrize("T", [1, 4, 8, 24])
def test_logits_at_matches_forward(T):
    torch.manual_seed(0)
    model = CFNet()
    x = synthetic_x(seed=7)
    outs = model.logits_at(x, [1, 4, 8, 24])
    torch.testing.assert_close(outs[T], model(x, T), atol=1e-6, rtol=0)


# ------------------------------------------------------------ 7. stochastic init
def test_stochastic_init_keeps_mass_and_positivity():
    torch.manual_seed(0)
    model = CFNet()
    x = random_grid_x(seed=8)
    c1_d, c2_d = model.initial_state(x)
    M_det = c1_d + 2.0 * c2_d
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(456)
    c1_a, c2_a = model.initial_state(x, stochastic=True, generator=g1)
    c1_b, c2_b = model.initial_state(x, stochastic=True, generator=g2)
    torch.testing.assert_close(c1_a + 2.0 * c2_a, M_det, atol=1e-6, rtol=0)
    torch.testing.assert_close(c1_b + 2.0 * c2_b, M_det, atol=1e-6, rtol=0)
    assert c1_a.min().item() >= 0 and c2_a.min().item() >= 0
    assert c1_b.min().item() >= 0 and c2_b.min().item() >= 0
    assert (c2_a - c2_b).abs().max().item() > 1e-3, \
        "two generators gave identical c2"


def test_stochastic_init_wallaware_keeps_exact_zeros_on_walls():
    torch.manual_seed(0)
    model = CFNet(wall_aware=True)
    x = random_grid_x(seed=11)
    wall = x[0, 0] == 0
    gen = torch.Generator().manual_seed(789)
    c1, c2 = model.initial_state(x, stochastic=True, generator=gen)
    assert (c1[:, :, wall] == 0).all(), "stochastic init put mass on walls (c1)"
    assert (c2[:, :, wall] == 0).all(), "stochastic init put mass on walls (c2)"


# --------------------------------------------------------------- 8. cross kernel
KERNEL_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))


def _backward_corner_grads(model, x):
    """Grad of sum(logits^2) w.r.t. the four corner taps of every 3x3 conv."""
    model(x, 8).square().sum().backward()
    convs = {"encoder": model.encoder,
             "rate_net[0]": model.rate_net[0],
             "rate_net[2]": model.rate_net[2]}
    return {name: {corner: conv.weight.grad[..., corner[0], corner[1]].clone()
                   for corner in KERNEL_CORNERS}
            for name, conv in convs.items()}


def test_cross_kernel_masks_rate_net_not_encoder():
    # packet 7.5: with cross_kernel=True ONLY the rate net's two 3x3 convs
    # are cross-masked; the encoder keeps its full 3x3 kernel by default.
    torch.manual_seed(0)
    model = CFNet(cross_kernel=True)
    x = synthetic_x(seed=0)
    corners = _backward_corner_grads(model, x)
    for name in ("rate_net[0]", "rate_net[2]"):
        for corner, grad in corners[name].items():
            assert (grad == 0).all(), f"corner {corner} of {name} got gradient"
    for corner, grad in corners["encoder"].items():
        assert (grad != 0).any(), f"encoder corner {corner} is dead (masked?)"


def test_cross_kernel_rate_net_weight_corners_exactly_zero():
    torch.manual_seed(0)
    model = CFNet(cross_kernel=True)
    for conv in (model.rate_net[0], model.rate_net[2]):
        w_eff = conv.weight * model.kernel_mask[None, None]
        for corner in KERNEL_CORNERS:
            assert (w_eff[..., corner[0], corner[1]] == 0).all(), \
                f"corner {corner} not masked in {conv}"
        with torch.no_grad():  # probe: impulse response through _conv3
            inp = torch.zeros(1, conv.in_channels, 5, 5)
            inp[0, :, 1, 1] = 1.0
            out = model._conv3(conv, inp)  # out = w_eff * inp + bias
            assert torch.equal(out[0, :, 2, 2], conv.bias), \
                "diagonal (corner-kernel) response nonzero"
            assert (out[0, :, 2, 1] - conv.bias).abs().sum() > 0, \
                "edge response zero: probe cannot detect a removed mask"


def test_cross_kernel_encoder_corner_taps_do_not_affect_output():
    torch.manual_seed(0)
    model = CFNet(cross_kernel=False).eval()  # encoder never masked when CK off
    x = synthetic_x(seed=0)
    with torch.no_grad():
        base = model(x, 8)
        for corner in KERNEL_CORNERS:
            model.encoder.weight.data[..., corner[0], corner[1]] += 0.5
        pert = model(x, 8)
    assert not torch.equal(base, pert), \
        "probe insensitive: encoder corner taps do nothing even unmasked"


def test_cross_kernel_rate_net_corner_taps_have_no_effect():
    torch.manual_seed(0)
    model = CFNet(cross_kernel=True).eval()
    x = synthetic_x(seed=0)
    with torch.no_grad():
        base = model(x, 16)
        for conv in (model.rate_net[0], model.rate_net[2]):
            for corner in KERNEL_CORNERS:
                conv.weight.data[..., corner[0], corner[1]] += 0.5
        pert = model(x, 16)
    assert torch.equal(base, pert), \
        "rate-net corner taps changed the output despite the cross mask"


def test_cross_kernel_encoder_option_restores_encoder_masking():
    # cross_kernel_encoder=True restores the pre-7.5 behaviour: all 3x3
    # convolutions (encoder included) are cross-masked when cross_kernel=True.
    torch.manual_seed(0)
    model = CFNet(cross_kernel=True, cross_kernel_encoder=True)
    x = synthetic_x(seed=0)
    corners = _backward_corner_grads(model, x)
    for name in ("encoder", "rate_net[0]", "rate_net[2]"):
        for corner, grad in corners[name].items():
            assert (grad == 0).all(), f"corner {corner} of {name} got gradient"
    # and the option really masks the encoder at runtime, not just in grad
    model.eval()
    with torch.no_grad():
        base = model(x, 16)
        for corner in KERNEL_CORNERS:
            model.encoder.weight.data[..., corner[0], corner[1]] += 0.5
        pert = model(x, 16)
    assert torch.equal(base, pert), \
        "cross_kernel_encoder=True left encoder corner taps live"


def test_cross_kernel_encoder_option_ignored_without_cross_kernel():
    # the legacy option only matters together with cross_kernel=True
    torch.manual_seed(0)
    model = CFNet(cross_kernel=False, cross_kernel_encoder=True)
    x = synthetic_x(seed=0)
    corners = _backward_corner_grads(model, x)
    for corner, grad in corners["encoder"].items():
        assert (grad != 0).any(), \
            f"encoder corner {corner} masked though cross_kernel is False"
