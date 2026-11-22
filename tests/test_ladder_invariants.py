"""Invariant tests for the Family B Becker-Doring commitment ladder.

Mirrors the style of tests/test_cf_invariants.py: float32 tolerances,
NEG_EPS = -1e-7 for non-negativity, relative mass drift tolerance 1e-5.
Covers the invariant list of docs/specs/family_b_ladder.md: exact mass
conservation and truncation, positivity under the joint flux cap (including
extreme-rate probes and after SGD), detailed balance by construction, the
frozen-rate free-energy descent, wall handling and leak-tightness, gradient
reach to every head, the two-timescale rate refresh, the random-start
variant, and forward / logits_at agreement.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from cf.models.ladder import LadderCFNet, MODEL_SPECS

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
    """Trace of one T-step rollout: per outer step c, M, J, V, residual."""
    _, trace = model.logits_at(x, Ts=[T], record=True)
    assert len(trace["c"]) == T
    return trace


# ---------------------------------------------------------------- 1. mass + truncation
@pytest.mark.parametrize("K", [4, 6, 8])
def test_mass_relative_drift_below_1e_minus_5(K):
    torch.manual_seed(0)
    model = LadderCFNet(**MODEL_SPECS["BD-Full"], K=K)
    x = random_grid_x(seed=2)
    c0 = model.initial_state(x)
    M0 = model.mass(c0).detach()
    scale = max(1.0, M0.max().item())
    trace = rollout_trace(model, x, 64)
    assert trace["c"][0].shape[2] == K          # rung map has exactly K rungs
    assert trace["J"][0].shape[2] == K - 1      # J_K = 0 by truncation
    for t, (c_t, M_t) in enumerate(zip(trace["c"], trace["M"]), start=1):
        assert (model.mass(c_t) - M_t).abs().max().item() < 1e-6, \
            f"recorded M disagrees with sum_k k c_k at t={t}"
        drift = (M_t - M0).abs().max().item() / scale
        assert drift < 1e-5, f"mass drifted by {drift} at t={t}"


@pytest.mark.parametrize("model_name", list(MODEL_SPECS))
def test_mass_relative_drift_free_rates_and_variants(model_name):
    torch.manual_seed(0)
    model = LadderCFNet(**MODEL_SPECS[model_name])
    x = random_grid_x(seed=3)
    c0 = model.initial_state(x)
    M0 = model.mass(c0).detach()
    scale = max(1.0, M0.max().item())
    for t, M_t in enumerate(rollout_trace(model, x, 64)["M"], start=1):
        drift = (M_t - M0).abs().max().item() / scale
        assert drift < 1e-5, f"mass drifted by {drift} at t={t}"


def test_mass_conserved_exactly_in_one_micro_step():
    # With rates frozen, one micro-step conserves M = sum_k k c_k to float
    # round-off for any rate magnitudes, and the top-rung stock can only
    # change by the last applied flux (no flux past rung K).
    torch.manual_seed(0)
    model = LadderCFNet()
    x = random_grid_x(seed=4)
    with torch.no_grad():
        c = model.initial_state(x)
        rates = model.refresh_rates(x, c)
        c_new, J = model.micro_step(c, rates)
        scale = max(1.0, model.mass(c).max().item())
        drift = (model.mass(c_new) - model.mass(c)).abs().max().item() / scale
        assert drift < 1e-7
        torch.testing.assert_close(c_new[:, :, -1] - c[:, :, -1], J[:, :, -1],
                                   atol=1e-7, rtol=0)   # truncation J_K = 0


# ------------------------------------------------------------------- 2. positivity
@pytest.mark.parametrize("model_name", list(MODEL_SPECS))
@pytest.mark.parametrize("x_factory", ["maze", "random"])
def test_positivity_at_init_and_every_step(model_name, x_factory):
    torch.manual_seed(0)
    model = LadderCFNet(**MODEL_SPECS[model_name])
    x = synthetic_x(seed=1) if x_factory == "maze" else random_grid_x(seed=1)
    c0 = model.initial_state(x)
    assert c0.min().item() >= NEG_EPS  # t = 0, before any step
    for t, c_t in enumerate(rollout_trace(model, x, 64)["c"], start=1):
        assert c_t.min().item() >= NEG_EPS, f"c < 0 at t={t}"


def test_positivity_after_50_sgd_steps():
    torch.manual_seed(0)
    model = LadderCFNet()  # BD-Full
    xs = torch.cat([synthetic_x(seed=s) for s in range(4)])
    ys = (torch.rand(xs.shape[0], xs.shape[2], xs.shape[3]) > 0.5).float()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(50):  # BCE on random labels
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xs, 8), ys)
        loss.backward()
        opt.step()
    x = random_grid_x(seed=2)
    assert model.initial_state(x).min().item() >= NEG_EPS
    for t, c_t in enumerate(rollout_trace(model, x, 64)["c"], start=1):
        assert c_t.min().item() >= NEG_EPS, f"c < 0 at t={t} after SGD"


@pytest.mark.parametrize("mode,gsign", [("merge-happy", -30.0), ("shed-happy", 30.0)])
def test_positivity_extreme_rates_joint_cap_protects(mode, gsign):
    # Drive the raw heads to extremes: mobility bias +30 (mu ~ 30) and a
    # one-sided energy profile (g_1 = 0, g_{k>1} = -30 wants to merge,
    # +30 wants to shed). Positivity and conservation then rest entirely
    # on the joint cap, not on the smallness of the rates (g weights are
    # exactly zero at initialisation, so the bias is the whole profile).
    torch.manual_seed(0)
    model = LadderCFNet()
    hid, K = model.hid, model.K
    with torch.no_grad():
        model.rate_net[2].bias[hid * K:] = 30.0            # mobility ~ 30
        for h in range(hid):
            model.rate_net[2].bias[h * K] = 0.0            # g_1 = 0 (gauge)
            model.rate_net[2].bias[h * K + 1:(h + 1) * K] = gsign
    x = synthetic_x(seed=11)
    c0 = model.initial_state(x)
    M0 = model.mass(c0)
    scale = max(1.0, M0.max().item())
    trace = rollout_trace(model, x, 16)
    for t, c_t in enumerate(trace["c"], start=1):
        assert c_t.min().item() >= NEG_EPS, f"{mode}: c < 0 at t={t}"
    for t, M_t in enumerate(trace["M"], start=1):
        drift = (M_t - M0).abs().max().item() / scale
        assert drift < 1e-5, f"{mode}: mass drifted by {drift} at t={t}"


def test_extreme_random_rates_64_step_rollout_stays_finite_and_conserves_mass():
    # Regression for the BD-WallAware NaN (results/ladder_train): the
    # trained run's V trace hit -inf and then NaN from outer step 11 on,
    # and the flux cap turned overflowing requests into 0 * inf = NaN. A
    # 64-step rollout under extreme random rates -- log-uniform over
    # [1e-6, 1e38], i.e. finite rates up to the float32 ceiling -- must
    # keep every rung and flux finite and non-negative and conserve M:
    # conservation is an identity of the rung balances, and where the cap
    # lambda is 0 (an overflowing request) the applied flux is exactly 0.
    torch.manual_seed(0)
    model = LadderCFNet()
    x = random_grid_x(seed=13)
    with torch.no_grad():
        c = model.initial_state(x)
        M0 = model.mass(c)
        scale = max(1.0, M0.max().item())
        B, hid, K, H, W = c.shape

        def draw():
            return 10.0 ** (44.0 * torch.rand(B, hid, K - 1, H, W) - 6.0)

        rates = (draw(), draw())                  # a, b ~ log-U(1e-6, 1e38)
        for t in range(64):
            c, J = model.micro_step(c, rates)
            assert torch.isfinite(c).all(), f"c not finite at step {t + 1}"
            assert torch.isfinite(J).all(), f"J not finite at step {t + 1}"
            assert c.min().item() >= NEG_EPS, f"c < 0 at step {t + 1}"
            drift = (model.mass(c) - M0).abs().max().item() / scale
            assert drift < 1e-5, f"mass drifted by {drift} at step {t + 1}"

        # Targeted probe of the cap guard: finite rates, a heavy rung 1
        # (c_1 = 10) and a = 1e38 make the merge request r_1 = a c_1^2
        # overflow to inf, so the cap is lambda = 0 and the applied flux
        # must be exactly 0 (0 * inf = NaN without the guard): the state
        # is unchanged and mass is conserved exactly.
        c = torch.zeros_like(c)
        c[:, :, 0] = 10.0
        a = torch.full((B, hid, K - 1, H, W), 1e38)
        b = torch.full((B, hid, K - 1, H, W), 1e-6)
        c_new, J = model.micro_step(c, (a, b))
        assert torch.isfinite(J).all(), "0 * inf leaked through the cap"
        assert torch.isfinite(c_new).all()
        torch.testing.assert_close(c_new, c, atol=0.0, rtol=0)
        drift = (model.mass(c_new) - model.mass(c)).abs().max().item()
        assert drift < 1e-5, f"mass drifted by {drift} under the capped step"


@pytest.mark.parametrize("gsign", [-1000.0, 1000.0])
def test_extreme_energy_profile_64_step_rollout_stays_finite(gsign):
    # The other half of the root cause: an energy profile far outside the
    # exp-safe range (here +/-1000, way past the g clamp) used to overflow
    # Q = exp(g_1 - g_k) to inf / underflow it to 0, so the shed rates
    # b = mu Q_k / Q_{k+1} came out as inf/inf or 0/0 NaN and the whole
    # rollout (state, fluxes, free energy) went NaN. With the clamp the
    # rates, the state, the fluxes and V all stay finite over 64 steps.
    torch.manual_seed(0)
    model = LadderCFNet()
    hid, K = model.hid, model.K
    with torch.no_grad():
        model.rate_net[2].bias[hid * K:] = 30.0            # mobility ~ 30
        for h in range(hid):
            model.rate_net[2].bias[h * K] = 0.0             # g_1 = 0 (gauge)
            model.rate_net[2].bias[h * K + 1:(h + 1) * K] = gsign
    x = random_grid_x(seed=13)
    c0 = model.initial_state(x)
    M0 = model.mass(c0)
    scale = max(1.0, M0.max().item())
    with torch.no_grad():
        c = c0
        rates = None
        for t in range(1, 65):
            if rates is None or (t - 1) % model.r_s == 0:
                rates = model.refresh_rates(x, c)
            a, b = rates
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), \
                f"NaN/inf rates at t={t}"
            c, J = model.fast_phase(c, rates)
            assert torch.isfinite(c).all(), f"c not finite at t={t}"
            assert torch.isfinite(J[-1]).all(), f"J not finite at t={t}"
            assert c.min().item() >= NEG_EPS, f"c < 0 at t={t}"
            drift = (model.mass(c) - M0).abs().max().item() / scale
            assert drift < 1e-5, f"mass drifted by {drift} at t={t}"
            assert np.isfinite(float(model.free_energy(c, rates))), \
                f"free energy not finite at t={t}"


# ------------------------------------------------- 3. detailed balance by construction
def test_detailed_balance_equilibrium_has_zero_flux():
    # For random (g, mu) outputs and a random activity z, the state
    # c_k = Q_k z^k has zero net flux on every rung. Checked in float64
    # (the algebra is exact) and in float32 at the spec tolerance
    # |J| < 1e-7 (1 + M); the mass M = sum_k k Q_k z^k follows from z.
    for dt in (torch.float64, torch.float32):
        torch.manual_seed(2)
        model = LadderCFNet()
        B, hid, K, H, W = 2, 3, model.K, 6, 6
        g = 0.5 * torch.randn(B, hid, K, H, W, dtype=dt)
        mu = F.softplus(0.5 * torch.randn(B, hid, 1, H, W, dtype=dt))
        Q = torch.exp(g[:, :, 0:1] - g)                    # gauge Q_1 = 1
        z = 0.3 + 0.5 * torch.rand(B, hid, 1, 1, 1, dtype=dt)
        kk = torch.arange(1, K + 1, dtype=dt).view(1, 1, -1, 1, 1)
        c = Q * z ** kk
        M = (kk * c).sum()
        a = mu.expand(B, hid, K - 1, H, W)
        b = mu * Q[:, :, :-1] / Q[:, :, 1:]                # b_{k+1} = mu Q_k/Q_{k+1}
        with torch.no_grad():
            _, J = model.micro_step(c, (a, b))
        assert J.abs().max().item() < 1e-7 * (1.0 + M.item()), \
            f"equilibrium state has nonzero flux (dtype {dt})"


def test_detailed_balance_arm_rates_satisfy_the_identity():
    # a_k = mu on every rung (one mobility), b ratios tied to a single profile.
    torch.manual_seed(0)
    model = LadderCFNet()  # BD-Full
    x = random_grid_x(seed=6)
    with torch.no_grad():
        c = model.initial_state(x)
        a, b = model.refresh_rates(x, c)
        assert (a.amax(dim=2) - a.amin(dim=2)).abs().max().item() == 0.0, \
            "a_k must be the same mobility on every rung"
        assert (a > 0).all() and (b > 0).all(), "rates must stay positive"


def test_free_rates_arm_lacks_the_structural_constraint():
    # The free-rates arm parameterizes a_k and b_{k+1} directly, so the
    # detailed-balance identity of the BD-Full arm is absent by design:
    # the rates vary across rungs and b/a is not a single profile's ratios.
    # The equilibrium-weight reconstruction stays well defined, so the same
    # free-energy formula can still be evaluated.
    torch.manual_seed(0)
    model = LadderCFNet(**MODEL_SPECS["BD-FreeRates"])
    x = random_grid_x(seed=6)
    with torch.no_grad():
        c = model.initial_state(x)
        a, b = model.refresh_rates(x, c)
        assert (a.amax(dim=2) - a.amin(dim=2)).max().item() > 0, \
            "a_k should vary across rungs in the free-rates arm"
        assert ((b / a).amax(dim=2) - (b / a).amin(dim=2)).max().item() > 0, \
            "b_{k+1}/a_k should vary across rungs in the free-rates arm"
        Q = model.rung_weights((a, b))
        assert torch.isfinite(Q).all() and (Q > 0).all()


# ------------------------------------------- 4. free energy under frozen rates
def test_free_energy_descends_and_settles_under_model_rates():
    # Freeze the slow phase (compute the rates once, then only micro-steps)
    # and record V = sum_k c_k (log(c_k/Q_k) - 1) after every micro-step.
    # Hard assertions per the spec: from the all-monomer start V decreases
    # strictly beyond the tolerance at least once before the state is
    # chemically at rest, and V at rest is below V at the start. Per
    # micro-step monotonicity is a monitored diagnostic (the capped
    # forward-Euler step is a descent only in continuous time), so the
    # largest increase is reported, not asserted.
    torch.manual_seed(0)
    model = LadderCFNet()
    x = random_grid_x(seed=9)
    V0, Vs, resid = _frozen_rate_run(model, x, outer=32)
    tol = [1e-6 * max(1.0, abs(V)) for V in Vs]
    rest = next((s for s, r in enumerate(resid, start=1) if r < 1e-5), None)
    horizon = rest if rest is not None else len(Vs) - 1
    strict = [s for s in range(horizon) if Vs[s + 1] < Vs[s] - tol[s]]
    assert strict, "V never decreased strictly before chemical rest"
    v_rest = Vs[rest] if rest is not None else Vs[-1]
    assert v_rest < V0, "V at rest is not below V at the start"
    increases = [(s, Vs[s + 1] - Vs[s]) for s in range(horizon)
                 if Vs[s + 1] > Vs[s] + tol[s]]
    max_inc = max(d for _, d in increases) if increases else 0.0
    print(f"[energy monitor] rest at micro-step {rest}, "
          f"largest per-micro-step V increase: {max_inc}")


def test_free_energy_non_increasing_under_fixed_rates_small_step():
    # With small mobilities the capped step is close to the continuous-time
    # descent, and V must be non-increasing at every micro-step.
    torch.manual_seed(3)
    model = LadderCFNet()
    x = random_grid_x(seed=5)
    B, hid, K, H, W = 1, model.hid, model.K, x.shape[2], x.shape[3]
    g = 0.5 * torch.randn(B, hid, K, H, W)
    Q = torch.exp(g[:, :, 0:1] - g)
    mu = 0.02  # small step: per-micro-step fluxes are a small fraction of stock
    a = torch.full((B, hid, K - 1, H, W), mu)
    b = mu * Q[:, :, :-1] / Q[:, :, 1:]
    with torch.no_grad():
        c = model.initial_state(x)
        rates = (a, b)
        Vs = [float(model.free_energy(c, rates))]
        for s in range(64):
            c, _ = model.micro_step(c, rates)
            Vs.append(float(model.free_energy(c, rates)))
    for s in range(len(Vs) - 1):
        assert Vs[s + 1] <= Vs[s] + 1e-6 * max(1.0, abs(Vs[s])), \
            f"V increased at micro-step {s + 1} ({Vs[s + 1]} > {Vs[s]})"
    assert Vs[-1] < Vs[0]


def _frozen_rate_run(model, x, outer):
    """Initial state, one rate refresh, then `outer` outer steps of frozen-rate
    fast phases; returns V0, per-micro-step V list (V0 first) and the
    per-micro-step residual flux sum|J| / sum M."""
    with torch.no_grad():
        c = model.initial_state(x)
        rates = model.refresh_rates(x, c)
        V0 = float(model.free_energy(c, rates))
        Vs, resid = [V0], []
        for _ in range(outer * model.n_f):
            c, J = model.micro_step(c, rates)
            Vs.append(float(model.free_energy(c, rates)))
            M_total = model.mass(c).sum().item()
            resid.append(J.abs().sum().item() / M_total if M_total > 0.0 else 0.0)
    return V0, Vs, resid


# ----------------------------------------------------------------------- 5. walls
def test_wallaware_state_exactly_zero_on_walls_every_step():
    torch.manual_seed(0)
    model = LadderCFNet()  # wall-aware is this family's default
    assert model.wall_aware
    for x in (synthetic_x(seed=5), random_grid_x(seed=5)):
        wall = x[0, 0] == 0  # (H, W) boolean wall mask
        assert (model.initial_state(x)[:, :, :, wall] == 0).all()
        for t, c_t in enumerate(rollout_trace(model, x, 16)["c"], start=1):
            assert (c_t[:, :, :, wall] == 0).all(), f"c != 0 on wall at t={t}"


def test_wallaware_stochastic_start_keeps_exact_zero_on_walls():
    torch.manual_seed(0)
    model = LadderCFNet()
    x = random_grid_x(seed=11)
    wall = x[0, 0] == 0
    gen = torch.Generator().manual_seed(789)
    c = model.initial_state(x, stochastic=True, generator=gen)
    assert (c[:, :, :, wall] == 0).all(), "stochastic init put mass on walls"
    for t, c_t in enumerate(rollout_trace(model, x, 16)["c"], start=1):
        assert (c_t[:, :, :, wall] == 0).all(), f"c != 0 on wall at t={t}"


def test_wallaware_wall_logits_are_exactly_minus_20():
    torch.manual_seed(0)
    model = LadderCFNet()
    x = synthetic_x(seed=10)
    wall = x[0, 0] == 0
    logits = model(x, 8)
    assert (logits[:, wall] == -20.0).all(), "wall logits are not exactly -20"


def test_wallaware_all_wall_grid_residual_flux_is_zero():
    # All-wall input: total mass is zero, residual_flux must be 0.0 (guarded),
    # not a ZeroDivisionError or NaN.
    torch.manual_seed(0)
    model = LadderCFNet()
    wall = np.zeros((9, 9), np.float32)
    x = torch.from_numpy(np.stack([wall, wall, wall]))[None]
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


def test_wallaware_no_leak_across_wall():
    # Start position moves within the left room: nothing to the right of the
    # full-height wall may change. The rate profile weights are drawn
    # (initialisation has them exactly zero) so the probe is sensitive.
    size = 9
    xa, xb = _split_room_x((2, 2)), _split_room_x((size - 3, 2))
    right = slice(size // 2 + 1, None)  # logits columns strictly right of wall
    torch.manual_seed(7)
    walled = LadderCFNet(wall_aware=True, cross_kernel=True).eval()
    plain = LadderCFNet(wall_aware=False, cross_kernel=False).eval()
    with torch.no_grad():
        for model in (walled, plain):
            model.rate_net[2].weight[: model.hid * model.K].normal_(0.0, 0.15)
        leak_walled = (walled(xa, 16) - walled(xb, 16))[:, :, right]
        leak_plain = (plain(xa, 16) - plain(xb, 16))[:, :, right]
    assert leak_walled.abs().max().item() < 1e-7, "start leaked across the wall"
    assert leak_plain.abs().max().item() > 1e-6, \
        "plain ladder should leak across the wall (test sensitivity check)"


# ----------------------------------------------------------------- 6. gradient reach
@pytest.mark.parametrize("model_name", list(MODEL_SPECS))
def test_backward_reaches_every_parameter_and_input(model_name):
    # One backward pass from the summed logits reaches every parameter and
    # the input, finite and non-zero. Extended to the heads: every output
    # channel of the rate net's last convolution (every g slice and the mu
    # head in the detailed-balance arm; every a and b head in the free-rates
    # arm) has a non-zero weight and bias gradient.
    torch.manual_seed(0)
    model = LadderCFNet(**MODEL_SPECS[model_name])
    x = synthetic_x(seed=3).requires_grad_(True)
    model(x, 8).square().sum().backward()
    assert x.grad is not None, "no grad on input x"
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        assert p.grad.abs().sum() > 0, f"zero grad on {name}"
    last = model.rate_net[2]
    per_channel = last.weight.grad.abs().flatten(1).sum(1)
    assert (per_channel > 0).all(), \
        "rate head output channels with zero weight grad"
    assert (last.bias.grad != 0).all(), "rate head bias channels with zero grad"
    assert (model.encoder.weight.grad != 0).any()
    assert (model.readout.weight.grad != 0).any()


# ------------------------------------------------------------- 7. two timescales
def test_rate_refresh_every_r_s_steps():
    # The rates are recomputed every r_s outer steps and frozen in between:
    # forward must equal a manual rollout that refreshes exactly on that
    # schedule, and must differ from rollouts that refresh at other cadences.
    torch.manual_seed(5)
    x = random_grid_x(seed=9)
    with torch.no_grad():
        # r_s = 2 refreshes at outer steps 1 and 3 (T = 4)
        model = LadderCFNet(r_s=2)
        c = model.initial_state(x)
        rates = model.refresh_rates(x, c)
        c, _ = model.fast_phase(c, rates)
        c, _ = model.fast_phase(c, rates)
        rates = model.refresh_rates(x, c)
        c, _ = model.fast_phase(c, rates)
        c, _ = model.fast_phase(c, rates)
        manual = model.read(x, c)
        torch.testing.assert_close(model(x, 4), manual, atol=0.0, rtol=0)

        # r_s = 1 refreshes before every outer step
        model1 = LadderCFNet(r_s=1)
        c = model1.initial_state(x)
        for _ in range(4):
            rates = model1.refresh_rates(x, c)
            c, _ = model1.fast_phase(c, rates)
        manual1 = model1.read(x, c)
        torch.testing.assert_close(model1(x, 4), manual1, atol=0.0, rtol=0)

        # r_s larger than the rollout: one refresh only, then frozen
        model_f = LadderCFNet(r_s=64)
        c = model_f.initial_state(x)
        rates = model_f.refresh_rates(x, c)
        for _ in range(8):
            c, _ = model_f.fast_phase(c, rates)
        manual_f = model_f.read(x, c)
        torch.testing.assert_close(model_f(x, 8), manual_f, atol=0.0, rtol=0)

        # and the cadence genuinely matters (a different refresh schedule
        # gives a different rollout)
        d = (model1(x, 8) - model(x, 8)).abs().max().item()
        assert d > 1e-4, f"r_s=1 and r_s=2 rollouts are identical ({d})"


def test_slow_phase_frozen_between_refreshes():
    # Within a fast phase the chemistry runs under the same frozen rates:
    # the outer step equals n_f consecutive micro-steps on one rate pair.
    torch.manual_seed(6)
    model = LadderCFNet()
    x = random_grid_x(seed=12)
    with torch.no_grad():
        c = model.initial_state(x)
        rates = model.refresh_rates(x, c)
        c2, Js = model.fast_phase(c, rates)
        c1, J1 = model.micro_step(c, rates)
        c1b, J2 = model.micro_step(c1, rates)
        torch.testing.assert_close(c2, c1b, atol=0.0, rtol=0)
        assert len(Js) == model.n_f


# ------------------------------------------------------ 8. random start + equivalence
def test_stochastic_start_keeps_mass_positivity_and_only_first_two_rungs():
    torch.manual_seed(0)
    model = LadderCFNet()
    x = random_grid_x(seed=8)
    M_det = model.mass(model.initial_state(x))
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(456)
    c_a = model.initial_state(x, stochastic=True, generator=g1)
    c_b = model.initial_state(x, stochastic=True, generator=g2)
    for c in (c_a, c_b):
        torch.testing.assert_close(model.mass(c), M_det, atol=1e-6, rtol=0)
        assert c.min().item() >= 0
        assert c[:, :, 2:].abs().max().item() == 0.0  # upper rungs start empty
    assert (c_a - c_b).abs().max().item() > 1e-3, \
        "two generators gave identical states"


@pytest.mark.parametrize("model_name", list(MODEL_SPECS))
def test_stochastic_start_conserves_mass_over_rollout(model_name):
    torch.manual_seed(0)
    model = LadderCFNet(**MODEL_SPECS[model_name])
    x = random_grid_x(seed=8)
    gen = torch.Generator().manual_seed(2024)
    c0 = model.initial_state(x, stochastic=True, generator=gen)
    M0 = model.mass(c0).detach()
    scale = max(1.0, M0.max().item())
    model.eval()
    with torch.no_grad():
        c = c0
        rates = None
        for t in range(1, 65):
            if rates is None or (t - 1) % model.r_s == 0:
                rates = model.refresh_rates(x, c)
            c, _ = model.fast_phase(c, rates)
            drift = (model.mass(c) - M0).abs().max().item() / scale
            assert drift < 1e-5, f"stochastic start: mass drift {drift} at t={t}"
            assert c.min().item() >= NEG_EPS


@pytest.mark.parametrize("T", [1, 4, 8, 24])
def test_logits_at_matches_forward(T):
    torch.manual_seed(0)
    model = LadderCFNet()
    x = synthetic_x(seed=7)
    outs = model.logits_at(x, [1, 4, 8, 24])
    torch.testing.assert_close(outs[T], model(x, T), atol=1e-6, rtol=0)


def test_logits_at_records_the_spec_trace():
    torch.manual_seed(0)
    model = LadderCFNet()
    x = synthetic_x(seed=2)
    _, trace = model.logits_at(x, Ts=[4], record=True)
    assert list(sorted(trace.keys())) == ["J", "M", "V", "V_micro", "c",
                                          "residual_flux"]
    assert len(trace["c"]) == 4
    assert len(trace["V_micro"]) == 4 * model.n_f  # every micro-step
    assert trace["V"][0] == trace["V_micro"][model.n_f - 1]
    assert all(isinstance(v, float) for v in trace["V"])


# ------------------------------------------------------------- 9. parameter counts
def test_param_counts():
    # BD-8 (default): encoder 3x3 -> 8, rate net 35 -> 8 -> 40, readout 1x1.
    assert sum(p.numel() for p in LadderCFNet().parameters()) == 5705
    assert sum(p.numel() for p in LadderCFNet(detailed_balance=False)
               .parameters()) == 6289
    # matched-budget arm BD-12 and the BD-16 variant of the spec table
    assert sum(p.numel() for p in LadderCFNet(hid=12).parameters()) == 12445
    assert sum(p.numel() for p in LadderCFNet(hid=16).parameters()) == 21777
