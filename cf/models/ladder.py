"""LadderCFNet -- Family B, the Becker-Doring commitment ladder.

Implementation of the specification card `docs/specs/family_b_ladder.md`,
following the conventions of `cf/models/cf.py` (the `forward(x, T)` /
`logits_at(Ts, record)` interface, the wall-aware and cross-kernel options,
the traced rollout).

The physics, in plain words
---------------------------
The Stage-A model gives every cell two states: loose evidence and committed
evidence, with nothing in between. This family gives every cell a small
ladder of states instead -- rung 1 is "no idea", the top rung is "certainly
on the path". A cell climbs one rung at a time when its own bottom rung
absorbs a loose piece of evidence (a monomer merge) and steps down one rung
at a time when it sheds a piece back. The total evidence a cell holds never
changes -- mass only redistributes across the rungs. A small network looks
at the cell's neighbourhood and sets how easy climbing is (the mobility)
and how the rungs compare (the energy profile); everything else -- the
stoichiometry, the conservation, the positivity cap, the joint bound -- is
fixed chemistry, not learned. Because the rates satisfy detailed balance,
the ladder has a genuine free energy: a number the chemistry always
decreases in continuous time while the rates are held fixed, the way a
ball rolls downhill and stops. The rates are refreshed only every `r_s` outer steps (the slow
phase); in between the chemistry relaxes under frozen rates in `n_f` micro
steps (the fast phase). That split is the "confidence with hysteresis":
what resists one weak signal is the frozen energy landscape, and sustained
contrary evidence is needed to move the equilibrium itself.

Order of one outer step (the spec "Discrete-time update"):
1. (slow, every `r_s` steps) rates from the rate network reading [x, c];
2. (fast, `n_f` micro-steps under frozen rates)
   - raw requests per rung: merge r_k = a_k c_1 c_k, shed d_k = b_{k+1} c_{k+1}
     (A_1 + A_k <-> A_{k+1}: only monomer moves, one rung at a time);
   - per-rung total requested outflow O_j (rung 1 donates both monomers of
     an A_1 + A_1 merge, hence the factor 2);
   - ONE joint cap lambda for the whole cell: every rung keeps at least a
     fraction (1 - zeta) of its stock, whatever the network asks for;
   - applied fluxes J_k = lambda (r_k - d_k) and the rung balances.
Truncation: J_K = 0 -- no flux past the top rung (K = 4 rungs by default).

Detailed balance (the default arm): the network outputs a per-rung energy
profile g; with the gauge Q_1 = 1 the equilibrium weights are
Q_k = exp(g_1 - g_k) (that is Q_k = exp(-g_k) in the gauge g_1 = 0), the
climb rate is the mobility a_k = mu, and the shed rate is
b_{k+1} = mu Q_k / Q_{k+1}, so a_k Q_1 Q_k = b_{k+1} Q_{k+1} holds for any
weights. The state c_k = Q_k z^k (any activity z > 0) then has zero net
flux on every rung, and V = sum_k c_k (log(c_k/Q_k) - 1) is the Lyapunov
free energy. The free-rates arm (BD-FreeRates, the ablation) drops the
(g, mu) parameterization for direct softplus rate heads a_k, b_{k+1}:
every discrete-time part is unchanged; the equilibrium weights needed by
V are reconstructed from the rates by the same recursion
Q_1 = 1, Q_{k+1} = a_k Q_1 Q_k / b_{k+1}.

Shape table (B x H x W grid; hid independent ladders per cell, K rungs)
-----------------------------------------------------------------------
x        (B, 3, H, W)         input channels [walkable, start, goal]
c        (B, hid, K, H, W)     rung concentrations; flattened to
                              (B, hid*K, H, W) rung-major [c_1..c_K] per
                              channel for every convolutional call
M        (B, hid, H, W)       conserved mass M = sum_k k c_k per cell/channel
J        (B, hid, K-1, H, W)  net upward rung fluxes
g        (B, hid, K, H, W)    per-rung energy profile (rate-net head)
mu       (B, hid, H, W)       mobility (rate-net head)
logits   (B, H, W)            per-cell path logits; exactly -20.0 in walls

Constants (fixed, not learned): K = 4 rungs, hid = 8 ladders, zeta = 0.5
joint flux cap, eps = 1e-12 request floor, g_max = 30 energy-profile clamp
(keeps every Q_k = exp(g_1 - g_k) and every rate ratio exp(g_{k+1} - g_k)
inside float32 range however steep the learned profile gets), r_s = 2 slow
refresh, n_f = 2 fast micro-steps, T_train = 8 outer steps.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SPECS = {
    "BD-Full": dict(detailed_balance=True),
    "BD-FreeRates": dict(detailed_balance=False),
}


class LadderCFNet(nn.Module):
    def __init__(self, in_ch=3, hid=8, K=4, detailed_balance=True,
                 wall_aware=True, cross_kernel=False,
                 zeta=0.5, eps=1e-12, r_s=2, n_f=2, g_max=30.0):
        super().__init__()
        self.hid, self.K = hid, K
        self.detailed_balance = detailed_balance
        self.wall_aware = wall_aware
        self.cross_kernel = cross_kernel
        self.zeta, self.eps = zeta, eps
        self.g_max = g_max
        self.r_s, self.n_f = r_s, n_f
        # energy profile g (hid*K channels) + raw mobility (hid channels)
        # in the detailed-balance arm; 2*(K-1)*hid direct rate heads a, b
        # in the free-rates arm.
        out_ch = (K + 1) * hid if detailed_balance else 2 * (K - 1) * hid
        self.encoder = nn.Conv2d(in_ch, hid, 3, padding=1)
        self.rate_net = nn.Sequential(
            nn.Conv2d(in_ch + hid * K, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, out_ch, 3, padding=1),
        )
        self.readout = nn.Conv2d(hid * K, 1, 1)
        if detailed_balance:
            # g starts identically zero: every Q_k = 1, the flat reference
            # ladder. The mu head starts with bias log(e - 1), whose softplus
            # is 1, so the mobility starts at mu ~ 1.
            with torch.no_grad():
                nn.init.zeros_(self.rate_net[2].weight[: hid * K])
                nn.init.zeros_(self.rate_net[2].bias[: hid * K])
                self.rate_net[2].bias[hid * K:].fill_(math.log(math.e - 1.0))
        if cross_kernel:
            self.register_buffer("kernel_mask",
                                 torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]))

    # ------------------------------------------------------------ plumbing

    def _conv3(self, conv, x, mask=True):
        """3x3 conv, optionally restricted to the 4-neighbourhood cross kernel.

        With cross_kernel=True the cross-shaped mask [[0,1,0],[1,1,1],[0,1,0]]
        is applied to the rate net's two 3x3 convolutions, exactly as in
        `cf/models/cf.py`; `mask` is False for every conv that is not
        cross-masked.
        """
        if not (self.cross_kernel and mask):
            return conv(x)
        w = conv.weight * self.kernel_mask.unsqueeze(0).unsqueeze(0)
        return F.conv2d(x, w, conv.bias, padding=1)

    def _flat(self, c):
        """(B, hid, K, H, W) -> (B, hid*K, H, W), rung-major [c_1..c_K] per channel."""
        B, hid, K, H, W = c.shape
        return c.reshape(B, hid * K, H, W)

    # --------------------------------------------------------------- state

    def mass(self, c):
        """M = sum_k k c_k per cell and channel -- the conserved evidence budget."""
        k = torch.arange(1, self.K + 1, device=c.device, dtype=c.dtype)
        return (c * k.view(1, 1, -1, 1, 1)).sum(dim=2)

    def initial_state(self, x, stochastic=False, generator=None):
        """M = softplus(encoder(x)); all mass starts on rung 1 (c_1 = M).

        The wall-aware option multiplies M by the walkable channel so walls
        hold exactly zero state. The random-start variant draws u ~ U(0, 0.5)
        per cell and channel and splits the mass over the first two rungs,
        c_2 = u M / 2 and c_1 = (1 - u) M, which keeps M = c_1 + 2 c_2 exactly.
        """
        M = F.softplus(self._conv3(self.encoder, x, mask=False))
        if self.wall_aware:
            M = M * x[:, 0:1]                       # no mass inside walls
        empty = torch.zeros_like(M)
        if not stochastic:
            return torch.stack([M] + [empty] * (self.K - 1), dim=2)
        # Draw u on the generator's own device (a CPU generator works even
        # when the model lives elsewhere); without one, draw on the
        # parameter device -- the same convention as cf.models.cf.
        dev = generator.device if generator is not None else M.device
        u = torch.rand(M.shape, generator=generator, device=dev, dtype=M.dtype)
        u = 0.5 * u.to(M.device)                    # u ~ U(0, 0.5)
        return torch.stack([(1.0 - u) * M, 0.5 * u * M]
                           + [empty] * (self.K - 2), dim=2)

    # ---------------------------------------------------------- slow phase

    def refresh_rates(self, x, c):
        """Slow phase: the rate network reads [x, rung stack] and outputs the
        rate pair (a, b), frozen until the next refresh.

        Detailed-balance arm: a_k = mu (the same mobility on every rung) and
        b_{k+1} = mu Q_k / Q_{k+1} with Q_k = exp(g_1 - g_k), so detailed
        balance a_k Q_1 Q_k = b_{k+1} Q_{k+1} holds for any weights and the
        gauge Q_1 = 1 is exact. The profile g is clamped to [-g_max, g_max]
        and b is formed in log space, b_{k+1} = mu exp(g_{k+1} - g_k):
        materialising Q first overflows exp to inf (or underflows to 0) on
        profiles steeper than ~88 nats, and the division Q_k / Q_{k+1}
        then yields inf/inf or 0/0 -- NaN rates that poison the rollout
        through the flux cap. Free-rates arm: a_k and b_{k+1} are direct
        softplus heads (no constraint, hence no energy guarantee).
        """
        B, hid, K, H, W = c.shape
        h = self._conv3(self.rate_net[0], torch.cat([x, self._flat(c)], dim=1))
        h = self.rate_net[1](h)
        if self.wall_aware:
            h = h * x[:, 0:1]                       # gate hidden activations by free cell
        raw = self._conv3(self.rate_net[2], h)
        if self.detailed_balance:
            g = raw[:, : hid * K].reshape(B, hid, K, H, W)
            g = torch.clamp(g, min=-self.g_max, max=self.g_max)
            mu = F.softplus(raw[:, hid * K:])       # (B, hid, H, W) mobility
            a = mu.unsqueeze(2).expand(B, hid, K - 1, H, W)
            b = mu.unsqueeze(2) * torch.exp(g[:, :, 1:] - g[:, :, :-1])
            return a, b
        a = F.softplus(raw[:, : (K - 1) * hid]).reshape(B, hid, K - 1, H, W)
        b = F.softplus(raw[:, (K - 1) * hid:]).reshape(B, hid, K - 1, H, W)
        return a, b

    def rung_weights(self, rates):
        """Equilibrium weights Q from the rate pair (a, b) by the detailed-balance
        recursion Q_1 = 1, Q_{k+1} = a_k Q_1 Q_k / b_{k+1}.

        For the detailed-balance arm this recovers Q = exp(g_1 - g_k); for
        the free-rates arm it is the reconstruction that keeps the free-energy
        formula V = sum_k c_k (log(c_k/Q_k) - 1) well-defined.
        """
        a, b = rates
        Qs = [torch.ones_like(a[:, :, 0])]
        for i in range(self.K - 1):
            Qs.append(Qs[-1] * a[:, :, i] / b[:, :, i])
        return torch.stack(Qs, dim=2)

    def free_energy(self, c, rates):
        """V = sum_k c_k (log(c_k/Q_k) - 1) over all rungs, cells and channels.

        The detailed-balance Lyapunov function: it decreases in continuous
        time under frozen rates. Numerically safe in float32:
        - log Q is accumulated in log space, never taken of a materialized
          Q. rung_weights chains the product Q_{k+1} = Q_k a_k / b_{k+1};
          at trained rate scales two consecutive a/b ratios overflow that
          chain to inf (or underflow to 0), and log Q then made V = -inf
          where c_k > 0 and NaN on empty rungs (0 * inf through the log
          difference). Here log Q_1 = 0 (gauge) and log Q_{k+1} =
          log Q_k + log a_k - log b_{k+1} -- the same recursion, so in the
          detailed-balance arm log Q_k is the (gauge g_1 = 0) profile -g_k
          itself, with no exp and no log of an exp anywhere.
        - the logs are taken of rates floored at the smallest positive
          normal, so an underflowed rate head (a or b exactly 0, seen at
          trained seeds) gives a finite log instead of log 0 = -inf and the
          0/0 the chained division produced.
        - each increment is clamped to [-2*g_max, 2*g_max], consistent
          with the g clamp (a ratio exp(g_{k+1} - g_k) spans at most that
          range); the clamp is a no-op on clamped profiles and keeps log Q
          finite for any rates, so c * log Q is never 0 * inf.
        - an empty rung contributes 0: xlogy gives 0 log 0 = 0, and the -c
          term is kept outside as before.
        """
        a, b = rates
        tiny = torch.finfo(a.dtype).tiny
        inc = torch.clamp(torch.log(torch.clamp(a, min=tiny))
                          - torch.log(torch.clamp(b, min=tiny)),
                          min=-2.0 * self.g_max, max=2.0 * self.g_max)
        logQ = [torch.zeros_like(a[:, :, 0])]
        for i in range(self.K - 1):
            logQ.append(logQ[-1] + inc[:, :, i])
        logQ = torch.stack(logQ, dim=2)
        return (torch.xlogy(c, c) - c * logQ - c).sum()

    # ---------------------------------------------------------- fast phase

    def micro_step(self, c, rates):
        """One capped chemistry micro-step under frozen rates.

        1. Requests: merge r_k = a_k c_1 c_k, shed d_k = b_{k+1} c_{k+1}
           (mass action, so a rung only ever requests outflow in proportion
           to its own stock).
        2. Per-rung total requested outflow O_j: rung 1 donates both
           monomers of A_1 + A_1 -> A_2 (factor 2), interior rungs can merge
           up or shed down, the top rung can only shed.
        3. Joint cap: lambda = min(1, zeta * min_j c_j / (O_j + eps)). A
           rung that makes no request (O_j = 0, hence c_j = 0 by mass
           action) does not constrain the cap -- otherwise the all-monomer
           start would deadlock with lambda = 0 and the cell could never
           climb its own ladder.
        4. Applied fluxes J_k = lambda (r_k - d_k) and the rung balances
           (J_0 = J_K = 0 by truncation):
           dc_1 = -2 J_1 - sum_{k>=2} J_k,   dc_k = J_{k-1} - J_k.
           Where lambda = 0 the cap grants no outflow at all, and J is
           forced to exactly 0: an overflowing request (r_k - d_k = inf)
           would otherwise turn the product into 0 * inf = NaN.

        Positivity is exact: rung j loses at most lambda O_j <= zeta c_j and
        only ever receives. Conservation is exact: sum_k k dc_k = 0
        identically in J.
        """
        a, b = rates
        B, hid, K, H, W = c.shape
        c1 = c[:, :, 0]                              # (B, hid, H, W) the monomer rung
        # r_k = a_k c_1 c_k, k = 1..K-1, with c_1 c_k formed first: the
        # left-to-right product (a c_1) c_k overflows to inf on extreme
        # rates and then inf * 0 on an empty rung is NaN, before the cap.
        r = a * (c1.unsqueeze(2) * c[:, :, :-1])
        d = b * c[:, :, 1:]                          # d_k = b_{k+1} c_{k+1}
        O = torch.cat([(2.0 * r[:, :, 0] + r[:, :, 1:].sum(dim=2)).unsqueeze(2),
                       r[:, :, 1:] + d[:, :, :-1],
                       d[:, :, -1].unsqueeze(2)], dim=2)   # (B, hid, K, H, W)
        ratio = torch.where(O > 0, self.zeta * c / (O + self.eps),
                            torch.full_like(c, float("inf")))
        lam = torch.clamp(ratio.amin(dim=2), max=1.0)      # (B, hid, H, W)
        J = lam.unsqueeze(2) * (r - d)                      # applied rung fluxes
        J = torch.where(lam.unsqueeze(2) > 0.0, J, torch.zeros_like(J))
        d1 = -2.0 * J[:, :, 0] - J[:, :, 1:].sum(dim=2)
        c_new = torch.cat([(c[:, :, 0] + d1).unsqueeze(2),
                           c[:, :, 1:-1] + (J[:, :, :-1] - J[:, :, 1:]),
                           (c[:, :, -1] + J[:, :, -1]).unsqueeze(2)], dim=2)
        return c_new, J

    def fast_phase(self, c, rates):
        """n_f micro-steps under the frozen rates; returns the state and the
        applied fluxes J of each micro-step."""
        Js = []
        for _ in range(self.n_f):
            c, J = self.micro_step(c, rates)
            Js.append(J)
        return c, Js

    def step(self, x, c, rates=None, refresh=False):
        """One outer step: refresh the rates when asked (the slow phase),
        then run the fast phase under them. Returns (c, rates, J of the
        last micro-step)."""
        if refresh or rates is None:
            rates = self.refresh_rates(x, c)
        c, Js = self.fast_phase(c, rates)
        return c, rates, Js[-1]

    # --------------------------------------------------------------- readout

    def read(self, x, c):
        """logits = Conv1x1 over the flattened rung stack, one per cell;
        exactly -20.0 in walls when the wall-aware option is on."""
        logits = self.readout(self._flat(c)).squeeze(1)
        if self.wall_aware:
            logits = logits.masked_fill(x[:, 0] == 0, -20.0)  # walls impossible
        return logits

    def forward(self, x, T):
        c = self.initial_state(x)
        rates = None
        for t in range(1, T + 1):
            c, rates, _ = self.step(x, c, rates, refresh=(t - 1) % self.r_s == 0)
        return self.read(x, c)

    def logits_at(self, x, Ts, record=False):
        """Read out logits after each step count in `Ts`, from ONE shared rollout.

        With record=True the returned trace holds, per outer step: the rung
        map c (as a stacked tensor), the mass M, the applied rung fluxes J of
        the last micro-step, the free energy V_t under the current rates, and
        the residual flux sum|J| / sum M; plus a flat `V_micro` list with V
        after every micro-step (the convergence diagnostic). The trace
        starts AFTER step 1: t = 0 (the initial state) is not included; get
        it from `initial_state(x)`. Rate refreshes follow the same `r_s`
        schedule as `forward`, so the two agree at every T.
        """
        Ts = sorted(set(Ts))
        c = self.initial_state(x)
        out, trace = {}, self._new_trace() if record else None
        rates, t = None, 0
        for T in Ts:
            for _ in range(T - t):
                if rates is None or t % self.r_s == 0:
                    rates = self.refresh_rates(x, c)
                t += 1
                if record:
                    Vs = []
                for _ in range(self.n_f):
                    c, J = self.micro_step(c, rates)
                    if record:
                        with torch.no_grad():
                            Vs.append(float(self.free_energy(c, rates)))
                if record:
                    self._record(trace, c, J, rates)
                    trace["V_micro"].extend(Vs)
            out[T] = self.read(x, c)
        if record:
            return out, trace
        return out

    @staticmethod
    def _new_trace():
        return {"c": [], "M": [], "J": [], "V": [], "V_micro": [],
                "residual_flux": []}

    def _record(self, trace, c, J, rates):
        with torch.no_grad():
            c = c.detach()
            M = self.mass(c)
            trace["c"].append(c.cpu())
            trace["M"].append(M.cpu())
            trace["J"].append(J.detach().cpu())
            trace["V"].append(float(self.free_energy(c, rates)))
        M_total = M.sum().item()
        # all-wall grid: total mass is zero, residual is 0.0 by convention
        trace["residual_flux"].append(J.abs().sum().item() / M_total
                                      if M_total > 0.0 else 0.0)
