"""CFNet — Stage-A Coagulation-Fragmentation looped processor.

Reversible reaction 2 A_1 <=> A_2 with mass M = c1 + 2 c2 conserved exactly at
every step (any flux magnitudes); the optional 1/2 flux bound additionally
guarantees non-negativity of c1, c2.

Shape table
-----------
x                  (B, 3, H, W)    input channels [walkable, start, goal]
c1, c2             (B, hid, H, W)  monomer / cluster concentrations
M = c1 + 2 c2      (B, hid, H, W)  conserved mass
m, s               (B, hid, H, W)  merge / split fluxes
h (rate-net hidden)(B, hid, H, W)
logits             (B, H, W)       per-cell path logits
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SPECS = {
    "CF-Full": dict(fragmentation=True),
    "CF-NoFrag": dict(fragmentation=False),
    "CF-NoBound": dict(bound_flux=False),
    "CF-WallAware": dict(wall_aware=True),
}


class CFNet(nn.Module):
    def __init__(self, in_ch=3, hid=16, fragmentation=True, bound_flux=True,
                 wall_aware=False, cross_kernel=False,
                 cross_kernel_encoder=False, damping=False):
        super().__init__()
        self.fragmentation = fragmentation
        self.bound_flux = bound_flux
        self.wall_aware = wall_aware
        self.cross_kernel = cross_kernel
        self.cross_kernel_encoder = cross_kernel_encoder
        self.damping = damping
        self.encoder = nn.Conv2d(in_ch, hid, 3, padding=1)
        self.rate_net = nn.Sequential(
            nn.Conv2d(in_ch + 2 * hid, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, 2 * hid, 3, padding=1),
        )
        self.readout = nn.Conv2d(2 * hid, 1, 1)
        if damping:
            self.damp_logit = nn.Parameter(torch.zeros(()))
        if cross_kernel:
            self.register_buffer("kernel_mask",
                                 torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]))

    # -- methods added below --

    def _conv3(self, conv, x, mask=True):
        """3x3 conv, optionally restricted to the 4-neighbourhood cross kernel.

        With cross_kernel=True the cross-shaped mask [[0,1,0],[1,1,1],[0,1,0]]
        is applied to the rate net's two 3x3 convolutions only (packet 7.5);
        `mask` is False for every conv that is not cross-masked. Set
        cross_kernel_encoder=True to additionally mask the encoder conv in
        initial_state — this restores the pre-7.5 behaviour (all 3x3
        convolutions masked) and has no effect when cross_kernel is False.
        """
        if not (self.cross_kernel and mask):
            return conv(x)
        w = conv.weight * self.kernel_mask.unsqueeze(0).unsqueeze(0)
        return F.conv2d(x, w, conv.bias, padding=1)

    def initial_state(self, x, stochastic=False, generator=None):
        # c1(0) = softplus(encoder(x)); c2(0) = 0
        M = F.softplus(self._conv3(self.encoder, x,
                                   mask=self.cross_kernel_encoder))
        if self.wall_aware:
            M = M * x[:, 0:1]                       # no mass inside walls
        if not stochastic:
            return M, torch.zeros_like(M)
        # Draw u on the generator's own device (a CPU generator works even
        # when the model lives on another device, e.g. CUDA — torch.rand with
        # device=M.device and a CPU generator would raise). Without a
        # generator, draw directly on the parameter device.
        dev = generator.device if generator is not None else M.device
        u = torch.rand(M.shape, generator=generator, device=dev, dtype=M.dtype)
        u = 0.5 * u.to(M.device)                  # u ~ U(0, 0.5)
        return (1.0 - u) * M, 0.5 * u * M         # c1, c2 (mass split, M kept)

    def fluxes(self, x, c1, c2):
        # h = rate_net([x, c1, c2]); heads raw_merge | raw_split
        h = self._conv3(self.rate_net[0], torch.cat([x, c1, c2], dim=1))
        h = self.rate_net[1](h)
        if self.wall_aware:
            h = h * x[:, 0:1]                        # gate hidden activations by free cell
        raw = self._conv3(self.rate_net[2], h)
        raw_merge, raw_split = raw.chunk(2, dim=1)
        factor = 0.5 if self.bound_flux else 1.0
        m = factor * c1 * torch.sigmoid(raw_merge)   # bounded by available mass
        s = c2 * torch.sigmoid(raw_split) if self.fragmentation else torch.zeros_like(c2)
        return m, s

    def step(self, x, c1, c2):
        # m_t = (1/2)*c1*sigmoid(raw_merge)  (or c1*sigmoid if not bound_flux)
        # s_t = c2*sigmoid(raw_split)       (0 if not fragmentation)
        m, s = self.fluxes(x, c1, c2)
        # g: damping factor, 1 unless damping (g = sigmoid(damp_logit)).
        # The fluxes are scaled by g BEFORE the state update, so the returned
        # m, s are the APPLIED fluxes the state actually changed by.
        g = 1.0
        if self.damping:
            g = torch.sigmoid(self.damp_logit)
            m = g * m
            s = g * s
        # c1 <- c1 + (-2 m + 2 s)  (m, s = applied fluxes)
        c1 = c1 + (-2.0 * m + 2.0 * s)
        # c2 <- c2 + (m - s)
        c2 = c2 + (m - s)
        return c1, c2, m, s, g

    def read(self, x, c1, c2):
        logits = self.readout(torch.cat([c1, c2], dim=1)).squeeze(1)
        if self.wall_aware:
            logits = logits.masked_fill(x[:, 0] == 0, -20.0)  # walls impossible
        return logits

    def forward(self, x, T):
        # (c1, c2) = (softplus(encoder(x)), 0)                     — initial state
        c1, c2 = self.initial_state(x)
        for _ in range(T):
            # (m, s) = fluxes from rate_net([x, c1, c2])
            # c1 <- c1 + (-2 m + 2 s);  c2 <- c2 + (m - s)          — reaction step
            c1, c2, _, _, _ = self.step(x, c1, c2)
        # logits = readout([c1, c2])                                — readout
        return self.read(x, c1, c2)

    def logits_at(self, x, Ts, record=False):
        """Read out logits after each step count in `Ts`, from ONE shared rollout.

        With record=True the returned trace holds per-step c1, c2, M, the
        APPLIED fluxes m, s (already scaled by the damping factor g) and g
        itself, plus a `residual_flux` list: sum|g*m - g*s| / sum M per step.
        The trace starts AFTER step 1: t=0 (the initial state) is not
        included; get it from `initial_state(x)`.
        """
        Ts = sorted(set(Ts))
        c1, c2 = self.initial_state(x)
        out, trace, residual = {}, self._new_trace() if record else None, []
        t = 0
        for T in Ts:
            for _ in range(T - t):
                c1, c2, m, s, g = self.step(x, c1, c2)
                t += 1
                if record:
                    self._record(trace, residual, c1, c2, m, s, g)
            out[T] = self.read(x, c1, c2)
        if record:
            trace["residual_flux"] = residual
            return out, trace
        return out

    @staticmethod
    def _new_trace():
        return {"c1": [], "c2": [], "M": [], "m": [], "s": [], "g": []}

    @staticmethod
    def _record(trace, residual, c1, c2, m, s, g):
        for key, val in (("c1", c1), ("c2", c2), ("m", m), ("s", s)):
            trace[key].append(val.detach().cpu())
        M = c1 + 2.0 * c2
        trace["M"].append(M.detach().cpu())
        trace["g"].append(float(g.detach()) if torch.is_tensor(g) else float(g))
        M_total = M.sum().item()
        # all-wall grid: total mass is zero, residual is 0.0 by convention
        residual.append((m - s).abs().sum().item() / M_total
                        if M_total > 0.0 else 0.0)
