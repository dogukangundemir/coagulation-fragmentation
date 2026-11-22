"""TransportCFNet — Family D, transport coagulation-fragmentation.

Implementation of the specification card `docs/specs/family_d_transport.md`.
The physics, in plain words: think of the maze as a network of pipes. Every
pair of adjacent free cells is a pipe; a wall is a missing pipe. Water is
poured into the start cell and drained from the goal cell, one unit per time
step, forever. A pipe that carries flow thickens; a pipe that carries nothing
thins. The total amount of pipe material never changes — material only moves
between pipes. After enough time, the only pipes with real thickness are the
ones on a route from start to goal: a dead-end corridor has no drain inside
it, so no steady flow can enter it, so its pipe thins to nothing. On a
maze without loops there is exactly one route, and the physics alone marks
the solution path. This is the slime-mould *Physarum* mechanism.

Precisely: every free edge *e* between neighbouring cells u, v is a
resistor of conductance **C_e = D_e / L_e** (thickness D_e, learned length
L_e) carrying signed flow **Q_e = C_e (p_u − p_v)** against a per-cell
potential **p**. The potentials solve the network equation — net outflow
+I at the start cell, −I at the goal cell, zero everywhere else
(Kirchhoff). Flow-carrying edges then gather material and empty edges shed
it, via the normalized redistribution

    D_e ← M_tube · D_e · f_e / Σ_{e'} D_{e'} f_{e'}     (spec "Update" step 3)

so the tube material M_tube = Σ_e D_e is conserved exactly. The layer's
relaxation minimizes the potential energy Ψ = ½ Σ_e C_e Δp_e² − Σ_u b_u p_u
of the resistor network (Δp_e = p_u − p_v), whose minimum is the steady
Kirchhoff flow; with unit lengths L ≡ 1 this is the classic dissipation
E = Σ_e Q_e²/D_e minimized over conserved flows (Thomson's principle).

Two arms (spec experiment X2): the *learned* arm learns edge lengths L_e,
the adaptation f_theta and a 1×1 readout (~200 parameters); the *physics
only* arm sets L ≡ 1, f_e = |Q_e| + eps_f and reads the path off the fixed
threshold |Q|_cell ≥ I/2 with zero learned parameters. The physics itself —
solves, conservation, wall structure — is never parameterized.

Shape table (B mazes of H x W cells; H = W = 2n+1 on the curriculum):

    x        (B, 3, H, W)    input channels [walkable, start one-hot, goal one-hot]
    walk     (B, H, W)       x[:, 0]: 1 free cell, 0 wall
    b        (B, H, W)       source field: +I on the start cell, −I on the goal cell
    Eh, Ev   (B, H, W−1), (B, H−1, W)  free-edge masks (both endpoints free)
    Dh, Dv   same shapes     edge thicknesses D_e (conductances); exact 0 on walls
    Lh, Lv   same shapes     learned edge lengths L_e (softplus head; ≥ 0)
    p        (B, H, W)       per-cell potential; fixed to 0 in wall cells
    Qh, Qv   same shapes     signed flows: Qh positive = eastward (from (r, c) to
                             (r, c+1)), Qv positive = southward
    qcell    (B, H, W)       |Q|_cell: Σ |Q_e| over the ≤ 4 incident edges
    dcell    (B, H, W)       D_cell: mean conductance of the incident free edges
    logits   (B, H, W)       per-cell path logits; exactly −20.0 in walls

One outer step runs the spec "Update" in order: (1) relax the potentials —
exactly (batched float64 linear solve, the default) or with K_rel
two-colour Gauss–Seidel sweeps — (2) compute the flows, (3) redistribute the
tube material, (4) re-impose the wall structure. The solve, flows and
divergence are computed in float64 and cast to float32 only for the material
step and the readout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus(LOG_E_MINUS_1) = log(1 + (e − 1)) = 1.0, so a head biased here and
# zero-initialised in its weights starts every output at exactly 1.0.
LOG_E_MINUS_1 = 0.5413248546129181

WALL_LOGIT = -20.0    # wall-aware logit convention of cf/models/cf.py
PATH_LOGIT = 20.0     # the physics-only arm emits ±20 logits (sigmoid=1/0)


class TransportCFNet(nn.Module):
    """Family D transport model with the CFNet interface.

    Constants (fixed, not learned — spec "Discrete-time update with all
    constants"): current I = 1.0 (unit mass per step), K_rel = 16 sweeps per
    outer step in sweep mode, eps_d = 1e-6 (denominator guard, applied only
    where a cell's conductance sum is zero), eps_f = 1e-3 (adaptation floor:
    a thinned edge keeps a tiny growth term and gradients keep flowing),
    M_tube = the number of free edges of the maze (so the uniform initial
    thickness is d_0 = 1.0), sigma_tie = 0.1 (random-start variant: the
    initial draw is normalized by its own sum, so Σ_e D_e(0) = M_tube
    exactly).

    Args: physics_only selects the X2 physics arm (zero learned parameters);
    relaxation is "exact" (default on the curriculum grids) or "sweep";
    k_rel is the relaxation budget in sweeps per outer step.
    """

    def __init__(self, physics_only=False, relaxation="exact", k_rel=16,
                 current=1.0, eps_d=1e-6, eps_f=1e-3, tie_noise=0.1,
                 solve_chunk=100):
        super().__init__()
        if relaxation not in ("exact", "sweep"):
            raise ValueError("relaxation must be 'exact' or 'sweep', got %r"
                             % (relaxation,))
        self.physics_only = physics_only
        self.relaxation = relaxation
        self.k_rel = int(k_rel)
        self.I = float(current)              # source current I
        self.eps_d = float(eps_d)            # denominator guard, isolated cells only
        self.eps_f = float(eps_f)            # adaptation floor
        self.sigma_tie = float(tie_noise)    # random-start initial thickness
        self.solve_chunk = int(solve_chunk)
        if not physics_only:
            # Learned, strictly local parts (spec "What is learned and what is
            # fixed"). Edge features h and per-edge lengths L_e = softplus of a
            # 3x3 head over h; the head's weights start at zero and its bias at
            # log(e − 1), so every L_e starts at exactly 1.0.
            self.encoder = nn.Conv2d(3, 4, 3, padding=1)          # 112 params
            self.length_head = nn.Conv2d(4, 2, 3, padding=1)       # 74 params
            nn.init.zeros_(self.length_head.weight)
            nn.init.constant_(self.length_head.bias, LOG_E_MINUS_1)
            # Adaptation f_theta: f_e = softplus(w_q |Q_e| + w_u . u_e + b_f) + eps_f,
            # with w_q = softplus(raw_wq) >= 0 so f is non-decreasing in the flow.
            self.raw_wq = nn.Parameter(torch.tensor(LOG_E_MINUS_1))  # -> w_q = 1.0
            # w_u starts small but positive: if it (and the length-head weights)
            # both started at 0, the encoder would receive exactly zero gradient
            # at initialisation and could never leave that state.
            self.w_u = nn.Parameter(torch.full((4,), 0.05))
            self.b_f = nn.Parameter(torch.zeros(()))
            # Readout: 1x1 conv over [x, |Q|_cell, p, D_cell] -> per-cell logits.
            self.readout = nn.Conv2d(6, 1, 1)                     # 7 params

    # ------------------------------------------------------------------ geometry

    def edge_masks(self, x):
        """Free-edge masks: Eh (B, H, W−1) between (r, c) and (r, c+1);
        Ev (B, H−1, W) between (r, c) and (r+1, c); 1 iff both cells are free."""
        walk = x[:, 0]
        Eh = walk[:, :, :-1] * walk[:, :, 1:]
        Ev = walk[:, :-1, :] * walk[:, 1:, :]
        return Eh, Ev

    def edge_features(self, x):
        """Learned edge quantities of the learned arm.

        h = Conv3x3(3 -> 4)(x) is the only learned conv before the readout;
        u_e is the mean of h over the edge's two endpoint cells:
            uh (B, 4, H, W−1), uv (B, 4, H−1, W)
        and L_e = softplus of a zero-weight, log(e−1)-biased 3x3 head over h,
        endpoint-averaged the same way (starts at exactly 1.0 everywhere):
            Lh (B, H, W−1), Lv (B, H−1, W), both >= softplus(0) > 0.
        """
        h = self.encoder(x)                                     # (B, 4, H, W)
        uh = 0.5 * (h[:, :, :, :-1] + h[:, :, :, 1:])           # (B, 4, H, W-1)
        uv = 0.5 * (h[:, :, :-1, :] + h[:, :, 1:, :])           # (B, 4, H-1, W)
        lcell = F.softplus(self.length_head(h))                 # (B, 2, H, W)
        Lh = 0.5 * (lcell[:, 0, :, :-1] + lcell[:, 0, :, 1:])    # (B, H, W-1)
        Lv = 0.5 * (lcell[:, 1, :-1, :] + lcell[:, 1, 1:, :])   # (B, H-1, W)
        return uh, uv, Lh, Lv

    # -------------------------------------------------------------------- state

    def initial_state(self, x, stochastic=False, generator=None):
        """Initial (Dh, Dv, p).

        Deterministic: D_e = d_0 = 1.0 on every free edge, 0 on wall edges,
        so Σ_e D_e(0) = M_tube = the per-maze free-edge count exactly, and
        p starts at 0 (the sweep mode relaxes from this zero potential).
        Stochastic (tie-break variant): D_e(0) = M_tube · exp(sigma_tie u_e)
        / Σ exp(sigma_tie u_e') over free edges, u_e ~ U(−1, 1) — the draw is
        normalized by its own sum, so Σ_e D_e(0) = M_tube exactly here too.
        A generator on a different device than x is honoured: the draw runs
        on the generator's device (torch.rand requires them to match) and is
        moved to x, so a CPU generator with a CUDA batch works.
        """
        Eh, Ev = self.edge_masks(x)
        if not stochastic:
            Dh = Eh.clone()                                     # d_0 = 1.0
            Dv = Ev.clone()
        else:
            gdev = generator.device if generator is not None else x.device
            gh = 2.0 * torch.rand(Eh.shape, generator=generator, device=gdev) - 1.0
            gv = 2.0 * torch.rand(Ev.shape, generator=generator, device=gdev) - 1.0
            if gdev != x.device:
                gh, gv = gh.to(x.device), gv.to(x.device)
            wh = torch.exp(self.sigma_tie * gh) * Eh           # free edges only
            wv = torch.exp(self.sigma_tie * gv) * Ev
            Dh, Dv = self._normalize_material(Eh, Ev, wh, wv)
        p = torch.zeros_like(x[:, 0])                          # walls stay at 0
        return Dh, Dv, p

    def _normalize_material(self, Eh, Ev, wh, wv):
        """D = M_tube · w / Σ w over each maze's free edges (wh, wv >= 0),
        guarded for the all-wall grid: no free edges -> D stays 0."""
        m_tube = Eh.sum(dim=(1, 2)) + Ev.sum(dim=(1, 2))        # (B,)
        denom = wh.sum(dim=(1, 2)) + wv.sum(dim=(1, 2))         # (B,)
        ok = denom > 0
        scale = torch.where(ok, m_tube / denom, torch.zeros_like(denom))
        Dh = torch.where(ok[:, None, None], wh * scale[:, None, None], wh)
        Dv = torch.where(ok[:, None, None], wv * scale[:, None, None], wv)
        return Dh * Eh, Dv * Ev

    # -------------------------------------------------------------- relaxation

    def _solve_potentials(self, walk, b, Ch, Cv):
        """Spec "Update" step 1, exact mode: solve A p = b per maze, in float64,
        in batch chunks of `solve_chunk` mazes. A is the weighted graph
        Laplacian (A_uu = Σ_{e ∋ u} C_e, A_uv = −C_e); the gauge is pinned by
        fixing the potential of the first free cell in row-major order to 0
        (b sums to zero, so the system is consistent). Returns p (B, H, W),
        float64, with p = 0 in walls."""
        B = walk.shape[0]
        outs = []
        for s in range(0, B, self.solve_chunk):
            e = min(s + self.solve_chunk, B)
            outs.append(self._solve_chunk(walk[s:e], b[s:e], Ch[s:e], Cv[s:e]))
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def _solve_chunk(self, walk, b, Ch, Cv):
        B, H, W = walk.shape
        dev = walk.device
        flat = walk.reshape(B, -1)                              # (B, H·W), row-major
        n_free = flat.sum(dim=1).double()                       # free-cell count per maze
        N = int(n_free.max().item())
        p = torch.zeros(B, H, W, dtype=torch.float64, device=dev)
        if N == 0:                                              # all-wall chunk
            return p
        # Local index of each free cell in row-major order (first free cell = 0).
        loc_flat = (torch.cumsum(flat.double(), dim=1) - 1.0)   # (B, H·W)
        # Incidence construction of the Laplacian: A = Zᵀ diag(C_e) Z, with
        # Z[b, u, e] = +1 at one endpoint, −1 at the other. Z is a constant
        # (no gradient flows through the entries themselves), C_e carries the
        # gradient, and the batched matmul is fully differentiable.
        hm = torch.nonzero((walk[:, :, :-1] > 0) & (walk[:, :, 1:] > 0))
        vm = torch.nonzero((walk[:, :-1, :] > 0) & (walk[:, 1:, :] > 0))
        M = hm.shape[0] + vm.shape[0]
        Zc = torch.zeros(B, N, M, dtype=torch.float64, device=dev)
        Z = torch.zeros(B, N, M, dtype=torch.float64, device=dev)
        k = 0
        if hm.shape[0] > 0:
            hb, hr, hc = hm[:, 0], hm[:, 1], hm[:, 2]
            u = loc_flat[hb, hr * W + hc].long()
            v = loc_flat[hb, hr * W + hc + 1].long()
            c = Ch[hb, hr, hc]
            e = torch.arange(hm.shape[0], device=dev)
            Z.index_put_((hb, u, e), torch.ones_like(c))
            Z.index_put_((hb, v, e), -torch.ones_like(c))
            Zc.index_put_((hb, u, e), c)
            Zc.index_put_((hb, v, e), -c)
            k = hm.shape[0]
        if vm.shape[0] > 0:
            vb, vr, vc = vm[:, 0], vm[:, 1], vm[:, 2]
            u = loc_flat[vb, vr * W + vc].long()
            v = loc_flat[vb, (vr + 1) * W + vc].long()
            c = Cv[vb, vr, vc]
            e = torch.arange(vm.shape[0], device=dev) + k
            Z.index_put_((vb, u, e), torch.ones_like(c))
            Z.index_put_((vb, v, e), -torch.ones_like(c))
            Zc.index_put_((vb, u, e), c)
            Zc.index_put_((vb, v, e), -c)
        A = torch.bmm(Zc, Z.transpose(1, 2))                    # (B, N, N) Laplacian
        # Right-hand side: the source field at the free cells.
        rhs = torch.zeros(B, N, dtype=torch.float64, device=dev)
        fb, ff = torch.nonzero(flat > 0, as_tuple=True)
        rhs.index_put_((fb, loc_flat[fb, ff].long()),
                       b.reshape(B, -1)[fb, ff])
        # Gauge pin: first free cell (row-major) gets p = 0 — zero its row and
        # set its diagonal to 1, and zero its right-hand side.
        eye = torch.eye(N, dtype=torch.float64, device=dev)
        keep = torch.ones(N, N, dtype=torch.float64, device=dev)
        keep[0, :] = 0.0
        pin = torch.zeros(N, N, dtype=torch.float64, device=dev)
        pin[0, 0] = 1.0
        A = A * keep + pin
        rhs = rhs * (torch.arange(N, device=dev) > 0).double()
        # Padding rows (a maze with fewer free cells than the chunk's N) and
        # isolated free cells (zero conductance sum; none exist on validated
        # mazes) need a nonzero diagonal for the solve to exist.
        pad = (torch.arange(N, device=dev).double()[None, :] >= n_free[:, None]).double()
        A = A + pad[:, :, None] * eye[None]
        diag = torch.diagonal(A, dim1=1, dim2=2)
        A = A + (diag == 0).double()[:, :, None] * eye[None] * self.eps_d
        sol = torch.linalg.solve(A, rhs)                        # (B, N)
        # Scatter the solution back onto the grid: p at wall cells is 0.
        loc = loc_flat.reshape(B, H, W)
        sol_grid = torch.gather(sol, 1, loc.flatten(1).clamp(min=0).long())
        p = (sol_grid * flat.double()).reshape(B, H, W)
        return p

    def _gauss_seidel(self, walk, b, Ch, Cv, p):
        """Spec "Update" step 1, sweep mode: K_rel two-colour Gauss–Seidel
        sweeps from the current p. Each half-sweep applies, to every free cell
        of one checkerboard colour,

            p_u ← (b_u + Σ_{v ~ u} C_uv p_v) / (Σ_{v ~ u} C_uv + eps_d·[Σ C = 0])

        (its neighbours are all the other colour, so the half-sweep is one
        vectorized neighbourhood gather and an exact coordinate-wise
        minimization of the potential energy for its colour). Wall cells are
        untouched (p = 0). Returns (p, psi_halves, residual_halves): the
        potential energy Ψ = ½ Σ Q²/C − Σ b·p after every half-sweep and the
        conservation residual max_u |div_u − b_u| as a per-half diagnostic.
        Ψ uses the conductances C_e = D_e/L_e — the C-weighted functional the
        update actually minimizes, which coincides with the length-blind
        ½ Σ Q²/D form only while L ≡ 1 — and both diagnostics are computed
        straight from p and the fixed conductances (no guarded divisions), so
        the per-half-sweep overhead stays small at large k_rel."""
        B, H, W = walk.shape
        Ce = torch.zeros_like(b)
        Ce[:, :, :-1] = Ch
        Cs = torch.zeros_like(b)
        Cs[:, :-1, :] = Cv
        Cw = torch.zeros_like(b)
        Cw[:, :, 1:] = Ch
        Cn = torch.zeros_like(b)
        Cn[:, 1:, :] = Cv
        sC = Ce + Cw + Cs + Cn
        denom = sC + self.eps_d * (sC == 0).to(sC.dtype)        # guard: isolated cells only
        free = walk > 0
        rr = torch.arange(H, device=walk.device)[:, None]
        cc = torch.arange(W, device=walk.device)[None, :]
        red = (rr + cc) % 2 == 0
        psi_halves, res_halves = [], []
        for half in range(2 * self.k_rel):                      # one sweep = red + black
            colour = red if half % 2 == 0 else ~red
            upd = free & colour
            pe = F.pad(p, (0, 1))[:, :, 1:]                    # p of the east neighbour
            pw = F.pad(p, (1, 0))[:, :, :-1]                   # p of the west neighbour
            ps = F.pad(p, (0, 0, 0, 1))[:, 1:, :]              # p of the south neighbour
            pn = F.pad(p, (0, 0, 1, 0))[:, :-1, :]             # p of the north neighbour
            num = b + Ce * pe + Cw * pw + Cs * ps + Cn * pn
            p = torch.where(upd, num / denom, p)
            Qh, Qv = self._flows(p, Ch, Cv)
            psi = self._potential_energy(b, p, Ch, Cv).detach()
            res = self._conservation_residual(b, walk, Qh, Qv).detach()
            psi_halves.append(float(psi))
            res_halves.append(float(res))
        return p, psi_halves, res_halves

    # ------------------------------------------------------------------- fluxes

    def _flows(self, p, Ch, Cv):
        """Spec "Update" step 2: Q_e = C_e (p_u − p_v), signed, positive eastward
        or southward; antisymmetric by construction (one signed map per edge,
        so the flow out of u into v is minus the flow out of v into u)."""
        Qh = Ch * (p[:, :, :-1] - p[:, :, 1:])                  # (B, H, W-1), eastward +
        Qv = Cv * (p[:, :-1, :] - p[:, 1:, :])                  # (B, H-1, W), southward +
        return Qh, Qv

    @staticmethod
    def _divergence(Qh, Qv):
        """Net outflow per cell: div[r, c] = Qh[r, c] − Qh[r, c−1] + Qv[r, c]
        − Qv[r−1, c] (zero padding at the borders); conservation means div = b."""
        east_out = F.pad(Qh, (0, 1))
        east_in = F.pad(Qh, (1, 0))
        south_out = F.pad(Qv, (0, 0, 0, 1))
        south_in = F.pad(Qv, (0, 0, 1, 0))
        return east_out - east_in + south_out - south_in

    @staticmethod
    def _dissipation(Qh, Qv, Dh, Dv):
        """E = Σ_e Q_e² / D_e over free edges. D_e = 0 occurs exactly where
        Q_e = 0 (wall edges), so the term is 0 there — guarded to keep 0/0
        out of the sum. Note this is the *unit-length* dissipation: with
        learned lengths L_e ≠ 1 the resistor network's true dissipation is
        Σ_e Q_e²/C_e = Σ_e Q_e² L_e/D_e, and the recorded E_t equals it only
        while L ≡ 1 (the physics arm, and the learned arm before its length
        head is trained). It is kept as the step's E_t diagnostic because the
        spec defines E_t this way and the coupled monotonicity claims are
        stated at uniform lengths."""
        zh = torch.where(Dh > 0, Qh * Qh / Dh, torch.zeros_like(Qh))
        zv = torch.where(Dv > 0, Qv * Qv / Dv, torch.zeros_like(Qv))
        return zh.sum() + zv.sum()

    def _potential_energy(self, b, p, Ch, Cv):
        """Ψ(p) = ½ Σ_e Q_e(p)²/C_e − Σ_u b_u p_u = ½ Σ_e C_e (p_u − p_v)²
        − Σ_u b_u p_u: the functional the relaxation (exact solve and every
        Gauss–Seidel half-sweep alike) actually minimizes, so it decreases
        monotonically per half-sweep for ANY weights — trained lengths L_e
        included, since the weights enter through C_e = D_e/L_e. It reduces to
        the length-blind ½ Σ Q²/D − Σ b·p form only while L ≡ 1 (the physics
        arm, and the learned arm at its zero-weight length init). C_e = 0
        (wall edges) contributes exactly 0, so no division guard is needed."""
        dph = p[:, :, :-1] - p[:, :, 1:]
        dpv = p[:, :-1, :] - p[:, 1:, :]
        return 0.5 * (Ch * dph * dph).sum() + 0.5 * (Cv * dpv * dpv).sum() \
            - (b * p).sum()

    def _conservation_residual(self, b, walk, Qh, Qv):
        """max_u |div_u − b_u| over free cells (0.0 on an all-wall grid)."""
        free = walk > 0
        if free.any():
            return (self._divergence(Qh, Qv) - b).abs()[free].max()
        return torch.zeros((), dtype=b.dtype, device=b.device)

    @staticmethod
    def _cell_absflow(Qh, Qv):
        """|Q|_cell (B, H, W): Σ of |Q_e| over the ≤ 4 incident edges."""
        ah, av = Qh.abs(), Qv.abs()
        ae = F.pad(ah, (0, 1))
        aw = F.pad(ah, (1, 0))
        asn = F.pad(av, (0, 0, 0, 1))
        an = F.pad(av, (0, 0, 1, 0))
        return ae + aw + asn + an

    @staticmethod
    def _cell_conductance(Dh, Dv, Eh, Ev):
        """D_cell (B, H, W): mean conductance of the incident free edges
        (0 where a cell has none)."""
        de = F.pad(Dh, (0, 1))
        dw = F.pad(Dh, (1, 0))
        ds = F.pad(Dv, (0, 0, 0, 1))
        dn = F.pad(Dv, (0, 0, 1, 0))
        ce = F.pad(Eh, (0, 1))
        cw = F.pad(Eh, (1, 0))
        cs = F.pad(Ev, (0, 0, 0, 1))
        cn = F.pad(Ev, (0, 0, 1, 0))
        cnt = ce + cw + cs + cn
        return torch.where(cnt > 0, (de + dw + ds + dn) / cnt.clamp(min=1.0),
                           torch.zeros_like(cnt))

    # -------------------------------------------------------------------- step

    def step(self, x, Dh, Dv, p):
        """One outer step (spec "Update" 1–4), from state (Dh, Dv, p) to the
        next. Returns (Dh, Dv, p, Qh, Qv, info); info carries the step's
        dissipation, conservation residual and (in sweep mode) the potential
        energy and residual after every half-sweep."""
        walk = x[:, 0]                                          # (B, H, W)
        b = (self.I * (x[:, 1] - x[:, 2])).double()             # source field b
        Eh, Ev = self.edge_masks(x)
        if self.physics_only:                                  # physics arm: L ≡ 1
            Lh = torch.ones_like(Dh)
            Lv = torch.ones_like(Dv)
        else:
            uh, uv, Lh, Lv = self.edge_features(x)
        Dh64, Dv64 = Dh.double(), Dv.double()
        Lh64, Lv64 = Lh.double(), Lv.double()
        Ch = Dh64 / Lh64                                        # C_e = D_e / L_e
        Cv = Dv64 / Lv64
        # 1. Relax the potentials (exact solve or K_rel two-colour sweeps).
        if self.relaxation == "exact":
            p64 = self._solve_potentials(walk, b, Ch, Cv)
            psi_halves, res_halves = [], []
        else:
            p64, psi_halves, res_halves = \
                self._gauss_seidel(walk, b, Ch, Cv, p.double())
        # 2. Flows and diagnostics, in float64.
        Qh64, Qv64 = self._flows(p64, Ch, Cv)
        resid = self._conservation_residual(b, walk, Qh64, Qv64)
        E_t = self._dissipation(Qh64, Qv64, Dh64, Dv64)
        Qh, Qv = Qh64.to(Dh.dtype), Qv64.to(Dv.dtype)           # cast to float32
        # 3. Redistribute the material: D_e ← M_tube · D_e f_e / Σ D_e f_e.
        if self.physics_only:                                  # f_e = |Q_e| + eps_f
            fh = Qh.abs() + self.eps_f
            fv = Qv.abs() + self.eps_f
        else:
            wq = F.softplus(self.raw_wq)                        # w_q >= 0
            zh = wq * Qh.abs() + (self.w_u[None, :, None, None] * uh).sum(dim=1) + self.b_f
            zv = wq * Qv.abs() + (self.w_u[None, :, None, None] * uv).sum(dim=1) + self.b_f
            fh = F.softplus(zh) + self.eps_f
            fv = F.softplus(zv) + self.eps_f
        Dh, Dv = self._normalize_material(Eh, Ev, Dh * fh, Dv * fv)
        # 4. Re-impose the wall structure: edges touching a wall are exactly 0.
        Dh = Dh * Eh
        Dv = Dv * Ev
        info = {"dissipation": float(E_t.detach()), "residual": float(resid.detach()),
                "psi_halves": psi_halves, "residual_halves": res_halves}
        return Dh, Dv, p64.to(x.dtype), Qh, Qv, info

    # ------------------------------------------------------------------ readout

    def read(self, x, Dh, Dv, p, Qh, Qv):
        """Per-cell path logits (B, H, W); walls are filled with exactly −20.0.

        Learned arm: Conv1x1 over [x, |Q|_cell, p, D_cell], thresholded at 0.5
        like the other models. Physics-only arm: the cell is on the path iff
        |Q|_cell >= I/2, emitted as a ±20 logit so the standard evaluation
        (sigmoid then 0.5 threshold) applies unchanged."""
        walk = x[:, 0]
        qcell = self._cell_absflow(Qh, Qv)                      # (B, H, W)
        if self.physics_only:
            logits = torch.where(qcell >= 0.5 * self.I,
                                 torch.full_like(qcell, PATH_LOGIT),
                                 torch.full_like(qcell, WALL_LOGIT))
        else:
            Eh, Ev = self.edge_masks(x)
            dcell = self._cell_conductance(Dh, Dv, Eh, Ev)      # (B, H, W)
            feats = torch.cat([x, qcell[:, None], p[:, None], dcell[:, None]], dim=1)
            logits = self.readout(feats).squeeze(1)
        return logits.masked_fill(walk == 0, WALL_LOGIT)

    # ----------------------------------------------------------------- interface

    def forward(self, x, T):
        """Logits (B, H, W) after T outer steps from the initial state."""
        Dh, Dv, p = self.initial_state(x)
        Qh = torch.zeros_like(Dh)
        Qv = torch.zeros_like(Dv)
        for _ in range(T):
            Dh, Dv, p, Qh, Qv, _ = self.step(x, Dh, Dv, p)
        return self.read(x, Dh, Dv, p, Qh, Qv)

    def logits_at(self, x, Ts, record=False):
        """Read out logits after each step count in `Ts`, from ONE shared
        rollout (same contract as cf/models/cf.py).

        With record=True the trace stores, per outer step (after that step),
        the D maps, the flow maps Qh/Qv and |Q|_cell, the potential p, the
        dissipation E_t of the step and the flow-conservation residual
        max_u |div_u − b_u|; in sweep mode it also stores the potential energy
        and residual after every half-sweep of the step's relaxation."""
        Ts = sorted(set(Ts))
        Dh, Dv, p = self.initial_state(x)
        Qh = torch.zeros_like(Dh)
        Qv = torch.zeros_like(Dv)
        out, trace = {}, self._new_trace() if record else None
        t = 0
        for T in Ts:
            for _ in range(T - t):
                Dh, Dv, p, Qh, Qv, info = self.step(x, Dh, Dv, p)
                t += 1
                if record:
                    self._record(trace, Dh, Dv, p, Qh, Qv, info)
            out[T] = self.read(x, Dh, Dv, p, Qh, Qv)
        if record:
            return out, trace
        return out

    @staticmethod
    def _new_trace():
        return {"Dh": [], "Dv": [], "p": [], "Qh": [], "Qv": [], "qcell": [],
                "dissipation": [], "residual": [],
                "psi_halves": [], "residual_halves": []}

    def _record(self, trace, Dh, Dv, p, Qh, Qv, info):
        trace["Dh"].append(Dh.detach().cpu())
        trace["Dv"].append(Dv.detach().cpu())
        trace["p"].append(p.detach().cpu())
        trace["Qh"].append(Qh.detach().cpu())
        trace["Qv"].append(Qv.detach().cpu())
        trace["qcell"].append(self._cell_absflow(Qh, Qv).detach().cpu())
        trace["dissipation"].append(float(info["dissipation"]))
        trace["residual"].append(float(info["residual"]))
        trace["psi_halves"].append(list(info["psi_halves"]))
        trace["residual_halves"].append(list(info["residual_halves"]))
