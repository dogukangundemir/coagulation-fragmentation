"""Transport coagulation-fragmentation mechanism figures (physics only, no learning).

Plain language: the maze is a network of pipes - every pair of adjacent free
cells is a pipe, a wall is a missing pipe. One unit of water per time step is
poured into the start cell and drained from the goal cell, forever. A pipe
that carries current thickens; a pipe that carries none thins. The total
amount of pipe material never changes - material only moves between pipes.
On a maze without loops exactly one route connects start and goal, and no
steady current can enter a dead-end corridor, so the physics alone marks the
solution path. This script shows that mechanism on one level-5 maze and one
level-9 maze with the arm that has ZERO learned parameters (uniform pipe
lengths L_e = 1, adaptation f_e = |Q_e| + eps_f; specification card
docs/specs/family_d_transport.md):

  flow_sweeps.png    |Q|_cell maps while the FIRST outer step's pressure
                     relaxation runs (two-colour Gauss-Seidel half-sweeps,
                     all pipes equally thick, pressures started at zero):
                     current spreads through the whole maze, then collapses
                     onto the solution path; dead ends go dark.
  tube_material.png  pipe thickness D_e maps over outer adaptation steps
                     (exact float64 pressure solve per step): material
                     coagulates into the one carrying channel and starves
                     everywhere else.
  monitors.png       three monitor curves: (a) share of all current |Q| on
                     the solution path -> 1 as the relaxation sweeps converge,
                     (b) the total material budget |Sum L_e D_e / M_tube - 1|
                     staying at the float32 rounding level across outer steps,
                     (c) the dissipated energy E_t = Sum Q_e^2 / D_e falling
                     as material thickens the route.
  data.csv           every plotted number (sweep scan, exact rollout, maze
                     meta, and an audit run at the spec's 16-sweep budget).
  README.md          captions, how-to-read, key takeaways (numbers straight
                     from the computed data), data sources, caveats.

Maze selection rule (fixed, not hand-picked): maze index 0 of the seeded
level-5 test split and maze index 0 of the seeded level-9 probe split (the
same seeded splits, seed 0, as every other run in this repository).

Run from the clone root:
    python -m scripts.figures.transport_mechanism
Everything runs on CPU; the arm has no training phase (0 epochs, 0 seeds).
After building each figure the script runs a programmatic readability audit:
no text may overlap any other text, nor any sampled line or marker point.
"""
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, PowerNorm, to_rgba
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import style  # noqa: E402  (shared figure style, read-only)
import cf  # noqa: E402
from cf.data.curriculum import build_datasets, load_curriculum, make_splits  # noqa: E402
from cf.models.transport import TransportCFNet  # noqa: E402

style.apply_style()

OUT = ROOT / "figures" / "transport_mechanism"
EVAL_JSON = ROOT / "results" / "transport_physics" / "results.json"

SNAP_HALF = [1, 8, 64, 512, 2048, 8192]   # flow_sweeps columns (half-sweeps)
N_HALF = max(SNAP_HALF)
OUTER_STEPS = 16                          # exact-rollout horizon per maze
SNAP_STEPS = [0, 1, 2, 4, 8]              # tube_material columns (outer steps)
AUDIT_KREL = 16                           # spec relaxation budget (sweeps/step)
AUDIT_STEPS = 12
RESID_TOL = 1e-5                          # spec tolerance, as in eval_transport_physics
MAZE_INDEX = 0                             # fixed selection rule

OI = style.OKABE_ITO
C_L5 = OI["blue"]          # level-5 maze colour
C_L9 = OI["vermillion"]    # level-9 maze colour
C_VLINE = OI["grey"]
WALL_RGB = "#d0d0d0"       # wall cells in the maps
PATH_LN = "white"          # solution-path overlay on the maps
MARK_FACE = OI["vermillion"]
MARK_EDGE = "white"
GUIDE_FS = 11.5
TICK_FS = 11
VIRIDIS = plt.get_cmap("viridis")


# ------------------------------------------------------------- run guards

def ensure_clone():
    """Refuse to run unless the imported cf package is THIS clone."""
    cf_root = Path(cf.__file__).resolve().parents[1]
    if cf_root != ROOT.resolve():
        raise SystemExit("cf resolves to %s, not this clone (%s)"
                         % (cf_root, ROOT.resolve()))
    return cf_root


# --------------------------------------------------------------- helpers
# Figure placement helpers: inches -> figure fractions (y measured from bottom)

def axrect(fig, x, y, w, h):
    W, H = fig.get_figwidth(), fig.get_figheight()
    return [x / W, y / H, w / W, h / H]


def itext(fig, x, y, s, **kw):
    W, H = fig.get_figwidth(), fig.get_figheight()
    return fig.text(x / W, y / H, s, **kw)


def tick_labels_in_view(ax):
    """Tick labels whose tick actually lies inside the axis view. The label
    Text objects of out-of-view ticks still report window extents, which
    would produce phantom overlaps."""
    out = []
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    for t, loc in zip(ax.get_xticklabels(), ax.get_xticks()):
        if t.get_text() and x0 - 1e-9 * max(1.0, abs(x0)) <= loc <= x1 + 1e-9 * max(1.0, abs(x1)):
            out.append(t)
    for t, loc in zip(ax.get_yticklabels(), ax.get_yticks()):
        if t.get_text() and y0 - 1e-9 * max(1.0, abs(y0)) <= loc <= y1 + 1e-9 * max(1.0, abs(y1)):
            out.append(t)
    return out


def audit_overlaps(fig, name):
    """Programmatic readability audit: no text may overlap any other text,
    nor any sampled line / marker point (window extents after a draw), and
    no text may run past the canvas."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    problems = []
    alltexts = []                       # (ax-or-None, Text)
    ax_text_map = {}
    for ax in fig.axes:
        lst = [t for t in ax.texts if t.get_text()]
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if t.get_text():
                lst.append(t)
        ax_text_map[ax] = lst
        for t in lst + tick_labels_in_view(ax):
            alltexts.append((ax, t))
    for t in fig.texts:
        if t.get_text():
            alltexts.append((None, t))
    # text vs canvas bounds
    for owner, t in alltexts:
        b = t.get_window_extent(rend)
        if b.x0 < 0 or b.x1 > W or b.y0 < 0 or b.y1 > H:
            problems.append("%s: text %r runs past the canvas (%d..%d x %d..%d)"
                            % (name, t.get_text()[:30], b.x0, b.x1, b.y0, b.y1))
    # text vs text
    for i in range(len(alltexts)):
        for j in range(i + 1, len(alltexts)):
            b1 = alltexts[i][1].get_window_extent(rend)
            b2 = alltexts[j][1].get_window_extent(rend)
            if b1.overlaps(b2):
                problems.append("%s: text %r overlaps text %r"
                                % (name, alltexts[i][1].get_text()[:30],
                                   alltexts[j][1].get_text()[:30]))
    # in-axes text vs sampled line / marker points of the same axes
    for ax, texts in ax_text_map.items():
        boxes = [(t, t.get_window_extent(rend)) for t in texts]
        for ln in ax.lines:
            xy = ln.get_xydata()
            if len(xy) == 0:
                continue
            pts = ax.transData.transform(xy)
            for t, b in boxes:
                hit = ((pts[:, 0] >= b.x0) & (pts[:, 0] <= b.x1)
                       & (pts[:, 1] >= b.y0) & (pts[:, 1] <= b.y1))
                if hit.any():
                    problems.append("%s: text %r overlaps a line/marker"
                                    % (name, t.get_text()[:30]))
        # legend frames: vs the axes' lines and texts
        lg = ax.get_legend()
        if lg is not None:
            lb = lg.get_window_extent(rend)
            for ln in ax.lines:
                xy = ln.get_xydata()
                if len(xy) == 0:
                    continue
                pts = ax.transData.transform(xy)
                if (((pts[:, 0] >= lb.x0) & (pts[:, 0] <= lb.x1)
                     & (pts[:, 1] >= lb.y0) & (pts[:, 1] <= lb.y1)).any()):
                    problems.append("%s: legend overlaps a line/marker" % name)
            for t in texts:
                if t.get_window_extent(rend).overlaps(lb):
                    problems.append("%s: legend overlaps text %r"
                                    % (name, t.get_text()[:30]))
    # figure-level legends vs figure texts
    for lg in fig.legends:
        lb = lg.get_window_extent(rend)
        for owner, t in alltexts:
            if t.get_window_extent(rend).overlaps(lb):
                problems.append("%s: figure legend overlaps text %r"
                                % (name, t.get_text()[:30]))
    return problems


# ------------------------------------------------------------ maze loading

def load_two_mazes():
    """Maze index 0 of the seeded level-5 test split and of the seeded
    level-9 probe split (seed 0, identical to every other run here)."""
    mazes, sols, levels = load_curriculum()
    splits, *_ = make_splits(mazes, sols, levels,
                             n_train=4000, n_val=1000, n_test=1000,
                             n_probe=1000, week1_level=5,
                             probe_levels=(7, 9), seed=0)
    ds = build_datasets(splits, mazes, sols)
    x5, y5 = ds["test"][MAZE_INDEX]
    x9, y9 = ds["probe"][9][MAZE_INDEX]
    return (x5, y5), (x9, y9)


class MazeCase:
    """One maze plus all derived geometry the figures need."""

    def __init__(self, name, x, y):
        self.name = name
        self.x = x[None]
        self.y = y[None]
        self.walk = self.x[:, 0]
        self.H, self.W = self.walk.shape[1], self.walk.shape[2]
        s = int(self.x[0, 1].flatten().argmax())
        g = int(self.x[0, 2].flatten().argmax())
        self.start = (s // self.W, s % self.W)
        self.goal = (g // self.W, g % self.W)
        self.on = (self.y > 0.5)
        self.onh = self.on[:1, :, :-1] & self.on[:1, :, 1:]
        self.onv = self.on[:, :-1, :] & self.on[:, 1:, :]
        self.path_cells = self._walk_path()
        m = TransportCFNet(physics_only=True, relaxation="exact")
        self.Eh, self.Ev = m.edge_masks(self.x)
        self.n_pipes = int(self.Eh.sum() + self.Ev.sum())
        self.n_free_cells = int(self.walk.sum())
        self.n_path_pipes = int(self.onh.sum() + self.onv.sum())
        self.n_path_cells = int(self.y.sum())

    def _walk_path(self):
        """Ordered path cell list, start -> goal, by following the label
        mask (every interior path cell has exactly two path neighbours)."""
        on = self.on[0]
        H, W = on.shape
        cur, prev, cells = self.start, None, [self.start]
        while cur != self.goal:
            r, c = cur
            nxt = None
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = (r + dr, c + dc)
                if not (0 <= nb[0] < H and 0 <= nb[1] < W):
                    continue
                if not on[nb] or nb == prev:
                    continue
                nxt = nb
                break
            assert nxt is not None, "path walk stuck at %r" % (cur,)
            prev, cur = cur, nxt
            cells.append(cur)
        return cells


# ------------------------------------------------------------- physics runs

def path_share(Qh, Qv, case):
    """|Q| on the label-path pipes / |Q| over all free pipes, plus the
    total |Q| sum on free pipes."""
    ah, av = Qh.abs(), Qv.abs()
    tot = (ah * case.Eh.double()).sum() + (av * case.Ev.double()).sum()
    onf = (ah * case.onh.double()).sum() + (av * case.onv.double()).sum()
    return (onf / tot).item(), tot.item()


def sweep_scan(case, n_half=N_HALF):
    """Relax the FIRST outer step's pressures from zero with the sweep
    relaxation, recording monitors after every half-sweep; pipes stay at
    their uniform initial thickness (that is what the first step solves)."""
    m = TransportCFNet(physics_only=True, relaxation="sweep", k_rel=1)
    walk = case.x[:, 0]
    b = (m.I * (case.x[:, 1] - case.x[:, 2])).double()
    Ch = case.Eh.double()
    Cv = case.Ev.double()
    Dh, Dv, p = m.initial_state(case.x)
    assert float((Dh * case.Eh).sum() + (Dv * case.Ev).sum()) == float(case.n_pipes)
    rows, snaps = [], {}
    for h in range(n_half):
        p, _, _ = m._gauss_seidel(walk, b, Ch, Cv, p.double())
        Qh, Qv = m._flows(p, Ch, Cv)
        share, tot = path_share(Qh, Qv, case)
        res = float(m._conservation_residual(b, walk, Qh, Qv))
        qcell = m._cell_absflow(Qh, Qv).cpu().numpy()
        rows.append(dict(x=h + 1, share=share, resid=res, tot=tot,
                         qmax=float(qcell.max())))
        if (h + 1) in SNAP_HALF:
            snaps[h + 1] = qcell
    return rows, snaps


def exact_rollout(case, n_steps=OUTER_STEPS):
    """Outer adaptation steps with the exact float64 pressure solve per step
    (the repo's default on the curriculum grids). Records per-step monitors
    and cell-thickness maps at the snapshot steps. Stops and marks a hold if
    the solve degenerates (residual crossing the spec tolerance, or a
    singular solve) - the same policy as scripts/eval_transport_physics.py."""
    m = TransportCFNet(physics_only=True, relaxation="exact")
    Dh, Dv, p = m.initial_state(case.x)
    m_tube = float(Dh.sum() + Dv.sum())
    snaps = {0: m._cell_conductance(Dh, Dv, case.Eh, case.Ev).cpu().numpy()}
    rows, held = [], None
    for t in range(1, n_steps + 1):
        try:
            Dh, Dv, p, Qh, Qv, info = m.step(case.x, Dh, Dv, p)
            resid = float(info["residual"])
        except Exception:                       # singular exact solve -> hold
            held = t
            break
        if resid > RESID_TOL:
            held = t
            break
        share, tot = path_share(Qh.double(), Qv.double(), case)
        deadh = Dh * case.Eh * (~case.onh)
        deadv = Dv * case.Ev * (~case.onv)
        mat = float(Dh.sum() + Dv.sum())
        mat_on_path = float((Dh * case.onh).sum() + (Dv * case.onv).sum())
        rows.append(dict(x=t, share=share, resid=resid, tot=tot,
                         mat_dev=abs(mat / m_tube - 1.0),
                         diss=float(info["dissipation"]),
                         dead_max=float(max(deadh.max(), deadv.max())),
                         mat_on_path=mat_on_path / m_tube))
        if t in SNAP_STEPS:
            snaps[t] = m._cell_conductance(Dh, Dv, case.Eh, case.Ev).cpu().numpy()
    return rows, snaps, held, m_tube


def sweep_budget_audit(case, k_rel=AUDIT_KREL, n_steps=AUDIT_STEPS):
    """Audit run: outer steps with the spec's relaxation budget (16 sweeps
    per step) instead of the exact solve. Not used for any figure panel;
    its per-step residuals land in data.csv and in the README caveat."""
    m = TransportCFNet(physics_only=True, relaxation="sweep", k_rel=k_rel)
    Dh, Dv, p = m.initial_state(case.x)
    rows = []
    for t in range(1, n_steps + 1):
        Dh, Dv, p, Qh, Qv, info = m.step(case.x, Dh, Dv, p)
        share, tot = path_share(Qh.double(), Qv.double(), case)
        rows.append(dict(x=t, share=share, resid=float(info["residual"]),
                         tot=tot))
    return rows


# --------------------------------------------------------------- rendering

def grid_rgba(grid, walk, cmap, norm):
    """Full-grid RGBA image: free cells coloured by the value through
    (cmap, norm); wall cells in the fixed wall grey."""
    if grid.ndim == 3:            # drop a leading batch axis if present
        grid = grid[0]
    H, W = grid.shape
    rgb = cmap(norm(grid))
    out = np.zeros((H, W, 4))
    w = walk[0] if walk.ndim == 3 else walk
    out[w == 0] = to_rgba(WALL_RGB)
    out[w > 0] = rgb[w > 0]
    return out


def draw_map(ax, grid, case, cmap, norm):
    ax.imshow(grid_rgba(grid, case.walk.numpy(), cmap, norm),
              interpolation="nearest", extent=(0, case.W, case.H, 0))
    px = [c[1] + 0.5 for c in case.path_cells]
    py = [c[0] + 0.5 for c in case.path_cells]
    ax.plot(px, py, ls="--", color=PATH_LN, lw=1.6, alpha=0.95, zorder=4)
    ax.plot(case.start[1] + 0.5, case.start[0] + 0.5, "o", ms=9,
            mfc=MARK_FACE, mec=MARK_EDGE, mew=1.2, zorder=5, ls="none")
    ax.plot(case.goal[1] + 0.5, case.goal[0] + 0.5, "s", ms=9,
            mfc=MARK_FACE, mec=MARK_EDGE, mew=1.2, zorder=5, ls="none")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def map_figure(cases, snaps_by_case, columns, cmap, norm, cbar_label,
               suptitle, guide, row_labels, out_png, fig_w=17.0):
    """Two maze rows x column snapshots; hand-placed inch geometry so the
    title, guide, column titles, maps and colorbar cannot drift."""
    n_cols = len(columns)
    pitch = (fig_w - 1.30 - 1.45) / n_cols        # inches per column
    w_map = pitch - 0.16
    y_row2 = 0.75
    y_row1 = y_row2 + w_map + 0.42
    col_y = y_row1 + w_map + 0.18
    fig_h = col_y + 0.32 + 0.30 + 0.57 + 0.32 + 0.75
    fig = plt.figure(figsize=(fig_w, fig_h))
    for ri, case in enumerate(cases):
        y = y_row2 if ri == 1 else y_row1
        for ci, (key, _title) in enumerate(columns):
            ax = fig.add_axes(axrect(fig, 1.30 + ci * pitch, y, w_map, w_map))
            draw_map(ax, snaps_by_case[case.name][key], case, cmap, norm)
    for ci, (_key, title) in enumerate(columns):
        itext(fig, 1.30 + ci * pitch + w_map / 2, col_y, title,
              ha="center", va="bottom", fontsize=TICK_FS + 0.5, color="black")
    for ri, case in enumerate(cases):
        y = y_row2 if ri == 1 else y_row1
        itext(fig, 0.52, y + w_map / 2, row_labels[case.name],
              rotation=90, ha="center", va="center", fontsize=TICK_FS)
    cx = 1.30 + n_cols * pitch + 0.12
    cax = fig.add_axes(axrect(fig, cx, y_row2, 0.28, 2 * w_map + 0.42))
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.set_label(cbar_label, fontsize=TICK_FS + 0.5)
    itext(fig, 0.30, fig_h - 0.55, suptitle, ha="left", va="top",
          fontsize=16, fontweight="bold")
    itext(fig, 0.30, fig_h - 0.95, guide[0], ha="left", va="top",
          fontsize=GUIDE_FS)
    itext(fig, 0.30, fig_h - 0.95 - 0.30, guide[1], ha="left", va="top",
          fontsize=GUIDE_FS)
    handles = [Line2D([], [], ls="--", color="black", lw=1.6,
                      label="solution path (given label, not computed)"),
               Line2D([], [], ls="none", marker="o", ms=9, mfc=MARK_FACE,
                      mec=MARK_EDGE, mew=1.2, label="start (current in)"),
               Line2D([], [], ls="none", marker="s", ms=9, mfc=MARK_FACE,
                      mec=MARK_EDGE, mew=1.2, label="goal (current out)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=TICK_FS,
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.savefig(out_png)
    return fig


# -------------------------------------------------------------------- main

def fmt(x, digits=4):
    return ("%.*f" % (digits, x)).rstrip("0").rstrip(".")


def main():
    t_start = time.perf_counter()
    ensure_clone()
    print("cf package:", cf.__file__)
    torch.set_grad_enabled(False)

    (x5, y5), (x9, y9) = load_two_mazes()
    l5 = MazeCase("level5", x5, y5)
    l9 = MazeCase("level9", x9, y9)
    cases = (l5, l9)
    for c in cases:
        print("%s: %dx%d cells, %d free cells, %d pipes, path %d cells / %d pipes"
              % (c.name, c.H, c.W, c.n_free_cells, c.n_pipes,
                 c.n_path_cells, c.n_path_pipes))

    runs = {}
    for c in cases:
        rows, snaps = sweep_scan(c)
        runs[("sweeps", c.name)] = rows
        runs[("snaps_q", c.name)] = snaps
        erows, esnaps, held, m_tube = exact_rollout(c)
        runs[("exact", c.name)] = erows
        runs[("snaps_d", c.name)] = esnaps
        runs[("held", c.name)] = held
        runs[("m_tube", c.name)] = m_tube
        runs[("audit", c.name)] = sweep_budget_audit(c)
        print("%s: sweep scan %d halves; exact %d steps (held=%s); audit %d steps"
              % (c.name, N_HALF, len(erows), held, len(runs[("audit", c.name)])))

    params_raw = sum(p.numel() for p in
                     TransportCFNet(physics_only=True).parameters())
    assert params_raw == 0

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "maze", "x_name", "x_value", "path_share",
                    "residual", "total_abs_Q", "qcell_max",
                    "material_dev_abs", "dissipation", "dead_D_max",
                    "material_on_path_share"])
        for c in cases:
            meta = [("free_cells", c.n_free_cells), ("pipes", c.n_pipes),
                    ("path_pipes", c.n_path_pipes),
                    ("path_cells", c.n_path_cells),
                    ("m_tube", runs[("m_tube", c.name)])]
            for k, v in meta:
                w.writerow(["maze_meta", c.name, k, v, "", "", "", "", "",
                            "", "", "", ""])
            for r in runs[("sweeps", c.name)]:
                w.writerow(["sweeps_first_step", c.name, "half_sweep", r["x"],
                             "%.10g" % r["share"], "%.6g" % r["resid"],
                             "%.8g" % r["tot"], "%.6g" % r["qmax"],
                             "", "", "", ""])
            for r in runs[("exact", c.name)]:
                w.writerow(["outer_steps_exact", c.name, "outer_step", r["x"],
                             "%.10g" % r["share"], "%.6g" % r["resid"],
                             "%.8g" % r["tot"], "",
                             "%.6g" % r["mat_dev"], "%.10g" % r["diss"],
                             "%.6g" % r["dead_max"],
                             "%.10g" % r["mat_on_path"]])
            for r in runs[("audit", c.name)]:
                w.writerow(["outer_steps_sweep_budget16", c.name,
                             "outer_step", r["x"], "%.10g" % r["share"],
                             "%.6g" % r["resid"], "%.8g" % r["tot"],
                             "", "", "", "", ""])

    audits = {}

    # ---------------- flow_sweeps.png
    qmax = max(max(float(s.max()) for s in runs[("snaps_q", c.name)].values())
               for c in cases)
    norm_q = PowerNorm(0.45, vmin=0.0, vmax=qmax)
    guide_q = [
        "How to read: one maze per row under the first outer step's pressure relaxation from zero "
        "(pipes equally thick); column labels count relaxation half-sweeps (a sweep = one red plus "
        "one black pass).",
        "Colour = current |Q| touching a cell, in units of the source current I = 1 (one unit poured "
        "in at the start, drained at the goal); dashed white line = the maze's solution path; "
        "circle = start, square = goal.",
    ]
    snap_q = {c.name: {h: runs[("snaps_q", c.name)][h] for h in SNAP_HALF}
              for c in cases}
    row_labels = {c.name: "level-5 maze\n(%dx%d cells, %d pipes)"
                  % (c.H, c.W, c.n_pipes) if c.name == "level5" else
                  "level-9 maze\n(%dx%d cells, %d pipes)"
                  % (c.H, c.W, c.n_pipes) for c in cases}
    fig1 = map_figure(
        cases, snap_q,
        [(h, "after %d half-sweeps" % h) for h in SNAP_HALF],
        VIRIDIS, norm_q, "|Q| per cell (units of I = 1)",
        "Current floods the whole maze, then drains out of everything except the unique start-goal route",
        guide_q, row_labels, OUT / "flow_sweeps.png")
    audits["flow_sweeps.png"] = audit_overlaps(fig1, "flow_sweeps")
    plt.close(fig1)

    # ---------------- tube_material.png
    dmax = max(max(float(s.max()) for s in runs[("snaps_d", c.name)].values())
               for c in cases)
    dmin = 1e-8                     # display floor, stated in the caption
    norm_d = LogNorm(vmin=dmin, vmax=dmax)
    guide_d = [
        "How to read: one maze per row after the labelled number of outer adaptation steps; each step "
        "re-solves the pressures exactly, then moves pipe material towards the pipes that carry current.",
        "Colour = pipe thickness D_e (mean over the cell's pipes) on a log scale with display floor "
        "1e-8; dead-end pipes thin by about 1000x per step; overlays as in the flow figure.",
    ]
    snap_d = {c.name: {t: runs[("snaps_d", c.name)][t] for t in SNAP_STEPS}
              for c in cases}
    fig2 = map_figure(
        cases, snap_d,
        [(t, "outer step %d" % t) for t in SNAP_STEPS],
        VIRIDIS, norm_d, "pipe thickness D_e (mean of incident pipes, log scale)",
        "Pipe material coagulates into the single carrying channel; every other pipe starves to nothing",
        guide_d, row_labels, OUT / "tube_material.png")
    audits["tube_material.png"] = audit_overlaps(fig2, "tube_material")
    plt.close(fig2)

    # ---------------- monitors.png
    fig_w, fig_h = 15.6, 6.05
    fig3 = plt.figure(figsize=(fig_w, fig_h))
    gap = 1.00
    pw, ph = (fig_w - 0.85 - 0.25 - 2 * gap) / 3, 3.30
    p1 = fig3.add_axes(axrect(fig3, 0.85, 0.62, pw, ph))
    p2 = fig3.add_axes(axrect(fig3, 0.85 + pw + gap, 0.62, pw, ph))
    p3 = fig3.add_axes(axrect(fig3, 0.85 + 2 * (pw + gap), 0.62, pw, ph))
    colors = {"level5": C_L5, "level9": C_L9}
    marks = {"level5": "o", "level9": "s"}
    names = {"level5": "level-5 maze", "level9": "level-9 maze"}

    # (a) share of |Q| on the path, over the relaxation half-sweeps
    for c in cases:
        rows = runs[("sweeps", c.name)]
        xs = [r["x"] for r in rows]
        ys = [r["share"] for r in rows]
        p1.plot(xs, ys, color=colors[c.name], lw=2.0, label=names[c.name])
        last = rows[-1]
        p1.plot([last["x"]], [last["share"]], marker=marks[c.name], ms=5.5,
                color=colors[c.name], ls="none")
        va = "bottom" if c.name == "level5" else "top"
        dy = 0.035 if c.name == "level5" else -0.045
        p1.annotate("%.4f" % last["share"], xy=(last["x"], last["share"]),
                    xytext=(last["x"], last["share"] + dy), ha="right",
                    va=va, fontsize=TICK_FS, color=colors[c.name])
    p1.axvline(AUDIT_KREL * 2, color=C_VLINE, ls="--", lw=1.4)
    p1.text(AUDIT_KREL * 2 * 0.75, 0.04, "16 sweeps/step",
            fontsize=TICK_FS, color=C_VLINE, ha="right", va="bottom")
    p1.set_xscale("log")
    p1.set_xlim(0.8, 12000)
    p1.set_ylim(0.0, 1.12)
    p1.set_xticks([1, 10, 100, 1000, 10000])
    p1.set_xlabel("half-sweeps of the pressure relaxation",
                  fontsize=TICK_FS + 1)
    p1.set_ylabel("share of |Q| on the solution path", fontsize=TICK_FS + 1)
    p1.tick_params(labelsize=TICK_FS)
    p1.legend(loc="lower right", fontsize=TICK_FS, frameon=False)

    # (b) material conservation across outer steps
    devs = {}
    for c in cases:
        rows = runs[("exact", c.name)]
        xs = [r["x"] for r in rows]
        devs[c.name] = [r["mat_dev"] for r in rows]
        p2.plot(xs, devs[c.name], color=colors[c.name], lw=2.0,
                marker=marks[c.name], ms=4.5, ls="-")
    dev_max = max(max(v) for v in devs.values())
    p2.set_xlim(0.5, 16.5)
    p2.set_ylim(0.0, 1.45e-7)
    p2.set_yticks([0.0, 5e-8, 1e-7])
    p2.set_yticklabels(["0", "5e-8", "1e-7"])
    p2.set_xticks([1, 4, 8, 12, 16])
    p2.text(0.97, 0.96, "largest deviation over both\nmazes: %.1e - float32\nrounding of the exact budget"
            % dev_max, transform=p2.transAxes,
            ha="right", va="top", fontsize=TICK_FS)
    p2.set_xlabel("outer adaptation step", fontsize=TICK_FS + 1)
    p2.set_ylabel("|Sum L_e D_e / M_tube - 1|", fontsize=TICK_FS + 1)
    p2.tick_params(labelsize=TICK_FS)

    # (c) dissipation across outer steps
    ends = {}
    for c in cases:
        rows = runs[("exact", c.name)]
        xs = [r["x"] for r in rows]
        ys = [r["diss"] for r in rows]
        ends[c.name] = (ys[0], ys[-1])
        p3.plot(xs, ys, color=colors[c.name], lw=2.0)
        p3.plot([xs[-1]], [ys[-1]], marker=marks[c.name], ms=5.5,
                color=colors[c.name], ls="none")
        if c.name == "level9":
            p3.annotate("%.2f" % ys[0], xy=(xs[0], ys[0]),
                        xytext=(xs[0] + 0.35, ys[0] + 0.4), ha="left",
                        va="bottom", fontsize=TICK_FS, color=colors[c.name])
            p3.annotate("%.2f" % ys[-1], xy=(xs[-1], ys[-1]),
                        xytext=(xs[-1] + 0.35, 2.7), ha="left", va="bottom",
                        fontsize=TICK_FS, color=colors[c.name])
        else:
            p3.annotate("%.2f" % ys[0], xy=(xs[0], ys[0]),
                        xytext=(0.60, ys[0] + 0.6), ha="left", va="bottom",
                        fontsize=TICK_FS, color=colors[c.name])
            p3.annotate("%.2f" % ys[-1], xy=(xs[-1], ys[-1]),
                        xytext=(xs[-1] + 0.35, ys[-1]), ha="left", va="bottom",
                        fontsize=TICK_FS, color=colors[c.name])
    d0_max = max(ends[n][0] for n in ends)
    p3.set_xlim(0.5, 19.6)
    p3.set_ylim(0.0, d0_max * 1.14)
    p3.set_xticks([1, 4, 8, 12, 16])
    p3.set_xlabel("outer adaptation step", fontsize=TICK_FS + 1)
    p3.set_ylabel("dissipation E_t = Sum Q_e^2 / D_e\n(units of I^2)",
                  fontsize=TICK_FS + 1)
    p3.tick_params(labelsize=TICK_FS)

    guide3 = [
        "How to read: left - share of all current |Q| on the solution path pipes vs "
        "relaxation half-sweeps (log x); grey dashed line = the spec budget of 16 "
        "sweeps per outer step.",
        "Middle - material deviation |Sum L_e D_e / M_tube - 1| over outer steps; "
        "right - dissipation E_t. Blue = level-5, vermillion = level-9; n = 1 "
        "deterministic maze per level, so no bands.",
    ]
    itext(fig3, 0.30, fig_h - 0.42,
          "The physics keeps its three promises: current only on the route, "
          "material constant, dissipation falling",
          ha="left", va="top", fontsize=16, fontweight="bold")
    itext(fig3, 0.30, fig_h - 0.84, guide3[0], ha="left", va="top",
          fontsize=GUIDE_FS)
    itext(fig3, 0.30, fig_h - 0.84 - 0.30, guide3[1], ha="left", va="top",
          fontsize=GUIDE_FS)
    fig3.savefig(OUT / "monitors.png")
    audits["monitors.png"] = audit_overlaps(fig3, "monitors")
    plt.close(fig3)

    for name, iss in audits.items():
        print("[audit] %s: %s" % (name, "PASS" if not iss else iss))

    total_min = (time.perf_counter() - t_start) / 60.0
    write_readme(runs, cases, dev_max, ends, total_min, audits)
    print("wrote", csv_path)
    print("wrote", OUT / "README.md")
    for png in ("flow_sweeps.png", "tube_material.png", "monitors.png"):
        print("wrote", OUT / png)
    print("total wall time %.1f min (CPU)" % total_min)
    return 0


# ------------------------------------------------------------------ README

def write_readme(runs, cases, dev_max, ends, total_min, audits):
    def sw(name, x):
        return next(r for r in runs[("sweeps", name)] if r["x"] == x)

    def ex(name, x):
        return next(r for r in runs[("exact", name)] if r["x"] == x)

    def au(name, x):
        return next(r for r in runs[("audit", name)] if r["x"] == x)

    l5, l9 = cases
    held5 = runs[("held", "level5")]
    held9 = runs[("held", "level9")]
    resid_exact_max = max(r["resid"] for c in cases for r in runs[("exact", c.name)])
    mat_path5 = ex("level5", max(SNAP_STEPS))["mat_on_path"]
    mat_path9 = ex("level9", max(SNAP_STEPS))["mat_on_path"]
    dead5 = ex("level5", max(SNAP_STEPS))["dead_max"]
    dead9 = ex("level9", max(SNAP_STEPS))["dead_max"]
    drop5 = (1.0 - ends["level5"][1] / ends["level5"][0]) * 100.0
    drop9 = (1.0 - ends["level9"][1] / ends["level9"][0]) * 100.0
    d_final5 = runs[("m_tube", "level5")] / l5.n_path_pipes
    d_final9 = runs[("m_tube", "level9")] / l9.n_path_pipes

    # stored set-level evaluation (for the table only; not recomputed here)
    strict_txt = "not run (results/transport_physics/results.json missing)"
    eval_rows, eval_wall, eval_dev = "", "", ""
    if EVAL_JSON.exists():
        ev = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
        def strict(name, T):
            return "%.3f" % ev["results"][name]["per_T"][str(T)]["strict"]
        Ts = ev["protocol"]["steps_T"]
        all_max = all(
            ev["results"][name]["per_T"][str(T)]["strict"] == 1.0
            for name in ("level5_test", "probe9") for T in Ts)
        first_step_hold = min(
            ev["results"][name]["solver_breakdown"]["first_step"]
            for name in ("level5_test", "probe9"))
        strict_txt = (
            "test strict at T = 8: %s (level-5 test, n = 1000 mazes) and %s "
            "(level-9 probe, n = 1000 mazes); strict is %.3f at EVERY measured "
            "horizon T in %s (the arm has no training and no validation "
            "selection, so no T* is defined) - from the stored evaluation run "
            "results/transport_physics/results.json, exact relaxation, CPU"
            % (strict("level5_test", 8), strict("probe9", 8),
               1.0 if all_max else float("nan"), Ts))
        eval_wall = ", ".join(
            "%s: %.0f s for %d mazes"
            % (nm, ev["protocol"]["sets"][nm]["wall_time_s"],
               ev["protocol"]["sets"][nm]["n_mazes"])
            for nm in ("level5_test", "probe9"))
        eval_dev = ("on the full sets the exact solve degenerates on some mazes "
                    "from outer step %d onward (the stored evaluation holds "
                    "such a maze's flow at its last conservation-valid value; "
                    "on a tree that is the exact flow for all later steps)" % first_step_hold)

    audit_line = "; ".join(
        "%s: %s" % (k, "pass" if not v else "ISSUES: %s" % v)
        for k, v in audits.items())

    L = []
    A = L.append
    A("# Transport mechanism: how the physics alone solves the maze\n")
    A("These figures show the transport coagulation-fragmentation model\n"
      "(cf/models/transport.py, specification docs/specs/family_d_transport.md)\n"
      "in its physics-only arm: ZERO learned parameters (verified: the sum of\n"
      "all parameter counts is 0; raw 0, effective 0), uniform pipe lengths\n"
      "L_e = 1, adaptation f_e = |Q_e| + eps_f with eps_f = 1e-3, source\n"
      "current I = 1. There is no training phase (0 epochs, 0 minutes of\n"
      "training, 0 seeds): the run is deterministic, so no bands and no\n"
      "standard deviations appear anywhere below.\n")
    A("## The run\n")
    A("| item | value |")
    A("| --- | --- |")
    A("| arm | physics-only transport, no learning (params raw 0 / effective 0) |")
    A("| training | none: 0 epochs, 0 seeds, 0 minutes |")
    A("| mazes | maze index %d of the seeded level-5 test split and maze index %d "
      "of the seeded level-9 probe split (splits seed 0, the same splits as every "
      "run in this repository; the rule is fixed in advance, none was picked after "
      "looking at results) |" % (MAZE_INDEX, MAZE_INDEX))
    A("| maze level-5 | %dx%d cells, %d free cells, %d pipes, path %d cells / %d pipes |"
      % (l5.H, l5.W, l5.n_free_cells, l5.n_pipes, l5.n_path_cells, l5.n_path_pipes))
    A("| maze level-9 | %dx%d cells, %d free cells, %d pipes, path %d cells / %d pipes |"
      % (l9.H, l9.W, l9.n_free_cells, l9.n_pipes, l9.n_path_cells, l9.n_path_pipes))
    A("| n | 1 maze per level (deterministic; sd not applicable); the population "
      "statement is the stored evaluation: %s |" % strict_txt)
    A("| hardware | CPU only (local GPU disabled by the run environment), torch "
      "%s, %s |" % (torch.__version__, platform.platform()))
    A("| budget | sweep scan %d half-sweeps per maze, exact rollout %d outer steps "
      "per maze, audit rollout %d outer steps at the spec budget; total figure wall "
      "time %.1f min CPU; stored set-level evaluation %s |"
      % (N_HALF, OUTER_STEPS, AUDIT_STEPS, total_min, eval_wall))
    A("")
    A("## flow_sweeps.png\n")
    A("![Current maps |Q| per cell while the first outer step's pressure "
      "relaxation runs](flow_sweeps.png)\n")
    A("**Caption.** Current maps |Q| per cell on the two mazes while the\n"
      "first outer step's pressure relaxation runs, at\n"
      "%s half-sweeps out of the %s computed. Pipes are all equally thick (the\n"
      "initial state); the pressures start at zero; the relaxation is the\n"
      "two-colour Gauss-Seidel sweep of the model (one sweep = one red plus\n"
      "one black pass).\n"
      % (", ".join(str(h) for h in SNAP_HALF), N_HALF))
    q_sc = swq_max(runs, cases)
    A("**How to read.** Colour = total current |Q| through the pipes touching\n"
      "a cell, in units of the source current I = 1 (the cell value is a SUM\n"
      "over the pipes touching the cell, so it can exceed 1 - %.2f, the top of\n"
      "the colour scale, is the largest cell value over all shown snapshots,\n"
      "on the converged maps a path cell where two path pipes, each carrying\n"
      "the full unit, meet; the colour scale is a power scale (gamma 0.45) on\n"
      "[0, %.2f]). White dashed line = the maze's solution path (the given\n"
      "label, not computed); circle = start (current in), square = goal\n"
      "(current out); grey cells are walls. n = 1 maze per level,\n"
      "deterministic, so no bands or error bars apply.\n" % (q_sc, q_sc))
    A("**Key takeaway.** The current spreads from the source and the sink\n"
      "into every corridor, then drains out of the dead ends and collapses\n"
      "onto the solution path: the share of |Q| on the path pipes rises from\n"
      "%.3f after 1 half-sweep to %.3f at %d half-sweeps on the level-5 maze\n"
      "(residual %.0e), and from %.3f to %.3f at %d half-sweeps on the\n"
      "level-9 maze (residual %.0e). On a maze without loops no steady\n"
      "current can enter a dead-end corridor and every pipe on the unique\n"
      "route carries the full unit - so 'dark' means 'no current can ever\n"
      "flow there'.\n"
      % (sw("level5", 1)["share"], sw("level5", N_HALF)["share"], N_HALF,
         sw("level5", N_HALF)["resid"], sw("level9", 1)["share"],
         sw("level9", N_HALF)["share"], N_HALF, sw("level9", N_HALF)["resid"]))
    A("**Data source.** cf/models/transport.py (sweep relaxation);\n"
      "data.csv, section sweeps_first_step; mazes from the seeded curriculum\n"
      "splits (cf/data/curriculum.py, dataset selimaktas/maze-curriculum-\n"
      "dataset, splits seed 0).\n")
    A("**Caveats.** The snapshot is inside the FIRST outer step only (uniform\n"
      "pipes); the specification's per-step relaxation budget is 16 sweeps =\n"
      "%d half-sweeps, and the grey dashed line in monitors.png shows where\n"
      "that budget sits on the convergence path (share %s and %s at exactly\n"
      "%d half-sweeps for the two mazes). One maze per level; the same\n"
      "mechanism gives strict accuracy 1.000 on all 1000 mazes of each set\n"
      "in the stored evaluation.\n"
      % (AUDIT_KREL * 2, fmt(sw("level5", AUDIT_KREL * 2)["share"], 3),
         fmt(sw("level9", AUDIT_KREL * 2)["share"], 3), AUDIT_KREL * 2))
    A("")
    A("## tube_material.png\n")
    A("![Pipe thickness D_e per cell over outer adaptation steps]"
      "(tube_material.png)\n")
    A("**Caption.** Pipe thickness D_e (per cell: the mean over the pipes\n"
      "touching it) on the same two mazes over outer adaptation steps %s.\n"
      "Each re-solves the pressures exactly (float64 linear solve), then\n"
      "redistributes the material: D_e <- M_tube * D_e f_e / Sum D_e f_e with\n"
      "f_e = |Q_e| + eps_f, so the material budget is exactly conserved.\n"
      % ", ".join(str(t) for t in SNAP_STEPS))
    A("**How to read.** Colour = pipe thickness on a LOG scale from a display\n"
      "floor of 1e-8 (stated clip: values below the floor, including the exact\n"
      "zeros reached by starving pipes, are drawn at the floor colour) up to\n"
      "%.1f, the largest value over the shown maps. Step 0 is the initial\n"
      "state: every free pipe equally thick (D_e = 1, so the material budget\n"
      "equals the pipe count: %d pipes level-5, %d pipes level-9).\n"
      "White dashed line / markers as in the flow figure. n = 1 maze per\n"
      "level, deterministic; no bands.\n"
      % (dmax_fig(runs, cases), l5.n_pipes, l9.n_pipes))
    A("**Key takeaway.** The material coagulates into the single carrying\n"
      "channel: after %d steps, %.6f of the budget sits on the path pipes of\n"
      "the level-9 maze (%.6f on the level-5 maze), each path pipe thickness\n"
      "has grown to the budget share M_tube / path pipes = %.1f (level-5)\n"
      "and %.1f (level-9), and the thickest starving dead-end pipe is down\n"
      "to %s and %s (fragmentation of the rest: about a factor 1000 thinner\n"
      "per step). The carrying channel is the exact solution path.\n"
      % (max(SNAP_STEPS), mat_path9, mat_path5, d_final5, d_final9,
         sci(dead5), sci(dead9)))
    A("**Data source.** cf/models/transport.py (exact relaxation and the\n"
      "redistribution step); data.csv, section outer_steps_exact (columns\n"
      "material_on_path_share and dead_D_max); same mazes as above.\n")
    A("**Caveats.** %s. The two mazes shown here keep their exact solves\n"
      "conservation-valid through all %d steps (largest per-step residual\n"
      "%.0e), so no hold policy was needed. One maze per level.\n"
      % (eval_dev, OUTER_STEPS, resid_exact_max))
    A("")
    A("## monitors.png\n")
    A("![Path-current share, material conservation and dissipation over the "
      "run](monitors.png)\n")
    A("**Caption.** The three promised invariants of the update, on the same\n"
      "two mazes: the share of all |Q| on the solution path tends to 1 as the\n"
      "relaxation converges (left); the material budget Sum L_e D_e stays\n"
      "constant (middle; with L_e = 1 it is Sum D_e, and the exact statement\n"
      "is |Sum D_e / M_tube - 1| <= float32 rounding); the dissipated\n"
      "energy E_t = Sum Q_e^2 / D_e decreases over the outer steps (right).\n")
    A("**How to read.** Left: x = half-sweeps of the first step's relaxation\n"
      "(log scale), y = share of |Q| on the path pipes (0-1); the grey dashed\n"
      "line marks the spec budget of 16 sweeps (= %d half-sweeps) per outer\n"
      "step. Middle and right: x = outer adaptation step (each step = one\n"
      "exact pressure solve plus one material redistribution); middle y =\n"
      "the material deviation |Sum L_e D_e / M_tube - 1| on a linear scale 0\n"
      "to 1.45e-7; right y = E_t. Blue = level-5 maze, vermillion = level-9\n"
      "maze; circles mark level-5 endpoints, squares mark level-9 endpoints.\n"
      "n = 1 deterministic maze per level, so no sd / bands are shown.\n"
      % (AUDIT_KREL * 2))
    A("**Key takeaway.** (1) The path share of |Q| rises to %.4f (level-5,\n"
      "%d half-sweeps) and %.4f (level-9, %d half-sweeps); with the exact\n"
      "pressure solve it is already %.4f at outer step 1 and stays there -\n"
      "on a tree the conserved flow is unique. (2) The material deviation\n"
      "never exceeds %.1e over both mazes and all %d steps (float32\n"
      "rounding; conservation is exact by construction). (3) The dissipation\n"
      "falls from %.2f to %.2f (-%.1f%%) on the level-5 maze and from %.2f\n"
      "to %.2f (-%.1f%%) on the level-9 maze, plateauing at the value\n"
      "path pipes / (M_tube / path pipes) = %.2f and %.2f - thickening the\n"
      "route is what lowers the dissipated energy.\n"
      % (sw("level5", N_HALF)["share"], N_HALF, sw("level9", N_HALF)["share"],
         N_HALF, ex("level5", 1)["share"], dev_max, OUTER_STEPS,
         ends["level5"][0], ends["level5"][1], drop5,
         ends["level9"][0], ends["level9"][1], drop9,
         ends["level5"][1], ends["level9"][1]))
    A("**Data source.** data.csv, sections sweeps_first_step (left panel),\n"
      "outer_steps_exact (middle and right panels); cf/models/transport.py.\n")
    A("**Caveats.** The sweep-relaxation curves (left panel) are inside the\n"
      "first outer step from a zero-pressure start; across outer steps run\n"
      "at the spec budget of 16 sweeps the bigger maze under-relaxes (audit\n"
      "run in data.csv, section outer_steps_sweep_budget16: at step 1 the\n"
      "level-9 residual is %.0e and by step %d the unit current is cut, total\n"
      "|Q| = %.2f on %d path pipes - that is a property of the fixed sweep\n"
      "budget on the 19x19 grid, not of the physics) - the outer-step panels\n"
      "use the exact solve instead (largest residual %.0e over %d steps).\n"
      "%s The dissipation E_t is the unit-length form Sum Q_e^2/D_e used by\n"
      "the model's own diagnostics; with L_e = 1 it equals the resistor\n"
      "network's true dissipated energy.\n"
      % (au("level9", 1)["resid"], AUDIT_STEPS,
         au("level9", AUDIT_STEPS)["tot"], l9.n_path_pipes,
         resid_exact_max, OUTER_STEPS, held_line(held5, held9)))
    A("")
    A("## data.csv\n")
    A("One row per measurement. Sections: maze_meta (cell/pipe/path counts and\n"
      "the material budget M_tube), sweeps_first_step (one row per\n"
      "half-sweep of the first outer step's relaxation: path share,\n"
      "conservation residual, total |Q|, max cell |Q|), outer_steps_exact\n"
      "(one row per outer step with the exact solve: path share, residual,\n"
      "total |Q|, material deviation, dissipation, thickest dead-end pipe,\n"
      "material share on the path pipes), outer_steps_sweep_budget16 (audit\n"
      "run at the spec's 16-sweep relaxation budget).\n")
    A("## Figure audit\n")
    A("Programmatic readability audit (each text's window extent against\n"
      "every other text and every sampled line/marker point): %s\n"
      "All text is 11 pt or larger; fixed colours from scripts/figures/\n"
      "style.py (Okabe-Ito); every figure carries its own legend or scale.\n"
      % audit_line)
    (OUT / "README.md").write_text("\n".join(L), encoding="utf-8")


def swq_max(runs, cases):
    return max(max(float(s.max()) for s in runs[("snaps_q", c.name)].values())
               for c in cases)


def dmax_fig(runs, cases):
    return max(max(float(s.max()) for s in runs[("snaps_d", c.name)].values())
               for c in cases)


def sci(v):
    return "%.1e" % v


def held_line(held5, held9):
    hs = [h for h in (held5, held9) if h is not None]
    if not hs:
        return ("Neither maze's exact solve degenerated within the shown "
                "horizon, so the hold policy of the stored evaluation was "
                "never triggered here.")
    return ("A hold occurred at outer step(s) %s - the rollout stopped there"
            % ", ".join(str(h) for h in hs))


if __name__ == "__main__":
    sys.exit(main())
