"""Concept figures for the handover packet: F1 (worked Stage-A step) and
F2 (pipeline diagram).

F1  One Stage-A reaction step 2A1 <=> A2 on a hand-set 3x3 grid (1 species):
    c1, c2 before the step, the merge flux m and split flux s, and c1, c2, M
    after the step -- computed by the real update, CFNet.step in
    cf/models/cf.py, with the rate network pinned to constant rates.
F2  The Stage-A pipeline as boxes and arrows: maze -> encoder -> c1(0), c2(0)
    -> T x (rate net -> m, s -> update) -> readout -> path. The maze and the
    path are a real sample from maze_cf.make_sample (fixed seed).

Writes figures/concept/{f1_stage_a_step.png, f2_pipeline.png, data.csv,
README.md}.

Run:  python scripts/figures/concept.py [--root REPO_ROOT]

Both figures assert at render time that no two text bounding boxes overlap,
that no text sits on an arrow, and (F1) that every cell value fits inside its
grid cell.
"""

import argparse
import csv
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                      # cf.models.cf, maze_cf
sys.path.insert(0, str(Path(__file__).resolve().parent))  # style
from style import DPI, OKABE_ITO, apply_style       # noqa: E402

from cf.models.cf import CFNet                     # noqa: E402
from maze_cf import bfs_distances, make_sample     # noqa: E402

torch.set_grad_enabled(False)

# hand-set initial state (1 species, 3x3 grid); the centre cell is the
# packet's worked example (c1 = 10, c2 = 0, M = 10)
C1_0 = np.array([[2.0, 5.0, 3.0],
                 [4.0, 10.0, 6.0],
                 [1.0, 8.0, 2.0]])
C2_0 = np.array([[0.0, 1.0, 0.0],
                 [2.0, 0.0, 1.0],
                 [0.0, 3.0, 0.0]])
MERGE_RATE = 0.6   # sigma(raw_merge), constant over cells
SPLIT_RATE = 0.2   # sigma(raw_split), constant over cells

MAZE_SEED = 0
MAZE_SIZE = 7


def logit(p):
    return float(np.log(p / (1.0 - p)))


def tint(color, f):
    """Blend the Okabe-Ito colour toward white by factor f in [0, 1]."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(255 - f * (255 - v)) for v in (r, g, b))


def cmap_of(color):
    return LinearSegmentedColormap.from_list("q", ["#ffffff", tint(color, 0.75)])


def fmt(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


# ----------------------------------------------------------------------------
# overlap checks
# ----------------------------------------------------------------------------
def _collect_texts(fig):
    arts = list(fig.texts)
    for ax in fig.axes:
        if getattr(ax, "axison", True):             # skip axis-off panels
            arts += [ax.title, ax.xaxis.label, ax.yaxis.label]
            arts += ax.get_xticklabels() + ax.get_yticklabels()
        arts += ax.texts
        leg = ax.get_legend()
        if leg is not None:
            arts += list(leg.get_texts())
    for leg in fig.legends:
        arts += list(leg.get_texts())
        if leg.get_title().get_text().strip():
            arts.append(leg.get_title())
    return [t for t in arts
            if t is not None and t.get_text().strip() and t.get_visible()]


def _densify(pts, step=4.0):
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / step))
        out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
                for i in range(n + 1)]
    return out


def assert_no_overlaps(fig, name, flow_artists=(), cell_texts=None):
    """No text-text overlap, no text on an arrow, and (F1) every cell value
    inside its own grid cell."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    labelled = [(t.get_text().split("\n")[0][:24], t.get_window_extent(ren))
                 for t in _collect_texts(fig)]
    for (t1, b1), (t2, b2) in combinations(labelled, 2):
        if b1.overlaps(b2):
            raise AssertionError(f"{name}: text {t1!r} overlaps text {t2!r}")
    for art in flow_artists:                       # arrows: no text on them
        tr = art.get_transform()
        for poly in art.get_path().to_polygons():
            for x, y in _densify([(p[0], p[1]) for p in tr.transform(poly)]):
                for txt, bb in labelled:
                    if bb.x0 - 1 <= x <= bb.x1 + 1 and bb.y0 - 1 <= y <= bb.y1 + 1:
                        raise AssertionError(
                            f"{name}: text {txt!r} sits on an arrow")
    for txt, bb_cell in cell_texts or []:          # cell values inside cells
        bb_cell = bb_cell.transformed(txt.axes.transData)
        bb = txt.get_window_extent(ren)
        if not (bb_cell.x0 - 1 <= bb.x0 and bb.x1 <= bb_cell.x1 + 1
                and bb_cell.y0 - 1 <= bb.y0 and bb.y1 <= bb_cell.y1 + 1):
            raise AssertionError(
                f"{name}: cell value {txt.get_text()!r} overflows its cell")


# ----------------------------------------------------------------------------
# F1: one Stage-A step on a 3x3 grid
# ----------------------------------------------------------------------------
def build_pinned_model():
    """CFNet with the rate network pinned to constant rates, so the real
    step() update runs on hand-set numbers."""
    model = CFNet(in_ch=3, hid=1, fragmentation=True, bound_flux=True)
    with torch.no_grad():
        for p in model.rate_net.parameters():
            p.zero_()
        model.rate_net[2].bias.copy_(
            torch.tensor([logit(MERGE_RATE), logit(SPLIT_RATE)]))
    return model


def draw_grid(ax, arr, color, vmax, title, cell_texts, highlight=False):
    n = arr.shape[0]
    ax.pcolormesh(np.arange(n + 1), np.arange(n, -0.01, -1), arr,
                  cmap=cmap_of(color), vmin=0.0, vmax=vmax,
                  edgecolors="white", linewidth=1.6)
    for i in range(n):
        for j in range(n):
            t = ax.text(j + 0.5, n - i - 0.5, fmt(arr[i, j]),
                        ha="center", va="center", fontsize=13)
            cell_texts.append((t, Bbox(np.array([[j, n - i - 1],
                                                 [j + 1, n - i]]))))
    if highlight:
        ax.add_patch(Rectangle((1, 1), 1, 1, fill=False,
                               edgecolor="black", linewidth=2.2))
    ax.set_title(title)
    # ticks at cell centres, labelled by cell index (row 0 at the top)
    ax.set_xticks([j + 0.5 for j in range(n)],
                  labels=[str(j) for j in range(n)])
    ax.set_yticks([n - i - 0.5 for i in range(n)],
                  labels=[str(i) for i in range(n)])
    ax.set_xlabel("cell column")
    ax.set_ylabel("cell row")
    ax.set_aspect("equal")


def make_f1(out, rows):
    model = build_pinned_model()
    x = torch.zeros(1, 3, 3, 3)
    x[:, 0] = 1.0                                   # all cells walkable
    c1 = torch.tensor(C1_0, dtype=torch.float32).reshape(1, 1, 3, 3)
    c2 = torch.tensor(C2_0, dtype=torch.float32).reshape(1, 1, 3, 3)
    c1n, c2n, m, s, g = model.step(x, c1, c2)

    c1_1 = c1n[0, 0].numpy()
    c2_1 = c2n[0, 0].numpy()
    m_a = m[0, 0].numpy()
    s_a = s[0, 0].numpy()
    M_0 = C1_0 + 2.0 * C2_0
    M_1 = c1_1 + 2.0 * c2_1
    drift = float(np.abs(M_1 - M_0).max())

    # the update must match the hand-computed reaction exactly
    np.testing.assert_allclose(m_a, 0.5 * MERGE_RATE * C1_0, atol=1e-6)
    np.testing.assert_allclose(s_a, SPLIT_RATE * C2_0, atol=1e-6)
    np.testing.assert_allclose(c1_1, C1_0 - 2 * m_a + 2 * s_a, atol=1e-6)
    np.testing.assert_allclose(c2_1, C2_0 + m_a - s_a, atol=1e-6)
    assert drift < 1e-5, drift
    # centre cell reproduces the packet's worked example
    assert abs(c1_1[1, 1] - 4.0) < 1e-6 and abs(c2_1[1, 1] - 3.0) < 1e-6
    assert abs(M_1[1, 1] - 10.0) < 1e-5

    for k, v in (("merge_rate", MERGE_RATE), ("split_rate", SPLIT_RATE),
                 ("M_total_before", float(M_0.sum())),
                 ("M_total_after", float(M_1.sum())),
                 ("max_abs_mass_drift_per_cell", drift)):
        rows.append(("F1", k, repr(v)))
    for name, a in (("c1_before", C1_0), ("c2_before", C2_0), ("m", m_a),
                    ("s", s_a), ("c1_after", c1_1), ("c2_after", c2_1),
                    ("M_before", M_0), ("M_after", M_1)):
        for i in range(3):
            for j in range(3):
                rows.append(("F1", f"{name}[{i},{j}]", repr(float(a[i, j]))))

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12.6))
    fig.subplots_adjust(hspace=0.42)
    cell_texts = []
    draw_grid(axes[0, 0], C1_0, OKABE_ITO["sky_blue"], C1_0.max(),
              "c1 before the step", cell_texts, highlight=True)
    draw_grid(axes[0, 1], C2_0, OKABE_ITO["bluish_green"], C2_0.max(),
              "c2 before the step", cell_texts, highlight=True)
    draw_grid(axes[0, 2], M_0, OKABE_ITO["yellow"], M_0.max(),
              "M = c1 + 2 c2 before", cell_texts, highlight=True)
    draw_grid(axes[1, 0], m_a, OKABE_ITO["vermillion"], m_a.max(),
              "merge flux m", cell_texts)
    draw_grid(axes[1, 1], s_a, OKABE_ITO["reddish_purple"], max(s_a.max(), 0.1),
              "split flux s", cell_texts)
    # after-panels are scaled to their own maximum, so no cell saturates
    draw_grid(axes[2, 0], c1_1, OKABE_ITO["sky_blue"], c1_1.max(),
              "c1 after one step", cell_texts, highlight=True)
    draw_grid(axes[2, 1], c2_1, OKABE_ITO["bluish_green"], c2_1.max(),
              "c2 after one step", cell_texts, highlight=True)
    draw_grid(axes[2, 2], M_1, OKABE_ITO["yellow"], M_1.max(),
              "M = c1 + 2 c2 after", cell_texts, highlight=True)

    ax = axes[1, 2]
    ax.set_axis_off()
    ax.text(0.5, 0.97,
            "one CFNet.step (cf/models/cf.py):\n"
            "m = 1/2 c1 sigma(merge),  s = c2 sigma(split)\n"
            "c1 <- c1 - 2 m + 2 s\n"
            "c2 <- c2 + m - s\n"
            "rates here: sigma(merge) = 0.6, sigma(split) = 0.2",
            ha="center", va="top", fontsize=12)
    ax.text(0.5, 0.52,
            "highlighted centre cell = the\npacket's worked example:\n"
            "c1 = 10, c2 = 0  ->  m = 3, s = 0\n"
            "->  c1 = 4, c2 = 3,  M = 10",
            ha="center", va="top", fontsize=12)

    fig.suptitle("Stage-A reaction 2 A1 <=> A2: one step on a 3x3 grid "
                 "(1 species, hand-set numbers)", fontsize=15)
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bb = Bbox.union([a.get_tightbbox(ren) for a in axes.flat])
    xf, yf = fig.transFigure.inverted().transform((bb.x0, bb.y0))
    handles = [Patch(facecolor=tint(c, 0.55), edgecolor=c, label=lab)
               for c, lab in ((OKABE_ITO["sky_blue"], "c1 monomers"),
                             (OKABE_ITO["bluish_green"], "c2 dimers"),
                             (OKABE_ITO["yellow"], "M mass"),
                             (OKABE_ITO["vermillion"], "m merge flux"),
                             (OKABE_ITO["reddish_purple"], "s split flux"))]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(xf, yf - 0.034),
               ncol=5, fontsize=10, frameon=False, title="panel colours",
               title_fontsize=10)
    fig.canvas.draw()
    leg_bb = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
    lx, ly = fig.transFigure.inverted().transform((leg_bb.x0, leg_bb.y0))
    fig.text(lx, ly - 0.008,
             f"Computed with CFNet.step on hand-set c1, c2 (float32); rates "
             f"constant over cells; total mass {fmt(M_0.sum())} before and "
             f"{fmt(M_1.sum())} after, max per-cell drift {drift:.1e}.",
             fontsize=10, color="#444444", ha="left", va="top")
    return fig, cell_texts


# ----------------------------------------------------------------------------
# F2: pipeline diagram
# ----------------------------------------------------------------------------
def maze_rgb(grid, start, goal, mask):
    rgb = np.ones(grid.shape + (3,))
    wall = np.array([0.55, 0.55, 0.55])
    rgb[grid == 1] = wall
    if mask is not None:
        col = np.array(matplotlib.colors.to_rgb(OKABE_ITO["bluish_green"]))
        rgb[mask] = 0.35 * col + 0.65 * np.ones(3)
    rgb[start] = matplotlib.colors.to_rgb(OKABE_ITO["sky_blue"])
    rgb[goal] = matplotlib.colors.to_rgb(OKABE_ITO["vermillion"])
    return rgb


def make_f2(out, rows):
    rng = np.random.default_rng(MAZE_SEED)
    grid, start, goal, mask = make_sample(rng, MAZE_SIZE)
    d = bfs_distances(grid, start)[goal]
    rows.append(("F2", "maze_size", str(MAZE_SIZE)))
    rows.append(("F2", "maze_seed", str(MAZE_SEED)))
    rows.append(("F2", "start", str(tuple(start))))
    rows.append(("F2", "goal", str(tuple(goal))))
    rows.append(("F2", "shortest_path_length", str(int(d))))
    rows.append(("F2", "path_cells_in_mask", str(int(mask.sum()))))
    rows.append(("F2", "T_train_40epoch_runs", "8"))

    fig, ax = plt.subplots(figsize=(16.0, 5.6))
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 40)
    ax.set_axis_off()

    # square extent per maze (14 x 14 units for 7 x 7 cells -> square cells,
    # imshow pins the aspect to equal)
    ax.imshow(maze_rgb(grid, start, goal, None),
              extent=[2, 16, 13, 27], interpolation="nearest")
    ax.imshow(maze_rgb(grid, start, goal, mask),
              extent=[124, 138, 13, 27], interpolation="nearest")
    ax.text(9, 12, "input maze x (7 x 7)", ha="center", va="top", fontsize=11)
    ax.text(131, 12, "path (readout target)", ha="center", va="top",
            fontsize=11)

    def box(x0, x1, y0, y1, text):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.4,rounding_size=1.2",
                                    facecolor="#f5f5f5", edgecolor="black",
                                    linewidth=1.2))
        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), text,
                ha="center", va="center", fontsize=11)

    box(20, 38, 15, 25, "encoder\nConv 3x3 + softplus")
    box(42, 63, 14, 26, "initial state\nc1(0) = softplus(enc(x))\nc2(0) = 0")
    box(66, 92, 10, 30,
        "T x CF step\n(T = 8 at training time)\nrate net -> m, s\n"
        "c1 <- c1 - 2 m + 2 s\nc2 <- c2 + m - s")
    box(95, 114, 15, 25, "readout\nConv 1x1 on [c1, c2]")
    ax.text(79, 7.5, "M = c1 + 2 c2 is conserved at every step",
            ha="center", va="top", fontsize=11, color="#444444")

    arrows = []
    for x0, x1 in ((16.5, 19.5), (38.5, 41.5), (63.5, 65.5),
                   (92.5, 94.5), (114.5, 123.5)):
        a = FancyArrowPatch((x0, 20), (x1, 20), arrowstyle="-|>",
                            mutation_scale=22, linewidth=1.6,
                            color="black", shrinkA=0, shrinkB=0)
        ax.add_patch(a)
        arrows.append(a)

    handles = [Patch(facecolor="#8c8c8c", edgecolor="#555555", label="wall"),
               Patch(facecolor="white", edgecolor="#555555", label="free cell"),
               Patch(facecolor=OKABE_ITO["sky_blue"], label="start"),
               Patch(facecolor=OKABE_ITO["vermillion"], label="goal"),
               Patch(facecolor=tint(OKABE_ITO["bluish_green"], 0.35),
                     edgecolor=OKABE_ITO["bluish_green"],
                     label="shortest-path cell")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.02),
              ncol=5, fontsize=10, frameon=False)

    ax.set_title("Stage-A CF pipeline: maze in, path out "
                 "(CF-Full, hid = 16, trained at fixed T = 8)",
                 fontsize=15, pad=14)
    return fig, arrows


# ----------------------------------------------------------------------------
def write_readme(out, n):
    M0, M1 = n["M_total_before"], n["M_total_after"]
    readme = f"""# Concept figures (packet F1, F2)

Generated by `scripts/figures/concept.py`; every number below is in
`data.csv` next to the PNGs.

## f1_stage_a_step.png

**Caption.** One Stage-A coagulation-fragmentation step (reaction
2 A1 <=> A2) on a hand-set 3x3 grid with a single species: the monomer
concentration c1, dimer concentration c2 and mass M = c1 + 2 c2 before the
step (top row), the merge flux m and split flux s the rate network produces
(middle row), and the state after one step (bottom row). The step is computed
by the real update, `CFNet.step` in `cf/models/cf.py`, with the rate network
pinned to constant rates sigma(merge) = {MERGE_RATE} and
sigma(split) = {SPLIT_RATE}.

**How to read.** Each panel is the 3x3 grid; the number in a cell is that
cell's value. Colour shade grows with the value (white = 0); every panel's
shade is scaled to that panel's own largest value, so shades compare values
within a panel, not across panels. The legend at
the bottom of the figure maps panel colours to quantities: sky blue = c1,
green = c2, yellow = M, orange = m, purple = s. Axis ticks sit at cell
centres and label the cell index (row 0 at the top). The black square marks
the centre cell, which is the packet's worked example. The middle-right panel
states the update equations that produced the bottom row from the top row.

**Key takeaway.** Mass is exactly conserved by the update: total
M = {M0:.1f} before the step and {M1:.1f} after (max per-cell drift
{n['max_abs_mass_drift_per_cell']:.1e} in float32), while the merge flux
moves mass from loose to committed: the centre cell goes from
c1 = 10, c2 = 0 to c1 = 4.0, c2 = 3.0 with m = 3.0, s = 0.0 and keeps
M = 10.0. The 1/2 bound on the merge flux keeps c1 >= 0 (here
m = 0.5 x 0.6 x c1 <= 0.3 c1 < 0.5 c1).

**Data source.** `cf/models/cf.py` (`CFNet.step`, `CFNet.fluxes`); the
hand-set inputs c1, c2 and all computed values are in `data.csv` (keys
`c1_before[*]`, `c2_before[*]`, `m[*]`, `s[*]`, `c1_after[*]`,
`c2_after[*]`, `M_before[*]`, `M_after[*]`). Script:
`scripts/figures/concept.py`.

**Caveats.** c1, c2 and the two rates are hand-set for illustration (not
learned, not from the encoder); one species (hid = 1) instead of the trained
model's 16; rates are constant over cells. No aggregation over seeds or
samples is involved, so there are no error bars or bands; the conservation
number is exact up to float32 rounding (drift stated above).

Packet figure: F1

## f2_pipeline.png

**Caption.** The Stage-A CF pipeline as a diagram: the input maze x (3
channels: walkable, start, goal) goes through a 3x3 convolution and a
softplus (encoder) that sets the initial monomer concentration c1(0) with
c2(0) = 0; T repetitions of the CF step (rate network produces the merge
flux m and split flux s, then the reaction update
c1 <- c1 - 2m + 2s, c2 <- c2 + m - s); a 1x1 convolution readout maps the
final [c1, c2] to one path logit per cell. The maze and the highlighted path
are a real sample from the project's generator (seed {MAZE_SEED},
{MAZE_SIZE} x {MAZE_SIZE}).

**How to read.** Left to right in processing order; arrows show the flow of
the single forward pass. The two images are the input maze and the
shortest-path mask the readout is trained to produce; the legend maps the
colours (grey = wall, white = free, blue = start, orange = goal, green =
shortest-path cell).

**Key takeaway.** The network only learns the encoder (initial mass), the
rate network (when to merge and split) and the readout; the reaction update
itself is fixed and conserves M = c1 + 2 c2 at every one of the T steps
(T = 8 at training time in the 40-epoch runs). The example maze's start
and goal are {n['shortest_path_length']} BFS steps apart ({n['path_cells_in_mask']}
cells in the ground-truth shortest-path mask the readout is trained on, start
and goal included).

**Data source.** `maze_cf.py` (`make_sample`, `bfs_distances`,
`shortest_path_mask`) for the maze and path; `cf/models/cf.py` for the stage
names and shapes; `week2_results/valT_selection.json` (`T_train` = 8) for
the training depth. Maze facts are in `data.csv` (keys `maze_size`,
`maze_seed`, `start`, `goal`, `shortest_path_length`,
`path_cells_in_mask`).

**Caveats.** The diagram shows the CF-Full variant (merge + split,
hid = 16); other variants differ as follows: CF-NoFrag has s = 0,
CF-WallAware masks walls, CF-NoBound drops the 1/2 bound on m. The path
shown is the ground-truth shortest-path mask, not a trained model's
prediction. No statistics are aggregated, so there are no error bars.

Packet figure: F2
"""
    (out / "README.md").write_text(readme, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    apply_style()
    out = args.root / "figures" / "concept"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    fig1, cell_texts = make_f1(out, rows)
    assert_no_overlaps(fig1, "f1", cell_texts=cell_texts)
    fig1.savefig(out / "f1_stage_a_step.png")
    plt.close(fig1)

    fig2, arrows = make_f2(out, rows)
    assert_no_overlaps(fig2, "f2", flow_artists=arrows)
    fig2.savefig(out / "f2_pipeline.png")
    plt.close(fig2)

    with open(out / "data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["figure", "key", "value"])
        w.writerows(rows)

    nums = {k: float(v) for _, k, v in rows
            if k in ("M_total_before", "M_total_after",
                     "max_abs_mass_drift_per_cell")}
    nums["shortest_path_length"] = next(
        int(v) for fk, k, v in rows if k == "shortest_path_length")
    nums["path_cells_in_mask"] = next(
        int(v) for fk, k, v in rows if k == "path_cells_in_mask")
    write_readme(out, nums)
    print(f"wrote {out / 'f1_stage_a_step.png'}, "
          f"{out / 'f2_pipeline.png'}, data.csv, README.md")


if __name__ == "__main__":
    main()
