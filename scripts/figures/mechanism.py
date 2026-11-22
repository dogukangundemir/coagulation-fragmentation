"""Packet figures F3 (c1(0) embedding of CF-Full) and F5 (trained invariants).

F3  channel maps of c1(0) (8 of 16 channels) for one maze + 2-D PCA of the
    per-cell c1(0) embeddings, coloured by cell type
    (wall / free / start / goal / path).
    Model: CF-Full, hid 16, 160 epochs, trained at fixed T=8, seed 0
    (week2_results/followups/E1_CF-Full_seed0.pt).
F5  guardrail invariants per trained run at T=64 from
    week2_results/trained_invariants.json: max relative mass drift and min
    concentration vs the tolerances 1e-5 and -1e-7.

Run:  python scripts/figures/mechanism.py [--results week2_results]
Outputs land in figures/mechanism/ (PNGs + data.csv).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullFormatter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import OKABE_ITO, apply_style  # noqa: E402
import run_week2  # noqa: E402

FIGDIR = REPO / "figures" / "mechanism"

CELL_COLORS = {
    "wall": OKABE_ITO["grey"],
    "free": OKABE_ITO["sky_blue"],
    "path": OKABE_ITO["blue"],
    "start": OKABE_ITO["vermillion"],
    "goal": OKABE_ITO["bluish_green"],
}
MODEL_COLORS = {"CF-Full": OKABE_ITO["blue"], "CF-NoFrag": OKABE_ITO["bluish_green"]}
TOL_COLOR = OKABE_ITO["vermillion"]


# ---------------------------------------------------------------- overlap check

def _axes_points(ax):
    """All marker points and densely sampled line points of one axes (px)."""
    pts = []
    for line in ax.get_lines():
        xy = np.asarray(line.get_xydata())
        if xy.size == 0:
            continue
        dxy = ax.transData.transform(xy)
        for k in range(len(dxy) - 1):
            a, b = dxy[k], dxy[k + 1]
            n = int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]))) + 1
            if n < 2:
                n = 2
            ts = np.linspace(0.0, 1.0, n)
            pts.extend(a + ts[:, None] * (b - a))
        if line.get_marker() not in ("", "None", None):
            pts.extend(dxy)
    for coll in ax.collections:
        off = np.asarray(coll.get_offsets())
        if off.size:
            pts.extend(ax.transData.transform(off))
    return np.asarray(pts) if pts else np.empty((0, 2))


def check_overlaps(fig, name):
    """Fail if any text overlaps another text, line, marker or scatter point."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        try:
            boxes.append((t, t.get_window_extent(renderer)))
        except Exception:
            continue
    problems = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append("text %r %s overlaps text %r %s"
                                % (boxes[i][0].get_text(), boxes[i][1],
                                   boxes[j][0].get_text(), boxes[j][1]))
    for ax in fig.axes:
        pts = _axes_points(ax)
        if pts.size == 0:
            continue
        for t, bb in boxes:
            inside = (pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) \
                & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)
            if inside.any():
                problems.append("text %r overlaps a line/marker of %s"
                                % (t.get_text(), ax.get_title() or "an axes"))
    if problems:
        raise RuntimeError("overlap check failed for %s:\n  %s"
                            % (name, "\n  ".join(problems)))
    print("[overlap] %s: OK (%d texts checked)" % (name, len(boxes)))


# ---------------------------------------------------------------- F3

def cell_types(x, y):
    """Cell-type label per cell; precedence start > goal > path > free > wall."""
    walk, start, goal = x[0] > 0.5, x[1] > 0.5, x[2] > 0.5
    path = (y > 0.5) & ~start & ~goal
    free = walk & ~start & ~goal & ~path
    labels = np.full(walk.shape, "wall", dtype=object)
    labels[free] = "free"
    labels[path] = "path"
    labels[goal] = "goal"
    labels[start] = "start"
    return labels


def pca2(emb):
    """2-component PCA via SVD; returns (components, explained variance)."""
    xc = emb - emb.mean(axis=0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    return xc @ vt[:2].T, var[:2]


def figure_f3(model, x, y, label, outdir, rows):
    with torch.no_grad():
        c1, _ = model.initial_state(x.unsqueeze(0))
    c1 = c1[0].numpy()                      # (16, 11, 11)
    n_ch = 8
    vmax = float(c1[:n_ch].max())
    labels = cell_types(x.numpy(), y.numpy())
    emb = c1.reshape(c1.shape[0], -1).T     # (121, 16)
    pc, var = pca2(emb)

    apply_style()
    fig = plt.figure(figsize=(14.5, 7.4))
    gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1.15, 1.15],
                          left=0.05, right=0.98, top=0.80, bottom=0.09,
                          wspace=0.55, hspace=0.42)
    ch_axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(n_ch)]
    pca_ax = fig.add_subplot(gs[:, 4:])

    for i, ax in enumerate(ch_axes):
        im = ax.imshow(c1[i], cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title("channel %d" % (i + 1), fontsize=11)
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.grid(False)
    cbar = fig.colorbar(im, ax=ch_axes, orientation="horizontal",
                        shrink=0.55, pad=0.04, aspect=40)
    cbar.set_label("c1(0) concentration (softplus units)", fontsize=11)

    order = ["wall", "free", "path", "goal", "start"]
    for ctype in order:
        sel = labels.flatten() == ctype
        pca_ax.scatter(pc[sel, 0], pc[sel, 1], s=34, alpha=0.85,
                       color=CELL_COLORS[ctype], label=ctype,
                       edgecolors="black", linewidths=0.4)
    pca_ax.set_xlabel("PC 1 (%.1f%% of variance)" % (100 * var[0]))
    pca_ax.set_ylabel("PC 2 (%.1f%% of variance)" % (100 * var[1]))
    pca_ax.set_title("per-cell c1(0) embedding, 2-D PCA", fontsize=13)
    pca_ax.legend(title="cell type", loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=11, title_fontsize=11)

    fig.suptitle("c1(0) already separates walls from corridors before any reaction step\n"
                 "(%s, one level-5 test maze)" % label, fontsize=15)

    path = outdir / "f3_c1_embedding.png"
    check_overlaps(fig, "F3")
    fig.savefig(path)
    plt.close(fig)

    flat = labels.flatten()
    rows.append(("F3", "maze", "grid", "11 x 11 cells", ""))
    rows.append(("F3", "cells", "n_total", "%d" % flat.size, "one maze"))
    for ctype in order:
        rows.append(("F3", "cells", "n_%s" % ctype, "%d" % int((flat == ctype).sum()),
                     "count"))
    rows.append(("F3", "pca", "var_pc1", "%.6f" % var[0], "fraction"))
    rows.append(("F3", "pca", "var_pc2", "%.6f" % var[1], "fraction"))
    for a, b in (("wall", "free"), ("path", "free")):
        pa, pb = pc[flat == a], pc[flat == b]
        dist = float(np.linalg.norm(pa.mean(0) - pb.mean(0)))
        sd = float(np.sqrt((pa.var(0).sum() + pb.var(0).sum()) / 2.0))
        rows.append(("F3", "pca", "sep_%s_%s_sd" % (a, b), "%.3f" % (dist / sd),
                     "centroid distance in pooled sd units"))
    rows.append(("F3", "c1_0", "min_all_16_ch", "%.6g" % float(c1.min()), ""))
    rows.append(("F3", "c1_0", "max_shown_8_ch", "%.6g" % vmax, ""))
    for i in range(n_ch):
        rows.append(("F3", "c1_0", "max_ch%d" % (i + 1),
                     "%.6g" % float(c1[i].max()), ""))
    return path


# ---------------------------------------------------------------- F5

def figure_f5(inv, outdir, rows):
    runs = [(tag, rec) for tag, rec in inv["runs"].items()
            if "max_relative_mass_drift" in rec.get("checks", {})]
    runs.sort(key=lambda tr: (tr[1]["model"], tr[1]["seed"]))
    tags = [tag for tag, _ in runs]
    drift = [rec["checks"]["max_relative_mass_drift"] for _, rec in runs]
    minc = [rec["checks"]["min_c1"] for _, rec in runs]
    tol_drift = inv["tolerances"]["relative_mass_drift_max"]
    tol_pos = inv["tolerances"]["positivity_min_c"]
    T = inv["config"]["T"]
    n_mazes = inv["config"]["n_mazes"]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    xpos = np.arange(len(tags))

    ax = axes[0]
    for (tag, rec), xp in zip(runs, xpos):
        ax.scatter(xp, rec["checks"]["max_relative_mass_drift"], s=90, zorder=3,
                   color=MODEL_COLORS[rec["model"]], edgecolors="black",
                   linewidths=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1e-3)
    ax.axhline(tol_drift, color=TOL_COLOR, linestyle="--", linewidth=2)
    # label in axes-fraction coords (Annotation extents are broken on log axes)
    ax.text(0.98, 0.545, "tolerance 1e-5", transform=ax.transAxes, ha="right",
            fontsize=11, color=TOL_COLOR)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["%s\nseed %d" % (rec["model"], rec["seed"])
                        for _, rec in runs], fontsize=10)
    ax.set_ylabel("max relative mass drift over t = 0..%d (log scale)" % T)
    ax.set_title("mass conservation: max |M_t - M_0| / max(1, max M_0)", fontsize=13)
    handles = [Line2D([], [], marker="o", linestyle="", color=MODEL_COLORS[m],
                      markeredgecolor="black", label=m)
               for m in ("CF-Full", "CF-NoFrag")]
    handles.append(Line2D([], [], color=TOL_COLOR, linestyle="--",
                          label="tolerance 1e-5"))
    ax.legend(handles=handles, loc="upper right", fontsize=11)

    ax = axes[1]
    for (tag, rec), xp in zip(runs, xpos):
        ax.scatter(xp, rec["checks"]["min_c1"], s=90, zorder=3,
                   color=MODEL_COLORS[rec["model"]], edgecolors="black",
                   linewidths=0.6)
    ax.set_yscale("symlog", linthresh=1e-7)
    ax.set_ylim(-2e-7, 1e-5)
    # the symlog locator emits major ticks beyond the view limits whose labels
    # float over the suptitle; fix explicit in-range ticks instead
    ax.set_yticks([-1e-7, 0.0, 1e-7, 1e-6, 1e-5])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(tol_pos, color=TOL_COLOR, linestyle="--", linewidth=2)
    ax.text(0.98, 0.015, "tolerance -1e-7", transform=ax.transAxes, ha="right",
            fontsize=11, color=TOL_COLOR)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["%s\nseed %d" % (rec["model"], rec["seed"])
                        for _, rec in runs], fontsize=10)
    ax.set_ylabel("min c1 over t = 0..%d (symlog)" % T)
    ax.set_title("positivity: min monomer concentration", fontsize=13)
    ax.legend(handles=handles[:2], loc="upper right", fontsize=11)

    fig.suptitle("Trained invariants hold on CPU: every CF run stays inside both "
                 "tolerances at T=%d" % T, fontsize=15)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.06, right=0.98, wspace=0.3)

    path = outdir / "f5_invariants.png"
    check_overlaps(fig, "F5")
    fig.savefig(path)
    plt.close(fig)

    rows.append(("F5", "config", "T", "%d" % T, "rollout steps"))
    rows.append(("F5", "config", "n_mazes", "%d" % n_mazes,
                 "level-5 test split, first N"))
    rows.append(("F5", "tolerance", "relative_mass_drift_max", "%.1e" % tol_drift,
                 ""))
    rows.append(("F5", "tolerance", "positivity_min_c", "%.1e" % tol_pos, ""))
    for (tag, rec), d, m in zip(runs, drift, minc):
        rows.append(("F5", tag, "max_relative_mass_drift", "%.6g" % d, ""))
        rows.append(("F5", tag, "min_c1", "%.6g" % m, ""))
        rows.append(("F5", tag, "status", rec["status"], ""))
    rows.append(("F5", "summary", "n_runs", "%d" % inv["summary"]["n_runs"],
                 "includes 3 RecurrentCNN runs (finite logits only)"))
    rows.append(("F5", "summary", "n_pass", "%d" % inv["summary"]["n_pass"], ""))
    return path


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(REPO / "week2_results"))
    ap.add_argument("--out", default=str(FIGDIR))
    ap.add_argument("--maze-index", type=int, default=0,
                    help="index into the level-5 test split (default 0)")
    args = ap.parse_args(argv)
    results = Path(args.results)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = run_week2.load_data(dict())
    k = args.maze_index
    x, y = data["test"].x[k], data["test"].y[k]

    model = run_week2.make_model("CF-Full")
    state = torch.load(str(results / "followups" / "E1_CF-Full_seed0.pt"),
                       map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.cpu().eval()

    fu = json.load(open(results / "followups" / "results.json", encoding="utf-8"))
    epochs = fu["runs"]["E1_CF-Full_seed0"]["epochs_done"]
    label = "CF-Full, hid 16, %d epochs, trained at fixed T=8, seed 0" % epochs

    inv = json.load(open(results / "trained_invariants.json", encoding="utf-8"))

    rows = []
    f3 = figure_f3(model, x, y, label, outdir, rows)
    f5 = figure_f5(inv, outdir, rows)

    csv_path = outdir / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "item", "quantity", "value", "detail"])
        w.writerows(rows)

    print("F3 -> %s" % f3)
    print("F5 -> %s" % f5)
    print("data -> %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
