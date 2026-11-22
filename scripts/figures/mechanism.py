"""Packet figures F3 (per-cell input embeddings) and F5 (trained invariants).

F3  per-cell input embedding of three models, each as channel maps plus a
    2-D PCA coloured by cell type:
    - CF-Full c1(0) (initial monomer concentration, first 8 of 16 channels),
      hid 16, 160 epochs, trained at fixed T=8, seed 0
      (week2_results/followups/E1_CF-Full_seed0.pt), one level-5 test maze;
    - AKOrN conditional input C (nn.Embedding of the given digit, first 8 of
      512 channels) from the final 100-epoch Sudoku EMA checkpoint, extracted
      with the official AKOrN code (SudokuAKOrN.feature's first lines);
    - WONN natural frequency field Omega (MazeInputEmbedding.omega_template
      lookup, first 8 of 128 channels) from the final 150-epoch maze-hard EMA
      checkpoint, extracted with the official WONN code (MazeWinfreeNet:
      omega = f_init(inp); with L=1 this field drives every step).
    The official baseline code, checkpoints, data and run records live in a
    read-only baselines checkout passed via --baselines-dir (or the
    CF_BASELINES_DIR environment variable); nothing outside the repo is
    written. All extraction runs on CPU.

F5  guardrail invariants per trained run at T=64 from
    week2_results/trained_invariants.json: max relative mass drift and min
    concentration vs the tolerances 1e-5 and -1e-7.

Run:  python scripts/figures/mechanism.py --baselines-dir DIR [--results week2_results]
      (or CF_BASELINES_DIR=DIR python scripts/figures/mechanism.py)
Outputs land in figures/mechanism/ (PNGs + data.csv).
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, NullFormatter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import OKABE_ITO, apply_style  # noqa: E402
import run_week2  # noqa: E402

FIGDIR = REPO / "figures" / "mechanism"

# relative locations inside the baselines checkout (never absolute paths)
AKORN_CKPT = "akorn/runs/sudoku_akorn/ema_99.pth"
AKORN_DATA = "akorn/data/sudoku"
AKORN_RECORD = "akorn_tools/out/run_record.json"
WONN_CKPT = "WONN/runs/maze/wonn_reduced_L1T24_ch128_s137_bs32/ema_model.pth"
WONN_DATA = "wonn_tools/data/maze-30x30-hard-1k"
WONN_RECORD = "wonn_tools/out/run_record.json"

CELL_COLORS = {
    "wall": OKABE_ITO["grey"],
    "free": OKABE_ITO["sky_blue"],
    "path": OKABE_ITO["blue"],
    "start": OKABE_ITO["vermillion"],
    "goal": OKABE_ITO["bluish_green"],
    "clue": OKABE_ITO["blue"],
    "blank": OKABE_ITO["grey"],
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
            inside = (pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & \
                     (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)
            if inside.any():
                problems.append("text %r overlaps a line/marker of %s"
                                % (t.get_text(), ax.get_title() or "an axes"))
    if problems:
        raise RuntimeError("overlap check failed for %s:\n  %s"
                            % (name, "\n  ".join(problems)))
    print("[overlap] %s: OK (%d texts checked)" % (name, len(boxes)))


# ---------------------------------------------------------------- F3 helpers

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


def _static_rows(field, cells, order, extra):
    """Per-instance rows shared by all three models of F3."""
    rows = []
    emb = field.reshape(field.shape[0], -1).T
    rows.append(("n_cells", "%d" % emb.shape[0], "one instance"))
    for ctype in order:
        n = int((cells == ctype).sum())
        if n:
            rows.append(("n_%s" % ctype, "%d" % n, "count"))
    rows.append(("n_distinct_embeddings", "%d" % np.unique(emb, axis=0).shape[0],
                 "unique per-cell embedding vectors"))
    rows.append(("field_min", "%.6g" % float(field.min()), "all channels"))
    rows.append(("field_max", "%.6g" % float(field.max()), "all channels"))
    for i in range(8):
        rows.append(("max_shown_ch%d" % (i + 1),
                     "%.6g" % float(np.abs(field[i]).max()),
                     "scale of shown channel map"))
    rows.extend(extra)
    return rows


def load_akorn_pack(baselines, puzzle_index):
    """AKOrN conditional input C for one test sudoku puzzle (official code)."""
    sys.path.insert(0, str(baselines / "akorn"))
    from source.models.sudoku.knet import SudokuAKOrN
    from source.data.datasets.sudoku.sudoku import (
        SudokuDataset, convert_onehot_to_int)

    # constructor flags match the official command in the run record
    net = SudokuAKOrN(n=4, ch=512, L=1, T=16, gamma=1.0, J="attn",
                     use_omega=True, global_omg=True, init_omg=0.5,
                     learn_omg=True, nl=True, heads=8)
    ck = torch.load(str(baselines / AKORN_CKPT), map_location="cpu",
                    weights_only=True)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    state = {k[len("ema_model."):]: v for k, v in sd.items()
             if k.startswith("ema_model.")}
    net.load_state_dict(state, strict=True)
    net.eval()

    ds = SudokuDataset(str(baselines / AKORN_DATA), train=False)
    X, _, is_input = ds[puzzle_index]
    with torch.no_grad():
        # the conditional input C, exactly as in SudokuAKOrN.feature:
        # inp -> self.embedding(inp).permute(0, 3, 1, 2)
        digits = convert_onehot_to_int(X.unsqueeze(0))
        c = net.embedding(digits).permute(0, 3, 1, 2)[0]
    c = c.numpy()                                   # (512, 9, 9)

    rec = json.load(open(baselines / AKORN_RECORD, encoding="utf-8"))
    cells = np.where(is_input[..., 0].numpy() > 0.5, "clue", "blank").flatten()
    order = ["blank", "clue"]
    extra = [
        ("instance", "sudoku test puzzle index %d" % puzzle_index,
         "SATNet sudoku test split (indices 9000-9999)"),
        ("grid", "9 x 9 cells", ""),
        ("n_shown_of_channels", "8 of 512", "first 8 channels shown"),
        ("epochs_run", "%d" % rec["epochs_run"],
         "official epochs %d; single unseeded run: preliminary (fewer than 3 seeds)"
         % rec["official_epochs"]),
        ("seeds_run", "%d" % rec["seeds"]["seeds_run"], "single run: preliminary"),
        ("T", "16", "model constructor, matches official command in run record"),
        ("ckpt", AKORN_CKPT, "final EMA checkpoint per run record"),
    ]
    return dict(
        tag="akorn", field=c, cmap="RdBu_r",
        vlim=(-float(np.abs(c[:8]).max()), float(np.abs(c[:8]).max())),
        cbar_label="conditional input C (embedding units, signed)",
        ticks=([0, 4, 8], [0, 4, 8]),
        header_lines=[
            "AKOrN conditional input C = the official code's embedding of the "
            "given digit (Sudoku, 512 channels)",
            "%d epochs (official budget), T=16, single unseeded run: "
            "preliminary" % rec["epochs_run"]],
        cells=cells, order=order, colors=CELL_COLORS,
        static_rows=_static_rows(c, cells, order, extra),
        sep_pairs=[("blank", "clue")], spread_groups=["blank", "clue"],
    )


def load_wonn_pack(baselines, maze_index):
    """WONN natural frequency field Omega for one test maze (official code)."""
    sys.path.insert(0, str(baselines / "WONN"))
    from maze.wnet import MazeWinfreeNet
    from maze.data import MazeDataset

    rec = json.load(open(baselines / WONN_RECORD, encoding="utf-8"))
    cfg = rec["config"]
    net = MazeWinfreeNet(ch=cfg["ch"], L=cfg["L"], T=cfg["T"],
                         coupling=cfg["coupling"], gamma=cfg["gamma"],
                         group_size=cfg["group_size"], norm=cfg["norm"],
                         heads=cfg["heads"], output_ksize=cfg["output_ksize"])
    ck = torch.load(str(baselines / WONN_CKPT), map_location="cpu",
                    weights_only=True)
    if isinstance(ck, dict):
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    else:
        sd = ck
    # official eval_maze.load_state: strip the EMA wrapper prefix if present
    if any(str(k).startswith("ema_model.") for k in sd.keys()):
        sd = {k[len("ema_model."):]: v for k, v in sd.items()
              if str(k).startswith("ema_model.")}
    net.load_state_dict(sd, strict=True)
    net.eval()

    ds = MazeDataset(str(baselines / WONN_DATA), split="test")
    inp = ds.inputs[maze_index]
    with torch.no_grad():
        # MazeWinfreeNet.feature: omega_base = self.f_init(inp); with L=1
        # (layer 0 has Identity transitions) this omega drives every step
        omega = net.f_init(torch.from_numpy(inp).unsqueeze(0))[0]
    omega = omega.numpy()                          # (ch, 30, 30)

    token_names = {0: "wall", 1: "free", 2: "start", 3: "goal"}
    cells = np.array([token_names[t] for t in inp.reshape(-1).tolist()])
    order = [t for t in ("wall", "free", "start", "goal")
             if t in set(cells.tolist())]
    extra = [
        ("instance", "maze index %d" % maze_index,
         "maze-30x30-hard-1k test split, converted per run record"),
        ("grid", "30 x 30 cells", ""),
        ("n_shown_of_channels", "8 of %d" % omega.shape[0],
         "first 8 channels shown"),
        ("epochs_trained", "%d" % rec["epochs_trained"],
         "official epochs %d: reduced budget, not a reproduction"
         % rec["official_epochs"]),
        ("seed", "%d" % rec["seed"], "single seed: preliminary"),
        ("T", "%d" % cfg["T"], "run record config"),
        ("ckpt", WONN_CKPT, "final EMA checkpoint per run record"),
    ]
    return dict(
        tag="wonn", field=omega, cmap="RdBu_r",
        vlim=(-float(np.abs(omega[:8]).max()),
              float(np.abs(omega[:8]).max())),
        cbar_label="natural frequency field omega (rad per step, signed)",
        ticks=([0, 10, 20, 29], [0, 10, 20, 29]),
        header_lines=[
            "WONN natural frequency field omega = per-token lookup into "
            "MazeInputEmbedding.omega_template (maze-hard, %d channels)"
            % omega.shape[0],
            "official code, %d of %d epochs (reduced budget, not a "
            "reproduction), T=%d, seed %d: preliminary"
            % (rec["epochs_trained"], rec["official_epochs"],
               cfg["T"], rec["seed"])],
        cells=cells, order=order, colors=CELL_COLORS,
        static_rows=_static_rows(omega, cells, order, extra),
        sep_pairs=[("wall", "free")],
        spread_groups=order,
    )


# ---------------------------------------------------------------- F3

def figure_f3(packs, n_shown, outdir, rows):
    apply_style()
    n_sec = len(packs)
    fig = plt.figure(figsize=(14.5, 18.2))
    gs = fig.add_gridspec(
        4 * n_sec, 7,
        height_ratios=[0.42, 1.0, 1.0, 0.18] * n_sec,
        width_ratios=[1, 1, 1, 1, 0.0, 1.15, 1.15],
        left=0.055, right=0.955, top=0.875, bottom=0.05,
        wspace=0.5, hspace=0.6)
    for si, p in enumerate(packs):
        hdr_ax = fig.add_subplot(gs[4 * si, :])
        hdr_ax.axis("off")
        hdr_ax.text(0.5, 0.42, "\n".join(p["header_lines"]), fontsize=13,
                    ha="center", va="center", transform=hdr_ax.transAxes)

        field = p["field"]
        ch_axes = [fig.add_subplot(gs[4 * si + 1 + i // 4, i % 4])
                   for i in range(n_shown)]
        cax = fig.add_subplot(gs[4 * si + 3, 0:4])
        pca_ax = fig.add_subplot(gs[4 * si + 1:4 * si + 3, 5:])
        vmin, vmax = p["vlim"]
        for i, ax in enumerate(ch_axes):
            im = ax.imshow(field[i], cmap=p["cmap"], vmin=vmin, vmax=vmax)
            ax.set_title("channel %d" % (i + 1), fontsize=11)
            ax.set_xticks(p["ticks"][0])
            ax.set_yticks(p["ticks"][1])
            ax.tick_params(labelsize=10)
            if i < 4:                     # top row: x labels repeat the
                ax.tick_params(labelbottom=False)   # bottom row's grid
            ax.grid(False)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(p["cbar_label"], fontsize=11)

        emb = field.reshape(field.shape[0], -1).T
        pc, var = pca2(emb)
        flat = p["cells"]
        for ctype in p["order"]:
            sel = flat == ctype
            if not sel.any():
                continue
            pca_ax.scatter(pc[sel, 0], pc[sel, 1], s=34, alpha=0.85,
                           color=p["colors"][ctype],
                           label="%s (%d)" % (ctype, int(sel.sum())),
                           edgecolors="black", linewidths=0.4)
        pca_ax.set_xlabel("PC 1 (%.1f%% of variance)" % (100 * var[0]),
                          fontsize=12)
        pca_ax.set_ylabel("PC 2 (%.1f%% of variance)" % (100 * var[1]),
                          fontsize=12, labelpad=6)
        pca_ax.tick_params(labelsize=10)
        # prune the corner ticks so the axis tick labels cannot collide
        # with each other or with the section header
        pca_ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
        pca_ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4, prune="both"))
        pca_ax.set_title("2-D PCA, one dot per cell", fontsize=12)
        pca_ax.legend(title="cell type", loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), fontsize=10,
                      title_fontsize=10)

        tag = p["tag"]
        for q, v, d in p["static_rows"]:
            rows.append(("F3", tag, q, v, d))
        rows.append(("F3", tag, "var_pc1", "%.6f" % var[0], "fraction"))
        rows.append(("F3", tag, "var_pc2", "%.6f" % var[1], "fraction"))
        for a, b in p["sep_pairs"]:
            pa, pb = pc[flat == a], pc[flat == b]
            dist = float(np.linalg.norm(pa.mean(0) - pb.mean(0)))
            sd = float(np.sqrt((pa.var(0).sum() + pb.var(0).sum()) / 2.0))
            same = (pa.size == 0 or pb.size == 0 or
                    bool(np.all(pa == pa[0])) and bool(np.all(pb == pb[0])))
            if same:
                rows.append(("F3", tag, "d_%s_%s_pc" % (a, b), "%.3f" % dist,
                             "centroid distance in PC units; every cell of "
                             "both types holds the same single point"))
            else:
                rows.append(("F3", tag, "sep_%s_%s_sd" % (a, b),
                             "%.3f" % (dist / sd),
                             "centroid distance in pooled sd units"))
        for g in p["spread_groups"]:
            pg = pc[flat == g]
            spread = float(max(pg[:, 0].max() - pg[:, 0].min(),
                               pg[:, 1].max() - pg[:, 1].min()))
            rows.append(("F3", tag, "spread_%s_pc" % g, "%.3f" % spread,
                         "max extent of one cell type in the PC plane"))

    fig.suptitle("Per-cell input embeddings: CF's c1(0) varies across cells, "
                 "while AKOrN's C and WONN's omega take one value per input "
                 "token\n"
                 "How to read: one section per model, top to bottom (CF-Full, "
                 "AKOrN, WONN); left: the first 8 channel maps of that model's "
                 "per-cell input tensor,\n"
                 "one shared colour scale per section; right: 2-D PCA of the "
                 "per-cell embedding vectors, one dot per cell, coloured by "
                 "cell type", fontsize=15)

    path = outdir / "f3_c1_embedding.png"
    check_overlaps(fig, "F3")
    fig.savefig(path)
    plt.close(fig)
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
                    help="index into the level-5 test split for CF (default 0)")
    ap.add_argument("--puzzle-index", type=int, default=0,
                    help="index into the sudoku ID test split for AKOrN "
                         "(default 0)")
    ap.add_argument("--wonn-maze-index", type=int, default=0,
                    help="index into the maze-hard test split for WONN "
                         "(default 0)")
    ap.add_argument("--baselines-dir",
                    default=os.environ.get("CF_BASELINES_DIR"),
                    help="read-only baselines checkout holding the official "
                         "AKOrN and WONN code, checkpoints, data and run "
                         "records (or set CF_BASELINES_DIR)")
    args = ap.parse_args(argv)
    if not args.baselines_dir:
        ap.error("--baselines-dir (or the CF_BASELINES_DIR environment "
                 "variable) is required: it points at the read-only "
                 "baselines checkout with the official model code and "
                 "checkpoints")
    baselines = Path(args.baselines_dir)
    results = Path(args.results)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # CF: c1(0) panel (unchanged model and instance)
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
    fu = json.load(open(results / "followups" / "results.json",
                        encoding="utf-8"))
    epochs = fu["runs"]["E1_CF-Full_seed0"]["epochs_done"]

    with torch.no_grad():
        c1, _ = model.initial_state(x.unsqueeze(0))
    c1 = c1[0].numpy()                       # (16, 11, 11)
    labels = cell_types(x.numpy(), y.numpy()).flatten()
    cf_order = ["wall", "free", "path", "goal", "start"]
    cf_extra = [
        ("instance", "level-5 test maze index %d" % k, "from --maze-index"),
        ("grid", "11 x 11 cells", ""),
        ("n_shown_of_channels", "8 of %d" % c1.shape[0],
         "first 8 channels shown"),
        ("epochs", "%d" % epochs,
         "single run, seed 0 (of the 3-seed follow-up family)"),
        ("T_train", "8", "fixed during training"),
        ("ckpt", "week2_results/followups/E1_CF-Full_seed0.pt", "repo-relative"),
    ]
    cf_pack = dict(
        tag="cf", field=c1, cmap="viridis",
        vlim=(0.0, float(c1[:8].max())),
        cbar_label="c1(0) concentration (softplus units)",
        ticks=([0, 5, 10], [0, 5, 10]),
        header_lines=[
            "CF-Full c1(0), initial monomer concentration (%d channels): "
            "hid 16, %d epochs, trained at fixed T=8, seed 0"
            % (c1.shape[0], epochs)],
        cells=labels, order=cf_order, colors=CELL_COLORS,
        static_rows=_static_rows(c1, labels, cf_order, cf_extra),
        sep_pairs=[("wall", "free"), ("path", "free")], spread_groups=[],
    )

    packs = [cf_pack,
             load_akorn_pack(baselines, args.puzzle_index),
             load_wonn_pack(baselines, args.wonn_maze_index)]

    inv = json.load(open(results / "trained_invariants.json",
                         encoding="utf-8"))

    rows = []
    f3 = figure_f3(packs, 8, outdir, rows)
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
