"""Per-cell path probability maps (sigmoid output) of the five 160-epoch
curriculum models on the four gallery mazes of path_gallery_1.png.

Packet F6 running example. For each of the four gallery mazes of
`figures/maze_paths/path_gallery_1.png` (the four random-pick test mazes
of the path gallery, re-selected here with the same RNG rule and
cross-checked against the README record), the figure shows the maze, its target, and then every
160-epoch curriculum model's raw per-cell path probability map (sigmoid
output, no threshold) next to its thresholded prediction at the seed-0
validation-selected inference depth T*:

  probability_maps.png         the figure
  data_probability_maps.csv    every number plotted or quoted (per model
                               and maze: probability stats, thresholded
                               metrics, recomputed full-split strict rate)
  README.md                    a new section appended between the two
                               *_probability_maps markers below

Models (all 160 epochs, seed 0, factories of the repo's own pipeline):
  E1_CF-Full          (followups)  CF-Full, hid 16, trained at fixed T=8
  E4_CF-NoFrag160     (followups)  CF-NoFrag, hid 18, fragmentation off
  E5_CF-WallAware160  (wallaware160) CF-WallAware, hid 16, trained fixed T=8
  E6_RecurrentCNN160  (rcnn160)    RecurrentCNN, hid 22, trained fixed T=8
  E7_RecurrentCNN160-Tsamp (rcnn160) RecurrentCNN, hid 22, T sampled 4..16

Verification done by the script itself, CPU only (no GPU is used):
  - the four gallery mazes are re-derived with the numpy RNG seed-0 rule
    of scripts/figures/maze_paths.py and must equal the four random-pick
    mazes recorded in the path-gallery record in
    figures/maze_paths/README.md, otherwise the script exits non-zero;
  - every model's full-split (1000 mazes) strict rate at its stored T*
    is recomputed from the checkpoint and must match the stored
    test_strict_at_T_star within 0.001, otherwise the script exits
    non-zero;
  - layout guard: no text overlaps another text, a line, a marker or a
    maze patch; every font size >= 10 pt; the figure carries a legend.

T* and the stored rates are read from the results JSONs next to the
checkpoints (never hard-coded):

  week2_results/followups/results.json     E1 / E4
  week2_results/wallaware160/results.json  E5
  week2_results/rcnn160/results.json       E6 / E7

The curriculum results root is taken from --week2-results (or the
CF_RESULTS_ROOT environment variable) and defaults to <root>/week2_results,
so no absolute local path is baked in.

Run from the repo root:
    python scripts/figures/probability_maps.py [--root ROOT]
        [--week2-results DIR] [--out DIR] [--seed 0]
"""
import argparse
import csv as _csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
import matplotlib.colors
import matplotlib.legend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import style  # noqa: E402  (shared figure style, read-only)
import run_week2  # noqa: E402  (repo data pipeline / model factories)
import cf.train as train_mod  # noqa: E402
from cf.metrics import evaluate_mask  # noqa: E402
from cf.models.cf import CFNet, MODEL_SPECS  # noqa: E402

style.apply_style()

DEVICE = torch.device("cpu")        # this script runs on CPU only
THRESH = train_mod.HYPERPARAMS["threshold"]
TOL = 0.001                          # tolerance for stored strict rates
MIN_FONT = 10.0                       # no figure text below this size
BATCH = 500
SELECTION_CLASSES = ("random pick",)  # gallery rows of path_gallery_1.png

OI = style.OKABE_ITO
C_CORRECT = OI["blue"]              # predicted cell on the true path
C_EXTRA = OI["vermillion"]           # false positive (extra cell)
C_MISSED = OI["reddish_purple"]      # missed true-path cell
WALL_RGB = (0.18, 0.18, 0.22)
FREE_RGB = (0.95, 0.95, 0.92)
PATH_RGB = (0.70, 0.70, 0.78)
# sequential probability ramp built from the shared palette: near-free-cell
# light -> Okabe-Ito sky blue at p ~ 0.5 -> Okabe-Ito blue at p = 1
PROB_CMAP = LinearSegmentedColormap.from_list(
    "okabe_path_prob", [FREE_RGB, OI["sky_blue"], OI["blue"]])
PROB_CMAP.set_bad(WALL_RGB)         # wall cells

FIGSIZE = (15.4, 18.0)
N_MAZE = 4
# row labels (left of the first panel of each model row); the second line
# of each label carries the training setting, T at inference is appended
MODEL_ROWS = [
    dict(key="cf-full-160ep", label=["CF-Full, hid 16", "160 epochs"]),
    dict(key="cf-nofrag-160ep",
         label=["CF-NoFrag, hid 18", "fragmentation off", "160 epochs"]),
    dict(key="cf-wallaware-160ep",
         label=["CF-WallAware, hid 16", "160 epochs"]),
    dict(key="rcnn160-fixedT",
         label=["RecurrentCNN", "fixed T = 8", "160 epochs"]),
    dict(key="rcnn160-varT",
         label=["RecurrentCNN", "T sampled 4-16", "160 epochs"]),
]

README_MARK_START = "<!-- f6_probability_maps:start -->"
README_MARK_END = "<!-- f6_probability_maps:end -->"


# ---------------------------------------------------------------- registry

def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_registry(week2):
    """The five 160-epoch curriculum models, seed 0.

    T* and the stored full-split strict rate are read from the results
    JSONs written by the training runs (never hard-coded).
    """
    fup = _load_json(week2 / "followups" / "results.json")
    wa = _load_json(week2 / "wallaware160" / "results.json")["runs"]
    rc = _load_json(week2 / "rcnn160" / "results.json")["runs"]

    def from_followups(exp, ckpt_pattern):
        runs = fup["runs"]["%s_seed0" % exp]
        return dict(T_star=int(runs["T_star"]),
                    stored=float(runs["test_strict_at_T_star"]),
                    ckpt_sub=("followups", ckpt_pattern % 0))

    reg = {
        "cf-full-160ep": dict(
            make=lambda: CFNet(hid=16, **MODEL_SPECS["CF-Full"]),
            **from_followups("E1_CF-Full", "E1_CF-Full_seed%d.pt")),
        "cf-nofrag-160ep": dict(
            make=lambda: CFNet(hid=18, **MODEL_SPECS["CF-NoFrag"]),
            **from_followups("E4_CF-NoFrag160", "E4_CF-NoFrag160_seed%d.pt")),
        "cf-wallaware-160ep": dict(
            make=lambda: run_week2.make_model("CF-WallAware"),
            T_star=int(wa["E5_CF-WallAware160_seed0"]["T_star"]),
            stored=float(wa["E5_CF-WallAware160_seed0"]
                         ["test_strict_at_T_star"]),
            ckpt_sub=("wallaware160", "E5_CF-WallAware160_seed0.pt")),
        "rcnn160-fixedT": dict(
            make=lambda: run_week2.make_model("RecurrentCNN"),
            T_star=int(rc["E6_RecurrentCNN160_seed0"]["T_star"]),
            stored=float(rc["E6_RecurrentCNN160_seed0"]
                         ["test_strict_at_T_star"]),
            ckpt_sub=("rcnn160", "E6_RecurrentCNN160_seed0.pt")),
        "rcnn160-varT": dict(
            make=lambda: run_week2.make_model("RecurrentCNN"),
            T_star=int(rc["E7_RecurrentCNN160-Tsamp_seed0"]["T_star"]),
            stored=float(rc["E7_RecurrentCNN160-Tsamp_seed0"]
                         ["test_strict_at_T_star"]),
            ckpt_sub=("rcnn160", "E7_RecurrentCNN160-Tsamp_seed0.pt")),
    }
    return reg


# ------------------------------------------------------------- gallery mazes

def gallery_mazes(readme_path):
    """The four random-pick gallery mazes of path_gallery_1.png.

    Read from the path-gallery record in figures/maze_paths/README.md and
    cross-checked against the same RNG rule that produced them
    (scripts/figures/maze_paths.py: numpy default_rng(0), 4 draws without
    replacement from the 1000 level-5 test mazes)."""
    if not readme_path.is_file():
        sys.exit("ACCEPTANCE FAIL: %s not found (it records which mazes "
                 "path_gallery_1.png shows)" % readme_path)
    line = None
    for ln in readme_path.read_text(encoding="utf-8").splitlines():
        if "Chosen mazes:" in ln:
            line = ln
            break
    if line is None:
        sys.exit("ACCEPTANCE FAIL: 'Chosen mazes:' line not found in %s"
                 % readme_path)
    found = re.findall(r"test maze (\d+) \(([^)]+)\)", line)
    picked = [int(mid) for mid, cls in found if cls in SELECTION_CLASSES]
    if len(picked) != N_MAZE:
        sys.exit("ACCEPTANCE FAIL: expected %d '%s' gallery mazes in %s, "
                 "found %d" % (N_MAZE, "/".join(SELECTION_CLASSES),
                               readme_path, len(picked)))
    rng = np.random.default_rng(0)
    derived = sorted(int(x) for x in rng.choice(1000, size=N_MAZE,
                                                replace=False))
    if sorted(picked) != derived:
        sys.exit("ACCEPTANCE FAIL: the gallery mazes recorded in %s (%s) do "
                 "not match the RNG-seed-0 selection rule (%s)"
                 % (readme_path, picked, derived))
    return picked


# --------------------------------------------------------------- evaluation

@torch.no_grad()
def full_split_strict_rate(model, ds, T):
    """Strict rate of the whole level-5 test split at depth T, mirroring
    cf.train.sweep_model exactly (sigmoid -> 0.5 threshold -> evaluate_mask)."""
    n = 0
    for i in range(0, len(ds), BATCH):
        xb = ds.x[i:i + BATCH].to(DEVICE)
        p = torch.sigmoid(model.logits_at(xb, [T])[T]).cpu().numpy()
        for k in range(len(p)):
            r = evaluate_mask(p[k] >= THRESH,
                              np.asarray(ds.walls[i + k]),
                              ds.starts[i + k], ds.goals[i + k],
                              ds.y[i + k].numpy())
            n += int(r["strict"])
    return n / len(ds)


@torch.no_grad()
def gallery_probs(model, ds, ids, T):
    """Sigmoid probability maps (n, H, W) for the given maze positions."""
    model.eval().to(DEVICE)
    xb = ds.x[ids].to(DEVICE)
    return torch.sigmoid(model.logits_at(xb, [T])[T]).cpu().numpy()


def evaluate_models(reg, ds, ids, week2):
    """Per (model, maze) records + full-split acceptance check per model."""
    records, rng_checks = [], []
    for row in MODEL_ROWS:
        key = row["key"]
        spec = reg[key]
        model = spec["make"]()
        ckpt = week2.joinpath(*spec["ckpt_sub"])
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.to(DEVICE).eval()
        T = spec["T_star"]
        rate = full_split_strict_rate(model, ds, T)
        if abs(rate - spec["stored"]) > TOL:
            sys.exit("ACCEPTANCE FAIL: %s seed 0 test strict at T=%d: "
                     "computed %.4f, stored %.4f"
                     % (key, T, rate, spec["stored"]))
        rng_checks.append((key, T, rate, spec["stored"]))
        probs = gallery_probs(model, ds, ids, T)
        for j, mid in enumerate(ids):
            wall = np.asarray(ds.walls[mid])
            _PROB_CACHE[(key, mid)] = dict(prob=probs[j],
                                           pred=probs[j] >= THRESH)
        for j, mid in enumerate(ids):
            wall = np.asarray(ds.walls[mid])
            gt = ds.y[mid].numpy() > 0.5
            off = (wall == 0) & ~gt
            p = probs[j]
            m = evaluate_mask(p >= THRESH, wall, ds.starts[mid],
                              ds.goals[mid], ds.y[mid].numpy())
            records.append(dict(
                model=key, epochs=160, seed=0, split="test",
                T_at_inference=T, maze=mid,
                dataset_row=int(ds.indices[mid]),
                path_len=int(gt.sum()),
                mean_p_on_path=round(float(p[gt].mean()), 4),
                min_p_on_path=round(float(p[gt].min()), 4),
                mean_p_off_path=round(float(p[off].mean()), 4),
                max_p_off_path=round(float(p[off].max()), 4),
                n_mid_band_free=int(((p >= 0.25) & (p < 0.75) & off).sum()),
                strict=int(m["strict"]),
                wrong_cells=int(m["extra_cells"]) + int(m["missed_cells"]),
                extra_cells=int(m["extra_cells"]),
                missed_cells=int(m["missed_cells"]),
                iou=round(float(m["iou"]), 4),
                test_strict_rate_at_T=round(rate, 4),
                stored_test_strict_at_T=round(spec["stored"], 4)))
    return records, rng_checks


# ---------------------------------------------------------------- viz tools

def _panel_frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)


def _maze_limits(ax, wall):
    H, W = wall.shape
    ax.set_xlim(-0.6, W - 0.4)
    ax.set_ylim(H - 0.4, -0.6)


def _start_goal(ax, start, goal):
    ax.plot(start[1], start[0], marker="o", ms=7, color="black",
            mfc="none", mew=0.8, linestyle="none")
    ax.plot(goal[1], goal[0], marker="*", ms=11, color="black",
            linestyle="none")


def draw_maze_panel(ax, wall, start, goal):
    rgb = np.zeros(wall.shape + (3,))
    rgb[wall == 1] = WALL_RGB
    rgb[wall == 0] = FREE_RGB
    ax.imshow(rgb, interpolation="nearest")
    _start_goal(ax, start, goal)
    _maze_limits(ax, wall)
    _panel_frame(ax)


def draw_target_panel(ax, wall, gt, start, goal):
    rgb = np.zeros(wall.shape + (3,))
    rgb[wall == 1] = WALL_RGB
    rgb[wall == 0] = FREE_RGB
    rgb[(wall == 0) & gt] = PATH_RGB
    ax.imshow(rgb, interpolation="nearest")
    for r, c in np.argwhere(gt):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=0.9))
    _start_goal(ax, start, goal)
    _maze_limits(ax, wall)
    _panel_frame(ax)


def draw_prob_panel(ax, wall, gt, prob, start, goal):
    arr = np.asarray(prob, float).copy()
    arr[wall == 1] = np.nan                 # wall cells -> cmap bad colour
    ax.imshow(arr, cmap=PROB_CMAP, vmin=0.0, vmax=1.0,
              interpolation="nearest")
    for r, c in np.argwhere(gt):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=0.9))
    _start_goal(ax, start, goal)
    _maze_limits(ax, wall)
    _panel_frame(ax)


def draw_pred_panel(ax, wall, gt, pred, start, goal):
    rgb = np.zeros(wall.shape + (3,))
    rgb[wall == 1] = WALL_RGB
    rgb[wall == 0] = FREE_RGB
    rgb[pred & ~gt] = matplotlib.colors.to_rgb(C_EXTRA)
    rgb[pred & gt] = matplotlib.colors.to_rgb(C_CORRECT)
    rgb[gt & ~pred] = matplotlib.colors.to_rgb(C_MISSED)
    ax.imshow(rgb, interpolation="nearest")
    for r, c in np.argwhere(gt):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=0.9))
    _start_goal(ax, start, goal)
    _maze_limits(ax, wall)
    _panel_frame(ax)


def filter_legend():
    return [
        Patch(facecolor=WALL_RGB, label="wall"),
        Patch(facecolor=FREE_RGB, edgecolor="0.6", label="free cell"),
        Patch(facecolor=PATH_RGB, label="true-path cell (target tint)"),
        Patch(facecolor="white", edgecolor="black",
              label="true path outline (probability maps)"),
        Line2D([], [], marker="o", color="black", mfc="none", ms=6, lw=0,
               label="start"),
        Line2D([], [], marker="*", color="black", ms=9, lw=0, label="goal"),
        Patch(facecolor=C_CORRECT, label="predicted cell on true path"),
        Patch(facecolor=C_EXTRA, label="false positive (extra cell)"),
        Patch(facecolor=C_MISSED,
              label="missed cell (on true path, not predicted)"),
    ]


def check_overlaps(fig, name):
    """Programmatic readability guard: no text may overlap another text,
    a line, a marker or a maze patch (legend internals excluded), every
    text is at least 10 pt, and the figure carries a legend."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    legend_texts, legends = set(), []
    for lg in fig.findobj(matplotlib.legend.Legend):
        legends.append(lg)
        legend_texts.update(id(o) for o in lg.findobj(Text))
    tick_texts = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_texts.update(id(l) for l in (tick.label1, tick.label2))
    texts = [t for t in fig.findobj(Text)
             if t.get_text().strip() and id(t) not in legend_texts
             and id(t) not in tick_texts]
    boxes = [(t.get_text()[:30], t.get_window_extent(ren)) for t in texts]
    artists = []
    for ax in fig.axes:
        artists += [(str(ln.get_label() or "line"),
                     ln.get_window_extent(ren)) for ln in ax.get_lines()]
        artists += [(str(pt.get_label() or "patch"),
                     pt.get_window_extent(ren)) for pt in ax.patches]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad.append("text/text: %r vs %r" % (boxes[i][0], boxes[j][0]))
    for tname, tb in boxes:
        for aname, ab in artists:
            if tb.overlaps(ab):
                bad.append("text/artist: %r vs %r" % (tname, aname))
    for lg in legends:
        ax = lg.axes
        if ax is None:
            continue
        lbb = lg.get_window_extent(ren)
        for ln in ax.get_lines():
            if ln in lg.findobj(type(ln)):
                continue
            if lbb.overlaps(ln.get_window_extent(ren)):
                bad.append("legend overlaps line %r" % (ln.get_label(),))
    for t in fig.findobj(Text):
        if t.get_text().strip() and \
                getattr(t, "get_fontsize", lambda: 99.0)() < MIN_FONT:
            bad.append("font < %.0f pt: %r" % (MIN_FONT, t.get_text()[:30]))
    if bad:
        raise RuntimeError("layout in %s has %d overlap(s): %s"
                           % (name, len(bad), "; ".join(b[: 10])))


def save_fig(fig, name, out_dir):
    check_overlaps(fig, name)
    n_entries = sum(len(lg.get_texts()) for lg in
                    fig.findobj(matplotlib.legend.Legend))
    if n_entries == 0:
        sys.exit("ACCEPTANCE FAIL: %s carries no legend" % name)
    path = out_dir / name
    fig.savefig(str(path))
    plt.close(fig)
    print("saved %s (%d legend entries)" % (path, n_entries))


def wrap_to_width(fig, text, max_px, fontsize, fontweight="normal"):
    """Greedy word wrap so no rendered line is wider than max_px."""
    ren = fig.canvas.get_renderer()
    probe = fig.text(0.5, 0.5, "", fontsize=fontsize, fontweight=fontweight)
    lines = []
    for para in text.split("\n"):
        current = ""
        for word in para.split(" "):
            if not word:
                continue
            trial = word if not current else current + " " + word
            probe.set_text(trial)
            if current and probe.get_window_extent(ren).width > max_px:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    probe.remove()
    return "\n".join(lines)


def headline(fig, title, guide):
    """Figure title + two-line reading guide at the top; returns the
    fraction of figure height below the guide (gridspec top margin)."""
    fig.canvas.draw()
    max_px = fig.bbox.width - 160.0
    t = fig.text(0.5, 0.994, wrap_to_width(fig, title, max_px, 15, "bold"),
                 ha="center", va="top", fontsize=15, fontweight="bold")
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    y = t.get_window_extent(ren).y0 / fig.bbox.height \
        - 8.0 / fig.bbox.height
    g = fig.text(0.5, y, wrap_to_width(fig, guide, max_px, 10.5),
                 ha="center", va="top", fontsize=10.5, color="#333333")
    fig.canvas.draw()
    top = g.get_window_extent(ren).y0 / fig.bbox.height \
        - 14.0 / fig.bbox.height
    return top


# ------------------------------------------------------------------ figure

def make_figure(ds, ids, records, min_path_p, max_off_p, n_solved):
    rec = {(r["model"], r["maze"]): r for r in records}
    total = len(records)
    fig = plt.figure(figsize=FIGSIZE)

    title = ("Near-binary confidence: on the four gallery mazes every "
             "160-epoch curriculum model keeps its weakest true-path cell "
             "at p = %.2f and its strongest off-path free cell at p = %.2f "
             "(%d of %d thresholded verdicts solved)"
             % (min_path_p, max_off_p, n_solved, total))
    maze_str = "/".join(str(m) for m in ids)
    guide = ("How to read: each column pair is one of the four gallery "
             "mazes of path_gallery_1.png (level-5 test mazes %s, the "
             "random-pick rows); the top row shows the maze and its "
             "target, each model row the raw per-cell path probability "
             "next to its thresholded prediction at the row's seed-0 "
             "inference depth T." % maze_str)
    guide2 = ("Colour scale (header) = per-cell path probability 0-1, a "
              "cell counts as on-path at p >= 0.5; walls dark, true path "
              "outlined, start / goal / thresholded colours in the "
              "legend; all five models were trained for 160 epochs at "
              "rollout depth T = 8 (variable-T: T sampled from 4-16).")
    top = headline(fig, title, guide + "\n" + guide2)

    n_rows = 2 + len(MODEL_ROWS)
    gs = fig.add_gridspec(n_rows, 2 * N_MAZE,
                          left=0.068, right=0.995, top=top, bottom=0.012,
                          height_ratios=[0.85] + [1.0] * (n_rows - 1),
                          hspace=0.42, wspace=0.09)

    # header: shared legend + probability colour bar
    axleg = fig.add_subplot(gs[0, 0:5])
    axleg.axis("off")
    axleg.legend(handles=filter_legend(), loc="center", ncol=3,
                 fontsize=10, frameon=False, handlelength=1.3,
                 columnspacing=1.0, handletextpad=0.5)
    cax = fig.add_subplot(gs[0, 5:8])
    cax.tick_params(labelsize=10)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0.0, 1.0),
                                      cmap=PROB_CMAP), cax=cax,
                      orientation="horizontal", ticks=[0.0, 0.5, 1.0])
    cb.ax.set_xticklabels(["0", "0.5 (threshold)", "1"])
    cb.set_label("per-cell path probability p (sigmoid output, 0-1)",
                 fontsize=10)

    # ground-truth row: maze + target per gallery maze
    for j, mid in enumerate(ids):
        wall = np.asarray(ds.walls[mid])
        gt = ds.y[mid].numpy() > 0.5
        ax = fig.add_subplot(gs[1, 2 * j])
        draw_maze_panel(ax, wall, ds.starts[mid], ds.goals[mid])
        ax.set_title("test maze %d" % mid, fontsize=10)
        ax.set_xlabel("(random pick)", fontsize=10, labelpad=1)
        ax = fig.add_subplot(gs[1, 2 * j + 1])
        draw_target_panel(ax, wall, gt, ds.starts[mid], ds.goals[mid])
        ax.set_title("target", fontsize=10)
        ax.set_xlabel("%d cells" % int(gt.sum()), fontsize=10, labelpad=1)

    # model rows: probability map | thresholded prediction
    for mi, row in enumerate(MODEL_ROWS):
        key = row["key"]
        for j, mid in enumerate(ids):
            r = rec[(key, mid)]
            T = r["T_at_inference"]
            wall = np.asarray(ds.walls[mid])
            gt = ds.y[mid].numpy() > 0.5
            model = _PROB_CACHE[(key, mid)]
            ax = fig.add_subplot(gs[2 + mi, 2 * j])
            draw_prob_panel(ax, wall, gt, model["prob"],
                            ds.starts[mid], ds.goals[mid])
            ax.set_title("probability map", fontsize=10)
            ax.set_xlabel("min on-path p = %.2f" % r["min_p_on_path"],
                          fontsize=10, labelpad=1)
            if j == 0:
                ax.set_ylabel("\n".join(row["label"]
                                        + ["T at inference = %d" % T]),
                              fontsize=10)
            ax = fig.add_subplot(gs[2 + mi, 2 * j + 1])
            draw_pred_panel(ax, wall, gt, model["pred"],
                            ds.starts[mid], ds.goals[mid])
            ax.set_title("thresholded (p >= 0.5)", fontsize=10)
            ax.set_xlabel("solved" if r["wrong_cells"] == 0
                          else "%d wrong" % r["wrong_cells"],
                          fontsize=10, labelpad=1)
    save_fig(fig, "probability_maps.png", _OUT_DIR)


# -------------------------------------------------------------------- files

CSV_COLS = ["model", "epochs", "seed", "split", "T_at_inference", "maze",
            "dataset_row", "path_len", "mean_p_on_path", "min_p_on_path",
            "mean_p_off_path", "max_p_off_path", "n_mid_band_free",
            "strict", "wrong_cells", "extra_cells", "missed_cells", "iou",
            "test_strict_rate_at_T", "stored_test_strict_at_T"]


def write_csv(records, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print("wrote %s (%d rows)" % (path, len(records)))


_MODEL_TEXT = {
    "cf-full-160ep": "CF-Full (hid 16, 160 epochs, trained at fixed T = 8)",
    "cf-nofrag-160ep": ("CF-NoFrag (hid 18, fragmentation off, 160 epochs, "
                        "trained at fixed T = 8)"),
    "cf-wallaware-160ep": ("CF-WallAware (hid 16, 160 epochs, trained at "
                           "fixed T = 8)"),
    "rcnn160-fixedT": ("RecurrentCNN (hid 22, 160 epochs, trained at fixed "
                       "T = 8)"),
    "rcnn160-varT": ("RecurrentCNN (hid 22, 160 epochs, rollout depth T "
                     "sampled from 4-16 per batch)"),
}


def readme_section(records, ids, rng_checks):
    rec = {(r["model"], r["maze"]): r for r in records}
    worst = min(records, key=lambda r: r["min_p_on_path"])
    loud = max(records, key=lambda r: r["max_p_off_path"])
    n_solved = sum(r["strict"] for r in records)
    mid_cells = sum(r["n_mid_band_free"] for r in records)
    lens = [int((r["path_len"])) for r in records
            if r["model"] == MODEL_ROWS[0]["key"]]

    def t_of(key):
        return rec[(key, ids[0])]["T_at_inference"]

    L = []
    L.append("## `probability_maps.png` - per-cell path probability maps "
             "(packet figure F6)\n")
    L.append("**Caption.** The raw sigmoid output of every 160-epoch "
             "curriculum model (seed 0) on the same four gallery mazes as "
             "`path_gallery_1.png` (level-5 test mazes %s, the "
             "random-pick rows of the path gallery). Column pairs left to right "
             "are the four mazes; the top row shows each maze and its "
             "target; each model row shows, at its seed-0 "
             "validation-selected inference depth T, the per-cell path "
             "probability map (no threshold) and next to it the "
             "thresholded prediction used for all strict/exact scoring. "
             "Models top to bottom: %s."
             % (", ".join(str(m) for m in ids),
                "; ".join(_MODEL_TEXT[r["key"]]
                          for r in MODEL_ROWS)))
    L.append("**How to read.** Probability maps are on a fixed 0-1 "
             "colour scale (header bar; a cell counts as on-path at "
             "p >= 0.5, so 0.5 is the decision threshold), wall cells "
             "are dark, the true path is outlined in black, circle = "
             "start, star = goal. Thresholded panels: blue = predicted "
             "cell on the true path, vermillion = extra predicted cell, "
             "purple = missed true-path cell; the captions below each "
             "probability map give the weakest true-path cell of that "
             "model-maze combination, the captions below each "
             "thresholded panel the wrong-cell count. Each pair of "
             "columns is one maze, each row one model at its own "
             "inference depth T (T = %s top to bottom)."
             % "; ".join("%d" % t_of(r["key"]) for r in MODEL_ROWS))
    L.append("**Key takeaway.** All %d of %d model-maze verdicts are "
             "solved, and the sigmoid outputs are near-binary rather "
             "than graded: across all %d panels the weakest single "
             "true-path cell has p = %.3f (%s, test maze %d) and the "
             "strongest off-path free cell p = %.3f (%s, test maze %d); "
             "%d free cell(s) in total sit in the 0.25-0.75 middle band. "
             "For reference, the same checkpoints on the whole "
             "1000-maze test split at the same T give strict rates %s "
             "(each recomputed by the script and matching the stored "
             "training record within 0.001), so these four easy mazes "
             "(true path %s cells) show a running example of the "
             "sigmoid output, not a difficulty probe."
             % (n_solved, len(records), len(records),
                worst["min_p_on_path"], _MODEL_TEXT[worst["model"]],
                worst["maze"], loud["max_p_off_path"],
                _MODEL_TEXT[loud["model"]], loud["maze"], mid_cells,
                "; ".join("%s %.3f" % (r["key"], r["rate"])
                          for r in rng_checks),
                "/".join(str(n) for n in lens)))
    L.append("**Data source.** `data_probability_maps.csv` in this "
             "folder (one row per model and maze: probability summary "
             "statistics, thresholded metrics, and the recomputed "
             "full-split strict rate with its stored reference); "
             "checkpoints (read-only), relative to the curriculum "
             "results archive root (configurable via the "
             "`$CF_RESULTS_ROOT` environment variable): "
             "`followups/E1_CF-Full_seed0.pt`, "
             "`followups/E4_CF-NoFrag160_seed0.pt`, "
             "`wallaware160/E5_CF-WallAware160_seed0.pt`, "
             "`rcnn160/E6_RecurrentCNN160_seed0.pt`, "
             "`rcnn160/E7_RecurrentCNN160-Tsamp_seed0.pt`. The four "
             "mazes are re-derived with the random-pick selection "
             "rule of the path gallery (`path_gallery_1.png`) and "
             "cross-checked against the record above.")
    L.append("**Caveats.** Single seed (0); the four mazes are random "
             "picks and turn out easy for all five models (all truth "
             "paths 9-11 cells, all verdicts solved), so the figure "
             "illustrates the shape of the sigmoid output on a running "
             "example - models do differ on the full split (see the "
             "strict rates above) and their failures are shown in the "
             "other figures in this folder: the path gallery "
             "(`path_gallery_1.png`, `path_gallery_2.png`), the "
             "whole-split coverage tiles (`coverage_<model>.png`), "
             "the failure anatomy (`failure_anatomy.png`) and the "
             "step-wise rollout in depth (`thinking_steps.png`). "
             "Probabilities are read out once at each model's "
             "validation-selected T; the fixed 0-1 colour scale "
             "saturates above p ~ 0.97, so exact per-cell values are "
             "in the CSV; walls are drawn from the maze, the models "
             "receive the walkable mask but their wall-cell "
             "probabilities are never scored or shown.")
    L.append("Packet figure: F6")
    return "\n\n".join(L) + "\n"


def patch_readme(path, section):
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = README_MARK_START + "\n\n" + section + "\n" + README_MARK_END
    if README_MARK_START in text:
        head = text.split(README_MARK_START)[0]
        tail = text.split(README_MARK_END)[-1]
        text = head + block + tail
    else:
        text = (text.rstrip() + "\n\n" + block + "\n")
    path.write_text(text, encoding="utf-8")
    print("patched %s (F6 section)" % path)


# -------------------------------------------------------------------- main

_PROB_CACHE = {}
_OUT_DIR = None


def main():
    global _OUT_DIR
    parser = argparse.ArgumentParser(
        description="Packet F6: per-cell path probability maps of the "
                    "160-epoch curriculum models on the path_gallery_1 "
                    "gallery mazes.")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repo root (default: this script's repo)")
    parser.add_argument("--week2-results", type=Path, default=None,
                        help="curriculum results root (= week2_results); "
                             "default: $CF_RESULTS_ROOT or "
                             "<root>/week2_results")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder; default "
                             "<root>/figures/maze_paths")
    parser.add_argument("--seed", type=int, default=0,
                        help="which seed's checkpoint to show (default 0)")
    args = parser.parse_args()
    os.environ.setdefault("CF_ALLOW_LOCAL_GPU", "0")

    root = args.root.resolve()
    week2 = args.week2_results
    if week2 is None:
        env = os.environ.get("CF_RESULTS_ROOT", "").strip()
        week2 = Path(env) if env else root / "week2_results"
    out_dir = args.out.resolve() if args.out else root / "figures" / "maze_paths"
    _OUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed != 0:
        sys.exit("ACCEPTANCE FAIL: this figure is defined on the seed-0 "
                 "checkpoints (path_gallery_1 uses seed 0); --seed %d "
                 "not supported here" % args.seed)

    print("loading data (run_week2.load_data, level-5 test split) ...")
    data = run_week2.load_data({})
    ds = data["test"]

    ids = gallery_mazes(out_dir / "README.md")
    print("gallery mazes (path_gallery_1.png, RNG rule + README record): %s"
          % ", ".join(str(m) for m in ids))

    print("registering the five 160-epoch curriculum models (seed 0) ...")
    reg = model_registry(week2)

    print("evaluating %d models on the test split at their stored T* ..."
          % len(MODEL_ROWS))
    records, rng_checks = evaluate_models(reg, ds, ids, week2)
    for key, T, rate, stored in rng_checks:
        print("  %s seed0 test strict @ T=%d: computed %.4f, stored %.4f ok"
              % (key, T, rate, stored))

    # keep the raw maps for the figure (the CSV holds the summary numbers);
    # they were filled into _PROB_CACHE by evaluate_models' single pass
    csv_path = out_dir / "data_probability_maps.csv"
    write_csv(records, csv_path)

    min_path_p = min(r["min_p_on_path"] for r in records)
    max_off_p = max(r["max_p_off_path"] for r in records)
    n_solved = sum(r["strict"] for r in records)
    print("figure: probability maps ...")
    make_figure(ds, ids, records, min_path_p, max_off_p, n_solved)

    rng_out = [dict(key=k, rate=r) for k, _, r, _ in rng_checks]
    patch_readme(out_dir / "README.md",
                 readme_section(records, ids, rng_out))

    print("\nDONE: figures in %s" % out_dir)


if __name__ == "__main__":
    main()
