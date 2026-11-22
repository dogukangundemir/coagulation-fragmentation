"""Commitment-over-time figures (packet figure F4): how CF-Full's path
probability per cell evolves over the recurrence steps.

Loads the CF-Full checkpoint (hid 16, 160 epochs, trained at fixed T=8,
seed 0) with the repo's own factory, rolls it out once on two level-5 test
mazes (selection rule: test maze 0 and the test maze with the longest true
path, ties by maze index) and writes into figures/commitment/:

  data.csv                    one row per (maze, t): median / IQR of the
                              per-cell path probability over the
                              true-path cells and over the off-path free
                              cells, plus wrong cells and strict at t
  commitment_maze<k>.png      one PNG per maze (each with its own legend):
                              top = ground truth + per-step probability
                              maps at t = 1, 2, 4, 8, 12, 16; bottom =
                              median +- IQR of P(path) over t = 1..16 for
                              the two cell groups
  README.md                   caption / how-to-read / key takeaway per
                              figure; every takeaway number is read back
                              from data.csv
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
import matplotlib.colors
import matplotlib.legend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT = ROOT / "figures" / "commitment"
CKPT = ROOT / "week2_results" / "followups" / "E1_CF-Full_seed0.pt"
RESULTS = ROOT / "week2_results" / "followups" / "results.json"
DEVICE = train_mod.DEVICE
THRESH = train_mod.HYPERPARAMS["threshold"]

OI = style.OKABE_ITO
C_TRUE = OI["blue"]            # true-path cells
C_OFF = OI["vermillion"]       # off-path free cells
C_THRESH = OI["grey"]
WALL_RGB = (0.18, 0.18, 0.22)
CMAP = plt.get_cmap("viridis")  # colourblind-safe sequential
MAP_STEPS = [1, 2, 4, 8, 12, 16]
CURVE_STEPS = list(range(1, 17))
MODEL_DESC = "CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0"


# ---------------------------------------------------------------- model/data

def load_model():
    model = CFNet(hid=16, **MODEL_SPECS["CF-Full"])
    model.load_state_dict(torch.load(str(CKPT), map_location="cpu"))
    return model.to(DEVICE).eval()


def pick_mazes(ds):
    """Selection rule: test maze 0 and the longest true path (ties by index)."""
    lens = np.array([int(v.sum().item()) for v in ds.y])
    longest = int(np.lexsort((np.arange(len(lens)), -lens))[0])
    return [0, longest], lens


@torch.no_grad()
def rollout_probs(model, ds, positions):
    """Sigmoid path probability per cell after each t = 1..16 (one rollout)."""
    xb = ds.x[positions].to(DEVICE)
    outs = model.logits_at(xb, CURVE_STEPS)
    return {t: torch.sigmoid(outs[t]).cpu().numpy() for t in outs}


def cell_stats(p, true_mask, off_mask):
    def grp(mask):
        v = p[mask]
        if v.size == 0:
            return dict(n=0, med=float("nan"), q25=float("nan"),
                        q75=float("nan"))
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        return dict(n=int(v.size), med=float(med), q25=float(q25),
                    q75=float(q75))
    return grp(true_mask), grp(off_mask)


def collect_rows(ds, mazes, probs):
    """data.csv rows: per (maze, t) group stats + thresholded metrics."""
    rows = []
    for k in mazes:
        gt = ds.y[k].numpy() > 0.5
        wall = ds.walls[k]
        off = (wall == 0) & ~gt
        for t in CURVE_STEPS:
            p = probs[t][mazes.index(k)]
            st_true, st_off = cell_stats(p, gt, off)
            r = evaluate_mask(p >= THRESH, wall, ds.starts[k],
                              ds.goals[k], ds.y[k].numpy())
            rows.append(dict(
                maze=k, dataset_row=int(ds.indices[k]),
                path_len=int(gt.sum()), t=t,
                n_true=st_true["n"], med_true=st_true["med"],
                q25_true=st_true["q25"], q75_true=st_true["q75"],
                n_off=st_off["n"], med_off=st_off["med"],
                q25_off=st_off["q25"], q75_off=st_off["q75"],
                wrong_cells=int(r["extra_cells"]) + int(r["missed_cells"]),
                strict=int(r["strict"])))
    return rows


def write_csv(rows, path):
    cols = ["maze", "dataset_row", "path_len", "t", "n_true", "med_true",
            "q25_true", "q75_true", "n_off", "med_off", "q25_off", "q75_off",
            "wrong_cells", "strict"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: (round(row[c], 4)
                            if isinstance(row[c], float) else row[c])
                        for c in cols})


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- drawing

def draw_prob_map(ax, p, wall, gt, start, goal, title, xlabel):
    """One panel: wall cells dark, free cells coloured by P(path) (viridis),
    true path outlined, start = circle, goal = star."""
    rgb = CMAP(np.clip(p, 0.0, 1.0))[..., :3]
    rgb[wall == 1] = WALL_RGB
    ax.imshow(rgb, interpolation="nearest", vmin=0.0, vmax=1.0)
    for r, c in np.argwhere(gt):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=0.9))
    ax.plot(start[1], start[0], marker="o", ms=6, color="black",
            mfc="none", mew=0.8, linestyle="none")
    ax.plot(goal[1], goal[0], marker="*", ms=10, color="black",
            linestyle="none")
    ax.set_xlim(-0.6, p.shape[1] - 0.4)
    ax.set_ylim(p.shape[0] - 0.4, -0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=1)


def check_overlaps(fig, name):
    """Programmatic readability guard (same rule as maze_paths.py): no text
    may overlap another text, a line, a marker or a patch, and a legend may
    not sit on top of plotted lines/markers."""
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
    if bad:
        raise RuntimeError("layout in %s has %d overlap(s): %s"
                           % (name, len(bad), "; ".join(bad[:10])))


def save(fig, name):
    check_overlaps(fig, name)
    n_entries = sum(len(lg.get_texts()) for lg in
                    fig.findobj(matplotlib.legend.Legend))
    if n_entries == 0:
        sys.exit("ACCEPTANCE FAIL: %s carries no legend" % name)
    fig.savefig(str(OUT / name))
    plt.close(fig)
    print("saved %s (%d legend entries)" % (OUT / name, n_entries))


def figure_one_maze(ds, k, probs_k, rows_k):
    """One self-contained PNG: probability maps + P(path) over t.

    probs_k: {t: (H, W) float array} of per-cell path probabilities for
    this maze, one shared rollout read out at every t.
    """
    gt = ds.y[k].numpy() > 0.5
    wall = ds.walls[k]
    off = (wall == 0) & ~gt
    plen = int(gt.sum())
    by_t = {int(r["t"]): r for r in rows_k}

    fig = plt.figure(figsize=(13.6, 9.6))
    gs = fig.add_gridspec(4, 1 + len(MAP_STEPS),
                          height_ratios=[0.22, 1.0, 0.78, 0.30],
                          hspace=0.62, wspace=0.10)

    axt = fig.add_subplot(gs[0, :])
    axt.axis("off")
    axt.set_title("Commitment over time: %s - level-5 test maze %d "
                  "(true path %d cells)\nper-cell path probability while "
                  "the loop runs (one shared rollout, read out at each t)"
                  % (MODEL_DESC, k, plen), fontsize=12)

    map_axes = []
    ax0 = fig.add_subplot(gs[1, 0])
    draw_prob_map(ax0, gt.astype(float), wall, gt,
                  ds.starts[k], ds.goals[k], "ground truth",
                  "%d cells" % plen)
    map_axes.append(ax0)
    for j, t in enumerate(MAP_STEPS):
        ax = fig.add_subplot(gs[1, j + 1])
        r = by_t[t]
        xlabel = "solved" if int(r["wrong_cells"]) == 0 \
            else "%d wrong" % int(r["wrong_cells"])
        draw_prob_map(ax, probs_k[t], wall, gt, ds.starts[k],
                      ds.goals[k], "t = %d" % t, xlabel)
        map_axes.append(ax)
    cbar = fig.colorbar(map_axes[1].images[0], ax=map_axes, shrink=0.85,
                        pad=0.02)
    cbar.set_label("P(path) per cell", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ---- P(path) over t: median + IQR per cell group
    axc = fig.add_subplot(gs[2, :])
    ts = np.array(CURVE_STEPS, dtype=float)
    med_true = np.array([float(by_t[t]["med_true"]) for t in CURVE_STEPS])
    q25_true = np.array([float(by_t[t]["q25_true"]) for t in CURVE_STEPS])
    q75_true = np.array([float(by_t[t]["q75_true"]) for t in CURVE_STEPS])
    med_off = np.array([float(by_t[t]["med_off"]) for t in CURVE_STEPS])
    q25_off = np.array([float(by_t[t]["q25_off"]) for t in CURVE_STEPS])
    q75_off = np.array([float(by_t[t]["q75_off"]) for t in CURVE_STEPS])
    # probabilities lie in [0, 1]; clip the band there and state it
    axc.fill_between(ts, np.clip(q25_true, 0.0, 1.0),
                     np.clip(q75_true, 0.0, 1.0), color=C_TRUE, alpha=0.25,
                     linewidth=0)
    axc.fill_between(ts, np.clip(q25_off, 0.0, 1.0),
                     np.clip(q75_off, 0.0, 1.0), color=C_OFF, alpha=0.25,
                     linewidth=0)
    lt, = axc.plot(ts, med_true, color=C_TRUE, lw=2.0,
                   label="median, true-path cells (n = %d)"
                         % int(by_t[1]["n_true"]))
    lo, = axc.plot(ts, med_off, color=C_OFF, lw=2.0,
                   label="median, off-path free cells (n = %d)"
                         % int(by_t[1]["n_off"]))
    axc.plot(MAP_STEPS, [float(by_t[t]["med_true"]) for t in MAP_STEPS],
             "o", ms=5, color=C_TRUE)
    axc.plot(MAP_STEPS, [float(by_t[t]["med_off"]) for t in MAP_STEPS],
             "o", ms=5, color=C_OFF)
    axc.axhline(THRESH, color=C_THRESH, lw=1.2, linestyle="--")
    axc.set_xticks(CURVE_STEPS)
    axc.set_xlim(0.5, 16.5)
    axc.set_ylim(-0.04, 1.06)
    axc.set_xlabel("recurrence step t", fontsize=11)
    axc.set_ylabel("P(path) per cell", fontsize=11)
    axc.set_title("P(path) over t: median (line) + IQR (band, 25th-75th "
                  "percentile over the cells of each group, clipped to "
                  "[0, 1]); dots mark the map steps t = 1, 2, 4, 8, 12, 16",
                  fontsize=11)

    axl = fig.add_subplot(gs[3, :])
    axl.axis("off")
    axl.legend(handles=[
        lt, lo,
        Patch(facecolor=C_TRUE, alpha=0.25,
              label="IQR band, true-path cells"),
        Patch(facecolor=C_OFF, alpha=0.25,
              label="IQR band, off-path free cells"),
        Line2D([], [], color=C_THRESH, lw=1.2, linestyle="--",
               label="decision threshold 0.5"),
        Patch(facecolor=WALL_RGB, label="wall (map panels)"),
        Patch(facecolor="white", edgecolor="black",
              label="true path (outline)"),
        Line2D([], [], marker="o", color="black", mfc="none", ms=6, lw=0,
               label="start"),
        Line2D([], [], marker="*", color="black", ms=9, lw=0,
               label="goal"),
    ], loc="center", ncol=5, fontsize=10, frameon=False, handlelength=1.5,
        columnspacing=1.3)
    save(fig, "commitment_maze%d.png" % k)


# ---------------------------------------------------------------- README

def write_readme(ds, mazes, rows, T_star, strict_at_Tstar, params):
    by_maze = {k: [r for r in rows if int(r["maze"]) == k] for k in mazes}
    L = []
    L.append("# Commitment over time: per-step path probability "
             "(packet figure F4)\n")
    L.append("All numbers below are recomputed by "
             "`scripts/figures/commitment.py` from the stored checkpoint and "
             "written to `data.csv` in this folder (one row per maze and "
             "recurrence step t); the README numbers are read back from "
             "that CSV.\n")
    L.append("Model: **%s**, %d parameters, checkpoint "
             "`week2_results/followups/E1_CF-Full_seed0.pt`; validation-"
             "selected inference depth T\\* = %d (test strict solve rate "
             "%.3f at T\\*, from `week2_results/followups/results.json`). "
             "Input: walkable mask + start + goal one-hot channels; output: "
             "per-cell path logit, sigmoid -> P(path), decision threshold "
             "0.5.\n" % (MODEL_DESC, params, T_star, strict_at_Tstar))
    L.append("Mazes (level-5 test split, 1000 mazes, 11x11, same pipeline "
             "as `run_week2.load_data`): selection rule = test maze 0 and "
             "the test maze with the longest true path (ties by maze "
             "index).\n")

    for k in mazes:
        rk = sorted(by_maze[k], key=lambda r: int(r["t"]))
        plen = int(rk[0]["path_len"])
        n_true = int(rk[0]["n_true"])
        n_off = int(rk[0]["n_off"])
        r1, r16 = rk[0], rk[-1]
        strict = [int(r["strict"]) for r in rk]
        solved_ts = [int(r["t"]) for r in rk if int(r["strict"]) == 1]
        stay = (solved_ts == list(range(min(solved_ts),
                                        min(solved_ts) + len(solved_ts)))
                if solved_ts else False)
        L.append("---\n")
        L.append("## `commitment_maze%d.png` - test maze %d "
                 "(true path %d cells)\n" % (k, k, plen))
        L.append("**Caption.** Top: the ground truth and the per-cell path "
                 "probability P(path) after t = 1, 2, 4, 8, 12, 16 "
                 "recurrence steps of %s (one shared rollout, read out at "
                 "each t; the number under each panel is the wrong-cell "
                 "count of the thresholded prediction at that t, "
                 "\\\"solved\\\" = 0). Bottom: median and IQR of P(path) "
                 "over t = 1..16, separately for the true-path cells and "
                 "the off-path free cells."
                 % MODEL_DESC)
        L.append("**How to read.** Map panels: free cells are coloured by "
                 "P(path) on the viridis scale (dark = 0, yellow = 1, see "
                 "colour bar), wall cells are dark grey, the true path is "
                 "outlined in black, start = circle, goal = star. Curve: "
                 "blue = median over the %d true-path cells, vermillion = "
                 "median over the %d off-path free cells; the shaded bands "
                 "are the IQR (25th-75th percentile over the cells of each "
                 "group, n stated in the legend), clipped to [0, 1] because "
                 "a probability cannot leave that range; the dashed grey "
                 "line is the 0.5 decision threshold; dots mark the six "
                 "steps shown as maps.\n" % (n_true, n_off))
        if solved_ts and stay:
            tail = ("first fully solved (strict) at t = %d and stays "
                    "solved through t = 16" % min(solved_ts))
        elif solved_ts:
            tail = ("fully solved (strict) at t = %s, not at every later "
                    "step" % "/".join(str(t) for t in solved_ts))
        else:
            best = min((int(r["wrong_cells"]), int(r["t"])) for r in rk)
            tail = ("never fully solved within 16 steps (best %d wrong "
                    "cells at t = %d)" % best)
        def verb(a, b):
            if b > a + 0.005:
                return "rises from"
            if b < a - 0.005:
                return "falls from"
            return "moves from"
        L.append("**Key takeaway.** On test maze %d the median P(path) on "
                 "the %d true-path cells %s %.2f at t = 1 to %.2f at t = 16 "
                 "(IQR width %.3f -> %.3f), while the median on the %d "
                 "off-path free cells %s %.2f to %.2f (IQR width %.3f -> "
                 "%.3f): the loop commits to the true path and suppresses "
                 "the alternatives. The thresholded prediction has %d "
                 "wrong cells at t = 1 and %s."
                 % (k, n_true, verb(float(r1["med_true"]),
                                    float(r16["med_true"])),
                    float(r1["med_true"]), float(r16["med_true"]),
                    float(r1["q75_true"]) - float(r1["q25_true"]),
                    float(r16["q75_true"]) - float(r16["q25_true"]),
                    n_off, verb(float(r1["med_off"]),
                                float(r16["med_off"])),
                    float(r1["med_off"]), float(r16["med_off"]),
                    float(r1["q75_off"]) - float(r1["q25_off"]),
                    float(r16["q75_off"]) - float(r16["q25_off"]),
                    int(r1["wrong_cells"]), tail))
        L.append("**Data source.** `data.csv` in this folder (rows with "
                 "maze = %d); checkpoint "
                 "`week2_results/followups/E1_CF-Full_seed0.pt`; data "
                 "pipeline `scripts/run_week2.py` (level-5 test split).\n"
                 % k)
        L.append("**Caveats.** Single maze, single seed (0); median + IQR "
                 "over cells within this one maze (n = %d true-path and "
                 "%d off-path free cells, no averaging over mazes); wall "
                 "cells are excluded from both groups (the model can still "
                 "assign them probability, which counts as wrong cells in "
                 "the strict metric); t = 9..16 is beyond the training "
                 "rollout depth T = 8; the IQR is a within-maze spread, "
                 "not a seed uncertainty.\n" % (n_true, n_off))
        L.append("**Packet figure: F4**\n")

    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("saved %s" % (OUT / "README.md"))


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading data (run_week2.load_data, level-5 test split) ...")
    data = run_week2.load_data({})
    ds = data["test"]
    mazes, lens = pick_mazes(ds)
    print("mazes: %d (index-0 pick) and %d (longest true path, %d cells)"
          % (mazes[0], mazes[1], int(lens[mazes[1]])))

    print("loading %s ..." % CKPT.name)
    model = load_model()
    print("rolling out t = 1..16 ...")
    probs = rollout_probs(model, ds, mazes)

    rows = collect_rows(ds, mazes, probs)
    csv_path = OUT / "data.csv"
    write_csv(rows, csv_path)
    print("wrote %s (%d rows)" % (csv_path, len(rows)))
    rows = read_csv(csv_path)          # README numbers come from the CSV

    for i, k in enumerate(mazes):
        probs_k = {t: probs[t][i] for t in probs}
        rows_k = [r for r in rows if int(r["maze"]) == k]
        print("figure: maze %d ..." % k)
        figure_one_maze(ds, k, probs_k, rows_k)

    fup = json.loads(RESULTS.read_text(encoding="utf-8"))
    run0 = fup["runs"]["E1_CF-Full_seed0"]
    write_readme(ds, mazes, rows, int(run0["T_star"]),
                 float(run0["test_strict_by_T"][str(run0["T_star"])]),
                 int(fup["config"]["e1_params"]))
    print("\nDONE: figures in %s" % OUT)


if __name__ == "__main__":
    main()
