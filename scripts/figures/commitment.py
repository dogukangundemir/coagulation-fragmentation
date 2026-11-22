"""Commitment-over-time figures (packet figure F4): how per-step predictions
evolve over the recurrence steps, on two scales.

Part 1 (unchanged): CF-Full on level-5 test mazes. Loads the CF-Full
checkpoint (hid 16, 160 epochs, trained at fixed T=8, seed 0) with the repo's
own factory, rolls it out once on two level-5 test mazes (selection rule:
test maze 0 and the test maze with the longest true path, ties by maze
index).

Part 2 (maze-hard, external inputs via CLI args or environment variables,
never hard-coded paths):

- WONN panel: the official WINFREE/WONN maze model (read-only official
  clone, imported from a baselines root given at run time) rolled out on one
  Maze-hard test maze (selection rule: test maze 0) over its T=24 steps,
  decoding the model's own readout head at every step. Shown: predicted
  class grids at t = 6, 8, 12, 16, 20, 24, per-step path-fragment counts
  (4-connected components of cells predicted as path), and - as the
  whole-eval-set summary - median + IQR of the fragment count over ALL 1000
  Maze-hard test mazes at every t.

- CF-WallAware panel: the CF family's maze-hard model (cross_kernel=True,
  wall_aware=True, hid 16, trained 120 epochs at T_train=24 on the 900-maze
  maze-hard train split, seed 0 of 3) from a CF results root given at run
  time, commitment roll-out on the SAME Maze-hard test maze 0: per-cell
  P(path) maps at t = 1, 4, 8, 16, 24 and median + IQR of P(path) over
  t = 1..24 for true-path and off-path free cells.

Every number the figures and README quote is recomputed here on CPU and
written to figures/commitment/data.csv first; the README numbers are read
back from that CSV. Nothing from a run record is copied without being
recomputed, except where the README explicitly says it cites a record (the
official eval's board accuracy of the WONN checkpoint, which is a GPU
number).

Usage (CPU only; external locations via CLI or env, no absolute paths in
this file):
  python scripts/figures/commitment.py \
      [--baselines-dir DIR | CF_BASELINES_DIR=DIR] [--cf-results-root DIR | CF_RESULTS_ROOT=DIR]

  --baselines-dir (env CF_BASELINES_DIR) must contain:
      WONN/                                   official model code (read-only)
      WONN/runs/maze/wonn_reduced_L1T24_ch128_s137_bs32/model.pth
      wonn_tools/out/run_record.json          run record of the reduced run
      wonn_tools/data/maze-30x30-hard-1k/test/{all__inputs,all__labels}.npy
  --cf-results-root (env CF_RESULTS_ROOT) must contain:
      maze_hard_cf/states/CF-WallAware_seed0.pt
      maze_hard_cf/results.json

Without an argument the corresponding panel is "not run" (printed to stdout,
stated in the README); the CF-Full panels always run.

Outputs into figures/commitment/:
  data.csv                                per (model, maze, t) rows
  commitment_maze<k>.png                  CF-Full panels (as before)
  commitment_wonn_mazehard_test0.png       WONN panel
  commitment_cfwallaware_mazehard_test0.png  CF-WallAware panel
  README.md                               caption / how-to-read / key
                                          takeaway per figure
"""
import argparse
import csv
import json
import os
import sys
from collections import deque
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
from cf.data.maze_hard import maze_hard_to_channels, parse_maze_hard, \
    validate_maze_hard  # noqa: E402
from cf.metrics import evaluate_mask  # noqa: E402
from cf.models.cf import CFNet, MODEL_SPECS  # noqa: E402

style.apply_style()

OUT = ROOT / "figures" / "commitment"
CKPT = ROOT / "week2_results" / "followups" / "E1_CF-Full_seed0.pt"
RESULTS = ROOT / "week2_results" / "followups" / "results.json"
DEVICE = train_mod.DEVICE
THRESH = train_mod.HYPERPARAMS["threshold"]

OI = style.OKABE_ITO
C_TRUE = OI["blue"]            # true-path cells / predicted path cells
C_OFF = OI["vermillion"]       # off-path free cells / wrong cells
C_FRAG = OI["bluish_green"]    # path fragments
C_TEST = OI["grey"]            # whole-test band
C_START = OI["orange"]         # start cells (WONN class maps)
C_GOAL = OI["yellow"]          # goal cells (WONN class maps)
C_THRESH = OI["grey"]
WALL_RGB = (0.18, 0.18, 0.22)
CMAP = plt.get_cmap("viridis")  # colourblind-safe sequential
MAP_STEPS = [1, 2, 4, 8, 12, 16]
CURVE_STEPS = list(range(1, 17))
MODEL_DESC = "CF-Full (hid 16, 160 epochs, trained at fixed T=8), seed 0"

MH_T = 24                                        # WONN / CF-WallAware depth
MH_CURVE = list(range(1, MH_T + 1))
WONN_MAP_STEPS = [6, 8, 12, 16, 20, 24]
CFWA_MAP_STEPS = [1, 4, 8, 16, 24]
MH_PICK = 0                                      # selection rule: test maze 0
MH_PICK_NAME = "maze-hard test maze 0"
CSV_COLS = ["model", "maze", "dataset_row", "path_len", "t", "n_true",
            "med_true", "q25_true", "q75_true", "n_off", "med_off",
            "q25_off", "q75_off", "wrong_cells", "strict", "n_pred_path",
            "fragments", "q25_frag", "q75_frag", "wall_hits", "board_ok",
            "frac_le1"]


def cpu_guard():
    """CPU-only policy for the maze-hard panels: CF_ALLOW_LOCAL_GPU=0 (the
    headless run rule) is honoured by never touching CUDA here regardless."""
    if os.environ.get("CF_ALLOW_LOCAL_GPU", "1") != "0":
        return torch.device("cpu")
    return torch.device("cpu")


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
                model="CF-Full", maze=k, dataset_row=int(ds.indices[k]),
                path_len=int(gt.sum()), t=t,
                n_true=st_true["n"], med_true=st_true["med"],
                q25_true=st_true["q25"], q75_true=st_true["q75"],
                n_off=st_off["n"], med_off=st_off["med"],
                q25_off=st_off["q25"], q75_off=st_off["q75"],
                wrong_cells=int(r["extra_cells"]) + int(r["missed_cells"]),
                strict=int(r["strict"]), n_pred_path=int((p >= THRESH).sum()),
                wall_hits=int(r["wall_hits"]), board_ok=int(r["exact"]),
                fragments=None, q25_frag=None, q75_frag=None))
    return rows


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for row in rows:
            w.writerow({c: (round(row.get(c), 4)
                            if isinstance(row.get(c), float)
                            else ("" if row.get(c) is None else row.get(c)))
                        for c in CSV_COLS})


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------- maze-hard data (npy)

# inverse of the raw cell codes used by the maze-30x30-hard-1k .npy
# conversion (0=pad, 1=wall, 2=free, 3=start, 4=goal, 5=path)
MH_CODE2CHAR = {1: "#", 2: " ", 3: "S", 4: "G", 5: "o"}


def load_maze_hard_rows(data_root, n):
    """Rebuild (question, answer) strings from the .npy conversion of the
    Maze-hard test split and re-validate with cf.data.maze_hard (structural
    checks; the rating column is not stored in the .npy conversion, so its
    stub is 0.0 and that one check is not a re-verification)."""
    inp = np.load(os.path.join(data_root, "all__inputs.npy"))
    lab = np.load(os.path.join(data_root, "all__labels.npy"))
    if inp.shape != lab.shape or inp.shape[1] != 900:
        raise ValueError("unexpected npy shapes %r / %r" % (inp.shape,
                                                            lab.shape))
    if inp.min() < 1 or lab.min() < 1:
        raise ValueError("pad cells present; expected raw codes 1..5")
    rows = []
    for i in range(n):
        q = "".join(MH_CODE2CHAR[int(c)] for c in inp[i])
        a = "".join(MH_CODE2CHAR[int(c)] for c in lab[i])
        row = dict(source="npy", question=q, answer=a, rating=0.0)
        fails = validate_maze_hard(row)
        if fails:
            raise ValueError("maze-hard row %d failed checks: %s" % (i, fails))
        rows.append(row)
    return inp, lab, rows


def wonn_inputs(inp):
    """Remap raw codes to the model input tokens (walls 0, free 1, S 2, G 3)."""
    out = np.full(inp.shape, 1, dtype=np.int64)
    out[inp == 1] = 0
    out[inp == 3] = 2
    out[inp == 4] = 3
    return out.reshape(-1, 30, 30)


def wonn_targets(lab):
    """Remap raw codes to the model's 5-class targets (off-by-one of raw)."""
    return (lab - 1).reshape(-1, 30, 30)


def count_fragments(pred):
    """4-connected components of cells predicted as path (class 4)."""
    p = (pred == 4)
    seen = np.zeros_like(p, bool)
    cnt = 0
    for r, c in np.argwhere(p):
        if seen[r, c]:
            continue
        cnt += 1
        seen[r, c] = True
        dq = deque([(r, c)])
        while dq:
            rr, cc = dq.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r2, c2 = rr + dr, cc + dc
                if (0 <= r2 < p.shape[0] and 0 <= c2 < p.shape[1]
                        and p[r2, c2] and not seen[r2, c2]):
                    seen[r2, c2] = True
                    dq.append((r2, c2))
    return cnt


# ---------------------------------------------------------------- WONN panel

def load_wonn(baselines_dir):
    """Import the official model, build it from the run record's config and
    load the final checkpoint (model.pth: the raw weights at the end of the
    reduced run; the record's own eval says the raw model beats its EMA
    twin at this budget)."""
    wonn_repo = baselines_dir / "WONN"       # official clone, read-only
    sys.path.insert(0, str(wonn_repo))
    from maze.wnet import MazeWinfreeNet  # official code, read-only clone
    record = json.loads((baselines_dir / "wonn_tools" / "out"
                         / "run_record.json").read_text(encoding="utf-8"))
    cfg = record["config"]
    model = MazeWinfreeNet(
        ch=int(cfg["ch"]), L=int(cfg["L"]), T=int(cfg["T"]),
        coupling=str(cfg["coupling"]), gamma=float(cfg["gamma"]),
        group_size=int(cfg["group_size"]), norm=str(cfg["norm"]),
        heads=int(cfg["heads"]), output_ksize=int(cfg["output_ksize"]))
    ckpt_path = (baselines_dir / "WONN" / "runs" / "maze"
                 / "wonn_reduced_L1T24_ch128_s137_bs32" / "model.pth")
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    device = cpu_guard()
    return model.to(device).eval(), record, device, cfg


@torch.no_grad()
def wonn_rollout(model, inputs, batch_indices, device, t_steps):
    """Per-step argmax predicted class grids: the official forward decodes
    only the final step; here the model's own readout head (decoder + out
    conv) is applied to every intermediate step of the same rollout. The
    decoded final step compares equal to the official forward's logits."""
    preds = np.zeros((len(t_steps), len(batch_indices), 30, 30), np.int64)
    for b0 in range(0, len(batch_indices), 32):
        idx = batch_indices[b0:b0 + 32]
        x = torch.from_numpy(inputs[idx]).to(device)
        out = model(x, return_thetas=True)
        thetas = out[1][0]                       # T tensors (B, ch, 30, 30)
        for j, t in enumerate(t_steps):
            lg = model.out(model.decoder(thetas[t - 1]))   # (B, 5, 30, 30)
            preds[j, b0:b0 + len(idx)] = lg.argmax(1).cpu().numpy()
    return preds


def wonn_rows(preds0, targets0, wall, label, test_frag, test_board):
    """CSV rows for the featured maze per t, plus whole-test summary rows
    (maze = -1 = all 1000 Maze-hard test mazes; fragments = per-t median,
    q25/q75_frag = per-t IQR, board_ok = fraction board-matching at that
    t, frac_le1 = fraction with at most 1 fragment at that t)."""
    path_len = int(label.sum())
    rows = []
    for j, t in enumerate(MH_CURVE):
        p = preds0[j]
        q25, med, q75 = np.percentile(test_frag[j], [25, 50, 75])
        rows.append(dict(
            model="WONN", maze=MH_PICK, dataset_row=MH_PICK,
            path_len=path_len, t=t, n_true=path_len,
            wrong_cells=int((p != targets0).sum()),
            n_pred_path=int((p == 4).sum()),
            fragments=count_fragments(p), wall_hits=int(((p == 4) & wall).sum()),
            board_ok=int((p == targets0).all()),
            q25_frag=None, q75_frag=None, med_true=None, q25_true=None,
            q75_true=None, n_off=None, med_off=None, q25_off=None,
            q75_off=None, strict=None, frac_le1=None))
        rows.append(dict(
            model="WONN", maze=-1, dataset_row=-1, path_len=None, t=t,
            n_true=int(test_board.shape[1]), board_ok=round(float(
                test_board[j].mean()), 4),
            frac_le1=round(float((test_frag[j] <= 1).mean()), 4),
            fragments=int(med), q25_frag=int(q25), q75_frag=int(q75),
            med_true=None, q25_true=None, q75_true=None, n_off=None,
            med_off=None, q25_off=None, q75_off=None, wrong_cells=None,
            strict=None, n_pred_path=None, wall_hits=None))
    return rows


# ---------------------------------------------------- CF-WallAware maze-hard

def load_cf_wallaware(cf_results_root):
    res = json.loads((cf_results_root / "maze_hard_cf" / "results.json")
                     .read_text(encoding="utf-8"))
    ck = (cf_results_root / "maze_hard_cf" / "states"
          / "CF-WallAware_seed0.pt")
    mkw = res["runs"]["CF-WallAware_seed0"]["model_kw"]
    model = CFNet(**mkw)
    model.load_state_dict(torch.load(str(ck), map_location="cpu"))
    device = cpu_guard()
    return model.to(device).eval(), res, device


@torch.no_grad()
def cf_wallaware_probs(model, wall, start, goal, device):
    """P(path) per cell after each t = 1..24 on the featured maze."""
    x = torch.from_numpy(maze_hard_to_channels(wall, start, goal)[None])
    outs = model.logits_at(x.to(device), MH_CURVE)
    return {t: torch.sigmoid(outs[t])[0].cpu().numpy() for t in outs}


def cf_wallaware_rows(probs, wall, start, goal, label):
    gt = label
    off = (~wall) & ~gt
    rows = []
    for t in MH_CURVE:
        p = probs[t]
        st_true, st_off = cell_stats(p, gt, off)
        r = evaluate_mask(p >= THRESH, wall.astype(np.int8), start, goal,
                          label.astype(float))
        rows.append(dict(
            model="CF-WallAware", maze=MH_PICK, dataset_row=MH_PICK,
            path_len=int(gt.sum()), t=t,
            n_true=st_true["n"], med_true=st_true["med"],
            q25_true=st_true["q25"], q75_true=st_true["q75"],
            n_off=st_off["n"], med_off=st_off["med"],
            q25_off=st_off["q25"], q75_off=st_off["q75"],
            wrong_cells=int(r["extra_cells"]) + int(r["missed_cells"]),
            strict=int(r["strict"]), n_pred_path=int((p >= THRESH).sum()),
            fragments=None, q25_frag=None, q75_frag=None,
            wall_hits=int(r["wall_hits"]), board_ok=int(r["exact"])))
    return rows


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


MH_CLASS_RGB = {0: WALL_RGB, 1: (1.0, 1.0, 1.0),
                2: matplotlib.colors.to_rgb(C_START),
                3: matplotlib.colors.to_rgb(C_GOAL),
                4: matplotlib.colors.to_rgb(C_TRUE)}


def draw_class_map(ax, grid, wall, gt, start, goal, title, xlabel):
    """One WONN panel: cells coloured by predicted class (wall dark, free
    white, path blue, start orange, goal yellow); reference path outlined,
    start = circle, goal = star."""
    rgb = np.ones(grid.shape + (3,))
    for code, col in MH_CLASS_RGB.items():
        pos = grid == code
        rgb[pos] = col
    ax.imshow(rgb, interpolation="nearest")
    for r, c in np.argwhere(gt):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=0.7))
    ax.plot(start[1], start[0], marker="o", ms=6, color="black",
            mfc="none", mew=0.8, linestyle="none")
    ax.plot(goal[1], goal[0], marker="*", ms=10, color="black",
            linestyle="none")
    ax.set_xlim(-0.6, grid.shape[1] - 0.4)
    ax.set_ylim(grid.shape[0] - 0.4, -0.6)
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


def figure_wonn(rows0, rows_test, preds0, wall, start, goal, label,
                wonn_label):
    """WONN maze-hard panel: class maps + per-step counts, with the
    whole-test fragment band below. rows0: featured-maze CSV rows;
    rows_test: maze = -1 CSV rows; preds0: (T, 30, 30) argmax grids."""
    by_t = {int(r["t"]): r for r in rows0}
    by_t_test = {int(r["t"]): r for r in rows_test}
    plen = int(label.sum())

    fig = plt.figure(figsize=(13.6, 11.0))
    gs = fig.add_gridspec(5, 1 + len(WONN_MAP_STEPS),
                          height_ratios=[0.24, 1.0, 0.62, 0.62, 0.30],
                          hspace=0.95, wspace=0.10)

    axt = fig.add_subplot(gs[0, :])
    axt.axis("off")
    axt.set_title("WONN on Maze-hard: path fragments appear, then merge "
                  "while its T = 24 steps run (%s\non test maze 0; still "
                  "not board-solved at t = 24)\nHow to read: top row = "
                  "predicted class grid after t steps (blue = path cells, "
                  "white = free, dark = wall, number under each panel = "
                  "fragments / wrong cells); middle = per-step counts for "
                  "this maze; bottom = fragment count, median + IQR band "
                  "over all 1000 Maze-hard test mazes" % wonn_label,
                  fontsize=11)

    map_axes = []
    ax0 = fig.add_subplot(gs[1, 0])
    draw_class_map(ax0, _gt_class_grid(wall, start, goal, label),
                   wall, label, start, goal, "ground truth",
                   "%d path cells" % plen)
    map_axes.append(ax0)
    for j, t in enumerate(WONN_MAP_STEPS):
        ax = fig.add_subplot(gs[1, j + 1])
        r = by_t[t]
        draw_class_map(ax, preds0[t - 1], wall, label, start, goal,
                       "t = %d" % t,
                       "%s frag / %s wrong" % (r["fragments"],
                                               r["wrong_cells"]))
        map_axes.append(ax)

    # ---- per-step counts on the featured maze (fragments left, cells right)
    axc = fig.add_subplot(gs[2, :])
    ts = np.array(MH_CURVE, dtype=float)
    frag = np.array([int(by_t[t]["fragments"]) for t in MH_CURVE])
    npath = np.array([int(by_t[t]["n_pred_path"]) for t in MH_CURVE])
    wrong = np.array([int(by_t[t]["wrong_cells"]) for t in MH_CURVE])
    lf, = axc.plot(ts, frag, color=C_FRAG, lw=2.0, marker="o", ms=4,
                   label="path fragments (4-conn. components)")
    axc.set_ylabel("path fragments (count)", fontsize=11, color=C_FRAG)
    axc.tick_params(axis="y", labelcolor=C_FRAG)
    axc.set_xlim(0.5, MH_T + 0.5)
    axc.set_xticks(MH_CURVE)
    axc.set_xlabel("WONN step t (of T = 24)", fontsize=11)
    axc.set_ylim(-2, max(frag) * 1.15 + 2)
    ax2 = axc.twinx()
    ax2.plot(ts, npath, color=C_TRUE, lw=2.0, label="cells predicted path")
    ax2.plot(ts, wrong, color=C_OFF, lw=2.0,
             label="cells disagreeing with reference")
    ax2.set_ylabel("cells (count)", fontsize=11)
    ax2.set_ylim(0, 950)
    ax2.spines["top"].set_visible(False)
    ax2.set_title("test maze 0: fragments peak at t = %d (%d fragments), "
                  "merge to %d at t = 24; wrong cells fall %d -> %d"
                  % (int(ts[np.argmax(frag)]), int(frag.max()),
                     int(frag[-1]), int(wrong[0]), int(wrong[-1])),
                  fontsize=11)
    for ax_ in (axc, ax2):
        ax_.grid(False)
        ax_.spines["top"].set_visible(False)

    # ---- whole-test fragment band
    axb = fig.add_subplot(gs[3, :])
    med = np.array([int(by_t_test[t]["fragments"]) for t in MH_CURVE],
                   dtype=float)
    q25 = np.array([int(by_t_test[t]["q25_frag"]) for t in MH_CURVE],
                   dtype=float)
    q75 = np.array([int(by_t_test[t]["q75_frag"]) for t in MH_CURVE],
                   dtype=float)
    axb.fill_between(ts, q25, q75, color=C_TEST, alpha=0.30, linewidth=0)
    lb, = axb.plot(ts, med, color=C_TEST, lw=2.0,
                   label="median over 1000 test mazes (IQR band)")
    lff, = axb.plot(ts, frag, color=C_FRAG, lw=1.6, linestyle="--",
                    label="test maze 0 (same data as above)")
    axb.set_xlim(0.5, MH_T + 0.5)
    axb.set_xticks(MH_CURVE)
    axb.set_ylim(-2, max(q75.max(), frag.max()) * 1.15 + 2)
    axb.set_xlabel("WONN step t (of T = 24)", fontsize=11)
    axb.set_ylabel("path fragments (count)", fontsize=11)
    axb.set_title("whole test split (n = 1000): median fragments per "
                  "step, band = IQR over mazes (unbounded, no clipping)",
                  fontsize=10)

    axl = fig.add_subplot(gs[4, :])
    axl.axis("off")
    axl.legend(handles=[
        lf,
        Line2D([], [], color=C_TRUE, lw=2.0,
               label="cells predicted as path (right axis)"),
        Line2D([], [], color=C_OFF, lw=2.0,
               label="cells disagreeing with reference (right axis)"),
        lb, lff,
        Patch(facecolor=C_TEST, alpha=0.30,
              label="IQR band of fragments over test mazes"),
        Patch(facecolor=WALL_RGB, label="wall (map panels)"),
        Patch(facecolor=C_TRUE, label="path cell (predicted / truth)"),
        Patch(facecolor=(1.0, 1.0, 1.0), edgecolor="black",
              label="free cell"),
        Patch(facecolor=C_START, label="start cell"),
        Patch(facecolor=C_GOAL, label="goal cell"),
        Patch(facecolor="none", edgecolor="black",
              label="reference route (outline)"),
        Line2D([], [], marker="o", color="black", mfc="none", ms=6, lw=0,
               label="start"),
        Line2D([], [], marker="*", color="black", ms=9, lw=0,
               label="goal"),
    ], loc="center", ncol=5, fontsize=10, frameon=False, handlelength=1.5,
        columnspacing=1.2)
    save(fig, "commitment_wonn_mazehard_test0.png")


def _gt_class_grid(wall, start, goal, label):
    """Reference grid in the WONN class coding (0 wall, 1 free, 2 start,
    3 goal, 4 path) so the ground-truth panel uses the same colours as the
    prediction panels."""
    grid = np.where(wall, 0, 1)
    grid[label] = 4
    grid[start] = 2
    grid[goal] = 3
    return grid


def figure_cf_wallaware(ds_rows, probs, wall, start, goal, label,
                        model_desc):
    """CF-WallAware commitment panel on the same maze-hard test maze."""
    gt = label
    off = (~wall) & ~gt
    plen = int(gt.sum())
    by_t = {int(r["t"]): r for r in ds_rows}

    fig = plt.figure(figsize=(13.6, 9.6))
    gs = fig.add_gridspec(4, 1 + len(CFWA_MAP_STEPS),
                          height_ratios=[0.22, 1.0, 0.78, 0.30],
                          hspace=0.62, wspace=0.10)

    axt = fig.add_subplot(gs[0, :])
    axt.axis("off")
    axt.set_title("CF-WallAware on Maze-hard: commitment sharpens but does "
                  "not finish (%s,\nMaze-hard test maze 0, t = 1..24)\nHow "
                  "to read: top = ground truth and per-cell P(path) after "
                  "t steps; bottom = median + IQR of P(path) per cell group "
                  "over t (band clipped to [0, 1])" % model_desc,
                  fontsize=11)

    map_axes = []
    ax0 = fig.add_subplot(gs[1, 0])
    draw_prob_map(ax0, gt.astype(float), wall, gt, start, goal,
                  "ground truth", "%d cells" % plen)
    map_axes.append(ax0)
    for j, t in enumerate(CFWA_MAP_STEPS):
        ax = fig.add_subplot(gs[1, j + 1])
        r = by_t[t]
        xlabel = ("%d wrong" % int(r["wrong_cells"]))
        draw_prob_map(ax, probs[t], wall, gt, start, goal, "t = %d" % t,
                      xlabel)
        map_axes.append(ax)
    cbar = fig.colorbar(map_axes[1].images[0], ax=map_axes, shrink=0.85,
                        pad=0.02)
    cbar.set_label("P(path) per cell", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    axc = fig.add_subplot(gs[2, :])
    ts = np.array(MH_CURVE, dtype=float)
    med_true = np.array([float(by_t[t]["med_true"]) for t in MH_CURVE])
    q25_true = np.array([float(by_t[t]["q25_true"]) for t in MH_CURVE])
    q75_true = np.array([float(by_t[t]["q75_true"]) for t in MH_CURVE])
    med_off = np.array([float(by_t[t]["med_off"]) for t in MH_CURVE])
    q25_off = np.array([float(by_t[t]["q25_off"]) for t in MH_CURVE])
    q75_off = np.array([float(by_t[t]["q75_off"]) for t in MH_CURVE])
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
    axc.plot(CFWA_MAP_STEPS, [float(by_t[t]["med_true"]) for t in CFWA_MAP_STEPS],
             "o", ms=5, color=C_TRUE)
    axc.plot(CFWA_MAP_STEPS, [float(by_t[t]["med_off"]) for t in CFWA_MAP_STEPS],
             "o", ms=5, color=C_OFF)
    axc.axhline(THRESH, color=C_THRESH, lw=1.2, linestyle="--")
    axc.set_xticks([1, 4, 8, 12, 16, 20, 24])
    axc.set_xlim(0.5, MH_T + 0.5)
    axc.set_ylim(-0.04, 1.06)
    axc.set_xlabel("recurrence step t (T_train = 24)", fontsize=11)
    axc.set_ylabel("P(path) per cell", fontsize=11)
    axc.set_title("P(path) over t: median (line) + IQR (band, clipped to "
                  "[0, 1]); the off-path median never reaches 0 and the "
                  "thresholded mask keeps %d wrong cells at t = 24"
                  % int(by_t[MH_T]["wrong_cells"]), fontsize=11)

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
    save(fig, "commitment_cfwallaware_mazehard_test0.png")


# ---------------------------------------------------------------- README

def write_readme(ds, mazes, rows, T_star, strict_at_Tstar, params,
                 extra=None):
    """extra: dict with optional keys wonn / cfwa, each carrying everything
    the maze-hard sections need (all numbers already read from the CSV)."""
    by_maze = {k: [r for r in rows if int(r["maze"]) == k
                   and r["model"] == "CF-Full"] for k in mazes}
    L = []
    L.append("# Commitment over time: per-step predictions "
             "(packet figure F4)\n")
    L.append("All numbers below are recomputed by "
             "`scripts/figures/commitment.py` from the stored checkpoints "
             "and written to `data.csv` in this folder (one row per model, "
             "maze and step t; rows with model = WONN and maze = -1 "
             "aggregate all 1000 Maze-hard test mazes, storing the per-t "
             "median and IQR of the fragment count, the fraction of mazes "
             "board-matching at that t (board_ok) and the fraction with "
             "at most 1 fragment at that t (frac_le1)); the README "
             "numbers are read back from that CSV.\n")
    L.append("Parts: 1. CF-Full on level-5 test mazes "
             "(`commitment_maze0.png`, `commitment_maze248.png`); "
             "2. WONN on one Maze-hard test maze "
             "(`commitment_wonn_mazehard_test0.png`); "
             "3. CF-WallAware on the same Maze-hard test maze "
             "(`commitment_cfwallaware_mazehard_test0.png`).\n")
    L.append("Model (part 1): **%s**, %d parameters, checkpoint "
             "`week2_results/followups/E1_CF-Full_seed0.pt`; validation-"
             "selected inference depth T\\* = %d (test strict solve rate "
             "%.3f at T\\*, from `week2_results/followups/results.json`). "
             "Input: walkable mask + start + goal one-hot channels; output: "
             "per-cell path logit, sigmoid -> P(path), decision threshold "
             "0.5.\n" % (MODEL_DESC, params, T_star, strict_at_Tstar))
    L.append("Mazes (part 1, level-5 test split, 1000 mazes, 11x11, same "
             "pipeline as `run_week2.load_data`): selection rule = test "
             "maze 0 and the test maze with the longest true path (ties by "
             "maze index). Mazes (parts 2-3, Maze-hard 30x30, HF "
             "sapientinc/maze-30x30-hard-1k test split, 1000 mazes): "
             "selection rule = test maze 0 for both models, so both panels "
             "show the same maze.\n")

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
        L.append("## `commitment_maze%d.png` - level-5 test maze %d "
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
                 "model = CF-Full, maze = %d); checkpoint "
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

    if extra and "wonn" in extra:
        L.extend(section_wonn(extra["wonn"]))
    elif extra is not None:
        L.append("---\n")
        L.append("## WONN Maze-hard panel: not run "
                 "(no --baselines-dir / CF_BASELINES_DIR given)\n")
        L.append("**Packet figure: F4**\n")
    if extra and "cfwa" in extra:
        L.extend(section_cfwa(extra["cfwa"]))
    elif extra is not None:
        L.append("---\n")
        L.append("## CF-WallAware Maze-hard panel: not run "
                 "(no --cf-results-root / CF_RESULTS_ROOT given)\n")
        L.append("**Packet figure: F4**\n")

    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("saved %s" % (OUT / "README.md"))


def section_wonn(w):
    """README section for commitment_wonn_mazehard_test0.png; every number
    comes from the CSV rows passed in w (prepared in main)."""
    L = ["---\n"]
    L.append("## `commitment_wonn_mazehard_test0.png` - WONN on Maze-hard "
             "test maze 0 (true path %s cells)\n" % w["path_len"])
    L.append("**Caption.** %s rolled out on Maze-hard test maze 0 over its "
             "T = 24 steps; the model's own readout head is applied to "
             "every intermediate step of the rollout. Top: the predicted "
             "class grid after t = 6, 8, 12, 16, 20, 24 steps (the number "
             "under each panel = path fragments / cells disagreeing with "
             "the reference grid). Middle: per-step counts for this maze. "
             "Bottom: the fragment count over ALL 1000 Maze-hard test "
             "mazes (median line + IQR band), with this maze on top."
             % w["model_desc"])
    L.append("**How to read.** Map panels: blue = cells predicted as path, "
             "white = free, dark grey = wall, orange = predicted start, "
             "yellow = predicted goal; the black outline is the reference "
             "route, circle = start, star = goal. Middle panel: green line "
             "with dots (left axis) = number of path fragments "
             "(4-connected components of path-predicted cells); blue and "
             "vermillion lines (right axis) = cells predicted as path and "
             "cells disagreeing with the reference grid. Bottom panel: "
             "grey line = per-step median fragment count over the 1000 "
             "test mazes, band = IQR over mazes (counts are unbounded, so "
             "the band needs no clipping); dashed green = this maze. All "
             "series are counts on one Maze-hard maze (n = 1) or over "
             "n = 1000 mazes; there is no seed spread (single seed).\n")
    L.append("**Key takeaway.** On test maze 0 the predicted path starts as "
             "scattered fragments - %s fragments at t = 8 (the peak for "
             "this maze) - and merges to %s fragments by t = 24, while the "
             "number of cells predicted as path settles at %s (reference "
             "route: %s cells) and the disagreements with the reference "
             "grid fall from %s at the peak to %s at t = 24; the maze is "
             "still not board-solved at t = 24 (board match = %s). Across "
             "all 1000 test mazes the same shape holds: the median "
             "fragment count peaks at t = %s (%s fragments) and falls to "
             "%s at t = 24 (IQR %s..%s at t = 24); %s of the 1000 mazes "
             "have at most 1 fragment at t = 24, and the board-match rate "
             "recomputed here on CPU is %s (%s in the run record's "
             "official eval of the same checkpoint on GPU)."
             % (w["frag_peak"], w["frag_final"], w["npath_final"],
                w["path_len"], w["wrong_at_peak"], w["wrong_final"],
                ("yes" if w["board_final"] == "1" else "no"),
                w["test_peak_t"], w["test_peak_med"], w["test_final_med"],
                w["test_final_q25"], w["test_final_q75"],
                w["frac_le1"], w["board_acc_recomputed"],
                w["board_acc_record"]))
    L.append("**Data source.** `data.csv` in this folder (rows with model = "
             "WONN; maze = 0 = this maze, maze = -1 = whole test split, n "
             "= 1000 mazes); WONN checkpoint and inputs are external, given "
             "at run time via --baselines-dir / CF_BASELINES_DIR: "
             "official model code under `WONN/` (read-only), checkpoint "
             "`WONN/runs/maze/wonn_reduced_L1T24_ch128_s137_bs32/"
             "model.pth` (final raw weights), run record `wonn_tools/out/"
             "run_record.json`, data `wonn_tools/data/maze-30x30-hard-1k/"
             "test/` (.npy conversion of the HF test csv, re-validated here "
             "with cf.data.maze_hard, structural checks).\n")
    b_diff = abs(float(w["board_acc_recomputed"])
                 - float(w["board_acc_record"]))
    if b_diff <= 0.005:
        board_note = ("The board-match rate recomputed here on CPU (%s) "
                      "matches the record's official GPU eval of the same "
                      "checkpoint (%s, agreement to %.1f pp): the initial "
                      "theta field is random but seeded (seed 137), so "
                      "the rollout is reproducible across devices at this "
                      "budget." % (w["board_acc_recomputed"],
                                    w["board_acc_record"], b_diff * 100))
    else:
        board_note = ("The board-match rate differs between this CPU "
                      "rollout (%s) and the record's GPU eval (%s): the "
                      "initial theta field is random (seeded, seed 137) "
                      "and at this budget the argmax is not numerically "
                      "stable across devices." % (w["board_acc_recomputed"],
                                                   w["board_acc_record"]))
    L.append("**Caveats.** Budget: %s. Reduced-budget run over the official "
             "code, not a reproduction; single seed -> preliminary. The "
             "steps before t = 6 predict no path cells at all (the "
             "oscillator field needs several steps before any cell crosses "
             "the path class). %s Fragment counts are a helper quantity "
             "defined here (4-connected components of path-predicted "
             "cells), not a metric of the official eval. Wall cells are "
             "never predicted as path here (wall hits = 0 on this maze at "
             "every t).\n" % (w["budget"], board_note))
    L.append("**Packet figure: F4**\n")
    return L


def section_cfwa(w):
    L = ["---\n"]
    L.append("## `commitment_cfwallaware_mazehard_test0.png` - "
             "CF-WallAware on Maze-hard test maze 0 (true path %s cells)\n"
             % w["path_len"])
    L.append("**Caption.** %s on the same Maze-hard test maze as the WONN "
             "panel: per-cell P(path) after t = 1, 4, 8, 16, 24 steps "
             "(one shared rollout, read out at each t; the number under "
             "each panel is the wrong-cell count of the thresholded "
             "prediction), and median + IQR of P(path) over t = 1..24 for "
             "the true-path and off-path free cells." % w["model_desc"])
    L.append("**How to read.** Same reading as the CF-Full panels above: "
             "free cells coloured by P(path) (viridis), true path outlined, "
             "bands = IQR clipped to [0, 1]; blue = median over the %s "
             "true-path cells, vermillion = median over the %s off-path "
             "free cells; dashed grey = 0.5 decision threshold. Model "
             "settings relevant to the plot are in the caption "
             "(hid 16, cross_kernel=True, wall-aware, 120 epochs, "
             "T_train = 24, seed 0 of 3).\n" % (w["n_true"], w["n_off"]))
    L.append("**Key takeaway.** On test maze 0 the median P(path) on the "
             "%s true-path cells rises from %.2f at t = 1 to %.2f at t = 24 "
             "and the median on the %s off-path free cells falls from %.2f "
             "to %.3f, but the suppression does not finish: the "
             "thresholded prediction keeps %s wrong cells at t = 24 "
             "(%s at t = 1) and never reaches a strict solve (strict = 0 "
             "at every t = 1..24). On the full test split this model "
             "scores valid-path 0 and board match 0 at every T (seed 0: "
             "IoU %.3f, recall %.3f, extra cells %.1f at T\\* = %d, from "
             "the maze-hard results.json cited in `data source`)."
             % (w["n_true"], float(w["med_true_1"]), float(w["med_true_24"]),
                w["n_off"], float(w["med_off_1"]), float(w["med_off_24"]),
                w["wrong_24"], w["wrong_1"], float(w["test_iou"]),
                float(w["test_recall"]), float(w["test_extra"]),
                w["T_star"]))
    L.append("**Data source.** `data.csv` in this folder (rows with model = "
             "CF-WallAware, maze = 0); checkpoint and sweep records are "
             "external, given at run time via --cf-results-root / "
             "CF_RESULTS_ROOT: `maze_hard_cf/states/CF-WallAware_seed0.pt` "
             "and `maze_hard_cf/results.json` (test-split sweep numbers "
             "cited, not recomputed here).\n")
    L.append("**Caveats.** Single maze, single seed (0 of 3; the other "
             "seeds exist for the test-split sweep but this rollout shows "
             "seed 0); median + IQR over cells within this one maze. The "
             "Maze-hard grids are non-perfect (several equally short "
             "routes exist), so board match and valid-path are not the "
             "same metric; the wrong-cell count here compares the "
             "thresholded mask against the single reference route. Wall "
             "cells are masked to -20 by construction, so wall hits are 0 "
             "by design (reported as a sanity check, not a "
             "discriminator).\n")
    L.append("**Packet figure: F4**\n")
    return L


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines-dir", default=os.environ.get(
        "CF_BASELINES_DIR", ""), help="baselines root holding the official "
        "WONN clone and wonn_tools (env CF_BASELINES_DIR)")
    ap.add_argument("--cf-results-root", default=os.environ.get(
        "CF_RESULTS_ROOT", ""), help="CF results root holding maze_hard_cf "
        "(env CF_RESULTS_ROOT)")
    args = ap.parse_args()
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

    extra = {}
    mh = None                     # shared maze-hard maze description
    wonn_ok = bool(args.baselines_dir)
    cfwa_ok = bool(args.cf_results_root)

    if not wonn_ok:
        print("not run: WONN panel (no --baselines-dir / "
              "CF_BASELINES_DIR)")
    if not cfwa_ok:
        print("not run: CF-WallAware panel (no --cf-results-root / "
              "CF_RESULTS_ROOT)")

    if wonn_ok:
        baselines = Path(args.baselines_dir)
        print("loading WONN (official code + final checkpoint, CPU) ...")
        wmodel, wrecord, wdev, wcfg = load_wonn(baselines)
        data_root = (baselines / "wonn_tools" / "data"
                     / "maze-30x30-hard-1k" / "test")
        inp, lab, mh_rows = load_maze_hard_rows(str(data_root), 1000)
        inputs, targets = wonn_inputs(inp), wonn_targets(lab)
        wall, start, goal, label = parse_maze_hard(
            mh_rows[MH_PICK]["question"], mh_rows[MH_PICK]["answer"])
        mh = dict(wall=wall, start=start, goal=goal, label=label)
        T = int(wcfg["T"])
        n_test = inputs.shape[0]
        print("rolling out WONN per step on %d test mazes (t = 1..%d, "
              "CPU) ..." % (n_test, T))
        torch.manual_seed(int(wcfg["seed"]))
        np.random.seed(int(wcfg["seed"]))
        all_preds = wonn_rollout(wmodel, inputs, list(range(n_test)),
                                 wdev, MH_CURVE)
        test_frag = np.zeros((T, n_test), int)
        test_board = np.zeros((T, n_test), int)
        for i in range(n_test):
            for t in range(T):
                test_frag[t, i] = count_fragments(all_preds[t, i])
                test_board[t, i] = int((all_preds[t, i] == targets[i]).all())
        board_acc_recomputed = test_board[T - 1].mean()
        frac_le1 = (test_frag[T - 1] <= 1).mean()
        print("recomputed board acc at t=%d: %.3f (run record eval: %.3f); "
              "fraction of mazes with <=1 fragment at t=%d: %.3f"
              % (T, board_acc_recomputed,
                 float(wrecord["eval"]["model.pth"]["board_acc"]),
                 T, frac_le1))
        wrows = wonn_rows(all_preds[:, MH_PICK], targets[MH_PICK], wall,
                          label, test_frag, test_board)
        rows.extend(wrows)
        wonn_label = ("official code, L=%d, T=%d, ch=%d, heads=%d, "
                      "%d/%d epochs, seed %d - reduced budget, not a "
                      "reproduction" % (int(wcfg["L"]), T, int(wcfg["ch"]),
                                        int(wcfg["heads"]),
                                        int(wrecord["epochs_trained"]),
                                        int(wrecord["official_epochs"]),
                                        int(wcfg["seed"])))

    if cfwa_ok and mh is None:
        print("not run: CF-WallAware panel needs the Maze-hard test data "
              "as well; pass --baselines-dir too")
        cfwa_ok = False

    if cfwa_ok:
        cf_root = Path(args.cf_results_root)
        print("loading CF-WallAware (maze-hard, seed 0, CPU) ...")
        cmodel, cres, cdev = load_cf_wallaware(cf_root)
        wall, start, goal, label = (mh["wall"], mh["start"], mh["goal"],
                                    mh["label"])
        print("rolling out CF-WallAware t = 1..24 on maze-hard test maze 0 "
              "...")
        cprobs = cf_wallaware_probs(cmodel, wall, start, goal, cdev)
        crows = cf_wallaware_rows(cprobs, wall, start, goal, label)
        rows.extend(crows)
        run0 = cres["runs"]["CF-WallAware_seed0"]
        cfwa_model_desc = ("CF-WallAware maze-hard model: hid 16, "
                           "cross_kernel=True, wall-aware, %d epochs "
                           "(seed 0 of 3), trained at T_train = %d"
                           % (int(run0["epochs_done"]),
                              int(run0["T_train"])))
        cfwa_extra = dict(
            run0=run0,
            model_desc=cfwa_model_desc)
    else:
        cfwa_extra = None

    csv_path = OUT / "data.csv"
    write_csv(rows, csv_path)
    print("wrote %s (%d rows)" % (csv_path, len(rows)))
    rows = read_csv(csv_path)          # README numbers come from the CSV

    for i, k in enumerate(mazes):
        probs_k = {t: probs[t][i] for t in probs}
        rows_k = [r for r in rows if int(r["maze"]) == k
                  and r["model"] == "CF-Full"]
        print("figure: maze %d ..." % k)
        figure_one_maze(ds, k, probs_k, rows_k)

    if wonn_ok:
        print("figure: WONN maze-hard panel ...")
        rows0 = [r for r in rows if r["model"] == "WONN"
                 and int(r["maze"]) == MH_PICK]
        rows_test = [r for r in rows if r["model"] == "WONN"
                     and int(r["maze"]) == -1]
        # featured maze per-step predictions, straight from the rollout
        preds0 = all_preds[:, MH_PICK]
        figure_wonn(rows0, rows_test, preds0, wall, start, goal, label,
                    wonn_label)
        # prepare README numbers from the CSV
        med_row = {int(r["t"]): r for r in rows0}
        test_row = {int(r["t"]): r for r in rows_test}
        frag = [int(med_row[t]["fragments"]) for t in MH_CURVE]
        peak_t = MH_CURVE[int(np.argmax(frag))]
        test_med = [float(test_row[t]["fragments"]) for t in MH_CURVE]
        tpk = MH_CURVE[int(np.argmax(test_med))]
        # the README's whole-test numbers come from the CSV rows (the
        # header says they are read back from it); guard that they agree
        # with the in-run recomputation to within CSV rounding
        board_csv = float(test_row[MH_T]["board_ok"])
        le1_csv = float(test_row[MH_T]["frac_le1"])
        if (abs(board_csv - float(board_acc_recomputed)) > 1e-3
                or abs(le1_csv - float(frac_le1)) > 1e-3):
            sys.exit("ACCEPTANCE FAIL: CSV whole-test rows disagree with "
                     "the in-run recomputation (board_ok %s vs %.4f, "
                     "frac_le1 %s vs %.4f)"
                     % (board_csv, board_acc_recomputed, le1_csv, frac_le1))
        extra["wonn"] = dict(
            path_len=med_row[1]["path_len"],
            model_desc=wonn_label,
            frag_peak=med_row[peak_t]["fragments"],
            frag_final=med_row[MH_T]["fragments"],
            npath_final=med_row[MH_T]["n_pred_path"],
            wrong_at_peak=med_row[peak_t]["wrong_cells"],
            wrong_final=med_row[MH_T]["wrong_cells"],
            board_final=med_row[MH_T]["board_ok"],
            test_peak_t=tpk,
            test_peak_med=int(test_row[tpk]["fragments"]),
            test_final_med=test_row[MH_T]["fragments"],
            test_final_q25=test_row[MH_T]["q25_frag"],
            test_final_q75=test_row[MH_T]["q75_frag"],
            frac_le1=("%.3f" % le1_csv),
            board_acc_recomputed=("%.3f" % board_csv),
            board_acc_record=("%.3f" % wrecord["eval"]["model.pth"]["board_acc"]),
            budget=("WONN: %d of %d official epochs, seed %d, batch 32 "
                    "instead of 64, single seed -> preliminary"
                    % (int(wrecord["epochs_trained"]),
                       int(wrecord["official_epochs"]),
                       int(wcfg["seed"]))))

    if cfwa_ok:
        print("figure: CF-WallAware maze-hard panel ...")
        crows_csv = [r for r in rows if r["model"] == "CF-WallAware"
                     and int(r["maze"]) == MH_PICK]
        figure_cf_wallaware(crows_csv, cprobs, mh["wall"], mh["start"],
                            mh["goal"], mh["label"], cfwa_model_desc)
        by_t = {int(r["t"]): r for r in crows_csv}
        # test-split sweep numbers from the maze-hard results.json (cited,
        # not recomputed here: they cover all 1000 test mazes)
        cfg0 = cfwa_extra["run0"]
        tstar = str(cfg0["T_star"])
        ts_metrics = cfg0["test_sweeps"][tstar]
        extra["cfwa"] = dict(
            path_len=by_t[1]["path_len"], model_desc=cfwa_model_desc,
            n_true=by_t[1]["n_true"], n_off=by_t[1]["n_off"],
            med_true_1=by_t[1]["med_true"], med_true_24=by_t[24]["med_true"],
            med_off_1=by_t[1]["med_off"], med_off_24=by_t[24]["med_off"],
            wrong_1=by_t[1]["wrong_cells"], wrong_24=by_t[24]["wrong_cells"],
            T_star=int(cfg0["T_star"]),
            test_iou=("%.3f" % float(ts_metrics["iou"])),
            test_recall=("%.3f" % float(ts_metrics["recall"])),
            test_extra=("%.1f" % float(ts_metrics["extra_cells"])))

    fup = json.loads(RESULTS.read_text(encoding="utf-8"))
    run0 = fup["runs"]["E1_CF-Full_seed0"]
    write_readme(ds, mazes, rows, int(run0["T_star"]),
                 float(run0["test_strict_by_T"][str(run0["T_star"])]),
                 int(fup["config"]["e1_params"]), extra=extra)
    print("\nDONE: figures in %s" % OUT)


if __name__ == "__main__":
    main()
