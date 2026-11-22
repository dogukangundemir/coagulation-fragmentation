"""Per-maze prediction figures: what each trained model actually outputs.

Loads the five trained model families (seed 0/1/2 checkpoints, built with the
repo's own factories exactly as run_followups.py / run_week2.py do),
recomputes every prediction on the level-5 test split (1000 mazes) at the
stored validation-selected horizon T* and writes into figures/maze_paths/:

  data.csv                  one row per (model, seed, maze): path length,
                            strict/exact, wrong/extra/missed cells, wall
                            hits, IoU
  path_gallery_1.png        the 8 selected test mazes (4 per figure) -
  path_gallery_2.png         ground truth | every model's prediction,
                             with the model / T-at-inference column
                             headers repeated on every maze row so each
                             PNG is readable on its own
  coverage_<model>.png      25x40 grid of all 1000 test mazes,
                            solved / not solved at T* (seed 0)
  failure_anatomy.png       per model: histogram of wrong cells and strict
                            solve rate by true path length (binned,
                            n per bin, seeds pooled)
  thinking_steps.png        two test mazes: thresholded predictions after
                            t = 1/2/4/8/12/16/24/32 recurrence steps,
                            CF-Full (160 epochs) vs RecurrentCNN (fixed T=8)
  README.md                 caption / how-to-read / key takeaway per figure;
                            every number comes from data.csv

Gallery-row selection rule (level-5 test split, seed-0 models at their own
T*): 4 mazes drawn with numpy RNG seed 0, 2 mazes where the models disagree
on the solve verdict (largest total wrong cells first, ties by maze index),
and the 2 mazes with the longest true path (ties by maze index).
"""
import json
import statistics
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

OUT = ROOT / "figures" / "maze_paths"
FOLLOWUPS = ROOT / "week2_results" / "followups"
WEEK2 = ROOT / "week2_results"
DEVICE = train_mod.DEVICE
THRESH = train_mod.HYPERPARAMS["threshold"]
SEEDS = [0, 1, 2]
SPLITS = ["test"]
COVERAGE_COLS = 40     # 40 x 25 = 1000 mazes per grid
TOL = 0.001

OI = style.OKABE_ITO
# per-model line colours for the aggregate figures (Okabe-Ito)
FC_COLORS = {
    "cf-full-160ep": OI["blue"],
    "cf-nofrag-160ep": OI["bluish_green"],
    "rcnn-varT": OI["vermillion"],
    "rcnn-fixedT8": OI["orange"],
    "cf-full-40ep": OI["reddish_purple"],
}
FC_BINS = [1, 10, 16, 22, 28, 34, 41]   # true path length (cells), 6 bins
THINK_MODELS = ["cf-full-160ep", "rcnn-fixedT8"]
THINK_SHORT = {"cf-full-160ep": "CF-Full, 160 epochs",
               "rcnn-fixedT8": "RecurrentCNN, fixed T=8"}
THINK_STEPS = [1, 2, 4, 8, 12, 16, 24, 32]
C_CORRECT = OI["blue"]            # predicted cell on the true path
C_EXTRA = OI["vermillion"]        # false positive (extra cell)
C_MISSED = OI["reddish_purple"]   # missed true-path cell
C_SOLVED = OI["bluish_green"]
C_UNSOLVED = OI["vermillion"]
C_BAND = OI["sky_blue"]
WALL_RGB = (0.18, 0.18, 0.22)
FREE_RGB = (0.95, 0.95, 0.92)
PATH_RGB = (0.70, 0.70, 0.78)

SPLIT_DESC = {
    "test": "level-5 test split (11x11 mazes)",
}


# ---------------------------------------------------------------- models

def load_T_stars():
    """Validation-selected T* per (model family, seed) from the stored
    selection JSONs (written by the repo's sweep / selection scripts)."""
    fup = json.loads((FOLLOWUPS / "results.json").read_text(encoding="utf-8"))
    t_fup = {(r["exp"], int(r["seed"])): int(r["T_star"])
             for r in fup["runs"].values()}
    sel = json.loads((WEEK2 / "valT_selection.json").read_text(encoding="utf-8"))
    t_w2 = {(r["model"], int(r["seed"])): int(r["T_star"])
            for r in sel["selections"]}
    return t_fup, t_w2, fup


def model_registry():
    t_fup, t_w2, fup = load_T_stars()
    cfg = fup["config"]

    def rcnn():
        return run_week2.make_model("RecurrentCNN")

    return [
        dict(key="cf-full-160ep",
             desc="CF-Full (hid 16, 160 epochs, trained at fixed T=8)",
             short="CF-Full\n160 epochs",
             make=lambda: CFNet(hid=16, **MODEL_SPECS["CF-Full"]),
             path=lambda s: FOLLOWUPS / ("E1_CF-Full_seed%d.pt" % s),
             tstar=lambda s: t_fup[("E1_CF-Full", s)],
             params=cfg["e1_params"],
             params_eff=cfg["e1_effective_params"],
             src="week2_results/followups/E1_CF-Full_seed{s}.pt"),
        dict(key="cf-nofrag-160ep",
             desc="CF-NoFrag (hid 18, fragmentation disabled, 160 epochs, "
                  "trained at fixed T=8)",
             short="CF-NoFrag\n160 epochs",
             make=lambda: CFNet(hid=18, **MODEL_SPECS["CF-NoFrag"]),
             path=lambda s: FOLLOWUPS / ("E4_CF-NoFrag160_seed%d.pt" % s),
             tstar=lambda s: t_fup[("E4_CF-NoFrag160", s)],
             params=cfg["e4_params"],
             params_eff=cfg["e4_effective_params"],
             src="week2_results/followups/E4_CF-NoFrag160_seed{s}.pt"),
        dict(key="rcnn-varT",
             desc="RecurrentCNN (hid 22, 40 epochs, rollout depth T drawn "
                  "uniformly from 4..16 during training)",
             short="RecurrentCNN\nT sampled 4-16",
             make=rcnn,
             path=lambda s: FOLLOWUPS / ("E3_RecurrentCNN_seed%d.pt" % s),
             tstar=lambda s: t_fup[("E3_RecurrentCNN", s)],
             params=cfg["e3_params"],
             params_eff=cfg["e3_effective_params"],
             src="week2_results/followups/E3_RecurrentCNN_seed{s}.pt"),
        dict(key="rcnn-fixedT8",
             desc="RecurrentCNN (hid 22, 40 epochs, trained at fixed T=8)",
             short="RecurrentCNN\nfixed T=8",
             make=rcnn,
             path=lambda s: WEEK2 / ("RecurrentCNN_seed%d.pt" % s),
             tstar=lambda s: t_w2[("RecurrentCNN", s)],
             params=cfg["e3_params"],
             params_eff=cfg["e3_effective_params"],
             src="week2_results/RecurrentCNN_seed{s}.pt"),
        dict(key="cf-full-40ep",
             desc="CF-Full (hid 16, 40 epochs, trained at fixed T=8)",
             short="CF-Full\n40 epochs",
             make=lambda: CFNet(hid=16, **MODEL_SPECS["CF-Full"]),
             path=lambda s: WEEK2 / ("CF-Full_seed%d.pt" % s),
             tstar=lambda s: t_w2[("CF-Full", s)],
             params=cfg["e1_params"],
             params_eff=cfg["e1_effective_params"],
             src="week2_results/CF-Full_seed{s}.pt"),
    ]


MODELS = model_registry()
MBY = {m["key"]: m for m in MODELS}


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def eval_rows(model, ds, Ts):
    """Per-maze metrics at every T in `Ts`, one shared rollout per batch.

    Mirrors cf.train.sweep_model exactly: logits_at -> sigmoid -> 0.5
    threshold -> cf.metrics.evaluate_mask per maze.
    """
    model.eval().to(DEVICE)
    Ts = sorted(set(Ts))
    probs = {T: [] for T in Ts}
    for i in range(0, len(ds), 500):
        xb = ds.x[i:i + 500].to(DEVICE)
        outs = model.logits_at(xb, Ts)
        for T in Ts:
            probs[T].append(torch.sigmoid(outs[T]).cpu())
    rows = {T: [] for T in Ts}
    for T in Ts:
        p = torch.cat(probs[T]).numpy()
        for k in range(len(ds)):
            rows[T].append(evaluate_mask(
                p[k] >= THRESH, ds.walls[k], ds.starts[k], ds.goals[k],
                ds.y[k].numpy()))
    return rows


@torch.no_grad()
def maze_masks(model, ds, positions, Ts):
    """Predicted boolean masks {T: (n, H, W)} for the given maze positions."""
    model.eval().to(DEVICE)
    xb = ds.x[positions].to(DEVICE)
    outs = model.logits_at(xb, sorted(set(Ts)))
    return {T: (torch.sigmoid(outs[T]).cpu().numpy() >= THRESH) for T in outs}


def collect_records(data):
    """Evaluate every (model, seed) on the test split at its own T*."""
    path_len = {s: np.array([int(v.sum().item()) for v in data[s].y])
                for s in SPLITS}
    EV, models, csv_rows = {}, {}, []
    for m in MODELS:
        for seed in SEEDS:
            model = m["make"]()
            model.load_state_dict(torch.load(str(m["path"](seed)),
                                             map_location="cpu"))
            model.to(DEVICE).eval()
            models[(m["key"], seed)] = model
            Ts = [m["tstar"](seed)]
            for split in SPLITS:
                ds = data[split]
                rows = eval_rows(model, ds, Ts)
                for T in Ts:
                    strict = np.array([r["strict"] for r in rows[T]], bool)
                    exact = np.array([r["exact"] for r in rows[T]], bool)
                    extra = np.array([r["extra_cells"] for r in rows[T]], int)
                    missed = np.array([r["missed_cells"] for r in rows[T]], int)
                    wrong = extra + missed
                    walls = np.array([r["wall_hits"] for r in rows[T]], int)
                    iou = np.array([r["iou"] for r in rows[T]], float)
                    EV[(m["key"], seed, split, T)] = dict(
                        strict=strict, exact=exact, wrong=wrong,
                        extra=extra, missed=missed, wallhits=walls, iou=iou)
                    for k in range(len(ds)):
                        csv_rows.append(dict(
                            model=m["key"], seed=seed, split=split, T=T,
                            maze=k, dataset_row=int(ds.indices[k]),
                            path_len=int(path_len[split][k]),
                            strict=int(strict[k]), exact=int(exact[k]),
                            wrong_cells=int(wrong[k]),
                            extra_cells=int(extra[k]),
                            missed_cells=int(missed[k]),
                            wall_hits=int(walls[k]),
                            iou=round(float(iou[k]), 4)))
    return EV, path_len, models, csv_rows


def write_csv(csv_rows, path):
    import csv as _csv
    cols = ["model", "seed", "split", "T", "maze", "dataset_row", "path_len",
            "strict", "exact", "wrong_cells", "extra_cells", "missed_cells",
            "wall_hits", "iou", "probe"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in csv_rows:
            row = dict(row)
            row.setdefault("probe", 0)          # probe = 1: figure-D step rows
            w.writerow(row)


def load_aggregate(path):
    """{(model, seed, split, T): strict rate} recomputed from data.csv.
    Per-step probe rows of figure D (probe = 1) are excluded: they
    re-evaluate two mazes at other step counts and would shift the rates."""
    import csv as _csv
    acc = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            if row.get("probe") == "1":
                continue
            key = (row["model"], int(row["seed"]), row["split"], int(row["T"]))
            acc.setdefault(key, []).append(int(row["strict"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def seed_stats(agg, mkey, split):
    """Strict rates at each seed's own T*: (mean, sample sd, per-seed list)."""
    rates = [agg[(mkey, s, split, MBY[mkey]["tstar"](s))] for s in SEEDS]
    return (statistics.mean(rates),
            statistics.stdev(rates) if len(rates) > 1 else 0.0, rates)


def acceptance_check(agg):
    """Seed-0 strict rates must match the stored results at the same T."""
    lines = []
    t_fup, t_w2, fup = load_T_stars()
    pairs = [("E1_CF-Full", "cf-full-160ep"),
             ("E3_RecurrentCNN", "rcnn-varT"),
             ("E4_CF-NoFrag160", "cf-nofrag-160ep")]
    for exp, mkey in pairs:
        run = fup["runs"]["%s_seed0" % exp]
        T = int(run["T_star"])
        stored = run["test_strict_by_T"][str(T)]
        mine = agg[(mkey, 0, "test", T)]
        if abs(stored - mine) > TOL:
            sys.exit("ACCEPTANCE FAIL: %s seed 0 test strict at T=%d: "
                     "computed %.4f, stored %.4f" % (mkey, T, mine, stored))
        lines.append("%s seed0 test strict @ T=%d: computed %.4f, "
                      "stored %.4f ok" % (mkey, T, mine, stored))
    sel = json.loads((WEEK2 / "valT_selection.json").read_text(encoding="utf-8"))
    for rec in sel["selections"]:
        mkey = {"RecurrentCNN": "rcnn-fixedT8",
                "CF-Full": "cf-full-40ep"}.get(rec["model"])
        if mkey is None:
            continue
        seed, T_star = int(rec["seed"]), int(rec["T_star"])
        stored, mine = rec["test_strict_at_T_star"], agg[(mkey, seed, "test", T_star)]
        if abs(stored - mine) > TOL:
            sys.exit("ACCEPTANCE FAIL: %s seed %d test strict at T*=%d: "
                      "computed %.4f, stored %.4f"
                      % (mkey, seed, T_star, mine, stored))
        lines.append("%s seed%d test strict @ T*=%d: computed %.4f, "
                      "stored %.4f ok" % (mkey, seed, T_star, mine, stored))
    return lines


# ---------------------------------------------------------------- maze art

def draw_maze(ax, wall, gt, pred=None, start=None, goal=None, lw=0.9,
              ms_start=7, ms_goal=11):
    """One maze panel: walls dark, free cells light, true path outlined.

    pred & gt -> blue (correct); pred & ~gt -> vermillion (false positive);
    gt & ~pred -> purple (missed cell); every true-path cell gets a black
    outline; start = circle, goal = star.
    """
    H, W = wall.shape
    rgb = np.zeros((H, W, 3))
    rgb[wall == 1] = WALL_RGB
    rgb[wall == 0] = FREE_RGB
    if pred is not None:
        p = np.asarray(pred, bool)
        rgb[p & ~gt] = matplotlib.colors.to_rgb(C_EXTRA)
        rgb[p & gt] = matplotlib.colors.to_rgb(C_CORRECT)
        rgb[gt & ~p] = matplotlib.colors.to_rgb(C_MISSED)
    ax.imshow(rgb, interpolation="nearest")
    for r, c in np.argwhere(gt):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=lw))
    if start is not None:
        ax.plot(start[1], start[0], marker="o", ms=ms_start, color="black",
                mfc="none", mew=0.8, linestyle="none")
    if goal is not None:
        ax.plot(goal[1], goal[0], marker="*", ms=ms_goal, color="black",
                linestyle="none")
    ax.set_xlim(-0.6, W - 0.4)
    ax.set_ylim(H - 0.4, -0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)


def gallery_legend():
    return [
        Patch(facecolor=WALL_RGB, label="wall"),
        Patch(facecolor=FREE_RGB, edgecolor="0.6", label="free cell"),
        Patch(facecolor="white", edgecolor="black",
              label="true path (outline)"),
        Line2D([], [], marker="o", color="black", mfc="none", ms=6, lw=0,
               label="start"),
        Line2D([], [], marker="*", color="black", ms=9, lw=0, label="goal"),
        Patch(facecolor=C_CORRECT, label="predicted cell on true path"),
        Patch(facecolor=C_EXTRA, label="false positive (extra cell)"),
        Patch(facecolor=C_MISSED,
              label="missed cell (on true path, not predicted)"),
    ]


def check_overlaps(fig, name):
    """Programmatic readability guard: no text may overlap another text, a
    line, a marker or a histogram bar, and a legend may not sit on top of
    plotted lines/markers (legend internals, which matplotlib lays out
    itself, are excluded)."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    legend_texts = set()
    legends = []
    for lg in fig.findobj(matplotlib.legend.Legend):
        legends.append(lg)
        legend_texts.update(id(o) for o in lg.findobj(Text))
    # tick labels of a switched-off axis are never drawn (axis("off")) -
    # exclude both label1 and label2 of every major/minor tick from the
    # check so hidden default ticks cannot trip it
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


# ---------------------------------------------------------------- figure A

def select_gallery_mazes(EV, path_len):
    keys = [m["key"] for m in MODELS]
    strict = np.stack([EV[(k, 0, "test", MBY[k]["tstar"](0))]["strict"]
                       for k in keys])           # (5, 1000)
    wrong = np.stack([EV[(k, 0, "test", MBY[k]["tstar"](0))]["wrong"]
                      for k in keys])
    picked, cls = [], []
    rng = np.random.default_rng(0)
    for k in sorted(int(x) for x in rng.choice(1000, size=4, replace=False)):
        picked.append(k)
        cls.append("random pick")
    n_solved = strict.sum(axis=0)
    disagree = np.array([k for k in np.where((n_solved > 0)
                                             & (n_solved < len(keys)))[0]
                         if k not in picked], int)
    score = wrong[:, disagree].sum(axis=0)
    for k in disagree[np.lexsort((disagree, -score))[:2]]:
        picked.append(int(k))
        cls.append("models disagree")
    lens = path_len["test"]
    order = np.lexsort((np.arange(len(lens)), -lens))
    n_hard = 0
    for k in order:
        if int(k) in picked:
            continue
        picked.append(int(k))
        cls.append("longest true path")
        n_hard += 1
        if n_hard == 2:
            break
    info = dict(n_disagree=int(((n_solved > 0) & (n_solved < len(keys))).sum()),
                n_all_solved=int((n_solved == len(keys)).sum()),
                n_none_solved=int((n_solved == 0).sum()))
    return picked, cls, info


def figure_gallery(data, EV, models, ids, cls, part, agg):
    n = len(ids)
    fig = plt.figure(figsize=(12.8, 1.6 + 3.0 * n))
    gs = fig.add_gridspec(n + 1, 1 + len(MODELS),
                          height_ratios=[0.55] + [1] * n,
                          hspace=0.62, wspace=0.11)
    axleg = fig.add_subplot(gs[0, :])
    axleg.axis("off")
    axleg.legend(handles=gallery_legend(), loc="center", ncol=4,
                 fontsize=10, frameon=False, handlelength=1.4,
                 columnspacing=1.2)
    solved_counts = 0
    solved_by_model = {}
    for i, (k, c) in enumerate(zip(ids, cls)):
        ds = data["test"]
        gt = ds.y[k].numpy() > 0.5
        wall = ds.walls[k]
        ax = fig.add_subplot(gs[i + 1, 0])
        draw_maze(ax, wall, gt, None, ds.starts[k], ds.goals[k])
        ax.set_title("ground truth", fontsize=10)
        ax.set_xlabel("%d cells" % int(gt.sum()), fontsize=10, labelpad=1)
        ax.set_ylabel("test maze %d\n(%s)" % (k, c), fontsize=10)
        for j, m in enumerate(MODELS):
            T = m["tstar"](0)
            pred = maze_masks(models[(m["key"], 0)], ds, [k], [T])[T][0]
            w = int((pred != gt).sum())
            solved = (w == 0)
            solved_counts += int(solved)
            solved_by_model[m["key"]] = solved_by_model.get(m["key"], 0) + int(solved)
            ax = fig.add_subplot(gs[i + 1, j + 1])
            draw_maze(ax, wall, gt, pred, ds.starts[k], ds.goals[k])
            ax.set_title(m["short"] + "\nT at inference = %d" % T,
                         fontsize=10)
            ax.set_xlabel("solved" if solved else "%d wrong" % w,
                          fontsize=10, labelpad=1)
    save(fig, "path_gallery_%s.png" % part)
    return solved_by_model


# ---------------------------------------------------------------- figure B

def coverage_canvas(ds, order, solved):
    H = ds.walls[order[0]].shape[0]
    ts = H + 2
    canvas = np.ones((25 * ts, COVERAGE_COLS * ts, 3))
    bc_s = matplotlib.colors.to_rgb(C_SOLVED)
    bc_u = matplotlib.colors.to_rgb(C_UNSOLVED)
    for idx, k in enumerate(order):
        wall, gt = ds.walls[k], ds.y[k].numpy() > 0.5
        r, c = divmod(idx, COVERAGE_COLS)
        tile = np.empty((ts, ts, 3))
        bc = bc_s if solved[k] else bc_u
        tile[0, :] = tile[-1, :] = tile[:, 0] = tile[:, -1] = bc
        inner = np.zeros((H, H, 3))
        inner[wall == 1] = WALL_RGB
        inner[(wall == 0) & gt] = PATH_RGB
        inner[(wall == 0) & ~gt] = FREE_RGB
        tile[1:1 + H, 1:1 + H] = inner
        canvas[r * ts:(r + 1) * ts, c * ts:(c + 1) * ts] = tile
    return canvas


def figure_coverage(m, split, ds, solved, order, agg):
    T = m["tstar"](0)
    rate = float(solved.mean())
    mean, sd, _ = seed_stats(agg, m["key"], split)
    canvas = coverage_canvas(ds, order, solved)
    fig = plt.figure(figsize=(13.0, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[18, 1], hspace=0.04)
    ax = fig.add_subplot(gs[0])
    ax.imshow(canvas, interpolation="nearest")
    ax.axis("off")
    ax.grid(False)
    ax.set_title(
        "%s - %s\n"
        % (m["short"].replace("\n", " "), SPLIT_DESC[split])
        + "each maze at its seed-0 validation-selected inference depth T = %d "
          "(tile border colour = outcome)\n" % T
        + "strict (exact) solve rate: %.3f (%d/1000, seed 0); 3 seeds each at "
          "own T: %.3f +- %.3f; tiles sorted by true path length ascending"
        % (rate, int(solved.sum()), mean, sd), fontsize=11)
    axl = fig.add_subplot(gs[1])
    axl.axis("off")
    axl.legend(handles=[
        Patch(facecolor=C_SOLVED, label="solved (strict = exact path)"),
        Patch(facecolor=C_UNSOLVED, label="not solved"),
        Patch(facecolor=PATH_RGB, label="true-path cells (tint)"),
        Patch(facecolor=WALL_RGB, label="wall"),
        Patch(facecolor=FREE_RGB, edgecolor="0.6", label="free cell")],
        loc="center", ncol=5, fontsize=10, frameon=False)
    save(fig, "coverage_%s.png" % m["key"])


# ---------------------------------------------------------------- figure C

def failure_stats(EV, path_len):
    """Numbers shared by figure C and its README section."""
    lens = np.asarray(path_len["test"])
    which = np.digitize(lens, FC_BINS[1:-1])          # bin index 0..5
    assert which.min() == 0 and which.max() == len(FC_BINS) - 2
    labels, nbin = [], []
    for b in range(len(FC_BINS) - 1):
        sel = lens[which == b]
        if len(sel) == 0:
            sys.exit("ACCEPTANCE FAIL: path-length bin %d is empty" % b)
        labels.append("%d" % int(sel.min()) if sel.min() == sel.max()
                      else "%d-%d" % (int(sel.min()), int(sel.max())))
        nbin.append(int(len(sel)))
    hist = {}
    for m in MODELS:
        wl = np.concatenate([
            EV[(m["key"], s, "test", m["tstar"](s))]["wrong"] for s in SEEDS])
        fails = wl[wl > 0]
        hist[m["key"]] = dict(
            wrong=wl, n_eval=int(len(wl)), n_unsolved=int(len(fails)),
            med=float(np.median(fails)) if len(fails) else 0.0,
            mx=int(wl.max()))
    binned = {}
    for m in MODELS:
        per_seed = []
        for s in SEEDS:
            st = EV[(m["key"], s, "test", m["tstar"](s))]["strict"]
            per_seed.append([float(st[which == b].mean())
                             for b in range(len(FC_BINS) - 1)])
        arr = np.stack(per_seed)                      # (seeds, bins)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        binned[m["key"]] = (mean, mean - np.clip(mean - sd, 0.0, 1.0),
                            np.clip(mean + sd, 0.0, 1.0) - mean)
    return dict(labels=labels, nbin=nbin, hist=hist, binned=binned)


def figure_failure_anatomy(FS):
    fig = plt.figure(figsize=(14.5, 10.8))
    outer = fig.add_gridspec(3, 1, height_ratios=[3.5, 3.3, 0.8],
                             hspace=0.90)
    top = outer[0].subgridspec(1, len(MODELS), wspace=0.32)
    mid = fig.add_subplot(outer[1])
    axl = fig.add_subplot(outer[2])
    axl.axis("off")

    allwrong = np.concatenate([FS["hist"][m["key"]]["wrong"] for m in MODELS])
    edges = np.arange(0.5, allwrong.max() + 1.5, 2.0)
    hi = 0.0
    for m in MODELS:
        f = FS["hist"][m["key"]]["wrong"]
        f = f[f > 0]
        hi = max(hi, np.histogram(f, bins=edges)[0].max())
    for j, m in enumerate(MODELS):
        ax = fig.add_subplot(top[j])
        h = FS["hist"][m["key"]]
        f = h["wrong"][h["wrong"] > 0]
        ax.hist(f, bins=edges, bottom=0.8, color=FC_COLORS[m["key"]],
                edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_ylim(0.5, hi * 2.2)
        ax.set_xlim(0, edges[-1])
        ax.set_title(m["short"].replace("\n", " "), fontsize=10, pad=10)
        ax.set_xlabel("wrong cells\nunsolved %d/%d\nmedian %.1f, max %d"
                      % (h["n_unsolved"], h["n_eval"], h["med"], h["mx"]),
                      fontsize=10)
        if j == 0:
            ax.set_ylabel("unsolved mazes (log scale)", fontsize=10)

    xs = np.arange(len(FS["labels"]))
    handles = []
    for m in MODELS:
        mean, dlo, dhi = FS["binned"][m["key"]]
        eb = mid.errorbar(xs, mean, yerr=[dlo, dhi],
                          color=FC_COLORS[m["key"]], marker="o", ms=4.5,
                          capsize=2.5, lw=1.8, elinewidth=1.2,
                          label=m["short"].replace("\n", " "))
        handles.append(eb[0])
    mid.set_xticks(xs)
    mid.set_xticklabels(["%s\nn=%d" % (lab, n)
                         for lab, n in zip(FS["labels"], FS["nbin"])],
                        fontsize=10)
    mid.set_xlabel("true path length (cells)", fontsize=10)
    mid.set_ylabel("strict solve rate (0-1)", fontsize=10)
    mid.set_ylim(-0.04, 1.08)
    mid.set_title("strict solve rate by true path length\n"
                  "binned means over 3 seeds, each at its own T*; error "
                  "bars = sample sd over 3 seeds, clipped to [0, 1]",
                  fontsize=10)
    axl.legend(handles=handles, loc="center", ncol=5, fontsize=10,
               frameon=False, handlelength=1.6, columnspacing=1.4)
    save(fig, "failure_anatomy.png")


# ---------------------------------------------------------------- figure D

def select_thinking_mazes(EV):
    """Two mazes for figure D: the worst failure (most wrong cells) of each
    of the two models at its own T*, seed 0; no repeats, ties by index.

    The shared no-repeat rule can push the second model past its outright
    worst maze when the first model already picked the same maze. The
    returned notes state, per pick, whether it is the model's outright
    worst maze or the worst remaining one after that duplicate was
    skipped, plus the wrong-cell counts and tie size for the caption."""
    picks, notes = [], []
    for key in THINK_MODELS:
        wrong = EV[(key, 0, "test", MBY[key]["tstar"](0))]["wrong"]
        order = np.lexsort((np.arange(len(wrong)), -wrong))
        outright = int(order[0])
        picked = None
        for k in order:
            if wrong[k] == 0:
                break
            if int(k) not in picks:
                picked = int(k)
                break
        picks.append(picked)
        notes.append(dict(maze=picked, wrong=int(wrong[picked]),
                          outright=outright,
                          outright_wrong=int(wrong[outright]),
                          is_outright=picked == outright,
                          ties=int((wrong == wrong[picked]).sum())))
    return picks, notes


def thinking_rows(data, models, mazes):
    """Predictions / per-step metrics for figure D (one shared rollout per
    model). The per-step rows are appended to data.csv so every number in
    figure D is verifiable there."""
    ds = data["test"]
    out, rows = {}, []
    for key in THINK_MODELS:
        masks = maze_masks(models[(key, 0)], ds, mazes, THINK_STEPS)
        for mi, k in enumerate(mazes):
            plen = int((ds.y[k].numpy() > 0.5).sum())
            per = {}
            for T in THINK_STEPS:
                pred = masks[T][mi]
                r = evaluate_mask(pred, ds.walls[k], ds.starts[k],
                                  ds.goals[k], ds.y[k].numpy())
                per[T] = r
                rows.append(dict(
                    model=key, seed=0, split="test", T=T, maze=k,
                    dataset_row=int(ds.indices[k]),
                    path_len=int((ds.y[k].numpy() > 0.5).sum()),
                    strict=int(r["strict"]), exact=int(r["exact"]),
                    wrong_cells=int(r["extra_cells"]) + int(r["missed_cells"]),
                    extra_cells=int(r["extra_cells"]),
                    missed_cells=int(r["missed_cells"]),
                    wall_hits=int(r["wall_hits"]),
                    iou=round(float(r["iou"]), 4), probe=1))
            out[(key, k)] = dict(per=per, path_len=plen,
                                 masks={T: masks[T][mi] for T in THINK_STEPS})
    return out, rows


def think_summary(TD, think_mazes, think_notes):
    """Lead takeaway sentence for figure D, assembled from the recomputed
    per-step probe metrics in TD (the probe = 1 rows of data.csv; no
    hard-coded numbers)."""
    def short(key):
        return THINK_SHORT[key].replace(", ", " (") + ")"

    i8 = THINK_STEPS.index(8)
    bits = []
    for i, k in enumerate(think_mazes):
        parts = []
        for key in THINK_MODELS:
            rec = TD[(key, k)]
            per = rec["per"]
            ws = [int(per[T]["extra_cells"]) + int(per[T]["missed_cells"])
                  for T in THINK_STEPS]
            ms = [int(per[T]["missed_cells"]) for T in THINK_STEPS]
            solved = [T for T, w in zip(THINK_STEPS, ws) if w == 0]
            if solved and ws[-1] == 0 and \
                    solved == THINK_STEPS[len(THINK_STEPS) - len(solved):]:
                parts.append("%s is solved from t = %d on"
                             % (short(key), min(solved)))
            elif key == THINK_MODELS[0] and solved:
                # solved once, then off the true path again: report the
                # longest tail of steps with a constant missed-cell count
                j = len(THINK_STEPS) - 1
                while (j > 0 and THINK_STEPS[j - 1] > min(solved)
                       and ms[j - 1] == ms[-1]):
                    j -= 1
                tspan = ("t = %d-%d" % (THINK_STEPS[j], THINK_STEPS[-1])
                         if j < len(THINK_STEPS) - 1
                         else "t = %d" % THINK_STEPS[-1])
                worst = ("its worst maze"
                         if k == think_notes[0]["maze"]
                         and think_notes[0]["is_outright"]
                         else "this worst-case pick")
                parts.append("%s is solved at t = %d and then loses the "
                             "path (%d of its %d true-path cells missed "
                             "at %s), i.e. a failure of running longer "
                             "on %s"
                             % (short(key), min(solved), ms[-1],
                                rec["path_len"], tspan, worst))
            elif ws[-1] > ws[i8]:
                parts.append("%s degrades after its training depth "
                             "(%d wrong cells at t = 8, %d at t = 32)"
                             % (short(key), ws[i8], ws[-1]))
            else:
                parts.append("%s never solves it within 32 steps (best "
                             "%d wrong cells at t = %d)"
                             % (short(key), min(ws),
                                THINK_STEPS[int(np.argmin(ws))]))
        bits.append("on test maze %d, %s, while %s"
                    % (k, parts[0], parts[1]))
    return ("What the two rows show, from the probe rows of data.csv: "
            + "; ".join(bits)
            + ". Both mazes are hand-selected worst cases under the "
              "selection rule above, not solve rates.")


def figure_thinking_steps(data, TD, mazes):
    ds = data["test"]
    nrow, ncol = 2 * len(mazes), 1 + len(THINK_STEPS)
    fig = plt.figure(figsize=(1.55 * ncol + 2.2, 1.95 * nrow + 2.8))
    gs = fig.add_gridspec(nrow + 1, ncol,
                          height_ratios=[0.55] + [1] * nrow,
                          hspace=0.75, wspace=0.10)
    axleg = fig.add_subplot(gs[0, :])
    axleg.axis("off")
    axleg.legend(handles=gallery_legend(), loc="center", ncol=4,
                 fontsize=10, frameon=False, handlelength=1.4,
                 columnspacing=1.2)
    for mi, k in enumerate(mazes):
        gt = ds.y[k].numpy() > 0.5
        wall = ds.walls[k]
        for ri, key in enumerate(THINK_MODELS):
            row = 2 * mi + ri + 1
            rec = TD[(key, k)]
            ax = fig.add_subplot(gs[row, 0])
            draw_maze(ax, wall, gt, None, ds.starts[k], ds.goals[k])
            ax.set_title("ground truth", fontsize=10)
            ax.set_ylabel("test maze %d (%d cells)\n%s"
                          % (k, int(gt.sum()), THINK_SHORT[key]),
                          fontsize=10)
            ax.set_xlabel("%d cells" % int(gt.sum()), fontsize=10, labelpad=1)
            for ci, T in enumerate(THINK_STEPS):
                ax = fig.add_subplot(gs[row, ci + 1])
                r = rec["per"][T]
                w = int(r["extra_cells"]) + int(r["missed_cells"])
                draw_maze(ax, wall, gt, rec["masks"][T],
                          ds.starts[k], ds.goals[k])
                ax.set_title("t = %d" % T, fontsize=10)
                ax.set_xlabel("solved" if w == 0 else "%d wrong" % w,
                              fontsize=10, labelpad=1)
    save(fig, "thinking_steps.png")


# ---------------------------------------------------------------- README

def write_readme(agg, gal_ids, gal_cls, gal_info, solved_by_model, FS,
                 TD, think_mazes, think_notes):
    L = []
    L.append("# Maze-path predictions: what every model actually outputs\n")
    L.append("All numbers below are recomputed by `scripts/figures/maze_paths.py` "
             "from the stored checkpoints and written to `data.csv` "
             "(one row per model, seed, inference depth T and maze); the "
             "README numbers are read back from that CSV.\n")
    L.append("Models (state dicts loaded with the repo's own factories; "
             "seed 0 shown in all instance-level figures, three seeds "
             "aggregated wherever a mean is shown):\n")
    L.append("| model | params (raw / effective) | training | check-point |")
    L.append("|---|---|---|---|")
    for m in MODELS:
        tr = ("fixed T=8, 160 epochs"
              if m["key"] in ("cf-full-160ep", "cf-nofrag-160ep")
              else "T sampled from 4..16 per batch, 40 epochs"
              if m["key"] == "rcnn-varT"
              else "fixed T=8, 40 epochs")
        pcell = ("%d" % m["params"] if m["params"] == m["params_eff"]
                 else "%d raw / %d effective"
                 % (m["params"], m["params_eff"]))
        L.append("| %s | %s | %s | `%s` |" % (m["desc"], pcell, tr,
                                              m["src"].replace("{s}", "0/1/2")))
    L.append("")
    m_nofrag = MBY["cf-nofrag-160ep"]
    L.append("Params: raw = state-dict element count; effective = "
             "parameters that receive gradients (nonzero-gradient rule, "
             "`week2_results/followups/results.json` `param_counts`, one "
             "real training batch at T = 8). With fragmentation disabled "
             "only the %d effective of the %d raw CF-NoFrag parameters "
             "train, so CF-NoFrag effectively has *fewer* trainable "
             "parameters than CF-Full (%d); for every other model raw = "
             "effective.\n"
             % (m_nofrag["params_eff"], m_nofrag["params"],
                MBY["cf-full-160ep"]["params"]))
    L.append("Scoring: a maze counts as **solved (strict)** if the thresholded "
             "prediction equals the true-path cell mask exactly (on these "
             "perfect mazes strict coincides with exact-match, so a wrong-cell "
             "count of zero is equivalent). Every model is evaluated at its "
             "validation-selected inference depth T\\*; the seed-0 rates at "
             "those depths match the stored selection results "
             "`week2_results/followups/results.json` and "
             "`week2_results/valT_selection.json` to within 0.001 (checked in "
             "the script, which exits non-zero on a mismatch). Inputs: "
             "walkable mask + start + goal one-hot channels; threshold 0.5 "
             "on the sigmoid output.\n")
    L.append("Split: level-5 test split (1000 mazes, 11x11) - the same "
             "pipeline as the repo's `run_week2.load_data`.\n")

    # ---- figure A
    L.append("---\n")
    L.append("## Figure A - path gallery (`path_gallery_1.png`, "
             "`path_gallery_2.png`)\n")
    L.append("**Selection rule for the eight rows** (level-5 test split, "
             "seed-0 models, each at its own T\\*): 4 mazes drawn with numpy "
             "RNG seed 0; 2 mazes where the five models disagree on the solve "
             "verdict (ranked by total wrong cells across models, largest "
             "first); the 2 mazes with the longest true path. Ties by maze "
             "index, no repeats. Chosen mazes: %s.\n"
             % ", ".join("test maze %d (%s)" % (i, c)
                         for i, c in zip(gal_ids, gal_cls)))
    L.append("**Caption.** Rows are test mazes; columns are the ground truth "
             "and every model's thresholded prediction at its seed-0 T\\* "
             "(model name and T at inference stated above each column, "
             "and the header row is repeated on every maze row so each "
             "PNG is readable on its own). Walls are dark, free cells "
             "light, the true path is outlined in black, start = circle, "
             "goal = star.")
    L.append("**How to read.** Blue = predicted cell on the true path "
             "(correct); vermillion = false positive (extra predicted cell); "
             "purple = missed cell (on the true path, not predicted); the "
             "number below each panel is the wrong-cell count, "
             "\"solved\" for zero.\n")
    per_model_gallery = ", ".join(
        "%s: %d/8" % (k, v) for k, v in solved_by_model.items())
    L.append("**Key takeaway.** On the whole test split, all five models "
             "solve the same %d mazes and none solves %d mazes; %d mazes get "
             "mixed verdicts (these two gallery rows show such disagreement "
             "cases). Across the 8 gallery rows each model solves: %s."
             % (gal_info["n_all_solved"], gal_info["n_none_solved"],
                gal_info["n_disagree"], per_model_gallery))
    L.append("**Data source.** `data.csv` in this folder (rows with "
             "split = test, seed = 0, T = the model's T\\*); checkpoints "
             "listed in the table above (`week2_results/followups/*.pt`, "
             "`week2_results/*.pt`).\n")
    L.append("**Caveats.** Single seed (0) per panel; the four random picks "
             "are a tiny sample; disagreement rows are selected to be hard "
             "and are not typical mazes.\n")

    # ---- figure B
    L.append("---\n")
    L.append("## Figure B - whole-test-split coverage "
             "(`coverage_<model>.png`, one PNG per model)\n")
    L.append("**Caption.** For every model, all 1000 test mazes as a 25x40 "
             "grid of tiles, one tile border colour per outcome at the "
             "seed-0 validation-selected T\\*: bluish green = solved, "
             "vermillion = not solved. The maze itself is drawn inside the "
             "tile (walls dark, true path tinted).")
    L.append("**How to read.** Tiles are sorted by true path length "
             "ascending, left to right, top to bottom; the title carries the "
             "seed-0 strict solve rate and the mean +- sample sd over the "
             "three seeds (each seed at its own T\\*, n = 1000 mazes per "
             "seed; sd is the sample sd over 3 seeds).\n")
    L.append("**Key takeaway.**")
    for m in MODELS:
        s0 = agg[(m["key"], 0, "test", m["tstar"](0))]
        mean, sd, _ = seed_stats(agg, m["key"], "test")
        L.append("- %s: strict solve rate %.3f (%d/1000, seed 0 at T\\* = %d);"
                 " %.3f +- %.3f over 3 seeds at their own T\\*"
                 % (m["desc"], s0, int(round(s0 * 1000)), m["tstar"](0),
                    mean, sd))
    L.append("\n**Data source.** `data.csv` in this folder (split = test, "
             "seed 0 for the tiles, all seeds for the mean).\n")
    L.append("**Caveats.** Tile colours are binary (strict pass/fail); a "
             "\"not solved\" tile may be one wrong cell or a completely "
             "wrong path; the 3-seed mean uses each seed's own T\\* (T\\* "
             "differs between seeds for some models).\n")

    # ---- figure C
    L.append("---\n")
    L.append("## Figure C - failure anatomy `failure_anatomy.png`\n")
    L.append("**Caption.** Top row: one histogram per model of the number "
             "of wrong cells per maze, over the unsolved test-maze "
             "evaluations (3 seeds pooled, each seed at its own validation-"
             "selected T\\*, n = %d evaluations per model = 3 seeds x 1000 "
             "mazes). Bottom: strict solve rate by true path length in 6 "
             "fixed bins (mean over the 3 seeds, error bars = sample sd, "
             "each seed at its own T\\*)."
             % FS["hist"]["cf-full-160ep"]["n_eval"])
    L.append("**How to read.** Histogram x = wrong cells (extra + missed, "
             "only unsolved evaluations, solved mazes are NOT in the "
             "histogram - their count is in the panel caption), y = number "
             "of mazes on a log scale, bars binned in steps of 2 wrong "
             "cells; each model keeps its Okabe-Ito colour in both panels. "
             "Bottom panel: x = true path length (bin ranges and n = "
             "mazes per bin are printed under each bin), y = strict solve "
             "rate between 0 and 1; error bars are clipped to [0, 1] "
             "because the rate cannot leave that range.\n")
    L.append("**Key takeaway.** Failure depth differs between models, not "
             "just failure frequency:")
    for m in MODELS:
        b = FS["binned"][m["key"]]
        L.append("- %s: unsolved in %d of %d evaluations; median %.1f "
                 "wrong cells per failed maze (max %d); solve rate %.3f on "
                 "the shortest bin (%s cells) vs %.3f on the longest "
                 "(%s cells)."
                 % (m["desc"], FS["hist"][m["key"]]["n_unsolved"],
                    FS["hist"][m["key"]]["n_eval"], FS["hist"][m["key"]]["med"],
                    FS["hist"][m["key"]]["mx"], b[0][0], FS["labels"][0],
                    b[0][-1], FS["labels"][-1]))
    L.append("")
    L.append("Per-bin strict solve rate (mean over 3 seeds +- sample sd, "
             "each seed at its own T\\*):\n")
    L.append("| true path length (cells) | n | %s |"
             % " | ".join(m["short"].replace("\n", " ") for m in MODELS))
    L.append("|---|---|---|")
    for i, lab in enumerate(FS["labels"]):
        cells = ["%.3f +- %.3f" % (FS["binned"][m["key"]][0][i],
                                   FS["binned"][m["key"]][1][i])
                 for m in MODELS]
        L.append("| %s | %d | %s |" % (lab, FS["nbin"][i],
                                       " | ".join(cells)))
    L.append("")
    L.append("**Data source.** `data.csv` in this folder: `wrong_cells` "
             "rows at each model's T\\* (histograms; 3 seeds pooled) and "
             "`strict` rows at T\\* (binned rates, sd over the 3 seeds). "
             "Checkpoints as in the model table above.\n")
    L.append("**Caveats.** The histogram pools the three seeds, which use "
             "different T\\* for some models; wrong cells = extra + missed, "
             "so a one-cell slip and one-cell duplication both count and "
             "a near-perfect prediction can still fail strict scoring; "
             "sample sd over only 3 seeds is a rough spread estimate; the "
             "longest bin (%s cells) holds only %d of the 1000 mazes.\n"
             % (FS["labels"][-1], FS["nbin"][-1]))

    # ---- figure D
    L.append("---\n")
    L.append("## Figure D - thinking steps `thinking_steps.png`\n")
    pick_bits = []
    for i, nt in enumerate(think_notes):
        key = THINK_MODELS[i]
        tie = ""
        if nt["ties"] > 1:
            tie = ("; %d mazes tie at that wrong-cell count, tie broken "
                   "by maze index" % nt["ties"])
        if nt["is_outright"]:
            pick_bits.append(
                "test maze %d = worst case of %s, %d wrong cells%s"
                % (nt["maze"], MBY[key]["desc"], nt["wrong"], tie))
        else:
            bj = think_mazes.index(nt["outright"])
            pick_bits.append(
                "test maze %d = worst remaining case of %s, %d wrong "
                "cells: its outright worst maze, test maze %d (%d wrong "
                "cells), was already picked for %s, so the shared "
                "no-repeats rule moves to the next-worst maze%s"
                % (nt["maze"], MBY[key]["desc"], nt["wrong"],
                   nt["outright"], nt["outright_wrong"],
                   THINK_SHORT[THINK_MODELS[bj]], tie))
    L.append("**Caption.** Two test mazes, the thresholded prediction "
             "after t = 1, 2, 4, 8, 12, 16, 24, 32 recurrence steps (one "
             "shared rollout per model, read out at each t). Maze rows: "
             "CF-Full (hid 16, 160 epochs, trained at fixed T = 8) above "
             "RecurrentCNN (hid 22, trained at fixed T = 8, 40 epochs); "
             "both seed 0. The left column repeats the ground truth; the "
             "maze selection rule: the test maze with the most wrong "
             "cells for each of the two models at its own T\\* (seed 0, "
             "no repeats, ties by maze index) - i.e. one worst case per "
             "model under that shared rule, and %s."
             % "; ".join(pick_bits))
    L.append("**How to read.** Same colour code as figure A: blue = "
             "predicted cell on the true path, vermillion = extra "
             "predicted cell, purple = missed true-path cell, black "
             "outline = true path, circle = start, star = goal. The "
             "number under each panel is the wrong-cell count at that t "
             "(\"solved\" = 0 wrong cells).\n")
    L.append("**Key takeaway.**\n")
    L.append(think_summary(TD, think_mazes, think_notes) + "\n")
    for k in think_mazes:
        for key in THINK_MODELS:
            per = TD[(key, k)]["per"]
            ws = [int(per[T]["extra_cells"]) + int(per[T]["missed_cells"])
                  for T in THINK_STEPS]
            solved_at = [T for T, w in zip(THINK_STEPS, ws) if w == 0]
            if solved_at:
                if ws[-1] == 0:
                    tail = ("first fully solved at t = %d and stays solved "
                            "through t = 32" % min(solved_at))
                else:
                    tail = ("fully solved only at t = %s, then off the "
                            "true path again (%d wrong cells at t = 32)"
                            % ("/".join(str(t) for t in solved_at), ws[-1]))
            else:
                tail = ("never solves it within 32 steps (best: %d wrong "
                        "cells at t = %d)"
                        % (min(ws), THINK_STEPS[int(np.argmin(ws))]))
            L.append("- Test maze %d, %s: %d wrong cells at t = 1; "
                     "%s (wrong cells at t = 1, 2, 4, 8, 12, 16, 24, 32: %s)."
                     % (k, THINK_SHORT[key] + ", seed 0", ws[0], tail,
                        ", ".join("%d" % w for w in ws)))
    L.append("")
    L.append("**Data source.** `data.csv` in this folder: rows with "
             "model = %s / %s, seed = 0 and T in "
             "1, 2, 4, 8, 12, 16, 24, 32 for the two mazes above"
             " (probe = 1 rows; all other rows have probe = 0, the "
             "T\\*-level evaluation, and are excluded from every rate "
             "recomputed here; checkpoints: "
             "`week2_results/followups/E1_CF-Full_seed0.pt`, "
             "`week2_results/RecurrentCNN_seed0.pt`).\n"
             % (THINK_MODELS[0], THINK_MODELS[1]))
    L.append("**Caveats.** Both models were trained with rollout depth 8, "
             "so predictions at t > 8 are beyond the training rollout "
             "depth (extrapolation for both, in different ways: the "
             "RecurrentCNN never saw a longer loop, CF-Full never saw a "
             "deeper coagulation-fragmentation process); the two mazes "
             "are worst cases selected to be hard (the second under the "
             "shared no-repeats rule), not typical test "
             "mazes; single seed (0); wrong cells = extra + missed.\n")

    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("saved %s" % (OUT / "README.md"))


# ---------------------------------------------------------------- main

def main():
    print("loading data (run_week2.load_data, level-5 test split) ...")
    data = run_week2.load_data({})
    print("evaluating %d models on the test split at their own T* ..."
          % (len(MODELS) * len(SEEDS)))
    EV, path_len, models, csv_rows = collect_records(data)
    # Figure D mazes need EV only; evaluate the step rollouts here so the
    # per-step rows land in data.csv (every figure-D number is in the CSV)
    print("figure D: selecting thinking-step mazes and rolling out ...")
    think_mazes, think_notes = select_thinking_mazes(EV)
    print("thinking-step mazes: %s" % ", ".join(
        "%d (worst case of %s)" % (nt["maze"], MBY[THINK_MODELS[i]]["key"])
        if nt["is_outright"] else
        "%d (worst remaining case of %s after its outright worst maze %d)"
        % (nt["maze"], MBY[THINK_MODELS[i]]["key"], nt["outright"])
        for i, nt in enumerate(think_notes)))
    TD, step_rows = thinking_rows(data, models, think_mazes)
    csv_rows.extend(step_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "data.csv"
    write_csv(csv_rows, csv_path)
    print("wrote %s (%d rows)" % (csv_path, len(csv_rows)))

    agg = load_aggregate(csv_path)
    print("\nacceptance check (seed-0 strict rates vs stored results):")
    for line in acceptance_check(agg):
        print("  " + line)

    # Figure A
    gal_ids, gal_cls, gal_info = select_gallery_mazes(EV, path_len)
    print("gallery mazes: %s" % ", ".join(
        "%d (%s)" % (i, c) for i, c in zip(gal_ids, gal_cls)))
    sb1 = figure_gallery(data, EV, models, gal_ids[:4], gal_cls[:4], "1", agg)
    sb2 = figure_gallery(data, EV, models, gal_ids[4:], gal_cls[4:], "2", agg)
    solved_by_model = {k: sb1.get(k, 0) + sb2.get(k, 0)
                       for k in [m["key"] for m in MODELS]}

    # Figure B
    ds = data["test"]
    order = np.lexsort((np.arange(len(ds)), path_len["test"]))
    for m in MODELS:
        solved = EV[(m["key"], 0, "test", m["tstar"](0))]["strict"]
        figure_coverage(m, "test", ds, solved, order, agg)

    # Figure C
    print("figure C: failure anatomy ...")
    FS = failure_stats(EV, path_len)
    figure_failure_anatomy(FS)

    # Figure D
    print("figure D: drawing thinking steps ...")
    figure_thinking_steps(data, TD, think_mazes)

    write_readme(agg, gal_ids, gal_cls, gal_info, solved_by_model, FS,
                 TD, think_mazes, think_notes)
    print("\nDONE: figures in %s" % OUT)


if __name__ == "__main__":
    main()
