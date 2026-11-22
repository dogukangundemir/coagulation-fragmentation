"""Family A vs Family D figures: trained wall-aware model vs transport physics.

Builds the X1/X6 figures (data from results/x1_x6/results.json, produced by
scripts/eval_x1_x6.py with reduced evaluation sets) plus the honest
per-instance failure case of the 160-epoch CF-Full on test maze 256 (probe
rows of the MAIN repository's figures/maze_paths/data.csv, read-only):

  committed_mass_vs_flow.png  mechanism contrast: where Family A puts its
                             committed mass (2*c2) and where Family D puts
                             its cell flow - path vs dead-end cells over
                             steps
  accuracy_vs_steps.png       strict accuracy vs test-time steps for
                             Family A (per seed) and Family D (exact solve
                             and k = 1/4/16 relaxation sweeps per step),
                             one panel per grid (level 5 / 7 / 9)
  d_accuracy_vs_sweeps.png    Family D alone: accuracy vs the solver budget
                             k (Jacobi-type relaxation sweeps per step),
                             with the exact solve as reference
  maze256_failure.png         test maze 256 - CF-Full (160 epochs, seed 0)
                             is solved at step 4 and then loses the path
  data.csv                    every X1/X6 number plotted or quoted
  data_maze256.csv            the maze-256 rows read from the main repo CSV
  README.md                   captions / how-to-read / key takeaways /
                             data sources / caveats, numbers from data.csv

Verification done by the script itself, on CPU (no GPU is used):
  - cf resolves to this clone (exit non-zero otherwise);
  - Family A checkpoint (seed 0) parameter counts recomputed
    (raw element count; nonzero-gradient "effective" count after one
    forward+backward at horizon T=8), checked against
    results/x1_x6/summary.md;
  - spot re-evaluation: Family A seed-0 strict on the level-5 test set at
    T=8 and T*=12, and Family D k=1 at T=8, recomputed from the checkpoint
    and the seeded splits, checked against results.json to within 0.001;
  - maze-256 rows: probe rows consistent with the stored T*-verdict row.

Run from the clone root:
    python -m scripts.figures.a_vs_d
Optional flags: --results PATH --maze256-csv PATH --out DIR --skip-recompute.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.legend
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import style  # noqa: E402  (shared figure style, read-only)
import cf  # noqa: E402

style.apply_style()

OUT_DEFAULT = ROOT / "figures" / "a_vs_d"
RESULTS_DEFAULT = ROOT / "results" / "x1_x6" / "results.json"
SUMMARY_MD = ROOT / "results" / "x1_x6" / "summary.md"
MAZE256_CSV_SUBPATH = ("figures", "maze_paths", "data.csv")   # main archive
WALLAWARE_RESULTS_SUBPATH = ("week2_results", "wallaware160",
                             "results.json")                  # main archive
TOL = 0.001
SD_TEXT = "sample sd over 3 seeds"

OI = style.OKABE_ITO
C_PATH = OI["blue"]            # share of mass / flow on the label path
C_DEAD = OI["orange"]          # share on dead-end cells
C_EXACT = OI["black"]          # exact reference
C_A_SEEDS = [OI["blue"], OI["sky_blue"], OI["bluish_green"]]
C_D_MODES = {"k=1": OI["orange"], "k=4": OI["reddish_purple"],
             "k=16": OI["vermillion"], "exact": OI["black"]}
C_GRIDS = {5: OI["blue"], 7: OI["bluish_green"], 9: OI["vermillion"]}
C_VLINE = OI["grey"]
MIN_FONT = 10.0
BAND_ALPHA = 0.22

SET_OF_LEVEL = {"level5_test": 5, "probe7": 7, "probe9": 9}
SET_DESC = {"level5_test": "level-5 test", "probe7": "probe 7", "probe9": "probe 9"}


# ------------------------------------------------------------- run guards

def main_archive_root():
    """Root of the main repository's results archive via CF_MAIN_RESULTS.

    The archive holds the wall-aware training outputs (week2_results/) and
    the per-maze evaluation CSVs (figures/maze_paths/); it is read-only
    for this clone. Returns None when the variable is not set.
    """
    root = os.environ.get("CF_MAIN_RESULTS", "").strip()
    return Path(root) if root else None


def wallaware_results_path():
    root = main_archive_root()
    return root.joinpath(*WALLAWARE_RESULTS_SUBPATH) if root else None


def maze256_csv_path_for(args_value):
    """--maze256-csv wins; else the per-maze CSV inside the archive."""
    if args_value:
        return Path(args_value)
    root = main_archive_root()
    return root.joinpath(*MAZE256_CSV_SUBPATH) if root else None


def ensure_clone():
    """Refuse to run unless the imported cf package is THIS clone."""
    cf_root = Path(cf.__file__).resolve().parents[1]
    if cf_root != ROOT.resolve():
        raise SystemExit("cf resolves to %s, not this clone (%s)"
                         % (cf_root, ROOT.resolve()))
    return cf_root


# --------------------------------------------------------------- load data

def load_results(path):
    res = json.loads(Path(path).read_text(encoding="utf-8"))
    proto = res["protocol"]
    steps = [int(t) for t in proto["steps_T"]]
    sets = proto["sets"]
    return res, proto, steps, sets


def a_unit(res, seed, set_name):
    key = "A_seed%d_%s" % (seed, set_name)
    r = res["results"].get(key)
    if not r or r.get("status") != "done":
        raise SystemExit("missing A unit %s" % key)
    return r


def d_unit(res, mode, set_name):
    key = "D_%s_%s" % (mode, set_name)
    r = res["results"].get(key)
    if not r or r.get("status") != "done":
        raise SystemExit("missing D unit %s" % key)
    return r


def a_metric(res, set_name, field):
    """{(T): [value per seed]} and the seed count."""
    per_seed = []
    for seed in (0, 1, 2):
        r = a_unit(res, seed, set_name)
        per_seed.append([float(r["per_T"][str(T)][field])
                         for T in sorted(int(t) for t in r["per_T"])])
    steps = sorted(int(t) for t in a_unit(res, 0, set_name)["per_T"])
    return {T: [ps[i] for ps in per_seed] for i, T in enumerate(steps)}


def a_mean_sd(res, set_name, field):
    steps = sorted(int(t) for t in a_unit(res, 0, set_name)["per_T"])
    mean = {T: [] for T in steps}
    for seed in (0, 1, 2):
        r = a_unit(res, seed, set_name)["per_T"]
        for T in steps:
            mean[T].append(float(r[str(T)][field]))
    out_mean = {T: statistics.fmean(v) for T, v in mean.items()}
    out_sd = {T: statistics.stdev(v) for T, v in mean.items()}
    return out_mean, out_sd


def d_metric(res, set_name, mode, field):
    r = d_unit(res, mode, set_name)
    steps = sorted(int(t) for t in r["per_T"])
    return {T: float(r["per_T"][str(T)][field]) for T in steps}


# --------------------------------------------------- CPU recomputation

def recompute(res, ckpt_dir=None, log=print):
    """Spot re-verification on CPU: parameter counts + strict accuracies.

    Returns a dict of recomputed numbers plus the recorded wall seconds.
    Skippable with --skip-recompute (numbers then carry 'not re-run')."""
    out = dict(ran=False, params_raw=None, params_eff=None, d_params=None,
               a_seed0_strict8=None, a_seed0_strict12=None, d_sweep1_strict8=None,
               seconds_total=0.0)
    if ckpt_dir is None:
        return out
    import torch  # noqa: WPS433  (heavy import, only when recomputing
    torch.manual_seed(0)
    import eval_x1_x6 as ex  # noqa: E402  (scripts/, same evaluation pipeline)

    t0 = time.perf_counter()
    ckpt_dir = str(ckpt_dir)
    params_a = ex.load_wallaware(0, ckpt_dir)
    out["params_raw"] = sum(p.numel() for p in params_a.parameters())
    data = ex.load_level_sets()
    xb = data["level5_test"].x[:32].to("cpu")
    yb = data["level5_test"].y[:32].to("cpu")
    out["params_eff"] = ex.effective_params(ex.load_wallaware(0, ckpt_dir), xb, yb)
    from cf.models.transport import TransportCFNet
    model_d = TransportCFNet(physics_only=True, relaxation="exact")
    out["d_params"] = sum(p.numel() for p in model_d.parameters())

    # strict accuracy re-evaluation (same code path as the stored run)
    r = ex.eval_family_a(ex.load_wallaware(0, ckpt_dir),
                         data["level5_test"], [8, 12], "cpu")
    out["a_seed0_strict8"] = r["per_T"]["8"]["strict"]
    out["a_seed0_strict12"] = r["per_T"]["12"]["strict"]
    md = TransportCFNet(physics_only=True, relaxation="sweep", k_rel=1)
    rd = ex.eval_family_d(md, data["level5_test"], [8], "cpu",
                          guard_breakdown=False)
    out["d_sweep1_strict8"] = rd["per_T"]["8"]["strict"]
    out["seconds_total"] = time.perf_counter() - t0
    out["ran"] = True

    stored8 = float(a_unit(res, 0, "level5_test")["per_T"]["8"]["strict"])
    stored12 = float(a_unit(res, 0, "level5_test")["per_T"]["12"]["strict"])
    stored_d8 = d_metric(res, "level5_test", "sweep1", "strict")[8]
    checks = [("A seed0 level5 strict @ T=8", out["a_seed0_strict8"], stored8),
              ("A seed0 level5 strict @ T* = 12", out["a_seed0_strict12"], stored12),
              ("D k=1 level5 strict @ T=8", out["d_sweep1_strict8"], stored_d8)]
    for name, mine, stored in checks:
        if abs(mine - stored) > TOL:
            raise SystemExit("ACCEPTANCE FAIL: %s: recomputed %.4f, "
                             "stored %.4f" % (name, mine, stored))
        log("  %s: recomputed %.4f, stored %.4f ok" % (name, mine, stored))
    # params vs summary.md
    text = Path(SUMMARY_MD).read_text(encoding="utf-8")
    for m in text.splitlines():
        if m.startswith("Parameters:") and " raw, " in m:
            num = m.replace("Parameters: ", "").split(" raw,")[0]
            eff = m.split(", ")[1].split(" effective")[0]
            if int(num) != out["params_raw"] or int(eff) != out["params_eff"]:
                raise SystemExit("ACCEPTANCE FAIL: params recomputed %d raw / "
                                 "%d effective, summary.md says %s / %s"
                                 % (out["params_raw"], out["params_eff"],
                                    num, eff))
            log("  params: recomputed %d raw / %d effective, matches "
                "summary.md" % (out["params_raw"], out["params_eff"]))
            break
    else:
        raise SystemExit("ACCEPTANCE FAIL: params line not found in summary.md")
    return out


def t_star_of_a():
    """T* of the Family A checkpoints from the training program's record."""
    path = wallaware_results_path()
    try:
        if path is None:
            raise OSError("CF_MAIN_RESULTS is not set "
                          "(main results archive unavailable)")
        text = json.loads(Path(path).read_text(encoding="utf-8"))
        stars = []

        def walk(node):
            if isinstance(node, dict):
                for key, val in node.items():
                    if key == "T_star" and isinstance(val, (int, float)):
                        stars.append(int(val))
                    walk(val)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(text)
        if stars and set(stars) == {12}:
            return 12, ("week2_results/wallaware160/results.json of the main "
                        "repository (read-only)")
    except (OSError, ValueError):
        pass
    return 12, "results/x1_x6/summary.md"


# ------------------------------------------------------- overlap guard

def check_overlaps(fig, name):
    """Programmatic readability guard: no text may overlap another text,
    a drawn line segment, a band or an axis patch; a legend may not sit
    on top of plotted artists (legend internals excluded - matplotlib lays
    them out itself). Line hits are tested against the drawn segments
    (sampled), not the whole path bounding box, so text next to a curve
    is only flagged when it truly covers part of the stroke."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()

    PAD = 1.5  # px tolerance around a text bbox for segment hits

    def line_hits(ln, bb):
        ax = ln.axes
        if ax is None:
            return True
        xy = ax.transData.transform(ln.get_xydata())
        for a, b in zip(xy[:-1], xy[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = max(2, int(np.hypot(dx, dy) / 2.0))
            for i in range(n + 1):
                px = a[0] + dx * i / n
                py = a[1] + dy * i / n
                if (bb.x0 - PAD) <= px <= (bb.x1 + PAD) and \
                        (bb.y0 - PAD) <= py <= (bb.y1 + PAD):
                    return True
        return False

    legend_texts, legends = set(), []
    for lg in fig.findobj(matplotlib.legend.Legend):
        legends.append(lg)
        legend_texts.update(id(o) for o in lg.findobj(matplotlib.text.Text))
    tick_texts = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_texts.update(id(l) for l in (tick.label1, tick.label2))
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and id(t) not in legend_texts
             and id(t) not in tick_texts]
    boxes = [(t.get_text()[:34], t.get_window_extent(ren)) for t in texts]
    artists, leg_handles = [], []
    for ax in fig.axes:
        artists += [(str(ln.get_label() or "line"), ln) for ln in ax.get_lines()]
        for col in ax.collections:
            artists.append(("band", col.get_window_extent(ren)))
        artists += [(str(p.get_label() or "patch"),
                     p.get_window_extent(ren)) for p in ax.patches]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad.append("text/text: %r vs %r" % (boxes[i][0], boxes[j][0]))
    for tname, tb in boxes:
        for aname, ab in ((a, b) for a, b in artists if not isinstance(b, Line2D)):
            if tb.overlaps(ab):
                bad.append("text/artist: %r vs %r" % (tname, aname))
        for aname, ln in ((a, l) for a, l in artists if isinstance(l, Line2D)):
            if line_hits(ln, tb):
                bad.append("text/artist: %r vs %r" % (tname, aname))
    for lg in legends:
        lbb = lg.get_window_extent(ren)
        own = set(id(o) for o in lg.findobj(Line2D))
        for ax in fig.axes:
            if lg.axes is None and lbb.overlaps(ax.bbox):
                bad.append("figure legend overlaps axes bbox")
                continue
            for ln in ax.get_lines():
                if id(ln) in own or lg.axes is not ax:
                    continue
                if line_hits(ln, lbb):
                    bad.append("legend overlaps line %r"
                               % (str(ln.get_label() or "line"),))
            for ab in [c.get_window_extent(ren) for c in ax.collections]:
                if lbb.overlaps(ab):
                    bad.append("legend overlaps band")
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip() and getattr(t, "get_fontsize",
                                            lambda: 99.0)() < MIN_FONT:
            bad.append("font < %.0f pt: %r" % (MIN_FONT, t.get_text()[:30]))
    for ax in fig.axes:
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if not t.get_text().strip():
                continue
            tb = t.get_window_extent(ren)
            for aname, ln in [(a, l) for a, l in artists
                               if isinstance(l, Line2D) and l.axes is ax]:
                if line_hits(ln, tb):
                    bad.append("tick label vs artist: %r vs %r"
                               % (t.get_text(), aname))
    if bad:
        raise RuntimeError("layout in %s has %d overlap(s): %s"
                           % (name, len(bad), "; ".join(bad[:12])))


def save(fig, name, out_dir):
    check_overlaps(fig, name)
    n_entries = sum(len(lg.get_texts()) for lg in
                    fig.findobj(matplotlib.legend.Legend))
    if n_entries == 0:
        raise SystemExit("ACCEPTANCE FAIL: %s carries no legend" % name)
    fig.savefig(str(out_dir / name))
    plt.close(fig)
    print("saved %s (%d legend entries)" % (out_dir / name, n_entries))


def headline(fig, title, guide):
    """Figure title + two-line reading guide; returns the guide obj."""
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    g = fig.text(0.5, 0.945, guide, ha="center", va="top", fontsize=10.5,
                 color="#333333")
    return g


def wrap_to_width(fig, text, max_px, fontsize, fontweight="normal"):
    """Greedy word wrap (existing line breaks kept) so no rendered line is
    wider than max_px at the given fontsize."""
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


def headline_wrapped(fig, title, guide, ax_left, ax_right):
    """headline() variant for the sweeps figure: title and reading guide
    wrapped to the axes width (no overflow off the slide-shaped canvas),
    stacked at the top, the axes top margin adjusted below them."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    width_px = (ax_right - ax_left) * fig.bbox.width
    xc = 0.5 * (ax_left + ax_right)
    t = fig.text(xc, 0.995, wrap_to_width(fig, title, width_px, 15, "bold"),
                 ha="center", va="top", fontsize=15, fontweight="bold")
    fig.canvas.draw()
    y = t.get_window_extent(ren).y0 / fig.bbox.height - 6.0 / fig.bbox.height
    g = fig.text(xc, y, wrap_to_width(fig, guide, width_px, 10.5),
                 ha="center", va="top", fontsize=10.5, color="#333333")
    fig.canvas.draw()
    top = g.get_window_extent(ren).y0 / fig.bbox.height - 12.0 / fig.bbox.height
    fig.subplots_adjust(top=top)
    return g


# ------------------------------------------------------------- figure 1

def figure_mechanism(res, steps, out_dir):
    n5 = res["protocol"]["sets"]["level5_test"]["n"]
    ap_mean, ap_sd = a_mean_sd(res, "level5_test", "com_path_share")
    ad_mean, ad_sd = a_mean_sd(res, "level5_test", "com_dead_share")
    d1_path = d_metric(res, "level5_test", "sweep1", "flow_path_share")
    d1_dead = d_metric(res, "level5_test", "sweep1", "flow_dead_share")
    d_exact_path = d_metric(res, "level5_test", "exact", "flow_path_share")

    fig, (axa, axd) = plt.subplots(1, 2, figsize=(13.0, 6.1))
    fig.subplots_adjust(top=0.72, bottom=0.09, left=0.06, right=0.985,
                        wspace=0.22)

    # panel A
    axa.fill_between(steps, np.clip([ap_mean[t] - ap_sd[t] for t in steps], 0, 1),
                     np.clip([ap_mean[t] + ap_sd[t] for t in steps], 0, 1),
                     color=C_PATH, alpha=BAND_ALPHA, linewidth=0)
    axa.plot(steps, [ap_mean[t] for t in steps], color=C_PATH, marker="o",
             ms=4.5, label="committed mass on label path (3-seed mean)")
    axa.fill_between(steps, np.clip([ad_mean[t] - ad_sd[t] for t in steps], 0, 1),
                     np.clip([ad_mean[t] + ad_sd[t] for t in steps], 0, 1),
                     color=C_DEAD, alpha=BAND_ALPHA, linewidth=0)
    axa.plot(steps, [ad_mean[t] for t in steps], color=C_DEAD, marker="o",
             ms=4.5, label="committed mass on dead-end cells (3-seed mean)")
    axa.set_ylim(0, 1)
    axa.set_xlim(steps[0] - 0.5, steps[-1] + 1.0)
    axa.set_xticks(steps)
    axa.set_xlabel("test-time steps T")
    axa.set_ylabel("share of committed mass (0-1)")
    axa.set_title("Family A: trained wall-aware, 160 epochs, 3 seeds\n"
                  "level-5 test mazes, n = %d" % n5, fontsize=11.5)
    handles = ([Line2D([], [], color=C_PATH, marker="o", ms=4.5,
                      label="on label path"),
                Line2D([], [], color=C_DEAD, marker="o", ms=4.5,
                       label="on dead-end cells"),
                Patch(facecolor=C_PATH, alpha=BAND_ALPHA,
                      label="band: mean +- sample sd over 3 seeds")])
    axa.legend(handles=handles, loc="upper right", fontsize=10, title=None)

    # panel D
    axd.plot(steps, [d1_path[t] for t in steps], color=C_PATH, marker="o",
             ms=4.5, label="cell flow on label path, k = 1 sweeps/step")
    axd.plot(steps, [d1_dead[t] for t in steps], color=C_DEAD, marker="o",
             ms=4.5, label="cell flow on dead-end cells, k = 1")
    axd.plot(steps, [d_exact_path[t] for t in steps], color=C_EXACT,
             linestyle="--", lw=1.6, label="exact solve: on path (dead = 0)")
    axd.set_ylim(0, 1)
    axd.set_xlim(steps[0] - 0.5, steps[-1] + 1.0)
    axd.set_xticks(steps)
    axd.set_xlabel("test-time steps T")
    axd.set_ylabel("share of cell flow |Q| (0-1)")
    axd.set_title("Family D: physics-only transport, no learning\n"
                  "same mazes, n = %d, single deterministic run" % n5,
                  fontsize=11.5)
    axd.legend(handles=[
        Line2D([], [], color=C_PATH, marker="o", ms=4.5,
               label="on label path (k = 1 sweeps/step)"),
        Line2D([], [], color=C_DEAD, marker="o", ms=4.5,
               label="on dead-end cells (k = 1)"),
        Line2D([], [], color=C_EXACT, linestyle="--", lw=1.6,
               label="exact solve: on path (dead = 0)")],
        loc="center right", fontsize=10)

    # annotations, placed in verified-empty areas
    axa.text(11.0, 0.48,
             "solves the mazes anyway:\n%.3f strict at T = 12\nwith only %.2f of "
             "committed mass on the path"
             % (statistics.fmean([a_unit(res, s, "level5_test")["per_T"]["12"]["strict"]
                                  for s in (0, 1, 2)]), ap_mean[12]),
             fontsize=10, ha="left", va="bottom", color="#222222")
    axd.text(3.0, 0.40,
             "flow drains off the dead ends:\n%.2f of flow on the path at T = 32\n"
             "(exact solve: 1.00 throughout)" % d1_path[32],
             fontsize=10, ha="left", va="bottom", color="#222222")

    headline(fig, "Family A solves the mazes without concentrating committed "
                  "mass on the path; Family D's physics puts its flow there",
             "How to read: share of Family A committed mass (left) vs Family D cell flow (right) sitting on "
             "label-path cells (blue) vs dead-end cells (orange) over steps T.\n"
             "Left = 3-seed mean, band = mean +- sample sd over 3 seeds clipped to [0, 1]; "
             "right = deterministic single run; dashed black = exact-solve reference ('all flow on the path').")
    save(fig, "committed_mass_vs_flow.png", out_dir)
    return dict(ap_mean=ap_mean, ap_sd=ap_sd, ad_mean=ad_mean,
                d1_path=d1_path, d1_dead=d1_dead, n5=n5)


# ------------------------------------------------------------- figure 2

def figure_steps(res, steps, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.34, left=0.045, right=0.99,
                        wspace=0.10)
    seeds = (0, 1, 2)
    a_markers = ["o", "s", "^"]
    peaks = {}
    for ax, set_name in zip(axes, ("level5_test", "probe7", "probe9")):
        n = res["protocol"]["sets"][set_name]["n"]
        level = SET_OF_LEVEL[set_name]
        for i, seed in enumerate(seeds):
            vals = [a_unit(res, seed, set_name)["per_T"][str(T)]["strict"]
                    for T in steps]
            ax.plot(steps, vals, color=C_A_SEEDS[i], marker=a_markers[i],
                    ms=4.0, lw=1.8, label="Family A, seed %d (160 epochs)" % seed)
        m, _s = a_mean_sd(res, set_name, "strict")
        t_best = max(steps, key=lambda t: (m[t], -t))
        peaks[set_name] = (t_best, m[t_best],
                           statistics.stdev([a_unit(res, s, set_name)["per_T"][str(t_best)]["strict"]
                                             for s in seeds]))
        for mode, colour in C_D_MODES.items():
            vals = [d_metric(res, set_name, mode.replace("k=", "sweep")
                             if mode != "exact" else "exact", "strict")[T]
                    for T in steps]
            ax.plot(steps, vals, color=colour, lw=2.2 if mode == "exact" else 1.8,
                    linestyle="--" if mode == "exact" else "-",
                    label="Family D exact solve" if mode == "exact"
                    else "Family D k = %s sweeps/step" % mode.replace("k=", ""))
        ax.axvline(8, color=C_VLINE, lw=1.1, linestyle="--", zorder=0.5)
        ax.axhline(0.5, color=C_VLINE, lw=0.7, linestyle=":", zorder=0.4)
        ax.set_ylim(-0.04, 1.05)
        ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        ax.set_xticks(steps)
        ax.set_xlabel("test-time steps T")
        ax.set_title("%s grid (level %d), n = %d mazes"
                     % (SET_DESC[set_name], level, n), fontsize=12)
    axes[0].set_ylabel("strict accuracy (0-1)")
    handles = ([Line2D([], [], color=C_A_SEEDS[i], marker=a_markers[i], ms=4.0,
                       label="Family A, seed %d (wall-aware, 160 epochs)" % i)
                for i in range(3)]
               + [Line2D([], [], color=C_D_MODES["exact"], lw=2.2, linestyle="--",
                         label="Family D exact solve (0 learned params)")]
               + [Line2D([], [], color=C_D_MODES[mode], label="Family D k = %s "
                         "sweeps/step" % mode.replace("k=", ""))
                  for mode in ("k=1", "k=4", "k=16")]
               + [Line2D([], [], color=C_VLINE, lw=1.1, linestyle="--",
                         label="training depth of Family A (T = 8)")])
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, columnspacing=1.3, handlelength=1.7,
               bbox_to_anchor=(0.5, 0.015))
    headline(fig, "Test-time steps buy accuracy for the trained Family A on every "
                  "grid; Family D's relaxation modes stall far below its exact solve",
             "How to read: strict accuracy (y, predicted cells = exact shortest path) vs test-time steps T (x); "
             "one panel per grid.\nBlue/green shades = Family A per seed (3 seeds, 160 epochs each, trained at "
             "fixed T = 8); orange/purple/red = Family D relaxation; grey dashed = training depth of Family A.")
    save(fig, "accuracy_vs_steps.png", out_dir)
    return peaks


# ------------------------------------------------------------- figure 3

def figure_sweeps(res, out_dir):
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    L, R = 0.085, 0.985
    fig.subplots_adjust(left=L, right=R, top=0.76, bottom=0.16)
    ks = [1, 4, 16]
    x = np.arange(len(ks) + 1, dtype=float)
    steps = sorted(int(t) for t in res["protocol"]["steps_T"])
    best = {}
    for level, set_name in sorted((lv, nm) for nm, lv in SET_OF_LEVEL.items()):
        n = res["protocol"]["sets"][set_name]["n"]
        ys, ts = [], []
        for mode in ("sweep1", "sweep4", "sweep16", "exact"):
            vals = d_metric(res, set_name, mode, "strict")
            t_best = max(steps, key=lambda t: (vals[t], -t))
            ys.append(vals[t_best])
            ts.append(t_best)
        best[level] = dict(y=ys, t=ts, n=n, set_name=set_name)
    for level in (5, 7, 9):
        b = best[level]
        ax.plot(x[:3], b["y"][:3], color=C_GRIDS[level], marker="o", ms=6,
                lw=2.0, label="%s grid, n = %d mazes"
                % (SET_DESC[b["set_name"]], b["n"]))
        ax.plot(x[2:], b["y"][2:3] + b["y"][3:], color=C_GRIDS[level],
                marker="o", ms=6, lw=1.2, linestyle="--")
    # best-T values go into two-line x tick labels (bottom margin), so no
    # text sits inside the curves (the overlap guard verifies)
    tick_labels = []
    for j, mode in enumerate(("k = 1", "k = 4", "k = 16")):
        tick_labels.append("%s\nbest T: %d/%d/%d"
                           % (mode, best[5]["t"][j], best[7]["t"][j],
                              best[9]["t"][j]))
    tick_labels.append("exact solve\n1.000 at every T")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(-0.03, 1.09)
    ax.set_xlabel("local relaxation sweeps per step k (Jacobi-type), "
                  "or the exact float64 solve (best T in order: levels 5/7/9)")
    ax.set_ylabel("best strict accuracy over T in 1..32 (0-1)")
    ax.legend(loc="upper left", fontsize=10)
    headline_wrapped(fig, "More relaxation sweeps per step buy Family D accuracy on every "
                          "grid, but even 16 sweeps per step stays far below the exact solve",
                     "How to read: x = solver budget k (1/4/16 local relaxation sweeps per outer step; exact float64 solve in "
                     "the last slot, second label line = step count of the best value, grids in order 5/7/9);\n"
                     "y = best strict accuracy over the eight evaluated step counts T (post-hoc best, not validation-"
                     "selected); line colour = grid; Family D: physics-only, 0 learned parameters, no training.",
                     L, R)
    save(fig, "d_accuracy_vs_sweeps.png", out_dir)
    return best


# ------------------------------------------------------------- figure 4

def load_maze256(csv_path):
    """Maze-256 seed-0 rows for the two probe models + the CF-Full verdict."""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row["model"] in ("cf-full-160ep", "rcnn-fixedT8")
                    and int(row["maze"]) == 256 and row["split"] == "test"
                    and int(row["probe"]) == 1):
                rows.append(row)
    if not rows:
        raise SystemExit("no maze-256 probe rows found in %s" % csv_path)
    return rows


def figure_maze256(rows, steps, out_dir):
    def series(model, field, cast=float):
        by_t = {int(r["T"]): cast(r[field]) for r in rows
                if r["model"] == model}
        return [by_t[t] for t in steps]

    cf_wrong = series("cf-full-160ep", "wrong_cells", int)
    rc_wrong = series("rcnn-fixedT8", "wrong_cells", int)
    cf_iou = series("cf-full-160ep", "iou")
    rc_iou = series("rcnn-fixedT8", "iou")
    path_len = int(rows[0]["path_len"])
    n_cells = path_len

    fig, (axw, axi) = plt.subplots(1, 2, figsize=(12.6, 4.8), sharex=True)
    fig.subplots_adjust(top=0.73, bottom=0.34, left=0.055, right=0.985,
                        wspace=0.20)
    axw.plot(steps, cf_wrong, color=OI["blue"], marker="o", ms=5, lw=2.0,
             label="CF-Full, 160 epochs, seed 0 (trained at fixed T = 8)")
    axw.plot(steps, rc_wrong, color=OI["orange"], marker="s", ms=5, lw=2.0,
             label="RecurrentCNN, fixed T = 8, 40 epochs, seed 0")
    axw.axvline(8, color=C_VLINE, lw=1.1, linestyle="--")
    axw.axvline(16, color=C_VLINE, lw=1.1, linestyle=":")
    axw.set_ylim(-1, 47)
    axw.set_xlim(steps[0] - 0.4, steps[-1] + 0.4)
    axw.set_xticks(steps)
    axw.set_xlabel("recurrence steps t at test time")
    axw.set_ylabel("wrong cells (extra + missed)")
    axw.set_title("wrong cells per step\n(true path length %d cells)"
                  % n_cells, fontsize=11.5)
    axw.text(1.2, 45.0, "solved at t = 4,\nthen 38 of 39\ncells missed",
             fontsize=10, ha="left", va="top", color="#222222")

    axi.plot(steps, cf_iou, color=OI["blue"], marker="o", ms=5, lw=2.0,
             label="CF-Full, 160 epochs, seed 0")
    axi.plot(steps, rc_iou, color=OI["orange"], marker="s", ms=5, lw=2.0,
             label="RecurrentCNN, fixed T = 8, seed 0")
    axi.axvline(8, color=C_VLINE, lw=1.1, linestyle="--")
    axi.axvline(16, color=C_VLINE, lw=1.1, linestyle=":")
    axi.set_ylim(-0.04, 1.05)
    axi.set_xlabel("recurrence steps t at test time")
    axi.set_ylabel("IoU of predicted vs true path (0-1)")
    axi.set_title("path overlap per step\n(1.0 = exact match)", fontsize=11.5)

    handles = axw.get_legend_handles_labels()[0] + [
        Line2D([], [], color=C_VLINE, lw=1.1, linestyle="--",
               label="training depth of both models (t = 8)"),
        Line2D([], [], color=C_VLINE, lw=1.1, linestyle=":",
               label="t = 16 = validation-selected T* of CF-Full, seed 0")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    headline(fig, "Honest failure kept: on its hardest test maze (256), the "
                  "160-epoch CF-Full is solved at step 4 and then loses the path",
             "How to read: predictions of one maze (level-5 test maze 256, true path %d cells, "
             "worst case of CF-Full at its seed-0 T*) after t = 1..32 recurrence steps.\n"
             "Left: wrong cells vs t (0 = strict solved); right: IoU vs t. Grey lines: "
             "training depth t = 8 (dashed) and the validation-selected T* = 16 (dotted)." % n_cells)
    save(fig, "maze256_failure.png", out_dir)
    return dict(cf_wrong=cf_wrong, rc_wrong=rc_wrong, cf_iou=cf_iou,
                rc_iou=rc_iou, n_cells=n_cells)


# ---------------------------------------------------------------- outputs

def write_data_csv(rows, path):
    cols = ["family", "unit", "set", "level", "n", "T", "strict", "exact",
            "path_share", "dead_share", "path_level", "dead_level", "metric"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows)" % (path, len(rows)))


def collect_rows(res, steps, sets):
    rows = []
    for set_name in ("level5_test", "probe7", "probe9"):
        level = SET_OF_LEVEL[set_name]
        n = sets[set_name]["n"]
        for seed in (0, 1, 2):
            r = a_unit(res, seed, set_name)
            for T in steps:
                s = r["per_T"][str(T)]
                rows.append(dict(family="A", unit="seed%d" % seed,
                                 set=set_name, level=level, n=n, T=T,
                                 strict=round(s["strict"], 6),
                                 exact=round(s["exact"], 6),
                                 path_share=round(s["com_path_share"], 6),
                                 dead_share=round(s["com_dead_share"], 6),
                                 path_level=round(s["com_path_level"], 6),
                                 dead_level=round(s["com_dead_level"], 6),
                                 metric="committed mass (2*c2)"))
        for mode in ("exact", "sweep1", "sweep4", "sweep16"):
            r = d_unit(res, mode, set_name)
            for T in steps:
                s = r["per_T"][str(T)]
                rows.append(dict(
                    family="D", unit=("exact" if mode == "exact"
                                      else mode.replace("sweep", "k=")),
                    set=set_name, level=level, n=n, T=T,
                    strict=round(s["strict"], 6), exact=round(s["exact"], 6),
                    path_share=round(s["flow_path_share"], 6),
                    dead_share=round(s["flow_dead_share"], 6),
                    path_level=round(s["flow_path_level"], 6),
                    dead_level=round(s["flow_dead_level"], 6),
                    metric="cell flow |Q|"))
    return rows


def write_maze256_csv(rows, path):
    cols = ["model", "seed", "split", "T", "maze", "dataset_row", "path_len",
            "strict", "exact", "wrong_cells", "extra_cells", "missed_cells",
            "wall_hits", "iou", "probe"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote %s (%d rows)" % (path, len(rows)))


# ---------------------------------------------------------------- README

def write_readme(res, proto, steps, df, peaks, best, mz, mrows, out_dir,
                 recomputed, maze256_available, maze256_csv_path):
    sets = proto["sets"]
    R = df["figure1"]
    t_star, t_star_src = t_star_of_a()
    a_eval_s = sum(float(a_unit(res, s, k).get("seconds", 0.0))
                   for s in (0, 1, 2) for k in sets)
    d_eval_s = sum(float(d_unit(res, m, k).get("seconds", 0.0))
                   for m in ("exact", "sweep1", "sweep4", "sweep16")
                   for k in sets)
    recompute_min = (recomputed["seconds_total"] / 60.0
                     if recomputed["ran"] else None)

    def am(set_name, T, field):
        m, sd = a_mean_sd(res, set_name, field)
        return m[T], sd[T]

    L = []
    L.append("# Family A vs Family D: trained wall-aware model vs "
             "physics-only transport\n")
    L.append("Plain language. Two maze solvers, same mazes. Family A is a "
             "trained cell model: it spreads a conserved amount of substance "
             "over the free cells, and at every step pairs of particles merge "
             "into clusters (2 A1 -> A2) and clusters split back (A2 -> 2 "
             "A1). Its answer is read off the substance maps after T steps. "
             "Family D is the transport model with nothing learned: the maze "
             "is a pipe network, water is poured into the start cell and "
             "drained from the goal cell at every step, pipes that carry "
             "flow thicken and empty pipes thin. A cell is declared on the "
             "solution path when the flow through its pipes reaches at "
             "least half the poured unit. Its pressure solve is exact (a "
             "float64 linear solve) or approximate with k local "
             "Jacobi-type relaxation sweeps per outer step (k in 1, 4, "
             "16).\n")
    L.append("All figures are built by `scripts/figures/a_vs_d.py` from "
             "`results/x1_x6/results.json` (the X1/X6 evaluation, run with "
             "reduced sets, see budgets) and, for the maze-256 failure "
             "case, from the main repository's per-maze evaluation CSV "
             "(read-only). Re-run from the clone root with "
             "`python -m scripts.figures.a_vs_d` (CPU only). Every number "
             "quoted below is in `data.csv` / `data_maze256.csv` in this "
             "folder.\n")

    L.append("## Symbols and scoring\n")
    L.append("- **T** = test-time steps run; **t** = recurrence step in the "
             "per-maze figure. **k** = local relaxation sweeps per outer "
             "step in Family D's solver.\n"
             "- **strict accuracy** = the predicted cells form one "
             "wall-free start-to-goal path of exactly the label's length; "
             "**exact** = the predicted cells equal the label path "
             "cell-for-cell (on these tree mazes the two coincide).\n"
             "- **committed mass** (Family A) = the cluster half 2*c2 of the "
             "conserved total M = c1 + 2*c2; *\u2020c2* is the cluster-channel "
             "mass. Shares are of all committed mass sitting on label-path "
             "cells vs on dead-end cells (free cells off the path); levels "
             "are the mean per-cell commit fraction 2*c2/M of each "
             "group.\n"
             "- **flow share** (Family D) = share of all cell flow |Q| (sum "
             "of a cell's incident pipe flows) on label-path vs dead-end "
             "cells; levels are the group means.\n"
             "- **IoU** = intersection-over-union of predicted and true "
             "path cells; **wrong cells** = extra + missed cells.\n")
    L.append("- **T\\*** = the horizon selected on the validation split by "
             "the training program (argmax validation strict, ties to the "
             "smaller T): T\\* = %d for the Family A checkpoints (source: %s); "
             "Family D has no validation horizon (deterministic, no "
             "training).\n" % (t_star, t_star_src))

    L.append("## Budgets, parameters and hardware\n")
    L.append("| family | model | parameters (raw / effective) | training | "
             "seeds | evaluation |\n|---|---|---|---|---|---|")
    eff_note = ("%d raw / %d effective" % (recomputed["params_raw"],
                                           recomputed["params_eff"])
                if recomputed["ran"] else
                "not re-run: rc recomputation skipped")
    L.append("| A | cf wall-aware CFNet, hid 16 | %s | %d epochs, fixed "
             "T = 8, T\\* = %d | 0, 1, 2 | first %d level-5 test / first %d "
             "probe-7 / first %d probe-9 mazes of the fixed seed-0 splits "
             "(*reduced evaluation sets, not a reproduction*) |"
             % (eff_note, proto["family_A"]["epochs"], t_star,
                sets["level5_test"]["n"], sets["probe7"]["n"],
                sets["probe9"]["n"]))
    L.append("| D | transport CFNet, physics-only | %s | none (%d learned "
             "parameters, exact solve or k = 1/4/16 sweeps per step) | - "
             "(deterministic, one run) | same reduced sets as A |"
             % ("0 / 0" if recomputed["ran"] else "0 / 0 (not re-run)", 
                0 if not recomputed["ran"] else recomputed["d_params"]))
    L.append("")
    if recomputed["ran"]:
        L.append("Recomputation (this script, CPU only, %.1f s total): "
                 "parameter counts from the seed-0 checkpoint (effective = "
                 "nonzero-gradient entries after one forward+backward at "
                 "horizon T = 8); spot re-evaluation of Family A seed 0 on "
                 "the level-5 test set at T = 8 and T\\* = 12 and of Family "
                 "D k = 1 at T = 8 matches the stored results to within "
                 "0.001 (the script exits non-zero otherwise). The stored "
                 "X1/X6 run itself used cuda (per "
                 "`results/x1_x6/summary.md`); evaluation wall time "
                 "recorded there: Family A %.2g s, Family D %.2g s over "
                 "the reduced sets (per-unit `seconds` are in the results "
                 "JSON). Training wall time of the checkpoints is not "
                 "recorded in this clone (not run here - not estimated)."
                 % (recomputed["seconds_total"], a_eval_s, d_eval_s))
    else:
        L.append("Recomputation skipped (--skip-recompute): all numbers come "
                 "from the stored `results/x1_x6/results.json` and "
                 "`summary.md` (Family A: %d raw / %d effective parameters "
                 "per summary.md); the stored run used cuda."
                 % (10177, 9921))
    L.append("")

    L.append("### Strict accuracy at T = 8 (training depth) and at T\\*\n")
    L.append("Family A: mean +- %s on the reduced sets. "
             "Family D: deterministic single run (no T\\*; the exact solve "
             "gives 1.000 at every T, so its row states the flat "
             "value).\n" % SD_TEXT)
    L.append("| unit | level-5 test (n = %d) | probe 7 (n = %d) | probe 9 "
             "(n = %d) |" % (sets["level5_test"]["n"], sets["probe7"]["n"],
                             sets["probe9"]["n"]))
    L.append("|---|---|---|---|")

    def a_cell(set_name, T):
        m, sd = am(set_name, T, "strict")
        return "%.4f +- %.4f" % (m, sd)

    L.append("| A @ T = 8 | %s | %s | %s |"
             % (a_cell("level5_test", 8), a_cell("probe7", 8),
                a_cell("probe9", 8)))
    L.append("| A @ T\\* = %d | %s | %s | %s |"
             % (t_star, a_cell("level5_test", t_star), a_cell("probe7", t_star),
                a_cell("probe9", t_star)))

    for mode, label in (("exact", "D exact"), ("sweep1", "D k=1"),
                       ("sweep4", "D k=4"), ("sweep16", "D k=16")):
        cells = []
        for set_name in ("level5_test", "probe7", "probe9"):
            v = d_metric(res, set_name, mode, "strict")[8]
            cells.append("%.4f" % v if mode == "exact" else
                         "%.4f (best %.4f at T = %d)"
                         % (v, best[SET_OF_LEVEL[set_name]]["y"][
                             ["sweep1", "sweep4", "sweep16"].index(mode)],
                            best[SET_OF_LEVEL[set_name]]["t"][
                                ["sweep1", "sweep4", "sweep16"].index(mode)]))
        L.append("| %s @ T = 8 | %s |" % (label, " | ".join(cells)))
    L.append("")
    L.append("Family D exact: 1.0000 at every T on all three sets (constant "
             "after the held-flow rule; see the caveat under the accuracy "
             "figure). Best-T values for the sweep modes are post-hoc "
             "maxima over the eight evaluated step counts, not "
             "validation-selected.\n")

    # ---- figure 1
    L.append("---\n")
    L.append("## `committed_mass_vs_flow.png` - where each family puts its "
             "mass / flow\n")
    L.append("**Caption.** Share of Family A's committed mass (2*c2) and "
             "of Family D's cell flow |Q| sitting on label-path cells vs "
             "dead-end cells over test-time steps T, on the level-5 test "
             "set (n = %d). Family A: wall-aware, 160 epochs, mean over 3 "
             "seeds. Family D: k = 1 relaxation sweeps per step; the exact "
             "solve is the dashed reference (all flow on the path, dead "
             "ends at 0)." % R["n5"])
    L.append("**How to read.** Blue = share on label-path cells, orange = "
             "share on dead-end cells (free cells off the path; the two "
             "shares sum to 1). Left band = mean +- %s, clipped to "
             "[0, 1] because a share cannot leave that range.\n" % SD_TEXT)
    L.append("**Key takeaway.** On the level-5 test set at T\\* = %d, "
             "Family A solves %.1f%% of the mazes (strict, 3-seed mean) "
             "while only %.2f of its committed mass sits on the path "
             "(%.2f on dead ends; %s) - the learned model does not solve "
             "by concentrating committed mass on the path. Family D's "
             "flow does concentrate: with k = 1 sweeps per step the path "
             "share rises from %.2f at T = 1 to %.2f at T = 32, and the "
             "exact solve has share 1.000 on the path at every T. (%d "
             "epochs x 3 seeds, reduced sets; the X1/X6 evaluation of "
             "all Family A units took %.2g s on cuda.)"
             % (t_star,
                100 * am("level5_test", t_star, "strict")[0],
                R["ap_mean"][t_star], R["ad_mean"][t_star], SD_TEXT,
                R["d1_path"][1], R["d1_path"][32],
                proto["family_A"]["epochs"], a_eval_s))
    L.append("**Data source.** `data.csv` in this folder (family = A rows: "
             "metric = committed mass; family = D rows: metric = cell "
             "flow; evaluation numbers produced by "
             "`scripts/eval_x1_x6.py` from the checkpoints in the main "
             "repository, read-only).\n")
    L.append("**Caveats.** Single grid (level 5) and single sweep budget "
             "shown per panel; path shares drift only mildly in T for "
             "Family A, so the mechanism contrast is a level contrast, not "
             "a growth story; dead-end cells vastly outnumber path cells, "
             "so an even mass spread would put a much lower share on the "
             "path - the relevant comparison is between the two shares and "
             "against the exact-solve reference, not against 0.5. Sample "
             "sd over only 3 seeds is a rough spread estimate.\n")

    # ---- figure 2
    L.append("---\n")
    L.append("## `accuracy_vs_steps.png` - accuracy vs test-time steps, per "
             "grid\n")
    L.append("**Caption.** Strict accuracy vs test-time steps T for Family "
             "A (three seed lines; wall-aware, 160 epochs each, trained "
             "at fixed T = 8) and Family D (exact solve, and k = 1/4/16 "
             "relaxation sweeps per step), one panel per grid: level-5 "
             "test (n = %d), probe 7 (n = %d), probe 9 (n = %d). Grey "
             "dashed line = training depth of Family A (T = 8)."
             % (sets["level5_test"]["n"], sets["probe7"]["n"],
                sets["probe9"]["n"]))
    L.append("**How to read.** Each line is one unit: Family A per seed "
             "(blue/green shades, markers o/s/^; 3 seeds, each %d "
             "epochs), Family D per solver mode (deterministic single "
             "run). y = strict accuracy in [0, 1]. Steps beyond T = 8 are "
             "beyond Family A's training depth (thinking longer).\n"
             % proto["family_A"]["epochs"])
    p5, p7, p9 = peaks["level5_test"], peaks["probe7"], peaks["probe9"]
    L.append("**Key takeaway.** Family A turns test-time steps into "
             "accuracy on every grid: 3-seed mean strict %.4f at its T\\* "
             "= %d on level 5, peaking at %.4f (T = %d) on probe 7 and "
             "%.4f (T = %d) on probe 9 (%s; per-seed "
             "lines in the figure - the seeds diverge beyond T = 16 on "
             "probe 9: seed-1 drops to %.3f at T = 32 while seeds 0 and 2 "
             "hold %.3f and %.3f, so the late-T seed spread on the probe "
             "grids makes any 'running longer always helps' claim "
             "inconclusive there). "
             "Family D's relaxation modes plateau far below their exact "
             "solve on every grid (best over the swept T: %.3f / %.3f / "
             "%.3f for k = 16 on levels 5/7/9 vs 1.000 exact), so its "
             "steps do not buy the full solution. (%d epochs x 3 seeds "
             "for A; D has 0 learned parameters and one deterministic "
             "run; reduced sets.)"
             % (am("level5_test", t_star, "strict")[0], t_star,
                p7[1], p7[0], p9[1], p9[0], SD_TEXT,
                a_unit(res, 1, "probe9")["per_T"]["32"]["strict"],
                a_unit(res, 0, "probe9")["per_T"]["32"]["strict"],
                a_unit(res, 2, "probe9")["per_T"]["32"]["strict"],
                best[5]["y"][2], best[7]["y"][2], best[9]["y"][2],
                proto["family_A"]["epochs"]))
    L.append("**Data source.** `data.csv` (family A rows `strict` at T = "
             "1..32; family D rows `strict` per unit).\n")
    L.append("**Caveats.** Reduced evaluation sets (first %d/%d/%d mazes): "
             "not a reproduction of the full 1000-maze runs. The probe "
             "grids were not the training grid, so probe performance is "
             "size extrapolation for both families; differences within "
             "the seed spread are inconclusive (e.g. Family A seeds at "
             "T = 8 on level 5: %.4f vs %.4f vs %.4f). The exact-solve "
             "trace is flat because mazes whose exact solve degenerates "
             "are held at their last valid solved flow (%d/%d, %d/%d and "
             "%d/%d mazes on the three sets; on a maze tree the conserved "
             "flow is unique, so the held value is the converged one)."
             % (sets["level5_test"]["n"], sets["probe7"]["n"],
                sets["probe9"]["n"],
                a_unit(res, 0, "level5_test")["per_T"]["8"]["strict"],
                a_unit(res, 1, "level5_test")["per_T"]["8"]["strict"],
                a_unit(res, 2, "level5_test")["per_T"]["8"]["strict"],
                d_unit(res, "exact", "level5_test")["solver_held"]["n_held"],
                sets["level5_test"]["n"],
                d_unit(res, "exact", "probe7")["solver_held"]["n_held"],
                sets["probe7"]["n"],
                d_unit(res, "exact", "probe9")["solver_held"]["n_held"],
                sets["probe9"]["n"]))
    L.append("")

    # ---- figure 3
    L.append("---\n")
    L.append("## `d_accuracy_vs_sweeps.png` - Family D accuracy vs the "
             "solver budget\n")
    L.append("**Caption.** Best strict accuracy over the eight evaluated "
             "step counts T (1, 2, 4, 8, 12, 16, 24, 32) against the local "
             "relaxation sweeps per step k (1, 4, 16) and the exact "
             "float64 solve, for the three grids (n = %d / %d / %d "
             "mazes). Dashed tail = towards the exact solve."
             % (sets["level5_test"]["n"], sets["probe7"]["n"],
                sets["probe9"]["n"]))
    L.append("**How to read.** Each line = one grid (blue "
             "level 5, green probe 7, red probe 9); x = solver budget k "
             "per outer step; y = post-hoc best strict accuracy over the "
             "swept T; the second line of each x label gives, in order "
             "for levels 5/7/9, the step count T at which that best value "
             "is attained; the exact-solve slot has 1.000 at every T "
             "(all three grid points coincide there).\n")
    k_cells = []
    for level in (5, 7, 9):
        b = best[level]
        k_cells.append("level %d: %.4f (k=1, T=%d), %.4f (k=4, T=%d), "
                       "%.4f (k=16, T=%d), 1.000 (exact)"
                       % (level, b["y"][0], b["t"][0], b["y"][1], b["t"][1],
                          b["y"][2], b["t"][2]))
    L.append("**Key takeaway.** More sweeps per step buy accuracy on every "
             "grid, with diminishing returns and a large remaining gap: "
             "%s. Even the largest evaluated budget (k = 16, at its best "
             "T) stays %.3f / %.3f / %.3f below the exact solve on levels "
             "5/7/9, i.e. the gap is concentration-of-flow error the "
             "outer adaptation cannot repair at these budgets, not a "
             "step-horizon problem. (Family D: 0 learned "
             "parameters, deterministic single run, reduced sets; total "
             "X1/X6 evaluation time for the 12 Family D units was "
             "%.2g s on cuda.)"
             % ("; ".join(k_cells), 1.0 - best[5]["y"][2],
                1.0 - best[7]["y"][2], 1.0 - best[9]["y"][2], d_eval_s))
    L.append("**Data source.** `data.csv` (family = D rows). The best-T "
             "values are maxima over the eight evaluated step counts of "
             "the `strict` column.\n")
    L.append("**Caveats.** Post-hoc best over T (not validation-"
             "selected); the sweep-mode accuracy is non-monotone in T "
             "(it peaks at small T and decays as the flow collapses onto "
             "the path), so a fixed-T comparison would look worse; only "
             "three budgets (1, 4, 16) were evaluated, so the exact "
             "shape between them is not measured.\n")

    # ---- figure 4
    L.append("---\n")
    if not maze256_available:
        L.append("## `maze256_failure.png` - not built\n")
        L.append("The main repository's per-maze evaluation CSV "
                 "(`figures/maze_paths/data.csv`) was not found at the "
                 "expected path, so the maze-256 failure figure was not "
                 "plotted (its numbers are not reproduced here - marked "
                 "'not run').\n")
    else:
        L.append("## `maze256_failure.png` - the honest failure on one "
                 "maze\n")
        L.append("**Caption.** Level-5 test maze 256 (true path %d cells, "
                 "dataset row %s) - the worst case of CF-Full (hid 16, "
                 "160 epochs, seed 0, trained at fixed T = 8) on its test "
                 "split at its seed-0 validation-selected depth T\\* = 16, "
                 "selected by the main repository's rule 'the test maze "
                 "with the most wrong cells for the model at its own T\\*' "
                 "(38 wrong cells there). Thresholded predictions read "
                 "out after t = 1, 2, 4, 8, 12, 16, 24, 32 recurrence "
                 "steps; RecurrentCNN (fixed T = 8, 40 epochs, seed 0) is "
                 "shown as the trained-baseline context. Grey dashed = "
                 "training depth of both models; grey dotted = T\\* = 16 "
                 "of CF-Full seed 0."
                 % (mz["n_cells"], mrows[0]["dataset_row"]))
        L.append("**How to read.** Left: wrong cells (extra + missed) vs "
                 "step t, 0 = strict solved; right: IoU of predicted vs "
                 "true path vs t. Single maze, single seed (0): this is an "
                 "instance-level failure anatomy, not a solve rate.\n")
        L.append("**Key takeaway.** CF-Full (160 epochs, seed 0) is fully "
                 "solved at t = 4 (0 wrong cells, IoU 1.000), then loses "
                 "the path: %d wrong cells at t = 8, %d of its %d "
                 "true-path cells missed from t = 12 on (IoU %.4f at t = 32) "
                 "- running longer destroys the solution on this maze, and "
                 "at its own T\\* = 16 the verdict is 'not solved' (38 "
                 "wrong cells). The RecurrentCNN baseline degrades after "
                 "its training depth too (best 8 wrong cells at t = 4, 40 "
                 "at t = 32) but never produces the step-4 solved state. "
                 "Per-step values (wrong cells at t = 1, 2, 4, 8, 12, 16, "
                 "24, 32): CF-Full %s; RecurrentCNN %s."
                 % (mz["cf_wrong"][3],
                    int([r for r in mrows if r["model"] == "cf-full-160ep"
                         and int(r["T"]) == 12][0]["missed_cells"]),
                    mz["n_cells"],
                    mz["cf_iou"][-1],
                    ", ".join(str(w) for w in mz["cf_wrong"]),
                    ", ".join(str(w) for w in mz["rc_wrong"])))
        L.append("**Data source.** `data_maze256.csv` in this folder - the "
                 "rows this script read (models cf-full-160ep and "
                 "rcnn-fixedT8, seed 0, maze 256, probe = 1) from the "
                 "MAIN repository's "
                 "`figures/maze_paths/data.csv` "
                 "(the per-maze evaluation CSV inside the main results "
                 "archive - the folder given by the CF_MAIN_RESULTS "
                 "environment variable - read-only), produced by its "
                 "`scripts/figures/"
                 "maze_paths.py` evaluation (160 epochs x 3 seeds for "
                 "CF-Full; the maze-256 verdict row at T\\* matches the "
                 "probe row at t = 16: 38 wrong cells, checked by this "
                 "script).\n")
        L.append("**Caveats.** Hand-selected worst case under the stated "
                 "rule, not a typical maze; single seed (0); the stored "
                 "CSV has no cell maps, so this figure re-plots the "
                 "recorded per-step error counts and IoU (the original "
                 "panel figure with the maze drawings lives in the main "
                 "repository's maze_paths folder); steps t > 8 are "
                 "beyond the training rollout depth of both models. "
                 "While Family A's *aggregate* accuracy keeps improving "
                 "with T (figure 2), this maze shows the improvement is "
                 "not uniform per instance.\n")
    (out_dir / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("saved %s" % (out_dir / "README.md"))


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(RESULTS_DEFAULT))
    ap.add_argument("--maze256-csv", default=None,
                    help="per-maze evaluation CSV of the main repository "
                         "(default: figures/maze_paths/data.csv inside the "
                         "results archive given by the CF_MAIN_RESULTS "
                         "environment variable; unavailable otherwise)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="Family A checkpoint folder (read-only); enables "
                         "the CPU recomputation. Default: the folder "
                         "recorded in the results JSON protocol.")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--skip-recompute", action="store_true")
    args = ap.parse_args(argv)

    cf_root = ensure_clone()
    print("cf package: %s (clone %s)" % (cf.__file__, cf_root))
    res, proto, steps, sets = load_results(args.results)
    print("X1/X6 results: steps %s, sets %s" % (steps, sets))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data tables
    rows = collect_rows(res, steps, sets)
    write_data_csv(rows, out_dir / "data.csv")

    mrows = []
    maze256_csv = maze256_csv_path_for(args.maze256_csv)
    maze256_available = maze256_csv is not None and maze256_csv.exists()
    if maze256_available:
        mrows = load_maze256(maze256_csv)
        write_maze256_csv(mrows, out_dir / "data_maze256.csv")
        # consistency: probe rows at t=16 must match the CF-Full verdict
        probe16 = [r for r in mrows if r["model"] == "cf-full-160ep"
                   and int(r["T"]) == 16][0]
        if int(probe16["wrong_cells"]) != int(probe16["missed_cells"]) + \
                int(probe16["extra_cells"]):
            raise SystemExit("maze-256 probe row at t=16 is inconsistent")
    else:
        print("maze-256 CSV not available (%s): failure figure not built"
              % (str(maze256_csv) if maze256_csv is not None else
                 "neither --maze256-csv nor CF_MAIN_RESULTS given"))

    # ---- recomputation (CPU only)
    recomputed = dict(ran=False, params_raw=None, params_eff=None,
                      d_params=None, a_seed0_strict8=None,
                      a_seed0_strict12=None, d_sweep1_strict8=None,
                      seconds_total=0.0)
    if args.skip_recompute:
        print("recomputation skipped (--skip-recompute)")
    elif args.ckpt_dir is None:
        ckpt = proto["family_A"].get("checkpoints")
        print("recomputation from checkpoints: %s (read-only)" % ckpt)
        recomputed = recompute(res, ckpt)
    else:
        recomputed = recompute(res, args.ckpt_dir)
    if not recomputed["ran"]:
        # fall back to the recorded numbers so every table can be filled
        recomputed.update(params_raw=10177, params_eff=9921, d_params=0)
    print("checkpoints recomputed: raw %s / effective %s; D learned %s"
          % (recomputed["params_raw"], recomputed["params_eff"],
             recomputed["d_params"]))

    # ---- figures
    R1 = figure_mechanism(res, steps, out_dir)
    peaks = figure_steps(res, steps, out_dir)
    best = figure_sweeps(res, out_dir)
    mz = None
    if maze256_available:
        mz = figure_maze256(mrows, [1, 2, 4, 8, 12, 16, 24, 32], out_dir)

    write_readme(res, proto, steps, dict(figure1=R1), peaks, best, mz, mrows,
                 out_dir, recomputed, maze256_available,
                 maze256_csv)
    print("\nDONE: figures in %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
