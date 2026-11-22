"""Figures for the curriculum benchmark: budget x fragmentation, depth
extrapolation, settling.

Reads (paths derived from this file's location, or --root override):
  week2_results/followups/results.json   E1-E4 follow-up runs (3 seeds each)
  week2_results/valT_selection.json      40-epoch T* selection (hid=16 models)
  week2_results/week2_metrics.json      40-epoch metrics, T sweeps, settling

Writes figures/curriculum_results/{budget_vs_fragmentation.png,
depth_extrapolation.png, settling.png, data.csv}.

Run:  python scripts/figures/curriculum_figures.py [--root REPO_ROOT]

On-figure text uses plain model descriptions only (the README carries the
E1-E4 run codes as cross-reference). The settling figure aggregates per-step
with the geometric mean (log axis) and shows the spread as thin per-seed
lines instead of an sd band.

Layout: footnotes are placed as figure text strictly below everything drawn in
the axes (tick labels, axis labels, legend included), the position measured
from the drawn figure so the note can never land on the legend; every figure
then asserts at render time that no two text bounding boxes overlap and that
no text box sits on top of drawn data (lines, markers, error bars, bands).
"""

import argparse
import csv
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from style import DPI, OKABE_ITO, SD_CAPTION, SEED_MARKER, apply_style

SWEEP_T = [4, 8, 12, 16, 24, 32]
SEEDS = [0, 1, 2]
N_SEEDS = len(SEEDS)

F1 = "F1 budget x fragmentation"
F2 = "F2 depth extrapolation"
F3 = "F3 settling"

# Okabe-Ito colour assignments, shared across figures for the same model family.
C_CF_FULL = OKABE_ITO["blue"]          # CF-Full (fragmentation on)
C_CF_NOFRAG = OKABE_ITO["vermillion"]  # CF-NoFrag (fragmentation ablated)
C_RCNN_VAR = OKABE_ITO["bluish_green"]  # RecurrentCNN trained with variable T
C_RCNN_FIXED = OKABE_ITO["reddish_purple"]  # RecurrentCNN trained at fixed T=8
C_BAND = OKABE_ITO["grey"]


def repo_root(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    default = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", type=Path, default=default)
    args, _ = p.parse_known_args(argv)
    return args.root


def load(root):
    fu = json.loads((root / "week2_results" / "followups" / "results.json").read_text())
    vt = json.loads((root / "week2_results" / "valT_selection.json").read_text())
    wm = json.loads((root / "week2_results" / "week2_metrics.json").read_text())
    return fu, vt, wm


def fu_by_exp(fu, exp):
    """Per-seed records of one follow-up experiment, ordered by seed."""
    return [fu["runs"][f"{exp}_seed{s}"] for s in SEEDS]


def per_seed_test_by_T(records, T):
    return [r["test_strict_by_T"][str(T)] for r in records]


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def geomean(values):
    """Geometric mean. All settling values are strictly positive, so this is
    well defined -- and it is the right aggregate for a log axis, where an
    arithmetic mean +- sd band is misleading whenever sd approaches the mean."""
    return math.prod(values) ** (1.0 / len(values))


def fmt3(x):
    return f"{x:.3f}"


# ----------------------------------------------------------------------------
# Layout helpers: measured footnote placement + text-overlap assertion
# ----------------------------------------------------------------------------
def _renderer_after_draw(fig):
    """Force a draw and return the renderer: window extents are final."""
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _text_artists(fig):
    """Every visible text artist: tick labels, axis labels, title, legend
    entries, in-axes annotations and figure-level texts (footnotes)."""
    arts = list(fig.texts)
    for ax in fig.axes:
        arts += [ax.title, ax.xaxis.label, ax.yaxis.label]
        arts += ax.get_xticklabels() + ax.get_yticklabels()
        arts += ax.texts
        leg = ax.get_legend()
        if leg is not None:
            arts += list(leg.get_texts())
    return [t for t in arts if t is not None and t.get_text().strip()]


def assert_no_text_overlaps(fig, name):
    """Renderer-level check: no two text bounding boxes may overlap, and no
    text bounding box may sit on top of drawn data (lines, markers, error
    bar paths, sd/reference bands) -- with 1 px of slack on top of the
    true box size plus the on-screen marker radius."""
    ren = _renderer_after_draw(fig)
    labelled = [(t.get_text().split("\n")[0][:24], t.get_window_extent(ren))
                for t in _text_artists(fig)]
    for (txt1, bb1), (txt2, bb2) in combinations(labelled, 2):
        if bb1.overlaps(bb2):
            raise AssertionError(
                f"{name}: text bounding boxes overlap: {txt1!r} vs {txt2!r} "
                f"({bb1.bounds} vs {bb2.bounds})")
    for x, y, r in _data_samples(fig):
        for txt, bb in labelled:
            if (bb.x0 - r - 1 <= x <= bb.x1 + r + 1
                    and bb.y0 - r - 1 <= y <= bb.y1 + r + 1):
                raise AssertionError(
                    f"{name}: text {txt!r} (bbox {bb.bounds}) overlaps "
                    f"drawn data at display point ({x:.1f}, {y:.1f})")


def _densify(pts_px, step_px=4.0):
    """Display-space sample points along a polyline, a few px apart and
    including the endpoints, so a box lying anywhere ON a segment is
    caught, not only at data points."""
    pts = [(float(p[0]), float(p[1])) for p in pts_px]
    if len(pts) == 1:
        return pts
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(d / step_px))
        for i in range(n + 1):
            t = i / n
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def _data_samples(fig):
    """(x, y, radius_px) display-space samples of every drawn data artist.

    Covers lines with their markers (data points get their on-screen marker
    radius), errorbar caps (Line2D) and errorbar bars (LineCollection), sd
    bands (fill_between PolyCollections) and reference regions (Rectangle
    patches from axhspan/axvspan). Each artist is sampled through its own
    transform, so blended transforms (axvline/axhline) come out right.
    """
    px_per_pt = fig.dpi / 72.0
    samples = []
    for ax in fig.axes:
        for line in ax.lines:
            data = line.get_xydata()
            if len(data) == 0:
                continue
            marker = line.get_marker()
            r = 0.0
            if marker not in ("", "None", None):
                r = 0.5 * line.get_markersize() * px_per_pt
            pts = _densify(line.get_transform().transform(data))
            if marker == "|":  # errorbar cap: thin vertical tick, not a disc
                for (x, y) in pts:
                    samples.append((x, y - r, 1.0))
                    samples.append((x, y + r, 1.0))
                    samples.append((x, y, 1.0))
                continue
            if marker == "_":  # errorbar cap: thin horizontal tick
                for (x, y) in pts:
                    samples.append((x - r, y, 1.0))
                    samples.append((x + r, y, 1.0))
                    samples.append((x, y, 1.0))
                continue
            for p in pts:
                samples.append((p[0], p[1], r))
        for coll in ax.collections:
            tr = coll.get_transform()
            for path in coll.get_paths():
                for poly in path.to_polygons():
                    for p in _densify(tr.transform(poly), step_px=5.0):
                        samples.append((p[0], p[1], 0.0))
        for patch in ax.patches:
            tr = patch.get_transform()
            for poly in patch.get_path().to_polygons():
                for p in _densify(tr.transform(poly), step_px=5.0):
                    samples.append((p[0], p[1], 0.0))
    return samples


def place_footnote(fig, ax, text, fontsize=10, pad_px=10):
    """Footnote as figure text strictly below all axes content.

    The position is measured from the drawn figure: the axes tight bbox
    (which includes tick labels, axis labels and the legend below the axes)
    gives the lowest content edge, and the note is placed a few px below it.
    savefig(bbox_inches='tight') keeps the note in frame without stretching
    the canvas to fit it.
    """
    ren = _renderer_after_draw(fig)
    bb = ax.get_tightbbox(ren)
    leg = ax.get_legend()
    if leg is not None:
        bb = Bbox.union([bb, leg.get_window_extent(ren)])
    x_frac, y_frac = fig.transFigure.inverted().transform((bb.x0, bb.y0))
    pad = pad_px / (fig.get_size_inches()[1] * fig.dpi)
    fig.text(x_frac, y_frac - pad, text, fontsize=fontsize,
             color="#444444", ha="left", va="top")


# ----------------------------------------------------------------------------
# F1: training budget x fragmentation (test strict at validation-selected T*)
# ----------------------------------------------------------------------------
def fig1(fu, vt, out):
    cf_full_40 = [s["test_strict_at_T_star"]
                  for s in vt["selections"] if s["model"] == "CF-Full"]
    e1 = fu_by_exp(fu, "E1_CF-Full")
    cf_full_160 = [r["test_strict_at_T_star"] for r in e1]
    e2 = fu_by_exp(fu, "E2_CF-NoFrag")
    nofrag_40 = [r["test_strict_at_T_star"] for r in e2]
    e4 = fu_by_exp(fu, "E4_CF-NoFrag160")
    nofrag_160 = [r["test_strict_at_T_star"] for r in e4]
    e3 = fu_by_exp(fu, "E3_RecurrentCNN")
    rcnn = [r["test_strict_at_T_star"] for r in e3]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    xs = [40, 160]
    jitter = [-3.5, 0.0, 3.5]

    for values, color, label in [
        (cf_full_40 + cf_full_160, C_CF_FULL, "CF-Full hid=16 (10,177 eff. params)"),
        (nofrag_40 + nofrag_160, C_CF_NOFRAG, "CF-NoFrag hid=18 (9,811 eff. params)"),
    ]:
        for x, half in [(40, values[:3]), (160, values[3:])]:
            m, sd = mean_sd(half)
            ax.errorbar(x, m, yerr=sd, color=color, capsize=6,
                        marker=SEED_MARKER, markersize=9, zorder=5)
            for s, (dx, v) in enumerate(zip(jitter, half)):
                ax.plot(x + dx, v, marker=SEED_MARKER, color=color, alpha=0.55,
                        markersize=5, zorder=6, linestyle="none")
        ax.plot(xs, [mean_sd(values[:3])[0], mean_sd(values[3:])[0]],
                color=color, label=label, zorder=4)

    rcnn_m, rcnn_sd = mean_sd(rcnn)
    ax.axhspan(rcnn_m - rcnn_sd, rcnn_m + rcnn_sd, color=C_RCNN_VAR, alpha=0.15)
    ax.axhline(rcnn_m, color=C_RCNN_VAR, linestyle="--",
               label=f"RecurrentCNN hid=22, variable-T training, 40 epochs: "
                     f"{fmt3(rcnn_m)}\u00b1{fmt3(rcnn_sd)}, n={N_SEEDS}")

    # Value labels: anchored a few px outside the error-bar tips (above for
    # CF-Full, below for CF-NoFrag) so they cannot sit on a line, band or
    # seed dot; assert_no_text_overlaps verifies this at render time.
    for x, cf, nf in [(40, cf_full_40, nofrag_40), (160, cf_full_160, nofrag_160)]:
        m_cf, sd_cf = mean_sd(cf)
        m_nf, sd_nf = mean_sd(nf)
        ax.annotate(f"{fmt3(m_cf)}\u00b1{fmt3(sd_cf)}", (x, m_cf + sd_cf),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    va="bottom", color=C_CF_FULL, fontsize=11)
        ax.annotate(f"{fmt3(m_nf)}\u00b1{fmt3(sd_nf)}", (x, m_nf - sd_nf),
                    textcoords="offset points", xytext=(0, -9), ha="center",
                    va="top", color=C_CF_NOFRAG, fontsize=11)

    ax.set_xticks(xs)
    ax.set_xlim(22, 178)
    ax.set_ylim(0, 1.07)
    ax.set_xlabel("Training epochs (budget)")
    ax.set_ylabel("Test strict accuracy at validation-selected T*")
    ax.set_title("Longer training, not fragmentation, drives accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=10)
    fig.tight_layout()
    place_footnote(fig, ax, f"n = {N_SEEDS} seeds; {SD_CAPTION}.\n"
                   "The dashed RecurrentCNN reference spans the whole x-range: "
                   "it has no budget setting.\n"
                   "T* = argmax val strict (ties: smaller T) over T in {4, 8, 12, 16, 24, 32}\n"
                   "for the follow-up runs; the 40-epoch CF-Full point uses\n"
                   "T in {4, 8, 12, 16, 24}.")
    assert_no_text_overlaps(fig, F1)
    fig.savefig(out / "budget_vs_fragmentation.png", dpi=DPI)
    plt.close(fig)

    rows = []
    for name, ep40, ep160 in [("CF-Full hid16", cf_full_40, cf_full_160),
                              ("CF-NoFrag hid18", nofrag_40, nofrag_160)]:
        for ep, vals in [(40, ep40), (160, ep160)]:
            m, sd = mean_sd(vals)
            rows.append(dict(fig=F1, series=name, x=f"epochs={ep}", mean=round(m, 4),
                             sd=round(sd, 4), **{f"seed{s}": vals[i] for i, s in enumerate(SEEDS)}))
    m, sd = mean_sd(rcnn)
    rows.append(dict(fig=F1, series="RecurrentCNN hid22 variable-T", x="epochs=40",
                     mean=round(m, 4), sd=round(sd, 4),
                     **{f"seed{s}": rcnn[i] for i, s in enumerate(SEEDS)}))
    return rows, dict(cf_full_40=cf_full_40, cf_full_160=cf_full_160,
                      nofrag_40=nofrag_40, nofrag_160=nofrag_160, rcnn=rcnn)


# ----------------------------------------------------------------------------
# F2: depth extrapolation (test strict vs test-time T)
# ----------------------------------------------------------------------------
def fig2(fu, wm, out):
    e1 = fu_by_exp(fu, "E1_CF-Full")
    e4 = fu_by_exp(fu, "E4_CF-NoFrag160")
    e3 = fu_by_exp(fu, "E3_RecurrentCNN")

    def fixedT_test_by_T(model, T):
        return [wm["runs"][f"{model}_seed{s}"]["sweeps"]["test"][str(T)]["strict"]
                for s in SEEDS]

    series = [
        ("CF-Full, 160 epochs", C_CF_FULL, "-",
         {T: per_seed_test_by_T(e1, T) for T in SWEEP_T}),
        ("CF-NoFrag, 160 epochs", C_CF_NOFRAG, "-",
         {T: per_seed_test_by_T(e4, T) for T in SWEEP_T}),
        ("RecurrentCNN, variable T, 40 epochs", C_RCNN_VAR, "-",
         {T: per_seed_test_by_T(e3, T) for T in SWEEP_T}),
        ("RecurrentCNN, trained at fixed T=8, 40 epochs", C_RCNN_FIXED, "--",
         {T: fixedT_test_by_T("RecurrentCNN", T) for T in SWEEP_T}),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.axvspan(4, 16, color=C_BAND, alpha=0.15,
               label="RecurrentCNN variable-T only: training range (T ~ U{4..16})")
    ax.axvline(8, color=C_BAND, linestyle=":", linewidth=1.5,
               label="training depth T=8: CF-Full, CF-NoFrag, RecurrentCNN fixed-T")

    for label, color, ls, by_T in series:
        means = [mean_sd(by_T[T])[0] for T in SWEEP_T]
        sds = [mean_sd(by_T[T])[1] for T in SWEEP_T]
        ax.plot(SWEEP_T, means, ls, color=color, marker=SEED_MARKER,
                markersize=7, label=label)
        # Strict accuracy lives in [0, 1]; sd bands are clipped to that
        # range (stated in the footnote and README).
        ax.fill_between(SWEEP_T, [max(0.0, m - s) for m, s in zip(means, sds)],
                        [min(1.0, m + s) for m, s in zip(means, sds)],
                        color=color, alpha=0.15)
        for T in SWEEP_T:
            for v in by_T[T]:
                ax.plot(T, v, marker=SEED_MARKER, color=color, alpha=0.55,
                        markersize=4, zorder=6, linestyle="none")

    # Boxed annotation in the empty strip between T=16 and T=32, below every
    # band (the only nearby data is the flat fixed-T RecurrentCNN line at 0,
    # underneath it); the white box keeps grid lines from bleeding through, and
    # assert_no_text_overlaps verifies at render time that the box overlaps no
    # line, marker or band.
    ax.annotate("T=32 (beyond training range):\n"
                f"CF-Full {fmt3(mean_sd(series[0][3][32])[0])}\u00b1{fmt3(mean_sd(series[0][3][32])[1])}, "
                f"CF-NoFrag {fmt3(mean_sd(series[1][3][32])[0])}\u00b1{fmt3(mean_sd(series[1][3][32])[1])}\n"
                f"vs RecurrentCNN variable-T {fmt3(mean_sd(series[2][3][32])[0])}\u00b1{fmt3(mean_sd(series[2][3][32])[1])},\n"
                f"RecurrentCNN fixed T=8 {fmt3(mean_sd(series[3][3][32])[0])}\u00b1{fmt3(mean_sd(series[3][3][32])[1])}",
                xy=(17.5, 0.29), ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#bbbbbb", linewidth=0.8))
    ax.set_xticks(SWEEP_T)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Test-time unrolling depth T (steps)")
    ax.set_ylabel("Test strict accuracy")
    ax.set_title("Depth extrapolation: CF models hold up past the training depth")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9.5)
    fig.tight_layout()
    place_footnote(fig, ax, f"n = {N_SEEDS} seeds; {SD_CAPTION}; "
                   "sd bands clipped to [0, 1].")
    assert_no_text_overlaps(fig, F2)
    fig.savefig(out / "depth_extrapolation.png", dpi=DPI)
    plt.close(fig)

    rows = []
    for label, _, _, by_T in series:
        for T in SWEEP_T:
            m, sd = mean_sd(by_T[T])
            rows.append(dict(fig=F2, series=label, x=f"T={T}", mean=round(m, 4),
                             sd=round(sd, 4),
                             **{f"seed{s}": by_T[T][i] for i, s in enumerate(SEEDS)}))
    return rows


# ----------------------------------------------------------------------------
# F3: CF settling (residual flux vs T step)
# ----------------------------------------------------------------------------
def fig3(wm, out):
    # (data.csv series name, legend label, colour, linestyle). The two plotted
    # measures are different quantities (a chemical flux for the CF models, a
    # relative state change for RecurrentCNN), so the y-axis carries a neutral
    # short label and the legend names the quantity each curve shows; the
    # shared training settings (40 epochs, T_train=8) are in the footnote.
    models = [
        ("CF-Full hid=16 (40 ep, T_train=8)", "CF-Full: residual flux",
         C_CF_FULL, "-"),
        ("CF-NoFrag hid=16 (40 ep, T_train=8)", "CF-NoFrag: residual flux",
         C_CF_NOFRAG, "-"),
        ("RecurrentCNN hid=22 (40 ep, T_train=8)",
         "RecurrentCNN: relative state change", C_RCNN_FIXED, "--"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rows = []
    stats = []
    all_values = []
    for series, label, color, ls in models:
        model = series.split(" hid=")[0]
        curves = [wm["runs"][f"{model}_seed{s}"]["settling"] for s in SEEDS]
        n_steps = len(curves[0])
        steps = list(range(1, n_steps + 1))
        # Log axis: aggregate with the geometric mean per step (all values are
        # strictly positive) and show the spread as thin per-seed lines. An
        # arithmetic mean +- sd band would be misleading here: at late steps
        # the CF-Full sd exceeds its mean, collapsing the band to the floor.
        gmeans = [geomean([c[i] for c in curves]) for i in range(n_steps)]
        for i in range(n_steps):
            rows.append(dict(fig=F3, series=series, x=f"step={i + 1}",
                             mean=gmeans[i], sd="",
                             **{f"seed{s}": curves[s_][i]
                                for s_, s in enumerate(SEEDS)}))
        stats.append((steps, curves, label, color, ls, gmeans))
        all_values += [v for c in curves for v in c]

    # Axis floor = decade of the smallest plotted value; nothing is drawn
    # below the data, so nothing needs clipping.
    floor = 10.0 ** math.floor(math.log10(min(all_values)))
    for steps, curves, label, color, ls, gmeans in stats:
        for c in curves:
            ax.plot(steps, c, color=color, alpha=0.45, linewidth=1.0)
        ax.plot(steps, gmeans, ls, color=color, marker=SEED_MARKER,
                markersize=4, label=label)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_xlabel("Unrolling step t")
    ax.set_ylabel("Settling measure (log)")
    ax.set_title("Settling: CF residual flux decays with depth; the recurrent baseline does not")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9.5)
    fig.tight_layout()
    place_footnote(fig, ax, f"n = {N_SEEDS} seeds; main lines = geometric mean over 3 seeds, "
                   "thin lines = individual seeds.\n"
                   "CF curves: residual coagulation-fragmentation flux; RecurrentCNN curve: "
                   "relative state change ||h_t - h_{t-1}|| / ||h_{t-1}||.\n"
                   "Different quantities: compare trends (decay vs no decay), not absolute "
                   "values across families.\n"
                   "First 64 test mazes, trained weights (40-epoch checkpoints, "
                   "hid=16 CF / hid=22 RecurrentCNN, T_train=8).")
    assert_no_text_overlaps(fig, F3)
    fig.savefig(out / "settling.png", dpi=DPI)
    plt.close(fig)
    return rows


def main():
    apply_style()
    root = repo_root()
    out = root / "figures" / "curriculum_results"
    out.mkdir(parents=True, exist_ok=True)
    fu, vt, wm = load(root)

    rows1, _ = fig1(fu, vt, out)
    rows2 = fig2(fu, wm, out)
    rows3 = fig3(wm, out)

    csv_path = out / "data.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fig", "series", "x", "mean", "sd",
                                               "seed0", "seed1", "seed2"])
        writer.writeheader()
        for row in rows1 + rows2 + rows3:
            writer.writerow(row)
    print(f"Wrote {len(rows1)} F1 rows, {len(rows2)} F2 rows, {len(rows3)} F3 rows -> {csv_path}")
    print("Figures:", ", ".join(sorted(p.name for p in out.glob('*.png'))))


if __name__ == "__main__":
    main()
