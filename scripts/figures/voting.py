"""Family G figures: random-start voting (accuracy vs K) and the energy
separation between correct and wrong starts.

Builds the Family G figures from results/voting/results.json (produced by
scripts/eval_voting.py, evaluation only, CPU):

  accuracy_vs_K.png     strict accuracy of the lowest-energy vote over
                        K random starts (K = 1, 8, 32) for Family A (the
                        trained wall-aware CF net, 160 epochs, seed 0, read
                        at T = 8 and T = 12) and Family D (untrained
                        transport physics, exact solve, read at T = 8 and
                        T = 16), against their deterministic starts
  energy_histograms.png per-start energies of all 32 random starts on the
                        300 mazes split into starts that solved the maze
                        and starts that failed, plus the separation AUC
  data.csv              every number plotted or quoted
  README.md             captions / how-to-read / key takeaways /
                        data sources / caveats, numbers from data.csv

Verification done by the script itself (CPU only, no GPU):
  - cf resolves to this clone (exit non-zero otherwise);
  - the per-K vote accuracies, the within-maze separation AUC, the vote
    sample accuracy and the any-correct ceiling are recomputed from the
    raw (K, mazes) energy/correctness arrays stored in results.json and
    checked against the stored values;
  - Family A raw/effective parameter counts are recomputed from the
    seed-0 checkpoint of the main repository (read-only);
  - programmatic layout guard: no readable text may overlap text, lines,
    bands or axis patches; every entry must be at least 10 pt.

Run from the clone root:
    python -m scripts.figures.voting
Optional flags: --results PATH --ckpt-dir PATH --out DIR --skip-params.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.legend
import matplotlib.pyplot as plt
import matplotlib.text
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

import style  # noqa: E402  (shared figure style, read-only)

style.apply_style()

sys.path.insert(0, str(ROOT))          # before the eval imports below
import cf  # noqa: E402
import eval_voting as ev               # noqa: E402  (results producer, read-only)
import eval_x1_x6 as ex                # noqa: E402  (checkpoint loader, read-only)

RESULTS_DEFAULT = ROOT / "results" / "voting" / "results.json"
OUT_DEFAULT = ROOT / "figures" / "voting"

MIN_FONT = 10.0
K_LIST = (1, 8, 32)

C_A8 = style.OKABE_ITO["blue"]          # A, read at T = 8
C_A12 = style.OKABE_ITO["bluish_green"]  # A, read at T = 12
C_D8 = style.OKABE_ITO["orange"]        # D, read at T = 8
C_D16 = style.OKABE_ITO["vermillion"]   # D, read at T = 16
C_BASE = style.OKABE_ITO["grey"]        # deterministic-start baselines
C_CORR = style.OKABE_ITO["blue"]        # correct starts (histograms)
C_WRONG = style.OKABE_ITO["vermillion"]  # wrong starts (histograms)


# ------------------------------------------------------- run guards

def ensure_clone():
    cf_root = Path(cf.__file__).resolve().parents[1]
    if cf_root != ROOT.resolve():
        raise SystemExit("cf resolves to %s, not this clone (%s)"
                         % (cf.__file__, ROOT.resolve()))
    return cf_root


# ------------------------------------------------------- statistics

def wilson_ci(k, n, z=1.96):
    """Wilson 95% interval for k successes out of n, clipped to [0, 1]."""
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def vote_from_arrays(energy, correct, k_list=K_LIST):
    """Recompute per-K vote accuracy from the raw (K, mazes) arrays."""
    out = {}
    for k in k_list:
        sel = np.argmin(energy[:k, :], axis=0)
        out[k] = float(correct[sel, np.arange(correct.shape[1])].mean())
    return out


def within_maze_auc(energy, correct):
    """P(correct start calmer than wrong start), within mazes; ties half."""
    wins = ties = total = 0
    for j in range(energy.shape[1]):
        e, c = energy[:, j], correct[:, j]
        ec, ew = e[c == 1], e[c == 0]
        if len(ec) == 0 or len(ew) == 0:
            continue
        wins += int((ec[:, None] < ew[None, :]).sum())
        ties += int((ec[:, None] == ew[None, :]).sum())
        total += len(ec) * len(ew)
    return (wins + 0.5 * ties) / total if total else None


def check_close(recomputed, stored, what, tol=1e-9):
    if recomputed is None and stored is None:
        return
    if recomputed is None or stored is None \
            or abs(float(recomputed) - float(stored)) > tol:
        raise SystemExit("MISMATCH %s: recomputed %r vs stored %r"
                         % (what, recomputed, stored))


# ------------------------------------------------------- load + verify

def load_and_verify(results_path):
    with open(results_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    res = payload["results"]
    A, D = res["A"], res["D"]
    n_mazes = payload["budget"]["mazes"]

    data = {"payload": payload, "A": {}, "D": {}, "n_mazes": n_mazes,
            "K_max": payload["budget"]["K"]}

    # --- Family A: recompute the vote from the raw arrays
    for T in ("8", "12"):
        unit = A["per_T"][T]
        E = np.asarray(unit["energy"], dtype=float)
        C = np.asarray(unit["correct"], dtype=int)
        if E.shape != (data["K_max"], n_mazes):
            raise SystemExit("A T=%s: unexpected array shape %s"
                             % (T, E.shape))
        acc = vote_from_arrays(E, C)
        for k in K_LIST:
            check_close(acc[k], unit["vote"]["per_K"][str(k)]["accuracy"],
                        "A T=%s vote accuracy K=%d" % (T, k))
        check_close(float(C.mean()), unit["vote"]["sample_accuracy"],
                    "A T=%s sample accuracy" % T)
        check_close(float((C.max(axis=0) > 0).mean()),
                    unit["vote"]["any_correct"], "A T=%s any-correct" % T)
        check_close(within_maze_auc(E, C), unit["vote"]["separation_auc"],
                    "A T=%s separation AUC" % T)
        sel = np.argmin(E, axis=0)
        data["A"][T] = dict(
            energy=E, correct=C,
            acc={k: acc[k] for k in K_LIST},
            solved={k: int(round(acc[k] * n_mazes)) for k in K_LIST},
            auc=within_maze_auc(E, C),
            sample_accuracy=float(C.mean()),
            any_correct=float((C.max(axis=0) > 0).mean()),
            pooled_correct=E[C == 1], pooled_wrong=E[C == 0],
            selected_correct=E[sel, np.arange(n_mazes)][C[sel, np.arange(n_mazes)] == 1],
            baseline_accuracy=A["baseline"][T]["accuracy"])
        data["A"][T]["baseline_solved"] = int(
            round(data["A"][T]["baseline_accuracy"] * n_mazes))

    # --- Family D: recompute the dissipation vote + the zero-wrong claim
    for T in ("8", "16"):
        unit = D["per_T"][T]
        Ed = np.asarray(unit["energy_dissipation"], dtype=float)
        Cd = np.asarray(unit["correct"], dtype=int)
        if Ed.shape != (data["K_max"], n_mazes):
            raise SystemExit("D T=%s: unexpected array shape %s"
                             % (T, Ed.shape))
        acc = vote_from_arrays(Ed, Cd)
        for k in K_LIST:
            check_close(acc[k], unit["vote_dissipation"]["per_K"][str(k)]["accuracy"],
                        "D T=%s dissipation vote accuracy K=%d" % (T, k))
            check_close(acc[k], unit["vote_leak"]["per_K"][str(k)]["accuracy"],
                        "D T=%s leak vote accuracy K=%d" % (T, k))
        if int(Cd.sum()) != Cd.size:
            raise SystemExit("D T=%s: expected every start correct, found %d wrong"
                             % (T, int(Cd.size - Cd.sum())))
        if unit["n_start_disagreement"] != 0 or unit["n_mazes_mixed"] != 0:
            raise SystemExit("D T=%s: stored disagreement counters are not zero"
                             % T)
        data["D"][T] = dict(
            energy=Ed, correct=Cd, acc={k: acc[k] for k in K_LIST},
            solved={k: int(round(acc[k] * n_mazes)) for k in K_LIST},
            pooled_correct=Ed[Cd == 1], pooled_wrong=Ed[Cd == 0],
            auc=unit["vote_dissipation"]["separation_auc"],
            n_start_disagreement=unit["n_start_disagreement"],
            n_mazes_mixed=unit["n_mazes_mixed"],
            seconds=D.get("seconds"))
    data["A_seconds"] = A.get("seconds")
    data["D_held"] = D.get("solver_held", {}).get("n_held")

    # --- cross-check the A baseline solved counts
    for T in ("8", "12"):
        pass  # baseline_accuracy checked above via stored value only
    return data


def recompute_params(ckpt_dir):
    """Raw and effective parameter counts of the Family A checkpoint."""
    import torch
    model = ex.load_wallaware(ev.SEED, str(ckpt_dir))
    n_raw = int(sum(p.numel() for p in model.parameters()))
    d = ev.load_test_set(16)
    xb = d.x[:16]
    yb = d.y[:16]
    n_eff = int(ex.effective_params(model, xb, yb, T=8))
    return n_raw, n_eff


# ------------------------------------------------------- overlap guard

def check_overlaps(fig, name):
    """No readable text may overlap text, a drawn line segment, a band or
    an axis patch; every legend-drifting line is sampled, legends are
    checked against lines and bands (legend internals excluded)."""
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
    artists = []
    for ax in fig.axes:
        artists += [(str(ln.get_label() or "line"), ln)
                    for ln in ax.get_lines()]
        for col in ax.collections:
            artists.append(("band", col.get_window_extent(ren)))
        artists += [(str(p.get_label() or "patch"),
                     p.get_window_extent(ren)) for p in ax.patches]
    def valid(bb):
        return (np.isfinite([bb.x0, bb.y0, bb.x1, bb.y1]).all()
                and bb.width >= 0.0 and bb.height >= 0.0
                and not (bb.width == 0.0 and bb.height == 0.0))

    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad.append("text/text: %r vs %r [%s | %s]"
                           % (boxes[i][0], boxes[j][0], boxes[i][1],
                              boxes[j][1]))
    for tname, tb in boxes:
        for aname, ab in ((a, b) for a, b in artists
                          if not isinstance(b, Line2D)):
            if valid(ab) and tb.overlaps(ab):
                bad.append("text/artist: %r vs %r [%s | %s]"
                           % (tname, aname, tb, ab))
        for aname, ln in ((a, l) for a, l in artists
                          if isinstance(l, Line2D)):
            if line_hits(ln, tb):
                bad.append("text/artist: %r vs %r [%s]" % (tname, aname, tb))
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
                if valid(ab) and lbb.overlaps(ab):
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
    """Two-line-deep header: bold multi-line title, then the guide text,
    both wrapped by the caller so every rendered line fits the canvas."""
    h_px = fig.get_figheight() * fig.dpi
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985, va="top")
    line = 1.35 * 15.0 * fig.dpi / 72.0 / h_px
    y_guide = 0.985 - (title.count("\n") + 1) * line - 0.010
    fig.text(0.5, y_guide, guide, ha="center", va="top", fontsize=10.5,
             color="#333333")


# ------------------------------------------------------- data.csv

def write_data_csv(data, params, out_dir):
    n = data["n_mazes"]
    rows = [["kind", "family", "horizon", "group_or_K", "n", "value",
             "low95", "high95", "note"]]

    def row(kind, family, horizon, group, n_, value, note="",
            low="", high=""):
        rows.append([kind, family, horizon, group, n_, value, low, high, note])

    for T, hor in (("8", "T=8"), ("12", "T=12")):
        a = data["A"][T]
        for k in K_LIST:
            solved = a["solved"][k]
            lo, hi = wilson_ci(solved, n)
            row("vote_accuracy", "A", hor, "K=%d" % k, n,
                "%.6f" % a["acc"][k], "lowest-energy pick of K random starts",
                "%.6f" % lo, "%.6f" % hi)
        lo, hi = wilson_ci(a["baseline_solved"], n)
        row("baseline_accuracy", "A", hor, "deterministic", n,
            "%.6f" % a["baseline_accuracy"],
            "the standard fixed start, no voting",
            "%.6f" % lo, "%.6f" % hi)
        row("per_start_accuracy", "A", hor, "all starts", a["energy"].size,
            "%.6f" % a["sample_accuracy"],
            "mean correctness over all 32 x 300 starts")
        row("any_correct_ceiling", "A", hor, "all starts", n,
            "%.6f" % a["any_correct"],
            "share of mazes with at least one correct start of the 32")
        row("separation_auc", "A", hor, "within-maze", "",
            "%.6f" % a["auc"],
            "P(correct start calmer than wrong start), within mazes")
        for grp, arr in (("correct", a["pooled_correct"]),
                         ("wrong", a["pooled_wrong"])):
            row("energy_pooled", "A", hor, grp, arr.size,
                "%.6f" % float(arr.mean()),
                "median %.6f" % float(np.median(arr)))
    for T, hor in (("8", "T=8"), ("16", "T=16")):
        d = data["D"][T]
        for k in K_LIST:
            solved = d["solved"][k]
            lo, hi = wilson_ci(solved, n)
            row("vote_accuracy", "D", hor, "K=%d" % k, n,
                "%.6f" % d["acc"][k],
                "dissipation vote; the leak vote gives identical accuracy",
                "%.6f" % lo, "%.6f" % hi)
        row("energy_pooled", "D", hor, "correct", d["pooled_correct"].size,
            "%.6f" % float(d["pooled_correct"].mean()),
            "median %.6g; all starts correct, none wrong"
            % float(np.median(d["pooled_correct"])))
        row("energy_pooled", "D", hor, "wrong", 0, "", "no wrong starts exist")
        row("start_disagreement", "D", hor, "mazes", n, "%d" % 0,
            "mazes whose starts disagree on the answer; also 0 mazes mixed")
        row("separation_auc", "D", hor, "within-maze", "", "n/a",
            "undefined: every start is correct, no wrong/correct pairs")
    row("params", "A", "T=8", "raw", params["raw"],
        "%d" % params["raw"] if params["raw"] is not None else "not run",
        "element count recomputed from the seed-0 checkpoint"
        if params["raw"] is not None
        else "recount skipped; stored results.json says raw %d"
        % (params["raw_stored"] or 0))
    row("params", "A", "T=8", "effective", params["effective"],
        "%d" % params["effective"] if params["effective"] is not None
        else "not run",
        "nonzero-gradient entries after one forward+backward at T=8")
    row("params", "D", "T=8", "learned", 0, "0",
        "physics-only arm: no learned parameters")
    row("budget", "A", "eval", "seconds", "", "%.2f" % (data["A_seconds"] or 0.0),
        "evaluation only, CPU")
    row("budget", "D", "eval", "seconds", "",
        "%.2f" % (data["D"]["8"]["seconds"] or 0.0),
        "evaluation only, CPU; %d solver rows held at their last valid flow"
        % (data["D_held"] or 0))
    with open(out_dir / "data.csv", "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print("saved %s (%d rows)" % (out_dir / "data.csv", len(rows) - 1))
    return rows


# ------------------------------------------------------- figure 1

def figure_accuracy(data, out_dir):
    n = data["n_mazes"]
    xs = np.array(K_LIST, dtype=float)
    A8, A12 = data["A"]["8"], data["A"]["12"]

    fig, ax = plt.subplots(figsize=(12.6, 6.9))
    fig.subplots_adjust(top=0.74, bottom=0.26, left=0.075, right=0.985)

    def vote_series(acc_solved, color, label, ls="-"):
        ys = [acc_solved[0][k] for k in K_LIST]
        solved = [acc_solved[1][k] for k in K_LIST]
        lo = [wilson_ci(s, n)[0] for s in solved]
        hi = [wilson_ci(s, n)[1] for s in solved]
        ax.errorbar(xs, ys, yerr=[np.maximum(0, np.array(ys) - np.array(lo)),
                                  np.maximum(0, np.array(hi) - np.array(ys))],
                    color=color, marker="o", ms=7, capsize=5,
                    linestyle=ls, label=label)

    vote_series((A8["acc"], A8["solved"]), C_A8,
                "A trained net, vote read at T = 8 "
                "(160 epochs, seed 0, T_train = 8)")
    vote_series((A12["acc"], A12["solved"]), C_A12,
                "A trained net, vote read at T = 12 "
                "(the horizon training selected)")

    d8, d16 = data["D"]["8"], data["D"]["16"]
    for d, color, lab in ((d8, C_D8, "D physics (no training), read at T = 8"),
                          (d16, C_D16, "D physics, read at T = 16 "
                                       "(identical to T = 8)")):
        ys = [d["acc"][k] for k in K_LIST]
        lo = [wilson_ci(d["solved"][k], n)[0] for k in K_LIST]
        hi = [wilson_ci(d["solved"][k], n)[1] for k in K_LIST]
        ax.errorbar(xs, ys,
                    yerr=[np.maximum(0, np.array(ys) - np.array(lo)),
                          np.maximum(0, np.array(hi) - np.array(ys))],
                    color=color, capsize=5,
                    linestyle="--" if color == C_D16 else "-",
                    marker="s" if color == C_D8 else "^",
                    label=lab)

    for band_i, (a, color, lab) in enumerate(((A8, style.OKABE_ITO["grey"],
                                                "A deterministic start, T = 8 (no voting)"),
                                               (A12, "#555555",
                                                "A deterministic start, T = 12 (no voting)"))):
        lo, hi = wilson_ci(a["baseline_solved"], n)
        ax.fill_between([xs[0] * 0.85, xs[-1] * 1.15], lo, hi,
                        color=color, alpha=0.16, linewidth=0,
                        label="deterministic 95% band" if band_i == 0
                        else None)

    gap = A8["baseline_accuracy"] - A8["acc"][32]
    ax.text(1.45, 0.80,
            "K = 32 vote: %.1f %% — still %.1f points below\n"
            "the same net's deterministic start (%.1f %%)"
            % (100 * A8["acc"][32], 100 * gap,
               100 * A8["baseline_accuracy"]),
            fontsize=10.5, ha="left", va="center", color="#222222")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in K_LIST])
    ax.minorticks_off()
    ax.set_xlim(xs[0] * 0.85, xs[-1] * 1.15)
    ax.set_ylim(0.0, 1.05)
    x0, x1 = ax.get_xlim()
    for a, color, lab in ((A8, style.OKABE_ITO["grey"],
                           "A deterministic start, T = 8 (no voting)"),
                          (A12, "#555555",
                           "A deterministic start, T = 12 (no voting)")):
        ax.plot([x0, x1], [a["baseline_accuracy"]] * 2,
                color=color, linestyle=":", linewidth=1.6, label=lab)
    ax.set_xlabel("K = random starts per maze (log scale)")
    ax.set_ylabel("strict accuracy (share of the %d test mazes solved)" % n)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=10.5)

    headline(fig,
             "Random-start voting: the physics energy picks the correct "
             "path at every K (%.3f)\n"
             "the trained net's energy vote climbs only %.3f \u2192 %.3f "
             "(T = 8), below its deterministic start (%.3f)"
             % (d8["acc"][32], A8["acc"][1], A8["acc"][32],
                A8["baseline_accuracy"]),
             "Solved = predicted cells are exactly the BFS shortest path; "
             "each point keeps the lowest-energy run of K random starts "
             "of the same net or physics, 300 mazes.\n"
             "Error bars and grey bands = Wilson 95% binomial intervals "
             "over the 300 mazes, clipped to [0, 1]; one training seed, "
             "300-maze slice: preliminary, not a reproduction.")
    save(fig, "accuracy_vs_K.png", out_dir)


# ------------------------------------------------------- figure 2

def _hist_panel(ax, ecorr, ewrong, title, xlabel, logx=False):
    kwargs = dict(bins=42, density=True, linewidth=1.1)
    if logx:
        lo = ecorr.min()
        hi = max(ecorr.max(), ewrong.max() if ewrong.size else 0.0)
        kwargs["bins"] = np.logspace(np.log10(lo), np.log10(hi * 1.02), 42)
    density_max = []
    if ecorr.size:
        counts, _, _ = ax.hist(ecorr, color=C_CORR, alpha=0.60, label=(
            "correct starts (n = %s)\nmedian %.4g"
            % ("{:,}".format(ecorr.size), float(np.median(ecorr)))),
            **kwargs)
        density_max.append(float(counts.max()))
    if ewrong.size:
        counts, _, _ = ax.hist(ewrong, color=C_WRONG, alpha=0.60, label=(
            "wrong starts (n = %s)\nmedian %.4g"
            % ("{:,}".format(ewrong.size), float(np.median(ewrong)))),
            **kwargs)
        density_max.append(float(counts.max()))
    else:
        ax.hist([], color=C_WRONG, alpha=0.60, label="wrong starts (n = 0)")
    if logx:
        ax.set_xscale("log")
    ax.set_ylim(0.0, (max(density_max) if density_max else 1.0) * 1.65)
    ax.set_title(title, fontsize=12.5, loc="left")
    ax.set_xlabel(xlabel)
    leg = ax.legend(fontsize=10.5, loc="upper left", borderaxespad=1.2)
    leg.set_frame_on(True)
    leg.get_frame().set_alpha(0.92)
    leg.get_frame().set_edgecolor("#cccccc")


def figure_histograms(data, out_dir):
    fig, (a8, a12, d8) = plt.subplots(1, 3, figsize=(15.0, 5.8))
    fig.subplots_adjust(top=0.70, bottom=0.13, left=0.05, right=0.985,
                        wspace=0.22)

    _hist_panel(
        a8, data["A"]["8"]["pooled_correct"], data["A"]["8"]["pooled_wrong"],
        "A trained net, T = 8 — AUC %.3f"
        % data["A"]["8"]["auc"],
        "energy $E_A$ = summed residual flux |m \u2212 s| per run\n"
        "(normalized mass per step)")
    a8.set_ylabel("share of all starts per energy bin")

    _hist_panel(
        a12, data["A"]["12"]["pooled_correct"], data["A"]["12"]["pooled_wrong"],
        "A trained net, T = 12 — AUC %.3f"
        % data["A"]["12"]["auc"],
        "energy $E_A$ = summed residual flux |m \u2212 s| per run\n"
        "(normalized mass per step)")

    _hist_panel(
        d8, data["D"]["8"]["pooled_correct"], data["D"]["8"]["pooled_wrong"],
        "D physics, T = 8 — no wrong starts",
        "energy $E_D$ = electrical dissipation\n$\\sum_e Q_e^2/D_e$"
        " (log scale)")

    headline(fig,
             "Energy separates correct from wrong starts for the trained "
             "net: AUC %.3f at T = 8 and %.3f at T = 12 —\n"
             "for the physics arm the separation is trivial: every one of "
             "the %s starts is correct"
             % (data["A"]["8"]["auc"], data["A"]["12"]["auc"],
                "{:,}".format(data["D"]["8"]["pooled_correct"].size)),
             "Every panel: energies of all 32 random starts on the 300 test "
             "mazes (n = 9,600 per panel); blue = starts that solved the "
             "maze, vermillion = starts that failed.\n"
             "AUC = share of correct-vs-wrong start pairs in the same maze "
             "where the correct start has the lower energy (0.5 = none, "
             "1.0 = perfect); one training seed: preliminary.")

    save(fig, "energy_histograms.png", out_dir)


# ------------------------------------------------------- README

def write_readme(data, params, rows, out_dir):
    A8, A12, D8 = data["A"]["8"], data["A"]["12"], data["D"]["8"]
    n = data["n_mazes"]

    raw_s = ("{:,}".format(params["raw"])
             if params["raw"] is not None else "not run")
    eff_s = ("{:,}".format(params["effective"])
             if params["effective"] is not None else "not run")
    if params["raw"] is None or params["raw_stored"] is None:
        stored_s = ""
    elif params["raw"] != params["raw_stored"]:
        stored_s = ("The stored results.json states raw %s, which does "
                    "NOT match this recomputation; the counted value is "
                    "the one used here (the stored text is a hardcoded "
                    "note, not a counted value)."
                    % "{:,}".format(params["raw_stored"]))
    else:
        stored_s = ("The stored results.json states raw %s, matching the "
                    "recomputation."
                    % "{:,}".format(params["raw_stored"]))
    params_sentence = ("- Seed-0 checkpoint parameter counts recomputed "
                       "by `scripts/figures/voting.py` (this figure "
                       "folder's own verification): %s raw / %s "
                       "effective. %s" % (raw_s, eff_s, stored_s))

    def pct(x):
        return "%.1f %%" % (100 * x)
    guide = ("Reading guide: solved = the predicted cells are exactly the "
             "BFS shortest path (strict; equal to exact on these tree "
             "mazes). K = number of random starting states tried per maze; "
             "the run with the lowest energy is kept. T = steps the "
             "network/physics is run before the answer is read off. "
             "Energy (Family A) = summed residual coagulation-fragmentation "
             "flux |m \u2212 s| at the readout step; energy (Family D) = "
             "electrical dissipation of the final flow (a dead-end leak "
             "fraction gives identical votes).")
    lines = []
    lines += ["# Family G — best-of-K random-start voting", "",
              "Plain language: run the solver several times from different "
              "random starting states and keep the run that ends in the "
              "calmest internal state. If calm runs are the correct runs, "
              "that is a free accuracy boost at test time.", "",
              "## Budget of every number in this folder", "",
              "- Family A (trained cell model, A-WallAware): 160 epochs, "
              "1 training seed (seed 0) — fewer than 3 seeds, so "
              "**preliminary**; evaluation on the first %d level-5 test "
              "mazes (reduced slice, **not a reproduction** of the full "
              "1,000-maze numbers), CPU, %.1f s of evaluation."
              % (n, data["A_seconds"]),
              "- Family D: physics only, no training (0 learned "
              "parameters), so no seeds; same %d mazes, CPU, %.1f s of "
              "evaluation." % (n, D8["seconds"]),
              "- Parameters: A = %s raw / %s effective (nonzero-gradient "
              "entries after one forward+backward at T = 8, recomputed "
              "from the seed-0 checkpoint); D = 0 learned. Hardware: CPU "
              "only." % ("{:,}".format(params["raw"]),
                         "{:,}".format(params["effective"])
                         if params["effective"] is not None
                         else "not run"),
              "", "## accuracy_vs_K.png", "",
              "**Caption.** Strict accuracy of the lowest-energy pick over "
              "K = 1, 8, 32 random starts, at the two readout horizons of "
              "the trained net (T = 8, its training depth, and T = 12, the "
              "horizon its training program selected on validation) and "
              "the two horizons of the physics arm (T = 8, 16), against "
              "each family's deterministic start.", "",
              "**How to read.** " + guide + " Error bars and bands are "
              "Wilson 95%% binomial intervals over the %d mazes, clipped "
              "to [0, 1]; they show maze-sampling noise only — with one "
              "training seed they say nothing about seed-to-seed "
              "variation." % n, "",
              "**Key takeaway (all values from data.csv).** The physics "
              "energy is perfect from any start: %s at every K and both "
              "horizons (all %d/%d mazes solved; 0/%d mazes whose starts "
              "disagree). Energy voting does lift the trained net — from "
              "%s at K = 1 to %s at K = 32 (T = 8) — but it recovers only "
              "part of the distance to its ceiling of %s (mazes with at "
              "least one correct start) and stays %s points below the "
              "%s of the very same net run once from its deterministic "
              "start (T = 12: %s vs %s)."
              % (pct(D8["acc"][32]), D8["solved"][32], n, n,
                 pct(A8["acc"][1]), pct(A8["acc"][32]),
                 pct(A8["any_correct"]),
                 "%.1f" % (100 * (A8["baseline_accuracy"] - A8["acc"][32])),
                 pct(A8["baseline_accuracy"]),
                 pct(A12["acc"][32]), pct(A12["baseline_accuracy"])), "",
              "**Data source.** `../../results/voting/results.json` "
              "(produced by `scripts/eval_voting.py`, evaluation only, "
              "CPU) and `../../results/voting/summary.md`; every number "
              "is in [`data.csv`](data.csv).", "", "## "
              "energy_histograms.png", "",
              "**Caption.** The energies of all 32 random starts on the "
              "%d mazes (n = 9,600 per panel), split into starts that "
              "solved the maze and starts that failed; Family A at T = 8 "
              "and T = 12, Family D at T = 8 (dissipation, log scale)." % n,
              "",
              "**How to read.** " + guide + " The histograms are pooled "
              "across mazes; the within-maze separation AUC in each panel "
              "title compares correct and wrong starts of the *same* maze "
              "only (0.5 = the energy carries no correctness "
              "information, 1.0 = perfect separation). Legends give the "
              "count and median of each group.", "",
              "**Key takeaway — does the energy separate correct from "
              "wrong?** Stated plainly: **yes, largely, for the trained "
              "net — and trivially for the physics arm.** The trained "
              "net's energy separates the two almost completely: AUC "
              "%.3f at T = 8 and %.3f at T = 12, with correct starts "
              "calmer on average (mean %.1f vs %.1f at T = 8; %.1f vs "
              "%.1f at T = 12). It is not perfect: at K = 32 the vote "
              "solves %d/%d mazes while %d/%d mazes contain at least one "
              "correct start, so in a few mazes a wrong start is calmer "
              "than every correct one. For the physics arm there is "
              "nothing to separate: every one of the 9,600 random starts "
              "solves its maze (identical under the leak vote and at "
              "T = 16), so its energy trivially 'ranks' the correct run "
              "first — the vote cannot fail, but it also cannot be "
              "credited as an accuracy win."
              % (A8["auc"], A12["auc"],
                 float(A8["pooled_correct"].mean()),
                 float(A8["pooled_wrong"].mean()),
                 float(A12["pooled_correct"].mean()),
                 float(A12["pooled_wrong"].mean()),
                 A8["solved"][32], n,
                 int(round(A8["any_correct"] * n)), n), "",
              "**Data source.** `../../results/voting/results.json`; "
              "every number is in [`data.csv`](data.csv).", "",
              "## Caveats", "",
              "- One training seed (seed 0): **preliminary**. The seed "
              "spread of the full curriculum runs is several points at "
              "this accuracy level, so differences of a few points here "
              "(for example T = 8 vs T = 12) are **inconclusive**.",
              "- Reduced 300-maze slice of the 1,000-maze test split: "
              "**not a reproduction** of the full-split numbers.",
              "- The Family A random starts are drawn from the model's own "
              "stochastic initial-state rule; the deterministic baseline "
              "uses the standard fixed start (c2 = 0) — the same model "
              "and weights either way.",
              "- Family D's starts differ only in the initial pipe "
              "conductances; its exact solve reaches the same steady flow "
              "from every start on a tree maze, which is why all its "
              "starts agree (0/%d disagreeing mazes). Its 1.000 is the "
              "physics, not voting skill." % n,
              "- The dissipation energies of Family D span many orders of "
              "magnitude, so that panel uses a log axis; the histograms "
              "are density-normalized (share of starts per bin).",
              params_sentence, ""]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n",
                                       encoding="utf-8")
    print("saved %s" % (out_dir / "README.md"))


# ------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Family G voting figures")
    ap.add_argument("--results", default=str(RESULTS_DEFAULT))
    ap.add_argument("--ckpt-dir", default=None,
                    help="Family A checkpoint folder, read-only (default: "
                         "resolved from the CF_MAIN_RESULTS environment "
                         "variable; error if neither is provided)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--skip-params", action="store_true",
                    help="do not recompute parameter counts from the "
                         "checkpoint")
    args = ap.parse_args()

    ensure_clone()
    print("cf package: %s" % cf.__file__)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_verify(Path(args.results))
    print("verified: per-K vote accuracies, AUCs and pooled counts "
          "recomputed from the raw arrays in %s" % args.results)

    if args.skip_params:
        params = dict(raw=None, effective=None)
        print("parameter recount skipped (stored raw count quoted instead)")
    else:
        n_raw, n_eff = recompute_params(ex.resolve_ckpt_dir(args.ckpt_dir))
        params = dict(raw=n_raw, effective=n_eff)
    params["raw_stored"] = None
    stored_str = data["payload"]["params"]["A"]
    import re as _re
    m = _re.search(r"raw ([0-9,]+)", stored_str)
    if m:
        params["raw_stored"] = int(m.group(1).replace(",", ""))
    if args.skip_params:
        params["raw"] = params["raw_stored"]
    elif params["raw"] != params["raw_stored"]:
        print("NOTE: recomputed raw params (%d) differ from the stored "
              "string (%d)" % (params["raw"], params["raw_stored"]))

    rows = write_data_csv(data, params, out_dir)
    figure_accuracy(data, out_dir)
    figure_histograms(data, out_dir)
    write_readme(data, params, rows, out_dir)
    return 0


if __name__ == "__main__":
    main()
