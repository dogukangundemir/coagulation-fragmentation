"""Main results figure (packet F7): test strict accuracy at the
validation-selected test-time depth T*, one bar per model.

Reads (paths derived from this file's location, or --root override):
  week2_results/valT_selection.json          40-epoch runs, T* per seed
  week2_results/week2_metrics.json           40-epoch runs, raw parameter counts
  week2_results/wallaware/valT_selection.json CF-WallAware, T* per seed
  week2_results/wallaware/week2_metrics.json  CF-WallAware raw/eff. parameters
  week2_results/wallaware160/results.json     CF-WallAware, 160-epoch runs
  week2_results/followups/results.json        160-epoch and variable-T runs

Writes figures/main_results/{main_results.png, data.csv}.

Run:  python scripts/figures/main_results.py [--root REPO_ROOT]

Each bar is the mean over 3 seeds of test strict accuracy at that seed's
validation-selected T* (T* = argmax val strict over candidate depths, ties
to smaller T). Error bars are the sample sd (n-1); dots are the individual
seeds. Labels carry the model, its training setting, its training depth
and its raw and effective parameter counts (effective = parameter
elements with nonzero gradient after one forward+backward on a real
training batch, as recorded in followups/results.json and in
wallaware/week2_metrics.json).
"""

import argparse
import csv
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import DPI, OKABE_ITO, SEED_MARKER, apply_style
from curriculum_figures import assert_no_text_overlaps, place_footnote

F7 = "F7 main results"


def repo_root(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", type=Path, default=None)
    args, _ = p.parse_known_args(argv)
    return args.root if args.root else Path(__file__).resolve().parents[2]


def load(root):
    vt = root / "week2_results" / "valT_selection.json"
    wm = root / "week2_results" / "week2_metrics.json"
    wa_vt = root / "week2_results" / "wallaware" / "valT_selection.json"
    wa_m = root / "week2_results" / "wallaware" / "week2_metrics.json"
    wa160 = root / "week2_results" / "wallaware160" / "results.json"
    fu = root / "week2_results" / "followups" / "results.json"
    import json
    with vt.open() as fh:
        vt = json.load(fh)
    with wm.open() as fh:
        wm = json.load(fh)
    with wa_vt.open() as fh:
        wa_vt = json.load(fh)
    with wa_m.open() as fh:
        wa_m = json.load(fh)
    with wa160.open() as fh:
        wa160 = json.load(fh)
    with fu.open() as fh:
        fu = json.load(fh)
    return vt, wm, wa_vt, wa_m, wa160, fu


def by_model(selections, model):
    return sorted((s for s in selections if s["model"] == model),
                  key=lambda s: s["seed"])


def fu_by_exp(fu, exp):
    return sorted((r for r in fu["runs"].values() if r["exp"] == exp),
                  key=lambda r: r["seed"])


def build_rows(vt, wm, wa_vt, wa_m, wa160, fu):
    """One row per model: seed values at T*, mean, sample sd, params."""
    wm_params = {r["model"]: r["n_params"] for r in wm["runs"].values()}
    wa_raw = {r["model"]: r["n_params"] for r in wa_m["runs"].values()}
    wa_eff = {r["model"]: r["n_params_effective"]
              for r in wa_m["runs"].values()}
    pc = fu["param_counts"]
    wa160_pc = wa160["param_counts"]["E5_CF-WallAware160"]
    wa160_runs = sorted(wa160["runs"].values(), key=lambda r: r["seed"])
    assert all(r["exp"] == "E5_CF-WallAware160" for r in wa160_runs)
    # Effective counts (parameter elements with a nonzero gradient after
    # one forward+backward on a real training batch; protocol recorded in
    # fu.param_counts.protocol) exist for the E1 hid-16 CF architecture,
    # the E3 RecurrentCNN, the hid-18 CF-NoFrag (E4), the CF-WallAware
    # architecture (E5 shares it) and CF-WallAware; the
    # 40-epoch CF-Full and hid-16 CF-NoFrag runs share the E1 architecture.

    def label(name, raw, eff):
        return f"{name}\n{raw:,} raw / {eff:,} effective params"

    def sel(model):
        return [s["test_strict_at_T_star"] for s in by_model(vt["selections"], model)]

    def sel_wa(model):
        return [s["test_strict_at_T_star"] for s in by_model(wa_vt["selections"], model)]

    def sel_fu(exp):
        return [r["test_strict_at_T_star"] for r in fu_by_exp(fu, exp)]

    def tstars(records, key="T_star"):
        return [r[key] for r in records]

    rows = [
        dict(model="RecurrentCNN fixed T=8",
             label=label("RecurrentCNN, fixed T=8, 40 epochs",
                         wm_params["RecurrentCNN"],
                         pc["E3_RecurrentCNN"]["effective"]),
             values=sel("RecurrentCNN"),
             params_raw=wm_params["RecurrentCNN"],
             params_eff=pc["E3_RecurrentCNN"]["effective"],
             T_star=tstars(by_model(vt["selections"], "RecurrentCNN"))),
        dict(model="CF-Full 40",
             label=label("CF-Full, hid 16, trained T=8, 40 epochs",
                         wm_params["CF-Full"],
                         pc["E1_CF-Full"]["effective"]),
             values=sel("CF-Full"),
             params_raw=wm_params["CF-Full"],
             params_eff=pc["E1_CF-Full"]["effective"],
             T_star=tstars(by_model(vt["selections"], "CF-Full"))),
        dict(model="CF-NoFrag 40",
             label=label("CF-NoFrag, hid 16, trained T=8, 40 epochs",
                         wm_params["CF-NoFrag"],
                         pc["E1_CF-Full"]["effective"]),
             values=sel("CF-NoFrag"),
             params_raw=wm_params["CF-NoFrag"],
             params_eff=pc["E1_CF-Full"]["effective"],
             T_star=tstars(by_model(vt["selections"], "CF-NoFrag"))),
        dict(model="CF-WallAware 40",
             label=label("CF-WallAware, hid 16, trained T=8, 40 epochs",
                         wa_raw["CF-WallAware"], wa_eff["CF-WallAware"]),
             values=sel_wa("CF-WallAware"),
             params_raw=wa_raw["CF-WallAware"],
             params_eff=wa_eff["CF-WallAware"],
             T_star=tstars(by_model(wa_vt["selections"], "CF-WallAware"))),
        dict(model="CF-WallAware 160",
             label=label("CF-WallAware, hid 16, trained T=8, 160 epochs",
                         wa160_pc["raw"], wa160_pc["effective"]),
             values=[r["test_strict_at_T_star"] for r in wa160_runs],
             params_raw=wa160_pc["raw"],
             params_eff=wa160_pc["effective"],
             T_star=tstars(wa160_runs)),
        dict(model="CF-Full 160",
             label=label("CF-Full, hid 16, trained T=8, 160 epochs",
                         pc["E1_CF-Full"]["raw"],
                         pc["E1_CF-Full"]["effective"]),
             values=sel_fu("E1_CF-Full"),
             params_raw=pc["E1_CF-Full"]["raw"],
             params_eff=pc["E1_CF-Full"]["effective"],
             T_star=tstars(fu_by_exp(fu, "E1_CF-Full"))),
        dict(model="CF-NoFrag hid18 160",
             label=label("CF-NoFrag, hid 18, trained T=8, 160 epochs",
                         pc["E4_CF-NoFrag160"]["raw"],
                         pc["E4_CF-NoFrag160"]["effective"]),
             values=sel_fu("E4_CF-NoFrag160"),
             params_raw=pc["E4_CF-NoFrag160"]["raw"],
             params_eff=pc["E4_CF-NoFrag160"]["effective"],
             T_star=tstars(fu_by_exp(fu, "E4_CF-NoFrag160"))),
        dict(model="RecurrentCNN variable T",
             label=label("RecurrentCNN, variable T, 40 epochs",
                         pc["E3_RecurrentCNN"]["raw"],
                         pc["E3_RecurrentCNN"]["effective"]),
             values=sel_fu("E3_RecurrentCNN"),
             params_raw=pc["E3_RecurrentCNN"]["raw"],
             params_eff=pc["E3_RecurrentCNN"]["effective"],
             T_star=tstars(fu_by_exp(fu, "E3_RecurrentCNN"))),
    ]
    for row in rows:
        assert len(row["values"]) == 3, row["model"]
        row["mean"] = statistics.mean(row["values"])
        row["sd"] = statistics.stdev(row["values"])
    return rows


COLORS = {
    "RecurrentCNN fixed T=8": OKABE_ITO["reddish_purple"],
    "RecurrentCNN variable T": OKABE_ITO["bluish_green"],
    "CF-Full 40": OKABE_ITO["blue"],
    "CF-Full 160": OKABE_ITO["blue"],
    "CF-NoFrag 40": OKABE_ITO["vermillion"],
    "CF-NoFrag hid18 160": OKABE_ITO["vermillion"],
    "CF-WallAware 40": OKABE_ITO["orange"],
    "CF-WallAware 160": OKABE_ITO["orange"],
}


def fig7(rows, out):
    rows = sorted(rows, key=lambda r: r["mean"])  # ascending: best on top
    n = len(rows)
    y = list(range(n))
    means = [r["mean"] for r in rows]
    sds = [r["sd"] for r in rows]
    colors = [COLORS[r["model"]] for r in rows]

    fig, ax = plt.subplots(figsize=(12.5, 5.8), layout="constrained")
    # NOTE: error bars are drawn as explicit line artists rather than
    # barh(xerr=...): barh stores every bar's cap points in one hidden
    # Line2D (linestyle 'None'), which the shared overlap checker would
    # sample as phantom diagonal lines between bars.
    ax.barh(y, means, height=0.62, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Test strict accuracy at validation-selected T* (fraction of test mazes)")
    ax.set_title("Main results: test strict accuracy at validation-selected T*")

    # Seed dots: small deterministic vertical offsets so a seed value that
    # coincides with an error-bar cap stays readable.
    offsets = [-0.16, 0.0, 0.16]
    for row, yi in zip(rows, y):
        for val, off in zip(row["values"], offsets):
            ax.plot(val, yi + off, SEED_MARKER, color="black",
                    markersize=5, markerfacecolor="white",
                    markeredgewidth=1.2, zorder=5)

    # Sample-sd error bars: horizontal line mean-sd .. mean+sd with small
    # vertical caps, drawn as plain visible lines.
    cap_h = 0.07
    for row, yi in zip(rows, y):
        lo, hi = row["mean"] - row["sd"], row["mean"] + row["sd"]
        ax.plot([lo, hi], [yi, yi], color="#333333", lw=1.4, zorder=4)
        for x in (lo, hi):
            ax.plot([x, x], [yi - cap_h, yi + cap_h], color="#333333",
                    lw=1.4, zorder=4)

    # Mean +- sd value labels, right of the error-bar cap.
    for row, yi in zip(rows, y):
        ax.text(row["mean"] + row["sd"] + 0.018, yi,
                f"{row['mean']:.3f} \u00b1 {row['sd']:.3f}",
                va="center", ha="left", fontsize=11, color="#222222")

    handles = [
        Patch(facecolor="#999999", alpha=0.85, label="mean over 3 seeds"),
        Line2D([0], [0], color="#333333", lw=1.4, label="sample sd (n\u22121)"),
        Line2D([0], [0], marker=SEED_MARKER, color="black",
                markerfacecolor="white", markersize=6, lw=0,
                label="individual seed"),
    ]
    ax.legend(handles=handles, loc="lower right")

    place_footnote(
        fig, ax,
        "\n".join(textwrap.wrap(
            "n = 3 seeds per model; error bars = sample sd (n\u22121); dots = "
            "individual seeds. T* = argmax validation strict accuracy over "
            "candidate test depths (ties to smaller T), selected per seed. "
            "Strict accuracy is a fraction in [0, 1]; no shading band is "
            "drawn, so no band clipping applies.", width=105)),
        fontsize=10)

    assert_no_text_overlaps(fig, F7)
    fig.savefig(out / "main_results.png", dpi=DPI)
    plt.close(fig)


def main():
    apply_style()
    root = repo_root()
    vt, wm, wa_vt, wa_m, wa160, fu = load(root)
    rows = build_rows(vt, wm, wa_vt, wa_m, wa160, fu)

    out = root / "figures" / "main_results"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "data.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "label", "params_raw", "params_effective",
                    "seed0", "seed1", "seed2", "mean", "sample_sd",
                    "T_star_seed0", "T_star_seed1", "T_star_seed2"])
        for r in rows:
            w.writerow([r["model"], r["label"], r["params_raw"], r["params_eff"],
                        *r["values"], f"{r['mean']:.6f}", f"{r['sd']:.6f}",
                        *r["T_star"]])
    print(f"Wrote {len(rows)} rows -> {csv_path}")

    fig7(rows, out)
    print(f"Figures: {', '.join(sorted(p.name for p in out.glob('*.png')))}")


if __name__ == "__main__":
    main()
