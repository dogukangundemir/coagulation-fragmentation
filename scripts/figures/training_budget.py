"""Training-budget figure: test strict accuracy at T* versus training
epochs (40 vs 160), one connected pair per model family (CF-Full,
CF-NoFrag, CF-WallAware, RecurrentCNN fixed T=8, RecurrentCNN variable T).

Reads (paths derived from this file's location, or --root override):
  week2_results/valT_selection.json           RecurrentCNN fixed-T and CF-Full
                                             40-epoch runs: test strict at T*
  week2_results/week2_metrics.json           raw params, training minutes
  week2_results/wallaware/valT_selection.json CF-WallAware 40, test strict at T*
  week2_results/wallaware/week2_metrics.json  CF-WallAware 40, minutes/hardware
  week2_results/wallaware160/results.json     CF-WallAware 160 epochs
  week2_results/followups/results.json        E1 CF-Full 160, E2 CF-NoFrag hid
                                             18 40, E3 RecurrentCNN variable T
                                             40, E4 CF-NoFrag hid 18 160
  week2_results/rcnn160/results.json         RecurrentCNN 160 epochs: fixed
                                             T=8 (E6) and variable T (E7)

Writes figures/training_budget/{training_budget.png, data.csv}.

Run:  python scripts/figures/training_budget.py [--root REPO_ROOT]

Each family is one coloured connected pair: mean test strict accuracy at
the validation-selected depth T* over 3 seeds, sample-sd (n-1) error bar,
open dots for the individual seeds (offset in x only for readability).
The CF-NoFrag pair uses the hid-18 architecture at both budgets (E2 at 40
epochs, E4 at 160 epochs, the only architecture-matched pair); the hid-16
40-epoch CF-NoFrag run is plotted in the main-results figure instead.
"""

import argparse
import csv
import json
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import DPI, OKABE_ITO, SEED_MARKER, apply_style
from curriculum_figures import assert_no_text_overlaps, place_footnote

FNAME = "training budget (40 vs 160 epochs)"

FAMILY_COLORS = {
    "CF-Full": OKABE_ITO["blue"],
    "CF-NoFrag": OKABE_ITO["vermillion"],
    "CF-WallAware": OKABE_ITO["orange"],
    "RecurrentCNN fixed": OKABE_ITO["reddish_purple"],
    "RecurrentCNN variable": OKABE_ITO["bluish_green"],
}


def repo_root(argv=None):
    argv = sys.argv[1:] if argv is None else sys.argv
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", type=Path, default=None)
    args, _ = p.parse_known_args(argv)
    return args.root if args.root else Path(__file__).resolve().parents[2]


def load(root):
    paths = {
        "vt": root / "week2_results" / "valT_selection.json",
        "wm": root / "week2_results" / "week2_metrics.json",
        "wa_vt": root / "week2_results" / "wallaware" / "valT_selection.json",
        "wa_m": root / "week2_results" / "wallaware" / "week2_metrics.json",
        "wa160": root / "week2_results" / "wallaware160" / "results.json",
        "fu": root / "week2_results" / "followups" / "results.json",
        "rc": root / "week2_results" / "rcnn160" / "results.json",
    }
    return {k: json.load(p.open()) for k, p in paths.items()}


def by_model(selections, model):
    return sorted((s for s in selections if s["model"] == model),
                  key=lambda s: s["seed"])


def by_exp(runs, exp):
    return sorted((r for r in runs.values() if r["exp"] == exp),
                  key=lambda r: r["seed"])


def mean_minutes_from_seconds(runs):
    return statistics.mean(r["train_seconds"] for r in runs) / 60.0


def mean_minutes(runs):
    return statistics.mean(r["wall_minutes"] for r in runs)


def build_rows(d):
    """One row per family: 40-epoch and 160-epoch seed values at T*."""
    vt, wm, wa_vt, wa_m = d["vt"], d["wm"], d["wa_vt"], d["wa_m"]
    wa160, fu, rc = d["wa160"], d["fu"], d["rc"]

    def wm_runs(src, model):
        return sorted((r for r in src["runs"].values()
                       if r["model"] == model), key=lambda r: r["seed"])

    wm_rcnn = wm_runs(wm, "RecurrentCNN")
    wm_full = wm_runs(wm, "CF-Full")
    wa_runs = sorted(wa_m["runs"].values(), key=lambda r: r["seed"])
    wa160_runs = sorted(wa160["runs"].values(), key=lambda r: r["seed"])
    e1 = by_exp(fu["runs"], "E1_CF-Full")
    e2 = by_exp(fu["runs"], "E2_CF-NoFrag")
    e3 = by_exp(fu["runs"], "E3_RecurrentCNN")
    e4 = by_exp(fu["runs"], "E4_CF-NoFrag160")
    e6 = by_exp(rc["runs"], "E6_RecurrentCNN160")
    e7 = by_exp(rc["runs"], "E7_RecurrentCNN160-Tsamp")

    hw_3070 = f"{wm['device_name']} ({wm['device']})"
    hw_rc = f"{rc['device_name']} ({rc['device']})"
    hw_cuda_unspec = "cuda (GPU model not recorded in results.json)"

    def vals(records):
        return [r["test_strict_at_T_star"] for r in records]

    def selt(sel_records):
        return [s["test_strict_at_T_star"] for s in sel_records]

    families = [
        dict(family="CF-Full", label="CF-Full, hid 16, trained T=8",
             v40=selt(by_model(vt["selections"], "CF-Full")),
             v160=vals(e1), T_star40=[s["T_star"] for s in by_model(vt["selections"], "CF-Full")],
             T_star160=[r["T_star"] for r in e1],
             min40=mean_minutes_from_seconds(wm_full), min160=mean_minutes(e1),
             hw40=hw_3070, hw160=hw_cuda_unspec,
             sources40="week2_results/valT_selection.json; week2_results/week2_metrics.json",
             sources160="week2_results/followups/results.json"),
        dict(family="CF-NoFrag",
             label="CF-NoFrag, hid 18, trained T=8",
             v40=vals(e2), v160=vals(e4),
             T_star40=[r["T_star"] for r in e2],
             T_star160=[r["T_star"] for r in e4],
             min40=mean_minutes(e2), min160=mean_minutes(e4),
             min40_seeds=[r["wall_minutes"] for r in e2],
             hw40=hw_cuda_unspec, hw160=hw_cuda_unspec,
             sources40="week2_results/followups/results.json (E2)",
             sources160="week2_results/followups/results.json (E4)"),
        dict(family="CF-WallAware", label="CF-WallAware, hid 16, trained T=8",
             v40=selt(by_model(wa_vt["selections"], "CF-WallAware")),
             v160=vals(wa160_runs),
             T_star40=[s["T_star"] for s in by_model(wa_vt["selections"], "CF-WallAware")],
             T_star160=[r["T_star"] for r in wa160_runs],
             min40=mean_minutes_from_seconds(wa_runs),
             min160=mean_minutes(wa160_runs),
             hw40=f"{wa_m['device_name']} ({wa_m['device']})",
             hw160=hw_cuda_unspec,
             sources40="week2_results/wallaware/valT_selection.json; week2_results/wallaware/week2_metrics.json",
             sources160="week2_results/wallaware160/results.json"),
        dict(family="RecurrentCNN fixed",
             label="RecurrentCNN, fixed T=8, hid 22",
             v40=selt(by_model(vt["selections"], "RecurrentCNN")),
             v160=vals(e6),
             T_star40=[s["T_star"] for s in by_model(vt["selections"], "RecurrentCNN")],
             T_star160=[r["T_star"] for r in e6],
             min40=mean_minutes_from_seconds(wm_rcnn), min160=mean_minutes(e6),
             hw40=hw_3070, hw160=hw_rc,
             sources40="week2_results/valT_selection.json; week2_results/week2_metrics.json",
             sources160="week2_results/rcnn160/results.json (E6)"),
        dict(family="RecurrentCNN variable",
             label="RecurrentCNN, variable T, hid 22",
             v40=vals(e3), v160=vals(e7),
             T_star40=[r["T_star"] for r in e3],
             T_star160=[r["T_star"] for r in e7],
             min40=mean_minutes(e3), min160=mean_minutes(e7),
             hw40=hw_cuda_unspec, hw160=hw_rc,
             sources40="week2_results/followups/results.json (E3)",
             sources160="week2_results/rcnn160/results.json (E7)"),
    ]

    for f in families:
        for e in (40, 160):
            v = f[f"v{e}"]
            assert len(v) >= 1, f["family"]
            f[f"n{e}"] = len(v)
            f[f"m{e}"] = statistics.mean(v)
            f[f"sd{e}"] = statistics.stdev(v) if len(v) > 1 else 0.0
            f[f"preliminary{e}"] = len(v) < 3
        f["gain"] = f["m160"] - f["m40"]
    families.sort(key=lambda f: f["m160"])
    return families


def write_csv(families, out):
    csv_path = out / "data.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "label",
                    "epochs40_seed0", "epochs40_seed1", "epochs40_seed2",
                    "epochs40_mean", "epochs40_sample_sd", "epochs40_n",
                    "T_star40_seed0", "T_star40_seed1", "T_star40_seed2",
                    "epochs160_seed0", "epochs160_seed1", "epochs160_seed2",
                    "epochs160_mean", "epochs160_sample_sd", "epochs160_n",
                    "T_star160_seed0", "T_star160_seed1", "T_star160_seed2",
                    "gain_mean160_minus_mean40",
                    "train_minutes40_mean_per_seed",
                    "train_minutes160_mean_per_seed",
                    "hardware40", "hardware160", "sources40", "sources160"])
        for f in families:
            w.writerow([
                f["family"], f["label"], *f["v40"],
                f"{f['m40']:.6f}", f"{f['sd40']:.6f}", f["n40"],
                *f["T_star40"], *f["v160"],
                f"{f['m160']:.6f}", f"{f['sd160']:.6f}", f["n160"],
                *f["T_star160"], f"{f['gain']:.6f}",
                f"{f['min40']:.3f}", f"{f['min160']:.3f}",
                f["hw40"], f["hw160"], f["sources40"], f["sources160"]])
    return csv_path


def fig_budget(families, out):
    fig, ax = plt.subplots(figsize=(10.2, 6.8), layout="constrained")
    x40, x160 = 0.0, 1.0

    for f in families:
        c = FAMILY_COLORS[f["family"]]
        ax.plot([x40, x160], [f["m40"], f["m160"]], color=c, lw=2.0, zorder=2)
        for x, k in ((x40, "40"), (x160, "160")):
            m, sd = f[f"m{k}"], f[f"sd{k}"]
            ax.errorbar(x, m, yerr=sd, color=c, capsize=4,
                        elinewidth=1.4, capthick=1.4, zorder=3)
            for i, seed_val in enumerate(f[f"v{k}"]):
                n = f[f"n{k}"]
                jitter = (i - (n - 1) / 2.0) * 0.028
                ax.plot(x + jitter, seed_val, SEED_MARKER, color="black",
                        markerfacecolor="white", markersize=5.5,
                        markeredgewidth=1.2, zorder=4)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([x40, x160])
    ax.set_xticklabels(["40", "160"])
    ax.set_xlabel("Training budget (epochs)")
    ax.set_ylabel("Test strict accuracy at T* (fraction of the 1,000-maze test set)")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.suptitle(
        "Four times the epochs (40 \u2192 160) lifts the CF families by "
        "+0.31 to +0.56 test strict accuracy at T*,\nwhile the RecurrentCNN "
        "baselines gain ~0.05 from an already-high 0.94", fontsize=15)

    handles = [Line2D([0], [0], color=FAMILY_COLORS[f["family"]], lw=2.0,
                      label=f["label"]) for f in families]
    handles += [
        Line2D([0], [0], color="#333333", lw=1.4, label="mean \u00b1 sample sd (n\u22121)"),
        Line2D([0], [0], marker=SEED_MARKER, color="black",
               markerfacecolor="white", markersize=6, lw=0,
               label="individual seed (offset in x only)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=11, ncol=2)

    # Key-comparison annotation in the empty region below the CF lines.
    nofrag = next(f for f in families if f["family"] == "CF-NoFrag")
    ax.annotate(
        f"largest gain: CF-NoFrag hid 18\n{nofrag['m40']:.2f} \u2192 "
        f"{nofrag['m160']:.2f} (+{nofrag['gain']:.2f})",
        xy=(0.35, nofrag["m40"] + 0.05 * (nofrag["m160"] - nofrag["m40"])),
        xytext=(0.30, 0.30), fontsize=11, color="#222222",
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2))

    min40 = min(f["min40"] for f in families)
    max40 = max(f["min40"] for f in families)
    min160 = min(f["min160"] for f in families)
    max160 = max(f["min160"] for f in families)
    nofrag = next(f for f in families if f["family"] == "CF-NoFrag")
    seeds40 = sorted(nofrag["min40_seeds"])
    slow, quick = seeds40[-1], seeds40[0]
    note = "\n".join(textwrap.wrap(
        "n = 3 seeds per family and budget; error bars = sample sd (n\u22121); "
        "open dots = individual seeds, offset in x for readability only. Each "
        "point plots test strict accuracy at the validation-selected depth T* "
        "(per-seed argmax of validation strict accuracy, ties to smaller T; "
        "T* values in data.csv). No shading band is drawn; all values are in "
        "[0, 1], so no band clipping applies. Mean wall-clock training time "
        f"per seed: {min40:.1f}\u2013{max40:.1f} minutes at 40 epochs (the "
        "CF-NoFrag hid 18 40-epoch mean is inflated by one slow seed: "
        f"{slow:.1f} minutes for that single seed vs {quick:.2f} for its "
        "fastest, as recorded) and "
        f"{min160:.1f}\u2013{max160:.1f} minutes at 160 epochs (per-family "
        "minutes and recorded hardware in data.csv: NVIDIA GeForce "
        "RTX 3070 Ti (CUDA) where logged).", width=104))
    place_footnote(fig, ax, note, fontsize=10)

    assert_no_text_overlaps(fig, FNAME)
    fig.savefig(out / "training_budget.png", dpi=DPI)
    plt.close(fig)


def main():
    apply_style()
    root = repo_root()
    d = load(root)
    families = build_rows(d)

    out = root / "figures" / "training_budget"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(families, out)
    print(f"Wrote {len(families)} rows -> {csv_path}")
    for f in families:
        print(f"{f['family']:20s} 40: {f['m40']:.4f}\u00b1{f['sd40']:.4f} "
              f"(n={f['n40']}, {f['min40']:.2f} min)  "
              f"160: {f['m160']:.4f}\u00b1{f['sd160']:.4f} "
              f"(n={f['n160']}, {f['min160']:.2f} min)  "
              f"gain=+{f['gain']:.4f}  hw40={f['hw40']} hw160={f['hw160']}")

    fig_budget(families, out)
    print(f"Figure: {out / 'training_budget.png'}")


if __name__ == "__main__":
    main()
