"""Main results figure: test strict accuracy at the trained depth (T=8)
AND at the validation-selected depth (T*), side by side for every model,
grouped by training budget (40 vs 160 epochs).

Reads (paths derived from this file's location, or --root override):
  week2_results/valT_selection.json           40-epoch runs, T=8 and T* per seed
  week2_results/week2_metrics.json            40-epoch runs, raw params, minutes
  week2_results/wallaware/valT_selection.json CF-WallAware 40, per-seed values
  week2_results/wallaware/week2_metrics.json  CF-WallAware 40, raw/eff. params
  week2_results/wallaware160/results.json     CF-WallAware 160 epochs
  week2_results/followups/results.json        160-epoch CF and 40-epoch follow-ups
  week2_results/rcnn160/results.json           RecurrentCNN 160 epochs (fixed T
                                              and variable T), 3 seeds each

Writes figures/main_results/{main_results.png, data.csv}.

Optionally adds the CF-families group from an external, read-only results
tree, given on the command line (--families-dir) or via the CF_FAMILIES_DIR
environment variable. The directory is either the cf-families working copy
root (it then must contain results/ladder_train3/results.json) or the
ladder_train3 directory itself. It contributes one row:
  B-WallAware ladder (BD-WallAware), 160 planned epochs per seed,
  n = 3 seeds, test strict at T=8 and at T* (results/ladder_train3).
Without it the figure keeps its previous two-panel content (the script
says so on stdout).

Run:  python scripts/figures/main_results.py [--root REPO_ROOT]
                          [--families-dir DIR | CF_FAMILIES_DIR=DIR]

For every model two paired horizontal bars are drawn: a faded bar for the
trained depth T=8 and a solid bar for the validation-selected depth T*
(T* = argmax val strict over candidate depths, ties to smaller T, per
seed).

The effective-parameter count for CF-NoFrag hid 16 (40 epochs) has no
recorded measurement in week2_results/, so this script runs the project's
own protocol (scripts/run_week2.py:effective_n_params, one training-style
forward+backward on a fixed synthetic curriculum batch, CPU, T=8) and
plots the measured value. With fragmentation=False the split-flux output
head is computed but immediately discarded (milestone1/maze_cf.py), so those entries
receive zero gradient and are not effective.
Each bar carries a sample-sd (n-1) error line and the individual
seed dots; the printed value is "mean +- sample sd". Models with fewer
than 3 seeds would be flagged "(preliminary)" in the label and the CSV;
with the current data every run has n = 3 seeds, so no flag applies.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import DPI, OKABE_ITO, SEED_MARKER, apply_style
from curriculum_figures import assert_no_text_overlaps
from matplotlib.transforms import Bbox

FNAME = "main results (test strict at T=8 and at T*)"

FAMILY_COLORS = {
    "CF-Full": OKABE_ITO["blue"],
    "CF-NoFrag": OKABE_ITO["vermillion"],
    "CF-WallAware": OKABE_ITO["orange"],
    "RecurrentCNN fixed": OKABE_ITO["reddish_purple"],
    "RecurrentCNN variable": OKABE_ITO["bluish_green"],
    "B-WallAware ladder": OKABE_ITO["sky_blue"],
}


def ladder_results_dir(argv=None):
    """Resolve the external CF-families ladder results directory, or None.

    --families-dir / CF_FAMILIES_DIR may point either at the cf-families
    working-copy root (containing results/ladder_train3) or at the
    ladder_train3 directory itself. Returns None when neither is given:
    the figure then keeps its previous two-panel content.
    """
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--families-dir", type=Path, default=None)
    args, _ = p.parse_known_args(argv)
    d = args.families_dir or os.environ.get("CF_FAMILIES_DIR")
    if not d:
        return None
    d = Path(d)
    for cand in (d, d / "results" / "ladder_train3"):
        if (cand / "results.json").is_file():
            return cand
    sys.exit(f"main_results: no results.json under {d} "
             f"(expected results/ladder_train3/results.json)")


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
    data = {k: json.load(p.open()) for k, p in paths.items()}
    return data, paths


def measure_effective_params(root):
    """Measured effective-parameter count for CF-NoFrag hid 16, T=8, CPU.

    The project's own protocol (scripts/run_week2.py:effective_n_params,
    the same one used for counts recorded in followups/results.json):
    one training-style forward+backward (BCEWithLogits) on a fixed
    synthetic curriculum batch, then count the parameter entries with a
    nonzero gradient. Deterministic: fixed torch seed for weight init,
    fixed synthetic-batch seed (seed=0, count=8) as in
    tests/test_effective_params.py.
    """
    import torch

    for sub in ("scripts", "tests"):
        p = str(Path(root) / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from maze_synthetic import make_batch
    from cf.data.curriculum import MazeDataset
    import run_week2

    pairs = make_batch(seed=0, count=8)
    mazes = [a for a, _ in pairs]
    sols = [s for _, s in pairs]
    ds = MazeDataset(list(range(len(mazes))), mazes, sols)
    torch.manual_seed(0)
    model = run_week2.make_model("CF-NoFrag")
    return run_week2.effective_n_params(model, ds.x, ds.y, T=8)


def by_model(selections, model):
    return sorted((s for s in selections if s["model"] == model),
                  key=lambda s: s["seed"])


def fu_by_exp(fu, exp):
    return sorted((r for r in fu["runs"].values() if r["exp"] == exp),
                  key=lambda r: r["seed"])


def rc_by_exp(rc, exp):
    return sorted((r for r in rc["runs"].values() if r["exp"] == exp),
                  key=lambda r: r["seed"])


def mean_minutes_from_seconds(runs):
    return statistics.mean(r["train_seconds"] for r in runs) / 60.0


def mean_minutes(runs):
    return statistics.mean(r["wall_minutes"] for r in runs)


def build_rows(d, eff_nofrag16=None):
    """One row per model variant: T=8 and T* seed values plus metadata.

    Values are read as recorded; nothing is estimated. Training minutes
    are the mean per-seed wall-clock training time from the same source.
    eff_nofrag16 is the measured effective-parameter count for CF-NoFrag
    hid 16 (40 epochs); when omitted it is measured here.
    """
    if eff_nofrag16 is None:
        eff_nofrag16 = measure_effective_params(repo_root())
    vt, wm, wa_vt, wa_m = d["vt"], d["wm"], d["wa_vt"], d["wa_m"]
    wa160, fu, rc = d["wa160"], d["fu"], d["rc"]

    def wm_runs(src, model):
        return sorted((r for r in src["runs"].values()
                       if r["model"] == model), key=lambda r: r["seed"])

    pc = fu["param_counts"]
    hw_3070 = f"{wm['device_name']} ({wm['device']})"
    hw_wa = f"{wa_m['device_name']} ({wa_m['device']})"
    hw_rc = f"{rc['device_name']} ({rc['device']})"
    hw_cuda_unspec = "cuda (GPU model not recorded in results.json)"

    rows = [
        dict(
            key="rcnn_fixed_40", family="RecurrentCNN fixed", epochs=40,
            name="RecurrentCNN, fixed T=8, hid 22\n9,989 raw / 9,989 eff. params",
            t8=[s["test_strict_at_T_train"]
                for s in by_model(vt["selections"], "RecurrentCNN")],
            ts=[s["test_strict_at_T_star"]
                for s in by_model(vt["selections"], "RecurrentCNN")],
            T_star=[s["T_star"] for s in by_model(vt["selections"], "RecurrentCNN")],
            params_raw=9989, params_eff=pc["E3_RecurrentCNN"]["effective"],
            minutes=mean_minutes_from_seconds(wm_runs(wm, "RecurrentCNN")),
            hardware=hw_3070,
            sources="week2_results/valT_selection.json; week2_results/week2_metrics.json; week2_results/followups/results.json (eff. params)"),
        dict(
            key="rcnn_var_40", family="RecurrentCNN variable", epochs=40,
            name="RecurrentCNN, variable T, hid 22\n9,989 raw / 9,989 eff. params",
            t8=[r["test_strict_by_T"]["8"] for r in fu_by_exp(fu, "E3_RecurrentCNN")],
            ts=[r["test_strict_at_T_star"] for r in fu_by_exp(fu, "E3_RecurrentCNN")],
            T_star=[r["T_star"] for r in fu_by_exp(fu, "E3_RecurrentCNN")],
            params_raw=pc["E3_RecurrentCNN"]["raw"],
            params_eff=pc["E3_RecurrentCNN"]["effective"],
            minutes=mean_minutes(fu_by_exp(fu, "E3_RecurrentCNN")),
            hardware=hw_cuda_unspec,
            sources="week2_results/followups/results.json"),
        dict(
            key="wa_40", family="CF-WallAware", epochs=40,
            name="CF-WallAware, hid 16\n10,177 raw / 9,921 eff. params",
            t8=[s["test_strict_at_T_train"]
                for s in by_model(wa_vt["selections"], "CF-WallAware")],
            ts=[s["test_strict_at_T_star"]
                for s in by_model(wa_vt["selections"], "CF-WallAware")],
            T_star=[s["T_star"] for s in by_model(wa_vt["selections"], "CF-WallAware")],
            params_raw=wa_m["runs"]["CF-WallAware_seed0"]["n_params"],
            params_eff=wa_m["runs"]["CF-WallAware_seed0"]["n_params_effective"],
            minutes=mean_minutes_from_seconds(wm_runs(wa_m, "CF-WallAware")),
            hardware=hw_wa,
            sources="week2_results/wallaware/valT_selection.json; week2_results/wallaware/week2_metrics.json"),
        dict(
            key="full_40", family="CF-Full", epochs=40,
            name="CF-Full, hid 16\n10,177 raw / 10,177 eff. params",
            t8=[s["test_strict_at_T_train"]
                for s in by_model(vt["selections"], "CF-Full")],
            ts=[s["test_strict_at_T_star"]
                for s in by_model(vt["selections"], "CF-Full")],
            T_star=[s["T_star"] for s in by_model(vt["selections"], "CF-Full")],
            params_raw=pc["E1_CF-Full"]["raw"],
            params_eff=pc["E1_CF-Full"]["effective"],
            minutes=mean_minutes_from_seconds(wm_runs(wm, "CF-Full")),
            hardware=hw_3070,
            sources="week2_results/valT_selection.json; week2_results/week2_metrics.json; week2_results/followups/results.json (eff. params)"),
        dict(
            key="nofrag_hid18_40", family="CF-NoFrag", epochs=40,
            name="CF-NoFrag, hid 18\n12,745 raw / 9,811 eff. params",
            t8=[r["test_strict_by_T"]["8"] for r in fu_by_exp(fu, "E2_CF-NoFrag")],
            ts=[r["test_strict_at_T_star"] for r in fu_by_exp(fu, "E2_CF-NoFrag")],
            T_star=[r["T_star"] for r in fu_by_exp(fu, "E2_CF-NoFrag")],
            params_raw=pc["E2_CF-NoFrag"]["raw"],
            params_eff=pc["E2_CF-NoFrag"]["effective"],
            minutes=mean_minutes(fu_by_exp(fu, "E2_CF-NoFrag")),
            hardware=hw_cuda_unspec,
            sources="week2_results/followups/results.json"),
        dict(
            key="nofrag_hid16_40", family="CF-NoFrag", epochs=40,
            name=(f"CF-NoFrag, hid 16\n"
                  f"{wm['runs']['CF-NoFrag_seed0']['n_params']:,} raw / "
                  f"{eff_nofrag16:,} eff. params"),
            t8=[s["test_strict_at_T_train"]
                for s in by_model(vt["selections"], "CF-NoFrag")],
            ts=[s["test_strict_at_T_star"]
                for s in by_model(vt["selections"], "CF-NoFrag")],
            T_star=[s["T_star"] for s in by_model(vt["selections"], "CF-NoFrag")],
            params_raw=wm["runs"]["CF-NoFrag_seed0"]["n_params"],
            params_eff=eff_nofrag16,
            minutes=mean_minutes_from_seconds(wm_runs(wm, "CF-NoFrag")),
            hardware=hw_3070,
            sources=("week2_results/valT_selection.json; "
                     "week2_results/week2_metrics.json; eff. params measured "
                     "by scripts/figures/main_results.py via "
                     "scripts/run_week2.py:effective_n_params (CPU, "
                     "synthetic batch, T=8)")),
        dict(
            key="rcnn_fixed_160", family="RecurrentCNN fixed", epochs=160,
            name="RecurrentCNN, fixed T=8, hid 22\n9,989 raw / 9,989 eff. params",
            t8=[r["test_strict_by_T"]["8"] for r in rc_by_exp(rc, "E6_RecurrentCNN160")],
            ts=[r["test_strict_at_T_star"] for r in rc_by_exp(rc, "E6_RecurrentCNN160")],
            T_star=[r["T_star"] for r in rc_by_exp(rc, "E6_RecurrentCNN160")],
            params_raw=rc["param_counts"]["E6_RecurrentCNN160"]["raw"],
            params_eff=rc["param_counts"]["E6_RecurrentCNN160"]["effective"],
            minutes=mean_minutes(rc_by_exp(rc, "E6_RecurrentCNN160")),
            hardware=hw_rc,
            sources="week2_results/rcnn160/results.json"),
        dict(
            key="rcnn_var_160", family="RecurrentCNN variable", epochs=160,
            name="RecurrentCNN, variable T, hid 22\n9,989 raw / 9,989 eff. params",
            t8=[r["test_strict_by_T"]["8"]
                for r in rc_by_exp(rc, "E7_RecurrentCNN160-Tsamp")],
            ts=[r["test_strict_at_T_star"]
                for r in rc_by_exp(rc, "E7_RecurrentCNN160-Tsamp")],
            T_star=[r["T_star"]
                    for r in rc_by_exp(rc, "E7_RecurrentCNN160-Tsamp")],
            params_raw=rc["param_counts"]["E7_RecurrentCNN160-Tsamp"]["raw"],
            params_eff=rc["param_counts"]["E7_RecurrentCNN160-Tsamp"]["effective"],
            minutes=mean_minutes(rc_by_exp(rc, "E7_RecurrentCNN160-Tsamp")),
            hardware=hw_rc,
            sources="week2_results/rcnn160/results.json"),
        dict(
            key="wa_160", family="CF-WallAware", epochs=160,
            name="CF-WallAware, hid 16\n10,177 raw / 9,921 eff. params",
            t8=[r["test_strict_by_T"]["8"]
                for r in sorted(wa160["runs"].values(), key=lambda r: r["seed"])],
            ts=[r["test_strict_at_T_star"]
                for r in sorted(wa160["runs"].values(), key=lambda r: r["seed"])],
            T_star=[r["T_star"]
                    for r in sorted(wa160["runs"].values(), key=lambda r: r["seed"])],
            params_raw=wa160["param_counts"]["E5_CF-WallAware160"]["raw"],
            params_eff=wa160["param_counts"]["E5_CF-WallAware160"]["effective"],
            minutes=mean_minutes(list(wa160["runs"].values())),
            hardware="cuda (GPU model not recorded in results.json)",
            sources="week2_results/wallaware160/results.json"),
        dict(
            key="full_160", family="CF-Full", epochs=160,
            name="CF-Full, hid 16\n10,177 raw / 10,177 eff. params",
            t8=[r["test_strict_by_T"]["8"] for r in fu_by_exp(fu, "E1_CF-Full")],
            ts=[r["test_strict_at_T_star"] for r in fu_by_exp(fu, "E1_CF-Full")],
            T_star=[r["T_star"] for r in fu_by_exp(fu, "E1_CF-Full")],
            params_raw=pc["E1_CF-Full"]["raw"],
            params_eff=pc["E1_CF-Full"]["effective"],
            minutes=mean_minutes(fu_by_exp(fu, "E1_CF-Full")),
            hardware=hw_cuda_unspec,
            sources="week2_results/followups/results.json"),
        dict(
            key="nofrag_hid18_160", family="CF-NoFrag", epochs=160,
            name="CF-NoFrag, hid 18\n12,745 raw / 9,811 eff. params",
            t8=[r["test_strict_by_T"]["8"] for r in fu_by_exp(fu, "E4_CF-NoFrag160")],
            ts=[r["test_strict_at_T_star"] for r in fu_by_exp(fu, "E4_CF-NoFrag160")],
            T_star=[r["T_star"] for r in fu_by_exp(fu, "E4_CF-NoFrag160")],
            params_raw=pc["E4_CF-NoFrag160"]["raw"],
            params_eff=pc["E4_CF-NoFrag160"]["effective"],
            minutes=mean_minutes(fu_by_exp(fu, "E4_CF-NoFrag160")),
            hardware=hw_cuda_unspec,
            sources="week2_results/followups/results.json"),
    ]

    # Deterministic ordering used by the figure and the CSV (within each
    # budget, ascending at T*, so the best model sits at the top row).
    rows.sort(key=lambda r: (r["epochs"], statistics.mean(r["ts"])))
    for row in rows:
        finalize_row(row, group="budget40" if row["epochs"] == 40 else "budget160")
    return rows


def finalize_row(row, group):
    n = len(row["ts"])
    assert n == len(row["t8"]) == len(row["T_star"]), row["key"]
    row["group"] = group
    row["n"] = n
    row["preliminary"] = n < 3
    row["t8_mean"] = statistics.mean(row["t8"])
    row["t8_sd"] = statistics.stdev(row["t8"]) if n > 1 else 0.0
    row["ts_mean"] = statistics.mean(row["ts"])
    row["ts_sd"] = statistics.stdev(row["ts"]) if n > 1 else 0.0
    if row["preliminary"]:
        row["name"] += f"\n(preliminary, n = {n})"


def load_ladder_row(ladder_dir):
    """Row for the CF-families group: B-WallAware ladder (BD-WallAware),
    160 planned epochs per seed, seeds 0/1/2, from the external read-only
    results tree (results/ladder_train3). Every value is read as recorded
    from results.json plus the per-seed ladder_metrics.json sidecars.

    Wall minutes are cumulative across the seeds' two training sessions
    (the runs were resumed from checkpoints); seed 2's prior-session
    minutes are an estimate from the per-epoch rate of seeds 0/1 (flagged
    in the payload, recorded here in the sources string).
    """
    res = json.loads((ladder_dir / "results.json").read_text())
    per_seed = sorted(res["per_seed"], key=lambda r: r["seed"])
    assert res["model"] == "BD-WallAware", res["model"]
    assert res["label"].startswith("planned budget"), res["label"]
    cfg = res["config"]
    assert len(per_seed) == 3 and sorted(res["seeds"]) == [0, 1, 2]
    for r in per_seed:
        assert r["epochs_done"] == cfg["epochs_planned"], r
        assert not r["stopped_early"], r
    metrics = []
    for r in per_seed:
        m = json.loads((ladder_dir / f"seed{r['seed']}"
                        / "ladder_metrics.json").read_text())
        assert m["config"]["model"] == "BD-WallAware"
        assert m["training"]["epochs_done"] == cfg["epochs_planned"]
        metrics.append(m)
    raw = {m["params"]["raw"] for m in metrics}
    eff = {m["params"]["effective"] for m in metrics}
    assert len(raw) == len(eff) == 1, "params differ between seeds"
    hw = {(m["hardware"]["device"], m["hardware"]["device_name"])
          for m in metrics}
    assert len(hw) == 1, "hardware differs between seeds"
    device, device_name = hw.pop()
    est = [r["wall_minutes_prev_estimated"] for r in per_seed]
    note = (""
            if not any(est)
            else " (seed %d's prior-session minutes estimated from the "
                 "per-epoch rate of the other seeds)" % per_seed[est.index(True)]["seed"])
    return dict(
        key="wladder_160", family="B-WallAware ladder", epochs=160,
        name=("B-WallAware ladder (BD-WallAware), hid 16, K 4 rungs\n"
              f"{raw.pop():,} raw / {eff.pop():,} eff. params"),
        t8=[r["strict_T_train"] for r in per_seed],
        ts=[r["strict_T_star"] for r in per_seed],
        T_star=[r["T_star"] for r in per_seed],
        params_raw=metrics[0]["params"]["raw"],
        params_eff=metrics[0]["params"]["effective"],
        minutes=statistics.mean(r["wall_minutes"] for r in per_seed),
        hardware=f"{device_name} ({device})",
        sources=("external cf-families results tree, read-only, "
                 "via --families-dir or CF_FAMILIES_DIR: "
                 "results/ladder_train3/results.json (per-seed strict at "
                 "T_train/T*, T*, cumulative wall minutes) and "
                 "seed*/ladder_metrics.json (params, hardware, epochs); "
                 f"wall minutes cumulative across two sessions{note}"),
    )


def write_csv(rows, out):
    csv_path = out / "data.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "family", "group", "epochs", "n_seeds",
                    "preliminary",
                    "params_raw", "params_effective",
                    "test_strict_T8_seed0", "test_strict_T8_seed1",
                    "test_strict_T8_seed2", "T8_mean", "T8_sample_sd",
                    "test_strict_Tstar_seed0", "test_strict_Tstar_seed1",
                    "test_strict_Tstar_seed2", "Tstar_mean", "Tstar_sample_sd",
                    "T_star_seed0", "T_star_seed1", "T_star_seed2",
                    "train_minutes_mean_per_seed", "hardware", "sources"])
        for r in rows:
            w.writerow([
                r["name"].split("\n")[0], r["family"], r["group"],
                r["epochs"], r["n"],
                int(r["preliminary"]), r["params_raw"], r["params_eff"],
                *r["t8"], f"{r['t8_mean']:.6f}", f"{r['t8_sd']:.6f}",
                *r["ts"], f"{r['ts_mean']:.6f}", f"{r['ts_sd']:.6f}",
                *r["T_star"], f"{r['minutes']:.3f}", r["hardware"],
                r["sources"]])
    return csv_path


def draw_model_row(ax, row, yi, offsets):
    """One model row: faded T=8 bar (lower) and solid T* bar (upper)."""
    color = FAMILY_COLORS[row["family"]]
    bars = [("ts", row["ts"], row["ts_mean"], row["ts_sd"], yi + 0.17, 1.0),
            ("t8", row["t8"], row["t8_mean"], row["t8_sd"], yi - 0.17, 0.42)]
    for _key, vals, mean, sd, yc, alpha in bars:
        ax.barh([yc], [mean], height=0.30, color=color, alpha=alpha,
                zorder=3 if alpha == 1.0 else 2)
        # Sample-sd error line with caps (plain visible lines: barh(xerr=...)
        # hides the caps inside one invisible Line2D artist).
        lo, hi = mean - sd, mean + sd
        ax.plot([lo, hi], [yc, yc], color="#333333", lw=1.4, zorder=4)
        cap = 0.06
        for x in (lo, hi):
            ax.plot([x, x], [yc - cap, yc + cap], color="#333333",
                    lw=1.4, zorder=4)
        for val, off in zip(vals, offsets):
            ax.plot(val, yc + off, SEED_MARKER, color="black",
                    markerfacecolor="white", markersize=4.5,
                    markeredgewidth=1.1, zorder=5)
        # Value label right of the error line and of any seed dot.
        lab_x = max(hi, max(vals)) + 0.018
        ax.text(lab_x, yc, f"{mean:.3f} \u00b1 {sd:.3f}",
                va="center", ha="left", fontsize=10, color="#222222")


def fig_main(rows, out):
    left = [r for r in rows if r["group"] == "budget40"]
    mid = [r for r in rows if r["group"] == "budget160"]
    fam = [r for r in rows if r["group"] == "families"]
    panels = [(left, "Trained 40 epochs"),
              (mid, "Trained 160 epochs")]
    if fam:
        panels.append((fam, "CF families: ladder, 160 epochs"))

    fig, axes = plt.subplots(
        1, len(panels), figsize=(7.3 * len(panels) - 0.4, 7.9),
        layout="constrained", sharex=True)
    nrow_max = max(len(left), len(mid))
    for ax, (group, title) in zip(axes, panels):
        ys = list(range(len(group)))
        for yi, row in zip(ys, group):
            draw_model_row(ax, row, yi, offsets=[-0.055, 0.0, 0.055])
        ax.set_yticks(ys)
        ax.set_yticklabels([r["name"] for r in group], fontsize=10.5)
        if len(group) == 1:
            ax.set_ylim(-0.75, 0.75)
        else:
            ax.set_ylim(-0.75, nrow_max - 0.25)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Test strict accuracy (fraction of the 1,000-maze test set)")
        ax.set_title(title, fontsize=13)

    ladder = fam[0] if fam else None
    if ladder is not None:
        t_stars = ", ".join(str(t) for t in ladder["T_star"])
        suptitle = (
            "At the trained depth T=8 and the validation-selected depth T*: "
            "160 epochs put every in-project model at 0.92+ at T* "
            "(their CF rows need T* = 12\u201316 and stay \u2264 0.91 at T=8),\n"
            f"while the CF-families B-WallAware ladder at the same 160-epoch "
            f"budget reaches {ladder['ts_mean']:.3f} \u00b1 {ladder['ts_sd']:.3f} "
            f"at T* (depths {t_stars}) vs {ladder['t8_mean']:.3f} \u00b1 "
            f"{ladder['t8_sd']:.3f} at T=8")
    else:
        suptitle = (
            "At the trained depth T=8 and the validation-selected depth T*: "
            "a 160-epoch budget puts every family at 0.92+ at T*,\n"
            "while the CF models peak only beyond their trained depth "
            "(T* = 12\u201316, and at T=8 they stay \u2264 0.91)")
    fig.suptitle(suptitle, fontsize=15)

    handles = [
        Patch(facecolor="#777777", alpha=1.0,
             label="validation-selected depth T*"),
        Patch(facecolor="#777777", alpha=0.42, label="trained depth T = 8"),
        Line2D([0], [0], color="#333333", lw=1.4, label="sample sd (n\u22121)"),
        Line2D([0], [0], marker=SEED_MARKER, color="black",
               markerfacecolor="white", markersize=6, lw=0,
               label="individual seed"),
    ]
    axes[0].legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, -0.16), fontsize=11, ncol=2)

    ladder_note = ""
    if ladder is not None:
        ladder_note = (
            " Right panel: CF-families group, B-WallAware ladder "
            "(BD-WallAware, hid 16, K 4 rungs), 160 planned epochs per seed, "
            "read from the external read-only cf-families results tree via "
            "--families-dir / CF_FAMILIES_DIR, same level-5 curriculum "
            "splits (4000/1000/1000) and strict metric; its training "
            "minutes are cumulative across two training sessions (seed 2's "
            "prior-session minutes an estimate); its T* candidates are "
            "{1, 2, 4, 8, 12, 16, 24, 32}.")
    footnote = "\n".join(textwrap.wrap(
        "How to read: one row per model, solid bar (upper) = test strict at "
        "the validation-selected depth T*, faded bar (lower) = at the "
        "trained depth T=8; dark line = sample sd (n\u22121), open dots = "
        "the three seeds; panels group the in-project models by training "
        "budget (40 epochs left, 160 epochs middle)." + ladder_note +
        " Every CF model and the fixed-T RecurrentCNN trained at depth "
        "T=8; the variable-T RecurrentCNN trained with per-batch sampled T "
        "in 4\u201316. T* = per-seed argmax of validation strict over "
        "candidate depths (ties to smaller T). No shading band is drawn; "
        "all values are in [0, 1], so no band clipping applies. Hardware "
        "and per-seed training minutes in data.csv.",
        width=165 if ladder is not None else 118))
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(ren) for ax in axes]
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            boxes.append(leg.get_window_extent(ren))
    bb = Bbox.union(boxes)
    x0, y0 = fig.transFigure.inverted().transform((bb.x0, bb.y0))
    fig.text(x0, y0 - 10 / (fig.get_size_inches()[1] * fig.dpi), footnote,
             fontsize=10, color="#444444", ha="left", va="top")

    assert_no_text_overlaps(fig, FNAME)
    fig.savefig(out / "main_results.png", dpi=DPI)
    plt.close(fig)


def main():
    apply_style()
    root = repo_root()
    ladder_dir = ladder_results_dir()
    d, paths = load(root)
    eff_nofrag16 = measure_effective_params(root)
    print(f"Measured CF-NoFrag hid 16: {eff_nofrag16} effective params "
          f"(run_week2.effective_n_params, one forward+backward on a "
          f"synthetic training batch, CPU, T=8)")
    rows = build_rows(d, eff_nofrag16)
    if ladder_dir is None:
        print("No --families-dir / CF_FAMILIES_DIR given: the CF-families "
              "group is omitted, the figure keeps its two budget panels.")
    else:
        ladder = load_ladder_row(ladder_dir)
        finalize_row(ladder, group="families")
        rows.append(ladder)

    out = root / "figures" / "main_results"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(rows, out)
    print(f"Wrote {len(rows)} rows -> {csv_path}")
    for r in rows:
        print(f"{r['key']:20s} group={r['group']:9s} epochs={r['epochs']:3d} "
              f"n={r['n']} "
              f"T8={r['t8_mean']:.4f}\u00b1{r['t8_sd']:.4f} "
              f"T*={r['ts_mean']:.4f}\u00b1{r['ts_sd']:.4f} "
              f"T*depths={r['T_star']} "
              f"minutes={r['minutes']:.2f} hw={r['hardware']}")

    fig_main(rows, out)
    print(f"Figure: {out / 'main_results.png'}")


if __name__ == "__main__":
    main()
