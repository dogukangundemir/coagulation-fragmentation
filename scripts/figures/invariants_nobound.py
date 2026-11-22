"""Packet figure F5, part 2: CF-Full vs CF-NoBound guardrail invariants at T=64.

Part 1 (figures/mechanism/f5_invariants.png) checks the trained CF-Full and
CF-NoFrag checkpoints against the two pre-registered guardrail tolerances.
This part isolates the 1/2 flux bound (cf/models/cf.py CFNet bound_flux):
CF-Full and CF-NoBound share the same architecture and - built under one
fixed seed - byte-identical initial weights, so any invariant difference
comes from the bound alone. Three runs are rolled out on the first 64
level-5 test mazes to T = 64:

- CF-Full, untrained (random init, seed 0)
- CF-NoBound (bound_flux=False), untrained (random init, seed 0)
- CF-Full, trained 160 epochs at fixed T=8, seed 0
  (week2_results/followups/E1_CF-Full_seed0.pt)

Plotted per step t: min c1 over all mazes / cells / channels (positivity;
NoBound can go negative) and the relative mass drift
max |M_t - M_0| / max(1, max M_0) of the worst maze (mass conservation is
structural and must hold with and without the bound), against the
pre-registered tolerances -1e-7 (positivity) and 1e-5 (mass drift).

Run:  python scripts/figures/invariants_nobound.py
      [--results week2_results] [--out figures/invariants_nobound]
      [--seed 0] [--train-seed 0] [--T 64] [--n-mazes 64]
CPU only (run with CF_ALLOW_LOCAL_GPU=0; every tensor and checkpoint is
loaded on the CPU). Outputs land in figures/invariants_nobound/
(PNG + data.csv with every plotted number).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from style import OKABE_ITO, apply_style          # noqa: E402
import run_week2                                  # noqa: E402
from mechanism import check_overlaps              # noqa: E402

FIGDIR = REPO / "figures" / "invariants_nobound"

TRAIN_EXP = "E1_CF-Full"                    # 160-epoch followup run family
NEG_EPS = -1e-7                             # positivity tolerance
DRIFT_TOL = 1e-5                            # relative mass-drift tolerance
# (both tolerances as in scripts/check_trained_invariants.py / tests)

TOL_COLOR = OKABE_ITO["vermillion"]


# ---------------------------------------------------------------- runs

def build_runs(args, results):
    """The three rolled-out models (untrained pair + trained CF-Full)."""
    runs = []
    for name, key in (("CF-NoBound", "nobound"), ("CF-Full", "full")):
        torch.manual_seed(args.seed)
        model = run_week2.make_model(name).cpu().eval()
        suffix = " (bound_flux=False)" if key == "nobound" else ""
        runs.append(dict(model_name=name, tag="%s_untrained_seed%d" % (name, args.seed),
                         model=model, color=(OKABE_ITO["reddish_purple"] if key == "nobound"
                                             else OKABE_ITO["blue"]),
                         linestyle="--",
                         label="%s, untrained (random init, seed %d)%s"
                               % (name, args.seed, suffix)))

    ts = args.train_seed
    ckpt = results / "followups" / ("%s_seed%d.pt" % (TRAIN_EXP, ts))
    fu = json.load(open(results / "followups" / "results.json", encoding="utf-8"))
    rec = fu["runs"]["%s_seed%d" % (TRAIN_EXP, ts)]
    epochs = rec["epochs_done"]
    T_train = fu["config"].get("T_train", 8)
    model = run_week2.make_model("CF-Full").cpu().eval()
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    runs.append(dict(model_name="CF-Full", tag="CF-Full_trained_%dep_seed%d" % (epochs, ts),
                     model=model.cpu().eval(), color=OKABE_ITO["blue"], linestyle="-",
                     label="CF-Full, trained %d epochs at fixed T=%d, seed %d"
                           % (epochs, T_train, ts),
                     epochs=epochs, T_train=T_train, checkpoint=ckpt.name))
    return runs


def rollout(model, x, T):
    """min c1 per step t = 0..T and relative mass drift per step t = 1..T.

    Drift follows scripts/check_trained_invariants.py positivity_and_mass:
    per maze, |M_t - M_0| is judged against that maze's own mass scale
    max(1, max M_0); the plotted value is the worst maze at each step.
    t = 0 is the reference state (drift 0 by definition, recorded as NaN so
    it is not plotted).
    """
    with torch.no_grad():
        c1, c2 = model.initial_state(x)
        M0 = c1 + 2.0 * c2
        dims = tuple(range(1, M0.dim()))
        scale = M0.amax(dim=dims).clamp(min=1.0)
        min_c1 = [float(c1.min())]
        drift = [float("nan")]
        for _ in range(T):
            c1, c2, _, _, _ = model.step(x, c1, c2)
            min_c1.append(float(c1.min()))
            drift.append(float(((c1 + 2.0 * c2 - M0).abs().amax(dim=dims)
                                / scale).max()))
    return min_c1, drift


# ---------------------------------------------------------------- figure

def make_figure(runs, T, n_mazes, outdir):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.5))
    t_all = np.arange(T + 1)

    # -- panel 1: positivity (min c1 over every maze, cell and channel)
    ax = axes[0]
    for r in runs:
        ax.plot(t_all, r["min_c1"], color=r["color"], linestyle=r["linestyle"])
    ax.set_yscale("symlog", linthresh=1e-7, linscale=4.0)
    ax.set_ylim(-1.2e-1, 3.6e-1)
    # every second decade: adjacent-decade labels would touch on symlog
    ax.set_yticks([-1e-1, -1e-3, 0.0, 1e-3, 1e-1])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(NEG_EPS, color=TOL_COLOR, linestyle="--", linewidth=2)
    ax.text(63, -1.6e-7, "tolerance -1e-7", ha="right", va="top",
            fontsize=11, color=TOL_COLOR)
    ax.set_xlim(-1.0, 65.0)
    ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_xlabel("reaction step t")
    ax.set_ylabel("min c1 (symlog; linear within $\\pm$1e-7)")
    ax.set_title("positivity: min c1 over %d mazes, all cells and channels"
                 % n_mazes, fontsize=13)

    nb = next(r for r in runs if r["model_name"] == "CF-NoBound")
    v_min = min(nb["min_c1"])
    t_min = int(np.argmin(nb["min_c1"]))
    dec_below = np.log10(-v_min / -NEG_EPS)
    # text sits in the empty band between the tolerance line and the trained
    # run's zero curve (axes-fraction coords; the arrow points to the dip)
    ax.annotate("min = %.2e at t = %d (CF-NoBound)\n"
                "%.1f decades below the positivity tolerance"
                % (v_min, t_min, dec_below),
                xy=(t_min, v_min), xytext=(0.10, 0.40),
                textcoords=ax.transAxes, ha="left", va="top", fontsize=11,
                arrowprops=dict(arrowstyle="->", lw=1.4,
                                color=OKABE_ITO["black"]))

    # -- panel 2: mass conservation (relative drift of the worst maze)
    ax = axes[1]
    for r in runs:
        # start at t = 1: t = 0 is the reference state (drift 0 by
        # definition, recorded in data.csv but not plotted)
        ax.plot(t_all[1:], r["drift"][1:], color=r["color"], linestyle=r["linestyle"])
    ax.set_yscale("symlog", linthresh=1e-8, linscale=2.0)
    ax.set_ylim(-1.5e-8, 3e-5)
    ax.set_yticks([0.0, 1e-7, 1e-6, 1e-5])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(DRIFT_TOL, color=TOL_COLOR, linestyle="--", linewidth=2)
    # label below the line at small t, where the curves are still low
    ax.text(13.0, 6.5e-6, "tolerance 1e-5", ha="center", va="top",
            fontsize=11, color=TOL_COLOR)
    ax.set_xlim(-1.0, 65.0)
    ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_xlabel("reaction step t (t = 0 is the drift-free reference)")
    ax.set_ylabel("relative mass drift (symlog; linear within $\\pm$1e-8)")
    ax.set_title("mass conservation: max |M_t - M_0| / max(1, max M_0), "
                 "worst maze", fontsize=13)

    handles = [Line2D([], [], color=r["color"], linestyle=r["linestyle"],
                      label=r["label"]) for r in runs]
    handles.append(Line2D([], [], color=TOL_COLOR, linestyle="--",
                          label="pre-registered tolerance (dashed)"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 0.01))
    worst_drift = max(d for r in runs for d in r["drift"] if d == d)
    fig.suptitle("Without the 1/2 flux bound min c1 goes negative (%.2e at t = %d); "
                 "mass conservation is structural:\nall three runs stay inside the "
                 "1e-5 drift tolerance at T = %d (worst drift %.2e)"
                 % (v_min, t_min, T, worst_drift), y=0.99, va="top", fontsize=15)
    fig.subplots_adjust(top=0.855, bottom=0.155, left=0.065, right=0.985, wspace=0.30)

    path = outdir / "f5_invariants_nobound.png"
    check_overlaps(fig, "F5-nobound")
    fig.savefig(path)
    plt.close(fig)
    return path, v_min, t_min, worst_drift


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(REPO / "week2_results"))
    ap.add_argument("--out", default=str(FIGDIR))
    ap.add_argument("--seed", type=int, default=0,
                    help="init seed of the untrained models (default 0)")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="seed of the trained CF-Full checkpoint (default 0)")
    ap.add_argument("--T", type=int, default=64,
                    help="rollout length in reaction steps (default 64)")
    ap.add_argument("--n-mazes", type=int, default=64,
                    help="first N mazes of the level-5 test split (default 64)")
    args = ap.parse_args(argv)
    results = Path(args.results)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    data = run_week2.load_data(dict())
    x = data["test"].x[:args.n_mazes].cpu()
    runs = build_runs(args, results)

    # the untrained pair differs ONLY in bound_flux: same seed -> same weights
    sd_a = runs[0]["model"].state_dict()
    sd_b = runs[1]["model"].state_dict()
    init_identical = all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)

    for r in runs:
        r["min_c1"], r["drift"] = rollout(r["model"], x, args.T)
        neg = min(r["min_c1"])
        mx = max(d for d in r["drift"] if d == d)
        print("[%s] min c1 = %.6g (t = %d), max drift = %.3g, "
              "positivity_ok = %s, mass_ok = %s"
              % (r["tag"], neg, int(np.argmin(r["min_c1"])), mx,
                 bool(neg >= NEG_EPS), bool(mx < DRIFT_TOL)))

    rows = [("F5_nobound", "config", "T", "%d" % args.T, "rollout steps"),
            ("F5_nobound", "config", "n_mazes", "%d" % args.n_mazes,
             "first N mazes of the level-5 test split"),
            ("F5_nobound", "config", "grid", "%d x %d" % (x.shape[-2], x.shape[-1]),
             "cells"),
            ("F5_nobound", "config", "untrained_seed", "%d" % args.seed,
             "random init of the untrained runs"),
            ("F5_nobound", "config", "device", "cpu", ""),
            ("F5_nobound", "tolerance", "positivity_min_c", "%.1e" % NEG_EPS,
             "as scripts/check_trained_invariants.py"),
            ("F5_nobound", "tolerance", "relative_mass_drift_max", "%.1e" % DRIFT_TOL,
             "as scripts/check_trained_invariants.py"),
            ("F5_nobound", "ablation", "untrained_init_identical", str(init_identical),
             "CF-Full vs CF-NoBound built under the same seed (same weights, "
             "only bound_flux differs)")]
    for r in runs:
        tag = r["tag"]
        if "epochs" in r:
            rows.append(("F5_nobound", tag, "epochs_trained", "%d" % r["epochs"],
                         "checkpoint %s" % r["checkpoint"]))
            rows.append(("F5_nobound", tag, "T_train", "%d" % r["T_train"],
                         "training depth of the checkpoint"))
        neg = min(r["min_c1"])
        mx = max(d for d in r["drift"] if d == d)
        rows.append(("F5_nobound", tag, "min_c1_overall", "%.6g" % neg,
                     "at t = %d" % int(np.argmin(r["min_c1"]))))
        rows.append(("F5_nobound", tag, "positivity_ok", str(bool(neg >= NEG_EPS)),
                     "min c1 >= -1e-7"))
        rows.append(("F5_nobound", tag, "max_relative_mass_drift", "%.6g" % mx,
                     "at t = %d" % int(np.nanargmax(r["drift"]))))
        rows.append(("F5_nobound", tag, "mass_ok", str(bool(mx < DRIFT_TOL)),
                     "max drift < 1e-5"))
        rows.append(("F5_nobound", tag, "final_min_c1", "%.6g" % r["min_c1"][-1],
                     "at t = %d" % args.T))
        rows.append(("F5_nobound", tag, "final_drift", "%.6g" % r["drift"][-1],
                     "at t = %d" % args.T))
        for t, v in enumerate(r["min_c1"]):
            rows.append(("F5_nobound", tag, "min_c1", "%.6g" % v, "t = %d" % t))
        for t in range(1, args.T + 1):
            rows.append(("F5_nobound", tag, "drift", "%.6g" % r["drift"][t],
                         "t = %d" % t))

    png, v_min, t_min, worst_drift = make_figure(runs, args.T, args.n_mazes, outdir)

    csv_path = outdir / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "item", "quantity", "value", "detail"])
        w.writerows(rows)

    print("F5-nobound -> %s" % png)
    print("data -> %s" % csv_path)
    print("key: NoBound min c1 = %.6g at t = %d; worst drift over runs = %.3g" %
          (v_min, t_min, worst_drift))
    return 0


if __name__ == "__main__":
    sys.exit(main())
