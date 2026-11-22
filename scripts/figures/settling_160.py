"""Settling figure for the 160-epoch models (packet figure F4).

Per-step settling measures of the trained 160-epoch checkpoints, unrolled
for 32 steps on the first 64 level-5 test mazes, 3 seeds each:

  CF-Full hid=16 (run E1, 160 ep, T_train=8)        residual CF flux
  CF-NoFrag hid=18 (run E4, 160 ep, T_train=8)      residual CF flux
  CF-WallAware hid=16 (run E5, 160 ep, T_train=8)   residual CF flux
  RecurrentCNN hid=22 fixed T=8 (run E6, 160 ep)    relative state change
  RecurrentCNN hid=22 variable T (run E7, 160 ep)   relative state change

The settling values are recomputed here from the trained checkpoint files
with cf.train.settling_curve (the exact function that produced the stored
settling curves of the 40-epoch figure); the runs' results.json files are
checked first so every plotted number provably carries its 160-epoch budget.

The script runs on CPU (set CF_ALLOW_LOCAL_GPU=1 to allow CUDA/MPS), reads
the checkpoints from a directory given by --checkpoints-root / $CF_RESULTS_ROOT
(default: <repo-root>/week2_results), and writes

  figures/curriculum_results/settling_160ep.png
  figures/curriculum_results/data.csv   (fig tag "F4 settling 160 ep";
        rows of the other figures are preserved)

Run:  python scripts/figures/settling_160.py [--root REPO_ROOT]
                 [--checkpoints-root DIR] [--out-dir DIR]

The 40-epoch settling figure (settling.png, written by
curriculum_figures.py) remains untouched and serves as the comparison.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

# Force CPU before torch is imported, unless explicitly allowed (the same
# knob the rest of the repository honours).
if os.environ.get("CF_ALLOW_LOCAL_GPU", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # dataset is in the HF cache

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))          # style.py, curriculum_figures.py
sys.path.insert(0, str(_HERE.parents[2]))      # cf package
sys.path.insert(0, str(_HERE.parents[2] / "scripts"))  # run_week2

import torch                                   # noqa: E402  (after env setup)
import matplotlib.pyplot as plt                # noqa: E402

import cf.train as train_mod                  # noqa: E402
import run_week2                              # noqa: E402
from cf.models.cf import CFNet, MODEL_SPECS   # noqa: E402
from cf.models.rcnn import RecurrentCNN       # noqa: E402
from curriculum_figures import (assert_no_text_overlaps, geomean,  # noqa: E402
                                place_footnote)
from style import DPI, OKABE_ITO, SEED_MARKER, apply_style  # noqa: E402

SEEDS = [0, 1, 2]
N_SEEDS = len(SEEDS)
T_MAX = 32          # unrolling steps (same horizon as the stored 40-ep curves)
N_MAZES = 64        # first 64 test mazes (same subset as the 40-ep figure)
F4 = "F4 settling 160 ep"
CSV_FIELDS = ["fig", "series", "x", "mean", "sd", "seed0", "seed1", "seed2"]

# Okabe-Ito colours: the CF-Full / CF-NoFrag / RCNN assignments match the other
# curriculum figures; CF-WallAware (only in this figure) takes orange.
C_CF_FULL = OKABE_ITO["blue"]
C_CF_NOFRAG = OKABE_ITO["vermillion"]
C_CF_WALLAWARE = OKABE_ITO["orange"]
C_RCNN_FIXED = OKABE_ITO["reddish_purple"]
C_RCNN_VAR = OKABE_ITO["bluish_green"]

# (results.json key, checkpoint dir under the checkpoints root, checkpoint
#  file pattern, state-dict factory, colour, linestyle, legend label,
#  data.csv series name)
MODELS = [
    ("E1_CF-Full", "followups", "E1_CF-Full_seed{s}.pt",
     lambda: CFNet(hid=16, **MODEL_SPECS["CF-Full"]),
     C_CF_FULL, "-",
     "CF-Full hid=16: residual flux",
     "CF-Full hid=16 (160 ep, T_train=8)"),
    ("E4_CF-NoFrag160", "followups", "E4_CF-NoFrag160_seed{s}.pt",
     lambda: CFNet(hid=18, **MODEL_SPECS["CF-NoFrag"]),
     C_CF_NOFRAG, "-",
     "CF-NoFrag hid=18: residual flux",
     "CF-NoFrag hid=18 (160 ep, T_train=8)"),
    ("E5_CF-WallAware160", "wallaware160", "E5_CF-WallAware160_seed{s}.pt",
     lambda: CFNet(hid=16, **MODEL_SPECS["CF-WallAware"]),
     C_CF_WALLAWARE, "-",
     "CF-WallAware hid=16: residual flux",
     "CF-WallAware hid=16 (160 ep, T_train=8)"),
    ("E6_RecurrentCNN160", "rcnn160", "E6_RecurrentCNN160_seed{s}.pt",
     lambda: RecurrentCNN(hid=22),
     C_RCNN_FIXED, "--",
     "RecurrentCNN fixed T=8 hid=22: relative state change",
     "RecurrentCNN hid=22 fixed T=8 (160 ep)"),
    ("E7_RecurrentCNN160-Tsamp", "rcnn160", "E7_RecurrentCNN160-Tsamp_seed{s}.pt",
     lambda: RecurrentCNN(hid=22),
     C_RCNN_VAR, "--",
     "RecurrentCNN variable T~U{4..16} hid=22: relative state change",
     "RecurrentCNN hid=22 variable T~U{4..16} (160 ep)"),
]


def parse_args():
    default_root = _HERE.parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=default_root,
                    help="repository root (default: inferred from this "
                         "script's location)")
    ap.add_argument("--checkpoints-root", type=Path, default=None,
                    help="directory containing followups/, wallaware160/ and "
                         "rcnn160/ with the trained 160-epoch checkpoints "
                         "(default: $CF_RESULTS_ROOT or <root>/week2_results)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory for the PNG and data.csv "
                         "(default: <root>/figures/curriculum_results)")
    return ap.parse_args()


def load_curves(ckpt_root):
    """Settling curves for every (model, seed); verifies the 160-epoch budget.

    Returns {series name: {seed: [32 floats]}}.
    """
    train_mod.DEVICE = torch.device("cpu")
    # the split is seeded (rng default_rng(0)) and so is the test set; the
    # first 64 test mazes are the exact mazes of the stored 40-ep curves
    data = run_week2.load_data({})
    test_ds = data["test"]
    print(f"Loaded test split: {len(test_ds)} mazes; settling on the first "
          f"{N_MAZES} unrolled for {T_MAX} steps, CPU")
    results = {}
    for run_key, sub, pattern, factory, *_ in MODELS:
        res_path = ckpt_root / sub / "results.json"
        payload = json.loads(res_path.read_text())
        per_seed = {}
        for seed in SEEDS:
            rec = payload["runs"][f"{run_key}_seed{seed}"]
            if rec["status"] != "done" or rec["epochs_done"] != 160 \
                    or rec["planned_epochs"] != 160:
                raise SystemExit(
                    f"{run_key}_seed{seed}: run is not a completed 160-epoch "
                    f"training (status={rec['status']!r}, "
                    f"epochs_done={rec.get('epochs_done')})")
            model = factory()
            sd = torch.load(str(ckpt_root / sub / pattern.format(s=seed)),
                            map_location="cpu")
            model.load_state_dict(sd, strict=True)
            per_seed[seed] = train_mod.settling_curve(model, test_ds,
                                                      T_MAX, n=N_MAZES)
            print(f"  {run_key}_seed{seed}: step1={per_seed[seed][0]:.6g} "
                  f"step{T_MAX}={per_seed[seed][-1]:.6g}")
        series = next(m[7] for m in MODELS if m[0] == run_key)
        results[series] = per_seed
    return results


def rows_from_curves(curves):
    """data.csv rows (geometric mean per step; no sd band is plotted)."""
    rows = []
    for _, _, _, _, _, _, _, series in MODELS:
        per_seed = curves[series]
        for i in range(T_MAX):
            vals = [per_seed[s][i] for s in SEEDS]
            rows.append(dict(fig=F4, series=series, x=f"step={i + 1}",
                             mean=geomean(vals), sd="",
                             **{f"seed{s_}": per_seed[s][i]
                                for s_, s in enumerate(SEEDS)}))
    return rows


def merge_data_csv(csv_path, rows):
    """Rewrite data.csv = rows of the other figures + the new F4 rows."""
    kept = []
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row["fig"] != F4:
                    kept.append(row)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in kept + rows:
            writer.writerow(row)
    print(f"data.csv: kept {len(kept)} existing rows, wrote {len(rows)} "
          f"{F4!r} rows -> {csv_path}")


def make_figure(curves, out_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    all_values = [v for per_seed in curves.values()
                  for c in per_seed.values() for v in c]
    for (run_key, _, _, _, color, ls, label, series) in MODELS:
        per_seed = curves[series]
        steps = list(range(1, T_MAX + 1))
        for seed in SEEDS:
            ax.plot(steps, per_seed[seed], color=color, alpha=0.45,
                    linewidth=1.0)
        gmeans = [geomean([per_seed[s][i] for s in SEEDS])
                  for i in range(T_MAX)]
        ax.plot(steps, gmeans, ls, color=color, marker=SEED_MARKER,
                markersize=4, label=label)
    # log axis: nothing is drawn below the axis floor (the decade of the
    # smallest plotted value), so no band and nothing is clipped. The y
    # ticks are all the labelled decades that carry data, from the decade
    # of the smallest value up to the decade of the largest; the floor
    # decade itself is skipped because matplotlib would place its label
    # straddling the axis bottom edge.
    floor = 10.0 ** math.floor(math.log10(min(all_values)))
    k_min = math.ceil(math.log10(min(all_values)))
    k_max = math.floor(math.log10(max(all_values)))
    yticks = [10.0 ** k for k in range(k_min, k_max + 1)
              if 10.0 ** k >= floor * 10]
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_yticks(yticks)
    ax.set_xlim(1, T_MAX)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.set_xlabel("Unrolling step t")
    ax.set_ylabel("Settling measure (log)")
    ax.set_title("Settling after 160 epochs: CF-Full and CF-NoFrag settle,\n"
                 "CF-WallAware stalls, fixed-T RecurrentCNN keeps churning",
                 pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    place_footnote(
        fig, ax,
        "How to read: x = unrolling step t (1-%d), y = settling measure, log "
        "scale. Thick = geometric mean over 3 seeds, thin = one seed.\n"
        "CF: residual CF flux; RCNN: relative state change (not comparable). "
        "All: 160 ep, first %d test mazes; T_train=8 except variable-T RCNN "
        "(T~U{4..16})." % (T_MAX, N_MAZES), fontsize=10)
    assert_no_text_overlaps(fig, F4)
    out = out_dir / "settling_160ep.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Figure:", out)
    return out


def main():
    apply_style()
    args = parse_args()
    ckpt_root = (args.checkpoints_root
                 or os.environ.get("CF_RESULTS_ROOT")
                 or args.root / "week2_results")
    out_dir = args.out_dir or args.root / "figures" / "curriculum_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt_root.is_dir():
        raise SystemExit(f"checkpoints root not found: {ckpt_root} "
                         "(pass --checkpoints-root or set CF_RESULTS_ROOT)")
    curves = load_curves(ckpt_root)
    rows = rows_from_curves(curves)
    merge_data_csv(out_dir / "data.csv", rows)
    make_figure(curves, out_dir)
    print("Geometric-mean settling (step 1 -> step %d):" % T_MAX)
    for _, _, _, _, _, _, label, series in MODELS:
        per_seed = curves[series]
        g1 = geomean([per_seed[s][0] for s in SEEDS])
        g32 = geomean([per_seed[s][-1] for s in SEEDS])
        print(f"  {label}: {g1:.6g} -> {g32:.6g} "
              f"(factor {g1 / g32:.1f}x lower)")


if __name__ == "__main__":
    main()
