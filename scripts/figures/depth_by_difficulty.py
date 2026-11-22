"""Packet figure F8: strict accuracy vs test-time horizon T, one panel per
difficulty (level-5 test, probe level 7, probe level 9), T_train marked.

Models (plain names, training settings in the legend):
  CF-Full hid=16, 160 epochs, trained at fixed T=8          (followups E1)
  CF-NoFrag hid=18, 160 epochs, trained at fixed T=8         (followups E4)
  RecurrentCNN hid=22, 40 epochs, trained at fixed T=8
  RecurrentCNN hid=22, 40 epochs, trained at variable T      (followups E3)

The test-time sweeps (T = 1, 2, 4, 8, 12, 16, 24, 32) are computed on CPU
from the saved checkpoints with cf.train.sweep_model (one rollout per batch,
scored with the repo's evaluate_mask/summarize path) and cached to
week2_results/probe_sweeps.json; a re-run skips everything already cached.

Run:  python scripts/figures/depth_by_difficulty.py [--root REPO_ROOT]

Writes figures/depth_by_difficulty/{depth_by_difficulty.png,
depth_by_difficulty_per_seed.png, data.csv, per_seed_tally.csv}.

The per-seed companion figure shows every seed's own curve (thin lines) plus
the 3-seed mean per model and difficulty; per_seed_tally.csv carries the
"keeps improving" tally used in the README: a seed keeps improving when its
strict accuracy at T=32 is at least its strict accuracy at T_train = 8.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (cf package)
sys.path.insert(0, str(Path(__file__).resolve().parent))      # style / helpers

from cf.data.curriculum import build_datasets, load_curriculum, make_splits
from cf.models.cf import CFNet, MODEL_SPECS
from cf.models.rcnn import RecurrentCNN
import cf.train as train_mod

from curriculum_figures import (assert_no_text_overlaps, mean_sd, place_footnote,
                               repo_root)
from style import (DPI, OKABE_ITO, SEED_MARKER, apply_style)

SWEEP_T = [1, 2, 4, 8, 12, 16, 24, 32]
SEEDS = [0, 1, 2]
N_SEEDS = len(SEEDS)
T_TRAIN = 8
MODELS = ["CF-Full", "CF-NoFrag", "RecurrentCNN-fixed", "RecurrentCNN-variable"]
F8 = "F8 depth by difficulty"
F8S = "F8 per-seed depth by difficulty"

C_CF_FULL = OKABE_ITO["blue"]
C_CF_NOFRAG = OKABE_ITO["vermillion"]
C_RCNN_FIXED = OKABE_ITO["reddish_purple"]
C_RCNN_VAR = OKABE_ITO["bluish_green"]
C_REF = OKABE_ITO["grey"]

DATASETS = ["test", "probe7", "probe9"]
PANEL_TITLES = {"test": "Level-5 test", "probe7": "Probe level 7", "probe9": "Probe level 9"}


# ----------------------------------------------------------------------------
# Checkpoints and sweep cache
# ----------------------------------------------------------------------------
def model_specs(root):
    """(key -> ckpt template, constructor, epochs, train-T description), with
    epoch counts read from the results JSONs (never hard-coded)."""
    wm = json.loads((root / "week2_results" / "week2_metrics.json").read_text())
    fu = json.loads((root / "week2_results" / "followups" / "results.json").read_text())

    def epochs_of(runs, tag):
        return runs[tag]["epochs_done"]

    specs = {
        "CF-Full": dict(
            ckpt="week2_results/followups/E1_CF-Full_seed{s}.pt",
            make=lambda: CFNet(hid=16, **MODEL_SPECS["CF-Full"]),
            epochs=epochs_of(fu["runs"], "E1_CF-Full_seed0"),
            train_T="trained at fixed T=8", color=C_CF_FULL),
        "CF-NoFrag": dict(
            ckpt="week2_results/followups/E4_CF-NoFrag160_seed{s}.pt",
            make=lambda: CFNet(hid=18, **MODEL_SPECS["CF-NoFrag"]),
            epochs=epochs_of(fu["runs"], "E4_CF-NoFrag160_seed0"),
            train_T="trained at fixed T=8", color=C_CF_NOFRAG),
        "RecurrentCNN-fixed": dict(
            ckpt="week2_results/RecurrentCNN_seed{s}.pt",
            make=lambda: RecurrentCNN(hid=22),
            epochs=epochs_of(wm["runs"], "RecurrentCNN_seed0"),
            train_T="trained at fixed T=8", color=C_RCNN_FIXED),
        "RecurrentCNN-variable": dict(
            ckpt="week2_results/followups/E3_RecurrentCNN_seed{s}.pt",
            make=lambda: RecurrentCNN(hid=22),
            epochs=epochs_of(fu["runs"], "E3_RecurrentCNN_seed0"),
            train_T="trained at variable T (4-16 per batch)", color=C_RCNN_VAR),
    }
    # every seed must have finished the same number of epochs
    for key, tag in [("CF-Full", "E1_CF-Full"), ("CF-NoFrag", "E4_CF-NoFrag160"),
                     ("RecurrentCNN-variable", "E3_RecurrentCNN")]:
        eps = [fu["runs"][f"{tag}_seed{s}"]["epochs_done"] for s in SEEDS]
        assert len(set(eps)) == 1 and eps[0] == specs[key]["epochs"], (key, eps)
    eps = [wm["runs"][f"RecurrentCNN_seed{s}"]["epochs_done"] for s in SEEDS]
    assert len(set(eps)) == 1 and eps[0] == specs["RecurrentCNN-fixed"]["epochs"], eps
    return specs


def load_data():
    """Curriculum pipeline: level-5 train/val/test, probe levels 7 and 9
    (fixed seeded splits, numpy default_rng(0))."""
    mazes, sols, levels = load_curriculum()
    splits, *_ = make_splits(mazes, sols, levels)
    ds = build_datasets(splits, mazes, sols)
    return {"test": ds["test"], "probe7": ds["probe"][7], "probe9": ds["probe"][9]}


def compute_sweeps(root, specs, data, cache_path):
    """Sweep every (model, seed, dataset) not yet in the cache; strict accuracy
    per T via cf.train.sweep_model on CPU."""
    cache = {"config": {"Ts": SWEEP_T, "seeds": SEEDS, "threshold": 0.5,
                        "metric": "strict (exact shortest path)",
                        "datasets": {k: len(v) for k, v in data.items()}},
             "sweeps": {}}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        assert cache["config"]["Ts"] == SWEEP_T and cache["config"]["seeds"] == SEEDS, \
            "existing probe_sweeps.json was written for a different sweep grid"
    sweeps = cache["sweeps"]
    t0 = time.time()
    for key, spec in specs.items():
        sweeps.setdefault(key, {})
        for seed in SEEDS:
            sweeps[key].setdefault(str(seed), {})
            for name in DATASETS:
                if name in sweeps[key][str(seed)]:
                    continue
                model = spec["make"]()
                model.load_state_dict(torch.load(
                    str(root / spec["ckpt"].format(s=seed)), map_location="cpu"))
                summary = train_mod.sweep_model(model, data[name], SWEEP_T)
                sweeps[key][str(seed)][name] = {str(T): summary[T]["strict"]
                                                for T in SWEEP_T}
                print(f"  swept {key} seed {seed} {name} "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def shared_x_axis(fig, ax):
    """x-axis styling shared by the mean and per-seed figures: linear in T.
    On a linear axis the T=1 and T=2 positions are too close for
    side-by-side tick labels, so those two are staggered onto two rows at
    their true x positions."""
    ax.set_xticks(SWEEP_T)
    seq = [str(T) if T not in (1, 2) else "" for T in SWEEP_T]
    ax.set_xticklabels(seq)
    tr = ax.get_xaxis_transform()
    row1 = mtransforms.offset_copy(tr, fig=fig, x=0.0, y=-4.0, units="points")
    row2 = mtransforms.offset_copy(tr, fig=fig, x=0.0, y=-18.0, units="points")
    ax.text(1, 0, "1", ha="center", va="top", fontsize=12, transform=row1)
    ax.text(2, 0, "2", ha="center", va="top", fontsize=12, transform=row2)
    ax.set_xlim(-0.5, SWEEP_T[-1] + 1.5)
    ax.set_xlabel("Test-time horizon T", labelpad=34)


def legend_label(key, spec):
    """Legend text: plain model name plus its full training settings."""
    name = "RecurrentCNN (fixed T)" if key == "RecurrentCNN-fixed" else key
    return f"{name}, {spec['epochs']} epochs, {spec['train_T']}"


def keeps_improvement(cache, key, panel, seed):
    """A seed keeps improving beyond the training horizon when its strict
    accuracy at T=32 is at least its strict accuracy at T_train = 8."""
    d = cache["sweeps"][key][str(seed)][panel]
    return d["32"] >= d[str(T_TRAIN)]

def fig8(cache, specs, data, out):
    sweeps = cache["sweeps"]
    xs = list(SWEEP_T)                   # x-axis linear in T
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.3), sharey=True)
    handles = []
    for ax, name in zip(axes, DATASETS):
        grid = data[name].x.shape[-1]
        for key in MODELS:
            spec = specs[key]
            means, sds = [], []
            for T in SWEEP_T:
                vals = [sweeps[key][str(s)][name][str(T)] for s in SEEDS]
                m, sd = mean_sd(vals)
                means.append(m)
                sds.append(sd)
            means, sds = np.array(means), np.array(sds)
            label = legend_label(key, spec)
            line, = ax.plot(xs, means, color=spec["color"],
                            marker=SEED_MARKER, markersize=4.5, label=label)
            ax.fill_between(xs, np.clip(means - sds, 0.0, 1.0),
                            np.clip(means + sds, 0.0, 1.0),
                            color=spec["color"], alpha=0.18, linewidth=0)
            if name == "test":
                handles.append(line)
        vline = ax.axvline(T_TRAIN, color=C_REF, linestyle="--", linewidth=1.4,
                           zorder=1)
        if name == "test":
            vline.set_label(f"Training horizon T = {T_TRAIN} (fixed-T models only)")
            handles.append(vline)
        shared_x_axis(fig, ax)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        n = len(data[name])
        ax.set_title(f"{PANEL_TITLES[name]} ({grid}x{grid}, n={n})", pad=12)
    axes[0].set_ylabel("Strict accuracy (exact path)")
    fig.suptitle("Thinking longer: strict accuracy vs test-time horizon T, one panel per difficulty",
                 fontsize=15)
    # explicit layout (the deep label stack below the axes is too tall for
    # the tight_layout solver), then the legend is placed by measurement a
    # few px below the axes' tight bbox (x labels included)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.87, bottom=0.16,
                        wspace=0.24)
    fig.canvas.draw()
    bb = axes[1].get_tightbbox(fig.canvas.get_renderer())
    leg = axes[1].legend(handles=handles, loc="upper center", ncol=2,
                         fontsize=10, bbox_transform=fig.transFigure,
                         bbox_to_anchor=(0.5, (bb.y0 - 6) / fig.bbox.y1))
    place_footnote(fig, axes[1],
                   f"n = {N_SEEDS} seeds; lines = mean strict accuracy, band = mean \u00b1 sample sd "
                   "over seeds, clipped to [0, 1].\n"
                   f"Vertical dashed line: training horizon T = {T_TRAIN} of the fixed-T models; "
                   "the variable-T RecurrentCNN was trained with T ~ U{4..16} per batch.\n"
                   "Threshold 0.5 on sigmoid probabilities; strict = prediction is exactly the "
                   "BFS shortest path (perfect mazes).\n"
                   "Sweeps computed on CPU from the saved checkpoints with cf.train.sweep_model; "
                   "cache: probe_sweeps.json.")
    assert_no_text_overlaps(fig, F8)
    fig.savefig(out / "depth_by_difficulty.png", dpi=DPI)
    plt.close(fig)


def fig8_per_seed(cache, specs, data, out):
    """Per-seed companion of F8: every seed's own curve (thin lines) plus the
    3-seed mean (bold), per model and difficulty panel. The seed lines show
    the spread directly, so no sd band is drawn (stated in the footnote)."""
    sweeps = cache["sweeps"]
    xs = list(SWEEP_T)
    cf_keep = sum(keeps_improvement(cache, m, p, s)
                  for m in ["CF-Full", "CF-NoFrag"]
                  for p in DATASETS for s in SEEDS)
    rcnn_fixed_keep = sum(keeps_improvement(cache, "RecurrentCNN-fixed", p, s)
                          for p in DATASETS for s in SEEDS)
    n_cf = 2 * len(DATASETS) * N_SEEDS

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.8), sharey=True)
    handles = []
    proxy = plt.Line2D([], [], color=C_REF, alpha=0.55, linewidth=1.0,
                       label="individual seed (thin line)")
    for ax, name in zip(axes, DATASETS):
        grid = data[name].x.shape[-1]
        for key in MODELS:
            spec = specs[key]
            for s in SEEDS:
                ys = [sweeps[key][str(s)][name][str(T)] for T in SWEEP_T]
                ax.plot(xs, ys, color=spec["color"], alpha=0.55,
                        linewidth=1.0, zorder=2)
            means = [mean_sd([sweeps[key][str(s)][name][str(T)]
                             for s in SEEDS])[0] for T in SWEEP_T]
            line, = ax.plot(xs, means, color=spec["color"],
                            marker=SEED_MARKER, markersize=4.5,
                            linewidth=2.2, zorder=4,
                            label=legend_label(key, spec))
            if name == "test":
                handles.append(line)
        vline = ax.axvline(T_TRAIN, color=C_REF, linestyle="--", linewidth=1.4,
                           zorder=1)
        if name == "test":
            vline.set_label(f"Training horizon T = {T_TRAIN} (fixed-T models only)")
            handles.append(vline)
        shared_x_axis(fig, ax)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        n = len(data[name])
        ax.set_title(f"{PANEL_TITLES[name]} ({grid}x{grid}, n={n})", pad=12)
    axes[0].set_ylabel("Strict accuracy (exact path)")
    # title states the finding; the two guide lines under it say how to read
    # the figure. Counts come from the sweeps themselves (keeps-improving
    # rule: strict at T=32 >= strict at T_train = 8).
    fig.suptitle(f"At T=32 the CF models hold or beat their T=8 accuracy in "
                 f"{cf_keep} of {n_cf} seed-panel curves, the fixed-T "
                 f"RecurrentCNN in {rcnn_fixed_keep} of {len(DATASETS) * N_SEEDS}",
                 fontsize=15, x=0.05, y=0.985, ha="left", va="top")
    fig.text(0.05, 0.925,
             "Reading guide: one colour per model (legend); thin lines are the "
             "individual seeds, the bold line with dots is their mean.\n"
             "Right of the dashed line is untested depth; a seed \"keeps "
             "improving\" when its strict accuracy at T=32 is at least its "
             f"accuracy at the training horizon T={T_TRAIN}.",
             fontsize=10.5, ha="left", va="top", color="#333333")
    # explicit layout (same deep label stack as F8), then the legend is
    # placed by measurement a few px below the axes' tight bbox
    fig.subplots_adjust(left=0.05, right=0.985, top=0.78, bottom=0.16,
                        wspace=0.24)
    fig.canvas.draw()
    bb = axes[1].get_tightbbox(fig.canvas.get_renderer())
    leg = axes[1].legend(handles=handles + [proxy], loc="upper center", ncol=3,
                         fontsize=10, bbox_transform=fig.transFigure,
                         bbox_to_anchor=(0.5, (bb.y0 - 6) / fig.bbox.y1))
    place_footnote(fig, axes[1],
                   f"n = {N_SEEDS} seeds per model (seeds 0, 1, 2); thin lines = strict accuracy "
                   "of each seed, bold line with dots = mean over seeds; no sd band is drawn "
                   "because the spread is shown directly by the seed lines.\n"
                   f"Vertical dashed line: training horizon T = {T_TRAIN} of the fixed-T models; "
                   "the variable-T RecurrentCNN was trained with T ~ U{4..16} per batch.\n"
                   "Threshold 0.5 on sigmoid probabilities; strict = prediction is exactly the "
                   "BFS shortest path (perfect mazes).\n"
                   "Sweeps computed on CPU from the saved checkpoints with cf.train.sweep_model; "
                   "cache: probe_sweeps.json. "
                   "Tally per model and panel in per_seed_tally.csv.")
    assert_no_text_overlaps(fig, F8S)
    fig.savefig(out / "depth_by_difficulty_per_seed.png", dpi=DPI)
    plt.close(fig)


def write_tally(cache, out):
    """"keeps improving" tally for the README, per (model, panel): one row per
    seed (strict accuracy at T_train = 8 and at T=32, plus the 0/1 verdict)
    and one aggregate row per (model, panel) counting the seeds that keep
    improving. Rule: a seed keeps improving when its strict accuracy at T=32
    is at least its strict accuracy at T_train = 8."""
    rows = []
    for name in DATASETS:
        for key in MODELS:
            for s in SEEDS:
                d = cache["sweeps"][key][str(s)][name]
                v_train, v32 = d[str(T_TRAIN)], d["32"]
                rows.append(dict(model=key, panel=name,
                                 seed=f"seed{s} (n=1 run)",
                                 strict_at_T_train=f"{v_train:.4f}",
                                 strict_at_T32=f"{v32:.4f}",
                                 keeps_improving=int(keeps_improvement(cache, key, name, s))))
            keep_n = sum(keeps_improvement(cache, key, name, s) for s in SEEDS)
            rows.append(dict(model=key, panel=name,
                             seed=f"ALL ({N_SEEDS} seeds)",
                             strict_at_T_train="",
                             strict_at_T32="",
                             keeps_improving=keep_n))
    path = out / "per_seed_tally.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "panel", "seed",
                                                "strict_at_T_train",
                                                "strict_at_T32",
                                                "keeps_improving"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_csv(cache, out):
    rows = []
    sweeps = cache["sweeps"]
    for name in DATASETS:
        for key in MODELS:
            for T in SWEEP_T:
                vals = [sweeps[key][str(s)][name][str(T)] for s in SEEDS]
                m, sd = mean_sd(vals)
                rows.append(dict(fig="F8", series=key, panel=name, x=T,
                                 mean=f"{m:.4f}", sd=f"{sd:.4f}",
                                 **{f"seed{s}": f"{v:.4f}" for s, v in zip(SEEDS, vals)}))
    path = out / "data.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fig", "series", "panel", "x",
                                                "mean", "sd", "seed0", "seed1", "seed2"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    apply_style()
    root = repo_root()
    out = root / "figures" / "depth_by_difficulty"
    out.mkdir(parents=True, exist_ok=True)

    specs = model_specs(root)
    data = load_data()
    cache = compute_sweeps(root, specs, data,
                           root / "week2_results" / "probe_sweeps.json")
    fig8(cache, specs, data, out)
    fig8_per_seed(cache, specs, data, out)
    rows = write_csv(cache, out)
    tally_rows = write_tally(cache, out)

    print(f"\nWrote {len(rows)} F8 rows -> {out / 'data.csv'}")
    print(f"Wrote {len(tally_rows)} tally rows -> {out / 'per_seed_tally.csv'}")
    print("Figures:", out / "depth_by_difficulty.png", ",", out / "depth_by_difficulty_per_seed.png")
    # tallies for the README: keeps improving (strict at T=32 >= strict at
    # T_train = 8) vs drops/collapses, per model and difficulty
    print(f"\nKeeps-improving tally (strict at T=32 >= strict at T={T_TRAIN}):")
    for key in MODELS:
        for name in DATASETS:
            keep_n = sum(keeps_improvement(cache, key, name, s) for s in SEEDS)
            detail = ", ".join(
                f"seed{s}: {cache['sweeps'][key][str(s)][name][str(T_TRAIN)]:.3f}"
                f"->{cache['sweeps'][key][str(s)][name]['32']:.3f}"
                f"({'keeps' if keeps_improvement(cache, key, name, s) else 'drops'})"
                for s in SEEDS)
            print(f"  {name:7s} {key:20s} {keep_n}/{N_SEEDS} seeds keep improving  [{detail}]")
    # key numbers for the README
    for name in DATASETS:
        for key in MODELS:
            at = {T: mean_sd([cache["sweeps"][key][str(s)][name][str(T)] for s in SEEDS])
                  for T in SWEEP_T}
            best_T = max(SWEEP_T, key=lambda T: (at[T][0], -T))
            m, sd = at[best_T]
            m8, sd8 = at[T_TRAIN]
            m32, sd32 = at[32]
            print(f"  {name:7s} {key:20s} T=8: {m8:.3f}+-{sd8:.3f} | "
                  f"best T={best_T}: {m:.3f}+-{sd:.3f} | T=32: {m32:.3f}+-{sd32:.3f}")


if __name__ == "__main__":
    main()
