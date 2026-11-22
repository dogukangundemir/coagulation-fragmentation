"""Family B (BD-WallAware) ladder energy and cell size-distribution figures.

Builds the two Family B energy/mechanism figures from the trained seed-0
commitment-ladder checkpoint and metrics that scripts/train_ladder.py writes
into results/ladder_train/ (read-only for this script):

  ladder_energy_V_steps.png        free energy V of the trained ladder over
                                   the 32-step rollout on the first 64 test
                                   mazes, per micro step, with the
                                   fixed-rate phases (frozen rates: V is the
                                   Lyapunov function) shaded and the
                                   slow-phase rate refreshes marked
  ladder_cell_size_distribution.png
                                   size distribution (rung shares) over
                                   time in one path cell vs one dead-end
                                   cell of one test maze, plus the maze map
                                   with both cells marked
  data.csv                         every number plotted or quoted (tidy long
                                   format: panel, series, item, step, rung,
                                   value)
  README.md                        captions / how-to-read / key takeaways /
                                   data sources / caveats; every quoted
                                   number is read back from data.csv

Model: LadderCFNet (cf/models/ladder.py) as configured in the metrics file
(defaults: hid=16, K=4 rungs, wall-aware, detailed-balance rates, seed 0,
T_train=8, r_s=2 outer steps between rate refreshes, n_f=2 chemistry
micro-steps per outer step). Verified by the script itself, on CPU:
  - cf resolves to this clone (exit non-zero otherwise);
  - the recomputed per-maze V (summed over mazes, in float64 - at these
    reduced-budget rates the equilibrium weights Q overflow float32
    mid-rollout, so the stored trace turns NaN while the chemistry state
    stays finite) matches the stored ladder_metrics.json energy trace to
    1e-3 relative over the stored trace's finite prefix (outer and micro
    steps); the float64 recomputation must itself stay finite;
  - the metrics config is the expected BD-WallAware arm (no smoke runs).

Run from the clone root:
    python -m scripts.figures.ladder_energy
Optional flags: --results results/ladder_train --out figures/ladder_energy.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.legend
import matplotlib.pyplot as plt
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

import run_week2  # noqa: E402  (repo data pipeline, read-only)
import train_ladder  # noqa: E402  (the B-WallAware factory)

OI = style.OKABE_ITO
C_V = OI["blue"]                 # free energy line
C_GRIDS = {"path": OI["bluish_green"], "dead": OI["orange"]}
C_REFRESH = OI["grey"]
C_PHASE = OI["grey"]
C_HORIZON = OI["vermillion"]
MIN_FONT = 10.0
TOL_REL = 1e-3        # the repo's established spot-check tolerance (a_vs_d)
WALK_OFF = "#f0f0f0"
WALL = "#3f3f3f"


# ------------------------------------------------------------- run guards

def ensure_clone():
    """Refuse to run unless the imported cf package is THIS clone."""
    cf_root = Path(cf.__file__).resolve().parents[1]
    if cf_root != ROOT.resolve():
        raise SystemExit("cf resolves to %s, not this clone (%s)"
                         % (cf_root, ROOT.resolve()))
    return cf_root


def check_metrics_arm(metrics, allow_smoke=False):
    cfg = metrics["config"]
    expect = dict(model="BD-WallAware", detailed_balance=True, wall_aware=True,
                  T_train=8)
    bad = {k: (cfg.get(k), v) for k, v in expect.items() if cfg.get(k) != v}
    if cfg.get("fast") and not allow_smoke:
        raise SystemExit("refusing to plot a --fast smoke run "
                         "(use --allow-smoke for a deliberate dry run)")
    if bad:
        raise SystemExit("metrics config mismatch: %s" % bad)
    return cfg


# ------------------------------------------------------------ rollouts (CPU)

def rung_weights64(rates):
    """Equilibrium weights Q in float64 (same recursion as
    LadderCFNet.rung_weights, but in double: the reduced-budget rates put
    some b ~ 1e-31, so the float32 recursion overflows to inf mid-rollout)."""
    a, b = rates
    ad, bd = a.double(), b.double()
    Qs = [torch.ones_like(ad[:, :, 0])]
    for i in range(ad.shape[2]):
        Qs.append(Qs[-1] * ad[:, :, i] / bd[:, :, i])
    return torch.stack(Qs, dim=2)


def v_per_maze(model, c, rates):
    """V = sum_k c_k (log(c_k/Q_k) - 1) per maze (free_energy, but per
    sample and in float64: the float32 free_energy NaNs once Q leaves the
    float32 range, while the chemistry state itself stays finite)."""
    Q = rung_weights64(rates)                         # (B, hid, K, H, W)
    cd = c.double()
    cc = torch.clamp(cd, min=1e-300)
    per = (cd * (torch.log(cc) - torch.log(Q)) - cd).sum(dim=(1, 2, 3, 4))
    return per                                         # (B,) float64


def rollout_v(model, x, T):
    """Mirror LadderCFNet.logits_at: per-maze V after every micro step.

    Returns V_init (B,), V_micro (B, T*n_f), V_outer (B, T). Rate refreshes
    follow the r_s schedule of forward/logits_at exactly.
    """
    B = x.shape[0]
    rs, nf = model.r_s, model.n_f
    V_micro = np.zeros((B, T * nf))
    V_outer = np.zeros((B, T))
    with torch.no_grad():
        c = model.initial_state(x)
        rates = model.refresh_rates(x, c)
        V_init = v_per_maze(model, c, rates).numpy()
        idx = 0
        for t in range(1, T + 1):
            if (t - 1) % rs == 0:
                rates = model.refresh_rates(x, c)
            for _ in range(nf):
                c, _ = model.micro_step(c, rates)
                V_micro[:, idx] = v_per_maze(model, c, rates).numpy()
                idx += 1
            V_outer[:, t - 1] = v_per_maze(model, c, rates).numpy()
    return V_init, V_micro, V_outer


def rollout_cell(model, x1, T):
    """Rung map of one maze after every outer step (t=0 = initial state),
    plus path logits after every outer step."""
    with torch.no_grad():
        c = model.initial_state(x1)
        c_steps = [c[0].numpy().copy()]
        logits = []
        rates = None
        for t in range(1, T + 1):
            if rates is None or (t - 1) % model.r_s == 0:
                rates = model.refresh_rates(x1, c)
            for _ in range(model.n_f):
                c, _ = model.micro_step(c, rates)
            c_steps.append(c[0].numpy().copy())
            logits.append(model.read(x1, c)[0].numpy())
    return np.stack(c_steps), np.stack(logits)          # (T+1, hid, K, H, W)


# ------------------------------------------------------------ cell geometry

def neighbours(r, c, H, W):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            yield rr, cc


def find_cells(wall, path_mask, start, goal):
    """(path cells excluding S/G, dead-end cells) of one maze.

    Dead end: walkable, not on the labelled shortest path, exactly one
    walkable 4-neighbour.
    """
    H, W = wall.shape
    walk = wall == 0
    path, dead = [], []
    for r in range(H):
        for c in range(W):
            if not walk[r, c]:
                continue
            n_walk = sum(1 for rr, cc in neighbours(r, c, H, W) if walk[rr, cc])
            if path_mask[r, c] and (r, c) not in (start, goal):
                path.append((r, c))
            elif not path_mask[r, c] and n_walk == 1:
                dead.append((r, c))
    return path, dead


def cell_timeseries(c_map, cell):
    """Number-weighted rung shares p_k(t) and mean size of one cell over time.

    c_map: (T+1, hid, K, H, W). p_k = sum_ch c[ch, k-1] / sum_ch sum_k' c;
    sums to 1 over rungs at every step; at t = 0 all mass is on rung 1 by
    initialisation. Mean cluster size = sum_k k p_k.
    """
    r, c = cell
    c_hist = c_map[:, :, :, r, c]                      # (T+1, hid, K)
    tot = c_hist.sum(axis=(1, 2))                      # (T+1,)
    shares = c_hist.sum(axis=1) / tot[:, None]         # (T+1, K)
    mean_size = (shares * np.arange(1, shares.shape[1] + 1)).sum(axis=1)
    return shares, mean_size


def median_pick(items, values):
    """Deterministic median rule: sort by value, take the upper-middle item."""
    order = sorted(range(len(items)), key=lambda i: (values[i], items[i]))
    return items[order[len(order) // 2]]


# ------------------------------------------------------------ overlap guard

def check_overlaps(fig, name):
    """Programmatic readability guard: no text may overlap other text, a
    drawn line segment, a band or a patch; a legend may not sit on plotted
    artists; no font below MIN_FONT. Adapted from scripts/figures/a_vs_d.py
    (the repo's established guard); marker-only lines are tested as marker
    hits (single points), not as interpolated segments."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    PAD = 1.5  # px tolerance around a text bbox for segment hits

    def line_hits(ln, bb):
        ax = ln.axes
        if ax is None:
            return True
        xy = ax.transData.transform(ln.get_xydata())
        if ln.get_linestyle() in ("", "None", "none"):   # markers only
            return any((bb.x0 - PAD) <= px <= (bb.x1 + PAD) and
                       (bb.y0 - PAD) <= py <= (bb.y1 + PAD)
                       for px, py in xy)
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
        if ax.get_label() == "<colorbar>":     # colorbar internals
            continue
        artists += [(str(ln.get_label() or "line"), ln) for ln in ax.get_lines()]
        for col in ax.collections:
            ab = col.get_window_extent(ren)
            if not np.isfinite([ab.x0, ab.y0, ab.x1, ab.y1]).all():
                continue                        # empty extent: nothing drawn
            artists.append(("band", ab))
        artists += [(str(p.get_label() or "patch"),
                     p.get_window_extent(ren)) for p in ax.patches]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                bad.append("text/text: %r vs %r" % (boxes[i][0], boxes[j][0]))
    for tname, tb in boxes:
        for aname, ab in ((a, b) for a, b in artists
                          if not isinstance(b, Line2D)):
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
            for ax2 in fig.axes:
                if ax2.get_label() == "<colorbar>":
                    continue
                for ab in [c.get_window_extent(ren) for c in ax2.collections]:
                    if not np.isfinite([ab.x0, ab.y0, ab.x1, ab.y1]).all():
                        continue
                    if lbb.overlaps(ab):
                        bad.append("legend overlaps band")
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip() and getattr(t, "get_fontsize",
                                            lambda: 99.0)() < MIN_FONT:
            bad.append("font < %.0f pt: %r" % (MIN_FONT, t.get_text()[:30]))
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
    """Figure title + two-line reading guide (both >= 10 pt)."""
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.945, guide, ha="center", va="top", fontsize=10.5,
             color="#333333")


# ---------------------------------------------------------------- CSV helpers

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "series", "item", "step", "rung", "value"])
        w.writerows(rows)


def csv_num(rows, panel, series, item="", step="", rung=""):
    """Read one value back from the rows that (will) constitute data.csv."""
    for panel_, series_, item_, step_, rung_, value in rows:
        if panel_ == panel and series_ == series and item_ == item \
                and str(step_) == str(step) and str(rung_) == str(rung):
            return value
    raise KeyError((panel, series, item, step, rung))


# ------------------------------------------------------------------- README

def write_readme(out_dir, rows, args, metrics):
    """README.md next to the PNGs; every quoted number is read from `rows`,
    i.e. from data.csv exactly as written."""
    n = lambda panel, series: csv_num(rows, panel, series)      # noqa: E731
    label = n("info", "label")
    epochs = int(n("info", "epochs_done"))
    planned = int(n("info", "epochs_planned"))
    budget = float(n("info", "budget_minutes"))
    hid = int(n("info", "hid"))
    K = int(n("info", "K"))
    rs = int(n("info", "r_s"))
    nf = int(n("info", "n_f"))
    T_train = int(n("info", "T_train"))
    T_star = int(n("info", "T_star"))
    n_V = int(n("info", "V_n_mazes"))
    T_V = int(n("info", "V_T"))
    pairs = int(n("info", "V_phase_pairs"))
    pairs_dec = int(n("info", "V_phase_pairs_decreasing"))
    frac = float(n("info", "V_phase_decrease_frac"))
    batch_dec = n("info", "V_phases_batch_decreasing")
    n_phase_v = int(pairs) // int(n_V)
    n_not = int(pairs) - int(pairs_dec)
    if n_not == 0:
        notfall_str = "no exception"
    elif n_not == 1:
        notfall_str = ("exactly one pair of the %d mazes x %d phases does "
                       "not fall" % (n_V, n_phase_v))
    else:
        notfall_str = "%d pairs do not fall" % n_not
    frac_pct = "%.1f%%" % (100.0 * frac)
    maze_i = int(n("info", "cell_maze_index"))
    path_len = int(n("info", "cell_maze_path_len"))
    n_path = int(n("info", "cell_n_path_cells"))
    n_dead = int(n("info", "cell_n_deadend_cells"))
    c_path = n("info", "cell_path_pick")
    c_dead = n("info", "cell_dead_pick")
    pick_rule = n("info", "cell_pick_rule")
    mp0 = float(csv_num(rows, "mean_size", "path_cell", c_path, 0))
    mpT = float(csv_num(rows, "mean_size", "path_cell", c_path, T_V))
    mpTs = float(csv_num(rows, "mean_size", "path_cell", c_path, T_star))
    md0 = float(csv_num(rows, "mean_size", "dead_cell", c_dead, 0))
    mdT = float(csv_num(rows, "mean_size", "dead_cell", c_dead, T_V))
    mdTs = float(csv_num(rows, "mean_size", "dead_cell", c_dead, T_star))
    p_reg = float(csv_num(rows, "size_dist", "path_cell", c_path, T_V, K))
    d_reg1 = float(csv_num(rows, "size_dist", "dead_cell", c_dead, T_V, 1))
    pred_strict = n("info", "cell_maze_pred_strict_at_Tstar")
    cell_diff = mpT - mdT
    solved_phrase = ("solved strictly at T*" if pred_strict == "True"
                     else "NOT solved strictly at T* (%s)" % pred_strict)
    # crossover story, read back from the mean_size panel of data.csv:
    # the step from which the path cell stays above the dead-end cell,
    # plus who leads at each step before it
    ms_rows = {(r[1], int(r[3])): float(r[5]) for r in rows
               if r[0] == "mean_size"}
    m_path = {t: v for (s, t), v in ms_rows.items() if s == "path_cell"}
    m_dead = {t: v for (s, t), v in ms_rows.items() if s == "dead_cell"}
    if sorted(m_path) != list(range(T_V + 1)) or \
            sorted(m_dead) != list(range(T_V + 1)):
        raise SystemExit("mean_size panel is incomplete in data.csv rows")
    final_cross = None
    if m_path[T_V] > m_dead[T_V]:
        final_cross = T_V
        while final_cross > 0 and \
                m_path[final_cross - 1] > m_dead[final_cross - 1]:
            final_cross -= 1
    if cell_diff >= 1.0:
        commit_verdict = ("the trained ladder puts its commitment mass on "
                          "the path and leaves dead ends uncommitted, the "
                          "hysteresis behaviour the family card predicts.")
    elif final_cross is None:
        commit_verdict = ("the path cell never ranks above the dead-end "
                          "cell in the recorded rollout - at this reduced "
                          "budget both stay near the bottom rungs, and the "
                          "sharp path/dead-end commitment split the family "
                          "card predicts (hysteresis) has not emerged yet.")
    else:
        pre = list(range(0, final_cross))
        dead_lead = [t for t in pre if m_dead[t] > m_path[t]]
        early_lead = [t for t in pre if m_path[t] > m_dead[t]]
        ties = [t for t in pre if m_dead[t] == m_path[t]]
        early_str = ", ".join(str(t) for t in early_lead) or "no step"
        tie_str = ("a tie at t = %s"
                   % ", ".join(str(t) for t in ties)) if ties else "no ties"
        commit_verdict = (
            "the path cell overtakes the dead-end cell only in the second "
            "half of the rollout: up to step %d the dead-end cell has the "
            "higher mean rung at %d of the %d recorded steps (the path "
            "cell is ahead only at step%s %s, %s), and the path cell stays "
            "above only from step %d on - even then both stay near the "
            "bottom rungs, and the sharp path/dead-end commitment split "
            "the family card predicts (hysteresis) has not emerged yet at "
            "this reduced budget."
            % (final_cross - 1, len(dead_lead), len(pre),
               "" if len(early_lead) == 1 else "s", early_str, tie_str,
               final_cross))
    eff = n("info", "params_effective")
    raw = n("info", "params_raw")
    hw = n("info", "hardware_device")
    test8 = n("info", "test_strict_at_T8")
    testS = n("info", "test_strict_at_Tstar")
    probe7 = n("info", "probe7_strict_at_Tstar")
    probe9 = n("info", "probe9_strict_at_Tstar")
    best_ep = n("info", "best_epoch")
    sd_type = n("info", "V_band_sd_type")
    clip_v = n("info", "V_clip_note")
    clip_share = n("info", "size_dist_share_clip")
    ckpt = n("info", "ckpt_file")
    nfin = int(n("info", "V_finite_prefix_micro"))
    nmic = int(n("info", "V_T")) * int(n("info", "n_f"))
    vrec = n("info", "V_recompute_note")
    vrec_cap = vrec[0].upper() + vrec[1:] + "."

    txt = f"""# Family B ladder: free energy and per-cell size distributions

Figures built by `scripts/figures/ladder_energy.py` from the trained
checkpoint and metrics in `{args.results}/` (BD-WallAware = Family B
Becker-Döring commitment ladder: wall-aware, detailed-balance rates, K = {K}
rungs, hid = {hid}, seed 0, trained at fixed T = {T_train}, r_s = {rs},
n_f = {nf}). Every number below is read back from `data.csv` in this folder.
Every number carries its training budget: **{label}** ({epochs}/{planned}
epochs done of a cosine schedule planned for {planned} epochs, stopped by a
{budget} min wall-clock budget, best epoch {best_ep}), trained on CPU
({hw}, reduced budget = not a reproduction of the 160-epoch protocol).

Numbers of the run for context (from `{args.results}/ladder_metrics.json`,
also mirrored in `data.csv`): {raw} raw parameters ({eff} effective),
test strict at T = 8 = {test8} and at the validation-selected T* = {T_star} =
{testS}, probe-7 strict at T* = {probe7}, probe-9 strict at T* = {probe9}.
Split: level-5 curriculum test split (1000 mazes); model input = walkable +
start + goal channels; readout = threshold 0.5 on per-cell logits.

---

## `ladder_energy_V_steps.png` - free energy over the rollout

**Caption.** Free energy V of the trained ladder, mean per maze over the
first {n_V} test mazes of the level-5 split, after every chemistry
micro-step of a {T_V}-outer-step rollout (n_f = {nf} micro-steps per outer
step; the rollout runs {T_V // T_train}x past the training horizon
T_train = {T_train}). Grey spans mark the fixed-rate phases: the rates
(g, mu) are recomputed only every r_s = {rs} outer steps, refreshing
before outer steps 1, 1+r_s, 1+2*r_s, ...; between two refreshes the
chemistry relaxes under frozen rates (a fixed-rate phase) and V is a
Lyapunov function, so V must fall inside each fixed-rate phase while the
refreshes can re-set the landscape.

**How to read.** Blue line = mean per-maze V after each micro step; blue
band = {sd_type}, n = {n_V} mazes ({clip_v}); grey spans = fixed-rate phases
(rates frozen; alternate shading only marks the segmentation, every phase
is fixed-rate); grey dashed line = slow-phase rate refresh; vermillion dotted
line = trained horizon. V is dimensionless (a sum of c log c terms, c in
evidence units); more negative = chemically more settled.

**Key takeaway.** Within fixed-rate phases the summed (batch) V decreases
in {batch_dec} of {n_phase_v} phases, and {pairs_dec} of {pairs} per-maze
phase pairs end below their start ({pairs_dec}/{pairs} = {frac_pct};
{notfall_str}); the visible upward steps happen
exactly at the rate refreshes, where the slow phase re-writes the energy
profile - the energy is a per-phase Lyapunov function, not a global one
(two-timescale design, as in the family card).

**Data source.** `data.csv` (panel `V_trace`, panel `V_phase`); checkpoint
`{args.results}/{ckpt}`; metrics
`{args.results}/ladder_metrics.json`; mazes via the seeded level-5
curriculum splits (`cf.data.curriculum`). Script recomputed V from the
checkpoint in float64 and cross-checked the summed trace against
`ladder_metrics.json` (agreement to 1e-3 relative, enforced over the
stored trace's finite prefix of {nfin} of {nmic} micro steps and the
matching outer prefix).

**Caveats.** Preliminary, 1 seed, reduced budget ({epochs}/{planned} epochs,
~{budget} min CPU); {n_V} of 1000 test mazes; V depends on the learned mass
scale, so absolute values are not comparable across training runs; the
discrete capped update contracts the continuous-time Lyapunov theorem, so
single micro-steps need not fall, phases do. {vrec_cap}

---

## `ladder_cell_size_distribution.png` - path cell vs dead-end cell

**Caption.** Left: test maze {maze_i} of the level-5 split (true path
{path_len} cells) with the path cells in green, every dead-end cell
(walkable, off-path, exactly one free neighbour) in orange, and the two
analysed cells outlined in black; the dots are the trained model's own
on-path prediction at T* = {T_star}. Middle and right: the size
distribution of the two outlined cells over the rollout - the share of the
cell's clusters on each of the K = {K} rungs (rung 1 = "no idea" to rung
{K} = "certainly on the path"), t = 0 being the initial state with all
evidence on rung 1. Bottom: the number-weighted mean rung (mean cluster
size) of the same two cells.

**How to read.** Maze map: green = true path (BFS shortest S-G path, S and
G marked), orange = dead-end cells, pale = other free cells, dark = walls.
Heatmaps: x = outer step (0 = initial state), y = ladder rung k, colour =
share of clusters on that rung, one shared colour scale clipped to [0, 1]
({clip_share}); a cell "commits" when its mass moves to the top rungs.
Line panel: green = the path cell, orange = the dead-end cell; grey dashed
= T* = {T_star} (validation-selected horizon).

**Key takeaway.** By step {T_V} the path cell ({c_path}) sits at a mean
rung of {mpT:.2f} (from {mp0:.2f} at t = 0; {p_reg:.1%} of its clusters on
the top rung {K}) and the dead-end cell ({c_dead}) at {mdT:.2f}
({d_reg1:.1%} of its clusters on rung 1) - a separation of
{cell_diff:.2f} rungs at step {T_V}: {commit_verdict} At the model's
selected horizon T* = {T_star} the mean rung is {mpTs:.2f} (path) vs
{mdTs:.2f} (dead end); the shown maze is {solved_phrase}.

**Data source.** `data.csv` (panels `size_dist`, `mean_size`, `info`);
checkpoint `{args.results}/{ckpt}`; metrics
`{args.results}/ladder_metrics.json`; maze {maze_i} from the seeded level-5
test split.

**Caveats.** Preliminary, 1 seed, reduced budget ({epochs}/{planned} epochs,
~{budget} min CPU). Cell selection is deterministic, not hand-picked: the
median path cell of {n_path} candidates and the median dead-end cell of
{n_dead} candidates by final mean rung ({pick_rule}); still a single maze
and one cell per kind (n = 1, an instance view - the level-5 aggregate
solve rate is in `data.csv`, see test strict above). Shares are
number-weighted over the hid = {hid} channels of the cell; the map's
prediction dots are the seed-0 model at its own T*.
"""
    (out_dir / "README.md").write_text(txt, encoding="utf-8")
    print("wrote %s" % (out_dir / "README.md"))


# ---------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/ladder_train",
                    help="train_ladder output dir (ladder_metrics.json + .pt)")
    ap.add_argument("--out", default="figures/ladder_energy")
    ap.add_argument("--maze", type=int, default=None,
                    help="test-split maze index for the cell figure "
                         "(default: first maze with a dead-end cell)")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="deliberate dry run against a --fast train_ladder "
                         "output (for development only)")
    args = ap.parse_args(argv)
    ensure_clone()

    results_dir = ROOT / args.results
    metrics = json.loads(
        (results_dir / "ladder_metrics.json").read_text(encoding="utf-8"))
    cfg = check_metrics_arm(metrics, allow_smoke=args.allow_smoke)
    if cfg["seed"] != 0:
        raise SystemExit("this figure shows the seed-0 run, metrics say seed "
                         "%d" % cfg["seed"])
    hid, K, seed = cfg["hid"], cfg["K"], cfg["seed"]
    ckpt_path = results_dir / ("BD-WallAware_seed%d.pt" % seed)
    if not ckpt_path.exists():
        ckpt_path = results_dir / ("BD-WallAware_seed%d.ckpt.pt" % seed)
    model = train_ladder.make_model(hid=hid, K=K)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "best_state" in state:
        # resumable train_ladder checkpoint: prefer the best-validation state
        state = state.get("best_state") or state["model_state"]
    model.load_state_dict(state)
    model.eval()
    print("loaded %s (%d epochs done, label %r)"
          % (ckpt_path.name, metrics["training"]["epochs_done"],
             metrics["label"]))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []                                   # the data.csv rows

    def info(series, value):
        rows.append(["info", series, "", "", "", str(value)])

    info("ckpt_file", ckpt_path.name)

    tr, sel = metrics["training"], metrics["selection"]
    info("model_name", cfg["model"])
    info("hid", hid)
    info("K", K)
    info("seed", seed)
    info("T_train", cfg["T_train"])
    info("epochs_done", tr["epochs_done"])
    info("epochs_planned", cfg["epochs_planned"])
    info("budget_minutes", cfg["budget_minutes"])
    info("best_epoch", tr.get("best_epoch", "not run"))
    info("params_raw", metrics["params"]["raw"])
    info("params_effective", metrics["params"]["effective"])
    info("hardware_device", metrics["hardware"]["device_name"])
    info("T_star", sel["T_star"])
    info("test_strict_at_T8", metrics["test_strict_at_T_train"])
    info("test_strict_at_Tstar", metrics["test_strict_at_T_star"])
    info("probe7_strict_at_T8", metrics["probe7_strict_at_T_train"])
    info("probe7_strict_at_Tstar", metrics["probe7_strict_at_T_star"])
    info("probe9_strict_at_T8", metrics["probe9_strict_at_T_train"])
    info("probe9_strict_at_Tstar", metrics["probe9_strict_at_T_star"])
    info("label", metrics["label"])
    info("r_s", model.r_s)
    info("n_f", model.n_f)
    info("rate_refreshes_note",
         "rates refresh before outer steps 1, 1+r_s, 1+2*r_s, ...; "
         "between two refreshes the chemistry relaxes under frozen rates "
         "(fixed-rate phase) and V is a Lyapunov function")

    # ---------------------------------------------------------------- data
    print("loading the seeded curriculum splits (level 5) ...")
    data = run_week2.load_data(dict(n_train=4000, n_val=1000, n_test=1000,
                                    n_probe=1000))
    n_V, T_V = int(metrics["energy"]["n_mazes"]), int(metrics["energy"]["T_V"])
    rs, nf = model.r_s, model.n_f
    x_V = data["test"].x[:n_V]
    V_init, V_micro, V_outer = rollout_v(model, x_V, T_V)

    # spot check: recomputed per-maze V summed over mazes vs stored trace.
    # The stored trace is float32 and overflows to NaN once the equilibrium
    # weights Q leave the float32 range mid-rollout (the reduced-budget
    # rates put b ~ 1e-31); the float64 recomputation must stay finite and
    # match the stored trace wherever the stored trace is finite.
    stored_outer = np.asarray(metrics["energy"]["V_by_outer_step"])
    stored_micro = np.asarray(metrics["energy"]["V_by_micro_step"])
    n_finite = {}
    for name, recomp, stored in [("outer", V_outer.sum(axis=0), stored_outer),
                                 ("micro", V_micro.sum(axis=0), stored_micro)]:
        if not np.isfinite(recomp).all():
            raise SystemExit("float64 %s-step V recomputation is not finite"
                             % name)
        fin = np.isfinite(stored)
        n_finite[name] = int(fin.sum())
        if not np.isclose(recomp[fin], stored[fin],
                          rtol=TOL_REL, atol=1e-3).all():
            m = np.max(np.abs(recomp[fin] - stored[fin])
                       / (np.abs(stored[fin]) + 1e-9))
            raise SystemExit("recomputed %s-step V trace mismatches "
                             "ladder_metrics.json over its finite prefix "
                             "(max rel err %.2e)" % (name, m))
    first_nan_micro = int(np.nonzero(~np.isfinite(stored_micro))[0][0]) + 1
    print("V trace cross-check ok (finite prefixes: %d/%d outer, %d/%d "
          "micro; float32 trace NaNs from micro step %d)"
          % (n_finite["outer"], len(stored_outer), n_finite["micro"],
             len(stored_micro), first_nan_micro))
    info("V_finite_prefix_micro", n_finite["micro"])
    info("V_first_nan_micro_step", first_nan_micro)
    info("V_recompute_note",
         "the stored float32 V trace overflows to NaN from micro step %d "
         "(the equilibrium weights Q leave the float32 range at these "
         "rates, b ~ 1e-31, while the chemistry state stays finite); the "
         "plotted V is recomputed in float64, which matches the stored "
         "trace to %.0e relative over its finite prefix (%d of %d micro "
         "steps)" % (first_nan_micro, TOL_REL, n_finite["micro"],
                     len(stored_micro)))

    # fixed-rate phases: micro steps of phase p are 4p+1 .. 4p+4
    # (r_s * n_f micro-steps each); dV = V(end of phase) - V(start of phase)
    n_phase = T_V // rs
    msz = rs * nf
    starts_phase = np.concatenate(
        [V_init[:, None], V_micro[:, :-msz]], axis=1)[:, 0::msz]
    ends_phase = V_micro[:, msz - 1::msz]
    dV = ends_phase - starts_phase
    dec_maze_phase = int((dV < 0.0).sum())
    n_pairs = int(dV.size)
    dec_frac = dec_maze_phase / n_pairs
    batch_dec = int((dV.sum(axis=0) < 0.0).sum())
    info("V_n_mazes", n_V)
    info("V_T", T_V)
    info("V_phase_pairs", n_pairs)
    info("V_phase_pairs_decreasing", dec_maze_phase)
    info("V_phase_decrease_frac", "%.4f" % dec_frac)
    info("V_phases_batch_decreasing", "%d/%d" % (batch_dec, n_phase))
    info("V_band_sd_type", "sample sd over %d mazes (n-1 denominator)" % n_V)
    info("V_clip_note",
         "V is unbounded below; the sd band is not clipped (no valid range)")
    for i in range(T_V * nf):
        rows.append(["V_trace", "mean_V", "%d mazes" % n_V, i + 1, "",
                     "%.6f" % V_micro[:, i].mean()])
        rows.append(["V_trace", "sd_V", "%d mazes" % n_V, i + 1, "",
                     "%.6f" % statistics.stdev(V_micro[:, i])])
    for j in range(n_phase):
        rows.append(["V_phase", "dV_batch_total", "all mazes", j + 1, "",
                     "%.6f" % dV[:, j].sum()])
        for i in range(n_V):
            rows.append(["V_phase", "dV_end_minus_start",
                         "maze%d" % i, j + 1, "", "%.6f" % dV[i, j]])

    # ------------------------------------------------------- figure 1 (V)
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    fig.subplots_adjust(top=0.76, bottom=0.30, left=0.08, right=0.97)
    xs = np.arange(1, T_V * nf + 1)
    mean_v = V_micro.mean(axis=0)
    sd_v = np.array([statistics.stdev(V_micro[:, i])
                     for i in range(T_V * nf)])
    ax.fill_between(xs, mean_v - sd_v, mean_v + sd_v, color=C_V,
                    alpha=0.22, lw=0)
    ax.plot(xs, mean_v, color=C_V, lw=2.0)
    for p in range(n_phase):
        if p % 2 == 0:
            ax.axvspan(p * msz + 0.5, (p + 1) * msz + 0.5,
                       color=C_PHASE, alpha=0.10, lw=0, zorder=0)
    refresh_xs = np.arange(0, n_phase) * msz + 1
    for x0 in refresh_xs:
        ax.axvline(x0, color=C_REFRESH, ls="--", lw=1.0, zorder=0)
    ax.axvline(nf * cfg["T_train"], color=C_HORIZON, ls=":", lw=1.4, zorder=1)
    ax.set_xlabel("micro step (n_f = %d chemistry micro-steps per outer "
                  "step; %d micro steps = %d outer steps)"
                  % (nf, T_V * nf, T_V))
    ax.set_ylabel("free energy V per maze (dimensionless)")
    handles = [Line2D([], [], color=C_V, lw=2.0,
                      label="mean V per maze (after every micro step)"),
               Patch(facecolor=C_V, alpha=0.22,
                     label="sample sd over %d mazes (n-1)" % n_V),
               Patch(facecolor=C_PHASE, alpha=0.35,
                     label="fixed-rate phase (frozen rates: V is Lyapunov)"),
               Line2D([], [], color=C_REFRESH, ls="--", lw=1.0,
                      label="rate refresh, slow phase (every r_s = %d outer)"
                      % rs),
               Line2D([], [], color=C_HORIZON, ls=":", lw=1.4,
                      label="trained horizon T_train = %d (outer steps)"
                      % cfg["T_train"])]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=10.0)
    headline(fig,
             "Free energy V falls in %d/%d fixed-rate phases (batch total) "
             "and in %d/%d = %.1f%% of per-maze phase pairs"
             % (batch_dec, n_phase, dec_maze_phase, n_pairs,
                100.0 * dec_frac),
             "Guide: one outer step = a rate refresh every r_s = %d steps + "
             "n_f = %d capped chemistry micro-steps under frozen rates.\n"
             "Line = mean per-maze V over %d test mazes after every micro "
             "step; band = sample sd; grey spans = fixed-rate phases, "
             "dashed = rate refresh. V recomputed in\\nfloat64 (the eval's "
             "float32 V overflows from micro step %d; the chemistry state "
             "itself stays finite)." % (rs, nf, n_V, first_nan_micro))
    save(fig, "ladder_energy_V_steps.png", out_dir)

    # ------------------------------------------------------- figure 2 (cells)
    maze_i = args.maze
    if maze_i is None:
        maze_i = 0
        while True:
            _, dead = find_cells(data["test"].walls[maze_i],
                                 data["test"].y[maze_i].numpy().astype(bool),
                                 tuple(data["test"].starts[maze_i]),
                                 tuple(data["test"].goals[maze_i]))
            if dead:
                break
            maze_i += 1
    wall = data["test"].walls[maze_i]
    y_mask = data["test"].y[maze_i].numpy().astype(bool)
    start = tuple(int(v) for v in data["test"].starts[maze_i])
    goal = tuple(int(v) for v in data["test"].goals[maze_i])
    path_cells, dead_cells = find_cells(wall, y_mask, start, goal)
    if not path_cells or not dead_cells:
        raise SystemExit("maze %d has no eligible path/dead-end cell pair"
                         % maze_i)
    c_map, logits = rollout_cell(model, data["test"].x[maze_i:maze_i + 1],
                                 T_V)
    T_star = int(sel["T_star"])
    if not (1 <= T_star <= T_V):
        raise SystemExit("T* = %d outside the recorded rollout" % T_star)

    pick, pick_share, mean_size = {}, {}, {}
    for kind, cells in (("path", path_cells), ("dead", dead_cells)):
        finals = [cell_timeseries(c_map, cell)[1][-1] for cell in cells]
        cell = median_pick(cells, finals)
        shares, msize = cell_timeseries(c_map, cell)
        pick[kind], pick_share[kind], mean_size[kind] = cell, shares, msize
    info("cell_maze_index", maze_i)
    info("cell_maze_path_len", int(y_mask.sum()))
    info("cell_n_path_cells", len(path_cells))
    info("cell_n_deadend_cells", len(dead_cells))
    info("cell_path_pick", "r%d,c%d" % pick["path"])
    info("cell_dead_pick", "r%d,c%d" % pick["dead"])
    info("cell_pick_rule",
         "median by final mean rung (upper middle of the value-sorted list)")
    pred_on = logits[T_star - 1] >= 0.0
    info("cell_maze_pred_strict_at_Tstar",
         bool(np.array_equal(pred_on, y_mask)))
    info("size_dist_share_clip",
         "shares sum to 1 over rungs; colour scale fixed to [0, 1] (clipped "
         "by construction, no values outside)")
    for kind in ("path", "dead"):
        r, c = pick[kind]
        for t in range(T_V + 1):
            for k in range(K):
                rows.append(["size_dist", kind + "_cell",
                             "r%d,c%d" % (r, c), t, k + 1,
                             "%.6f" % pick_share[kind][t, k]])
            rows.append(["mean_size", kind + "_cell", "r%d,c%d" % (r, c),
                         t, "", "%.6f" % mean_size[kind][t]])

    fig = plt.figure(figsize=(13.5, 7.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.3, 1.3],
                          height_ratios=[1.0, 0.72])
    gs.update(top=0.70, bottom=0.09, left=0.03, right=0.985,
              hspace=0.42, wspace=0.30)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_hp = fig.add_subplot(gs[0, 1])
    ax_hd = fig.add_subplot(gs[0, 2])
    ax_line = fig.add_subplot(gs[1, 1:])

    # --- maze map (left, spans both rows)
    H, W = wall.shape
    for r in range(H):
        for c in range(W):
            if wall[r, c]:
                fc = WALL
            elif y_mask[r, c]:
                fc = C_GRIDS["path"]
            elif (r, c) in dead_cells:
                fc = C_GRIDS["dead"]
            else:
                fc = WALK_OFF
            ax_map.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=fc, edgecolor="white",
                                           lw=0.3))
    for kind in ("path", "dead"):
        r, c = pick[kind]
        ax_map.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor="none",
                                       edgecolor="black", lw=2.5))
    pred_c, pred_r = np.nonzero(pred_on)
    ax_map.plot(pred_c, pred_r, ls="none", marker="o", ms=6.5,
                mfc="none", mec="black", mew=1.2)
    ax_map.plot(start[1], start[0], ls="none", marker="o", ms=9,
                mfc="white", mec="black", mew=1.6)
    ax_map.plot(goal[1], goal[0], ls="none", marker="*", ms=14,
                mfc="black", mec="black")
    ax_map.set_xlim(-0.6, W - 0.4)
    ax_map.set_ylim(H - 0.4, -0.6)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for s in ax_map.spines.values():
        s.set_visible(False)
    ax_map.set_title("test maze %d (true path %d cells)"
                     % (maze_i, int(y_mask.sum())), fontsize=11.5)
    ax_map.legend(handles=[
        Patch(facecolor=C_GRIDS["path"], label="true path cell"),
        Patch(facecolor=C_GRIDS["dead"], label="dead-end cell"),
        Patch(facecolor=WALK_OFF, edgecolor="#b0b0b0",
              label="other free cell"),
        Patch(facecolor=WALL, label="wall"),
        Patch(facecolor="none", edgecolor="black", lw=2.5,
              label="cells shown right (outline)"),
        Line2D([], [], ls="none", marker="o", ms=6.5, mfc="none",
               mec="black", mew=1.2,
               label="predicted on-path at T* = %d" % T_star),
        Line2D([], [], ls="none", marker="o", ms=9, mfc="white",
               mec="black", mew=1.6, label="start"),
        Line2D([], [], ls="none", marker="*", ms=14, mfc="black",
               mec="black", label="goal"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=10.0)

    # --- rung-share heatmaps (shared [0, 1] scale)
    im = None
    for ax_h, kind in ((ax_hp, "path"), (ax_hd, "dead")):
        mat = pick_share[kind].T                    # (K, T+1)
        im = ax_h.imshow(mat, aspect="auto", origin="lower",
                         cmap="viridis", vmin=0.0, vmax=1.0,
                         interpolation="nearest",
                         extent=[-0.5, T_V + 0.5, 0.5, K + 0.5])
        ax_h.set_xticks([0, 8, 16, 24, 32])
        ax_h.set_yticks(range(1, K + 1))
        ax_h.set_xlabel("outer step (t = 0 is the initial state)",
                        fontsize=11.0)
        ax_h.set_title("%s cell (r%d, c%d) - final mean rung %.2f"
                       % (kind, pick[kind][0], pick[kind][1],
                          mean_size[kind][-1]), fontsize=11.5)
    ax_hp.set_ylabel("ladder rung k (k = 1 bottom ... %d top)" % K,
                     fontsize=11.0)
    cbar = fig.colorbar(im, ax=[ax_hp, ax_hd], shrink=0.85, pad=0.015)
    cbar.set_label("cluster share on rung k\n"
                   "(number-weighted, clipped to [0, 1])",
                   fontsize=10.0)
    cbar.ax.tick_params(labelsize=10.0)

    # --- mean rung lines (bottom right)
    steps = np.arange(0, T_V + 1)
    for kind, label in (("path", "path cell (r%d, c%d)" % pick["path"]),
                        ("dead", "dead-end cell (r%d, c%d)" % pick["dead"])):
        ax_line.plot(steps, mean_size[kind], color=C_GRIDS[kind], lw=2.2,
                     label=label)
        ax_line.plot(steps[-1], mean_size[kind][-1], marker="o", ms=5,
                     color=C_GRIDS[kind], ls="none")
    ax_line.plot([T_star, T_star], [1.0, K + 0.06], color=OI["grey"],
                 ls="--", lw=1.0,
                 label="T* = %d (validation-selected horizon)" % T_star)
    ax_line.set_ylim(1.0, K + 0.06)
    ax_line.set_yticks(range(1, K + 1))
    ax_line.set_xlim(-0.5, T_V + 0.5)
    ax_line.set_xlabel("outer step (t = 0 is the initial state)",
                       fontsize=11.0)
    ax_line.set_ylabel("number-weighted mean rung\n"
                       "(= mean cluster size, 1..%d)" % K, fontsize=11.0)
    ax_line.legend(loc="upper right", fontsize=10.0)

    cell_diff = mean_size["path"][-1] - mean_size["dead"][-1]
    # first step from which the path cell stays above the dead-end cell
    # (mean_size covers t = 0..T_V); drives the honest title wording
    final_cross = None
    if mean_size["path"][-1] > mean_size["dead"][-1]:
        final_cross = T_V
        while final_cross > 0 and \
                mean_size["path"][final_cross - 1] > \
                mean_size["dead"][final_cross - 1]:
            final_cross -= 1
    if cell_diff >= 1.0:
        f2_title = ("The path cell climbs the commitment ladder (%.2f by "
                    "step %d), the dead-end cell stays at the bottom rung "
                    "(%.2f)" % (mean_size["path"][-1], T_V,
                                mean_size["dead"][-1]))
    elif final_cross is None:
        f2_title = ("The dead-end cell stays above the path cell for the "
                    "whole rollout (%.2f vs %.2f at step %d): at %d/%d "
                    "epochs the ladder does not separate the two"
                    % (mean_size["dead"][-1], mean_size["path"][-1], T_V,
                       tr["epochs_done"], cfg["epochs_planned"]))
    else:
        f2_title = ("Path cell overtakes the dead-end cell only from step "
                    "%d, ending %.2f rungs above: at %d/%d epochs the "
                    "separation is still mild"
                    % (final_cross, cell_diff, tr["epochs_done"],
                       cfg["epochs_planned"]))
    headline(fig,
             f2_title,
             "Guide: left - the maze with the two analysed cells outlined; "
             "middle/right - the share of the cell's clusters on each rung "
             "over time (t = 0:\nall mass on rung 1 by initialisation; "
             "colour scale shared, clipped to [0, 1]); bottom - the "
             "number-weighted mean rung of the same two cells.")
    save(fig, "ladder_cell_size_distribution.png", out_dir)

    # ------------------------------------------------------------ outputs
    write_csv(out_dir / "data.csv", rows)
    print("wrote %s (%d rows)" % (out_dir / "data.csv", len(rows)))
    write_readme(out_dir, rows, args, metrics)


if __name__ == "__main__":
    main()
