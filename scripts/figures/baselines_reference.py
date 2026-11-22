"""Baselines reference figure (packet F10): published numbers vs our own
reduced-budget runs of the official baseline code, as a clean table.

Reads the two baseline run records from a baselines directory given on the
command line (--baselines-dir) or via the CF_BASELINES_DIR environment
variable:

  <baselines-dir>/akorn_tools/out/run_record.json
  <baselines-dir>/wonn_tools/out/run_record.json
  <baselines-dir>/wonn_tools/out/train_log.txt
                    (only for the raw parameter count; "not recorded" if absent)

If neither --baselines-dir nor CF_BASELINES_DIR is set, no run records are
read: the figure and data.csv are regenerated from the tracked
figures/baselines_reference/data.csv only (the script says so on stdout);
fields that live only in the run records (commit hashes, export
cross-check) then read "not recorded".

Writes figures/baselines_reference/{baselines_reference.png, data.csv}.

Run (CPU only):
  python scripts/figures/baselines_reference.py --baselines-dir DIR [--out-dir DIR] [--root DIR]
  python scripts/figures/baselines_reference.py          # or CF_BASELINES_DIR=DIR

Published reference numbers are constants below, cited to the tables they
come from; every number of ours is recomputed from the run records at run
time (or, in the fallback without a baselines directory, from the tracked
data.csv). If an OOD T_eval=128 result is added to the AKOrN record later, the
script picks it up automatically on re-run (find_akorn_t128 sweeps the
record for it, skipping paper-reference entries); until then the cell reads
"not run" -- nothing in the figure or the CSV is estimated.

Labels: both our rows are reduced-budget pipeline checks over the official
code, not reproductions. Both are single runs, hence "preliminary" (fewer
than 3 seeds, no seed spread). The layout is checked at render time: no
text box may overlap any other text box or any drawn line/patch, and every
cell must stay inside its column band.
"""

import argparse
import csv
import json
import os
import re
import sys
import textwrap
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import DPI, OKABE_ITO, apply_style  # noqa: E402

F10 = "F10 baselines reference"

# Published reference numbers (constants, cited; we did not re-run these):
# AKOrN paper, Table 19, p. 36 (Sudoku, board accuracy in %, sd as reported
# in the paper): ID 99.8 +- 0.1 and OOD 16.6 at T=16, OOD 51.7 +- 3.3 at
# T_eval=128.
PUB_AKORN_ID_T16 = (99.8, 0.1)
PUB_AKORN_OOD_T16 = (16.6, None)
PUB_AKORN_OOD_T128 = (51.7, 3.3)
# WONN paper, Table 3, p. 9 (maze-hard, exact-board accuracy in %):
# 76.2 at K=1, T=24, model with 0.396 M parameters.
PUB_WONN_BOARD = (76.2, None)
PUB_WONN_PARAMS = "0.396 M"
PUB_AKORN_EPOCHS = "100 (official protocol)"
PUB_WONN_EPOCHS = "9000 (official protocol)"
PUB_HW = "not stated in cited table"
PUB_SEEDS = "sd reported; n not stated"

GREY = OKABE_ITO["grey"]
BLACK = "#333333"
BLUE = OKABE_ITO["blue"]
HAIR = "#B8B8B8"

LEFT_PX = 64.0
RIGHT_PX = 48.0
STRIP_W = 8.0
CELL_PAD = 7.0
ROW_PAD = 8.0
LINE_GAP = 1.42  # line height in font ems

PT2PX = DPI / 72.0


def repo_root():
    return Path(__file__).resolve().parents[2]


def dig(obj, *path):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def find_akorn_t128(record):
    """OOD-at-T=128 result of ours, if the record gains one later.

    Sweeps every dict for a numeric leaf whose own key, or an ancestor key,
    carries both an eval/ood marker and 128. Segments matching
    official|paper|reference are skipped: those are the paper numbers, not
    our run. Returns (value_fraction_or_pct, json_path) or (None, None).
    """
    hits = []

    def num(v):
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def walk(node, path):
        if not isinstance(node, dict):
            return
        for key, val in node.items():
            seg = f"{path}.{key}" if path else key
            if re.search(r"(?i)official|paper|reference", seg):
                continue
            v = num(val)
            low = seg.lower()
            if v is not None and "128" in low and ("ood" in low or "eval" in low):
                hits.append((v, seg))
            elif isinstance(val, dict):
                walk(val, seg)

    walk(record, "")
    if not hits:
        return None, None
    oods = [h for h in hits if "ood" in h[1].lower()]
    value, seg = (oods or hits)[0]
    pct = value * 100.0 if value <= 1.5 else value
    return pct, seg


def wonn_raw_params(train_log):
    if not train_log or not Path(train_log).is_file():
        return None
    text = Path(train_log).read_text(errors="replace")
    match = re.search(r"(?i)total number of parameters:\s*([0-9,]+)", text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def pct1(x):
    """Fraction-or-percent from a run record -> 'NN.N' percent string."""
    if x is None:
        return "not recorded"
    v = x * 100.0 if x <= 1.5 else x
    return f"{v:.1f}"


def pct_s(x):
    return "?" if x is None else f"{x:.1f}"


def build(akorn, wonn, t128, params_raw=None):
    """All figure/CSV content, as plain strings recomputed from the records."""
    ak_hw = (dig(akorn, "gpu") or "").replace("NVIDIA GeForce ", "")
    wn_gpu = dig(wonn, "gpu", "model") or ""
    seed_w = dig(wonn, "seed")
    bs = dig(wonn, "config", "batchsize")
    bs_off = dig(wonn, "config", "official_batchsize")

    ak_id = dig(akorn, "eval", "board_accuracy", "id_T16")
    ak_ood = dig(akorn, "eval", "board_accuracy", "ood_T16")
    ak_raw_id = dig(akorn, "eval", "raw_model_id_T16_from_train_log")
    ak_x_id = dig(akorn, "eval", "export_cross_check", "id_T16")
    ak_x_ood = dig(akorn, "eval", "export_cross_check", "ood_T16")

    wn_board = dig(wonn, "eval", "ema_model", "board_acc")
    wn_f1 = dig(wonn, "eval", "ema_model", "path_f1")
    wn_board_raw = dig(wonn, "eval", "model.pth", "board_acc")
    wn_f1_raw = dig(wonn, "eval", "model.pth", "path_f1")

    ak_hours = f"{dig(akorn, 'training_wall_minutes'):.1f}" if dig(
        akorn, "training_wall_minutes") is not None else "not recorded"
    wn_hours = dig(wonn, "train_wall_minutes")
    wn_hours = f"{wn_hours:.1f}" if wn_hours is not None else "not recorded"

    if params_raw is None:
        params_raw = wonn_raw_params(ARGS.wonn_train_log)

    # ---------------- figure panel A: AKOrN, Sudoku -----------------
    ak_pub_source = (
        f"AKOrN paper, Table 19, p. 36\n"
        f"{PUB_AKORN_EPOCHS} epochs\n"
        f"hardware: {PUB_HW}\n"
        f"seeds: {PUB_SEEDS}"
    )
    ak_pub_config = (
        "official config; board acc in %\n"
        "(exact solved boards);\n"
        "eval at T=16 and T_eval=128"
    )
    ak_ours_source = (
        f"this repo, official code\n"
        f"{dig(akorn, 'epochs_run')}/{dig(akorn, 'official_epochs')} epochs, "
        f"{ak_hours} min\n"
        f"1 run (seed not recorded)\n"
        f"{ak_hw}"
    )
    ak_ours_label_run = (
        f"{dig(akorn, 'epochs_run')}/{dig(akorn, 'official_epochs')} epochs, "
        f"{ak_hours} min, 1 run (seed not recorded)"
    )
    ak_ours_config = (
        "T=16, lr 1e-3, global learnable\n"
        "omega (init 0.5), EMA ckpt,\n"
        "official eval script at T=16;\n"
        "n = 1,000 ID / 18,000 OOD puzzles"
    )
    ak_t128_val = "not run" if t128[0] is None else f"{t128[0]:.1f}"
    title = (
        "Baselines at a reduced budget: AKOrN Sudoku ID "
        f"{pct1(ak_id)} vs published {PUB_AKORN_ID_T16[0]} \u00b1 "
        f"{PUB_AKORN_ID_T16[1]} at T=16 (ours: "
        f"{dig(akorn, 'epochs_run')}/{dig(akorn, 'official_epochs')} epochs);\n"
        f"WONN maze-hard {pct1(wn_board)} vs published {PUB_WONN_BOARD[0]} "
        f"(ours: {dig(wonn, 'epochs_trained')}/{dig(wonn, 'official_epochs')} "
        f"epochs) \u2014 pipeline checks, not reproductions"
    )

    panel_a = dict(
        header="A. AKOrN \u2014 Sudoku (official code, commit "
               f"{dig(akorn, 'commit_hash')[:7]}), full-board accuracy in %",
        cols=[
            ("Source (budget,\nhardware, seeds)", 0.000, 0.235, "left"),
            ("Config /\neval protocol", 0.235, 0.500, "left"),
            ("ID board acc\n@ T=16 (%)", 0.500, 0.625, "center"),
            ("OOD board acc\n@ T=16 (%)", 0.625, 0.750, "center"),
            ("OOD board acc\n@ T_eval=128 (%)", 0.750, 0.875, "center"),
            ("Verdict", 0.875, 1.000, "center"),
        ],
        rows=[
            dict(strip=GREY, cells=[
                ak_pub_source, ak_pub_config,
                f"{PUB_AKORN_ID_T16[0]} \u00b1 {PUB_AKORN_ID_T16[1]}",
                f"{PUB_AKORN_OOD_T16[0]}",
                f"{PUB_AKORN_OOD_T128[0]} \u00b1 {PUB_AKORN_OOD_T128[1]}",
                "reference\n(paper)"],
                styles=["pub", "pub", "val", "val", "val", "ref"]),
            dict(strip=BLUE, cells=[
                ak_ours_source, ak_ours_config,
                f"{pct1(ak_id)}", f"{pct1(ak_ood)}",
                ak_t128_val, "not a\nreproduction"],
                styles=["ours", "ours", "val", "val",
                        ("missing" if t128[0] is None else "val"), "verdict"]),
        ])

    # ---------------- figure panel B: WONN, maze-hard -----------------
    wn_pub_source = (
        f"WONN paper, Table 3, p. 9\n"
        f"{PUB_WONN_EPOCHS} epochs\n"
        f"hardware: {PUB_HW}\n"
        f"seeds: {PUB_SEEDS}"
    )
    wn_pub_config = (
        "K=1, T=24; board acc =\n"
        "exact full-grid match\n"
        "(not any valid shortest\n"
        "path); test split"
    )
    wn_ours_source = (
        f"this repo, official code\n"
        f"{dig(wonn, 'epochs_trained')}/{dig(wonn, 'official_epochs')} epochs, "
        f"{wn_hours} min\n"
        f"1 run, seed {seed_w}\n"
        f"batch {bs} (official {bs_off})\n"
        f"{wn_gpu}"
    )
    wn_ours_label_run = (
        f"{dig(wonn, 'epochs_trained')}/{dig(wonn, 'official_epochs')} epochs, "
        f"{wn_hours} min, 1 run, seed {seed_w}, batch {bs} (official {bs_off})"
    )
    wn_ours_config = (
        "L=1, T=24, ch=128, heads 8,\n"
        "gamma 0.1, GroupNorm, bf16 AMP,\n"
        "EMA ckpt; official eval_maze.py\n"
        "on test (n = 1,000 mazes)"
    )
    params_cell = (
        "396,293 raw\n(effective\nnot measured)"
        if params_raw is not None else "not recorded"
    )
    panel_b = dict(
        header="B. WONN \u2014 maze-30x30-hard-1k (official code, commit "
               f"{dig(wonn, 'commit_hash')}), metrics in %",
        cols=[
            ("Source (budget,\nhardware, seeds)", 0.000, 0.245, "left"),
            ("Config /\neval protocol", 0.245, 0.505, "left"),
            ("Board acc\n@ K=1, T=24 (%)", 0.505, 0.635, "center"),
            ("Path F1\n@ T=24 (%)", 0.635, 0.750, "center"),
            ("Params", 0.750, 0.875, "center"),
            ("Verdict", 0.875, 1.000, "center"),
        ],
        rows=[
            dict(strip=GREY, cells=[
                wn_pub_source, wn_pub_config,
                f"{PUB_WONN_BOARD[0]}", "not stated\nin table",
                f"{PUB_WONN_PARAMS}\n(raw, paper)", "reference\n(paper)"],
                styles=["pub", "pub", "val", "missing", "pub", "ref"]),
            dict(strip=BLUE, cells=[
                wn_ours_source, wn_ours_config,
                f"{pct1(wn_board)}", f"{pct1(wn_f1)}",
                params_cell, "pipeline check\nonly, not a\nreproduction"],
                styles=["ours", "ours", "val", "val", "ours", "verdict"]),
        ])

    foot = [
        "Published \u00b1 values are the sd reported by the papers (n not "
        "stated in the cited tables); our numbers are single runs (AKOrN: "
        "seed not recorded; WONN: seed "
        f"{seed_w}) \u2014 preliminary, no seed spread, small differences "
        "are inconclusive.",
        f"AKOrN ours: EMA checkpoint, official eval script; raw (non-EMA) "
        f"model ID {pct1(ak_raw_id)} at T=16 (training log); export "
        f"cross-check {pct1(ak_x_id)} ID / {pct1(ak_x_ood)} OOD.",
        "WONN ours: exact full-grid match required, so path F1 "
        f"{pct1(wn_f1)} with {pct1(wn_board)} board acc; non-EMA model.pth "
        f"reached path F1 {pct1(wn_f1_raw)}, board still "
        f"{pct1(wn_board_raw)}; params "
        + (f"{params_raw:,} raw" if params_raw is not None
           else "not recorded") + ", effective not measured.",
        "Hardware (our runs): NVIDIA RTX 3070 Ti, 8 GB; reduced-budget "
        "checks of the official code (AKOrN commit "
        f"{dig(akorn, 'commit_hash')[:7]}, WONN commit "
        f"{dig(wonn, 'commit_hash')}) \u2014 not reproductions.",
        "'not run' or 'not stated in table' = the number was never measured "
        "by us / not given in the cited paper table; nothing in the figure "
        "is estimated.",
    ]

    # ---------------- data.csv rows -----------------
    def csvrow(block, source, metric, val, sd, run, src):
        return dict(block=block, source=source, metric=metric,
                    value_percent=val, sd_percent=sd or "", budget=run,
                    hardware=("RTX 3070 Ti (8 GB), this repo" if source ==
                              "ours" else PUB_HW), data_source=src)

    ak_pub_run = f"{PUB_AKORN_EPOCHS} epochs; not trained by us"
    wn_pub_run = f"{PUB_WONN_EPOCHS} epochs; not trained by us"
    csv_rows = [
        csvrow("akorn_sudoku", "published", "ID board acc @ T=16",
               f"{PUB_AKORN_ID_T16[0]}", f"{PUB_AKORN_ID_T16[1]}", ak_pub_run,
               "AKOrN paper, Table 19, p. 36"),
        csvrow("akorn_sudoku", "published", "OOD board acc @ T=16",
               f"{PUB_AKORN_OOD_T16[0]}", "", ak_pub_run,
               "AKOrN paper, Table 19, p. 36"),
        csvrow("akorn_sudoku", "published", "OOD board acc @ T_eval=128",
               f"{PUB_AKORN_OOD_T128[0]}", f"{PUB_AKORN_OOD_T128[1]}",
               ak_pub_run, "AKOrN paper, Table 19, p. 36"),
        csvrow("akorn_sudoku", "ours", "ID board acc @ T=16", pct1(ak_id),
               "", ak_ours_label_run + ", preliminary",
               "akorn_tools/out/run_record.json"),
        csvrow("akorn_sudoku", "ours", "OOD board acc @ T=16", pct1(ak_ood),
               "", ak_ours_label_run + ", preliminary",
               "akorn_tools/out/run_record.json"),
        csvrow("akorn_sudoku", "ours", "OOD board acc @ T_eval=128",
               (t128[0] and f"{t128[0]:.1f}") or "not run", "",
               ak_ours_label_run + ", preliminary",
               "akorn_tools/out/run_record.json"),
        csvrow("akorn_sudoku", "ours", "raw (non-EMA) ID board acc @ T=16",
               pct1(ak_raw_id), "", ak_ours_label_run,
               "akorn_tools/out/run_record.json (training log value)"),
        csvrow("wonn_maze_hard", "published", "exact-board acc @ K=1, T=24",
               f"{PUB_WONN_BOARD[0]}", "", wn_pub_run,
               "WONN paper, Table 3, p. 9"),
        csvrow("wonn_maze_hard", "published", "params",
               PUB_WONN_PARAMS + " (raw, paper)", "", wn_pub_run,
               "WONN paper, Table 3, p. 9"),
        csvrow("wonn_maze_hard", "ours", "exact-board acc @ K=1, T=24",
               pct1(wn_board), "", wn_ours_label_run + ", preliminary",
               "wonn_tools/out/run_record.json"),
        csvrow("wonn_maze_hard", "ours", "path F1 @ K=1, T=24",
               pct1(wn_f1), "", wn_ours_label_run + ", preliminary",
               "wonn_tools/out/run_record.json"),
        csvrow("wonn_maze_hard", "ours",
               "exact-board acc @ K=1, T=24 (non-EMA model.pth)",
               pct1(wn_board_raw), "", wn_ours_label_run,
               "wonn_tools/out/run_record.json"),
        csvrow("wonn_maze_hard", "ours",
               "path F1 @ K=1, T=24 (non-EMA model.pth)",
               pct1(wn_f1_raw), "", wn_ours_label_run,
               "wonn_tools/out/run_record.json"),
        csvrow("wonn_maze_hard", "ours", "params raw",
               (params_raw is not None and f"{params_raw:,}")
               or "not recorded",
               "", wn_ours_label_run,
               "wonn_tools/out/train_log.txt; effective not measured"),
    ]

    return title, [panel_a, panel_b], foot, csv_rows


# --------------------------------------------------------------------------
# rendering helpers (pure figure coordinates, pixel-measured checks)
# --------------------------------------------------------------------------

class Canvas:
    """Figure-coordinate painter with a full overlap self-check."""

    def __init__(self, fig_w_in, fig_h_in):
        self.w = fig_w_in * DPI
        self.h = fig_h_in * DPI
        self.fig = plt.figure(figsize=(fig_w_in, fig_h_in))
        self.texts = []   # [(artist, column_band_or_None)]
        self.geoms = []   # sampled artists (lines, patches), never under text

    def text(self, x_px, y_top_px, s, size=11, color=BLACK, weight="normal",
             style="normal", ha="left", band=None, linespacing=LINE_GAP):
        t = self.fig.text(
            x_px / self.w, 1.0 - y_top_px / self.h, s, fontsize=size,
            color=color, fontweight=weight, fontstyle=style, ha=ha, va="top",
            linespacing=linespacing)
        self.texts.append((t, band))
        return t

    def hline(self, x0_px, x1_px, y_px, color=HAIR, lw=1.0, alpha=1.0):
        self._line((x0_px, y_px), (x1_px, y_px), color, lw, alpha)

    def rect(self, x_px, y_top_px, w_px, h_px, color):
        r = Rectangle((x_px / self.w, 1.0 - (y_top_px + h_px) / self.h),
                      w_px / self.w, h_px / self.h, transform=
                      self.fig.transFigure, facecolor=color, edgecolor="none")
        self.fig.add_artist(r)
        self.geoms.append(("patch", r))
        return r

    def _line(self, p0, p1, color, lw, alpha):
        from matplotlib.lines import Line2D
        ln = Line2D([p0[0] / self.w, p1[0] / self.w],
                    [1.0 - p0[1] / self.h, 1.0 - p1[1] / self.h],
                    transform=self.fig.transFigure, color=color,
                    linewidth=lw, alpha=alpha)
        self.fig.add_artist(ln)
        self.geoms.append(("line", ln, p0, p1))

    # -- geometry sampling in display pixels, like the shared helpers --
    def _samples(self):
        pts = []
        fig = self.fig
        scale = DPI / fig.dpi
        for g in self.geoms:
            if g[0] == "line":
                _, ln, p0, p1 = g
                steps = 60
                for i in range(steps + 1):
                    f = i / steps
                    x = p0[0] + (p1[0] - p0[0]) * f
                    y = p0[1] + (p1[1] - p0[1]) * f
                    # stored as (x from left, y from top) in DPI px ->
                    # display px (y from bottom) at the current draw dpi
                    pts.append((x * scale, (self.h - y) * scale))
            else:
                patch = g[1]
                tr = patch.get_transform()
                for poly in patch.get_path().to_polygons():
                    for x, y in poly:
                        dx, dy = tr.transform((x, y))
                        pts.append((dx * scale, dy * scale))
        return [(float(x), float(y)) for x, y in pts]

    def check(self, name, pad=1.25):
        """Assert: no two text boxes overlap; no text box covers a line or
        patch sample; every cell stays inside its column band."""
        self.fig.canvas.draw()
        ren = self.fig.canvas.get_renderer()
        boxes = [(t.get_text().split("\n")[0][:26], t.get_window_extent(ren))
                 for t, _ in self.texts]
        for (t1, b1), (t2, b2) in combinations(boxes, 2):
            e1 = Bbox.from_bounds(b1.x0 - pad, b1.y0 - pad,
                                  b1.width + 2 * pad, b1.height + 2 * pad)
            if e1.overlaps(b2):
                raise AssertionError(
                    f"{name}: text boxes overlap: {t1!r} vs {t2!r} "
                    f"({b1.bounds} vs {b2.bounds})")
        pad_px = pad
        for gx, gy in self._samples():
            for txt, bb in boxes:
                if (bb.x0 - pad_px <= gx <= bb.x1 + pad_px
                        and bb.y0 - pad_px <= gy <= bb.y1 + pad_px):
                    raise AssertionError(
                        f"{name}: text {txt!r} (bbox {bb.bounds}) sits on "
                        f"drawn line/patch at ({gx:.1f}, {gy:.1f})")
        for t, band in self.texts:
            if band is None:
                continue
            bb = t.get_window_extent(ren)
            x0, x1 = band
            if bb.x0 < x0 - 2.0 or bb.x1 > x1 + 2.0:
                raise AssertionError(
                    f"{name}: cell {t.get_text()[:26]!r} "
                    f"(bbox {bb.bounds}) escapes its column band "
                    f"({x0:.0f}, {x1:.0f})")
        return self


def lh_pt(size_pt):
    """Line advance in px: matplotlib line box plus a small real gap, so
    consecutive text lines never touch (checked at render time)."""
    return size_pt * PT2PX * LINE_GAP + 3.0


def nl(s):
    return s.count("\n") + 1


# --------------------------------------------------------------------------
# figure layout
# --------------------------------------------------------------------------

def render(title, panels, foot, out_png):
    fig_w, fig_h = 13.6, 9.0
    c = Canvas(fig_w, fig_h)
    W, H = c.w, c.h
    cx0 = LEFT_PX
    cw = W - LEFT_PX - RIGHT_PX

    y = 16.0
    for line in title.split("\n"):
        c.text(W / 2, y, line, size=15, weight="bold", ha="center")
        y += lh_pt(15)
    y += 4

    guide = [
        "Reading guide: one row per number source \u2014 published value "
        "(grey strip) above, our run (blue strip) below.",
        "All numbers are exact full-board accuracy in %; each row carries "
        "its budget (epochs, minutes, seeds, hardware).",
    ]
    for line in guide:
        c.text(W / 2, y, line, size=10.5, color="#555555", ha="center")
        y += lh_pt(10.5)
    y += 10

    # legend, top-right, clear of the centred reading guide
    leg_x = W - RIGHT_PX - 470
    leg_y = 16.0 + lh_pt(15) * 2 + 8
    sw_w, sw_h = 30.0, 14.0
    c.rect(leg_x, leg_y, sw_w, sw_h, GREY)
    c.text(leg_x + sw_w + 10, leg_y - 1, "published number (paper)",
           size=11, color="#555555", band=(leg_x + sw_w + 10, W - RIGHT_PX + 2))
    leg2_y = leg_y + lh_pt(11) + 6
    c.rect(leg_x, leg2_y, sw_w, sw_h, BLUE)
    c.text(leg_x + sw_w + 10, leg2_y - 1,
           "our reduced-budget run", size=11, color="#555555",
           band=(leg_x + sw_w + 10, W - RIGHT_PX + 2))

    strip_x = LEFT_PX - 18.0
    cell_x0 = LEFT_PX + STRIP_W + 12.0

    for panel in panels:
        # panel header
        c.text(cx0, y, panel["header"], size=13, weight="bold")
        y += lh_pt(13) + 6

        # column headers
        hdr_top = y
        wrapped_hdrs = []
        for label, f0, f1, align in panel["cols"]:
            band = (cx0 + f0 * cw, cx0 + f1 * cw)
            wrapped = label  # headers are pre-wrapped to fit their band
            wrapped_hdrs.append(wrapped)
            x = band[0] if align == "left" else (band[0] + band[1]) / 2
            ha = "left" if align == "left" else "center"
            c.text(x, hdr_top, wrapped,
                   size=11, weight="semibold", color="#555555", ha=ha,
                   band=band)
        n_hdr = max(nl(s) for s in wrapped_hdrs)
        y = hdr_top + n_hdr * lh_pt(11) + 5
        c.hline(cx0, cx0 + cw, y, color="#888888", lw=1.1)
        y += 1

        # rows
        for r_idx, row in enumerate(panel["rows"]):
            if r_idx > 0:
                c.hline(cx0, cx0 + cw, y, color=HAIR, lw=0.7, alpha=0.6)
                y += 1
            row_top = y + ROW_PAD
            n_lines = max(nl(s) for s in row["cells"])
            row_h = n_lines * lh_pt(11) + 2 * ROW_PAD
            c.rect(strip_x, row_top, STRIP_W, row_h - 2 * ROW_PAD,
                   row["strip"])

            styles = row["styles"]
            bands = [(cx0 + f0 * cw, cx0 + f1 * cw)
                     for _, f0, f1, _ in panel["cols"]]
            aligns = [a for _, _, _, a in panel["cols"]]
            for s, st, (b0, b1), align in zip(row["cells"], styles, bands,
                                              aligns):
                full_band = (b0 + (0 if align == "left" else 0), b1)
                txt_style = dict(
                    pub=dict(size=11, color=BLACK),
                    ours=dict(size=11, color=BLACK),
                    val=dict(size=12.5, color=BLACK),
                    missing=dict(size=11.5, color="#777777",
                                 style="italic"),
                    ref=dict(size=11.5, color="#777777"),
                    verdict=dict(size=11.5, color=BLUE, weight="bold"),
                )[st]
                if st in ("val", "missing") and s == "not run":
                    txt_style = dict(size=11.5, color="#777777",
                                     style="italic")
                if st == "val" and row["strip"] == BLUE:
                    txt_style = dict(size=12.5, color=BLUE, weight="bold")
                x = b0 if align == "left" else (b0 + b1) / 2
                c.text(x, row_top, s, ha=align,
                       band=(b0 - 2, b1 + 2), **txt_style)
            y = row_top + row_h - ROW_PAD + 4
        y += 2
        c.hline(cx0, cx0 + cw, y, color="#888888", lw=1.1)
        y += 20

    # footnotes and data-source line
    for line in foot:
        wrapped = "\n".join(textwrap.wrap(line, 178))
        c.text(LEFT_PX, y, wrapped, size=10, color="#666666")
        y += nl(wrapped) * lh_pt(10) + 4

    c.check(F10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    c.fig.savefig(out_png, dpi=DPI)
    print(f"wrote {out_png}")


def records_from_csv(csv_path):
    """Fallback without a baselines directory: rebuild record-like dicts
    from the tracked data.csv so the figure can be regenerated from the
    tracked data alone. Fields that live only in the run records (commit
    hashes, AKOrN export cross-check) come back as "not recorded"."""
    if not csv_path.is_file():
        raise SystemExit(
            f"{csv_path} not found and no baselines directory given: pass "
            "--baselines-dir (or set CF_BASELINES_DIR)")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def val(block, metric):
        for r in rows:
            if (r["block"], r["source"], r["metric"]) == (block, "ours", metric):
                return r["value_percent"]
        return None

    def budget(block):
        for r in rows:
            if r["block"] == block and r["source"] == "ours":
                return r["budget"]
        return ""

    def num(s):
        if s is None:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def epochs_min(block):
        m = re.match(r"(\d+)/(\d+) epochs, ([0-9.]+) min", budget(block))
        return int(m.group(1)), int(m.group(2)), float(m.group(3))

    def hw():
        for r in rows:
            if r["source"] == "ours":
                return r["hardware"].split(" (")[0]
        return ""

    ak_epochs, ak_off, ak_min = epochs_min("akorn_sudoku")
    ak_gpu = hw()
    akorn = dict(
        gpu=f"NVIDIA GeForce {ak_gpu}",
        epochs_run=ak_epochs, official_epochs=ak_off,
        training_wall_minutes=ak_min,
        commit_hash="n/a (run record not given)",
        eval=dict(
            board_accuracy=dict(
                id_T16=num(val("akorn_sudoku", "ID board acc @ T=16")),
                ood_T16=num(val("akorn_sudoku", "OOD board acc @ T=16")),
            ),
            raw_model_id_T16_from_train_log=num(
                val("akorn_sudoku", "raw (non-EMA) ID board acc @ T=16")),
            export_cross_check=dict(id_T16=None, ood_T16=None),
        ),
    )

    wn_budget = budget("wonn_maze_hard")
    wn_epochs, wn_off, wn_min = epochs_min("wonn_maze_hard")
    seed_m = re.search(r"seed (\d+)", wn_budget)
    batch_m = re.search(r"batch (\d+) \(official (\d+)\)", wn_budget)
    wonn = dict(
        gpu=dict(model=hw()),
        seed=int(seed_m.group(1)) if seed_m else None,
        config=dict(
            batchsize=int(batch_m.group(1)) if batch_m else None,
            official_batchsize=int(batch_m.group(2)) if batch_m else None,
        ),
        epochs_trained=wn_epochs, official_epochs=wn_off,
        train_wall_minutes=wn_min,
        commit_hash="n/a (run record not given)",
        eval=dict(
            ema_model=dict(
                board_acc=num(val("wonn_maze_hard",
                                  "exact-board acc @ K=1, T=24")),
                path_f1=num(val("wonn_maze_hard", "path F1 @ K=1, T=24")),
            ),
            **{"model.pth": dict(
                board_acc=num(val(
                    "wonn_maze_hard",
                    "exact-board acc @ K=1, T=24 (non-EMA model.pth)")),
                path_f1=num(val(
                    "wonn_maze_hard",
                    "path F1 @ K=1, T=24 (non-EMA model.pth)")),
            )},
        ),
    )

    t128_val = num(val("akorn_sudoku", "OOD board acc @ T_eval=128"))
    t128 = (t128_val, "data.csv") if t128_val is not None else (None, None)

    params_raw = None
    p = val("wonn_maze_hard", "params raw")
    if p and re.fullmatch(r"[0-9,]+", p):
        params_raw = int(p.replace(",", ""))

    return akorn, wonn, t128, params_raw


def main(argv=None):
    global ARGS
    ap = argparse.ArgumentParser(
        description="F10 published-vs-ours baselines table figure.")
    ap.add_argument(
        "--baselines-dir", type=Path, default=None,
        help="Directory holding akorn_tools/out/run_record.json and "
             "wonn_tools/out/run_record.json; also read from the "
             "CF_BASELINES_DIR environment variable. Without it, the "
             "figure is regenerated from the tracked data.csv only.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None)
    ARGS = ap.parse_args(argv)
    baselines_dir = ARGS.baselines_dir or os.environ.get("CF_BASELINES_DIR")

    root = ARGS.root if ARGS.root else repo_root()
    out_dir = ARGS.out_dir if ARGS.out_dir else root / "figures" / \
        "baselines_reference"

    params_from_csv = None
    if baselines_dir:
        akorn_record = Path(baselines_dir) / "akorn_tools" / "out" / \
            "run_record.json"
        wonn_record = Path(baselines_dir) / "wonn_tools" / "out" / \
            "run_record.json"
        ARGS.wonn_train_log = wonn_record.parent / "train_log.txt"
        akorn = json.loads(akorn_record.read_text())
        wonn = json.loads(wonn_record.read_text())
        t128 = find_akorn_t128(akorn)
        if t128[0] is not None:
            print(f"AKOrN T_eval=128 result found in record at {t128[1]}: "
                  f"{t128[0]:.1f}%")
        else:
            print("AKOrN T_eval=128: not present in record -> 'not run'")
    else:
        print("--baselines-dir not given and CF_BASELINES_DIR not set: "
              "regenerating from the tracked "
              "figures/baselines_reference/data.csv only; record-only "
              "fields (commit hashes, export cross-check) read "
              "'not recorded'.")
        ARGS.wonn_train_log = None
        akorn, wonn, t128, params_from_csv = records_from_csv(
            out_dir / "data.csv")

    title, panels, foot, csv_rows = build(akorn, wonn, t128,
                                          params_raw=params_from_csv)

    render(title, panels, foot, out_dir / "baselines_reference.png")

    csv_path = out_dir / "data.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    apply_style()
    main()
