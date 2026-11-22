"""Baselines reference figure (packet F10): published numbers vs our own
reduced-budget runs of the official baseline code, as a clean table.

Reads the two baseline run records from a baselines directory given on the
command line (--baselines-dir) or via the CF_BASELINES_DIR environment
variable:

  <baselines-dir>/akorn_tools/out/run_record.json
  <baselines-dir>/wonn_tools/out/run_record.json
  <baselines-dir>/wonn_tools/out/train_log.txt
                    (only for the raw parameter count; "not recorded" if absent)

Also read when present (relative to the baselines directory):
  akorn_tools/out/eval_100ep_T16.log, akorn_tools/out/eval_100ep_T128.log
                    (the re-run official evals the record points to via
                    eval.full_stdout_logs; re-read as a cross-check and
                    quoted in the unseeded re-eval footnote)
  akorn_tools/out/_export_id.log, akorn_tools/out/_export_ood.log
                    (exact-board accuracy of the exported prediction
                    grids; the export cross-check of the final EMA
                    checkpoint)
  akorn_tools/out/train_reduced.log
                    (surviving training log; its last epoch and final
                    raw/EMA test accuracies give the provenance of the
                    raw (non-EMA) ID value in the run record)
The run record quotes the superseded earlier-revision eval values in its
notes; those are parsed for the re-eval-drift footnote (unseeded GPU
re-evals of the same checkpoint are nondeterministic).

If neither --baselines-dir nor CF_BASELINES_DIR is set, no run records are
read: the figure and data.csv are regenerated from the tracked
figures/baselines_reference/data.csv only (the script says so on stdout);
fields that live only in the run records (commit hashes, export
cross-check) then read "not recorded", while the re-eval and surviving-log
numbers fall back to their data.csv rows (written there on the last
record-based run), so the footnotes regenerate.

Writes figures/baselines_reference/{baselines_reference.png, data.csv}.

Run (CPU only):
  python scripts/figures/baselines_reference.py --baselines-dir DIR [--out-dir DIR] [--root DIR]
  python scripts/figures/baselines_reference.py          # or CF_BASELINES_DIR=DIR

Published reference numbers are constants below, cited to the tables they
come from; every number of ours is recomputed from the run records at run
time (or, in the fallback without a baselines directory, from the tracked
data.csv). Both AKOrN record layouts are supported: the current one holds
eval.board_accuracy_T16 and eval.board_accuracy_T_eval_128 (older ones used
eval.board_accuracy); an OOD number missing from those is swept for by
find_akorn_t128 (skipping paper-reference entries) and otherwise reads "not
run" -- nothing in the figure or the CSV is estimated.

Labels: our AKOrN row carries the run's own budget. When the record shows
the official config run to the full official epoch count (epochs_run >=
official_epochs), the verdict reads "full official budget, 1 seed
(preliminary)" -- the record states a single unseeded seed, so it stays
preliminary; below the full count the verdict stays "not a reproduction".
The WONN row is a reduced-budget pipeline check and stays "not a
reproduction". Both are single runs, hence "preliminary" (fewer than 3
seeds, no seed spread). The layout is checked at render time: no text box
may overlap any other text box or any drawn line/patch, every cell must
stay inside its column band, and the content must fit the canvas height.
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


def dig_any(obj, *paths):
    """First non-None dig() over several alternative key paths."""
    for path in paths:
        v = dig(obj, *path)
        if v is not None:
            return v
    return None


def to_pct(v):
    """Record fraction-or-percent leaf -> percent float, or None."""
    if v is None:
        return None
    return v * 100.0 if v <= 1.5 else v


def akorn_t128_id(akorn):
    """Our ID board accuracy at T_eval=128 in %, or None ("not run")."""
    return to_pct(dig_any(
        akorn,
        ("eval", "board_accuracy", "id_T128"),
        ("eval", "board_accuracy_T_eval_128", "id_T128"),
    ))


def akorn_t128_ood(akorn):
    """Our OOD board accuracy at T_eval=128 in %, or None ("not run")."""
    v = dig_any(
        akorn,
        ("eval", "board_accuracy", "ood_T128"),
        ("eval", "board_accuracy_T_eval_128", "ood_T128"),
    )
    if v is not None:
        return to_pct(v)
    return find_akorn_t128(akorn)[0]


def wonn_raw_params(train_log):
    if not train_log or not Path(train_log).is_file():
        return None
    text = Path(train_log).read_text(errors="replace")
    match = re.search(r"(?i)total number of parameters:\s*([0-9,]+)", text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def akorn_surviving_logs(baselines_dir):
    """Surviving AKOrN logs next to the run record.

    Returns id16/ood16/id128/ood128: the Vote accuracy values of the
    re-run official evals (first block = ID, second = OOD of each
    eval_100ep_T*.log), exp_id/exp_ood (n/tot): the exact-board accuracy
    and grid counts of the prediction-grid export re-checks
    (_export_id.log / _export_ood.log), and log_epoch/log_raw/log_ema:
    the last epoch and the final raw/EMA test accuracies of
    train_reduced.log. Each key reads None when its file is missing or
    holds no match -- nothing is estimated."""
    sv = dict(id16=None, ood16=None, id128=None, ood128=None,
              exp_id=None, exp_id_n=None, exp_id_tot=None,
              exp_ood=None, exp_ood_n=None, exp_ood_tot=None,
              log_epoch=None, log_raw=None, log_ema=None)
    out_dir = Path(baselines_dir) / "akorn_tools" / "out"
    for fname, keys in (("eval_100ep_T16.log", ("id16", "ood16")),
                        ("eval_100ep_T128.log", ("id128", "ood128"))):
        f = out_dir / fname
        if not f.is_file():
            continue
        accs = re.findall(r"Vote accuracy:\s*([01]\.\d+)",
                          f.read_text(errors="replace"))
        for key, acc in zip(keys, accs):
            sv[key] = float(acc)
    for fname, vkey, nkey, tkey in (("_export_id.log", "exp_id",
                                     "exp_id_n", "exp_id_tot"),
                                    ("_export_ood.log", "exp_ood",
                                     "exp_ood_n", "exp_ood_tot")):
        f = out_dir / fname
        if not f.is_file():
            continue
        m = re.search(r"board_acc = ([01]\.\d+) \((\d+)/(\d+)\)",
                      f.read_text(errors="replace"))
        if m:
            sv[vkey] = float(m.group(1))
            sv[nkey] = int(m.group(2))
            sv[tkey] = int(m.group(3))
    f = out_dir / "train_reduced.log"
    if f.is_file():
        text = f.read_text(errors="replace")
        eps = re.findall(r"Epoch \[(\d+)/", text)
        raw = re.findall(r"\[Test\]: Total boards:\d+, Accuracy: ([01]\.\d+)",
                         text)
        ema = re.findall(
            r"\[EMA Test\]: Total boards:\d+, Accuracy: ([01]\.\d+)", text)
        if eps:
            sv["log_epoch"] = int(eps[-1])
        if raw:
            sv["log_raw"] = float(raw[-1])
        if ema:
            sv["log_ema"] = float(ema[-1])
    return sv


def akorn_prior_revision(record):
    """Earlier-record eval values quoted in the refreshed run record's
    eval notes ("where the earlier record differed (...), the logs
    win"), or all None when the record quotes no such sentence."""
    out = dict(pr_id16=None, pr_ood16=None, pr_ood128=None)
    strings = []
    for holder in (dig(record, "eval", "notes"), dig(record, "notes")):
        if isinstance(holder, str):
            strings.append(holder)
        elif isinstance(holder, list):
            strings += [s for s in holder if isinstance(s, str)]
    text = " ".join(strings)
    m = re.search(r"earlier record differed \(id_T16 ([01]\.\d+), "
                  r"ood_T16 ([01]\.\d+), ood_T128 ([01]\.\d+)\)", text)
    if m:
        out["pr_id16"] = float(m.group(1))
        out["pr_ood16"] = float(m.group(2))
        out["pr_ood128"] = float(m.group(3))
    return out


def pct1(x):
    """Fraction-or-percent from a run record -> 'NN.N' percent string."""
    if x is None:
        return "not recorded"
    v = x * 100.0 if x <= 1.5 else x
    return f"{v:.1f}"


def pct_s(x):
    return "?" if x is None else f"{x:.1f}"


def build(akorn, wonn, t128, params_raw=None, surviving=None):
    """All figure/CSV content, as plain strings recomputed from the records.

    t128 = (id_pct, ood_pct): our AKOrN board accuracies at T_eval=128 in
    percent, either value None when the record does not hold it ("not run").
    surviving = akorn_surviving_logs() (or the same keys read back from
    data.csv): post-record re-eval values of the final EMA checkpoint and
    the tail of the surviving training log, for the drift footnote and the
    provenance note on the raw (non-EMA) ID value.
    """
    ak_id128, ak_ood128 = t128
    sv = dict(id16=None, ood16=None, id128=None, ood128=None,
              exp_id=None, exp_id_n=None, exp_id_tot=None,
              exp_ood=None, exp_ood_n=None, exp_ood_tot=None,
              log_epoch=None, log_raw=None, log_ema=None,
              pr_id16=None, pr_ood16=None, pr_ood128=None)
    if surviving:
        for k, v in surviving.items():
            if v is not None:
                sv[k] = v
    ak_hw = (dig(akorn, "gpu") or "").replace("NVIDIA GeForce ", "")
    wn_gpu = dig(wonn, "gpu", "model") or ""
    seed_w = dig(wonn, "seed")
    bs = dig(wonn, "config", "batchsize")
    bs_off = dig(wonn, "config", "official_batchsize")

    ak_id = dig_any(akorn, ("eval", "board_accuracy", "id_T16"),
                    ("eval", "board_accuracy_T16", "id_T16"))
    ak_ood = dig_any(akorn, ("eval", "board_accuracy", "ood_T16"),
                     ("eval", "board_accuracy_T16", "ood_T16"))
    ak_raw_id = dig_any(
        akorn,
        ("eval", "raw_model_id_T16_from_train_log"),
        ("eval", "raw_model_id_T16_from_train_log_epoch_100"),
    )
    # only the stale records carried a prediction-export cross-check; the
    # 100-epoch record exports no predictions for the final checkpoint
    ak_x_id = dig(akorn, "eval", "export_cross_check", "id_T16")
    ak_x_ood = dig(akorn, "eval", "export_cross_check", "ood_T16")
    ak_pub_id128 = dig(akorn, "eval", "official_paper_reference", "id")

    wn_board = dig(wonn, "eval", "ema_model", "board_acc")
    wn_f1 = dig(wonn, "eval", "ema_model", "path_f1")
    wn_board_raw = dig(wonn, "eval", "model.pth", "board_acc")
    wn_f1_raw = dig(wonn, "eval", "model.pth", "path_f1")

    ak_min = dig(akorn, "training_wall_minutes")
    ak_cont_min = dig(akorn, "continuation_wall_minutes")
    if ak_min is not None and ak_cont_min is not None:
        ak_hours = f"{ak_min + ak_cont_min:.1f}"
    elif ak_min is not None:
        ak_hours = f"{ak_min:.1f}"
    else:
        ak_hours = "not recorded"
    wn_hours = dig(wonn, "train_wall_minutes")
    wn_hours = f"{wn_hours:.1f}" if wn_hours is not None else "not recorded"

    ak_seeds_info = dig(akorn, "seeds")
    if isinstance(ak_seeds_info, dict) and ak_seeds_info.get("seeds_run"):
        ak_seed_n = ak_seeds_info["seeds_run"]
        ak_seed_lbl = f"{ak_seed_n} run (unseeded)"
    else:
        ak_seed_n = 1
        ak_seed_lbl = "1 run (seed not recorded)"
    ak_run = dig(akorn, "epochs_run")
    ak_off = dig(akorn, "official_epochs")
    # drop "not a reproduction" only when the record shows the official
    # config trained to the full official epoch count (single seed recorded,
    # and the figure says so)
    ak_full = (isinstance(ak_run, int) and isinstance(ak_off, int)
               and ak_run >= ak_off and bool(dig(akorn, "official_command")))

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
        f"{ak_run}/{ak_off} epochs, {ak_hours} min\n"
        f"{ak_seed_lbl}\n"
        f"{ak_hw}"
    )
    ak_ours_label_run = (
        f"{ak_run}/{ak_off} epochs, {ak_hours} min, {ak_seed_lbl}"
    )
    ak_ours_config = (
        "T=16, lr 1e-3, global learnable\n"
        "omega (init 0.5), EMA ckpt,\n"
        "official eval at T=16 and\n"
        "T_eval=128; n = 1,000 ID /\n"
        "18,000 OOD puzzles"
    )
    ak_t128_val = "not run" if ak_ood128 is None else f"{ak_ood128:.1f}"
    ak_verdict = (
        f"full official\nbudget, {ak_seed_n} seed\n(preliminary)"
        if ak_full else "not a\nreproduction")
    ak_budget_word = "full" if ak_full else "reduced"
    if ak_ood128 is None:
        title_l1 = (
            f"AKOrN Sudoku, {ak_budget_word} {ak_run}/{ak_off} epochs, "
            f"single seed: OOD at T_eval=128 not run;")
    else:
        within = (abs(ak_ood128 - PUB_AKORN_OOD_T128[0])
                  <= PUB_AKORN_OOD_T128[1])
        verdict_word = "matches" if within else "stays below"
        title_l1 = (
            f"AKOrN Sudoku, {ak_budget_word} {ak_run}/{ak_off} epochs, "
            f"single seed: OOD at T_eval=128 {ak_ood128:.1f} vs published "
            f"{PUB_AKORN_OOD_T128[0]} \u00b1 {PUB_AKORN_OOD_T128[1]} "
            f"\u2014 {verdict_word};")
    title_l2 = (
        f"WONN maze-hard at {dig(wonn, 'epochs_trained')}/"
        f"{dig(wonn, 'official_epochs')} epochs: board acc "
        f"{pct1(wn_board)} vs published {PUB_WONN_BOARD[0]} at K=1, "
        f"T=24 \u2014 pipeline check only")
    title = title_l1 + "\n" + title_l2

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
                ak_t128_val, ak_verdict],
                styles=["ours", "ours", "val", "val",
                        ("missing" if ak_ood128 is None else "val"),
                        "verdict"]),
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

    if ak_x_id is not None and ak_x_ood is not None:
        ak_x_txt = (f"export cross-check {pct1(ak_x_id)} ID / "
                    f"{pct1(ak_x_ood)} OOD")
    elif sv["exp_id"] is not None or sv["exp_ood"] is not None:
        parts = []
        if sv["exp_id"] is not None:
            counts = ("" if sv["exp_id_n"] is None
                      else f" ({sv['exp_id_n']}/{sv['exp_id_tot']} grids)")
            parts.append(f"ID {pct1(sv['exp_id'])}{counts}")
        if sv["exp_ood"] is not None:
            counts = ("" if sv["exp_ood_n"] is None
                      else f" ({sv['exp_ood_n']}/{sv['exp_ood_tot']} grids)")
            parts.append(f"OOD {pct1(sv['exp_ood'])}{counts}")
        ak_x_txt = ("export cross-check from the exported prediction "
                    "grids: " + " / ".join(parts))
    else:
        ak_x_txt = "no export cross-check readable ('not run')"
    ak_id128_txt = "not run" if ak_id128 is None else f"{ak_id128:.1f}"
    ak_pub_id128_txt = ak_pub_id128 if ak_pub_id128 else "not stated"

    def rtxt(v):
        return "not run" if v is None else pct1(v)

    if (sv["log_epoch"] is not None and isinstance(ak_run, int)
            and sv["log_epoch"] < ak_run):
        ak_raw_note = (
            f"a run-record value: the surviving training log ends at "
            f"epoch {sv['log_epoch']} (final raw {rtxt(sv['log_raw'])} / "
            f"EMA {rtxt(sv['log_ema'])}), so the epoch-{ak_run} raw value "
            f"cannot be re-derived from preserved logs")
    else:
        ak_raw_note = ("a run-record value (the training log of the "
                       "final epochs is not preserved)")

    def drift_pp(rc, rec):
        a, b = to_pct(rc), to_pct(rec)
        return None if a is None or b is None else abs(a - b)

    prior_parts, drifts = [], []
    for label, rc, rec in (("ID at T=16", sv["pr_id16"], ak_id),
                           ("OOD at T=16", sv["pr_ood16"], ak_ood),
                           ("OOD at T_eval=128", sv["pr_ood128"], ak_ood128)):
        if rc is None:
            continue
        prior_parts.append(f"{label} {pct1(rc)}")
        d = drift_pp(rc, rec)
        if d is not None:
            drifts.append(d)
    drift_txt = (
        "AKOrN unseeded re-evals: the plotted values are the record's "
        "full-stdout re-run official eval of the final EMA checkpoint "
        "(eval_100ep_T16.log / eval_100ep_T128.log in the baselines "
        "directory), re-read here from those logs; re-evaluating that "
        "unseeded checkpoint is nondeterministic -- "
        + ((f"an earlier record revision for the same checkpoint read "
            f"{', '.join(prior_parts)}"
            + (f", up to {max(drifts):.1f} pp of run-to-run drift"
               if drifts else "")
            + ", so single-run differences of that size are re-eval "
            "noise, not seed spread.")
           if prior_parts
           else "the record quotes no superseded revision ('not run')."))

    foot = [
        "Published \u00b1 values are the sd reported by the papers (n not "
        "stated in the cited tables); our numbers are single runs (AKOrN: "
        f"{ak_seed_lbl}; WONN: seed {seed_w}) \u2014 preliminary, no seed "
        "spread, small differences are inconclusive.",
        f"AKOrN ours: final EMA checkpoint, official eval script; raw "
        f"(non-EMA) model ID {pct1(ak_raw_id)} at T=16 is {ak_raw_note}; "
        f"our ID at T_eval=128 is {ak_id128_txt} (paper reference quoted "
        f"in the run record: {ak_pub_id128_txt}); {ak_x_txt}.",
        drift_txt,
        "AKOrN training deviations recorded in the run record: run split "
        "into two phases (epochs 1-40, then a minimal resume wrapper "
        "continued 41-100 restoring model, Adam and EMA state; no epoch "
        "restarted), --checkpoint_every=10, --num_workers=0 (Windows), no "
        "seed fixed (single continuous unseeded run).",
        "WONN ours: exact full-grid match required, so path F1 "
        f"{pct1(wn_f1)} with {pct1(wn_board)} board acc; non-EMA model.pth "
        f"reached path F1 {pct1(wn_f1_raw)}, board still "
        f"{pct1(wn_board_raw)}; params "
        + (f"{params_raw:,} raw" if params_raw is not None
           else "not recorded") + ", effective not measured.",
        "Hardware (our runs): NVIDIA RTX 3070 Ti, 8 GB; our own runs of "
        "the official code (AKOrN commit "
        f"{dig(akorn, 'commit_hash')[:7]}, WONN commit "
        f"{dig(wonn, 'commit_hash')}); the published rows were not run "
        "by us.",
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
        csvrow("akorn_sudoku", "published", "ID board acc @ T_eval=128",
               (ak_pub_id128 or "not stated"), "", ak_pub_run,
               "akorn_tools/out/run_record.json (paper reference quoted "
               "there)"),
        csvrow("akorn_sudoku", "ours", "ID board acc @ T=16", pct1(ak_id),
               "", ak_ours_label_run + ", preliminary",
               "akorn_tools/out/run_record.json"),
        csvrow("akorn_sudoku", "ours", "OOD board acc @ T=16", pct1(ak_ood),
               "", ak_ours_label_run + ", preliminary",
               "akorn_tools/out/run_record.json"),
        csvrow("akorn_sudoku", "ours", "OOD board acc @ T_eval=128",
               ak_t128_val, "",
               ak_ours_label_run + ", preliminary",
               "akorn_tools/out/run_record.json"),
        csvrow("akorn_sudoku", "ours", "ID board acc @ T_eval=128",
               (f"{ak_id128:.1f}" if ak_id128 is not None else "not run"),
               "", ak_ours_label_run + ", preliminary",
               "akorn_tools/out/run_record.json"),
        csvrow("akorn_sudoku", "ours", "raw (non-EMA) ID board acc @ T=16",
               pct1(ak_raw_id), "", ak_ours_label_run,
               "akorn_tools/out/run_record.json (survives only in the run "
               "record; the preserved training log does not reach the "
               "final epoch)"),
        csvrow("akorn_sudoku", "ours",
               "ID board acc @ T=16 (re-run eval log re-read)",
               pct1(sv["id16"]), "", ak_ours_label_run,
               "akorn_tools/out/eval_100ep_T16.log (the full-stdout eval "
               "the run record points to; unseeded re-run)"),
        csvrow("akorn_sudoku", "ours",
               "OOD board acc @ T=16 (re-run eval log re-read)",
               pct1(sv["ood16"]), "", ak_ours_label_run,
               "akorn_tools/out/eval_100ep_T16.log (the full-stdout eval "
               "the run record points to; unseeded re-run)"),
        csvrow("akorn_sudoku", "ours",
               "ID board acc @ T_eval=128 (re-run eval log re-read)",
               pct1(sv["id128"]), "", ak_ours_label_run,
               "akorn_tools/out/eval_100ep_T128.log (the full-stdout eval "
               "the run record points to; unseeded re-run)"),
        csvrow("akorn_sudoku", "ours",
               "OOD board acc @ T_eval=128 (re-run eval log re-read)",
               pct1(sv["ood128"]), "", ak_ours_label_run,
               "akorn_tools/out/eval_100ep_T128.log (the full-stdout eval "
               "the run record points to; unseeded re-run)"),
        csvrow("akorn_sudoku", "ours",
               "ID board acc @ T=16 (earlier record revision)",
               pct1(sv["pr_id16"]), "", ak_ours_label_run,
               "akorn_tools/out/run_record.json (superseded earlier-"
               "revision value quoted in the eval notes)"),
        csvrow("akorn_sudoku", "ours",
               "OOD board acc @ T=16 (earlier record revision)",
               pct1(sv["pr_ood16"]), "", ak_ours_label_run,
               "akorn_tools/out/run_record.json (superseded earlier-"
               "revision value quoted in the eval notes)"),
        csvrow("akorn_sudoku", "ours",
               "OOD board acc @ T_eval=128 (earlier record revision)",
               pct1(sv["pr_ood128"]), "", ak_ours_label_run,
               "akorn_tools/out/run_record.json (superseded earlier-"
               "revision value quoted in the eval notes)"),
        csvrow("akorn_sudoku", "ours",
               "ID board acc @ T_eval=128 (prediction-grid export re-check)",
               pct1(sv["exp_id"]), "", ak_ours_label_run,
               "akorn_tools/out/_export_id.log (board_acc of the exported "
               "prediction grids, final EMA checkpoint)"),
        csvrow("akorn_sudoku", "ours",
               "OOD board acc @ T_eval=128 (prediction-grid export re-check)",
               pct1(sv["exp_ood"]), "", ak_ours_label_run,
               "akorn_tools/out/_export_ood.log (board_acc of the exported "
               "prediction grids, final EMA checkpoint)"),
        csvrow("akorn_sudoku", "ours", "surviving training log reaches epoch",
               (str(sv["log_epoch"]) if sv["log_epoch"] is not None
                else "not recorded"), "", ak_ours_label_run,
               "akorn_tools/out/train_reduced.log"),
        csvrow("akorn_sudoku", "ours",
               "raw (non-EMA) board acc, end of surviving training log",
               pct1(sv["log_raw"]), "", ak_ours_label_run,
               "akorn_tools/out/train_reduced.log"),
        csvrow("akorn_sudoku", "ours",
               "EMA board acc, end of surviving training log",
               pct1(sv["log_ema"]), "", ak_ours_label_run,
               "akorn_tools/out/train_reduced.log"),
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
    # tall enough for both modes: the tracked record-based render and the
    # no-baselines fallback, whose commit hashes read "n/a (...)" and so
    # wrap one footnote line more
    fig_w, fig_h = 13.6, 11.0
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
           "our run (this repo)", size=11, color="#555555",
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

    if y > c.h - 6.0:
        raise AssertionError(
            f"{F10}: content bottom {y:.0f}px exceeds the figure canvas "
            f"height {c.h:.0f}px; enlarge fig_h or shorten the footnotes")
    c.check(F10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    c.fig.savefig(out_png, dpi=DPI)
    print(f"wrote {out_png}")


def records_from_csv(csv_path):
    """Fallback without a baselines directory: rebuild record-like dicts
    from the tracked data.csv so the figure can be regenerated from the
    tracked data alone. Fields that live only in the run records (commit
    hashes, the AKOrN paper-reference quote, export cross-check) come back
    as "not recorded" / "not stated"."""
    if not csv_path.is_file():
        raise SystemExit(
            f"{csv_path} not found and no baselines directory given: pass "
            "--baselines-dir (or set CF_BASELINES_DIR)")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def val_src(block, metric, source):
        for r in rows:
            if (r["block"], r["source"], r["metric"]) == (block, source,
                                                         metric):
                return r["value_percent"]
        return None

    def val(block, metric):
        return val_src(block, metric, "ours")

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

    def frac(s):
        """tracked data.csv percent string -> record-style fraction.

        The run records hold fractions (0.001) while data.csv holds
        percents (0.1); build()'s pct1()/to_pct() auto-detection expects
        fractions, so every percent read back here is divided by 100 --
        otherwise sub-1.5 percents (board acc 0.1) would render x100."""
        v = num(s)
        return None if v is None else v / 100.0

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
    seed_m = re.search(r"(\d+) run \(unseeded\)", budget("akorn_sudoku"))
    akorn = dict(
        gpu=f"NVIDIA GeForce {ak_gpu}",
        epochs_run=ak_epochs, official_epochs=ak_off,
        training_wall_minutes=ak_min,
        commit_hash="n/a (run record not given)",
        # the tracked budget column records the run against the official
        # epoch total, so the full-budget verdict stays reproducible here
        official_command="official protocol (per tracked data.csv)",
        seeds=dict(seeds_run=int(seed_m.group(1)) if seed_m else None),
        eval=dict(
            board_accuracy_T16=dict(
                id_T16=frac(val("akorn_sudoku", "ID board acc @ T=16")),
                ood_T16=frac(val("akorn_sudoku", "OOD board acc @ T=16")),
            ),
            board_accuracy_T_eval_128=dict(
                id_T128=frac(val("akorn_sudoku",
                               "ID board acc @ T_eval=128")),
                ood_T128=frac(val("akorn_sudoku",
                                "OOD board acc @ T_eval=128")),
            ),
            raw_model_id_T16_from_train_log_epoch_100=num(
                val("akorn_sudoku", "raw (non-EMA) ID board acc @ T=16")),
            official_paper_reference=dict(
                id=val_src("akorn_sudoku", "ID board acc @ T_eval=128",
                           "published") or ""),
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
                board_acc=frac(val("wonn_maze_hard",
                                  "exact-board acc @ K=1, T=24")),
                path_f1=frac(val("wonn_maze_hard", "path F1 @ K=1, T=24")),
            ),
            **{"model.pth": dict(
                board_acc=frac(val(
                    "wonn_maze_hard",
                    "exact-board acc @ K=1, T=24 (non-EMA model.pth)")),
                path_f1=frac(val(
                    "wonn_maze_hard",
                    "path F1 @ K=1, T=24 (non-EMA model.pth)")),
            )},
        ),
    )

    params_raw = None
    p = val("wonn_maze_hard", "params raw")
    if p and re.fullmatch(r"[0-9,]+", p):
        params_raw = int(p.replace(",", ""))

    # surviving-log and re-eval numbers fall back to their own data.csv
    # rows (written there on the last record-based run), so the footnotes
    # regenerate; export-grid counts exist only next to the run itself
    sv = dict(
        id16=frac(val("akorn_sudoku",
                     "ID board acc @ T=16 (re-run eval log re-read)")),
        ood16=frac(val("akorn_sudoku",
                      "OOD board acc @ T=16 (re-run eval log re-read)")),
        id128=frac(val("akorn_sudoku",
                      "ID board acc @ T_eval=128 (re-run eval log re-read)")),
        ood128=frac(val(
            "akorn_sudoku",
            "OOD board acc @ T_eval=128 (re-run eval log re-read)")),
        exp_id=frac(val(
            "akorn_sudoku",
            "ID board acc @ T_eval=128 (prediction-grid export re-check)")),
        exp_ood=frac(val(
            "akorn_sudoku",
            "OOD board acc @ T_eval=128 (prediction-grid export re-check)")),
        log_epoch=None, log_raw=None, log_ema=None,
        pr_id16=frac(val("akorn_sudoku",
                        "ID board acc @ T=16 (earlier record revision)")),
        pr_ood16=frac(val("akorn_sudoku",
                         "OOD board acc @ T=16 (earlier record revision)")),
        pr_ood128=frac(
            val("akorn_sudoku",
                "OOD board acc @ T_eval=128 (earlier record revision)")),
    )
    ep = val("akorn_sudoku", "surviving training log reaches epoch")
    if ep is not None:
        try:
            sv["log_epoch"] = int(ep)
        except ValueError:
            pass
    sv["log_raw"] = frac(val(
        "akorn_sudoku", "raw (non-EMA) board acc, end of surviving "
        "training log"))
    sv["log_ema"] = frac(val(
        "akorn_sudoku", "EMA board acc, end of surviving training log"))

    return akorn, wonn, params_raw, sv


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
        surviving = akorn_surviving_logs(baselines_dir)
        surviving.update(akorn_prior_revision(akorn))
    else:
        print("--baselines-dir not given and CF_BASELINES_DIR not set: "
              "regenerating from the tracked "
              "figures/baselines_reference/data.csv only; record-only "
              "fields (commit hashes, paper-reference quote, export "
              "cross-check) read 'not recorded' / 'not stated'.")
        ARGS.wonn_train_log = None
        akorn, wonn, params_from_csv, surviving = records_from_csv(
            out_dir / "data.csv")

    t128 = (akorn_t128_id(akorn), akorn_t128_ood(akorn))
    if t128[1] is not None:
        print(f"AKOrN T_eval=128 from record: ID "
              f"{'' if t128[0] is None else format(t128[0], '.1f')}, "
              f"OOD {t128[1]:.1f}")
    else:
        print("AKOrN T_eval=128 OOD: not present in record -> 'not run'")
    rc = [f"{k}={surviving[k]}" for k in ("id16", "ood16", "id128",
                                          "ood128", "exp_id", "exp_ood",
                                          "pr_id16", "pr_ood16",
                                          "pr_ood128")
          if surviving[k] is not None]
    print("AKOrN surviving re-eval/export/prior-revision logs: "
          + ("; ".join(rc) if rc else "not present -> 'not run'"))

    title, panels, foot, csv_rows = build(akorn, wonn, t128,
                                          params_raw=params_from_csv,
                                          surviving=surviving)

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
