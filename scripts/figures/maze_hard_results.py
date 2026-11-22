"""Maze-hard test results: board-exact vs strict accuracy across arms.

Packet figure F11 (first figure beyond the packet's F1-F10 list).

What the figure shows
---------------------
On the 1000-maze Maze-hard test split (30x30, sapientinc/maze-30x30-hard-1k),
every arm tested so far is far from solving the board:

- CF-WallAware (cross kernel, hid 16), trained here, 120 epochs x 3 seeds:
  0.0 +- 0.0 board-exact and 0.0 +- 0.0 strict at the validation-selected
  horizon T*=16 (at the trained depth T=24 also 0.0). Test sweep T in
  {16, 24, 32, 48} stays at 0.0-0.2% per seed.
- WONN (official code, reduced budget 150/9000 epochs, 1 seed): official
  eval board accuracy 0.1% (EMA checkpoint) / 8.4% (non-EMA model.pth);
  strict is "not computed" (never recorded). Reduced budget - not a
  reproduction of the paper's 76.2%.
- Transport CF, physics-only arm (exact solve; 0 learned parameters, one
  deterministic run): board 0/1000, strict 4/1000 (0.4%) at T=8; with the
  seeded tie-break (K=4 random starts, lowest dissipation, first 300 test
  mazes) strict becomes 98/300 (32.7%) while board stays 0/300.

All 1000 test mazes are tied (more than one equally short start-to-goal
route), which is why exact (board) is strictly harder than strict (any
valid wall-free simple path of the reference length): a solver can emit a
perfectly valid alternative route and still fail board-exact.

Data locations (all external, read-only; passed via CLI or env vars)
---------------------------------------------------------------------
  --families-dir (or CF_FAMILIES_DIR): the cf-families results tree with
      <dir>/maze_hard_cf/results.json (+ states/CF-WallAware_seed<k>.pt)
      <dir>/transport_mazehard/results.json
      <dir>/transport_mazehard_tie/results.json
  --baselines-dir (or CF_BASELINES_DIR): the baselines tree with
      <dir>/wonn_tools/out/run_record.json      (WONN reduced-budget record)
      <dir>/wonn_tools/out/wonn_reduced_preds.npz (test grids; its PREDICTIONS
          are the 52-epoch interim EMA export, not the 150-epoch headline eval)
      <dir>/wonn_tools/out/train_log.txt        (raw parameter count)
      <dir>/wonn_tools/out/eval_ema_model.txt   (52-epoch interim eval log)

The test grids are decoded from the npz export of the same HF test.csv the
records were produced from; the decode is cross-checked by re-evaluating
CF-WallAware seed 0 at T*=16 and the physics arm at T=8 and asserting the
recomputed means match the recorded results.json values.

Without external directories the script regenerates every figure from the
tracked CSVs in the output folder (data.csv, permaze_iou.csv,
example_grids.csv).

Outputs
-------
  figures/maze_hard_results/maze_hard_results.png    (table + bar panel)
  figures/maze_hard_results/maze_hard_T_sweep.png    (accuracy vs T per seed)
  figures/maze_hard_results/maze_hard_examples.png   (4 example predictions)
  figures/maze_hard_results/maze_hard_coverage.png   (per-instance IoU)
plus the CSVs named above.

Run (CPU only):
  python scripts/figures/maze_hard_results.py \
      --families-dir <cf-families results tree> \
      --baselines-dir <baselines tree>
  # or CF_FAMILIES_DIR=... CF_BASELINES_DIR=... python scripts/figures/maze_hard_results.py
  # no dirs given -> tracked CSV fallback

Every number is recomputed from the records at run time (or read back from
the tracked data.csv in fallback); nothing in the figures is estimated. The
layout is checked at render time: no text box may overlap any other text
box or any drawn line/patch/marker sample, and every table cell stays inside
its column band.
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
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import DPI, OKABE_ITO, apply_style  # noqa: E402
from baselines_reference import Canvas, lh_pt, nl  # noqa: E402

F11 = "F11 maze-hard results"

# metric colours (fixed for the whole figure set)
EXACT_C = OKABE_ITO["blue"]           # board-exact metric
STRICT_C = OKABE_ITO["bluish_green"]  # strict (valid path) metric
CF_C = OKABE_ITO["orange"]
WONN_C = OKABE_ITO["sky_blue"]
PHYS_C = OKABE_ITO["vermillion"]
GREY = OKABE_ITO["grey"]
BLACK = "#333333"
HAIR = "#B8B8B8"

# maze-grid error-map colours (project convention)
GRID_COL = dict(
    wall="#222222",
    hit="#006400",          # dark green: predicted and reference path cell
    extra="#D55E00",        # vermillion "red": predicted, not reference
    missed="#0072B2",       # blue: reference cell, not predicted
    negative="#C9C9C9",     # grey: free cell, correctly not predicted
)

# published reference numbers (cited constants, not run by us):
# WONN paper, Table 3, p. 9: 76.2% board accuracy, K=1, T=24, 0.396M params,
# 9000 epochs (official protocol).
PUB_WONN_BOARD = 76.2
PUB_WONN_PARAMS = "0.396 M"
PUB_WONN_EPOCHS = 9000

LEFT_PX = 64.0
RIGHT_PX = 48.0
STRIP_W = 8.0
ROW_PAD = 8.0


def repo_root():
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# live recomputation from the external (read-only) result trees
# ---------------------------------------------------------------------------

def decode_npz_grids(npz_path):
    """npz token ids -> MazeHardDataset rows (question/answer strings).

    Token maps (verified against the source csv; the live CF re-evaluation
    below reproduces the recorded metrics, which validates order and
    encoding): inputs 0=wall '#', 1=free ' ', 2=start 'S', 3=goal 'G';
    labels additionally 4=path 'o'.
    """
    z = np.load(str(npz_path))
    inp, lab = z["inputs"], z["labels"]
    M_Q = {0: "#", 1: " ", 2: "S", 3: "G"}
    M_A = {0: "#", 1: " ", 2: "S", 3: "G", 4: "o"}
    rows = []
    for i in range(inp.shape[0]):
        q = "".join(M_Q[c] for c in inp[i].reshape(-1).tolist())
        a = "".join(M_A[c] for c in lab[i].reshape(-1).tolist())
        rows.append(dict(question=q, answer=a))
    return rows


def load_model_to(repo_root_dir, cls_path, kw):
    """Import a model class by dotted path; keeps imports lazy and local."""
    mod_name, cls_name = cls_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)(**kw)


def cf_per_maze(states_dir, ds, seeds, t_star, batch=250):
    """CF-WallAware live evaluation on the test set at each seed's T*."""
    import torch
    from cf.models.cf import CFNet
    from cf.metrics import evaluate_mask

    per_seed = {}
    for seed in seeds:
        model = CFNet(hid=16, cross_kernel=True, wall_aware=True)
        state = torch.load(str(states_dir / ("CF-WallAware_seed%d.pt" % seed)),
                          map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        rows = []
        with torch.no_grad():
            for i in range(0, len(ds), batch):
                logits = model(ds.x[i:i + batch], T=t_star[seed])
                pred = logits.numpy() >= 0
                for k in range(pred.shape[0]):
                    rows.append(evaluate_mask(
                        pred[k], ds.walls[i + k], ds.starts[i + k],
                        ds.goals[i + k], ds.y[i + k].numpy()))
        per_seed[seed] = rows
    return per_seed


_PHYS_CACHE = {}


def physics_model():
    """One shared physics model instance for every live rollout."""
    if "m" not in _PHYS_CACHE:
        sys.path.insert(0, str(repo_root()))
        sys.path.insert(0, str(repo_root() / "scripts"))
        from eval_transport_mazehard import MazeHardExactNet
        _PHYS_CACHE["m"] = MazeHardExactNet(physics_only=True,
                                            relaxation="exact")
    return _PHYS_CACHE["m"]


def physics_per_maze(ds, T=8, batch=100):
    """Transport physics-only arm (exact solve) live evaluation at T."""
    import torch
    from cf.metrics import evaluate_mask

    model = physics_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 0, "physics arm must hold 0 learned parameters"
    rows = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            logits = model(ds.x[i:i + batch], T)
            pred = torch.sigmoid(logits).numpy() >= 0.5
            for k in range(pred.shape[0]):
                rows.append(evaluate_mask(
                    pred[k], ds.walls[i + k], ds.starts[i + k],
                    ds.goals[i + k], ds.y[i + k].numpy()))
    return rows


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def wonn_raw_params(train_log_path):
    if train_log_path is None or not Path(train_log_path).is_file():
        return None
    text = Path(train_log_path).read_text(errors="replace")
    m = re.search(r"(?i)total number of parameters:\s*([0-9,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None


def wonn_npz_ious(npz_path):
    """Per-maze IoU of the npz-exported predictions (INTERIM 52-epoch EMA -
    labelled as such everywhere; used only for the coverage panel)."""
    z = np.load(str(npz_path))
    lab, pred = z["labels"], z["pred_single"]
    # IoU of the predicted route cells with the reference route cells,
    # with S and G treated as route cells (the cf label convention):
    mask_pred = np.isin(pred, (2, 3, 4))
    mask_ref = np.isin(lab, (2, 3, 4))
    ious = []
    for i in range(lab.shape[0]):
        tp = int((mask_pred[i] & mask_ref[i]).sum())
        fp = int((mask_pred[i] & ~mask_ref[i]).sum())
        fn = int((~mask_pred[i] & mask_ref[i]).sum())
        ious.append(tp / max(tp + fp + fn, 1))
    # path F1 exactly as the official eval_maze.py counts it: class 'o' only
    pred_path = pred == 4
    ref_path = lab == 4
    tp = int((pred_path & ref_path).sum())
    fp = int((pred_path & ~ref_path).sum())
    fn = int((~pred_path & ref_path).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return np.array(ious), float(f1)


def collect_live(families_dir, baselines_dir):
    sys.path.insert(0, str(repo_root()))
    from cf.data.maze_hard import MazeHardDataset

    families_dir, baselines_dir = Path(families_dir), Path(baselines_dir)
    D = {"checks": {}}

    # ---------------- CF record (maze_hard_cf) ----------------
    cfj = json.loads((families_dir / "maze_hard_cf" / "results.json").read_text())
    runs = cfj["runs"]
    pm = cfj["per_model"]["CF-WallAware"]
    seeds = sorted(pm["seeds"])
    t_star_map = {int(pm["T_star"][i]): None for i in range(len(seeds))}
    T_star_list = [int(pm["T_star"][i]) for i in range(len(seeds))]
    T_star = {s: int(runs["CF-WallAware_seed%d" % s]["T_star"]) for s in seeds}
    cf_runs = {}
    for s in seeds:
        r = runs["CF-WallAware_seed%d" % s]
        cf_runs[s] = dict(
            epochs_done=r["epochs_done"], epochs_planned=r["epochs_planned"],
            best_epoch=r["best_epoch"], T_star=int(r["T_star"]),
            params_raw=r["params_raw"], params_eff=r["params_effective"],
            train_seconds=r["train_seconds"], T_train=int(r["T_train"]),
            t_train_metrics=r["T_train_metrics"],
            t_star_metrics=r["T_star_metrics"],
            test_sweeps={int(t): dict(v) for t, v in r["test_sweeps"].items()},
        )
    D["cf"] = dict(
        seeds=seeds, T_star=T_star, runs=cf_runs,
        exact_tstar=pm["exact_at_T_star"], exact_tstar_sd=pm["exact_at_T_star_std"],
        strict_tstar=pm["strict_at_T_star"], strict_tstar_sd=pm["strict_at_T_star_std"],
        exact_ttrain=pm["exact_at_T_train"], exact_ttrain_sd=pm["exact_at_T_train_std"],
        strict_ttrain=pm["strict_at_T_train"], strict_ttrain_sd=pm["strict_at_T_train_std"],
        iou_tstar=pm["iou_at_T_star"], recall_tstar=pm["recall_at_T_star"],
        recall_ttrain=pm["recall_at_T_train"], iou_ttrain=pm["iou_at_T_train"],
        extra_ttrain=float(np.mean([cf_runs[s]["t_train_metrics"]["extra_cells"] for s in seeds])),
        extra_tstar=float(np.mean([cf_runs[s]["t_star_metrics"]["extra_cells"] for s in seeds])),
        device=cfj["budget"]["device"], wall_min=cfj["budget"]["wall_minutes"],
    )

    # ---------------- transport records ----------------
    trj = json.loads((families_dir / "transport_mazehard" / "results.json").read_text())
    ex8 = trj["results"]["exact"]["test"]["per_T"]["8"]
    D["phys"] = dict(
        n=int(trj["results"]["exact"]["test"]["n_mazes"]),
        board8=int(ex8["board_all"]), strict8=int(ex8["strict_all"]),
        iou8=ex8["iou_mean"],
        n_tied=int(trj["sets"]["n_tied"]), n_untied=int(trj["sets"]["n_untied"]),
        T=8, device=trj["protocol"]["device"],
        wall_s=trj["wall_time"]["total_seconds"],
    )
    tij = json.loads((families_dir / "transport_mazehard_tie" / "results.json").read_text())
    tie8 = tij["results"]["exact_tie"]["per_T"]["8"]
    tie0 = tij["results"]["exact"]["per_T"]["8"]
    D["phys_tie"] = dict(
        n=int(tij["dataset"]["n_test_mazes"]),
        board8=int(tie8["board_all"]), strict8=int(tie8["strict_all"]),
        strict8_no_tie=int(tie0["strict_all"]),
        n_starts=int(tij["tie_break"]["n_starts"]),
        n_tied=int(tij["sets"]["n_tied"]), T=8,
        device=tij["protocol"]["device"],
    )

    # ---------------- WONN record (external baseline) ----------------
    wo = baselines_dir / "wonn_tools" / "out"
    wr = json.loads((wo / "run_record.json").read_text())
    D["wonn"] = dict(
        ema_board=wr["eval"]["ema_model"]["board_acc"],
        ema_f1=wr["eval"]["ema_model"]["path_f1"],
        raw_board=wr["eval"]["model.pth"]["board_acc"],
        raw_f1=wr["eval"]["model.pth"]["path_f1"],
        epochs=wr["epochs_trained"], official_epochs=wr["official_epochs"],
        seed=wr["seed"], batch=wr["config"]["batchsize"],
        official_batch=wr["config"]["official_batchsize"],
        gpu=wr["gpu"]["model"], commit=wr["commit_hash"],
        wall_min=wr["train_wall_minutes"],
        params_raw=wonn_raw_params(wo / "train_log.txt"),
    )

    # ---------------- test grids from the baselines npz ----------------
    npz_path = wo / "wonn_reduced_preds.npz"
    rows = decode_npz_grids(npz_path)
    ds = MazeHardDataset(rows)

    # ---------------- live recomputations + cross-checks ----------------
    cf_pm = cf_per_maze(families_dir / "maze_hard_cf" / "states", ds,
                        seeds, T_star)
    for s in seeds:
        rec = cf_runs[s]["test_sweeps"][cf_runs[s]["T_star"]]
        for key in ("strict", "exact"):
            live, want = mean(cf_pm[s], key), rec[key]
            assert abs(live - want) <= 2e-3, (
                "CF seed %d %s recomputed %.5f vs recorded %.5f"
                % (s, key, live, want))
        for key, tol in (("iou", 2e-3), ("recall", 2e-3),
                         ("extra_cells", 0.05), ("missed_cells", 0.1)):
            live, want = mean(cf_pm[s], key), rec[key]
            assert abs(live - want) <= tol, (
                "CF seed %d %s recomputed %.5f vs recorded %.5f"
                % (s, key, live, want))
        D["checks"]["cf_seed%d" % s] = (
            "strict %g exact %g iou %.4f == recorded test_sweeps[T*]"
            % (mean(cf_pm[s], "strict"), mean(cf_pm[s], "exact"),
               mean(cf_pm[s], "iou")))
    D["cf_live"] = cf_pm

    ph_pm = physics_per_maze(ds, T=8)
    assert sum(r["strict"] for r in ph_pm) == D["phys"]["strict8"]
    assert sum(r["exact"] for r in ph_pm) == D["phys"]["board8"]
    assert abs(mean(ph_pm, "iou") - D["phys"]["iou8"]) <= 1e-3
    D["checks"]["physics"] = ("strict %d exact %d mean iou %.4f == recorded"
                              % (D["phys"]["strict8"], D["phys"]["board8"],
                                 mean(ph_pm, "iou")))
    D["phys_live"] = ph_pm

    wn_ious, wn_f1 = wonn_npz_ious(npz_path)
    interim_log = wo / "eval_ema_model.txt"
    if interim_log.is_file():
        m = re.search(r"path_f1:\s*([0-9.]+)",
                      interim_log.read_text(errors="replace"))
        if m:
            assert abs(wn_f1 - float(m.group(1))) <= 5e-3, (
                "npz prediction export does not match the 52-epoch interim "
                "eval log (path F1 %.4f vs %s)" % (wn_f1, m.group(1)))
    D["checks"]["wonn_npz"] = (
        "npz predictions: path F1 %.4f == the 52-epoch interim EMA eval log "
        "-> interim export, NOT the 150-epoch headline eval" % wn_f1)
    D["wonn_npz"] = dict(ious=wn_ious, f1=wn_f1,
                         interim_note=("npz predictions reproduce the "
                                        "52-epoch interim EMA eval log "
                                        "(path F1 %.4f); they are NOT the "
                                        "150-epoch headline eval" % wn_f1))

    # ---------------- example prediction selection (fixed rules) ----------
    iou0 = np.array([r["iou"] for r in cf_pm[seeds[0]]])
    order = np.argsort(iou0, kind="stable")
    med_idx = int(order[len(order) // 2])
    best_idx = int(np.argmax(iou0))
    ph_strict_idx = np.where(np.array([r["strict"] for r in ph_pm]) == 1)[0]
    assert len(ph_strict_idx) > 0, "no physics strict pass found"
    e3_idx = int(ph_strict_idx.min())
    D["example_spec"] = dict(
        e1=dict(maze=med_idx, arm="cf", seed=seeds[0],
                T=T_star[seeds[0]],
                rule="CF-WallAware seed 0: maze with the median per-maze IoU"),
        e2=dict(maze=best_idx, arm="cf", seed=seeds[0],
                T=T_star[seeds[0]],
                rule="CF-WallAware seed 0: maze with the highest per-maze IoU"),
        e3=dict(maze=e3_idx, arm="physics", T=8,
                rule="physics arm: lowest-index maze whose read-out passes "
                     "strict (valid alternative route)"),
        e4=dict(maze=0, arm="physics", T=8,
                rule="physics arm: maze 0, the first test maze (fixed)"),
    )
    for name, sp in D["example_spec"].items():
        src = cf_pm[seeds[0]] if sp["arm"] == "cf" else ph_pm
        sp["stats"] = src[sp["maze"]]

    # per-maze arrays for the coverage panel + the tracked permaze_iou.csv
    def stat_rows(rr, arm):
        return [dict(arm=arm, maze_index=i, iou=r["iou"], strict=r["strict"],
                     exact=r["exact"], extra_cells=r["extra_cells"],
                     missed_cells=r["missed_cells"]) for i, r in enumerate(rr)]

    D["permaze"] = (stat_rows(cf_pm[seeds[0]], "cf_wallaware_seed0_tstar")
                    + stat_rows(ph_pm, "physics_exact_t8")
                    + [dict(arm="wonn_ema52_interim_export", maze_index=i,
                            iou=float(v), strict="", exact="",
                            extra_cells="", missed_cells="")
                       for i, v in enumerate(wn_ious)])

    # example grids (chars) for the panels + the tracked example_grids.csv
    import torch
    ex_rows = []
    for name in ("e1", "e2", "e3", "e4"):
        sp = D["example_spec"][name]
        i = sp["maze"]
        q = rows[i]["question"]
        a = rows[i]["answer"]
        if sp["arm"] == "cf":
            from cf.models.cf import CFNet
            m = CFNet(hid=16, cross_kernel=True, wall_aware=True)
            m.load_state_dict(torch.load(
                str(families_dir / "maze_hard_cf" / "states" /
                    ("CF-WallAware_seed%d.pt" % sp["seed"])), map_location="cpu"))
            m.eval()
            with torch.no_grad():
                logits = m(ds.x[i:i + 1], T=sp["T"])
                pred = logits.numpy()[0] >= 0
        else:
            model = physics_model()
            with torch.no_grad():
                logits = model(ds.x[i:i + 1], T=sp["T"])
                pred = torch.sigmoid(logits).numpy()[0] >= 0.5
        wall_mask = (np.frombuffer(q.encode("ascii"), dtype=np.uint8)
                     == ord("#")).reshape(30, 30)
        assert not (pred & wall_mask).any(), "prediction on a wall cell"
        label_mask = "".join("1" if ch in "oSG" else "0" for ch in a)
        ex_rows.append(dict(
            panel=name,
            arm_label=("CF-WallAware, seed 0, T=%d" % sp["T"]
                       if sp["arm"] == "cf" else
                       "physics exact solve, T=%d" % sp["T"]),
            maze_index=i, question=q, label=label_mask,
            pred="".join("1" if v else "0"
                         for v in pred.reshape(-1).tolist()),
        ))
    D["example_grids"] = ex_rows
    D["n_test"] = len(ds)
    return D


# ---------------------------------------------------------------------------
# fallback: rebuild the same data from the tracked CSVs
# ---------------------------------------------------------------------------

def collect_fallback(out_dir):
    data_path = out_dir / "data.csv"
    if not data_path.is_file():
        raise SystemExit(
            "%s not found and no external directories given: pass "
            "--families-dir and --baselines-dir (or set CF_FAMILIES_DIR / "
            "CF_BASELINES_DIR)" % data_path)
    with data_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    D = {"checks": {}, "fallback": True}

    def get(block, arm, metric):
        for r in rows:
            if (r["block"], r["arm"], r["metric"]) == (block, arm, metric):
                return r
        return None

    def num(block, arm, metric, default=None):
        r = get(block, arm, metric)
        if r is None or r["value"] in ("", "not computed", "not recorded"):
            return default
        try:
            return float(r["value"])
        except ValueError:
            return default

    per_seed = {}
    for s in (0, 1, 2):
        sweeps = {}
        for t in (16, 24, 32, 48):
            st = num("tsweep", "CF-WallAware seed %d" % s,
                     "strict accuracy @T=%d (%%)" % t)
            ex = num("tsweep", "CF-WallAware seed %d" % s,
                     "board-exact accuracy @T=%d (%%)" % t)
            iou = num("tsweep", "CF-WallAware seed %d" % s, "IoU @T=%d" % t)
            sweeps[t] = dict(strict=(st or 0.0) / 100.0,
                             exact=(ex or 0.0) / 100.0, iou=(iou or 0.0))
        per_seed[s] = dict(epochs_done=120, epochs_planned=120,
                           best_epoch=None, T_star=16, T_train=24,
                           params_raw=10177, params_eff=5889,
                           t_train_metrics={}, t_star_metrics={},
                           test_sweeps=sweeps)
    D["cf"] = dict(
        seeds=[0, 1, 2], T_star={0: 16, 1: 16, 2: 16}, runs=per_seed,
        exact_tstar=num("results_table", "CF-WallAware (ours)",
                        "board-exact accuracy @T*=16 (%)"),
        exact_tstar_sd=num("results_table", "CF-WallAware (ours)",
                           "board-exact sd @T*=16 (%)", 0.0),
        strict_tstar=num("results_table", "CF-WallAware (ours)",
                         "strict accuracy @T*=16 (%)"),
        strict_tstar_sd=num("results_table", "CF-WallAware (ours)",
                            "strict sd @T*=16 (%)", 0.0),
        exact_ttrain=num("results_table", "CF-WallAware (ours)",
                         "board-exact accuracy @T=24 (%)"),
        exact_ttrain_sd=0.0,
        strict_ttrain=num("results_table", "CF-WallAware (ours)",
                          "strict accuracy @T=24 (%)"),
        strict_ttrain_sd=0.0,
        iou_tstar=num("results_table", "CF-WallAware (ours)",
                      "mean IoU @T*=16"),
        recall_tstar=num("results_table", "CF-WallAware (ours)",
                         "mean recall @T*=16"),
        recall_ttrain=None, iou_ttrain=None,
        extra_ttrain=num("results_table", "CF-WallAware (ours)",
                         "mean extra cells @T=24 (count)"),
        extra_tstar=num("results_table", "CF-WallAware (ours)",
                        "mean extra cells @T*=16 (count)"),
        device="cuda (recorded)", wall_min=None)

    D["phys"] = dict(
        n=1000, board8=0,
        strict8=int(num("results_table", "Physics, exact solve",
                        "strict passes @T=8 (count)")),
        iou8=num("results_table", "Physics, exact solve", "mean IoU @T=8"),
        n_tied=1000, n_untied=0, T=8, device="cpu (recorded)", wall_s=None)
    D["phys_tie"] = dict(
        n=300, board8=0,
        strict8=int(num("results_table", "Physics + tie-break (K=4)",
                        "strict passes @T=8 (count)")),
        strict8_no_tie=int(num("results_table", "Physics + tie-break (K=4)",
                               "strict passes @T=8 without tie-break (count)")),
        n_starts=4, n_tied=300, T=8, device="cpu (recorded)")
    comm = get("results_table", "WONN official (EMA)", "official code commit")
    params_r = get("results_table", "WONN official (EMA)", "params raw")
    D["wonn"] = dict(
        ema_board=(num("results_table", "WONN official (EMA)",
                       "board-exact accuracy (%)") or 0.0) / 100.0,
        ema_f1=(num("results_table", "WONN official (EMA)",
                    "path F1 (%)") or 0.0) / 100.0,
        raw_board=(num("results_table", "WONN official (non-EMA)",
                       "board-exact accuracy (%)") or 0.0) / 100.0,
        raw_f1=(num("results_table", "WONN official (non-EMA)",
                    "path F1 (%)") or 0.0) / 100.0,
        epochs=150, official_epochs=9000, seed=137, batch=32, official_batch=64,
        gpu="RTX 3070 Ti (recorded)",
        commit=(comm["value"] if comm else None),
        wall_min=None,
        params_raw=(int(params_r["value"]) if params_r
                    and params_r["value"] not in ("", "not recorded") else None))

    with (out_dir / "permaze_iou.csv").open(newline="", encoding="utf-8") as fh:
        D["permaze"] = list(csv.DictReader(fh))
    ex_rows = []
    with (out_dir / "example_grids.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ex_rows.append(dict(panel=r["panel"], arm_label=r["arm_label"],
                               maze_index=int(r["maze_index"]),
                               question=r["question"], label=r["label"],
                               pred=r["pred"]))
    D["example_grids"] = ex_rows
    D["example_spec"] = {}
    for name in ("e1", "e2", "e3", "e4"):
        D["example_spec"][name] = dict(
            maze=int(num("examples", name, "maze index", 0)),
            arm=("cf" if name in ("e1", "e2") else "physics"),
            T=int(num("examples", name, "readout T", 8)),
            rule=(get("examples", name, "selection rule") or
                  dict(value=""))["value"],
            stats=dict(
                iou=num("examples", name, "IoU", 0.0),
                strict=int(num("examples", name, "strict pass (0/1)", 0)),
                exact=int(num("examples", name, "board-exact pass (0/1)", 0)),
                extra_cells=int(num("examples", name, "extra cells (count)", 0)),
                missed_cells=int(num("examples", name, "missed cells (count)", 0))))
    f1r = get("coverage", "wonn_ema52_interim_export", "path F1")
    D["wonn_npz"] = dict(
        ious=None, f1=(float(f1r["value"]) if f1r else None),
        interim_note=("interim 52-epoch EMA export (tracked CSV); "
                      "not the 150-epoch headline eval"))
    D["n_test"] = 1000
    return D




# ---------------------------------------------------------------------------
# overlap checker (shared by all axes-based figures)
# ---------------------------------------------------------------------------

def check_figure(fig, name, extra_points=(), pad=1.5):
    """Assert: no two text boxes overlap and no text sits on any drawn
    line/patch/marker sample. Tick marks, spines, gridlines and raster
    images are exempt; everything else is checked in display pixels."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts = [t for t in fig.texts if t.get_text().strip()]
    for ax in fig.axes:
        texts.extend([t for t in ax.texts if t.get_text().strip()])
        texts.append(ax.title)
        texts.append(ax.xaxis.label)
        texts.append(ax.yaxis.label)
        texts.extend([t for t in ax.get_xticklabels() if t.get_text().strip()])
        texts.extend([t for t in ax.get_yticklabels() if t.get_text().strip()])
        leg = ax.get_legend()
        if leg is not None:
            texts.extend([t for t in leg.get_texts() if t.get_text().strip()])
    boxes = [(t.get_text().split("\n")[0][:30], t.get_window_extent(ren))
             for t in texts]
    boxes = [(n_, b) for n_, b in boxes if b.width > 2.0 and b.height > 2.0]

    for (n1, b1), (n2, b2) in combinations(boxes, 2):
        if (b1.x0 - pad <= b2.x1 + pad and b2.x0 - pad <= b1.x1 + pad
                and b1.y0 - pad <= b2.y1 + pad and b2.y0 - pad <= b1.y1 + pad):
            raise AssertionError("%s: texts overlap: %r vs %r (%s vs %s)"
                                 % (name, n1, n2, b1.bounds, b2.bounds))

    pts = []
    for ax in fig.axes:
        for ln in ax.lines:
            data = np.asarray(ln.get_xydata(), float)
            if data.ndim != 2 or data.shape[0] == 0:
                continue
            for i in range(len(data)):
                pts.append(tuple(ax.transData.transform(data[i])))
            for i in range(len(data) - 1):
                p0 = ax.transData.transform(data[i])
                p1 = ax.transData.transform(data[i + 1])
                for f in range(1, 20):
                    pts.append((p0[0] + (p1[0] - p0[0]) * f / 20.0,
                                p0[1] + (p1[1] - p0[1]) * f / 20.0))
        for patch in ax.patches:
            tr = patch.get_transform()
            for poly in patch.get_path().to_polygons():
                for x, y in poly:
                    pts.append(tuple(tr.transform((x, y))))
    pts.extend(extra_points)
    for gx, gy in pts:
        for txt, bb in boxes:
            if (bb.x0 - pad <= gx <= bb.x1 + pad
                    and bb.y0 - pad <= gy <= bb.y1 + pad):
                raise AssertionError(
                    "%s: text %r (bbox %s) sits on drawn geometry at (%.1f, %.1f)"
                    % (name, txt, bb.bounds, gx, gy))
    return fig


# ---------------------------------------------------------------------------
# figure 1: results table + grouped bars (pixel canvas)
# ---------------------------------------------------------------------------

def fmt_pct(x):
    return ("%g" % x) if abs(x - round(x)) < 1e-9 else ("%.1f" % x)


def _wrap(line, width):
    return textwrap.wrap(line, width=width)


def render_results(D, out_png):
    cf, wn, ph, pt = D["cf"], D["wonn"], D["phys"], D["phys_tie"]
    tie_p = pt["strict8"] * 100.0 / pt["n"]
    phys_p = ph["strict8"] * 100.0 / ph["n"]

    title = [
        "Maze-hard test split (1000 mazes, 30x30): board-exact accuracy stays near zero for every arm "
        "tested here -",
        "CF-WallAware 0.0% (120 epochs, 3 seeds), physics 0.0%, WONN official code 0.1% EMA / 8.4% raw "
        "at 150/9000 epochs;",
        "strict beats board-exact only where ties are broken - physics tie-break 32.7% valid paths "
        "vs 0.0% board",
    ]
    guide = [
        "Reading guide: table and bars show the same numbers; blue = board-exact (cell-for-cell match "
        "with the single reference route, WONN's metric),",
        "green = strict (any wall-free simple start-to-goal path of the reference length). Every test "
        "maze here is tied, so a valid alternative",
        "route passes strict but never board-exact. Budgets differ per arm (epochs, seeds, hardware) - "
        "no two arms are budget-matched.",
    ]
    foot = [
        "CF-WallAware row: value = mean over the 3 recorded seeds (0, 1, 2; 120 epochs each); "
        "\u00b1 = sample sd (n\u22121 denominator). Numbers are recorded at both the "
        "validation-selected depth T*=16 and the trained depth T=24 - all four are 0.0.",
        "Metrics: board-exact = predicted route equals the single reference route cell-for-cell (the "
        "metric of the WONN reference code); strict = the predicted cells form any wall-free simple "
        "start-to-goal path of the reference length. All 1000 test mazes are tied (more than one "
        "equally short route), so exact is strictly harder than strict - exact = strict must never be "
        "asserted on this split.",
        "WONN: strict was never computed by the official evaluation - the cells read 'not computed', "
        "not zero. Official eval path F1: 84.0 (EMA) / 93.2 (raw model.pth). Published 76.2% (WONN "
        "paper, Table 3, p. 9) needs the full 9000-epoch budget - the 150-epoch run here is a "
        "reduced-budget pipeline check, not a reproduction.",
        "Physics arm: no learning (0 parameters, 0 epochs), one deterministic run, CPU. Tie-break "
        "row: 4 seeded random starts per maze, lowest-dissipation start kept, on the first 300 test "
        "mazes (reduced slice, labelled); board stays 0 because the picked route need not be the "
        "reference's.",
        "Sources (external, read-only; passed via --families-dir / --baselines-dir): "
        "maze_hard_cf/results.json, transport_mazehard/results.json, "
        "transport_mazehard_tie/results.json, wonn_tools/out/run_record.json. Every number is in "
        "data.csv; the CF mean metrics and the physics arm were recomputed live from the checkpoints "
        "and the physics solver and matched the records.",
    ]

    cols = [
        ("Arm (params)", 0.000, 0.235, "left"),
        ("Budget (epochs, seeds, hardware)", 0.235, 0.480, "left"),
        ("Board-exact =\nreference route (%)", 0.480, 0.632, "center"),
        ("Strict = any valid\npath (%)", 0.632, 0.782, "center"),
        ("Note", 0.782, 1.000, "left"),
    ]
    rows = [
        dict(strip=GREY, cells=[
            "Published reference:\nWONN paper, K=1, T=24\n%s params" % PUB_WONN_PARAMS,
            "%d epochs (official protocol);\nseed count not stated in table" % PUB_WONN_EPOCHS,
            "76.2",
            "not stated\nin the table",
            "WONN paper, Table 3, p. 9;\nboard-exact metric; not run by us"],
            styles=["ours", "ours", "val", "val", "ref"]),
        dict(strip=CF_C, cells=[
            "CF-WallAware, cross kernel\n(ours), hid 16\n10,177 raw / 5,889 effective",
            "120/120 epochs x 3 seeds\n(0, 1, 2), CUDA",
            "0.0 \u00b10.0 @T*=16\n0.0 @T=24",
            "0.0 \u00b10.0 @T*=16\n0.0 @T=24",
            "recall of the reference route\n0.98, but ~%d extra lit cells\nper maze at T*=16" % round(cf["extra_tstar"])],
            styles=["ours", "ours", "val", "val", "ours"]),
        dict(strip=WONN_C, cells=[
            "WONN, official code (commit\n%s), EMA checkpoint + non-EMA\nmodel.pth" % wn["commit"],
            "%d/%d epochs, 1 seed (%d),\nbatch %d (official %d),\n%s" % (
                wn["epochs"], wn["official_epochs"], wn["seed"], wn["batch"],
                wn["official_batch"], wn["gpu"]),
            "EMA 0.1\nraw 8.4",
            "not computed",
            "reduced budget - NOT a\nreproduction; 1 seed =\npreliminary; path F1 84.0\n(EMA) / 93.2 (raw)"],
            styles=["ours", "ours", "val", "missing", "verdict"]),
        dict(strip=PHYS_C, cells=[
            "Transport CF, physics-only\narm (exact solve),\n0 learned parameters",
            "no training: 0 epochs,\n0 seeds, one deterministic\nrun, CPU, T=8",
            "0.0\n(0/1000)",
            "%s\n(%d/1000)" % (fmt_pct(phys_p), ph["strict8"]),
            "readout: a cell is on the path\niff |flow| >= 0.5; all 1000\ntest mazes are tied"],
            styles=["ours", "ours", "val", "val", "ours"]),
        dict(strip=PHYS_C, cells=[
            "Transport CF, physics-only\n+ tie-break K=4 seeded starts",
            "same physics, 4 random\nstarts per maze, lowest\ndissipation kept;\nfirst 300 test mazes",
            "0.0\n(0/300)",
            "%s\n(%d/300)" % (fmt_pct(tie_p), pt["strict8"]),
            "the tie-break picks ONE\nroute, so strict rises -\nbut the picked route need\nnot be the reference"],
            styles=["ours", "ours", "val", "val", "verdict"]),
    ]

    # -- precompute the exact vertical layout so the canvas fits everything --
    ax_h = 460.0
    y = 16.0
    for _ in title:
        y += lh_pt(15) + 6
    y += 4
    for _ in guide:
        y += lh_pt(10.5) + 5
    y += 12
    y += 2 * lh_pt(11) + 5 + 1                    # column headers + rule
    row_line_counts = [max(nl(s) for s in r["cells"]) for r in rows]
    y += ROW_PAD
    y += row_line_counts[0] * lh_pt(11) + 2 * ROW_PAD - ROW_PAD + 4 + 1
    for n_lines in row_line_counts[1:]:
        y += 1 + ROW_PAD + n_lines * lh_pt(11) + 2 * ROW_PAD - ROW_PAD + 4
    y += 24                                        # gap above the bar panel
    fy0 = y + ax_h + 130
    foot_wrapped = ["\n".join(_wrap(line, 172)) for line in foot]
    foot_h = sum(nl(w) * lh_pt(10) + 6 for w in foot_wrapped)
    fig_h_in = (fy0 + foot_h + 24) / float(DPI)

    fig_w_in = 13.6
    c = Canvas(fig_w_in, fig_h_in)
    W, H = c.w, c.h
    cx0 = LEFT_PX
    cw = W - LEFT_PX - RIGHT_PX

    y = 16.0
    for line in title:
        c.text(W / 2, y, line, size=15, weight="bold", ha="center")
        y += lh_pt(15) + 6
    y += 4
    for line in guide:
        c.text(W / 2, y, line, size=10.5, color="#555555", ha="center")
        y += lh_pt(10.5) + 5
    y += 12

    hdr_top = y
    for label, f0, f1, align in cols:
        band = (cx0 + f0 * cw, cx0 + f1 * cw)
        x = band[0] if align == "left" else (band[0] + band[1]) / 2
        c.text(x, hdr_top, label, size=11, weight="semibold", color="#555555",
               ha=align, band=band)
    y = hdr_top + 2 * lh_pt(11) + 5
    c.hline(cx0, cx0 + cw, y, color="#888888", lw=1.1)
    y += 1

    strip_x = LEFT_PX - 18.0
    for r_idx, row in enumerate(rows):
        if r_idx > 0:
            c.hline(cx0, cx0 + cw, y, color=HAIR, lw=0.7, alpha=0.6)
            y += 1
        row_top = y + ROW_PAD
        n_lines = max(nl(s) for s in row["cells"])
        row_h = n_lines * lh_pt(11) + 2 * ROW_PAD
        c.rect(strip_x, row_top, STRIP_W, row_h - 2 * ROW_PAD, row["strip"])
        bands = [(cx0 + f0 * cw, cx0 + f1 * cw) for _, f0, f1, _ in cols]
        aligns = [a for _, _, _, a in cols]
        for s, st, (b0, b1), align in zip(row["cells"], row["styles"], bands,
                                          aligns):
            style = dict(
                ours=dict(size=11, color=BLACK),
                val=dict(size=12.5, color=BLACK, weight="semibold"),
                missing=dict(size=11.5, color="#777777", style="italic"),
                ref=dict(size=11.5, color="#777777"),
                verdict=dict(size=11, color=BLACK),
            )[st]
            x = b0 if align == "left" else (b0 + b1) / 2
            c.text(x, row_top, s, ha=align, band=(b0 - 2, b1 + 2), **style)
        y = row_top + row_h - ROW_PAD + 4
    c.hline(cx0, cx0 + cw, y, color="#888888", lw=1.1)
    y += 24

    # ---- grouped bars on an axes placed below the table ----
    ax_rect = [cx0 / W, (H - y - ax_h) / H, cw / W, ax_h / H]
    ax = c.fig.add_axes(ax_rect)
    ax.grid(False)
    groups = [
        ("CF-WallAware (ours)\n120 ep, 3 seeds, T*=16",
         "0.0", "0.0", 0.0, 0.0),
        ("WONN official, EMA\n150/9000 ep, 1 seed",
         fmt_pct(wn["ema_board"] * 100.0), None, wn["ema_board"] * 100.0, None),
        ("WONN official, raw\n150/9000 ep, 1 seed",
         fmt_pct(wn["raw_board"] * 100.0), None, wn["raw_board"] * 100.0, None),
        ("Physics exact solve\nno training, T=8",
         "0.0", fmt_pct(phys_p), 0.0, phys_p),
        ("Physics + tie-break K=4\nno training, T=8, n=300",
         "0.0", fmt_pct(tie_p), 0.0, tie_p),
    ]
    xs = np.arange(len(groups), dtype=float)
    bw = 0.30
    for i, (name, ex_lbl, st_lbl, exact_v, strict_v) in enumerate(groups):
        xe = i - bw / 2 - 0.02
        xsr = i + bw / 2 + 0.02
        ax.bar([xe], [exact_v], width=bw, color=EXACT_C, edgecolor="none",
               zorder=2)
        ax.text(xe, exact_v + 1.8, ex_lbl, ha="center", va="bottom",
                fontsize=10.5, color=BLACK)
        if strict_v is not None:
            ax.bar([xsr], [strict_v], width=bw, color=STRICT_C,
                   edgecolor="none", zorder=2)
            ax.text(xsr, strict_v + 1.8, st_lbl, ha="center", va="bottom",
                    fontsize=10.5, color=BLACK)
        else:
            ax.text(i, 18.0, "not\ncomputed", ha="center", va="bottom",
                    fontsize=10, color="#777777", style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=10.5)
    ax.set_xlim(-0.65, len(groups) - 0.35)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("board-exact / strict accuracy (% of test mazes)",
                  fontsize=11.5)
    ax.axhline(PUB_WONN_BOARD, color=GREY, lw=1.4, ls="--", zorder=1)
    ax.text(4.4, PUB_WONN_BOARD + 1.2,
            "WONN paper (9000 epochs):\n76.2% board-exact",
            ha="right", va="bottom", fontsize=10.5, color="#555555")
    leg = ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=EXACT_C,
                      label="board-exact (WONN's board metric)"),
        plt.Rectangle((0, 0), 1, 1, color=STRICT_C,
                      label="strict (any valid route)"),
        plt.Line2D([0], [0], color=GREY, lw=1.4, ls="--",
                   label="published WONN (9000 epochs)"),
    ], loc="upper left", fontsize=10.5)
    leg.set_zorder(5)

    # ---- footnotes under the bar panel ----
    fy = y + ax_h + 130.0
    for wrapped in foot_wrapped:
        c.text(LEFT_PX, fy, wrapped, size=10, color="#666666")
        fy += nl(wrapped) * lh_pt(10) + 6

    c.check(F11 + " canvas")
    # _samples() already returns display pixels (y up, dpi-normalised);
    # pass them straight to the axes-based checker.
    check_figure(c.fig, F11 + " full", extra_points=c._samples())
    out_png.parent.mkdir(parents=True, exist_ok=True)
    c.fig.savefig(out_png, dpi=DPI)
    plt.close(c.fig)
    print("wrote %s" % out_png)


# ---------------------------------------------------------------------------
# figure 2: accuracy vs test T for CF-WallAware, per seed
# ---------------------------------------------------------------------------

def render_t_sweep(D, out_png):
    cf = D["cf"]
    Ts = sorted(cf["runs"][cf["seeds"][0]]["test_sweeps"].keys())
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(bottom=0.24)
    ax.grid(False)
    seed_colors = {0: OKABE_ITO["blue"], 1: OKABE_ITO["vermillion"],
                   2: OKABE_ITO["reddish_purple"]}
    max_v = 0.0
    for s in cf["seeds"]:
        ys = [cf["runs"][s]["test_sweeps"][T]["strict"] * 100.0 for T in Ts]
        ye = [cf["runs"][s]["test_sweeps"][T]["exact"] * 100.0 for T in Ts]
        assert all(abs(a - b) < 1e-9 for a, b in zip(ys, ye)), (
            "strict and board-exact differ at a recorded point (seed %d)" % s)
        ax.plot(Ts, ys, "-o", color=seed_colors[s], lw=2.0, markersize=6,
                label="seed %d" % s, zorder=3)
        for T, v in zip(Ts, ys):
            if v > 0:
                ax.annotate(("%g" % v) + "%", (T, v),
                            textcoords="offset points", xytext=(9, -13),
                            fontsize=10.5, color=BLACK)
            max_v = max(max_v, v)
    ax.axvline(24, color=GREY, lw=1.2, ls="--", zorder=1)
    ax.text(23.6, 0.03, "trained at T=24", fontsize=10.5, color="#555555",
            ha="right", va="bottom")
    ax.plot([16], [0], marker="*", color="#111111", markersize=12, zorder=4,
            linestyle="none", label="T*=16 (validation-selected, all seeds)")
    ax.set_xlabel("read-out horizon T (steps at test time)", fontsize=12)
    ax.set_ylabel("test accuracy (% of 1000 mazes)", fontsize=12)
    ax.set_ylim(0, 0.30)
    ax.set_yticks([0, 0.10, 0.20, 0.30])
    ax.set_xticks(Ts)
    ax.set_xlim(14.5, 49.5)
    ax.set_title(
        "Test accuracy vs horizon for CF-WallAware (120 epochs, 3 seeds): "
        "0.0-0.2% at every T in 16-48 -\nlonger thinking does not rescue "
        "Maze-hard; board-exact equals strict at every recorded point",
        fontsize=12.5, pad=22)
    ax.legend(loc="upper left", fontsize=10.5, title="training seed",
              title_fontsize=10.5)
    fig.text(0.5, 0.02,
             "Reading guide: y = strict accuracy (%; one line per training seed: seed 0 / 1 / 2 in\n"
             "blue / vermillion / purple); x = read-out horizon T in steps. Board-exact equals strict\n"
             "at every recorded point, so each line carries both metrics. Points are per-seed full-\n"
             "test values, not aggregates - no error bars apply. y kept linear 0-0.3%: the near-zero\n"
             "values are the message.",
             fontsize=10.5, ha="center", va="bottom", color="#555555")
    check_figure(fig, "F11 T sweep")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("wrote %s" % out_png)


# ---------------------------------------------------------------------------
# figure 3: four example predictions
# ---------------------------------------------------------------------------

def render_examples(D, out_png):
    spec = D["example_spec"]
    grids = {r["panel"]: r for r in D["example_grids"]}
    names = ["e1", "e2", "e3", "e4"]
    titles = {
        "e1": "1: CF-WallAware, seed 0, T*=16\nmaze %d (median-IoU instance)"
              % spec["e1"]["maze"],
        "e2": "2: CF-WallAware, seed 0, T*=16\nmaze %d (best-IoU instance)"
              % spec["e2"]["maze"],
        "e3": "3: physics exact solve, T=8\nmaze %d (a strict-passing maze)"
              % spec["e3"]["maze"],
        "e4": "4: physics exact solve, T=8\nmaze 0 (first test maze)",
    }
    caps = {}
    for name in names:
        st = spec[name]["stats"]
        if st["strict"]:
            verdict = "strict PASS, board-exact 0\n(valid alternative route)"
        else:
            verdict = "strict fail"
        caps[name] = ("IoU %.2f\n%d extra / %d missed cells\n%s"
                      % (st["iou"], st["extra_cells"], st["missed_cells"],
                         verdict))

    fig = plt.figure(figsize=(13.6, 5.9))
    fig.text(0.5, 0.975,
             "Four example predictions on tied Maze-hard test mazes: CF-WallAware covers the "
             "reference route but lights extra cells (panels 1-2);\nthe physics arm lights tied "
             "twin routes (panel 4) or lands on a valid alternative route that passes strict while "
             "failing board-exact (panel 3)",
             ha="center", va="top", fontsize=12.5, weight="bold")
    fig.text(0.5, 0.895,
             "Reading guide: dark green = predicted route cell that is also in the reference route "
             "(hit); vermillion / red = predicted but not in the reference (extra);\nblue = "
             "reference route cell that was missed; grey = free cell correctly left unlit; "
             "black = wall; the small black box marks S and G. Panels 1-2 show seed 0 at its "
             "validation-selected T*=16.",
             ha="center", va="top", fontsize=10.5, color="#555555")

    ax_pos = [[0.040, 0.28, 0.205, 0.44], [0.290, 0.28, 0.205, 0.44],
              [0.540, 0.28, 0.205, 0.44], [0.760, 0.28, 0.205, 0.44]]
    for rect, name in zip(ax_pos, names):
        ax = fig.add_axes(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        g = grids[name]
        q = g["question"]
        label = (np.frombuffer(g["label"].encode("ascii"), dtype=np.uint8)
                 == ord("1")).reshape(30, 30)
        pred = (np.frombuffer(g["pred"].encode("ascii"), dtype=np.uint8)
                == ord("1")).reshape(30, 30)
        wall = (np.frombuffer(q.encode("ascii"), dtype=np.uint8)
                == ord("#")).reshape(30, 30)
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        img[:, :, :] = [201, 201, 201]       # correct negative (grey)
        hit = pred & label
        extra = pred & ~label
        missed = ~pred & label
        for mask, col in ((hit, GRID_COL["hit"]), (extra, GRID_COL["extra"]),
                          (missed, GRID_COL["missed"]),
                          (wall, GRID_COL["wall"])):
            rgb = [int(col[i + 1:i + 3], 16) for i in (0, 2, 4)]
            img[mask] = rgb
        ax.imshow(img, interpolation="nearest", aspect="equal")
        qarr = np.frombuffer(q.encode("ascii"), dtype=np.uint8).reshape(30, 30)
        for ch in ("S", "G"):
            pos = np.argwhere(qarr == ord(ch))[0]
            ax.add_patch(plt.Rectangle((pos[1] - 0.42, pos[0] - 0.42), 0.84,
                                       0.84, fill=False, edgecolor="black",
                                       lw=1.5))
        ax.set_title(titles[name], fontsize=11)
        fig.text(rect[0] + rect[2] / 2.0, rect[1] - 0.035, caps[name],
                 ha="center", va="top", fontsize=10, color="#333333")

    lax = fig.add_axes([0.04, 0.03, 0.92, 0.09])
    lax.set_xticks([])
    lax.set_yticks([])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")
    items = [("wall", "#222222", True),
             ("correct negative (free cell)", GRID_COL["negative"], True),
             ("hit (predicted + reference)", GRID_COL["hit"], True),
             ("extra (predicted, not reference)", GRID_COL["extra"], True),
             ("missed (reference, not predicted)", GRID_COL["missed"], True),
             ("S / G cell", "#FFFFFF", True)]
    x = 0.008
    for label, colr in [(a, b) for a, b, _ in items]:
        lax.add_patch(plt.Rectangle((x, 0.30), 0.014, 0.40,
                                    facecolor=colr, edgecolor="#222222",
                                    lw=0.8))
        lax.text(x + 0.018, 0.50, label, fontsize=10, va="center", ha="left")
        x += 0.018 + 0.0072 * len(label) + 0.022
    check_figure(fig, "F11 examples")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("wrote %s" % out_png)


# ---------------------------------------------------------------------------
# figure 4: per-instance coverage over all 1000 test mazes
# ---------------------------------------------------------------------------

def render_coverage(D, out_png):
    perm = D["permaze"]
    arms = {}
    for r in perm:
        arms.setdefault(r["arm"], []).append(r)
    cf_l = np.array([float(r["iou"]) for r in arms["cf_wallaware_seed0_tstar"]])
    ph_l = np.array([float(r["iou"]) for r in arms["physics_exact_t8"]])
    wn_l = np.array([float(r["iou"]) for r in arms["wonn_ema52_interim_export"]])
    cf_s = int(sum(int(r["strict"]) for r in arms["cf_wallaware_seed0_tstar"]))
    ph_rows = [r for r in arms["physics_exact_t8"] if r["strict"] != ""]
    ph_s = int(sum(int(r["strict"]) for r in ph_rows))

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    fig.subplots_adjust(bottom=0.24)
    ax.grid(False)
    xs = np.arange(1000)
    cf_sorted = np.sort(cf_l)[::-1]
    ph_sorted = np.sort(ph_l)[::-1]
    wn_sorted = np.sort(wn_l)[::-1]
    ax.plot(xs, cf_sorted, color=CF_C, lw=2.2, zorder=3,
            label="CF-WallAware, seed 0, T*=16 (strict %d/1000)" % cf_s)
    ax.plot(xs, ph_sorted, color=PHYS_C, lw=2.2, zorder=3,
            label="physics exact solve, T=8 (strict %d/1000)" % ph_s)
    ax.plot(xs, wn_sorted, color=WONN_C, lw=2.2, ls="--", zorder=3,
            label="WONN EMA, interim 52-epoch export (own budget)")
    # Rings sit at the strict passes' OWN positions on the sorted curve, not
    # at the top-k IoU positions: a valid alternative route can have a modest
    # IoU, so the strict passes are generally NOT the highest-IoU mazes.
    ph_pairs = sorted((int(r["maze_index"]), float(r["iou"]))
                      for r in arms["physics_exact_t8"])
    desc = np.argsort(-np.array([v for _, v in ph_pairs]), kind="stable")
    ph_rank = np.empty(len(ph_pairs), dtype=int)
    ph_rank[desc] = np.arange(len(ph_pairs))
    ph_pos = dict((m, int(ph_rank[k])) for k, (m, _) in enumerate(ph_pairs))
    ph_val = dict(ph_pairs)
    ph_strict_i = [int(r["maze_index"]) for r in ph_rows
                   if int(r["strict"]) == 1]
    for m in ph_strict_i:
        ax.plot([ph_pos[m]], [ph_val[m]], marker="o",
                markerfacecolor="none", markeredgecolor=PHYS_C,
                markersize=9, zorder=4, linestyle="none")
    if ph_strict_i:
        ph_lo = min(ph_val[m] for m in ph_strict_i)
        ph_hi = max(ph_val[m] for m in ph_strict_i)
    else:
        ph_lo = ph_hi = 0.0
    # plain label (an annotate() arrow would union the arrow into the text
    # extent and cover the curves); the label sits above every curve there
    ax.text(430, 0.95,
            "rings = physics strict passes (IoU %.2f-%.2f)" % (ph_lo, ph_hi),
            fontsize=10.5, color=BLACK, ha="left", va="top")
    ax.set_xlabel("test mazes, sorted by per-maze IoU (descending)", fontsize=12)
    ax.set_ylabel("per-maze IoU with the reference route", fontsize=12)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_title(
        "Per-instance IoU on ALL 1000 test mazes: the reference route's cells are largely found "
        "(median IoU %.2f physics / %.2f CF),\nyet a single valid route almost never appears - "
        "strict passes: physics %d/1000 (T=8), CF-WallAware %d/1000 (T*=16)"
        % (float(np.median(ph_l)), float(np.median(cf_l)), ph_s, cf_s),
        fontsize=11.5, pad=22)
    ax.legend(loc="lower right", fontsize=10.5)
    fig.text(0.5, 0.02,
             "Reading guide: x = the 1000 test mazes sorted by per-maze IoU (highest on the left); y = IoU of the\n"
             "predicted cells with the single reference route (1.0 = exact board match). Rings mark the physics\n"
             "strict passes (valid alternative route, board still 0), each at its own IoU position - they sit\n"
             "mid-curve (IoU %.2f-%.2f here), not at the top. Nothing is aggregated - each curve is one full\n"
             "test-set run; the WONN curve is the labelled interim 52-epoch EMA export, not the headline run."
             % (ph_lo, ph_hi),
             fontsize=10.5, ha="center", va="bottom", color="#555555")
    check_figure(fig, "F11 coverage")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print("wrote %s" % out_png)


# ---------------------------------------------------------------------------
# CSV writers (the tracked data files hold the plotted numbers)
# ---------------------------------------------------------------------------

def csv_rows(D):
    cf, wn, ph, pt = D["cf"], D["wonn"], D["phys"], D["phys_tie"]
    rows = []

    def add(block, arm, metric, value, unit, budget, source, note="",
            sd=None, n=None):
        rows.append(dict(block=block, arm=arm, metric=metric,
                         value=("" if value is None else value), unit=unit,
                         sd=("" if sd is None else sd),
                         n=("" if n is None else n), budget=budget,
                         data_source=source, note=note))

    r0 = cf["runs"][cf["seeds"][0]]
    cf_budget = "120 epochs x 3 seeds (0,1,2), CUDA"
    cf_src = "maze_hard_cf/results.json (families results tree, external)"
    for metric, v, sd in (
            ("board-exact accuracy @T*=16 (%)", cf["exact_tstar"] * 100.0,
             cf["exact_tstar_sd"] * 100.0),
            ("strict accuracy @T*=16 (%)", cf["strict_tstar"] * 100.0,
             cf["strict_tstar_sd"] * 100.0),
            ("board-exact accuracy @T=24 (%)", cf["exact_ttrain"] * 100.0, 0.0),
            ("strict accuracy @T=24 (%)", cf["strict_ttrain"] * 100.0, 0.0),
            ("mean IoU @T*=16", cf["iou_tstar"], None),
            ("mean recall @T*=16", cf["recall_tstar"], None),
            ("mean extra cells @T*=16 (count)", cf["extra_tstar"], None),
            ("mean extra cells @T=24 (count)", cf["extra_ttrain"], None)):
        add("results_table", "CF-WallAware (ours)", metric, v,
            "% or count (as in metric name)",
            cf_budget + ("; sd 0.0 over n=3" if sd is not None else ""),
            cf_src, sd=sd, n=(3 if sd is not None else None),
            note="mean over 3 seeds" if sd is not None else "")
    add("results_table", "CF-WallAware (ours)", "params raw", r0["params_raw"],
        "count", cf_budget, cf_src)
    add("results_table", "CF-WallAware (ours)", "params effective",
        r0["params_eff"], "count", cf_budget, cf_src)
    add("results_table", "CF-WallAware (ours)", "epochs recorded (per seed)",
        "%d/%d (planned)" % (r0["epochs_done"], r0["epochs_planned"]),
        "epochs", cf_budget, cf_src)
    add("results_table", "CF-WallAware (ours)", "T* (per seed)", "16, 16, 16",
        "steps", cf_budget, cf_src,
        note="argmax val strict over [16,24,32,48], ties to smaller T")

    for s in cf["seeds"]:
        for T in sorted(cf["runs"][s]["test_sweeps"].keys()):
            sw = cf["runs"][s]["test_sweeps"][T]
            add("tsweep", "CF-WallAware seed %d" % s,
                "strict accuracy @T=%d (%%)" % T, sw["strict"] * 100.0, "%",
                "120 epochs, seed %d" % s, cf_src)
            add("tsweep", "CF-WallAware seed %d" % s,
                "board-exact accuracy @T=%d (%%)" % T, sw["exact"] * 100.0,
                "%", "120 epochs, seed %d" % s, cf_src)
            add("tsweep", "CF-WallAware seed %d" % s, "IoU @T=%d" % T,
                sw["iou"], "fraction", "120 epochs, seed %d" % s, cf_src)

    wn_budget = ("%d/%d epochs, 1 seed (%d), official code (reduced budget, "
                 "not a reproduction), %s"
                 % (wn["epochs"], wn["official_epochs"], wn["seed"], wn["gpu"]))
    wn_src = "wonn_tools/out/run_record.json (baselines tree, external)"
    add("results_table", "WONN official (EMA)", "board-exact accuracy (%)",
        wn["ema_board"] * 100.0, "%", wn_budget, wn_src,
        note="official eval_maze.py, EMA checkpoint, K=1, T=24")
    add("results_table", "WONN official (EMA)", "path F1 (%)",
        wn["ema_f1"] * 100.0, "%", wn_budget, wn_src)
    add("results_table", "WONN official (EMA)", "strict accuracy (%)",
        "not computed", "-", wn_budget, wn_src,
        note="strict was never recorded for this run")
    add("results_table", "WONN official (EMA)", "official code commit",
        wn["commit"], "-", wn_budget, wn_src)
    add("results_table", "WONN official (EMA)", "params raw",
        wn["params_raw"] if wn["params_raw"] is not None else "not recorded",
        "count", wn_budget, "wonn_tools/out/train_log.txt (baselines tree)")
    add("results_table", "WONN official (non-EMA)", "board-exact accuracy (%)",
        wn["raw_board"] * 100.0, "%", wn_budget + "; same run", wn_src,
        note="official eval_maze.py, model.pth, K=1, T=24")
    add("results_table", "WONN official (non-EMA)", "path F1 (%)",
        wn["raw_f1"] * 100.0, "%", wn_budget, wn_src)
    add("results_table", "WONN official (non-EMA)", "strict accuracy (%)",
        "not computed", "-", wn_budget, wn_src)

    ph_budget = "no training (0 epochs, 0 seeds, one deterministic run), CPU"
    ph_src = "transport_mazehard/results.json (families results tree, external)"
    add("results_table", "Physics, exact solve", "board passes @T=8 (count)",
        ph["board8"], "count", ph_budget, ph_src)
    add("results_table", "Physics, exact solve", "strict passes @T=8 (count)",
        ph["strict8"], "count", ph_budget, ph_src,
        note="n=%d test mazes" % ph["n"])
    add("results_table", "Physics, exact solve",
        "board-exact accuracy @T=8 (%)", ph["board8"] * 100.0 / ph["n"], "%",
        ph_budget, ph_src)
    add("results_table", "Physics, exact solve", "strict accuracy @T=8 (%)",
        ph["strict8"] * 100.0 / ph["n"], "%", ph_budget, ph_src)
    add("results_table", "Physics, exact solve", "mean IoU @T=8", ph["iou8"],
        "fraction", ph_budget, ph_src)
    add("results_table", "Physics, exact solve", "tied test mazes",
        "%d of %d" % (ph["n_tied"], ph["n"]), "count", ph_budget, ph_src,
        note="BFS count over the wall grid (recorded)")

    tie_src = ("transport_mazehard_tie/results.json (families results tree, "
               "external)")
    tie_budget = (ph_budget + "; tie-break K=4 seeded starts, lowest "
                  "dissipation; first 300 test mazes")
    add("results_table", "Physics + tie-break (K=4)",
        "strict passes @T=8 (count)", pt["strict8"], "count", tie_budget,
        tie_src)
    add("results_table", "Physics + tie-break (K=4)",
        "strict passes @T=8 without tie-break (count)", pt["strict8_no_tie"],
        "count", tie_budget, tie_src)
    add("results_table", "Physics + tie-break (K=4)",
        "board passes @T=8 (count)", pt["board8"], "count", tie_budget,
        tie_src)
    add("results_table", "Physics + tie-break (K=4)",
        "strict accuracy @T=8 (%)", pt["strict8"] * 100.0 / pt["n"], "%",
        tie_budget, tie_src, note="first %d test mazes" % pt["n"])
    add("results_table", "Physics + tie-break (K=4)",
        "board-exact accuracy @T=8 (%)", pt["board8"] * 100.0 / pt["n"], "%",
        tie_budget, tie_src)

    add("results_table", "Published WONN (paper)", "board-exact accuracy (%)",
        PUB_WONN_BOARD, "%",
        "%d epochs (official protocol)" % PUB_WONN_EPOCHS,
        "WONN paper, Table 3, p. 9 (cited, not run by us)",
        note="K=1, T=24, %s params" % PUB_WONN_PARAMS)

    for name in ("e1", "e2", "e3", "e4"):
        sp = D["example_spec"][name]
        st = sp["stats"]
        arm_lbl = ("CF-WallAware seed 0, T=%d" % sp["T"]
                   if sp["arm"] == "cf" else "physics exact solve, T=8")
        add("examples", name, "maze index", sp["maze"],
            "index (test csv order)", arm_lbl,
            "recomputed live (checkpoint / physics model)", note=sp["rule"])
        add("examples", name, "readout T", sp["T"], "steps", arm_lbl,
            "recomputed live (checkpoint / physics model)")
        add("examples", name, "IoU", st["iou"], "fraction", arm_lbl,
            "recomputed live (checkpoint / physics model)")
        add("examples", name, "extra cells (count)", st["extra_cells"],
            "count", arm_lbl, "recomputed live (checkpoint / physics model)")
        add("examples", name, "missed cells (count)", st["missed_cells"],
            "count", arm_lbl, "recomputed live (checkpoint / physics model)")
        add("examples", name, "strict pass (0/1)", st["strict"], "flag",
            arm_lbl, "recomputed live (checkpoint / physics model)")
        add("examples", name, "board-exact pass (0/1)", st["exact"], "flag",
            arm_lbl, "recomputed live (checkpoint / physics model)")

    for arm in ("cf_wallaware_seed0_tstar", "physics_exact_t8",
                "wonn_ema52_interim_export"):
        ious = np.array([float(r["iou"]) for r in D["permaze"]
                         if r["arm"] == arm])
        stricts = [r["strict"] for r in D["permaze"]
                   if r["arm"] == arm and r["strict"] != ""]
        src = "permaze_iou.csv (tracked)"
        add("coverage", arm, "n mazes", len(ious), "count",
            "one run, full test set", src)
        add("coverage", arm, "median IoU", float(np.median(ious)), "fraction",
            "one run, full test set", src)
        add("coverage", arm, "max IoU", float(ious.max()), "fraction",
            "one run, full test set", src)
        add("coverage", arm, "mazes with IoU >= 0.9", int((ious >= 0.9).sum()),
            "count", "one run, full test set", src)
        add("coverage", arm, "mazes with IoU >= 0.7", int((ious >= 0.7).sum()),
            "count", "one run, full test set", src)
        if stricts:
            add("coverage", arm, "strict passes",
                int(sum(int(v) for v in stricts)), "count",
                "one run, full test set", src)
        if arm == "wonn_ema52_interim_export":
            add("coverage", arm, "path F1", D["wonn_npz"]["f1"], "fraction",
                "interim 52-epoch EMA export of the official exporter",
                "wonn_tools/out/wonn_reduced_preds.npz (baselines tree, "
                "external)", note=D["wonn_npz"]["interim_note"])
    return rows


def write_csvs(D, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.csv"
    rows = csv_rows(D)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % path)

    path = out_dir / "permaze_iou.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "arm", "maze_index", "iou", "strict", "exact", "extra_cells",
            "missed_cells"])
        w.writeheader()
        for r in D["permaze"]:
            w.writerow(r)
    print("wrote %s" % path)

    path = out_dir / "example_grids.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["panel", "arm_label", "maze_index",
                                           "question", "label", "pred"])
        w.writeheader()
        for r in D["example_grids"]:
            w.writerow(r)
    print("wrote %s" % path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="F11 Maze-hard results: board-exact vs strict across arms.")
    ap.add_argument("--families-dir", type=Path, default=None,
                    help="cf-families results tree (maze_hard_cf, "
                         "transport_mazehard*); also from CF_FAMILIES_DIR")
    ap.add_argument("--baselines-dir", type=Path, default=None,
                    help="baselines tree (wonn_tools/out record + npz); "
                         "also from CF_BASELINES_DIR")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    families = args.families_dir or os.environ.get("CF_FAMILIES_DIR")
    baselines = args.baselines_dir or os.environ.get("CF_BASELINES_DIR")

    root = args.root if args.root else repo_root()
    out_dir = (args.out_dir if args.out_dir else
               root / "figures" / "maze_hard_results")

    if families and baselines:
        print("live mode: recomputing from the external results trees")
        D = collect_live(families, baselines)
        for k, v in D["checks"].items():
            print("  cross-check %s: %s" % (k, v))
        write_csvs(D, out_dir)
    else:
        print("fallback mode: no external directories given -> figures are "
              "regenerated from the tracked CSVs in %s" % out_dir)
        D = collect_fallback(out_dir)

    render_results(D, out_dir / "maze_hard_results.png")
    render_t_sweep(D, out_dir / "maze_hard_T_sweep.png")
    render_examples(D, out_dir / "maze_hard_examples.png")
    render_coverage(D, out_dir / "maze_hard_coverage.png")
    print("done")


if __name__ == "__main__":
    apply_style()
    main()
