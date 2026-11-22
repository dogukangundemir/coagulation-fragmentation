"""Training / evaluation utilities shared by all Week-2+ experiments.

predict_probs / evaluate_model / train_model / plot_history are copied
verbatim from CF_Milestone2_Week1_HF.ipynb Section 5; the notebook's globals
(DEVICE, RESULTS, ds, pos_weight) are explicit arguments here.  The
hyperparameters are the packet 9.3 configuration: Adam lr 1e-3, cosine decay
to 0 per step, grad-norm clip 1.0, batch 32, 40 epochs, BCEWithLogits with
pos_weight = neg/pos of the train labels.
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cf.metrics import evaluate_mask, summarize
from cf.utils import set_seed

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

HYPERPARAMS = dict(
    optimizer="Adam", lr=1e-3, lr_schedule="cosine decay to 0 (per step)", grad_clip_norm=1.0,
    batch_size=32, epochs=40,
    loss="BCEWithLogits over all cells, pos_weight = neg/pos ratio of train labels",
    pos_weight=None,            # None -> computed from the train split's label density
    threshold=0.5,
    T_train_recurrent=8, model_selection="best val strict (ties: val IoU)",
    seeds=[0, 1, 2],
)
MODEL_CFG = {"RecurrentCNN": dict(hid=22), "PlainCNN": dict(hid=18, depth=6)}


@torch.no_grad()
def predict_probs(model, d, T, bs=500):
    model.eval()
    out = [torch.sigmoid(model(d.x[i:i + bs].to(DEVICE), T=T)).cpu() for i in range(0, len(d), bs)]
    return torch.cat(out).numpy()


def evaluate_model(model, d, T, thresh=None, assert_strict_equals_exact=True):
    thresh = HYPERPARAMS["threshold"] if thresh is None else thresh
    probs = predict_probs(model, d, T)
    rows = [evaluate_mask(probs[k] >= thresh, d.walls[k], d.starts[k], d.goals[k], d.y[k].numpy())
            for k in range(len(d))]
    # the assertion only holds for perfect mazes: Maze-hard grids admit several
    # shortest paths, so a prediction can be strict-valid yet differ from the label.
    if assert_strict_equals_exact:
        assert all(r["strict"] == r["exact"] for r in rows), "perfect maze: strict must equal exact"
    return summarize(rows), rows, probs


def train_model(make_model, T, seed, tag, train_ds, val_ds, pos_weight=None,
                results_dir=None, verbose=True, assert_strict_equals_exact=True,
                max_minutes=None, checkpoint_path=None, _clock=None):
    # budget checks use an injectable monotonic clock (tests pass a fake one);
    # time.time() is kept only for the human-facing train_time_s
    clock = time.monotonic if _clock is None else _clock
    hp = HYPERPARAMS
    if pos_weight is None:
        density = train_ds.y.mean().item()
        pos_weight = (1.0 - density) / density
    results_dir = Path(results_dir) if results_dir is not None else Path.cwd() / "week1_results"
    results_dir.mkdir(exist_ok=True)
    budget = None if max_minutes is None else float(max_minutes) * 60.0
    set_seed(seed)
    model = make_model().to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    # the cosine schedule is ALWAYS defined over the full planned epochs; a
    # wall-clock budget stop therefore never compresses the LR schedule of
    # the steps that did run
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"] * len(loader))
    pw = torch.tensor(float(pos_weight), device=DEVICE)
    hist = {k: [] for k in ["train_loss", "val_strict", "val_iou", "val_covers_path"]}
    best_key, best_state = (-1.0, -1.0), None
    start_ep = 0
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        # resume: restore the exact training trajectory (weights, optimizer,
        # per-step scheduler state, history and model-selection state) plus
        # the DataLoader generator, so resumed epochs draw the same shuffled
        # batch order as an uninterrupted run would
        ck = torch.load(str(checkpoint_path), map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        sched.load_state_dict(ck["sched_state"])
        hist = ck["hist"]
        best_key, best_state = ck["best_key"], ck["best_state"]
        start_ep = ck["epoch"] + 1
        if ck.get("gen_state") is not None:
            gen.set_state(ck["gen_state"].cpu())

    def save_ckpt(ep):
        torch.save(dict(model_state=model.state_dict(), opt_state=opt.state_dict(),
                        sched_state=sched.state_dict(),
                        gen_state=gen.get_state(), epoch=ep, hist=hist,
                        best_key=best_key, best_state=best_state),
                   str(checkpoint_path))

    stop = False
    t_start = clock()
    t_wall = time.time()
    for ep in range(start_ep, hp["epochs"]):
        model.train()
        total = 0.0
        for step, (xb, yb) in enumerate(loader):
            # wall-clock budget: stop at the next step boundary; the first step
            # of the run is always allowed to execute
            if (budget is not None and (ep > start_ep or step > 0)
                    and clock() - t_start > budget):
                stop = True
                break
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(xb, T=T), yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip_norm"])
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(xb)
        if stop:
            # mid-epoch budget stop: do NOT save a checkpoint here. The last
            # completed-epoch checkpoint already on disk stays valid, and a
            # resume restarts exactly from that epoch (weights / optimizer /
            # per-step scheduler / shuffle RNG), re-running the interrupted
            # epoch from its start
            break
        v, _, _ = evaluate_model(model, val_ds, T, assert_strict_equals_exact=assert_strict_equals_exact)
        hist["train_loss"].append(total / len(train_ds))
        hist["val_strict"].append(v["strict"])
        hist["val_iou"].append(v["iou"])
        hist["val_covers_path"].append(v["covers_path"])
        if (v["strict"], v["iou"]) > best_key:
            best_key = (v["strict"], v["iou"])
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            hist["best_epoch"] = ep
        if verbose and (ep % 5 == 4 or ep == 0):
            print("  [%s seed %d] epoch %2d | train BCE %.4f | val strict %.3f IoU %.3f covers %.3f"
                  % (tag, seed, ep, hist["train_loss"][-1], v["strict"], v["iou"], v["covers_path"]))
        if checkpoint_path is not None:
            save_ckpt(ep)
    # keep the best-so-far model by the existing selection rule (val strict,
    # ties by val IoU); a budget stop before any completed epoch keeps the
    # current weights (there is no scored epoch yet to select from)
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["train_time_s"] = time.time() - t_wall
    if max_minutes is not None:
        hist["stopped_early"] = stop
        hist["epochs_done"] = len(hist["train_loss"])
        hist["seconds"] = clock() - t_start
    ckpt = results_dir / ("%s_seed%d.pt" % (tag, seed))
    torch.save(model.state_dict(), str(ckpt))
    if verbose and "best_epoch" in hist:
        print("  [%s seed %d] best epoch %d (val strict %.3f) | %.0fs | saved %s"
              % (tag, seed, hist["best_epoch"], best_key[0], hist["train_time_s"], ckpt.name))
    return model, hist


def plot_history(hist, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3))
    a1.plot(hist["train_loss"]); a1.set_yscale("log")
    a1.set_xlabel("epoch"); a1.set_ylabel("train BCE (log)"); a1.set_title(title + ": training loss")
    a2.plot(hist["val_strict"], label="val strict (exact path)")
    a2.plot(hist["val_covers_path"], "--", label="val covers_path (lenient)")
    a2.plot(hist["val_iou"], ":", label="val IoU")
    a2.axvline(hist["best_epoch"], color="grey", lw=0.8)
    a2.set_ylim(0, 1.02); a2.set_xlabel("epoch"); a2.legend(fontsize=8)
    a2.set_title(title + ": validation (grey = selected epoch)")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def sweep_model(model, d, Ts, thresh=None, assert_strict_equals_exact=True):
    """Test-time T-sweep from ONE rollout: {T: summary}.

    All horizons in `Ts` are read out of a single rollout per batch via
    model.logits_at, then scored with the exact evaluate_mask / summarize path
    used by evaluate_model, so sweep_model(model, d, [T])[T] equals the summary
    returned by evaluate_model(model, d, T).
    """
    thresh = HYPERPARAMS["threshold"] if thresh is None else thresh
    model.eval().to(DEVICE)
    Ts = sorted(set(Ts))
    probs = {T: [] for T in Ts}
    for i in range(0, len(d), 500):
        xb = d.x[i:i + 500].to(DEVICE)
        outs = model.logits_at(xb, Ts)          # {T: logits (B, H, W)}
        for T in Ts:
            probs[T].append(torch.sigmoid(outs[T]).cpu())
    summaries = {}
    for T in Ts:
        p = torch.cat(probs[T]).numpy()
        rows = [evaluate_mask(p[k] >= thresh, d.walls[k], d.starts[k], d.goals[k], d.y[k].numpy())
                for k in range(len(d))]
        if assert_strict_equals_exact:
            assert all(r["strict"] == r["exact"] for r in rows), "perfect maze: strict must equal exact"
        summaries[T] = summarize(rows)
    return summaries


@torch.no_grad()
def settling_curve(model, d, T_max, n=64):
    """Per-step settling values on the first `n` mazes of `d` (list of T_max).

    CF models (attribute `fluxes`): mean residual_flux from logits_at(record=True).
    RecurrentCNN: relative state change ||h_t - h_{t-1}|| / ||h_{t-1}||, with the
    documented update h <- h + cell([x, h]) replayed here through the model's
    own modules (rcnn.py is untouched); index t-1 of the list is step t.
    """
    model.eval().to(DEVICE)
    xs = d.x[:n].to(DEVICE)
    if hasattr(model, "fluxes"):
        _, trace = model.logits_at(xs, [T_max], record=True)
        return [float(v) for v in trace["residual_flux"]]
    h = model.encoder(xs)
    curve = []
    for _ in range(T_max):
        h_new = h + model.cell(torch.cat([xs, h], dim=1))
        denom = h.pow(2).sum().sqrt()
        curve.append(float((h_new - h).pow(2).sum().sqrt() / denom) if denom > 0 else 0.0)
        h = h_new
    return curve
