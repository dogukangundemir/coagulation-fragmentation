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


def evaluate_model(model, d, T, thresh=None):
    thresh = HYPERPARAMS["threshold"] if thresh is None else thresh
    probs = predict_probs(model, d, T)
    rows = [evaluate_mask(probs[k] >= thresh, d.walls[k], d.starts[k], d.goals[k], d.y[k].numpy())
            for k in range(len(d))]
    assert all(r["strict"] == r["exact"] for r in rows), "perfect maze: strict must equal exact"
    return summarize(rows), rows, probs


def train_model(make_model, T, seed, tag, train_ds, val_ds, pos_weight=None,
                results_dir=None, verbose=True):
    hp = HYPERPARAMS
    if pos_weight is None:
        density = train_ds.y.mean().item()
        pos_weight = (1.0 - density) / density
    results_dir = Path(results_dir) if results_dir is not None else Path.cwd() / "week1_results"
    results_dir.mkdir(exist_ok=True)
    set_seed(seed)
    model = make_model().to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"] * len(loader))
    pw = torch.tensor(float(pos_weight), device=DEVICE)
    hist = {k: [] for k in ["train_loss", "val_strict", "val_iou", "val_covers_path"]}
    best_key, best_state = (-1.0, -1.0), None
    t_start = time.time()
    for ep in range(hp["epochs"]):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(xb, T=T), yb, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip_norm"])
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(xb)
        v, _, _ = evaluate_model(model, val_ds, T)
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
    model.load_state_dict(best_state)
    hist["train_time_s"] = time.time() - t_start
    ckpt = results_dir / ("%s_seed%d.pt" % (tag, seed))
    torch.save(model.state_dict(), str(ckpt))
    if verbose:
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
