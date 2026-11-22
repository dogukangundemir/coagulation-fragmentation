"""Train and evaluate B-WallAware, Family B's Becker-Doring commitment ladder.

B-WallAware = LadderCFNet (cf/models/ladder.py) with K = 4 rungs, the
wall-aware option on and the detailed-balance rate parameterisation the
family card recommends (the free-rates arm is the ablation, not trained
here). The run follows the shared curriculum protocol: level-5 maze
splits via cf.data.curriculum (4000/1000/1000, probe levels 7 and 9 at
1000 each, split seed 0), T_train = 8, the cf.train hyperparameters
(Adam lr 1e-3, cosine decay to 0 per step, grad-norm clip 1.0, batch 32,
BCEWithLogits with the train pos_weight, threshold 0.5, model selection
on best validation strict, ties by validation IoU), with the cosine
schedule planned for 160 epochs. Training runs under a wall-clock budget
with resumable per-epoch checkpoints, so a stop before 160 epochs is a
reduced-budget run, not a lost one: relaunching the same command resumes
from the last completed epoch.

After training the script records: raw and effective parameter counts,
the validation strict sweep over the candidate horizons and the
validation-selected horizon T* = argmax val strict (ties: smaller T),
test strict at T = 8 and at T*, probe-7/9 strict at both horizons, the
test-time T sweep, and the free energy V over the rollout steps on the
first 64 test mazes with the fixed-rate phases marked (the slow phase
refreshes the rates every r_s steps; between refreshes the chemistry
relaxes under frozen rates and V is a Lyapunov function). One seed only,
so every number is preliminary. Results are written atomically to
<out>/ladder_metrics.json.

Usage:
  python -m scripts.train_ladder                      # 30-min budget, seed 0
  python -m scripts.train_ladder --eval-only          # re-evaluate the checkpoint
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cf.train as train_mod
import run_week2
from cf.models.ladder import LadderCFNet
from cf.models.rcnn import n_params
from select_T_val import pick_T_star

MODEL_NAME = "BD-WallAware"
RATE_MAX = 100.0


class LadderStable(LadderCFNet):
    """BD-WallAware with the two training-stability guards.

    The plain LadderCFNet poisons its parameters with NaN within ~40
    SGD steps of this protocol (reproduced on CPU; the loss stays finite,
    so it is a backward, not a forward, failure): on near-empty rungs the
    backward through the joint-flux cap ratio zeta*c/(O+eps) is
    O(1/O^2) stiff (eps = 1e-12), and on steep learned profiles the
    backward through b = mu*Q_k/Q_{k+1} explodes with 1/Q_{k+1}^2.

    Guard 1: the cap Lambda is treated as a constant within a micro-step
    (computed under no_grad) -- it is a numerical safeguard in the same
    spirit as a positivity projection, and every chemistry gradient (through
    the rates a, b and the rungs c) is kept. Guard 2: the rates are clamped
    at RATE_MAX, well above the trained range but below the explosive zone;
    the gradient path is cut exactly where the rates leave that range.
    Mass conservation, positivity and the detailed-balance structure are
    untouched by both guards, and the forward behaviour is identical while
    the rates stay under the clamp.
    """

    def refresh_rates(self, x, c):
        a, b = LadderCFNet.refresh_rates(self, x, c)
        return a.clamp(max=RATE_MAX), b.clamp(max=RATE_MAX)

    def micro_step(self, c, rates):
        a, b = rates
        B, hid, K, H, W = c.shape
        c1 = c[:, :, 0]
        r = a * c1.unsqueeze(2) * c[:, :, :-1]
        d = b * c[:, :, 1:]
        O = torch.cat([(2.0 * r[:, :, 0] + r[:, :, 1:].sum(dim=2)).unsqueeze(2),
                       r[:, :, 1:] + d[:, :, :-1],
                       d[:, :, -1].unsqueeze(2)], dim=2)
        with torch.no_grad():
            ratio = torch.where(O > 0, self.zeta * c / (O + self.eps),
                                torch.full_like(c, float("inf")))
            lam = torch.clamp(ratio.amin(dim=2), max=1.0)
        J = lam.unsqueeze(2) * (r - d)
        d1 = -2.0 * J[:, :, 0] - J[:, :, 1:].sum(dim=2)
        c_new = torch.cat([(c[:, :, 0] + d1).unsqueeze(2),
                           c[:, :, 1:-1] + (J[:, :, :-1] - J[:, :, 1:]),
                           (c[:, :, -1] + J[:, :, -1]).unsqueeze(2)], dim=2)
        return c_new, J
DEFAULT_SWEEP_T = [1, 2, 4, 8, 12, 16, 24, 32]
DEFAULT_EPOCHS = 160                 # planned budget of the cosine schedule
FAST = dict(n_train=200, n_val=100, n_test=100, n_probe=100, epochs=2,
            sweep_T=[1, 4, 8], budget_minutes=2.0)

READOUT_RULE = (
    "readout: logits = 1x1 conv over the rung stack, masked to -20.0 in "
    "walls; a cell is on the predicted path iff sigmoid(logit) >= 0.5; a "
    "maze counts as strict only if the predicted support forms a "
    "wall-free simple S->G path of exactly the BFS shortest-path length "
    "(on these perfect mazes strict == exact, asserted); T* = argmax "
    "validation strict over the candidate horizons, ties to the smaller T"
)
SEEDS_NOTE = "preliminary, 1 seed, reduced budget (%d/%d epochs)"


def make_model(hid=16, K=4):
    """The B-WallAware arm: wall-aware, detailed-balance ladder with the
    two training-stability guards of LadderStable (see its docstring)."""
    return LadderStable(hid=hid, K=K, detailed_balance=True, wall_aware=True)


def budget_label(epochs_done, planned_epochs=DEFAULT_EPOCHS):
    """Every reported number carries its training budget."""
    return SEEDS_NOTE % (epochs_done, planned_epochs)


def rate_refresh_marks(T, r_s=2):
    """(refresh_steps, phase_of_step) of the slow phase over steps 1..T.

    Mirrors LadderCFNet.forward / logits_at exactly: the rates refresh
    before outer steps 1, 1 + r_s, 1 + 2*r_s, ..., and the steps between
    two refreshes form one fixed-rate phase (frozen rates, V Lyapunov).
    """
    refresh = [1 + i * r_s for i in range(T) if 1 + i * r_s <= T]
    phase = [(t - 1) // r_s for t in range(1, T + 1)]
    return refresh, phase


def load_data(cfg):
    """The shared curriculum pipeline (delegates to run_week2.load_data)."""
    return run_week2.load_data(cfg)


def evaluate_arm(model, data, sweep_T, T_train=8, n_V=64, T_V=32):
    """All post-training measurements for one trained ladder."""
    val_sweep = train_mod.sweep_model(model, data["val"], sweep_T)
    val_strict = {int(T): s["strict"] for T, s in val_sweep.items()}
    T_star = pick_T_star(val_strict)
    sweeps = {split: train_mod.sweep_model(model, data[split], sweep_T)
              for split in ("test", "probe7", "probe9")}
    # free energy over the rollout on n_V test mazes, fixed-rate phases marked
    xs = data["test"].x[:n_V].to(train_mod.DEVICE)
    _, trace = model.logits_at(xs, [T_V], record=True)
    refresh, phase = rate_refresh_marks(T_V, model.r_s)
    assert len(trace["V"]) == T_V and len(phase) == T_V
    assert len(trace["V_micro"]) == T_V * model.n_f
    energy = dict(n_mazes=n_V, T_V=T_V,
                  V_by_outer_step=trace["V"], V_by_micro_step=trace["V_micro"],
                  residual_flux_by_outer_step=trace["residual_flux"],
                  rate_refresh_steps=refresh, fixed_rate_phase_by_step=phase)
    return dict(val_strict_by_T=val_strict, T_star=T_star,
                val_strict_at_T_star=val_strict[T_star],
                sweeps=sweeps, energy=energy)


def hardware_info():
    caps = dict(power_limit_w=None, default_power_limit_w=None,
                memory_fraction=0.70,
                clock_cap_note="SM clock cap not applied (needs admin; "
                               "power + memory caps active)")
    if train_mod.DEVICE.type == "cuda":
        caps["power_limit_w"], caps["default_power_limit_w"] = 217, 310
    return dict(device=str(train_mod.DEVICE),
                device_name=(torch.cuda.get_device_name(0)
                             if train_mod.DEVICE.type == "cuda" else "CPU"),
                torch_version=torch.__version__,
                git_commit=run_week2.git_commit(), gpu_caps=caps)


def run(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    planned_epochs = FAST["epochs"] if args.fast else args.epochs
    sweep_T = FAST["sweep_T"] if args.fast else args.sweep_T
    budget = FAST["budget_minutes"] if args.fast else args.budget_minutes
    sizes = FAST if args.fast else dict(n_train=args.n_train, n_val=args.n_val,
                                        n_test=args.n_test, n_probe=args.n_probe)
    # the cosine schedule is planned over the FULL epoch budget even when the
    # wall-clock budget stops the run early
    train_mod.HYPERPARAMS["epochs"] = planned_epochs
    data = load_data(dict(sizes))
    tag = "%s_seed%d" % (MODEL_NAME, args.seed)
    ckpt_path = out_dir / ("%s.ckpt.pt" % tag)
    print("[%s] device %s, planned %d epochs, budget %.1f min"
          % (tag, train_mod.DEVICE, planned_epochs, budget))

    n_raw = n_params(make_model(args.hid, args.K))
    if args.eval_only:
        ck = torch.load(str(ckpt_path), map_location="cpu")
        model = make_model(args.hid, args.K).to(train_mod.DEVICE)
        state = ck.get("best_state") or ck["model_state"]
        model.load_state_dict(state)
        hist = dict(ck["hist"])
        epochs_done = len(hist.get("train_loss", []))
        print("[%s] eval-only: restored best state from %s (epochs done %d)"
              % (tag, ckpt_path.name, epochs_done))
    else:
        xb = data["train"].x[:train_mod.HYPERPARAMS["batch_size"]].to(train_mod.DEVICE)
        yb = data["train"].y[:train_mod.HYPERPARAMS["batch_size"]].to(train_mod.DEVICE)
        n_eff = run_week2.effective_n_params(make_model(args.hid, args.K), xb, yb,
                                             args.T_train,
                                             pos_weight=run_week2.train_pos_weight(
                                                 data["train"]))
        print("[%s] %d params (%d effective)" % (tag, n_raw, n_eff))
        model, hist = train_mod.train_model(
            lambda: make_model(args.hid, args.K), T=args.T_train, seed=args.seed,
            tag=MODEL_NAME, train_ds=data["train"], val_ds=data["val"],
            pos_weight=run_week2.train_pos_weight(data["train"]),
            results_dir=out_dir, checkpoint_path=ckpt_path, run_id=tag,
            max_minutes=budget)
        epochs_done = hist.get("epochs_done", len(hist.get("train_loss", [])))
    print("[%s] evaluating: val sweep / T* selection, test, probes, T sweep, V trace"
          % tag)
    rec = evaluate_arm(model, data, sweep_T, T_train=args.T_train,
                       n_V=args.V_mazes, T_V=args.V_T)

    label = budget_label(epochs_done, planned_epochs)
    T_star = rec["T_star"]
    payload = dict(
        label=label, n_seeds=1, readout_rule=READOUT_RULE,
        config=dict(model=MODEL_NAME, hid=args.hid, K=args.K,
                    detailed_balance=True, wall_aware=True, seed=args.seed,
                    gradient_guards="flux cap detached within micro-step; "
                                    "rates clamped at %.0f" % RATE_MAX,
                    T_train=args.T_train, sweep_T=list(sweep_T),
                    epochs_planned=planned_epochs,
                    budget_minutes=budget, out=str(out_dir), fast=args.fast),
        hyperparams=dict(train_mod.HYPERPARAMS),
        hardware=hardware_info(),
        params=dict(raw=n_raw,
                    effective=(n_eff if not args.eval_only else "not run (eval-only)")),
        training=dict(epochs_done=epochs_done, best_epoch=hist.get("best_epoch"),
                      stopped_early=hist.get("stopped_early"),
                      seconds=hist.get("seconds"), label=label),
        selection=dict(val_strict_by_T={str(k): v for k, v in rec["val_strict_by_T"].items()},
                        T_star=T_star, val_strict_at_T_star=rec["val_strict_at_T_star"]),
        test_strict_at_T_train=rec["sweeps"]["test"][args.T_train]["strict"],
        test_strict_at_T_star=rec["sweeps"]["test"][T_star]["strict"],
        probe7_strict_at_T_train=rec["sweeps"]["probe7"][args.T_train]["strict"],
        probe7_strict_at_T_star=rec["sweeps"]["probe7"][T_star]["strict"],
        probe9_strict_at_T_train=rec["sweeps"]["probe9"][args.T_train]["strict"],
        probe9_strict_at_T_star=rec["sweeps"]["probe9"][T_star]["strict"],
        sweeps={sp: {str(T): s for T, s in sw.items()} for sp, sw in rec["sweeps"].items()},
        energy=rec["energy"])
    out_path = run_week2.atomic_write_json(out_dir / "ladder_metrics.json", payload)
    print(print_table(payload))
    return payload, out_path


def print_table(p):
    cfg, tr = p["config"], p["training"]
    eff = p["params"]["effective"]
    eff_s = str(eff) if isinstance(eff, str) else "%d" % eff
    lines = ["%-12s hid %d K %d seed %d | %d params (%s effective) | %s"
             % (cfg["model"], cfg["hid"], cfg["K"], cfg["seed"],
                p["params"]["raw"], eff_s, p["hardware"]["device"]),
             "budget: %d/%d epochs, %.1f min wall-clock budget, stopped_early=%s | %s"
             % (tr["epochs_done"], cfg["epochs_planned"], cfg["budget_minutes"],
                tr["stopped_early"], p["label"]),
             "%-10s %6s %6s" % ("set", "T=%d" % cfg["T_train"], "T*=%d" % p["selection"]["T_star"])]
    for name, p8, ps in [
            ("test", p["test_strict_at_T_train"], p["test_strict_at_T_star"]),
            ("probe7", p["probe7_strict_at_T_train"], p["probe7_strict_at_T_star"]),
            ("probe9", p["probe9_strict_at_T_train"], p["probe9_strict_at_T_star"])]:
        lines.append("%-10s %6.4f %6.4f" % (name, p8, ps))
    lines.append("T sweep (test strict): " + "  ".join(
        "T%s=%.3f" % (T, s["strict"]) for T, s in sorted(p["sweeps"]["test"].items(),
                                                         key=lambda kv: int(kv[0]))))
    lines.append("V trace on %d test mazes to T=%d, rate refreshes at steps %s"
                 % (p["energy"]["n_mazes"], p["energy"]["T_V"],
                    p["energy"]["rate_refresh_steps"]))
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hid", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T-train", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                    help="planned epochs of the cosine schedule (default 160)")
    ap.add_argument("--budget-minutes", type=float, default=30.0,
                    help="wall-clock training budget (resume on relaunch)")
    ap.add_argument("--sweep-T", nargs="+", type=int, default=DEFAULT_SWEEP_T)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-probe", type=int, default=1000)
    ap.add_argument("--V-mazes", type=int, default=64)
    ap.add_argument("--V-T", type=int, default=32)
    ap.add_argument("--out", default="results/ladder_train")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, evaluate the resumable checkpoint")
    ap.add_argument("--fast", action="store_true",
                    help="smoke mode: 200/100/100 mazes, 2 epochs, T sweep 1 4 8")
    return ap.parse_args(argv)


def main(argv=None):
    run(argv)


if __name__ == "__main__":
    main()
