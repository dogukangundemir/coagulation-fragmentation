"""Helper checks for scripts/eval_transport_physics.py (no dataset loading).

The evaluation pipeline of experiment X2's physics half is exercised on small
synthetic perfect mazes (the offline stand-in helpers next to this file),
in the curriculum layout: channels [walkable, start one-hot, goal one-hot],
label path replayed from the provided solution, and the same strict/exact
metrics as every other arm. A stand-in model whose exact solve degenerates
verifies the hold-and-continue handling: a maze whose pressure solve goes
singular is held at its last solved flow (unique on a maze tree), recorded in
the solver-breakdown bookkeeping, and never stops the run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root: cf
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_transport_physics as etp                                  # noqa: E402
from cf.data.curriculum import maze_to_channels, parse_maze, replay_solution  # noqa: E402
from cf.models.transport import TransportCFNet                        # noqa: E402

from maze_synthetic import make_example                               # noqa: E402


class FakeLevelSet:
    """The (x, y, walls, starts, goals, solutions) surface evaluate_level uses."""

    def __init__(self, count=4, n=5, seed=0, min_moves=4):
        rng = np.random.default_rng(seed)
        xs, ys, walls, starts, goals = [], [], [], [], []
        for _ in range(count):
            ascii_maze, solution = make_example(rng, n=n, min_moves=min_moves)
            wall, start, goal = parse_maze(ascii_maze)
            mask, end, legal, revisited = replay_solution(wall, start, solution)
            assert legal and end == goal and not revisited
            xs.append(maze_to_channels(wall, start, goal))
            ys.append(mask)
            walls.append(wall)
            starts.append(start)
            goals.append(goal)
        self.x = torch.from_numpy(np.stack(xs)).float()
        self.y = torch.stack([torch.from_numpy(m).float() for m in ys])
        self.walls, self.starts, self.goals = walls, starts, goals

    def __len__(self):
        return self.x.shape[0]


class BreakdownModel:
    """Transport-model stand-in whose pressure solve breaks for ONE maze.

    Follows the surface evaluate_level uses: edge_masks, initial_state,
    step, read (the spec's fixed-threshold physics readout). Mazes are
    identified by their content fingerprint; the marked maze raises
    torch.linalg.LinAlgError at its marked outer step (and at every later
    attempt), all other steps return finite unit flows on the free edges
    with a zero conservation residual. Successful steps add a small 1e-3
    to every free pipe so a stepped maze is distinguishable from a held
    one in the maps.
    """

    def __init__(self, x_ref, broken_idx, broken_step, trigger="singular"):
        self.I = 1.0
        self.x_ref = x_ref
        self.keys = [x_ref[b].contiguous().numpy().tobytes()
                     for b in range(x_ref.shape[0])]
        assert len(set(self.keys)) == len(self.keys), "maze fingerprints collide"
        self.broken_key = self.keys[broken_idx]
        self.broken_step = int(broken_step)
        assert trigger in ("singular", "residual")
        self.trigger = trigger
        self.done = {}

    def _key(self, xb, j):
        return xb[j].contiguous().numpy().tobytes()

    def to(self, device):
        return self

    def eval(self):
        return self

    def edge_masks(self, x):
        walk = x[:, 0]
        return walk[:, :, :-1] * walk[:, :, 1:], walk[:, :-1, :] * walk[:, 1:, :]

    def initial_state(self, x):
        Eh, Ev = self.edge_masks(x)
        return Eh.clone(), Ev.clone(), torch.zeros_like(x[:, 0])

    def step(self, x, Dh, Dv, p):
        broken_here = False
        for j in range(x.shape[0]):
            k = self._key(x, j)
            if k == self.broken_key and self.done.get(k, 0) >= self.broken_step - 1:
                broken_here = True
        Eh, Ev = self.edge_masks(x)
        for j in range(x.shape[0]):
            if self._key(x, j) == self.broken_key and broken_here:
                continue        # the broken maze never completes its step
            k = self._key(x, j)
            self.done[k] = self.done.get(k, 0) + 1
        if broken_here and self.trigger == "singular":
            raise torch.linalg.LinAlgError(
                "(Batch element ?): The solver failed because the input "
                "matrix is singular.")
        residual = 2e-4 if (broken_here and self.trigger == "residual") else 0.0
        info = {"residual": residual, "dissipation": 1.0,
                "psi_halves": [], "residual_halves": []}
        return (Dh + 1e-3 * Eh, Dv + 1e-3 * Ev, p,
                Eh.clone(), Ev.clone(), info)

    def read(self, x, Dh, Dv, p, Qh, Qv):
        ah, av = Qh.abs(), Qv.abs()
        qcell = (torch.nn.functional.pad(ah, (0, 1))
                 + torch.nn.functional.pad(ah, (1, 0))
                 + torch.nn.functional.pad(av, (0, 0, 0, 1))
                 + torch.nn.functional.pad(av, (0, 0, 1, 0)))
        logits = torch.where(qcell >= 0.5 * self.I,
                             torch.full_like(qcell, 20.0),
                             torch.full_like(qcell, -20.0))
        return logits.masked_fill(x[:, 0] == 0, -20.0)


def physics_batch(d, T=1):
    """(edge masks, flows, labels) of the physics arm after T exact-solve steps."""
    model = TransportCFNet(physics_only=True).eval()
    x = d.x
    with torch.no_grad():
        Eh, Ev = model.edge_masks(x)
        Dh, Dv, p = model.initial_state(x)
        for _ in range(T):
            Dh, Dv, p, Qh, Qv, _ = model.step(x, Dh, Dv, p)
    return model, Eh, Ev, Qh, Qv, d.y


def test_ensure_clone_passes_in_this_repository():
    assert etp.ensure_clone() == etp.REPO_ROOT.resolve()


def test_ensure_clone_rejects_a_foreign_cf_path():
    foreign = "C:/somewhere/else/cf/models/transport.py"
    try:
        etp.ensure_clone(foreign)
    except SystemExit as exc:
        assert "not this clone" in str(exc)
    else:
        raise AssertionError("ensure_clone must refuse a foreign cf path")


def test_path_edge_masks_match_the_label_corridor():
    d = FakeLevelSet()
    Ph, Pv = etp.path_edge_masks(d.y)
    # both endpoints on the label path, per edge layout
    on = d.y > 0.5
    assert torch.equal(Ph, on[:, :, :-1] & on[:, :, 1:])
    assert torch.equal(Pv, on[:, :-1, :] & on[:, 1:, :])
    # every maze has at least one path edge (start != goal, connected)
    assert (Ph.sum(dim=(1, 2)) + Pv.sum(dim=(1, 2)) > 0).all()


def test_dead_end_fraction_detects_off_path_flow():
    d = FakeLevelSet()
    _, Eh, Ev, _, _, y = physics_batch(d)
    # fabricated flows: every free pipe carries exactly one unit
    Qh = Eh.clone()
    Qv = Ev.clone()
    fracs = etp.dead_end_flow_fractions(Qh, Qv, y, Eh, Ev)
    Ph, Pv = etp.path_edge_masks(y)
    n_off = (Eh.bool() & ~Ph).sum(dim=(1, 2)) + (Ev.bool() & ~Pv).sum(dim=(1, 2))
    n_free = Eh.sum(dim=(1, 2)) + Ev.sum(dim=(1, 2))
    expected = (n_off / n_free).float()
    assert torch.allclose(fracs, expected, atol=1e-6)
    assert (fracs > 0).all() and (fracs < 1).all()   # off-path pipes exist


def test_dead_end_fraction_is_zero_for_the_exact_physics_flow():
    # the spec's guarantee: on a perfect maze the steady flow is exactly zero
    # on every pipe off the solution path - the fraction must be ~0
    d = FakeLevelSet(count=3)
    _, Eh, Ev, Qh, Qv, y = physics_batch(d, T=1)
    fracs = etp.dead_end_flow_fractions(Qh, Qv, y, Eh, Ev)
    assert float(fracs.max()) < 1e-9


def test_evaluate_level_ceiling_accuracy_and_payload_shape():
    d = FakeLevelSet(count=3)
    model = TransportCFNet(physics_only=True)
    out = etp.evaluate_level(model, d, Ts=[1, 2, 4])
    assert out["n_mazes"] == 3
    assert out["wall_time_s"] >= 0.0
    assert set(out["per_T"]) == {"1", "2", "4"}
    for T, s in out["per_T"].items():
        assert s["strict"] == 1.0 and s["exact"] == 1.0
        assert 0.0 <= s["dead_end_flow_fraction"] < 1e-9
        assert s["dead_end_flow_fraction_max"] < 1e-8
        assert s["n_static_flow"] == 0              # no solver breakdown here
        assert s["wall_seconds"] >= 0.0
        assert s["flow_conservation_residual_mean"] < etp.RESIDUAL_TOL
        assert s["dissipation_mean_per_maze"] > 0.0
    assert out["solver_breakdown"]["n_held"] == 0
    assert out["solver_breakdown"]["n_full_rollout"] == 3
    assert out["solver_breakdown"]["first_step"] is None


def test_step_active_isolates_only_the_failing_maze():
    d = FakeLevelSet(count=4)
    model = BreakdownModel(d.x, broken_idx=2, broken_step=1)
    with torch.no_grad():
        Eh, Ev = model.edge_masks(d.x)
        Dh, Dv, p = model.initial_state(d.x)
        Qh_prev, Qv_prev = torch.zeros_like(Dh), torch.zeros_like(Dv)
        Dh2, Dv2, p2, Qh2, Qv2, info, failed = etp._step_active(
            model, d.x, Dh, Dv, p, Qh_prev, Qv_prev)
    assert failed == [2]
    assert info is not None and info["residual"] == 0.0
    assert torch.equal(Dh2[2], Dh[2]) and torch.equal(Qh2[2], Qh_prev[2])
    for k in (0, 1, 3):
        assert torch.allclose(Dh2[k], Dh[k] + 1e-3 * Eh[k])
        assert torch.equal(Qh2[k], Eh[k])
    # the broken maze never completed a step
    assert model.done.get(model.keys[2]) is None


def test_evaluate_level_holds_maze_whose_solve_breaks():
    d = FakeLevelSet(count=4)
    model = BreakdownModel(d.x, broken_idx=1, broken_step=2)
    out = etp.evaluate_level(model, d, Ts=[1, 2, 3], batch_size=4)
    # the run completes; the breakdown is recorded, not fatal
    assert out["n_mazes"] == 4
    bk = out["solver_breakdown"]
    assert bk["n_held"] == 1 and bk["n_full_rollout"] == 3
    assert bk["first_step"] == 2 and bk["last_step"] == 2
    assert bk["held_at_step_histogram"] == {"2": 1}
    assert out["per_T"]["1"]["n_static_flow"] == 0
    assert out["per_T"]["2"]["n_static_flow"] == 1
    assert out["per_T"]["3"]["n_static_flow"] == 1
    # the held maze's flows stay at their last solved unit flows at every
    # later horizon: the breakdown never corrupts the maps
    assert out["per_T"]["2"]["flow_conservation_residual_mean"] < etp.RESIDUAL_TOL


def test_evaluate_level_holds_maze_whose_residual_crosses_tolerance():
    # a solve that is NOT singular but whose conservation residual exceeds
    # the 1e-5 tolerance is equally invalid: the maze must be held, not run
    d = FakeLevelSet(count=4)
    model = BreakdownModel(d.x, broken_idx=3, broken_step=1, trigger="residual")
    out = etp.evaluate_level(model, d, Ts=[1, 2], batch_size=4)
    bk = out["solver_breakdown"]
    assert bk["n_held"] == 1 and bk["n_full_rollout"] == 3
    assert bk["first_step"] == 1
    assert out["per_T"]["1"]["n_static_flow"] == 1
    assert out["per_T"]["2"]["n_static_flow"] == 1


def test_breakdown_model_raises_the_real_solver_exception():
    d = FakeLevelSet(count=2)
    model = BreakdownModel(d.x, broken_idx=0, broken_step=1)
    with torch.no_grad():
        Dh, Dv, p = model.initial_state(d.x)
        try:
            model.step(d.x, Dh, Dv, p)
        except torch.linalg.LinAlgError:
            pass
        else:
            raise AssertionError("the stand-in must raise the solver exception")


def test_shared_rollout_matches_forward_horizons():
    # logits read after T steps of one shared rollout equal model.forward(x, T)
    d = FakeLevelSet(count=2)
    model = TransportCFNet(physics_only=True).eval()
    Ts = [1, 3]
    with torch.no_grad():
        Dh, Dv, p = model.initial_state(d.x)
        Qh = torch.zeros_like(Dh)
        Qv = torch.zeros_like(Dv)
        t = 0
        for T in Ts:
            for _ in range(T - t):
                Dh, Dv, p, Qh, Qv, _ = model.step(d.x, Dh, Dv, p)
                t += 1
            read = model.read(d.x, Dh, Dv, p, Qh, Qv)
            fwd = model(d.x, T=T)
            assert torch.equal(read, fwd)


def test_build_payload_states_rule_and_threshold():
    d = FakeLevelSet(count=2)
    model = TransportCFNet(physics_only=True)
    res = etp.evaluate_level(model, d, Ts=[1])
    payload = etp.build_payload(model, [1], {"level5_test": dict(n_mazes=2)},
                                {"level5_test": res}, 1.5)
    assert payload["model"]["learned_parameters"] == 0
    assert payload["model"]["edge_costs"].startswith("uniform")
    assert ">= 0.5" in payload["readout"]["rule"]
    assert "fixed_in_advance" in payload["readout"]
    assert payload["protocol"]["steps_T"] == [1]
    assert payload["results"]["level5_test"]["per_T"]["1"]["strict"] == 1.0
    assert payload["wall_time"]["total_seconds"] == 1.5


def test_write_outputs_creates_and_rewrites(tmp_path):
    d = FakeLevelSet(count=2)
    model = TransportCFNet(physics_only=True)
    res = etp.evaluate_level(model, d, Ts=[1])
    payload = etp.build_payload(model, [1], {"level5_test": dict(n_mazes=2)},
                                {"level5_test": res}, 0.5)
    summary = etp.render_summary(payload, {"level5_test": res["wall_time_s"]},
                                 {"level5_test": 2})
    rj, sm = etp.write_outputs(tmp_path, payload, summary)
    assert rj.name == "results.json" and sm.name == "summary.md"
    assert json.loads(rj.read_text(encoding="utf-8")) == payload
    text = sm.read_text(encoding="utf-8")
    assert "strict" in text and "dead-end" in text.lower()
    assert "0.5" in text            # the fixed readout threshold is stated
    assert "held" in text.lower()   # the solver-range note is stated
    # a rewrite replaces both files without leftover temp files
    etp.write_outputs(tmp_path, payload, summary + "\nrewritten\n")
    assert json.loads(rj.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("*.tmp"))
    assert "rewritten" in sm.read_text(encoding="utf-8")


def test_parse_args_defaults():
    args = etp.parse_args([])
    assert args.out == "results/transport_physics"
    assert args.steps == [1, 2, 4, 8, 16, 32, 64]
    assert args.batch_size == 100
    args = etp.parse_args(["--steps", "1", "2", "--out", "x"])
    assert args.steps == [1, 2] and args.out == "x"
