"""One controlled query-independent response-gating experiment for PointMaze."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT))

from crl.envs import TwoRouteSwampWindyF4Env
from ett.convex_action_transition import convex_box, energy_score, gates, matrices
from ett.diagonal_transition import POINTMAZE_WALLS, _project_samples, sample_displacement
from ett.pointmaze_region_pilot import Kernel
from ett.rollout_return import DISCOUNT, GOAL, HORIZON, task_reward


SOURCE = ROOT / "outputs" / "supervised_stochastic_ett_20260914_v1"
ATOM_SOURCE = ROOT / "outputs" / "supervised_stochastic_ett_atom_suppression_20260914_v1"
EXTERNAL = Path(r"C:\Users\trhua\Documents\Codex\2026-09-08\f")
TRAIN_DATA = EXTERNAL / "outputs" / "supervised_ett_native_probe_v1" / "paired_transitions.npz"
OLD_COLLECTOR = EXTERNAL / "work" / "supervised-ett-audit" / "pointmaze_supervised_probe.py"
OLD_ENV = EXTERNAL / "work" / "branch-source-031d430" / "crl" / "envs.py"
OLD_TEACHER = EXTERNAL / "work" / "branch-source-f85a5f4" / "scripts" / "collect_swamp_windy.py"
OLD_BLIND = EXTERNAL / "work" / "branch-source-f85a5f4" / "scripts" / "collect_swamp_windy_baddemo.py"
PAIRED_EVALUATION = OUT / "paired_evaluation.npz"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
STATIONARY_THRESHOLD = float(CONFIG["stationary_threshold"])


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(plain(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def full_energy_score(samples, target):
    samples = np.asarray(samples, np.float64)
    target = np.asarray(target, np.float64)
    n, k, _ = samples.shape
    result = np.empty(n, np.float64)
    for start in range(0, n, 128):
        stop = min(n, start + 128)
        value = samples[start:stop]
        first = np.linalg.norm(value - target[start:stop, None], axis=-1).mean(1)
        pairs = np.linalg.norm(value[:, :, None] - value[:, None, :], axis=-1)
        result[start:stop] = first - pairs.sum((1, 2)) / (2 * k * (k - 1))
    return result


def native_legal(xy):
    xy = np.asarray(xy)
    floor = np.floor(xy).astype(np.int64)
    bounded = (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= 9)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= 5)
    )
    cell = POINTMAZE_WALLS[
        np.clip(floor[..., 0], 0, 8), np.clip(floor[..., 1], 0, 4)
    ]
    return bounded & (cell == 0)


class Ledger:
    def __init__(self, resume=False):
        if resume:
            saved = json.loads((OUT / "ledger.json").read_text(encoding="utf-8"))
            self.model_counts = OrderedDict(saved["model"]["counts"])
            self.native_counts = OrderedDict(saved["native"]["counts"])
        else:
            self.model_counts = OrderedDict()
            self.native_counts = OrderedDict()
            self.save()

    @property
    def model_total(self):
        return int(sum(self.model_counts.values()))

    @property
    def native_total(self):
        return int(sum(self.native_counts.values()))

    def model(self, number, purpose):
        number = int(number)
        if self.model_total + number > CONFIG["model_successor_cap"]:
            raise RuntimeError("model-successor cap exhausted")
        self.model_counts[purpose] = self.model_counts.get(purpose, 0) + number
        self.save()

    def native(self, number, purpose):
        number = int(number)
        if self.native_total + number > CONFIG["native_step_cap"]:
            raise RuntimeError("native-step cap exhausted")
        self.native_counts[purpose] = self.native_counts.get(purpose, 0) + number
        self.save()

    def save(self):
        write_json(
            OUT / "ledger.json",
            {
                "model": {
                    "counts": self.model_counts,
                    "total": self.model_total,
                    "cap": CONFIG["model_successor_cap"],
                },
                "native": {
                    "counts": self.native_counts,
                    "total": self.native_total,
                    "cap": CONFIG["native_step_cap"],
                },
            },
        )


class ExperimentEngine:
    """Original sampler plus the one gated emitter; production code is untouched."""

    def __init__(self, state_mean, state_std):
        self.kernel = Kernel()
        self.state_mean = jnp.asarray(state_mean, jnp.float32)
        self.state_std = jnp.asarray(state_std, jnp.float32)
        self.control_sample = jax.jit(self._control_sample, static_argnums=(5,))
        self.candidate_sample = jax.jit(self._candidate_sample, static_argnums=(5,))
        self.control_rollout = self._make_rollout(self.control_sample)
        self.candidate_rollout = self._make_rollout(self.candidate_sample)

    def _control_sample(self, theta, state, action, xb, key, count):
        output, detail = self.kernel._sample(theta, state, action, xb, key, count)
        detail = dict(detail)
        detail["h"] = jnp.ones(detail["stationary_atom"].shape, output.dtype)
        return output, detail

    def _candidate_sample(self, theta, state, action, xb, key, count):
        common, w = theta[:48], theta[48:]
        goal = jnp.broadcast_to(jnp.asarray(GOAL), state.shape)
        context = jnp.concatenate([state, xb, xb, goal], axis=-1)
        distribution = self.kernel.base._distribution(
            self.kernel.parameters(common), context
        )
        delta, atom = sample_displacement(
            distribution,
            key,
            count,
            self.kernel.base.delta_mean,
            self.kernel.base.delta_std,
            self.kernel.base.spec,
        )
        anchor_state, base_detail = _project_samples(
            state, delta, self.kernel.base.spec
        )
        anchor = anchor_state[..., :2]
        low, high, valid, rectangle = convex_box(state, anchor)
        gate_weight = gates(state)
        response_matrix = jnp.einsum(
            "bj,jkl->bkl", gate_weight, matrices(common[16:], 1.0)
        )
        response = jnp.einsum("bij,bj->bi", response_matrix, action - xb)
        batch, draws = atom.shape
        broadcast = lambda value: jnp.broadcast_to(
            value[:, None], (batch, draws) + value.shape[1:]
        )
        features = jnp.concatenate(
            [
                broadcast((state - self.state_mean) / self.state_std),
                broadcast(xb),
                anchor - state[:, None, :2],
                atom[..., None].astype(state.dtype),
                broadcast(gate_weight),
            ],
            axis=-1,
        )
        if features.shape[-1] != CONFIG["gate_dimension"]:
            raise ValueError("gate feature width changed")
        gate_linear = jnp.einsum("bki,i->bk", features, w)
        h = jnp.clip(1.0 + gate_linear, 0.0, 1.0)
        proposal = anchor + h[..., None] * response[:, None, :]
        xy = jnp.clip(proposal, low, high)
        xy = jnp.where(valid[..., None], xy, jnp.nan)
        old = jnp.broadcast_to(state[:, None, :6], xy.shape[:-1] + (6,))
        output = jnp.concatenate([xy, old], axis=-1)
        return output, {
            "anchor_xy": anchor,
            "box_low": low,
            "box_high": high,
            "box_valid": valid,
            "rectangle": rectangle,
            "response": response,
            "h": h,
            "gate_linear": gate_linear,
            "stationary_atom": atom,
            "projection_corrected": jnp.any(xy != proposal, axis=-1),
            "projected_to_boundary": jnp.any((xy == low) | (xy == high), axis=-1),
            "emitted_change": jnp.linalg.norm(xy - anchor, axis=-1),
            "base_corrected": jnp.any(
                base_detail["raw_position"] != anchor, axis=-1
            ),
        }

    def _make_rollout(self, sample_function):
        nominal, actor = self.kernel.nominal, self.kernel.actor

        def generate(theta, states, goals, key):
            def step(state, step_key):
                nominal_key, actor_key, transition_key = jax.random.split(step_key, 3)
                xb = nominal.sample(state, nominal_key, 1, goal=goals)
                action = actor(state, goals, actor_key)
                output, detail = sample_function(
                    theta, state, action, xb, transition_key, 1
                )
                following = output[:, 0]
                record = {
                    "next_state": following,
                    "action": action,
                    "xb": xb,
                    "reward": task_reward(following, goals),
                    "atom": detail["stationary_atom"][:, 0],
                    "h": detail["h"][:, 0],
                    "projection": detail["projection_corrected"][:, 0],
                }
                return following, record

            _, record = jax.lax.scan(
                step, states, jax.random.split(key, CONFIG["return_horizon"])
            )
            record = {name: jnp.swapaxes(value, 0, 1) for name, value in record.items()}
            record["states"] = jnp.concatenate(
                [states[:, None], record.pop("next_state")], axis=1
            )
            record["return"] = jnp.sum(
                record["reward"]
                * jnp.power(CONFIG["return_discount"], jnp.arange(CONFIG["return_horizon"])),
                axis=-1,
            )
            return record

        return jax.jit(generate)


def dependency_provenance():
    original = json.loads((SOURCE / "provenance.json").read_text(encoding="utf-8"))
    atom = json.loads((ATOM_SOURCE / "provenance.json").read_text(encoding="utf-8"))
    paired = json.loads(
        (TRAIN_DATA.parent / "provenance.json").read_text(encoding="utf-8")
    )
    required = [
        OUT / "PROTOCOL.md",
        OUT / "config.json",
        OUT / "collect_evaluation.py",
        Path(__file__),
        SOURCE / "REPORT.md",
        SOURCE / "run.py",
        SOURCE / "B_s0.npz",
        SOURCE / "B_s1.npz",
        ATOM_SOURCE / "REPORT.md",
        ATOM_SOURCE / "results.json",
        TRAIN_DATA,
        OLD_COLLECTOR,
        OLD_ENV,
        OLD_TEACHER,
        OLD_BLIND,
        ROOT / "ett" / "pointmaze_region_pilot.py",
        ROOT / "ett" / "convex_action_transition.py",
        ROOT / "ett" / "diagonal_transition.py",
        ROOT / "ett" / "rollout_return.py",
        ROOT / "crl" / "envs.py",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {str(path): sha256(path) for path in required}
    for path in [
        TRAIN_DATA,
        OLD_COLLECTOR,
        ROOT / "ett" / "pointmaze_region_pilot.py",
        ROOT / "ett" / "convex_action_transition.py",
        ROOT / "ett" / "diagonal_transition.py",
        ROOT / "crl" / "envs.py",
    ]:
        if original["sources"][str(path)] != hashes[str(path)]:
            raise RuntimeError(f"completed-experiment dependency changed: {path}")
    for path in [SOURCE / "B_s0.npz", SOURCE / "B_s1.npz", TRAIN_DATA]:
        if atom["input_sha256"][str(path)] != hashes[str(path)]:
            raise RuntimeError(f"atom-diagnostic dependency changed: {path}")
    external_expected = {
        OLD_COLLECTOR: paired["source_files"]["work\\supervised-ett-audit\\pointmaze_supervised_probe.py"],
        OLD_ENV: paired["source_files"]["work\\branch-source-031d430\\crl\\envs.py"],
        OLD_TEACHER: paired["source_files"]["work\\branch-source-f85a5f4\\scripts\\collect_swamp_windy.py"],
        OLD_BLIND: paired["source_files"]["work\\branch-source-f85a5f4\\scripts\\collect_swamp_windy_baddemo.py"],
    }
    for path, expected in external_expected.items():
        if hashes[str(path)] != expected:
            raise RuntimeError(f"original paired-collector dependency changed: {path}")
    rollout_config = json.loads(
        (ROOT / "artifacts" / "ett_rollout_return" / "residual6_s01" / "config.json").read_text(encoding="utf-8")
    )
    expected_rollout = rollout_config["source_file_sha256"]["ett/rollout_return.py"]
    if hashes[str(ROOT / "ett" / "rollout_return.py")] != expected_rollout:
        raise RuntimeError("task-return implementation changed")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "working_head_recorded_not_required": head,
        "historical_parent_commit": CONFIG["parent_commit"],
        "dependency_sha256": hashes,
        "completed_experiment_provenance_sha256": sha256(SOURCE / "provenance.json"),
        "atom_diagnostic_provenance_sha256": sha256(ATOM_SOURCE / "provenance.json"),
        "hash_contract_passed": True,
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_devices": list(map(str, jax.devices())),
    }


def load_training_data(save_scaling=True):
    with np.load(TRAIN_DATA, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    assert data["s"].shape == (19200, 8)
    train = data["episode"] < 72
    diagonal = np.all(data["xq"] == data["xb"], axis=1)
    diagonal_ids = np.flatnonzero(train & diagonal)
    off_ids = np.flatnonzero(train & ~diagonal)
    assert len(diagonal_ids) == 3860 and len(off_ids) == 10540
    state_mean = data["s"][train].mean(axis=0).astype(np.float32)
    state_std = np.maximum(
        data["s"][train].std(axis=0), CONFIG["state_scale_floor"]
    ).astype(np.float32)
    if save_scaling:
        np.savez_compressed(
            OUT / "gate_feature_scaling.npz",
            state_mean=state_mean,
            state_std=state_std,
            train_episodes=np.arange(72),
        )
        write_json(
            OUT / "gate_features.json",
            {
                "dimension": 21,
                "order": CONFIG["gate_features"],
                "state_mean": state_mean,
                "state_std": state_std,
                "estimated_from": "all original rows in episodes 0-71 only",
                "xq_included": False,
                "labels_or_outcomes_included": False,
            },
        )
    else:
        with np.load(OUT / "gate_feature_scaling.npz", allow_pickle=False) as scaling:
            np.testing.assert_array_equal(scaling["state_mean"], state_mean)
            np.testing.assert_array_equal(scaling["state_std"], state_std)
    return data, diagonal_ids, off_ids, state_mean, state_std


def parent_checkpoint(seed):
    with np.load(SOURCE / f"B_s{seed}.npz", allow_pickle=False) as loaded:
        checkpoint = {name: loaded[name] for name in loaded.files}
    assert checkpoint["theta"].shape == (48,)
    assert checkpoint["adam_m"].shape == (48,)
    assert checkpoint["adam_v"].shape == (48,)
    assert checkpoint["trajectory"].shape == (120, 48)
    np.testing.assert_array_equal(checkpoint["trajectory"][-1], checkpoint["theta"])
    return checkpoint


def make_training_functions(engine):
    def control_loss(theta, state, xb, xq, target, key):
        output, _ = engine.control_sample(theta, state, xq, xb, key, CONFIG["training_samples"])
        return energy_score(output[..., :2], target[:, :2]).mean()

    def candidate_loss(theta, state, xb, xq, target, key):
        output, detail = engine.candidate_sample(
            theta, state, xq, xb, key, CONFIG["training_samples"]
        )
        h = detail["h"]
        return (
            energy_score(output[..., :2], target[:, :2]).mean(),
            h.mean(),
            (h < 1.0).mean(),
            (h <= 0.0).mean(),
        )

    return (
        jax.jit(jax.vmap(control_loss, in_axes=(0, None, None, None, None, None))),
        jax.jit(jax.vmap(candidate_loss, in_axes=(0, None, None, None, None, None))),
    )


def train_arm(engine, data, diagonal_ids, off_ids, seed, arm, ledger, functions):
    parent = parent_checkpoint(seed)
    if arm == "control":
        theta = parent["theta"].copy()
        adam_m = parent["adam_m"].astype(np.float64).copy()
        adam_v = parent["adam_v"].astype(np.float64).copy()
        dimension = 48
    else:
        theta = np.concatenate(
            [parent["theta"], np.zeros(CONFIG["gate_dimension"], np.float32)]
        )
        adam_m = np.concatenate(
            [parent["adam_m"].astype(np.float64), np.zeros(CONFIG["gate_dimension"])]
        )
        adam_v = np.concatenate(
            [parent["adam_v"].astype(np.float64), np.zeros(CONFIG["gate_dimension"])]
        )
        dimension = 48 + CONFIG["gate_dimension"]
    sigma = np.concatenate(
        [
            np.full(16, CONFIG["diagonal_sigma"]),
            np.full(32, CONFIG["response_sigma"]),
            np.full(dimension - 48, CONFIG["gate_sigma"]),
        ]
    )
    rate = np.concatenate(
        [
            np.full(16, CONFIG["diagonal_learning_rate"]),
            np.full(32, CONFIG["response_learning_rate"]),
            np.full(dimension - 48, CONFIG["gate_learning_rate"]),
        ]
    )
    history, trajectory = [], []
    control_batch, candidate_batch = functions
    for step in range(CONFIG["additional_updates"]):
        rng = np.random.default_rng(
            CONFIG["training_direction_seed"] + seed * 100000 + step * 10
        )
        ids_diagonal = rng.choice(diagonal_ids, CONFIG["batch_per_term"])
        ids_off = rng.choice(off_ids, CONFIG["batch_per_term"])
        common_directions = rng.normal(size=(CONFIG["directions"], 48))
        if arm == "candidate":
            gate_directions = rng.normal(
                size=(CONFIG["directions"], CONFIG["gate_dimension"])
            )
            directions = np.concatenate([common_directions, gate_directions], axis=1)
        else:
            directions = common_directions
        trials = np.stack(
            [theta + sigma * directions, theta - sigma * directions], axis=1
        ).reshape(CONFIG["signed_candidates"], dimension).astype(np.float32)
        term_values = []
        term_h = []
        for term, indices in enumerate([ids_diagonal, ids_off]):
            ledger.model(
                CONFIG["signed_candidates"]
                * CONFIG["batch_per_term"]
                * CONFIG["training_samples"],
                f"training_{arm}_s{seed}",
            )
            key = jax.random.PRNGKey(
                CONFIG["training_model_seed"] + seed * 100000 + step * 10 + term
            )
            args = [
                jnp.asarray(data[name][indices])
                for name in ["s", "xb", "xq", "y"]
            ]
            if arm == "candidate":
                values, h_mean, h_below, h_zero = candidate_batch(
                    jnp.asarray(trials), *args, key
                )
                term_h.append(
                    {
                        "mean": float(np.asarray(h_mean).mean()),
                        "below_one_fraction": float(np.asarray(h_below).mean()),
                        "zero_fraction": float(np.asarray(h_zero).mean()),
                    }
                )
            else:
                values = control_batch(jnp.asarray(trials), *args, key)
                term_h.append(
                    {"mean": 1.0, "below_one_fraction": 0.0, "zero_fraction": 0.0}
                )
            values = np.asarray(values).reshape(CONFIG["directions"], 2)
            if not np.isfinite(values).all():
                raise FloatingPointError("non-finite signed training loss")
            term_values.append(values)
        components = np.stack(
            [
                np.mean(
                    (values[:, 0] - values[:, 1])[:, None] * directions,
                    axis=0,
                )
                / (2 * sigma)
                for values in term_values
            ]
        )
        components[0, 16:] = 0.0
        gradient = components.sum(axis=0)
        global_step = 120 + step + 1
        adam_m = CONFIG["adam_beta1"] * adam_m + (1 - CONFIG["adam_beta1"]) * gradient
        adam_v = CONFIG["adam_beta2"] * adam_v + (1 - CONFIG["adam_beta2"]) * gradient**2
        update = rate * (
            adam_m / (1 - CONFIG["adam_beta1"] ** global_step)
        ) / (
            np.sqrt(adam_v / (1 - CONFIG["adam_beta2"] ** global_step))
            + CONFIG["adam_epsilon"]
        )
        groups = [
            (slice(0, 16), CONFIG["diagonal_update_cap"]),
            (slice(16, 48), CONFIG["response_update_cap"]),
        ]
        if arm == "candidate":
            groups.append((slice(48, dimension), CONFIG["gate_update_cap"]))
        for section, cap in groups:
            norm = np.linalg.norm(update[section])
            update[section] *= min(1.0, cap / max(norm, 1e-12))
        theta = (theta - update).astype(np.float32)
        if not np.isfinite(theta).all():
            raise FloatingPointError("non-finite trained parameters")
        trajectory.append(theta.copy())
        history.append(
            {
                "additional_step": step + 1,
                "global_adam_step": global_step,
                "signed_diagonal_losses": term_values[0],
                "signed_off_losses": term_values[1],
                "diagonal_gradient_norm": float(np.linalg.norm(components[0])),
                "off_gradient_norm": float(np.linalg.norm(components[1])),
                "head_update_norm": float(np.linalg.norm(update[:16])),
                "response_update_norm": float(np.linalg.norm(update[16:48])),
                "gate_gradient_norm": (
                    float(np.linalg.norm(components[1, 48:]))
                    if arm == "candidate"
                    else 0.0
                ),
                "gate_update_norm": (
                    float(np.linalg.norm(update[48:])) if arm == "candidate" else 0.0
                ),
                "gate_weight_norm": (
                    float(np.linalg.norm(theta[48:])) if arm == "candidate" else 0.0
                ),
                "signed_gate_diagnostics": term_h,
            }
        )
        if (step + 1) % 30 == 0:
            print(
                f"trained {arm} seed {seed}: {step + 1}/120, "
                f"model outputs {ledger.model_total:,}",
                flush=True,
            )
    trajectory = np.asarray(trajectory)
    np.savez_compressed(
        OUT / f"{arm}_s{seed}.npz",
        theta=theta,
        adam_m=adam_m,
        adam_v=adam_v,
        trajectory=trajectory,
        parent_theta=parent["theta"],
        inherited_adam_step=120,
    )
    write_json(OUT / f"{arm}_s{seed}_training.json", history)
    if arm == "candidate":
        effective = {
            "nonzero_gate_gradient_updates": int(
                sum(row["gate_gradient_norm"] > 0 for row in history)
            ),
            "nonzero_gate_parameter_updates": int(
                sum(row["gate_update_norm"] > 0 for row in history)
            ),
            "final_gate_weight_norm": float(np.linalg.norm(theta[48:])),
            "signed_off_trials_below_one_fraction_mean": float(
                np.mean(
                    [row["signed_gate_diagnostics"][1]["below_one_fraction"] for row in history]
                )
            ),
        }
        if (
            effective["nonzero_gate_parameter_updates"] == 0
            or effective["final_gate_weight_norm"] == 0
        ):
            raise RuntimeError("gate received no effective updates")
        write_json(OUT / f"candidate_s{seed}_gate_effectiveness.json", effective)
    return theta


def initialization_checks(engine, data, ledger):
    rows = np.flatnonzero(data["episode"] < 72)[:128]
    result = {}
    for seed in [0, 1]:
        parent = parent_checkpoint(seed)["theta"]
        candidate = np.concatenate(
            [parent, np.zeros(CONFIG["gate_dimension"], np.float32)]
        )
        key = jax.random.PRNGKey(CONFIG["check_seed"] + seed)
        args = [jnp.asarray(data[name][rows]) for name in ["s", "xq", "xb"]]
        ledger.model(128 * 8, "checks_initialization")
        original, original_detail = engine.control_sample(parent, *args, key, 8)
        ledger.model(128 * 8, "checks_initialization")
        gated, gated_detail = engine.candidate_sample(candidate, *args, key, 8)
        original, gated = np.asarray(original), np.asarray(gated)
        np.testing.assert_array_equal(original, gated)
        np.testing.assert_array_equal(
            np.asarray(original_detail["stationary_atom"]),
            np.asarray(gated_detail["stationary_atom"]),
        )
        np.testing.assert_array_equal(np.asarray(gated_detail["h"]), np.ones((128, 8)))
        result[f"s{seed}"] = {
            "exact_output_equivalence": True,
            "exact_atom_equivalence": True,
            "gate_exactly_one": True,
        }
    write_json(OUT / "initialization_checks.json", result)


def load_theta(name):
    if name.startswith("parent"):
        return parent_checkpoint(int(name[-1]))["theta"]
    with np.load(OUT / f"{name}.npz", allow_pickle=False) as loaded:
        return loaded["theta"]


def sample_function(engine, name):
    return engine.candidate_sample if name.startswith("candidate") else engine.control_sample


def collect_paired_evaluation(ledger):
    ledger.native(
        CONFIG["planned_native_steps"]["paired_branch_and_prefix"],
        "paired_branch_and_prefix",
    )
    command = [
        sys.executable,
        str(OUT / "collect_evaluation.py"),
        "--out",
        str(PAIRED_EVALUATION),
        "--episodes",
        str(CONFIG["paired_evaluation_episodes"]),
        "--seed",
        str(CONFIG["paired_collection_seed"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    manifest = json.loads(
        (OUT / "paired_evaluation_collection.json").read_text(encoding="utf-8")
    )
    assert manifest["native_steps"] == CONFIG["planned_native_steps"]["paired_branch_and_prefix"]


def evaluation_groups(data):
    state, target = data["s"], data["y"][:, :2]
    diagonal = np.all(data["xq"] == data["xb"], axis=1)
    actual_stationary = (
        np.linalg.norm(target - state[:, :2], axis=1) <= STATIONARY_THRESHOLD
    )
    alive = ~data["dead_before"]
    q0_stationary = np.empty(len(state), bool)
    for offset in range(0, len(state), 4):
        block = np.arange(offset, offset + 4)
        assert len(set(zip(data["episode"][block], data["time"][block]))) == 1
        assert set(data["query"][block]) == {0, 1, 2, 3}
        assert np.all(state[block] == state[block[0]])
        assert np.all(data["xb"][block] == data["xb"][block[0]])
        q0 = block[data["query"][block] == 0][0]
        assert np.array_equal(data["xq"][q0], data["xb"][q0])
        q0_stationary[block] = actual_stationary[q0]
    groups = OrderedDict(
        [
            ("diagonal", diagonal),
            ("all_off_diagonal", ~diagonal),
            ("already_dead", (~diagonal) & (~alive)),
            (
                "alive_outside_goal",
                (~diagonal)
                & alive
                & (np.linalg.norm(state[:, :2] - GOAL[:2], axis=1) >= 2),
            ),
            ("alive_moving", (~diagonal) & alive & (~actual_stationary)),
            ("fatal_onset", (~diagonal) & alive & data["dead_after"]),
            ("alive_stationary", (~diagonal) & alive & actual_stationary),
            (
                "alive_stationary_diagonal_moving_off_diagonal",
                (~diagonal) & alive & q0_stationary & (~actual_stationary),
            ),
        ]
    )
    return groups, actual_stationary, q0_stationary


def summarize_evaluation(saved, data, mask):
    if not mask.any():
        return {"rows": 0, "episodes": 0, "available": False}
    atom = saved["atom"][mask]
    h = saved["h"][mask]
    result = {
        "rows": int(mask.sum()),
        "episodes": int(len(np.unique(data["episode"][mask]))),
        "energy_score": float(saved["es"][mask].mean()),
        "mean_xy_rmse": float(np.sqrt(saved["squared_error"][mask].mean())),
        "mean_xy_euclidean_error": float(saved["euclidean_error"][mask].mean()),
        "sample_stationary_fraction": float(saved["stationary"][mask].mean()),
        "sample_moving_fraction": float((~saved["stationary"][mask]).mean()),
        "mean_sample_displacement": float(saved["displacement"][mask].mean()),
        "atom_fraction": float(atom.mean()),
        "illegal_sample_fraction": float((~saved["legal"][mask]).mean()),
        "illegal_mean_fraction": float((~saved["mean_legal"][mask]).mean()),
        "gate": {
            "mean": float(h.mean()),
            "quantiles": np.quantile(h, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]),
            "below_one_fraction": float((h < 1).mean()),
            "zero_fraction": float((h <= 0).mean()),
            "atom_mean": float(h[atom].mean()) if atom.any() else None,
            "nonatom_mean": float(h[~atom].mean()) if (~atom).any() else None,
        },
    }
    return result


def bootstrap_difference(left, right, episode, mask, weights, transform=None):
    if not mask.any():
        return {"available": False, "reason": "zero rows"}
    episodes = np.arange(CONFIG["paired_evaluation_episodes"])
    count = np.array([np.sum(mask & (episode == item)) for item in episodes])
    a = np.array([left[mask & (episode == item)].sum() for item in episodes])
    b = np.array([right[mask & (episode == item)].sum() for item in episodes])
    denominator = weights @ count
    valid = denominator > 0
    av = (weights[valid] @ a) / denominator[valid]
    bv = (weights[valid] @ b) / denominator[valid]
    point_a, point_b = left[mask].mean(), right[mask].mean()
    if transform is not None:
        av, bv = transform(av), transform(bv)
        point_a, point_b = transform(point_a), transform(point_b)
    return {
        "direction": "left_minus_right",
        "difference": float(point_a - point_b),
        "ci95": np.quantile(av - bv, [0.025, 0.975]),
        "contributing_episodes": int(np.sum(count > 0)),
        "empty_replicates": int(np.sum(~valid)),
    }


def evaluate_one_step(engine, ledger, reuse_saved=False):
    with np.load(PAIRED_EVALUATION, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    assert data["s"].shape == (9600, 8)
    groups, actual_stationary, q0_stationary = evaluation_groups(data)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    weights = rng.multinomial(
        CONFIG["paired_evaluation_episodes"],
        np.full(CONFIG["paired_evaluation_episodes"], 1 / CONFIG["paired_evaluation_episodes"]),
        size=CONFIG["bootstrap_replicates"],
    )
    names = [
        "parent_s0", "control_s0", "candidate_s0",
        "parent_s1", "control_s1", "candidate_s1",
    ]
    arrays, metrics = {}, {}
    for name in names:
        saved_path = OUT / f"{name}_evaluation.npz"
        if reuse_saved:
            with np.load(saved_path, allow_pickle=False) as loaded:
                saved = {key: loaded[key] for key in loaded.files}
            samples = saved["samples_xy"]
            np.testing.assert_allclose(
                saved["es"], full_energy_score(samples, data["y"][:, :2]), atol=2e-12
            )
        else:
            theta = load_theta(name)
            function = sample_function(engine, name)
            outputs = {key: [] for key in ["samples_xy", "atom", "h", "projection"]}
            for start in range(0, len(data["s"]), 128):
                stop = min(len(data["s"]), start + 128)
                ledger.model((stop - start) * CONFIG["evaluation_samples"], "one_step_evaluation")
                key = jax.random.PRNGKey(CONFIG["one_step_model_seed"] + start)
                output, detail = function(
                    jnp.asarray(theta),
                    jnp.asarray(data["s"][start:stop]),
                    jnp.asarray(data["xq"][start:stop]),
                    jnp.asarray(data["xb"][start:stop]),
                    key,
                    CONFIG["evaluation_samples"],
                )
                output = np.asarray(output)
                np.testing.assert_array_equal(
                    output[..., 2:],
                    np.broadcast_to(data["s"][start:stop, None, :6], output[..., 2:].shape),
                )
                outputs["samples_xy"].append(output[..., :2])
                outputs["atom"].append(np.asarray(detail["stationary_atom"]))
                outputs["h"].append(np.asarray(detail["h"]))
                outputs["projection"].append(np.asarray(detail["projection_corrected"]))
            saved = {key: np.concatenate(value) for key, value in outputs.items()}
            samples = saved["samples_xy"]
            mean = samples.mean(axis=1)
            saved.update(
                es=full_energy_score(samples, data["y"][:, :2]),
                mean=mean,
                squared_error=np.sum((mean - data["y"][:, :2]) ** 2, axis=1),
                euclidean_error=np.linalg.norm(mean - data["y"][:, :2], axis=1),
                stationary=np.linalg.norm(samples - data["s"][:, None, :2], axis=-1)
                <= STATIONARY_THRESHOLD,
                displacement=np.linalg.norm(samples - data["s"][:, None, :2], axis=-1),
                legal=native_legal(samples),
                mean_legal=native_legal(mean),
            )
            if not saved["legal"].all():
                raise RuntimeError(f"illegal emitted sample in {name}")
            np.savez_compressed(saved_path, **saved)
        arrays[name] = saved
        metrics[name] = {
            label: summarize_evaluation(saved, data, mask)
            for label, mask in groups.items()
        }
        print(f"evaluated one-step {name}", flush=True)
    contrasts = {}
    row_fields = {
        "energy_score": ("es", None),
        "mean_xy_rmse": ("squared_error", np.sqrt),
        "mean_xy_euclidean_error": ("euclidean_error", None),
        "sample_stationary_fraction": ("stationary", None),
        "mean_sample_displacement": ("displacement", None),
        "illegal_sample_fraction": ("legal", None),
        "illegal_mean_fraction": ("mean_legal", None),
    }
    for seed in [0, 1]:
        contrasts[str(seed)] = {}
        for comparison, left_name, right_name in [
            ("candidate_minus_control", f"candidate_s{seed}", f"control_s{seed}"),
            ("control_minus_parent", f"control_s{seed}", f"parent_s{seed}"),
        ]:
            contrasts[str(seed)][comparison] = {}
            for label, mask in groups.items():
                contrasts[str(seed)][comparison][label] = {}
                for metric, (field, transform) in row_fields.items():
                    left = arrays[left_name][field]
                    right = arrays[right_name][field]
                    if field == "stationary":
                        left, right = left.mean(1), right.mean(1)
                    elif field == "displacement":
                        left, right = left.mean(1), right.mean(1)
                    elif field == "legal":
                        left, right = (~left).mean(1), (~right).mean(1)
                    elif field == "mean_legal":
                        left, right = (~left).astype(float), (~right).astype(float)
                    contrasts[str(seed)][comparison][label][metric] = bootstrap_difference(
                        left,
                        right,
                        data["episode"],
                        mask,
                        weights,
                        transform,
                    )
    write_json(OUT / "one_step_metrics.json", metrics)
    write_json(OUT / "one_step_contrasts.json", contrasts)
    np.savez_compressed(
        OUT / "one_step_groups.npz",
        episode=data["episode"],
        actual_stationary=actual_stationary,
        q0_stationary=q0_stationary,
        **groups,
    )
    return data, groups, arrays, metrics, contrasts


def invariant_checks(engine, data, ledger):
    diagonal = np.all(data["xq"] == data["xb"], axis=1)
    diagonal_rows = np.flatnonzero(diagonal)[:64]
    rows = np.arange(64)
    rng = np.random.default_rng(CONFIG["check_seed"])
    action_left = rng.uniform(-1, 1, (64, 2)).astype(np.float32)
    action_right = rng.uniform(-1, 1, (64, 2)).astype(np.float32)
    result = {}
    for number, name in enumerate(
        [
            "parent_s0", "control_s0", "candidate_s0",
            "parent_s1", "control_s1", "candidate_s1",
        ]
    ):
        theta = load_theta(name)
        function = sample_function(engine, name)
        key = jax.random.PRNGKey(CONFIG["check_seed"] + 100 + number)
        ledger.model(64 * 8, "checks_invariants")
        diagonal_output, diagonal_detail = function(
            jnp.asarray(theta),
            jnp.asarray(data["s"][diagonal_rows]),
            jnp.asarray(data["xq"][diagonal_rows]),
            jnp.asarray(data["xb"][diagonal_rows]),
            key,
            8,
        )
        diagonal_output = np.asarray(diagonal_output)
        np.testing.assert_array_equal(
            diagonal_output[..., :2], np.asarray(diagonal_detail["anchor_xy"])
        )
        np.testing.assert_array_equal(
            diagonal_output[..., 2:],
            np.broadcast_to(
                data["s"][diagonal_rows, None, :6], diagonal_output[..., 2:].shape
            ),
        )
        ledger.model(64 * 8, "checks_invariants")
        left, left_detail = function(
            jnp.asarray(theta),
            jnp.asarray(data["s"][rows]),
            jnp.asarray(action_left),
            jnp.asarray(data["xb"][rows]),
            key,
            8,
        )
        ledger.model(64 * 8, "checks_invariants")
        right, right_detail = function(
            jnp.asarray(theta),
            jnp.asarray(data["s"][rows]),
            jnp.asarray(action_right),
            jnp.asarray(data["xb"][rows]),
            key,
            8,
        )
        left, right = np.asarray(left), np.asarray(right)
        np.testing.assert_array_equal(
            np.asarray(left_detail["anchor_xy"]), np.asarray(right_detail["anchor_xy"])
        )
        np.testing.assert_array_equal(
            np.asarray(left_detail["stationary_atom"]),
            np.asarray(right_detail["stationary_atom"]),
        )
        np.testing.assert_array_equal(
            np.asarray(left_detail["h"]), np.asarray(right_detail["h"])
        )
        denominator = np.linalg.norm(action_left - action_right, axis=1)[:, None]
        ratio = np.linalg.norm(left[..., :2] - right[..., :2], axis=-1) / denominator
        max_ratio = float(ratio.max())
        if max_ratio > 1 + 2e-6:
            raise RuntimeError(f"coupled action bound failed for {name}: {max_ratio}")
        if not native_legal(left[..., :2]).all() or not native_legal(right[..., :2]).all():
            raise RuntimeError("invariant check emitted illegal sample")
        h = np.asarray(left_detail["h"])
        if np.any((h < 0) | (h > 1)):
            raise RuntimeError("gate bound failed")
        result[name] = {
            "diagonal_identity": True,
            "f4_shift": True,
            "coupled_anchor_and_atom": True,
            "query_independent_h_under_coupling": True,
            "coupled_action_ratio_max": max_ratio,
            "all_samples_native_legal": True,
            "h_min": float(h.min()),
            "h_max": float(h.max()),
        }
    write_json(OUT / "invariant_checks.json", result)


def collect_native_actor(engine, ledger):
    episodes = CONFIG["return_native_episodes"]
    states, actions, rewards = [], [], []
    ledger.native(
        CONFIG["planned_native_steps"]["fixed_actor_returns"],
        "fixed_actor_returns",
    )
    for episode in range(episodes):
        env = TwoRouteSwampWindyF4Env(
            seed=CONFIG["return_environment_seed"] + episode,
            active_prob=CONFIG["paired_active_probability"],
        )
        observation = env.reset()
        episode_states = [observation[:8].copy()]
        episode_actions, episode_rewards = [], []
        for step in range(CONFIG["return_horizon"]):
            key = jax.random.PRNGKey(
                CONFIG["return_actor_seed"] + episode * CONFIG["return_horizon"] + step
            )
            action = np.asarray(
                engine.kernel.actor(
                    jnp.asarray(observation[None, :8]),
                    jnp.asarray(observation[None, 8:]),
                    key,
                )
            )[0]
            following, reward, done, _ = env.step(action)
            if done:
                raise RuntimeError("fixed-length native actor episode terminated")
            reconstructed = float(task_reward(following[:8], following[8:]))
            if reward != reconstructed:
                raise RuntimeError("native reward disagrees with fixed visible reward")
            episode_states.append(following[:8].copy())
            episode_actions.append(action)
            episode_rewards.append(reward)
            observation = following
        states.append(episode_states)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
    states = np.asarray(states)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)
    returns = np.sum(
        rewards * CONFIG["return_discount"] ** np.arange(CONFIG["return_horizon"]), axis=1
    )
    np.savez_compressed(
        OUT / "native_actor_episodes.npz",
        states=states,
        actions=actions,
        rewards=rewards,
        returns=returns,
        environment_seed=np.arange(episodes) + CONFIG["return_environment_seed"],
    )
    return {"states": states, "actions": actions, "rewards": rewards, "returns": returns}


def scalar_episode_bootstrap(value, weights, transform=None):
    value = np.asarray(value, np.float64)
    estimate = value.mean()
    replicates = weights @ value / weights.sum(axis=1)
    if transform is not None:
        estimate, replicates = transform(estimate), transform(replicates)
    return {"estimate": float(estimate), "ci95": np.quantile(replicates, [0.025, 0.975])}


def paired_episode_difference(left, right, weights, transform=None):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    point_left, point_right = left.mean(), right.mean()
    l = weights @ left / weights.sum(axis=1)
    r = weights @ right / weights.sum(axis=1)
    if transform is not None:
        point_left, point_right = transform(point_left), transform(point_right)
        l, r = transform(l), transform(r)
    return {
        "direction": "left_minus_right",
        "difference": float(point_left - point_right),
        "ci95": np.quantile(l - r, [0.025, 0.975]),
    }


def evaluate_returns(engine, native, ledger):
    episodes = CONFIG["return_native_episodes"]
    repeats = CONFIG["return_paths_per_initial_state"]
    roots = np.repeat(native["states"][:, 0], repeats, axis=0)
    goals = np.broadcast_to(GOAL, roots.shape)
    names = [
        "parent_s0", "control_s0", "candidate_s0",
        "parent_s1", "control_s1", "candidate_s1",
    ]
    records = {}
    for name in names:
        saved_path = OUT / f"{name}_return_rollouts.npz"
        if saved_path.is_file():
            with np.load(saved_path, allow_pickle=False) as loaded:
                record = {key: loaded[key] for key in loaded.files}
        else:
            theta = load_theta(name)
            function = engine.candidate_rollout if name.startswith("candidate") else engine.control_rollout
            ledger.model(
                episodes * repeats * CONFIG["return_horizon"], "return_rollouts"
            )
            record = function(
                jnp.asarray(theta),
                jnp.asarray(roots),
                jnp.asarray(goals),
                jax.random.PRNGKey(CONFIG["return_model_seed"]),
            )
            record = {
                key: np.asarray(value).reshape((episodes, repeats) + value.shape[1:])
                for key, value in record.items()
            }
        np.testing.assert_array_equal(
            record["states"][:, :, 1:, 2:], record["states"][:, :, :-1, :6]
        )
        if not native_legal(record["states"][..., 1:, :2]).all():
            raise RuntimeError(f"illegal return-rollout sample for {name}")
        recomputed_reward = np.asarray(
            task_reward(
                record["states"][:, :, 1:],
                np.broadcast_to(GOAL, record["states"][:, :, 1:].shape),
            )
        )
        np.testing.assert_array_equal(record["reward"], recomputed_reward)
        discount32 = np.asarray(
            jnp.power(
                jnp.float32(CONFIG["return_discount"]),
                jnp.arange(CONFIG["return_horizon"]),
            )
        )
        recomputed_return = np.sum(
            record["reward"].astype(np.float32) * discount32,
            axis=-1,
            dtype=np.float32,
        )
        # JAX/XLA and NumPy can use different float32 reduction trees.  The
        # reward bits and discount vector are exact; only the summation order
        # is allowed this one-ulp-scale tolerance.
        np.testing.assert_allclose(
            record["return"], recomputed_return, rtol=0.0, atol=1e-6
        )
        if not saved_path.is_file():
            np.savez_compressed(saved_path, **record)
        records[name] = record
        print(f"evaluated return rollouts {name}", flush=True)

    rng = np.random.default_rng(CONFIG["bootstrap_seed"] + 1)
    weights = rng.multinomial(
        episodes, np.full(episodes, 1 / episodes), size=CONFIG["bootstrap_replicates"]
    )
    native_return = native["returns"]
    native_reward = native["rewards"]
    metrics = {
        "native": {
            "episodes": episodes,
            "discounted_return": scalar_episode_bootstrap(native_return, weights),
            "positive_return_fraction": scalar_episode_bootstrap(native_return > 0, weights),
            "mean_reward_by_step": native_reward.mean(axis=0),
            "mean_rewarded_steps": float(native_reward.sum(axis=1).mean()),
        }
    }
    per_episode = {}
    for name, record in records.items():
        predicted_return = record["return"].mean(axis=1)
        predicted_reward = record["reward"].mean(axis=1)
        error = predicted_return - native_return
        reward_time_mae = np.abs(predicted_reward - native_reward).mean(axis=1)
        per_episode[name] = {
            "predicted_return": predicted_return,
            "absolute_error": np.abs(error),
            "squared_error": error**2,
            "reward_time_mae": reward_time_mae,
        }
        metrics[name] = {
            "paths_per_initial_state": repeats,
            "predicted_discounted_return": scalar_episode_bootstrap(predicted_return, weights),
            "return_bias_predicted_minus_native": scalar_episode_bootstrap(error, weights),
            "return_mae": scalar_episode_bootstrap(np.abs(error), weights),
            "return_rmse": scalar_episode_bootstrap(error**2, weights, np.sqrt),
            "reward_time_mae": scalar_episode_bootstrap(reward_time_mae, weights),
            "mean_reward_by_step": predicted_reward.mean(axis=0),
            "mean_rewarded_steps": float(record["reward"].sum(axis=-1).mean()),
            "positive_return_path_fraction": float((record["return"] > 0).mean()),
            "sample_stationary_fraction": float(
                (
                    np.linalg.norm(
                        np.diff(record["states"][..., :2], axis=2), axis=-1
                    )
                    <= STATIONARY_THRESHOLD
                ).mean()
            ),
            "gate_mean": float(record["h"].mean()),
            "gate_below_one_fraction": float((record["h"] < 1).mean()),
        }
    contrasts = {}
    for seed in [0, 1]:
        contrasts[str(seed)] = {}
        for comparison, left, right in [
            ("candidate_minus_control", f"candidate_s{seed}", f"control_s{seed}"),
            ("control_minus_parent", f"control_s{seed}", f"parent_s{seed}"),
        ]:
            contrasts[str(seed)][comparison] = {
                "predicted_return": paired_episode_difference(
                    per_episode[left]["predicted_return"],
                    per_episode[right]["predicted_return"],
                    weights,
                ),
                "absolute_return_error": paired_episode_difference(
                    per_episode[left]["absolute_error"],
                    per_episode[right]["absolute_error"],
                    weights,
                ),
                "return_rmse": paired_episode_difference(
                    per_episode[left]["squared_error"],
                    per_episode[right]["squared_error"],
                    weights,
                    np.sqrt,
                ),
                "reward_time_mae": paired_episode_difference(
                    per_episode[left]["reward_time_mae"],
                    per_episode[right]["reward_time_mae"],
                    weights,
                ),
            }
    write_json(OUT / "return_metrics.json", metrics)
    write_json(OUT / "return_contrasts.json", contrasts)
    np.savez_compressed(
        OUT / "return_reward_curves.npz",
        native=native_reward.mean(axis=0),
        **{name: record["reward"].mean(axis=(0, 1)) for name, record in records.items()},
    )
    return metrics, contrasts


def main():
    resume = "--resume" in sys.argv[1:]
    if (OUT / "execution_started.json").exists() and not resume:
        raise RuntimeError("fresh execution required; this experiment has already started")
    if resume:
        if not (OUT / "failure.json").is_file() or (OUT / "completion.json").exists():
            raise RuntimeError("resume requires one incomplete failed execution")
    planned_model = sum(
        value
        for key, value in CONFIG["planned_model_successors"].items()
        if key != "total"
    )
    planned_native = sum(
        value for key, value in CONFIG["planned_native_steps"].items() if key != "total"
    )
    assert planned_model == CONFIG["planned_model_successors"]["total"] == 20349952
    assert planned_native == CONFIG["planned_native_steps"]["total"] == 14400
    assert planned_model <= CONFIG["model_successor_cap"]
    assert planned_native <= CONFIG["native_step_cap"]
    if resume:
        provenance = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
        sealed = json.loads((OUT / "execution_started.json").read_text(encoding="utf-8"))
        resume_number = 1
        while (OUT / f"resume_{resume_number}.json").exists():
            resume_number += 1
        write_json(
            OUT / f"resume_{resume_number}.json",
            {
                "reason": (
                    "resume after a non-model implementation or verification failure; "
                    "completed artifacts are reused"
                ),
                "previous_failure": json.loads((OUT / "failure.json").read_text(encoding="utf-8")),
                "original_driver_sha256": sealed["driver_sha256"],
                "resumed_driver_sha256": sha256(Path(__file__)),
                "repeated_training": False,
                "repeated_model_evaluation": False,
                "repeated_native_collection": False,
                "resume_scope": "saved-array summaries, remaining invariants, native actor collection, return rollouts",
            },
        )
    else:
        provenance = dependency_provenance()
        write_json(OUT / "provenance.json", provenance)
        sealed = {
            "status": "running",
            "started_unix_time": time.time(),
            "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
            "config_sha256": sha256(OUT / "config.json"),
            "driver_sha256": sha256(Path(__file__)),
            "collector_sha256": sha256(OUT / "collect_evaluation.py"),
            "planned_model_successors": planned_model,
            "planned_native_steps": planned_native,
        }
        write_json(OUT / "execution_started.json", sealed)
    ledger = Ledger(resume=resume)
    try:
        data, diagonal_ids, off_ids, state_mean, state_std = load_training_data(
            save_scaling=not resume
        )
        engine = ExperimentEngine(state_mean, state_std)
        if not resume:
            initialization_checks(engine, data, ledger)
            functions = make_training_functions(engine)
            for seed in [0, 1]:
                train_arm(
                    engine, data, diagonal_ids, off_ids, seed, "control", ledger, functions
                )
                train_arm(
                    engine, data, diagonal_ids, off_ids, seed, "candidate", ledger, functions
                )
            collect_paired_evaluation(ledger)
        else:
            for seed in [0, 1]:
                for arm in ["control", "candidate"]:
                    if not (OUT / f"{arm}_s{seed}.npz").is_file():
                        raise FileNotFoundError(f"missing completed checkpoint {arm}_s{seed}")
            if not PAIRED_EVALUATION.is_file():
                raise FileNotFoundError(PAIRED_EVALUATION)
        evaluation_data, _, _, _, _ = evaluate_one_step(
            engine, ledger, reuse_saved=resume
        )
        if not (resume and (OUT / "invariant_checks.json").is_file()):
            invariant_checks(engine, evaluation_data, ledger)
        if resume and (OUT / "native_actor_episodes.npz").is_file():
            with np.load(OUT / "native_actor_episodes.npz", allow_pickle=False) as loaded:
                native = {key: loaded[key] for key in ["states", "actions", "rewards", "returns"]}
        else:
            native = collect_native_actor(engine, ledger)
        evaluate_returns(engine, native, ledger)
        failure_overhead = ledger.model_total - planned_model
        if failure_overhead < 0 or failure_overhead % (
            CONFIG["return_native_episodes"]
            * CONFIG["return_paths_per_initial_state"]
            * CONFIG["return_horizon"]
        ) != 0:
            raise RuntimeError("model accounting does not reconcile")
        if ledger.model_total > CONFIG["model_successor_cap"]:
            raise RuntimeError("model-successor hard cap exceeded")
        assert ledger.native_total == planned_native
        current = dependency_provenance()
        old_dependencies = dict(provenance["dependency_sha256"])
        new_dependencies = dict(current["dependency_sha256"])
        old_driver = old_dependencies.pop(str(Path(__file__)))
        new_driver = new_dependencies.pop(str(Path(__file__)))
        if new_dependencies != old_dependencies:
            raise RuntimeError("non-driver dependency changed during execution")
        if resume:
            assert old_driver == sealed["driver_sha256"] and new_driver == sha256(Path(__file__))
        elif new_driver != old_driver:
            raise RuntimeError("driver changed during execution")
        write_json(
            OUT / "completion.json",
            {
                "status": "complete",
                "model_successors": ledger.model_total,
                "planned_model_successors": planned_model,
                "failed_verification_model_successor_overhead": failure_overhead,
                "native_steps": ledger.native_total,
                "training_updates": 480,
                "actor_updates": 0,
                "critic_updates": 0,
                "nominal_updates": 0,
                "architecture_candidates": 1,
                "selection": CONFIG["selection"],
                "source_hashes_unchanged": True,
            },
        )
        print("EXPERIMENT COMPLETE", flush=True)
    except BaseException as error:
        write_json(
            OUT / "failure.json",
            {
                "status": "failed",
                "error": repr(error),
                "model_successors_charged": ledger.model_total,
                "native_steps_charged": ledger.native_total,
            },
        )
        raise


if __name__ == "__main__":
    main()
