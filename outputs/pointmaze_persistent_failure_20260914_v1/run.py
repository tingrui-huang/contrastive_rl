"""Bounded failure-label-supervised persistent-mode PointMaze diagnostic."""
from __future__ import annotations

import hashlib
import importlib.util
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
RESPONSE = ROOT / "outputs" / "pointmaze_response_gate_20260914_v1"
EXTERNAL = Path(r"C:\Users\trhua\Documents\Codex\2026-09-08\f")
DIAGNOSIS = EXTERNAL / "outputs" / "pointmaze_return_gap_diagnosis_8bb3c31_v1"
TRAIN_DATA = EXTERNAL / "outputs" / "supervised_ett_native_probe_v1" / "paired_transitions.npz"
OLD_ENV = EXTERNAL / "work" / "branch-source-031d430" / "crl" / "envs.py"
OLD_TEACHER = EXTERNAL / "work" / "branch-source-f85a5f4" / "scripts" / "collect_swamp_windy.py"
OLD_BLIND = EXTERNAL / "work" / "branch-source-f85a5f4" / "scripts" / "collect_swamp_windy_baddemo.py"
PAIRED_EVALUATION = OUT / "paired_evaluation.npz"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))

sys.path.insert(0, str(ROOT))
from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402


def load_response_module():
    spec = importlib.util.spec_from_file_location(
        "completed_pointmaze_response_gate", RESPONSE / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RESPONSE_MODULE = load_response_module()
GOAL = np.asarray(RESPONSE_MODULE.GOAL, np.float32)


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


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hazardous(xy):
    xy = np.asarray(xy)
    return (
        (xy[..., 0] >= 3)
        & (xy[..., 0] < 6)
        & (xy[..., 1] >= 3)
        & (xy[..., 1] < 4)
    )


def hazardous_jax(xy):
    return (
        (xy[..., 0] >= 3)
        & (xy[..., 0] < 6)
        & (xy[..., 1] >= 3)
        & (xy[..., 1] < 4)
    )


class Ledger:
    def __init__(self, resume=False):
        if resume and (OUT / "ledger.json").exists():
            saved = json.loads((OUT / "ledger.json").read_text(encoding="utf-8"))
            self.position_counts = OrderedDict(saved["position_model"]["counts"])
            self.native_counts = OrderedDict(saved["native"]["counts"])
            self.head_counts = OrderedDict(saved["head_only"]["counts"])
        else:
            self.position_counts = OrderedDict()
            self.native_counts = OrderedDict()
            self.head_counts = OrderedDict()
            self.save()

    @property
    def position_total(self):
        return int(sum(self.position_counts.values()))

    @property
    def native_total(self):
        return int(sum(self.native_counts.values()))

    def position(self, number, purpose):
        number = int(number)
        if self.position_total + number > CONFIG["position_successor_cap"]:
            raise RuntimeError("position-model successor cap exhausted")
        self.position_counts[purpose] = self.position_counts.get(purpose, 0) + number
        self.save()

    def native(self, number, purpose):
        number = int(number)
        if self.native_total + number > CONFIG["native_step_cap"]:
            raise RuntimeError("native-step cap exhausted")
        self.native_counts[purpose] = self.native_counts.get(purpose, 0) + number
        self.save()

    def head(self, number, purpose):
        self.head_counts[purpose] = self.head_counts.get(purpose, 0) + int(number)
        self.save()

    def save(self):
        write_json(
            OUT / "ledger.json",
            {
                "position_model": {
                    "counts": self.position_counts,
                    "total": self.position_total,
                    "cap": CONFIG["position_successor_cap"],
                },
                "native": {
                    "counts": self.native_counts,
                    "total": self.native_total,
                    "cap": CONFIG["native_step_cap"],
                },
                "head_only": {"counts": self.head_counts, "total": int(sum(self.head_counts.values()))},
            },
        )


def dependency_provenance():
    required = [
        OUT / "PROTOCOL.md",
        OUT / "config.json",
        OUT / "collect_evaluation.py",
        OUT / "equivalent_experiment_search.json",
        Path(__file__),
        RESPONSE / "REPORT.md",
        RESPONSE / "PROTOCOL.md",
        RESPONSE / "config.json",
        RESPONSE / "provenance.json",
        RESPONSE / "completion.json",
        RESPONSE / "verification.json",
        RESPONSE / "run.py",
        RESPONSE / "gate_feature_scaling.npz",
        RESPONSE / "candidate_s0.npz",
        RESPONSE / "candidate_s1.npz",
        DIAGNOSIS / "REPORT.md",
        DIAGNOSIS / "results.json",
        DIAGNOSIS / "native_replay_audit.npz",
        TRAIN_DATA,
        TRAIN_DATA.parent / "provenance.json",
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
    response_completion = json.loads((RESPONSE / "completion.json").read_text())
    response_verification = json.loads((RESPONSE / "verification.json").read_text())
    response_resume = json.loads((RESPONSE / "resume_2.json").read_text())
    assert response_completion["status"] == "complete"
    assert response_verification["status"] == "pass"
    assert response_resume["resumed_driver_sha256"] == sha256(RESPONSE / "run.py")
    assert not json.loads((OUT / "equivalent_experiment_search.json").read_text())[
        "equivalent_completed_experiment_found"
    ]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head != CONFIG["start_commit"]:
        raise RuntimeError(f"expected start commit {CONFIG['start_commit']}, got {head}")
    result = {
        "working_head_at_start": head,
        "dependency_sha256": {str(path): sha256(path) for path in required},
        "response_experiment_complete_and_verified": True,
        "equivalent_completed_experiment_found": False,
        "explicit_failure_label_supervision": True,
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
    }
    write_json(OUT / "provenance.json", result)
    return result


HEAD_DIM = CONFIG["head_input_dimension"]
HIDDEN = CONFIG["head_hidden_units"][0]
W1 = slice(0, HEAD_DIM * HIDDEN)
B1 = slice(W1.stop, W1.stop + HIDDEN)
W2 = slice(B1.stop, B1.stop + HIDDEN * HIDDEN)
B2 = slice(W2.stop, W2.stop + HIDDEN)
W3 = slice(B2.stop, B2.stop + HIDDEN)
B3 = W3.stop
PARAM_DIM = B3 + 1


def head_initial(seed):
    rng = np.random.default_rng(CONFIG["head_initialization_seed_base"] + seed)
    theta = np.zeros(PARAM_DIM, np.float32)
    theta[W1] = rng.uniform(
        -np.sqrt(6 / (HEAD_DIM + HIDDEN)),
        np.sqrt(6 / (HEAD_DIM + HIDDEN)),
        W1.stop - W1.start,
    )
    theta[W2] = rng.uniform(
        -np.sqrt(6 / (2 * HIDDEN)),
        np.sqrt(6 / (2 * HIDDEN)),
        W2.stop - W2.start,
    )
    theta[W3] = rng.uniform(
        -np.sqrt(6 / (HIDDEN + 1)),
        np.sqrt(6 / (HIDDEN + 1)),
        W3.stop - W3.start,
    )
    return theta


def head_logits(theta, features):
    w1 = theta[W1].reshape(HEAD_DIM, HIDDEN)
    b1 = theta[B1]
    w2 = theta[W2].reshape(HIDDEN, HIDDEN)
    b2 = theta[B2]
    w3 = theta[W3]
    b3 = theta[B3]
    first = jnp.tanh(features @ w1 + b1)
    second = jnp.tanh(first @ w2 + b2)
    return second @ w3 + b3


def head_loss(theta, features, target):
    logits = head_logits(theta, features)
    return jnp.mean(jax.nn.softplus(logits) - target * logits)


HEAD_VALUE_GRAD = jax.jit(jax.value_and_grad(head_loss))
HEAD_PREDICT = jax.jit(lambda theta, features: jax.nn.sigmoid(head_logits(theta, features)))


def compute_scaling(train):
    mask = train["episode"] <= CONFIG["training_episode_last"]
    mean = train["s"][mask].mean(axis=0).astype(np.float32)
    std = train["s"][mask].std(axis=0).astype(np.float32)
    std = np.maximum(std, CONFIG["state_scale_floor"]).astype(np.float32)
    np.savez_compressed(OUT / "head_feature_scaling.npz", state_mean=mean, state_std=std)
    return mean, std


def head_features(state, xb, xq, next_xy, mean, std):
    next_xy = np.asarray(next_xy, np.float32)
    prefix = next_xy.shape[:-1]
    state_b = np.broadcast_to(np.asarray(state, np.float32).reshape(state.shape[:-1] + (1,) * (len(prefix) - (state.ndim - 1)) + (8,)), prefix + (8,)) if len(prefix) > state.ndim - 1 else np.asarray(state, np.float32)
    xb_b = np.broadcast_to(np.asarray(xb, np.float32).reshape(xb.shape[:-1] + (1,) * (len(prefix) - (xb.ndim - 1)) + (2,)), prefix + (2,)) if len(prefix) > xb.ndim - 1 else np.asarray(xb, np.float32)
    xq_b = np.broadcast_to(np.asarray(xq, np.float32).reshape(xq.shape[:-1] + (1,) * (len(prefix) - (xq.ndim - 1)) + (2,)), prefix + (2,)) if len(prefix) > xq.ndim - 1 else np.asarray(xq, np.float32)
    standardized = (state_b - mean) / std
    displacement = next_xy - state_b[..., :2]
    result = np.concatenate([standardized, xb_b, xq_b, displacement], axis=-1)
    assert result.shape[-1] == HEAD_DIM
    return result.astype(np.float32)


def predict_numpy(theta, features, chunk=65536):
    flat = np.asarray(features, np.float32).reshape(-1, HEAD_DIM)
    values = []
    for start in range(0, len(flat), chunk):
        values.append(np.asarray(HEAD_PREDICT(jnp.asarray(theta), jnp.asarray(flat[start:start + chunk]))))
    return np.concatenate(values).reshape(features.shape[:-1])


def train_head(seed, train, mean, std, ledger):
    path = OUT / f"head_s{seed}.npz"
    if path.exists():
        return load_npz(path)["theta"]
    alive = (train["episode"] <= CONFIG["training_episode_last"]) & (~train["dead_before"])
    indices = np.flatnonzero(alive)
    features = head_features(
        train["s"][indices], train["xb"][indices], train["xq"][indices],
        train["y"][indices, :2], mean, std,
    )
    target = train["dead_after"][indices].astype(np.float32)
    theta = head_initial(seed)
    initial_theta = theta.copy()
    adam_m = np.zeros(PARAM_DIM, np.float64)
    adam_v = np.zeros(PARAM_DIM, np.float64)
    rng = np.random.default_rng(CONFIG["head_batch_seed_base"] + seed)
    history = {key: [] for key in ["loss", "gradient_norm", "update_norm", "batch_positive_fraction", "theta_norm"]}
    ledger.head(CONFIG["head_updates"] * CONFIG["head_batch_size"], f"training_head_s{seed}_row_presentations")
    for step in range(CONFIG["head_updates"]):
        batch = rng.choice(len(indices), CONFIG["head_batch_size"])
        loss, gradient = HEAD_VALUE_GRAD(
            jnp.asarray(theta), jnp.asarray(features[batch]), jnp.asarray(target[batch])
        )
        gradient = np.asarray(gradient, np.float64)
        adam_m = CONFIG["adam_beta1"] * adam_m + (1 - CONFIG["adam_beta1"]) * gradient
        adam_v = CONFIG["adam_beta2"] * adam_v + (1 - CONFIG["adam_beta2"]) * gradient**2
        age = step + 1
        update = CONFIG["head_learning_rate"] * (
            adam_m / (1 - CONFIG["adam_beta1"] ** age)
        ) / (
            np.sqrt(adam_v / (1 - CONFIG["adam_beta2"] ** age)) + CONFIG["adam_epsilon"]
        )
        theta = (theta - update).astype(np.float32)
        history["loss"].append(float(loss))
        history["gradient_norm"].append(float(np.linalg.norm(gradient)))
        history["update_norm"].append(float(np.linalg.norm(update)))
        history["batch_positive_fraction"].append(float(target[batch].mean()))
        history["theta_norm"].append(float(np.linalg.norm(theta)))
        if age % 500 == 0:
            print(f"trained failure head {seed}: {age}/2000", flush=True)
    probability = predict_numpy(theta, features)
    if probability.std() <= 1e-4 or np.ptp(probability) <= 1e-3:
        raise RuntimeError(f"head {seed} predictions are effectively constant")
    np.savez_compressed(
        path,
        theta=theta,
        initial_theta=initial_theta,
        adam_m=adam_m,
        adam_v=adam_v,
        **{key: np.asarray(value) for key, value in history.items()},
    )
    summary = {
        "training_rows_alive_before": len(indices),
        "training_positive_onsets": int(target.sum()),
        "training_observed_onset_rate": float(target.mean()),
        "final_full_training_bce": float(head_loss(jnp.asarray(theta), jnp.asarray(features), jnp.asarray(target))),
        "prediction_mean": float(probability.mean()),
        "prediction_std": float(probability.std()),
        "prediction_min": float(probability.min()),
        "prediction_max": float(probability.max()),
        "prediction_mean_positive": float(probability[target == 1].mean()),
        "prediction_mean_negative": float(probability[target == 0].mean()),
        "not_constant": True,
    }
    write_json(OUT / f"head_s{seed}_training_summary.json", summary)
    return theta


def load_position_engine():
    scaling = load_npz(RESPONSE / "gate_feature_scaling.npz")
    return RESPONSE_MODULE.ExperimentEngine(scaling["state_mean"], scaling["state_std"])


def position_theta(seed):
    return load_npz(RESPONSE / f"candidate_s{seed}.npz")["theta"]


def make_joint_rollout(engine, state_mean, state_std, horizon):
    nominal, actor = engine.kernel.nominal, engine.kernel.actor
    mean = jnp.asarray(state_mean, jnp.float32)
    std = jnp.asarray(state_std, jnp.float32)

    def generate(position_parameters, head_parameters, states, goals, key, enabled):
        def step(carry, step_key):
            state, failed = carry
            nominal_key, actor_key, position_key, failure_key = jax.random.split(step_key, 4)
            xb = nominal.sample(state, nominal_key, 1, goal=goals)
            action = actor(state, goals, actor_key)
            proposed, detail = engine.candidate_sample(
                position_parameters, state, action, xb, position_key, 1
            )
            proposed = proposed[:, 0]
            feature = jnp.concatenate(
                [(state - mean) / std, xb, action, proposed[:, :2] - state[:, :2]], axis=-1
            )
            if enabled:
                probability_raw = jax.nn.sigmoid(head_logits(head_parameters, feature))
                probability = jnp.where(failed, 0.0, probability_raw)
                onset = (~failed) & (jax.random.uniform(failure_key, probability.shape) < probability)
            else:
                probability = jnp.zeros(failed.shape, jnp.float32)
                onset = jnp.zeros(failed.shape, bool)
            absorbed = jnp.concatenate([state[:, :2], state[:, :6]], axis=-1)
            following = jnp.where(failed[:, None], absorbed, proposed)
            next_failed = failed | onset
            visible_reward = jnp.linalg.norm(following[:, :2] - goals[:, :2], axis=-1) < 2
            reward = (visible_reward & (~next_failed)).astype(jnp.float32)
            record = {
                "next_state": following,
                "action": action,
                "xb": xb,
                "position_proposal_xy": proposed[:, :2],
                "d_before": failed,
                "d_after": next_failed,
                "onset": onset,
                "failure_probability": probability,
                "hazard_landing": hazardous_jax(proposed[:, :2]),
                "reward": reward,
                "atom": detail["stationary_atom"][:, 0],
                "h": detail["h"][:, 0],
                "projection": detail["projection_corrected"][:, 0],
            }
            return (following, next_failed), record

        initial_failed = jnp.zeros(states.shape[0], bool)
        (_, _), record = jax.lax.scan(
            step, (states, initial_failed), jax.random.split(key, horizon)
        )
        record = {name: jnp.swapaxes(value, 0, 1) for name, value in record.items()}
        record["states"] = jnp.concatenate([states[:, None], record.pop("next_state")], axis=1)
        record["d"] = jnp.concatenate([initial_failed[:, None], record["d_after"]], axis=1)
        record["return"] = jnp.sum(
            record["reward"]
            * jnp.power(jnp.float32(CONFIG["return_discount"]), jnp.arange(horizon)),
            axis=-1,
        )
        return record

    return jax.jit(generate, static_argnums=(5,))


def contract_checks(engine, heads, mean, std, train, ledger):
    saved_path = OUT / "contract_checks_initial.json"
    if saved_path.exists():
        return read_json(saved_path.name)
    alive_rows = np.flatnonzero(
        (train["episode"] <= CONFIG["training_episode_last"]) & (~train["dead_before"])
    )[:32]
    states = train["s"][alive_rows]
    goals = np.broadcast_to(GOAL, states.shape)
    rollout = make_joint_rollout(engine, mean, std, 8)
    checks = {}
    for seed in (0, 1):
        ledger.position(2 * len(states) * 8, "contract_checks")
        first = rollout(
            jnp.asarray(position_theta(seed)), jnp.asarray(heads[0]),
            jnp.asarray(states), jnp.asarray(goals),
            jax.random.PRNGKey(CONFIG["check_seed"] + seed), False,
        )
        second = rollout(
            jnp.asarray(position_theta(seed)), jnp.asarray(heads[1]),
            jnp.asarray(states), jnp.asarray(goals),
            jax.random.PRNGKey(CONFIG["check_seed"] + seed), False,
        )
        first, second = ({k: np.asarray(v) for k, v in item.items()} for item in (first, second))
        for field in first:
            np.testing.assert_array_equal(first[field], second[field])
        assert not first["d"].any() and not first["onset"].any()
        assert RESPONSE_MODULE.native_legal(first["states"][..., 1:, :2]).all()
        np.testing.assert_array_equal(first["states"][:, 1:, 2:], first["states"][:, :-1, :6])
        checks[f"p{seed}"] = {
            "disabled_mechanism_exact_across_head_parameters": True,
            "disabled_mode_always_alive": True,
            "f4_shift": True,
            "position_geometry_legal": True,
        }
    checks["contract"] = {
        "actor_input": "observed F4 and fixed goal only",
        "nominal_input": "observed F4 and fixed goal only",
        "head_input": "observed F4, xb, actor xq, sampled next XY displacement",
        "internal_mode_not_given_to_actor_or_nominal": True,
        "native_labels_not_referenced_by_rollout": True,
        "joint_action_lipschitz_claim_made": False,
    }
    write_json(saved_path, checks)
    return checks


def collect_paired(ledger):
    if PAIRED_EVALUATION.exists():
        return
    ledger.native(CONFIG["planned_native_steps"]["paired_branch_and_prefix"], "paired_branch_and_prefix")
    subprocess.run(
        [
            sys.executable,
            str(OUT / "collect_evaluation.py"),
            "--out",
            str(PAIRED_EVALUATION),
            "--episodes",
            str(CONFIG["paired_evaluation_episodes"]),
            "--seed",
            str(CONFIG["paired_collection_seed"]),
        ],
        cwd=ROOT,
        check=True,
    )


def onset_groups(data):
    alive = ~data["dead_before"]
    diagonal = np.all(data["xb"] == data["xq"], axis=1)
    hazard = hazardous(data["y"][:, :2])
    return OrderedDict(
        [
            ("all_alive", alive),
            ("diagonal", alive & diagonal),
            ("off_diagonal", alive & ~diagonal),
            ("hazardous_landing", alive & hazard),
            ("nonhazardous_landing", alive & ~hazard),
            ("diagonal_hazardous", alive & diagonal & hazard),
            ("diagonal_nonhazardous", alive & diagonal & ~hazard),
            ("off_diagonal_hazardous", alive & ~diagonal & hazard),
            ("off_diagonal_nonhazardous", alive & ~diagonal & ~hazard),
        ]
    )


def episode_bootstrap_rows(value, episode, mask, weights):
    counts = np.array([np.sum(mask & (episode == item)) for item in range(48)])
    totals = np.array([np.asarray(value)[mask & (episode == item)].sum() for item in range(48)])
    denominator = weights @ counts
    valid = denominator > 0
    replicates = (weights[valid] @ totals) / denominator[valid]
    return {
        "estimate": float(np.asarray(value)[mask].mean()),
        "ci95": np.quantile(replicates, [0.025, 0.975]),
        "contributing_episodes": int(np.sum(counts > 0)),
        "empty_replicates": int(np.sum(~valid)),
    }


def episode_bootstrap_row_difference(left, right, episode, mask, weights):
    counts = np.array([np.sum(mask & (episode == item)) for item in range(48)])
    left_totals = np.array(
        [np.asarray(left)[mask & (episode == item)].sum() for item in range(48)]
    )
    right_totals = np.array(
        [np.asarray(right)[mask & (episode == item)].sum() for item in range(48)]
    )
    denominator = weights @ counts
    valid = denominator > 0
    replicates = (
        (weights[valid] @ left_totals) - (weights[valid] @ right_totals)
    ) / denominator[valid]
    return {
        "estimate": float(np.asarray(left)[mask].mean() - np.asarray(right)[mask].mean()),
        "ci95": np.quantile(replicates, [0.025, 0.975]),
        "contributing_episodes": int(np.sum(counts > 0)),
        "empty_replicates": int(np.sum(~valid)),
    }


def summarize_onset(probability, data, groups, weights):
    target = data["dead_after"].astype(np.float64)
    probability = np.asarray(probability, np.float64)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    brier = (probability - target) ** 2
    log_loss = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    result = {}
    edges = np.asarray(CONFIG["calibration_bin_edges"])
    for label, mask in groups.items():
        if not mask.any():
            result[label] = {"rows": 0, "episodes": 0, "available": False}
            continue
        bins = []
        ece = 0.0
        group_count = int(mask.sum())
        for index in range(len(edges) - 1):
            upper_inclusive = index == len(edges) - 2
            selected = mask & (probability >= edges[index]) & (
                (probability <= edges[index + 1]) if upper_inclusive else (probability < edges[index + 1])
            )
            if selected.any():
                count = int(selected.sum())
                predicted = float(probability[selected].mean())
                observed = float(target[selected].mean())
                ece += count / group_count * abs(predicted - observed)
                bins.append(
                    {
                        "lower": float(edges[index]),
                        "upper": float(edges[index + 1]),
                        "count": count,
                        "predicted_rate": predicted,
                        "observed_rate": observed,
                    }
                )
        predicted_rate = episode_bootstrap_rows(probability, data["episode"], mask, weights)
        observed_rate = episode_bootstrap_rows(target, data["episode"], mask, weights)
        result[label] = {
            "rows": group_count,
            "episodes": int(len(np.unique(data["episode"][mask]))),
            "positive_onsets": int(target[mask].sum()),
            "observed_rate": observed_rate,
            "predicted_rate": predicted_rate,
            "predicted_minus_observed": episode_bootstrap_row_difference(
                probability, target, data["episode"], mask, weights
            ),
            "brier": episode_bootstrap_rows(brier, data["episode"], mask, weights),
            "log_loss": episode_bootstrap_rows(log_loss, data["episode"], mask, weights),
            "expected_calibration_error": float(ece),
            "calibration_bins": bins,
        }
    return result


def evaluate_onset(engine, heads, mean, std, ledger):
    data = load_npz(PAIRED_EVALUATION)
    assert data["s"].shape == (9600, 8)
    groups = onset_groups(data)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    weights = rng.multinomial(48, np.full(48, 1 / 48), CONFIG["bootstrap_replicates"])
    metrics = {"actual_successor": {}, "sampled_position_joint": {}}
    actual_features = head_features(data["s"], data["xb"], data["xq"], data["y"][:, :2], mean, std)
    for head_seed, theta in enumerate(heads):
        probability = predict_numpy(theta, actual_features)
        np.savez_compressed(
            OUT / f"head_h{head_seed}_actual_successor_evaluation.npz",
            probability=probability,
            target=data["dead_after"],
            alive_before=~data["dead_before"],
        )
        metrics["actual_successor"][f"h{head_seed}"] = summarize_onset(
            probability, data, groups, weights
        )
    for position_seed in (0, 1):
        sample_path = OUT / f"position_p{position_seed}_one_step_samples.npz"
        if sample_path.exists():
            sampled = load_npz(sample_path)
        else:
            outputs = {key: [] for key in ["samples_xy", "atom", "h", "projection"]}
            theta = position_theta(position_seed)
            for start in range(0, len(data["s"]), 128):
                stop = min(start + 128, len(data["s"]))
                ledger.position((stop - start) * CONFIG["one_step_position_samples"], "one_step_joint_evaluation")
                output, detail = engine.candidate_sample(
                    jnp.asarray(theta),
                    jnp.asarray(data["s"][start:stop]),
                    jnp.asarray(data["xq"][start:stop]),
                    jnp.asarray(data["xb"][start:stop]),
                    jax.random.PRNGKey(CONFIG["one_step_position_seed"] + position_seed * 100000 + start),
                    CONFIG["one_step_position_samples"],
                )
                output = np.asarray(output)
                outputs["samples_xy"].append(output[..., :2])
                outputs["atom"].append(np.asarray(detail["stationary_atom"]))
                outputs["h"].append(np.asarray(detail["h"]))
                outputs["projection"].append(np.asarray(detail["projection_corrected"]))
            sampled = {key: np.concatenate(value) for key, value in outputs.items()}
            sampled["legal"] = RESPONSE_MODULE.native_legal(sampled["samples_xy"])
            if not sampled["legal"].all():
                raise RuntimeError("frozen position model emitted illegal one-step sample")
            np.savez_compressed(sample_path, **sampled)
        metrics["sampled_position_joint"][f"p{position_seed}"] = {}
        features = head_features(
            data["s"], data["xb"], data["xq"], sampled["samples_xy"], mean, std
        )
        for head_seed, theta in enumerate(heads):
            sample_probability = predict_numpy(theta, features)
            mean_probability = sample_probability.mean(axis=1)
            np.savez_compressed(
                OUT / f"joint_p{position_seed}_h{head_seed}_onset_evaluation.npz",
                sample_probability=sample_probability,
                mean_probability=mean_probability,
            )
            metrics["sampled_position_joint"][f"p{position_seed}"][f"h{head_seed}"] = summarize_onset(
                mean_probability, data, groups, weights
            )
    write_json(OUT / "onset_metrics.json", metrics)
    np.savez_compressed(
        OUT / "onset_groups.npz", episode=data["episode"], **groups
    )
    return data, metrics


def collect_native_actor(engine, ledger):
    path = OUT / "native_actor_episodes.npz"
    if path.exists():
        return load_npz(path)
    episodes = CONFIG["return_native_episodes"]
    ledger.native(CONFIG["planned_native_steps"]["fixed_actor_returns"], "fixed_actor_returns")
    states, actions, rewards = [], [], []
    dead_before, dead_after, onset = [], [], []
    for episode in range(episodes):
        env = TwoRouteSwampWindyF4Env(
            seed=CONFIG["return_environment_seed"] + episode,
            active_prob=CONFIG["paired_active_probability"],
        )
        observation = env.reset()
        episode_states = [observation[:8].copy()]
        episode_actions, episode_rewards = [], []
        episode_before, episode_after, episode_onset = [], [], []
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
            before = bool(env.dead)
            following, reward, done, _ = env.step(action)
            after = bool(env.dead)
            if done:
                raise RuntimeError("fixed-length native actor episode terminated")
            expected_reward = float(
                (np.linalg.norm(following[:2] - GOAL[:2]) < 2) and (not after)
            )
            if reward != expected_reward:
                raise RuntimeError("native reward disagrees with joint task semantics")
            episode_states.append(following[:8].copy())
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_before.append(before)
            episode_after.append(after)
            episode_onset.append((not before) and after)
            observation = following
        states.append(episode_states)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        dead_before.append(episode_before)
        dead_after.append(episode_after)
        onset.append(episode_onset)
    rewards = np.asarray(rewards)
    result = {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "rewards": rewards,
        "dead_before": np.asarray(dead_before),
        "dead_after": np.asarray(dead_after),
        "onset": np.asarray(onset),
        "returns": np.sum(
            rewards * CONFIG["return_discount"] ** np.arange(CONFIG["return_horizon"]), axis=1
        ),
        "environment_seed": np.arange(episodes) + CONFIG["return_environment_seed"],
    }
    np.savez_compressed(path, **result)
    return result


def evaluate_rollouts(engine, heads, mean, std, native, ledger):
    roots = np.repeat(native["states"][:, 0], CONFIG["return_paths_per_initial_state"], axis=0)
    goals = np.broadcast_to(GOAL, roots.shape)
    rollout = make_joint_rollout(engine, mean, std, CONFIG["return_horizon"])
    arms = [
        ("baseline_p0", 0, 0, False),
        ("baseline_p1", 1, 0, False),
        ("joint_p0_h0", 0, 0, True),
        ("joint_p0_h1", 0, 1, True),
        ("joint_p1_h0", 1, 0, True),
        ("joint_p1_h1", 1, 1, True),
    ]
    records = {}
    for name, position_seed, head_seed, enabled in arms:
        path = OUT / f"{name}_rollouts.npz"
        if path.exists():
            record = load_npz(path)
        else:
            ledger.position(
                len(roots) * CONFIG["return_horizon"], "complete_rollouts"
            )
            generated = rollout(
                jnp.asarray(position_theta(position_seed)),
                jnp.asarray(heads[head_seed]),
                jnp.asarray(roots),
                jnp.asarray(goals),
                jax.random.PRNGKey(CONFIG["return_model_seed"]),
                enabled,
            )
            record = {
                key: np.asarray(value).reshape(
                    (CONFIG["return_native_episodes"], CONFIG["return_paths_per_initial_state"])
                    + value.shape[1:]
                )
                for key, value in generated.items()
            }
            np.savez_compressed(path, **record)
        np.testing.assert_array_equal(record["states"][:, :, 1:, 2:], record["states"][:, :, :-1, :6])
        if not RESPONSE_MODULE.native_legal(record["states"][:, :, 1:, :2]).all():
            raise RuntimeError(f"illegal generated successor in {name}")
        if not enabled:
            assert not record["d"].any() and not record["onset"].any()
        records[name] = record
        print(f"completed rollout arm {name}", flush=True)
    return records


def scalar_bootstrap(value, weights):
    value = np.asarray(value, np.float64)
    replicates = weights @ value / weights.sum(axis=1)
    return {"estimate": float(value.mean()), "ci95": np.quantile(replicates, [0.025, 0.975])}


def ratio_bootstrap(numerator, denominator, weights):
    numerator = np.asarray(numerator, np.float64)
    denominator = np.asarray(denominator, np.float64)
    point_denominator = denominator.sum()
    if point_denominator == 0:
        return {"available": False, "reason": "zero denominator"}
    boot_num = weights @ numerator
    boot_den = weights @ denominator
    valid = boot_den > 0
    return {
        "estimate": float(numerator.sum() / point_denominator),
        "ci95": np.quantile(boot_num[valid] / boot_den[valid], [0.025, 0.975]),
        "empty_replicates": int((~valid).sum()),
    }


def paired_difference(left, right, weights):
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    difference = left - right
    return scalar_bootstrap(difference, weights)


def paired_ratio_difference(left_num, left_den, right_num, right_den, weights):
    left_num = np.asarray(left_num, np.float64)
    left_den = np.asarray(left_den, np.float64)
    right_num = np.asarray(right_num, np.float64)
    right_den = np.asarray(right_den, np.float64)
    valid_point = left_den.sum() > 0 and right_den.sum() > 0
    if not valid_point:
        return {"available": False, "reason": "zero point denominator"}
    ln, ld = weights @ left_num, weights @ left_den
    rn, rd = weights @ right_num, weights @ right_den
    valid = (ld > 0) & (rd > 0)
    point = left_num.sum() / left_den.sum() - right_num.sum() / right_den.sum()
    return {
        "estimate": float(point),
        "ci95": np.quantile(ln[valid] / ld[valid] - rn[valid] / rd[valid], [0.025, 0.975]),
        "empty_replicates": int((~valid).sum()),
    }


def trajectory_metrics(native, records):
    rng = np.random.default_rng(CONFIG["bootstrap_seed"] + 1)
    weights = rng.multinomial(48, np.full(48, 1 / 48), CONFIG["bootstrap_replicates"])
    native_any_reward = native["returns"] > 0
    native_failed = native["dead_after"].any(axis=1)
    native_survive_no_reward = (~native_failed) & (~native_any_reward)
    native_onset_time = np.where(
        native_failed, np.argmax(native["onset"], axis=1) + 1, 0
    )
    native_components = {
        "any_reward": native_any_reward.astype(float),
        "failed": native_failed.astype(float),
        "survive_no_reward": native_survive_no_reward.astype(float),
        "return": native["returns"],
        "conditional_return_num": native["returns"] * native_any_reward,
        "conditional_return_den": native_any_reward.astype(float),
        "failure_time_num": native_onset_time * native_failed,
        "failure_time_den": native_failed.astype(float),
    }
    result = {
        "native": {
            "episodes": 48,
            "probability_any_reward": scalar_bootstrap(native_components["any_reward"], weights),
            "probability_failure": scalar_bootstrap(native_components["failed"], weights),
            "probability_survive_without_reward": scalar_bootstrap(native_components["survive_no_reward"], weights),
            "mean_discounted_return": scalar_bootstrap(native_components["return"], weights),
            "return_conditional_on_reward": ratio_bootstrap(
                native_components["conditional_return_num"], native_components["conditional_return_den"], weights
            ),
            "failure_time_conditional_on_failure": ratio_bootstrap(
                native_components["failure_time_num"], native_components["failure_time_den"], weights
            ),
            "failure_count": int(native_failed.sum()),
            "rewarded_episode_count": int(native_any_reward.sum()),
            "survive_without_reward_count": int(native_survive_no_reward.sum()),
            "failure_outside_hazard_count": int(
                (native["onset"] & ~hazardous(native["states"][:, 1:, :2])).sum()
            ),
        },
        "models": {},
        "gaps_from_native": {},
    }
    for name, record in records.items():
        any_reward_path = record["return"] > 0
        failed_path = record["d"][:, :, -1]
        survive_no_reward_path = (~failed_path) & (~any_reward_path)
        onset_time = np.where(failed_path, np.argmax(record["onset"], axis=2) + 1, 0)
        alive_transition = ~record["d_before"]
        outside = ~record["hazard_landing"]
        sampled_outside_onset = record["onset"] & outside
        components = {
            "any_reward": any_reward_path.mean(axis=1),
            "failed": failed_path.mean(axis=1),
            "survive_no_reward": survive_no_reward_path.mean(axis=1),
            "return": record["return"].mean(axis=1),
            "conditional_return_num": (record["return"] * any_reward_path).sum(axis=1),
            "conditional_return_den": any_reward_path.sum(axis=1),
            "failure_time_num": (onset_time * failed_path).sum(axis=1),
            "failure_time_den": failed_path.sum(axis=1),
            "outside_probability_num": (
                record["failure_probability"] * alive_transition * outside
            ).sum(axis=(1, 2)),
            "outside_probability_den": (alive_transition * outside).sum(axis=(1, 2)),
            "outside_onset_num": sampled_outside_onset.sum(axis=(1, 2)),
            "onset_den": record["onset"].sum(axis=(1, 2)),
            "outside_onset_paths": sampled_outside_onset.any(axis=2).mean(axis=1),
        }
        d_monotone = np.all(record["d"][:, :, 1:] >= record["d"][:, :, :-1])
        frozen = record["d_before"]
        frozen_xy_ok = np.all(
            record["states"][:, :, 1:, :2][frozen]
            == record["states"][:, :, :-1, :2][frozen]
        )
        onset_movement = np.linalg.norm(
            record["states"][:, :, 1:, :2] - record["states"][:, :, :-1, :2], axis=-1
        )[record["onset"]]
        fatal_proposal_exact = np.array_equal(
            record["states"][:, :, 1:, :2][record["onset"]],
            record["position_proposal_xy"][record["onset"]],
        )
        result["models"][name] = {
            "paths": int(record["return"].size),
            "probability_any_reward": scalar_bootstrap(components["any_reward"], weights),
            "probability_failure": scalar_bootstrap(components["failed"], weights),
            "probability_survive_without_reward": scalar_bootstrap(components["survive_no_reward"], weights),
            "mean_discounted_return": scalar_bootstrap(components["return"], weights),
            "return_conditional_on_reward": ratio_bootstrap(
                components["conditional_return_num"], components["conditional_return_den"], weights
            ),
            "failure_time_conditional_on_failure": ratio_bootstrap(
                components["failure_time_num"], components["failure_time_den"], weights
            ),
            "expected_failure_probability_outside_hazard": ratio_bootstrap(
                components["outside_probability_num"], components["outside_probability_den"], weights
            ),
            "fraction_sampled_onsets_outside_hazard": ratio_bootstrap(
                components["outside_onset_num"], components["onset_den"], weights
            ),
            "probability_path_has_outside_hazard_onset": scalar_bootstrap(
                components["outside_onset_paths"], weights
            ),
            "sampled_onset_count": int(record["onset"].sum()),
            "sampled_outside_hazard_onset_count": int(sampled_outside_onset.sum()),
            "mode_monotone": bool(d_monotone),
            "absorbing_xy_exact": bool(frozen_xy_ok),
            "fatal_onset_proposal_preserved_exactly": bool(fatal_proposal_exact),
            "fatal_onset_count": int(record["onset"].sum()),
            "fatal_onset_moving_fraction": float(
                (onset_movement > CONFIG["stationary_threshold"]).mean()
            ) if len(onset_movement) else None,
            "persistence_violations": int((record["d"][:, :, 1:] < record["d"][:, :, :-1]).sum()),
        }
        result["gaps_from_native"][name] = {
            "probability_any_reward": paired_difference(
                components["any_reward"], native_components["any_reward"], weights
            ),
            "probability_failure": paired_difference(
                components["failed"], native_components["failed"], weights
            ),
            "probability_survive_without_reward": paired_difference(
                components["survive_no_reward"], native_components["survive_no_reward"], weights
            ),
            "mean_discounted_return": paired_difference(
                components["return"], native_components["return"], weights
            ),
            "return_conditional_on_reward": paired_ratio_difference(
                components["conditional_return_num"], components["conditional_return_den"],
                native_components["conditional_return_num"], native_components["conditional_return_den"], weights,
            ),
            "failure_time_conditional_on_failure": paired_ratio_difference(
                components["failure_time_num"], components["failure_time_den"],
                native_components["failure_time_num"], native_components["failure_time_den"], weights,
            ),
        }
    write_json(OUT / "trajectory_metrics.json", result)
    np.savez_compressed(
        OUT / "reward_curves.npz",
        native=native["rewards"].mean(axis=0),
        **{name: record["reward"].mean(axis=(0, 1)) for name, record in records.items()},
    )
    return result


def final_contract_checks(records):
    checks = read_json("contract_checks_initial.json")
    for position_seed in (0, 1):
        baseline = records[f"baseline_p{position_seed}"]
        for head_seed in (0, 1):
            joint = records[f"joint_p{position_seed}_h{head_seed}"]
            alive = ~joint["d_before"]
            for field in ["action", "xb", "position_proposal_xy"]:
                np.testing.assert_array_equal(joint[field][alive], baseline[field][alive])
            onset = joint["onset"]
            np.testing.assert_array_equal(
                joint["states"][:, :, 1:, :2][onset], joint["position_proposal_xy"][onset]
            )
            failed = joint["d_before"]
            np.testing.assert_array_equal(
                joint["states"][:, :, 1:, :2][failed], joint["states"][:, :, :-1, :2][failed]
            )
            assert np.all(joint["d"][:, :, 1:] >= joint["d"][:, :, :-1])
            checks[f"joint_p{position_seed}_h{head_seed}"] = {
                "pre_onset_actor_nominal_and_position_proposal_match_baseline": True,
                "fatal_onset_landing_preserved": True,
                "failed_mode_absorbing": True,
                "absorbing_xy_exact": True,
                "absorbing_f4_shift": True,
                "position_geometry_legal": True,
                "failure_support_mask_applied": False,
            }
    write_json(OUT / "contract_checks.json", checks)


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    resume = "--resume" in sys.argv[1:]
    if (OUT / "execution_started.json").exists() and not resume:
        raise RuntimeError("experiment already started; use --resume only after a recorded failure")
    if resume and ((OUT / "completion.json").exists() or not (OUT / "failure.json").exists()):
        raise RuntimeError("resume requires an incomplete recorded failure")
    planned_position = sum(
        value for key, value in CONFIG["planned_position_successors"].items() if key != "total"
    )
    planned_native = sum(
        value for key, value in CONFIG["planned_native_steps"].items() if key != "total"
    )
    assert planned_position == CONFIG["planned_position_successors"]["total"] == 2151424
    assert planned_native == CONFIG["planned_native_steps"]["total"] == 14400
    ledger = Ledger(resume=resume)
    try:
        if not resume:
            provenance = dependency_provenance()
            write_json(
                OUT / "execution_started.json",
                {
                    "status": "running",
                    "started_unix_time": time.time(),
                    "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
                    "config_sha256": sha256(OUT / "config.json"),
                    "driver_sha256": sha256(Path(__file__)),
                    "collector_sha256": sha256(OUT / "collect_evaluation.py"),
                    "planned_position_successors": planned_position,
                    "planned_native_steps": planned_native,
                    "explicit_failure_label_supervision": True,
                },
            )
        else:
            provenance = read_json("provenance.json")
            resume_index = len(list(OUT.glob("resume*.json")))
            write_json(
                OUT / f"resume_{resume_index}.json",
                {
                    "reason": "resume after recorded implementation/reporting failure; completed stages are reused",
                    "driver_sha256": sha256(Path(__file__)),
                    "repeated_head_training": False,
                    "repeated_paired_collection": False,
                    "repeated_native_collection": False,
                },
            )
        train = load_npz(TRAIN_DATA)
        mean, std = (
            (load_npz(OUT / "head_feature_scaling.npz")["state_mean"], load_npz(OUT / "head_feature_scaling.npz")["state_std"])
            if (OUT / "head_feature_scaling.npz").exists()
            else compute_scaling(train)
        )
        engine = load_position_engine()
        heads = [train_head(seed, train, mean, std, ledger) for seed in (0, 1)]
        if not (OUT / "frozen_settings.json").exists():
            write_json(
                OUT / "frozen_settings.json",
                {
                    "frozen_before_new_evaluation_collection": True,
                    "position_checkpoint_sha256": {
                        f"p{seed}": sha256(RESPONSE / f"candidate_s{seed}.npz") for seed in (0, 1)
                    },
                    "head_checkpoint_sha256": {
                        f"h{seed}": sha256(OUT / f"head_s{seed}.npz") for seed in (0, 1)
                    },
                    "configuration_sha256": sha256(OUT / "config.json"),
                    "protocol_sha256": sha256(OUT / "PROTOCOL.md"),
                    "selection": CONFIG["selection"],
                },
            )
        contract_checks(engine, heads, mean, std, train, ledger)
        collect_paired(ledger)
        data, onset = evaluate_onset(engine, heads, mean, std, ledger)
        native = collect_native_actor(engine, ledger)
        records = evaluate_rollouts(engine, heads, mean, std, native, ledger)
        metrics = trajectory_metrics(native, records)
        final_contract_checks(records)
        current_hashes = {
            path: sha256(path)
            for path in provenance["dependency_sha256"]
            if Path(path) != Path(__file__)
        }
        expected_hashes = {
            path: value
            for path, value in provenance["dependency_sha256"].items()
            if Path(path) != Path(__file__)
        }
        if current_hashes != expected_hashes:
            raise RuntimeError("non-driver dependency changed during execution")
        assert ledger.position_total == planned_position
        assert ledger.native_total == planned_native
        completion = {
            "status": "complete",
            "position_model_successors": ledger.position_total,
            "native_steps": ledger.native_total,
            "head_training_updates": 2 * CONFIG["head_updates"],
            "head_training_row_presentations": 2 * CONFIG["head_updates"] * CONFIG["head_batch_size"],
            "position_model_updates": 0,
            "actor_updates": 0,
            "critic_updates": 0,
            "nominal_updates": 0,
            "head_configurations": 1,
            "head_seeds": 2,
            "joint_combinations": 4,
            "baseline_arms": 2,
            "source_hashes_unchanged": True,
            "explicit_failure_label_supervision": True,
            "evaluation_tuning": False,
        }
        write_json(OUT / "completion.json", completion)
        print(json.dumps(completion), flush=True)
    except Exception as exc:
        write_json(
            OUT / "failure.json",
            {
                "status": "failed",
                "error": repr(exc),
                "position_successors_charged": ledger.position_total,
                "native_steps_charged": ledger.native_total,
            },
        )
        raise


if __name__ == "__main__":
    main()
