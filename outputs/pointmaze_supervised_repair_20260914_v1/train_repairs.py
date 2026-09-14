"""Train the two predeclared supervised repair components and freeze validation selections."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common import (
    CONFIG,
    OUT,
    PARENT,
    RESPONSE_MODULE,
    Ledger,
    exact_diagonal,
    full_energy_score,
    hazardous,
    input_groups,
    load_head_scaling,
    load_npz,
    old_head,
    old_position,
    write_json,
)
from ett.convex_action_transition import energy_score


def alive_rows(split):
    loaded = load_npz(OUT / f"{split}_paired.npz")
    mask = ~loaded["dead_before"]
    names = ("episode", "time", "prefix_source", "s", "xb", "xq", "y", "onset", "reward", "query_type")
    return {name: loaded[name][mask] for name in names}


def make_position_loss(engine):
    count = CONFIG["position_training_samples"]

    def loss(theta, state, xb, xq, target, key):
        output, detail = engine.candidate_sample(theta, state, xq, xb, key, count)
        return energy_score(output[..., :2], target[:, :2]).mean(), detail["h"].mean()

    return jax.jit(jax.vmap(loss, in_axes=(0, None, None, None, None, None)))


def sample_position(engine, theta, data, seed):
    count = CONFIG["position_validation_samples"]
    outputs, atoms, gates = [], [], []
    for start in range(0, len(data["s"]), 256):
        stop = min(start + 256, len(data["s"]))
        output, detail = engine.candidate_sample(
            jnp.asarray(theta),
            jnp.asarray(data["s"][start:stop]),
            jnp.asarray(data["xq"][start:stop]),
            jnp.asarray(data["xb"][start:stop]),
            jax.random.PRNGKey(CONFIG["position_validation_seed_base"] + seed * 100000 + start),
            count,
        )
        outputs.append(np.asarray(output))
        atoms.append(np.asarray(detail["stationary_atom"]))
        gates.append(np.asarray(detail["h"]))
    output = np.concatenate(outputs)
    samples = output[..., :2]
    np.testing.assert_array_equal(
        output[..., 2:], np.broadcast_to(data["s"][:, None, :6], output[..., 2:].shape)
    )
    if not RESPONSE_MODULE.native_legal(samples).all():
        raise RuntimeError("validation position sample is illegal")
    return samples, np.concatenate(atoms), np.concatenate(gates)


def event_point(probability, target, mask):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return {"rows": 0, "observed": None, "predicted": None, "bias": None, "brier": None}
    probability = np.asarray(probability)[mask]
    target = np.asarray(target, np.float64)[mask]
    return {
        "rows": int(mask.sum()),
        "observed": float(target.mean()),
        "predicted": float(probability.mean()),
        "bias": float((probability - target).mean()),
        "brier": float(((probability - target) ** 2).mean()),
    }


def position_validation_metrics(samples, data):
    current = data["s"][:, :2]
    actual = data["y"][:, :2]
    sample_delta = samples - current[:, None]
    actual_delta = actual - current
    diag = exact_diagonal(data)
    score = full_energy_score(samples, actual)
    current_hazard = hazardous(current)
    actual_hazard = hazardous(actual)
    sampled_hazard = hazardous(samples)
    predicted_hazard = sampled_hazard.mean(axis=1)
    result = {
        "rows": len(current),
        "diagonal_rows": int(diag.sum()),
        "off_diagonal_rows": int((~diag).sum()),
        "diagonal_energy": float(score[diag].mean()),
        "off_diagonal_energy": float(score[~diag].mean()),
        "selection_score": float(0.5 * (score[diag].mean() + score[~diag].mean())),
        "hazard_entry": event_point(predicted_hazard, actual_hazard, ~current_hazard),
        "hazard_residence": event_point(predicted_hazard, actual_hazard, current_hazard),
        "hazard_exit": event_point(1 - predicted_hazard, ~actual_hazard, current_hazard),
        "rightward_progress": event_point(
            (sample_delta[..., 0] > CONFIG["stationary_threshold"]).mean(axis=1),
            actual_delta[:, 0] > CONFIG["stationary_threshold"],
            np.ones(len(current), bool),
        ),
        "horizontal_displacement": {
            "actual_mean": float(actual_delta[:, 0].mean()),
            "predicted_mean": float(sample_delta[..., 0].mean()),
            "bias": float(sample_delta[..., 0].mean(axis=1).mean() - actual_delta[:, 0].mean()),
        },
        "current_hazard_horizontal_displacement": {
            "rows": int(current_hazard.sum()),
            "actual_mean": float(actual_delta[current_hazard, 0].mean()) if current_hazard.any() else None,
            "predicted_mean": float(sample_delta[current_hazard, :, 0].mean()) if current_hazard.any() else None,
            "bias": float(sample_delta[current_hazard, :, 0].mean() - actual_delta[current_hazard, 0].mean()) if current_hazard.any() else None,
        },
        "mean_xy_error": float(np.linalg.norm(samples.mean(axis=1) - actual, axis=1).mean()),
    }
    return result, score


def train_position(seed, train, validation, ledger):
    checkpoint_path = OUT / f"position_repaired_s{seed}.npz"
    if checkpoint_path.exists():
        return load_npz(checkpoint_path)["theta"]
    engine = PARENT.load_position_engine()
    theta = old_position(seed).astype(np.float32).copy()
    assert theta.shape == (69,)
    adam_m = np.zeros(theta.shape, np.float64)
    adam_v = np.zeros(theta.shape, np.float64)
    trajectory = []
    training_history = []
    validation_history = []
    validation_thetas = []
    best_score = np.inf
    best_step = None
    best_theta = None
    best_samples = None
    best_atom = None
    best_gate = None
    diag_ids = np.flatnonzero(exact_diagonal(train))
    off_ids = np.flatnonzero(~exact_diagonal(train))
    if not len(diag_ids) or not len(off_ids):
        raise RuntimeError("missing position training partition")
    training_calls = (
        CONFIG["position_updates"] * 2 * CONFIG["position_signed_candidates"]
        * CONFIG["position_batch_per_term"] * CONFIG["position_training_samples"]
    )
    validation_points = CONFIG["position_updates"] // CONFIG["position_validation_frequency"] + 1
    ledger.position(training_calls, f"position_training_s{seed}")
    ledger.position(
        validation_points * len(validation["s"]) * CONFIG["position_validation_samples"],
        f"position_validation_s{seed}",
    )
    ledger.updates(CONFIG["position_updates"], f"position_training_s{seed}")
    batch_loss = make_position_loss(engine)
    sigma = np.concatenate([
        np.full(16, CONFIG["position_sigma"]["diagonal"]),
        np.full(32, CONFIG["position_sigma"]["response"]),
        np.full(21, CONFIG["position_sigma"]["gate"]),
    ])
    rate = np.concatenate([
        np.full(16, CONFIG["position_learning_rate"]["diagonal"]),
        np.full(32, CONFIG["position_learning_rate"]["response"]),
        np.full(21, CONFIG["position_learning_rate"]["gate"]),
    ])

    def validate(step):
        nonlocal best_score, best_step, best_theta, best_samples, best_atom, best_gate
        samples, atom, gate = sample_position(engine, theta, validation, seed)
        metrics, score = position_validation_metrics(samples, validation)
        metrics["step"] = step
        metrics["atom_fraction"] = float(atom.mean())
        metrics["gate_mean"] = float(gate.mean())
        validation_history.append(metrics)
        validation_thetas.append(theta.copy())
        if metrics["selection_score"] < best_score:
            best_score = metrics["selection_score"]
            best_step = step
            best_theta = theta.copy()
            best_samples = samples.copy()
            best_atom = atom.copy()
            best_gate = gate.copy()

    validate(0)
    for step in range(CONFIG["position_updates"]):
        rng = np.random.default_rng(
            CONFIG["position_training_direction_seed_base"] + seed * 100000 + step
        )
        batches = [
            rng.choice(diag_ids, CONFIG["position_batch_per_term"]),
            rng.choice(off_ids, CONFIG["position_batch_per_term"]),
        ]
        directions = rng.normal(size=(CONFIG["position_directions"], len(theta)))
        trials = np.stack([theta + sigma * directions, theta - sigma * directions], axis=1)
        trials = trials.reshape(CONFIG["position_signed_candidates"], len(theta)).astype(np.float32)
        values, gate_means = [], []
        for term, indices in enumerate(batches):
            losses, h_mean = batch_loss(
                jnp.asarray(trials),
                jnp.asarray(train["s"][indices]),
                jnp.asarray(train["xb"][indices]),
                jnp.asarray(train["xq"][indices]),
                jnp.asarray(train["y"][indices]),
                jax.random.PRNGKey(
                    CONFIG["position_training_model_seed_base"] + seed * 100000 + step * 2 + term
                ),
            )
            values.append(np.asarray(losses).reshape(CONFIG["position_directions"], 2))
            gate_means.append(float(np.asarray(h_mean).mean()))
        components = np.stack([
            np.mean((loss[:, 0] - loss[:, 1])[:, None] * directions, axis=0) / (2 * sigma)
            for loss in values
        ])
        components[0, 16:] = 0
        gradient = components.sum(axis=0)
        adam_m = CONFIG["adam_beta1"] * adam_m + (1 - CONFIG["adam_beta1"]) * gradient
        adam_v = CONFIG["adam_beta2"] * adam_v + (1 - CONFIG["adam_beta2"]) * gradient**2
        age = step + 1
        update = rate * (adam_m / (1 - CONFIG["adam_beta1"]**age)) / (
            np.sqrt(adam_v / (1 - CONFIG["adam_beta2"]**age)) + CONFIG["adam_epsilon"]
        )
        for section, cap in (
            (slice(0, 16), CONFIG["position_update_cap"]["diagonal"]),
            (slice(16, 48), CONFIG["position_update_cap"]["response"]),
            (slice(48, 69), CONFIG["position_update_cap"]["gate"]),
        ):
            norm = np.linalg.norm(update[section])
            update[section] *= min(1.0, cap / max(norm, 1e-12))
        theta = (theta - update).astype(np.float32)
        if not np.isfinite(theta).all():
            raise FloatingPointError("non-finite position parameters")
        trajectory.append(theta.copy())
        training_history.append({
            "step": age,
            "diagonal_signed_losses": values[0],
            "off_diagonal_signed_losses": values[1],
            "diagonal_gradient_norm": float(np.linalg.norm(components[0])),
            "off_diagonal_gradient_norm": float(np.linalg.norm(components[1])),
            "update_norms": [
                float(np.linalg.norm(update[:16])),
                float(np.linalg.norm(update[16:48])),
                float(np.linalg.norm(update[48:])),
            ],
            "signed_gate_means": gate_means,
        })
        if age % CONFIG["position_validation_frequency"] == 0:
            validate(age)
            print(f"position seed {seed}: {age}/{CONFIG['position_updates']}", flush=True)
    assert best_theta is not None
    np.savez_compressed(
        checkpoint_path,
        theta=best_theta,
        initialization_theta=old_position(seed),
        selected_step=np.asarray(best_step),
        validation_theta=np.asarray(validation_thetas),
        training_trajectory=np.asarray(trajectory),
        adam_m=adam_m,
        adam_v=adam_v,
    )
    np.savez_compressed(
        OUT / f"position_validation_selected_s{seed}.npz",
        samples_xy=best_samples,
        atom=best_atom,
        gate=best_gate,
        target_xy=validation["y"][:, :2],
    )
    write_json(OUT / f"position_training_s{seed}.json", {
        "train_alive_rows": len(train["s"]),
        "train_diagonal_rows": len(diag_ids),
        "train_off_diagonal_rows": len(off_ids),
        "updates": CONFIG["position_updates"],
        "optimizer_state_reset": True,
        "trainable_parameters": {"diagonal_offsets": 16, "response_coordinates": 32, "gate_weights": 21},
        "history": training_history,
    })
    write_json(OUT / f"position_validation_s{seed}.json", {
        "selection_rule": CONFIG["position_validation_selection"],
        "selected_step": best_step,
        "selected_score": best_score,
        "history": validation_history,
    })
    return best_theta


def head_validation(theta, data, mean, std):
    features = PARENT.head_features(data["s"], data["xb"], data["xq"], data["y"][:, :2], mean, std)
    logits = np.asarray(PARENT.head_logits(jnp.asarray(theta), jnp.asarray(features)))
    actual_hazard = hazardous(data["y"][:, :2])
    target = data["onset"].astype(np.float64)
    probability = np.asarray(jax.nn.sigmoid(jnp.asarray(logits))) * actual_hazard
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    loss = -(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    raw_loss = np.asarray(jax.nn.softplus(jnp.asarray(logits)) - jnp.asarray(target) * jnp.asarray(logits))
    current = input_groups(data["s"])
    other = ~(current["approach_left"] | current["inside_hazard"])
    selection_masks = [current["approach_left"] & actual_hazard, current["inside_hazard"] & actual_hazard, other & actual_hazard]
    group_losses = [float(raw_loss[mask].mean()) for mask in selection_masks if mask.any()]
    metrics = {
        "selection_score": float(np.mean(group_losses)),
        "selection_group_hazardous_rows": [int(mask.sum()) for mask in selection_masks],
        "all": event_point(probability, target, np.ones(len(target), bool)),
        "actual_hazardous": event_point(probability, target, actual_hazard),
        "actual_nonhazardous": event_point(probability, target, ~actual_hazard),
        "approach_left": event_point(probability, target, current["approach_left"]),
        "inside_hazard": event_point(probability, target, current["inside_hazard"]),
        "supported_log_loss_mean": float(loss.mean()),
        "outside_support_probability_max": float(probability[~actual_hazard].max(initial=0)),
    }
    return metrics, probability, features, actual_hazard


def train_head(seed, train, validation, ledger):
    checkpoint_path = OUT / f"head_repaired_s{seed}.npz"
    if checkpoint_path.exists():
        return load_npz(checkpoint_path)["theta"]
    mean, std = load_head_scaling()
    theta = old_head(seed).astype(np.float32).copy()
    assert theta.shape == (PARENT.PARAM_DIM,)
    adam_m = np.zeros(theta.shape, np.float64)
    adam_v = np.zeros(theta.shape, np.float64)
    train_features = PARENT.head_features(train["s"], train["xb"], train["xq"], train["y"][:, :2], mean, std)
    train_target = train["onset"].astype(np.float32)
    train_hazard = hazardous(train["y"][:, :2]).astype(np.float32)
    if np.any(train_target > train_hazard):
        raise RuntimeError("native onset outside hazardous support")

    def supported_loss(parameters, features, target, support):
        logits = PARENT.head_logits(parameters, features)
        return jnp.mean(support * (jax.nn.softplus(logits) - target * logits))

    value_grad = jax.jit(jax.value_and_grad(supported_loss))
    ledger.heads(CONFIG["head_updates"] * CONFIG["head_batch_size"], f"head_training_s{seed}")
    validation_points = CONFIG["head_updates"] // CONFIG["head_validation_frequency"] + 1
    ledger.heads(validation_points * len(validation["s"]), f"head_validation_s{seed}")
    ledger.updates(CONFIG["head_updates"], f"head_training_s{seed}")
    history = []
    validation_history = []
    validation_theta = []
    best_score = np.inf
    best_step = None
    best_theta = None
    best_probability = None

    def validate(step):
        nonlocal best_score, best_step, best_theta, best_probability
        metrics, probability, _, _ = head_validation(theta, validation, mean, std)
        metrics["step"] = step
        validation_history.append(metrics)
        validation_theta.append(theta.copy())
        if metrics["selection_score"] < best_score:
            best_score = metrics["selection_score"]
            best_step = step
            best_theta = theta.copy()
            best_probability = probability.copy()

    validate(0)
    for step in range(CONFIG["head_updates"]):
        rng = np.random.default_rng(CONFIG["head_batch_seed_base"] + seed * 100000 + step)
        ids = rng.choice(len(train_features), CONFIG["head_batch_size"])
        loss, gradient = value_grad(
            jnp.asarray(theta),
            jnp.asarray(train_features[ids]),
            jnp.asarray(train_target[ids]),
            jnp.asarray(train_hazard[ids]),
        )
        gradient = np.asarray(gradient, np.float64)
        adam_m = CONFIG["adam_beta1"] * adam_m + (1 - CONFIG["adam_beta1"]) * gradient
        adam_v = CONFIG["adam_beta2"] * adam_v + (1 - CONFIG["adam_beta2"]) * gradient**2
        age = step + 1
        update = CONFIG["head_learning_rate"] * (adam_m / (1 - CONFIG["adam_beta1"]**age)) / (
            np.sqrt(adam_v / (1 - CONFIG["adam_beta2"]**age)) + CONFIG["adam_epsilon"]
        )
        theta = (theta - update).astype(np.float32)
        if not np.isfinite(theta).all():
            raise FloatingPointError("non-finite head parameters")
        history.append({
            "step": age,
            "batch_supported_fraction": float(train_hazard[ids].mean()),
            "batch_onset_fraction": float(train_target[ids].mean()),
            "supported_bce_scaled_by_all_rows": float(loss),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "update_norm": float(np.linalg.norm(update)),
        })
        if age % CONFIG["head_validation_frequency"] == 0:
            validate(age)
            if age % 250 == 0:
                print(f"head seed {seed}: {age}/{CONFIG['head_updates']}", flush=True)
    assert best_theta is not None
    np.savez_compressed(
        checkpoint_path,
        theta=best_theta,
        initialization_theta=old_head(seed),
        selected_step=np.asarray(best_step),
        validation_theta=np.asarray(validation_theta),
        adam_m=adam_m,
        adam_v=adam_v,
    )
    np.savez_compressed(
        OUT / f"head_validation_selected_s{seed}.npz",
        probability=best_probability,
        target=validation["onset"],
        actual_hazard=hazardous(validation["y"][:, :2]),
    )
    write_json(OUT / f"head_training_s{seed}.json", {
        "train_alive_rows": len(train["s"]),
        "train_onsets": int(train_target.sum()),
        "train_hazardous_landings": int(train_hazard.sum()),
        "updates": CONFIG["head_updates"],
        "optimizer_state_reset": True,
        "all_1569_parameters_trainable": True,
        "uniform_alive_row_sampling": True,
        "history": history,
    })
    write_json(OUT / f"head_validation_s{seed}.json", {
        "selection_rule": CONFIG["head_validation_selection"],
        "selected_step": best_step,
        "selected_score": best_score,
        "history": validation_history,
    })
    return best_theta


def train_all(ledger):
    train = alive_rows("train")
    validation = alive_rows("validation")
    summary = {
        "train_alive_rows": len(train["s"]),
        "validation_alive_rows": len(validation["s"]),
        "train_diagonal_rows": int(exact_diagonal(train).sum()),
        "validation_diagonal_rows": int(exact_diagonal(validation).sum()),
        "train_onsets": int(train["onset"].sum()),
        "validation_onsets": int(validation["onset"].sum()),
        "train_prefix_episode_counts": {
            "expert": int(len(np.unique(train["episode"][train["prefix_source"] == 0]))),
            "actor": int(len(np.unique(train["episode"][train["prefix_source"] == 1]))),
        },
        "validation_prefix_episode_counts": {
            "expert": int(len(np.unique(validation["episode"][validation["prefix_source"] == 0]))),
            "actor": int(len(np.unique(validation["episode"][validation["prefix_source"] == 1]))),
        },
    }
    write_json(OUT / "training_data_summary.json", summary)
    positions = [train_position(seed, train, validation, ledger) for seed in (0, 1)]
    heads = [train_head(seed, train, validation, ledger) for seed in (0, 1)]
    write_json(OUT / "checkpoints_frozen.json", {
        "status": "frozen_before_final_collection",
        "position_selected_steps": [int(load_npz(OUT / f"position_repaired_s{s}.npz")["selected_step"]) for s in (0, 1)],
        "head_selected_steps": [int(load_npz(OUT / f"head_repaired_s{s}.npz")["selected_step"]) for s in (0, 1)],
        "final_outcomes_inspected": False,
    })
    return positions, heads


if __name__ == "__main__":
    train_all(Ledger())
