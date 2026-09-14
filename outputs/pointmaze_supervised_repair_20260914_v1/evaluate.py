"""Frozen final matched and complete-trajectory evaluation for the repair factorial."""
from __future__ import annotations

from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from common import (
    CONFIG,
    OUT,
    PARENT,
    RESPONSE_MODULE,
    Ledger,
    binary_summary,
    bootstrap_weights,
    exact_diagonal,
    full_energy_score,
    hazardous,
    input_groups,
    load_head_scaling,
    load_npz,
    old_head,
    old_position,
    paired_ratio_difference,
    paired_row_difference,
    paired_scalar_difference,
    ratio_summary,
    row_mean_summary,
    scalar_summary,
    write_json,
)


def final_alive_rows():
    loaded = load_npz(OUT / "final_paired.npz")
    mask = ~loaded["dead_before"]
    names = ("episode", "time", "prefix_source", "s", "xb", "xq", "y", "onset", "reward", "query_type")
    return loaded, {name: loaded[name][mask] for name in names}


def head_theta(seed, variant):
    if variant == "old":
        return old_head(seed)
    return load_npz(OUT / f"head_repaired_s{seed}.npz")["theta"]


def position_theta(seed, variant):
    if variant == "old":
        return old_position(seed)
    return load_npz(OUT / f"position_repaired_s{seed}.npz")["theta"]


def evaluation_groups(data):
    groups = input_groups(data["s"])
    groups["query_shadow_diagonal"] = data["query_type"] == 0
    groups["query_frozen_actor"] = data["query_type"] == 1
    groups["query_negative_shadow"] = data["query_type"] == 2
    groups["query_uniform_random"] = data["query_type"] == 3
    groups["exact_diagonal"] = exact_diagonal(data)
    groups["off_diagonal"] = ~exact_diagonal(data)
    return groups


def evaluate_actual_heads(data, groups, mean, std, weights, ledger):
    arrays = {}
    metrics = {}
    actual_hazard = hazardous(data["y"][:, :2])
    masks = OrderedDict(groups)
    masks["actual_hazardous_landing"] = actual_hazard
    masks["actual_nonhazardous_landing"] = ~actual_hazard
    features = PARENT.head_features(data["s"], data["xb"], data["xq"], data["y"][:, :2], mean, std)
    for seed in (0, 1):
        metrics[f"s{seed}"] = {}
        probabilities = {}
        for variant in ("old", "repaired"):
            raw = PARENT.predict_numpy(head_theta(seed, variant), features)
            probability = raw * actual_hazard
            ledger.heads(len(probability), f"final_actual_head_s{seed}_{variant}")
            arrays[f"probability_s{seed}_{variant}"] = probability.astype(np.float32)
            probabilities[variant] = probability
            metrics[f"s{seed}"][variant] = {
                name: binary_summary(probability, data["onset"], data["episode"], mask, weights)
                for name, mask in masks.items()
            }
        old = probabilities["old"]
        repaired = probabilities["repaired"]
        clipped_old = np.clip(old, 1e-12, 1 - 1e-12)
        clipped_repaired = np.clip(repaired, 1e-12, 1 - 1e-12)
        target = data["onset"].astype(float)
        old_log = -(target * np.log(clipped_old) + (1 - target) * np.log1p(-clipped_old))
        repaired_log = -(target * np.log(clipped_repaired) + (1 - target) * np.log1p(-clipped_repaired))
        contrasts = {}
        for name, mask in masks.items():
            contrasts[name] = {
                "prediction_repaired_minus_old": paired_row_difference(repaired, old, data["episode"], mask, weights),
                "brier_repaired_minus_old": paired_row_difference((repaired - target) ** 2, (old - target) ** 2, data["episode"], mask, weights),
                "log_loss_repaired_minus_old": paired_row_difference(repaired_log, old_log, data["episode"], mask, weights),
            }
        metrics[f"s{seed}"]["contrast"] = contrasts
    np.savez_compressed(OUT / "final_actual_head_predictions.npz", target=data["onset"], actual_hazard=actual_hazard, **arrays)
    write_json(OUT / "final_actual_head_metrics.json", metrics)
    return arrays, metrics


def sample_final_positions(data, ledger):
    engine = PARENT.load_position_engine()
    count = CONFIG["final_position_samples"]
    saved = {}
    for seed in (0, 1):
        for variant in ("old", "repaired"):
            name = f"s{seed}_{variant}"
            path = OUT / f"final_position_{name}.npz"
            if path.exists():
                saved[name] = load_npz(path)
                continue
            ledger.position(len(data["s"]) * count, f"final_matched_position_{name}")
            parts = {key: [] for key in ("samples_xy", "atom", "gate", "projection")}
            theta = position_theta(seed, variant)
            for start in range(0, len(data["s"]), 128):
                stop = min(start + 128, len(data["s"]))
                output, detail = engine.candidate_sample(
                    jnp.asarray(theta),
                    jnp.asarray(data["s"][start:stop]),
                    jnp.asarray(data["xq"][start:stop]),
                    jnp.asarray(data["xb"][start:stop]),
                    jax.random.PRNGKey(CONFIG["final_position_seed_base"] + seed * 100000 + start),
                    count,
                )
                output = np.asarray(output)
                parts["samples_xy"].append(output[..., :2])
                parts["atom"].append(np.asarray(detail["stationary_atom"]))
                parts["gate"].append(np.asarray(detail["h"]))
                parts["projection"].append(np.asarray(detail["projection_corrected"]))
                np.testing.assert_array_equal(
                    output[..., 2:],
                    np.broadcast_to(data["s"][start:stop, None, :6], output[..., 2:].shape),
                )
            result = {key: np.concatenate(value) for key, value in parts.items()}
            result["legal"] = RESPONSE_MODULE.native_legal(result["samples_xy"])
            if not result["legal"].all():
                raise RuntimeError("illegal final position sample")
            np.savez_compressed(path, **result)
            saved[name] = result
    return saved


def position_model_metrics(samples, data, groups, weights):
    current = data["s"][:, :2]
    actual = data["y"][:, :2]
    delta = samples - current[:, None]
    actual_delta = actual - current
    sample_hazard = hazardous(samples)
    actual_hazard = hazardous(actual)
    current_hazard = hazardous(current)
    score = full_energy_score(samples, actual)
    event_probability = {
        "hazard_landing": sample_hazard.mean(axis=1),
        "hazard_entry": sample_hazard.mean(axis=1),
        "hazard_residence": sample_hazard.mean(axis=1),
        "hazard_exit": (~sample_hazard).mean(axis=1),
        "rightward_progress": (delta[..., 0] > CONFIG["stationary_threshold"]).mean(axis=1),
        "stationary": (np.linalg.norm(delta, axis=-1) <= CONFIG["stationary_threshold"]).mean(axis=1),
    }
    targets = {
        "hazard_landing": actual_hazard,
        "hazard_entry": actual_hazard,
        "hazard_residence": actual_hazard,
        "hazard_exit": ~actual_hazard,
        "rightward_progress": actual_delta[:, 0] > CONFIG["stationary_threshold"],
        "stationary": np.linalg.norm(actual_delta, axis=1) <= CONFIG["stationary_threshold"],
    }
    eligibility = {
        "hazard_landing": np.ones(len(actual), bool),
        "hazard_entry": ~current_hazard,
        "hazard_residence": current_hazard,
        "hazard_exit": current_hazard,
        "rightward_progress": np.ones(len(actual), bool),
        "stationary": np.ones(len(actual), bool),
    }
    result = {"groups": {}}
    for group_name, group_mask in groups.items():
        entry = {
            "energy_score": row_mean_summary(score, data["episode"], group_mask, weights),
            "mean_xy_error": row_mean_summary(np.linalg.norm(samples.mean(axis=1) - actual, axis=1), data["episode"], group_mask, weights),
            "horizontal_displacement": {
                "actual": row_mean_summary(actual_delta[:, 0], data["episode"], group_mask, weights),
                "predicted": row_mean_summary(delta[..., 0].mean(axis=1), data["episode"], group_mask, weights),
                "bias": row_mean_summary(delta[..., 0].mean(axis=1) - actual_delta[:, 0], data["episode"], group_mask, weights),
            },
        }
        for event in event_probability:
            mask = group_mask & eligibility[event]
            entry[event] = binary_summary(event_probability[event], targets[event], data["episode"], mask, weights)
        result["groups"][group_name] = entry
    result["raw"] = {"energy": score, "mean_xy_error": np.linalg.norm(samples.mean(axis=1) - actual, axis=1), "dx_prediction": delta[..., 0].mean(axis=1), "events": event_probability, "targets": targets, "eligibility": eligibility}
    return result


def evaluate_positions(data, groups, samples, weights):
    metrics = {}
    internal = {}
    for seed in (0, 1):
        metrics[f"s{seed}"] = {}
        old = position_model_metrics(samples[f"s{seed}_old"]["samples_xy"], data, groups, weights)
        repaired = position_model_metrics(samples[f"s{seed}_repaired"]["samples_xy"], data, groups, weights)
        internal[seed] = {"old": old, "repaired": repaired}
        for variant, values in (("old", old), ("repaired", repaired)):
            metrics[f"s{seed}"][variant] = {"groups": values["groups"]}
        contrasts = {}
        for group_name, mask in groups.items():
            entry = {
                "energy_repaired_minus_old": paired_row_difference(repaired["raw"]["energy"], old["raw"]["energy"], data["episode"], mask, weights),
                "mean_xy_error_repaired_minus_old": paired_row_difference(repaired["raw"]["mean_xy_error"], old["raw"]["mean_xy_error"], data["episode"], mask, weights),
                "dx_prediction_repaired_minus_old": paired_row_difference(repaired["raw"]["dx_prediction"], old["raw"]["dx_prediction"], data["episode"], mask, weights),
            }
            for event in old["raw"]["events"]:
                event_mask = mask & old["raw"]["eligibility"][event]
                target = old["raw"]["targets"][event].astype(float)
                entry[event] = {
                    "probability_repaired_minus_old": paired_row_difference(repaired["raw"]["events"][event], old["raw"]["events"][event], data["episode"], event_mask, weights),
                    "brier_repaired_minus_old": paired_row_difference((repaired["raw"]["events"][event] - target) ** 2, (old["raw"]["events"][event] - target) ** 2, data["episode"], event_mask, weights),
                }
            contrasts[group_name] = entry
        metrics[f"s{seed}"]["contrast"] = contrasts
    write_json(OUT / "final_position_metrics.json", metrics)
    return internal, metrics


def evaluate_joint(data, groups, samples, mean, std, weights, ledger):
    arrays = {}
    metrics = {}
    target = data["onset"].astype(float)
    for seed in (0, 1):
        metrics[f"s{seed}"] = {"arms": {}}
        raw_arm = {}
        for position_variant in ("old", "repaired"):
            xy = samples[f"s{seed}_{position_variant}"]["samples_xy"]
            support = hazardous(xy)
            features = PARENT.head_features(data["s"], data["xb"], data["xq"], xy, mean, std)
            for head_variant in ("old", "repaired"):
                name = f"p{position_variant}_h{head_variant}"
                probability = PARENT.predict_numpy(head_theta(seed, head_variant), features) * support
                ledger.heads(probability.size, f"final_joint_s{seed}_{name}")
                q = probability.mean(axis=1)
                q_hazard = (probability * support).mean(axis=1)
                q_nonhazard = (probability * ~support).mean(axis=1)
                arrays[f"sample_probability_s{seed}_{name}"] = probability.astype(np.float32)
                arrays[f"q_s{seed}_{name}"] = q.astype(np.float32)
                raw_arm[name] = q
                metrics[f"s{seed}"]["arms"][name] = {
                    group_name: {
                        "onset": binary_summary(q, target, data["episode"], mask, weights),
                        "hazard_component": row_mean_summary(q_hazard, data["episode"], mask, weights),
                        "nonhazard_component": row_mean_summary(q_nonhazard, data["episode"], mask, weights),
                        "generated_hazard_probability": row_mean_summary(support.mean(axis=1), data["episode"], mask, weights),
                    }
                    for group_name, mask in groups.items()
                }
                assert np.max(np.abs(q_nonhazard)) == 0
        contrast = {}
        losses = {
            name: {
                "brier": (q - target) ** 2,
                "log": -(target * np.log(np.clip(q, 1e-12, 1 - 1e-12)) + (1 - target) * np.log1p(-np.clip(q, 1e-12, 1 - 1e-12))),
            }
            for name, q in raw_arm.items()
        }
        pairs = {
            "head_repair_at_old_position": ("pold_hrepaired", "pold_hold"),
            "head_repair_at_repaired_position": ("prepaired_hrepaired", "prepaired_hold"),
            "position_repair_at_old_head": ("prepaired_hold", "pold_hold"),
            "position_repair_at_repaired_head": ("prepaired_hrepaired", "pold_hrepaired"),
            "combined_repair": ("prepaired_hrepaired", "pold_hold"),
        }
        for label, (left, right) in pairs.items():
            contrast[label] = {}
            for group_name, mask in groups.items():
                contrast[label][group_name] = {
                    "probability_difference": paired_row_difference(raw_arm[left], raw_arm[right], data["episode"], mask, weights),
                    "brier_difference": paired_row_difference(losses[left]["brier"], losses[right]["brier"], data["episode"], mask, weights),
                    "log_loss_difference": paired_row_difference(losses[left]["log"], losses[right]["log"], data["episode"], mask, weights),
                }
        interaction_probability = raw_arm["prepaired_hrepaired"] - raw_arm["prepaired_hold"] - raw_arm["pold_hrepaired"] + raw_arm["pold_hold"]
        for group_name, mask in groups.items():
            contrast.setdefault("factorial_interaction", {})[group_name] = {
                "probability_difference_in_differences": row_mean_summary(interaction_probability, data["episode"], mask, weights),
                "brier_difference_in_differences": row_mean_summary(
                    losses["prepaired_hrepaired"]["brier"] - losses["prepaired_hold"]["brier"] - losses["pold_hrepaired"]["brier"] + losses["pold_hold"]["brier"],
                    data["episode"], mask, weights,
                ),
            }
        metrics[f"s{seed}"]["contrasts"] = contrast
    np.savez_compressed(OUT / "final_joint_predictions.npz", target=target, **arrays)
    write_json(OUT / "final_joint_metrics.json", metrics)
    return metrics


def make_rollout(engine, mean, std):
    nominal, actor = engine.kernel.nominal, engine.kernel.actor
    mean_j = jnp.asarray(mean, jnp.float32)
    std_j = jnp.asarray(std, jnp.float32)

    def generate(position_parameters, head_parameters, states, goals, key):
        def step(carry, step_key):
            state, failed = carry
            nominal_key, actor_key, position_key, onset_key = jax.random.split(step_key, 4)
            xb = nominal.sample(state, nominal_key, 1, goal=goals)
            action = actor(state, goals, actor_key)
            proposed, detail = engine.candidate_sample(position_parameters, state, action, xb, position_key, 1)
            proposed = proposed[:, 0]
            features = jnp.concatenate([(state - mean_j) / std_j, xb, action, proposed[:, :2] - state[:, :2]], axis=-1)
            support = PARENT.hazardous_jax(proposed[:, :2])
            raw_probability = jax.nn.sigmoid(PARENT.head_logits(head_parameters, features))
            probability = jnp.where(failed, 0.0, jnp.where(support, raw_probability, 0.0))
            onset = (~failed) & (jax.random.uniform(onset_key, probability.shape) < probability)
            absorbed = jnp.concatenate([state[:, :2], state[:, :6]], axis=-1)
            following = jnp.where(failed[:, None], absorbed, proposed)
            next_failed = failed | onset
            reward = ((jnp.linalg.norm(following[:, :2] - goals[:, :2], axis=-1) < 2) & (~next_failed)).astype(jnp.float32)
            record = {
                "next_state": following,
                "position_proposal_xy": proposed[:, :2],
                "action": action,
                "xb": xb,
                "failed_before": failed,
                "failed_after": next_failed,
                "onset": onset,
                "failure_probability": probability,
                "hazard_landing": support,
                "reward": reward,
                "atom": detail["stationary_atom"][:, 0],
                "gate": detail["h"][:, 0],
            }
            return (following, next_failed), record

        initial_failed = jnp.zeros(states.shape[0], bool)
        (_, _), record = jax.lax.scan(step, (states, initial_failed), jax.random.split(key, CONFIG["horizon"]))
        record = {name: jnp.swapaxes(value, 0, 1) for name, value in record.items()}
        record["states"] = jnp.concatenate([states[:, None], record.pop("next_state")], axis=1)
        record["failed"] = jnp.concatenate([initial_failed[:, None], record["failed_after"]], axis=1)
        record["return"] = jnp.sum(record["reward"] * jnp.power(jnp.float32(CONFIG["discount"]), jnp.arange(CONFIG["horizon"])), axis=-1)
        return record

    return jax.jit(generate)


def first_event_components(event, positions):
    event = np.asarray(event, bool)
    positions = np.asarray(positions)
    entered = event.any(axis=-1)
    first = np.argmax(event, axis=-1)
    index = np.broadcast_to(first[..., None, None], positions.shape[:-2] + (1, 2))
    xy = np.take_along_axis(positions, index, axis=-2)[..., 0, :]
    xy = np.where(entered[..., None], xy, np.nan)
    if event.ndim == 2:
        event = event[:, None]
        entered = entered[:, None]
        first = first[:, None]
        xy = xy[:, None]
    paths = entered.shape[1]
    return {
        "probability": entered.mean(axis=1),
        "denominator": entered.sum(axis=1) / paths,
        "time_num": np.where(entered, first, 0).sum(axis=1) / paths,
        "x_num": np.where(entered, xy[..., 0], 0).sum(axis=1) / paths,
        "y_num": np.where(entered, xy[..., 1], 0).sum(axis=1) / paths,
    }


def trajectory_components(record, native=False):
    if native:
        reward = record["prefix_rewards"]
        failed_before = record["prefix_dead_before"]
        failed_after = record["prefix_dead_after"]
        onset = record["prefix_onset"]
        states = record["prefix_states"]
        positions = states[:, 1:, :2]
        hazard = hazardous(positions)
        returns = np.sum(reward * CONFIG["discount"] ** np.arange(CONFIG["horizon"]), axis=1)
        reward_any = reward.any(axis=1)
        failure_any = failed_after.any(axis=1)
        survival_no_reward = (~failure_any) & (~reward_any)
        onset_components = first_event_components(onset, positions)
        first_hazard = first_event_components(hazard, positions)
        at_risk = (hazard & ~failed_before).sum(axis=1)
        return {
            "reward_probability": reward_any.astype(float),
            "failure_probability": failure_any.astype(float),
            "survival_without_reward": survival_no_reward.astype(float),
            "return": returns,
            "reward_return_num": returns * reward_any,
            "reward_den": reward_any.astype(float),
            "onset": onset_components,
            "first_hazard": first_hazard,
            "at_risk_exposure": at_risk.astype(float),
        }
    reward = record["reward"]
    failed_before = record["failed_before"]
    failed_after = record["failed_after"]
    onset = record["onset"]
    positions = record["position_proposal_xy"]
    hazard = hazardous(positions) & ~failed_before
    returns = record["return"]
    reward_any = reward.any(axis=-1)
    failure_any = failed_after.any(axis=-1)
    survival_no_reward = (~failure_any) & (~reward_any)
    onset_components = first_event_components(onset, positions)
    first_hazard = first_event_components(hazard, positions)
    return {
        "reward_probability": reward_any.mean(axis=1),
        "failure_probability": failure_any.mean(axis=1),
        "survival_without_reward": survival_no_reward.mean(axis=1),
        "return": returns.mean(axis=1),
        "reward_return_num": (returns * reward_any).mean(axis=1),
        "reward_den": reward_any.mean(axis=1),
        "onset": onset_components,
        "first_hazard": first_hazard,
        "at_risk_exposure": hazard.sum(axis=-1).mean(axis=1),
    }


def summarize_trajectory(component, weights):
    return {
        "reward_occurrence": scalar_summary(component["reward_probability"], weights),
        "failure_probability": scalar_summary(component["failure_probability"], weights),
        "survival_without_reward": scalar_summary(component["survival_without_reward"], weights),
        "mean_discounted_return": scalar_summary(component["return"], weights),
        "return_conditional_on_reward": ratio_summary(component["reward_return_num"], component["reward_den"], weights),
        "onset_time_zero_based_conditional_on_failure": ratio_summary(component["onset"]["time_num"], component["onset"]["denominator"], weights),
        "first_hazard_probability": scalar_summary(component["first_hazard"]["probability"], weights),
        "first_hazard_time_zero_based_conditional": ratio_summary(component["first_hazard"]["time_num"], component["first_hazard"]["denominator"], weights),
        "first_hazard_mean_xy": {
            "x": ratio_summary(component["first_hazard"]["x_num"], component["first_hazard"]["denominator"], weights),
            "y": ratio_summary(component["first_hazard"]["y_num"], component["first_hazard"]["denominator"], weights),
        },
        "at_risk_hazardous_opportunities": scalar_summary(component["at_risk_exposure"], weights),
    }


def compare_trajectory(left, right, weights):
    result = {
        name: paired_scalar_difference(left[name], right[name], weights)
        for name in ("reward_probability", "failure_probability", "survival_without_reward", "return", "at_risk_exposure")
    }
    result["return_conditional_on_reward"] = paired_ratio_difference(
        left["reward_return_num"], left["reward_den"], right["reward_return_num"], right["reward_den"], weights
    )
    result["onset_time_conditional"] = paired_ratio_difference(
        left["onset"]["time_num"], left["onset"]["denominator"], right["onset"]["time_num"], right["onset"]["denominator"], weights
    )
    result["first_hazard_time_conditional"] = paired_ratio_difference(
        left["first_hazard"]["time_num"], left["first_hazard"]["denominator"], right["first_hazard"]["time_num"], right["first_hazard"]["denominator"], weights
    )
    return result


def evaluate_rollouts(full_final, mean, std, weights, ledger):
    native_component = trajectory_components(full_final, native=True)
    engine = PARENT.load_position_engine()
    rollout = make_rollout(engine, mean, std)
    roots = np.repeat(full_final["prefix_states"][:, 0], CONFIG["rollout_paths_per_native_root"], axis=0)
    goals = np.broadcast_to(PARENT.GOAL, roots.shape)
    records = {}
    components = {}
    summaries = {"native": summarize_trajectory(native_component, weights), "arms": {}}
    for seed in (0, 1):
        summaries["arms"][f"s{seed}"] = {}
        for position_variant in ("old", "repaired"):
            for head_variant in ("old", "repaired"):
                name = f"s{seed}_p{position_variant}_h{head_variant}"
                path = OUT / f"rollout_{name}.npz"
                if path.exists():
                    record = load_npz(path)
                else:
                    ledger.position(len(roots) * CONFIG["horizon"], f"complete_rollout_{name}")
                    ledger.paths(len(roots), f"complete_rollout_{name}")
                    generated = rollout(
                        jnp.asarray(position_theta(seed, position_variant)),
                        jnp.asarray(head_theta(seed, head_variant)),
                        jnp.asarray(roots),
                        jnp.asarray(goals),
                        jax.random.PRNGKey(CONFIG["rollout_seed"]),
                    )
                    record = {
                        key: np.asarray(value).reshape(
                            (CONFIG["splits"]["final"]["episodes"], CONFIG["rollout_paths_per_native_root"]) + value.shape[1:]
                        )
                        for key, value in generated.items()
                    }
                    np.savez_compressed(path, **record)
                np.testing.assert_array_equal(record["states"][:, :, 1:, 2:], record["states"][:, :, :-1, :6])
                if not RESPONSE_MODULE.native_legal(record["states"][:, :, 1:, :2]).all():
                    raise RuntimeError("illegal rollout state")
                assert not (record["failure_probability"] * (~record["hazard_landing"])).any()
                assert np.all(record["failed_after"] >= record["failed_before"])
                before_failed = record["failed_before"]
                current_xy = record["states"][:, :, :-1, :2]
                next_xy = record["states"][:, :, 1:, :2]
                np.testing.assert_array_equal(next_xy[before_failed], current_xy[before_failed])
                onset = record["onset"]
                np.testing.assert_array_equal(next_xy[onset], record["position_proposal_xy"][onset])
                component = trajectory_components(record)
                records[name] = record
                components[name] = component
                summaries["arms"][f"s{seed}"][f"p{position_variant}_h{head_variant}"] = {
                    "metrics": summarize_trajectory(component, weights),
                    "difference_from_native": compare_trajectory(component, native_component, weights),
                }
    contrasts = {}
    simple = ("reward_probability", "failure_probability", "survival_without_reward", "return", "at_risk_exposure")
    for seed in (0, 1):
        prefix = f"s{seed}_"
        arm = {
            "old_old": components[prefix + "pold_hold"],
            "old_repaired_head": components[prefix + "pold_hrepaired"],
            "repaired_position_old": components[prefix + "prepaired_hold"],
            "repaired_repaired": components[prefix + "prepaired_hrepaired"],
        }
        contrasts[f"s{seed}"] = {
            "head_repair_at_old_position": compare_trajectory(arm["old_repaired_head"], arm["old_old"], weights),
            "head_repair_at_repaired_position": compare_trajectory(arm["repaired_repaired"], arm["repaired_position_old"], weights),
            "position_repair_at_old_head": compare_trajectory(arm["repaired_position_old"], arm["old_old"], weights),
            "position_repair_at_repaired_head": compare_trajectory(arm["repaired_repaired"], arm["old_repaired_head"], weights),
            "combined_repair": compare_trajectory(arm["repaired_repaired"], arm["old_old"], weights),
            "factorial_interaction": {
                name: scalar_summary(
                    arm["repaired_repaired"][name] - arm["repaired_position_old"][name] - arm["old_repaired_head"][name] + arm["old_old"][name], weights
                )
                for name in simple
            },
        }
    summaries["factorial_contrasts"] = contrasts
    write_json(OUT / "trajectory_metrics.json", summaries)
    return native_component, components, summaries


def acceptance(actual_metrics, position_internal, trajectory_summary):
    result = {"head": {}, "position": {}, "combined": {}}
    native = trajectory_summary["native"]
    for seed in (0, 1):
        head = actual_metrics[f"s{seed}"]
        head_checks = {}
        for group in ("all", "actual_hazardous_landing"):
            old = head["old"][group]
            repaired = head["repaired"][group]
            head_checks[group] = {
                "brier_improved": repaired["brier"]["estimate"] < old["brier"]["estimate"],
                "absolute_bias_improved": abs(repaired["calibration_bias"]["estimate"]) < abs(old["calibration_bias"]["estimate"]),
            }
        result["head"][f"s{seed}"] = {"checks": head_checks, "passed": all(all(v.values()) for v in head_checks.values())}

        old = position_internal[seed]["old"]
        repaired = position_internal[seed]["repaired"]
        def energy(group, variant):
            return variant["groups"][group]["energy_score"]["estimate"]
        old_res = old["groups"]["all"]["hazard_residence"]["calibration_bias"]["estimate"]
        new_res = repaired["groups"]["all"]["hazard_residence"]["calibration_bias"]["estimate"]
        old_dx = old["groups"]["inside_hazard"]["horizontal_displacement"]["bias"]["estimate"]
        new_dx = repaired["groups"]["inside_hazard"]["horizontal_displacement"]["bias"]["estimate"]
        old_entry = old["groups"]["all"]["hazard_entry"]["calibration_bias"]["estimate"]
        new_entry = repaired["groups"]["all"]["hazard_entry"]["calibration_bias"]["estimate"]
        position_checks = {
            "diagonal_energy_improved": energy("exact_diagonal", repaired) < energy("exact_diagonal", old),
            "off_diagonal_energy_improved": energy("off_diagonal", repaired) < energy("off_diagonal", old),
            "residence_absolute_bias_improved": abs(new_res) < abs(old_res),
            "inside_hazard_dx_absolute_bias_improved": abs(new_dx) < abs(old_dx),
            "entry_absolute_bias_not_worse_by_over_0p03": abs(new_entry) <= abs(old_entry) + 0.03,
        }
        result["position"][f"s{seed}"] = {"checks": position_checks, "passed": all(position_checks.values())}

        old_arm = trajectory_summary["arms"][f"s{seed}"]["pold_hold"]["metrics"]
        new_arm = trajectory_summary["arms"][f"s{seed}"]["prepaired_hrepaired"]["metrics"]
        reward_native = native["reward_occurrence"]["estimate"]
        failure_native = native["failure_probability"]["estimate"]
        combined_checks = {
            "reward_occurrence_absolute_gap_improved": abs(new_arm["reward_occurrence"]["estimate"] - reward_native) < abs(old_arm["reward_occurrence"]["estimate"] - reward_native),
            "failure_absolute_gap_improved": abs(new_arm["failure_probability"]["estimate"] - failure_native) < abs(old_arm["failure_probability"]["estimate"] - failure_native),
            "head_component_passed": result["head"][f"s{seed}"]["passed"],
            "position_component_passed": result["position"][f"s{seed}"]["passed"],
        }
        result["combined"][f"s{seed}"] = {"checks": combined_checks, "passed": all(combined_checks.values()), "return_not_an_acceptance_input": True}
    result["overall"] = {
        "head_passed_both_seeds": all(result["head"][f"s{s}"]["passed"] for s in (0, 1)),
        "position_passed_both_seeds": all(result["position"][f"s{s}"]["passed"] for s in (0, 1)),
        "combined_passed_both_seeds": all(result["combined"][f"s{s}"]["passed"] for s in (0, 1)),
    }
    write_json(OUT / "acceptance.json", result)
    return result


def evaluate_all(ledger):
    full_final, data = final_alive_rows()
    weights = bootstrap_weights()
    groups = evaluation_groups(data)
    np.savez_compressed(OUT / "final_groups.npz", episode=data["episode"], **groups)
    mean, std = load_head_scaling()
    _, actual_metrics = evaluate_actual_heads(data, groups, mean, std, weights, ledger)
    samples = sample_final_positions(data, ledger)
    position_internal, position_metrics = evaluate_positions(data, groups, samples, weights)
    joint_metrics = evaluate_joint(data, groups, samples, mean, std, weights, ledger)
    _, _, trajectory_summary = evaluate_rollouts(full_final, mean, std, weights, ledger)
    accepted = acceptance(actual_metrics, position_internal, trajectory_summary)
    scope = {
        "final_episodes": CONFIG["splits"]["final"]["episodes"],
        "final_rows": len(full_final["episode"]),
        "alive_rows": len(data["episode"]),
        "alive_onsets": int(data["onset"].sum()),
        "alive_actual_hazardous_landings": int(hazardous(data["y"][:, :2]).sum()),
        "input_group_counts": {name: {"rows": int(mask.sum()), "episodes": int(len(np.unique(data["episode"][mask])))} for name, mask in groups.items()},
        "all_arms_use_hazard_support": True,
        "position_samples_per_context": CONFIG["final_position_samples"],
        "model_paths_per_native_root": CONFIG["rollout_paths_per_native_root"],
    }
    write_json(OUT / "final_scope.json", scope)
    return {"scope": scope, "actual": actual_metrics, "position": position_metrics, "joint": joint_metrics, "trajectory": trajectory_summary, "acceptance": accepted}


if __name__ == "__main__":
    evaluate_all(Ledger())
