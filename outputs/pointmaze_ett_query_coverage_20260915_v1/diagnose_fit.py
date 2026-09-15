"""Read-only post-decision audit of goal support and final critic fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
CONFIG = json.loads((OUT / "config.json").read_text())
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
sys.path.insert(0, str(ROOT))

from crl import checkpoint, networks  # noqa: E402


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


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


def make_network():
    return networks.make_networks(
        8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
        hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
        use_image_obs=False, use_layer_norm=False, obs_scale=None,
    )


def scores(network, params, states, actions):
    goal = np.broadcast_to(GOAL, states.shape)
    observation = jnp.asarray(np.concatenate([states, goal], axis=1))
    output = []
    for action in actions:
        batch_action = jnp.asarray(np.broadcast_to(action, (len(states), 2)))
        phi, psi = network.representation_network.apply(params, observation, batch_action)
        value = jnp.sum(phi * psi, axis=1)
        if value.ndim == 2:
            value = value[:, 0]
        output.append(np.asarray(value, np.float32))
    return np.stack(output, axis=1)


def margin_summary(score, bootstrap):
    central = score[:, 1] - score[:, 4]
    grouped = score[:, :3].mean(axis=1) - score[:, 3:].mean(axis=1)
    return {
        "central_down_minus_right": float(central.mean()),
        "central_ci95": np.quantile(central[bootstrap].mean(axis=1), [0.025, 0.975]),
        "central_roots_positive": int((central > 0).sum()),
        "grouped_down_minus_right": float(grouped.mean()),
        "grouped_ci95": np.quantile(grouped[bootstrap].mean(axis=1), [0.025, 0.975]),
        "grouped_roots_positive": int((grouped > 0).sum()),
        "candidate_logit_means": score.mean(axis=0),
    }


def replay_batch_fit(network, q_params, replay, lineage, update_ids):
    @jax.jit
    def logits_fn(observation, action):
        return network.q_network.apply(q_params, observation, action)

    candidate_names = CONFIG["query_actions"]["names"]
    all_stats = []
    t0_stats = []
    candidate_stats = {name: [] for name in candidate_names}
    for update in update_ids:
        trajectory = lineage["trajectory"][update]
        anchor = lineage["anchor_time"][update]
        future = lineage["future_time"][update]
        state = replay["obs"][trajectory, anchor, :8]
        goal = replay["obs"][trajectory, future, :8]
        observation = np.concatenate([state, goal], axis=1).astype(np.float32)
        action = replay["act"][trajectory, anchor].astype(np.float32)
        logits = np.asarray(logits_fn(jnp.asarray(observation), jnp.asarray(action)))
        diagonal = np.diag(logits)
        negative_logit_mean = (logits.sum(axis=1) - diagonal) / (len(logits) - 1)
        positive_loss = np.logaddexp(0, -diagonal)
        all_negative_loss = np.logaddexp(0, logits)
        negative_loss_mean = (
            all_negative_loss.sum(axis=1) - np.logaddexp(0, diagonal)
        ) / (len(logits) - 1)
        row = np.stack([
            diagonal,
            negative_logit_mean,
            diagonal - negative_logit_mean,
            np.argmax(logits, axis=1) == np.arange(len(logits)),
            (positive_loss + (len(logits) - 1) * negative_loss_mean) / len(logits),
        ], axis=1)
        all_stats.append(row)
        selected_t0 = (trajectory >= 3300) & (anchor == 0)
        t0_stats.append(row[selected_t0])
        candidate = replay["audit_candidate_id"][trajectory]
        for index, name in enumerate(candidate_names):
            selected = selected_t0 & (candidate == index)
            candidate_stats[name].append(row[selected])

    def summarize(chunks):
        joined = np.concatenate(chunks) if chunks and any(len(x) for x in chunks) else np.empty((0, 5))
        return {
            "rows": len(joined),
            "positive_logit": float(joined[:, 0].mean()) if len(joined) else None,
            "negative_logit": float(joined[:, 1].mean()) if len(joined) else None,
            "positive_minus_mean_negative": float(joined[:, 2].mean()) if len(joined) else None,
            "categorical_accuracy": float(joined[:, 3].mean()) if len(joined) else None,
            "original_sigmoid_nce_row_loss": float(joined[:, 4].mean()) if len(joined) else None,
        }

    return {
        "all_rows": summarize(all_stats),
        "synthetic_time0_rows": summarize(t0_stats),
        "candidate_time0_rows": {name: summarize(chunks) for name, chunks in candidate_stats.items()},
    }


def model_outcomes(generated):
    states = generated["states"][:, :50]
    failed = generated["failed"][:, :50]
    distance = np.linalg.norm(states[..., :2] - GOAL[:2], axis=2)
    reached = ((distance < 2) & ~failed).any(axis=1)
    strict = ((distance < 0.5) & ~failed).any(axis=1)
    result = {}
    for index, name in enumerate(CONFIG["query_actions"]["names"]):
        selected = generated["candidate_id"] == index
        result[name] = {
            "paths": int(selected.sum()),
            "region_reach": float(reached[selected].mean()),
            "strict_reach": float(strict[selected].mean()),
        }
    return result


def paired_goal_margins(network, q_params, roots, goals, actions):
    """Score exact root/action/goal tuples without forming cross-goal matrices."""
    output = np.empty((len(roots), len(goals)), np.float32)
    for root_index, root in enumerate(roots):
        repeated_root = np.broadcast_to(root, (len(goals), 8)).astype(np.float32)
        observation = jnp.asarray(np.concatenate([repeated_root, goals], axis=1))
        action_scores = []
        for action in (actions[1], actions[4]):
            repeated_action = jnp.asarray(np.broadcast_to(action, (len(goals), 2)))
            phi, psi = network.representation_network.apply(
                q_params, observation, repeated_action
            )
            value = jnp.sum(phi * psi, axis=1)
            if value.ndim == 2:
                value = value[:, 0]
            action_scores.append(np.asarray(value, np.float32))
        output[root_index] = action_scores[0] - action_scores[1]
    root_means = output.mean(axis=1)
    return {
        "goals": len(goals),
        "mean_down_minus_right": float(output.mean()),
        "roots_with_positive_mean": int((root_means > 0).sum()),
        "goal_fraction_with_positive_root_mean": float((output.mean(axis=0) > 0).mean()),
        "per_root_mean": root_means,
    }


def main():
    network = make_network()
    construction = load_npz(OUT / "construction_roots.npz")
    heldout_path = ROOT / CONFIG["inputs"]["heldout_roots"]
    heldout = load_npz(heldout_path)["state"].astype(np.float32)
    actions = construction["candidate_action"].astype(np.float32)
    rng = np.random.default_rng(2026091514)
    construction_bootstrap = rng.integers(
        0, len(construction["state"]), size=(5000, len(construction["state"]))
    )
    heldout_bootstrap = rng.integers(0, len(heldout), size=(5000, len(heldout)))
    lineage = load_npz(OUT / "nce_lineage.npz")
    update_ids = np.unique(np.linspace(0, 29999, 512, dtype=np.int32))
    replay_c = load_npz(OUT / "replay_C.npz")
    synthetic_t0 = (lineage["trajectory"] >= 3300) & (lineage["anchor_time"] == 0)
    row, column = np.nonzero(synthetic_t0)
    trajectory = lineage["trajectory"][row, column]
    future = lineage["future_time"][row, column]
    candidate = replay_c["audit_candidate_id"][trajectory]
    sampled_goal = replay_c["obs"][trajectory, future, :8]
    full_distance = np.linalg.norm(sampled_goal - GOAL, axis=1)
    xy_distance = np.linalg.norm(sampled_goal[:, :2] - GOAL[:2], axis=1)
    goal_support = {}
    goal_banks = {}
    for name, selected in {
        "down": candidate < 3,
        "right": candidate >= 3,
    }.items():
        distances = full_distance[selected]
        xy = xy_distance[selected]
        goal_support[name] = {
            "positive_pairs": int(selected.sum()),
            "minimum_full_f4_distance": float(distances.min()),
            "full_f4_distance_quantiles": np.quantile(
                distances, [0, 0.01, 0.05, 0.25, 0.5]
            ),
            "full_f4_threshold_counts": {
                str(threshold): int((distances <= threshold).sum())
                for threshold in (0.05, 0.1, 0.25, 0.5, 1.0)
            },
            "xy_distance_lt_2": int((xy < 2).sum()),
            "exact_canonical_matches": int(np.all(sampled_goal[selected] == GOAL, axis=1).sum()),
        }
        bank_mask = selected & (xy_distance < 2)
        bank = sampled_goal[bank_mask]
        _, unique_index = np.unique(bank, axis=0, return_index=True)
        goal_banks[name] = bank[np.sort(unique_index)[:512]].astype(np.float32)
    result = {
        "status": "post_decision_read_only_diagnostic",
        "training_or_checkpoint_selection": False,
        "new_native_interaction": False,
        "sampled_exact_training_updates": update_ids,
        "arms": {},
        "model_outcomes_C": model_outcomes(load_npz(OUT / "generated_C.npz")),
        "sampled_time0_goal_support": goal_support,
    }
    for arm in ("B", "C"):
        _, state = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        construction_score = scores(network, state.q_params, construction["state"], actions)
        heldout_score = scores(network, state.q_params, heldout, actions)
        replay = load_npz(OUT / f"replay_{arm}.npz")
        result["arms"][arm] = {
            "canonical_goal_construction_roots": margin_summary(
                construction_score, construction_bootstrap
            ),
            "canonical_goal_heldout_roots": margin_summary(heldout_score, heldout_bootstrap),
            "exact_lineage_batch_fit": replay_batch_fit(
                network, state.q_params, replay, lineage, update_ids
            ),
            "heldout_margin_for_sampled_goal_banks": {
                name: paired_goal_margins(
                    network, state.q_params, heldout, bank, actions
                )
                for name, bank in goal_banks.items()
            },
        }
    result["interpretation_guard"] = (
        "Exact-batch fit measures the original sigmoid-NCE discrimination on sampled future goals; "
        "canonical-goal margins separately measure route ordering. Neither is a calibrated success probability."
    )
    (OUT / "fit_diagnostic.json").write_text(
        json.dumps(plain(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    concise = {
        arm: {
            "construction": result["arms"][arm]["canonical_goal_construction_roots"],
            "heldout": result["arms"][arm]["canonical_goal_heldout_roots"],
            "nce_all": result["arms"][arm]["exact_lineage_batch_fit"]["all_rows"],
            "nce_t0": result["arms"][arm]["exact_lineage_batch_fit"]["synthetic_time0_rows"],
        }
        for arm in ("B", "C")
    }
    print(json.dumps(plain(concise), indent=2), flush=True)


if __name__ == "__main__":
    main()
