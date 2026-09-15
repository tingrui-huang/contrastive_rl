"""Read-only post-hoc route-ranking diagnostic on established fork roots."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
ROOTS_DIR = ROOT / "outputs" / "pointmaze_matched_fork_20260914_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from crl import checkpoint, networks  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    with np.load(ROOTS_DIR / "root_selection.npz", allow_pickle=False) as loaded:
        roots = {key: loaded[key] for key in loaded.files}
    with np.load(ROOTS_DIR / "primary_episode_results.npz", allow_pickle=False) as loaded:
        prior = {key: loaded[key] for key in loaded.files}

    network = networks.make_networks(
        8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
        hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
        use_image_obs=False, use_layer_norm=False, obs_scale=None,
    )
    states = roots["state"].astype(np.float32)
    actions = roots["candidate_action"].astype(np.float32)
    goal = np.broadcast_to(np.tile(np.array([8.5, 3.5], np.float32), 4), states.shape)
    observation = jnp.asarray(np.concatenate([states, goal], axis=1))

    def scores(q_params, action):
        batch_action = jnp.asarray(np.broadcast_to(action, (len(states), 2)))
        phi, psi = network.representation_network.apply(q_params, observation, batch_action)
        value = jnp.sum(phi * psi, axis=1)
        if value.ndim == 2:
            value = value[:, 0]
        return np.asarray(value, np.float32)

    rng = np.random.default_rng(2026091599)
    bootstrap = rng.integers(0, len(states), size=(2000, len(states)))
    arms = {}
    for arm in ("O", "P"):
        _, state = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        down = scores(state.q_params, actions[0])
        right = scores(state.q_params, actions[1])
        margin = down - right
        boot_mean = margin[bootstrap].mean(axis=1)
        arms[arm] = {
            "down_logit_mean": float(down.mean()),
            "right_logit_mean": float(right.mean()),
            "down_minus_right_mean": float(margin.mean()),
            "down_minus_right_root_bootstrap_ci95": np.quantile(boot_mean, [0.025, 0.975]).tolist(),
            "roots_preferring_down": int((margin > 0).sum()),
            "ranking_accuracy_against_established_native_teacher": float((margin > 0).mean()),
            "per_root_down_minus_right": margin.tolist(),
        }

    native_return = prior["native_whole_episode_return"].mean(axis=2)
    native_success = prior["native_strict_success"].mean(axis=2)
    result = {
        "status": "post_hoc_read_only_diagnostic",
        "not_used_for_fitting_selection_or_protocol_changes": True,
        "no_new_native_interaction": True,
        "root_set": {
            "description": "16 previously established coherent alive fork roots; down and right were already natively evaluated with 64 continuations per root/action",
            "root_selection_sha256": sha256(ROOTS_DIR / "root_selection.npz"),
            "prior_episode_results_sha256": sha256(ROOTS_DIR / "primary_episode_results.npz"),
            "roots": len(states),
            "candidate_order": roots["candidate_name"].tolist(),
        },
        "established_native_teacher": {
            "down_minus_right_return_mean": float((native_return[:, 0] - native_return[:, 1]).mean()),
            "down_minus_right_success_rate": float((native_success[:, 0] - native_success[:, 1]).mean()),
            "roots_with_down_higher_return": int((native_return[:, 0] > native_return[:, 1]).sum()),
        },
        "arms": arms,
        "interpretation": (
            "P ranks the established beneficial down action above right on every saved root"
            if arms["P"]["roots_preferring_down"] == len(states)
            else "P does not consistently rank the established beneficial down action above right"
        ),
    }
    (OUT / "critic_ranking_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
