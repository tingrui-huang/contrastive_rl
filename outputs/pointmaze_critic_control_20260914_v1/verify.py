"""Independent saved-artifact checks for the PointMaze critic control."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
MATCHED_DIR = ROOT / "outputs" / "pointmaze_matched_fork_20260914_v1"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUN = load_module("verify_critic_control", OUT / "run.py")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    RUN.verify_training_seal()
    discovery = load_npz(OUT / "discovery_native.npz")
    roots = load_npz(OUT / "roots_and_split.npz")
    streams = load_npz(OUT / "fork_streams.npz")
    native = load_npz(OUT / "native_fork_traces.npz")
    model = load_npz(OUT / "model_fork_traces.npz")
    labels = load_npz(OUT / "ranking_labels.npz")
    train_index = load_npz(OUT / "nce_train_indices.npz")
    validation_index = load_npz(OUT / "nce_validation_indices.npz")
    training = load_json(OUT / "training_results.json")
    completion = load_json(OUT / "completion.json")

    assert discovery["states"].shape == (CONFIG["discovery_episode_count"], 51, 8)
    assert discovery["action"].shape == (CONFIG["discovery_episode_count"], 50, 2)
    np.testing.assert_array_equal(discovery["states"][:, 1:, 2:], discovery["states"][:, :-1, :6])
    assert len(np.unique(roots["reset_seed"])) == CONFIG["root_target"]
    assert len(roots["train_root"]) == CONFIG["train_root_count"]
    assert len(roots["validation_root"]) == CONFIG["validation_root_count"]
    assert not set(roots["train_root"]).intersection(set(roots["validation_root"]))
    assert set(np.concatenate([roots["train_root"], roots["validation_root"]])) == set(range(CONFIG["root_target"]))
    old = load_npz(MATCHED_DIR / "root_selection.npz")
    assert not set(roots["reset_seed"]).intersection(set(old["reset_seed"]))
    state = roots["state"]
    assert np.all((state[:, 0] >= 1) & (state[:, 0] < 2) & (state[:, 1] >= 3) & (state[:, 1] < 4))
    selected_episode = roots["discovery_episode"]
    selected_time = roots["time"]
    assert np.all(~discovery["failed_before"][selected_episode, selected_time])
    for root_index, (episode, time_index) in enumerate(zip(selected_episode, selected_time)):
        simulator = RUN.reconstruct_discovery_root(discovery, int(episode), int(time_index))
        np.testing.assert_array_equal(simulator.observation()[:8], roots["state"][root_index])
        np.testing.assert_array_equal(simulator.env.swamp_bits, roots["root_bits"][root_index])

    candidate = RUN.candidate_actions()
    for action_index in range(2):
        actual = native["action"][:, action_index, :, 0]
        np.testing.assert_array_equal(actual, np.broadcast_to(candidate[action_index], actual.shape))
        actual = model["action"][:, action_index, :, 0]
        np.testing.assert_array_equal(actual, np.broadcast_to(candidate[action_index], actual.shape))
    expected_root_bits = np.broadcast_to(
        roots["root_bits"][:, None, :], native["swamp_bits_before"][:, 0, :, 0].shape
    )
    np.testing.assert_array_equal(native["swamp_bits_before"][:, 0, :, 0], expected_root_bits)
    np.testing.assert_array_equal(native["swamp_bits_before"][:, 1, :, 0], expected_root_bits)
    for action_index in range(2):
        expected_bits = streams["native_bits_after"][:, :, : native["valid"].shape[-1]]
        np.testing.assert_array_equal(
            native["swamp_bits_after"][:, action_index][native["valid"][:, action_index]],
            expected_bits[native["valid"][:, action_index]],
        )

    weights_native = CONFIG["discount"] ** np.arange(native["reward"].shape[-1])
    weights_model = CONFIG["discount"] ** np.arange(model["reward"].shape[-1])
    np.testing.assert_allclose(
        native["discounted_return"], np.sum(native["reward"] * native["valid"] * weights_native, axis=-1),
        rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        model["discounted_return"], np.sum(model["reward"] * model["valid"] * weights_model, axis=-1),
        rtol=0, atol=1e-6
    )
    model_gap = model["discounted_return"][:, 0].mean(axis=1) - model["discounted_return"][:, 1].mean(axis=1)
    native_gap = native["discounted_return"][:, 0].mean(axis=1) - native["discounted_return"][:, 1].mean(axis=1)
    np.testing.assert_allclose(labels["model_return_gap"], model_gap, rtol=0, atol=1e-7)
    np.testing.assert_allclose(labels["native_return_gap"], native_gap, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(labels["preferred_sign"], np.sign(model_gap))

    train_set = set(int(value) for value in roots["train_root"])
    validation_set = set(int(value) for value in roots["validation_root"])
    assert set(np.unique(train_index["root"])) <= train_set
    assert set(np.unique(validation_index["root"])) <= validation_set
    assert np.all(train_index["future"] > train_index["anchor"])
    assert np.all(validation_index["future"] > validation_index["anchor"])
    for update in range(CONFIG["critic_updates_per_condition"]):
        identity = np.stack(
            [train_index["root"][update], train_index["action"][update], train_index["repeat"][update]], axis=1
        )
        assert len(np.unique(identity, axis=0)) == CONFIG["critic_batch_size"]
        root_anchor = train_index["anchor"][update] == 0
        assert int(np.sum(root_anchor & (train_index["action"][update] == 0))) == 64
        assert int(np.sum(root_anchor & (train_index["action"][update] == 1))) == 64

    _, _, _, initial, nominal, head, state_mean, state_std = RUN.frozen_components()
    initial_actor_hash = RUN.tree_sha(initial.policy_params)
    assert training["updates_per_condition"] == CONFIG["critic_updates_per_condition"]
    assert training["actor_updates"] == training["model_updates"] == 0
    final_critic_hashes = []
    for condition in ("A_nce_only", "B_nce_plus_rank"):
        path = OUT / training["final_checkpoints"][condition]["path"]
        assert sha256(path) == training["final_checkpoints"][condition]["sha256"]
        _, final = RUN.checkpoint.load_checkpoint(path)
        assert RUN.tree_sha(final.policy_params) == initial_actor_hash
        assert RUN.tree_sha(final.q_params) != RUN.tree_sha(initial.q_params)
        final_critic_hashes.append(RUN.tree_sha(final.q_params))
    assert final_critic_hashes[0] != final_critic_hashes[1]
    assert RUN.tree_sha((initial.policy_params, nominal.params, head, state_mean, state_std)) == training["frozen_model_bundle_sha256"]
    assert CONFIG["rank_weight_A"] == 0.0 and CONFIG["rank_weight_B"] == 1.0

    for name, expected in completion["artifact_sha256"].items():
        assert sha256(OUT / name) == expected, name
    result = {
        "status": "passed",
        "initial_and_training_seals": True,
        "new_native_episodes_complete": True,
        "roots_naturally_reached_alive_and_episode_disjoint": True,
        "old_16_roots_external_only": True,
        "fixed_first_actions": True,
        "native_hidden_timing_and_paired_streams": True,
        "discounted_returns_recomputed": True,
        "ranking_labels_derived_from_model_returns": True,
        "A_B_common_native_data_and_nce_indices": True,
        "future_goals_strictly_later": True,
        "per_batch_fork_anchor_balance": True,
        "actor_and_model_frozen": True,
        "critic_updates_per_condition": CONFIG["critic_updates_per_condition"],
        "checkpoint_searches": 0,
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    completion["status"] = "complete_verified"
    completion["verification_sha256"] = sha256(OUT / "verification.json")
    (OUT / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
