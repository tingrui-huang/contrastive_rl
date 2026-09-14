"""Collect one sealed PointMaze repair split with paired same-snapshot outcomes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
PERSISTENT = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
TEACHER_SOURCE = Path(
    r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\branch-source-f85a5f4"
    r"\scripts\collect_swamp_windy.py"
)
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))
from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARENT = load_module("repair_collection_parent", PERSISTENT / "run.py")


def load_original_teacher():
    return load_module("repair_original_teacher", TEACHER_SOURCE).make_windy_teacher


MAKE_TEACHER = load_original_teacher()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(env):
    return {
        "state": np.asarray(env.state).copy(),
        "frames": np.asarray(env.frames).copy(),
        "goal": np.asarray(env.goal).copy(),
        "bits": np.asarray(env.swamp_bits).copy(),
        "dead": bool(env.dead),
        "auto": bool(env._auto_resample),
        "rng": json.dumps(env._rng.bit_generator.state, sort_keys=True, separators=(",", ":")),
    }


def assert_same(left, right):
    for name in ("state", "frames", "goal", "bits"):
        np.testing.assert_array_equal(left[name], right[name])
    for name in ("dead", "auto", "rng"):
        assert left[name] == right[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=sorted(CONFIG["splits"]), required=True)
    args = parser.parse_args()
    split = args.split
    spec = CONFIG["splits"][split]
    output_path = OUT / f"{split}_paired.npz"
    manifest_path = OUT / f"{split}_collection.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"{split} collection already exists; reuse it")

    engine = PARENT.load_position_engine()
    actor = engine.kernel.actor
    rows = {name: [] for name in (
        "episode", "time", "prefix_source", "s", "xb", "xq", "y",
        "dead_before", "dead_after", "onset", "reward", "query_type",
        "swamp_bits_before", "shadow_force_safe", "shadow_teacher_mode",
    )}
    prefix = {name: [] for name in (
        "states", "actions", "rewards", "dead_before", "dead_after", "onset",
        "environment_seed", "prefix_source",
    )}
    mode_code = {"forced_safe": 1, "immediate_shortcut": 2, "wait_shortcut": 3}
    native_steps = 0
    teacher_queries = 0
    actor_queries = 0
    exact_prefix_branches = 0

    for episode in range(spec["episodes"]):
        env = TwoRouteSwampWindyF4Env(
            seed=spec["environment_seed_base"] + episode,
            active_prob=CONFIG["active_probability"],
        )
        observation = env.reset()
        teacher_rng = np.random.default_rng(spec["teacher_seed_base"] + episode)
        random_rng = np.random.default_rng(spec["random_seed_base"] + episode)
        teacher = MAKE_TEACHER(env, teacher_rng, CONFIG["force_safe_probability"])
        memo = {}
        prefix_source = 1 if split == "final" or episode % 2 else 0
        episode_states = [observation[:8].copy()]
        episode_actions, episode_rewards = [], []
        episode_before, episode_after, episode_onset = [], [], []

        for time_index in range(CONFIG["horizon"]):
            before_query = snapshot(env)
            advice_raw = np.asarray(
                teacher(env.state.copy(), env.goal.copy(), memo), np.float32
            )
            if np.any(advice_raw != 0):
                advice = np.clip(
                    advice_raw + teacher_rng.normal(0, CONFIG["teacher_noise"], 2), -1, 1
                ).astype(np.float32)
            else:
                advice = advice_raw.copy()
            assert_same(before_query, snapshot(env))
            teacher_queries += 1

            actor_key = jax.random.PRNGKey(
                spec["actor_seed_base"] + episode * CONFIG["horizon"] + time_index
            )
            actor_action = np.asarray(
                actor(
                    jnp.asarray(observation[None, :8]),
                    jnp.asarray(observation[None, 8:]),
                    actor_key,
                )
            )[0].astype(np.float32)
            actor_queries += 1
            random_action = random_rng.uniform(-1, 1, 2).astype(np.float32)
            actions = [advice, actor_action, -advice, random_action]
            branches = []
            for query_type, action in enumerate(actions):
                branch = copy.deepcopy(env)
                following, reward, done, _ = branch.step(action)
                native_steps += 1
                if done:
                    raise RuntimeError("fixed-length branch unexpectedly terminated")
                dead_before = bool(env.dead)
                dead_after = bool(branch.dead)
                expected_reward = float(
                    (np.linalg.norm(following[:2] - PARENT.GOAL[:2]) < 2) and (not dead_after)
                )
                assert reward == expected_reward
                values = {
                    "episode": episode,
                    "time": time_index,
                    "prefix_source": prefix_source,
                    "s": observation[:8].copy(),
                    "xb": advice.copy(),
                    "xq": action.copy(),
                    "y": following[:8].copy(),
                    "dead_before": dead_before,
                    "dead_after": dead_after,
                    "onset": (not dead_before) and dead_after,
                    "reward": reward,
                    "query_type": query_type,
                    "swamp_bits_before": before_query["bits"],
                    "shadow_force_safe": bool(memo["force_safe"]),
                    "shadow_teacher_mode": mode_code[memo["teacher_mode"]],
                }
                for name, value in values.items():
                    rows[name].append(value)
                branches.append((following, reward, dead_after))

            selected = prefix_source
            dead_before = bool(env.dead)
            following, reward, done, _ = env.step(actions[selected])
            native_steps += 1
            if done:
                raise RuntimeError("fixed-length prefix unexpectedly terminated")
            np.testing.assert_array_equal(following, branches[selected][0])
            assert reward == branches[selected][1] and bool(env.dead) == branches[selected][2]
            exact_prefix_branches += 1
            episode_states.append(following[:8].copy())
            episode_actions.append(actions[selected].copy())
            episode_rewards.append(reward)
            episode_before.append(dead_before)
            episode_after.append(bool(env.dead))
            episode_onset.append((not dead_before) and bool(env.dead))
            observation = following

        prefix["states"].append(episode_states)
        prefix["actions"].append(episode_actions)
        prefix["rewards"].append(episode_rewards)
        prefix["dead_before"].append(episode_before)
        prefix["dead_after"].append(episode_after)
        prefix["onset"].append(episode_onset)
        prefix["environment_seed"].append(spec["environment_seed_base"] + episode)
        prefix["prefix_source"].append(prefix_source)
        if (episode + 1) % 32 == 0:
            print(f"collected {split}: {episode + 1}/{spec['episodes']}", flush=True)

    arrays = {name: np.asarray(value) for name, value in rows.items()}
    arrays.update({f"prefix_{name}": np.asarray(value) for name, value in prefix.items()})
    np.savez_compressed(output_path, **arrays)
    manifest = {
        "split": split,
        "episodes": spec["episodes"],
        "rows": len(arrays["episode"]),
        "native_steps": native_steps,
        "teacher_queries": teacher_queries,
        "actor_queries": actor_queries,
        "exact_prefix_branch_matches": exact_prefix_branches,
        "prefix_source_episode_counts": {
            "expert": int(np.sum(arrays["prefix_prefix_source"] == 0)),
            "actor": int(np.sum(arrays["prefix_prefix_source"] == 1)),
        },
        "all_outcomes_retained": True,
        "teacher_query_preserved_environment_state_bits_and_rng": True,
        "same_snapshot_branch_rng_coupled": True,
        "output_sha256": sha256(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
