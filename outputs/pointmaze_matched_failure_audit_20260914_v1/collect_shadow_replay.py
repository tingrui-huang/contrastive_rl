"""Replay saved native actor actions and attach same-context shadow advice."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / "outputs" / "pointmaze_persistent_failure_20260914_v1"
TEACHER_SOURCE = Path(
    r"C:\Users\trhua\Documents\Codex\2026-09-08\f\work\branch-source-f85a5f4"
    r"\scripts\collect_swamp_windy.py"
)
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))

sys.path.insert(0, str(ROOT))
from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402


def load_original_teacher():
    spec = importlib.util.spec_from_file_location("pinned_original_windy_teacher", TEACHER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.make_windy_teacher


MAKE_TEACHER = load_original_teacher()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def rng_state_bytes(env):
    return json.dumps(
        env._rng.bit_generator.state, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def snapshot(env):
    return {
        "state": np.asarray(env.state).copy(),
        "frames": np.asarray(env.frames).copy(),
        "goal": np.asarray(env.goal).copy(),
        "bits": np.asarray(env.swamp_bits).copy(),
        "dead": bool(env.dead),
        "auto_resample": bool(env._auto_resample),
        "rng": rng_state_bytes(env),
    }


def assert_same_snapshot(before, after):
    for name in ("state", "frames", "goal", "bits"):
        np.testing.assert_array_equal(before[name], after[name])
    assert before["dead"] == after["dead"]
    assert before["auto_resample"] == after["auto_resample"]
    assert before["rng"] == after["rng"]


def main():
    output_path = OUT / "augmented_shadow_replay.npz"
    verification_path = OUT / "replay_verification.json"
    if output_path.exists() or verification_path.exists():
        raise FileExistsError("replay output already exists; reuse it instead of repeating native calls")

    native = load_npz(SOURCE / "native_actor_episodes.npz")
    assert native["states"].shape == (CONFIG["native_episodes"], CONFIG["horizon"] + 1, 8)
    assert native["actions"].shape == (CONFIG["native_episodes"], CONFIG["horizon"], 2)
    rows = {name: [] for name in (
        "episode", "time", "s", "xb_shadow", "xq_saved_actor", "y_real",
        "dead_before", "dead_after", "onset", "reward", "swamp_bits_before",
        "shadow_force_safe", "shadow_teacher_mode", "shadow_wait_count",
    )}
    teacher_mode_codes = {"forced_safe": 1, "immediate_shortcut": 2, "wait_shortcut": 3}
    checked_queries = 0
    exact_observations = 0
    exact_successors = 0
    exact_death_states = 0
    exact_rewards = 0

    for episode in range(CONFIG["native_episodes"]):
        env = TwoRouteSwampWindyF4Env(
            seed=int(native["environment_seed"][episode]),
            active_prob=CONFIG["active_probability"],
        )
        observation = env.reset()
        teacher_rng = np.random.default_rng(CONFIG["shadow_teacher_seed_base"] + episode)
        teacher = MAKE_TEACHER(env, teacher_rng, CONFIG["shadow_force_safe_probability"])
        memo = {}
        for time in range(CONFIG["horizon"]):
            np.testing.assert_array_equal(observation[:8], native["states"][episode, time])
            exact_observations += 1
            before = snapshot(env)
            unnoised = np.asarray(
                teacher(env.state.copy(), env.goal.copy(), memo), np.float32
            )
            if np.any(unnoised != 0):
                xb = np.clip(
                    unnoised + teacher_rng.normal(0, CONFIG["shadow_teacher_noise"], 2),
                    -1,
                    1,
                ).astype(np.float32)
            else:
                xb = unnoised.copy()
            after_query = snapshot(env)
            assert_same_snapshot(before, after_query)
            checked_queries += 1

            action = native["actions"][episode, time]
            dead_before = bool(env.dead)
            following, reward, done, _ = env.step(action)
            if done:
                raise RuntimeError("fixed-length replay unexpectedly terminated")
            dead_after = bool(env.dead)
            onset = (not dead_before) and dead_after
            np.testing.assert_array_equal(following[:8], native["states"][episode, time + 1])
            exact_successors += 1
            assert dead_before == bool(native["dead_before"][episode, time])
            assert dead_after == bool(native["dead_after"][episode, time])
            assert onset == bool(native["onset"][episode, time])
            exact_death_states += 1
            assert reward == float(native["rewards"][episode, time])
            exact_rewards += 1

            values = {
                "episode": episode,
                "time": time,
                "s": observation[:8].copy(),
                "xb_shadow": xb,
                "xq_saved_actor": action.copy(),
                "y_real": following[:8].copy(),
                "dead_before": dead_before,
                "dead_after": dead_after,
                "onset": onset,
                "reward": reward,
                "swamp_bits_before": before["bits"],
                "shadow_force_safe": bool(memo["force_safe"]),
                "shadow_teacher_mode": teacher_mode_codes[memo["teacher_mode"]],
                "shadow_wait_count": int(memo["wait_count"]),
            }
            for name, value in values.items():
                rows[name].append(value)
            observation = following

    arrays = {name: np.asarray(values) for name, values in rows.items()}
    np.savez_compressed(output_path, **arrays)
    verification = {
        "status": "pass",
        "episodes": CONFIG["native_episodes"],
        "rows": len(arrays["episode"]),
        "native_replay_steps": len(arrays["episode"]),
        "teacher_queries": checked_queries,
        "exact_saved_observations": exact_observations,
        "exact_saved_successors": exact_successors,
        "exact_saved_death_state_tuples": exact_death_states,
        "exact_saved_rewards": exact_rewards,
        "teacher_query_preserved_native_state_bits_and_rng": True,
        "teacher_rng": "independent per episode: seed 14140000 + episode",
        "advice_status": "newly sampled shadow advice on the original hidden situation; not historical advice recovery",
        "native_source_sha256": sha256(SOURCE / "native_actor_episodes.npz"),
        "teacher_source_sha256": sha256(TEACHER_SOURCE),
        "output_sha256": sha256(output_path),
    }
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
