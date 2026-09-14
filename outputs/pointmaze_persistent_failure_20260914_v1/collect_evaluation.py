"""Collect one new paired PointMaze evaluation set with frozen seeds."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXTERNAL = Path(r"C:\Users\trhua\Documents\Codex\2026-09-08\f")
sys.path[:0] = [
    str(EXTERNAL / "work" / "branch-source-031d430"),
    str(EXTERNAL / "work" / "branch-source-f85a5f4"),
]
from crl.envs import TwoRouteSwampWindyF4Env  # noqa: E402
from scripts.collect_swamp_windy import make_windy_teacher  # noqa: E402
from scripts.collect_swamp_windy_baddemo import make_bad_demonstrator  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    rows = {
        key: []
        for key in (
            "s", "xb", "xq", "y", "episode", "time", "query",
            "prefix_mode", "dead_before", "dead_after", "reward"
        )
    }
    native_steps = 0
    exact_prefix_replays = 0
    for episode in range(args.episodes):
        env = TwoRouteSwampWindyF4Env(seed=args.seed + episode, active_prob=0.3)
        observation = env.reset()
        teacher_rng = np.random.default_rng(args.seed + 10000 + episode)
        query_rng = np.random.default_rng(args.seed + 20000 + episode)
        teacher = make_windy_teacher(env, teacher_rng, 0.05)
        blind = make_bad_demonstrator(env)
        teacher_memo, blind_memo = {}, {}
        prefix_mode = episode % 2
        for time in range(50):
            xb = np.asarray(
                teacher(env.state.copy(), env.goal.copy(), teacher_memo), np.float32
            )
            if np.any(xb != 0):
                xb = np.clip(xb + teacher_rng.normal(0, 0.15, 2), -1, 1).astype(np.float32)
            forward = np.asarray(
                blind(env.state.copy(), env.goal.copy(), blind_memo), np.float32
            )
            actions = [
                xb,
                forward,
                -xb,
                query_rng.uniform(-1, 1, 2).astype(np.float32),
            ]
            paired = []
            for query, action in enumerate(actions):
                branch = copy.deepcopy(env)
                next_observation, reward, done, _ = branch.step(action)
                native_steps += 1
                if done:
                    raise RuntimeError("fixed-length collector unexpectedly terminated")
                np.testing.assert_array_equal(next_observation[2:8], observation[:6])
                values = dict(
                    s=observation[:8].copy(),
                    xb=xb.copy(),
                    xq=action.copy(),
                    y=next_observation[:8].copy(),
                    episode=episode,
                    time=time,
                    query=query,
                    prefix_mode=prefix_mode,
                    dead_before=env.dead,
                    dead_after=branch.dead,
                    reward=reward,
                )
                for key, value in values.items():
                    rows[key].append(value)
                paired.append(next_observation)
            action_index = 0 if prefix_mode == 0 else 1
            observation, _, done, _ = env.step(actions[action_index])
            native_steps += 1
            if done:
                raise RuntimeError("fixed-length prefix unexpectedly terminated")
            np.testing.assert_array_equal(observation, paired[action_index])
            exact_prefix_replays += 1
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    assert len(arrays["s"]) == args.episodes * 50 * 4
    np.savez_compressed(out, **arrays)
    manifest = {
        "episodes": args.episodes,
        "rows": len(arrays["s"]),
        "native_steps": native_steps,
        "exact_prefix_replays": exact_prefix_replays,
        "balanced_prefix_episode_counts": {
            "teacher": int(np.sum(np.arange(args.episodes) % 2 == 0)),
            "blind": int(np.sum(np.arange(args.episodes) % 2 == 1)),
        },
        "base_seed": args.seed,
        "all_outcomes_retained": True,
        "output_sha256": sha256(out),
    }
    out.with_name("paired_evaluation_collection.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
