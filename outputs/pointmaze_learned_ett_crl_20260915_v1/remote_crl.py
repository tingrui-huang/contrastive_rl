"""Torch-free GPU entrypoint for the two sealed CRL arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT))

from crl import checkpoint  # noqa: E402
from crl.config import Config  # noqa: E402


CONFIG = json.loads((OUT / "config.json").read_text())


def original_dataset():
    choices = [
        ROOT / "datasets" / "swamp_windy_f4_merged_s0.npz",
        ROOT / "artifacts" / "f4_p30_server_30076" / "results" / "datasets" / "swamp_windy_f4_merged_s0.npz",
    ]
    return next(path for path in choices if path.exists())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_sha(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.ascontiguousarray(np.asarray(leaf))
        h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_config(arm):
    dataset = original_dataset() if arm == "O" else OUT / "learned_ett_replay.npz"
    return Config(
        env_name="point_two_route_swamp_windy_f4_v0", offline_dataset=str(dataset),
        obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
        start_index=0, end_index=-1,
        max_number_of_steps=30000,
        fail_bank_path="", fail_neg_alpha=0.0, obs_norm_mode="", obs_norm_z_scale=0.0,
        anchor_cut_mode="", balanced_sampling=False,
        use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
        bc_coef=0.05, random_goals=0.5,
        entropy_coefficient=0.0, target_entropy=0.0,
        batch_size=256, repr_dim=64, hidden_layer_sizes=(256, 256), discount=0.95,
        learning_rate=3e-4, actor_learning_rate=3e-4,
        num_sgd_steps_per_step=10, num_actors=0,
        guard_abort=True, jit=True, seed=0,
        eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=1_000,
        ckpt_every_steps=30000, ckpt_dir=str(OUT / "crl" / arm),
    )


def train_arm(arm):
    from crl.train import train
    cfg = build_config(arm)
    final = OUT / "crl" / arm / "final.pkl"
    if final.exists():
        print(f"arm {arm} already complete")
        return
    print(f"SEALED ARM {arm}: bc_coef={cfg.bc_coef}, fixed 30000, no training eval", flush=True)
    started = time.time()
    train(cfg)
    if not final.exists():
        raise RuntimeError("missing fixed final checkpoint")
    write_json(OUT / "crl" / arm / "run_summary.json", {
        "arm": arm, "bc_coef": cfg.bc_coef, "steps": 30000,
        "dataset": str(cfg.offline_dataset), "dataset_sha256": sha256(cfg.offline_dataset),
        "native_training_evaluations": 0,
        "initial_checkpoint_sha256": sha256(OUT / "crl" / arm / "init.pkl"),
        "final_checkpoint_sha256": sha256(final), "wall_seconds": time.time() - started,
        "jax_devices": [str(device) for device in jax.devices()],
    })


def verify():
    result = {"bc_coef_every_arm": 0.05, "arms": {}, "matched": {}}
    for arm in ("O", "P"):
        _, initial = checkpoint.load_checkpoint(OUT / "crl" / arm / "init.pkl")
        step, final = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        result["arms"][arm] = {
            "step": int(step), "initial_tree_sha256": tree_sha(initial),
            "initial_policy_sha256": tree_sha(initial.policy_params),
            "initial_critic_sha256": tree_sha(initial.q_params),
            "final_policy_sha256": tree_sha(final.policy_params),
            "final_critic_sha256": tree_sha(final.q_params),
            "actor_updated": tree_sha(final.policy_params) != tree_sha(initial.policy_params),
            "critic_updated": tree_sha(final.q_params) != tree_sha(initial.q_params),
        }
    o, p = result["arms"]["O"], result["arms"]["P"]
    result["matched"] = {
        "initial_full_state": o["initial_tree_sha256"] == p["initial_tree_sha256"],
        "initial_policy": o["initial_policy_sha256"] == p["initial_policy_sha256"],
        "initial_critic": o["initial_critic_sha256"] == p["initial_critic_sha256"],
        "budget": o["step"] == p["step"] == 30000,
        "both_actor_and_critic_updated": all(result["arms"][a][k] for a in ("O", "P")
                                             for k in ("actor_updated", "critic_updated")),
        "no_training_native_evaluation": True,
    }
    result["passed"] = all(result["matched"].values())
    write_json(OUT / "crl_verification.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("O", "P", "verify"))
    args = parser.parse_args()
    if args.command == "verify": verify()
    else: train_arm(args.command)


if __name__ == "__main__":
    main()
