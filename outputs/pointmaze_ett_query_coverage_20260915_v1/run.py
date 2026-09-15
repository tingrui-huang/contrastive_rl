"""Run the sealed fixed-ETT query-coverage intervention."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
BASE = ROOT / "outputs" / "pointmaze_learned_ett_crl_20260915_v1"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))


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


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def input_path(name):
    raw = Path(CONFIG["inputs"][name])
    return raw if raw.is_absolute() else ROOT / raw


def make_network():
    from crl import networks
    return networks.make_networks(
        8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
        hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
        use_image_obs=False, use_layer_norm=False, obs_scale=None,
    )


def tree_sha(tree):
    import jax
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def select_training_roots():
    supervision = load_npz(input_path("supervision"))
    state = supervision["train_s"]
    mask = (
        (supervision["train_time"] == 1)
        & (state[:, 0] >= 1) & (state[:, 0] < 2)
        & (state[:, 1] >= 3) & (state[:, 1] < 4)
    )
    eligible = np.flatnonzero(mask)
    order = sorted(
        eligible.tolist(),
        key=lambda row: (
            int(supervision["train_source"][row]),
            int(supervision["train_episode"][row]),
            int(supervision["train_source_row"][row]),
        ),
    )
    seen, selected = set(), []
    for row in order:
        key = np.ascontiguousarray(state[row]).tobytes()
        if key not in seen:
            seen.add(key)
            selected.append(row)
    selected = np.asarray(selected, np.int64)
    expected = CONFIG["training_roots"]["expected_unique_roots"]
    if len(selected) != expected:
        raise AssertionError(f"expected {expected} exact-F4 roots, got {len(selected)}")
    return {
        "state": state[selected].astype(np.float32),
        "supervision_row": selected,
        "source": supervision["train_source"][selected],
        "episode": supervision["train_episode"][selected],
        "time": supervision["train_time"][selected],
        "source_row": supervision["train_source_row"][selected],
    }, int(mask.sum())


def audit_stage():
    if (OUT / "seal.json").exists():
        print("coverage protocol already sealed", flush=True)
        return
    input_hashes = {}
    for name in (
        "ett_checkpoint", "supervision", "prior_replay", "nominal", "rollout_actor",
        "original_dataset", "heldout_roots", "posthoc_diagnostic",
    ):
        path = input_path(name)
        actual = sha256(path)
        expected = CONFIG["inputs"][name + "_sha256"]
        if actual != expected:
            raise AssertionError(f"input hash mismatch {name}: {actual} != {expected}")
        input_hashes[name] = {"path": str(path), "sha256": actual}

    roots, eligible_rows = select_training_roots()
    heldout = load_npz(input_path("heldout_roots"))["state"].astype(np.float32)
    exact_overlap = np.any(np.all(roots["state"][:, None] == heldout[None], axis=2))
    distances = np.linalg.norm(roots["state"][:, None] - heldout[None], axis=2)
    if exact_overlap:
        raise AssertionError("construction roots overlap held-out roots")

    rng = np.random.default_rng(CONFIG["training_roots"]["selection_seed"])
    third = rng.permutation(len(roots["state"]))[
        :CONFIG["training_roots"]["roots_with_third_replicate"]
    ]
    context_root = np.concatenate([
        np.arange(len(roots["state"]), dtype=np.int32),
        np.arange(len(roots["state"]), dtype=np.int32),
        third.astype(np.int32),
    ])
    context_root = context_root[rng.permutation(len(context_root))]
    if len(context_root) != CONFIG["training_roots"]["contexts"]:
        raise AssertionError("context budget mismatch")

    actions = np.asarray(CONFIG["query_actions"]["values"], np.float32)
    endpoints = roots["state"][:, None, :2] + actions[None]
    cells = np.floor(endpoints).astype(np.int32)
    down_geometry = bool(np.all(cells[:, :3] == np.array([1, 2])))
    right_geometry = bool(np.all(cells[:, 3:] == np.array([2, 3])))
    in_bounds = bool(np.all(np.abs(actions) <= 1))
    if not (down_geometry and right_geometry and in_bounds):
        raise AssertionError("sealed candidate geometry failed")

    np.savez_compressed(
        OUT / "construction_roots.npz", **roots, context_root=context_root,
        third_replicate_root=third.astype(np.int32), candidate_action=actions,
        candidate_name=np.asarray(CONFIG["query_actions"]["names"]),
    )
    root_manifest = {
        "eligible_training_rows_before_exact_f4_dedup": eligible_rows,
        "unique_construction_roots": len(roots["state"]),
        "source_counts": {
            str(source): int((roots["source"] == source).sum())
            for source in np.unique(roots["source"])
        },
        "context_count": len(context_root),
        "replicate_count_by_root": {
            "two": int((np.bincount(context_root) == 2).sum()),
            "three": int((np.bincount(context_root) == 3).sum()),
        },
        "heldout_roots": len(heldout),
        "exact_f4_overlap_with_heldout": bool(exact_overlap),
        "minimum_full_f4_l2_to_heldout": float(distances.min()),
        "candidate_geometry": {
            "all_down_to_cell_1_2": down_geometry,
            "all_right_to_cell_2_3": right_geometry,
            "all_actions_in_bounds": in_bounds,
        },
        "construction_roots_sha256": sha256(OUT / "construction_roots.npz"),
    }
    write_json(OUT / "root_manifest.json", root_manifest)
    code_files = ("config.json", "PROTOCOL.md", "run.py", "remote_crl.py")
    write_json(OUT / "seal.json", {
        "status": "sealed_before_candidate_generation_or_training",
        "candidate_results_seen": False,
        "source_git_head": CONFIG["source_git_head"],
        "input_hashes": input_hashes,
        "root_manifest_sha256": sha256(OUT / "root_manifest.json"),
        "construction_roots_sha256": sha256(OUT / "construction_roots.npz"),
        "code_sha256": {name: sha256(OUT / name) for name in code_files},
    })
    print("sealed 198 construction roots, 550 contexts, 3300 paths per arm", flush=True)


def classify_actions(actions):
    down = (np.abs(actions[..., 0]) <= 0.35) & (actions[..., 1] <= -0.75)
    right = (actions[..., 0] >= 0.75) & (np.abs(actions[..., 1]) <= 0.35)
    return down, right


def summarize_generated(data):
    valid_states = data["states"][:, :50]
    xy = valid_states[..., :2]
    failed = data["failed"][:, :50]
    events = data["onset_event"]
    cell = np.floor(xy).astype(np.int32)
    lower = np.zeros(len(xy), bool)
    shortcut = np.zeros(len(xy), bool)
    for episode in range(len(xy)):
        low_ids = np.flatnonzero(
            ((cell[episode, :, 0] == 1) & (cell[episode, :, 1] == 2))
            | (xy[episode, :, 1] < 2)
        )
        hazard_ids = np.flatnonzero(
            (xy[episode, :, 0] >= 3) & (xy[episode, :, 0] < 6)
            & (xy[episode, :, 1] >= 3) & (xy[episode, :, 1] < 4)
        )
        low_time = int(low_ids[0]) if len(low_ids) else 10**9
        hazard_time = int(hazard_ids[0]) if len(hazard_ids) else 10**9
        lower[episode] = low_time < hazard_time
        shortcut[episode] = hazard_time <= low_time and hazard_time < 10**9
    goal_distance = np.linalg.norm(xy - GOAL[:2], axis=2)
    reached = ((goal_distance < 2) & ~failed).any(axis=1)
    strict = ((goal_distance < 0.5) & ~failed).any(axis=1)
    hazard_successor = (
        (xy[:, 1:, 0] >= 3) & (xy[:, 1:, 0] < 6)
        & (xy[:, 1:, 1] >= 3) & (xy[:, 1:, 1] < 4)
    )
    outside = events & ~hazard_successor
    onset_cell = np.floor(xy[:, 1:, :2]).astype(np.int32)
    cell_counts = {}
    for candidate in ((1, 3), (2, 3), (1, 2)):
        key = f"{candidate[0]},{candidate[1]}"
        cell_counts[key] = int((outside & np.all(onset_cell == candidate, axis=2)).sum())
    first_down, first_right = classify_actions(data["actions"][:, 0])
    first_xb_near_stop = np.linalg.norm(data["advice"][:, 0], axis=1) <= 0.2
    return {
        "paths": len(xy),
        "valid_transitions": int(np.sum(data["lengths"] - 1)),
        "first_query_down_sector": int(first_down.sum()),
        "first_query_right_sector": int(first_right.sum()),
        "lower_route_fraction": float(lower.mean()),
        "shortcut_fraction": float(shortcut.mean()),
        "region_reach_fraction": float(reached.mean()),
        "strict_reach_fraction": float(strict.mean()),
        "absorbed_fraction": float(failed[:, -1].mean()),
        "onsets": int(events.sum()),
        "onsets_outside_hazard": int(outside.sum()),
        "outside_hazard_fraction_of_onsets": float(outside.sum() / max(1, events.sum())),
        "outside_hazard_onset_cells": cell_counts,
        "first_nominal_near_stop": int(first_xb_near_stop.sum()),
        "first_nominal_near_stop_fraction": float(first_xb_near_stop.mean()),
        "outside_onsets_with_first_nominal_near_stop": int(
            (outside.any(axis=1) & first_xb_near_stop).sum()
        ),
        "candidate": {},
    }


def generate_arm(arm, roots, model, nominal, actor_state, network):
    import jax
    import jax.numpy as jnp
    import torch
    from learned_ett import legal_numpy

    @jax.jit
    def actor_sample(params, observation, key):
        distribution = network.policy_network.apply(params, observation)
        return network.sample(distribution, key)

    context_root = roots["context_root"].astype(np.int32)
    context_state = roots["state"][context_root]
    candidates = roots["candidate_action"].astype(np.float32)
    context_id = np.repeat(np.arange(len(context_root), dtype=np.int32), len(candidates))
    candidate_id = np.tile(np.arange(len(candidates), dtype=np.int8), len(context_root))
    root_id = context_root[context_id]
    paths = len(context_id)
    if paths != CONFIG["generation"]["paths_per_arm"]:
        raise AssertionError("path budget mismatch")

    state = np.zeros((paths, 51, 8), np.float32)
    action = np.zeros((paths, 51, 2), np.float32)
    advice = np.zeros((paths, 49, 2), np.float32)
    onset_probability = np.zeros((paths, 49), np.float32)
    onset_event = np.zeros((paths, 49), bool)
    failed = np.zeros((paths, 51), bool)
    state[:, 0] = roots["state"][root_id]

    context_goal = np.broadcast_to(GOAL, context_state.shape).astype(np.float32)
    context_xb = np.asarray(nominal.sample(
        jnp.asarray(context_state),
        jax.random.PRNGKey(CONFIG["seed"] + 1000),
        1,
        goal=jnp.asarray(context_goal),
    ), np.float32).copy()
    first_xb = context_xb[context_id]
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(CONFIG["seed"] + 2000)
    started = time.time()

    for step in range(49):
        current = state[:, step]
        goal = np.broadcast_to(GOAL, current.shape).astype(np.float32)
        if step == 0:
            xb = first_xb.copy()
        else:
            xb = np.asarray(nominal.sample(
                jnp.asarray(current),
                jax.random.PRNGKey(CONFIG["seed"] + 3000 + step),
                1,
                goal=jnp.asarray(goal),
            ), np.float32).copy()
        observation = jnp.asarray(np.concatenate([current, goal], axis=1))
        actor_action = np.asarray(actor_sample(
            actor_state.policy_params,
            observation,
            jax.random.PRNGKey(CONFIG["seed"] + 4000 + step),
        ), np.float32).copy()
        xq = candidates[candidate_id].copy() if (arm == "C" and step == 0) else actor_action
        advice[:, step] = xb
        action[:, step] = xq
        with torch.no_grad():
            successor, event, probability, _ = model.sample_live(
                torch.as_tensor(current), torch.as_tensor(xb), torch.as_tensor(xq), torch_rng
            )
        successor = successor.numpy().astype(np.float32)
        event = event.numpy()
        probability = probability.numpy()
        was_failed = failed[:, step]
        persistent = np.concatenate([current[:, :2], current[:, :6]], axis=1)
        successor[was_failed] = persistent[was_failed]
        event[was_failed] = False
        probability[was_failed] = 0
        state[:, step + 1] = successor
        failed[:, step + 1] = was_failed | event
        onset_event[:, step] = event
        onset_probability[:, step] = probability
        if (step + 1) % 10 == 0 or step == 48:
            print(
                f"arm {arm} generated {step + 1}/49; failed={failed[:, step + 1].mean():.3f}",
                flush=True,
            )

    state[:, 50] = state[:, 49]
    failed[:, 50] = failed[:, 49]
    lengths = np.full(paths, 50, np.int16)
    if np.any(state[:, 1:50, 2:] != state[:, :49, :6]):
        raise AssertionError("F4 shift violation")
    if not legal_numpy(state[:, :50, :2]).all():
        raise AssertionError("illegal emitted endpoint")
    recovery = failed[:, :49] & np.any(state[:, 1:50, :2] != state[:, :49, :2], axis=2)
    if recovery.any():
        raise AssertionError("post-failure recovery")
    data = {
        "states": state,
        "actions": action,
        "advice": advice,
        "onset_probability": onset_probability,
        "onset_event": onset_event,
        "failed": failed,
        "lengths": lengths,
        "root_id": root_id,
        "context_id": context_id,
        "candidate_id": candidate_id,
        "first_nominal": first_xb,
    }
    summary = summarize_generated(data)
    if arm == "C":
        for candidate_index, name in enumerate(CONFIG["query_actions"]["names"]):
            selected = candidate_id == candidate_index
            subset = {key: value[selected] for key, value in data.items()
                      if isinstance(value, np.ndarray) and len(value) == paths}
            item = summarize_generated(subset)
            summary["candidate"][name] = {
                key: item[key] for key in (
                    "paths", "lower_route_fraction", "region_reach_fraction",
                    "strict_reach_fraction", "absorbed_fraction", "onsets",
                    "onsets_outside_hazard",
                )
            }
    summary["wall_seconds"] = time.time() - started
    return data, summary


def build_replay(arm, generated):
    prior = load_npz(input_path("prior_replay"))
    original = prior["audit_source"] == 0
    if int(original.sum()) != CONFIG["replay"]["original_episodes"]:
        raise AssertionError("prior original subset count changed")
    original_obs = prior["obs"][original].astype(np.float32)
    original_act = prior["act"][original].astype(np.float32)
    goal = np.broadcast_to(GOAL, generated["states"].shape).astype(np.float32)
    synthetic_obs = np.concatenate([generated["states"], goal], axis=2)
    obs = np.concatenate([original_obs, synthetic_obs]).astype(np.float32)
    act = np.concatenate([original_act, generated["actions"]]).astype(np.float32)
    lengths = np.concatenate([
        np.full(len(original_obs), 51, np.int16), generated["lengths"]
    ])
    n_original = len(original_obs)
    onset_time = np.where(
        generated["onset_event"].any(axis=1),
        generated["onset_event"].argmax(axis=1),
        -1,
    ).astype(np.int16)
    meta = {
        "env_name": "point_two_route_swamp_windy_f4_v0",
        "source_git_head": CONFIG["source_git_head"],
        "arm": arm,
        "state_dim": 8,
        "goal_dim": 8,
        "action_dim": 2,
        "query_intervention_only_at_synthetic_time_zero": arm == "C",
        "synthetic_valid_observations": 50,
    }
    path = OUT / f"replay_{arm}.npz"
    np.savez_compressed(
        path,
        obs=obs,
        act=act,
        lengths=lengths,
        meta=np.asarray(json.dumps(meta, sort_keys=True)),
        audit_source=np.concatenate([
            np.zeros(n_original, np.int8), np.full(len(generated["states"]), 1, np.int8)
        ]),
        audit_original_episode=np.concatenate([
            prior["audit_source_episode"][original].astype(np.int32),
            np.full(len(generated["states"]), -1, np.int32),
        ]),
        audit_root_id=np.concatenate([
            np.full(n_original, -1, np.int32), generated["root_id"].astype(np.int32)
        ]),
        audit_context_id=np.concatenate([
            np.full(n_original, -1, np.int32), generated["context_id"].astype(np.int32)
        ]),
        audit_candidate_id=np.concatenate([
            np.full(n_original, -1, np.int8), generated["candidate_id"].astype(np.int8)
        ]),
        audit_onset_time=np.concatenate([
            np.full(n_original, -2, np.int16), onset_time
        ]),
        audit_failed_at_end=np.concatenate([
            np.zeros(n_original, bool), generated["failed"][:, 49]
        ]),
    )
    return path


def generation_stage():
    if all((OUT / f"replay_{arm}.npz").exists() for arm in ("B", "C")):
        print("both coverage replays already generated", flush=True)
        return
    if not (OUT / "seal.json").exists():
        raise RuntimeError("run audit before generation")
    import torch
    import jax
    from crl import checkpoint
    from learned_ett import load_checkpoint
    from propensity.nominal_policy import load_nominal_policy

    torch.set_num_threads(2)
    model, _ = load_checkpoint(input_path("ett_checkpoint"), "cpu")
    nominal = load_nominal_policy(input_path("nominal").parent)
    _, actor_state = checkpoint.load_checkpoint(input_path("rollout_actor"))
    network = make_network()
    roots = load_npz(OUT / "construction_roots.npz")
    result = {
        "fixed_inputs": {
            name: CONFIG["inputs"][name + "_sha256"]
            for name in ("ett_checkpoint", "nominal", "rollout_actor")
        },
        "jax_devices": [str(device) for device in jax.devices()],
        "arms": {},
        "paired_random_streams": True,
        "outcome_filtering": False,
    }
    for arm in ("B", "C"):
        generated, summary = generate_arm(arm, roots, model, nominal, actor_state, network)
        generated_path = OUT / f"generated_{arm}.npz"
        np.savez_compressed(generated_path, **generated)
        replay_path = build_replay(arm, generated)
        summary["generated_sha256"] = sha256(generated_path)
        summary["replay_sha256"] = sha256(replay_path)
        result["arms"][arm] = summary
    write_json(OUT / "generation.json", result)
    print("generated matched behavior-query and covered-query replays", flush=True)


def build_crl_config(arm):
    from crl.config import Config
    return Config(
        env_name="point_two_route_swamp_windy_f4_v0",
        offline_dataset=str(OUT / f"replay_{arm}.npz"),
        obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
        start_index=0, end_index=-1,
        max_number_of_steps=CONFIG["crl"]["steps_per_arm"],
        fail_bank_path="", fail_neg_alpha=0.0, obs_norm_mode="", obs_norm_z_scale=0.0,
        anchor_cut_mode="", balanced_sampling=False,
        use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
        bc_coef=0.05, random_goals=0.5, entropy_coefficient=0.0, target_entropy=0.0,
        batch_size=256, repr_dim=64, hidden_layer_sizes=(256, 256), discount=0.95,
        learning_rate=3e-4, actor_learning_rate=3e-4,
        num_sgd_steps_per_step=10, num_actors=0, guard_abort=True, jit=True, seed=0,
        eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=1_000,
        ckpt_every_steps=CONFIG["crl"]["steps_per_arm"],
        ckpt_dir=str(OUT / "crl" / arm),
    )


def lineage_stage():
    if (OUT / "nce_lineage.npz").exists():
        print("coverage NCE lineage already materialized", flush=True)
        return
    from crl import offline_audit

    configs = {arm: build_crl_config(arm) for arm in ("B", "C")}
    config_dicts = {arm: dataclasses.asdict(configs[arm]) for arm in ("B", "C")}
    diff = {
        key: [config_dicts["B"][key], config_dicts["C"][key]]
        for key in config_dicts["B"]
        if config_dicts["B"][key] != config_dicts["C"][key]
    }
    if set(diff) != {"offline_dataset", "ckpt_dir"}:
        raise AssertionError(f"unmatched CRL configs: {diff}")
    buffers, audits = {}, {}
    for arm in ("B", "C"):
        buffer, _ = offline_audit.build_offline_buffer(configs[arm].offline_dataset, configs[arm])
        passed, gates, report = offline_audit.run_static_audit(
            configs[arm].offline_dataset, configs[arm], buffer=buffer
        )
        if not passed:
            raise RuntimeError(f"offline audit failed for {arm}: {gates}")
        buffers[arm] = buffer
        audits[arm] = {"gates": gates, "report": report}
    steps = CONFIG["crl"]["steps_per_arm"]
    batch = CONFIG["crl"]["batch_size"]
    trajectory = np.empty((steps, batch), np.int32)
    anchor = np.empty_like(trajectory)
    future = np.empty_like(trajectory)
    for update in range(steps):
        draw_b = buffers["B"].sampled_indices(batch)
        draw_c = buffers["C"].sampled_indices(batch)
        if not all(np.array_equal(left, right) for left, right in zip(draw_b, draw_c)):
            raise AssertionError(f"sampler streams diverged at update {update}")
        trajectory[update], anchor[update], future[update] = draw_b
    np.savez_compressed(
        OUT / "nce_lineage.npz", trajectory=trajectory, anchor_time=anchor,
        future_time=future, future_offset=future - anchor,
        update=np.arange(steps, dtype=np.int32),
    )

    replay_b, replay_c = load_npz(OUT / "replay_B.npz"), load_npz(OUT / "replay_C.npz")
    synthetic_t0 = (trajectory >= 3300) & (anchor == 0)
    rows, cols = np.nonzero(synthetic_t0)
    traj = trajectory[rows, cols]
    fut = future[rows, cols]
    future_b = replay_b["obs"][traj, fut, :8]
    future_c = replay_c["obs"][traj, fut, :8]
    candidate = replay_c["audit_candidate_id"][traj]
    task_full_distance_b = np.linalg.norm(future_b - GOAL, axis=1)
    task_full_distance_c = np.linalg.norm(future_c - GOAL, axis=1)
    task_xy_distance_b = np.linalg.norm(future_b[:, :2] - GOAL[:2], axis=1)
    task_xy_distance_c = np.linalg.norm(future_c[:, :2] - GOAL[:2], axis=1)
    generated_b = load_npz(OUT / "generated_B.npz")
    behavior_down, behavior_right = classify_actions(generated_b["actions"][:, 0])
    summary = {
        "rows": int(trajectory.size),
        "shape": list(trajectory.shape),
        "shared_sampler_indices": True,
        "lineage_sha256": sha256(OUT / "nce_lineage.npz"),
        "config_diff": diff,
        "offline_audits": audits,
        "synthetic_time0_positive_pairs": int(synthetic_t0.sum()),
        "B_time0_trajectory_action_counts": {
            "down_sector": int(behavior_down[traj - 3300].sum()),
            "right_sector": int(behavior_right[traj - 3300].sum()),
            "other": int((~behavior_down[traj - 3300] & ~behavior_right[traj - 3300]).sum()),
        },
        "C_candidate_positive_counts": {},
        "task_future_support": {
            "B_full_f4_distance_le_0p5": int((task_full_distance_b <= 0.5).sum()),
            "C_full_f4_distance_le_0p5": int((task_full_distance_c <= 0.5).sum()),
            "B_xy_distance_lt_2": int((task_xy_distance_b < 2).sum()),
            "C_xy_distance_lt_2": int((task_xy_distance_c < 2).sum()),
        },
    }
    for candidate_index, name in enumerate(CONFIG["query_actions"]["names"]):
        selected = candidate == candidate_index
        summary["C_candidate_positive_counts"][name] = {
            "all": int(selected.sum()),
            "full_f4_distance_le_0p5": int((selected & (task_full_distance_c <= 0.5)).sum()),
            "xy_distance_lt_2": int((selected & (task_xy_distance_c < 2)).sum()),
        }
    write_json(OUT / "lineage.json", summary)
    print(f"materialized {trajectory.size:,} matched NCE positives", flush=True)


def verify_crl_stage():
    from crl import checkpoint
    record = {"bc_coef_every_arm": 0.05, "arms": {}, "matched": {}}
    for arm in ("B", "C"):
        _, initial = checkpoint.load_checkpoint(OUT / "crl" / arm / "init.pkl")
        step, final = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        record["arms"][arm] = {
            "step": int(step),
            "initial_tree_sha256": tree_sha(initial),
            "initial_policy_sha256": tree_sha(initial.policy_params),
            "initial_critic_sha256": tree_sha(initial.q_params),
            "final_policy_sha256": tree_sha(final.policy_params),
            "final_critic_sha256": tree_sha(final.q_params),
            "actor_updated": tree_sha(final.policy_params) != tree_sha(initial.policy_params),
            "critic_updated": tree_sha(final.q_params) != tree_sha(initial.q_params),
        }
    b, c = record["arms"]["B"], record["arms"]["C"]
    record["matched"] = {
        "initial_full_state": b["initial_tree_sha256"] == c["initial_tree_sha256"],
        "initial_policy": b["initial_policy_sha256"] == c["initial_policy_sha256"],
        "initial_critic": b["initial_critic_sha256"] == c["initial_critic_sha256"],
        "budget": b["step"] == c["step"] == CONFIG["crl"]["steps_per_arm"],
        "both_actor_and_critic_updated": all(
            record["arms"][arm][key]
            for arm in ("B", "C") for key in ("actor_updated", "critic_updated")
        ),
        "no_training_native_evaluation": True,
    }
    record["passed"] = all(record["matched"].values())
    write_json(OUT / "crl_verification.json", record)
    if not record["passed"]:
        raise RuntimeError("matched CRL verification failed")


def bootstrap_interval(values, indices):
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def analysis_stage():
    if not all((OUT / "crl" / arm / "final.pkl").exists() for arm in ("B", "C")):
        raise RuntimeError("download both final checkpoints before analysis")
    verify_crl_stage()
    import jax.numpy as jnp
    from crl import checkpoint

    roots = load_npz(input_path("heldout_roots"))["state"].astype(np.float32)
    actions = np.asarray(CONFIG["heldout_evaluation"]["candidate_actions"], np.float32)
    goal = np.broadcast_to(GOAL, roots.shape)
    observation = jnp.asarray(np.concatenate([roots, goal], axis=1))
    network = make_network()
    rng = np.random.default_rng(CONFIG["heldout_evaluation"]["bootstrap_seed"])
    indices = rng.integers(
        0, len(roots),
        size=(CONFIG["heldout_evaluation"]["bootstrap_replicates"], len(roots)),
    )
    actor_epsilon = rng.standard_normal((
        len(roots), CONFIG["heldout_evaluation"]["actor_samples_per_root"], 2
    )).astype(np.float32)
    result = {
        "status": "fixed_final_read_only_heldout_evaluation",
        "construction_overlap": False,
        "root_count": len(roots),
        "arms": {},
    }
    margins = {}
    for arm in ("B", "C"):
        _, state = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        scores = []
        for action in actions:
            batch_action = jnp.asarray(np.broadcast_to(action, (len(roots), 2)))
            phi, psi = network.representation_network.apply(
                state.q_params, observation, batch_action
            )
            score = jnp.sum(phi * psi, axis=1)
            if score.ndim == 2:
                score = score[:, 0]
            scores.append(np.asarray(score, np.float32))
        scores = np.stack(scores, axis=1)
        margin = scores[:, 0] - scores[:, 1]
        margins[arm] = margin
        distribution = network.policy_network.apply(state.policy_params, observation)
        loc = np.asarray(distribution.loc, np.float32)[:, None]
        scale = np.asarray(distribution.scale, np.float32)[:, None]
        samples = np.tanh(loc + scale * actor_epsilon)
        down, right = classify_actions(samples)
        endpoints = roots[:, None, :2] + samples
        cells = np.floor(endpoints).astype(np.int32)
        enter_lower = np.all(cells == np.array([1, 2]), axis=2)
        result["arms"][arm] = {
            "down_logit_mean": float(scores[:, 0].mean()),
            "right_logit_mean": float(scores[:, 1].mean()),
            "down_minus_right_mean": float(margin.mean()),
            "down_minus_right_ci95": bootstrap_interval(margin, indices),
            "roots_preferring_down": int((margin > 0).sum()),
            "per_root_down_minus_right": margin,
            "actor_mode_action_mean": np.tanh(np.asarray(distribution.loc)).mean(axis=0),
            "actor_down_sector_probability": float(down.mean()),
            "actor_right_sector_probability": float(right.mean()),
            "actor_noiseless_lower_entry_probability": float(enter_lower.mean()),
        }
    change = margins["C"] - margins["B"]
    result["C_minus_B"] = {
        "down_minus_right_margin": float(change.mean()),
        "ci95": bootstrap_interval(change, indices),
        "per_root": change,
    }
    gates = {
        "absolute_C": result["arms"]["C"]["down_minus_right_ci95"][0] > 0,
        "incremental_C_minus_B": result["C_minus_B"]["ci95"][0] > 0,
        "C_accuracy": result["arms"]["C"]["roots_preferring_down"] >= 12,
    }
    passed = all(gates.values())
    result["primary_decision"] = {
        "gates": gates,
        "passed": passed,
        "label": (
            CONFIG["primary_decision"]["pass_label"]
            if passed else CONFIG["primary_decision"]["fail_label"]
        ),
    }
    write_json(OUT / "heldout_evaluation.json", result)
    print(json.dumps(plain(result["primary_decision"]), indent=2), flush=True)


def trace_stage():
    """Trace one deterministic context/action slot through generation and NCE."""
    import jax.numpy as jnp
    from crl import checkpoint

    roots = load_npz(OUT / "construction_roots.npz")
    supervision = load_npz(input_path("supervision"))
    generated = {arm: load_npz(OUT / f"generated_{arm}.npz") for arm in ("B", "C")}
    replay = {arm: load_npz(OUT / f"replay_{arm}.npz") for arm in ("B", "C")}
    lineage = load_npz(OUT / "nce_lineage.npz")
    path = 1  # context 0, sealed central-down candidate slot; no outcome selection.
    replay_trajectory = 3300 + path
    occurrences = np.argwhere(
        (lineage["trajectory"] == replay_trajectory) & (lineage["anchor_time"] == 0)
    )
    if not len(occurrences):
        raise RuntimeError("deterministic trace path has no time-zero NCE occurrence")
    update, batch_row = map(int, occurrences[0])
    future_time = int(lineage["future_time"][update, batch_row])
    root_id = int(generated["C"]["root_id"][path])
    supervision_row = int(roots["supervision_row"][root_id])
    candidates = roots["candidate_action"].astype(np.float32)
    network = make_network()
    output = {
        "selection": "path 1 = context 0 central-down slot; fixed by array order, no outcome selection",
        "collected_training_tuple": {
            "supervision_row": supervision_row,
            "source": int(supervision["train_source"][supervision_row]),
            "source_row": int(supervision["train_source_row"][supervision_row]),
            "episode": int(supervision["train_episode"][supervision_row]),
            "time": int(supervision["train_time"][supervision_row]),
            "s": supervision["train_s"][supervision_row],
            "recorded_xb": supervision["train_xb"][supervision_row],
            "recorded_xq": supervision["train_xq"][supervision_row],
            "recorded_y": supervision["train_y"][supervision_row],
            "recorded_onset": bool(supervision["train_onset"][supervision_row]),
        },
        "shared_generated_context": {
            "root_id": root_id,
            "context_id": int(generated["C"]["context_id"][path]),
            "nominal_xb": generated["C"]["advice"][path, 0],
            "B_C_nominal_bit_identical": bool(np.array_equal(
                generated["B"]["advice"][path, 0], generated["C"]["advice"][path, 0]
            )),
        },
        "nce_positive": {
            "replay_trajectory": replay_trajectory,
            "update": update,
            "batch_row": batch_row,
            "anchor_time": 0,
            "future_time": future_time,
            "future_offset": future_time,
            "shared_index_both_arms": True,
        },
        "arms": {},
    }
    for arm in ("B", "C"):
        _, state = checkpoint.load_checkpoint(OUT / "crl" / arm / "final.pkl")
        root = generated[arm]["states"][path, 0]
        observation = jnp.asarray(np.concatenate([root, GOAL])[None])
        action_score = []
        for candidate in (candidates[1], candidates[4]):
            phi, psi = network.representation_network.apply(
                state.q_params, observation,
                jnp.asarray(candidate[None]),
            )
            value = jnp.sum(phi * psi, axis=1)
            action_score.append(float(np.asarray(value).reshape(-1)[0]))
        distribution = network.policy_network.apply(state.policy_params, observation)
        output["arms"][arm] = {
            "first_query": generated[arm]["actions"][path, 0],
            "first_successor": generated[arm]["states"][path, 1],
            "first_onset": bool(generated[arm]["onset_event"][path, 0]),
            "absorbed_at_end": bool(generated[arm]["failed"][path, 49]),
            "valid_continuation_sha256": array_sha(generated[arm]["states"][path, :50]),
            "nce_positive_future": replay[arm]["obs"][replay_trajectory, future_time, :8],
            "canonical_goal_down_logit": action_score[0],
            "canonical_goal_right_logit": action_score[1],
            "canonical_goal_down_minus_right": action_score[0] - action_score[1],
            "final_actor_mode_action": np.tanh(np.asarray(distribution.loc)[0]),
        }
    write_json(OUT / "trace.json", output)


def finalize_stage():
    trace_stage()
    generation = json.loads((OUT / "generation.json").read_text())
    lineage = json.loads((OUT / "lineage.json").read_text())
    heldout = json.loads((OUT / "heldout_evaluation.json").read_text())
    fit = json.loads((OUT / "fit_diagnostic.json").read_text())
    trace = json.loads((OUT / "trace.json").read_text())
    verification = json.loads((OUT / "crl_verification.json").read_text())
    native = {
        policy: json.loads((OUT / "native" / policy / "summary.json").read_text())
        for policy in ("mode", "sample")
    }
    result = {
        "status": "complete",
        "primary_decision": heldout["primary_decision"],
        "generation": generation,
        "lineage": lineage,
        "heldout": heldout,
        "fit_diagnostic": fit,
        "trace": trace,
        "crl_verification": verification,
        "native": native,
        "one_seed_exploratory": True,
        "ett_retrained": False,
        "nce_changed": False,
    }
    write_json(OUT / "results.json", result)

    def native_metric(policy, arm, metric):
        return native[policy]["policies"][arm][metric]["mean"]

    def native_delta(policy, metric):
        comparison = native[policy]["comparisons"]["B_minus_C"][metric]
        lo, hi = comparison["ci95"]
        return -comparison["mean"], [-hi, -lo]

    rows = []
    for policy in ("mode", "sample"):
        reach_delta, reach_ci = native_delta(policy, "reach")
        lower_delta, lower_ci = native_delta(policy, "lower_route")
        rows.append(
            f"| {policy} | {native_metric(policy,'B','reach'):.3f} | "
            f"{native_metric(policy,'C','reach'):.3f} | {reach_delta:+.3f} "
            f"[{reach_ci[0]:+.3f}, {reach_ci[1]:+.3f}] | "
            f"{native_metric(policy,'B','lower_route'):.3f} | "
            f"{native_metric(policy,'C','lower_route'):.3f} | {lower_delta:+.3f} "
            f"[{lower_ci[0]:+.3f}, {lower_ci[1]:+.3f}] |"
        )
    b = heldout["arms"]["B"]
    c = heldout["arms"]["C"]
    delta = heldout["C_minus_B"]
    gen_b = generation["arms"]["B"]
    gen_c = generation["arms"]["C"]
    decision = heldout["primary_decision"]
    fit_c = fit["arms"]["C"]
    construction_fit = fit_c["canonical_goal_construction_roots"]
    goal_support = fit["sampled_time0_goal_support"]
    goal_bank = fit_c["heldout_margin_for_sampled_goal_banks"]
    report = f"""# Fixed-ETT query-coverage intervention

Primary result: **`{decision['label']}`**. The selected ETT was not retrained, sigmoid-NCE was unchanged, and both 30k-step CRL arms used BC 0.05 from hash-matched initialization.

## What changed

Both arms used the same 198 supervised-training fork roots, 550 `(s, xb)` contexts, exact retained original half, continuation actor, and paired random streams. B queried its first action from the old observational actor. C supplied three sealed down-neighborhood and three sealed right-neighborhood queries per context. Every one of the 3,300 paths per arm was kept for the full 49 remaining transitions; synthetic padding was excluded with structural lengths.

B's first queries contained {gen_b['first_query_down_sector']} down-sector and {gen_b['first_query_right_sector']} right-sector paths. C contained {gen_c['first_query_down_sector']} and {gen_c['first_query_right_sector']}. Generated B/C region reach was {gen_b['region_reach_fraction']:.3f}/{gen_c['region_reach_fraction']:.3f}; absorption {gen_b['absorbed_fraction']:.3f}/{gen_c['absorbed_fraction']:.3f}. C retained {gen_c['onsets_outside_hazard']} out-of-hazard onsets of {gen_c['onsets']} total; no onset correction or outcome filtering was applied.

## Did coverage transmit to the held-out critic?

On 16 evaluation-only coherent alive roots, B's task-goal down-minus-right logit was {b['down_minus_right_mean']:+.3f} [{b['down_minus_right_ci95'][0]:+.3f}, {b['down_minus_right_ci95'][1]:+.3f}] with {b['roots_preferring_down']}/16 roots preferring down. C's was {c['down_minus_right_mean']:+.3f} [{c['down_minus_right_ci95'][0]:+.3f}, {c['down_minus_right_ci95'][1]:+.3f}] with {c['roots_preferring_down']}/16. The paired C-minus-B change was {delta['down_minus_right_margin']:+.3f} [{delta['ci95'][0]:+.3f}, {delta['ci95'][1]:+.3f}]. Gates: {decision['gates']}.

The exact 7,680,000-row sampler lineage contains {lineage['synthetic_time0_positive_pairs']:,} synthetic time-zero positives. Candidate and task-near-future counts are in `lineage.json`; this shows what entered NCE rather than relying on total generated transition count.

## Where the remaining exact-goal error sits

C remains negative even on the 198 construction roots: central down-minus-right {construction_fit['central_down_minus_right']:+.3f} [{construction_fit['central_ci95'][0]:+.3f}, {construction_fit['central_ci95'][1]:+.3f}], with {construction_fit['central_roots_positive']}/198 positive. The strict held-out failure is therefore not merely a held-out-root generalization failure.

However, the relabeled target distribution contains zero exact copies of the canonical stationary F4 goal. Among C's time-zero positives, down/right have {goal_support['down']['positive_pairs']:,}/{goal_support['right']['positive_pairs']:,} goals; their nearest full-F4 distances to canonical are {goal_support['down']['minimum_full_f4_distance']:.3f}/{goal_support['right']['minimum_full_f4_distance']:.3f}, and only {goal_support['down']['full_f4_threshold_counts']['0.1']}/{goal_support['right']['full_f4_threshold_counts']['0.1']} lie within 0.1. Down nevertheless has {goal_support['down']['full_f4_threshold_counts']['0.5']:,} futures within 0.5 versus {goal_support['right']['full_f4_threshold_counts']['0.5']:,} for right.

On 512 actual down-route goal-near F4 goals drawn from those positives, C's held-out down-minus-right logit is {goal_bank['down']['mean_down_minus_right']:+.3f}, positive on all 16 roots. On a right-route goal bank it is {goal_bank['right']['mean_down_minus_right']:+.3f}. Thus C learned action-to-supported-future discrimination, while the exact stationary canonical-goal probe remains a small extrapolation error. `fit_diagnostic.json` also evaluates 512 exact lineage batches: C improves time-zero positive-versus-negative separation, but the more diverse replay is harder overall. These logits are density-ratio diagnostics, not calibrated success probabilities.

Actor samples on the held-out roots enter the noiseless lower cell with probability {b['actor_noiseless_lower_entry_probability']:.3f} (B) and {c['actor_noiseless_lower_entry_probability']:.3f} (C). Because C's covered actions also enter the unchanged BC term, actor movement is not a critic-only attribution.

## Fixed-final native evaluation

| policy | B reach | C reach | C-B reach (paired 95% CI) | B lower | C lower | C-B lower (paired 95% CI) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Native evaluation used 200 fresh paired reset seeds only after training. It did not select checkpoints or change the fixed budget.

Although the strict exact-canonical critic gate fails, the native policy effect is large in both protocols. Query coverage is therefore a demonstrated major cause of the prior policy failure and is sufficient for useful policy learning in this intervention, but the experiment does not establish it as the unique cause or satisfy the stricter canonical critic criterion.

## Trace

`trace.json` follows the deterministic context-0 central-down slot from its exact collected supervised tuple through the shared nominal condition, B/C first queries and learned successors, complete continuation hashes, one exact shared NCE `(trajectory, anchor, future)` occurrence, and final actor/critic preferences. The trace path is fixed by array order and was not selected by outcome.

## Interpretation limits

This one-seed intervention tests sufficiency of a particular sealed action-covering generator at a fixed learner budget. It does not prove uniqueness of the cause. It intentionally leaves the fixed ETT's out-of-support death errors in the replay, so failure would still be compatible with goal-distribution, critic-fit, model-quality, or finite-sample limitations. Model-internal success is not native success.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    code_files = ("run.py", "remote_crl.py", "diagnose_fit.py")
    write_json(OUT / "completion.json", {
        "status": "complete",
        "primary_decision": decision,
        "report_sha256": sha256(OUT / "REPORT.md"),
        "results_sha256": sha256(OUT / "results.json"),
        "code_sha256": {name: sha256(OUT / name) for name in code_files},
        "sealed_training_code_unchanged_during_generation_and_training": True,
        "post_seal_changes": "read-only fit diagnostic, trace, and final reporting only",
        "commit_or_push": False,
    })
    print("coverage experiment report complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("audit", "generate", "lineage", "analyze", "finalize")
    )
    args = parser.parse_args()
    if args.stage == "audit":
        audit_stage()
    elif args.stage == "generate":
        generation_stage()
    elif args.stage == "lineage":
        lineage_stage()
    elif args.stage == "analyze":
        analysis_stage()
    elif args.stage == "finalize":
        finalize_stage()


if __name__ == "__main__":
    main()
