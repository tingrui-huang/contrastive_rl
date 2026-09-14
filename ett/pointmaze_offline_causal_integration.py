"""Strictly offline matched observational-vs-ETT future integration.

The production objective is unchanged sigmoid NCE.  This driver changes only
the conditional positive-future source at matched recorded anchors, then runs
the unchanged offline actor objective.  It deliberately has no environment,
simulator, native-transition, or oracle imports.
"""
from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from crl import checkpoint, networks
from crl.losses import Transition
from ett import pointmaze_early_pilot as early
from ett import pointmaze_phase_sampling as phase
from ett import pointmaze_region_pilot as region
from ett import pointmaze_update_reference as update_reference
from ett.finite_crl import write, sha


DEFAULT_OUT = Path("outputs/pointmaze_offline_causal_integration_20260915_v1")
PROTOCOL_SOURCE = Path("notes/pointmaze_offline_causal_integration.md")
DATASET = Path("artifacts/f4_p30_server_30076/results/datasets/swamp_windy_f4_merged_s0.npz")
DATASET_MANIFEST = DATASET.with_name(DATASET.name + ".manifest.json")
INITIAL = Path("artifacts/f4_p30_server_30076/results/runs/f4_p30_sweep/p30_a0_a01_a03_s0_s1/alpha0_seed0/final.pkl")
INITIAL_PROVENANCE = INITIAL.with_name("arm_provenance.json")
NOMINAL = Path("artifacts/nominal_policy/f4_p30_expert_only_mdn_k5_s0/best.pkl")
NOMINAL_PROVENANCE = NOMINAL.with_name("expert_population_manifest.json")
DIAGONAL = Path("artifacts/ett_distribution_matching/f4_p30_s01_guarded/s0_L0p25_lambda0/final.pkl")
DIAGONAL_HISTORY = DIAGONAL.with_name("training_history.json")
DIAGONAL_REPORT = DIAGONAL.parents[1] / "REPORT.md"
NOMINAL_CONFIG = NOMINAL.with_name("config.json")
ROUTING_MANIFEST = Path("artifacts/ett_rollout_return/residual6_s01/config.json")
TEST_SOURCE = Path("scripts/test_pointmaze_offline_causal_integration.py")
PAPER = Path("D:/Tue/Research/Contrastive_Causal_RL/526_Causal_Contrastive (1).pdf")

EXPECTED = {
    str(DATASET): "83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c",
    str(INITIAL): "ea8a71d3cb8d963259a54d47250b454462dae2ae62f03a55c370ca81a2c8ec54",
    str(NOMINAL): "6376e60185aa616c2d09c5e0250d762cde2fb845c5042488e64910fb3b745f23",
    str(DIAGONAL): "9e9e1c56387d446b5788612e27e5b7f7743e120186bebc90a78fc6774476184e",
    str(DATASET_MANIFEST): "591dd4bc2983b0ca96a050538357e3c78805dd5ab40e99e9313e65b551bd933e",
    str(INITIAL_PROVENANCE): "5a7f45b8a8142c56870bea7bdce35b7564695c3179212359ac19f1dc52f89f57",
    str(NOMINAL_PROVENANCE): "2dabe55a4bcb474c7b1aa49e1b32d4a28759e38d9299ebb72dff1b018a419cee",
    str(NOMINAL_CONFIG): "9d2b96d13307638494ff6c6432b2ac639d58d584000b40f6879ef179e15b4379",
    str(DIAGONAL_HISTORY): "a06b1faef70d5d37ef07290f7c78cb1c27fcabd7a0affb99366093f63fc02553",
    str(ROUTING_MANIFEST): "fd83efeca7f5bc3c5561e7ed32b91758dcfc48c0a55f6fca15cd712ed6db1969",
}
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
PROBES = np.array([[0., -1.], [-.1, -.9], [0., -.9], [.1, -.9],
                   [1., 0.], [.9, -.1], [.9, 0.], [.9, .1]], np.float32)

CONFIG = {
    "experiment": "strictly_offline_sampling_based_causal_contrastive",
    "base_commit": "b79f1a62cf093dad303ef268b7e75a5b0f5d336e",
    "seed": 541000000,
    "discount": .95,
    "batch_size": 256,
    "fork_region": "1<=x<2 and 3<=y<4",
    "down": "ay<=-0.5 and abs(ay)>=abs(ax)",
    "right": "ax>=0.5 and abs(ax)>abs(ay)",
    "ett_roots": {"ordinary_nonfork": 48, "down": 24, "right": 24},
    "ett_root_repeats": 4,
    "auxiliary_prefit_steps": 1000,
    "auxiliary_refresh_steps": 400,
    "ett_updates": 3,
    "ett_directions": 4,
    "ett_queries": 32,
    "ett_successors": 4,
    "ett_actor_draws": 4,
    "ett_sigma": {"diagonal": .01, "response": .1},
    "ett_rate": {"diagonal": .01, "response": .5},
    "ett_step_cap": {"diagonal": .03, "response": .1},
    "ett_lambda": 1.,
    "diagonal_tolerance": .02,
    "train_cache": {"ordinary_nonfork": 2048, "down": 1024, "right": 1024},
    "validation_cache": {"ordinary_nonfork": 512, "down": 256, "right": 256},
    "production_batch": {"ordinary_nonfork": 128, "down": 64, "right": 64},
    "critic_updates": 400,
    "actor_updates": 1000,
    "critic_learning_rate": 3e-4,
    "actor_learning_rate": 3e-4,
    "adam_eps": 1e-7,
    "tau": .005,
    "bc_coef": .5,
    "random_goals": .5,
    "entropy_coefficient": 0.,
    "eval_nce_batches": 32,
    "fork_actor_samples": 128,
    "local_gradient_samples": 8,
    "away_rows": 1024,
    "away_actor_samples": 64,
    "gradient_batches": 4,
    "bootstrap_replicates": 2000,
    "model_output_cap": 480000,
    "environment_calls": 0,
    "native_steps": 0,
    "final_checkpoint_rule": "fixed final iterate only",
    "failure_support": "none: no explicit learned failure latent or defensible off-diagonal entry probability",
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def tree_sha(tree) -> str:
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(leaf)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def tree_norm(tree) -> float:
    return float(np.sqrt(sum(np.sum(np.asarray(x, np.float64) ** 2)
                             for x in jax.tree_util.tree_leaves(tree))))


def tree_delta_norm(a, b) -> float:
    return tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, a, b))


def json_default(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return value.as_posix()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class Ledger:
    def __init__(self, out: Path):
        self.out = out
        self.data = {"charged": 0, "cap": CONFIG["model_output_cap"], "entries": []}

    def add(self, number, purpose):
        number = int(number)
        if self.data["charged"] + number > self.data["cap"]:
            raise RuntimeError("sealed model-output cap exhausted")
        self.data["charged"] += number
        self.data["entries"].append({"purpose": str(purpose), "outputs": number})
        write_json(self.out / "model_ledger.json", self.data)


def load_eligible():
    for path, digest in EXPECTED.items():
        assert file_sha(Path(path)) == digest, path
    manifest = read_json(DATASET_MANIFEST)
    actor_prov = read_json(INITIAL_PROVENANCE)
    nominal_prov = read_json(NOMINAL_PROVENANCE)
    diagonal_history = read_json(DIAGONAL_HISTORY)
    assert manifest["content_sha256"] == actor_prov["dataset_content_sha256"]
    assert actor_prov["code_commit"] == "4f3da3f1e94cbad647bc56b0f24e4171776090ee"
    assert actor_prov["bank"] is None and actor_prov["alpha"] == 0.
    assert nominal_prov["selection_uses_outcome_or_hidden_state"] is False
    assert nominal_prov["dataset_sha256"] == EXPECTED[str(DATASET)]
    # Lambda-zero training history has total loss exactly equal to diagonal NLL.
    assert all(abs(r["total_loss"] - r["diagonal_training_nll"]) < 1e-9
               for r in diagonal_history)
    with np.load(DATASET, allow_pickle=False) as z:
        obs, act = np.asarray(z["obs"]), np.asarray(z["act"])
    assert obs.shape == (6600, 51, 16) and act.shape == (6600, 51, 2)
    assert np.isfinite(obs).all() and np.isfinite(act[:, :50]).all()
    np.testing.assert_array_equal(obs[:, 1:, 2:8], obs[:, :-1, :6])
    np.testing.assert_array_equal(obs[:, :, 8:], np.broadcast_to(GOAL, obs[:, :, 8:].shape))
    step, initial = checkpoint.load_checkpoint(INITIAL)
    assert step == 150000
    return obs, act, initial, manifest, actor_prov, nominal_prov


def split_ids():
    heldout = np.random.default_rng(0).permutation(6600)[:660].astype(np.int32)
    train = np.setdiff1d(np.arange(6600, dtype=np.int32), heldout)
    assert len(train) == 5940 and len(heldout) == 660
    assert not np.intersect1d(train, heldout).size
    return train, heldout


def fork_mask(state):
    xy = np.asarray(state)[..., :2]
    return ((xy[..., 0] >= 1.) & (xy[..., 0] < 2.) &
            (xy[..., 1] >= 3.) & (xy[..., 1] < 4.))


def direction_masks(action):
    action = np.asarray(action)
    down = ((action[..., 1] <= -.5) &
            (np.abs(action[..., 1]) >= np.abs(action[..., 0])))
    right = ((action[..., 0] >= .5) &
             (np.abs(action[..., 0]) > np.abs(action[..., 1])))
    assert not np.any(down & right)
    return down, right


def candidates(obs, act, ids):
    episode, time_index = np.indices((len(ids), 50))
    episode = ids[episode.ravel()]
    time_index = time_index.ravel().astype(np.int32)
    fm = fork_mask(obs[episode, time_index])
    down, right = direction_masks(act[episode, time_index])
    rows = np.column_stack([episode, time_index]).astype(np.int32)
    return {"ordinary_nonfork": rows[~fm], "down": rows[fm & down],
            "right": rows[fm & right], "fork": rows[fm]}


def sample_rows(rng, rows, count, replace=None):
    if replace is None:
        replace = count > len(rows)
    index = rng.choice(len(rows), count, replace=replace)
    return rows[index]


def distinct_episode_roots(rng, groups, counts):
    selected, used = {}, set()
    for name in ("down", "right", "ordinary_nonfork"):
        rows = groups[name][rng.permutation(len(groups[name]))]
        keep = []
        for row in rows:
            if int(row[0]) in used:
                continue
            keep.append(row)
            used.add(int(row[0]))
            if len(keep) == counts[name]:
                break
        assert len(keep) == counts[name]
        selected[name] = np.asarray(keep, np.int32)
    return np.concatenate([selected["ordinary_nonfork"], selected["down"], selected["right"]]), \
        np.repeat(np.array([0, 1, 2], np.int8),
                  [counts["ordinary_nonfork"], counts["down"], counts["right"]])


def draw_offsets(rng, lengths):
    lengths = np.asarray(lengths, np.int32)
    arange = np.arange(1, 51, dtype=np.int32)
    valid = arange[None, :] <= lengths[:, None]
    logits = np.where(valid, arange[None, :] * math.log(CONFIG["discount"]), -np.inf)
    u = rng.uniform(size=logits.shape).clip(1e-20, 1.)
    offsets = arange[np.argmax(logits - np.log(-np.log(u)), axis=1)]
    assert np.all((offsets >= 1) & (offsets <= lengths))
    return offsets.astype(np.int32)


def draw_ordinary_replay(rng, ids, batches):
    e = np.empty((batches, 256), np.int32)
    t = np.empty_like(e); f = np.empty_like(e)
    for u in range(batches):
        e[u] = ids[rng.integers(0, len(ids), 256)]
        t[u] = rng.integers(0, 50, 256)
        f[u] = t[u] + draw_offsets(rng, 50 - t[u])
    return e, t, f


def draw_pool_batches(rng, source, lengths, batches):
    slots = [np.flatnonzero(source == k) for k in (0, 1, 2)]
    ids = np.empty((batches, 256), np.int32)
    offsets = np.empty_like(ids)
    row_source = np.empty((batches, 256), np.int8)
    for u in range(batches):
        chosen = np.concatenate([rng.choice(slots[0], 128), rng.choice(slots[1], 64),
                                 rng.choice(slots[2], 64)]).astype(np.int32)
        src = np.repeat(np.array([0, 1, 2], np.int8), [128, 64, 64])
        off = draw_offsets(rng, lengths[chosen])
        order = rng.permutation(256)
        ids[u], offsets[u], row_source[u] = chosen[order], off[order], src[order]
    return ids, offsets, row_source


def pool_concentration(rows):
    key = rows[:, 0].astype(np.int64) * 51 + rows[:, 1]
    _, count = np.unique(key, return_counts=True)
    p = count / count.sum()
    return {"draws": int(len(rows)), "unique_anchors": int(len(count)),
            "unique_episodes": int(len(np.unique(rows[:, 0]))),
            "duplicate_fraction": float(1 - len(count) / len(rows)),
            "max_multiplicity": int(count.max()),
            "effective_anchor_count": float(1 / np.sum(p ** 2))}


def build_plan(obs, act, train, heldout):
    rng = np.random.default_rng(CONFIG["seed"])
    train_groups, val_groups = candidates(obs, act, train), candidates(obs, act, heldout)
    ett_rows, ett_source = distinct_episode_roots(rng, train_groups, CONFIG["ett_roots"])

    def make_pool(groups, spec):
        rows = np.concatenate([sample_rows(rng, groups[name], spec[name])
                               for name in ("ordinary_nonfork", "down", "right")])
        source = np.repeat(np.array([0, 1, 2], np.int8),
                           [spec["ordinary_nonfork"], spec["down"], spec["right"]])
        order = rng.permutation(len(rows))
        return rows[order], source[order]

    train_pool, train_source = make_pool(train_groups, CONFIG["train_cache"])
    val_pool, val_source = make_pool(val_groups, CONFIG["validation_cache"])
    train_ids, train_offsets, train_batch_source = draw_pool_batches(
        rng, train_source, 50 - train_pool[:, 1], CONFIG["critic_updates"])
    eval_ids, eval_offsets, eval_batch_source = draw_pool_batches(
        rng, val_source, 50 - val_pool[:, 1], CONFIG["eval_nce_batches"])
    actor = draw_ordinary_replay(rng, train, CONFIG["actor_updates"])
    actor_keys = np.asarray(jax.random.split(jax.random.PRNGKey(CONFIG["seed"] + 1),
                                             CONFIG["actor_updates"]), np.uint32)

    ctx = []
    for ep in heldout:
        times = np.flatnonzero(fork_mask(obs[ep, :50]))
        if len(times): ctx.append((ep, times[0]))
    ctx = np.asarray(ctx, np.int32)
    ctx_future = ctx[:, 1] + draw_offsets(rng, 50 - ctx[:, 1])
    all_val = np.column_stack([np.repeat(heldout, 50), np.tile(np.arange(50), len(heldout))]).astype(np.int32)
    away = all_val[~fork_mask(obs[all_val[:, 0], all_val[:, 1]])]
    away = sample_rows(rng, away, CONFIG["away_rows"], replace=False)
    away_future = away[:, 1] + draw_offsets(rng, 50 - away[:, 1])
    fork_eps = rng.normal(size=(len(ctx), CONFIG["fork_actor_samples"], 2)).astype(np.float32)
    away_eps = rng.normal(size=(len(away), CONFIG["away_actor_samples"], 2)).astype(np.float32)

    teacher_train = np.intersect1d(train, np.arange(1200, 6000, dtype=np.int32))
    teacher_val = np.intersect1d(heldout, np.arange(1200, 6000, dtype=np.int32))
    diagonal_train = sample_rows(rng, np.column_stack([
        np.repeat(teacher_train, 50), np.tile(np.arange(50), len(teacher_train))]), 512, False)
    diagonal_val = sample_rows(rng, np.column_stack([
        np.repeat(teacher_val, 50), np.tile(np.arange(50), len(teacher_val))]), 512, False)

    plan = {
        "ett_episode": ett_rows[:, 0], "ett_time": ett_rows[:, 1], "ett_source": ett_source,
        "train_pool_episode": train_pool[:, 0], "train_pool_time": train_pool[:, 1],
        "train_pool_source": train_source,
        "validation_pool_episode": val_pool[:, 0], "validation_pool_time": val_pool[:, 1],
        "validation_pool_source": val_source,
        "critic_pool_id": train_ids, "critic_offset": train_offsets,
        "critic_source": train_batch_source,
        "eval_pool_id": eval_ids, "eval_offset": eval_offsets,
        "eval_source": eval_batch_source,
        "actor_episode": actor[0], "actor_time": actor[1], "actor_future": actor[2],
        "actor_keys": actor_keys,
        "fork_context_episode": ctx[:, 0], "fork_context_time": ctx[:, 1],
        "fork_context_future": ctx_future,
        "away_episode": away[:, 0], "away_time": away[:, 1], "away_future": away_future,
        "fork_eps": fork_eps, "away_eps": away_eps, "action_probes": PROBES,
        "diagonal_train_episode": diagonal_train[:, 0], "diagonal_train_time": diagonal_train[:, 1],
        "diagonal_validation_episode": diagonal_val[:, 0], "diagonal_validation_time": diagonal_val[:, 1],
    }
    coverage = {
        "train_candidates": {k: {"anchors": int(len(v)), "episodes": int(len(np.unique(v[:, 0])))}
                             for k, v in train_groups.items()},
        "validation_candidates": {k: {"anchors": int(len(v)), "episodes": int(len(np.unique(v[:, 0])))}
                                  for k, v in val_groups.items()},
        "ett_roots": pool_concentration(ett_rows),
        "train_cache": {str(k): pool_concentration(train_pool[train_source == k]) for k in (0, 1, 2)},
        "validation_cache": {str(k): pool_concentration(val_pool[val_source == k]) for k in (0, 1, 2)},
    }
    return plan, coverage


def validate_plan(plan, obs, act, train, heldout):
    train_set, held_set = set(map(int, train)), set(map(int, heldout))
    for prefix, allowed in (("train_pool", train_set), ("validation_pool", held_set)):
        e, t, src = plan[prefix + "_episode"], plan[prefix + "_time"], plan[prefix + "_source"]
        assert set(map(int, np.unique(e))) <= allowed
        assert np.all((t >= 0) & (t < 50))
        fm = fork_mask(obs[e, t]); dm, rm = direction_masks(act[e, t])
        assert np.all(~fm[src == 0]) and np.all(fm[src != 0])
        assert np.all(dm[src == 1]) and np.all(rm[src == 2])
    for ids, offsets, source, pool_source, pool_time in (
            (plan["critic_pool_id"], plan["critic_offset"], plan["critic_source"],
             plan["train_pool_source"], plan["train_pool_time"]),
            (plan["eval_pool_id"], plan["eval_offset"], plan["eval_source"],
             plan["validation_pool_source"], plan["validation_pool_time"])):
        assert np.all(pool_source[ids] == source)
        assert np.all((offsets >= 1) & (offsets <= 50 - pool_time[ids]))
        assert np.all(np.sum(source == 0, 1) == 128)
        assert np.all(np.sum(source == 1, 1) == 64)
        assert np.all(np.sum(source == 2, 1) == 64)
    assert len(np.unique(plan["ett_episode"])) == len(plan["ett_episode"]) == 96
    e, t, f = plan["actor_episode"], plan["actor_time"], plan["actor_future"]
    assert set(map(int, np.unique(e))) <= train_set and np.all(f > t) and np.all(f <= 50)


def prepare(out: Path):
    if out.exists(): raise ValueError("fresh output directory required")
    obs, act, initial, manifest, actor_prov, nominal_prov = load_eligible()
    train, heldout = split_ids()
    plan, coverage = build_plan(obs, act, train, heldout)
    validate_plan(plan, obs, act, train, heldout)
    out.mkdir(parents=True); (out / "checkpoints").mkdir(); (out / "ett_updates").mkdir()
    (out / "PROTOCOL.md").write_bytes(PROTOCOL_SOURCE.read_bytes())
    write_json(out / "config.json", CONFIG)
    np.savez_compressed(out / "partition.npz", train=train, heldout=heldout)
    np.savez_compressed(out / "sampler_plan.npz", **plan)
    write_json(out / "coverage.json", coverage)
    historical = [
        Path("artifacts/pointmaze_region_pilot/full_f4_integration_s01_v1/REPORT.md"),
        Path("artifacts/pointmaze_region_pilot/update_reference_s01_v1/REPORT.md"),
        Path("outputs/supervised_stochastic_ett_20260914_v1/REPORT.md"),
        Path("outputs/pointmaze_oracle_alive_motion_20260914_v1/REPORT.md"),
        Path("outputs/pointmaze_oracle_training_pilot_20260914_v1/REPORT.md"),
        Path("outputs/pointmaze_critic_control_20260914_v1/REPORT.md"),
        Path("outputs/pointmaze_offline_replay_control_20260915_v1/REPORT.md"),
    ]
    sources = [Path(__file__), TEST_SOURCE, PROTOCOL_SOURCE,
               Path("ett/pointmaze_region_pilot.py"),
               Path("ett/pointmaze_early_pilot.py"), Path("ett/pointmaze_update_reference.py"),
               Path("ett/convex_action_transition.py"), Path("ett/diagonal_transition.py"),
               Path("crl/losses.py"), Path("crl/networks.py"), Path("crl/checkpoint.py")]
    provenance = {
        "eligible": {
            "dataset": {"path": DATASET, "sha256": EXPECTED[str(DATASET)],
                        "content_sha256": manifest["content_sha256"], "loaded_arrays": ["obs", "act"]},
            "production_initial": {"path": INITIAL, "sha256": EXPECTED[str(INITIAL)],
                                   "step": 150000, "ancestry_commit": actor_prov["code_commit"],
                                   "full_state_sha256": tree_sha(initial),
                                   "actor_sha256": tree_sha(initial.policy_params),
                                   "critic_sha256": tree_sha(initial.q_params),
                                   "actor_optimizer_sha256": tree_sha(initial.policy_optimizer_state),
                                   "critic_optimizer_sha256": tree_sha(initial.q_optimizer_state)},
            "nominal": {"path": NOMINAL, "sha256": EXPECTED[str(NOMINAL)],
                        "population": nominal_prov["population"],
                        "selection_uses_outcome_or_hidden_state": False},
            "diagonal": {"path": DIAGONAL, "sha256": EXPECTED[str(DIAGONAL)],
                         "training": "lambda=0; total loss equals diagonal NLL; original offline expert-source rows"},
            "operational_metadata": {
                "dataset_manifest": {"path": DATASET_MANIFEST,
                                     "sha256": EXPECTED[str(DATASET_MANIFEST)]},
                "actor_provenance": {"path": INITIAL_PROVENANCE,
                                     "sha256": EXPECTED[str(INITIAL_PROVENANCE)]},
                "nominal_config": {"path": NOMINAL_CONFIG,
                                   "sha256": EXPECTED[str(NOMINAL_CONFIG)]},
                "nominal_population": {"path": NOMINAL_PROVENANCE,
                                       "sha256": EXPECTED[str(NOMINAL_PROVENANCE)]},
                "diagonal_history": {"path": DIAGONAL_HISTORY,
                                     "sha256": EXPECTED[str(DIAGONAL_HISTORY)]},
                "eligible_path_router": {"path": ROUTING_MANIFEST,
                                         "sha256": EXPECTED[str(ROUTING_MANIFEST)]},
            },
        },
        "excluded": [
            {"path": "artifacts/pointmaze_region_pilot/update_reference_s01_v1/final_kernels.npz",
             "reason": "ETT optimization ancestry includes native-collected early contexts"},
            {"commits": ["b3d1423", "5acf766"], "reason": "new native diagnostic trajectories/checkpoints"},
            {"names": ["C1", "supervised heads", "oracle-motion/replay checkpoints"],
             "reason": "oracle or newly collected native supervision"}],
        "failure_mechanism": {"explicit_failure_latent": False,
                              "learned_entry_probability": False,
                              "stationary_atom_is_not_death": True,
                              "reason": "offline diagonal observations do not identify off-diagonal failure entry"},
        "split_limitation": "initial production checkpoint saw all 6600 episodes",
        "paper": {"path": PAPER, "sha256": file_sha(PAPER), "reviewed_pages": [2, 5, 6]},
        "source_sha256": {str(p): file_sha(p) for p in sources},
        "historical_sha256_before": {str(p): file_sha(p) for p in historical},
        "git": {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "reviewed_commits": ["5acf766", "b79f1a62cf093dad303ef268b7e75a5b0f5d336e"]},
        "runtime": {"python": platform.python_version(), "jax": jax.__version__,
                    "torch": torch.__version__, "devices": [str(x) for x in jax.devices()]},
    }
    write_json(out / "provenance.json", provenance)
    sealed = ["PROTOCOL.md", "config.json", "partition.npz", "sampler_plan.npz",
              "coverage.json", "provenance.json"]
    write_json(out / "seal.json", {
        "sealed_before_optimizer_updates": True,
        "optimizer_updates_so_far": 0,
        "sha256": {name: file_sha(out / name) for name in sealed},
        "initial_state_sha256_both_arms": tree_sha(initial),
        "fixed_final_only": True,
    })
    print("sealed strictly offline protocol and sampler plan", flush=True)


def load_prepared(out: Path):
    obs, act, initial, _, _, _ = load_eligible()
    partition = np.load(out / "partition.npz", allow_pickle=False)
    train, heldout = partition["train"], partition["heldout"]
    z = np.load(out / "sampler_plan.npz", allow_pickle=False)
    plan = {k: z[k] for k in z.files}
    validate_plan(plan, obs, act, train, heldout)
    return obs, act, initial, train, heldout, plan


def diagonal_context(obs, act, plan):
    result = {}
    for part, prefix in (("train", "diagonal_train"),
                         ("validation", "diagonal_validation")):
        e, t = plan[prefix + "_episode"], plan[prefix + "_time"]
        result[part + "_state"] = obs[e, t, :8].astype(np.float32)
        result[part + "_action"] = act[e, t].astype(np.float32)
        result[part + "_target"] = obs[e, t + 1, :8].astype(np.float32)
    state = obs[:, :50, :8]
    # Normalization is computed from learner-visible F4 only. The caller
    # overwrites these two fields with the continuation-training partition.
    result["mean"] = state.reshape(-1, 8).mean(0)
    result["std"] = np.maximum(state.reshape(-1, 8).std(0), .1)
    return result


def save_records(path: Path, records):
    phase.save_records(path, records)


def generate_ett_records(engine, theta, obs, act, plan, seed, ledger, purpose):
    e, t = plan["ett_episode"], plan["ett_time"]
    roots, first = obs[e, t, :8], act[e, t]
    return early.collect(engine, theta, roots, t, seed, ledger, purpose,
                         repeats=CONFIG["ett_root_repeats"], first=first)


def generate_cache(engine, theta, obs, act, episode, time_index, seed,
                   ledger, purpose):
    n = len(episode)
    lengths = (50 - time_index).astype(np.int32)
    states = np.full((n, 51, 8), np.nan, np.float32)
    actions = np.full((n, 50, 2), np.nan, np.float32)
    x_prime = np.full((n, 50, 2), np.nan, np.float32)
    reward = np.full((n, 50), np.nan, np.float32)
    atom = np.zeros((n, 50), bool)
    projected = np.zeros((n, 50), bool)
    valid = np.zeros((n, 50), bool)
    for h in sorted(np.unique(lengths)):
        ix = np.flatnonzero(lengths == h)
        ledger.add(len(ix) * int(h), purpose + f"/h{h}")
        record = engine.rollout(theta, obs[episode[ix], time_index[ix], :8],
                                jax.random.PRNGKey(seed + int(h)), int(h),
                                act[episode[ix], time_index[ix]])
        states[ix, :h + 1] = record["states"]
        actions[ix, :h] = record["action"]
        x_prime[ix, :h] = record["x_prime"]
        reward[ix, :h] = record["reward"]
        atom[ix, :h] = record["atom"]
        projected[ix, :h] = record["projected"]
        valid[ix, :h] = True
        np.testing.assert_array_equal(record["action"][:, 0], act[episode[ix], time_index[ix]])
        np.testing.assert_array_equal(record["states"][:, 1:, 2:], record["states"][:, :-1, :6])
        print(purpose, "remaining horizon", int(h), "anchors", len(ix), flush=True)
    state_valid = np.arange(51)[None, :] <= lengths[:, None]
    transition_valid = np.arange(50)[None, :] < lengths[:, None]
    assert np.isfinite(states[state_valid]).all()
    assert np.isnan(states[~state_valid]).all()
    assert np.array_equal(valid, transition_valid)
    assert np.isfinite(actions[transition_valid]).all()
    assert np.isnan(actions[~transition_valid]).all()
    return {"states": states, "action": actions, "x_prime": x_prime,
            "reward": reward, "atom": atom, "projected": projected,
            "valid": valid, "length": lengths, "episode": episode,
            "anchor_time": time_index,
            "anchor_action": act[episode, time_index].astype(np.float32),
            "theta_sha256": np.array(array_sha(theta)),
            "base_diagonal_file_sha256": np.array(EXPECTED[str(DIAGONAL)]),
            "nominal_file_sha256": np.array(EXPECTED[str(NOMINAL)]),
            "fixed_actor_file_sha256": np.array(EXPECTED[str(INITIAL)])}


def save_cache(path: Path, record):
    np.savez_compressed(path, **record)


def load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def cache_audit(record):
    n = len(record["length"])
    first_ok = np.array_equal(record["action"][:, 0], record["anchor_action"])
    history_error = 0.
    task_reached, sampled_atom, atom_then_later_motion = [], 0, 0
    returns = []
    for i, h in enumerate(record["length"]):
        s = record["states"][i, :h + 1]
        history_error = max(history_error, float(np.max(np.abs(s[1:, 2:] - s[:-1, :6]))))
        rr = record["reward"][i, :h]
        returns.append(float(rr @ (.95 ** np.arange(h))))
        task_reached.append(bool(np.any(rr > 0)))
        for t in np.flatnonzero(record["atom"][i, :h]):
            sampled_atom += 1
            if t + 1 < h:
                later = np.linalg.norm(np.diff(s[t + 1:, :2], axis=0), axis=1)
                atom_then_later_motion += int(np.any(later > 1e-7))
    valid_count = int(np.sum(record["valid"]))
    return {"paths": n, "valid_transitions": valid_count,
            "first_recorded_action_exact": first_ok,
            "history_shift_max_abs_error": history_error,
            "task_region_reached_path_fraction": float(np.mean(task_reached)),
            "mean_discounted_model_reward": float(np.mean(returns)),
            "raw_stationary_atom_fraction": float(sampled_atom / valid_count),
            "atom_events_with_later_step_available": int(sum(
                np.sum(record["atom"][i, :max(int(h) - 1, 0)])
                for i, h in enumerate(record["length"]))),
            "atom_then_any_later_motion_count": atom_then_later_motion,
            "failure_frequency": None,
            "post_failure_consistency": None,
            "failure_note": "unsupported: raw stationary atoms are not classified as death"}


def cache_pair_audit(initial, final, source):
    """Compare common-random-number paths without assigning death semantics."""
    assert np.array_equal(initial["episode"], final["episode"])
    assert np.array_equal(initial["anchor_time"], final["anchor_time"])
    assert np.array_equal(initial["length"], final["length"])
    result = {}
    for name, paths in (("all", np.ones(len(source), bool)),
                        ("ordinary_nonfork", source == 0),
                        ("down", source == 1), ("right", source == 2),
                        ("fork", source != 0)):
        rows, episode, delta, initial_region, final_region = [], [], [], [], []
        for i in np.flatnonzero(paths):
            h = int(final["length"][i])
            rows.append(np.arange(1, h + 1))
            episode.append(np.full(h, final["episode"][i], np.int32))
            left, right = initial["states"][i, 1:h + 1], final["states"][i, 1:h + 1]
            delta.append(np.linalg.norm(right[:, :2] - left[:, :2], axis=1))
            initial_region.append(np.linalg.norm(left[:, :2] - GOAL[:2], axis=1) < 2.)
            final_region.append(np.linalg.norm(right[:, :2] - GOAL[:2], axis=1) < 2.)
        delta, episode = np.concatenate(delta), np.concatenate(episode)
        initial_region, final_region = np.concatenate(initial_region), np.concatenate(final_region)
        result[name] = {
            "paths": int(np.sum(paths)),
            "episodes": int(len(np.unique(episode))),
            "future_state_rows": int(len(delta)),
            "changed_xy_fraction": float(np.mean(delta > 1e-7)),
            "mean_xy_l2": float(np.mean(delta)),
            "initial_task_region_fraction": float(np.mean(initial_region)),
            "final_task_region_fraction": float(np.mean(final_region)),
            "paired_final_minus_initial_task_region_fraction": cluster_bootstrap(
                final_region.astype(float) - initial_region.astype(float), episode,
                CONFIG["seed"] + 900 + {"all": 0, "ordinary_nonfork": 1,
                                         "down": 2, "right": 3, "fork": 4}[name]),
        }
    return result


def fit_ett(out: Path):
    if (out / "ett_training.json").exists(): raise ValueError("ETT phase already complete")
    obs, act, initial, train, _, plan = load_prepared(out)
    engine = region.Kernel()
    assert tree_sha(initial.policy_params) == tree_sha(
        checkpoint.load_checkpoint(INITIAL)[1].policy_params)
    immutable = tree_sha((engine.base.params, engine.nominal.params))
    context = diagonal_context(obs, act, plan)
    train_states = obs[train, :50, :8].reshape(-1, 8)
    context["mean"] = train_states.mean(0)
    context["std"] = np.maximum(train_states.std(0), .1)
    ledger = Ledger(out)
    theta = np.zeros(48, np.float32)
    guard_rows = np.arange(256)
    initial_guard = float(region.diagonal(
        engine, theta, context, "train", guard_rows, 16,
        CONFIG["seed"] + 10, ledger, "ett/initial_diagonal_guard").mean())

    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    critic = early.Critic(CONFIG["seed"] % (2 ** 31 - 1))
    optimizer = torch.optim.Adam(critic.parameters(), lr=.003)
    prefit = generate_ett_records(engine, theta, obs, act, plan,
                                  CONFIG["seed"] + 100, ledger, "ett/auxiliary_prefit_paths")
    save_records(out / "ett_updates" / "auxiliary_prefit_paths.npz", prefit)
    prefit_info = update_reference.refresh(
        critic, optimizer, prefit, context, CONFIG["seed"] + 101,
        steps=CONFIG["auxiliary_prefit_steps"])
    history = []
    for iteration in range(CONFIG["ett_updates"]):
        folder = out / "ett_updates" / f"u{iteration}"
        folder.mkdir()
        key = CONFIG["seed"] + 10000 + iteration * 1000
        preupdate = theta.copy()
        records = generate_ett_records(engine, preupdate, obs, act, plan, key,
                                       ledger, f"ett/u{iteration}/visitation")
        save_records(folder / "paths.npz", records)
        refresh = update_reference.refresh(
            critic, optimizer, records, context, key + 1,
            steps=CONFIG["auxiliary_refresh_steps"])
        queries = update_reference.visitation(records, key + 2, count=CONFIG["ett_queries"])
        np.savez_compressed(folder / "queries.npz", **queries)
        rng = np.random.default_rng(key + 3)
        directions = rng.normal(size=(CONFIG["ett_directions"], 48))
        diagonal_indices = rng.integers(512, size=128)
        sigma = np.r_[np.full(16, CONFIG["ett_sigma"]["diagonal"]),
                      np.full(32, CONFIG["ett_sigma"]["response"])]
        signed = np.empty((CONFIG["ett_directions"], 2, 2), np.float64)
        for d, direction in enumerate(directions):
            for si, sign in enumerate((1., -1.)):
                candidate = (preupdate + sign * sigma * direction).astype(np.float32)
                signed[d, si, 0] = region.diagonal(
                    engine, candidate, context, "train", diagonal_indices, 8,
                    key + 30, ledger, f"ett/u{iteration}/signed_diagonal").mean()
                value, raw = update_reference.surrogate(
                    engine, candidate, preupdate, queries, critic, context,
                    key + 40, ledger)
                signed[d, si, 1] = value
                np.savez_compressed(folder / f"d{d}_sign{si}.npz", **raw)
        proposed, info = update_reference.propose(preupdate, directions, signed, True)
        candidate_guard = float(region.diagonal(
            engine, proposed, context, "train", guard_rows, 16,
            CONFIG["seed"] + 10, ledger, f"ett/u{iteration}/candidate_guard").mean())
        accepted = update_reference.guard_accept(candidate_guard, initial_guard)
        if accepted: theta = proposed
        assert np.isfinite(theta).all()
        row = {"iteration": iteration, "accepted": accepted,
               "initial_guard": initial_guard, "candidate_guard": candidate_guard,
               "preupdate_sha256": array_sha(preupdate),
               "accepted_theta_sha256": array_sha(theta),
               "auxiliary_refresh": refresh, "signed_terms": signed, **info}
        np.savez_compressed(folder / "update.npz", preupdate=preupdate,
                            proposed=proposed, accepted_theta=theta,
                            directions=directions, diagonal_indices=diagonal_indices,
                            signed=signed, **info)
        write_json(folder / "update.json", row)
        history.append(row)
        print("ETT update", iteration + 1, "accepted", accepted,
              "guard", round(candidate_guard, 6), flush=True)

    np.savez_compressed(out / "checkpoints" / "ett_final.npz", theta=theta)
    torch.save({"model": critic.state_dict(), "optimizer": optimizer.state_dict()},
               out / "checkpoints" / "auxiliary_region_value_final.pt")
    validation_initial = region.diagonal(
        engine, np.zeros(48, np.float32), context, "validation", np.arange(512), 32,
        CONFIG["seed"] + 200, ledger, "ett/validation_diagonal_initial")
    validation_final = region.diagonal(
        engine, theta, context, "validation", np.arange(512), 32,
        CONFIG["seed"] + 200, ledger, "ett/validation_diagonal_final")

    train_cache = generate_cache(
        engine, theta, obs, act, plan["train_pool_episode"], plan["train_pool_time"],
        CONFIG["seed"] + 300, ledger, "production/train_final_ett_cache")
    validation_cache = generate_cache(
        engine, theta, obs, act, plan["validation_pool_episode"], plan["validation_pool_time"],
        CONFIG["seed"] + 400, ledger, "production/validation_final_ett_cache")
    initial_validation_cache = generate_cache(
        engine, np.zeros(48, np.float32), obs, act,
        plan["validation_pool_episode"], plan["validation_pool_time"],
        CONFIG["seed"] + 400, ledger, "audit/validation_initial_ett_cache")
    save_cache(out / "train_trajectory_pool.npz", train_cache)
    save_cache(out / "validation_trajectory_pool.npz", validation_cache)
    save_cache(out / "initial_validation_trajectory_pool.npz", initial_validation_cache)
    interface = {
        "initial_validation": cache_audit(initial_validation_cache),
        "final_validation": cache_audit(validation_cache),
        "paired_initial_to_final_validation": cache_pair_audit(
            initial_validation_cache, validation_cache,
            plan["validation_pool_source"]),
        "final_training": cache_audit(train_cache),
        "diagonal_validation_energy_initial": float(np.mean(validation_initial)),
        "diagonal_validation_energy_final": float(np.mean(validation_final)),
        "diagonal_validation_energy_change": float(np.mean(validation_final - validation_initial)),
        "explicit_failure_supported": False,
        "failure_assumption": "failure would be irreversible, but no entry event is inferred or sampled",
        "learned_failure_entry_probability": None,
        "stationary_atom_is_failure": False,
    }
    write_json(out / "sampling_interface_pretraining.json", interface)
    assert immutable == tree_sha((engine.base.params, engine.nominal.params))
    training = {"status": "complete", "theta_initial_sha256": array_sha(np.zeros(48, np.float32)),
                "theta_final_sha256": array_sha(theta), "theta_l2": float(np.linalg.norm(theta)),
                "accepted_updates": int(sum(r["accepted"] for r in history)),
                "history": history, "auxiliary_prefit": prefit_info,
                "auxiliary_updates": CONFIG["auxiliary_prefit_steps"] +
                                     CONFIG["ett_updates"] * CONFIG["auxiliary_refresh_steps"],
                "model_outputs": ledger.data["charged"],
                "model_output_cap": ledger.data["cap"],
                "immutable_diagonal_and_nominal_sha256": immutable,
                "fixed_actor_sha256": tree_sha(initial.policy_params),
                "failure_model": "unsupported; no event sampled"}
    write_json(out / "ett_training.json", training)
    print("eligible offline-rooted ETT phase complete; model outputs",
          ledger.data["charged"], flush=True)


def make_network_and_optimizers():
    network = networks.make_networks(8, 8, 2, repr_dim=64, repr_norm=False,
        repr_norm_temp=True, hidden_layer_sizes=(256, 256), actor_min_std=1e-6,
        twin_q=False, use_image_obs=False, use_layer_norm=False, obs_scale=None)
    q_optimizer = optax.adam(CONFIG["critic_learning_rate"], eps=CONFIG["adam_eps"])
    policy_optimizer = optax.adam(CONFIG["actor_learning_rate"], eps=CONFIG["adam_eps"])
    return network, q_optimizer, policy_optimizer


def transition_batch(state, action, goal, following):
    n = len(state)
    batch = Transition(np.concatenate([state, goal], -1).astype(np.float32),
                       action.astype(np.float32),
                       np.full(n, np.nan, np.float32),
                       np.full(n, np.nan, np.float32),
                       np.concatenate([following, goal], -1).astype(np.float32),
                       np.full((n, 2), np.nan, np.float32))
    return jax.tree_util.tree_map(jnp.asarray, batch)


def rows_from_pool(obs, act, plan, cache, pool_prefix, ids, offsets, arm):
    episode = plan[pool_prefix + "_episode"][ids]
    time_index = plan[pool_prefix + "_time"][ids]
    state = obs[episode, time_index, :8]
    action = act[episode, time_index]
    following = obs[episode, time_index + 1, :8]
    if arm == "O":
        goal = obs[episode, time_index + offsets, :8]
    elif arm == "P":
        goal = cache["states"][ids, offsets]
        assert np.all(offsets <= cache["length"][ids])
    else:
        raise ValueError(arm)
    assert np.isfinite(goal).all()
    return transition_batch(state, action, goal, following), goal


def ordinary_batch(obs, act, plan, update):
    e, t, f = (plan["actor_episode"][update], plan["actor_time"][update],
               plan["actor_future"][update])
    return transition_batch(obs[e, t, :8], act[e, t], obs[e, f, :8], obs[e, t + 1, :8])


def build_critic_step(network, optimizer):
    def objective(params, batch):
        logits = network.q_network.apply(params, batch.observation, batch.action)
        labels = jnp.eye(logits.shape[0], dtype=logits.dtype)
        matrix = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
        pos = jnp.mean(jnp.diag(logits))
        neg = (jnp.sum(logits) - jnp.sum(jnp.diag(logits))) / (logits.size - logits.shape[0])
        return jnp.mean(matrix), (pos, neg, pos - neg,
                                  jnp.mean((logits > 0) == labels),
                                  jnp.mean(jnp.argmax(logits, 1) == jnp.arange(logits.shape[0])))
    grad_fn = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def step(params, optimizer_state, target, batch):
        (loss, aux), gradient = grad_fn(params, batch)
        update, optimizer_state = optimizer.update(gradient, optimizer_state)
        params = optax.apply_updates(params, update)
        target = jax.tree_util.tree_map(
            lambda x, y: x * (1 - CONFIG["tau"]) + y * CONFIG["tau"], target, params)
        return (params, optimizer_state, target, loss, aux,
                optax.global_norm(gradient), optax.global_norm(update))
    return step


def actor_parts(network, policy_params, q_params, batch, key):
    state, goal = batch.observation[:, :8], batch.observation[:, 8:]
    new_state = jnp.concatenate([state, state], 0)
    new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, 0)], 0)
    original_action = jnp.concatenate([batch.action, batch.action], 0)
    observation = jnp.concatenate([new_state, new_goal], 1)
    distribution = network.policy_network.apply(policy_params, observation)
    sampled_action = network.sample(distribution, key)
    log_probability = network.log_prob(distribution, sampled_action)
    q = network.q_network.apply(q_params, observation, sampled_action)
    critic_term = -jnp.mean(jnp.diag(q))
    bc_nll = -jnp.mean(network.log_prob(distribution, original_action))
    auxiliary = (jnp.mean(-log_probability), jnp.median(distribution.scale),
                 jnp.mean(jnp.abs(distribution.loc)),
                 jnp.mean((jnp.abs(jnp.tanh(distribution.loc)) > .99).astype(jnp.float32)))
    return bc_nll, critic_term, auxiliary


def build_actor_step(network, optimizer):
    def objective(policy_params, q_params, batch, key):
        bc, critic, auxiliary = actor_parts(network, policy_params, q_params, batch, key)
        return .5 * bc + .5 * critic, (bc, critic, auxiliary)
    grad_fn = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def step(policy_params, optimizer_state, q_params, batch, key):
        (loss, (bc, critic, auxiliary)), gradient = grad_fn(
            policy_params, q_params, batch, key)
        update, optimizer_state = optimizer.update(gradient, optimizer_state)
        policy_params = optax.apply_updates(policy_params, update)
        return (policy_params, optimizer_state, loss, bc, critic, auxiliary,
                optax.global_norm(gradient), optax.global_norm(update))
    return step


def finite_tree(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))


def train(out: Path):
    if (out / "production_training.json").exists(): raise ValueError("production phase complete")
    obs, act, initial, _, _, plan = load_prepared(out)
    cache = load_cache(out / "train_trajectory_pool.npz")
    network, q_optimizer, policy_optimizer = make_network_and_optimizers()
    critic_step = build_critic_step(network, q_optimizer)
    curves, states, summary = {}, {}, {
        "initial": {"full_state_sha256": tree_sha(initial),
                    "policy_sha256": tree_sha(initial.policy_params),
                    "critic_sha256": tree_sha(initial.q_params),
                    "actor_optimizer_sha256": tree_sha(initial.policy_optimizer_state),
                    "critic_optimizer_sha256": tree_sha(initial.q_optimizer_state)}}
    lineage = {"pool_id": plan["critic_pool_id"], "future_offset": plan["critic_offset"],
               "source": plan["critic_source"]}
    O_goals, P_goals = [], []
    for arm in ("O", "P"):
        state = initial
        rows, goals = [], []
        for u in range(CONFIG["critic_updates"]):
            ids, offsets = plan["critic_pool_id"][u], plan["critic_offset"][u]
            batch, goal = rows_from_pool(obs, act, plan, cache, "train_pool", ids, offsets, arm)
            q, qo, tq, loss, auxiliary, gn, un = critic_step(
                state.q_params, state.q_optimizer_state, state.target_q_params, batch)
            values = np.asarray([loss, *auxiliary, gn, un], np.float64)
            if not np.isfinite(values).all() or not finite_tree((q, qo, tq)):
                raise FloatingPointError((arm, u, values))
            state = state._replace(q_params=q, q_optimizer_state=qo, target_q_params=tq)
            rows.append(values); goals.append(goal)
            if (u + 1) % 50 == 0:
                print("production critic", arm, u + 1, "loss", round(float(loss), 6), flush=True)
        assert tree_sha(state.policy_params) == tree_sha(initial.policy_params)
        assert tree_sha(state.policy_optimizer_state) == tree_sha(initial.policy_optimizer_state)
        checkpoint.save_named(out / "checkpoints", f"critic_{arm}_final", 150400, state)
        curves[f"critic_{arm}"] = np.asarray(rows)
        states[arm] = state
        summary[f"critic_{arm}"] = {"updates": 400, "final_loss": float(rows[-1][0]),
            "critic_sha256": tree_sha(state.q_params),
            "critic_optimizer_sha256": tree_sha(state.q_optimizer_state),
            "critic_parameter_delta_l2": tree_delta_norm(state.q_params, initial.q_params),
            "actor_frozen": True}
        (O_goals if arm == "O" else P_goals).append(np.concatenate(goals, 0))
    lineage["observational_goal"] = O_goals[0].reshape(400, 256, 8)
    lineage["pessimistic_goal"] = P_goals[0].reshape(400, 256, 8)
    np.savez_compressed(out / "nce_row_lineage.npz", **lineage)

    actor_step = build_actor_step(network, policy_optimizer)
    for arm in ("O", "P"):
        critic_state = states[arm]
        state = initial._replace(q_params=critic_state.q_params,
            target_q_params=critic_state.target_q_params,
            q_optimizer_state=critic_state.q_optimizer_state)
        rows = []
        for u in range(CONFIG["actor_updates"]):
            batch = ordinary_batch(obs, act, plan, u)
            pp, po, loss, bc, critic, auxiliary, gn, un = actor_step(
                state.policy_params, state.policy_optimizer_state, state.q_params,
                batch, jnp.asarray(plan["actor_keys"][u]))
            values = np.asarray([loss, bc, critic, *auxiliary, gn, un], np.float64)
            if not np.isfinite(values).all() or not finite_tree((pp, po)):
                raise FloatingPointError((arm, u, values))
            state = state._replace(policy_params=pp, policy_optimizer_state=po)
            rows.append(values)
            if (u + 1) % 100 == 0:
                print("production actor", arm, u + 1, "loss", round(float(loss), 6), flush=True)
        assert tree_sha(state.q_params) == tree_sha(critic_state.q_params)
        checkpoint.save_named(out / "checkpoints", f"actor_{arm}_final", 151400, state)
        curves[f"actor_{arm}"] = np.asarray(rows)
        summary[f"actor_{arm}"] = {"updates": 1000,
            "policy_sha256": tree_sha(state.policy_params),
            "actor_optimizer_sha256": tree_sha(state.policy_optimizer_state),
            "critic_sha256": tree_sha(state.q_params),
            "policy_parameter_delta_l2": tree_delta_norm(state.policy_params, initial.policy_params),
            "mean_raw_gradient_l2": float(np.mean(np.asarray(rows)[:, -2])),
            "mean_optimizer_update_l2": float(np.mean(np.asarray(rows)[:, -1])),
            "final_raw_gradient_l2": float(rows[-1][-2]),
            "final_optimizer_update_l2": float(rows[-1][-1])}
    summary["integrity"] = {
        "same_initial_full_state": True,
        "matched_anchor_and_future_offset_plan": True,
        "O_positive_source": "100% original recorded continuation",
        "P_positive_source": "100% final ETT cached continuation",
        "actor_batches_identical": True,
        "actor_keys_sha256": array_sha(plan["actor_keys"]),
        "actor_updates_occurred": all(summary[f"actor_{a}"]["policy_parameter_delta_l2"] > 0
                                      for a in ("O", "P"))}
    np.savez_compressed(out / "learning_curves.npz", **curves,
        critic_columns=np.array(["loss", "positive_logit", "negative_logit", "logit_gap",
                                 "binary_accuracy", "categorical_accuracy", "raw_gradient_l2",
                                 "optimizer_update_l2"]),
        actor_columns=np.array(["loss", "bc_nll", "critic_actor_term", "sample_entropy",
                                "scale_median", "loc_abs_mean", "saturation_fraction",
                                "raw_gradient_l2", "optimizer_update_l2"]))
    write_json(out / "production_training.json", summary)
    print("both production critics and actors complete", flush=True)


def paired_scores(network, q_params, observation, action):
    phi, psi = network.representation_network.apply(q_params, observation, action)
    return jnp.mean(jnp.sum(phi * psi, axis=1), axis=-1)


def nce_metrics(network, q_params, batch):
    logits = network.q_network.apply(q_params, batch.observation, batch.action)
    labels = jnp.eye(logits.shape[0], dtype=logits.dtype)
    loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    diagonal = jnp.diag(logits)
    negative = (jnp.sum(logits) - jnp.sum(diagonal)) / (logits.size - logits.shape[0])
    return jnp.array([jnp.mean(loss), jnp.mean(diagonal), negative,
                      jnp.mean(diagonal) - negative,
                      jnp.mean((logits > 0) == labels),
                      jnp.mean(jnp.argmax(logits, 1) == jnp.arange(logits.shape[0]))])


def bootstrap_mean(values, seed):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values),
                         size=(CONFIG["bootstrap_replicates"], len(values)))
    means = values[index].mean(1)
    return {"mean": float(values.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}


def cluster_bootstrap(values, episode, seed):
    """Paired episode bootstrap retaining every sampled row in each episode."""
    values, episode = np.asarray(values, np.float64), np.asarray(episode)
    unique, inverse = np.unique(episode, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    total = np.bincount(inverse, weights=values).astype(np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(unique),
                         size=(CONFIG["bootstrap_replicates"], len(unique)))
    means = total[index].sum(1) / count[index].sum(1)
    return {"mean": float(values.mean()), "episodes": int(len(unique)),
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "resampling_unit": "source episode"}


def action_distribution(network, policy_params, observation, eps):
    distribution = network.policy_network.apply(policy_params, jnp.asarray(observation))
    loc, scale = np.asarray(distribution.loc), np.asarray(distribution.scale)
    sample = np.tanh(loc[:, None] + scale[:, None] * eps).astype(np.float32)
    return loc, scale, sample


def action_summary(sample):
    down, right = direction_masks(sample)
    return {"down_probability": float(np.mean(down)),
            "right_probability": float(np.mean(right)),
            "other_probability": float(np.mean(~(down | right))),
            "action_x_mean": float(np.mean(sample[..., 0])),
            "action_y_mean": float(np.mean(sample[..., 1])),
            "action_x_std": float(np.std(sample[..., 0])),
            "action_y_std": float(np.std(sample[..., 1])),
            "action_x_quantiles": np.quantile(sample[..., 0], [.1, .25, .5, .75, .9]),
            "action_y_quantiles": np.quantile(sample[..., 1], [.1, .25, .5, .75, .9])}


def local_gradients(network, q_params, observation, action):
    return jax.grad(lambda a: jnp.sum(paired_scores(network, q_params, observation, a)))(action)


def flat_dot(a, b):
    left, right = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    dot = sum(float(jnp.vdot(x, y)) for x, y in zip(left, right))
    na = math.sqrt(sum(float(jnp.vdot(x, x)) for x in left))
    nb = math.sqrt(sum(float(jnp.vdot(x, x)) for x in right))
    return dot, na, nb, dot / (na * nb) if na and nb else float("nan")


def component_gradients(network, policy_params, q_params, batch, key):
    bc_value, bc_grad = jax.value_and_grad(
        lambda p: actor_parts(network, p, q_params, batch, key)[0])(policy_params)
    q_value, q_grad = jax.value_and_grad(
        lambda p: actor_parts(network, p, q_params, batch, key)[1])(policy_params)
    dot, bn, qn, cosine = flat_dot(bc_grad, q_grad)
    combined = jax.tree_util.tree_map(lambda x, y: .5 * x + .5 * y, bc_grad, q_grad)
    return {"bc_nll": float(bc_value), "critic_actor_term": float(q_value),
            "bc_raw_gradient_l2": bn, "critic_raw_gradient_l2": qn,
            "bc_weighted_gradient_l2": .5 * bn,
            "critic_weighted_gradient_l2": .5 * qn,
            "component_dot": dot, "component_cosine": cosine,
            "combined_raw_gradient_l2": tree_norm(combined)}


def future_stats(goals, anchors, source, episode, pool_id):
    result = {}
    for name, mask in (("all", np.ones(len(goals), bool)),
                       ("ordinary_nonfork", source == 0),
                       ("down", source == 1), ("right", source == 2),
                       ("fork", source != 0)):
        g, s = goals[mask], anchors[mask]
        result[name] = {
            "rows": int(len(g)),
            "episodes": int(len(np.unique(episode[mask]))),
            "unique_cached_anchors": int(len(np.unique(pool_id[mask]))),
            "task_region_fraction": float(np.mean(np.linalg.norm(g[:, :2] - GOAL[:2], axis=1) < 2.)),
            "stationary_relative_to_anchor_fraction": float(np.mean(np.linalg.norm(g[:, :2] - s[:, :2], axis=1) <= 1e-7)),
            "goal_xy_mean": np.mean(g[:, :2], 0),
            "goal_xy_std": np.std(g[:, :2], 0),
            "goal_xy_covariance": np.cov(g[:, :2].T),
        }
    return result


def evaluate(out: Path):
    if (out / "results.json").exists(): raise ValueError("evaluation complete")
    obs, act, initial, _, _, plan = load_prepared(out)
    train_cache = load_cache(out / "train_trajectory_pool.npz")
    validation_cache = load_cache(out / "validation_trajectory_pool.npz")
    critic_states = {a: checkpoint.load_checkpoint(
        out / "checkpoints" / f"critic_{a}_final.pkl")[1] for a in ("O", "P")}
    actor_states = {a: checkpoint.load_checkpoint(
        out / "checkpoints" / f"actor_{a}_final.pkl")[1] for a in ("O", "P")}
    network, _, _ = make_network_and_optimizers()
    qsets = {"initial": initial.q_params, "O": critic_states["O"].q_params,
             "P": critic_states["P"].q_params}
    psets = {"initial": initial.policy_params, "O": actor_states["O"].policy_params,
             "P": actor_states["P"].policy_params}
    results = {"sampling": {}, "critic": {}, "actor": {}, "gradient_components": {}}
    saved = {}

    # The exact production NCE rows: matched anchors/offsets, different futures.
    lineage = np.load(out / "nce_row_lineage.npz", allow_pickle=False)
    ids, offsets, source = lineage["pool_id"], lineage["future_offset"], lineage["source"]
    episode = plan["train_pool_episode"][ids]
    time_index = plan["train_pool_time"][ids]
    anchors = obs[episode, time_index, :8].reshape(-1, 8)
    episode_flat = episode.reshape(-1)
    O_goal = lineage["observational_goal"].reshape(-1, 8)
    P_goal = lineage["pessimistic_goal"].reshape(-1, 8)
    source_flat = source.reshape(-1)
    flat_ids = ids.reshape(-1)
    results["sampling"]["observational_future"] = future_stats(
        O_goal, anchors, source_flat, episode_flat, flat_ids)
    results["sampling"]["pessimistic_future"] = future_stats(
        P_goal, anchors, source_flat, episode_flat, flat_ids)
    difference = np.linalg.norm(P_goal[:, :2] - O_goal[:, :2], axis=1)
    results["sampling"]["matched_future_change"] = {
        "rows": int(len(difference)), "changed_xy_fraction": float(np.mean(difference > 1e-7)),
        "mean_xy_l2": float(np.mean(difference)), "median_xy_l2": float(np.median(difference)),
        "p90_xy_l2": float(np.quantile(difference, .9)),
        "anchor_identity_sha256": array_sha(ids), "future_offset_sha256": array_sha(offsets),
        "same_anchor_and_offset_O_P": True, "P_positive_fraction_from_ETT": 1.0}
    results["sampling"]["paired_change_by_group"] = {}
    for name, mask in (("all", np.ones(len(P_goal), bool)),
                       ("ordinary_nonfork", source_flat == 0),
                       ("down", source_flat == 1),
                       ("right", source_flat == 2),
                       ("fork", source_flat != 0)):
        o_region = np.linalg.norm(O_goal[mask, :2] - GOAL[:2], axis=1) < 2.
        p_region = np.linalg.norm(P_goal[mask, :2] - GOAL[:2], axis=1) < 2.
        results["sampling"]["paired_change_by_group"][name] = {
            "rows": int(np.sum(mask)),
            "episodes": int(len(np.unique(episode_flat[mask]))),
            "changed_xy_fraction": float(np.mean(difference[mask] > 1e-7)),
            "mean_xy_l2": float(np.mean(difference[mask])),
            "P_minus_O_task_region_fraction": cluster_bootstrap(
                p_region.astype(float) - o_region.astype(float), episode_flat[mask],
                CONFIG["seed"] + 1000 + {"all": 0, "ordinary_nonfork": 1,
                                          "down": 2, "right": 3, "fork": 4}[name]),
        }
    saved.update(production_anchor_state=anchors, production_episode=episode_flat,
                 production_source=source_flat,
                 observational_future_goal=O_goal, pessimistic_future_goal=P_goal,
                 matched_future_xy_l2=difference)

    # Common continuation-held-out NCE rows under both future sources.
    nce_fn = jax.jit(lambda q, b: nce_metrics(network, q, b))
    for qname, qparams in qsets.items():
        results["critic"][qname] = {}
        for future_source, arm in (("observational", "O"), ("pessimistic", "P")):
            rows = []
            for u in range(CONFIG["eval_nce_batches"]):
                batch, _ = rows_from_pool(
                    obs, act, plan, validation_cache, "validation_pool",
                    plan["eval_pool_id"][u], plan["eval_offset"][u], arm)
                rows.append(np.asarray(nce_fn(qparams, batch)))
            rows = np.asarray(rows)
            saved[f"nce_{qname}_{future_source}"] = rows
            results["critic"][qname][future_source + "_nce"] = dict(zip(
                ["loss", "positive_logit", "negative_logit", "logit_gap",
                 "binary_accuracy", "categorical_accuracy"], rows.mean(0)))

    # Canonical task probes at one first observable fork state per held-out episode.
    ce, ct, cf = (plan["fork_context_episode"], plan["fork_context_time"],
                  plan["fork_context_future"])
    state = obs[ce, ct, :8].astype(np.float32)
    canonical = np.concatenate([state, np.broadcast_to(GOAL, state.shape)], 1)
    ordinary_goal = np.concatenate([state, obs[ce, cf, :8]], 1).astype(np.float32)
    score_fn = jax.jit(lambda q, o, a: paired_scores(network, q, o, a))
    for qname, qparams in qsets.items():
        scores = np.stack([np.asarray(score_fn(
            qparams, jnp.asarray(canonical),
            jnp.asarray(np.broadcast_to(action, (len(state), 2))))) for action in PROBES], 1)
        saved[f"probe_scores_{qname}"] = scores
        endpoint = scores[:, 0] - scores[:, 4]
        neighborhood = scores[:, :4].mean(1) - scores[:, 4:].mean(1)
        seed = CONFIG["seed"] + {"initial": 500, "O": 510, "P": 520}[qname]
        results["critic"][qname]["endpoint_down_minus_right"] = bootstrap_mean(endpoint, seed)
        results["critic"][qname]["neighborhood_down_minus_right"] = bootstrap_mean(neighborhood, seed + 1)
    results["critic"]["paired_P_minus_O_endpoint_gap"] = bootstrap_mean(
        saved["probe_scores_P"][:, 0] - saved["probe_scores_P"][:, 4] -
        saved["probe_scores_O"][:, 0] + saved["probe_scores_O"][:, 4], CONFIG["seed"] + 530)
    for future_source in ("observational", "pessimistic"):
        results["critic"][f"paired_P_minus_O_{future_source}_nce_loss"] = bootstrap_mean(
            saved[f"nce_P_{future_source}"][:, 0] - saved[f"nce_O_{future_source}"][:, 0],
            CONFIG["seed"] + (531 if future_source == "observational" else 532))

    # Actor distributions and episode-paired changes.
    saved.update(fork_context_episode=ce, fork_context_time=ct, fork_context_future=cf,
                 action_probes=PROBES)
    for pname, pp in psets.items():
        results["actor"][pname] = {}
        for goal_name, observation in (("canonical", canonical), ("ordinary_goal", ordinary_goal)):
            loc, scale, sample = action_distribution(network, pp, observation, plan["fork_eps"])
            saved[f"actor_{pname}_{goal_name}_loc"] = loc
            saved[f"actor_{pname}_{goal_name}_scale"] = scale
            saved[f"actor_{pname}_{goal_name}_sample"] = sample
            results["actor"][pname][goal_name] = action_summary(sample)
    for arm, seed in (("O", CONFIG["seed"] + 600), ("P", CONFIG["seed"] + 610)):
        for goal_name, add in (("canonical", 0), ("ordinary_goal", 2)):
            current = saved[f"actor_{arm}_{goal_name}_sample"]
            initial_sample = saved[f"actor_initial_{goal_name}_sample"]
            cd, cr = direction_masks(current); idown, iright = direction_masks(initial_sample)
            results["actor"][arm][goal_name]["paired_down_probability_change"] = bootstrap_mean(
                cd.mean(1) - idown.mean(1), seed + add)
            results["actor"][arm][goal_name]["paired_right_probability_change"] = bootstrap_mean(
                cr.mean(1) - iright.mean(1), seed + add + 1)
            results["actor"][arm][goal_name]["paired_action_x_mean_change"] = bootstrap_mean(
                current[..., 0].mean(1) - initial_sample[..., 0].mean(1), seed + add + 20)
            results["actor"][arm][goal_name]["paired_action_y_mean_change"] = bootstrap_mean(
                current[..., 1].mean(1) - initial_sample[..., 1].mean(1), seed + add + 21)
    results["actor"]["paired_P_minus_O"] = {}
    for goal_name, add in (("canonical", 0), ("ordinary_goal", 10)):
        ps, os = (saved[f"actor_{arm}_{goal_name}_sample"] for arm in ("P", "O"))
        pd, pr = direction_masks(ps); od, ora = direction_masks(os)
        results["actor"]["paired_P_minus_O"][goal_name] = {
            "down_probability": bootstrap_mean(pd.mean(1) - od.mean(1),
                                                CONFIG["seed"] + 650 + add),
            "right_probability": bootstrap_mean(pr.mean(1) - ora.mean(1),
                                                 CONFIG["seed"] + 651 + add),
            "action_x_mean": bootstrap_mean(ps[..., 0].mean(1) - os[..., 0].mean(1),
                                             CONFIG["seed"] + 652 + add),
            "action_y_mean": bootstrap_mean(ps[..., 1].mean(1) - os[..., 1].mean(1),
                                             CONFIG["seed"] + 653 + add),
        }

    # Local critic derivatives at actual common actor samples.
    lg = CONFIG["local_gradient_samples"]
    for arm in ("O", "P"):
        for stage, pp in (("initial_actor", psets["initial"]),
                          ("final_actor", psets[arm])):
            for goal_name, observation in (("canonical", canonical),
                                           ("ordinary_goal", ordinary_goal)):
                _, _, sample = action_distribution(network, pp, observation,
                                                    plan["fork_eps"][:, :lg])
                flat_obs = np.repeat(observation, lg, 0)
                gradient = np.asarray(local_gradients(
                    network, qsets[arm], jnp.asarray(flat_obs),
                    jnp.asarray(sample.reshape(-1, 2)))).reshape(len(state), lg, 2)
                saved[f"local_gradient_{arm}_{stage}_{goal_name}"] = gradient
                per_episode = gradient.mean(1)
                tangent = -(per_episode[:, 0] + per_episode[:, 1])
                results["critic"][arm][f"local_gradient_{stage}_{goal_name}"] = {
                    "mean_dx": float(per_episode[:, 0].mean()),
                    "mean_dy": float(per_episode[:, 1].mean()),
                    "mean_norm": float(np.linalg.norm(per_episode, axis=1).mean()),
                    "down_minus_right_tangent": bootstrap_mean(
                        tangent, CONFIG["seed"] + (700 if arm == "O" else 720) +
                        (5 if stage == "final_actor" else 0) +
                        (2 if goal_name == "ordinary_goal" else 0))}

    # Common fixed actor batches: raw BC/critic gradients and conflict.
    for arm in ("O", "P"):
        for stage, pp in (("initial", psets["initial"]), ("final", psets[arm])):
            rows = [component_gradients(network, pp, qsets[arm], ordinary_batch(obs, act, plan, u),
                                        jnp.asarray(plan["actor_keys"][u]))
                    for u in range(CONFIG["gradient_batches"])]
            results["gradient_components"][f"{arm}_{stage}"] = {
                key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    # Non-fork ordinary-goal retention.
    ae, at, af = plan["away_episode"], plan["away_time"], plan["away_future"]
    away_obs = np.concatenate([obs[ae, at, :8], obs[ae, af, :8]], 1).astype(np.float32)
    away_action = act[ae, at].astype(np.float32)
    away_modes, away_bc = {}, {}
    for pname, pp in psets.items():
        loc, scale, _ = action_distribution(network, pp, away_obs, plan["away_eps"])
        mode = np.tanh(loc)
        dist = network.policy_network.apply(pp, jnp.asarray(away_obs))
        bc = -np.asarray(network.log_prob(dist, jnp.asarray(away_action)))
        away_modes[pname], away_bc[pname] = mode, bc
        saved[f"away_mode_{pname}"] = mode; saved[f"away_bc_nll_{pname}"] = bc
        results["actor"][pname]["away_ordinary_goal"] = {
            "bc_nll_mean": float(bc.mean()), "mode_action_x_mean": float(mode[:, 0].mean()),
            "mode_action_y_mean": float(mode[:, 1].mean()), "scale_mean": float(scale.mean())}
    for arm, seed in (("O", CONFIG["seed"] + 800), ("P", CONFIG["seed"] + 810)):
        results["actor"][arm]["away_ordinary_goal"]["mode_l2_change"] = bootstrap_mean(
            np.linalg.norm(away_modes[arm] - away_modes["initial"], axis=1), seed)
        results["actor"][arm]["away_ordinary_goal"]["bc_nll_change"] = bootstrap_mean(
            away_bc[arm] - away_bc["initial"], seed + 1)

    results["counts"] = {"production_rows_per_arm": 400 * 256,
        "heldout_nce_rows_per_source": 32 * 256,
        "heldout_fork_context_rows": int(len(ce)),
        "heldout_fork_context_episodes": int(len(np.unique(ce))),
        "away_rows": int(len(ae)), "away_episodes": int(len(np.unique(ae)))}
    np.savez_compressed(out / "offline_evaluation_arrays.npz", **saved)
    write_json(out / "results.json", results)
    print("offline production evaluation complete", flush=True)


def estimate_text(record, scale=1., digits=4):
    mean = record["mean"] * scale
    low, high = (x * scale for x in record["ci95"])
    return f"{mean:.{digits}f} (95% CI {low:.{digits}f}, {high:.{digits}f})"


def percent(value):
    return f"{100 * value:.2f}%"


def interval_excludes_zero(record):
    low, high = record["ci95"]
    return low > 0 or high < 0


def finalize(out: Path):
    if not (out / "verification.json").exists():
        raise ValueError("verification must pass before report generation")
    results = read_json(out / "results.json")
    interface = read_json(out / "sampling_interface_pretraining.json")
    ett = read_json(out / "ett_training.json")
    training = read_json(out / "production_training.json")
    coverage = read_json(out / "coverage.json")
    verification = read_json(out / "verification.json")

    sampling = results["sampling"]
    o_all, p_all = sampling["observational_future"]["all"], sampling["pessimistic_future"]["all"]
    o_fork, p_fork = sampling["observational_future"]["fork"], sampling["pessimistic_future"]["fork"]
    paired_all = sampling["paired_change_by_group"]["all"]
    paired_fork = sampling["paired_change_by_group"]["fork"]
    path_pair = interface["paired_initial_to_final_validation"]["all"]
    critic = results["critic"]
    actor = results["actor"]
    critic_delta = critic["paired_P_minus_O_endpoint_gap"]
    actor_delta = actor["paired_P_minus_O"]["canonical"]

    if paired_fork["P_minus_O_task_region_fraction"]["mean"] >= 0:
        recommendation = (
            "Run one ETT-only controlled repair before any more production learning: fit an "
            "observable full-F4 multi-step persistence/response component from eligible original "
            "offline suffixes, then require this same matched-anchor sampler audit to show that "
            "fork futures are no longer more favorable. Do not label stationary rows as death.")
    elif not interval_excludes_zero(critic_delta):
        recommendation = (
            "Keep the loss unchanged and run one preregistered critic optimization/exposure fork "
            "on newly sealed P caches: hold the final ETT fixed, preserve ordinary coverage, and "
            "test whether more independent generated continuations per fork anchor make the "
            "conditional-future change learnable. Do not alter the goal labels or add ranking loss.")
    elif not (interval_excludes_zero(actor_delta["down_probability"]) or
              interval_excludes_zero(actor_delta["right_probability"])):
        recommendation = (
            "Run one actor-only coefficient fork with both critics frozen and identical replay, "
            "varying only the existing BC/critic mixture. This directly tests the observed local "
            "critic-versus-BC conflict without changing the critic objective or adding route labels.")
    else:
        recommendation = (
            "Collect new complete PointMaze episodes under the fixed final actors in a separately "
            "authorized confirmation experiment. The present offline actor response should not be "
            "promoted to a native-success claim without that independent confirmation.")

    lines = [
        "# Strictly offline PointMaze causal-contrastive integration",
        "",
        "## Result",
        "",
        "This bounded experiment restored the requested sampling pipeline without changing the "
        "production sigmoid-NCE loss or actor objective. It used only the fixed original offline "
        "dataset and eligible offline-trained checkpoints. No environment, simulator, native "
        "trajectory, oracle transition, hidden state, audit label, failure bank, ranking target, "
        "or return-regression target entered the run.",
        "",
        "The historical optimized ETT checkpoint was not reused because its optimization ancestry "
        "included native-collected contexts. The eligible lambda-zero diagonal law, nominal model, "
        "and frozen original actor instead seeded a fresh three-update ETT fit on 96 offline roots. "
        f"{ett['accepted_updates']} of 3 proposed ETT updates passed the fixed diagonal guard; the "
        f"final offset norm was {ett['theta_l2']:.6f}. The run emitted "
        f"{ett['model_outputs']:,} model transitions under the sealed {ett['model_output_cap']:,} cap.",
        "",
        "## 1. Did the eligible pessimistic ETT change conditional futures?",
        "",
        f"Yes descriptively: under common validation anchors and random streams, "
        f"{percent(path_pair['changed_xy_fraction'])} of {path_pair['future_state_rows']:,} valid "
        f"future-state rows changed in XY, with mean paired displacement {path_pair['mean_xy_l2']:.4f}. "
        f"The all-future task-region fraction changed from {percent(path_pair['initial_task_region_fraction'])} "
        f"to {percent(path_pair['final_task_region_fraction'])}; the paired episode-bootstrap change was "
        f"{estimate_text(path_pair['paired_final_minus_initial_task_region_fraction'], 100., 2)} percentage points.",
        "",
        f"The final validation paths reached the geometric task region on "
        f"{percent(interface['final_validation']['task_region_reached_path_fraction'])} of paths and had "
        f"raw atom frequency {percent(interface['final_validation']['raw_stationary_atom_fraction'])}. "
        f"There were {interface['final_validation']['atom_then_any_later_motion_count']:,} atom events "
        "followed by later motion. These atom statistics are not failure statistics.",
        "",
        "Irreversible-failure frequency and post-failure consistency are **unsupported**. The eligible "
        "model has no learned failure-entry variable, and the original offline observations do not "
        "identify off-diagonal death. The user-provided absorption assumption says what should happen "
        "after a genuine failure; it does not supply the probability of entering one. No stationary "
        "outcome was relabeled as death.",
        "",
        "## 2. Did the actual NCE batches preserve the change?",
        "",
        f"Yes. O and P used the same {o_all['rows']:,} production rows per arm, the same recorded "
        "anchor/action identities, and the same strictly-later discounted offsets. P sourced "
        "100% of positive goals from the complete final-ETT cache; O sourced 100% from the matching "
        f"recorded continuation. Across all rows, {percent(paired_all['changed_xy_fraction'])} of paired "
        f"goals changed in XY (mean L2 {paired_all['mean_xy_l2']:.4f}). The sampled task-region fraction "
        f"was O {percent(o_all['task_region_fraction'])} versus P {percent(p_all['task_region_fraction'])}; "
        f"P-O was {estimate_text(paired_all['P_minus_O_task_region_fraction'], 100., 2)} percentage points.",
        "",
        f"Fork rows were {o_fork['rows']:,}/{o_all['rows']:,} ({percent(o_fork['rows'] / o_all['rows'])}), "
        f"supported by {o_fork['episodes']:,} source episodes and {o_fork['unique_cached_anchors']:,} "
        f"cached anchor identities. Their task-region fraction was O "
        f"{percent(o_fork['task_region_fraction'])} versus P {percent(p_fork['task_region_fraction'])}; "
        f"P-O was {estimate_text(paired_fork['P_minus_O_task_region_fraction'], 100., 2)} percentage points. "
        "Each production batch contained exactly 128 ordinary, 64 down-action fork, and 64 "
        "right-action fork rows. Cache duplication and supporting-episode counts are in coverage.json.",
        "",
        "These are achieved future-goal marginals, not returns. A complete path reaching the task "
        "region, a discounted sampled future in that region, and a canonical-goal critic score remain "
        "separate quantities.",
        "",
        "## 3. Did the unchanged critic learn a different task preference?",
        "",
        f"At 656 held-out first observable fork contexts, the endpoint down-minus-right canonical-goal "
        f"gap was initially {estimate_text(critic['initial']['endpoint_down_minus_right'])}, after O "
        f"{estimate_text(critic['O']['endpoint_down_minus_right'])}, and after P "
        f"{estimate_text(critic['P']['endpoint_down_minus_right'])}. The paired P-O endpoint-gap change "
        f"was {estimate_text(critic_delta)}. This is "
        f"{'evidence of a treatment-specific score-preference change' if interval_excludes_zero(critic_delta) else 'not a resolved treatment-specific score-preference change'} "
        "under the episode bootstrap; it is not a calibrated return difference.",
        "",
        f"On common fixed held-out batches, P-minus-O NCE loss was "
        f"{estimate_text(critic['paired_P_minus_O_observational_nce_loss'])} for observational goals "
        f"and {estimate_text(critic['paired_P_minus_O_pessimistic_nce_loss'])} for ETT goals. These "
        "intervals resample fixed evaluation batches because all-pairs negatives make an individual "
        "row loss non-separable. Local action derivatives at actual actor samples are saved in "
        "offline_evaluation_arrays.npz and summarized in results.json.",
        "",
        "## 4. Did the unchanged actor objective translate the preference?",
        "",
        f"At the same canonical-goal fork contexts, initial/O/P downward probabilities were "
        f"{percent(actor['initial']['canonical']['down_probability'])}, "
        f"{percent(actor['O']['canonical']['down_probability'])}, and "
        f"{percent(actor['P']['canonical']['down_probability'])}; rightward probabilities were "
        f"{percent(actor['initial']['canonical']['right_probability'])}, "
        f"{percent(actor['O']['canonical']['right_probability'])}, and "
        f"{percent(actor['P']['canonical']['right_probability'])}. Paired P-O changes were "
        f"{estimate_text(actor_delta['down_probability'], 100., 2)} percentage points down and "
        f"{estimate_text(actor_delta['right_probability'], 100., 2)} percentage points right. "
        f"The continuous P-O mean-action shifts were dx {estimate_text(actor_delta['action_x_mean'])} "
        f"and dy {estimate_text(actor_delta['action_y_mean'])}.",
        "",
        f"Both arms applied all 1,000 actor updates and moved from the common initial actor "
        f"(parameter L2 deltas O {training['actor_O']['policy_parameter_delta_l2']:.4f}, "
        f"P {training['actor_P']['policy_parameter_delta_l2']:.4f}). The mean raw BC/critic gradient "
        f"cosines at the final actors were O "
        f"{results['gradient_components']['O_final']['component_cosine']:.4f} and P "
        f"{results['gradient_components']['P_final']['component_cosine']:.4f}. Non-fork ordinary-goal "
        "retention, local critic gradients, policy scales, continuous samples, and optimizer-applied "
        "changes are all retained in the numerical outputs. Any action-distribution change is only a "
        "strictly offline policy response, not measured native route entry or success.",
        "",
        "## What is established and what is unresolved",
        "",
        "Established: an eligible-data-only ETT optimization can be connected to complete cached "
        "continuations; its future goals can replace all production positives at exactly matched "
        "offline anchors; and the unchanged critic/actor response can be measured end to end with "
        "full-F4 lineage. All sealed lineage, frozen-tree, padding, first-action, update-count, hash, "
        f"and static no-environment checks passed ({verification['checks_passed']} checks).",
        "",
        "Unresolved: the ETT is not a certified worst-case transition kernel; native success and true "
        "counterfactual outcomes were intentionally not measured; the initial production checkpoint "
        "had already seen all 6,600 episodes; and no eligible evidence supplies an irreversible-failure "
        "entry probability. This one-seed bounded result is exploratory evidence, not causal "
        "identification or a lower bound.",
        "",
        "## Recommended next intervention",
        "",
        recommendation,
        "",
        "## Reproduction",
        "",
        "Run from the PointMaze worktree with a fresh output directory:",
        "",
        "```powershell",
        "python -m pytest -q scripts/test_pointmaze_offline_causal_integration.py",
        "python -m ett.pointmaze_offline_causal_integration all --out outputs/pointmaze_offline_causal_integration_20260915_v1",
        "```",
        "",
        "The final checkpoints are fixed iterates, not validation-selected checkpoints. No commit or "
        "push is part of this experiment.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = sorted(p for p in out.rglob("*")
                   if p.is_file() and p.name != "completion_manifest.json")
    write_json(out / "completion_manifest.json", {
        "status": "complete", "files": len(files),
        "sha256": {p.relative_to(out).as_posix(): file_sha(p) for p in files},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print("English report and completion manifest written", flush=True)


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def verify(out: Path):
    obs, act, initial, train, heldout, plan = load_prepared(out)
    provenance = read_json(out / "provenance.json")
    seal = read_json(out / "seal.json")
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    check(read_json(out / "config.json") == CONFIG, "sealed config equals implementation config")
    check(seal["sealed_before_optimizer_updates"] and seal["optimizer_updates_so_far"] == 0,
          "protocol sealed before optimizer updates")
    for name, digest in seal["sha256"].items():
        check(file_sha(out / name) == digest, f"sealed hash unchanged: {name}")
    for path, digest in EXPECTED.items():
        check(file_sha(Path(path)) == digest, f"eligible input hash unchanged: {path}")
    for path, digest in provenance["source_sha256"].items():
        check(file_sha(Path(path)) == digest, f"sealed source hash unchanged: {path}")
    for path, digest in provenance["historical_sha256_before"].items():
        check(file_sha(Path(path)) == digest, f"historical artifact unchanged: {path}")

    modules = imported_modules(Path(__file__))
    forbidden = ("gym", "gymnasium", "dm_control", "native_alive_motion",
                 "true_transition", "oracle")
    check(not any(any(token in module.lower() for token in forbidden) for module in modules),
          "driver imports no environment, native, true-transition, or oracle module")
    check(CONFIG["environment_calls"] == 0 and CONFIG["native_steps"] == 0,
          "sealed environment and native-step budgets are zero")
    eligible_runtime = {str(DATASET), str(DATASET_MANIFEST), str(INITIAL),
                        str(INITIAL_PROVENANCE), str(NOMINAL), str(NOMINAL_CONFIG),
                        str(NOMINAL_PROVENANCE), str(DIAGONAL), str(DIAGONAL_HISTORY),
                        str(ROUTING_MANIFEST)}
    recorded_runtime = set(EXPECTED)
    check(recorded_runtime == eligible_runtime,
          "runtime artifact whitelist contains only eligible offline inputs and metadata")
    check(provenance["failure_mechanism"]["explicit_failure_latent"] is False and
          provenance["failure_mechanism"]["stationary_atom_is_not_death"] is True,
          "unsupported failure entry is not fabricated")
    check(np.array_equal(train, split_ids()[0]) and np.array_equal(heldout, split_ids()[1]),
          "established episode split preserved")
    validate_plan(plan, obs, act, train, heldout)
    checks.append("sampler plan strata, split, and strictly-later offsets valid")

    final_theta = np.load(out / "checkpoints" / "ett_final.npz", allow_pickle=False)["theta"]
    cache_specs = (("train_trajectory_pool.npz", "train_pool", final_theta),
                   ("validation_trajectory_pool.npz", "validation_pool", final_theta),
                   ("initial_validation_trajectory_pool.npz", "validation_pool",
                    np.zeros(48, np.float32)))
    loaded_cache = {}
    for filename, prefix, theta in cache_specs:
        cache = load_cache(out / filename); loaded_cache[filename] = cache
        episode, anchor_time = plan[prefix + "_episode"], plan[prefix + "_time"]
        check(np.array_equal(cache["episode"], episode) and
              np.array_equal(cache["anchor_time"], anchor_time),
              f"{filename}: anchor identities exact")
        check(np.array_equal(cache["states"][:, 0], obs[episode, anchor_time, :8]),
              f"{filename}: recorded anchor states exact")
        check(np.array_equal(cache["action"][:, 0], act[episode, anchor_time]) and
              np.array_equal(cache["anchor_action"], act[episode, anchor_time]),
              f"{filename}: recorded first actions exact")
        check(str(cache["theta_sha256"]) == array_sha(theta),
              f"{filename}: ETT version identifier exact")
        check(str(cache["base_diagonal_file_sha256"]) == EXPECTED[str(DIAGONAL)] and
              str(cache["nominal_file_sha256"]) == EXPECTED[str(NOMINAL)] and
              str(cache["fixed_actor_file_sha256"]) == EXPECTED[str(INITIAL)],
              f"{filename}: component identifiers exact")
        state_valid = np.arange(51)[None, :] <= cache["length"][:, None]
        transition_valid = np.arange(50)[None, :] < cache["length"][:, None]
        check(np.array_equal(cache["valid"], transition_valid),
              f"{filename}: validity mask exact")
        check(np.isfinite(cache["states"][state_valid]).all() and
              np.isnan(cache["states"][~state_valid]).all(),
              f"{filename}: state padding cannot be sampled")
        check(np.isfinite(cache["action"][transition_valid]).all() and
              np.isnan(cache["action"][~transition_valid]).all(),
              f"{filename}: action padding explicit")
        history_error = max(float(np.max(np.abs(cache["states"][i, 1:h + 1, 2:] -
                                                  cache["states"][i, :h, :6])))
                            for i, h in enumerate(cache["length"]))
        check(history_error == 0., f"{filename}: full-F4 history shifts exact")

    lineage = np.load(out / "nce_row_lineage.npz", allow_pickle=False)
    ids, offsets = lineage["pool_id"], lineage["future_offset"]
    check(np.array_equal(ids, plan["critic_pool_id"]) and
          np.array_equal(offsets, plan["critic_offset"]) and
          np.array_equal(lineage["source"], plan["critic_source"]),
          "O/P production row identity, strata, and future offsets matched")
    episode = plan["train_pool_episode"][ids]
    anchor_time = plan["train_pool_time"][ids]
    expected_o = obs[episode, anchor_time + offsets, :8]
    expected_p = loaded_cache["train_trajectory_pool.npz"]["states"][ids, offsets]
    check(np.array_equal(lineage["observational_goal"], expected_o),
          "every O positive is the matching recorded future")
    check(np.array_equal(lineage["pessimistic_goal"], expected_p),
          "every P positive is the matching final-ETT future")
    check(np.isfinite(expected_p).all(), "no padded P future entered NCE")

    training = read_json(out / "production_training.json")
    check(training["integrity"]["P_positive_source"].startswith("100%") and
          training["integrity"]["O_positive_source"].startswith("100%"),
          "production positive-source fractions are 100% O and 100% P")
    for arm in ("O", "P"):
        cstep, critic_state = checkpoint.load_checkpoint(
            out / "checkpoints" / f"critic_{arm}_final.pkl")
        astep, actor_state = checkpoint.load_checkpoint(
            out / "checkpoints" / f"actor_{arm}_final.pkl")
        check(cstep == 150400 and astep == 151400,
              f"{arm}: fixed checkpoint update counts exact")
        check(tree_sha(critic_state.policy_params) == tree_sha(initial.policy_params) and
              tree_sha(critic_state.policy_optimizer_state) ==
              tree_sha(initial.policy_optimizer_state),
              f"{arm}: actor and actor optimizer frozen during critic stage")
        check(tree_sha(actor_state.q_params) == tree_sha(critic_state.q_params) and
              tree_sha(actor_state.q_optimizer_state) ==
              tree_sha(critic_state.q_optimizer_state),
              f"{arm}: critic and critic optimizer frozen during actor stage")
        check(tree_delta_norm(actor_state.policy_params, initial.policy_params) > 0.,
              f"{arm}: actor optimizer applied nonzero changes")
    curves = np.load(out / "learning_curves.npz", allow_pickle=False)
    check(curves["critic_O"].shape == (400, 8) and curves["critic_P"].shape == (400, 8),
          "exactly 400 production critic updates per arm")
    check(curves["actor_O"].shape == (1000, 9) and curves["actor_P"].shape == (1000, 9),
          "exactly 1000 actor updates per arm")
    ett = read_json(out / "ett_training.json")
    ledger = read_json(out / "model_ledger.json")
    check(ett["auxiliary_updates"] == 2200 and len(ett["history"]) == 3,
          "ETT and auxiliary update budgets exact")
    check(ledger["charged"] == ett["model_outputs"] and ledger["charged"] <= ledger["cap"],
          "model-generation ledger within sealed cap")
    check(read_json(out / "results.json")["sampling"]["matched_future_change"]
          ["P_positive_fraction_from_ETT"] == 1., "evaluation confirms all P positives use ETT")

    payload = {
        "status": "passed", "checks_passed": len(checks), "checks": checks,
        "environment_interactions": 0, "native_steps": 0,
        "runtime_artifact_whitelist": sorted(eligible_runtime),
        "excluded_native_or_oracle_training_inputs": [],
        "historical_artifacts_modified": False,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    }
    write_json(out / "verification.json", payload)
    print("verification passed", len(checks), "checks", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit-ett", "train", "evaluate",
                                          "verify", "finalize", "all"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.phase in ("prepare", "all"): prepare(args.out)
    if args.phase in ("fit-ett", "all"): fit_ett(args.out)
    if args.phase in ("train", "all"): train(args.out)
    if args.phase in ("evaluate", "all"): evaluate(args.out)
    if args.phase in ("verify", "all"): verify(args.out)
    if args.phase in ("finalize", "all"): finalize(args.out)


if __name__ == "__main__":
    main()
