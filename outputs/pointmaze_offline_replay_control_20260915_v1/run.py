"""Strictly offline PointMaze frequency-vs-route replay control.

This program intentionally has no environment or transition-model imports.  Its
only learned-data reads are the ``obs`` and ``act`` members of one frozen NPZ and
one original full learner checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from crl import checkpoint, networks
from crl.losses import Transition


OUT = Path("outputs/pointmaze_offline_replay_control_20260915_v1")
DATASET = Path("artifacts/f4_p30_server_30076/results/datasets/swamp_windy_f4_merged_s0.npz")
DATASET_MANIFEST = DATASET.with_name(DATASET.name + ".manifest.json")
INITIAL = Path("artifacts/f4_p30_server_30076/results/runs/f4_p30_sweep/p30_a0_a01_a03_s0_s1/alpha0_seed0/final.pkl")
INITIAL_PROVENANCE = INITIAL.with_name("arm_provenance.json")

EXPECTED_DATASET_SHA = "83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c"
EXPECTED_DATASET_CONTENT_SHA = "ad8b4470df0ebcc1c03421dfb59b61f9faee48cfbf51e2b9c0a402e0403f544d"
EXPECTED_INITIAL_SHA = "ea8a71d3cb8d963259a54d47250b454462dae2ae62f03a55c370ca81a2c8ec54"
EXPECTED_ANCESTRY_COMMIT = "4f3da3f1e94cbad647bc56b0f24e4171776090ee"

GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
PROBES = np.array([
    [0.0, -1.0], [-0.1, -0.9], [0.0, -0.9], [0.1, -0.9],
    [1.0, 0.0], [0.9, -0.1], [0.9, 0.0], [0.9, 0.1],
], np.float32)

CONFIG = {
    "experiment": "strictly_offline_pointmaze_replay_control",
    "source_commit": "5acf76669f1378aaa88809c390fe3f3b625aff71",
    "batch_size": 256,
    "critic_updates": 400,
    "actor_updates": 1000,
    "discount": 0.95,
    "critic_learning_rate": 3e-4,
    "actor_learning_rate": 3e-4,
    "adam_eps": 1e-7,
    "tau": 0.005,
    "bc_coef": 0.5,
    "random_goals": 0.5,
    "entropy_coefficient": 0.0,
    "fork_region": {"x_min": 1.0, "x_max_exclusive": 2.0,
                    "y_min": 3.0, "y_max_exclusive": 4.0},
    "down": "action_y <= -0.5 and abs(action_y) >= abs(action_x)",
    "right": "action_x >= 0.5 and abs(action_x) > abs(action_y)",
    "R_batch": {"ordinary": 128, "down": 64, "right": 64},
    "seeds": {
        "critic_ordinary": 530915000,
        "critic_focus": 530915001,
        "critic_shuffle": 530915002,
        "actor_replay": 530915100,
        "actor_keys": 530915101,
        "eval_ordinary": 530915200,
        "eval_focus": 530915201,
        "eval_context_goal": 530915202,
        "eval_away": 530915203,
        "eval_actor_fork_eps": 530915204,
        "eval_actor_away_eps": 530915205,
        "bootstrap": 530915300,
    },
    "eval_nce_batches": 32,
    "eval_fork_actor_samples": 128,
    "eval_local_gradient_samples": 8,
    "eval_away_rows": 1024,
    "eval_away_actor_samples": 64,
    "gradient_batches": 4,
    "bootstrap_replicates": 2000,
    "final_checkpoint_rule": "fixed final iterate only",
    "numerical_failure": "abort on any nonfinite loss, gradient, update, or parameter",
}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def array_sha(x) -> str:
    a = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def tree_sha(tree) -> str:
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves(tree):
        a = np.ascontiguousarray(x)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def tree_norm(tree) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(x, np.float64) ** 2))
                             for x in jax.tree_util.tree_leaves(tree))))


def tree_delta_norm(a, b) -> float:
    return tree_norm(jax.tree_util.tree_map(lambda x, y: x - y, a, b))


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return x.as_posix()
    raise TypeError(type(x).__name__)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_network_and_optimizers():
    network = networks.make_networks(
        8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
        hidden_layer_sizes=(256, 256), actor_min_std=1e-6,
        twin_q=False, use_image_obs=False, use_layer_norm=False,
        obs_scale=None)
    q_opt = optax.adam(CONFIG["critic_learning_rate"], eps=CONFIG["adam_eps"])
    p_opt = optax.adam(CONFIG["actor_learning_rate"], eps=CONFIG["adam_eps"])
    return network, q_opt, p_opt


def load_eligible():
    assert file_sha(DATASET) == EXPECTED_DATASET_SHA
    assert file_sha(INITIAL) == EXPECTED_INITIAL_SHA
    manifest = read_json(DATASET_MANIFEST)
    provenance = read_json(INITIAL_PROVENANCE)
    assert manifest["sha256"] == EXPECTED_DATASET_SHA
    assert manifest["content_sha256"] == EXPECTED_DATASET_CONTENT_SHA
    assert provenance["dataset_content_sha256"] == EXPECTED_DATASET_CONTENT_SHA
    assert provenance["code_commit"] == EXPECTED_ANCESTRY_COMMIT
    assert provenance["arm"] == "zbase" and provenance["alpha"] == 0.0
    assert provenance["seed"] == 0 and provenance["steps"] == 150000
    # Deliberately access exactly these two saved members, and no audit arrays.
    with np.load(DATASET, allow_pickle=False) as z:
        obs = np.asarray(z["obs"])
        act = np.asarray(z["act"])
    assert obs.shape == (6600, 51, 16) and act.shape == (6600, 51, 2)
    assert np.isfinite(obs).all() and np.isfinite(act[:, :50]).all()
    np.testing.assert_array_equal(obs[:, 1:, 2:8], obs[:, :-1, :6])
    np.testing.assert_array_equal(obs[:, :, 8:], np.broadcast_to(GOAL, obs[:, :, 8:].shape))
    step, state = checkpoint.load_checkpoint(INITIAL)
    assert step == 150000
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
    return obs, act, state, manifest, provenance


def split_ids():
    heldout = np.random.default_rng(0).permutation(6600)[:660].astype(np.int32)
    train = np.setdiff1d(np.arange(6600, dtype=np.int32), heldout)
    assert len(train) == 5940 and len(heldout) == 660
    assert not np.intersect1d(train, heldout).size
    return train, heldout


def fork_mask(obs_rows):
    xy = obs_rows[..., :2]
    return ((xy[..., 0] >= 1.0) & (xy[..., 0] < 2.0) &
            (xy[..., 1] >= 3.0) & (xy[..., 1] < 4.0))


def direction_masks(actions):
    down = ((actions[..., 1] <= -0.5) &
            (np.abs(actions[..., 1]) >= np.abs(actions[..., 0])))
    right = ((actions[..., 0] >= 0.5) &
             (np.abs(actions[..., 0]) > np.abs(actions[..., 1])))
    assert not np.any(down & right)
    return down, right


def draw_future(rng, times):
    times = np.asarray(times, np.int32)
    arange = np.arange(51, dtype=np.int32)
    valid = arange[None, :] > times[:, None]
    logits = np.where(valid,
                      (arange[None, :] - times[:, None]) * math.log(CONFIG["discount"]),
                      -np.inf)
    u = rng.uniform(size=logits.shape).clip(1e-20, 1.0)
    gumbel = -np.log(-np.log(u))
    future = np.argmax(logits + gumbel, axis=1).astype(np.int32)
    assert np.all(future > times) and np.all(future <= 50)
    return future


def draw_ordinary(rng, episode_ids, batches, size=256):
    e = np.empty((batches, size), np.int32)
    t = np.empty_like(e)
    f = np.empty_like(e)
    for u in range(batches):
        local = rng.integers(0, len(episode_ids), size=size)
        e[u] = episode_ids[local]
        t[u] = rng.integers(0, 50, size=size)
        f[u] = draw_future(rng, t[u])
    return e, t, f


def eligible_anchors(obs, act, episode_ids):
    fm = fork_mask(obs[episode_ids, :50])
    dm, rm = direction_masks(act[episode_ids, :50])
    out = {}
    for name, mask in (("fork", fm), ("down", fm & dm), ("right", fm & rm)):
        local_e, t = np.nonzero(mask)
        out[name] = (episode_ids[local_e].astype(np.int32), t.astype(np.int32))
    return out


def draw_focus(rng, anchors, batches, size):
    ae, at = anchors
    e = np.empty((batches, size), np.int32)
    t = np.empty_like(e)
    f = np.empty_like(e)
    for u in range(batches):
        ix = rng.integers(0, len(ae), size=size)
        e[u], t[u] = ae[ix], at[ix]
        f[u] = draw_future(rng, t[u])
    return e, t, f


def concentration(episode, time_index):
    key = np.asarray(episode, np.int64).ravel() * 51 + np.asarray(time_index).ravel()
    _, count = np.unique(key, return_counts=True)
    p = count / count.sum()
    return {
        "draws": int(count.sum()),
        "unique_anchors": int(len(count)),
        "duplicate_draw_fraction": float(1 - len(count) / count.sum()),
        "max_multiplicity": int(count.max()),
        "top_anchor_share": float(count.max() / count.sum()),
        "empirical_effective_anchor_count": float(1 / np.sum(p ** 2)),
    }


def make_plan(obs, act, train, heldout):
    s = CONFIG["seeds"]
    ordinary = draw_ordinary(np.random.default_rng(s["critic_ordinary"]), train,
                             CONFIG["critic_updates"])
    tr_anchor = eligible_anchors(obs, act, train)
    ho_anchor = eligible_anchors(obs, act, heldout)
    frng = np.random.default_rng(s["critic_focus"])
    de, dt, df = draw_focus(frng, tr_anchor["down"], CONFIG["critic_updates"], 64)
    re, rt, rf = draw_focus(frng, tr_anchor["right"], CONFIG["critic_updates"], 64)
    R_e = np.concatenate([ordinary[0][:, :128], de, re], axis=1)
    R_t = np.concatenate([ordinary[1][:, :128], dt, rt], axis=1)
    R_f = np.concatenate([ordinary[2][:, :128], df, rf], axis=1)
    R_source = np.broadcast_to(np.repeat(np.array([0, 1, 2], np.int8), [128, 64, 64]),
                               R_e.shape).copy()
    srng = np.random.default_rng(s["critic_shuffle"])
    for u in range(CONFIG["critic_updates"]):
        p = srng.permutation(256)
        R_e[u], R_t[u], R_f[u], R_source[u] = (
            R_e[u, p], R_t[u, p], R_f[u, p], R_source[u, p])

    actor = draw_ordinary(np.random.default_rng(s["actor_replay"]), train,
                          CONFIG["actor_updates"])
    actor_keys = np.asarray(jax.random.split(jax.random.PRNGKey(s["actor_keys"]),
                                             CONFIG["actor_updates"]), np.uint32)

    eval_ord = draw_ordinary(np.random.default_rng(s["eval_ordinary"]), heldout,
                             CONFIG["eval_nce_batches"])
    efrng = np.random.default_rng(s["eval_focus"])
    ede, edt, edf = draw_focus(efrng, ho_anchor["down"],
                               CONFIG["eval_nce_batches"], 128)
    ere, ert, erf = draw_focus(efrng, ho_anchor["right"],
                               CONFIG["eval_nce_batches"], 128)
    eval_f_e = np.concatenate([ede, ere], 1)
    eval_f_t = np.concatenate([edt, ert], 1)
    eval_f_f = np.concatenate([edf, erf], 1)
    eval_f_source = np.broadcast_to(np.repeat(np.array([1, 2], np.int8), 128),
                                    eval_f_e.shape).copy()
    for u in range(CONFIG["eval_nce_batches"]):
        p = efrng.permutation(256)
        eval_f_e[u], eval_f_t[u], eval_f_f[u], eval_f_source[u] = (
            eval_f_e[u, p], eval_f_t[u, p], eval_f_f[u, p], eval_f_source[u, p])

    # Exactly one observable context per held-out episode: its first fork row.
    ctx_e, ctx_t = [], []
    for ep in heldout:
        times = np.flatnonzero(fork_mask(obs[ep, :50]))
        if len(times):
            ctx_e.append(ep)
            ctx_t.append(times[0])
    ctx_e = np.asarray(ctx_e, np.int32)
    ctx_t = np.asarray(ctx_t, np.int32)
    assert len(ctx_e) == len(np.unique(ctx_e))
    ctx_future = draw_future(np.random.default_rng(s["eval_context_goal"]), ctx_t)

    # A fixed held-out ordinary-goal cohort away from the fork, no replacement.
    he, ht = np.indices((len(heldout), 50))
    candidate = np.column_stack([heldout[he.ravel()], ht.ravel()]).astype(np.int32)
    candidate = candidate[~fork_mask(obs[candidate[:, 0], candidate[:, 1]])]
    arng = np.random.default_rng(s["eval_away"])
    choose = arng.choice(len(candidate), CONFIG["eval_away_rows"], replace=False)
    away_e, away_t = candidate[choose].T
    away_f = draw_future(arng, away_t)

    fork_eps = np.random.default_rng(s["eval_actor_fork_eps"]).normal(
        size=(len(ctx_e), CONFIG["eval_fork_actor_samples"], 2)).astype(np.float32)
    away_eps = np.random.default_rng(s["eval_actor_away_eps"]).normal(
        size=(len(away_e), CONFIG["eval_away_actor_samples"], 2)).astype(np.float32)

    arrays = {
        "critic_F_episode": ordinary[0], "critic_F_time": ordinary[1],
        "critic_F_future": ordinary[2],
        "critic_R_episode": R_e, "critic_R_time": R_t, "critic_R_future": R_f,
        "critic_R_source": R_source,
        "actor_episode": actor[0], "actor_time": actor[1], "actor_future": actor[2],
        "actor_keys": actor_keys,
        "eval_ordinary_episode": eval_ord[0], "eval_ordinary_time": eval_ord[1],
        "eval_ordinary_future": eval_ord[2],
        "eval_focus_episode": eval_f_e, "eval_focus_time": eval_f_t,
        "eval_focus_future": eval_f_f, "eval_focus_source": eval_f_source,
        "fork_context_episode": ctx_e, "fork_context_time": ctx_t,
        "fork_context_future": ctx_future,
        "away_episode": away_e, "away_time": away_t, "away_future": away_f,
        "fork_eps": fork_eps, "away_eps": away_eps,
        "action_probes": PROBES,
    }
    coverage = {}
    for part, ids, anchors in (("train", train, tr_anchor),
                               ("heldout", heldout, ho_anchor)):
        coverage[part] = {
            name: {"anchors": int(len(v[0])), "episodes": int(len(np.unique(v[0])))}
            for name, v in anchors.items()
        }
        coverage[part]["episodes_total"] = int(len(ids))
    coverage["R_training_exposure"] = {
        "down": concentration(de, dt),
        "right": concentration(re, rt),
        "ordinary": concentration(ordinary[0][:, :128], ordinary[1][:, :128]),
        "source_rows": {str(k): int(np.sum(R_source == k)) for k in (0, 1, 2)},
    }
    coverage["F_training_exposure"] = concentration(ordinary[0], ordinary[1])
    return arrays, coverage


def validate_indices(plan, obs, act, train, heldout):
    train_set = set(map(int, train))
    heldout_set = set(map(int, heldout))
    for prefix, allowed in (("critic_F", train_set), ("critic_R", train_set),
                            ("actor", train_set), ("eval_ordinary", heldout_set),
                            ("eval_focus", heldout_set)):
        e, t, f = (plan[prefix + "_episode"], plan[prefix + "_time"],
                   plan[prefix + "_future"])
        assert set(map(int, np.unique(e))) <= allowed
        assert np.all((t >= 0) & (t < 50))
        assert np.all((f > t) & (f <= 50))
    src = plan["critic_R_source"]
    assert np.all(np.sum(src == 0, axis=1) == 128)
    assert np.all(np.sum(src == 1, axis=1) == 64)
    assert np.all(np.sum(src == 2, axis=1) == 64)
    e, t = plan["critic_R_episode"], plan["critic_R_time"]
    fm = fork_mask(obs[e, t])
    dm, rm = direction_masks(act[e, t])
    assert np.all(fm[src != 0])
    assert np.all(dm[src == 1]) and np.all(rm[src == 2])
    es = plan["eval_focus_source"]
    e, t = plan["eval_focus_episode"], plan["eval_focus_time"]
    fm = fork_mask(obs[e, t])
    dm, rm = direction_masks(act[e, t])
    assert np.all(fm) and np.all(dm[es == 1]) and np.all(rm[es == 2])
    # After the sealed within-batch shuffle, R's ordinary multiset must still be
    # exactly the first half of F's common ordinary draw (episode, time, future).
    for u in range(src.shape[0]):
        got = np.column_stack([plan["critic_R_episode"][u, src[u] == 0],
                               plan["critic_R_time"][u, src[u] == 0],
                               plan["critic_R_future"][u, src[u] == 0]])
        want = np.column_stack([plan["critic_F_episode"][u, :128],
                                plan["critic_F_time"][u, :128],
                                plan["critic_F_future"][u, :128]])
        order_g = np.lexsort((got[:, 2], got[:, 1], got[:, 0]))
        order_w = np.lexsort((want[:, 2], want[:, 1], want[:, 0]))
        assert np.array_equal(got[order_g], want[order_w])


def prepare():
    if (OUT / "seal.json").exists():
        raise RuntimeError("sealed preparation already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    obs, act, state, manifest, provenance = load_eligible()
    train, heldout = split_ids()
    plan, coverage = make_plan(obs, act, train, heldout)
    validate_indices(plan, obs, act, train, heldout)
    np.savez_compressed(OUT / "partition.npz", train=train, heldout=heldout)
    np.savez_compressed(OUT / "sampling_plan.npz", **plan)
    write_json(OUT / "config.json", CONFIG)
    write_json(OUT / "coverage.json", coverage)

    historical = [
        Path("outputs/pointmaze_matched_fork_20260914_v1/REPORT.md"),
        Path("outputs/pointmaze_matched_fork_20260914_v1/PROTOCOL.md"),
        Path("outputs/pointmaze_critic_control_20260914_v1/REPORT.md"),
        Path("outputs/pointmaze_critic_control_20260914_v1/PROTOCOL.md"),
        Path("artifacts/pointmaze_region_pilot/phase_sampling_s01_v1/REPORT.md"),
        Path("artifacts/pointmaze_region_pilot/phase_sampling_s01_v1/PROTOCOL.md"),
        Path("outputs/pointmaze_oracle_training_pilot_20260914_v1/REPORT.md"),
        Path("outputs/pointmaze_oracle_training_pilot_20260914_v1/PROTOCOL.md"),
        Path("ett/run_policy_improvement.py"), Path("ett/policy_improvement.py"),
        Path("crl/losses.py"), Path("crl/networks.py"), Path("crl/replay.py"),
    ]
    hhash = {p.as_posix(): file_sha(p) for p in historical}
    prov = {
        "dataset": {
            "path": DATASET.as_posix(), "file_sha256": EXPECTED_DATASET_SHA,
            "content_sha256": EXPECTED_DATASET_CONTENT_SHA,
            "episodes": 6600, "rows_per_episode": 51,
            "training_members": "partition.npz:train (5,940 complete episodes)",
            "heldout_members": "partition.npz:heldout (660 complete episodes)",
            "loaded_members": ["obs", "act"],
            "not_loaded": manifest["meta"]["audit_fields"] + ["meta"],
            "source_membership": manifest["meta"]["merge"],
        },
        "initial_checkpoint": {
            "path": INITIAL.as_posix(), "file_sha256": EXPECTED_INITIAL_SHA,
            "step": 150000, "ancestry_commit": EXPECTED_ANCESTRY_COMMIT,
            "dataset_content_sha256": provenance["dataset_content_sha256"],
            "full_state_sha256": tree_sha(state),
            "policy_sha256": tree_sha(state.policy_params),
            "critic_sha256": tree_sha(state.q_params),
            "actor_optimizer_sha256": tree_sha(state.policy_optimizer_state),
            "critic_optimizer_sha256": tree_sha(state.q_optimizer_state),
            "eligible": True,
            "ancestry": "original zbase alpha=0 seed=0; fixed dataset only; no failure bank",
        },
        "split": {
            "method": "numpy default_rng(0).permutation(6600)[:660]",
            "train_sha256": array_sha(train), "heldout_sha256": array_sha(heldout),
            "limitation": "initial step-150000 checkpoint saw all 6,600 episodes",
        },
        "excluded": [
            {"commit": "b3d1423", "reason": "new native matched-fork diagnostic data"},
            {"commit": "5acf766", "reason": "new native critic-control data and trained critics"},
            {"name": "C1", "reason": "checkpoint ancestry includes oracle-generated replay"},
            {"name": "oracle pilot B/C", "reason": "C is contaminated; comparison not equivalent and both joint-train actors"},
        ],
        "actual_training_inputs": [DATASET.as_posix() + "::{obs,act}", INITIAL.as_posix()],
        "historical_sha256_before": hhash,
        "git": {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        },
    }
    write_json(OUT / "provenance.json", prov)
    write_json(OUT / "equivalence_review.json", {
        "equivalent_eligible_completed_experiment_found": False,
        "phase_sampling_s01_v1": "different generated dataset, region critic/objective, native final collection, zero actor updates",
        "oracle_training_pilot": "uses oracle-generated replay in C, joint actor/critic training, and is ineligible",
        "critic_control_20260914_v1": "uses newly collected native continuations, contaminated C1 initialization, zero actor updates",
        "decision": "run both sealed critic arms and both required actor arms",
    })
    sealed = ["PROTOCOL.md", "run.py", "config.json", "coverage.json",
              "provenance.json", "equivalence_review.json", "partition.npz",
              "sampling_plan.npz"]
    seal = {
        "sealed_before_optimizer_updates": True,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": {name: file_sha(OUT / name) for name in sealed},
        "initial_full_state_sha256_both_arms": tree_sha(state),
        "optimizer_updates_so_far": 0,
    }
    write_json(OUT / "seal.json", seal)
    print("prepared and sealed", OUT)


def load_plan_and_inputs():
    obs, act, initial, _, _ = load_eligible()
    part = np.load(OUT / "partition.npz", allow_pickle=False)
    train, heldout = part["train"], part["heldout"]
    plan_npz = np.load(OUT / "sampling_plan.npz", allow_pickle=False)
    plan = {k: plan_npz[k] for k in plan_npz.files}
    validate_indices(plan, obs, act, train, heldout)
    return obs, act, initial, train, heldout, plan


def batch_from_indices(obs, act, episode, time_index, future):
    state = obs[episode, time_index, :8].astype(np.float32, copy=False)
    goal = obs[episode, future, :8].astype(np.float32, copy=False)
    following = obs[episode, time_index + 1, :8].astype(np.float32, copy=False)
    action = act[episode, time_index].astype(np.float32, copy=False)
    n = len(np.asarray(episode).ravel())
    batch = Transition(
        np.concatenate([state, goal], -1), action,
        np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32),
        np.concatenate([following, goal], -1), np.full((n, 2), np.nan, np.float32))
    assert batch.observation.shape == (n, 16)
    return jax.tree_util.tree_map(jnp.asarray, batch)


def build_critic_step(network, optimizer):
    def objective(params, batch):
        logits = network.q_network.apply(params, batch.observation, batch.action)
        labels = jnp.eye(logits.shape[0], dtype=logits.dtype)
        matrix = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
        loss = jnp.mean(matrix)
        pos = jnp.mean(jnp.diag(logits))
        neg = (jnp.sum(logits) - jnp.sum(jnp.diag(logits))) / (logits.size - logits.shape[0])
        return loss, (pos, neg, pos - neg,
                      jnp.mean((logits > 0) == labels),
                      jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0])))

    grad_fn = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def step(params, opt_state, target, batch):
        (loss, aux), grad = grad_fn(params, batch)
        update, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, update)
        target = jax.tree_util.tree_map(
            lambda x, y: x * (1 - CONFIG["tau"]) + y * CONFIG["tau"], target, params)
        return params, opt_state, target, loss, aux, optax.global_norm(grad), optax.global_norm(update)
    return step


def actor_parts(network, policy_params, q_params, batch, key):
    state = batch.observation[:, :8]
    goal = batch.observation[:, 8:]
    new_state = jnp.concatenate([state, state], axis=0)
    new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
    original_action = jnp.concatenate([batch.action, batch.action], axis=0)
    new_obs = jnp.concatenate([new_state, new_goal], axis=1)
    dist = network.policy_network.apply(policy_params, new_obs)
    action = network.sample(dist, key)
    log_prob = network.log_prob(dist, action)
    q_action = network.q_network.apply(q_params, new_obs, action)
    q_term = -jnp.diag(q_action)  # fixed alpha=0
    bc_nll = -network.log_prob(dist, original_action)
    return jnp.mean(bc_nll), jnp.mean(q_term), (
        jnp.mean(-log_prob), jnp.median(dist.scale),
        jnp.mean(jnp.abs(dist.loc)),
        jnp.mean((jnp.abs(jnp.tanh(dist.loc)) > .99).astype(jnp.float32)))


def build_actor_step(network, optimizer):
    def objective(policy_params, q_params, batch, key):
        bc, q, aux = actor_parts(network, policy_params, q_params, batch, key)
        return CONFIG["bc_coef"] * bc + (1 - CONFIG["bc_coef"]) * q, (bc, q, aux)
    grad_fn = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def step(policy_params, opt_state, q_params, batch, key):
        (loss, (bc, q, aux)), grad = grad_fn(policy_params, q_params, batch, key)
        update, opt_state = optimizer.update(grad, opt_state)
        policy_params = optax.apply_updates(policy_params, update)
        return policy_params, opt_state, loss, bc, q, aux, optax.global_norm(grad), optax.global_norm(update)
    return step


def finite_tree(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))


def train():
    if (OUT / "training_results.json").exists():
        raise RuntimeError("training already completed")
    obs, act, initial, _, _, plan = load_plan_and_inputs()
    network, q_opt, p_opt = make_network_and_optimizers()
    critic_step = build_critic_step(network, q_opt)
    training = {"initial": {
        "full": tree_sha(initial), "policy": tree_sha(initial.policy_params),
        "critic": tree_sha(initial.q_params),
        "actor_optimizer": tree_sha(initial.policy_optimizer_state),
        "critic_optimizer": tree_sha(initial.q_optimizer_state)}}
    curve_arrays = {}
    critic_states = {}
    for arm in ("F", "R"):
        state = initial
        rows = []
        for u in range(CONFIG["critic_updates"]):
            batch = batch_from_indices(obs, act, plan[f"critic_{arm}_episode"][u],
                                       plan[f"critic_{arm}_time"][u],
                                       plan[f"critic_{arm}_future"][u])
            q, qo, tq, loss, aux, gn, un = critic_step(
                state.q_params, state.q_optimizer_state, state.target_q_params, batch)
            values = np.asarray([loss, *aux, gn, un], np.float64)
            if not np.isfinite(values).all() or not finite_tree((q, qo, tq)):
                raise FloatingPointError((arm, u, values))
            state = state._replace(q_params=q, q_optimizer_state=qo, target_q_params=tq)
            rows.append(values)
            if (u + 1) % 50 == 0:
                print("critic", arm, u + 1, "loss", round(float(loss), 6), flush=True)
        assert tree_sha(state.policy_params) == tree_sha(initial.policy_params)
        assert tree_sha(state.policy_optimizer_state) == tree_sha(initial.policy_optimizer_state)
        checkpoint.save_named(OUT / "checkpoints", f"critic_{arm}_final", 150400, state)
        curve_arrays[f"critic_{arm}"] = np.asarray(rows)
        critic_states[arm] = state
        training[f"critic_{arm}"] = {
            "updates": 400, "final_loss": float(rows[-1][0]),
            "q_sha256": tree_sha(q), "q_optimizer_sha256": tree_sha(qo),
            "policy_sha256": tree_sha(state.policy_params),
            "policy_optimizer_sha256": tree_sha(state.policy_optimizer_state),
            "q_parameter_delta_l2": tree_delta_norm(q, initial.q_params),
        }

    actor_step = build_actor_step(network, p_opt)
    for arm in ("F", "R"):
        critic_state = critic_states[arm]
        state = initial._replace(q_params=critic_state.q_params,
                                 target_q_params=critic_state.target_q_params,
                                 q_optimizer_state=critic_state.q_optimizer_state)
        rows = []
        for u in range(CONFIG["actor_updates"]):
            batch = batch_from_indices(obs, act, plan["actor_episode"][u],
                                       plan["actor_time"][u], plan["actor_future"][u])
            pp, po, loss, bc, qterm, aux, gn, un = actor_step(
                state.policy_params, state.policy_optimizer_state, state.q_params,
                batch, jnp.asarray(plan["actor_keys"][u]))
            values = np.asarray([loss, bc, qterm, *aux, gn, un], np.float64)
            if not np.isfinite(values).all() or not finite_tree((pp, po)):
                raise FloatingPointError((arm, u, values))
            state = state._replace(policy_params=pp, policy_optimizer_state=po)
            rows.append(values)
            if (u + 1) % 100 == 0:
                print("actor", arm, u + 1, "loss", round(float(loss), 6), flush=True)
        assert tree_sha(state.q_params) == tree_sha(critic_state.q_params)
        checkpoint.save_named(OUT / "checkpoints", f"actor_{arm}_final", 151400, state)
        curve_arrays[f"actor_{arm}"] = np.asarray(rows)
        training[f"actor_{arm}"] = {
            "updates": 1000, "final_loss": float(rows[-1][0]),
            "policy_sha256": tree_sha(state.policy_params),
            "actor_optimizer_sha256": tree_sha(state.policy_optimizer_state),
            "critic_sha256": tree_sha(state.q_params),
            "policy_parameter_delta_l2": tree_delta_norm(state.policy_params, initial.policy_params),
            "mean_raw_gradient_l2": float(np.mean(np.asarray(rows)[:, -2])),
            "mean_optimizer_update_l2": float(np.mean(np.asarray(rows)[:, -1])),
            "final_raw_gradient_l2": float(rows[-1][-2]),
            "final_optimizer_update_l2": float(rows[-1][-1]),
        }
    np.savez_compressed(OUT / "training_curves.npz", **curve_arrays,
                        critic_columns=np.array(["loss", "positive_logit", "negative_logit",
                                                 "logit_gap", "binary_accuracy", "categorical_accuracy",
                                                 "raw_gradient_l2", "optimizer_update_l2"]),
                        actor_columns=np.array(["loss", "bc_nll", "critic_actor_term",
                                                "sample_entropy", "scale_median", "loc_abs_mean",
                                                "saturation_fraction", "raw_gradient_l2",
                                                "optimizer_update_l2"]))
    training["integrity"] = {
        "same_initial_full_state": True,
        "only_critic_sampling_differed": True,
        "actor_index_arrays_shared": array_sha(plan["actor_episode"]),
        "actor_goal_arrays_shared": array_sha(plan["actor_future"]),
        "actor_key_arrays_shared": array_sha(plan["actor_keys"]),
        "actor_updates_occurred": all(training[f"actor_{a}"]["policy_parameter_delta_l2"] > 0
                                      for a in ("F", "R")),
    }
    write_json(OUT / "training_results.json", training)
    print("critic and actor stages complete")


def paired_scores(network, q_params, observation, action):
    phi, psi = network.representation_network.apply(q_params, observation, action)
    return jnp.mean(jnp.sum(phi * psi, axis=1), axis=-1)


def nce_metrics(network, q_params, batch):
    logits = network.q_network.apply(q_params, batch.observation, batch.action)
    labels = jnp.eye(logits.shape[0], dtype=logits.dtype)
    matrix = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    diag = jnp.diag(logits)
    neg = (jnp.sum(logits) - jnp.sum(diag)) / (logits.size - logits.shape[0])
    return jnp.array([jnp.mean(matrix), jnp.mean(diag), neg,
                      jnp.mean(diag) - neg,
                      jnp.mean((logits > 0) == labels),
                      jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0]))])


def bootstrap_mean(values, seed):
    x = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(0, len(x), size=(CONFIG["bootstrap_replicates"], len(x)))], axis=1)
    return {"mean": float(np.mean(x)), "ci95": np.quantile(means, [.025, .975]).tolist()}


def route_summary(samples):
    down, right = direction_masks(samples)
    return {
        "down_probability": float(np.mean(down)),
        "right_probability": float(np.mean(right)),
        "other_probability": float(np.mean(~(down | right))),
        "action_x_mean": float(np.mean(samples[..., 0])),
        "action_y_mean": float(np.mean(samples[..., 1])),
        "action_x_std": float(np.std(samples[..., 0])),
        "action_y_std": float(np.std(samples[..., 1])),
        "action_x_quantiles": np.quantile(samples[..., 0], [.1, .25, .5, .75, .9]).tolist(),
        "action_y_quantiles": np.quantile(samples[..., 1], [.1, .25, .5, .75, .9]).tolist(),
    }


def action_distribution(network, policy_params, observation, eps):
    dist = network.policy_network.apply(policy_params, jnp.asarray(observation))
    loc, scale = np.asarray(dist.loc), np.asarray(dist.scale)
    samples = np.tanh(loc[:, None, :] + scale[:, None, :] * eps)
    return loc, scale, samples.astype(np.float32)


def local_q_gradients(network, q_params, observation, action):
    def total(a):
        return jnp.sum(paired_scores(network, q_params, observation, a))
    return jax.grad(total)(action)


def flat_dot(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    dot = sum(float(jnp.vdot(x, y)) for x, y in zip(la, lb))
    na = math.sqrt(sum(float(jnp.vdot(x, x)) for x in la))
    nb = math.sqrt(sum(float(jnp.vdot(x, x)) for x in lb))
    return dot, na, nb, dot / (na * nb) if na and nb else float("nan")


def component_gradients(network, policy_params, q_params, batch, key):
    bc_fn = lambda p: actor_parts(network, p, q_params, batch, key)[0]
    q_fn = lambda p: actor_parts(network, p, q_params, batch, key)[1]
    bc_value, bc_grad = jax.value_and_grad(bc_fn)(policy_params)
    q_value, q_grad = jax.value_and_grad(q_fn)(policy_params)
    dot, bn, qn, cosine = flat_dot(bc_grad, q_grad)
    combined = jax.tree_util.tree_map(lambda x, y: .5 * x + .5 * y, bc_grad, q_grad)
    return {
        "bc_nll": float(bc_value), "critic_actor_term": float(q_value),
        "bc_raw_gradient_l2": bn, "critic_raw_gradient_l2": qn,
        "bc_weighted_gradient_l2": .5 * bn,
        "critic_weighted_gradient_l2": .5 * qn,
        "component_dot": dot, "component_cosine": cosine,
        "combined_raw_gradient_l2": tree_norm(combined),
    }


def evaluate():
    if (OUT / "results.json").exists():
        raise RuntimeError("evaluation already completed")
    obs, act, initial, _, _, plan = load_plan_and_inputs()
    network, _, _ = make_network_and_optimizers()
    critic_states = {a: checkpoint.load_checkpoint(OUT / "checkpoints" / f"critic_{a}_final.pkl")[1]
                     for a in ("F", "R")}
    actor_states = {a: checkpoint.load_checkpoint(OUT / "checkpoints" / f"actor_{a}_final.pkl")[1]
                    for a in ("F", "R")}
    qsets = {"initial": initial.q_params, "F": critic_states["F"].q_params,
             "R": critic_states["R"].q_params}
    psets = {"initial": initial.policy_params, "F": actor_states["F"].policy_params,
             "R": actor_states["R"].policy_params}

    saved = {}
    result = {"critic": {}, "actor": {}, "gradient_components": {}}
    nce_fn = jax.jit(lambda q, b: nce_metrics(network, q, b))
    for qname, qparams in qsets.items():
        result["critic"][qname] = {}
        for cohort, prefix in (("ordinary", "eval_ordinary"), ("fork", "eval_focus")):
            values = []
            for u in range(CONFIG["eval_nce_batches"]):
                b = batch_from_indices(obs, act, plan[prefix + "_episode"][u],
                                       plan[prefix + "_time"][u], plan[prefix + "_future"][u])
                values.append(np.asarray(nce_fn(qparams, b)))
            values = np.asarray(values)
            saved[f"nce_{qname}_{cohort}"] = values
            result["critic"][qname][cohort] = dict(zip(
                ["loss", "positive_logit", "negative_logit", "logit_gap",
                 "binary_accuracy", "categorical_accuracy"], np.mean(values, axis=0).tolist()))

    ce, ct, cf = (plan["fork_context_episode"], plan["fork_context_time"],
                  plan["fork_context_future"])
    state = obs[ce, ct, :8].astype(np.float32)
    canonical_obs = np.concatenate([state, np.broadcast_to(GOAL, state.shape)], axis=1)
    ordinary_goal_obs = np.concatenate([state, obs[ce, cf, :8]], axis=1).astype(np.float32)
    saved.update(fork_context_episode=ce, fork_context_time=ct,
                 fork_context_future=cf, action_probes=PROBES)
    score_fn = jax.jit(lambda q, o, a: paired_scores(network, q, o, a))
    for qname, qparams in qsets.items():
        score = []
        for probe in PROBES:
            aa = np.broadcast_to(probe, (len(state), 2))
            score.append(np.asarray(score_fn(qparams, jnp.asarray(canonical_obs), jnp.asarray(aa))))
        score = np.stack(score, axis=1)
        saved[f"probe_scores_{qname}"] = score
        endpoint = score[:, 0] - score[:, 4]
        neighborhood = np.mean(score[:, :4], axis=1) - np.mean(score[:, 4:], axis=1)
        seed = CONFIG["seeds"]["bootstrap"] + {"initial": 0, "F": 10, "R": 20}[qname]
        result["critic"][qname]["endpoint_down_minus_right"] = bootstrap_mean(endpoint, seed)
        result["critic"][qname]["neighborhood_down_minus_right"] = bootstrap_mean(neighborhood, seed + 1)

    endpoint_diff = saved["probe_scores_R"][:, 0] - saved["probe_scores_R"][:, 4] - (
        saved["probe_scores_F"][:, 0] - saved["probe_scores_F"][:, 4])
    result["critic"]["paired_R_minus_F_endpoint_gap"] = bootstrap_mean(
        endpoint_diff, CONFIG["seeds"]["bootstrap"] + 30)
    result["critic"]["paired_R_minus_F_ordinary_nce_loss"] = bootstrap_mean(
        saved["nce_R_ordinary"][:, 0] - saved["nce_F_ordinary"][:, 0],
        CONFIG["seeds"]["bootstrap"] + 31)
    result["critic"]["paired_R_minus_F_fork_nce_loss"] = bootstrap_mean(
        saved["nce_R_fork"][:, 0] - saved["nce_F_fork"][:, 0],
        CONFIG["seeds"]["bootstrap"] + 32)

    # Fork actor distributions under the canonical and same-episode future goals.
    for pname, pp in psets.items():
        result["actor"][pname] = {}
        for gname, oo in (("canonical", canonical_obs), ("ordinary_goal", ordinary_goal_obs)):
            loc, scale, samples = action_distribution(network, pp, oo, plan["fork_eps"])
            saved[f"actor_{pname}_{gname}_loc"] = loc
            saved[f"actor_{pname}_{gname}_scale"] = scale
            saved[f"actor_{pname}_{gname}_samples"] = samples
            result["actor"][pname][gname] = route_summary(samples)

    # Paired episode-bootstrap actor changes.
    for arm, offset in (("F", 100), ("R", 110)):
        for gname, goff in (("canonical", 0), ("ordinary_goal", 2)):
            sm = saved[f"actor_{arm}_{gname}_samples"]
            si = saved[f"actor_initial_{gname}_samples"]
            dm, rm = direction_masks(sm)
            di, ri = direction_masks(si)
            result["actor"][arm][gname]["paired_down_probability_change"] = bootstrap_mean(
                np.mean(dm, 1) - np.mean(di, 1), CONFIG["seeds"]["bootstrap"] + offset + goff)
            result["actor"][arm][gname]["paired_right_probability_change"] = bootstrap_mean(
                np.mean(rm, 1) - np.mean(ri, 1), CONFIG["seeds"]["bootstrap"] + offset + goff + 1)

    # Local critic gradients at actual stochastic actor samples (first 8 fixed eps).
    lg = CONFIG["eval_local_gradient_samples"]
    for arm, qparams in (("F", qsets["F"]), ("R", qsets["R"])):
        for stage, pp in (("initial_actor", psets["initial"]), ("final_actor", psets[arm])):
            for gname, oo in (("canonical", canonical_obs), ("ordinary_goal", ordinary_goal_obs)):
                loc, scale, samples = action_distribution(network, pp, oo, plan["fork_eps"][:, :lg])
                flat_o = np.repeat(oo, lg, axis=0)
                flat_a = samples.reshape(-1, 2)
                grad = np.asarray(local_q_gradients(network, qparams,
                                                    jnp.asarray(flat_o), jnp.asarray(flat_a))).reshape(len(ce), lg, 2)
                saved[f"local_grad_{arm}_{stage}_{gname}"] = grad
                per_ep = np.mean(grad, axis=1)
                tangent = -(per_ep[:, 0] + per_ep[:, 1])
                result["critic"][arm][f"local_gradient_{stage}_{gname}"] = {
                    "mean_dx": float(np.mean(per_ep[:, 0])),
                    "mean_dy": float(np.mean(per_ep[:, 1])),
                    "mean_norm": float(np.mean(np.linalg.norm(per_ep, axis=1))),
                    "down_minus_right_tangent": bootstrap_mean(
                        tangent, CONFIG["seeds"]["bootstrap"] +
                        (200 if arm == "F" else 220) +
                        (0 if stage == "initial_actor" else 5) +
                        (0 if gname == "canonical" else 2)),
                }

    # Away-from-fork ordinary-goal retention and BC fit.
    ae, at, af = plan["away_episode"], plan["away_time"], plan["away_future"]
    away_obs = np.concatenate([obs[ae, at, :8], obs[ae, af, :8]], axis=1).astype(np.float32)
    away_action = act[ae, at].astype(np.float32)
    saved.update(away_episode=ae, away_time=at, away_future=af)
    away_modes = {}
    away_bc = {}
    for pname, pp in psets.items():
        loc, scale, samples = action_distribution(network, pp, away_obs, plan["away_eps"])
        mode = np.tanh(loc)
        dist = network.policy_network.apply(pp, jnp.asarray(away_obs))
        bc = -np.asarray(network.log_prob(dist, jnp.asarray(away_action)))
        away_modes[pname], away_bc[pname] = mode, bc
        saved[f"away_mode_{pname}"] = mode
        saved[f"away_bc_nll_{pname}"] = bc
        result["actor"][pname]["away_ordinary_goal"] = {
            "bc_nll_mean": float(np.mean(bc)),
            "mode_action_x_mean": float(np.mean(mode[:, 0])),
            "mode_action_y_mean": float(np.mean(mode[:, 1])),
            "scale_mean": float(np.mean(scale)),
        }
    for arm, off in (("F", 300), ("R", 310)):
        shift = np.linalg.norm(away_modes[arm] - away_modes["initial"], axis=1)
        result["actor"][arm]["away_ordinary_goal"]["mode_l2_change"] = bootstrap_mean(
            shift, CONFIG["seeds"]["bootstrap"] + off)
        result["actor"][arm]["away_ordinary_goal"]["bc_nll_change"] = bootstrap_mean(
            away_bc[arm] - away_bc["initial"], CONFIG["seeds"]["bootstrap"] + off + 1)

    # Raw BC and critic gradient components on exactly the same four actor batches.
    for arm in ("F", "R"):
        qparams = qsets[arm]
        for stage, pp in (("initial", psets["initial"]), ("final", psets[arm])):
            rows = []
            for u in range(CONFIG["gradient_batches"]):
                b = batch_from_indices(obs, act, plan["actor_episode"][u],
                                       plan["actor_time"][u], plan["actor_future"][u])
                rows.append(component_gradients(network, pp, qparams, b,
                                                jnp.asarray(plan["actor_keys"][u])))
            result["gradient_components"][f"{arm}_{stage}"] = {
                k: float(np.mean([r[k] for r in rows])) for k in rows[0]
            }

    np.savez_compressed(OUT / "offline_evaluation_arrays.npz", **saved)
    result["counts"] = {
        "heldout_fork_context_rows": int(len(ce)),
        "heldout_fork_context_episodes": int(len(np.unique(ce))),
        "heldout_away_rows": int(len(ae)),
        "heldout_away_episodes": int(len(np.unique(ae)),),
        "ordinary_nce_rows": int(CONFIG["eval_nce_batches"] * 256),
        "fork_nce_rows": int(CONFIG["eval_nce_batches"] * 256),
    }
    write_json(OUT / "results.json", result)
    print("offline evaluation complete")


def fmt_est(x, digits=4):
    return f"{x['mean']:.{digits}f} [{x['ci95'][0]:.{digits}f}, {x['ci95'][1]:.{digits}f}]"


def finalize():
    r = read_json(OUT / "results.json")
    tr = read_json(OUT / "training_results.json")
    cov = read_json(OUT / "coverage.json")
    c = r["critic"]
    a = r["actor"]
    gap_f = c["F"]["endpoint_down_minus_right"]["mean"]
    gap_r = c["R"]["endpoint_down_minus_right"]["mean"]
    dp_f = a["F"]["canonical"]["paired_down_probability_change"]["mean"]
    dp_r = a["R"]["canonical"]["paired_down_probability_change"]["mean"]
    if gap_r > gap_f and dp_r > dp_f:
        decision = "R moved the endpoint preference toward down and its actor moved further toward recorded-down actions than F."
    elif gap_r > gap_f:
        decision = "R moved the endpoint preference toward down, but the actor did not follow it more strongly than F."
    else:
        decision = "R did not improve the down-versus-right endpoint preference relative to F."
    lines = [
        "# Strictly offline PointMaze replay-control report", "",
        "## Result", "",
        decision + " This is an offline sampling/actor-response diagnostic, not evidence of route entry, native success, or recovered intervention Q.", "",
        "Both arms began from the eligible original step-150,000 zbase checkpoint and used only `obs` and `act` from the fixed 6,600-episode dataset. F used ordinary replay; R replaced half of every critic batch with 64 recorded-down and 64 recorded-right fork anchors. Each arm then ran the required 1,000 actor updates on byte-identical ordinary batches and Gaussian innovations.", "",
        "## Coverage and training", "",
        f"The preserved split contains 5,940 continuation-training and 660 held-out episodes. The initial checkpoint had already seen all 6,600, so this is a continuation-held-out comparison, not an independently unseen test. Training support was {cov['train']['down']['anchors']:,} down anchors from {cov['train']['down']['episodes']:,} episodes and {cov['train']['right']['anchors']:,} right anchors from {cov['train']['right']['episodes']:,} episodes. R drew 25,600 rows per direction; it touched {cov['R_training_exposure']['down']['unique_anchors']:,} distinct down and {cov['R_training_exposure']['right']['unique_anchors']:,} distinct right anchors, with maximum multiplicities {cov['R_training_exposure']['down']['max_multiplicity']} and {cov['R_training_exposure']['right']['max_multiplicity']}. Ordinary replay remained 128/256 rows in every R batch.", "",
        f"All 400 critic updates and all 1,000 actor updates completed in both arms. Actor parameter L2 changes were {tr['actor_F']['policy_parameter_delta_l2']:.4f} (F) and {tr['actor_R']['policy_parameter_delta_l2']:.4f} (R). Mean raw actor-gradient versus optimizer-applied update norms were {tr['actor_F']['mean_raw_gradient_l2']:.4f}/{tr['actor_F']['mean_optimizer_update_l2']:.6f} and {tr['actor_R']['mean_raw_gradient_l2']:.4f}/{tr['actor_R']['mean_optimizer_update_l2']:.6f}; raw gradients therefore must not be read as applied step sizes.", "",
        "## Critic", "",
        f"On 8,192 common ordinary held-out rows, final NCE loss was {c['F']['ordinary']['loss']:.6f} for F and {c['R']['ordinary']['loss']:.6f} for R; paired R-minus-F across fixed batches was {fmt_est(c['paired_R_minus_F_ordinary_nce_loss'], 6)}. On 8,192 balanced fork rows, losses were {c['F']['fork']['loss']:.6f} and {c['R']['fork']['loss']:.6f}; paired difference {fmt_est(c['paired_R_minus_F_fork_nce_loss'], 6)}.", "",
        f"Across {r['counts']['heldout_fork_context_episodes']} held-out episodes, canonical-task endpoint down-minus-right logits were {fmt_est(c['initial']['endpoint_down_minus_right'])} initially, {fmt_est(c['F']['endpoint_down_minus_right'])} after F, and {fmt_est(c['R']['endpoint_down_minus_right'])} after R. The paired R-minus-F change was {fmt_est(c['paired_R_minus_F_endpoint_gap'])}. Four-point endpoint neighborhoods gave {fmt_est(c['F']['neighborhood_down_minus_right'])} (F) and {fmt_est(c['R']['neighborhood_down_minus_right'])} (R). These are preference probes, not accuracy labels.", "",
        f"At actual stochastic samples from the final canonical-goal actors, the local maximizing-gradient projection along down-minus-right was {fmt_est(c['F']['local_gradient_final_actor_canonical']['down_minus_right_tangent'])} for F and {fmt_est(c['R']['local_gradient_final_actor_canonical']['down_minus_right_tangent'])} for R. Endpoint rankings and these local derivatives are distinct objects.", "",
        "## Actor response and retention", "",
        f"Under the canonical goal, the original actor sampled the recorded-down stratum with probability {a['initial']['canonical']['down_probability']:.4f} and recorded-right with {a['initial']['canonical']['right_probability']:.4f}. Final F was {a['F']['canonical']['down_probability']:.4f}/{a['F']['canonical']['right_probability']:.4f}; final R was {a['R']['canonical']['down_probability']:.4f}/{a['R']['canonical']['right_probability']:.4f}. Paired down-probability changes were {fmt_est(a['F']['canonical']['paired_down_probability_change'])} (F) and {fmt_est(a['R']['canonical']['paired_down_probability_change'])} (R). These probabilities classify continuous samples only; they do not establish lower-route entry.", "",
        f"With ordinary same-episode future goals at the same fork states, down/right probabilities were {a['F']['ordinary_goal']['down_probability']:.4f}/{a['F']['ordinary_goal']['right_probability']:.4f} (F) and {a['R']['ordinary_goal']['down_probability']:.4f}/{a['R']['ordinary_goal']['right_probability']:.4f} (R). On 1,024 non-fork held-out ordinary-goal rows, mode-action L2 changes were {fmt_est(a['F']['away_ordinary_goal']['mode_l2_change'])} and {fmt_est(a['R']['away_ordinary_goal']['mode_l2_change'])}; BC-NLL changes were {fmt_est(a['F']['away_ordinary_goal']['bc_nll_change'])} and {fmt_est(a['R']['away_ordinary_goal']['bc_nll_change'])}.", "",
        f"On four common fixed actor batches, final BC/critic gradient cosines were {r['gradient_components']['F_final']['component_cosine']:.4f} (F) and {r['gradient_components']['R_final']['component_cosine']:.4f} (R). Weighted BC and critic gradient norms were {r['gradient_components']['F_final']['bc_weighted_gradient_l2']:.4f}/{r['gradient_components']['F_final']['critic_weighted_gradient_l2']:.4f} and {r['gradient_components']['R_final']['bc_weighted_gradient_l2']:.4f}/{r['gradient_components']['R_final']['critic_weighted_gradient_l2']:.4f}.", "",
        "## What this establishes", "",
        "The experiment isolates whether changing exposure to already-recorded observable fork/action strata, while retaining ordinary replay and the unchanged all-pairs NCE objective, changes the critic probes and propagates through the original actor objective. It does not remove behavior-policy confounding, add matched counterfactuals, or test current-policy continuations. A failure would therefore mean this particular reweighting was insufficient, not that contrastive learning or the model family is incapable.", "",
        "The old matched-fork and critic-control reports were hypothesis context only. Their trajectories, labels, critics, and optimizer states were excluded. No model rollout or environment interaction occurred, and no historical artifact was modified.", "",
        "## Recommended next intervention", "",
    ]
    if gap_r > gap_f and dp_r > dp_f:
        lines.append("Because the offline reweighting moved both critic probes and actor samples, the next justified intervention is a separately confirmed, complete-episode evaluation of the fixed R actor under a newly preregistered protocol. The present 660-row continuation-held-out split remains exploratory because initialization saw it; do not infer native performance before that confirmation.")
    elif gap_r > gap_f:
        lines.append("Keep critic R fixed and run a bounded actor-objective diagnostic on saved ordinary batches that maps endpoint preference, local gradients, BC conflict, and optimizer-applied directions. Do not add new data until the failure to transmit the changed critic preference into the actor is localized.")
    else:
        lines.append("Do not increase the replay weight blindly. The next justified intervention is to obtain genuinely matched continuation evidence under a separately authorized complete-episode protocol, because reweighting behavior-policy trajectories cannot manufacture the missing counterfactual/current-policy continuation information.")
    lines += ["", "Numerical arrays, exact hashes, sealed indices, curves, and fixed checkpoints accompany this report.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("report written")


def verify():
    obs, act, initial, train, heldout, plan = load_plan_and_inputs()
    validate_indices(plan, obs, act, train, heldout)
    prov = read_json(OUT / "provenance.json")
    training = read_json(OUT / "training_results.json")
    seal = read_json(OUT / "seal.json")
    assert seal["sealed_before_optimizer_updates"] and seal["optimizer_updates_so_far"] == 0
    for name, digest in seal["sha256"].items():
        assert file_sha(OUT / name) == digest, name
    for path, digest in prov["historical_sha256_before"].items():
        assert file_sha(Path(path)) == digest, path
    for arm in ("F", "R"):
        _, cs = checkpoint.load_checkpoint(OUT / "checkpoints" / f"critic_{arm}_final.pkl")
        _, ps = checkpoint.load_checkpoint(OUT / "checkpoints" / f"actor_{arm}_final.pkl")
        assert tree_sha(cs.policy_params) == tree_sha(initial.policy_params)
        assert tree_sha(cs.policy_optimizer_state) == tree_sha(initial.policy_optimizer_state)
        assert tree_sha(ps.q_params) == tree_sha(cs.q_params)
        assert tree_sha(ps.policy_params) != tree_sha(initial.policy_params)
    assert training["integrity"]["actor_updates_occurred"]
    source = (OUT / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden_import_fragments = ("env", "native", "rollout", "transition_model", "oracle")
    assert not [x for x in imports if any(f in x.lower() for f in forbidden_import_fragments)]
    status = subprocess.check_output(["git", "diff", "--name-only"], text=True).splitlines()
    verification = {
        "status": "pass",
        "no_environment_or_model_imports": True,
        "runtime_training_arrays": ["obs", "act"],
        "no_excluded_paths_in_actual_training_inputs": True,
        "input_hashes_match": True,
        "split_preserved_and_disjoint": True,
        "same_eligible_starting_parameters_and_optimizer_states": True,
        "critic_only_difference_is_sampling_plan": True,
        "R_all_batches_have_128_ordinary_64_down_64_right": True,
        "all_future_indices_strictly_later_and_same_episode": True,
        "actor_batches_goals_and_random_streams_identical": True,
        "actor_updates_completed": {"F": 1000, "R": 1000},
        "historical_artifact_hashes_unchanged": True,
        "tracked_diff_paths": status,
        "note": "No commit or push performed; new output directory is untracked by design.",
    }
    write_json(OUT / "verification.json", verification)
    deliverables = ["PROTOCOL.md", "provenance.json", "config.json", "coverage.json",
                    "equivalence_review.json", "partition.npz", "sampling_plan.npz",
                    "run.py", "training_curves.npz", "offline_evaluation_arrays.npz",
                    "training_results.json", "results.json", "REPORT.md", "verification.json",
                    "checkpoints/critic_F_final.pkl", "checkpoints/critic_R_final.pkl",
                    "checkpoints/actor_F_final.pkl", "checkpoints/actor_R_final.pkl"]
    write_json(OUT / "completion.json", {
        "status": "complete", "deliverable_sha256": {
            p: file_sha(OUT / p) for p in deliverables},
        "critic_updates": {"F": 400, "R": 400},
        "actor_updates": {"F": 1000, "R": 1000},
        "native_interactions": 0, "model_rollouts": 0,
    })
    print("verification passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "train", "evaluate", "finalize", "verify", "all"],
                        default="all")
    args = parser.parse_args()
    phases = ["prepare", "train", "evaluate", "finalize", "verify"] if args.phase == "all" else [args.phase]
    for phase in phases:
        globals()[phase]()


if __name__ == "__main__":
    main()
