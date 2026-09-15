"""Faithful-actor consistency audit (Part A, no training) and uniform-anchor
critic purity check (Part B). See PROTOCOL.md."""
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crl import checkpoint  # noqa: E402
from ett import absorbing_ett as ab  # noqa: E402
from ett import pointmaze_absorbing_integration as g1mod  # noqa: E402
from ett import pointmaze_offline_causal_integration as oci  # noqa: E402
from ett import pointmaze_region_pilot as region  # noqa: E402
from ett.diagonal_transition import POINTMAZE_WALLS  # noqa: E402
from ett.rollout_return import GOAL  # noqa: E402

OUT = Path(__file__).resolve().parent
G1 = ROOT / "outputs/pointmaze_absorbing_integration_20260915_v1"
G2 = ROOT / "outputs/pointmaze_absorbing_actor_sweep_20260915_v1"
CHECKPOINTS = [("bc0p5", 0.5), ("bc0p2", 0.2), ("bc0p1", 0.1), ("bc0p05", 0.05)]
ARMS = ["O", "P"]
FORK_PATHS, FORK_PATHS_DIAG, HORIZON = 32, 16, 49
ROLLOUT_CAP = 30_000_000
BOOT_SEED = 0
WALLS = np.asarray(POINTMAZE_WALLS)
GOAL2 = np.asarray(GOAL[:2], np.float64)
CELL_NAMES = {(1, 2): "(1,2) lower entry", (2, 3): "(2,3) right", (1, 3): "(1,3) stay", (0, 3): "(0,3) left"}


def write(path, value):
    oci.write_json(path, value)


def bs(values, seed):
    return oci.bootstrap_mean(np.asarray(values, np.float64), seed)


def physical_landing(xy, a):
    p = np.array(xy, np.float64).copy()
    a = np.clip(np.asarray(a, np.float64), -1, 1)
    for _ in range(10):
        for axis in range(2):
            q = p.copy(); q[:, axis] += .1 * a[:, axis]
            out = (q[:, 0] < 0) | (q[:, 1] < 0) | (q[:, 0] > 9) | (q[:, 1] > 5)
            c = np.clip(np.floor(q).astype(int), [0, 0], [8, 4])
            p = np.where((out | (WALLS[c[:, 0], c[:, 1]] == 1))[:, None], p, q)
    return np.clip(np.floor(p).astype(int), [0, 0], [8, 4])


def cell_name(c):
    c = (int(c[0]), int(c[1]))
    if c in CELL_NAMES:
        return CELL_NAMES[c]
    return "wall cell" if WALLS[c] == 1 else f"other free {c}"


def distribution(cells):
    n = len(cells)
    counts = Counter(cell_name(c) for c in cells)
    return {k: v / n for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}


def make_actor(network, policy_params):
    @jax.jit
    def sample(s, g, key):
        d = network.policy_network.apply(policy_params, jnp.concatenate([s, g], -1))
        return jnp.tanh(d.loc + d.scale * jax.random.normal(key, d.loc.shape))
    return sample


def in_cell(xy, c):
    cc = ab.landing_cell(xy)
    return (cc[..., 0] == c[0]) & (cc[..., 1] == c[1])


# --------------------------------------------------------------------------- #
# Part A                                                                        #
# --------------------------------------------------------------------------- #
def first_step_audit(name, pp, ctx, backend_a3, key):
    network, plan, fs, canonical = ctx["network"], ctx["plan"], ctx["fork_state"], ctx["canonical"]
    loc, scale, sample = oci.action_distribution(network, pp, canonical, plan["fork_eps"])
    n, k, _ = sample.shape
    S8 = np.repeat(fs, k, 0); S = S8[:, :2]; A = sample.reshape(-1, 2).astype(np.float32)
    naive = ab.landing_cell(S + A)
    phys = physical_landing(S, A)
    y, diag = backend_a3.sample(np.zeros(48, np.float32), S8, A, A, key, 1)
    y = np.asarray(y)[:, 0]; atom = np.asarray(diag["stationary_atom"])[:, 0]
    model = ab.landing_cell(y[:, :2])
    # Manski onset flag for this first transition: x' drawn from the nominal at s
    xp = np.asarray(ctx["engine"].nominal.sample(jnp.asarray(S8), jax.random.fold_in(key, 1), 1,
                                                 goal=jnp.broadcast_to(jnp.asarray(GOAL), S8.shape)))
    agree = np.all(ab.landing_cell(S + A) == ab.landing_cell(S + xp), axis=1)
    ins = ctx["support"][model[:, 0], model[:, 1]]
    onset = (~agree) & ins & (~atom)
    # eligible Kernel at zero offsets (action-blind): anchor at the nominal draw
    yk, _ = ctx["engine"].sample(jnp.zeros(48), jnp.asarray(S8), jnp.asarray(A), jnp.asarray(xp), jax.random.fold_in(key, 2), 1)
    kernel = ab.landing_cell(np.asarray(yk)[:, 0, :2])
    p12 = (phys[:, 0] == 1) & (phys[:, 1] == 2)
    m12 = (model[:, 0] == 1) & (model[:, 1] == 2)
    naive_wall = WALLS[naive[:, 0], naive[:, 1]] == 1
    diag_wall = naive_wall & p12
    joint = Counter((cell_name(p), cell_name(m)) for p, m in zip(phys, model))
    per_root_disagree = np.mean(np.any(model != phys, axis=1).reshape(n, k), axis=1)
    per_root_p12 = p12.reshape(n, k).mean(1); per_root_m12 = m12.reshape(n, k).mean(1)
    cond = m12.reshape(n, k).sum(1) / np.maximum(p12.reshape(n, k).sum(1), 1)
    result = {
        "samples": int(n * k),
        "naive": distribution(naive), "physical": distribution(phys),
        "a1_a3_first_transition": distribution(model), "kernel_zero_offsets": distribution(kernel),
        "model_atom_fraction": float(atom.mean()),
        "model_onset_fraction_first_step": float(onset.mean()),
        "physical_12_fraction": float(p12.mean()), "model_12_fraction": float(m12.mean()),
        "model_cell_given_physical_12": distribution(model[p12]),
        "model_cell_given_physical_23": distribution(model[(phys[:, 0] == 2) & (phys[:, 1] == 3)]),
        "diagonal_wall_actions": {
            "fraction_of_samples": float(diag_wall.mean()),
            "definition": "naive cell is a wall cell and physical landing is (1,2)",
            "model_cell": distribution(model[diag_wall]) if diag_wall.any() else {},
            "mean_action": np.mean(A[diag_wall], 0).tolist() if diag_wall.any() else None,
            "mean_model_displacement": float(np.linalg.norm(y[diag_wall, :2] - S[diag_wall], axis=1).mean()) if diag_wall.any() else None,
            "mean_physical_displacement": float(np.linalg.norm(
                physical_position(S[diag_wall], A[diag_wall]) - S[diag_wall], axis=1).mean()) if diag_wall.any() else None},
        "joint_physical_x_model_top": {f"{p} -> {m}": v / (n * k) for (p, m), v in joint.most_common(12)},
        "per_root": {
            "mean_disagreement_fraction": float(per_root_disagree.mean()),
            "roots_with_disagreement_over_half": int(np.sum(per_root_disagree > .5)),
            "physical_12": bs(per_root_p12, BOOT_SEED + 1), "model_12": bs(per_root_m12, BOOT_SEED + 2),
            "model_minus_physical_12": bs(per_root_m12 - per_root_p12, BOOT_SEED + 3),
            "mean_P_model12_given_physical12": float(np.mean(cond[p12.reshape(n, k).sum(1) > 0]))},
    }
    return result, dict(phys12=per_root_p12, model12=per_root_m12, disagree=per_root_disagree)


def physical_position(xy, a):
    p = np.array(xy, np.float64).copy()
    a = np.clip(np.asarray(a, np.float64), -1, 1)
    for _ in range(10):
        for axis in range(2):
            q = p.copy(); q[:, axis] += .1 * a[:, axis]
            out = (q[:, 0] < 0) | (q[:, 1] < 0) | (q[:, 0] > 9) | (q[:, 1] > 5)
            c = np.clip(np.floor(q).astype(int), [0, 0], [8, 4])
            p = np.where((out | (WALLS[c[:, 0], c[:, 1]] == 1))[:, None], p, q)
    return p


def continuation_audit(name, pp, ctx, key3, key1):
    network, fs, fl = ctx["network"], ctx["fork_state"], ctx["fork_len"]
    actor_fn = make_actor(network, pp)
    fake = SimpleNamespace(base=ctx["engine"].base, nominal=ctx["engine"].nominal, actor=actor_fn, actor_jit=None)
    a3 = ab.AbsorbingRollout(fake, ctx["support"], "manski_absorbing")
    a1 = ab.AbsorbingRollout(fake, ctx["support"], "diagonal_motion")
    n = len(fs)
    s = np.repeat(fs, FORK_PATHS, 0)
    out = a3.rollout(np.zeros(48, np.float32), s, key3, HORIZON)
    L = np.minimum(np.repeat(fl, FORK_PATHS), HORIZON)
    xy = out["states"][:, :, :2]; frozen = out["frozen"]
    T = np.arange(HORIZON + 1)[None]
    valid = T <= L[:, None]
    below = valid & (xy[:, :, 1] < 2.)
    lower = below.any(1)
    t_lower = np.where(lower, np.argmax(below, axis=1), -1)
    first_cell = ab.landing_cell(xy[:, 1])
    first12 = (first_cell[:, 0] == 1) & (first_cell[:, 1] == 2)
    ever12 = (valid & in_cell(xy, (1, 2))).any(1)
    absorbed = frozen[np.arange(len(s)), L]
    reward = out["reward"] * (np.arange(HORIZON)[None] < L[:, None])
    reach = (reward > 0).any(1)
    # outcome classes for paths whose first model cell is (1,2)
    classes = []
    for i in np.flatnonzero(first12):
        h = int(L[i]); cells = ab.landing_cell(xy[i, 1:h + 1])
        below_t = np.flatnonzero(xy[i, 1:h + 1, 1] < 2.)
        back = np.flatnonzero(((cells[:, 0] == 1) & (cells[:, 1] == 3)) | ((cells[:, 0] == 2) & (cells[:, 1] == 3)))
        tb = below_t[0] if len(below_t) else None
        tr = back[0] if len(back) else None
        if tb is not None and (tr is None or tb < tr):
            classes.append("continued (y<2 before revisiting fork/right)")
        elif tr is not None:
            classes.append("returned to fork/right" + (" then absorbed" if absorbed[i] else ""))
        elif absorbed[i]:
            classes.append("absorbed without leaving")
        else:
            stayed = np.all((cells[:, 0] == 1) & (cells[:, 1] == 2))
            classes.append("stuck in (1,2)" if stayed else "wandered elsewhere")
    class_dist = {k: v / max(len(classes), 1) for k, v in Counter(classes).most_common()}
    # actor at the first (1,2) state of each such path: physical landing of a fresh sample
    idx = np.flatnonzero(ever12)
    first12_state = np.stack([out["states"][i, np.argmax(valid[i] & in_cell(xy[i], (1, 2)))] for i in idx]) if len(idx) else np.zeros((0, 8), np.float32)
    at12 = {}
    if len(idx):
        g = jnp.broadcast_to(jnp.asarray(GOAL), first12_state.shape)
        a12 = np.asarray(actor_fn(jnp.asarray(first12_state), g, jax.random.fold_in(key3, 99)))
        at12 = {"states": int(len(idx)), "physical_landing_of_actor_sample": distribution(physical_landing(first12_state[:, :2], a12)),
                "mean_action": a12.mean(0).tolist(), "mean_state_xy": first12_state[:, :2].mean(0).tolist()}
    # A1 rollouts
    s1 = np.repeat(fs, FORK_PATHS_DIAG, 0)
    out1 = a1.rollout(np.zeros(48, np.float32), s1, key1, HORIZON)
    L1 = np.minimum(np.repeat(fl, FORK_PATHS_DIAG), HORIZON)
    r1 = out1["reward"] * (np.arange(HORIZON)[None] < L1[:, None])
    xy1 = out1["states"][:, :, :2]
    lower1 = ((np.arange(HORIZON + 1)[None] <= L1[:, None]) & (xy1[:, :, 1] < 2.)).any(1)
    first1 = ab.landing_cell(xy1[:, 1]); first12_1 = (first1[:, 0] == 1) & (first1[:, 1] == 2)
    per_root = dict(a3_first12=first12.reshape(n, FORK_PATHS).mean(1), a3_ever12=ever12.reshape(n, FORK_PATHS).mean(1),
                    a3_lower=lower.reshape(n, FORK_PATHS).mean(1), a3_absorbed=absorbed.reshape(n, FORK_PATHS).mean(1),
                    a3_reach=reach.reshape(n, FORK_PATHS).mean(1),
                    a1_first12=first12_1.reshape(n, FORK_PATHS_DIAG).mean(1), a1_lower=lower1.reshape(n, FORK_PATHS_DIAG).mean(1),
                    a1_reach=(r1 > 0).any(1).reshape(n, FORK_PATHS_DIAG).mean(1))
    result = {
        "a3_first_step_cell": distribution(first_cell),
        "a3_first_step_12": bs(per_root["a3_first12"], BOOT_SEED + 10),
        "a3_ever_in_12": bs(per_root["a3_ever12"], BOOT_SEED + 11),
        "a3_lower_route": bs(per_root["a3_lower"], BOOT_SEED + 12),
        "a3_absorbed": bs(per_root["a3_absorbed"], BOOT_SEED + 13),
        "a3_reach": bs(per_root["a3_reach"], BOOT_SEED + 14),
        "a3_lower_given_first12": float(lower[first12].mean()) if first12.any() else None,
        "a3_absorbed_given_first12": float(absorbed[first12].mean()) if first12.any() else None,
        "a3_lower_given_not_first12": float(lower[~first12].mean()),
        "mean_time_to_lower_given_lower": float(t_lower[lower].mean()) if lower.any() else None,
        "first12_outcome_classes": class_dist, "first12_paths": int(first12.sum()),
        "actor_at_first_12_state": at12,
        "a1_first_step_12": bs(per_root["a1_first12"], BOOT_SEED + 15),
        "a1_lower_route": bs(per_root["a1_lower"], BOOT_SEED + 16),
        "a1_reach": bs(per_root["a1_reach"], BOOT_SEED + 17),
    }
    cost = len(s) * HORIZON + len(s1) * HORIZON
    return result, per_root, cost


def recorded_actions_at_12(obs, act, train):
    xy = obs[train, :50, :2].reshape(-1, 2); a = act[train, :50].reshape(-1, 2)
    m = in_cell(xy, (1, 2))
    return {"rows": int(m.sum()), "physical_landing": distribution(physical_landing(xy[m], a[m])),
            "mean_action": a[m].mean(0).tolist()}


# --------------------------------------------------------------------------- #
# Part B                                                                        #
# --------------------------------------------------------------------------- #
def purity_check(ctx, obs, act, train, initial):
    rng = np.random.default_rng(0)
    rows = rng.choice(len(train) * 50, 4096, replace=False)
    episode = train[rows // 50].astype(np.int32); time_index = (rows % 50).astype(np.int32)
    engine = ctx["engine"]
    backend = ab.AbsorbingRollout(engine, ctx["support"], "manski_absorbing")
    ledger = oci.Ledger(OUT)
    cache = g1mod.generate_cache(backend, np.zeros(48, np.float32), obs, act, episode, time_index,
                                 oci.CONFIG["seed"] + 300, ledger, "purity/uniform_pool_A3_cache")
    oci.save_cache(OUT / "uniform_pool_trajectory_cache.npz", cache)
    fm = oci.fork_mask(obs[episode, time_index]); down, right = oci.direction_masks(act[episode, time_index])
    pool_info = {"anchors": 4096, "fork_anchors": int(fm.sum()), "fork_down_anchors": int((fm & down).sum()),
                 "fork_right_anchors": int((fm & right).sum()), "sampling": "uniform rows of the training partition, seed 0, without replacement"}
    ids = rng.integers(0, 4096, (oci.CONFIG["critic_updates"], 256)).astype(np.int32)
    offsets = np.stack([oci.draw_offsets(rng, 50 - time_index[ids[u]]) for u in range(oci.CONFIG["critic_updates"])])
    plan_like = {"train_pool_episode": episode, "train_pool_time": time_index}
    network, q_optimizer, _ = oci.make_network_and_optimizers()
    critic_step = oci.build_critic_step(network, q_optimizer)
    score_fn = jax.jit(lambda q, o, a: oci.paired_scores(network, q, o, a))
    canonical, probes = ctx["canonical"], ctx["plan"]["action_probes"]

    def endpoint(qparams, seed):
        sc = np.stack([np.asarray(score_fn(qparams, jnp.asarray(canonical), jnp.asarray(np.broadcast_to(p, (len(canonical), 2))))) for p in probes], 1)
        return {"endpoint_down_minus_right": bs(sc[:, 0] - sc[:, 4], seed),
                "neighborhood_down_minus_right": bs(sc[:, :4].mean(1) - sc[:, 4:].mean(1), seed + 1)}, sc[:, 0] - sc[:, 4]
    results = {"pool": pool_info, "initial": endpoint(initial.q_params, 500)[0]}
    gaps = {}
    for arm in ARMS:
        state = initial; losses = []
        for u in range(oci.CONFIG["critic_updates"]):
            batch, _ = oci.rows_from_pool(obs, act, plan_like, cache, "train_pool", ids[u], offsets[u], arm)
            q, qo, tq, loss, aux, gn, un = critic_step(state.q_params, state.q_optimizer_state, state.target_q_params, batch)
            assert np.isfinite(float(loss))
            state = state._replace(q_params=q, q_optimizer_state=qo, target_q_params=tq); losses.append(float(loss))
        assert oci.tree_sha(state.policy_params) == oci.tree_sha(initial.policy_params)
        checkpoint.save_named(OUT / "checkpoints", f"critic_{arm}_uniform_anchors", 150400, state)
        results[arm], gaps[arm] = endpoint(state.q_params, 510 if arm == "O" else 520)
        results[arm]["final_loss"] = float(np.mean(losses[-50:]))
        results[arm]["critic_parameter_delta_l2"] = oci.tree_delta_norm(state.q_params, initial.q_params)
    results["paired_P_minus_O_endpoint_gap"] = bs(gaps["P"] - gaps["O"], 530)
    results["model_outputs"] = ledger.data["charged"]
    g1 = oci.read_json(G1 / "results.json")["critic"]
    results["G1_stratified_reference"] = {k: g1[k]["endpoint_down_minus_right"] for k in ("initial", "O", "P")}
    results["G1_stratified_reference"]["paired_P_minus_O"] = g1["paired_P_minus_O_endpoint_gap"]
    return results


def main():
    t0 = time.time()
    obs, act, initial, train, heldout, plan = oci.load_prepared(G1)
    freeze = oci.read_json(G1 / "freeze_tables.json")
    support = np.zeros(ab.GRID, bool)
    for i, j in freeze["support_cells"]:
        support[i, j] = True
    network, _, _ = oci.make_network_and_optimizers()
    engine = region.Kernel()
    ce, ct = plan["fork_context_episode"], plan["fork_context_time"]
    fs = obs[ce, ct, :8].astype(np.float32)
    ctx = dict(network=network, plan=plan, engine=engine, support=support, fork_state=fs, fork_len=(50 - ct).astype(np.int32),
               canonical=np.concatenate([fs, np.broadcast_to(GOAL, fs.shape)], 1).astype(np.float32))
    backend_a3 = ab.AbsorbingRollout(engine, support, "manski_absorbing")
    config = dict(checkpoints=CHECKPOINTS, arms=ARMS, fork_paths=FORK_PATHS, fork_paths_diagonal=FORK_PATHS_DIAG, horizon=HORIZON,
                  rollout_cap=ROLLOUT_CAP, bootstrap_seed=BOOT_SEED,
                  sha256={p: oci.file_sha(ROOT / p) for p in ["ett/absorbing_ett.py", "ett/pointmaze_absorbing_integration.py",
                                                              "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_O_final.pkl",
                                                              "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_P_final.pkl",
                                                              "outputs/pointmaze_absorbing_integration_20260915_v1/sampler_plan.npz"] +
                          [f"outputs/pointmaze_absorbing_actor_sweep_20260915_v1/checkpoints/actor_{a}_{t}.pkl" for t, _ in CHECKPOINTS for a in ARMS]},
                  protocol_sha256=oci.file_sha(OUT / "PROTOCOL.md"),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  fields_read=["obs", "act"], environment_calls=0, native_steps=0, actor_training=0)
    write(OUT / "config.json", config)
    results = {"config": config, "part_a": {}, "recorded_actions_at_1_2_train": recorded_actions_at_12(obs, act, train)}
    per_root = {}
    charged = 0
    key_first = jax.random.PRNGKey(4100)
    for tag_, bc in CHECKPOINTS:
        for arm in ARMS:
            name = f"{arm}_{tag_}"
            pp = checkpoint.load_checkpoint(G2 / "checkpoints" / f"actor_{arm}_{tag_}.pkl")[1].policy_params
            fa, pr_a = first_step_audit(name, pp, ctx, backend_a3, key_first)
            ca, pr_c, cost = continuation_audit(name, pp, ctx, jax.random.PRNGKey(2100), jax.random.PRNGKey(2200))
            charged += cost; assert charged <= ROLLOUT_CAP
            results["part_a"][name] = {"bc_coef": bc, "arm": arm, "first_step": fa, "continuation": ca}
            for k, v in {**pr_a, **pr_c}.items(): per_root[f"{name}/{k}"] = v
            write(OUT / f"audit_{name}.json", results["part_a"][name])
            print(f"{name}: phys(1,2) {fa['physical_12_fraction']:.3f} model(1,2) {fa['model_12_fraction']:.3f} "
                  f"P(model12|phys12) {fa['per_root']['mean_P_model12_given_physical12']:.3f} | A3 first12 {ca['a3_first_step_12']['mean']:.3f} "
                  f"ever12 {ca['a3_ever_in_12']['mean']:.3f} lower {ca['a3_lower_route']['mean']:.3f} lower|first12 {ca['a3_lower_given_first12']} "
                  f"| A1 first12 {ca['a1_first_step_12']['mean']:.3f} lower {ca['a1_lower_route']['mean']:.3f} reach {ca['a1_reach']['mean']:.3f} | {time.time() - t0:.0f}s", flush=True)
    results["rollout_transitions"] = charged
    print("part B: uniform-anchor critic purity check", flush=True)
    results["part_b"] = purity_check(ctx, obs, act, train, initial)
    results["elapsed_seconds"] = time.time() - t0
    np.savez_compressed(OUT / "per_root.npz", fork_context_episode=ce, fork_context_time=ct, **per_root)
    write(OUT / "results.json", results)
    pb = results["part_b"]
    print("purity: initial", pb["initial"]["endpoint_down_minus_right"]["mean"], "O", pb["O"]["endpoint_down_minus_right"]["mean"],
          "P", pb["P"]["endpoint_down_minus_right"]["mean"], "P-O", pb["paired_P_minus_O_endpoint_gap"]["mean"])


if __name__ == "__main__":
    main()
