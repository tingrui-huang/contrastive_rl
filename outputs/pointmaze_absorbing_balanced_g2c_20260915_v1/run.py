"""G2c: balanced (cell, landing-cell) sampling of the actor batches; paper objective.

Strictly offline. See PROTOCOL.md.  The actor objective, critics, roots, goal,
actor keys and budgets are the sealed G1 ones; only the rows in the 1,000
actor batches change (key-balanced instead of uniform).
"""
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crl import checkpoint  # noqa: E402
from ett import absorbing_ett as ab  # noqa: E402
from ett import pointmaze_offline_causal_integration as oci  # noqa: E402
from ett import pointmaze_region_pilot as region  # noqa: E402
from ett.diagonal_transition import POINTMAZE_WALLS  # noqa: E402
from ett.rollout_return import GOAL  # noqa: E402

OUT = Path(__file__).resolve().parent
G1 = ROOT / "outputs/pointmaze_absorbing_integration_20260915_v1"
G2 = ROOT / "outputs/pointmaze_absorbing_actor_sweep_20260915_v1"
ARMS = ["O", "P"]
ROW_SEEDS = [0, 1, 2]
KEY_MIN_ROWS = 10
FORK_PATHS, FORK_PATHS_DIAG, AWAY_PATHS, HORIZON = 32, 16, 8, 49
ROLLOUT_CAP = 30_000_000
BOOT_SEED = 0
GOAL2 = np.asarray(GOAL[:2], np.float64)
WALLS = np.asarray(POINTMAZE_WALLS)
CELLS = {"(1,2) lower entry": (1, 2), "(2,3) right": (2, 3), "(1,3) stay": (1, 3), "(0,3) left": (0, 3)}


def write(path, value):
    oci.write_json(path, value)


def summary_stats(values, seed):
    return oci.bootstrap_mean(np.asarray(values, np.float64), seed)


# --------------------------------------------------------------------------- #
# balanced key sampling                                                         #
# --------------------------------------------------------------------------- #
def build_keys(obs, act, train):
    e_idx, t_idx = np.meshgrid(train, np.arange(50), indexing="ij")
    e_idx, t_idx = e_idx.ravel(), t_idx.ravel()
    xy = obs[e_idx, t_idx, :2]
    a = act[e_idx, t_idx]
    c, l = ab.landing_cell(xy), ab.landing_cell(xy + a)
    key = (c[:, 0] * 1000 + c[:, 1] * 100 + l[:, 0] * 10 + l[:, 1]).astype(np.int64)
    uniq, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep = counts >= KEY_MIN_ROWS
    rows_by_key = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    info = {"distinct_keys": int(len(uniq)), "kept_keys": int(keep.sum()),
            "excluded_keys": int((~keep).sum()), "excluded_rows": int(counts[~keep].sum()),
            "total_rows": int(len(key)),
            "fork_keys": {f"({(k // 10) % 10},{k % 10})": int(n) for k, n in zip(uniq, counts)
                          if k // 1000 == 1 and (k // 100) % 10 == 3}}
    kept = [(int(uniq[k]), rows_by_key[k]) for k in range(len(uniq)) if keep[k]]
    return kept, e_idx, t_idx, info


def draw_balanced_batches(rng, kept, e_idx, t_idx, batches):
    e = np.empty((batches, 256), np.int32); t = np.empty_like(e); f = np.empty_like(e)
    keys = np.empty((batches, 256), np.int64)
    for u in range(batches):
        ksel = rng.integers(0, len(kept), 256)
        rows = np.array([kept[k][1][rng.integers(0, len(kept[k][1]))] for k in ksel])
        keys[u] = [kept[k][0] for k in ksel]
        e[u], t[u] = e_idx[rows], t_idx[rows]
        f[u] = t[u] + oci.draw_offsets(rng, 50 - t[u])
    return e, t, f, keys


def balanced_batch(obs, act, rows, u):
    e, t, f = rows["episode"][u], rows["time"][u], rows["future"][u]
    return oci.transition_batch(obs[e, t, :8], act[e, t], obs[e, f, :8], obs[e, t + 1, :8])


def train_actor(network, optimizer, initial, critic_state, rows, plan, obs, act):
    step = oci.build_actor_step(network, optimizer)           # the sealed .5/.5 objective
    state = initial._replace(q_params=critic_state.q_params,
                             target_q_params=critic_state.target_q_params,
                             q_optimizer_state=critic_state.q_optimizer_state)
    curve = []
    for u in range(oci.CONFIG["actor_updates"]):
        batch = balanced_batch(obs, act, rows, u)
        pp, po, loss, bc, critic, auxiliary, gn, un = step(
            state.policy_params, state.policy_optimizer_state, state.q_params,
            batch, jnp.asarray(plan["actor_keys"][u]))
        values = np.asarray([loss, bc, critic, *auxiliary, gn, un], np.float64)
        if not np.isfinite(values).all() or not oci.finite_tree((pp, po)):
            raise FloatingPointError((u, values))
        state = state._replace(policy_params=pp, policy_optimizer_state=po)
        curve.append(values)
    assert oci.tree_sha(state.q_params) == oci.tree_sha(critic_state.q_params)
    return state, np.asarray(curve)


# --------------------------------------------------------------------------- #
# evaluation helpers (G2/G2b) plus the physical landing                          #
# --------------------------------------------------------------------------- #
def physical_landing(xy, a):
    """Ten substeps, X then Y, blocked coordinate updates rejected; no noise."""
    p = np.array(xy, np.float64).copy()
    a = np.clip(np.asarray(a, np.float64), -1, 1)
    for _ in range(10):
        for axis in range(2):
            q = p.copy(); q[:, axis] += .1 * a[:, axis]
            out = (q[:, 0] < 0) | (q[:, 1] < 0) | (q[:, 0] > 9) | (q[:, 1] > 5)
            c = np.clip(np.floor(q).astype(int), [0, 0], [8, 4])
            blocked = out | (WALLS[c[:, 0], c[:, 1]] == 1)
            p = np.where(blocked[:, None], p, q)
    return np.clip(np.floor(p).astype(int), [0, 0], [8, 4])


def make_actor(network, policy_params):
    @jax.jit
    def sample(s, g, key):
        d = network.policy_network.apply(policy_params, jnp.concatenate([s, g], -1))
        return jnp.tanh(d.loc + d.scale * jax.random.normal(key, d.loc.shape))
    return sample


def rollout_metrics(backend, states, lengths, paths, key):
    n = len(states)
    s = np.repeat(states, paths, 0)
    out = backend.rollout(np.zeros(48, np.float32), s, key, HORIZON)
    L = np.minimum(np.repeat(lengths, paths), HORIZON)
    mask = np.arange(HORIZON)[None] < L[:, None]
    r = out["reward"] * mask
    disc = .95 ** np.arange(HORIZON)
    xy = out["states"][:, :, :2]
    smask = np.arange(HORIZON + 1)[None] <= L[:, None]
    dist = np.where(smask, np.linalg.norm(xy - GOAL2, axis=-1), np.inf)
    lower = np.where(smask, xy[:, :, 1] < 2., False).any(1)
    step_disp = np.linalg.norm(np.diff(xy, axis=1), axis=-1) * mask
    metrics = dict(
        discounted_return=(r * disc).sum(1), reach=(r > 0).any(1).astype(float),
        strict_success=(dist.min(1) < .5).astype(float), lower_route=lower.astype(float),
        absorbed=out["frozen"][np.arange(n * paths), L].astype(float),
        entered_support=(out["hazard_alive"] & mask).any(1).astype(float),
        manski_onset=(out["onset_manski"] & mask).any(1).astype(float),
        mean_step_displacement=step_disp.sum(1) / np.maximum(mask.sum(1), 1))
    return {k: v.reshape(n, paths).mean(1) for k, v in metrics.items()}, int(n * paths * HORIZON)


def observed_pairs(obs, act, train):
    xy = obs[train, :50, :2].reshape(-1, 2)
    a = act[train, :50].reshape(-1, 2)
    c = ab.landing_cell(xy); l = ab.landing_cell(xy + a)
    table = np.zeros(ab.GRID + ab.GRID, bool)
    table[c[:, 0], c[:, 1], l[:, 0], l[:, 1]] = True
    return table


def ood_fraction(table, states, actions):
    c = ab.landing_cell(states[:, :2]); l = ab.landing_cell(states[:, :2] + actions)
    return float(np.mean(~table[c[:, 0], c[:, 1], l[:, 0], l[:, 1]]))


def bin_agreement(states, actions, recorded, physical=False):
    f = (lambda s, a: physical_landing(s, a)) if physical else (lambda s, a: ab.landing_cell(s + a))
    return float(np.mean(np.all(f(states[:, :2], actions) == f(states[:, :2], recorded), axis=1)))


def landing_distribution(states, sample):
    n, k, _ = sample.shape
    S = np.repeat(states[:, :2], k, 0); A = sample.reshape(-1, 2)
    naive, phys = ab.landing_cell(S + A), physical_landing(S, A)
    out = {}
    for tag_, land in (("physical", phys), ("naive", naive)):
        d = {name: float(np.mean((land[:, 0] == c[0]) & (land[:, 1] == c[1]))) for name, c in CELLS.items()}
        d["wall cell"] = float(np.mean(WALLS[land[:, 0], land[:, 1]] == 1))
        d["other free cell"] = float(1. - sum(d.values()))
        out[tag_] = d
    down, right = oci.direction_masks(A)
    out["mask_down"], out["mask_right"] = float(down.mean()), float(right.mean())
    per_root = {"phys_12": ((phys[:, 0] == 1) & (phys[:, 1] == 2)).reshape(n, k).mean(1),
                "phys_23": ((phys[:, 0] == 2) & (phys[:, 1] == 3)).reshape(n, k).mean(1),
                "naive_12": ((naive[:, 0] == 1) & (naive[:, 1] == 2)).reshape(n, k).mean(1)}
    return out, per_root


def evaluate_actor(name, pp, arm, ctx):
    network, critics, engine, support, plan, pairs = (ctx[k] for k in ("network", "critics", "engine", "support", "plan", "pairs"))
    fork_state, fork_len, canonical = ctx["fork_state"], ctx["fork_len"], ctx["canonical"]
    away_state, away_len, away_obs, away_action = ctx["away_state"], ctx["away_len"], ctx["away_obs"], ctx["away_action"]
    score_fn = ctx["score_fn"]
    rec = {"arm": arm}
    loc, scale, sample = oci.action_distribution(network, pp, canonical, plan["fork_eps"])
    land, per_root = landing_distribution(fork_state, sample)
    rec["fork_landing"] = land
    rec["fork_action"] = oci.action_summary(sample)
    rec["fork_action"]["saturation_fraction"] = float(np.mean(np.any(np.abs(sample) > .99, -1)))
    rec["fork_action"]["mode_ood_fraction"] = ood_fraction(pairs, fork_state, np.tanh(loc))
    flat = sample.reshape(-1, 2)
    q_samples = np.asarray(score_fn(critics[arm].q_params, jnp.asarray(np.repeat(canonical, sample.shape[1], 0)),
                                    jnp.asarray(flat))).reshape(len(canonical), -1).mean(1)
    probe = plan["action_probes"]
    qd = np.asarray(score_fn(critics[arm].q_params, jnp.asarray(canonical), jnp.asarray(np.broadcast_to(probe[0], (len(canonical), 2)))))
    qr = np.asarray(score_fn(critics[arm].q_params, jnp.asarray(canonical), jnp.asarray(np.broadcast_to(probe[4], (len(canonical), 2)))))
    rec["critic_at_actor"] = {"mean_q_actor_samples": float(q_samples.mean()), "mean_q_down_probe": float(qd.mean()),
                              "mean_q_right_probe": float(qr.mean()),
                              "actor_minus_right_probe": summary_stats(q_samples - qr, BOOT_SEED + 3),
                              "actor_minus_down_probe": summary_stats(q_samples - qd, BOOT_SEED + 4)}
    actor_fn = make_actor(network, pp)
    fake = SimpleNamespace(base=engine.base, nominal=engine.nominal, actor=actor_fn, actor_jit=None)
    a3 = ab.AbsorbingRollout(fake, support, "manski_absorbing")
    a1 = ab.AbsorbingRollout(fake, support, "diagonal_motion")
    m3, n3 = rollout_metrics(a3, fork_state, fork_len, FORK_PATHS, ctx["fork_key_a3"])
    m1, n1 = rollout_metrics(a1, fork_state, fork_len, FORK_PATHS_DIAG, ctx["fork_key_a1"])
    rec["fork_rollout_manski_absorbing"] = {k: summary_stats(v, BOOT_SEED + 10 + i) for i, (k, v) in enumerate(m3.items())}
    rec["fork_rollout_diagonal_motion"] = {k: summary_stats(m1[k], BOOT_SEED + 20 + i) for i, k in enumerate(
        ["reach", "lower_route", "strict_success", "discounted_return", "mean_step_displacement"])}
    for k, v in m3.items(): per_root[f"a3_{k}"] = v
    for k in ["reach", "lower_route"]: per_root[f"a1_{k}"] = m1[k]
    aloc, ascale, _ = oci.action_distribution(network, pp, away_obs, plan["away_eps"])
    amode = np.tanh(aloc)
    adist = network.policy_network.apply(pp, jnp.asarray(away_obs))
    abc = -np.asarray(network.log_prob(adist, jnp.asarray(away_action)))
    ma, na = rollout_metrics(a3, away_state, away_len, AWAY_PATHS, ctx["away_key"])
    rec["away"] = {
        "bc_nll_mean": float(abc.mean()),
        "bc_nll_change_vs_initial": summary_stats(abc - ctx["away_init_bc"], BOOT_SEED + 30),
        "mode_l2_change_vs_initial": summary_stats(np.linalg.norm(amode - ctx["away_init_mode"], axis=1), BOOT_SEED + 31),
        "mode_l2_to_recorded": float(np.linalg.norm(amode - away_action, axis=1).mean()),
        "landing_bin_agreement_mode_vs_recorded": bin_agreement(away_state, amode, away_action),
        "physical_landing_agreement_mode_vs_recorded": bin_agreement(away_state, amode, away_action, physical=True),
        "mode_ood_fraction": ood_fraction(pairs, away_state, amode),
        "scale_mean": float(ascale.mean()),
        "saturation_fraction_mode": float(np.mean(np.any(np.abs(amode) > .99, -1))),
        "rollout_manski_absorbing": {k: summary_stats(ma[k], BOOT_SEED + 40 + i) for i, k in enumerate(
            ["reach", "absorbed", "discounted_return", "entered_support"])}}
    comps = [oci.component_gradients(network, pp, critics[arm].q_params, oci.ordinary_batch(ctx["obs"], ctx["act"], plan, u),
                                     jnp.asarray(plan["actor_keys"][u])) for u in range(oci.CONFIG["gradient_batches"])]
    rec["gradient_components"] = {k: float(np.mean([c[k] for c in comps])) for k in comps[0]}
    return rec, per_root, n3 + n1 + na


def main():
    t0 = time.time()
    obs, act, initial, train, heldout, plan = oci.load_prepared(G1)
    freeze = oci.read_json(G1 / "freeze_tables.json")
    support = np.zeros(ab.GRID, bool)
    for i, j in freeze["support_cells"]:
        support[i, j] = True
    critics = {arm: checkpoint.load_checkpoint(G1 / "checkpoints" / f"critic_{arm}_final.pkl")[1] for arm in ARMS}
    g1_actors = {arm: checkpoint.load_checkpoint(G1 / "checkpoints" / f"actor_{arm}_final.pkl")[1] for arm in ARMS}
    network, _, policy_optimizer = oci.make_network_and_optimizers()
    engine = region.Kernel()
    immutable = oci.tree_sha((engine.base.params, engine.nominal.params))
    kept, e_idx, t_idx, key_info = build_keys(obs, act, train)

    ce, ct = plan["fork_context_episode"], plan["fork_context_time"]
    fork_state = obs[ce, ct, :8].astype(np.float32)
    ae, at, af = plan["away_episode"], plan["away_time"], plan["away_future"]
    away_state = obs[ae, at, :8].astype(np.float32)
    away_obs = np.concatenate([away_state, obs[ae, af, :8]], 1).astype(np.float32)
    away_action = act[ae, at].astype(np.float32)
    away_init_loc, _, _ = oci.action_distribution(network, initial.policy_params, away_obs, plan["away_eps"])
    init_dist = network.policy_network.apply(initial.policy_params, jnp.asarray(away_obs))
    ctx = dict(network=network, critics=critics, engine=engine, support=support, plan=plan,
               pairs=observed_pairs(obs, act, train), obs=obs, act=act,
               fork_state=fork_state, fork_len=(50 - ct).astype(np.int32),
               canonical=np.concatenate([fork_state, np.broadcast_to(GOAL, fork_state.shape)], 1).astype(np.float32),
               away_state=away_state, away_len=(50 - at).astype(np.int32), away_obs=away_obs, away_action=away_action,
               away_init_mode=np.tanh(away_init_loc),
               away_init_bc=-np.asarray(network.log_prob(init_dist, jnp.asarray(away_action))),
               score_fn=jax.jit(lambda q, o, a: oci.paired_scores(network, q, o, a)),
               fork_key_a3=jax.random.PRNGKey(2100), fork_key_a1=jax.random.PRNGKey(2200), away_key=jax.random.PRNGKey(2300))

    config = dict(arms=ARMS, row_seeds=ROW_SEEDS, key="(current cell, landing cell of recorded action), training partition",
                  key_min_rows=KEY_MIN_ROWS, key_info=key_info, objective="sealed production actor objective .5*bc_nll + .5*critic_term",
                  actor_updates=oci.CONFIG["actor_updates"], batch_rows=256,
                  fork_paths=FORK_PATHS, fork_paths_diagonal=FORK_PATHS_DIAG, away_paths=AWAY_PATHS,
                  rollout_horizon=HORIZON, rollout_cap=ROLLOUT_CAP, bootstrap_seed=BOOT_SEED,
                  sha256={p: oci.file_sha(ROOT / p) for p in [
                      "ett/absorbing_ett.py", "ett/pointmaze_absorbing_integration.py", "ett/pointmaze_offline_causal_integration.py",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_O_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_P_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/actor_O_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/actor_P_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/sampler_plan.npz",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/freeze_tables.json"]},
                  protocol_sha256=oci.file_sha(OUT / "PROTOCOL.md"),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  fields_read=["obs", "act"], environment_calls=0, native_steps=0, critic_updates=0)
    write(OUT / "config.json", config)

    results = {"config": config, "reference_uniform_G1": {}, "arms": {arm: {} for arm in ARMS}, "exposure": {}}
    per_root_all, curves, charged = {}, {}, 0
    # reference: the G1 uniform-row actors under the same evaluation
    for arm in ARMS:
        rec, per_root, n = evaluate_actor(f"{arm}_uniform_G1", g1_actors[arm].policy_params, arm, ctx)
        charged += n
        results["reference_uniform_G1"][arm] = rec
        for k, v in per_root.items(): per_root_all[f"{arm}_uniform/{k}"] = v
        print(f"{arm}_uniform_G1: phys (1,2) {rec['fork_landing']['physical']['(1,2) lower entry']:.3f} "
              f"| A3 reach {rec['fork_rollout_manski_absorbing']['reach']['mean']:.3f} abs {rec['fork_rollout_manski_absorbing']['absorbed']['mean']:.3f} "
              f"low {rec['fork_rollout_manski_absorbing']['lower_route']['mean']:.3f} | {time.time() - t0:.0f}s", flush=True)
    # uniform-plan exposure of fork keys for comparison
    pe, pt = plan["actor_episode"], plan["actor_time"]
    pc, pl = ab.landing_cell(obs[pe, pt, :2]), ab.landing_cell(obs[pe, pt, :2] + act[pe, pt])
    results["exposure"]["uniform_plan"] = {
        "fork_rows_share": float(np.mean((pc[..., 0] == 1) & (pc[..., 1] == 3))),
        "fork_to_1_2_share": float(np.mean((pc[..., 0] == 1) & (pc[..., 1] == 3) & (pl[..., 0] == 1) & (pl[..., 1] == 2))),
        "fork_to_2_3_share": float(np.mean((pc[..., 0] == 1) & (pc[..., 1] == 3) & (pl[..., 0] == 2) & (pl[..., 1] == 3)))}

    for seed in ROW_SEEDS:
        rng = np.random.default_rng(3_000_000 + seed)
        e, t, f, keys = draw_balanced_batches(rng, kept, e_idx, t_idx, oci.CONFIG["actor_updates"])
        rows = {"episode": e, "time": t, "future": f, "key": keys}
        np.savez_compressed(OUT / f"balanced_rows_seed{seed}.npz", **rows)
        assert np.isin(e, train).all() and np.all(f > t) and np.all(f <= 50)
        results["exposure"][f"balanced_seed{seed}"] = {
            "fork_rows_share": float(np.mean((keys // 1000 == 1) & ((keys // 100) % 10 == 3))),
            "fork_to_1_2_share": float(np.mean(keys == 1312)), "fork_to_2_3_share": float(np.mean(keys == 1323)),
            "distinct_keys_used": int(len(np.unique(keys)))}
        for arm in ARMS:
            name = f"{arm}_balanced_seed{seed}"
            state, curve = train_actor(network, policy_optimizer, initial, critics[arm], rows, plan, obs, act)
            checkpoint.save_named(OUT / "checkpoints", f"actor_{name}", 151400, state)
            curves[name] = curve
            rec, per_root, n = evaluate_actor(name, state.policy_params, arm, ctx)
            charged += n
            assert charged <= ROLLOUT_CAP
            rec["row_seed"] = seed
            rec["policy_parameter_delta_l2_vs_initial"] = oci.tree_delta_norm(state.policy_params, initial.policy_params)
            rec["policy_parameter_delta_l2_vs_G1_uniform"] = oci.tree_delta_norm(state.policy_params, g1_actors[arm].policy_params)
            tail = curve[-100:].mean(0)
            rec["curve_tail_mean"] = dict(zip(["loss", "bc_nll", "critic_actor_term", "sample_entropy", "scale_median",
                                              "loc_abs_mean", "saturation_fraction", "raw_gradient_l2", "optimizer_update_l2"], tail.tolist()))
            results["arms"][arm][f"seed{seed}"] = rec
            for k, v in per_root.items(): per_root_all[f"{name}/{k}"] = v
            write(OUT / f"actor_audit_{name}.json", rec)
            fl, m3, w = rec["fork_landing"], rec["fork_rollout_manski_absorbing"], rec["away"]
            print(f"{name}: phys (1,2) {fl['physical']['(1,2) lower entry']:.3f} (2,3) {fl['physical']['(2,3) right']:.3f} naive (1,2) {fl['naive']['(1,2) lower entry']:.3f} "
                  f"| A3 reach {m3['reach']['mean']:.3f} abs {m3['absorbed']['mean']:.3f} low {m3['lower_route']['mean']:.3f} "
                  f"| A1 reach {rec['fork_rollout_diagonal_motion']['reach']['mean']:.3f} | away agree {w['landing_bin_agreement_mode_vs_recorded']:.3f} "
                  f"reach {w['rollout_manski_absorbing']['reach']['mean']:.3f} | cos {rec['gradient_components']['component_cosine']:.3f} | {time.time() - t0:.0f}s", flush=True)

    # paired contrasts (seed 0 primary)
    results["paired"] = {}
    for seed in ROW_SEEDS:
        s = f"seed{seed}"; entry = {}
        for arm in ARMS:
            for metric, key in (("phys_12", "phys_12"), ("a3_reach", "a3_reach"), ("a3_absorbed", "a3_absorbed"), ("a3_lower_route", "a3_lower_route")):
                entry[f"{arm}_{metric}_change_vs_uniform"] = summary_stats(
                    per_root_all[f"{arm}_balanced_{s}/{key}"] - per_root_all[f"{arm}_uniform/{key}"], BOOT_SEED + 50)
        entry["P_minus_O_phys_12"] = summary_stats(per_root_all[f"P_balanced_{s}/phys_12"] - per_root_all[f"O_balanced_{s}/phys_12"], BOOT_SEED + 53)
        entry["P_minus_O_a3_lower_route"] = summary_stats(per_root_all[f"P_balanced_{s}/a3_lower_route"] - per_root_all[f"O_balanced_{s}/a3_lower_route"], BOOT_SEED + 54)
        entry["P_minus_O_a3_absorbed"] = summary_stats(per_root_all[f"P_balanced_{s}/a3_absorbed"] - per_root_all[f"O_balanced_{s}/a3_absorbed"], BOOT_SEED + 55)
        entry["P_minus_O_a3_reach"] = summary_stats(per_root_all[f"P_balanced_{s}/a3_reach"] - per_root_all[f"O_balanced_{s}/a3_reach"], BOOT_SEED + 56)
        results["paired"][s] = entry

    refP = results["reference_uniform_G1"]["P"]
    P0 = results["arms"]["P"]["seed0"]; O0 = results["arms"]["O"]["seed0"]
    e12 = P0["fork_landing"]["physical"]["(1,2) lower entry"]
    others = [results["arms"]["P"][f"seed{s}"]["fork_landing"]["physical"]["(1,2) lower entry"] for s in ROW_SEEDS[1:]]
    retention = (P0["away"]["landing_bin_agreement_mode_vs_recorded"] >= .85 * refP["away"]["landing_bin_agreement_mode_vs_recorded"]
                 and P0["away"]["rollout_manski_absorbing"]["reach"]["mean"] >= refP["away"]["rollout_manski_absorbing"]["reach"]["mean"] - .05
                 and P0["fork_rollout_diagonal_motion"]["reach"]["mean"] >= .9 * refP["fork_rollout_diagonal_motion"]["reach"]["mean"])
    decision = dict(P_physical_lower_entry_seed0=e12, P_physical_lower_entry_other_seeds=others,
                    hard_pass=bool(e12 >= .3), strong_pass=bool(e12 >= .4),
                    seeds_consistent=bool(all(abs(x - e12) <= .1 for x in others)),
                    retention=bool(retention),
                    secondary=bool(P0["fork_rollout_manski_absorbing"]["absorbed"]["mean"] < refP["fork_rollout_manski_absorbing"]["absorbed"]["mean"]
                                   and P0["fork_rollout_manski_absorbing"]["reach"]["mean"] > refP["fork_rollout_manski_absorbing"]["reach"]["mean"]),
                    O_physical_lower_entry_seed0=O0["fork_landing"]["physical"]["(1,2) lower entry"],
                    O_spurious_flag=bool(O0["fork_landing"]["physical"]["(1,2) lower entry"] > .25))
    decision["g3_candidate"] = bool(decision["hard_pass"] and decision["retention"] and decision["seeds_consistent"])
    decision["recommendation"] = ("G2c P actor (seed 0) is the G3 candidate" if decision["g3_candidate"]
                                  else "primary not met; next is G1' (balanced critic batches), not a new objective")
    results["decision"] = decision
    results["rollout_transitions"] = charged
    results["integrity"] = {"critic_updates": 0, "environment_calls": 0, "native_steps": 0,
                            "diagonal_and_nominal_unchanged": immutable == oci.tree_sha((engine.base.params, engine.nominal.params)),
                            "elapsed_seconds": time.time() - t0}
    np.savez_compressed(OUT / "learning_curves.npz", **curves)
    np.savez_compressed(OUT / "per_root.npz", fork_context_episode=ce, fork_context_time=ct, **per_root_all)
    write(OUT / "results.json", results)
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
