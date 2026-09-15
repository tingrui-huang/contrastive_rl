"""G2: BC-coefficient sweep of the actor stage on the frozen G1 critics.

Strictly offline. See PROTOCOL.md in this directory.  Reuses the sealed G1
plan, batches, keys, roots, critics and evaluation helpers; changes only the
BC coefficient of the production actor objective.
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
import optax

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crl import checkpoint  # noqa: E402
from ett import absorbing_ett as ab  # noqa: E402
from ett import pointmaze_offline_causal_integration as oci  # noqa: E402
from ett import pointmaze_region_pilot as region  # noqa: E402
from ett.rollout_return import GOAL  # noqa: E402

OUT = Path(__file__).resolve().parent
G1 = ROOT / "outputs/pointmaze_absorbing_integration_20260915_v1"
COEFS = [0.5, 0.2, 0.1, 0.05, 0.0]
ARMS = ["O", "P"]
FORK_PATHS, FORK_PATHS_DIAG, AWAY_PATHS, HORIZON = 32, 16, 8, 49
ROLLOUT_CAP = 30_000_000
BOOT_SEED = 0
PROBE_DOWN, PROBE_RIGHT = 0, 4  # indices into plan["action_probes"]
GOAL2 = np.asarray(GOAL[:2], np.float64)


def tag(coef):
    return f"bc{coef:g}".replace(".", "p")


def write(path, value):
    oci.write_json(path, value)


# --------------------------------------------------------------------------- #
# actor stage with a configurable BC coefficient                                #
# --------------------------------------------------------------------------- #
def build_actor_step(network, optimizer, bc_coef):
    def objective(policy_params, q_params, batch, key):
        bc, critic, auxiliary = oci.actor_parts(network, policy_params, q_params, batch, key)
        return bc_coef * bc + (1. - bc_coef) * critic, (bc, critic, auxiliary)
    grad_fn = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def step(policy_params, optimizer_state, q_params, batch, key):
        (loss, (bc, critic, auxiliary)), gradient = grad_fn(policy_params, q_params, batch, key)
        update, optimizer_state = optimizer.update(gradient, optimizer_state)
        policy_params = optax.apply_updates(policy_params, update)
        return (policy_params, optimizer_state, loss, bc, critic, auxiliary,
                optax.global_norm(gradient), optax.global_norm(update))
    return step


def train_actor(network, optimizer, initial, critic_state, coef, obs, act, plan):
    step = build_actor_step(network, optimizer, coef)
    state = initial._replace(q_params=critic_state.q_params,
                             target_q_params=critic_state.target_q_params,
                             q_optimizer_state=critic_state.q_optimizer_state)
    rows = []
    for u in range(oci.CONFIG["actor_updates"]):
        batch = oci.ordinary_batch(obs, act, plan, u)
        pp, po, loss, bc, critic, auxiliary, gn, un = step(
            state.policy_params, state.policy_optimizer_state, state.q_params,
            batch, jnp.asarray(plan["actor_keys"][u]))
        values = np.asarray([loss, bc, critic, *auxiliary, gn, un], np.float64)
        if not np.isfinite(values).all() or not oci.finite_tree((pp, po)):
            raise FloatingPointError((coef, u, values))
        state = state._replace(policy_params=pp, policy_optimizer_state=po)
        rows.append(values)
    assert oci.tree_sha(state.q_params) == oci.tree_sha(critic_state.q_params)
    return state, np.asarray(rows)


# --------------------------------------------------------------------------- #
# evaluation helpers                                                            #
# --------------------------------------------------------------------------- #
def make_actor(network, policy_params):
    @jax.jit
    def sample(s, g, key):
        d = network.policy_network.apply(policy_params, jnp.concatenate([s, g], -1))
        return jnp.tanh(d.loc + d.scale * jax.random.normal(key, d.loc.shape))
    return sample


def rollout_metrics(backend, states, lengths, paths, key):
    """Roll all roots to HORIZON, mask to remaining length, average per root."""
    n = len(states)
    s = np.repeat(states, paths, 0)
    out = backend.rollout(np.zeros(48, np.float32), s, key, HORIZON)
    L = np.minimum(np.repeat(lengths, paths), HORIZON)  # roots at t=0 lose their 50th step
    t = np.arange(HORIZON)[None]
    mask = t < L[:, None]
    r = out["reward"] * mask
    disc = .95 ** np.arange(HORIZON)
    xy = out["xy"] if "xy" in out else out["states"][:, :, :2]
    smask = np.arange(HORIZON + 1)[None] <= L[:, None]
    dist = np.where(smask, np.linalg.norm(xy - GOAL2, axis=-1), np.inf)
    lower = np.where(smask, xy[:, :, 1] < 2., False).any(1)
    absorbed = out["frozen"][np.arange(n * paths), L]
    metrics = dict(
        discounted_return=(r * disc).sum(1),
        reach=(r > 0).any(1).astype(float),
        strict_success=(dist.min(1) < .5).astype(float),
        lower_route=lower.astype(float),
        absorbed=absorbed.astype(float),
        entered_support=(out["hazard_alive"] & mask).any(1).astype(float),
        manski_onset=(out["onset_manski"] & mask).any(1).astype(float))
    return {k: v.reshape(n, paths).mean(1) for k, v in metrics.items()}, int(n * paths * HORIZON)


def observed_pairs(obs, act, train):
    """(current cell, landing cell of the recorded action) pairs seen in training."""
    xy = obs[train, :50, :2].reshape(-1, 2)
    a = act[train, :50].reshape(-1, 2)
    c = ab.landing_cell(xy); l = ab.landing_cell(xy + a)
    table = np.zeros(ab.GRID + ab.GRID, bool)
    table[c[:, 0], c[:, 1], l[:, 0], l[:, 1]] = True
    return table


def ood_fraction(table, states, actions):
    c = ab.landing_cell(states[:, :2]); l = ab.landing_cell(states[:, :2] + actions)
    return float(np.mean(~table[c[:, 0], c[:, 1], l[:, 0], l[:, 1]]))


def bin_agreement(states, actions, recorded):
    return float(np.mean(np.all(ab.landing_cell(states[:, :2] + actions) ==
                                ab.landing_cell(states[:, :2] + recorded), axis=1)))


def summary_stats(values, seed):
    return oci.bootstrap_mean(np.asarray(values, np.float64), seed)


def main():
    t0 = time.time()
    obs, act, initial, train, heldout, plan = oci.load_prepared(G1)
    g1_results = oci.read_json(G1 / "results.json")
    freeze = oci.read_json(G1 / "freeze_tables.json")
    support = np.zeros(ab.GRID, bool)
    for i, j in freeze["support_cells"]:
        support[i, j] = True
    critics = {arm: checkpoint.load_checkpoint(G1 / "checkpoints" / f"critic_{arm}_final.pkl")[1] for arm in ARMS}
    g1_actors = {arm: checkpoint.load_checkpoint(G1 / "checkpoints" / f"actor_{arm}_final.pkl")[1] for arm in ARMS}
    network, _, policy_optimizer = oci.make_network_and_optimizers()
    engine = region.Kernel()
    immutable = oci.tree_sha((engine.base.params, engine.nominal.params))
    pairs = observed_pairs(obs, act, train)

    ce, ct = plan["fork_context_episode"], plan["fork_context_time"]
    fork_state = obs[ce, ct, :8].astype(np.float32)
    fork_len = (50 - ct).astype(np.int32)
    canonical = np.concatenate([fork_state, np.broadcast_to(GOAL, fork_state.shape)], 1).astype(np.float32)
    probes = plan["action_probes"]
    ae, at, af = plan["away_episode"], plan["away_time"], plan["away_future"]
    away_state = obs[ae, at, :8].astype(np.float32)
    away_len = (50 - at).astype(np.int32)
    away_obs = np.concatenate([away_state, obs[ae, af, :8]], 1).astype(np.float32)
    away_action = act[ae, at].astype(np.float32)
    score_fn = jax.jit(lambda q, o, a: oci.paired_scores(network, q, o, a))
    probe_scores = {arm: np.stack([np.asarray(score_fn(critics[arm].q_params, jnp.asarray(canonical),
                                                       jnp.asarray(np.broadcast_to(p, (len(canonical), 2)))))
                                   for p in probes], 1) for arm in ARMS}

    config = dict(coefs=COEFS, arms=ARMS, actor_updates=oci.CONFIG["actor_updates"],
                  objective="bc_coef * bc_nll + (1 - bc_coef) * critic_term",
                  fork_paths=FORK_PATHS, fork_paths_diagonal=FORK_PATHS_DIAG, away_paths=AWAY_PATHS,
                  rollout_horizon=HORIZON, rollout_cap=ROLLOUT_CAP, bootstrap_seed=BOOT_SEED,
                  g1_dir=str(G1.relative_to(ROOT)).replace("\\", "/"),
                  sha256={p: oci.file_sha(ROOT / p) for p in [
                      "ett/absorbing_ett.py", "ett/pointmaze_absorbing_integration.py",
                      "ett/pointmaze_offline_causal_integration.py",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_O_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_P_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/sampler_plan.npz",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/freeze_tables.json"]},
                  protocol_sha256=oci.file_sha(OUT / "PROTOCOL.md"),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  fields_read=["obs", "act"], environment_calls=0, native_steps=0)
    write(OUT / "config.json", config)

    # -- initial-actor references -------------------------------------------
    init_loc, _, init_sample = oci.action_distribution(network, initial.policy_params, canonical, plan["fork_eps"])
    away_init_loc, _, _ = oci.action_distribution(network, initial.policy_params, away_obs, plan["away_eps"])
    away_init_mode = np.tanh(away_init_loc)
    init_dist = network.policy_network.apply(initial.policy_params, jnp.asarray(away_obs))
    away_init_bc = -np.asarray(network.log_prob(init_dist, jnp.asarray(away_action)))

    results = {"config": config, "arms": {arm: {} for arm in ARMS}, "reproduction": {}, "rollout_transitions": 0}
    per_root = {}
    curves = {}
    charged = 0
    fork_key_a3 = jax.random.PRNGKey(2100)
    fork_key_a1 = jax.random.PRNGKey(2200)
    away_key = jax.random.PRNGKey(2300)
    for coef in COEFS:
        for arm in ARMS:
            name = f"{arm}_{tag(coef)}"
            state, rows = train_actor(network, policy_optimizer, initial, critics[arm], coef, obs, act, plan)
            checkpoint.save_named(OUT / "checkpoints", f"actor_{name}", 151400, state)
            curves[name] = rows
            pp = state.policy_params
            if coef == 0.5:
                results["reproduction"][arm] = {
                    "policy_sha256_equal_to_G1": oci.tree_sha(pp) == oci.tree_sha(g1_actors[arm].policy_params),
                    "policy_delta_l2_vs_G1": oci.tree_delta_norm(pp, g1_actors[arm].policy_params)}
            tail = rows[-100:].mean(0)
            rec = {"bc_coef": coef, "arm": arm,
                   "policy_parameter_delta_l2_vs_initial": oci.tree_delta_norm(pp, initial.policy_params),
                   "curve_tail_mean": dict(zip(["loss", "bc_nll", "critic_actor_term", "sample_entropy",
                                                "scale_median", "loc_abs_mean", "saturation_fraction",
                                                "raw_gradient_l2", "optimizer_update_l2"], tail.tolist()))}
            # 1. fork action distribution
            loc, scale, sample = oci.action_distribution(network, pp, canonical, plan["fork_eps"])
            down, right = oci.direction_masks(sample)
            rec["fork"] = oci.action_summary(sample)
            rec["fork"]["saturation_fraction"] = float(np.mean(np.any(np.abs(sample) > .99, -1)))
            rec["fork"]["mode_ood_fraction"] = ood_fraction(pairs, fork_state, np.tanh(loc))
            rec["fork"]["sample_ood_fraction"] = ood_fraction(
                pairs, np.repeat(fork_state, sample.shape[1], 0), sample.reshape(-1, 2))
            per_root[f"{name}/fork_down"] = down.mean(1); per_root[f"{name}/fork_right"] = right.mean(1)
            idown, iright = oci.direction_masks(init_sample)
            rec["fork"]["paired_down_change_vs_initial"] = summary_stats(down.mean(1) - idown.mean(1), BOOT_SEED + 1)
            rec["fork"]["paired_right_change_vs_initial"] = summary_stats(right.mean(1) - iright.mean(1), BOOT_SEED + 2)
            # 3. critic-induced preference at actor samples (own frozen critic)
            flat = sample.reshape(-1, 2)
            q_samples = np.asarray(score_fn(critics[arm].q_params, jnp.asarray(np.repeat(canonical, sample.shape[1], 0)),
                                            jnp.asarray(flat))).reshape(len(canonical), -1).mean(1)
            rec["critic_at_actor"] = {
                "mean_q_actor_samples": float(q_samples.mean()),
                "mean_q_down_probe": float(probe_scores[arm][:, PROBE_DOWN].mean()),
                "mean_q_right_probe": float(probe_scores[arm][:, PROBE_RIGHT].mean()),
                "actor_minus_right_probe": summary_stats(q_samples - probe_scores[arm][:, PROBE_RIGHT], BOOT_SEED + 3),
                "actor_minus_down_probe": summary_stats(q_samples - probe_scores[arm][:, PROBE_DOWN], BOOT_SEED + 4),
                "critic_endpoint_down_minus_right_G1": g1_results["critic"][arm]["endpoint_down_minus_right"]}
            # 4. model rollouts from fork roots
            actor_fn = make_actor(network, pp)
            fake = SimpleNamespace(base=engine.base, nominal=engine.nominal, actor=actor_fn, actor_jit=None)
            a3 = ab.AbsorbingRollout(fake, support, "manski_absorbing")
            a1 = ab.AbsorbingRollout(fake, support, "diagonal_motion")
            m3, n3 = rollout_metrics(a3, fork_state, fork_len, FORK_PATHS, fork_key_a3)
            m1, n1 = rollout_metrics(a1, fork_state, fork_len, FORK_PATHS_DIAG, fork_key_a1)
            charged += n3 + n1
            rec["fork_rollout_manski_absorbing"] = {k: summary_stats(v, BOOT_SEED + 10 + i) for i, (k, v) in enumerate(m3.items())}
            rec["fork_rollout_diagonal_motion"] = {k: summary_stats(m1[k], BOOT_SEED + 20 + i) for i, k in enumerate(["reach", "lower_route", "discounted_return", "strict_success"])}
            for k, v in m3.items(): per_root[f"{name}/a3_{k}"] = v
            for k in ["reach", "lower_route"]: per_root[f"{name}/a1_{k}"] = m1[k]
            # 5. non-fork retention on away rows
            aloc, ascale, _ = oci.action_distribution(network, pp, away_obs, plan["away_eps"])
            amode = np.tanh(aloc)
            adist = network.policy_network.apply(pp, jnp.asarray(away_obs))
            abc = -np.asarray(network.log_prob(adist, jnp.asarray(away_action)))
            ma, na = rollout_metrics(a3, away_state, away_len, AWAY_PATHS, away_key)
            charged += na
            rec["away"] = {
                "bc_nll_mean": float(abc.mean()),
                "bc_nll_change_vs_initial": summary_stats(abc - away_init_bc, BOOT_SEED + 30),
                "mode_l2_change_vs_initial": summary_stats(np.linalg.norm(amode - away_init_mode, axis=1), BOOT_SEED + 31),
                "mode_l2_to_recorded": float(np.linalg.norm(amode - away_action, axis=1).mean()),
                "landing_bin_agreement_mode_vs_recorded": bin_agreement(away_state, amode, away_action),
                "landing_bin_agreement_initial_actor": bin_agreement(away_state, away_init_mode, away_action),
                "mode_ood_fraction": ood_fraction(pairs, away_state, amode),
                "scale_mean": float(ascale.mean()),
                "saturation_fraction_mode": float(np.mean(np.any(np.abs(amode) > .99, -1))),
                "rollout_manski_absorbing": {k: summary_stats(ma[k], BOOT_SEED + 40 + i) for i, k in enumerate(["reach", "absorbed", "discounted_return", "entered_support"])}}
            assert charged <= ROLLOUT_CAP
            # 6. gradient components on the sealed batches
            comps = [oci.component_gradients(network, pp, critics[arm].q_params, oci.ordinary_batch(obs, act, plan, u),
                                             jnp.asarray(plan["actor_keys"][u])) for u in range(oci.CONFIG["gradient_batches"])]
            g = {k: float(np.mean([c[k] for c in comps])) for k in comps[0]}
            g["bc_coef_weighted_gradient_l2"] = coef * g["bc_raw_gradient_l2"]
            g["critic_coef_weighted_gradient_l2"] = (1 - coef) * g["critic_raw_gradient_l2"]
            rec["gradient_components"] = g
            results["arms"][arm][tag(coef)] = rec
            write(OUT / f"actor_audit_{name}.json", rec)
            print(f"{name}: fork down {rec['fork']['down_probability']:.3f} right {rec['fork']['right_probability']:.3f} "
                  f"| A3 reach {m3['reach'].mean():.3f} absorbed {m3['absorbed'].mean():.3f} lower {m3['lower_route'].mean():.3f} "
                  f"| away agree {rec['away']['landing_bin_agreement_mode_vs_recorded']:.3f} reach {ma['reach'].mean():.3f} "
                  f"| cos {g['component_cosine']:.3f} | {time.time() - t0:.0f}s", flush=True)

    # -- paired contrasts ---------------------------------------------------
    results["paired"] = {}
    for coef in COEFS:
        t = tag(coef)
        entry = {}
        for arm in ARMS:
            base = per_root[f"{arm}_bc0p5/fork_down"], per_root[f"{arm}_bc0p5/fork_right"]
            entry[f"{arm}_down_change_vs_bc0p5"] = summary_stats(per_root[f"{arm}_{t}/fork_down"] - base[0], BOOT_SEED + 50)
            entry[f"{arm}_right_change_vs_bc0p5"] = summary_stats(per_root[f"{arm}_{t}/fork_right"] - base[1], BOOT_SEED + 51)
            entry[f"{arm}_a3_reach_change_vs_bc0p5"] = summary_stats(per_root[f"{arm}_{t}/a3_reach"] - per_root[f"{arm}_bc0p5/a3_reach"], BOOT_SEED + 52)
        entry["P_minus_O_down"] = summary_stats(per_root[f"P_{t}/fork_down"] - per_root[f"O_{t}/fork_down"], BOOT_SEED + 53)
        entry["P_minus_O_right"] = summary_stats(per_root[f"P_{t}/fork_right"] - per_root[f"O_{t}/fork_right"], BOOT_SEED + 54)
        entry["P_minus_O_a3_reach"] = summary_stats(per_root[f"P_{t}/a3_reach"] - per_root[f"O_{t}/a3_reach"], BOOT_SEED + 55)
        entry["P_minus_O_a3_absorbed"] = summary_stats(per_root[f"P_{t}/a3_absorbed"] - per_root[f"O_{t}/a3_absorbed"], BOOT_SEED + 56)
        results["paired"][t] = entry

    # -- decision -----------------------------------------------------------
    decision = {"coefficients": {}}
    baseline_P = results["arms"]["P"]["bc0p5"]
    for coef in COEFS:
        t = tag(coef)
        P, O = results["arms"]["P"][t], results["arms"]["O"][t]
        down, right = P["fork"]["down_probability"], P["fork"]["right_probability"]
        primary = down >= .5 and (right == 0 or down / right >= 1.5)
        retention = (P["away"]["landing_bin_agreement_mode_vs_recorded"] >=
                     .85 * baseline_P["away"]["landing_bin_agreement_mode_vs_recorded"] and
                     P["away"]["rollout_manski_absorbing"]["reach"]["mean"] >=
                     baseline_P["away"]["rollout_manski_absorbing"]["reach"]["mean"] - .05)
        o_flag = O["fork"]["down_probability"] > .25
        decision["coefficients"][t] = dict(bc_coef=coef, P_down=down, P_right=right,
                                           P_down_over_right=(down / right if right > 0 else None),
                                           primary=bool(primary), retention=bool(retention),
                                           O_down=O["fork"]["down_probability"], O_spurious_flag=bool(o_flag))
    passing = [c for c in COEFS if c > 0 and decision["coefficients"][tag(c)]["primary"]
               and decision["coefficients"][tag(c)]["retention"]]
    zero = decision["coefficients"][tag(0.0)]
    if passing:
        decision["selected_bc_coef"] = max(passing)
        decision["recommendation"] = "G3 candidate selected"
    elif zero["primary"] and zero["retention"]:
        decision["selected_bc_coef"] = 0.0
        decision["recommendation"] = "only bc=0 flips with intact retention; diagnostic, confirm before G3"
    else:
        decision["selected_bc_coef"] = None
        decision["recommendation"] = "no coefficient flips the P actor with retention; run AWR extraction (G2b)"
    results["decision"] = decision
    results["rollout_transitions"] = charged
    results["integrity"] = {"critic_updates": 0, "environment_calls": 0, "native_steps": 0,
                            "diagonal_and_nominal_unchanged": immutable == oci.tree_sha((engine.base.params, engine.nominal.params)),
                            "elapsed_seconds": time.time() - t0}
    np.savez_compressed(OUT / "learning_curves.npz", **curves)
    np.savez_compressed(OUT / "per_root.npz", fork_context_episode=ce, fork_context_time=ct, **per_root)
    write(OUT / "results.json", results)
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
