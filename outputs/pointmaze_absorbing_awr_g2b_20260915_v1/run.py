"""G2b: AWR extraction on recorded actions with the frozen G1 critics.

Strictly offline. See PROTOCOL.md in this directory.  Reuses the sealed G1
plan, batches, keys, roots and critics; the actor loss is the only change.
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
from ett.diagonal_transition import POINTMAZE_WALLS  # noqa: E402
from ett.rollout_return import GOAL  # noqa: E402

OUT = Path(__file__).resolve().parent
G1 = ROOT / "outputs/pointmaze_absorbing_integration_20260915_v1"
G2 = ROOT / "outputs/pointmaze_absorbing_actor_sweep_20260915_v1"
BETAS = [0.0, 0.5, 1.0, 2.0, 5.0]
ARMS = ["O", "P"]
BANK_SIZE, BANK_SEED, NOMINAL_DRAWS = 32, 0, 32
LOG_W_CLIP = 5.
FORK_PATHS, FORK_PATHS_DIAG, AWAY_PATHS, HORIZON = 32, 16, 8, 49
ROLLOUT_CAP = 30_000_000
BOOT_SEED = 0
GOAL2 = np.asarray(GOAL[:2], np.float64)
CELLS = {"(1,2) lower entry": (1, 2), "(2,3) right": (2, 3), "(1,3) stay": (1, 3), "(0,3) left": (0, 3)}


def tag(beta):
    return f"beta{beta:g}".replace(".", "p")


def write(path, value):
    oci.write_json(path, value)


def summary_stats(values, seed):
    return oci.bootstrap_mean(np.asarray(values, np.float64), seed)


# --------------------------------------------------------------------------- #
# recorded-action bank and precomputed advantages                               #
# --------------------------------------------------------------------------- #
def build_bank(obs, act, train):
    xy = obs[train, :50, :2].reshape(-1, 2)
    a = act[train, :50].reshape(-1, 2).astype(np.float32)
    cell = ab.landing_cell(xy)
    rng = np.random.default_rng(BANK_SEED)
    bank = np.full(ab.GRID + (BANK_SIZE, 2), np.nan, np.float32)
    counts = np.zeros(ab.GRID, np.int64)
    for i in range(ab.GRID[0]):
        for j in range(ab.GRID[1]):
            rows = np.flatnonzero((cell[:, 0] == i) & (cell[:, 1] == j))
            counts[i, j] = len(rows)
            if len(rows) == 0:
                continue
            pick = rng.choice(rows, size=min(BANK_SIZE, len(rows)), replace=False)
            bank[i, j, :len(pick)] = a[pick]
    return bank, counts


def doubled_rows(obs, act, plan, u):
    """The production actor stage's row doubling, in the same order."""
    e, t, f = plan["actor_episode"][u], plan["actor_time"][u], plan["actor_future"][u]
    state = obs[e, t, :8].astype(np.float32)
    goal = obs[e, f, :8].astype(np.float32)
    action = act[e, t].astype(np.float32)
    new_state = np.concatenate([state, state], 0)
    new_goal = np.concatenate([goal, np.roll(goal, 1, 0)], 0)
    original = np.concatenate([action, action], 0)
    return new_state, new_goal, original, np.concatenate([e, e]), np.concatenate([t, t])


def precompute_advantages(network, q_params, bank, obs, act, plan, nominal):
    score = jax.jit(lambda o, a: oci.paired_scores(network, q_params, o, a))
    n_updates = oci.CONFIG["actor_updates"]
    out = {k: [] for k in ("q_rec", "v_bank", "v_nominal", "cell_x", "cell_y", "land_x", "land_y",
                           "episode", "time")}
    nominal_key = jax.random.PRNGKey(7)
    for u in range(n_updates):
        s, g, a, e, t = doubled_rows(obs, act, plan, u)
        o = np.concatenate([s, g], 1)
        q_rec = np.asarray(score(jnp.asarray(o), jnp.asarray(a)))
        cell = ab.landing_cell(s[:, :2])
        cand = bank[cell[:, 0], cell[:, 1]]                       # [rows, K, 2]
        valid = np.isfinite(cand[..., 0])
        cand_flat = np.where(valid[..., None], cand, 0.).reshape(-1, 2).astype(np.float32)
        o_rep = np.repeat(o, BANK_SIZE, 0)
        q_bank = np.asarray(score(jnp.asarray(o_rep), jnp.asarray(cand_flat))).reshape(len(s), BANK_SIZE)
        v_bank = np.where(valid, q_bank, 0.).sum(1) / np.maximum(valid.sum(1), 1)
        nominal_key, sub = jax.random.split(nominal_key)
        x_nom = np.asarray(nominal.sample(jnp.asarray(s), sub, NOMINAL_DRAWS,
                                          goal=jnp.broadcast_to(jnp.asarray(GOAL), s.shape)))
        q_nom = np.asarray(score(jnp.asarray(np.repeat(o, NOMINAL_DRAWS, 0)),
                                 jnp.asarray(x_nom.reshape(-1, 2)))).reshape(len(s), NOMINAL_DRAWS)
        land = ab.landing_cell(s[:, :2] + a)
        out["q_rec"].append(q_rec); out["v_bank"].append(v_bank); out["v_nominal"].append(q_nom.mean(1))
        out["cell_x"].append(cell[:, 0]); out["cell_y"].append(cell[:, 1])
        out["land_x"].append(land[:, 0]); out["land_y"].append(land[:, 1])
        out["episode"].append(e); out["time"].append(t)
        if (u + 1) % 250 == 0:
            print("advantages", u + 1, flush=True)
    return {k: np.stack(v) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# AWR actor stage                                                               #
# --------------------------------------------------------------------------- #
def build_awr_step(network, optimizer):
    def objective(policy_params, observation, action, weight):
        distribution = network.policy_network.apply(policy_params, observation)
        log_prob = network.log_prob(distribution, action)
        loss = -jnp.mean(weight * log_prob)
        auxiliary = (jnp.mean(-log_prob), jnp.median(distribution.scale),
                     jnp.mean(jnp.abs(distribution.loc)),
                     jnp.mean((jnp.abs(jnp.tanh(distribution.loc)) > .99).astype(jnp.float32)))
        return loss, auxiliary
    grad_fn = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def step(policy_params, optimizer_state, observation, action, weight):
        (loss, auxiliary), gradient = grad_fn(policy_params, observation, action, weight)
        update, optimizer_state = optimizer.update(gradient, optimizer_state)
        policy_params = optax.apply_updates(policy_params, update)
        return (policy_params, optimizer_state, loss, auxiliary,
                optax.global_norm(gradient), optax.global_norm(update))
    return step


def weights_for(beta, advantage):
    log_w = np.clip(beta * advantage, -LOG_W_CLIP, LOG_W_CLIP)
    clipped = np.abs(beta * advantage) > LOG_W_CLIP
    w = np.exp(log_w)
    w = w / w.mean(1, keepdims=True)
    return w.astype(np.float32), clipped


def train_awr(network, optimizer, initial, critic_state, weights, obs, act, plan):
    step = build_awr_step(network, optimizer)
    state = initial._replace(q_params=critic_state.q_params,
                             target_q_params=critic_state.target_q_params,
                             q_optimizer_state=critic_state.q_optimizer_state)
    rows = []
    for u in range(oci.CONFIG["actor_updates"]):
        s, g, a, _, _ = doubled_rows(obs, act, plan, u)
        o = jnp.asarray(np.concatenate([s, g], 1))
        pp, po, loss, auxiliary, gn, un = step(state.policy_params, state.policy_optimizer_state,
                                                o, jnp.asarray(a), jnp.asarray(weights[u]))
        values = np.asarray([loss, *auxiliary, gn, un], np.float64)
        if not np.isfinite(values).all() or not oci.finite_tree((pp, po)):
            raise FloatingPointError((u, values))
        state = state._replace(policy_params=pp, policy_optimizer_state=po)
        rows.append(values)
    assert oci.tree_sha(state.q_params) == oci.tree_sha(critic_state.q_params)
    return state, np.asarray(rows)


# --------------------------------------------------------------------------- #
# evaluation helpers (as in G2)                                                 #
# --------------------------------------------------------------------------- #
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
    t = np.arange(HORIZON)[None]
    mask = t < L[:, None]
    r = out["reward"] * mask
    disc = .95 ** np.arange(HORIZON)
    xy = out["states"][:, :, :2]
    smask = np.arange(HORIZON + 1)[None] <= L[:, None]
    dist = np.where(smask, np.linalg.norm(xy - GOAL2, axis=-1), np.inf)
    lower = np.where(smask, xy[:, :, 1] < 2., False).any(1)
    step_disp = np.linalg.norm(np.diff(xy, axis=1), axis=-1) * mask
    metrics = dict(
        discounted_return=(r * disc).sum(1),
        reach=(r > 0).any(1).astype(float),
        strict_success=(dist.min(1) < .5).astype(float),
        lower_route=lower.astype(float),
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


def bin_agreement(states, actions, recorded):
    return float(np.mean(np.all(ab.landing_cell(states[:, :2] + actions) ==
                                ab.landing_cell(states[:, :2] + recorded), axis=1)))


def landing_distribution(states, sample):
    """First-step landing cells of [roots, K, 2] samples; per-root (1,2) rate too."""
    n, k, _ = sample.shape
    land = ab.landing_cell(np.repeat(states[:, :2], k, 0) + sample.reshape(-1, 2))
    wall = np.asarray(POINTMAZE_WALLS)[land[:, 0], land[:, 1]] == 1
    result = {name: float(np.mean((land[:, 0] == c[0]) & (land[:, 1] == c[1]))) for name, c in CELLS.items()}
    result["wall cell"] = float(np.mean(wall))
    result["other free cell"] = float(1. - sum(result.values()))
    down, right = oci.direction_masks(sample.reshape(-1, 2))
    is_12 = (land[:, 0] == 1) & (land[:, 1] == 2)
    result["mask_down"] = float(np.mean(down)); result["mask_right"] = float(np.mean(right))
    result["mask_down_false_positive"] = float(np.mean(down & ~is_12))
    result["mask_down_not_lower_entry_share"] = float(np.mean(~is_12[down])) if down.any() else None
    per_root_12 = is_12.reshape(n, k).mean(1)
    per_root_23 = ((land[:, 0] == 2) & (land[:, 1] == 3)).reshape(n, k).mean(1)
    per_root_wall = wall.reshape(n, k).mean(1)
    return result, per_root_12, per_root_23, per_root_wall


def main():
    t0 = time.time()
    obs, act, initial, train, heldout, plan = oci.load_prepared(G1)
    g1_results = oci.read_json(G1 / "results.json")
    g2_results = oci.read_json(G2 / "results.json")
    freeze = oci.read_json(G1 / "freeze_tables.json")
    support = np.zeros(ab.GRID, bool)
    for i, j in freeze["support_cells"]:
        support[i, j] = True
    critics = {arm: checkpoint.load_checkpoint(G1 / "checkpoints" / f"critic_{arm}_final.pkl")[1] for arm in ARMS}
    network, _, policy_optimizer = oci.make_network_and_optimizers()
    engine = region.Kernel()
    immutable = oci.tree_sha((engine.base.params, engine.nominal.params))
    pairs = observed_pairs(obs, act, train)
    bank, bank_counts = build_bank(obs, act, train)
    np.savez_compressed(OUT / "bank.npz", bank=bank, counts=bank_counts)

    config = dict(betas=BETAS, arms=ARMS, actor_updates=oci.CONFIG["actor_updates"],
                  objective="-mean(w * log pi(a_rec|s,g)); log_w = clip(beta*(Q - V_bank), -5, 5); w normalised to mean 1 per doubled batch",
                  baseline="mean Q over a fixed bank of 32 recorded training-partition actions from the current cell",
                  bank_size=BANK_SIZE, bank_seed=BANK_SEED, nominal_draws_diagnostic=NOMINAL_DRAWS,
                  log_weight_clip=LOG_W_CLIP, row_doubling="production actor stage (own goal + rolled goal)",
                  fork_paths=FORK_PATHS, fork_paths_diagonal=FORK_PATHS_DIAG, away_paths=AWAY_PATHS,
                  rollout_horizon=HORIZON, rollout_cap=ROLLOUT_CAP, bootstrap_seed=BOOT_SEED,
                  sha256={p: oci.file_sha(ROOT / p) for p in [
                      "ett/absorbing_ett.py", "ett/pointmaze_absorbing_integration.py",
                      "ett/pointmaze_offline_causal_integration.py",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_O_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/checkpoints/critic_P_final.pkl",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/sampler_plan.npz",
                      "outputs/pointmaze_absorbing_integration_20260915_v1/freeze_tables.json",
                      "outputs/pointmaze_absorbing_actor_sweep_20260915_v1/results.json"]},
                  protocol_sha256=oci.file_sha(OUT / "PROTOCOL.md"),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  fields_read=["obs", "act"], environment_calls=0, native_steps=0, critic_updates=0)
    write(OUT / "config.json", config)

    ce, ct = plan["fork_context_episode"], plan["fork_context_time"]
    fork_state = obs[ce, ct, :8].astype(np.float32)
    fork_len = (50 - ct).astype(np.int32)
    canonical = np.concatenate([fork_state, np.broadcast_to(GOAL, fork_state.shape)], 1).astype(np.float32)
    ae, at, af = plan["away_episode"], plan["away_time"], plan["away_future"]
    away_state = obs[ae, at, :8].astype(np.float32)
    away_len = (50 - at).astype(np.int32)
    away_obs = np.concatenate([away_state, obs[ae, af, :8]], 1).astype(np.float32)
    away_action = act[ae, at].astype(np.float32)
    score_fn = jax.jit(lambda q, o, a: oci.paired_scores(network, q, o, a))

    init_loc, _, init_sample = oci.action_distribution(network, initial.policy_params, canonical, plan["fork_eps"])
    init_land, init_12, _, _ = landing_distribution(fork_state, init_sample)
    away_init_loc, _, _ = oci.action_distribution(network, initial.policy_params, away_obs, plan["away_eps"])
    away_init_mode = np.tanh(away_init_loc)
    init_dist = network.policy_network.apply(initial.policy_params, jnp.asarray(away_obs))
    away_init_bc = -np.asarray(network.log_prob(init_dist, jnp.asarray(away_action)))

    # -- advantages (fixed critics, fixed batches, fixed bank) ---------------
    advantages, adv_summary = {}, {}
    for arm in ARMS:
        adv = precompute_advantages(network, critics[arm].q_params, bank, obs, act, plan, engine.nominal)
        adv["advantage"] = adv["q_rec"] - adv["v_bank"]
        adv["advantage_nominal"] = adv["q_rec"] - adv["v_nominal"]
        advantages[arm] = adv
        A = adv["advantage"]
        fork_rows = (adv["cell_x"] == 1) & (adv["cell_y"] == 3)
        land12 = (adv["land_x"] == 1) & (adv["land_y"] == 2)
        land23 = (adv["land_x"] == 2) & (adv["land_y"] == 3)
        by_cell = {}
        for i, j in zip(*np.nonzero(bank_counts > 0)):
            m = (adv["cell_x"] == i) & (adv["cell_y"] == j)
            if m.sum() >= 50:
                by_cell[f"({i},{j})"] = dict(rows=int(m.sum()), mean=float(A[m].mean()), std=float(A[m].std()),
                                             q05=float(np.quantile(A[m], .05)), q95=float(np.quantile(A[m], .95)))
        by_land = {}
        for i, j in zip(*np.nonzero(bank_counts > 0)):
            m = (adv["land_x"] == i) & (adv["land_y"] == j)
            if m.sum() >= 50:
                by_land[f"({i},{j})"] = dict(rows=int(m.sum()), mean=float(A[m].mean()), std=float(A[m].std()))
        adv_summary[arm] = dict(
            rows=int(A.size), mean=float(A.mean()), std=float(A.std()),
            quantiles={q: float(np.quantile(A, q)) for q in (.01, .05, .25, .5, .75, .95, .99)},
            correlation_with_nominal_baseline=float(np.corrcoef(A.ravel(), adv["advantage_nominal"].ravel())[0, 1]),
            mean_v_bank_minus_v_nominal=float((adv["v_bank"] - adv["v_nominal"]).mean()),
            fork_cell=dict(rows=int(fork_rows.sum()),
                           land_1_2=dict(rows=int((fork_rows & land12).sum()), mean=float(A[fork_rows & land12].mean()),
                                         std=float(A[fork_rows & land12].std()),
                                         mean_nominal_baseline=float(adv["advantage_nominal"][fork_rows & land12].mean())),
                           land_2_3=dict(rows=int((fork_rows & land23).sum()), mean=float(A[fork_rows & land23].mean()),
                                         std=float(A[fork_rows & land23].std()),
                                         mean_nominal_baseline=float(adv["advantage_nominal"][fork_rows & land23].mean())),
                           other=dict(rows=int((fork_rows & ~land12 & ~land23).sum()),
                                      mean=float(A[fork_rows & ~land12 & ~land23].mean()))),
            by_current_cell=by_cell, by_recorded_landing_cell=by_land)
        np.savez_compressed(OUT / f"advantages_{arm}.npz", **adv)
        print(arm, "advantages: fork (1,2) mean %.3f vs (2,3) mean %.3f" % (
            adv_summary[arm]["fork_cell"]["land_1_2"]["mean"], adv_summary[arm]["fork_cell"]["land_2_3"]["mean"]), flush=True)
    write(OUT / "advantage_summary.json", adv_summary)

    results = {"config": config, "advantages": adv_summary, "arms": {arm: {} for arm in ARMS},
               "weights": {arm: {} for arm in ARMS}, "initial_fork_landing": init_land}
    per_root, curves, charged = {}, {}, 0
    fork_key_a3, fork_key_a1, away_key = jax.random.PRNGKey(2100), jax.random.PRNGKey(2200), jax.random.PRNGKey(2300)
    for beta in BETAS:
        for arm in ARMS:
            name = f"{arm}_{tag(beta)}"
            adv = advantages[arm]
            w, clipped = weights_for(beta, adv["advantage"])
            fork_rows = (adv["cell_x"] == 1) & (adv["cell_y"] == 3)
            land12 = (adv["land_x"] == 1) & (adv["land_y"] == 2)
            ess = (w.sum(1) ** 2 / (w ** 2).sum(1))
            results["weights"][arm][tag(beta)] = dict(
                beta=beta, clipping_fraction=float(clipped.mean()),
                mean_effective_sample_size=float(ess.mean()), min_effective_sample_size=float(ess.min()),
                rows_per_batch=int(w.shape[1]), max_normalised_weight=float(w.max()),
                mean_max_weight_per_batch=float(w.max(1).mean()),
                weight_share_fork_rows=float(w[fork_rows].sum() / w.sum()),
                row_share_fork_rows=float(fork_rows.mean()),
                weight_share_fork_lower_entry_rows=float(w[fork_rows & land12].sum() / w.sum()),
                row_share_fork_lower_entry_rows=float((fork_rows & land12).mean()),
                mean_weight_fork_lower_entry=float(w[fork_rows & land12].mean()),
                mean_weight_fork_right=float(w[fork_rows & (adv["land_x"] == 2) & (adv["land_y"] == 3)].mean()))
            state, rows = train_awr(network, policy_optimizer, initial, critics[arm], w, obs, act, plan)
            checkpoint.save_named(OUT / "checkpoints", f"actor_{name}", 151400, state)
            curves[name] = rows
            pp = state.policy_params
            tail = rows[-100:].mean(0)
            rec = {"beta": beta, "arm": arm,
                   "policy_parameter_delta_l2_vs_initial": oci.tree_delta_norm(pp, initial.policy_params),
                   "curve_tail_mean": dict(zip(["awr_loss", "nll_recorded", "scale_median", "loc_abs_mean",
                                                "saturation_fraction", "raw_gradient_l2", "optimizer_update_l2"], tail.tolist()))}
            # 1. fork landing distribution
            loc, scale, sample = oci.action_distribution(network, pp, canonical, plan["fork_eps"])
            land, r12, r23, rwall = landing_distribution(fork_state, sample)
            rec["fork_landing"] = land
            rec["fork_landing"]["paired_lower_entry_change_vs_initial"] = summary_stats(r12 - init_12, BOOT_SEED + 1)
            rec["fork_action"] = oci.action_summary(sample)
            rec["fork_action"]["saturation_fraction"] = float(np.mean(np.any(np.abs(sample) > .99, -1)))
            rec["fork_action"]["mode_ood_fraction"] = ood_fraction(pairs, fork_state, np.tanh(loc))
            per_root[f"{name}/fork_12"], per_root[f"{name}/fork_23"], per_root[f"{name}/fork_wall"] = r12, r23, rwall
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
            # 2/3. model rollouts
            actor_fn = make_actor(network, pp)
            fake = SimpleNamespace(base=engine.base, nominal=engine.nominal, actor=actor_fn, actor_jit=None)
            a3 = ab.AbsorbingRollout(fake, support, "manski_absorbing")
            a1 = ab.AbsorbingRollout(fake, support, "diagonal_motion")
            m3, n3 = rollout_metrics(a3, fork_state, fork_len, FORK_PATHS, fork_key_a3)
            m1, n1 = rollout_metrics(a1, fork_state, fork_len, FORK_PATHS_DIAG, fork_key_a1)
            charged += n3 + n1
            rec["fork_rollout_manski_absorbing"] = {k: summary_stats(v, BOOT_SEED + 10 + i) for i, (k, v) in enumerate(m3.items())}
            rec["fork_rollout_diagonal_motion"] = {k: summary_stats(m1[k], BOOT_SEED + 20 + i) for i, k in enumerate(
                ["reach", "lower_route", "strict_success", "discounted_return", "mean_step_displacement"])}
            for k, v in m3.items(): per_root[f"{name}/a3_{k}"] = v
            for k in ["reach", "lower_route"]: per_root[f"{name}/a1_{k}"] = m1[k]
            # 4. retention
            aloc, ascale, _ = oci.action_distribution(network, pp, away_obs, plan["away_eps"])
            amode = np.tanh(aloc)
            adist = network.policy_network.apply(pp, jnp.asarray(away_obs))
            abc = -np.asarray(network.log_prob(adist, jnp.asarray(away_action)))
            ma, na = rollout_metrics(a3, away_state, away_len, AWAY_PATHS, away_key)
            charged += na
            assert charged <= ROLLOUT_CAP
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
                "rollout_manski_absorbing": {k: summary_stats(ma[k], BOOT_SEED + 40 + i) for i, k in enumerate(
                    ["reach", "absorbed", "discounted_return", "entered_support"])}}
            results["arms"][arm][tag(beta)] = rec
            write(OUT / f"actor_audit_{name}.json", rec)
            print(f"{name}: fork (1,2) {land['(1,2) lower entry']:.3f} (2,3) {land['(2,3) right']:.3f} wall {land['wall cell']:.3f} "
                  f"| A3 reach {m3['reach'].mean():.3f} abs {m3['absorbed'].mean():.3f} low {m3['lower_route'].mean():.3f} "
                  f"| A1 reach {m1['reach'].mean():.3f} | away agree {rec['away']['landing_bin_agreement_mode_vs_recorded']:.3f} "
                  f"reach {ma['reach'].mean():.3f} | clip {clipped.mean():.3f} ess {ess.mean():.0f} | {time.time() - t0:.0f}s", flush=True)

    # -- paired contrasts and decision --------------------------------------
    results["paired"] = {}
    for beta in BETAS:
        t = tag(beta)
        entry = {}
        for arm in ARMS:
            entry[f"{arm}_lower_entry_change_vs_beta0"] = summary_stats(per_root[f"{arm}_{t}/fork_12"] - per_root[f"{arm}_beta0/fork_12"], BOOT_SEED + 50)
            entry[f"{arm}_a3_reach_change_vs_beta0"] = summary_stats(per_root[f"{arm}_{t}/a3_reach"] - per_root[f"{arm}_beta0/a3_reach"], BOOT_SEED + 51)
            entry[f"{arm}_a3_absorbed_change_vs_beta0"] = summary_stats(per_root[f"{arm}_{t}/a3_absorbed"] - per_root[f"{arm}_beta0/a3_absorbed"], BOOT_SEED + 52)
        entry["P_minus_O_lower_entry"] = summary_stats(per_root[f"P_{t}/fork_12"] - per_root[f"O_{t}/fork_12"], BOOT_SEED + 53)
        entry["P_minus_O_a3_reach"] = summary_stats(per_root[f"P_{t}/a3_reach"] - per_root[f"O_{t}/a3_reach"], BOOT_SEED + 54)
        entry["P_minus_O_a3_absorbed"] = summary_stats(per_root[f"P_{t}/a3_absorbed"] - per_root[f"O_{t}/a3_absorbed"], BOOT_SEED + 55)
        entry["P_minus_O_a3_lower_route"] = summary_stats(per_root[f"P_{t}/a3_lower_route"] - per_root[f"O_{t}/a3_lower_route"], BOOT_SEED + 56)
        results["paired"][t] = entry

    g2p = g2_results["arms"]["P"]["bc0p5"]
    base = results["arms"]["P"]["beta0"]
    decision = {"g2_reference_P_bc0p5": {"a3_reach": g2p["fork_rollout_manski_absorbing"]["reach"]["mean"],
                                         "a3_absorbed": g2p["fork_rollout_manski_absorbing"]["absorbed"]["mean"]},
                "betas": {}}
    for beta in BETAS:
        t = tag(beta)
        P, O, W = results["arms"]["P"][t], results["arms"]["O"][t], results["weights"]["P"][t]
        e12 = P["fork_landing"]["(1,2) lower entry"]
        retention = (P["away"]["landing_bin_agreement_mode_vs_recorded"] >= .85 * base["away"]["landing_bin_agreement_mode_vs_recorded"]
                     and P["away"]["rollout_manski_absorbing"]["reach"]["mean"] >= base["away"]["rollout_manski_absorbing"]["reach"]["mean"] - .05
                     and P["fork_rollout_diagonal_motion"]["reach"]["mean"] >= .9 * base["fork_rollout_diagonal_motion"]["reach"]["mean"])
        secondary = (P["fork_rollout_manski_absorbing"]["absorbed"]["mean"] < decision["g2_reference_P_bc0p5"]["a3_absorbed"]
                     and P["fork_rollout_manski_absorbing"]["reach"]["mean"] > decision["g2_reference_P_bc0p5"]["a3_reach"])
        decision["betas"][t] = dict(beta=beta, P_lower_entry=e12, hard_pass=bool(e12 >= .3), strong_pass=bool(e12 >= .4),
                                    retention=bool(retention), secondary_vs_G2=bool(secondary),
                                    clipping_fraction=W["clipping_fraction"], clipping_excessive=bool(W["clipping_fraction"] > .25),
                                    O_lower_entry=O["fork_landing"]["(1,2) lower entry"],
                                    O_spurious_flag=bool(O["fork_landing"]["(1,2) lower entry"] > .25))
    passing = [b for b in BETAS if b > 0 and decision["betas"][tag(b)]["hard_pass"] and decision["betas"][tag(b)]["retention"]
               and not (b == 5. and decision["betas"][tag(b)]["clipping_excessive"])]
    if passing:
        decision["selected_beta"] = min(passing)
        decision["recommendation"] = "AWR candidate selected for G3"
    else:
        decision["selected_beta"] = None
        decision["recommendation"] = "AWR insufficient under the sealed rule; see diagnosis"
    results["decision"] = decision
    results["rollout_transitions"] = charged
    results["integrity"] = {"critic_updates": 0, "environment_calls": 0, "native_steps": 0,
                            "diagonal_and_nominal_unchanged": immutable == oci.tree_sha((engine.base.params, engine.nominal.params)),
                            "elapsed_seconds": time.time() - t0}
    np.savez_compressed(OUT / "learning_curves.npz", **curves)
    np.savez_compressed(OUT / "per_root.npz", fork_context_episode=ce, fork_context_time=ct, **per_root)
    write(OUT / "weight_diagnostics.json", results["weights"])
    write(OUT / "results.json", results)
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
