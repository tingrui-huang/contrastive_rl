"""The REAL-UPDATE check (user's request, 2026-09-20): does the actual actor
update -- real actor batches, real critic batches, the real Adam states, from
a copy of the clipped CF checkpoint -- move the policy at the states where a
detour candidate is VERIFIED (the user's frozen-model diagnostic: at that
reset state, with the task goal, the candidate's continuation entered the
far route) toward or away from that candidate?  With the critic-term and
BC-term contributions separated and the scale watched.  No proxies: the BC
pull is the one the real batches exert through the shared parameters.

Procedure per training seed s (the clipped CF final = the 30,000-update
state incl. both Adam states; `variants/critic_clip0.1/CF/seed_s/final.pkl`):
the learner is rebuilt exactly as in training (critic clip 0.1, separate
actor batch), the critic and actor streams are fast-forwarded by 30,000
batches (the run's own next batches), and N further real updates are
applied in the training's scan groups of 4.  Around every logged group:

  * at the verified states V (reset state s_v + task goal g; candidate c_v =
    the verified far-route candidate with the highest own-critic score in
    that context): the policy mode m_v = tanh(loc_v), the scale, the distance
    ||m_v - c_v||, the critic's f(s_v, m_v, g) and f(s_v, c_v, g);
  * the parameter direction d = grad_theta mean_v <tanh(loc_v), u_v> with
    u_v = (c_v - m_v) / ||.|| held fixed ("toward the candidates" in
    parameter space, policy parameters only);
  * on the group's real actor batches, with the pre-update critic (as the
    update does): the gradients of the two actor-loss terms, weighted as in
    the loss, g_q = grad (1 - bc) * mean(-min_h f) and g_bc = grad bc *
    mean(-log pi(a_logged)); reported as cosines and projections of the
    descent directions -g_q, -g_bc onto d;
  * the REAL update (the training step, batches and RNG as the run), then
    the realised change: the projection of the policy-parameter step onto
    d, and at V the change of ||m_v - c_v|| and the projection of the mode
    movement onto u_v.

Counterfactual replays from the same checkpoint with the same batches:
`--bc-coef 0` (critic term only) and `--bc-coef 1` (BC term only) show where
each term alone would take the modes (Adam steps; not additive).

Readings (user's): the critic update itself does not support the
candidates -> query coverage in the training context comes first; the
critic supports and BC cancels -> an objective conflict; the full update
supports -> ask how later updates cancel the improvement.

  python scripts/diag_v6_actor_real_update.py --seed s --updates 2000 [--bc-coef 0|1] --candidates /root/codex_joined.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import exp_v6_mainline_pilot as MP  # noqa: E402
import diag_v6_training_replay as DR  # noqa: E402

VARIANT = 'critic_clip0.1'
OUT = MP.OUT / 'diag_traj' / 'real_update'
STATE_DIM, OBS_W = MP.STATE_DIM, MP.OBS_W
G = MP.G
LOG_DENSE, LOG_EVERY = 50, 25       # every group for the first 50 groups (200 updates), then every 25 groups


def verified_contexts(J, seed):
  """Stable-shortcut contexts of seed `seed` with at least one verified far-route candidate; c_v = the verified candidate with the highest own-critic score."""
  out = []
  for c in J:
    if int(c['seed']) != seed or c.get('saved_route') != 'shortcut':
      continue
    names, sc = c['candidate_names'], c['scores']
    cands = [(sc[i], nm, np.asarray(c['actions'][i], np.float32)) for i, nm in enumerate(names) if nm != 'mode' and c['outcomes'].get(nm, {}).get('route') == 'detour']
    if not cands:
      continue
    best = max(cands, key=lambda t: t[0])
    out.append({'episode': int(c['episode']), 'candidate': best[1], 'user_score': float(best[0]), 'user_mode_score': float(sc[names.index('mode')]), 'action': best[2], 'n_verified': len(cands)})
  return out


def main():
  import jax
  import jax.numpy as jnp
  import optax
  from crl import checkpoint
  from crl import losses as losses_mod
  import diag_v6_pilot_trajectories as DT
  ap = argparse.ArgumentParser()
  ap.add_argument('--seed', type=int, required=True)
  ap.add_argument('--updates', type=int, default=2000)
  ap.add_argument('--bc-coef', type=float, default=None, help='counterfactual replay with this bc_coef (0 = critic term only, 1 = BC term only); default = the training value 0.05')
  ap.add_argument('--candidates', default='/root/codex_joined.json')
  ap.add_argument('--probe-key', type=int, default=211_000_000)
  args = ap.parse_args()
  OUT.mkdir(parents=True, exist_ok=True)
  s = args.seed
  tag = f'seed{s}' + ('' if args.bc_coef is None else f'_bc{args.bc_coef:g}')
  J = json.loads(Path(args.candidates).read_text(encoding='utf-8'))
  V = verified_contexts(J, s)
  T = DT.Traj(f'CF_s{s}@{VARIANT}')
  O_v = np.stack([T.obs(v['episode'])[0][:OBS_W] for v in V]).astype(np.float32)          # reset state + task goal (the evaluation draw)
  C_v = np.stack([v['action'] for v in V]).astype(np.float32)
  print(f'seed {s}: {len(V)} verified stable-shortcut contexts', flush=True)
  # the learner exactly as train_arm (clip), with an optional bc_coef override for the counterfactual replays
  cfg = MP.recipe_config(s, OUT / '_cfg'); cfg.batch_size = MP.BATCH; MP.fill_dims(cfg)
  bc_used = cfg.bc_coef if args.bc_coef is None else float(args.bc_coef)
  cfg.bc_coef = bc_used
  nets = MP.make_nets(cfg)
  pol_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7); q_opt = MP.critic_optimizer(cfg, float(MP.VARIANTS[VARIANT]['critic_clip']))
  gidx = np.asarray(cfg.goal_indices)

  def obs_to_goal(states):
    return states[:, jnp.asarray(gidx)]
  _, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, pol_opt, q_opt, separate_actor_batch=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); obs, act, lengths, _ = MP.load_dataset()
  futures = MP.BranchFutures(anchors, MP.OUT / 'branches_cf.npz')
  critic_stream = MP.CriticStream(anchors, futures, cfg.batch_size, cfg.discount, MP.CRITIC_STREAM_SEED0 + s)
  actor_stream = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + s)
  src = MP.run_dir('CF', s, MP.variant_base(VARIANT)) / 'final.pkl'
  step0, state = checkpoint.load_checkpoint(src)
  assert jax.tree_util.tree_structure(state.q_optimizer_state) == jax.tree_util.tree_structure(q_opt.init(state.q_params)), 'critic optimizer state layout'
  ff = DR.fast_forward(critic_stream, actor_stream, int(step0))
  print(f'loaded {src} @ {step0}; streams fast-forwarded in {ff:.0f} s; bc_coef {bc_used}', flush=True)

  def _multi(state, pair):
    state, metrics = jax.lax.scan(update_step, state, pair)
    return state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
  multi_update = jax.jit(_multi)

  # probes at the verified states
  @jax.jit
  def probe(pp, qp, O, C):
    d = nets.policy_network.apply(pp, O); mode = jnp.tanh(d.loc)
    phi_m, psi_m = nets.representation_network.apply(qp, O, mode); f_m = jnp.min(jnp.sum(phi_m * psi_m, axis=1), axis=1)
    phi_c, psi_c = nets.representation_network.apply(qp, O, C); f_c = jnp.min(jnp.sum(phi_c * psi_c, axis=1), axis=1)
    return mode, d.scale, f_m, f_c

  def toward_direction(pp, O, U):
    """grad_theta mean_v <tanh(loc_v), u_v> (policy params)."""
    def J(p):
      return jnp.mean(jnp.sum(jnp.tanh(nets.policy_network.apply(p, O).loc) * U, axis=1))
    return jax.grad(J)(pp)
  toward_direction = jax.jit(toward_direction)

  def term_grads(pp, qp, batch, key):
    """Weighted gradients of the two actor-loss terms on one real actor batch (sampled action from the current policy, as the loss)."""
    Ob = batch.observation; A = batch.action

    def q_term(p):
      d = nets.policy_network.apply(p, Ob); a = nets.sample(d, key)
      q = nets.q_network.apply(qp, Ob, a); q = jnp.min(q, axis=-1) if q.ndim == 3 else q
      return (1.0 - bc_used) * jnp.mean(-jnp.diag(q))

    def bc_term(p):
      return bc_used * jnp.mean(-nets.log_prob(nets.policy_network.apply(p, Ob), A))
    return jax.grad(q_term)(pp), jax.grad(bc_term)(pp)
  term_grads = jax.jit(term_grads)

  def flat(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])

  def proj(g, d):
    gf, df = flat(g), flat(d); dn = jnp.linalg.norm(df) + 1e-12
    return float(jnp.dot(gf, df) / dn), float(jnp.dot(gf, df) / (dn * (jnp.linalg.norm(gf) + 1e-12)))
  key = jax.random.PRNGKey(args.probe_key + s)
  rec, t0 = [], time.time()
  n_groups = args.updates // G
  m0, sc0, fm0, fc0 = (np.asarray(v) for v in probe(state.policy_params, state.q_params, jnp.asarray(O_v), jnp.asarray(C_v)))
  d_start = np.linalg.norm(m0 - C_v, axis=1)
  for it in range(n_groups):
    cbs = [critic_stream.sample()[0] for _ in range(G)]
    abs_ = [actor_stream.sample(cfg.batch_size) for _ in range(G)]
    log = it < LOG_DENSE or (it + 1) % LOG_EVERY == 0 or it == n_groups - 1
    if log:
      mode, scale, f_m, f_c = (np.asarray(v) for v in probe(state.policy_params, state.q_params, jnp.asarray(O_v), jnp.asarray(C_v)))
      diff = C_v - mode; U = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8)
      d_dir = toward_direction(state.policy_params, jnp.asarray(O_v), jnp.asarray(U))
      gq_list, gb_list = [], []
      for j, b in enumerate(abs_):
        gq, gb = term_grads(state.policy_params, state.q_params, b, jax.random.fold_in(key, it * 10 + j)); gq_list.append(gq); gb_list.append(gb)
      gq = jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *gq_list); gb = jax.tree_util.tree_map(lambda *x: sum(x) / len(x), *gb_list)
      neg = lambda t: jax.tree_util.tree_map(lambda x: -x, t)
      pq, cq = proj(neg(gq), d_dir); pb, cb = proj(neg(gb), d_dir); pt, ct = proj(neg(jax.tree_util.tree_map(lambda a, b: a + b, gq, gb)), d_dir)
      pp_before = state.policy_params
    state, metrics = multi_update(state, (MP._stack(cbs), MP._stack(abs_)))
    if log:
      dtheta = jax.tree_util.tree_map(lambda a, b: a - b, state.policy_params, pp_before)
      ps_, cs_ = proj(dtheta, d_dir)
      mode2, scale2, f_m2, f_c2 = (np.asarray(v) for v in probe(state.policy_params, state.q_params, jnp.asarray(O_v), jnp.asarray(C_v)))
      move = np.sum((mode2 - mode) * U, axis=1)
      d_now = np.linalg.norm(mode2 - C_v, axis=1)
      m = {k: float(v) for k, v in metrics.items()}
      rec.append({'update': int(step0) + (it + 1) * G, 'dist_mean': float(d_now.mean()), 'dist_change_mean': float((d_now - np.linalg.norm(mode - C_v, axis=1)).mean()),
                  'mode_move_toward_mean': float(move.mean()), 'share_moved_toward': float((move > 0).mean()), 'scale_median': float(np.median(scale2)),
                  'f_mode_mean': float(f_m2.mean()), 'f_cand_mean': float(f_c2.mean()), 'share_cand_above_mode': float((f_c2 > f_m2).mean()),
                  'critic_term_proj': pq, 'critic_term_cos': cq, 'bc_term_proj': pb, 'bc_term_cos': cb, 'total_grad_proj': pt, 'total_grad_cos': ct,
                  'actual_step_proj': ps_, 'actual_step_cos': cs_, 'actual_step_norm': float(jnp.linalg.norm(flat(dtheta))),
                  'train_bc_nll': m.get('bc_nll', float('nan')), 'train_q_term': m.get('actor_q_term', float('nan')), 'train_scale_median': m.get('policy_scale_median', float('nan'))})
      if it < 5 or (it + 1) % (LOG_EVERY * 4) == 0 or it == n_groups - 1:
        r = rec[-1]
        print(f'[{tag} upd {r["update"]:>6}] dist {r["dist_mean"]:.3f} (d {r["dist_change_mean"]:+.4f}) move-toward {r["mode_move_toward_mean"]:+.4f} ({r["share_moved_toward"]:.2f}) scale {r["scale_median"]:.3f} | '
              f'f(mode) {r["f_mode_mean"]:+.3f} f(c) {r["f_cand_mean"]:+.3f} c>mode {r["share_cand_above_mode"]:.2f} | cos toward: critic {r["critic_term_cos"]:+.3f} bc {r["bc_term_cos"]:+.3f} total {r["total_grad_cos"]:+.3f} step {r["actual_step_cos"]:+.3f} '
              f'({(it + 1) * G / (time.time() - t0):.1f} upd/s)', flush=True)
  m_end, sc_end, fm_end, fc_end = (np.asarray(v) for v in probe(state.policy_params, state.q_params, jnp.asarray(O_v), jnp.asarray(C_v)))
  d_end = np.linalg.norm(m_end - C_v, axis=1)
  R = np.array([(r['critic_term_cos'], r['bc_term_cos'], r['total_grad_cos'], r['actual_step_cos'], r['mode_move_toward_mean']) for r in rec])
  summ = {'seed': s, 'bc_coef': bc_used, 'updates': args.updates, 'n_contexts': len(V), 'contexts': [{k: v for k, v in c.items() if k != 'action'} for c in V],
          'dist_start_mean': float(d_start.mean()), 'dist_end_mean': float(d_end.mean()), 'dist_change_per_context': (d_end - d_start).round(4).tolist(),
          'share_contexts_closer_at_end': float((d_end < d_start).mean()), 'scale_start_median': float(np.median(sc0)), 'scale_end_median': float(np.median(sc_end)),
          'f_mode_start_mean': float(fm0.mean()), 'f_cand_start_mean': float(fc0.mean()), 'f_mode_end_mean': float(fm_end.mean()), 'f_cand_end_mean': float(fc_end.mean()),
          'share_cand_above_mode_start': float((fc0 > fm0).mean()), 'share_cand_above_mode_end': float((fc_end > fm_end).mean()),
          'share_logged_groups': {'critic_cos_positive': float((R[:, 0] > 0).mean()), 'bc_cos_positive': float((R[:, 1] > 0).mean()), 'total_cos_positive': float((R[:, 2] > 0).mean()), 'actual_step_cos_positive': float((R[:, 3] > 0).mean()), 'mode_moved_toward': float((R[:, 4] > 0).mean())},
          'mean_logged': {'critic_cos': float(R[:, 0].mean()), 'bc_cos': float(R[:, 1].mean()), 'total_cos': float(R[:, 2].mean()), 'actual_step_cos': float(R[:, 3].mean()), 'mode_move_toward': float(R[:, 4].mean())},
          'source_ckpt': str(src), 'source_sha256': MP.sha256(src), 'wall_seconds': time.time() - t0}
  MP.write_json(OUT / f'real_update_{tag}.json', {'summary': summ, 'records': rec})
  print(json.dumps(summ, indent=1, default=str), flush=True)


if __name__ == '__main__':
  main()
