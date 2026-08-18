"""V0 Flow Matching evaluation: clean held-out sanity checks + the
Policy-B -> Flow -> Critic-C interface check.

PLUMBING CHECK ONLY. Everything here runs on CLEAN VALIDATION trajectories
(the episode ids held out by scripts/train_flow_v0_clean.py). It never
touches the failure bank, the 39 held-out same-anchor pairs, the 40 fresh
death stream, or any settled-failure artifact, and it never scores fatal
coverage.

Provides sample_next_states(s, K): normalize s -> K Gaussian noises ->
fixed-step Euler integration of dx/dt = v_theta(x, t, s_norm) from t=0 to
t=1 -> denormalize the generated delta -> return s + delta.

Diagnostics (clean validation anchors only):
  * one-step displacement scale, generated vs real ||s_{t+1} - s||;
  * nearest-candidate error d_min = min_k ||delta_hat_k - delta_true|| in
    NORMALIZED delta space, with a marginal-shuffle reference (deltas drawn
    from the empirical marginal instead of the conditional model) so the
    number is interpretable -- reference only, not a scientific claim;
  * candidate diversity (mean pairwise distance among the K candidates);
  * numerical sanity (NaN/inf, max |coord|, max |delta| vs data ranges).

Interface check: a_B = pi_B(s, g) with the deployed legacy-bank alpha=0.1
policy (deterministic tanh(loc) eval convention, g = the episode task goal
already encoded in the 58-dim observation), candidates from the flow, and
scores f_C(s, a_B, s'_k) from the settled-bank alpha=0.1 critic with the
candidate in the GOAL SLOT via crl.replay.obs_to_goal. The critic is NEVER
used to train or guide the flow, and argmin_k f_C here is only a numerical
plausibility check -- not a causal worst case.

Usage:
  python scripts/eval_flow_v0_clean.py [--flow-dir artifacts/flow_v0_clean]
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import networks as networks_mod   # noqa: E402
from crl import checkpoint as ckpt_mod     # noqa: E402
from crl.replay import obs_to_goal         # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402
from train_flow_v0_clean import make_net, CLEAN_NPZ, OBS_DIM  # noqa: E402

FLOW_DIR = 'artifacts/flow_v0_clean'
B_CKPT = 'failneg_clean_p30_h800_resetfix_a01_s0_300k/best.pkl'
C_CKPT = ('failneg_settledbank_a01_s0_300k/'
          'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--flow-dir', default=FLOW_DIR)
  ap.add_argument('--npz', default=CLEAN_NPZ)
  ap.add_argument('--b-ckpt', default=B_CKPT)
  ap.add_argument('--c-ckpt', default=C_CKPT)
  ap.add_argument('--k', type=int, default=32, help='candidates per anchor')
  ap.add_argument('--ode-steps', type=int, default=50)
  ap.add_argument('--n-anchors', type=int, default=256)
  ap.add_argument('--n-interface-anchors', type=int, default=64)
  ap.add_argument('--n-fixed-anchors', type=int, default=8,
                  help='anchors whose raw candidates are saved verbatim')
  ap.add_argument('--seed', type=int, default=1234)
  args = ap.parse_args()
  rng = np.random.default_rng(args.seed)

  with open(os.path.join(args.flow_dir, 'flow_v0.pkl'), 'rb') as f:
    ck = pickle.load(f)
  split = json.load(open(os.path.join(args.flow_dir,
                                      'split_manifest.json')))
  nrm = {k: (np.asarray(v, np.float32) if isinstance(v, list) else v)
         for k, v in ck['norm'].items()}
  net = make_net(tuple(ck['hidden']), OBS_DIM)
  params = ck['params']

  # ---- clean VALIDATION anchors only --------------------------------------
  with np.load(args.npz, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    lengths = np.asarray(d['lengths'], np.int64)
  val_eps = np.asarray(split['val_episode_ids'], np.int64)
  assert not (set(val_eps.tolist())
              & set(split['train_episode_ids'])), 'split leak'
  anchors58, next58 = [], []
  for e in val_eps:
    t = rng.integers(0, lengths[e] - 1, size=8)
    anchors58.append(obs[e, t])
    next58.append(obs[e, t + 1])
  anchors58 = np.concatenate(anchors58)
  next58 = np.concatenate(next58)
  sel = rng.permutation(len(anchors58))[:args.n_anchors]
  anchors58, next58 = anchors58[sel], next58[sel]
  S = anchors58[:, :OBS_DIM]
  S_next = next58[:, :OBS_DIM]
  D_true = S_next - S

  # ---- sampler -------------------------------------------------------------
  @jax.jit
  def _v(x, t, s):
    return net.apply(params, x, t, s)

  def sample_next_states(s_raw, K, ode_steps, key):
    """s_raw [N, 29] -> candidates [N, K, 29] (see module docstring)."""
    n = s_raw.shape[0]
    s_n = (s_raw - nrm['state_mean']) / nrm['state_std']
    s_rep = jnp.asarray(np.repeat(s_n, K, axis=0))          # [N*K, 29]
    x = jax.random.normal(key, (n * K, OBS_DIM))
    dt = 1.0 / ode_steps
    for i in range(ode_steps):                              # fixed-step Euler
      tt = jnp.full((n * K, 1), i * dt)
      x = x + dt * _v(x, tt, s_rep)
    dlt = np.asarray(x) * nrm['delta_std'] + nrm['delta_mean']
    dlt = dlt.reshape(n, K, OBS_DIM)
    return np.repeat(s_raw[:, None], K, axis=1) + dlt, dlt

  key = jax.random.PRNGKey(args.seed)
  key, ks = jax.random.split(key)
  t0 = time.time()
  cand, dlt = sample_next_states(S, args.k, args.ode_steps, ks)
  wall = time.time() - t0
  latency = {'n_anchors': int(len(S)), 'k': args.k,
             'ode_steps': args.ode_steps,
             'total_sec': wall,
             'sec_per_anchor_of_k_candidates': wall / len(S),
             'candidates_per_sec': float(len(S) * args.k / wall),
             'solver': 'fixed-step explicit Euler'}

  # ---- diagnostics ---------------------------------------------------------
  finite = bool(np.isfinite(cand).all())
  disp_gen = np.linalg.norm(dlt, axis=2)          # [N, K]
  disp_true = np.linalg.norm(D_true, axis=1)      # [N]
  # normalized delta space (mean-centred + scaled, same map for both sides)
  dn = (dlt - nrm['delta_mean']) / nrm['delta_std']
  dtn = (D_true - nrm['delta_mean']) / nrm['delta_std']
  dmin = np.linalg.norm(dn - dtn[:, None], axis=2).min(axis=1)
  # marginal-shuffle reference: K deltas drawn from the empirical marginal
  ridx = rng.integers(0, len(D_true), size=(len(D_true), args.k))
  dmarg = (D_true[ridx] - nrm['delta_mean']) / nrm['delta_std']
  dmin_marg = np.linalg.norm(dmarg - dtn[:, None], axis=2).min(axis=1)
  # diversity: mean pairwise distance among candidates (normalized space)
  diff = dn[:, :, None, :] - dn[:, None, :, :]
  pair = np.linalg.norm(diff, axis=3)
  iu = np.triu_indices(args.k, k=1)
  diversity = pair[:, iu[0], iu[1]].mean(axis=1)

  def q(a):
    return {'mean': float(np.mean(a)), 'median': float(np.median(a)),
            'p10': float(np.percentile(a, 10)),
            'p90': float(np.percentile(a, 90)),
            'max': float(np.max(a))}

  diag = {
      'n_anchors': int(len(S)), 'k': args.k,
      'all_finite': finite,
      'displacement_norm_generated': q(disp_gen),
      'displacement_norm_real': q(disp_true),
      'nearest_candidate_error_normalized': q(dmin),
      'nearest_candidate_error_marginal_reference': q(dmin_marg),
      'reference_note': ('marginal reference = K deltas resampled from the '
                         'empirical marginal delta distribution instead of '
                         'the conditional flow; context only'),
      'candidate_diversity_mean_pairwise_normalized': q(diversity),
      'numerical': {
          'max_abs_candidate_coord': float(np.abs(cand).max()),
          'max_abs_real_state_coord': float(np.abs(S_next).max()),
          'max_abs_generated_delta': float(np.abs(dlt).max()),
          'max_abs_real_delta': float(np.abs(D_true).max()),
          'n_nonfinite': int((~np.isfinite(cand)).sum())},
      'sampling': latency}

  # ---- Policy B -> Flow -> Critic C interface ------------------------------
  cfg = build_offline_cfg()
  nets = networks_mod.make_networks(
      obs_dim=OBS_DIM, goal_dim=OBS_DIM, action_dim=8,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
  b_step, b_state = ckpt_mod.load_checkpoint(args.b_ckpt)
  c_step, c_state = ckpt_mod.load_checkpoint(args.c_ckpt)

  @jax.jit
  def b_act(o58, p=b_state.policy_params):
    return jnp.tanh(nets.policy_network.apply(p, o58).loc)

  @jax.jit
  def c_score(og, a, p=c_state.q_params):
    qv = nets.q_network.apply(p, og, a)
    return jnp.diagonal(qv, axis1=0, axis2=1).T          # [B, 2]

  m = args.n_interface_anchors
  o58 = anchors58[:m]
  aB = np.asarray(b_act(jnp.asarray(o58)), np.float32)
  cand_i = cand[:m]
  F = np.zeros((m, args.k, 2), np.float32)
  for k in range(args.k):
    g = obs_to_goal(cand_i[:, k, :].astype(np.float32), 0, -1,
                    tuple(range(OBS_DIM)))
    og = np.concatenate([o58[:, :OBS_DIM], np.asarray(g, np.float32)], 1)
    F[:, k] = np.asarray(c_score(jnp.asarray(og), jnp.asarray(aB)))
  Fmin = F.min(axis=2)                                    # twin-min per cand
  kstar = Fmin.argmin(axis=1)
  chosen = cand_i[np.arange(m), kstar]
  chosen_delta = chosen - o58[:, :OBS_DIM]
  iface = {
      'b_ckpt': args.b_ckpt, 'b_step': int(b_step),
      'c_ckpt': args.c_ckpt, 'c_step': int(c_step),
      'n_anchors': m, 'k': args.k,
      'all_scores_finite': bool(np.isfinite(F).all()),
      'f1_finite': bool(np.isfinite(F[..., 0]).all()),
      'f2_finite': bool(np.isfinite(F[..., 1]).all()),
      'score_range_fmin': [float(Fmin.min()), float(Fmin.max())],
      'score_mean_fmin': float(Fmin.mean()),
      'score_std_fmin': float(Fmin.std()),
      'within_anchor_score_spread_fmin': q(Fmin.max(1) - Fmin.min(1)),
      'f1_stats': {'mean': float(F[..., 0].mean()),
                   'range': [float(F[..., 0].min()),
                             float(F[..., 0].max())]},
      'f2_stats': {'mean': float(F[..., 1].mean()),
                   'range': [float(F[..., 1].min()),
                             float(F[..., 1].max())]},
      'argmin_candidate': {
          'delta_norm': q(np.linalg.norm(chosen_delta, axis=1)),
          'torso_z': q(chosen[:, 2]),
          'all_finite': bool(np.isfinite(chosen).all())},
      'real_next_state_reference': {
          'torso_z': q(S_next[:m, 2]),
          'delta_norm': q(np.linalg.norm(D_true[:m], axis=1))},
      'note': ('interface/numerical check only -- argmin_k f_C over V0 '
               'candidates is NOT a causal worst case; the critic never '
               'trains or guides the flow')}

  # ---- fixed anchor set: raw candidates saved verbatim ---------------------
  nf = args.n_fixed_anchors
  np.savez_compressed(
      os.path.join(args.flow_dir, 'fixed_anchor_samples.npz'),
      anchor_obs58=anchors58[:nf], real_next_state=S_next[:nf],
      candidates=cand[:nf], candidate_deltas=dlt[:nf],
      real_delta=D_true[:nf], policy_B_action=aB[:nf],
      critic_C_scores_twin=F[:nf], critic_C_fmin=Fmin[:nf],
      argmin_index=kstar[:nf],
      meta=json.dumps({'k': args.k, 'ode_steps': args.ode_steps,
                       'source': 'clean validation episodes only'}))

  summary = {'stage': 'flow V0 (clean factual, plumbing only)',
             'flow_dir': args.flow_dir,
             'flow_config': ck['config'], 'n_params': ck['n_params'],
             'split': {k: split[k] for k in
                       ('n_train_episodes', 'n_val_episodes',
                        'n_train_transitions', 'n_val_transitions',
                        'split_level', 'seed')},
             'clean_diagnostics': diag,
             'interface_B_flow_C': iface,
             'scope_note': ('V0 learns q(s\'|s) from clean factual adjacent '
                            'transitions only; it has no reason to generate '
                            'fatal branches and no causal claim is made.')}
  json.dump(summary, open(os.path.join(args.flow_dir,
                                       'v0_summary.json'), 'w'), indent=2)

  print('== clean held-out diagnostics ({} anchors, K={}) =='.format(
      len(S), args.k))
  print('  finite: {} | max |coord| gen {:.2f} vs real {:.2f}'.format(
      finite, diag['numerical']['max_abs_candidate_coord'],
      diag['numerical']['max_abs_real_state_coord']))
  print('  ||delta|| generated median {:.4f} (p10 {:.4f} p90 {:.4f})'.format(
      diag['displacement_norm_generated']['median'],
      diag['displacement_norm_generated']['p10'],
      diag['displacement_norm_generated']['p90']))
  print('  ||delta|| real      median {:.4f} (p10 {:.4f} p90 {:.4f})'.format(
      diag['displacement_norm_real']['median'],
      diag['displacement_norm_real']['p10'],
      diag['displacement_norm_real']['p90']))
  print('  nearest-candidate err (norm) median {:.3f} | marginal ref '
        '{:.3f}'.format(diag['nearest_candidate_error_normalized']['median'],
                        diag['nearest_candidate_error_marginal_reference']
                        ['median']))
  print('  diversity (mean pairwise, norm) median {:.3f}'.format(
      diag['candidate_diversity_mean_pairwise_normalized']['median']))
  print('  sampling: {:.3f}s for {}x{} candidates ({:.0f} cand/s, {} Euler '
        'steps)'.format(wall, len(S), args.k,
                        latency['candidates_per_sec'], args.ode_steps))
  print('\n== Policy B -> Flow -> Critic C ==')
  print('  scores finite: all {} (f1 {}, f2 {})'.format(
      iface['all_scores_finite'], iface['f1_finite'], iface['f2_finite']))
  print('  f_min range [{:.2f}, {:.2f}] mean {:.2f} std {:.2f}'.format(
      iface['score_range_fmin'][0], iface['score_range_fmin'][1],
      iface['score_mean_fmin'], iface['score_std_fmin']))
  print('  within-anchor spread median {:.3f}'.format(
      iface['within_anchor_score_spread_fmin']['median']))
  print('  argmin candidate: ||delta|| median {:.4f}, torso z median '
        '{:.3f} (real z median {:.3f})'.format(
            iface['argmin_candidate']['delta_norm']['median'],
            iface['argmin_candidate']['torso_z']['median'],
            iface['real_next_state_reference']['torso_z']['median']))
  print('\nsaved v0_summary.json + fixed_anchor_samples.npz -> '
        + args.flow_dir)


if __name__ == '__main__':
  main()
