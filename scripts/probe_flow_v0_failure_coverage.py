"""DEV PROBE: does the clean-only V0 Flow naturally cover the settled fatal
mode when placed at a real pre-death anchor?

DEVELOPMENT DATA ONLY -- the 16 ORIGINAL PILOT death episodes (the ones that
already fed the failure bank / settled-death reconstruction / env debugging).
The 39 fresh same-anchor held-out pairs, the 40 fresh death stream, and every
final Flow test artifact are NOT opened here.

Inputs (all pre-existing, validated artifacts):
  artifacts/same_anchor_candidate_probe/pairs_debug16_pilot.npz
      anchor_obs      s_i            (29-dim learner state at collapse_step)
      fatal_candidate s'_i,fatal     (N=80 physically settled fatal outcome;
                                      bitwise == the production settled bank)
      safe_candidate  s'_i,safe      (same-anchor forced-mask safe successor,
                                      ordinary one-step transition)
  artifacts/flow_v0_clean/flow_v0.pkl + norm_stats.npz   (FROZEN V0)

The Flow receives ONLY s_i. No fatal target, no failure label, no mask, no
severity, no _dead, no critic score, no policy action, no hidden U, and no
guidance of any kind enters sampling.

Primary statistic (normalized delta space, the space V0 was trained in, using
the FROZEN training normalization):
    d_fatal(K) = min_{k<=K} || norm(dhat_ik) - norm(ds_i,fatal) ||_2

Reported against three descriptive references (never used to tune anything):
  * d_safe(K): same statistic toward the ordinary safe successor -- the
    CONTROL that separates "Flow cannot model this region" from "Flow models
    the local safe support but misses the fatal mode";
  * d(safe -> fatal): how far the real safe outcome is from the real fatal
    outcome at the same anchor;
  * nearest-neighbour spread among the 16 REAL settled fatal deltas -- the
    natural width of the fatal mode itself.

Sampling uses the frozen V0 checkpoint, normalization, architecture and
fixed-step Euler solver (50 steps), identical to scripts/eval_flow_v0_clean.py.
For each seed a single K_MAX draw is taken and the K-grid is evaluated on
NESTED PREFIXES, so the coverage-vs-K curve is monotone by construction and
seed-to-seed variability is visible.

Usage:
  python scripts/probe_flow_v0_failure_coverage.py
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import make_net, OBS_DIM  # noqa: E402

FLOW_DIR = 'artifacts/flow_v0_clean'
PAIRS = 'artifacts/same_anchor_candidate_probe/pairs_debug16_pilot.npz'
OUT_DIR = 'artifacts/flow_v0_failure_coverage_dev16'
K_GRID = (32, 64, 128, 256)              # the specified grid (primary)
K_EXTRA = (512, 1024, 2048)              # supplementary: tail-mass check
ODE_STEPS = 50                           # frozen V0 sampler setting
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)


def phys(s):
  """Physical observables used in the settled-death analysis."""
  return {'x': float(s[0]), 'y': float(s[1]), 'torso_z': float(s[2]),
          'up_z': float(1.0 - 2.0 * (s[4] ** 2 + s[5] ** 2)),
          'v_xy': float(np.linalg.norm(s[15:17]))}


def qstats(a):
  a = np.asarray(a, float)
  return {'mean': float(a.mean()), 'median': float(np.median(a)),
          'min': float(a.min()), 'max': float(a.max()),
          'p10': float(np.percentile(a, 10)),
          'p90': float(np.percentile(a, 90))}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--flow-dir', default=FLOW_DIR)
  ap.add_argument('--pairs', default=PAIRS)
  ap.add_argument('--out', default=OUT_DIR)
  ap.add_argument('--ode-steps', type=int, default=ODE_STEPS)
  ap.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
  ap.add_argument('--secondary-critic', action='store_true', default=True,
                  help='record Critic-C scores as a SECONDARY diagnostic '
                       'only (never determines coverage success)')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  seeds = [int(s) for s in args.seeds.split(',')]
  assert args.ode_steps == ODE_STEPS, 'sampler must stay frozen at 50 steps'

  # guard: development pilot data only
  assert 'debug16_pilot' in args.pairs, 'this probe uses the 16 PILOT deaths'
  for bad in ('heldout40', 'deaths_extended'):
    assert bad not in args.pairs, 'reserved evaluation data must not be used'

  # ---- frozen V0 ----------------------------------------------------------
  with open(os.path.join(args.flow_dir, 'flow_v0.pkl'), 'rb') as f:
    ck = pickle.load(f)
  nrm = {k: (np.asarray(v, np.float32) if isinstance(v, list) else v)
         for k, v in ck['norm'].items()}
  net = make_net(tuple(ck['hidden']), OBS_DIM)
  params = ck['params']

  @jax.jit
  def _v(x, t, s):
    return net.apply(params, x, t, s)

  def sample(s_raw, K, key):
    """IDENTICAL to eval_flow_v0_clean.sample_next_states (frozen)."""
    n = s_raw.shape[0]
    s_n = (s_raw - nrm['state_mean']) / nrm['state_std']
    s_rep = jnp.asarray(np.repeat(s_n, K, axis=0))
    x = jax.random.normal(key, (n * K, OBS_DIM))
    dt = 1.0 / args.ode_steps
    for i in range(args.ode_steps):
      tt = jnp.full((n * K, 1), i * dt)
      x = x + dt * _v(x, tt, s_rep)
    dlt = np.asarray(x) * nrm['delta_std'] + nrm['delta_mean']
    return dlt.reshape(n, K, OBS_DIM)

  # ---- development anchors -------------------------------------------------
  d = np.load(args.pairs, allow_pickle=True)
  S = np.asarray(d['anchor_obs'], np.float32)          # [16, 29]
  Sf = np.asarray(d['fatal_candidate'], np.float32)
  Ss = np.asarray(d['safe_candidate'], np.float32)
  eps = np.asarray(d['episode_id'], np.int64)
  n = len(S)
  Df, Ds = Sf - S, Ss - S

  def nz(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']
  nDf, nDs = nz(Df), nz(Ds)

  # ---- references ----------------------------------------------------------
  d_safe_to_fatal = np.linalg.norm(nDf - nDs, axis=1)          # [16]
  pw = np.linalg.norm(nDf[:, None] - nDf[None], axis=2)
  np.fill_diagonal(pw, np.inf)
  fatal_nn = pw.min(axis=1)                                    # [16]
  fatal_pairwise = pw[np.isfinite(pw)]

  # ---- sampling + nested-prefix coverage ----------------------------------
  k_all = sorted(set(K_GRID) | set(K_EXTRA))
  k_max = max(k_all)
  t0 = time.time()
  dmin_f = {K: np.zeros((len(seeds), n)) for K in k_all}
  dmin_s = {K: np.zeros((len(seeds), n)) for K in k_all}
  best_cand = None
  best_d = np.full(n, np.inf)
  cloud = None
  for si, sd in enumerate(seeds):
    dlt = sample(S, k_max, jax.random.PRNGKey(sd))             # [16, kmax, 29]
    ndlt = nz(dlt)
    df = np.linalg.norm(ndlt - nDf[:, None], axis=2)           # [16, kmax]
    ds = np.linalg.norm(ndlt - nDs[:, None], axis=2)
    for K in k_all:
      dmin_f[K][si] = df[:, :K].min(axis=1)
      dmin_s[K][si] = ds[:, :K].min(axis=1)
    j = df.argmin(axis=1)
    cur = df[np.arange(n), j]
    imp = cur < best_d
    if best_cand is None:
      best_cand = S + dlt[np.arange(n), j]
      best_d = cur.copy()
    else:
      best_cand[imp] = (S + dlt[np.arange(n), j])[imp]
      best_d[imp] = cur[imp]
    if si == 0:
      cloud = S[:, None] + dlt[:, :256]                        # for the plot
  wall = time.time() - t0

  # ---- coverage-vs-K -------------------------------------------------------
  cov = {}
  for K in k_all:
    per_anchor = dmin_f[K].mean(axis=0)          # mean over seeds
    cov[K] = {
        'd_fatal_mean_over_seeds_per_anchor': per_anchor.tolist(),
        'd_fatal': qstats(dmin_f[K]),
        'd_safe_control': qstats(dmin_s[K]),
        'seed_spread_of_anchor_mean': float(dmin_f[K].mean(axis=1).std()),
    }

  # ---- physical comparison at the single best (nearest-to-fatal) candidate -
  rows = []
  for i in range(n):
    rows.append({
        'episode_id': int(eps[i]),
        'd_fatal_best_over_all_seeds': float(best_d[i]),
        'd_fatal_K256_meanseed': float(dmin_f[256][:, i].mean()),
        'd_safe_K256_meanseed': float(dmin_s[256][:, i].mean()),
        'd_safe_to_fatal': float(d_safe_to_fatal[i]),
        'fatal_mode_nn_spread': float(fatal_nn[i]),
        'nearest_flow': phys(best_cand[i]),
        'true_fatal': phys(Sf[i]),
        'safe_successor': phys(Ss[i]),
        'anchor': phys(S[i])})

  def agg(key, sub):
    return qstats([r[key][sub] for r in rows])

  physical = {feat: {'nearest_flow_candidate': agg('nearest_flow', feat),
                     'true_fatal': agg('true_fatal', feat),
                     'safe_successor': agg('safe_successor', feat),
                     'anchor': agg('anchor', feat)}
              for feat in ('torso_z', 'v_xy', 'up_z', 'x', 'y')}

  # ---- SECONDARY (non-determining) critic diagnostic ----------------------
  secondary = None
  if args.secondary_critic:
    from crl import networks as networks_mod
    from crl import checkpoint as ckpt_mod
    from crl.replay import obs_to_goal
    from verify_offline_d4rl import build_offline_cfg
    cfg = build_offline_cfg()
    nets = networks_mod.make_networks(
        obs_dim=OBS_DIM, goal_dim=OBS_DIM, action_dim=8,
        repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
        repr_norm_temp=cfg.repr_norm_temp,
        hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
        use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
    c_step, c_state = ckpt_mod.load_checkpoint(
        'failneg_settledbank_a01_s0_300k/'
        'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')

    @jax.jit
    def c_score(og, a, p=c_state.q_params):
      qv = nets.q_network.apply(p, og, a)
      return jnp.diagonal(qv, axis1=0, axis2=1).T

    a_rec = np.asarray(d['anchor_action'], np.float32)   # recorded action
    out = {}
    for name, cand in (('nearest_flow', best_cand), ('true_fatal', Sf),
                       ('safe_successor', Ss)):
      g = np.asarray(obs_to_goal(cand.astype(np.float32), 0, -1,
                                 tuple(range(OBS_DIM))), np.float32)
      og = np.concatenate([S, g], axis=1)
      f = np.asarray(c_score(jnp.asarray(og), jnp.asarray(a_rec)))
      out[name] = qstats(f.min(axis=1))
    secondary = {'critic_C_f_min': out, 'action': 'recorded anchor action',
                 'note': ('SECONDARY ONLY -- computed after the geometric/'
                          'physical analysis was fixed; does NOT define '
                          'coverage. A low critic score on an off-manifold '
                          'sample is not evidence of fatal-mode coverage.')}

  # ---- outputs -------------------------------------------------------------
  summary = {
      'probe': 'V0 clean-only Flow failure-coverage, 16 PILOT deaths (dev)',
      'flow': {'dir': args.flow_dir,
               'ckpt_sha256': C.sha256_file(
                   os.path.join(args.flow_dir, 'flow_v0.pkl')),
               'config': ck['config'], 'n_params': ck['n_params'],
               'ode_steps': args.ode_steps,
               'solver': 'fixed-step explicit Euler (frozen)'},
      'data': {'pairs': args.pairs,
               'pairs_sha256': C.sha256_file(args.pairs),
               'anchor_episode_ids': eps.tolist(),
               'n_anchors': n,
               'note': 'anchors + settled fatal + same-anchor safe successor '
                       'all taken from existing validated pilot artifacts; '
                       'no counterfactual world was regenerated'},
      'sampling': {'seeds': seeds, 'k_grid_primary': list(K_GRID),
                   'k_grid_supplementary': list(K_EXTRA),
                   'k_max_drawn_per_seed': k_max,
                   'nested_prefix_evaluation': True,
                   'total_candidates': int(len(seeds) * n * k_max),
                   'wall_sec': wall},
      'primary_metric': 'min_k || norm(delta_hat) - norm(delta_fatal) ||_2 '
                        'in the frozen V0 normalized delta space',
      'coverage_vs_K': {str(K): cov[K] for K in k_all},
      'references': {
          'd_safe_to_fatal': qstats(d_safe_to_fatal),
          'fatal_mode_nearest_neighbour_spread': qstats(fatal_nn),
          'fatal_mode_pairwise': qstats(fatal_pairwise),
          'note': 'descriptive only; never used to tune the Flow'},
      'physical_comparison': physical,
      'per_anchor': rows,
      'secondary_critic_diagnostic': secondary,
      'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'),
            indent=2)

  with open(os.path.join(args.out, 'per_anchor.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['episode_id', 'd_fatal_best', 'd_fatal_K256', 'd_safe_K256',
                'd_safe_to_fatal', 'fatal_nn_spread',
                'flow_z', 'fatal_z', 'safe_z',
                'flow_vxy', 'fatal_vxy', 'safe_vxy'])
    for r in rows:
      w.writerow([r['episode_id'], round(r['d_fatal_best_over_all_seeds'], 4),
                  round(r['d_fatal_K256_meanseed'], 4),
                  round(r['d_safe_K256_meanseed'], 4),
                  round(r['d_safe_to_fatal'], 4),
                  round(r['fatal_mode_nn_spread'], 4),
                  round(r['nearest_flow']['torso_z'], 4),
                  round(r['true_fatal']['torso_z'], 4),
                  round(r['safe_successor']['torso_z'], 4),
                  round(r['nearest_flow']['v_xy'], 4),
                  round(r['true_fatal']['v_xy'], 4),
                  round(r['safe_successor']['v_xy'], 4)])

  np.savez_compressed(
      os.path.join(args.out, 'nearest_candidates.npz'),
      anchor=S, true_fatal=Sf, safe_successor=Ss,
      nearest_flow_candidate=best_cand, d_fatal_best=best_d,
      episode_id=eps, seeds=np.asarray(seeds))

  # ---- plots ---------------------------------------------------------------
  ks = np.array(k_all)
  fmean = np.array([dmin_f[K].mean() for K in k_all])
  flo = np.array([dmin_f[K].mean(axis=1).min() for K in k_all])
  fhi = np.array([dmin_f[K].mean(axis=1).max() for K in k_all])
  smean = np.array([dmin_s[K].mean() for K in k_all])
  fig, ax = plt.subplots(figsize=(7.6, 5))
  ax.plot(ks, fmean, 'o-', color='crimson', label='d to SETTLED FATAL (mean)')
  ax.fill_between(ks, flo, fhi, color='crimson', alpha=0.18,
                  label='seed range')
  ax.plot(ks, smean, 's-', color='tab:blue',
          label='d to safe successor (control)')
  ax.axhline(np.median(d_safe_to_fatal), color='gray', ls='--', lw=1.2,
             label='real safe -> real fatal (median)')
  ax.axhline(np.median(fatal_nn), color='green', ls=':', lw=1.4,
             label='fatal-mode NN spread (median)')
  ax.axvline(256, color='k', lw=0.6, alpha=0.4)
  ax.set_xscale('log', base=2)
  ax.set_xlabel('K (candidates per anchor; nested prefixes)')
  ax.set_ylabel('min normalized delta distance')
  ax.set_title('V0 clean-only Flow: coverage of the settled fatal mode\n'
               '16 pilot death anchors, %d seeds' % len(seeds))
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'coverage_vs_K.png'), dpi=140)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(7.2, 5.2))
  cl = cloud.reshape(-1, OBS_DIM)
  ax.scatter(np.linalg.norm(cl[:, 15:17], axis=1), cl[:, 2], s=3, alpha=0.08,
             color='tab:gray', label='Flow candidates (seed 0, K=256)')
  ax.scatter([r['safe_successor']['v_xy'] for r in rows],
             [r['safe_successor']['torso_z'] for r in rows], s=55,
             marker='^', color='tab:blue', label='real safe successor')
  ax.scatter([r['nearest_flow']['v_xy'] for r in rows],
             [r['nearest_flow']['torso_z'] for r in rows], s=55,
             marker='o', facecolors='none', edgecolors='darkorange',
             linewidths=1.6, label='nearest Flow candidate to fatal')
  ax.scatter([r['true_fatal']['v_xy'] for r in rows],
             [r['true_fatal']['torso_z'] for r in rows], s=70, marker='X',
             color='crimson', label='real SETTLED FATAL')
  ax.set_xlabel('|v_xy| [m/s]')
  ax.set_ylabel('torso z')
  ax.set_title('Where Flow samples land vs the settled fatal mode')
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'physical_scatter.png'), dpi=140)
  plt.close(fig)

  # ---- console -------------------------------------------------------------
  print('anchors: %d pilot deaths | seeds %s | %d candidates in %.1fs'
        % (n, seeds, len(seeds) * n * k_max, wall))
  print('\ncoverage vs K (mean over %d seeds x %d anchors):' % (len(seeds), n))
  for K in k_all:
    tag = '' if K in K_GRID else '  (supplementary)'
    print('  K=%5d  d_fatal %.3f [%.3f, %.3f]   d_safe(control) %.3f%s'
          % (K, dmin_f[K].mean(), dmin_f[K].min(), dmin_f[K].max(),
             dmin_s[K].mean(), tag))
  print('\nreferences:')
  print('  real safe -> real fatal : median %.3f (min %.3f max %.3f)'
        % (np.median(d_safe_to_fatal), d_safe_to_fatal.min(),
           d_safe_to_fatal.max()))
  print('  fatal-mode NN spread    : median %.3f (min %.3f max %.3f)'
        % (np.median(fatal_nn), fatal_nn.min(), fatal_nn.max()))
  print('\nphysical (median over 16 anchors):')
  for feat in ('torso_z', 'v_xy', 'up_z'):
    p = physical[feat]
    print('  %-8s anchor %.3f | safe %.3f | nearest-flow %.3f | FATAL %.3f'
          % (feat, p['anchor']['median'], p['safe_successor']['median'],
             p['nearest_flow_candidate']['median'], p['true_fatal']['median']))
  if secondary:
    print('\n[secondary, non-determining] critic C f_min medians: '
          'flow %.2f | fatal %.2f | safe %.2f'
          % (secondary['critic_C_f_min']['nearest_flow']['median'],
             secondary['critic_C_f_min']['true_fatal']['median'],
             secondary['critic_C_f_min']['safe_successor']['median']))
  print('\nsaved -> %s' % args.out)


if __name__ == '__main__':
  main()
