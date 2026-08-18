"""V0 vs V0.5 controlled comparison: q(s'|s) against q(s'|s,a).

Runs BOTH models through identical code paths on identical data so every
difference is attributable to the conditioning set:

  (1) clean-validation metrics on the SAME anchors (same rng seed and
      selection procedure as scripts/eval_flow_v0_clean.py): FM-space
      nearest-candidate error, candidate diversity, delta scale/tail,
      sampling speed;
  (2) dev16 failure coverage on the SAME 16 PILOT development death anchors,
      same K-grid and same 8 sampling seeds as
      scripts/probe_flow_v0_failure_coverage.py, plus the same-anchor SAFE
      successor control. The recomputed V0 numbers are cross-checked against
      the frozen V0 probe artifact to prove no drift;
  (3) the Policy-B -> Flow -> Critic-C interface with V0.5 (interface/
      numerical check only; the critic never trains or guides the flow and
      never defines coverage).

The 39 fresh held-out same-anchor pairs and the 40 fresh death stream are
NOT opened.

Usage:
  python scripts/eval_flow_v05_clean_action.py
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
from train_flow_v05_clean_action import make_net_a  # noqa: E402
from probe_flow_v0_failure_coverage import phys, qstats  # noqa: E402

V0_DIR = 'artifacts/flow_v0_clean'
V05_DIR = 'artifacts/flow_v05_clean_action'
V0_COV = 'artifacts/flow_v0_failure_coverage_dev16/summary.json'
PAIRS16 = 'artifacts/same_anchor_candidate_probe/pairs_debug16_pilot.npz'
B_CKPT = 'failneg_clean_p30_h800_resetfix_a01_s0_300k/best.pkl'
C_CKPT = ('failneg_settledbank_a01_s0_300k/'
          'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')
ODE_STEPS = 50
K_ALL = (32, 64, 128, 256, 512, 1024, 2048)
DEV_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)   # identical to the V0 probe


def load_models(v0_dir, v05_dir):
  with open(os.path.join(v0_dir, 'flow_v0.pkl'), 'rb') as f:
    c0 = pickle.load(f)
  with open(os.path.join(v05_dir, 'flow_v05.pkl'), 'rb') as f:
    c1 = pickle.load(f)
  nrm = {k: np.asarray(v, np.float32) for k, v in c0['norm'].items()
         if k in ('state_mean', 'state_std', 'delta_mean', 'delta_std')}
  n1 = {k: np.asarray(v, np.float32) for k, v in c1['norm'].items()}
  for k in nrm:
    assert np.array_equal(nrm[k], n1[k]), 'V0.5 must reuse V0 normalization'
  net0 = make_net(tuple(c0['hidden']), OBS_DIM)
  net1 = make_net_a(tuple(c1['hidden']), OBS_DIM)

  @jax.jit
  def v0(x, t, s):
    return net0.apply(c0['params'], x, t, s)

  @jax.jit
  def v1(x, t, s, a):
    return net1.apply(c1['params'], x, t, s, a)

  def sample(s_raw, K, key, act=None):
    """Frozen V0 sampler; `act` present => the V0.5 conditional."""
    n = s_raw.shape[0]
    s_n = (s_raw - nrm['state_mean']) / nrm['state_std']
    s_rep = jnp.asarray(np.repeat(s_n, K, axis=0))
    a_rep = None if act is None else jnp.asarray(np.repeat(act, K, axis=0))
    x = jax.random.normal(key, (n * K, OBS_DIM))
    dt = 1.0 / ODE_STEPS
    for i in range(ODE_STEPS):
      tt = jnp.full((n * K, 1), i * dt)
      x = x + dt * (v0(x, tt, s_rep) if a_rep is None
                    else v1(x, tt, s_rep, a_rep))
    dlt = np.asarray(x) * nrm['delta_std'] + nrm['delta_mean']
    return dlt.reshape(n, K, OBS_DIM)

  return c0, c1, nrm, sample


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--v0-dir', default=V0_DIR)
  ap.add_argument('--v05-dir', default=V05_DIR)
  ap.add_argument('--out', default=V05_DIR)
  ap.add_argument('--k', type=int, default=32)
  ap.add_argument('--n-anchors', type=int, default=256)
  ap.add_argument('--n-interface-anchors', type=int, default=64)
  ap.add_argument('--seed', type=int, default=1234)   # == V0 clean eval seed
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  c0, c1, nrm, sample = load_models(args.v0_dir, args.v05_dir)

  def nzd(x):
    return (x - nrm['delta_mean']) / nrm['delta_std']

  # ================= (1) clean validation, identical anchors ===============
  split = json.load(open(os.path.join(args.v0_dir, 'split_manifest.json')))
  rng = np.random.default_rng(args.seed)         # same seed/procedure as V0
  with np.load(split['npz'], allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    lengths = np.asarray(d['lengths'], np.int64)
  val_eps = np.asarray(split['val_episode_ids'], np.int64)
  A58, N58, ACT = [], [], []
  for e in val_eps:
    t = rng.integers(0, lengths[e] - 1, size=8)
    A58.append(obs[e, t])
    N58.append(obs[e, t + 1])
    ACT.append(act[e, t])
  A58, N58, ACT = (np.concatenate(A58), np.concatenate(N58),
                   np.concatenate(ACT))
  sel = rng.permutation(len(A58))[:args.n_anchors]
  A58, N58, ACT = A58[sel], N58[sel], ACT[sel]
  S, Snext = A58[:, :OBS_DIM], N58[:, :OBS_DIM]
  Dtrue = Snext - S
  nDtrue = nzd(Dtrue)

  clean = {}
  cand_store = {}
  for tag, use_a in (('V0', False), ('V0.5', True)):
    key = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    dlt = sample(S, args.k, key, ACT if use_a else None)
    wall = time.time() - t0
    cand_store[tag] = dlt
    nd = nzd(dlt)
    dmin = np.linalg.norm(nd - nDtrue[:, None], axis=2).min(axis=1)
    diff = nd[:, :, None, :] - nd[:, None, :, :]
    iu = np.triu_indices(args.k, k=1)
    div = np.linalg.norm(diff, axis=3)[:, iu[0], iu[1]].mean(axis=1)
    dnorm = np.linalg.norm(dlt, axis=2)
    clean[tag] = {
        'nearest_candidate_error_normalized': qstats(dmin),
        'candidate_diversity_mean_pairwise_normalized': qstats(div),
        'displacement_norm_generated': qstats(dnorm),
        'max_abs_generated_delta': float(np.abs(dlt).max()),
        'max_abs_candidate_coord': float(np.abs(S[:, None] + dlt).max()),
        'n_nonfinite': int((~np.isfinite(dlt)).sum()),
        'sampling': {'sec': wall, 'candidates_per_sec':
                     float(len(S) * args.k / wall), 'ode_steps': ODE_STEPS}}
  clean['real_reference'] = {
      'displacement_norm_real': qstats(np.linalg.norm(Dtrue, axis=1)),
      'max_abs_real_delta': float(np.abs(Dtrue).max()),
      'max_abs_real_state_coord': float(np.abs(Snext).max())}
  clean['n_anchors'] = int(len(S))
  clean['k'] = args.k

  # ================= (2) dev16 failure coverage ============================
  p = np.load(PAIRS16, allow_pickle=True)
  S16 = np.asarray(p['anchor_obs'], np.float32)
  A16 = np.asarray(p['anchor_action'], np.float32)
  Sf = np.asarray(p['fatal_candidate'], np.float32)
  Ss = np.asarray(p['safe_candidate'], np.float32)
  eps16 = np.asarray(p['episode_id'], np.int64)
  nDf, nDs = nzd(Sf - S16), nzd(Ss - S16)
  d_safe_to_fatal = np.linalg.norm(nDf - nDs, axis=1)
  pw = np.linalg.norm(nDf[:, None] - nDf[None], axis=2)
  np.fill_diagonal(pw, np.inf)
  fatal_nn = pw.min(axis=1)
  kmax = max(K_ALL)

  dev = {}
  best = {}
  for tag, use_a in (('V0', False), ('V0.5', True)):
    df_k = {K: np.zeros((len(DEV_SEEDS), len(S16))) for K in K_ALL}
    ds_k = {K: np.zeros((len(DEV_SEEDS), len(S16))) for K in K_ALL}
    bd = np.full(len(S16), np.inf)
    bc = np.zeros_like(S16)
    for si, sd in enumerate(DEV_SEEDS):
      dlt = sample(S16, kmax, jax.random.PRNGKey(sd),
                   A16 if use_a else None)
      nd = nzd(dlt)
      df = np.linalg.norm(nd - nDf[:, None], axis=2)
      ds = np.linalg.norm(nd - nDs[:, None], axis=2)
      for K in K_ALL:
        df_k[K][si] = df[:, :K].min(axis=1)
        ds_k[K][si] = ds[:, :K].min(axis=1)
      j = df.argmin(axis=1)
      cur = df[np.arange(len(S16)), j]
      imp = cur < bd
      bc[imp] = (S16 + dlt[np.arange(len(S16)), j])[imp]
      bd[imp] = cur[imp]
    dev[tag] = {str(K): {'d_fatal': qstats(df_k[K]),
                         'd_safe_control': qstats(ds_k[K])} for K in K_ALL}
    dev[tag]['best_d_fatal_any_seed'] = qstats(bd)
    dev[tag]['n_anchors_inside_fatal_mode_spread'] = int((bd < fatal_nn).sum())
    best[tag] = (bc, bd)

  # cross-check the recomputed V0 numbers against the frozen probe artifact
  frozen = json.load(open(V0_COV))
  drift = {K: abs(dev['V0'][str(K)]['d_fatal']['mean']
                  - frozen['coverage_vs_K'][str(K)]['d_fatal']['mean'])
           for K in K_ALL}
  dev['v0_recompute_max_drift_vs_frozen_artifact'] = float(max(drift.values()))
  dev['references'] = {'d_safe_to_fatal': qstats(d_safe_to_fatal),
                       'fatal_mode_nn_spread': qstats(fatal_nn)}

  phys_cmp = {}
  for feat in ('torso_z', 'v_xy', 'up_z'):
    phys_cmp[feat] = {
        'true_fatal': qstats([phys(x)[feat] for x in Sf]),
        'safe_successor': qstats([phys(x)[feat] for x in Ss]),
        'V0_nearest': qstats([phys(x)[feat] for x in best['V0'][0]]),
        'V0.5_nearest': qstats([phys(x)[feat] for x in best['V0.5'][0]])}

  # ================= (3) Policy-B -> V0.5 Flow -> Critic-C =================
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
  b_step, b_state = ckpt_mod.load_checkpoint(B_CKPT)
  c_step, c_state = ckpt_mod.load_checkpoint(C_CKPT)

  @jax.jit
  def b_act(o58, pp=b_state.policy_params):
    return jnp.tanh(nets.policy_network.apply(pp, o58).loc)

  @jax.jit
  def c_score(og, a, pp=c_state.q_params):
    qv = nets.q_network.apply(pp, og, a)
    return jnp.diagonal(qv, axis1=0, axis2=1).T

  m = args.n_interface_anchors
  o58 = A58[:m]
  aB = np.asarray(b_act(jnp.asarray(o58)), np.float32)
  dlt_b = sample(o58[:, :OBS_DIM], args.k, jax.random.PRNGKey(args.seed + 5),
                 aB)                                   # a_B drives the flow
  cnd = o58[:, None, :OBS_DIM] + dlt_b
  F = np.zeros((m, args.k, 2), np.float32)
  for k in range(args.k):
    g = np.asarray(obs_to_goal(cnd[:, k].astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    og = np.concatenate([o58[:, :OBS_DIM], g], axis=1)
    F[:, k] = np.asarray(c_score(jnp.asarray(og), jnp.asarray(aB)))
  Fmin = F.min(axis=2)
  kstar = Fmin.argmin(axis=1)
  chosen = cnd[np.arange(m), kstar]
  iface = {'b_ckpt': B_CKPT, 'b_step': int(b_step), 'c_ckpt': C_CKPT,
           'c_step': int(c_step), 'n_anchors': m, 'k': args.k,
           'flow_conditioned_on': 'a_B (the deployed policy action)',
           'all_scores_finite': bool(np.isfinite(F).all()),
           'f1_finite': bool(np.isfinite(F[..., 0]).all()),
           'f2_finite': bool(np.isfinite(F[..., 1]).all()),
           'score_range_fmin': [float(Fmin.min()), float(Fmin.max())],
           'score_mean_fmin': float(Fmin.mean()),
           'score_std_fmin': float(Fmin.std()),
           'within_anchor_spread_fmin': qstats(Fmin.max(1) - Fmin.min(1)),
           'argmin_candidate': {
               'delta_norm': qstats(np.linalg.norm(
                   chosen - o58[:, :OBS_DIM], axis=1)),
               'torso_z': qstats(chosen[:, 2]),
               'all_finite': bool(np.isfinite(chosen).all())},
           'note': 'interface/numerical check only; not a causal worst case'}

  # ================= outputs ===============================================
  summary = {
      'comparison': 'V0 q(ds|s) vs V0.5 q(ds|s,a), clean-only, same split, '
                    'same normalization, same objective/optimizer/sampler',
      'models': {
          'V0': {'dir': args.v0_dir, 'n_params': c0['n_params'],
                 'config': c0['config']},
          'V0.5': {'dir': args.v05_dir, 'n_params': c1['n_params'],
                   'config': c1['config'],
                   'action_transform': c1['action_transform']}},
      'clean_validation': clean,
      'dev16_coverage': dev,
      'dev16_physical': phys_cmp,
      'interface_B_flow_C_v05': iface,
      'protected': ('39 held-out same-anchor pairs and the 40 fresh death '
                    'stream were NOT opened'),
      'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out, 'v05_vs_v0_summary.json'),
                          'w'), indent=2)

  with open(os.path.join(args.out, 'dev16_coverage.csv'), 'w',
            newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['episode_id', 'V0_best_d_fatal', 'V05_best_d_fatal',
                'd_safe_to_fatal', 'fatal_nn_spread',
                'V0_z', 'V05_z', 'fatal_z', 'V0_vxy', 'V05_vxy', 'fatal_vxy'])
    for i in range(len(S16)):
      w.writerow([int(eps16[i]), round(float(best['V0'][1][i]), 4),
                  round(float(best['V0.5'][1][i]), 4),
                  round(float(d_safe_to_fatal[i]), 4),
                  round(float(fatal_nn[i]), 4),
                  round(phys(best['V0'][0][i])['torso_z'], 4),
                  round(phys(best['V0.5'][0][i])['torso_z'], 4),
                  round(phys(Sf[i])['torso_z'], 4),
                  round(phys(best['V0'][0][i])['v_xy'], 4),
                  round(phys(best['V0.5'][0][i])['v_xy'], 4),
                  round(phys(Sf[i])['v_xy'], 4)])

  ks = np.array(K_ALL)
  fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
  for tag, col in (('V0', 'tab:purple'), ('V0.5', 'crimson')):
    ax[0].plot(ks, [dev[tag][str(K)]['d_fatal']['mean'] for K in K_ALL],
               'o-', color=col, label=tag + ' -> fatal')
    ax[0].plot(ks, [dev[tag][str(K)]['d_safe_control']['mean'] for K in K_ALL],
               's--', color=col, alpha=0.55, label=tag + ' -> safe (control)')
  ax[0].axhline(np.median(d_safe_to_fatal), color='gray', ls='--', lw=1.1,
                label='real safe -> real fatal')
  ax[0].axhline(np.median(fatal_nn), color='green', ls=':', lw=1.3,
                label='fatal-mode NN spread')
  ax[0].set_xscale('log', base=2)
  ax[0].set_xlabel('K')
  ax[0].set_ylabel('min normalized delta distance')
  ax[0].set_title('dev16 coverage: action conditioning changes nothing\n'
                  'about the fatal mode')
  ax[0].legend(fontsize=7)
  b = np.arange(2)
  ax[1].bar(b - 0.2, [clean['V0']['nearest_candidate_error_normalized']['median'],
                      clean['V0']['candidate_diversity_mean_pairwise_normalized']['median']],
            0.4, label='V0', color='tab:purple')
  ax[1].bar(b + 0.2, [clean['V0.5']['nearest_candidate_error_normalized']['median'],
                      clean['V0.5']['candidate_diversity_mean_pairwise_normalized']['median']],
            0.4, label='V0.5', color='crimson')
  ax[1].set_xticks(b)
  ax[1].set_xticklabels(['nearest-candidate err', 'diversity'])
  ax[1].set_title('clean validation (median, %d anchors, K=%d)'
                  % (clean['n_anchors'], args.k))
  ax[1].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'v05_vs_v0.png'), dpi=140)
  plt.close(fig)

  # ---- console -------------------------------------------------------------
  print('== clean validation (%d anchors, K=%d) ==' % (clean['n_anchors'],
                                                       args.k))
  for tag in ('V0', 'V0.5'):
    c = clean[tag]
    print('  %-5s dmin %.3f | diversity %.3f | |d| med %.3f | max|d| %.2f '
          '| %.0f cand/s'
          % (tag, c['nearest_candidate_error_normalized']['median'],
             c['candidate_diversity_mean_pairwise_normalized']['median'],
             c['displacement_norm_generated']['median'],
             c['max_abs_generated_delta'],
             c['sampling']['candidates_per_sec']))
  print('  real  |d| med %.3f | max|d| %.2f'
        % (clean['real_reference']['displacement_norm_real']['median'],
           clean['real_reference']['max_abs_real_delta']))
  print('\n== dev16 coverage (16 pilot deaths, %d seeds) ==' % len(DEV_SEEDS))
  print('  V0 recompute drift vs frozen artifact: %.4f'
        % dev['v0_recompute_max_drift_vs_frozen_artifact'])
  for K in K_ALL:
    print('  K=%5d  fatal: V0 %.3f  V0.5 %.3f   |  safe: V0 %.3f  V0.5 %.3f'
          % (K, dev['V0'][str(K)]['d_fatal']['mean'],
             dev['V0.5'][str(K)]['d_fatal']['mean'],
             dev['V0'][str(K)]['d_safe_control']['mean'],
             dev['V0.5'][str(K)]['d_safe_control']['mean']))
  print('  refs: safe->fatal %.3f | fatal-mode NN spread %.3f'
        % (np.median(d_safe_to_fatal), np.median(fatal_nn)))
  print('  anchors inside fatal-mode spread: V0 %d/16, V0.5 %d/16'
        % (dev['V0']['n_anchors_inside_fatal_mode_spread'],
           dev['V0.5']['n_anchors_inside_fatal_mode_spread']))
  print('\n== physical (median) ==')
  for feat in ('torso_z', 'v_xy'):
    p_ = phys_cmp[feat]
    print('  %-8s fatal %.3f | safe %.3f | V0 %.3f | V0.5 %.3f'
          % (feat, p_['true_fatal']['median'], p_['safe_successor']['median'],
             p_['V0_nearest']['median'], p_['V0.5_nearest']['median']))
  print('\n== B -> V0.5 Flow -> C interface ==')
  print('  finite %s | f_min [%.2f, %.2f] mean %.2f | spread med %.2f'
        % (iface['all_scores_finite'], iface['score_range_fmin'][0],
           iface['score_range_fmin'][1], iface['score_mean_fmin'],
           iface['within_anchor_spread_fmin']['median']))
  print('\nsaved -> %s' % args.out)


if __name__ == '__main__':
  main()
