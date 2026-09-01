"""Critic-semantic audit of the Z-state failure-aware CRL runs.

Compares zbase (alpha=0) against zfail (alpha=0.1) on the question the whole
Z variant exists to answer:

    with XY held EXACTLY fixed, does the critic separate
        g_safe = (x', y', 0)   from   g_fail = (x', y', -0.5) ?

Everything here is read-only. No training, no dataset regeneration.

THE SCORE. f(s, a, g) is the critic logit, computed the way the actor consumes
it: crl/networks.py q_network.apply on the flat [state | goal] observation,
taking the diagonal of the B x B outer product so element i is
critic(s_i, a_i, g_i). z_physical is applied INSIDE the network, so this audit
feeds RAW physical coordinates exactly as training did -- feeding pre-scaled
values here would double-apply it.

  8A safe-entry anchors  -- transitions that enter a swamp cell and survive,
     so the factual outcome is z=0 and the synthetic counterpart is z=-0.5.
     m_safe = f(g_safe) - f(g_fail) SHOULD be positive and should grow with
     alpha if the failure term is doing semantic work.

  8B fatal-entry anchors -- transitions that actually died. The factual
     outcome IS g_fail, so m_fatal > 0 is NOT required; this checks that
     alpha=0.1 does not destroy the factual association.

  9  z<0 positives -- how often the ordinary future relabeling legitimately
     hands back a z<0 goal, and what the critic scores it. This is overlap
     between the positive occupancy distribution and the failure reference
     distribution, not a label conflict; nothing is masked.

  10 failure-bank effect on safe anchors -- four goal families scored under
     both models, reported raw AND after subtracting each model's own mean
     logit, so a global calibration shift cannot masquerade as a ranking
     change.

  12 invariants -- re-asserted here, not assumed from the earlier audit.

Usage:
  python scripts/audit_swamp_z_failneg.py
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.15')

import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402
from crl.report_maze import load_nets             # noqa: E402
from run_swamp_windy_z_failneg import (           # noqa: E402
    BANK, DATASET, ENV, build_cfg)

OUT_DIR = 'artifacts/swamp_windy_z_failneg_v1'
RUNS = {'zbase (alpha=0)': 'swamp_windy_z_zbase_s0/final.pkl',
        'zfail (alpha=0.1)': 'swamp_windy_z_zfail_s0/final.pkl'}
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
Z_FAIL = -0.5


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  return {'n': int(v.size), 'mean': float(v.mean()),
          'median': float(np.median(v)),
          'p10': float(np.percentile(v, 10)),
          'p90': float(np.percentile(v, 90)),
          'std': float(v.std())}


def critic_scores(nets, state, s, a, g, chunk=4096):
  """f(s_i, a_i, g_i) for each i -- the diagonal of the outer product."""
  out = []
  for k in range(0, len(s), chunk):
    obs = np.concatenate([s[k:k + chunk], g[k:k + chunk]], axis=1)
    r = np.asarray(nets.q_network.apply(state.q_params, jnp.asarray(obs,
                                                                   jnp.float32),
                                        jnp.asarray(a[k:k + chunk],
                                                    jnp.float32)))
    if r.ndim == 3:
      r = r.min(axis=-1)
    out.append(np.einsum('ii->i', r) if r.ndim == 2 else r)
  return np.concatenate(out)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()

  def git(*a):
    try:
      return subprocess.check_output(['git'] + list(a),
                                     cwd=os.path.dirname(_HERE)).decode().strip()
    except Exception:                              # pylint: disable=broad-except
      return ''
  code_commit = git('log', '-1', '--format=%H', '--', 'crl', 'scripts')
  head = git('rev-parse', 'HEAD')
  dirty = bool(git('status', '--porcelain', '--', 'crl', 'scripts'))

  out = {'analysis_script': 'scripts/audit_swamp_z_failneg.py',
         'code_commit': code_commit, 'head_at_runtime': head,
         'audited_files_dirty': dirty, 'dataset': DATASET, 'bank': BANK}
  print('=' * 96)
  print('Z FAILURE-AWARE CRL -- CRITIC SEMANTIC AUDIT')
  print('=' * 96)
  print('  code commit %s%s' % (code_commit,
                                '  (WORKING TREE DIRTY)' if dirty else ''))

  # ---------------------------------------------------------------- data
  with np.load(DATASET, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    died = np.asarray(d['entered_active_swamp']).astype(bool)
  n_eps, L, _ = obs.shape
  xyz = obs[:, :, :3]
  z = xyz[:, :, 2]
  s_t = xyz[:, :-1, :].reshape(-1, 3)
  s_n = xyz[:, 1:, :].reshape(-1, 3)
  a_t = act[:, :-1, :].reshape(-1, 2)
  ep_of = np.repeat(np.arange(n_eps)[:, None], L - 1, axis=1).reshape(-1)

  cell = np.clip(np.floor(s_n[:, :2]).astype(int), 0, [8, 4])
  lands_swamp = np.zeros(len(s_n), bool)
  for cx, cy in SWAMP_CELLS:
    lands_swamp |= (cell[:, 0] == cx) & (cell[:, 1] == cy)
  fatal = (s_t[:, 2] == 0) & (s_n[:, 2] < 0)
  safe_entry = lands_swamp & (s_t[:, 2] == 0) & (s_n[:, 2] == 0)
  print('  transitions %s   swamp-landing %s   safe-entry %s   fatal-entry %s'
        % (format(len(s_t), ','), format(int(lands_swamp.sum()), ','),
           format(int(safe_entry.sum()), ','), format(int(fatal.sum()), ',')))

  rng = np.random.default_rng(args.seed)
  i_safe = np.where(safe_entry)[0]
  if len(i_safe) > 20000:
    i_safe = rng.choice(i_safe, 20000, replace=False)
  i_fatal = np.where(fatal)[0]
  with np.load(BANK, allow_pickle=False) as b:
    bank = np.asarray(b['goals'], np.float32)

  # ---------------------------------------------------------------- 12
  print('\n' + '=' * 96)
  print('12. INVARIANTS (re-asserted, not assumed)')
  print('=' * 96)
  cfg_env = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg_env, seed=0)
  env.reset(); env.set_auto_resample(False)
  env.set_swamp([1, 1, 1]); oa = env._get_obs().copy()
  env.set_swamp([0, 0, 0]); oc = env._get_obs().copy()
  inv = {'obs_width': int(oa.shape[0]),
         'hidden_bits_change_obs_by': float(np.abs(oa - oc).max()),
         'safe_outcomes_all_z0': bool((s_n[safe_entry, 2] == 0).all()),
         'fatal_outcomes_all_zfail': bool(
             np.allclose(s_n[fatal, 2], Z_FAIL, atol=1e-6)),
         'bank_all_below_ground': bool((bank[:, 2] < 0).all()),
         'z_values_in_dataset': sorted(float(v) for v in np.unique(z)),
         'obs_norm_mode': 'z_physical', 'obs_norm_applied_in':
             'crl/networks.py via crl.obs_norm.obs_scale_vector (once)',
         'empirical_z_standardization': False}
  cfg2 = Config(env_name='point_two_route_swamp_windy_v0')
  e2 = envs_mod.make_env('point_two_route_swamp_windy_v0', cfg2, seed=0)
  inv['old_2d_env_unaffected'] = bool(cfg2.obs_dim == 2 and cfg2.goal_dim == 2
                                      and e2.reset().shape == (4,))
  for k, v in inv.items():
    print('  %-34s %s' % (k, v))
  assert inv['hidden_bits_change_obs_by'] == 0.0
  assert inv['safe_outcomes_all_z0'] and inv['fatal_outcomes_all_zfail']
  assert inv['old_2d_env_unaffected']
  assert inv['z_values_in_dataset'] == [Z_FAIL, 0.0]
  out['12_invariants'] = inv

  # ---------------------------------------------------------------- models
  results = {}
  for label, ckpt in RUNS.items():
    if not os.path.exists(ckpt):
      print('\n  MISSING %s -- skipping %s' % (ckpt, label))
      continue
    arm = 'zbase' if 'zbase' in ckpt else 'zfail'
    cfg = build_cfg(arm, '', steps=1, seed=args.seed)
    envs_mod.make_env(ENV, cfg, seed=0)
    nets, state, _, step = load_nets(ENV, ckpt, cfg)
    r = {'checkpoint': ckpt, 'step': int(step), 'alpha': cfg.fail_neg_alpha}

    # ---- 8A safe-entry paired ranking ----
    S, A = s_t[i_safe], a_t[i_safe]
    g_safe = s_n[i_safe].copy()
    g_fail = g_safe.copy(); g_fail[:, 2] = Z_FAIL
    f_s = critic_scores(nets, state, S, A, g_safe)
    f_f = critic_scores(nets, state, S, A, g_fail)
    m = f_s - f_f
    r['8A_safe_entry'] = {'margin': dist(m), 'frac_positive': float((m > 0).mean()),
                          'f_safe': dist(f_s), 'f_fail': dist(f_f)}

    # ---- 8B fatal-entry paired ranking ----
    S2, A2 = s_t[i_fatal], a_t[i_fatal]
    g_f2 = s_n[i_fatal].copy()
    g_s2 = g_f2.copy(); g_s2[:, 2] = 0.0
    f_s2 = critic_scores(nets, state, S2, A2, g_s2)
    f_f2 = critic_scores(nets, state, S2, A2, g_f2)
    m2 = f_s2 - f_f2
    r['8B_fatal_entry'] = {'margin': dist(m2),
                           'frac_positive': float((m2 > 0).mean()),
                           'f_safe_synthetic': dist(f_s2),
                           'f_fail_factual': dist(f_f2)}

    # ---- 10 goal families on safe anchors, raw and mean-centred ----
    n = len(i_safe)
    g_rand = xyz.reshape(-1, 3)[rng.choice(n_eps * L, n)]
    g_bankg = bank[rng.integers(0, len(bank), n)]
    fam = {'factual_safe_future': f_s, 'ordinary_random': None,
           'same_xy_synthetic_failure': f_f, 'failure_bank_3d': None}
    fam['ordinary_random'] = critic_scores(nets, state, S, A, g_rand)
    fam['failure_bank_3d'] = critic_scores(nets, state, S, A, g_bankg)
    allf = np.concatenate(list(fam.values()))
    mu = float(allf.mean())
    r['10_goal_families'] = {
        k: {'raw': dist(v), 'centred': dist(v - mu)} for k, v in fam.items()}
    r['10_overall_mean_logit'] = mu

    # ---- 9 z<0 positives through ordinary relabeling ----
    from crl.offline_audit import build_offline_buffer
    cfgb = build_cfg(arm, '', steps=1, seed=args.seed)
    envs_mod.make_env(ENV, cfgb, seed=0)
    buf, _ = build_offline_buffer(DATASET, cfgb)
    pos_z, pos_f = [], []
    for _ in range(40):
      tr = buf.sample(cfgb.batch_size)
      st = tr.observation[:, :3]
      gg = tr.observation[:, 3:]
      pos_z.append(gg[:, 2].copy())
      pos_f.append(critic_scores(nets, state, st, tr.action, gg))
    pos_z = np.concatenate(pos_z); pos_f = np.concatenate(pos_f)
    neg = pos_z < 0
    r['9_z_negative_positives'] = {
        'n_sampled': int(len(pos_z)),
        'frac_sampled_positives_with_z_negative': float(neg.mean()),
        'critic_on_z_negative_positives': dist(pos_f[neg]),
        'critic_on_z_zero_positives': dist(pos_f[~neg])}
    results[label] = r
    print('\n  %s  (step %d, alpha %.2f)' % (label, step, cfg.fail_neg_alpha))
    print('    8A margin mean %+.4f median %+.4f  frac>0 %.4f'
          % (m.mean(), np.median(m), (m > 0).mean()))
    print('    8B margin mean %+.4f median %+.4f  frac>0 %.4f'
          % (m2.mean(), np.median(m2), (m2 > 0).mean()))
    print('    9  frac positives with z<0 %.5f   critic on them %+.4f'
          % (neg.mean(), pos_f[neg].mean() if neg.any() else float('nan')))

  out['runs'] = results

  # ---------------------------------------------------------------- report
  if len(results) == 2:
    ka, kb = 'zbase (alpha=0)', 'zfail (alpha=0.1)'
    A_, B_ = results[ka], results[kb]
    print('\n' + '=' * 96)
    print('8A SAFE-ENTRY PAIRED RANKING   m_safe = f(x,y,0) - f(x,y,-0.5)')
    print('=' * 96)
    print('  %-22s%10s%10s%10s%10s%12s' % ('', 'mean', 'median', 'p10', 'p90',
                                           'frac>0'))
    for k, v in ((ka, A_), (kb, B_)):
      d = v['8A_safe_entry']['margin']
      print('  %-22s%10.4f%10.4f%10.4f%10.4f%12.4f'
            % (k, d['mean'], d['median'], d['p10'], d['p90'],
               v['8A_safe_entry']['frac_positive']))
    da = A_['8A_safe_entry']['margin']['mean']
    db = B_['8A_safe_entry']['margin']['mean']
    print('  DELTA (zfail - zbase) mean margin: %+.4f' % (db - da))
    print('\n  component scores')
    print('  %-22s%14s%14s' % ('', 'f(g_safe)', 'f(g_fail)'))
    for k, v in ((ka, A_), (kb, B_)):
      print('  %-22s%14.4f%14.4f' % (k, v['8A_safe_entry']['f_safe']['mean'],
                                     v['8A_safe_entry']['f_fail']['mean']))

    print('\n' + '=' * 96)
    print('8B FATAL-ENTRY  m_fatal = f(x,y,0) - f(x,y,-0.5)   '
          '(m>0 NOT required)')
    print('=' * 96)
    print('  %-22s%10s%10s%12s%16s' % ('', 'mean', 'median', 'frac>0',
                                       'f(factual fail)'))
    for k, v in ((ka, A_), (kb, B_)):
      d = v['8B_fatal_entry']['margin']
      print('  %-22s%10.4f%10.4f%12.4f%16.4f'
            % (k, d['mean'], d['median'], v['8B_fatal_entry']['frac_positive'],
               v['8B_fatal_entry']['f_fail_factual']['mean']))

    print('\n' + '=' * 96)
    print('10 GOAL FAMILIES ON SAFE ANCHORS  (raw | mean-centred per model)')
    print('=' * 96)
    fams = ['factual_safe_future', 'ordinary_random',
            'same_xy_synthetic_failure', 'failure_bank_3d']
    print('  %-28s%11s%11s%13s%11s' % ('family', 'raw a=0', 'raw a=.1',
                                       'ctr a=0', 'ctr a=.1'))
    for fm in fams:
      print('  %-28s%11.4f%11.4f%13.4f%11.4f'
            % (fm, A_['10_goal_families'][fm]['raw']['mean'],
               B_['10_goal_families'][fm]['raw']['mean'],
               A_['10_goal_families'][fm]['centred']['mean'],
               B_['10_goal_families'][fm]['centred']['mean']))
    print('  overall mean logit: a=0 %.4f   a=0.1 %.4f'
          % (A_['10_overall_mean_logit'], B_['10_overall_mean_logit']))

    print('\n' + '=' * 96)
    print('9 z<0 POSITIVES THROUGH ORDINARY RELABELING (no masking)')
    print('=' * 96)
    print('  failed episodes: %d / %d = %.4f'
          % (int(died.sum()), n_eps, died.mean()))
    fut = np.array([bool((z[e] < 0).any()) for e in range(n_eps)])
    print('  episodes whose future contains z<0: %.4f' % fut.mean())
    for k, v in ((ka, A_), (kb, B_)):
      d = v['9_z_negative_positives']
      print('  %-22s frac sampled positives z<0 %.5f   f(z<0 pos) %+.4f   '
            'f(z=0 pos) %+.4f'
            % (k, d['frac_sampled_positives_with_z_negative'],
               d['critic_on_z_negative_positives']['mean'],
               d['critic_on_z_zero_positives']['mean']))
    out['9_dataset_level'] = {
        'frac_failed_episodes': float(died.mean()),
        'frac_episodes_with_z_negative_future': float(fut.mean())}

  os.makedirs(args.out_dir, exist_ok=True)
  p = os.path.join(args.out_dir, 'critic_audit.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % p)


if __name__ == '__main__':
  main()
