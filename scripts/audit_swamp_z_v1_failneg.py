"""Critic-semantic audit of the Z-v1 failure-aware CRL runs.

Read-only. No training, no dataset regeneration. Helpers are imported from
scripts/audit_swamp_z_failneg.py (the accepted v0 audit) so the scoring
function is literally the same code.

WHAT CHANGES FROM v0. In v0 the only failure state was the settled -0.5, so
the paired diagnostic was (x,y,0) vs (x,y,-0.5) -- a 0.5 raw gap. v1 exposes
the sinking as a trajectory, so the PRIMARY pair is now the real one-step
alternative outcome:

    g_safe       = (x', y', 0)        the clear-swamp entry actually observed
    g_fail_entry = (x', y', -0.12)    the same XY under an active bit

a 0.12 raw gap (0.24 after the accepted z/|z_min| scaling). These are NOT
numerically comparable tasks -- v1's physical separation is 4.17x smaller -- so
the question is directional, not whether the margin matches v0's.

New v1-only sections: a depth-response curve over the whole sinking ladder
(section 9), a five-family calibration audit that separates first-contact from
settled failures (10), and the positive-goal distribution stratified by sinking
depth (11).

Usage:
  python scripts/audit_swamp_z_v1_failneg.py
"""
import argparse
import collections
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.15')

import jax                                        # noqa: E402

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402
from crl.report_maze import load_nets             # noqa: E402
from audit_swamp_z_failneg import critic_scores   # noqa: E402
import run_swamp_windy_z_failneg as L             # noqa: E402

OUT_DIR = 'artifacts/swamp_windy_z_v1_failneg_v1'
RUNS = {'zbase (alpha=0)': 'swamp_windy_z_v1_zbase_s0/final.pkl',
        'zfail (alpha=0.1)': 'swamp_windy_z_v1_zfail_s0/final.pkl'}
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
# The full learner-visible sinking ladder for the depth-response curve.
DEPTHS = [0.0, -0.12, -0.24, -0.36, -0.48, -0.5]
Z_ENTRY = -0.12
Z_SETTLED = -0.5


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  return {'n': int(v.size), 'mean': float(v.mean()),
          'median': float(np.median(v)), 'std': float(v.std()),
          'p10': float(np.percentile(v, 10)),
          'p90': float(np.percentile(v, 90))}


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  L.select_version('v1')

  def git(*a):
    try:
      return subprocess.check_output(
          ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
    except Exception:                              # pylint: disable=broad-except
      return ''
  out = {'analysis_script': 'scripts/audit_swamp_z_v1_failneg.py',
         'env': L.ENV, 'dataset': L.DATASET, 'bank': L.BANK,
         'code_commit': git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
         'head_at_runtime': git('rev-parse', 'HEAD'),
         'audited_files_dirty': bool(git('status', '--porcelain', '--',
                                         'crl', 'scripts'))}
  print('=' * 96)
  print('Z-v1 FAILURE-AWARE CRL -- CRITIC SEMANTIC AUDIT')
  print('=' * 96)
  print('  env %s' % L.ENV)
  print('  code commit %s%s' % (out['code_commit'],
                                '  (DIRTY)' if out['audited_files_dirty']
                                else ''))

  # ---------------------------------------------------------------- data
  with np.load(L.DATASET, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    died = np.asarray(d['entered_active_swamp']).astype(bool)
  n_eps, Lr, _ = obs.shape
  xyz = obs[:, :, :3]
  z = xyz[:, :, 2]
  s_t = xyz[:, :-1, :].reshape(-1, 3)
  s_n = xyz[:, 1:, :].reshape(-1, 3)
  a_t = act[:, :-1, :].reshape(-1, 2)

  cell = np.clip(np.floor(s_n[:, :2]).astype(int), 0, [8, 4])
  lands_swamp = np.zeros(len(s_n), bool)
  for cx, cy in SWAMP_CELLS:
    lands_swamp |= (cell[:, 0] == cx) & (cell[:, 1] == cy)
  # FIRST CONTACT only: z was 0 and became negative on this transition.
  first_contact = (s_t[:, 2] == 0) & (s_n[:, 2] < 0)
  safe_entry = lands_swamp & (s_t[:, 2] == 0) & (s_n[:, 2] == 0)
  print('  transitions %s | swamp-landing %s | safe-entry %s | first-contact %s'
        % (format(len(s_t), ','), format(int(lands_swamp.sum()), ','),
           format(int(safe_entry.sum()), ','),
           format(int(first_contact.sum()), ',')))
  fc_z = np.unique(np.round(s_n[first_contact, 2], 4))
  print('  first-contact z values: %s   <-- must be ~%.2f, not %.2f'
        % (fc_z.tolist(), Z_ENTRY, Z_SETTLED))
  assert np.allclose(fc_z, Z_ENTRY, atol=1e-3), (
      'first-contact z is not the first increment -- wrong env version?')
  out['first_contact_z_values'] = [float(v) for v in fc_z]

  rng = np.random.default_rng(args.seed)
  i_safe = np.where(safe_entry)[0]
  if len(i_safe) > 20000:
    i_safe = rng.choice(i_safe, 20000, replace=False)
  i_fc = np.where(first_contact)[0]
  with np.load(L.BANK, allow_pickle=False) as b:
    bank = np.asarray(b['goals'], np.float32)
  print('  bank %s  z values %s' % (bank.shape,
                                    sorted(set(np.round(bank[:, 2], 4)))))

  # ------------------------------------------------------------ invariants
  cfg_env = Config(env_name=L.ENV)
  env = envs_mod.make_env(L.ENV, cfg_env, seed=0)
  env.reset(); env.set_auto_resample(False)
  env.set_swamp([1, 1, 1]); oa = env._get_obs().copy()
  env.set_swamp([0, 0, 0]); oc = env._get_obs().copy()
  inv = {'obs_width': int(oa.shape[0]),
         'hidden_bits_change_obs_by': float(np.abs(oa - oc).max()),
         'z_values_in_dataset': sorted(float(v) for v in np.unique(np.round(z, 4))),
         'bank_all_below_ground': bool((bank[:, 2] < 0).all()),
         'bank_all_first_contact': bool(np.allclose(bank[:, 2], Z_ENTRY,
                                                    atol=1e-3))}
  assert inv['hidden_bits_change_obs_by'] == 0.0
  out['invariants'] = inv
  print('  invariants: obs width %d, hidden-bit leak %.1e, dataset z %s'
        % (inv['obs_width'], inv['hidden_bits_change_obs_by'],
           inv['z_values_in_dataset']))

  # ---------------------------------------------------------------- models
  results = {}
  for label, ckpt in RUNS.items():
    if not os.path.exists(ckpt):
      print('  MISSING %s -- skipping %s' % (ckpt, label))
      continue
    arm = 'zbase' if 'zbase' in ckpt else 'zfail'
    cfg = L.build_cfg(arm, '', steps=1, seed=args.seed)
    envs_mod.make_env(L.ENV, cfg, seed=0)
    nets, state, _, step = load_nets(L.ENV, ckpt, cfg)
    r = {'checkpoint': ckpt, 'step': int(step), 'alpha': cfg.fail_neg_alpha}
    S, A = s_t[i_safe], a_t[i_safe]

    # ---- 7 PRIMARY: safe entry vs same-XY FIRST-CONTACT failure ----
    g_safe = s_n[i_safe].copy()
    g_fe = g_safe.copy(); g_fe[:, 2] = Z_ENTRY
    f_s = critic_scores(nets, state, S, A, g_safe)
    f_fe = critic_scores(nets, state, S, A, g_fe)
    m = f_s - f_fe
    r['7_entry_ranking'] = {'margin': dist(m),
                            'frac_positive': float((m > 0).mean()),
                            'f_safe': dist(f_s), 'f_fail_entry': dist(f_fe)}

    # ---- 8 real fatal (first-contact) anchors ----
    S2, A2 = s_t[i_fc], a_t[i_fc]
    g_ff = s_n[i_fc].copy()
    g_sc = g_ff.copy(); g_sc[:, 2] = 0.0
    f_sc = critic_scores(nets, state, S2, A2, g_sc)
    f_ff = critic_scores(nets, state, S2, A2, g_ff)
    m2 = f_sc - f_ff
    r['8_fatal_ranking'] = {'margin': dist(m2),
                            'frac_positive': float((m2 > 0).mean()),
                            'f_safe_counterpart': dist(f_sc),
                            'f_fail_factual': dist(f_ff)}

    # ---- 9 depth-response curve ----
    curve = {}
    for zv in DEPTHS:
      gz = g_safe.copy(); gz[:, 2] = zv
      fz = critic_scores(nets, state, S, A, gz)
      curve['%.2f' % zv] = {'f': dist(fz), 'margin': dist(f_s - fz)}
    r['9_depth_curve'] = curve

    # ---- 10 five goal families ----
    n = len(i_safe)
    g_rand = xyz.reshape(-1, 3)[rng.choice(n_eps * Lr, n)]
    g_set = g_safe.copy(); g_set[:, 2] = Z_SETTLED
    g_bank = bank[rng.integers(0, len(bank), n)]
    fam = {'factual_safe_future': f_s,
           'ordinary_random': critic_scores(nets, state, S, A, g_rand),
           'same_xy_first_contact_-0.12': f_fe,
           'same_xy_settled_-0.5': critic_scores(nets, state, S, A, g_set),
           'first_contact_bank': critic_scores(nets, state, S, A, g_bank)}
    mu = float(np.concatenate(list(fam.values())).mean())
    r['10_goal_families'] = {k: {'raw': dist(v), 'centred': dist(v - mu)}
                             for k, v in fam.items()}
    r['10_overall_mean_logit'] = mu

    # ---- 11 positives stratified by sinking depth ----
    from crl.offline_audit import build_offline_buffer
    buf, _ = build_offline_buffer(L.DATASET, cfg)
    pz, pf = [], []
    for _ in range(60):
      tr = buf.sample(cfg.batch_size)
      gg = tr.observation[:, 3:]
      pz.append(gg[:, 2].copy())
      pf.append(critic_scores(nets, state, tr.observation[:, :3], tr.action,
                              gg))
    pz = np.round(np.concatenate(pz), 4)
    pf = np.concatenate(pf)
    strat = {}
    for zv in DEPTHS:
      msk = np.isclose(pz, zv, atol=1e-3)
      strat['%.2f' % zv] = {'frac': float(msk.mean()),
                            'critic': dist(pf[msk]) if msk.any() else {'n': 0}}
    r['11_positive_depths'] = {'n_sampled': int(len(pz)), 'by_depth': strat,
                               'frac_z_negative': float((pz < 0).mean())}
    results[label] = r
    print('\n  %s (step %d, alpha %.2f)' % (label, step, cfg.fail_neg_alpha))
    print('    7 entry margin  mean %+.4f  frac>0 %.4f'
          % (m.mean(), (m > 0).mean()))
    print('    8 fatal margin  mean %+.4f  frac>0 %.4f'
          % (m2.mean(), (m2 > 0).mean()))
  out['runs'] = results

  # ---------------------------------------------------------------- tables
  if len(results) == 2:
    ka, kb = 'zbase (alpha=0)', 'zfail (alpha=0.1)'
    A_, B_ = results[ka], results[kb]
    print('\n' + '=' * 96)
    print('7. PRIMARY -- safe entry vs same-XY FIRST-CONTACT failure (-0.12)')
    print('=' * 96)
    print('  %-20s%9s%9s%9s%9s%9s%10s%12s%12s'
          % ('', 'mean', 'median', 'std', 'p10', 'p90', 'frac>0', 'f(safe)',
             'f(fail)'))
    for k, v in ((ka, A_), (kb, B_)):
      d = v['7_entry_ranking']
      print('  %-20s%9.4f%9.4f%9.4f%9.4f%9.4f%10.4f%12.4f%12.4f'
            % (k, d['margin']['mean'], d['margin']['median'],
               d['margin']['std'], d['margin']['p10'], d['margin']['p90'],
               d['frac_positive'], d['f_safe']['mean'],
               d['f_fail_entry']['mean']))
    dm = (B_['7_entry_ranking']['margin']['mean']
          - A_['7_entry_ranking']['margin']['mean'])
    print('  DELTA m (zfail - zbase): %+.4f' % dm)
    out['delta_m_entry'] = float(dm)

    print('\n' + '=' * 96)
    print('8. FATAL (first-contact) anchors -- m>0 NOT required')
    print('=' * 96)
    print('  %-20s%9s%9s%10s%16s' % ('', 'mean', 'median', 'frac>0',
                                     'f(factual)'))
    for k, v in ((ka, A_), (kb, B_)):
      d = v['8_fatal_ranking']
      print('  %-20s%9.4f%9.4f%10.4f%16.4f'
            % (k, d['margin']['mean'], d['margin']['median'],
               d['frac_positive'], d['f_fail_factual']['mean']))

    print('\n' + '=' * 96)
    print('9. DEPTH-RESPONSE CURVE  (no monotonicity was trained or imposed)')
    print('=' * 96)
    hdr = '  %-20s' % 'z' + ''.join('%10s' % ('%.2f' % v) for v in DEPTHS)
    print(hdr)
    for k, v in ((ka, A_), (kb, B_)):
      print('  %-20s' % ('E[f]  ' + k.split()[0])
            + ''.join('%10.4f' % v['9_depth_curve']['%.2f' % d]['f']['mean']
                      for d in DEPTHS))
    for k, v in ((ka, A_), (kb, B_)):
      print('  %-20s' % ('m(z)  ' + k.split()[0])
            + ''.join('%10.4f' % v['9_depth_curve']['%.2f' % d]['margin']['mean']
                      for d in DEPTHS))

    print('\n' + '=' * 96)
    print('10. GOAL FAMILIES  (raw | mean-centred per model)')
    print('=' * 96)
    print('  %-30s%11s%11s%13s%11s' % ('family', 'raw a=0', 'raw a=.1',
                                       'ctr a=0', 'ctr a=.1'))
    for fm in A_['10_goal_families']:
      print('  %-30s%11.4f%11.4f%13.4f%11.4f'
            % (fm, A_['10_goal_families'][fm]['raw']['mean'],
               B_['10_goal_families'][fm]['raw']['mean'],
               A_['10_goal_families'][fm]['centred']['mean'],
               B_['10_goal_families'][fm]['centred']['mean']))
    print('  overall mean logit: a=0 %.4f   a=0.1 %.4f'
          % (A_['10_overall_mean_logit'], B_['10_overall_mean_logit']))

    print('\n' + '=' * 96)
    print('11. POSITIVE GOALS BY SINKING DEPTH (not masked, not rebalanced)')
    print('=' * 96)
    print('  %-10s%12s%14s%14s' % ('z', 'frac', 'f a=0', 'f a=.1'))
    for zv in DEPTHS:
      k = '%.2f' % zv
      fa = A_['11_positive_depths']['by_depth'][k]
      fb = B_['11_positive_depths']['by_depth'][k]
      print('  %-10s%12.5f%14s%14s'
            % (k, fa['frac'],
               '%.4f' % fa['critic']['mean'] if fa['critic']['n'] else '-',
               '%.4f' % fb['critic']['mean'] if fb['critic']['n'] else '-'))
    print('  frac of positives with z<0: a=0 %.5f  a=0.1 %.5f'
          % (A_['11_positive_depths']['frac_z_negative'],
             B_['11_positive_depths']['frac_z_negative']))

    # ---- 13 v0 vs v1 ----
    print('\n' + '=' * 96)
    print('13. V0 vs V1  (V0 numbers quoted from the accepted result, NOT '
          'retrained)')
    print('=' * 96)
    print('  NOT the same task: v0 paired 0 vs -0.5 (raw gap 0.50), v1 pairs')
    print('  0 vs -0.12 (raw gap 0.12) -- 4.17x smaller physical separation.')
    print('  %-10s%-22s%12s%12s' % ('', 'paired goals', 'E[m] a=0',
                                    'E[m] a=.1'))
    print('  %-10s%-22s%12.4f%12.4f' % ('Z-v0', '(x,y,0) vs -0.50',
                                        -0.1623, 0.4260))
    print('  %-10s%-22s%12.4f%12.4f'
          % ('Z-v1', '(x,y,0) vs -0.12',
             A_['7_entry_ranking']['margin']['mean'],
             B_['7_entry_ranking']['margin']['mean']))
    print('  %-10s%-22s%12s%12s' % ('', 'P(m>0)', 'a=0', 'a=.1'))
    print('  %-10s%-22s%12.4f%12.4f' % ('Z-v0', '', 0.3488, 0.6606))
    print('  %-10s%-22s%12.4f%12.4f'
          % ('Z-v1', '', A_['7_entry_ranking']['frac_positive'],
             B_['7_entry_ranking']['frac_positive']))
    out['13_v0_vs_v1'] = {
        'v0_reference': {'paired': '(x,y,0) vs (x,y,-0.5)', 'raw_gap': 0.5,
                         'mean_margin_a0': -0.1623, 'mean_margin_a01': 0.4260,
                         'frac_pos_a0': 0.3488, 'frac_pos_a01': 0.6606,
                         'note': 'quoted from the accepted v0 result; NOT '
                                 'retrained here'},
        'v1': {'paired': '(x,y,0) vs (x,y,-0.12)', 'raw_gap': 0.12,
               'mean_margin_a0': A_['7_entry_ranking']['margin']['mean'],
               'mean_margin_a01': B_['7_entry_ranking']['margin']['mean'],
               'frac_pos_a0': A_['7_entry_ranking']['frac_positive'],
               'frac_pos_a01': B_['7_entry_ranking']['frac_positive']},
        'comparability': 'the two margins are NOT numerically comparable; v1 '
                         'physical separation is 4.17x smaller. The question '
                         'is directional.'}

  os.makedirs(args.out_dir, exist_ok=True)
  p = os.path.join(args.out_dir, 'critic_audit.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % p)


if __name__ == '__main__':
  main()
