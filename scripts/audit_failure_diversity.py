"""Phase 4+5: build D_fail^diverse and run the DIVERSITY GATE.

Phase 4 -- pool construction. From each V3 arm's frozen factual collection,
take exactly ONE settled-fatal transition per naturally occurring death,
    (s_predeath, a_predeath, s'_settled)
    = (obs[e, c], act[e, c], obs[e, lengths[e]-1]),
identical to the V2 criterion. No pre-death windows, no artificial
anchor/final pairing, no mild contacts, no wall collisions, no hidden-state
derived samples. The original D_fail^196 is preserved and unioned in; exact
duplicate transitions are removed, near-duplicates are NOT (and nothing is
removed using final39 outcomes).

Phase 5 -- gate. Using the same frozen state/action normalization as V2:
  5A new-to-old distance: how far each NEW failure anchor is from the
     nearest of the original 196, with the threshold taken from the OLD
     pool's own nearest-neighbour geometry (its median NN distance), not
     invented post hoc;
  5B old-vs-combined coverage geometry on development anchors (dev16, and
     the REUSED final39 clearly labelled as a post-hoc development
     diagnostic -- it is no longer a sealed final test);
  5C per-arm contribution (metadata only; the arm id is never a Flow input);
  5D qualitative PASS/FAIL.

Usage:
  python scripts/audit_failure_diversity.py
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402
from train_flow_v0_clean import OBS_DIM    # noqa: E402
from train_flow_v2 import build_fail_pool  # noqa: E402

ROOT = 'artifacts/flow_v3_diverse_failure'
COLLECT = os.path.join(ROOT, 'bad_demo_diverse')
OUT = os.path.join(ROOT, 'failure_pool_diversity_audit')
V0_DIR = 'artifacts/flow_v0_clean'
ACT_STATS = 'artifacts/flow_v05_clean_action/action_stats.json'
OLD_BAD_DIR = 'artifacts/bad_demo_fixed'
OLD_BAD_NAME = 'bad_demo_blind_p30_h800_settle80'
DEV16 = 'artifacts/same_anchor_candidate_probe/pairs_debug16_pilot.npz'
FINAL39 = 'artifacts/flow_v2_final39/candidates_and_scores.npz'
FINAL39_CSV = 'artifacts/flow_v2_final39/per_case.csv'


def desc(a):
  a = np.asarray(a, float)
  if not len(a):
    return None
  return {'n': int(len(a)), 'median': float(np.median(a)),
          'mean': float(a.mean()), 'min': float(a.min()),
          'max': float(a.max()),
          'p10': float(np.percentile(a, 10)),
          'p90': float(np.percentile(a, 90))}


def arm_fail_pool(npz, side):
  d = np.load(npz, allow_pickle=True)
  s = np.load(side, allow_pickle=True)
  obs, act = np.asarray(d['obs'], np.float32), np.asarray(d['act'], np.float32)
  ln = np.asarray(d['lengths'], np.int64)
  dead = np.asarray(s['dead'], bool)
  col = np.asarray(s['collapse_step'], np.int64)
  S, A, S2, ep = [], [], [], []
  for e in np.where(dead)[0]:
    c, last = int(col[e]), int(ln[e]) - 1
    assert last == c + 1, 'dead episode not truncated at collapse+1'
    S.append(obs[e, c, :OBS_DIM])
    A.append(act[e, c])
    S2.append(obs[e, last, :OBS_DIM])
    ep.append(int(e))
  if not S:
    z = np.zeros((0, OBS_DIM), np.float32)
    return z, np.zeros((0, 8), np.float32), z, np.array([], np.int64)
  S, A, S2 = np.stack(S), np.stack(A), np.stack(S2)
  return S, A, S2 - S, np.asarray(ep, np.int64)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  nz_ = np.load(os.path.join(V0_DIR, 'norm_stats.npz'))
  s_mean = np.asarray(nz_['state_mean'], np.float32)
  s_std = np.asarray(nz_['state_std'], np.float32)
  d_mean = np.asarray(nz_['delta_mean'], np.float32)
  d_std = np.asarray(nz_['delta_std'], np.float32)
  ast = json.load(open(ACT_STATS))
  a_mean = np.asarray(ast['per_dim_mean'], np.float32)
  a_std = np.asarray(ast['per_dim_std'], np.float32)
  ns = lambda x: (x - s_mean) / s_std          # noqa: E731
  na = lambda x: (x - a_mean) / a_std          # noqa: E731

  # ---- Phase 4: pools ------------------------------------------------------
  So, Ao, Do, _ = build_fail_pool(OLD_BAD_DIR, OLD_BAD_NAME)   # 196
  arms, Sn, An, Dn, arm_id = [], [], [], [], []
  for npz in sorted(glob.glob(os.path.join(COLLECT, 'bad_demo_*.npz'))):
    if npz.endswith('_sidecar.npz'):
      continue
    side = npz.replace('.npz', '_sidecar.npz')
    if not os.path.exists(side):
      continue
    arm = json.loads(str(np.load(npz, allow_pickle=True)['meta']))['arm']
    s_, a_, d_, ep_ = arm_fail_pool(npz, side)
    arms.append({'arm': arm, 'npz': npz,
                 'npz_sha256': C.sha256_file(npz),
                 'n_fatal': int(len(s_)), 'episode_ids': ep_.tolist()})
    if len(s_):
      Sn.append(s_)
      An.append(a_)
      Dn.append(d_)
      arm_id += [arm] * len(s_)
    print('  arm %-16s fatal transitions: %d' % (arm, len(s_)), flush=True)
  assert arms, 'no arm collections found'
  Sn = np.concatenate(Sn) if Sn else np.zeros((0, OBS_DIM), np.float32)
  An = np.concatenate(An) if An else np.zeros((0, 8), np.float32)
  Dn = np.concatenate(Dn) if Dn else np.zeros((0, OBS_DIM), np.float32)
  arm_id = np.asarray(arm_id)

  # combined pool with EXACT-duplicate removal only
  allS = np.concatenate([So, Sn])
  allA = np.concatenate([Ao, An])
  allD = np.concatenate([Do, Dn])
  src = np.array(['old196'] * len(So) + list(arm_id))
  key = np.concatenate([allS, allA, allD], axis=1)
  _, uniq = np.unique(key, axis=0, return_index=True)
  uniq = np.sort(uniq)
  n_dup = len(allS) - len(uniq)
  allS, allA, allD, src = allS[uniq], allA[uniq], allD[uniq], src[uniq]
  print('\npool: old %d + new %d -> combined %d (exact duplicates removed: %d)'
        % (len(So), len(Sn), len(allS), n_dup), flush=True)

  NSo, NAo = ns(So), na(Ao)
  NSn, NAn = ns(Sn), na(An)

  # ---- 5A: new-to-old distance --------------------------------------------
  # threshold derived from the OLD pool's own NN geometry (not invented)
  Doo = np.linalg.norm(NSo[:, None] - NSo[None], axis=2)
  np.fill_diagonal(Doo, np.inf)
  old_nn = Doo.min(1)
  scale_s = float(np.median(old_nn))
  Doo_sa = np.linalg.norm(
      np.concatenate([NSo, NAo], 1)[:, None]
      - np.concatenate([NSo, NAo], 1)[None], axis=2)
  np.fill_diagonal(Doo_sa, np.inf)
  scale_sa = float(np.median(Doo_sa.min(1)))

  d_new_old_s = (np.linalg.norm(NSn[:, None] - NSo[None], axis=2).min(1)
                 if len(NSn) else np.array([]))
  d_new_old_sa = (np.linalg.norm(
      np.concatenate([NSn, NAn], 1)[:, None]
      - np.concatenate([NSo, NAo], 1)[None], axis=2).min(1)
      if len(NSn) else np.array([]))
  a5 = {
      'old_pool_own_nn_median_state': scale_s,
      'old_pool_own_nn_median_state_action': scale_sa,
      'threshold_note': ('"close" / "outside" are defined by the OLD pool\'s '
                         'own median nearest-neighbour distance; no post-hoc '
                         'threshold was invented'),
      'new_to_old_state': desc(d_new_old_s),
      'new_to_old_state_action': desc(d_new_old_sa),
      'frac_new_within_old_nn_scale_state':
          float((d_new_old_s <= scale_s).mean()) if len(d_new_old_s) else None,
      'frac_new_beyond_old_nn_scale_state':
          float((d_new_old_s > scale_s).mean()) if len(d_new_old_s) else None,
      'frac_new_beyond_2x_old_nn_scale_state':
          float((d_new_old_s > 2 * scale_s).mean()) if len(d_new_old_s)
          else None,
  }

  # ---- 5B: old vs combined coverage geometry on development anchors -------
  def dev_block(name, Sd, Ad, extra=None):
    NSd, NAd = ns(Sd), na(Ad)
    d_old = np.linalg.norm(NSd[:, None] - NSo[None], axis=2).min(1)
    d_div = np.linalg.norm(NSd[:, None] - ns(allS)[None], axis=2).min(1)
    d_old_sa = np.linalg.norm(
        np.concatenate([NSd, NAd], 1)[:, None]
        - np.concatenate([NSo, NAo], 1)[None], axis=2).min(1)
    d_div_sa = np.linalg.norm(
        np.concatenate([NSd, NAd], 1)[:, None]
        - np.concatenate([ns(allS), na(allA)], 1)[None], axis=2).min(1)
    out = {'n_anchors': int(len(Sd)),
           'd_old_state': desc(d_old), 'd_diverse_state': desc(d_div),
           'delta_reduction_state': desc(d_old - d_div),
           'd_old_state_action': desc(d_old_sa),
           'd_diverse_state_action': desc(d_div_sa),
           'delta_reduction_state_action': desc(d_old_sa - d_div_sa),
           'frac_anchors_improved': float((d_old - d_div > 1e-6).mean())}
    if extra is not None:
      unc = extra == 0
      out['previously_uncovered'] = {
          'n': int(unc.sum()),
          'd_old_state': desc(d_old[unc]),
          'd_diverse_state': desc(d_div[unc]),
          'delta_reduction_state': desc((d_old - d_div)[unc]),
          'frac_improved': float((d_old - d_div > 1e-6)[unc].mean())}
      out['previously_covered'] = {
          'n': int((~unc).sum()),
          'delta_reduction_state': desc((d_old - d_div)[~unc])}
    return out, d_old, d_div

  p16 = np.load(DEV16, allow_pickle=True)
  b16, _, _ = dev_block('dev16', np.asarray(p16['anchor_obs'], np.float32),
                        np.asarray(p16['anchor_action'], np.float32))
  z39 = np.load(FINAL39, allow_pickle=True)
  cov39 = np.array([int(r['covered']) for r in
                    csv.DictReader(open(FINAL39_CSV))])
  b39, d39_old, d39_div = dev_block(
      'final39', np.asarray(z39['anchor'], np.float32),
      np.asarray(z39['action'], np.float32), extra=cov39)
  b39['LABEL'] = ('REUSED DEVELOPMENT DIAGNOSTIC ONLY -- final39 stopped '
                  'being a sealed final test when it was used to motivate V3')

  # ---- 5C: per-arm contribution -------------------------------------------
  per_arm = {}
  for a in arms:
    m = arm_id == a['arm']
    if not m.any():
      per_arm[a['arm']] = {'n_fatal': 0}
      continue
    dn = d_new_old_s[m]
    # unique contribution: distance to failures from all OTHER sources
    other = np.concatenate([NSo, NSn[~m]]) if (~m).any() else NSo
    d_other = np.linalg.norm(NSn[m][:, None] - other[None], axis=2).min(1)
    per_arm[a['arm']] = {
        'n_fatal': int(m.sum()),
        'nearest_old_state_distance': desc(dn),
        'frac_beyond_old_nn_scale': float((dn > scale_s).mean()),
        'distance_to_all_other_failure_sources': desc(d_other),
        'frac_unique_beyond_scale_vs_all_others':
            float((d_other > scale_s).mean())}

  # ---- 5D: gate ------------------------------------------------------------
  n_new = int(len(Sn))
  frac_beyond = a5['frac_new_beyond_old_nn_scale_state'] or 0.0
  red_unc = b39['previously_uncovered']['delta_reduction_state']['median']
  frac_imp_unc = b39['previously_uncovered']['frac_improved']
  gate = {
      'criteria': {
          'substantial_new_fatal_transitions': n_new >= 100,
          'not_overwhelmingly_near_duplicates': frac_beyond >= 0.5,
          'reduces_distance_for_previously_sparse_regions':
              (red_unc > 0 and frac_imp_unc >= 0.5)},
      'values': {'n_new_fatal_transitions': n_new,
                 'frac_new_beyond_old_nn_scale': frac_beyond,
                 'median_distance_reduction_previously_uncovered': red_unc,
                 'frac_previously_uncovered_improved': frac_imp_unc},
      'note': ('qualitative gate as specified; death count alone is not '
               'sufficient -- the pool must expand the failure manifold')}
  gate['PASS'] = all(gate['criteria'].values())

  summary = {'phase4_pool': {
                 'n_old': int(len(So)), 'n_new': n_new,
                 'n_combined_after_exact_dedup': int(len(allS)),
                 'n_exact_duplicates_removed': int(n_dup),
                 'per_arm': arms,
                 'construction': ('one settled-fatal transition per natural '
                                  'death; no windows, no artificial pairing, '
                                  'no hidden-state samples')},
             'phase5A_new_to_old': a5,
             'phase5B_dev16': b16,
             'phase5B_final39_reused_development_diagnostic': b39,
             'phase5C_per_arm': per_arm,
             'phase5D_gate': gate,
             'normalization': {'state': os.path.join(V0_DIR,
                                                     'norm_stats.npz'),
                               'action': ACT_STATS},
             'git_commit': C.git_commit()}
  json.dump(summary, open(os.path.join(args.out, 'diversity_audit.json'),
                          'w'), indent=2)
  np.savez_compressed(os.path.join(args.out, 'failure_pool_diverse.npz'),
                      state=allS, action=allA, delta=allD, source=src,
                      meta=json.dumps({
                          'definition': 'D_fail^diverse = D_fail^196 U '
                                        'D_fail^new (exact dups removed)',
                          'n': int(len(allS))}))

  # ---- plots ---------------------------------------------------------------
  fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
  ax[0].hist(d_new_old_s, bins=30, color='crimson', alpha=0.8)
  ax[0].axvline(scale_s, color='k', ls='--', lw=1.4,
                label='old-pool median NN = %.2f' % scale_s)
  ax[0].set_xlabel('new failure anchor -> nearest OLD failure anchor')
  ax[0].set_ylabel('count')
  ax[0].set_title('5A: does new data leave the old manifold?')
  ax[0].legend(fontsize=8)
  unc = cov39 == 0
  ax[1].scatter(d39_old[~unc], d39_div[~unc], s=40, color='tab:green',
                label='previously covered')
  ax[1].scatter(d39_old[unc], d39_div[unc], s=40, color='tab:red',
                marker='x', label='previously uncovered')
  lim = [0, max(d39_old.max(), d39_div.max()) * 1.05]
  ax[1].plot(lim, lim, 'k--', lw=1)
  ax[1].set_xlabel('nearest-failure distance (old 196)')
  ax[1].set_ylabel('nearest-failure distance (diverse pool)')
  ax[1].set_title('5B: support distance, old vs diverse')
  ax[1].legend(fontsize=8)
  names = list(per_arm)
  vals = [per_arm[k].get('n_fatal', 0) for k in names]
  ax[2].bar(range(len(names)), vals, color='tab:blue')
  ax[2].set_xticks(range(len(names)))
  ax[2].set_xticklabels(names, rotation=20, fontsize=8)
  ax[2].set_ylabel('settled-fatal transitions')
  ax[2].set_title('5C: per-arm contribution')
  fig.tight_layout()
  fig.savefig(os.path.join(args.out, 'diversity_audit.png'), dpi=140)
  plt.close(fig)

  print('\n5A new->old state distance: median %.3f (old-pool NN scale %.3f)'
        % (a5['new_to_old_state']['median'], scale_s))
  print('   beyond old NN scale: %.3f | beyond 2x: %.3f'
        % (frac_beyond, a5['frac_new_beyond_2x_old_nn_scale_state']))
  print('5B dev16   d_old %.3f -> d_diverse %.3f (improved %.0f%%)'
        % (b16['d_old_state']['median'], b16['d_diverse_state']['median'],
           100 * b16['frac_anchors_improved']))
  print('5B final39 d_old %.3f -> d_diverse %.3f (improved %.0f%%)  '
        '[reused dev diagnostic]'
        % (b39['d_old_state']['median'], b39['d_diverse_state']['median'],
           100 * b39['frac_anchors_improved']))
  u = b39['previously_uncovered']
  print('   previously UNCOVERED (n=%d): %.3f -> %.3f, median reduction '
        '%.3f, improved %.0f%%'
        % (u['n'], u['d_old_state']['median'], u['d_diverse_state']['median'],
           u['delta_reduction_state']['median'], 100 * u['frac_improved']))
  print('\n5C per arm:')
  for k, v in per_arm.items():
    if v.get('n_fatal'):
      print('   %-16s n=%4d | nearest-old median %.3f | beyond scale %.2f | '
            'unique vs all others %.2f'
            % (k, v['n_fatal'], v['nearest_old_state_distance']['median'],
               v['frac_beyond_old_nn_scale'],
               v['frac_unique_beyond_scale_vs_all_others']))
  print('\nDIVERSITY GATE: %s' % ('PASS' if gate['PASS'] else 'FAIL'))
  for k, v in gate['criteria'].items():
    print('   %-46s %s' % (k, v))
  print('saved -> %s' % args.out)


if __name__ == '__main__':
  main()
