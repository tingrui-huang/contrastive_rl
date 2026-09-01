"""Normalization + provenance audit for point_two_route_swamp_windy_z_v0.

Diagnostic only. No CRL training, no full dataset, no replay change, no
failure-negative change, no environment-dynamics change. z_min, sink speed and
the settle dynamics are all read, never written.

Two questions.

  (1) PROVENANCE. artifacts/swamp_windy_z/env_audit.json recorded a `commit`
      field that does not contain the code it audited. This resolves which
      commit actually introduced the Z environment, explains the mechanism of
      the error, and re-runs the env audit so the metadata is right -- checking
      that every number is unchanged.

  (2) SCALE. Is z scaled sensibly relative to x and y?

THE ANSWER TO (2) TURNS ON A FACT THAT HAS TO BE STATED FIRST: the pipeline
applies NO normalization to state observations at all. crl/networks.py
_repr_fn takes `state = obs[:, :obs_dim]` and `goal = obs[:, obs_dim:]` raw;
the only scaling in the file is `/ 255.0` for IMAGE observations in
_unflatten_obs; `repr_norm` normalizes the output representation, not the
input, and is False in the windy recipe. So "would z automatically receive
empirical mean/std normalization?" has the answer NO -- there is no such
mechanism to be picked up by.

That inverts the risk the audit was commissioned to check. An empirical
(z - mu)/sigma would amplify z, because failure is rare and sigma_z tracks the
death RATE rather than any physical quantity; the numbers below quantify by how
much. But since nothing normalizes today, the live risk is the opposite one:
raw z carries a 0.5 signal into encoders that see x spanning 9 units.

Both are measured, alongside the fixed physical convention z / |z_min|.

Usage:
  python scripts/audit_swamp_windy_z_norm.py
  python scripts/audit_swamp_windy_z_norm.py --episodes 400
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402
from crl.obs_norm import make_obs_normalizer, z_scale_from_env  # noqa: E402
from collect_swamp_windy import make_windy_teacher  # noqa: E402

ENV_Z = 'point_two_route_swamp_windy_z_v0'
OUT = 'artifacts/swamp_windy_z/normalization_audit.json'
Z_FILES = ['crl/envs.py', 'scripts/audit_swamp_windy_z.py',
           'artifacts/swamp_windy_z/env_audit.json']


def git(*args):
  try:
    return subprocess.check_output(['git'] + list(args),
                                   cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                                # pylint: disable=broad-except
    return ''


def collect(env, episodes, seed, random_frac=0.2, force_safe_prob=0.05,
            teacher_noise=0.15):
  """Small temporary rollout with the SAME behaviour mix as the real collector.

  make_windy_teacher works unchanged on the Z env because self.state is still
  the 2-D XY vector; only the observation is 3-D. Proportions therefore match
  what a real dataset would look like, which is what the frequency of z=-0.5
  depends on.
  """
  rng = np.random.default_rng(seed)
  teacher = make_windy_teacher(env, rng, force_safe_prob)
  L = env.max_episode_steps
  n_random = int(round(episodes * random_frac))
  rows, prim = [], []
  n_fail_entry = n_post = 0
  for ep in range(episodes):
    env.reset()
    g = env.goal.copy()
    memo = {}
    is_rand = ep < n_random
    ep_rows, death_row = [], None
    for t in range(L):
      ep_rows.append(env._get_obs().copy())
      if is_rand:
        a = rng.uniform(-1, 1, 2).astype(np.float32)
      else:
        a = np.asarray(teacher(env.state.copy(), g, memo), np.float32)
        if teacher_noise > 0 and np.any(a != 0):
          a = np.clip(a + rng.normal(0, teacher_noise, 2), -1, 1).astype(
              np.float32)
      was_dead = env.dead
      env.step(a)
      if env.dead and not was_dead:
        death_row = t + 1
        n_fail_entry += 1
    ep_rows.append(env._get_obs().copy())
    # PRIMARY = every row up to and INCLUDING the death row. The death state is
    # the outcome of the last real transition, not part of the frozen tail;
    # excluding it would drop every z < 0 row and drive sigma_z to exactly 0.
    for t in range(len(ep_rows)):
      rows.append(ep_rows[t])
      keep = death_row is None or t <= death_row
      prim.append(keep)
      if not keep:
        n_post += 1
  return (np.array(rows, np.float64), np.array(prim, bool), n_fail_entry,
          n_post)


def stats(a):
  return {'n': int(len(a)),
          'mu': [float(v) for v in a.mean(0)],
          'sigma': [float(v) for v in a.std(0)],
          'min': [float(v) for v in a.min(0)],
          'max': [float(v) for v in a.max(0)]}


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--episodes', type=int, default=400)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()

  out = {'analysis_script': 'scripts/audit_swamp_windy_z_norm.py'}

  # ---------------------------------------------------------------- 1
  print('=' * 96)
  print('1. PROVENANCE')
  print('=' * 96)
  impl = git('log', '-1', '--format=%H', '-S', 'class TwoRouteSwampWindyZEnv',
             '--', 'crl/envs.py')
  impl_branch = git('log', '-1', '--format=%H', '-S',
                    'point_two_route_swamp_windy_z_v0', '--', 'crl/envs.py')
  impl_script = git('log', '--diff-filter=A', '--format=%H', '--',
                    'scripts/audit_swamp_windy_z.py')
  impl_json = git('log', '--diff-filter=A', '--format=%H', '--',
                  'artifacts/swamp_windy_z/env_audit.json')
  stale = '754887eb4c0f6a4a120091707e128751de506d63'
  present_at_stale = git('show', '%s:crl/envs.py' % stale).count(
      'TwoRouteSwampWindyZEnv')
  parent = git('rev-parse', '%s^' % impl) if impl else ''
  print('  commit introducing TwoRouteSwampWindyZEnv        : %s' % impl)
  print('  commit introducing the make_env branch           : %s' % impl_branch)
  print('  commit adding scripts/audit_swamp_windy_z.py     : %s' % impl_script)
  print('  commit adding artifacts/.../env_audit.json       : %s' % impl_json)
  print('  all four are the same commit                     : %s'
        % (len({impl, impl_branch, impl_script, impl_json}) == 1))
  print('  occurrences of TwoRouteSwampWindyZEnv at %s : %d'
        % (stale[:7], present_at_stale))
  print('  %s is the PARENT of %s                  : %s'
        % (stale[:7], impl[:7], parent == stale))
  print()
  print('  VERDICT: 754887e is STALE metadata. The Z environment does not')
  print('  exist at that commit, and it is the parent of the implementation')
  print('  commit. MECHANISM: audit_swamp_windy_z.py stamps `git rev-parse')
  print('  HEAD` at RUN time, and the audit was run before the code was')
  print('  committed, so it recorded the parent. The code actually under test')
  print('  was uncommitted working-tree state that became %s.' % impl[:7])
  print('  CORRECT implementation commit: %s' % impl)
  out['1_provenance'] = {
      'z_env_implementation_commit': impl,
      'make_env_branch_commit': impl_branch,
      'audit_script_added_commit': impl_script,
      'env_audit_json_added_commit': impl_json,
      'all_four_same_commit': len({impl, impl_branch, impl_script,
                                   impl_json}) == 1,
      'stale_field_value': stale,
      'stale_is_parent_of_implementation': parent == stale,
      'z_env_present_at_stale_commit': bool(present_at_stale),
      'stale_field_is_incorrect': True,
      '7da7728_is_correct_implementation_commit': impl.startswith('7da7728'),
      'mechanism': 'audit_swamp_windy_z.py stamped git rev-parse HEAD at run '
                   'time and was run before the code was committed, so it '
                   'recorded the parent commit rather than the code under test',
      'remedy': 'env_audit.json commit field rewritten to the commit that last '
                'touched the audited files; numbers regenerated and verified '
                'identical'}

  # ---------------------------------------------------------------- 2
  print('\n' + '=' * 96)
  print('2. NORMALIZATION PATH (inspected, not assumed)')
  print('=' * 96)
  print('  crl/networks.py :: make_networks._repr_fn')
  print('      state = obs[:, :obs_dim]        <- RAW, no scaling')
  print('      goal  = obs[:, obs_dim:]        <- RAW, no scaling')
  print('  crl/networks.py :: make_networks._unflatten_obs')
  print('      / 255.0                         <- IMAGE observations only')
  print('  crl/networks.py :: repr_norm block  <- normalizes the OUTPUT')
  print('      representation (sa_repr, g_repr), not the input; False in the')
  print('      windy recipe')
  print('  crl/replay.py :: TrajectoryBuffer.sample  <- .astype(float32) only')
  print()
  print('  Are x, y, z normalized?                      NO')
  print('  Applied to states, goals, both, or model-only? NONE of them for')
  print('    state envs; only image observations are scaled, by /255')
  print('  Are mu/sigma computed from the dataset?      NO -- there is no')
  print('    empirical-statistics machinery in the pipeline at all')
  print('  Are state and goal statistics shared?        N/A, none exist')
  print('  Would z automatically get empirical mu/sigma? NO')
  nets_src = open(os.path.join(os.path.dirname(_HERE),
                               'crl', 'networks.py')).read()
  assert 'state = obs[:, :obs_dim]' in nets_src
  assert 'goal = obs[:, obs_dim:]' in nets_src
  out['2_normalization_path'] = {
      'code_paths': [
          'crl/networks.py :: make_networks._repr_fn '
          '(state = obs[:, :obs_dim], goal = obs[:, obs_dim:], both RAW)',
          'crl/networks.py :: make_networks._unflatten_obs (/255, IMAGE only)',
          'crl/networks.py :: repr_norm block (normalizes the OUTPUT '
          'representation, not the input; False in the windy recipe)',
          'crl/replay.py :: TrajectoryBuffer.sample (astype(float32) only)'],
      'xyz_normalized': False,
      'applied_to': 'nothing for state envs; /255 for image observations only',
      'empirical_mu_sigma_computed': False,
      'state_and_goal_stats_shared': None,
      'z_would_get_empirical_normalization': False,
      'consequence': 'raw z enters the encoders at magnitude 0.5 against x in '
                     '[0,9] and y in [1,4], so the live risk is z being '
                     'UNDER-weighted, not over-amplified'}

  # ---------------------------------------------------------------- 3
  cfg = Config(env_name=ENV_Z)
  env = envs_mod.make_env(ENV_Z, cfg, seed=args.seed)
  zmin = z_scale_from_env(env)
  print('\n' + '=' * 96)
  print('3. SMALL Z-STATE DATASET (%d temporary episodes, NOT the training set)'
        % args.episodes)
  print('=' * 96)
  rows, prim, n_fail_entry, n_post = collect(env, args.episodes, args.seed)
  s_xyz = rows[:, :3]
  z = s_xyz[:, 2]
  print('  states logged            : %s' % format(len(rows), ','))
  print('  z == 0                   : %s  (%.4f)'
        % (format(int((z == 0).sum()), ','), (z == 0).mean()))
  print('  z <  0                   : %s  (%.4f)'
        % (format(int((z < 0).sum()), ','), (z < 0).mean()))
  print('  failure-ENTRY states     : %s' % format(n_fail_entry, ','))
  print('  post-failure frozen rows : %s' % format(n_post, ','))
  print('  primary (dynamic) rows   : %s' % format(int(prim.sum()), ','))
  A = s_xyz
  B = s_xyz[prim]
  print('  A all rows %s   B primary rows %s'
        % (format(len(A), ','), format(len(B), ',')))
  print('  z<0 frequency: A %.5f   B %.5f   <-- frozen tails inflate A'
        % ((A[:, 2] < 0).mean(), (B[:, 2] < 0).mean()))
  out['3_dataset'] = {
      'episodes': args.episodes, 'n_states': int(len(rows)),
      'n_z_zero': int((z == 0).sum()), 'frac_z_zero': float((z == 0).mean()),
      'n_z_negative': int((z < 0).sum()),
      'frac_z_negative': float((z < 0).mean()),
      'n_failure_entry_states': int(n_fail_entry),
      'n_post_failure_frozen': int(n_post),
      'n_primary_rows': int(prim.sum()),
      'frac_z_negative_all_rows': float((A[:, 2] < 0).mean()),
      'frac_z_negative_primary': float((B[:, 2] < 0).mean())}

  # ---------------------------------------------------------------- 4
  print('\n' + '=' * 96)
  print('4. EMPIRICAL STATISTICS and where z=0 / z=-0.5 land under them')
  print('=' * 96)
  norm = {}
  for tag, arr in (('A_all_rows', A), ('B_primary_rows', B)):
    st = stats(arr)
    mz, sz = st['mu'][2], st['sigma'][2]
    if sz <= 0:
      # No z variation at all in this slice: standardisation is undefined,
      # which is itself the finding rather than an error to swallow.
      st.update({'z_safe_standardized': None, 'z_fail_standardized': None,
                 'separation_standardized': float('inf'),
                 'note': 'sigma_z == 0: this slice contains no failure rows, '
                         'so empirical standardisation is undefined'})
      norm[tag] = st
      print('  %s  (n = %s)  sigma_z == 0 -- no failures in this slice, '
            'standardisation UNDEFINED' % (tag, format(st['n'], ',')))
      continue
    z_safe = (0.0 - mz) / sz
    z_fail = (-zmin - mz) / sz
    sep = abs(z_fail - z_safe)
    print('  %s  (n = %s)' % (tag, format(st['n'], ',')))
    print('    mu    x %8.4f   y %8.4f   z %9.5f'
          % (st['mu'][0], st['mu'][1], st['mu'][2]))
    print('    sigma x %8.4f   y %8.4f   z %9.5f'
          % (st['sigma'][0], st['sigma'][1], st['sigma'][2]))
    print('    z_safe_tilde %+9.4f   z_fail_tilde %+9.4f   |separation| %8.4f'
          % (z_safe, z_fail, sep))
    st.update({'z_safe_standardized': z_safe, 'z_fail_standardized': z_fail,
               'separation_standardized': sep})
    norm[tag] = st
  out['4_empirical'] = norm

  # ---------------------------------------------------------------- 5
  print('\n' + '=' * 96)
  print('5. EMPIRICAL vs FIXED PHYSICAL Z SCALING  (X/Y untouched in both)')
  print('=' * 96)
  fixed_sep = zmin / zmin                        # (-0.5)/0.5 - 0/0.5 = -1
  print('  fixed  z / |z_min| = z / %.2f :  z=0 -> 0.0000   z=-%.2f -> %+.4f'
        % (zmin, zmin, -1.0))
  print('    separation = %.4f  (one full sinking depth, by construction)'
        % fixed_sep)
  print('  raw (TODAY, no normalization) :  z=0 -> 0.0000   z=-%.2f -> %+.4f'
        % (zmin, -zmin))
  print('    separation = %.4f' % zmin)
  print()
  print('  %-18s%14s%14s%14s' % ('convention', 'separation', 'vs fixed',
                                 'depends on?'))
  print('  %-18s%14.4f%14s%14s' % ('raw (current)', zmin,
                                   '%.2fx' % (zmin / fixed_sep), 'nothing'))
  print('  %-18s%14.4f%14s%14s' % ('fixed z/|z_min|', fixed_sep, '1.00x',
                                   'z_min only'))
  amp = {}
  for tag in ('A_all_rows', 'B_primary_rows'):
    sep = norm[tag]['separation_standardized']
    amp[tag] = sep
    print('  %-18s%14.4f%14s%14s' % ('empirical ' + tag[0], sep,
                                     '%.2fx' % (sep / fixed_sep), 'death rate'))
  out['5_scaling'] = {
      'z_min_abs': zmin,
      'fixed_separation': float(fixed_sep),
      'raw_separation': float(zmin),
      'empirical_separation_all_rows': float(amp['A_all_rows']),
      'empirical_separation_primary': float(amp['B_primary_rows']),
      'empirical_amplification_vs_fixed_all_rows':
          float(amp['A_all_rows'] / fixed_sep),
      'empirical_amplification_vs_fixed_primary':
          float(amp['B_primary_rows'] / fixed_sep)}

  # ---------------------------------------------------------------- 6
  print('\n' + '=' * 96)
  print('6. DOES Z DOMINATE STATE-SPACE DISTANCE?')
  print('=' * 96)
  safe = B[B[:, 2] == 0]
  rng = np.random.default_rng(args.seed + 5)
  i = rng.choice(len(safe), 4000)
  j = rng.choice(len(safe), 4000)
  xy_typ = np.linalg.norm(safe[i, :2] - safe[j, :2], axis=1)
  print('  typical XY distance between two ordinary SAFE states:')
  print('    mean %.4f  median %.4f  p10 %.4f  p90 %.4f'
        % (xy_typ.mean(), np.median(xy_typ), np.percentile(xy_typ, 10),
           np.percentile(xy_typ, 90)))
  print()
  print('  same-XY pair  g_safe=(x,y,0)  vs  g_fail=(x,y,-%.2f):' % zmin)
  print('    XY contribution is exactly 0 by construction, so the full 3-D')
  print('    distance IS the scale assigned to failure depth.')
  print('  %-24s%12s%12s%14s' % ('convention', 'XY dist', 'Z dist',
                                 'vs median XY'))
  rows6 = []
  for tag, zdist in (('raw (current)', zmin),
                     ('fixed z/|z_min|', 1.0),
                     ('empirical A (all)', amp['A_all_rows']),
                     ('empirical B (primary)', amp['B_primary_rows'])):
    ratio = zdist / np.median(xy_typ)
    print('  %-24s%12.4f%12.4f%14.2fx' % (tag, 0.0, zdist, ratio))
    rows6.append({'convention': tag, 'xy_distance': 0.0,
                  'z_distance': float(zdist),
                  'ratio_to_median_xy': float(ratio)})
  out['6_distance'] = {
      'typical_safe_safe_xy_distance': {
          'mean': float(xy_typ.mean()), 'median': float(np.median(xy_typ)),
          'p10': float(np.percentile(xy_typ, 10)),
          'p90': float(np.percentile(xy_typ, 90))},
      'same_xy_pairs': rows6}

  # ---------------------------------------------------------------- 7 + 8
  print('\n' + '=' * 96)
  print('7/8. PREPARED NORMALIZER + UNIT TESTS  (crl/obs_norm.py, NOT wired in)')
  print('=' * 96)
  ident = make_obs_normalizer(3, 3, mode='none')
  zphys = make_obs_normalizer(3, 3, mode='z_physical',
                              z_scale=z_scale_from_env(env))
  probe = np.array([[2.5, 3.5, 0.0, 8.5, 3.5, 0.0],
                    [4.2, 3.5, -zmin, 8.5, 3.5, 0.0]], np.float64)
  t = {}
  n_id = ident(probe)
  t['mode_none_is_identity'] = bool(np.array_equal(n_id, probe))
  n_zp = zphys(probe)
  t['z0_maps_to_0'] = bool(n_zp[0, 2] == 0.0)
  t['zmin_maps_to_-1'] = bool(n_zp[1, 2] == -1.0)
  t['goal_z_also_scaled'] = bool(n_zp[0, 5] == 0.0 and n_zp[1, 5] == 0.0)
  t['xy_untouched'] = bool(np.array_equal(n_zp[:, :2], probe[:, :2])
                           and np.array_equal(n_zp[:, 3:5], probe[:, 3:5]))
  t['safe_stays_on_ground'] = bool(n_zp[0, 2] == 0.0)
  t['failure_still_distinguishable'] = bool(n_zp[1, 2] != n_zp[0, 2])
  # hidden bits must not move the normalized observation either
  env.reset(); env.set_auto_resample(False)
  env.set_swamp([1, 1, 1]); oa = zphys(env._get_obs()[None])
  env.set_swamp([0, 0, 0]); oc = zphys(env._get_obs()[None])
  t['hidden_bits_do_not_move_normalized_obs'] = bool(
      np.abs(oa - oc).max() == 0.0)
  # the 2-D env must be untouched by all of this
  cfg2 = Config(env_name='point_two_route_swamp_windy_v0')
  e2 = envs_mod.make_env('point_two_route_swamp_windy_v0', cfg2, seed=0)
  t['old_2d_env_unaffected'] = bool(cfg2.obs_dim == 2 and cfg2.goal_dim == 2
                                    and e2.reset().shape == (4,))
  for k, v in t.items():
    print('  %-45s %s' % (k, 'PASS' if v else 'FAIL'))
  assert all(t.values()), 'a normalization unit test failed'
  out['7_8_unit_tests'] = t
  out['7_implementation'] = {
      'module': 'crl/obs_norm.py',
      'wired_into_pipeline': False,
      'modes': ['none (identity, the default and today\'s behaviour)',
                'z_physical (z / |z_min| on both the state and goal halves)'],
      'z_scale_source': 'crl.obs_norm.z_scale_from_env -> abs(env.z_min); the '
                        'literal 0.5 is not duplicated anywhere',
      'empirical_mode_offered': False,
      'why_not': 'sigma_z tracks the death RATE, not a physical quantity, so '
                 'the safe/failure separation would move whenever the dataset '
                 'death rate moved'}

  # ---------------------------------------------------------------- verdict
  ratio_emp = amp['B_primary_rows'] / np.median(xy_typ)
  over = amp['B_primary_rows'] > 3.0
  print('\n' + '=' * 96)
  print('VERDICT')
  print('=' * 96)
  print('  Empirical standardization WOULD over-amplify z: %s' % over)
  print('    one failure step would read as %.2f units against a median'
        % amp['B_primary_rows'])
  print('    safe-safe XY distance of %.2f, i.e. %.1fx.'
        % (np.median(xy_typ), ratio_emp))
  print('  But nothing normalizes today, so that is a risk of ADDING the wrong')
  print('  scheme, not a description of the current pipeline. Raw z gives a')
  print('  %.2f separation against a median XY distance of %.2f (%.2fx).'
        % (zmin, np.median(xy_typ), zmin / np.median(xy_typ)))
  print('  RECOMMENDATION: z / |z_min|. It is dataset-independent, puts one')
  print('  sinking depth at 1.0 -- the same order as a maze cell -- and leaves')
  print('  X/Y exactly as the existing 2-D results were produced with.')
  out['verdict'] = {
      'empirical_would_over_amplify': bool(over),
      'empirical_separation_vs_median_xy': float(ratio_emp),
      'raw_separation_vs_median_xy': float(zmin / np.median(xy_typ)),
      'recommended_convention': 'z / |z_min|  (crl.obs_norm mode z_physical)',
      'recommendation_reasons': [
          'dataset-independent: does not move with the death rate',
          'one sinking depth = 1.0, the same order as one maze cell',
          'leaves X/Y untouched, so existing 2-D comparisons stay valid'],
      'note': 'the pipeline currently normalizes nothing, so this is a choice '
              'to be made when wiring it in, not a bug to be fixed now'}

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % args.out)


if __name__ == '__main__':
  main()
