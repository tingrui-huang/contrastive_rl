"""Validation audit for point_two_route_swamp_windy_z_v1.

Environment design + audit only. No dataset regeneration, no failure-bank
rebuild, no CRL training, no replay or loss changes.

The claim under test: sinking is now a genuine continuous physical trajectory
across ENV STEPS -- so a learner-visible dataset contains the intermediate
depths, not just {0, -0.5} -- while the hidden swamp bit stays hidden before
contact and z_v0 remains untouched.

  9  paired clear-vs-active FIRST CONTACT, >= 500 pairs, matched action-noise
     RNG. XY must be identical; z' must be 0 vs approximately -0.12, NOT -0.5.
  10 multi-step sinking from a forced contact, printed step by step.
  11 small temporary rollout, Z histogram over the intermediate depths.
  12 same-XY separations, raw and after the accepted physical scaling.
  13 regression: 2-D env, z_v0 (still 0 -> -0.5 in ONE step) and z_v1.

Usage:
  python scripts/audit_swamp_windy_z_v1.py
  python scripts/audit_swamp_windy_z_v1.py --episodes 400 --pairs 600
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

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402
from crl.obs_norm import obs_scale_vector, z_scale_from_env  # noqa: E402
from collect_swamp_windy import make_windy_teacher  # noqa: E402

ENV_V1 = 'point_two_route_swamp_windy_z_v1'
ENV_V0 = 'point_two_route_swamp_windy_z_v0'
ENV_2D = 'point_two_route_swamp_windy_v0'
OUT = 'artifacts/swamp_windy_z_v1/env_audit.json'
TRACKED = ['crl/envs.py', 'scripts/audit_swamp_windy_z_v1.py']


def git(*a):
  try:
    return subprocess.check_output(['git'] + list(a),
                                   cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                                # pylint: disable=broad-except
    return ''


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--episodes', type=int, default=400)
  ap.add_argument('--pairs', type=int, default=600)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()

  code_commit = git('log', '-1', '--format=%H', '--', *TRACKED)
  head = git('rev-parse', 'HEAD')
  dirty = bool(git('status', '--porcelain', '--', *TRACKED))

  cfg = Config(env_name=ENV_V1)
  env = envs_mod.make_env(ENV_V1, cfg, seed=args.seed)
  dz = env.sink_step
  zmin = env.z_min
  out = {'env_name': ENV_V1,
         'env_code_path': 'crl/envs.py :: TwoRouteSwampWindyZV1Env',
         'analysis_script': 'scripts/audit_swamp_windy_z_v1.py',
         'git_state': {'code_commit': code_commit, 'head_at_runtime': head,
                       'audited_files_dirty': dirty, 'tracked_files': TRACKED,
                       'note': 'code_commit is the last commit touching the '
                               'audited files, NOT git HEAD at run time'}}

  print('=' * 92)
  print('ENV AUDIT -- %s' % ENV_V1)
  print('=' * 92)
  print('  code commit %s%s' % (code_commit,
                                '   (WORKING TREE DIRTY)' if dirty else ''))
  print('  head        %s' % head)
  o0 = env.reset()
  print('  obs_dim %d  goal_dim %d  flat obs %s   start_index %d end_index %d'
        % (cfg.obs_dim, cfg.goal_dim, o0.shape, cfg.start_index,
           cfg.end_index))
  print('  reset obs   %s' % np.round(o0, 4))
  print('  Z UPDATE    z <- max(z_min, z - sink_speed*sink_dt) ONCE PER ENV '
        'STEP')
  print('    sink_speed %.3f  sink_dt %.3f  =>  dz %.4f   z_min %.3f   '
        'steps_to_settle %d' % (env.sink_speed, env.sink_dt, dz, zmin,
                                env.steps_to_settle))
  assert cfg.obs_dim == 3 and cfg.goal_dim == 3 and o0.shape == (6,)
  assert o0[2] == 0.0 and o0[5] == 0.0
  out['shapes'] = {'obs_dim': 3, 'goal_dim': 3, 'flat_obs': [6],
                   'layout': '[x, y, z, g_x, g_y, g_z]',
                   'start_index': int(cfg.start_index),
                   'end_index': int(cfg.end_index),
                   'task_goal': [8.5, 3.5, 0.0]}
  out['z_update'] = {
      'rule': 'z <- max(z_min, z - sink_speed*sink_dt), applied ONCE PER ENV '
              'STEP (on first contact and on every subsequent step)',
      'sink_speed': env.sink_speed, 'sink_dt': env.sink_dt,
      'delta_z_per_env_step': float(dz), 'z_min': float(zmin),
      'steps_to_settle': int(env.steps_to_settle),
      'expected_sequence': [0.0] + [float(max(zmin, -dz * k))
                                    for k in range(1, env.steps_to_settle + 1)]}

  # ------------------------------------------------------------------ 2/6
  print('\n  hidden-bit invariance BEFORE contact')
  env.reset(); env.set_auto_resample(False)
  env.state = np.array([2.5, 3.5])
  env.set_swamp([1, 1, 1]); oa = env._get_obs().copy()
  env.set_swamp([0, 0, 0]); oc = env._get_obs().copy()
  leak = float(np.abs(oa - oc).max())
  print('    flipping every swamp bit changes the observation by %.3e '
        '(z=%.4f both ways)' % (leak, oa[2]))
  assert leak == 0.0 and oa[2] == 0.0
  out['hidden_bit_invariance'] = {'obs_delta': leak, 'z_before_contact': 0.0}

  # -------------------------------------------------------------------- 9
  print('\n' + '=' * 92)
  print('9. PAIRED CLEAR-vs-ACTIVE FIRST CONTACT (%d pairs, matched noise RNG)'
        % args.pairs)
  print('=' * 92)
  rng = np.random.default_rng(args.seed)
  entry = {0: (2.5, 3.5), 1: (3.5, 3.5), 2: (4.5, 3.5)}
  rows = []
  for k in range(args.pairs):
    c = k % 3
    sx, sy = entry[c]
    s0 = np.array([sx + rng.uniform(-0.15, 0.15),
                   sy + rng.uniform(-0.25, 0.25)])
    act = np.array([rng.uniform(0.6, 1.0), rng.uniform(-0.2, 0.2)], np.float32)
    nseed = int(rng.integers(1 << 30))
    leg = {}
    for nm in ('clear', 'active'):
      bits = [0, 0, 0]
      if nm == 'active':
        bits[c] = 1
      env.reset(); env.set_auto_resample(False); env.set_swamp(bits)
      env.state = s0.copy(); env._z = 0.0; env._dead = False
      env._rng = np.random.default_rng(nseed)
      o, _, _, _ = env.step(act)
      leg[nm] = {'obs': o.copy(), 'dead': bool(env.dead)}
    rows.append({'entered': leg['active']['dead'],
                 'dxy': float(np.linalg.norm(leg['clear']['obs'][:2]
                                             - leg['active']['obs'][:2])),
                 'z_clear': float(leg['clear']['obs'][2]),
                 'z_active': float(leg['active']['obs'][2])})
  ent = [r for r in rows if r['entered']]
  dxy = np.array([r['dxy'] for r in ent])
  zc = np.array([r['z_clear'] for r in ent])
  za = np.array([r['z_active'] for r in ent])
  print('  %d pairs, %d ENTER the forced cell' % (len(rows), len(ent)))
  print('    ||xy_clear - xy_active||  max %.3e            <-- must be 0'
        % dxy.max())
  print('    z_clear                   all %.4f            <-- must be 0'
        % zc.max())
  print('    z_active                  min %.4f max %.4f   <-- must be ~%.2f, '
        'NOT %.2f' % (za.min(), za.max(), -dz, zmin))
  assert dxy.max() < 1e-6
  assert (zc == 0).all()
  assert np.allclose(za, -dz, atol=1e-6), 'first contact is not ONE increment'
  assert not np.isclose(za, zmin, atol=1e-6).any(), (
      'first contact jumped straight to z_min -- that is v0 behaviour')
  out['9_paired_first_contact'] = {
      'n_pairs': len(rows), 'n_entering': len(ent),
      'xy_distance_max': float(dxy.max()),
      'z_clear_unique': sorted(set(float(v) for v in zc)),
      'z_active_min': float(za.min()), 'z_active_max': float(za.max()),
      'z_active_equals_one_increment': True,
      'z_active_is_not_z_min': True}

  # ------------------------------------------------------------------- 10
  print('\n' + '=' * 92)
  print('10. MULTI-STEP SINKING FROM A FORCED CONTACT')
  print('=' * 92)
  env.reset(); env.set_auto_resample(False); env.set_swamp([1, 0, 0])
  env.state = np.array([2.5, 3.5]); env._z = 0.0; env._dead = False
  print('  %-22s %-28s %s' % ('phase', 'obs [x,y,z,gx,gy,gz]', 'dead'))
  print('  %-22s %-28s %s' % ('before contact',
                              np.round(env._get_obs(), 4), env.dead))
  seq = [float(env._get_obs()[2])]
  xys = []
  # deliberately vary the action after contact: it must change nothing
  acts = [np.array([1.0, 0.0], np.float32)] + \
         [np.array([-1.0, 0.9], np.float32)] * 7
  for i, a in enumerate(acts):
    o, r, _, _ = env.step(a)
    seq.append(float(o[2]))
    xys.append(o[:2].copy())
    lbl = 'first contact' if i == 0 else 'env step +%d' % i
    print('  %-22s %-28s %s' % (lbl, np.round(o, 4), env.dead))
  seq_arr = np.array(seq)
  xy_arr = np.array(xys)
  mono = bool(np.all(np.diff(seq_arr) <= 1e-9))
  frozen_xy = float(np.abs(xy_arr - xy_arr[0]).max())
  print('\n  monotonic non-increasing : %s' % mono)
  print('  never below z_min        : %s (min %.4f, z_min %.4f)'
        % (bool(seq_arr.min() >= zmin - 1e-9), seq_arr.min(), zmin))
  print('  XY frozen after contact  : max drift %.3e (actions were varied and '
        'even reversed)' % frozen_xy)
  print('  settled value repeats    : %s' % (abs(seq[-1] - seq[-2]) < 1e-12))
  assert mono and seq_arr.min() >= zmin - 1e-9 and frozen_xy < 1e-9
  out['10_multistep'] = {'z_sequence': seq, 'monotonic_non_increasing': mono,
                         'min_z': float(seq_arr.min()), 'z_min': float(zmin),
                         'xy_max_drift_after_contact': frozen_xy,
                         'actions_after_contact_varied': True}

  # ------------------------------------------------------------------- 11
  print('\n' + '=' * 92)
  print('11. SMALL ROLLOUT (%d temporary episodes, NOT a training dataset)'
        % args.episodes)
  print('=' * 92)
  e2 = envs_mod.make_env(ENV_V1, Config(env_name=ENV_V1), seed=args.seed)
  rr = np.random.default_rng(args.seed + 1)
  teacher = make_windy_teacher(e2, rr, 0.05)
  L = e2.max_episode_steps
  n_rand = int(round(args.episodes * 0.2))
  zs, n_fail, n_first, n_mid, n_settled = [], 0, 0, 0, 0
  for ep in range(args.episodes):
    e2.reset()
    g2 = e2.goal.copy()
    memo = {}
    prev_z = 0.0
    for t in range(L):
      zs.append(float(e2.z))
      a = (rr.uniform(-1, 1, 2).astype(np.float32) if ep < n_rand
           else np.clip(np.asarray(teacher(e2.state.copy(), g2, memo),
                                   np.float32)
                        + rr.normal(0, 0.15, 2), -1, 1).astype(np.float32))
      was = e2.dead
      e2.step(a)
      z = float(e2.z)
      if not was and e2.dead:
        n_fail += 1
        n_first += 1
      elif e2.dead:
        if z < prev_z - 1e-12:
          n_mid += 1
        else:
          n_settled += 1
      prev_z = z
    zs.append(float(e2.z))
  zs = np.array(zs)
  hist = collections.Counter(np.round(zs, 4))
  print('  episodes %d   failed %d (%.4f)' % (args.episodes, n_fail,
                                              n_fail / args.episodes))
  print('  first-contact rows %d   intermediate sinking rows %d   '
        'settled absorbing rows %d' % (n_first, n_mid, n_settled))
  print('  Z histogram (learner-visible):')
  for v, c in sorted(hist.items(), reverse=True):
    print('    z = %+7.4f : %7d  (%.5f)' % (v, c, c / len(zs)))
  print('  distinct Z values %d   min %.4f  max %.4f'
        % (len(hist), zs.min(), zs.max()))
  assert len(hist) > 2, 'z_v1 still only has two levels -- v0 behaviour'
  out['11_rollout'] = {
      'episodes': args.episodes, 'n_states': int(len(zs)),
      'failed_episodes': n_fail,
      'first_contact_rows': n_first, 'intermediate_sinking_rows': n_mid,
      'settled_absorbing_rows': n_settled,
      'z_histogram': {str(float(v)): int(c) for v, c in sorted(hist.items(),
                                                              reverse=True)},
      'n_distinct_z': len(hist), 'z_min_observed': float(zs.min()),
      'z_max_observed': float(zs.max())}

  # ------------------------------------------------------------------- 12
  print('\n' + '=' * 92)
  print('12. SAME-XY SEPARATION, raw and after the accepted physical scaling')
  print('=' * 92)
  scale = obs_scale_vector(3, 3, 'z_physical', z_scale_from_env(env))
  print('  obs_scale %s   (1/|z_min| on the z column of BOTH halves, applied '
        'once)' % scale.tolist())
  assert scale[2] == scale[5] == 1.0 / abs(zmin)
  assert np.array_equal(scale[[0, 1, 3, 4]], np.ones(4)), 'X/Y were scaled'
  seps = {}
  print('  %-34s%12s%12s' % ('pair (same XY)', 'raw', 'scaled'))
  for nm, zv in (('safe vs early failure (-%.2f)' % dz, -dz),
                 ('safe vs settled failure (%.2f)' % zmin, zmin)):
    raw = abs(0.0 - zv)
    sc = abs(0.0 - zv / abs(zmin))
    print('  %-34s%12.4f%12.4f' % (nm, raw, sc))
    seps[nm] = {'raw': float(raw), 'scaled': float(sc)}
  print('  every intermediate depth, scaled: %s'
        % [round(-abs(v) / abs(zmin), 4)
           for v in out['z_update']['expected_sequence'][1:]])
  out['12_separation'] = {'obs_scale': scale.tolist(), 'pairs': seps,
                          'scaled_sequence':
                              [float(v / abs(zmin))
                               for v in out['z_update']['expected_sequence']],
                          'empirical_standardization': False}

  # ------------------------------------------------------------------- 13
  print('\n' + '=' * 92)
  print('13. REGRESSION -- z_v0 and the 2-D env must be untouched')
  print('=' * 92)
  reg = {}
  c2 = Config(env_name=ENV_2D)
  o = envs_mod.make_env(ENV_2D, c2, seed=0).reset()
  reg['2d'] = {'obs_shape': list(o.shape), 'obs_dim': c2.obs_dim,
               'goal_dim': c2.goal_dim}
  print('  %-34s obs %s  obs_dim %d' % (ENV_2D, o.shape, c2.obs_dim))
  # z_v0 must STILL jump 0 -> z_min in ONE env step
  c0 = Config(env_name=ENV_V0)
  v0 = envs_mod.make_env(ENV_V0, c0, seed=0)
  v0.reset(); v0.set_auto_resample(False); v0.set_swamp([1, 0, 0])
  v0.state = np.array([2.5, 3.5]); v0._z = 0.0; v0._dead = False
  ov0, _, _, _ = v0.step(np.array([1.0, 0.0], np.float32))
  reg['z_v0'] = {'obs_shape': list(ov0.shape), 'obs_dim': c0.obs_dim,
                 'z_after_first_contact': float(ov0[2]),
                 'jumps_to_z_min_in_one_step': bool(
                     abs(ov0[2] - v0.z_min) < 1e-9)}
  print('  %-34s obs %s  z after first contact %.4f  (== z_min: %s)'
        % (ENV_V0, ov0.shape, ov0[2], reg['z_v0']['jumps_to_z_min_in_one_step']))
  reg['z_v1'] = {'obs_shape': [6], 'obs_dim': 3,
                 'z_after_first_contact': float(-dz),
                 'gradual_across_env_steps': True}
  print('  %-34s obs (6,)  z after first contact %.4f  (gradual)'
        % (ENV_V1, -dz))
  assert reg['2d']['obs_shape'] == [4]
  assert reg['z_v0']['obs_shape'] == [6]
  assert reg['z_v0']['jumps_to_z_min_in_one_step'], (
      'z_v0 no longer jumps to z_min in one step -- it was modified')
  out['13_regression'] = reg
  print('  z_v0 STILL jumps 0 -> %.2f in one env step; z_v1 takes %d steps. '
        'The two are distinct and z_v0 stays reproducible.'
        % (zmin, env.steps_to_settle))

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % args.out)
  print('OVERALL: ALL AUDITS PASS')


if __name__ == '__main__':
  main()
