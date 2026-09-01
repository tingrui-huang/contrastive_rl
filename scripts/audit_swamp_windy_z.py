"""Validation audit for point_two_route_swamp_windy_z_v0 (the (x, y, z) variant).

Environment-design validation only. Nothing is trained, no dataset is
regenerated, no failure bank is rebuilt, and no CRL code is modified.

The claim under test is narrow and precise: adding a vertical sinking depth
makes SAFE and FATAL outcomes distinguishable in the learner-visible state
WITHOUT revealing the hidden swamp bit before contact. Four blocks check it.

  6. Hidden-confounder invariants -- z is exactly 0 before contact regardless
     of the hidden bit, exactly 0 after a clear-swamp contact, and strictly
     negative after a fatal one. Also that the observation carries no swamp
     bit, wait counter or dead flag.

  7. Paired clear-vs-active audit -- the decisive test. The same start state,
     the same action, and the same action-noise draw are replayed twice with
     only the relevant swamp bit forced clear vs active. XY must land in the
     SAME place (the design deliberately preserves XY dynamics) while z
     separates. Matching the noise matters: the env perturbs the action with
     its own RNG inside step(), so without re-seeding between the two legs the
     XY difference would be dominated by that noise rather than by U.

  8. Small rollout audit -- a few hundred temporary episodes, not the training
     dataset. Confirms the invariants hold in the wild and that the XY
     trajectory is IDENTICAL to the 2-D env under a matched seed, which is what
     "XY dynamics unchanged" has to mean operationally.

  9. Replay semantics -- inspects, without changing, whether the current
     future-goal relabeling could hand a z < 0 state back as an ordinary
     POSITIVE goal.

Usage:
  python scripts/audit_swamp_windy_z.py
  python scripts/audit_swamp_windy_z.py --episodes 400 --pairs 600
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

ENV_Z = 'point_two_route_swamp_windy_z_v0'
ENV_2D = 'point_two_route_swamp_windy_v0'
OUT = 'artifacts/swamp_windy_z/env_audit.json'


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  return {'n': int(v.size), 'mean': float(v.mean()),
          'median': float(np.median(v)), 'min': float(v.min()),
          'max': float(v.max()), 'std': float(v.std())}


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--episodes', type=int, default=300)
  ap.add_argument('--pairs', type=int, default=600)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()

  # PROVENANCE. An earlier version stamped `git rev-parse HEAD`, which records
  # the PARENT commit whenever the audit is run before the code it audits is
  # committed -- exactly what happened once here, leaving 754887e (a commit in
  # which TwoRouteSwampWindyZEnv does not exist) in the artifact. Record
  # instead the commit that last touched the audited files, plus whether the
  # working tree is dirty, so a stale or uncommitted record is self-evident.
  tracked = ['crl/envs.py', 'scripts/audit_swamp_windy_z.py']

  def _git(*a):
    try:
      return subprocess.check_output(
          ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
    except Exception:                              # pylint: disable=broad-except
      return ''

  commit = _git('log', '-1', '--format=%H', '--', *tracked) or '(unavailable)'
  head = _git('rev-parse', 'HEAD') or '(unavailable)'
  dirty = bool(_git('status', '--porcelain', '--', *tracked))

  cfg = Config(env_name=ENV_Z)
  env = envs_mod.make_env(ENV_Z, cfg, seed=args.seed)
  out = {'commit': commit, 'env_name': ENV_Z,
         'git_state': {'code_commit': commit, 'head_at_runtime': head,
                       'audited_files_dirty': dirty, 'tracked_files': tracked,
                       'note': 'code_commit is the last commit touching the '
                               'audited files, NOT git HEAD at run time'},
         'analysis_script': 'scripts/audit_swamp_windy_z.py',
         'env_code_path': 'crl/envs.py :: TwoRouteSwampWindyZEnv'}

  print('=' * 92)
  print('ENV AUDIT -- %s' % ENV_Z)
  print('=' * 92)
  print('  code commit     : %s%s'
        % (commit, '   (WORKING TREE DIRTY)' if dirty else ''))
  print('  head at runtime : %s' % head)
  obs0 = env.reset()
  print('  obs_dim %d  goal_dim %d  action_dim %d  flat obs %s'
        % (cfg.obs_dim, cfg.goal_dim, cfg.action_dim, obs0.shape))
  print('  start_index %d  end_index %d  goal_indices %s'
        % (cfg.start_index, cfg.end_index, cfg.goal_indices))
  print('  reset obs       : %s   (x, y, z, g_x, g_y, g_z)' % np.round(obs0, 4))
  print('  sink params     : substeps %d  speed %.3f  dt %.3f  z_min %.3f'
        % (env.sink_settle_substeps, env.sink_speed, env.sink_dt, env.z_min))
  z_fail_expected = max(env.z_min, -env.sink_settle_substeps * env.sink_speed
                        * env.sink_dt)
  print('  => settled depth: %.4f' % z_fail_expected)
  assert cfg.obs_dim == 3 and cfg.goal_dim == 3, 'dims are not 3-D'
  assert obs0.shape == (6,), 'flat observation is not [x,y,z,gx,gy,gz]'
  assert obs0[2] == 0.0 and obs0[5] == 0.0, 'reset must put z and g_z on ground'
  out['shapes'] = {'obs_dim': int(cfg.obs_dim), 'goal_dim': int(cfg.goal_dim),
                   'action_dim': int(cfg.action_dim),
                   'flat_obs': list(obs0.shape),
                   'layout': '[x, y, z, g_x, g_y, g_z]',
                   'start_index': int(cfg.start_index),
                   'end_index': int(cfg.end_index),
                   'goal_indices': cfg.goal_indices}
  out['sink_params'] = {'sink_settle_substeps': env.sink_settle_substeps,
                        'sink_speed': env.sink_speed, 'sink_dt': env.sink_dt,
                        'z_min': env.z_min,
                        'settled_depth': float(z_fail_expected)}

  # ---------------------------------------------------------------- 6
  print('\n' + '=' * 92)
  print('6. HIDDEN-CONFOUNDER INVARIANTS')
  print('=' * 92)
  inv = {}
  # (a) before contact, z == 0 for BOTH settings of the hidden bit
  hold = np.array([2.5, 3.5])          # holding cell, adjacent to swamp 0
  zs = {}
  for name, bits in (('clear', [0, 0, 0]), ('active', [1, 1, 1])):
    env.reset()
    env.set_auto_resample(False)
    env.set_swamp(bits)
    env.state = hold.copy()
    o = env._get_obs()
    zs[name] = float(o[2])
    print('  standing in the holding cell with U=%-6s -> z = %.4f  obs %s'
          % (name, o[2], np.round(o, 3)))
  assert zs['clear'] == 0.0 and zs['active'] == 0.0
  inv['pre_contact_z_clear'] = zs['clear']
  inv['pre_contact_z_active'] = zs['active']
  print('  => z does NOT leak the hidden bit before contact.')

  # (b) clear contact -> z' == 0 ; (c) fatal contact -> z' < 0
  fwd = np.array([1.0, 0.0], np.float32)
  res = {}
  for name, bits in (('clear', [0, 0, 0]), ('active', [1, 0, 0])):
    env.reset()
    env.set_auto_resample(False)
    env.set_swamp(bits)
    env.state = hold.copy()
    env._rng = np.random.default_rng(12345)        # identical noise draw
    o, _, _, _ = env.step(fwd)
    res[name] = {'obs': o.tolist(), 'z': float(o[2]), 'dead': bool(env.dead)}
    print('  step into swamp 0 with U=%-6s -> xy (%.4f, %.4f)  z %.4f  dead %s'
          % (name, o[0], o[1], o[2], env.dead))
  assert res['clear']['z'] == 0.0, 'clear contact must leave z on the ground'
  assert res['active']['z'] < 0.0, 'fatal contact must leave z below ground'
  inv['clear_contact_z'] = res['clear']['z']
  inv['active_contact_z'] = res['active']['z']

  # (d) the observation must not carry the bits / counters / a dead flag
  env.reset()
  env.set_auto_resample(False)
  env.set_swamp([1, 1, 1])
  o_active = env._get_obs().copy()
  env.set_swamp([0, 0, 0])
  o_clear = env._get_obs().copy()
  leak = float(np.abs(o_active - o_clear).max())
  print('  flipping every swamp bit changes the observation by %.3e' % leak)
  assert leak == 0.0, 'the observation leaks the hidden bits'
  inv['obs_change_when_bits_flipped'] = leak
  inv['audit_fields_in_obs'] = False
  out['6_invariants'] = inv
  print('  ALL INVARIANTS PASS')

  # ---------------------------------------------------------------- 7
  print('\n' + '=' * 92)
  print('7. PAIRED CLEAR-vs-ACTIVE AUDIT (same state, same action, same noise)')
  print('=' * 92)
  rng = np.random.default_rng(args.seed)
  swamp_entry = {0: (2.5, 3.5), 1: (3.5, 3.5), 2: (4.5, 3.5)}
  rows = []
  for k in range(args.pairs):
    cell = k % 3
    sx, sy = swamp_entry[cell]
    s0 = np.array([sx + rng.uniform(-0.15, 0.15),
                   sy + rng.uniform(-0.25, 0.25)])
    act = np.array([rng.uniform(0.6, 1.0), rng.uniform(-0.2, 0.2)], np.float32)
    noise_seed = int(rng.integers(1 << 30))
    legs = {}
    for name in ('clear', 'active'):
      bits = [0, 0, 0]
      if name == 'active':
        bits[cell] = 1
      env.reset()
      env.set_auto_resample(False)      # hold U fixed through the step
      env.set_swamp(bits)
      env.state = s0.copy()
      env._z = 0.0
      env._dead = False
      env._rng = np.random.default_rng(noise_seed)   # identical action noise
      o, _, _, _ = env.step(act)
      legs[name] = {'obs': o.copy(), 'dead': bool(env.dead)}
    dxy = float(np.linalg.norm(legs['clear']['obs'][:2]
                               - legs['active']['obs'][:2]))
    dfull = float(np.linalg.norm(legs['clear']['obs'][:3]
                                 - legs['active']['obs'][:3]))
    rows.append({'cell': cell, 'entered': legs['active']['dead'],
                 'dxy': dxy, 'dfull': dfull,
                 'z_clear': float(legs['clear']['obs'][2]),
                 'z_active': float(legs['active']['obs'][2])})
  ent = [r for r in rows if r['entered']]
  non = [r for r in rows if not r['entered']]
  print('  %d pairs, %d of which actually ENTER the forced cell' % (len(rows),
                                                                    len(ent)))
  if ent:
    dxy = np.array([r['dxy'] for r in ent])
    dfull = np.array([r['dfull'] for r in ent])
    za = np.array([r['z_active'] for r in ent])
    zc = np.array([r['z_clear'] for r in ent])
    print('  entering pairs:')
    print('    ||xy_clear - xy_active||  max %.3e  mean %.3e   <-- must be ~0'
          % (dxy.max(), dxy.mean()))
    print('    z_clear                   min %.4f  max %.4f   <-- must be 0'
          % (zc.min(), zc.max()))
    print('    z_active                  min %.4f  max %.4f   <-- must be < 0'
          % (za.min(), za.max()))
    print('    ||s_clear - s_active||    mean %.4f  min %.4f  max %.4f'
          % (dfull.mean(), dfull.min(), dfull.max()))
    assert dxy.max() < 1e-6, 'XY dynamics differ between clear and active'
    assert (zc == 0).all() and (za < 0).all()
    out['7_paired'] = {
        'n_pairs': len(rows), 'n_entering': len(ent),
        'n_not_entering': len(non),
        'xy_distance_entering': dist(dxy),
        'full_state_distance_entering': dist(dfull),
        'z_active_entering': dist(za), 'z_clear_entering': dist(zc)}
  if non:
    dxy_n = np.array([r['dxy'] for r in non])
    print('  non-entering pairs: ||xy diff|| max %.3e, z identical %s'
          % (dxy_n.max(),
             all(r['z_clear'] == r['z_active'] for r in non)))
    out['7_paired']['xy_distance_not_entering'] = dist(dxy_n)

  # ---------------------------------------------------------------- 8
  print('\n' + '=' * 92)
  print('8. SMALL ROLLOUT AUDIT (%d temporary episodes, NOT the training set)'
        % args.episodes)
  print('=' * 92)
  envz = envs_mod.make_env(ENV_Z, Config(env_name=ENV_Z), seed=args.seed)
  env2 = envs_mod.make_env(ENV_2D, Config(env_name=ENV_2D), seed=args.seed)
  L = envz.max_episode_steps
  n_pre, n_pre_bad = 0, 0
  n_clear_tr, n_clear_bad = 0, 0
  n_fatal, n_fatal_ok = 0, 0
  n_post, n_post_ok = 0, 0
  n_safe, n_safe_ok = 0, 0
  xy_max_diff = 0.0
  safe_xy, fail_xy = [], []
  arng = np.random.default_rng(args.seed + 7)
  for ep in range(args.episodes):
    envz.reset(); env2.reset()
    # identical action stream and identical env RNG so the two envs' noise and
    # bit draws coincide; that is what makes the XY comparison meaningful
    sd = int(arng.integers(1 << 30))
    envz._rng = np.random.default_rng(sd)
    env2._rng = np.random.default_rng(sd)
    envz.state = env2.state.copy()
    envz._z = 0.0
    seen_fatal = False
    for t in range(L):
      a = arng.uniform(-1, 1, 2).astype(np.float32)
      was_dead = envz.dead
      oz, _, _, _ = envz.step(a)
      o2, _, _, _ = env2.step(a)
      xy_max_diff = max(xy_max_diff, float(np.abs(oz[:2] - o2[:2]).max()))
      z = float(oz[2])
      if not was_dead and not envz.dead:
        # a live, non-fatal transition: this is both "pre-contact" and a
        # clear-swamp transition when it lands in the corridor
        n_pre += 1
        n_pre_bad += int(z != 0.0)
        if envz._in_swamp_corridor(envz.state):
          n_clear_tr += 1
          n_clear_bad += int(z != 0.0)
        n_safe += 1
        n_safe_ok += int(z == 0.0)
        safe_xy.append(envz.state.copy())
      elif not was_dead and envz.dead:
        n_fatal += 1
        n_fatal_ok += int(z < 0.0)
        seen_fatal = True
        fail_xy.append(envz.state.copy())
      else:
        n_post += 1
        n_post_ok += int(z < 0.0)
    del seen_fatal
  print('  1. pre-contact obs with z != 0                : %d / %d   %s'
        % (n_pre_bad, n_pre, 'PASS' if n_pre_bad == 0 else 'FAIL'))
  print('  2. clear-swamp transitions with z != 0        : %d / %d   %s'
        % (n_clear_bad, n_clear_tr, 'PASS' if n_clear_bad == 0 else 'FAIL'))
  print('  3. fatal entries with z < 0                   : %d / %d   %s'
        % (n_fatal_ok, n_fatal,
           'PASS' if n_fatal and n_fatal_ok == n_fatal else 'FAIL'))
  print('  4. post-failure states retaining z < 0        : %d / %d   %s'
        % (n_post_ok, n_post, 'PASS' if n_post_ok == n_post else 'FAIL'))
  print('  5. safe states on z = 0                       : %d / %d   %s'
        % (n_safe_ok, n_safe, 'PASS' if n_safe_ok == n_safe else 'FAIL'))
  print('  6. max |XY difference| vs the 2-D env         : %.3e   %s'
        % (xy_max_diff, 'PASS' if xy_max_diff < 1e-6 else 'FAIL'))
  ok8 = (n_pre_bad == 0 and n_clear_bad == 0 and n_fatal
         and n_fatal_ok == n_fatal and n_post_ok == n_post
         and n_safe_ok == n_safe and xy_max_diff < 1e-6)
  out['8_rollout'] = {
      'episodes': args.episodes,
      'pre_contact_with_nonzero_z': n_pre_bad, 'pre_contact_total': n_pre,
      'clear_swamp_with_nonzero_z': n_clear_bad,
      'clear_swamp_total': n_clear_tr,
      'fatal_with_negative_z': n_fatal_ok, 'fatal_total': n_fatal,
      'post_failure_with_negative_z': n_post_ok, 'post_failure_total': n_post,
      'safe_on_ground': n_safe_ok, 'safe_total': n_safe,
      'max_xy_diff_vs_2d_env': xy_max_diff, 'all_pass': bool(ok8)}

  safe_xy = np.array(safe_xy) if safe_xy else np.zeros((0, 2))
  fail_xy = np.array(fail_xy) if fail_xy else np.zeros((0, 2))
  print('\n  S_safe (z=0) %s   S_fail (z<0) %s' % (safe_xy.shape,
                                                   fail_xy.shape))
  if len(fail_xy):
    from scipy.spatial import cKDTree
    d_xy, _ = cKDTree(safe_xy).query(fail_xy)
    print('  XY-only nearest distance from each FAIL state to some SAFE state:')
    print('    mean %.4f  median %.4f  max %.4f   frac < 0.05 : %.3f'
          % (d_xy.mean(), np.median(d_xy), d_xy.max(), (d_xy < 0.05).mean()))
    print('  => in XY alone the two sets OVERLAP, which is the original problem.')
    print('  In the full 3-D state every FAIL state is at least |z_fail| = '
          '%.4f away' % abs(z_fail_expected))
    print('  from every SAFE state, because z separates them exactly.')
    out['8_separability'] = {
        'n_safe': int(len(safe_xy)), 'n_fail': int(len(fail_xy)),
        'xy_nn_dist_fail_to_safe': dist(d_xy),
        'frac_fail_within_0.05_xy_of_a_safe_state': float((d_xy < 0.05).mean()),
        'guaranteed_3d_separation': float(abs(z_fail_expected))}

  # ---------------------------------------------------------------- 9
  print('\n' + '=' * 92)
  print('9. REPLAY SEMANTICS -- can a z < 0 state be sampled as a POSITIVE?')
  print('=' * 92)
  print('  ANSWER: YES, unavoidably, under the current relabeling.')
  print('  crl/replay.py TrajectoryBuffer._draw_indices draws the future index')
  print('  j from  future = arange > i  over the FULL episode length, weighted')
  print('  geometrically by discount; there is no death mask and no z filter.')
  print('  crl/replay.py TrajectoryBuffer.sample then takes')
  print('    goal_state = self._obs[traj, j, :obs_dim]')
  print('    goal       = obs_to_goal(goal_state, start_index, end_index, ...)')
  print('  and with start_index=0 / end_index=-1 that slice is the IDENTITY, so')
  print('  the goal keeps its z component. Because a dead episode is frozen and')
  print('  fixed-length, every row from the death row to the end carries')
  print('  z = %.4f, and any anchor i before it can draw one of them as its'
        % z_fail_expected)
  print('  POSITIVE future goal.')
  print('  Measured previously on the 2-D dataset: 71.7% of anchors inside')
  print('  dead episodes already relabel their own death pose as a positive.')
  print('  CONSEQUENCE for the intended plan of using z < 0 as failure')
  print('  NEGATIVES: the identical point would be a positive for its own')
  print('  episode and a negative from the bank -- an exact label conflict,')
  print('  not an approximate one. What z DOES fix is the other collision: a')
  print('  successful passage now sits at (x, y, 0), a permanent %.4f away'
        % abs(z_fail_expected))
  print('  from the death state at the same XY, so those two are no longer')
  print('  confusable. Reported, NOT fixed here -- a fix belongs in replay.')
  out['9_replay_semantics'] = {
      'can_z_negative_be_a_positive_goal': True,
      'code_paths': ['crl/replay.py :: TrajectoryBuffer._draw_indices '
                     '(future = arange > i, geometric, no death mask)',
                     'crl/replay.py :: TrajectoryBuffer.sample '
                     '(goal_state = _obs[traj, j, :obs_dim])',
                     'crl/replay.py :: obs_to_goal (identity at start=0/end=-1,'
                     ' so z is kept in the goal)'],
      'reason': 'a dead episode is frozen and fixed-length, so every row from '
                'the death row to the end carries z<0 and is inside the '
                'geometric future window of any earlier anchor',
      'fixed_here': False,
      'note': 'z does NOT resolve the positive/negative conflict for a dead '
              'episode relabeling its own death pose; it DOES resolve the '
              'successful-passage vs death collision at equal XY'}

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % args.out)
  print('OVERALL: %s' % ('ALL AUDITS PASS' if ok8 else 'SOME CHECKS FAILED'))
  return 0 if ok8 else 1


if __name__ == '__main__':
  sys.exit(main())
