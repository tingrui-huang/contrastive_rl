r"""Which goal representation can NAME a V5 rock death? Measured, not assumed.

Read-only probe. No training, no dataset is written, no env code is changed:
it drives crl/rockfall_clock_v5.py through the existing ClockV5Teacher and
measures the 29-dim learner state.

THE QUESTION. crl/losses.py implements failure-aware negatives by splicing a
bank of states into the GOAL half of the observation, so every bank entry is
a point in goal space. Under the frozen V5 default (dc6690a) that space is the
torso XY pair. A PointMaze f4 failure state survives its projection because
the f4 goal IS the four-frame position stack and a dead agent is frozen in it.
An AntMaze V5 death has no such luck: the ant is killed by a RULE (a flagged
rock contact), at a point a safe crossing also passes through. This script
measures whether anything of the death survives the projection, and if not,
which minimal extension of the goal columns keeps it.

WHAT IS MEASURED, per candidate column subset:

  AUC        5-fold held-out Fisher-LDA AUC separating real death states from
             real safe-crossing states MATCHED ON XY (3 nearest by torso XY).
             Folds are grouped by source episode. 0.5 = chance.
  gap_band   median nearest-neighbour distance from a death state to the
             expert's safe band crossings, in units of the data's own
             per-column sigma. How far outside normal corridor traffic a
             bank entry sits.
  gap_hold   the same to the expert's zero-torque HOLD at the mouth. This is
             the false positive that matters here: the benchmark's safe blind
             policy is to stand still, and an ant standing still is slow, so a
             failure signature built on "slow" would penalise the behaviour
             the benchmark is trying to elicit.
  goal_off   distance from the COMMANDED task goal to the nearest state in
             the expert data, same sigma units. This is the "XY + 27 zeros"
             pathology as a number: a goal vector no training positive can
             ever resemble.

DEATH-SETTLE IS ALSO MEASURED, AND IS A DISTINCT QUESTION. crl/rockfall_ant.py
(the V1/V2 rockfall family) has an opt-in ``death_settle_substeps``: on a fatal
contact the actor loses control and physics runs on, so the fatal outcome
develops into the observation. This probe evaluates the same idea for V5
WITHOUT changing the env -- it steps the underlying mujoco sim with zero ctrl
after the env has already reported the death, purely as a readout -- and
reports what settling does to gap_hold. Settling is not free here: a settled
ant is a MOTIONLESS ant, and motionless is exactly what the expert's safe hold
looks like.

Usage:
  python scripts/probe_v5_failure_representation.py
  python scripts/probe_v5_failure_representation.py --n-death 300 --n-normal 300
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                                # noqa: E402
from crl import rockfall_clock_v5 as V5                         # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X, HAZARD_HALF_Y    # noqa: E402
import rockfall_clock_v5_teacher as CT                          # noqa: E402

OUT_DIR = 'artifacts/v5_failneg/representation_probe'
HORIZON = 400
#: 29-dim ant state layout (crl/d4rl_ant.py _Sim._obs_dict):
#: [0:2] torso xy | [2] z | [3:7] quat | [7:15] joint angles
#: [15:18] linear vel | [18:21] angular vel | [21:29] joint vel
STATE_NAMES = (['x', 'y', 'z'] + ['qw', 'qx', 'qy', 'qz']
               + ['j%d' % i for i in range(8)]
               + ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
               + ['jv%d' % i for i in range(8)])
CANDIDATES = {
    'xy   (frozen V5 default)':       V5.GOAL_INDICES_XY,
    'xy+z':                           (0, 1, 2),
    'xy+linvel':                      (0, 1, 15, 16, 17),
    'xy+angvel':                      (0, 1, 18, 19, 20),
    'xy+linvel+angvel   (XYV)':       V5.GOAL_INDICES_XYV,
    'xy+z+linvel+angvel':             (0, 1, 2, 15, 16, 17, 18, 19, 20),
    'xy+all velocity (incl joints)':  (0, 1) + tuple(range(15, 29)),
    'xy+full 27 (the padded goal)':   tuple(range(29)),
}
SETTLE_STEPS = (0, 10, 20, 40, 80, 160)


def in_band(xy):
  return ((np.abs(xy[..., 1]) < HAZARD_HALF_Y)
          & (xy[..., 0] >= HAZARD_X[0]) & (xy[..., 0] <= HAZARD_X[1]))


def state_of(sim):
  d = sim._obs_dict()
  return np.concatenate([d['achieved_goal'],
                         d['observation']]).astype(np.float32)


def collect_deaths(env, teacher, n, settle_steps, horizon=HORIZON):
  """Real V5 rock deaths, plus the privileged post-mortem settle readout.

  do(go) under an ACTIVE latent, which the teacher audit measures at >0.90
  failure. ``settle_steps[0]`` must be 0: that entry is the state the env
  ITSELF returns on the death step, the only one that costs no change to the
  dynamics. The later entries keep stepping the raw sim with zero ctrl AFTER
  the env has declared the episode over -- a readout of what the ant would
  have done, never anything the env does on its own.
  """
  assert settle_steps[0] == 0
  out = {k: [] for k in settle_steps}
  meta = []
  for _ in range(n):
    o = env.reset(rockfall_active=True)
    teacher.fresh(route='shortcut')
    info = {}
    for t in range(horizon):
      o, r, done, info = env.step(teacher.act(o, env.schedule, 'go'))
      if done or r > 0:
        break
    if not info.get('failure'):
      continue
    out[0].append(o[:29].copy())
    meta.append({'t': int(info.get('t', -1)),
                 'in_band': bool(in_band(o[:2])),
                 'rockfall_start': info.get('rockfall_start')})
    sim = env._env
    sim.data.ctrl[:] = 0.0
    prev = 0
    for k in settle_steps[1:]:
      for _ in range(k - prev):
        mujoco.mj_step(sim.model, sim.data)
      prev = k
      out[k].append(state_of(sim))
  return {k: np.asarray(v, np.float32) for k, v in out.items()}, meta


def collect_normal(env, teacher, n, p_active, p_far, seed, horizon=HORIZON):
  """The expert's own states, tagged by what the expert was doing."""
  rng = np.random.default_rng(seed)
  states, tags, episode_ids = [], [], []
  for episode_id in range(n):
    u = bool(rng.random() < p_active)
    route = 'detour' if rng.random() < p_far else 'shortcut'
    o = env.reset(rockfall_active=u)
    teacher.fresh(route=route)
    info = {}
    for t in range(horizon):
      a = teacher.act(o, env.schedule)
      holding = teacher.holding
      o, r, done, info = env.step(a)
      s = o[:29].copy()
      states.append(s)
      episode_ids.append(episode_id)
      if holding:
        tags.append('hold')
      elif route == 'detour':
        tags.append('detour')
      elif in_band(s[:2]):
        tags.append('band')
      else:
        tags.append('other')
      if done or r > 0:
        break
    assert not info.get('failure'), 'the sighted expert must never die'
  return (np.asarray(states, np.float32), np.asarray(tags),
          np.asarray(episode_ids, np.int64))


def xy_matched(anchors, pool, pool_groups, k=3):
  """For every anchor, the k pool rows nearest in torso XY."""
  d2 = ((anchors[:, None, 0] - pool[None, :, 0]) ** 2
        + (anchors[:, None, 1] - pool[None, :, 1]) ** 2)
  order = np.argsort(d2, axis=1)[:, :k]
  gap = np.sqrt(np.take_along_axis(d2, order, axis=1))
  flat = order.ravel()
  return pool[flat], np.asarray(pool_groups)[flat], float(gap.mean())


def _auc(sa, sb):
  """Rank AUC for positive scores ``sa`` versus negative scores ``sb``."""
  rank = np.concatenate([sa, sb]).argsort().argsort()[:len(sa)]
  return float((rank.sum() - len(sa) * (len(sa) - 1) / 2) / (len(sa) * len(sb)))


def lda_auc(a, b, cols, a_groups, b_groups, folds=5, seed=1729):
  """Held-out Fisher-LDA AUC; fit and scored episodes never overlap.

  Folds are assigned by source episode, not row: all states from one normal
  expert episode stay together, and every death already has its own episode.
  Thus neither exact rows nor correlated within-episode states cross the
  train/test boundary. AUCs are computed within each held-out fold and averaged.
  """
  A = np.asarray(a[:, cols], np.float64)
  B = np.asarray(b[:, cols], np.float64)
  ag, bg = np.asarray(a_groups), np.asarray(b_groups)
  if len(ag) != len(A) or len(bg) != len(B):
    raise ValueError('group arrays must align with class rows')
  if min(len(np.unique(ag)), len(np.unique(bg))) < folds:
    raise ValueError('need at least %d source episodes per class for held-out '
                     'AUC' % folds)
  rng = np.random.default_rng(seed)
  aug, bug = rng.permutation(np.unique(ag)), rng.permutation(np.unique(bg))
  afold = {g: i % folds for i, g in enumerate(aug)}
  bfold = {g: i % folds for i, g in enumerate(bug)}
  aucs = []
  for fold in range(folds):
    ate = np.flatnonzero([afold[g] == fold for g in ag])
    bte = np.flatnonzero([bfold[g] == fold for g in bg])
    atr = np.flatnonzero([afold[g] != fold for g in ag])
    btr = np.flatnonzero([bfold[g] != fold for g in bg])
    AA, BB = A[atr], B[btr]
    mu = AA.mean(0) - BB.mean(0)
    cov = np.cov(np.vstack([AA - AA.mean(0), BB - BB.mean(0)]).T)
    cov = np.atleast_2d(cov) + 1e-6 * np.eye(len(cols))
    w = np.linalg.solve(cov, mu)
    aucs.append(_auc(A[ate] @ w, B[bte] @ w))
  return float(np.mean(aucs))


def nn_sigma(a, b, cols, sigma, chunk=256):
  """Median nearest-neighbour distance a -> b in per-column sigma units."""
  A, B = a[:, cols] / sigma, b[:, cols] / sigma
  best = []
  for i in range(0, len(A), chunk):
    best.append(np.linalg.norm(A[i:i + chunk, None, :] - B[None, :, :],
                               axis=2).min(1))
  return float(np.median(np.concatenate(best)))


def dprime(a, b):
  s = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2.0) + 1e-9
  return np.abs(a.mean(0) - b.mean(0)) / s


def git(*a):
  try:
    return subprocess.check_output(
        ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                          # pylint: disable=broad-except
    return ''


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--n-death', type=int, default=300)
  ap.add_argument('--n-normal', type=int, default=300)
  ap.add_argument('--seed', type=int, default=4242)
  ap.add_argument('--p-active', type=float, default=CT.P_ACTIVE)
  ap.add_argument('--p-far', type=float, default=CT.P_FAR)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  cfg, teacher = CT.make_teacher()
  cfg.rockfall_max_steps = args.horizon
  env_d = envs_mod.make_env(CT.ENV_NAME, cfg, seed=args.seed)
  env_n = envs_mod.make_env(CT.ENV_NAME, cfg, seed=args.seed + 3000)

  print('=' * 92)
  print('V5 FAILURE-REPRESENTATION PROBE  (env %s)' % CT.ENV_NAME)
  print('=' * 92)
  deaths, dmeta = collect_deaths(env_d, teacher, args.n_death, SETTLE_STEPS,
                                 args.horizon)
  normal, tags, normal_ep = collect_normal(
      env_n, teacher, args.n_normal, args.p_active, args.p_far,
      args.seed + 11, args.horizon)
  band = normal[tags == 'band']
  band_ep = normal_ep[tags == 'band']
  hold = normal[tags == 'hold']
  print('  deaths                 %d (do(go) | active, %d episodes)'
        % (len(deaths[0]), args.n_death))
  print('  deaths inside the band %.3f'
        % float(np.mean([m['in_band'] for m in dmeta])))
  print('  expert states          %d  %s'
        % (len(normal), {t: int((tags == t).sum()) for t in sorted(set(tags))}))

  #: the commanded goal under the offline contract: zeros(29) with [:2] = gxy.
  task_goal = np.zeros(29, np.float32)
  task_goal[:2] = env_n._eval_goal_xy()

  report = {
      'probe_script': 'scripts/probe_v5_failure_representation.py',
      'env': CT.ENV_NAME,
      'code_commit': git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
      'head': git('rev-parse', 'HEAD'),
      'dirty': bool(git('status', '--porcelain', '--', 'crl', 'scripts')),
      'n_deaths': int(len(deaths[0])), 'n_normal': int(len(normal)),
      'n_band': int(len(band)), 'n_hold': int(len(hold)),
      'death_in_band_frac': float(np.mean([m['in_band'] for m in dmeta])),
      'state_layout': {'xy': [0, 1], 'z': [2], 'quat': [3, 7],
                       'joint_angles': [7, 15], 'linear_velocity': [15, 18],
                       'angular_velocity': [18, 21],
                       'joint_velocity': [21, 29]},
      'task_goal_xy': [float(v) for v in task_goal[:2]],
      'settle_note':
          'settle > 0 is a privileged READOUT: after the env has already '
          'declared the death, the raw mujoco sim is stepped with zero ctrl. '
          'crl/rockfall_clock_v5.py is not modified and nothing about the '
          'benchmark dynamics changes.',
      'auc_protocol': '5-fold source-episode-grouped Fisher LDA. All states '
                      'from one normal expert episode remain in one fold; '
                      'death episodes are likewise disjoint. No episode is '
                      'used both to fit and score a fold.',
  }

  # ------------------------------------------------------- representations
  d0 = deaths[0]
  death_ep = np.arange(len(d0), dtype=np.int64)
  band_m, band_m_ep, band_gap_xy = xy_matched(d0, band, band_ep)
  print('\n' + '-' * 92)
  print('REPRESENTATIONS, at the REAL V5 death step (settle 0). XY match gap '
        '%.4f maze units.' % band_gap_xy)
  print('-' * 92)
  print('%-32s %4s %7s %10s %10s %9s' % ('goal columns', 'dim', 'AUC',
                                         'gap_band', 'gap_hold', 'goal_off'))
  rep_rows = {}
  for name, cols in CANDIDATES.items():
    cols = list(cols)
    sigma = normal[:, cols].std(0) + 1e-6
    row = {
        'columns': cols, 'dim': len(cols),
        'auc_vs_xy_matched_band': lda_auc(
            d0, band_m, cols, death_ep, band_m_ep),
        'gap_to_band_sigma': nn_sigma(d0, band, cols, sigma),
        'gap_to_hold_sigma': nn_sigma(d0, hold, cols, sigma),
        'task_goal_offdist_sigma': nn_sigma(task_goal[None, :], normal, cols,
                                            sigma),
    }
    rep_rows[name] = row
    print('%-32s %4d %7.3f %10.3f %10.3f %9.3f'
          % (name, row['dim'], row['auc_vs_xy_matched_band'],
             row['gap_to_band_sigma'], row['gap_to_hold_sigma'],
             row['task_goal_offdist_sigma']))
  report['representations'] = rep_rows
  report['xy_match_gap_maze_units'] = band_gap_xy

  # --------------------------------------------------------- settle sweep
  print('\n' + '-' * 92)
  print('DEATH-SETTLE SWEEP, in the XYV columns. settle 0 = what V5 returns.')
  print('-' * 92)
  cols = list(V5.GOAL_INDICES_XYV)
  sigma = normal[:, cols].std(0) + 1e-6
  print('%8s %7s %10s %10s %10s %10s'
        % ('settle', 'AUC', 'gap_band', 'gap_hold', 'mean |v|', 'mean |w|'))
  settle_rows = {}
  for k in SETTLE_STEPS:
    dk = deaths[k]
    bm, bm_ep, _ = xy_matched(dk, band, band_ep)
    row = {'auc_vs_xy_matched_band': lda_auc(
               dk, bm, cols, death_ep, bm_ep),
           'gap_to_band_sigma': nn_sigma(dk, band, cols, sigma),
           'gap_to_hold_sigma': nn_sigma(dk, hold, cols, sigma),
           'mean_linear_speed': float(
               np.linalg.norm(dk[:, 15:18], axis=1).mean()),
           'mean_angular_speed': float(
               np.linalg.norm(dk[:, 18:21], axis=1).mean())}
    settle_rows[str(k)] = row
    print('%8d %7.3f %10.3f %10.3f %10.3f %10.3f'
          % (k, row['auc_vs_xy_matched_band'], row['gap_to_band_sigma'],
             row['gap_to_hold_sigma'], row['mean_linear_speed'],
             row['mean_angular_speed']))
  settle_rows['reference_hold'] = {
      'mean_linear_speed': float(np.linalg.norm(hold[:, 15:18], axis=1).mean()),
      'mean_angular_speed': float(
          np.linalg.norm(hold[:, 18:21], axis=1).mean())}
  settle_rows['reference_band'] = {
      'mean_linear_speed': float(np.linalg.norm(band[:, 15:18], axis=1).mean()),
      'mean_angular_speed': float(
          np.linalg.norm(band[:, 18:21], axis=1).mean())}
  print('%8s %7s %10s %10s %10.3f %10.3f'
        % ('(hold)', '', '', '',
           settle_rows['reference_hold']['mean_linear_speed'],
           settle_rows['reference_hold']['mean_angular_speed']))
  print('%8s %7s %10s %10s %10.3f %10.3f'
        % ('(band)', '', '', '',
           settle_rows['reference_band']['mean_linear_speed'],
           settle_rows['reference_band']['mean_angular_speed']))
  report['settle_sweep'] = settle_rows

  # ------------------------------------------------------------ per-column
  dp = dprime(d0, band_m)
  top = np.argsort(-dp)[:8]
  report['per_column_dprime_death_vs_matched_band'] = {
      STATE_NAMES[i]: float(dp[i]) for i in range(29)}
  print('\ntop separating columns at settle 0: %s'
        % ', '.join('%s %.2f' % (STATE_NAMES[i], dp[i]) for i in top))

  # false positive the benchmark actually cares about
  band_speed = np.linalg.norm(band[:, 15:17], axis=1)
  hold_speed = np.linalg.norm(hold[:, 15:17], axis=1)
  death_speed = np.linalg.norm(d0[:, 15:17], axis=1)
  report['speed_check'] = {
      'note': 'a "slow" failure signature would also fire on the expert hold, '
              'which is the safe blind behaviour. These are the three planar '
              'speed distributions that decide whether it does.',
      'band_p1': float(np.percentile(band_speed, 1)),
      'band_median': float(np.median(band_speed)),
      'band_frac_below_0.3': float((band_speed < 0.3).mean()),
      'hold_median': float(np.median(hold_speed)),
      'hold_p95': float(np.percentile(hold_speed, 95)),
      'death_median': float(np.median(death_speed)),
      'death_p5': float(np.percentile(death_speed, 5))}
  print('planar speed  band p1 %.3f median %.3f | hold median %.3f p95 %.3f | '
        'death median %.3f p5 %.3f'
        % (report['speed_check']['band_p1'],
           report['speed_check']['band_median'],
           report['speed_check']['hold_median'],
           report['speed_check']['hold_p95'],
           report['speed_check']['death_median'],
           report['speed_check']['death_p5']))

  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'representation_probe.json')
  with open(path, 'w') as f:
    json.dump(report, f, indent=2)
  np.savez_compressed(
      os.path.join(args.out_dir, 'probe_states.npz'),
      deaths_settle0=deaths[0], band=band, hold=hold, task_goal=task_goal)
  print('\n-> %s' % path)


if __name__ == '__main__':
  main()
