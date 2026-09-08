r"""Collect V6 failure candidates for the composed failure bank.

NEGATIVE-BANK DATA ONLY. Nothing here touches the normal V6 offline dataset,
and these trajectories are never anchors, positives or hindsight positives.
They exist to build q_fail and nothing else.

The V6 benchmark itself is unchanged: the maze, the rocks, both hazard coins,
both absolute clocks, p_active and the horizon are the frozen ones. Only the
BEHAVIOUR POLICY changes, plus two opt-in interfaces that are inert in every
ordinary episode -- the teacher's zone-specific deliberate override and the
env's ``death_settle_substeps``.

FOUR ARMS
---------
  random          uniform torque in [-1, 1]^8. Kept because "we tried it and
                  it produced nothing" is a result, not an omission. On the V5
                  map a uniform-torque ant never left the start cell (mean
                  furthest x 0.93, band at 2.6); V6's first band is at x = 6.6.
                  The arm bails with a quantified negative result.
  noisy           the frozen walking controller driving the shortcut BLIND
                  (intent='go', handed a redacted timetable so it structurally
                  cannot peek) plus Gaussian action noise. This is the arm that
                  supplies the random/noisy 60% of the bank, and it is called
                  noisy-controller everywhere because that is what it is.
  deliberate_z1   the NORMAL privileged sighted teacher, with one local
                  override at zone 1 (see below).
  deliberate_z2   the same, targeted at zone 2, with zone 1 handled normally
                  and safely.

WHAT MAKES A DELIBERATE FAILURE DELIBERATE
------------------------------------------
The collector never asks for a failure. It asks the normal sighted rule what it
would do and only overrides a decision the rule made:

    normal_decision[z]   == 'wait'      (the rule knew the crossing was unsafe)
    executed_decision[z] == 'go'        (we made it walk in anyway)
    deliberate_override[z] is True
    actual failure_zone  == z           (real MuJoCo rocks did the killing)

All four are recorded per episode and re-asserted by the bank builder. An
episode whose normal decision was 'go' is NOT a deliberate failure however it
ended, and is dropped with a reason. A zone-2 candidate that dies at zone 1 is
dropped: it never got to make the zone-2 mistake.

The 5% teacher detour coin is DISABLED in this collector only, so a candidate
reliably reaches the targeted shortcut hazard. That is collection-only
behaviour and is recorded in the manifest; the benchmark's teacher and the
frozen dataset keep their 0.05 coin.

The hazard latent of the targeted zone is FORCED ACTIVE -- a zone that is not
armed can never make its rule say 'wait'. That is generation using a
privileged field, recorded as such. The absolute clocks t0_1 / t0_2 are NEVER
forced: the env draws them from its own rngs every reset, exactly as in the
benchmark.

FAILURE STATE
-------------
One entry per failed episode: the observation the fatal transition returns
after ``--settle`` ctrl-free MuJoCo substeps, which is the AntMaze settled
convention (scripts/rebuild_failure_bank_settled.py, SETTLE_N = 80). The
actor loses control at the fatal contact; no observation field is written by
hand. Because V6's failure is absorbing and the loop breaks on ``done``, that
row is the last valid row of the episode and no episode can gain weight from
repeated post-death frames.

Usage:
  python scripts/collect_v6_failure_candidates.py --arm noisy --episodes 400
  python scripts/collect_v6_failure_candidates.py --arm deliberate_z1 --episodes 200
  python scripts/collect_v6_failure_candidates.py --arm deliberate_z2 --episodes 200
  python scripts/collect_v6_failure_candidates.py --all --episodes 300
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                       # noqa: E402
from crl import rockfall_clock_v6 as V6                # noqa: E402
import rockfall_clock_v6_teacher as CT                 # noqa: E402

OUT_DIR = os.path.join('artifacts', 'v6_failneg', 'candidates')
NAME = 'v6_failures'
ENV_NAME = CT.ENV_NAME
HORIZON = int(CT.HORIZON)
STATE_DIM = 29
ACTION_DIM = 8
ARMS = ('random', 'noisy', 'deliberate_z1', 'deliberate_z2')
BLIND_ARMS = ('random', 'noisy')
#: the AntMaze settled-failure convention (scripts/rebuild_failure_bank_settled)
SETTLE_N = 80
DEFAULT_NOISE = 0.3


class RedactedSchedule(dict):
  """The timetable with the timetable removed.

  The blind arms are handed this instead of ``env.schedule``. Reading any
  hazard field raises, so "it did not peek" is a structural property rather
  than a promise. ``t`` stays readable: the ant can count its own steps.
  """

  def __getitem__(self, key):
    if key in ('zones', 'active', 'start', 'end', 'sampled_start'):
      raise KeyError('a blind arm read the privileged timetable field %r -- '
                     'that is a bug, not a data point' % key)
    return dict.__getitem__(self, key)


class Futile(Exception):
  """An arm that does not produce usable candidates, with the numbers."""

  def __init__(self, arm, attempts, kept, drops, furthest):
    super().__init__(
        'arm %r kept %d of %d episodes; drop reasons %s; mean furthest torso '
        'x %.2f (zone 1 starts at x = %.1f)'
        % (arm, kept, attempts, drops, furthest, V6.HAZARD_X[1][0]))
    self.arm, self.attempts, self.kept = arm, attempts, kept
    self.drops, self.furthest_x = drops, furthest


def make_env(seed, horizon, settle, p_active_1, p_active_2):
  """The frozen V6 env, plus the opt-in settle knob (0 => untouched)."""
  cfg, teacher = CT.make_teacher()
  cfg.rockfall_max_steps = int(horizon)
  cfg.max_episode_steps = int(horizon)
  cfg.rockfall_p_active_1 = float(p_active_1)
  cfg.rockfall_p_active_2 = float(p_active_2)
  cfg.rockfall_death_settle_substeps = int(settle)
  env = envs_mod.make_env(ENV_NAME, cfg, seed=seed)
  assert env.death_settle_substeps == int(settle)
  return cfg, env, teacher


def run_episode(env, teacher, arm, rng, noise, horizon, record):
  """One episode of one arm. Returns its provenance row.

  Latents: the targeted zone is forced active; everything else is the env's
  own draw. Clocks are never forced.
  """
  blind = arm in BLIND_ARMS
  target = {'deliberate_z1': 1, 'deliberate_z2': 2}.get(arm)
  o = env.reset(rockfall_active_1=True if target == 1 else None,
                rockfall_active_2=True if target == 2 else None)
  #: the 5% detour coin is OFF here -- collection-only, see the module doc.
  teacher.fresh(route='shortcut', deliberate_override_zone=target)
  info = {}
  far_x = -np.inf
  t = 0
  for t in range(horizon):
    sched = (RedactedSchedule(t=env.schedule['t']) if blind else env.schedule)
    if arm == 'random':
      a = rng.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
    elif arm == 'noisy':
      a = teacher.act(o, sched, 'go')
      a = np.clip(a + rng.normal(0.0, noise, ACTION_DIM),
                  -1.0, 1.0).astype(np.float32)
    else:
      a = teacher.act(o, sched)          # the NORMAL sighted rule, overridden
    o2, r, done, info = env.step(a)
    record(o, a, o2)
    o = o2
    far_x = max(far_x, float(o[0]))
    if done or r > 0:
      break

  normal = teacher.normal_decisions
  executed = teacher.executed_decisions
  override = teacher.deliberate_overrides
  failure = bool(info.get('failure'))
  fzone = info.get('failure_zone')
  #: survived zone 1 = it got through the first band alive. A candidate that
  #: died at zone 1 has fzone == 1 and is excluded by this for zone 2.
  survived_z1 = bool(info.get('entered_hazard_1')
                     and (not failure or fzone != 1))
  return {
      'source_arm': arm,
      'source_type': 'deliberate' if target else 'random',
      'targeted_zone': target,
      'outcome': ('rock_death' if failure
                  else ('success' if info.get('success') else 'timeout')),
      'failure': failure,
      'actual_failure_zone': int(fzone) if fzone is not None else -1,
      'survived_zone1': survived_z1,
      'normal_decision_zone1': normal[1], 'normal_decision_zone2': normal[2],
      'executed_decision_zone1': executed[1],
      'executed_decision_zone2': executed[2],
      'deliberate_override_zone1': bool(override[1]),
      'deliberate_override_zone2': bool(override[2]),
      'teacher_route': teacher.route,
      'route_realized': info.get('route'),
      'u1': bool(info.get('rockfall_active_1')),
      'u2': bool(info.get('rockfall_active_2')),
      't0_1': info.get('rockfall_sampled_start_1'),
      't0_2': info.get('rockfall_sampled_start_2'),
      'rockfall_start_1': info.get('rockfall_start_1'),
      'rockfall_start_2': info.get('rockfall_start_2'),
      'rockfall_end_1': info.get('rockfall_end_1'),
      'rockfall_end_2': info.get('rockfall_end_2'),
      'mouth_step_zone1': info.get('mouth_step_1'),
      'mouth_step_zone2': info.get('mouth_step_2'),
      'band_entry_step_zone1': info.get('band_entry_step_1'),
      'band_entry_step_zone2': info.get('band_entry_step_2'),
      'entered_hazard_zone1': bool(info.get('entered_hazard_1')),
      'entered_hazard_zone2': bool(info.get('entered_hazard_2')),
      'hold_steps_zone1': int(teacher.hold_steps[1]),
      'hold_steps_zone2': int(teacher.hold_steps[2]),
      'rock_contact_zone1': bool(info.get('rock_contact_1')),
      'rock_contact_zone2': bool(info.get('rock_contact_2')),
      'env_seed': int(info.get('env_seed', -1)),
      'reset_index': int(info.get('reset_index', -1)),
      'failure_step': int(info.get('t', t + 1)) if failure else -1,
      'ep_length': int(t + 1),
      'furthest_x': round(float(far_x), 4),
      'final_xy': [round(float(o[0]), 4), round(float(o[1]), 4)],
  }


def keep_rule(row):
  """Accept only what the arm set out to produce. Nothing is repaired."""
  arm = row['source_arm']
  if not row['failure']:
    return False, 'not_a_rock_death:' + row['outcome']
  z = row['actual_failure_zone']
  if z not in (1, 2):
    return False, 'failure_zone_unidentified'
  if arm in BLIND_ARMS:
    return True, ''
  target = row['targeted_zone']
  if row['normal_decision_zone%d' % target] != 'wait':
    return False, 'normal_decision_was_not_wait'
  if row['executed_decision_zone%d' % target] != 'go':
    return False, 'executed_decision_was_not_go'
  if not row['deliberate_override_zone%d' % target]:
    return False, 'override_did_not_fire'
  if z != target:
    return False, 'died_in_zone_%d_not_the_targeted_%d' % (z, target)
  if target == 2 and not row['survived_zone1']:
    return False, 'did_not_survive_zone_1'
  return True, ''


def drop_summary(dropped):
  out = {}
  for d in dropped:
    out[d['drop_reason']] = out.get(d['drop_reason'], 0) + 1
  return out


def collect(arm, episodes, seed, args):
  cfg, env, teacher = make_env(seed, args.horizon, args.settle,
                               args.p_active_1, args.p_active_2)
  rng = np.random.default_rng(seed + 90_001)
  L = args.horizon + 1
  obs = np.zeros((episodes, L, 58), np.float32)
  act = np.zeros((episodes, L, ACTION_DIM), np.float32)
  lengths = np.zeros(episodes, np.int64)
  eval_goals = np.zeros((episodes, 2), np.float32)
  rows, dropped = [], []
  kept = attempts = 0
  while kept < episodes:
    attempts += 1
    box = {'n': 0}

    def record(o, a, o2, _box=box, _k=kept):
      i = _box['n']
      if i == 0:
        obs[_k, 0] = o
        eval_goals[_k] = o[29:31]
      act[_k, i] = a
      obs[_k, i + 1] = o2
      _box['n'] = i + 1

    row = run_episode(env, teacher, arm, rng, args.noise, args.horizon, record)
    ok, why = keep_rule(row)
    n = box['n']
    if not ok or n < 2:
      obs[kept, :] = 0.0
      act[kept, :] = 0.0
      dropped.append(dict(row, drop_reason=why or 'fewer_than_two_obs'))
      if kept == 0 and attempts >= args.futility_attempts:
        raise Futile(arm, attempts, kept, drop_summary(dropped),
                     float(np.mean([d['furthest_x'] for d in dropped])))
      if attempts > args.max_attempt_factor * episodes:
        raise Futile(arm, attempts, kept, drop_summary(dropped),
                     float(np.mean([d['furthest_x'] for d in dropped])))
      continue
    lengths[kept] = n + 1
    row['episode_id'] = kept
    row['death_row'] = n
    rows.append(row)
    kept += 1
    if kept % args.progress_every == 0:
      print('  %-14s %d/%d kept (%d attempts, %d dropped)'
            % (arm, kept, episodes, attempts, len(dropped)), flush=True)
  return dict(obs=obs, act=act, lengths=lengths, eval_goals=eval_goals,
              rows=rows, dropped=dropped, attempts=attempts, cfg=cfg)


def write(arm, data, args, seed):
  os.makedirs(args.out_dir, exist_ok=True)
  rows = data['rows']
  name = '%s_%s_s%d' % (args.name, arm, seed)
  drops = drop_summary(data['dropped'])
  zone_counts = {str(z): int(sum(r['actual_failure_zone'] == z for r in rows))
                 for z in (1, 2)}
  meta = {
      'name': name,
      'env': ENV_NAME,
      'env_version': V6.ENV_VERSION,
      'source_arm': arm,
      'source_type': 'deliberate' if arm.startswith('deliberate') else 'random',
      'targeted_zone': {'deliberate_z1': 1, 'deliberate_z2': 2}.get(arm),
      'arm_definition': {
          'random': 'uniform torque U[-1,1]^8; kept so its emptiness is on '
                    'the record',
          'noisy': 'the frozen walking controller driving the shortcut BLIND '
                   '(intent=go against a redacted timetable) + Gaussian '
                   'action noise -- NOISY-CONTROLLER, not uniform random',
          'deliberate_z1': 'the normal privileged sighted teacher with a '
                           'zone-1 deliberate override: normal decision WAIT '
                           'preserved, executed decision GO',
          'deliberate_z2': 'the same targeted at zone 2; zone 1 is handled '
                           'normally and must be survived',
      }[arm],
      'obs_dim': STATE_DIM, 'observation_width': 58, 'action_dim': ACTION_DIM,
      'horizon': args.horizon,
      'collection_seed': int(seed),
      'noise_sigma': float(args.noise) if arm == 'noisy' else 0.0,
      'death_settle_substeps': int(args.settle),
      'failure_state_definition':
          'the observation the fatal transition returns after %d ctrl-free '
          'MuJoCo substeps (the AntMaze settled convention, see '
          'scripts/rebuild_failure_bank_settled.py); it is the last valid row '
          'of the episode, one per failed episode' % int(args.settle),
      'p_active_1': float(args.p_active_1),
      'p_active_2': float(args.p_active_2),
      't0_ranges': {'1': [V6.T0_MIN_1, V6.T0_MAX_1],
                    '2': [V6.T0_MIN_2, V6.T0_MAX_2]},
      'clocks_forced': False,
      'generation_uses_privileged_fields': (
          ['rockfall_active_%d (targeted zone forced active)'
           % {'deliberate_z1': 1, 'deliberate_z2': 2}[arm]]
          if arm.startswith('deliberate') else []),
      'teacher_detour_coin': 'DISABLED for this collector only (route forced '
                             'to shortcut). The benchmark teacher and the '
                             'frozen dataset keep their 0.05 coin.',
      'failure_rule': "the env's own info['failure'] -- a flagged dropped-rock "
                      'contact, with info["failure_zone"] naming the zone. V6 '
                      'has no fall-death, so an ant that merely falls over '
                      'runs to the horizon and is recorded as a timeout.',
      'n_kept': len(rows),
      'n_attempts': int(data['attempts']),
      'keep_rate': round(len(rows) / max(1, data['attempts']), 4),
      'dropped_counts': drops,
      'failure_zone_counts': zone_counts,
      'mean_ep_length': (round(float(np.mean([r['ep_length'] for r in rows])),
                               2) if rows else None),
      'privileged_fields_in_learner_npz':
          'none -- obs/act/lengths/eval_goals only; all provenance is sidecar',
  }
  path = os.path.join(args.out_dir, name + '.npz')
  np.savez_compressed(path, obs=data['obs'], act=data['act'],
                      lengths=data['lengths'], eval_goals=data['eval_goals'],
                      meta=json.dumps(meta))

  def col(key, dtype=None, none=-1):
    vals = [r[key] if r[key] is not None else none for r in rows]
    return np.array(vals, dtype) if dtype else np.array(vals)

  side = os.path.join(args.out_dir, name + '_sidecar.npz')
  np.savez_compressed(
      side,
      episode_id=col('episode_id', np.int64),
      death_row=col('death_row', np.int64),
      source_arm=col('source_arm'), source_type=col('source_type'),
      targeted_zone=col('targeted_zone', np.int64),
      outcome=col('outcome'), failure=col('failure'),
      actual_failure_zone=col('actual_failure_zone', np.int64),
      survived_zone1=col('survived_zone1'),
      normal_decision_zone1=col('normal_decision_zone1', none='none'),
      normal_decision_zone2=col('normal_decision_zone2', none='none'),
      executed_decision_zone1=col('executed_decision_zone1', none='none'),
      executed_decision_zone2=col('executed_decision_zone2', none='none'),
      deliberate_override_zone1=col('deliberate_override_zone1'),
      deliberate_override_zone2=col('deliberate_override_zone2'),
      teacher_route=col('teacher_route'),
      route_realized=col('route_realized', none='none'),
      u1=col('u1'), u2=col('u2'),
      t0_1=col('t0_1', np.int64), t0_2=col('t0_2', np.int64),
      rockfall_start_1=col('rockfall_start_1', np.int64),
      rockfall_start_2=col('rockfall_start_2', np.int64),
      rockfall_end_1=col('rockfall_end_1', np.int64),
      rockfall_end_2=col('rockfall_end_2', np.int64),
      mouth_step_zone1=col('mouth_step_zone1', np.int64),
      mouth_step_zone2=col('mouth_step_zone2', np.int64),
      band_entry_step_zone1=col('band_entry_step_zone1', np.int64),
      band_entry_step_zone2=col('band_entry_step_zone2', np.int64),
      entered_hazard_zone1=col('entered_hazard_zone1'),
      entered_hazard_zone2=col('entered_hazard_zone2'),
      hold_steps_zone1=col('hold_steps_zone1', np.int64),
      hold_steps_zone2=col('hold_steps_zone2', np.int64),
      rock_contact_zone1=col('rock_contact_zone1'),
      rock_contact_zone2=col('rock_contact_zone2'),
      env_seed=col('env_seed', np.int64),
      reset_index=col('reset_index', np.int64),
      failure_step=col('failure_step', np.int64),
      ep_length=col('ep_length', np.int64),
      furthest_x=col('furthest_x', np.float64),
      collection_seed=np.int64(seed),
      death_settle_substeps=np.int64(args.settle),
      source_npz=np.array(name + '.npz'))
  print('%-14s kept %4d / %5d attempts (%.3f) | zones %s | mean length %.1f '
        '| dropped %s' % (arm, len(rows), data['attempts'], meta['keep_rate'],
                          zone_counts, meta['mean_ep_length'] or 0.0,
                          drops or '{}'))
  print('  ->', path)
  print('  ->', side, flush=True)
  return path


def build_parser():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--arm', choices=ARMS, default='noisy')
  ap.add_argument('--all', action='store_true',
                  help='collect every arm in turn')
  ap.add_argument('--episodes', type=int, default=300,
                  help='KEPT candidates per arm')
  ap.add_argument('--seed', type=int, default=707)
  ap.add_argument('--noise', type=float, default=DEFAULT_NOISE)
  ap.add_argument('--settle', type=int, default=SETTLE_N,
                  help='ctrl-free MuJoCo substeps inside the fatal transition '
                       '(the AntMaze settled convention); 0 = freeze at '
                       'contact')
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--p-active-1', type=float, default=V6.P_ACTIVE_1)
  ap.add_argument('--p-active-2', type=float, default=V6.P_ACTIVE_2)
  ap.add_argument('--max-attempt-factor', type=int, default=40)
  ap.add_argument('--futility-attempts', type=int, default=60)
  ap.add_argument('--progress-every', type=int, default=50)
  ap.add_argument('--name', default=NAME)
  ap.add_argument('--out-dir', default=OUT_DIR)
  return ap


def main():
  args = build_parser().parse_args()
  arms = list(ARMS) if args.all else [args.arm]
  print('=' * 92)
  print('V6 FAILURE CANDIDATE COLLECTION  env %s  settle %d  p_active %.2f/%.2f'
        % (ENV_NAME, args.settle, args.p_active_1, args.p_active_2))
  print('=' * 92)
  written = []
  for i, arm in enumerate(arms):
    seed = args.seed + 137 * i
    print('\narm %s  (seed %d, %d kept wanted)' % (arm, seed, args.episodes),
          flush=True)
    try:
      data = collect(arm, args.episodes, seed, args)
    except Futile as e:
      print('  ARM PRODUCED NO USABLE CANDIDATES: %s' % e)
      print('  Recorded as a measured negative result; nothing is written for '
            'this arm and it contributes nothing to any bank.')
      os.makedirs(args.out_dir, exist_ok=True)
      with open(os.path.join(args.out_dir,
                             '%s_%s_s%d_EMPTY.json'
                             % (args.name, arm, seed)), 'w') as f:
        json.dump({'source_arm': arm, 'n_kept': e.kept,
                   'n_attempts': e.attempts, 'drop_reasons': e.drops,
                   'mean_furthest_torso_x': e.furthest_x,
                   'zone1_starts_at_x': float(V6.HAZARD_X[1][0]),
                   'verdict': 'arm produces no task-relevant failures on this '
                              'benchmark'}, f, indent=2)
      continue
    written.append(write(arm, data, args, seed))
  print()
  print('wrote %d arm file(s)' % len(written))


if __name__ == '__main__':
  main()
