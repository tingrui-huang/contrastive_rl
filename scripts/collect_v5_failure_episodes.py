r"""Collect REAL V5 rock deaths, with honest provenance, for the failure bank.

The frozen V5 expert never dies (the collector asserts it, and the audited
far05 dataset has n_deaths = 0), so a failure bank needs its own episodes.
Nothing about the benchmark changes to get them: crl/rockfall_clock_v5.py is
untouched, the rockfall keeps running on its own clock, and the only thing
this script does is drive the SAME env with different behaviour policies.

FIVE SOURCE ARMS, each recorded by what it actually did, never by what it is
called. ``schedule_read`` is not a claim -- the blind arms are handed a
REDACTED schedule object that raises on 'active' / 'start' / 'end', so it is
structurally impossible for them to consult the timetable.

  random      uniform torque in [-1, 1]^8. MEASURED to be task-irrelevant on
              this benchmark: 100/100 episodes time out, mean furthest x =
              0.96, and the band starts at x = 2.6, so the ant never gets
              near the rocks. The arm is kept because "we tried uniform
              random and it produced nothing" is a result, not an omission.
  noisy       the repo's walking controller driving straight through, plus
              Gaussian action noise (--noise). This is the arm that actually
              supplies the "random / noisy" half of the bank, and it is
              called noisy-controller everywhere because that is what it is:
              competent navigation with perturbed actions, NOT uniform random.
  blind       the same controller, no noise, never reads the timetable. The
              ant walks into a burst it has no way to know about. (The f4
              'unaware' analogue.)
  deliberate  the SIGHTED expert. At the mouth it reads env.schedule, applies
              its own overlap rule, and decides WAIT -- and is then overridden
              to GO for the rest of the episode. Knew the burst overlapped its
              crossing, had the hold available, entered anyway. (The f4
              'deliberate' analogue, and like f4's it is a CURATION label:
              the resulting state carries no provenance.)
  mistimed    the sighted expert decides WAIT and does hold, but the hold is
              cut short by --cut steps, so it enters while the burst is still
              open. A local timing error near the safe policy rather than a
              rejection of it.

WHAT COUNTS AS A FAILURE is the env's own rule and nothing else:
``info['failure']``, which V5 sets only on a flagged dropped-rock contact.
An intervention that does not kill is NOT a failure -- it is recorded as
'success' or 'timeout' and dropped. V5 has no fall-death, so an ant that
merely falls over runs to the horizon and is recorded as 'timeout'; that is
reported separately so a bank can never quietly fill up with ants that
tripped.

PRIVILEGED FIELDS. The latent is FORCED ACTIVE by default (--p-active 1.0):
deaths only exist under an active latent and drawing it at 0.30 would throw
away 70% of the compute. That is generation, not just selection, and it is
recorded as such in the manifest. The burst schedule is never forced: t0 is
drawn by the env's own rng every reset, exactly as in the benchmark. Nothing
privileged enters obs/act.

Output (artifacts/v5_failneg/failures/<name>.npz + _sidecar.npz):
  learner npz: obs [N, L, 58], act [N, L, 8], lengths [N], eval_goals [N, 2],
               meta (json). Same contract as the V5 dataset collector, so
               these episodes CAN be merged into a training set if that is
               ever wanted -- it is a separate, explicit choice and is off by
               default (see scripts/run_v5_failneg.py --train-npz).
  sidecar npz: per-episode provenance -- source arm, teacher decision,
               executed intent, whether the timetable was read, intervention
               class, outcome, death row, schedule, mouth/band steps. AUDIT
               ONLY. Never a training input, never a bank input.

Usage:
  python scripts/collect_v5_failure_episodes.py --arm noisy --episodes 400
  python scripts/collect_v5_failure_episodes.py --arm deliberate --episodes 300
  python scripts/collect_v5_failure_episodes.py --all --episodes 400
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                                # noqa: E402
from crl import rockfall_clock_v5 as V5                         # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X, HAZARD_HALF_Y    # noqa: E402
import rockfall_clock_v5_teacher as CT                          # noqa: E402

OUT_DIR = 'artifacts/v5_failneg/failures'
NAME = 'v5_failures'
HORIZON = 400
ARMS = ('random', 'noisy', 'blind', 'deliberate', 'mistimed')
#: the "random / noisy" half of the bank and the "competent unsafe entry" half.
NOISY_ARMS = ('random', 'noisy')
COMPETENT_ARMS = ('deliberate', 'mistimed')
DEFAULT_NOISE = 0.3
DEFAULT_CUT = 30


class RedactedSchedule(dict):
  """The timetable with the timetable removed.

  ClockV5Teacher.act() needs a schedule object, but only its sighted branch
  dereferences 'active' / 'start' / 'end'. Handing the blind arms this object
  turns "we promise it did not peek" into "it could not have peeked": any read
  of a timetable field raises. 't' stays readable because it is the env step
  counter, which the ant can count for itself.
  """

  FORBIDDEN = ('active', 'start', 'end')

  def __getitem__(self, key):
    if key in self.FORBIDDEN:
      raise KeyError(
          'blind arm read the privileged timetable field %r -- this is a bug, '
          'not a data point' % key)
    return dict.__getitem__(self, key)


class TrackedSchedule(dict):
  """Snapshot of the privileged timetable that records actual field reads."""

  def __init__(self, source):
    super().__init__(source)
    self.fields_read = set()

  def __getitem__(self, key):
    self.fields_read.add(key)
    return dict.__getitem__(self, key)


def in_band(xy):
  return (abs(float(xy[1])) < HAZARD_HALF_Y
          and HAZARD_X[0] <= float(xy[0]) <= HAZARD_X[1])


def in_corridor_row(xy):
  """Bottom corridor, the row the band sits in. The rocks' lethal reach runs
  ~0.9 west and ~0.7 east of the band edges (see the teacher's CROSS_MAX /
  MOUTH_TO_BAND_MIN derivation), so a real rock death just outside the band
  is still a rock death; it is not, however, allowed to be somewhere else in
  the maze entirely."""
  return abs(float(xy[1])) < HAZARD_HALF_Y


class Futile(Exception):
  """An arm that does not produce failures on this benchmark. Carries the
  numbers that say so, because "we tried it and it gave nothing" is only a
  result if the nothing is quantified."""

  def __init__(self, arm, attempts, kept, drops, furthest, dropped_rows):
    super().__init__(
        'arm %r produced %d failures in %d episodes; drop reasons %s; mean '
        'furthest torso x %.2f (the band starts at x = %.1f)'
        % (arm, kept, attempts, drops, furthest, HAZARD_X[0]))
    self.arm, self.attempts, self.kept = arm, attempts, kept
    self.drops, self.furthest_x = drops, furthest
    self.dropped_rows = dropped_rows


def drop_summary(dropped):
  out = {}
  for d in dropped:
    out[d['drop_reason']] = out.get(d['drop_reason'], 0) + 1
  return out


def furthest_x(dropped):
  xs = [d['furthest_x'] for d in dropped if d.get('furthest_x') is not None]
  return float(np.mean(xs)) if xs else float('nan')


def run_episode(env, teacher, arm, rng, noise, cut, horizon, record):
  """One episode of one arm. ``record(o, a, o2, info)`` stores the rows.

  Returns the provenance row. The teacher's own state (decision, release_step,
  hold_steps) is read AFTER the episode, so the row says what happened rather
  than what was intended.
  """
  o = env.reset(rockfall_active=True)
  teacher.fresh(route='shortcut')
  blind = arm in ('random', 'noisy', 'blind')
  schedule_fields_read = set()
  intervention_step = None
  info = {}
  t = 0
  far_x = -np.inf
  for t in range(horizon):
    tracked = None
    if blind:
      sched = RedactedSchedule(t=env.schedule['t'])
    else:
      tracked = TrackedSchedule(env.schedule)
      sched = tracked
    if arm == 'random':
      a = rng.uniform(-1.0, 1.0, 8).astype(np.float32)
    elif arm == 'noisy':
      a = teacher.act(o, sched, 'go')
      a = np.clip(a + rng.normal(0.0, noise, 8), -1.0, 1.0).astype(np.float32)
    elif arm == 'blind':
      a = teacher.act(o, sched, 'go')
    elif arm == 'deliberate':
      #: run the sighted rule until it commits to WAIT at the mouth, then
      #: force GO. teacher.decision stays 'wait': the record keeps what the
      #: expert knew, not what it was made to do.
      if teacher.decision == 'wait':
        if intervention_step is None:
          intervention_step = t
        a = teacher.act(o, sched, 'go')
      else:
        a = teacher.act(o, sched)
    elif arm == 'mistimed':
      if teacher.decision == 'wait' and teacher.release_step is not None:
        if t >= teacher.release_step - cut:
          if intervention_step is None:
            intervention_step = t
          a = teacher.act(o, sched, 'go')
        else:
          a = teacher.act(o, sched)
      else:
        a = teacher.act(o, sched)
    else:
      raise ValueError('unknown arm %r' % arm)
    if tracked is not None:
      schedule_fields_read.update(tracked.fields_read)
    o2, r, done, info = env.step(a)
    record(o, a, o2, info)
    o = o2
    far_x = max(far_x, float(o[0]))
    if done or r > 0:
      break
  failure = bool(info.get('failure'))
  success = bool(info.get('success'))
  outcome = 'rock_death' if failure else ('success' if success else 'timeout')
  return {
      'source_arm': arm,
      'outcome': outcome,
      'failure': failure,
      'teacher_decision': teacher.decision if not blind else None,
      'schedule_read': bool(schedule_fields_read & {'active', 'start', 'end'}),
      'schedule_fields_read': ','.join(sorted(schedule_fields_read)),
      'intervention': {'random': 'uniform_action',
                       'noisy': 'action_noise',
                       'blind': 'none',
                       'deliberate': 'override_wait_to_go',
                       'mistimed': 'early_release'}[arm],
      'intervention_step': intervention_step,
      'executed_intent': 'go' if blind or intervention_step is not None
                         else (teacher.decision or 'go'),
      'noise_sigma': float(noise) if arm == 'noisy' else 0.0,
      'cut_steps': int(cut) if arm == 'mistimed' else 0,
      'rockfall_active': bool(env.privileged_rockfall_active),
      'rockfall_start': info.get('rockfall_start'),
      'rockfall_end': info.get('rockfall_end'),
      'mouth_step': info.get('mouth_step'),
      'band_entry_step': info.get('band_entry_step'),
      'entered_hazard': bool(info.get('entered_hazard')),
      'rock_waves': int(info.get('rock_waves', 0)),
      'hold_steps': int(teacher.hold_steps_done),
      'release_step': teacher.release_step,
      'ep_length': int(t + 1),
      'furthest_x': round(float(far_x), 4),
      'final_xy': [round(float(o[0]), 4), round(float(o[1]), 4)],
      'death_in_band': failure and in_band(o[:2]),
      'death_in_corridor_row': failure and in_corridor_row(o[:2]),
  }


def collect(arm, episodes, seed, noise, cut, horizon, keep_rule,
            max_attempt_factor=40, futility_attempts=60):
  cfg, teacher = CT.make_teacher()
  cfg.rockfall_max_steps = horizon
  env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=seed)
  rng = np.random.default_rng(seed + 90_001)
  L = horizon + 1
  obs = np.zeros((episodes, L, 58), np.float32)
  act = np.zeros((episodes, L, 8), np.float32)
  lengths = np.zeros(episodes, np.int64)
  eval_goals = np.zeros((episodes, 2), np.float32)
  death_row = np.full(episodes, -1, np.int64)
  rows, dropped = [], []
  kept = 0
  attempts = 0
  while kept < episodes:
    attempts += 1
    buf = {'n': 0}

    def record(o, a, o2, info, _buf=buf, _k=kept):
      i = _buf['n']
      if i == 0:
        obs[_k, 0] = o
        eval_goals[_k] = o[29:31]
      act[_k, i] = a
      obs[_k, i + 1] = o2
      _buf['n'] = i + 1

    row = run_episode(env, teacher, arm, rng, noise, cut, horizon, record)
    n = buf['n']
    keep, why = keep_rule(row, n)
    if not keep:
      obs[kept, :] = 0.0
      act[kept, :] = 0.0
      dropped.append(dict(row, drop_reason=why))
      #: futility bail: an arm that has produced NOTHING in this many
      #: attempts is not slow, it is the wrong arm. Stop and report it as a
      #: measured negative rather than burning the node on it.
      if kept == 0 and attempts >= futility_attempts:
        raise Futile(arm, attempts, kept, drop_summary(dropped),
                     furthest_x(dropped), dropped)
      if attempts > max_attempt_factor * episodes:
        raise Futile(arm, attempts, kept, drop_summary(dropped),
                     furthest_x(dropped), dropped)
      continue
    #: the death observation is the LAST recorded row: V5 returns the
    #: post-contact obs on the fatal step and the loop breaks there, so there
    #: is exactly one post-death frame and it cannot be double counted.
    lengths[kept] = n + 1
    death_row[kept] = n
    row['episode_id'] = kept
    row['death_row'] = int(n)
    rows.append(row)
    kept += 1
    if kept % 50 == 0:
      print('  %s: %d/%d kept (%d attempts, %d dropped)'
            % (arm, kept, episodes, attempts, len(dropped)), flush=True)
  return dict(obs=obs, act=act, lengths=lengths, eval_goals=eval_goals,
              death_row=death_row, rows=rows, dropped=dropped,
              attempts=attempts)


def keep_rule_factory(require_in_band):
  def keep(row, n_steps):
    if not row['failure']:
      return False, 'not_a_rock_death:' + row['outcome']
    if n_steps < 2:
      return False, 'truncated_before_two_observations'
    if not row['death_in_corridor_row']:
      return False, 'death_outside_the_corridor_row'
    if require_in_band and not row['death_in_band']:
      return False, 'death_outside_the_band'
    return True, ''
  return keep


def write(arm, data, args, out_dir, collection_seed):
  os.makedirs(out_dir, exist_ok=True)
  rows = data['rows']
  # --all deliberately gives every arm an independent RNG stream.  Name and
  # provenance by the seed ACTUALLY passed to collect(), not merely the base
  # seed supplied on the command line.
  name = f'{args.name}_{arm}_s{collection_seed}'
  drop_counts = {}
  for d in data['dropped']:
    drop_counts[d['drop_reason']] = drop_counts.get(d['drop_reason'], 0) + 1
  meta = {
      'name': name,
      'env': CT.ENV_NAME,
      'source_arm': arm,
      'arm_definition': {
          'random': 'uniform torque U[-1,1]^8; measured task-irrelevant on '
                    'this benchmark (never reaches the band)',
          'noisy': 'the repo walking controller driving straight through '
                   '(blind) + Gaussian action noise; NOISY-CONTROLLER, not '
                   'uniform random',
          'blind': 'the walking controller driving straight through, never '
                   'reads the timetable (unaware)',
          'deliberate': 'the sighted expert read env.schedule at the mouth, '
                        'its overlap rule decided WAIT, and it was overridden '
                        'to GO for the rest of the episode',
          'mistimed': 'the sighted expert decided WAIT and held, but the hold '
                      'was cut short by cut_steps so it entered while the '
                      'burst was still open',
      }[arm],
      'obs_dim': 29, 'goal_dim': 29, 'action_dim': 8,
      'horizon': args.horizon,
      'collection_seed': int(collection_seed),
      'collection_base_seed': int(args.seed),
      'noise_sigma': float(args.noise) if arm == 'noisy' else 0.0,
      'cut_steps': int(args.cut) if arm == 'mistimed' else 0,
      'p_active': 1.0,
      'generation_uses_privileged_fields': ['rockfall_active'],
      'generation_note':
          'the latent is FORCED ACTIVE (deaths exist only under an active '
          'latent). The burst schedule t0 is NOT forced: the env draws it '
          'from its own rng every reset, exactly as in the benchmark. The '
          'rockfall is never made always-on and is never triggered by the '
          "ant's arrival.",
      'failure_rule': "env info['failure'] -- a flagged dropped-rock contact. "
                      'V5 has no fall-death, so an ant that merely falls over '
                      'runs to the horizon and is recorded as a timeout.',
      'keep_rule': {'require_rock_death': True,
                    'require_corridor_row': True,
                    'require_in_band': bool(args.require_in_band)},
      'n_kept': len(rows),
      'n_attempts': int(data['attempts']),
      'keep_rate': round(len(rows) / max(1, data['attempts']), 4),
      'dropped_counts': drop_counts,
      'death_in_band_frac': round(float(np.mean(
          [r['death_in_band'] for r in rows])), 4) if rows else None,
      'schedule_read_frac': round(float(np.mean(
          [r['schedule_read'] for r in rows])), 4) if rows else None,
      'mean_ep_length': round(float(np.mean(
          [r['ep_length'] for r in rows])), 2) if rows else None,
      'privileged_fields_in_learner_npz': 'none -- obs/act/lengths/eval_goals '
                                          'only; all provenance is sidecar',
  }
  path = os.path.join(out_dir, f'{name}.npz')
  np.savez_compressed(path, obs=data['obs'], act=data['act'],
                      lengths=data['lengths'], eval_goals=data['eval_goals'],
                      meta=json.dumps(meta))

  def col(key, dtype=None, none=-1):
    v = [r[key] if r[key] is not None else none for r in rows]
    return np.array(v, dtype) if dtype else np.array(v)

  side = os.path.join(out_dir, f'{name}_sidecar.npz')
  np.savez_compressed(
      side,
      episode_id=col('episode_id', np.int64),
      source_arm=col('source_arm'),
      outcome=col('outcome'),
      teacher_decision=col('teacher_decision', none='none'),
      schedule_read=col('schedule_read'),
      schedule_fields_read=col('schedule_fields_read'),
      intervention=col('intervention'),
      intervention_step=col('intervention_step', np.int64),
      executed_intent=col('executed_intent'),
      noise_sigma=col('noise_sigma', np.float64),
      cut_steps=col('cut_steps', np.int64),
      rockfall_active=col('rockfall_active'),
      rockfall_start=col('rockfall_start', np.int64),
      rockfall_end=col('rockfall_end', np.int64),
      mouth_step=col('mouth_step', np.int64),
      band_entry_step=col('band_entry_step', np.int64),
      entered_hazard=col('entered_hazard'),
      rock_waves=col('rock_waves', np.int64),
      hold_steps=col('hold_steps', np.int64),
      release_step=col('release_step', np.int64),
      ep_length=col('ep_length', np.int64),
      death_row=col('death_row', np.int64),
      death_in_band=col('death_in_band'),
      death_in_corridor_row=col('death_in_corridor_row'),
      collection_seed=np.int64(collection_seed),
      collection_base_seed=np.int64(args.seed))
  if data['dropped']:
    dropped_path = os.path.join(out_dir, f'{name}_dropped.json')
    with open(dropped_path, 'w') as f:
      json.dump({'source_arm': arm,
                 'collection_seed': int(collection_seed),
                 'collection_base_seed': int(args.seed),
                 'definition': 'non-fatal or otherwise ineligible attempts; '
                               'audit only, never learner or bank input',
                 'attempts': data['dropped']}, f, indent=2)
  print('%-11s kept %4d / %5d attempts (%.3f) | death in band %.3f | '
        'mean length %.1f | dropped %s'
        % (arm, len(rows), data['attempts'], meta['keep_rate'],
           meta['death_in_band_frac'] or 0.0, meta['mean_ep_length'] or 0.0,
           drop_counts))
  print('  ->', path)
  print('  ->', side, flush=True)
  if data['dropped']:
    print('  ->', dropped_path, '(dropped-attempt audit)', flush=True)
  return path


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--arm', choices=ARMS, default='noisy')
  ap.add_argument('--all', action='store_true',
                  help='collect every arm in turn (the random arm included, '
                       'so its emptiness is on the record)')
  ap.add_argument('--episodes', type=int, default=400,
                  help='number of KEPT failure episodes per arm')
  ap.add_argument('--seed', type=int, default=808)
  ap.add_argument('--noise', type=float, default=DEFAULT_NOISE,
                  help='Gaussian action-noise sigma for the noisy arm')
  ap.add_argument('--cut', type=int, default=DEFAULT_CUT,
                  help='steps by which the mistimed arm cuts its hold short')
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--require-in-band', action='store_true',
                  help='drop rock deaths that happened just outside the band '
                       '(default: keep them; they are real rock deaths in the '
                       "corridor row, inside the pattern's lethal reach)")
  ap.add_argument('--max-attempt-factor', type=int, default=40,
                  help='give up after this many attempts per wanted episode')
  ap.add_argument('--futility-attempts', type=int, default=60,
                  help='give up immediately if an arm has produced ZERO '
                       'failures in this many episodes')
  ap.add_argument('--name', default=NAME)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  arms = list(ARMS) if args.all else [args.arm]
  keep = keep_rule_factory(args.require_in_band)
  written = []
  print('=' * 88)
  print('V5 FAILURE COLLECTION  env %s  seed %d' % (CT.ENV_NAME, args.seed))
  print('=' * 88)
  for i, arm in enumerate(arms):
    arm_seed = args.seed + 137 * i
    print('\narm %s (%d kept episodes wanted)' % (arm, args.episodes),
          flush=True)
    print('  rng seed %d (base seed %d, arm offset %d)'
          % (arm_seed, args.seed, 137 * i), flush=True)
    try:
      data = collect(arm, args.episodes, arm_seed, args.noise,
                     args.cut, args.horizon, keep,
                     max_attempt_factor=args.max_attempt_factor,
                     futility_attempts=args.futility_attempts)
    except Futile as e:
      print('  ARM PRODUCED NO USABLE FAILURES: %s' % e)
      print('  This is a measured negative result, not an omission. No '
            'episodes are written for this arm and it contributes nothing '
            'to any bank.')
      os.makedirs(args.out_dir, exist_ok=True)
      with open(os.path.join(args.out_dir,
                             f'{args.name}_{arm}_s{arm_seed}_EMPTY.json'),
                'w') as f:
        json.dump({'source_arm': arm, 'n_kept': int(e.kept),
                   'collection_seed': int(arm_seed),
                   'collection_base_seed': int(args.seed),
                   'n_attempts': int(e.attempts),
                   'drop_reasons': e.drops,
                   'mean_furthest_torso_x': e.furthest_x,
                   'band_starts_at_x': float(HAZARD_X[0]),
                   'verdict': 'arm produces no task-relevant failures on this '
                              'benchmark',
                   'attempts_detail': e.dropped_rows}, f, indent=2)
      continue
    written.append(write(arm, data, args, args.out_dir, arm_seed))
  print()
  print('wrote %d arm file(s)' % len(written))


if __name__ == '__main__':
  main()
