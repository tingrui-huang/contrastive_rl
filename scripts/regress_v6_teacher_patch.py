r"""Regression: the deliberate-override patch does not change normal V6.

The patch in scripts/rockfall_clock_v6_teacher.py separates the NORMAL
privileged decision from the EXECUTED one and adds a zone-specific deliberate
override. Nothing else may move. This script proves that three ways, from
strongest to weakest:

  R1  REPLAY AGAINST THE FROZEN DATASET. Re-run the demonstrator collection
      protocol for the first --episodes episodes at the dataset's own seed and
      byte-compare obs, act and lengths against
      artifacts/rockfall_clock_v6/dataset/antmaze_rockfall_clock_v6.npz, plus
      the sidecar's decision / hold / mouth / band-entry / outcome columns.
      Identical actions imply identical trajectories, so this is the whole
      behaviour of the teacher, not a summary of it.

  R2  INVARIANT, NO OVERRIDE. With deliberate_override_zone=None every zone
      that decided has executed_decision == normal_decision and
      deliberate_override False.

  R3  INVARIANT, BLIND INTENT. With intent='go' the executed decisions are all
      'go' exactly as before, and the normal decisions are None -- the rule was
      not consulted, which is what lets the blind collector run against a
      redacted timetable. A 'go' recorded there would be a decision the teacher
      never made.

R1 is the acceptance evidence; R2 and R3 are the properties the collectors
rely on.

Usage:
  python scripts/regress_v6_teacher_patch.py               # 40 episodes
  python scripts/regress_v6_teacher_patch.py --episodes 120
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                 # noqa: E402
from crl import rockfall_clock_v6 as V6          # noqa: E402
import rockfall_clock_v6_teacher as CT           # noqa: E402

FROZEN_DIR = os.path.join('artifacts', 'rockfall_clock_v6', 'dataset')
FROZEN_NAME = 'antmaze_rockfall_clock_v6'
COLLECTOR = os.path.join('scripts', 'collect_rockfall_clock_v6_dataset.py')
#: columns of the frozen sidecar that are pure teacher behaviour
BEHAVIOUR_COLUMNS = (
    'decision_zone_1', 'decision_zone_2',
    'hold_steps_zone_1', 'hold_steps_zone_2',
    'release_step_zone_1', 'release_step_zone_2',
    'mouth_step_zone_1', 'mouth_step_zone_2',
    'band_entry_step_zone_1', 'band_entry_step_zone_2',
    'wait_zone_1', 'wait_zone_2', 'go_direct_zone_1', 'go_direct_zone_2',
    'route', 'route_realized', 'success', 'failure', 'timeout',
    'ep_length', 'final_xy', 'nudges',
)
RESULTS = []


def check(name, ok, detail=''):
  RESULTS.append((name, bool(ok), detail))
  print('  %-4s %-34s %s' % ('PASS' if ok else 'FAIL', name, detail),
        flush=True)
  return bool(ok)


def r1_replay(episodes, workdir, python=sys.executable):
  """Re-collect the first N episodes with the patched teacher and diff."""
  frozen = os.path.join(FROZEN_DIR, FROZEN_NAME + '.npz')
  frozen_side = os.path.join(FROZEN_DIR, FROZEN_NAME + '_sidecar.npz')
  if not (os.path.exists(frozen) and os.path.exists(frozen_side)):
    return check('R1_REPLAY_FROZEN_DATASET', False,
                 'frozen dataset missing: %s' % frozen)
  with np.load(frozen, allow_pickle=False) as d:
    meta = json.loads(str(d['meta']))
  cmd = [python, COLLECTOR,
         '--episodes', str(episodes),
         '--seed', str(int(meta['collection_seed'])),
         '--p-active-1', repr(float(meta['p_active_1'])),
         '--p-active-2', repr(float(meta['p_active_2'])),
         '--teacher-detour-prob', repr(float(meta['teacher_detour_prob'])),
         '--horizon', str(int(meta['horizon'])),
         '--out-dir', workdir, '--name', 'replay',
         '--progress-every', '10000']
  print('  replaying: %s' % ' '.join(cmd[1:]), flush=True)
  proc = subprocess.run(cmd, capture_output=True, text=True)
  if proc.returncode != 0:
    return check('R1_REPLAY_FROZEN_DATASET', False,
                 'collector failed: %s' % proc.stderr.strip().splitlines()[-1:])

  with np.load(frozen, allow_pickle=False) as a, \
       np.load(os.path.join(workdir, 'replay.npz'), allow_pickle=False) as b:
    n = int(b['obs'].shape[0])
    obs_same = bool(np.array_equal(a['obs'][:n], b['obs']))
    act_same = bool(np.array_equal(a['act'][:n], b['act']))
    len_same = bool(np.array_equal(a['lengths'][:n], b['lengths']))
    ngoal = bool(np.array_equal(a['eval_goals'][:n], b['eval_goals']))
  mismatched = []
  with np.load(frozen_side, allow_pickle=False) as a, \
       np.load(os.path.join(workdir, 'replay_sidecar.npz'),
               allow_pickle=False) as b:
    for col in BEHAVIOUR_COLUMNS:
      if col not in a.files or col not in b.files:
        mismatched.append(col + '(absent)')
        continue
      if not np.array_equal(np.asarray(a[col])[:n], np.asarray(b[col])):
        mismatched.append(col)
  ok = obs_same and act_same and len_same and ngoal and not mismatched
  return check(
      'R1_REPLAY_FROZEN_DATASET', ok,
      '%d episodes: obs %s, act %s, lengths %s, eval_goals %s; behaviour '
      'columns differing: %s'
      % (n, 'same' if obs_same else 'DIFFER', 'same' if act_same else 'DIFFER',
         'same' if len_same else 'DIFFER', 'same' if ngoal else 'DIFFER',
         mismatched or 'none'))


def _episode(env, teacher, route, override_zone, intent, horizon):
  o = env.reset()
  teacher.fresh(route=route, deliberate_override_zone=override_zone)
  info = {}
  for _ in range(horizon):
    o, r, done, info = env.step(teacher.act(o, env.schedule, intent))
    if done or r > 0:
      break
  return info


def r2_r3_invariants(episodes, seed, horizon):
  cfg, teacher = CT.make_teacher()
  cfg.rockfall_max_steps = horizon
  env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=seed)
  rng = np.random.default_rng(seed + 17)

  bad2, decided2 = [], 0
  for _ in range(episodes):
    route = 'detour' if rng.random() < CT.TEACHER_DETOUR_PROB else 'shortcut'
    _episode(env, teacher, route, None, None, horizon)
    nd, ed, ov = (teacher.normal_decisions, teacher.executed_decisions,
                  teacher.deliberate_overrides)
    for z in (1, 2):
      if nd[z] is not None or ed[z] is not None:
        decided2 += 1
      if nd[z] != ed[z] or ov[z]:
        bad2.append((z, nd[z], ed[z], ov[z]))
  check('R2_NO_OVERRIDE_NORMAL_EQ_EXEC', not bad2,
        '%d zone decisions over %d episodes; mismatches %s'
        % (decided2, episodes, bad2[:3] or 'none'))

  bad3, decided3 = [], 0
  for _ in range(episodes):
    _episode(env, teacher, 'shortcut', None, 'go', horizon)
    nd, ed = teacher.normal_decisions, teacher.executed_decisions
    for z in (1, 2):
      if ed[z] is not None:
        decided3 += 1
        if ed[z] != 'go' or nd[z] is not None:
          bad3.append((z, nd[z], ed[z]))
  check('R3_BLIND_INTENT_UNCHANGED', not bad3,
        '%d executed decisions, all go, normal all None; violations %s'
        % (decided3, bad3[:3] or 'none'))


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--episodes', type=int, default=40,
                  help='episodes replayed against the frozen dataset')
  ap.add_argument('--invariant-episodes', type=int, default=12)
  ap.add_argument('--seed', type=int, default=4321)
  ap.add_argument('--horizon', type=int, default=CT.HORIZON)
  ap.add_argument('--keep-workdir', default='')
  args = ap.parse_args()

  print('=' * 92)
  print('V6 TEACHER PATCH REGRESSION  (deliberate override off => unchanged)')
  print('=' * 92)
  workdir = args.keep_workdir or tempfile.mkdtemp(prefix='v6_regress_')
  try:
    r1_replay(args.episodes, workdir)
  finally:
    if not args.keep_workdir:
      shutil.rmtree(workdir, ignore_errors=True)
  r2_r3_invariants(args.invariant_episodes, args.seed, args.horizon)

  print('=' * 92)
  bad = [n for n, ok, _ in RESULTS if not ok]
  print('VERDICT: %s  (%d/%d)'
        % ('ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad),
           len(RESULTS) - len(bad), len(RESULTS)))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
