r"""Do z_v1 and f4 induce the SAME (position, action) -> sink map?

Diagnostic only. Nothing is trained and nothing is written outside --out-dir.

WHY THIS EXISTS. Everything in artifacts/propensity_manski (the Manski bound
audit) and artifacts/propensity_lipschitz (the L_p audit) is computed from
three quantities:

    pi_B(s, A)     the behaviour policy's propensity over an action bin
    Ttilde(s,A)    the observational P(sink | s, X in A)
    p(s, do(a))    the true interventional sink probability, from the simulator

None of them reads the observation. They depend only on the positions, the
actions, the hidden bits and the sink rule. Frame stacking changes the
observation and nothing else -- TwoRouteSwampWindyF4Env overrides only
_get_obs, reset and step, and self.state stays the 2-D vector the parent's
substep loop operates on -- so those audits should carry over to f4 verbatim,
and re-running them would only reproduce known numbers.

"Should" is not "does". This script checks it instead of asserting it, at a
cost of seconds rather than the twenty minutes a full re-run would take. Under
common random numbers -- same start position, same hidden bits, same
action-noise stream -- the two envs must agree on the endpoint EXACTLY and on
the death label with zero disagreements. If they do not, the carry-over claim
is false and both audits have to be re-run against the f4 dataset.

The behaviour-policy half of the claim is checked elsewhere and from the other
direction: the two collectors produce identical died and reached@0.5 rates to
four decimals, which is what pins pi_B and Ttilde.

Usage:
  python scripts/check_f4_matches_z_dynamics.py
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402

ENV_Z = 'point_two_route_swamp_windy_z_v1'
ENV_F4 = 'point_two_route_swamp_windy_f4_v0'
OUT_DIR = 'artifacts/f4_dynamics_check'
TRACKED = ['crl/envs.py', 'scripts/check_f4_matches_z_dynamics.py']
QUERIES = (('holding_mid', (2.50, 3.50)), ('holding_edge', (2.90, 3.50)),
           ('mouth', (3.10, 3.50)), ('corridor_1', (4.10, 3.50)))
A_ENTER = np.array([1.0, 0.0], np.float32)


def step_dead(env, xy, bits, action, seed):
  """One step from a clean alive state under a pinned (U, xi)."""
  env.set_auto_resample(False)
  env._dead = False
  if hasattr(env, '_z'):
    env._z = 0.0
  if hasattr(env, '_frames'):
    env._frames = None                 # reseeded lazily from the new position
  env.state = np.asarray(xy, float).copy()
  env.set_swamp(bits)
  env._rng = np.random.default_rng(int(seed))
  env.step(np.asarray(action, np.float32))
  return env.state.copy(), bool(env._dead)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--pairs', type=int, default=20000)
  ap.add_argument('--oracle', type=int, default=8000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  def git(*a):
    try:
      return subprocess.check_output(['git'] + list(a),
                                     cwd=_ROOT).decode().strip()
    except Exception:                             # pylint: disable=broad-except
      return ''

  os.makedirs(args.out_dir, exist_ok=True)
  out = {'analysis_script': 'scripts/check_f4_matches_z_dynamics.py',
         'env_a': ENV_Z, 'env_b': ENV_F4,
         'code_commit': git('log', '-1', '--format=%H', '--', *TRACKED),
         'dirty': bool(git('status', '--porcelain', '--', *TRACKED)),
         'config': {'pairs': args.pairs, 'oracle': args.oracle,
                    'seed': args.seed}}

  print('=' * 92)
  print('DO z_v1 AND f4 INDUCE THE SAME (position, action) -> sink MAP?')
  print('=' * 92)
  print('  code commit %s  dirty %s' % (out['code_commit'][:12], out['dirty']))

  ez = envs_mod.make_env(ENV_Z, Config(env_name=ENV_Z), seed=args.seed)
  ef = envs_mod.make_env(ENV_F4, Config(env_name=ENV_F4), seed=args.seed)
  rng = np.random.default_rng(args.seed)

  worst, n_dis = 0.0, 0
  for _ in range(args.pairs):
    xy = np.array([rng.uniform(2.0, 6.0), rng.uniform(3.0, 4.0)])
    bits = (rng.random(3) < ez.active_prob).tolist()
    act = rng.uniform(-1, 1, 2).astype(np.float32)
    s = int(rng.integers(1 << 30))
    pz, dz = step_dead(ez, xy, bits, act, s)
    pf, df = step_dead(ef, xy, bits, act, s)
    worst = max(worst, float(np.abs(pz - pf).max()))
    n_dis += int(dz != df)
  print('\n  %s paired one-step comparisons under CRN'
        % format(args.pairs, ','))
  print('    max |endpoint_z_v1 - endpoint_f4| = %.3e   (must be 0)' % worst)
  print('    death-label disagreements         = %d   (must be 0)' % n_dis)
  out['paired'] = {'n': args.pairs, 'max_endpoint_diff': worst,
                   'death_label_disagreements': int(n_dis)}

  print('\n  the interventional probability the Manski audit reports:')
  print('    %-14s%12s%12s%10s' % ('query', 'z_v1', 'f4', 'equal'))
  probs = {}
  for name, xy in QUERIES:
    vals = []
    for env in (ez, ef):
      r = np.random.default_rng(7)
      n = 0
      for _ in range(args.oracle):
        bits = (r.random(3) < env.active_prob).tolist()
        n += step_dead(env, xy, bits, A_ENTER, int(r.integers(1 << 30)))[1]
      vals.append(n / float(args.oracle))
    probs[name] = {'z_v1': vals[0], 'f4': vals[1], 'equal': vals[0] == vals[1]}
    print('    %-14s%12.4f%12.4f%10s'
          % (name, vals[0], vals[1], 'YES' if vals[0] == vals[1] else 'NO'))
  out['oracle_p_sink_do_enter'] = probs

  ok = (worst == 0.0 and n_dis == 0
        and all(v['equal'] for v in probs.values()))
  out['carry_over_valid'] = bool(ok)
  print('\n%s' % ('=' * 92))
  if ok:
    print('CARRY-OVER VALID. artifacts/propensity_manski and')
    print('artifacts/propensity_lipschitz apply to f4 unchanged; re-running')
    print('them against the f4 dataset would only reproduce known numbers.')
  else:
    print('CARRY-OVER INVALID -- the two envs disagree. Both audits must be')
    print('re-run against the f4 dataset before any f4 result is interpreted.')
  p = os.path.join(args.out_dir, 'f4_dynamics_check.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('wrote %s' % p)
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
