r"""Validation audit for point_two_route_swamp_windy_f4_v0 (frame stacking).

Diagnostic / acceptance only. No training, no dataset is written, and no
existing 2-D, z_v0 or z_v1 artifact is touched.

WHAT IS BEING ACCEPTED. The 27-run null on the 2-D benchmark was root-caused
structurally: a frozen dead agent and an agent walking through occupy the SAME
(x, y), so obs_dim = 2 cannot represent the difference and the failure bank
cancels into a constant shift. The z envs fixed that by adding physics (a
sinking depth). This variant fixes it by adding MEMORY instead -- the learner
sees the last 4 positions -- so the separating quantity is the velocity, and
the physics stay exactly two-dimensional.

That is only worth having if it actually separates. Gate 6 therefore re-measures
the number that motivated the whole line: the fraction of dead states that have
a LIVE state essentially on top of them, first in the 2-D metric (where it was
0.9915) and then in the stacked metric.

THE GATES

  1  registration and dims        obs_dim = goal_dim = 8, obs width 16
  2  ordering                     state[0:2] is the CURRENT position
  3  hidden-bit invariance        flipping U leaves the observation identical
  4  frozen signature             after death the stack converges to [s*]*4
  5  moving signature             a passing agent's frames differ
  6  separation at matched XY     the 2-D collision vs the stacked one
  7  old envs unaffected          2-D / z_v0 / z_v1 still build with old dims
  8  warm-up cost                 how many steps before the stack is informative

GATE 8 IS A COST, NOT A PASS/FAIL. At reset there is no history, so the stack
is seeded with copies of the start position and a fresh agent momentarily looks
"frozen". That is unavoidable for any frame stack; it is measured and reported
rather than papered over, because it bounds how early the zero-velocity
signature can be trusted.

The behaviour policy is imported from the accepted 2-D collector, not
reimplemented, so the state distribution in gate 6 is the one the real datasets
would have.

Usage:
  python scripts/audit_swamp_windy_f4.py --episodes 300     # smoke
  python scripts/audit_swamp_windy_f4.py                    # full
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from crl import envs as envs_mod                          # noqa: E402
from crl.config import Config                             # noqa: E402
from scripts.collect_swamp_windy import make_windy_teacher  # noqa: E402

ENV_F4 = 'point_two_route_swamp_windy_f4_v0'
OLD_ENVS = {'point_two_route_swamp_windy_v0': (2, 2),
            'point_two_route_swamp_windy_z_v0': (3, 3),
            'point_two_route_swamp_windy_z_v1': (3, 3)}
OUT_DIR = 'artifacts/swamp_windy_f4'
TRACKED = ['crl/envs.py', 'scripts/audit_swamp_windy_f4.py']
NF = 4
TAU = 0.05                       # the "essentially on top of it" radius


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--episodes', type=int, default=2000)
  ap.add_argument('--random-frac', type=float, default=0.2)
  ap.add_argument('--force-safe-prob', type=float, default=0.05)
  ap.add_argument('--teacher-noise', type=float, default=0.15)
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
  out = {'analysis_script': 'scripts/audit_swamp_windy_f4.py', 'env': ENV_F4,
         'code_commit': git('log', '-1', '--format=%H', '--', *TRACKED),
         'dirty': bool(git('status', '--porcelain', '--', *TRACKED)),
         'config': {'episodes': args.episodes, 'seed': args.seed,
                    'n_frames': NF, 'tau': TAU}}
  gates = {}

  def gate(name, ok, detail=''):
    gates[name] = {'pass': bool(ok), 'detail': detail}
    print('  %-4s %-28s %s' % ('PASS' if ok else 'FAIL', name, detail))

  print('=' * 98)
  print('ACCEPTANCE AUDIT -- point_two_route_swamp_windy_f4_v0 (frame stacking)')
  print('=' * 98)
  print('  code commit %s  dirty %s' % (out['code_commit'][:12], out['dirty']))

  cfg = Config(env_name=ENV_F4)
  env = envs_mod.make_env(ENV_F4, cfg, seed=args.seed)

  # ------------------------------------------------------------------- 1, 2
  print('\n1-2. REGISTRATION, DIMS AND ORDERING')
  obs = env.reset()
  gate('1_class', type(env).__name__ == 'TwoRouteSwampWindyF4Env',
       type(env).__name__)
  gate('1_dims', env.obs_dim == 2 * NF and env.goal_dim == 2 * NF,
       'obs_dim %d goal_dim %d' % (env.obs_dim, env.goal_dim))
  gate('1_obs_width', obs.shape == (4 * NF,), 'obs shape %s' % (obs.shape,))
  gate('1_config_filled', cfg.obs_dim == 2 * NF and cfg.goal_dim == 2 * NF,
       'config obs_dim %d goal_dim %d' % (cfg.obs_dim, cfg.goal_dim))
  for _ in range(6):
    obs, _, _, _ = env.step(np.array([0.7, 0.1], np.float32))
  # the observation is float32 by design, so the comparison is against the
  # float32 cast of the (float64) internal state, not against the float64 value
  gate('2_current_first',
       float(np.abs(obs[:2]
                    - np.asarray(env.state, np.float32)).max()) == 0.0,
       'obs[0:2] %s == float32(state) exactly' % np.round(obs[:2], 4))
  gate('2_goal_tiled',
       float(np.abs(obs[2 * NF:].reshape(NF, 2)
                    - np.asarray(env.goal, np.float32)).max()) == 0.0,
       'goal block is tile(GOAL, %d)' % NF)

  # ---------------------------------------------------------------------- 3
  print('\n3. HIDDEN-BIT INVARIANCE  (U must never reach the learner)')
  worst = 0.0
  rng = np.random.default_rng(args.seed)
  for _ in range(400):
    env.reset()
    env.set_auto_resample(False)
    seq = rng.uniform(-1, 1, (5, 2)).astype(np.float32)
    obs_by_bits = []
    for bits in ([0, 0, 0], [1, 1, 1], [1, 0, 1]):
      env.reset()
      env.set_auto_resample(False)
      env.set_swamp(bits)
      env._rng = np.random.default_rng(12345)
      o = None
      for a in seq:
        o, _, _, _ = env.step(a)
        if env.dead:                 # only PRE-contact obs must be invariant
          o = None
          break
      if o is not None:
        obs_by_bits.append(o)
    if len(obs_by_bits) > 1:
      ref = obs_by_bits[0]
      for o in obs_by_bits[1:]:
        worst = max(worst, float(np.abs(o - ref).max()))
  gate('3_hidden_bits_invariant', worst == 0.0,
       'max |obs difference| across bit patterns = %.3e' % worst)
  env.set_auto_resample(True)

  # ------------------------------------------------------------------- 4, 5
  print('\n4-5. FROZEN vs MOVING SIGNATURE')

  def rollout(policy, n_steps=50, seed=0):
    env.reset()
    env._rng = np.random.default_rng(seed)
    fr, dead = [], []
    for t in range(n_steps):
      env.step(policy(t))
      fr.append(env.frames.copy())
      dead.append(env.dead)
    return np.array(fr), np.array(dead)

  # drive straight into the corridor until something dies
  spread_dead, spread_live, n_dead_seen = [], [], 0
  for s in range(400):
    fr, dd = rollout(lambda t: np.array([1.0, 0.0], np.float32), 30, seed=s)
    sp = np.array([np.abs(f - f[0]).max() for f in fr])   # frame spread
    if dd.any():
      i = int(np.argmax(dd))
      if i + NF < len(fr):
        spread_dead.append(float(sp[i + NF]))             # NF steps after death
        n_dead_seen += 1
    live = np.where(~dd)[0]
    live = live[live >= NF]
    if len(live):
      spread_live.append(float(sp[live].mean()))
  sd = np.array(spread_dead)
  sl = np.array(spread_live)
  gate('4_frozen_signature', sd.size > 0 and sd.max() == 0.0,
       '%d deaths; max frame spread %d steps after death = %.3e'
       % (n_dead_seen, NF, sd.max() if sd.size else float('nan')))
  gate('5_moving_signature', sl.size > 0 and sl.min() > 0.0,
       'min mean frame spread while alive = %.4f'
       % (sl.min() if sl.size else float('nan')))

  # ---------------------------------------------------------------------- 6
  print('\n6. SEPARATION AT MATCHED XY  (the number that motivated all of this)')
  print('  Rolling out %d episodes with the ACCEPTED behaviour policy '
        '(imported,' % args.episodes)
  print('  not reimplemented) so the state distribution matches the real '
        'datasets.')
  rng = np.random.default_rng(args.seed)
  teacher = make_windy_teacher(env, rng, args.force_safe_prob)
  n_random = int(round(args.episodes * args.random_frac))
  S, D = [], []
  for ep in range(args.episodes):
    env.reset()
    memo = {}
    use_rand = ep < n_random
    for t in range(50):
      if use_rand:
        a = rng.uniform(-1, 1, 2).astype(np.float32)
      else:
        a = np.asarray(teacher(env.state, env.goal, memo), np.float32)
        a = np.clip(a + rng.normal(0, args.teacher_noise, 2), -1, 1)
      env.step(a)
      if t >= NF:                       # skip the warm-up window (gate 8)
        S.append(env.state_f4.copy())
        D.append(env.dead)
  S = np.asarray(S, np.float32)
  D = np.asarray(D, bool)
  reg = ((S[:, 0] >= 3.0) & (S[:, 0] < 6.0)
         & (S[:, 1] >= 3.0) & (S[:, 1] < 4.0))
  dm, lm = reg & D, reg & (~D)
  nd, nl = int(dm.sum()), int(lm.sum())
  print('  swamp-region states: %s dead / %s alive' % (format(nd, ','),
                                                       format(nl, ',')))
  sep = {}
  if nd > 0 and nl > 0:
    m = min(nd, nl, 4000)
    di = rng.choice(np.where(dm)[0], m, False)
    li = rng.choice(np.where(lm)[0], m, False)
    for label, cols in (('xy_only (the 2-D benchmark)', slice(0, 2)),
                        ('stacked (%d frames)' % NF, slice(0, 2 * NF))):
      Xd, Xl = S[di, cols], S[li, cols]
      nn = np.empty(len(Xd))
      for b in range(0, len(Xd), 512):
        d2 = ((Xd[b:b + 512, None, :] - Xl[None, :, :]) ** 2).sum(-1)
        nn[b:b + 512] = np.sqrt(d2.min(1))
      sep[label] = {'n': int(m), 'median_nn': float(np.median(nn)),
                    'frac_within_tau': float((nn < TAU).mean())}
      print('  %-28s  median d(dead, nearest alive) %.4f   frac < %.2f: %.4f'
            % (label, np.median(nn), TAU, (nn < TAU).mean()))
    f2 = sep['xy_only (the 2-D benchmark)']['frac_within_tau']
    f8 = sep['stacked (%d frames)' % NF]['frac_within_tau']
    gate('6_stacking_separates', f8 < f2,
         'collision %.4f (2-D) -> %.4f (stacked)' % (f2, f8))
  out['6_separation'] = sep

  # ---------------------------------------------------------------------- 7
  print('\n7. OLD ENVS UNAFFECTED')
  for name, (od, gd) in OLD_ENVS.items():
    c = Config(env_name=name)
    e = envs_mod.make_env(name, c, seed=0)
    ok = (e.obs_dim, e.goal_dim) == (od, gd)
    gate('7_%s' % name.replace('point_two_route_swamp_', ''), ok,
         'obs_dim %d goal_dim %d (expected %d/%d)' % (e.obs_dim, e.goal_dim,
                                                      od, gd))

  # ---------------------------------------------------------------------- 8
  print('\n8. THE TWO LAGS  (reported, not a pass/fail)')
  print('  There are two warm-up costs and they are NOT the same size. Saying')
  print('  "the first n_frames steps carry no separation" would be wrong.')
  env.reset()
  spreads = []
  for t in range(8):
    env.step(np.array([0.8, 0.0], np.float32))
    spreads.append(float(np.abs(env.frames - env.frames[0]).max()))
  first_informative = next((i + 1 for i, s in enumerate(spreads) if s > 0), None)
  print('\n  (a) RESET LAG. At t = 0 the stack is seeded with copies of the')
  print('      start position, so a fresh agent momentarily looks frozen. ONE')
  print('      step breaks that, because the newest frame has already moved.')
  print('      frame spread by step: ' + ' '.join('%.3f' % s for s in spreads))
  print('      first step with non-zero spread: %s' % first_informative)
  lags = []
  for s in range(400):
    env.reset()
    env._rng = np.random.default_rng(s)
    hit = None
    for t in range(40):
      env.step(np.array([1.0, 0.0], np.float32))
      if env.dead and hit is None:
        hit = t
      if hit is not None and float(
          np.abs(env.frames - env.frames[0]).max()) == 0.0:
        lags.append(t - hit)
        break
  lags = np.array(lags)
  print('\n  (b) DEATH LAG -- the cost that actually matters. A newly dead')
  print('      agent still carries its pre-contact frames, so the all-equal')
  print('      signature only appears once the stack has rolled over. For')
  print('      those steps a dead agent and a moving one still look alike.')
  if lags.size:
    print('      measured over %d deaths: median %.1f, max %d steps'
          % (lags.size, float(np.median(lags)), int(lags.max())))
  out['8_lags'] = {
      'reset_spread_by_step': spreads,
      'reset_first_informative_step': first_informative,
      'death_lag_n': int(lags.size),
      'death_lag_median': float(np.median(lags)) if lags.size else None,
      'death_lag_max': int(lags.max()) if lags.size else None,
      'note': 'the reset lag is 1 step; the death lag is the real cost and is '
              'the n_frames-1 steps during which the frozen signature has not '
              'yet formed'}

  out['gates'] = gates
  n_pass = sum(1 for g in gates.values() if g['pass'])
  print('\n%s' % ('=' * 98))
  print('%d/%d gates pass' % (n_pass, len(gates)))
  out['gates_passed'] = n_pass
  out['gates_total'] = len(gates)
  p = os.path.join(args.out_dir, 'env_audit.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('wrote %s' % p)
  return 0 if n_pass == len(gates) else 1


if __name__ == '__main__':
  sys.exit(main())
