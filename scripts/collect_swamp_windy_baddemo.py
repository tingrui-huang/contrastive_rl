"""Collect the BAD DEMONSTRATOR dataset for the windy-LETHAL swamp env.

WHY THIS EXISTS. In the main dataset (datasets/swamp_windy_teacher_s0.npz) the
branch we need to reason about has no competent support:

    teacher_mode 2/3 (gate-aware)   P(enter | U active) =   0 / 501
    teacher_mode 0   (random)       P(enter | U active) = 118 / 466

i.e. the ONLY data in which somebody walks forward while the gate is active
comes from a uniform-random policy, which is equally blind when the gate is
clear (0.228) and behaves like noise for the rest of the episode. Positivity is
strictly violated for the gate-aware behavior policy. A failure bank built from
that dataset is therefore 79% random-walk deaths, not demonstrations.

The bad demonstrator is the missing arm: BFS-competent everywhere, heading for
the shortcut, with the ONE dodge step removed. It reads no bits and never
waits. Concretely, in make_windy_teacher the loop

    for cand in (a, zeros, -a):        # forward, wait, backward
        if not deadly(landing_cell(s, cand)): return cand

collapses to ``return a``.

WIND IS LEFT NATURAL BY DEFAULT (--gate_active_prob unset). That is deliberate
and is the point of the dataset: with p=0.10 per cell per step the SAME
demonstrator sometimes dies in the corridor and sometimes walks straight
through it, which yields matched pairs -- same state, same action, different U,
different outcome. Those pairs are what a worst-case bound is identified from.
Forcing the gate always-on (--gate_active_prob 1.0) reproduces the literal
"swamp is 100% active and the expert walks in anyway" reading, but every
episode then dies in swamp cell 0 and the matched survivals are gone.

These trajectories are POSITIVES, not bank entries. An episode that dies at
(4.2, 3.5) truthfully demonstrates that (4.2, 3.5) IS reachable from (2.5, 3.5)
under a=forward; that is correct dynamics and belongs on the positive side.
Adding the same points as psi-side negative goals is what the 27-run alpha
sweep already falsified -- 94.8% of the states surviving episodes occupy in the
corridor sit within 0.10 of some death state, and the learner observation is
[x, y] only (crl.envs: "U is NOT exposed"), so the two labels land on identical
inputs and cancel into a constant.

Output layout is byte-compatible with scripts/collect_swamp_windy.py (same
keys, same widths) so the two can be concatenated or loaded side by side.
teacher_mode is a NEW code 4 = bad_demo, so every downstream audit can select
or exclude these episodes.

Run:
  python -m scripts.collect_swamp_windy_baddemo --episodes 600 --seed 0 \
      --out datasets/swamp_windy_baddemo_s0.npz
"""
import argparse
import hashlib
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config
from crl.report_maze import make_oracle

ENV = 'point_two_route_swamp_windy_v0'
# Codes 0-3 are collect_swamp_windy.MODE; 4 is new and belongs to this file.
MODE = {'random': 0, 'forced_safe': 1, 'immediate_shortcut': 2,
        'wait_shortcut': 3, 'bad_demo': 4}
ROUTE = {'random': 0, 'shortcut': 1, 'safe_detour': 2, 'other': 3}
HOLDING_CELL = (2, 3)


def _sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(chunk), b''):
      h.update(block)
  return h.hexdigest()


def make_bad_demonstrator(env):
  """BFS-competent shortcut policy that never consults the swamp bits.

  Identical to make_windy_teacher's non-force_safe branch with the dodge
  candidate loop deleted. It is a DEMONSTRATOR: purposeful, repeatable, and
  wrong for exactly one reason.
  """
  base_oracle = make_oracle(env._walls)

  def policy(s, g, memo):
    if '_m' not in memo:
      memo['_m'] = {}
      memo['teacher_mode'] = 'bad_demo'
      memo['wait_count'] = 0
    return np.asarray(base_oracle(s, g, memo['_m']), np.float32)
  return policy


def _audit(obs, bits, died, act):
  """P(enter | U) at the holding cell + where the deaths land."""
  xy = obs[:, :, :2]
  cell = np.clip(np.floor(xy).astype(int), 0, [8, 4])
  at_hold = ((cell[:, :, 0] == HOLDING_CELL[0])
             & (cell[:, :, 1] == HOLDING_CELL[1]))
  na = nc = fa = fc = 0
  for e in range(obs.shape[0]):
    for t in np.where(at_hold[e, :-1])[0]:
      # bits[e, t, 0] governs step t and is the bit for swamp cell (3,3), the
      # cell a forward move from the holding cell lands in.
      entered = bool(cell[e, t + 1, 0] == 3 and cell[e, t + 1, 1] == 3)
      if bits[e, t, 0]:
        na += 1
        fa += entered
      else:
        nc += 1
        fc += entered
  # death cell histogram
  swamp = ((3, 3), (4, 3), (5, 3))
  hist = {f'S{k}_{c}': 0 for k, c in enumerate(swamp)}
  for e in np.where(died.astype(bool))[0]:
    for t in range(obs.shape[1] - 1):
      # int(), not the raw numpy scalars: tuple(cell[...]) is a tuple of
      # np.int64, which compares equal to the python tuple but formats as
      # "(np.int64(4), np.int64(3))" and so misses the pre-built key.
      c = (int(cell[e, t + 1, 0]), int(cell[e, t + 1, 1]))
      if c in swamp and bits[e, t, swamp.index(c)]:
        hist[f'S{swamp.index(c)}_{c}'] += 1
        break
  return {
      'holding_cell_visits_gate_active': int(na),
      'holding_cell_visits_gate_clear': int(nc),
      'entered_given_active': int(fa),
      'entered_given_clear': int(fc),
      'P_enter_given_active': (fa / na) if na else None,
      'P_enter_given_clear': (fc / nc) if nc else None,
      'deaths_per_swamp_cell': hist,
  }


def collect(episodes, teacher_noise, seed, out, gate_active_prob=None,
            force=False):
  if os.path.exists(out) and not force:
    raise SystemExit(f'REFUSING to overwrite {out} (use --force).')
  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=seed)
  if gate_active_prob is not None:
    # _resample() reads self.active_prob every step, so this is enough; 1.0
    # gives the literal always-lethal reading.
    env.active_prob = float(gate_active_prob)
  rng = np.random.default_rng(seed)
  demo = make_bad_demonstrator(env)
  L = env.max_episode_steps + 1
  D, A = cfg.obs_dim + cfg.goal_dim, cfg.action_dim

  obs_out = np.zeros((episodes, L, D), np.float32)
  act_out = np.zeros((episodes, L, A), np.float32)
  bits_out = np.zeros((episodes, L, 3), np.uint8)
  route_out = np.zeros((episodes,), np.int64)
  mode_out = np.full((episodes,), MODE['bad_demo'], np.int64)
  fsafe_out = np.zeros((episodes,), np.uint8)
  wait_out = np.zeros((episodes,), np.int64)
  died_out = np.zeros((episodes,), np.uint8)
  succ = []

  for ep in range(episodes):
    env.reset()
    g = env.goal.copy()
    memo = {}
    traj = [env.state.copy()]
    for t in range(env.max_episode_steps):
      obs_out[ep, t] = np.concatenate([env.state, g]).astype(np.float32)
      bits_out[ep, t] = env.swamp_bits.astype(np.uint8)
      a = demo(env.state.copy(), g, memo)
      if teacher_noise > 0 and np.any(a != 0):
        a = np.clip(a + rng.normal(0, teacher_noise, A), -1, 1).astype(np.float32)
      act_out[ep, t] = a
      env.step(a)
      traj.append(env.state.copy())
    obs_out[ep, -1] = np.concatenate([env.state, g]).astype(np.float32)
    bits_out[ep, -1] = env.swamp_bits.astype(np.uint8)
    traj = np.array(traj)
    died_out[ep] = int(env.dead)
    used_safe = bool(np.any(traj[:, 1] < 2.0))
    crossed = bool(np.any(traj[:, 0] > 6.0)) and not used_safe
    route_out[ep] = (ROUTE['shortcut'] if crossed else
                     ROUTE['safe_detour'] if used_safe else ROUTE['other'])
    succ.append(float(np.min(np.linalg.norm(traj - g, axis=1)) < 0.5))
    if (ep + 1) % 200 == 0:
      print(f'  {ep + 1}/{episodes} (reached@0.5 {np.mean(succ):.3f}, '
            f'died {died_out[:ep + 1].mean():.3f})', flush=True)

  succ = np.asarray(succ)
  audit = _audit(obs_out, bits_out, died_out, act_out)
  meta = {
      'env_name': ENV, 'setting': 'windy_lethal_bad_demonstrator',
      'episodes': int(episodes), 'seed': int(seed),
      'behavior_policy': 'BFS shortcut oracle with the dodge step REMOVED; '
                         'never reads swamp_bits, never waits; '
                         f'teacher_noise={teacher_noise}',
      'role': 'POSITIVE-side support for the (forward | U active) branch that '
              'the gate-aware teacher visits 0/501 times. NOT a psi-side '
              'negative bank -- see module docstring.',
      'gate_active_prob': float(env.active_prob),
      'gate_active_prob_overridden': gate_active_prob is not None,
      'teacher_noise': float(teacher_noise),
      'resample': 'every step, everywhere (wind); active swamp = terminal',
      'obs_dim': int(cfg.obs_dim), 'goal_dim': int(cfg.goal_dim),
      'action_dim': int(A), 'max_episode_steps': int(env.max_episode_steps),
      'episode_len_rows_L': int(L), 'obs_width_D': int(D),
      'n_transitions': int(episodes * (L - 1)),
      'teacher_mode_code': MODE, 'route_label_code': ROUTE,
      'teacher_mode_frequencies': {'bad_demo': int(episodes)},
      'died_rate_overall': float(died_out.mean()),
      'reached_0p5_rate': float(succ.mean()),
      'audit': audit,
      'audit_fields': ['swamp_bits', 'route_label', 'teacher_mode',
                       'force_safe', 'wait_count', 'entered_active_swamp'],
      'note': 'layout matches scripts/collect_swamp_windy.py exactly; '
              'teacher_mode == 4 marks every episode as bad_demo',
  }
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  np.savez(out, obs=obs_out, act=act_out, swamp_bits=bits_out,
           route_label=route_out, teacher_mode=mode_out, force_safe=fsafe_out,
           wait_count=wait_out, entered_active_swamp=died_out,
           meta=np.array(json.dumps(meta)))
  digest = _sha256(out)
  json.dump(dict(path=os.path.abspath(out), sha256=digest,
                 size_bytes=int(os.path.getsize(out)),
                 obs_shape=list(obs_out.shape), act_shape=list(act_out.shape),
                 frozen=True, meta=meta),
            open(out + '.manifest.json', 'w'), indent=2)
  try:
    os.chmod(out, 0o444)
  except OSError:
    pass

  print(f'\nBAD DEMONSTRATOR -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')
  print(f'sha256 = {digest}')
  print(f'\ndied {died_out.mean():.3f}   reached@0.5 {succ.mean():.3f}   '
        f'gate p={env.active_prob}')
  a = audit
  print(f'holding cell, U ACTIVE : entered {a["entered_given_active"]}/'
        f'{a["holding_cell_visits_gate_active"]} '
        f'= {a["P_enter_given_active"]}')
  print(f'holding cell, U clear  : entered {a["entered_given_clear"]}/'
        f'{a["holding_cell_visits_gate_clear"]} '
        f'= {a["P_enter_given_clear"]}')
  print('deaths per swamp cell  : '
        + '  '.join(f'{k}={v}' for k, v in a['deaths_per_swamp_cell'].items()))
  if not a['P_enter_given_active']:
    print('\nWARNING: P(enter | active) is still 0 -- the dodge was not '
          'actually removed, or the demonstrator never reaches the corridor.')


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument('--episodes', type=int, default=600)
  p.add_argument('--teacher_noise', type=float, default=0.15)
  p.add_argument('--gate_active_prob', type=float, default=None,
                 help='override the env per-cell activation probability. '
                      'Leave unset (0.10, natural wind) to get matched '
                      'survive/die pairs; 1.0 = always lethal, every episode '
                      'dies in swamp cell 0 and the matched pairs are lost.')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', required=True)
  p.add_argument('--force', action='store_true')
  a = p.parse_args()
  if a.force and os.path.exists(a.out):
    try:
      os.chmod(a.out, 0o644)
    except OSError:
      pass
  collect(a.episodes, a.teacher_noise, a.seed, a.out,
          gate_active_prob=a.gate_active_prob, force=a.force)


if __name__ == '__main__':
  main()
