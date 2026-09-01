"""Does the learner's critic PREFER the middle, or does it just end up there?
Analysis only, no retraining.

The middle route has two possible origins and identical-looking trajectories:

  GENUINE   the centre is the only mask-independent safe lane, and a blind
            learner is supposed to find it (the benchmark's own
            'found-center-shortcut' verdict). Then the critic should SCORE
            centre states above side-lane states at matched progress.
  ARTIFACT  the entrance action is the average of an unobservable left/right
            coin flip, and the average happens to be drivable. Then the critic
            has no particular opinion about the centre -- the behaviour comes
            from the actor being unable to represent two answers, not from the
            value function liking one.

Protocol: take the teacher's own pre-handoff states from the pilot, slice them
by corridor x so progress-to-goal is matched, splice ONE fixed goal into every
state's goal block so the contrastive scores are commensurable, and compare
f(s, pi(s), g) across the three lanes. The critic returns the full [B, B]
logit matrix; the matched pair is the diagonal, twin-min as in the actor loss.

Usage: python scripts/probe_critic_lane_preference.py [--per-cell 60]
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from leak_probe_clean_ant import mannwhitney  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/clean_ant_leak_probe'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
CLEAN = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
         'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
SIDECAR = ('artifacts/rockfall_v2_p30_h800_resetfix/pilot/'
           'antmaze_rockfall_v2_p30_h800_resetfix_pilot_sidecar.npz')
X_SLICES = [(2.4, 2.8), (2.8, 3.2), (3.2, 3.6), (3.6, 4.0), (4.0, 4.4),
            (4.4, 4.8), (4.8, 5.2), (5.2, 5.6)]


def build_critic(cfg, ckpt_path):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  pp, qp = st.policy_params, st.q_params

  @jax.jit
  def score(o):
    """f(s, pi(s), g) for each row of o, twin-min, as the actor sees it."""
    a = jnp.tanh(nets.policy_network.apply(pp, o).loc)
    q = nets.q_network.apply(qp, o, a)
    if q.ndim == 3:
      q = jnp.min(q, axis=-1)
    return jnp.diag(q)

  return score, int(step)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--per-cell', type=int, default=60,
                  help='states per (lane, x-slice) cell')
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  d = np.load(CLEAN, allow_pickle=True)
  s = np.load(SIDECAR, allow_pickle=True)
  dead = np.asarray(s['dead'], bool)
  ci = np.where(~dead)[0]
  route = np.asarray(s['route'])[ci]
  X = np.asarray(s['step_torso_x'])[ci]
  H = np.asarray(s['step_handoff'])[ci]
  obs = d['obs']

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  score, step = build_critic(cfg, args.ckpt)
  print(f'ckpt {args.ckpt} @ step {step}', flush=True)

  #: one fixed goal for every state, so the contrastive scores are on the
  #: same scale; the modal d4rl eval goal of the pilot.
  goals = np.stack([obs[e, 0, 29:58] for e in range(len(obs))])
  g_fixed = np.median(goals, axis=0).astype(np.float32)
  print('fixed goal xy', np.round(g_fixed[:2], 3),
        '| goal xy spread across episodes',
        np.round(goals[:, :2].std(axis=0), 4), flush=True)

  cells = {}
  for lane in ('left', 'center', 'right'):
    for si, (a, b) in enumerate(X_SLICES):
      pool = []
      for e in range(len(obs)):
        if route[e] != lane:
          continue
        pre = (H[e] == 0) & np.isfinite(X[e]) & (X[e] >= a) & (X[e] < b)
        idx = np.where(pre)[0]
        for t in idx[::4]:
          if t < obs.shape[1]:
            pool.append(obs[e, t])
        if len(pool) >= args.per_cell:
          break
      if len(pool) < 8:
        continue
      o = np.stack(pool[:args.per_cell]).astype(np.float32)
      o[:, 29:58] = g_fixed
      cells[(lane, si)] = np.asarray(score(jnp.asarray(o)))

  print(f'\n{"x slice":>12s} ' + ''.join(f'{l:>12s}' for l in
                                         ('left', 'center', 'right'))
        + '    centre - best side')
  rows = []
  for si, (a, b) in enumerate(X_SLICES):
    vals = {l: cells.get((l, si)) for l in ('left', 'center', 'right')}
    if vals['center'] is None:
      continue
    m = {l: (float(np.mean(v)) if v is not None else None)
         for l, v in vals.items()}
    sides = [m[l] for l in ('left', 'right') if m[l] is not None]
    gap = (m['center'] - max(sides)) if sides else None
    print(f'  [{a:.1f},{b:.1f})  ' +
          ''.join(f'{(m[l] if m[l] is not None else float("nan")):12.3f}'
                  for l in ('left', 'center', 'right')) +
          f'{gap:12.3f}' if gap is not None else '')
    rows.append({'x_slice': [a, b],
                 'mean': {l: (round(m[l], 4) if m[l] is not None else None)
                          for l in m},
                 'n': {l: (len(v) if v is not None else 0)
                       for l, v in vals.items()},
                 'center_minus_best_side': (round(gap, 4) if gap is not None
                                            else None)})

  allc = np.concatenate([v for (l, _), v in cells.items() if l == 'center'])
  alls = np.concatenate([v for (l, _), v in cells.items() if l != 'center'])
  mw = mannwhitney(allc, alls)
  gaps = [r['center_minus_best_side'] for r in rows
          if r['center_minus_best_side'] is not None]
  verdict = ('critic PREFERS the centre' if gaps and np.mean(gaps) > 0
             else 'critic does NOT prefer the centre')
  print(f'\npooled: centre {allc.mean():.3f} (n={len(allc)})  '
        f'side {alls.mean():.3f} (n={len(alls)})  '
        f'diff {allc.mean() - alls.mean():+.3f}')
  print('Mann-Whitney centre vs side:', mw)
  print('mean (centre - best side) across x slices:',
        round(float(np.mean(gaps)), 4) if gaps else None, '->', verdict)

  out = {'ckpt': args.ckpt, 'step': step, 'per_cell': args.per_cell,
         'fixed_goal_xy': [round(float(v), 4) for v in g_fixed[:2]],
         'by_x_slice': rows,
         'pooled': {'center_mean': round(float(allc.mean()), 4),
                    'center_n': int(len(allc)),
                    'side_mean': round(float(alls.mean()), 4),
                    'side_n': int(len(alls)),
                    'diff': round(float(allc.mean() - alls.mean()), 4),
                    'mannwhitney': mw},
         'mean_center_minus_best_side': (round(float(np.mean(gaps)), 4)
                                         if gaps else None),
         'verdict': verdict}
  p = os.path.join(args.out_dir, 'critic_lane_preference.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('->', p, flush=True)


if __name__ == '__main__':
  main()
