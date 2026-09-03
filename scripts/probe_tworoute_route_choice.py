"""Does the coin-free learner CHOOSE a route, or average the two modes?

With the heading coin retired, `shortcut_rate` alone cannot answer this: the
actor is deterministic (tanh(loc)) and the only variation left across
episodes is the +-0.1 qpos reset jitter, so a rate near 0.5 can equally mean
"every episode runs the same averaged behaviour that gets LABELLED shortcut
half the time". Two probes settle it.

P1 -- ACTION-SPACE MODE AVERAGING (the decisive one). Replay the dataset's own
early states through the trained actor and score its action against four
candidates: the shortcut mode mean, the detour mode mean, their mass-weighted
mixture, and the 50/50 midpoint. A port of scripts/probe_center_route_cause.py
to the two-route fork.
  VERDICT: mode-averaging CONFIRMED if the mixture is the argmin on >= 8 of
  the first 12 phase-matched steps AND ||pi - mix|| < 0.25 * ||a_sc - a_dt||.
  Prerequisite (checked and reported): the dataset's own mode separation must
  be large enough for the comparison to mean anything. On the coin-carrying
  v1 dataset it was 0.178 against a within-mode spread of ~1.7 -- there the
  probe would be vacuous. On v2 it is 1.415 (2.757 at t=0).

P2 -- THE GEOMETRIC CONSEQUENCE, measured rather than modelled. Deterministic
rollouts from the canonical pose; classify each episode as reaching a route
or JAMMED (torso xy static for 50+ consecutive steps). This separates
"averaged into the wall" from "stalled at the fork" from "actually chose",
and it protects the causal reading: info['route'] latches to 'shortcut' on
band entry, so an ant that drifts into the band while aiming at a wall is
labelled a shortcut-taker.

Sector table from the maze geometry, measured from the start at (0,0):
  detour 0-18.4 deg | WALL 18.4-71.6 deg (59%) | shortcut 71.6-90 deg

Usage:
  python scripts/probe_tworoute_route_choice.py --ckpt <run>/best.pkl [--n 60]
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
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/tworoute_rockfall_v0'
NPZ = ('artifacts/tworoute_rockfall_v0/dataset/'
       'antmaze_tworoute_rockfall_v2.npz')
HORIZON = 400
JAM_EPS = 0.004        #: per-step torso displacement below this = not moving
JAM_RUN = 50           #: consecutive static steps that count as jammed


def build_policy(ckpt_path, seed=1):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg, seed=seed)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  return act, int(step)


def p1_mode_averaging(act, npz_path, n_steps=12):
  """Score the actor against the two dataset modes and their combinations."""
  a = np.load(npz_path, allow_pickle=True)
  s = np.load(npz_path.replace('.npz', '_sidecar.npz'), allow_pickle=True)
  obs, acts = a['obs'], a['act']
  intent = np.asarray(s['route_intent'])
  sc, dt = intent == 'shortcut', intent == 'detour'
  w = float(sc.mean())                       # mass weight of the shortcut mode
  rows = []
  for t in range(n_steps):
    m_sc = acts[sc, t, :].mean(0)
    m_dt = acts[dt, t, :].mean(0)
    mix = w * m_sc + (1.0 - w) * m_dt
    mid = 0.5 * (m_sc + m_dt)
    #: the actor's action on the SAME states the dataset visited at step t
    pi = np.asarray(act(jnp.asarray(obs[:, t, :])))
    pi_bar = pi.mean(0)
    cand = {'shortcut': m_sc, 'detour': m_dt, 'mixture': mix, 'midpoint': mid}
    dists = {k: float(np.linalg.norm(pi_bar - v)) for k, v in cand.items()}
    rows.append({'t': t,
                 'mode_separation': round(float(np.linalg.norm(m_sc - m_dt)), 4),
                 'dist': {k: round(v, 4) for k, v in dists.items()},
                 'argmin': min(dists, key=dists.get)})
  seps = np.array([r['mode_separation'] for r in rows])
  argmins = [r['argmin'] for r in rows]
  n_mix = sum(x == 'mixture' for x in argmins)
  d_mix = np.array([r['dist']['mixture'] for r in rows])
  confirmed = bool(n_mix >= 8 and np.mean(d_mix) < 0.25 * np.mean(seps))
  return {'shortcut_mass_weight': round(w, 4),
          'mean_mode_separation_t_lt_12': round(float(seps.mean()), 4),
          'separation_at_t0': round(float(seps[0]), 4),
          'argmin_counts': {k: argmins.count(k)
                            for k in ('shortcut', 'detour', 'mixture',
                                      'midpoint')},
          'n_steps_argmin_is_mixture': n_mix,
          'mean_dist_to_mixture': round(float(d_mix.mean()), 4),
          'threshold_0p25_x_separation': round(float(0.25 * seps.mean()), 4),
          'mode_averaging_confirmed': confirmed,
          'per_step': rows}


def p2_rollouts(act, n, seed):
  """Deterministic rollouts from the canonical pose; jam vs route."""
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = HORIZON
  env = envs_mod.make_env('offline_ant_umaze_tworoute_rockfall', cfg,
                          seed=seed)
  rows = []
  for k in range(n):
    o = env.reset()
    u = env.privileged_rockfall_active
    xs, ys = [], []
    jam_run, jam_max, prev, info = 0, 0, None, {}
    for t in range(HORIZON):
      o, r, done, info = env.step(np.asarray(act(jnp.asarray(o[None]))[0]))
      x, y = float(o[0]), float(o[1])
      xs.append(x)
      ys.append(y)
      if prev and abs(x - prev[0]) < JAM_EPS and abs(y - prev[1]) < JAM_EPS:
        jam_run += 1
        jam_max = max(jam_max, jam_run)
      else:
        jam_run = 0
      prev = (x, y)
      if done or r > 0:
        break
    xs, ys = np.array(xs), np.array(ys)
    d = np.hypot(xs, ys)
    #: displacement heading once the ant has travelled 3 world units
    i3 = int(np.argmax(d >= 3.0)) if (d >= 3.0).any() else -1
    head = (round(float(np.degrees(np.arctan2(ys[i3], xs[i3]))), 1)
            if i3 >= 0 else None)
    sector = (None if head is None else
              'detour' if head < 18.4 else
              'wall' if head < 71.6 else 'shortcut')
    rows.append({'u': bool(u), 'success': bool(info.get('success')),
                 'failure': bool(info.get('failure')),
                 'route': info.get('route'),
                 'jammed': bool(jam_max >= JAM_RUN),
                 'max_jam_run': int(jam_max),
                 'reached_3u': i3 >= 0,
                 'heading_at_3u_deg': head, 'sector_at_3u': sector,
                 'final_xy': [round(float(xs[-1]), 3), round(float(ys[-1]), 3)],
                 'steps': int(len(xs))})
    if (k + 1) % 20 == 0:
      print(f'  P2 {k + 1}/{n}', flush=True)

  def frac(f):
    return round(float(np.mean([f(r) for r in rows])), 4)

  sect = [r['sector_at_3u'] for r in rows]
  return {'n': n,
          'success': frac(lambda r: r['success']),
          'failure': frac(lambda r: r['failure']),
          'jammed': frac(lambda r: r['jammed']),
          'never_reached_3u': frac(lambda r: not r['reached_3u']),
          'route': {k: round(float(np.mean([r['route'] == k for r in rows])), 4)
                    for k in ('shortcut', 'detour')},
          'route_none': frac(lambda r: r['route'] is None),
          'sector_at_3u': {k: sect.count(k) for k in
                           ('detour', 'wall', 'shortcut', None)},
          'mean_heading_at_3u_deg': (
              round(float(np.mean([r['heading_at_3u_deg'] for r in rows
                                   if r['heading_at_3u_deg'] is not None])), 1)
              if any(r['reached_3u'] for r in rows) else None),
          'mean_max_jam_run': round(float(np.mean(
              [r['max_jam_run'] for r in rows])), 1),
          'episodes': rows}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--n', type=int, default=60)
  ap.add_argument('--seed', type=int, default=777)
  ap.add_argument('--npz', default=NPZ)
  ap.add_argument('--label', default=None)
  args = ap.parse_args()
  act, step = build_policy(args.ckpt)
  label = args.label or os.path.basename(os.path.dirname(args.ckpt))
  print(f'{label}: ckpt {args.ckpt} @ step {step}', flush=True)

  p1 = p1_mode_averaging(act, args.npz)
  print('P1 mode averaging:', json.dumps(
      {k: v for k, v in p1.items() if k != 'per_step'}, indent=2), flush=True)
  p2 = p2_rollouts(act, args.n, args.seed)
  print('P2 rollouts:', json.dumps(
      {k: v for k, v in p2.items() if k != 'episodes'}, indent=2), flush=True)

  os.makedirs(OUT, exist_ok=True)
  path = os.path.join(OUT, f'route_choice_probe_{label}.json')
  with open(path, 'w') as f:
    json.dump({'label': label, 'ckpt': args.ckpt, 'ckpt_step': step,
               'p1_mode_averaging': p1, 'p2_rollouts': p2}, f, indent=2)
  print('->', path, flush=True)


if __name__ == '__main__':
  main()
