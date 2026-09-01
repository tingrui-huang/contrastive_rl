"""Why does the actor take a route its own critic scores lowest? Analysis only.

The critic ranks the lanes right > left > centre by roughly 10 logits (probe_
critic_lane_preference.py). The actor nevertheless spends 39% of its episodes
in the middle. That is only a contradiction if the actor could act on the
critic's lane preference. It cannot, and this probe measures why.

The actor objective is evaluated ONE STEP at a time: it moves the action to
raise f(s, a, g) at the current state. A lane is not an action -- it is the
accumulation of a hundred steps of small lateral drift. So the question is:

  at a single entrance state, how much does f change between the action that
  eventually leads left and the action that eventually leads right?

Two measurements per state:
  Q SPREAD   f(s, a, g) for the teacher's left-mode action, the teacher's
             right-mode action, the actor's own action, and random actions.
             The random actions calibrate what "the critic cares" looks like:
             if f separates good from garbage by a lot but left from right by
             almost nothing, the critic has no lateral opinion to act on.
  STEP GAIN  the actual one-step |dy| those actions produce from that state,
             which says how many steps of committed drift a lane costs.

Usage: python scripts/probe_q_lateral_signal.py [--n-states 120]
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

import mujoco                              # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from leak_probe_clean_ant import set_state  # noqa: E402
from probe_center_route_cause import qpos_qvel_from_obs  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/clean_ant_leak_probe'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
CLEAN = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
         'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
SIDECAR = ('artifacts/rockfall_v2_p30_h800_resetfix/pilot/'
           'antmaze_rockfall_v2_p30_h800_resetfix_pilot_sidecar.npz')
SEED = 20_357
#: the entrance window: the lane is still undecided here.
TMAX = 12
N_RANDOM = 16


def build(cfg, ckpt_path):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  pp, qp = st.policy_params, st.q_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)

  @jax.jit
  def score(o, a):
    """f(s, a_i, g) for a batch of candidate actions at ONE state."""
    q = nets.q_network.apply(qp, o, a)
    if q.ndim == 3:
      q = jnp.min(q, axis=-1)
    return jnp.diag(q)

  return act, score, int(step)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--n-states', type=int, default=120)
  ap.add_argument('--out-dir', default=OUT)
  ap.add_argument('--seed', type=int, default=SEED)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  d = np.load(CLEAN, allow_pickle=True)
  s = np.load(SIDECAR, allow_pickle=True)
  dead = np.asarray(s['dead'], bool)
  ci = np.where(~dead)[0]
  route = np.asarray(s['route'])[ci]
  base_side = np.asarray(s['base_side'])[ci]
  masks = np.asarray(s['rockfall_mask'])[ci]
  sevs = np.asarray(s['severity'])[ci]
  obs, acts = d['obs'], d['act']

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  cfg.rockfall_severity = V2.SEVERITY_V2
  cfg.rockfall_p_active = 0.30
  cfg.rockfall_max_steps = 800
  cfg.rockfall_reset_fix = True
  act, score, step = build(cfg, args.ckpt)
  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=args.seed)
  print(f'ckpt {args.ckpt} @ step {step}', flush=True)

  # the two teacher modes at each early step, gait-phase matched
  mL, mR = {}, {}
  for t in range(TMAX):
    aL = [acts[e, t] for e in range(len(obs))
          if route[e] != 'center' and base_side[e] == 'left']
    aR = [acts[e, t] for e in range(len(obs))
          if route[e] != 'center' and base_side[e] == 'right']
    mL[t] = np.mean(aL, axis=0)
    mR[t] = np.mean(aR, axis=0)

  rows = []
  eps = [e for e in range(len(obs)) if route[e] != 'center']
  rng.shuffle(eps)
  for e in eps[:args.n_states]:
    t = int(rng.integers(1, TMAX))
    o0 = obs[e, t].astype(np.float32)
    cands = {'actor': np.asarray(act(jnp.asarray(o0[None])))[0],
             'teacher_LEFT_mode': mL[t].astype(np.float32),
             'teacher_RIGHT_mode': mR[t].astype(np.float32),
             'midpoint': (0.5 * (mL[t] + mR[t])).astype(np.float32)}
    names = list(cands)
    A = np.stack([cands[k] for k in names] +
                 [rng.uniform(-1, 1, 8).astype(np.float32)
                  for _ in range(N_RANDOM)])
    O = np.repeat(o0[None], len(A), axis=0)
    f = np.asarray(score(jnp.asarray(O), jnp.asarray(A)))
    r = {k: float(f[i]) for i, k in enumerate(names)}
    r['random_mean'] = float(f[len(names):].mean())
    r['random_min'] = float(f[len(names):].min())

    # one-step lateral effect of each candidate from this exact state
    qpos, qvel = qpos_qvel_from_obs(o0)
    goal = o0[29:31].copy()
    dys = {}
    for k in ('teacher_LEFT_mode', 'teacher_RIGHT_mode', 'actor'):
      set_state(env, qpos, qvel, goal, tuple(int(b) for b in masks[e]),
                tuple(str(x) for x in sevs[e]))
      y0 = float(env._env.data.qpos[1])
      env.step(np.clip(cands[k], -1, 1))
      dys[k] = float(env._env.data.qpos[1]) - y0
    r['dy_left'] = dys['teacher_LEFT_mode']
    r['dy_right'] = dys['teacher_RIGHT_mode']
    r['dy_actor'] = dys['actor']
    r['t'] = t
    rows.append(r)
    if len(rows) % 30 == 0:
      print(f'  {len(rows)}/{args.n_states} states', flush=True)

  def col(k):
    return np.array([r[k] for r in rows])

  fL, fR, fA, fM = col('teacher_LEFT_mode'), col('teacher_RIGHT_mode'), \
      col('actor'), col('midpoint')
  frand, frmin = col('random_mean'), col('random_min')
  lat = np.abs(fL - fR)
  good = np.minimum(fL, fR) - frand
  print('\n--- critic score f(s,a,g) at entrance states (n=%d) ---' % len(rows))
  print(f'  teacher LEFT-mode action   {fL.mean():8.3f}')
  print(f'  teacher RIGHT-mode action  {fR.mean():8.3f}')
  print(f'  midpoint action            {fM.mean():8.3f}')
  print(f'  actor action               {fA.mean():8.3f}')
  print(f'  random actions (mean)      {frand.mean():8.3f}')
  print(f'  random actions (worst)     {frmin.mean():8.3f}')
  print(f'\n  LATERAL signal  |f(left) - f(right)|      = {lat.mean():7.3f}'
        f'  (median {np.median(lat):.3f})')
  print(f'  QUALITY signal  min(f(L),f(R)) - f(random) = {good.mean():7.3f}')
  print(f'  BETWEEN-LANE gap measured downstream       =  ~10.3   '
        f'(critic_lane_preference.py)')
  ratio = float(lat.mean() / max(abs(good.mean()), 1e-9))
  print(f'\n  lateral / quality = {ratio:.3f}')
  dyl, dyr = col('dy_left'), col('dy_right')
  print(f'\n--- one-step lateral effect ---')
  print(f'  |dy| from the LEFT-mode action   {np.abs(dyl).mean():.5f}')
  print(f'  |dy| from the RIGHT-mode action  {np.abs(dyr).mean():.5f}')
  print(f'  |dy_left - dy_right| per step    {np.abs(dyl - dyr).mean():.5f}')
  steps_to_lane = 1.0 / max(np.abs(dyl - dyr).mean(), 1e-9)
  print(f'  -> steps of committed drift to separate the lanes by 1.0 in y: '
        f'{steps_to_lane:.0f}')

  out = {'ckpt': args.ckpt, 'step': step, 'n_states': len(rows),
         'mean_f': {'teacher_LEFT_mode': round(float(fL.mean()), 4),
                    'teacher_RIGHT_mode': round(float(fR.mean()), 4),
                    'midpoint': round(float(fM.mean()), 4),
                    'actor': round(float(fA.mean()), 4),
                    'random_mean': round(float(frand.mean()), 4),
                    'random_worst': round(float(frmin.mean()), 4)},
         'lateral_signal_mean': round(float(lat.mean()), 4),
         'lateral_signal_median': round(float(np.median(lat)), 4),
         'quality_signal_mean': round(float(good.mean()), 4),
         'lateral_over_quality': round(ratio, 4),
         'between_lane_gap_downstream': 10.3,
         'one_step_dy': {
             'left_mode': round(float(np.abs(dyl).mean()), 6),
             'right_mode': round(float(np.abs(dyr).mean()), 6),
             'left_minus_right': round(float(np.abs(dyl - dyr).mean()), 6),
             'steps_to_separate_lanes_by_1': round(float(steps_to_lane), 1)},
         'states': rows}
  p = os.path.join(args.out_dir, 'q_lateral_signal.json')
  with open(p, 'w') as f_:
    json.dump(out, f_, indent=2)
  print('\n->', p, flush=True)


if __name__ == '__main__':
  main()
