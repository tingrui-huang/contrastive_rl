"""AntMaze V6 mainline pilot: frozen-model trajectory diagnostic (no retraining).

Two questions on the existing start / O / CF checkpoints, on the SAME 300
evaluation episodes (env seed 3909, mode policy):

  1. Where exactly do the CF timeouts stop?  ``rollout`` replays every policy
     with full per-step capture (31-column observation = 29-dim Ant state
     incl. xy / torso height / quaternion / joints / velocities, plus the goal
     xy; the executed torque; the env's route, hazard-entry and mouth flags),
     then ``timeouts`` classifies each timeout by its end position, route
     progress, last-100-step motion, wall clearance and posture (stopped /
     oscillating / wall / fallen / wrong direction / slow), separately for
     "on the detour", "no route label yet" and "on the shortcut".
  2. Where does the route-choice signal act?  ``fork`` (a) scores the seven
     policies' actual reset-state torques with each arm's own critic (and the
     start agent's), (b) swaps ONLY the first action at the identical reset
     state and hidden draw under a fixed continuation policy and reads the
     realised route, (c) locates the step where the O and CF paths of the same
     episode actually separate and repeats the single-action swap there, and
     (d) reads what the CF branches (the critic's positive futures) said at the
     t = 0 anchors, plus the CF actor's own samples / log-probs at those states.
  ``cont`` answers the third case (entered the detour, could not finish): from
  the CF policy's own pre-stall state the continuation is handed to CF itself
  (control), the start d05 policy, the O policy and the teacher's blind
  position driver.

Everything is deterministic: episode k of the evaluation is reproduced from
its recorded hidden draw (latents, clocks, rock jitter) and initial pose, so
every variant of an episode shares its environment exactly.

  python scripts/diag_v6_pilot_trajectories.py rollout --workers 7
  python scripts/diag_v6_pilot_trajectories.py timeouts
  python scripts/diag_v6_pilot_trajectories.py fork --workers 18
  python scripts/diag_v6_pilot_trajectories.py cont --workers 18
  python scripts/diag_v6_pilot_trajectories.py entries      # hazard / leg entry times from the xy records
  python scripts/diag_v6_pilot_trajectories.py objective    # q-term vs BC term of the actor loss at start-region rows
  python scripts/diag_v6_pilot_trajectories.py report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
  os.environ.setdefault(_v, '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import exp_v6_mainline_pilot as MP  # noqa: E402  (pins the pilot's V6_* variables, eval seed 3909)

OUT = MP.OUT / 'diag_traj'
EVAL_SEED, N_EP, HORIZON = MP.EVAL['seed'], MP.EVAL['n'], MP.HORIZON
SEEDS = MP.SEEDS
STATE_DIM, OBS_W, ACTION_DIM = MP.STATE_DIM, MP.OBS_W, MP.ACTION_DIM
NQ, NV = 15, 14
DIAG_SEED = 141_000_000
N_SAMPLES = 64
GAP_DIVERGE = 0.5          # xy distance at which two paths of one episode count as separated
STALL_BACK = 20            # steps before the stall onset at which the continuation test starts
WALL_CLEAR = 0.6           # clearance below which a stalled ant counts as against a wall
TORSO_FALLEN = 0.30        # torso height below which the ant counts as fallen


def policies():
  P = {'start': MP.START_CKPT}
  for arm in MP.ARMS:
    for s in SEEDS:
      P[f'{arm}_s{s}'] = MP.run_dir(arm, s) / 'final.pkl'
  return P


# --------------------------------------------------------------- geometry
def route_arc(x, y):
  """Progress coordinate along the maze corridors: shortcut s = x (0..24);
  detour s = y on the west column (0..8), 8 + x on the top corridor (8..32),
  32 + (8 - y) on the east column (32..40).  Returns (route, s, clearance)
  where clearance is the perpendicular distance to the corridor walls."""
  cands = []
  if -2.5 <= y <= 2.5 and -2.5 <= x <= 26.5:            # shortcut corridor (row y = 0)
    cands.append(('shortcut', float(np.clip(x, 0, 24)), 2.0 - abs(y)))
  if -2.5 <= x <= 2.5 and -2.5 <= y <= 10.5:            # west column
    cands.append(('detour', float(np.clip(y, 0, 8)), 2.0 - abs(x)))
  if 5.5 <= y <= 10.5 and -2.5 <= x <= 26.5:            # top corridor (row y = 8)
    cands.append(('detour', 8.0 + float(np.clip(x, 0, 24)), 2.0 - abs(y - 8)))
  if 21.5 <= x <= 26.5 and -2.5 <= y <= 10.5:           # east column
    cands.append(('detour', 32.0 + float(np.clip(8 - y, 0, 8)), 2.0 - abs(x - 24)))
  if not cands:
    return ('off', float('nan'), float('nan'))
  best = max(cands, key=lambda c: c[2])
  return best


def region(x, y):
  if x < 2 and y < 2:
    return 'start'
  if x < 2 and y >= 2:
    return 'west_column' if y < 6 else 'top_west_corner'
  if y >= 6 and 2 <= x < 22:
    return 'top_corridor'
  if x >= 22 and y >= 2:
    return 'east_column' if y > 2 else 'goal_area'
  if x >= 22:
    return 'goal_area'
  return 'shortcut_corridor'


def up_z(quat):
  """z-component of the body's up vector from the wxyz quaternion (1 = upright, -1 = flipped)."""
  w, x, y, z = quat
  return 1.0 - 2.0 * (x * x + y * y)


# ---------------------------------------------------------------- rollout
def _eval_args(ckpt):
  import eval_rockfall_clock_v6_baseline as EV
  return EV.parse_args(['--ckpt', str(ckpt), '--n', str(N_EP), '--seed', str(EVAL_SEED), '--policy', 'mean',
                        '--action-seed', '9909', '--p-active-1', '0.5', '--p-active-2', '0.5'])


def _rollout_worker(args):
  """Replay the evaluation of one policy exactly (same env construction, same
  reset order, tanh(loc)) with full capture."""
  name, ckpt, n_ep = args
  import jax.numpy as jnp
  import eval_rockfall_clock_v6_baseline as EV
  a = _eval_args(ckpt)
  act, _, _ = EV.build_mean_policy(str(ckpt), a)
  _, env = EV._configure_env(a, seed=a.seed)
  obs_rows, act_rows, lengths, eps = [], [], [], []
  for k in range(n_ep):
    o = env.reset()
    hid = {'u1': bool(env.privileged_rockfall_active_1), 'u2': bool(env.privileged_rockfall_active_2),
           't0_1': int(env.privileged_sampled_start(1)), 't0_2': int(env.privileged_sampled_start(2)),
           'jitter': np.stack([np.asarray(env._drop_jitter[1]), np.asarray(env._drop_jitter[2])]).astype(np.float64)}
    rows_o, rows_a = [o[:OBS_W].astype(np.float32)], []
    info, reward, done = {}, 0.0, False
    for t in range(HORIZON):
      ac = np.asarray(act(jnp.asarray(o[None], jnp.float32))[0], np.float32)
      o, reward, done, info = env.step(ac)
      rows_a.append(ac); rows_o.append(o[:OBS_W].astype(np.float32))
      if done or reward > 0:
        break
    rows_a.append(np.zeros(ACTION_DIM, np.float32))
    success = bool(info.get('success', reward > 0)); failure = bool(info.get('failure', False))
    eps.append({'episode': k, **{x: hid[x] for x in ('u1', 'u2', 't0_1', 't0_2')}, 'steps': len(rows_a) - 1,
                'success': success, 'failure': failure, 'timeout': (not success and not failure),
                'route': info.get('route'), 'failure_zone': info.get('failure_zone'),
                'entered_hazard_1': bool(info.get('entered_hazard_1')), 'entered_hazard_2': bool(info.get('entered_hazard_2')),
                'mouth_step_1': info.get('mouth_step_1'), 'mouth_step_2': info.get('mouth_step_2'),
                'jitter': hid['jitter']})
    obs_rows.append(np.stack(rows_o)); act_rows.append(np.stack(rows_a)); lengths.append(len(rows_o))
  return name, obs_rows, act_rows, lengths, eps


def save_traj(name, obs_rows, act_rows, lengths, eps):
  lengths = np.asarray(lengths, np.int64)
  offset = np.concatenate([[0], np.cumsum(lengths)[:-1]])
  keys = ('episode', 'u1', 'u2', 't0_1', 't0_2', 'steps', 'success', 'failure', 'timeout', 'entered_hazard_1', 'entered_hazard_2')
  np.savez_compressed(OUT / f'traj_{name}.npz', obs_rows=np.concatenate(obs_rows), act_rows=np.concatenate(act_rows), offset=offset, length=lengths,
                      route=np.array([str(e['route']) for e in eps]), failure_zone=np.array([-1 if e['failure_zone'] is None else int(e['failure_zone']) for e in eps]),
                      mouth_step_1=np.array([-1 if e['mouth_step_1'] is None else int(e['mouth_step_1']) for e in eps]),
                      mouth_step_2=np.array([-1 if e['mouth_step_2'] is None else int(e['mouth_step_2']) for e in eps]),
                      jitter=np.stack([e['jitter'] for e in eps]), **{k: np.array([e[k] for e in eps]) for k in keys})


class Traj:
  def __init__(self, name):
    with np.load(OUT / f'traj_{name}.npz', allow_pickle=False) as d:
      self.d = {k: d[k] for k in d.files}
    self.name = name

  def obs(self, k):
    o, n = int(self.d['offset'][k]), int(self.d['length'][k])
    return self.d['obs_rows'][o:o + n]

  def act(self, k):
    o, n = int(self.d['offset'][k]), int(self.d['length'][k])
    return self.d['act_rows'][o:o + n]

  def ep(self, k):
    return {key: (self.d[key][k].item() if self.d[key].ndim == 1 else self.d[key][k]) for key in self.d if key not in ('obs_rows', 'act_rows', 'offset')}

  def hidden(self, k):
    return {'u1': bool(self.d['u1'][k]), 'u2': bool(self.d['u2'][k]), 't0_1': int(self.d['t0_1'][k]), 't0_2': int(self.d['t0_2'][k]), 'jitter': self.d['jitter'][k]}


def mode_rollout(args):
  OUT.mkdir(parents=True, exist_ok=True)
  from multiprocessing import get_context
  P = policies()
  n_ep = N_EP if args.limit is None else min(args.limit, N_EP)
  todo = [(n, p, n_ep) for n, p in P.items() if not (OUT / f'traj_{n}.npz').exists() or args.force]
  if not todo:
    print('all trajectories exist', flush=True); return
  t0 = time.time()
  if args.workers <= 1:
    res = [_rollout_worker(a) for a in todo]
  else:
    with get_context('spawn').Pool(min(args.workers, len(todo))) as pool:
      res = pool.map(_rollout_worker, todo)
  for name, obs_rows, act_rows, lengths, eps in res:
    save_traj(name, obs_rows, act_rows, lengths, eps)
  # consistency with the evaluation JSONs and identical hidden draws across policies
  chk = {}
  ref = None
  for name in P:
    T = Traj(name)
    ev = MP.read_json(MP.eval_paths()[name if name == 'start' else f'{name[:-3]}/seed_{name[-1]}'])['episodes']
    same = sum(int(bool(T.d['success'][k]) == bool(ev[k]['success']) and bool(T.d['failure'][k]) == bool(ev[k]['failure'])
                   and str(T.d['route'][k]) == str(ev[k]['route']) and int(T.d['steps'][k]) == int(ev[k]['steps'])) for k in range(len(T.d['steps'])))
    hid = np.concatenate([T.d['u1'], T.d['u2'], T.d['t0_1'], T.d['t0_2'], T.d['jitter'].ravel()])
    if ref is None:
      ref = hid
    chk[name] = {'episodes_matching_eval_json (outcome, route, steps)': f'{same}/{len(T.d["steps"])}', 'hidden_draws_identical_to_first_policy': bool(np.array_equal(hid, ref)),
                 'success': float(T.d['success'].mean()), 'timeout': float(T.d['timeout'].mean())}
  MP.write_json(OUT / 'rollout_check.json', {'wall_seconds': time.time() - t0, 'checks': chk})
  print(json.dumps(chk, indent=1), flush=True)


# --------------------------------------------------------------- timeouts
def classify_timeout(obs, act):
  """Per-episode end-state classification for a timeout trajectory."""
  xy = obs[:, :2]
  x, y = float(xy[-1, 0]), float(xy[-1, 1])
  route, s_end, clear = route_arc(x, y)
  arcs = np.array([route_arc(float(p[0]), float(p[1]))[1] for p in xy])
  routes = np.array([route_arc(float(p[0]), float(p[1]))[0] for p in xy])
  det = np.where(routes == 'detour', arcs, np.nan); sc = np.where(routes == 'shortcut', arcs, np.nan)
  prog = det if route == 'detour' else sc
  prog_max = float(np.nanmax(prog)) if np.isfinite(prog).any() else float('nan')
  t_max = int(np.nanargmax(np.where(np.isfinite(prog), prog, -np.inf))) if np.isfinite(prog).any() else -1
  # stall onset: the last step after which progress never again exceeds its running max by 0.5
  stall = t_max
  last100 = xy[-101:]
  d100 = float(np.linalg.norm(last100[-1] - last100[0])); L100 = float(np.linalg.norm(np.diff(last100, axis=0), axis=1).sum())
  prog100 = (float(np.nanmax(prog[-101:]) - np.nanmin(prog[-101:])) if np.isfinite(prog[-101:]).any() else float('nan'))
  going = (float(prog[-1] - np.nanmean(prog[-101:-51])) if np.isfinite(prog[-101:-51]).any() and np.isfinite(prog[-1]) else float('nan'))
  torso_z = float(obs[-1, 2]); upz = float(up_z(obs[-1, 3:7]))
  mean_z_last = float(obs[-101:, 2].mean())
  sat = float((np.abs(act[-101:-1]) > 0.99).mean())
  if upz < 0.0 or mean_z_last < TORSO_FALLEN:
    cls = 'fallen'
  elif d100 < 0.5 and L100 < 3.0:
    cls = 'stopped'
  elif d100 < 1.0 and clear < WALL_CLEAR:
    cls = 'wall_stuck'
  elif d100 < 1.0:
    cls = 'oscillating'
  elif np.isfinite(going) and going < -0.5:
    cls = 'wrong_direction'
  else:
    cls = 'slow_but_moving'
  first_north = int(np.argmax(xy[:, 1] >= 2.0)) if (xy[:, 1] >= 2.0).any() else -1
  first_east = int(np.argmax(xy[:, 0] >= 2.0)) if (xy[:, 0] >= 2.0).any() else -1
  first_top = int(np.argmax(xy[:, 1] >= 6.0)) if (xy[:, 1] >= 6.0).any() else -1
  first_eastcol = int(np.argmax((xy[:, 0] >= 22.0) & (xy[:, 1] >= 6.0))) if ((xy[:, 0] >= 22.0) & (xy[:, 1] >= 6.0)).any() else -1
  return {'end_xy': [round(x, 2), round(y, 2)], 'region': region(x, y), 'route_by_arc': route, 'arc_end': s_end, 'arc_max': prog_max,
          'stall_onset': stall, 'stall_len': int(len(xy) - 1 - stall) if stall >= 0 else None, 'clearance': clear,
          'd100': d100, 'L100': L100, 'prog100': prog100, 'torso_z': torso_z, 'up_z': upz, 'mean_z_last100': mean_z_last, 'saturation_last100': sat,
          'dist_goal': float(np.linalg.norm(xy[-1] - obs[-1, STATE_DIM:OBS_W])), 'max_y': float(xy[:, 1].max()), 'max_x': float(xy[:, 0].max()),
          'first_north': first_north, 'first_east': first_east, 'first_top': first_top, 'first_east_column': first_eastcol, 'class': cls}


def mode_timeouts(args):
  P = policies()
  res = {}
  for name in P:
    T = Traj(name)
    rows = []
    for k in range(len(T.d['steps'])):
      if not bool(T.d['timeout'][k]):
        continue
      c = classify_timeout(T.obs(k), T.act(k))
      c.update({'episode': k, 'route_label': str(T.d['route'][k]), 'u1': bool(T.d['u1'][k]), 'u2': bool(T.d['u2'][k])})
      rows.append(c)
    # summary by route label x class
    by = {}
    for r in rows:
      by.setdefault(r['route_label'], {}).setdefault(r['class'], []).append(r)
    summ = {lab: {cls: {'n': len(v), 'region': {reg: sum(1 for q in v if q['region'] == reg) for reg in set(q['region'] for q in v)},
                        'arc_max_mean': float(np.nanmean([q['arc_max'] for q in v])), 'stall_len_mean': float(np.mean([q['stall_len'] for q in v if q['stall_len'] is not None])) if any(q['stall_len'] is not None for q in v) else None,
                        'torso_z_mean': float(np.mean([q['mean_z_last100'] for q in v])), 'clearance_mean': float(np.nanmean([q['clearance'] for q in v]))}
                  for cls, v in d.items()} for lab, d in by.items()}
    # posture reference from the successful episodes
    zs = [float(T.obs(k)[:, 2].mean()) for k in range(len(T.d['steps'])) if bool(T.d['success'][k])]
    res[name] = {'n_timeouts': len(rows), 'summary': summ, 'success_torso_z_mean': float(np.mean(zs)) if zs else None, 'episodes': rows}
    print(name, 'timeouts', len(rows), {lab: {c: v['n'] for c, v in d.items()} for lab, d in summ.items()}, flush=True)
  MP.write_json(OUT / 'timeouts.json', res)


# ------------------------------------------------------------- branching
_POL_CACHE = {}


def _policy_bundle(name):
  """Mode / distribution / critic of one checkpoint (CPU, cached per process)."""
  if name in _POL_CACHE:
    return _POL_CACHE[name]
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, OUT / '_cfg')
  MP.fill_dims(cfg)
  nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(policies()[name])
  pp, qp = st.policy_params, st.q_params

  @jax.jit
  def _mode(o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)

  @jax.jit
  def _dist(o):
    p = nets.policy_network.apply(pp, o)
    return p.loc, p.scale

  @jax.jit
  def _logp(o, a):
    return nets.log_prob(nets.policy_network.apply(pp, o), a)

  @jax.jit
  def _f(o, a):
    phi, psi = nets.representation_network.apply(qp, o, a)
    return jnp.min(jnp.sum(phi * psi, axis=1), axis=1)
  def _q_term(params, o, key):
    # the actor loss's critic term with alpha = 0: -f(s, a_sampled, g) on the batch diagonal (crl.losses.actor_loss)
    d = nets.policy_network.apply(params, o)
    a = nets.sample(d, key)
    phi, psi = nets.representation_network.apply(qp, o, a)
    return -jnp.mean(jnp.min(jnp.sum(phi * psi, axis=1), axis=1))

  def _bc_nll(params, o, a_logged):
    return -jnp.mean(nets.log_prob(nets.policy_network.apply(params, o), a_logged))
  _gq, _gb = jax.jit(jax.grad(_q_term)), jax.jit(jax.grad(_bc_nll))

  def _grad_norms(O, A, seed):
    import optax
    o = jnp.asarray(O[:, :OBS_W], jnp.float32)
    gq = _gq(pp, o, jax.random.PRNGKey(int(seed)))
    gb = _gb(pp, o, jnp.asarray(A, jnp.float32))
    return float(optax.global_norm(gq)), float(optax.global_norm(gb))
  b = {'mode': lambda o: np.asarray(_mode(jnp.asarray(o[None, :OBS_W], jnp.float32)))[0],
       'grad_norms': _grad_norms,
       'mode_batch': lambda O: np.asarray(_mode(jnp.asarray(O[:, :OBS_W], jnp.float32))),
       'dist': lambda O: tuple(np.asarray(v) for v in _dist(jnp.asarray(O[:, :OBS_W], jnp.float32))),
       'logp': lambda O, A: np.asarray(_logp(jnp.asarray(O[:, :OBS_W], jnp.float32), jnp.asarray(A, jnp.float32))),
       'f': lambda O, A: np.asarray(_f(jnp.asarray(O[:, :OBS_W], jnp.float32), jnp.asarray(A, jnp.float32)))}
  _POL_CACHE[name] = b
  return b


def restore_episode(env, obs0, hidden):
  """The evaluation episode's exact environment: its latents, clocks, rock
  jitter and initial Ant pose, at time 0."""
  import build_v6_branch_replay as B
  o = B.restore(env, obs0[:STATE_DIM].astype(np.float64), obs0[STATE_DIM:OBS_W].astype(np.float64), 0)
  env._active = {1: bool(hidden['u1']), 2: bool(hidden['u2'])}
  env._t0 = {1: int(hidden['t0_1']), 2: int(hidden['t0_2'])}
  env._drop_jitter = {1: np.array(hidden['jitter'][0], np.float64), 2: np.array(hidden['jitter'][1], np.float64)}
  return o


def run_from(env, teacher, o, t, act_fn, max_steps, capture=False):
  """Closed-loop from obs o at time t: act_fn(o) -> torque, or 'driver' = the
  teacher's blind position rule (diag_v6_first_step_crossover._continue)."""
  import diag_v6_first_step_crossover as DG
  if act_fn == 'driver':
    r = DG._continue(env, teacher, o, t, None)
    xy = r['obs'][:, :2]
    return {'success': r['success'], 'failure': r['failure'], 'steps': r['steps'], 'timeout': (not r['success'] and not r['failure']),
            'route': env._route, 'max_y': float(xy[:, 1].max()), 'final_xy': xy[-1].tolist(), 'obs': (r['obs'] if capture else None)}
  rows = [o[:OBS_W].copy()]
  done, reward, step, info = False, 0.0, 0, {}
  while step < max_steps and not done and not reward > 0:
    a = act_fn(o)
    o, reward, done, info = env.step(np.asarray(a, np.float32))
    rows.append(o[:OBS_W].copy()); step += 1
  xy = np.stack(rows)[:, :2]
  return {'success': bool(reward > 0), 'failure': bool(info.get('failure', False)) or (done and not reward > 0), 'steps': step,
          'timeout': (not reward > 0 and not (bool(info.get('failure', False)) or done)), 'route': env._route,
          'max_y': float(xy[:, 1].max()), 'final_xy': xy[-1].tolist(), 'obs': (np.stack(rows) if capture else None)}


def _branch_worker(args):
  """Jobs: (tag, episode, obs0, hidden, prefix_policy, tau, first_policy, cont_policy)."""
  jobs, worker_seed = args
  import build_v6_branch_replay as B
  env, teacher = B._worker_env(worker_seed)
  out = []
  for (tag, k, obs0, hidden, prefix, tau, first, cont) in jobs:
    o = restore_episode(env, obs0, hidden)
    t, done, reward = 0, False, 0.0
    if prefix is not None and tau > 0:
      pf = _policy_bundle(prefix)['mode']
      while t < tau and not done and not reward > 0:
        o, reward, done, _ = env.step(pf(o)); t += 1
    if done or reward > 0:
      out.append({'tag': tag, 'episode': k, 'prefix_ended': True, 'success': bool(reward > 0), 'failure': bool(done and not reward > 0),
                  'timeout': False, 'steps': t, 'route': env._route, 'max_y': None, 'final_xy': o[:2].tolist(), 'tau': tau})
      continue
    swap_xy = o[:2].tolist()
    if first is not None:
      a1 = _policy_bundle(first)['mode'](o)
      o, reward, done, info = env.step(a1); t += 1
      if done or reward > 0:
        out.append({'tag': tag, 'episode': k, 'prefix_ended': False, 'success': bool(reward > 0), 'failure': bool(done and not reward > 0),
                    'timeout': False, 'steps': t, 'route': env._route, 'max_y': float(o[1]), 'final_xy': o[:2].tolist(), 'tau': tau, 'swap_xy': swap_xy})
        continue
    cf = 'driver' if cont == 'driver' else _policy_bundle(cont)['mode']
    r = run_from(env, teacher, o, t, cf, HORIZON - t)
    r.pop('obs', None)
    out.append({'tag': tag, 'episode': k, 'prefix_ended': False, 'tau': tau, 'swap_xy': swap_xy, **r, 'steps': r['steps'] + t})
  return out


def run_branches(jobs, workers, seed0=DIAG_SEED):
  from multiprocessing import get_context
  n = max(1, workers * 4)
  parts = [jobs[i::n] for i in range(n)]
  parts = [p for p in parts if p]
  args = [(p, seed0 + i) for i, p in enumerate(parts)]
  if workers <= 1:
    res = [_branch_worker(a) for a in args]
  else:
    with get_context('spawn').Pool(workers) as pool:
      res = pool.map(_branch_worker, args)
  return [r for part in res for r in part]


# ------------------------------------------------------------------- fork
def mode_fork(args):
  """(a) critic scores of the actual reset-state torques, (b) first-action
  swaps at t = 0, (c) path divergence and the swap at the divergence step,
  (d) the CF branches' support at the t = 0 anchors and the CF actor's own
  samples at the reset states."""
  P = policies()
  T = {n: Traj(n) for n in P}
  n_have = int(len(T['start'].d['steps']))
  obs0 = np.stack([T['start'].obs(k)[0] for k in range(n_have)])            # identical across policies (checked in rollout)
  for n in P:
    assert np.allclose(np.stack([T[n].obs(k)[0] for k in range(n_have)]), obs0, atol=1e-6), f'{n}: reset states differ'
  eps = range(n_have) if args.limit is None else range(min(args.limit, n_have))
  # (a) actual torques and critic scores
  A = {n: _policy_bundle(n)['mode_batch'](obs0) for n in P}
  scores, logps, samples = {}, {}, {}
  rng = np.random.default_rng(DIAG_SEED + 7)
  for s in SEEDS:
    for arm in ('CF', 'O'):
      b = _policy_bundle(f'{arm}_s{s}')
      for n in P:
        scores[f'{arm}_s{s}|{n}'] = b['f'](obs0, A[n]).tolist()
        logps[f'{arm}_s{s}|{n}'] = b['logp'](obs0, A[n]).tolist()
      loc, scale = b['dist'](obs0)
      eps_ = rng.standard_normal((N_SAMPLES, *loc.shape))
      smp = np.tanh(loc[None] + scale[None] * eps_).astype(np.float32)          # [K, N, 8]
      fs = np.stack([b['f'](obs0, smp[j]) for j in range(N_SAMPLES)])            # [K, N]
      fm = b['f'](obs0, A[f'{arm}_s{s}'])
      samples[f'{arm}_s{s}'] = {'mode_score': fm.tolist(), 'sample_score_mean': fs.mean(0).tolist(), 'sample_score_max': fs.max(0).tolist(),
                                'mode_rank_frac_below': (fs < fm[None]).mean(0).tolist(), 'scale_mean': float(scale.mean())}
  van = _policy_bundle('start')
  for n in P:
    scores[f'van|{n}'] = van['f'](obs0, A[n]).tolist()
  # (b) first-action swaps at t = 0 within a seed: first in {start, O_s, CF_s} x continuation in {start, O_s, CF_s}
  jobs = []
  for s in SEEDS:
    names = ['start', f'O_s{s}', f'CF_s{s}']
    for k in eps:
      for first in names:
        for cont in names:
          jobs.append((f'swap0|s{s}|{first}>{cont}', k, obs0[k], T['start'].hidden(k), None, 0, first, cont))
  # replay-fidelity check: the full episode under one policy from the restored state must reproduce the rollout
  for n in ('start', 'CF_s2', 'O_s2'):
    for k in list(eps)[:40]:
      jobs.append((f'fidelity|{n}', k, obs0[k], T['start'].hidden(k), None, 0, None, n))
  # (c) divergence between O_s and CF_s paths of the same episode; swap at that step
  div = {}
  for s in SEEDS:
    div[s] = []
    for k in eps:
      xo, xc = T[f'O_s{s}'].obs(k)[:, :2], T[f'CF_s{s}'].obs(k)[:, :2]
      L = min(len(xo), len(xc))
      gap = np.linalg.norm(xo[:L] - xc[:L], axis=1)
      idx = np.flatnonzero(gap > GAP_DIVERGE)
      tau = int(idx[0]) if len(idx) else -1
      d = {'episode': k, 'tau': tau, 'xy_at_tau_O': (xo[tau].tolist() if tau >= 0 else None), 'xy_at_tau_CF': (xc[tau].tolist() if tau >= 0 else None),
           'route_O': str(T[f'O_s{s}'].d['route'][k]), 'route_CF': str(T[f'CF_s{s}'].d['route'][k]),
           'gap_at_10': float(gap[10]) if L > 10 else None, 'gap_at_50': float(gap[50]) if L > 50 else None,
           'CF_first_north': (int(np.argmax(xc[:, 1] >= 2.0)) if (xc[:, 1] >= 2.0).any() else -1), 'O_first_east': (int(np.argmax(xo[:, 0] >= 2.0)) if (xo[:, 0] >= 2.0).any() else -1)}
      div[s].append(d)
      if tau > 0:
        # at CF's own state at tau: O's single action then CF continues; and at O's state: CF's action then O continues
        jobs.append((f'swapT|s{s}|CF@tau:O>CF', k, obs0[k], T['start'].hidden(k), f'CF_s{s}', tau, f'O_s{s}', f'CF_s{s}'))
        jobs.append((f'swapT|s{s}|O@tau:CF>O', k, obs0[k], T['start'].hidden(k), f'O_s{s}', tau, f'CF_s{s}', f'O_s{s}'))
        jobs.append((f'swapT|s{s}|CF@tau:CF>CF', k, obs0[k], T['start'].hidden(k), f'CF_s{s}', tau, f'CF_s{s}', f'CF_s{s}'))
  print(f'fork: {len(jobs)} branch rollouts on {args.workers} workers', flush=True)
  t0 = time.time()
  res = run_branches(jobs, args.workers)
  print(f'done in {time.time() - t0:.0f} s', flush=True)
  # (d) the CF branches at the t = 0 anchors
  support = branch_support_t0()
  MP.write_json(OUT / 'fork.json', {'torques': {n: A[n].tolist() for n in P}, 'critic_scores': scores, 'logp_under_policy': logps, 'own_samples': samples,
                                    'divergence': div, 'branch_results': res, 'branch_support_t0': support, 'n_episodes': len(list(eps))})


def branch_support_t0():
  """What the CF critic's positive futures said at the reset-row anchors."""
  import build_v6_branch_replay as B
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  bf = MP.BranchFutures(anchors, MP.OUT / 'branches_cf.npz')
  rows = []
  for k in np.flatnonzero(anchors.t == 0):
    o, a = bf.path(int(k))
    goal = anchors.goal_xy[k]
    rows.append({'anchor': int(k), 'episode': int(anchors.episode[k]), 'outcome': str(bf.outcome[k]), 'len': int(bf.lengths[k]),
                 'max_y': float(o[:, 1].max()), 'around': bool(o[:, 1].max() >= B.DETOUR_Y), 'p_goal': B.goal_frame_probability(o, goal, MP.GAMMA, 0.5),
                 'action_first': a[0].tolist()})
  ar = [r for r in rows if r['around']]; st = [r for r in rows if not r['around']]
  summ = {'n_t0_anchors': len(rows), 'n_around': len(ar), 'around_success': float(np.mean([r['outcome'] == 'success' for r in ar])) if ar else None,
          'straight_success': float(np.mean([r['outcome'] == 'success' for r in st])) if st else None,
          'p_goal_around_mean': float(np.mean([r['p_goal'] for r in ar])) if ar else None, 'p_goal_straight_mean': float(np.mean([r['p_goal'] for r in st])) if st else None,
          'straight_death': float(np.mean([r['outcome'] == 'death' for r in st])) if st else None}
  return {'summary': summ, 'rows': rows}


# ------------------------------------------------------------------- cont
def mode_cont(args):
  """Continuation from the CF policy's own pre-stall state (its recorded
  timeout trajectories): CF itself (control), the start policy, the O policy
  of the same seed, the teacher's blind driver.  Also from the step the CF
  path entered the west column (y >= 2) for the detour timeouts."""
  P = policies()
  tos = MP.read_json(OUT / 'timeouts.json')
  T = {n: Traj(n) for n in P}
  jobs = []
  for s in SEEDS:
    name = f'CF_s{s}'
    for r in tos[name]['episodes']:
      k = r['episode']
      hid, o0 = T[name].hidden(k), T[name].obs(k)[0]
      taus = {}
      if r['stall_onset'] is not None and r['stall_onset'] >= 0:
        taus['prestall'] = max(0, int(r['stall_onset']) - STALL_BACK)
      if r['route_label'] == 'detour' and r['first_north'] >= 0:
        taus['enter_detour'] = int(r['first_north'])
      if r['route_label'] == 'None' or r['route_label'] == 'none':
        taus['t100'] = 100
      for tag, tau in taus.items():
        for cont in (name, 'start', f'O_s{s}', 'driver'):
          jobs.append((f'cont|s{s}|{tag}|{cont}', k, o0, hid, name, tau, None, cont))
  print(f'cont: {len(jobs)} branch rollouts', flush=True)
  t0 = time.time()
  res = run_branches(jobs, args.workers, seed0=DIAG_SEED + 500_000)
  print(f'done in {time.time() - t0:.0f} s', flush=True)
  MP.write_json(OUT / 'cont.json', {'results': res})


# ---------------------------------------------------------------- entries
def mode_entries(args):
  """First-entry steps from the recorded xy: hazard zones 1 / 2 (x in [6.6, 9.4] /
  [14.6, 17.4], |y| < 2), the turn north (y >= 2), the shortcut (x >= 2), the
  top corridor (y >= 6), the east column; deaths relative to the zone's burst."""
  from crl import rockfall_clock_v6 as V6
  P = policies()
  res = {}
  for name in P:
    T = Traj(name)
    rows = []
    for k in range(len(T.d['steps'])):
      xy = T.obs(k)[:, :2]
      first = lambda m: (int(np.argmax(m)) if m.any() else -1)
      z1 = first((xy[:, 0] >= V6.HAZARD_X[1][0]) & (xy[:, 0] <= V6.HAZARD_X[1][1]) & (np.abs(xy[:, 1]) < V6.HAZARD_HALF_Y))
      z2 = first((xy[:, 0] >= V6.HAZARD_X[2][0]) & (xy[:, 0] <= V6.HAZARD_X[2][1]) & (np.abs(xy[:, 1]) < V6.HAZARD_HALF_Y))
      r = {'episode': k, 'route': str(T.d['route'][k]), 'outcome': ('success' if T.d['success'][k] else ('death' if T.d['failure'][k] else 'timeout')),
           'u1': bool(T.d['u1'][k]), 'u2': bool(T.d['u2'][k]), 't0_1': int(T.d['t0_1'][k]), 't0_2': int(T.d['t0_2'][k]), 'steps': int(T.d['steps'][k]),
           'first_zone1': z1, 'first_zone2': z2, 'first_north': first(xy[:, 1] >= 2.0), 'first_east': first(xy[:, 0] >= 2.0),
           'first_top': first(xy[:, 1] >= 6.0), 'first_east_column': first((xy[:, 0] >= 22.0) & (xy[:, 1] >= 6.0)),
           'mouth_step_1': int(T.d['mouth_step_1'][k]), 'mouth_step_2': int(T.d['mouth_step_2'][k]), 'failure_zone': int(T.d['failure_zone'][k])}
      if r['outcome'] == 'death' and r['failure_zone'] in (1, 2):
        t0 = r['t0_%d' % r['failure_zone']]
        r['death_in_burst_window'] = bool(t0 <= r['steps'] <= t0 + V6.ROCKFALL_STEPS)
      rows.append(r)
    n = len(rows)
    med = lambda v: (float(np.median(v)) if len(v) else None)
    summ = {'n': n, 'enter_zone1': sum(r['first_zone1'] >= 0 for r in rows) / n, 'enter_zone2': sum(r['first_zone2'] >= 0 for r in rows) / n,
            'turn_north': sum(r['first_north'] >= 0 for r in rows) / n, 'reach_top': sum(r['first_top'] >= 0 for r in rows) / n,
            'reach_east_column': sum(r['first_east_column'] >= 0 for r in rows) / n,
            'median_first_zone1': med([r['first_zone1'] for r in rows if r['first_zone1'] >= 0]), 'median_first_zone2': med([r['first_zone2'] for r in rows if r['first_zone2'] >= 0]),
            'median_first_north': med([r['first_north'] for r in rows if r['first_north'] >= 0]), 'median_first_top': med([r['first_top'] for r in rows if r['first_top'] >= 0]),
            'median_first_east_column': med([r['first_east_column'] for r in rows if r['first_east_column'] >= 0]),
            'deaths': sum(r['outcome'] == 'death' for r in rows), 'deaths_in_burst_window': sum(bool(r.get('death_in_burst_window')) for r in rows),
            'entered_zone1_while_active': sum(1 for r in rows if r['first_zone1'] >= 0 and r['u1']), 'died_zone1': sum(r['failure_zone'] == 1 for r in rows),
            'entered_zone2_while_active': sum(1 for r in rows if r['first_zone2'] >= 0 and r['u2']), 'died_zone2': sum(r['failure_zone'] == 2 for r in rows),
            'detour_episodes_touching_a_zone': sum(1 for r in rows if r['route'] == 'detour' and (r['first_zone1'] >= 0 or r['first_zone2'] >= 0))}
    res[name] = {'summary': summ, 'episodes': rows}
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}, flush=True)
  MP.write_json(OUT / 'entries.json', res)


# -------------------------------------------------------------- objective
def mode_objective(args):
  """The actor objective's two terms at real start-region dataset rows (x < 2,
  y < 2, t <= 5; a 2,048-row sample) and at the 196 reset-row anchors, for
  every actor with its own critic: f at the mode, at an own sample and at
  the logged (teacher) torque; the BC NLL of the logged torque; the weighted
  terms as they enter the loss ((1 - bc) q-term, bc * NLL); the gradient
  norms of the two weighted terms w.r.t. the policy parameters; and the
  critic's row margin between logged north-moving and east-moving torques."""
  P = policies()
  obs, act, lengths, _ = MP.load_dataset()
  n, L = obs.shape[:2]
  t = np.arange(L)[None, :]
  valid = t < (lengths[:, None] - 1)
  m = valid & (obs[:, :, 0] < 2.0) & (obs[:, :, 1] < 2.0) & (t <= 5)
  e, i = np.nonzero(m)
  rng = np.random.default_rng(DIAG_SEED + 11)
  sel = rng.choice(len(e), size=min(2048, len(e)), replace=False)
  O1, A1 = obs[e[sel], i[sel], :OBS_W].astype(np.float32), act[e[sel], i[sel]].astype(np.float32)
  disp = obs[e[sel], i[sel] + 1, :2] - obs[e[sel], i[sel], :2]
  north = (disp[:, 1] > 0.05) & (disp[:, 1] > np.abs(disp[:, 0])); east = (disp[:, 0] > 0.05) & (disp[:, 0] > np.abs(disp[:, 1]))
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  k0 = np.flatnonzero(anchors.t == 0)
  O0, A0 = anchors.obs31[k0].astype(np.float32), anchors.action[k0].astype(np.float32)
  bc = 0.05
  res = {'start_region_rows': int(len(e)), 'sampled_rows': int(len(sel)), 'north_rows': int(north.sum()), 'east_rows': int(east.sum()), 'reset_rows': int(len(k0)), 'per_actor': {}}
  for name in P:
    b = _policy_bundle(name)
    out = {}
    for label, OO, AA in (('start_region_t<=5', O1, A1), ('reset_rows_t0', O0, A0)):
      mode = b['mode_batch'](OO)
      loc, scale = b['dist'](OO)
      a_s = np.tanh(loc + scale * rng.standard_normal(loc.shape)).astype(np.float32)
      f_mode, f_sample, f_logged = b['f'](OO, mode), b['f'](OO, a_s), b['f'](OO, AA)
      nll = -b['logp'](OO, AA)
      gq, gb = b['grad_norms'](OO, AA, DIAG_SEED + 13)
      d = {'f_mode': float(f_mode.mean()), 'f_own_sample': float(f_sample.mean()), 'f_logged': float(f_logged.mean()),
           'P(f_mode > f_logged)': float((f_mode > f_logged).mean()), 'bc_nll_logged': float(nll.mean()), 'bc_nll_median': float(np.median(nll)),
           'policy_scale_mean': float(scale.mean()), 'mode_minus_logged_torque_L2': float(np.linalg.norm(mode - AA, axis=1).mean()),
           'weighted_q_term': float((1 - bc) * (-f_sample.mean())), 'weighted_bc_term': float(bc * nll.mean()),
           'grad_norm_weighted_q_term': (1 - bc) * gq, 'grad_norm_weighted_bc_term': bc * gb, 'grad_ratio_bc_over_q': (bc * gb) / max(1e-9, (1 - bc) * gq)}
      if label == 'start_region_t<=5':
        d['f_logged_north_minus_east'] = float(f_logged[north].mean() - f_logged[east].mean())
        d['f_logged_north'], d['f_logged_east'] = float(f_logged[north].mean()), float(f_logged[east].mean())
        d['nll_north_minus_east'] = float(nll[north].mean() - nll[east].mean())
      out[label] = d
    res['per_actor'][name] = out
    print(name, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out.items()}), flush=True)
  MP.write_json(OUT / 'objective.json', res)


# ----------------------------------------------------------------- report
def _tab(rows, cols):
  L = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
  for r in rows:
    L.append('| ' + ' | '.join(str(v) for v in r) + ' |')
  return L


def mode_report(args):
  P = policies()
  L = ['# Frozen-model trajectory diagnostic: where the CF timeouts stop, and where the route signal acts', '',
       f'Same 300 evaluation episodes (seed {EVAL_SEED}, mode policy) replayed with full capture; every variant of an episode reuses its '
       'recorded latents, clocks, rock jitter and initial pose.  `rollout_check.json`, `timeouts.json`, `fork.json`, `cont.json`.', '']
  chk = MP.read_json(OUT / 'rollout_check.json')['checks'] if (OUT / 'rollout_check.json').exists() else {}
  if chk:
    L += ['## Replay fidelity', '', *_tab([[n, v['episodes_matching_eval_json (outcome, route, steps)'], v['hidden_draws_identical_to_first_policy'], f"{v['success']:.3f}", f"{v['timeout']:.3f}"] for n, v in chk.items()],
                                        ['policy', 'episodes matching the eval JSON (of 300)', 'hidden draws identical', 'success', 'timeout']), '']
  if (OUT / 'timeouts.json').exists():
    tos = MP.read_json(OUT / 'timeouts.json')
    L += ['## 1. Where the timeouts stop', '', 'Class: fallen (torso down / flipped), stopped (last 100 steps: net < 0.5, path < 3), wall_stuck (net < 1, clearance < 0.6), '
          'oscillating (net < 1 otherwise), wrong_direction (progress falling), slow_but_moving.  arc = progress along the route (shortcut 0-24; detour 0-8 west column, 8-32 top corridor, 32-40 east column).', '']
    rows = []
    for n, v in tos.items():
      for lab, d in v['summary'].items():
        for cls, q in d.items():
          reg = ', '.join(f'{r} {c}' for r, c in sorted(q['region'].items(), key=lambda t: -t[1]))
          rows.append([n, lab, cls, q['n'], f"{q['arc_max_mean']:.1f}", (f"{q['stall_len_mean']:.0f}" if q['stall_len_mean'] is not None else '-'), f"{q['torso_z_mean']:.2f}", f"{q['clearance_mean']:.2f}", reg])
    L += _tab(rows, ['policy', 'route label', 'class', 'n', 'mean max arc', 'mean stall length (steps)', 'torso z (last 100)', 'clearance', 'end regions'])
    zs = {n: v['success_torso_z_mean'] for n, v in tos.items()}
    L += ['', 'Torso height reference (mean over successful episodes): ' + ', '.join(f'{n} {z:.2f}' for n, z in zs.items() if z is not None), '']
    # per-episode detail for the CF timeouts
    for n in [x for x in P if x.startswith('CF')]:
      v = tos[n]
      if not v['episodes']:
        continue
      L += [f'### {n}: every timeout', '', *_tab([[r['episode'], r['route_label'], 'U%d%d' % (r['u1'], r['u2']), r['class'], r['region'], f"{r['end_xy'][0]:.1f},{r['end_xy'][1]:.1f}",
                                                    (f"{r['arc_max']:.1f}" if r['arc_max'] == r['arc_max'] else '-'), r['stall_onset'], r['stall_len'], f"{r['d100']:.2f}", f"{r['L100']:.1f}", f"{r['clearance']:.2f}" if r['clearance'] == r['clearance'] else '-', f"{r['mean_z_last100']:.2f}", f"{r['up_z']:.2f}", r['first_north'], r['first_top'], r['first_east_column']]
                                                   for r in v['episodes']],
                                                  ['ep', 'route', 'U', 'class', 'end region', 'end xy', 'max arc', 'stall onset', 'stall len', 'net 100', 'path 100', 'clearance', 'torso z', 'up z', 'first y>=2', 'first y>=6', 'first east col']), '']
  if (OUT / 'fork.json').exists():
    F = MP.read_json(OUT / 'fork.json')
    L += ['## 2. Where the route signal acts', '']
    # (a) critic scores
    L += ['### 2a. Critic scores of the seven policies\' actual reset-state torques (min over the twin heads, task goal; mean over the 300 reset states)', '']
    crit = sorted({k.split('|')[0] for k in F['critic_scores']})
    rows = []
    for c in crit:
      rows.append([c] + [f"{np.mean(F['critic_scores'][f'{c}|{n}']):+.2f}" for n in P])
    L += _tab(rows, ['critic \\ torque of'] + list(P)) + ['']
    L += ['Per-state preference: share of reset states where the critic scores the CF torque of its own seed above the O torque / above the start torque:', '']
    rows = []
    for s in SEEDS:
      for c in (f'CF_s{s}', f'O_s{s}', 'van'):
        a, b, st = np.array(F['critic_scores'][f'{c}|CF_s{s}']), np.array(F['critic_scores'][f'{c}|O_s{s}']), np.array(F['critic_scores'][f'{c}|start'])
        rows.append([c, s, f'{(a > b).mean():.2f}', f'{(a > st).mean():.2f}', f'{(b > st).mean():.2f}'])
    L += _tab(rows, ['critic', 'seed', 'P(f(CF) > f(O))', 'P(f(CF) > f(start))', 'P(f(O) > f(start))']) + ['']
    L += ['The CF actor at its own reset states: its mode torque\'s score vs 64 samples of its own tanh-normal under its own critic (and the O actor likewise):', '']
    rows = [[n, f"{np.mean(v['mode_score']):+.2f}", f"{np.mean(v['sample_score_mean']):+.2f}", f"{np.mean(v['sample_score_max']):+.2f}", f"{np.mean(v['mode_rank_frac_below']):.2f}", f"{v['scale_mean']:.3f}"] for n, v in F['own_samples'].items()]
    L += _tab(rows, ['actor', 'f(mode)', 'mean f(samples)', 'max f(samples)', 'frac samples below the mode', 'policy scale']) + ['']
    L += ['Log-likelihood of the other policies\' torques under each actor (mean over reset states): ', '']
    rows = [[c] + [f"{np.mean(F['logp_under_policy'][f'{c}|{n}']):+.1f}" for n in P] for c in sorted({k.split('|')[0] for k in F['logp_under_policy']})]
    L += _tab(rows, ['log pi_actor(torque of ...)'] + list(P)) + ['']
    # (b) swaps at t = 0
    L += ['### 2b. First action swapped at the identical reset state and hidden draw (route realised; n = number of episodes)', '']
    res = F['branch_results']
    rows = []
    for s in SEEDS:
      names = ['start', f'O_s{s}', f'CF_s{s}']
      for cont in names:
        for first in names:
          rs = [r for r in res if r['tag'] == f'swap0|s{s}|{first}>{cont}']
          if rs:
            rows.append([s, cont, first, len(rs), f"{np.mean([r['route'] == 'detour' for r in rs]):.3f}", f"{np.mean([r['success'] for r in rs]):.3f}", f"{np.mean([r['failure'] for r in rs]):.3f}", f"{np.mean([r['timeout'] for r in rs]):.3f}"])
    L += _tab(rows, ['seed', 'continuation', 'first action from', 'n', 'detour', 'success', 'death', 'timeout']) + ['']
    fid = {}
    for n in ('start', 'CF_s2', 'O_s2'):
      rs = [r for r in res if r['tag'] == f'fidelity|{n}']
      if rs:
        T = Traj(n)
        same = sum(int(bool(r['success']) == bool(T.d['success'][r['episode']]) and str(r['route']) == str(T.d['route'][r['episode']]) and abs(int(r['steps']) - int(T.d['steps'][r['episode']])) <= 1) for r in rs)
        fid[n] = f'{same}/{len(rs)}'
    L += [f'Replay fidelity of the branch primitive (restored episode, one policy end-to-end, outcome+route+steps equal to the rollout): {fid}', '']
    # (c) divergence
    L += ['### 2c. Where the O and CF paths of the same episode separate (first step with xy gap > 0.5), and the single-action swap there', '']
    rows = []
    for s in SEEDS:
      dv = F['divergence'][str(s)] if str(s) in F['divergence'] else F['divergence'][s]
      taus = np.array([d['tau'] for d in dv]); diff = [d for d in dv if d['route_O'] != d['route_CF']]
      rows.append([s, len(dv), int((taus >= 0).sum()), f'{np.median(taus[taus >= 0]):.0f}' if (taus >= 0).any() else '-', f'{np.mean(taus[taus >= 0]):.0f}' if (taus >= 0).any() else '-',
                   len(diff), (f"{np.median([d['tau'] for d in diff if d['tau'] >= 0]):.0f}" if any(d['tau'] >= 0 for d in diff) else '-'),
                   (f"{np.median([d['CF_first_north'] for d in diff if d['CF_first_north'] >= 0]):.0f}" if any(d['CF_first_north'] >= 0 for d in diff) else '-'),
                   (f"{np.mean([np.hypot(*d['xy_at_tau_CF']) for d in diff if d['tau'] >= 0]):.1f}" if any(d['tau'] >= 0 for d in diff) else '-')])
    L += _tab(rows, ['seed', 'episodes', 'paths separate', 'median tau (all)', 'mean tau', 'episodes with different routes', 'median tau (different routes)', 'median step CF reaches y >= 2', 'mean |xy| at tau (CF)']) + ['']
    rows = []
    for s in SEEDS:
      for tag, lab in ((f'swapT|s{s}|CF@tau:CF>CF', 'CF state at tau, CF action, CF continues (control)'), (f'swapT|s{s}|CF@tau:O>CF', 'CF state at tau, O action once, CF continues'), (f'swapT|s{s}|O@tau:CF>O', 'O state at tau, CF action once, O continues')):
        rs = [r for r in res if r['tag'] == tag]
        if rs:
          rows.append([s, lab, len(rs), f"{np.mean([r['route'] == 'detour' for r in rs]):.3f}", f"{np.mean([r['success'] for r in rs]):.3f}"])
    L += _tab(rows, ['seed', 'variant', 'n', 'detour', 'success']) + ['']
    # (d) support
    sp = F['branch_support_t0']['summary']
    L += ['### 2d. What the CF branches (the CF critic\'s positive futures) said at the t = 0 anchors', '',
          f"{sp['n_t0_anchors']} reset-row anchors: {sp['n_around']} branches went around (y >= 6); success around {sp['around_success']} vs straight {sp['straight_success']} "
          f"(straight death {sp['straight_death']}); mean P_goal (gamma 0.999, r 0.5) around {sp['p_goal_around_mean']} vs straight {sp['p_goal_straight_mean']}.", '']
  if (OUT / 'cont.json').exists():
    C = MP.read_json(OUT / 'cont.json')['results']
    L += ['## 3. Continuation from the CF policy\'s own pre-stall states (its timeout episodes)', '',
          'prestall = 20 steps before the last progress maximum; enter_detour = the step the CF path first reached y >= 2; t100 = step 100 for the no-route timeouts.  '
          'Continuations: the CF policy itself (control), the start d05 policy, the O policy of the seed, the teacher\'s blind position driver.', '']
    rows = []
    for s in SEEDS:
      for tag in ('prestall', 'enter_detour', 't100'):
        for cont in (f'CF_s{s}', 'start', f'O_s{s}', 'driver'):
          rs = [r for r in C if r['tag'] == f'cont|s{s}|{tag}|{cont}']
          if rs:
            rows.append([s, tag, cont, len(rs), f"{np.mean([r['success'] for r in rs]):.2f}", f"{np.mean([r['failure'] for r in rs]):.2f}", f"{np.mean([r['timeout'] for r in rs]):.2f}",
                         f"{np.mean([r['prefix_ended'] for r in rs]):.2f}", f"{np.median([r['steps'] for r in rs if r['success']]):.0f}" if any(r['success'] for r in rs) else '-'])
    L += _tab(rows, ['seed', 'start point', 'continuation', 'n', 'reach', 'death', 'timeout', 'prefix ended early', 'median total steps (reached)']) + ['']
  if (OUT / 'entries.json').exists():
    E = MP.read_json(OUT / 'entries.json')
    L += ['## 4. Entry times from the recorded xy (per policy, 300 episodes)', '',
          *_tab([[n, f"{v['summary']['turn_north']:.2f}", v['summary']['median_first_north'], f"{v['summary']['reach_top']:.2f}", v['summary']['median_first_top'], f"{v['summary']['reach_east_column']:.2f}", v['summary']['median_first_east_column'],
                  f"{v['summary']['enter_zone1']:.2f}", v['summary']['median_first_zone1'], f"{v['summary']['enter_zone2']:.2f}", v['summary']['median_first_zone2'],
                  f"{v['summary']['entered_zone1_while_active']} / {v['summary']['died_zone1']}", f"{v['summary']['entered_zone2_while_active']} / {v['summary']['died_zone2']}", f"{v['summary']['deaths_in_burst_window']} / {v['summary']['deaths']}", v['summary']['detour_episodes_touching_a_zone']]
                 for n, v in E.items()],
                ['policy', 'turn north (y>=2)', 'median step', 'reach top (y>=6)', 'median step', 'reach east column', 'median step', 'enter zone 1', 'median step', 'enter zone 2', 'median step',
                 'entered zone 1 while active / died there', 'entered zone 2 while active / died there', 'deaths inside the burst window / deaths', 'detour episodes that touched a zone']), '']
  if (OUT / 'objective.json').exists():
    Ob = MP.read_json(OUT / 'objective.json')
    L += ['## 5. The actor objective at real start-region rows (each actor with its own critic)', '',
          f"{Ob['sampled_rows']} of {Ob['start_region_rows']} logged rows with x < 2, y < 2, t <= 5 ({Ob['north_rows']} next-frame north-moving, {Ob['east_rows']} east-moving) and the {Ob['reset_rows']} reset-row anchors.  "
          'q-term = -f(s, a_sampled, g) (alpha 0), BC = -log pi(a_logged | s, g); weights 0.95 / 0.05 as in the loss; gradient norms w.r.t. the policy parameters.', '']
    for label in ('start_region_t<=5', 'reset_rows_t0'):
      rows = []
      for n, v in Ob['per_actor'].items():
        d = v[label]
        rows.append([n, f"{d['f_mode']:+.2f}", f"{d['f_own_sample']:+.2f}", f"{d['f_logged']:+.2f}", f"{d['P(f_mode > f_logged)']:.2f}", f"{d['bc_nll_logged']:.1f}", f"{d['bc_nll_median']:.1f}", f"{d['policy_scale_mean']:.2f}", f"{d['mode_minus_logged_torque_L2']:.2f}",
                     f"{d['weighted_q_term']:+.2f}", f"{d['weighted_bc_term']:+.2f}", f"{d['grad_norm_weighted_q_term']:.2f}", f"{d['grad_norm_weighted_bc_term']:.2f}", f"{d['grad_ratio_bc_over_q']:.2f}"]
                    + ([f"{d['f_logged_north']:+.2f}", f"{d['f_logged_east']:+.2f}", f"{d['f_logged_north_minus_east']:+.2f}"] if 'f_logged_north' in d else []))
      L += [f'### {label}', '', *_tab(rows, ['actor', 'f(mode)', 'f(own sample)', 'f(logged)', 'P(f mode > f logged)', 'BC NLL logged (mean)', 'median', 'scale', '|mode - logged|', '0.95 q-term', '0.05 BC', '|grad| q', '|grad| BC', 'BC / q grad ratio'] + (['f logged north', 'f logged east', 'north - east'] if label.startswith('start') else [])), '']
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('rollout', 'timeouts', 'fork', 'cont', 'entries', 'objective', 'report'))
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  {'rollout': mode_rollout, 'timeouts': mode_timeouts, 'fork': mode_fork, 'cont': mode_cont, 'entries': mode_entries, 'objective': mode_objective, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
