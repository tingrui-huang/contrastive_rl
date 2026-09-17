"""AntMaze V6 branch replay: interventional futures for recorded anchors (oracle model).

The PointMaze Step 12 rule carried to the long two-rockfall V6 benchmark
(`notes/antmaze_branch_replay_plan.md`).  For a recorded anchor (s_t, a_t)
the future is what a BLIND agent gets after it: the Ant's 29-dim state is
restored in the simulator, the two hazard latents and both clocks are
REDRAWN from their priors (the recorded values were the ones the sighted
teacher acted on -- keeping them would keep the confounding), the rocks
are parked, the absolute time is set to t, and the path continues

  * for the first ``--k-recorded`` steps with the recorded torques (the
    anchor's own action, as PointMaze kept the recorded first action), then
  * with the frozen walker driven by the teacher's own leg tables, never
    consulting a schedule: the detour legs when the position says detour
    (y >= 6, or x < 2 with y > 2), the shortcut legs with the blind 'go'
    intent otherwise;

until success, death or the horizon, exactly as the collector ends recorded
episodes.  At start-region anchors (x < 2, y < 2) BOTH routes are branched
from step 0 (the query coverage of PointMaze arm C).  The simulator is the
model here -- an oracle upper bound outside the offline setting, declared as
such; a learned macro-model replaces it in Phase 2.

Modes
  phase0   the interventional and observational law at the start region
           (reach / death / reach time per route, and the relabeling law's
           goal-frame probability at gamma 0.999 and 0.99), plus the
           placed-on-detour continuation audit
  replay   the branch replay: anchors every ``--every`` recorded steps of
           every episode, plus the start-region query branches; written in
           the dataset's gxy contract (state 29 | goal_xy 2, act 8, lengths)

Rows are generated with multiprocessing (one env + teacher per worker).
Outputs: outputs/antmaze_branch_replay_v1/.
"""
from __future__ import annotations

import argparse
import json
import os

# one worker = one core: keep BLAS / XLA thread pools from multiplying across
# the process pool (20 workers x 22 BLAS threads exhausted the machine)
for _v in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
  os.environ.setdefault(_v, '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

DATASET = ROOT / 'artifacts' / 'rockfall_clock_v6' / 'dataset' / 'antmaze_rockfall_clock_v6_p040_gxy.npz'
SIDECAR = ROOT / 'artifacts' / 'rockfall_clock_v6' / 'dataset' / 'antmaze_rockfall_clock_v6_p040_sidecar.npz'
OUT = Path(os.environ.get('V6_BRANCH_OUT', ROOT / 'outputs' / 'antmaze_branch_replay_v1'))
STATE_DIM, NQ, NV, ACTION_DIM = 29, 15, 14, 8
HORIZON = 800
START_REGION = dict(x_max=2.0, y_max=2.0)
DETOUR_Y = 6.0
GEN_SEED = 606_000
PARK = os.environ.get('V6_BRANCH_PARK', '1') == '1'    # continue to the horizon after reaching

_ENV = None
_TEACHER = None


def _worker_env(seed):
  """One env + teacher per process, built lazily."""
  global _ENV, _TEACHER
  if _ENV is None:
    from crl import envs as envs_mod
    import rockfall_clock_v6_teacher as CT
    cfg, teacher = CT.make_teacher()
    _ENV = envs_mod.make_env(CT.ENV_NAME, cfg, seed=int(seed))
    _TEACHER = teacher
  return _ENV, _TEACHER


def route_from_position(x, y):
  return 'detour' if (y >= DETOUR_Y or (x < START_REGION['x_max'] and y > 2.0)) else 'shortcut'


def restore(env, state, goal_xy, t):
  """Fresh hazards / clocks / jitter (env.reset draws them), then the Ant's
  recorded state, the recorded goal and the absolute time."""
  import mujoco
  env.reset()
  u = env._env
  u.data.qpos[:NQ] = state[:NQ]
  u.data.qvel[:NV] = state[NQ:NQ + NV]
  mujoco.mj_forward(u.model, u.data)
  full = np.array(env._goal_state_full, np.float32)
  full[:2] = goal_xy
  env._goal_state_full = full
  env._goal_vec = full[list(env.goal_indices)]
  u.goal = np.asarray(goal_xy, float).copy()
  env._last_obs = u._obs_dict()
  env._t = int(t)
  o = env._flatten(env._last_obs)
  return o


def rollout(env, teacher, state, goal_xy, t, recorded_actions, intent, max_steps, park=True):
  """One branch.  ``recorded_actions`` (possibly empty) are applied first,
  then the blind driver with ``intent`` ('detour' / 'go') or, if intent is
  None, the route the position implies after the recorded segment."""
  o = restore(env, state, goal_xy, t)
  rows_o, rows_a = [o[:STATE_DIM + 2].copy()], []
  done, reward, info = False, 0.0, {}
  step, reached, reach_step = 0, False, None
  for a in recorded_actions:
    if step >= max_steps:
      break
    if reached:
      a = np.zeros(ACTION_DIM, np.float32)
    o, reward, done, info = env.step(np.asarray(a, np.float32))
    rows_a.append(np.asarray(a, np.float32)); rows_o.append(o[:STATE_DIM + 2].copy())
    step += 1
    if reward > 0 and reach_step is None:
      reach_step = step
    reached = reached or reward > 0
    if done or (reward > 0 and not park):
      break
  if not (done or (reached and not park)) and step < max_steps:
    if intent is None:
      route = route_from_position(float(o[0]), float(o[1]))
    else:
      route = 'detour' if intent == 'detour' else 'shortcut'
    drive_intent = 'detour' if route == 'detour' else 'go'
    teacher.fresh(route=route)
    while step < max_steps:
      # after the goal is reached the path HOLDS (zero torque, the teacher's
      # own waiting action): goal states are absorbing for the relabeling
      # law, and the hold is the same for both routes -- the drivers' own
      # post-goal behaviour (hovering vs pushing into a wall) is not
      a = np.zeros(ACTION_DIM, np.float32) if reached else teacher.act(o, env.schedule, intent=drive_intent)
      o, reward, done, info = env.step(np.asarray(a, np.float32))
      rows_a.append(np.asarray(a, np.float32)); rows_o.append(o[:STATE_DIM + 2].copy())
      step += 1
      if reward > 0 and reach_step is None:
        reach_step = step
      reached = reached or reward > 0
      if done or (reward > 0 and not park):
        break
  rows_a.append(np.zeros(ACTION_DIM, np.float32))          # dummy action on the last obs row
  return {'obs': np.stack(rows_o), 'act': np.stack(rows_a), 'steps': step, 'reach_step': reach_step,
          'success': bool(reached), 'failure': bool(info.get('failure', False)),
          'route': info.get('route'), 'u1': bool(env.privileged_rockfall_active_1),
          'u2': bool(env.privileged_rockfall_active_2)}


def _run_jobs(args):
  """Worker: a list of jobs (episode, t, intent, kind, k_recorded) -> results."""
  jobs, seed = args
  env, teacher = _worker_env(seed)
  with np.load(DATASET, allow_pickle=False) as d:
    obs, act, lengths = d['obs'], d['act'], d['lengths']
  out = []
  for (e, t, intent, kind, k_rec) in jobs:
    state = obs[e, t, :STATE_DIM].astype(np.float64)
    goal_xy = obs[e, t, STATE_DIM:STATE_DIM + 2].astype(np.float64)
    last = int(lengths[e]) - 1                  # last obs row; actions valid for rows < last
    rec = act[e, t:min(t + k_rec, last)] if k_rec > 0 else act[e, t:t]
    r = rollout(env, teacher, state, goal_xy, t, rec, intent, HORIZON - t, park=PARK)
    r.update({'episode': int(e), 't': int(t), 'kind': kind, 'intent': intent, 'k_rec': int(len(rec))})
    out.append(r)
  return out


def _chunks(jobs, n):
  k = max(1, len(jobs) // n + (len(jobs) % n > 0))
  return [jobs[i:i + k] for i in range(0, len(jobs), k)]


def run_parallel(jobs, workers, seed0):
  parts = _chunks(jobs, workers * 4)
  args = [(p, seed0 + i) for i, p in enumerate(parts)]
  if workers <= 1:
    res = [_run_jobs(a) for a in args]
  else:
    with Pool(workers) as pool:
      res = pool.map(_run_jobs, args)
  return [r for part in res for r in part]


def goal_frame_probability(path_obs, goal_xy, gamma, radius):
  """The buffer's discounted relabeling law from row 0: P(future frame within
  ``radius`` of the goal)."""
  L = len(path_obs)
  if L < 2:
    return 0.0
  j = np.arange(1, L)
  w = gamma ** j
  near = np.linalg.norm(path_obs[1:, :2] - goal_xy[None], axis=1) <= radius
  return float((w * near).sum() / w.sum())


# ----------------------------------------------------------------- phase 0
def phase0(args):
  rng = np.random.default_rng(args.seed)
  with np.load(DATASET, allow_pickle=False) as d:
    obs, lengths = d['obs'], d['lengths']
  with np.load(SIDECAR, allow_pickle=True) as sc:
    route_rec = sc['route_realized'].astype(str)
  n_eps, L = obs.shape[:2]
  valid = np.arange(L)[None, :] < (lengths[:, None] - 1)
  x, y = obs[:, :, 0], obs[:, :, 1]
  start = valid & (x < START_REGION['x_max']) & (y < START_REGION['y_max']) & (np.arange(L)[None, :] <= args.t_max)
  ee, tt = np.nonzero(start)
  # half of the anchors from t <= 5 (the decision the deployed actor faces), half from the rest
  early = np.flatnonzero(tt <= 5); late = np.flatnonzero(tt > 5)
  pick = np.concatenate([rng.choice(early, size=min(args.n_start // 2, len(early)), replace=False),
                         rng.choice(late, size=min(args.n_start - args.n_start // 2, len(late)), replace=False)])
  jobs = []
  for i in pick:
    for intent in ('detour', 'go'):
      jobs.append((int(ee[i]), int(tt[i]), intent, 'start_query', 0))
  # placed-on-detour audit: rows of recorded detour episodes on the north leg and the east leg
  det = np.flatnonzero(route_rec == 'detour')
  north = [(int(e), int(t)) for e in det for t in range(1, int(lengths[e]) - 1)
           if x[e, t] < 2.0 and 2.0 < y[e, t] < 7.5]
  east = [(int(e), int(t)) for e in det for t in range(1, int(lengths[e]) - 1)
          if y[e, t] >= 7.5 and 2.0 < x[e, t] < 22.0]
  for pool_, name in ((north, 'placed_north'), (east, 'placed_east')):
    if pool_:
      sel = rng.choice(len(pool_), size=min(args.n_placed, len(pool_)), replace=False)
      jobs += [(pool_[i][0], pool_[i][1], 'detour', name, 0) for i in sel]
  print(f'phase0: {len(pick)} start anchors x 2 routes, {sum(j[3] != "start_query" for j in jobs)} placed rows; '
        f'{args.workers} workers', flush=True)
  t0 = time.time()
  res = run_parallel(jobs, args.workers, GEN_SEED + 1)
  print(f'rollouts done in {time.time() - t0:.0f}s', flush=True)
  summary = {'n_start_anchors': int(len(pick)), 'wall_seconds': time.time() - t0, 'branches': {}}
  L_ = ['# Phase 0: the interventional law at the start region, and the placed-on-detour audit', '',
        f'{len(pick)} recorded start-region anchors (x < 2, y < 2, t <= {args.t_max}); each branched '
        'north (detour legs) and east (shortcut legs, blind go) from its restored Ant state with '
        'hazards and clocks redrawn from the priors.  Goal-frame probability = the discounted '
        'relabeling law from the anchor over the branch path (paths end at success / death / horizon, '
        'unless parking is on: with V6_BRANCH_PARK=1 (default) a path continues to the horizon after reaching, '
        'holding with zero torque, so the goal is an absorbing parked tail instead of the single last frame the recorded episodes end on); '
        'radius 0.5 = the success distance.', '',
        '| branch | n | reach | death | timeout | reach step (median) | P_goal g=0.999 r=0.5 | r=1.0 | P_goal g=0.99 r=0.5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for kind, intent in (('start_query', 'detour'), ('start_query', 'go'), ('placed_north', 'detour'), ('placed_east', 'detour')):
    rs = [r for r in res if r['kind'] == kind and r['intent'] == intent]
    if not rs:
      continue
    reach = np.array([r['success'] for r in rs]); death = np.array([r['failure'] for r in rs])
    steps = np.array([r['t'] + (r['reach_step'] if r['reach_step'] is not None else r['steps']) for r in rs])
    p999 = np.mean([goal_frame_probability(r['obs'], r['obs'][0, STATE_DIM:STATE_DIM + 2], 0.999, 0.5) for r in rs])
    p999w = np.mean([goal_frame_probability(r['obs'], r['obs'][0, STATE_DIM:STATE_DIM + 2], 0.999, 1.0) for r in rs])
    p99 = np.mean([goal_frame_probability(r['obs'], r['obs'][0, STATE_DIM:STATE_DIM + 2], 0.99, 0.5) for r in rs])
    row = {'n': len(rs), 'reach': float(reach.mean()), 'death': float(death.mean()),
           'timeout': float((~reach & ~death).mean()),
           'reach_step_median': float(np.median(steps[reach])) if reach.any() else None,
           'p_goal_0999_r05': float(p999), 'p_goal_0999_r10': float(p999w), 'p_goal_099_r05': float(p99)}
    summary['branches'][f'{kind}|{intent}'] = row
    L_.append(f'| {kind} / {intent} | {row["n"]} | {row["reach"]:.3f} | {row["death"]:.3f} | {row["timeout"]:.3f} '
              f'| {row["reach_step_median"]} | {p999:.5f} | {p999w:.5f} | {p99:.5f} |')
  # the decision the deployed actor faces is at the reset pose: targets by anchor time
  L_.append('')
  L_.append('| anchor time | n per route | detour reach | go reach | target g=0.999 r=0.5 | r=1.0 |')
  L_.append('|---|---:|---:|---:|---:|---:|')
  summary['by_anchor_time'] = {}
  for lo, hi in ((0, 0), (0, 5), (0, 10), (11, 60)):
    rd = [r for r in res if r['kind'] == 'start_query' and r['intent'] == 'detour' and lo <= r['t'] <= hi]
    rg = [r for r in res if r['kind'] == 'start_query' and r['intent'] == 'go' and lo <= r['t'] <= hi]
    if not rd or not rg:
      continue
    row = {}
    for radius in (0.5, 1.0):
      pd_ = np.mean([goal_frame_probability(r['obs'], r['obs'][0, STATE_DIM:STATE_DIM + 2], 0.999, radius) for r in rd])
      pg_ = np.mean([goal_frame_probability(r['obs'], r['obs'][0, STATE_DIM:STATE_DIM + 2], 0.999, radius) for r in rg])
      row[f'target_r{radius}'] = float(np.log(pd_ / max(pg_, 1e-12)))
    row.update({'n': len(rd), 'detour_reach': float(np.mean([r['success'] for r in rd])),
                'go_reach': float(np.mean([r['success'] for r in rg]))})
    summary['by_anchor_time'][f'{lo}-{hi}'] = row
    L_.append(f'| {lo}-{hi} | {len(rd)} | {row["detour_reach"]:.2f} | {row["go_reach"]:.2f} | '
              f'**{row["target_r0.5"]:+.2f}** | {row["target_r1.0"]:+.2f} |')
  b = summary['branches']
  if 'start_query|detour' in b and 'start_query|go' in b:
    for key, label in (('p_goal_0999_r05', 'gamma 0.999, r 0.5'), ('p_goal_0999_r10', 'gamma 0.999, r 1.0'),
                       ('p_goal_099_r05', 'gamma 0.99, r 0.5')):
      lr = float(np.log(b['start_query|detour'][key] / max(b['start_query|go'][key], 1e-12)))
      summary[f'target_{key}'] = lr
      L_.append(f'\ninterventional target at the start region, {label}: log P(detour) / P(shortcut) = **{lr:+.3f}**')
  # observational counterpart: the recorded futures of the same start-region rows
  obs_rows = {'detour': [], 'shortcut': []}
  for i in pick:
    e, t = int(ee[i]), int(tt[i])
    path = obs[e, t:int(lengths[e])]
    obs_rows[route_rec[e]].append(goal_frame_probability(path, path[0, STATE_DIM:STATE_DIM + 2], 0.999, 0.5))
  if obs_rows['detour'] and obs_rows['shortcut']:
    lr_obs = float(np.log(np.mean(obs_rows['detour']) / np.mean(obs_rows['shortcut'])))
    summary['observational_target_0999_r05'] = lr_obs
    L_.append(f'\nobservational counterpart (recorded futures, sighted teacher; {len(obs_rows["detour"])} detour / '
              f'{len(obs_rows["shortcut"])} shortcut rows), gamma 0.999, r 0.5: **{lr_obs:+.3f}**')
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / 'phase0_REPORT.md').write_text('\n'.join(L_) + '\n', encoding='utf-8')
  (OUT / 'phase0_results.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
  np.savez_compressed(OUT / 'phase0_branches.npz',
                      episode=np.array([r['episode'] for r in res]), t=np.array([r['t'] for r in res]),
                      kind=np.array([r['kind'] for r in res]), intent=np.array([r['intent'] for r in res]),
                      success=np.array([r['success'] for r in res]), failure=np.array([r['failure'] for r in res]),
                      steps=np.array([r['steps'] for r in res]), u1=np.array([r['u1'] for r in res]),
                      u2=np.array([r['u2'] for r in res]))
  print('\n'.join(L_), flush=True)


# ------------------------------------------------------------------ replay
def build_replay(args):
  with np.load(DATASET, allow_pickle=False) as d:
    obs, lengths, meta = d['obs'], d['lengths'], str(d['meta'])
  n_eps, L = obs.shape[:2]
  jobs = []
  for e in range(n_eps):
    last = int(lengths[e]) - 1
    for t in range(0, last, args.every):
      jobs.append((e, t, None, 'branch', args.k_recorded))
    for t in range(0, min(last, args.t_max + 1)):
      xx, yy = obs[e, t, 0], obs[e, t, 1]
      if xx < START_REGION['x_max'] and yy < START_REGION['y_max'] and t % args.query_every == 0:
        jobs.append((e, t, 'detour', 'query', 0))
        jobs.append((e, t, 'go', 'query', 0))
  if args.limit:
    jobs = jobs[:args.limit]
  print(f'replay: {sum(j[3] == "branch" for j in jobs)} branch paths + {sum(j[3] == "query" for j in jobs)} '
        f'query paths, {args.workers} workers', flush=True)
  t0 = time.time()
  res = run_parallel(jobs, args.workers, GEN_SEED + 100)
  n = len(res)
  obs_p = np.zeros((n, L, STATE_DIM + 2), np.float32)
  act_p = np.zeros((n, L, ACTION_DIM), np.float32)
  lengths_p = np.zeros(n, np.int64)
  for i, r in enumerate(res):
    k = len(r['obs'])
    obs_p[i, :k] = r['obs']; act_p[i, :k] = r['act']; lengths_p[i] = k
    obs_p[i, k:] = r['obs'][-1]                     # never sampled: masked by lengths
  m = json.loads(meta)
  m.update({'arm': 'branch_replay_oracle', 'source_dataset': str(DATASET), 'every': args.every,
            'k_recorded': args.k_recorded, 'query_every': args.query_every, 'gen_seed': GEN_SEED,
            'anchor_rule': 'row 0 of every path (set_anchor_strata in the driver)',
            'hazards': 'redrawn from the priors at every branch (interventional continuation)'})
  OUT.mkdir(parents=True, exist_ok=True)
  path = OUT / ('replay_branch.npz' if not args.limit else 'replay_branch_smoke.npz')
  np.savez_compressed(
      path, obs=obs_p, act=act_p, lengths=lengths_p.astype(np.int64),
      eval_goals=obs_p[:, 0, STATE_DIM:STATE_DIM + 2].astype(np.float32),
      meta=np.asarray(json.dumps(m, sort_keys=True)),
      audit_kind=np.array([r['kind'] for r in res]), audit_intent=np.array([str(r['intent']) for r in res]),
      audit_episode=np.array([r['episode'] for r in res], np.int32), audit_anchor_time=np.array([r['t'] for r in res], np.int16),
      audit_success=np.array([r['success'] for r in res]), audit_failure=np.array([r['failure'] for r in res]),
      audit_u1=np.array([r['u1'] for r in res]), audit_u2=np.array([r['u2'] for r in res]))
  succ = np.array([r['success'] for r in res]); fail = np.array([r['failure'] for r in res])
  kinds = np.array([r['kind'] for r in res]); intents = np.array([str(r['intent']) for r in res])
  summary = {'paths': n, 'wall_seconds': time.time() - t0, 'reach': float(succ.mean()), 'death': float(fail.mean()),
             'mean_length': float(lengths_p.mean()),
             'by_kind': {f'{k}|{i}': {'n': int(((kinds == k) & (intents == i)).sum()),
                                     'reach': float(succ[(kinds == k) & (intents == i)].mean()),
                                     'death': float(fail[(kinds == k) & (intents == i)].mean())}
                         for k, i in {(k, i) for k, i in zip(kinds, intents)}},
             'bytes': int(os.path.getsize(path)), 'path': str(path)}
  (OUT / ('generation.json' if not args.limit else 'generation_smoke.json')).write_text(json.dumps(summary, indent=1), encoding='utf-8')
  print(json.dumps(summary, indent=1), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('phase0', 'replay'))
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n-start', type=int, default=400, help='phase0: start-region anchors')
  ap.add_argument('--n-placed', type=int, default=100, help='phase0: placed-on-detour rows per leg')
  ap.add_argument('--t-max', type=int, default=60, help='start-region anchors: latest anchor time')
  ap.add_argument('--every', type=int, default=20, help='replay: anchor every k recorded steps')
  ap.add_argument('--query-every', type=int, default=10, help='replay: start-region query anchors every k steps')
  ap.add_argument('--k-recorded', type=int, default=25, help='replay: recorded torques before the driver')
  ap.add_argument('--limit', type=int, default=0, help='replay: smoke with the first N jobs')
  args = ap.parse_args(argv)
  if args.mode == 'phase0':
    phase0(args)
  else:
    build_replay(args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
