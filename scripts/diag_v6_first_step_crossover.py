"""Oracle-stage diagnostic: how much of a start-region outcome does the FIRST
torque explain, against the 24-step continuation the critic never sees?

For each start-region anchor (a recorded row with x < 2, y < 2, t <= 5) take
one north donor and one east donor (recorded episodes whose 25-step
displacement from time t points north / east, as the replay generator does)
and cross their segments:

    NN  north first torque + north continuation (steps 2..25)
    NE  north first torque + east continuation
    EN  east first torque  + north continuation
    EE  east first torque  + east continuation

The four arms restore the same Ant state at the same absolute time with the
SAME hazards, clocks and rock jitter (the env's six hidden streams are
reseeded per anchor before every reset), run the 25 torques open-loop, and
then hand over to the same blind driver, which picks its route from the
position it finds itself in (never from a query label) and holds with zero
torque after reaching.  Read: the XY direction after the 25 steps, the route
the driver then takes, and the full outcome (reach / death / timeout, the
goal-frame probability under the replay's own law).

Mode ``walker``: the same detour-corridor rows Phase 0 placed (north leg,
east leg of recorded detour episodes), each run (a) by the generation walker
(frozen walker + blind driver, intent detour) and (b) by a deployment BC
actor's mode, paired hazards -- separates "wrong route" from "right route,
cannot finish".

  V6_DATASET_STEM=... V6_P_ACTIVE=... V6_BRANCH_OUT=... \\
    python scripts/diag_v6_first_step_crossover.py crossover --n-anchors 300
    python scripts/diag_v6_first_step_crossover.py walker --bc-ckpt <pure-BC final.pkl>

Diagnostic only: it uses the simulator as the model (Phase 1's oracle) and
is not an offline training result.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402

SEG = 25
GAMMA, RADIUS = 0.999, 0.5
ARMS = ('NN', 'NE', 'EN', 'EE')
DIAG_SEED = 909_000


def reseed(env, seed):
  """Identical hidden draws (U1, U2, both clocks, both jitters) at the next reset."""
  from crl import rockfall_clock_v6 as V6
  env._active_rng = {1: np.random.default_rng(seed + V6._ACTIVE_SEED_OFFSET_1),
                     2: np.random.default_rng(seed + V6._ACTIVE_SEED_OFFSET_2)}
  env._sched_rng = {1: np.random.default_rng(seed + V6._SCHED_SEED_OFFSET_1),
                    2: np.random.default_rng(seed + V6._SCHED_SEED_OFFSET_2)}
  env._jitter_rng = {1: np.random.default_rng(seed + V6._JITTER_SEED_OFFSET_1),
                     2: np.random.default_rng(seed + V6._JITTER_SEED_OFFSET_2)}


def direction(d):
  dx, dy = float(d[0]), float(d[1])
  if dy > 1.0 and dy > abs(dx):
    return 'north'
  if dx > 1.0 and dx > abs(dy):
    return 'east'
  return 'neither'


def _run_crossover_jobs(args):
  jobs, seed = args
  env, teacher = B._worker_env(seed)
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
  out = []
  for (i, e, t, arm, d_n, d_e, pair_seed) in jobs:
    state = obs[e, t, :B.STATE_DIM].astype(np.float64)
    goal_xy = obs[e, t, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float64)
    seg_n, seg_e = act[d_n, t:t + SEG], act[d_e, t:t + SEG]
    first = seg_n[:1] if arm[0] == 'N' else seg_e[:1]
    rest = seg_n[1:] if arm[1] == 'N' else seg_e[1:]
    seg = np.concatenate([first, rest])
    reseed(env, pair_seed)
    r = B.rollout(env, teacher, state, goal_xy, t, seg, None, B.HORIZON - t, park=True)
    k = min(SEG, len(r['obs']) - 1)
    disp = r['obs'][k, :2] - r['obs'][0, :2]
    out.append({'anchor': int(i), 'episode': int(e), 't': int(t), 'arm': arm, 'donor_north': int(d_n), 'donor_east': int(d_e),
                'disp25': disp.tolist(), 'dir25': direction(disp), 'steps_recorded': int(k),
                'driver_route': B.route_from_position(float(r['obs'][k, 0]), float(r['obs'][k, 1])),
                'success': r['success'], 'failure': r['failure'], 'steps': r['steps'], 'reach_step': r['reach_step'],
                'p_goal': B.goal_frame_probability(r['obs'], goal_xy, GAMMA, RADIUS),
                'u1': r['u1'], 'u2': r['u2'], 'final_xy': r['obs'][-1, :2].tolist()})
  return out


def _run_walker_jobs(args):
  jobs, seed, bc_ckpt = args
  env, teacher = B._worker_env(seed)
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs = d['obs']
  act_fn = None
  if bc_ckpt:
    import jax
    import jax.numpy as jnp
    from crl import checkpoint, networks
    cfg = teacher_cfg()
    nets = networks.make_networks(
        obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim, repr_dim=int(cfg.repr_dim),
        repr_norm=cfg.repr_norm, repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
        twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
    _, st = checkpoint.load_checkpoint(bc_ckpt)
    pp = st.policy_params

    @jax.jit
    def _mode(o):
      return jnp.tanh(nets.policy_network.apply(pp, o).loc)
    act_fn = lambda o: np.asarray(_mode(jnp.asarray(o[None, :B.STATE_DIM + 2], jnp.float32)))[0]
  out = []
  for (i, e, t, leg, walker, pair_seed) in jobs:
    state = obs[e, t, :B.STATE_DIM].astype(np.float64)
    goal_xy = obs[e, t, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float64)
    reseed(env, pair_seed)
    if walker == 'driver':
      r = B.rollout(env, teacher, state, goal_xy, t, [], 'detour', B.HORIZON - t, park=True)
      rec = {'success': r['success'], 'failure': r['failure'], 'steps': r['steps'], 'reach_step': r['reach_step'],
             'final_xy': r['obs'][-1, :2].tolist(), 'p_goal': B.goal_frame_probability(r['obs'], goal_xy, GAMMA, RADIUS)}
    else:
      o = B.restore(env, state, goal_xy, t)
      rows = [o[:B.STATE_DIM + 2].copy()]
      done, reached, reach_step, step, info = False, False, None, 0, {}
      while step < B.HORIZON - t and not done:
        a = np.zeros(B.ACTION_DIM, np.float32) if reached else act_fn(o)
        o, reward, done, info = env.step(np.asarray(a, np.float32))
        rows.append(o[:B.STATE_DIM + 2].copy()); step += 1
        if reward > 0 and reach_step is None:
          reach_step = step
        reached = reached or reward > 0
      rows = np.stack(rows)
      rec = {'success': bool(reached), 'failure': bool(info.get('failure', False)), 'steps': step, 'reach_step': reach_step,
             'final_xy': rows[-1, :2].tolist(), 'p_goal': B.goal_frame_probability(rows, goal_xy, GAMMA, RADIUS)}
    rec.update({'row': int(i), 'episode': int(e), 't': int(t), 'leg': leg, 'walker': walker})
    out.append(rec)
  return out


def teacher_cfg():
  import rockfall_clock_v6_teacher as CT
  cfg, _ = CT.make_teacher(p_active_1=B.P_ACTIVE, p_active_2=B.P_ACTIVE)
  return cfg


def chunks(jobs, n):
  k = max(1, len(jobs) // n + (len(jobs) % n > 0))
  return [jobs[i:i + k] for i in range(0, len(jobs), k)]


def run(fn, jobs, workers, extra=()):
  parts = chunks(jobs, workers * 4)
  args = [(p, DIAG_SEED + 100 + i, *extra) for i, p in enumerate(parts)]
  if workers <= 1:
    res = [fn(a) for a in args]
  else:
    with Pool(workers) as pool:
      res = pool.map(fn, args)
  return [r for part in res for r in part]


# --------------------------------------------------------------- crossover
def crossover(args):
  rng = np.random.default_rng(args.seed)
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs, lengths = d['obs'], d['lengths']
  n_eps, L = obs.shape[:2]
  valid = np.arange(L)[None, :] < (lengths[:, None] - 1)
  x, y = obs[:, :, 0], obs[:, :, 1]
  start = valid & (x < B.START_REGION['x_max']) & (y < B.START_REGION['y_max']) & (np.arange(L)[None, :] <= args.t_max)
  ee, tt = np.nonzero(start)
  donors_north, donors_east = {}, {}
  for t in range(0, args.t_max + 1):
    ok = (lengths - 1) > t + SEG
    dd = obs[:, min(t + SEG, L - 1), :2] - obs[:, t, :2]
    donors_north[t] = np.flatnonzero(ok & (dd[:, 1] > 1.0) & (dd[:, 1] > np.abs(dd[:, 0])))
    donors_east[t] = np.flatnonzero(ok & (dd[:, 0] > 1.0) & (dd[:, 0] > np.abs(dd[:, 1])))
  pick = rng.choice(len(ee), size=min(args.n_anchors, len(ee)), replace=False)
  jobs = []
  for i in pick:
    e, t = int(ee[i]), int(tt[i])
    d_n, d_e = int(rng.choice(donors_north[t])), int(rng.choice(donors_east[t]))
    pair_seed = DIAG_SEED + 10_000 + int(i)
    for arm in ARMS:
      jobs.append((int(i), e, t, arm, d_n, d_e, pair_seed))
  print(f'crossover: {len(pick)} anchors x 4 arms, {args.workers} workers', flush=True)
  t0 = time.time()
  res = run(_run_crossover_jobs, jobs, args.workers)
  print(f'rollouts done in {time.time() - t0:.0f}s', flush=True)
  out = B.OUT / 'diag_crossover'
  out.mkdir(parents=True, exist_ok=True)
  (out / 'rollouts.json').write_text(json.dumps(res), encoding='utf-8')
  report_crossover(res, out)


def report_crossover(res, out):
  by = {a: [r for r in res if r['arm'] == a] for a in ARMS}
  L_ = ['# First torque x continuation crossover at the start region (paired hazards, oracle model)', '',
        f'{len(by["NN"])} start-region anchors (t <= 5); each arm = the recorded first torque of one donor + steps 2..25 of '
        f'the other or the same donor, then the blind driver by position, zero-torque hold after reaching; gamma {GAMMA}, '
        f'radius {RADIUS}.  Direction after 25 steps: north = dy > 1 and dy > |dx|, east likewise.', '',
        '| arm | first | continuation | north @25 | east @25 | neither | driver picks detour | reach | death | timeout | P_goal | reach step (median) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
  P = {}
  for a in ARMS:
    rs = by[a]; n = len(rs)
    dirs = np.array([r['dir25'] for r in rs])
    P[a] = {'north25': float(np.mean(dirs == 'north')), 'east25': float(np.mean(dirs == 'east')),
            'neither25': float(np.mean(dirs == 'neither')),
            'detour': float(np.mean([r['driver_route'] == 'detour' for r in rs])),
            'reach': float(np.mean([r['success'] for r in rs])), 'death': float(np.mean([r['failure'] for r in rs])),
            'timeout': float(np.mean([(not r['success']) and (not r['failure']) for r in rs])),
            'p_goal': float(np.mean([r['p_goal'] for r in rs])),
            'reach_step_median': (float(np.median([r['reach_step'] for r in rs if r['reach_step'] is not None]))
                                  if any(r['reach_step'] is not None for r in rs) else None)}
    p = P[a]
    L_.append(f'| {a} | {"north" if a[0] == "N" else "east"} | {"north" if a[1] == "N" else "east"} | {p["north25"]:.2f} | '
              f'{p["east25"]:.2f} | {p["neither25"]:.2f} | {p["detour"]:.2f} | {p["reach"]:.2f} | {p["death"]:.2f} | '
              f'{p["timeout"]:.2f} | {p["p_goal"]:.3f} | {p["reach_step_median"]} |')
  # main effects and interaction (differences of means over the 2 x 2), per outcome
  L_ += ['', '| outcome | first-torque effect (N - E, averaged over continuations) | continuation effect (N - E, averaged over first torques) | interaction |',
         '|---|---:|---:|---:|']
  eff = {}
  for k, lab in (('north25', 'P(north after 25)'), ('detour', 'P(driver takes detour)'), ('reach', 'reach'),
                 ('death', 'death'), ('p_goal', 'P_goal')):
    first = 0.5 * ((P['NN'][k] - P['EN'][k]) + (P['NE'][k] - P['EE'][k]))
    cont = 0.5 * ((P['NN'][k] - P['NE'][k]) + (P['EN'][k] - P['EE'][k]))
    inter = (P['NN'][k] - P['NE'][k]) - (P['EN'][k] - P['EE'][k])
    eff[k] = {'first': first, 'continuation': cont, 'interaction': inter}
    L_.append(f'| {lab} | {first:+.3f} | {cont:+.3f} | {inter:+.3f} |')
  # the law's numbers: log P_goal ratios the critic would be asked to learn for the first torque alone
  lr = {}
  for a, b in (('NN', 'EE'), ('NE', 'EE'), ('EN', 'EE'), ('NN', 'EN'), ('NE', 'EE')):
    lr[f'{a}/{b}'] = float(np.log(max(P[a]['p_goal'], 1e-9) / max(P[b]['p_goal'], 1e-9)))
  L_ += ['', f'log P_goal ratios: NN/EE {lr["NN/EE"]:+.2f} (both differ), NE/EE {lr["NE/EE"]:+.2f} (only the first torque is north), '
             f'EN/EE {lr["EN/EE"]:+.2f} (only the continuation is north), NN/EN {lr["NN/EN"]:+.2f} (first torque, given a north continuation).',
         '', 'Reading: the first-torque effect is what a critic scoring (s, a_1) can carry at best; the continuation effect is what the '
             'replay row\'s realised future carries but the critic\'s input does not see.']
  # paired, within-anchor contrasts (same hazards): does swapping only the first torque change the direction / the outcome?
  anchors = sorted(set(r['anchor'] for r in res))
  idx = {(r['anchor'], r['arm']): r for r in res}
  flips_first, flips_cont, n_pairs = 0, 0, 0
  for i in anchors:
    if all((i, a) in idx for a in ARMS):
      n_pairs += 1
      flips_first += int(idx[(i, 'NN')]['dir25'] != idx[(i, 'EN')]['dir25']) + int(idx[(i, 'NE')]['dir25'] != idx[(i, 'EE')]['dir25'])
      flips_cont += int(idx[(i, 'NN')]['dir25'] != idx[(i, 'NE')]['dir25']) + int(idx[(i, 'EN')]['dir25'] != idx[(i, 'EE')]['dir25'])
  L_ += ['', f'Within-anchor, paired hazards ({n_pairs} anchors): swapping ONLY the first torque changes the 25-step direction in '
             f'{flips_first / (2 * n_pairs):.2f} of the pairs; swapping ONLY the continuation changes it in {flips_cont / (2 * n_pairs):.2f}.']
  (out / 'REPORT.md').write_text('\n'.join(L_) + '\n', encoding='utf-8')
  (out / 'summary.json').write_text(json.dumps({'arms': P, 'effects': eff, 'log_ratios': lr, 'n_anchors': len(anchors),
                                                'flip_rate_first': flips_first / (2 * n_pairs), 'flip_rate_continuation': flips_cont / (2 * n_pairs)},
                                               indent=1), encoding='utf-8')
  print('\n'.join(L_), flush=True)


# ------------------------------------------------------------------ walker
def walker(args):
  rng = np.random.default_rng(args.seed + 1)
  with np.load(B.DATASET, allow_pickle=False) as d:
    obs, lengths = d['obs'], d['lengths']
  with np.load(B.SIDECAR, allow_pickle=True) as sc:
    route_rec = sc['route_realized'].astype(str)
  x, y = obs[:, :, 0], obs[:, :, 1]
  det = np.flatnonzero(route_rec == 'detour')
  north = [(int(e), int(t)) for e in det for t in range(1, int(lengths[e]) - 1) if x[e, t] < 2.0 and 2.0 < y[e, t] < 7.5]
  east = [(int(e), int(t)) for e in det for t in range(1, int(lengths[e]) - 1) if y[e, t] >= 7.5 and 2.0 < x[e, t] < 22.0]
  jobs = []
  for pool_, leg in ((north, 'north'), (east, 'east')):
    sel = rng.choice(len(pool_), size=min(args.n_placed, len(pool_)), replace=False)
    for k, j in enumerate(sel):
      e, t = pool_[j]
      pair_seed = DIAG_SEED + 20_000 + (0 if leg == 'north' else 1000) + int(k)
      for w in ('driver', 'bc'):
        jobs.append((int(k), e, t, leg, w, pair_seed))
  print(f'walker: {len(jobs) // 2} placed rows x 2 walkers, {args.workers} workers', flush=True)
  t0 = time.time()
  res = run(_run_walker_jobs, jobs, args.workers, extra=(args.bc_ckpt,))
  print(f'rollouts done in {time.time() - t0:.0f}s', flush=True)
  out = B.OUT / 'diag_walker'
  out.mkdir(parents=True, exist_ok=True)
  (out / 'rollouts.json').write_text(json.dumps(res), encoding='utf-8')
  L_ = ['# Detour-corridor completion: generation walker vs deployment BC walker (paired hazards, same placed rows)', '',
        f'BC actor: {args.bc_ckpt}', '',
        '| leg | walker | n | reach | death | timeout | reach step (median) | P_goal |', '|---|---|---:|---:|---:|---:|---:|---:|']
  summ = {}
  for leg in ('north', 'east'):
    for w in ('driver', 'bc'):
      rs = [r for r in res if r['leg'] == leg and r['walker'] == w]
      if not rs:
        continue
      s = {'n': len(rs), 'reach': float(np.mean([r['success'] for r in rs])), 'death': float(np.mean([r['failure'] for r in rs])),
           'timeout': float(np.mean([(not r['success']) and (not r['failure']) for r in rs])),
           'reach_step_median': (float(np.median([r['reach_step'] for r in rs if r['reach_step'] is not None]))
                                 if any(r['reach_step'] is not None for r in rs) else None),
           'p_goal': float(np.mean([r['p_goal'] for r in rs]))}
      summ[f'{leg}/{w}'] = s
      L_.append(f'| {leg} | {"generation (frozen walker + blind driver)" if w == "driver" else "deployment (BC mode)"} | {s["n"]} | '
                f'{s["reach"]:.2f} | {s["death"]:.2f} | {s["timeout"]:.2f} | {s["reach_step_median"]} | {s["p_goal"]:.3f} |')
  (out / 'REPORT.md').write_text('\n'.join(L_) + '\n', encoding='utf-8')
  (out / 'summary.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')
  print('\n'.join(L_), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('crossover', 'walker'))
  ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n-anchors', type=int, default=300)
  ap.add_argument('--t-max', type=int, default=5)
  ap.add_argument('--n-placed', type=int, default=100)
  ap.add_argument('--bc-ckpt', default='')
  args = ap.parse_args(argv)
  if args.mode == 'crossover':
    crossover(args)
  else:
    if not args.bc_ckpt:
      ap.error('walker mode needs --bc-ckpt')
    walker(args)


if __name__ == '__main__':
  main()
