"""Environment, physics, teacher, and timing audit for rockfall-clock V6.

This is deliberately separate from every V5 gate.  It validates the new
rectangular two-hazard environment before an expensive dataset collection or
baseline run:

* registration, geometry, goal, dimensions, and Ant-only observations;
* disjoint free-joint rock sets and independently advancing absolute clocks;
* reset reproducibility and empirical independence of U1/U2 and t0_1/t0_2;
* paired same-action active/clear rollouts remain learner-identical until a
  physical rock is close enough to influence the Ant;
* physical behavior under forced U00/U10/U01/U11 and simultaneous falls;
* sighted shortcut and forced-detour behavior for every latent combination;
* forced-go lethality of each hazard in isolation;
* clean shortcut mouth/entry/crossing timing and episode-length headroom.

The learner never receives an audit value: hidden state is read only through
the environment's explicit privileged interface and step ``info``.  All
results, including failed checks, are written to JSON before the process exits.

Run a quick integration pass while developing::

    python scripts/rockfall_clock_v6_audit.py --quick

Run the default evidence pass::

    python scripts/rockfall_clock_v6_audit.py
"""
import argparse
import importlib
import json
import os
import sys

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod                         # noqa: E402
from crl import rockfall_clock_v6 as V6                  # noqa: E402
from crl.rockfall_ant import NQ_ANT, NV_ANT              # noqa: E402


OUT = 'artifacts/rockfall_clock_v6/timing_audit.json'
LATENTS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _env_kwargs(params):
  return {
      'p_active_1': params['p_active_1'],
      'p_active_2': params['p_active_2'],
      't0_min_1': params['t0_range_1'][0],
      't0_max_1': params['t0_range_1'][1],
      't0_min_2': params['t0_range_2'][0],
      't0_max_2': params['t0_range_2'][1],
  }


def _make_teacher(teacher_module, horizon, params):
  return teacher_module.make_teacher(
      horizon=horizon,
      p_active_1=params['p_active_1'],
      p_active_2=params['p_active_2'],
      t0_range_1=params['t0_range_1'],
      t0_range_2=params['t0_range_2'])


def _load_teacher_module():
  """Load the sibling teacher lazily so this file remains importable alone."""
  try:
    module = importlib.import_module('rockfall_clock_v6_teacher')
  except ModuleNotFoundError as exc:
    raise RuntimeError(
        'scripts/rockfall_clock_v6_teacher.py is required to run the V6 '
        'teacher/timing audit') from exc
  missing = [name for name in ('ENV_NAME', 'HORIZON', 'CROSSING_STEPS',
                               'SAFETY_MARGIN', 'make_teacher',
                               'teacher_episode')
             if not hasattr(module, name)]
  if missing:
    raise RuntimeError(f'V6 teacher interface is missing {missing}')
  return module


def _describe(values):
  values = np.asarray([v for v in values if v is not None], np.float64)
  if not len(values):
    return {'n': 0, 'mean': None, 'std': None, 'min': None, 'max': None,
            'median': None, 'p05': None, 'p95': None}
  return {
      'n': int(len(values)),
      'mean': round(float(np.mean(values)), 3),
      'std': round(float(np.std(values)), 3),
      'min': round(float(np.min(values)), 3),
      'max': round(float(np.max(values)), 3),
      'median': round(float(np.median(values)), 3),
      'p05': round(float(np.quantile(values, 0.05)), 3),
      'p95': round(float(np.quantile(values, 0.95)), 3),
  }


def _rate(rows, key):
  return (round(float(np.mean([bool(row[key]) for row in rows])), 4)
          if rows else None)


def _corr(a, b):
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
    return None
  return round(float(np.corrcoef(a, b)[0, 1]), 6)


def _snapshot(env):
  return {
      'u1': bool(env.privileged_rockfall_active_1),
      'u2': bool(env.privileged_rockfall_active_2),
      't0_1': int(env.privileged_sampled_start(1)),
      't0_2': int(env.privileged_sampled_start(2)),
  }


def _latent_name(u1, u2):
  return f'U{int(bool(u1))}{int(bool(u2))}'


class Checks:
  def __init__(self):
    self.rows = []

  def add(self, name, passed, detail=None):
    row = {'name': str(name), 'passed': bool(passed)}
    if detail is not None:
      row['detail'] = detail
    self.rows.append(row)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}", flush=True)

  @property
  def passed(self):
    return all(row['passed'] for row in self.rows)


def _make_registered_env(teacher_module, cfg, seed):
  env = envs_mod.make_env(teacher_module.ENV_NAME, cfg, seed=seed)
  if not isinstance(env, V6.RockfallClockV6Env):
    raise TypeError(f'{teacher_module.ENV_NAME} made {type(env).__name__}, '
                    'not RockfallClockV6Env')
  return env


def structural_audit(teacher_module, horizon, seed, params, checks):
  """Registration, observation-hiddenness, rocks, map, and goal checks."""
  cfg, _ = _make_teacher(teacher_module, horizon, params)
  env = _make_registered_env(teacher_module, cfg, seed)
  obs = env.reset(rockfall_active_1=False, rockfall_active_2=False,
                  rockfall_start_1=V6.T0_MIN_1,
                  rockfall_start_2=V6.T0_MIN_2)
  model, data = env._env.model, env._env.data
  expected_nq = NQ_ANT + 2 * V6.ROCKS_PER_ZONE * 7
  expected_nv = NV_ANT + 2 * V6.ROCKS_PER_ZONE * 6
  sets_disjoint = (
      set(env._rock_qadr[1]).isdisjoint(env._rock_qadr[2])
      and set(env._rock_vadr[1]).isdisjoint(env._rock_vadr[2])
      and env._rock_gids[1].isdisjoint(env._rock_gids[2]))
  rock_names = {
      str(zone): [
          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
          for gid in sorted(env._rock_gids[zone])]
      for zone in (1, 2)
  }
  parked_x = {
      str(zone): [round(float(data.qpos[qadr]), 6)
                  for qadr in env._rock_qadr[zone]]
      for zone in (1, 2)
  }
  goal_xy = obs[env.obs_dim:env.obs_dim + 2]
  detail = {
      'env_name': teacher_module.ENV_NAME,
      'environment_version': V6.ENV_VERSION,
      'map_cells': [V6.MAP_ROWS, V6.MAP_COLS],
      'geometry': V6.geometry_metadata(),
      'flat_observation_shape': list(obs.shape),
      'obs_dim': int(env.obs_dim), 'goal_dim': int(env.goal_dim),
      'action_dim': int(env.action_dim),
      'model_nq': int(model.nq), 'model_nv': int(model.nv),
      'model_nu': int(model.nu),
      'expected_nq': expected_nq, 'expected_nv': expected_nv,
      'rock_geom_names': rock_names, 'parked_x': parked_x,
      'goal_cell_xy': [round(float(v), 6)
                       for v in env._eval_goal_cell_xy],
      'goal_xy': [round(float(v), 6) for v in goal_xy],
      'goal_sampler_bounds': {'x': [24.0, 25.5], 'y': [0.0, 1.5]},
      'schedule_at_reset': env.schedule,
  }
  structural_ok = (
      obs.shape == (58,) and env.obs_dim == 29 and env.goal_dim == 29
      and env.action_dim == 8 and model.nq == expected_nq
      and model.nv == expected_nv and model.nu == 8
      and len(env._rock_qadr[1]) == V6.ROCKS_PER_ZONE
      and len(env._rock_qadr[2]) == V6.ROCKS_PER_ZONE
      and sets_disjoint and V6.MAP_ROWS == 5 and V6.MAP_COLS == 9
      and np.array_equal(np.asarray(env._eval_goal_cell_xy),
                         np.array(V6.GOAL_CELL_XY))
      and 24.0 <= float(goal_xy[0]) <= 25.5
      and 0.0 <= float(goal_xy[1]) <= 1.5)
  checks.add('registration/map/model/goal contract', structural_ok, detail)

  # Four separate, identically seeded worlds differ only in privileged reset
  # values.  If a latent, timer, or rock DOF leaked, these observations would
  # differ or grow beyond the 29-Ant + 29-goal contract.
  paired = []
  forced_starts = ((10, 90), (45, 125), (17, 111), (33, 101))
  for (u1, u2), (t01, t02) in zip(LATENTS, forced_starts):
    penv = V6.RockfallClockV6Env(
        max_episode_steps=horizon, seed=seed + 1, eval_goal_mode='fixed',
        **_env_kwargs(params))
    po = penv.reset(rockfall_active_1=u1, rockfall_active_2=u2,
                    rockfall_start_1=t01, rockfall_start_2=t02)
    paired.append({'latent': _latent_name(u1, u2), 'obs': po,
                   'schedule': penv.schedule})
  max_diff = max(float(np.max(np.abs(row['obs'] - paired[0]['obs'])))
                 for row in paired[1:])
  hidden_detail = {
      'max_initial_obs_difference_across_forced_latents_and_clocks': max_diff,
      'flat_width': int(paired[0]['obs'].shape[0]),
      'schedules': [row['schedule'] for row in paired],
  }
  checks.add('U1/U2/t0/rock state hidden from learner observation',
             max_diff == 0.0 and paired[0]['obs'].shape == (58,),
             hidden_detail)
  return detail, hidden_detail


def independence_audit(n, seed, params, checks):
  """Empirical independence plus exact same-seed reproducibility checks."""
  env = V6.RockfallClockV6Env(seed=seed, eval_goal_mode='fixed',
                              **_env_kwargs(params))
  rows = []
  for _ in range(n):
    env.reset()
    rows.append(_snapshot(env))
  u1 = np.array([row['u1'] for row in rows], np.float64)
  u2 = np.array([row['u2'] for row in rows], np.float64)
  t01 = np.array([row['t0_1'] for row in rows], np.float64)
  t02 = np.array([row['t0_2'] for row in rows], np.float64)
  counts = {_latent_name(a, b): int(sum(
      row['u1'] == bool(a) and row['u2'] == bool(b) for row in rows))
            for a, b in LATENTS}
  p1, p2 = params['p_active_1'], params['p_active_2']
  expected = {'U00': (1 - p1) * (1 - p2),
              'U10': p1 * (1 - p2),
              'U01': (1 - p1) * p2,
              'U11': p1 * p2}
  observed = {key: round(value / n, 6) for key, value in counts.items()}
  # Four-sigma binomial tolerance, with a small finite-sample floor.
  tolerances = {
      key: max(0.02, 4.0 * np.sqrt(prob * (1.0 - prob) / n))
      for key, prob in expected.items()}
  joint_ok = all(abs(observed[key] - expected[key]) <= tolerances[key]
                 for key in expected)
  corr_limit = max(0.08, 4.0 / np.sqrt(n))
  corr_u = _corr(u1, u2)
  corr_t0 = _corr(t01, t02)
  ranges_ok = (
      np.all((params['t0_range_1'][0] <= t01)
             & (t01 <= params['t0_range_1'][1]))
      and np.all((params['t0_range_2'][0] <= t02)
                 & (t02 <= params['t0_range_2'][1])))

  env_a = V6.RockfallClockV6Env(seed=seed + 1, eval_goal_mode='fixed',
                                **_env_kwargs(params))
  env_b = V6.RockfallClockV6Env(seed=seed + 1, eval_goal_mode='fixed',
                                **_env_kwargs(params))
  seq_a, seq_b = [], []
  for _ in range(50):
    env_a.reset()
    env_b.reset()
    seq_a.append(_snapshot(env_a))
    seq_b.append(_snapshot(env_b))
  reproducible = seq_a == seq_b

  # Overrides must still consume every stream.  After equal reset counts the
  # next natural episode must align with an all-natural twin.
  env_n = V6.RockfallClockV6Env(seed=seed + 2, eval_goal_mode='fixed',
                                **_env_kwargs(params))
  env_f = V6.RockfallClockV6Env(seed=seed + 2, eval_goal_mode='fixed',
                                **_env_kwargs(params))
  for idx in range(30):
    env_n.reset()
    env_f.reset(rockfall_active_1=idx % 2,
                rockfall_active_2=(idx // 2) % 2,
                rockfall_start_1=-100 - idx,
                rockfall_start_2=1000 + idx)
  obs_n = env_n.reset()
  obs_f = env_f.reset()
  override_stream_safe = (_snapshot(env_n) == _snapshot(env_f)
                          and np.array_equal(obs_n, obs_f))

  detail = {
      'n': int(n), 'counts': counts, 'observed_frequencies': observed,
      'expected_frequencies': {k: round(v, 6) for k, v in expected.items()},
      'frequency_tolerances': {k: round(float(v), 6)
                               for k, v in tolerances.items()},
      'p_active_1_observed': round(float(np.mean(u1)), 6),
      'p_active_2_observed': round(float(np.mean(u2)), 6),
      'corr_u1_u2': corr_u,
      't0_1': _describe(t01), 't0_2': _describe(t02),
      'corr_t0_1_t0_2': corr_t0, 'correlation_limit': round(corr_limit, 6),
      'same_seed_sequences_identical': reproducible,
      'override_consumption_preserves_next_natural_episode':
          override_stream_safe,
  }
  passed = (joint_ok and ranges_ok and reproducible and override_stream_safe
            and corr_u is not None and abs(corr_u) <= corr_limit
            and corr_t0 is not None and abs(corr_t0) <= corr_limit)
  checks.add('independent/reproducible latent and schedule streams', passed,
             detail)
  return detail


def _stationary_schedule(combo, starts, seed, params):
  u1, u2 = combo
  env = V6.RockfallClockV6Env(seed=seed, eval_goal_mode='fixed',
                              **_env_kwargs(params))
  env.reset(rockfall_active_1=u1, rockfall_active_2=u2,
            rockfall_start_1=starts[1], rockfall_start_2=starts[2])
  stop = max(starts.values()) + V6.ROCKFALL_STEPS + 2
  first_wave = {1: [], 2: []}
  previous_waves = {1: 0, 2: 0}
  info = env._info(False)
  for step in range(stop):
    _, _, done, info = env.step(np.zeros(8, np.float32))
    if done:
      break
    for zone in (1, 2):
      waves = int(info[f'rock_waves_{zone}'])
      if waves > previous_waves[zone]:
        first_wave[zone].append(step)
        previous_waves[zone] = waves
  return {
      'latent': _latent_name(u1, u2),
      'forced_starts': {str(k): int(v) for k, v in starts.items()},
      'wave_steps': {str(k): v for k, v in first_wave.items()},
      'waves': {str(z): int(info[f'rock_waves_{z}']) for z in (1, 2)},
      'passed': {str(z): bool(info[f'rockfall_passed_{z}']) for z in (1, 2)},
      'dropped_at_end': {
          str(z): bool(info[f'rock_dropped_{z}']) for z in (1, 2)},
      'failure': bool(info['failure']), 'steps_run': int(step + 1),
  }


def physics_schedule_audit(seed, params, checks):
  """Force all combinations and prove both body sets can coexist in flight."""
  starts = {1: params['t0_range_1'][0], 2: params['t0_range_2'][0]}
  rows = [_stationary_schedule(combo, starts, seed + idx, params)
          for idx, combo in enumerate(LATENTS)]
  expected_wave_steps = {
      str(zone): [starts[zone] + k * V6.WAVE_PERIOD for k in range(6)]
      for zone in (1, 2)}
  mechanics_ok = True
  for row, combo in zip(rows, LATENTS):
    for zone, active in zip((1, 2), combo):
      expected_n = 6 if active else 0
      mechanics_ok &= row['waves'][str(zone)] == expected_n
      mechanics_ok &= row['wave_steps'][str(zone)] == (
          expected_wave_steps[str(zone)] if active else [])
      mechanics_ok &= row['passed'][str(zone)] == bool(active)
      mechanics_ok &= not row['dropped_at_end'][str(zone)]
    mechanics_ok &= not row['failure']

  env = V6.RockfallClockV6Env(seed=seed + 100, eval_goal_mode='fixed',
                              **_env_kwargs(params))
  env.reset(rockfall_active_1=True, rockfall_active_2=True,
            rockfall_start_1=0, rockfall_start_2=6)
  info = {}
  for _ in range(7):
    _, _, _, info = env.step(np.zeros(8, np.float32))
  x_positions = {
      str(zone): [round(float(env._env.data.qpos[qadr]), 6)
                  for qadr in env._rock_qadr[zone]]
      for zone in (1, 2)}
  simultaneous = bool(info['rock_dropped_1'] and info['rock_dropped_2'])
  separate_footprints = (
      max(x_positions['1']) < min(x_positions['2'])
      and all(V6.AIM_X[1][0] <= x <= V6.ROCK_X_MAX[1]
              for x in x_positions['1'])
      and all(V6.AIM_X[2][0] <= x <= V6.ROCK_X_MAX[2]
              for x in x_positions['2']))
  disjoint_addresses = (
      set(env._rock_qadr[1]).isdisjoint(env._rock_qadr[2])
      and set(env._rock_vadr[1]).isdisjoint(env._rock_vadr[2])
      and env._rock_gids[1].isdisjoint(env._rock_gids[2]))
  detail = {
      'forced_combinations': rows,
      'expected_wave_steps': expected_wave_steps,
      'simultaneous_probe': {
          'forced_starts': {'1': 0, '2': 6},
          'both_sets_dropped': simultaneous,
          'x_positions': x_positions,
          'separate_footprints': separate_footprints,
          'disjoint_qpos_qvel_geom_addresses': disjoint_addresses,
      },
  }
  checks.add('forced U00/U10/U01/U11 clock and wave mechanics',
             mechanics_ok, detail['forced_combinations'])
  checks.add('simultaneous hazards use separate physical rock bodies',
             simultaneous and separate_footprints and disjoint_addresses,
             detail['simultaneous_probe'])
  return detail


def paired_hiddenness_audit(teacher_module, horizon, seed, params, checks):
  """V5-style same-action hiddenness check for each isolated rock system.

  Each pair has the same environment seed, Ant reset, fixed goal, sampled
  jitter, and forced timetable.  The action is computed once from the clear
  twin by the deterministic shortcut controller and replayed verbatim into the
  active twin.  Thus an observation difference before ``rock_within_reach``
  cannot be attributed to different behavior.

  MuJoCo's global constraint solve can create minute floating-point changes
  when remote rocks leave/rejoin the floor (the V5 audit documents this), so
  the scientific gate is a 1e-6 learner-observation tolerance rather than
  bit-exact equality.  Exact divergence is still recorded as a diagnostic.
  """
  _, teacher = _make_teacher(teacher_module, horizon, params)
  pairs = {}
  all_ok = True
  for zone in (1, 2):
    pair_seed = int(seed + zone - 1)
    starts = {1: int(params['t0_range_1'][0]),
              2: int(params['t0_range_2'][0])}
    active_bits = {1: zone == 1, 2: zone == 2}
    active_env = V6.RockfallClockV6Env(
        max_episode_steps=horizon, seed=pair_seed, eval_goal_mode='fixed',
        **_env_kwargs(params))
    clear_env = V6.RockfallClockV6Env(
        max_episode_steps=horizon, seed=pair_seed, eval_goal_mode='fixed',
        **_env_kwargs(params))
    active_obs = active_env.reset(
        rockfall_active_1=active_bits[1],
        rockfall_active_2=active_bits[2],
        rockfall_start_1=starts[1], rockfall_start_2=starts[2])
    clear_obs = clear_env.reset(
        rockfall_active_1=False, rockfall_active_2=False,
        rockfall_start_1=starts[1], rockfall_start_2=starts[2])
    teacher.fresh(route='shortcut')

    initial_diff = float(np.max(np.abs(active_obs - clear_obs)))
    max_pre_reach_diff = initial_diff
    max_until_contact_diff = initial_diff
    first_exact_divergence = None
    first_over_tolerance = None
    first_drop_clock = None
    first_within_reach_step = None
    first_flagged_contact_step = None
    final_info = {}
    steps_run = 0
    for _ in range(horizon):
      # Generate ONE action from the clear twin and replay it in both worlds.
      pre_step_clock = int(clear_env.schedule['t'])
      action = teacher.act(clear_obs, clear_env.schedule, intent='go')
      active_next, _, active_done, active_info = active_env.step(action)
      clear_next, clear_reward, _, _ = clear_env.step(action)
      steps_run += 1
      final_info = active_info

      diff = float(np.max(np.abs(active_next - clear_next)))
      max_until_contact_diff = max(max_until_contact_diff, diff)
      post_step = int(active_info['t'])
      if diff > 0.0 and first_exact_divergence is None:
        first_exact_divergence = post_step
      if diff > 1e-6 and first_over_tolerance is None:
        first_over_tolerance = post_step
      if (first_drop_clock is None
          and int(active_info[f'rock_waves_{zone}']) > 0):
        first_drop_clock = pre_step_clock

      within_reach = active_env.rock_within_reach(zone)
      if within_reach and first_within_reach_step is None:
        first_within_reach_step = post_step
      if first_within_reach_step is None:
        max_pre_reach_diff = max(max_pre_reach_diff, diff)
      if (bool(active_info[f'rock_contact_{zone}'])
          and first_flagged_contact_step is None):
        first_flagged_contact_step = post_step

      active_obs, clear_obs = active_next, clear_next
      if active_done or clear_reward > 0:
        break

    pair_ok = (
        initial_diff == 0.0
        and active_obs.shape == clear_obs.shape == (58,)
        and first_drop_clock == starts[zone]
        and first_within_reach_step is not None
        and first_flagged_contact_step is not None
        and first_within_reach_step <= first_flagged_contact_step
        and max_pre_reach_diff <= 1e-6
        and (first_over_tolerance is None
             or first_over_tolerance >= first_within_reach_step)
        and bool(final_info.get('failure'))
        and final_info.get('failure_zone') == zone)
    all_ok &= pair_ok
    pairs[str(zone)] = {
        'zone': zone,
        'pair_seed': pair_seed,
        'active_latent': _latent_name(active_bits[1], active_bits[2]),
        'forced_t0_1': starts[1], 'forced_t0_2': starts[2],
        'learner_observation_shape': list(active_obs.shape),
        'initial_max_abs_difference': initial_diff,
        'first_drop_pre_step_clock': first_drop_clock,
        'first_within_reach_post_step': first_within_reach_step,
        'first_flagged_contact_post_step': first_flagged_contact_step,
        'first_exact_observation_divergence_post_step':
            first_exact_divergence,
        'first_observation_divergence_over_1e-6_post_step':
            first_over_tolerance,
        'max_abs_observation_difference_before_within_reach':
            max_pre_reach_diff,
        'max_abs_observation_difference_through_contact':
            max_until_contact_diff,
        'steps_run': steps_run,
        'failure': bool(final_info.get('failure')),
        'failure_zone': final_info.get('failure_zone'),
        'passed': bool(pair_ok),
    }

  detail = {
      'protocol': ('same seed/reset/fixed goal and identical action replay; '
                   'action generated only from clear twin'),
      'pre_within_reach_tolerance': 1e-6,
      'isolated_pairs': pairs,
  }
  checks.add('paired same-action observations stay hidden before rock reach',
             all_ok, detail)
  return detail


def _trace_timing(trace):
  out = {}
  for zone in (1, 2):
    x0, x1 = V6.HAZARD_X[zone]
    mouth = next((row['step'] for row in trace
                  if V6.MOUTH_X[zone] <= row['x'] < x0
                  and abs(row['y']) < V6.HAZARD_HALF_Y), None)
    entry = next((row['step'] for row in trace
                  if x0 <= row['x'] <= x1
                  and abs(row['y']) < V6.HAZARD_HALF_Y), None)
    exit_step = None
    if entry is not None:
      exit_step = next((row['step'] for row in trace
                        if row['step'] > entry and row['x'] > x1
                        and abs(row['y']) < V6.HAZARD_HALF_Y), None)
    out[zone] = {
        'mouth_step_measured': mouth,
        'entry_step_measured': entry,
        'exit_step_measured': exit_step,
        'mouth_to_entry_steps': (entry - mouth
                                 if mouth is not None and entry is not None
                                 else None),
        'crossing_steps': (exit_step - entry
                           if entry is not None and exit_step is not None
                           else None),
        'mouth_to_clear_steps': (exit_step - mouth
                                 if mouth is not None and exit_step is not None
                                 else None),
    }
  return out


def _teacher_rollout(teacher_module, env, teacher, hazards, route,
                     horizon, intent=None, starts=None):
  trace = []
  last_info = {}

  def on_step(obs, action, next_obs, reward, done, info):
    nonlocal last_info
    if not trace:
      trace.append({'step': 0, 'x': float(obs[0]), 'y': float(obs[1])})
    trace.append({'step': len(trace), 'x': float(next_obs[0]),
                  'y': float(next_obs[1]),
                  'action_norm': float(np.linalg.norm(action)),
                  'reward': float(reward), 'done': bool(done)})
    last_info = dict(info)

  row = teacher_module.teacher_episode(
      env, teacher, hazards={1: bool(hazards[0]), 2: bool(hazards[1])},
      route=route, intent=intent, starts=starts, horizon=horizon,
      on_step=on_step)
  timing = _trace_timing(trace)
  row['timing'] = {str(zone): timing[zone] for zone in (1, 2)}
  row['entered_zone_1'] = bool(last_info.get('entered_hazard_1'))
  row['entered_zone_2'] = bool(last_info.get('entered_hazard_2'))
  row['passed_zone_1'] = bool(last_info.get('rockfall_passed_1'))
  row['passed_zone_2'] = bool(last_info.get('rockfall_passed_2'))
  row['dropped_zone_1_at_end'] = bool(last_info.get('rock_dropped_1'))
  row['dropped_zone_2_at_end'] = bool(last_info.get('rock_dropped_2'))
  return row


def _outcome_summary(rows, horizon):
  successes = [row['steps'] for row in rows if row['success']]
  return {
      'n': len(rows), 'success': _rate(rows, 'success'),
      'failure': _rate(rows, 'failure'), 'timeout': _rate(rows, 'timeout'),
      'steps': _describe([row['steps'] for row in rows]),
      'successful_episode_steps': _describe(successes),
      'horizon': int(horizon),
      'headroom_from_longest_episode': (
          int(horizon - max(row['steps'] for row in rows)) if rows else None),
      'headroom_from_longest_success': (
          int(horizon - max(successes)) if successes else None),
      'failure_zone_1': int(sum(row.get('failure_zone') == 1 for row in rows)),
      'failure_zone_2': int(sum(row.get('failure_zone') == 2 for row in rows)),
  }


def teacher_timing_audit(teacher_module, horizon, n_clean,
                         n_teacher_per_combo, n_detour_per_combo,
                         n_forced_go, seed, params, checks):
  """Exercise the public teacher API and derive clean trajectory timings."""
  cfg, teacher = _make_teacher(teacher_module, horizon, params)
  env = _make_registered_env(teacher_module, cfg, seed)

  clean_rows = [
      _teacher_rollout(teacher_module, env, teacher, (0, 0), 'shortcut',
                       horizon)
      for _ in range(n_clean)]
  timing = {}
  for zone in (1, 2):
    zrows = [row['timing'][str(zone)] for row in clean_rows]
    timing[str(zone)] = {
        'mouth_arrival_steps': _describe(
            [row['mouth_step_measured'] for row in zrows]),
        'band_arrival_steps': _describe(
            [row['entry_step_measured'] for row in zrows]),
        'band_exit_steps': _describe(
            [row['exit_step_measured'] for row in zrows]),
        'mouth_to_entry_steps': _describe(
            [row['mouth_to_entry_steps'] for row in zrows]),
        'crossing_steps': _describe(
            [row['crossing_steps'] for row in zrows]),
        'mouth_to_clear_steps': _describe(
            [row['mouth_to_clear_steps'] for row in zrows]),
        'teacher_config_crossing_steps': int(
            teacher_module.CROSSING_STEPS[zone]),
        'teacher_safety_margin': int(teacher_module.SAFETY_MARGIN),
        't0_range': list(env.t0_ranges[zone]),
        'unsafe_window_end_range': [
            int(env.t0_ranges[zone][0] + V6.ROCKFALL_STEPS),
            int(env.t0_ranges[zone][1] + V6.ROCKFALL_STEPS)],
    }
  clean_summary = _outcome_summary(clean_rows, horizon)
  clean_complete = all(row['success'] and not row['failure']
                       and row['entered_zone_1'] and row['entered_zone_2']
                       and row['timing']['1']['crossing_steps'] is not None
                       and row['timing']['2']['crossing_steps'] is not None
                       for row in clean_rows)
  checks.add('clean shortcut succeeds and traverses both zones',
             clean_complete and clean_summary['success'] >= 0.90,
             {'outcomes': clean_summary, 'timing': timing})

  sighted = {}
  all_sighted = []
  for combo_idx, combo in enumerate(LATENTS):
    key = _latent_name(*combo)
    rows = [
        _teacher_rollout(teacher_module, env, teacher, combo, 'shortcut',
                         horizon)
        for _ in range(n_teacher_per_combo)]
    sighted[key] = {'outcomes': _outcome_summary(rows, horizon),
                    'wait_zone_1': _rate(rows, 'wait_zone_1'),
                    'wait_zone_2': _rate(rows, 'wait_zone_2'),
                    'rows': rows}
    all_sighted.extend(rows)
  sighted_summary = _outcome_summary(all_sighted, horizon)
  sighted_ok = (sighted_summary['success'] >= 0.90
                and sighted_summary['failure'] == 0.0
                and sighted_summary['timeout'] <= 0.05)
  checks.add('sighted shortcut teacher handles all four latent combinations',
             sighted_ok,
             {'overall': sighted_summary,
              'by_latent': {key: value['outcomes']
                            for key, value in sighted.items()}})

  detour = {}
  all_detour = []
  for combo in LATENTS:
    key = _latent_name(*combo)
    rows = [
        _teacher_rollout(teacher_module, env, teacher, combo, 'detour',
                         horizon, intent='detour')
        for _ in range(n_detour_per_combo)]
    detour[key] = {'outcomes': _outcome_summary(rows, horizon), 'rows': rows}
    all_detour.extend(rows)
  detour_summary = _outcome_summary(all_detour, horizon)
  detour_safe = all(
      not row['failure'] and not row['entered_zone_1']
      and not row['entered_zone_2']
      and row['route_realized'] != 'shortcut'
      and not row['wait_zone_1'] and not row['wait_zone_2']
      for row in all_detour)
  checks.add('forced detour bypasses both hazards under U00/U10/U01/U11',
             detour_safe and detour_summary['success'] >= 0.90
             and detour_summary['timeout'] <= 0.10,
             {'overall': detour_summary,
              'by_latent': {key: value['outcomes']
                            for key, value in detour.items()}})

  forced_go = {}
  for zone, combo in ((1, (1, 0)), (2, (0, 1))):
    rows = [
        _teacher_rollout(teacher_module, env, teacher, combo, 'shortcut',
                         horizon, intent='go')
        for _ in range(n_forced_go)]
    forced_go[str(zone)] = {'latent': _latent_name(*combo),
                            'outcomes': _outcome_summary(rows, horizon),
                            'rows': rows}
  zone1_real = (forced_go['1']['outcomes']['failure_zone_1'] > 0
                and forced_go['1']['outcomes']['failure_zone_2'] == 0)
  zone2_real = (forced_go['2']['outcomes']['failure_zone_2'] > 0
                and forced_go['2']['outcomes']['failure_zone_1'] == 0)
  checks.add('each isolated active hazard is physically dangerous to do(go)',
             zone1_real and zone2_real,
             {zone: value['outcomes'] for zone, value in forced_go.items()})

  categories = {
      'clean_shortcut_U00': clean_summary,
      'sighted_shortcut_U10': sighted['U10']['outcomes'],
      'sighted_shortcut_U01': sighted['U01']['outcomes'],
      'sighted_shortcut_U11': sighted['U11']['outcomes'],
      'safe_detour_all_latents': detour_summary,
      'sighted_all_latents': sighted_summary,
  }
  legitimate = clean_rows + all_sighted + all_detour
  timeout_rate = _rate(legitimate, 'timeout')
  max_steps = max(row['steps'] for row in legitimate)
  successful_steps = [row['steps'] for row in legitimate if row['success']]
  horizon_detail = {
      'configured_horizon': int(horizon),
      'legitimate_episode_count': len(legitimate),
      'timeout_rate': timeout_rate,
      'max_observed_steps': int(max_steps),
      'minimum_observed_headroom': int(horizon - max_steps),
      'max_successful_episode_steps': int(max(successful_steps)),
      'minimum_successful_headroom': int(horizon - max(successful_steps)),
      'categories': categories,
  }
  checks.add('horizon leaves legitimate teacher behavior timeout-safe',
             timeout_rate <= 0.05 and sighted_summary['success'] >= 0.90
             and detour_summary['success'] >= 0.90,
             horizon_detail)
  return {
      'walker_controller': dict(getattr(
          teacher, 'walker_provenance', {})),
      'clean_shortcut_timing': timing,
      'clean_shortcut': {'summary': clean_summary, 'rows': clean_rows},
      'sighted_shortcut_by_latent': sighted,
      'forced_detour_by_latent': detour,
      'forced_go_isolated_hazards': forced_go,
      'horizon_analysis': horizon_detail,
  }


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=701)
  parser.add_argument('--horizon', type=int, default=None)
  parser.add_argument('--p-active-1', type=float, default=V6.P_ACTIVE_1)
  parser.add_argument('--p-active-2', type=float, default=V6.P_ACTIVE_2)
  parser.add_argument('--t0-min-1', type=int, default=V6.T0_MIN_1)
  parser.add_argument('--t0-max-1', type=int, default=V6.T0_MAX_1)
  parser.add_argument('--t0-min-2', type=int, default=V6.T0_MIN_2)
  parser.add_argument('--t0-max-2', type=int, default=V6.T0_MAX_2)
  parser.add_argument('--n-independence', type=int, default=2000)
  parser.add_argument('--n-clean', type=int, default=30)
  parser.add_argument('--n-teacher-per-combo', type=int, default=20)
  parser.add_argument('--n-detour-per-combo', type=int, default=10)
  parser.add_argument('--n-forced-go', type=int, default=12)
  parser.add_argument('--quick', action='store_true')
  parser.add_argument('--out', default=OUT)
  return parser.parse_args()


def main():
  args = parse_args()
  teacher_module = _load_teacher_module()
  horizon = int(teacher_module.HORIZON if args.horizon is None
                else args.horizon)
  params = {
      'p_active_1': float(args.p_active_1),
      'p_active_2': float(args.p_active_2),
      't0_range_1': (int(args.t0_min_1), int(args.t0_max_1)),
      't0_range_2': (int(args.t0_min_2), int(args.t0_max_2)),
  }
  if args.quick:
    args.n_independence = min(args.n_independence, 300)
    args.n_clean = min(args.n_clean, 3)
    args.n_teacher_per_combo = min(args.n_teacher_per_combo, 2)
    args.n_detour_per_combo = min(args.n_detour_per_combo, 1)
    args.n_forced_go = min(args.n_forced_go, 3)
  counts = {
      'independence_resets': int(args.n_independence),
      'clean_shortcut_episodes': int(args.n_clean),
      'sighted_shortcut_per_latent': int(args.n_teacher_per_combo),
      'forced_detour_per_latent': int(args.n_detour_per_combo),
      'forced_go_per_isolated_hazard': int(args.n_forced_go),
  }
  print(f'V6 environment/timing audit | seed {args.seed} | H={horizon} | '
        f'{counts}', flush=True)
  checks = Checks()
  report = {
      'audit': 'rockfall_clock_v6_environment_teacher_timing',
      'environment_version': V6.ENV_VERSION,
      'env_name': teacher_module.ENV_NAME,
      'seed': int(args.seed), 'horizon': horizon,
      'quick': bool(args.quick), 'sample_counts': counts,
      'defaults': {
          'p_active_1': params['p_active_1'],
          'p_active_2': params['p_active_2'],
          't0_range_1': list(params['t0_range_1']),
          't0_range_2': list(params['t0_range_2']),
          'rockfall_steps': V6.ROCKFALL_STEPS,
          'wave_period': V6.WAVE_PERIOD,
          'rocks_per_zone': V6.ROCKS_PER_ZONE,
          'teacher_crossing_steps': teacher_module.CROSSING_STEPS,
          'teacher_safety_margin': teacher_module.SAFETY_MARGIN,
          'teacher_detour_turn_bias': getattr(
              teacher_module, 'DETOUR_TURN_BIAS', None),
          'teacher_detour_turn_bias_steps': getattr(
              teacher_module, 'DETOUR_TURN_BIAS_STEPS', None),
      },
  }
  try:
    structural, hidden = structural_audit(
        teacher_module, horizon, args.seed, params, checks)
    report['structural'] = structural
    report['hidden_observation'] = hidden
    report['independence'] = independence_audit(
        args.n_independence, args.seed + 1000, params, checks)
    report['physics_and_schedules'] = physics_schedule_audit(
        args.seed + 2000, params, checks)
    report['paired_same_action_hiddenness'] = paired_hiddenness_audit(
        teacher_module, horizon, args.seed + 2500, params, checks)
    report['teacher_and_timing'] = teacher_timing_audit(
        teacher_module, horizon, args.n_clean, args.n_teacher_per_combo,
        args.n_detour_per_combo, args.n_forced_go, args.seed + 3000, params,
        checks)
  except Exception as exc:  # Preserve partial evidence before surfacing error.
    report['runtime_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    checks.add('audit completed without runtime error', False,
               report['runtime_error'])
  report['checks'] = checks.rows
  report['passed'] = checks.passed
  out_dir = os.path.dirname(os.path.abspath(args.out))
  os.makedirs(out_dir, exist_ok=True)
  with open(args.out, 'w') as handle:
    json.dump(report, handle, indent=2)
  print(json.dumps({
      'passed': report['passed'],
      'checks': [{'name': row['name'], 'passed': row['passed']}
                 for row in checks.rows],
  }, indent=2), flush=True)
  print('->', args.out, flush=True)
  raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
  main()
