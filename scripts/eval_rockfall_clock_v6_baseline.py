"""Authoritative natural-draw evaluation for the V6 vanilla CRL baseline.

The headline is the deterministic actor mean, identical to the repository's
standard evaluation policy.  Every episode calls ``env.reset()`` without
latent/timing overrides, so U1, U2, t0_1, and t0_2 come from the environment's
independent reset-time streams.  Privileged fields are read only after reset
for the audit breakdown and are never passed to the policy.

Example::

  python scripts/eval_rockfall_clock_v6_baseline.py \
      --ckpt artifacts/rockfall_clock_v6/runs/<run>/final.pkl --n 1000
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import checkpoint as ckpt_mod             # noqa: E402
from crl import envs as envs_mod                    # noqa: E402
from crl import networks as networks_mod            # noqa: E402
from crl import rockfall_clock_v6 as V6             # noqa: E402
import rockfall_clock_v6_teacher as CT               # noqa: E402
from verify_offline_d4rl import build_offline_cfg    # noqa: E402

OUT_ROOT = 'artifacts/rockfall_clock_v6'
ENV_BASE = 'offline_antmaze_rockfall_clock_v6'
ENV_XY = ENV_BASE + '_gxy'
ENV_NAMES = (ENV_BASE, ENV_XY)
HORIZON = int(CT.HORIZON)
assert HORIZON == 800, 'V6 train/eval/teacher horizons must stay synchronized'
GAMMA = 0.99
LATENT_GROUPS = ((0, 0), (1, 0), (0, 1), (1, 1))
ALGORITHM_CONTRACT = {
    'recipe': 'vanilla_crl_v5',
    'binary_nce': True,
    'use_td': False,
    'use_cpc': False,
    'use_gcbc': False,
    'add_mc_to_td': False,
    'twin_q': True,
    'bc_coef': 0.05,
    'random_goals': 0.0,
    'batch_size': 1024,
    'repr_dim': 16,
    'repr_norm': False,
    'repr_norm_temp': True,
    'hidden_layer_sizes': [1024, 1024],
    'use_image_obs': False,
    'use_layer_norm': False,
    'discount': 0.99,
    'tau': 0.005,
    'actor_learning_rate': 3e-4,
    'learning_rate': 3e-4,
    'num_sgd_steps_per_step': 4,
    'updates_per_step': 1,
}


def _validate_benchmark_args(args):
  if args.n <= 0:
    raise ValueError(f'--n must be positive, got {args.n}')
  if args.horizon <= 0:
    raise ValueError(f'--horizon must be positive, got {args.horizon}')
  for name in ('p_active_1', 'p_active_2'):
    value = float(getattr(args, name))
    if not 0.0 <= value <= 1.0:
      raise ValueError(f'--{name.replace("_", "-")} must be in [0, 1], '
                       f'got {value}')
  for zone in (1, 2):
    lo = int(getattr(args, f't0_min_{zone}'))
    hi = int(getattr(args, f't0_max_{zone}'))
    if lo > hi:
      raise ValueError(f'zone {zone} t0 minimum exceeds maximum: {lo}>{hi}')


def _training_contract(args):
  """Require eval knobs to match the checkpoint's recorded train/dataset."""
  run_dir = os.path.dirname(os.path.abspath(args.ckpt))
  manifest_path = os.path.join(run_dir, 'benchmark_config.json')
  fingerprint_path = os.path.join(run_dir, 'offline_dataset.sha256')
  if not os.path.isfile(manifest_path):
    raise FileNotFoundError(
        f'authoritative V6 eval requires {manifest_path}; checkpoint alone '
        'cannot establish its environment/timing contract')
  if not os.path.isfile(fingerprint_path):
    raise FileNotFoundError(
        f'authoritative V6 eval requires {fingerprint_path}; the checkpoint '
        'does not have the offline dataset audit sidecar written by train()')
  with open(manifest_path, encoding='utf-8') as handle:
    manifest = json.load(handle)
  with open(fingerprint_path, encoding='utf-8') as handle:
    fingerprint = json.load(handle)
  dataset_meta = fingerprint.get('meta') or {}

  goal_dim = 2 if args.env_name == ENV_XY else 29
  expected = {
      'environment_version': V6.ENV_VERSION,
      'env_name': args.env_name,
      'horizon': int(args.horizon),
      'p_active_1': float(args.p_active_1),
      'p_active_2': float(args.p_active_2),
      'teacher_detour_prob': float(CT.TEACHER_DETOUR_PROB),
  }
  manifest_values = {
      name: manifest.get(name) for name in expected}
  dataset_values = {
      name: dataset_meta.get(name) for name in expected}
  timing_wanted = {
      't0_range_1': [int(args.t0_min_1), int(args.t0_max_1)],
      't0_range_2': [int(args.t0_min_2), int(args.t0_max_2)],
  }
  mismatches = {}
  for name, wanted in expected.items():
    for source, values in (('manifest', manifest_values),
                           ('dataset_meta', dataset_values)):
      got = values[name]
      if isinstance(wanted, float):
        matches = (isinstance(got, (int, float))
                   and np.isclose(float(got), wanted,
                                  rtol=0.0, atol=1e-12))
      else:
        matches = got == wanted
      if not matches:
        mismatches[f'{source}.{name}'] = {
            'recorded': got, 'cli_expected': wanted}
  dataset_timing = dataset_meta.get('timing') or {}
  for name, wanted in timing_wanted.items():
    got_manifest = manifest.get(name)
    dataset_name = f'{name}_inclusive'
    got_dataset = dataset_timing.get(dataset_name)
    if got_manifest != wanted:
      mismatches[f'manifest.{name}'] = {
          'recorded': got_manifest, 'cli_expected': wanted}
    if got_dataset != wanted:
      mismatches[f'dataset_meta.timing.{dataset_name}'] = {
          'recorded': got_dataset, 'cli_expected': wanted}
  learner = manifest.get('learner_contract') or {}
  for name, wanted in (('state_dim', 29), ('goal_dim', goal_dim),
                       ('flat_dim', 29 + goal_dim)):
    if learner.get(name) != wanted:
      mismatches[f'manifest.learner_contract.{name}'] = {
          'recorded': learner.get(name), 'cli_expected': wanted}
  fail = manifest.get('failure_bank') or {}
  if fail != {'path': '', 'alpha': 0.0, 'enabled': False}:
    mismatches['manifest.failure_bank'] = {
        'recorded': fail,
        'cli_expected': {'path': '', 'alpha': 0.0, 'enabled': False},
    }
  if dataset_meta.get('failure_bank') is not False:
    mismatches['dataset_meta.failure_bank'] = {
        'recorded': dataset_meta.get('failure_bank'),
        'cli_expected': False,
    }
  if manifest.get('algorithm_contract') != ALGORITHM_CONTRACT:
    mismatches['manifest.algorithm_contract'] = {
        'recorded': manifest.get('algorithm_contract'),
        'cli_expected': ALGORITHM_CONTRACT,
    }
  if mismatches:
    raise ValueError(
        f'checkpoint/eval benchmark mismatch for {args.ckpt}: {mismatches}')
  digest = fingerprint.get('sha256')
  if not isinstance(digest, str) or len(digest) != 64:
    raise ValueError(f'{fingerprint_path} has no valid dataset SHA-256')
  if manifest.get('dataset_sha256') != digest:
    raise ValueError(
        f'{manifest_path} dataset_sha256 does not match '
        f'{fingerprint_path}: manifest={manifest.get("dataset_sha256")!r}, '
        f'fingerprint={digest!r}')
  return manifest, digest


def _configure_env(args, seed):
  cfg = build_offline_cfg()
  cfg.env_name = args.env_name
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = int(args.horizon)
  cfg.max_episode_steps = int(args.horizon)
  cfg.rockfall_p_active_1 = float(args.p_active_1)
  cfg.rockfall_p_active_2 = float(args.p_active_2)
  cfg.rockfall_t0_min_1 = int(args.t0_min_1)
  cfg.rockfall_t0_max_1 = int(args.t0_max_1)
  cfg.rockfall_t0_min_2 = int(args.t0_min_2)
  cfg.rockfall_t0_max_2 = int(args.t0_max_2)
  cfg.fail_bank_path = ''
  cfg.fail_neg_alpha = 0.0
  # Match the explicitly pinned vanilla architecture in the V6 trainer even
  # if the caller's shell happens to set OFFLINE_LAYER_NORM.
  cfg.use_layer_norm = False
  env = envs_mod.make_env(args.env_name, cfg, seed=int(seed))
  expected_goal_dim = 2 if args.env_name == ENV_XY else 29
  assert (cfg.obs_dim, cfg.goal_dim, cfg.action_dim) == (
      29, expected_goal_dim, 8), (
          cfg.obs_dim, cfg.goal_dim, cfg.action_dim)
  return cfg, env


def build_mean_policy(ckpt_path, args):
  """Restore a checkpoint and return the deterministic tanh(actor.loc)."""
  cfg, _ = _configure_env(args, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, state = ckpt_mod.load_checkpoint(ckpt_path)
  policy_params = state.policy_params

  @jax.jit
  def act_mean(observation):
    return jnp.tanh(nets.policy_network.apply(
        policy_params, observation).loc)

  width = int(cfg.obs_dim) + int(cfg.goal_dim)
  try:
    probe = act_mean(jnp.zeros((1, width), jnp.float32))
    np.asarray(probe)
  except Exception as exc:  # give a contract error instead of a deep JAX trace
    raise ValueError(
        f'checkpoint is incompatible with {args.env_name} ({width}-column '
        'state+goal input); use the environment representation it was '
        'trained with') from exc
  return act_mean, int(step), cfg


def _yaw_deg(quaternion):
  q = np.asarray(quaternion, float)
  w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
  return float(np.degrees(np.arctan2(2 * (w * z + x * y),
                                     1 - 2 * (y * y + z * z))))


def evaluate(act_mean, args):
  """Evaluate only natural environment draws; never force U or t0."""
  _, env = _configure_env(args, seed=args.seed)
  expected_width = 31 if args.env_name == ENV_XY else 58
  rows = []
  progress_every = max(1, min(100, args.n // 10))
  for episode in range(args.n):
    observation = env.reset()  # authoritative natural U1/U2/t0 draws
    if tuple(observation.shape) != (expected_width,):
      raise RuntimeError(
          f'{args.env_name} emitted {observation.shape}, expected '
          f'({expected_width},)')
    yaw = _yaw_deg(observation[3:7])
    if abs(yaw) > 15.0:
      raise RuntimeError(
          f'episode {episode}: non-canonical start yaw {yaw:.1f} degrees')

    # Privileged audit capture only.  These values are never concatenated to
    # ``observation`` and never enter ``act_mean``.
    u1 = bool(env.privileged_rockfall_active_1)
    u2 = bool(env.privileged_rockfall_active_2)
    sampled_t0_1 = int(env.privileged_sampled_start(1))
    sampled_t0_2 = int(env.privileged_sampled_start(2))
    active_t0_1 = env.privileged_rockfall_start(1)
    active_t0_2 = env.privileged_rockfall_start(2)

    info = {}
    episode_return = 0.0
    reward = 0.0
    for time_step in range(args.horizon):
      action = np.asarray(
          act_mean(jnp.asarray(observation[None], jnp.float32))[0])
      observation, reward, done, info = env.step(action)
      episode_return += float(reward)
      if done or reward > 0:
        break

    success = bool(info.get('success', reward > 0))
    failure = bool(info.get('failure', False))
    timeout = bool(not success and not failure)
    if int(success) + int(failure) + int(timeout) != 1:
      raise RuntimeError(
          f'episode {episode}: invalid outcome success={success}, '
          f'failure={failure}, timeout={timeout}')
    failure_zone = info.get('failure_zone')
    if failure:
      if failure_zone not in (1, 2):
        raise RuntimeError(
            f'episode {episode}: physical failure lacks zone attribution: '
            f'{failure_zone!r}')
      failure_zone = int(failure_zone)
    else:
      failure_zone = None
    if (bool(info.get('rockfall_active_1')) != u1
        or bool(info.get('rockfall_active_2')) != u2):
      raise RuntimeError(f'episode {episode}: reset latents changed in flight')

    route = info.get('route')
    if route not in ('shortcut', 'detour', None):
      raise RuntimeError(f'episode {episode}: unknown route {route!r}')
    steps = int(time_step + 1)
    rows.append({
        'episode_id': int(episode),
        'u1': u1,
        'u2': u2,
        'latent': f'U{int(u1)}{int(u2)}',
        'sampled_t0_1': sampled_t0_1,
        'sampled_t0_2': sampled_t0_2,
        'rockfall_start_1': (None if active_t0_1 is None
                             else int(active_t0_1)),
        'rockfall_start_2': (None if active_t0_2 is None
                             else int(active_t0_2)),
        'success': success,
        'failure': failure,
        'timeout': timeout,
        'failure_zone': failure_zone,
        'zone1_death': bool(failure_zone == 1),
        'zone2_death': bool(failure_zone == 2),
        'route': route,
        'entered_hazard_1': bool(info.get('entered_hazard_1')),
        'entered_hazard_2': bool(info.get('entered_hazard_2')),
        'mouth_step_1': info.get('mouth_step_1'),
        'mouth_step_2': info.get('mouth_step_2'),
        'band_entry_step_1': info.get('band_entry_step_1'),
        'band_entry_step_2': info.get('band_entry_step_2'),
        'steps': steps,
        'return': float(episode_return),
        'discounted': (round(float(GAMMA ** steps), 6)
                       if success else 0.0),
        'start_yaw_deg': round(yaw, 2),
    })
    if (episode + 1) % progress_every == 0 or episode + 1 == args.n:
      print(f'  {episode + 1}/{args.n}', flush=True)
  return rows


def _rate(rows, key):
  if not rows:
    return None
  return round(float(np.mean([bool(row[key]) for row in rows])), 4)


def _mean(rows, key):
  if not rows:
    return None
  return round(float(np.mean([float(row[key]) for row in rows])), 4)


def _block(rows):
  n = len(rows)
  route_counts = {
      'shortcut': int(sum(row['route'] == 'shortcut' for row in rows)),
      'detour': int(sum(row['route'] == 'detour' for row in rows)),
      'no_route': int(sum(row['route'] is None for row in rows)),
  }
  routes = {
      name: {'n': count,
             'rate': (round(count / n, 4) if n else None)}
      for name, count in route_counts.items()
  }
  success_n = int(sum(row['success'] for row in rows))
  failure_n = int(sum(row['failure'] for row in rows))
  timeout_n = int(sum(row['timeout'] for row in rows))
  return {
      'n': n,
      'success': _rate(rows, 'success'),
      'failure': _rate(rows, 'failure'),
      'timeout': _rate(rows, 'timeout'),
      'success_n': success_n,
      'failure_n': failure_n,
      'timeout_n': timeout_n,
      'zone1_deaths': int(sum(row['zone1_death'] for row in rows)),
      'zone2_deaths': int(sum(row['zone2_death'] for row in rows)),
      'routes': routes,
      'mean_steps': _mean(rows, 'steps'),
      'mean_discounted': _mean(rows, 'discounted'),
  }


def summarize(rows, p_active_1=V6.P_ACTIVE_1,
              p_active_2=V6.P_ACTIVE_2):
  """Overall and required U00/U10/U01/U11 outcome/route breakdowns."""
  by_latent = {}
  expected = {}
  for u1, u2 in LATENT_GROUPS:
    key = f'U{u1}{u2}'
    group = [row for row in rows
             if row['u1'] == bool(u1) and row['u2'] == bool(u2)]
    by_latent[key] = _block(group)
    expected[key] = round(
        (p_active_1 if u1 else 1 - p_active_1)
        * (p_active_2 if u2 else 1 - p_active_2), 6)
  n = len(rows)
  observed = {key: (round(value['n'] / n, 6) if n else None)
              for key, value in by_latent.items()}
  return {
      'policy_headline': 'deterministic_actor_mean',
      'overall': _block(rows),
      'by_latent': by_latent,
      'natural_latent_draws': {
          'expected_under_independence': expected,
          'observed': observed,
          'u1_rate': _rate(rows, 'u1'),
          'u2_rate': _rate(rows, 'u2'),
      },
  }


def _write_results(rows, summary, step, args, dataset_sha256):
  label = args.method_label or os.path.basename(
      os.path.dirname(os.path.abspath(args.ckpt)))
  label = f'{label}_mean'
  record = {
      'label': label,
      'policy': 'deterministic_actor_mean',
      'env': args.env_name,
      'environment_version': V6.ENV_VERSION,
      'ckpt': args.ckpt,
      'ckpt_step': int(step),
      'n_eval': int(args.n),
      'eval_seed': int(args.seed),
      'reset_protocol': 'natural env.reset(); no latent or timing overrides',
      'horizon': int(args.horizon),
      'p_active_1': float(args.p_active_1),
      'p_active_2': float(args.p_active_2),
      't0_range_1': [int(args.t0_min_1), int(args.t0_max_1)],
      't0_range_2': [int(args.t0_min_2), int(args.t0_max_2)],
      'learner_input_dim': 31 if args.env_name == ENV_XY else 58,
      'failure_bank_enabled': False,
      'training_dataset_sha256': dataset_sha256,
      'summary': summary,
  }
  out_dir = args.out_dir or (os.path.dirname(args.ckpt) or '.')
  os.makedirs(out_dir, exist_ok=True)
  local_path = os.path.join(out_dir, 'eval_rockfall_clock_v6_mean.json')
  with open(local_path, 'w', encoding='utf-8') as handle:
    json.dump({**record, 'episodes': rows}, handle, indent=2)

  os.makedirs(args.results_root, exist_ok=True)
  aggregate_path = os.path.join(args.results_root, 'baseline_results.json')
  aggregate = []
  if os.path.exists(aggregate_path):
    with open(aggregate_path, encoding='utf-8') as handle:
      aggregate = json.load(handle)
    if not isinstance(aggregate, list):
      raise ValueError(f'{aggregate_path} must contain a JSON list')
  aggregate = [item for item in aggregate
               if item.get('label') != label] + [record]
  with open(aggregate_path, 'w', encoding='utf-8') as handle:
    json.dump(aggregate, handle, indent=2)
  return local_path, aggregate_path


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Natural-draw deterministic-mean V6 baseline evaluation.')
  parser.add_argument('--ckpt', required=True)
  parser.add_argument('--n', type=int, default=1000)
  parser.add_argument('--seed', type=int, default=909)
  parser.add_argument(
      '--env-name', choices=ENV_NAMES, default=ENV_XY,
      help='must match checkpoint training; default is 31-column XY arm')
  parser.add_argument('--horizon', type=int, default=HORIZON)
  parser.add_argument('--p-active-1', type=float, default=V6.P_ACTIVE_1)
  parser.add_argument('--p-active-2', type=float, default=V6.P_ACTIVE_2)
  parser.add_argument('--t0-min-1', type=int, default=V6.T0_MIN_1)
  parser.add_argument('--t0-max-1', type=int, default=V6.T0_MAX_1)
  parser.add_argument('--t0-min-2', type=int, default=V6.T0_MIN_2)
  parser.add_argument('--t0-max-2', type=int, default=V6.T0_MAX_2)
  parser.add_argument('--method-label', default=None)
  parser.add_argument('--out-dir', default=None)
  parser.add_argument('--results-root', default=OUT_ROOT)
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  _validate_benchmark_args(args)
  _, dataset_sha256 = _training_contract(args)
  act_mean, step, cfg = build_mean_policy(args.ckpt, args)
  print(
      f'ckpt {args.ckpt} @ step {step} | deterministic actor mean '
      f'| env {args.env_name} | input {cfg.obs_dim + cfg.goal_dim} '
      f'| natural draws n={args.n}, seed={args.seed} | H={args.horizon} '
      f'| p=({args.p_active_1:g},{args.p_active_2:g}) '
      f'| t0=([{args.t0_min_1},{args.t0_max_1}],'
      f'[{args.t0_min_2},{args.t0_max_2}])', flush=True)
  rows = evaluate(act_mean, args)
  summary = summarize(rows, args.p_active_1, args.p_active_2)
  print(json.dumps(summary, indent=2), flush=True)
  local_path, aggregate_path = _write_results(
      rows, summary, step, args, dataset_sha256)
  print('->', local_path, 'and', aggregate_path, flush=True)


if __name__ == '__main__':
  main()
