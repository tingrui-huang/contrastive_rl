"""Collect the offline dataset for point_two_route_swamp_windy_z_v0.

Deliberately a thin shell around the ACCEPTED 2-D collectors rather than a
reimplementation: the behaviour policies are imported, not copied.

  --mode teacher   scripts.collect_swamp_windy.make_windy_teacher
                   (per-step reactive gate-aware teacher + force_safe coin +
                   random_frac uniform episodes)
  --mode baddemo   scripts.collect_swamp_windy_baddemo.make_bad_demonstrator
                   (BFS shortcut oracle with the dodge step removed)

Both work unchanged on the Z env because TwoRouteSwampWindyZEnv keeps
self.state as the 2-D XY vector; only the OBSERVATION is 3-D. So the maze,
start/goal, horizon, hidden-bit mechanism, p_active, teacher logic and
proportions are all identical to the 2-D dataset by construction, and the only
thing that changes is the environment.

Output layout matches the 2-D collectors key-for-key so
scripts/merge_swamp_windy_baddemo.py and crl.offline_audit work unchanged; the
only difference is width:

    obs  [N, 51, 6]   [x, y, z, g_x, g_y, g_z]     (was [N, 51, 4])
    act  [N, 51, 2]   act[:, -1] dummy

swamp_bits / route_label / teacher_mode / force_safe / wait_count /
entered_active_swamp stay AUDIT-ONLY per-episode arrays and never enter the
learner observation.

Z IS STORED RAW (0 or z_min = -0.5). The z_physical scaling lives in the model
preprocessing path (crl/obs_norm.obs_scale_vector, applied inside
crl/networks.py), so nothing here pre-normalizes and nothing downstream
normalizes twice.

Run:
  python -m scripts.collect_swamp_windy_z --mode teacher --episodes 6000 \
      --random_frac 0.2 --force_safe_prob 0.05 --teacher_noise 0.15 --seed 0 \
      --out datasets/swamp_windy_z_teacher_s0.npz
  python -m scripts.collect_swamp_windy_z --mode baddemo --episodes 600 \
      --seed 0 --out datasets/swamp_windy_z_baddemo_s0.npz
"""
import argparse
import hashlib
import json
import os

import numpy as np

from crl import envs as envs_mod
from crl.config import Config
from scripts.collect_swamp_windy import MODE as MODE_2D
from scripts.collect_swamp_windy import ROUTE, make_windy_teacher
from scripts.collect_swamp_windy_baddemo import make_bad_demonstrator

ENV = 'point_two_route_swamp_windy_z_v0'
ENVS = ('point_two_route_swamp_windy_z_v0', 'point_two_route_swamp_windy_z_v1')
MODE = dict(MODE_2D)
MODE['bad_demo'] = 4


def _sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def content_sha(path):
  """sha256 over the ARRAY CONTENTS; datasets/ is gitignored and regenerated."""
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for k in sorted(d.files):
      a = d[k]
      h.update(k.encode())
      h.update(str(a.dtype).encode())
      h.update(str(a.shape).encode())
      h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def collect(mode, episodes, random_frac, force_safe_prob, teacher_noise, seed,
            out, force=False, env_name=ENV):
  if os.path.exists(out) and not force:
    raise SystemExit('REFUSING to overwrite frozen dataset %s (use --force).'
                     % out)
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  assert cfg.obs_dim == 3 and cfg.goal_dim == 3, 'not the 3-D Z env'
  rng = np.random.default_rng(seed)
  policy = (make_windy_teacher(env, rng, force_safe_prob) if mode == 'teacher'
            else make_bad_demonstrator(env))
  L = env.max_episode_steps + 1
  D, A = cfg.obs_dim + cfg.goal_dim, cfg.action_dim
  n_random = int(round(episodes * random_frac)) if mode == 'teacher' else 0

  obs_out = np.zeros((episodes, L, D), np.float32)
  act_out = np.zeros((episodes, L, A), np.float32)
  bits_out = np.zeros((episodes, L, 3), np.uint8)
  route_out = np.zeros((episodes,), np.int64)
  mode_out = np.zeros((episodes,), np.int64)
  fsafe_out = np.zeros((episodes,), np.uint8)
  wait_out = np.zeros((episodes,), np.int64)
  died_out = np.zeros((episodes,), np.uint8)
  succ = []

  for ep in range(episodes):
    env.reset()
    g2 = env.goal.copy()                      # 2-D, for the oracles
    memo = {}
    is_random = ep < n_random
    traj = [env.state.copy()]
    for t in range(env.max_episode_steps):
      obs_out[ep, t] = env._get_obs()          # [x, y, z, g_x, g_y, g_z]
      bits_out[ep, t] = env.swamp_bits.astype(np.uint8)
      if is_random:
        a = rng.uniform(-1, 1, A).astype(np.float32)
      else:
        a = np.asarray(policy(env.state.copy(), g2, memo), np.float32)
        if teacher_noise > 0 and np.any(a != 0):
          a = np.clip(a + rng.normal(0, teacher_noise, A), -1, 1).astype(
              np.float32)
      act_out[ep, t] = a
      env.step(a)
      traj.append(env.state.copy())
    obs_out[ep, -1] = env._get_obs()
    bits_out[ep, -1] = env.swamp_bits.astype(np.uint8)
    traj = np.array(traj)
    died_out[ep] = int(env.dead)
    if is_random:
      mode_out[ep] = MODE['random']
      route_out[ep] = ROUTE['random']
    else:
      mode_out[ep] = MODE[memo.get('teacher_mode', 'immediate_shortcut')]
      fsafe_out[ep] = int(memo.get('force_safe', False))
      wait_out[ep] = int(memo.get('wait_count', 0))
      used_safe = bool(np.any(traj[:, 1] < 2.0))
      crossed = bool(np.any(traj[:, 0] > 6.0)) and not used_safe
      route_out[ep] = (ROUTE['shortcut'] if crossed else
                       ROUTE['safe_detour'] if used_safe else ROUTE['other'])
    succ.append(float(np.min(np.linalg.norm(traj - g2, axis=1)) < 0.5))
    if (ep + 1) % 500 == 0:
      print('  %d/%d (reached@0.5 %.3f, died %.3f)'
            % (ep + 1, episodes, np.mean(succ), died_out[:ep + 1].mean()),
            flush=True)

  succ = np.asarray(succ)
  z = obs_out[:, :, 2]
  meta = {
      'env_name': env_name, 'setting': 'windy_lethal_z_%s' % mode,
      'episodes': int(episodes), 'seed': int(seed), 'mode': mode,
      'behavior_policy': ('windy_reactive_teacher(force_safe=%s, per-step '
                          'dodge) + random_frac=%s' % (force_safe_prob,
                                                       random_frac))
                         if mode == 'teacher' else
                         'BFS shortcut oracle with the dodge step REMOVED',
      'random_frac': float(random_frac if mode == 'teacher' else 0.0),
      'force_safe_prob': float(force_safe_prob if mode == 'teacher' else 0.0),
      'teacher_noise': float(teacher_noise),
      'per_cell_swamp_prob': float(env.active_prob),
      'obs_dim': int(cfg.obs_dim), 'goal_dim': int(cfg.goal_dim),
      'obs_layout': '[x, y, z, g_x, g_y, g_z]',
      'action_dim': int(A), 'max_episode_steps': int(env.max_episode_steps),
      'episode_len_rows_L': int(L), 'obs_width_D': int(D),
      'n_transitions': int(episodes * (L - 1)),
      # z_v1 has no sink_settle_substeps: its settle length is DERIVED from
      # z_min / (sink_speed*sink_dt), so the field is absent rather than
      # misleadingly present.
      'z_min': float(env.z_min), 'sink_speed': float(env.sink_speed),
      'sink_dt': float(env.sink_dt),
      'sink_settle_substeps': (int(env.sink_settle_substeps)
                               if hasattr(env, 'sink_settle_substeps')
                               else None),
      'steps_to_settle': (int(env.steps_to_settle)
                          if hasattr(env, 'steps_to_settle') else None),
      'sinking_exposed_across_env_steps':
          bool(hasattr(env, 'steps_to_settle')),
      'z_stored_raw': True,
      'z_scaling_note': 'z is stored RAW; z_physical is applied inside '
                        'crl/networks.py via crl.obs_norm.obs_scale_vector',
      'frac_z_zero': float((z == 0).mean()),
      'frac_z_negative': float((z < 0).mean()),
      'teacher_mode_code': MODE, 'route_label_code': ROUTE,
      'died_rate_overall': float(died_out.mean()),
      'reached_0p5_rate': float(succ.mean()),
      'audit_fields': ['swamp_bits', 'route_label', 'teacher_mode',
                       'force_safe', 'wait_count', 'entered_active_swamp'],
      'note': 'entered_active_swamp == DIED; audit fields are AUDIT-ONLY; the '
              'learner obs is [x,y,z,gx,gy,gz] only',
  }
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  np.savez(out, obs=obs_out, act=act_out, swamp_bits=bits_out,
           route_label=route_out, teacher_mode=mode_out, force_safe=fsafe_out,
           wait_count=wait_out, entered_active_swamp=died_out,
           meta=np.array(json.dumps(meta)))
  fsha, csha = _sha256(out), content_sha(out)
  json.dump(dict(path=os.path.abspath(out), sha256=fsha, content_sha256=csha,
                 size_bytes=int(os.path.getsize(out)),
                 obs_shape=list(obs_out.shape), act_shape=list(act_out.shape),
                 frozen=True, meta=meta),
            open(out + '.manifest.json', 'w'), indent=2)
  try:
    os.chmod(out, 0o444)
  except OSError:
    pass
  print('\nFROZEN Z dataset -> %s (%.1f MB)' % (out,
                                                os.path.getsize(out) / 1e6))
  print('  obs %s  act %s' % (obs_out.shape, act_out.shape))
  print('  died %.4f  reached@0.5 %.4f' % (died_out.mean(), succ.mean()))
  print('  z==0 %.5f   z<0 %.5f' % ((z == 0).mean(), (z < 0).mean()))
  print('  file sha256    %s' % fsha)
  print('  content sha256 %s' % csha)


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument('--mode', choices=('teacher', 'baddemo'), default='teacher')
  p.add_argument('--env', choices=ENVS, default=ENV,
                 help='z_v0 (default, unchanged) or z_v1 (sinking spread over '
                      'env steps). The behaviour policies are identical; only '
                      'the environment version differs.')
  p.add_argument('--episodes', type=int, default=6000)
  p.add_argument('--random_frac', type=float, default=0.2)
  p.add_argument('--force_safe_prob', type=float, default=0.05)
  p.add_argument('--teacher_noise', type=float, default=0.15)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', required=True)
  p.add_argument('--force', action='store_true')
  a = p.parse_args()
  if a.force and os.path.exists(a.out):
    try:
      os.chmod(a.out, 0o644)
    except OSError:
      pass
  collect(a.mode, a.episodes, a.random_frac, a.force_safe_prob,
          a.teacher_noise, a.seed, a.out, force=a.force, env_name=a.env)


if __name__ == '__main__':
  main()
