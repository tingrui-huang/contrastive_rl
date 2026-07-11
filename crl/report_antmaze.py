"""AntMaze evidence pipeline (mirrors report_maze for the Ant locomotion maze).

State layout from crl.envs.MazeEnv:  state[0:2]=torso xy, state[2]=torso z,
goal = flat[obs_dim:obs_dim+2] = desired xy.  Metrics logged:

  * torso XY displacement + goal-directed velocity
  * torso height + fall rate (unhealthy)
  * episode length + success (xy-dist < 0.5)
  * action norm + saturation
  * wall collisions/crossings (torso xy inside a wall cell)
  * top-down XY trajectory plot (+ optional rendered video on a GL backend)
  * perturbation-based action-sensitivity: actor action vs local/random action
    perturbations via short MuJoCo clone-rollouts, scored by goal-directed xy
    progress (no assumption that individual joints map to nav directions).

Run:  python -m crl.report_antmaze --env_name antmaze_umaze --ckpt run/best.pkl --out artifacts/ant
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

from crl import envs as envs_mod
from crl.config import Config
from crl.report_maze import load_nets

FALL_Z = 0.3          # torso below this height counts as fallen/unhealthy
SUCC = 0.5


def _maze_map(env):
  return np.asarray(env._env.unwrapped.maze.maze_map)


def _wall_cell(env, xy):
  mz = env._env.unwrapped.maze
  r, c = mz.cell_xy_to_rowcol(np.asarray(xy))
  m = mz.maze_map
  if 0 <= r < len(m) and 0 <= c < len(m[0]):
    return m[r][c] == 1
  return True


def rollout(env, policy, dt=0.05):
  """policy(flat_obs)->action, or None for uniform-random."""
  flat = env.reset()
  od = env.obs_dim
  goal = flat[od:od + 2].copy()
  xy = [flat[0:2].copy()]
  zs = [float(flat[2])]
  acts, wall_steps = [], 0
  hit = 0.0
  dists = [float(np.linalg.norm(flat[0:2] - goal))]
  for _ in range(env.max_episode_steps):
    if policy is None:
      a = np.random.default_rng().uniform(-1, 1, env.action_dim).astype(np.float32)
    else:
      a = np.asarray(policy(flat), np.float32)
    flat, r, _, _ = env.step(a)
    hit = max(hit, float(r))
    xy.append(flat[0:2].copy()); zs.append(float(flat[2])); acts.append(a)
    dists.append(float(np.linalg.norm(flat[0:2] - goal)))
    if _wall_cell(env, flat[0:2]):
      wall_steps += 1
  xy = np.array(xy); acts = np.array(acts); zs = np.array(zs)
  gdir = goal - xy[0]
  gdir = gdir / (np.linalg.norm(gdir) + 1e-9)
  step_xy = np.diff(xy, axis=0)
  goal_vel = float(np.mean(step_xy @ gdir) / dt)       # goal-directed velocity
  return dict(
      xy=xy, goal=goal,
      success=hit, final_goal_dist=float(dists[-1]), min_goal_dist=float(min(dists)),
      torso_xy_disp=float(np.linalg.norm(xy[-1] - xy[0])),
      path_len=float(np.sum(np.linalg.norm(step_xy, axis=1))),
      goal_directed_velocity=goal_vel,
      min_z=float(zs.min()), mean_z=float(zs.mean()),
      fell=bool(zs.min() < FALL_Z),
      action_norm=float(np.mean(np.linalg.norm(acts, axis=1))),
      action_saturation=float(np.mean(np.abs(acts) > 0.99)),
      wall_crossings=int(wall_steps), ep_len=len(acts))


def aggregate(eps):
  ks = ['success', 'final_goal_dist', 'min_goal_dist', 'torso_xy_disp', 'path_len',
        'goal_directed_velocity', 'min_z', 'mean_z', 'fell', 'action_norm',
        'action_saturation', 'wall_crossings', 'ep_len']
  out = {'n_episodes': len(eps)}
  for k in ks:
    v = np.array([e[k] for e in eps], float)
    out[k + '_mean'] = float(v.mean())
    out[k + '_std'] = float(v.std())
  return out


# --------------------------------------------------------------------------- #
# perturbation-based action-sensitivity (short MuJoCo clone-rollouts)
# --------------------------------------------------------------------------- #
def perturbation_scan(env, greedy, n_states=8, k_steps=5, n_perturb=6,
                      noise=0.3, seed=0):
  u = env._env.unwrapped
  rng = np.random.default_rng(seed)

  def progress_from(qpos, qvel, goal, a):
    u.data.qpos[:] = qpos; u.data.qvel[:] = qvel
    mujoco.mj_forward(u.model, u.data)
    d0 = float(np.linalg.norm(np.asarray(u.data.qpos[:2]) - goal))
    for _ in range(k_steps):
      u.step(np.asarray(a, np.float32))
    return d0 - float(np.linalg.norm(np.asarray(u.data.qpos[:2]) - goal))

  beat_local, beat_random, actor_prog = [], [], []
  for _ in range(n_states):
    flat = env.reset()
    goal = flat[env.obs_dim:env.obs_dim + 2].copy()
    for _ in range(int(rng.integers(5, 25))):       # wander to a non-initial state
      flat, _, _, _ = env.step(rng.uniform(-1, 1, env.action_dim).astype(np.float32))
    a0 = np.asarray(greedy(flat), np.float32)
    qpos0 = np.asarray(u.data.qpos).copy(); qvel0 = np.asarray(u.data.qvel).copy()
    pa = progress_from(qpos0, qvel0, goal, a0)
    pl = [progress_from(qpos0, qvel0, goal,
                        np.clip(a0 + rng.normal(0, noise, env.action_dim), -1, 1))
          for _ in range(n_perturb)]
    pr = [progress_from(qpos0, qvel0, goal,
                        rng.uniform(-1, 1, env.action_dim).astype(np.float32))
          for _ in range(n_perturb)]
    actor_prog.append(pa)
    beat_local.append(int(pa >= np.mean(pl)))
    beat_random.append(int(pa >= np.mean(pr)))
  return {'n_states': n_states, 'k_steps': k_steps,
          'actor_goal_progress_mean': float(np.mean(actor_prog)),
          'actor_beats_local_frac': float(np.mean(beat_local)),
          'actor_beats_random_frac': float(np.mean(beat_random))}


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_topdown(env, eps, out):
  m = _maze_map(env)
  fig, ax = plt.subplots(figsize=(5, 5))
  mz = env._env.unwrapped.maze
  # draw wall cells as squares in world coords
  for r in range(len(m)):
    for c in range(len(m[0])):
      if m[r][c] == 1:
        x, y = mz.cell_rowcol_to_xy(np.array([r, c]))
        s = mz.maze_size_scaling
        ax.add_patch(plt.Rectangle((x - s / 2, y - s / 2), s, s,
                                   color='0.7', zorder=0))
  for e in eps[:10]:
    ax.plot(e['xy'][:, 0], e['xy'][:, 1], '-', lw=1, alpha=0.7)
    ax.scatter(*e['xy'][0], c='tab:green', s=25, zorder=3)
    ax.scatter(*e['goal'], c='red', marker='*', s=90, zorder=3)
  ax.set_aspect('equal'); ax.set_title('AntMaze top-down XY trajectories')
  fig.tight_layout(); fig.savefig(out, dpi=100); plt.close(fig)


def render_video(env_name, ckpt, out, episodes=1, seed=0):
  """Optional rendered rollout (needs MUJOCO_GL=egl/glfw). Best on Colab."""
  import imageio
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed, render_mode='rgb_array')
  _, _, greedy, _ = load_nets(env_name, ckpt, cfg)
  frames = []
  for _ in range(episodes):
    flat = env.reset()
    for _ in range(env.max_episode_steps):
      flat, _, _, _ = env.step(greedy(flat))
      frames.append(np.asarray(env.render()))
  imageio.mimsave(out, frames, duration=1 / 30)
  return out


# --------------------------------------------------------------------------- #
# smoke health (integration / optimization, NOT task success)
# --------------------------------------------------------------------------- #
def smoke_health(metrics_json):
  m = json.load(open(metrics_json))
  finite = all(bool(np.all(np.isfinite(
      [v for v in r.values() if isinstance(v, (int, float))]))) for r in m)
  first, last = m[0], m[-1]
  return {
      'evals': len(m), 'all_finite': bool(finite),
      'critic_loss_first': first.get('critic_loss'),
      'critic_loss_last': last.get('critic_loss'),
      'logits_gap_first': first.get('logits_gap'),
      'logits_gap_last': last.get('logits_gap'),
      'cat_acc_first': first.get('categorical_accuracy'),
      'cat_acc_last': last.get('categorical_accuracy'),
      'critic_loss_decreased': (last.get('critic_loss', 9e9) < first.get('critic_loss', 0)),
      'logits_gap_grew': (last.get('logits_gap', 0) > first.get('logits_gap', 9e9)),
  }


def env_audit(env_name='antmaze_umaze', steps=50, seed=0):
  """Structural + rollout audit of the AntMaze wrapper. Returns (passed, details).
  Checks invariants (not hard-coded dims) + finite random rollout."""
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  flat = env.reset()
  rng = np.random.default_rng(seed)
  finite = bool(np.all(np.isfinite(flat)))
  for _ in range(steps):
    flat, r, _, _ = env.step(rng.uniform(-1, 1, cfg.action_dim).astype(np.float32))
    if not (np.all(np.isfinite(flat)) and np.isfinite(r)):
      finite = False
  checks = {
      'goal_dim_is_2': cfg.goal_dim == 2,
      'goal_slice_0_2': (cfg.start_index, cfg.end_index) == (0, 2),
      'action_dim_8': cfg.action_dim == 8,
      'has_proprio': cfg.obs_dim > cfg.goal_dim,
      'flat_len_ok': env.reset().shape[0] == cfg.obs_dim + cfg.goal_dim,
      'rollout_finite': finite,
  }
  details = {**checks, 'obs_dim': cfg.obs_dim, 'goal_dim': cfg.goal_dim,
             'action_dim': cfg.action_dim, 'max_steps': cfg.max_episode_steps}
  return all(checks.values()), details


def qualification_verdict(metrics_json, ckpt_dir, min_success=0.2,
                          sat_warn=0.9, fall_warn=0.8):
  """PASS / WARN / FAIL from metrics.json + checkpoint dir. FAIL on NaN or
  missing checkpoints; WARN on weak optimization / high saturation / falls /
  low success (AntMaze is hard, so low success is a WARN, not a FAIL)."""
  h = smoke_health(metrics_json)
  last = json.load(open(metrics_json))[-1]
  issues = []
  if not h['all_finite']:
    issues.append(('FAIL', 'NaN/inf present in metrics'))
  missing = [f for f in ('init', 'early', 'mid', 'final', 'best', 'latest')
             if not os.path.exists(os.path.join(ckpt_dir, f + '.pkl'))]
  if missing:
    issues.append(('FAIL', f'missing checkpoints: {missing}'))
  if not h['critic_loss_decreased']:
    issues.append(('WARN', 'critic_loss did not decrease'))
  if not h['logits_gap_grew']:
    issues.append(('WARN', 'logits_gap did not grow'))
  succ = last.get('success', 0.0)
  if succ < min_success:
    issues.append(('WARN', f'success {succ:.2f} < {min_success} (AntMaze is hard)'))
  sat = last.get('ant_action_saturation')
  if sat is not None and sat > sat_warn:
    issues.append(('WARN', f'action_saturation {sat:.2f} > {sat_warn}'))
  fall = last.get('ant_fall_fraction')
  if fall is not None and fall > fall_warn:
    issues.append(('WARN', f'fall_fraction {fall:.2f} > {fall_warn}'))
  verdict = ('FAIL' if any(s == 'FAIL' for s, _ in issues)
             else 'WARN' if any(s == 'WARN' for s, _ in issues) else 'PASS')
  return verdict, issues, {'success': succ, 'action_saturation': sat,
                           'fall_fraction': fall, **h}


def full_report(env_name, ckpt=None, episodes=10, seed=123, out=None):
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=seed)
  greedy = None; step = None
  if ckpt:
    _, _, greedy, step = load_nets(env_name, ckpt, cfg)
  rand_eps = [rollout(env, None) for _ in range(episodes)]
  report = {'env_name': env_name, 'ckpt': ckpt, 'step': step, 'episodes': episodes,
            'random': aggregate(rand_eps)}
  eps_for_plot = rand_eps
  if ckpt:
    tr_eps = [rollout(env, greedy) for _ in range(episodes)]
    report['trained'] = aggregate(tr_eps)
    report['perturbation_scan'] = perturbation_scan(env, greedy)
    eps_for_plot = tr_eps
  if out:
    os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, 'antmaze_report.json'), 'w'), indent=2)
    plot_topdown(env, eps_for_plot, os.path.join(out, 'topdown_trajectories.png'))
  return report


def _print(report):
  print('=' * 64)
  print(f'ANTMAZE REPORT  {report["env_name"]}  (step {report["step"]}, '
        f'{report["episodes"]} eps)')
  print('=' * 64)
  for who in ('random', 'trained'):
    if who not in report:
      continue
    a = report[who]
    print(f'\n[{who}]  n={a["n_episodes"]}')
    for k in ('success', 'min_goal_dist', 'torso_xy_disp', 'goal_directed_velocity',
              'mean_z', 'fell', 'action_norm', 'action_saturation', 'wall_crossings'):
      print(f'   {k:24s}: {a[k+"_mean"]:.3f} +/- {a[k+"_std"]:.3f}')
  if 'perturbation_scan' in report:
    s = report['perturbation_scan']
    print(f'\nperturbation action-scan ({s["n_states"]} states, {s["k_steps"]} clone steps):')
    print(f'   actor goal-progress mean : {s["actor_goal_progress_mean"]:.4f}')
    print(f'   actor beats local pert.  : {s["actor_beats_local_frac"]:.2f}')
    print(f'   actor beats random pert. : {s["actor_beats_random_frac"]:.2f}')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--env_name', default='antmaze_umaze')
  p.add_argument('--ckpt', default=None)
  p.add_argument('--episodes', type=int, default=10)
  p.add_argument('--seed', type=int, default=123)
  p.add_argument('--out', default=None)
  p.add_argument('--smoke_health', default=None, help='metrics.json to health-check')
  args = p.parse_args()
  if args.smoke_health:
    print('smoke health:', json.dumps(smoke_health(args.smoke_health), indent=2))
    return
  report = full_report(args.env_name, ckpt=args.ckpt, episodes=args.episodes,
                       seed=args.seed, out=args.out)
  _print(report)
  if args.out:
    print(f'\nsaved report + plot under {args.out}')


if __name__ == '__main__':
  main()
