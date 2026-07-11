"""EasyPush evaluation report: the conedir-success / L2C-failure contrast.

State-only numeric probes (no rendering, runs anywhere MuJoCo imports). Given a
trained checkpoint + env name, produces:

  1. overall success (greedy) + final / min object-goal distance
  2. random-policy floor and a scripted-oracle ceiling (same step() reward)
  3. success by cone bin      (for fetch_push_easy_conedir)
  4. success by goal quadrant (for fetch_push_easy_neutral_dir) + (+x vs -x)
  5. displacement-direction probe: cos(object net displacement, goal dir) vs
     cos(object net displacement, +x world) -- exposes a fixed +x/contact-side
     shortcut when cos(+x) >= cos(goal).

The montage GIF is intentionally NOT here (it needs a renderer): use
`crl.visualize.rollout_gif` from the notebook for that.

Run:
    python -m crl.report_push --env_name fetch_push_easy_conedir \
        --ckpt rob_conedir_s1/best.pkl --episodes 100
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from crl.config import Config

GRIP = slice(0, 3)
OBJ = slice(3, 6)          # object == achieved_goal (goal slice 3:6)
DES = slice(25, 28)        # desired_goal
SUCCESS_THRESH = 0.05


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
def _load_greedy(env_name, ckpt, config):
  nets = networks_mod.make_networks(
      obs_dim=config.obs_dim, goal_dim=config.goal_dim,
      action_dim=config.action_dim, repr_dim=int(config.repr_dim),
      repr_norm=config.repr_norm, repr_norm_temp=config.repr_norm_temp,
      hidden_layer_sizes=config.hidden_layer_sizes, twin_q=config.twin_q,
      use_image_obs=config.use_image_obs)
  step, state = ckpt_mod.load_checkpoint(ckpt)

  @jax.jit
  def greedy(obs):
    dist = nets.policy_network.apply(state.policy_params, obs)
    return nets.sample_eval(dist, None)

  def act(obs):
    return np.asarray(greedy(jnp.asarray(obs[None]))[0])
  return act, step


def _oracle_action(obs):
  """Scripted lift->over->descend->push controller (same 4-D action interface
  as the policy). To reach the contact point behind the object w.r.t. the goal
  direction WITHOUT knocking the object off-course, it lifts to a safe height,
  travels over the behind-point, descends, then pushes through toward the goal.
  Serves as a solvable-task ceiling; it must reposition for L2C where the goal
  can point away from the fixed -x gripper start.
  """
  grip = obs[GRIP]; obj = obs[OBJ]; goal = obs[DES]
  gdir = goal - obj
  n = float(np.linalg.norm(gdir[:2]))
  if n < SUCCESS_THRESH:
    return np.zeros(4, np.float32)                   # object already at goal
  u = gdir[:2] / (n + 1e-9)                          # 2-D unit goal direction
  behind_xy = obj[:2] - 0.055 * u                    # contact point behind obj
  safe_z = obj[2] + 0.10
  d_xy = float(np.linalg.norm(grip[:2] - behind_xy))
  if d_xy > 0.03:                                     # not yet behind the object
    if grip[2] < safe_z - 0.02:
      target = np.array([grip[0], grip[1], safe_z])          # lift straight up
    else:
      target = np.array([behind_xy[0], behind_xy[1], safe_z])  # travel over
  elif grip[2] > obj[2] + 0.02:
    target = np.array([behind_xy[0], behind_xy[1], obj[2]])    # descend behind
  else:
    target = np.array([goal[0], goal[1], obj[2]])             # push through
  a = np.clip((target - grip) * 10.0, -1.0, 1.0).astype(np.float32)
  return np.concatenate([a, [0.0]]).astype(np.float32)  # gripper closed/neutral


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #
def _rollout(env, policy, n_eps, rng):
  """policy(obs)->action, or None for uniform-random. Returns per-episode dicts."""
  eps = []
  for _ in range(n_eps):
    obs = env.reset()
    obj0 = obs[OBJ].copy()
    goal = obs[DES].copy()
    hit = 0.0
    dists = [float(np.linalg.norm(obs[OBJ] - goal))]
    for _ in range(env.max_episode_steps):
      if policy is None:
        a = rng.uniform(-1, 1, size=env.action_dim).astype(np.float32)
      else:
        a = policy(obs)
      obs, r, _, _ = env.step(a)
      hit = max(hit, float(r))
      dists.append(float(np.linalg.norm(obs[OBJ] - goal)))
    eps.append({
        'obj0': obj0, 'obj_final': obs[OBJ].copy(), 'goal': goal,
        'success': hit, 'final_dist': dists[-1], 'min_dist': float(min(dists)),
    })
  return eps


def _agg(eps):
  return {
      'success': float(np.mean([e['success'] for e in eps])),
      'final_dist': float(np.mean([e['final_dist'] for e in eps])),
      'min_dist': float(np.mean([e['min_dist'] for e in eps])),
      'n': len(eps),
  }


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _goal_angle(e):
  g = e['goal'] - e['obj0']
  return float(np.arctan2(g[1], g[0]))            # radians, 0 == +x world


def _success_by_bins(eps, edges, labels):
  """Bin episodes by goal angle into [edges[i], edges[i+1]) -> mean success."""
  out = {}
  ang = np.array([_goal_angle(e) for e in eps])
  suc = np.array([e['success'] for e in eps])
  for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
    m = (ang >= lo) & (ang < hi)
    out[lab] = {'success': float(suc[m].mean()) if m.any() else None,
                'n': int(m.sum())}
  return out


def cone_bins(eps, cone_hw):
  """conedir: split the +/-cone_hw cone around +x into center / +edge / -edge."""
  third = cone_hw / 3.0
  edges = [-cone_hw - 1e-6, -third, third, cone_hw + 1e-6]
  return _success_by_bins(eps, edges, ['-edge', 'center', '+edge'])


def quadrants(eps):
  """L2C: bin goal direction into +x / +y / -x / -y world quadrants."""
  q = np.pi / 4.0
  ang = np.array([_goal_angle(e) for e in eps])
  suc = np.array([e['success'] for e in eps])
  masks = {
      '+x (toward gripper-forward)': (ang >= -q) & (ang < q),
      '+y': (ang >= q) & (ang < 3 * q),
      '-x (behind gripper)': (ang >= 3 * q) | (ang < -3 * q),
      '-y': (ang >= -3 * q) & (ang < -q),
  }
  out = {k: {'success': float(suc[m].mean()) if m.any() else None,
             'n': int(m.sum())} for k, m in masks.items()}
  return out


def displacement_probe(eps):
  """cos(object net displacement, goal dir) vs cos(same, +x world).

  A goal-directed policy has cos_goal >> cos_posx. A fixed +x/contact-side
  shortcut has cos_posx >= cos_goal.
  """
  cos_goal, cos_posx = [], []
  posx = np.array([1.0, 0.0, 0.0])
  for e in eps:
    disp = e['obj_final'] - e['obj0']
    dn = np.linalg.norm(disp)
    if dn < 1e-4:
      continue                                     # object never moved
    gdir = e['goal'] - e['obj0']
    gdir = gdir / (np.linalg.norm(gdir) + 1e-9)
    u = disp / dn
    cos_goal.append(float(np.dot(u, gdir)))
    cos_posx.append(float(np.dot(u, posx)))
  moved = len(cos_goal)
  return {
      'frac_object_moved': float(moved / max(1, len(eps))),
      'cos_disp_goal_mean': float(np.mean(cos_goal)) if moved else None,
      'cos_disp_posx_mean': float(np.mean(cos_posx)) if moved else None,
      'shortcut_flag': bool(moved and np.mean(cos_posx) >= np.mean(cos_goal)),
  }


# --------------------------------------------------------------------------- #
# top-level
# --------------------------------------------------------------------------- #
def evaluate_report(env_name, ckpt, episodes=100, seed=123, config=None,
                    random_episodes=None, oracle_episodes=None):
  config = config or Config(env_name=env_name)
  config.env_name = env_name
  env = envs_mod.make_env(env_name, config, seed=seed)
  greedy, step = _load_greedy(env_name, ckpt, config)
  rng = np.random.default_rng(seed)

  eps = _rollout(env, greedy, episodes, rng)
  rand_eps = _rollout(env, None, random_episodes or episodes, rng)
  orac_eps = _rollout(env, _oracle_action, oracle_episodes or episodes, rng)

  rep = {
      'env_name': env_name, 'ckpt': ckpt, 'step': int(step),
      'episodes': episodes,
      'trained': _agg(eps),
      'random': _agg(rand_eps),
      'oracle': _agg(orac_eps),
      'displacement_probe': displacement_probe(eps),
  }
  if getattr(env, '_cone_dir', False):
    rep['success_by_cone_bin'] = cone_bins(eps, env._cone_hw)
  if getattr(env, '_neutral_dir', False):
    rep['success_by_quadrant'] = quadrants(eps)
    q = rep['success_by_quadrant']
    px = q['+x (toward gripper-forward)']['success']
    nx = q['-x (behind gripper)']['success']
    rep['plus_minus_x_asymmetry'] = {
        'plus_x': px, 'minus_x': nx,
        'gap': (px - nx) if (px is not None and nx is not None) else None}
  return rep


def print_report(rep):
  def line(k, v):
    print(f'   {k:34s}: {v}')
  print('=' * 70)
  print(f'EASYPUSH REPORT  {rep["env_name"]}  (ckpt step {rep["step"]}, '
        f'{rep["episodes"]} eps)')
  print('=' * 70)
  print('[1] SUCCESS + OBJECT-GOAL DISTANCE')
  for who in ('trained', 'random', 'oracle'):
    a = rep[who]
    line(who, f'success={a["success"]:.3f}  final_dist={a["final_dist"]:.3f}  '
              f'min_dist={a["min_dist"]:.3f}  (n={a["n"]})')
  if 'success_by_cone_bin' in rep:
    print('\n[2] SUCCESS BY CONE BIN (goal angle around +x)')
    for k, v in rep['success_by_cone_bin'].items():
      line(k, f'success={v["success"]}  (n={v["n"]})')
  if 'success_by_quadrant' in rep:
    print('\n[2] SUCCESS BY GOAL QUADRANT')
    for k, v in rep['success_by_quadrant'].items():
      line(k, f'success={v["success"]}  (n={v["n"]})')
    a = rep['plus_minus_x_asymmetry']
    print(f'   +x vs -x asymmetry: +x={a["plus_x"]}  -x={a["minus_x"]}  '
          f'gap={a["gap"]}')
  print('\n[3] DISPLACEMENT-DIRECTION PROBE')
  d = rep['displacement_probe']
  line('frac_object_moved', f'{d["frac_object_moved"]:.3f}')
  line('cos(disp, goal dir)', d['cos_disp_goal_mean'])
  line('cos(disp, +x world)', d['cos_disp_posx_mean'])
  line('fixed-+x shortcut?', d['shortcut_flag'])
  print('=' * 70)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--env_name', required=True)
  p.add_argument('--ckpt', required=True)
  p.add_argument('--episodes', type=int, default=100)
  p.add_argument('--seed', type=int, default=123)
  p.add_argument('--out', default=None)
  args = p.parse_args()
  rep = evaluate_report(args.env_name, args.ckpt, episodes=args.episodes,
                        seed=args.seed)
  print_report(rep)
  if args.out:
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(rep, open(args.out, 'w'), indent=2)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
  main()
