"""Step 1 of the seed-instability diagnosis: where is the actor stuck at the fork?

Read-only on the sealed query-coverage checkpoints (arm C, one successful and
one failed learner seed).  At the same fork states and the same canonical
task goal, for each seed and each root:

  * the actor's deterministic action tanh(loc) and 1024 sampled actions;
  * the critic score q(s, a, g) = phi(s, a) . psi(g) over the whole 2-D action
    square, plus its value at the mode, at the samples, at DOWN [0,-1] and
    RIGHT [1,0], and its argmax;
  * the critic's pull at the mode, grad_a q, and the BC pull, i.e. where the
    replay actions taken at (near) this state sit relative to the mode;
  * ONE small actual update of a copy of the actor with (a) only the critic
    term, (b) only the BC term, (c) the full 0.05/0.95 loss -- both on real
    training batches from the C replay (the effect at the roots through the
    shared parameters) and on a local batch made of the roots themselves --
    reporting how tanh(loc) at the roots actually moved, projected onto DOWN
    and RIGHT.  Plain SGD steps on a copy; the checkpoints are untouched and
    no environment step is taken.

Outputs: per-seed action-score maps (PNG), per-root numbers (JSON) and a
REPORT.md under outputs/pointmaze_actor_fork_diagnosis_v1/.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
OUT = ROOT / 'outputs' / 'pointmaze_actor_fork_diagnosis_v1'
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
DOWN = np.array([0.0, -1.0], np.float32)
RIGHT = np.array([1.0, 0.0], np.float32)
GRID = 61
N_SAMPLES = 1024
BC_COEF = 0.05
LOCAL_RADIUS = 0.15          # xy radius for "replay actions taken near the root"
SGD_LR = 3e-4                # the actor learning rate of the sealed recipe
G_BATCHES, BATCH = 10, 256   # one training step = 10 SGD steps on 256-batches

SEEDS = {
    # arm C final checkpoints; seed 1 (the 100% seed) is not on disk -- only a
    # truncated B copy and the JSONs came back from the remote node.
    0: SEALED / 'crl' / 'C' / 'final.pkl',
    2: SEALED / 'cross_seeds' / 'seed_2' / 'crl' / 'C' / 'final.pkl',
}
SEED_LABEL = {0: 'seed 0 (mode success 0.845, lower 0.78)',
              2: 'seed 2 (mode success 0.305, lower 0.00)'}


def plain(v):
  if isinstance(v, dict):
    return {str(k): plain(x) for k, x in v.items()}
  if isinstance(v, (list, tuple)):
    return [plain(x) for x in v]
  if isinstance(v, np.ndarray):
    return v.tolist()
  if isinstance(v, np.generic):
    return v.item()
  return v


def sector(a):
  a = np.asarray(a)
  down = (np.abs(a[..., 0]) <= 0.35) & (a[..., 1] <= -0.75)
  right = (a[..., 0] >= 0.75) & (np.abs(a[..., 1]) <= 0.35)
  return down, right


def make_network():
  from crl import networks
  return networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)


def crl_config():
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0',
      offline_dataset=str(SEALED / 'replay_C.npz'),
      obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
      start_index=0, end_index=-1, max_number_of_steps=30000,
      fail_bank_path='', fail_neg_alpha=0.0, obs_norm_mode='',
      obs_norm_z_scale=0.0, anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=BC_COEF, random_goals=0.5, entropy_coefficient=0.0,
      target_entropy=0.0, batch_size=BATCH, repr_dim=64,
      hidden_layer_sizes=(256, 256), discount=0.95, learning_rate=3e-4,
      actor_learning_rate=SGD_LR, num_sgd_steps_per_step=G_BATCHES,
      num_actors=0, guard_abort=True, jit=True, seed=0)


class Actor:
  """Thin wrapper: policy/critic functions of one checkpoint."""

  def __init__(self, nets, policy_params, q_params):
    import jax
    import jax.numpy as jnp
    self.nets, self.policy_params, self.q_params = nets, policy_params, q_params

    @jax.jit
    def dist(params, obs):
      return nets.policy_network.apply(params, obs)

    @jax.jit
    def score(q_params, obs, action):
      # phi, psi: [B, repr_dim, n_critics]; single critic here -> [B]
      phi, psi = nets.representation_network.apply(q_params, obs, action)
      return jnp.sum(phi * psi, axis=1)[:, 0]

    @jax.jit
    def score_grad(q_params, obs, action):
      f = lambda a: jnp.sum(score(q_params, obs, a))
      return jax.grad(f)(action)

    @jax.jit
    def samples(params, obs, key):
      keys = jax.random.split(key, N_SAMPLES)
      p = nets.policy_network.apply(params, obs)
      return jax.vmap(lambda k: nets.sample(p, k))(keys)   # [N, B, 2]

    self._dist, self._score, self._score_grad, self._samples = (
        dist, score, score_grad, samples)

  def mode(self, obs, params=None):
    p = self._dist(self.policy_params if params is None else params, obs)
    return np.tanh(np.asarray(p.loc)), np.asarray(p.scale)

  def score(self, obs, action):
    return np.asarray(self._score(self.q_params, obs, action))

  def score_grad(self, obs, action):
    return np.asarray(self._score_grad(self.q_params, obs, action))

  def samples(self, obs, key):
    return np.asarray(self._samples(self.policy_params, obs, key))


def actor_term_grads(nets, cfg, policy_params, q_params, transitions, key):
  """Gradients of the critic term, the BC term and the full actor loss.

  Mirrors crl/losses.py actor_loss with random_goals 0.5 and alpha 0.
  """
  import jax
  import jax.numpy as jnp
  obs_dim = cfg.obs_dim

  def parts(params):
    obs = transitions.observation
    state, goal = obs[:, :obs_dim], obs[:, obs_dim:]
    new_state = jnp.concatenate([state, state], axis=0)
    new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
    orig_action = jnp.concatenate([transitions.action, transitions.action], 0)
    new_obs = jnp.concatenate([new_state, new_goal], axis=1)
    dist = nets.policy_network.apply(params, new_obs)
    action = nets.sample(dist, key)
    q = nets.q_network.apply(q_params, new_obs, action)
    q_term = -jnp.diag(q)
    bc_nll = -nets.log_prob(dist, orig_action)
    return jnp.mean(q_term), jnp.mean(bc_nll)

  g_critic = jax.grad(lambda p: parts(p)[0])(policy_params)
  g_bc = jax.grad(lambda p: parts(p)[1])(policy_params)
  g_full = jax.grad(lambda p: (1 - cfg.bc_coef) * parts(p)[0]
                    + cfg.bc_coef * parts(p)[1])(policy_params)
  return {'critic': g_critic, 'bc': g_bc, 'full': g_full}


def sgd(params, grads, lr):
  import jax
  return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def tree_norm(t):
  import jax
  return float(np.sqrt(sum(float(np.sum(np.asarray(x) ** 2))
                           for x in jax.tree_util.tree_leaves(t))))


def make_env():
  from crl import envs as envs_mod
  from crl.config import Config
  name = 'point_two_route_swamp_windy_f4_v0'
  env = envs_mod.make_env(name, Config(env_name=name), seed=0)
  env.reset()
  env._action_noise = 0.0                      # noise-free kinematics only
  return env


def physics_outcome(env, physical_xy, actions, steps=8):
  """Route each constant action leads to from the root, noise-free.

  'lower' once y < 2 (the safe lower corridor), 'shortcut' once x >= 2.5 with
  y >= 3 (the hazard corridor), else 'other'.  Wind bits and death are
  ignored: this is the deterministic wall geometry only.
  """
  out = np.zeros(len(actions), np.int8)      # 0 other, 1 lower, 2 shortcut
  for k, a in enumerate(actions):
    env.reset()
    env._dead = False
    env.state = np.array(physical_xy, float)
    for _ in range(steps):
      env.step(np.array(a, float))
      x, y = env.state
      if y < 2.0:
        out[k] = 1
        break
      if x >= 2.5 and y >= 3.0:
        out[k] = 2
        break
  return out


def local_replay_actions(replay, roots):
  """Replay actions taken within LOCAL_RADIUS (xy) of each root."""
  xy = replay['obs'][:, :50, :2].reshape(-1, 2)
  act = replay['act'][:, :50, :].reshape(-1, 2)
  out = []
  for s in roots:
    d = np.linalg.norm(xy - s[:2], axis=1)
    sel = d <= LOCAL_RADIUS
    a = act[sel]
    down, right = sector(a) if len(a) else (np.zeros(0, bool),) * 2
    out.append({'n': int(sel.sum()),
                'mean_action': (a.mean(0) if len(a) else np.full(2, np.nan)),
                'down_fraction': (float(down.mean()) if len(a) else None),
                'right_fraction': (float(right.mean()) if len(a) else None),
                'actions': a})
  return out


def one_step_effects(actor, nets, cfg, buffer, roots_obs, local_bc_batch,
                     key):
  """Delta of tanh(loc) at the roots after one small SGD step per term."""
  import jax
  from crl import losses as losses_mod
  mode0, _ = actor.mode(roots_obs)
  results = {}
  # (i) real training batches: G x B from the frozen C replay
  batches = [buffer.sample(BATCH) for _ in range(G_BATCHES)]
  acc = None
  for i, b in enumerate(batches):
    tr = losses_mod.Transition(*[np.asarray(getattr(b, f))
                                 for f in losses_mod.Transition._fields])
    g = actor_term_grads(nets, cfg, actor.policy_params, actor.q_params,
                         tr, jax.random.fold_in(key, i))
    acc = g if acc is None else {k: jax.tree_util.tree_map(
        lambda a, b_: a + b_, acc[k], g[k]) for k in g}
  grads = {k: jax.tree_util.tree_map(lambda a: a / G_BATCHES, acc[k])
           for k in acc}
  results['training_batches'] = _apply_and_measure(actor, grads, roots_obs,
                                                   mode0)
  # (ii) local batch: the roots with the canonical goal (critic term) and the
  # replay actions taken near the roots (BC term)
  local = {}
  obs_dim = cfg.obs_dim
  import jax.numpy as jnp

  def critic_local(params):
    dist = nets.policy_network.apply(params, roots_obs)
    a = nets.sample(dist, key)
    q = nets.q_network.apply(actor.q_params, roots_obs, a)
    return -jnp.mean(jnp.diag(q))

  def bc_local(params):
    dist = nets.policy_network.apply(params, local_bc_batch['obs'])
    return -jnp.mean(nets.log_prob(dist, local_bc_batch['act']))

  local['critic'] = jax.grad(critic_local)(actor.policy_params)
  local['bc'] = jax.grad(bc_local)(actor.policy_params)
  local['full'] = jax.tree_util.tree_map(
      lambda c, b: (1 - BC_COEF) * c + BC_COEF * b, local['critic'],
      local['bc'])
  results['local_root_batch'] = _apply_and_measure(actor, local, roots_obs,
                                                   mode0)
  return results


def _apply_and_measure(actor, grads, roots_obs, mode0):
  out = {}
  for term, g in grads.items():
    gn = tree_norm(g)
    row = {'grad_norm': gn, 'steps': {}}
    for lr in (SGD_LR, 10 * SGD_LR):
      new_params = sgd(actor.policy_params, g, lr)
      mode1, _ = actor.mode(roots_obs, params=new_params)
      delta = mode1 - mode0
      q0 = actor.score(roots_obs, mode0)
      q1 = actor.score(roots_obs, mode1)
      row['steps'][f'lr_{lr:g}'] = {
          'delta_mode_mean': delta.mean(0),
          'delta_mode_norm_mean': float(np.linalg.norm(delta, axis=1).mean()),
          'proj_down_mean': float((delta @ DOWN).mean()),
          'proj_right_mean': float((delta @ RIGHT).mean()),
          'critic_score_change_mean': float((q1 - q0).mean()),
          'per_root_delta': delta,
      }
    # direction only: unit-norm parameter step of size 1e-3 * |theta| is not
    # needed; the two learning rates above show the direction is linear.
    out[term] = row
  return out


def analyse_seed(seed, path, nets, roots, root_names, replay_local, buffer,
                 cfg, local_bc_batch, key, outcome_maps, physical):
  import jax
  from crl import checkpoint
  step, state = checkpoint.load_checkpoint(path)
  actor = Actor(nets, state.policy_params, state.q_params)
  goal = np.broadcast_to(GOAL, roots.shape).astype(np.float32)
  obs = np.concatenate([roots, goal], axis=1).astype(np.float32)
  n = len(roots)
  mode, scale = actor.mode(obs)
  samples = actor.samples(obs, key)                      # [N_SAMPLES, n, 2]
  axis = np.linspace(-1, 1, GRID, dtype=np.float32)
  gx, gy = np.meshgrid(axis, axis, indexing='xy')
  grid = np.stack([gx.ravel(), gy.ravel()], axis=1)      # [GRID*GRID, 2]
  per_root = []
  score_maps = np.zeros((n, GRID, GRID), np.float32)
  for i in range(n):
    obs_rep = np.repeat(obs[i:i + 1], len(grid), axis=0)
    q_grid = actor.score(obs_rep, grid).reshape(GRID, GRID)
    score_maps[i] = q_grid
    q_mode = float(actor.score(obs[i:i + 1], mode[i:i + 1])[0])
    q_down = float(actor.score(obs[i:i + 1], DOWN[None])[0])
    q_right = float(actor.score(obs[i:i + 1], RIGHT[None])[0])
    s_i = samples[:, i]
    q_samples = actor.score(np.repeat(obs[i:i + 1], N_SAMPLES, 0), s_i)
    grad = actor.score_grad(obs[i:i + 1], mode[i:i + 1])[0]
    k = int(np.argmax(q_grid))
    argmax_a = grid[k]
    down_s, right_s = sector(s_i)
    loc_a = replay_local[i]
    # physics: which route does each grid action, the mode and the samples take
    om = outcome_maps[i].ravel()
    q_flat = q_grid.ravel()
    q_lower_max = float(q_flat[om == 1].max()) if (om == 1).any() else None
    q_short_max = float(q_flat[om == 2].max()) if (om == 2).any() else None
    env = make_env()
    mode_out = int(physics_outcome(env, physical[i], mode[i:i + 1])[0])
    samp_out = physics_outcome(env, physical[i], s_i[:256])
    names_out = {0: 'other', 1: 'lower', 2: 'shortcut'}
    bc_pull = (loc_a['mean_action'] - mode[i]) if loc_a['n'] else np.full(2, np.nan)
    # is there a local maximum near the mode that is not the global one?
    near = np.linalg.norm(grid - mode[i], axis=1) <= 0.25
    q_near_max = float(q_grid.ravel()[near].max())
    per_root.append({
        'root': root_names[i], 'state_xy': roots[i, :2],
        'mode_action': mode[i], 'policy_scale': scale[i],
        'mode_sector': ('down' if sector(mode[i:i + 1])[0][0] else
                        'right' if sector(mode[i:i + 1])[1][0] else 'other'),
        'samples_down_fraction': float(down_s.mean()),
        'samples_right_fraction': float(right_s.mean()),
        'samples_mean_action': s_i.mean(0),
        'q_mode': q_mode, 'q_samples_mean': float(q_samples.mean()),
        'q_down': q_down, 'q_right': q_right,
        'q_down_minus_right': q_down - q_right,
        'q_grid_max': float(q_grid.max()), 'q_grid_argmax_action': argmax_a,
        'q_grid_argmax_sector': ('down' if sector(argmax_a[None])[0][0] else
                                 'right' if sector(argmax_a[None])[1][0]
                                 else 'other'),
        'q_local_max_within_0p25_of_mode': q_near_max,
        'mode_trapped_by_local_ridge':
            bool(q_near_max > q_mode - 1e-6 and q_down > q_near_max + 0.05),
        'physics': {
            'mode_outcome': names_out[mode_out],
            'samples_lower_fraction': float((samp_out == 1).mean()),
            'samples_shortcut_fraction': float((samp_out == 2).mean()),
            'grid_lower_fraction': float((om == 1).mean()),
            'grid_shortcut_fraction': float((om == 2).mean()),
            'q_max_in_lower_region': q_lower_max,
            'q_max_in_shortcut_region': q_short_max,
            'q_argmax_region': names_out[int(om[k])],
            'lower_region_has_higher_max': (
                None if q_lower_max is None or q_short_max is None
                else bool(q_lower_max > q_short_max)),
        },
        'critic_grad_at_mode': grad,
        'critic_grad_proj_down': float(grad @ DOWN),
        'critic_grad_proj_right': float(grad @ RIGHT),
        'replay_actions_near_root': {
            'n': loc_a['n'], 'mean_action': loc_a['mean_action'],
            'down_fraction': loc_a['down_fraction'],
            'right_fraction': loc_a['right_fraction']},
        'bc_pull_towards_local_actions': bc_pull,
        'bc_pull_proj_down': (float(bc_pull @ DOWN) if loc_a['n'] else None),
        'bc_pull_proj_right': (float(bc_pull @ RIGHT) if loc_a['n'] else None),
    })
  one_step = one_step_effects(actor, nets, cfg, buffer, obs, local_bc_batch,
                              jax.random.fold_in(key, 7))
  summary = {
      'seed': seed, 'label': SEED_LABEL[seed], 'checkpoint': str(path),
      'step': int(step), 'n_roots': n,
      'mode_sector_counts': {
          s: int(sum(r['mode_sector'] == s for r in per_root))
          for s in ('down', 'right', 'other')},
      'argmax_sector_counts': {
          s: int(sum(r['q_grid_argmax_sector'] == s for r in per_root))
          for s in ('down', 'right', 'other')},
      'roots_q_down_gt_q_right': int(sum(r['q_down_minus_right'] > 0
                                         for r in per_root)),
      'roots_q_down_gt_q_mode': int(sum(r['q_down'] > r['q_mode']
                                        for r in per_root)),
      'roots_mode_trapped_by_local_ridge': int(sum(
          r['mode_trapped_by_local_ridge'] for r in per_root)),
      'physics_mode_outcome_counts': {
          o: int(sum(r['physics']['mode_outcome'] == o for r in per_root))
          for o in ('lower', 'shortcut', 'other')},
      'physics_argmax_region_counts': {
          o: int(sum(r['physics']['q_argmax_region'] == o for r in per_root))
          for o in ('lower', 'shortcut', 'other')},
      'roots_lower_region_max_gt_shortcut_region_max': int(sum(
          bool(r['physics']['lower_region_has_higher_max']) for r in per_root)),
      'mean_samples_lower_fraction': float(np.mean(
          [r['physics']['samples_lower_fraction'] for r in per_root])),
      'mean_samples_shortcut_fraction': float(np.mean(
          [r['physics']['samples_shortcut_fraction'] for r in per_root])),
      'mean_q_mode': float(np.mean([r['q_mode'] for r in per_root])),
      'mean_q_down': float(np.mean([r['q_down'] for r in per_root])),
      'mean_q_right': float(np.mean([r['q_right'] for r in per_root])),
      'mean_q_grid_max': float(np.mean([r['q_grid_max'] for r in per_root])),
      'mean_samples_down_fraction': float(np.mean(
          [r['samples_down_fraction'] for r in per_root])),
      'mean_samples_right_fraction': float(np.mean(
          [r['samples_right_fraction'] for r in per_root])),
      'mean_policy_scale': float(np.mean([r['policy_scale'] for r in per_root])),
      'mean_critic_grad_proj_down': float(np.mean(
          [r['critic_grad_proj_down'] for r in per_root])),
      'mean_critic_grad_proj_right': float(np.mean(
          [r['critic_grad_proj_right'] for r in per_root])),
      'mean_bc_pull_proj_down': float(np.nanmean(
          [r['bc_pull_proj_down'] for r in per_root
           if r['bc_pull_proj_down'] is not None])),
      'mean_bc_pull_proj_right': float(np.nanmean(
          [r['bc_pull_proj_right'] for r in per_root
           if r['bc_pull_proj_right'] is not None])),
      'one_step': one_step,
  }
  return summary, per_root, score_maps, mode, samples


def plot_seed(seed, per_root, score_maps, mode, samples, out_png,
              outcome_maps=None):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  n = len(per_root)
  cols = 4
  rows = int(np.ceil(n / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
  axes = np.atleast_2d(axes)
  vmin = min(float(m.min()) for m in score_maps)
  vmax = max(float(m.max()) for m in score_maps)
  for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    if i >= n:
      ax.axis('off')
      continue
    r = per_root[i]
    im = ax.imshow(score_maps[i], origin='lower', extent=[-1, 1, -1, 1],
                   cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal')
    ax.contour(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID),
               score_maps[i], levels=8, colors='white', linewidths=0.4,
               alpha=0.6)
    if outcome_maps is not None:
      # physics: hatch the actions that lead into the shortcut corridor and
      # outline the ones that drop into the lower corridor
      ax.contourf(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID),
                  (outcome_maps[i] == 2).astype(float), levels=[0.5, 1.5],
                  colors='none', hatches=['////'], alpha=0)
      ax.contour(np.linspace(-1, 1, GRID), np.linspace(-1, 1, GRID),
                 (outcome_maps[i] == 1).astype(float), levels=[0.5],
                 colors='cyan', linewidths=1.2)
    s = samples[:, i]
    ax.scatter(s[:, 0], s[:, 1], s=3, c='white', alpha=0.15, linewidths=0)
    ax.scatter([DOWN[0]], [DOWN[1]], marker='v', s=90, c='cyan',
               edgecolors='black', label=f'down q={r["q_down"]:.2f}')
    ax.scatter([RIGHT[0]], [RIGHT[1]], marker='>', s=90, c='orange',
               edgecolors='black', label=f'right q={r["q_right"]:.2f}')
    ax.scatter([r['mode_action'][0]], [r['mode_action'][1]], marker='*',
               s=160, c='red', edgecolors='black',
               label=f'mode q={r["q_mode"]:.2f}')
    g = r['critic_grad_at_mode']
    gn = np.linalg.norm(g) + 1e-9
    ax.arrow(r['mode_action'][0], r['mode_action'][1], 0.3 * g[0] / gn,
             0.3 * g[1] / gn, color='red', width=0.012, length_includes_head=True)
    bp = r['bc_pull_towards_local_actions']
    if np.all(np.isfinite(bp)):
      ax.arrow(r['mode_action'][0], r['mode_action'][1], bp[0], bp[1],
               color='magenta', width=0.008, length_includes_head=True)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ph = r['physics']
    ax.set_title(f'{r["root"]}  xy=({r["state_xy"][0]:.2f},{r["state_xy"][1]:.2f})'
                 f'  mode->{ph["mode_outcome"]}'
                 f'\nsamples lower {ph["samples_lower_fraction"]:.2f} / '
                 f'shortcut {ph["samples_shortcut_fraction"]:.2f}; q-argmax '
                 f'in {ph["q_argmax_region"]}', fontsize=8)
    ax.legend(fontsize=6, loc='upper left')
  fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='critic score')
  fig.suptitle(f'{SEED_LABEL[seed]} -- critic score over the action square; '
               'star = actor mode, white = samples, red arrow = grad_a q at the '
               'mode, magenta = replay-action pull; cyan outline = actions that '
               'drop into the LOWER corridor, hatched = actions that enter the '
               'SHORTCUT corridor (noise-free physics from this root)',
               fontsize=10)
  fig.savefig(out_png, dpi=110, bbox_inches='tight')
  plt.close(fig)


def write_report(summaries, out_md):
  lines = ['# Step 1: where the query-coverage actor sits at the fork', '',
           'Read-only on the sealed arm-C checkpoints, 16 held-out fork roots, '
           'canonical goal (8.5, 3.5). Seed 1 (the 100% seed) has no '
           'checkpoint on disk, so the successful seed here is seed 0.', '',
           '| seed | mode sector (down/right/other) | argmax sector | roots '
           'q_down>q_right | roots q_down>q_mode | trapped-by-ridge roots | '
           'mean q mode / down / right / grid-max | samples down / right | '
           'scale |', '|---|---|---|---:|---:|---:|---|---|---:|']
  for s in summaries:
    ms, ag = s['mode_sector_counts'], s['argmax_sector_counts']
    lines.append(
        f'| {s["seed"]} | {ms["down"]}/{ms["right"]}/{ms["other"]} | '
        f'{ag["down"]}/{ag["right"]}/{ag["other"]} | '
        f'{s["roots_q_down_gt_q_right"]}/16 | {s["roots_q_down_gt_q_mode"]}/16 '
        f'| {s["roots_mode_trapped_by_local_ridge"]}/16 | '
        f'{s["mean_q_mode"]:.3f} / {s["mean_q_down"]:.3f} / '
        f'{s["mean_q_right"]:.3f} / {s["mean_q_grid_max"]:.3f} | '
        f'{s["mean_samples_down_fraction"]:.2f} / '
        f'{s["mean_samples_right_fraction"]:.2f} | '
        f'{s["mean_policy_scale"]:.3f} |')
  lines += ['', '## What the actions physically do (noise-free kinematics '
            'from each root, 8 steps)', '',
            '| seed | mode outcome (lower/shortcut/other) | q-argmax region | '
            'roots where max q in lower region > max q in shortcut region | '
            'samples lower / shortcut |', '|---|---|---|---:|---|']
  for s in summaries:
    mo, ar = s['physics_mode_outcome_counts'], s['physics_argmax_region_counts']
    lines.append(f'| {s["seed"]} | {mo["lower"]}/{mo["shortcut"]}/{mo["other"]} '
                 f'| {ar["lower"]}/{ar["shortcut"]}/{ar["other"]} | '
                 f'{s["roots_lower_region_max_gt_shortcut_region_max"]}/16 | '
                 f'{s["mean_samples_lower_fraction"]:.2f} / '
                 f'{s["mean_samples_shortcut_fraction"]:.2f} |')
  lines += ['', '## Pulls at the mode (mean over roots; projection on DOWN '
            '(0,-1) and RIGHT (1,0))', '',
            '| seed | grad_a q . down | grad_a q . right | BC pull . down | '
            'BC pull . right |', '|---|---:|---:|---:|---:|']
  for s in summaries:
    lines.append(f'| {s["seed"]} | {s["mean_critic_grad_proj_down"]:+.3f} | '
                 f'{s["mean_critic_grad_proj_right"]:+.3f} | '
                 f'{s["mean_bc_pull_proj_down"]:+.3f} | '
                 f'{s["mean_bc_pull_proj_right"]:+.3f} |')
  lines += ['', '## One actual SGD step on an actor copy: movement of '
            'tanh(loc) at the 16 roots', '',
            'critic = only the critic term, bc = only the BC term, full = '
            '0.95 critic + 0.05 bc. "training batches" = 10 x 256 relabeled '
            'transitions from the C replay (the real training signal, acting '
            'on the roots through shared parameters); "local" = the roots '
            'themselves (critic) and the replay actions within 0.15 of them '
            '(bc). Values are mean displacement projected on DOWN / RIGHT '
            'and the mean change of the critic score at the new mode.', '',
            '| seed | batch | term | lr | |grad| | d.down | d.right | |d| | '
            'dq(mode) |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
  for s in summaries:
    for batch, terms in s['one_step'].items():
      for term, row in terms.items():
        for lr, x in row['steps'].items():
          lines.append(
              f'| {s["seed"]} | {batch} | {term} | {lr[3:]} | '
              f'{row["grad_norm"]:.3f} | {x["proj_down_mean"]:+.4f} | '
              f'{x["proj_right_mean"]:+.4f} | {x["delta_mode_norm_mean"]:.4f} '
              f'| {x["critic_score_change_mean"]:+.4f} |')
  Path(out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print('\n'.join(lines))


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--roots', choices=('heldout', 'construction'),
                  default='heldout')
  ap.add_argument('--seed-key', type=int, default=2026091601)
  args = ap.parse_args(argv)
  import jax
  from crl import offline_audit
  OUT.mkdir(parents=True, exist_ok=True)
  if args.roots == 'heldout':
    with np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                 / 'root_selection.npz', allow_pickle=False) as d:
      roots = d['state'].astype(np.float32)
    names = [f'root {i}' for i in range(len(roots))]
  else:
    with np.load(SEALED / 'construction_roots.npz', allow_pickle=False) as d:
      roots = d['state'].astype(np.float32)
    names = [f'croot {i}' for i in range(len(roots))]
  with np.load(SEALED / 'replay_C.npz', allow_pickle=False) as d:
    replay = {'obs': d['obs'], 'act': d['act']}
  replay_local = local_replay_actions(replay, roots)
  # local BC batch: every replay transition within the radius of any root
  xy = replay['obs'][:, :50, :].reshape(-1, 16)
  act = replay['act'][:, :50, :].reshape(-1, 2)
  sel = np.zeros(len(xy), bool)
  for s in roots:
    sel |= np.linalg.norm(xy[:, :2] - s[:2], axis=1) <= LOCAL_RADIUS
  local_bc_batch = {'obs': np.concatenate(
      [xy[sel, :8], np.broadcast_to(GOAL, (int(sel.sum()), 8))], 1
      ).astype(np.float32), 'act': act[sel].astype(np.float32)}
  print(f'{args.roots} roots: {len(roots)}; replay transitions within '
        f'{LOCAL_RADIUS} of any root: {int(sel.sum())}', flush=True)
  if args.roots == 'heldout':
    with np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                 / 'root_selection.npz', allow_pickle=False) as d:
      physical = d['physical_state'].astype(float)
  else:
    physical = roots[:, :2].astype(float)
  axis = np.linspace(-1, 1, GRID, dtype=np.float32)
  gx, gy = np.meshgrid(axis, axis, indexing='xy')
  grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
  env = make_env()
  outcome_maps = np.stack([physics_outcome(env, physical[i], grid)
                           .reshape(GRID, GRID) for i in range(len(roots))])
  print('physics outcome maps computed: lower fraction of the action square '
        f'{(outcome_maps == 1).mean():.2f}, shortcut '
        f'{(outcome_maps == 2).mean():.2f}', flush=True)
  nets = make_network()
  cfg = crl_config()
  buffer, _ = offline_audit.build_offline_buffer(cfg.offline_dataset, cfg)
  key = jax.random.PRNGKey(args.seed_key)
  summaries = []
  for seed, path in SEEDS.items():
    if not path.exists():
      print(f'seed {seed}: missing {path}', flush=True)
      continue
    summary, per_root, maps, mode, samples = analyse_seed(
        seed, path, nets, roots, names, replay_local, buffer, cfg,
        local_bc_batch, key, outcome_maps, physical)
    summaries.append(summary)
    tag = f'seed{seed}_{args.roots}'
    with open(OUT / f'{tag}.json', 'w', encoding='utf-8') as f:
      json.dump(plain({'summary': summary, 'per_root': per_root}), f,
                indent=2)
    np.savez_compressed(OUT / f'{tag}_score_maps.npz', score_maps=maps,
                        mode=mode, samples=samples, roots=roots,
                        outcome_maps=outcome_maps,
                        grid_axis=np.linspace(-1, 1, GRID))
    if args.roots == 'heldout':
      plot_seed(seed, per_root, maps, mode, samples, OUT / f'{tag}.png',
                outcome_maps)
    print(f'seed {seed}: mode sectors {summary["mode_sector_counts"]}, '
          f'q_down>q_right on {summary["roots_q_down_gt_q_right"]}/16, '
          f'trapped {summary["roots_mode_trapped_by_local_ridge"]}/16',
          flush=True)
  write_report(summaries, OUT / f'REPORT_{args.roots}.md')
  return 0


if __name__ == '__main__':
  sys.exit(main())
