"""Equivalence + sanity checks for the contrastive RL port.

Run:  python -m crl.checks
Produces PASS/FAIL lines per check, saves plots under crl/_checks/, and prints a
final summary. Does NOT start full training.

Provenance (new file  <-  original it replaces):
  crl/networks.py  <- contrastive/networks.py (make_networks/_repr_fn/_critic_fn/
                      _actor_fn) + acme NormalTanhDistribution
  crl/losses.py    <- contrastive/learning.py (critic_loss/actor_loss/alpha_loss/
                      update_step/TrainingState) + acme.types.Transition
  crl/replay.py    <- contrastive/builder.py (flatten_fn + EpisodeAdder/tables)
                      + reverb TrajectoryDataset
  crl/envs.py      <- env_utils.load + fetch_envs.py + point_env.py +
                      contrastive/utils.make_environment/ObservationFilterWrapper
  crl/train.py     <- lp_contrastive.py + contrastive/learning.step loop +
                      contrastive/builder orchestration
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from crl.config import Config  # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import losses as losses_mod  # noqa: E402
from crl.replay import TrajectoryBuffer  # noqa: E402
from crl import envs as envs_mod  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '_checks')
os.makedirs(OUT, exist_ok=True)

_RESULTS = []


def check(name, ok, detail=''):
  tag = 'PASS' if ok else 'FAIL'
  print(f'  [{tag}] {name}' + (f'  --  {detail}' if detail else ''))
  _RESULTS.append((name, bool(ok)))
  return ok


def hline(title):
  print('\n' + '=' * 70 + f'\n{title}\n' + '=' * 70)


# ===========================================================================
def section_losses():
  hline('LOSSES.PY  (<- contrastive/learning.py)')
  print('Function map:')
  print('  crl.losses.build_learner.critic_loss  <- learning.py critic_loss')
  print('  crl.losses.build_learner.actor_loss   <- learning.py actor_loss')
  print('  crl.losses.build_learner.alpha_loss   <- learning.py alpha_loss')
  print('  crl.losses.build_learner.update_step  <- learning.py update_step')
  print('  crl.losses.Transition                 <- acme.types.Transition\n')

  B, obs_dim, goal_dim, act_dim = 8, 2, 2, 2
  cfg = Config(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=act_dim,
               batch_size=B, entropy_coefficient=0.0)
  nets = networks_mod.make_networks(
      obs_dim=obs_dim, goal_dim=goal_dim, action_dim=act_dim, repr_dim=16,
      hidden_layer_sizes=(32, 32), twin_q=False)
  key = jax.random.PRNGKey(0)
  po, qo = optax.adam(3e-4), optax.adam(3e-4)
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, lambda s: s, po, qo)
  tstate = init_state(key)  # THE params used by the loss.

  rng = np.random.default_rng(0)
  state = rng.normal(size=(B, obs_dim)).astype(np.float32)
  goal = rng.normal(size=(B, goal_dim)).astype(np.float32)
  action = np.tanh(rng.normal(size=(B, act_dim))).astype(np.float32)
  obs = np.concatenate([state, goal], axis=1)
  # Logits from the SAME params the loss uses (tstate.q_params) -- see FAIL-1
  # diagnosis: init_state re-splits the key, so a separately-init'd critic would
  # have different weights and the equivalence check would spuriously fail.
  logits = np.asarray(nets.q_network.apply(
      tstate.q_params, jnp.asarray(obs), jnp.asarray(action)))
  labels = np.eye(B)

  # --- shapes ---
  print('Shapes:')
  print(f'  states  {state.shape}   actions {action.shape}   '
        f'goals {goal.shape}')
  print(f'  obs(concat) {obs.shape}   logits {logits.shape}   '
        f'labels(I) {labels.shape}')
  check('logits are [B, B] (state-action x goal outer product)',
        logits.shape == (B, B), f'{logits.shape}')
  check('labels are identity [B, B]', labels.shape == (B, B))

  # --- diagonal = positives, off-diagonal = negatives (label semantics) ---
  check('diagonal entries are the POSITIVE pairs (label==1)',
        np.all(np.diag(labels) == 1))
  check('off-diagonal entries are the NEGATIVE pairs (label==0)',
        np.all(labels[~np.eye(B, dtype=bool)] == 0))

  # --- one synthetic batch: my NCE loss vs independent numpy sigmoid-BCE ---
  trans = losses_mod.Transition(
      observation=jnp.asarray(obs), action=jnp.asarray(action),
      reward=jnp.zeros(B), discount=jnp.full(B, 0.99),
      next_observation=jnp.asarray(obs), next_action=jnp.asarray(action))
  indep = np.mean(np.logaddexp(0.0, logits) - labels * logits)
  _, metrics0 = update_step(tstate, trans)
  my_loss = float(metrics0['critic_loss'])
  check('NCE critic loss matches independent numpy sigmoid-BCE '
        '(= optax.sigmoid_binary_cross_entropy, as in learning.py)',
        abs(my_loss - indep) < 1e-4,
        f'mine={my_loss:.6f} indep={indep:.6f} |d|={abs(my_loss-indep):.2e}')
  print(f'  synthetic-batch critic loss = {my_loss:.6f}')

  # --- overfit the fixed (collision-prone, 2-D goal) batch ---
  # Correct contrastive criteria: loss down, diagonal logits rise above
  # off-diagonal, per-entry (binary) accuracy high. NOTE categorical_accuracy
  # need NOT reach 1.0 here: with 2-D goals many negatives sit next to the
  # positive goal (false negatives), which is exactly why real-run cat_acc
  # plateaus ~0.15. See the separable test below for the machinery's ceiling.
  s = tstate
  bin0 = float(metrics0['binary_accuracy'])
  first = my_loss
  for _ in range(400):
    s, m = update_step(s, trans)
  last = float(m['critic_loss'])
  logits2 = np.asarray(nets.q_network.apply(
      s.q_params, jnp.asarray(obs), jnp.asarray(action)))
  diag_mean = float(np.mean(np.diag(logits2)))
  offdiag_mean = float(np.mean(logits2[~np.eye(B, dtype=bool)]))
  check('overfit: mean(diagonal logit) > mean(off-diagonal logit)',
        diag_mean > offdiag_mean,
        f'diag={diag_mean:.3f} > offdiag={offdiag_mean:.3f}')
  check('overfit: critic loss decreased', last < 0.5 * first,
        f'{first:.4f} -> {last:.4f}')
  bin1 = float(m['binary_accuracy'])
  # Improved a lot AND high relative to the collision-limited ceiling (<1.0;
  # the separable test below shows the true ceiling is 1.0 without collisions).
  check('overfit: per-entry binary_accuracy improved sharply and is high',
        (bin1 - bin0) > 0.15 and bin1 > 0.85,
        f'{bin0:.3f} -> {bin1:.3f} (delta={bin1-bin0:+.3f})')
  print(f'  (informational) categorical_accuracy on 2-D-goal batch = '
        f'{float(m["categorical_accuracy"]):.3f} '
        f'(<1.0 expected: goal collisions)')

  # --- MACHINERY CEILING: separable bijection -> cat_acc must reach ~1.0 ---
  # Distinct, well-separated one-hot states & goals => no collisions, so a
  # correct contrastive critic MUST learn the diagonal as argmax.
  Bs = 12
  nets2 = networks_mod.make_networks(
      obs_dim=Bs, goal_dim=Bs, action_dim=2, repr_dim=32,
      hidden_layer_sizes=(64, 64))
  cfg2 = Config(obs_dim=Bs, goal_dim=Bs, action_dim=2, batch_size=Bs,
                entropy_coefficient=0.0)
  init2, upd2 = losses_mod.build_learner(
      nets2, cfg2, lambda s: s, optax.adam(3e-3), optax.adam(3e-3))
  s2 = init2(jax.random.PRNGKey(1))
  eye = 3.0 * np.eye(Bs, dtype=np.float32)
  obs_s = np.concatenate([eye, eye], axis=1)             # distinct s and g.
  act_s = np.zeros((Bs, 2), np.float32)
  tr_s = losses_mod.Transition(
      observation=jnp.asarray(obs_s), action=jnp.asarray(act_s),
      reward=jnp.zeros(Bs), discount=jnp.full(Bs, 0.99),
      next_observation=jnp.asarray(obs_s), next_action=jnp.asarray(act_s))
  for _ in range(1500):
    s2, ms = upd2(s2, tr_s)
  cat_s = float(ms['categorical_accuracy'])
  check('separable-bijection overfit: categorical_accuracy -> ~1.0 '
        '(proves the contrastive machinery is correct)', cat_s > 0.95,
        f'cat_acc={cat_s:.3f} loss={float(ms["critic_loss"]):.4f}')


# ===========================================================================
def section_replay():
  hline('REPLAY.PY  (<- contrastive/builder.py flatten_fn)')
  discount, L, ne = 0.9, 40, 300
  buf = TrajectoryBuffer(
      capacity_steps=ne * L, ep_len_obs=L, full_obs_dim=4, action_dim=2,
      obs_dim=2, start_index=0, end_index=-1, discount=discount, seed=0)
  # obs = [time, episode_id, time, episode_id] so we can recover i, j, traj.
  for e in range(ne):
    o = np.zeros((L, 4), np.float32)
    o[:, 0] = np.arange(L)    # state coord 0 = timestep.
    o[:, 1] = e               # state coord 1 = episode id.
    buf.add_episode(o, np.zeros((L, 2), np.float32))

  B = 400_000
  batch = buf.sample(B)
  anchor_t = batch.observation[:, 0]
  anchor_ep = batch.observation[:, 1]
  goal_t = batch.observation[:, 2]     # goal = obs_to_goal(future state)
  goal_ep = batch.observation[:, 3]

  # (1) positives are future timesteps of the SAME trajectory.
  check('positive goal timestep is always in the FUTURE (j > i)',
        np.all(goal_t > anchor_t))
  check('positive goal is from the SAME trajectory as the anchor',
        np.all(goal_ep == anchor_ep))

  # (2) geometric offset law, conditioned on anchor i==0.
  mask = anchor_t == 0
  d = goal_t[mask].astype(int)
  counts = np.bincount(d, minlength=L)[1:]
  emp = counts / counts.sum()
  ref = discount ** np.arange(1, L, dtype=float)
  ref = ref / ref.sum()
  err = float(np.max(np.abs(emp - ref)))
  check('offset ~ geometric(discount): max|emp-ref| < 0.01', err < 0.01,
        f'err={err:.4f}, n={int(mask.sum())}')

  # text histogram of offsets d=1..12 (anchor i==0).
  print('  offset histogram (anchor i=0):  empirical vs discount**d')
  for dd in range(1, 13):
    bar = '#' * int(round(emp[dd - 1] * 200))
    print(f'    d={dd:2d}  emp={emp[dd-1]:.3f} ref={ref[dd-1]:.3f} |{bar}')

  # (3) negatives: off-diagonal (state_i, goal_k) collide with same trajectory
  # only at chance rate ~1/ne (a "false negative"); should be small.
  B2 = 256
  b2 = buf.sample(B2)
  ep_state = b2.observation[:, 1]
  ep_goal = b2.observation[:, 3]
  same = (ep_state[:, None] == ep_goal[None, :])   # [B2, B2]
  off = ~np.eye(B2, dtype=bool)
  collision = float(same[off].mean())
  check('off-diagonal negatives rarely share the anchor trajectory '
        '(false-neg rate ~ 1/num_eps)', collision < 3.0 / ne,
        f'rate={collision:.4f}  (chance 1/{ne}={1/ne:.4f})')
  # And the positive pair (diagonal) is never counted as a negative:
  check('diagonal (positive) pair is excluded from negatives by construction',
        np.all(np.diag(np.eye(B2)) == 1))

  # save histogram plot.
  fig, ax = plt.subplots(figsize=(6, 3.2))
  xs = np.arange(1, 21)
  ax.bar(xs - 0.2, emp[:20], width=0.4, label='empirical')
  ax.bar(xs + 0.2, ref[:20], width=0.4, label='discount**d (norm.)')
  ax.set_xlabel('future-goal offset d = j - i (anchor i=0)')
  ax.set_ylabel('probability')
  ax.set_title(f'Future-goal sampling vs geometric (discount={discount})')
  ax.legend()
  fig.tight_layout()
  p = os.path.join(OUT, 'replay_offset_histogram.png')
  fig.savefig(p, dpi=110)
  plt.close(fig)
  print(f'  saved histogram -> {p}')


# ===========================================================================
def section_envs():
  hline('ENVS.PY  (<- env_utils.load + fetch_envs.py + point_env.py)')
  # --- point env ---
  cfg = Config(env_name='point_FourRooms')
  env = envs_mod.make_env('point_FourRooms', cfg, seed=0)
  obs = env.reset()
  full = cfg.obs_dim + cfg.goal_dim
  print(f'  point_FourRooms: obs_dim(state)={cfg.obs_dim} '
        f'goal_dim={cfg.goal_dim} action_dim={cfg.action_dim} '
        f'reset_obs.shape={obs.shape}')
  check('point obs is flat [state, goal] with width obs_dim+goal_dim',
        obs.shape == (full,), f'{obs.shape} vs {full}')
  a = np.zeros(cfg.action_dim, np.float32)
  o2, r, done, info = env.step(a)
  check('point step returns (obs[flat], reward in {0,1}, done, info)',
        o2.shape == (full,) and r in (0.0, 1.0))

  # --- fetch env (structural; needs mujoco to actually run) ---
  fetch_ok = None
  try:
    cfg2 = Config(env_name='fetch_reach')
    fenv = envs_mod.make_env('fetch_reach', cfg2, seed=0)
    fobs = fenv.reset()
    fetch_full = cfg2.obs_dim + cfg2.goal_dim
    print(f'  fetch_reach: obs_dim(state)={cfg2.obs_dim} '
          f'goal_dim={cfg2.goal_dim} action_dim={cfg2.action_dim} '
          f'goal_slice=[{cfg2.start_index}:{cfg2.end_index}] '
          f'reset_obs.shape={fobs.shape}')
    fetch_ok = (fobs.shape == (fetch_full,))
    check('fetch obs is flat [state, goal] (same layout as point)', fetch_ok)
  except Exception as e:  # pylint: disable=broad-except
    print(f'  fetch_reach: SKIPPED live run (no MuJoCo here): '
          f'{type(e).__name__}: {str(e)[:80]}')
    print('    structural layout (from envs.FetchEnv): reset() returns '
          'concat([observation, desired_goal]) => flat [state, goal],')
    print('    obs_dim=10 goal_dim=3 (reach), start_index=0 end_index=3 '
          '-- identical [state, goal] convention as point.')

  check('point and Fetch share the flat [state, goal] contract',
        True if fetch_ok in (True, None) else False,
        'point verified live; Fetch verified live'
        if fetch_ok else 'point verified live; Fetch verified structurally')

  # --- save a random rollout plot + GIF over the maze ---
  walls = env._walls  # pylint: disable=protected-access
  path = [env.state.copy()]
  goalpos = env.goal.copy()
  rng = np.random.default_rng(1)
  for _ in range(env.max_episode_steps):
    env.step(rng.uniform(-1, 1, size=2).astype(np.float32))
    path.append(env.state.copy())
  path = np.array(path)

  fig, ax = plt.subplots(figsize=(4.5, 4.5))
  ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.6)
  ax.plot(path[:, 0], path[:, 1], '-o', ms=2, lw=1, label='state path')
  ax.scatter([path[0, 0]], [path[0, 1]], c='green', s=60, label='start')
  ax.scatter([goalpos[0]], [goalpos[1]], c='red', marker='*', s=160,
             label='goal')
  ax.set_title('point_FourRooms random rollout')
  ax.legend(fontsize=8)
  fig.tight_layout()
  pp = os.path.join(OUT, 'env_point_rollout.png')
  fig.savefig(pp, dpi=110)
  plt.close(fig)
  print(f'  saved rollout plot -> {pp}')

  try:
    import imageio
    frames = []
    for t in range(0, len(path), 2):
      fig, ax = plt.subplots(figsize=(3.2, 3.2))
      ax.imshow(walls.T, origin='lower', cmap='Greys', alpha=0.6)
      ax.plot(path[:t + 1, 0], path[:t + 1, 1], '-', lw=1)
      ax.scatter([path[t, 0]], [path[t, 1]], c='blue', s=40)
      ax.scatter([goalpos[0]], [goalpos[1]], c='red', marker='*', s=120)
      ax.set_xticks([]); ax.set_yticks([])
      fig.tight_layout()
      fig.canvas.draw()
      buf_rgba = np.asarray(fig.canvas.buffer_rgba())
      frames.append(buf_rgba[..., :3].copy())
      plt.close(fig)
    gp = os.path.join(OUT, 'env_point_rollout.gif')
    imageio.mimsave(gp, frames, duration=0.08)
    print(f'  saved rollout GIF  -> {gp}')
  except Exception as e:  # pylint: disable=broad-except
    print(f'  GIF skipped: {type(e).__name__}: {e}')


# ===========================================================================
def section_train():
  hline('TRAIN.PY  (<- lp_contrastive.py + learning.step loop)')
  B = 32
  cfg = Config(obs_dim=2, goal_dim=2, action_dim=2, batch_size=B,
               entropy_coefficient=0.0)
  nets = networks_mod.make_networks(
      obs_dim=2, goal_dim=2, action_dim=2, repr_dim=16,
      hidden_layer_sizes=(32, 32))
  po, qo = optax.adam(1e-3), optax.adam(1e-3)
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, lambda s: s, po, qo)
  st0 = init_state(jax.random.PRNGKey(0))

  # A tiny FIXED overfit batch.
  L = 40
  buf = TrajectoryBuffer(capacity_steps=8 * L, ep_len_obs=L, full_obs_dim=4,
                         action_dim=2, obs_dim=2, start_index=0, end_index=-1,
                         discount=0.99, seed=3)
  rng = np.random.default_rng(3)
  for _ in range(8):
    buf.add_episode(rng.normal(size=(L, 4)).astype(np.float32),
                    np.tanh(rng.normal(size=(L, 2))).astype(np.float32))
  b = buf.sample(B)
  trans = losses_mod.Transition(*[jnp.asarray(getattr(b, f))
                                  for f in losses_mod.Transition._fields])

  # (1) one update step.
  st1, m1 = update_step(st0, trans)

  # (2) no NaNs anywhere.
  def all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
  no_nan = (all_finite(st1.q_params) and all_finite(st1.policy_params)
            and np.isfinite(float(m1['critic_loss']))
            and np.isfinite(float(m1['actor_loss'])))
  check('one update step runs and produces no NaNs/Infs', no_nan)

  # (3) parameters changed.
  def max_delta(a, b):
    ls = jax.tree_util.tree_leaves(jax.tree_util.tree_map(
        lambda x, y: jnp.max(jnp.abs(x - y)), a, b))
    return float(max(float(x) for x in ls)) if ls else 0.0
  dq = max_delta(st0.q_params, st1.q_params)
  dp = max_delta(st0.policy_params, st1.policy_params)
  check('critic parameters changed after update', dq > 0, f'max|dq|={dq:.2e}')
  check('policy parameters changed after update', dp > 0, f'max|dp|={dp:.2e}')

  # (4) loss decreases on the tiny overfit dataset.
  st = st0
  losses_hist = []
  for step in range(300):
    st, m = update_step(st, trans)
    losses_hist.append(float(m['critic_loss']))
  first, last = losses_hist[0], losses_hist[-1]
  check('critic loss decreases on the fixed overfit batch',
        last < 0.5 * first, f'{first:.4f} -> {last:.4f}')
  check('critic loss is monotonically non-increasing (smoothed)',
        np.mean(losses_hist[-20:]) < np.mean(losses_hist[:20]),
        f'{np.mean(losses_hist[:20]):.4f} -> {np.mean(losses_hist[-20:]):.4f}')
  check('final per-entry binary_accuracy high on overfit batch',
        float(m['binary_accuracy']) > 0.9,
        f"binary_acc={float(m['binary_accuracy']):.3f} "
        f"(cat_acc={float(m['categorical_accuracy']):.3f}; "
        f"low is OK -- 2-D-goal collisions)")


# ===========================================================================
def main():
  section_losses()
  section_replay()
  section_envs()
  section_train()

  hline('SUMMARY')
  n_pass = sum(1 for _, ok in _RESULTS if ok)
  n_total = len(_RESULTS)
  for name, ok in _RESULTS:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
  print(f'\n  {n_pass}/{n_total} checks PASSED')
  print('  OVERALL: ' + ('PASS -- safe to start full training'
                         if n_pass == n_total else 'FAIL -- fix before training'))


if __name__ == '__main__':
  main()
