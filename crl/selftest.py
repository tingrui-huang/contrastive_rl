"""Local self-tests for the contrastive RL port (no MuJoCo needed).

Run:  python -m crl.selftest
Checks:
  1. The future-goal sampler follows the geometric law prob(j-i) ~ discount**d,
     and only ever picks future indices (j > i).
  2. Relabeled batch shapes are correct.
  3. One full end-to-end learner update runs for contrastive_nce AND c_learning.
"""
import numpy as np


def test_sampler_distribution():
  from crl.replay import TrajectoryBuffer
  discount = 0.9
  L = 40  # observations per episode.
  buf = TrajectoryBuffer(
      capacity_steps=200 * L, ep_len_obs=L, full_obs_dim=4, action_dim=2,
      obs_dim=2, start_index=0, end_index=-1, discount=discount, seed=0)
  # Episodes where state == time index, so we can recover i and j exactly.
  for e in range(200):
    obs = np.zeros((L, 4), np.float32)
    obs[:, 0] = np.arange(L)          # state x = time  -> recovers i and j.
    obs[:, 1] = e                     # state y = episode id.
    act = np.zeros((L, 2), np.float32)
    buf.add_episode(obs, act)

  B = 400_000
  batch = buf.sample(B)
  i = batch.observation[:, 0]                    # anchor time (state x).
  j = batch.observation[:, 2]                    # goal x == future time.
  assert np.all(j > i), 'sampler picked a non-future goal!'

  # Condition on anchor i==0: the offset j should be TRUNCATED-geometric over
  # j in [1, L-1] with prob ~ discount**j (exactly the flatten_fn categorical).
  # (Pooling over a uniform anchor would instead give a mixture, so we condition.)
  mask = i == 0
  d = j[mask].astype(int)                         # offset == j since i==0.
  counts = np.bincount(d, minlength=L)[1:]        # d = 1 .. L-1
  emp = counts / counts.sum()
  ref = discount ** np.arange(1, L, dtype=float)
  ref = ref / ref.sum()
  max_err = np.max(np.abs(emp - ref))
  print(f'[sampler] anchor i=0, n={mask.sum()}, '
        f'max |empirical - truncated-geometric| = {max_err:.4f}')
  assert max_err < 0.01, f'offset distribution off geometric: {max_err}'
  print('[sampler] OK: future-only + geometric(discount) offsets.')


def test_shapes_and_update():
  import jax
  import optax
  from crl.config import Config
  from crl import networks as networks_mod
  from crl import losses as losses_mod
  from crl.replay import TrajectoryBuffer

  for name, kw in [('contrastive_nce', {}),
                   ('c_learning', {'use_td': True, 'twin_q': True})]:
    config = Config(obs_dim=2, goal_dim=2, action_dim=2, max_episode_steps=39,
                    batch_size=32, entropy_coefficient=0.0, **kw)
    L = config.max_episode_steps + 1
    nets = networks_mod.make_networks(
        obs_dim=2, goal_dim=2, action_dim=2, repr_dim=16,
        hidden_layer_sizes=(32, 32), twin_q=config.twin_q)
    po = optax.adam(3e-4)
    qo = optax.adam(3e-4)
    def obs_to_goal(s):
      return s  # start=0,end=-1 -> full state.
    init_state, update_step = losses_mod.build_learner(
        nets, config, obs_to_goal, po, qo)
    state = init_state(jax.random.PRNGKey(0))

    buf = TrajectoryBuffer(
        capacity_steps=50 * L, ep_len_obs=L, full_obs_dim=4, action_dim=2,
        obs_dim=2, start_index=0, end_index=-1, discount=config.discount)
    for _ in range(50):
      buf.add_episode(np.random.randn(L, 4).astype(np.float32),
                      np.tanh(np.random.randn(L, 2)).astype(np.float32))
    batch = buf.sample(config.batch_size)
    assert batch.observation.shape == (32, 4), batch.observation.shape
    import jax.numpy as jnp
    tb = losses_mod.Transition(*[jnp.asarray(getattr(batch, f))
                                 for f in losses_mod.Transition._fields])
    state, metrics = update_step(state, tb)
    cl = float(metrics['critic_loss'])
    al = float(metrics['actor_loss'])
    assert np.isfinite(cl) and np.isfinite(al), (name, cl, al)
    print(f'[update:{name}] OK: critic_loss={cl:.3f} actor_loss={al:.3f} '
          f'cat_acc={float(metrics.get("categorical_accuracy", 0)):.3f}')


if __name__ == '__main__':
  test_sampler_distribution()
  test_shapes_and_update()
  print('\nAll self-tests passed.')
