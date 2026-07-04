"""Contrastive RL losses (Acme-free port of ``contrastive/learning.py``).

The loss bodies (critic: NCE / CPC / C-learning; actor: SAC with the diagonal-Q
+ random-goals trick; adaptive alpha) are copied faithfully from the original
learner. Removed Acme dependencies:

  * ``acme.types.Transition``            -> the local ``Transition`` namedtuple.
  * ``jax.tree_multimap``                -> ``jax.tree_util.tree_map``.
  * ``utils.process_multiple_batches``   -> ``jax.lax.scan`` in ``train.py``.

Reward and discount are carried in ``Transition`` for completeness but, as in
the original, the contrastive losses never read them.
"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax


class Transition(NamedTuple):
  observation: jnp.ndarray       # concat([state, relabeled_goal])
  action: jnp.ndarray
  reward: jnp.ndarray
  discount: jnp.ndarray
  next_observation: jnp.ndarray  # concat([next_state, same relabeled_goal])
  next_action: jnp.ndarray


class TrainingState(NamedTuple):
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: object
  q_params: object
  target_q_params: object
  key: jnp.ndarray
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[jnp.ndarray] = None


def build_learner(networks, config, obs_to_goal, policy_optimizer,
                  q_optimizer):
  """Returns ``(init_state, update_step)`` closures for the given config.

  ``obs_to_goal`` maps a batch of states [B, obs_dim] -> goal coords
  [B, goal_dim] (slice ``start_index:end_index``); used only by the TD path.
  """
  adaptive_entropy_coefficient = config.entropy_coefficient is None
  obs_dim = config.obs_dim

  if adaptive_entropy_coefficient:
    log_alpha_init = jnp.asarray(0., dtype=jnp.float32)
    alpha_optimizer = optax.adam(learning_rate=3e-4)
    alpha_optimizer_state_init = alpha_optimizer.init(log_alpha_init)
  else:
    if config.target_entropy:
      raise ValueError('target_entropy should not be set when '
                       'entropy_coefficient is provided')

  # ------------------------------------------------------------------ alpha
  def alpha_loss(log_alpha, policy_params, transitions, key):
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = networks.policy_network.apply(
        policy_params, transitions.observation)
    action = networks.sample(dist_params, key)
    log_prob = networks.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    loss = alpha * jax.lax.stop_gradient(-log_prob - config.target_entropy)
    return jnp.mean(loss)

  # ----------------------------------------------------------------- critic
  def critic_loss(q_params, policy_params, target_q_params, transitions, key):
    batch_size = transitions.observation.shape[0]
    if config.use_td:
      # For TD learning, diagonal elements are the immediate next state.
      s, g = jnp.split(transitions.observation, [obs_dim], axis=1)
      next_s, _ = jnp.split(transitions.next_observation, [obs_dim], axis=1)
      if config.add_mc_to_td:
        next_fraction = (1 - config.discount) / ((1 - config.discount) + 1)
        num_next = int(batch_size * next_fraction)
        new_g = jnp.concatenate([
            obs_to_goal(next_s[:num_next]),
            g[num_next:],
        ], axis=0)
      else:
        new_g = obs_to_goal(next_s)
      obs = jnp.concatenate([s, new_g], axis=1)
      transitions = transitions._replace(observation=obs)
    I = jnp.eye(batch_size)  # pylint: disable=invalid-name
    logits = networks.q_network.apply(
        q_params, transitions.observation, transitions.action)

    if config.use_td:
      assert len(logits.shape) == 3  # twin Q required.
      s, g = jnp.split(transitions.observation, [obs_dim], axis=1)
      del s
      next_s = transitions.next_observation[:, :obs_dim]
      goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)
      g = g[goal_indices]
      transitions = transitions._replace(
          next_observation=jnp.concatenate([next_s, g], axis=1))
      next_dist_params = networks.policy_network.apply(
          policy_params, transitions.next_observation)
      next_action = networks.sample(next_dist_params, key)
      next_q = networks.q_network.apply(target_q_params,
                                        transitions.next_observation,
                                        next_action)
      next_q = jax.nn.sigmoid(next_q)
      next_v = jnp.min(next_q, axis=-1)
      next_v = jax.lax.stop_gradient(next_v)
      next_v = jnp.diag(next_v)
      w = next_v / (1 - next_v)
      w_clipping = 20.0
      w = jnp.clip(w, 0, w_clipping)
      pos_logits = jax.vmap(jnp.diag, -1, -1)(logits)
      loss_pos = optax.sigmoid_binary_cross_entropy(
          logits=pos_logits, labels=1)  # [B, 2]

      neg_logits = logits[jnp.arange(batch_size), goal_indices]
      loss_neg1 = w[:, None] * optax.sigmoid_binary_cross_entropy(
          logits=neg_logits, labels=1)  # [B, 2]
      loss_neg2 = optax.sigmoid_binary_cross_entropy(
          logits=neg_logits, labels=0)  # [B, 2]

      if config.add_mc_to_td:
        loss = ((1 + (1 - config.discount)) * loss_pos
                + config.discount * loss_neg1 + 2 * loss_neg2)
      else:
        loss = ((1 - config.discount) * loss_pos
                + config.discount * loss_neg1 + loss_neg2)
      logits = jnp.mean(logits, axis=-1)

    else:  # Monte-Carlo contrastive losses.
      def loss_fn(_logits):  # pylint: disable=invalid-name
        if config.use_cpc:
          return (optax.softmax_cross_entropy(logits=_logits, labels=I)
                  + 0.01 * jax.nn.logsumexp(_logits, axis=1)**2)
        else:
          return optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I)
      if len(logits.shape) == 3:  # twin q
        loss = jax.vmap(loss_fn, in_axes=2, out_axes=-1)(logits)
        loss = jnp.mean(loss, axis=-1)
        logits = jnp.mean(logits, axis=-1)
      else:
        loss = loss_fn(logits)

    loss = jnp.mean(loss)
    correct = (jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1))
    logits_pos = jnp.sum(logits * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
    if len(logits.shape) == 3:
      logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1)**2
    else:
      logsumexp = jax.nn.logsumexp(logits, axis=1)**2
    metrics = {
        'binary_accuracy': jnp.mean((logits > 0) == I),
        'categorical_accuracy': jnp.mean(correct),
        'logits_pos': logits_pos,
        'logits_neg': logits_neg,
        'logsumexp': logsumexp.mean(),
    }
    return loss, metrics

  # ------------------------------------------------------------------ actor
  def actor_loss(policy_params, q_params, alpha, transitions, key):
    obs = transitions.observation
    if config.use_gcbc:
      dist_params = networks.policy_network.apply(policy_params, obs)
      log_prob = networks.log_prob(dist_params, transitions.action)
      loss = -1.0 * jnp.mean(log_prob)
      return loss

    state = obs[:, :obs_dim]
    goal = obs[:, obs_dim:]
    if config.random_goals == 0.0:
      new_state = state
      new_goal = goal
    elif config.random_goals == 0.5:
      new_state = jnp.concatenate([state, state], axis=0)
      new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
    else:
      assert config.random_goals == 1.0
      new_state = state
      new_goal = jnp.roll(goal, 1, axis=0)

    new_obs = jnp.concatenate([new_state, new_goal], axis=1)
    dist_params = networks.policy_network.apply(policy_params, new_obs)
    action = networks.sample(dist_params, key)
    log_prob = networks.log_prob(dist_params, action)
    q_action = networks.q_network.apply(q_params, new_obs, action)
    if len(q_action.shape) == 3:  # twin q trick
      assert q_action.shape[2] == 2
      q_action = jnp.mean(q_action, axis=-1)
    loss = alpha * log_prob - jnp.diag(q_action)
    return jnp.mean(loss)

  alpha_grad = jax.value_and_grad(alpha_loss)
  critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
  actor_grad = jax.value_and_grad(actor_loss)

  # ------------------------------------------------------------- update step
  def update_step(state, transitions):
    key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
    if adaptive_entropy_coefficient:
      alpha_loss_value, alpha_grads = alpha_grad(
          state.alpha_params, state.policy_params, transitions, key_alpha)
      alpha = jnp.exp(state.alpha_params)
    else:
      alpha = config.entropy_coefficient

    if not config.use_gcbc:
      (critic_loss_value, critic_metrics), critic_grads = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          transitions, key_critic)

    actor_loss_value, actor_grads = actor_grad(
        state.policy_params, state.q_params, alpha, transitions, key_actor)

    actor_update, policy_optimizer_state = policy_optimizer.update(
        actor_grads, state.policy_optimizer_state)
    policy_params = optax.apply_updates(state.policy_params, actor_update)

    if config.use_gcbc:
      metrics = {}
      critic_loss_value = 0.0
      q_params = state.q_params
      q_optimizer_state = state.q_optimizer_state
      new_target_q_params = state.target_q_params
    else:
      critic_update, q_optimizer_state = q_optimizer.update(
          critic_grads, state.q_optimizer_state)
      q_params = optax.apply_updates(state.q_params, critic_update)
      new_target_q_params = jax.tree_util.tree_map(
          lambda x, y: x * (1 - config.tau) + y * config.tau,
          state.target_q_params, q_params)
      metrics = critic_metrics

    metrics.update({
        'critic_loss': critic_loss_value,
        'actor_loss': actor_loss_value,
    })

    new_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        q_optimizer_state=q_optimizer_state,
        policy_params=policy_params,
        q_params=q_params,
        target_q_params=new_target_q_params,
        key=key,
        alpha_optimizer_state=state.alpha_optimizer_state,
        alpha_params=state.alpha_params,
    )
    if adaptive_entropy_coefficient:
      alpha_update, alpha_optimizer_state = alpha_optimizer.update(
          alpha_grads, state.alpha_optimizer_state)
      alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
      metrics.update({'alpha_loss': alpha_loss_value,
                      'alpha': jnp.exp(alpha_params)})
      new_state = new_state._replace(
          alpha_optimizer_state=alpha_optimizer_state,
          alpha_params=alpha_params)
    return new_state, metrics

  # ------------------------------------------------------------ init state
  def init_state(key):
    key_policy, key_q, key = jax.random.split(key, 3)
    policy_params = networks.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = networks.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)
    state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        q_optimizer_state=q_optimizer_state,
        policy_params=policy_params,
        q_params=q_params,
        target_q_params=q_params,
        key=key)
    if adaptive_entropy_coefficient:
      state = state._replace(
          alpha_optimizer_state=alpha_optimizer_state_init,
          alpha_params=log_alpha_init)
    return state

  return init_state, update_step
