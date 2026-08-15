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
                  q_optimizer, fail_bank=None):
  """Returns ``(init_state, update_step)`` closures for the given config.

  ``obs_to_goal`` maps a batch of states [B, obs_dim] -> goal coords
  [B, goal_dim] (slice ``start_index:end_index``); used only by the TD path.

  ``fail_bank`` (optional, [N_bank, goal_dim]): failure-state bank in GOAL
  coordinates for failure-aware negative sampling. Used only when
  ``config.fail_neg_alpha > 0`` (see critic_loss); ``None`` or alpha 0 leaves
  every loss byte-identical to the baseline.
  """
  adaptive_entropy_coefficient = config.entropy_coefficient is None
  obs_dim = config.obs_dim

  # --- Failure-aware negatives (Part 1): static setup -----------------------
  # Negative-distribution mixture q_alpha = (1-alpha)*p_clean + alpha*q_fail.
  # Implemented as a loss-level mixture that PRESERVES the original
  # positive/negative weighting exactly (see critic_loss): the positive term
  # keeps its original coefficient; only the negative term becomes
  # (1-alpha)*L_ordinary-neg + alpha*L_failure-neg, where L_failure-neg is the
  # EXACT expectation over the (small) bank -- all bank states scored, uniform
  # average -- so no sampling noise enters. At alpha=0 the loss and gradients
  # are byte-identical to the baseline (the fail branch is skipped entirely).
  fail_alpha = float(getattr(config, 'fail_neg_alpha', 0.0) or 0.0)
  fail_enabled = fail_bank is not None and fail_alpha > 0.0
  if fail_alpha > 0.0 and fail_bank is None:
    raise ValueError('fail_neg_alpha > 0 requires a failure bank '
                     '(config.fail_bank_path).')
  if fail_enabled:
    if config.use_td or config.use_cpc or config.use_gcbc:
      raise ValueError('failure-aware negatives are implemented only for the '
                       'Monte-Carlo NCE critic (use_td=False, use_cpc=False).')
    if not 0.0 < fail_alpha < 1.0:
      raise ValueError(f'fail_neg_alpha must be in (0, 1), got {fail_alpha}')
    fail_bank_arr = jnp.asarray(fail_bank, jnp.float32)
    assert fail_bank_arr.ndim == 2 and fail_bank_arr.shape[0] > 0
    if fail_bank_arr.shape[0] > config.batch_size:
      raise ValueError(
          f'failure bank ({fail_bank_arr.shape[0]} states) larger than '
          f'batch_size={config.batch_size}; the padded second critic apply '
          'requires n_bank <= batch_size.')

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

    fail_metrics = {}
    if fail_enabled and not config.use_td:
      # Failure-aware negatives -- loss-level mixture that PRESERVES the
      # original positive/negative weighting. Decompose the original
      # jnp.mean(loss) over the B x B elementwise-BCE matrix:
      #
      #   L_orig = S_pos/B^2 + S_neg/B^2
      #     S_pos = sum of the B diagonal (positive) elements,
      #     S_neg = sum of the B(B-1) off-diagonal (negative) elements
      #           = B(B-1)/B^2 * E_offdiag[BCE]  (total negative mass (B-1)/B).
      #
      # Only the negative DISTRIBUTION changes, per q_alpha:
      #
      #   L(alpha) = S_pos/B^2                       (positive term UNCHANGED)
      #            + (1-alpha) * S_neg/B^2           (ordinary negatives)
      #            + alpha * (B-1)/B * E_fail[BCE]   (same total negative mass)
      #
      # E_fail is computed EXACTLY: every bank state scored against every
      # in-batch anchor (s_i, a_i) via a second critic apply on the SAME
      # states/actions with the goal half of the first n_bank rows replaced by
      # the bank -- sa_repr rows are identical, so column j of the result is
      # exactly critic(s_i, a_i, g_fail_j) -- then uniformly averaged
      # (q_fail = uniform over the bank). All labels 0. At alpha=0 the
      # expression reduces algebraically to jnp.mean(loss) (the else branch).
      n_bank = fail_bank_arr.shape[0]
      state = transitions.observation[:, :obs_dim]
      goal_half = transitions.observation[:, obs_dim:]
      goal2 = jnp.concatenate([fail_bank_arr, goal_half[n_bank:]], axis=0)
      obs2 = jnp.concatenate([state, goal2], axis=1)
      fail_logits = networks.q_network.apply(
          q_params, obs2, transitions.action)[:, :n_bank]   # [B, n_bank(, 2)]

      fail_loss = optax.sigmoid_binary_cross_entropy(
          logits=fail_logits, labels=jnp.zeros_like(fail_logits))
      if len(fail_logits.shape) == 3:  # twin q
        fail_loss = jnp.mean(fail_loss, axis=-1)
        fail_logits = jnp.mean(fail_logits, axis=-1)

      pos_term = jnp.sum(loss * I) / (batch_size ** 2)
      neg_ord = jnp.sum(loss * (1 - I)) / (batch_size ** 2)
      neg_fail = (batch_size - 1) / batch_size * jnp.mean(fail_loss)
      loss = pos_term + (1 - fail_alpha) * neg_ord + fail_alpha * neg_fail
      fail_metrics = {
          # exact loss decomposition (weighted terms as they enter the loss)
          'critic_pos_term': pos_term,
          'critic_neg_ord_term': (1 - fail_alpha) * neg_ord,
          'critic_neg_fail_term': fail_alpha * neg_fail,
          # unweighted components + the mixture weight for auditability
          'critic_neg_ord_raw': neg_ord,
          'critic_neg_fail_raw': neg_fail,
          'fail_neg_alpha': jnp.asarray(fail_alpha, jnp.float32),
          'fail_bank_size': jnp.asarray(n_bank, jnp.float32),
          'logits_fail_neg': jnp.mean(fail_logits),
      }
    else:
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
        'logits_gap': logits_pos - logits_neg,  # NCE sanity: should be > 0.
        'logsumexp': logsumexp.mean(),
        **fail_metrics,
    }
    return loss, metrics

  # ------------------------------------------------------------------ actor
  def actor_loss(policy_params, q_params, alpha, transitions, key):
    obs = transitions.observation
    if config.use_gcbc:
      dist_params = networks.policy_network.apply(policy_params, obs)
      log_prob = networks.log_prob(dist_params, transitions.action)
      loss = -1.0 * jnp.mean(log_prob)
      return loss, {}

    state = obs[:, :obs_dim]
    goal = obs[:, obs_dim:]
    if config.random_goals == 0.0:
      new_state = state
      new_goal = goal
      orig_action = transitions.action
    elif config.random_goals == 0.5:
      new_state = jnp.concatenate([state, state], axis=0)
      new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
      orig_action = jnp.concatenate(
          [transitions.action, transitions.action], axis=0)
    else:
      assert config.random_goals == 1.0
      new_state = state
      new_goal = jnp.roll(goal, 1, axis=0)
      orig_action = transitions.action

    new_obs = jnp.concatenate([new_state, new_goal], axis=1)
    dist_params = networks.policy_network.apply(policy_params, new_obs)
    action = networks.sample(dist_params, key)
    log_prob = networks.log_prob(dist_params, action)
    q_action = networks.q_network.apply(q_params, new_obs, action)
    if len(q_action.shape) == 3:  # twin q trick
      assert q_action.shape[2] == 2
      # Upstream master uses the pessimistic MIN over the twin critics in the
      # actor objective (learning.py); the 2022 snapshot's jnp.mean is stale.
      q_action = jnp.min(q_action, axis=-1)
    q_term = alpha * log_prob - jnp.diag(q_action)

    # --- Actor-behavior diagnostics (additive; do not affect the loss) --------
    # These surface the saturation/collapse signatures that a fixed alpha=0 run
    # needs to be judged by (see crl/train.py logging). loc/scale are the
    # pre-tanh Gaussian params; the deterministic (mode) action is tanh(loc).
    loc = dist_params.loc
    scale = dist_params.scale
    mode_action = jnp.tanh(loc)
    diag = {
        # SAC-style entropy estimate of the current policy: E[-log pi(a|s)].
        'policy_entropy': jnp.mean(-log_prob),
        'policy_scale_median': jnp.median(scale),
        # fraction of action-dim scales pinned near the actor_min_std floor.
        'policy_scale_floor_fraction': jnp.mean((scale < 1e-3).astype(jnp.float32)),
        'pre_tanh_loc_abs_mean': jnp.mean(jnp.abs(loc)),
        'pre_tanh_loc_abs_max': jnp.max(jnp.abs(loc)),
        # fraction of mode-action components saturated against the tanh bound.
        'action_saturation_fraction':
            jnp.mean((jnp.abs(mode_action) > 0.99).astype(jnp.float32)),
    }

    if config.bc_coef > 0:
      # Offline actor objective (paper Eq 7-8 / WindyCorridor recipe):
      # max (1-bc)*E_pi[f] + bc*log pi(a_orig|s,g). log_prob clips boundary
      # actions internally, so dataset actions at exactly +/-1 are safe.
      bc_nll = -networks.log_prob(dist_params, orig_action)
      loss = config.bc_coef * bc_nll + (1 - config.bc_coef) * q_term
      bc_nll_mean = jnp.mean(bc_nll)
      q_term_mean = jnp.mean(q_term)
      aux = {
          'actor_q_term': q_term_mean, 'bc_nll': bc_nll_mean,
          # raw = unweighted component means; weighted = as they enter the loss.
          'bc_nll_raw': bc_nll_mean,
          'bc_loss_weighted': config.bc_coef * bc_nll_mean,
          'critic_actor_term_raw': q_term_mean,
          'critic_actor_term_weighted': (1 - config.bc_coef) * q_term_mean,
      }
    else:
      loss = q_term
      q_term_mean = jnp.mean(q_term)
      aux = {'critic_actor_term_raw': q_term_mean,
             'critic_actor_term_weighted': q_term_mean}
    aux.update(diag)
    return jnp.mean(loss), aux

  alpha_grad = jax.value_and_grad(alpha_loss)
  critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
  actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

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

    (actor_loss_value, actor_aux), actor_grads = actor_grad(
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
        # Gradient-norm health (additive diagnostics): a collapsed actor tends to
        # a near-zero actor grad norm; a diverging critic shows a growing one.
        'actor_grad_norm': optax.global_norm(actor_grads),
        'critic_grad_norm': (optax.global_norm(critic_grads)
                             if not config.use_gcbc else 0.0),
    })
    metrics.update(actor_aux)

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
