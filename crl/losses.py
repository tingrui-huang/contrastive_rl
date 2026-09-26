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
  # --- task-goal supervision (used only when config.task_goal_coef > 0) ------
  # The episode's OWN task goal in goal coordinates, and whether that episode
  # actually reached it. Unlike the relabeled goal above, task_goal does not
  # depend on the anchor time, and task_success is a trajectory-level outcome
  # that hindsight relabeling cannot express: a failed episode never contains
  # the task goal among its future states, so the ordinary contrastive loss
  # never scores (s, a, g_task) with a negative label. Both are None on the
  # paths that do not supply them; critic_loss reads them only behind the flag.
  task_goal: Optional[jnp.ndarray] = None      # [B, goal_dim]
  task_success: Optional[jnp.ndarray] = None   # [B], 1.0 reached / 0.0 not
  # --- death-trajectory hindsight negatives (config.death_neg_coef > 0) -----
  # An INDEPENDENT batch drawn only from the failure episodes and relabeled by
  # the ordinary rule: anchor time i, goal from a geometric future j > i of
  # the same trajectory. Hindsight would call such a pair a positive; this
  # term calls it a 0, because the place it reached is on a path that ended in
  # a death. Rows here do not correspond to the rows above.
  death_observation: Optional[jnp.ndarray] = None   # [B_d, obs_dim+goal_dim]
  death_action: Optional[jnp.ndarray] = None        # [B_d, action_dim]
  # --- generated-trajectory negatives (config.gen_neg_coef > 0) -------------
  # An INDEPENDENT batch drawn from the causal transition model's generated
  # trajectories and relabeled by the ordinary rule inside each one. Read only
  # by the critic's generated-negative term; the contrastive loss and the
  # actor never see it. Rows here do not correspond to the rows above.
  gen_observation: Optional[jnp.ndarray] = None     # [B_g, obs_dim+goal_dim]
  gen_action: Optional[jnp.ndarray] = None          # [B_g, action_dim]

  def apply_fields(self, fn):
    """Rebuild this Transition with ``fn`` applied to every PRESENT field.

    The optional fields above are None on the paths that do not supply them,
    and ``jnp.asarray(None)`` raises, so every call site that converts a whole
    batch (to device arrays, to numpy, ...) goes through here rather than
    splatting ``_fields`` blindly."""
    return self._replace(**{
        name: (None if getattr(self, name) is None else fn(getattr(self, name)))
        for name in self._fields})


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
                  q_optimizer, fail_bank=None, separate_actor_batch=False,
                  anchor_params=None, anchor_coef=0.0):
  """Returns ``(init_state, update_step)`` closures for the given config.

  ``obs_to_goal`` maps a batch of states [B, obs_dim] -> goal coords
  [B, goal_dim] (slice ``start_index:end_index``); used only by the TD path.

  ``fail_bank`` (optional, [N_bank, goal_dim]): failure-state bank in GOAL
  coordinates for failure-aware negative sampling. Used only when
  ``config.fail_neg_alpha > 0`` (see critic_loss); ``None`` or alpha 0 leaves
  every loss byte-identical to the baseline.

  ``separate_actor_batch`` (sampling-interface change, notes/MAINLINE_CONTRACT.md):
  ``update_step`` takes ``(critic_transitions, actor_transitions)``.  The
  critic loss is taken on the first batch; the actor loss -- its critic term
  AND its BC term, on the same rows, exactly the ``bc_transitions is None``
  pairing -- on the second.  The loss bodies are the ones above, unchanged;
  what changes is only WHICH rows each loss is evaluated on.  With both
  batches drawn from the same stream this reduces to the shared-batch
  learner up to the RNG stream.  Requires ``bc_sampling == 'shared'``.

  ``anchor_params`` / ``anchor_coef`` (diagnostic variant, AntMaze V6 pilot):
  adds ``anchor_coef * mean_i ||tanh(loc_i) - tanh(loc_ref_i)||^2`` to the
  actor loss, where ``loc_ref`` is the mode of a FIXED reference policy
  (``anchor_params``, e.g. the policy the actor was initialised from) at the
  same (state, goal) rows -- a trust region to the previous iterate that uses
  no labels and leaves the BC and critic terms untouched.  Inactive at 0.
  """
  adaptive_entropy_coefficient = config.entropy_coefficient is None
  if separate_actor_batch and (getattr(config, 'bc_sampling', 'shared') or 'shared') != 'shared':
    raise ValueError('separate_actor_batch keeps the shared-row actor pairing; '
                     'bc_sampling must be "shared"')
  obs_dim = config.obs_dim
  bc_sampling = getattr(config, 'bc_sampling', 'shared') or 'shared'
  if bc_sampling not in ('shared', 'independent', 'balanced'):
    raise ValueError(f'unknown bc_sampling {bc_sampling!r}')
  if bc_sampling != 'shared' and (config.random_goals != 0.0
                                  or config.bc_coef <= 0):
    raise ValueError('bc_sampling independent/balanced requires '
                     'random_goals 0 and bc_coef > 0')

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

  # --- Task-goal supervision: the negative term ------------------------------
  # The Monte-Carlo contrastive critic is trained purely by hindsight: every
  # positive is (s, a, g) with g a FUTURE state of the same trajectory. A
  # failure trajectory therefore teaches "this (s, a) reaches the place where
  # you died" and never "this (s, a) fails to reach the task goal" -- the task
  # goal is simply absent from its future, so it is never scored with a
  # negative label. The only pressure it puts on f(s, a, g_task) is as an
  # off-diagonal negative of OTHER anchors, which is the same pressure a
  # successful trajectory gets.
  #
  # This term supplies the missing label directly: score f(s, a, g_task) for
  # the anchor's OWN task goal and regress it, with binary cross-entropy,
  # onto whether that episode actually reached the goal. Successful
  # transitions get label 1, negative-dataset transitions label 0. Both label
  # classes carry the same goal (the task goal), so nothing is separable from
  # the goal input alone -- the margin has to be carried by sa_repr(s, a).
  #
  # Added to the critic loss with weight ``config.task_goal_coef``; at 0 the
  # branch is skipped entirely and the loss is byte-identical to the baseline.
  task_coef = float(getattr(config, 'task_goal_coef', 0.0) or 0.0)
  task_enabled = task_coef > 0.0
  if task_enabled:
    if config.use_td or config.use_cpc or config.use_gcbc:
      raise ValueError('task-goal supervision is implemented only for the '
                       'Monte-Carlo NCE critic (use_td=False, use_cpc=False).')
    if task_coef < 0.0:
      raise ValueError(f'task_goal_coef must be >= 0, got {task_coef}')

  # --- death-trajectory hindsight negatives ---------------------------------
  # fail_neg_alpha pairs banked death STATES with arbitrary anchors, so the
  # only input that can carry the label is the goal -- and in the 2-dim XY
  # goal the critic sees, a death state and a safe crossing of the same band
  # are the same point (measured AUC 0.505, notes/v6_failure_negatives.md).
  # This term keeps the anchor and the goal in the SAME death trajectory and
  # relabels by the ordinary geometric-future rule, so the pair is one
  # hindsight would score as a positive; labeling it 0 puts the pressure on
  # sa_repr(s, a) instead of on the goal.
  death_coef = float(getattr(config, 'death_neg_coef', 0.0) or 0.0)
  death_enabled = death_coef > 0.0
  if death_enabled:
    if config.use_td or config.use_cpc or config.use_gcbc:
      raise ValueError('death-trajectory negatives are implemented only for '
                       'the Monte-Carlo NCE critic (use_td=False, '
                       'use_cpc=False, use_gcbc=False).')
    if death_coef < 0.0:
      raise ValueError(f'death_neg_coef must be >= 0, got {death_coef}')

  # --- generated-trajectory negatives (crl/ETT_train_v2.py) -----------------
  # Real hindsight pairs labeled 1, the transition model's generated hindsight
  # pairs labeled 0; the real side is the diagonal of the contrastive batch.
  gen_coef = float(getattr(config, 'gen_neg_coef', 0.0) or 0.0)
  gen_enabled = gen_coef > 0.0
  if gen_enabled:
    if config.use_td or config.use_cpc or config.use_gcbc:
      raise ValueError('generated-trajectory negatives are implemented only '
                       'for the Monte-Carlo NCE critic (use_td=False, '
                       'use_cpc=False, use_gcbc=False).')
  elif gen_coef < 0.0:
    raise ValueError(f'gen_neg_coef must be >= 0, got {gen_coef}')

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
    q_logits = logits   # per-head [B, B(, 2)], before any twin averaging.

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

    task_metrics = {}
    if task_enabled:
      # f(s, a, g_task) for the anchor's OWN task goal, one score per row (the
      # diagonal of the outer product), regressed onto the episode outcome.
      if transitions.task_goal is None or transitions.task_success is None:
        raise ValueError(
            'task_goal_coef > 0 but the batch carries no task_goal / '
            'task_success; the replay buffer was not given episode labels.')
      state_t = transitions.observation[:, :obs_dim]
      obs_task = jnp.concatenate([state_t, transitions.task_goal], axis=1)
      task_all = networks.q_network.apply(q_params, obs_task,
                                          transitions.action)
      if task_all.ndim == 3:                       # twin q -> [B, 2]
        task_logits = jax.vmap(jnp.diag, -1, -1)(task_all)
        task_labels = transitions.task_success[:, None]
      else:                                        # single q -> [B]
        task_logits = jnp.diag(task_all)
        task_labels = transitions.task_success
      task_term = jnp.mean(optax.sigmoid_binary_cross_entropy(
          logits=task_logits, labels=task_labels))
      loss = loss + task_coef * task_term

      f_task = (jnp.mean(task_logits, axis=-1) if task_logits.ndim == 2
                else task_logits)                  # [B], twin-averaged
      w1 = transitions.task_success
      w0 = 1.0 - w1
      f1 = jnp.sum(f_task * w1) / jnp.maximum(jnp.sum(w1), 1.0)
      f0 = jnp.sum(f_task * w0) / jnp.maximum(jnp.sum(w0), 1.0)
      task_metrics = {
          'task_goal_coef': jnp.asarray(task_coef, jnp.float32),
          'task_goal_term': task_coef * task_term,   # as it enters the loss
          'task_goal_term_raw': task_term,
          'task_logits_success': f1,
          'task_logits_failure': f0,
          # the quantity the term exists to open up.
          'task_logits_margin': f1 - f0,
          'task_success_fraction': jnp.mean(w1),
          'task_goal_accuracy': jnp.mean(
              ((f_task > 0) == (w1 > 0.5)).astype(jnp.float32)),
      }

    death_metrics = {}
    if death_enabled:
      # A pair (s, a, g) drawn entirely inside a death trajectory by the same
      # geometric-future rule the positives use, scored with label 0. Row k's
      # own state-action against row k's own relabeled goal -- the diagonal of
      # the outer product, as in the task-goal term above.
      if (transitions.death_observation is None
          or transitions.death_action is None):
        raise ValueError(
            'death_neg_coef > 0 but the batch carries no death_observation / '
            'death_action; the replay buffer was not asked for a death batch '
            '(see TrajectoryBuffer.enable_death_negatives).')
      death_all = networks.q_network.apply(
          q_params, transitions.death_observation, transitions.death_action)
      if death_all.ndim == 3:                      # twin q -> [B_d, 2]
        death_logits = jax.vmap(jnp.diag, -1, -1)(death_all)
      else:                                        # single q -> [B_d]
        death_logits = jnp.diag(death_all)
      death_term = jnp.mean(optax.sigmoid_binary_cross_entropy(
          logits=death_logits, labels=jnp.zeros_like(death_logits)))
      loss = loss + death_coef * death_term

      f_death = (jnp.mean(death_logits, axis=-1) if death_logits.ndim == 2
                 else death_logits)                # [B_d], twin-averaged
      death_metrics = {
          'death_neg_coef': jnp.asarray(death_coef, jnp.float32),
          'death_neg_term': death_coef * death_term,   # as it enters the loss
          'death_neg_term_raw': death_term,
          # f on the pairs the term is pushing down. The ordinary positive
          # diagonal ('logits_pos') is the reference: these pairs ARE
          # hindsight positives, so the gap between the two is the term's
          # whole effect on the critic.
          'death_pair_logits': jnp.mean(f_death),
          'death_pair_frac_pos': jnp.mean((f_death > 0).astype(jnp.float32)),
          # Per-element weight against the ordinary positive diagonal, which
          # the B x B mean gives 1/B^2 each: death_coef * B. At 1.0 this term
          # exactly cancels the positive pressure on a death-trajectory pair.
          'death_vs_pos_weight': jnp.asarray(death_coef * batch_size,
                                             jnp.float32),
      }

    gen_metrics = {}
    if gen_enabled:
      # Real side: row k of the contrastive batch against its own relabeled
      # goal, i.e. the diagonal already computed above -- label 1. Generated
      # side: the same diagonal on an independent batch drawn from the model's
      # trajectories -- label 0. Each class is averaged on its own, so the two
      # batch sizes do not set the class prior.
      if transitions.gen_observation is None or transitions.gen_action is None:
        raise ValueError(
            'gen_neg_coef > 0 but the batch carries no gen_observation / '
            'gen_action; the generated buffer was not sampled '
            '(see crl/ETT_train_v2.py).')
      gen_all = networks.q_network.apply(
          q_params, transitions.gen_observation, transitions.gen_action)
      if gen_all.ndim == 3:                        # twin q -> [B, 2], [B_g, 2]
        real_diag = jax.vmap(jnp.diag, -1, -1)(q_logits)
        gen_diag = jax.vmap(jnp.diag, -1, -1)(gen_all)
      else:                                        # single q -> [B], [B_g]
        real_diag = jnp.diag(q_logits)
        gen_diag = jnp.diag(gen_all)
      gen_real_term = jnp.mean(optax.sigmoid_binary_cross_entropy(
          logits=real_diag, labels=jnp.ones_like(real_diag)))
      gen_fake_term = jnp.mean(optax.sigmoid_binary_cross_entropy(
          logits=gen_diag, labels=jnp.zeros_like(gen_diag)))
      gen_term = gen_real_term + gen_fake_term
      loss = loss + gen_coef * gen_term

      f_real = (jnp.mean(real_diag, axis=-1) if real_diag.ndim == 2
                else real_diag)                    # [B], twin-averaged
      f_gen = (jnp.mean(gen_diag, axis=-1) if gen_diag.ndim == 2
               else gen_diag)                      # [B_g], twin-averaged
      gen_metrics = {
          'gen_neg_coef': jnp.asarray(gen_coef, jnp.float32),
          'gen_neg_term': gen_coef * gen_term,     # as it enters the loss
          'gen_neg_term_raw': gen_term,
          'gen_neg_real_bce': gen_real_term,
          'gen_neg_gen_bce': gen_fake_term,
          'gen_real_logits': jnp.mean(f_real),
          'gen_pair_logits': jnp.mean(f_gen),
          # the quantity the term exists to open up.
          'gen_logits_margin': jnp.mean(f_real) - jnp.mean(f_gen),
          'gen_accuracy': 0.5 * (jnp.mean((f_real > 0).astype(jnp.float32))
                                 + jnp.mean((f_gen < 0).astype(jnp.float32))),
      }

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
        **death_metrics,
        **task_metrics,
        **gen_metrics,
    }
    return loss, metrics

  # ------------------------------------------------------------------ actor
  def actor_loss(policy_params, q_params, alpha, transitions, key,
                 bc_transitions=None):
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
      if bc_transitions is None:
        bc_nll = -networks.log_prob(dist_params, orig_action)
      else:
        # BC rows drawn separately (config.bc_sampling != 'shared'): the
        # critic term above keeps the buffer batch, the BC term gets its own
        # (state, goal, recorded action) rows, e.g. region-balanced ones.
        bc_dist = networks.policy_network.apply(
            policy_params, bc_transitions.observation)
        bc_nll = -networks.log_prob(bc_dist, bc_transitions.action)
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
    if anchor_params is not None and anchor_coef > 0.0:
      ref = networks.policy_network.apply(anchor_params, new_obs)
      pen = jnp.sum((mode_action - jax.lax.stop_gradient(jnp.tanh(ref.loc))) ** 2, axis=1)
      loss = loss + anchor_coef * pen
      aux['anchor_penalty_raw'] = jnp.mean(pen)
      aux['anchor_penalty_weighted'] = anchor_coef * jnp.mean(pen)
    aux.update(diag)
    return jnp.mean(loss), aux

  alpha_grad = jax.value_and_grad(alpha_loss)
  critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
  actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

  # ------------------------------------------------------------- update step
  def update_step(state, transitions):
    # ``transitions`` is a Transition, or a (Transition, Transition-or-None)
    # pair whose second element holds the BC term's own rows; with
    # ``separate_actor_batch`` it is the (critic rows, actor rows) pair.
    if separate_actor_batch:
      transitions, actor_transitions = transitions
      bc_transitions = None
    elif isinstance(transitions, Transition):
      bc_transitions = None
      actor_transitions = transitions
    else:
      transitions, bc_transitions = transitions
      actor_transitions = transitions
    key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
    if adaptive_entropy_coefficient:
      alpha_loss_value, alpha_grads = alpha_grad(
          state.alpha_params, state.policy_params, actor_transitions, key_alpha)
      alpha = jnp.exp(state.alpha_params)
    else:
      alpha = config.entropy_coefficient

    if not config.use_gcbc:
      (critic_loss_value, critic_metrics), critic_grads = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          transitions, key_critic)

    (actor_loss_value, actor_aux), actor_grads = actor_grad(
        state.policy_params, state.q_params, alpha, actor_transitions, key_actor,
        bc_transitions)

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
