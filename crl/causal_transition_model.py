"""Causal transition model (ETT stage 0): dynamics and a blind expert policy.

``CausalTransitionModel`` bundles the supervised heads that are fitted on
the frozen offline dataset before the first contrastive gradient step:

  * ``transition`` -- the one-step dynamics
    ``f_theta(s_t, a_t, a'_t) -> s_{t+1}`` described at length below
    (unchanged; this is the old ``TransitionModel``);
  * ``policy`` -- a BLIND expert policy over actions at ``s_t``. The dataset
    was collected by a SIGHTED expert that reads the rockfall latent at the
    band mouth, but that latent is audit-only: it never enters the stored
    observation (``crl/offline_audit.py`` gate G6). So the head sees exactly
    the learner state the agent sees -- no rock-trap status -- and is fitted
    to reproduce the sighted expert's action. Where the two disagree is
    precisely where the decision needed the hidden latent, which is the
    quantity the causal-transition line of work is after.

Both heads are fitted in ONE loop (``fit_causal``) on the same minibatches,
the same episode-level holdout and one Adam optimizer over the joint parameter
tree, so a single fit produces a single self-consistent artifact. The
dynamics-only ``fit`` is kept for callers that want just ``f(s, a)``.

``CausalTransitionModel.sample`` composes the fitted heads forward: it draws
states from the frozen dataset, draws ``a_hat`` from the expert head and then
``s'_hat`` from the dynamics head, and INHERITS the drawn rows' goal because
no head predicts one. With ``off_diagonal='agent'`` the third input block is
the CURRENT contrastive learner's action at the same state, so a generated
transition reads "the expert would have played ``a``, the agent plays ``a'``":
the diagonal where the two agree, the failure branch where they do not. Which
of the two a given step takes is decided by the expert head's own coin,
``sum_i w_i(s) ** 2`` -- the probability that two independent draws of the
hidden confounder land in the same branch -- so the counterfactual query fires
in proportion to how much the confounder matters at that state, and never at a
state where one branch holds all the weight. Handed the stored actions and next states it returns a
self-comparing batch (``SampledTransitions.compare``), which is the cheapest
end-to-end check that the two heads agree with each other and not merely with
their own regression targets -- and it splits the next-state error into the
dynamics head's own and the part the generated action handed it. ``rollout``
is the same step repeated to a horizon, branched off a real dataset episode.

**A note on what used to be here.** A third head, a dataset-membership
discriminator ``D_psi(s_t, a_t)``, has been REMOVED. Its one job at sample
time was to supply the coin that chose between the dynamics head's factual
diagonal and its counterfactual off-diagonal, via the density reading
``logit D - log Vol(box)``, and the mixture expert head below does that job
properly: a normalized conditional instead of an unnormalized ratio against a
uniform, and a coin with no temperature to tune. The measured problem with the
old coin is on the record: on the V5 far05 fit ``D(s, pi(s)) > 0.99`` at 95.8%
of dataset states, so the counterfactual branch fired on about 1% of rows and
the only way to move it was a temperature that turned a probability into a
hyperparameter. Checkpoints written with the head still load -- ``save``
stores plain dicts, never class instances, so ``load`` simply drops the
payload with a note.

The expert head, specifically. Two are available, and the choice is
``dyn_policy_head``:

  * **Input**, both heads. The raw 29-dim learner state, standardized with the
    SAME train-split statistics the dynamics head uses. The goal half of the
    stored observation is dropped exactly as it is for dynamics.
  * ``'mdn'`` **(default): a mixture density network**, ``k =
    dyn_policy_components`` diagonal Gaussian branches over standardized
    actions, censored to the train-split action box, fitted by maximum
    likelihood. This is the head that makes ``sample`` meaningful. The
    conditional being fitted is the sighted teacher's action distribution
    MARGINALIZED over the hidden confounder ``u``,
    ``p(a|s) = sum_u p(u|s) pi_sighted(a | s, u)``, so it is multi-modal by
    construction -- and an MSE fit to a multi-modal conditional returns its
    conditional MEAN, which at the band mouth is an average of "go straight"
    and "detour": a torque in neither branch. Three details that are decisions
    rather than defaults, each argued where it is implemented:
      - the box is modelled by CENSORING, not by a tanh squash. The action
        box is exactly the actuator's ``[-1, 1]``, some torques sit exactly on
        it (0.11% of coordinates on V5 far05), and a squashed Gaussian's
        log-density is ``-inf`` on any such row -- one is enough to make the
        loss infinite. A clip is also what the actuator physically did. See
        ``censored_component_log_prob``;
      - ``log_sigma`` is clipped to ``[dyn_policy_log_sigma_min,
        dyn_policy_log_sigma_max]``, because a Gaussian mixture's likelihood
        is unbounded above and a component can otherwise collapse onto one
        data point. See ``split_mixture``;
      - the component means are pushed apart at init, because Haiku's zero
        bias init starts them all in the same place. See
        ``_spread_mixture_means``.
  * ``'deterministic'``: the historical MSE regression, kept so the two can be
    compared on the same data and so every pre-mixture config refits
    unchanged. ``dyn_policy_squash=True`` emits pre-squash logits and maps
    them through a tanh onto the train-split action box;
    ``dyn_policy_squash=False`` regresses standardized actions linearly and
    clips at predict time.
  * **Do not compare the two on RMSE.** A better mixture can have a worse
    point error, because the mean of a bimodal conditional is closer to both
    modes than either mode is to the other. ``mdn_val_nll`` is the mixture's
    own headline number and ``dyn_policy_components=1`` is its control: a k=1
    fit that matches k=3 says the multi-modality is not in the actions.
  * **Baseline.** Every report carries the error of the constant predictor
    "always play the train-split mean action", the expert-policy analogue of
    the persistence baseline below. A policy that does not beat it has learned
    nothing.

The dynamics head, in detail:

A plain ReLU MLP fitted by supervised regression on the SAME frozen offline
dataset the contrastive learner trains on, and fitted BEFORE the first CRL
gradient step (see ``crl/ETT_train.py``). States are the 29-dim learner state
half of the stored observation; the env's stored goal half and the relabeled
contrastive goal play no part here -- dynamics are goal-independent.

Fitting details, and why each is here:

  * **Standardized inputs.** The ant state mixes units (torso xy in metres up
    to ~9, quaternions in [-1, 1], joint velocities in the tens), so an
    unnormalized MSE would be dominated by a handful of columns. Inputs and
    regression targets are standardized with statistics computed on the TRAIN
    split only.
  * **Residual target (default).** The network predicts the normalized
    ``delta = s_{t+1} - s_t`` and the prediction is ``s_t + delta``. Over a
    single MuJoCo control step the state barely moves, so regressing the delta
    removes the identity map the network would otherwise have to memorize.
    ``dyn_predict_delta=False`` regresses the absolute next state instead.
  * **Episode-level holdout.** The validation split is whole episodes, never
    random transitions: consecutive transitions within an episode are near
    duplicates, so a transition-level split reports an optimistic error.
  * **Activation and LR schedule.** ``dyn_activation`` / ``dyn_lr_schedule``
    both default to the historical setting (``relu`` / ``const``), so any
    existing config refits bit-for-bit. On the rockfall ant, ``silu`` plus a
    ``cosine`` decay is a large, repeatable win: SiLU because a ReLU net is
    piecewise linear and rigid-body flow is smooth, and the decay because a
    fixed LR leaves the fit oscillating on minibatch noise for its whole
    second half.
  * **Spectral normalization (opt-in).** ``dyn_spectral_norm=True`` divides
    every layer's weight -- both heads, output layer included -- by its own
    largest singular value and rescales by ``dyn_spectral_norm_coef``, so the
    whole map is Lipschitz-bounded by ``coef ** n_layers``. That constrains
    exactly what a one-step model is asked for: a bounded change of predicted
    next state per unit change of input, which is what keeps multi-step
    rollouts and off-manifold queries from exploding. The estimate is
    recomputed from the weights on every forward pass rather than carried in
    module state, so a reloaded model predicts precisely as it trained -- at
    a measured ~35% on fit wall time.
  * **Counterfactual-action input (default).** ``dyn_next_action_input=True``
    widens the input from ``concat(s, a)`` to ``concat(s, a, a')``, where
    ``a'`` is the COUNTERFACTUAL action: the action that could have been
    played at the SAME state ``s_t`` in place of the stored ``a_t``. It is not
    ``a_{t+1}`` -- despite the legacy ``next_action`` spelling kept on the
    config field and throughout this module, nothing about ``a'`` is a
    temporal successor. It lives in the same action space as ``a`` and is
    standardized by the same train-split statistics. Because nothing
    counterfactual was ever executed in the environment, every stored
    transition is FACTUAL: ``a' = a`` on every row of the frozen offline
    dataset, the extra block is an exact COPY of the previous one, and the
    POSITIVE data supervises the diagonal alone. Two consequences, and both
    matter before any number from an off-diagonal (genuinely counterfactual)
    query on a positive-only fit is believed:

      - the fit itself neither gains nor loses anything measurable -- the
        diagonal is all the data there is, and the network can represent the
        old ``f(s, a)`` by ignoring either block;
      - the off-diagonal map ``a' != a`` -- the counterfactual direction, and
        the only one worth carrying the block for -- is NOT IDENTIFIED, and on
        this architecture that is provable rather than merely likely. Because
        every row has ``a' = a``, the gradient with respect to the ``a`` block
        of the first layer is EQUAL to the gradient with respect to the ``a'``
        block at every step, and Adam's update is elementwise -- so the
        DIFFERENCE between the two blocks never moves from its initialization.
        Measured on a 3k-step fit of the V5 far05 dataset: the difference
        changes by 7e-7 (float noise) while the sum moves by 0.29. The model's
        entire response to ``a' != a`` is therefore the random init, frozen.
        Consistently with that, ``action_input_sensitivity`` (below) measures a
        counterfactual-block share of 0.50 +- 0.01 across seeds with a block
        cosine of only ~0.4-0.5, and swapping in another transition's ``a'``
        moves the prediction by about half of one true step -- a large response
        that no gradient ever supervised. ``fit`` and ``fit_causal`` put those
        numbers in the stats so the arbitrariness is on the record rather
        than hidden in the pickle. Only the factual direction ``da = da'``
        reproduces the physical Jacobian; use the counterfactual input as
        plumbing for the objective below, not as a counterfactual oracle on
        its own.

  * **Off-diagonal supervision from a NEGATIVE dataset.** That objective now
    exists, and ``config.dyn_negative_dataset`` turns it on. Given a
    companion collection of FAILURE episodes for the same benchmark, every
    fit step adds a second dynamics term: a minibatch of negative transitions
    ``(s, a, s')``, a counterfactual ``a' != a`` drawn for each row and
    REDRAWN at every step (``config.dyn_off_diagonal_train_action``), and the
    same head regressed ``f(s, a, a') -> s'`` on it. The total loss becomes
    ``dyn_mse + dyn_off_diagonal_coef * dyn_offdiag_mse + dyn_policy_coef *
    policy_loss``. What the head then holds is two branches of one map: at
    ``a' = a`` the expert dynamics the positive data recorded, at ``a'``
    unlike ``a`` the dynamics the failure data recorded. The gradient
    argument above is exactly what this repairs -- the two action blocks stop
    receiving identical gradients, so their difference moves.

    What it is NOT: ``a'`` is drawn independently of ``s'``, so the
    off-diagonal fits ``E[s' | s, a]`` over the negative data marginalized
    over ``a'``. It is a branch selected by ``a'`` being unlike ``a``, not a
    response to the particular ``a'``. ``TransitionModel.off_diagonal_trained``
    records whether a given artifact was fitted this way, because an
    off-diagonal prediction means completely different things in the two
    regimes and the weights do not say which.

  * **Persistence baseline.** Every report also carries the error of the
    trivial predictor ``s_{t+1} = s_t``. A dynamics model that does not beat it
    by a wide margin has learned nothing, and on a residual parameterization
    that failure looks like a small, healthy-seeming loss.

The fitted model enters no contrastive LOSS: it is saved, reported, and read
by the dataset-augmentation path in ``crl/ETT_train.py``, which generates
trajectories from it and mixes them into the minibatches. See
``notes/CAUSAL_TRANSITION_V0.md``.
"""
import dataclasses
import json
import os
import pickle
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


def _standardizer(x, floor=1e-6):
  """(mean, std) over axis 0; constant columns get std=1 (no divide by zero)."""
  mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
  std = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
  std = np.where(std < floor, np.float32(1.0), std).astype(np.float32)
  return mean, std


#: Selectable hidden activations. ``relu`` is the historical default and is
#: what every model pickled before this knob existed was fitted with, so it
#: stays the default everywhere (Config, ``build_network``, ``TransitionModel``).
ACTIVATIONS = {'relu': jax.nn.relu, 'silu': jax.nn.silu, 'gelu': jax.nn.gelu,
               'tanh': jnp.tanh}


#: Weight init used by every layer of both heads. Pulled out of the two
#: builders so the spectral-norm layers below start from EXACTLY the same
#: draw as the plain ones: with the same seed, an SN fit and a plain fit have
#: identical initial weights and differ only by the normalization.
_W_INIT = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')


def _spectral_scale(w, coef, n_steps, eps=1e-12):
  """``coef / sigma(w)``: sigma from ``n_steps`` power iterations on ``w``.

  Deliberately STATELESS. Haiku's own ``hk.SpectralNorm`` carries the power
  iteration vector in module state, which would force
  ``hk.transform_with_state`` through every fit / predict / save / load path
  in this module and invalidate every pickle already written. Starting each
  forward pass from a FIXED vector instead makes sigma a pure function of the
  weights, so a reloaded params tree reproduces the training-time forward pass
  exactly. The mat-vecs are not free: they are launch-latency bound rather
  than FLOP bound, and cost ~35% of the wall time of a 3k-step (512, 512)
  ``fit_causal`` on this machine's GPU. At ``n_steps=10`` the estimate lands
  within ~2% BELOW the true sigma on 512-wide weights (so the effective
  per-layer norm is ~1.02 * coef); 20 iterations roughly halves that for
  another ~6% of wall time.

  The iterates are ``stop_gradient``-ed, which is the usual SN gradient:
  ``d sigma / dW = v u^T`` is exact once the iteration has converged.
  """
  in_dim, out_dim = w.shape
  frozen = jax.lax.stop_gradient(w)
  # Fixed key, not an hk.next_rng_key: the estimate must not depend on any
  # RNG the caller happens to be threading (``predict`` threads none).
  u = jax.random.normal(jax.random.PRNGKey(0), (out_dim,), w.dtype)
  u = u / (jnp.linalg.norm(u) + eps)
  v = jnp.zeros((in_dim,), w.dtype)
  for _ in range(max(1, int(n_steps))):
    v = frozen @ u                       # w is [in, out]: maps R^out -> R^in.
    v = v / (jnp.linalg.norm(v) + eps)
    u = frozen.T @ v
    u = u / (jnp.linalg.norm(u) + eps)
  sigma = jnp.einsum('i,ij,j->', v, w, u)
  return coef / (jnp.abs(sigma) + eps)


class _SNLinear(hk.Module):
  """``hk.Linear`` whose weight is rescaled to spectral norm ``coef``.

  Same parameters as ``hk.Linear`` (``w`` [in, out] and ``b``) and the same
  init, so only the forward pass differs. The param PATHS do differ from
  ``hk.nets.MLP``'s (``transition_mlp/linear_0`` rather than
  ``transition_mlp/~/linear_0``), so a params tree fitted with the flag on
  cannot be loaded with it off; ``spectral_norm`` rides inside the pickled
  model precisely so that never has to be guessed.
  """

  def __init__(self, output_size, coef, n_steps, name=None):
    super().__init__(name=name)
    self.output_size = int(output_size)
    self.coef = float(coef)
    self.n_steps = int(n_steps)

  def __call__(self, x):
    w = hk.get_parameter('w', [x.shape[-1], self.output_size], x.dtype,
                         init=_W_INIT)
    b = hk.get_parameter('b', [self.output_size], x.dtype, init=jnp.zeros)
    return jnp.dot(x, w * _spectral_scale(w, self.coef, self.n_steps)) + b


class _SNMLP(hk.Module):
  """``hk.nets.MLP`` with spectral normalization on EVERY layer.

  The output layer is normalized too, which is what bounds the Lipschitz
  constant of the whole map by ``coef ** n_layers``. That also caps how much
  the head can stretch its (standardized) targets, so a fit that underfits
  visibly wants a ``coef`` above 1 rather than the SN switched off.
  """

  def __init__(self, output_sizes, activation, coef, n_steps, name=None):
    super().__init__(name=name)
    self.output_sizes = list(output_sizes)
    self.activation = activation
    self.coef = float(coef)
    self.n_steps = int(n_steps)

  def __call__(self, x):
    last = len(self.output_sizes) - 1
    for i, out in enumerate(self.output_sizes):
      x = _SNLinear(out, self.coef, self.n_steps, name=f'linear_{i}')(x)
      if i != last:
        x = self.activation(x)
    return x


def _sn_settings(config):
  """The three ``dyn_spectral_norm*`` knobs, defaulted for older configs."""
  return {
      'spectral_norm': bool(getattr(config, 'dyn_spectral_norm', False)),
      'sn_coef': float(getattr(config, 'dyn_spectral_norm_coef', 1.0)),
      'sn_iters': int(getattr(config, 'dyn_spectral_norm_iters', 10)),
  }


def build_network(state_dim: int, hidden_layer_sizes: Sequence[int],
                  activation: str = 'relu', spectral_norm: bool = False,
                  sn_coef: float = 1.0, sn_iters: int = 10):
  """The MLP itself: ``concat([s_norm, a_norm(, a'_norm)]) -> state_dim``.

  The input width is inferred from the first argument Haiku sees, so this
  builder is shared by the ``(s, a)`` and ``(s, a, a')`` parameterizations
  (``a'`` the counterfactual action); the legacy-named
  ``TransitionModel.next_action_input`` flag records which one a given params
  tree was fitted with.

  With ``spectral_norm`` every layer (the output layer included) divides its
  weight by its own spectral norm and multiplies by ``sn_coef``.
  """
  sizes = list(hidden_layer_sizes) + [int(state_dim)]
  if activation not in ACTIVATIONS:
    raise ValueError(f'unknown dyn_activation {activation!r}; '
                     f'expected one of {sorted(ACTIVATIONS)}')
  act_fn = ACTIVATIONS[activation]

  def _fn(x):
    if spectral_norm:
      return _SNMLP(sizes, act_fn, sn_coef, sn_iters, name='transition_mlp')(x)
    return hk.nets.MLP(
        sizes,
        w_init=_W_INIT,
        activation=act_fn,
        name='transition_mlp')(x)

  return hk.without_apply_rng(hk.transform(_fn))


def build_policy_network(action_dim: int, hidden_layer_sizes: Sequence[int],
                         activation: str = 'relu',
                         spectral_norm: bool = False, sn_coef: float = 1.0,
                         sn_iters: int = 10):
  """The blind expert head: ``s_norm -> action_dim`` pre-squash logits."""
  sizes = list(hidden_layer_sizes) + [int(action_dim)]
  if activation not in ACTIVATIONS:
    raise ValueError(f'unknown activation {activation!r}; '
                     f'expected one of {sorted(ACTIVATIONS)}')
  act_fn = ACTIVATIONS[activation]

  def _fn(x):
    if spectral_norm:
      return _SNMLP(sizes, act_fn, sn_coef, sn_iters,
                    name='expert_policy_mlp')(x)
    return hk.nets.MLP(
        sizes,
        w_init=_W_INIT,
        activation=act_fn,
        name='expert_policy_mlp')(x)

  return hk.without_apply_rng(hk.transform(_fn))


def build_mixture_policy_network(action_dim: int, n_components: int,
                                 hidden_layer_sizes: Sequence[int],
                                 activation: str = 'relu',
                                 spectral_norm: bool = False,
                                 sn_coef: float = 1.0, sn_iters: int = 10):
  """The mixture expert head: ``s_norm -> k * (1 + 2 * action_dim)``.

  One flat vector per state, which ``split_mixture`` slices into the mixture
  weight LOGITS ``[k]``, the component means ``[k, A]`` and the component log
  standard deviations ``[k, A]``. Emitting logits and logs rather than
  normalized weights and positive sigmas keeps the head unconstrained, so the
  optimizer never has to respect a simplex or a positivity bound.

  Diagonal covariance per component, deliberately: a full one costs
  ``k * A * (A + 1) / 2`` outputs and a Cholesky factorization per state for
  an 8-dim action, and the components are here to represent the hidden
  confounder's BRANCHES, not the within-branch torque correlations. If a
  diagonal mixture turns out to need extra components purely to fake
  correlation, that shows up as held-out likelihood still improving in ``k``
  past the number of branches -- which is the reason the fit reports the
  likelihood per ``k`` rather than fixing it.
  """
  k = int(n_components)
  a = int(action_dim)
  if k < 1:
    raise ValueError(f'n_components must be at least 1, got {n_components}')
  sizes = list(hidden_layer_sizes) + [k * (1 + 2 * a)]
  if activation not in ACTIVATIONS:
    raise ValueError(f'unknown activation {activation!r}; '
                     f'expected one of {sorted(ACTIVATIONS)}')
  act_fn = ACTIVATIONS[activation]

  def _fn(x):
    if spectral_norm:
      return _SNMLP(sizes, act_fn, sn_coef, sn_iters,
                    name='mixture_policy_mlp')(x)
    return hk.nets.MLP(
        sizes,
        w_init=_W_INIT,
        activation=act_fn,
        name='mixture_policy_mlp')(x)

  return hk.without_apply_rng(hk.transform(_fn))


def split_mixture(out, action_dim, n_components,
                  log_sigma_min=-5.0, log_sigma_max=2.0):
  """Slice a flat mixture-head output into ``(log_w, mu, log_sigma)``.

  ``log_w`` is ``[..., k]`` and already log-normalized (a log-softmax over the
  weight logits), ``mu`` and ``log_sigma`` are ``[..., k, A]``.

  ``log_sigma`` is CLIPPED to ``[log_sigma_min, log_sigma_max]``. The floor is
  not cosmetic: the likelihood of a Gaussian mixture is unbounded above -- a
  component can drive its sigma to zero on top of a single data point and take
  the objective to ``-inf`` -- and the floor is what makes the maximum
  well-posed. The ceiling stops a component that has stopped winning any
  responsibility from drifting out to a flat prior it cannot return from.
  Clipping (rather than a softplus or an exp of a bounded activation) is
  chosen so the gradient is exactly zero outside the range instead of merely
  small, which makes a saturated component visible in the diagnostics rather
  than slowly moving.
  """
  k, a = int(n_components), int(action_dim)
  logits = out[..., :k]
  mu = out[..., k:k + k * a].reshape(*out.shape[:-1], k, a)
  ls = out[..., k + k * a:].reshape(*out.shape[:-1], k, a)
  return (jax.nn.log_softmax(logits, axis=-1), mu,
          jnp.clip(ls, log_sigma_min, log_sigma_max))


def censored_component_log_prob(mu, log_sigma, a, lo, hi):
  """Per-component log-likelihood of ``a`` under a mixture CENSORED to the box.

  ``mu`` / ``log_sigma`` are ``[..., k, A]``, ``a`` is ``[..., A]`` and ``lo``
  / ``hi`` are ``[A]``; the result is ``[..., k]``.

  Why censored rather than a plain Gaussian or a tanh squash. Some of the
  stored torques sit EXACTLY on the box limit -- measured at 0.11% of action
  coordinates on the V5 far05 dataset, whose train-split box is exactly
  [-1, 1] per dimension, i.e. the actuator's own range, so a censoring limit
  here IS the physical limit and not an artifact of the observed extremes.
  (The "a third of the torques saturate" figure quoted elsewhere in this repo
  is the v2.1 PILOT's, not this dataset's.) A boundary atom is rare here but
  not negligible, and one of the three candidate parameterizations breaks
  outright on a single such row:

    * a tanh-squashed Gaussian is UNDEFINED on those rows. Its log-density
      carries ``-log(1 - u^2)`` with ``u`` the action rescaled to ``[-1, 1]``,
      which is ``-inf`` at a saturated torque. Rarity does not help: ONE such
      row makes the minibatch loss infinite, so 0.11% is not a small error
      but a fit that does not run;
    * a plain Gaussian is finite but wrong in the same place, spreading the
      atom's mass smoothly across the boundary and out of the box;
    * censoring is what the data-generating process actually did. A saturating
      controller IS a clip: the torque it wanted fell outside the actuator's
      range and the actuator returned the limit. So the model here is
      ``a = clip(z, lo, hi)`` with ``z`` from a diagonal Gaussian mixture, and
      the likelihood of a boundary observation is the TAIL MASS the clip piled
      up there rather than a density.

  Concretely, per dimension: the Gaussian log-density strictly inside the box,
  ``log Phi((lo - mu) / sigma)`` at the lower limit and
  ``log Phi((mu - hi) / sigma)`` at the upper one, summed over dimensions
  because the components are diagonal.

  The result therefore mixes log-densities (interior dimensions) with log
  PROBABILITIES (censored ones) and is not a pure density: it is comparable
  across fits on the same data and the same units, which is what a model
  selection over ``k`` needs, but it is not a quantity whose exponential
  integrates to one over the box.
  """
  sigma = jnp.exp(log_sigma)
  a_k = a[..., None, :]
  z = (a_k - mu) / sigma
  log_pdf = -0.5 * z ** 2 - log_sigma - 0.5 * jnp.log(2.0 * jnp.pi)
  # log Phi of the standardized distance to each limit -- log_ndtr is the
  # stable form; the literal log(ndtr(x)) underflows to -inf by x ~ -6, which
  # is exactly where a confident component sits.
  log_lo = jax.scipy.special.log_ndtr((lo - mu) / sigma)
  log_hi = jax.scipy.special.log_ndtr((mu - hi) / sigma)
  at_lo = (a <= lo)[..., None, :]
  at_hi = (a >= hi)[..., None, :]
  per_dim = jnp.where(at_lo, log_lo, jnp.where(at_hi, log_hi, log_pdf))
  return jnp.sum(per_dim, axis=-1)


def censored_mixture_log_prob(log_w, mu, log_sigma, a, lo, hi):
  """``log p(a|s)`` of the censored mixture: logsumexp over the components."""
  return jax.nn.logsumexp(
      log_w + censored_component_log_prob(mu, log_sigma, a, lo, hi), axis=-1)


def _spread_mixture_means(policy_params, n_components, action_dim, spread,
                          key):
  """Push the k component means apart at init. Returns a new params tree.

  Haiku initializes every bias to zero, so a freshly initialized mixture head
  starts with all ``k`` component means at the same point (the dataset's mean
  action, in standardized units) and they differ only through the random
  weight rows of the output layer. That is a weak symmetry to break: the
  components see nearly equal responsibilities, receive nearly equal
  gradients, and a mixture that starts collapsed can stay collapsed -- which
  looks exactly like "the data is unimodal" and is not.

  So the ``mu`` slice of the output layer's BIAS is seeded with
  ``N(0, spread ** 2)`` draws, in standardized action units: ``spread = 1``
  puts the components one action-standard-deviation apart on average, which is
  the scale the branches actually differ on. The weight logits and log-sigmas
  are left at zero -- equal weights and unit sigmas are a fine starting point,
  and it is the means that need separating.

  The output layer is found by SHAPE, not by name: ``hk.nets.MLP`` and the
  spectral-norm MLP give their layers different parameter paths, and the head
  has exactly one bias of width ``k * (1 + 2 * A)``.
  """
  k, a = int(n_components), int(action_dim)
  width = k * (1 + 2 * a)
  hits = [name for name, p in policy_params.items()
          if 'b' in p and p['b'].shape == (width,)]
  if len(hits) != 1:
    raise ValueError(f'expected exactly one output bias of width {width} in '
                     f'the mixture head, found {len(hits)}: {hits}')
  name = hits[0]
  b = policy_params[name]['b']
  mu = jax.random.normal(key, (k * a,)) * float(spread)
  out = dict(policy_params)
  out[name] = dict(out[name])
  out[name]['b'] = b.at[k:k + k * a].set(mu)
  return out


def build_optimizer(config, steps):
  """Adam with the configured learning-rate schedule.

  ``dyn_lr_schedule='const'`` reproduces the original fixed-LR optimizer
  exactly. ``'cosine'`` warms up from lr/10 to lr over
  ``dyn_lr_warmup_frac`` of the run and then cosine-decays to
  ``dyn_lr_end_frac * lr``; a fixed LR otherwise leaves the fit bouncing
  around its own optimum on minibatch noise for the whole second half.
  """
  lr = float(config.dyn_learning_rate)
  sched = str(config.dyn_lr_schedule)
  if sched == 'const':
    return optax.adam(lr, eps=1e-7), lr
  if sched == 'cosine':
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr,
        warmup_steps=int(float(config.dyn_lr_warmup_frac) * steps),
        decay_steps=max(int(steps), 1),
        end_value=lr * float(config.dyn_lr_end_frac))
    return optax.adam(schedule, eps=1e-7), schedule
  raise ValueError(f'unknown dyn_lr_schedule {sched!r} '
                   "(expected 'const' or 'cosine')")


# -- THE MAZE FLOOR PLAN ----------------------------------------------------
#
# The feasible-box clip on `TransitionModel` bounds xy by the maze's BOUNDING
# BOX, and a box is a product of intervals: on every map in this family that
# has an interior wall -- the U-maze's central bar, the V6/V7 ring -- the box
# also contains the wall. A generated chain is therefore free to walk straight
# through the block in the middle of the map and still be "in bounds", which
# on a two-route benchmark is the one excursion that matters, because it
# invents a shortcut between the two routes the whole benchmark is about.
#
# `maze_free_cells` reads the traversable cells off a live env so that
# `clip_state` can project xy onto the union of the OPEN cells instead, which
# is the room rather than its bounding box. It duck-types the env -- `_open`
# and `_torso_offset`, the same two attributes the env's own `_cell_xy` uses
# -- and takes the cell size as an argument, so this module never imports an
# environment module and nothing about a particular map is written down here.
# The caller that knows which env it is holding (crl/ETT_train.py and the
# evaluation scripts) is the one that passes the two together.
def maze_free_cells(env, scaling):
  """``(centres, half)``: the maze's open cells as squares in torso xy units.

  ``centres`` is a ``[m, 2]`` float32 array of open-cell CENTRES and ``half``
  is half a cell, so the traversable region is the union of the m squares
  ``[cx - half, cx + half] x [cy - half, cy + half]`` -- exactly the region
  the env's wall geoms leave open, since a wall occupies a whole cell and the
  inner face of a wall is half a cell from the centre of its open neighbour.

  Args:
    env: an ant maze env exposing ``_open`` (traversable ``(row, col)`` cells)
      and ``_torso_offset``, i.e. a ``D4rlAntUMazeEnv`` or any subclass. The
      cells are static geometry: rocks and gates that open and close during an
      episode are NOT walls and do not appear here.
    scaling: the maze cell size the env's XML was built with
      (``crl.d4rl_ant.SCALING`` for this family).

  Raises:
    TypeError: if ``env`` does not look like an ant maze env, so a caller
      cannot silently get a floor plan belonging to a different map.
    ValueError: if the env reports no open cells at all.
  """
  cells = getattr(env, '_open', None)
  offset = getattr(env, '_torso_offset', None)
  if cells is None or offset is None:
    raise TypeError(
        f'{type(env).__name__} does not look like an ant maze env: '
        'maze_free_cells needs _open and _torso_offset')
  if not len(cells):
    raise ValueError(f'{type(env).__name__} reports no open cells')
  tx, ty = float(offset[0]), float(offset[1])
  s = float(scaling)
  # The env's own _cell_xy, vectorized: (r, c) -> (c * scaling - tx,
  # r * scaling - ty). Kept identical to it deliberately -- a floor plan that
  # disagreed with the env about where a cell is would clip generated states
  # onto the wrong squares and nothing downstream would notice.
  centres = np.array([[int(c) * s - tx, int(r) * s - ty] for r, c in cells],
                     np.float32)
  return centres, 0.5 * s


@dataclasses.dataclass
class TransitionModel:
  """Fitted dynamics model: parameters + the normalization it was fitted with.

  ``predict(states, actions)`` takes and returns RAW (unnormalized) state units,
  so callers never have to know the standardization exists. When the model was
  fitted with ``next_action_input`` it accepts a third argument, the
  COUNTERFACTUAL action ``a'`` -- the action that could have been played at
  the same ``s_t`` instead of ``a_t``, not ``a_{t+1}``; omitting it queries the
  FACTUAL diagonal ``a' = a``, which is the only place the training data ever
  put it (see the module docstring).
  """
  params: Any
  state_dim: int
  action_dim: int
  hidden_layer_sizes: Tuple[int, ...]
  predict_delta: bool
  state_mean: np.ndarray
  state_std: np.ndarray
  action_mean: np.ndarray
  action_std: np.ndarray
  target_mean: np.ndarray
  target_std: np.ndarray
  # Defaulted so that models pickled before this field existed still load;
  # every one of those was fitted with ReLU.
  activation: str = 'relu'
  # Likewise defaulted: every model pickled before these existed was fitted
  # without spectral normalization. They are part of the artifact because the
  # forward pass cannot be reconstructed without them.
  spectral_norm: bool = False
  spectral_norm_coef: float = 1.0
  spectral_norm_iters: int = 10
  # Defaulted False: every model pickled before the counterfactual action
  # existed was fitted on concat(s, a), and its input width has to be
  # reconstructed from the artifact rather than from whatever the current
  # config says. The field name is legacy -- a' is the counterfactual action,
  # not a_{t+1} -- and is kept unchanged so existing pickles still load.
  next_action_input: bool = False
  # Whether a gradient ever supervised a' != a, i.e. whether the fit was given
  # a NEGATIVE dataset for the off-diagonal term. It is part of the artifact
  # because it is the difference between an off-diagonal prediction that means
  # something and one that is the random init: a consumer cannot tell them
  # apart from the weights. Defaulted False, which is what every model pickled
  # before the off-diagonal objective existed was.
  off_diagonal_trained: bool = False
  # The FEASIBLE BOX of the state, two [state_dim] vectors, or None for an
  # unbounded model. `predict` never touches them -- it stays the raw head, so
  # the one-step held-out error is still the head's own -- but `clip_state`
  # applies them, and `rollout` calls it on every state it GENERATES. A chain
  # that feeds its own output back in is the only place the head is asked
  # about a state no episode ever visited, and the only place a single bad
  # step is inherited by every step after it.
  # crl.d4rl_ant.ant_state_bounds derives the box for this family of envs.
  # Defaulted None so that models pickled before these fields existed still
  # load, and load unclipped, which is what they were fitted and evaluated as.
  state_lo: Optional[np.ndarray] = None
  state_hi: Optional[np.ndarray] = None
  # THE MAZE FLOOR PLAN, the part of the feasibility constraint the box above
  # cannot express. `free_xy` is a [m, 2] array of open-cell centres and
  # `free_half` half a cell, so the traversable region is the union of m
  # squares; `clip_state` projects the xy of a generated state onto it. This
  # is what stops a generated chain from walking through the block in the
  # middle of the map, which the bounding box lets through by construction.
  # crl.causal_transition_model.maze_free_cells reads the pair off a live env.
  # Defaulted None/0 so that models pickled before these fields existed still
  # load, and load with walls ignored, which is what they were evaluated as.
  free_xy: Optional[np.ndarray] = None
  free_half: float = 0.0
  _apply: Optional[Any] = dataclasses.field(
      default=None, repr=False, compare=False)

  @property
  def clips_states(self):
    """True when a feasible BOX was attached (the per-dimension clip)."""
    return self.state_lo is not None and self.state_hi is not None

  @property
  def clips_walls(self):
    """True when a maze floor plan was attached (the xy wall projection)."""
    return self.free_xy is not None and self.free_half > 0.0

  @property
  def constrains_states(self):
    """True when `clip_state` does any work at all -- box, walls or both."""
    return self.clips_states or self.clips_walls

  def set_state_bounds(self, bounds):
    """Attach (or with None, drop) the feasible box. Returns self.

    Both vectors are checked against `state_dim` and against each other here
    rather than at rollout time, because a box that is empty on one dimension
    would collapse that coordinate of every generated trajectory to a constant
    and nothing downstream would report it.
    """
    if bounds is None:
      self.state_lo = self.state_hi = None
      return self
    lo, hi = bounds
    lo = np.asarray(lo, np.float32).reshape(-1)
    hi = np.asarray(hi, np.float32).reshape(-1)
    if lo.shape != (self.state_dim,) or hi.shape != (self.state_dim,):
      raise ValueError(f'state bounds must both be [{self.state_dim}], got '
                       f'{lo.shape} and {hi.shape}')
    if not np.all(lo < hi):
      bad = [int(i) for i in np.flatnonzero(~(lo < hi))]
      raise ValueError(f'empty state box on dimensions {bad}')
    self.state_lo, self.state_hi = lo, hi
    return self

  def set_free_cells(self, cells):
    """Attach (or with None, drop) the maze floor plan. Returns self.

    ``cells`` is the ``(centres, half)`` pair ``maze_free_cells`` returns.
    Validated here rather than at rollout time, because a floor plan built
    from the wrong map or the wrong cell size would quietly pull every
    generated trajectory onto squares that are not this maze's, and the only
    symptom downstream would be a rollout that looks oddly well behaved.
    """
    if cells is None:
      self.free_xy, self.free_half = None, 0.0
      return self
    centres, half = cells
    centres = np.asarray(centres, np.float32)
    if centres.ndim != 2 or centres.shape[1] != 2:
      raise ValueError(f'free cell centres must be [m, 2], got {centres.shape}')
    if not centres.shape[0]:
      raise ValueError('free cell centres are empty: no traversable region')
    half = float(half)
    if not half > 0.0:
      raise ValueError(f'free cell half-width must be positive, got {half}')
    if self.state_dim < 2:
      raise ValueError(f'a maze floor plan needs an xy state, but this model '
                       f'has state_dim={self.state_dim}')
    self.free_xy, self.free_half = centres, half
    return self

  def clip_xy(self, xy, chunk=65_536):
    """Project xy onto the union of the maze's open cells; a no-op without one.

    The exact Euclidean projection: the projection onto a union of closed sets
    is the nearest of the projections onto its members, and the projection
    onto an axis-aligned square is a per-coordinate clamp. A point that is
    already inside some open cell clamps to itself there at distance zero, so
    this is the identity on every feasible state and moves only the ones that
    are inside a wall -- or, if the box clip is off, outside the maze
    entirely.

    A projected point lands ON the wall face rather than a torso radius clear
    of it, which is the same convention the feasible box uses: the purpose is
    to stop a chain from crossing a wall, not to model contact.

    Args:
      xy: [..., 2] raw xy in torso units.
      chunk: rows per block of the [n, m] distance matrix, since m is the
        whole floor plan and n can be a full dataset.
    """
    p = np.asarray(xy, np.float32)
    if not self.clips_walls:
      return p
    shape = p.shape
    p = p.reshape(-1, 2)
    lo = self.free_xy - self.free_half                      # [m, 2]
    hi = self.free_xy + self.free_half
    out = np.empty_like(p)
    for i in range(0, p.shape[0], int(chunk)):
      b = p[i:i + int(chunk)]                               # [k, 2]
      q = np.clip(b[:, None, :], lo[None], hi[None])        # [k, m, 2]
      d = np.sum((q - b[:, None, :]) ** 2, axis=-1)         # [k, m]
      out[i:i + b.shape[0]] = q[np.arange(b.shape[0]), np.argmin(d, axis=1)]
    return out.reshape(shape)

  def clip_state(self, states):
    """Force raw states back into the feasible region; a no-op without one.

    Two constraints, in this order, and neither is a fix for a bad fit -- both
    only stop one bad step from being inherited by every step after it:

      * THE BOX, element-wise, which is all a per-dimension box can be. It
        says the ant is inside the outer walls, above the floor, with its
        joints at or inside their stops and no coordinate running away, which
        is what a diverging rollout violates first and worst.
      * THE WALLS, on xy alone. The box is the maze's bounding box, so on any
        map with an interior wall it contains that wall too; `clip_xy`
        projects xy onto the union of the traversable cells, which is the
        room the box cannot describe. Only xy is corrected -- the velocities
        are left as the head produced them, so a state pulled out of a wall
        still carries the momentum that took it there and the next step is
        the head's own answer to that, not a hand-made stop.
    """
    s = np.asarray(states, np.float32)
    if not self.constrains_states:
      return s
    # Either branch yields a fresh array, so the in-place xy write below can
    # never reach through to the caller's own buffer.
    s = (np.clip(s, self.state_lo, self.state_hi).astype(np.float32)
         if self.clips_states else s.copy())
    if self.clips_walls:
      s[..., :2] = self.clip_xy(s[..., :2])
    return s

  @property
  def input_dim(self):
    """Width of the vector the MLP actually consumes."""
    return int(self.state_dim + self.action_dim *
               (2 if self.next_action_input else 1))

  def _predict_fn(self):
    """Lazily built (and cached) jitted raw-units forward pass."""
    if self._apply is None:
      net = build_network(self.state_dim, self.hidden_layer_sizes,
                          self.activation, self.spectral_norm,
                          self.spectral_norm_coef, self.spectral_norm_iters)
      s_mean = jnp.asarray(self.state_mean)
      s_std = jnp.asarray(self.state_std)
      a_mean = jnp.asarray(self.action_mean)
      a_std = jnp.asarray(self.action_std)
      t_mean = jnp.asarray(self.target_mean)
      t_std = jnp.asarray(self.target_std)
      predict_delta = self.predict_delta

      pair = self.next_action_input

      def _raw(params, s, a, a2):
        blocks = [(s - s_mean) / s_std, (a - a_mean) / a_std]
        if pair:
          # The counterfactual a' is in the SAME space as a, so it gets the
          # SAME statistics -- anything else would make the factual diagonal
          # a' = a two different inputs.
          blocks.append((a2 - a_mean) / a_std)
        y = net.apply(params, jnp.concatenate(blocks, axis=-1)) * t_std + t_mean
        return s + y if predict_delta else y

      self._apply = jax.jit(_raw)
    return self._apply

  def predict(self, states, actions, next_actions=None, chunk=65_536):
    """Predicted ``s_{t+1}`` for raw ``(s_t, a_t[, a'_t])`` (numpy in/out).

    ``next_actions`` is the COUNTERFACTUAL action ``a'`` (the parameter name
    is legacy); ``None`` means the FACTUAL diagonal ``a' = a``, which is what
    every stored transition is and therefore the only query the fit
    supervised. A model fitted WITHOUT the counterfactual input ignores the
    argument entirely, so passing one by mistake cannot silently change an old
    model's answer.
    """
    fn = self._predict_fn()
    s = np.asarray(states, np.float32)
    a = np.asarray(actions, np.float32)
    a2 = a if next_actions is None else np.asarray(next_actions, np.float32)
    if a2.shape != a.shape:
      raise ValueError(f'next_actions {a2.shape} must match actions {a.shape}')
    if s.ndim == 1:                       # single transition -> keep it simple.
      return np.asarray(fn(self.params, jnp.asarray(s[None]),
                           jnp.asarray(a[None]), jnp.asarray(a2[None]))[0])
    out = [np.asarray(fn(self.params, jnp.asarray(s[i:i + chunk]),
                         jnp.asarray(a[i:i + chunk]),
                         jnp.asarray(a2[i:i + chunk])))
           for i in range(0, s.shape[0], chunk)]
    return np.concatenate(out) if out else np.zeros_like(s)


@dataclasses.dataclass
class ExpertPolicy:
  """Fitted blind expert ``pi_phi(s_t) -> a_t``: parameters + normalization.

  The network emits pre-squash ``logits``; ``act()`` turns them into an action
  in RAW units, so callers never have to know about the standardization or the
  squash. The input is the learner state ONLY -- the rockfall latent the
  sighted teacher read at the band mouth is not in it, by construction.
  """
  params: Any
  state_dim: int
  action_dim: int
  hidden_layer_sizes: Tuple[int, ...]
  state_mean: np.ndarray
  state_std: np.ndarray
  action_mean: np.ndarray
  action_std: np.ndarray
  # Action box measured on the train split; the squashed head maps
  # tanh(logits) -> action_bias + action_scale * tanh(logits), and the linear
  # head clips to [bias - scale, bias + scale].
  action_scale: np.ndarray
  action_bias: np.ndarray
  squash: bool = True
  activation: str = 'relu'
  spectral_norm: bool = False
  spectral_norm_coef: float = 1.0
  spectral_norm_iters: int = 10
  _apply: Optional[Any] = dataclasses.field(
      default=None, repr=False, compare=False)

  def _act_fn(self):
    """Lazily built (and cached) jitted raw-units forward pass."""
    if self._apply is None:
      net = build_policy_network(self.action_dim, self.hidden_layer_sizes,
                                 self.activation, self.spectral_norm,
                                 self.spectral_norm_coef,
                                 self.spectral_norm_iters)
      s_mean = jnp.asarray(self.state_mean)
      s_std = jnp.asarray(self.state_std)
      a_mean = jnp.asarray(self.action_mean)
      a_std = jnp.asarray(self.action_std)
      scale = jnp.asarray(self.action_scale)
      bias = jnp.asarray(self.action_bias)
      squash = self.squash

      def _raw(params, s):
        logits = net.apply(params, (s - s_mean) / s_std)
        if squash:
          return bias + scale * jnp.tanh(logits)
        return jnp.clip(logits * a_std + a_mean, bias - scale, bias + scale)

      self._apply = jax.jit(_raw)
    return self._apply

  def act(self, states, chunk=65_536):
    """Predicted expert action for a batch of raw states (numpy in/out)."""
    fn = self._act_fn()
    s = np.asarray(states, np.float32)
    if s.ndim == 1:
      return np.asarray(fn(self.params, jnp.asarray(s[None]))[0])
    out = [np.asarray(fn(self.params, jnp.asarray(s[i:i + chunk])))
           for i in range(0, s.shape[0], chunk)]
    return (np.concatenate(out) if out
            else np.zeros((0, self.action_dim), np.float32))


@dataclasses.dataclass
class MixtureExpertPolicy:
  """Fitted blind expert ``pi_phi(a|s)`` as a MIXTURE: params + normalization.

  The stochastic counterpart of ``ExpertPolicy``, and the head the sampler
  actually needs. The dataset was collected by a SIGHTED teacher that reads
  the rockfall latent at the band mouth; that latent never enters the stored
  observation, so the conditional this head is fitting is the teacher's action
  distribution MARGINALIZED over the hidden confounder ``u``:

      p(a|s) = sum_u p(u|s) pi_sighted(a | s, u)

  which is multi-modal by construction. A deterministic MSE head returns the
  conditional MEAN of that mixture, and at the band mouth the mean of "go
  straight" and "detour" is a torque that does neither -- which is why
  ``ExpertPolicy`` cannot be used to sample even though its RMSE looks fine.

  The generative model, exactly:

      i   ~ Categorical(w(s))                      k branches
      z   ~ N(mu_i(s), diag(sigma_i(s) ** 2))      in STANDARDIZED action units
      a   = clip(z, lo, hi)                        the actuator's own clip

  so the box is modelled by CENSORING, not by a squash -- see
  ``censored_component_log_prob`` for why that is the only one of the three
  candidate parameterizations that is even defined on this dataset, some of
  whose torques sit exactly on the limit.

  Inputs and outputs are RAW units. The mixture itself lives in standardized
  action coordinates (better conditioned for the network, and standardization
  is monotone per dimension so it commutes with the clip), and every method
  that returns an action de-standardizes on the way out.

  ``n_components = 1`` is a plain censored Gaussian and is the honest control
  for the whole design: if it matches ``k = 3`` on held-out likelihood, the
  multi-modality this head exists for is not in the actions.
  """
  params: Any
  state_dim: int
  action_dim: int
  n_components: int
  hidden_layer_sizes: Tuple[int, ...]
  state_mean: np.ndarray
  state_std: np.ndarray
  action_mean: np.ndarray
  action_std: np.ndarray
  # The censoring box, RAW action units, measured on the TRAIN split. `hi` is
  # held strictly above `lo` by the fit: a dimension with zero width would put
  # every observation at BOTH limits at once and the per-dimension likelihood
  # would stop being defined.
  action_lo: np.ndarray = None
  action_hi: np.ndarray = None
  log_sigma_min: float = -5.0
  log_sigma_max: float = 2.0
  activation: str = 'relu'
  spectral_norm: bool = False
  spectral_norm_coef: float = 1.0
  spectral_norm_iters: int = 10
  # -- SAMPLE-time entropy knobs. They change what the sampler draws and
  # nothing about the fitted parameters, so they are mutable after a load:
  #     m = load(path); m.policy.sample_sigma_scale = 2.0
  # and a sweep over them costs no refit. Defaulted to the identity, which is
  # also what every pickle written before they existed loads as.
  #
  # `sample_sigma_scale` multiplies every component's sigma in `sample` and
  # `sample_other_component`. `weight_temperature` divides the weight LOGITS
  # in those two and in `diagonal_prob`, which must see the same weights the
  # draw does or the coin would describe a mixture the sampler is not using.
  #
  # Deliberately NOT applied by `log_prob`, `responsibilities`, `weights`,
  # `component_means` or `mode_action`: those report the FITTED head, and a
  # held-out NLL that moved when a sampling knob moved would be meaningless.
  sample_sigma_scale: float = 1.0
  weight_temperature: float = 1.0
  _apply: Optional[Any] = dataclasses.field(
      default=None, repr=False, compare=False)

  def _sampling_log_w(self, log_w):
    """Weight log-probabilities AS THE SAMPLER SEES THEM. [..., k].

    ``log_w / T`` renormalized. ``T = 1`` returns the argument unchanged (by
    identity, not approximately), ``T > 1`` flattens the branches toward
    uniform and ``T < 1`` sharpens them toward the arg-max branch.
    """
    t = float(self.weight_temperature)
    if t <= 0.0:
      raise ValueError(f'weight_temperature must be positive, got {t}')
    if t == 1.0:
      return log_w
    scaled = np.asarray(log_w, np.float32) / t
    scaled -= scaled.max(axis=-1, keepdims=True)
    return (scaled - np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
            ).astype(np.float32)

  def _sampling_sigma(self, log_sigma):
    """Component standard deviations AS THE SAMPLER SEES THEM, [..., k, A].

    Standardized action units, like ``log_sigma`` itself. Scaling sigma rather
    than shifting log-sigma is the parameterization that keeps the knob
    readable: 2.0 means "draws twice as wide as the fit", whatever the fit.
    """
    c = float(self.sample_sigma_scale)
    if c <= 0.0:
      raise ValueError(f'sample_sigma_scale must be positive, got {c}')
    return np.exp(log_sigma) * c

  # -- the standardized-space box the mixture is actually censored to.
  @property
  def _lo_std(self):
    return (np.asarray(self.action_lo, np.float32)
            - self.action_mean) / self.action_std

  @property
  def _hi_std(self):
    return (np.asarray(self.action_hi, np.float32)
            - self.action_mean) / self.action_std

  def _mixture_fn(self):
    """Lazily built (and cached) jitted ``s_raw -> (log_w, mu, log_sigma)``.

    ``mu`` and ``log_sigma`` come back in STANDARDIZED action units, which is
    the frame the censored likelihood and the sampler both work in.
    """
    if self._apply is None:
      net = build_mixture_policy_network(
          self.action_dim, self.n_components, self.hidden_layer_sizes,
          self.activation, self.spectral_norm, self.spectral_norm_coef,
          self.spectral_norm_iters)
      s_mean = jnp.asarray(self.state_mean)
      s_std = jnp.asarray(self.state_std)
      a_dim, k = int(self.action_dim), int(self.n_components)
      ls_min, ls_max = float(self.log_sigma_min), float(self.log_sigma_max)

      def _raw(params, s):
        out = net.apply(params, (s - s_mean) / s_std)
        return split_mixture(out, a_dim, k, ls_min, ls_max)

      self._apply = jax.jit(_raw)
    return self._apply

  def mixture(self, states, chunk=65_536):
    """``(log_w, mu, log_sigma)`` for raw states: ``[n,k]``, ``[n,k,A]`` x2.

    ``mu`` / ``log_sigma`` are STANDARDIZED action units. Use ``component_means``
    for the same means in raw units.
    """
    fn = self._mixture_fn()
    s = np.asarray(states, np.float32)
    single = s.ndim == 1
    if single:
      s = s[None]
    parts = [fn(self.params, jnp.asarray(s[i:i + chunk]))
             for i in range(0, s.shape[0], chunk)]
    if not parts:
      k, a = int(self.n_components), int(self.action_dim)
      return (np.zeros((0, k), np.float32), np.zeros((0, k, a), np.float32),
              np.zeros((0, k, a), np.float32))
    out = tuple(np.concatenate([np.asarray(p[j]) for p in parts])
                for j in range(3))
    return tuple(o[0] for o in out) if single else out

  def component_means(self, states, chunk=65_536):
    """The k component means in RAW action units, clipped to the box. [n,k,A].

    Clipped because the generative model clips: a component whose mean sits
    outside the actuator's range describes a branch that saturates, and the
    action it actually produces is the limit.
    """
    _, mu, _ = self.mixture(states, chunk=chunk)
    raw = mu * self.action_std + self.action_mean
    return np.clip(raw, self.action_lo, self.action_hi).astype(np.float32)

  def weights(self, states, chunk=65_536):
    """Mixture weights ``w(s)`` in (0, 1), summing to one per row. [n, k]."""
    log_w, _, _ = self.mixture(states, chunk=chunk)
    return np.exp(log_w).astype(np.float32)

  def diagonal_prob(self, states, chunk=65_536):
    """``sum_i w_i(s) ** 2`` -- the FACTUAL-branch probability. [n].

    The probability that two independent draws of the hidden confounder land
    in the SAME branch, and the coin ``sample`` / ``rollout`` use to choose
    between the dynamics head's factual diagonal and its counterfactual
    off-diagonal. It is 1 exactly when one branch has all the weight (there is
    no counterfactual to ask about at such a state) and ``1/k`` when the k
    branches are equally likely (maximally confounded), so the counterfactual
    query fires in proportion to how much the confounder matters HERE -- with
    no temperature to tune, unlike the discriminator coin it replaced.

    This is the state-level form. Note it is also the average of the
    per-component form ``w_i`` over a draw of ``i``, since
    ``sum_i P(draw i) * w_i = sum_i w_i ** 2``; taking it at the state level
    keeps the branch choice INDEPENDENT of which action was drawn, so a
    rare-branch draw is not systematically the one sent to the unidentified
    off-diagonal map.

    Computed on the TEMPERED weights, because the coin has to describe the
    mixture the sampler actually draws from: with ``weight_temperature > 1``
    the branches the draw sees are flatter than the fitted ones, and a coin
    taken on the fitted weights would over-report the factual branch.
    """
    log_w, _, _ = self.mixture(states, chunk=chunk)
    w = np.exp(self._sampling_log_w(log_w)).astype(np.float32)
    return np.sum(w ** 2, axis=-1).astype(np.float32)

  def log_prob(self, states, actions, chunk=65_536):
    """``log p(a|s)`` of the CENSORED mixture, for raw ``(s, a)``. [n].

    Mixes log-densities on interior dimensions with log probabilities on
    censored ones, and is stated in STANDARDIZED action units -- see
    ``censored_component_log_prob`` for both caveats. Comparable across fits
    on the same data, which is what selecting ``k`` needs; not a pure density.
    """
    s = np.asarray(states, np.float32)
    a = np.asarray(actions, np.float32)
    single = s.ndim == 1
    if single:
      s, a = s[None], a[None]
    if a.shape != (s.shape[0], self.action_dim):
      raise ValueError(f'actions must be [{s.shape[0]}, {self.action_dim}], '
                       f'got {a.shape}')
    log_w, mu, ls = self.mixture(s, chunk=chunk)
    a_std = (a - self.action_mean) / self.action_std
    out = np.asarray(censored_mixture_log_prob(
        jnp.asarray(log_w), jnp.asarray(mu), jnp.asarray(ls),
        jnp.asarray(a_std), jnp.asarray(self._lo_std),
        jnp.asarray(self._hi_std)))
    return float(out[0]) if single else out

  def responsibilities(self, states, actions, chunk=65_536):
    """``p(i | s, a)`` -- which branch this action most likely came from. [n,k].

    The mixture posterior, i.e. the weights reweighted by how well each
    component explains the observed action. The diagnostic that says whether
    the k components have actually split the data: a component that never
    takes responsibility anywhere is dead weight, and a dataset whose rows all
    load on one component has no multi-modality for the head to find.
    """
    s = np.asarray(states, np.float32)
    a = np.asarray(actions, np.float32)
    single = s.ndim == 1
    if single:
      s, a = s[None], a[None]
    log_w, mu, ls = self.mixture(s, chunk=chunk)
    a_std = (a - self.action_mean) / self.action_std
    joint = log_w + np.asarray(censored_component_log_prob(
        jnp.asarray(mu), jnp.asarray(ls), jnp.asarray(a_std),
        jnp.asarray(self._lo_std), jnp.asarray(self._hi_std)))
    joint = joint - joint.max(axis=-1, keepdims=True)
    w = np.exp(joint)
    out = (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
    return out[0] if single else out

  def sample(self, states, seed=0, chunk=65_536):
    """Draw ``a ~ p(a|s)``. Returns ``(actions_raw, component_index)``.

    The generative model of the class docstring, run forward: a categorical
    draw over the branches, a diagonal Gaussian draw inside the chosen one,
    and the actuator's clip. The component index is returned because the
    sampler KNOWS which branch it drew -- there is no need to classify the
    action afterwards, and an argmax over responsibilities would be a strictly
    worse estimate of a quantity that is already exact here.
    """
    s = np.asarray(states, np.float32)
    single = s.ndim == 1
    if single:
      s = s[None]
    log_w, mu, ls = self.mixture(s, chunk=chunk)
    log_w = self._sampling_log_w(log_w)
    sigma = self._sampling_sigma(ls)
    rng = np.random.default_rng(seed)
    n, k = log_w.shape
    # Gumbel-max: one uniform per (row, component) gives the categorical draw
    # without a Python loop over rows.
    g = -np.log(-np.log(rng.random((n, k)) + 1e-300) + 1e-300)
    i = np.argmax(log_w + g, axis=-1)
    rows = np.arange(n)
    z = (mu[rows, i] + sigma[rows, i] * rng.standard_normal(
        (n, self.action_dim))).astype(np.float32)
    a = np.clip(z * self.action_std + self.action_mean,
                self.action_lo, self.action_hi).astype(np.float32)
    return (a[0], int(i[0])) if single else (a, i.astype(np.int64))

  def sample_other_component(self, states, exclude, seed=0, chunk=65_536):
    """Draw ``a'`` from a branch OTHER than ``exclude``. [n, A].

    What ``dyn_off_diagonal_action='mixture'`` uses for the counterfactual
    action: an action the expert plausibly would have played at this state had
    the hidden confounder taken a different value. Unlike a uniform draw it
    stays on the action manifold, so an off-diagonal query is extrapolating in
    ONE way (an unsupervised map) rather than two (an unsupervised map at an
    input the data never visits).

    With ``n_components == 1`` there is no other branch and this raises: a
    single-component head has no counterfactual of this kind to offer.
    """
    k = int(self.n_components)
    if k < 2:
      raise ValueError('sample_other_component needs at least 2 mixture '
                       f'components, this head has {k}')
    s = np.asarray(states, np.float32)
    single = s.ndim == 1
    if single:
      s = s[None]
    ex = np.asarray(exclude, np.int64).reshape(-1)
    if ex.shape[0] != s.shape[0]:
      raise ValueError(f'exclude must have {s.shape[0]} entries, got '
                       f'{ex.shape[0]}')
    log_w, mu, ls = self.mixture(s, chunk=chunk)
    log_w = self._sampling_log_w(log_w)
    sigma = self._sampling_sigma(ls)
    rng = np.random.default_rng(seed)
    n = log_w.shape[0]
    # Renormalize over the remaining branches by masking the excluded one to
    # -inf before the Gumbel-max, so a' is drawn from p(i | i != exclude)
    # rather than from a uniform choice among the others.
    masked = np.array(log_w, np.float32)
    masked[np.arange(n), ex] = -np.inf
    g = -np.log(-np.log(rng.random((n, k)) + 1e-300) + 1e-300)
    j = np.argmax(masked + g, axis=-1)
    rows = np.arange(n)
    z = (mu[rows, j] + sigma[rows, j] * rng.standard_normal(
        (n, self.action_dim))).astype(np.float32)
    a = np.clip(z * self.action_std + self.action_mean,
                self.action_lo, self.action_hi).astype(np.float32)
    return a[0] if single else a

  def mode_action(self, states, chunk=65_536):
    """The mean of the HIGHEST-WEIGHT component, clipped. [n, A].

    The deterministic point action of a mixture head, and what ``act``
    returns. Deliberately NOT the mixture mean: the mixture mean is exactly
    the artifact this head exists to avoid, an average over branches that is
    itself in no branch.
    """
    log_w, _, _ = self.mixture(states, chunk=chunk)
    means = self.component_means(states, chunk=chunk)
    single = np.ndim(log_w) == 1
    if single:
      return means[int(np.argmax(log_w))]
    return means[np.arange(log_w.shape[0]), np.argmax(log_w, axis=-1)]

  def mean_action(self, states, chunk=65_536):
    """``sum_i w_i(s) mu_i(s)``, clipped. [n, A]. A DIAGNOSTIC only.

    Reported so a mixture fit can be scored against the same mean-action
    baseline and the same RMSE the deterministic head was, which is the only
    way to compare the two on one number. It is not a usable action on a
    multi-modal state, and a mixture head whose mean-RMSE beats its own mode's
    is telling you the branches have not separated.
    """
    log_w, mu, _ = self.mixture(states, chunk=chunk)
    w = np.exp(log_w)
    if np.ndim(log_w) == 1:
      z = np.sum(w[:, None] * mu, axis=0)
    else:
      z = np.sum(w[..., None] * mu, axis=-2)
    return np.clip(z * self.action_std + self.action_mean,
                   self.action_lo, self.action_hi).astype(np.float32)

  def act(self, states, seed=None, chunk=65_536):
    """A point action per state, for callers that want one.

    ``seed=None`` returns ``mode_action`` -- deterministic, so this method is
    a drop-in for ``ExpertPolicy.act``. An integer seed DRAWS instead, which
    is what ``sample`` / ``rollout`` do; they pass one explicitly rather than
    relying on this default, because a silently deterministic sampler is the
    failure this head was built to remove.
    """
    if seed is None:
      return self.mode_action(states, chunk=chunk)
    a, _ = self.sample(states, seed=seed, chunk=chunk)
    return a


def build_done_network(hidden_layer_sizes: Sequence[int],
                       activation: str = 'relu'):
  """The done head: ``s_norm -> 1`` logit of ``P(done | s)``.

  No spectral norm: the head never feeds back into the chain (it only decides
  where a rollout stops), so there is no compounding to bound, and a
  Lipschitz cap would blur the terminal boundary it exists to draw.
  """
  sizes = list(hidden_layer_sizes) + [1]
  if activation not in ACTIVATIONS:
    raise ValueError(f'unknown activation {activation!r}; '
                     f'expected one of {sorted(ACTIVATIONS)}')
  act_fn = ACTIVATIONS[activation]

  def _fn(x):
    return hk.nets.MLP(sizes, w_init=_W_INIT, activation=act_fn,
                       name='done_mlp')(x)[..., 0]

  return hk.without_apply_rng(hk.transform(_fn))


@dataclasses.dataclass
class DoneHead:
  """Fitted terminal classifier ``P(done | s_t)`` on the raw learner state.

  Supervised by the TERMINAL states of the offline data -- the goal-arrival
  state of every successful positive episode and the settled death state of
  every failure episode (``config.dyn_done_dataset``) -- against the
  non-terminal states of the same episodes. The input is the 29-dim state
  only, standardized with the dynamics head's train-split statistics; the goal
  is not an input.

  ``threshold`` is sample-time only and mutable on a loaded pickle, like the
  mixture head's sampling knobs: ``rollout`` stops a row at the first
  generated state with ``prob >= threshold``.
  """
  params: Any
  state_dim: int
  hidden_layer_sizes: Tuple[int, ...]
  state_mean: np.ndarray
  state_std: np.ndarray
  activation: str = 'relu'
  threshold: float = 0.5
  _apply: Optional[Any] = dataclasses.field(
      default=None, repr=False, compare=False)

  def _logit_fn(self):
    if self._apply is None:
      net = build_done_network(self.hidden_layer_sizes, self.activation)
      s_mean = jnp.asarray(self.state_mean)
      s_std = jnp.asarray(self.state_std)
      self._apply = jax.jit(
          lambda params, s: net.apply(params, (s - s_mean) / s_std))
    return self._apply

  def prob(self, states, chunk=65_536):
    """``P(done | s)`` for a batch of raw states (numpy in/out)."""
    fn = self._logit_fn()
    s = np.asarray(states, np.float32)
    out = [np.asarray(jax.nn.sigmoid(fn(self.params,
                                        jnp.asarray(s[i:i + chunk]))))
           for i in range(0, s.shape[0], chunk)]
    return np.concatenate(out) if out else np.zeros((0,), np.float32)

  def is_done(self, states, chunk=65_536):
    """Boolean done flag per raw state, at ``self.threshold``."""
    return self.prob(states, chunk=chunk) >= float(self.threshold)


@dataclasses.dataclass
class SampledTransitions:
  """Model-generated transitions next to the dataset rows they were drawn from.

  What ``CausalTransitionModel.sample`` returns. Exactly two of these fields
  are GENERATED -- ``actions`` from the blind expert head and ``next_states``
  from the dynamics head -- and everything else is either copied out of the
  offline dataset or a diagnostic computed from the pair:

    * ``states`` are dataset rows, verbatim: neither head models the state
      distribution, so a starting state can only be drawn, not invented;
    * ``goals`` are INHERITED from the same dataset rows. No head predicts a
      goal (dynamics are goal-independent and the expert head is fitted on the
      state half alone), so the goal is carried along purely to keep the
      generated transition consumable by a goal-conditioned learner;
    * ``data_actions`` / ``data_next_states`` are the stored ground truth for
      the same rows, which is what makes the batch self-comparing;
    * ``next_states_data_action`` is ``f(s, a_data)`` -- the dynamics head
      queried with the STORED action instead of the generated one. It splits
      the next-state error into the part the dynamics head owns and the part
      it inherited from the policy head's action (see ``compare``).

  With a MIXTURE expert head the batch is a draw from a stochastic model, not
  just a draw of rows: the same ``index`` reproduces the same actions only at
  the same ``seed``. ``components`` records which branch of the hidden
  confounder each action was drawn from, which is exact rather than inferred
  -- the sampler chose it.
  """
  index: np.ndarray                                 # [n] dataset row numbers.
  states: np.ndarray                                # [n, S] drawn, verbatim.
  actions: np.ndarray                               # [n, A] GENERATED: a~pi(s).
  next_states: np.ndarray                           # [n, S] GENERATED: f(s,a).
  goals: Optional[np.ndarray] = None                # [n, G] inherited.
  data_actions: Optional[np.ndarray] = None         # [n, A] stored a_t.
  data_next_states: Optional[np.ndarray] = None     # [n, S] stored s_{t+1}.
  next_states_data_action: Optional[np.ndarray] = None   # [n, S] f(s, a_data).
  # The mixture component each generated action was drawn from; None when the
  # expert head is the deterministic one, which has no branches.
  components: Optional[np.ndarray] = None           # [n] int64 in [0, k).
  # Set only when sample() ran with off_diagonal='mixture'. `off_diagonal`
  # marks the rows whose next_state came from a COUNTERFACTUAL query
  # f(s, a, a') with a' != a, and `counterfactual_actions` is the a' that was
  # used on those rows (NaN elsewhere). Those rows are extrapolation: the fit
  # only ever saw a' = a. See sample().
  off_diagonal: Optional[np.ndarray] = None         # [n] bool.
  counterfactual_actions: Optional[np.ndarray] = None    # [n, A] a'.
  # Which generated next states the feasibility clip MOVED -- outside the
  # feasible box, or inside a maze wall. None when the model carries neither
  # constraint. It is recorded rather than swallowed because clipping changes
  # the one-step error `compare()` reports: on a clipped row the number is the
  # error of the CORRECTED state, not of the head's raw output, and a reader
  # who does not know that would read a constraint as a better fit.
  clipped: Optional[np.ndarray] = None              # [n] bool.
  # Whether the model that produced this batch had its off-diagonal
  # supervised by a negative dataset. It rides along because it is what
  # decides whether an off-diagonal row is data or an artifact of the init,
  # and the consumer of a batch has no other way to find out.
  off_diagonal_trained: bool = False

  @property
  def n(self):
    return int(self.states.shape[0])

  def _check_consumable(self, allow_off_diagonal):
    """Refuse to hand out a batch a consumer would misread.

    Two refusals, both the same kind of bug caught at the boundary rather than
    absorbed silently:

      * no goals -- a zero-padded goal half is exactly what cost this repo 0.3
        success on every AntMaze benchmark (commit 22ae675);
      * off-diagonal rows from an UNSUPERVISED off-diagonal -- those next
        states are "where s would have gone had a' been played instead", NOT
        "where s went after a", and on a fit with no negative dataset the map
        that produced them is unidentified: no gradient has ever supervised
        a' != a, so a consumer that eats them as dynamics is training on the
        network's initialization. Pass ``allow_off_diagonal=True`` to say you
        meant it. A model whose off-diagonal WAS supervised
        (``off_diagonal_trained``) is not refused: there the second branch is
        the point.
    """
    if self.goals is None:
      raise ValueError('this batch was sampled without goals; pass goals= to '
                       'sample() if you need full observations')
    if (not allow_off_diagonal and not self.off_diagonal_trained
        and self.off_diagonal is not None
        and bool(self.off_diagonal.any())):
      n_off = int(self.off_diagonal.sum())
      raise ValueError(
          f'{n_off} of {self.n} rows are COUNTERFACTUAL (off-diagonal): their '
          "next_state is f(s, a, a') for a' != a, which is not the transition "
          'that follows `a` and comes from a map the fit never supervised. '
          'Filter on .off_diagonal, or pass allow_off_diagonal=True if a '
          'mixed batch is genuinely what you want.')

  def observations(self, allow_off_diagonal=False):
    """``concat(state, goal)`` -- the observation an agent network consumes."""
    self._check_consumable(allow_off_diagonal)
    return np.concatenate([self.states, self.goals], axis=-1)

  def next_observations(self, allow_off_diagonal=False):
    """``concat(next_state, goal)``: the goal is constant within a transition."""
    self._check_consumable(allow_off_diagonal)
    return np.concatenate([self.next_states, self.goals], axis=-1)

  def compare(self):
    """How far the generated action and next state are from the stored ones.

    Returns None when the batch was drawn without ground truth. Each head is
    reported against the same trivial baseline its fit is reported against --
    the train-split mean action for the policy, persistence ``s' = s`` for the
    dynamics -- because a raw RMSE on this state space is not interpretable on
    its own; ``*_gain`` is baseline / model, so above 1 is better than trivial.

    The three next-state numbers answer three different questions:

      ``next_state_rmse_raw``        the end-to-end error of the generated
                                     transition, which is what a consumer of
                                     this batch actually eats;
      ``dyn_only_rmse_raw``          ``f(s, a_data)`` against the truth -- the
                                     dynamics head's own error, with the
                                     policy head taken out of the loop;
      ``policy_induced_rmse``        how far swapping the generated action in
                                     for the stored one moves the dynamics
                                     head's own prediction.

    A large end-to-end error with a small ``dyn_only_rmse_raw`` means the
    disagreement is the POLICY's, which on this dataset is the interesting
    outcome rather than a defect: the stored actions come from a sighted
    teacher and the expert head is blind by construction, so the steps where
    it cannot reproduce the action are the steps whose decision needed the
    hidden rockfall latent.
    """
    if self.data_actions is None or self.data_next_states is None:
      return None
    a_gen, a_true = self.actions, self.data_actions
    sn_gen, sn_true = self.next_states, self.data_next_states

    out = {'n': self.n}
    out.update(_action_error_stats(a_gen, a_true, 'action'))
    out.update(_raw_error_stats(sn_gen, sn_true, 'next_state'))

    # Baselines, computed on THESE rows: a mean action taken over the drawn
    # rows is the honest constant predictor for the same slice.
    mean_a = np.broadcast_to(np.mean(a_true, axis=0), a_true.shape)
    out['action_mean_baseline_rmse_raw'] = float(
        np.sqrt(np.mean((mean_a - a_true) ** 2)))
    out['action_gain'] = (out['action_mean_baseline_rmse_raw'] /
                          max(out['action_rmse_raw'], 1e-12))
    # The persistence baseline is also the SCALE reference, since the error of
    # ``s' = s`` is exactly how far one true step moves the state. So the same
    # number serves twice: as the baseline the dynamics head has to beat, and
    # as the unit that makes the raw RMSE readable -- an error well under one
    # step is a generated transition that lands where the real one did.
    step = float(np.sqrt(np.mean((sn_true - self.states) ** 2)))
    out['next_state_persistence_rmse_raw'] = step
    out['next_state_gain'] = step / max(out['next_state_rmse_raw'], 1e-12)
    out['next_state_error_per_step'] = out['next_state_rmse_raw'] / max(step,
                                                                        1e-12)

    # Mean per-row cosine between the two action vectors: unlike RMSE this
    # says whether the generated action points the same WAY as the stored one.
    num = np.sum(a_gen * a_true, axis=-1)
    den = (np.linalg.norm(a_gen, axis=-1) * np.linalg.norm(a_true, axis=-1))
    ok = den > 1e-12
    out['action_cosine'] = (float(np.mean(num[ok] / den[ok])) if ok.any()
                            else float('nan'))

    if self.next_states_data_action is not None:
      out['dyn_only_rmse_raw'] = float(
          np.sqrt(np.mean((self.next_states_data_action - sn_true) ** 2)))
      out['policy_induced_rmse'] = float(np.sqrt(np.mean(
          (sn_gen - self.next_states_data_action) ** 2)))
    if self.clipped is not None:
      out['clipped_frac'] = float(np.mean(self.clipped))
    if self.off_diagonal is not None:
      out['off_diagonal_frac'] = float(np.mean(self.off_diagonal))
    if self.components is not None and self.components.size:
      # How the draws spread over the branches. A component that never gets
      # drawn is dead weight in the head; all the mass on one is a state
      # distribution with no confounding for the mixture to represent.
      k = int(self.components.max()) + 1
      out['component_frac'] = [
          float(np.mean(self.components == j)) for j in range(k)]
    return out

  def summary(self):
    """Human-readable form of ``compare`` (empty string with no ground truth)."""
    c = self.compare()
    if c is None:
      return f'  {self.n} generated transitions (no ground truth to compare)'
    lines = [
        f'  {c["n"]} sampled transitions'
        f'{"" if self.goals is None else " (goals inherited from the dataset)"}',
        f'  action   rmse={c["action_rmse_raw"]:.5f} '
        f'(mean-action baseline {c["action_mean_baseline_rmse_raw"]:.5f}, '
        f'{c["action_gain"]:.2f}x) '
        f'expl_var={c["action_explained_variance"]:.4f} '
        f'cos={c["action_cosine"]:+.3f}',
        f'  next s   rmse={c["next_state_rmse_raw"]:.5f} '
        f'xy={c["next_state_rmse_xy"]:.5f} '
        f'(persistence {c["next_state_persistence_rmse_raw"]:.5f}, '
        f'{c["next_state_gain"]:.2f}x) '
        f'expl_var={c["next_state_explained_variance"]:.4f}',
        f'  next s   error is {c["next_state_error_per_step"]:.2f}x one true '
        f'step (one step moves the state by '
        f'{c["next_state_persistence_rmse_raw"]:.5f})',
    ]
    if 'dyn_only_rmse_raw' in c:
      lines.append(
          f'  attribution: dynamics alone f(s, a_data) rmse='
          f'{c["dyn_only_rmse_raw"]:.5f}; swapping in the generated action '
          f'moves the prediction {c["policy_induced_rmse"]:.5f}')
    if 'clipped_frac' in c:
      lines.append(
          f'  feasibility clip moved {100 * c["clipped_frac"]:.2f}% of the '
          f'generated next states, whose next_state error above is therefore '
          f'the CORRECTED state\'s, not the raw head\'s '
          f'(dyn_only above is never clipped)')
    if 'component_frac' in c:
      lines.append('  branches drawn: '
                   + ' '.join(f'{f:.3f}' for f in c['component_frac']))
    if 'off_diagonal_frac' in c:
      lines.append(
          f'  off-diagonal (COUNTERFACTUAL, '
          f'{"supervised" if self.off_diagonal_trained else "UNIDENTIFIED"}): '
          f'{100 * c["off_diagonal_frac"]:.1f}% of rows')
    return '\n'.join(lines)


@dataclasses.dataclass
class RolloutTrajectories:
  """Generated trajectories: T model steps from a real INITIAL state.

  What ``CausalTransitionModel.rollout`` returns. Each row starts at the first
  state of a dataset episode -- the state the environment resets to -- and
  everything after it was generated by iterating the two heads,
  ``a_t ~ pi_phi(. | s_t)`` into ``s_{t+1} = f_theta(s_t, a_t, a'_t)``. No
  real prefix is copied in and no mid-episode state is ever used as a start,
  so index 0 is the only real observation in a row and all ``T`` transitions
  are the model's.

  The START is always real. ``episode`` names the dataset episode whose reset
  state and goal the row carries, so a generated trajectory can never be the
  base of another one.

  Why a trajectory and not a bag of transition pairs: the contrastive learner
  relabels the goal to a FUTURE state sampled from the SAME trajectory
  (``crl/replay.py``), so generated data is only usable by it if it has an
  internal future to relabel from. That is the whole reason this exists next
  to ``sample`` -- ``sample`` is an evaluation of the heads, this is data.

  ``obs``, ``act`` and ``lengths`` are exactly the three arguments of
  ``crl.replay.TrajectoryBuffer.add_episode``, in its layout: observations are
  the full ``concat(state, goal)`` the agent network consumes, rows past
  ``lengths[b]`` are zero padding the relabeler must not sample, and
  ``pad_to`` sizes the time axis to a target buffer's ``ep_len_obs``.

  The goal half is INHERITED from the source trajectory and held constant
  across the generated segment. In these datasets the stored goal is constant
  over an episode's valid span anyway (the varying values are the zero-padded
  tail), and the relabeler reads only the state half, so this is exact rather
  than an approximation -- but it does mean a generated trajectory carries its
  ORIGINAL episode's goal, not one the rollout achieved.
  """
  episode: np.ndarray            # [n] source episode id per row.
  obs: np.ndarray                # [n, L, S + G] concat(state, goal), padded.
  act: np.ndarray                # [n, L, A] padded.
  lengths: np.ndarray            # [n] valid observation count, all T+1.
  generated: np.ndarray          # [n, L] bool: obs rows the model produced.
  steps: int                     # T.
  state_dim: int
  data_states: Optional[np.ndarray] = None   # [n, L, S] the real continuation.
  data_act: Optional[np.ndarray] = None      # [n, L, A] the real actions.
  data_valid: Optional[np.ndarray] = None    # [n, L] bool: real row exists.
  # Per-step bookkeeping of the mixture head, both indexed like `generated`:
  # `components` at the slot of the ACTION it describes (-1 on padding and on
  # any row a deterministic head produced), `off_diagonal` at the DESTINATION
  # of the transition it describes. Both None when the feature they record was
  # not in play.
  components: Optional[np.ndarray] = None    # [n, L] int64 branch drawn.
  off_diagonal: Optional[np.ndarray] = None  # [n, L] bool: counterfactual step.
  # Where the feasibility clip MOVED the state, also indexed at the
  # DESTINATION: True on a generated observation that `clip_state` changed --
  # a coordinate the dynamics head put outside the feasible box, or an xy it
  # put inside a maze wall and the floor-plan projection pulled out. The two
  # are not distinguished here; `model.clips_states` and `model.clips_walls`
  # say which constraints were even active. None when the model carries
  # neither. Kept because a clip that fires is a report about the fit, not a
  # fix: a rollout whose states are pinned to the walls is telling you the
  # chain diverged, and `clipped_frac` is the cheapest place to see it.
  clipped: Optional[np.ndarray] = None       # [n, L] bool: the clip moved it.
  # Whether the model that generated these rows had its off-diagonal
  # supervised by a negative dataset -- the difference between a
  # counterfactual step that carries the failure branch and one that carries
  # the network's initialization. See TransitionModel.off_diagonal_trained.
  off_diagonal_trained: bool = False
  # [n] bool: the row was STOPPED by the done head (its last valid
  # observation, index lengths[b] - 1, is the state the head flagged). None
  # when the rollout ran without a done head; then every row is T + 1 long.
  terminated: Optional[np.ndarray] = None

  @property
  def terminated_frac(self):
    """Share of rows the done head stopped before the horizon."""
    return 0.0 if self.terminated is None else float(np.mean(self.terminated))

  @property
  def off_diagonal_frac(self):
    """Share of GENERATED observations that came from a counterfactual step.

    The compounding number: read it before believing anything downstream of a
    rollout run with ``off_diagonal='mixture'``, because a trajectory is
    unsupervised from its first off-diagonal step onward -- every later state
    was predicted from a state no physics produced.
    """
    if self.off_diagonal is None:
      return 0.0
    g = self.generated
    return (float(np.mean(self.off_diagonal[g])) if bool(g.any()) else 0.0)

  @property
  def any_off_diagonal_frac(self):
    """Share of ROWS with at least one counterfactual step anywhere in them.

    The number that matters for using a rollout as data: a row is factual only
    if EVERY one of its steps was, so this is the fraction of trajectories
    that are contaminated rather than the fraction of steps.
    """
    if self.off_diagonal is None:
      return 0.0
    return float(np.mean(np.any(self.off_diagonal & self.generated, axis=1)))

  @property
  def clipped_frac(self):
    """Share of GENERATED observations the feasibility clip touched.

    0.0 on an unconstrained model, which is indistinguishable here from a
    constraint that never bound; ``model.constrains_states`` tells those
    apart, and ``clips_states`` / ``clips_walls`` say which were active. A
    small non-zero value is the clip doing its job on the tail of a chain; a
    large one means the rollout is mostly the box rather than the model, and
    the right response is a shorter ``dyn_rollout_steps`` or a better fit, not
    a wider box.
    """
    if self.clipped is None:
      return 0.0
    g = self.generated
    return float(np.mean(self.clipped[g])) if bool(g.any()) else 0.0

  @property
  def any_clipped_frac(self):
    """Share of ROWS the clip touched at least once, anywhere in them."""
    if self.clipped is None:
      return 0.0
    return float(np.mean(np.any(self.clipped & self.generated, axis=1)))

  @property
  def n(self):
    return int(self.obs.shape[0])

  def states(self):
    """[n, L, S] the state half of every stored observation."""
    return self.obs[..., :self.state_dim]

  def goals(self):
    """[n, L, G] the inherited goal half."""
    return self.obs[..., self.state_dim:]

  def episodes(self):
    """Iterate ``(obs, act, length)`` triples -- ``add_episode``'s arguments.

    Kept as a generator over rows rather than a method that takes a buffer,
    so this module stays free of any dependency on ``crl.replay``::

        for o, a, n in traj.episodes():
          buffer.add_episode(o, a, length=n)
    """
    for b in range(self.n):
      yield self.obs[b], self.act[b], int(self.lengths[b])

  def drift(self):
    """Closed-loop error at each horizon h = 1..T against the real continuation.

    Returns a list of per-horizon dicts, or None when the rollout was produced
    without ground truth. Only rows whose SOURCE episode actually ran h steps
    from its initial state contribute at horizon h, so ``n`` shrinks with h
    and the deep horizons are measured on the long episodes only -- read ``n``
    before reading the error.

    ``state_rmse`` is the closed-loop error: at every step the policy head was
    fed the dynamics head's own previous output, so this compounds in a way
    none of the one-step numbers in ``SampledTransitions.compare`` do. The
    scale reference is ``true_move_rmse``, how far the REAL trajectory
    travelled over the same h steps; ``error_per_move`` is their ratio, and it
    is the number that says whether a generated state at horizon h is still
    near the real one (<< 1) or has decorrelated from it (>= 1).
    """
    if self.data_states is None:
      return None
    rows = []
    for h in range(1, self.steps + 1):
      # Every row starts at index 0; a row the done head stopped has no
      # generated state past its own length.
      ok = self.data_valid[:, h] & (h < self.lengths)
      if not ok.any():
        rows.append({'horizon': h, 'n': 0})
        continue
      idx = np.flatnonzero(ok)
      gen = self.obs[idx, h, :self.state_dim]
      true = self.data_states[idx, h]
      start = self.obs[idx, 0, :self.state_dim]
      err = gen - true
      move = float(np.sqrt(np.mean((true - start) ** 2)))
      row = {
          'horizon': h,
          'n': int(idx.size),
          'state_rmse': float(np.sqrt(np.mean(err ** 2))),
          'state_rmse_xy': float(np.sqrt(np.mean(err[:, :2] ** 2))),
          'true_move_rmse': move,
      }
      row['error_per_move'] = row['state_rmse'] / max(move, 1e-12)
      # The generated action at this step is the one that PRODUCED the state
      # at horizon h, i.e. the action stored at time h - 1.
      a_gen = self.act[idx, h - 1]
      a_true = self.data_act[idx, h - 1]
      row['action_rmse'] = float(np.sqrt(np.mean((a_gen - a_true) ** 2)))
      rows.append(row)
    return rows

  def summary(self):
    """Human-readable drift at a few horizons (empty with no ground truth)."""
    d = self.drift()
    head = (f'  {self.n} rollouts of T={self.steps} steps, each from the '
            f'initial state of one of {np.unique(self.episode).size} '
            'episodes (no real prefix: every transition is generated)')
    if self.off_diagonal is not None:
      head += (f'\n  COUNTERFACTUAL: {100 * self.off_diagonal_frac:.1f}% of '
               f'generated steps took the '
               + ('supervised' if self.off_diagonal_trained
                  else 'unidentified')
               + f' off-diagonal, leaving '
                 f'{100 * self.any_off_diagonal_frac:.1f}% of ROWS with at '
                 f'least one such step'
               + ('' if self.off_diagonal_trained else
                  ' -- and every state after one was predicted from a state '
                  'no physics produced'))
    if self.terminated is not None:
      head += (f'\n  DONE HEAD: stopped {100 * self.terminated_frac:.1f}% of '
               f'rows before T; mean length {self.lengths.mean():.1f} obs')
    if self.clipped is not None:
      head += (f'\n  FEASIBILITY CLIP: {100 * self.clipped_frac:.2f}% of '
               f'generated states were moved back into the feasible region, '
               f'touching {100 * self.any_clipped_frac:.1f}% of ROWS')
    if self.off_diagonal is not None and self.components is not None:
      gen = self.generated
      c = self.components[gen & (self.components >= 0)]
      if c.size:
        k = int(c.max()) + 1
        head += ('\n  branches drawn: '
                 + ' '.join(f'{float(np.mean(c == j)):.3f}' for j in range(k)))
    if d is None:
      return head + '\n  (no ground truth: drift not measured)'
    lines = [head,
             f'  {"horizon":>8}{"n":>8}{"state rmse":>12}{"xy rmse":>10}'
             f'{"true move":>11}{"err/move":>10}{"action rmse":>13}']
    picks = [h for h in (1, 2, 5, 10, 25, 50, 75, 100, self.steps)
             if 1 <= h <= self.steps]
    for h in sorted(set(picks)):
      r = d[h - 1]
      if not r['n']:
        lines.append(f'  {h:>8}{0:>8}{"--":>12}')
        continue
      lines.append(f'  {h:>8}{r["n"]:>8}{r["state_rmse"]:>12.4f}'
                   f'{r["state_rmse_xy"]:>10.4f}{r["true_move_rmse"]:>11.4f}'
                   f'{r["error_per_move"]:>10.3f}{r["action_rmse"]:>13.4f}')
    return '\n'.join(lines)


@dataclasses.dataclass
class CausalTransitionModel:
  """The ETT stage-0 artifact: dynamics + a blind expert policy.

  A thin container; each head keeps its own parameters and normalization, so
  ``model.transition`` is exactly the object ``fit`` has always returned and
  every existing consumer of a ``TransitionModel`` keeps working on it.

  ``policy`` is either a ``MixtureExpertPolicy`` (``dyn_policy_head='mdn'``,
  the default) or the historical deterministic ``ExpertPolicy``. Only the
  mixture head can drive the counterfactual branch of ``sample`` /
  ``rollout``, because that branch's coin is a function of the mixture
  weights.
  """
  transition: TransitionModel
  policy: Any
  # Optional terminal classifier (``config.dyn_done_dataset``); when present,
  # ``rollout`` stops each row at the first generated state it flags.
  done: Optional[DoneHead] = None

  @property
  def has_done(self):
    return self.done is not None

  # -- pass-throughs, so a CausalTransitionModel can stand in for either head.
  def predict(self, states, actions, next_actions=None, chunk=65_536):
    """Predicted ``s_{t+1}`` for raw ``(s_t, a_t[, a'_t])`` (dynamics head)."""
    return self.transition.predict(states, actions, next_actions, chunk=chunk)

  def clip_state(self, states):
    """Force raw states into the dynamics head's feasible region (see it)."""
    return self.transition.clip_state(states)

  def clip_xy(self, xy, chunk=65_536):
    """Project xy onto the dynamics head's maze floor plan (see it)."""
    return self.transition.clip_xy(xy, chunk=chunk)

  @property
  def clips_states(self):
    """True when a feasible state box is attached to the dynamics head."""
    return self.transition.clips_states

  @property
  def clips_walls(self):
    """True when a maze floor plan is attached to the dynamics head."""
    return self.transition.clips_walls

  @property
  def constrains_states(self):
    """True when `clip_state` does any work -- box, walls or both."""
    return self.transition.constrains_states

  def set_state_bounds(self, bounds):
    """Attach (or drop, with None) the feasible state box. Returns self."""
    self.transition.set_state_bounds(bounds)
    return self

  def set_free_cells(self, cells):
    """Attach (or drop, with None) the maze floor plan. Returns self."""
    self.transition.set_free_cells(cells)
    return self

  def act(self, states, seed=None, chunk=65_536):
    """A point expert action for raw ``s_t`` (the blind policy head).

    ``seed`` reaches a mixture head, where ``None`` is its deterministic mode
    and an integer draws; the deterministic head ignores it, and passing one
    to it raises rather than silently returning the same action.
    """
    if isinstance(self.policy, MixtureExpertPolicy):
      return self.policy.act(states, seed=seed, chunk=chunk)
    if seed is not None:
      raise ValueError('seed= is meaningless for a deterministic ExpertPolicy '
                       "head; refit with dyn_policy_head='mdn' to sample")
    return self.policy.act(states, chunk=chunk)

  @property
  def is_mixture(self):
    """True when the expert head is a mixture and can be sampled from."""
    return isinstance(self.policy, MixtureExpertPolicy)

  def _require_mixture(self, what):
    if not self.is_mixture:
      raise ValueError(
          f'{what} needs a mixture expert head; this model carries a '
          f'{type(self.policy).__name__}. Refit with '
          "dyn_policy_head='mdn'.")
    return self.policy

  @property
  def state_dim(self):
    return self.transition.state_dim

  @property
  def action_dim(self):
    return self.transition.action_dim

  @property
  def next_action_input(self):
    return self.transition.next_action_input

  def _draw_actions(self, s, seed, chunk):
    """One step's actions: ``(a, component_index_or_None)`` for raw states.

    A mixture head DRAWS (and reports which branch it drew); the deterministic
    head returns its single action and ``None``.
    """
    if self.is_mixture:
      return self.policy.sample(s, seed=seed, chunk=chunk)
    return self.policy.act(s, chunk=chunk), None

  def _branch(self, s, a, comp, off_diagonal, off_diagonal_action, rng,
              chunk, agent_actions=None):
    """Pick the diagonal or the off-diagonal per row; return ``(a', off)``.

    ``a'`` is the third input block of the dynamics head, ``a`` on the factual
    rows and a counterfactual action on the rest; ``off`` is the boolean mask
    of the counterfactual rows. See ``sample`` for the coin and both warnings.

    BOTH counterfactual modes are gated by the SAME coin, ``sum_i w_i(s) ** 2``
    from the mixture expert head -- the probability that two independent draws
    of the hidden confounder land in the same branch. They differ only in
    where the counterfactual action comes from when the coin says
    off-diagonal: ``'mixture'`` draws one (uniform over the action box, or
    from another branch of the expert mixture) and ``'agent'`` takes the
    learner's own action at the same state.

    In ``'agent'`` mode the coin and the agent/expert disagreement are ANDed,
    because a row can only be a counterfactual query if the query actually
    differs: where the learner already plays what the expert would, ``a' = a``
    and the row is the diagonal whatever the coin said. So a batch's
    off-diagonal share is at most ``1 - E[sum_i w_i ** 2]`` OVER THE STATES IT
    ACTUALLY VISITED, and thins out further as the agent converges onto the
    expert. In a ``rollout`` those are generated states, not dataset states,
    and the two means differ -- a chain that drifts into more confounded
    states tosses a lower coin than the dataset average would predict, so
    read the reported share against the states of the batch, not against a
    number measured on the frozen data.
    """
    n = s.shape[0]
    if off_diagonal not in OFF_DIAGONAL_MODES:
      raise ValueError(f'off_diagonal={off_diagonal!r}; expected one of '
                       f'{sorted(OFF_DIAGONAL_MODES)}')
    if off_diagonal_action not in OFF_DIAGONAL_ACTIONS:
      raise ValueError(f'off_diagonal_action={off_diagonal_action!r}; '
                       f'expected one of {sorted(OFF_DIAGONAL_ACTIONS)}')
    if off_diagonal == 'none':
      return a, np.zeros(n, bool)
    pi = self._require_mixture(f'off_diagonal={off_diagonal!r}')
    # The coin: sum_i w_i(s) ** 2, the probability the confounder repeats.
    p_diag = pi.diagonal_prob(s, chunk=chunk)
    # Both draws happen on every row whatever the coin says, so the RNG stream
    # does not depend on how the coins fell and a re-run reproduces the batch.
    off = rng.random(n) >= p_diag
    if off_diagonal == 'agent':
      if agent_actions is None:
        raise ValueError(
            "off_diagonal='agent' needs the learner's actions: pass "
            'agent_action_fn=, a callable (states, goals) -> [n, action_dim]')
      a_cf = np.asarray(agent_actions, np.float32)
      if a_cf.shape != a.shape:
        raise ValueError(f'agent_action_fn returned {a_cf.shape}, expected '
                         f'{a.shape}')
      off &= np.any(a_cf != a, axis=-1)
      return np.where(off[:, None], a_cf, a).astype(np.float32), off
    if off_diagonal_action == 'uniform':
      a_cf = rng.uniform(np.asarray(pi.action_lo, np.float64),
                         np.asarray(pi.action_hi, np.float64),
                         size=a.shape).astype(np.float32)
    elif off_diagonal_action == 'mixture':
      if comp is None:
        raise ValueError("off_diagonal_action='mixture' needs the component "
                         'the factual action was drawn from')
      a_cf = pi.sample_other_component(
          s, comp, seed=int(rng.integers(0, 2 ** 32)), chunk=chunk)
    return np.where(off[:, None], a_cf, a).astype(np.float32), off

  def sample(self, states, actions=None, next_states=None, goals=None,
             n=None, seed=0, index=None, replace=False, chunk=65_536,
             off_diagonal='none', off_diagonal_action='uniform',
             agent_action_fn=None):
    """Generate transitions by running BOTH heads on drawn dataset states.

    The forward composition of the stage-0 artifact, and the cheapest end-to-
    end check that the two heads are consistent with each other rather than
    only with their own targets:

      1. draw ``n`` rows uniformly from the offline dataset (without
         replacement unless ``replace``) -- the state distribution is the
         dataset's, since neither head models it;
      2. draw ``(a_hat, i)`` from the blind expert head: a branch
         ``i ~ Categorical(w(s))`` of the hidden confounder and an action
         inside it. A deterministic head returns its single action instead;
      3. ``s'_hat`` from the dynamics head, either the FACTUAL diagonal
         ``f(s, a_hat, a_hat)`` or the COUNTERFACTUAL off-diagonal
         ``f(s, a_hat, a')`` -- see ``off_diagonal`` -- and then through the
         feasibility clip, so a one-step draw and a ``rollout`` step agree
         about what a physically possible state is. ``result.clipped`` marks
         the rows that moved, because on those rows ``compare()`` scores the
         corrected state rather than the head's raw output;
      4. carry the drawn rows' GOAL through unchanged. No head predicts a
         goal, so it is inherited, not generated.

    Pass ``actions`` and ``next_states`` -- the stored ones, row-aligned with
    ``states`` -- to get a self-comparing batch: the result also carries the
    ground truth for the drawn rows and ``result.compare()`` / ``.summary()``
    report how far the generated pair landed from it. Without them the batch
    is generation-only and ``compare()`` returns None.

    Args:
      states: [N, S] raw dataset states, e.g. the first array out of
        ``ReplayBuffer.flat_transitions()``.
      actions: [N, A] the stored actions for those rows, optional.
      next_states: [N, S] the stored next states for those rows, optional.
      goals: [N, G] the stored goal half, optional -- ``flat_goals()`` is
        row-for-row aligned with ``flat_transitions()`` and is what belongs
        here.
      n: how many rows to draw; None (or >= N) takes every row in order.
      seed: seeds the row draw, the action draw and the branch coin. A
        mixture head is STOCHASTIC, so unlike the pre-mixture version of this
        method two calls that draw the same ``index`` agree only if they also
        share ``seed``.
      index: explicit rows to use, which overrides ``n`` and ``seed``'s effect
        on the row draw -- for regenerating the same batch, or for sampling a
        slice (the held-out episodes, the detour steps) chosen by the caller.
      replace: draw with replacement, so ``n`` may exceed N.
      chunk: batch size for the two forward passes.
      off_diagonal: ``'none'`` (default) emits only the factual diagonal, the
        only query the fit ever supervised. ``'mixture'`` emits, per row
        independently,

            f(s, a_hat, a_hat)   with probability sum_i w_i(s) ** 2
            f(s, a_hat, a')      otherwise

        i.e. the factual transition where the hidden confounder is settled at
        this state and a counterfactual where it is not. ``sum_i w_i ** 2`` is
        the probability that two independent draws of the confounder land in
        the same branch, so the counterfactual fires in proportion to how much
        the confounder matters here: never at a state where one branch has all
        the weight, and with probability ``1 - 1/k`` where the k branches are
        equally likely. Taking the coin at the STATE level rather than as the
        drawn component's own weight keeps the branch choice independent of
        which action was drawn, so a rare-branch draw is not systematically
        the row sent off-diagonal. Requires a mixture head.

        THE WARNING that outlives the discriminator this replaced: the
        off-diagonal rows are UNIDENTIFIED. The fit saw ``a' = a`` on every
        transition, so no gradient has ever supervised ``a' != a``; the
        measured off-diagonal sensitivity says a swap moves the prediction by
        about half a real step, with nothing establishing that it moves it
        correctly. Those rows are the network's initialization, not physics,
        until an off-diagonal training set exists.
        ``'agent'`` is the third mode and the one dataset augmentation uses.
        It runs the SAME coin as ``'mixture'`` and differs only in where the
        counterfactual action comes from:

            f(s, a_hat, a_hat)   with probability sum_i w_i(s) ** 2
            f(s, a_hat, a_pi)    otherwise, a_pi from ``agent_action_fn``

        so a step is factual where the confounder is settled at this state,
        and asks "the expert would have played a_hat, what if the CURRENT
        learner's a_pi were played instead" where it is not. The coin is ANDed
        with agent/expert disagreement, since a row where the learner already
        plays the expert's action is the diagonal whatever the coin said; the
        off-diagonal share therefore sits at or below
        ``1 - E[sum_i w_i ** 2]`` over the rows of THIS batch, and thins out
        further as the agent converges. Unlike ``'mixture'`` this is only as
        meaningful as the fit behind it: check
        ``model.transition.off_diagonal_trained``.
      off_diagonal_action: where ``a'`` comes from on the counterfactual rows.
        ``'uniform'`` draws from the train-split action box, which is off the
        action manifold and so extrapolates twice over. ``'mixture'`` draws
        from a DIFFERENT branch of the mixture than the factual action came
        from, which keeps ``a'`` admissible and gives the query a reading --
        "what if the confounder had been branch j instead of branch i" -- and
        needs ``k >= 2``. Ignored by ``off_diagonal='agent'``, which takes
        ``a'`` from the learner instead of drawing one.
      agent_action_fn: ``(states, goals) -> [n, action_dim]``, the learner's
        policy. Required by ``off_diagonal='agent'`` and unused otherwise.
        ``goals`` is the drawn rows' inherited goal half, or None when the
        call was made without goals -- a goal-conditioned learner needs them,
        so pass ``goals=`` too.

    Returns:
      A ``SampledTransitions``.
    """
    s_all = np.asarray(states, np.float32)
    if s_all.ndim != 2:
      raise ValueError(f'states must be [N, state_dim], got {s_all.shape}')
    if s_all.shape[1] != self.state_dim:
      raise ValueError(f'states have width {s_all.shape[1]} but the model was '
                       f'fitted on state_dim={self.state_dim}')
    n_rows = s_all.shape[0]

    def _aligned(x, name, width=None):
      """Optional companion array, checked against the state rows."""
      if x is None:
        return None
      x = np.asarray(x, np.float32)
      if x.ndim != 2 or x.shape[0] != n_rows:
        raise ValueError(f'{name} must have {n_rows} rows to align with '
                         f'states, got {x.shape}')
      if width is not None and x.shape[1] != width:
        raise ValueError(f'{name} must have width {width}, got {x.shape[1]}')
      return x

    a_all = _aligned(actions, 'actions', self.action_dim)
    sn_all = _aligned(next_states, 'next_states', self.state_dim)
    g_all = _aligned(goals, 'goals')

    if index is not None:
      idx = np.asarray(index, np.int64)
      if idx.ndim != 1:
        raise ValueError(f'index must be 1-D, got {idx.shape}')
      if idx.size and (idx.min() < 0 or idx.max() >= n_rows):
        raise ValueError(f'index out of range for {n_rows} rows')
    elif n is None or (not replace and n >= n_rows):
      idx = np.arange(n_rows, dtype=np.int64)
    else:
      if n < 0:
        raise ValueError(f'n must be non-negative, got {n}')
      idx = np.random.default_rng(seed).choice(n_rows, size=int(n),
                                               replace=bool(replace))
      idx = np.asarray(idx, np.int64)

    s = s_all[idx]
    g = None if g_all is None else g_all[idx]
    # Own streams off the same seed: the row draw above must keep the draws it
    # always made, so the action draw and the branch coin get their own primes.
    a_gen, comp = self._draw_actions(s, seed + 7_919, chunk)          # step 2.
    a_agent = (agent_action_fn(s, g) if off_diagonal == 'agent'
               and agent_action_fn is not None else None)
    a_cf, off = self._branch(s, a_gen, comp, off_diagonal,            # step 3.
                             off_diagonal_action,
                             np.random.default_rng(seed + 15_487), chunk,
                             agent_actions=a_agent)
    sn_raw = self.predict(s, a_gen, a_cf, chunk=chunk)
    # The same feasibility clip `rollout` applies, for the same reason: these
    # rows are generated next states and a consumer must not be handed one
    # inside a wall just because this entry point takes a single step. See
    # `TransitionModel.clip_state`.
    sn_gen = self.clip_state(sn_raw)
    clipped = (np.any(sn_gen != sn_raw, axis=-1)
               if self.constrains_states else None)
    a_true = None if a_all is None else a_all[idx]
    # Dynamics head on the STORED action: the control that separates its own
    # error from the error the generated action handed it. Always the factual
    # diagonal -- it is the dynamics head's own error that is wanted here --
    # and deliberately NOT clipped, so `dyn_only_rmse_raw` stays the raw head's
    # number and is comparable with the held-out error the fit reports.
    sn_data_action = (None if a_true is None
                      else self.predict(s, a_true, chunk=chunk))

    return SampledTransitions(
        index=idx, states=s, actions=a_gen, next_states=sn_gen,
        goals=g,                                       # step 4: inherited.
        data_actions=a_true,
        data_next_states=None if sn_all is None else sn_all[idx],
        next_states_data_action=sn_data_action,
        components=comp,
        off_diagonal_trained=bool(self.transition.off_diagonal_trained),
        clipped=clipped,
        off_diagonal=None if off_diagonal == 'none' else off,
        counterfactual_actions=(
            None if off_diagonal == 'none'
            else np.where(off[:, None], a_cf, np.nan).astype(np.float32)))

  def rollout(self, states, actions, next_states, episode_ids, steps,
              goals=None, n=1, seed=0, episodes=None, pad_to=None,
              ground_truth=True, off_diagonal='none',
              off_diagonal_action='uniform', agent_action_fn=None,
              stop_on_done=True):
    """Generate trajectories: up to ``steps`` model steps from an INITIAL state.

    Every rollout starts at the first state of its source episode -- the state
    the environment resets to -- and nothing else is inherited from the
    dataset. No prefix of real transitions is copied in, and no state from
    partway through an episode is ever used as a starting point::

        s_0                              the source episode's INITIAL state
        a_hat_t = pi_phi(s_t)            for t = 0 .. T-1
        s_hat_{t+1} = f_theta(s_t, a_hat_t)

    so every row has ``T + 1`` observations, of which exactly one (the first)
    is real and ``T`` were generated, and every transition in the batch is the
    model's. The row carries the source episode's goal throughout.

    Starting from the reset state is what makes a generated trajectory a
    trajectory the policy could actually live: the contrastive learner relabels
    goals to a future state of the same row, and a row that began at a state
    reached by 200 real steps would hand it futures conditioned on a history
    the model never produced. The same reason makes the starts comparable to
    each other -- in these environments the reset distribution is narrow, so
    all rows begin in essentially the same place and differ only through what
    the heads generated.

    All ``n`` rollouts are stepped TOGETHER, so the cost is ``T`` batched
    forward passes rather than ``n * T`` calls.

    The START of a rollout is always real: the initial states come from the
    arrays passed in, which are the frozen dataset's. No generated trajectory
    can become the base of another one -- ``crl/ETT_train.py`` puts generated
    rows in a SEPARATE buffer and asserts the dataset buffer neither changed
    content nor grew.

    COMPOUNDING, with either counterfactual mode: the chain advances through
    the off-diagonal map whenever the coin says so, and every later step
    inherits that state. At a confounded state the coin fires about
    ``1 - 1/k`` of the time, so a long rollout is all-factual with vanishing
    probability -- at k=3 weights near ``(0.8, 0.15, 0.05)`` the per-step rate
    is 0.335 and a 20-step rollout avoids it with probability 2e-4. Nearly
    every generated trajectory will therefore contain at least one
    off-diagonal step, which is exactly the point when the off-diagonal was
    SUPERVISED by a negative dataset and a defect when it was not: with
    ``off_diagonal='mixture'`` on an unsupervised fit those steps are the
    network's initialization rather than physics, usable as a diagnostic and
    not as training data. Check ``model.transition.off_diagonal_trained``
    before rolling out into it.

    One action is generated past the last observation (at index ``T``). That
    slot is not a transition of the trajectory, but ``crl/replay.py`` reads
    ``act[traj, i + 1]`` up to the last valid observation index, so leaving it
    zero would feed the learner a fabricated zero action.

    THE FEASIBILITY CLIP. When the dynamics head carries a state box (see
    ``TransitionModel.state_lo`` and ``crl.d4rl_ant.ant_state_bounds``) or a
    maze floor plan (``TransitionModel.free_xy``, from ``maze_free_cells``),
    every generated state is forced back into the feasible region by
    ``clip_state`` before it is stored AND before it is fed back in, so a
    chain that starts to leave the environment's state space -- or to walk
    into a wall -- is corrected at the step that left rather than at the end.
    That ordering is the whole point: the expert head and the dynamics head of
    step ``t+1`` see the corrected state, so one excursion cannot compound.
    The initial state is real and is never clipped.
    ``RolloutTrajectories.clipped`` records where the clip moved the state and
    ``clipped_frac`` summarizes it -- read it as a diagnostic of the FIT,
    because a constraint that has to bind often is a chain that is diverging,
    and pinning a divergent state to the wall does not make it data.

    THE DONE HEAD. When the model carries one and ``stop_on_done`` is True,
    every generated state is scored by ``done.is_done`` (after the clip) and a
    row STOPS at the first state it flags: that state is the row's last valid
    observation, ``lengths[b]`` is set to its index + 1, the trailing action is
    drawn at it as usual, and nothing past it is generated or stored (padding
    stays zero, goal half included). The initial state is never scored. Rows
    that are not flagged run the full ``T`` steps. The batch keeps stepping
    together; a stopped row is simply no longer written.

    Args:
      states, actions, next_states, episode_ids: the flat dataset arrays, as
        ``ReplayBuffer.flat_transitions()`` returns them. Transitions of one
        episode must be CONTIGUOUS and in time order (they are, from that
        method); this is checked, not assumed, because both the initial state
        and the real continuation used for scoring depend on it.
      steps: T, how many steps to generate. ``config.dyn_rollout_steps``,
        whose documented ceiling is 70 -- the whole row is generated now, so T
        is also the length of the trajectory the relabeler sees.
      goals: [N, G] the stored goal half (``flat_goals()``). Omitted means the
        generated observations are the state half alone.
      n: how many trajectories to generate.
      seed: seeds the episode draw, the action draws and the coin.
      episodes: explicit source episode ids, one per rollout -- overrides the
        episode draw (and sets ``n``). Only their INITIAL states are used.
      pad_to: length of the padded time axis. Defaults to what the batch
        needs; set it to a target buffer's ``ep_len_obs`` to hand the rows
        straight to ``add_episode``.
      ground_truth: also return the real continuation from the same initial
        state, so ``drift()`` can score the rollout. Costs one gather, no
        extra forward passes.
      off_diagonal: ``'none'`` (default), ``'mixture'`` or ``'agent'``, per
        step, exactly as in ``sample`` -- including the warning that
        off-diagonal rows are unidentified unless the fit was given a
        negative dataset, and the compounding note above.
      off_diagonal_action: where ``a'`` comes from, exactly as in ``sample``.
      agent_action_fn: ``(states, goals) -> [n, action_dim]``, the learner's
        policy, called once per generated step on the CURRENT states and the
        rows' inherited goals. Required by ``off_diagonal='agent'``. Because
        it reads the learner, a batch of rollouts is only as current as the
        parameters it was generated with -- ``crl/ETT_train.py`` regenerates
        the whole augmentation buffer on a schedule for exactly that reason.
      stop_on_done: stop each row at the first state the done head flags (see
        above). Ignored when the model has no done head.

    Returns:
      A ``RolloutTrajectories``.
    """
    T = int(steps)
    if T < 1:
      raise ValueError(f'steps must be at least 1, got {steps}')
    s_all = np.asarray(states, np.float32)
    a_all = np.asarray(actions, np.float32)
    sn_all = np.asarray(next_states, np.float32)
    eid = np.asarray(episode_ids)
    if s_all.ndim != 2 or s_all.shape[1] != self.state_dim:
      raise ValueError(f'states must be [N, {self.state_dim}], got '
                       f'{s_all.shape}')
    for name, arr, width in (('actions', a_all, self.action_dim),
                             ('next_states', sn_all, self.state_dim)):
      if arr.shape != (s_all.shape[0], width):
        raise ValueError(f'{name} must be [{s_all.shape[0]}, {width}], got '
                         f'{arr.shape}')
    if eid.shape != (s_all.shape[0],):
      raise ValueError(f'episode_ids must be [{s_all.shape[0]}], got '
                       f'{eid.shape}')
    g_all = None
    if goals is not None:
      g_all = np.asarray(goals, np.float32)
      if g_all.ndim != 2 or g_all.shape[0] != s_all.shape[0]:
        raise ValueError(f'goals must have {s_all.shape[0]} rows, got '
                         f'{g_all.shape}')

    # -- episode index: first flat row and transition count of each episode.
    ep_ids, first, count = np.unique(eid, return_index=True,
                                     return_counts=True)
    for e, f, c in zip(ep_ids, first, count):
      if not np.array_equal(eid[f:f + c], np.full(c, e)):
        raise ValueError(f'episode {e} is not a contiguous run of rows in '
                         'episode_ids; rollout needs the episode in time '
                         'order to take its initial state')
    slot = {int(e): (int(f), int(c)) for e, f, c in zip(ep_ids, first, count)}

    # -- draw the source episodes. There is no branch draw: the start is the
    # episode's own first state, so all a draw selects is which reset (and
    # which goal) the row carries.
    rng = np.random.default_rng(seed)
    if episodes is None:
      ep = ep_ids[rng.integers(0, ep_ids.size, size=int(n))]
    else:
      ep = np.asarray(episodes)
      if ep.ndim != 1:
        raise ValueError(f'episodes must be 1-D, got {ep.shape}')
      for e in np.unique(ep):
        if int(e) not in slot:
          raise ValueError(f'episode {e} is not in episode_ids')
    rows = ep.size

    # Every row is the same length now: one real initial observation plus the
    # T the model generates.
    lengths = np.full(rows, T + 1, np.int64)
    L = T + 1
    if pad_to is not None:
      if int(pad_to) < L:
        raise ValueError(f'pad_to={pad_to} is shorter than the rollouts in '
                         f'this batch ({L})')
      L = int(pad_to)

    goal_dim = 0 if g_all is None else g_all.shape[1]
    obs = np.zeros((rows, L, self.state_dim + goal_dim), np.float32)
    act = np.zeros((rows, L, self.action_dim), np.float32)
    gen_mask = np.zeros((rows, L), bool)
    # Only allocated when asked for: these are the same size as `obs` again,
    # and the augmentation path regenerates its whole batch on a schedule
    # with ground_truth=False -- paying for three unused arrays every round.
    d_states = (np.zeros((rows, L, self.state_dim), np.float32)
                if ground_truth else None)
    d_act = (np.zeros((rows, L, self.action_dim), np.float32)
             if ground_truth else None)
    d_valid = np.zeros((rows, L), bool) if ground_truth else None

    # -- seed each row with its episode's initial state, and copy the source's
    # own continuation from that same state for scoring.
    for b in range(rows):
      f, c = slot[int(ep[b])]
      obs[b, 0, :self.state_dim] = s_all[f]
      if g_all is not None:
        obs[b, :T + 1, self.state_dim:] = g_all[f]
      if ground_truth:
        # Every state the source episode has: s_0..s_{c-1} then its terminal.
        src_s = np.concatenate([s_all[f:f + c], sn_all[f + c - 1][None]])
        # How much of this row's time axis the SOURCE episode actually covers:
        # a short episode simply stops contributing past its own end.
        k = min(T + 1, c + 1)                     # real observations.
        m = min(T + 1, c)                         # real actions.
        d_states[b, :k] = src_s[:k]
        d_act[b, :m] = a_all[f:f + m]
        d_valid[b, :k] = True

    # -- step every rollout together: T policy calls and T dynamics calls.
    # Two streams of their own, off the same seed and separate from the
    # episode draw above, so adding the action draw and the coin did not
    # disturb which episodes an existing call selects.
    a_rng = np.random.default_rng(seed + 7_919)
    cf_rng = np.random.default_rng(seed + 15_487)
    off_steps = np.zeros((rows, L), bool)
    clip_steps = (np.zeros((rows, L), bool)
                  if self.constrains_states else None)
    comp_steps = np.full((rows, L), -1, np.int64)
    use_done = bool(stop_on_done) and self.has_done
    alive = np.ones(rows, bool)
    terminated = np.zeros(rows, bool)
    cur = obs[:, 0, :self.state_dim].copy()
    for t in range(T + 1):                          # +1: the trailing action.
      a_hat, comp = self._draw_actions(
          cur, int(a_rng.integers(0, 2 ** 32)), 65_536)
      # A row is written up to its last valid observation, which for a row
      # the done head stopped at the previous step is this slot: that is its
      # trailing action.
      w = t < lengths
      act[w, t] = a_hat[w]
      if comp is not None:
        comp_steps[w, t] = comp[w]
      if t == T or not alive.any():
        break
      a_agent = None
      if off_diagonal == 'agent' and agent_action_fn is not None:
        # The goal half is constant over a row's valid span, so reading it at
        # the current slot is the same as reading it anywhere in the row.
        g_cur = None if goal_dim == 0 else obs[:, t, self.state_dim:]
        a_agent = agent_action_fn(cur, g_cur)
      a_cf, off = self._branch(cur, a_hat, comp, off_diagonal,
                               off_diagonal_action, cf_rng, 65_536,
                               agent_actions=a_agent)
      # The clip is INSIDE the chain, not applied to the finished array: the
      # clipped state is what the next step's expert head and dynamics head
      # are queried on, so an excursion is corrected once instead of being
      # compounded and then hidden at the end.
      raw = self.predict(cur, a_hat, a_cf)
      nxt = self.clip_state(raw)
      if clip_steps is not None:
        clip_steps[alive, t + 1] = np.any(nxt != raw, axis=-1)[alive]
      obs[alive, t + 1, :self.state_dim] = nxt[alive]
      gen_mask[alive, t + 1] = True
      # Indexed at the DESTINATION, like `generated`: the flag describes the
      # transition that produced this observation.
      off_steps[alive, t + 1] = off[alive]
      cur = np.where(alive[:, None], nxt, cur)
      if use_done:
        stop = alive & self.done.is_done(nxt)
        lengths[stop] = t + 2
        terminated |= stop
        alive &= ~stop

    if use_done:
      # Padding past a stopped row's end is zero, goal half included.
      for b in np.flatnonzero(terminated):
        obs[b, lengths[b]:] = 0.0

    return RolloutTrajectories(
        episode=ep, obs=obs, act=act, lengths=lengths,
        generated=gen_mask, steps=T, state_dim=self.state_dim,
        data_states=d_states if ground_truth else None,
        data_act=d_act if ground_truth else None,
        data_valid=d_valid if ground_truth else None,
        components=None if not self.is_mixture else comp_steps,
        off_diagonal_trained=bool(self.transition.off_diagonal_trained),
        off_diagonal=None if off_diagonal == 'none' else off_steps,
        clipped=clip_steps,
        terminated=terminated if use_done else None)


def _split_by_episode(episode_ids, holdout_frac, rng):
  """Train/val masks that keep whole episodes together (see module docstring)."""
  eps = np.unique(episode_ids)
  n_val = int(round(holdout_frac * eps.size))
  n_val = min(max(n_val, 1 if holdout_frac > 0 else 0), max(eps.size - 1, 0))
  if n_val == 0:
    return np.ones(episode_ids.shape, bool), np.zeros(episode_ids.shape, bool), 0
  val_eps = rng.permutation(eps)[:n_val]
  val = np.isin(episode_ids, val_eps)
  return ~val, val, n_val


def _raw_error_stats(pred, truth, prefix):
  """Raw-unit error summary: MSE, RMSE, XY RMSE, explained variance."""
  err = pred - truth
  mse_per_dim = np.mean(err ** 2, axis=0)
  var_per_dim = np.var(truth, axis=0)
  # Explained variance is aggregated over dims (sum of MSE / sum of variance)
  # so a near-constant column cannot dominate the ratio.
  denom = float(np.sum(var_per_dim))
  return {
      prefix + '_mse_raw': float(np.mean(mse_per_dim)),
      prefix + '_rmse_raw': float(np.sqrt(np.mean(mse_per_dim))),
      prefix + '_rmse_xy': float(np.sqrt(np.mean(mse_per_dim[:2]))),
      prefix + '_rmse_per_dim': [float(v) for v in np.sqrt(mse_per_dim)],
      prefix + '_explained_variance': (
          float(1.0 - np.sum(mse_per_dim) / denom) if denom > 0 else float('nan')),
  }


def _action_error_stats(pred, truth, prefix):
  """Raw-unit action error summary (the expert-policy analogue of the above)."""
  err = pred - truth
  mse_per_dim = np.mean(err ** 2, axis=0)
  var_per_dim = np.var(truth, axis=0)
  denom = float(np.sum(var_per_dim))
  return {
      prefix + '_mse_raw': float(np.mean(mse_per_dim)),
      prefix + '_rmse_raw': float(np.sqrt(np.mean(mse_per_dim))),
      prefix + '_rmse_per_dim': [float(v) for v in np.sqrt(mse_per_dim)],
      prefix + '_explained_variance': (
          float(1.0 - np.sum(mse_per_dim) / denom) if denom > 0 else float('nan')),
  }


POLICY_HEADS = ('mdn', 'deterministic')
OFF_DIAGONAL_MODES = ('none', 'mixture', 'agent')
OFF_DIAGONAL_ACTIONS = ('uniform', 'mixture')
#: Where the FIT draws the counterfactual a' for a negative minibatch. Not the
#: same set as OFF_DIAGONAL_ACTIONS above, which is the SAMPLER's: 'mixture'
#: is a sampling-time device (draw from another branch of the expert head) and
#: 'shuffle' a fitting-time one (take another negative row's stored action).
OFF_DIAGONAL_TRAIN_ACTIONS = ('uniform', 'shuffle')


def _mixture_stats(policy, states, actions, prefix):
  """Diagnostics for a fitted mixture head on one split.

  Everything here is about whether the k components did what they are for --
  split the data by the hidden confounder -- rather than about point accuracy,
  which a mixture head is not trying to optimize:

    ``nll``               mean negative log-likelihood of the STORED actions,
                          the fit's own objective and the number that selects
                          k. Lower is better; comparable across fits on the
                          same split and the same units, and NOT a pure
                          density (see ``censored_component_log_prob``).
    ``weight_mean``       the average mixture weight per component. A
                          component whose mean weight is ~0 everywhere is dead
                          capacity, and k should come down.
    ``resp_frac``         share of rows whose most likely branch is component
                          j. Unlike ``weight_mean`` this is posterior, so it
                          says whether the components actually claim distinct
                          DATA rather than merely carrying prior weight.
    ``diag_prob_mean``    mean of ``sum_i w_i ** 2``, i.e. the average rate at
                          which the sampler will take the factual diagonal --
                          in BOTH counterfactual modes, ``'mixture'`` and the
                          augmentation path's ``'agent'``. 1.0 means the head
                          found no confounding anywhere and the counterfactual
                          branch would never fire. Taken through
                          ``policy.diagonal_prob``, so it is the coin AFTER
                          ``weight_temperature`` -- this row is a statement
                          about the sampler, not about the fit.
    ``branch_entropy``    mean Shannon entropy of ``w(s)`` in nats, 0 at a
                          state with one branch and ``log k`` at a uniform
                          one -- the direct readout of how multi-modal the
                          FITTED conditional is, untempered, so it stays
                          comparable across sampling settings.
    ``sample_branch_entropy``
                          the same entropy on the tempered weights the
                          sampler draws from. Equal to ``branch_entropy`` at
                          ``weight_temperature = 1``; the gap between the two
                          is exactly what the temperature bought.
    ``mode_rmse_raw``     RMSE of the highest-weight component's mean against
                          the stored action, and
    ``mean_rmse_raw``     the same for the mixture mean. Reported only so a
                          mixture fit can be put next to the deterministic
                          head's single number; a BETTER mixture can score
                          WORSE here, and a fit whose mean beats its own mode
                          is telling you the branches have not separated.
    ``censored_frac``     share of stored action COORDINATES sitting on a box
                          limit, i.e. how much of the likelihood is carried by
                          censoring rather than by density. Measured at 0.0011
                          on V5 far05 -- small, but a single such row is what
                          would make a squashed-Gaussian loss infinite, which
                          is why the head censors rather than squashes.
  """
  lp = policy.log_prob(states, actions)
  w = policy.weights(states)
  resp = policy.responsibilities(states, actions)
  mode = policy.mode_action(states)
  mean = policy.mean_action(states)
  lo = np.asarray(policy.action_lo, np.float32)
  hi = np.asarray(policy.action_hi, np.float32)
  a = np.asarray(actions, np.float32)
  ent = -np.sum(w * np.log(np.clip(w, 1e-30, None)), axis=-1)
  # The sampler's own weights: identical to `w` unless weight_temperature != 1.
  ws = np.exp(policy._sampling_log_w(np.log(np.clip(w, 1e-30, None))))
  ent_s = -np.sum(ws * np.log(np.clip(ws, 1e-30, None)), axis=-1)
  arg = np.argmax(resp, axis=-1)
  k = int(policy.n_components)
  return {
      f'{prefix}_nll': float(-np.mean(lp)),
      f'{prefix}_weight_mean': [float(x) for x in np.mean(w, axis=0)],
      f'{prefix}_resp_frac': [float(np.mean(arg == j)) for j in range(k)],
      f'{prefix}_diag_prob_mean': float(np.mean(policy.diagonal_prob(states))),
      f'{prefix}_branch_entropy': float(np.mean(ent)),
      f'{prefix}_sample_branch_entropy': float(np.mean(ent_s)),
      f'{prefix}_branch_entropy_max': float(np.log(k)),
      f'{prefix}_mode_rmse_raw': float(np.sqrt(np.mean((mode - a) ** 2))),
      f'{prefix}_mean_rmse_raw': float(np.sqrt(np.mean((mean - a) ** 2))),
      f'{prefix}_censored_frac': float(np.mean((a <= lo) | (a >= hi))),
  }


def action_input_sensitivity(model, states, actions, n=4096, seed=0,
                             eps=0.25):
  """How an ``(s, a, a')`` dynamics head splits its action dependence.

  ``a'`` is the COUNTERFACTUAL action -- an alternative to ``a`` at the same
  state, not ``a_{t+1}``. The offline dataset only ever shows the model the
  factual ``a' = a``, so the SUM of the two Jacobian blocks is what the fit is
  supervised on and their individual values are not identified: an equally
  good fit exists for any split. This measures the split that this particular
  initialization happened to land on, so the report says out loud how much of
  a claimed "intervention on a'" is really just an arbitrary internal
  weighting.

  Returned keys (all on the raw, unstandardized scale):

    * ``action_jac_a`` / ``action_jac_next_a`` -- mean Frobenius norm of
      ``d s'/d a`` and ``d s'/d a'`` over ``n`` sampled held-out states.
    * ``action_jac_diag`` -- norm of ``d s'/d a + d s'/d a'``, the FACTUAL
      diagonal derivative. This is the only physically meaningful one: it is
      what a plain ``f(s, a)`` model's Jacobian would be, and it is the direction the
      data constrains.
    * ``action_next_a_share`` -- ``|J_a'| / (|J_a| + |J_a'|)``. 0.5 means the
      net simply halved a single response across the two copies; 0 or 1 means
      it effectively ignored one block. Neither is better; both are arbitrary.
    * ``action_block_cosine`` -- cosine between the two flattened blocks. Near
      +1 says the blocks are the same map scaled, i.e. the net learned one
      action response and distributed it.
    * ``offdiag_shift_rmse`` -- raw-unit RMSE between ``f(s, a, a')`` with the
      counterfactual ``a'`` drawn from a DIFFERENT transition and the factual
      ``f(s, a, a)``,
      next to ``true_step_rmse``, the size of the real one-step state change.
      Their ratio says how large the model's unconstrained off-diagonal
      response is compared to the motion it was actually fitted on.
  """
  if not getattr(model, 'next_action_input', False):
    return {}
  transition = getattr(model, 'transition', model)
  s = np.asarray(states, np.float32)
  a = np.asarray(actions, np.float32)
  rng = np.random.default_rng(int(seed))
  if s.shape[0] > n:
    idx = rng.choice(s.shape[0], size=int(n), replace=False)
    s, a = s[idx], a[idx]

  fn = transition._predict_fn()
  params = transition.params
  jac = jax.jit(jax.vmap(jax.jacrev(lambda si, ai, a2i: fn(params, si, ai, a2i),
                                    argnums=(1, 2))))
  j_a, j_a2 = jac(jnp.asarray(s), jnp.asarray(a), jnp.asarray(a))
  j_a, j_a2 = np.asarray(j_a), np.asarray(j_a2)
  fro = lambda j: float(np.mean(np.sqrt(np.sum(j ** 2, axis=(1, 2)))))
  flat_a = j_a.reshape(j_a.shape[0], -1)
  flat_a2 = j_a2.reshape(j_a2.shape[0], -1)
  denom = (np.linalg.norm(flat_a, axis=1) * np.linalg.norm(flat_a2, axis=1))
  cos = np.sum(flat_a * flat_a2, axis=1) / np.maximum(denom, 1e-12)
  n_a, n_a2 = fro(j_a), fro(j_a2)

  # Counterfactual probe: keep (s, a), swap in another transition's action as
  # the counterfactual a'.
  perm = rng.permutation(s.shape[0])
  diag = transition.predict(s, a)
  off = transition.predict(s, a, a[perm])
  # Same probe on the SUPERVISED input for scale: perturbing a itself.
  step = transition.predict(s, a) - s
  return {
      'action_jac_a': n_a,
      'action_jac_next_a': n_a2,
      'action_jac_diag': fro(j_a + j_a2),
      'action_next_a_share': float(n_a2 / max(n_a + n_a2, 1e-12)),
      'action_block_cosine': float(np.mean(cos)),
      'offdiag_shift_rmse': float(np.sqrt(np.mean((off - diag) ** 2))),
      'true_step_rmse': float(np.sqrt(np.mean(step ** 2))),
      'action_sensitivity_n': int(s.shape[0]),
  }


@dataclasses.dataclass
class _Prepared:
  """Everything ``fit`` / ``fit_causal`` share: split, normalization, tensors."""
  states: np.ndarray
  actions: np.ndarray
  next_states: np.ndarray
  n: int
  state_dim: int
  action_dim: int
  # Width of the design matrix `x_tr` / `x_va`: state_dim + action_dim, or
  # state_dim + 2 * action_dim with the action pair.
  in_dim: int
  next_action_input: bool
  next_actions: np.ndarray
  rng: Any
  tr_mask: np.ndarray
  va_mask: np.ndarray
  n_val_eps: int
  n_tr: int
  n_va: int
  s_mean: np.ndarray
  s_std: np.ndarray
  a_mean: np.ndarray
  a_std: np.ndarray
  t_mean: np.ndarray
  t_std: np.ndarray
  x_tr: Any
  y_tr: Any
  x_va: Any
  y_va: Any
  activation: str


def _prepare(states, actions, next_states, episode_ids, config, seed,
             next_actions=None):
  """Cast, split by episode, standardize and build the network tensors.

  Shared verbatim by ``fit`` and ``fit_causal``, and in that exact order: the
  ``rng`` returned here has consumed only the holdout permutation, so both
  entry points go on to draw the identical minibatch stream from it.

  ``next_actions`` is the counterfactual action ``a'`` (legacy parameter
  name) and defaults to ``a`` itself, which is what the frozen offline dataset
  holds: every stored transition is factual, on the diagonal ``a' = a``. It is
  a parameter rather than a hard-wired copy only so a future dataset that
  really does carry counterfactual actions can be fitted here without touching
  this function.
  """
  states = np.asarray(states, np.float32)
  actions = np.asarray(actions, np.float32)
  next_states = np.asarray(next_states, np.float32)
  n, state_dim = states.shape
  action_dim = actions.shape[1]
  assert next_states.shape == states.shape, 'state/next_state shape mismatch'
  assert n > 0, 'no transitions to fit the transition model on'
  pair = bool(getattr(config, 'dyn_next_action_input', False))
  if next_actions is None:
    next_actions = actions
  next_actions = np.asarray(next_actions, np.float32)
  assert next_actions.shape == actions.shape, 'action/next_action shape mismatch'

  rng = np.random.default_rng(seed)
  tr_mask, va_mask, n_val_eps = _split_by_episode(
      np.asarray(episode_ids), float(config.dyn_holdout_frac), rng)
  n_tr, n_va = int(tr_mask.sum()), int(va_mask.sum())

  # Normalization from the TRAIN split only (val must stay unseen).
  s_mean, s_std = _standardizer(states[tr_mask])
  a_mean, a_std = _standardizer(actions[tr_mask])
  raw_target = (next_states - states) if config.dyn_predict_delta else next_states
  t_mean, t_std = _standardizer(raw_target[tr_mask])

  # The counterfactual a' shares a's statistics on purpose: the two blocks
  # must be the same standardized coordinates, or the factual diagonal a' = a
  # would not be a diagonal in the network's input space.
  cols = [(states - s_mean) / s_std, (actions - a_mean) / a_std]
  if pair:
    cols.append((next_actions - a_mean) / a_std)
  x = np.concatenate(cols, axis=1)
  y = (raw_target - t_mean) / t_std
  return _Prepared(
      states=states, actions=actions, next_states=next_states,
      n=n, state_dim=state_dim, action_dim=action_dim,
      in_dim=int(x.shape[1]), next_action_input=pair,
      next_actions=next_actions,
      rng=rng, tr_mask=tr_mask, va_mask=va_mask, n_val_eps=n_val_eps,
      n_tr=n_tr, n_va=n_va,
      s_mean=s_mean, s_std=s_std, a_mean=a_mean, a_std=a_std,
      t_mean=t_mean, t_std=t_std,
      x_tr=jnp.asarray(x[tr_mask]), y_tr=jnp.asarray(y[tr_mask]),
      x_va=jnp.asarray(x[va_mask]), y_va=jnp.asarray(y[va_mask]),
      activation=str(getattr(config, 'dyn_activation', 'relu')))


@dataclasses.dataclass
class _PreparedNegative:
  """The NEGATIVE dataset, standardized into the positive fit's coordinates.

  Everything the off-diagonal term of ``fit_causal`` needs. The three blocks
  are kept apart rather than concatenated because the third one -- the
  counterfactual ``a'`` -- is redrawn at every training step, so only
  ``states`` and ``actions`` are fixed; the design matrix is assembled inside
  the jitted update.

  The normalization is the POSITIVE fit's, deliberately and without
  exception: the network has one output head in one set of standardized
  units, and re-centering the negative rows on their own statistics would ask
  the same weights to answer in two different frames. Same for the action
  box, the holdout rule (whole episodes) and the target parameterization
  (delta or absolute) -- all inherited.
  """
  n: int
  n_tr: int
  n_va: int
  n_val_eps: int
  n_episodes: int
  states: np.ndarray            # [N, S] raw, kept for the raw-unit stats.
  actions: np.ndarray           # [N, A] raw.
  next_states: np.ndarray       # [N, S] raw.
  tr_mask: np.ndarray
  va_mask: np.ndarray
  s_tr: Any                     # [n_tr, S] standardized.
  a_tr: Any                     # [n_tr, A] standardized.
  y_tr: Any                     # [n_tr, S] standardized regression target.
  s_va: Any
  a_va: Any
  y_va: Any
  acf_va: Any                   # [n_va, A] the FIXED validation a' (see below).
  acf_va_raw: np.ndarray        # the same draw in raw units.
  lo_std: np.ndarray            # action box, standardized: uniform a' draws.
  hi_std: np.ndarray
  train_action: str


def _prepare_negative(neg_states, neg_actions, neg_next_states,
                      neg_episode_ids, d, a_lo, a_hi, config, seed):
  """Cast, split and standardize the negative transitions for the fit.

  ``d`` is the positive ``_Prepared``; its standardization is reused verbatim
  (see ``_PreparedNegative``). ``a_lo`` / ``a_hi`` are the action box measured
  on the POSITIVE train split, which is what a uniform ``a'`` is drawn from --
  the box is the actuator's range, and the negative episodes were collected
  with the same actuator.

  The validation ``a'`` is drawn ONCE here rather than per evaluation, so the
  off-diagonal validation MSE is a curve of the model against a fixed target
  instead of a curve against a moving one. The training ``a'`` is the
  opposite: redrawn every step (see ``fit_causal``), so a negative transition
  is supervised against many counterfactual partners over the fit.
  """
  s = np.asarray(neg_states, np.float32)
  a = np.asarray(neg_actions, np.float32)
  sn = np.asarray(neg_next_states, np.float32)
  eid = np.asarray(neg_episode_ids)
  if s.ndim != 2 or s.shape[1] != d.state_dim:
    raise ValueError(f'negative states must be [N, {d.state_dim}] to share '
                     f'the positive fit\'s state space, got {s.shape}')
  if a.shape != (s.shape[0], d.action_dim):
    raise ValueError(f'negative actions must be [{s.shape[0]}, '
                     f'{d.action_dim}], got {a.shape}')
  if sn.shape != s.shape:
    raise ValueError('negative state/next_state shape mismatch: '
                     f'{s.shape} vs {sn.shape}')
  if eid.shape != (s.shape[0],):
    raise ValueError(f'negative episode_ids must be [{s.shape[0]}], got '
                     f'{eid.shape}')
  if s.shape[0] == 0:
    raise ValueError('the negative dataset holds no transitions')

  train_action = str(getattr(config, 'dyn_off_diagonal_train_action',
                             'uniform')).lower()
  if train_action not in OFF_DIAGONAL_TRAIN_ACTIONS:
    raise ValueError(f'dyn_off_diagonal_train_action={train_action!r}; '
                     f'expected one of {sorted(OFF_DIAGONAL_TRAIN_ACTIONS)}')

  # Its own RNG: the positive minibatch stream must keep every draw it made
  # before an off-diagonal term existed, so a run WITHOUT a negative dataset
  # is bit-for-bit the old one.
  rng = np.random.default_rng(seed)
  holdout = float(getattr(config, 'dyn_negative_holdout_frac',
                          config.dyn_holdout_frac))
  tr_mask, va_mask, n_val_eps = _split_by_episode(eid, holdout, rng)

  raw_target = (sn - s) if config.dyn_predict_delta else sn
  s_std_all = (s - d.s_mean) / d.s_std
  a_std_all = (a - d.a_mean) / d.a_std
  y_std_all = (raw_target - d.t_mean) / d.t_std

  lo_std = ((a_lo - d.a_mean) / d.a_std).astype(np.float32)
  hi_std = ((a_hi - d.a_mean) / d.a_std).astype(np.float32)
  n_va = int(va_mask.sum())
  acf_va_raw = rng.uniform(a_lo.astype(np.float64), a_hi.astype(np.float64),
                           size=(n_va, d.action_dim)).astype(np.float32)
  acf_va = (acf_va_raw - d.a_mean) / d.a_std

  return _PreparedNegative(
      n=int(s.shape[0]), n_tr=int(tr_mask.sum()), n_va=n_va,
      n_val_eps=int(n_val_eps), n_episodes=int(np.unique(eid).size),
      states=s, actions=a, next_states=sn,
      tr_mask=tr_mask, va_mask=va_mask,
      s_tr=jnp.asarray(s_std_all[tr_mask]),
      a_tr=jnp.asarray(a_std_all[tr_mask]),
      y_tr=jnp.asarray(y_std_all[tr_mask]),
      s_va=jnp.asarray(s_std_all[va_mask]),
      a_va=jnp.asarray(a_std_all[va_mask]),
      y_va=jnp.asarray(y_std_all[va_mask]),
      acf_va=jnp.asarray(acf_va), acf_va_raw=acf_va_raw,
      lo_std=lo_std, hi_std=hi_std, train_action=train_action)


def _draw_train_counterfactual(neg, idx, rng):
  """One training step's ``a'`` for the negative minibatch at rows ``idx``.

  Returns STANDARDIZED actions, the frame the design matrix is in. Both
  sources guarantee ``a' != a`` in the sense the objective needs:

    * ``'uniform'`` draws from the (continuous) action box, where landing
      exactly on the stored torque has probability zero -- though a draw can
      of course land NEAR it, which is correct rather than a flaw: the
      off-diagonal has to join up continuously with the diagonal;
    * ``'shuffle'`` takes row ``(i + k) % n`` for a random ``k`` in
      ``[1, n-1]``, so the partner row is never the row itself.
  """
  if neg.train_action == 'shuffle':
    n = neg.n_tr
    step = rng.integers(1, max(n, 2), size=idx.shape[0])
    return neg.a_tr[(idx + step) % n, :]
  u = rng.random((idx.shape[0], neg.lo_std.shape[0]), dtype=np.float32)
  return jnp.asarray(neg.lo_std + u * (neg.hi_std - neg.lo_std))


def _fit_done(d, neg, done_states, done_outcome, config, seed, log_fn):
  """Fit the done head. Returns ``(DoneHead, stats)``.

  Positives: ``done_states``, the terminal states. Negatives: every state of
  the positive flat transitions plus, when given, the negative dataset's --
  both are ``s_t`` rows with ``t < length - 1``, so none of them is terminal.
  Holdout: the negatives inherit the dynamics fit's whole-episode split; the
  positives (one per episode) are split at random with the same fraction.

  The two classes are ~1:250, so every minibatch is drawn BALANCED, half
  positives and half negatives. The head is therefore calibrated to a 50/50
  prior, not the natural one; the report gives recall and the per-state
  false-positive rate at ``dyn_done_threshold`` on held-out rows, and the
  latter is the number that decides how often a rollout is cut short by
  mistake (over T steps, roughly ``1 - (1 - fpr) ** T``).

  Its own optimizer, parameters and RNG streams, so the dynamics and expert
  heads fit exactly as they do without it.
  """
  pos = np.asarray(done_states, np.float32)
  if pos.ndim != 2 or pos.shape[1] != d.state_dim:
    raise ValueError(f'done states must be [N, {d.state_dim}], got '
                     f'{pos.shape}')
  outcome = (None if done_outcome is None
             else np.asarray(done_outcome).astype(np.int64))
  rng = np.random.default_rng(seed)
  holdout = float(config.dyn_holdout_frac)
  n_pos = pos.shape[0]
  n_pos_va = min(max(int(round(holdout * n_pos)), 1 if holdout > 0 else 0),
                 n_pos - 1)
  perm = rng.permutation(n_pos)
  pos_va_idx, pos_tr_idx = perm[:n_pos_va], perm[n_pos_va:]

  neg_s = [d.states[d.tr_mask]]
  neg_s_va = [d.states[d.va_mask]]
  if neg is not None:
    neg_s.append(neg.states[neg.tr_mask])
    neg_s_va.append(neg.states[neg.va_mask])
  # A state labelled terminal must not also be a negative: with
  # dyn_done_dataset built from the LAST LIVE state of a failure episode, that
  # exact row is also a non-terminal state of the negative dataset.
  pos_keys = {r.tobytes() for r in pos}
  n_dropped = 0
  for lst in (neg_s, neg_s_va):
    for j, arr in enumerate(lst):
      keep = np.array([r.tobytes() not in pos_keys for r in arr], bool)
      n_dropped += int((~keep).sum())
      lst[j] = arr[keep]
  std = lambda x: (x - d.s_mean) / d.s_std
  p_tr = jnp.asarray(std(pos[pos_tr_idx]))
  p_va = pos[pos_va_idx]                 # raw: head.prob standardizes.
  q_tr = jnp.asarray(std(np.concatenate(neg_s)))
  q_va = np.concatenate(neg_s_va)

  sizes = tuple(int(h) for h in config.dyn_done_hidden_layer_sizes)
  net = build_done_network(sizes, d.activation)
  params = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, d.state_dim)))
  steps = int(config.dyn_done_train_steps or config.dyn_train_steps)
  opt, _ = build_optimizer(config, steps)
  opt_state = opt.init(params)
  half = max(1, int(config.dyn_batch_size) // 2)

  def _loss(params, sp, sq):
    lp, lq = net.apply(params, sp), net.apply(params, sq)
    # BCE with logits: -log sigmoid(lp) for positives, -log(1 - sigmoid(lq)).
    return 0.5 * (jnp.mean(jax.nn.softplus(-lp)) +
                  jnp.mean(jax.nn.softplus(lq)))

  def _update(params, opt_state, sp, sq):
    loss, grad = jax.value_and_grad(_loss)(params, sp, sq)
    updates, opt_state = opt.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

  update = jax.jit(_update) if config.jit else _update
  threshold = float(config.dyn_done_threshold)
  log_fn(f'  DONE HEAD: P(done|s) MLP {sizes} {d.state_dim}->1 on '
         f'{n_pos} terminal states ({pos_tr_idx.size} train / '
         f'{n_pos_va} val) vs {q_tr.shape[0]} + {q_va.shape[0]} non-terminal'
         + (f' ({n_dropped} dropped: identical to a terminal state)'
            if n_dropped else '') + '; '
         f'balanced batches of {2 * half}, {steps} steps, rollout threshold '
         f'{threshold:g}')
  head = DoneHead(params=None, state_dim=int(d.state_dim),
                  hidden_layer_sizes=sizes, state_mean=d.s_mean,
                  state_std=d.s_std, activation=d.activation,
                  threshold=threshold)
  log_every = max(1, int(config.dyn_log_every_steps))
  curve = []

  def _val(params):
    head.params = jax.tree_util.tree_map(np.asarray, params)
    pp = head.prob(p_va) if n_pos_va else np.zeros(0)
    pq = head.prob(q_va)
    return pp, pq

  for step in range(1, steps + 1):
    ip = rng.integers(0, p_tr.shape[0], size=half)
    iq = rng.integers(0, q_tr.shape[0], size=half)
    params, opt_state, loss = update(params, opt_state, p_tr[ip], q_tr[iq])
    if step % log_every == 0 or step == steps:
      pp, pq = _val(params)
      rec = float(np.mean(pp >= threshold)) if pp.size else float('nan')
      fpr = float(np.mean(pq >= threshold))
      curve.append({'step': step, 'train_bce': float(loss),
                    'val_recall': rec, 'val_fpr': fpr})
      log_fn(f'  [done {step:>7}/{steps}] bce={float(loss):.5f} | val '
             f'recall={rec:.4f} fpr={fpr:.2e}')

  pp, pq = _val(params)
  stats = {
      'done_hidden_layer_sizes': list(sizes),
      'done_steps': steps,
      'done_threshold': threshold,
      'done_n_pos_train': int(pos_tr_idx.size),
      'done_n_pos_val': int(n_pos_va),
      'done_n_neg_train': int(q_tr.shape[0]),
      'done_n_neg_val': int(q_va.shape[0]),
      'done_n_neg_dropped_as_terminal': n_dropped,
      'done_val_recall': float(np.mean(pp >= threshold)) if pp.size else None,
      'done_val_fpr': float(np.mean(pq >= threshold)),
      'done_curve': curve,
  }
  if outcome is not None and pp.size:
    o_va = outcome[pos_va_idx]
    for code, name in ((1, 'success'), (0, 'failure')):
      m = o_va == code
      if m.any():
        stats[f'done_val_recall_{name}'] = float(np.mean(pp[m] >= threshold))
        stats[f'done_n_pos_val_{name}'] = int(m.sum())
  return head, stats


def fit(states, actions, next_states, episode_ids, config, seed=0,
        log_fn=print, next_actions=None):
  """Fit ``f(s, a[, a']) -> s'`` on the transitions. Returns (model, stats).

  Args:
    states / actions / next_states: [N, obs_dim], [N, action_dim], [N, obs_dim]
      raw arrays, e.g. straight from ``TrajectoryBuffer.flat_transitions()``.
    episode_ids: [N] episode index per transition (drives the holdout split).
    config: a ``crl.config.Config``; reads only the ``dyn_*`` fields.
    seed: RNG seed for the split, the init and the minibatch stream. Kept
      SEPARATE from the CRL streams so fitting cannot perturb the agent run.
    next_actions: [N, action_dim] counterfactual actions ``a'`` (legacy
      argument name), used only when ``config.dyn_next_action_input``.
      Defaults to ``actions`` -- the factual diagonal the offline dataset
      holds.
  """
  t0 = time.time()
  d = _prepare(states, actions, next_states, episode_ids, config, seed,
               next_actions)
  states, actions, next_states = d.states, d.actions, d.next_states
  n, state_dim, action_dim = d.n, d.state_dim, d.action_dim
  rng, tr_mask, va_mask, n_val_eps = d.rng, d.tr_mask, d.va_mask, d.n_val_eps
  n_tr, n_va = d.n_tr, d.n_va
  s_mean, s_std, a_mean, a_std = d.s_mean, d.s_std, d.a_mean, d.a_std
  t_mean, t_std = d.t_mean, d.t_std
  x_tr, y_tr, x_va, y_va = d.x_tr, d.y_tr, d.x_va, d.y_va

  activation = d.activation
  sn = _sn_settings(config)
  net = build_network(state_dim, config.dyn_hidden_layer_sizes, activation,
                      **sn)
  key = jax.random.PRNGKey(seed)
  params = net.init(key, jnp.zeros((1, d.in_dim), jnp.float32))
  opt, _ = build_optimizer(config, int(config.dyn_train_steps))
  opt_state = opt.init(params)

  def _mse(params, xb, yb):
    return jnp.mean((net.apply(params, xb) - yb) ** 2)

  def _update(params, opt_state, xb, yb):
    loss, grad = jax.value_and_grad(_mse)(params, xb, yb)
    updates, opt_state = opt.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

  update = jax.jit(_update) if config.jit else _update
  eval_mse = jax.jit(_mse) if config.jit else _mse

  steps = int(config.dyn_train_steps)
  batch = min(int(config.dyn_batch_size), n_tr)
  log_every = max(1, int(config.dyn_log_every_steps))
  log_fn(f'TRANSITION MODEL: fitting '
         f'f(s,a{",a\'" if d.next_action_input else ""})->s\' on {n} '
         f'transitions '
         f'({n_tr} train / {n_va} val, {n_val_eps} val episodes), '
         f'MLP {tuple(config.dyn_hidden_layer_sizes)} '
         f'{d.in_dim}->{state_dim}, '
         f'target={"delta" if config.dyn_predict_delta else "absolute"}, '
         f'act={activation}, '
         + (f'spectral-norm all layers (coef {sn["sn_coef"]}, '
            f'{sn["sn_iters"]} power iters), ' if sn['spectral_norm'] else '')
         + f'{steps} steps @ batch {batch}, lr {config.dyn_learning_rate} '
           f'({config.dyn_lr_schedule})')

  curve = []
  train_loss = float('nan')
  for step in range(1, steps + 1):
    idx = rng.integers(0, n_tr, size=batch)
    params, opt_state, loss = update(params, opt_state, x_tr[idx, :],
                                     y_tr[idx, :])
    if step % log_every == 0 or step == steps:
      train_loss = float(loss)
      val_loss = float(eval_mse(params, x_va, y_va)) if n_va else float('nan')
      curve.append({'step': step, 'train_mse_norm': train_loss,
                    'val_mse_norm': val_loss})
      log_fn(f'  [dyn {step:>7}/{steps}] train_mse_norm={train_loss:.5f} '
             f'val_mse_norm={val_loss:.5f}', )

  model = TransitionModel(
      params=jax.tree_util.tree_map(np.asarray, params),
      state_dim=int(state_dim), action_dim=int(action_dim),
      hidden_layer_sizes=tuple(int(h) for h in config.dyn_hidden_layer_sizes),
      predict_delta=bool(config.dyn_predict_delta),
      state_mean=s_mean, state_std=s_std, action_mean=a_mean, action_std=a_std,
      target_mean=t_mean, target_std=t_std, activation=activation,
      spectral_norm=sn['spectral_norm'], spectral_norm_coef=sn['sn_coef'],
      spectral_norm_iters=sn['sn_iters'],
      next_action_input=d.next_action_input)

  stats = {
      'n_transitions': int(n), 'n_train': n_tr, 'n_val': n_va,
      'n_val_episodes': int(n_val_eps),
      'state_dim': int(state_dim), 'action_dim': int(action_dim),
      'hidden_layer_sizes': list(model.hidden_layer_sizes),
      'input_dim': int(d.in_dim),
      'next_action_input': bool(d.next_action_input),
      'predict_delta': model.predict_delta,
      'activation': activation,
      'spectral_norm': sn['spectral_norm'],
      'spectral_norm_coef': sn['sn_coef'],
      'spectral_norm_iters': sn['sn_iters'],
      'steps': steps, 'batch_size': batch,
      'learning_rate': float(config.dyn_learning_rate),
      'lr_schedule': str(config.dyn_lr_schedule),
      'seed': int(seed),
      'final_train_mse_norm': train_loss,
      'curve': curve,
  }
  if n_va:
    stats['final_val_mse_norm'] = float(eval_mse(params, x_va, y_va))
    pred = model.predict(states[va_mask], actions[va_mask])
    stats.update(_raw_error_stats(pred, next_states[va_mask], 'val'))
    # Trivial "nothing moves" predictor on the SAME split -- the number the
    # model has to beat before it is worth anything.
    stats.update(_raw_error_stats(states[va_mask], next_states[va_mask],
                                  'persistence'))
    ratio = stats['persistence_rmse_raw'] / max(stats['val_rmse_raw'], 1e-12)
    stats['improvement_over_persistence'] = float(ratio)
    if d.next_action_input:
      stats.update(action_input_sensitivity(model, states[va_mask],
                                            actions[va_mask], seed=seed))
  stats['wall_time_s'] = float(time.time() - t0)
  return model, stats


def fit_causal(states, actions, next_states, episode_ids, config, seed=0,
               log_fn=print, next_actions=None, neg_states=None,
               neg_actions=None, neg_next_states=None, neg_episode_ids=None,
               state_bounds=None, free_cells=None, done_states=None,
               done_outcome=None):
  """Fit the dynamics head and the blind expert policy. Returns (model, stats).

  One loop, one optimizer, one minibatch stream: at every step the same
  transitions supervise ``f(s, a) -> s'`` and the expert head, and the total
  loss is ``dyn_mse + dyn_policy_coef * policy_loss``. The heads are separate
  networks (no shared trunk), so the expert head cannot distort the dynamics
  fit through the representation -- only through the shared Adam state, which
  ``dyn_policy_coef=0`` removes entirely.

  THE OFF-DIAGONAL TERM. Passing ``neg_states / neg_actions /
  neg_next_states / neg_episode_ids`` -- a NEGATIVE dataset, i.e. failure
  episodes collected on the same benchmark -- adds a third term and turns the
  total loss into

      dyn_mse(diagonal, positive data)
        + dyn_off_diagonal_coef * dyn_mse(off-diagonal, negative data)
        + dyn_policy_coef * policy_loss.

  The two dynamics terms supervise DISJOINT parts of the same head's input:

    * the diagonal, as always, is the frozen positive dataset with ``a' = a``
      on every row -- unchanged, bit-for-bit, including its minibatch stream;
    * the off-diagonal draws its own minibatch of negative transitions
      ``(s, a, s')``, draws a counterfactual ``a' != a`` for each row (per
      ``config.dyn_off_diagonal_train_action``, REDRAWN every step) and
      regresses ``f(s, a, a') -> s'`` on it.

  That is what makes the third input block mean something. Until now every
  row the head ever saw had ``a' = a``, so the gradient with respect to the
  two action blocks was identical at every step and their DIFFERENCE never
  moved from its initialization (measured: 7e-7 over a 3k-step fit) -- the
  model's whole response to ``a' != a`` was random. With this term the
  difference is supervised, and the head becomes a two-branch object: play
  the expert's own action back at it and it returns the expert's next state,
  hand it an ``a'`` unlike ``a`` and it returns the next state the FAILURE
  data recorded.

  What the off-diagonal is NOT. ``a'`` is drawn independently of ``s'``, so
  the term fits ``E[s' | s, a]`` over the negative data marginalized over
  ``a'``; it is a branch selected by ``a'`` being unlike ``a``, not a
  per-action counterfactual response. Reading an off-diagonal prediction as
  "where ``a'`` would have taken the ant" is over-reading it. The artifact
  records which regime it was fitted in (``TransitionModel.off_diagonal_
  trained``) so a consumer does not have to guess.

  Without a negative dataset nothing above runs, the loss is the two-term one
  it always was, and the fit reproduces exactly.

  Two expert heads are available, and they are fitted by DIFFERENT objectives:

    * ``dyn_policy_head='mdn'`` (the default) fits a ``MixtureExpertPolicy``
      of ``dyn_policy_components`` branches by MAXIMUM LIKELIHOOD, and
      ``policy_loss`` is the mean negative log-likelihood of the stored
      actions under a mixture censored to the train-split action box. This is
      the head to use, because the conditional being fitted is the sighted
      teacher's action distribution marginalized over the hidden rockfall
      latent -- multi-modal by construction, and an MSE fit to a multi-modal
      conditional returns its MEAN, which at the band mouth is an average of
      "go straight" and "detour": a torque in neither branch.
    * ``dyn_policy_head='deterministic'`` fits the historical
      ``ExpertPolicy`` by raw-unit MSE, kept so the two can be compared and so
      every pre-mixture config refits unchanged.

  A consequence worth stating before any table is read: the two heads are NOT
  comparable on RMSE. A better mixture can have a worse point error than a
  deterministic regression, because the mean of a bimodal conditional is
  closer to both modes than either mode is to the other. ``policy_val_nll`` is
  the mixture's own headline number, and ``dyn_policy_components=1`` is its
  control -- a k=1 fit that matches k=3 says the multi-modality is not in the
  actions.

  Both heads run for the whole ``dyn_train_steps`` budget. On the v2.1 pilot
  the dynamics head is still improving at 20k while the deterministic policy
  head has plateaued since ~9k, but over that stretch its validation curve is
  flat rather than climbing, so keeping its best-step parameters instead of
  its last-step ones was measured at about 1% of RMSE -- less than the
  run-to-run spread noted below. Hence: no early stopping, no snapshot
  bookkeeping.

  With ``dyn_policy_coef=0`` the dynamics head reproduces ``fit``: the split,
  the standardization, the dynamics init key and the minibatch stream are
  identical by construction, and the two agree bit-for-bit in-process. That is
  a statement about construction, not about the GPU -- long fits on this
  machine have been seen to diverge slightly BETWEEN processes, so treat a
  cross-run difference in the last digits as expected rather than as a bug.

  Args:
    states / actions / next_states: [N, obs_dim], [N, action_dim], [N, obs_dim]
      raw arrays, e.g. straight from ``TrajectoryBuffer.flat_transitions()``.
      ``states`` is the learner state half only -- the rockfall latent the
      sighted teacher saw is NOT in it, which is the whole point of the
      policy head.
    episode_ids: [N] episode index per transition (drives the holdout split).
    config: a ``crl.config.Config``; reads only the ``dyn_*`` fields.
    seed: RNG seed for the split, both inits and the minibatch stream. Kept
      SEPARATE from the CRL streams so fitting cannot perturb the agent run.
    next_actions: [N, action_dim] counterfactual actions ``a'`` for the
      dynamics head (legacy argument name), used only when
      ``config.dyn_next_action_input``; defaults to ``actions`` (the factual
      diagonal ``a' = a`` the dataset holds). The expert head never sees it --
      it models ``a``, not the pair.
    neg_states / neg_actions / neg_next_states / neg_episode_ids: the NEGATIVE
      dataset, in exactly the layout of the four positive arrays: flat
      transitions in the same learner STATE space, episode ids driving their
      own whole-episode holdout. All four together or none; giving them needs
      ``config.dyn_next_action_input`` (there is no a' block to supervise
      otherwise). They are standardized with the POSITIVE train split's
      statistics and never touch the expert head, which models the sighted
      teacher and has no business learning from failures.
    state_bounds: ``(lo, hi)``, two ``[obs_dim]`` vectors giving the FEASIBLE
      BOX of the state, or None for an unbounded model. Nothing in the fit
      reads them -- the box is a property of the environment, not of the data,
      and ``crl.d4rl_ant.ant_state_bounds`` derives it from the live env --
      but they are stored on the returned artifact, and
      ``CausalTransitionModel.rollout`` clips every state it generates into
      them. See ``TransitionModel.clip_state``.
    free_cells: ``(centres, half)`` from ``maze_free_cells``, the maze's
      traversable region, or None. Like ``state_bounds`` it is a property of
      the environment rather than of the data and nothing in the fit reads
      it; it is stored on the artifact so that every consumer projects the xy
      of a generated state onto the same floor plan. It is what stops a
      generated chain from crossing an INTERIOR wall, which the box cannot.
    done_states: [M, obs_dim] TERMINAL states (``config.dyn_done_dataset``),
      or None. When given, a third head ``P(done | s)`` is fitted against the
      non-terminal states of the positive and negative data (see
      ``_fit_done``) and attached as ``model.done``; ``rollout`` then stops
      each row at the first state it flags. Fitted after, and independently
      of, the two heads above.
    done_outcome: [M] optional 1 = success / 0 = failure per done state, used
      only to split the held-out recall in the report.
  """
  t0 = time.time()
  d = _prepare(states, actions, next_states, episode_ids, config, seed,
               next_actions)
  states, actions, next_states = d.states, d.actions, d.next_states
  n, state_dim, action_dim = d.n, d.state_dim, d.action_dim
  rng, tr_mask, va_mask = d.rng, d.tr_mask, d.va_mask
  n_tr, n_va = d.n_tr, d.n_va
  x_tr, y_tr, x_va, y_va = d.x_tr, d.y_tr, d.x_va, d.y_va
  activation = d.activation

  # Policy inputs are the state block of the same standardized design matrix,
  # so the two heads see the identical rows in the identical order.
  s_tr, s_va = x_tr[:, :state_dim], x_va[:, :state_dim]
  head = str(getattr(config, 'dyn_policy_head', 'mdn')).lower()
  if head not in POLICY_HEADS:
    raise ValueError(f'dyn_policy_head={head!r}; expected one of '
                     f'{sorted(POLICY_HEADS)}')
  mdn = head == 'mdn'
  squash = bool(getattr(config, 'dyn_policy_squash', True))
  # Action box from the TRAIN split (val must stay unseen), with a floor so a
  # constant action dimension cannot produce a degenerate squash -- or, for
  # the mixture head, a censoring interval of zero width, where an observation
  # would sit on BOTH limits at once and the per-dimension likelihood would
  # stop being defined. The floor widens such a dimension symmetrically about
  # its own constant value.
  a_lo = np.min(actions[tr_mask], axis=0).astype(np.float32)
  a_hi = np.max(actions[tr_mask], axis=0).astype(np.float32)
  thin = (a_hi - a_lo) < np.float32(2e-3)
  mid = (a_hi + a_lo) / 2.0
  a_lo = np.where(thin, mid - 1e-3, a_lo).astype(np.float32)
  a_hi = np.where(thin, mid + 1e-3, a_hi).astype(np.float32)
  a_scale = ((a_hi - a_lo) / 2.0).astype(np.float32)
  a_bias = mid.astype(np.float32)
  # The box in STANDARDIZED coordinates -- the frame the mixture lives in, so
  # the censoring limits have to be in it too. Standardization is monotone per
  # dimension, so it commutes with the clip that produced the censoring.
  lo_std = jnp.asarray((a_lo - d.a_mean) / d.a_std)
  hi_std = jnp.asarray((a_hi - d.a_mean) / d.a_std)

  # --- the negative dataset, i.e. the off-diagonal's supervision -----------
  neg_given = [x is not None for x in (neg_states, neg_actions,
                                       neg_next_states, neg_episode_ids)]
  if any(neg_given) and not all(neg_given):
    raise ValueError('neg_states, neg_actions, neg_next_states and '
                     'neg_episode_ids must be given together (got '
                     f'{sum(neg_given)} of 4)')
  neg = None
  off_coef = float(getattr(config, 'dyn_off_diagonal_coef', 1.0))
  if all(neg_given):
    if not d.next_action_input:
      raise ValueError(
          'a negative dataset was given but dyn_next_action_input=False, so '
          "the network has no a' block and there is no off-diagonal to "
          'supervise. Set dyn_next_action_input=True or drop the negatives.')
    neg = _prepare_negative(neg_states, neg_actions, neg_next_states,
                            neg_episode_ids, d, a_lo, a_hi, config,
                            seed + 104_729)

  # The policy target. The mixture head and the linear deterministic head both
  # work in standardized action units, which is the FIRST action block of x
  # (with the pair input, x also carries a copy of it as a'); the squashed
  # deterministic head regresses RAW actions instead.
  a_cols = slice(state_dim, state_dim + action_dim)
  if mdn or not squash:
    p_tr, p_va = x_tr[:, a_cols], x_va[:, a_cols]
  else:
    p_tr = jnp.asarray(actions[tr_mask])
    p_va = jnp.asarray(actions[va_mask])

  policy_sizes = tuple(int(h) for h in config.dyn_policy_hidden_layer_sizes)
  policy_coef = float(config.dyn_policy_coef)
  k = int(getattr(config, 'dyn_policy_components', 3))
  ls_min = float(getattr(config, 'dyn_policy_log_sigma_min', -5.0))
  ls_max = float(getattr(config, 'dyn_policy_log_sigma_max', 2.0))
  if mdn:
    if k < 1:
      raise ValueError(f'dyn_policy_components must be at least 1, got {k}')
    if not ls_min < ls_max:
      raise ValueError(f'dyn_policy_log_sigma_min={ls_min} must be below '
                       f'dyn_policy_log_sigma_max={ls_max}')
  init_spread = float(getattr(config, 'dyn_policy_init_spread', 1.0))
  # Sample-time only: carried onto the fitted head so a config-only run gets
  # them, and mutable afterwards so a sweep needs no refit.
  sigma_scale = float(getattr(config, 'dyn_policy_sample_sigma_scale', 1.0))
  weight_temp = float(getattr(config, 'dyn_policy_weight_temperature', 1.0))
  if mdn:
    if sigma_scale <= 0.0:
      raise ValueError('dyn_policy_sample_sigma_scale must be positive, got '
                       f'{sigma_scale}')
    if weight_temp <= 0.0:
      raise ValueError('dyn_policy_weight_temperature must be positive, got '
                       f'{weight_temp}')

  sn = _sn_settings(config)
  dyn_net = build_network(state_dim, config.dyn_hidden_layer_sizes, activation,
                          **sn)
  pi_net = (build_mixture_policy_network(action_dim, k, policy_sizes,
                                         activation, **sn) if mdn
            else build_policy_network(action_dim, policy_sizes, activation,
                                      **sn))
  # PRNGKey(seed) for the dynamics head is what `fit` uses, so a causal fit and
  # a dynamics-only fit start from the same dynamics weights.
  params = {
      'dyn': dyn_net.init(
          jax.random.PRNGKey(seed),
          jnp.zeros((1, d.in_dim), jnp.float32)),
      'policy': pi_net.init(
          jax.random.PRNGKey(seed + 7919),
          jnp.zeros((1, state_dim), jnp.float32)),
  }
  if mdn and init_spread > 0 and k > 1:
    params['policy'] = _spread_mixture_means(
        params['policy'], k, action_dim, init_spread,
        jax.random.PRNGKey(seed + 21_701))
  opt, _ = build_optimizer(config, int(config.dyn_train_steps))
  opt_state = opt.init(params)

  scale_j, bias_j = jnp.asarray(a_scale), jnp.asarray(a_bias)

  def _policy_loss(params, sb, pb):
    """The expert head's loss on one minibatch: NLL for the mixture, MSE else.

    The two are on different scales and only one of them is a squared error,
    so ``dyn_policy_coef`` does not mean the same thing under both heads --
    the NLL of an 8-dim censored mixture runs to tens of nats while the MSE is
    O(1). Set the coefficient per head rather than carrying one over.
    """
    out = pi_net.apply(params['policy'], sb)
    if not mdn:
      pred = bias_j + scale_j * jnp.tanh(out) if squash else out
      return jnp.mean((pred - pb) ** 2)
    log_w, mu, ls = split_mixture(out, action_dim, k, ls_min, ls_max)
    return -jnp.mean(
        censored_mixture_log_prob(log_w, mu, ls, pb, lo_std, hi_std))

  def _losses(params, xb, yb, sb, pb):
    dyn = jnp.mean((dyn_net.apply(params['dyn'], xb) - yb) ** 2)
    return dyn, _policy_loss(params, sb, pb)

  def _off_loss(params, so, ao, ac, yo):
    """The off-diagonal MSE on a negative minibatch.

    The design matrix is assembled HERE rather than passed in because only
    two of its three blocks are fixed: a' is redrawn every step, so
    concatenating outside the jit would move a full copy of the negative
    states across the host boundary at every step for nothing.
    """
    xo = jnp.concatenate([so, ao, ac], axis=-1)
    return jnp.mean((dyn_net.apply(params['dyn'], xo) - yo) ** 2)

  def _total(params, xb, yb, sb, pb):
    dyn, pol = _losses(params, xb, yb, sb, pb)
    return dyn + policy_coef * pol, (dyn, pol)

  def _total_off(params, xb, yb, sb, pb, so, ao, ac, yo):
    dyn, pol = _losses(params, xb, yb, sb, pb)
    off = _off_loss(params, so, ao, ac, yo)
    return dyn + off_coef * off + policy_coef * pol, (dyn, pol, off)

  def _update(params, opt_state, xb, yb, sb, pb):
    (_, parts), grad = jax.value_and_grad(_total, has_aux=True)(
        params, xb, yb, sb, pb)
    updates, opt_state = opt.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, parts

  def _update_off(params, opt_state, xb, yb, sb, pb, so, ao, ac, yo):
    (_, parts), grad = jax.value_and_grad(_total_off, has_aux=True)(
        params, xb, yb, sb, pb, so, ao, ac, yo)
    updates, opt_state = opt.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, parts

  # The no-negatives path keeps its OWN closure, untouched, rather than
  # passing zeros through the wider one: that is what makes a fit without a
  # negative dataset reproduce bit-for-bit.
  _upd = _update if neg is None else _update_off
  update = jax.jit(_upd) if config.jit else _upd
  eval_losses = jax.jit(_losses) if config.jit else _losses
  eval_off = (None if neg is None
              else (jax.jit(_off_loss) if config.jit else _off_loss))

  steps = int(config.dyn_train_steps)
  batch = min(int(config.dyn_batch_size), n_tr)
  log_every = max(1, int(config.dyn_log_every_steps))
  pol_name = 'pi(a|s)' if mdn else 'pi(s)->a'
  sig = f'f(s,a{",a\'" if d.next_action_input else ""})->s\''
  log_fn(f'CAUSAL TRANSITION MODEL: fitting {sig} AND blind {pol_name} '
         f'on {n} transitions ({n_tr} train / {n_va} val, '
         f'{d.n_val_eps} val episodes)')
  log_fn(f'  dynamics: MLP {tuple(config.dyn_hidden_layer_sizes)} '
         f'{d.in_dim}->{state_dim}, '
         f'target={"delta" if config.dyn_predict_delta else "absolute"}'
         + (f", action pair (a' = a on every stored transition)"
            if d.next_action_input else ''))
  if neg is not None:
    log_fn(f"  OFF-DIAGONAL: f(s, a, a') -> s' on {neg.n} NEGATIVE "
           f'transitions from {neg.n_episodes} failure episodes '
           f'({neg.n_tr} train / {neg.n_va} val, {neg.n_val_eps} val '
           f"episodes); a' redrawn every step from "
           + ('the train-split action box (uniform)'
              if neg.train_action == 'uniform'
              else "another negative row's stored action (shuffle)")
           + f', weight {off_coef}')
  elif d.next_action_input:
    log_fn("  OFF-DIAGONAL: UNSUPERVISED (no negative dataset) -- a' != a "
           'queries on this fit return the initialization, not physics')
  if mdn:
    log_fn(f'  expert policy: MIXTURE MLP {policy_sizes} {state_dim}->'
           f'{k * (1 + 2 * action_dim)} = {k} x (w, mu[{action_dim}], '
           f'log_sigma[{action_dim}]), censored to the train action box, '
           f'log_sigma in [{ls_min}, {ls_max}], NLL weight {policy_coef}')
  else:
    log_fn(f'  expert policy: MLP {policy_sizes} {state_dim}->{action_dim}, '
           f'{"tanh-squashed logits" if squash else "linear on standardized"}, '
           f'MSE weight {policy_coef}')
  if sn['spectral_norm']:
    log_fn(f'  spectral norm on EVERY layer of both heads: '
           f'coef {sn["sn_coef"]}, {sn["sn_iters"]} power iterations')
  log_fn(f'  act={activation}, {steps} steps @ batch {batch}, '
         f'lr {config.dyn_learning_rate} ({config.dyn_lr_schedule})')

  pol_key = 'nll' if mdn else 'mse'
  curve = []
  train_dyn = train_pol = float('nan')
  train_off = float('nan')
  # The off-diagonal's own minibatch stream and a' draws, seeded apart from
  # everything above so the positive stream is exactly what it always was.
  off_rng = (None if neg is None
             else np.random.default_rng(seed + 1_299_709))
  off_batch = 0 if neg is None else min(int(config.dyn_batch_size), neg.n_tr)
  for step in range(1, steps + 1):
    idx = rng.integers(0, n_tr, size=batch)
    args = (x_tr[idx, :], y_tr[idx, :], s_tr[idx, :], p_tr[idx, :])
    if neg is not None:
      oidx = off_rng.integers(0, neg.n_tr, size=off_batch)
      acf = _draw_train_counterfactual(neg, oidx, off_rng)
      args += (neg.s_tr[oidx, :], neg.a_tr[oidx, :], acf, neg.y_tr[oidx, :])
    params, opt_state, parts = update(params, opt_state, *args)
    if step % log_every == 0 or step == steps:
      train_dyn, train_pol = float(parts[0]), float(parts[1])
      if n_va:
        v_dyn, v_pol = eval_losses(params, x_va, y_va, s_va, p_va)
        val_dyn, val_pol = float(v_dyn), float(v_pol)
      else:
        val_dyn = val_pol = float('nan')
      row = {'step': step, 'train_mse_norm': train_dyn,
             'val_mse_norm': val_dyn,
             f'train_policy_{pol_key}': train_pol,
             f'val_policy_{pol_key}': val_pol}
      line = (f'  [causal {step:>7}/{steps}] '
              f'dyn train={train_dyn:.5f} val={val_dyn:.5f} | '
              f'pi {pol_key} train={train_pol:.5f} val={val_pol:.5f}')
      if neg is not None:
        train_off = float(parts[2])
        val_off = (float(eval_off(params, neg.s_va, neg.a_va, neg.acf_va,
                                  neg.y_va)) if neg.n_va else float('nan'))
        row['train_offdiag_mse_norm'] = train_off
        row['val_offdiag_mse_norm'] = val_off
        line += f' | off train={train_off:.5f} val={val_off:.5f}'
      curve.append(row)
      log_fn(line)

  to_np = lambda t: jax.tree_util.tree_map(np.asarray, t)
  transition = TransitionModel(
      params=to_np(params['dyn']),
      state_dim=int(state_dim), action_dim=int(action_dim),
      hidden_layer_sizes=tuple(int(h) for h in config.dyn_hidden_layer_sizes),
      predict_delta=bool(config.dyn_predict_delta),
      state_mean=d.s_mean, state_std=d.s_std,
      action_mean=d.a_mean, action_std=d.a_std,
      target_mean=d.t_mean, target_std=d.t_std, activation=activation,
      spectral_norm=sn['spectral_norm'], spectral_norm_coef=sn['sn_coef'],
      spectral_norm_iters=sn['sn_iters'],
      next_action_input=d.next_action_input,
      off_diagonal_trained=neg is not None)
  # The feasible box is not fitted -- it is a property of the ENVIRONMENT the
  # data came from, handed in by the caller (crl/ETT_train.py reads it off the
  # live eval env). It is attached to the artifact so that a pickle carries
  # its own box and every later consumer of `rollout` clips the same way this
  # run did.
  transition.set_state_bounds(state_bounds)
  transition.set_free_cells(free_cells)
  if mdn:
    policy = MixtureExpertPolicy(
        params=to_np(params['policy']),
        state_dim=int(state_dim), action_dim=int(action_dim),
        n_components=k, hidden_layer_sizes=policy_sizes,
        state_mean=d.s_mean, state_std=d.s_std,
        action_mean=d.a_mean, action_std=d.a_std,
        action_lo=a_lo, action_hi=a_hi,
        log_sigma_min=ls_min, log_sigma_max=ls_max,
        activation=activation,
        spectral_norm=sn['spectral_norm'], spectral_norm_coef=sn['sn_coef'],
        spectral_norm_iters=sn['sn_iters'],
        sample_sigma_scale=sigma_scale, weight_temperature=weight_temp)
  else:
    policy = ExpertPolicy(
        params=to_np(params['policy']),
        state_dim=int(state_dim), action_dim=int(action_dim),
        hidden_layer_sizes=policy_sizes,
        state_mean=d.s_mean, state_std=d.s_std,
        action_mean=d.a_mean, action_std=d.a_std,
        action_scale=a_scale, action_bias=a_bias,
        squash=squash, activation=activation,
        spectral_norm=sn['spectral_norm'], spectral_norm_coef=sn['sn_coef'],
        spectral_norm_iters=sn['sn_iters'])
  done_head, done_stats = None, {}
  if done_states is not None:
    done_head, done_stats = _fit_done(d, neg, done_states, done_outcome,
                                      config, seed + 2_750_159, log_fn)
  model = CausalTransitionModel(transition=transition, policy=policy,
                                done=done_head)

  stats = {
      'n_transitions': int(n), 'n_train': n_tr, 'n_val': n_va,
      'n_val_episodes': int(d.n_val_eps),
      'state_dim': int(state_dim), 'action_dim': int(action_dim),
      'hidden_layer_sizes': list(transition.hidden_layer_sizes),
      'input_dim': int(d.in_dim),
      'next_action_input': bool(d.next_action_input),
      'off_diagonal_trained': neg is not None,
      'predict_delta': transition.predict_delta,
      'activation': activation,
      'spectral_norm': sn['spectral_norm'],
      'spectral_norm_coef': sn['sn_coef'],
      'spectral_norm_iters': sn['sn_iters'],
      'steps': steps, 'batch_size': batch,
      'learning_rate': float(config.dyn_learning_rate),
      'lr_schedule': str(config.dyn_lr_schedule),
      'seed': int(seed),
      'final_train_mse_norm': train_dyn,
      'policy_head': head,
      'policy_hidden_layer_sizes': list(policy_sizes),
      'policy_coef': policy_coef,
      f'final_train_policy_{pol_key}': train_pol,
      'action_lo': [float(v) for v in a_lo],
      'action_hi': [float(v) for v in a_hi],
      'state_clip': bool(transition.clips_states),
      'state_lo': (None if not transition.clips_states
                   else [float(v) for v in transition.state_lo]),
      'state_hi': (None if not transition.clips_states
                   else [float(v) for v in transition.state_hi]),
      'curve': curve,
  }
  stats.update(done_stats)
  if neg is not None:
    stats.update({
        'off_diagonal_coef': off_coef,
        'off_diagonal_train_action': neg.train_action,
        'n_negative_transitions': neg.n,
        'n_negative_episodes': neg.n_episodes,
        'n_negative_train': neg.n_tr,
        'n_negative_val': neg.n_va,
        'n_negative_val_episodes': neg.n_val_eps,
        'negative_batch_size': off_batch,
        'final_train_offdiag_mse_norm': train_off,
    })
  if mdn:
    stats.update({'policy_components': k,
                  'policy_log_sigma_min': ls_min,
                  'policy_log_sigma_max': ls_max,
                  'policy_init_spread': init_spread,
                  'policy_sample_sigma_scale': sigma_scale,
                  'policy_weight_temperature': weight_temp})
  else:
    stats['policy_squash'] = squash
  if n_va:
    v_dyn, v_pol = eval_losses(params, x_va, y_va, s_va, p_va)
    stats['final_val_mse_norm'] = float(v_dyn)
    stats[f'final_val_policy_{pol_key}'] = float(v_pol)
    pred = transition.predict(states[va_mask], actions[va_mask])
    stats.update(_raw_error_stats(pred, next_states[va_mask], 'val'))
    # Trivial "nothing moves" predictor on the SAME split -- the number the
    # dynamics head has to beat before it is worth anything.
    stats.update(_raw_error_stats(states[va_mask], next_states[va_mask],
                                  'persistence'))
    ratio = stats['persistence_rmse_raw'] / max(stats['val_rmse_raw'], 1e-12)
    stats['improvement_over_persistence'] = float(ratio)
    # Same idea for the policy: beat "always play the train-split mean action".
    a_val = actions[va_mask]
    point = (policy.mode_action(states[va_mask]) if mdn
             else policy.act(states[va_mask]))
    stats.update(_action_error_stats(point, a_val, 'policy_val'))
    stats.update(_action_error_stats(
        np.broadcast_to(d.a_mean, a_val.shape), a_val, 'policy_mean_baseline'))
    ratio = (stats['policy_mean_baseline_rmse_raw'] /
             max(stats['policy_val_rmse_raw'], 1e-12))
    stats['policy_improvement_over_mean'] = float(ratio)
    if d.next_action_input:
      stats.update(action_input_sensitivity(transition, states[va_mask],
                                            actions[va_mask], seed=seed))
  if neg is not None and neg.n_va:
    # The off-diagonal in RAW units, on the negative holdout, against the
    # same two references the diagonal is reported against: persistence
    # (what "nothing moves" would have scored on these rows) and -- the one
    # that matters here -- what the DIAGONAL predicts for the same (s, a).
    # If the two agree, the head has one branch, not two, and the negative
    # data changed nothing a consumer can use.
    ns, na = neg.states[neg.va_mask], neg.actions[neg.va_mask]
    nsn = neg.next_states[neg.va_mask]
    stats['final_val_offdiag_mse_norm'] = float(
        eval_off(params, neg.s_va, neg.a_va, neg.acf_va, neg.y_va))
    off_pred = transition.predict(ns, na, neg.acf_va_raw)
    diag_pred = transition.predict(ns, na)
    stats.update(_raw_error_stats(off_pred, nsn, 'negative_val'))
    stats.update(_raw_error_stats(ns, nsn, 'negative_persistence'))
    stats['negative_improvement_over_persistence'] = float(
        stats['negative_persistence_rmse_raw'] /
        max(stats['negative_val_rmse_raw'], 1e-12))
    # The diagonal on the SAME negative rows: how much worse the expert
    # branch is at predicting a failure step, i.e. how far apart the two
    # branches actually are.
    stats.update(_raw_error_stats(diag_pred, nsn, 'negative_diag'))
    stats['off_vs_diag_rmse_raw'] = float(
        np.sqrt(np.mean((off_pred - diag_pred) ** 2)))
  if mdn:
    # The mixture diagnostics on BOTH splits: the held-out ones say whether
    # the branches generalize, the train ones how far the fit got. These are
    # the numbers that select k -- the RMSEs above cannot, since a better
    # mixture can score worse on them.
    for name, m in (('mdn_train', tr_mask), ('mdn_val', va_mask)):
      if int(m.sum()) == 0:
        continue
      stats.update(_mixture_stats(policy, states[m], actions[m], name))
  stats['wall_time_s'] = float(time.time() - t0)
  return model, stats


def _fields_of(model):
  return {k: v for k, v in dataclasses.asdict(model).items()
          if not k.startswith('_')}


def save(model, path, stats=None):
  """Pickle a ``TransitionModel`` or ``CausalTransitionModel`` atomically.

  A causal payload is tagged with ``kind='causal'`` and stores the two heads
  under separate keys, so ``load`` can tell the two artifacts apart and old
  dynamics-only pickles keep loading untouched. ``policy_head`` records which
  expert head the payload carries; a pickle written before that key existed is
  a deterministic one.

  Nothing here pickles a class -- every head goes in as a dict of its fields
  and numpy arrays -- which is why removing the discriminator head could not
  break any existing checkpoint.
  """
  if isinstance(model, CausalTransitionModel):
    payload = {'kind': 'causal',
               'policy_head': ('mdn' if isinstance(model.policy,
                                                   MixtureExpertPolicy)
                               else 'deterministic'),
               'transition': _fields_of(model.transition),
               'policy': _fields_of(model.policy)}
    if model.done is not None:
      payload['done'] = _fields_of(model.done)
  else:
    payload = _fields_of(model)
  payload['stats'] = stats
  tmp = path + '.tmp'
  with open(tmp, 'wb') as f:
    pickle.dump(payload, f)
  os.replace(tmp, path)


def load(path, log_fn=print):
  """Restore whatever ``save`` wrote: a ``TransitionModel`` or a causal one.

  A checkpoint written before the discriminator head was removed still loads:
  its ``discriminator`` payload is a plain dict of fields (no class reference),
  so it is DROPPED here with a note rather than reconstructed. The expert head
  is rebuilt as whatever ``policy_head`` says, defaulting to the deterministic
  one for pickles older than that key.
  """
  with open(path, 'rb') as f:
    payload = pickle.load(f)
  payload.pop('stats', None)
  if payload.pop('kind', None) != 'causal':
    return TransitionModel(**payload)
  if payload.pop('discriminator', None) is not None and log_fn:
    log_fn(f'  note: {path} carries a discriminator head, which this version '
           'no longer implements -- dropped. Its role at sample time (the '
           'coin choosing the dynamics head\'s diagonal or off-diagonal) is '
           'now the mixture expert head\'s sum_i w_i(s)**2.')
  head = payload.pop('policy_head', 'deterministic')
  if head == 'mdn':
    policy = MixtureExpertPolicy(**payload['policy'])
  elif head == 'deterministic':
    policy = ExpertPolicy(**payload['policy'])
  else:
    raise ValueError(f'{path} declares policy_head={head!r}, which this '
                     f'version cannot build (expected one of '
                     f'{sorted(POLICY_HEADS)})')
  done = payload.get('done')
  return CausalTransitionModel(
      transition=TransitionModel(**payload['transition']), policy=policy,
      done=None if done is None else DoneHead(**done))


def write_report(ckpt_dir, stats, filename='transition_model_metrics.json'):
  """Write the fit report next to the checkpoints (JSON, human-readable)."""
  if not ckpt_dir:
    return None
  os.makedirs(ckpt_dir, exist_ok=True)
  path = os.path.join(ckpt_dir, filename)
  with open(path, 'w') as f:
    json.dump(stats, f, indent=2)
  return path


def summary_line(stats):
  """One-line human summary of a fit (printed after training).

  Grows extra lines when the stats came from ``fit_causal``: the expert
  policy next to the same mean-action baseline it has to beat, and -- for a
  mixture head -- the numbers that say whether the branches separated, since
  its RMSE deliberately does not.
  """
  if 'val_rmse_raw' not in stats:
    line = (f'  train_mse_norm={stats["final_train_mse_norm"]:.5f} '
            '(no holdout)')
    for key, label in (('final_train_policy_nll', 'policy train_nll'),
                       ('final_train_policy_mse', 'policy train_mse')):
      if key in stats:
        line += f' | {label}={stats[key]:.5f}'
    return line
  line = (f'  val: mse_norm={stats["final_val_mse_norm"]:.5f} '
          f'rmse_raw={stats["val_rmse_raw"]:.5f} '
          f'rmse_xy={stats["val_rmse_xy"]:.5f} '
          f'expl_var={stats["val_explained_variance"]:.4f} | '
          f'persistence rmse_raw={stats["persistence_rmse_raw"]:.5f} '
          f'(model is {stats["improvement_over_persistence"]:.1f}x better) | '
          f'{stats["wall_time_s"]:.0f}s')
  if 'action_next_a_share' in stats:
    trained = bool(stats.get('off_diagonal_trained'))
    line += (f"\n  action pair: |dS/da|={stats['action_jac_a']:.4f} "
             f"|dS/da'|={stats['action_jac_next_a']:.4f} "
             f"(a' share {stats['action_next_a_share']:.2f}, "
             f"block cos {stats['action_block_cosine']:+.2f}) | "
             f"off-diagonal swap moves the prediction "
             f"{stats['offdiag_shift_rmse']:.4f} vs a true step of "
             f"{stats['true_step_rmse']:.4f}"
             + ('' if trained else
                " -- UNIDENTIFIED, the fit only ever saw a' = a"))
  if 'negative_val_rmse_raw' in stats:
    line += (f"\n  off-diagonal val ({stats['n_negative_val']} negative "
             f"transitions, a' {stats['off_diagonal_train_action']}): "
             f"mse_norm={stats['final_val_offdiag_mse_norm']:.5f} "
             f"rmse_raw={stats['negative_val_rmse_raw']:.5f} "
             f"xy={stats['negative_val_rmse_xy']:.5f} | persistence "
             f"{stats['negative_persistence_rmse_raw']:.5f} "
             f"({stats['negative_improvement_over_persistence']:.1f}x better)"
             f"\n  off-diagonal val: the DIAGONAL scores "
             f"{stats['negative_diag_rmse_raw']:.5f} on the same rows and the "
             f"two branches differ by {stats['off_vs_diag_rmse_raw']:.5f} -- "
             f"if that is ~0 the negative data bought nothing")
  if 'done_val_fpr' in stats:
    rec = stats['done_val_recall']
    line += (f"\n  done head val (threshold {stats['done_threshold']:g}): "
             f"recall {'--' if rec is None else f'{rec:.4f}'}"
             + ''.join(f" ({name} {stats[f'done_val_recall_{name}']:.4f})"
                       for name in ('success', 'failure')
                       if f'done_val_recall_{name}' in stats)
             + f" | false-positive rate on non-terminal states "
               f"{stats['done_val_fpr']:.2e}")
  if 'policy_val_rmse_raw' in stats:
    what = 'mode' if stats.get('policy_head') == 'mdn' else 'pi(s)'
    line += (f'\n  expert policy val ({what}): rmse_raw='
             f'{stats["policy_val_rmse_raw"]:.5f} '
             f'expl_var={stats["policy_val_explained_variance"]:.4f} | '
             f'mean-action rmse_raw='
             f'{stats["policy_mean_baseline_rmse_raw"]:.5f} '
             f'(policy is {stats["policy_improvement_over_mean"]:.2f}x better)')
  if 'mdn_val_nll' in stats:
    k = stats.get('policy_components')
    w = ' '.join(f'{x:.3f}' for x in stats['mdn_val_weight_mean'])
    r = ' '.join(f'{x:.3f}' for x in stats['mdn_val_resp_frac'])
    line += (f'\n  mixture val (k={k}): NLL={stats["mdn_val_nll"]:.4f} '
             f'(train {stats["mdn_train_nll"]:.4f}) | mean weights [{w}] '
             f'| rows claimed [{r}]'
             f'\n  mixture val: branch entropy '
             f'{stats["mdn_val_branch_entropy"]:.4f} of a possible '
             f'{stats["mdn_val_branch_entropy_max"]:.4f} nats; the sampler '
             f'will take the FACTUAL diagonal '
             f'{100 * stats["mdn_val_diag_prob_mean"]:.1f}% of the time '
             f'(mode rmse {stats["mdn_val_mode_rmse_raw"]:.5f} vs mixture-mean '
             f'{stats["mdn_val_mean_rmse_raw"]:.5f}; '
             f'{100 * stats["mdn_val_censored_frac"]:.1f}% of action '
             f'coordinates are censored at a box limit)')
    ss = stats.get('policy_sample_sigma_scale', 1.0)
    wt = stats.get('policy_weight_temperature', 1.0)
    if ss != 1.0 or wt != 1.0:
      # Only printed when a knob is off its identity, so an untuned fit's log
      # is unchanged and a tuned one says loudly what the sampler is doing.
      line += (f'\n  mixture SAMPLING: sigma x{ss:g}, weight temperature '
               f'{wt:g} -> branch entropy '
               f'{stats["mdn_val_sample_branch_entropy"]:.4f} nats at draw '
               f'time (fit {stats["mdn_val_branch_entropy"]:.4f}). The fit is '
               'unchanged; the NLL above is the untempered head.')
  return line
