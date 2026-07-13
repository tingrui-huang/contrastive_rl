"""Contrastive RL networks (Acme-free port of ``contrastive/networks.py``).

The critic and its representation function are copied essentially verbatim from
the original (two MLP encoders whose inner product is the goal-conditioned value
function). The only real change is dropping Acme:

  * ``networks_lib.FeedForwardNetwork``  -> the tiny ``FeedForward`` namedtuple.
  * ``networks_lib.NormalTanhDistribution`` (tfp-backed) -> a self-contained
    tanh-squashed diagonal Gaussian (``TanhNormalParams`` + free functions),
    so we depend only on jax + haiku, no tfp / distrax.
  * ``networks_lib.AtariTorso`` -> a small conv torso reimplemented here (only
    used when ``use_image_obs=True``).
"""
import dataclasses
from typing import Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Tanh-squashed diagonal Gaussian policy (replacement for acme's
# NormalTanhDistribution + tfd.Independent(TanhTransformedDistribution)).
# ---------------------------------------------------------------------------
class TanhNormalParams(NamedTuple):
  loc: jnp.ndarray
  scale: jnp.ndarray


def tanh_normal_sample(params: TanhNormalParams, key) -> jnp.ndarray:
  """Reparameterized sample of a tanh-squashed Gaussian."""
  u = params.loc + params.scale * jax.random.normal(key, params.loc.shape)
  return jnp.tanh(u)


def tanh_normal_mode(params: TanhNormalParams) -> jnp.ndarray:
  """Deterministic action (used at eval): tanh of the Gaussian mean."""
  return jnp.tanh(params.loc)


def tanh_normal_log_prob(params: TanhNormalParams, actions: jnp.ndarray,
                         eps: float = 1e-6) -> jnp.ndarray:
  """log pi(a|s), summed over action dims, with the tanh change-of-variables.

  Matches distrax/acme's TanhTransformedDistribution wrapped in
  Independent(..., 1): Gaussian log-prob of the pre-tanh value minus the
  log-det of the tanh Jacobian, summed over the action dimension.
  """
  y = jnp.clip(actions, -1.0 + eps, 1.0 - eps)
  x = jnp.arctanh(y)  # pre-tanh value.
  # Diagonal Gaussian log-prob of x.
  log_unnormalized = -0.5 * jnp.square((x - params.loc) / params.scale)
  log_normalization = 0.5 * jnp.log(2.0 * np.pi) + jnp.log(params.scale)
  normal_log_prob = log_unnormalized - log_normalization
  # Stable log(1 - tanh(x)^2) = 2*(log2 - x - softplus(-2x)).
  log_det = 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))
  return jnp.sum(normal_log_prob - log_det, axis=-1)


# ---------------------------------------------------------------------------
# Small conv torso for image observations (replacement for acme AtariTorso).
# ---------------------------------------------------------------------------
class _AtariTorso(hk.Module):
  """DQN-style conv torso: (B, 64, 64, C) -> (B, features)."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)

  def __call__(self, x):
    conv = hk.Sequential([
        hk.Conv2D(32, kernel_shape=8, stride=4), jax.nn.relu,
        hk.Conv2D(64, kernel_shape=4, stride=2), jax.nn.relu,
        hk.Conv2D(64, kernel_shape=3, stride=1), jax.nn.relu,
    ])
    return hk.Flatten()(conv(x))


class _LayerNormMLP(hk.Module):
  """MLP with LayerNorm before each ReLU (Linear -> LayerNorm -> ReLU per hidden
  layer); the final layer is a plain Linear unless ``activate_final``. Being a
  hk.Module, twin-critic reuse auto-numbers the second instance (sa_encoder,
  sa_encoder_1), exactly like hk.nets.MLP -- so the two critic heads stay
  independent. Used only when use_layer_norm=True (Stabilizing-Contrastive-RL)."""

  def __init__(self, sizes, w_init, activate_final, name=None):
    super().__init__(name=name)
    self._sizes = list(sizes)
    self._w_init = w_init
    self._activate_final = activate_final

  def __call__(self, x):
    n = len(self._sizes)
    for i, w in enumerate(self._sizes):
      x = hk.Linear(w, w_init=self._w_init, name=f'linear_{i}')(x)
      if i < n - 1 or self._activate_final:
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                         name=f'ln_{i}')(x)
        x = jax.nn.relu(x)
    return x


class FeedForward(NamedTuple):
  """Minimal replacement for acme's FeedForwardNetwork."""
  init: Callable
  apply: Callable


@dataclasses.dataclass
class ContrastiveNetworks:
  """Pure functions for the contrastive RL agent."""
  policy_network: FeedForward
  q_network: FeedForward
  # policy-distribution helpers (operate on TanhNormalParams):
  log_prob: Callable
  sample: Callable
  sample_eval: Callable


def make_networks(
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    repr_dim: int = 64,
    repr_norm: bool = False,
    repr_norm_temp: bool = True,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    actor_min_std: float = 1e-6,
    twin_q: bool = False,
    use_image_obs: bool = False,
    use_layer_norm: bool = False,
) -> ContrastiveNetworks:
  """Creates the contrastive RL networks.

  Args:
    obs_dim: size of the STATE part of the observation.
    goal_dim: size of the GOAL part of the observation.
    action_dim: number of action dimensions.
    (remaining args mirror the original make_networks).
  The full observation fed to the networks is ``concat([state, goal])`` with
  width ``obs_dim + goal_dim``.
  """
  full_obs_dim = obs_dim + goal_dim

  def _unflatten_obs(obs):
    state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
    goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    return state, goal

  # ---- Critic representation (verbatim from the original) ----
  def _repr_fn(obs, action, hidden=None):
    if hidden is None:
      if use_image_obs:
        state, goal = _unflatten_obs(obs)
        img_encoder = _AtariTorso()
        state = img_encoder(state)
        goal = img_encoder(goal)
      else:
        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]
    else:
      state, goal = hidden

    enc_w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    enc_sizes = list(hidden_layer_sizes) + [repr_dim]
    sa_in = jnp.concatenate([state, action], axis=-1)
    if use_layer_norm:
      sa_repr = _LayerNormMLP(enc_sizes, enc_w_init, False,
                              name='sa_encoder')(sa_in)
      g_repr = _LayerNormMLP(enc_sizes, enc_w_init, False,
                             name='g_encoder')(goal)
    else:
      sa_repr = hk.nets.MLP(enc_sizes, w_init=enc_w_init,
                            activation=jax.nn.relu, name='sa_encoder')(sa_in)
      g_repr = hk.nets.MLP(enc_sizes, w_init=enc_w_init,
                           activation=jax.nn.relu, name='g_encoder')(goal)

    if repr_norm:
      sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
      g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)
      if repr_norm_temp:
        log_scale = hk.get_parameter('repr_log_scale', [], dtype=sa_repr.dtype,
                                     init=jnp.zeros)
        sa_repr = sa_repr / jnp.exp(log_scale)
    return sa_repr, g_repr, (state, goal)

  def _combine_repr(sa_repr, g_repr):
    return jnp.einsum('ik,jk->ij', sa_repr, g_repr)

  def _critic_fn(obs, action):
    sa_repr, g_repr, hidden = _repr_fn(obs, action)
    outer = _combine_repr(sa_repr, g_repr)
    if twin_q:
      sa_repr2, g_repr2, _ = _repr_fn(obs, action, hidden=hidden)
      outer2 = _combine_repr(sa_repr2, g_repr2)
      outer = jnp.stack([outer, outer2], axis=-1)  # [B, B, 2]
    return outer

  # ---- Actor (returns TanhNormalParams instead of a tfp distribution) ----
  def _actor_fn(obs):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      obs = jnp.concatenate([state, goal], axis=-1)
      obs = _AtariTorso()(obs)
    act_w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform')
    if use_layer_norm:
      h = _LayerNormMLP(list(hidden_layer_sizes), act_w_init, True,
                        name='actor_torso')(obs)
    else:
      h = hk.nets.MLP(
          list(hidden_layer_sizes), w_init=act_w_init,
          activation=jax.nn.relu, activate_final=True)(obs)
    loc = hk.Linear(
        action_dim,
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
        name='loc')(h)
    scale = hk.Linear(
        action_dim,
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
        name='scale')(h)
    scale = jax.nn.softplus(scale) + actor_min_std
    return TanhNormalParams(loc=loc, scale=scale)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))

  dummy_obs = jnp.zeros((1, full_obs_dim), dtype=jnp.float32)
  dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

  return ContrastiveNetworks(
      policy_network=FeedForward(
          init=lambda key: policy.init(key, dummy_obs),
          apply=policy.apply),
      q_network=FeedForward(
          init=lambda key: critic.init(key, dummy_obs, dummy_action),
          apply=critic.apply),
      log_prob=tanh_normal_log_prob,
      sample=tanh_normal_sample,
      sample_eval=lambda params, key: tanh_normal_mode(params),
  )
