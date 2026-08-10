"""Stage 2: conditional flow-matching model of the offline behavior policy.

Learns ``mu_omega(a | c)`` where the decision context is

    c = concat(s, g_cmd)          # [B, context_dim]

with ``g_cmd`` the PRE-ACTION commanded task goal stored in the frozen dataset
(see the Stage-2 goal-provenance audit: it is sampled at ``env.reset()`` by the
d4rl goal_sampler, held constant for the episode, and is never a hindsight or
future-achieved variable). This is NOT contrastive RL's future-goal relabeling;
nothing here touches ``crl/replay.py``.

Objective (standard conditional flow matching, linear/rectified path):

    x1 = observed offline action
    x0 ~ N(0, I)
    t  ~ U(0, 1)
    xt = (1 - t) * x0 + t * x1
    L  = E || v_omega(c, xt, t) - (x1 - x0) ||^2

Sampling integrates the learned velocity field with explicit Euler steps from
t=0 to t=1 starting at z ~ N(0, I), clipping to the environment action box
after EVERY step -- the official CFQL sampler's behavior (flow_steps = 10).
That clip belongs to numerical integration, NOT to the training objective: the
loss below is unchanged, the network output is never squashed, and the stored
offline actions are never modified. ``sample_actions_raw`` keeps the unclipped
path available for diagnostics so the effect stays measurable.

The model generates ACTIONS ONLY -- never states or goals.

Conventions follow the repo: haiku + optax + jax, no Flax, no new dependency.
This module contains NO discriminator, no behavior-vs-target classifier, no
causal weighting and no worst-case branch -- those are later stages.
"""
import dataclasses
from typing import Callable, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

#: Environment action box for every conforming dataset in this repo (the ant's
#: actuator range). Used ONLY for diagnostics/clipping, never during training.
ACTION_BOX = (-1.0, 1.0)


@dataclasses.dataclass
class FlowConfig:
  """Velocity-network hyperparameters."""

  context_dim: int
  action_dim: int
  #: Velocity MLP width/depth. (256, 256, 256) = the repo's (256, 256) critic
  #: torso plus one layer, a conservative default for an 8-D action density.
  hidden_sizes: Tuple[int, ...] = (256, 256, 256)
  #: Sinusoidal time-embedding width. 0 => feed the raw scalar t instead.
  time_features: int = 32
  #: Highest Fourier frequency for the time embedding.
  time_max_freq: float = 1000.0
  #: LayerNorm after each hidden layer. Off by default (repo default).
  use_layer_norm: bool = False

  def asdict(self):
    d = dataclasses.asdict(self)
    d['hidden_sizes'] = list(self.hidden_sizes)
    return d


class FlowNetwork(NamedTuple):
  """Pure-function pair, mirroring crl.networks' FeedForward namedtuple."""

  init: Callable
  apply: Callable      # apply(params, context, x_t, t) -> velocity [B, A]


def time_embedding(t, num_features, max_freq=1000.0):
  """Sinusoidal Fourier features of t in [0, 1]. t: [B, 1] -> [B, num_features].

  ``num_features == 0`` returns the raw scalar time, so the ablation is a config
  change rather than a code change."""
  if num_features <= 0:
    return t
  half = num_features // 2
  freqs = jnp.exp(jnp.linspace(0.0, jnp.log(max_freq), half))       # [half]
  ang = 2.0 * jnp.pi * t * freqs[None, :]                           # [B, half]
  return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def make_flow_network(config: FlowConfig) -> FlowNetwork:
  """Build v_omega(c, x_t, t) -> [B, action_dim] as a haiku pure function."""

  def _forward(context, x_t, t):
    # t is accepted as [B] or [B, 1]; normalize to [B, 1].
    t = jnp.reshape(t, (t.shape[0], 1))
    h = jnp.concatenate(
        [context, x_t,
         time_embedding(t, config.time_features, config.time_max_freq)],
        axis=-1)
    for width in config.hidden_sizes:
      h = hk.Linear(width)(h)
      if config.use_layer_norm:
        h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)
      h = jax.nn.relu(h)
    # Final layer zero-initialized: training starts from the zero velocity
    # field, i.e. an identity-ish flow, which is a stable place to begin.
    return hk.Linear(config.action_dim, w_init=jnp.zeros,
                     b_init=jnp.zeros)(h)

  transformed = hk.without_apply_rng(hk.transform(_forward))
  return FlowNetwork(init=transformed.init, apply=transformed.apply)


# --------------------------------------------------------------------------- #
# Training objective
# --------------------------------------------------------------------------- #
def flow_matching_loss(apply, params, context, action, key):
  """L = mean_over_batch || v(c, xt, t) - (x1 - x0) ||^2 (summed over dims).

  Args:
    apply: FlowNetwork.apply
    params: haiku params
    context: [B, context_dim]   the decision context c = (s, g_cmd)
    action:  [B, action_dim]    x1, the observed offline action
    key:     PRNGKey for (x0, t)
  Returns:
    (scalar loss, aux dict)
  """
  key_x0, key_t = jax.random.split(key)
  x1 = action
  x0 = jax.random.normal(key_x0, x1.shape)
  t = jax.random.uniform(key_t, (x1.shape[0], 1))
  x_t = (1.0 - t) * x0 + t * x1
  v_target = x1 - x0
  v_pred = apply(params, context, x_t, t)
  per_example = jnp.sum(jnp.square(v_pred - v_target), axis=-1)   # [B]
  return jnp.mean(per_example), {'per_example': per_example}


def flow_matching_loss_fixed(apply, params, context, action, x0, t):
  """Same loss with EXTERNALLY supplied (x0, t).

  Used for validation so the reported number is a deterministic function of the
  parameters -- otherwise the val curve would carry the noise of a fresh (x0, t)
  draw at every eval and be uncomparable across steps."""
  x_t = (1.0 - t) * x0 + t * action
  v_target = action - x0
  v_pred = apply(params, context, x_t, t)
  return jnp.mean(jnp.sum(jnp.square(v_pred - v_target), axis=-1))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
#: Euler steps for sampling. ALIGNED WITH THE OFFICIAL CFQL IMPLEMENTATION,
#: which uses flow_steps = 10. (Earlier revisions of this file labeled 10 a
#: prototype guess because no CFQL source was available to check against; that
#: is no longer the case.) Override with --flow-steps.
DEFAULT_FLOW_STEPS = 10

#: Per-step action clipping during Euler integration. The official CFQL flow
#: sampler clips to the environment action box after EVERY integration step:
#:
#:     for i in range(flow_steps):
#:         t = i / flow_steps
#:         v = flow(...)
#:         actions = actions + v / flow_steps
#:         actions = clip(actions, -1, 1)
#:
#: This is a property of ACTION SAMPLING / numerical integration ONLY. It is
#: NOT part of the flow-matching training target: ``flow_matching_loss`` still
#: regresses the unconstrained velocity ``x1 - x0`` on the linear path, the
#: network output is never squashed, and the stored offline actions are never
#: modified. Clipping the integration path keeps generated actions inside the
#: same support as the data, which also removes an out-of-range shortcut that a
#: later behavior-vs-target discriminator could otherwise exploit.
DEFAULT_PER_STEP_CLIP = True
CLIP_PROVENANCE = 'official CFQL sampling implementation'
FLOW_STEPS_PROVENANCE = 'aligned with official CFQL implementation'


def _euler_integrate(apply, params, context, key, action_dim, num_steps,
                     clip, box, x0=None):
  """Shared Euler loop. Returns (x, n_clipped_coord_updates).

  ``x0`` (optional) supplies the base noise explicitly. Passing it makes the
  sample a deterministic function of (params, context, x0, num_steps), which is
  what the integration-step sweep needs: holding x0 fixed while varying
  num_steps isolates discretization from everything else. When omitted, x0 is
  drawn from ``key`` as before."""
  batch = context.shape[0]
  lo, hi = box
  if x0 is None:
    x0 = jax.random.normal(key, (batch, action_dim))
  dt = 1.0 / num_steps

  def body(k, carry):
    x, n_clipped = carry
    t = jnp.full((batch, 1), k * dt)
    x_next = x + apply(params, context, x, t) * dt
    if clip:
      x_clipped = jnp.clip(x_next, lo, hi)
      n_clipped = n_clipped + jnp.sum(x_next != x_clipped)
      x_next = x_clipped
    return (x_next, n_clipped)

  return jax.lax.fori_loop(0, num_steps, body, (x0, jnp.int32(0)))


def sample_actions(apply, params, context, key, action_dim,
                   num_steps=DEFAULT_FLOW_STEPS, clip=DEFAULT_PER_STEP_CLIP,
                   box=ACTION_BOX):
  """Euler-integrate the velocity field from t=0 to t=1 (CFQL-aligned).

      x <- z ~ N(0, I)
      for k in range(num_steps):
          t = k / num_steps
          x <- x + v(c, x, t) / num_steps
          x <- clip(x, -1, 1)          # when clip=True (the default)

  This is THE BehaviorFlow sampler for downstream stages: the clip is applied
  after EVERY Euler update, matching the official CFQL sampler, not once at the
  end. Pass ``clip=False`` for the diagnostic-only raw sampler.

  Returns [B, action_dim]."""
  x, _ = _euler_integrate(apply, params, context, key, action_dim, num_steps,
                          clip, box)
  return x


def sample_actions_raw(apply, params, context, key, action_dim,
                       num_steps=DEFAULT_FLOW_STEPS, box=ACTION_BOX):
  """DIAGNOSTIC ONLY: unclipped Euler integration.

  Kept so out-of-box mass stays measurable (it is what the corrected sampler
  removes). Never use this as the BehaviorFlow sampler for later stages."""
  del box
  return sample_actions(apply, params, context, key, action_dim,
                        num_steps=num_steps, clip=False)


def sample_actions_with_clip_stats(apply, params, context, key, action_dim,
                                   num_steps=DEFAULT_FLOW_STEPS,
                                   box=ACTION_BOX):
  """Clipped sampler + how often the clip actually fired.

  Returns (x, n_clipped_coord_updates, n_total_coord_updates) so a caller can
  report whether clipping is a rare numerical correction or dominates."""
  x, n_clipped = _euler_integrate(apply, params, context, key, action_dim,
                                  num_steps, True, box)
  total = context.shape[0] * action_dim * num_steps
  return x, n_clipped, total


def sample_actions_from_noise(apply, params, context, x0,
                              num_steps=DEFAULT_FLOW_STEPS,
                              clip=DEFAULT_PER_STEP_CLIP, box=ACTION_BOX):
  """Same integrator, with the base noise supplied explicitly.

  ``context``: [B, C], ``x0``: [B, action_dim]. Identical mathematics to
  ``sample_actions``; only the source of z differs. Use this whenever two runs
  must share the same noise (e.g. the num_steps sweep)."""
  x, _ = _euler_integrate(apply, params, context, None, x0.shape[-1],
                          num_steps, clip, box, x0=x0)
  return x


def integrate_with_overshoot(apply, params, context, x0,
                             num_steps=DEFAULT_FLOW_STEPS, box=ACTION_BOX):
  """Clipped integration that also returns the PER-STEP overshoot tensor.

  Returns (x_final [B, A], overshoot [num_steps, B, A]) where ``overshoot`` is
  ``max(|x_next| - hi_or_lo, 0)`` measured BEFORE each clip, i.e. how far that
  Euler update tried to leave the box. Zero entries are steps where the clip
  did not fire, so the clip-trigger rate and the overshoot distribution both
  come from the same tensor. Uses scan (not fori_loop) to stack the per-step
  values; memory is num_steps * B * A floats, so call it on a bounded batch."""
  lo, hi = box
  dt = 1.0 / num_steps

  def step(x, k):
    t = jnp.full((x.shape[0], 1), k * dt)
    x_next = x + apply(params, context, x, t) * dt
    over = jnp.maximum(x_next - hi, 0.0) + jnp.maximum(lo - x_next, 0.0)
    return jnp.clip(x_next, lo, hi), over

  x_final, overshoot = jax.lax.scan(step, x0, jnp.arange(num_steps))
  return x_final, overshoot


def sample_actions_multi(apply, params, context, key, action_dim, num_samples,
                         num_steps=DEFAULT_FLOW_STEPS,
                         clip=DEFAULT_PER_STEP_CLIP, box=ACTION_BOX):
  """K independent samples per context row.

  context: [B, C] -> returns [B, K, action_dim]. Each of the B*K rows gets its
  own z, so a collapsed sampler shows up as zero spread. Per-step clipping is
  on by default, as in ``sample_actions``."""
  b = context.shape[0]
  rep = jnp.repeat(context, num_samples, axis=0)                  # [B*K, C]
  flat = sample_actions(apply, params, rep, key, action_dim,
                        num_steps=num_steps, clip=clip, box=box)
  return jnp.reshape(flat, (b, num_samples, action_dim))


def clip_to_box(actions, box=ACTION_BOX):
  """Environment-valid action. Reported SEPARATELY from the raw sample so that
  out-of-box mass stays visible in the diagnostics."""
  lo, hi = box
  return jnp.clip(actions, lo, hi)


def out_of_box_fraction(actions, box=ACTION_BOX):
  """Fraction of raw scalar entries outside the action box."""
  lo, hi = box
  a = np.asarray(actions)
  return float(np.mean((a < lo) | (a > hi)))
