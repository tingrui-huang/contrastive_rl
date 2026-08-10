"""Behavior-vs-target support discriminator (Stage 3A).

WHAT THIS IS NOT
----------------
The output of this model is NOT the continuous propensity ``P(A = a | S = s)``
and must NOT be used as a causal mixture weight. Trained on a balanced
behavior-vs-target classification problem, an ideal discriminator recovers

    D*(x) = p_behavior(x) / (p_behavior(x) + p_target(x))

a RELATIVE density/discrepancy score under an artificial 50/50 class prior --
not the Manski propensity mass the discrete formulation used. ``sigmoid(logit)``
is the posterior of that artificial classification problem and nothing more.
Throughout this package the output is called a support_score /
relative_support_score, never a propensity.

INPUT SPECIFICATIONS
--------------------
The general causal formulation is ``D(s, g_cmd, a)``: condition on the state and
the PRE-ACTION commanded goal, never on the replay query goal ``g_query`` (an
achieved future state, hence a descendant of the action).

  state_cmdgoal_action  [s | g_cmd[live] | a]  -- design B, the primary model
  state_action          [s | a]                -- design A, goal-marginalized
  action                [a]                    -- diagnostic: how much is
                                                  explainable by the global
                                                  action distribution alone
  context               [s | g_cmd[live]]      -- diagnostic: MUST be at chance,
                                                  since positives and negatives
                                                  share the identical context

For the rockfall dataset only ``g_cmd`` dims [0, 1] are live (the other 27 are
identically zero), so the prototype feeds ``g_cmd[:2]``. That is an
environment-specific narrowing of the general D(s, g_cmd, a) formulation, NOT a
redefinition of the method; the indices are configurable and recorded in the
run metadata.
"""
import dataclasses
from typing import Callable, NamedTuple, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

#: Which parts of (state, g_cmd, action) each input spec consumes.
INPUT_SPECS = {
    'state_cmdgoal_action': ('state', 'cmdgoal', 'action'),
    'state_action': ('state', 'action'),
    'action': ('action',),
    'context': ('state', 'cmdgoal'),
}


@dataclasses.dataclass
class DiscriminatorConfig:
  """Deliberately small MLP -- this is a support probe, not a capacity study."""

  input_dim: int
  hidden_sizes: Tuple[int, ...] = (256, 256)

  def asdict(self):
    d = dataclasses.asdict(self)
    d['hidden_sizes'] = list(self.hidden_sizes)
    return d


class Discriminator(NamedTuple):
  init: Callable
  apply: Callable      # apply(params, x) -> raw logit [B]


def make_discriminator(config: DiscriminatorConfig) -> Discriminator:
  """Linear(256) - ReLU - Linear(256) - ReLU - Linear(1), raw logit output."""

  def _forward(x):
    h = x
    for width in config.hidden_sizes:
      h = jax.nn.relu(hk.Linear(width)(h))
    return jnp.squeeze(hk.Linear(1)(h), axis=-1)

  t = hk.without_apply_rng(hk.transform(_forward))
  return Discriminator(init=t.init, apply=t.apply)


def assemble_inputs(spec, state, cmdgoal, action):
  """Concatenate exactly the parts named by ``spec``, in a fixed order.

  ``g_query`` is deliberately absent from every spec -- see the module
  docstring. Callers pass ``cmdgoal`` already restricted to the live indices."""
  if spec not in INPUT_SPECS:
    raise ValueError(f'unknown input spec {spec!r}')
  parts = []
  for name in INPUT_SPECS[spec]:
    parts.append({'state': state, 'cmdgoal': cmdgoal, 'action': action}[name])
  return jnp.concatenate([jnp.asarray(p) for p in parts], axis=-1)


def input_dim_for(spec, state_dim, cmdgoal_dim, action_dim):
  sizes = {'state': state_dim, 'cmdgoal': cmdgoal_dim, 'action': action_dim}
  return int(sum(sizes[n] for n in INPUT_SPECS[spec]))


def bce_with_logits(logits, labels):
  """Mean binary cross-entropy. labels: 1 = behavior (real), 0 = CRL target."""
  logits = jnp.asarray(logits)
  labels = jnp.asarray(labels, logits.dtype)
  # log(1 + exp(-|z|)) + max(z, 0) - z * y  -- numerically stable form.
  return jnp.mean(jnp.maximum(logits, 0) - logits * labels
                  + jnp.log1p(jnp.exp(-jnp.abs(logits))))


class Standardizer(NamedTuple):
  """Per-feature affine normalization fitted on the TRAIN split only."""

  mean: np.ndarray
  std: np.ndarray

  def apply(self, x):
    return (jnp.asarray(x) - jnp.asarray(self.mean)) / jnp.asarray(self.std)

  def asdict(self):
    return {'mean': np.asarray(self.mean).tolist(),
            'std': np.asarray(self.std).tolist()}


def fit_standardizer(x):
  """Ant state dims span orders of magnitude, so the MLP needs scaling. Fitted
  on training-split inputs only; the dev and test splits never contribute."""
  x = np.asarray(x, np.float64)
  mean = x.mean(axis=0)
  std = x.std(axis=0)
  std = np.where(std < 1e-8, 1.0, std)        # constant features -> no scaling
  return Standardizer(mean=mean.astype(np.float32),
                      std=std.astype(np.float32))
