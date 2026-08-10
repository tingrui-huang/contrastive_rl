"""Canonical behavior-vs-target AGREEMENT score API (integration-ready).

    This is a learned behavior-vs-target agreement surrogate.
    It is NOT a literal continuous propensity probability.

``D_psi(s, g_cmd, a) in [0, 1]`` answers, softly:

    high D  -- the queried action agrees with / resembles actions supported by
               the observational behavior policy in this PRE-ACTION context
    low  D  -- the queried action disagrees with the observational behavior
               policy there

Motivation. In the discrete causal formulation, exact action agreement
``1[a = a_obs]`` decides how much mass goes to the observational (nominal)
branch versus the pessimistic branch. In a continuous action space the exact
point mass ``P(A = a | S = s)`` is zero, so no exact analogue exists. Following
the same theory-to-practice step CFQL takes, we replace the hard agreement
indicator with a LEARNED SOFT AGREEMENT SCORE.

Difference from CFQL. CFQL's positives are BC-Flow samples because its FQL
backbone already contains a behavior generator. The Contrastive RL backbone
does not, so we use the REAL offline behavior actions directly as positives.
That removes an unnecessary generative model and avoids the severe
Flow-generated boundary artifact measured in the Stage-2.5 audit
(boundary-only AUC ~0.97 for flow-vs-CRL versus ~0.52 for real-vs-CRL).

g_cmd vs g_query -- do not confuse these:

    g_cmd    the actual PRE-ACTION commanded goal from the offline data. Part
             of the behavior decision context. IT IS AN INPUT TO D.

    g_query  a hindsight/future achieved state used by CRL replay to index
             which goal-conditioned value function is being queried. It is a
             DESCENDANT of the action. It is used ONLY to ask the CRL actor for
             a proposed action. IT IS NOT AN INPUT TO D, and substituting it
             for g_cmd would turn the object into a hindsight posterior rather
             than a behavior-support quantity.

What this module deliberately does NOT contain: no BehaviorFlow, no uniform
reference distribution, no action-neighborhood bandwidth h, no density or
density-ratio estimation, and no calibration into a claimed propensity.

Downstream use (a future causal learner needs nothing else):

    from propensity.agreement import load_agreement_model, agreement_score_batch

    model = load_agreement_model('artifacts/support_discriminator/'
                                 'D_state_cmdgoal_action')
    w = agreement_score_batch(model.params, model.spec, states, g_cmd, actions)

``w`` is a pure, jit/vmap-friendly forward pass: no environment, no CRL policy
call, no dataset access.
"""
import dataclasses
import json
import os
import pickle
import sys
from typing import NamedTuple, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from propensity import discriminator as disc_mod

#: The one active formulation. Diagnostics/baselines (state_action, action,
#: context) exist in the repository but must NOT be used by the causal learner.
ACTIVE_INPUT_SPEC = 'state_cmdgoal_action'


@dataclasses.dataclass(frozen=True)
class AgreementSpec:
  """Everything needed to rebuild the forward pass, from checkpoint metadata."""

  state_dim: int
  action_dim: int
  g_cmd_dim_full: int          # full stored width (29 here); general form
  g_cmd_indices: Tuple[int, ...]   # LIVE indices actually fed (env-specific)
  hidden_sizes: Tuple[int, ...]
  input_spec: str = ACTIVE_INPUT_SPEC

  @property
  def g_cmd_dim(self):
    """Width actually consumed by the network."""
    return len(self.g_cmd_indices)

  @property
  def input_dim(self):
    return disc_mod.input_dim_for(self.input_spec, self.state_dim,
                                  self.g_cmd_dim, self.action_dim)

  def asdict(self):
    d = dataclasses.asdict(self)
    d['g_cmd_indices'] = list(self.g_cmd_indices)
    d['hidden_sizes'] = list(self.hidden_sizes)
    d['g_cmd_dim'] = self.g_cmd_dim
    d['input_dim'] = self.input_dim
    return d


class AgreementParams(NamedTuple):
  """Pytree carrying everything the forward pass consumes.

  Kept as a pytree (not a closure) so jit/vmap/grad treat it as a normal
  argument and a caller can hold several seeds' parameters side by side."""

  net_params: object
  input_mean: jnp.ndarray
  input_std: jnp.ndarray


class AgreementModel(NamedTuple):
  params: AgreementParams
  spec: AgreementSpec
  meta: dict


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_agreement_model(run_dir):
  """Load a trained agreement model from a Stage-3A/3B run directory."""
  with open(os.path.join(run_dir, 'config.json')) as f:
    meta = json.load(f)
  if meta['input_spec'] != ACTIVE_INPUT_SPEC:
    raise ValueError(
        f'{run_dir} holds input_spec={meta["input_spec"]!r}, but the active '
        f'formulation is {ACTIVE_INPUT_SPEC!r} = D(s, g_cmd, a). The '
        'state_action / action / context models are DIAGNOSTIC BASELINES and '
        'must not be used by the causal learner.')
  with open(os.path.join(run_dir, 'model.pkl'), 'rb') as f:
    blob = pickle.load(f)
  spec = AgreementSpec(
      state_dim=int(meta['state_dim']),
      action_dim=int(meta['action_dim']),
      g_cmd_dim_full=int(meta.get('g_cmd_dim_full', 29)),
      g_cmd_indices=tuple(int(i) for i in meta['cmdgoal_indices']),
      hidden_sizes=tuple(int(h) for h in meta['architecture']['hidden_sizes']),
      input_spec=meta['input_spec'])
  params = AgreementParams(
      net_params=jax.tree_util.tree_map(jnp.asarray, blob['params']),
      input_mean=jnp.asarray(blob['standardizer']['mean']),
      input_std=jnp.asarray(blob['standardizer']['std']))
  return AgreementModel(params=params, spec=spec, meta=meta)


# --------------------------------------------------------------------------- #
# Core forward pass -- pure, vectorized, jit/vmap friendly
# --------------------------------------------------------------------------- #
def _select_g_cmd(g_cmd, spec):
  """Accept the FULL stored g_cmd and slice to the live indices.

  Passing an already-sliced g_cmd of width ``len(g_cmd_indices)`` is also
  accepted for convenience. (If a future environment has
  ``len(indices) == g_cmd_dim_full`` the two cases coincide and no slicing is
  needed, so the ambiguity is harmless.)"""
  g = jnp.asarray(g_cmd)
  width = g.shape[-1]
  if width == spec.g_cmd_dim_full and spec.g_cmd_dim != spec.g_cmd_dim_full:
    return jnp.take(g, jnp.asarray(spec.g_cmd_indices), axis=-1)
  if width == spec.g_cmd_dim:
    return g
  raise ValueError(
      f'g_cmd last dim {width} is neither the full stored width '
      f'{spec.g_cmd_dim_full} nor the live width {spec.g_cmd_dim}')


def agreement_logit_batch(params, spec, states, g_cmd, actions):
  """Raw logit, for diagnostics and for anything that needs an unsquashed score.

  Shapes: states [..., state_dim], g_cmd [..., g_cmd_dim_full] (or already
  sliced), actions [..., action_dim] -> [...]. Arbitrary leading dimensions are
  supported natively because every layer acts on the last axis, so jit and vmap
  need no reshaping. Pure forward pass: no environment, no CRL policy call, no
  dataset access."""
  net = disc_mod.make_discriminator(disc_mod.DiscriminatorConfig(
      input_dim=spec.input_dim, hidden_sizes=spec.hidden_sizes))
  x = disc_mod.assemble_inputs(spec.input_spec, jnp.asarray(states),
                               _select_g_cmd(g_cmd, spec), jnp.asarray(actions))
  x = (x - params.input_mean) / params.input_std
  return net.apply(params.net_params, x)


def agreement_score_batch(params, spec, states, g_cmd, actions):
  """AGREEMENT SCORE in [0, 1] -- the downstream-facing quantity.

  ``sigmoid`` of the discriminator logit. This is a learned behavior-vs-target
  agreement surrogate, NOT a literal continuous propensity probability and NOT
  a calibrated causal branch probability: it is the posterior of an artificial
  balanced behavior-vs-target classification problem."""
  return jax.nn.sigmoid(
      agreement_logit_batch(params, spec, states, g_cmd, actions))


#: Readable aliases. All three name the same soft agreement surrogate.
behavior_support_score = agreement_score_batch
soft_agreement = agreement_score_batch


def make_agreement_fns(spec, jit=True):
  """Return ``(logit_fn, score_fn)`` bound to ``spec``, optionally jitted.

  Both take ``(params, states, g_cmd, actions)``, so a caller can hold several
  seeds' ``AgreementParams`` and reuse one compiled function for all of them."""
  def logit_fn(params, states, g_cmd, actions):
    return agreement_logit_batch(params, spec, states, g_cmd, actions)

  def score_fn(params, states, g_cmd, actions):
    return agreement_score_batch(params, spec, states, g_cmd, actions)

  return (jax.jit(logit_fn), jax.jit(score_fn)) if jit else (logit_fn, score_fn)


class AgreementScorer:
  """Convenience object wrapper. ``scorer.score(s, g_cmd, a) -> [0, 1]``."""

  def __init__(self, model: AgreementModel, jit=True):
    self.model = model
    self.spec = model.spec
    self._logit_fn, self._score_fn = make_agreement_fns(model.spec, jit=jit)

  def score(self, state, g_cmd, action):
    """Agreement score in [0, 1]. Not a propensity."""
    return self._score_fn(self.model.params, state, g_cmd, action)

  def logit(self, state, g_cmd, action):
    return self._logit_fn(self.model.params, state, g_cmd, action)

  # Aliases, same object.
  agreement_score = score
  behavior_support_score = score
  soft_agreement = score
