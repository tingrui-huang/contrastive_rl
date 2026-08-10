"""Read-only adapter for the frozen CRL target policy + its future-goal rule.

Supplies the NEGATIVE class of the Stage-3A support discriminator: actions the
CRL learner would propose at a held-out offline state. Nothing here trains,
modifies, or checkpoints the CRL actor, and no environment is ever constructed
-- states come from the frozen offline npz and go through a frozen forward pass.

Two roles of "goal" are kept strictly apart (see the Stage-2.5b audit):

  g_cmd    the commanded task goal, drawn at env.reset(), constant within the
           episode, PRE-action, and part of the observation the teacher saw.
           This is the only goal the support discriminator conditions on.

  g_query  an ACHIEVED FUTURE STATE, sampled by crl/replay.py at replay time to
           index which goal-conditioned value function is being queried. It is a
           descendant of the action, so it must NEVER enter a behavior-support
           conditioning set. Here it only decides WHICH action the learner
           proposes -- it is not a discriminator input.

``sample_future_goal_index`` faithfully reproduces the future-goal law in
``crl/replay.py:_draw_indices`` (variable-length path):

    for a transition at time t in episode e with valid obs count L_e,
        j ~ Categorical( p(j) proportional to discount ** (j - t) )
        over j in [t+1, L_e-1]

drawn with the same Gumbel-max mechanism, so the terminal observation at
L_e - 1 is selectable exactly as it is in replay. The goal vector is
``obs[e, j, :obs_dim]`` restricted to ``goal_indices`` (= range(29) for the
rockfall runs, i.e. the full achieved state).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

#: CRL actor config -- scripts/verify_offline_d4rl.build_offline_cfg, the config
#: every rockfall run was trained with. Verified against checkpoint shapes.
CRL_HIDDEN = (1024, 1024)
CRL_REPR_DIM = 16
CRL_TWIN_Q = True
CRL_LAYER_NORM = False
CRL_DISCOUNT = 0.99            # crl replay future-goal discount


def load_frozen_crl_actor(ckpt_path, obs_dim, goal_dim, action_dim):
  """Load the frozen CRL policy. Returns (step, apply_fn, info).

  ``apply_fn(obs58) -> TanhNormalParams``. The checkpoint is opened read-only;
  the parameters are never written back."""
  from crl import checkpoint as crl_ckpt
  from crl import networks as crl_networks
  step, state = crl_ckpt.load_checkpoint(ckpt_path)
  nets = crl_networks.make_networks(
      obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim,
      repr_dim=CRL_REPR_DIM, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=CRL_HIDDEN, twin_q=CRL_TWIN_Q,
      use_image_obs=False, use_layer_norm=CRL_LAYER_NORM)
  params = state.policy_params
  w0 = np.asarray(params['mlp/~/linear_0']['w'])
  full = obs_dim + goal_dim
  if w0.shape != (full, CRL_HIDDEN[0]):
    raise ValueError(f'CRL actor input width {w0.shape[0]} != {full}; the '
                     'recorded architecture does not match this checkpoint.')
  apply_fn = jax.jit(lambda obs: nets.policy_network.apply(params, obs))
  info = {'checkpoint': os.path.abspath(ckpt_path), 'step': int(step),
          'hidden_layer_sizes': list(CRL_HIDDEN), 'repr_dim': CRL_REPR_DIM,
          'twin_q': CRL_TWIN_Q, 'use_layer_norm': CRL_LAYER_NORM,
          'action_range': 'tanh squash, no clipping',
          'frozen': True, 'trained_on': 'pi(a | s, g_query)'}
  return step, apply_fn, info


def sample_future_goal_index(lengths, ep_idx, t_idx, rng,
                             discount=CRL_DISCOUNT, chunk=4096):
  """j ~ Categorical(prop. discount**(j-t)) over j in [t+1, L_e-1].

  Faithful Gumbel-max reproduction of crl/replay.py:_draw_indices. Chunked so
  the [B, L] logit array stays bounded for long-horizon episodes."""
  ep_idx = np.asarray(ep_idx, np.int64)
  t_idx = np.asarray(t_idx, np.int64)
  L = int(np.max(lengths))
  arange = np.arange(L)
  log_d = float(np.log(discount)) if discount > 0 else -np.inf
  out = np.empty(len(ep_idx), np.int64)
  for lo in range(0, len(ep_idx), chunk):
    hi = min(lo + chunk, len(ep_idx))
    Lt = lengths[ep_idx[lo:hi]]                            # [b] valid obs count
    t = t_idx[lo:hi]
    valid = arange[None, :] < Lt[:, None]                  # within episode
    future = (arange[None, :] > t[:, None]) & valid        # strictly future
    logp = (arange[None, :] - t[:, None]) * log_d
    logits = np.where(future, logp, -np.inf)
    g = -np.log(-np.log(rng.uniform(size=logits.shape).clip(1e-20, 1.0)))
    out[lo:hi] = np.argmax(logits + g, axis=1)
  return out


def crl_actions(apply_fn, state, goal_query, eps):
  """Frozen-policy actions at (s, g_query).

  Returns (a_sample, a_mode):
    a_sample = tanh(loc + scale * eps)   one fixed STOCHASTIC draw per context
                                         (the primary negative)
    a_mode   = tanh(loc)                 the deterministic eval action
                                         (secondary sensitivity check)
  Range comes from the tanh squash; no clipping is applied anywhere."""
  obs = jnp.concatenate([jnp.asarray(state), jnp.asarray(goal_query)], axis=1)
  p = apply_fn(obs)
  return (np.asarray(jnp.tanh(p.loc + p.scale * jnp.asarray(eps))),
          np.asarray(jnp.tanh(p.loc)))


def load_goal_source(npz_path, obs_dim):
  """Episode-indexed achieved-state tensor ``[E, L, obs_dim]`` for goal lookup.

  Read directly from the frozen npz (obs only) rather than through the Stage-1
  loader, because the future-goal law can select the TERMINAL observation at
  index ``L_e - 1``, which is deliberately not a transition row and therefore
  absent from the loader's flattened (state, action) view. Clamping to the last
  transition row instead would silently alter the goal distribution."""
  with np.load(npz_path, allow_pickle=False) as d:
    obs = np.ascontiguousarray(d['obs'][:, :, :obs_dim])
  obs.flags.writeable = False
  return obs


def replay_semantics(discount=CRL_DISCOUNT, goal_indices=None):
  """Machine-readable record of the reproduced future-goal law."""
  return {
      'source': 'crl/replay.py:_draw_indices (variable-length path)',
      'discount': discount,
      'eligible_future_range': 'j in [t+1, L_e - 1] (same episode)',
      'probability_rule': 'P(j) proportional to discount ** (j - t)',
      'sampling_mechanism': 'Gumbel-max over masked log-probs (as in replay)',
      'terminal_observation_selectable': True,
      'goal_extraction': 'obs[e, j, :obs_dim] restricted to goal_indices',
      'goal_indices': (list(range(29)) if goal_indices is None
                       else list(goal_indices)),
      'crosses_episode_boundary': False,
      'role': 'g_query selects WHICH action the learner proposes; it is NOT a '
              'discriminator input (it is a descendant of the action).',
  }
