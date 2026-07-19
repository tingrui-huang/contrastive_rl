"""Lane/speed-conditioned residual probe controller (Stage 1A).

One controller produces all four gate arms by COMMAND only:

    a_t = clip(a_base(s_t, g_t) + ALPHA * da_theta(s_t, g_t, y_ref, v_ref))

  * a_base: the frozen 0.89 offline-CRL actor (deterministic tanh(loc)),
    driven by the same continuous carrot goal g_t = (x + LOOKAHEAD, y_ref)
    the geometry gate used. NEVER trained here.
  * da_theta: a small residual MLP, zero-initialized (final layer scale
    ~1e-4) so training starts exactly at the validated base behavior;
    output squashed to [-ALPHA, ALPHA].
  * commands: y_ref in {-LANE_Y, 0, +LANE_Y}; v_ref in {V_SLOW, V_FAST}
    (target forward speed along +x in the corridor).

The low-level controller deliberately does NOT see U: the privileged
teacher's high-level rule maps U -> y_ref later (U -> y_ref -> A), keeping
the confounding pathway explicit and auditable.
"""
import pickle
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

LANE_Y = 1.1
V_FAST = 1.4          # ~natural corridor speed of the base actor (1.3-1.6)
V_SLOW = 0.4
ALPHA = 0.3           # residual action amplitude cap
LOOKAHEAD = 3.0       # carrot distance (matches litter_geometry_gate)
CARROT_CAP_X = 7.0
RES_OBS_DIM = 29 + 2 + 1 + 1   # state, carrot xy, y_ref, v_ref


def residual_obs(state29, carrot_xy, y_ref, v_ref):
  return np.concatenate([state29, carrot_xy,
                         [y_ref, v_ref]]).astype(np.float32)


def make_residual_networks(hidden=(256, 256)):
  """Returns (actor, critic) haiku transforms.

  actor(obs[RES_OBS_DIM]) -> residual in [-ALPHA, ALPHA]^8
  critic(obs, act8) -> (q1, q2) twin scalars
  """
  def actor_fn(obs):
    h = obs
    for w in hidden:
      h = jax.nn.relu(hk.Linear(w)(h))
    out = hk.Linear(8, w_init=hk.initializers.VarianceScaling(1e-4),
                    b_init=jnp.zeros)(h)
    return ALPHA * jnp.tanh(out)

  def critic_fn(obs, act):
    x = jnp.concatenate([obs, act], axis=-1)
    qs = []
    for _ in range(2):
      h = x
      for w in hidden:
        h = jax.nn.relu(hk.Linear(w)(h))
      qs.append(hk.Linear(1)(h)[..., 0])
    return tuple(qs)

  return (hk.without_apply_rng(hk.transform(actor_fn)),
          hk.without_apply_rng(hk.transform(critic_fn)))


class ProbeController:
  """Deployment-time controller: frozen base + frozen residual."""

  def __init__(self, base_act_fn, residual_params=None, actor=None):
    self._base = base_act_fn             # (obs58[None]) -> action[None, 8]
    self._actor = actor
    self._params = residual_params
    if residual_params is not None:
      self._res = jax.jit(lambda p, o: self._actor.apply(p, o))

  def carrot_goal(self, xy, y_ref, ramp_x=None):
    """Continuous carrot: lane point LOOKAHEAD ahead; optional exit ramp."""
    y = y_ref
    if ramp_x is not None:
      y = y_ref * min(1.0, max(0.0, (ramp_x - xy[0]) / 0.8))
    return np.array([min(xy[0] + LOOKAHEAD, CARROT_CAP_X), y], np.float32)

  def __call__(self, obs58, y_ref, v_ref, carrot=None):
    xy = obs58[:2]
    g = self.carrot_goal(xy, y_ref) if carrot is None else carrot
    o_cmd = obs58.copy()
    o_cmd[29:] = 0.0
    o_cmd[29:31] = g
    a = np.asarray(self._base(o_cmd[None])[0])
    if self._params is not None:
      ro = residual_obs(obs58[:29], g, y_ref, v_ref)
      a = a + np.asarray(self._res(self._params, ro[None])[0])
    return np.clip(a, -1.0, 1.0), g


def save_residual(path, params, meta):
  with open(path, 'wb') as f:
    pickle.dump({'params': jax.device_get(params), 'meta': meta}, f)


def load_residual(path):
  with open(path, 'rb') as f:
    blob = pickle.load(f)
  return blob['params'], blob.get('meta', {})
