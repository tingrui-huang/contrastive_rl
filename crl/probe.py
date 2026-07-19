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
V_SLOW = 0.6          # 0.4 fought the base gait's stability envelope; 0.6
                      # still gives ~5x kinetic-energy separation vs fast
ALPHA = 0.6           # residual action amplitude cap; 0.3 lacked the
                      # authority to brake 1.4->0.4 or hold side lanes
                      # (carrot distance does NOT modulate speed: measured
                      # vx 1.06-1.42 across LOOKAHEAD 0.6-3.0)
LOOKAHEAD = 3.0       # carrot distance (matches litter_geometry_gate)
CARROT_CAP_X = 7.0
#: state, carrot xy, y_ref, v_ref, and EXPLICIT tracking errors (y - y_ref,
#: vx - v_ref): both are derivable from the state, but making them linearly
#: available shortcuts what TD3 was failing to discover on its own (60k
#: steps with zero speed-command separation before this change).
RES_OBS_DIM = 29 + 2 + 1 + 1 + 2


def residual_obs(state29, carrot_xy, y_ref, v_ref):
  y_err = state29[1] - y_ref
  v_err = state29[15] - v_ref            # qvel[0] = forward speed
  return np.concatenate([state29, carrot_xy,
                         [y_ref, v_ref, y_err, v_err]]).astype(np.float32)


#: carrot-steering head: the residual's 9th output is a lateral OFFSET added
#: to the carrot goal's y before the base actor sees it. Steering the base
#: through its own goal input has far more lateral authority than fighting
#: its torques (pure 8-dim torque residual plateaued at y_err ~1.3 for 150k
#: steps), and the base keeps its own balance while turning.
DY_CAP = 0.8
ACT_DIM = 9            # 8 torque residual + 1 carrot dy
ACT_LO = np.array([-ALPHA] * 8 + [-DY_CAP], np.float32)
ACT_HI = -ACT_LO


def make_residual_networks(hidden=(256, 256)):
  """Returns (actor, critic) haiku transforms.

  actor(obs[RES_OBS_DIM]) -> [torque residual (8, +-ALPHA), carrot dy
  (+-DY_CAP)]
  critic(obs, act9) -> (q1, q2) twin scalars
  """
  scale = jnp.asarray(ACT_HI)

  def actor_fn(obs):
    h = obs
    for w in hidden:
      h = jax.nn.relu(hk.Linear(w)(h))
    out = hk.Linear(ACT_DIM, w_init=hk.initializers.VarianceScaling(1e-4),
                    b_init=jnp.zeros)(h)
    return scale * jnp.tanh(out)

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
    da = np.zeros(ACT_DIM, np.float32)
    if self._params is not None:
      ro = residual_obs(obs58[:29], g, y_ref, v_ref)
      da = np.asarray(self._res(self._params, ro[None])[0])
    g_steered = np.array([g[0], np.clip(g[1] + da[8], -1.5, 1.5)],
                         np.float32)
    o_cmd = obs58.copy()
    o_cmd[29:] = 0.0
    o_cmd[29:31] = g_steered
    a = np.asarray(self._base(o_cmd[None])[0]) + da[:8]
    return np.clip(a, -1.0, 1.0), g_steered


def save_residual(path, params, meta):
  with open(path, 'wb') as f:
    pickle.dump({'params': jax.device_get(params), 'meta': meta}, f)


def load_residual(path):
  with open(path, 'rb') as f:
    blob = pickle.load(f)
  return blob['params'], blob.get('meta', {})
