"""Optional observation normalization for the (x, y, z) PointMaze variant.

NOT WIRED INTO ANYTHING. Nothing in crl/ imports this module, so adding it
changes no existing behaviour. It exists so the normalization choice for the
z dimension is explicit, configurable and unit-tested BEFORE any training run
turns it on.

WHY THIS EXISTS AT ALL. The current pipeline applies NO normalization to state
observations. crl/networks.py `_repr_fn` slices the flat observation raw:

    state = obs[:, :obs_dim]
    goal  = obs[:, obs_dim:]

and the only scaling anywhere is `/ 255.0` for image observations in
`_unflatten_obs`. `repr_norm` normalizes the OUTPUT representation, not the
input, and is False in the windy recipe. So there is no empirical mean/std
machinery that a new dimension would silently be picked up by -- z would go in
raw, at a magnitude of 0.5 against x in [0, 9] and y in [1, 4].

THE TWO CANDIDATE CONVENTIONS.

  'none'        identity. Exactly the current pipeline, bit-for-bit. Default.

  'z_physical'  divide ONLY the z column, in both the state and the goal half,
                by |z_min| taken from the environment. So z = 0 -> 0 and
                z = z_min -> -1, a unit of "one full sinking depth". X and Y
                are untouched, which is deliberate: this step is not a licence
                to renormalize the coordinates the existing 2-D results were
                produced with.

An empirical (z - mu_z) / sigma_z standardisation is deliberately NOT offered.
Failure is rare, so sigma_z is small and driven by the failure RATE rather than
by any physical quantity: the separation between a safe and a failed state
would then change whenever the dataset's death rate changed, which is exactly
the artifact the normalization audit was run to avoid. See
artifacts/swamp_windy_z/normalization_audit.json for the measured numbers.

The z column index is derived, not hard-coded: for this env the state is
(x, y, z) so z is the LAST state column, and the flat observation is
[state | goal] so the goal's z is the last column overall.
"""
import numpy as np


def z_scale_from_env(env):
  """|z_min| for the (x, y, z) env. Never duplicate the literal 0.5."""
  z_min = getattr(env, 'z_min', None)
  if z_min is None:
    raise ValueError('env has no z_min; this normalizer is for the (x, y, z) '
                     'variant only')
  scale = abs(float(z_min))
  if scale <= 0:
    raise ValueError('z_min must be non-zero, got %r' % z_min)
  return scale


def make_obs_normalizer(obs_dim, goal_dim, mode='none', z_scale=None):
  """Returns f(flat_obs [..., obs_dim + goal_dim]) -> flat_obs, same shape.

  mode='none' returns the input unchanged and is the default, so a caller that
  forgets to configure anything gets today's behaviour exactly.

  mode='z_physical' divides the z column of BOTH halves by z_scale. Both
  halves matter: the critic's g_encoder consumes the goal half, and leaving it
  raw while scaling the state half would put the two encoders on different
  units for the same physical quantity.
  """
  if mode == 'none':
    def identity(obs):
      return obs
    return identity

  if mode != 'z_physical':
    raise ValueError('unknown normalization mode %r (expected none|z_physical)'
                     % mode)
  if obs_dim != 3 or goal_dim != 3:
    raise ValueError('z_physical expects the 3-D (x, y, z) variant, got '
                     'obs_dim=%d goal_dim=%d' % (obs_dim, goal_dim))
  if z_scale is None or z_scale <= 0:
    raise ValueError('z_physical needs a positive z_scale (use '
                     'z_scale_from_env)')
  zs = float(z_scale)
  i_state_z = obs_dim - 1                 # (x, y, z) -> z is the last state col
  i_goal_z = obs_dim + goal_dim - 1       # ... and the last column overall

  def normalize(obs):
    obs = np.asarray(obs)
    out = obs.copy()
    out[..., i_state_z] = obs[..., i_state_z] / zs
    out[..., i_goal_z] = obs[..., i_goal_z] / zs
    return out
  return normalize
