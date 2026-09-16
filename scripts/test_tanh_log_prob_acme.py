"""Verify crl.networks.tanh_normal_log_prob_acme against dm-acme 0.4.0's rule.

Reference: an independent float64 implementation of Acme 0.4.0's
TanhTransformedDistribution.log_prob (acme/jax/networks/distributional.py,
threshold 0.999) built from scipy.stats.norm, plus two semantic checks that do
not go through that formula at all: the boundary value equals the average of
the tanh-Gaussian density over the band [0.999, 1] obtained by quadrature, and
interior density + both band masses integrate to one.  Gradients w.r.t. loc
and scale are checked against central finite differences of the float64
reference.  Also reports how far the port's historical 'clip' log-prob is
from Acme's at the boundary.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax                                     # noqa: E402
import jax.numpy as jnp                        # noqa: E402
from scipy import integrate, stats             # noqa: E402

from crl import networks                       # noqa: E402

THRESHOLD = 0.999


def ref_log_prob_1d(a, loc, scale):
  """Acme 0.4.0 rule in float64, one action dimension."""
  inv_t = np.arctanh(THRESHOLD)
  log_eps = np.log(1.0 - THRESHOLD)
  a = np.clip(a, -THRESHOLD, THRESHOLD)
  if a >= THRESHOLD:
    return stats.norm.logsf(inv_t, loc=loc, scale=scale) - log_eps
  if a <= -THRESHOLD:
    return stats.norm.logcdf(-inv_t, loc=loc, scale=scale) - log_eps
  x = np.arctanh(a)
  return stats.norm.logpdf(x, loc=loc, scale=scale) - np.log1p(-np.tanh(x) ** 2)


def ref_log_prob(actions, loc, scale):
  return np.array([sum(ref_log_prob_1d(a[d], loc[d], scale[d])
                       for d in range(len(a))) for a in actions])


def tanh_normal_pdf(y, loc, scale):
  """Density of tanh(Normal) at y in (-1, 1), float64."""
  x = np.arctanh(y)
  return stats.norm.pdf(x, loc=loc, scale=scale) / (1.0 - y * y)


def main():
  rng = np.random.default_rng(0)
  failures = []

  def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}  {detail}')
    if not ok:
      failures.append(name)

  jl = lambda p, a: np.asarray(networks.tanh_normal_log_prob_acme(p, jnp.asarray(a, jnp.float32)))
  # --- 1. values: interior, exactly at +/-1, at +/-0.9995, exactly at the threshold
  loc = np.array([0.3, -1.2, 4.0, -5.5])
  scale = np.array([0.5, 1.0, 2.0, 0.2])
  params = networks.TanhNormalParams(loc=jnp.asarray(loc, jnp.float32),
                                     scale=jnp.asarray(scale, jnp.float32))
  interior = rng.uniform(-0.99, 0.99, size=(64, 4))
  err = np.abs(jl(params, interior) - ref_log_prob(interior, loc, scale)).max()
  check('interior actions match Acme reference', err < 2e-4, f'max abs err {err:.2e}')
  legacy = np.asarray(networks.tanh_normal_log_prob(params, jnp.asarray(interior, jnp.float32)))
  err = np.abs(jl(params, interior) - legacy).max()
  check('interior branch equals the historical clip log-prob', err < 1e-4, f'max abs err {err:.2e}')
  for name, a in (('actions exactly +1', np.ones((1, 4))), ('actions exactly -1', -np.ones((1, 4))),
                  ('actions at +0.9995 (inside the band)', np.full((1, 4), 0.9995)),
                  ('actions at -0.9995', np.full((1, 4), -0.9995)),
                  ('actions exactly at the threshold 0.999', np.full((1, 4), THRESHOLD)),
                  ('mixed: [1, -1, 0.5, -0.9995]', np.array([[1.0, -1.0, 0.5, -0.9995]]))):
    got, want = jl(params, a), ref_log_prob(a, loc, scale)
    err = (np.abs(got - want) / np.maximum(1.0, np.abs(want))).max()   # float32 vs float64
    check(name, err < 2e-5, f'max rel err {err:.2e} (values {np.round(want, 3)})')

  # --- 2. semantics: band value == average density over the band (quadrature)
  for d in range(4):
    mass, _ = integrate.quad(lambda y: tanh_normal_pdf(y, loc[d], scale[d]), THRESHOLD, 1.0)
    if mass < 1e-290:          # quadrature underflows in the far tail; the closed form does not
      print(f'  skip  quadrature underflow for loc {loc[d]:+.1f} scale {scale[d]:.1f} (mass {mass:.1e})')
      continue
    avg = np.log(mass / (1.0 - THRESHOLD))
    got = float(networks.tanh_normal_log_prob_acme(
        networks.TanhNormalParams(loc=jnp.asarray([loc[d]], jnp.float32),
                                  scale=jnp.asarray([scale[d]], jnp.float32)),
        jnp.asarray([[1.0]], jnp.float32))[0])
    check(f'right band value = log(avg density over [0.999, 1]) for loc {loc[d]:+.1f} scale {scale[d]:.1f}',
          abs(got - avg) < 2e-3, f'{got:.4f} vs quadrature {avg:.4f}')
    mass_in, _ = integrate.quad(lambda y: tanh_normal_pdf(y, loc[d], scale[d]), -THRESHOLD, THRESHOLD, limit=200)
    mass_l, _ = integrate.quad(lambda y: tanh_normal_pdf(y, loc[d], scale[d]), -1.0, -THRESHOLD)
    mass_r, _ = integrate.quad(lambda y: tanh_normal_pdf(y, loc[d], scale[d]), THRESHOLD, 1.0)
    check(f'interior + both bands integrate to one (loc {loc[d]:+.1f})',
          abs(mass_in + mass_l + mass_r - 1.0) < 1e-3, f'{mass_in + mass_l + mass_r:.5f}')

  # --- 3. gradients w.r.t. loc and scale vs finite differences of the reference
  def grad_fd(a, loc, scale, which, h=1e-4):
    g = np.zeros(4)
    for d in range(4):
      lp, lm = loc.copy(), loc.copy()
      sp, sm = scale.copy(), scale.copy()
      if which == 'loc':
        lp[d] += h; lm[d] -= h
      else:
        sp[d] += h; sm[d] -= h
      g[d] = (ref_log_prob(a, lp, sp)[0] - ref_log_prob(a, lm, sm)[0]) / (2 * h)
    return g
  for name, a in (('interior', np.array([[0.2, -0.7, 0.95, -0.5]])),
                  ('boundary +1', np.ones((1, 4))), ('boundary -1', -np.ones((1, 4))),
                  ('mixed', np.array([[1.0, -1.0, 0.5, 0.9995]]))):
    for which in ('loc', 'scale'):
      def f(v):
        p = (networks.TanhNormalParams(loc=v, scale=params.scale) if which == 'loc'
             else networks.TanhNormalParams(loc=params.loc, scale=v))
        return jnp.sum(networks.tanh_normal_log_prob_acme(p, jnp.asarray(a, jnp.float32)))
      g = np.asarray(jax.grad(f)(params.loc if which == 'loc' else params.scale))
      g_ref = grad_fd(a, loc, scale, which)
      err = np.abs(g - g_ref).max() / (np.abs(g_ref).max() + 1e-6)
      check(f'd log_prob / d {which} at {name} matches finite differences', err < 2e-2,
            f'rel err {err:.2e}; jax {np.round(g, 4)} ref {np.round(g_ref, 4)}')
  finite = np.all(np.isfinite(jl(params, np.array([[1.0, -1.0, 1.0, -1.0]]))))
  check('boundary log-probs finite', bool(finite))

  # --- 4. how different the two rules are at the boundary (for the record)
  p1 = networks.TanhNormalParams(loc=jnp.asarray([[0.0]], jnp.float32), scale=jnp.asarray([[1.0]], jnp.float32))
  one = jnp.asarray([[1.0]], jnp.float32)
  print(f'  loc 0, scale 1, action +1: clip rule {float(networks.tanh_normal_log_prob(p1, one)[0]):.2f}  '
        f'acme rule {float(networks.tanh_normal_log_prob_acme(p1, one)[0]):.2f}')
  p2 = networks.TanhNormalParams(loc=jnp.asarray([[0.0]], jnp.float32), scale=jnp.asarray([[0.3]], jnp.float32))
  print(f'  loc 0, scale 0.3, action +1: clip rule {float(networks.tanh_normal_log_prob(p2, one)[0]):.2f}  '
        f'acme rule {float(networks.tanh_normal_log_prob_acme(p2, one)[0]):.2f}')
  print('ALL PASS' if not failures else f'FAILED: {failures}')
  return 0 if not failures else 1


if __name__ == '__main__':
  sys.exit(main())
