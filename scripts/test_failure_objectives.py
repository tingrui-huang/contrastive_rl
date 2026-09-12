"""Formula and numerical checks independent of PointMaze checkpoints."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ett.anchored_transition import conditional_mmd2
from ett.failure_objectives import (
    mmd_components, nearest_reference, conditional_set_cost, shift_with_newest)


def f4(x):
  return jnp.tile(jnp.asarray([x, 0.]), 4)


class ObjectiveTests(unittest.TestCase):
  def test_scalar_formula_and_grouping(self):
    y = jnp.stack([jnp.stack([f4(0), f4(0)]), jnp.stack([f4(1), f4(1)])])
    f = jnp.stack([f4(0), f4(1)])
    parts = mmd_components(y, f, 0., 1., [1.])
    cross = np.exp(-2.)
    np.testing.assert_allclose(parts['gg'], 1.)
    np.testing.assert_allclose(parts['rr'], (1+cross)/2)
    np.testing.assert_allclose(parts['gr'], (1+cross)/2)
    np.testing.assert_allclose(parts['mmd'], (1-cross)/2, rtol=1e-6)
    np.testing.assert_allclose(parts['mmd'], conditional_mmd2(y, f, 0., 1., [1.]))
    self.assertAlmostEqual(float(mmd_components(y.reshape(1, 4, 8), f, 0., 1., [1.])['mmd'][0]), 0., places=6)

  def test_duplicate_invariance_and_frequency_sensitivity(self):
    f = jnp.stack([f4(0), f4(1)])
    reweighted = jnp.stack([f4(0), f4(0), f4(0), f4(1)])
    y = jnp.stack([f4(.2), f4(.4)])[None]
    np.testing.assert_allclose(conditional_set_cost(y, f, 0., 1.), conditional_set_cost(y, reweighted, 0., 1.))
    self.assertGreater(abs(float(mmd_components(y, f, 0., 1., [1.])['mmd'][0] -
                                 mmd_components(y, reweighted, 0., 1., [1.])['mmd'][0])), .01)

  def test_single_failure_mode_collapse_has_zero_set_cost(self):
    f = jnp.stack([f4(0), f4(1)])
    collapsed = jnp.repeat(f[:1], 16, axis=0)[None]
    np.testing.assert_array_equal(conditional_set_cost(collapsed, f, 0., 1.), 0.)
    self.assertGreater(float(mmd_components(collapsed, f, 0., 1., [1.])['mmd'][0]), .1)
    np.testing.assert_array_equal(nearest_reference(f, f, 0., 1.)['distance2'], 0.)

  def test_same_reference_decomposition_and_history_floor(self):
    y = jnp.asarray([[[0., 0., 1., 0., 1., 0., 1., 0.]]])
    f = jnp.stack([f4(0), f4(1)])
    detail = nearest_reference(y, f, 0., jnp.ones(8))
    self.assertEqual(int(detail['index'][0, 0]), 1)
    self.assertAlmostEqual(float(detail['newest_distance2'][0, 0]), 1.)
    self.assertAlmostEqual(float(detail['history_distance2'][0, 0]), 0.)
    np.testing.assert_allclose(detail['distance2'], detail['newest_distance2']+detail['history_distance2'])
    self.assertTrue(np.all(np.asarray(detail['history_floor']) <= np.asarray(detail['distance2'])))

  def test_piecewise_gradient_and_tie_convention(self):
    f = jnp.stack([f4(-1), f4(1)])
    state = f4(.25)[None]
    std = jnp.array([2., 1.] * 4)
    function = lambda xy: conditional_set_cost(shift_with_newest(state, xy), f, 7., std).sum()
    xy = state[:, None, :2]
    gradient = np.asarray(jax.grad(function)(xy))
    self.assertAlmostEqual(float(gradient[0, 0, 0]), 2*(.25-1)/4, places=6)
    h = 1e-3
    finite_difference = (float(function(xy.at[0, 0, 0].add(h))) - float(function(xy.at[0, 0, 0].add(-h))))/(2*h)
    self.assertAlmostEqual(finite_difference, float(gradient[0, 0, 0]), places=3)
    tie = f4(0)[None, None]
    self.assertEqual(int(nearest_reference(tie, f, 0., jnp.ones(8))['exact_tie_count'][0, 0]), 2)
    self.assertEqual(int(nearest_reference(tie, f, 0., jnp.ones(8))['index'][0, 0]), 0)
    # Branch selection is documented; no averaging to a fictitious zero gradient.
    self.assertGreater(float(jax.grad(lambda v: conditional_set_cost(v, f, 0., jnp.ones(8)).sum())(tie)[0, 0, 0]), 0.)

  def test_v_statistic_k_effect_is_exactly_accounted_for(self):
    f = jnp.stack([f4(-1), f4(1)])
    y = jnp.stack([f4(-.5), f4(.5)])[None]
    parts = mmd_components(y, f, 0., 1., [1.])
    gg_off_diagonal = np.exp(-2.)
    expected = gg_off_diagonal + float(parts['rr'][0]) - 2*float(parts['gr'][0])
    self.assertAlmostEqual(float(parts['mmd_generated_u'][0]), expected, places=6)
    np.testing.assert_allclose(parts['mmd'] - parts['mmd_generated_u'], parts['generated_self_pair_correction'], atol=1e-7)

  def test_history_is_preserved_and_newest_only_cost_is_not_substituted(self):
    s = f4(2)[None]
    y = shift_with_newest(s, jnp.zeros((1, 8, 2)))
    np.testing.assert_array_equal(y[..., 2:], jnp.broadcast_to(s[:, None, :6], (1, 8, 6)))
    self.assertEqual(float(conditional_set_cost(y, f4(0)[None], 0., jnp.ones(8))[0]), 12.)


if __name__ == '__main__':
  unittest.main(verbosity=2)
