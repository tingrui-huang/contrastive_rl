"""Behavioral tests for the bounded, conditional ETT sampler (CPU compatible)."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ett.anchored_transition import AnchoredTransition, conditional_mmd2
from ett.diagonal_transition import (
    DiagonalTransitionModel, DiagonalTransitionSpec, make_transition_network,
    sample_displacement, TransitionDistribution)
from ett.eval_diagonal_transition import _open_endpoint


class AnchoredTests(unittest.TestCase):
  def setUp(self):
    spec = DiagonalTransitionSpec(hidden_sizes=(16, 16))
    params = make_transition_network(spec).init(jax.random.PRNGKey(1), jnp.zeros((1, 20)))
    self.base = DiagonalTransitionModel(params, spec, np.zeros(20), np.ones(20),
                                        np.zeros(2), np.ones(2) * 0.1)
    self.model = AnchoredTransition.initialize(self.base, jax.random.PRNGKey(2), 0.75)
    self.state = jnp.array([[2.9, 3.05, 2.5, 3.1, 2.2, 3.1, 1.8, 3.2],
                            [7.1, 1.5, 7.2, 1.6, 7.3, 1.7, 7.4, 1.8]])
    self.goal = jnp.tile(jnp.array([8.5, 3.5]), (2, 4))
    self.action = jnp.array([[0.8, -0.3], [-1., 1.]])

  def test_mmd_against_scalar_formula_and_conditional_grouping(self):
    generated = jnp.array([[[0.], [0.]], [[2.], [2.]]])
    reference = jnp.array([[0.], [2.]])
    value = np.asarray(conditional_mmd2(generated, reference, 0., 1., [1.]))
    expected = 0.5 * (1.0 - np.exp(-2.0))
    np.testing.assert_allclose(value, expected, rtol=1e-6)
    pooled = conditional_mmd2(generated.reshape(1, 4, 1), reference, 0., 1., [1.])
    np.testing.assert_allclose(pooled, 0., atol=2e-7)
    np.testing.assert_allclose(conditional_mmd2(reference[None], reference,
                                              0., 1., [.5, 1., 2.]), 0., atol=1e-7)

  def test_diagonal_identity_after_nonzero_residual_and_exact_shift(self):
    self.model.params['residual'] = jax.tree_util.tree_map(
        lambda p: p + 2., self.model.params['residual'])
    key = jax.random.PRNGKey(9)
    actual = self.model.sample(self.state, self.action, self.action, key, 31, self.goal)
    expected = self.base.sample(self.state, self.action, self.action, key, 31, self.goal)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        self.model.log_prob(self.state, self.action, self.action, actual[:, 0], self.goal),
        self.base.log_prob(self.state, self.action, self.action, actual[:, 0], self.goal))
    with self.assertRaises(ValueError):
      self.model.log_prob(self.state, -self.action, self.action, actual[:, 0], self.goal)
    off = self.model.sample(self.state, -self.action, self.action, key, 31, self.goal)
    np.testing.assert_array_equal(off[..., 2:], np.broadcast_to(
        np.asarray(self.state[:, None, :6]), (2, 31, 6)))

  def test_final_bound_geometry_shapes_and_nearby_actions(self):
    self.model.params['residual'] = jax.tree_util.tree_map(
        lambda p: p + 4., self.model.params['residual'])
    for distance in (0., 1e-6, 0.01, 0.4, 2.):
      x = jnp.clip(self.action + distance, -1, 1)
      values, detail = self.model.sample_with_diagnostics(
          self.state, x, self.action, jax.random.PRNGKey(11), 128, self.goal)
      self.assertEqual(values.shape, (2, 128, 8))
      self.assertTrue(np.isfinite(values).all())
      self.assertTrue(_open_endpoint(values[..., :2]).all())
      self.assertTrue(np.all(np.asarray(detail['emitted_change']) <=
                             np.asarray(detail['radius']) + 1e-7))
    one = self.model.sample(self.state[0], self.action[0], self.action[0],
                            jax.random.PRNGKey(0), goal=self.goal[0])
    self.assertEqual(one.shape, (8,))

  def test_marginalization_keeps_state_and_action_fixed_with_independent_draws(self):
    class NominalFixture:
      def sample(self, state, key, count, goal):
        return jax.random.uniform(key, (len(state), count, 2), minval=-1, maxval=1)
    nominal = NominalFixture()
    key = jax.random.PRNGKey(17)
    actual, _ = self.model.marginal_flat(self.model.params, nominal, self.state,
                                        self.action, self.goal, key, 8)
    ka, kt = jax.random.split(key)
    xp = nominal.sample(self.state, ka, 8, self.goal)
    direct, _ = self.model.sample_flat(
        self.model.params, jnp.repeat(self.state, 8, axis=0),
        jnp.repeat(self.action, 8, axis=0), xp.reshape(-1, 2),
        jnp.repeat(self.goal, 8, axis=0), kt, 1)
    np.testing.assert_array_equal(actual, direct.reshape(2, 8, 8))
    self.assertGreater(float(xp.std(axis=1).min()), 0.1)
    np.testing.assert_array_equal(actual[..., 2:],
                                  np.broadcast_to(self.state[:, None, :6], (2, 8, 6)))

  def test_collision_projection_that_breaks_bound_falls_back_to_anchor(self):
    # A .9-step diagonal anchor is safe. A short downward residual hits a
    # wall; endpoint rejection would move .9 back, exceeding the .8 radius.
    self.base._distribution = lambda params, context: TransitionDistribution(
        jnp.full((len(context),), -100.), jnp.zeros((len(context), 3)),
        jnp.broadcast_to(jnp.array([9., 0.]), (len(context), 3, 2)),
        jnp.zeros((len(context), 3, 2)))
    residual = self.model.params['residual']
    final = [name for name, values in residual.items() if values['b'].shape == (2,)][0]
    residual[final]['b'] = jnp.array([0., -100.])
    state = jnp.tile(jnp.array([2.5, 3.5]), 4)[None]
    xp = jnp.array([[0., .4 / .75]])
    action = -xp
    value, detail = self.model.sample_flat(self.model.params, state, action, xp,
                                          self.goal[:1], jax.random.PRNGKey(0), 8)
    self.assertTrue(np.asarray(detail['candidate_blocked']).all())
    self.assertTrue(np.asarray(detail['bound_fallback']).all())
    np.testing.assert_array_equal(value[..., :2], detail['anchor_xy'])
    self.assertTrue(_open_endpoint(value[..., :2]).all())

  def test_saturated_residual_has_no_roundoff_fallback_in_open_space(self):
    self.base._distribution = lambda params, context: TransitionDistribution(
        jnp.full((len(context),), 100.), jnp.zeros((len(context), 3)),
        jnp.zeros((len(context), 3, 2)), jnp.zeros((len(context), 3, 2)))
    residual = self.model.params['residual']
    final = [name for name, values in residual.items() if values['b'].shape == (2,)][0]
    residual[final]['b'] = jnp.array([100., 100.])
    current = jnp.stack([jnp.linspace(3.2, 5.5, 256), jnp.full(256, 3.3)], axis=-1)
    state = jnp.tile(current, (1, 4))
    xp = jnp.zeros((256, 2))
    action = jax.random.uniform(jax.random.PRNGKey(9), (256, 2), minval=.001, maxval=.1)
    value, detail = self.model.sample_flat(self.model.params, state, action, xp,
                                          jnp.tile(self.goal[:1], (256, 1)), jax.random.PRNGKey(0), 8)
    self.assertFalse(np.asarray(detail['bound_fallback']).any())
    self.assertTrue(np.all(np.asarray(detail['emitted_change']) <= np.asarray(detail['radius'])))
    self.assertTrue(_open_endpoint(value[..., :2]).all())

  def test_residual_gradient_matches_finite_difference(self):
    reference = jnp.tile(jnp.array([4., 3.5]), (6, 4))
    def objective(params):
      value, _ = self.model.sample_flat(params, self.state, -self.action,
                                         self.action, self.goal, jax.random.PRNGKey(3), 8)
      return conditional_mmd2(value, reference, 0., 1., [1., 2., 4.]).mean()
    gradient = jax.grad(objective)(self.model.params)
    leaves = jax.tree_util.tree_leaves(gradient['residual'])
    self.assertTrue(all(np.isfinite(v).all() for v in leaves))
    self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves), 1e-8)
    # Directional derivative probes the actual loss, not its implementation.
    direction = jax.tree_util.tree_map(lambda v: v / (jnp.linalg.norm(v) + 1e-8),
                                        gradient['residual'])
    def shifted(amount):
      params = dict(self.model.params)
      params['residual'] = jax.tree_util.tree_map(
          lambda p, d: p + amount*d, params['residual'], direction)
      return float(objective(params))
    numerical = (shifted(1e-3) - shifted(-1e-3)) / 2e-3
    analytic = sum(float(jnp.sum(g*d)) for g, d in zip(
        leaves, jax.tree_util.tree_leaves(direction)))
    np.testing.assert_allclose(numerical, analytic, rtol=.015, atol=2e-5)

  def test_discrete_sampling_does_not_claim_logit_gradients(self):
    distribution = TransitionDistribution(jnp.array([-1.]), jnp.zeros((1, 3)),
                                           jnp.ones((1, 3, 2)), jnp.ones((1, 3, 2)))
    derivative = jax.grad(lambda d: sample_displacement(
        d, jax.random.PRNGKey(3), 64, jnp.zeros(2), jnp.ones(2), self.base.spec)[0].sum())(distribution)
    np.testing.assert_array_equal(derivative.stationary_logit, 0.)
    np.testing.assert_array_equal(derivative.mixture_logits, 0.)
    self.assertGreater(float(jnp.linalg.norm(derivative.loc)), 0.)


if __name__ == '__main__':
  unittest.main(verbosity=2)
