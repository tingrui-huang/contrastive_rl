"""Independent timing, partition, feature and model-input contract tests."""
import inspect
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ett.failure_scoring import (
    FEATURES, SCOPE, FailureScorer, labels_from_death_rows,
    scorer_episode_split, visible_features)
from scripts.diagnose_f4_death_observability import _make_probe


class FailureScoringTests(unittest.TestCase):
  def test_fatal_action_is_alive_and_first_returned_observation_is_dead(self):
    episode = np.repeat([0, 1, 2], 6)
    time = np.tile(np.arange(6), 3)
    result = labels_from_death_rows(episode, time, [2, -1, 6])
    np.testing.assert_array_equal(result['death_age'][:6], [-1, -1, 0, 1, 2, 3])
    np.testing.assert_array_equal(result['dies_during_transition'].reshape(3, 6).sum(1), [1, 0, 1])
    self.assertFalse(result['dead_when_observed'][1])
    self.assertFalse(result['dead_when_observed'][6:].any())

  def test_episode_partitions_preserve_upstream_holdout(self):
    split = scorer_episode_split(6600)
    combined = np.concatenate(list(split.values()))
    self.assertEqual(len(np.unique(combined)), 6600)
    self.assertEqual(len(combined), 6600)
    expected = np.sort(np.random.default_rng(0).permutation(6600)[:660])
    np.testing.assert_array_equal(split['test'], expected)
    for key, value in split.items():
      np.testing.assert_array_equal(value, scorer_episode_split(6600)[key])
    # Expanding complete episodes to transition rows cannot cross partitions.
    row_episode = np.repeat(np.arange(6600), 50)
    masks = [np.isin(row_episode, ids) for ids in split.values()]
    np.testing.assert_array_equal(np.stack(masks).sum(0), 1)

  def test_newest_first_motion_and_goal_order(self):
    s = jnp.asarray([[4., 8., 3., 6., 1., 3., 0., 0.]])
    g = jnp.arange(8.)[None]
    np.testing.assert_array_equal(visible_features(s, g, 'motion')[0, :6], [1, 2, 2, 3, 1, 3])
    for kind, width in zip(FEATURES, (2, 6, 8)):
      value = visible_features(s, g, kind)
      self.assertEqual(value.shape, (1, width+8))
      np.testing.assert_array_equal(value[0, -8:], g[0])
    translated = s+jnp.tile(jnp.array([2., -1.]), 4)
    np.testing.assert_array_equal(visible_features(s, g, 'motion'), visible_features(translated, g, 'motion'))
    self.assertFalse(np.array_equal(visible_features(s, g, 'f4'), visible_features(translated, g, 'f4')))

  def test_hidden_fields_cannot_enter_the_feature_api(self):
    self.assertEqual(tuple(inspect.signature(visible_features).parameters), ('state', 'goal', 'kind'))
    for kind in FEATURES:
      with self.assertRaises(ValueError):
        visible_features(jnp.zeros((2, 9)), jnp.zeros(8), kind)
      with self.assertRaises(TypeError):
        visible_features(jnp.zeros((2, 8)), jnp.zeros(8), kind, death_flag=True)

  def test_frozen_score_shapes_and_newest_gradient(self):
    net = _make_probe((8, 8))
    cp = {'scope': SCOPE, 'hidden_sizes': [8, 8], 'feature_name': 'f4',
          'params': net.init(jax.random.PRNGKey(0), jnp.ones((1, 16))),
          'mean': np.zeros(16), 'std': np.ones(16), 'fixed_commanded_goal': np.zeros(8)}
    model = FailureScorer(cp)
    for shape in ((8,), (3, 8), (2, 4, 8)):
      s = jnp.ones(shape)
      score = model.score(s)
      self.assertEqual(score.shape, shape[:-1])
      self.assertTrue(np.all((score >= 0) & (score <= 1)))
      self.assertTrue(np.isfinite(jax.grad(lambda v: model.score(v).sum())(s)).all())
    with self.assertRaises(ValueError):
      FailureScorer({**cp, 'scope': 'offline_objective'})


if __name__ == '__main__':
  unittest.main(verbosity=2)
