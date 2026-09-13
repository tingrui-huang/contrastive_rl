"""Deterministic component fixtures only: no training or trajectory generation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from ett.convex_action_transition import (
    ConvexActionTransition, emit, emit_with_geometry, fork_segment, project_segment,
    RECTANGLE_HIGH, RECTANGLE_LOW,
)
from ett.diagonal_transition import DiagonalTransitionSpec

BASELINE = Path('artifacts/ett_convex_set_component/fork_segment_v1/pre_change/convex_action_transition.py')
TOL = 2e-6


def historical_module():
    spec = importlib.util.spec_from_file_location('historical_convex_transition', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures():
    q = np.array([[1.5, 3.5], [1., 3.], [1.5, 3.5], [2., 3.5],
                  [1.5, 4.], [1.5, 3.5], [1.5, 3.5], [1.5, 3.5],
                  [1.5, np.nextafter(np.float32(4), np.float32(0))]], np.float32)
    a = np.array([[2.4, 3.4], [1., 3.], [2.4, 3.], [2.4, 3.4],
                  [1.5, 3.5], [1.5, 2.5], [RECTANGLE_HIGH[2, 0], 3.],
                  [2.4, 2.5], [1.5, RECTANGLE_HIGH[0, 1]]], np.float32)
    return np.tile(q, (1, 4)), a[:, None]


def check_geometry(test, state, output, detail):
    xy = np.asarray(output)[..., :2]
    a, b = np.asarray(detail['segment_start']), np.asarray(detail['segment_end'])
    chosen = np.asarray(detail['segment_available'])
    fraction = np.asarray(detail['segment_fraction'])
    expected = a + fraction[..., None] * (b - a)
    np.testing.assert_allclose(xy[chosen], expected[chosen], atol=TOL, rtol=0)
    test.assertTrue(np.all((fraction >= 0) & (fraction <= 1)))
    free = np.any(np.all((xy[..., None, :] >= RECTANGLE_LOW - TOL) &
                         (xy[..., None, :] <= RECTANGLE_HIGH + TOL), axis=-1), axis=-1)
    test.assertTrue(free.all())
    test.assertTrue(np.all(xy >= (state[:, None, :2] - 1.) - TOL))
    test.assertTrue(np.all(xy <= (state[:, None, :2] + 1.) + TOL))
    np.testing.assert_array_equal(np.asarray(output)[..., 2:],
                                  np.broadcast_to(state[:, None, :6], output.shape[:-1] + (6,)))


class ConvexSetChecks(unittest.TestCase):
    def test_historical_default_and_explicit_rectangle(self):
        historical = historical_module()
        s, anchor = fixtures()
        action = np.tile([-.75, -1.], (len(s), 1)).astype(np.float32)
        xp = -action
        theta = np.arange(32, dtype=np.float32) - 12
        old = jax.jit(historical.emit)(theta, s, action, xp, anchor)
        for fn in [emit, emit_with_geometry]:
            new = jax.jit(fn)(theta, s, action, xp, anchor)
            self.assertEqual(old[1].keys(), new[1].keys())
            for x, y in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
                np.testing.assert_array_equal(x, y)

    def test_selector_boundary_cases_and_degenerate_projection(self):
        s, anchor = fixtures()
        end, available = fork_segment(s, anchor)
        np.testing.assert_array_equal(available[:, 0], [True, True, False, False, False,
                                                       False, True, False, False])
        np.testing.assert_array_equal(np.asarray(end)[~np.asarray(available)], anchor[~np.asarray(available)])
        xy, fraction = project_segment(anchor + 1., anchor, anchor)
        np.testing.assert_array_equal(xy, anchor)
        np.testing.assert_array_equal(fraction, np.zeros(anchor.shape[:-1]))
        # The uncovered anchor must stay invalid in either mode.
        for mode in ['rectangle', 'fork_segment']:
            y, d = emit_with_geometry(np.zeros(32), s, s[:, :2]*0, s[:, :2]*0, anchor, geometry_mode=mode)
            self.assertFalse(bool(d['box_valid'][7, 0]))
            self.assertTrue(np.isnan(y[7, 0, :2]).all())

    def test_diagonal_multiple_draws_and_f4(self):
        s, a = fixtures()
        s, a = s[:3], a[:3]
        a = np.repeat(a, 3, axis=1)
        a[:, 1, 0] -= .125
        xp = np.tile([.25, -.5], (len(s), 1)).astype(np.float32)
        for mode in ['rectangle', 'fork_segment']:
            output, _ = emit_with_geometry(np.arange(32, dtype=np.float32), s, xp, xp, a, geometry_mode=mode)
            np.testing.assert_array_equal(output[..., :2], a)
            np.testing.assert_array_equal(output[..., 2:], np.broadcast_to(s[:, None, :6], (3, 3, 6)))

    def test_fixed_selection_and_action_pair_bound(self):
        s, a = fixtures()
        s, a = s[[0, 1, 2, 3, 5, 6]], a[[0, 1, 2, 3, 5, 6]]
        grid = np.array([(x, y) for x in np.linspace(-1, 1, 5) for y in np.linspace(-1, 1, 5)] +
                        [(.25, -.5), (.250001, -.499999)], np.float32)
        xp = np.tile([.25, -.5], (len(s), 1)).astype(np.float32)
        theta = np.tile(np.array([[0., 0.], [0., 1.]], np.float32).ravel(), 8)
        for mode in ['rectangle', 'fork_segment']:
            outputs, selected = [], None
            for action in grid:
                y, d = emit_with_geometry(theta, s, np.broadcast_to(action, xp.shape), xp, a, geometry_mode=mode)
                outputs.append(np.asarray(y))
                if mode == 'fork_segment':
                    current = [d[k] for k in ['segment_start', 'segment_end', 'segment_available']]
                    if selected is not None:
                        for before, after in zip(selected, current):
                            np.testing.assert_array_equal(before, after)
                    selected = current
                    check_geometry(self, s, y, d)
            outputs = np.asarray(outputs, dtype=np.float64)
            differences = np.linalg.norm(outputs[:, None] - outputs[None, :], axis=-1)
            inputs = np.linalg.norm(grid.astype(float)[:, None] - grid.astype(float)[None, :], axis=-1)
            self.assertLessEqual(float(np.max(differences - inputs[..., None, None])), TOL)

    def test_geometry_endpoint_certificate(self):
        s, a = fixtures()
        end, available = fork_segment(s, a)
        for q, start, stop in zip(s[np.asarray(available[:, 0]), :2], a[np.asarray(available[:, 0]), 0],
                                  np.asarray(end)[np.asarray(available[:, 0]), 0]):
            start, stop = start.astype(float), stop.astype(float)
            if start[0] > float(RECTANGLE_HIGH[2, 0]):
                cross_x = start[0] + (3-start[1])*(stop[0]-start[0])/(stop[1]-start[1])
                self.assertLessEqual(cross_x, 1.9375)
                self.assertGreaterEqual(cross_x, 1.)
            points = start + np.linspace(0, 1, 101)[:, None]*(stop-start)
            self.assertTrue(np.any(np.all((points[:, None] >= RECTANGLE_LOW) &
                                         (points[:, None] <= RECTANGLE_HIGH), axis=-1), axis=-1).all())
            self.assertTrue(np.all(points >= q-1.))
            self.assertTrue(np.all(points <= q+1.))

    def test_public_sampler_uses_same_diagonal_context_and_draws(self):
        # Stub only the stochastic displacement draw, using exact dyadic fixtures.
        # Exercise the real base projector, F4 shift, public batching and JIT.
        s = np.tile([1.5, 3.5], 4).astype(np.float32)[None]
        xp = np.array([[.5, -.25]], np.float32)
        action = np.array([[-.5, -1.]], np.float32)
        goal = np.tile([8.5, 3.5], 4).astype(np.float32)[None]
        seen = []
        def distribution(params, context):
            return context
        def draw(context, key, count, mean, std, spec):
            # Checked outside JIT by its deterministic numeric return as well.
            seen.append(context)
            delta = jnp.broadcast_to(jnp.array([.75, -.125]), (len(context), count, 2))
            return delta, jnp.zeros((len(context), count), dtype=bool)
        base = SimpleNamespace(spec=DiagonalTransitionSpec(), params={}, delta_mean=0., delta_std=1.,
                               _distribution=distribution)
        theta = np.tile([0., 0., 0., 1.], 8).astype(np.float32)
        with patch('ett.convex_action_transition.sample_displacement', draw):
            for mode in ['rectangle', 'fork_segment']:
                model = ConvexActionTransition(base, theta, geometry_mode=mode)
                _, _ = model.sample_flat(theta, jnp.array(s), jnp.array(action), jnp.array(xp), jnp.array(goal), jax.random.PRNGKey(0), 2)
                np.testing.assert_array_equal(seen[-1], np.concatenate([s, xp, xp, goal], axis=-1))
                off, d = model.sample_with_diagnostics(s, action, xp, jax.random.PRNGKey(0), 2, goal)
                diag, dd = model.sample_with_diagnostics(s, xp, xp, jax.random.PRNGKey(0), 2, goal)
                np.testing.assert_array_equal(diag[..., :2], dd['anchor_xy'])
                np.testing.assert_array_equal(d['anchor_xy'], dd['anchor_xy'])
                np.testing.assert_array_equal(off[..., 2:], np.broadcast_to(s[:, None, :6], (1, 2, 6)))
                expected, _ = emit_with_geometry(theta, s, action, xp, d['anchor_xy'], geometry_mode=mode)
                np.testing.assert_allclose(off, expected, atol=TOL, rtol=0)
                one = model.sample(s, action, xp, jax.random.PRNGKey(0), goal=goal)
                self.assertEqual(one.shape, (1, 8))
                with self.assertRaises(AttributeError):
                    model.geometry_mode = 'rectangle'
        with self.assertRaises(ValueError):
            ConvexActionTransition(base, geometry_mode='nonconvex_union')


if __name__ == '__main__':
    unittest.main()
