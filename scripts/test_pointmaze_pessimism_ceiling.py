"""Static/unit checks for the one-shot PointMaze pessimism-ceiling diagnostic."""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ett.diagonal_transition import DiagonalTransitionSpec
import ett.pointmaze_pessimism_ceiling as ceiling
from ett.pointmaze_pessimism_ceiling import (
    CONFIG, _farthest_corner, _global_farthest, apply_arm, endpoint_valid, protocol_text)


def main():
    assert CONFIG['planned_main_outputs'] == (
        36 * 64 * 10 * 6 + 128 * 50 * 6)
    assert CONFIG['planned_probe_outputs'] == (
        128 * 8 * 6 + 128 * 16 * 2 * 6)
    assert CONFIG['planned_a0_a4_outputs'] == 207360 < CONFIG['a0_a4_output_cap']
    a5 = CONFIG['a5_protocol']
    assert a5['lookahead_outputs'] == 29440 * 5 * 8 * 3
    assert a5['total_including_a0_a4'] == 3774720 > CONFIG['a5_output_cap']

    state = jnp.asarray([[.5, 3.5, .5, 3.5, .5, 3.5, .5, 3.5],
                         [7.5, 3.5, 7.5, 3.5, 7.5, 3.5, 7.5, 3.5]], jnp.float32)
    goal = jnp.asarray([[8.5, 3.5], [8.5, 3.5]], jnp.float32)
    target, valid = _global_farthest(state, goal)
    assert np.asarray(valid).all()
    assert endpoint_valid(np.asarray(target)).all()
    assert np.max(np.abs(np.asarray(target) - np.asarray(state[:, :2]))) <= 1

    low = jnp.asarray([[[0., 3.], [7., 3.]]], jnp.float32)
    high = jnp.asarray([[[1., 4.], [8., 4.]]], jnp.float32)
    corner = np.asarray(_farthest_corner(low, high, jnp.asarray([[8.5, 3.5]])))
    np.testing.assert_array_equal(corner[0, 0], np.array([0., 3.], np.float32))
    np.testing.assert_array_equal(corner[0, 1], np.array([7., 3.], np.float32))

    ceiling._BASE_SPEC.value = DiagonalTransitionSpec()
    anchor = jnp.asarray([[[.75, 3.25]], [[7.25, 3.25]]], jnp.float32)
    xp = jnp.asarray([[0., 0.], [.2, -.2]], jnp.float32)
    goal8 = jnp.broadcast_to(jnp.asarray([8.5, 3.5] * 4, jnp.float32), state.shape)
    compiled = jax.jit(apply_arm)
    for code in range(5):
        diagonal, detail = compiled(jnp.int32(code), jnp.zeros(32), state, xp, xp, goal8, anchor)
        np.testing.assert_array_equal(np.asarray(diagonal[..., :2]), np.asarray(anchor))
        assert np.asarray(detail['valid']).all()
        off, detail = compiled(jnp.int32(code), jnp.zeros(32), state,
                               jnp.clip(xp + .25, -1, 1), xp, goal8, anchor)
        assert np.isfinite(np.asarray(off)).all() and np.asarray(detail['valid']).all()
        assert endpoint_valid(np.asarray(off[..., :2])).all()

    text = protocol_text()
    for required in ('207,360', '3,774,720', 'lower bounds', 'source times',
                     'Decision rules', 'A5'):
        assert required in text
    for source in ('ett/pointmaze_pessimism_ceiling.py',
                   'ett/report_pointmaze_pessimism_ceiling.py'):
        ast.parse(Path(source).read_text(encoding='utf-8'))
    print('pointmaze pessimism ceiling static/unit checks passed')


if __name__ == '__main__':
    main()
