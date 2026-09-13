"""Unit checks for consistent Frobenius/spectral response constraints."""
import jax
import jax.numpy as j
import numpy as np

from ett.pointmaze_norm_comparison import (CONFIG, MODES, block_norms, emit_mode,
    matrices_mode, project_response)
from ett.rollout_return import GOAL, task_reward


def main():
    parts = ['training_visitation_outputs', 'optimizer_probe_outputs',
             'matched_audit_outputs', 'heldout_evaluation_outputs',
             'reset_evaluation_outputs', 'constraint_outputs']
    assert sum(CONFIG[key] for key in parts) == CONFIG['hard_cap'] == 2101824
    assert CONFIG['meaningful_return_effect'] == .01

    identity = np.tile(np.eye(2, dtype=np.float32), (8, 1, 1)).ravel()
    frobenius, f_info = project_response(identity, 'frobenius')
    spectral, s_info = project_response(identity, 'spectral')
    expected = np.tile(np.eye(2) / np.sqrt(2), (8, 1, 1))
    np.testing.assert_allclose(frobenius.reshape(8, 2, 2), expected, atol=1e-7)
    np.testing.assert_array_equal(spectral, identity)
    assert f_info['active_count'] == 8 and s_info['active_count'] == 0
    assert block_norms(spectral, 'spectral').max() == 1
    assert np.linalg.svd(np.asarray(matrices_mode(identity, 'spectral')),
                         compute_uv=False)[..., 0].max() <= 1

    state = j.asarray([[1.5, 3.5] * 4], j.float32)
    anchor = j.asarray([[[1.5, 3.5]]], j.float32)
    xp = j.asarray([[0., 0.]], j.float32)
    theta = j.asarray(np.linspace(-1.5, 1.5, 32), j.float32)
    for mode in MODES:
        diagonal, detail = jax.jit(lambda action: emit_mode(
            theta, state, action, xp, anchor, mode))(xp)
        np.testing.assert_array_equal(np.asarray(diagonal[..., :2]), np.asarray(anchor))
        assert np.asarray(detail['box_valid']).all()
        actions = [j.asarray([[x, y]], j.float32) for x, y in [(-1., -1.), (0., 0.), (1., 1.)]]
        outputs = [np.asarray(emit_mode(theta, state, action, xp, anchor, mode)[0]) for action in actions]
        for i in range(len(actions)):
            for k in range(i):
                distance = np.linalg.norm(outputs[i][..., :2] - outputs[k][..., :2], axis=-1)
                allowed = np.linalg.norm(np.asarray(actions[i] - actions[k]), axis=-1)
                assert np.max(distance - allowed) <= 2e-6

    next_state = np.asarray([[8.5, 3.5] * 4, [6.5, 3.5] * 4], np.float32)
    goal = np.broadcast_to(GOAL, next_state.shape)
    np.testing.assert_array_equal(np.asarray(task_reward(next_state, goal)), np.array([1., 0.], np.float32))
    print('Frobenius/spectral ETT comparison unit checks passed')


if __name__ == '__main__':
    main()
