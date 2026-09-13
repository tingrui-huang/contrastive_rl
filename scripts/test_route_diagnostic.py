"""Zero-transition checks for the diagnostic's controller and path validator."""
import inspect
import unittest
import numpy as np
from ett.route_diagnostic import ROUTES, controller, passage_crossed, validate
from ett.rollout_return import GOAL


class RouteTests(unittest.TestCase):
    def test_threshold_clip_and_final(self):
        points = ROUTES['lower']
        for xy, index, expected_index, action in [
            ([.5, 3.5], 0, 0, [1, 0]),
            ([1.25, 3.5], 0, 1, [.25, -1]),
            ([1.249, 3.5], 0, 0, [.251, 0]),
            ([8.5, 3.5], 4, 4, [0, 0]),
            ([9., 4.], 4, 4, [-.5, -.5]),
        ]:
            a, i = controller(np.array(xy, np.float32), np.int32(index), points)
            self.assertEqual(int(i), expected_index)
            np.testing.assert_allclose(a, action, atol=1e-7, rtol=0)
        self.assertEqual(list(inspect.signature(controller).parameters), ['xy', 'index', 'waypoints'])

    def test_lower_passage_requires_crossing(self):
        self.assertFalse(passage_crossed(np.array([[1.5, 1.5], [7.5, 1.5]]), True))
        self.assertTrue(passage_crossed(np.array([[x+.5, 1.5] for x in range(8)]), True))
        self.assertFalse(passage_crossed(np.array([[x+.5, 3.5] for x in range(8)]), True))
        xy = np.array([[x+.5, 1.5] for x in range(8)])
        xy[4, 1] = 3.5
        self.assertFalse(passage_crossed(xy, True))

    def test_independent_reward_history_validation(self):
        # Fabricated arithmetic fixture only; no native or model transition.
        xy = np.tile([8.5, 3.5], (51, 1)).astype(np.float32)
        states = np.tile(xy, (1, 4))[None]
        actions = np.tile([-1., 0.], (1, 50, 1)).astype(np.float32)
        reward = np.ones((1, 50))
        record = dict(states=states, action=actions, waypoint=np.zeros((1, 50), int),
                      reward=reward, goals=np.tile(GOAL, (1, 51, 1)),
                      **{'return': reward @ (.95**np.arange(50))})
        validate(record, 'shortcut')
        record['reward'][0, 0] = 0
        with self.assertRaises(AssertionError):
            validate(record, 'shortcut')


if __name__ == '__main__':
    unittest.main()
