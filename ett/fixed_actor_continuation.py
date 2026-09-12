"""Fresh native-simulator continuation evaluation; no model fitting.

Opaque snapshots are confined to the simulator layer. Selection and scoring
receive observations, actions, parent identifiers and external time only.
"""
import copy
import numpy as np

from crl.envs import TwoRouteSwampWindyF4Env


class TimedSimulator:
    """Enforce the native task's external 50-step collection contract."""

    def __init__(self, seed):
        self.env = TwoRouteSwampWindyF4Env(seed=seed, active_prob=.3)
        self.elapsed = 0

    def observation(self):
        return self.env._get_obs().copy()

    def snapshot(self):
        # Full instance state includes geometry, RNG, hidden bits, absorption,
        # frame stack and configuration; the caller owns the elapsed counter.
        return copy.deepcopy((self.env, self.elapsed))

    @classmethod
    def restore(cls, snapshot, future_seed=None):
        result = cls.__new__(cls)
        result.env, result.elapsed = copy.deepcopy(snapshot)
        if future_seed is not None:
            # Do not reset or resample current bits or absorption here.
            result.env._rng = np.random.default_rng(future_seed)
        return result

    def require_horizon(self, horizon=20):
        if self.elapsed+horizon > self.env.max_episode_steps:
            raise ValueError('continuation exceeds original episode time limit')

    def step(self, action):
        self.require_horizon(1)
        action = np.asarray(action, np.float32)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('expected a bounded finite execution action')
        before = self.observation()
        observation, reward, done, _ = self.env.step(action)
        self.elapsed += 1
        if done:
            raise RuntimeError('unexpected native termination in fixed-length task')
        np.testing.assert_array_equal(observation[2:8], before[:6])
        np.testing.assert_array_equal(observation[8:], before[8:])
        visible_reward = float(np.linalg.norm(observation[:2].astype(float)-observation[8:10])<2)
        if visible_reward != reward:
            raise RuntimeError('native reward and visible reconstruction disagree')
        return observation, reward, self.elapsed == self.env.max_episode_steps


def observable_strata(observation):
    xy = observation[:, :2]
    distance = np.linalg.norm(xy-observation[:, 8:10], axis=1)
    region = np.where(xy[:, 0]<3, 0, np.where(xy[:, 0]>=6, 2,
                      np.where((xy[:, 1]>=3)&(xy[:, 1]<4), 1, 3)))
    motion = np.sqrt(np.mean(np.sum(np.diff(observation[:, :8].reshape(-1,4,2), axis=1)**2, axis=2), axis=1))
    return dict(region=region, progress=np.where(distance<2, 2, np.where(distance<6, 1, 0)),
                motion=np.where(motion<.02, 0, np.where(motion<.35, 1, 2)))


def select_contexts(observation, parent, seed=12002001):
    """Two contexts per parent, inverse observable-cell frequency weighting.

    All candidates are fixed time-index positions of natural trajectories.
    No outcome, scorer or hidden simulator value enters this function.
    """
    strata = observable_strata(observation)
    cells = np.stack(list(strata.values()), axis=1)
    _, cell, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    weights = np.where(strata['progress']==2, 1., 4.) / counts[cell]
    rng = np.random.default_rng(seed)
    selected = []
    for p in np.unique(parent):
        rows = np.flatnonzero(parent==p)
        selected.extend(rng.choice(rows, size=min(2,len(rows)), replace=False,
                                   p=weights[rows]/weights[rows].sum()).tolist())
    return np.sort(selected)


def coverage_action(source, observation, timestep, rng, memo):
    """Visible-only uniform or predeclared waypoint coverage; never a teacher."""
    if source == 3:
        return rng.uniform(-1,1,2).astype(np.float32)
    points = ([[2.5,3.5],[7.5,3.5],[8.5,3.5]] if source==1 else
              [[1.5,3.5],[1.5,1.5],[7.5,1.5],[7.5,3.5],[8.5,3.5]])
    index = memo.setdefault('waypoint', 0)
    if np.linalg.norm(observation[:2]-points[index])<.3 and index<len(points)-1:
        index += 1
        memo['waypoint'] = index
    # Predetermined pauses supply valid stationary histories without inspecting
    # absorption. Speed alternates by parent, not by hidden state or outcomes.
    if timestep in (5,6,13):
        action = np.zeros(2)
    else:
        action = np.clip(np.asarray(points[index])-observation[:2], -memo['speed'], memo['speed'])
    return np.clip(action+rng.normal(0,.12,2), -1,1).astype(np.float32)


def return_summary(values):
    values = np.asarray(values)
    maximum = np.sum(.95**np.arange(20))
    return dict(count=values.size, mean=float(values.mean()), std=float(values.std()),
                quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist(),
                zero_fraction=float(np.mean(values==0)),
                maximum_fraction=float(np.mean(np.isclose(values,maximum,atol=1e-10,rtol=0))))
