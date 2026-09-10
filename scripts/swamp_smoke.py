"""CPU smoke for PointMaze swamp defaults, hiddenness and blind survival.

Run: python -m scripts.swamp_smoke
F4 behavior audit: python -m scripts.audit_swamp_windy_f4 --episodes 300
"""
import itertools
import json

import numpy as np

from crl import envs
from crl.config import Config


def main():
  report = {}
  variants = {
      'matched_v0': (2, envs.TwoRouteSwampMatchedEnv),
      'windy_v0': (2, envs.TwoRouteSwampWindyEnv),
      'windy_z_v0': (3, envs.TwoRouteSwampWindyZEnv),
      'windy_z_v1': (3, envs.TwoRouteSwampWindyZV1Env),
      'windy_f4_v0': (8, envs.TwoRouteSwampWindyF4Env),
  }
  for variant, (dim, cls) in variants.items():
    name = f'point_two_route_swamp_{variant}'
    cfg = Config(env_name=name)
    env = envs.make_env(name, cfg, seed=20260910)
    assert env.active_prob == 0.30
    assert cls().active_prob == 0.30
    assert (cfg.obs_dim, cfg.goal_dim, cfg.action_dim) == (dim, dim, 2)
    obs = env.reset()
    assert obs.shape == (2 * dim,) and obs.dtype == np.float32
    for bits in itertools.product((0, 1), repeat=3):
      env.set_swamp(bits)
      np.testing.assert_array_equal(env._get_obs(), obs)

    draws = []
    for _ in range(20000):
      env.reset()
      draws.append(env.swamp_bits)
    draws = np.asarray(draws)
    rates = draws.mean(axis=0)
    clear_rate = float((~draws.any(axis=1)).mean())
    assert np.all(np.abs(rates - 0.30) < 0.015), rates
    assert abs(clear_rate - 0.343) < 0.015, clear_rate
    corr = np.corrcoef(draws.T)
    assert np.max(np.abs(corr[~np.eye(3, dtype=bool)])) < 0.03
    report[variant] = dict(per_cell_rates=rates.tolist(),
                           all_clear_rate=clear_rate, hidden_obs=True)

  # Exactly one landing in each swamp cell, with no action noise or bit access
  # by the controller. Death must be absorbing until reset for every mask.
  env = envs.TwoRouteSwampWindyEnv(action_noise=0.0, seed=20260910)
  for bits in itertools.product((0, 1), repeat=3):
    env.set_auto_resample(False)
    env.reset()
    env.set_swamp(bits)
    for _ in range(8):
      obs, reward, done, info = env.step([1.0, 0.0])
      assert not done and info == {}
    assert env.dead == any(bits)
    if env.dead:
      death_obs = obs.copy()
      for _ in range(3):
        obs, reward, done, info = env.step([-1.0, -1.0])
        np.testing.assert_array_equal(obs, death_obs)
        assert reward == 0.0 and not done and info == {}
    else:
      assert np.linalg.norm(env.state - env.goal) < 0.5

  env.set_auto_resample(True)
  successes = 0
  n = 2000
  for _ in range(n):
    env.reset()
    for _ in range(8):
      env.step([1.0, 0.0])
      if env.dead:
        break
    successes += int(not env.dead and np.linalg.norm(env.state - env.goal) < 0.5)
  rate = successes / n
  assert abs(rate - 0.343) < 0.04, rate
  report['windy_v0'].update(blind_episodes=n, blind_success=rate,
                         theory=0.343, forced_masks_passed=8)

  # The existing safe route still works even with every swamp cell active.
  env.set_auto_resample(False)
  env.set_swamp([1, 1, 1])
  env.reset()
  waypoints = ([1.5, 3.5], [1.5, 1.5], [7.5, 1.5], [7.5, 3.5], [8.5, 3.5])
  for target in waypoints:
    for _ in range(10):
      if np.linalg.norm(env.state - target) < 1e-6:
        break
      env.step(np.clip(np.asarray(target) - env.state, -1, 1))
      assert not env.dead
  assert np.linalg.norm(env.state - env.goal) < 0.5
  report['windy_v0']['safe_route_all_active'] = True

  # Preserve explicit overrides and the original strong reference default.
  for _, cls in variants.values():
    for p in (0.0, 0.1, 1.0):
      env = cls(active_prob=p)
      assert env.active_prob == p
      if p in (0.0, 1.0):
        assert np.all(env.swamp_bits == bool(p))
  assert envs.TwoRouteSwampEnv().active_prob == 0.2
  print(json.dumps(report, indent=2))
  print('PASS: defaults, sampling, hiddenness, blind shortcut, death, safe route, overrides')


if __name__ == '__main__':
  main()
