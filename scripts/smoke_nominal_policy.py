"""Fast API/gradient smoke test for the nominal bounded action density."""
import argparse
import json
import os
import pickle
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from propensity.nominal_policy import (
    NominalPolicySpec, censored_mixture_log_prob, load_nominal_policy,
    make_policy_network)


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--out-dir', default='artifacts/nominal_policy_smoke_api')
  args = parser.parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)

  seed = 41
  rng = np.random.default_rng(seed)
  context = rng.normal(size=(256, 16)).astype(np.float32)
  choose = rng.random(256) < 0.55
  loc = np.where(choose[:, None], np.array([0.9, 0.25]),
                 np.array([-0.65, -0.4]))
  action = np.clip(loc + 0.12 * rng.normal(size=(256, 2)), -1, 1).astype(np.float32)
  # Exercise all three likelihood cases: lower atom, interior density, upper atom.
  action[:3] = np.array([[-1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]], np.float32)

  spec = NominalPolicySpec(8, 8, 2, num_components=3,
                           hidden_sizes=(32, 32))
  network = make_policy_network(spec)
  key = jax.random.PRNGKey(seed)
  params = network.init(key, jnp.asarray(context[:1]))
  optimizer = optax.adam(1e-3)
  opt_state = optimizer.init(params)

  @jax.jit
  def update(p, o, c, a):
    def loss_fn(parameters):
      return -jnp.mean(censored_mixture_log_prob(
          network.apply(parameters, c), a, spec))
    loss, grad = jax.value_and_grad(loss_fn)(p)
    updates, o = optimizer.update(grad, o, p)
    return optax.apply_updates(p, updates), o, loss

  losses = []
  for _ in range(40):
    params, opt_state, loss = update(
        params, opt_state, jnp.asarray(context), jnp.asarray(action))
    losses.append(float(loss))
  assert np.isfinite(losses).all(), losses

  config = {
      'model': spec.asdict(),
      'context_normalization': {
          'mean': np.zeros(16).tolist(), 'std': np.ones(16).tolist()},
      'smoke': True,
  }
  with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)
  with open(os.path.join(args.out_dir, 'best.pkl'), 'wb') as f:
    pickle.dump({'params': jax.device_get(params), 'step': 40,
                 'validation_nll': losses[-1]}, f)

  model = load_nominal_policy(args.out_dir)
  single = np.asarray(model.sample(context[0], jax.random.PRNGKey(1)))
  multiple = np.asarray(model.sample(context[:5], jax.random.PRNGKey(2), 11))
  separated = np.asarray(model.sample(
      context[0, :8], jax.random.PRNGKey(3), 7, goal=context[0, 8:]))
  log_prob = np.asarray(model.log_prob(context[:8], action[:8]))
  assert single.shape == (2,)
  assert multiple.shape == (5, 11, 2)
  assert separated.shape == (7, 2)
  assert np.isfinite(log_prob).all()
  assert (multiple >= -1).all() and (multiple <= 1).all()
  assert (separated >= -1).all() and (separated <= 1).all()
  # Also validate the inexpensive comparison architecture.
  one = NominalPolicySpec(8, 8, 2, num_components=1, hidden_sizes=(16,))
  one_net = make_policy_network(one)
  one_params = one_net.init(jax.random.PRNGKey(4), jnp.asarray(context[:2]))
  one_lp = censored_mixture_log_prob(
      one_net.apply(one_params, jnp.asarray(context[:2])),
      jnp.asarray(action[:2]), one)
  assert np.isfinite(np.asarray(one_lp)).all()

  result = {
      'pass': True, 'loss_first': losses[0], 'loss_last': losses[-1],
      'single_shape': list(single.shape), 'multiple_shape': list(multiple.shape),
      'separate_state_goal_shape': list(separated.shape),
      'sample_min': multiple.min(axis=(0, 1)).tolist(),
      'sample_max': multiple.max(axis=(0, 1)).tolist(),
      'haiku_version': hk.__version__, 'jax_backend': jax.default_backend(),
  }
  with open(os.path.join(args.out_dir, 'smoke_report.json'), 'w') as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write('\n')
  print(json.dumps(result, indent=2, sort_keys=True))
  return result


if __name__ == '__main__':
  main()
