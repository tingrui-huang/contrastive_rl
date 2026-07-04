"""Checkpointing for the contrastive RL port.

Saves the full TrainingState (params + optimizer state, as numpy) plus a JSON
metrics history, so a long Colab run survives disconnects and you keep a record
of success-rate-over-time. Designed to write to a Google Drive folder.

Layout under ``ckpt_dir``:
  latest.pkl    -- most recent full state (for --resume). Overwritten each save.
  best.pkl      -- state at the highest eval success so far.
  metrics.json  -- list of {step, success, ...} dicts, appended each eval.
"""
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def _to_numpy(tree):
  return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _to_jax(tree):
  return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


def save_checkpoint(ckpt_dir, step, state, metrics_history, success,
                    best_success):
  """Writes latest.pkl + metrics.json (+ best.pkl if success improved).

  Returns the (possibly updated) best_success.
  """
  if not ckpt_dir:
    return best_success
  os.makedirs(ckpt_dir, exist_ok=True)
  payload = {'step': int(step), 'state': _to_numpy(state)}

  # Write latest atomically-ish (temp then replace) to survive interruptions.
  latest = os.path.join(ckpt_dir, 'latest.pkl')
  tmp = latest + '.tmp'
  with open(tmp, 'wb') as f:
    pickle.dump(payload, f)
  os.replace(tmp, latest)

  with open(os.path.join(ckpt_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics_history, f, indent=2)

  if success is not None and success >= best_success:
    best_success = success
    with open(os.path.join(ckpt_dir, 'best.pkl'), 'wb') as f:
      pickle.dump(payload, f)
    print(f'    [ckpt] new best success={success:.3f} -> best.pkl')
  print(f'    [ckpt] saved step {step} -> {latest}', flush=True)
  return best_success


def load_checkpoint(path):
  """Loads a checkpoint file; returns (step, state_with_jax_arrays)."""
  with open(path, 'rb') as f:
    payload = pickle.load(f)
  return payload['step'], _to_jax(payload['state'])


def load_policy_params(path):
  """Convenience for eval/visualize: just the policy params (jax arrays)."""
  _, state = load_checkpoint(path)
  return state.policy_params
