"""Checkpointing for the Stage-2 behavior flow model.

Mirrors ``crl/checkpoint.py`` (pickle payload + JSON metrics sidecar, atomic
temp-then-replace write) but carries the Stage-2 provenance block instead of the
CRL TrainingState.

Layout under ``ckpt_dir``:
  latest.pkl    -- most recent {step, params, opt_state}
  best.pkl      -- state at the LOWEST validation flow-MSE so far
  config.json   -- run config + dataset/model provenance metadata
  metrics.json  -- list of {step, train_loss, val_loss, elapsed_s} dicts
"""
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def _to_numpy(tree):
  return jax.tree_util.tree_map(np.asarray, tree)


def _to_jax(tree):
  return jax.tree_util.tree_map(jnp.asarray, tree)


def _atomic_pickle(path, payload):
  tmp = path + '.tmp'
  with open(tmp, 'wb') as f:
    pickle.dump(payload, f)
  os.replace(tmp, path)


def save_metadata(ckpt_dir, metadata):
  """Write config.json -- the run's full provenance record."""
  os.makedirs(ckpt_dir, exist_ok=True)
  with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
    json.dump(metadata, f, indent=2)


def save_checkpoint(ckpt_dir, step, params, opt_state, metrics_history,
                    val_loss, best_val):
  """Write latest.pkl + metrics.json (+ best.pkl on strict improvement).

  ``best`` tracks the LOWEST validation flow-MSE (lower is better), so the
  comparison is inverted relative to crl/checkpoint.py's success rate.
  Returns the (possibly updated) best_val."""
  if not ckpt_dir:
    return best_val
  os.makedirs(ckpt_dir, exist_ok=True)
  payload = {'step': int(step), 'params': _to_numpy(params),
             'opt_state': _to_numpy(opt_state)}
  _atomic_pickle(os.path.join(ckpt_dir, 'latest.pkl'), payload)

  with open(os.path.join(ckpt_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics_history, f, indent=2)

  if val_loss is not None and val_loss < best_val:
    best_val = float(val_loss)
    _atomic_pickle(os.path.join(ckpt_dir, 'best.pkl'), payload)
    print(f'    [ckpt] new best val flow-MSE={best_val:.6f} -> best.pkl')
  return best_val


def save_milestone(ckpt_dir, step, params, opt_state):
  """Write ``checkpoint_<step>.pkl`` -- a named, never-overwritten snapshot.

  Same payload schema as latest/best, so every evaluation script loads it
  without modification. Used by continuation runs to pin the 100k/150k states
  independently of best/latest, which keep moving."""
  if not ckpt_dir:
    return
  os.makedirs(ckpt_dir, exist_ok=True)
  path = os.path.join(ckpt_dir, f'checkpoint_{int(step)}.pkl')
  _atomic_pickle(path, {'step': int(step), 'params': _to_numpy(params),
                        'opt_state': _to_numpy(opt_state)})
  print(f'    [ckpt] milestone -> {path}')


def load_checkpoint(path):
  """Returns (step, params, opt_state) with jax arrays."""
  with open(path, 'rb') as f:
    payload = pickle.load(f)
  return (payload['step'], _to_jax(payload['params']),
          _to_jax(payload.get('opt_state')))


def load_metadata(ckpt_dir):
  with open(os.path.join(ckpt_dir, 'config.json')) as f:
    return json.load(f)
