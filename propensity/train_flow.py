"""Stage 2 trainer: conditional flow matching for the behavior policy mu(a|c).

Trains ``v_omega(c, x_t, t)`` on the FROZEN offline dataset only, using the
Stage-1 loader and its deterministic episode-level train/validation split. The
decision context is ``c = concat(s, g_cmd)`` -- the pre-action commanded goal,
audited in the Stage-2 report; NOT a contrastive future-goal relabel.

No environment is constructed, no rollout is performed, and nothing is written
back into the dataset. There is no discriminator, no causal weighting and no
worst-case branch here.

Run (defaults are conservative; see --help):

  python -m propensity.train_flow \
      --dataset artifacts/rockfall_v2_p30_h800_resetfix/pilot/antmaze_rockfall_v2_p30_h800_resetfix_pilot.npz \
      --ckpt-dir artifacts/propensity_flow/rockfall_v2_p30_h800_smoke \
      --steps 3000
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from propensity import checkpoint as ckpt_mod
from propensity import flow as flow_mod
from propensity.dataset import BehaviorDataset


def build_parser():
  p = argparse.ArgumentParser(
      description='Stage 2: train a conditional flow-matching model of the '
                  'offline behavior action distribution mu(a | s, g_cmd).')
  # data
  p.add_argument('--dataset', required=True,
                 help='frozen offline .npz (crl/offline_audit.py contract)')
  p.add_argument('--context-mode', choices=('context', 'state_only'),
                 default='context',
                 help="'context' (default) conditions on c=(s, g_cmd) = the "
                      "full stored obs; 'state_only' conditions on s alone.")
  p.add_argument('--val-frac', type=float, default=0.1)
  p.add_argument('--split-seed', type=int, default=0,
                 help='Stage-1 episode-level split seed (default 0).')
  # optimization
  p.add_argument('--seed', type=int, default=0, help='training/init seed')
  p.add_argument('--batch-size', type=int, default=256,
                 help='matches crl/config.py Config.batch_size')
  p.add_argument('--learning-rate', type=float, default=3e-4,
                 help='matches crl/config.py Config.learning_rate')
  p.add_argument('--steps', type=int, default=50_000,
                 help='gradient steps. With --resume-from this is the FINAL '
                      'GLOBAL step target, not an additional count.')
  # --- continuation -------------------------------------------------------- #
  p.add_argument('--resume-from', default=None,
                 help='checkpoint .pkl to continue from. Restores params AND '
                      'the full Optax state (Adam mu/nu/count), so the '
                      'optimization trajectory continues rather than '
                      'restarting. NOTE: the checkpoint carries no PRNG state, '
                      'so the minibatch/noise stream restarts from '
                      '--stream-seed; draws are i.i.d. so this is a '
                      'statistically equivalent, not bit-identical, '
                      'continuation.')
  p.add_argument('--stream-seed', type=int, default=None,
                 help='seed for the minibatch + flow-noise stream during a '
                      'continuation (default: --seed). Recorded in metadata.')
  p.add_argument('--resume-metrics', default=None,
                 help='metrics.json from the resumed run, prepended so the '
                      'loss trajectory stays continuous.')
  p.add_argument('--milestone-steps', default='',
                 help='comma-separated global steps at which to save '
                      'checkpoint_<step>.pkl (e.g. 100000,150000).')
  p.add_argument('--eval-every', type=int, default=1_000)
  p.add_argument('--log-every', type=int, default=500)
  # model
  p.add_argument('--hidden-sizes', default='256,256,256')
  p.add_argument('--time-features', type=int, default=32,
                 help='sinusoidal time-embedding width; 0 = raw scalar t')
  p.add_argument('--layer-norm', action='store_true')
  # sampling (recorded in metadata; used by eval_flow)
  p.add_argument('--flow-steps', type=int, default=flow_mod.DEFAULT_FLOW_STEPS,
                 help='Euler steps for sampling (default 10, aligned with the '
                      'official CFQL implementation). Sampling-only: the '
                      'training objective does not use it.')
  # validation protocol
  p.add_argument('--val-batches', type=int, default=20,
                 help='fixed validation batches; (x0, t) are drawn ONCE with '
                      '--val-noise-seed so the val curve is a deterministic '
                      'function of the parameters.')
  p.add_argument('--val-noise-seed', type=int, default=12345)
  # io
  p.add_argument('--ckpt-dir', default='')
  return p


def _fixed_val_batches(ds, batch_size, n_batches, noise_seed, action_dim):
  """Freeze a validation set AND its (x0, t) draws once, up front."""
  rng = np.random.default_rng(noise_seed)
  key = jax.random.PRNGKey(noise_seed)
  batches = []
  pool = ds._indices('val')                                   # noqa: SLF001
  if pool.size == 0:
    return batches
  take = min(batch_size, pool.size)
  for _ in range(n_batches):
    rows = pool[rng.integers(0, pool.size, size=take)]
    ctx = jnp.asarray(ds._state[rows])                        # noqa: SLF001
    act = jnp.asarray(ds._action[rows])                       # noqa: SLF001
    key, k0, k1 = jax.random.split(key, 3)
    x0 = jax.random.normal(k0, (take, action_dim))
    t = jax.random.uniform(k1, (take, 1))
    batches.append((ctx, act, x0, t))
  return batches


def main(argv=None):
  args = build_parser().parse_args(argv)
  t_start = time.time()

  # ---- data (Stage 1; frozen, episode-level split) ------------------------- #
  state_mode = 'obs' if args.context_mode == 'context' else 'state'
  ds = BehaviorDataset(args.dataset, val_frac=args.val_frac,
                       seed=args.split_seed, state_mode=state_mode,
                       split_level='episode')
  ok, gates, details = ds.check()
  if not ok:
    print('Stage-1 dataset gates FAILED:', gates)
    return 1
  context_dim, action_dim = ds.context_dim, ds.action_dim

  print('=' * 74)
  print('propensity.train_flow -- Stage 2 behavior flow matching')
  print('=' * 74)
  print(f'dataset        {args.dataset}')
  print(f'sha256         {ds.fingerprint["sha256"][:32]}...')
  print(f'context mode   {ds.context_mode}  '
        f'(c = s[{ds._obs_dim}] | g_cmd[{ds._goal_dim}] -> {context_dim})')
  print(f'action_dim     {action_dim}')
  print(f'transitions    {ds.n_transitions}  '
        f'(train {ds.n_train} / val {ds.n_val}, split seed {ds.split_seed})')
  print(f'split disjoint index_overlap={details["index_overlap"]} '
        f'episode_overlap={details["episode_overlap"]}')
  print()

  # ---- model -------------------------------------------------------------- #
  hidden = tuple(int(x) for x in args.hidden_sizes.split(',') if x)
  fcfg = flow_mod.FlowConfig(context_dim=context_dim, action_dim=action_dim,
                             hidden_sizes=hidden,
                             time_features=args.time_features,
                             use_layer_norm=args.layer_norm)
  net = flow_mod.make_flow_network(fcfg)

  key = jax.random.PRNGKey(args.seed)
  key, init_key = jax.random.split(key)
  params = net.init(init_key,
                    jnp.zeros((1, context_dim), jnp.float32),
                    jnp.zeros((1, action_dim), jnp.float32),
                    jnp.zeros((1, 1), jnp.float32))
  n_params = int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))
  print(f'velocity net   MLP{hidden} -> {action_dim}  '
        f'| time_features={args.time_features} | params={n_params}')
  print(f'optimizer      adam(lr={args.learning_rate})  batch={args.batch_size}'
        f'  steps={args.steps}  seed={args.seed}')
  print()

  opt = optax.adam(args.learning_rate)
  opt_state = opt.init(params)

  # ---- continuation: restore params AND optimizer state -------------------- #
  start_step = 0
  resume_info = None
  if args.resume_from:
    r_step, r_params, r_opt = ckpt_mod.load_checkpoint(args.resume_from)
    if r_opt is None:
      print(f'ABORT: {args.resume_from} has no optimizer state. Refusing to '
            'continue with a zero-initialized Adam -- that would be a restart '
            'wearing a continuation label.')
      return 1
    # Structural check: the restored trees must match the freshly built ones.
    if (jax.tree_util.tree_structure(r_params)
        != jax.tree_util.tree_structure(params)):
      print('ABORT: checkpoint parameter tree does not match this config.')
      return 1
    if (jax.tree_util.tree_structure(r_opt)
        != jax.tree_util.tree_structure(opt_state)):
      print('ABORT: checkpoint optimizer tree does not match optax.adam '
            f'(lr={args.learning_rate}).')
      return 1
    adam_count = int(jax.tree_util.tree_leaves(r_opt)[0])
    params, opt_state, start_step = r_params, r_opt, int(r_step)
    resume_info = {
        'resumed_from': os.path.abspath(args.resume_from),
        'resumed_at_step': start_step,
        'adam_count_restored': adam_count,
        'optimizer_state_restored': True,
        'param_tree_matched': True,
        'prng_state_in_checkpoint': False,
        'stream_continuation': 'restarted from stream_seed; minibatch and '
                               '(x0, t) draws are i.i.d., so this is a '
                               'statistically equivalent continuation, NOT a '
                               'bit-identical one',
    }
    print(f'  [resume] {args.resume_from}')
    print(f'  [resume] step={start_step}  adam count={adam_count}  '
          f'optimizer state RESTORED (not reinitialized)')
    if adam_count != start_step:
      print(f'  [resume] WARNING: adam count {adam_count} != step {start_step}')
    if args.steps <= start_step:
      print(f'ABORT: --steps {args.steps} <= resumed step {start_step}.')
      return 1

  @jax.jit
  def update(params, opt_state, context, action, key):
    def loss_fn(p):
      loss, _ = flow_mod.flow_matching_loss(net.apply, p, context, action, key)
      return loss
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

  @jax.jit
  def val_loss_fn(params, context, action, x0, t):
    return flow_mod.flow_matching_loss_fixed(net.apply, params, context,
                                             action, x0, t)

  val_batches = _fixed_val_batches(ds, args.batch_size, args.val_batches,
                                   args.val_noise_seed, action_dim)

  def evaluate(params):
    if not val_batches:
      return float('nan')
    return float(np.mean([float(val_loss_fn(params, *b)) for b in val_batches]))

  # ---- metadata ----------------------------------------------------------- #
  metadata = {
      'stage': 2,
      'model': 'conditional_flow_matching_behavior',
      'dataset_path': os.path.abspath(args.dataset),
      'dataset_sha256': ds.fingerprint['sha256'],
      'dataset_manifest_verified': ds.fingerprint['manifest_verified'],
      'dataset_content_sha256': ds.content_sha256,
      'split_seed': ds.split_seed,
      'val_frac': args.val_frac,
      'split_level': 'episode',
      'context_mode': ds.context_mode,
      'state_dim': ds._obs_dim,                               # noqa: SLF001
      'goal_dim': ds._goal_dim,                               # noqa: SLF001
      'context_dim': context_dim,
      'action_dim': action_dim,
      'n_transitions': ds.n_transitions,
      'n_train': ds.n_train,
      'n_val': ds.n_val,
      'training_seed': args.seed,
      'batch_size': args.batch_size,
      'learning_rate': args.learning_rate,
      'steps': args.steps,
      'flow_config': fcfg.asdict(),
      'n_params': n_params,
      'resume': resume_info,
      'start_step': start_step,
      'stream_seed': (args.stream_seed if args.stream_seed is not None
                      else args.seed),
      'flow_steps_default': args.flow_steps,
      'flow_steps_provenance': flow_mod.FLOW_STEPS_PROVENANCE,
      # Sampling-only properties, recorded here for reproducibility. They do
      # NOT enter the flow-matching training target.
      'per_step_action_clip': list(flow_mod.ACTION_BOX),
      'clip_provenance': flow_mod.CLIP_PROVENANCE,
      'val_batches': args.val_batches,
      'val_noise_seed': args.val_noise_seed,
      'action_box': list(flow_mod.ACTION_BOX),
      'actions_normalized_by_loader': False,
      'contains_discriminator': False,
      'contains_causal_weighting': False,
  }
  if args.ckpt_dir:
    ckpt_mod.save_metadata(args.ckpt_dir, metadata)

  # ---- train -------------------------------------------------------------- #
  stream_seed = args.stream_seed if args.stream_seed is not None else args.seed
  sample_rng = np.random.default_rng(stream_seed)
  key = jax.random.PRNGKey(stream_seed) if args.resume_from else key
  milestones = sorted(int(x) for x in args.milestone_steps.split(',') if x)

  history, best_val = [], float('inf')
  if args.resume_metrics and os.path.exists(args.resume_metrics):
    with open(args.resume_metrics) as f:
      history = json.load(f)
    prior = [h['val_loss'] for h in history if h.get('val_loss') is not None]
    if prior:
      best_val = float(min(prior))
    print(f'  [resume] carried {len(history)} prior metric rows; '
          f'best_val so far = {best_val:.6f}')

  init_val = evaluate(params)
  print(f'{"step":>8}  {"train_flow_mse":>14}  {"val_flow_mse":>12}  '
        f'{"elapsed_s":>9}')
  print(f'{start_step:>8}  {"--":>14}  {init_val:>12.6f}  '
        f'{time.time()-t_start:>9.1f}'
        + ('   <- resumed state (must match the resumed run)'
           if args.resume_from else ''))
  if not args.resume_from:
    history.append({'step': 0, 'train_loss': None, 'val_loss': init_val,
                    'elapsed_s': round(time.time() - t_start, 2)})
    best_val = float('inf')

  running, running_n = 0.0, 0
  for step in range(start_step + 1, args.steps + 1):
    batch = ds.sample_batch(args.batch_size, split='train', rng=sample_rng)
    key, sub = jax.random.split(key)
    params, opt_state, loss = update(
        params, opt_state, jnp.asarray(batch.state), jnp.asarray(batch.action),
        sub)
    running += float(loss)
    running_n += 1

    if step % args.log_every == 0 or step == args.steps:
      train_loss = running / max(running_n, 1)
      running, running_n = 0.0, 0
      do_eval = (step % args.eval_every == 0) or (step == args.steps)
      vl = evaluate(params) if do_eval else None
      elapsed = time.time() - t_start
      print(f'{step:>8}  {train_loss:>14.6f}  '
            f'{(f"{vl:.6f}" if vl is not None else "-"):>12}  '
            f'{elapsed:>9.1f}', flush=True)
      history.append({'step': step, 'train_loss': train_loss, 'val_loss': vl,
                      'elapsed_s': round(elapsed, 2)})
      if not np.isfinite(train_loss) or (vl is not None and not np.isfinite(vl)):
        print('ABORT: non-finite loss')
        return 1
      if do_eval and args.ckpt_dir:
        best_val = ckpt_mod.save_checkpoint(
            args.ckpt_dir, step, params, opt_state, history, vl, best_val)
    if step in milestones and args.ckpt_dir:
      ckpt_mod.save_milestone(args.ckpt_dir, step, params, opt_state)

  final_val = evaluate(params)
  metadata['final_val_flow_mse'] = final_val
  metadata['best_val_flow_mse'] = (None if best_val == float('inf')
                                   else best_val)
  metadata['init_val_flow_mse'] = init_val
  metadata['wall_clock_s'] = round(time.time() - t_start, 2)
  if args.ckpt_dir:
    ckpt_mod.save_metadata(args.ckpt_dir, metadata)
    with open(os.path.join(args.ckpt_dir, 'metrics.json'), 'w') as f:
      json.dump(history, f, indent=2)

  print()
  print(f'init val flow-MSE   {init_val:.6f}')
  print(f'final val flow-MSE  {final_val:.6f}')
  print(f'best  val flow-MSE  {best_val:.6f}')
  print(f'elapsed             {time.time() - t_start:.1f}s')
  if args.ckpt_dir:
    print(f'checkpoints         {os.path.abspath(args.ckpt_dir)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
