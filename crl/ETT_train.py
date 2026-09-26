"""ETT variant of the single-process train + eval loop (crl/train.py).

This module is OFFLINE-ONLY: ``config.offline_dataset`` is required and no
training environment or collection path exists. Same contrastive RL loop as
``crl.train``, with one addition: a CAUSAL TRANSITION MODEL is fitted on the
frozen dataset BEFORE the first contrastive gradient step. It has two heads,
trained jointly on the same minibatches over the learner STATE space -- the
one-step dynamics ``f_theta(s_t, a_t, a'_t) -> s_{t+1}``, and a BLIND expert
policy that imitates the sighted teacher which collected the data without
ever seeing the rockfall trap status that teacher acted on. By default that
second head is a MIXTURE density network over actions
(``config.dyn_policy_head='mdn'``, ``config.dyn_policy_components`` branches),
because what it is fitting is the teacher's action distribution marginalized
over the hidden latent and is therefore multi-modal -- a point regression
would return the mean of "go straight" and "detour". See
``crl/causal_transition_model.py``.

The fit itself has two dynamics terms when ``config.dyn_negative_dataset``
names a collection of FAILURE episodes: the usual diagonal ``f(s, a, a) ->
s'`` on the frozen positive dataset, and an off-diagonal ``f(s, a, a') ->
s'`` with a random ``a' != a`` on the negative transitions. The negative file
is read straight off disk and never enters the replay buffer -- the offline
contract covers ``config.offline_dataset`` and nothing else touches it.

The fitted model is reported and checkpointed but NOT consumed by any
contrastive loss; its one consumer is the augmentation path below.
``dyn_enable=False`` is the master switch over the whole stage -- both the
fit and the rollout sampling -- and ``dyn_train_steps=0`` disables just the
fit; either makes this module behave exactly like ``crl.train``.

One OPTIONAL consumer exists, off by default: with ``dyn_augment_frac > 0``
the two fitted heads are composed forward to generate trajectories -- each one
the INITIAL state of a dataset episode continued by ``dyn_rollout_steps``
closed-loop model steps -- which are mixed into each contrastive minibatch at
that fraction. They go into a SEPARATE buffer, never the frozen one: the offline
contract is that the dataset does not grow or change, and it is enforced by
gate G8 plus content-hash assertions after the fit, at every eval and at the
end of the run. At ``dyn_augment_frac=0`` -- or at ``dyn_enable=False``,
whatever the fraction -- the sampling stream is bit-for-bit the
pre-augmentation one, so existing offline runs reproduce exactly.

That generated buffer is REBUILT DURING TRAINING, at the top of every
``dyn_augment_refresh_every`` iterations of the main loop (default: every
one), from the policy as it stands at that moment. The reason is the
dynamics head's third input block: with ``dyn_off_diagonal='agent'`` the
counterfactual action ``a'`` is the CURRENT learner's action at the
generated state, so every generated transition reads "the blind expert would
play ``a`` here, the agent plays ``a'``". Which of the two a step takes is
the expert head's own coin, ``sum_i w_i(s) ** 2``: with that probability the
step is the factual diagonal ``f(s, a, a)``, and otherwise it is
``f(s, a, a')`` -- the branch that ``config.dyn_negative_dataset`` supervised
out of the failure episodes. So the counterfactual fires in proportion to how
much the hidden confounder matters at that state, and a state where the
expert mixture has collapsed onto one branch is never asked a counterfactual
it has no answer for. Data generated against a policy ten thousand gradient
steps ago answers a question about a policy that no longer exists, hence the
refresh.

Usage:
    python -m crl.ETT_train --offline_dataset path/to/data.npz
    python ant_maze/train_ctl_ant_maze.py configs/ant_maze_configs.yml

Algorithm selection mirrors lp_contrastive.py:
    (default)            contrastive_nce
    --use_cpc            contrastive_cpc
    --use_td --twin_q    c_learning
    --use_td --twin_q --add_mc_to_td   nce+c_learning
    --use_gcbc           gcbc
"""
import argparse
import dataclasses
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from crl import checkpoint as ckpt_mod
from crl import d4rl_ant
from crl import envs
from crl import losses as losses_mod
from crl import networks as networks_mod
from crl import causal_transition_model as causal_mod
from crl import replay as replay_mod
from crl.config import Config


def _build_arg_parser():
  p = argparse.ArgumentParser(description='Contrastive RL (Acme-free port).')
  for f in dataclasses.fields(Config):
    name = '--' + f.name
    if f.type is bool or isinstance(f.default, bool):
      # Support --flag / --no-flag for booleans.
      p.add_argument(name, dest=f.name, action='store_true', default=None)
      p.add_argument('--no-' + f.name, dest=f.name, action='store_false')
    else:
      p.add_argument(name, dest=f.name, default=None)
  return p


def _apply_overrides(config, args):
  for f in dataclasses.fields(Config):
    val = getattr(args, f.name)
    if val is None:
      continue
    if isinstance(getattr(config, f.name), bool) or f.type is bool:
      setattr(config, f.name, bool(val))
    elif f.name in ('hidden_layer_sizes', 'dyn_hidden_layer_sizes',
                    'dyn_policy_hidden_layer_sizes'):
      setattr(config, f.name, tuple(int(x) for x in str(val).split(',')))
    elif f.name == 'entropy_coefficient':
      setattr(config, f.name, None if val in ('None', 'none') else float(val))
    else:
      # Cast to the type of the current default.
      cur = getattr(config, f.name)
      caster = type(cur) if cur is not None else str
      setattr(config, f.name, caster(val))
  return config


def load_negative_transitions(path, obs_dim, action_dim, log_fn=print):
  """Flat ``(states, actions, next_states, episode_ids)`` from a failure npz.

  The NEGATIVE dataset -- failure episodes for the same benchmark -- is what
  supervises the dynamics head's off-diagonal ``f(s, a, a') -> s'`` for
  ``a' != a`` (see ``crl.causal_transition_model.fit_causal``). It is read
  here and handed to the fit, and it NEVER enters the replay buffer: the
  offline contract is that the frozen dataset is exactly
  ``config.offline_dataset``, and the content-hash assertions around the fit
  would catch it if it did.

  The file is an episode npz in ``offline_dataset``'s own schema
  (``obs [E, L, *]``, ``act [E, L, A]``, optional ``lengths [E]``) -- the
  candidate files under ``artifacts/v6_failneg/candidates_*/`` are exactly
  that. Only the leading ``obs_dim`` columns of ``obs`` are read, so a
  collection stored under an older goal contract (V6 candidates are 58 wide
  against a 31-wide _gxy positive dataset) needs no conversion: the learner
  STATE half is the same 29 columns in both, and dynamics never see a goal.

  Pointing this at a ``*_sidecar.npz`` is a common slip -- those hold the
  per-episode outcome metadata, not trajectories -- so it is named in the
  error rather than surfacing as a KeyError.
  """
  with np.load(path, allow_pickle=False) as d:
    keys = set(d.files)
    if not {'obs', 'act'} <= keys:
      raise ValueError(
          f'{path} has no obs/act arrays (keys: {sorted(keys)}), so it is not '
          'an episode dataset. If this is a *_sidecar.npz of outcome '
          'metadata, point dyn_negative_dataset at the candidate npz next to '
          'it instead.')
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    lengths = (np.asarray(d['lengths'], np.int64) if 'lengths' in keys
               else np.full(obs.shape[0], obs.shape[1], np.int64))
  if obs.ndim != 3 or act.ndim != 3:
    raise ValueError(f'{path}: expected obs [E, L, *] and act [E, L, A], got '
                     f'{obs.shape} and {act.shape}')
  if obs.shape[2] < obs_dim:
    raise ValueError(f'{path}: obs are {obs.shape[2]} wide, narrower than the '
                     f'learner state (obs_dim={obs_dim})')
  if act.shape[2] != action_dim:
    raise ValueError(f'{path}: actions are {act.shape[2]} wide, expected '
                     f'action_dim={action_dim}')
  s_l, a_l, sn_l, e_l = [], [], [], []
  for e in range(obs.shape[0]):
    n = int(lengths[e]) - 1                    # transitions in this episode.
    if n < 1:
      continue
    s_l.append(obs[e, :n, :obs_dim])
    sn_l.append(obs[e, 1:n + 1, :obs_dim])
    a_l.append(act[e, :n])
    e_l.append(np.full(n, e, np.int64))
  if not s_l:
    raise ValueError(f'{path} holds no usable transitions (every episode is '
                     'shorter than two observations)')
  states = np.concatenate(s_l)
  log_fn(f'NEGATIVE DATASET: {path} -> {states.shape[0]} transitions from '
         f'{len(s_l)} failure episodes (obs {obs.shape[2]} wide, sliced to '
         f'the leading {obs_dim} learner-state columns); it supervises the '
         "dynamics head's OFF-DIAGONAL and never enters the replay buffer")
  return (states, np.concatenate(a_l), np.concatenate(sn_l),
          np.concatenate(e_l))


def route_stats(infos, successes):
  """Per-episode route / hazard outcome of one eval, aggregated.

  Returns {} for envs that report nothing in ``info`` (every env but the
  rockfall benchmarks). On those benchmarks the scalar success rate is NOT a
  measure of the policy: an agent that walks the shortcut reaches the goal
  exactly when neither hazard zone was armed, and the arming is a reset-time
  draw from a stream seeded off ``config.seed``, identical across runs that
  share it. These are the quantities that do vary with the policy -- which
  route it picked, whether it died, and the success rate SPLIT by whether the
  episode was survivable at all.

  ``success_given_armed`` is the one to read: it is near 0 for any policy that
  rushes the shortcut, and only route choice or waiting can lift it.
  """
  if not infos or 'route' not in infos[0]:
    return {}
  succ = [bool(s) for s in successes]
  route = [i.get('route') for i in infos]
  armed = [bool(i.get('rockfall_active_1')) or bool(i.get('rockfall_active_2'))
           for i in infos]

  def _rate(mask):
    hits = [s for s, k in zip(succ, mask) if k]
    return float(np.mean(hits)) if hits else float('nan')

  def _mean_step(key):
    #: None means the ant never got there; averaging only over the ones that
    #: did keeps this a timing measure rather than a mixed arrival/timing one.
    vals = [i[key] for i in infos
            if i.get(key) is not None and i.get(key) >= 0]
    return float(np.mean(vals)) if vals else float('nan')

  out = {
      'route_detour_frac': float(np.mean([r == 'detour' for r in route])),
      'route_shortcut_frac': float(np.mean([r == 'shortcut' for r in route])),
      'route_undecided_frac': float(np.mean([r is None for r in route])),
      'hazard_armed_frac': float(np.mean(armed)),
      'death_frac': float(np.mean([bool(i.get('dead')) for i in infos])),
      'success_given_armed': _rate(armed),
      'success_given_clear': _rate([not a for a in armed]),
      # Waiting shows up as a LATER band entry for an unchanged mouth arrival:
      # the teacher holds at the mouth with zero torque and then walks in.
      'mouth_step_1': _mean_step('mouth_step_1'),
      'band_entry_step_1': _mean_step('band_entry_step_1'),
      'mouth_step_2': _mean_step('mouth_step_2'),
      'band_entry_step_2': _mean_step('band_entry_step_2'),
  }
  # Route choice is made blind, so it must NOT track the arming; logged so a
  # spurious correlation (an observation leak) would be visible rather than
  # assumed away.
  out['detour_frac_given_armed'] = float(np.mean(
      [r == 'detour' for r, k in zip(route, armed) if k])) if any(armed) \
      else float('nan')
  return out


def evaluate(env, eval_act_fn, params, episodes, np_rng, action_dim,
             obs_dim, start_index, end_index, goal_indices=None, raw=False):
  """Greedy rollouts. Returns (success, final_dist, min_dist, route_stats).

  success  = any reward==1 within an episode.
  *_dist   = L2 distance ||achieved_goal - desired_goal||, where achieved_goal
             is obs_to_goal(state) (state[start:end]) and desired_goal is the
             goal half of the observation -- the same quantity the reward uses.
  With ``goal_indices`` (rich-goal ablation) the distance is XY-ONLY (the
  first two goal coords), so metrics stay comparable across goal arms.
  route_stats: see ``route_stats`` above; {} for envs that report no info.

  The env's terminal flag is deliberately still ignored: every episode runs
  the full horizon so the eval-env step count stays exactly
  ``episodes * max_episode_steps``, which the offline contract asserts. The
  rockfall envs make post-death steps absorbing, so this costs nothing but
  wall time.

  ``raw=True`` returns the per-episode lists (successes, final_dists,
  min_dists, infos) instead, so ``evaluate_seeds`` can pool episodes.
  """
  successes, final_dists, min_dists, infos = [], [], [], []
  for _ in range(episodes):
    obs = env.reset()
    hit = 0.0
    dists = []
    info = {}
    for _ in range(env.max_episode_steps):
      a = np.asarray(eval_act_fn(params, jnp.asarray(obs[None]))[0])
      obs, r, _, info = env.step(a)
      hit = max(hit, float(r))
      # float cast: uint8 (image obs) subtraction would wrap around.
      state = obs[:obs_dim].astype(np.float32, copy=False)
      goal = obs[obs_dim:].astype(np.float32, copy=False)
      if goal_indices is not None:
        ag, goal = state[:2], goal[:2]           # XY primary metric
      else:
        ag = state[start_index:] if end_index == -1 else state[start_index:end_index]
      dists.append(float(np.linalg.norm(ag - goal)))
    successes.append(hit)
    final_dists.append(dists[-1])
    min_dists.append(min(dists))
    # The info dict carries the episode's whole history (route, arming, death),
    # and post-death steps keep returning it, so the LAST one is the summary.
    infos.append(info if isinstance(info, dict) else {})
  if raw:
    return successes, final_dists, min_dists, infos
  return (float(np.mean(successes)), float(np.mean(final_dists)),
          float(np.mean(min_dists)), route_stats(infos, successes))


def evaluate_seeds(eval_envs, eval_act_fn, params, episodes, *args, **kw):
  """``evaluate`` on each independently seeded eval env, pooled.

  Returns (success, final_dist, min_dist, route_stats, per_seed_success):
  the first four pool all ``len(eval_envs) * episodes`` episodes, so the
  route stats' conditional rates are computed over the pooled episodes rather
  than averaged per seed; the last is one success rate per env.
  """
  succ, fd, md, infos, per_seed = [], [], [], [], []
  for env in eval_envs:
    s, f, m, i = evaluate(env, eval_act_fn, params, episodes, *args,
                          raw=True, **kw)
    succ += s; fd += f; md += m; infos += i
    per_seed.append(float(np.mean(s)))
  return (float(np.mean(succ)), float(np.mean(fd)), float(np.mean(md)),
          route_stats(infos, succ), per_seed)


def evaluate_push_physical(env, eval_act_fn, params, episodes, np_rng):
  """Greedy rollouts for a FetchPush IMAGE env, scored on PHYSICAL coordinates.

  Returns (success, final_dist, min_dist) where success is the sparse env
  reward (object within 0.05 m of the goal, from sim state) and the distances
  are ``||object_xyz - desired_goal_xyz||`` read from the simulator -- NOT the
  flattened image-L2 that ``evaluate`` would compute for image observations.
  ``env`` must be a ``crl.envs.FetchEnv`` (exposes ``_env.unwrapped._get_obs``).
  """
  u = env._env.unwrapped
  successes, final_dists, min_dists = [], [], []
  for _ in range(episodes):
    env.reset()
    desired = np.asarray(env._desired, dtype=np.float32)
    obs_img = np.concatenate([env._frame(), env._goal_img])
    hit = 0.0
    dists = []
    for _ in range(env.max_episode_steps):
      a = np.asarray(eval_act_fn(params, jnp.asarray(obs_img[None]))[0])
      obs_img, r, _, _ = env.step(a)
      hit = max(hit, float(r))
      d = u._get_obs()
      obj = np.asarray(d['achieved_goal'], dtype=np.float32)
      dists.append(float(np.linalg.norm(obj - desired)))
    successes.append(hit)
    final_dists.append(dists[-1])
    min_dists.append(min(dists))
  return (float(np.mean(successes)), float(np.mean(final_dists)),
          float(np.mean(min_dists)))


def ETT_train(config: Config, stop_on_done: bool = True):
  print('Config:', config)
  key = jax.random.PRNGKey(config.seed)
  np_rng = np.random.default_rng(config.seed)

  # --- STRICT OFFLINE MODE (the only mode this module supports) ---------
  # No TRAINING environment is ever created and there is no collection path:
  # the eval env is the ONLY env; it fills the config dims and is stepped
  # solely inside evaluate(). Online-only knobs are rejected / disabled.
  if not config.offline_dataset:
    raise ValueError(
        'ETT_train is offline-only: set offline_dataset '
        '(--offline_dataset path/to/data.npz).')
  if int(config.num_actors) > 1:
    raise ValueError(
        f'offline mode rejects num_actors={config.num_actors} (>1): there '
        'is no collection to parallelize.')
  eval_env = envs.make_env(config.env_name, config, seed=config.seed + 10_000)
  # One eval env per eval seed; each draws its own hazard latents, clocks and
  # rock jitter from its seed. Env 0 is the one above, unchanged, so its
  # episodes are the ones single-seed runs were scored on.
  if int(config.eval_num_seeds) < 1:
    raise ValueError(f'eval_num_seeds={config.eval_num_seeds} must be >= 1')
  eval_envs = [eval_env] + [
      envs.make_env(config.env_name, config,
                    seed=config.seed + 10_000 + 100_003 * k)
      for k in range(1, int(config.eval_num_seeds))]
  # random_steps / min_replay_size are meaningless with a frozen dataset.
  if config.random_steps:
    print(f'  [offline] disabling random_steps={config.random_steps} -> 0')
  config.random_steps = 0
  print(f'obs_dim={config.obs_dim} goal_dim={config.goal_dim} '
        f'action_dim={config.action_dim} '
        f'max_episode_steps={config.max_episode_steps} '
        f'goal_slice=[{config.start_index}:{config.end_index}]')

  # --- Networks + learner ---
  nets = networks_mod.make_networks(
      obs_dim=config.obs_dim, goal_dim=config.goal_dim,
      action_dim=config.action_dim, repr_dim=int(config.repr_dim),
      repr_norm=config.repr_norm, repr_norm_temp=config.repr_norm_temp,
      hidden_layer_sizes=config.hidden_layer_sizes,
      twin_q=config.twin_q, use_image_obs=config.use_image_obs,
      use_layer_norm=config.use_layer_norm)

  policy_optimizer = optax.adam(config.actor_learning_rate, eps=1e-7)
  q_optimizer = optax.adam(config.learning_rate, eps=1e-7)

  start, end = config.start_index, config.end_index
  gidx = (None if config.goal_indices is None
          else jnp.asarray(config.goal_indices))
  def obs_to_goal(states):
    if gidx is not None:
      return states[:, gidx]
    return states[:, start:] if end == -1 else states[:, start:end]

  # Failure-aware negative sampling (Part 1): optional failure-state bank.
  # Bank states are stored in the learner STATE space; slice them to goal
  # coords with the same rule the relabeler uses (goal_indices / start:end).
  fail_bank = None
  if getattr(config, 'fail_bank_path', ''):
    from crl.replay import obs_to_goal as np_obs_to_goal
    with np.load(config.fail_bank_path) as fb:
      bank_states = np.asarray(fb['goals'], np.float32)
    fail_bank = np_obs_to_goal(
        bank_states, config.start_index, config.end_index,
        config.goal_indices)
    print(f'FAILURE-NEGATIVE BANK: {config.fail_bank_path} '
          f'({bank_states.shape[0]} states -> goal dim {fail_bank.shape[1]}), '
          f'alpha={config.fail_neg_alpha}')

  init_state, update_step = losses_mod.build_learner(
      nets, config, obs_to_goal, policy_optimizer, q_optimizer,
      fail_bank=fail_bank)

  key, key_init = jax.random.split(key)
  state = init_state(key_init)

  # Optionally resume params/optimizer from a previous run (buffer refills).
  start_step = 0
  if config.resume and config.ckpt_dir:
    latest = os.path.join(config.ckpt_dir, 'latest.pkl')
    if os.path.exists(latest):
      start_step, state = ckpt_mod.load_checkpoint(latest)
      print(f'Resumed from {latest} at step {start_step}.')
    else:
      print(f'--resume set but no checkpoint at {latest}; starting fresh.')

  # --- Jitted helpers ---
  # Only the greedy eval policy is needed: offline training never acts in a
  # training env, so there is no stochastic collection policy.
  def _eval_act(params, obs):
    return nets.sample_eval(nets.policy_network.apply(params, obs), None)

  eval_act_fn = jax.jit(_eval_act) if config.jit else _eval_act

  def _multi_update(state, trans_G):
    # trans_G: Transition with leading dim G; scan does G sequential updates.
    state, metrics = jax.lax.scan(update_step, state, trans_G)
    metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
    return state, metrics
  multi_update = jax.jit(_multi_update) if config.jit else _multi_update

  # --- Replay ---
  min_replay = config.min_replay_size
  dyn_model, dyn_stats = None, None   # ETT stage 0.
  # Load the fixed dataset ONCE, sized exactly to it, and FREEZE it. Then run
  # the full static + buffer audit BEFORE any gradient step; a single failed
  # gate aborts training. See crl/offline_audit.py for the gate definitions.
  from crl import offline_audit
  buffer, off_fp = offline_audit.build_offline_buffer(
      config.offline_dataset, config)
  passed, gates, audit_report = offline_audit.run_static_audit(
      config.offline_dataset, config, buffer=buffer)
  print('OFFLINE AUDIT (pre-training gates):')
  for gname, ok in gates.items():
    print(f'  {"PASS" if ok else "FAIL"}  {gname}')
  print(f'  dataset={config.offline_dataset}')
  print(f'  sha256={off_fp["sha256"][:16]}...  eps={off_fp["n_episodes"]}  '
        f'trans={off_fp["n_transitions"]}  obs={off_fp["obs_shape"]}  '
        f'act={off_fp["act_shape"]}  ep_len<=[{off_fp["ep_lengths_min"]},'
        f'{off_fp["ep_lengths_max"]}]')
  if off_fp['keys']['audit']:
    print(f'  audit-only fields kept OUT of the learner: '
          f'{off_fp["keys"]["audit"]}')
  if not passed:
    raise RuntimeError(
        f'OFFLINE AUDIT FAILED (gates={gates}); refusing to train.')
  if config.min_replay_size:
    print(f'  [offline] disabling min_replay_size={config.min_replay_size} '
          '-> 0 (the frozen dataset IS the buffer)')
  min_replay = 0   # the frozen dataset IS the buffer; learn from step one.

  # Resume must use the identical dataset; a fresh run records its hash.
  if config.ckpt_dir:
    if config.resume:
      same, recorded = offline_audit.require_same_dataset_hash(
          config.ckpt_dir, off_fp['sha256'])
      if not same:
        raise RuntimeError(
            'OFFLINE RESUME dataset MISMATCH: this checkpoint was trained on '
            f'{recorded} but the current dataset hashes {off_fp["sha256"]}.')
    else:
      offline_audit.record_dataset_hash(
          config.ckpt_dir, off_fp['sha256'], off_fp['meta'])
    # The hash sidecar pins WHICH data this run trained on. The manifest pins
    # everything an evaluation process cannot recover from the checkpoint: the
    # environment and hazard timing its eval env must be rebuilt with, the
    # contrastive learning math, and -- the part unique to this module -- which
    # ETT arm the run is, standard or augmented. Written before the first
    # gradient step so an interrupted run is still identifiable.
    manifest_path = offline_audit.record_benchmark_manifest(
        config.ckpt_dir, config, off_fp)
    ett_arm = offline_audit.ett_contract(config)
    print(f'  benchmark manifest -> {manifest_path}')
    print(f'  ARM: {ett_arm["arm"]} (dyn_enable={ett_arm["dyn_enable"]}, '
          f'dyn_augment_frac={ett_arm["dyn_augment_frac"]})')

  # Runtime immutability watchdog (checked at every eval): the collection env
  # does not exist (env is None) and the buffer is frozen, so collection is
  # structurally impossible; here we also pin the content hash and count the
  # eval env's steps so eval cannot mutate or over-step.
  offline_frozen_sha = buffer.content_sha256()
  offline_frozen_ptr = (buffer._num_eps, buffer.ready_steps)
  offline_eval_steps = {'n': 0, 'evals': 0}
  def _count_steps(real_step):
    def _counted_eval_step(a):
      offline_eval_steps['n'] += 1
      return real_step(a)
    return _counted_eval_step
  for _env in eval_envs:
    _env.step = _count_steps(_env.step)

  # --- ETT stage 0: the causal transition model ----------------------------
  # Both supervised heads -- the dynamics f(s_t, a_t, a'_t) -> s_{t+1} and the
  # BLIND expert policy over a_t -- are fitted (or loaded) on the frozen
  # dataset BEFORE any contrastive gradient step. Four properties are
  # deliberate:
  #   * it reads the buffer through flat_transitions(), which copies out of
  #     storage and never advances buffer._rng -- so the goal-relabeling stream
  #     the CRL loop consumes is bit-for-bit what it would have been without
  #     this stage, and offline runs stay comparable to the baseline;
  #   * that same call hands the policy head the LEARNER STATE only. The
  #     rockfall trap status the sighted teacher read when it collected these
  #     episodes is audit-only and never entered the stored observation
  #     (offline_audit gate G6), so pi is blind to it by construction;
  #   * it uses its own seed stream (config.seed + 1_000_003), so it cannot
  #     perturb the agent's PRNG either;
  #   * the frozen-content hash is pinned right after, proving the fit did not
  #     mutate the dataset.
  # Nothing below consumes `dyn_model` in a LOSS -- see the module docstring.
  # The optional dataset-augmentation stage further down generates data with
  # it, which is the one consumer, and it is off unless dyn_augment_frac > 0.
  # `dyn_enable` is the master switch over both halves of the stage (the fit
  # here and the rollout sampling below). With it off this module reduces to
  # offline crl.train, which is what makes a baseline arm a one-line edit of an
  # otherwise-unchanged augmented config -- so the suppression is announced
  # rather than silent.
  ett_enabled = bool(getattr(config, 'dyn_enable', True))
  if not ett_enabled:
    suppressed = []
    if config.dyn_train_steps > 0 or config.dyn_model_path:
      suppressed.append('transition-model fit' if not config.dyn_model_path
                        else f'transition-model load ({config.dyn_model_path})')
    if float(getattr(config, 'dyn_augment_frac', 0.0)) > 0.0:
      suppressed.append(
          f'dataset augmentation (dyn_augment_frac='
          f'{config.dyn_augment_frac})')
    print('ETT STAGE 0 DISABLED (dyn_enable=False): '
          + (', '.join(suppressed) + ' skipped' if suppressed else
             'nothing to skip')
          + '; this run is plain offline CRL.', flush=True)
  if ett_enabled and (config.dyn_train_steps > 0 or config.dyn_model_path):
    # THE FEASIBLE STATE BOX the rollouts are clipped into. It is read off the
    # LIVE eval env -- its traversable cells, its torso offset, its compiled
    # MuJoCo model -- and not written down anywhere, so a run on a different
    # map gets a different box from the same call and a run on a benchmark
    # that is not an ant maze gets none. See crl.d4rl_ant.ant_state_bounds for
    # what bounds each block of the 29 coordinates, and config.dyn_clip_states
    # for why the clip exists at all.
    state_bounds = None
    if bool(getattr(config, 'dyn_clip_states', True)):
      try:
        state_bounds = d4rl_ant.ant_state_bounds(eval_env)
      except TypeError as exc:
        print(f'STATE CLIP: unavailable for {config.env_name} ({exc}); '
              'generated rollouts run unclipped.', flush=True)
      else:
        lo, hi = state_bounds
        if lo.size != config.obs_dim:
          raise ValueError(
              f'{config.env_name} gives a {lo.size}-dim state box but the '
              f'learner state is {config.obs_dim}-dim')
        print(f'STATE CLIP: generated rollouts clipped to the feasible box '
              f'of {config.env_name} -- '
              f'xy [{lo[0]:.1f}, {hi[0]:.1f}] x [{lo[1]:.1f}, {hi[1]:.1f}], '
              f'z [{lo[2]:.1f}, {hi[2]:.1f}], '
              f'joints [{lo[7]:.2f}, {hi[7]:.2f}] and their stops, '
              f'|qvel| <= {hi[-1]:.0f}', flush=True)
    # THE MAZE FLOOR PLAN, the other half of the same constraint. The box
    # above is a product of intervals, so its xy is the maze's BOUNDING box
    # and contains every interior wall; this projects generated xy onto the
    # union of the traversable cells instead. Read off the same live env, and
    # separately switchable because it is a different claim about the state:
    # see config.dyn_clip_walls.
    free_cells = None
    if bool(getattr(config, 'dyn_clip_walls', True)):
      try:
        free_cells = causal_mod.maze_free_cells(eval_env, d4rl_ant.SCALING)
      except (TypeError, ValueError) as exc:
        print(f'WALL CLIP: unavailable for {config.env_name} ({exc}); '
              'generated xy is bounded by the box alone.', flush=True)
      else:
        centres, half = free_cells
        print(f'WALL CLIP: generated xy projected onto the '
              f'{centres.shape[0]} traversable cells of {config.env_name} '
              f'({2 * half:.1f} units square), so a generated chain cannot '
              f'cross an interior wall.', flush=True)
    if config.dyn_model_path:
      dyn_model = causal_mod.load(config.dyn_model_path)
      t = getattr(dyn_model, 'transition', dyn_model)
      print(f'CAUSAL TRANSITION MODEL: loaded {config.dyn_model_path} '
            f'(dynamics MLP {t.hidden_layer_sizes}, '
            f'{t.input_dim}->{t.state_dim}, '
            f'input=(s,a{",a\'" if t.next_action_input else ""}), '
            f'target={"delta" if t.predict_delta else "absolute"}'
            + (f'; expert policy '
               f'{"MIXTURE k=%d" % dyn_model.policy.n_components if dyn_model.is_mixture else "MLP"} '
               f'{dyn_model.policy.hidden_layer_sizes}, '
               f'{dyn_model.policy.state_dim}->{dyn_model.policy.action_dim}'
               if isinstance(dyn_model, causal_mod.CausalTransitionModel)
               else '; dynamics-only checkpoint, no expert policy') + ')',
            flush=True)
      # A pickle carries the box and floor plan it was fitted with, which may
      # be older ones or none at all. This run's env is the authority, so both
      # are re-attached here -- including being REMOVED when dyn_clip_states /
      # dyn_clip_walls are False, so those flags mean the same thing on a
      # loaded model as on a fresh fit.
      t.set_state_bounds(state_bounds)
      t.set_free_cells(free_cells)
    else:
      dyn_s, dyn_a, dyn_sn, dyn_eid = buffer.flat_transitions()
      # The NEGATIVE dataset, when one is named: failure episodes read
      # straight off disk (never into the replay buffer) whose transitions
      # supervise the dynamics head's OFF-DIAGONAL f(s, a, a') for a' != a.
      # Without it that half of the head's input is unsupervised and an
      # off-diagonal query returns the initialization -- which is exactly
      # what the augmentation path below asks for, so the two are set
      # together or not at all.
      neg = (None, None, None, None)
      if getattr(config, 'dyn_negative_dataset', ''):
        neg = load_negative_transitions(
            config.dyn_negative_dataset, config.obs_dim, config.action_dim)
      # Both heads are supervised by these rows alone, so the fit touches
      # nothing of the agent's -- neither its parameters nor its PRNG.
      dyn_model, dyn_stats = causal_mod.fit_causal(
          dyn_s, dyn_a, dyn_sn, dyn_eid, config,
          seed=config.seed + 1_000_003,
          neg_states=neg[0], neg_actions=neg[1], neg_next_states=neg[2],
          neg_episode_ids=neg[3], state_bounds=state_bounds,
          free_cells=free_cells)
      print(causal_mod.summary_line(dyn_stats), flush=True)
      if config.ckpt_dir:
        os.makedirs(config.ckpt_dir, exist_ok=True)
        dyn_path = os.path.join(config.ckpt_dir, 'causal_transition_model.pkl')
        causal_mod.save(dyn_model, dyn_path, dyn_stats)
        rep = causal_mod.write_report(
            config.ckpt_dir, dyn_stats,
            filename='causal_transition_model_metrics.json')
        print(f'  saved {dyn_path} + {rep}', flush=True)
    assert buffer.content_sha256() == offline_frozen_sha, \
        'offline contract violated: transition-model fit changed replay CONTENT'

  # --- ETT stage 0b: augment the dataset with model rollouts (OFF by default)
  # Generated trajectories CANNOT be added to `buffer`. It is sized exactly to
  # the dataset, frozen (gate G8), and its content hash is asserted right
  # above, at every eval and at the end of the run -- "the frozen dataset IS
  # the buffer" is the offline contract, not an implementation detail. So the
  # rollouts go into a SECOND TrajectoryBuffer and the two are mixed per
  # minibatch below. Consequences worth stating:
  #   * they have to be TRAJECTORIES, not transition pairs: both buffers
  #     relabel each goal to a future state of the same trajectory, so
  #     generated data is only consumable if it carries an internal future;
  #   * a rollout inherits NOTHING from its source episode except that
  #     episode's initial state and goal, so every transition in this buffer
  #     is the model's and mixing it in cannot silently reweight real
  #     transitions;
  #   * the rollout draws read the frozen buffer only through
  #     flat_transitions() / flat_goals(), which copy out of storage without
  #     advancing its RNG -- so with dyn_augment_frac = 0 the CRL sampling
  #     stream below is bit-for-bit what it was before this stage existed.
  #
  # THE BUFFER IS REGENERATED DURING TRAINING, every
  # `dyn_augment_refresh_every` iterations of the main loop (default: every
  # one). That is not an optimization, it is what makes the counterfactual
  # input mean anything: with dyn_off_diagonal='agent' the third block a' is
  # the CURRENT learner's action at the generated state, so a buffer built
  # once before the first gradient step would spend the whole run answering
  # "what if a randomly initialized policy acted here". Each regeneration
  # draws fresh source episodes and fresh action draws too (the seed advances
  # with the round), so the generated data is resampled, not merely relabeled.
  aug = {'buffer': None, 'rounds': 0}
  # Mean of the branch coin over dataset states, reported once in the banner
  # so a run records how often its generated steps COULD go off-diagonal.
  # NaN until augmentation is set up and measures it.
  p_diag_mean = float('nan')
  aug_frac = float(getattr(config, 'dyn_augment_frac', 0.0)) if ett_enabled \
      else 0.0
  build_aug_buffer = None
  aug_refresh_every = max(1, int(getattr(config,
                                         'dyn_augment_refresh_every', 1)))
  if aug_frac > 0.0:
    if not 0.0 < aug_frac < 1.0:
      raise ValueError(f'dyn_augment_frac must be in (0, 1), got {aug_frac}')
    if dyn_model is None or not isinstance(
        dyn_model, causal_mod.CausalTransitionModel):
      raise ValueError(
          'dyn_augment_frac > 0 needs a fitted CausalTransitionModel to '
          'generate with: set dyn_train_steps > 0 or dyn_model_path to a '
          'causal (not dynamics-only) checkpoint')
    T = int(config.dyn_rollout_steps)
    if config.use_image_obs:
      # Both heads are fitted on the STATE vector, so there is nothing to roll
      # out for a pixel observation -- and the real buffer stores uint8 frames
      # while generated states are float32, so the two could not be mixed even
      # if there were.
      raise ValueError('dyn_augment_frac > 0 is not supported with '
                       'use_image_obs: the transition model operates on the '
                       'state vector, not on frames')
    aug_off = str(getattr(config, 'dyn_off_diagonal', 'none')).lower()
    if aug_off == 'agent' and not dyn_model.transition.off_diagonal_trained:
      print("  [WARN] dyn_off_diagonal='agent' on a model whose off-diagonal "
            'was never supervised (no dyn_negative_dataset at fit time): '
            "every step where the agent disagrees with the expert queries a "
            'map that is the network initialization, not physics.',
            flush=True)
    aug_s, aug_a, aug_sn, aug_eid = buffer.flat_transitions()
    aug_g = buffer.flat_goals()
    # Where rollouts START: v1 at the reset state, v2 at the route split with
    # the real prefix kept (crl/causal_transition_model_v2.py).
    aug_start = str(getattr(config, 'dyn_rollout_start', 'initial')).lower()
    gen_model, route_kwargs = dyn_model, {}
    aug_start_desc = 'each from an episode INITIAL state (no real prefix)'
    if aug_start == 'route_split':
      from crl import causal_transition_model_v2 as causal_v2
      gen_model = causal_v2.CausalTransitionModelV2.from_v1(dyn_model)
      label_path = (getattr(config, 'dyn_rollout_route_labels', '')
                    or causal_v2.sidecar_path(config.offline_dataset))
      routes = causal_v2.load_route_labels(label_path, buffer._num_eps)
      shortcut_zone = int(getattr(config, 'dyn_rollout_shortcut_zone',
                                  causal_v2.DEFAULT_SHORTCUT_ZONE))
      route_kwargs = dict(
          routes=routes,
          detour_min_x=float(getattr(config, 'dyn_rollout_detour_min_x',
                                     causal_v2.DEFAULT_DETOUR_MIN_X)),
          detour_max_y=float(getattr(config, 'dyn_rollout_detour_max_y',
                                     causal_v2.DEFAULT_DETOUR_MAX_Y)),
          trap_box=causal_v2.trap_box_for(shortcut_zone))
      aug_start_desc = (
          'each from a ROUTE SPLIT with the real prefix kept (detour: first '
          f"x > {route_kwargs['detour_min_x']:g} and "
          f"y < {route_kwargs['detour_max_y']:g}; shortcut: first state in "
          f"hazard zone {shortcut_zone} {route_kwargs['trap_box']}; route "
          'labels from '
          + (label_path if routes is not None else 'geometry (no sidecar at '
             f'{label_path})') + ')')
    elif aug_start != 'initial':
      raise ValueError(f"dyn_rollout_start must be 'initial' or "
                       f"'route_split', got {aug_start!r}")
    if aug_off != 'none' and dyn_model.is_mixture:
      # E[sum_i w_i(s)**2] over the dataset, on a capped subsample: the coin
      # is a per-state quantity and the banner wants one number, not a pass
      # over all 286k rows. It is the reference point, NOT a prediction of the
      # share the banner reports: the rollout tosses the coin at the states it
      # generates, and a chain that drifts into more confounded states than
      # the dataset average goes off-diagonal more often than this.
      _p_idx = np.random.default_rng(config.seed + 4_000_003).choice(
          aug_s.shape[0], size=int(min(aug_s.shape[0], 65_536)),
          replace=False)
      p_diag_mean = float(
          dyn_model.policy.diagonal_prob(aug_s[_p_idx]).mean())

    # a' = the CURRENT agent's action at the generated state. It is a DRAW
    # from the policy, not its mode: the learner's own behaviour is
    # stochastic, and a draw is also what keeps the T steps of one rollout
    # from being a single deterministic curve. Its key stream is seeded off
    # the stage-0 offset, never split from the learner's `key`, so turning
    # augmentation on cannot shift the agent's own randomness.
    def _agent_act(params, obs, k):
      return nets.sample(nets.policy_network.apply(params, obs), k)
    agent_act_fn = jax.jit(_agent_act) if config.jit else _agent_act
    agent_key = {'k': jax.random.PRNGKey(config.seed + 3_000_003)}

    def agent_actions(policy_params, states, goals):
      if goals is None:
        raise ValueError(
            "dyn_off_diagonal='agent' needs the goal half to build the "
            'observation the policy network reads; the rollout was called '
            'without goals')
      obs = np.concatenate([np.asarray(states, np.float32),
                            np.asarray(goals, np.float32)], axis=-1)
      agent_key['k'], sub = jax.random.split(agent_key['k'])
      return np.asarray(agent_act_fn(policy_params, jnp.asarray(obs), sub))

    def build_aug_buffer(policy_params, round_idx):
      """Regenerate the augmentation buffer from the CURRENT policy."""
      traj = gen_model.rollout(
          aug_s, aug_a, aug_sn, aug_eid, T, goals=aug_g,
          n=int(config.dyn_augment_episodes),
          # The seed advances with the round so each refresh draws different
          # source episodes (and different action draws); it stays offset from
          # every other stream in the run.
          seed=config.seed + 1_000_003 + 7_919 * int(round_idx),
          ground_truth=False,
          # A loaded pickle may carry a done head; v3 (crl/ETT_train_v3.py)
          # passes stop_on_done=False so every row runs the full T.
          stop_on_done=stop_on_done,
          off_diagonal=aug_off,
          off_diagonal_action=str(getattr(config, 'dyn_off_diagonal_action',
                                          'uniform')).lower(),
          agent_action_fn=(
              None if aug_off != 'agent'
              else lambda st, go: agent_actions(policy_params, st, go)),
          **route_kwargs)
      aug_L = int(traj.lengths.max())
      buf = replay_mod.TrajectoryBuffer(
          capacity_steps=traj.n * aug_L, ep_len_obs=aug_L,
          full_obs_dim=config.obs_dim + config.goal_dim,
          action_dim=config.action_dim, obs_dim=config.obs_dim,
          start_index=config.start_index, end_index=config.end_index,
          discount=config.discount,
          # Its own relabeling stream, offset off the stage-0 seed so it
          # cannot collide with the frozen buffer's (seeded at config.seed),
          # and advanced per round so two rounds do not relabel identically.
          seed=config.seed + 2_000_003 + 7_919 * int(round_idx),
          goal_indices=config.goal_indices,
          obs_dtype=np.float32)
      for o, a_row, n_valid in traj.episodes():
        buf.add_episode(o[:aug_L], a_row[:aug_L], length=n_valid)
      buf.freeze()      # immutable until the next round replaces it whole.
      return buf, traj

  G = max(1, config.num_sgd_steps_per_step)
  B = config.batch_size
  # Split of each minibatch between the two buffers. Fixed here, not per step,
  # so the mix is a property of the run rather than something that drifts.
  B_aug = 0 if build_aug_buffer is None else int(round(aug_frac * B))
  if build_aug_buffer is not None and not 0 < B_aug < B:
    raise ValueError(
        f'dyn_augment_frac={aug_frac} gives {B_aug} generated rows of a '
        f'{B}-row batch; pick a fraction that splits the batch')
  B_real = B - B_aug

  def refresh_aug_buffer(policy_params, iteration):
    """Rebuild the generated buffer at the top of a main-loop iteration.

    Returns a short status string for the periodic log line, or '' when this
    iteration was not a refresh one. The FIRST build prints the full banner;
    later ones would repeat it once per iteration for no new information, so
    they report the numbers that actually move -- the off-diagonal share (how
    often the agent disagreed with the expert), the share of generated states
    the feasible-box clip had to pull back, and the transition count.
    """
    if build_aug_buffer is None or iteration % aug_refresh_every:
      return ''
    round_idx = aug['rounds']
    buf, traj = build_aug_buffer(policy_params, round_idx)
    aug['buffer'], aug['rounds'] = buf, round_idx + 1
    off = traj.off_diagonal_frac
    clipped = traj.clipped_frac
    if round_idx == 0:
      print(f'DATASET AUGMENTATION: {traj.n} generated trajectories, T={T}, '
            f'{aug_start_desc} -> {buf.ready_steps} transitions, '
            + (f'{int(traj.generated.sum())} model-generated '
               f'(mean start step {traj.mean_start_step:.0f}, detour rows '
               f"{traj.route_frac('detour'):.0%}); "
               if aug_start == 'route_split' else 'all model-generated; ')
            + f'{aug_frac:.0%} of every minibatch will come from them; '
            f"a' = {aug_off}"
            + (f' ({100 * off:.1f}% of generated steps went off-diagonal; '
               f'the coin is sum_i w_i(s)**2, which averages {p_diag_mean:.3f} '
               f'at DATASET states -- the rollout tosses it at its own '
               f'generated states, so the two need not agree)'
               if aug_off != 'none' else '')
            + (f'; the feasibility clip ('
               + ' + '.join(
                   ([] if not dyn_model.clips_states else ['box'])
                   + ([] if not dyn_model.clips_walls else ['maze walls']))
               + f') moved {100 * clipped:.2f}% of '
               f'generated states and touched {100 * traj.any_clipped_frac:.1f}% '
               f'of rows -- a large share means the chain is DIVERGING and the '
               f'constraint, not the model, is writing the trajectory'
               if traj.clipped is not None else
               '; no feasibility clip on these rollouts')
            + f'; regenerated every {aug_refresh_every} iteration(s) from '
              'the current policy', flush=True)
      # Only on the first build: the content hash walks the whole frozen
      # buffer, and the eval-time assertion already covers the rest of the
      # run. The pointer check below is O(1) and stays on every round.
      assert buffer.content_sha256() == offline_frozen_sha, \
          'offline contract violated: rollout generation changed replay CONTENT'
    assert (buffer._num_eps, buffer.ready_steps) == offline_frozen_ptr, \
        'offline contract violated: rollout generation grew the replay buffer'
    return (f' aug=r{round_idx}/{buf.ready_steps}'
            + (f'/off{100 * off:.0f}%' if aug_off != 'none' else '')
            + (f'/clip{100 * clipped:.1f}%' if traj.clipped is not None
               else ''))

  def sample_G():
    aug_buffer = aug['buffer']
    if aug_buffer is None:
      # Untouched path: exactly one sample(B) per gradient step, same order,
      # same RNG draws as before dataset augmentation existed.
      batches = [buffer.sample(B) for _ in range(G)]
    else:
      # Concatenate rather than interleave: the contrastive loss treats the
      # batch as an unordered set of anchors whose off-diagonal pairs are the
      # negatives, so real anchors get generated goals as negatives and vice
      # versa. That cross-talk is the point -- it is what makes the generated
      # states inform the real ones' representation.
      batches = []
      for _ in range(G):
        real = buffer.sample(B_real)
        gen = aug_buffer.sample(B_aug)
        batches.append(losses_mod.Transition(*[
            None if getattr(real, f) is None
            else np.concatenate([getattr(real, f), getattr(gen, f)], axis=0)
            for f in losses_mod.Transition._fields]))
    stacked = losses_mod.Transition(*[
        None if getattr(batches[0], field) is None
        else jnp.asarray(np.stack([getattr(b, field) for b in batches],
                                  axis=0))
        for field in losses_mod.Transition._fields])
    return stacked

  # --- Main loop ---
  env_steps = start_step
  last_log = start_step
  last_eval = start_step
  last_ckpt = start_step
  t0 = time.time()
  metrics_history = []
  best_success = -1.0
  ckpt_every = config.ckpt_every_steps or config.eval_every_steps

  # Milestone checkpoints (init/early/mid/final) + optional TensorBoard mirror.
  saved_phases = set()
  if config.ckpt_dir:
    ckpt_mod.save_named(config.ckpt_dir, 'init', env_steps, state)
    saved_phases.add('init')
  writer = None
  if config.tensorboard:
    tb_dir = os.path.join(config.ckpt_dir or '.', 'tb')
    try:
      # Use tensorboardX (pure-python). Do NOT use torch.utils.tensorboard:
      # importing torch alongside JAX on the same GPU can hard-crash the kernel.
      from tensorboardX import SummaryWriter
      writer = SummaryWriter(tb_dir)
    except Exception as ex:  # pylint: disable=broad-except
      print(f'  [tensorboard requested but tensorboardX unavailable ({ex}); '
            'skipping TB. `pip install tensorboardX`]')

  # Maze-scalar hooks (guarded so they never block training).
  is_point_maze = config.env_name.startswith('point_')
  is_antmaze = config.env_name.startswith('antmaze_')
  def _maze_scalars():
    from crl import report_maze
    def _mp(s, g, memo):
      obs = np.concatenate([s, g]).astype(np.float32)
      return np.asarray(eval_act_fn(state.policy_params, jnp.asarray(obs[None]))[0])
    a = report_maze.eval_scalars(eval_env, _mp, episodes=min(config.eval_episodes, 30))
    keep = ('success@2.0', 'success@1.0', 'success@0.5', 'spl', 'collisions',
            'wp_completion')
    return {'maze_' + k: float(v) for k, v in a.items()
            if k in keep and v is not None}
  def _antmaze_scalars():
    from crl import report_antmaze
    def _p(flat):
      return np.asarray(eval_act_fn(state.policy_params, jnp.asarray(flat[None]))[0])
    eps = [report_antmaze.rollout(eval_env, _p)
           for _ in range(min(config.eval_episodes, 3))]
    a = report_antmaze.aggregate(eps)
    return {'ant_action_saturation': a['action_saturation_mean'],
            'ant_torso_height': a['mean_z_mean'],
            'ant_fall_fraction': a['fell_mean'],
            'ant_goal_velocity': a['goal_directed_velocity_mean']}

  # Restore the replay snapshot for staged/multi-stage runs (exact resume).
  learner_updates = 0
  replay_path = (os.path.join(config.ckpt_dir, 'replay.npz')
                 if config.ckpt_dir else None)
  if config.save_replay and config.resume and replay_path and \
     os.path.exists(replay_path):
    n = buffer.load(replay_path)
    print(f'Restored replay snapshot: {n} episodes '
          f'({buffer.ready_steps} transitions).')

  iteration = 0
  aug_note = ''
  while env_steps < config.max_number_of_steps:
    # Frozen dataset: no collection, ever; env_steps is the gradient clock,
    # advancing one episode's worth of learner budget per iteration.
    env_steps += config.max_episode_steps

    # Resample the generated trajectories from the CURRENT policy, before
    # this iteration's gradient steps consume them. With
    # dyn_off_diagonal='agent' the counterfactual block a' is the learner's
    # own action on the steps the sum_i w_i(s)**2 coin sends off-diagonal, so
    # the generated data is only valid for the parameters it was generated
    # with; dyn_augment_refresh_every controls how stale it is allowed to
    # get. A no-op when augmentation is off.
    aug_note = refresh_aug_buffer(state.policy_params, iteration) or aug_note
    iteration += 1

    # Learn. Ratio preserved: 1 update-batch per env step on the gradient
    # clock, i.e. one episode's worth of updates per iteration.
    metrics = {}
    if buffer.ready_steps >= min_replay:
      learner_steps = max(1, config.updates_per_step *
                          config.max_episode_steps // G)
      for _ in range(learner_steps):
        state, metrics = multi_update(state, sample_G())
      learner_updates += learner_steps * G

    # Numerical guard (opt-in): abort on non-finite / exploding learner state.
    if config.guard_abort and metrics:
      reason = None
      m = {k: float(v) for k, v in metrics.items()}
      for k in ('actor_loss', 'critic_loss', 'logits_pos', 'logits_neg',
                'alpha', 'alpha_loss'):
        if k in m and not np.isfinite(m[k]):
          reason = f'non-finite {k}={m[k]}'
          break
      if reason is None and abs(m.get('actor_loss', 0.0)) > config.guard_actor_loss_max:
        reason = (f'|actor_loss|={abs(m["actor_loss"]):.3g} > '
                  f'{config.guard_actor_loss_max:.3g}')
      if reason is None:
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in
                     jax.tree_util.tree_leaves((state.policy_params,
                                                state.q_params)))
        if not finite:
          reason = 'non-finite policy/critic parameters'
      if reason is not None:
        print(f'GUARD_ABORT at step {env_steps}: {reason}', flush=True)
        metrics_history.append({'step': int(env_steps), 'guard_abort': reason})
        if config.ckpt_dir:
          ckpt_mod.save_named(config.ckpt_dir, 'abort', env_steps, state)
          best_success = ckpt_mod.save_checkpoint(
              config.ckpt_dir, env_steps, state, metrics_history, None,
              best_success)
        break

    # Logging.
    if env_steps - last_log >= config.log_every_steps:
      sps = (env_steps) / (time.time() - t0)
      msg = f'[step {env_steps:>8}] sps={sps:6.0f}{aug_note}'
      if metrics:
        m = {k: float(v) for k, v in metrics.items()}
        msg += (f' critic={m.get("critic_loss", 0):.3f}'
                f' actor={m.get("actor_loss", 0):.3f}'
                f' cat_acc={m.get("categorical_accuracy", 0):.3f}')
        if 'bc_nll' in m:
          msg += f' bc_nll={m["bc_nll"]:.3f}'
      elif buffer.ready_steps < min_replay:
        msg += f' (filling buffer {buffer.ready_steps}/{min_replay})'
      print(msg, flush=True)
      last_log = env_steps

    # Eval.
    if env_steps - last_eval >= config.eval_every_steps:
      # Snapshot the eval-env step counter BEFORE evaluate() so the offline
      # contract can check evaluate()'s OWN consumption via a delta -- the
      # read-only maze/antmaze scalar rollouts below also legitimately step the
      # eval env (never touching replay), so an absolute count would be wrong.
      eval_n_before = offline_eval_steps['n']
      if config.physical_eval_push:
        # FetchPush image run: score on PHYSICAL object-goal coordinates, never
        # flattened image-L2 (which is meaningless as a control metric).
        push = [evaluate_push_physical(
            env, eval_act_fn, state.policy_params, config.eval_episodes,
            np_rng) for env in eval_envs]
        per_seed = [p[0] for p in push]
        succ, fdist, mdist = (float(np.mean(v)) for v in zip(*push))
        rstats = {}
        dist_tag = 'phys_dist'
      else:
        succ, fdist, mdist, rstats, per_seed = evaluate_seeds(
            eval_envs, eval_act_fn, state.policy_params, config.eval_episodes,
            np_rng, config.action_dim, config.obs_dim, config.start_index,
            config.end_index, config.goal_indices)
        dist_tag = 'final_dist'
      print(f'  >> EVAL step {env_steps}: success_rate={succ:.3f} '
            f'(std {np.std(per_seed):.3f} over {len(per_seed)} eval seeds x '
            f'{config.eval_episodes} eps: '
            + ' '.join(f'{p:.3f}' for p in per_seed) + ') '
            f'{dist_tag}={fdist:.3f} min_dist={mdist:.3f}', flush=True)
      if rstats:
        # The unconditional success rate above is pinned to the hazard draw;
        # these are the numbers that respond to the policy.
        print(f'     route: detour={rstats["route_detour_frac"]:.3f} '
              f'shortcut={rstats["route_shortcut_frac"]:.3f} '
              f'undecided={rstats["route_undecided_frac"]:.3f} | '
              f'died={rstats["death_frac"]:.3f} '
              f'armed={rstats["hazard_armed_frac"]:.3f} | '
              f'succ|armed={rstats["success_given_armed"]:.3f} '
              f'succ|clear={rstats["success_given_clear"]:.3f} | '
              f'mouth1={rstats["mouth_step_1"]:.0f} '
              f'band1={rstats["band_entry_step_1"]:.0f}', flush=True)
      last_eval = env_steps
      # HARD offline contract (see the audit block above): the frozen dataset
      # is immutable (pointers + full content hash unchanged), and evaluate()
      # consumed EXACTLY eval_num_seeds*eval_episodes*max_episode_steps
      # eval-env steps this
      # eval. There is no collection env at all and the buffer is frozen, so
      # collection is structurally impossible.
      assert (buffer._num_eps, buffer.ready_steps) == offline_frozen_ptr, \
          'offline contract violated: replay pointers changed'
      assert buffer.content_sha256() == offline_frozen_sha, \
          'offline contract violated: replay CONTENT changed'
      offline_eval_steps['evals'] += 1
      consumed = offline_eval_steps['n'] - eval_n_before
      expected = (len(eval_envs) * config.eval_episodes
                  * eval_env.max_episode_steps)
      assert consumed == expected, (
          'offline contract violated: evaluate() eval-env steps '
          f'{consumed} != {expected} (per-eval)')
      rec = {'step': int(env_steps), 'success': float(succ),
             'final_dist': float(fdist), 'min_dist': float(mdist),
             'success_std': float(np.std(per_seed)),
             'eval_num_seeds': len(per_seed),
             **{f'success_seed{k}': float(p) for k, p in enumerate(per_seed)},
             'learner_updates': int(learner_updates),
             'num_actors': 0,
             'per_actor_steps': 0}
      # Non-finite route stats are MEANINGFUL ABSENCE, not a numerical fault:
      # mouth_step_1 is nan when no episode reached that mouth at all. Dropping
      # them keeps the key out of metrics.json for that eval (and out of the
      # NaN guard below, which exists to catch a diverging learner).
      rec.update({k: float(v) for k, v in rstats.items()
                  if np.isfinite(v)})
      if config.physical_eval_push:
        # Unambiguous physical aliases (final_dist/min_dist above are already
        # physical in this mode; image-L2 is never logged).
        rec['physical_success_rate'] = float(succ)
        rec['physical_final_object_goal_distance'] = float(fdist)
        rec['physical_min_object_goal_distance'] = float(mdist)
      rec.update({k: float(v) for k, v in metrics.items()})
      if is_point_maze:
        try:
          rec.update(_maze_scalars())
        except Exception as ex:  # pylint: disable=broad-except
          print('  [maze scalars skipped]', ex)
      if is_antmaze:
        try:
          rec.update(_antmaze_scalars())
        except Exception as ex:  # pylint: disable=broad-except
          print('  [antmaze scalars skipped]', ex)
      # NaN guard: surface non-finite metrics immediately.
      if not np.all(np.isfinite([v for v in rec.values()
                                 if isinstance(v, (int, float))])):
        print(f'  [WARN] non-finite metric at step {env_steps}', flush=True)
      metrics_history.append(rec)
      if writer is not None:
        for k, v in rec.items():
          if isinstance(v, (int, float)):
            writer.add_scalar(k.replace('@', '_'), float(v), env_steps)

      # Periodic checkpoint to (Drive) ckpt_dir -- not just once at the end.
      if config.ckpt_dir and env_steps - last_ckpt >= ckpt_every:
        best_success = ckpt_mod.save_checkpoint(
            config.ckpt_dir, env_steps, state, metrics_history, succ,
            best_success, strict=config.best_strict_improvement)
        last_ckpt = env_steps

      # Milestone checkpoints at 25% / 50% of the budget.
      if config.ckpt_dir:
        for nm, frac in (('early', 0.25), ('mid', 0.5)):
          if nm not in saved_phases and env_steps >= frac * config.max_number_of_steps:
            ckpt_mod.save_named(config.ckpt_dir, nm, env_steps, state)
            saved_phases.add(nm)

      # Explicit step-numbered milestones (e.g. 10k/20k/30k/50k/70k) saved as
      # <step>.pkl the first eval at or past each target.
      if config.ckpt_dir and config.ckpt_milestone_steps:
        for ms in config.ckpt_milestone_steps:
          tag = f'ckpt_{int(ms)}'
          if tag not in saved_phases and env_steps >= ms:
            ckpt_mod.save_named(config.ckpt_dir, str(int(ms)), env_steps, state)
            saved_phases.add(tag)

  # Final offline-contract check (content hash) before the last save.
  assert buffer.content_sha256() == offline_frozen_sha, \
      'offline contract violated: replay CONTENT changed by end of run'
  # Final save (also runs after a guard abort's break).
  if config.ckpt_dir:
    ckpt_mod.save_named(config.ckpt_dir, 'final', env_steps, state)
    ckpt_mod.save_checkpoint(config.ckpt_dir, env_steps, state,
                             metrics_history, None, best_success)
  if config.save_replay and replay_path:
    buffer.save(replay_path)
    print(f'Replay snapshot saved: {buffer.ready_steps} transitions '
          f'-> {replay_path}', flush=True)
  if writer is not None:
    writer.close()
  return state


def main():
  args = _build_arg_parser().parse_args()
  config = _apply_overrides(Config(), args)
  ETT_train(config)


if __name__ == '__main__':
  main()
