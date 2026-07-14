"""Formal OFFLINE image FetchPush conedir qualification -- original-style actor.

Actor objective (verified in crl/losses.py):
    actor_loss = bc_coef * BC_NLL(data_action | state, future_goal)
               + (1 - bc_coef) * (-min_Q(state, policy_action, future_goal))
with the requested target config:
    entropy_coefficient = 0.0   (fixed alpha=0, NO adaptive-alpha optimizer)
    bc_coef             = 0.05  (5% goal-conditioned BC anchor)
    twin_q              = True  (actor uses the pessimistic min over the twins)
    random_goals        = 0.0   (positives come from the SAME offline trajectory)
Strictly offline: the frozen .npz IS the buffer; no collection env is created and
evaluation cannot mutate replay (enforced by crl/train.py's offline contract).

This driver:
  1. builds the exact config and runs an abort-on-failure PREFLIGHT,
  2. trains 100k gradient steps (physical eval every 10k; milestone checkpoints
     init/10k/20k/30k/50k/70k/final/latest/best; strict physical best),
  3. POST-hoc, on a FIXED evaluation set, writes rich PHYSICAL metrics per
     checkpoint (fixed_eval.csv), checkpoint diagnostics at 20k/50k/100k
     (checkpoint_diagnostics.csv), and greedy rollout GIFs at 20k/50k/100k,
  4. emits report.md + summary.json with the qualification verdict.

All distances are PHYSICAL object-goal coordinates from the simulator; flattened
image-L2 is never used as a control metric.

Run (Colab GPU):
  python -m scripts.qualify_image_offline_original_style \
      --dataset /content/drive/MyDrive/.../push_image_conedir_<...>.npz \
      --run_dir /content/drive/MyDrive/.../fetch_push_image_conedir_off_bc005_alpha0_twin_s0 \
      --seed 0 --steps 100000

Smoke self-test (tiny, local, proves the plumbing):
  MUJOCO_GL=glfw python -m scripts.qualify_image_offline_original_style \
      --dataset datasets/push_image_conedir_smoke.npz \
      --run_dir /tmp/img_qual_smoke --smoke
"""
import argparse
import hashlib
import json
import os

import numpy as np

IMG = 64 * 64 * 3
CONTACT_THRESH = 0.06     # gripper-object distance counted as contact (m)
MOVE_EPS = 1e-3           # per-step object motion counted as "moving" (m)
TABLE_Z = 0.40            # object below this z counts as fallen/invalid


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight(dataset, cfg, run_dir):
  """Verify config + dataset before training. Returns (ok, audit_dict)."""
  import jax
  from crl import envs as envs_mod, networks as networks_mod, losses as losses_mod
  import optax

  a = {}
  # dataset fingerprint
  sha = hashlib.sha256(open(dataset, 'rb').read()).hexdigest()
  z = np.load(dataset, allow_pickle=True)
  obs, act = z['obs'], z['act']
  a['1_dataset_path'] = os.path.abspath(dataset)
  a['2_sha256'] = sha
  a['3_episodes'] = int(obs.shape[0])
  a['3_transitions'] = int(obs.shape[0] * (obs.shape[1] - 1))
  a['4_obs_shape'] = list(obs.shape)
  a['4_obs_dtype'] = str(obs.dtype)
  a['5_action_shape'] = list(act.shape)
  a['5_action_range'] = [float(act.min()), float(act.max())]
  a['6_action_aligned'] = (obs.shape[0] == act.shape[0]
                           and obs.shape[1] == act.shape[1])
  # config flags
  a['10_random_goals'] = cfg.random_goals
  a['11_twin_q'] = cfg.twin_q
  a['12_bc_coef'] = cfg.bc_coef
  a['13_entropy_coefficient'] = cfg.entropy_coefficient
  a['13_adaptive_alpha'] = (cfg.entropy_coefficient is None)
  a['14_physical_eval_push'] = cfg.physical_eval_push

  # empirically confirm objective + no adaptive-alpha optimizer + twin-min
  env = envs_mod.make_env(cfg.env_name, cfg, seed=cfg.seed)  # fills dims
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=True)
  s, e = cfg.start_index, cfg.end_index
  o2g = (lambda x: x[:, s:] if e == -1 else x[:, s:e])
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, o2g, optax.adam(3e-4, eps=1e-7), optax.adam(3e-4, eps=1e-7))
  st = init_state(jax.random.PRNGKey(0))
  a['13_no_alpha_optimizer'] = (st.alpha_params is None
                                and st.alpha_optimizer_state is None)

  checks = {
      '1_dataset_exists': os.path.exists(dataset),
      '6_action_aligned': a['6_action_aligned'],
      '10_random_goals_0': cfg.random_goals == 0.0,
      '11_twin_q_true': cfg.twin_q is True,
      '12_bc_coef_0.05': abs(cfg.bc_coef - 0.05) < 1e-9,
      '13_fixed_alpha0': cfg.entropy_coefficient == 0.0,
      '13_no_adaptive_optimizer': a['13_no_alpha_optimizer'],
      '14_physical_eval_on': cfg.physical_eval_push is True,
      '4_obs_uint8_image': (str(obs.dtype) == 'uint8'
                            and obs.shape[-1] == 2 * IMG),
      '5_action_in_range': a['5_action_range'][0] >= -1.0 - 1e-6
                           and a['5_action_range'][1] <= 1.0 + 1e-6,
  }
  a['checks'] = checks
  a['verdict'] = 'PASS' if all(checks.values()) else 'ABORT_PREFLIGHT_FAILED'
  os.makedirs(run_dir, exist_ok=True)
  return all(checks.values()), a


# --------------------------------------------------------------------------- #
# Rich physical evaluation on a FIXED set of episodes
# --------------------------------------------------------------------------- #
def _greedy_fn(nets):
  import jax, jax.numpy as jnp
  @jax.jit
  def g(params, o):
    return nets.sample_eval(nets.policy_network.apply(params, o), None)
  return g


def physical_eval(env, nets, params, episodes, seed, gif_path=None):
  """Fixed-seed greedy rollouts scored on PHYSICAL coordinates. Returns a dict
  of all required fields; optionally saves a rollout GIF."""
  import jax.numpy as jnp
  g = _greedy_fn(nets)
  u = env._env.unwrapped
  env._rng = np.random.default_rng(seed)   # fix the episode goals/starts
  succ, fdist, mdist, disp = [], [], [], []
  grip_min, contact_steps, moving_frac, fall = [], [], [], []
  acts = []
  frames = []
  for ep in range(episodes):
    env.reset()
    desired = np.asarray(env._desired, dtype=np.float32)
    obs = np.concatenate([env._frame(), env._goal_img])
    d0 = u._get_obs()
    obj0 = np.asarray(d0['achieved_goal'], np.float32)
    dists, gmins, csteps, moves, fell = [], [], 0, 0, False
    prev_obj = obj0
    for t in range(env.max_episode_steps):
      a = np.asarray(g(params, jnp.asarray(obs[None]))[0])
      acts.append(a.copy())
      obs, r, _, _ = env.step(a)
      dd = u._get_obs()
      obj = np.asarray(dd['achieved_goal'], np.float32)
      grip = np.asarray(dd['observation'][0:3], np.float32)
      dists.append(float(np.linalg.norm(obj - desired)))
      gd = float(np.linalg.norm(grip - obj))
      gmins.append(gd)
      csteps += int(gd < CONTACT_THRESH)
      moves += int(np.linalg.norm(obj - prev_obj) > MOVE_EPS)
      fell = fell or bool(obj[2] < TABLE_Z)
      prev_obj = obj
      if gif_path and ep < 2:
        frames.append(np.kron(obs[:IMG].reshape(64, 64, 3),
                              np.ones((4, 4, 1))).astype(np.uint8))
    hit = float(max([r] + [0.0]))  # r is last-step; success tracked below
    # success = any step within threshold (physical): recompute from dists
    succ.append(float(min(dists) < env._success_threshold))
    fdist.append(dists[-1]); mdist.append(min(dists))
    disp.append(float(np.linalg.norm(prev_obj - obj0)))
    grip_min.append(min(gmins)); contact_steps.append(csteps)
    moving_frac.append(moves / env.max_episode_steps); fall.append(int(fell))
  acts = np.stack(acts, 0)
  if gif_path and frames:
    import imageio
    imageio.mimsave(gif_path, frames, duration=0.08)
  return {
      'physical_success_rate': float(np.mean(succ)),
      'physical_final_object_goal_distance': float(np.mean(fdist)),
      'physical_min_object_goal_distance': float(np.mean(mdist)),
      'object_displacement': float(np.mean(disp)),
      'minimum_gripper_object_distance': float(np.mean(grip_min)),
      'contact_steps': float(np.mean(contact_steps)),
      'moving_object_fraction': float(np.mean(moving_frac)),
      'fall_or_invalid_rate': float(np.mean(fall)),
      'action_mean_per_dimension': [float(x) for x in acts.mean(0)],
      'action_std_per_dimension': [float(x) for x in acts.std(0)],
      'action_saturation_fraction': float(np.mean(np.abs(acts) > 0.99)),
  }


# --------------------------------------------------------------------------- #
# Checkpoint diagnostics on FIXED dataset (image_state, image_goal) inputs
# --------------------------------------------------------------------------- #
def checkpoint_diagnostics(nets, policy_params, q_params, dataset, n_inputs, seed):
  """For fixed dataset inputs: policy mode action, per-checkpoint action
  sensitivity, policy scale, and critic scores for {policy, zero, aligned-data}
  action. The reference action is the ALIGNED DATASET action (never invented)."""
  import jax, jax.numpy as jnp
  z = np.load(dataset, allow_pickle=True)
  obs, act = z['obs'], z['act']
  rng = np.random.default_rng(seed)
  E, L = obs.shape[0], obs.shape[1] - 1
  picks = [(int(rng.integers(0, E)), int(rng.integers(1, L))) for _ in range(n_inputs)]

  @jax.jit
  def pol(p, o):
    dp = nets.policy_network.apply(p, o)
    return jnp.tanh(dp.loc), dp.scale
  @jax.jit
  def crit(p, o, a):
    q = nets.q_network.apply(p, o, a)
    if q.ndim == 3:              # twin -> pessimistic min, diagonal element
      q = jnp.min(q, axis=-1)
    return jnp.diag(q)

  rows, modes = [], []
  for i, (ep, t) in enumerate(picks):
    o = obs[ep, t].astype(np.float32)[None]
    a_data = act[ep, t].astype(np.float32)[None]
    mode, scale = pol(policy_params, jnp.asarray(o))
    mode = np.asarray(mode)[0]; scale = np.asarray(scale)[0]
    modes.append(mode)
    q_pol = float(crit(q_params, jnp.asarray(o), jnp.asarray(mode[None]))[0])
    q_zero = float(crit(q_params, jnp.asarray(o), jnp.zeros((1, act.shape[-1]), np.float32))[0])
    q_data = float(crit(q_params, jnp.asarray(o), jnp.asarray(a_data))[0])
    rows.append({
        'input_id': f'ep{ep}_t{t}',
        'policy_mode_action': [round(float(x), 4) for x in mode],
        'policy_scale_median': float(np.median(scale)),
        'critic_score_policy_action': q_pol,
        'critic_score_zero_action': q_zero,
        'critic_score_aligned_data_action': q_data,
    })
  modes = np.stack(modes, 0)
  # input-sensitivity: mean over action dims of the std across the fixed inputs.
  sensitivity = float(np.mean(modes.std(0)))
  return rows, sensitivity


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def decide_verdict(final_eval, diag_100k_sensitivity, dataset_meta):
  s = final_eval['physical_success_rate']
  contact = final_eval['contact_steps'] > 0
  disp = final_eval['object_displacement'] > 0.01
  sat = final_eval['action_saturation_fraction']
  sens = diag_100k_sensitivity
  scale_ok = True  # scale collapse is caught via saturation + sensitivity below
  # dataset sufficiency: behavior policy actually pushed to the goal.
  ds_ok = float(dataset_meta.get('behavior_success', 0.0)) >= 0.5
  if not ds_ok:
    return 'OFFLINE_DATASET_INSUFFICIENT'
  saturating = sat > 0.5 and sens < 1e-2
  if saturating:
    return 'ACTOR_STILL_SATURATES'
  if s > 0.2 and contact and disp and sens > 1e-2:
    return 'IMAGE_OFFLINE_CONTROL_QUALIFIED'
  if sat < 0.5 and contact and disp and s <= 0.2:
    return 'ACTOR_BC_STABILIZED_BUT_CONTROL_WEAK'
  if not (contact and disp):
    return 'CRITIC_RETRIEVAL_WITHOUT_CONTROL'
  return 'INCONCLUSIVE'


# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', required=True)
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--steps', type=int, default=100_000)
  ap.add_argument('--eval_episodes', type=int, default=50)
  ap.add_argument('--resume', action='store_true')
  ap.add_argument('--smoke', action='store_true',
                  help='tiny fast self-test of the whole pipeline')
  ap.add_argument('--posthoc-only', dest='posthoc_only', action='store_true',
                  help='skip preflight+training; regenerate the physical fixed-eval, '
                       'checkpoint diagnostics, GIFs and verdict from checkpoints '
                       'already in --run_dir (e.g. after a Colab disconnect).')
  ap.add_argument('--out', default='artifacts/image_conedir_offline_original_style')
  args = ap.parse_args()

  from crl.config import Config
  from crl.train import train
  from crl import envs as envs_mod, networks as networks_mod
  from crl import checkpoint as ckpt_mod

  if args.smoke:
    steps, evalp, milestones, diag_steps, n_inputs = 200, 4, (100,), (100, 200), 6
    eval_every = 100
  else:
    steps, evalp = args.steps, args.eval_episodes
    milestones = (10_000, 20_000, 30_000, 50_000, 70_000)
    diag_steps = (20_000, 50_000, 100_000)
    n_inputs = 24
    eval_every = 10_000

  cfg = Config(
      env_name='fetch_push_image_conedir',
      entropy_coefficient=0.0, bc_coef=0.05, twin_q=True, random_goals=0.0,
      use_td=False, use_cpc=False, use_gcbc=False,
      offline_dataset=args.dataset, batch_size=256,
      max_number_of_steps=steps, eval_every_steps=eval_every,
      eval_episodes=evalp, log_every_steps=max(1000, eval_every // 5),
      seed=args.seed, ckpt_dir=args.run_dir, resume=args.resume,
      tensorboard=True, guard_abort=True,
      physical_eval_push=True, best_strict_improvement=True,
      ckpt_milestone_steps=milestones,
  )
  # hard invariants (fail loud rather than silently drift)
  assert cfg.entropy_coefficient == 0.0 and cfg.bc_coef == 0.05
  assert cfg.twin_q and cfg.random_goals == 0.0 and cfg.physical_eval_push

  os.makedirs(args.out, exist_ok=True)
  if args.posthoc_only:
    # Reuse checkpoints already in --run_dir; do NOT preflight or retrain. Fill
    # cfg dims from the env (preflight/train normally do this) so the post-hoc
    # network build below matches the trained graph.
    envs_mod.make_env(cfg.env_name, cfg, seed=cfg.seed)
    print(f'POSTHOC-ONLY: skipping preflight+train; using checkpoints in '
          f'{args.run_dir}', flush=True)
  else:
    ok, audit = preflight(args.dataset, cfg, args.run_dir)
    with open(os.path.join(args.out, 'preflight_audit.json'), 'w') as f:
      json.dump(audit, f, indent=2)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
      json.dump({'target_config': cfg.__dict__,
                 'objective': 'actor_loss = 0.05*BC_NLL + 0.95*(-min_Q)'},
                f, indent=2, default=str)
    if not ok:
      print('PREFLIGHT FAILED -> aborting before training.', audit['checks'])
      return

    # ---------------------------------------------------------------- TRAIN
    train(cfg)

  # ------------------------------------------------------- POST-HOC ANALYSIS
  rcfg = Config(env_name=cfg.env_name)
  env = envs_mod.make_env(cfg.env_name, rcfg, seed=args.seed + 777)
  # Networks MUST match the trained config (esp. twin_q=True), not Config
  # defaults -- otherwise the loaded q_params won't fit the critic graph.
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=True)

  # milestone step -> checkpoint file
  ck = {'init': 'init.pkl', 'best': 'best.pkl', 'final': 'final.pkl'}
  for ms in milestones:
    ck[str(ms)] = f'{ms}.pkl'
  ck[str(steps)] = 'final.pkl'

  # ---- fixed_eval.csv: rich PHYSICAL metrics per checkpoint (fixed 50-ep set)
  fixed_rows = []
  for name, fn in ck.items():
    path = os.path.join(args.run_dir, fn)
    if not os.path.exists(path):
      continue
    step, tstate = ckpt_mod.load_checkpoint(path)
    gif = None
    if name in (str(x) for x in diag_steps):
      gif = os.path.join(args.run_dir, f'rollout_{name}.gif')
    m = physical_eval(env, nets, tstate.policy_params, evalp, args.seed + 100,
                      gif_path=gif)
    m['checkpoint'] = name
    m['step'] = int(step)
    fixed_rows.append(m)

  cols = ['checkpoint', 'step', 'physical_success_rate',
          'physical_final_object_goal_distance', 'physical_min_object_goal_distance',
          'object_displacement', 'minimum_gripper_object_distance', 'contact_steps',
          'moving_object_fraction', 'fall_or_invalid_rate',
          'action_mean_per_dimension', 'action_std_per_dimension',
          'action_saturation_fraction']
  with open(os.path.join(args.out, 'fixed_eval.csv'), 'w') as f:
    f.write(','.join(cols) + '\n')
    for r in fixed_rows:
      f.write(','.join(('"%s"' % r[c] if isinstance(r[c], list) else str(r[c]))
                       for c in cols) + '\n')

  # ---- checkpoint_diagnostics.csv at 20k/50k/100k
  diag_cols = ['checkpoint_step', 'input_id', 'policy_mode_action',
               'policy_action_sensitivity', 'policy_scale_median',
               'critic_score_policy_action', 'critic_score_zero_action',
               'critic_score_aligned_data_action']
  diag_rows = []
  sens_by_step = {}
  for ds in diag_steps:
    fn = ck.get(str(ds), 'final.pkl' if ds == steps else None)
    if fn is None:
      continue
    path = os.path.join(args.run_dir, fn)
    if not os.path.exists(path):
      continue
    step, tstate = ckpt_mod.load_checkpoint(path)
    rows, sens = checkpoint_diagnostics(nets, tstate.policy_params, tstate.q_params,
                                        args.dataset, n_inputs, args.seed + 9)
    sens_by_step[int(ds)] = sens
    for r in rows:
      r['checkpoint_step'] = int(step)
      r['policy_action_sensitivity'] = sens
      diag_rows.append(r)
  with open(os.path.join(args.out, 'checkpoint_diagnostics.csv'), 'w') as f:
    f.write(','.join(diag_cols) + '\n')
    for r in diag_rows:
      f.write(','.join(('"%s"' % r[c] if isinstance(r[c], list) else str(r[c]))
                       for c in diag_cols) + '\n')

  # ---- verdict + report
  z = np.load(args.dataset, allow_pickle=True)
  ds_meta = json.loads(str(z['meta'])) if 'meta' in z else {}
  final_eval = next((r for r in fixed_rows if r['checkpoint'] in ('final', str(steps))),
                    fixed_rows[-1] if fixed_rows else {})
  verdict = decide_verdict(final_eval, sens_by_step.get(diag_steps[-1], 0.0), ds_meta)

  summary = {
      'run_dir': args.run_dir, 'dataset': args.dataset, 'seed': args.seed,
      'steps': steps, 'objective': 'actor_loss = 0.05*BC_NLL + 0.95*(-min_Q)',
      'qualification_verdict': verdict,
      'final_checkpoint_physical': final_eval,
      'diagnostic_sensitivity_by_step': sens_by_step,
      'dataset_meta': ds_meta,
  }
  with open(os.path.join(args.out, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

  with open(os.path.join(args.out, 'report.md'), 'w') as f:
    f.write(_render_report(summary, fixed_rows, diag_rows))
  print('QUALIFICATION VERDICT:', verdict)


def _render_report(summary, fixed_rows, diag_rows):
  fe = summary['final_checkpoint_physical']
  L = ['# Offline image FetchPush conedir -- original-style qualification\n',
       f"**Verdict:** `{summary['qualification_verdict']}`\n",
       f"Objective: `{summary['objective']}` (fixed alpha=0, bc=0.05, twin-min, "
       "random_goals=0, strictly offline).\n",
       '## Final checkpoint (physical, fixed eval set)\n']
  for k in ('physical_success_rate', 'object_displacement', 'contact_steps',
            'minimum_gripper_object_distance', 'moving_object_fraction',
            'fall_or_invalid_rate', 'action_saturation_fraction'):
    L.append(f'- {k}: {fe.get(k)}')
  L.append('\n## Required answers\n')
  L.append('1. Avoided previous actor collapse? '
           'See action_saturation_fraction + sensitivity above.')
  L.append('2. Learned image pushing? See physical_success_rate + object_displacement.')
  L.append('3. Dataset sufficient? behavior_success='
           f"{summary['dataset_meta'].get('behavior_success')}.")
  L.append('4. BC dominate or anchor? bc weight fixed 0.05 (see metrics.json '
           'bc_loss_weighted vs critic_actor_term_weighted).')
  L.append('5. Critic action ranking vs retrieval? compare '
           'critic_score_aligned_data_action vs _zero_action vs _policy_action '
           'in checkpoint_diagnostics.csv.')
  L.append('6. Next experiment: chosen by the verdict (do NOT auto-launch).')
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  main()
