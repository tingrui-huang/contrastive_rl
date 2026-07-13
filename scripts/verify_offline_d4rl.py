"""Pre-training gates for the faithful OFFLINE antmaze-umaze-v2 path.

Proves, WITHOUT training the learner:
  G1  dataset->buffer BC-tuple alignment: exact RNG replication of
      TrajectoryBuffer.sample() shows every sampled (state, action, goal)
      is (obs[traj,i], act[traj,i], obs[traj,j>i]) of the SAME episode.
  G2  zero-padded goal contract: eval obs = [state(29), goal_xy, 0*27],
      goals drawn from the dataset's empirical eval_goals; reward=1 at goal.
  G3  twin-MIN (not mean) enters the actor objective: the real update_step's
      reported actor_loss equals a manual recomputation with jnp.min and
      differs from the jnp.mean version.
  G4  BC gradients increase log pi(a_dataset): 3 update steps at bc_coef=1
      strictly increase mean log-prob of dataset actions.
  G5  offline dry-run (max_number_of_steps=0): dataset ingested once, exact
      buffer sizing, 'env collection DISABLED' + frozen-sha stdout evidence,
      no learner step, no env interaction.
  G6  runtime hard-assertion wiring present in crl/train.py (poisoned
      collection env.step, replay sha check each eval, eval-step accounting).
  G7  run-config contract: bc_coef=0.05, twin_q=True, min semantics,
      random_goals=0.0, batch=1024, repr=16, hidden=(1024,1024), alpha=0,
      zero target entropy, 1M learner-step budget.

The exact offline actor objective this prepares (upstream master):
  L_actor = mean[ 0.05 * (-log pi(a_data | s,g))
                + 0.95 * (alpha*log pi(a|s,g) - min(Q1,Q2)(s,a,g)) ],
  alpha = 0, goals g = geometric future-state relabels (gamma=0.99),
  critic = binary NCE on twin logits.
"""
import io
import json
import os
import sys
import contextlib
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import optax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from crl.config import Config
from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import losses as losses_mod
from crl.replay import TrajectoryBuffer

# Dataset path is env-overridable so the SAME builder serves the local
# workstation run and the Colab run (which sets OFFLINE_NPZ=/content/...npz).
NPZ = os.environ.get(
    'OFFLINE_NPZ',
    r'D:\Users\trhua\Research\datasets\d4rl\antmaze_umaze_v2_offline.npz')
OUT = 'artifacts/offline_d4rl'
# Dry-run scratch: use the OS temp dir (Linux/Colab have no TEMP env, which
# previously defaulted to '.', writing the scratch INTO the repo and dirtying
# the tree so the next git checkout refused to run).
SCRATCH = os.environ.get('OFFLINE_DRYRUN_DIR', os.path.join(
    tempfile.gettempdir(), 'crl_offline_dryrun'))
GATE = {}


def build_offline_cfg(max_steps=1_000_000, ckpt_dir=''):
  """Canonical faithful-offline run config (upstream offline_ant recipe)."""
  return Config(
      env_name='offline_ant_umaze',
      offline_dataset=NPZ,
      # LayerNorm arm toggled by env (OFFLINE_LAYER_NORM=1); default off keeps
      # the faithful google-research recipe.
      use_layer_norm=os.environ.get('OFFLINE_LAYER_NORM', '0') == '1',
      num_actors=0,                         # OFFLINE: zero collection actors
      use_td=False, use_cpc=False,          # binary NCE
      twin_q=True,                          # 2 critics; actor uses min
      bc_coef=0.05,
      random_goals=0.0,
      entropy_coefficient=0.0,              # alpha = 0 (fixed)
      target_entropy=0.0,                   # must be unset with fixed alpha
      batch_size=1024, repr_dim=16, hidden_layer_sizes=(1024, 1024),
      discount=0.99, actor_learning_rate=3e-4, learning_rate=3e-4,
      max_number_of_steps=max_steps,        # OFFLINE: learner-update budget
      num_sgd_steps_per_step=4, updates_per_step=1,
      guard_abort=True,
      eval_every_steps=10_000, eval_episodes=10, log_every_steps=5_000,
      seed=0, ckpt_dir=ckpt_dir, tensorboard=bool(ckpt_dir))


def main():
  os.makedirs(OUT, exist_ok=True)

  with np.load(NPZ) as d:
    ds_obs = d['obs'].copy()
    ds_act = d['act'].copy()
    ds_goals = d['eval_goals'].copy()
  E, L, W = ds_obs.shape
  assert (E, L, W) == (1426, 701, 58), (E, L, W)

  # ---------------- G1: dataset->buffer BC-tuple alignment ----------------
  SEED = 123
  B = 512
  buf = TrajectoryBuffer(capacity_steps=E * L, ep_len_obs=L, full_obs_dim=W,
                         action_dim=8, obs_dim=29, start_index=0,
                         end_index=-1, discount=0.99, seed=SEED,
                         goal_indices=tuple(range(29)))
  for k in range(E):
    buf.add_episode(ds_obs[k], ds_act[k])
  # replicate sample()'s exact RNG draw order on a twin generator
  rng2 = np.random.default_rng(SEED)
  traj = rng2.integers(0, E, size=B)
  i = rng2.integers(0, L - 1, size=B)
  arange = np.arange(L)
  logp = np.where(arange[None, :] > i[:, None],
                  (arange[None, :] - i[:, None]) * np.log(0.99), -np.inf)
  g = -np.log(-np.log(rng2.uniform(size=logp.shape).clip(1e-20, 1.0)))
  j = np.argmax(logp + g, axis=1)
  tr = buf.sample(B)
  assert np.array_equal(tr.observation[:, :29], ds_obs[traj, i, :29])
  assert np.array_equal(tr.action, ds_act[traj, i])
  assert np.array_equal(tr.observation[:, 29:], ds_obs[traj, j, :29])
  assert np.array_equal(tr.next_observation[:, :29], ds_obs[traj, i + 1, :29])
  assert np.all(j > i)
  GATE['G1_bc_tuple_alignment'] = {
      'pass': True, 'pairs_checked': int(B),
      'note': ('sampled (s,a,g)=(obs[traj,i],act[traj,i],obs[traj,j>i]) '
               'bit-exact vs hdf5-derived npz; same-episode future goals')}
  print('G1 PASS: BC tuples trajectory-aligned (bit-exact,', B, 'pairs)')

  # ---------------- G2: zero-padded eval goal contract --------------------
  cfg = build_offline_cfg()
  env = envs_mod.make_env('offline_ant_umaze', cfg, seed=5)
  assert cfg.obs_dim == 29 and cfg.goal_dim == 29 and cfg.action_dim == 8
  assert cfg.max_episode_steps == 700
  goals_seen = []
  for _ in range(10):
    obs = env.reset()
    assert obs.shape == (58,)
    assert np.all(obs[31:] == 0.0), 'goal not zero-padded beyond xy'
    gxy = obs[29:31]
    match = np.isclose(ds_goals, gxy[None], atol=1e-5).all(1)
    assert match.any(), f'eval goal {gxy} not from the empirical dataset set'
    goals_seen.append(tuple(np.round(gxy, 3)))
  assert len(set(goals_seen)) >= 5, 'eval goals do not vary across resets'
  u = env._env.unwrapped
  u.data.qpos[:2] = np.asarray(u.goal)          # teleport onto the goal
  import mujoco
  mujoco.mj_forward(u.model, u.data)
  _, r, _, _ = env.step(np.zeros(8, np.float32))
  assert r == 1.0, f'reward at goal = {r}'
  GATE['G2_zero_padded_goal'] = {
      'pass': True, 'distinct_goals_in_10_resets': len(set(goals_seen)),
      'reward_at_goal': float(r)}
  print('G2 PASS: zero-padded XY goal contract exact; empirical eval goals;'
        ' reward=1 at goal')

  # ---------------- G3 + G4: objective gates on the real update path ------
  nets = networks_mod.make_networks(
      obs_dim=29, goal_dim=29, action_dim=8, repr_dim=16, repr_norm=False,
      repr_norm_temp=True, hidden_layer_sizes=(1024, 1024), twin_q=True,
      use_image_obs=False)
  pol_opt = optax.adam(3e-4, eps=1e-7)
  q_opt = optax.adam(3e-4, eps=1e-7)
  def obs_to_goal(states):
    return states[:, jnp.arange(29)]
  gate_cfg = build_offline_cfg()
  gate_cfg.obs_dim, gate_cfg.goal_dim, gate_cfg.action_dim = 29, 29, 8
  init_state, update_step = losses_mod.build_learner(
      nets, gate_cfg, obs_to_goal, pol_opt, q_opt)
  state = init_state(jax.random.PRNGKey(7))
  batch = buf.sample(256)
  trans = losses_mod.Transition(*[jnp.asarray(x) for x in batch])

  # manual recomputation with the SAME key split as update_step
  _, _, _, key_actor = jax.random.split(state.key, 4)
  dist = nets.policy_network.apply(state.policy_params, trans.observation)
  a_pi = nets.sample(dist, key_actor)
  logp = nets.log_prob(dist, a_pi)
  q = nets.q_network.apply(state.q_params, trans.observation, a_pi)
  assert q.ndim == 3 and q.shape[2] == 2, 'twin critic not active'
  qterm_min = 0.0 * logp - jnp.diag(jnp.min(q, axis=-1))
  qterm_mean = 0.0 * logp - jnp.diag(jnp.mean(q, axis=-1))
  bc_nll = -nets.log_prob(dist, trans.action)
  manual_min = float(jnp.mean(0.05 * bc_nll + 0.95 * qterm_min))
  manual_mean = float(jnp.mean(0.05 * bc_nll + 0.95 * qterm_mean))
  _, metrics = update_step(state, trans)
  reported = float(metrics['actor_loss'])
  assert abs(reported - manual_min) < 1e-3, (reported, manual_min)
  assert abs(reported - manual_mean) > 1e-4, \
      'min and mean agree; twin heads degenerate?'
  GATE['G3_twin_min_in_actor'] = {
      'pass': True, 'reported_actor_loss': reported,
      'manual_min_version': manual_min, 'manual_mean_version': manual_mean}
  print(f'G3 PASS: actor uses min(Q1,Q2): reported={reported:.6f} == '
        f'min-version={manual_min:.6f} != mean-version={manual_mean:.6f}')

  # G4: pure-BC gradient ascends log pi(a_data)
  bc_cfg = build_offline_cfg()
  bc_cfg.obs_dim, bc_cfg.goal_dim, bc_cfg.action_dim = 29, 29, 8
  bc_cfg.bc_coef = 1.0
  init2, update2 = losses_mod.build_learner(nets, bc_cfg, obs_to_goal,
                                            pol_opt, q_opt)
  st2 = init2(jax.random.PRNGKey(11))
  def mean_logp(params):
    d = nets.policy_network.apply(params, trans.observation)
    return float(jnp.mean(nets.log_prob(d, trans.action)))
  lp_hist = [mean_logp(st2.policy_params)]
  for _ in range(3):
    st2, m2 = update2(st2, trans)
    lp_hist.append(mean_logp(st2.policy_params))
  assert lp_hist[-1] > lp_hist[0], lp_hist
  assert all(b >= a - 1e-6 for a, b in zip(lp_hist, lp_hist[1:])), lp_hist
  GATE['G4_bc_gradient_ascends_logprob'] = {
      'pass': True, 'mean_logprob_trajectory': [round(v, 4) for v in lp_hist]}
  print('G4 PASS: BC gradient increases log pi(a_data):',
        [round(v, 3) for v in lp_hist])

  # ---------------- G5: offline dry-run (0 learner steps) -----------------
  os.makedirs(SCRATCH, exist_ok=True)
  dry_cfg = build_offline_cfg(max_steps=0, ckpt_dir=SCRATCH)
  dry_cfg.tensorboard = False
  from crl.train import train
  buf_stdout = io.StringIO()
  with contextlib.redirect_stdout(buf_stdout):
    train(dry_cfg)
  out = buf_stdout.getvalue()
  assert 'OFFLINE AUDIT (pre-training gates):' in out
  assert 'FAIL' not in out, out
  assert 'eps=1426' in out and 'trans=998200' in out
  assert '>> EVAL' not in out and 'GUARD_ABORT' not in out
  GATE['G5_dry_run'] = {
      'pass': True,
      'evidence': [l.strip() for l in out.splitlines()
                   if 'PASS' in l or 'sha256=' in l or 'offline' in l.lower()]}
  print('G5 PASS: dry run ingests 1426 eps / 998200 transitions via the '
        'static offline audit, no env interaction, no learner step')

  # ---------------- G6: runtime hard-assertion wiring ---------------------
  src = open(os.path.join(os.path.dirname(_HERE), 'crl', 'train.py'),
             encoding='utf-8').read()
  need = ['env = None',                       # no training env is ever built
          'offline mode rejects num_actors',
          'buffer.content_sha256() == offline_frozen_sha',
          'consumed == expected',
          'require_same_dataset_hash']
  missing = [s for s in need if s not in src]
  assert not missing, missing
  src_r = open(os.path.join(os.path.dirname(_HERE), 'crl', 'replay.py'),
               encoding='utf-8').read()
  assert 'def freeze' in src_r and 'frozen' in src_r
  GATE['G6_runtime_asserts_armed'] = {'pass': True, 'checks': need + [
      'TrajectoryBuffer.freeze (add_episode raises when frozen)']}
  print('G6 PASS: structural no-env offline mode + frozen buffer + per-eval '
        'sha + eval-step accounting + resume dataset-hash guard, all armed')

  # ---------------- G7: run-config contract -------------------------------
  run_cfg = build_offline_cfg(max_steps=1_000_000)
  assert run_cfg.bc_coef == 0.05 and run_cfg.twin_q is True
  assert run_cfg.random_goals == 0.0 and run_cfg.batch_size == 1024
  assert run_cfg.repr_dim == 16
  assert run_cfg.hidden_layer_sizes == (1024, 1024)
  assert run_cfg.entropy_coefficient == 0.0 and run_cfg.target_entropy == 0.0
  assert run_cfg.max_number_of_steps == 1_000_000
  assert run_cfg.offline_dataset == NPZ and run_cfg.use_td is False
  GATE['G7_run_config'] = {
      'pass': True,
      'objective': ('L_actor = mean[0.05*(-log pi(a_data|s,g)) + 0.95*'
                    '(0*log pi(a|s,g) - min(Q1,Q2)(s,a,g))]; '
                    'critic: binary NCE on twin logits; goals: geometric '
                    'future-state relabels gamma=0.99; '
                    'budget: 1e6 gradient updates (offline step clock)')}
  print('G7 PASS: run config matches the upstream offline recipe')

  GATE['ALL'] = all(v.get('pass') for k, v in GATE.items() if k != 'ALL')
  json.dump(GATE, open(f'{OUT}/pretraining_gates.json', 'w'), indent=2)
  print('\nALL GATES PASS' if GATE['ALL'] else '\nGATE FAILURE')
  print('saved', f'{OUT}/pretraining_gates.json')


if __name__ == '__main__':
  main()
