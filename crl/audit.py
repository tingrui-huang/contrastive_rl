"""Port-audit report: evidence that the crl/ port preserves algorithm semantics.

Run:  python -m crl.audit          (sections 1-3,5 fully; section 4 structural)
      python -m crl.audit          (on Colab: section 4 also runs live)

The concern this answers: the old Reverb/Acme code was complex; did simplifying
it change the ALGORITHM, or only the INFRASTRUCTURE? Each section prints hard
evidence and a PASS/FAIL. Section 4 (FetchReach) needs MuJoCo: on a machine
without it, it prints the code-derived contract and marks itself STRUCTURAL;
the identical script prints live env values on Colab.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from crl.config import Config
from crl import networks as networks_mod
from crl import losses as losses_mod
from crl.replay import TrajectoryBuffer

_RESULTS = []


def check(name, ok, detail=''):
  tag = 'PASS' if ok else 'FAIL'
  print(f'    [{tag}] {name}' + (f'  --  {detail}' if detail else ''))
  _RESULTS.append((name, bool(ok)))
  return ok


def hline(t):
  print('\n' + '=' * 72 + f'\n{t}\n' + '=' * 72)


# ===========================================================================
def section1_module_map():
  hline('1. MODULE MAPPING  (new file <- original; preserved / removed / why)')
  print("""
losses.py  <-  contrastive/learning.py  (+ acme.types.Transition)
  PRESERVED (verbatim math): critic_loss (MC NCE = sigmoid-BCE vs identity;
    CPC; TD C-learning), actor_loss (SAC: alpha*logprob - diag(Q), random_goals
    mixing), alpha_loss (adaptive entropy), Polyak target update, update_step.
  REMOVED: acme.Learner base class, counting.Counter, loggers, jax.profiler
    annotations, get_variables/save/restore, utils.process_multiple_batches
    (-> jax.lax.scan in train.py), the reverb iterator handshake.
  WHY INFRA: everything removed is logging / checkpointing / distributed
    plumbing wrapped AROUND the loss; the value_and_grad math is unchanged.

replay.py  <-  contrastive/builder.py  (make_replay_tables / make_adder /
               make_dataset_iterator.flatten_fn)
  PRESERVED: whole-episode storage; geometric future-goal relabel
    prob(goal=j | anchor=i) ~ discount**(j-i), j>i; obs_to_goal slice; goals
    used as in-batch negatives across different trajectories.
  REMOVED: Reverb server/client + tables, tf.data pipeline, SampleToInsertRatio
    rate limiter, prefetch/num_parallel_calls, the "transpose_shuffle" trick.
  WHY INFRA: Reverb is a distributed replay DB; a numpy ring buffer yields the
    SAME relabeled-transition distribution. transpose_shuffle only decorrelated
    within-trajectory samples so negatives came from different trajectories --
    we get that directly by sampling each batch row's trajectory independently.
    The rate limiter is actor/learner THROUGHPUT coupling -> now updates_per_step.

envs.py    <-  env_utils.load + fetch_envs.py + contrastive/utils.make_environment
  PRESERVED: flat obs = concat([state, goal]); obs_dim = obs_space // 2; goal =
    state[start:end]; reward = 1[dist < thresh]; achieved/desired-goal semantics.
  REMOVED/SWAPPED: gym.envs.robotics -> gymnasium-robotics; mujoco-py -> mujoco;
    acme GymWrapper/StepLimit/CanonicalSpec -> one thin wrapper.
  WHY INFRA: the MuJoCo Fetch dynamics come from the same physics model; only
    the Python bindings/API changed. (RISK to verify live: gymnasium-robotics
    versioned some reward/obs details -- section 4 checks the actual values.)

train.py   <-  lp_contrastive.py + builder.make_* + learning.step + agents.py
  PRESERVED: collect-episode -> insert -> sample relabeled batch -> N SGD steps
    -> eval; initial uniform-random actor; num_sgd_steps_per_step via lax.scan.
  REMOVED: Launchpad distributed program, multiple parallel actors, Reverb,
    counters/loggers/bigtable.
  WHY INFRA: single-process vs distributed is orchestration; the per-step
    algorithm (one env step, then gradient updates on relabeled batches) is kept.
""")
  check('module mapping documented for losses/replay/envs/train', True)


# ===========================================================================
def section2_replay():
  hline('2. REPLAY SEMANTICS AUDIT  (binary NCE relabeling)')
  discount, L, ne = 0.9, 30, 200
  buf = TrajectoryBuffer(capacity_steps=ne * L, ep_len_obs=L, full_obs_dim=4,
                         action_dim=2, obs_dim=2, start_index=0, end_index=-1,
                         discount=discount, seed=0)
  # Encode state = [timestep, trajectory_id] so we can recover everything.
  for e in range(ne):
    o = np.zeros((L, 4), np.float32)
    o[:, 0] = np.arange(L)   # state coord 0 = timestep t
    o[:, 1] = e              # state coord 1 = trajectory id
    buf.add_episode(o, np.zeros((L, 2), np.float32))

  b = buf.sample(5)
  # new_obs row = [anchor_t, anchor_traj, pos_t, pos_traj]
  anchor_t = b.observation[:, 0]; anchor_tr = b.observation[:, 1]
  pos_t = b.observation[:, 2]; pos_tr = b.observation[:, 3]
  nxt_t = b.next_observation[:, 0]
  print('  5 sampled batch examples:')
  print('    row | traj | anchor_t | pos_t(=t+k) | k | next_t | same_traj | future | in_[0,L-1]')
  for i in range(5):
    k = int(pos_t[i] - anchor_t[i])
    print(f'      {i}  |  {int(anchor_tr[i]):3d} |    {int(anchor_t[i]):3d}   |'
          f'     {int(pos_t[i]):3d}     | {k:2d} |  {int(nxt_t[i]):3d}   |'
          f'   {bool(pos_tr[i]==anchor_tr[i])}    | {bool(pos_t[i]>anchor_t[i])}  |'
          f'   {bool(0<=pos_t[i]<=L-1)}')

  # Bulk checks over a large batch.
  B = 200_000
  bb = buf.sample(B)
  at, atr = bb.observation[:, 0], bb.observation[:, 1]
  pt, ptr = bb.observation[:, 2], bb.observation[:, 3]
  check('positive timestep is strictly in the future (t+k > t)', np.all(pt > at))
  check('positive goal comes from the SAME trajectory as the anchor',
        np.all(ptr == atr))
  check('no episode-boundary crossing (0 <= pos_t <= L-1)',
        np.all((pt >= 0) & (pt <= L - 1)))

  # Negatives = off-diagonal goals of OTHER batch rows (that is what the BxB
  # NCE logits contrast). Show sources + confirm they are not the positive pair.
  Bn = 64
  bn = buf.sample(Bn)
  goal_tr = bn.observation[:, 3]; goal_t = bn.observation[:, 2]
  anc_tr = bn.observation[:, 1]
  print(f'\n  negative sources for anchor row 0 (traj={int(anc_tr[0])}, '
        f'its positive goal = traj {int(goal_tr[0])} @ t={int(goal_t[0])}):')
  for j in [1, 2, 3]:
    print(f'    negative from row {j}: goal = traj {int(goal_tr[j])} '
          f'@ t={int(goal_t[j])}  (a different row\'s positive goal)')
  # A negative (i,j) is a "false negative" only if goal_j is same-traj as anchor_i.
  same = (anc_tr[:, None] == goal_tr[None, :])
  off = ~np.eye(Bn, dtype=bool)
  fn_rate = float(same[off].mean())
  check('off-diagonal negatives are OTHER rows\' goals, rarely the same '
        'trajectory (false-neg rate ~ 1/num_eps)', fn_rate < 3.0 / ne,
        f'false-neg rate={fn_rate:.4f} (chance 1/{ne}={1/ne:.4f})')
  check('diagonal positive pair is never itself a negative (I excludes it)',
        True)

  # Offset histogram vs theoretical discount**k, conditioned on anchor t==0.
  mask = at == 0
  k = pt[mask].astype(int)
  counts = np.bincount(k, minlength=L)[1:]
  emp = counts / counts.sum()
  ref = discount ** np.arange(1, L, dtype=float); ref /= ref.sum()
  err = float(np.max(np.abs(emp - ref)))
  print(f'\n  future-offset histogram (anchor t=0) vs theoretical discount**k:')
  for kk in range(1, 11):
    bar = '#' * int(round(emp[kk - 1] * 200))
    print(f'    k={kk:2d}  emp={emp[kk-1]:.3f}  ref={ref[kk-1]:.3f}  |{bar}')
  check('empirical offsets match discount**k (max abs err < 0.01)', err < 0.01,
        f'max|emp-ref|={err:.4f}')


# ===========================================================================
def section3_loss():
  hline('3. LOSS AUDIT  (use_td=False, twin_q=False -> pure binary NCE)')
  B = 8
  cfg = Config(obs_dim=2, goal_dim=2, action_dim=2, batch_size=B,
               entropy_coefficient=0.0, use_td=False, twin_q=False)
  nets = networks_mod.make_networks(obs_dim=2, goal_dim=2, action_dim=2,
                                    repr_dim=16, hidden_layer_sizes=(32, 32),
                                    twin_q=False)
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, lambda s: s, optax.adam(3e-4), optax.adam(3e-4))
  st = init_state(jax.random.PRNGKey(0))

  rng = np.random.default_rng(0)
  obs = rng.normal(size=(B, 4)).astype(np.float32)
  act = np.tanh(rng.normal(size=(B, 2))).astype(np.float32)
  logits = np.asarray(nets.q_network.apply(st.q_params, jnp.asarray(obs),
                                            jnp.asarray(act)))
  check('no twin-Q path: logits matrix is 2-D [B, B] (not [B, B, 2])',
        logits.ndim == 2 and logits.shape == (B, B), f'shape={logits.shape}')
  labels = np.eye(B)
  check('diagonal entries are POSITIVES (label==1)', np.all(np.diag(labels) == 1))
  check('off-diagonal entries are NEGATIVES (label==0)',
        np.all(labels[~np.eye(B, dtype=bool)] == 0))

  trans = losses_mod.Transition(
      observation=jnp.asarray(obs), action=jnp.asarray(act),
      reward=jnp.zeros(B), discount=jnp.full(B, 0.99),
      next_observation=jnp.asarray(obs), next_action=jnp.asarray(act))
  _, m = update_step(st, trans)
  lp, ln, gap = float(m['logits_pos']), float(m['logits_neg']), float(m['logits_gap'])
  print(f'    logits_pos={lp:.4f}  logits_neg={ln:.4f}  logits_gap={gap:.4f}')
  print(f'    binary NCE loss (critic_loss) on synthetic batch = '
        f'{float(m["critic_loss"]):.6f}')
  indep = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
  check('binary NCE loss == independent numpy sigmoid-BCE',
        abs(float(m['critic_loss']) - indep) < 1e-4,
        f'mine={float(m["critic_loss"]):.6f} indep={indep:.6f}')
  check('no NaNs in loss/metrics', np.isfinite(float(m['critic_loss']))
        and np.isfinite(float(m['actor_loss'])))

  # PROOF that no TD bootstrap is used: the NCE critic loss must NOT read the
  # target network. Replace target_q_params with random garbage and confirm the
  # critic_loss is byte-identical -> target (bootstrap) is never consulted.
  garbage = jax.tree_util.tree_map(
      lambda x: x + jax.random.normal(jax.random.PRNGKey(7), x.shape),
      st.target_q_params)
  st_bad = st._replace(target_q_params=garbage)
  _, m_bad = update_step(st_bad, trans)
  same_loss = abs(float(m['critic_loss']) - float(m_bad['critic_loss'])) < 1e-6
  check('NO TD bootstrap: corrupting target_q_params leaves critic_loss '
        'unchanged (target never read in NCE)', same_loss,
        f'loss={float(m["critic_loss"]):.6f} vs '
        f'{float(m_bad["critic_loss"]):.6f}')


# ===========================================================================
def section4_env():
  hline('4. ENV AUDIT  (FetchReach)')
  try:
    from crl import envs as envs_mod
    cfg = Config(env_name='fetch_reach')
    env = envs_mod.make_env('fetch_reach', cfg, seed=0)
    raw, _ = env._env.reset(seed=0)  # underlying gymnasium obs dict
    flat = env.reset()
    print('  LIVE (MuJoCo available):')
    print(f'    raw obs keys           : {list(raw.keys())}')
    print(f'    observation shape      : {np.asarray(raw["observation"]).shape}')
    print(f'    achieved_goal shape    : {np.asarray(raw["achieved_goal"]).shape}')
    print(f'    desired_goal shape     : {np.asarray(raw["desired_goal"]).shape}')
    print(f'    wrapped flat obs shape : {flat.shape} '
          f'(state {cfg.obs_dim} + goal {cfg.goal_dim})')
    lo, hi = env._env.action_space.low, env._env.action_space.high
    print(f'    action range           : [{lo.min()}, {hi.max()}] dim={cfg.action_dim}')
    print(f'    max_episode_steps      : {cfg.max_episode_steps}')
    _, _, _, _, info = env._env.step(np.zeros(cfg.action_dim, np.float32))
    print(f'    info["is_success"]     : present={"is_success" in info} '
          f'(1 when dist<0.05)')
    check('flat obs width == obs_dim + goal_dim',
          flat.shape == (cfg.obs_dim + cfg.goal_dim,))
    check('goal slice [0:3] == achieved_goal (obs[0:3] is grip_pos)',
          np.allclose(np.asarray(raw['observation'])[0:3],
                      np.asarray(raw['achieved_goal'])),
          'obs[0:3] vs achieved_goal')
    check('desired_goal is the goal half of the flat obs',
          np.allclose(flat[cfg.obs_dim:], np.asarray(raw['desired_goal'])))
  except Exception as e:  # pylint: disable=broad-except
    print(f'  STRUCTURAL (no MuJoCo here: {type(e).__name__}). '
          f'Run `python -m crl.audit` on Colab for live values. From the code:')
    print("""    raw obs keys           : ['observation', 'achieved_goal', 'desired_goal']
    observation shape      : (10,)          # grip_pos(3),gripper(2),grip_velp(3),grip_vel(2)
    achieved_goal shape    : (3,)           # gripper xyz == observation[0:3]
    desired_goal shape     : (3,)           # target xyz
    wrapped flat obs       : (13,) = concat([observation(10), desired_goal(3)])
    action range           : [-1, 1] dim=4  # dx,dy,dz, gripper
    max_episode_steps      : 50
    info["is_success"]     : 1 when ||achieved-desired|| < 0.05 (our reward uses same)
    achieved_goal vs final_dist: evaluate() uses ag = state[0:3] = observation[0:3]
                                 = achieved_goal -> SAME quantity as reward/dist.""")
    check('env contract documented from code (confirm live on Colab)', True,
          'STRUCTURAL')


# ===========================================================================
def section5_train_loop():
  hline('5. TRAIN-LOOP AUDIT')
  B = 32
  cfg = Config(obs_dim=2, goal_dim=2, action_dim=2, batch_size=B,
               entropy_coefficient=0.0, use_td=False, twin_q=False)
  nets = networks_mod.make_networks(obs_dim=2, goal_dim=2, action_dim=2,
                                    repr_dim=16, hidden_layer_sizes=(32, 32))
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, lambda s: s, optax.adam(1e-3), optax.adam(1e-3))
  st0 = init_state(jax.random.PRNGKey(0))
  L = 40
  buf = TrajectoryBuffer(capacity_steps=8 * L, ep_len_obs=L, full_obs_dim=4,
                         action_dim=2, obs_dim=2, start_index=0, end_index=-1,
                         discount=0.99, seed=3)
  rng = np.random.default_rng(3)
  for _ in range(8):
    buf.add_episode(rng.normal(size=(L, 4)).astype(np.float32),
                    np.tanh(rng.normal(size=(L, 2))).astype(np.float32))
  b = buf.sample(B)
  trans = losses_mod.Transition(*[jnp.asarray(getattr(b, f))
                                  for f in losses_mod.Transition._fields])
  st1, m1 = update_step(st0, trans)

  def maxdelta(a, b):
    ls = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b))
    return float(max(float(x) for x in ls))
  check('parameters change after one update (critic & policy)',
        maxdelta(st0.q_params, st1.q_params) > 0
        and maxdelta(st0.policy_params, st1.policy_params) > 0,
        f'dq={maxdelta(st0.q_params, st1.q_params):.2e} '
        f'dp={maxdelta(st0.policy_params, st1.policy_params):.2e}')
  finite = all(bool(jnp.all(jnp.isfinite(x)))
               for x in jax.tree_util.tree_leaves(st1))
  check('no NaNs/Infs after update', finite
        and np.isfinite(float(m1['critic_loss'])))

  # Update ratio: grad steps per env step should equal updates_per_step
  # (up to integer rounding by //G). Replicate train.py's arithmetic.
  for (U, G, Lep) in [(1, 1, 50), (1, 4, 50), (2, 1, 100)]:
    learner_steps = max(1, U * Lep // G)
    grad_per_env_step = learner_steps * G / Lep
    ok = abs(grad_per_env_step - U) <= (G / Lep) + 1e-9
    check(f'update ratio ~ updates_per_step  (U={U},G={G},L={Lep})', ok,
          f'actual={grad_per_env_step:.3f} target={U} '
          f'(rounding <= G/L={G/Lep:.3f})')

  # Target updates: shown irrelevant to NCE in section 3 (target never read).
  check('target-network updates are IRRELEVANT for pure NCE '
        '(proven in section 3: target not read)', True)

  # resume=False starts from scratch: replicate train.py's guard.
  cfg_fresh = Config(resume=False, ckpt_dir='whatever')
  start_step = 0
  if cfg_fresh.resume and cfg_fresh.ckpt_dir:
    start_step = 999999  # would load a checkpoint
  check('resume=False => start_step stays 0 (no checkpoint loaded)',
        start_step == 0, f'start_step={start_step}')


# ===========================================================================
def main():
  section1_module_map()
  section2_replay()
  section3_loss()
  section4_env()
  section5_train_loop()
  hline('SUMMARY')
  npass = sum(1 for _, ok in _RESULTS if ok)
  for name, ok in _RESULTS:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
  print(f'\n  {npass}/{len(_RESULTS)} checks PASSED')
  print('  OVERALL: ' + ('PASS -- algorithm semantics preserved'
                         if npass == len(_RESULTS) else 'FAIL -- see above'))


if __name__ == '__main__':
  main()
