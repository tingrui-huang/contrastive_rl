"""Stage 1A (revised): from-scratch direct-torque lane/speed TD3 walker.

No 0.89 base in the loop: the walker owns its gait. Corridor specialist:
learns to walk +x while tracking a commanded lane y_ref and forward speed
v_ref. Commands sampled CONTINUOUSLY per episode (y_ref ~ U(-1.2, 1.2),
v_ref ~ U(0.4, 1.6)) for robustness; qualification evaluates the six
canonical commands (y in {-1.1, 0, +1.1} x v in {0.6, 1.4}).

Phase 1: empty corridor (offline_ant_umaze). Command qualification targets:
  in-zone (x in [2.5, 5.5]) y_err_p90 <= 0.25 on every canonical command,
  slow/fast vx within +-0.2/0.25 of command, fall <= 5%, reach >= 95%.
Phase 2: litter geometry, collapse OFF (--phase 2 --init-from ...).
Only after freezing: collapse recalibration + the 4-arm gate.

Start states randomized along the corridor (x ~ U(-0.5, 3), y ~ U(-1.3,
1.3)) so lane-keeping is trained everywhere, not just from the R cell.

Usage:
  python scripts/train_walker.py --phase 1 --steps 2000000
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                             # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import probe                     # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

EP_LEN = 250
X_DONE = 6.0
ZONE = (2.5, 5.5)
GAMMA = 0.98
TAU = 0.005
LR = 3e-4
BATCH = 256
BUFFER = 1_000_000
WARMUP = 10_000            # uniform random actions
EXPL_NOISE = 0.2
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2
EVAL_EVERY = 25_000
EVAL_COMMANDS = [(y, v) for y in (-probe.LANE_Y, 0.0, probe.LANE_Y)
                 for v in (probe.V_SLOW, probe.V_FAST)]

CP, CY, CV, CPSI, CC, CU = 10.0, 2.5, 1.0, 0.3, 0.5, 0.05
FALL_PENALTY, DONE_BONUS = 5.0, 5.0
DT = 0.1

QUAL = {'y_err_p90': 0.25, 'v_tol_slow': 0.2, 'v_tol_fast': 0.25,
        'fall_rate': 0.05, 'reach_rate': 0.95}


def torso_up_z(qpos):
  w, x, y, _ = qpos[3:7]
  return 1.0 - 2.0 * (x * x + y * y)


def torso_yaw(qpos):
  w, x, y, z = qpos[3:7]
  return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Trainer:

  def __init__(self, seed):
    self.rng = np.random.default_rng(seed)
    self.actor_t, self.critic_t = probe.make_walker_networks()
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    dummy_o = jnp.zeros((1, probe.WALKER_OBS_DIM))
    dummy_a = jnp.zeros((1, 8))
    self.pi = self.actor_t.init(k1, dummy_o)
    self.q = self.critic_t.init(k2, dummy_o, dummy_a)
    self.pi_targ, self.q_targ = self.pi, self.q
    self.opt_pi = optax.adam(LR)
    self.opt_q = optax.adam(LR)
    self.pi_state = self.opt_pi.init(self.pi)
    self.q_state = self.opt_q.init(self.q)
    self.updates = 0

    self.obs = np.zeros((BUFFER, probe.WALKER_OBS_DIM), np.float32)
    self.act = np.zeros((BUFFER, 8), np.float32)
    self.rew = np.zeros(BUFFER, np.float32)
    self.nobs = np.zeros((BUFFER, probe.WALKER_OBS_DIM), np.float32)
    self.mask = np.zeros(BUFFER, np.float32)
    self.ptr, self.full = 0, False

    actor_t, critic_t = self.actor_t, self.critic_t

    @jax.jit
    def q_step(q, q_state, pi_targ, q_targ, batch, key):
      o, a, r, no, m = batch
      noise = jnp.clip(POLICY_NOISE * jax.random.normal(key, (BATCH, 8)),
                       -NOISE_CLIP, NOISE_CLIP)
      na = jnp.clip(actor_t.apply(pi_targ, no) + noise, -1.0, 1.0)
      tq1, tq2 = critic_t.apply(q_targ, no, na)
      target = r + GAMMA * m * jnp.minimum(tq1, tq2)

      def loss_fn(qp):
        q1, q2 = critic_t.apply(qp, o, a)
        return jnp.mean((q1 - target) ** 2 + (q2 - target) ** 2)

      loss, grads = jax.value_and_grad(loss_fn)(q)
      upd, q_state = self.opt_q.update(grads, q_state)
      return optax.apply_updates(q, upd), q_state, loss

    @jax.jit
    def pi_step(pi, pi_state, q, batch):
      o = batch[0]

      def loss_fn(pp):
        a = actor_t.apply(pp, o)
        q1, _ = critic_t.apply(q, o, a)
        return -jnp.mean(q1)

      loss, grads = jax.value_and_grad(loss_fn)(pi)
      upd, pi_state = self.opt_pi.update(grads, pi_state)
      return optax.apply_updates(pi, upd), pi_state, loss

    @jax.jit
    def polyak(a, b):
      return jax.tree.map(lambda x, y: x + TAU * (y - x), a, b)

    @jax.jit
    def act_fn(pi, o):
      return actor_t.apply(pi, o)

    self._q_step, self._pi_step = q_step, pi_step
    self._polyak, self._act = polyak, act_fn
    self.key = jax.random.PRNGKey(seed + 1)

  def store(self, o, a, r, no, mask):
    i = self.ptr
    self.obs[i], self.act[i], self.rew[i] = o, a, r
    self.nobs[i], self.mask[i] = no, mask
    self.ptr = (i + 1) % BUFFER
    self.full = self.full or self.ptr == 0

  def sample(self):
    hi = BUFFER if self.full else self.ptr
    idx = self.rng.integers(hi, size=BATCH)
    return (jnp.asarray(self.obs[idx]), jnp.asarray(self.act[idx]),
            jnp.asarray(self.rew[idx]), jnp.asarray(self.nobs[idx]),
            jnp.asarray(self.mask[idx]))

  def update(self):
    self.key, k = jax.random.split(self.key)
    batch = self.sample()
    self.q, self.q_state, qloss = self._q_step(
        self.q, self.q_state, self.pi_targ, self.q_targ, batch, k)
    self.updates += 1
    if self.updates % POLICY_DELAY == 0:
      self.pi, self.pi_state, _ = self._pi_step(
          self.pi, self.pi_state, self.q, batch)
      self.pi_targ = self._polyak(self.pi_targ, self.pi)
      self.q_targ = self._polyak(self.q_targ, self.q)
    return float(qloss)

  def action(self, wo, mode):
    if mode == 'random':
      return self.rng.uniform(-1.0, 1.0, 8)
    a = np.asarray(self._act(self.pi, jnp.asarray(wo[None]))[0])
    if mode == 'explore':
      a = np.clip(a + self.rng.normal(0, EXPL_NOISE, 8), -1.0, 1.0)
    return a


def make_train_env(phase, cfg, seed):
  name = 'offline_ant_umaze' if phase == 1 else 'offline_ant_umaze_litter'
  env = envs_mod.make_env(name, cfg, seed=seed)
  if hasattr(env, 'collapse_force'):
    env.collapse_force = None            # collapse only after freezing
  return env


def randomize_start(env, rng):
  u = env._env
  u.data.qpos[0] = rng.uniform(-0.5, 3.0)
  u.data.qpos[1] = rng.uniform(-1.3, 1.3)
  mujoco.mj_forward(u.model, u.data)


def run_episode(env, trainer, y_ref, v_ref, mode, rng=None, store=True):
  o = env.reset()
  if rng is not None:
    randomize_start(env, rng)
    o = env._flatten(env._env._obs_dict())
    env._last_obs = env._env._obs_dict()
  prev_x = o[0]
  stats = {'y_err': [], 'vx': [], 'contact': 0, 'fell': False,
           'reached': False}
  prev_a = np.zeros(8)
  for t in range(EP_LEN):
    wo = probe.walker_obs(o[:29], y_ref, v_ref)
    a = (trainer.action(wo, mode) if trainer is not None else np.zeros(8))
    o2, _, _, info = env.step(a)
    q = env._env.data.qpos
    qv = env._env.data.qvel
    y, vx, yaw = float(o2[1]), float(qv[0]), abs(torso_yaw(np.asarray(q)))
    contact = int(info.get('pile_contacts', 0) > 0
                  or info.get('rubble_contacts', 0) > 0)
    fell = torso_up_z(np.asarray(q)) < 0.3 or float(q[2]) < 0.2
    reached = float(o2[0]) >= X_DONE
    r = (CP * min(float(o2[0]) - prev_x, v_ref * DT) - CY * abs(y - y_ref)
         - CV * abs(vx - v_ref) - CPSI * min(yaw, np.pi - yaw)
         - CC * contact - CU * float(np.sum((a - prev_a) ** 2)))
    if fell:
      r -= FALL_PENALTY
    if reached:
      r += DONE_BONUS
    terminal = fell or reached
    if store and trainer is not None:
      wo2 = probe.walker_obs(o2[:29], y_ref, v_ref)
      trainer.store(wo, a, r, wo2, 0.0 if terminal else 1.0)
    if ZONE[0] <= float(o2[0]) <= ZONE[1]:
      stats['y_err'].append(abs(y - y_ref))
      stats['vx'].append(vx)
    stats['contact'] += contact
    prev_x, prev_a, o = float(o2[0]), a, o2
    if terminal:
      stats['fell'] = fell
      stats['reached'] = reached
      break
  stats['steps'] = t + 1
  return stats


def evaluate(env, trainer, eps=3):
  rows = []
  for y_ref, v_ref in EVAL_COMMANDS:
    for _ in range(eps):
      s = run_episode(env, trainer, y_ref, v_ref, mode='eval', store=False)
      zone = s['y_err'] or [2.0]
      rows.append({'y_ref': y_ref, 'v_ref': v_ref,
                   'y_err_p90': float(np.percentile(zone, 90)),
                   'vx_mean': float(np.mean(s['vx'] or [0.0])),
                   'reached': bool(s['reached']), 'fell': bool(s['fell']),
                   'steps': s['steps'], 'contact': s['contact']})
  slow = [r for r in rows if r['v_ref'] == probe.V_SLOW]
  fast = [r for r in rows if r['v_ref'] == probe.V_FAST]
  agg = {'y_err_p90_max': max(r['y_err_p90'] for r in rows),
         'reach_rate': float(np.mean([r['reached'] for r in rows])),
         'fall_rate': float(np.mean([r['fell'] for r in rows])),
         'slow_vx': float(np.mean([r['vx_mean'] for r in slow])),
         'fast_vx': float(np.mean([r['vx_mean'] for r in fast]))}
  agg['qualified'] = bool(
      agg['y_err_p90_max'] <= QUAL['y_err_p90']
      and abs(agg['slow_vx'] - probe.V_SLOW) <= QUAL['v_tol_slow']
      and abs(agg['fast_vx'] - probe.V_FAST) <= QUAL['v_tol_fast']
      and agg['fall_rate'] <= QUAL['fall_rate']
      and agg['reach_rate'] >= QUAL['reach_rate'])
  return agg, rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--phase', type=int, choices=(1, 2), required=True)
  ap.add_argument('--steps', type=int, default=2_000_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--init-from', default=None)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()
  out = args.out or f'artifacts/walker/phase{args.phase}'
  os.makedirs(out, exist_ok=True)

  cfg = build_offline_cfg()
  trainer = Trainer(args.seed)
  if args.init_from:
    trainer.pi, meta = probe.load_residual(args.init_from)
    trainer.pi_targ = trainer.pi
    print(f'initialized walker from {args.init_from} ({meta})', flush=True)

  env = make_train_env(args.phase, cfg, args.seed)
  eval_env = make_train_env(args.phase, cfg, args.seed + 1000)
  start_rng = np.random.default_rng(args.seed + 7)

  log_path = os.path.join(out, 'train_log.jsonl')
  best_err = np.inf
  steps = 0
  t0 = time.time()
  with open(log_path, 'a') as log:
    while steps < args.steps:
      y_ref = float(start_rng.uniform(-1.2, 1.2))
      v_ref = float(start_rng.uniform(0.4, 1.6))
      mode = 'random' if steps < WARMUP else 'explore'
      s = run_episode(env, trainer, y_ref, v_ref, mode, rng=start_rng)
      n = s['steps']
      if steps > WARMUP:
        for _ in range(n):
          trainer.update()
      new_steps = steps + n
      if new_steps // EVAL_EVERY > steps // EVAL_EVERY:
        agg, rows = evaluate(eval_env, trainer)
        agg.update(step=new_steps, sps=new_steps / (time.time() - t0))
        log.write(json.dumps({'eval': agg, 'rows': rows}) + '\n')
        log.flush()
        print(f'[{new_steps:8d}] y_err {agg["y_err_p90_max"]:.2f}  '
              f'reach {agg["reach_rate"]:.2f}  fall {agg["fall_rate"]:.2f}  '
              f'slow_vx {agg["slow_vx"]:.2f}  fast_vx {agg["fast_vx"]:.2f}  '
              f'qual {agg["qualified"]}  ({agg["sps"]:.0f} sps)', flush=True)
        probe.save_residual(os.path.join(out, 'walker_latest.pkl'),
                            trainer.pi, {'step': new_steps, **agg})
        if (agg['fall_rate'] <= 0.1 and agg['reach_rate'] >= 0.8
            and agg['y_err_p90_max'] < best_err):
          best_err = agg['y_err_p90_max']
          probe.save_residual(os.path.join(out, 'walker_best.pkl'),
                              trainer.pi, {'step': new_steps, **agg})
        if agg['qualified']:
          probe.save_residual(os.path.join(out, 'walker_qualified.pkl'),
                              trainer.pi, {'step': new_steps, **agg})
          print('QUALIFIED -- checkpoint saved; continuing to budget end.',
                flush=True)
      steps = new_steps

  print('done. best y_err_p90_max:', best_err, flush=True)


if __name__ == '__main__':
  main()
