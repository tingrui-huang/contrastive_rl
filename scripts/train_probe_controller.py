"""Stage 1A: train the lane/speed residual on top of the FROZEN 0.89 actor.

Residual TD3 (deterministic residual + twin critics + target smoothing) on a
dense corridor-tracking reward. The base actor is never updated; the residual
is zero-initialized and amplitude-capped (crl/probe.py ALPHA), so training
starts exactly at the validated locomotion.

Phases (both with collapse DISABLED):
  1: empty corridor (offline_ant_umaze). Commands sampled per episode from
     y_ref in {-1.1, 0, +1.1} x v_ref in {slow, fast}. Goal: lateral error
     from ~+-0.5-0.8 down to ~+-0.2, and actual speed tracking v_ref.
  2: litter geometry present (offline_ant_umaze_litter, collapse off),
     init from phase 1. Learns lane keeping among litter + composure on
     contact. Commands: clean/blocked/middle lanes emerge from y_ref anyway.

Shaping reward (TRAINING TOOL ONLY -- final gates use raw task metrics):
  r = cp*dx - cy*|y - y_ref| - cv*|vx - v_ref| - cpsi*|yaw|
      - cc*1[litter contact] - cu*||da||^2  (+ fall penalty, terminal)

Episodes: reset at the R cell, terminate at x >= X_DONE (success, bonus),
on fall (penalty), or at EP_LEN steps.

Usage:
  python scripts/train_probe_controller.py --phase 1 --steps 150000 \
      --ckpt offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl
  python scripts/train_probe_controller.py --phase 2 --steps 150000 \
      --init-from artifacts/probe_controller/phase1/residual_best.pkl ...
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

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl import probe                     # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

EP_LEN = 250
X_DONE = 6.0
GAMMA = 0.98
TAU = 0.005
LR = 3e-4
BATCH = 256
BUFFER = 300_000
WARMUP = 3_000
EXPL_NOISE = 0.15          # in residual units (pre-clip), scaled by ALPHA
POLICY_NOISE = 0.1
NOISE_CLIP = 0.3
POLICY_DELAY = 2
EVAL_EVERY = 10_000
COMMANDS = [(y, v) for y in (-probe.LANE_Y, 0.0, probe.LANE_Y)
            for v in (probe.V_SLOW, probe.V_FAST)]

CP, CY, CV, CPSI, CC, CU = 10.0, 1.0, 0.5, 0.3, 0.5, 0.1
FALL_PENALTY, DONE_BONUS = 5.0, 5.0


def torso_up_z(qpos):
  w, x, y, _ = qpos[3:7]
  return 1.0 - 2.0 * (x * x + y * y)


def torso_yaw(qpos):
  w, x, y, z = qpos[3:7]
  return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Trainer:

  def __init__(self, seed, base_act):
    self.rng = np.random.default_rng(seed)
    self.base_act = base_act
    self.actor_t, self.critic_t = probe.make_residual_networks()
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    dummy_o = jnp.zeros((1, probe.RES_OBS_DIM))
    dummy_a = jnp.zeros((1, 8))
    self.pi = self.actor_t.init(k1, dummy_o)
    self.q = self.critic_t.init(k2, dummy_o, dummy_a)
    self.pi_targ, self.q_targ = self.pi, self.q
    self.opt_pi = optax.adam(LR)
    self.opt_q = optax.adam(LR)
    self.pi_state = self.opt_pi.init(self.pi)
    self.q_state = self.opt_q.init(self.q)
    self.updates = 0

    self.obs = np.zeros((BUFFER, probe.RES_OBS_DIM), np.float32)
    self.act = np.zeros((BUFFER, 8), np.float32)
    self.rew = np.zeros(BUFFER, np.float32)
    self.nobs = np.zeros((BUFFER, probe.RES_OBS_DIM), np.float32)
    self.mask = np.zeros(BUFFER, np.float32)   # 0 where terminal
    self.ptr, self.full = 0, False

    actor_t, critic_t = self.actor_t, self.critic_t

    @jax.jit
    def q_step(q, q_state, pi_targ, q_targ, batch, key):
      o, a, r, no, m = batch
      noise = jnp.clip(POLICY_NOISE * jax.random.normal(key, (BATCH, 8)),
                       -NOISE_CLIP, NOISE_CLIP)
      na = jnp.clip(actor_t.apply(pi_targ, no) + noise,
                    -probe.ALPHA, probe.ALPHA)
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
    def res_act(pi, o):
      return actor_t.apply(pi, o)

    self._q_step, self._pi_step = q_step, pi_step
    self._polyak, self._res_act = polyak, res_act
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

  def residual(self, ro, explore):
    a = np.asarray(self._res_act(self.pi, jnp.asarray(ro[None]))[0])
    if explore:
      a = a + self.rng.normal(0, EXPL_NOISE * probe.ALPHA, 8)
    return np.clip(a, -probe.ALPHA, probe.ALPHA)


def make_train_env(phase, cfg, seed):
  name = 'offline_ant_umaze' if phase == 1 else 'offline_ant_umaze_litter'
  env = envs_mod.make_env(name, cfg, seed=seed)
  if hasattr(env, 'collapse_force'):
    env.collapse_force = None            # Stage 1A trains WITHOUT collapse
  return env


def run_episode(env, base_act, trainer, y_ref, v_ref, explore, store=True):
  o = env.reset()
  prev_x = o[0]
  stats = {'y_err': [], 'vx': [], 'contact': 0, 'fell': False, 'reached': False}
  prev_a_res = np.zeros(8)
  ro = None
  for t in range(EP_LEN):
    xy = o[:2]
    g = np.array([min(xy[0] + probe.LOOKAHEAD, probe.CARROT_CAP_X), y_ref],
                 np.float32)
    o_cmd = o.copy()
    o_cmd[29:] = 0.0
    o_cmd[29:31] = g
    a_base = np.asarray(base_act(jnp.asarray(o_cmd[None]))[0])
    ro_new = probe.residual_obs(o[:29], g, y_ref, v_ref)
    a_res = (trainer.residual(ro_new, explore) if trainer is not None
             else np.zeros(8))
    a = np.clip(a_base + a_res, -1.0, 1.0)
    o2, _, _, info = env.step(a)
    q = env._env.data.qpos
    qv = env._env.data.qvel
    y, vx, yaw = float(o2[1]), float(qv[0]), abs(torso_yaw(np.asarray(q)))
    contact = int(info.get('pile_contacts', 0) > 0
                  or info.get('rubble_contacts', 0) > 0)
    fell = torso_up_z(np.asarray(q)) < 0.3 or float(q[2]) < 0.2
    reached = float(o2[0]) >= X_DONE
    r = (CP * (float(o2[0]) - prev_x) - CY * abs(y - y_ref)
         - CV * abs(vx - v_ref) - CPSI * min(yaw, np.pi - yaw)
         - CC * contact - CU * float(np.sum((a_res - prev_a_res) ** 2)))
    if fell:
      r -= FALL_PENALTY
    if reached:
      r += DONE_BONUS
    terminal = fell or reached
    if store and trainer is not None:
      ro2 = probe.residual_obs(
          o2[:29],
          np.array([min(o2[0] + probe.LOOKAHEAD, probe.CARROT_CAP_X), y_ref],
                   np.float32), y_ref, v_ref)
      trainer.store(ro_new, a_res, r, ro2, 0.0 if terminal else 1.0)
    stats['y_err'].append(abs(y - y_ref))
    stats['vx'].append(vx)
    stats['contact'] += contact
    prev_x, prev_a_res, o, ro = float(o2[0]), a_res, o2, ro_new
    if terminal:
      stats['fell'] = fell
      stats['reached'] = reached
      break
  stats['steps'] = t + 1
  return stats


def evaluate(env, base_act, trainer):
  rows = []
  for y_ref, v_ref in COMMANDS:
    for _ in range(3):
      s = run_episode(env, base_act, trainer, y_ref, v_ref,
                      explore=False, store=False)
      zone = [e for i, e in enumerate(s['y_err'])]   # whole run
      rows.append({'y_ref': y_ref, 'v_ref': v_ref,
                   'y_err_p90': float(np.percentile(zone, 90)),
                   'vx_mean': float(np.mean(s['vx'])),
                   'reached': bool(s['reached']), 'fell': bool(s['fell']),
                   'steps': s['steps'], 'contact': s['contact']})
  agg = {
      'y_err_p90_max': max(r['y_err_p90'] for r in rows),
      'reach_rate': float(np.mean([r['reached'] for r in rows])),
      'fall_rate': float(np.mean([r['fell'] for r in rows])),
      'slow_vx': float(np.mean([r['vx_mean'] for r in rows
                                if r['v_ref'] == probe.V_SLOW])),
      'fast_vx': float(np.mean([r['vx_mean'] for r in rows
                                if r['v_ref'] == probe.V_FAST])),
  }
  return agg, rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default='offline_umaze_bc005_twinmin_s0_50k/'
                                    'checkpoints/best.pkl')
  ap.add_argument('--phase', type=int, choices=(1, 2), required=True)
  ap.add_argument('--steps', type=int, default=150_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--init-from', default=None)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()
  out = args.out or f'artifacts/probe_controller/phase{args.phase}'
  os.makedirs(out, exist_ok=True)

  cfg = build_offline_cfg()
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  _, st = ckpt_mod.load_checkpoint(args.ckpt)
  params = st.policy_params

  @jax.jit
  def base_act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  trainer = Trainer(args.seed, base_act)
  if args.init_from:
    trainer.pi, meta = probe.load_residual(args.init_from)
    trainer.pi_targ = trainer.pi
    print(f'initialized residual from {args.init_from} ({meta})')

  env = make_train_env(args.phase, cfg, args.seed)
  eval_env = make_train_env(args.phase, cfg, args.seed + 1000)

  log_path = os.path.join(out, 'train_log.jsonl')
  best = {'y_err_p90_max': np.inf}
  steps = 0
  t0 = time.time()
  with open(log_path, 'a') as log:
    while steps < args.steps:
      y_ref, v_ref = COMMANDS[trainer.rng.integers(len(COMMANDS))]
      s = run_episode(env, base_act, trainer, y_ref, v_ref, explore=True)
      n = s['steps']
      if steps > WARMUP:
        for _ in range(n):
          trainer.update()
      new_steps = steps + n
      if new_steps // EVAL_EVERY > steps // EVAL_EVERY:
        agg, rows = evaluate(eval_env, base_act, trainer)
        agg.update(step=new_steps, sps=new_steps / (time.time() - t0))
        log.write(json.dumps({'eval': agg, 'rows': rows}) + '\n')
        log.flush()
        print(f'[{new_steps:7d}] y_err_p90_max {agg["y_err_p90_max"]:.2f}  '
              f'reach {agg["reach_rate"]:.2f}  fall {agg["fall_rate"]:.2f}  '
              f'slow_vx {agg["slow_vx"]:.2f}  fast_vx {agg["fast_vx"]:.2f}  '
              f'({agg["sps"]:.0f} sps)', flush=True)
        probe.save_residual(os.path.join(out, 'residual_latest.pkl'),
                            trainer.pi, {'step': new_steps, **agg})
        if (agg['fall_rate'] <= 0.1
            and agg['y_err_p90_max'] < best['y_err_p90_max']):
          best = agg
          probe.save_residual(os.path.join(out, 'residual_best.pkl'),
                              trainer.pi, {'step': new_steps, **agg})
      steps = new_steps

  print('done.', json.dumps(best))


if __name__ == '__main__':
  main()
