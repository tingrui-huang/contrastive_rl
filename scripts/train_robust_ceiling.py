"""Diagnostic 1: the U-BLIND robust ceiling in offline_ant_umaze_litter.

An online-TD3 high-level controller that NEVER observes u_side or any
privileged variable. It sees only the 29-dim proprioceptive state and, each
step, commands the FROZEN walker (lane y_ref, speed v_ref); after the litter
corridor (geometric handoff x>=6 or y>=2) the FROZEN base policy drives the
true goal. Collapse is ON (real env). Nothing frozen is modified.

Unlike the fixed middle_slow controller, this learns adaptive speed,
contact-aware slowing (contact perturbs proprioception), recovery/lateral
corrections, all U-blind. "When to hand off" is the fixed post-litter
geometric rule (learned handoff is out of scope for this quick diagnostic).

Training MDP (corridor steps only): reward = 10*dx (route progress) - 0.02
- 2*fell; terminal +5 on reaching handoff ALIVE, -8 on collapse, 0 on
timeout. Eval runs the FULL episode (corridor policy + base handoff) and
reports true goal-reaching success.

Usage: python scripts/train_robust_ceiling.py --steps 600000 [--eval-only CKPT]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import probe as P                # noqa: E402
import litter_pilot_common as C           # noqa: E402

OBS = 29                     # proprioceptive state only (U-blind)
ACT = 2                      # [y_ref, v_ref] command to the frozen walker
HANDOFF_X, ZONE = 6.0, (2.5, 5.5)
Y_MAX, V_LO, V_HI = 1.2, 0.4, 1.6
EP_LEN = 700
GAMMA, TAU, LR, BATCH = 0.98, 0.005, 3e-4, 256
BUFFER, WARMUP = 500_000, 10_000
EXPL, PNOISE, NCLIP, PDELAY = 0.2, 0.2, 0.5, 2
CP, CTIME, CFALL, RDONE, RDEAD = 10.0, 0.02, 2.0, 5.0, 8.0
FROZEN_MIDDLE_SLOW = 79 / 143   # 0.552, full-dataset middle_slow success


def decode(a):
  y = float(np.clip(a[0], -1, 1)) * Y_MAX
  v = V_LO + (float(np.clip(a[1], -1, 1)) + 1) * 0.5 * (V_HI - V_LO)
  return y, v


def torso_up(qpos):
  x, y = qpos[4], qpos[5]
  return 1.0 - 2.0 * (x * x + y * y)


def make_nets(hidden=(256, 256)):
  def actor(o):
    h = o
    for w in hidden:
      h = jax.nn.relu(hk.Linear(w)(h))
    return jnp.tanh(hk.Linear(ACT)(h))

  def critic(o, a):
    x = jnp.concatenate([o, a], -1)
    qs = []
    for _ in range(2):
      h = x
      for w in hidden:
        h = jax.nn.relu(hk.Linear(w)(h))
      qs.append(hk.Linear(1)(h)[..., 0])
    return tuple(qs)
  return (hk.without_apply_rng(hk.transform(actor)),
          hk.without_apply_rng(hk.transform(critic)))


class TD3:
  def __init__(self, seed):
    self.rng = np.random.default_rng(seed)
    self.a_t, self.c_t = make_nets()
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    do, da = jnp.zeros((1, OBS)), jnp.zeros((1, ACT))
    self.pi, self.q = self.a_t.init(k1, do), self.c_t.init(k2, do, da)
    self.pit, self.qt = self.pi, self.q
    self.opi, self.oq = optax.adam(LR), optax.adam(LR)
    self.spi, self.sq = self.opi.init(self.pi), self.oq.init(self.q)
    self.up = 0
    self.O = np.zeros((BUFFER, OBS), np.float32)
    self.A = np.zeros((BUFFER, ACT), np.float32)
    self.R = np.zeros(BUFFER, np.float32)
    self.O2 = np.zeros((BUFFER, OBS), np.float32)
    self.M = np.zeros(BUFFER, np.float32)
    self.ptr, self.full = 0, False
    a_t, c_t = self.a_t, self.c_t

    @jax.jit
    def q_step(q, sq, pit, qt, b, key):
      o, a, r, o2, m = b
      noise = jnp.clip(PNOISE * jax.random.normal(key, (BATCH, ACT)),
                       -NCLIP, NCLIP)
      na = jnp.clip(a_t.apply(pit, o2) + noise, -1, 1)
      t1, t2 = c_t.apply(qt, o2, na)
      tgt = r + GAMMA * m * jnp.minimum(t1, t2)

      def loss(qp):
        q1, q2 = c_t.apply(qp, o, a)
        return jnp.mean((q1 - tgt) ** 2 + (q2 - tgt) ** 2)
      lo, g = jax.value_and_grad(loss)(q)
      u, sq = self.oq.update(g, sq)
      return optax.apply_updates(q, u), sq, lo

    @jax.jit
    def pi_step(pi, spi, q, b):
      o = b[0]

      def loss(pp):
        return -jnp.mean(c_t.apply(q, o, a_t.apply(pp, o))[0])
      lo, g = jax.value_and_grad(loss)(pi)
      u, spi = self.opi.update(g, spi)
      return optax.apply_updates(pi, u), spi, lo

    @jax.jit
    def poly(a, b):
      return jax.tree.map(lambda x, y: x + TAU * (y - x), a, b)

    @jax.jit
    def act(pi, o):
      return a_t.apply(pi, o)
    self._q, self._pi, self._poly, self._act = q_step, pi_step, poly, act
    self.key = jax.random.PRNGKey(seed + 1)

  def store(self, o, a, r, o2, m):
    i = self.ptr
    self.O[i], self.A[i], self.R[i], self.O2[i], self.M[i] = o, a, r, o2, m
    self.ptr = (i + 1) % BUFFER
    self.full = self.full or self.ptr == 0

  def update(self):
    hi = BUFFER if self.full else self.ptr
    idx = self.rng.integers(hi, size=BATCH)
    b = (jnp.asarray(self.O[idx]), jnp.asarray(self.A[idx]),
         jnp.asarray(self.R[idx]), jnp.asarray(self.O2[idx]),
         jnp.asarray(self.M[idx]))
    self.key, k = jax.random.split(self.key)
    self.q, self.sq, _ = self._q(self.q, self.sq, self.pit, self.qt, b, k)
    self.up += 1
    if self.up % PDELAY == 0:
      self.pi, self.spi, _ = self._pi(self.pi, self.spi, self.q, b)
      self.pit = self._poly(self.pit, self.pi)
      self.qt = self._poly(self.qt, self.q)

  def action(self, o, explore):
    a = np.asarray(self._act(self.pi, jnp.asarray(o[None]))[0])
    if explore:
      a = a + self.rng.normal(0, EXPL, ACT)
    return np.clip(a, -1, 1)


def episode(env, walker, base_act, td3, mode, u_side=None):
  """mode: 'train' | 'eval'. Returns rollout stats (+ stores if train)."""
  o = env.reset(u_side=u_side) if u_side is not None else env.reset()
  u = int(env.u_side)
  true_goal = o[29:31].copy()
  handoff, hit, dead_at = False, 0.0, -1
  prev_x = float(o[0])
  fell_any = 0
  zone_y = []
  for t in range(EP_LEN):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a_low = np.asarray(base_act(jnp.asarray(oc[None]))[0])
      a_hl = None
    else:
      a_hl = (td3.action(o[:29], explore=(mode == 'train'))
              if td3 is not None else np.array([0.0, (0.8 - V_LO) /
                                                (V_HI - V_LO) * 2 - 1]))
      y_ref, v_ref = decode(a_hl)
      a_low = walker(o, y_ref, v_ref)
    o2, r_env, _, info = env.step(a_low)
    hit = max(hit, float(r_env))
    dead = bool(info.get('dead'))
    if dead and dead_at < 0:
      dead_at = t
    q = env._env.data.qpos
    fell = torso_up(np.asarray(q)) < 0.3 or float(q[2]) < 0.2
    fell_any += int(fell)
    if ZONE[0] <= x <= ZONE[1] and abs(y) < 2.0 and not handoff:
      zone_y.append(y)
    if mode == 'train' and a_hl is not None:
      dx = float(o2[0]) - prev_x
      reached_handoff = (float(o2[0]) >= HANDOFF_X or float(o2[1]) >= 2.0)
      r = CP * dx - CTIME - (CFALL if fell else 0.0)
      term = dead or reached_handoff
      if dead:
        r -= RDEAD
      elif reached_handoff:
        r += RDONE
      td3.store(o[:29], a_hl, r, o2[:29], 0.0 if term else 1.0)
      if term:
        prev_x = float(o2[0])
        o = o2
        break
    prev_x = float(o2[0])
    o = o2
    if hit > 0 or (dead_at >= 0 and t > dead_at + 3):
      break
  zm = float(np.mean(zone_y)) if zone_y else 0.0
  return {'u': u, 'success': hit, 'dead': dead_at >= 0, 'fell': fell_any > 0,
          'timeout': hit == 0 and dead_at < 0, 'zone_mean_y': zm,
          'steps': t + 1}


def evaluate(env, walker, base_act, td3, eps=200):
  rows = [episode(env, walker, base_act, td3, 'eval', u_side=(i % 2))
          for i in range(eps)]
  def rate(key, sub=None):
    r = [x for x in rows if sub is None or x['u'] == sub]
    return float(np.mean([x[key] for x in r])) if r else 0.0
  side = float(np.mean([abs(x['zone_mean_y']) >= 0.5 for x in rows
                        if x['zone_mean_y'] != 0.0])) if rows else 0.0
  return {'success': rate('success'),
          'success_u0': rate('success', 0), 'success_u1': rate('success', 1),
          'collapse': rate('dead'), 'fall': rate('fell'),
          'timeout': rate('timeout'), 'side_fraction': side,
          'center_fraction': 1.0 - side, 'n': len(rows)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--steps', type=int, default=600_000)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--eval-only', default=None)
  ap.add_argument('--out', default='artifacts/robust_ceiling')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  cfg, walker, base_act, bstep, wmeta = C.load_controllers(
      'artifacts/walker/phase1/walker_best.pkl',
      'offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl')
  cfg.offline_dataset = ''
  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=args.seed)
  eval_env = envs_mod.make_env('offline_ant_umaze_litter', cfg,
                               seed=args.seed + 10_000)
  assert env.collapse_force == 80.0 and env.collapse_speed == 1.2

  td3 = TD3(args.seed)
  if args.eval_only:
    td3.pi, _ = P.load_residual(args.eval_only)
    r = evaluate(eval_env, walker, base_act, td3, eps=200)
    print(json.dumps(r, indent=2))
    return

  # baseline: fixed middle_slow (td3=None uses y=0,v=0.8) in THIS eval harness
  base_ms = evaluate(eval_env, walker, base_act, None, eps=200)
  print('fixed middle_slow (this harness):', json.dumps(base_ms), flush=True)

  log = open(os.path.join(args.out, 'train_log.jsonl'), 'a')
  best, steps, t0 = -1.0, 0, time.time()
  EVAL_EVERY = 50_000
  while steps < args.steps:
    s = episode(env, walker, base_act, td3, 'train')
    n = s['steps']
    if steps > WARMUP:
      for _ in range(n):
        td3.update()
    ns = steps + n
    if ns // EVAL_EVERY > steps // EVAL_EVERY:
      ev = evaluate(eval_env, walker, base_act, td3, eps=100)
      ev.update(step=ns, sps=ns / (time.time() - t0))
      log.write(json.dumps(ev) + '\n')
      log.flush()
      print(f'[{ns:7d}] succ {ev["success"]:.3f} (u0 {ev["success_u0"]:.2f} '
            f'u1 {ev["success_u1"]:.2f}) collapse {ev["collapse"]:.2f} '
            f'timeout {ev["timeout"]:.2f} center {ev["center_fraction"]:.2f} '
            f'({ev["sps"]:.0f} sps)', flush=True)
      P.save_residual(os.path.join(args.out, 'ceiling_latest.pkl'), td3.pi,
                      {'step': ns, **ev})
      if ev['success'] > best:
        best = ev['success']
        P.save_residual(os.path.join(args.out, 'ceiling_best.pkl'), td3.pi,
                        {'step': ns, **ev})
    steps = ns

  # final 200-episode balanced-U eval of the best checkpoint
  td3.pi, meta = P.load_residual(os.path.join(args.out, 'ceiling_best.pkl'))
  final = evaluate(eval_env, walker, base_act, td3, eps=200)
  final['fixed_middle_slow_harness'] = base_ms
  final['fixed_middle_slow_dataset'] = FROZEN_MIDDLE_SLOW
  final['best_step'] = meta.get('step')
  json.dump(final, open(os.path.join(args.out, 'final_eval.json'), 'w'),
            indent=2)
  print('\n=== ROBUST CEILING (best, 200 eps balanced U) ===')
  print(json.dumps(final, indent=2))


if __name__ == '__main__':
  main()
