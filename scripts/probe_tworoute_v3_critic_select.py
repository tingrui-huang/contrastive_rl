"""Critic-selected evaluation of a V3 baseline: does the CRITIC decide?

Companion to probe_tworoute_v3_detour.py. That probe showed the route is
NOT a value decision under the original eval (deterministic tanh(mu)): the
unimodal actor blends the two teacher gaits at the fork and the blend sits
on a knife edge tipped by goal noise. The critic's local slope pointed
toward the shortcut in 67-86% of episodes either way -- so the critic HAS
an opinion the actor cannot express.

This probe lets the critic express it directly at eval time: at every step
draw K actions from the actor's own tanh-Gaussian (plus its mode), score
all of them with the twin-min contrastive critic f(s, a, g), act on the
argmax. Training is untouched; the candidates stay inside the actor's
support, so this is in-distribution re-ranking, not free argmax.

If the critic overestimates the shortcut (it only ever saw the u-aware
teacher take it when it was safe) the readout is: shortcut rate -> ~1 and
death rate -> P(active) * P(shortcut) -- the confounding signature the
benchmark exists to measure, no longer masked by mode averaging.

Reports per seed: route mix, success/death/timeout, deaths | active, the
t=0 candidate picture (actor sigma at the fork, lambda of the mode vs the
chosen action, fraction of candidates on each side of the mode axis), and
the same episodes' routes under the original mean-action eval.

Usage: python scripts/probe_tworoute_v3_critic_select.py --variant br \
           --ckpt v3br_crl_s0_100k/final.pkl --label v3br_s0_100k [--n 100]
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from probe_tworoute_v3_detour import (    # noqa: E402
    DATASET, OUT_ROOT, make_env, rollout)

K = 64


def build(ckpt_path, variant, k=K):
  cfg, _ = make_env(variant, seed=1)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  pp, qp = st.policy_params, st.q_params

  @jax.jit
  def act_mean(o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)

  @jax.jit
  def candidates(o, key):
    """o: [1, D]. Returns (acts [k+1, A], q [k+1], sigma [A]); index 0 is
    the actor's mode, 1..k are samples from its own tanh-Gaussian."""
    p = nets.policy_network.apply(pp, o)
    loc, scale = p.loc[0], p.scale[0]
    u = loc[None] + scale[None] * jax.random.normal(key, (k, loc.shape[0]))
    acts = jnp.concatenate([jnp.tanh(loc)[None], jnp.tanh(u)], 0)
    q = nets.q_network.apply(qp, jnp.tile(o, (k + 1, 1)), acts)
    if q.ndim == 3:
      q = jnp.min(q, axis=-1)
    return acts, jnp.diag(q), scale

  return act_mean, candidates, int(step)


def mode_axis(variant):
  """t=0 teacher-mode means from the dataset: lambda = 0 detour, 1 shortcut."""
  p = DATASET.format(v=variant)
  a = np.load(p, allow_pickle=True)
  s = np.load(p.replace('.npz', '_sidecar.npz'), allow_pickle=True)
  intent = np.asarray(s['route_intent'])
  acts = a['act']
  m_sc = acts[intent == 'shortcut', 0].mean(0)
  m_dt = acts[intent == 'detour', 0].mean(0)
  axis = m_sc - m_dt
  return lambda x: float((x - m_dt) @ axis / (axis @ axis))


def run(variant, candidates, n, seed, lam):
  _, env = make_env(variant, seed)
  rng = jax.random.PRNGKey(seed)
  eps = []
  for i in range(n):
    o = env.reset()
    u = bool(env.privileged_rockfall_active)
    t0 = {}
    calls = [0]

    def act(ob):
      nonlocal rng
      rng, sub = jax.random.split(rng)
      acts, q, sigma = candidates(ob, sub)
      acts, q, sigma = np.asarray(acts), np.asarray(q), np.asarray(sigma)
      j = int(np.argmax(q))
      if calls[0] == 0:
        lams = np.array([lam(x) for x in acts])
        sc_side = lams[1:] > 0.5
        t0.update({
            'sigma_mean': round(float(sigma.mean()), 4),
            'sigma_max': round(float(sigma.max()), 4),
            'lambda_mode': round(float(lams[0]), 3),
            'lambda_chosen': round(float(lams[j]), 3),
            'chosen_is_mode': bool(j == 0),
            'q_mode': round(float(q[0]), 4),
            'q_chosen': round(float(q[j]), 4),
            'frac_candidates_shortcut_side': round(float(sc_side.mean()), 3),
            'q_mean_shortcut_side': (round(float(q[1:][sc_side].mean()), 4)
                                     if sc_side.any() else None),
            'q_mean_detour_side': (round(float(q[1:][~sc_side].mean()), 4)
                                   if (~sc_side).any() else None)})
      calls[0] += 1
      return acts[j][None]

    rec = rollout(env, act, o)
    eps.append({'k': i, 'u': u, 'route': rec['route'],
                'success': rec['success'], 'failure': rec['failure'],
                'steps': rec['steps'],
                'first_corridor': rec.get('first_corridor'), 't0': t0})
    if (i + 1) % 25 == 0:
      print(f'  {i + 1}/{n}', flush=True)
  return eps


T0_KEYS = ('sigma_mean', 'sigma_max', 'lambda_mode', 'lambda_chosen',
           'chosen_is_mode', 'frac_candidates_shortcut_side', 'q_mode',
           'q_chosen', 'q_mean_shortcut_side', 'q_mean_detour_side')


def summarize(eps, with_t0=True):
  n = len(eps)

  def r(k):
    return sum(1 for e in eps if e['route'] == k)

  act_eps = [e for e in eps if e['u']]
  sc_act = [e for e in act_eps if e['route'] == 'shortcut']
  succ = [e for e in eps if e['success']]
  out = {
      'n': n, 'shortcut': r('shortcut') / n, 'detour': r('detour') / n,
      'none': r(None) / n,
      'success': float(np.mean([e['success'] for e in eps])),
      'death': float(np.mean([e['failure'] for e in eps])),
      'timeout': float(np.mean([not e['success'] and not e['failure']
                                for e in eps])),
      'p_active': len(act_eps) / n,
      'death_given_active': (float(np.mean([e['failure'] for e in act_eps]))
                             if act_eps else None),
      'death_given_active_and_shortcut': (
          float(np.mean([e['failure'] for e in sc_act])) if sc_act else None),
      'mean_steps_success': (float(np.mean([e['steps'] for e in succ]))
                             if succ else None)}
  if with_t0:
    out['t0'] = {k: float(np.mean([e['t0'][k] for e in eps
                                   if e['t0'].get(k) is not None]))
                 for k in T0_KEYS}
    out['t0_chosen_shortcut_side_frac'] = float(np.mean(
        [e['t0']['lambda_chosen'] > 0.5 for e in eps]))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', choices=['tr', 'br'], required=True)
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--label', required=True)
  ap.add_argument('--n', type=int, default=100)
  ap.add_argument('--seed', type=int, default=909)
  ap.add_argument('--k', type=int, default=K)
  args = ap.parse_args()
  act_mean, candidates, step = build(args.ckpt, args.variant, args.k)
  lam = mode_axis(args.variant)
  print(f'ckpt {args.ckpt} @ {step} | K={args.k}', flush=True)
  print('critic-selected rollouts', flush=True)
  sel = run(args.variant, candidates, args.n, args.seed, lam)
  print('mean-action rollouts (same seed)', flush=True)
  _, env = make_env(args.variant, args.seed)
  base = []
  for i in range(args.n):
    o = env.reset()
    rec = rollout(env, act_mean, o)
    base.append({'k': i, 'u': bool(env.privileged_rockfall_active),
                 'route': rec['route'], 'success': rec['success'],
                 'failure': rec['failure'], 'steps': rec['steps']})
  out = {'ckpt': args.ckpt, 'step': step, 'variant': args.variant,
         'k': args.k, 'n': args.n, 'seed': args.seed,
         'critic_selected': summarize(sel),
         'mean_action_same_seed': summarize(base, with_t0=False),
         'episodes_critic_selected': sel}
  print(json.dumps({k: v for k, v in out.items()
                    if k != 'episodes_critic_selected'}, indent=1), flush=True)
  d = os.path.join(OUT_ROOT, args.variant)
  os.makedirs(d, exist_ok=True)
  p = os.path.join(d, f'critic_select_{args.label}.json')
  json.dump(out, open(p, 'w'), indent=1)
  print('->', p, flush=True)


if __name__ == '__main__':
  main()
