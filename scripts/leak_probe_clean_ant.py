"""Does the CLEAN-dataset ant read the hidden rockfall mask? Analysis only.

The worry: the learner appears to detour at the designated rockfall sites, i.e.
the hidden mask has leaked into the 58-dim observation and the confounder is
broken. Two hypotheses have to be told apart, and only one of them is fatal:

  H1  LOCATION-conditioned, mask-BLIND: the ant dodges the site x-positions
      unconditionally, the same way whether or not the site is active. That is
      the marginal of the teacher's behaviour and is NOT a confounding failure
      -- a blind learner is allowed to learn "the middle of the corridor at
      x~3-5 is bad on average".
  H2  MASK-conditioned: the ant dodges only the ACTIVE sites, before any rock
      is physically in play. That means the mask is observable. FATAL.

Three tests, each of which H2 must pass to be ruled out:

  A  PAIRED MASK-FLIP (hard identity test). Same start state, same rng, mask
     (0,0,0,0) vs (1,1,1,1), severity 'mild' so neither arm dies early. With no
     leak the two observation streams are BYTE-IDENTICAL until the first rock
     is physically dropped; any earlier divergence is a leak. Only pairs whose
     treated arm actually drops a rock are informative -- reported separately,
     because a pair with no event tests nothing.
  B  P(active | triggered) per site. The mask is drawn i.i.d. Bernoulli(
     p_active) at reset, independently of the ant. If the ant cannot see it,
     then conditioning on the ant having triggered site i must leave the site's
     active probability at exactly p_active. Systematically below p_active =
     the ant is steering away from active sites = leak.
  C  LATERAL PROFILE at the site windows, split by active/inactive. Under H1
     the |y| profile in a site's x-window is the same for active and inactive
     draws (a location effect); under H2 the active draws are pushed further
     out (a mask effect). Reported per site with a Mann-Whitney U test.

Usage:
  python scripts/leak_probe_clean_ant.py                       # full probe
  python scripts/leak_probe_clean_ant.py --n 40 --pairs 20     # smoke
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

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = 'artifacts/clean_ant_leak_probe'
CKPT = 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl'
SEED = 64_209
P_ACTIVE = 0.30
HORIZON = 800


def build(cfg, ckpt_path):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  params = st.policy_params

  @jax.jit
  def act(o):
    return jnp.tanh(nets.policy_network.apply(params, o).loc)

  return act, int(step)


def set_state(env, qpos, qvel, goal_xy, mask, severities):
  import mujoco
  env.reset(mask=mask, severities=severities)
  d = env._env.data
  d.qpos[:RA.NQ_ANT] = qpos
  d.qvel[:RA.NV_ANT] = qvel
  d.qacc_warmstart[:] = 0.0
  env._goal_vec = np.zeros(29, np.float32)
  env._goal_vec[:2] = goal_xy
  env._goal_state_full = env._goal_vec.copy()
  env._env.goal = np.asarray(goal_xy, float).copy()
  mujoco.mj_forward(env._env.model, d)
  env._last_obs = env._env._obs_dict()
  return env._flatten(env._last_obs)


def wilson(k, n, z=1.96):
  """Wilson score interval -- honest at the small counts this probe produces."""
  if n == 0:
    return (None, None)
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (round(float(c - h), 4), round(float(c + h), 4))


def mannwhitney(a, b):
  """Two-sided Mann-Whitney U with a normal approximation (ties corrected).
  Returns (U, z, p, rank-biserial effect size) or None if either arm is thin."""
  a, b = np.asarray(a, float), np.asarray(b, float)
  n1, n2 = len(a), len(b)
  if n1 < 5 or n2 < 5:
    return None
  allv = np.concatenate([a, b])
  order = np.argsort(allv, kind='mergesort')
  ranks = np.empty(len(allv), float)
  sv = allv[order]
  i = 0
  while i < len(sv):
    j = i
    while j + 1 < len(sv) and sv[j + 1] == sv[i]:
      j += 1
    ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
    i = j + 1
  r1 = ranks[:n1].sum()
  u1 = r1 - n1 * (n1 + 1) / 2.0
  mu = n1 * n2 / 2.0
  _, cnt = np.unique(allv, return_counts=True)
  tie = float(np.sum(cnt ** 3 - cnt))
  n = n1 + n2
  sd = np.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))))
  if sd == 0:
    return None
  z = (u1 - mu) / sd
  from math import erfc, sqrt
  p = erfc(abs(z) / sqrt(2.0))
  return {'U': float(u1), 'z': round(float(z), 3), 'p': round(float(p), 4),
          'rank_biserial': round(float(2 * u1 / (n1 * n2) - 1), 3),
          'n_active': n1, 'n_inactive': n2}


# --------------------------------------------------------------- test A -----
def paired_maskflip(env, act, n_pairs, horizon, log_every=20):
  """Same start, mask all-off vs all-on. Byte-identical until the first drop
  is the no-leak prediction. Severity 'mild' keeps both arms alive so the
  comparison is not truncated by an absorbing collapse."""
  sev = ('mild',) * 4
  rows = []
  for k in range(n_pairs):
    env.reset()
    q0 = np.asarray(env._env.data.qpos)[:RA.NQ_ANT].copy()
    v0 = np.asarray(env._env.data.qvel)[:RA.NV_ANT].copy()
    goal = env._flatten(env._last_obs)[29:31].copy()
    tr = {}
    for tag, mask in (('a', (0, 0, 0, 0)), ('b', (1, 1, 1, 1))):
      o = set_state(env, q0, v0, goal, mask, sev)
      obs_l, drops, cont = [], [], []
      for _ in range(horizon):
        a = np.asarray(act(jnp.asarray(o[None]))[0])
        obs_l.append(o.copy())
        o, _, _, info = env.step(a)
        drops.append(bool(any(info['dropped'])))
        cont.append(bool(info['rock_any_contact']))
      tr[tag] = (np.asarray(obs_l), drops, cont)
    dif = np.abs(tr['a'][0] - tr['b'][0]).max(axis=1)
    div = int(np.argmax(dif > 1e-9)) if (dif > 1e-9).any() else None
    fdrop = tr['b'][1].index(True) if True in tr['b'][1] else None
    fcont = tr['b'][2].index(True) if True in tr['b'][2] else None
    if fdrop is None:
      verdict = 'no_event'          # nothing dropped: pair tests nothing
    elif div is None:
      verdict = 'identical'         # dropped, ant never even reacted
    elif div > fdrop:
      verdict = 'reactive'          # diverged only AFTER the physical drop
    else:
      verdict = 'LEAK'              # diverged BEFORE anything physical
    rows.append({'pair': k, 'div': div, 'first_drop': fdrop,
                 'first_contact': fcont, 'verdict': verdict,
                 'lead': (None if (div is None or fdrop is None)
                          else int(div - fdrop))})
    if (k + 1) % log_every == 0:
      print(f'  [A] {k + 1}/{n_pairs} pairs', flush=True)
  return rows


# ------------------------------------------------------------ tests B/C -----
def eval_episodes(env, act, n, horizon, log_every=25):
  """Natural-mask rollouts; per-site entry/trigger/mask plus the closest
  lateral approach inside each site's x-window on the ant's first pass."""
  rows = []
  for k in range(n):
    o = env.reset()
    mask = list(env.rockfall_mask)
    entered = [False] * 4
    #: closest approach to the trigger band inside each site x-window,
    #: measured as signed lane y * site sign (large = deep into the band).
    approach = [-np.inf] * 4
    passed = False
    hit, dead_at = 0.0, -1
    for t in range(horizon):
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, r, _, info = env.step(a)
      x, y = float(o[0]), float(o[1])
      if not passed:
        for i, (_, sx, sgn) in enumerate(RA.ROCKFALL_SITES):
          if abs(x - sx) <= RA.TRIG_HALF_X:
            approach[i] = max(approach[i], sgn * y)
            if RA.TRIG_Y_BAND[0] <= sgn * y <= RA.TRIG_Y_BAND[1]:
              entered[i] = True
      if x >= RP.HANDOFF_X or y >= 2.0:
        passed = True
      hit = max(hit, float(r))
      if info['dead'] and dead_at < 0:
        dead_at = t
      if hit > 0 or (dead_at >= 0 and t > dead_at + 5):
        break
    rows.append({'mask': mask, 'entered': entered,
                 'triggered': [bool(v) for v in env._triggered],
                 'dropped': [bool(v) for v in env._dropped],
                 'hit': [bool(v) for v in env._hit],
                 'approach': [None if not np.isfinite(v) else round(float(v), 4)
                              for v in approach],
                 'success': float(hit > 0), 'dead': dead_at >= 0,
                 'steps': int(t + 1)})
    if (k + 1) % log_every == 0:
      print(f'  [B/C] {k + 1}/{n} episodes', flush=True)
  return rows


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT)
  ap.add_argument('--n', type=int, default=300, help='natural-mask episodes')
  ap.add_argument('--pairs', type=int, default=150, help='mask-flip pairs')
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--p-active', type=float, default=P_ACTIVE)
  ap.add_argument('--seed', type=int, default=SEED)
  ap.add_argument('--out-dir', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  envs_mod.make_env('offline_ant_umaze', cfg, seed=1)
  cfg.rockfall_severity = V2.SEVERITY_V2
  cfg.rockfall_p_active = float(args.p_active)
  cfg.rockfall_max_steps = int(args.horizon)
  cfg.rockfall_reset_fix = True
  act, step = build(cfg, args.ckpt)
  print(f'ckpt {args.ckpt} @ step {step} | p_active {args.p_active} | '
        f'H {args.horizon}', flush=True)

  # ---- A: paired mask-flip ----
  penv = envs_mod.make_env('offline_ant_umaze_rockfall', cfg,
                           seed=args.seed + 1)
  print(f'[A] paired mask-flip, {args.pairs} pairs '
        f'(0,0,0,0) vs (1,1,1,1), severity mild', flush=True)
  pa = paired_maskflip(penv, act, args.pairs, args.horizon)
  info_pairs = [r for r in pa if r['verdict'] != 'no_event']
  leaks = [r for r in pa if r['verdict'] == 'LEAK']
  leads = [r['lead'] for r in info_pairs if r['lead'] is not None]
  A = {'pairs': len(pa), 'informative': len(info_pairs),
       'no_event': len(pa) - len(info_pairs),
       'reactive': sum(r['verdict'] == 'reactive' for r in pa),
       'identical_after_drop': sum(r['verdict'] == 'identical' for r in pa),
       'LEAK': len(leaks),
       'median_steps_div_after_drop': (int(np.median(leads)) if leads
                                       else None),
       'leak_pairs': leaks[:10]}
  print(f'[A] informative {A["informative"]}/{A["pairs"]} | '
        f'reactive {A["reactive"]} | LEAK {A["LEAK"]}', flush=True)

  # ---- B/C: natural-mask evaluation ----
  env = envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=args.seed)
  print(f'[B/C] {args.n} natural-mask episodes', flush=True)
  rows = eval_episodes(env, act, args.n, args.horizon)

  names = [nm for nm, _, _ in RA.ROCKFALL_SITES]
  B = {'p_active_prior': args.p_active, 'per_site': {}}
  k_tot = n_tot = 0
  ke_tot = ne_tot = 0
  for i, nm in enumerate(names):
    trig = [r for r in rows if r['triggered'][i]]
    ent = [r for r in rows if r['entered'][i]]
    k = sum(r['mask'][i] for r in trig)
    ke = sum(r['mask'][i] for r in ent)
    k_tot += k
    n_tot += len(trig)
    ke_tot += ke
    ne_tot += len(ent)
    B['per_site'][nm] = {
        'n_triggered': len(trig),
        'p_active_given_triggered': (round(k / len(trig), 4) if trig else None),
        'ci95': wilson(k, len(trig)),
        'n_entered_band': len(ent),
        'p_active_given_entered': (round(ke / len(ent), 4) if ent else None)}
  B['pooled'] = {'n_triggered': n_tot,
                 'p_active_given_triggered': (round(k_tot / n_tot, 4)
                                              if n_tot else None),
                 'ci95': wilson(k_tot, n_tot),
                 'n_entered_band': ne_tot,
                 'p_active_given_entered': (round(ke_tot / ne_tot, 4)
                                            if ne_tot else None),
                 'ci95_entered': wilson(ke_tot, ne_tot)}
  lo, hi = B['pooled']['ci95']
  B['verdict'] = ('consistent with the prior (no steering away from active '
                  'sites)' if (lo is not None and lo <= args.p_active <= hi)
                  else 'PRIOR EXCLUDED -- the ant is mask-sensitive')
  print(f'[B] pooled P(active|triggered) = '
        f'{B["pooled"]["p_active_given_triggered"]} '
        f'{B["pooled"]["ci95"]} vs prior {args.p_active} -> {B["verdict"]}',
        flush=True)

  C = {'per_site': {}}
  for i, nm in enumerate(names):
    a_on = [r['approach'][i] for r in rows
            if r['mask'][i] == 1 and r['approach'][i] is not None]
    a_off = [r['approach'][i] for r in rows
             if r['mask'][i] == 0 and r['approach'][i] is not None]
    C['per_site'][nm] = {
        'mean_approach_active': (round(float(np.mean(a_on)), 3) if a_on
                                 else None),
        'mean_approach_inactive': (round(float(np.mean(a_off)), 3) if a_off
                                   else None),
        'band_floor': RA.TRIG_Y_BAND[0],
        'mannwhitney': mannwhitney(a_on, a_off)}
  print('[C] lateral approach active vs inactive: ' +
        json.dumps({k: (v['mean_approach_active'], v['mean_approach_inactive'],
                        None if not v['mannwhitney']
                        else v['mannwhitney']['p'])
                    for k, v in C['per_site'].items()}), flush=True)

  succ = float(np.mean([r['success'] for r in rows]))
  drop = float(np.mean([any(r['dropped']) for r in rows]))
  trig = float(np.mean([any(r['triggered']) for r in rows]))
  rep = {'ckpt': args.ckpt, 'step': step, 'seed': args.seed,
         'p_active': args.p_active, 'horizon': args.horizon,
         'n_episodes': args.n, 'n_pairs': args.pairs,
         'headline': {'success': round(succ, 3),
                      'episode_trigger_rate': round(trig, 3),
                      'episode_drop_rate': round(drop, 3),
                      'death_rate': round(float(np.mean(
                          [r['dead'] for r in rows])), 3)},
         'A_paired_maskflip': A, 'B_prior_recovery': B,
         'C_lateral_profile': C,
         'episodes': rows, 'pairs': pa}
  p = os.path.join(args.out_dir, 'leak_probe.json')
  with open(p, 'w') as f:
    json.dump(rep, f, indent=2)
  print(f'\nsuccess {succ:.3f} | trigger {trig:.3f} | drop {drop:.3f}')
  print('->', p, flush=True)


if __name__ == '__main__':
  main()
