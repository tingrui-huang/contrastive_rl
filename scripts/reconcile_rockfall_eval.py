"""Authoritative evaluation-aggregation reconciliation for the v2.1 rockfall
benchmark.

Motivation
----------
The committed naive-300k diagnosis (e37ebf3) reported a single "success" number
(final 0.78) computed as a POOLED mean over episodes whose masks were drawn from
the NATURAL per-site Bernoulli(0.2) distribution -- so `all_clear` (~41% of
episodes) dominates and `both_sides` (~13%) is starved (n=18/200). The teacher /
center anchors in rockfall_v2_teacher.py use the SAME natural-pooled scheme. No
policy was ever scored on a balanced-by-pattern design. This script fixes that:
ONE job, ONE seed, ALL policies, ALL aggregations.

Design
------
* p_active = 0.2 per site (unchanged, verified) -> analytic mask-PATTERN weights
  all_clear/left_only/right_only/both_sides = 0.4096/0.2304/0.2304/0.1296.
* Balanced block: K episodes PER PATTERN (within-pattern raw masks drawn from the
  natural conditional via rejection sampling). Gives clean equal-n per-pattern
  success, balanced macro, worst-mask.
* Natural block: N episodes with masks drawn straight from Bernoulli(0.2). Gives
  the pooled (natural-sample) number and the empirical per-pattern counts -- this
  is the aggregation the old report used.
* PAIRED: forcing a mask still advances the env's mask RNG (rockfall_ant
  _begin_episode draws then overrides), so every policy sees byte-identical
  (initial state, forced mask, per-site severity) at each episode index. All
  four policies are compared on the same episodes.

Policies: naive (neural checkpoint), teacher (sighted local-detour), blind
(same base side, no detour), center (mask-invariant center route).

Analysis only. No env/dataset/checkpoint modification.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod            # noqa: E402
from crl import rockfall_ant as RA          # noqa: E402
import litter_pilot_common as C             # noqa: E402
import rockfall_pilot as RP                 # noqa: E402
import rockfall_v2_teacher as T             # noqa: E402
from diagnose_naive_rockfall import build_policy  # noqa: E402

PATTERNS = ('all_clear', 'left_only', 'right_only', 'both_sides')


def natural_weights(p_active):
  """Analytic mask-PATTERN probabilities for per-site Bernoulli(p_active)."""
  sa = 1.0 - (1.0 - p_active) ** 2           # a side (2 sites) has >=1 active
  return {
      'all_clear': (1 - sa) ** 2,
      'left_only': sa * (1 - sa),
      'right_only': (1 - sa) * sa,
      'both_sides': sa * sa,
  }


def mask_pattern(m):
  la, ra = (m[0] or m[1]), (m[2] or m[3])
  return ('all_clear' if not (la or ra) else
          'left_only' if la and not ra else
          'right_only' if ra and not la else 'both_sides')


def sha256(path):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def draw_balanced_masks(seed, k, p_active):
  """k raw masks per pattern; within-pattern draws follow the natural
  Bernoulli(p_active) conditional (rejection sampling). Returns list of
  (pattern, mask_tuple) in pattern-blocked order."""
  rng = np.random.default_rng(seed)
  buckets = {p: [] for p in PATTERNS}
  need = set(PATTERNS)
  while need:
    m = tuple(int(rng.random() < p_active) for _ in range(4))
    p = mask_pattern(m)
    if len(buckets[p]) < k:
      buckets[p].append(m)
      if len(buckets[p]) == k:
        need.discard(p)
  return [(p, m) for p in PATTERNS for m in buckets[p]]


def draw_natural_masks(seed, n, p_active):
  rng = np.random.default_rng(seed)
  return [tuple(int(rng.random() < p_active) for _ in range(4))
          for _ in range(n)]


def draw_sides(seed, n):
  rng = np.random.default_rng(seed)
  return ['left' if rng.random() < 0.5 else 'right' for _ in range(n)]


def rollout_naive(env, act, o):
  """Neural-policy episode from obs o; success = max reward (matches the
  committed diagnose_naive_rockfall rollout semantics)."""
  hit, dead_at = 0.0, -1
  T_ = env.max_episode_steps
  for t in range(T_):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if hit > 0 or (dead_at >= 0 and t > dead_at + 5):
      break
  return float(hit)


def eval_policy(kind, mask_seq, side_seq, cfg, walker, base_act, act,
                seed, p_active, horizon=None, reset_fix=False):
  """Run one policy over a fixed forced-mask sequence. Returns per-episode
  {pattern, success}. Same seed + same forced masks => paired across kinds."""
  env = T.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed),
      p_active, reset_fix=reset_fix)
  if horizon is not None:
    env.max_episode_steps = int(horizon)
  assert tuple(env.severity_probs) == tuple(T.SEVERITY_V2)
  assert abs(float(env.p_active) - p_active) < 1e-9
  rows = []
  for i, m in enumerate(mask_seq):
    o = env.reset(mask=m)
    assert tuple(env.rockfall_mask) == tuple(m)
    if kind == 'naive':
      s = rollout_naive(env, act, o)
    elif kind == 'teacher':
      s = T.run_sighted(env, o, walker, base_act, side_seq[i],
                        use_detour=True)['success']
    elif kind == 'blind':
      s = T.run_sighted(env, o, walker, base_act, side_seq[i],
                        use_detour=False)['success']
    elif kind == 'center':
      s = RP.run_route(env, o, walker, base_act, 'center')['success']
    else:
      raise ValueError(kind)
    rows.append({'pattern': mask_pattern(m), 'success': float(s > 0)})
  return rows


def _wilson(k, n, z=1.96):
  if n == 0:
    return (None, None)
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (round(c - h, 3), round(c + h, 3))


def summarize(rows_bal, rows_nat, nat_weight):
  # per-pattern from the BALANCED block (equal n -> clean estimates)
  by = {p: [] for p in PATTERNS}
  for r in rows_bal:
    by[r['pattern']].append(r['success'])
  per = {}
  for p in PATTERNS:
    v = by[p]
    n = len(v)
    k = int(sum(v))
    mean = (k / n) if n else None
    per[p] = {'n': n, 'success': (round(mean, 3) if mean is not None else None),
              'wilson95': _wilson(k, n)}
  means = [per[p]['success'] for p in PATTERNS if per[p]['success'] is not None]
  balanced_macro = round(float(np.mean(means)), 4) if means else None
  worst_p = min((p for p in PATTERNS if per[p]['success'] is not None),
                key=lambda p: per[p]['success'], default=None)
  worst = {'pattern': worst_p,
           'success': (per[worst_p]['success'] if worst_p else None),
           'wilson95': (per[worst_p]['wilson95'] if worst_p else None)}
  # natural-distribution success: analytic reweight of balanced per-pattern
  nat_analytic = None
  if all(per[p]['success'] is not None for p in PATTERNS):
    nat_analytic = round(sum(nat_weight[p] * per[p]['success']
                             for p in PATTERNS), 4)
  # NATURAL block: empirical pooled + empirical counts (old-report scheme)
  nat_by = {p: [] for p in PATTERNS}
  for r in rows_nat:
    nat_by[r['pattern']].append(r['success'])
  nat_counts = {p: {'n': len(nat_by[p]),
                    'success': (round(float(np.mean(nat_by[p])), 3)
                                if nat_by[p] else None)}
                for p in PATTERNS}
  pooled_nat = round(float(np.mean([r['success'] for r in rows_nat])), 4) \
      if rows_nat else None
  return {
      'per_pattern_balanced': per,
      'balanced_macro_success': balanced_macro,
      'worst_mask': worst,
      'natural_distribution_success_analytic': nat_analytic,
      'natural_block': {'pooled_success': pooled_nat,
                        'per_pattern_counts': nat_counts,
                        'n_total': len(rows_nat)},
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--naive-ckpt',
                  default='naive_rockfall_v2_s0_300k/final.pkl')
  ap.add_argument('--k', type=int, default=100, help='balanced eps per pattern')
  ap.add_argument('--n-nat', type=int, default=200, help='natural-block eps')
  ap.add_argument('--seed', type=int, default=20_260_726)
  ap.add_argument('--out', default='artifacts/rockfall_reconcile/reconcile.json')
  ap.add_argument('--policies', nargs='+',
                  default=['naive', 'teacher', 'blind', 'center'])
  ap.add_argument('--p-active', type=float, default=RA.P_ACTIVE,
                  help='mask density for this condition (0.20/0.30/0.50). '
                       'Sets the env instance AND the natural weights/draws.')
  ap.add_argument('--horizon', type=int, default=None,
                  help='episode horizon (H=800 experiment); None keeps 700')
  ap.add_argument('--reset-fix', action='store_true',
                  help='use the canonical episode-independent full reset')
  args = ap.parse_args()
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  p_active = float(args.p_active)
  nat_weight = natural_weights(p_active)

  # provenance
  try:
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:
    commit = None

  # shared episode plan (identical across policies)
  bal = draw_balanced_masks(args.seed, args.k, p_active)
  nat = draw_natural_masks(args.seed + 1, args.n_nat, p_active)
  bal_masks = [m for _, m in bal]
  bal_sides = draw_sides(args.seed + 2, len(bal_masks))
  nat_sides = draw_sides(args.seed + 3, len(nat))
  print(f'plan: balanced {len(bal_masks)} eps ({args.k}/pattern), '
        f'natural {len(nat)} eps | seed {args.seed} | p_active {p_active}',
        flush=True)

  # controllers (walker + base handoff policy) shared by scripted policies
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'

  results = {}
  for kind in args.policies:
    act = None
    if kind == 'naive':
      cfg_n, act, step = build_policy(args.naive_ckpt)
      cfg_n.offline_dataset = ''
      cfg_n.eval_goal_mode = 'd4rl'
      cfg_use = cfg_n
    else:
      cfg_use = cfg
    print(f'--- {kind} ---', flush=True)
    rows_bal = eval_policy(kind, bal_masks, bal_sides, cfg_use, walker,
                           base_act, act, args.seed, p_active, args.horizon,
                           args.reset_fix)
    rows_nat = eval_policy(kind, nat, nat_sides, cfg_use, walker, base_act,
                           act, args.seed + 10, p_active, args.horizon,
                           args.reset_fix)
    s = summarize(rows_bal, rows_nat, nat_weight)
    results[kind] = s
    print(f'{kind}: balanced_macro={s["balanced_macro_success"]} '
          f'pooled_nat={s["natural_block"]["pooled_success"]} '
          f'nat_analytic={s["natural_distribution_success_analytic"]} '
          f'worst={s["worst_mask"]["pattern"]}={s["worst_mask"]["success"]}',
          flush=True)

  provenance = {
      'commit': commit,
      'eval_seed': args.seed,
      'severity_v2': list(T.SEVERITY_V2),
      'p_active': p_active,
      'horizon': (int(args.horizon) if args.horizon is not None else 700),
      'reset_fix': bool(args.reset_fix),
      'natural_pattern_weights': {p: round(nat_weight[p], 4)
                                  for p in PATTERNS},
      'balanced_k_per_pattern': args.k,
      'natural_block_n': args.n_nat,
      'checkpoints': {
          'naive': {'path': args.naive_ckpt, 'sha256': sha256(args.naive_ckpt)},
          'walker': {'path': RP.WALKER, 'sha256': sha256(RP.WALKER)},
          'base_handoff': {'path': RP.BASE, 'sha256': sha256(RP.BASE)},
      },
  }
  report = {'provenance': provenance, 'results': results}
  json.dump(report, open(args.out, 'w'), indent=2)
  print('\nwrote', args.out, flush=True)


if __name__ == '__main__':
  main()
