"""PART 2: quantify the legacy-reset error. One fixed N=500 natural bank,
cap=700, evaluate teacher / center / blind / legacy-H700-naive under BOTH the
legacy reset and the corrected (full) reset. Diagnostic only -- NOT the final
benchmark. Output artifacts/reset_fix/legacy_vs_corrected_n500.json.
"""
import argparse
import json
import os
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
import rockfall_v2_teacher as V2            # noqa: E402
from diagnose_naive_rockfall import build_policy      # noqa: E402
from reconcile_rockfall_eval import (draw_natural_masks, draw_sides,
                                     mask_pattern, natural_weights,
                                     PATTERNS)          # noqa: E402

SEED = 20_260_726
PA = 0.30
CAP = 700
NAIVE = 'naive_rockfall_v2_p30_s0_300k/final.pkl'


def scripted(env, o, walker, base_act, mode, base_side):
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = (V2.active_site_windows(base_sgn, env.rockfall_mask)
          if mode == 'teacher' else [])
  true_goal = o[29:31].copy()
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  succ = dead = -1
  for t in range(CAP):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy(); oc[29:] = 0.0; oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    elif mode == 'center':
      x_hist.append(x)
      yc, vc = RP.route_command('center', t, x_hist, nudge); a = walker(o, yc, vc)
    else:
      x_hist.append(x)
      yc, vc = V2.detour_command(base_sgn, wins, x, t, x_hist, nudge, RP.V_SIDE)
      a = walker(o, yc, vc)
    o, r, _, info = env.step(a)
    if r > 0 and succ < 0:
      succ = t
    if info['dead'] and dead < 0:
      dead = t
    if succ >= 0 or (dead >= 0 and t > dead + 5):
      break
  return {'succ_step': succ, 'success': int(succ >= 0), 'dead': int(dead >= 0),
          'pattern': mask_pattern(env.rockfall_mask)}


def neural(env, act, o):
  succ = dead = -1
  ys = []
  for t in range(CAP):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    o, r, _, info = env.step(a)
    if 2.3 <= float(o[0]) <= 5.7:
      ys.append(float(o[1]))
    if r > 0 and succ < 0:
      succ = t
    if info['dead'] and dead < 0:
      dead = t
    if succ >= 0 or (dead >= 0 and t > dead + 5):
      break
  my = float(np.mean(ys)) if ys else 0.0
  route = 'left' if my > 0.5 else 'right' if my < -0.5 else 'center'
  return {'succ_step': succ, 'success': int(succ >= 0), 'dead': int(dead >= 0),
          'pattern': mask_pattern(env.rockfall_mask), 'route': route}


def summarize(rows, weights):
  n = len(rows)
  pooled = round(sum(r['success'] for r in rows) / n, 4)
  steps = [r['succ_step'] + 1 for r in rows if r['success']]
  per = {}
  for p in PATTERNS:
    v = [r['success'] for r in rows if r['pattern'] == p]
    if v:
      per[p] = round(float(np.mean(v)), 3)
  macro = round(float(np.mean(list(per.values()))), 4) if per else None
  worst = min(per.values()) if per else None
  return {'pooled_success': pooled, 'n_success': int(sum(r['success'] for r in rows)),
          'median_steps': int(np.median(steps)) if steps else None,
          'mean_steps': round(float(np.mean(steps)), 1) if steps else None,
          'per_pattern': per, 'balanced_macro': macro, 'worst_mask': worst}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n', type=int, default=500)
  ap.add_argument('--out', default='artifacts/reset_fix/legacy_vs_corrected_n500.json')
  args = ap.parse_args()
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  weights = natural_weights(PA)
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''; cfg.eval_goal_mode = 'd4rl'
  cfgn, act, _ = build_policy(NAIVE)
  cfgn.offline_dataset = ''; cfgn.eval_goal_mode = 'd4rl'
  masks = draw_natural_masks(SEED + 1, args.n, PA)
  sides = draw_sides(SEED + 2, args.n)

  def run(kind, reset_fix):
    ccfg = cfgn if kind == 'naive' else cfg
    env = V2.apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', ccfg, seed=SEED), PA,
        reset_fix=reset_fix)
    rows = []
    for i, m in enumerate(masks):
      o = env.reset(mask=m)
      if kind == 'naive':
        rows.append(neural(env, act, o))
      else:
        rows.append(scripted(env, o, walker, base_act, kind, sides[i]))
    return rows

  out = {'provenance': {'seed': SEED, 'p_active': PA, 'cap': CAP, 'n': args.n,
                        'naive_ckpt': NAIVE,
                        'reset_fix_version': V2.RESET_FIX_VERSION}, 'policies': {}}
  for kind in ('teacher', 'center', 'blind', 'naive'):
    leg = run(kind, False); cor = run(kind, True)
    dis = [i for i in range(len(masks)) if leg[i]['success'] != cor[i]['success']]
    l2cF = sum(1 for i in dis if leg[i]['success'] and not cor[i]['success'])
    l2cS = sum(1 for i in dis if not leg[i]['success'] and cor[i]['success'])
    out['policies'][kind] = {
        'legacy': summarize(leg, weights),
        'corrected': summarize(cor, weights),
        'episode_disagreements': len(dis),
        'legacy_success_to_corrected_failure': int(l2cF),
        'legacy_failure_to_corrected_success': int(l2cS)}
    lp = out['policies'][kind]['legacy']['pooled_success']
    cp = out['policies'][kind]['corrected']['pooled_success']
    print(f'{kind}: legacy pooled {lp} -> corrected {cp} | disagree '
          f'{len(dis)} (L->F {l2cF}, F->S {l2cS})', flush=True)
  json.dump(out, open(args.out, 'w'), indent=2)
  print('wrote', args.out, flush=True)


if __name__ == '__main__':
  main()
