"""PART 1: MuJoCo reset-independence correctness tests (A-E).

Gate for the corrected-reset rerun. Corrected reset (full_reset=True) MUST pass
every test; we also run the key tests under legacy reset to demonstrate the bug
the fix removes. Machine-readable report -> artifacts/reset_fix/reset_tests.json.
Exit nonzero if any corrected-reset test fails.

Determinism note: the env's per-episode init is fully determined by (env seed,
reset-count, forced mask). We never reseed the global RNG; episodes are pinned by
their explicit spec.
"""
import argparse
import hashlib
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
from reconcile_rockfall_eval import (draw_natural_masks, mask_pattern)  # noqa

SEED = 20_260_726
NAIVE = 'naive_rockfall_v2_p30_s0_300k/final.pkl'
PA = 0.30


def make(cfg, seed, reset_fix):
  return V2.apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed), PA,
      reset_fix=reset_fix)


def naive_ep(env, act, o, cap=700, record=False):
  """Deterministic neural episode; returns outcome + optional obs/state trace."""
  succ = dead = -1
  obs_tr, acts, rews = [], [], []
  for t in range(cap):
    a = np.asarray(act(jnp.asarray(o[None]))[0])
    if record:
      obs_tr.append(o.copy()); acts.append(a.copy())
    o, r, _, info = env.step(a)
    if record:
      rews.append(float(r))
    if r > 0 and succ < 0:
      succ = t
    if info['dead'] and dead < 0:
      dead = t
    if succ >= 0 or (dead >= 0 and t > dead + 5):
      break
  d = env._env.data
  out = {'succ': succ, 'dead': dead, 'steps': t + 1,
         'final_qpos': np.asarray(d.qpos).copy(),
         'final_qvel': np.asarray(d.qvel).copy(),
         'success': int(succ >= 0)}
  if record:
    out['obs'] = np.asarray(obs_tr); out['acts'] = np.asarray(acts)
    out['rews'] = np.asarray(rews)
  return out


def center_ep(env, o, walker, base_act, cap=700, record=False):
  true_goal = o[29:31].copy()
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  succ = dead = -1
  obs_tr = []
  for t in range(cap):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy(); oc[29:] = 0.0; oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      x_hist.append(x)
      y_cmd, v_cmd = RP.route_command('center', t, x_hist, nudge)
      a = walker(o, y_cmd, v_cmd)
    if record:
      obs_tr.append(o.copy())
    o, r, _, info = env.step(a)
    if r > 0 and succ < 0:
      succ = t
    if info['dead'] and dead < 0:
      dead = t
    if succ >= 0 or (dead >= 0 and t > dead + 5):
      break
  return {'succ': succ, 'dead': dead, 'steps': t + 1,
          'success': int(succ >= 0),
          'n_trig': int(sum(env._triggered)), 'n_drop': int(sum(env._dropped)),
          'n_hit': int(sum(env._hit)),
          'final_x': float(o[0]), 'final_y': float(o[1]),
          'obs': np.asarray(obs_tr) if record else None}


def traj_equal(a, b, tol=0.0):
  if a['obs'].shape != b['obs'].shape:
    return False, float('inf')
  d = float(np.abs(a['obs'] - b['obs']).max()) if a['obs'].size else 0.0
  d = max(d, float(np.abs(a['final_qpos'] - b['final_qpos']).max()),
          float(np.abs(a['final_qvel'] - b['final_qvel']).max()))
  ok = (a['success'] == b['success'] and a['succ'] == b['succ']
        and a['dead'] == b['dead'] and d <= tol)
  return ok, d


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default='artifacts/reset_fix/reset_tests.json')
  args = ap.parse_args()
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfgn, act, _ = build_policy(NAIVE)
  cfgn.offline_dataset = ''
  cfgn.eval_goal_mode = 'd4rl'
  masks = draw_natural_masks(SEED + 1, 60, PA)
  R = {'reset_fix_version': V2.RESET_FIX_VERSION}

  # ---- Test A: repeated determinism (corrected) ----
  e1 = make(cfgn, SEED, True); e2 = make(cfgn, SEED, True)
  a1 = naive_ep(e1, act, e1.reset(mask=masks[0]), record=True)
  a2 = naive_ep(e2, act, e2.reset(mask=masks[0]), record=True)
  okA, dA = traj_equal(a1, a2)
  R['A_repeated_determinism'] = {'pass': bool(okA), 'max_state_div': dA}

  # ---- Test B: previous-episode independence (corrected must pass; legacy fails)
  # Target = 2nd reset with a FIXED forced mask. The preceding episode runs for a
  # DIFFERENT number of steps in each case (100/300/500/700), which is exactly the
  # mechanism that leaves different solver warmstart. Corrected reset must make the
  # target byte-identical regardless; legacy is expected to DIFFER.
  tgt_mask = (1, 0, 1, 0)
  pre_mask = (1, 1, 1, 1)
  pre_caps = [100, 300, 500, 700]

  def run_fixed(env, o, nsteps):
    """Step the naive policy EXACTLY nsteps (no early break) -> a length-
    dependent end state (feeds the next episode's warmstart under legacy)."""
    for _ in range(nsteps):
      a = np.asarray(act(jnp.asarray(o[None]))[0])
      o, _, _, _ = env.step(a)
    return o

  def run_B(reset_fix):
    outs, pre_states = [], []
    for pcap in pre_caps:
      env = make(cfgn, SEED, reset_fix)
      run_fixed(env, env.reset(mask=pre_mask), pcap)          # preceding, pcap steps
      pre_states.append(float(np.asarray(env._env.data.qpos)[0]))
      t = naive_ep(env, act, env.reset(mask=tgt_mask), record=True)  # target
      outs.append(t)
    divs = [traj_equal(outs[0], o)[1] for o in outs[1:]]
    alleq = all(traj_equal(outs[0], o)[0] for o in outs[1:])
    return alleq, max(divs) if divs else 0.0, pre_states
  okB, dB, preB = run_B(True)
  legB_eq, legB_div, _ = run_B(False)
  R['B_prev_episode_independence'] = {
      'pass': bool(okB), 'max_target_div_corrected': dB,
      'preceding_caps': pre_caps,
      'preceding_end_x_distinct': [round(x, 4) for x in preB],
      'legacy_target_identical': bool(legB_eq),
      'legacy_max_target_div': legB_div,
      'note': 'corrected must be identical (pass); legacy expected to DIFFER '
              '(nonzero legacy_max_target_div confirms the bug the fix removes)'}

  # ---- Test C: cap-prefix identity over a bank (corrected must pass) ----
  def run_bank(cap, reset_fix):
    env = make(cfgn, SEED, reset_fix)
    return [naive_ep(env, act, env.reset(mask=m), cap=cap) for m in masks]
  b700 = run_bank(700, True); b1000 = run_bank(1000, True)
  mismC = sum(1 for r7, r10 in zip(b700, b1000)
              if not (r7['success'] == (0 <= r10['succ'] < 700)
                      and (r7['succ'] if r7['succ'] < 700 else -1)
                      == (r10['succ'] if 0 <= r10['succ'] < 700 else -1)))
  lb700 = run_bank(700, False); lb1000 = run_bank(1000, False)
  legC = sum(1 for r7, r10 in zip(lb700, lb1000)
             if r7['success'] != (0 <= r10['succ'] < 700))
  R['C_cap_prefix_identity'] = {
      'pass': mismC == 0, 'corrected_mismatch': int(mismC),
      'legacy_mismatch': int(legC), 'n': len(masks)}

  # ---- Test D: harness agreement at cap=700 (corrected) ----
  # reconcile rollout_naive vs this evaluator vs horizon naive_rollout, same bank.
  from reconcile_rockfall_eval import rollout_naive as recon_roll
  from horizon_sweep_p30 import naive_rollout as horiz_roll
  eR = make(cfgn, SEED, True); eS = make(cfgn, SEED, True)
  eH = make(cfgn, SEED, True); eH.max_episode_steps = 1000
  sR = [int(recon_roll(eR, act, eR.reset(mask=m)) > 0) for m in masks]
  sS = [naive_ep(eS, act, eS.reset(mask=m))['success'] for m in masks]
  sH = [int(0 <= horiz_roll(eH, act, eH.reset(mask=m))['succ_step'] < 700)
        for m in masks]
  disD = sum(1 for i in range(len(masks)) if not (sR[i] == sS[i] == sH[i]))
  R['D_harness_agreement'] = {'pass': disD == 0, 'disagreements': int(disD),
                              'n': len(masks)}

  # ---- Test E: center mask invariance (corrected) ----
  PATS = {'all_clear': (0, 0, 0, 0), 'left_only': (1, 0, 0, 0),
          'right_only': (0, 0, 1, 0), 'both_sides': (1, 0, 1, 0)}
  envs = {p: make(cfg, SEED + 900, True) for p in PATS}
  spread = 0.0; disagree = 0; rockfall = 0; base = None
  for _ in range(20):
    rows = {}
    for p, m in PATS.items():
      rows[p] = center_ep(envs[p], envs[p].reset(mask=m), walker, base_act,
                          record=True)
    succs = [rows[p]['success'] for p in PATS]
    if len(set(succs)) > 1:
      disagree += 1
    ref = rows['all_clear']['obs']
    for p in PATS:
      rockfall += rows[p]['n_trig'] + rows[p]['n_drop'] + rows[p]['n_hit']
      ob = rows[p]['obs']
      if ob.shape == ref.shape and ob.size:
        spread = max(spread, float(np.abs(ob - ref).max()))
      elif ob.shape != ref.shape:
        spread = float('inf')
  R['E_center_mask_invariance'] = {
      'pass': bool(disagree == 0 and rockfall == 0 and spread == 0.0),
      'success_disagreements': int(disagree), 'total_rockfall_events': int(rockfall),
      'max_traj_spread_across_masks': spread}

  corrected_tests = ['A_repeated_determinism', 'B_prev_episode_independence',
                     'C_cap_prefix_identity', 'D_harness_agreement',
                     'E_center_mask_invariance']
  all_pass = all(R[k]['pass'] for k in corrected_tests)
  R['ALL_CORRECTED_PASS'] = bool(all_pass)
  json.dump(R, open(args.out, 'w'), indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
  for k in corrected_tests:
    print(f"{'PASS' if R[k]['pass'] else 'FAIL'}  {k}  {R[k]}")
  print('legacy contrast: B identical =', R['B_prev_episode_independence']
        ['legacy_target_identical'], '| C legacy mismatch =',
        R['C_cap_prefix_identity']['legacy_mismatch'])
  print('ALL CORRECTED TESTS', 'PASS' if all_pass else 'FAILED', '->', args.out)
  return 0 if all_pass else 1


if __name__ == '__main__':
  sys.exit(main())
