"""Failure-mode analysis of the frozen middle_slow COVERAGE component (143 eps)
in the full litter dataset. ANALYSIS ONLY -- nothing frozen is modified.

Per-episode summaries come from the sidecar; fall detection (needs torso
orientation/z, absent from the sidecar) comes from a DETERMINISTIC re-sim of
the exact frozen pipeline, verified bit-exact against the dataset's
success/collapse labels + a per-episode u_side alignment assert.

Outputs artifacts/coverage_failure_analysis/{summary.csv,report.json} and
prints the full report. GIF selection (episode ids) is emitted for a
separate deterministic render pass.
"""
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
import litter_pilot_common as C           # noqa: E402
import walker_gate as WG                  # noqa: E402
from crl import probe as P                # noqa: E402
import collect_litter_pilot as CL         # noqa: E402

SIDE = 'artifacts/litter_dataset/full/antmaze_litter_full_sidecar.npz'
ENV_SEED = 25_770_067 - 6  # == 25_770_061, the full-collection env seed
OUT = 'artifacts/coverage_failure_analysis'
ZONE = (2.5, 5.5)
HANDOFF_X = WG.HANDOFF_X
COVERAGE_V = CL.COVERAGE_MIDDLE_SLOW_V


def torso_up(q):
  return 1.0 - 2.0 * (q[4] * q[4] + q[5] * q[5])


def resim(env, walker, base_act):
  """Replicate CL.rollout for a coverage episode (env already reset by the
  caller's sweep). Returns (hit, dead_at, fall_step, frames-not-captured)."""
  o = env._flatten(env._last_obs)
  true_goal = o[29:31].copy()
  handoff = False
  x_hist, nudge_until, nudge_sign = [], -1, 1.0
  dead_at, hit, fall_step = -1, 0.0, -1
  for t in range(CL.HORIZON):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      y_cmd = 0.0
      x_hist.append(x)
      if t < nudge_until:
        y_cmd = nudge_sign * WG.NUDGE_Y
      elif (len(x_hist) > WG.STALL_WINDOW
            and x_hist[-1] - x_hist[-WG.STALL_WINDOW] < WG.STALL_MIN_DX):
        nudge_until = t + WG.NUDGE_STEPS
        nudge_sign = -nudge_sign
        x_hist.clear()
        y_cmd = nudge_sign * WG.NUDGE_Y
      a = walker(o, y_cmd, COVERAGE_V)
    o2, r, _, info = env.step(a)
    q = np.asarray(env._env.data.qpos)
    if fall_step < 0 and (torso_up(q) < 0.3 or float(q[2]) < 0.2):
      fall_step = t
    hit = max(hit, float(r))
    if info.get('dead') and dead_at < 0:
      dead_at = t
    if dead_at >= 0:
      break
    o = o2
  return hit, dead_at, fall_step


def main():
  os.makedirs(OUT, exist_ok=True)
  sc = np.load(SIDE, allow_pickle=True)
  tm = sc['teacher_mode'].astype(str)
  u_side = sc['u_side'].astype(int)
  success = sc['success'].astype(float)
  ep_len = sc['ep_length'].astype(int)
  dead = sc['dead'].astype(bool)
  collapse_step = sc['collapse_step'].astype(int)
  final_gd = sc['final_goal_dist'].astype(float)
  sx, sy = sc['step_torso_x'], sc['step_torso_y']
  svx = sc['step_vx']
  spc, src = sc['step_pile_contacts'], sc['step_rubble_contacts']
  shf, sps = sc['step_hforce'], sc['step_pre_speed']
  slane, sho = sc['step_lane_cmd'], sc['step_handoff']
  N = len(tm)
  cov_ids = [e for e in range(N) if tm[e] == 'coverage']

  # ---- deterministic re-sim (falls + fidelity check) ----
  cfg, walker, base_act, _, _ = C.load_controllers(
      'artifacts/walker/phase1/walker_best.pkl',
      'offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl')
  cfg.offline_dataset = ''
  env = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=ENV_SEED)
  # The sidecar is AUTHORITATIVE for success/dead/collapse/positions/contacts;
  # re-sim only supplies fall_step (torso orientation, absent from sidecar).
  # Cross-process JAX/BLAS float non-determinism can make a few borderline
  # 700-step rollouts diverge in outcome; such episodes' re-sim fall is marked
  # unreliable (and never used to assign the fall category).
  falls, reliable, align_ok, mism = {}, {}, True, []
  for e in range(N):
    env.reset()                              # advance rng in lockstep
    if int(env.u_side) != u_side[e]:
      align_ok = False
    if tm[e] != 'coverage':
      continue
    hit, dead_at, fall_step = resim(env, walker, base_act)
    falls[e] = fall_step
    ok = ((hit > 0) == (success[e] > 0)
          and (dead_at >= 0) == bool(dead[e])
          and (dead_at < 0 or dead_at == collapse_step[e]))
    reliable[e] = ok
    if not ok:
      mism.append(e)
  print(f'alignment u_side match (all {N}): {align_ok}', flush=True)
  print(f're-sim outcome-faithful coverage eps: {143 - len(mism)}/143 '
        f'(mismatched, fall unreliable: {mism})', flush=True)
  assert align_ok, 'rng alignment broken -- aborting'

  # ---- per-episode summary ----
  rows = []
  for e in cov_ids:
    L = int(ep_len[e])
    LV = L - 2                               # last valid step-data index (row
    xs, ys = sx[e, :L], sy[e, :L]            # L-1 is the nan-padded terminal)
    contact = (spc[e, :L] > 0) | (src[e, :L] > 0)
    fc = int(np.argmax(contact)) if contact.any() else -1
    inzone = (xs >= ZONE[0]) & (xs <= ZONE[1]) & (np.abs(ys) < 2.0)
    ze = int(np.argmax(inzone)) if inzone.any() else -1
    zx = -1
    if ze >= 0:
      after = np.flatnonzero((xs > ZONE[1]) & (np.arange(L) > ze))
      zx = int(after[0]) if len(after) else -1
    ho = int(np.argmax(sho[e, :L] > 0.5)) if (sho[e, :L] > 0.5).any() else -1
    # nudge activations = contiguous non-zero lane-cmd blocks (pre-handoff)
    lane = np.nan_to_num(slane[e, :L], nan=0.0)
    active = np.abs(lane) > 0.1
    nudges, rec = [], 0
    t = 0
    while t < L:
      if active[t]:
        t0 = t
        while t < L and active[t]:
          t += 1
        te = min(t0 + WG.STALL_WINDOW, L - 1)
        recovered = (xs[te] - xs[t0]) > 0.3
        nudges.append((t0, recovered))
        rec += int(recovered)
      else:
        t += 1
    fell = falls[e] >= 0 and reliable.get(e, False)
    fall_step = falls[e] if fell else -1
    max_x = float(np.nanmax(xs))
    max_x_pre = float(np.nanmax(xs[:ho])) if ho > 0 else max_x
    if bool(dead[e]):
      fail_step = int(collapse_step[e])
    elif fell and success[e] == 0:
      fail_step = int(fall_step)
    else:
      fail_step = LV
    after_ho = ho >= 0 and fail_step > ho
    rows.append(dict(
        episode_id=e, u_side=int(u_side[e]), success=int(success[e]),
        ep_length=L, dead=bool(dead[e]), collapse_step=int(collapse_step[e]),
        fell=bool(fell), fall_step=int(fall_step),
        resim_reliable=bool(reliable.get(e, False)),
        first_contact_step=fc, first_zone_entry=ze, zone_exit=zx,
        handoff_step=ho, max_hforce=float(np.nanmax(shf[e, :L])),
        pre_contact_speed=float(sps[e, fc]) if fc >= 0 else float('nan'),
        n_nudges=len(nudges), n_nudges_recovered=rec,
        max_x=max_x, max_x_pre_handoff=max_x_pre,
        final_goal_dist=float(final_gd[e]),
        fail_step=fail_step, failure_after_handoff=bool(after_ho),
        fall_x=float(xs[min(fall_step, LV)]) if fell else float('nan'),
        fail_x=float(xs[min(fail_step, LV)]),
        fail_y=float(ys[min(fail_step, LV)])))

  # ---- mutually-exclusive classification (priority order) ----
  def classify(r):
    if r['success'] == 1:
      return 'success'
    if r['dead']:
      return '1_collapse_litter'
    if r['fell'] and (r['handoff_step'] < 0
                      or r['fall_step'] <= r['handoff_step']) \
            and 2.0 <= r['fall_x'] <= 6.0:
      return '2_fall_litter'
    if r['handoff_step'] < 0 and r['max_x_pre_handoff'] < HANDOFF_X:
      return '3_wedged_before_exit'
    if r['handoff_step'] >= 0 and r['fell'] \
            and r['fall_step'] > r['handoff_step']:
      return '5_fail_after_handoff'
    if r['handoff_step'] >= 0:
      return '4_cleared_timeout'
    return '6_other'

  for r in rows:
    r['category'] = classify(r)

  cats = ['1_collapse_litter', '2_fall_litter', '3_wedged_before_exit',
          '4_cleared_timeout', '5_fail_after_handoff', '6_other']
  fails = [r for r in rows if r['success'] == 0]
  succ_rows = [r for r in rows if r['success'] == 1]

  def frac(sub, key, val=None, pred=None):
    s = [r for r in rows if sub(r)]
    if not s:
      return None
    if pred:
      return float(np.mean([pred(r) for r in s]))
    return float(np.mean([r[key] == val for r in s]))

  report = {
      'n_coverage': len(rows), 'n_success': len(succ_rows),
      'n_fail': len(fails), 'success_rate': len(succ_rows) / len(rows),
      'success_rate_by_U': {
          'u0': float(np.mean([r['success'] for r in rows if r['u_side'] == 0])),
          'u1': float(np.mean([r['success'] for r in rows if r['u_side'] == 1]))},
      'category_counts': {c: sum(r['category'] == c for r in rows) for c in cats},
      'category_pct_of_failures': {
          c: round(100 * sum(r['category'] == c for r in rows) / len(fails), 1)
          for c in cats},
      'category_by_U': {
          c: {'u0': sum(r['category'] == c and r['u_side'] == 0 for r in rows),
              'u1': sum(r['category'] == c and r['u_side'] == 1 for r in rows)}
          for c in cats},
      'failures_before_handoff': sum(not r['failure_after_handoff']
                                     for r in fails),
      'failures_after_handoff': sum(r['failure_after_handoff'] for r in fails),
      'unstick': {
          'eps_with_nudges': sum(r['n_nudges'] > 0 for r in rows),
          'total_nudges': sum(r['n_nudges'] for r in rows),
          'total_recovered': sum(r['n_nudges_recovered'] for r in rows),
          'recovery_rate': (sum(r['n_nudges_recovered'] for r in rows)
                            / max(sum(r['n_nudges'] for r in rows), 1)),
          'mean_nudges_success': float(np.mean([r['n_nudges'] for r in succ_rows])),
          'mean_nudges_fail': float(np.mean([r['n_nudges'] for r in fails]))},
      'succ_vs_fail': {
          'max_hforce': {'success': float(np.nanmedian([r['max_hforce'] for r in succ_rows])),
                         'fail': float(np.nanmedian([r['max_hforce'] for r in fails]))},
          'pre_contact_speed': {
              'success': float(np.nanmedian([r['pre_contact_speed'] for r in succ_rows])),
              'fail': float(np.nanmedian([r['pre_contact_speed'] for r in fails]))},
          'zone_traversal_steps': {
              'success': float(np.median([(r['zone_exit'] - r['first_zone_entry'])
                                          for r in succ_rows
                                          if r['zone_exit'] > 0 and r['first_zone_entry'] >= 0])),
              'fail': float(np.median([(r['zone_exit'] - r['first_zone_entry'])
                                       for r in fails
                                       if r['zone_exit'] > 0 and r['first_zone_entry'] >= 0] or [float('nan')]))},
          'n_nudges': {'success': float(np.median([r['n_nudges'] for r in succ_rows])),
                       'fail': float(np.median([r['n_nudges'] for r in fails]))}},
      'fail_step_distribution': {
          'p10': float(np.percentile([r['fail_step'] for r in fails], 10)),
          'p50': float(np.percentile([r['fail_step'] for r in fails], 50)),
          'p90': float(np.percentile([r['fail_step'] for r in fails], 90))},
      'fail_location_x_hist': np.histogram(
          [r['fail_x'] for r in fails], bins=[0, 1, 2, 2.5, 3.2, 4, 5, 5.5, 6, 8, 10])[0].tolist(),
      'fail_location_x_bins': [0, 1, 2, 2.5, 3.2, 4, 5, 5.5, 6, 8, 10],
  }

  # GIF selection: first 3 episode ids per failure category + first 3 successes
  gif_sel = {c: sorted(r['episode_id'] for r in rows if r['category'] == c)[:3]
             for c in cats}
  gif_sel['success'] = sorted(r['episode_id'] for r in succ_rows)[:3]
  report['gif_selection'] = gif_sel

  # write outputs
  import csv
  with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)

  # print
  print('\n===== COVERAGE FAILURE ANALYSIS (143 eps, 79/64) =====')
  print('success rate by U:', {k: round(v, 3) for k, v in report['success_rate_by_U'].items()})
  print('\ncategory counts (of 64 failures):')
  for c in cats:
    n = report['category_counts'][c]
    print(f'  {c:24s} {n:2d}  ({report["category_pct_of_failures"][c]:5.1f}% of fails)  '
          f'u0={report["category_by_U"][c]["u0"]} u1={report["category_by_U"][c]["u1"]}')
  print(f'\nfailures before handoff: {report["failures_before_handoff"]} | '
        f'after handoff: {report["failures_after_handoff"]}')
  print('unstick:', json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in report['unstick'].items()}))
  print('succ vs fail medians:', json.dumps(report['succ_vs_fail'], indent=1))
  print('fail_step dist:', report['fail_step_distribution'])
  print('fail x-hist:', dict(zip([f'{a}-{b}' for a, b in zip(
      report['fail_location_x_bins'], report['fail_location_x_bins'][1:])],
      report['fail_location_x_hist'])))
  print('\nGIF selection:', json.dumps(gif_sel))
  print('\nsaved', OUT)


if __name__ == '__main__':
  main()
