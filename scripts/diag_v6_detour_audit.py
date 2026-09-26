"""AntMaze V6 mainline pilot, variant critic_clip0.1: audit of the far-route
timeouts of the clipped CF policies (frozen models, no retraining).

Starting point (user's recount, 2026-09-19): on the 300 development
evaluation episodes the clipped CF policies reach the far route (the env's
'detour' label = the top-west corner, y >= 6 at x < 2) in 170 / 155 / 108
episodes and finish 111 / 122 / 87 of those (65 / 79 / 81 %, pooled 74 %); no
far-route episode dies, every far-route loss is a timeout.  So two things
remain open -- fewer than half of the episodes take the far route, and about
a quarter of those that do run out of time -- and the earlier reading
"the detour is completed 40-70 % of the time" was wrong.

This audit takes the far-route timeouts of the CURRENT (clipped) policies --
not the base CF's entrances of diag_traj/cont.json -- and separates

  (a) entered the far route too late for the remaining budget: the steps left
      at the top-west corner against the completion times of the successful
      far-route episodes from the same corner;
  (b) time enough, but stalled / oscillating / against a wall on the way
      (the diag_v6_pilot_trajectories end-state classes, with the progress
      reached and the stall onset);
  (c) an unfavourable entrance pose / velocity: torso height, uprightness,
      planar speed and heading at the corner, timeouts against successes.

Then, before any claim of an execution regression caused by the update, the
continuation test: from the SAME entrance state (the clipped CF's own prefix
up to the corner, and up to the turn north at y >= 2) with the SAME
remaining time (horizon - tau), the continuation is handed to the clipped CF
itself (control: must reproduce the recorded outcome), the start policy (the
actor's initialisation), the O policy of the same seed (same recipe,
recorded futures), the base CF of the same seed and the teacher's blind
position driver -- from the timeout entrances AND from the success entrances,
so that every continuation policy is scored on both.  Reading rules, fixed
before the run:

  R1  the driver fails from a timeout entrance -> that entrance is not
      completable in the time left (a / c), whatever the policy;
  R2  the start policy reaches from the timeout entrances at about its rate
      from the success entrances while the clipped CF does not -> the loss
      lies in the updated policy's execution after the entrance (b, an
      update-caused regression);
  R3  the start policy reaches from the timeout entrances well below its
      rate from the success entrances -> the entrance (state or time)
      carries the difference, not the update.

Usage (CPU; the trajectories of the six clipped policies are captured first):
  python scripts/diag_v6_detour_audit.py rollout --variant critic_clip0.1 --workers 6
  python scripts/diag_v6_detour_audit.py audit   --variant critic_clip0.1 --workers 18
  python scripts/diag_v6_detour_audit.py report  --variant critic_clip0.1
Outputs: diag_traj/traj_<policy>@<variant>.npz, diag_traj/detour_audit_<variant>.json,
diag_traj/DETOUR_AUDIT_<variant>.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import diag_v6_pilot_trajectories as DT  # noqa: E402  (CPU JAX, the pilot's pins, the capture / classification / branching machinery)
import exp_v6_mainline_pilot as MP  # noqa: E402

OUT = DT.OUT
HORIZON, SEEDS, STATE_DIM, OBS_W = DT.HORIZON, DT.SEEDS, DT.STATE_DIM, DT.OBS_W
DETOUR_Y, DETOUR_MAX_X, NORTH_Y = 6.0, 2.0, 2.0      # the env's far-route label (crl.rockfall_clock_v6.DETOUR_Y / DETOUR_MAX_X); the turn north
AUDIT_SEED = 151_000_000
PCT_LATE = 10                                        # "too late" = steps left below this percentile of the successful completion times


def vnames(variant):
  return {f'{arm}_s{s}': f'{arm}_s{s}@{variant}' for arm in MP.ARMS for s in SEEDS}


# ---------------------------------------------------------------- rollout
def mode_rollout(args):
  """The six clipped policies on the same 300 evaluation episodes, full capture
  (diag_v6_pilot_trajectories._rollout_worker), checked against their eval
  JSONs and against the base trajectories' hidden draws."""
  OUT.mkdir(parents=True, exist_ok=True)
  from multiprocessing import get_context
  v = args.variant
  P = {n: p for n, p in DT.policies(v).items() if n.endswith(f'@{v}')}
  n_ep = DT.N_EP if args.limit is None else min(args.limit, DT.N_EP)
  todo = [(n, p, n_ep) for n, p in P.items() if not (OUT / f'traj_{n}.npz').exists() or args.force]
  t0 = time.time()
  if todo:
    if args.workers <= 1:
      res = [DT._rollout_worker(a) for a in todo]
    else:
      with get_context('spawn').Pool(min(args.workers, len(todo))) as pool:
        res = pool.map(DT._rollout_worker, todo)
    for name, obs_rows, act_rows, lengths, eps in res:
      DT.save_traj(name, obs_rows, act_rows, lengths, eps)
  chk, ref = {}, None
  if (OUT / 'traj_start.npz').exists():
    T0 = DT.Traj('start'); ref = np.concatenate([T0.d['u1'], T0.d['u2'], T0.d['t0_1'], T0.d['t0_2'], T0.d['jitter'].ravel()])
  evp = MP.eval_paths(MP.variant_base(v))
  for name in P:
    T = DT.Traj(name)
    ev = MP.read_json(evp[f'{name.split("@")[0][:-3]}/seed_{name.split("@")[0][-1]}'])['episodes']
    same = sum(int(bool(T.d['success'][k]) == bool(ev[k]['success']) and bool(T.d['failure'][k]) == bool(ev[k]['failure'])
                   and str(T.d['route'][k]) == str(ev[k]['route']) and int(T.d['steps'][k]) == int(ev[k]['steps'])) for k in range(len(T.d['steps'])))
    hid = np.concatenate([T.d['u1'], T.d['u2'], T.d['t0_1'], T.d['t0_2'], T.d['jitter'].ravel()])
    chk[name] = {'episodes_matching_eval_json (outcome, route, steps)': f'{same}/{len(T.d["steps"])}',
                 'hidden_draws_identical_to_base_start_trajectories': (bool(np.array_equal(hid, ref)) if ref is not None else None),
                 'success': float(T.d['success'].mean()), 'timeout': float(T.d['timeout'].mean())}
  MP.write_json(OUT / f'rollout_check_{v}.json', {'wall_seconds': time.time() - t0, 'checks': chk})
  print(json.dumps(chk, indent=1), flush=True)


# ------------------------------------------------------------------ audit
def first(mask):
  return int(np.argmax(mask)) if mask.any() else -1


def entrance_steps(obs):
  xy = obs[:, :2]
  return {'t_north': first(xy[:, 1] >= NORTH_Y), 't_top': first((xy[:, 1] >= DETOUR_Y) & (xy[:, 0] < DETOUR_MAX_X))}


def entrance_features(o):
  """State features at one entrance row: torso height, uprightness, planar
  speed, heading alignment with the next leg (+x from the corner), the mean
  |joint velocity|."""
  v = o[15:17]
  sp = float(np.linalg.norm(v))
  return {'torso_z': float(o[2]), 'up_z': float(DT.up_z(o[3:7])), 'speed': sp, 'vx': float(v[0]), 'vy': float(v[1]),
          'heading_x': (float(v[0] / sp) if sp > 1e-6 else 0.0), 'joint_speed': float(np.abs(o[21:29]).mean())}


def auc(a, b):
  """P(a > b) + 0.5 P(a == b) for two samples (rank statistic)."""
  a, b = np.asarray(a, float), np.asarray(b, float)
  if not len(a) or not len(b):
    return float('nan')
  return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def pct(v, q):
  return float(np.percentile(v, q)) if len(v) else float('nan')


def mode_audit(args):
  v = args.variant
  names = vnames(v)
  T = {n: DT.Traj(names[n]) for n in names}
  Tb = {n: DT.Traj(n) for n in ('start', *[f'{arm}_s{s}' for arm in MP.ARMS for s in SEEDS])}
  # 1. per-episode table of every far-route episode of the clipped CF policies (+ the base CF and start for reference)
  per = {}
  for lab, Tr in [(names[f'CF_s{s}'], T[f'CF_s{s}']) for s in SEEDS] + [(f'CF_s{s}', Tb[f'CF_s{s}']) for s in SEEDS] + [('start', Tb['start'])]:
    rows = []
    for k in range(len(Tr.d['steps'])):
      if str(Tr.d['route'][k]) != 'detour':
        continue
      obs = Tr.obs(k)
      e = entrance_steps(obs)
      steps = int(Tr.d['steps'][k])
      r = {'episode': k, 'outcome': ('success' if Tr.d['success'][k] else ('death' if Tr.d['failure'][k] else 'timeout')), 'steps': steps,
           'u': 'U%d%d' % (bool(Tr.d['u1'][k]), bool(Tr.d['u2'][k])), **e,
           'completion_from_top': (steps - e['t_top']) if Tr.d['success'][k] and e['t_top'] >= 0 else None,
           'completion_from_north': (steps - e['t_north']) if Tr.d['success'][k] and e['t_north'] >= 0 else None,
           'left_at_top': (HORIZON - e['t_top']) if e['t_top'] >= 0 else None, 'left_at_north': (HORIZON - e['t_north']) if e['t_north'] >= 0 else None,
           'entrance': (entrance_features(obs[e['t_top']]) if e['t_top'] >= 0 else None)}
      if r['outcome'] == 'timeout':
        c = DT.classify_timeout(obs, Tr.act(k))
        r['class'] = c['class']; r['region'] = c['region']; r['arc_max'] = c['arc_max']; r['arc_end'] = c['arc_end']
        r['stall_onset'] = c['stall_onset']; r['stall_len'] = c['stall_len']; r['first_east_column'] = c['first_east_column']
        r['stall_after_top'] = (int(c['stall_onset']) - e['t_top']) if (c['stall_onset'] is not None and c['stall_onset'] >= 0 and e['t_top'] >= 0) else None
      rows.append(r)
    per[lab] = rows
  # 2. timing: the successful completion times from the corner (per clipped seed and pooled) -> the "too late" line
  comp = {lab: np.array([r['completion_from_top'] for r in rows if r['completion_from_top'] is not None]) for lab, rows in per.items()}
  pooled = np.concatenate([comp[names[f'CF_s{s}']] for s in SEEDS])
  timing = {lab: {'n_success': int(len(c)), 'min': float(c.min()) if len(c) else None, 'p10': pct(c, 10), 'median': pct(c, 50), 'p90': pct(c, 90), 'max': float(c.max()) if len(c) else None}
            for lab, c in comp.items()}
  timing['pooled_clipped_CF'] = {'n_success': int(len(pooled)), 'min': float(pooled.min()), 'p10': pct(pooled, PCT_LATE), 'median': pct(pooled, 50), 'p90': pct(pooled, 90), 'max': float(pooled.max())}
  late_line = timing['pooled_clipped_CF']['p10']; late_min = timing['pooled_clipped_CF']['min']
  for lab, rows in per.items():
    for r in rows:
      if r['outcome'] == 'timeout' and r['left_at_top'] is not None:
        r['too_late_p10'] = bool(r['left_at_top'] < late_line); r['too_late_min'] = bool(r['left_at_top'] < late_min)
        r['left_percentile_in_success_completions'] = float((pooled <= r['left_at_top']).mean())
  # 3. cases (a) / (b) / (c) for the clipped CF timeouts
  cases = {}
  for s in SEEDS:
    lab = names[f'CF_s{s}']; tos = [r for r in per[lab] if r['outcome'] == 'timeout']; suc = [r for r in per[lab] if r['outcome'] == 'success']
    a = [r for r in tos if r.get('too_late_p10')]
    rest = [r for r in tos if not r.get('too_late_p10')]
    by_class = {}
    for r in rest:
      by_class.setdefault(r['class'], []).append(r)
    feats = {}
    for f in ('torso_z', 'up_z', 'speed', 'heading_x', 'joint_speed'):
      ft = [r['entrance'][f] for r in tos if r['entrance']]; fs = [r['entrance'][f] for r in suc if r['entrance']]
      feats[f] = {'timeout_mean': float(np.mean(ft)) if ft else None, 'success_mean': float(np.mean(fs)) if fs else None, 'auc_timeout_gt_success': auc(ft, fs)}
    cases[lab] = {'n_far_route': len(per[lab]), 'n_success': len(suc), 'n_timeout': len(tos),
                  'a_too_late_p10': {'n': len(a), 'episodes': [r['episode'] for r in a], 'left_at_top': [r['left_at_top'] for r in a]},
                  'a_too_late_min': int(sum(bool(r.get('too_late_min')) for r in tos)),
                  'left_at_top_timeouts': {'min': min((r['left_at_top'] for r in tos), default=None), 'median': float(np.median([r['left_at_top'] for r in tos])) if tos else None, 'max': max((r['left_at_top'] for r in tos), default=None)},
                  't_top_success': {'median': float(np.median([r['t_top'] for r in suc])) if suc else None, 'p90': pct([r['t_top'] for r in suc], 90)},
                  't_top_timeout': {'median': float(np.median([r['t_top'] for r in tos])) if tos else None, 'p90': pct([r['t_top'] for r in tos], 90)},
                  'b_time_enough_by_class': {c: {'n': len(R), 'arc_max_mean': float(np.nanmean([r['arc_max'] for r in R])), 'arc_end_mean': float(np.nanmean([r['arc_end'] for r in R])),
                                                 'stall_after_top_median': float(np.median([r['stall_after_top'] for r in R if r['stall_after_top'] is not None])) if any(r['stall_after_top'] is not None for r in R) else None,
                                                 'stall_len_median': float(np.median([r['stall_len'] for r in R if r['stall_len'] is not None])) if any(r['stall_len'] is not None for r in R) else None,
                                                 'regions': {reg: sum(1 for r in R if r['region'] == reg) for reg in set(r['region'] for r in R)}}
                                             for c, R in by_class.items()},
                  'c_entrance_features': feats}
  # 4. the continuation test from the clipped CF's own entrances, same remaining time
  jobs = []
  for s in SEEDS:
    me, o_, cfb = names[f'CF_s{s}'], names[f'O_s{s}'], f'CF_s{s}'
    Tr = T[f'CF_s{s}']
    for r in per[me]:
      k = r['episode']; hid, o0 = Tr.hidden(k), Tr.obs(k)[0]
      conts = [me, 'start', o_, cfb, 'driver'] if r['outcome'] == 'timeout' else ['start', o_, cfb, 'driver']
      for ent in ('top', 'north'):
        tau = r['t_top'] if ent == 'top' else r['t_north']
        if tau is None or tau < 0:
          continue
        for cont in conts:
          jobs.append((f'audit|s{s}|{ent}|{r["outcome"]}|{cont}', k, o0, hid, me, int(tau), None, cont))
  print(f'audit {v}: {len(jobs)} continuation rollouts', flush=True)
  t0 = time.time()
  res = DT.run_branches(jobs, args.workers, seed0=AUDIT_SEED)
  print(f'done in {time.time() - t0:.0f} s', flush=True)
  # control fidelity: the clipped CF from its own entrance must reproduce the recorded timeout
  ctrl = [r for r in res if r['tag'].split('|')[3] == 'timeout' and r['tag'].split('|')[4].startswith('CF_s') and r['tag'].split('|')[4].endswith(f'@{v}')]
  fidelity = {'n': len(ctrl), 'reproduced_timeout': int(sum(1 for r in ctrl if r['timeout'])), 'reached': int(sum(1 for r in ctrl if r['success'])), 'died': int(sum(1 for r in ctrl if r['failure']))}
  # decision table
  table = []
  for s in SEEDS:
    me, o_, cfb = names[f'CF_s{s}'], names[f'O_s{s}'], f'CF_s{s}'
    for ent in ('top', 'north'):
      for outcome in ('timeout', 'success'):
        for cont, clab in ((me, 'CF clip (control)'), ('start', 'start'), (o_, 'O clip'), (cfb, 'CF base'), ('driver', 'driver')):
          rs = [r for r in res if r['tag'] == f'audit|s{s}|{ent}|{outcome}|{cont}']
          if not rs:
            continue
          row = {'seed': s, 'entrance': ent, 'from': f'{outcome} entrances', 'continuation': clab, 'n': len(rs), 'reach': float(np.mean([r['success'] for r in rs])),
                 'timeout': float(np.mean([r['timeout'] for r in rs])), 'death': float(np.mean([r['failure'] for r in rs]))}
          if outcome == 'timeout' and ent == 'top':
            late = {r['episode'] for r in per[me] if r.get('too_late_p10')}
            rl = [r for r in rs if r['episode'] in late]; rn = [r for r in rs if r['episode'] not in late]
            row['reach_too_late'] = (float(np.mean([r['success'] for r in rl])) if rl else None); row['n_too_late'] = len(rl)
            row['reach_time_enough'] = (float(np.mean([r['success'] for r in rn])) if rn else None); row['n_time_enough'] = len(rn)
          table.append(row)
  # pooled over seeds
  for ent in ('top', 'north'):
    for outcome in ('timeout', 'success'):
      for clab in ('CF clip (control)', 'start', 'O clip', 'CF base', 'driver'):
        rs = [r for r in table if r['entrance'] == ent and r['from'] == f'{outcome} entrances' and r['continuation'] == clab]
        if rs:
          n = sum(r['n'] for r in rs)
          table.append({'seed': 'pooled', 'entrance': ent, 'from': f'{outcome} entrances', 'continuation': clab, 'n': n,
                        'reach': sum(r['reach'] * r['n'] for r in rs) / n, 'timeout': sum(r['timeout'] * r['n'] for r in rs) / n, 'death': sum(r['death'] * r['n'] for r in rs) / n})
  # the ledger of every policy (eval JSONs), for the report
  ledger = {}
  for lab, p in MP.eval_paths(MP.variant_base(v)).items():
    if lab != 'start' and p.exists():
      ledger[f'{lab}@{v}'] = MP.route_ledger(MP.read_json(p)['episodes'])
  for lab, p in MP.eval_paths().items():
    if p.exists():
      ledger[lab] = MP.route_ledger(MP.read_json(p)['episodes'])
  # the no-route timeouts of the clipped CF (where they end)
  noroute = {}
  for s in SEEDS:
    Tr = T[f'CF_s{s}']; rows = []
    for k in range(len(Tr.d['steps'])):
      if str(Tr.d['route'][k]) in ('None', 'none') and bool(Tr.d['timeout'][k]):
        c = DT.classify_timeout(Tr.obs(k), Tr.act(k)); e = entrance_steps(Tr.obs(k))
        rows.append({'episode': k, 'class': c['class'], 'region': c['region'], 'max_y': c['max_y'], 'max_x': c['max_x'], 'stall_onset': c['stall_onset'], **e})
    noroute[names[f'CF_s{s}']] = {'n': len(rows), 'by_region': {reg: sum(1 for r in rows if r['region'] == reg) for reg in set(r['region'] for r in rows)},
                                  'by_class': {c: sum(1 for r in rows if r['class'] == c) for c in set(r['class'] for r in rows)},
                                  'turned_north': int(sum(1 for r in rows if r['t_north'] >= 0)), 'stall_onset_median': float(np.median([r['stall_onset'] for r in rows])) if rows else None, 'episodes': rows}
  MP.write_json(OUT / f'detour_audit_{v}.json', {'variant': v, 'late_line': {'percentile': PCT_LATE, 'steps': late_line, 'min_success_completion': late_min},
                                                 'timing': timing, 'cases': cases, 'fidelity': fidelity, 'table': table, 'ledger': ledger, 'no_route_timeouts': noroute,
                                                 'per_episode': per, 'results': res})
  print(json.dumps({'fidelity': fidelity, 'late_line': late_line, 'cases': {k: {**{kk: vv for kk, vv in c.items() if kk in ('n_far_route', 'n_success', 'n_timeout', 'a_too_late_min')}, 'a_too_late_p10': c['a_too_late_p10']['n']} for k, c in cases.items()}}, indent=1), flush=True)
  for r in table:
    if r['seed'] == 'pooled':
      print(r, flush=True)


# ----------------------------------------------------------------- report
def mode_report(args):
  v = args.variant
  A = MP.read_json(OUT / f'detour_audit_{v}.json')
  names = vnames(v)
  L = [f'# Far-route timeouts of the clipped CF policies ({v}): late entrance, stalls, entrance state, and the continuation test', '',
       f'Same 300 evaluation episodes (seed {DT.EVAL_SEED}, mode policy) replayed with full capture for the six {v} policies '
       f'(`rollout_check_{v}.json`); far route = the env\'s detour label (top-west corner, y >= {DETOUR_Y:.0f} at x < {DETOUR_MAX_X:.0f}); '
       f'turn north = y >= {NORTH_Y:.0f}.  `detour_audit_{v}.json`.', '']
  # ledger
  L += ['## 1. Route ledger', '', '| policy | far route n (share) | completed | far timeouts / deaths | completion | shortcut n / success / deaths | no route n / deaths / timeouts | success | if far timeouts rescued |',
        '|---|---:|---:|---|---:|---|---|---:|---:|']
  order = ['start'] + [f'O/seed_{s}' for s in SEEDS] + [f'CF/seed_{s}' for s in SEEDS] + [f'O/seed_{s}@{v}' for s in SEEDS] + [f'CF/seed_{s}@{v}' for s in SEEDS]
  for lab in order:
    if lab not in A['ledger']:
      continue
    g = A['ledger'][lab]; f, c, z = g['far_route'], g['shortcut'], g['no_route']
    L.append(f'| {lab} | {f["n"]} ({f["share"]:.2f}) | {f["success"]} | {f["timeout"]} / {f["death"]} | {f["completion"]:.3f} | {c["n"]} / {c["success"]} / {c["death"]} | {z["n"]} / {z["death"]} / {z["timeout"]} | {g["success"]:.3f} | {g["success_if_far_route_timeouts_rescued"]:.3f} |')
  L += ['', 'The last column is bookkeeping (every far-route timeout counted as a success, nothing else changed), an upper bound for fixing the far-route walking alone.', '']
  # timing
  tm = A['timing']; ll = A['late_line']
  L += ['## 2. (a) Entered too late?', '', 'Completion time from the corner (steps) of the successful far-route episodes:', '',
        '| policy | n | min | p10 | median | p90 | max |', '|---|---:|---:|---:|---:|---:|---:|']
  for lab in [names[f'CF_s{s}'] for s in SEEDS] + ['pooled_clipped_CF'] + [f'CF_s{s}' for s in SEEDS] + ['start']:
    if lab in tm and tm[lab]['n_success']:
      t = tm[lab]; L.append(f'| {lab} | {t["n_success"]} | {t["min"]:.0f} | {t["p10"]:.0f} | {t["median"]:.0f} | {t["p90"]:.0f} | {t["max"]:.0f} |')
  L += ['', f'"Too late" line = steps left at the corner below the pooled p{ll["percentile"]} of the successful completions = {ll["steps"]:.0f} steps '
        f'(below the minimum {ll["min_success_completion"]:.0f}: certainly too late).', '',
        '| policy | far route | success | timeout | too late (p10) | too late (min) | steps left at the corner, timeouts: min / median / max | corner reached, successes: median / p90 | corner reached, timeouts: median / p90 |',
        '|---|---:|---:|---:|---:|---:|---|---|---|']
  for s in SEEDS:
    c = A['cases'][names[f'CF_s{s}']]; lt = c['left_at_top_timeouts']
    L.append(f'| CF_s{s}@{v} | {c["n_far_route"]} | {c["n_success"]} | {c["n_timeout"]} | {c["a_too_late_p10"]["n"]} | {c["a_too_late_min"]} | '
             f'{lt["min"]} / {lt["median"]:.0f} / {lt["max"]} | {c["t_top_success"]["median"]:.0f} / {c["t_top_success"]["p90"]:.0f} | {c["t_top_timeout"]["median"]:.0f} / {c["t_top_timeout"]["p90"]:.0f} |')
  # classes
  L += ['', '## 3. (b) Time enough: how the timeouts end', '', 'Classes of diag_v6_pilot_trajectories (last 100 steps); arc = progress along the far route (0-8 west column, 8-32 top corridor, 32-40 east column).', '',
        '| policy | class | n | mean max arc | mean end arc | stall onset after the corner (median) | stall length (median) | end regions |', '|---|---|---:|---:|---:|---:|---:|---|']
  for s in SEEDS:
    c = A['cases'][names[f'CF_s{s}']]
    for cls, q in sorted(c['b_time_enough_by_class'].items(), key=lambda t: -t[1]['n']):
      reg = ', '.join(f'{r} {n}' for r, n in sorted(q['regions'].items(), key=lambda t: -t[1]))
      L.append(f'| CF_s{s}@{v} | {cls} | {q["n"]} | {q["arc_max_mean"]:.1f} | {q["arc_end_mean"]:.1f} | {q["stall_after_top_median"] if q["stall_after_top_median"] is None else int(q["stall_after_top_median"])} | '
               f'{q["stall_len_median"] if q["stall_len_median"] is None else int(q["stall_len_median"])} | {reg} |')
  # entrance features
  L += ['', '## 4. (c) Entrance state at the corner: timeouts vs successes', '', 'AUC = P(timeout value > success value); 0.5 = indistinguishable.', '',
        '| policy | feature | timeout mean | success mean | AUC |', '|---|---|---:|---:|---:|']
  for s in SEEDS:
    c = A['cases'][names[f'CF_s{s}']]
    for f, q in c['c_entrance_features'].items():
      if q['timeout_mean'] is not None and q['success_mean'] is not None:
        L.append(f'| CF_s{s}@{v} | {f} | {q["timeout_mean"]:.3f} | {q["success_mean"]:.3f} | {q["auc_timeout_gt_success"]:.2f} |')
  # continuation test
  fd = A['fidelity']
  L += ['', '## 5. The continuation test: same entrance state, same remaining time', '',
        f'Control fidelity: the clipped CF from its own corner entrance reproduces the recorded timeout in {fd["reproduced_timeout"]} / {fd["n"]} cases (reached {fd["reached"]}, died {fd["died"]}).', '',
        '| seed | entrance | from | continuation | n | reach | timeout | death | reach: too late / time enough (n) |', '|---|---|---|---|---:|---:|---:|---:|---|']
  for r in A['table']:
    extra = ''
    if 'reach_too_late' in r:
      extra = f'{r["reach_too_late"] if r["reach_too_late"] is None else round(r["reach_too_late"], 2)} / {r["reach_time_enough"] if r["reach_time_enough"] is None else round(r["reach_time_enough"], 2)} ({r["n_too_late"]} / {r["n_time_enough"]})'
    L.append(f'| {r["seed"]} | {r["entrance"]} | {r["from"]} | {r["continuation"]} | {r["n"]} | {r["reach"]:.2f} | {r["timeout"]:.2f} | {r["death"]:.2f} | {extra} |')
  # reading by the pre-stated rules, per seed at the corner
  L += ['', '## 6. Reading (rules R1-R3 of the docstring, corner entrance)', '']
  for s in list(SEEDS) + ['pooled']:
    g = lambda frm, cont: next((r for r in A['table'] if r['seed'] == s and r['entrance'] == 'top' and r['from'] == frm and r['continuation'] == cont), None)
    st_t, st_s, dr_t, dr_s, cf_t, o_t, o_s, b_t, b_s = (g('timeout entrances', 'start'), g('success entrances', 'start'), g('timeout entrances', 'driver'), g('success entrances', 'driver'),
                                                        g('timeout entrances', 'CF clip (control)'), g('timeout entrances', 'O clip'), g('success entrances', 'O clip'), g('timeout entrances', 'CF base'), g('success entrances', 'CF base'))
    if not (st_t and st_s):
      continue
    ctrl = '' if cf_t is None else f', the clipped CF itself {cf_t["reach"]:.2f}'
    L.append(f'* seed {s}: from the timeout entrances (n = {st_t["n"]}) the start policy reaches {st_t["reach"]:.2f}, O clip {o_t["reach"]:.2f}, CF base {b_t["reach"]:.2f}, the driver {dr_t["reach"]:.2f}'
             f'{ctrl}; from the success entrances (n = {st_s["n"]}) start {st_s["reach"]:.2f}, O clip {o_s["reach"]:.2f}, CF base {b_s["reach"]:.2f}, driver {dr_s["reach"]:.2f}.')
  L += ['', 'R1: driver fails from a timeout entrance -> not completable in the time left.  R2: start reaches from the timeout entrances at about its success-entrance rate while the clipped CF does not -> '
        'the updated policy\'s execution after the entrance.  R3: start falls well below its success-entrance rate -> the entrance (state / time) carries the difference.  The numbers above are the evidence; the sentence is written after them in the SUMMARY.', '']
  # no-route timeouts
  L += ['## 7. The no-route timeouts (neither the corner nor a hazard zone reached)', '', '| policy | n | turned north (y >= 2) | end regions | classes | stall onset (median) |', '|---|---:|---:|---|---|---:|']
  for s in SEEDS:
    q = A['no_route_timeouts'][names[f'CF_s{s}']]
    L.append(f'| CF_s{s}@{v} | {q["n"]} | {q["turned_north"]} | ' + ', '.join(f'{r} {n}' for r, n in sorted(q['by_region'].items(), key=lambda t: -t[1])) + ' | ' + ', '.join(f'{c} {n}' for c, n in sorted(q['by_class'].items(), key=lambda t: -t[1])) + f' | {q["stall_onset_median"] if q["stall_onset_median"] is None else int(q["stall_onset_median"])} |')
  (OUT / f'DETOUR_AUDIT_{v}.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('rollout', 'audit', 'report'))
  ap.add_argument('--variant', default='critic_clip0.1')
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument('--limit', type=int, default=None)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  {'rollout': mode_rollout, 'audit': mode_audit, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
