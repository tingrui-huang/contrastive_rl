#!/usr/bin/env python
"""AntMaze V6 -- the three-layer acceptance of the CF-motion revision (user's step (3) after 5650f8d, 2026-09-21; no training).

On the TEST split of the CF-controlled rollouts (start points never seen by the fit or the selection) and for each model dir
(v4 = the original, v4c = the control continuation, v4r = the revision):
  L1  teacher forcing: every step gets the REAL state -- the one-step motion error on CF rows by row type (moving / slow / static),
      by region (turn = the first 30 rows from reset, far route, corridor, zones) and the stationary gate's confusion on the real
      static rows; the predicted speed vs the real speed on the slow rows (swaying / slow progress must be predicted, not frozen).
  L2  open loop: the REAL action sequence, the model rolled forward from the sequence start (the stationary gate as deployed; the onset
      OFF) -- xy / pose / velocity error at steps 5 / 10 / 20 / 30 / 50 / 100, and on the rollouts with a real stall segment (>= 20 rows
      below the slow speed) the model's displacement over that segment vs the real one (a stall predicted as progress).
  L2b stall entry (user's review of 37b990f): the model restarted at the REAL entry state of each real stall segment (and, separately,
      20 rows before it) and fed the recorded actions through the segment -- the model's displacement over the segment vs the real one
      and the gate's firing; separates "the local stall dynamics are not learned" from "the state error accumulated before the entry".
  L3  closed loop: the CF actor chooses the action from the MODEL's state (onset OFF) -- heading north at exactly step 30, far entry
      within 100 steps, and the reach / stall outcome INSIDE THE FIRST 400 STEPS for both the model and the real rollout (its hazards
      inactive in both zones only, so the real path is deterministic and never dies).
  Plus the no-regression check on the ORIGINAL validation rows (start-agent / logged paths): the fit's own val metrics at the selected
  step are read from the model json.
Outputs under cf_motion/accept/: report.json, REPORT.md.
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

import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_repeated_draws as RD  # noqa: E402
import fit_v6_ett_one_step_v2 as F2  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of, FAR  # noqa: E402

CFM = MP.OUT / 'cf_motion'
OUT = CFM / 'accept'
STATE_DIM, OBS_W, ACTION_DIM = MP.STATE_DIM, MP.OBS_W, MP.ACTION_DIM
SLOW = 0.03
STEPS_L2 = (5, 10, 20, 30, 50, 100)


class Motion:
  """The motion regression + stationary gate of a model dir (onset unused): s' = s + delta(s, a_q), delta zeroed where the gate fires."""

  def __init__(self, model_dir, fold):
    F2.OUT = Path(model_dir); self.P = F2.Predictor2(F2.load_model(fold))

  def step(self, s, a):
    n = len(s); ab = np.zeros((n, ACTION_DIM), np.float32)
    s2, _, p_stat = self.P.step(s.astype(np.float32), ab, a.astype(np.float32), np.zeros(n, np.int64), np.zeros(n, np.int64))
    return s2, p_stat


ROWS_FILE = 'rollouts_cf.npz'
MODEL_DIRS = None


def load_rows():
  d = np.load(CFM / ROWS_FILE, allow_pickle=False)
  return {k: d[k] for k in d.files if k != 'meta'}


def fold_of_rollouts(R):
  """The model fold for a rollout: its start anchor's fold (the fit's cross-fitting); the test rollouts are new sequences, any fold's
  model has not seen them -- the anchor's fold is used for consistency with the pipeline."""
  sup = np.load(MP.OUT / 'ett_context' / 'supervision.npz', allow_pickle=False); fa = sup['fold_anchor'].astype(np.int64)
  return fa[R['start_anchor'][R['roll_start']]]


def layer1(models, R, test_rows):
  s, a, s2 = R['s'][test_rows], R['a'][test_rows], R['s2'][test_rows]
  d_real = s2 - s; speed = R['speed'][test_rows]; static = R['static'][test_rows]; slow = R['slow'][test_rows]; reg = R['region'][test_rows]; step = R['step'][test_rows]
  grp = R['roll_group'][R['rollout'][test_rows]].astype(str)
  kind = np.where(static, 'static', np.where(slow, 'slow', 'moving'))
  phase = np.where((grp == 'reset') & (step < 30), 'turn30', np.where(np.isin(reg, FAR), 'far', np.where(np.isin(reg, [REGIONS.index('zone1'), REGIONS.index('zone2')]), 'zone', 'corridor')))
  res = {}
  for name, M in models.items():
    pred = np.zeros_like(s2); pst = np.zeros(len(s)); folds = R['_fold'][R['rollout'][test_rows]]
    for f in np.unique(folds):
      m = folds == f; p2, ps = M[int(f)].step(s[m], a[m]); pred[m] = p2; pst[m] = ps
    d_pred = pred - s; err = np.linalg.norm((d_pred - d_real)[:, :2], axis=1); err_all = np.abs(d_pred - d_real).mean(axis=1)
    row = {'n': int(len(s))}
    for k in ('moving', 'slow', 'static'):
      m = kind == k
      row[f'xy_err_{k}'] = {'median': float(np.median(err[m])), 'p90': float(np.percentile(err[m], 90)), 'n': int(m.sum())}
      row[f'all_dims_err_{k}'] = float(np.median(err_all[m]))
    for ph in ('turn30', 'far', 'corridor', 'zone'):
      m = phase == ph
      if m.sum():
        row[f'xy_err_{ph}'] = {'median': float(np.median(err[m])), 'p90': float(np.percentile(err[m], 90)), 'n': int(m.sum())}
    gate = pst > 0.5
    row['gate'] = {'fires_on_static': float(gate[static].mean()), 'fires_on_slow': float(gate[slow].mean()), 'fires_on_moving': float(gate[kind == 'moving'].mean())}
    pspeed = np.linalg.norm(d_pred[:, :2], axis=1)
    row['slow_rows_speed'] = {'real_median': float(np.median(speed[slow])), 'pred_median': float(np.median(pspeed[slow])), 'pred_frozen_share': float((pspeed[slow] == 0).mean())}
    row['static_rows_pred_speed_median'] = float(np.median(pspeed[static]))
    res[name] = row
  return res


def sequences(R, test_rollouts):
  out = []
  for i in test_rollouts:
    rows = np.flatnonzero(R['rollout'] == i)
    out.append({'id': int(i), 'rows': rows, 's': np.concatenate([R['s'][rows], R['s2'][rows[-1:]]]), 'a': R['a'][rows], 'speed': R['speed'][rows], 'group': str(R['roll_group'][i]), 'outcome': str(R['roll_outcome'][i]),
                'u1': bool(R['roll_u1'][i]), 'u2': bool(R['roll_u2'][i]), 'fold': int(R['_fold'][i]), 'goal': R['start_goal'][R['roll_start'][i]]})
  return out


def stall_segments(speed, min_len=20):
  segs = []; start = None
  for j, sp in enumerate(np.concatenate([speed, [1.0]])):
    if sp < SLOW and start is None:
      start = j
    elif sp >= SLOW and start is not None:
      if j - start >= min_len:
        segs.append((start, j))
      start = None
  return segs


def layer2(models, seqs):
  res = {}
  for name, M in models.items():
    errs = {k: [] for k in STEPS_L2}; pose = {k: [] for k in STEPS_L2}; vel = {k: [] for k in STEPS_L2}; stall = []
    for q in seqs:
      S, A = q['s'], q['a']; T = len(A); s = S[0:1].copy(); pred = [s[0]]
      for j in range(min(T, 400)):
        s, _ = M[q['fold']].step(s, A[j:j + 1]); pred.append(s[0])
      pred = np.stack(pred); n = len(pred)
      for k in STEPS_L2:
        if k < n:
          d = pred[k] - S[k]; errs[k].append(np.linalg.norm(d[:2])); pose[k].append(np.abs(d[2:15]).mean()); vel[k].append(np.abs(d[15:29]).mean())
      for (a0, b0) in stall_segments(q['speed']):
        if b0 < n:
          real = np.linalg.norm(S[b0, :2] - S[a0, :2]); mod = np.linalg.norm(pred[b0, :2] - pred[a0, :2])
          stall.append({'len': b0 - a0, 'real_disp': float(real), 'model_disp': float(mod), 'group': q['group']})
    row = {'n_seq': len(seqs), 'xy_err': {k: {'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)), 'n': len(v)} for k, v in errs.items() if v},
           'pose_err_median': {k: float(np.median(v)) for k, v in pose.items() if v}, 'vel_err_median': {k: float(np.median(v)) for k, v in vel.items() if v}}
    if stall:
      rd = np.array([x['real_disp'] for x in stall]); md = np.array([x['model_disp'] for x in stall])
      row['stall_segments'] = {'n': len(stall), 'len_median': float(np.median([x['len'] for x in stall])), 'real_disp_median': float(np.median(rd)), 'model_disp_median': float(np.median(md)),
                               'share_model_disp_gt_0.5': float((md > 0.5).mean()), 'share_model_disp_gt_2.0': float((md > 2.0).mean()), 'share_real_disp_gt_0.5': float((rd > 0.5).mean())}
    res[name] = row
  return res


def layer2b(models, seqs, pre=20, cf_mode=None):
  """Restart at the real stall-segment entry (offset 0) or `pre` rows before it; (a) the recorded actions fed through the segment, and
  (b) the SAME CF actor choosing the action from the model's state for the same number of steps (user's plan after 20b9401): separates
  'the model cannot propagate even fixed actions' from 'the actor's feedback on the predicted state amplifies the deviation'."""
  res = {}
  for name, M in models.items():
    rows = {0: [], pre: []}
    for q in seqs:
      S, A = q['s'], q['a']; goal = q['goal'].astype(np.float32)
      for (a0, b0) in stall_segments(q['speed']):
        for off in (0, pre):
          st = a0 - off
          if st < 0:
            continue
          s = S[st:st + 1].copy(); gate_fires = 0; pred = [s[0]]
          for j in range(st, b0):
            s, pst = M[q['fold']].step(s, A[j:j + 1]); pred.append(s[0]); gate_fires += int(pst[0] > 0.5)
          pred = np.stack(pred); k0 = a0 - st
          real = float(np.linalg.norm(S[b0, :2] - S[a0, :2])); mod = float(np.linalg.norm(pred[-1, :2] - pred[k0, :2])); entry_err = float(np.linalg.norm(pred[k0, :2] - S[a0, :2]))
          row = {'len': b0 - a0, 'real_disp': real, 'model_disp': mod, 'entry_xy_err': entry_err, 'gate_share': gate_fires / max(b0 - st, 1), 'real_speed_median': float(np.median(q['speed'][a0:b0]))}
          if cf_mode is not None:                                            # (b) closed loop from the same start state
            s = S[st:st + 1].copy(); pred_cl = [s[0]]; adev = []
            for j in range(st, b0):
              a = cf_mode(np.concatenate([s, goal[None]], axis=1)); adev.append(float(np.abs(a[0] - A[j]).mean())); s, _ = M[q['fold']].step(s, a); pred_cl.append(s[0])
            pred_cl = np.stack(pred_cl)
            row['model_disp_closed'] = float(np.linalg.norm(pred_cl[-1, :2] - pred_cl[k0, :2])); row['action_dev_closed'] = float(np.mean(adev)); row['entry_xy_err_closed'] = float(np.linalg.norm(pred_cl[k0, :2] - S[a0, :2]))
          rows[off].append(row)
    out = {}
    for off, rr in rows.items():
      if not rr:
        continue
      md = np.array([x['model_disp'] for x in rr]); rd = np.array([x['real_disp'] for x in rr])
      e = {'n': len(rr), 'len_median': float(np.median([x['len'] for x in rr])), 'real_disp_median': float(np.median(rd)), 'model_disp_median': float(np.median(md)), 'model_disp_p90': float(np.percentile(md, 90)),
           'share_model_disp_gt_0.5': float((md > 0.5).mean()), 'share_model_disp_lt_0.2': float((md < 0.2).mean()), 'entry_xy_err_median': float(np.median([x['entry_xy_err'] for x in rr])),
           'gate_share_median': float(np.median([x['gate_share'] for x in rr])), 'real_speed_median': float(np.median([x['real_speed_median'] for x in rr]))}
      if 'model_disp_closed' in rr[0]:
        mc = np.array([x['model_disp_closed'] for x in rr])
        e.update({'closed_model_disp_median': float(np.median(mc)), 'closed_model_disp_p90': float(np.percentile(mc, 90)), 'closed_share_gt_0.5': float((mc > 0.5).mean()), 'closed_share_lt_0.2': float((mc < 0.2).mean()),
                  'closed_entry_xy_err_median': float(np.median([x['entry_xy_err_closed'] for x in rr])), 'closed_action_dev_mean': float(np.mean([x['action_dev_closed'] for x in rr])),
                  'share_closed_gt_recorded_by_0.3': float(((mc - md) > 0.3).mean())})
      out[f'restart_{off}_before_entry'] = e
    res[name] = out
  return res


def layer3(models, seqs, cf_mode):
  """Closed loop with the CF actor on the model state; compared with the real rollout where both hazards were inactive (deterministic real path)."""
  res = {}
  seqs = [q for q in seqs if not q['u1'] and not q['u2']]
  for name, M in models.items():
    heads, fars, reach, stallp = [], [], [], []
    for q in seqs:
      S = q['s']; goal = q['goal'].astype(np.float32); s = S[0:1].copy(); xy = [s[0, :2]]
      for j in range(400):
        o31 = np.concatenate([s, goal[None]], axis=1); a = cf_mode(o31); s, _ = M[q['fold']].step(s, a); xy.append(s[0, :2])
        if np.linalg.norm(s[0, :2] - goal) <= RD.REACH_R:
          break
      xy = np.stack(xy); real = S[:401, :2]                                     # both judged inside the first 400 steps
      if len(real) > 30 and len(xy) > 30:
        heads.append((float(real[30, 1] > 1.5), float(xy[30, 1] > 1.5)))
      rf = np.isin(region_of(real[:101].astype(np.float64)), FAR).any(); mf = np.isin(region_of(xy[:101].astype(np.float64)), FAR).any(); fars.append((float(rf), float(mf)))
      rr = bool(np.linalg.norm(real - goal, axis=1).min() <= RD.REACH_R); mr = bool(np.linalg.norm(xy - goal, axis=1).min() <= RD.REACH_R); reach.append((float(rr), float(mr)))
      rs = float(np.linalg.norm(real[-1] - real[max(0, len(real) - 100)]) < 0.5) if len(real) >= 100 and not rr else 0.0; ms = float(np.linalg.norm(xy[-1] - xy[max(0, len(xy) - 100)]) < 0.5) if len(xy) >= 100 and not mr else 0.0
      stallp.append((rs, ms))
    H = np.array(heads); Fa = np.array(fars); Rc = np.array(reach); St = np.array(stallp)
    res[name] = {'n_seq': len(seqs), 'heading_north_30': {'real': float(H[:, 0].mean()), 'model': float(H[:, 1].mean()), 'agreement': float((H[:, 0] == H[:, 1]).mean())} if len(H) else None,
                 'far_entry_100': {'real': float(Fa[:, 0].mean()), 'model': float(Fa[:, 1].mean()), 'agreement': float((Fa[:, 0] == Fa[:, 1]).mean())},
                 'reach_by_400': {'real': float(Rc[:, 0].mean()), 'model': float(Rc[:, 1].mean()), 'agreement': float((Rc[:, 0] == Rc[:, 1]).mean())},
                 'stalled_last_100': {'real': float(St[:, 0].mean()), 'model': float(St[:, 1].mean()), 'agreement': float((St[:, 0] == St[:, 1]).mean())}}
  return res


def run(args):
  OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
  R = load_rows(); R['_fold'] = fold_of_rollouts(R)
  test_rollouts = np.flatnonzero(R['roll_split'].astype(str) == 'test'); test_rows = np.flatnonzero(np.isin(R['rollout'], test_rollouts))
  dirs = {'v4': MP.OUT / 'ett_one_step_v4', 'v4c': MP.OUT / 'ett_one_step_v4c', 'v4r': MP.OUT / 'ett_one_step_v4r'} if not MODEL_DIRS else {k: MP.OUT / v for k, v in MODEL_DIRS.items()}
  dirs = {k: v for k, v in dirs.items() if all((v / f'model_fold{f}.pkl').exists() for f in range(3))}
  models = {name: {f: Motion(d, f) for f in range(3)} for name, d in dirs.items()}
  print('models', list(models), 'test rollouts', len(test_rollouts), 'rows', len(test_rows), flush=True)
  res = {'models': {k: str(v) for k, v in dirs.items()}, 'test': {'rollouts': int(len(test_rollouts)), 'rows': int(len(test_rows))}}
  res['L1_teacher_forced'] = layer1(models, R, test_rows); print(f'L1 {time.time() - t0:.0f}s', flush=True)
  seqs = sequences(R, test_rollouts)
  res['L2_open_loop'] = layer2(models, seqs); print(f'L2 {time.time() - t0:.0f}s', flush=True)
  cf_mode = RD_cf_mode()
  res['L2b_stall_entry'] = layer2b(models, seqs, cf_mode=cf_mode); print(f'L2b {time.time() - t0:.0f}s', flush=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); ep = anchors.episode[R['start_anchor']]; sp = R['start_split'].astype(str)
  res['split_episode_overlap'] = {'train_episodes': int(len(set(ep[sp == 'train']))), 'test_episodes': int(len(set(ep[sp == 'test']))), 'test_episodes_also_in_train': int(len(set(ep[sp == 'test']) & set(ep[sp == 'train']))),
                                  'val_episodes_also_in_train': int(len(set(ep[sp == 'val']) & set(ep[sp == 'train']))), 'note': 'the split is by start anchor; anchors of one source episode can fall into different splits (their CF rollouts are different trajectories from different states)'}
  res['L3_closed_loop_no_hazard_seqs'] = layer3(models, seqs, cf_mode); print(f'L3 {time.time() - t0:.0f}s', flush=True)
  # the original validation (no-regression) from the fit jsons
  res['original_validation_at_selected_step'] = {}
  for name, d in dirs.items():
    per = {}
    for f in range(3):
      j = MP.read_json(d / f'model_fold{f}.json'); h = next((x for x in j['history'] if x['step'] == j['best_step']), None)
      per[f'fold{f}'] = {k: h.get(k) for k in ('step', 'val_mse_diag', 'val_mse_off', 'val_onset_bce', 'val_auroc', 'val_stat_acc', 'cf_val_mse_moving', 'cf_val_mse_all', 'cf_val_stat_acc')} if h else None
    res['original_validation_at_selected_step'][name] = per
  res['rows_file'] = ROWS_FILE
  MP.write_json(OUT / 'report.json', res)
  (OUT / 'REPORT.md').write_text(write_md(res), encoding='utf-8')
  print(f'done {time.time() - t0:.0f}s', flush=True)


def RD_cf_mode():
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, OUT / '_cfg'); MP.fill_dims(cfg); nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(RD.ckpt('CF')); pp = st.policy_params
  f = jax.jit(lambda o: jnp.tanh(nets.policy_network.apply(pp, o).loc))
  return lambda o31: np.asarray(f(jnp.asarray(o31, jnp.float32)), np.float32)


def write_md(res):
  names = list(res['models'])
  L = ['# The three-layer acceptance of the CF-motion revision (test split of the CF-controlled rollouts; no training)', '', f"Models {res['models']}; test rollouts {res['test']['rollouts']}, rows {res['test']['rows']}.", '']
  L += ['## L1 -- teacher forcing (real state every step): one-step xy error (median / p90) by row type and phase; the stationary gate; slow rows', '', '| model | moving | slow | static | turn30 | far | corridor | zone | gate fires on static / slow / moving | slow rows: real speed / pred speed / pred frozen share | static rows pred speed |', '|---|---|---|---|---|---|---|---|---|---|---|']
  for n in names:
    r = res['L1_teacher_forced'][n]
    c = lambda k: (f"{r[k]['median']:.4f} / {r[k]['p90']:.4f} (n {r[k]['n']})" if k in r else '-')
    L.append(f"| {n} | {c('xy_err_moving')} | {c('xy_err_slow')} | {c('xy_err_static')} | {c('xy_err_turn30')} | {c('xy_err_far')} | {c('xy_err_corridor')} | {c('xy_err_zone')} | {r['gate']['fires_on_static']:.3f} / {r['gate']['fires_on_slow']:.3f} / {r['gate']['fires_on_moving']:.3f} | {r['slow_rows_speed']['real_median']:.4f} / {r['slow_rows_speed']['pred_median']:.4f} / {r['slow_rows_speed']['pred_frozen_share']:.2f} | {r['static_rows_pred_speed_median']:.4f} |")
  L += ['', '## L2 -- open loop (real action sequence, model rolled forward; onset off): xy error median / p90 at steps; stall segments', '', '| model | ' + ' | '.join(f'step {k}' for k in STEPS_L2) + ' | pose err median at 30 / 100 | vel err median at 30 / 100 | stall segments (n, len): real disp / model disp median; share model disp > 0.5 / > 2.0 (real > 0.5) |', '|---|' + '---|' * len(STEPS_L2) + '---|---|---|']
  for n in names:
    r = res['L2_open_loop'][n]; st = r.get('stall_segments')
    cells = [(f"{r['xy_err'][str(k)]['median']:.3f} / {r['xy_err'][str(k)]['p90']:.3f}" if str(k) in r['xy_err'] else (f"{r['xy_err'][k]['median']:.3f} / {r['xy_err'][k]['p90']:.3f}" if k in r['xy_err'] else '-')) for k in STEPS_L2]
    pe = r['pose_err_median']; ve = r['vel_err_median']; g = lambda d, k: d.get(str(k), d.get(k))
    L.append(f"| {n} | " + ' | '.join(cells) + f" | {g(pe, 30):.3f} / {g(pe, 100) if g(pe, 100) is not None else float('nan'):.3f} | {g(ve, 30):.3f} / {g(ve, 100) if g(ve, 100) is not None else float('nan'):.3f} | " + (f"({st['n']}, {st['len_median']:.0f}): {st['real_disp_median']:.2f} / {st['model_disp_median']:.2f}; {st['share_model_disp_gt_0.5']:.2f} / {st['share_model_disp_gt_2.0']:.2f} ({st['share_real_disp_gt_0.5']:.2f})" if st else '-') + ' |')
  L += ['', '## L2b -- restart at the real stall-segment entry (and 20 rows before it): (a) recorded actions vs (b) the CF actor on the model state, same start, same number of steps', '', '| model | restart | n | seg len | real disp | (a) model disp median / p90; share > 0.5 / < 0.2 | (b) closed-loop disp median / p90; share > 0.5 / < 0.2 | share (b) - (a) > 0.3 | closed-loop mean action dev | entry xy err (a) / (b) | gate |', '|---|---|---:|---:|---:|---|---|---:|---:|---|---:|']
  for n in names:
    for k, e in res['L2b_stall_entry'][n].items():
      cl = (f"{e['closed_model_disp_median']:.3f} / {e['closed_model_disp_p90']:.3f}; {e['closed_share_gt_0.5']:.2f} / {e['closed_share_lt_0.2']:.2f}" if 'closed_model_disp_median' in e else '-')
      L.append(f"| {n} | {k} | {e['n']} | {e['len_median']:.0f} | {e['real_disp_median']:.3f} | {e['model_disp_median']:.3f} / {e['model_disp_p90']:.3f}; {e['share_model_disp_gt_0.5']:.2f} / {e['share_model_disp_lt_0.2']:.2f} | {cl} | {e.get('share_closed_gt_recorded_by_0.3', float('nan')):.2f} | {e.get('closed_action_dev_mean', float('nan')):.3f} | {e['entry_xy_err_median']:.3f} / {e.get('closed_entry_xy_err_median', float('nan')):.3f} | {e['gate_share_median']:.2f} |")
  L += ['', f"Split / episode overlap: {json.dumps(res['split_episode_overlap'])}", '']
  L += ['', '## L3 -- closed loop (CF actor on the model state; onset off) vs the real hazard-free rollouts, both inside the first 400 steps', '', '| model | n seq | heading north at 30: real / model / agreement | far entry by 100: real / model / agreement | reach by 400: real / model / agreement | stalled in the last 100: real / model / agreement |', '|---|---:|---|---|---|---|']
  for n in names:
    r = res['L3_closed_loop_no_hazard_seqs'][n]; h = r['heading_north_30']
    c = lambda d: f"{d['real']:.2f} / {d['model']:.2f} / {d['agreement']:.2f}"
    L.append(f"| {n} | {r['n_seq']} | {c(h) if h else '-'} | {c(r['far_entry_100'])} | {c(r['reach_by_400'])} | {c(r['stalled_last_100'])} |")
  L += ['', '## The original validation at the selected step (no-regression on the start-agent / logged rows) and the CF val', '', '| model | fold | step | val mse diag / off | onset bce / auroc | stat acc | CF val mse moving / all | CF val stat acc |', '|---|---|---:|---|---|---:|---|---:|']
  for n in names:
    for f, h in res['original_validation_at_selected_step'][n].items():
      if h:
        fm = lambda v: '-' if v is None else f'{v:.4f}'
        L.append(f"| {n} | {f} | {h['step']} | {fm(h['val_mse_diag'])} / {fm(h['val_mse_off'])} | {fm(h['val_onset_bce'])} / {fm(h['val_auroc'])} | {fm(h['val_stat_acc'])} | {fm(h.get('cf_val_mse_moving'))} / {fm(h.get('cf_val_mse_all'))} | {fm(h.get('cf_val_stat_acc'))} |")
  L.append('')
  return '\n'.join(L)


def main(argv=None):
  ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('run',)); ap.add_argument('--rows', default='rollouts_cf.npz'); ap.add_argument('--models', nargs='*', default=None, help='name=dir pairs (dirs under the pilot output)'); ap.add_argument('--tag', default=None)
  args = ap.parse_args(argv)
  global ROWS_FILE, MODEL_DIRS, OUT
  ROWS_FILE = args.rows
  if args.models:
    MODEL_DIRS = dict(m.split('=') for m in args.models)
  if args.tag:
    OUT = CFM / f'accept_{args.tag}'
  run(args); return 0


if __name__ == '__main__':
  sys.exit(main())
