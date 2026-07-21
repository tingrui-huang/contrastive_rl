"""Rockfall v2 (LOCAL-DETOUR) PRIMARY pilot audit. Read-only.

Gates tuned to the 90/0/10 local-detour spec (no blind in the primary set;
no severe/impaired/mild impact-coverage requirement -- the local-detour
teacher is designed to AVOID active sites, so hits are rare by design and
impact coverage is an ENVIRONMENT qualification, not a dataset gate).

  V1 frozen integrity      walker/base sha + rockfall freeze + npz sha
  V2 contract & loader     learner keys/shapes/finite + offline loader
  V3 hiddenness            no privileged keys + paired-reset identity
  V4 boundaries            lengths / padding / dead truncation
  V5 mask stream           4-site activation ~0.2 + pairwise independence
  V6 mixture 90/0/10       exact 270 sighted / 0 blind / 30 center
  V7 base side             balanced within sighted + independent of mask
  V8 detour behaviour      active inward dip vs inactive straight (selective)
                           + return to base lane after the window
  V9 center compliance     coverage episodes stay on the center route
  V10 pre-drop probe       mask unpredictable from coverage obs (chance)

Usage: python scripts/audit_rockfall_v2_pilot.py
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import offline_audit as OA       # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from collect_rockfall_v2_pilot import check_rockfall_freeze  # noqa: E402
from audit_litter_pilot import logistic_cv  # noqa: E402

LEARNER_KEYS = {'obs', 'act', 'eval_goals', 'lengths', 'meta'}
BIT = V2.BIT


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dir', default='artifacts/rockfall_v2_dataset/pilot')
  ap.add_argument('--name', default='antmaze_rockfall_v2_pilot')
  ap.add_argument('--pairs', type=int, default=6)
  args = ap.parse_args()
  npz_path = os.path.join(args.dir, f'{args.name}.npz')
  side_path = os.path.join(args.dir, f'{args.name}_sidecar.npz')
  man = json.load(open(os.path.join(args.dir, 'pilot_manifest.json')))
  d = np.load(npz_path, allow_pickle=True)
  s = np.load(side_path, allow_pickle=True)
  n = int(d['lengths'].shape[0])
  gates, notes = {}, {}

  # ---- V1 integrity ----
  hard_ok, disc, info = C.check_frozen_integrity()
  rf_ok, rf_diffs, _ = check_rockfall_freeze()
  sha_ok = (C.sha256_file(npz_path) == man['npz_sha256']
            and C.sha256_file(side_path) == man['sidecar_sha256'])
  gates['V1_frozen_integrity'] = bool(hard_ok and rf_ok and sha_ok)
  notes['V1'] = {'litter': hard_ok, 'rockfall': rf_ok, 'sha': sha_ok,
                 'discrepancies': disc + rf_diffs}

  # ---- V2 contract & loader ----
  lengths, obs, act = d['lengths'], d['obs'], d['act']
  keys_ok = set(d.files) == LEARNER_KEYS
  shape_ok = (obs.shape == (n, 701, 58) and act.shape == (n, 701, 8)
              and obs.dtype == np.float32)
  finite_ok = all(np.isfinite(obs[e, :lengths[e]]).all() for e in range(n))
  from verify_offline_d4rl import build_offline_cfg
  cfg = build_offline_cfg()
  cfg.obs_dim, cfg.goal_dim, cfg.action_dim = 29, 29, 8
  cfg.start_index, cfg.end_index = 0, -1
  cfg.goal_indices = tuple(range(29))
  cfg.max_episode_steps = 700
  cfg.use_image_obs = False
  buf, _ = OA.build_offline_buffer(npz_path, cfg)
  buf.freeze()
  loader_ok = len(buf) == int((lengths - 1).sum()) == man['n_transitions_total']
  gates['V2_contract_loader'] = bool(keys_ok and shape_ok and finite_ok
                                     and loader_ok)
  notes['V2'] = {'keys': sorted(d.files), 'transitions': len(buf)}

  # ---- V3 hiddenness ----
  priv_leak = LEARNER_KEYS ^ set(d.files)
  cfg2 = build_offline_cfg()
  cfg2.offline_dataset = ''
  cfg2.eval_goal_mode = 'd4rl'
  penv = envs_mod.make_env('offline_ant_umaze_rockfall', cfg2,
                           seed=man['env_seed'] + 9)
  _, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  pairs = RP.paired_hiddenness(penv, walker, base_act, args.pairs)
  gates['V3_hiddenness'] = bool(not priv_leak and all(p['ok'] for p in pairs))
  notes['V3'] = {'extra_keys': sorted(priv_leak), 'pairs': pairs}

  # ---- V4 boundaries ----
  dead = s['dead']
  cstep = s['collapse_step']
  pad_ok = all(not obs[e, lengths[e]:].any() for e in range(n))
  trunc_ok = all((not dead[e]) or lengths[e] == min(cstep[e] + 2, 701)
                 for e in range(n))
  gates['V4_boundaries'] = bool(pad_ok and trunc_ok
                                and (lengths >= 2).all())
  notes['V4'] = {'pad_ok': pad_ok, 'trunc_ok': trunc_ok,
                 'dead_eps': int(dead.sum())}

  # ---- V5 mask stream ----
  m = s['rockfall_mask'].astype(int)
  freq = m.mean(0)
  corr = np.corrcoef(m.T)
  off = corr[~np.eye(4, dtype=bool)]
  gates['V5_mask_stream'] = bool(np.all((freq >= 0.13) & (freq <= 0.27))
                                 and np.all(np.abs(off) <= 0.15))
  notes['V5'] = {'site_freq': [round(float(f), 3) for f in freq],
                 'max_abs_pairwise_corr': round(float(np.abs(off).max()), 3)}

  # ---- V6 mixture 90/0/10 ----
  tm = s['teacher_mode']
  counts = {mode: int((tm == mode).sum())
            for mode in ('sighted', 'blind', 'coverage')}
  n_sight = int(round(0.90 * n))
  n_cover = int(round(0.10 * n))
  gates['V6_mixture_90_0_10'] = bool(counts == {'sighted': n_sight,
                                                'blind': 0,
                                                'coverage': n_cover})
  notes['V6'] = {'counts': counts,
                 'expected': {'sighted': n_sight, 'blind': 0,
                              'coverage': n_cover}}

  # ---- V7 base side balance + mask independence (within sighted) ----
  bs = s['base_side']
  sight = tm == 'sighted'
  bs_left = (bs[sight] == 'left').astype(int)
  left_active = np.array([1 if (m[e, 0] or m[e, 1]) else 0
                          for e in np.where(sight)[0]])
  frac_left = float(bs_left.mean())
  corr_sm = float(np.corrcoef(bs_left, left_active)[0, 1]) if bs_left.std() \
      else 0.0
  gates['V7_base_side'] = bool(0.42 <= frac_left <= 0.58
                               and abs(corr_sm) < 0.12)
  notes['V7'] = {'frac_base_left': round(frac_left, 3),
                 'corr_base_vs_leftactive': round(corr_sm, 3)}

  # ---- V8 detour behaviour (from sidecar trajectories) ----
  tx, ty, hf = s['step_torso_x'], s['step_torso_y'], s['step_handoff']
  det = {'active': [], 'inactive': [], 'recover': []}
  trig = {'active_total': 0, 'active_triggered': 0}
  for e in np.where(sight)[0]:
    base = bs[e]
    sgn = 1.0 if base == 'left' else -1.0
    x, y, h = tx[e], ty[e], hf[e]
    base_sites = [(nm, sx) for nm, sx, sg in RA.ROCKFALL_SITES if sg == sgn]
    active_x = [sx for nm, sx in base_sites if m[e, BIT[nm]]]
    for nm, sx in base_sites:
      w = (x >= sx - V2.DETOUR_PRE) & (x <= sx + V2.DETOUR_POST) & (h == 0)
      if w.sum() < 2:
        continue
      mn = float(np.min(np.abs(y[w])))
      if m[e, BIT[nm]]:
        det['active'].append(mn)
        trig['active_total'] += 1
        if s['triggered'][e, BIT[nm]]:
          trig['active_triggered'] += 1
        rlo, rhi = sx + V2.DETOUR_POST + 0.3, sx + V2.DETOUR_POST + 1.1
        if not any(ax != sx and (ax - V2.DETOUR_PRE) < rhi
                   and (ax + V2.DETOUR_POST) > rlo for ax in active_x):
          rw = (x >= rlo) & (x <= rhi) & (h == 0)
          if rw.any():
            det['recover'].append(float(np.mean(np.abs(y[rw]))))
      else:
        det['inactive'].append(mn)
  am = float(np.mean(det['active'])) if det['active'] else None
  im = float(np.mean(det['inactive'])) if det['inactive'] else None
  rm = float(np.mean(det['recover'])) if det['recover'] else None
  trig_rate = trig['active_triggered'] / max(trig['active_total'], 1)
  gates['V8_detour_behaviour'] = bool(
      am is not None and im is not None and rm is not None
      and am < 0.80 and im > 0.82 and (im - am) >= 0.10
      and rm >= am + 0.12 and trig_rate <= 0.15)
  notes['V8'] = {'active_dip_mean': round(am, 3) if am else None,
                 'inactive_mean': round(im, 3) if im else None,
                 'separation': round(im - am, 3) if (am and im) else None,
                 'recover_mean': round(rm, 3) if rm else None,
                 'active_site_trigger_rate': round(trig_rate, 3),
                 'active_total': trig['active_total']}

  # ---- V9 center compliance ----
  cov = np.where(tm == 'coverage')[0]
  center_ok = []
  for e in cov:
    x, y, h = tx[e], ty[e], hf[e]
    zone = (x >= 2.3) & (x <= 5.7) & (h == 0) & np.isfinite(y)
    if zone.sum() >= 5:
      center_ok.append(abs(float(np.nanmean(y[zone]))) < 0.5)
  center_drop = int(sum(bool(s['dropped'][e].any()) for e in cov))
  gates['V9_center_compliance'] = bool(all(center_ok) and center_drop == 0)
  notes['V9'] = {'coverage_eps': int(len(cov)),
                 'center_mean_y_ok': int(sum(center_ok)),
                 'center_drop_eps': center_drop}

  # ---- V10 pre-drop probe (coverage-only) ----
  fdrop = s['first_drop_step']
  X, groups = [], []
  for e in cov:
    tmax = int(lengths[e] - 1 if fdrop[e] < 0 else min(fdrop[e],
                                                       lengths[e] - 1))
    sel = np.arange(1, tmax)
    if len(sel) > 80:
      sel = sel[np.linspace(0, len(sel) - 1, 80).astype(int)]
    X.append(obs[e, sel, :29])
    groups += [e] * len(sel)
  X = np.concatenate(X)
  groups = np.asarray(groups)
  probe, probe_ok = {}, True
  for b in range(4):
    y = m[groups, b].astype(float)
    base = max(y.mean(), 1 - y.mean())
    acc, auc = logistic_cv(X, y, groups)
    probe[f'site{b}'] = {'acc': round(float(acc), 3),
                         'base': round(float(base), 3),
                         'margin': round(float(acc - base), 3)}
    if acc - base > 0.07:
      probe_ok = False
  gates['V10_predrop_probe'] = bool(probe_ok)
  notes['V10'] = {'coverage_eps': int(len(cov)), 'per_site': probe}

  all_pass = all(gates.values())
  report = {'dataset': npz_path, 'variant': 'local_detour_v2_primary_90_0_10',
            'n_episodes': n, 'gates': gates, 'notes': notes,
            'all_pass': all_pass, 'git_commit': C.git_commit(),
            'impact_coverage_note': 'severe/impaired/mild coverage is NOT a '
            'primary-dataset gate (teacher avoids active sites by design); it '
            'is an environment qualification only.'}
  out = os.path.join(args.dir, 'audit_report.json')
  json.dump(report, open(out, 'w'), indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
  for k, v in gates.items():
    print(f'{"PASS" if v else "FAIL"}  {k}')
  print('V2 AUDIT ' + ('ALL PASS' if all_pass else 'FAILED'), '->', out)
  return 0 if all_pass else 1


if __name__ == '__main__':
  sys.exit(main())
