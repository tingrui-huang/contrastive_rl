"""Rockfall Stage-A pilot audit: gates R1-R9 (adapted from the litter A1-A9
scheme). Read-only: recomputes checksums, replays probes, never modifies any
frozen artifact or the dataset.

  R1 frozen integrity      walker/base sha256 + rockfall freeze manifest
  R2 contract & loader     npz keys/shapes/finiteness + real offline loader
  R3 hiddenness            structural (no privileged keys) + paired-reset
  R4 boundaries            lengths / padding / dead truncation semantics
  R5 mask stream           per-site activation, pairwise independence,
                           mode ~ mask independence (mixture vs env rng)
  R6 route compliance      sighted == teacher rule, blind alternation,
                           coverage == center; trajectory follows command
  R7 outcome sanity        per-mode success inside loose pilot bands
  R8 pre-drop probe        mask bits unpredictable from obs WITHIN the
                           U-independent coverage behaviour (episode-grouped
                           logistic probe; sighted route choice is the
                           INTENDED U->A pathway and is not tested here)
  R9 support               successes per route, crash/impair examples

Run:  python scripts/audit_rockfall_pilot.py [--dir artifacts/rockfall_dataset/pilot]
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
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from collect_rockfall_pilot import check_rockfall_freeze  # noqa: E402
from audit_litter_pilot import logistic_cv  # noqa: E402

LEARNER_KEYS = {'obs', 'act', 'eval_goals', 'lengths', 'meta'}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dir', default='artifacts/rockfall_dataset/pilot')
  ap.add_argument('--name', default=None)
  ap.add_argument('--pairs', type=int, default=4)
  args = ap.parse_args()
  name = args.name
  if name is None:
    cands = [f[:-4] for f in os.listdir(args.dir)
             if f.endswith('.npz') and not f.endswith('_sidecar.npz')]
    assert len(cands) == 1, cands
    name = cands[0]
  npz_path = os.path.join(args.dir, f'{name}.npz')
  side_path = os.path.join(args.dir, f'{name}_sidecar.npz')
  man = json.load(open(os.path.join(args.dir, 'pilot_manifest.json')))
  d = np.load(npz_path, allow_pickle=True)
  s = np.load(side_path, allow_pickle=True)
  n = int(d['lengths'].shape[0])
  gates, notes = {}, {}

  # ---- R1 frozen integrity -------------------------------------------------
  hard_ok, disc, info = C.check_frozen_integrity()
  rf_ok, rf_diffs, _ = check_rockfall_freeze()
  sha_ok = (C.sha256_file(npz_path) == man['npz_sha256']
            and C.sha256_file(side_path) == man['sidecar_sha256'])
  gates['R1_frozen_integrity'] = bool(hard_ok and rf_ok and sha_ok)
  notes['R1'] = {'litter_hard_ok': hard_ok, 'rockfall_ok': rf_ok,
                 'sha_match': sha_ok,
                 'discrepancies': disc + rf_diffs}

  # ---- R2 contract & loader ------------------------------------------------
  keys_ok = set(d.files) == LEARNER_KEYS
  lengths = d['lengths']
  obs, act = d['obs'], d['act']
  shape_ok = (obs.shape == (n, 701, 58) and act.shape == (n, 701, 8)
              and obs.dtype == np.float32)
  finite_ok = all(np.isfinite(obs[e, :lengths[e]]).all()
                  and np.isfinite(act[e, :max(lengths[e] - 1, 0)]).all()
                  for e in range(n))
  from verify_offline_d4rl import build_offline_cfg
  cfg = build_offline_cfg()
  cfg.obs_dim, cfg.goal_dim, cfg.action_dim = 29, 29, 8
  cfg.start_index, cfg.end_index = 0, -1
  cfg.goal_indices = tuple(range(29))
  cfg.max_episode_steps = 700
  cfg.use_image_obs = False
  buf, _ = OA.build_offline_buffer(npz_path, cfg)
  buf.freeze()
  loader_ok = len(buf) == int((lengths - 1).sum()) == \
      man['n_transitions_total']
  gates['R2_contract_loader'] = bool(keys_ok and shape_ok and finite_ok
                                     and loader_ok)
  notes['R2'] = {'keys': sorted(d.files), 'loader_transitions': len(buf),
                 'manifest_transitions': man['n_transitions_total']}

  # ---- R3 hiddenness ---------------------------------------------------------
  priv_leak = LEARNER_KEYS ^ set(d.files)
  cfg2 = build_offline_cfg()
  cfg2.offline_dataset = ''
  cfg2.eval_goal_mode = 'd4rl'
  penv = envs_mod.make_env('offline_ant_umaze_rockfall', cfg2,
                           seed=man['env_seed'] + 9)
  _, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  pairs = RP.paired_hiddenness(penv, walker, base_act, args.pairs)
  pair_ok = all(p['ok'] for p in pairs)
  gates['R3_hiddenness'] = bool(not priv_leak and pair_ok)
  notes['R3'] = {'extra_keys': sorted(priv_leak), 'pairs': pairs}

  # ---- R4 boundaries ----------------------------------------------------------
  pad_ok = all(not obs[e, lengths[e]:].any() for e in range(n))
  dead = s['dead']
  cstep = s['collapse_step']
  trunc_ok = all((not dead[e]) or lengths[e] == min(cstep[e] + 2, 701)
                 for e in range(n))
  len_ok = bool((lengths >= 2).all() and (lengths <= 701).all())
  gates['R4_boundaries'] = bool(pad_ok and trunc_ok and len_ok)
  notes['R4'] = {'pad_ok': pad_ok, 'trunc_ok': trunc_ok,
                 'dead_eps': int(dead.sum())}

  # ---- R5 mask stream ---------------------------------------------------------
  m = s['rockfall_mask'].astype(int)          # (n, 4)
  freq = m.mean(0)
  freq_ok = bool(np.all((freq >= 0.13) & (freq <= 0.27)))
  corr = np.corrcoef(m.T)
  off = corr[~np.eye(4, dtype=bool)]
  corr_ok = bool(np.all(np.abs(off) <= 0.15))
  tm = s['teacher_mode']
  mode_dep = []
  for mode in ('sighted', 'coverage'):
    sel = tm == mode
    if sel.sum() >= 20:
      mode_dep.append(float(np.abs(m[sel].mean(0) - freq).max()))
  mode_ok = bool(all(x <= 0.12 for x in mode_dep))
  gates['R5_mask_stream'] = bool(freq_ok and corr_ok and mode_ok)
  notes['R5'] = {'site_freq': [round(float(f), 3) for f in freq],
                 'max_abs_pairwise_corr': round(float(np.abs(off).max()), 3),
                 'mode_vs_mask_maxdev': [round(x, 3) for x in mode_dep]}

  # ---- R6 route compliance ------------------------------------------------------
  route = s['route']
  sight = tm == 'sighted'
  ok_rule = []
  for e in np.where(sight)[0]:
    mask = tuple(int(b) for b in m[e])
    la = mask[0] or mask[1]
    ra = mask[2] or mask[3]
    if la and ra:
      ok_rule.append(route[e] == 'center')
    elif la:
      ok_rule.append(route[e] == 'right')
    elif ra:
      ok_rule.append(route[e] == 'left')
    else:
      ok_rule.append(route[e] in ('left', 'right'))
  rule_ok = bool(all(ok_rule))
  blind_routes = route[tm == 'blind']
  blind_ok = bool(all(r == ('left' if i % 2 == 0 else 'right')
                      for i, r in enumerate(blind_routes)))
  cover_ok = bool(all(r == 'center' for r in route[tm == 'coverage']))
  # trajectory follows the commanded route (zone mean-y classification)
  ty, tx, hf = s['step_torso_y'], s['step_torso_x'], s['step_handoff']
  follow = []
  for e in range(n):
    zone = ((tx[e] >= 2.3) & (tx[e] <= 5.7) & (hf[e] == 0)
            & np.isfinite(ty[e]))
    if zone.sum() < 5:
      continue
    my = float(np.nanmean(ty[e][zone]))
    obs_route = 'left' if my > 0.5 else 'right' if my < -0.5 else 'center'
    follow.append(obs_route == route[e])
  follow_rate = float(np.mean(follow))
  gates['R6_route_compliance'] = bool(rule_ok and blind_ok and cover_ok
                                      and follow_rate >= 0.90)
  notes['R6'] = {'sighted_rule_ok': rule_ok, 'blind_alternation': blind_ok,
                 'coverage_center': cover_ok,
                 'trajectory_follow_rate': round(follow_rate, 3),
                 'both_clear_left_frac': round(float(np.mean(
                     [route[e] == 'left' for e in np.where(sight)[0]
                      if not (m[e, 0] or m[e, 1] or m[e, 2] or m[e, 3])]
                 )), 3)}

  # ---- R7 outcome sanity ----------------------------------------------------------
  succ = s['success']
  by_mode = {mode: float(succ[tm == mode].mean())
             for mode in ('sighted', 'blind', 'coverage')}
  bands = {'sighted': (0.75, 0.97), 'blind': (0.20, 0.90),
           'coverage': (0.55, 0.95)}
  band_ok = all(bands[k][0] <= v <= bands[k][1] for k, v in by_mode.items())
  gates['R7_outcomes'] = bool(band_ok)
  notes['R7'] = {'success_by_mode': {k: round(v, 3)
                                     for k, v in by_mode.items()},
                 'bands': bands,
                 'overall': round(float(succ.mean()), 3),
                 'dead_frac': round(float(dead.mean()), 3)}

  # ---- R8 pre-drop hiddenness probe (coverage-only) --------------------------------
  # Coverage is U-independent by construction; its obs stream before any
  # drop must carry no mask information. (Sighted route choice IS the
  # intended U->A confounding pathway -- excluded by design.)
  cov = np.where(tm == 'coverage')[0]
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
  probe = {}
  probe_ok = True
  for b in range(4):
    y = m[groups, b].astype(float)
    base = max(y.mean(), 1 - y.mean())
    acc, auc = logistic_cv(X, y, groups)
    margin = float(acc - base)
    probe[f'site{b}'] = {'acc': round(float(acc), 3),
                         'base': round(float(base), 3),
                         'auc': round(float(auc), 3),
                         'margin': round(margin, 3)}
    if margin > 0.07:
      probe_ok = False
  gates['R8_predrop_probe'] = bool(probe_ok)
  notes['R8'] = {'coverage_eps': int(len(cov)), 'states': int(len(X)),
                 'per_site': probe}

  # ---- R9 support -------------------------------------------------------------------
  succ_route = {r: int(((route == r) & (succ > 0)).sum())
                for r in ('left', 'right', 'center')}
  n_collapse = int(dead.sum())
  n_impaired = int(s['impaired'].sum())
  n_hit = int((s['hit'].any(axis=1)).sum())
  gates['R9_support'] = bool(succ_route['left'] >= 15
                             and succ_route['right'] >= 15
                             and succ_route['center'] >= 15
                             and n_collapse >= 3 and n_impaired >= 2)
  notes['R9'] = {'successes_by_route': succ_route,
                 'collapse_eps': n_collapse, 'impaired_eps': n_impaired,
                 'rock_hit_eps': n_hit}

  # ---- report ----
  all_pass = all(gates.values())
  report = {'dataset': npz_path, 'n_episodes': n, 'gates': gates,
            'notes': notes, 'all_pass': all_pass,
            'git_commit': C.git_commit()}
  out = os.path.join(args.dir, 'audit_report.json')
  json.dump(report, open(out, 'w'), indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
  for k, v in gates.items():
    print(f'{"PASS" if v else "FAIL"}  {k}')
  print('AUDIT ' + ('ALL PASS' if all_pass else 'FAILED'), '->', out)
  return 0 if all_pass else 1


if __name__ == '__main__':
  sys.exit(main())
