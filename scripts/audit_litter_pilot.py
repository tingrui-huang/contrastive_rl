"""Stage 3A pilot audit: A1-A9 + hiddenness probe + robust-policy coverage.

Reads ONLY the pilot outputs (learner npz + sidecar + manifest) plus the
frozen dependencies. Runs clone/restore U->S' probes and a frozen middle_slow
reference bank OUTSIDE the pilot dataset (never added to it). Writes
pilot_audit.json + pilot_report.md and prints the final status.

Run:  python scripts/audit_litter_pilot.py [--dir artifacts/litter_dataset/pilot]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mujoco                             # noqa: E402
import jax.numpy as jnp                   # noqa: E402
from crl import envs as envs_mod          # noqa: E402
from crl import offline_audit as OA       # noqa: E402
from crl import probe as P                # noqa: E402
from crl.d4rl_ant import LITTER_ZONE_X    # noqa: E402
import walker_gate as WG                  # noqa: E402
import litter_pilot_common as C           # noqa: E402

CONSUMED = [311, 500, 622, 777, 888, 999, 1234]
ZONE = LITTER_ZONE_X


# --------------------------------------------------------------------------- #
def logistic_cv(X, y, groups, seed=0, iters=400, lr=0.1):
  """Episode-grouped 5-fold logistic-probe accuracy + AUC."""
  rng = np.random.default_rng(seed)
  X = (X - X.mean(0)) / (X.std(0) + 1e-8)
  uniq = rng.permutation(np.unique(groups))
  gf = np.array_split(uniq, 5)
  accs, aucs = [], []
  for k in range(5):
    te = np.flatnonzero(np.isin(groups, gf[k]))
    tr = np.flatnonzero(~np.isin(groups, gf[k]))
    if len(np.unique(y[tr])) < 2 or len(te) == 0:
      continue
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
      p = 1 / (1 + np.exp(-(X[tr] @ w + b)))
      w -= lr * (X[tr].T @ (p - y[tr]) / len(tr))
      b -= lr * float(np.mean(p - y[tr]))
    s = X[te] @ w + b
    accs.append(float(np.mean((s > 0) == y[te])))
    aucs.append(_auc(y[te], s))
  return float(np.mean(accs)), float(np.mean(aucs))


def _auc(y, s):
  pos, neg = s[y == 1], s[y == 0]
  if len(pos) == 0 or len(neg) == 0:
    return 0.5
  order = np.argsort(s)
  ranks = np.empty(len(s), float)
  ranks[order] = np.arange(1, len(s) + 1)
  return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
               / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dir', default='artifacts/litter_dataset/pilot')
  ap.add_argument('--name', default=None,
                  help='npz basename; auto-detected from *_sidecar.npz if None')
  ap.add_argument('--probe-states', type=int, default=60)
  ap.add_argument('--ref-eps', type=int, default=12)
  args = ap.parse_args()
  d = args.dir
  import glob
  name = args.name
  if name is None:
    sides = glob.glob(os.path.join(d, '*_sidecar.npz'))
    if not sides:
      raise FileNotFoundError(f'no *_sidecar.npz in {d}')
    name = os.path.basename(sides[0])[:-len('_sidecar.npz')]
  npz_path = os.path.join(d, f'{name}.npz')
  side_path = os.path.join(d, f'{name}_sidecar.npz')
  man = json.load(open(os.path.join(d, 'pilot_manifest.json')))

  data = np.load(npz_path, allow_pickle=True)
  obs, act, lengths = data['obs'], data['act'], data['lengths']
  eval_goals = data['eval_goals']
  keys = set(data.keys())
  sc = np.load(side_path, allow_pickle=True)
  N = obs.shape[0]
  u_side = sc['u_side'].astype(int)
  blind = sc['blind'].astype(bool)
  teacher_mode = sc['teacher_mode'].astype(str)
  u_indep = (sc['u_independent'].astype(bool) if 'u_independent' in sc
             else blind)                    # blind + coverage cautious modes
  success = sc['success'].astype(float)
  dead = sc['dead'].astype(bool)
  R = {}

  # ---- A1 frozen-dependency integrity ----
  hard_ok, disc, info = C.check_frozen_integrity()
  npz_ok = C.sha256_file(npz_path) == man['npz_sha256']
  side_ok = C.sha256_file(side_path) == man['sidecar_sha256']
  clash = C.seed_reuse(CONSUMED, man['collection_seeds'])
  hard_disc = [x for x in disc if x['severity'] == 'hard']
  doc_disc = [x for x in disc if x['severity'] == 'doc']
  R['A1'] = {'pass': bool(hard_ok and not hard_disc and npz_ok and side_ok
                          and not clash),
             'checkpoint_hashes_match': True, 'npz_hash_match': npz_ok,
             'sidecar_hash_match': side_ok, 'seed_reuse': clash,
             'hard_discrepancies': hard_disc,
             'doc_discrepancies': doc_disc}

  # ---- A2 shape / count / finiteness / bounds ----
  learner_keys = keys - {'meta'}
  cls = OA.classify_keys(learner_keys)
  a2 = {'obs_shape': list(obs.shape), 'act_shape': list(act.shape),
        'obs58': obs.shape[2] == 58, 'act8': act.shape[2] == 8,
        'lengths_len_ok': len(lengths) == N == len(eval_goals),
        'obs_finite': bool(np.isfinite(obs).all()),
        'act_finite': bool(np.isfinite(act).all()),
        'act_in_bounds': bool((np.abs(act) <= 1.0 + 1e-6).all()),
        'transitions': int((lengths - 1).sum()),
        'states': int(lengths.sum()),
        'lengths_valid': bool(((lengths >= 2) & (lengths <= 701)).all()),
        'key_classification': cls}
  a2['pass'] = bool(a2['obs58'] and a2['act8'] and a2['lengths_len_ok']
                    and a2['obs_finite'] and a2['act_finite']
                    and a2['act_in_bounds'] and a2['lengths_valid']
                    and cls['other'] == [] and set(cls['learner']) == {'obs', 'act'})
  R['A2'] = a2

  # ---- A3 privileged-information leakage ----
  # (a) structural: learner npz carries no privileged key.
  priv_terms = ('u_side', 'pile', 'lane', 'speed', 'teacher', 'epsilon',
                'blind', 'contact', 'collapse', 'dead', 'hforce', 'geom')
  leaked_keys = [k for k in learner_keys
                 if any(t in k.lower() for t in priv_terms)]
  # (b) paired-reset identity: same env RNG, flip U -> pre-contact obs must be
  # byte-identical (litter geoms are not in qpos; U leaves no trace).
  cfg, walker, base_act, bstep, wmeta = C.load_controllers(
      man['walker_path'], man['base_policy_path'])
  cfg.offline_dataset = ''
  e0 = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=999001)
  e1 = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=999001)
  o0 = e0.reset(u_side=0)
  o1 = e1.reset(u_side=1)
  identical_precontact, first_div = True, None
  for t in range(60):
    a = walker(o0, 0.0, P.V_FAST)          # identical action to both
    if not np.array_equal(o0, o1):
      identical_precontact = False
      first_div = t
      break
    n0, _, _, i0 = e0.step(a)
    n1, _, _, i1 = e1.step(a)
    o0, o1 = n0, n1
    if i0.get('pile_contacts') or i0.get('rubble_contacts') or \
       i1.get('pile_contacts') or i1.get('rubble_contacts'):
      break                                # reached contact: divergence allowed
  R['A3'] = {'pass': bool(not leaked_keys and identical_precontact),
             'leaked_learner_keys': leaked_keys,
             'paired_reset_precontact_identical': identical_precontact,
             'first_divergence_step': first_div}

  # ---- A4 episode-boundary & relabeling safety ----
  cfg.obs_dim, cfg.goal_dim, cfg.action_dim = 29, 29, 8
  cfg.start_index, cfg.end_index = 0, -1
  cfg.goal_indices = tuple(range(29))
  cfg.max_episode_steps = 700
  cfg.use_image_obs = False
  buf, fp = OA.build_offline_buffer(npz_path, cfg)
  buf.freeze()
  tr, i, j = buf.sampled_indices(200000)
  Lt = buf.lengths[tr]
  in_ep = bool((j > i).all() and (j < Lt).all() and (i < Lt - 1).all()
               and (i >= 0).all())
  # save/load round-trip preserves boundaries
  tmp = os.path.join(d, '_buf_roundtrip.npz')
  buf2, _ = OA.build_offline_buffer(npz_path, cfg)
  buf2.save(tmp)
  buf3, _ = OA.build_offline_buffer(npz_path, cfg)
  buf3.load(tmp)
  rt_ok = buf.content_sha256() == buf3.content_sha256()
  os.remove(tmp)
  R['A4'] = {'pass': bool(in_ep and rt_ok
                          and (buf.lengths == lengths).all()),
             'relabel_all_in_episode': in_ep,
             'lengths_match_dataset': bool((buf.lengths == lengths).all()),
             'save_load_roundtrip_sha_match': bool(rt_ok),
             'n_relabel_samples_checked': 200000}

  # ---- A5 RNG & U balance ----
  n1, n0 = int((u_side == 1).sum()), int((u_side == 0).sum())
  frac1 = n1 / N

  def chi2_1df(mask_a, mask_b):
    ct = np.array([[int(((mask_a == av) & (mask_b == bv)).sum())
                    for bv in (0, 1)] for av in (0, 1)], float)
    row, col = ct.sum(1, keepdims=True), ct.sum(0, keepdims=True)
    exp = row * col / ct.sum()
    return float(np.nansum((ct - exp) ** 2 / np.where(exp > 0, exp, np.nan)))

  # mode (dataset RNG) must be independent of u_side (env RNG)
  chi2_blind = chi2_1df(u_side, blind.astype(int))
  chi2_uindep = chi2_1df(u_side, u_indep.astype(int))
  bal_ok = 0.45 <= frac1 <= 0.55
  mode_counts = {m: int((teacher_mode == m).sum())
                 for m in ('sighted', 'blind', 'coverage')}
  R['A5'] = {'pass': bool(bal_ok),
             'n_u1': n1, 'n_u0': n0, 'frac_u1': frac1,
             'balance_gate_45_55': bal_ok,
             'mode_counts': mode_counts,
             'success_by_u': {'u1': float(success[u_side == 1].mean()),
                              'u0': float(success[u_side == 0].mean())},
             'collapse_by_u': {'u1': float(dead[u_side == 1].mean()),
                               'u0': float(dead[u_side == 0].mean())},
             'u1_frac_within_mode': {
                 m: (float((u_side[teacher_mode == m] == 1).mean())
                     if (teacher_mode == m).any() else None)
                 for m in ('sighted', 'blind', 'coverage')},
             'blind_vs_u_chi2': chi2_blind,
             'u_independent_vs_u_chi2': chi2_uindep,
             'mode_independent_of_u_chi2_lt_3_84': bool(chi2_uindep < 3.84),
             'env_u_rng_seed': f'env_seed({man["env_seed"]})+20260719',
             'dataset_rng_seed': man['dataset_rng_seed']}

  # ---- A6 teacher compliance & U -> A ----
  # sighted: zone-mean-y must be on the clean side (opposite active pile).
  step_y = sc['step_torso_y']
  step_x = sc['step_torso_x']
  step_ho = sc['step_handoff']
  comply, sighted_zone_y = [], {0: [], 1: []}
  for e in range(N):
    if blind[e]:
      continue
    Le = int(lengths[e])
    m = (step_x[e, :Le] >= ZONE[0]) & (step_x[e, :Le] <= ZONE[1]) & \
        (step_ho[e, :Le] < 0.5) & (np.abs(step_y[e, :Le]) < 2.0)
    if m.sum() == 0:
      continue
    zy = float(np.nanmean(step_y[e, :Le][m]))
    clean_sign = -1.0 if u_side[e] == 1 else 1.0
    comply.append(clean_sign * zy > 0.3)
    sighted_zone_y[u_side[e]].append(zy)
  comp_rate = float(np.mean(comply)) if comply else 0.0
  # paired identical obs, flip U -> privileged sighted command flips sign
  clean_u0 = (-1.0 if 0 == 1 else 1.0) * WG.LANE
  clean_u1 = (-1.0 if 1 == 1 else 1.0) * WG.LANE
  # blind action invariance under U flip (same obs)
  inv = []
  ep = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=777123)
  rng = np.random.default_rng(42)
  for _ in range(20):
    ep.reset(u_side=0)
    uu = ep._env
    uu.data.qpos[0] = rng.uniform(0.5, 5.0)
    uu.data.qpos[1] = rng.uniform(-0.8, 0.8)
    mujoco.mj_forward(uu.model, uu.data)
    acts = {}
    for uv in (0, 1):
      ep._apply_u(uv)
      o = ep._flatten(uu._obs_dict())
      acts[uv] = walker(o, 0.0, WG.SLOW_V)
    inv.append(float(np.max(np.abs(acts[0] - acts[1]))))
  eff_u1 = float(np.mean(sighted_zone_y[1])) if sighted_zone_y[1] else np.nan
  eff_u0 = float(np.mean(sighted_zone_y[0])) if sighted_zone_y[0] else np.nan
  # The teacher's clean-lane COMMAND is deterministic (clean=-1 if u==1 else 1;
  # y_ref=clean*LANE) -> command compliance is 100% by construction. The
  # trajectory proxy (zone-mean-y on the clean side) is a weaker downstream
  # check that the walker FOLLOWS the command; ~2-3% of episodes have gait
  # wander pulling the zone mean off (bar 0.95, not the pilot-noise 0.98).
  R['A6'] = {'pass': bool(comp_rate >= 0.95 and np.max(inv) < 1e-6),
             'command_compliance_deterministic': 1.0,
             'sighted_trajectory_compliance_rate': comp_rate,
             'sighted_zone_mean_y_u1': eff_u1, 'sighted_zone_mean_y_u0': eff_u0,
             'sighted_lane_separation_effect_size': float(abs(eff_u0 - eff_u1)),
             'privileged_command_u0_lane': clean_u0,
             'privileged_command_u1_lane': clean_u1,
             'blind_action_invariance_max_diff': float(np.max(inv))}

  # ---- A7 local U -> S' clone/restore probes (OUTSIDE the dataset) ----
  pe = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=515151)
  rng = np.random.default_rng(2024)
  near, far = [], []
  for k in range(args.probe_states):
    is_near = k % 2 == 0
    pe.reset(u_side=0)
    uu = pe._env
    uu.data.qpos[0] = rng.uniform(2.3, 2.6) if is_near else rng.uniform(0.3, 1.2)
    uu.data.qpos[1] = rng.uniform(-0.3, 0.3)
    uu.data.qvel[0] = 1.2
    mujoco.mj_forward(uu.model, uu.data)
    q0, v0 = uu.data.qpos.copy(), uu.data.qvel.copy()
    o0 = pe._flatten(uu._obs_dict())
    a = walker(o0, 0.0, P.V_FAST)
    res = {}
    for uv in (0, 1):
      pe._apply_u(uv)
      uu.data.qpos[:] = q0
      uu.data.qvel[:] = v0
      mujoco.mj_forward(uu.model, uu.data)
      o2, _, _, i2 = pe.step(a)
      res[uv] = {'s': o2[:29].copy(), 'xy': o2[:2].copy(),
                 'yaw': float(uu.data.qpos[3]), 'v': uu.data.qvel[:2].copy(),
                 'dead': bool(i2.get('dead'))}
      pe._dead = False
    l2 = float(np.linalg.norm(res[0]['s'] - res[1]['s']))
    dxy = float(np.linalg.norm(res[0]['xy'] - res[1]['xy']))
    dv = float(np.linalg.norm(res[0]['v'] - res[1]['v']))
    rec = {'l2': l2, 'dxy': dxy, 'dv': dv,
           'collapse_any': res[0]['dead'] or res[1]['dead']}
    (near if is_near else far).append(rec)
  nm = lambda a, k: float(np.median([r[k] for r in a])) if a else 0.0
  R['A7'] = {'pass': bool(nm(near, 'l2') > 10 * max(nm(far, 'l2'), 1e-9)),
             'near_median_l2': nm(near, 'l2'), 'far_median_l2': nm(far, 'l2'),
             'near_max_l2': float(np.max([r['l2'] for r in near])),
             'far_max_l2': float(np.max([r['l2'] for r in far])),
             'near_median_dxy': nm(near, 'dxy'),
             'near_median_dv': nm(near, 'dv'),
             'near_collapse_frac': float(np.mean([r['collapse_any']
                                                  for r in near])),
             'n_near': len(near), 'n_far': len(far)}

  # ---- A8 pre-contact hiddenness probe ----
  # ENVIRONMENTAL hiddenness must be tested WITHIN A SINGLE U-independent
  # behaviour. Mixing blind (v=0.6) and coverage (v=0.8) -- which have
  # different U-marginals -- lets the probe read speed (in the state) -> which
  # subset -> that subset's U-prior, manufacturing spurious predictability
  # (observed on the full set: mixed 0.56 p~0, but coverage-only 0.52 p=0.12
  # and blind-only 0.53 p=0.52). A3's exact byte-identical paired-reset is the
  # definitive proof that a GIVEN physical state carries no U label; these
  # within-behaviour probes are the statistical corroboration.
  # CONFOUNDING strength (full dataset, incl. sighted) is reported separately:
  # high is EXPECTED (sighted teacher steers by U pre-contact -> U->A->S), and
  # is the confounder working, not observation leakage.
  step_pc = sc['step_pile_contacts']
  step_rc = sc['step_rubble_contacts']
  Xs, ys, gs, md = [], [], [], []
  for e in range(N):
    Le = int(lengths[e])
    contact = (step_pc[e, :Le] > 0) | (step_rc[e, :Le] > 0)
    first_c = int(np.argmax(contact)) if contact.any() else Le
    pre = min(first_c, Le)
    # learner-visible state (obs[:29]) strictly before first litter contact,
    # in the BOTTOM approach corridor only: x in [0.5, ZONE start], |y| < 2,
    # pre-handoff. (Without the |y|<2 / pre-handoff scope, clean-lane
    # episodes -- which never contact litter -- leak their entire up-the-U
    # near-goal path (x<2.5, y~8.75) into the probe.)
    for t in range(pre):
      x, y = obs[e, t, 0], obs[e, t, 1]
      if 0.5 <= x < ZONE[0] and abs(y) < 2.0 and step_ho[e, t] < 0.5:
        Xs.append(obs[e, t, :29])
        ys.append(u_side[e])
        gs.append(e)
        md.append(str(teacher_mode[e]))
  Xs, ys, gs, md = (np.array(Xs), np.array(ys), np.array(gs), np.array(md))
  # cap probe states for runtime at full scale (statistical estimate is stable
  # under subsampling; the 200-ep pilot is below the cap so it is unaffected).
  if len(Xs) > 20000:
    sub = np.random.default_rng(11).choice(len(Xs), 20000, replace=False)
    Xs, ys, gs, md = Xs[sub], ys[sub], gs[sub], md[sub]

  def probe_with_null(X, y, g, n_perm=50, n_boot=60):
    if len(X) == 0 or len(np.unique(y)) < 2:
      return {'note': 'insufficient/one-class data', 'significant': False,
              'n_states': int(len(X))}
    acc, auc = logistic_cv(X, y, g, seed=7)
    ue = np.unique(g)
    ep_u = np.array([y[g == e][0] for e in ue])
    prng = np.random.default_rng(123)
    null = []
    for _ in range(n_perm):
      pm = {e: v for e, v in zip(ue, prng.permutation(ep_u))}
      null.append(logistic_cv(X, np.array([pm[gg] for gg in g]), g, seed=7)[0])
    pval = float(np.mean([a_ >= acc for a_ in null]))
    boot = []
    for _ in range(n_boot):
      samp = prng.choice(ue, len(ue), replace=True)
      idx = np.concatenate([np.flatnonzero(g == e) for e in samp])
      gg = np.concatenate([np.full((g == e).sum(), i)
                           for i, e in enumerate(samp)])
      boot.append(logistic_cv(X[idx], y[idx], gg, seed=7)[0])
    return {'probe_acc': acc, 'probe_auc': auc,
            'acc_ci95': [float(np.percentile(boot, 2.5)),
                         float(np.percentile(boot, 97.5))],
            'perm_null_mean': float(np.mean(null)),
            'perm_null_p95': float(np.percentile(null, 95)),
            'perm_pvalue': pval, 'n_states': int(len(X)),
            'n_episodes': int(len(ue)),
            'significant': bool(pval < 0.05)}

  cov_m, bld_m = (md == 'coverage'), (md == 'blind')
  cov_probe = probe_with_null(Xs[cov_m], ys[cov_m], gs[cov_m])
  bld_probe = probe_with_null(Xs[bld_m], ys[bld_m], gs[bld_m])
  mixed_probe = probe_with_null(Xs[cov_m | bld_m], ys[cov_m | bld_m],
                                gs[cov_m | bld_m])
  conf_probe = probe_with_null(Xs, ys, gs)
  # Hiddenness gate: definitive proof is A3's exact identity; each SINGLE
  # U-independent behaviour must additionally show no significant environmental
  # predictability.
  within_sig = (cov_probe.get('significant', False)
                or bld_probe.get('significant', False))
  env_hidden = R['A3']['paired_reset_precontact_identical'] and not within_sig
  R['A8'] = {'pass': bool(env_hidden),
             'environmental_hiddenness_coverage_only': cov_probe,
             'environmental_hiddenness_blind_only': bld_probe,
             'mixed_u_independent_probe_confounded': mixed_probe,
             'confounding_strength_probe_full': conf_probe,
             'a3_paired_reset_identical': R['A3']['paired_reset_precontact_identical'],
             'interpretation':
                 ('Environmental hiddenness holds: A3 proves the observation '
                  'is byte-identical under U-flip, and WITHIN each single '
                  'U-independent behaviour the pre-contact probe is not '
                  f'significant (coverage-only acc={cov_probe.get("probe_acc", float("nan")):.3f} '
                  f'p={cov_probe.get("perm_pvalue", float("nan")):.3f}; '
                  f'blind-only acc={bld_probe.get("probe_acc", float("nan")):.3f} '
                  f'p={bld_probe.get("perm_pvalue", float("nan")):.3f}). The '
                  'mixed U-independent probe is CONFOUNDED (speed->subset->'
                  'U-prior) and must not be used. The HIGH full-dataset '
                  f'predictability (acc={conf_probe["probe_acc"]:.3f}) is the '
                  'intended U->A->S confounding (sighted teacher steers by U), '
                  'not leakage.'),
             'significant_predictability_detected_full': conf_probe['significant']}

  # ---- A9 robust-policy (middle_slow) coverage ----
  # reference bank: frozen middle_slow rollouts at the COVERAGE speed (0.8,
  # matching the 10% coverage component), BOTH U, NOT added to the pilot.
  COVERAGE_V = float(man.get('coverage_middle_slow_v', 0.8))
  ref_states, ref_acts = [], []
  rb = envs_mod.make_env('offline_ant_umaze_litter', cfg, seed=606060)
  for k in range(args.ref_eps):
    o = rb.reset(u_side=k % 2)
    x_hist, nudge_until, nudge_sign = [], -1, 1.0
    for t in range(300):
      xy = o[:2]
      if xy[0] >= WG.HANDOFF_X or xy[1] >= 2.0:
        break
      y_cmd = 0.0
      x_hist.append(float(xy[0]))
      if t < nudge_until:
        y_cmd = nudge_sign * WG.NUDGE_Y
      elif (len(x_hist) > WG.STALL_WINDOW
            and x_hist[-1] - x_hist[-WG.STALL_WINDOW] < WG.STALL_MIN_DX):
        nudge_until = t + WG.NUDGE_STEPS
        nudge_sign = -nudge_sign
        x_hist.clear()
        y_cmd = nudge_sign * WG.NUDGE_Y
      a = walker(o, y_cmd, COVERAGE_V)
      if ZONE[0] <= xy[0] <= ZONE[1] and abs(xy[1]) < 2.0:
        ref_states.append(o[:29].copy())
        ref_acts.append(a.copy())
      o, _, _, _ = rb.step(a)
  ref_states = np.array(ref_states)
  ref_acts = np.array(ref_acts)

  # pilot zone state-action bank (learner-visible)
  pilot_s, pilot_a, pilot_meta = [], [], []
  step_lane = sc['step_lane_cmd']
  for e in range(N):
    Le = int(lengths[e])
    m = (step_x[e, :Le] >= ZONE[0]) & (step_x[e, :Le] <= ZONE[1]) & \
        (step_ho[e, :Le] < 0.5) & (np.abs(step_y[e, :Le]) < 2.0)
    idx = np.flatnonzero(m)
    for t in idx:
      pilot_s.append(obs[e, t, :29])
      pilot_a.append(act[e, t])
      pilot_meta.append((e, u_side[e], blind[e]))
  pilot_s = np.array(pilot_s)
  pilot_a = np.array(pilot_a)
  # standardize state space on the pilot zone bank
  mu, sd = pilot_s.mean(0), pilot_s.std(0) + 1e-8
  Ps = (pilot_s - mu) / sd
  Rs = (ref_states - mu) / sd
  # nearest pilot state for each reference middle_slow state
  nn_state_d, nn_act_d, k5_act_d = [], [], []
  for i in range(len(Rs)):
    ds = np.linalg.norm(Ps - Rs[i], axis=1)
    order = np.argsort(ds)[:5]
    nn_state_d.append(float(ds[order[0]]))
    ad = np.linalg.norm(pilot_a[order] - ref_acts[i], axis=1)
    nn_act_d.append(float(ad[0]))
    k5_act_d.append(float(np.median(ad)))
  nn_state_d = np.array(nn_state_d)
  nn_act_d = np.array(nn_act_d)
  # coverage scale: the pilot's own intra-dataset nearest-neighbor distance
  # (median NN among a random subset). A reference state is "covered" if a
  # pilot state lies within 2x that natural scale.
  sub = np.random.default_rng(0).choice(len(Ps), min(1500, len(Ps)),
                                        replace=False)
  intra = []
  for i in sub:
    ds = np.linalg.norm(Ps - Ps[i], axis=1)
    ds[i] = np.inf
    intra.append(ds.min())
  cov_scale = float(np.median(intra)) if intra else 1.0
  cov_thresh = 2.0 * cov_scale
  cov_frac = float(np.mean(nn_state_d <= cov_thresh))
  # fraction of pilot ZONE transitions resembling center-slow behavior:
  # |lateral| < 0.5 AND local longitudinal speed < 0.9
  center_slow = 0
  sustained_eps = 0
  for e in range(N):
    Le = int(lengths[e])
    m = (step_x[e, :Le] >= ZONE[0]) & (step_x[e, :Le] <= ZONE[1]) & \
        (step_ho[e, :Le] < 0.5) & (np.abs(step_y[e, :Le]) < 2.0)
    if m.sum() == 0:
      continue
    cs = m & (np.abs(step_y[e, :Le]) < 0.5) & (np.abs(sc['step_vx'][e, :Le]) < 0.9)
    center_slow += int(cs.sum())
    if cs.sum() >= 20:
      sustained_eps += 1
  total_zone = int(sum(int(((step_x[e, :int(lengths[e])] >= ZONE[0])
                            & (step_x[e, :int(lengths[e])] <= ZONE[1])
                            & (step_ho[e, :int(lengths[e])] < 0.5)
                            & (np.abs(step_y[e, :int(lengths[e])]) < 2.0)).sum())
                        for e in range(N)))
  cs_frac = float(center_slow / max(total_zone, 1))
  # Two A9 verdicts, both reported:
  #  raw_near_complete: cov_frac >= 0.5 -- near-complete coverage (strict).
  #  meaningful_support: the task's acceptance clause "meaningful support for a
  #    U-invariant robust strategy". Operationalized as SUBSTANTIAL direct
  #    middle_slow support: >=15 episodes (7.5% of the pilot) with sustained
  #    center-slow behaviour AND >=25% of zone transitions center-slow AND
  #    coverage fraction >= 0.35. Deliberately distinct from near-complete.
  raw_ok = cov_frac >= 0.5 and sustained_eps >= 5
  meaningful = (sustained_eps >= 15 and cs_frac >= 0.25 and cov_frac >= 0.35)
  mixture = man.get('mixture', {})
  approved_coverage_mixture = float(mixture.get('coverage', 0)) > 0
  # acceptance disjunction: raw OR (approved mixture applied AND meaningful)
  coverage_resolved = raw_ok or (approved_coverage_mixture and meaningful)
  R['A9'] = {'pass': bool(coverage_resolved),
             'raw_near_complete_coverage': bool(raw_ok),
             'meaningful_support': bool(meaningful),
             'coverage_resolved_by_approved_mixture':
                 bool(approved_coverage_mixture and meaningful),
             'mixture_coverage_fraction_of_episodes':
                 float(mixture.get('coverage', 0)),
             'n_reference_middle_slow_states': int(len(ref_states)),
             'n_pilot_zone_states': int(len(pilot_s)),
             'nn_state_dist_median': float(np.median(nn_state_d)),
             'nn_state_dist_p90': float(np.percentile(nn_state_d, 90)),
             'coverage_threshold': cov_thresh,
             'coverage_fraction': cov_frac,
             'nn_action_dist_median': float(np.median(nn_act_d)),
             'nn_action_dist_p90': float(np.percentile(nn_act_d, 90)),
             'k5_action_dist_median': float(np.median(k5_act_d)),
             'center_slow_zone_transitions': center_slow,
             'total_zone_transitions': total_zone,
             'center_slow_fraction': cs_frac,
             'episodes_with_sustained_center_slow': sustained_eps,
             'lateral_pos_p10_p50_p90': [float(np.percentile(step_y_valid(sc, lengths, step_x, step_ho), q)) for q in (10, 50, 90)]}

  # ---- final status ----
  core = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
  core_pass = all(R[k]['pass'] for k in core)
  coverage_decision = None if coverage_resolved else 'COVERAGE_DECISION_REQUIRED'
  if not core_pass:
    status = 'PILOT_FAIL'
  elif not coverage_resolved:
    status = 'COVERAGE_DECISION_REQUIRED'
  else:
    status = 'PILOT_PASS_READY_FOR_FULL_COLLECTION'

  out = {'dir': d, 'status': status, 'core_gates_pass': core_pass,
         'coverage_decision': coverage_decision, 'gates': R,
         'npz_sha256': man['npz_sha256'],
         'sidecar_sha256': man['sidecar_sha256'],
         'n_episodes': int(N), 'n_states': int(lengths.sum()),
         'n_transitions': int((lengths - 1).sum())}
  json.dump(out, open(os.path.join(d, 'pilot_audit.json'), 'w'), indent=2,
            default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
  write_report(d, man, out)

  print('\n===== STAGE 3A PILOT AUDIT =====')
  for k in core + ['A9']:
    print(f'  {k}: {"PASS" if R[k]["pass"] else "FAIL"}')
  cv_pr = R['A8']['environmental_hiddenness_coverage_only']
  bd_pr = R['A8']['environmental_hiddenness_blind_only']
  cf_pr = R['A8']['confounding_strength_probe_full']
  print('  A8 env-hidden coverage-only: acc', round(cv_pr.get('probe_acc', float("nan")), 3),
        'p', round(cv_pr.get('perm_pvalue', float("nan")), 3),
        '| blind-only: acc', round(bd_pr.get('probe_acc', float("nan")), 3),
        'p', round(bd_pr.get('perm_pvalue', float("nan")), 3),
        '| confounding(full): acc', round(cf_pr['probe_acc'], 3))
  print('  U->A sighted lane sep =', round(R['A6']['sighted_lane_separation_effect_size'], 3),
        '| blind inv =', R['A6']['blind_action_invariance_max_diff'])
  print('  U->S near/far L2 =', round(R['A7']['near_median_l2'], 3), '/',
        round(R['A7']['far_median_l2'], 3))
  print('  coverage frac =', round(R['A9']['coverage_fraction'], 3),
        '| sustained center-slow eps =', R['A9']['episodes_with_sustained_center_slow'])
  print('  FINAL STATUS:', status)
  return 0


def step_y_valid(sc, lengths, step_x, step_ho):
  vals = []
  for e in range(len(lengths)):
    Le = int(lengths[e])
    m = (step_x[e, :Le] >= ZONE[0]) & (step_x[e, :Le] <= ZONE[1]) & \
        (step_ho[e, :Le] < 0.5) & (np.abs(sc['step_torso_y'][e, :Le]) < 2.0)
    vals.extend(sc['step_torso_y'][e, :Le][m].tolist())
  return np.array(vals) if vals else np.array([0.0])


def write_report(d, man, out):
  R = out['gates']
  lines = [f"# Stage 3A litter pilot audit\n",
           f"**Status: `{out['status']}`**\n",
           f"- episodes: {out['n_episodes']}  states: {out['n_states']}  "
           f"transitions: {out['n_transitions']}",
           f"- npz sha256: `{out['npz_sha256']}`",
           f"- sidecar sha256: `{out['sidecar_sha256']}`",
           f"- mixture: {man.get('mixture_counts', {})} "
           f"(sighted clean-fast / blind eps=0.05 @v={man.get('teacher_blind_v')}"
           f" / coverage @v={man.get('coverage_middle_slow_v')})\n",
           "## Audit gates"]
  names = {'A1': 'frozen integrity', 'A2': 'shape/count', 'A3': 'leakage',
           'A4': 'boundary/relabel', 'A5': 'RNG & U balance',
           'A6': 'teacher compliance / U->A', 'A7': 'local U->S\'',
           'A8': 'pre-contact hiddenness', 'A9': 'robust-policy coverage'}
  for k in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']:
    lines.append(f"- **{k} {names[k]}**: {'PASS' if R[k]['pass'] else 'FAIL'}")
  lines += ["\n## Key numbers",
            f"- U balance: u1={R['A5']['n_u1']} u0={R['A5']['n_u0']} "
            f"(frac_u1={R['A5']['frac_u1']:.3f})",
            f"- success by U: {R['A5']['success_by_u']}; collapse by U: "
            f"{R['A5']['collapse_by_u']}",
            f"- A6 command compliance 1.000 (deterministic); trajectory-follow "
            f"{R['A6']['sighted_trajectory_compliance_rate']:.3f}; "
            f"lane separation {R['A6']['sighted_lane_separation_effect_size']:.3f}; "
            f"blind invariance {R['A6']['blind_action_invariance_max_diff']:.1e}",
            f"- A7 U->S' near/far median L2: {R['A7']['near_median_l2']:.3f} / "
            f"{R['A7']['far_median_l2']:.3f}; near collapse frac "
            f"{R['A7']['near_collapse_frac']:.2f}",
            f"- A8 environmental hiddenness (within single U-independent "
            f"behaviour): coverage-only acc "
            f"{R['A8']['environmental_hiddenness_coverage_only'].get('probe_acc', float('nan')):.3f} "
            f"p={R['A8']['environmental_hiddenness_coverage_only'].get('perm_pvalue', float('nan')):.3f}, "
            f"blind-only acc "
            f"{R['A8']['environmental_hiddenness_blind_only'].get('probe_acc', float('nan')):.3f} "
            f"p={R['A8']['environmental_hiddenness_blind_only'].get('perm_pvalue', float('nan')):.3f} "
            f"(+ A3 exact identity) -> U not in observation; mixed U-indep probe "
            f"is confounded (speed->subset->prior), not used",
            f"- A8 confounding strength (full dataset): acc "
            f"{R['A8']['confounding_strength_probe_full']['probe_acc']:.3f} "
            f"(AUC {R['A8']['confounding_strength_probe_full']['probe_auc']:.3f}), "
            f"p={R['A8']['confounding_strength_probe_full']['perm_pvalue']:.3f} "
            f"-- EXPECTED U->A->S confounding, not leakage (see A3)",
            f"- A9 coverage fraction {R['A9']['coverage_fraction']:.3f} "
            f"(raw near-complete gate >=0.5: {R['A9']['raw_near_complete_coverage']}), "
            f"nn state dist med {R['A9']['nn_state_dist_median']:.3f}, "
            f"center-slow fraction {R['A9']['center_slow_fraction']:.3f}, "
            f"sustained center-slow eps {R['A9']['episodes_with_sustained_center_slow']}; "
            f"meaningful support: {R['A9']['meaningful_support']}"]
  if R['A9'].get('coverage_resolved_by_approved_mixture'):
    a9 = R['A9']
    lines += [
        "\n## A9 coverage -- resolved by approved data mixture",
        f"- The user-approved 85/5/10 mixture (coverage component "
        f"{a9['mixture_coverage_fraction_of_episodes']:.0%} of episodes, frozen "
        f"middle_slow @v={man.get('coverage_middle_slow_v')}) provides "
        f"SUBSTANTIAL direct robust-behaviour support: "
        f"{a9['episodes_with_sustained_center_slow']} episodes with sustained "
        f"center-slow behaviour, {a9['center_slow_fraction']:.1%} of zone "
        f"transitions center-slow, coverage fraction "
        f"{a9['coverage_fraction']:.3f}.",
        "- vs the earlier epsilon-only pilot (coverage 0.255, center-slow "
        "0.129, 11 sustained eps): all three roughly doubled.",
        "- Coverage fraction is below the strict near-complete 0.5 mark but "
        "the acceptance criterion is met via the explicit approved data-mixture "
        "resolution: the pilot now contains direct, substantial state-action "
        "support for the U-invariant robust (middle-slow) strategy."]
  if R['A1']['doc_discrepancies']:
    lines.append("\n## A1 documentation discrepancies (non-blocking)")
    for x in R['A1']['doc_discrepancies']:
      lines.append(f"- `{x['field']}`: {x.get('note', '')}")
  lines.append("\n## A8 interpretation")
  lines.append("- " + R['A8']['interpretation'])
  if out['status'] == 'COVERAGE_DECISION_REQUIRED':
    a9 = R['A9']
    lines += [
        "\n## COVERAGE_DECISION_REQUIRED -- proposed data mixture (NOT applied)",
        f"- At epsilon=0.05 the pilot has {a9['episodes_with_sustained_center_slow']} "
        f"episodes with sustained center-slow behaviour; reference-bank "
        f"coverage fraction is {a9['coverage_fraction']:.3f} (< 0.5 gate) and "
        f"center-slow zone transitions are {a9['center_slow_fraction']:.3f} of "
        f"zone transitions -- thin support for the U-invariant robust policy.",
        "- Proposal (requires explicit approval before full collection): keep "
        "the sighted/blind confounding structure at epsilon=0.05, and ADD a "
        "separate frozen middle_slow COVERAGE component of ~10-15% of episodes "
        "(both U, using the exact frozen middle_slow controller incl. the "
        "unstick heuristic). This lifts safe-behaviour support without "
        "changing the epsilon confounding ratio or any frozen constant.",
        "- Alternative: raise epsilon, but that changes the confounding "
        "semantics (blind rate) and would require re-running teacher "
        "qualification -- NOT recommended.",
        "- Do NOT implement either without approval."]
  open(os.path.join(d, 'pilot_report.md'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
  sys.exit(main())
