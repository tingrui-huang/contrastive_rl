"""Two small checks before deciding what to fix (user request, after bf977d3).

1. restore   Does the same (state, torque, hidden seeds) give the same physical
             start regardless of what the simulator executed before?
             `build_v6_branch_replay.restore` resets the env (mj_resetData:
             zero warmstart / qacc, rocks parked), writes qpos / qvel, sets
             the goal and the absolute time, and calls mj_forward -- it does
             not copy the full MjData.  For 60 recorded anchors (12 per
             stratum) we compare, with the recorded torque and one fixed
             hidden-seed set: (a) a fresh env instance, (b) a shared env that
             has just executed 150 continuation steps of another anchor, and
             (c) the recorded next frame obs[e, t+1] of the original episode
             (whose solver warm start and rock configuration were different);
             the first-step observation and the 10-step trajectory under the
             frozen policy, plus the MjData fields that restore does not set.

2. onestep   One-step consequence vs long-horizon consequence: at the SAME
             evaluation keys (A: familiar states + training torques; B: new
             torques at those states; old C and Cnew: new episodes), is the
             one-step state change Delta s = s1 - s0 predictable from (s, a)
             by a model fitted on the R-replay training keys, while the
             long-horizon P_goal is not?  Training keys: R replay row 0 -> row
             1 (deterministic given (s, a): rocks are parked at the anchor).
             Evaluation keys: one simulator step each (no continuation).
             Predictors: kNN (k 10, standardised (s, a)) and an MLP of the S
             predictor's architecture (1024, 1024, relu, linear 29-d output,
             squared error on standardised Delta s, Adam 3e-4, batch 1024, 30k
             updates, seeds 0 / 1) -- against the saved S predictors of P_goal
             on the same keys (their R^2 vs the 16-draw sample means).  Also,
             within an anchor, the relation between candidates' one-step
             differences and their P_goal differences.

Oracle diagnostic; nothing is trained that enters the method; no replay
changes; ~2,100 single simulator steps and 120 short restores.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402
import build_v6_policy_replay as R  # noqa: E402
import diag_v6_first_step_crossover as DG  # noqa: E402

OUT_DIR = 'diag_restore_onestep'
STRATA = ('start_early', 'start_late', 'shortcut_early', 'turn', 'north_leg')


def out_dir():
  return B.OUT / OUT_DIR


def make_env(seed):
  from crl import envs as envs_mod
  import rockfall_clock_v6_teacher as CT
  cfg, teacher = CT.make_teacher(p_active_1=B.P_ACTIVE, p_active_2=B.P_ACTIVE)
  return envs_mod.make_env(CT.ENV_NAME, cfg, seed=int(seed)), teacher


def mj_fields(env):
  d = env._env.data
  return {'time': float(d.time), 'qacc_warmstart_norm': float(np.linalg.norm(d.qacc_warmstart)), 'qacc_norm': float(np.linalg.norm(d.qacc)),
          'act_norm': float(np.linalg.norm(d.act)) if d.act.size else 0.0, 'ctrl_norm': float(np.linalg.norm(d.ctrl)), 'ncon': int(d.ncon)}


# ----------------------------------------------------------------- restore
def restore_check(args):
  man = json.loads((B.OUT / 'diag_r1_abc' / 'manifest.json').read_text(encoding='utf-8'))
  obs, act, lengths, meta, route = R.load_dataset()
  rng = np.random.default_rng(args.seed)
  picks = []
  for s in STRATA:
    pool = [a for a in man['anchors'] if a['layer'] == 'AB' and a['stratum'] == s and a['t'] + 1 < int(lengths[a['episode']]) - 1]
    picks += [pool[i] for i in rng.choice(len(pool), size=min(args.n_per_stratum, len(pool)), replace=False)]
  act_fn = DG._bc_act_fn(args.cont_ckpt)
  env_seq, _ = make_env(12345)
  prev = None
  rows = []
  for a in picks:
    e, t = a['episode'], a['t']
    state = obs[e, t, :B.STATE_DIM].astype(np.float64); goal = obs[e, t, B.STATE_DIM:B.STATE_DIM + 2].astype(np.float64)
    torque = act[e, t].astype(np.float32); seed = a['draw_seeds'][0]

    def run(env, k_steps=10):
      DG.reseed(env, seed); o0 = B.restore(env, state, goal, t)
      f0 = mj_fields(env)
      o1, r1, d1, _ = env.step(torque)
      traj = [o1[:B.STATE_DIM + 2].copy()]
      o = o1
      for _ in range(k_steps):
        if d1:
          break
        o, r1, d1, _ = env.step(act_fn(o)); traj.append(o[:B.STATE_DIM + 2].copy())
      return o0[:B.STATE_DIM + 2].copy(), np.array(traj), f0
    # (a) fresh instance
    env_iso, _ = make_env(777 + len(rows))
    o0_iso, tr_iso, f_iso = run(env_iso)
    # (b) shared instance after 150 continuation steps of the previous anchor (or of this one with other seeds)
    if prev is not None:
      DG.reseed(env_seq, prev['seed'] + 1); o = B.restore(env_seq, prev['state'], prev['goal'], prev['t'])
      o, _, d, _ = env_seq.step(prev['torque'])
      for _ in range(150):
        if d:
          break
        o, _, d, _ = env_seq.step(act_fn(o))
    o0_seq, tr_seq, f_seq = run(env_seq)
    prev = {'seed': seed, 'state': state, 'goal': goal, 't': t, 'torque': torque}
    n = min(len(tr_iso), len(tr_seq))
    rec_next = obs[e, t + 1, :B.STATE_DIM + 2]
    rows.append({'episode': int(e), 't': int(t), 'stratum': a['stratum'], 'anchor_id': a['anchor_id'],
                 'restore_vs_dataset_row_maxabs': float(np.abs(o0_iso - obs[e, t, :B.STATE_DIM + 2]).max()),
                 'iso_vs_seq_step1_maxabs': float(np.abs(tr_iso[0] - tr_seq[0]).max()),
                 'iso_vs_seq_step10_maxabs': float(np.abs(tr_iso[n - 1] - tr_seq[n - 1]).max()), 'steps_compared': int(n),
                 'iso_vs_recorded_next_maxabs': float(np.abs(tr_iso[0] - rec_next).max()),
                 'iso_vs_recorded_next_xy_dist': float(np.linalg.norm(tr_iso[0, :2] - rec_next[:2])),
                 'recorded_step_size_maxabs': float(np.abs(rec_next - obs[e, t, :B.STATE_DIM + 2]).max()),
                 'mj_after_restore_iso': f_iso, 'mj_after_restore_seq': f_seq})
  df = rows
  def agg(key):
    v = np.array([r[key] for r in df]); return {'max': float(v.max()), 'median': float(np.median(v)), 'mean': float(v.mean())}
  res = {'n_anchors': len(df), 'restore_vs_dataset_row': agg('restore_vs_dataset_row_maxabs'), 'iso_vs_seq_step1': agg('iso_vs_seq_step1_maxabs'),
         'iso_vs_seq_step10': agg('iso_vs_seq_step10_maxabs'), 'iso_vs_recorded_next': agg('iso_vs_recorded_next_maxabs'),
         'iso_vs_recorded_next_xy': agg('iso_vs_recorded_next_xy_dist'), 'recorded_step_size': agg('recorded_step_size_maxabs'),
         'per_stratum_iso_vs_recorded_next_median': {s: float(np.median([r['iso_vs_recorded_next_maxabs'] for r in df if r['stratum'] == s])) for s in STRATA},
         'mj_fields_after_restore': {'iso_warmstart_norm_max': max(r['mj_after_restore_iso']['qacc_warmstart_norm'] for r in df),
                                     'seq_warmstart_norm_max': max(r['mj_after_restore_seq']['qacc_warmstart_norm'] for r in df),
                                     'time_after_restore_iso': sorted({r['mj_after_restore_iso']['time'] for r in df})[:3],
                                     'ncon_after_restore_median': float(np.median([r['mj_after_restore_iso']['ncon'] for r in df]))},
         'rows': df}
  out_dir().mkdir(parents=True, exist_ok=True)
  (out_dir() / 'restore_check.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
  print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1, default=float), flush=True)


# ----------------------------------------------------------------- onestep
def build_model(out_dim):
  import haiku as hk
  import jax

  def fn(x):
    init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    h = hk.nets.MLP([1024, 1024], w_init=init, activation=jax.nn.relu, activate_final=True)(x)
    return hk.Linear(out_dim, w_init=init)(h)
  return hk.without_apply_rng(hk.transform(fn))


def r2(pred, y):
  ss = float(((y - y.mean(0)) ** 2).sum())
  return float(1 - ((pred - y) ** 2).sum() / ss) if ss > 0 else None


def onestep(args):
  import jax
  import jax.numpy as jnp
  import optax
  obs, act, lengths, meta, route = R.load_dataset()
  # --- training keys: R replay rows 0 -> 1
  with np.load(B.OUT / 'exp_r1_coverage' / 'replay_armR.npz', allow_pickle=False) as d:
    o0 = d['obs'][:, 0, :B.STATE_DIM + 2].astype(np.float32); o1 = d['obs'][:, 1, :B.STATE_DIM].astype(np.float32); a0 = d['act'][:, 0].astype(np.float32)
    ln = d['lengths']; aid = d['audit_anchor_id']; cand = d['audit_cand'].astype(str); pg = d['audit_p_goal']; w = d['audit_weight']; kind = d['audit_kind'].astype(str)
  ok = ln >= 2
  keys = {}
  for i in np.flatnonzero(ok):
    k = (int(aid[i]), cand[i]); e = keys.setdefault(k, {'o': o0[i], 'a': a0[i], 'd': [], 'p': [], 'w': [], 'kind': kind[i]})
    e['d'].append(o1[i] - o0[i, :B.STATE_DIM]); e['p'].append(pg[i]); e['w'].append(w[i])
  ks = list(keys)
  Xo = np.stack([keys[k]['o'] for k in ks]); Xa = np.stack([keys[k]['a'] for k in ks])
  D = np.stack([np.mean(keys[k]['d'], 0) for k in ks]); Dvar = np.array([np.mean(np.var(keys[k]['d'], 0)) for k in ks])
  P = np.array([np.average(keys[k]['p'], weights=keys[k]['w']) for k in ks]); W = np.array([np.sum(keys[k]['w']) for k in ks])
  print(f'training keys {len(ks)}; one-step change identical across draws: mean within-key variance {Dvar.mean():.2e} (across-key variance {D.var(0).mean():.2e})', flush=True)
  mu_o, sd_o = Xo.mean(0), Xo.std(0) + 1e-6; mu_a, sd_a = Xa.mean(0), Xa.std(0) + 1e-6; mu_d, sd_d = D.mean(0), D.std(0) + 1e-6
  feat = lambda o, a: np.concatenate([(o[:, :B.STATE_DIM + 2] - mu_o) / sd_o, (a - mu_a) / sd_a], 1).astype(np.float32)
  Xtr = feat(Xo, Xa); Ytr = ((D - mu_d) / sd_d).astype(np.float32)
  # --- evaluation keys: one simulator step each
  env, _ = make_env(4242)
  sets = {}
  for diag in ('diag_r1_abc', 'diag_cnew'):
    man = json.loads((B.OUT / diag / 'manifest.json').read_text(encoding='utf-8'))
    T = R.load_table(B.OUT / diag / 'outcomes.npz')
    p16 = {}
    for i in range(len(T['p_goal'])):
      p16.setdefault((int(T['anchor_id'][i]), str(T['cand'][i])), []).append(float(T['p_goal'][i]))
    for a in man['anchors']:
      state = np.asarray(a['obs'][:B.STATE_DIM], np.float64); goal = np.asarray(a['goal_xy'], np.float64)
      for c, v in a['candidates'].items():
        layer = c.split('|')[0]
        name = {'diag_r1_abc': {'A': 'A', 'B': 'B', 'C': 'C_old'}, 'diag_cnew': {'C': 'C_new'}}[diag].get(layer)
        if name is None or (a['anchor_id'], c) not in p16:
          continue
        DG.reseed(env, a['draw_seeds'][0]); o = B.restore(env, state, goal, a['t'])
        o1_, _, _, _ = env.step(np.asarray(v, np.float32))
        sets.setdefault(name, {'o': [], 'a': [], 'd': [], 'p': [], 'aid': [], 'stratum': [], 'cand': []})
        S = sets[name]; S['o'].append(np.asarray(a['obs'], np.float32)); S['a'].append(np.asarray(v, np.float32)); S['d'].append(o1_[:B.STATE_DIM] - np.asarray(a['obs'][:B.STATE_DIM], np.float32))
        S['p'].append(float(np.mean(p16[(a['anchor_id'], c)]))); S['aid'].append(a['anchor_id']); S['stratum'].append(a['stratum']); S['cand'].append(c.split('|')[1])
  for S in sets.values():
    for k in ('o', 'a', 'd', 'p'):
      S[k] = np.array(S[k])
  print({k: len(v['p']) for k, v in sets.items()}, flush=True)
  # --- kNN one-step and P predictors (training keys)
  def knn_pred(Xq, k=10):
    idx = []
    for i in range(0, len(Xq), 256):
      d2 = ((Xq[i:i + 256, None, :] - Xtr[None]) ** 2).sum(-1); idx.append(np.argpartition(d2, k, axis=1)[:, :k])
    idx = np.concatenate(idx)
    return Ytr[idx].mean(1), P[idx].mean(1)
  # --- MLP one-step predictors
  results = {'training_keys': len(ks), 'onestep_within_key_variance': float(Dvar.mean()), 'sets': {k: {'n_keys': int(len(v['p'])), 'n_anchors': int(len(set(v['aid'])))} for k, v in sets.items()}}
  mlps = []
  for seed in (0, 1):
    model = build_model(B.STATE_DIM); key = jax.random.PRNGKey(500 + seed)
    params = model.init(key, jnp.zeros((1, Xtr.shape[1]), jnp.float32)); opt = optax.adam(3e-4, eps=1e-7); st = opt.init(params)

    def loss_fn(p, x, y):
      return jnp.mean((model.apply(p, x) - y) ** 2)

    @jax.jit
    def step(p, s, x, y):
      l, g = jax.value_and_grad(loss_fn)(p, x, y); u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s, l
    rng = np.random.default_rng(seed); pw = W / W.sum(); t0 = time.time()
    for it in range(args.steps):
      idx = rng.choice(len(ks), size=1024, replace=True, p=pw)
      params, st, l = step(params, st, jnp.asarray(Xtr[idx]), jnp.asarray(Ytr[idx]))
    pred = lambda X, p=params: np.concatenate([np.asarray(model.apply(p, jnp.asarray(X[i:i + 8192]))) for i in range(0, len(X), 8192)])
    mlps.append(pred); results.setdefault('mlp_train_seconds', []).append(time.time() - t0)
  # --- S predictors of P (saved)
  sman = json.loads((B.OUT / 'diag_supervised_pgoal' / 'manifest.json').read_text(encoding='utf-8'))
  s_preds = []
  import diag_v6_supervised_pgoal as SP
  for seed in (0, 1, 2):
    fn, wa = SP.load_predictor(B.OUT / 'diag_supervised_pgoal' / f'S_seed{seed}' / 'final.pkl'); s_preds.append((fn, wa))
  # --- evaluate
  results['r2'] = {}
  results['r2']['train'] = {'onestep_mlp': [r2(m(Xtr), Ytr) for m in mlps], 'P_S': [r2(fn(SP.features(sman, Xo, Xa, wa)), P) for fn, wa in s_preds]}
  for name, S in sets.items():
    Xq = feat(S['o'], S['a']); Yq = ((S['d'] - mu_d) / sd_d).astype(np.float32)
    kd, kp = knn_pred(Xq)
    results['r2'][name] = {'onestep_knn': r2(kd, Yq), 'onestep_mlp': [r2(m(Xq), Yq) for m in mlps], 'P_knn': r2(kp, S['p']),
                           'P_S': [r2(fn(SP.features(sman, S['o'], S['a'], wa)), S['p']) for fn, wa in s_preds],
                           'P_var': float(S['p'].var()), 'onestep_var_std_units': float(Yq.var())}
    # within-anchor pairs: one-step distance vs P difference
    by = {}
    for i, a_ in enumerate(S['aid']):
      by.setdefault(a_, []).append(i)
    dd, dp, pred_dd = [], [], []
    for a_, ids in by.items():
      for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
          i, j = ids[x], ids[y]
          dd.append(float(np.linalg.norm(Yq[i] - Yq[j]))); dp.append(abs(S['p'][i] - S['p'][j]))
          pred_dd.append(float(np.linalg.norm(mlps[0](Xq[[i]])[0] - mlps[0](Xq[[j]])[0])))
    dd, dp, pred_dd = np.array(dd), np.array(dp), np.array(pred_dd)
    if len(dd) > 2:
      rk = lambda v: np.argsort(np.argsort(v))
      results['r2'][name]['within_anchor'] = {'n_pairs': int(len(dd)), 'spearman_onestep_dist_vs_P_diff': float(np.corrcoef(rk(dd), rk(dp))[0, 1]),
                                              'spearman_predicted_onestep_dist_vs_true': float(np.corrcoef(rk(pred_dd), rk(dd))[0, 1]),
                                              'frac_small_onestep_but_P_diff_gt_0.1': float(np.mean((dd < np.median(dd)) & (dp > 0.1))),
                                              'frac_large_onestep_but_P_diff_lt_0.02': float(np.mean((dd >= np.median(dd)) & (dp < 0.02)))}
  (out_dir()).mkdir(parents=True, exist_ok=True)
  (out_dir() / 'onestep.json').write_text(json.dumps(results, indent=1, default=float), encoding='utf-8')
  print(json.dumps(results, indent=1, default=float), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('restore', 'onestep'))
  ap.add_argument('--cont-ckpt', required=True)
  ap.add_argument('--seed', type=int, default=3)
  ap.add_argument('--n-per-stratum', type=int, default=12)
  ap.add_argument('--steps', type=int, default=30_000)
  args = ap.parse_args(argv)
  {'restore': restore_check, 'onestep': onestep}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
