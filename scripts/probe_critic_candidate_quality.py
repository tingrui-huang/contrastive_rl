"""Probe: does the trained CRL critic rank candidate next states by quality?

EVALUATION/DIAGNOSIS ONLY -- no training, no dataset, no loss, no policy
changes. Tests whether score(s') = f(s', pi(s',g), g) (the existing critic,
actor-mode continuation action, TRUE task goal g) separates states that lead
to the rock-death outcome from comparable states on successful trajectories.
This is the candidate-ranking component the future flow pipeline would use:
s'_wc = argmin_k f(s'_k, pi(s'_k,g), g).

State groups (from the authoritative H=800 resetfix pilot + sidecar, the same
source Part 1's failure split was cut from; scripts/make_rockfall_failure_split.py):
  bad_post  the single stored post-collapse buried pose per dead episode,
            obs[e, collapse_step+1] == obs[e, lengths-1] (identical to the
            16-state failure bank). The ant has observably entered the
            absorbing rock-death mode.
  bad_pre   the last PRE-hit state obs[e, collapse_step] of each dead episode.
            The next step kills, but the incoming rock is invisible in the
            29-dim ant obs -- falsification control: if the critic "ranks"
            these low it is reading lane pose, not physical doom.
  good      states from clean (dead=False) SUCCESSFUL episodes restricted to
            the rockfall corridor zone x in [2.3, 5.7] (the zone the diagnose
            script uses; site trigger windows span x 2.4..5.5), subsampled
            with a fixed stride. Same task stage as the deaths (which all
            occur inside the zone), so low score cannot merely mean
            "far from goal / different trajectory stage".

Goal handling: every state is scored against ITS OWN episode's commanded task
goal (sidecar goal_xy == npz eval_goals), embedded with the deployment
convention goal_vec = [gxy, 0*27] (zero-padded XY; d4rl_ant.py reset()).

Action handling: a' = pi(s',g) = tanh(loc) -- the actor's eval-mode action
(networks.sample_eval convention). No action search.

Critic aggregation: the repo's actor objective uses the pessimistic MIN over
the twin critics (losses.py); f_min is the headline score. Per-critic (f1,f2)
and f_mean are also reported so twin disagreement is visible.

Metrics per checkpoint (ranking is within-checkpoint only; absolute logits
are not comparable across checkpoints):
  * n / mean / median / std / quantiles per group;
  * AUC of badness = -f_min separating bad vs good (Mann-Whitney, ties=0.5),
    with a 95% CI from a cluster bootstrap over EPISODES (states within an
    episode are correlated; dead eps contribute 1 state per bad tier);
  * pairwise P(f_bad < f_good) (strict) over all bad x good pairs, plus the
    same restricted to x-comparable pairs |x_bad - x_good| <= dx (deaths
    cluster at x~2.7-2.9; this guards against a zone-position confound).

Outputs: per-state CSV + npz + summary JSON under --out.

Lagged mode (--bad-npz + --lags, the AUC(k) diagnostic): bad states are
rebuilt from an EXTENDED death collection
(scripts/collect_rockfall_death_extended.py) as obs[e, collapse_step + k] for
each lag k, and every ranking statistic is recomputed per lag against the
unchanged good group. Adds a label-shuffle control (must land at AUC ~0.5)
and the Spearman(f_min, x | upright good) positive control to the summary.

Usage:
  python scripts/probe_critic_candidate_quality.py \
      [--ckpt a01_best=failneg_clean_p30_h800_resetfix_a01_s0_300k/best.pkl ...]
      [--bad-npz artifacts/rockfall_death_extended/deaths_extended.npz \
       --lags 0,1,2,5,10,20,30,50]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import networks as networks_mod   # noqa: E402
from crl import checkpoint as ckpt_mod     # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

PILOT_DIR = 'artifacts/rockfall_v2_p30_h800_resetfix/pilot'
PILOT_NAME = 'antmaze_rockfall_v2_p30_h800_resetfix_pilot'
DEFAULT_CKPTS = [
    'a01_best=failneg_clean_p30_h800_resetfix_a01_s0_300k/best.pkl',
    'a01_final=failneg_clean_p30_h800_resetfix_a01_s0_300k/final.pkl',
    'a0_best=failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl',
    'a0_final=failneg_clean_p30_h800_resetfix_a0_s0_300k/final.pkl',
]
ZONE_X = (2.3, 5.7)     # rockfall corridor zone (diagnose_naive_rockfall.py)
STRIDE = 10             # good-state subsample stride within an episode
DX_MATCH = 0.5          # |x_bad - x_good| window for the x-matched pair stat
N_BOOT = 1000
CHUNK = 256


def auc_bad_lower(f_bad, f_good):
  """AUC of badness=-f: P(f_bad < f_good) + 0.5*P(tie), rank-based."""
  nb, ng = len(f_bad), len(f_good)
  if nb == 0 or ng == 0:
    return float('nan')
  allv = np.concatenate([f_bad, f_good])
  order = np.argsort(allv, kind='mergesort')
  ranks = np.empty(len(allv))
  ranks[order] = np.arange(1, len(allv) + 1)
  # average ranks for ties
  sv = allv[order]
  i = 0
  while i < len(sv):
    j = i
    while j + 1 < len(sv) and sv[j + 1] == sv[i]:
      j += 1
    if j > i:
      ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
    i = j + 1
  # U for the GOOD group being higher
  u_good = ranks[nb:].sum() - ng * (ng + 1) / 2.0
  return float(u_good / (nb * ng))


def pairwise_stats(f_bad, x_bad, f_good, x_good, dx):
  """Strict P(f_bad < f_good), tie fraction, and the |dx|-restricted version."""
  fb = f_bad[:, None]
  fg = f_good[None, :]
  lt = (fb < fg)
  eq = (fb == fg)
  out = {'p_bad_lt_good': float(lt.mean()), 'tie_frac': float(eq.mean())}
  m = np.abs(x_bad[:, None] - x_good[None, :]) <= dx
  out['n_pairs'] = int(lt.size)
  out['n_pairs_xmatched'] = int(m.sum())
  out['p_bad_lt_good_xmatched'] = (float(lt[m].mean()) if m.any()
                                   else float('nan'))
  return out


def cluster_bootstrap_auc(f_bad, ep_bad, f_good, ep_good, n_boot, rng):
  """95% CI on AUC from resampling EPISODES with replacement per group."""
  ub, ug = np.unique(ep_bad), np.unique(ep_good)
  idx_b = {e: np.where(ep_bad == e)[0] for e in ub}
  idx_g = {e: np.where(ep_good == e)[0] for e in ug}
  vals = []
  for _ in range(n_boot):
    eb = rng.choice(ub, size=len(ub), replace=True)
    eg = rng.choice(ug, size=len(ug), replace=True)
    fb = np.concatenate([f_bad[idx_b[e]] for e in eb])
    fg = np.concatenate([f_good[idx_g[e]] for e in eg])
    vals.append(auc_bad_lower(fb, fg))
  return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def shuffle_control_auc(f_bad, f_good, n_shuffle, rng):
  """Label-shuffle null: AUC after randomly re-splitting the pooled scores
  into groups of the same sizes. Mean must land at ~0.5."""
  pool = np.concatenate([f_bad, f_good])
  nb = len(f_bad)
  vals = []
  for _ in range(n_shuffle):
    perm = rng.permutation(len(pool))
    vals.append(auc_bad_lower(pool[perm[:nb]], pool[perm[nb:]]))
  return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
          'n_shuffle': int(n_shuffle)}


def group_summary(f):
  qs = [5, 10, 25, 50, 75, 90, 95]
  return {'n': int(len(f)), 'mean': float(np.mean(f)),
          'median': float(np.median(f)), 'std': float(np.std(f)),
          'quantiles': {f'p{q}': float(np.percentile(f, q)) for q in qs}}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--data-dir', default=PILOT_DIR)
  ap.add_argument('--data-name', default=PILOT_NAME)
  ap.add_argument('--ckpt', action='append', default=None,
                  metavar='TAG=PATH',
                  help='checkpoint(s) to probe; default = a01/a0 best+final')
  ap.add_argument('--out', default='artifacts/critic_candidate_probe')
  ap.add_argument('--stride', type=int, default=STRIDE)
  ap.add_argument('--dx-match', type=float, default=DX_MATCH)
  ap.add_argument('--n-boot', type=int, default=N_BOOT)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--bad-npz', default='',
                  help='extended death collection npz; bad states become '
                       'obs[e, collapse_step+k] per --lags (replaces the '
                       'default bad_pre/bad_post construction)')
  ap.add_argument('--lags', default='0,1,2,5,10,20,30,50',
                  help='comma-separated lags k for --bad-npz mode')
  ap.add_argument('--n-shuffle', type=int, default=200,
                  help='label-shuffle control resamples per bad group')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  ckpts = [c.split('=', 1) for c in (args.ckpt or DEFAULT_CKPTS)]

  # ---------------- data: probe state construction ----------------
  npz = os.path.join(args.data_dir, f'{args.data_name}.npz')
  sidecar = os.path.join(args.data_dir, f'{args.data_name}_sidecar.npz')
  d = np.load(npz, allow_pickle=True)
  s = np.load(sidecar, allow_pickle=True)
  meta = json.loads(str(d['meta']))
  obs_dim, goal_dim, act_dim = (meta['obs_dim'], meta['goal_dim'],
                                meta['action_dim'])
  obs, lengths = d['obs'], np.asarray(d['lengths'], np.int64)
  goal_xy = np.asarray(s['goal_xy'], np.float32)
  assert np.allclose(goal_xy, d['eval_goals']), 'sidecar/npz goal mismatch'
  dead = np.asarray(s['dead'], bool)
  succ = np.asarray(s['success'], float)
  collapse = np.asarray(s['collapse_step'], np.int64)

  rows = []   # (group, ep, t, state[29], goal_xy[2])
  if args.bad_npz:
    lags = [int(k) for k in args.lags.split(',')]
    bd = np.load(args.bad_npz, allow_pickle=True)
    b_obs = bd['obs']
    b_col = np.asarray(bd['collapse_step'], np.int64)
    b_end = np.asarray(bd['end_t'], np.int64)
    b_goal = np.asarray(bd['goal_xy'], np.float32)
    bad_names = [f'bad_k{k}' for k in lags]
    for k, name in zip(lags, bad_names):
      for i in range(len(b_col)):
        t = int(b_col[i]) + k
        if t <= int(b_end[i]):
          # 100000+i: probe-local episode id, disjoint from pilot ids
          rows.append((name, 100000 + i, t, b_obs[i, t, :obs_dim],
                       b_goal[i]))
  else:
    lags = None
    bad_names = ['bad_pre', 'bad_post']
    for e in np.where(dead)[0]:
      c, last = int(collapse[e]), int(lengths[e]) - 1
      assert last == c + 1, 'dead episode not truncated at collapse+1 obs'
      assert s['step_dead'][e, c] == 1.0, 'step_dead misaligned with collapse'
      rows.append(('bad_pre', e, c, obs[e, c, :obs_dim], goal_xy[e]))
      rows.append(('bad_post', e, last, obs[e, last, :obs_dim], goal_xy[e]))

  good_eps = np.where((~dead) & (succ == 1.0))[0]
  for e in good_eps:
    x = s['step_torso_x'][e, :lengths[e] - 1]
    tz = np.where((x >= ZONE_X[0]) & (x <= ZONE_X[1]))[0]
    for t in tz[::args.stride]:
      rows.append(('good', e, int(t), obs[e, t, :obs_dim], goal_xy[e]))

  group = np.array([r[0] for r in rows])
  ep = np.array([r[1] for r in rows], np.int64)
  tstep = np.array([r[2] for r in rows], np.int64)
  states = np.stack([r[3] for r in rows]).astype(np.float32)
  state_goal_xy = np.stack([r[4] for r in rows]).astype(np.float32)
  # deployment goal convention: zero-padded XY of the episode's task goal
  goals = np.zeros((len(rows), goal_dim), np.float32)
  goals[:, :2] = state_goal_xy
  og = np.concatenate([states, goals], axis=1)
  print(f'probe states: {dict(zip(*np.unique(group, return_counts=True)))} '
        f'(good eps {len(good_eps)}, dead eps {int(dead.sum())}, '
        f'zone x in {ZONE_X}, stride {args.stride}'
        + (f', bad-npz {args.bad_npz} lags {lags}' if args.bad_npz else '')
        + ')', flush=True)

  # ---------------- networks (exact run config; no env needed) -------------
  cfg = build_offline_cfg()
  nets = networks_mod.make_networks(
      obs_dim=obs_dim, goal_dim=goal_dim, action_dim=act_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=False,
      use_layer_norm=cfg.use_layer_norm)

  rng = np.random.default_rng(args.seed)
  all_groups = ['good'] + bad_names
  summary = {'data': npz, 'zone_x': ZONE_X, 'stride': args.stride,
             'dx_match': args.dx_match,
             'bad_npz': args.bad_npz or None, 'lags': lags,
             'goal_convention': 'zero-padded XY (deployment/eval contract)',
             'twin_aggregation': 'min (actor-objective convention)',
             'n_states': {g: int((group == g).sum()) for g in all_groups},
             'per_ckpt': {}}
  per_state = {}   # tag -> dict of score arrays

  for tag, path in ckpts:
    step, st = ckpt_mod.load_checkpoint(path)
    assert 'sa_encoder/~/linear_0' in st.q_params, (
        'unexpected critic params (LayerNorm run?); this probe assumes the '
        'faithful no-LayerNorm recipe')

    @jax.jit
    def score_chunk(o, pp=st.policy_params, qp=st.q_params):
      a = jnp.tanh(nets.policy_network.apply(pp, o).loc)      # pi(s', g) mode
      q = nets.q_network.apply(qp, o, a)                      # [b, b, 2]
      f = jnp.diagonal(q, axis1=0, axis2=1).T                 # [b, 2]
      return a, f

    acts = np.empty((len(og), act_dim), np.float32)
    f12 = np.empty((len(og), 2), np.float32)
    for i in range(0, len(og), CHUNK):
      a, f = score_chunk(jnp.asarray(og[i:i + CHUNK]))
      acts[i:i + CHUNK] = np.asarray(a)
      f12[i:i + CHUNK] = np.asarray(f)
    scores = {'f1': f12[:, 0], 'f2': f12[:, 1],
              'f_min': f12.min(axis=1), 'f_mean': f12.mean(axis=1)}
    per_state[tag] = scores

    x = states[:, 0]
    rep = {'path': path, 'step': int(step), 'groups': {}, 'ranking': {}}
    for g in all_groups:
      rep['groups'][g] = {k: group_summary(v[group == g])
                          for k, v in scores.items()}
    # positive control: on upright good states, f_min should track corridor
    # progress x (geodesic progress toward the goal within the zone).
    from scipy.stats import spearmanr
    mg_all = group == 'good'
    up = mg_all & (states[:, 2] >= 0.35)
    rep['spearman_fmin_x_upright_good'] = float(
        spearmanr(scores['f_min'][up], x[up]).statistic)
    for bad in bad_names:
      mb, mg = group == bad, group == 'good'
      r = {k: {'auc_badness': auc_bad_lower(v[mb], v[mg])}
           for k, v in scores.items()}
      r['f_min'].update(pairwise_stats(scores['f_min'][mb], x[mb],
                                       scores['f_min'][mg], x[mg],
                                       args.dx_match))
      r['f_min']['auc_ci95'] = cluster_bootstrap_auc(
          scores['f_min'][mb], ep[mb], scores['f_min'][mg], ep[mg],
          args.n_boot, rng)
      r['f_min']['shuffle_control'] = shuffle_control_auc(
          scores['f_min'][mb], scores['f_min'][mg], args.n_shuffle, rng)
      rep['ranking'][f'{bad}_vs_good'] = r
    summary['per_ckpt'][tag] = rep

    fm = rep['groups']
    rk = rep['ranking']
    print(f"\n== {tag} (step {step}) ==")
    print('  n: ' + '  '.join(f"{g} {fm[g]['f_min']['n']}"
                              for g in all_groups))
    print(f"  positive control spearman(f_min, x | upright good): "
          f"{rep['spearman_fmin_x_upright_good']:+.3f}")
    for g in all_groups:
      gs = fm[g]['f_min']
      print(f"  f_min[{g:8s}] mean {gs['mean']:8.3f}  median "
            f"{gs['median']:8.3f}  p10 {gs['quantiles']['p10']:8.3f}  "
            f"p90 {gs['quantiles']['p90']:8.3f}")
    for bad in bad_names:
      r = rk[f'{bad}_vs_good']['f_min']
      sh = r['shuffle_control']
      print(f"  {bad:8s} vs good: AUC(-f_min) {r['auc_badness']:.3f} "
            f"CI95 [{r['auc_ci95'][0]:.3f}, {r['auc_ci95'][1]:.3f}]  "
            f"P(f_bad<f_good) {r['p_bad_lt_good']:.3f}  "
            f"x-matched {r['p_bad_lt_good_xmatched']:.3f} "
            f"(n={r['n_pairs_xmatched']})  "
            f"shuffle {sh['mean']:.3f}+-{sh['std']:.3f}")
      aucs = {k: rk[f'{bad}_vs_good'][k]['auc_badness']
              for k in ('f1', 'f2', 'f_mean')}
      print(f"           per-critic AUC: f1 {aucs['f1']:.3f}  "
            f"f2 {aucs['f2']:.3f}  f_mean {aucs['f_mean']:.3f}")

  # ---------------- outputs ----------------
  with open(os.path.join(args.out, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  npz_out = {'group': group, 'ep': ep, 't': tstep, 'states': states,
             'goal_xy': state_goal_xy}
  for tag, sc in per_state.items():
    for k, v in sc.items():
      npz_out[f'{tag}__{k}'] = v
  np.savez_compressed(os.path.join(args.out, 'per_state_scores.npz'),
                      **npz_out)
  csv_path = os.path.join(args.out, 'per_state_scores.csv')
  tags = [t for t, _ in ckpts]
  with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group', 'ep', 't', 'x', 'y', 'z']
               + [f'{t}__{k}' for t in tags
                  for k in ('f1', 'f2', 'f_min', 'f_mean')])
    for i in range(len(group)):
      w.writerow([group[i], ep[i], tstep[i],
                  f'{states[i, 0]:.4f}', f'{states[i, 1]:.4f}',
                  f'{states[i, 2]:.4f}']
                 + [f'{per_state[t][k][i]:.5f}' for t in tags
                    for k in ('f1', 'f2', 'f_min', 'f_mean')])
  print(f"\nsaved {os.path.join(args.out, 'summary.json')}, "
        f"per_state_scores.npz/.csv", flush=True)


if __name__ == '__main__':
  main()
