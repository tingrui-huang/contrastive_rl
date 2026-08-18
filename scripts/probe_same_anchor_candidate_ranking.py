"""Same-anchor safe/fatal candidate ranking probe (paired critic test).

For each pair (s, a, s'_safe, s'_fatal) built by
scripts/build_same_anchor_pairs.py -- identical anchor (s, a), the two
candidates being the env's own one-step outcomes under the factual vs
counterfactual hidden hazard -- score the CANDIDATES IN THE GOAL SLOT:

    f_safe  = f(s, a, s'_safe)
    f_fatal = f(s, a, s'_fatal)
    Delta   = f_safe - f_fatal          (correct ranking: Delta > 0)

No actor call, no task goal, no bank distances, no classifier: the only
critic inputs are the anchor state-action and the candidate converted with
the CRL implementation's own state->goal rule (crl.replay.obs_to_goal,
goal_indices = range(29)).

PRIMARY metric: paired win rate P(Delta > 0) with a bootstrap 95% CI over
pairs. Secondary: median/mean Delta (+CIs), ties, per-head (f1, f2) win
rates, and a descriptive unpaired AUC. The repo's standard twin aggregation
for decision-making is the pessimistic MIN (actor objective, losses.py);
the critic LOSS trains the twin MEAN -- both are reported.

Usage:
  python scripts/probe_same_anchor_candidate_ranking.py \
      --checkpoint <run>/best.pkl \
      --pairs artifacts/same_anchor_candidate_probe/pairs_heldout40.npz \
      --output artifacts/same_anchor_candidate_probe/results_<tag>

The held-out pair file (pairs_heldout40.npz) is a SEALED test set: score it
only with the frozen probe definition once the production settled-bank
alpha=0.1 checkpoint exists, and never use its scores to tune anything.
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
from crl.replay import obs_to_goal         # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OBS_DIM, GOAL_DIM, ACT_DIM = 29, 29, 8
N_BOOT = 10_000


def boot_ci(vals, stat_fn, n_boot, rng):
  n = len(vals)
  stats = [stat_fn(vals[rng.integers(0, n, n)]) for _ in range(n_boot)]
  return [float(np.percentile(stats, 2.5)),
          float(np.percentile(stats, 97.5))]


def summarize(delta, n_boot, rng):
  n = len(delta)
  wins = int((delta > 0).sum())
  ties = int((delta == 0).sum())
  return {
      'n_pairs': n,
      'paired_win_rate': wins / n,
      'paired_win_rate_ci95': boot_ci(delta, lambda d: float((d > 0).mean()),
                                      n_boot, rng),
      'n_ties': ties,
      'median_delta': float(np.median(delta)),
      'median_delta_ci95': boot_ci(delta, np.median, n_boot, rng),
      'mean_delta': float(np.mean(delta)),
      'mean_delta_ci95': boot_ci(delta, np.mean, n_boot, rng),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--checkpoint', required=True)
  ap.add_argument('--pairs', required=True)
  ap.add_argument('--output', required=True,
                  help='output prefix/dir: results.json + per_pair.csv')
  ap.add_argument('--n-boot', type=int, default=N_BOOT)
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  os.makedirs(args.output, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  d = np.load(args.pairs, allow_pickle=True)
  meta = json.loads(str(d['meta']))
  s = np.asarray(d['anchor_obs'], np.float32)          # [N, 29]
  a = np.asarray(d['anchor_action'], np.float32)       # [N, 8]
  cands = {'safe': np.asarray(d['safe_candidate'], np.float32),
           'fatal': np.asarray(d['fatal_candidate'], np.float32)}
  n = len(s)
  assert s.shape == (n, OBS_DIM) and a.shape == (n, ACT_DIM)
  print(f"pairs: {n} ({meta['label']}) | ckpt: {args.checkpoint}")
  if 'heldout' in meta['label']:
    print('NOTE: scoring the SEALED held-out set -- the probe definition '
          'must already be frozen.')

  cfg = build_offline_cfg()
  nets = networks_mod.make_networks(
      obs_dim=OBS_DIM, goal_dim=GOAL_DIM, action_dim=ACT_DIM,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=False,
      use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(args.checkpoint)
  assert 'sa_encoder/~/linear_0' in st.q_params, 'unexpected critic params'

  @jax.jit
  def score(og, act, qp=st.q_params):
    q = nets.q_network.apply(qp, og, act)               # [B, B, 2]
    return jnp.diagonal(q, axis1=0, axis2=1).T          # [B, 2]

  f = {}
  for name, cand in cands.items():
    g = obs_to_goal(cand, 0, -1, tuple(range(OBS_DIM)))  # CRL state->goal
    og = np.concatenate([s, g.astype(np.float32)], axis=1)
    f[name] = np.asarray(score(jnp.asarray(og), jnp.asarray(a)))  # [N, 2]

  conv = {
      'f1': (f['safe'][:, 0], f['fatal'][:, 0]),
      'f2': (f['safe'][:, 1], f['fatal'][:, 1]),
      'f_min': (f['safe'].min(1), f['fatal'].min(1)),
      'f_mean': (f['safe'].mean(1), f['fatal'].mean(1)),
  }
  results = {'checkpoint': args.checkpoint, 'step': int(step),
             'pairs': args.pairs, 'pairs_label': meta['label'],
             'n_pairs': n,
             'twin_aggregation_note': ('f_min = actor-objective convention '
                                       '(pessimistic min); f_mean = critic-'
                                       'loss convention; f1/f2 = heads'),
             'per_convention': {}}
  for name, (fs, ff) in conv.items():
    delta = fs - ff
    r = summarize(delta, args.n_boot, rng)
    # secondary, descriptive only (data are explicitly paired):
    lab = np.r_[np.ones(n), np.zeros(n)]
    sc = np.r_[fs, ff]
    order = sc.argsort(kind='mergesort')
    ranks = np.empty(2 * n)
    ranks[order] = np.arange(1, 2 * n + 1)
    sv = sc[order]
    i = 0
    while i < len(sv):                       # average ranks over ties
      j = i
      while j + 1 < len(sv) and sv[j + 1] == sv[i]:
        j += 1
      if j > i:
        ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
      i = j + 1
    auc = float((ranks[lab == 1].sum() - n * (n + 1) / 2) / (n * n))
    r['secondary_unpaired_auc'] = auc
    results['per_convention'][name] = r

  csv_path = os.path.join(args.output, 'per_pair.csv')
  with open(csv_path, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['pair', 'episode_id', 'f1_safe', 'f1_fatal', 'f2_safe',
                'f2_fatal', 'fmin_safe', 'fmin_fatal', 'delta_fmin'])
    for i in range(n):
      w.writerow([i, int(d['episode_id'][i]),
                  f"{f['safe'][i, 0]:.5f}", f"{f['fatal'][i, 0]:.5f}",
                  f"{f['safe'][i, 1]:.5f}", f"{f['fatal'][i, 1]:.5f}",
                  f"{f['safe'][i].min():.5f}", f"{f['fatal'][i].min():.5f}",
                  f"{f['safe'][i].min() - f['fatal'][i].min():.5f}"])
  with open(os.path.join(args.output, 'results.json'), 'w') as fh:
    json.dump(results, fh, indent=2)

  print(f"\n== {args.checkpoint} (step {step}) on {meta['label']} ==")
  for name in ('f_min', 'f_mean', 'f1', 'f2'):
    r = results['per_convention'][name]
    print(f"  {name:6s}: win rate {r['paired_win_rate']:.3f} "
          f"CI95 [{r['paired_win_rate_ci95'][0]:.3f}, "
          f"{r['paired_win_rate_ci95'][1]:.3f}]  "
          f"median D {r['median_delta']:+.3f}  mean D {r['mean_delta']:+.3f}"
          f"  ties {r['n_ties']}  (aux AUC {r['secondary_unpaired_auc']:.3f})")
  print(f"saved {os.path.join(args.output, 'results.json')}, per_pair.csv")


if __name__ == '__main__':
  main()
