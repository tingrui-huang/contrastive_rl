"""M0: agreement-coin == propensity-table equivalence tests (analysis only).

Background. The archived Thm-2 sampler's per-step coin is
Bernoulli(P_hat(bin(x_t)|cell)) -- a probability LOOKUP. The CFQL-inspired
"agreement" coin replaces the lookup with an EVENT: draw a fresh behavior
action at the same state, heads iff it matches the logged action. In the
discrete/binned world these are the same coin,

    E[ 1{bin(X_fresh) = bin(x_t)} | cell ] = P_emp(bin(x_t) | cell),

and M0's job is to certify that identity ON THE REAL MACHINERY before any
M1 work builds on it -- plus to pin down, with numbers, the confusion that
was caught in review: a SOURCE CLASSIFIER (which pile did this action come
from?) does NOT estimate a propensity (how often does behavior choose it?).

  T1  pointwise: MC exact-match frequency (resampling real dataset actions
      per cell) == unsmoothed empirical bin frequency within binomial CI;
      the saved table differs from the empirical frequency only by the
      exact Laplace formula (checked bitwise on counts, exactly on probs).
  T2  sampler-level: AgreementCoinSampler (walk_from verbatim except the
      coin lines) vs the archived table-coin ManskiSampler, same anchors,
      both ball-N+hazard and reachable-N modes. Endpoint-cell TV distance
      is judged against a seed-vs-seed null of the table sampler; teleport
      fraction and the G7 pessimism statistic (mean endpoint BFS) must
      match. An unsmoothed-table arm decomposes any gap into
      "Laplace smoothing" vs "coin mechanism".
  T3  three-quantities demo (synthetic beta=(.7,.3) and the real HOLDING
      cell):
        (a) agreement estimator            -> beta(x|s)        (frequency)
        (b) source classifier data-vs-BC   -> 0.5              (source,
            degenerate when the BC model is good -- NOT a propensity)
        (c) source classifier beta-vs-pi   -> beta/(beta+pi)   (CFQL's D)
      using the closed-form Bayes classifier for binned actions
      (count ratio), so no NN noise clouds the point.

Nothing frozen is modified. The archived sampler is imported verbatim from
scripts/manski_archive_c1368c7.py.

Run:  python -m scripts.m0_agreement_equivalence
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import manski_archive_c1368c7 as MA        # noqa: E402
from crl import envs as envs_mod           # noqa: E402
from crl.config import Config              # noqa: E402

MSWAMP = ((3, 3), (4, 3), (5, 3))          # matched-setting swamp cells


# ---------------------------------------------------------------------------
# Agreement-coin sampler: identical walk, only the coin lines differ.
# ---------------------------------------------------------------------------
class AgreementCoinSampler(MA.ManskiSampler):
  """Thm-2 walk whose coin is the EVENT {fresh behavior draw agrees with the
  logged action's bin}, instead of Bernoulli(P_hat). Fresh draws come from
  the same per-cell anchor pools the sampler already maintains, so the
  expectation of the coin is the UNSMOOTHED empirical bin frequency."""

  def __init__(self, obs, act, walls, probs, n_sectors, zero_thresh, dist,
               gamma, seed=0, hazard_cells=(), reachable_n=False):
    super().__init__(obs, act, walls, probs, n_sectors, zero_thresh, dist,
                     gamma, seed=seed, hazard_cells=hazard_cells,
                     reachable_n=reachable_n)
    self._bins = MA.action_bins(act.reshape(-1, 2), n_sectors, zero_thresh)
    self.n_fallback = 0                    # cells without a pool (should be 0)

  def _agreement_event(self, flat):
    """One fresh behavior draw per walker (grouped by cell); True = agree."""
    rng = self._rng
    agree = np.zeros(len(flat), bool)
    key = (self._cell_i[flat] * self._walls.shape[1] + self._cell_j[flat])
    for k in np.unique(key):
      sel = np.where(key == k)[0]
      pool = self._pool.get((int(k) // self._walls.shape[1],
                             int(k) % self._walls.shape[1]))
      if pool is None:                     # no anchorable data in this cell
        self.n_fallback += len(sel)
        agree[sel] = rng.random(len(sel)) < self._phat[flat[sel]]
        continue
      draws = pool[rng.integers(len(pool), size=len(sel))]
      agree[sel] = self._bins[draws] == self._bins[flat[sel]]
    return agree

  def walk_from(self, start_flat, p_override=None, max_steps=2000,
                collect_maps=False):
    # Verbatim copy of MA.ManskiSampler.walk_from except the coin lines,
    # which are marked with  # <<< COIN.
    rng = self._rng
    cur = np.asarray(start_flat, np.int64).copy()
    batch = len(cur)
    alive = np.ones(batch, bool)
    teleports = np.zeros(self._walls.shape, np.int64) if collect_maps else None
    visits = np.zeros(self._walls.shape, np.int64) if collect_maps else None
    n_steps = 0
    while alive.any() and n_steps < max_steps:
      n_steps += 1
      alive &= rng.random(batch) < self._gamma
      idx = np.where(alive)[0]
      if not len(idx):
        break
      trunc = idx[self._t_of[cur[idx]] == self._length - 1]
      if len(trunc):
        self._group_reanchor(cur, trunc, self._pool)
      ci, cj = self._cell_i[cur[idx]], self._cell_j[cur[idx]]
      if collect_maps:
        np.add.at(visits, (ci, cj), 1)
      if p_override is None:                                    # <<< COIN
        agree = self._agreement_event(cur[idx])                 # <<< COIN
      else:                                                     # <<< COIN
        agree = rng.random(len(idx)) < p_override               # <<< COIN
      if self._reachable and p_override is None:
        stuck = self._u_flag[cur[idx]] & ~agree                 # <<< COIN
        cur[idx[~stuck]] += 1
        if stuck.any():
          if collect_maps:
            np.add.at(teleports, (ci[stuck], cj[stuck]), 1)
          alive[idx[stuck]] = False
        continue
      walk = agree                                              # <<< COIN
      cur[idx[walk]] += 1
      tele = idx[~walk]
      if len(tele):
        if collect_maps:
          np.add.at(teleports, (ci[~walk], cj[~walk]), 1)
        self._group_reanchor(cur, tele, self._worst_pool)
        dead = self._hazard_mask[self._cell_i[cur[tele]],
                                 self._cell_j[cur[tele]]]
        if dead.any():
          alive[tele[dead]] = False
    return (cur, teleports, visits) if collect_maps else cur


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def cell_hist(walls, cell_i, cell_j):
  h = np.zeros(walls.shape)
  np.add.at(h, (cell_i, cell_j), 1.0)
  return h / h.sum()


def tv(h1, h2):
  return 0.5 * float(np.abs(h1 - h2).sum())


def run_arm(cls, obs, act, walls, probs, tcfg, dist, gamma, seed, anchors,
            hazard, reachable):
  s = cls(obs, act, walls, probs, tcfg['sectors'], tcfg['zero_thresh'], dist,
          gamma, seed=seed, hazard_cells=hazard, reachable_n=reachable)
  cur, tele_map, visit_map = s.walk_from(anchors.copy(), collect_maps=True)
  ci, cj = s._cell_i[cur], s._cell_j[cur]
  return dict(
      hist=cell_hist(walls, ci, cj),
      teleport_frac=float(tele_map.sum() / max(visit_map.sum(), 1)),
      mean_bfs=float(np.mean(dist[ci, cj])),
      se_bfs=float(np.std(dist[ci, cj]) / np.sqrt(len(cur))),
      n_fallback=int(getattr(s, 'n_fallback', 0)),
  )


def bayes_classifier(pos_bins, neg_bins, k):
  """Closed-form optimal P(pos | bin) from labeled samples (count ratio)."""
  cp = np.bincount(pos_bins, minlength=k).astype(float)
  cn = np.bincount(neg_bins, minlength=k).astype(float)
  with np.errstate(invalid='ignore'):
    return cp / (cp + cn)


# ---------------------------------------------------------------------------
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--dataset', default='datasets/swamp_matched_teacher_s0.npz')
  ap.add_argument('--env_name', default='point_two_route_swamp_matched_v0')
  ap.add_argument('--table', default='artifacts/manski_port/propensity_table.npz')
  ap.add_argument('--gamma', type=float, default=0.95)
  ap.add_argument('--batch', type=int, default=20000)
  ap.add_argument('--mc', type=int, default=20000, help='T1 draws per cell')
  ap.add_argument('--min_cell_count', type=int, default=500)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default='artifacts/m0_agreement')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(args.seed)
  report = {'config': vars(args), 'gates': []}

  def gate(name, passed, **metrics):
    report['gates'].append(dict(name=name, passed=bool(passed),
                                metrics=metrics))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  "
          + ' '.join(f'{k}={v}' for k, v in metrics.items()), flush=True)

  cfg = Config(env_name=args.env_name)
  env = envs_mod.make_env(args.env_name, cfg, seed=args.seed)
  walls = env._walls
  goal_cell = tuple(MA.cells_of(walls, np.asarray(env.GOAL)))
  holding = tuple(env.HOLDING_CELL)
  data = np.load(args.dataset, allow_pickle=True)
  obs, act = data['obs'], data['act']
  table = np.load(args.table, allow_pickle=True)
  tcfg = json.loads(str(table['config']))
  probs_smoothed, counts_saved = table['probs'], table['counts']
  k = MA.n_bins(tcfg['sectors'])
  dist = MA.bfs_dist_map(walls, goal_cell)

  # ---- shared flat views (t <= L-2 rows carry a real action) --------------
  ne, L = obs.shape[0], obs.shape[1]
  xy = obs[:, :, :2].reshape(-1, 2)
  t_of = np.tile(np.arange(L), ne)
  rows = t_of <= L - 2
  cells = MA.cells_of(walls, xy)
  bins = MA.action_bins(act.reshape(-1, 2), tcfg['sectors'],
                        tcfg['zero_thresh'])

  # =========================================================================
  # T1 -- pointwise: MC agreement == empirical frequency; table == Laplace
  # =========================================================================
  counts, _ = MA.fit_propensity(walls, cells[rows], bins[rows],
                                tcfg['sectors'], alpha=tcfg['alpha'])
  gate('T1a_counts_replicate_exactly',
       np.array_equal(counts, counts_saved),
       max_abs_diff=float(np.abs(counts - counts_saved).max()))

  lap = (counts + tcfg['alpha']) / (counts.sum(-1, keepdims=True)
                                    + tcfg['alpha'] * k)
  gate('T1b_table_is_exact_laplace',
       np.allclose(lap, probs_smoothed, atol=1e-12),
       max_abs_diff=float(np.abs(lap - probs_smoothed).max()))

  tot = counts.sum(-1)
  p_emp = np.divide(counts, np.maximum(tot[..., None], 1))
  test_cells = [tuple(c) for c in np.argwhere(tot >= args.min_cell_count)]
  violations, n_tests, max_z = 0, 0, 0.0
  worst = None
  for (i, j) in test_cells:
    pool = np.where(rows & (cells[:, 0] == i) & (cells[:, 1] == j))[0]
    draws = bins[pool[rng.integers(len(pool), size=args.mc)]]
    freq = np.bincount(draws, minlength=k) / args.mc
    for b in range(k):
      p = p_emp[i, j, b]
      se = np.sqrt(max(p * (1 - p), 1e-12) / args.mc)
      z = abs(freq[b] - p) / max(se, 1e-12)
      n_tests += 1
      if z > max_z:
        max_z, worst = z, (i, j, b)
      if z > 4.0 and abs(freq[b] - p) > 5e-4:
        violations += 1
  gate('T1c_mc_agreement_matches_empirical', violations == 0,
       cells_tested=len(test_cells), tests=n_tests, violations=violations,
       max_z=round(max_z, 2), worst=str(worst))
  report['t1_smoothing_gap'] = float(
      np.abs(probs_smoothed - p_emp)[tot >= args.min_cell_count].max())

  # =========================================================================
  # T2 -- sampler-level equivalence, both modes
  # =========================================================================
  probs_emp = np.divide(counts, np.maximum(tot[..., None], 1),
                        out=np.full_like(counts, 1.0 / k),
                        where=tot[..., None] > 0)
  anchorable = np.where(rows & (walls[cells[:, 0], cells[:, 1]] == 0))[0]
  anchors = anchorable[rng.integers(len(anchorable), size=args.batch)]

  for mode, reachable in (('ballN_hazard', False), ('reachableN', True)):
    arm = lambda cls, probs_, seed_: run_arm(
        cls, obs, act, walls, probs_, tcfg, dist, args.gamma, seed_,
        anchors, MSWAMP, reachable)
    A = arm(MA.ManskiSampler, probs_smoothed, 10)     # table, smoothed
    A0 = arm(MA.ManskiSampler, probs_emp, 20)         # table, unsmoothed
    nulls = [arm(MA.ManskiSampler, probs_emp, 20 + 7 * m) for m in (1, 2, 3)]
    B = arm(AgreementCoinSampler, probs_emp, 50)      # event coin

    null_tv = [tv(A0['hist'], n['hist']) for n in nulls]
    tv_b = tv(A0['hist'], B['hist'])
    tv_smooth = tv(A0['hist'], A['hist'])
    thr = max(2.0 * max(null_tv), 0.02)
    gate(f'T2a_{mode}_endpoint_TV_at_null_level', tv_b <= thr,
         tv_agreement_vs_table=round(tv_b, 4),
         null_tv=[round(v, 4) for v in null_tv],
         tv_smoothing_effect=round(tv_smooth, 4), threshold=round(thr, 4))

    d_tf = abs(B['teleport_frac'] - A0['teleport_frac'])
    gate(f'T2b_{mode}_teleport_fraction_matches',
         d_tf <= max(0.01, 0.05 * A0['teleport_frac']),
         table=round(A0['teleport_frac'], 4),
         agreement=round(B['teleport_frac'], 4), abs_diff=round(d_tf, 4))

    d_bfs = abs(B['mean_bfs'] - A0['mean_bfs'])
    se = np.hypot(B['se_bfs'], A0['se_bfs'])
    gate(f'T2c_{mode}_pessimism_G7_stat_matches', d_bfs <= 4.0 * se + 0.02,
         table_mean_bfs=round(A0['mean_bfs'], 3),
         agreement_mean_bfs=round(B['mean_bfs'], 3),
         smoothed_table_mean_bfs=round(A['mean_bfs'], 3),
         diff=round(d_bfs, 4), se=round(float(se), 4),
         fallback_draws=B['n_fallback'])

  # =========================================================================
  # T3 -- three quantities, numerically (the reviewed confusion, as numbers)
  # =========================================================================
  n = 200_000
  beta = np.array([0.7, 0.3])
  pi = np.array([0.9, 0.1])
  s_beta = rng.choice(2, size=n, p=beta)     # "data" actions
  s_bc = rng.choice(2, size=n, p=beta)       # perfect BC model's actions
  s_pi = rng.choice(2, size=n, p=pi)         # target-policy actions

  agree_L = float(np.mean(s_bc == 0))                    # (a) frequency
  d_same = bayes_classifier(s_beta, s_bc, 2)             # (b) source, same
  d_pi = bayes_classifier(s_beta, s_pi, 2)               # (c) CFQL-style
  expect_c = beta / (beta + pi)
  gate('T3a_agreement_estimates_propensity',
       abs(agree_L - 0.7) < 0.005, value=round(agree_L, 4), expected=0.7)
  gate('T3b_source_classifier_same_dist_degenerates_to_half',
       np.all(np.abs(d_same - 0.5) < 0.01),
       value=[round(v, 4) for v in d_same],
       note='outputs 0.5, NOT beta=0.7 -- source classification is not '
            'propensity estimation')
  gate('T3c_source_classifier_vs_pi_is_density_ratio',
       np.all(np.abs(d_pi - expect_c) < 0.01),
       value=[round(v, 4) for v in d_pi],
       expected=[round(v, 4) for v in expect_c],
       note='CFQL D* = beta/(beta+pi): a discrepancy score, not beta')

  # real HOLDING cell replay of (a)+(b)
  hp = np.where(rows & (cells[:, 0] == holding[0])
                & (cells[:, 1] == holding[1]))[0]
  hb = bins[hp]
  half = rng.permutation(len(hb))
  d_real = bayes_classifier(hb[half[::2]], hb[half[1::2]], k)
  seen = np.bincount(hb, minlength=k) > 200
  agree_real = np.bincount(hb[rng.integers(len(hb), size=args.mc)],
                           minlength=k) / args.mc
  p_hold = p_emp[holding[0], holding[1]]
  gate('T3d_holding_cell_real_data',
       np.all(np.abs(d_real[seen] - 0.5) < 0.05)
       and np.all(np.abs(agree_real[seen] - p_hold[seen]) < 0.02),
       holding_cell=str(holding), bins_tested=int(seen.sum()),
       classifier_range=[round(float(d_real[seen].min()), 3),
                         round(float(d_real[seen].max()), 3)],
       max_agree_err=round(float(np.abs(agree_real - p_hold)[seen].max()), 4))

  # -------------------------------------------------------------------------
  report['all_passed'] = all(g['passed'] for g in report['gates'])
  with open(os.path.join(args.out, 'report.json'), 'w') as f:
    json.dump(report, f, indent=1)
  print(f"\nALL {'PASSED' if report['all_passed'] else 'FAILED'} -> "
        f"{args.out}/report.json", flush=True)


if __name__ == '__main__':
  main()
