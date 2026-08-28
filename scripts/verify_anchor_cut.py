"""Verification for the windy-swamp pipeline:

    original 6000 episodes retained  ->  anchors by scheme C
    ->  ordinary in-batch negatives unchanged  ->  + failure-state negatives.

Checks (all CPU, well under a minute):

  V1  DEFAULT PATH UNTOUCHED. With anchor_cut off the buffer consumes the RNG
      exactly as before -- traj=integers(0,ne), i=integers(0,L-1), then the
      Gumbel draw -- reproduced bit-for-bit from a fresh Generator.
  V2  SCHEME C LAW. Episodes uniform; row uniform inside [0, cut); never >=cut.
      Checked on a synthetic buffer with deliberately uneven cuts.
  V3  CUT BITES ON THE REAL DATASET, and goals are STILL drawn at/past the cut
      -- the half that separates scheme C from the 'lengths' path.
  V4  RELABELING LAW UNCHANGED. Conditional P(j|i) still matches
      discount**(j-i) normalised over the FULL row range, not a truncated one.
  V5  MEASURED EFFECT on the real dataset (anchor composition, decision-point
      density) -- reproduces the numbers the design decision was made on.
  V6  BANK CONTRACT: dtype/shape, goal-space width, n_bank <= batch_size,
      every state inside a swamp cell.
  V7  BANK-vs-POSITIVE OVERLAP (INFORMATIONAL, not a gate). Unlike the AntMaze
      bank, these states DO occur on retained trajectories; the overlap is
      measured and reported so it stays visible, not asserted away.
  V8  LOSS ALGEBRA: pos + (1-a)*ord + a*fail == critic_loss exactly, and
      alpha=0 with a bank present is byte-identical to the baseline.

Usage: python scripts/verify_anchor_cut.py
"""
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import jax                                    # noqa: E402
import optax                                  # noqa: E402

from crl import losses as losses_mod          # noqa: E402
from crl import networks as networks_mod      # noqa: E402
from crl import offline_audit                 # noqa: E402
from crl.config import Config                 # noqa: E402
from crl.replay import TrajectoryBuffer       # noqa: E402

NPZ = 'datasets/swamp_windy_teacher_s0.npz'
BANK = 'artifacts/swamp_windy_failure_bank/failure_bank.npz'
ENV = 'point_two_route_swamp_windy_v0'
OBS_DIM = GOAL_DIM = ACT_DIM = 2
SWAMP = [(3, 3), (4, 3), (5, 3)]
FORK, HOLD = (1, 3), (2, 3)
SEED = 7
RESULTS = {}
FAILED = []


def check(name, ok, detail):
  RESULTS[name] = {'pass': bool(ok), 'detail': detail}
  print(f'  {"PASS" if ok else "FAIL"}  {name}: {detail}')
  if not ok:
    FAILED.append(name)


def leaf_hash(tree):
  h = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    h.update(np.asarray(leaf).tobytes())
  return h.hexdigest()


def make_buffer(obs, act, cuts=None, seed=SEED):
  E, L, W = obs.shape
  b = TrajectoryBuffer(
      capacity_steps=E * L, ep_len_obs=L, full_obs_dim=W, action_dim=ACT_DIM,
      obs_dim=OBS_DIM, start_index=0, end_index=-1, discount=0.99, seed=seed)
  for e in range(E):
    b.add_episode(obs[e], act[e])
  if cuts is not None:
    b.set_anchor_cuts(cuts)
  b.freeze()
  return b


def main():
  d = np.load(NPZ, allow_pickle=True)
  obs, act = d['obs'], d['act']
  bits = d['swamp_bits']
  mode = np.asarray(d['teacher_mode'])
  E, L, W = obs.shape
  cell = np.clip(np.floor(obs[:, :, :2]).astype(int), 0, [8, 4])
  B = 4096

  cfg = Config(env_name=ENV, obs_dim=OBS_DIM, goal_dim=GOAL_DIM,
               action_dim=ACT_DIM, max_episode_steps=L - 1,
               start_index=0, end_index=-1, anchor_cut_mode='arrival',
               anchor_cut_radius=0.5)
  cuts, cut_stats = offline_audit.compute_anchor_cuts(obs, OBS_DIM, 0.5)

  # ---------------------------------------------------------------- V1
  print('\nV1  default path untouched')
  b_off = make_buffer(obs, act)
  traj_b, i_b, j_b = b_off.sampled_indices(B)
  r = np.random.default_rng(SEED)
  traj_r = r.integers(0, E, size=B)
  i_r = r.integers(0, L - 1, size=B)
  ar = np.arange(L)
  logp = np.where(ar[None, :] > i_r[:, None],
                  (ar[None, :] - i_r[:, None]) * np.log(0.99), -np.inf)
  g = -np.log(-np.log(r.uniform(size=logp.shape).clip(1e-20, 1.0)))
  j_r = np.argmax(logp + g, axis=1)
  ok = (np.array_equal(traj_b, traj_r) and np.array_equal(i_b, i_r)
        and np.array_equal(j_b, j_r))
  check('V1_default_rng_identical', ok,
        f'{B} draws reproduce traj/i/j bit-for-bit from a fresh Generator '
        f'(use_anchor_cut={b_off.use_anchor_cut})')

  # ---------------------------------------------------------------- V2
  print('\nV2  scheme C law (synthetic, uneven cuts)')
  nE, nL = 40, 21
  sobs = np.random.default_rng(0).normal(size=(nE, nL, 2 * OBS_DIM)).astype(
      np.float32)
  sact = np.zeros((nE, nL, ACT_DIM), np.float32)
  scuts = np.array([1 + (e % (nL - 1)) for e in range(nE)], np.int64)
  sb = make_buffer(sobs, sact, cuts=scuts, seed=3)
  N = 400000
  st, si, sj = sb.sampled_indices(N)
  viol = int(np.sum(si >= scuts[st]))
  ep_cnt = np.bincount(st, minlength=nE)
  ep_dev = float(np.abs(ep_cnt / (N / nE) - 1).max())
  row_dev = 0.0
  for e in range(nE):
    m = st == e
    c = np.bincount(si[m], minlength=scuts[e])[:scuts[e]]
    exp = m.sum() / scuts[e]
    if exp > 200:                       # only judge well-populated episodes
      row_dev = max(row_dev, float(np.abs(c / exp - 1).max()))
  ok = viol == 0 and ep_dev < 0.06 and row_dev < 0.12
  check('V2_scheme_C_law', ok,
        f'{N} draws: anchors past cut = {viol}; episode-uniform max dev '
        f'{ep_dev:.3f}; within-episode row-uniform max dev {row_dev:.3f}')

  # ---------------------------------------------------------------- V3
  print('\nV3  cut bites on the real dataset; future window NOT truncated')
  b_on = make_buffer(obs, act, cuts=cuts)
  t2, i2, j2 = b_on.sampled_indices(200000)
  past_anchor = int(np.sum(i2 >= cuts[t2]))
  past_goal = int(np.sum(j2 >= cuts[t2]))
  ok = past_anchor == 0 and past_goal > 0
  check('V3_cut_and_full_future', ok,
        f'anchors past cut = {past_anchor} (must be 0); goals at/past cut = '
        f'{past_goal:,} = {past_goal/len(j2):.1%} (must be > 0, proving the '
        f'goal window is still the full {L} rows)')
  ok10, s10 = offline_audit.check_anchor_cut(b_on)
  check('V3b_G10_gate', ok10, json.dumps(s10))

  # ---------------------------------------------------------------- V4
  # A/B against the DEFAULT buffer rather than against an absolute tolerance:
  # with ~50 bins the multinomial noise floor on total variation is itself
  # ~0.02, so an absolute threshold would only be measuring sample size.
  print('\nV4  relabeling law unchanged (A/B vs the default sampler)')
  i0 = 0
  NBIG = 1_500_000
  _, i_off, j_off = b_off.sampled_indices(NBIG)
  _, i_on, j_on = b_on.sampled_indices(NBIG)

  def law(i_arr, j_arr):
    m = i_arr == i0
    c = np.bincount(j_arr[m], minlength=L)[i0 + 1:].astype(float)
    return c / c.sum(), int(m.sum()), int(j_arr[m].max())

  p_off, n_off, max_off = law(i_off, j_off)
  p_on, n_on, max_on = law(i_on, j_on)
  th = 0.99 ** (np.arange(i0 + 1, L) - i0)
  th /= th.sum()
  tv = lambda a, b: 0.5 * float(np.abs(a - b).sum())
  # analytic multinomial noise floor for the SMALLER of the two samples
  floor = 0.5 * float(np.sum(np.sqrt(2 * th * (1 - th)
                                     / (np.pi * min(n_off, n_on)))))
  d_ab, d_on, d_off = tv(p_on, p_off), tv(p_on, th), tv(p_off, th)
  ok = (d_ab < 4 * floor and d_on < 4 * floor and max_on == L - 1
        and max_off == L - 1)
  check('V4_geometric_law', ok,
        f'anchors at i={i0} (n_on={n_on:,}, n_off={n_off:,}); noise floor '
        f'{floor:.5f}; TV(schemeC, default) = {d_ab:.5f}; '
        f'TV(schemeC, discount**(j-i)) = {d_on:.5f}; TV(default, law) = '
        f'{d_off:.5f}; max j reached {max_on} of {L-1} in both')

  # ---------------------------------------------------------------- V5
  print('\nV5  measured effect on the real dataset')
  def shares(t, i):
    c = cell[t, i]
    dec = (((c[:, 0] == FORK[0]) & (c[:, 1] == FORK[1]))
           | ((c[:, 0] == HOLD[0]) & (c[:, 1] == HOLD[1])))
    ha = ((c[:, 0] == HOLD[0]) & (c[:, 1] == HOLD[1])
          & bits[t, i, 0].astype(bool))
    src = {int(k): float((mode[t] == k).mean()) for k in range(4)}
    return dec.mean(), ha.mean(), src
  # Same sample size on both arms: at B=4096 the composition estimate alone
  # carries ~0.8pp of noise, which would swamp the comparison.
  NCMP = 400000
  t_a, i_a, _ = b_off.sampled_indices(NCMP)
  t3, i3, _ = b_on.sampled_indices(NCMP)
  dec_a, ha_a, src_a = shares(t_a, i_a)
  dec_c, ha_c, src_c = shares(t3, i3)
  comp_shift = max(abs(src_a[k] - src_c[k]) for k in src_a)
  comp_ok = comp_shift < 0.01      # A and C share the episode marginal.
  ok = dec_c > 3 * dec_a and comp_ok
  check('V5_effect', ok,
        f'decision-point anchors {dec_a:.2%} -> {dec_c:.2%} '
        f'({dec_c/dec_a:.2f}x, {1024*dec_c:.0f} per batch of 1024); '
        f'holding+swamp0-active {ha_a:.3%} -> {ha_c:.3%} '
        f'({1024*ha_c:.1f} per batch); source composition preserved '
        f'(max shift {comp_shift:.4f} over {NCMP:,} draws each, '
        f'random {src_a[0]:.1%}->{src_c[0]:.1%}) -- scheme C keeps the '
        f'episode marginal, unlike a flat draw over acting rows')

  # ---------------------------------------------------------------- V6
  print('\nV6  bank contract')
  bk = np.load(BANK, allow_pickle=True)
  bank_states = np.asarray(bk['goals'], np.float32)
  bcell = np.clip(np.floor(bank_states).astype(int), 0, [8, 4])
  in_swamp = np.array([tuple(c) in SWAMP for c in bcell])
  ok = (bank_states.ndim == 2 and bank_states.shape[1] == OBS_DIM
        and bank_states.dtype == np.float32 and in_swamp.all())
  check('V6_bank_contract', ok,
        f'{bank_states.shape} float32, all {int(in_swamp.sum())}/'
        f'{len(bank_states)} states inside a swamp cell; '
        f'REQUIRES batch_size >= {len(bank_states)}')

  # ---------------------------------------------------------------- V7
  # scipy is NOT in the node environment (scripts/failneg_h800_node_setup.sh),
  # so this falls back to a chunked numpy scan -- the bank is only 514 points.
  print('\nV7  bank-vs-positive overlap (INFORMATIONAL, not a gate)')
  t4, _, j4 = b_on.sampled_indices(200000)
  gs = obs[t4, j4, :2].astype(np.float64)
  radii = (0.02, 0.05, 0.1)
  try:
    from scipy.spatial import cKDTree
    tree = cKDTree(bank_states)
    over = {f'{r}': float((tree.query_ball_point(gs, r, return_length=True) > 0)
                          .mean()) for r in radii}
  except ImportError:
    bs = bank_states.astype(np.float64)
    hits = {r: 0 for r in radii}
    for s in range(0, len(gs), 20000):
      chunk = gs[s:s + 20000]
      dmin = np.sqrt(((chunk[:, None, :] - bs[None, :, :]) ** 2).sum(2)).min(1)
      for r in radii:
        hits[r] += int((dmin < r).sum())
    over = {f'{r}': hits[r] / len(gs) for r in radii}
  RESULTS['V7_bank_positive_overlap'] = {'pass': None, 'detail': over}
  print(f'  INFO  V7_bank_positive_overlap: fraction of relabeled POSITIVE '
        f'goals within r of some bank state: ' +
        ', '.join(f'r={k} -> {v:.3%}' for k, v in over.items()))
  print('        (the AntMaze bank had zero such overlap by construction; here '
        'the dataset is not split, so this is expected and is the known '
        'positive/negative tension -- kept visible on purpose)')

  # ---------------------------------------------------------------- V8
  print('\nV8  loss algebra with the bank')
  nb = 32
  sub = bank_states[np.linspace(0, len(bank_states) - 1, nb).astype(int)]
  BB = 64

  def small_cfg(alpha, bank_path=''):
    return Config(env_name=ENV, obs_dim=OBS_DIM, goal_dim=GOAL_DIM,
                  action_dim=ACT_DIM, max_episode_steps=L - 1,
                  start_index=0, end_index=-1, use_td=False, use_cpc=False,
                  twin_q=True, bc_coef=0.05, random_goals=0.0,
                  entropy_coefficient=0.0, target_entropy=0.0,
                  batch_size=BB, repr_dim=16, hidden_layer_sizes=(64, 64),
                  discount=0.99, fail_neg_alpha=alpha,
                  fail_bank_path=bank_path, seed=SEED)

  def build(c, bank=None):
    nets = networks_mod.make_networks(
        OBS_DIM, GOAL_DIM, ACT_DIM, repr_dim=c.repr_dim,
        hidden_layer_sizes=c.hidden_layer_sizes, twin_q=c.twin_q)
    o2g = lambda s: s[:, :GOAL_DIM]
    init, step = losses_mod.build_learner(
        nets, c, o2g, optax.adam(3e-4), optax.adam(3e-4), fail_bank=bank)
    return init, jax.jit(step)

  batch = b_on.sample(BB)
  key = jax.random.PRNGKey(SEED)
  alpha = 0.1
  init_f, step_f = build(small_cfg(alpha), bank=sub)
  s = init_f(key)
  s, m = step_f(s, batch)
  tot = float(m['critic_pos_term'] + m['critic_neg_ord_term']
              + m['critic_neg_fail_term'])
  lhs = float(m['critic_loss'])
  ok = abs(tot - lhs) < 1e-5 and int(m['fail_bank_size']) == nb
  check('V8a_loss_decomposition', ok,
        f'pos {float(m["critic_pos_term"]):.6f} + ord '
        f'{float(m["critic_neg_ord_term"]):.6f} + fail '
        f'{float(m["critic_neg_fail_term"]):.6f} = {tot:.6f} vs critic_loss '
        f'{lhs:.6f} (|diff| {abs(tot-lhs):.2e}); bank size '
        f'{int(m["fail_bank_size"])}, alpha {float(m["fail_neg_alpha"])}')

  init_a, step_a = build(small_cfg(0.0), bank=None)
  init_b, step_b = build(small_cfg(0.0), bank=sub)
  sa, sb2 = init_a(key), init_b(key)
  for _ in range(5):
    sa, ma = step_a(sa, batch)
    sb2, mb = step_b(sb2, batch)
  ok = (leaf_hash(sa.q_params) == leaf_hash(sb2.q_params)
        and leaf_hash(sa.policy_params) == leaf_hash(sb2.policy_params)
        and 'fail_neg_alpha' not in mb)
  check('V8b_alpha0_identical', ok,
        '5 updates: params byte-identical with and without a bank at alpha=0; '
        'no fail metrics emitted')

  # ---------------------------------------------------------------- V9
  print('\nV9  balanced (s,a) sampling')
  tj, rw, bkt, bal_stats = offline_audit.compute_balanced_buckets(
      obs, act, OBS_DIM, 1.0, 4, cuts=cuts)
  b_bal = TrajectoryBuffer(
      capacity_steps=E * L, ep_len_obs=L, full_obs_dim=W, action_dim=ACT_DIM,
      obs_dim=OBS_DIM, start_index=0, end_index=-1, discount=0.99, seed=SEED)
  for e in range(E):
    b_bal.add_episode(obs[e], act[e])
  b_bal.set_anchor_cuts(cuts)
  b_bal.set_balanced_buckets(tj, rw, bkt, cap=300)
  b_bal.freeze()
  tb, ib, jb = b_bal.sampled_indices(300000)

  # (a) anchors still respect the cut, (b) goals still span the full window
  past_cut = int(np.sum(ib >= cuts[tb]))
  reach_end = int(jb.max())

  def fork_ratio(t, i):
    c = cell[t, i]
    nx = cell[t, i + 1]
    at = (c[:, 0] == FORK[0]) & (c[:, 1] == FORK[1])
    f = int((at & (nx[:, 0] == HOLD[0]) & (nx[:, 1] == HOLD[1])).sum())
    s = int((at & (nx[:, 0] == 1) & (nx[:, 1] == 2)).sum())
    return f / max(s, 1)

  r_cut = fork_ratio(t3, i3)                       # scheme C alone (from V5)
  r_bal = fork_ratio(tb, ib)
  ok = (past_cut == 0 and reach_end == L - 1 and r_bal < r_cut / 3)
  check('V9_balanced_sampling', ok,
        f'{bal_stats["n_buckets"]} buckets (cap 300, rotated sectors + wait); '
        f'anchors past cut {past_cut} (must be 0); max j {reach_end} of {L-1} '
        f'(future window intact); fork shortcut:safe {r_cut:.2f}x -> '
        f'{r_bal:.2f}x')

  # ---------------------------------------------------------------- report
  out = 'artifacts/swamp_windy_failure_bank/verify_anchor_cut.json'
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, 'w') as f:
    json.dump({'results': RESULTS, 'anchor_cut_stats': cut_stats,
               'dataset': NPZ, 'bank': BANK}, f, indent=2)
  print(f'\nanchor-cut stats: {json.dumps(cut_stats)}')
  print(f'report -> {out}')
  if FAILED:
    print(f'\n{len(FAILED)} CHECK(S) FAILED: {FAILED}')
    return 1
  print(f'\nALL {len(RESULTS)-1} GATES PASS (V7 is informational).')
  return 0


if __name__ == '__main__':
  sys.exit(main())
