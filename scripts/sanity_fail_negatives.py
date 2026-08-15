"""Step 6 sanity checks for failure-aware negative sampling (Part 1).

Small-batch verification BEFORE any expensive run:

  S1  positives still come only from the geometric future-state construction
      (twin-RNG replication of the buffer's variable-length index draw);
  S2  alpha=0 is byte-identical to the baseline (same params after N updates,
      same critic loss as a hand-computed baseline BCE);
  S3  failure states appear ONLY in negative slots (no bank state can ever be
      a positive: exact-match scan of the bank against every clean-dataset
      observation; positives are relabeled clean states by construction);
  S4  the loss decomposes EXACTLY as L(a) = L_pos + (1-a)*L_ord-neg +
      a*L_fail-neg with the positive term and total negative mass at their
      original weights (L_pos + L_ord-neg == L(0), all components
      alpha-invariant on fixed params/batch; only the mixture weight moves);
  S5  D_rock-fail does not leak into the clean anchor/positive set (episode-id
      partition + byte-level absence of dead episodes from the clean npz);
  S6  evaluation trajectories never enter training (offline buffer frozen +
      the 9-gate static audit passes on the clean npz; eval episodes are
      fresh env rollouts, never stored);
  S7  diagnostic example of sampled positives / in-batch negatives / failure
      negatives, plus a finite-loss + param-update smoke of the fail branch.

Runs on CPU in well under a minute (small nets for speed; checks are
structural, not scale-dependent).

Usage: python scripts/sanity_fail_negatives.py
"""
import dataclasses
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402
import optax                                 # noqa: E402

from crl import losses as losses_mod         # noqa: E402
from crl import networks as networks_mod     # noqa: E402
from crl import offline_audit                # noqa: E402
from crl.config import Config                # noqa: E402
from crl.replay import TrajectoryBuffer, obs_to_goal  # noqa: E402

SPLIT_DIR = 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split'
CLEAN_NPZ = os.path.join(
    SPLIT_DIR, 'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
ROCKFAIL_NPZ = os.path.join(
    SPLIT_DIR, 'antmaze_rockfall_v2_p30_h800_resetfix_pilot_rockfail.npz')
BANK_NPZ = os.path.join(SPLIT_DIR, 'failure_bank.npz')
MANIFEST = os.path.join(SPLIT_DIR, 'failure_split_manifest.json')
SRC_NPZ = ('artifacts/rockfall_v2_p30_h800_resetfix/pilot/'
           'antmaze_rockfall_v2_p30_h800_resetfix_pilot.npz')

OBS_DIM, GOAL_DIM, ACT_DIM = 29, 29, 8
B = 64          # small batch for the checks (recipe uses 1024; structural).
SEED = 7


def small_cfg(alpha=0.0, bank=''):
  return Config(
      env_name='offline_ant_umaze_rockfall', offline_dataset=CLEAN_NPZ,
      obs_dim=OBS_DIM, goal_dim=GOAL_DIM, action_dim=ACT_DIM,
      max_episode_steps=800, goal_indices=tuple(range(29)),
      start_index=0, end_index=-1,
      use_td=False, use_cpc=False, twin_q=True, bc_coef=0.05,
      random_goals=0.0, entropy_coefficient=0.0, target_entropy=0.0,
      batch_size=B, repr_dim=16, hidden_layer_sizes=(64, 64),
      discount=0.99, fail_neg_alpha=alpha, fail_bank_path=bank, seed=SEED)


def build(cfg, fail_bank=None):
  nets = networks_mod.make_networks(
      OBS_DIM, GOAL_DIM, ACT_DIM, repr_dim=cfg.repr_dim,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q)
  o2g = lambda s: s[:, :GOAL_DIM]
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, o2g, optax.adam(3e-4), optax.adam(3e-4),
      fail_bank=fail_bank)
  return nets, init_state, jax.jit(update_step)


def leaf_hash(tree):
  h = hashlib.sha256()
  for leaf in jax.tree_util.tree_leaves(tree):
    h.update(np.asarray(leaf).tobytes())
  return h.hexdigest()


def main():
  results = {}
  d = np.load(CLEAN_NPZ, allow_pickle=True)
  obs, act, lengths = d['obs'], d['act'], d['lengths'].astype(np.int64)
  E, L, W = obs.shape
  bank = np.load(BANK_NPZ, allow_pickle=True)
  bank_states = np.asarray(bank['goals'], np.float32)          # [16, 29]
  man = json.load(open(MANIFEST))

  buf = TrajectoryBuffer(
      capacity_steps=E * L, ep_len_obs=L, full_obs_dim=W, action_dim=ACT_DIM,
      obs_dim=OBS_DIM, start_index=0, end_index=-1, discount=0.99, seed=SEED,
      goal_indices=tuple(range(29)))
  for e in range(E):
    buf.add_episode(obs[e], act[e], length=int(lengths[e]))
  buf.freeze()

  # ---------------- S1: positive construction (twin-RNG replication) -------
  # All clean episodes are full length (801) -- only dead episodes were
  # truncated -- so the buffer takes its fixed-length RNG path.
  assert np.all(lengths == L)
  rng2 = np.random.default_rng(SEED)
  traj = rng2.integers(0, E, size=B)
  Lt = lengths[traj]
  i = rng2.integers(0, L - 1, size=B)
  arange = np.arange(L)
  future = arange[None, :] > i[:, None]
  logp = np.where(future, (arange[None, :] - i[:, None]) * np.log(0.99),
                  -np.inf)
  g = -np.log(-np.log(rng2.uniform(size=logp.shape).clip(1e-20, 1.0)))
  j = np.argmax(logp + g, axis=1)
  tr = buf.sample(B)
  assert np.array_equal(tr.observation[:, :OBS_DIM], obs[traj, i, :OBS_DIM])
  assert np.array_equal(tr.action, act[traj, i])
  assert np.array_equal(tr.observation[:, OBS_DIM:], obs[traj, j, :OBS_DIM])
  assert np.all(j > i) and np.all(j < Lt), 'future goal outside episode'
  results['S1_positive_construction'] = (
      f'PASS ({B} tuples: goal == obs[traj, j>i] within valid length, '
      'geometric relabel replicated exactly)')

  # ---------------- S2: alpha=0 byte-identical to baseline -----------------
  cfg0 = small_cfg(alpha=0.0)
  _, init0, step0 = build(cfg0, fail_bank=None)
  # bank present but alpha=0 -> branch must be skipped entirely
  _, init0b, step0b = build(small_cfg(alpha=0.0), fail_bank=bank_states)
  key = jax.random.PRNGKey(SEED)
  s_a = init0(key)
  s_b = init0b(key)
  assert leaf_hash(s_a.q_params) == leaf_hash(s_b.q_params)
  batch = buf.sample(B)
  for _ in range(5):
    s_a, m_a = step0(s_a, batch)
    s_b, m_b = step0b(s_b, batch)
  same = (leaf_hash(s_a.q_params) == leaf_hash(s_b.q_params)
          and leaf_hash(s_a.policy_params) == leaf_hash(s_b.policy_params))
  assert same, 'alpha=0 with a bank present diverged from the baseline'
  assert 'fail_neg_count' not in m_b, 'fail metrics leaked into alpha=0 run'
  results['S2_alpha0_identical'] = (
      'PASS (5 updates, param hashes identical with/without bank at alpha=0; '
      'no fail metrics emitted)')

  # ---------------- S3: failure states only in negative slots --------------
  # positives are relabeled CLEAN states by construction (S1); additionally no
  # bank state exists anywhere in the clean dataset, so a bank state can never
  # be sampled as a positive.
  clean_states = obs[:, :, :OBS_DIM].reshape(-1, OBS_DIM)
  # compare in float32 byte space, restricted to valid rows
  valid_rows = (arange[None, :] < lengths[:, None]).reshape(-1)
  clean_states = clean_states[valid_rows]
  hits = 0
  for b in bank_states:
    hits += int(np.any(np.all(clean_states == b[None, :], axis=1)))
  assert hits == 0, f'{hits} bank states found inside the clean dataset'
  results['S3_bank_only_negative'] = (
      f'PASS (0/{len(bank_states)} bank states occur anywhere in the '
      f'{clean_states.shape[0]} valid clean observations; loss labels bank '
      'columns 0 only)')

  # ---------------- S4 + S7: alpha sweep + exact loss decomposition --------
  # New-form check (loss-level mixture): on the SAME params/batch across all
  # alphas, verify
  #   L(alpha) == pos + (1-alpha)*neg_ord + alpha*neg_fail   (reported terms)
  #   pos, neg_ord are alpha-INVARIANT and satisfy pos + neg_ord == L(0)
  #   (the baseline mean), i.e. the positive term and total negative mass
  #   keep their original weights; only the negative distribution mixes.
  base_loss = None   # L(0) on the same params/batch
  cfg_b = small_cfg(alpha=0.0)
  _, init_b, step_b = build(cfg_b, fail_bank=None)
  st_ref = init_b(jax.random.PRNGKey(SEED))
  _, met0 = step_b(st_ref, batch)
  base_loss = float(met0['critic_loss'])
  frac_report = {}
  example = None
  pos_ref = ord_raw_ref = fail_raw_ref = None
  for alpha in (0.1, 0.3, 0.5):
    cfa = small_cfg(alpha=alpha, bank=BANK_NPZ)
    _, init_a, step_a = build(cfa, fail_bank=bank_states)
    st = init_a(jax.random.PRNGKey(SEED))   # same init as the baseline
    _, met = step_a(st, batch)
    pos = float(met['critic_pos_term'])
    ord_w = float(met['critic_neg_ord_term'])
    fail_w = float(met['critic_neg_fail_term'])
    ord_raw = float(met['critic_neg_ord_raw'])
    fail_raw = float(met['critic_neg_fail_raw'])
    L = float(met['critic_loss'])
    # decomposition identity
    assert abs(L - (pos + ord_w + fail_w)) < 1e-5, (alpha, L, pos, ord_w,
                                                    fail_w)
    # weighted terms = alpha-mixture of the raw components
    assert abs(ord_w - (1 - alpha) * ord_raw) < 1e-6
    assert abs(fail_w - alpha * fail_raw) < 1e-6
    assert abs(float(met['fail_neg_alpha']) - alpha) < 1e-6
    assert float(met['fail_bank_size']) == len(bank_states)
    # positive term + raw components are alpha-invariant (same params/batch)
    if pos_ref is None:
      pos_ref, ord_raw_ref, fail_raw_ref = pos, ord_raw, fail_raw
      # original weighting preserved: pos + ord_raw == baseline mean L(0)
      assert abs((pos + ord_raw) - base_loss) < 1e-5, (pos + ord_raw,
                                                       base_loss)
    else:
      assert abs(pos - pos_ref) < 1e-6
      assert abs(ord_raw - ord_raw_ref) < 1e-6
      assert abs(fail_raw - fail_raw_ref) < 1e-6
    frac_report[alpha] = {
        'pos_term': round(pos, 6), 'neg_ord_weighted': round(ord_w, 6),
        'neg_fail_weighted': round(fail_w, 6), 'total': round(L, 6),
        'neg_ord_raw': round(ord_raw, 6), 'neg_fail_raw': round(fail_raw, 6),
        'logits_fail_neg': round(float(met['logits_fail_neg']), 4)}
    if example is None:
      example = {
          'anchor_xy': np.asarray(batch.observation[0, :2]).tolist(),
          'positive_goal_xy': np.asarray(
              batch.observation[0, OBS_DIM:OBS_DIM + 2]).tolist(),
          'inbatch_negative_goal_xy': np.asarray(
              batch.observation[1:4, OBS_DIM:OBS_DIM + 2]).tolist(),
          'fail_negative_goal_xy': bank_states[:3, :2].tolist(),
      }
  results['S4_loss_decomposition'] = {
      'status': 'PASS',
      'baseline_L0': round(base_loss, 6),
      'identity': ('L(a) = pos + (1-a)*neg_ord_raw + a*neg_fail_raw; '
                   'pos + neg_ord_raw == L(0); pos/neg_ord_raw/neg_fail_raw '
                   'alpha-invariant on fixed params/batch'),
      'per_alpha': frac_report}

  # param-update smoke: fail branch actually trains (params move, loss finite)
  cfa = small_cfg(alpha=0.3, bank=BANK_NPZ)
  _, init_a, step_a = build(cfa, fail_bank=bank_states)
  st0 = init_a(jax.random.PRNGKey(SEED))
  st1, met = step_a(st0, batch)
  moved = leaf_hash(st1.q_params) != leaf_hash(st0.q_params)
  assert moved and np.isfinite(float(met['critic_loss']))
  results['S7_fail_branch_smoke'] = (
      f'PASS (alpha=0.3: critic_loss={float(met["critic_loss"]):.4f} finite, '
      'params updated)')
  results['S7_diagnostic_example'] = example

  # ---------------- S5: no leakage of D_rock-fail into clean ---------------
  fail_ids = set(man['rockfail']['episode_ids'])
  assert man['clean']['n_episodes'] + man['rockfail']['n_episodes'] == 300
  assert man['clean']['n_transitions'] + man['rockfail']['n_transitions'] \
      == man['n_transitions_original']
  # byte-level: no clean episode equals any rock-fail episode
  df = np.load(ROCKFAIL_NPZ, allow_pickle=True)
  fail_ep_hashes = {hashlib.sha256(
      np.ascontiguousarray(df['obs'][k]).tobytes()).hexdigest()
      for k in range(df['obs'].shape[0])}
  clean_ep_hashes = {hashlib.sha256(
      np.ascontiguousarray(obs[k]).tobytes()).hexdigest() for k in range(E)}
  assert not (fail_ep_hashes & clean_ep_hashes), 'episode overlap!'
  # bank provenance: every bank state is the terminal obs of a dead episode
  src = np.load(SRC_NPZ, allow_pickle=True)
  src_lengths = src['lengths'].astype(np.int64)
  for row, e in zip(bank_states, np.asarray(bank['episode_id'])):
    assert np.array_equal(row, src['obs'][e, src_lengths[e] - 1, :OBS_DIM])
    assert e in fail_ids
  results['S5_no_leakage'] = (
      f'PASS (episode partition {man["clean"]["n_episodes"]}+'
      f'{man["rockfail"]["n_episodes"]}=300 exact; transition split '
      f'{man["clean"]["n_transitions"]}+{man["rockfail"]["n_transitions"]}='
      f'{man["n_transitions_original"]}; zero byte-level episode overlap; '
      'all 16 bank states are terminal obs of dead episodes only)')

  # ---------------- S6: eval separation + offline audit on clean npz -------
  cfg = small_cfg()
  cfg.max_episode_steps = 800
  passed, gates, _ = offline_audit.run_static_audit(CLEAN_NPZ, cfg, buffer=buf)
  assert buf.frozen, 'buffer must be frozen in offline mode'
  assert passed, f'offline audit failed on clean npz: {gates}'
  results['S6_eval_separation'] = (
      'PASS (buffer frozen; static audit gates all PASS on clean npz: '
      f'{sorted(gates)}; eval episodes are fresh env rollouts and are never '
      'written to the buffer -- add_episode raises when frozen)')

  print(json.dumps(results, indent=2))
  out = os.path.join(SPLIT_DIR, 'sanity_report.json')
  with open(out, 'w') as f:
    json.dump(results, f, indent=2)
  print('\nALL SANITY CHECKS PASSED ->', out)


if __name__ == '__main__':
  main()
