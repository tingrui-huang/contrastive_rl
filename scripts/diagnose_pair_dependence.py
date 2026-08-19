"""Task section 11: is the same failure-like goal handled PAIR-DEPENDENTLY?

The pessimistic branch can hand a settled-failure-like state to Critic C as a
POSITIVE goal for a risky anchor, while the SAME kind of state is a failure
NEGATIVE for unrelated anchors. That is only a contradiction if the critic
scores a goal globally, ignoring the anchor. This measures whether it does.

  Q1  f_C(s_i, a_i, g^f_b) across anchors i for a FIXED bank goal b -- if the
      spread across anchors is large, the score is anchor-dependent, not a
      per-goal constant.
  Q2  two-way variance decomposition of f over (anchor x goal): how much of
      the variance is the goal main effect (global) vs the anchor main effect
      and the anchor-goal interaction (pair-dependent).
  Q3  risky vs safe anchors, same 16 failure goals. "Risky" = the anchor's own
      frozen s'_wc lands close to the negative bank. If risky anchors score
      failure goals HIGHER than safe anchors do, the critic already resolves
      the same goal differently by anchor -- exactly the pair-dependence the
      integration needs.
  Q4  the pessimistic-positive score f_C(s_i, a_i, obs_to_goal(s'_wc,i))
      against the failure-negative score for unrelated anchors.

Diagnostic only. Nothing is trained; Critic C is read-only and never enters
worst-case selection.

Usage:  python scripts/diagnose_pair_dependence.py [--n-anchors 512]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import jax                                        # noqa: E402
import jax.numpy as jnp                           # noqa: E402
from crl import static_worstcase as sw            # noqa: E402
from crl import networks as networks_mod          # noqa: E402
from crl import checkpoint as ckpt_mod            # noqa: E402
from crl.replay import obs_to_goal                # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT = os.path.join(_ROOT, 'artifacts/static_worstcase_rl')
CLEAN = os.path.join(
    _ROOT, 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
    'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
C_CKPT = os.path.join(
    _ROOT, 'failneg_settledbank_a01_s0_300k/'
    'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl')
OBS_DIM = 29


def qstats(x):
  x = np.asarray(x, np.float64).ravel()
  return {'mean': float(x.mean()), 'std': float(x.std()),
          'median': float(np.median(x)), 'p10': float(np.percentile(x, 10)),
          'p90': float(np.percentile(x, 90))}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--n-anchors', type=int, default=512)
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)

  # ---- anchors -----------------------------------------------------------
  with np.load(CLEAN, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    lengths = np.asarray(d['lengths'], np.int64)
  rng = np.random.default_rng(args.seed)
  E = obs.shape[0]
  e = rng.integers(0, E, size=args.n_anchors)
  t = np.array([rng.integers(0, int(lengths[x]) - 1) for x in e])
  S = obs[e, t, :OBS_DIM]
  A = act[e, t]

  # ---- frozen worst-case state per anchor --------------------------------
  m = sw.StaticWorstCase(root=_ROOT)
  s_wc = np.zeros_like(S)
  d_neg = np.zeros(len(S), np.float32)
  root = jax.random.PRNGKey(sw.SEED)
  CH = 256
  for c in range(int(np.ceil(len(S) / CH))):
    lo, hi = c * CH, min((c + 1) * CH, len(S))
    o, aux = m.worst_case_next_state(S[lo:hi], A[lo:hi], return_aux=True,
                                     key=jax.random.fold_in(root, c))
    s_wc[lo:hi] = o
    d_neg[lo:hi] = aux['d_neg']

  # ---- Critic C (read-only) ----------------------------------------------
  cfg = build_offline_cfg()
  nets = networks_mod.make_networks(
      obs_dim=OBS_DIM, goal_dim=OBS_DIM, action_dim=8,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
  _, c_state = ckpt_mod.load_checkpoint(C_CKPT)

  @jax.jit
  def c_diag(og, a, pp=c_state.q_params):
    qv = nets.q_network.apply(pp, og, a)
    return jnp.diagonal(qv, axis1=0, axis2=1).T

  def score(states, goals, actions):
    g = np.asarray(obs_to_goal(goals.astype(np.float32), 0, -1,
                               tuple(range(OBS_DIM))), np.float32)
    return np.asarray(c_diag(jnp.asarray(np.concatenate([states, g], 1)),
                             jnp.asarray(actions))).min(1)

  bank = np.asarray(np.load(os.path.join(_ROOT, sw.BANK_NPZ),
                            allow_pickle=True)['goals'], np.float32)
  n, nb = len(S), len(bank)

  # F[i, b] = f_C(s_i, a_i, g^f_b)
  F = np.zeros((n, nb), np.float32)
  for b in range(nb):
    F[:, b] = score(S, np.repeat(bank[b][None], n, axis=0), A)
  # pessimistic-positive score: each anchor against its OWN s'_wc
  f_own = score(S, s_wc, A)

  # ---- Q1 spread across anchors for a fixed goal -------------------------
  per_goal_anchor_std = F.std(axis=0)          # [nb]
  per_anchor_goal_std = F.std(axis=1)          # [n]
  rep = {'n_anchors': n, 'n_bank_goals': nb,
         'Q1_fixed_goal_spread_across_anchors': {
             'per_goal_std_over_anchors': qstats(per_goal_anchor_std),
             'per_anchor_std_over_goals': qstats(per_anchor_goal_std),
             'interpretation': (
                 'A globally-scored goal would give ~0 std across anchors. '
                 'Non-trivial per-goal spread over anchors means f depends on '
                 'the anchor, i.e. the label is pair-dependent.')}}

  # ---- Q2 two-way variance decomposition ---------------------------------
  grand = F.mean()
  a_eff = F.mean(1) - grand                    # anchor main effect
  g_eff = F.mean(0) - grand                    # goal main effect
  resid = F - grand - a_eff[:, None] - g_eff[None, :]
  ss_a = float((a_eff ** 2).sum() * nb)
  ss_g = float((g_eff ** 2).sum() * n)
  ss_r = float((resid ** 2).sum())
  ss_t = ss_a + ss_g + ss_r
  rep['Q2_variance_decomposition'] = {
      'goal_main_effect_frac': ss_g / ss_t,
      'anchor_main_effect_frac': ss_a / ss_t,
      'anchor_goal_interaction_frac': ss_r / ss_t,
      'pair_dependent_frac': (ss_a + ss_r) / ss_t,
      'interpretation': (
          'goal_main_effect_frac ~ 1 would mean the critic assigns each goal a '
          'global score and the two roles WOULD conflict. Anchor + '
          'interaction mass is the pair-dependent part.')}

  # ---- Q3 risky vs safe anchors, SAME goals ------------------------------
  q1, q3 = np.percentile(d_neg, [25, 75])
  risky = d_neg <= q1                          # own s_wc lands near the bank
  safe = d_neg >= q3
  fr, fs = F[risky].mean(1), F[safe].mean(1)
  boot = [float(rng.choice(fr, len(fr)).mean() - rng.choice(fs, len(fs)).mean())
          for _ in range(2000)]
  rep['Q3_risky_vs_safe_same_failure_goals'] = {
      'risky_definition': 'anchor d_neg(own s_wc) <= p25 (%.3f)' % q1,
      'safe_definition': 'anchor d_neg(own s_wc) >= p75 (%.3f)' % q3,
      'n_risky': int(risky.sum()), 'n_safe': int(safe.sum()),
      'mean_f_failure_goals_risky': float(fr.mean()),
      'mean_f_failure_goals_safe': float(fs.mean()),
      'difference': float(fr.mean() - fs.mean()),
      'bootstrap_ci95': [float(np.percentile(boot, 2.5)),
                         float(np.percentile(boot, 97.5))],
      'interpretation': (
          'The SAME 16 failure goals scored by different anchors. A non-zero, '
          'CI-excluding-zero difference is direct evidence the critic already '
          'resolves the same goal differently depending on (s, a).')}

  # ---- Q4 pessimistic-positive vs failure-negative -------------------------
  rep['Q4_role_comparison'] = {
      'f_own_swc_as_positive': qstats(f_own),
      'f_bank_goals_as_negative_all_anchors': qstats(F),
      'f_bank_goals_for_risky_anchors': qstats(F[risky]),
      'note': ('f_own is each anchor paired with ITS OWN frozen s_wc (the '
               'pessimistic-positive role); F is the bank goals paired with '
               'unrelated anchors (the failure-negative role).')}

  rep['verdict'] = (
      'PAIR-DEPENDENT' if rep['Q2_variance_decomposition'][
          'pair_dependent_frac'] > 0.5 else 'GOAL-DOMINATED (investigate)')
  rep['caveat'] = (
      'Measured on the FROZEN Critic C, which was trained WITHOUT any '
      'pessimistic-positive branch. It shows the architecture resolves goals '
      'pair-dependently; it does not predict what training with the branch '
      'would do.')
  rep['git_commit'] = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()
  json.dump(rep, open(os.path.join(OUT, 'pair_dependence.json'), 'w'), indent=2)

  v = rep['Q2_variance_decomposition']
  q3r = rep['Q3_risky_vs_safe_same_failure_goals']
  print('Q1 per-goal std across anchors : mean %.4f (per-anchor std across '
        'goals %.4f)' % (rep['Q1_fixed_goal_spread_across_anchors']
                         ['per_goal_std_over_anchors']['mean'],
                         rep['Q1_fixed_goal_spread_across_anchors']
                         ['per_anchor_std_over_goals']['mean']))
  print('Q2 variance: goal %.3f | anchor %.3f | interaction %.3f -> '
        'pair-dependent %.3f'
        % (v['goal_main_effect_frac'], v['anchor_main_effect_frac'],
           v['anchor_goal_interaction_frac'], v['pair_dependent_frac']))
  print('Q3 same 16 failure goals: risky %.4f vs safe %.4f | diff %.4f '
        'CI95 [%.4f, %.4f]'
        % (q3r['mean_f_failure_goals_risky'], q3r['mean_f_failure_goals_safe'],
           q3r['difference'], q3r['bootstrap_ci95'][0],
           q3r['bootstrap_ci95'][1]))
  print('Q4 f(own s_wc) mean %.4f | f(bank, unrelated) mean %.4f'
        % (rep['Q4_role_comparison']['f_own_swc_as_positive']['mean'],
           rep['Q4_role_comparison']['f_bank_goals_as_negative_all_anchors']
           ['mean']))
  print('VERDICT: %s' % rep['verdict'])


if __name__ == '__main__':
  main()
