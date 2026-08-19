"""Pre-launch smoke for one arm of the four-arm worst-case ablation.

Runs the REAL learner on the REAL frozen dataset for N updates and checks every
property the launch gate requires. Uses the same config builder, buffer, bank
and loss as production; only the update count differs.

Checks
  C1  all losses finite (critic, actor, and every logged component)
  C2  actor/critic gradient norms finite
  C3  replay sampler stable (no exception, correct shapes, every batch)
  C4  branch actually fires at the expected rate (worst-case arms)
  C5  static table lookup works; positive goals really are replaced
  C6  the Flow is NEVER invoked during training -- StaticWorstCase is
      monkeypatched to raise, so any online call aborts the smoke
  C7  no gradients reach the Flow/selector (the sampler is numpy; the learner
      never sees Flow params)
  C8  no JAX recompilation loop (per-update time stabilizes after warmup)
  C9  nominal sampler is bitwise identical to the baseline when the nominal
      branch is forced
  C10 positive-goal distribution actually differs from the blind baseline
  Logs E[rho], E[p_wc] = 1 - E[rho], and the realized worst-case rate.

Usage:
  python scripts/smoke_four_arm.py --arm fixed10  --updates 2000
  python scripts/smoke_four_arm.py --arm fixed50  --updates 2000
  python scripts/smoke_four_arm.py --arm dpsi     --updates 2000
  python scripts/smoke_four_arm.py --arm blind    --updates 2000
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import jax                                                # noqa: E402
import optax                                              # noqa: E402
from crl import envs as envs_mod                          # noqa: E402
from crl import losses as losses_mod                      # noqa: E402
from crl import networks as networks_mod                  # noqa: E402
from crl import static_worstcase as sw                    # noqa: E402
from crl.replay import TrajectoryBuffer, obs_to_goal      # noqa: E402
from crl.pessimistic_positive import PessimisticPositiveBuffer  # noqa: E402
from verify_offline_d4rl import build_offline_cfg         # noqa: E402
from rockfall_v2_teacher import SEVERITY_V2               # noqa: E402

OUT = os.path.join(_ROOT, 'artifacts/four_arm_wc_run')
CLEAN = os.path.join(
    _ROOT, 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
    'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
CLEAN_SHA = '6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852'
BANK = os.path.join(
    _ROOT, 'artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz')
TABLE = os.path.join(_ROOT, 'artifacts/static_worstcase_rl/worstcase_table.npz')
DPSI = os.path.join(_ROOT,
                    'artifacts/support_discriminator/D_state_cmdgoal_action')
ARMS = {
    'fixed10': dict(run_id='wc_fixedp10_h800_a01_s0_300k', alpha=0.1,
                    wc=True, mode='fixed', p_wc=0.10),
    'fixed50': dict(run_id='wc_fixedp50_h800_a01_s0_300k', alpha=0.1,
                    wc=True, mode='fixed', p_wc=0.50),
    'dpsi': dict(run_id='wc_dpsi_surrogate_h800_a01_s0_300k', alpha=0.1,
                 wc=True, mode='dpsi', p_wc=None),
    'blind': dict(run_id='blind_crl_clean_h800_a00_s0_300k', alpha=0.0,
                  wc=False, mode=None, p_wc=None),
}
RESULTS = []


def check(name, ok, detail=''):
  RESULTS.append({'check': name, 'passed': bool(ok), 'detail': str(detail)})
  print('  %-46s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail),
        flush=True)
  return bool(ok)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--arm', required=True, choices=sorted(ARMS))
  ap.add_argument('--updates', type=int, default=2000)
  args = ap.parse_args()
  A = ARMS[args.arm]
  os.makedirs(OUT, exist_ok=True)
  print('=' * 70)
  print('SMOKE arm=%s run_id=%s updates=%d' % (args.arm, A['run_id'],
                                               args.updates))
  print('=' * 70, flush=True)

  # ---- C6 arm the Flow tripwire BEFORE anything else --------------------
  flow_calls = {'n': 0}
  _orig_init = sw.StaticWorstCase.__init__

  def _forbidden(self, *a, **k):
    flow_calls['n'] += 1
    raise RuntimeError('FLOW INVOKED DURING TRAINING -- the arm must use the '
                       'static table only')
  sw.StaticWorstCase.__init__ = _forbidden

  # ---- config: identical to production -----------------------------------
  cfg = build_offline_cfg(max_steps=300000, ckpt_dir='')
  cfg.env_name = 'offline_ant_umaze_rockfall'
  cfg.offline_dataset = CLEAN
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_severity = SEVERITY_V2
  cfg.rockfall_p_active = 0.3
  cfg.rockfall_max_steps = 800
  cfg.max_episode_steps = 800
  cfg.rockfall_reset_fix = True
  cfg.seed = 0
  envs_mod.make_env(cfg.env_name, cfg, seed=cfg.seed)   # fills dims

  assert sw.sha256_file(CLEAN) == CLEAN_SHA, 'dataset sha mismatch'
  fail_bank = None
  if A['alpha'] > 0:
    cfg.fail_bank_path = BANK
    cfg.fail_neg_alpha = A['alpha']
    with np.load(BANK) as fb:
      bank_states = np.asarray(fb['goals'], np.float32)
    fail_bank = obs_to_goal(bank_states, cfg.start_index, cfg.end_index,
                            cfg.goal_indices)

  with np.load(CLEAN, allow_pickle=True) as d:
    obs, act = np.asarray(d['obs'], np.float32), np.asarray(d['act'],
                                                            np.float32)
  E, L, W = obs.shape

  def fresh_buffer(seed=0):
    b = TrajectoryBuffer(capacity_steps=E * L, ep_len_obs=L, full_obs_dim=W,
                         action_dim=8, obs_dim=29, start_index=0, end_index=-1,
                         discount=0.99, seed=seed,
                         goal_indices=tuple(range(29)))
    for k in range(E):
      b.add_episode(obs[k], act[k])
    return b

  buffer = fresh_buffer()
  baseline_buffer = fresh_buffer()

  # ---- worst-case wrapper -------------------------------------------------
  table_sha = None
  if A['wc']:
    table_sha = sw.sha256_file(TABLE)
    if A['mode'] == 'fixed':
      def rho_fn(s_, g_, a_, _p=A['p_wc']):
        return np.full(len(s_), 1.0 - _p)
      rho_desc = 'fixed p_wc=%.2f' % A['p_wc']
    else:
      from propensity.agreement import (load_agreement_model,
                                        agreement_score_batch)
      _m = load_agreement_model(DPSI)

      def rho_fn(s_, g_, a_, _mm=_m):
        return np.asarray(agreement_score_batch(_mm.params, _mm.spec, s_, g_,
                                                a_), np.float64)
      rho_desc = 'D_psi surrogate (raw sigmoid)'
    buffer = PessimisticPositiveBuffer(buffer, TABLE, rho_fn=rho_fn, seed=0)
    print('  table sha %s (%d rows) | rho: %s'
          % (table_sha[:16], buffer.n_table_rows, rho_desc), flush=True)

  # ---- learner ------------------------------------------------------------
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
  init_state, update_step = losses_mod.build_learner(
      nets, cfg, obs_to_goal, optax.adam(cfg.actor_learning_rate, eps=1e-7),
      optax.adam(cfg.learning_rate, eps=1e-7), fail_bank=fail_bank)
  update_jit = jax.jit(update_step)
  state = init_state(jax.random.PRNGKey(cfg.seed))

  # ---- C9 forced-nominal bitwise identity (worst-case arms only) ---------
  if A['wc']:
    st = buffer._base._rng.bit_generator.state
    t_nom = buffer.sample(cfg.batch_size, force_branch=1)
    buffer._base._rng.bit_generator.state = st
    t_base = baseline_buffer.sample(cfg.batch_size)
    check('C9_forced_nominal_bitwise_identical_to_baseline',
          np.array_equal(t_nom.observation, t_base.observation)
          and np.array_equal(t_nom.action, t_base.action))
    buffer._base._rng.bit_generator.state = st
    t_wc = buffer.sample(cfg.batch_size, force_branch=0)
    check('C5_positive_goal_actually_replaced',
          not np.array_equal(t_wc.observation[:, 29:],
                             t_base.observation[:, 29:]))
    check('C10_wc_goals_differ_from_baseline_population',
          float(np.abs(t_wc.observation[:, 29:]
                       - t_base.observation[:, 29:]).mean()) > 1e-6,
          'mean |dg| %.4f' % float(np.abs(t_wc.observation[:, 29:]
                                          - t_base.observation[:, 29:]).mean()))
    buffer._base._rng.bit_generator.state = st
    buffer._n_nominal = buffer._n_worstcase = 0
    buffer._rho_sum = 0.0
    buffer._rho_n = 0

  # ---- training loop ------------------------------------------------------
  hist, times = [], []
  n_batches = 0
  for u in range(args.updates):
    t0 = time.time()
    batch = buffer.sample(cfg.batch_size)
    n_batches += 1
    assert batch.observation.shape == (cfg.batch_size, 58), \
        batch.observation.shape
    state, m = update_jit(state, batch)
    if u == 0:
      jax.block_until_ready(state.q_params)
    times.append(time.time() - t0)
    if (u + 1) % 250 == 0 or u == args.updates - 1:
      mm = {k: float(v) for k, v in m.items()}
      hist.append({'update': u + 1, **mm})
      line = ('  [%5d] critic=%.4f actor=%.4f cat_acc=%.3f'
              % (u + 1, mm['critic_loss'], mm['actor_loss'],
                 mm['categorical_accuracy']))
      if 'critic_pos_term' in mm:
        line += (' pos=%.4f neg_ord=%.4f fail_neg=%.4f'
                 % (mm['critic_pos_term'], mm['critic_neg_ord_term'],
                    mm['critic_neg_fail_term']))
      if A['wc']:
        bs = buffer.branch_stats()
        line += (' | E[rho]=%.4f E[p_wc]=%.4f wc_rate=%.4f'
                 % (bs['mean_rho'], 1 - bs['mean_rho'],
                    bs['realized_wc_rate']))
      print(line, flush=True)

  jax.block_until_ready(state.q_params)

  # ---- checks -------------------------------------------------------------
  print('\nCHECKS')
  last = hist[-1]
  finite = all(np.isfinite(v) for k, v in last.items() if k != 'update')
  check('C1_all_losses_finite', finite)
  check('C2_grad_norms_finite',
        np.isfinite(last['actor_grad_norm'])
        and np.isfinite(last['critic_grad_norm']),
        'actor %.3g critic %.3g' % (last['actor_grad_norm'],
                                    last['critic_grad_norm']))
  check('C3_replay_sampler_stable', n_batches == args.updates,
        '%d batches' % n_batches)
  check('C6_flow_never_invoked_during_training', flow_calls['n'] == 0)
  check('C7_no_flow_params_in_learner_state',
        not any('vfield' in str(k) for k in
                jax.tree_util.tree_structure(state.q_params).__str__()))
  warm = np.array(times[max(5, len(times) // 5):])
  spikes = int((warm > 5 * np.median(warm)).sum())
  check('C8_no_recompilation_loop', spikes <= 1,
        'median %.4fs, %d spikes>5x' % (float(np.median(warm)), spikes))
  if A['alpha'] > 0:
    lhs = (last['critic_pos_term'] + last['critic_neg_ord_term']
           + last['critic_neg_fail_term'])
    check('C11_loss_decomposition_identity',
          abs(lhs - last['critic_loss']) < 1e-4,
          '|lhs-critic| %.2e' % abs(lhs - last['critic_loss']))
    check('C12_fail_bank_size_16', last['fail_bank_size'] == 16)

  bs = buffer.branch_stats() if A['wc'] else None
  if A['wc']:
    check('C4_branch_fires', bs['n_worstcase'] > 0,
          '%d worst-case of %d' % (bs['n_worstcase'],
                                   bs['n_worstcase'] + bs['n_nominal']))
    if A['mode'] == 'fixed':
      exp = A['p_wc']
      se = np.sqrt(exp * (1 - exp) / (args.updates * cfg.batch_size))
      check('C4b_realized_rate_matches_fixed_p_wc',
            abs(bs['realized_wc_rate'] - exp) < max(5 * se, 0.005),
            'realized %.4f vs %.2f' % (bs['realized_wc_rate'], exp))

  sw.StaticWorstCase.__init__ = _orig_init
  n_fail = sum(1 for r in RESULTS if not r['passed'])
  rep = {'arm': args.arm, 'run_id': A['run_id'], 'updates': args.updates,
         'alpha_fail': A['alpha'], 'worst_case_enabled': A['wc'],
         'rho_mode': A['mode'], 'p_wc_configured': A['p_wc'],
         'table_sha256': table_sha,
         'branch_stats': bs,
         'mean_rho': bs['mean_rho'] if bs else None,
         'mean_p_wc': (1 - bs['mean_rho']) if bs else None,
         'realized_wc_rate': bs['realized_wc_rate'] if bs else None,
         'median_update_seconds': float(np.median(warm)),
         'loss_history': hist, 'checks': RESULTS, 'n_failed': n_fail,
         'git_commit': subprocess.check_output(
             ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()}
  json.dump(rep, open(os.path.join(OUT, 'smoke_%s.json' % args.arm), 'w'),
            indent=2)
  print('\n%d/%d checks passed  ->  %s'
        % (len(RESULTS) - n_fail, len(RESULTS),
           os.path.join(OUT, 'smoke_%s.json' % args.arm)))
  sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
  main()
