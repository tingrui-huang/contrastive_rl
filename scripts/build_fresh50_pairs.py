"""Phase 2: build the FRESH50 sealed same-anchor evaluation set.

Reuses the already-validated same-anchor pair protocol verbatim -- only the
environment/dataset seeds change. Nothing about the definition, the gates or
the action source is redesigned.

  fatal world  a fresh episode driven live by the same teacher mixture used
               by the validated oracle-pair stream (sighted 90% / coverage
               10%), in the patched env with death_settle_substeps=80. Its
               recorded actions ARE the frozen action source (anchor_action),
               exactly as in the final39 protocol.
  safe world   the same episode replayed from a forced-mask reset with the
               severe site(s) responsible for the fatal hit switched off, the
               SAME recorded actions applied. The two envs share one seed and
               reset in lockstep so goal / init pose / severities / jitter are
               unchanged; the env's reset RNG draw order is mask-independent
               by design.

Gates (identical to scripts/build_same_anchor_pairs.py):
  * safe-world obs equals fatal-world obs BITWISE for every t <= collapse;
  * the paired action is the same stored float32 array in both worlds;
  * fatal branch dies on the paired step and returns the N=80 settled state;
  * safe branch is non-fatal on the paired step and finite.
Every attempt is logged with its exclusion reason. Pairs are never dropped
for looking difficult, and no Flow/Critic output is consulted while building.

Disjointness: fresh env/dataset seeds, checked against every consumed seed
(pilot, 40-death stream, bad-demo, all four V3 arms).

Usage:
  python scripts/build_fresh50_pairs.py [--target 50]
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
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from rockfall_v2_teacher import apply_v2_config, SEVERITY_V2  # noqa: E402
from collect_rockfall_pilot import (check_rockfall_freeze,     # noqa: E402
                                    prescreen_env_seed)
from collect_rockfall_v2_pilot import V2_CONSUMED, MIX         # noqa: E402
from collect_rockfall_death_extended import rollout_extended   # noqa: E402
from build_same_anchor_pairs import (replay_actions,           # noqa: E402
                                     responsible_sites)
from verify_offline_d4rl import build_offline_cfg              # noqa: E402

OUT = 'artifacts/flow_v3_fresh50'
P_ACTIVE, HORIZON, SETTLE_N, OBS_DIM = 0.30, 800, 80, 29
SEED_BASE = 96_500_019
DATASET_SEED = 96_990_013
#: every seed consumed by an earlier collection / probe
EXTRA_CONSUMED = [52_400_019, 71_400_019, 82_500_019, 91_500_019,
                  92_500_019, 93_500_019, 94_500_019,
                  51_990_013, 51_990_014, 71_990_013, 82_990_013,
                  91_990_013, 92_990_013, 93_990_013, 94_990_013]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--target', type=int, default=50)
  ap.add_argument('--max-episodes', type=int, default=3000)
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  hard_ok, disc, info = C.check_frozen_integrity()
  rf_ok, rf_diffs, rf_man = check_rockfall_freeze()
  assert hard_ok and rf_ok, 'frozen-integrity failure %s' % (disc + rf_diffs)
  excl = V2_CONSUMED + EXTRA_CONSUMED
  env_seed, prescreen = prescreen_env_seed(
      args.max_episodes, [SEED_BASE + 97 * k for k in range(400)],
      exclude=excl, p_active=P_ACTIVE)
  clash = C.seed_reuse(excl, [env_seed], [DATASET_SEED])
  assert not clash, 'seed clash %s' % clash
  print('fresh env_seed %d | dataset_seed %d | prescreen %s'
        % (env_seed, DATASET_SEED, prescreen), flush=True)

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_death_settle_substeps = SETTLE_N

  def mk(seed):
    e = apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed),
        P_ACTIVE, reset_fix=True)
    e.death_settle_substeps = SETTLE_N
    assert e.death_settle_substeps == SETTLE_N
    return e
  fatal_env, safe_env = mk(env_seed), mk(env_seed)

  # teacher mixture, same convention as the validated stream
  ds_rng = np.random.default_rng(DATASET_SEED)
  side_rng = np.random.default_rng(DATASET_SEED + 1)
  n_cover = int(round(MIX['coverage'] * args.max_episodes))
  modes = np.array(['sighted'] * (args.max_episodes - n_cover)
                   + ['coverage'] * n_cover)
  ds_rng.shuffle(modes)

  pairs, attempts = [], []
  for e in range(args.max_episodes):
    mode = str(modes[e])
    side = 'left' if side_rng.random() < 0.5 else 'right'
    o_f = fatal_env.reset()
    obs, act, _, _, ep = rollout_extended(fatal_env, o_f, walker, base_act,
                                          mode, side, HORIZON, extend=2)
    # safe_env gets EXACTLY ONE reset per episode (plain when there is no
    # death, forced-mask when there is) so the two RNG streams stay in
    # lockstep -- the env consumes randomness only in reset(), never step().
    if not ep['dead']:
      safe_env.reset()
      attempts.append({'episode': e, 'mode': mode, 'status': 'no_death'})
      continue
    c = int(ep['collapse_step'])
    resp = responsible_sites(fatal_env, c)
    rec = {'episode': e, 'mode': mode, 'base_side': side,
           'collapse_step': c,
           'original_mask': [int(b) for b in ep['rockfall_mask']],
           'severities': [str(s) for s in ep['severity']]}
    if not resp:
      safe_env.reset()                   # keep the streams aligned
      attempts.append({**rec, 'status': 'excluded',
                       'reason': 'no_responsible_severe_site'})
      continue
    cf_mask = [int(b) for b in ep['rockfall_mask']]
    for j in resp:
      cf_mask[j] = 0
    rec['fatal_site'] = [fatal_env.site_names[j] for j in resp]
    rec['counterfactual_mask'] = cf_mask

    # the safe world must restart from the SAME reset draw with a forced mask
    o_s = safe_env.reset(mask=tuple(cf_mask))
    ok, div_t, anchor_safe, _ = replay_actions(safe_env, o_s, act, obs, c)
    if not ok:
      attempts.append({**rec, 'status': 'excluded',
                       'reason': 'safe_world_divergence_before_anchor_at_t%d'
                                 % div_t})
      continue
    if not np.array_equal(anchor_safe, obs[c]):
      attempts.append({**rec, 'status': 'excluded',
                       'reason': 'anchor_mismatch'})
      continue
    s_safe, _, _, info_s = safe_env.step(act[c])
    if info_s['dead'] or safe_env.dead:
      attempts.append({**rec, 'status': 'excluded',
                       'reason': 'safe_branch_fatal_from_other_site'})
      continue
    if not np.isfinite(s_safe).all():
      attempts.append({**rec, 'status': 'excluded',
                       'reason': 'safe_branch_nonfinite'})
      continue
    pairs.append({'anchor_obs': obs[c, :OBS_DIM].astype(np.float32),
                  'anchor_action': act[c].astype(np.float32),
                  'fatal_candidate': obs[c + 1, :OBS_DIM].astype(np.float32),
                  'safe_candidate': s_safe[:OBS_DIM].astype(np.float32),
                  'goal_xy': ep['goal_xy'], 'episode_id': e,
                  'collapse_step': c,
                  'original_mask': np.asarray(ep['rockfall_mask'], np.int8),
                  'counterfactual_mask': np.asarray(cf_mask, np.int8)})
    attempts.append({**rec, 'status': 'ACCEPTED',
                     'pair_index': len(pairs) - 1,
                     'safe_rock_contact_on_paired_step':
                         bool(info_s['rock_ant_contact'])})
    print('  pair %2d/%d (episode %d, site %s, c=%d)'
          % (len(pairs), args.target, e, rec['fatal_site'], c), flush=True)
    if len(pairs) >= args.target:
      break

  n_ep = e + 1
  assert len(pairs) >= args.target, \
      'only %d valid pairs in %d episodes' % (len(pairs), n_ep)

  npz = os.path.join(args.out, 'fresh50_pairs.npz')
  meta = {'label': 'fresh50', 'n': len(pairs),
          'protocol': ('validated same-anchor protocol, reused verbatim; '
                       'only the seeds are new'),
          'action_source': ('anchor_action = the recorded factual action at '
                            'collapse_step, identical convention to final39'),
          'death_settle_substeps': SETTLE_N,
          'transition_convention': ('fatal branch = patched one-step '
                                    'transition with the internal N=80 '
                                    'settle; safe branch = the ordinary '
                                    'one-step transition'),
          'env_seed': env_seed, 'dataset_seed': DATASET_SEED,
          'sealed_note': ('paired hidden-world intervention is EVALUATION '
                          'only and never enters training')}
  with open(npz + '.tmp', 'wb') as f:
    np.savez_compressed(
        f,
        anchor_obs=np.stack([p['anchor_obs'] for p in pairs]),
        anchor_action=np.stack([p['anchor_action'] for p in pairs]),
        fatal_candidate=np.stack([p['fatal_candidate'] for p in pairs]),
        safe_candidate=np.stack([p['safe_candidate'] for p in pairs]),
        goal_xy=np.stack([p['goal_xy'] for p in pairs]),
        episode_id=np.array([p['episode_id'] for p in pairs], np.int64),
        collapse_step=np.array([p['collapse_step'] for p in pairs], np.int64),
        original_mask=np.stack([p['original_mask'] for p in pairs]),
        counterfactual_mask=np.stack([p['counterfactual_mask']
                                      for p in pairs]),
        meta=json.dumps(meta))
  os.replace(npz + '.tmp', npz)

  n_dead = sum(1 for a in attempts if a.get('status') != 'no_death')
  man = {
      'target': args.target, 'n_pairs': len(pairs),
      'n_episodes_run': n_ep,
      'n_death_episodes': n_dead,
      'n_excluded': sum(1 for a in attempts if a.get('status') == 'excluded'),
      'exclusion_reasons': {r: sum(1 for a in attempts
                                   if a.get('reason') == r)
                            for r in sorted({a['reason'] for a in attempts
                                             if 'reason' in a})},
      'env_seed': env_seed, 'dataset_seed': DATASET_SEED,
      'prescreen_freq': prescreen,
      'seed_disjointness_checked_against': excl,
      'teacher_mixture': {'sighted': 1 - MIX['coverage'],
                          'coverage': MIX['coverage']},
      'p_active': P_ACTIVE, 'horizon': HORIZON,
      'death_settle_substeps': SETTLE_N, 'reset_fix': True,
      'severity_probs_v2': list(SEVERITY_V2),
      'gates': ('anchor bitwise across worlds for all t<=collapse; identical '
                'stored action; fatal branch dead with settled successor; '
                'safe branch non-fatal and finite'),
      'no_model_output_consulted': True,
      'attempts': attempts,
      'pairs_npz': npz, 'pairs_sha256': C.sha256_file(npz),
      'walker_sha256': info['walker_sha256'],
      'git_commit': C.git_commit()}
  json.dump(man, open(os.path.join(args.out,
                                   'pair_generation_manifest.json'), 'w'),
            indent=2)
  print('\nfresh50: %d pairs from %d episodes (%d deaths, %d excluded)'
        % (len(pairs), n_ep, n_dead, man['n_excluded']))
  print('exclusions: %s' % man['exclusion_reasons'])
  print('sha %s' % man['pairs_sha256'])
  print('-> %s' % npz)


if __name__ == '__main__':
  main()
