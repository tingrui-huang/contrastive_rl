"""Stage 1: rebuild the production failure bank from N=80 SETTLED fatal
observations (rock-death observability patch, death_settle_substeps=80).

Bank definition (unchanged convention, new physics): one failure state per
original authoritative pilot death episode -- the observation returned by the
fatal transition, now after the 80-substep internal ctrl-free physics settle:

    g_fail = s'_settled = obs[e, collapse_step + 1, :29]

EXACTLY the 16 original pilot deaths enter the bank. The 40 fresh deaths from
the physics diagnostic are HELD OUT (later similarity evaluation only) and
are recorded in an explicit held-out manifest; they must never influence
training or selection.

Validations performed here (all must pass or the script aborts):
  V1  bank size == 16;
  V2  every state comes from an original pilot death episode (episode ids ==
      the authoritative failure-split ids);
  V3  every fatal prefix (obs[0:collapse+1], act[0:collapse+1]) reproduces
      the frozen pilot npz BITWISE;
  V4  the extracted state is the settled observation: bitwise equal to the
      diagnostic sweep's trace at substep 80, and != the legacy frozen state;
  V5  new bank observably differs from the old bank (per-state physics);
  V6  the clean training npz hash equals the authoritative failure-split
      manifest hash AND the hash recorded by the previous alpha=0.1 run;
  V7  representation contract: 29-dim ant-only state, same keys as the old
      bank; no rock/severity/dead/hidden metadata fields;
  V8  goal-encoder contract: np_obs_to_goal(bank) has the expected width;
  V9  zero overlap between the bank and the 40 held-out fresh deaths.

The old bank is NOT overwritten. Output goes to
artifacts/settled_failure_bank_alpha01/.
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from death_settle_sweep import make_patched_env, run_death_episode  # noqa: E402
from collect_rockfall_death_extended import PILOT_DIR, PILOT_NAME  # noqa: E402

OUT_DIR = 'artifacts/settled_failure_bank_alpha01'
SETTLE_N = 80
OLD_BANK = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
            'failure_bank.npz')
SPLIT_MANIFEST = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
                  'failure_split_manifest.json')
CLEAN_NPZ = ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
             'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
PREV_A01_SHA_FILE = 'failneg_clean_p30_h800_resetfix_a01_s0_300k/offline_dataset.sha256'
SWEEP_TRACES = 'artifacts/rockfall_death_physics_patch/settle_traces.npz'
OBS_DIM = 29


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  checks = {}

  # ---- V6 first (stop condition 1): clean dataset authoritative ------------
  clean_sha = C.sha256_file(CLEAN_NPZ)
  split_man = json.load(open(SPLIT_MANIFEST))
  prev_sha = json.load(open(PREV_A01_SHA_FILE))['sha256']
  checks['V6_clean_hash_vs_split_manifest'] = (
      clean_sha == split_man['clean']['sha256'])
  checks['V6_clean_hash_vs_prev_a01_run'] = clean_sha == prev_sha
  if not all(v for k, v in checks.items()):
    print('ABORT: clean dataset hash mismatch.', checks)
    return 2

  d0 = np.load(os.path.join(PILOT_DIR, f'{PILOT_NAME}.npz'),
               allow_pickle=True)
  s0 = np.load(os.path.join(PILOT_DIR, f'{PILOT_NAME}_sidecar.npz'),
               allow_pickle=True)
  pman = json.load(open(os.path.join(PILOT_DIR, 'pilot_manifest.json')))
  pilot_seed = int(pman['env_seed'])
  dead_ids = np.where(np.asarray(s0['dead'], bool))[0]
  auth_ids = split_man['rockfail']['episode_ids']
  checks['V2_episode_ids_match_failure_split'] = (
      dead_ids.tolist() == list(auth_ids))
  checks['V1_bank_size_16'] = len(dead_ids) == 16

  # ---- replay the 16 pilot deaths under the PATCHED env (N=80) -------------
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  env = make_patched_env(cfg, pilot_seed, SETTLE_N)

  sweep = np.load(SWEEP_TRACES, allow_pickle=True)
  sw_pilot = sweep['source'] == 'pilot'
  sw_ids = np.asarray(sweep['episode_id'])[sw_pilot]
  sw_traces = sweep['traces'][sw_pilot]

  bank_states, prov = [], []
  ok3 = ok4 = True
  for e in range(int(dead_ids.max()) + 1):
    o = env.reset()
    if e not in dead_ids:
      continue
    mode, side = str(s0['teacher_mode'][e]), str(s0['base_side'][e])
    obs, act, ep, _ = run_death_episode(env, o, mode, side, walker,
                                        base_act, extend=2)
    c = int(s0['collapse_step'][e])
    pre_ok = (ep['collapse_step'] == c
              and np.array_equal(obs[:c + 1], d0['obs'][e, :c + 1])
              and np.array_equal(act[:c + 1], d0['act'][e, :c + 1]))
    settled = obs[c + 1, :OBS_DIM].astype(np.float32)
    j = int(np.where(sw_ids == e)[0][0])
    trace_ok = np.array_equal(settled, sw_traces[j, SETTLE_N - 1])
    differs = not np.array_equal(settled, d0['obs'][e, c + 1, :OBS_DIM])
    ok3 &= pre_ok
    ok4 &= trace_ok and differs
    bank_states.append(settled)
    prov.append({'episode_id': int(e), 'collapse_step': c,
                 'first_hit_step': int(s0['first_hit_step'][e]),
                 'ep_length': int(s0['ep_length'][e]),
                 'prefix_bitwise_ok': bool(pre_ok),
                 'settled_matches_sweep_trace': bool(trace_ok),
                 'differs_from_legacy': bool(differs),
                 'z_new': float(settled[2]),
                 'v_xy_new': float(np.linalg.norm(settled[15:17])),
                 'z_old': float(d0['obs'][e, c + 1, 2]),
                 'v_xy_old': float(np.linalg.norm(
                     d0['obs'][e, c + 1, 15:17]))})
    print(f"  ep {e}: prefix {pre_ok} trace {trace_ok} "
          f"z {prov[-1]['z_old']:.3f}->{prov[-1]['z_new']:.3f} "
          f"v {prov[-1]['v_xy_old']:.2f}->{prov[-1]['v_xy_new']:.3f}",
          flush=True)
  checks['V3_all_prefixes_bitwise'] = bool(ok3)
  checks['V4_settled_state_n80'] = bool(ok4)
  bank = np.stack(bank_states)

  # ---- V5: observably different from the old bank --------------------------
  old = np.load(OLD_BANK, allow_pickle=True)
  old_goals = np.asarray(old['goals'], np.float32)
  checks['V5_all_states_differ_from_old_bank'] = bool(
      (np.abs(bank - old_goals).max(axis=1) > 1e-6).all())
  checks['V2_old_bank_episode_ids_match'] = (
      np.asarray(old['episode_id']).tolist() == dead_ids.tolist())

  # ---- V7/V8: representation contract --------------------------------------
  checks['V7_state_dim_29'] = bank.shape == (16, OBS_DIM)
  bank_keys = ['goals', 'episode_id', 'collapse_step', 'first_hit_step',
               'ep_length', 'meta']
  checks['V7_same_keys_as_old_bank'] = sorted(old.files) == sorted(bank_keys)
  from crl.replay import obs_to_goal
  g = obs_to_goal(bank, 0, -1, tuple(range(OBS_DIM)))
  checks['V8_goal_encoder_width_29'] = g.shape == (16, 29)

  # ---- V9: zero overlap with the 40 held-out fresh deaths ------------------
  sw_fresh = sweep['source'] == 'fresh'
  fresh_settled = sweep['traces'][sw_fresh][:, SETTLE_N - 1]
  overlap = (np.abs(bank[:, None] - fresh_settled[None]).max(axis=2)
             < 1e-9).any()
  checks['V9_zero_overlap_with_heldout'] = not bool(overlap)
  heldout = {
      'purpose': ('held-out similarity evaluation ONLY; must never enter '
                  'failure-negative training, hyperparameter selection, or '
                  'checkpoint selection'),
      'n': int(sw_fresh.sum()),
      'env_seed': int(np.asarray(sweep['env_seed'])[sw_fresh][0]),
      'episode_ids': np.asarray(sweep['episode_id'])[sw_fresh].tolist(),
      'source_npz': SWEEP_TRACES,
      'settled_states_sha256_of_array': C.sha256_file(SWEEP_TRACES)}

  if not all(checks.values()):
    print('ABORT: validation failed:',
          {k: v for k, v in checks.items() if not v})
    return 3

  # ---- write the new bank ---------------------------------------------------
  bank_path = os.path.join(OUT_DIR, 'failure_bank_settled.npz')
  meta = {
      'definition': ('N=80 SETTLED post-fatal observation obs[e, '
                     'collapse_step+1, :29] of each authoritative pilot '
                     'dead episode, produced by the patched env '
                     '(death_settle_substeps=80: actor ctrl zeroed at the '
                     'fatal contact, 80 extra MuJoCo substeps of '
                     'gravity/rock/contact/mud physics inside the fatal '
                     'transition). Replaces the legacy healthy-looking '
                     'frozen pose bank (preserved unchanged at '
                     f'{OLD_BANK}).'),
      'death_settle_substeps': SETTLE_N,
      'source_npz': os.path.join(PILOT_DIR, f'{PILOT_NAME}.npz'),
      'source_sha256': C.sha256_file(
          os.path.join(PILOT_DIR, f'{PILOT_NAME}.npz')),
      'pilot_env_seed': pilot_seed,
      'obs_dim': OBS_DIM, 'goal_indices': 'range(29)',
      'git_commit': C.git_commit()}
  tmp = bank_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(
        f, goals=bank,
        episode_id=dead_ids.astype(np.int64),
        collapse_step=np.asarray(s0['collapse_step'])[dead_ids],
        first_hit_step=np.asarray(s0['first_hit_step'])[dead_ids],
        ep_length=np.asarray(s0['ep_length'])[dead_ids],
        meta=json.dumps(meta))
  os.replace(tmp, bank_path)

  manifest = {
      'checks': checks,
      'bank': {'path': bank_path, 'sha256': C.sha256_file(bank_path),
               'n_states': 16, 'state_dim': OBS_DIM,
               'death_settle_substeps': SETTLE_N},
      'old_bank': {'path': OLD_BANK, 'sha256': C.sha256_file(OLD_BANK),
                   'preserved': True},
      'clean_npz': {'path': CLEAN_NPZ, 'sha256': clean_sha},
      'heldout_fresh_deaths': heldout,
      'physical_stats': {
          'new': {'z_median': float(np.median(bank[:, 2])),
                  'v_xy_median': float(np.median(
                      np.linalg.norm(bank[:, 15:17], axis=1)))},
          'old': {'z_median': float(np.median(old_goals[:, 2])),
                  'v_xy_median': float(np.median(
                      np.linalg.norm(old_goals[:, 15:17], axis=1)))}},
      'per_state_provenance': prov,
      'git_commit': C.git_commit()}
  man_path = os.path.join(OUT_DIR, 'bank_manifest.json')
  json.dump(manifest, open(man_path, 'w'), indent=2)
  print(f"\nALL CHECKS PASS: {json.dumps(checks, indent=1)}")
  print(f"new bank -> {bank_path} ({manifest['bank']['sha256'][:16]}...)")
  print(f"old bank preserved ({manifest['old_bank']['sha256'][:16]}...)")
  print(f"manifest -> {man_path}")
  return 0


if __name__ == '__main__':
  sys.exit(main())
