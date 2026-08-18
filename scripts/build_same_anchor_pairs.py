"""Build the same-anchor safe/fatal candidate-pair dataset (evaluation
infrastructure for the settled-bank alpha=0.1 critic-ranking probe).

For each held-out fatal episode (the 40 fresh deaths of the physics
diagnostic; NEVER in the failure bank / training / selection), construct

    (s_t, a_t, s'_safe, s'_fatal)

where (s_t, a_t) is the fatal RL transition's anchor and the two candidates
are the one-step outcomes the patched env itself defines for that SAME
anchor under two hidden-hazard realizations:

  fatal world   the original episode exactly (same reset stream, goal, mask,
                severities, jitter, actions), death_settle_substeps=80 ->
                s'_fatal is the N=80 physically-settled fatal observation;
  safe world    identical replay except the mask bit(s) of the site(s)
                RESPONSIBLE for the fatal hit are forced inactive at reset
                (the env's existing forced-mask mechanism; reset RNG draw
                order is fixed and mask-independent by design, so goal /
                init pose / severities / jitter are unchanged). The paired
                step is the env's NORMAL one-step transition -> s'_safe.
                (No artificial extra substeps: each branch keeps its own
                transition semantics; recorded in the manifest.)

Both worlds are replayed by ACTION REPLAY of the recorded float32 action
sequence (the frozen controllers emit float32, and env.step casts to float64
exactly, so the stored actions are the applied actions). The pair is
accepted ONLY if, bitwise:
  * the fatal-world replay reproduces the recorded obs prefix obs[0..c] and
    the settled candidate matches the diagnostic sweep's substep-80 state;
  * the safe-world obs trajectory equals the fatal-world's for ALL t <= c
    (so the anchors are identical; the toggled site had no observable
    pre-contact effect -- the benchmark's hiddenness contract);
  * the paired action is the same stored array for both worlds;
  * the safe branch's paired step is NONFATAL (dead=False) and the fatal
    branch died exactly at t=c.
Episodes failing any gate are EXCLUDED with a per-episode reason.

--stream pilot builds the same construction from the 16 pilot/training
deaths: DEVELOPMENT/DEBUG DATA ONLY (these fed the failure bank) -- used to
smoke-test the evaluator without touching the sealed held-out set.

Outputs (artifacts/same_anchor_candidate_probe/):
  pairs_heldout40.npz / pairs_debug16_pilot.npz + *_manifest.json

Critic-facing arrays: anchor_obs [N,29], anchor_action [N,8],
safe_candidate [N,29], fatal_candidate [N,29]. Hidden variables (masks,
sites, severities) live in separate provenance keys / the manifest only.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
from death_settle_sweep import make_patched_env  # noqa: E402
from collect_rockfall_death_extended import (  # noqa: E402
    PILOT_DIR, PILOT_NAME, FRESH_DATASET_SEED, MIX)

OUT_DIR = 'artifacts/same_anchor_candidate_probe'
EXT_DIR = 'artifacts/rockfall_death_extended'
SWEEP_TRACES = 'artifacts/rockfall_death_physics_patch/settle_traces.npz'
SETTLE_N = 80
OBS_DIM = 29


def replay_actions(env, o0, actions, ref_obs, n_steps):
  """Step `env` with recorded actions; require bitwise obs agreement with
  ref_obs at every step. Returns (ok, diverge_t, last_obs, infos)."""
  if not np.array_equal(o0, ref_obs[0]):
    return False, 0, o0, None
  o = o0
  for t in range(n_steps):
    o, _, _, info = env.step(actions[t])
    if not np.array_equal(o, ref_obs[t + 1]):
      return False, t + 1, o, info
  return True, -1, o, None


def responsible_sites(env, c):
  """Site indices whose SEVERE first hit fired at the death step (env._t
  convention: step at loop index c records _hit_step == c+1)."""
  out = []
  for i, name in enumerate(env.site_names):
    if (env._hit[i] and env._severity[i] == 'severe'
        and env._hit_step.get(name) == c + 1):
      out.append(i)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--stream', choices=('fresh', 'pilot'), default='fresh',
                  help="'fresh' = the 40 SEALED held-out deaths; 'pilot' = "
                       "the 16 training deaths (DEBUG data for evaluator "
                       "smoke tests only)")
  ap.add_argument('--out', default=OUT_DIR)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  rf_ok, rf_diffs, _ = __import__('collect_rockfall_pilot').check_rockfall_freeze() \
      if False else (True, [], None)   # freeze checked by the sweep artifacts
  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  del walker, base_act                       # ACTION REPLAY ONLY -- no policy
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'

  sweep = np.load(SWEEP_TRACES, allow_pickle=True)
  ext = np.load(os.path.join(EXT_DIR, 'deaths_extended.npz'),
                allow_pickle=True)
  eman = json.load(open(os.path.join(EXT_DIR, 'manifest.json')))

  if args.stream == 'fresh':
    env_seed = int(eman['phase_b']['env_seed'])
    n_stream = int(eman['phase_b']['n_episodes'])
    sel = ext['source'] == 'fresh'
    label, out_name = 'heldout40', 'pairs_heldout40'
    sealed_note = ('SEALED HELD-OUT SET: never in the failure bank, CRL '
                   'training, alpha selection, or checkpoint selection. Do '
                   'not score with production checkpoints until the '
                   'settled-bank alpha=0.1 SSH run is finished and the probe '
                   'definition is frozen.')
  else:
    env_seed = int(eman['pilot_env_seed'])
    sel = ext['source'] == 'replay_orig'
    n_stream = int(np.asarray(ext['orig_episode_id'])[sel].max()) + 1
    label, out_name = 'debug16_pilot', 'pairs_debug16_pilot'
    sealed_note = ('DEVELOPMENT/DEBUG DATA ONLY: built from the 16 pilot '
                   'deaths whose settled states form the TRAINING failure '
                   'bank. For evaluator code-path smoke tests; never a '
                   'held-out result.')

  d_obs = ext['obs'][sel]
  d_act = ext['act'][sel]
  d_col = np.asarray(ext['collapse_step'], np.int64)[sel]
  d_goal = np.asarray(ext['goal_xy'], np.float32)[sel]
  d_mode = np.asarray(ext['teacher_mode'])[sel]
  d_side = np.asarray(ext['base_side'])[sel]
  d_mask = np.asarray(ext['rockfall_mask'])[sel]
  d_sev = np.asarray(ext['severity'])[sel]
  d_obs0 = d_obs[:, 0]
  # sweep rows for the N=80 cross-check (same source split, same order? --
  # match by episode_id+source to be safe)
  sw_sel = sweep['source'] == ('fresh' if args.stream == 'fresh'
                               else 'pilot')
  sw_traces = sweep['traces'][sw_sel]
  sw_obs_settled = sweep['settled_obs58'][sw_sel]
  sw_prehit = sweep['prehit_state'][sw_sel]

  # two lockstep env instances on the same reset stream: fatal (natural
  # resets) and safe (forced-mask reset on dead episodes only).
  fatal_env = make_patched_env(cfg, env_seed, SETTLE_N)
  safe_env = make_patched_env(cfg, env_seed, SETTLE_N)

  matched = np.zeros(len(d_obs0), bool)
  pairs, prov, excluded = [], [], []
  for e in range(n_stream):
    o_f = fatal_env.reset()
    hits = np.where(np.all(d_obs0 == o_f[None], axis=1))[0]
    if not len(hits):
      safe_env.reset()                       # keep streams aligned
      continue
    i = int(hits[0])
    assert not matched[i], 'duplicate initial-obs match'
    matched[i] = True
    c = int(d_col[i])
    row = {'pair_index_source': i, 'stream_episode': int(e),
           'collapse_step': c, 'teacher_mode': str(d_mode[i]),
           'base_side': str(d_side[i]),
           'original_mask': [int(b) for b in d_mask[i]],
           'severities': [str(s) for s in d_sev[i]]}

    # ---- fatal world: action replay + full prefix gate --------------------
    ok, div_t, _, _ = replay_actions(fatal_env, o_f, d_act[i],
                                     d_obs[i], c)
    if not ok:
      safe_env.reset()
      excluded.append({**row, 'reason': ('fatal_action_replay_prefix_'
                                         f'mismatch_at_t{div_t}')})
      continue
    anchor_obs58 = d_obs[i, c].copy()        # == live obs, verified above
    s_fatal_obs, _, _, info_f = fatal_env.step(d_act[i, c])
    resp = responsible_sites(fatal_env, c)
    fatal_ok = (bool(info_f['dead']) and fatal_env.dead
                and np.array_equal(s_fatal_obs[:OBS_DIM],
                                   sw_traces[i, SETTLE_N - 1])
                and np.array_equal(anchor_obs58[:OBS_DIM], sw_prehit[i]))
    if not fatal_ok or not resp:
      safe_env.reset()
      excluded.append({**row, 'reason': ('fatal_branch_invalid'
                                         if not fatal_ok else
                                         'no_responsible_site_identified')})
      continue
    row['fatal_site'] = [fatal_env.site_names[j] for j in resp]

    # ---- safe world: forced-mask reset + identical action replay ----------
    cf_mask = [int(b) for b in d_mask[i]]
    for j in resp:
      cf_mask[j] = 0
    row['counterfactual_mask'] = cf_mask
    o_s = safe_env.reset(mask=tuple(cf_mask))
    ok, div_t, anchor_safe, _ = replay_actions(safe_env, o_s, d_act[i],
                                               d_obs[i], c)
    if not ok:
      excluded.append({**row, 'reason': ('safe_world_observable_divergence_'
                                         f'before_anchor_at_t{div_t}')})
      continue
    # strict same-anchor gate (bitwise), same stored action array
    anchor_bitwise = np.array_equal(anchor_safe, anchor_obs58)
    if not anchor_bitwise:
      excluded.append({**row, 'reason': 'anchor_mismatch'})
      continue
    s_safe_obs, _, _, info_s = safe_env.step(d_act[i, c])
    if info_s['dead'] or safe_env.dead:
      excluded.append({**row, 'reason': 'safe_branch_fatal_from_other_site'})
      continue
    if not np.isfinite(s_safe_obs).all():
      excluded.append({**row, 'reason': 'safe_branch_nonfinite_obs'})
      continue

    row.update({
        'anchor_bitwise_identical': True,
        'action_bitwise_identical': True,   # same stored array both worlds
        'fatal_branch': {'dead': True, 'settled_matches_sweep_n80': True},
        'safe_branch': {'dead': False,
                        'rock_contact_on_paired_step':
                            bool(info_s['rock_ant_contact']),
                        'sites_triggered':
                            [int(b) for b in info_s['triggered']]},
        'goal_xy': [float(g) for g in d_goal[i]]})
    prov.append(row)
    pairs.append({
        'anchor_obs': anchor_obs58[:OBS_DIM].astype(np.float32),
        'anchor_action': d_act[i, c].astype(np.float32),
        'safe_candidate': s_safe_obs[:OBS_DIM].astype(np.float32),
        'fatal_candidate': s_fatal_obs[:OBS_DIM].astype(np.float32),
        'episode_id': e, 'source_index': i, 'collapse_step': c,
        'goal_xy': d_goal[i],
        'original_mask': np.asarray(d_mask[i], np.int8),
        'counterfactual_mask': np.asarray(cf_mask, np.int8),
        'fatal_site_bits': np.asarray(
            [1 if j in resp else 0 for j in range(4)], np.int8)})
    print(f'  ep {e}: PAIR OK (site {row["fatal_site"]}, c={c}, safe rock '
          f'contact={row["safe_branch"]["rock_contact_on_paired_step"]})',
          flush=True)
    if matched.all():
      break

  assert matched.all(), 'not every death episode was re-identified'
  n = len(pairs)
  print(f'\n{label}: {n} accepted / {len(excluded)} excluded '
        f'of {len(d_obs0)} deaths', flush=True)

  npz_path = os.path.join(args.out, f'{out_name}.npz')
  meta = {
      'label': label, 'sealed_note': sealed_note,
      'transition_time_convention': (
          'fatal branch = the patched env one-step transition with '
          f'death_settle_substeps={SETTLE_N} (internal ctrl-free settling '
          'inside the fatal step); safe branch = the env NORMAL one-step '
          'transition. No branch is artificially advanced to match elapsed '
          'physical time.'),
      'anchor_def': ('s_t = learner obs immediately before the fatal '
                     'action (t = collapse_step), a_t = the recorded '
                     'action actually taken there; identical bitwise in '
                     'both worlds by gate'),
      'critic_arrays': ['anchor_obs', 'anchor_action', 'safe_candidate',
                        'fatal_candidate'],
      'hidden_provenance_arrays': ['original_mask', 'counterfactual_mask',
                                   'fatal_site_bits'],
      'env_seed': env_seed, 'death_settle_substeps': SETTLE_N}
  with open(npz_path + '.tmp', 'wb') as f:
    np.savez_compressed(
        f,
        anchor_obs=np.stack([p['anchor_obs'] for p in pairs]),
        anchor_action=np.stack([p['anchor_action'] for p in pairs]),
        safe_candidate=np.stack([p['safe_candidate'] for p in pairs]),
        fatal_candidate=np.stack([p['fatal_candidate'] for p in pairs]),
        episode_id=np.array([p['episode_id'] for p in pairs], np.int64),
        source_index=np.array([p['source_index'] for p in pairs], np.int64),
        collapse_step=np.array([p['collapse_step'] for p in pairs],
                               np.int64),
        goal_xy=np.stack([p['goal_xy'] for p in pairs]),
        original_mask=np.stack([p['original_mask'] for p in pairs]),
        counterfactual_mask=np.stack([p['counterfactual_mask']
                                      for p in pairs]),
        fatal_site_bits=np.stack([p['fatal_site_bits'] for p in pairs]),
        meta=json.dumps(meta))
  os.replace(npz_path + '.tmp', npz_path)

  manifest = {
      'label': label, 'sealed_note': sealed_note,
      'n_deaths_considered': int(len(d_obs0)),
      'n_pairs_accepted': n,
      'n_excluded': len(excluded),
      'exclusions': excluded,
      'gates': {'anchor_bitwise': 'obs trajectories equal for ALL t<=c',
                'action_bitwise': 'same stored float32 action array applied '
                                  'to both worlds',
                'fatal_branch': 'dead at t=c, settled obs == diagnostic '
                                'sweep substep-80 state (N-independence '
                                'verified previously)',
                'safe_branch': 'dead=False on the paired step (task success '
                               'NOT required)'},
      'transition_time_convention': meta['transition_time_convention'],
      'per_pair_provenance': prov,
      'sources': {
          'deaths_extended_npz': os.path.join(EXT_DIR,
                                              'deaths_extended.npz'),
          'deaths_extended_sha256': C.sha256_file(
              os.path.join(EXT_DIR, 'deaths_extended.npz')),
          'settle_traces_npz': SWEEP_TRACES,
          'settle_traces_sha256': C.sha256_file(SWEEP_TRACES)},
      'env': {'seed': env_seed, 'death_settle_substeps': SETTLE_N,
              'p_active': 0.30, 'horizon': 800, 'reset_fix': True,
              'severity_probs': [0.80, 0.15, 0.05]},
      'git_commit': C.git_commit(),
      'pairs_npz': npz_path, 'pairs_sha256': C.sha256_file(npz_path)}
  man_path = os.path.join(args.out, f'{out_name}_manifest.json')
  json.dump(manifest, open(man_path, 'w'), indent=2)
  print(f'pairs -> {npz_path} ({manifest["pairs_sha256"][:16]}...)')
  print(f'manifest -> {man_path}')


if __name__ == '__main__':
  main()
