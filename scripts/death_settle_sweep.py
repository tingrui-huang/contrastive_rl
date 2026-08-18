"""Death-settle physics calibration sweep + invariance gates (Stages C/D of
the rock-death observability patch).

Replays the two reproducible fatal-episode streams (the authoritative pilot's
16 deaths and the extended collection's 40 fresh deaths) with the PATCHED env
(crl/rockfall_ant.py death_settle_substeps > 0, trace recording ON) and:

  * records the full per-substep settle trace (29-dim ant state after each of
    the N settle substeps of the fatal transition) -- one N=max run yields
    every smaller horizon of the sweep for free (deterministic physics, the
    state after k substeps is independent of the configured N; verified
    empirically below);
  * Gate 1  pre-contact reproduction: obs[0:collapse+1] and act[0:collapse+1]
    (incl. the fatal action, chosen from the unchanged obs[collapse]) must be
    BITWISE identical to the stored authoritative arrays; the settled
    obs[collapse+1] must differ (that is the point of the patch);
  * Gate 5  terminal absorption: the episodes are rolled 10 steps past the
    fatal transition; every post-death obs must be bitwise-frozen at the
    settled observation and reward 0;
  * N-independence check: the first pilot death is re-run with a smaller
    configured N; its settled obs must equal the max-N run's trace at that
    substep;
  * Gate 3 (config side): the frozen rockfall-config check must still pass
    (the patch adds an INSTANCE parameter, no frozen module constant).

--pilot-gate2 additionally rolls out every NON-death pilot episode to the
full horizon under the patched env and verifies obs/act bitwise against the
frozen pilot npz (Gate 2: the patch cannot touch episodes without a fatal
contact). Expensive; run in the background.

Outputs (artifacts/rockfall_death_physics_patch/):
  settle_traces.npz   traces [n_dead, N_max, 29] + pre-hit/settled obs +
                      goal/collapse/source/provenance
  sweep_gates.json    per-episode Gate 1/5 results, N-independence, config
                      gate ('sweep' mode) / gate2_nondeath.json ('gate2')

Analysis/collection only -- no training artifact is touched.
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
from rockfall_v2_teacher import apply_v2_config  # noqa: E402
from collect_rockfall_pilot import check_rockfall_freeze  # noqa: E402
from collect_rockfall_death_extended import (  # noqa: E402
    rollout_extended, PILOT_DIR, PILOT_NAME, P_ACTIVE, HORIZON, MIX,
    FRESH_DATASET_SEED)

OUT_DIR = 'artifacts/rockfall_death_physics_patch'
EXT_DIR = 'artifacts/rockfall_death_extended'
N_MAX = 80                    # 16 x frame_skip; sweep reads {5,10,20,40,80}
GATE5_EXTEND = 10


def make_patched_env(cfg, seed, settle):
  env = apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=seed),
      P_ACTIVE, reset_fix=True)
  env.death_settle_substeps = int(settle)
  env.death_settle_record_trace = True
  return env


def run_death_episode(env, o, mode, side, walker, base_act, extend):
  obs, act, vlen, sc, ep = rollout_extended(
      env, o, walker, base_act, mode, side, HORIZON, extend)
  trace = (np.stack(env._death_settle_trace)
           if env._death_settle_trace else None)
  return obs, act, ep, trace


def gate1_check(obs, act, ep, ref_obs, ref_act, ref_collapse):
  c = int(ref_collapse)
  r = {'collapse_step_match': ep['collapse_step'] == c,
       'obs_precontact_bitwise': bool(
           np.array_equal(obs[:c + 1], ref_obs[:c + 1])),
       'act_incl_fatal_bitwise': bool(
           np.array_equal(act[:c + 1], ref_act[:c + 1])),
       'settled_obs_differs': not bool(
           np.array_equal(obs[c + 1], ref_obs[c + 1]))}
  r['ok'] = (r['collapse_step_match'] and r['obs_precontact_bitwise']
             and r['act_incl_fatal_bitwise'] and r['settled_obs_differs'])
  return r


def gate5_check(obs, ep):
  c, et = int(ep['collapse_step']), int(ep['end_t'])
  post = obs[c + 2:et + 1]
  return {'n_post_steps': int(et - (c + 1)),
          'frozen_at_settled': bool(
              np.all(post == obs[c + 1][None]) if len(post) else True)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT_DIR)
  ap.add_argument('--n-max', type=int, default=N_MAX)
  ap.add_argument('--smoke', action='store_true',
                  help='pilot stream only, first 2 deaths')
  ap.add_argument('--pilot-gate2', action='store_true',
                  help='Gate 2: roll every NON-death pilot episode fully '
                       'under the patched env and compare bitwise')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  # Gate 3 (config side): frozen rockfall config unchanged by the patch.
  rf_ok, rf_diffs, _ = check_rockfall_freeze()
  hard_ok, disc, cinfo = C.check_frozen_integrity()
  assert rf_ok and hard_ok, f'frozen-integrity failure: {rf_diffs + disc}'

  d0 = np.load(os.path.join(PILOT_DIR, f'{PILOT_NAME}.npz'),
               allow_pickle=True)
  s0 = np.load(os.path.join(PILOT_DIR, f'{PILOT_NAME}_sidecar.npz'),
               allow_pickle=True)
  pman = json.load(open(os.path.join(PILOT_DIR, 'pilot_manifest.json')))
  pilot_seed = int(pman['env_seed'])
  dead0 = np.asarray(s0['dead'], bool)
  dead_ids = np.where(dead0)[0]
  if args.smoke:
    dead_ids = dead_ids[:2]

  cfg, walker, base_act, _, _ = C.load_controllers(RP.WALKER, RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'

  # ==================== Gate 2 mode (non-death invariance) ==================
  if args.pilot_gate2:
    env = make_patched_env(cfg, pilot_seed, args.n_max)
    rows = []
    for e in range(len(dead0)):
      o = env.reset()
      mode = str(s0['teacher_mode'][e])
      side = str(s0['base_side'][e])
      obs, act, ep, _ = run_death_episode(env, o, mode, side, walker,
                                          base_act, GATE5_EXTEND)
      if dead0[e]:
        c = int(s0['collapse_step'][e])
        rows.append({'e': int(e), 'kind': 'death',
                     'ok': bool(np.array_equal(obs[:c + 1],
                                               d0['obs'][e, :c + 1]))
                           and ep['collapse_step'] == c})
      else:
        L = int(d0['lengths'][e])          # clean pilot eps store the full
        ok = (bool(np.array_equal(obs[:L], d0['obs'][e, :L]))     # horizon
              and bool(np.array_equal(act[:L - 1], d0['act'][e, :L - 1]))
              and not ep['dead'])
        rows.append({'e': int(e), 'kind': 'clean', 'ok': bool(ok),
                     'success_match':
                         float(ep['success']) == float(s0['success'][e])})
      if (e + 1) % 25 == 0:
        n_ok = sum(r['ok'] for r in rows)
        print(f'  gate2 {e + 1}/{len(dead0)}: ok {n_ok}/{len(rows)}',
              flush=True)
    out = {'n_episodes': len(rows),
           'n_clean': sum(r['kind'] == 'clean' for r in rows),
           'n_death_prefix': sum(r['kind'] == 'death' for r in rows),
           'all_ok': all(r['ok'] for r in rows),
           'success_all_match': all(r.get('success_match', True)
                                    for r in rows),
           'failures': [r for r in rows if not r['ok']],
           'death_settle_substeps': args.n_max,
           'pilot_env_seed': pilot_seed}
    json.dump(out, open(os.path.join(args.out, 'gate2_nondeath.json'), 'w'),
              indent=2)
    print(f"GATE2: all_ok={out['all_ok']} over {out['n_clean']} clean "
          f"episodes (+{out['n_death_prefix']} death prefixes)", flush=True)
    return 0

  # ==================== sweep mode ==========================================
  kept, g1_rows, g5_rows = [], [], []

  # ---- pilot stream (16 deaths) ----
  env = make_patched_env(cfg, pilot_seed, args.n_max)
  for e in range(int(dead_ids.max()) + 1 if len(dead_ids) else 0):
    o = env.reset()
    if e not in dead_ids:
      continue
    mode, side = str(s0['teacher_mode'][e]), str(s0['base_side'][e])
    obs, act, ep, trace = run_death_episode(env, o, mode, side, walker,
                                            base_act, GATE5_EXTEND)
    assert trace is not None and len(trace) == args.n_max
    g1 = gate1_check(obs, act, ep, d0['obs'][e], d0['act'][e],
                     s0['collapse_step'][e])
    g5 = gate5_check(obs, ep)
    g1_rows.append({'stream': 'pilot', 'e': int(e), **g1})
    g5_rows.append({'stream': 'pilot', 'e': int(e), **g5})
    c = int(ep['collapse_step'])
    kept.append({'trace': trace, 'settled_obs58': obs[c + 1].copy(),
                 'prehit_state': obs[c, :29].copy(),
                 'legacy_frozen_state': d0['obs'][e, c + 1, :29].copy(),
                 'goal_xy': ep['goal_xy'], 'collapse_step': c,
                 'source': 'pilot', 'episode_id': int(e),
                 'env_seed': pilot_seed})
    print(f"  pilot ep {e}: G1 {g1['ok']} G5 {g5['frozen_at_settled']}",
          flush=True)
    if not g1['ok']:
      print('ABORT: Gate 1 failed on the pilot stream.')
      return 3

  # ---- N-independence: rerun FIRST pilot death with settle = n_max // 4 ----
  n_small = max(args.n_max // 4, 1)
  env2 = make_patched_env(cfg, pilot_seed, n_small)
  first_dead = int(dead_ids[0])
  for e in range(first_dead + 1):
    o = env2.reset()
  obs2, _, ep2, trace2 = run_death_episode(
      env2, o, str(s0['teacher_mode'][first_dead]),
      str(s0['base_side'][first_dead]), walker, base_act, GATE5_EXTEND)
  n_indep = bool(np.array_equal(trace2[n_small - 1],
                                kept[0]['trace'][n_small - 1]))
  n_indep_obs = bool(np.array_equal(
      obs2[ep2['collapse_step'] + 1, :29], kept[0]['trace'][n_small - 1]))
  print(f'N-independence (N={n_small} vs trace of N={args.n_max}): '
        f'{n_indep and n_indep_obs}', flush=True)

  # ---- fresh stream (40 deaths) ----
  if not args.smoke:
    ext = np.load(os.path.join(EXT_DIR, 'deaths_extended.npz'),
                  allow_pickle=True)
    eman = json.load(open(os.path.join(EXT_DIR, 'manifest.json')))
    fresh_seed = int(eman['phase_b']['env_seed'])
    n_eps_b = int(eman['phase_b']['n_episodes'])
    is_fresh = ext['source'] == 'fresh'
    f_obs0 = ext['obs'][is_fresh][:, 0]          # initial obs, 58-dim
    f_obs = ext['obs'][is_fresh]
    f_act = ext['act'][is_fresh]
    f_col = np.asarray(ext['collapse_step'], np.int64)[is_fresh]
    f_mode = np.asarray(ext['teacher_mode'])[is_fresh]
    f_side = np.asarray(ext['base_side'])[is_fresh]
    matched = np.zeros(len(f_obs0), bool)
    # regenerate the phase-B mode/side streams (collector convention)
    ds_rng = np.random.default_rng(FRESH_DATASET_SEED)
    side_rng = np.random.default_rng(FRESH_DATASET_SEED + 1)
    n_planned = 1500                       # collector's --max-episodes
    n_cover = int(round(MIX['coverage'] * n_planned))
    modes = np.array(['sighted'] * (n_planned - n_cover)
                     + ['coverage'] * n_cover)
    ds_rng.shuffle(modes)

    envB = make_patched_env(cfg, fresh_seed, args.n_max)
    for e in range(n_eps_b):
      mode = str(modes[e])
      side = 'left' if side_rng.random() < 0.5 else 'right'
      o = envB.reset()
      hits = np.where(np.all(f_obs0 == o[None], axis=1))[0]
      if not len(hits):
        continue
      i = int(hits[0])
      assert not matched[i], 'duplicate initial-obs match'
      matched[i] = True
      assert mode == str(f_mode[i]) and side == str(f_side[i]), (
          'regenerated mode/side stream disagrees with the recorded episode')
      obs, act, ep, trace = run_death_episode(envB, o, mode, side, walker,
                                              base_act, GATE5_EXTEND)
      assert trace is not None and len(trace) == args.n_max
      g1 = gate1_check(obs, act, ep, f_obs[i], f_act[i], f_col[i])
      g5 = gate5_check(obs, ep)
      g1_rows.append({'stream': 'fresh', 'e': int(e), **g1})
      g5_rows.append({'stream': 'fresh', 'e': int(e), **g5})
      c = int(ep['collapse_step'])
      kept.append({'trace': trace, 'settled_obs58': obs[c + 1].copy(),
                   'prehit_state': obs[c, :29].copy(),
                   'legacy_frozen_state': f_obs[i, c + 1, :29].copy(),
                   'goal_xy': ep['goal_xy'], 'collapse_step': c,
                   'source': 'fresh', 'episode_id': int(e),
                   'env_seed': fresh_seed})
      if not g1['ok']:
        print(f'ABORT: Gate 1 failed on fresh ep {e}.')
        return 3
      if matched.all():
        break
    print(f'fresh stream: matched {int(matched.sum())}/{len(matched)} deaths',
          flush=True)
    assert matched.all(), 'not every fresh death episode was re-identified'

  # ---- save --------------------------------------------------------------
  traces = np.stack([k['trace'] for k in kept])          # [n, N_max, 29]
  npz_path = os.path.join(args.out, 'settle_traces.npz')
  meta = {'n_max': args.n_max, 'frame_skip': 5, 'substep_dt': 0.02,
          'protocol': 'local_detour_v2.1_sev0.80_p30_h800_resetfix_v1',
          'trace_def': ('29-dim ant state (qpos[:15], qvel[:14] order as in '
                        'the learner obs) after each settle substep of the '
                        'fatal transition, actor ctrl zeroed at the fatal '
                        'contact substep')}
  with open(npz_path + '.tmp', 'wb') as f:
    np.savez_compressed(
        f, traces=traces,
        settled_obs58=np.stack([k['settled_obs58'] for k in kept]),
        prehit_state=np.stack([k['prehit_state'] for k in kept]),
        legacy_frozen_state=np.stack([k['legacy_frozen_state']
                                      for k in kept]),
        goal_xy=np.stack([k['goal_xy'] for k in kept]),
        collapse_step=np.array([k['collapse_step'] for k in kept], np.int64),
        source=np.array([k['source'] for k in kept]),
        episode_id=np.array([k['episode_id'] for k in kept], np.int64),
        env_seed=np.array([k['env_seed'] for k in kept], np.int64),
        meta=json.dumps(meta))
  os.replace(npz_path + '.tmp', npz_path)

  gates = {
      'gate1_precontact_reproduction': {
          'n': len(g1_rows), 'all_ok': all(r['ok'] for r in g1_rows),
          'rows': g1_rows},
      'gate5_terminal_absorbing': {
          'n': len(g5_rows),
          'all_frozen': all(r['frozen_at_settled'] for r in g5_rows),
          'rows': g5_rows},
      'n_independence_check': {'n_small': n_small, 'ok': n_indep
                               and n_indep_obs},
      'gate3_frozen_config_check_passes': bool(rf_ok and hard_ok),
      'death_settle_substeps_max': args.n_max,
      'git_commit': C.git_commit(),
      'walker_sha256': cinfo['walker_sha256'],
      'base_policy_sha256': cinfo['base_sha256'],
      'n_death_episodes': len(kept),
      'npz': npz_path, 'npz_sha256': C.sha256_file(npz_path)}
  json.dump(gates, open(os.path.join(args.out, 'sweep_gates.json'), 'w'),
            indent=2)
  print(f"\nGATES: g1 {gates['gate1_precontact_reproduction']['all_ok']} "
        f"g5 {gates['gate5_terminal_absorbing']['all_frozen']} "
        f"n-indep {gates['n_independence_check']['ok']} | "
        f"{len(kept)} deaths -> {npz_path}", flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
