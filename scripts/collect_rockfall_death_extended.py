"""Extended-horizon rock-death collection (probe-local; D1 of the
"is rock-death observable, and when?" diagnostic).

DATA COLLECTION ONLY. Does not touch the frozen pilot npz, the failure split,
the failure bank, or any training artifact. Output goes to a NEW directory.

Two phases, identical env protocol to the authoritative pilot
(rockfall_v2_p30_h800_resetfix: p_active=0.30, H=800, severity 0.80/0.15/0.05,
resetfix_v1, frozen walker + base policy, 90/0/10 teacher mixture):

  A(replay)  Re-create the pilot's env (SAME env_seed) and replay its reset
             stream episode-for-episode. Every episode consumes exactly its
             reset RNG draws; the 16 recorded dead episodes are additionally
             ROLLED OUT with the recorded teacher mode/side and, instead of
             the pilot's truncation at collapse_step+2, stepped on to
             collapse_step + EXTEND (or the horizon). The re-collected prefix
             obs[0:collapse+2] / act[0:collapse+1] must match the frozen
             pilot BITWISE (per-episode check recorded in the manifest; any
             mismatch aborts). Valid because the env consumes RNG only in
             reset (rockfall_ant.py: _begin_episode + goal sampling; step()
             draws nothing) and resetfix_v1 makes resets state-independent --
             the bitwise check verifies both assumptions empirically.

  B(fresh)   New prescreened env_seed (disjoint from all consumed seeds),
             fresh mode/side streams built with the pilot's exact convention,
             episodes collected until >= --fresh-deaths NEW death episodes.
             Dead episodes are kept in full (extended past collapse as in A);
             clean episodes are counted, summarized, and discarded. A clean
             episode may be aborted once death is impossible (past the
             handoff with every drop resolved) -- see --no-early-abort.

Also records, per extended death episode, whether the post-death observation
stream is FROZEN (obs[collapse+1+k] bitwise == obs[collapse+1] for all k),
which crl/rockfall_ant.py step() implies (dead => returns _last_obs without
mj_step). This is the load-bearing structural fact for the D2/D3 analysis.

Usage:
  python scripts/collect_rockfall_death_extended.py            # full run
  python scripts/collect_rockfall_death_extended.py --smoke    # phase A on
                                                               # first 2 deaths
"""
import argparse
import json
import os
import sys

import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod          # noqa: E402
from crl import rockfall_ant as RA        # noqa: E402
import litter_pilot_common as C           # noqa: E402
import rockfall_pilot as RP               # noqa: E402
import rockfall_v2_teacher as V2          # noqa: E402
from rockfall_v2_teacher import (apply_v2_config, SEVERITY_V2,  # noqa: E402
                                 protocol_version)
from collect_rockfall_pilot import (check_rockfall_freeze, prescreen_env_seed,
                                    CONSUMED)  # noqa: E402
from collect_rockfall_v2_pilot import V2_CONSUMED  # noqa: E402

PILOT_DIR = 'artifacts/rockfall_v2_p30_h800_resetfix/pilot'
PILOT_NAME = 'antmaze_rockfall_v2_p30_h800_resetfix_pilot'
OUT_DIR = 'artifacts/rockfall_death_extended'
P_ACTIVE = 0.30
HORIZON = 800
EXTEND = 50                    # steps recorded past collapse_step
MIX = {'sighted': 0.90, 'blind': 0.00, 'coverage': 0.10}   # pilot convention
#: fresh phase-B seeds (offset far from every pilot/dev base; prescreen +
#: clash check still enforce disjointness explicitly)
FRESH_SEED_BASE = 71_400_019
FRESH_DATASET_SEED = 71_990_013
#: clean-episode early abort: past the handoff line with margin and every
#: drop resolved for this many steps, a rockfall death is impossible (all
#: trigger windows end by x=5.5; rocks land within ~5 steps of their drop).
ABORT_X = 6.2
ABORT_DROP_QUIET = 25


def rollout_extended(env, o, walker, base_act, mode, base_side,
                     horizon=HORIZON, extend=EXTEND):
  """The pilot's rollout (collect_rockfall_v2_pilot.rollout) with ONE change:
  a death does not break the loop -- stepping continues to
  collapse_step + extend (capped at the horizon), applying the same teacher
  law throughout. Returns the same (obs, act, valid_len, sc, ep) contract
  plus ep['end_t'] = last recorded obs index."""
  L = horizon + 1
  true_goal = o[29:31].copy()
  obs = np.zeros((L, 58), np.float32)
  act = np.zeros((L, 8), np.float32)
  obs[0] = o
  sc = {k: np.full(L, np.nan, np.float32) for k in
        ('lane_cmd', 'speed_cmd', 'torso_x', 'torso_y', 'vx')}
  sc.update({k: np.zeros(L, np.float32) for k in
             ('handoff', 'rock_ant_contact', 'dead', 'in_detour')})

  is_center = mode == 'coverage'
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = (V2.active_site_windows(base_sgn, env.rockfall_mask)
          if mode == 'sighted' else [])
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  dead_at, hit = -1, 0.0
  end_t = horizon
  for t in range(horizon):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
      lane_cmd = speed_cmd = np.nan
      in_det = False
    else:
      x_hist.append(x)
      if is_center:
        y_cmd, v_cmd = RP.route_command('center', t, x_hist, nudge)
        in_det = False
      else:
        in_det = any(x0 <= x <= x1 for x0, x1 in wins)
        y_cmd, v_cmd = V2.detour_command(base_sgn, wins, x, t, x_hist, nudge,
                                         RP.V_SIDE)
      a = walker(o, y_cmd, v_cmd)
      lane_cmd, speed_cmd = float(y_cmd), float(v_cmd)
    o2, r, _, info = env.step(a)
    act[t] = a
    obs[t + 1] = o2
    sc['handoff'][t] = float(handoff)
    sc['lane_cmd'][t] = lane_cmd
    sc['speed_cmd'][t] = speed_cmd
    sc['torso_x'][t] = float(o2[0])
    sc['torso_y'][t] = float(o2[1])
    sc['vx'][t] = float(env._env.data.qvel[0])
    sc['rock_ant_contact'][t] = float(bool(info['rock_ant_contact']))
    sc['dead'][t] = float(bool(info['dead']))
    sc['in_detour'][t] = float(bool(in_det))
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if dead_at >= 0 and t >= dead_at + extend:   # extended stop (was +2 break)
      end_t = t + 1
      break
    o = o2
  else:
    end_t = horizon
  drop_steps = env._drop_step
  hit_steps = env._hit_step
  route = 'center' if is_center else base_side
  valid_len = min(dead_at + 2, L) if dead_at >= 0 else end_t + 1  # pilot rule
  ep = {'rockfall_mask': np.asarray(env.rockfall_mask, np.int8),
        'severity': np.asarray(env.privileged_severity),
        'triggered': np.asarray(env._triggered, bool),
        'dropped': np.asarray(env._dropped, bool),
        'hit': np.asarray(env._hit, bool),
        'first_drop_step': int(min(drop_steps.values()) if drop_steps
                               else -1),
        'first_hit_step': int(min(hit_steps.values()) if hit_steps else -1),
        'impaired': bool(env._impaired_leg_ids),
        'teacher_mode': mode, 'route': route, 'base_side': base_side,
        'collapse_step': int(dead_at), 'dead': bool(dead_at >= 0),
        'success': float(hit), 'ep_length': int(valid_len),
        'end_t': int(end_t), 'goal_xy': true_goal.astype(np.float32)}
  return obs, act, valid_len, sc, ep


def rollout_fresh(env, o, walker, base_act, mode, base_side,
                  horizon=HORIZON, extend=EXTEND, early_abort=True):
  """Phase-B episode: same law; clean episodes may abort once death is
  impossible (past ABORT_X with every drop >= ABORT_DROP_QUIET steps old).
  Returns rollout_extended's contract; aborted episodes have ep['aborted']."""
  L = horizon + 1
  true_goal = o[29:31].copy()
  obs = np.zeros((L, 58), np.float32)
  act = np.zeros((L, 8), np.float32)
  obs[0] = o
  is_center = mode == 'coverage'
  base_sgn = 1.0 if base_side == 'left' else -1.0
  wins = (V2.active_site_windows(base_sgn, env.rockfall_mask)
          if mode == 'sighted' else [])
  handoff = False
  x_hist, nudge = [], {'until': -1, 'sign': 1.0}
  dead_at, hit = -1, 0.0
  aborted = False
  sc = {k: np.zeros(L, np.float32) for k in
        ('rock_ant_contact', 'dead', 'torso_x', 'torso_y', 'vx')}
  end_t = horizon
  for t in range(horizon):
    x, y = float(o[0]), float(o[1])
    if not handoff and (x >= RP.HANDOFF_X or y >= 2.0):
      handoff = True
    if handoff:
      oc = o.copy()
      oc[29:] = 0.0
      oc[29:31] = true_goal
      a = np.asarray(base_act(jnp.asarray(oc[None]))[0])
    else:
      x_hist.append(x)
      if is_center:
        y_cmd, v_cmd = RP.route_command('center', t, x_hist, nudge)
      else:
        y_cmd, v_cmd = V2.detour_command(base_sgn, wins, x, t, x_hist, nudge,
                                         RP.V_SIDE)
      a = walker(o, y_cmd, v_cmd)
    o2, r, _, info = env.step(a)
    act[t] = a
    obs[t + 1] = o2
    sc['rock_ant_contact'][t] = float(bool(info['rock_ant_contact']))
    sc['dead'][t] = float(bool(info['dead']))
    sc['torso_x'][t] = float(o2[0])
    sc['torso_y'][t] = float(o2[1])
    sc['vx'][t] = float(env._env.data.qvel[0])
    hit = max(hit, float(r))
    if info['dead'] and dead_at < 0:
      dead_at = t
    if dead_at >= 0 and t >= dead_at + extend:
      end_t = t + 1
      break
    if (early_abort and dead_at < 0 and float(o2[0]) >= ABORT_X
        and all(t - s >= ABORT_DROP_QUIET for s in env._drop_step.values())):
      aborted = True                       # death now impossible; stop early
      end_t = t + 1
      break
    o = o2
  else:
    end_t = horizon
  ep = {'rockfall_mask': np.asarray(env.rockfall_mask, np.int8),
        'severity': np.asarray(env.privileged_severity),
        'hit': np.asarray(env._hit, bool),
        'first_hit_step': int(min(env._hit_step.values())
                              if env._hit_step else -1),
        'teacher_mode': mode, 'base_side': base_side,
        'collapse_step': int(dead_at), 'dead': bool(dead_at >= 0),
        'success': float(hit), 'aborted': bool(aborted), 'end_t': int(end_t),
        'goal_xy': true_goal.astype(np.float32)}
  return obs, act, sc, ep


def frozen_check(obs, c, end_t):
  """Is the post-death obs stream frozen at obs[c+1]? (expected True)."""
  if end_t <= c + 1:
    return True, 0
  same = np.all(obs[c + 2:end_t + 1] == obs[c + 1][None], axis=1)
  return bool(same.all()), int(end_t - (c + 1))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT_DIR)
  ap.add_argument('--fresh-deaths', type=int, default=40,
                  help='NEW death episodes to collect in phase B (total = '
                       '16 replayed + this)')
  ap.add_argument('--max-episodes', type=int, default=1500,
                  help='phase-B episode cap (runaway guard)')
  ap.add_argument('--extend', type=int, default=EXTEND)
  ap.add_argument('--smoke', action='store_true',
                  help='phase A only, first 2 dead episodes (fast check)')
  ap.add_argument('--no-early-abort', action='store_true',
                  help='phase B: run clean episodes to the full horizon '
                       'instead of aborting once death is impossible')
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)

  # ---- frozen-stack integrity (same gates as the pilot collector) ----------
  hard_ok, disc, cinfo = C.check_frozen_integrity()
  rf_ok, rf_diffs, rf_man = check_rockfall_freeze()
  if not (hard_ok and rf_ok):
    print('ABORT: frozen-integrity failure.', disc + rf_diffs)
    return 2

  pman = json.load(open(os.path.join(PILOT_DIR, 'pilot_manifest.json')))
  orig_env_seed = int(pman['env_seed'])
  assert pman['p_active'] == P_ACTIVE and pman['horizon'] == HORIZON
  assert pman['reset_fix'] is True
  d0 = np.load(os.path.join(PILOT_DIR, f'{PILOT_NAME}.npz'),
               allow_pickle=True)
  s0 = np.load(os.path.join(PILOT_DIR, f'{PILOT_NAME}_sidecar.npz'),
               allow_pickle=True)
  dead0 = np.asarray(s0['dead'], bool)
  n_pilot = len(dead0)
  dead_ids = np.where(dead0)[0]
  if args.smoke:
    dead_ids = dead_ids[:2]
  print(f'pilot: {n_pilot} eps, {dead0.sum()} dead '
        f'(replaying {len(dead_ids)}), env_seed {orig_env_seed}', flush=True)

  cfg, walker, base_act, base_step, wmeta = C.load_controllers(RP.WALKER,
                                                               RP.BASE)
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'

  kept = []          # per-death dict: obs/act/sc arrays + ep row + provenance

  # ================= Phase A: replay + bitwise verify + extend ==============
  env = apply_v2_config(
      envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=orig_env_seed),
      P_ACTIVE, reset_fix=True)
  bitwise = []
  replay_upto = int(dead_ids.max()) if len(dead_ids) else -1
  for e in range(replay_upto + 1):
    o = env.reset()                       # consume this episode's reset draws
    if e not in dead_ids:
      continue
    mode = str(s0['teacher_mode'][e])
    side = str(s0['base_side'][e])
    obs, act, vlen, sc, ep = rollout_extended(
        env, o, walker, base_act, mode, side, HORIZON, args.extend)
    c = int(s0['collapse_step'][e])
    ok_obs = bool(np.array_equal(obs[:c + 2], d0['obs'][e, :c + 2]))
    ok_act = bool(np.array_equal(act[:c + 1], d0['act'][e, :c + 1]))
    ok_c = ep['collapse_step'] == c
    bitwise.append({'episode_id': int(e), 'collapse_step': c,
                    'obs_prefix_bitwise': ok_obs, 'act_prefix_bitwise': ok_act,
                    'collapse_step_match': ok_c})
    print(f'  replay ep {e}: collapse {c} -> end_t {ep["end_t"]} | '
          f'obs bitwise {ok_obs} act bitwise {ok_act}', flush=True)
    if not (ok_obs and ok_act and ok_c):
      json.dump({'phase_a_bitwise': bitwise, 'verdict': 'REPRODUCTION FAILED'},
                open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
      print('ABORT: extended re-collection does not reproduce the original '
            'prefix bitwise; states are not comparable. See manifest.')
      return 3
    fro, n_post = frozen_check(obs, c, ep['end_t'])
    kept.append({'obs': obs, 'act': act, 'sc': sc, 'ep': ep,
                 'source': 'replay_orig', 'orig_episode_id': int(e),
                 'env_seed': orig_env_seed, 'frozen_after_death': fro,
                 'n_post_frames': n_post})
  n_replayed = len(kept)
  print(f'phase A done: {n_replayed} deaths replayed+extended, all bitwise OK',
        flush=True)

  # ================= Phase B: fresh episodes to >= fresh-deaths =============
  clean_rows = []
  fresh_seed = None
  prescreen_freq = None
  if not args.smoke and args.fresh_deaths > 0:
    exclude = (V2_CONSUMED + [orig_env_seed, pman['dataset_rng_seed'],
                              pman['base_side_rng_seed']])
    fresh_seed, prescreen_freq = prescreen_env_seed(
        args.max_episodes, [FRESH_SEED_BASE + 97 * k for k in range(400)],
        exclude=exclude, p_active=P_ACTIVE)
    clash = C.seed_reuse(exclude, [fresh_seed], [FRESH_DATASET_SEED])
    assert not clash, f'seed clash {clash}'
    print(f'phase B: env_seed {fresh_seed} prescreen {prescreen_freq}',
          flush=True)
    envB = apply_v2_config(
        envs_mod.make_env('offline_ant_umaze_rockfall', cfg, seed=fresh_seed),
        P_ACTIVE, reset_fix=True)
    # mode/side streams: pilot convention (shuffled 90/0/10 block + side rng)
    ds_rng = np.random.default_rng(FRESH_DATASET_SEED)
    side_rng = np.random.default_rng(FRESH_DATASET_SEED + 1)
    n_max = args.max_episodes
    n_cover = int(round(MIX['coverage'] * n_max))
    modes = np.array(['sighted'] * (n_max - n_cover) + ['coverage'] * n_cover)
    ds_rng.shuffle(modes)
    n_fresh_dead = 0
    for e in range(n_max):
      mode = str(modes[e])
      side = 'left' if side_rng.random() < 0.5 else 'right'
      o = envB.reset()
      obs, act, sc, ep = rollout_fresh(
          envB, o, walker, base_act, mode, side, HORIZON, args.extend,
          early_abort=not args.no_early_abort)
      if ep['dead']:
        c = int(ep['collapse_step'])
        fro, n_post = frozen_check(obs, c, ep['end_t'])
        kept.append({'obs': obs, 'act': act, 'sc': sc, 'ep': ep,
                     'source': 'fresh', 'orig_episode_id': -1,
                     'fresh_episode_index': int(e), 'env_seed': fresh_seed,
                     'frozen_after_death': fro, 'n_post_frames': n_post})
        n_fresh_dead += 1
      else:
        clean_rows.append({'e': int(e), 'mode': mode, 'side': side,
                           'success': ep['success'],
                           'aborted': ep['aborted'], 'end_t': ep['end_t'],
                           'hit_any': bool(ep['hit'].any())})
      if (e + 1) % 50 == 0 or ep['dead']:
        print(f'  phase B ep {e + 1}/{n_max}: deaths {n_fresh_dead}/'
              f'{args.fresh_deaths}', flush=True)
      if n_fresh_dead >= args.fresh_deaths:
        break
    n_episodes_b = e + 1
    print(f'phase B done: {n_fresh_dead} fresh deaths in {n_episodes_b} eps '
          f'(death rate {n_fresh_dead / n_episodes_b:.3f})', flush=True)

  # ================= write artifact =========================================
  n = len(kept)
  Lmax = max(k['ep']['end_t'] + 1 for k in kept)
  obs_a = np.zeros((n, Lmax, 58), np.float32)
  act_a = np.zeros((n, Lmax, 8), np.float32)
  step_keys = ('rock_ant_contact', 'dead', 'torso_x', 'torso_y', 'vx')
  sc_a = {k: np.zeros((n, Lmax), np.float32) for k in step_keys}
  for i, k in enumerate(kept):
    et = k['ep']['end_t']
    obs_a[i, :et + 1] = k['obs'][:et + 1]
    act_a[i, :et + 1] = k['act'][:et + 1]
    for key in step_keys:
      sc_a[key][i, :et + 1] = k['sc'][key][:et + 1]
  ep_arr = {
      'collapse_step': np.array([k['ep']['collapse_step'] for k in kept],
                                np.int64),
      'end_t': np.array([k['ep']['end_t'] for k in kept], np.int64),
      'first_hit_step': np.array([k['ep']['first_hit_step'] for k in kept],
                                 np.int64),
      'goal_xy': np.stack([k['ep']['goal_xy'] for k in kept]),
      'teacher_mode': np.array([k['ep']['teacher_mode'] for k in kept]),
      'base_side': np.array([k['ep']['base_side'] for k in kept]),
      'rockfall_mask': np.stack([k['ep']['rockfall_mask'] for k in kept]),
      'severity': np.stack([k['ep']['severity'] for k in kept]),
      'source': np.array([k['source'] for k in kept]),
      'orig_episode_id': np.array([k['orig_episode_id'] for k in kept],
                                  np.int64),
      'env_seed': np.array([k['env_seed'] for k in kept], np.int64),
      'frozen_after_death': np.array([k['frozen_after_death'] for k in kept],
                                     bool),
      'n_post_frames': np.array([k['n_post_frames'] for k in kept], np.int64),
  }
  meta = {'env_name': 'offline_ant_umaze_rockfall', 'obs_dim': 29,
          'goal_dim': 29, 'action_dim': 8,
          'protocol': protocol_version(P_ACTIVE) + '_h800_resetfix_v1',
          'extend_steps': args.extend,
          'note': ('Extended rock-death episodes (probe-local; NOT a training '
                   'set, NOT the failure bank). Dead episodes stepped to '
                   'collapse_step+extend instead of the pilot truncation at '
                   'collapse_step+2.')}
  npz_path = os.path.join(args.out, 'deaths_extended.npz')
  tmp = npz_path + '.tmp'
  with open(tmp, 'wb') as f:
    np.savez_compressed(f, obs=obs_a, act=act_a,
                        **{f'step_{k}': v for k, v in sc_a.items()},
                        **ep_arr, meta=json.dumps(meta))
  os.replace(tmp, npz_path)

  frozen_all = bool(ep_arr['frozen_after_death'].all())
  man = {
      'protocol': meta['protocol'], 'p_active': P_ACTIVE, 'horizon': HORIZON,
      'reset_fix': True, 'severity_probs_v2': list(SEVERITY_V2),
      'extend_steps': args.extend,
      'git_commit': C.git_commit(),
      'rockfall_frozen_code_commit': rf_man.get('frozen_code_commit'),
      'walker_sha256': cinfo['walker_sha256'],
      'base_policy_sha256': cinfo['base_sha256'],
      'walker_step': int(wmeta['step']), 'base_policy_step': int(base_step),
      'pilot_manifest': os.path.join(PILOT_DIR, 'pilot_manifest.json'),
      'pilot_env_seed': orig_env_seed,
      'phase_a': {'n_replayed_deaths': n_replayed,
                  'bitwise_checks': bitwise,
                  'all_bitwise_ok': all(b['obs_prefix_bitwise']
                                        and b['act_prefix_bitwise']
                                        for b in bitwise)},
      'phase_b': ({'env_seed': fresh_seed,
                   'dataset_seed': FRESH_DATASET_SEED,
                   'prescreen_freq': prescreen_freq,
                   'early_abort': not args.no_early_abort,
                   'abort_rule': f'x >= {ABORT_X} and every drop >= '
                                 f'{ABORT_DROP_QUIET} steps old',
                   'n_episodes': int(n_episodes_b),
                   'n_fresh_deaths': int(n_fresh_dead),
                   'death_rate': round(n_fresh_dead / n_episodes_b, 4),
                   'clean_summary': {
                       'n_clean': len(clean_rows),
                       'n_aborted': int(sum(r['aborted']
                                            for r in clean_rows)),
                       'success_rate_completed': (round(float(np.mean(
                           [r['success'] for r in clean_rows
                            if not r['aborted']])), 3)
                           if any(not r['aborted'] for r in clean_rows)
                           else None)}}
                  if fresh_seed is not None else None),
      'n_death_episodes_total': n,
      'post_death_obs_frozen_all_episodes': frozen_all,
      'post_death_frames_checked': int(ep_arr['n_post_frames'].sum()),
      'npz': npz_path, 'npz_sha256': C.sha256_file(npz_path),
  }
  json.dump(man, open(os.path.join(args.out, 'manifest.json'), 'w'), indent=2)
  print(f'\nwrote {npz_path} ({n} death episodes; '
        f'post-death obs frozen in all: {frozen_all})')
  print(f"manifest -> {os.path.join(args.out, 'manifest.json')}")
  return 0


if __name__ == '__main__':
  sys.exit(main())
