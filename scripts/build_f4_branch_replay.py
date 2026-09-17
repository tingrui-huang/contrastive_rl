"""Branch replay: every recorded (s_t, a_t) keeps its anchor, its future comes from the model.

Step 11 showed that the critic learns the fork's action dependence once the
sighted teacher's recorded continuation is no longer among its positives at
the fork -- but it did so by selecting rows with a provenance label at one
place, which is a diagnosis, not a method.  The fair rule is uniform: the
positive-future distribution of the critic is the interventional one,
p(g | s, do(a)), estimated by rolling the fixed ETT out of EVERY recorded
anchor, and the recorded futures are used to fit the ETT, never as
positives.  Vanilla CRL is the same recipe with the recorded continuation in
place of the model's.

For each original episode e of replay_D's 3300-episode subset and each
anchor time t in [0, 49]: state[0] = the recorded s_t, action[0] = the
recorded a_t, then the fixed ETT / nominal / continuation actor exactly as
arms C, D and E generated their query paths, for the 50 - t steps that
remain of the episode's horizon, so the path ends at the same absolute time
as the record.  The 7,700 query paths of replay_E (arms C, D, E: a queried
first action at one of the 550 contexts, then the same continuation) are
kept as they are -- they have the same form.  Anchors are the first row of
every path (TrajectoryBuffer.set_anchor_strata with row 0 in the driver);
later rows are goals only.

Recorded frozen rows (four identical frames after a death) are handled by
the model like every other state; how often the model moves out of a
recorded stationary state is reported, not corrected.

Output: outputs/pointmaze_branch_replay_v1/replay_P.npz (+ generation.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_f4_corner_coverage_fix import (   # noqa: E402
    CONFIG, FIX, GOAL, PILOT, REPLAY_E, load_npz, make_network, sha256, write_json)
sys.path.insert(0, str(PILOT))

OUT = ROOT / 'outputs' / 'pointmaze_branch_replay_v1'
REPLAY_P = OUT / 'replay_P.npz'
GEN_SEED = int(CONFIG['seed']) + 30_000        # distinct stream from C (0), D (+10k), E (+20k)
H = 50                                         # obs rows 0..50 per recorded episode


def generate(anchor_times, chunk, frozen_stay=False):
  import jax
  import jax.numpy as jnp
  import torch
  from crl import checkpoint
  from learned_ett import legal_numpy, load_checkpoint
  from propensity.nominal_policy import load_nominal_policy
  torch.set_num_threads(8)
  model, _ = load_checkpoint(ROOT / CONFIG['inputs']['ett_checkpoint'], 'cpu')
  nominal = load_nominal_policy((ROOT / CONFIG['inputs']['nominal']).parent)
  _, actor_state = checkpoint.load_checkpoint(ROOT / CONFIG['inputs']['rollout_actor'])
  network = make_network()

  @jax.jit
  def actor_sample(params, obs, key):
    return network.sample(network.policy_network.apply(params, obs), key)

  d = load_npz(FIX / 'replay_D.npz')
  orig = np.flatnonzero(d['audit_source'] == 0)
  obs_o, act_o = d['obs'][orig, :, :8], d['act'][orig]
  n_eps = len(orig)
  rows = []
  t0 = time.time()
  for t in anchor_times:
    steps = H - t                                    # model steps; path has steps + 1 obs rows
    for c0 in range(0, n_eps, chunk):
      sl = slice(c0, min(c0 + chunk, n_eps))
      n = sl.stop - sl.start
      state = np.zeros((n, H + 1, 8), np.float32)
      action = np.zeros((n, H + 1, 2), np.float32)
      failed = np.zeros((n, H + 1), bool)
      onset = np.zeros((n, H), bool)
      state[:, 0] = obs_o[sl, t]
      if frozen_stay and t >= 4:
        # the F4 death signature -- four identical frames after the reset fill --
        # is carried forward as an already-absorbed path (the record itself
        # never leaves such a state); a variant, not the uniform rule
        failed[:, 0] = np.all(obs_o[sl, t, 2:] == np.tile(obs_o[sl, t, :2], 3), axis=1)
      rng = torch.Generator(device='cpu')
      rng.manual_seed(GEN_SEED + 2000 + 100 * t + c0)
      for step in range(steps):
        cur = state[:, step]
        goal = np.broadcast_to(GOAL, cur.shape).astype(np.float32)
        xb = np.asarray(nominal.sample(
            jnp.asarray(cur), jax.random.PRNGKey(GEN_SEED + 3000 + 100 * t + step),
            1, goal=jnp.asarray(goal)), np.float32)
        obs = jnp.asarray(np.concatenate([cur, goal], axis=1))
        a_actor = np.asarray(actor_sample(actor_state.policy_params, obs,
                                          jax.random.PRNGKey(GEN_SEED + 4000 + 100 * t + step)),
                             np.float32)
        xq = act_o[sl, t].astype(np.float32).copy() if step == 0 else a_actor
        action[:, step] = xq
        with torch.no_grad():
          succ, event, _, _ = model.sample_live(
              torch.as_tensor(cur), torch.as_tensor(xb), torch.as_tensor(xq), rng)
        succ = succ.numpy().astype(np.float32)
        event = event.numpy()
        was = failed[:, step]
        persistent = np.concatenate([cur[:, :2], cur[:, :6]], axis=1)
        succ[was] = persistent[was]
        event[was] = False
        state[:, step + 1] = succ
        failed[:, step + 1] = was | event
        onset[:, step] = event
      valid = steps + 1
      if np.any(state[:, 1:valid, 2:] != state[:, :valid - 1, :6]):
        raise AssertionError('F4 shift violation')
      if not legal_numpy(state[:, :valid, :2]).all():
        raise AssertionError('illegal emitted endpoint')
      # rows past the horizon: repeat the last valid row (never sampled; lengths mask them)
      state[:, valid:] = state[:, valid - 1:valid]
      failed[:, valid:] = failed[:, valid - 1:valid]
      rows.append({'state': state, 'action': action, 'failed': failed, 'onset': onset,
                   'length': np.full(n, valid, np.int16), 'episode': orig[sl].astype(np.int32),
                   'anchor_time': np.full(n, t, np.int16)})
    done = sum(len(r['state']) for r in rows)
    print(f'anchor t={t:2d}: {done:,} paths, {time.time() - t0:.0f}s', flush=True)
  return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--chunk', type=int, default=3300)
  ap.add_argument('--anchor-times', type=int, nargs='*', default=list(range(H)))
  ap.add_argument('--smoke', action='store_true', help='anchor times 0, 25, 49 and 200 episodes')
  ap.add_argument('--frozen-stay', action='store_true',
                  help='variant: recorded stationary anchors (t >= 4) stay absorbed; writes replay_P_frozen.npz')
  args = ap.parse_args(argv)
  name = 'replay_P_frozen.npz' if args.frozen_stay else 'replay_P.npz'
  out = (OUT if not args.smoke else OUT / '_smoke') / name
  if out.exists():
    print(f'{out} exists', flush=True)
    return 0
  times = [0, 25, 49] if args.smoke else args.anchor_times
  g = generate(times, args.chunk if not args.smoke else 200, frozen_stay=args.frozen_stay)
  if args.smoke:
    keep = np.arange(len(g['state'])) % 3300 < 200
    g = {k: v[keep] for k, v in g.items()}
  paths = len(g['state'])
  # ---- assemble: the branch paths + the query paths of replay_E (sources 1, 2, 3) ----
  e = load_npz(REPLAY_E)
  q = np.flatnonzero(e['audit_source'] != 0)
  goal_s = np.broadcast_to(GOAL, g['state'].shape).astype(np.float32)
  obs_p = np.concatenate([g['state'], goal_s], axis=2).astype(np.float32)
  meta = json.loads(str(e['meta']))
  meta.update({'arm': 'P_frozen' if args.frozen_stay else 'P', 'branch_paths': int(paths), 'query_paths': int(len(q)),
               'base_replay': 'replay_E.npz', 'base_replay_sha256': sha256(REPLAY_E),
               'anchor_rule': 'row 0 of every path (set_anchor_strata in the driver)',
               'gen_seed': GEN_SEED})
  onset_time = np.where(g['onset'].any(1), g['onset'].argmax(1), -1).astype(np.int16)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      out,
      obs=np.concatenate([obs_p, e['obs'][q]]).astype(np.float32),
      act=np.concatenate([g['action'], e['act'][q]]).astype(np.float32),
      lengths=np.concatenate([g['length'], e['lengths'][q]]).astype(np.int16),
      meta=np.asarray(json.dumps(meta, sort_keys=True)),
      audit_source=np.concatenate([np.zeros(paths, np.int8), e['audit_source'][q]]),
      audit_original_episode=np.concatenate([g['episode'], e['audit_original_episode'][q]]).astype(np.int32),
      audit_anchor_time=np.concatenate([g['anchor_time'], np.zeros(len(q), np.int16)]).astype(np.int16),
      audit_onset_time=np.concatenate([onset_time, e['audit_onset_time'][q]]).astype(np.int16),
      audit_failed_at_end=np.concatenate([g['failed'][np.arange(paths), g['length'].astype(int) - 1],
                                          e['audit_failed_at_end'][q]]))
  # ---- summary: the new law at the fork, and what the model does at recorded stationary rows ----
  xy = g['state'][:, :, :2]
  L = g['length'].astype(int)
  reach = np.array([np.any(np.linalg.norm(xy[k, :L[k]] - GOAL[:2], axis=1) < 0.5) for k in range(paths)])
  s0 = g['state'][:, 0]
  cell = np.floor(s0[:, :2]).astype(int)
  fork = (cell[:, 0] == 1) & (cell[:, 1] == 3)
  a0 = g['action'][:, 0]
  ang = np.degrees(np.arctan2(a0[:, 1], a0[:, 0]))
  sec = {'DOWN': (ang < -67.5) & (ang > -112.5), 'RIGHT': (ang > -22.5) & (ang < 22.5),
         'DR': (ang <= -22.5) & (ang >= -67.5)}
  frozen = np.all(s0[:, 2:] == np.tile(s0[:, :2], 3), axis=1) & (g['anchor_time'] >= 4)
  moved = np.linalg.norm(g['state'][:, 1, :2] - s0[:, :2], axis=1) > 0.05
  summary = {
      'branch_paths': int(paths), 'query_paths': int(len(q)), 'replay_sha256': sha256(out),
      'reach_0p5_fraction': float(reach.mean()),
      'absorbed_at_end': float(g['failed'][np.arange(paths), L - 1].mean()),
      'fork_anchors': int(fork.sum()),
      'fork_reach_by_sector': {k: {'n': int((fork & m).sum()), 'reach': float(reach[fork & m].mean())}
                               for k, m in sec.items()},
      'recorded_stationary_anchors': int(frozen.sum()),
      'model_moves_from_recorded_stationary': float(moved[frozen].mean()) if frozen.any() else None,
      'wall_seconds': None}
  summary['frozen_stay'] = bool(args.frozen_stay)
  write_json(out.parent / ('generation_frozen.json' if args.frozen_stay else 'generation.json'), summary)
  print(json.dumps(summary, indent=1), flush=True)
  return 0


if __name__ == '__main__':
  sys.exit(main())
