"""Training replay and full-state resumption of the round-1 CF runs (the pilot's own learner, streams and futures).

Two uses, both starting from a SAVED TrainingState (params, Adam states, target
params, sampling key -- crl.checkpoint stores the whole state) with the critic
and actor streams fast-forwarded to that checkpoint's position, so the batches
are the training's own sequence (up to GPU non-determinism):

  replay   Re-run the round-1 CF training of one seed from `init.pkl` (or a
           milestone) and save a checkpoint every --ckpt-every updates, to see
           WHEN the reset torques turn toward the detour and WHEN the detour
           walking degrades (mode `probe` measures both on every checkpoint).
  resume   Step-2 control: continue the round-1 CF agent from `final.pkl` with
           its full learner state (actor, critic, target, both Adam states) on
           the round-1 futures for another --updates updates -- the same data,
           continued -- then the standard evaluation; compared with CFold2
           (same futures, restarted critic and Adam) and CF1.
  probe    For every checkpoint of a replay: (a) route probe -- the mode from
           the 300 evaluation resets for 100 steps, share that turns north into
           the west column (max y >= 2.5); (b) walking probe -- continuation
           from the SAME detour-entrance handover states as diag_traj/cont.json
           (the round-1 CF policy's own prefix up to tau), reach / timeout;
           (c) the checkpoint's own critic at the 192 logged-shortcut reset
           rows: best teacher detour torque minus f(mode) and minus f(logged);
           (d) the mode's distance from the start agent's mode at those rows.
  report   Tables across the checkpoints.

  python scripts/diag_v6_training_replay.py replay --seed 0 --from init --to 30000 --ckpt-every 1000
  python scripts/diag_v6_training_replay.py probe --seed 0 --workers 18
  python scripts/diag_v6_training_replay.py resume --seed 0 --updates 30000
  python scripts/diag_v6_training_replay.py report
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP  # noqa: E402

REPLAY = MP.OUT / 'diag_replay'
RESUME = MP.R2 / 'resume'
G = MP.G
PROBE_STEPS = 100
NORTH_Y = 2.5
MAX_ENTRIES = 60          # walking probe: cap on the recorded CF detour entrances used as handover states


# ------------------------------------------------------------- learner
def build(seed, arm, branch_path, cfg_dir):
  import jax.numpy as jnp
  import optax
  from crl import losses as losses_mod
  cfg = MP.recipe_config(seed, cfg_dir)
  cfg.batch_size = MP.BATCH
  MP.fill_dims(cfg)
  nets = MP.make_nets(cfg)
  pol_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7)
  q_opt = optax.adam(cfg.learning_rate, eps=1e-7)
  gidx = np.asarray(cfg.goal_indices)

  def obs_to_goal(states):
    return states[:, jnp.asarray(gidx)]
  _, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, pol_opt, q_opt, separate_actor_batch=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  obs, act, lengths, _ = MP.load_dataset()
  futures = MP.RecordedFutures(anchors, obs, act) if arm == 'O' else MP.BranchFutures(anchors, branch_path)
  critic_stream = MP.CriticStream(anchors, futures, cfg.batch_size, cfg.discount, MP.CRITIC_STREAM_SEED0 + seed)
  actor_stream = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + seed)
  return cfg, nets, update_step, critic_stream, actor_stream


def fast_forward(critic_stream, actor_stream, n_batches):
  """Consume n_batches of both streams' RNGs exactly as train_arm did (CriticStream.draw: two uniform(B) draws;
  TrajectoryBuffer._draw_indices, variable-length path: integers(B), random(B), uniform(B, L))."""
  buf = actor_stream.buffer
  assert getattr(buf, '_use_lengths', False) and not getattr(buf, '_use_strata', False) and not getattr(buf, '_use_balanced', False) \
      and not getattr(buf, '_use_anchor_cut', False), 'fast-forward assumes the variable-length sampling path'
  B, L, ne, rng = critic_stream.B, buf._L, buf._num_eps, buf._rng
  t0 = time.time()
  for _ in range(n_batches):
    critic_stream.draw()
    rng.integers(0, ne, size=B); rng.random(B); rng.uniform(size=(B, L))
  return time.time() - t0


def selfcheck_fast_forward(seed=0, n=7):
  """The manual RNG consumption reproduces TrajectoryBuffer.sampled_indices: fast-forward n batches on one buffer,
  draw n batches on a twin, and compare the (n + 1)-th draw."""
  cfg = MP.recipe_config(seed, REPLAY / '_cfg'); cfg.batch_size = MP.BATCH; MP.fill_dims(cfg)
  a = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + seed); b = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + seed)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')

  class _F:                                            # a stand-in futures object with unit-length branches for CriticStream
    lengths = np.full(anchors.n, 2, np.int64)
  ca = MP.CriticStream(anchors, _F(), cfg.batch_size, cfg.discount, 1); cb = MP.CriticStream(anchors, _F(), cfg.batch_size, cfg.discount, 1)
  fast_forward(ca, a, n)
  for _ in range(n):
    b.buffer.sampled_indices(cfg.batch_size); cb.draw()
  x, y = a.buffer.sampled_indices(cfg.batch_size), b.buffer.sampled_indices(cfg.batch_size)
  ok = all(np.array_equal(p, q) for p, q in zip(x, y)) and all(np.array_equal(p, q) for p, q in zip(ca.draw(), cb.draw()))
  print(f'fast-forward self-check: {"OK" if ok else "MISMATCH"}', flush=True)
  return ok


def continue_training(src_ckpt, n_updates, out_dir, ckpt_every, seed, arm, branch_path, name_final='final', log_every=500):
  import jax
  from crl import checkpoint
  out_dir.mkdir(parents=True, exist_ok=True)
  cfg, nets, update_step, critic_stream, actor_stream = build(seed, arm, branch_path, out_dir / '_cfg')
  step0, state = checkpoint.load_checkpoint(src_ckpt)
  ff = fast_forward(critic_stream, actor_stream, int(step0))
  print(f'loaded {src_ckpt} @ step {step0}; streams fast-forwarded by {step0} batches in {ff:.0f} s', flush=True)
  h0 = {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)}
  checkpoint.save_named(str(out_dir), 'init', int(step0), state)

  def _multi(state, pair):
    state, metrics = jax.lax.scan(update_step, state, pair)
    return state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
  multi_update = jax.jit(_multi)
  assert n_updates % G == 0 and ckpt_every % G == 0
  n, t0, hist = 0, time.time(), []
  for it in range(n_updates // G):
    cbs = [critic_stream.sample()[0] for _ in range(G)]
    abs_ = [actor_stream.sample(cfg.batch_size) for _ in range(G)]
    state, metrics = multi_update(state, (MP._stack(cbs), MP._stack(abs_)))
    n += G
    if n % log_every == 0 or n == n_updates:
      m = {k: float(v) for k, v in metrics.items()}
      hist.append({'update': int(step0) + n, **m})
      print(f'[{arm} s{seed} upd {int(step0) + n:>6}] critic {m.get("critic_loss", 0):.4f} actor {m.get("actor_loss", 0):.4f} bc_nll {m.get("bc_nll", 0):.3f} '
            f'q_term {m.get("actor_q_term", 0):.3f} {n / (time.time() - t0):.1f} upd/s', flush=True)
    if n % ckpt_every == 0 and n < n_updates:
      checkpoint.save_named(str(out_dir), f'upd_{int(step0) + n}', int(step0) + n, state)
  checkpoint.save_named(str(out_dir), name_final, int(step0) + n, state)
  MP.write_json(out_dir / 'train_manifest.json', {
      'source_ckpt': str(src_ckpt), 'source_ckpt_sha256': MP.sha256(src_ckpt), 'source_step': int(step0), 'updates': n, 'final_step': int(step0) + n,
      'state_carried': ['policy_params', 'q_params', 'target_q_params', 'policy_optimizer_state', 'q_optimizer_state', 'key'],
      'streams': {'critic_seed': MP.CRITIC_STREAM_SEED0 + seed, 'actor_seed': MP.ACTOR_STREAM_SEED0 + seed, 'fast_forwarded_batches': int(step0)},
      'futures': ('recorded' if arm == 'O' else str(branch_path)), 'branch_sha256': (None if arm == 'O' else MP.sha256(branch_path)),
      'ckpt_every': ckpt_every, 'config': MP.config_dump(cfg), 'params_at_load': h0,
      'params_final': {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)},
      'device': str(jax.devices()[0]), 'wall_seconds': time.time() - t0, 'history': hist})
  print(f'{arm} s{seed}: {n} updates from step {step0} in {time.time() - t0:.0f} s -> {out_dir}', flush=True)


def mode_replay(args):
  src = MP.run_dir('CF', args.seed) / (f'{args.from_}.pkl')
  out = REPLAY / f'CF_s{args.seed}'
  if (out / 'final.pkl').exists() and not args.force:
    print(f'{out} exists', flush=True); return
  from crl import checkpoint
  step0 = checkpoint.load_checkpoint(src)[0]
  continue_training(src, args.to - int(step0), out, args.ckpt_every, args.seed, 'CF', MP.OUT / 'branches_cf.npz')


def mode_resume(args):
  src = MP.run_dir('CF', args.seed) / 'final.pkl'
  out = RESUME / 'CF' / f'seed_{args.seed}'
  if (out / 'final.pkl').exists() and not args.force:
    print(f'{out} exists', flush=True)
  else:
    continue_training(src, args.updates, out, args.ckpt_every, args.seed, 'CF', MP.OUT / 'branches_cf.npz')
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = MP.EVAL['seed'], MP.EVAL['n']
  print(MP.D.evaluate_ckpt(out / 'final.pkl', out, MP.EVAL['policy']), flush=True)


# --------------------------------------------------------------- probes
def _ckpt_list(seed, d=None):
  import pickle
  d = d or (REPLAY / f'CF_s{seed}')

  def step_of(p):
    with open(p, 'rb') as f:
      return int(pickle.load(f)['step'])
  cks = sorted(d.glob('upd_*.pkl'), key=lambda p: int(p.stem.split('_')[1]))
  return [(step_of(d / 'init.pkl'), d / 'init.pkl')] + [(int(p.stem.split('_')[1]), p) for p in cks] + [(step_of(d / 'final.pkl'), d / 'final.pkl')]


def mode_probe(args):
  import diag_v6_pilot_trajectories as DT
  import jax.numpy as jnp
  s = args.seed
  T0 = DT.Traj('start')
  Tcf = DT.Traj(f'CF_s{s}')
  C = MP.read_json(DT.OUT / 'cont.json')['results']
  hand = [(r['episode'], int(r['tau'])) for r in C if r['tag'] == f'cont|s{s}|enter_detour|CF_s{s}']     # cont.json: the CF detour TIMEOUT episodes' entrance
  # the broader set: every recorded CF episode that entered the west column (first step with y >= 2 at x < 2.5), capped at MAX_ENTRIES
  entries = []
  for k in range(int(len(Tcf.d['steps']))):
    xy = Tcf.obs(k)[:, :2]
    m = (xy[:, 1] >= 2.0) & (xy[:, 0] < 2.5)
    if m.any():
      entries.append((k, int(np.argmax(m))))
  rng = np.random.default_rng(DT.DIAG_SEED + 31 + s)
  if len(entries) > MAX_ENTRIES:
    entries = [entries[i] for i in sorted(rng.choice(len(entries), size=MAX_ENTRIES, replace=False))]
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  k0 = np.flatnonzero((anchors.t == 0) & (route[anchors.episode] == 'shortcut'))
  O0, A0 = anchors.obs31[k0].astype(np.float32), anchors.action[k0].astype(np.float32)
  teach = DT._teacher_reset_torques()
  det = np.stack([v for k, v in teach.items() if k.startswith('teacher_detour')])
  start_mode = DT._policy_bundle('start')['mode_batch'](O0)
  cf_final_mode = DT._policy_bundle(f'CF_s{s}')['mode_batch'](O0)
  rdir = Path(args.replay_dir) if args.replay_dir else (REPLAY / f'CF_s{s}')     # --replay-dir: e.g. the resumption run's milestones
  cks = _ckpt_list(s, rdir)
  if args.only_final:
    cks = [cks[0], cks[-1]]
  out_path = rdir / 'probe.json'
  done = MP.read_json(out_path)['checkpoints'] if out_path.exists() and not args.force else {}
  n_ep = int(len(T0.d['steps']))
  for step, ck in cks:
    if str(step) in done and not (args.entry_only and 'entry_probe' not in done[str(step)]):
      continue
    name = f'ckpt:{ck}'
    if args.entry_only and str(step) in done:
      # a probe.json written by the earlier script version: add the entrance probe to the existing record
      jobs = [(f'entry|{step}', k, Tcf.obs(k)[0], Tcf.hidden(k), f'CF_s{s}', tau, None, name, False, None) for k, tau in entries]
      t0 = time.time()
      ee = DT.run_branches(jobs, args.workers, seed0=DT.DIAG_SEED + 900_000 + step)
      done[str(step)]['entry_probe'] = {'n': len(ee), 'reach': float(np.mean([r['success'] for r in ee])), 'timeout': float(np.mean([r['timeout'] for r in ee])),
                                        'failure': float(np.mean([r['failure'] for r in ee])), 'steps_mean': float(np.mean([r['steps'] for r in ee]))}
      print(f'step {step:>6}: entry reach {done[str(step)]["entry_probe"]["reach"]:.3f} timeout {done[str(step)]["entry_probe"]["timeout"]:.3f} ({time.time() - t0:.0f} s)', flush=True)
      MP.write_json(out_path, {'seed': s, 'handover_states': len(hand), 'entry_states': len(entries), 'probe_steps': PROBE_STEPS, 'north_y': NORTH_Y, 'checkpoints': done})
      continue
    jobs = [(f'route|{step}', k, T0.obs(k)[0], T0.hidden(k), None, 0, None, name, False, PROBE_STEPS) for k in range(n_ep)]
    jobs += [(f'walk|{step}', k, Tcf.obs(k)[0], Tcf.hidden(k), f'CF_s{s}', tau, None, name, False, None) for k, tau in hand]
    jobs += [(f'entry|{step}', k, Tcf.obs(k)[0], Tcf.hidden(k), f'CF_s{s}', tau, None, name, False, None) for k, tau in entries]
    t0 = time.time()
    res = DT.run_branches(jobs, args.workers, seed0=DT.DIAG_SEED + 900_000 + step)
    rr = [r for r in res if r['tag'].startswith('route|')]; ww = [r for r in res if r['tag'].startswith('walk|')]; ee = [r for r in res if r['tag'].startswith('entry|')]

    def _blk(rows):
      return {'n': len(rows), 'reach': (float(np.mean([r['success'] for r in rows])) if rows else None), 'timeout': (float(np.mean([r['timeout'] for r in rows])) if rows else None),
              'failure': (float(np.mean([r['failure'] for r in rows])) if rows else None), 'steps_mean': (float(np.mean([r['steps'] for r in rows])) if rows else None)}
    b = DT._policy_bundle(name)
    mode = b['mode_batch'](O0)
    Fd = np.stack([b['f'](O0, np.repeat(det[j][None], len(O0), 0)) for j in range(len(det))])      # [4, N]
    F_det, jbest = Fd.max(0), Fd.argmax(0)
    f_mode, f_logged = b['f'](O0, mode), b['f'](O0, A0)
    d_mode, d_det = mode - start_mode, det[jbest] - start_mode
    cos = (d_mode * d_det).sum(1) / (np.linalg.norm(d_mode, axis=1) * np.linalg.norm(d_det, axis=1) + 1e-8)
    rec = {'step': step, 'ckpt': str(ck),
           'route_probe': {'n': len(rr), 'turned_north': float(np.mean([r['max_y'] >= NORTH_Y for r in rr])), 'success_within_probe': float(np.mean([r['success'] for r in rr])),
                           'failure_within_probe': float(np.mean([r['failure'] for r in rr])), 'max_y_mean': float(np.mean([r['max_y'] for r in rr]))},
           'walking_probe': _blk(ww), 'entry_probe': _blk(ee),
           'critic_at_reset_rows': {'best_detour_minus_f_mode': float((F_det - f_mode).mean()), 'P_best_detour_above_mode': float((F_det > f_mode).mean()),
                                    'best_detour_minus_f_logged': float((F_det - f_logged).mean()), 'P_best_detour_above_logged': float((F_det > f_logged).mean())},
           'mode_at_reset_rows': {'dist_to_start_mode': float(np.linalg.norm(mode - start_mode, axis=1).mean()),
                                  'dist_to_cf_final_mode': float(np.linalg.norm(mode - cf_final_mode, axis=1).mean()),
                                  'cos_to_best_detour_direction': float(cos.mean())},
           'wall_seconds': time.time() - t0}
    done[str(step)] = rec
    print(f'step {step:>6}: north {rec["route_probe"]["turned_north"]:.3f}  walk reach {rec["walking_probe"]["reach"]} timeout {rec["walking_probe"]["timeout"]}  '
          f'entry reach {rec["entry_probe"]["reach"]} timeout {rec["entry_probe"]["timeout"]}  '
          f'critic best_detour - f_mode {rec["critic_at_reset_rows"]["best_detour_minus_f_mode"]:+.2f} (P {rec["critic_at_reset_rows"]["P_best_detour_above_mode"]:.2f})  '
          f'|mode - start| {rec["mode_at_reset_rows"]["dist_to_start_mode"]:.3f}  ({rec["wall_seconds"]:.0f} s)', flush=True)
    MP.write_json(out_path, {'seed': s, 'handover_states': len(hand), 'entry_states': len(entries), 'probe_steps': PROBE_STEPS, 'north_y': NORTH_Y, 'checkpoints': done})


def mode_calib(args):
  """The same probes on the ORIGINAL round-1 CF final checkpoint (and the start agent): calibrates the replay
  (GPU non-determinism makes it a statistical replicate) and the 100-step route probe against the 300-episode evaluation."""
  import diag_v6_pilot_trajectories as DT
  s = args.seed
  T0, Tcf = DT.Traj('start'), DT.Traj(f'CF_s{s}')
  C = MP.read_json(DT.OUT / 'cont.json')['results']
  hand = [(r['episode'], int(r['tau'])) for r in C if r['tag'] == f'cont|s{s}|enter_detour|CF_s{s}']
  entries = []
  for k in range(int(len(Tcf.d['steps']))):
    xy = Tcf.obs(k)[:, :2]
    m = (xy[:, 1] >= 2.0) & (xy[:, 0] < 2.5)
    if m.any():
      entries.append((k, int(np.argmax(m))))
  rng = np.random.default_rng(DT.DIAG_SEED + 31 + s)
  if len(entries) > MAX_ENTRIES:
    entries = [entries[i] for i in sorted(rng.choice(len(entries), size=MAX_ENTRIES, replace=False))]
  n_ep = int(len(T0.d['steps']))
  out = {}
  for label, name in (('orig_CF_final', f'CF_s{s}'), ('start', 'start'), ('resume_final', f'ckpt:{RESUME / "CF" / f"seed_{s}" / "final.pkl"}')):
    if name.startswith('ckpt:') and not Path(name[5:]).exists():
      continue
    jobs = [(f'route|{label}', k, T0.obs(k)[0], T0.hidden(k), None, 0, None, name, False, PROBE_STEPS) for k in range(n_ep)]
    jobs += [(f'walk|{label}', k, Tcf.obs(k)[0], Tcf.hidden(k), f'CF_s{s}', tau, None, name, False, None) for k, tau in hand]
    jobs += [(f'entry|{label}', k, Tcf.obs(k)[0], Tcf.hidden(k), f'CF_s{s}', tau, None, name, False, None) for k, tau in entries]
    res = DT.run_branches(jobs, args.workers, seed0=DT.DIAG_SEED + 950_000)
    blk = lambda rows: {'n': len(rows), 'reach': float(np.mean([r['success'] for r in rows])) if rows else None, 'timeout': float(np.mean([r['timeout'] for r in rows])) if rows else None}
    rr = [r for r in res if r['tag'].startswith('route|')]
    out[label] = {'route_probe': {'n': len(rr), 'turned_north': float(np.mean([r['max_y'] >= NORTH_Y for r in rr])), 'failure_within_probe': float(np.mean([r['failure'] for r in rr]))},
                  'walking_probe': blk([r for r in res if r['tag'].startswith('walk|')]), 'entry_probe': blk([r for r in res if r['tag'].startswith('entry|')])}
    print(label, json.dumps(out[label]), flush=True)
  MP.write_json(REPLAY / f'CF_s{s}' / 'calibration.json', {'seed': s, 'handover_states': len(hand), 'entry_states': len(entries), 'probes': out})


def mode_report(args):
  L = ['# Training replay of the round-1 CF runs (full saved state, own batch sequence) and the full-state resumption control', '',
       f'Route probe: the mode from the 300 evaluation resets for {PROBE_STEPS} steps, share with max y >= {NORTH_Y} (turned north).  '
       'Walking probe: continuation from the same detour-entrance handover states as diag_traj/cont.json (round-1 CF prefix up to tau).  '
       'Critic: the checkpoint\'s own critic at the 192 logged-shortcut reset rows (task goal).', '']
  for s in MP.SEEDS:
    p = REPLAY / f'CF_s{s}' / 'probe.json'
    if not p.exists():
      continue
    P = MP.read_json(p)
    L += [f'## Replay CF seed {s} ({P["handover_states"]} handover states)', '',
          '| update | turned north | success in probe | walk reach (timeout entrances) | walk timeout | entry reach (all entrances) | entry timeout | best detour - f(mode) | P | best detour - f(logged) | \\|mode - start\\| | \\|mode - CF final\\| |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fm = lambda v: ('-' if v is None else format(v, '.3f'))
    for step in sorted(P['checkpoints'], key=int):
      r = P['checkpoints'][step]
      w, c, m, e = r['walking_probe'], r['critic_at_reset_rows'], r['mode_at_reset_rows'], r.get('entry_probe', {'reach': None, 'timeout': None})
      L.append(f'| {step} | {r["route_probe"]["turned_north"]:.3f} | {r["route_probe"]["success_within_probe"]:.3f} | {fm(w["reach"])} | {fm(w["timeout"])} | {fm(e["reach"])} | {fm(e["timeout"])} | '
               f'{c["best_detour_minus_f_mode"]:+.2f} | {c["P_best_detour_above_mode"]:.2f} | {c["best_detour_minus_f_logged"]:+.2f} | '
               f'{m["dist_to_start_mode"]:.3f} | {m["dist_to_cf_final_mode"]:.3f} |')
    L.append('')
  # the resumption control
  E1 = {n: MP._episodes(p) for n, p in MP.eval_paths().items() if p.exists()}
  E2 = {n: MP._episodes(p) for n, p in MP.r2_eval_paths().items() if p.exists()}
  ER = {}
  for s in MP.SEEDS:
    p = RESUME / 'CF' / f'seed_{s}' / f'eval_mean_s{MP.EVAL["seed"]}.json'
    if p.exists():
      ER[s] = MP._episodes(p)
  if ER:
    L += ['## Full-state resumption of the round-1 CF agents (round-1 futures, +30,000 updates, actor / critic / target / Adam carried)', '',
          '| policy | success | detour | death | timeout | success hazard | mean steps |', '|---|---:|---:|---:|---:|---:|---:|']
    rows = [(f'round1 CF/seed_{s}', E1.get(f'CF/seed_{s}')) for s in MP.SEEDS] + [(f'resume CF/seed_{s}', ER.get(s)) for s in MP.SEEDS] + \
           [(f'round2 CFold/seed_{s}', E2.get(f'CFold/seed_{s}')) for s in MP.SEEDS] + [(f'round2 O/seed_{s}', E2.get(f'O/seed_{s}')) for s in MP.SEEDS]
    for n, e in rows:
      if e is not None:
        h = MP._headline(e)
        L.append(f'| {n} | {h["success"]:.3f} | {h["detour"]:.3f} | {h["death"]:.3f} | {h["timeout"]:.3f} | {h["success_hazard"]:.3f} | {h["mean_steps"]:.0f} |')
    if len(ER) == len(MP.SEEDS):
      by_cf1 = {s: E1[f'CF/seed_{s}'] for s in MP.SEEDS}; by_cfold = {s: E2[f'CFold/seed_{s}'] for s in MP.SEEDS}; by_o2 = {s: E2[f'O/seed_{s}'] for s in MP.SEEDS}
      res = {}
      L += ['', '### Paired differences (per seed; mean, seed s.e., episode-bootstrap s.e.)', '']
      for key in ('success', 'detour', 'failure', 'timeout'):
        L += [f'#### {key}', '', '| comparison | per seed | mean | seed s.e. | boot s.e. | same direction |', '|---|---|---:|---:|---:|---|']
        for cname, a, b in (('resume - CF1 (continued vs its start)', ER, by_cf1), ('resume - CFold2 (carried vs restarted learner, same futures)', ER, by_cfold), ('resume - O2', ER, by_o2)):
          r = MP.paired_block(a, b, key=key); res[f'{cname}:{key}'] = r
          per = ' / '.join(f'{v["mean"]:+.3f}' for v in r['per_seed'].values())
          L.append(f'| {cname} | {per} | {r["mean"]:+.3f} | {r["seed_se"]:.3f} | {r["episode_bootstrap_se_of_mean"]:.3f} | {r["seeds_same_direction"]} |')
        L.append('')
      MP.write_json(RESUME / 'results.json', {'headlines': {f'resume CF/seed_{s}': MP._headline(e) for s, e in ER.items()}, 'paired': res})
  (REPLAY / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('mode', choices=('replay', 'resume', 'probe', 'calib', 'report', 'selfcheck'))
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--from', dest='from_', default='init', help='replay: init | 10000 | 20000')
  ap.add_argument('--to', type=int, default=MP.UPDATES)
  ap.add_argument('--updates', type=int, default=MP.UPDATES, help='resume: additional updates')
  ap.add_argument('--ckpt-every', type=int, default=1000)
  ap.add_argument('--workers', type=int, default=18)
  ap.add_argument('--only-final', action='store_true')
  ap.add_argument('--replay-dir', default=None, help='probe: directory with init.pkl / upd_*.pkl / final.pkl (default diag_replay/CF_s<seed>)')
  ap.add_argument('--entry-only', action='store_true', help='probe: add the entrance probe to records that lack it')
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args()
  if args.mode == 'selfcheck':
    return 0 if selfcheck_fast_forward() else 1
  {'replay': mode_replay, 'resume': mode_resume, 'probe': mode_probe, 'calib': mode_calib, 'report': mode_report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
