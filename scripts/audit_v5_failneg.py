r"""Critic-semantic audit of the V5 failure-negative runs.

Read-only. No training, no dataset regeneration, no checkpoint is written.

WHY THIS EXISTS. A deployment eval cannot tell "the bank did nothing to the
critic" apart from "the critic moved and the actor did not follow". On the f4
benchmark the deployment cells were identical across 27 runs and the only
quantity that responded was f(., ., g_bank) -- invisible to a deployment eval.
So the same three questions get asked here, in V5's own terms:

  1  does the critic score the FAILURE GOALS lower?
  2  does it separate a failure state from a NORMAL state at the same place --
     and does it leave the expert's safe HOLD alone while doing so?
  3  does the actor's behaviour at the mouth change?

WHAT IS NOT COPIED FROM THE f4 AUDIT. f4's velocity-response curve and its
21x21 two-dimensional action grid are both artifacts of a 2-D point mass; the
ant has an 8-D action space and a different failure mechanism. The behaviour
probe here scores the ACTOR'S OWN action against the zero action (the expert's
hold) at real mouth states, which is the question this benchmark actually asks.

PAIRING IS REAL BY CONSTRUCTION. The failure states come from a HELD-OUT
failure collection (its own seed, its own episodes, never in the bank -- the
script asserts the intersection is empty); the normal states come from the
frozen training dataset. Every pair is therefore two different episodes from
two different collection runs, with no synthesis anywhere. The one synthetic
diagnostic -- a real crossing state with its velocity columns zeroed, i.e. "the
same ant, stopped here" -- is reported in its own section and labelled as
synthetic, because a synthesised goal is not evidence about real states.

Runs are DISCOVERED, not hard-coded: every <glob>/final.pkl is audited and its
alpha and goal representation are read from that run's own
arm_provenance.json, so a sweep split across machines audits correctly once
the directories are collected in one place.

Usage:
  python scripts/audit_v5_failneg.py --goal-rep xyv
  python scripts/audit_v5_failneg.py --goal-rep xyv --runs-glob 'v5fn_xyv_*_s0'
"""
import argparse
import glob as globmod
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.20')

import jax                                                      # noqa: E402
import jax.numpy as jnp                                         # noqa: E402

from crl import checkpoint as ckpt_mod                          # noqa: E402
from crl import networks as networks_mod                        # noqa: E402
from crl import rockfall_clock_v5 as V5                         # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X, HAZARD_HALF_Y    # noqa: E402
import run_v5_failneg as L                                      # noqa: E402

OUT_DIR = 'artifacts/v5_failneg/audit'
HELDOUT_DIR = 'artifacts/v5_failneg/failures_heldout'
HELDOUT_PATTERN = 'v5_failures_heldout_*_s*.npz'
BANK_DEFAULT = L.BANK_DEFAULT
STATE_DIM = 29
STOP_V = 0.15          #: planar speed below which a step counts as standing


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  return {'n': int(v.size), 'mean': float(v.mean()),
          'median': float(np.median(v)), 'std': float(v.std()),
          'p10': float(np.percentile(v, 10)),
          'p90': float(np.percentile(v, 90))}


def in_band(xy):
  return ((np.abs(xy[..., 1]) < HAZARD_HALF_Y)
          & (xy[..., 0] >= HAZARD_X[0]) & (xy[..., 0] <= HAZARD_X[1]))


def at_mouth(xy):
  return ((np.abs(xy[..., 1]) < HAZARD_HALF_Y)
          & (xy[..., 0] >= V5.MOUTH_X) & (xy[..., 0] < HAZARD_X[0]))


def git(*a):
  try:
    return subprocess.check_output(
        ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
  except Exception:                            # pylint: disable=broad-except
    return ''


def discover_runs(pattern, include_smoke=False):
  runs = []
  for d in sorted(globmod.glob(pattern)):
    ckpt = os.path.join(d, 'final.pkl')
    if not os.path.exists(ckpt):
      continue
    prov_path = os.path.join(d, 'arm_provenance.json')
    prov = json.load(open(prov_path)) if os.path.exists(prov_path) else {}
    if prov.get('smoke', False) and not include_smoke:
      continue
    runs.append({'run_dir': d, 'checkpoint': ckpt,
                 'alpha': float(prov.get('alpha', 0.0)),
                 'goal_rep': prov.get('goal_rep'),
                 'bank': prov.get('bank'),
                 'dataset_content_sha256': prov.get('dataset_content_sha256'),
                 'bank_content_sha256': prov.get('bank_content_sha256'),
                 'seed': prov.get('seed'),
                 'label': os.path.basename(d)})
  return sorted(runs, key=lambda r: (r['alpha'], r['label']))


def load_heldout(hdir, pattern):
  """Held-out failure states + rename-proof (seed, arm, episode) identity."""
  files = sorted(f for f in globmod.glob(os.path.join(hdir, pattern))
                 if not f.endswith('_sidecar.npz'))
  if not files:
    raise SystemExit(
        'no held-out failure files matched %s in %s. Collect them with a seed '
        'DIFFERENT from the bank\'s:\n'
        '  python scripts/collect_v5_failure_episodes.py --all --episodes 200 '
        '--seed 909 \\\n'
        '      --out-dir %s --name v5_failures_heldout'
        % (pattern, hdir, hdir))
  states, ident, arms = [], [], []
  for f in files:
    with np.load(f, allow_pickle=False) as d:
      obs, lengths = d['obs'], np.asarray(d['lengths'], np.int64)
    with np.load(f.replace('.npz', '_sidecar.npz'), allow_pickle=False) as s:
      ep = np.asarray(s['episode_id'], np.int64)
      dr = np.asarray(s['death_row'], np.int64)
      arm = np.asarray(s['source_arm'])
      collection_seed = int(s['collection_seed'])
    assert np.array_equal(dr, lengths[:len(dr)] - 1)
    states.append(obs[ep, dr, :STATE_DIM].astype(np.float32))
    ident.extend((collection_seed, str(a), int(e))
                 for a, e in zip(arm.tolist(), ep.tolist()))
    arms.extend(str(a) for a in arm)
  return np.concatenate(states, 0), ident, np.asarray(arms), files


def make_scorer(ckpt_path, cfg):
  """f(s, a, g) for the twin-min contrastive critic of one checkpoint."""
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  qp, pp = st.q_params, st.policy_params
  trained = int(pp['mlp/~/linear_0']['w'].shape[0])
  want = int(cfg.obs_dim) + int(cfg.goal_dim)
  assert trained == want, (
      'checkpoint expects %d-dim observations but goal rep %r produces %d'
      % (trained, L.REP, want))

  # Keep parameters explicit in the jitted function.  Closing over a Haiku
  # parameter tree here is unsafe when this audit changes batch shape (the
  # anchor probes and replay-positive probes normally do): on current JAX the
  # nested closure can reuse a trace whose positional arguments no longer
  # match, making the 29-dim state appear at the 8-dim goal encoder.  Training
  # is unaffected; this is strictly an audit-side apply wrapper.
  @jax.jit
  def scores_apply(params, s, a, g):
    q = nets.q_network.apply(params, jnp.concatenate([s, g], axis=1), a)
    if q.ndim == 3:
      q = jnp.min(q, axis=-1)
    return jnp.diag(q)

  @jax.jit
  def actor_mode(o):
    p = nets.policy_network.apply(pp, o)
    return jnp.tanh(p.loc), p.scale

  def scores_np(s, a, g, chunk=2048):
    s, a, g = np.asarray(s), np.asarray(a), np.asarray(g)
    assert s.ndim == a.ndim == g.ndim == 2
    assert s.shape[0] == a.shape[0] == g.shape[0]
    assert s.shape[1] == int(cfg.obs_dim), s.shape
    assert a.shape[1] == int(cfg.action_dim), a.shape
    assert g.shape[1] == int(cfg.goal_dim), g.shape
    out = []
    for i in range(0, len(s), chunk):
      out.append(np.asarray(scores_apply(
          qp, jnp.asarray(s[i:i + chunk]), jnp.asarray(a[i:i + chunk]),
          jnp.asarray(g[i:i + chunk]))))
    return np.concatenate(out) if out else np.zeros(0)

  return scores_np, actor_mode, int(step)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--goal-rep', choices=sorted(L.GOAL_REPS), default='xyv')
  ap.add_argument('--runs-glob', default=None)
  ap.add_argument('--bank', default=BANK_DEFAULT)
  ap.add_argument('--heldout-dir', default=HELDOUT_DIR)
  ap.add_argument('--heldout-pattern', default=HELDOUT_PATTERN)
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n-anchors', type=int, default=8000)
  ap.add_argument('--include-smoke', action='store_true',
                  help='include runs whose provenance says smoke=true; off by '
                       'default so production audits cannot mix in smoke runs')
  ap.add_argument('--allow-provenance-mismatch', action='store_true',
                  help='audit legacy checkpoints despite a dataset/bank hash '
                       'mismatch; production audits should never use this')
  args = ap.parse_args()
  L.select_rep(args.goal_rep)
  gi = list(L.GOAL_REPS[args.goal_rep]['indices'])
  runs_glob = args.runs_glob or ('v5fn_%s_*_s*' % args.goal_rep)

  print('=' * 100)
  print('V5 FAILURE-NEGATIVE -- CRITIC SEMANTIC AUDIT   (goal rep %s)'
        % args.goal_rep)
  print('=' * 100)
  out = {'analysis_script': 'scripts/audit_v5_failneg.py',
         'goal_rep': args.goal_rep, 'goal_indices': gi,
         'env': L.ENV, 'dataset': L.DATASET, 'runs_glob': runs_glob,
         'code_commit': git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
         'head_at_runtime': git('rev-parse', 'HEAD'),
         'audited_files_dirty': bool(git('status', '--porcelain', '--',
                                         'crl', 'scripts'))}
  runs = discover_runs(runs_glob, include_smoke=args.include_smoke)
  if not runs:
    raise SystemExit('no runs matched %r (need <dir>/final.pkl)' % runs_glob)
  print('  runs: %s' % ', '.join('%s (alpha %g)' % (r['label'], r['alpha'])
                                 for r in runs))
  bad_rep = [r['label'] for r in runs
             if r['goal_rep'] not in (None, args.goal_rep)]
  if bad_rep:
    raise SystemExit('runs trained under a different goal representation are '
                     'in the glob: %s' % bad_rep)

  # ------------------------------------------------------------- the data
  with np.load(L.DATASET, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    lengths = np.asarray(d['lengths'], np.int64)
  dataset_sha = L.content_sha(L.DATASET)
  n_eps, Lrow, W = obs.shape
  assert W == STATE_DIM + len(gi), (W, STATE_DIM + len(gi))
  #: only rows that are inside a valid episode; the padded tail is not data.
  valid = np.arange(Lrow - 1)[None, :] < (lengths[:, None] - 1)
  s_t = obs[:, :-1, :STATE_DIM].reshape(-1, STATE_DIM)[valid.ravel()]
  s_n = obs[:, 1:, :STATE_DIM].reshape(-1, STATE_DIM)[valid.ravel()]
  a_t = act[:, :-1, :].reshape(-1, 8)[valid.ravel()]
  ep_of = np.repeat(np.arange(n_eps), Lrow - 1)[valid.ravel()]
  task_goal = obs[0, 0, STATE_DIM:].astype(np.float32).copy()
  print('  training transitions %s | task goal %s'
        % (format(len(s_t), ','), np.round(task_goal, 3).tolist()))

  band_sel = in_band(s_n[:, :2])
  hold_sel = at_mouth(s_n[:, :2]) & (
      np.linalg.norm(s_n[:, 15:17], axis=1) < STOP_V)
  band_states = s_n[band_sel]
  hold_states = s_n[hold_sel]
  print('  real band crossings %s | real mouth holds %s'
        % (format(int(band_sel.sum()), ','), format(int(hold_sel.sum()), ',')))
  assert len(band_states) and len(hold_states)

  # ----------------------------------------------------- bank and held-out
  with np.load(args.bank, allow_pickle=False) as b:
    bank_states = np.asarray(b['goals'], np.float32)
    bank_arm = np.asarray(b['source_arm'])
    bank_ident = set(zip(b['collection_seed'].astype(np.int64).tolist(),
                         b['source_arm'].tolist(),
                         b['episode_id'].astype(np.int64).tolist()))
    bank_meta = json.loads(str(b['meta']))
  bank_sha = L.content_sha(args.bank)
  provenance_errors = []
  for r in runs:
    if r['dataset_content_sha256'] != dataset_sha:
      provenance_errors.append(
          '%s dataset %r != %s' % (r['label'],
                                    r['dataset_content_sha256'], dataset_sha))
    if r['alpha'] > 0 and r['bank_content_sha256'] != bank_sha:
      provenance_errors.append(
          '%s bank %r != %s' % (r['label'], r['bank_content_sha256'], bank_sha))
  if provenance_errors and not args.allow_provenance_mismatch:
    raise SystemExit(
        'checkpoint provenance does not match the audit artifacts:\n  '
        + '\n  '.join(provenance_errors)
        + '\nRefusing an unpaired alpha audit. Use --allow-provenance-mismatch '
          'only for a labelled legacy diagnostic.')
  if provenance_errors:
    print('  WARNING: explicitly allowing provenance mismatch:\n    '
          + '\n    '.join(provenance_errors))
  out['artifact_provenance'] = {
      'dataset_content_sha256': dataset_sha,
      'bank_content_sha256': bank_sha,
      'mismatches': provenance_errors,
      'mismatch_override_used': bool(provenance_errors),
  }
  held, held_ident, held_arm, held_files = load_heldout(args.heldout_dir,
                                                        args.heldout_pattern)
  overlap = bank_ident & set(held_ident)
  assert not overlap, (
      'the audit failure states are NOT isolated from the bank: %d shared '
      '(collection seed, source arm, episode) tuples' % len(overlap))
  print('  bank %s  n=%d  %s' % (args.bank, len(bank_states),
                                 bank_meta.get('compose')))
  print('  held-out failures %d from %s  -- (seed, arm, episode) intersection with '
        'the bank is EMPTY' % (len(held), [os.path.basename(f)
                                           for f in held_files]))
  out['isolation'] = {
      'bank_path': args.bank,
      'bank_content_sha256': bank_sha,
      'dataset_content_sha256': dataset_sha,
      'bank_n': int(len(bank_states)),
      'bank_compose': bank_meta.get('compose'),
      'heldout_files': [os.path.basename(f) for f in held_files],
      'heldout_n': int(len(held)),
      'shared_seed_arm_episode_tuples': 0,
      'note': 'the bank is built from base-seed-808 arm streams and the audit '
              'failures from disjoint base-seed-909 arm streams; the eval side '
              'is never used to '
              'select bank entries.'}

  rng = np.random.default_rng(args.seed)

  # anchors: real (state, action) pairs from the training data, taken in the
  # approach zone and the band -- where the decision actually happens.
  anchor_sel = np.flatnonzero(at_mouth(s_t[:, :2]) | in_band(s_t[:, :2]))
  if len(anchor_sel) > args.n_anchors:
    anchor_sel = rng.choice(anchor_sel, args.n_anchors, replace=False)
  S, A = s_t[anchor_sel], a_t[anchor_sel]
  print('  anchors (mouth + band states) %s' % format(len(S), ','))
  out['n_anchors'] = int(len(S))

  # XY-nearest REAL band crossing for every held-out death state. Different
  # episodes by construction: the crossings are training episodes, the deaths
  # are the held-out collection.
  def nearest_by_xy(anchors, pool, chunk=256):
    idx = np.empty(len(anchors), np.int64)
    gap = np.empty(len(anchors), np.float64)
    for i in range(0, len(anchors), chunk):
      blk = anchors[i:i + chunk, :2]
      d2 = ((blk[:, None, 0] - pool[None, :, 0]) ** 2
            + (blk[:, None, 1] - pool[None, :, 1]) ** 2)
      j = np.argmin(d2, axis=1)
      idx[i:i + chunk] = j
      gap[i:i + chunk] = np.sqrt(d2[np.arange(len(blk)), j])
    return idx, gap

  nn_idx, nn_gap = nearest_by_xy(held, band_states)
  print('  real pairing: XY gap between a held-out death and its nearest '
        'training crossing, median %.4f maze units' % np.median(nn_gap))

  def as_goal(states):
    return np.ascontiguousarray(states[:, gi], np.float32)

  # Draw every stochastic probe ONCE, then reuse the same indices for every
  # alpha/checkpoint.  Otherwise alpha would be confounded with which deaths,
  # crossings, holds, and bank rows happened to be sampled for that run.
  probe_pick = rng.integers(0, len(bank_states), len(S))
  probe_dead = rng.integers(0, len(held), len(S))
  probe_hold = rng.integers(0, len(hold_states), len(S))
  probe_random = rng.integers(0, len(s_n), len(S))
  probe_stopped = rng.integers(0, len(band_states), len(S))
  mouth_sel = np.flatnonzero(at_mouth(s_t[:, :2]))
  if len(mouth_sel) > 2048:
    mouth_sel = rng.choice(mouth_sel, 2048, replace=False)

  results = {}
  for r in runs:
    cfg = L.build_cfg('base' if r['alpha'] == 0 else 'fail', '', steps=1,
                      seed=args.seed,
                      alpha=None if r['alpha'] == 0 else r['alpha'])
    #: the goal rep decides the widths; the env is not instantiated (this is a
    #: read-only audit) so fill the dims from the registry directly.
    cfg.obs_dim, cfg.goal_dim, cfg.action_dim = STATE_DIM, len(gi), 8
    # build_offline_buffer below also needs the representation selector.  The
    # normal trainer obtains this from make_env(); this read-only audit does
    # not instantiate an env, so leaving the build_cfg default (None) would
    # make replay relabel goals with the full 29-dim state and return a
    # 58-column observation to an XYV checkpoint.
    cfg.goal_indices = tuple(gi)
    score, actor_mode, step = make_scorer(r['checkpoint'], cfg)
    res = {'run_dir': r['run_dir'], 'checkpoint': r['checkpoint'],
           'step': step, 'alpha': r['alpha'], 'seed': r['seed'],
           'bank': r['bank']}

    # --- 1 BANK RESPONSE ------------------------------------------------
    pick = probe_pick
    g_bank = as_goal(bank_states[pick])
    f_bank = score(S, A, g_bank)
    res['1_bank_response'] = {'f_bank': dist(f_bank)}
    for c in sorted(set(bank_arm.tolist())):
      m = bank_arm[pick] == c
      if m.any():
        res['1_bank_response']['f_bank_' + c] = dist(f_bank[m])

    # --- 2 REAL PAIRING (PRIMARY): held-out death vs training crossing ---
    k = probe_dead
    g_dead = as_goal(held[k])
    g_alive = as_goal(band_states[nn_idx[k]])
    f_dead = score(S, A, g_dead)
    f_alive = score(S, A, g_alive)
    margin = f_alive - f_dead
    res['2_real_pairing'] = {
        'source': 'held-out failure episodes (base seed 909) vs the frozen training '
                  "dataset's own band crossings; different episodes, nothing "
                  'synthesised',
        'margin_alive_minus_dead': dist(margin),
        'frac_positive': float((margin > 0).mean()),
        'f_alive': dist(f_alive), 'f_dead': dist(f_dead),
        'pair_xy_gap': dist(nn_gap[k]),
        'by_source_arm': {c: dist(margin[held_arm[k] == c])
                          for c in sorted(set(held_arm.tolist()))}}

    # --- 3 THE HOLD MUST SURVIVE ---------------------------------------
    kh = probe_hold
    g_hold = as_goal(hold_states[kh])
    f_hold = score(S, A, g_hold)
    res['3_hold_preserved'] = {
        'question': 'the benchmark rewards standing still at the mouth. If the '
                    'bank has taught the critic that "slow" is death, f(hold) '
                    'drops with alpha too and the failure negatives have '
                    'damaged the very behaviour they should encourage.',
        'f_hold': dist(f_hold),
        'margin_hold_minus_dead': dist(f_hold - f_dead),
        'frac_hold_above_dead': float((f_hold > f_dead).mean()),
        'margin_hold_minus_alive': dist(f_hold - f_alive)}

    # --- 4 GOAL FAMILIES, centred ---------------------------------------
    g_rand = as_goal(s_n[probe_random])
    fam = {'factual_future_state': score(S, A, as_goal(s_n[anchor_sel])),
           'ordinary_random_state': score(S, A, g_rand),
           'real_band_crossing': f_alive,
           'real_mouth_hold': f_hold,
           'heldout_failure_state': f_dead,
           'failure_bank': f_bank,
           'task_goal': score(S, A, np.tile(task_goal, (len(S), 1)))}
    mu = float(np.concatenate(list(fam.values())).mean())
    res['4_goal_families'] = {k2: {'raw': dist(v), 'centred': dist(v - mu)}
                              for k2, v in fam.items()}
    res['4_overall_mean_logit'] = mu

    # --- 5 SYNTHETIC DIAGNOSTIC (labelled; not evidence about real states) -
    moving = band_states[probe_stopped].copy()
    stopped = moving.copy()
    stopped[:, 15:21] = 0.0            # the same ant, at rest, same XY
    f_moving = score(S, A, as_goal(moving))
    f_stop = score(S, A, as_goal(stopped))
    res['5_synthetic_stopped_in_band'] = {
        'SYNTHETIC': True,
        'construction': 'a REAL training band-crossing state with its six '
                        'torso velocity columns set to zero: the same ant, at '
                        'the same point, stopped. Under goal rep xy this is '
                        'identical to the unmodified state and the section is '
        'vacuous by design.',
        'f_moving_in_band': dist(f_moving),
        'f_stopped_in_band': dist(f_stop),
        'margin_moving_minus_stopped': dist(f_moving - f_stop)}

    # --- 6 BEHAVIOUR AT THE MOUTH: actor action vs the expert hold -------
    Sm = s_t[mouth_sel]
    obs_m = np.concatenate([Sm, np.tile(task_goal, (len(Sm), 1))], axis=1)
    a_mode, a_scale = actor_mode(jnp.asarray(obs_m))
    a_mode = np.asarray(a_mode)
    g_task_m = np.tile(task_goal, (len(Sm), 1)).astype(np.float32)
    q_mode = score(Sm, a_mode, g_task_m)
    q_zero = score(Sm, np.zeros_like(a_mode), g_task_m)
    res['6_mouth_behaviour'] = {
        'question': 'at a real mouth state, does the critic rank the actor own '
                    'action above the zero action (the expert hold), and how '
                    'big is the commanded action?',
        'n_mouth_states': int(len(Sm)),
        'q_actor_mode': dist(q_mode), 'q_zero_action': dist(q_zero),
        'margin_mode_minus_zero': dist(q_mode - q_zero),
        'frac_prefers_moving': float((q_mode > q_zero).mean()),
        'actor_action_norm': dist(np.linalg.norm(a_mode, axis=1)),
        'actor_sigma_mean': float(np.asarray(a_scale).mean()),
        'note': 'this is the critic ranking, not the rollout. The deployment '
                'eval (scripts/eval_rockfall_clock_v5_baseline.py) measures '
                'what the policy actually does.'}

    # --- 7 POSITIVES: does the relabeler ever hand over a failure goal? --
    from crl.offline_audit import build_offline_buffer
    buf, _ = build_offline_buffer(L.DATASET, cfg)
    sigma = s_n[:, gi].std(0) + 1e-6
    bank_g = bank_states[:, gi] / sigma
    pv, pf, pnn = [], [], []
    for _ in range(20):
      tr = buf.sample(cfg.batch_size)
      gg = np.asarray(tr.observation[:, STATE_DIM:], np.float32)
      pf.append(score(np.asarray(tr.observation[:, :STATE_DIM]),
                      np.asarray(tr.action), gg))
      pv.append(np.linalg.norm(gg[:, 2:5], axis=1) if len(gi) > 2
                else np.zeros(len(gg)))
      d = np.linalg.norm(gg[:, None, :] / sigma - bank_g[None, :, :], axis=2)
      pnn.append(d.min(1))
    pf, pv, pnn = (np.concatenate(pf), np.concatenate(pv),
                   np.concatenate(pnn))
    res['7_positive_goals'] = {
        'n_sampled': int(len(pf)),
        'critic_positive': dist(pf),
        'nn_distance_to_bank_sigma': dist(pnn),
        'frac_within_0.5_sigma_of_a_bank_entry': float((pnn < 0.5).mean()),
        'positive_goal_linear_speed': dist(pv),
        'note': 'the frozen training set contains no deaths, so a positive '
                'goal landing on a bank entry would mean the bank is naming '
                'ordinary expert states rather than failures. There is no '
                'death mask on the relabeler, by design.'}

    results[r['label']] = res
    print('\n  %s (alpha %g, step %d)' % (r['label'], r['alpha'], step))
    print('    1 f(bank)              mean %+.4f' % f_bank.mean())
    print('    2 alive - dead         mean %+.4f  frac>0 %.4f  (xy gap %.3f)'
          % (margin.mean(), (margin > 0).mean(), np.median(nn_gap[k])))
    print('    3 f(hold)              mean %+.4f  hold-dead %+.4f  frac>0 %.4f'
          % (f_hold.mean(), (f_hold - f_dead).mean(),
             (f_hold > f_dead).mean()))
    print('    6 mouth mode-zero      mean %+.4f  frac moving %.4f  |a| %.3f'
          % ((q_mode - q_zero).mean(), (q_mode > q_zero).mean(),
             np.linalg.norm(a_mode, axis=1).mean()))
    print('    7 positives nn to bank median %.3f sigma  within 0.5 %.4f'
          % (np.median(pnn), (pnn < 0.5).mean()))

  out['runs'] = results

  # -------------------------------------------------------------- table
  ks = sorted(results, key=lambda k2: results[k2]['alpha'])
  print('\n' + '=' * 100)
  print('ALPHA RESPONSE   (f(bank) moving alone is the f4 null: a real effect')
  print('                  needs columns 2 and 6 to move, with 3 intact)')
  print('=' * 100)
  hdr = ('%-24s%8s%11s%11s%9s%11s%11s%9s'
         % ('run', 'alpha', 'f(bank)', 'alive-dead', 'frac>0', 'f(hold)',
            'mode-zero', 'moving'))
  print(hdr)
  print('-' * len(hdr))
  for k2 in ks:
    r = results[k2]
    print('%-24s%8.2f%11.4f%11.4f%9.4f%11.4f%11.4f%9.4f'
          % (k2, r['alpha'],
             r['1_bank_response']['f_bank']['mean'],
             r['2_real_pairing']['margin_alive_minus_dead']['mean'],
             r['2_real_pairing']['frac_positive'],
             r['3_hold_preserved']['f_hold']['mean'],
             r['6_mouth_behaviour']['margin_mode_minus_zero']['mean'],
             r['6_mouth_behaviour']['frac_prefers_moving']))

  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'audit_%s.json' % args.goal_rep)
  with open(path, 'w') as f:
    json.dump(out, f, indent=2)
  print('\n-> %s' % path)


if __name__ == '__main__':
  main()
