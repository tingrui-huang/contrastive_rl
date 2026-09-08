r"""Small-scale verification of the V5 failure-negative pipeline. No training.

Every check is a hard PASS/FAIL and the script exits non-zero if any fails.
This is the gate that has to be green before a sweep is worth starting; it is
not a substitute for the sweep and it says nothing about whether the failure
negatives help.

  V1  BANK COMPOSITION      the mixture is what was asked for, entry counts and
                            fractions are reported, and every entry is a real
                            rock death from a real failure episode.
  V2  NO DUPLICATE EPISODES no (collection seed, source arm, episode) appears
                            twice; no two entries are the same vector.
  V3  DIMENSIONS            the bank stores 29-dim learner states, projects to
                            the right goal width under both goal contracts, and
                            fits inside the training batch (crl/losses.py pads
                            the second critic apply and needs n_bank <= B).
  V4  NORMALISATION         nothing rescales anything: obs_norm is off for this
                            env, so the bank enters the critic in exactly the
                            units the training states are in. Checked, not
                            assumed, by comparing a bank row against the raw
                            failure-episode row it came from.
  V5  NO PRIVILEGED INPUT   the bank npz's 'goals', the failure-episode npz and
                            the training npz carry no latent, no schedule, no
                            route label. The bank's provenance arrays exist but
                            are not the array crl/train.py reads.
  V6  ALPHA = 0 EQUIVALENCE two learners built from the same key, one with no
                            bank and one with the bank at alpha 0, produce
                            BIT-IDENTICAL parameters after several updates.
  V7  ALPHA > 0 IS LIVE     at alpha > 0 the parameters DIFFER, the reported
                            loss decomposition adds up to the reported loss,
                            and at alpha -> 0 the decomposition converges to
                            the baseline mean loss.
  V8  BANK/EVAL ISOLATION   the audit's held-out failures share no (file,
                            episode) with the bank, when a held-out set exists.

Usage:
  python scripts/verify_v5_failneg.py
  python scripts/verify_v5_failneg.py --bank <path> --goal-rep xyv
"""
import argparse
import glob as globmod
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.20')

import jax                                                # noqa: E402
import jax.numpy as jnp                                   # noqa: E402
import optax                                              # noqa: E402

from crl import losses as losses_mod                      # noqa: E402
from crl import networks as networks_mod                  # noqa: E402
from crl.losses import Transition                         # noqa: E402
from crl.replay import obs_to_goal as np_obs_to_goal      # noqa: E402
import run_v5_failneg as L                                # noqa: E402

STATE_DIM = 29
ACTION_DIM = 8
FAIL_DIR = 'artifacts/v5_failneg/failures'
HELDOUT_DIR = 'artifacts/v5_failneg/failures_heldout'
RESULTS = []


def check(name, ok, detail=''):
  RESULTS.append((name, bool(ok), detail))
  print('  %-4s %-26s %s' % ('PASS' if ok else 'FAIL', name, detail),
        flush=True)
  return bool(ok)


def tree_max_abs_diff(a, b):
  leaves = jax.tree_util.tree_leaves(
      jax.tree_util.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b))
  return float(max(float(v) for v in leaves)) if leaves else 0.0


def fake_batch(rng, obs_dim, goal_dim, n):
  o = rng.standard_normal((n, obs_dim + goal_dim)).astype(np.float32)
  o2 = rng.standard_normal((n, obs_dim + goal_dim)).astype(np.float32)
  a = rng.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32)
  a2 = rng.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32)
  return Transition(observation=jnp.asarray(o), action=jnp.asarray(a),
                    reward=jnp.zeros(n), discount=jnp.ones(n),
                    next_observation=jnp.asarray(o2),
                    next_action=jnp.asarray(a2))


def build(cfg, bank, key):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  start, end = cfg.start_index, cfg.end_index
  gidx = (None if cfg.goal_indices is None else jnp.asarray(cfg.goal_indices))

  def obs_to_goal(states):
    if gidx is not None:
      return states[:, gidx]
    return states[:, start:] if end == -1 else states[:, start:end]

  init, step = losses_mod.build_learner(
      nets, cfg, obs_to_goal, optax.adam(cfg.actor_learning_rate, eps=1e-7),
      optax.adam(cfg.learning_rate, eps=1e-7), fail_bank=bank)
  return init(key), step


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--bank', default=L.BANK_DEFAULT)
  ap.add_argument('--goal-rep', choices=sorted(L.GOAL_REPS), default='xyv')
  ap.add_argument('--fail-dir', default=FAIL_DIR)
  ap.add_argument('--heldout-dir', default=HELDOUT_DIR)
  ap.add_argument('--expect-compose', default='noisy=0.6,deliberate=0.4')
  ap.add_argument('--batch', type=int, default=0,
                  help='batch for the equivalence checks (0 = exactly the bank '
                       'size). The real runs use 1024; the identity is exact '
                       'at any size >= n_bank, which crl/losses.py requires.')
  ap.add_argument('--updates', type=int, default=3)
  ap.add_argument('--tolerance', type=float, default=0.02,
                  help='allowed absolute deviation of an achieved class '
                       'fraction from the requested one')
  args = ap.parse_args()
  L.select_rep(args.goal_rep)
  gi = list(L.GOAL_REPS[args.goal_rep]['indices'])

  print('=' * 88)
  print('V5 FAILURE-NEGATIVE VERIFICATION   bank %s   goal rep %s'
        % (args.bank, args.goal_rep))
  print('=' * 88)

  with np.load(args.bank, allow_pickle=False) as b:
    bank_states = np.asarray(b['goals'], np.float32)
    arm = np.asarray(b['source_arm'])
    ep = np.asarray(b['episode_id'], np.int64)
    src = np.asarray(b['source_file'])
    bank_seed = np.asarray(b['collection_seed'], np.int64)
    meta = json.loads(str(b['meta']))
    bank_keys = list(b.files)

  # ------------------------------------------------------------------- V1
  want = {}
  for part in args.expect_compose.split(','):
    k, v = part.split('=')
    want[k.strip()] = float(v)
  got = {c: float((arm == c).mean()) for c in sorted(set(arm.tolist()))}
  ok = (set(got) == set(want)
        and all(abs(got[c] - want[c]) <= args.tolerance for c in want))
  check('V1_BANK_COMPOSITION', ok,
        'n=%d  got %s  want %s' % (len(bank_states),
                                   {k: round(v, 3) for k, v in got.items()},
                                   want))
  # Verify EVERY selected row against its source learner file and privileged
  # sidecar.  Merely trusting the bank builder's prose would let a stale or
  # hand-edited bank pass this gate.
  source_cache = {}
  every_death, death_errors = True, []
  for i, (source, episode, source_arm, seed) in enumerate(
      zip(src.tolist(), ep.tolist(), arm.tolist(), bank_seed.tolist())):
    src_path = os.path.join(args.fail_dir, source)
    side_path = src_path.replace('.npz', '_sidecar.npz')
    try:
      if source not in source_cache:
        with np.load(src_path, allow_pickle=False) as d:
          source_cache[source] = {
              'obs': np.asarray(d['obs']),
              'lengths': np.asarray(d['lengths'], np.int64)}
        with np.load(side_path, allow_pickle=False) as s:
          source_cache[source].update({k: np.asarray(s[k]) for k in s.files})
      raw = source_cache[source]
      row = int(np.flatnonzero(raw['episode_id'] == episode)[0])
      death_row_i = int(raw['death_row'][row])
      valid = (
          str(raw['outcome'][row]) == 'rock_death'
          and bool(raw['death_in_corridor_row'][row])
          and str(raw['source_arm'][row]) == str(source_arm)
          and int(raw['collection_seed']) == int(seed)
          and death_row_i == int(raw['lengths'][episode]) - 1
          and np.array_equal(raw['obs'][episode, death_row_i, :STATE_DIM]
                             .astype(np.float32), bank_states[i]))
      if not valid:
        death_errors.append('%s:ep%d' % (source, episode))
    except (OSError, KeyError, IndexError, ValueError) as exc:
      death_errors.append('%s:ep%d (%s)' % (source, episode, exc))
    every_death = every_death and not death_errors
  fail_meta_ok = (meta.get('extraction_moment', '').startswith('settle 0')
                  and meta.get('n_bank') == len(bank_states)
                  and len(meta.get('entries', [])) == len(bank_states)
                  and every_death)
  check('V1b_REAL_FAILURES_ONLY', fail_meta_ok,
        'checked %d/%d source rows; extraction %s%s'
        % (len(bank_states) - len(death_errors), len(bank_states),
           meta.get('extraction_moment', '')[:40],
           ('; errors ' + repr(death_errors[:3])) if death_errors else ''))

  # ------------------------------------------------------------------- V2
  ident = list(zip(bank_seed.tolist(), arm.tolist(), ep.tolist()))
  uniq_rows = len(np.unique(bank_states, axis=0))
  check('V2_NO_DUPLICATE_EPISODES',
        len(set(ident)) == len(ident) and uniq_rows == len(bank_states),
        '%d unique (seed, arm, episode) / %d entries; %d unique vectors'
        % (len(set(ident)), len(ident), uniq_rows))

  # ------------------------------------------------------------------- V3
  cfg = L.build_cfg('fail', '', steps=1, seed=0, alpha=0.3)
  cfg.obs_dim, cfg.goal_dim, cfg.action_dim = STATE_DIM, len(gi), ACTION_DIM
  cfg.goal_indices = tuple(gi)
  proj = np_obs_to_goal(bank_states, cfg.start_index, cfg.end_index,
                        cfg.goal_indices)
  widths = {rep: len(spec['indices']) for rep, spec in L.GOAL_REPS.items()}
  ok3 = (bank_states.shape[1] == STATE_DIM
         and proj.shape[1] == len(gi)
         and len(bank_states) <= cfg.batch_size
         and all(bank_states[:, list(spec['indices'])].shape[1] == w
                 for (rep, spec), w in zip(L.GOAL_REPS.items(),
                                           widths.values())))
  check('V3_DIMENSIONS', ok3,
        'stored %s -> %s under %r; batch_size %d; widths %s'
        % (bank_states.shape, proj.shape, args.goal_rep, cfg.batch_size,
           widths))

  # ------------------------------------------------------------------- V4
  #: a bank row must be BYTE-IDENTICAL to the failure-episode row it came from.
  e0 = meta['entries'][0]
  src_path = os.path.join(args.fail_dir, e0['source_file'])
  ok4, detail4 = False, 'source episode file not found: %s' % src_path
  if os.path.exists(src_path):
    with np.load(src_path, allow_pickle=False) as d:
      raw = d['obs'][e0['episode_id'], e0['death_row'], :STATE_DIM]
    same = bool(np.array_equal(raw.astype(np.float32), bank_states[0]))
    norm_off = (getattr(cfg, 'obs_norm_mode', '') in ('', None))
    ok4 = same and norm_off
    detail4 = ('bank row 0 == raw death row (max |d| %.3g); obs_norm_mode %r'
               % (float(np.abs(raw - bank_states[0]).max()),
                  getattr(cfg, 'obs_norm_mode', '(field absent)')))
  check('V4_NORMALISATION', ok4, detail4)

  # ------------------------------------------------------------------- V5
  forbidden = ('rockfall_active', 'rockfall_start', 'rockfall_end', 'route',
               'intent', 'schedule', 'u', 'latent', 'teacher_decision')
  bank_leak = [k for k in bank_keys if k in forbidden]
  with np.load(L.DATASET, allow_pickle=False) as d:
    train_keys = set(d.files)
    train_w = d['obs'].shape[-1]
  train_leak = sorted(train_keys - {'obs', 'act', 'lengths', 'eval_goals',
                                    'meta'})
  fail_leak = []
  for f in sorted(globmod.glob(os.path.join(args.fail_dir, '*.npz'))):
    if f.endswith('_sidecar.npz'):
      continue
    with np.load(f, allow_pickle=False) as d:
      extra = sorted(set(d.files) - {'obs', 'act', 'lengths', 'eval_goals',
                                     'meta'})
    if extra:
      fail_leak.append((os.path.basename(f), extra))
  ok5 = (not bank_leak and not train_leak and not fail_leak
         and train_w == STATE_DIM + len(gi)
         and 'goals' in bank_keys)
  check('V5_NO_PRIVILEGED_INPUT', ok5,
        'bank arrays %s (train.py reads "goals" only); training npz keys %s, '
        'width %d; failure npz leaks %s'
        % (bank_keys, sorted(train_keys), train_w, fail_leak or 'none'))

  # ------------------------------------------------------------------- V6
  rng = np.random.default_rng(0)
  #: crl/losses.py pads the second critic apply, so the check batch can never
  #: be smaller than the bank. Keeping it AT the bank size makes the whole
  #: verification cheap without weakening any identity: alpha = 0 equivalence
  #: and the loss decomposition are exact at every batch size >= n_bank.
  batch = int(args.batch) if args.batch else len(bank_states)
  if batch < len(bank_states):
    raise SystemExit('--batch %d < bank size %d; crl/losses.py requires '
                     'n_bank <= batch_size' % (batch, len(bank_states)))
  batches = [fake_batch(rng, STATE_DIM, len(gi), batch)
             for _ in range(args.updates)]
  key = jax.random.PRNGKey(0)
  cfg0 = L.build_cfg('base', '', steps=1, seed=0)
  cfg0.obs_dim, cfg0.goal_dim, cfg0.action_dim = (STATE_DIM, len(gi),
                                                  ACTION_DIM)
  cfg0.goal_indices = tuple(gi)
  cfg0.batch_size = batch
  st_a, step_a = build(cfg0, None, key)
  st_b, step_b = build(cfg0, proj, key)          # bank present, alpha 0
  for tr in batches:
    st_a, _ = step_a(st_a, tr)
    st_b, _ = step_b(st_b, tr)
  d_q = tree_max_abs_diff(st_a.q_params, st_b.q_params)
  d_p = tree_max_abs_diff(st_a.policy_params, st_b.policy_params)
  check('V6_ALPHA0_EQUIVALENCE', d_q == 0.0 and d_p == 0.0,
        'after %d updates: max |dq| %.3g, max |dpi| %.3g (bank loaded but '
        'alpha 0 -> the fail branch is skipped entirely)'
        % (args.updates, d_q, d_p))

  # ------------------------------------------------------------------- V7
  cfg1 = L.build_cfg('fail', '', steps=1, seed=0, alpha=0.3)
  cfg1.obs_dim, cfg1.goal_dim, cfg1.action_dim = (STATE_DIM, len(gi),
                                                  ACTION_DIM)
  cfg1.goal_indices = tuple(gi)
  cfg1.batch_size = batch
  st_c, step_c = build(cfg1, proj, key)
  m_last = {}
  for tr in batches:
    st_c, m_last = step_c(st_c, tr)
  d_live = tree_max_abs_diff(st_a.q_params, st_c.q_params)
  pos = float(m_last['critic_pos_term'])
  ord_t = float(m_last['critic_neg_ord_term'])
  fail_t = float(m_last['critic_neg_fail_term'])
  total = float(m_last['critic_loss'])
  adds_up = abs((pos + ord_t + fail_t) - total) < 1e-5
  #: alpha -> 0 must reproduce the baseline mean loss algebraically.
  cfg_eps = L.build_cfg('fail', '', steps=1, seed=0, alpha=1e-8)
  cfg_eps.obs_dim, cfg_eps.goal_dim, cfg_eps.action_dim = (STATE_DIM, len(gi),
                                                           ACTION_DIM)
  cfg_eps.goal_indices = tuple(gi)
  cfg_eps.batch_size = batch
  st_e, step_e = build(cfg_eps, proj, key)
  st_e2, m_eps = step_e(st_e, batches[0])
  st_a2, m_base = build(cfg0, None, key)[1](build(cfg0, None, key)[0],
                                            batches[0])
  gap = abs(float(m_eps['critic_loss']) - float(m_base['critic_loss']))
  check('V7_ALPHA_LIVE', d_live > 0.0 and adds_up and gap < 1e-5,
        'alpha 0.3 moves params (max |dq| %.3g); pos %.5f + ord %.5f + fail '
        '%.5f = %.5f vs reported %.5f; alpha 1e-8 vs baseline loss gap %.3g'
        % (d_live, pos, ord_t, fail_t, pos + ord_t + fail_t, total, gap))

  # ------------------------------------------------------------------- V8
  held = sorted(f for f in globmod.glob(os.path.join(args.heldout_dir,
                                                     '*.npz'))
                if not f.endswith('_sidecar.npz'))
  if not held:
    check('V8_BANK_EVAL_ISOLATION', False,
          'no held-out failure set in %s -- the audit has nothing isolated to '
          'pair against. Collect one with a different seed.' % args.heldout_dir)
  else:
    hid, held_rows = set(), []
    for f in held:
      with np.load(f, allow_pickle=False) as d:
        hobs = np.asarray(d['obs'])
      with np.load(f.replace('.npz', '_sidecar.npz'), allow_pickle=False) as s:
        seed = int(s['collection_seed'])
        for row, (source_arm, episode, death_row_i) in enumerate(zip(
            s['source_arm'].tolist(), s['episode_id'].tolist(),
            s['death_row'].tolist())):
          hid.add((seed, str(source_arm), int(episode)))
          held_rows.append(hobs[int(episode), int(death_row_i), :STATE_DIM])
    shared = set(ident) & hid
    held_rows = np.asarray(held_rows, np.float32)
    # Exact state overlap is an additional rename-proof guard.  The semantic
    # identity above catches a copied collection; this catches copied vectors
    # even if its provenance was also rewritten.
    bank_row_bytes = {np.ascontiguousarray(x).tobytes() for x in bank_states}
    shared_rows = sum(np.ascontiguousarray(x).tobytes() in bank_row_bytes
                      for x in held_rows)
    check('V8_BANK_EVAL_ISOLATION', not shared and shared_rows == 0,
          '%d held-out episodes, %d shared (seed, arm, episode), %d exact '
          'shared death states' % (len(hid), len(shared), shared_rows))

  print('=' * 88)
  bad = [n for n, ok, _ in RESULTS if not ok]
  print('VERDICT: %s  (%d/%d checks passed)'
        % ('ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad),
           len(RESULTS) - len(bad), len(RESULTS)))
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
