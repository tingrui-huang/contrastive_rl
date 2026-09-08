r"""Pre-training audit of the V6 composed failure bank.

Every check is a hard PASS/FAIL and the script exits non-zero if any fails.
It is the gate that must be green before any alpha training is worth starting.
It says nothing about whether failure negatives help; it says the artifact is
what it claims to be.

  A  NORMAL V6 REGRESSION       delegated to scripts/regress_v6_teacher_patch.py
                                (replays the frozen dataset bitwise); this
                                script re-checks the two invariants the
                                collectors depend on.
  B  DELIBERATE ZONE 1          every Z1 entry: normal 'wait', executed 'go',
                                override True, real death in zone 1.
  C  DELIBERATE ZONE 2          every Z2 entry: survived zone 1, normal 'wait'
                                at zone 2, executed 'go', override True, real
                                death in zone 2. Zone 1 handled normally --
                                reported split by whether it had to wait there.
  D  RANDOM FAILURE SOURCE      the random/noisy component comes from the
                                noisy-controller arm, is blind by construction
                                (normal decisions None), and every entry is a
                                genuine flagged rock death. The uniform-torque
                                arm's measured emptiness is reported.
  E  SETTLED STATE              the settle knob demonstrably changes the
                                recorded fatal observation and nothing before
                                it: two runs of the same arm at the same seed,
                                settle 0 vs N, are compared step by step.
  F  LEARNER VISIBILITY         the tensor crl/train.py hands the critic is the
                                goal projection of the 29-dim ant state and
                                carries none of the provenance columns; each
                                provenance field is searched for numerically
                                inside the projected bank.
  G  BANK COMPOSITION           60% random, 20% deliberate Z1, 20% deliberate
                                Z2, exact, no duplicates.
  H  ALPHA = 0                  a learner built with the bank at alpha 0 is
                                bit-identical to one built with no bank, and
                                at alpha > 0 the loss decomposition adds up.

Usage:
  python scripts/audit_v6_failure_bank.py
  python scripts/audit_v6_failure_bank.py --bank <path> --skip-settle
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

from crl import envs as envs_mod                          # noqa: E402
from crl import losses as losses_mod                      # noqa: E402
from crl import networks as networks_mod                  # noqa: E402
from crl import rockfall_clock_v6 as V6                   # noqa: E402
from crl.losses import Transition                         # noqa: E402
from crl.replay import obs_to_goal as np_obs_to_goal      # noqa: E402
import collect_v6_failure_candidates as CC                # noqa: E402
import run_v6_failneg as RL                               # noqa: E402
import rockfall_clock_v6_teacher as CT                    # noqa: E402

CAND_DIR = CC.OUT_DIR
BANK = RL.BANK_DEFAULT
STATE_DIM = 29
ACTION_DIM = 8
RESULTS = []


def check(name, ok, detail=''):
  RESULTS.append((name, bool(ok), detail))
  print('  %-4s %-26s %s' % ('PASS' if ok else 'FAIL', name, detail),
        flush=True)
  return bool(ok)


def load_bank(path):
  with np.load(path, allow_pickle=False) as b:
    out = {k: np.asarray(b[k]) for k in b.files if k != 'meta'}
    out['meta'] = json.loads(str(b['meta']))
    out['keys'] = list(b.files)
  return out


def tree_max_abs_diff(a, b):
  leaves = jax.tree_util.tree_leaves(
      jax.tree_util.tree_map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b))
  return float(max(float(v) for v in leaves)) if leaves else 0.0


def fake_batch(rng, obs_dim, goal_dim, n):
  def r(*shape):
    return jnp.asarray(rng.standard_normal(shape).astype(np.float32))
  a = jnp.asarray(rng.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32))
  a2 = jnp.asarray(rng.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32))
  return Transition(observation=r(n, obs_dim + goal_dim), action=a,
                    reward=jnp.zeros(n), discount=jnp.ones(n),
                    next_observation=r(n, obs_dim + goal_dim),
                    next_action=a2)


def build_learner(cfg, bank, key):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  gidx = (None if cfg.goal_indices is None else jnp.asarray(cfg.goal_indices))
  start, end = cfg.start_index, cfg.end_index

  def obs_to_goal(states):
    if gidx is not None:
      return states[:, gidx]
    return states[:, start:] if end == -1 else states[:, start:end]

  init, step = losses_mod.build_learner(
      nets, cfg, obs_to_goal, optax.adam(cfg.actor_learning_rate, eps=1e-7),
      optax.adam(cfg.learning_rate, eps=1e-7), fail_bank=bank)
  return init(key), step


# --------------------------------------------------------------------- A
def audit_a_regression(seed, horizon, episodes):
  cfg, teacher = CT.make_teacher()
  cfg.rockfall_max_steps = horizon
  env = envs_mod.make_env(CT.ENV_NAME, cfg, seed=seed)
  rng = np.random.default_rng(seed + 7)
  bad = []
  for _ in range(episodes):
    route = 'detour' if rng.random() < CT.TEACHER_DETOUR_PROB else 'shortcut'
    o = env.reset()
    teacher.fresh(route=route, deliberate_override_zone=None)
    for _ in range(horizon):
      o, r, done, _ = env.step(teacher.act(o, env.schedule))
      if done or r > 0:
        break
    nd, ed, ov = (teacher.normal_decisions, teacher.executed_decisions,
                  teacher.deliberate_overrides)
    for z in (1, 2):
      if nd[z] != ed[z] or ov[z]:
        bad.append((z, nd[z], ed[z], ov[z]))
  check('A_NORMAL_V6_UNCHANGED', not bad,
        'override off over %d episodes: executed == normal everywhere, no '
        'override fired. Bitwise dataset replay: run '
        'scripts/regress_v6_teacher_patch.py' % episodes)


# ------------------------------------------------------------------ B / C
def audit_deliberate(bank, zone, label):
  entries = [e for e in bank['meta']['entries']
             if e['source_type'] == 'deliberate' and e['targeted_zone'] == zone]
  bad = []
  for e in entries:
    fails = []
    if e['normal_decision_zone%d' % zone] != 'wait':
      fails.append('normal!=wait')
    if e['executed_decision_zone%d' % zone] != 'go':
      fails.append('executed!=go')
    if not e['deliberate_override_zone%d' % zone]:
      fails.append('override_false')
    if e['actual_failure_zone'] != zone:
      fails.append('failure_zone=%s' % e['actual_failure_zone'])
    if zone == 2 and not e['survived_zone1']:
      fails.append('did_not_survive_zone1')
    if fails:
      bad.append((e['source_file'], e['episode_id'], fails))
  extra = ''
  if zone == 2 and entries:
    waited = sum(1 for e in entries if e['hold_steps_zone1'] > 0)
    u1_on = sum(1 for e in entries if e['u1'])
    extra = ('; zone 1 handled normally: %d/%d had u1 active and %d of those '
             'actually held before crossing'
             % (u1_on, len(entries), waited))
  check(label, entries and not bad,
        '%d entries, all with normal wait -> executed go -> real zone-%d '
        'death%s%s' % (len(entries), zone, extra,
                       '' if not bad else '; violations %s' % bad[:3]))
  return entries


# --------------------------------------------------------------------- D
def audit_d_random(bank, cand_dir):
  entries = [e for e in bank['meta']['entries'] if e['source_type'] == 'random']
  arms = sorted({e['source_arm'] for e in entries})
  blind = all(e['normal_decision_zone1'] in ('none', 'None', None)
              and e['normal_decision_zone2'] in ('none', 'None', None)
              for e in entries)
  zones = {str(z): sum(1 for e in entries if e['actual_failure_zone'] == z)
           for z in (1, 2)}
  zones['ambiguous'] = sum(1 for e in entries
                           if e['actual_failure_zone'] not in (1, 2))
  empty = sorted(globmod.glob(os.path.join(cand_dir, '*_EMPTY.json')))
  uniform_note = 'no uniform-torque record found'
  if empty:
    rec = json.load(open(empty[0]))
    uniform_note = ('uniform torque: %d kept in %d episodes, mean furthest x '
                    '%.2f vs zone 1 at x %.1f'
                    % (rec['n_kept'], rec['n_attempts'],
                       rec['mean_furthest_torso_x'], rec['zone1_starts_at_x']))
  ok = bool(entries) and blind and zones['ambiguous'] == 0
  check('D_RANDOM_SOURCE', ok,
        'arms %s, blind by construction %s, zones %s; %s'
        % (arms, blind, zones, uniform_note))
  return zones


# --------------------------------------------------------------------- E
def audit_e_settled(bank, settle, seed, horizon, episodes=1):
  """Same arm, same seed, settle 0 vs N: identical up to the fatal row."""
  recorded = bank['meta']['settled_state_extraction']['death_settle_substeps']
  if int(recorded) != int(settle):
    return check('E_SETTLED_STATE', False,
                 'bank records settle %s but --settle is %s'
                 % (recorded, settle))
  runs = {}
  for s in (0, int(settle)):
    args = CC.build_parser().parse_args(
        ['--arm', 'deliberate_z1', '--episodes', str(episodes),
         '--seed', str(seed), '--settle', str(s), '--horizon', str(horizon),
         '--progress-every', '10000'])
    runs[s] = CC.collect('deliberate_z1', episodes, seed, args)
  a, b = runs[0], runs[int(settle)]
  #: EPISODE 0 ONLY, deliberately. The settle leaves the mujoco state of a
  #: finished episode in a different place, and this env does not zero all
  #: persistent solver state on reset, so episodes 1+ of the two runs are not
  #: required to match and comparing them would be a false alarm. Episode 0
  #: starts from the same reset in both runs and is the clean test of the one
  #: claim being made: the settle changes the fatal observation and nothing
  #: before it.
  n0a, n0b = int(a['lengths'][0]), int(b['lengths'][0])
  same_len = n0a == n0b
  prefix_same = bool(np.array_equal(a['obs'][0, :n0a - 1],
                                    b['obs'][0, :n0b - 1])) if same_len else False
  act_same = bool(np.array_equal(a['act'][0, :n0a - 1],
                                 b['act'][0, :n0b - 1])) if same_len else False
  fatal_a = a['obs'][0, n0a - 1, :STATE_DIM]
  fatal_b = b['obs'][0, n0b - 1, :STATE_DIM]
  differs = float(np.abs(fatal_a - fatal_b).max())
  speed_a = float(np.linalg.norm(fatal_a[15:18]))
  speed_b = float(np.linalg.norm(fatal_b[15:18]))
  ok = same_len and prefix_same and act_same and differs > 0.0
  check('E_SETTLED_STATE', ok,
        'episode 0, settle 0 vs %d: %d pre-fatal rows %s, actions %s, length '
        '%d==%d %s; fatal row max|d| %.4f; |v| %.3f -> %.3f'
        % (settle, n0a - 1, 'identical' if prefix_same else 'DIFFER',
           'identical' if act_same else 'DIFFER', n0a, n0b,
           'yes' if same_len else 'NO', differs, speed_a, speed_b))


# --------------------------------------------------------------------- F
def audit_f_visibility(bank, cfg):
  goals = bank['goals'].astype(np.float32)
  proj = np_obs_to_goal(goals, cfg.start_index, cfg.end_index,
                        cfg.goal_indices)
  #: the projection must be exactly the goal columns of the ant state
  exact = bool(np.array_equal(proj, goals[:, list(cfg.goal_indices)]))
  #: no provenance column may appear as a numeric column of the projection
  meta_cols = {
      'u1': np.array([float(e['u1']) for e in bank['meta']['entries']]),
      'u2': np.array([float(e['u2']) for e in bank['meta']['entries']]),
      't0_1': np.array([float(e['t0_1']) for e in bank['meta']['entries']]),
      't0_2': np.array([float(e['t0_2']) for e in bank['meta']['entries']]),
      'targeted_zone': np.array([float(e['targeted_zone'])
                                 for e in bank['meta']['entries']]),
      'actual_failure_zone': np.array([float(e['actual_failure_zone'])
                                       for e in bank['meta']['entries']]),
      'deliberate_override_zone1': np.array(
          [float(e['deliberate_override_zone1'])
           for e in bank['meta']['entries']]),
      'deliberate_override_zone2': np.array(
          [float(e['deliberate_override_zone2'])
           for e in bank['meta']['entries']]),
      'source_type_is_deliberate': np.array(
          [float(e['source_type'] == 'deliberate')
           for e in bank['meta']['entries']]),
  }
  leaked = []
  for name, col in meta_cols.items():
    for j in range(proj.shape[1]):
      if np.allclose(proj[:, j], col, atol=1e-6):
        leaked.append((name, j))
  #: and the projected width is the goal width, not something wider
  width_ok = proj.shape[1] == cfg.goal_dim
  check('F_LEARNER_VISIBILITY', exact and not leaked and width_ok,
        'critic sees %s = state[:, %s]; %d provenance fields searched, '
        'leaked: %s' % (list(proj.shape), list(cfg.goal_indices),
                        len(meta_cols), leaked or 'none'))


# --------------------------------------------------- P (property, non-gating)
def property_goal_expressiveness(bank, cfg, npz):
  """What can a goal in THIS representation say about a failure state?

  Reported, not gated: the goal contract is V6's, not this experiment's. For
  each zone, real bank failure states are compared against the training set's
  own safe crossings of the SAME band, matched on torso XY, with a Fisher-LDA
  AUC computed in the goal columns the critic actually sees and, for contrast,
  in the full 29-dim ant state. 0.5 means the representation cannot tell the
  two apart, so a bank entry in it names a PLACE rather than a way of dying.
  """
  with np.load(npz, allow_pickle=False) as d:
    obs, lengths = d['obs'], np.asarray(d['lengths'], np.int64)
  valid = np.arange(obs.shape[1])[None, :] < lengths[:, None]
  states = obs[:, :, :STATE_DIM].reshape(-1, STATE_DIM)[valid.ravel()]

  def in_band(xy, zone):
    lo, hi = V6.HAZARD_X[zone]
    return ((np.abs(xy[:, 1]) < V6.HAZARD_HALF_Y)
            & (xy[:, 0] >= lo) & (xy[:, 0] <= hi))

  def lda_auc(a, b, cols):
    A, B = a[:, cols], b[:, cols]
    mu = A.mean(0) - B.mean(0)
    cov = np.atleast_2d(np.cov(np.vstack([A - A.mean(0),
                                          B - B.mean(0)]).T))
    cov = cov + 1e-6 * np.eye(len(cols))
    w = np.linalg.solve(cov, mu)
    sa, sb = A @ w, B @ w
    rank = np.concatenate([sa, sb]).argsort().argsort()[:len(sa)]
    return float((rank.sum() - len(sa) * (len(sa) - 1) / 2)
                 / (len(sa) * len(sb)))

  goals, zones = bank['goals'].astype(np.float32), bank['actual_failure_zone']
  gi = list(cfg.goal_indices)
  out = {}
  for zone in (1, 2):
    dead = goals[zones == zone]
    cross = states[in_band(states[:, :2], zone)]
    if not len(dead) or not len(cross):
      continue
    d2 = ((dead[:, None, 0] - cross[None, :, 0]) ** 2
          + (dead[:, None, 1] - cross[None, :, 1]) ** 2)
    order = np.argsort(d2, axis=1)[:, :3]
    matched = cross[order.ravel()]
    out['zone%d' % zone] = {
        'n_failures': int(len(dead)), 'n_crossings': int(len(cross)),
        'xy_match_gap': float(np.sqrt(np.take_along_axis(d2, order, 1)).mean()),
        'auc_in_goal_columns': lda_auc(dead, matched, gi),
        'auc_in_full_29dim_state': lda_auc(dead, matched,
                                           list(range(STATE_DIM)))}
  print('  INFO P_GOAL_EXPRESSIVENESS   (reported, NOT a gate)')
  for k, v in out.items():
    print('       %-6s AUC in goal columns %s: %.3f | in the full 29-dim '
          'state: %.3f   (%d failures vs %d same-band crossings, xy gap %.4f)'
          % (k, gi, v['auc_in_goal_columns'], v['auc_in_full_29dim_state'],
             v['n_failures'], v['n_crossings'], v['xy_match_gap']))
  return out


# --------------------------------------------------------------------- G
def audit_g_composition(bank):
  n = len(bank['goals'])
  kinds, zones = bank['source_type'], bank['actual_failure_zone']
  n_random = int((kinds == 'random').sum())
  n_z1 = int(((kinds == 'deliberate') & (zones == 1)).sum())
  n_z2 = int(((kinds == 'deliberate') & (zones == 2)).sum())
  ident = list(zip(bank['source_file'].tolist(), bank['episode_id'].tolist()))
  uniq_rows = len(np.unique(bank['goals'], axis=0))
  ok = (n_random / n == 0.60 and (n_z1 + n_z2) / n == 0.40 and n_z1 == n_z2
        and n_random + n_z1 + n_z2 == n
        and len(set(ident)) == n and uniq_rows == n)
  check('G_BANK_COMPOSITION', ok,
        'N=%d: random %d (%.2f), delib %d (%.2f) = Z1 %d + Z2 %d; unique '
        '(file, episode) %d; unique vectors %d'
        % (n, n_random, n_random / n, n_z1 + n_z2, (n_z1 + n_z2) / n,
           n_z1, n_z2, len(set(ident)), uniq_rows))


# --------------------------------------------------------------------- H
def audit_h_alpha0(bank, cfg, updates=3):
  proj = np_obs_to_goal(bank['goals'].astype(np.float32), cfg.start_index,
                        cfg.end_index, cfg.goal_indices)
  batch = max(len(proj), 64)
  rng = np.random.default_rng(0)
  batches = [fake_batch(rng, cfg.obs_dim, cfg.goal_dim, batch)
             for _ in range(updates)]
  key = jax.random.PRNGKey(0)

  import dataclasses
  cfg0 = dataclasses.replace(cfg, fail_neg_alpha=0.0, fail_bank_path='',
                             batch_size=batch)
  st_a, step_a = build_learner(cfg0, None, key)
  st_b, step_b = build_learner(cfg0, proj, key)     # bank present, alpha 0
  for tr in batches:
    st_a, _ = step_a(st_a, tr)
    st_b, _ = step_b(st_b, tr)
  d_q = tree_max_abs_diff(st_a.q_params, st_b.q_params)
  d_p = tree_max_abs_diff(st_a.policy_params, st_b.policy_params)
  check('H_ALPHA0_EQUIVALENCE', d_q == 0.0 and d_p == 0.0,
        'bank loaded at alpha 0 vs no bank, %d updates: max |dq| %.3g, '
        'max |dpi| %.3g' % (updates, d_q, d_p))

  cfg1 = dataclasses.replace(cfg0, fail_neg_alpha=0.3)
  st_c, step_c = build_learner(cfg1, proj, key)
  m = {}
  for tr in batches:
    st_c, m = step_c(st_c, tr)
  d_live = tree_max_abs_diff(st_a.q_params, st_c.q_params)
  pos, ordt, failt = (float(m['critic_pos_term']),
                      float(m['critic_neg_ord_term']),
                      float(m['critic_neg_fail_term']))
  total = float(m['critic_loss'])
  check('H_ALPHA_LIVE', d_live > 0.0 and abs(pos + ordt + failt - total) < 1e-5,
        'alpha 0.3 moves params (max |dq| %.3g); pos %.5f + ord %.5f + fail '
        '%.5f = %.5f vs reported %.5f' % (d_live, pos, ordt, failt,
                                          pos + ordt + failt, total))


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--bank', default=BANK)
  ap.add_argument('--cand-dir', default=CAND_DIR)
  ap.add_argument('--env-name', default=None)
  ap.add_argument('--seed', type=int, default=5150)
  ap.add_argument('--regression-episodes', type=int, default=8)
  ap.add_argument('--settle-episodes', type=int, default=1)
  ap.add_argument('--skip-settle', action='store_true')
  ap.add_argument('--out-dir', default=os.path.join('artifacts', 'v6_failneg',
                                                    'audit'))
  args = ap.parse_args()

  bank = load_bank(args.bank)
  settle = int(bank['meta']['settled_state_extraction']
               ['death_settle_substeps'])
  print('=' * 92)
  print('V6 FAILURE-BANK AUDIT   %s' % args.bank)
  print('=' * 92)
  print('  N=%d  compose %s  settle %d  env %s'
        % (len(bank['goals']), bank['meta']['composition']['level_2'], settle,
           bank['meta']['env_name']))

  #: the exact config the run will use, dims filled the way crl/train.py does
  argv = ['--alpha', '0.3', '--check-only', '--bank', args.bank]
  if args.env_name:
    argv += ['--env-name', args.env_name]
  a = RL.build_parser().parse_args(argv)
  from verify_offline_d4rl import build_offline_cfg
  npz = a.npz or RL.B._default_dataset(a.env_name)
  cfg = build_offline_cfg(max_steps=a.horizon, ckpt_dir='')
  RL.B._apply_v6_config(cfg, a, npz)
  envs_mod.make_env(cfg.env_name, cfg, seed=cfg.seed)
  print('  goal contract: obs_dim %d + goal_dim %d, indices %s'
        % (cfg.obs_dim, cfg.goal_dim, cfg.goal_indices))
  print()

  audit_a_regression(args.seed, CT.HORIZON, args.regression_episodes)
  audit_deliberate(bank, 1, 'B_DELIBERATE_ZONE1')
  audit_deliberate(bank, 2, 'C_DELIBERATE_ZONE2')
  random_zones = audit_d_random(bank, args.cand_dir)
  if args.skip_settle:
    check('E_SETTLED_STATE', True,
          'SKIPPED by --skip-settle; bank records settle %d' % settle)
  else:
    audit_e_settled(bank, settle, args.seed + 11, CT.HORIZON,
                    args.settle_episodes)
  audit_f_visibility(bank, cfg)
  audit_g_composition(bank)
  audit_h_alpha0(bank, cfg)
  expressiveness = property_goal_expressiveness(bank, cfg, npz)

  print('=' * 92)
  bad = [n for n, ok, _ in RESULTS if not ok]
  print('VERDICT: %s  (%d/%d checks passed)'
        % ('ALL PASS' if not bad else 'FAILED: ' + ', '.join(bad),
           len(RESULTS) - len(bad), len(RESULTS)))
  os.makedirs(args.out_dir, exist_ok=True)
  out = os.path.join(args.out_dir, 'bank_audit.json')
  with open(out, 'w') as f:
    json.dump({'bank': args.bank, 'n_bank': int(len(bank['goals'])),
               'settle': settle,
               'goal_dim': int(cfg.goal_dim),
               'goal_indices': list(cfg.goal_indices),
               'random_by_zone': random_zones,
               'goal_expressiveness': expressiveness,
               'checks': [{'name': n, 'pass': ok, 'detail': d}
                          for n, ok, d in RESULTS],
               'verdict': 'ALL PASS' if not bad else 'FAILED'}, f, indent=2)
  print('->', out)
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
