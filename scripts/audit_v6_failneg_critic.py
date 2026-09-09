r"""Post-training critic audit for the V6 failure-negative sweep.

Read-only. No training, no dataset regeneration, nothing is written into a run
directory. It answers the one question the deployment numbers cannot:

    the arms differ in loss and in parameters, and their deployment outcomes
    are the same -- did the BANK move the critic at all, or did the critic
    move and the actor not follow?

Three sections, in the order they rule things out:

  1  BANK RESPONSE     f(s, a, g_bank) per alpha, split by bank class
                       (random / deliberate zone 1 / deliberate zone 2).
                       If this does not move with alpha, the failure term is
                       not reaching the critic and sections 2-3 are moot.

  2  REAL PAIRING      held-out failure states (their own collection seed,
                       never in the bank -- asserted) against the training
                       set's own band crossings matched on torso XY, per zone.
                       Different episodes from different collection runs,
                       nothing synthesised. The margin f(alive) - f(dead) is
                       the quantity a failure bank is supposed to open up.
                       Reported next to a POSITION CONTROL: the same margin
                       between two safe crossings matched the same way, which
                       is what "the critic just learned the place" looks like.

  3  MOUTH BEHAVIOUR   at real approach states in front of each band, the
                       critic's score of the ACTOR's own action against the
                       zero action (standing still), plus the commanded action
                       norm. This is where a policy that has learned to wait
                       would show it, and it uses the 8-D actions the actor
                       actually emits rather than a grid.

Also reported, not gated: the goal families (factual future, random state,
task goal, real crossing, held-out failure, bank), centred, so the bank's
score can be read against the scale the critic is using.

Usage:
  python scripts/audit_v6_failneg_critic.py
  python scripts/audit_v6_failneg_critic.py --ckpt-dir <dir> --bank <npz>
"""
import argparse
import glob as globmod
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.25')

import jax                                                # noqa: E402
import jax.numpy as jnp                                   # noqa: E402

from crl import checkpoint as ckpt_mod                    # noqa: E402
from crl import envs as envs_mod                          # noqa: E402
from crl import networks as networks_mod                  # noqa: E402
from crl import rockfall_clock_v6 as V6                   # noqa: E402
from crl.replay import obs_to_goal as np_obs_to_goal      # noqa: E402
import run_v6_failneg as RL                               # noqa: E402
import train_rockfall_clock_v6_baseline as B              # noqa: E402

CKPT_DIR = os.path.join('artifacts', 'v6_failneg', 'results', 'ckpt_p040_s0')
HELDOUT_DIR = os.path.join('artifacts', 'v6_failneg',
                           'candidates_p040_heldout')
OUT_DIR = os.path.join('artifacts', 'v6_failneg', 'audit_critic')
STATE_DIM = 29
STOP_V = 0.15
#: 'a0' -> 0.0, 'a1' -> 0.1, 'a05' -> 0.05 (the launcher's alpha_tag inverse)
ALPHA_RE = re.compile(r'v6fn_a(\d+)_s(\d+)_')


def parse_arm(path):
  m = ALPHA_RE.search(os.path.basename(path))
  if not m:
    return None
  digits, seed = m.group(1), int(m.group(2))
  alpha = 0.0 if digits == '0' else float('0.' + digits)
  return alpha, seed


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  return {'n': int(v.size), 'mean': float(v.mean()),
          'median': float(np.median(v)), 'std': float(v.std()),
          'p10': float(np.percentile(v, 10)),
          'p90': float(np.percentile(v, 90))}


def in_band(xy, zone):
  lo, hi = V6.HAZARD_X[zone]
  return ((np.abs(xy[:, 1]) < V6.HAZARD_HALF_Y)
          & (xy[:, 0] >= lo) & (xy[:, 0] <= hi))


def at_mouth(xy, zone):
  lo = V6.HAZARD_X[zone][0]
  return ((np.abs(xy[:, 1]) < V6.HAZARD_HALF_Y)
          & (xy[:, 0] >= V6.MOUTH_X[zone]) & (xy[:, 0] < lo))


def make_scorer(ckpt_path, cfg):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  qp, pp = st.q_params, st.policy_params
  want = int(cfg.obs_dim) + int(cfg.goal_dim)
  got = int(pp['mlp/~/linear_0']['w'].shape[0])
  assert got == want, ('checkpoint expects %d-dim observations, this goal '
                       'contract produces %d' % (got, want))

  @jax.jit
  def _scores(params, s, a, g):
    q = nets.q_network.apply(params, jnp.concatenate([s, g], axis=1), a)
    if q.ndim == 3:
      q = jnp.min(q, axis=-1)
    return jnp.diag(q)

  @jax.jit
  def _mode(params, o):
    p = nets.policy_network.apply(params, o)
    return jnp.tanh(p.loc)

  def scores(s, a, g, chunk=2048):
    out = []
    for i in range(0, len(s), chunk):
      out.append(np.asarray(_scores(qp, jnp.asarray(s[i:i + chunk]),
                                    jnp.asarray(a[i:i + chunk]),
                                    jnp.asarray(g[i:i + chunk]))))
    return np.concatenate(out) if out else np.zeros(0)

  def actor_mode(o, chunk=2048):
    out = []
    for i in range(0, len(o), chunk):
      out.append(np.asarray(_mode(pp, jnp.asarray(o[i:i + chunk]))))
    return np.concatenate(out) if out else np.zeros((0, cfg.action_dim))

  return scores, actor_mode, int(step)


def load_heldout(hdir):
  files = sorted(f for f in globmod.glob(os.path.join(hdir, '*.npz'))
                 if not f.endswith('_sidecar.npz'))
  if not files:
    raise SystemExit(
        'no held-out failure files in %s. Collect them with a seed different '
        'from the bank\'s:\n  python scripts/collect_v6_failure_candidates.py '
        '--all --episodes 100 --seed 3030 \\\n      --out-dir %s --name '
        'v6_failures_p040_heldout' % (hdir, hdir))
  states, zones, ident = [], [], []
  for f in files:
    with np.load(f, allow_pickle=False) as d:
      obs, lengths = d['obs'], np.asarray(d['lengths'], np.int64)
    with np.load(f.replace('.npz', '_sidecar.npz'), allow_pickle=False) as s:
      ep = np.asarray(s['episode_id'], np.int64)
      dr = np.asarray(s['death_row'], np.int64)
      fz = np.asarray(s['actual_failure_zone'], np.int64)
      seed = int(s['collection_seed'])
    assert np.array_equal(dr, lengths[:len(dr)] - 1)
    states.append(obs[ep, dr, :STATE_DIM].astype(np.float32))
    zones.append(fz)
    ident.extend((seed, os.path.basename(f), int(e)) for e in ep)
  return (np.concatenate(states, 0), np.concatenate(zones, 0), ident, files)


def nearest_by_xy(anchors, pool, chunk=256, exclude_self=False):
  idx = np.empty(len(anchors), np.int64)
  gap = np.empty(len(anchors), np.float64)
  for i in range(0, len(anchors), chunk):
    blk = anchors[i:i + chunk, :2]
    d2 = ((blk[:, None, 0] - pool[None, :, 0]) ** 2
          + (blk[:, None, 1] - pool[None, :, 1]) ** 2)
    if exclude_self:
      #: a crossing is its own nearest neighbour; take the second nearest.
      order = np.argsort(d2, axis=1)[:, 1]
    else:
      order = np.argmin(d2, axis=1)
    idx[i:i + chunk] = order
    gap[i:i + chunk] = np.sqrt(d2[np.arange(len(blk)), order])
  return idx, gap


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--ckpt-dir', default=CKPT_DIR)
  ap.add_argument('--ckpt-glob', default='v6fn_*_100k_gxy_p0.4-0.4_h800_final.pkl')
  ap.add_argument('--bank', default=RL.BANK_DEFAULT)
  ap.add_argument('--heldout-dir', default=HELDOUT_DIR)
  ap.add_argument('--env-name', default=B.ENV_XY)
  ap.add_argument('--npz', default=None)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n-anchors', type=int, default=6000)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  ckpts = sorted(globmod.glob(os.path.join(args.ckpt_dir, args.ckpt_glob)))
  arms = [(parse_arm(p), p) for p in ckpts]
  arms = sorted([(a[0], a[1], p) for a, p in arms if a])
  if not arms:
    raise SystemExit('no checkpoints matched %s in %s'
                     % (args.ckpt_glob, args.ckpt_dir))

  print('=' * 104)
  print('V6 FAILURE-NEGATIVE -- POST-TRAINING CRITIC AUDIT')
  print('=' * 104)

  #: the exact config the runs used, dims filled the way crl/train.py does
  from types import SimpleNamespace
  from verify_offline_d4rl import build_offline_cfg
  a = SimpleNamespace(env_name=args.env_name, npz=args.npz, horizon=B.HORIZON,
                      p_active_1=V6.P_ACTIVE_1, p_active_2=V6.P_ACTIVE_2,
                      t0_min_1=V6.T0_MIN_1, t0_max_1=V6.T0_MAX_1,
                      t0_min_2=V6.T0_MIN_2, t0_max_2=V6.T0_MAX_2,
                      seed=args.seed, resume=False)
  npz = args.npz or B._default_dataset(args.env_name)
  cfg = build_offline_cfg(max_steps=B.HORIZON, ckpt_dir='')
  B._apply_v6_config(cfg, a, npz)
  envs_mod.make_env(cfg.env_name, cfg, seed=cfg.seed)
  gi = list(cfg.goal_indices)
  print('  env %s   obs_dim %d + goal_dim %d, indices %s'
        % (cfg.env_name, cfg.obs_dim, cfg.goal_dim, gi))
  print('  dataset %s' % npz)
  print('  arms: %s' % ', '.join('alpha %.1f (s%d)' % (al, sd)
                                 for al, sd, _ in arms))

  # ------------------------------------------------------------ the data
  with np.load(npz, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    lengths = np.asarray(d['lengths'], np.int64)
  n_eps, L, W = obs.shape
  valid = (np.arange(L - 1)[None, :] < (lengths[:, None] - 1)).ravel()
  s_t = obs[:, :-1, :STATE_DIM].reshape(-1, STATE_DIM)[valid]
  s_n = obs[:, 1:, :STATE_DIM].reshape(-1, STATE_DIM)[valid]
  a_t = act[:, :-1, :].reshape(-1, 8)[valid]
  task_goal = obs[0, 0, STATE_DIM:].astype(np.float32).copy()
  print('  training transitions %s   task goal %s'
        % (format(len(s_t), ','), np.round(task_goal, 2).tolist()))

  crossings = {z: s_n[in_band(s_n[:, :2], z)] for z in (1, 2)}
  mouths = {z: np.flatnonzero(at_mouth(s_t[:, :2], z)) for z in (1, 2)}
  print('  real band crossings: Z1 %s, Z2 %s | approach states: Z1 %s, Z2 %s'
        % tuple(format(len(x), ',') for x in
                (crossings[1], crossings[2], mouths[1], mouths[2])))

  with np.load(args.bank, allow_pickle=False) as b:
    bank = np.asarray(b['goals'], np.float32)
    bank_kind = np.asarray(b['source_type'])
    bank_zone = np.asarray(b['actual_failure_zone'])
    bank_ident = set(zip(b['source_file'].tolist(), b['episode_id'].tolist()))
  held, held_zone, held_ident, held_files = load_heldout(args.heldout_dir)
  overlap = bank_ident & {(f, e) for _, f, e in held_ident}
  assert not overlap, ('the audit failure states are NOT isolated from the '
                       'bank: %d shared' % len(overlap))
  print('  bank %s  n=%d | held-out %d from %s -- disjoint from the bank'
        % (os.path.basename(args.bank), len(bank), len(held),
           [os.path.basename(f) for f in held_files]))

  rng = np.random.default_rng(args.seed)
  #: anchors are real (state, action) pairs in the approach zones and bands,
  #: i.e. where the decision this benchmark is about actually happens.
  sel = np.flatnonzero(at_mouth(s_t[:, :2], 1) | at_mouth(s_t[:, :2], 2)
                       | in_band(s_t[:, :2], 1) | in_band(s_t[:, :2], 2))
  if len(sel) > args.n_anchors:
    sel = rng.choice(sel, args.n_anchors, replace=False)
  S, A = s_t[sel], a_t[sel]
  print('  anchors %s' % format(len(S), ','))

  def as_goal(x):
    return np.ascontiguousarray(x[:, gi], np.float32)

  #: pairings, computed once and reused for every arm so the arms are
  #: compared on exactly the same pairs.
  pairs = {}
  for z in (1, 2):
    dead = held[held_zone == z]
    if not len(dead) or not len(crossings[z]):
      continue
    nn, gap = nearest_by_xy(dead, crossings[z])
    #: POSITION CONTROL: the same construction between two safe crossings.
    ctrl_src = crossings[z][rng.choice(len(crossings[z]),
                                       min(len(dead), len(crossings[z])),
                                       replace=False)]
    nn_c, gap_c = nearest_by_xy(ctrl_src, crossings[z], exclude_self=True)
    pairs[z] = dict(dead=dead, alive=crossings[z][nn], gap=gap,
                    ctrl_a=ctrl_src, ctrl_b=crossings[z][nn_c], gap_c=gap_c)
    print('  zone %d pairing: %d held-out deaths, median XY gap %.4f '
          '(control gap %.4f)' % (z, len(dead), np.median(gap),
                                  np.median(gap_c)))

  out = {'env': cfg.env_name, 'dataset': npz, 'bank': args.bank,
         'goal_indices': gi, 'n_anchors': int(len(S)),
         'heldout_files': [os.path.basename(f) for f in held_files],
         'arms': {}}

  for alpha, seed, path in arms:
    scores, actor_mode, step = make_scorer(path, cfg)
    res = {'alpha': alpha, 'seed': seed, 'checkpoint': path, 'step': step}

    # ---- 1 BANK RESPONSE -------------------------------------------
    pick = rng.integers(0, len(bank), len(S))
    f_bank = scores(S, A, as_goal(bank[pick]))
    res['1_bank_response'] = {'f_bank': dist(f_bank)}
    for label, m in (('random', bank_kind[pick] == 'random'),
                     ('deliberate_z1', (bank_kind[pick] == 'deliberate')
                      & (bank_zone[pick] == 1)),
                     ('deliberate_z2', (bank_kind[pick] == 'deliberate')
                      & (bank_zone[pick] == 2))):
      if m.any():
        res['1_bank_response']['f_bank_' + label] = dist(f_bank[m])

    # ---- 2 REAL PAIRING + POSITION CONTROL --------------------------
    res['2_real_pairing'] = {}
    for z, p in pairs.items():
      k = rng.integers(0, len(p['dead']), len(S))
      f_dead = scores(S, A, as_goal(p['dead'][k]))
      f_alive = scores(S, A, as_goal(p['alive'][k]))
      margin = f_alive - f_dead
      kc = rng.integers(0, len(p['ctrl_a']), len(S))
      f_c1 = scores(S, A, as_goal(p['ctrl_b'][kc]))
      f_c2 = scores(S, A, as_goal(p['ctrl_a'][kc]))
      res['2_real_pairing']['zone%d' % z] = {
          'margin_alive_minus_dead': dist(margin),
          'frac_positive': float((margin > 0).mean()),
          'f_alive': dist(f_alive), 'f_dead': dist(f_dead),
          'pair_xy_gap': dist(p['gap'][k]),
          'position_control_margin': dist(f_c1 - f_c2),
          'position_control_note':
              'the same nearest-XY construction between two SAFE crossings; '
              'a real failure effect has to exceed this'}

    # ---- 3 MOUTH BEHAVIOUR -----------------------------------------
    res['3_mouth_behaviour'] = {}
    for z in (1, 2):
      idx = mouths[z]
      if not len(idx):
        continue
      if len(idx) > 2048:
        idx = rng.choice(idx, 2048, replace=False)
      Sm = s_t[idx]
      g_task = np.tile(task_goal, (len(Sm), 1)).astype(np.float32)
      a_mode = actor_mode(np.concatenate([Sm, g_task], axis=1))
      q_mode = scores(Sm, a_mode, g_task)
      q_zero = scores(Sm, np.zeros_like(a_mode), g_task)
      res['3_mouth_behaviour']['zone%d' % z] = {
          'n_states': int(len(Sm)),
          'q_actor_mode': dist(q_mode), 'q_zero_action': dist(q_zero),
          'margin_mode_minus_zero': dist(q_mode - q_zero),
          'frac_prefers_moving': float((q_mode > q_zero).mean()),
          'actor_action_norm': dist(np.linalg.norm(a_mode, axis=1))}

    # ---- goal families, centred -------------------------------------
    fam = {'factual_future': scores(S, A, as_goal(s_n[sel])),
           'random_state': scores(S, A, as_goal(
               s_n[rng.integers(0, len(s_n), len(S))])),
           'task_goal': scores(S, A, np.tile(task_goal, (len(S), 1))),
           'failure_bank': f_bank}
    mu = float(np.concatenate(list(fam.values())).mean())
    res['4_goal_families'] = {k: {'raw': dist(v), 'centred': dist(v - mu)}
                              for k, v in fam.items()}
    res['4_overall_mean'] = mu

    out['arms']['alpha_%g' % alpha] = res
    print('\n  alpha %.1f (step %d)' % (alpha, step))
    print('    1 f(bank)             mean %+.4f   random %+.4f  Z1 %+.4f  '
          'Z2 %+.4f'
          % (f_bank.mean(),
             res['1_bank_response'].get('f_bank_random', {}).get('mean',
                                                                 float('nan')),
             res['1_bank_response'].get('f_bank_deliberate_z1',
                                        {}).get('mean', float('nan')),
             res['1_bank_response'].get('f_bank_deliberate_z2',
                                        {}).get('mean', float('nan'))))
    for z in sorted(pairs):
      r = res['2_real_pairing']['zone%d' % z]
      print('    2 Z%d alive-dead      mean %+.4f  frac>0 %.3f   | position '
            'control %+.4f' % (z, r['margin_alive_minus_dead']['mean'],
                               r['frac_positive'],
                               r['position_control_margin']['mean']))
    for z in (1, 2):
      r = res['3_mouth_behaviour'].get('zone%d' % z)
      if r:
        print('    3 Z%d mode-zero       mean %+.4f  frac moving %.3f  '
              '|a| %.3f' % (z, r['margin_mode_minus_zero']['mean'],
                            r['frac_prefers_moving'],
                            r['actor_action_norm']['mean']))

  # ------------------------------------------------------------ table
  print('\n' + '=' * 104)
  print('ALPHA RESPONSE')
  print('=' * 104)
  hdr = ('%7s %11s %11s %11s | %11s %11s | %11s %11s'
         % ('alpha', 'f(bank)', 'f(bank)Z1', 'f(bank)Z2', 'Z1 a-d',
            'Z1 ctrl', 'Z2 a-d', 'Z2 ctrl'))
  print(hdr)
  print('-' * len(hdr))
  for key in sorted(out['arms'], key=lambda k: out['arms'][k]['alpha']):
    r = out['arms'][key]
    b = r['1_bank_response']
    p1 = r['2_real_pairing'].get('zone1', {})
    p2 = r['2_real_pairing'].get('zone2', {})
    g = lambda d, *ks: (d.get(ks[0], {}).get(ks[1], float('nan'))
                        if d else float('nan'))
    print('%7.1f %11.4f %11.4f %11.4f | %11.4f %11.4f | %11.4f %11.4f'
          % (r['alpha'], b['f_bank']['mean'],
             g(b, 'f_bank_deliberate_z1', 'mean'),
             g(b, 'f_bank_deliberate_z2', 'mean'),
             g(p1, 'margin_alive_minus_dead', 'mean'),
             g(p1, 'position_control_margin', 'mean'),
             g(p2, 'margin_alive_minus_dead', 'mean'),
             g(p2, 'position_control_margin', 'mean')))

  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'critic_audit.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=2)
  print('\n-> %s' % path)


if __name__ == '__main__':
  main()
