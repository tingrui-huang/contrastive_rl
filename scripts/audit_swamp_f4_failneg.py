r"""Critic-semantic audit of the frame-stacked (f4) failure-aware CRL runs.

Read-only. No training, no dataset regeneration, no checkpoint is written.
``critic_scores`` is imported from scripts/audit_swamp_z_failneg.py so the
scoring function is literally the same code the z audits used.

WHY THIS EXISTS. The deployment report alone cannot distinguish "the bank did
nothing to the critic" from "the critic moved and the actor did not follow".
The 27-run 2-D null looked identical in every deployment cell -- zero variance
everywhere -- and the ONLY quantity that responded was f(., ., g_bank). That
quantity is invisible to a deployment eval, so it needs this script.

WHAT CHANGES FROM THE Z AUDITS, AND WHY IT IS NOT A PORT. In the z envs a
failure state is (x, y, z<0) and its safe counterpart is (x, y, 0): one column
differs, so the counterfactual goal is synthesised by overwriting that column.
f4 has no such column. Here the separating quantity is VELOCITY --

    dead     [p, p, p, p]            zero velocity, the freeze signature
    passing  [p, p_1, p_2, p_3]      non-zero velocity, same current position p

-- so the paired diagnostic zeroes the velocity of a REAL observed stack
instead of overwriting a depth. Section 2 does that synthetically (the current
position matches EXACTLY, which is the property that made the z pairing
informative); section 3 refuses to synthesise anything and pairs real frozen
stacks against real moving ones found nearest to them, so the two sections
fail in different ways if the pairing is what is driving a result. Section 3
reports its pairing BOTH ways: the unrestricted nearest neighbour, which is
usually the anchor's own death step (identical frame 0, tightest possible pair,
but the same trajectory), and a cross-episode-only variant, which is the
"a survivor walked through here" comparison people will read it as.

Section 5 replaces the z depth-response curve with a VELOCITY-response curve,
which is its exact analogue: scale a real stack's velocity from 1.0 down to
0.0 and watch the critic slide from "passing" to "dead".

THE FORK MARGIN (section 6) USES DISJOINT ACTION MASKS. An earlier probe on
this benchmark reported inflated margins because the safe-ward (a_y < -0.3)
and shortcut-ward (a_x > 0.3) half-planes OVERLAPPED on 56/441 grid actions,
so one action could be the argmax of both sides. The partition here is by
dominant component and is disjoint by construction; the script asserts it.

Runs are DISCOVERED, not hard-coded: every swamp_windy_f4_z*_s*/final.pkl is
audited and its alpha is read from that run's own arm_provenance.json, so an
alpha sweep split across machines audits correctly once the directories are
collected in one place.

Usage:
  python scripts/audit_swamp_f4_failneg.py
  python scripts/audit_swamp_f4_failneg.py --runs-glob 'swamp_windy_f4_z*_s0'
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.15')

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402
from crl.report_maze import load_nets             # noqa: E402
from audit_swamp_z_failneg import critic_scores   # noqa: E402
import run_swamp_windy_z_failneg as L             # noqa: E402

OUT_DIR = 'artifacts/swamp_windy_f4_failneg'
RUNS_GLOB = 'swamp_windy_f4_z*_s*'
SWAMP_CELLS = ((3, 3), (4, 3), (5, 3))
MOUTH_CELL = (2, 3)                 # the holding cell in front of the corridor
NF = 4
GD = 2 * NF                         # goal / state width of the stack
VEL_FRACTIONS = (1.0, 0.75, 0.5, 0.25, 0.0)
FROZEN_TOL = 0.0                    # the freeze is exact; no tolerance needed


def dist(v):
  v = np.asarray(v, np.float64)
  if v.size == 0:
    return {'n': 0}
  return {'n': int(v.size), 'mean': float(v.mean()),
          'median': float(np.median(v)), 'std': float(v.std()),
          'p10': float(np.percentile(v, 10)),
          'p90': float(np.percentile(v, 90))}


def stack_velocity(stacks):
  """Mean per-frame displacement magnitude of [N, GD] stacks (newest first)."""
  f = stacks.reshape(len(stacks), NF, 2)
  return np.linalg.norm(f[:, :-1] - f[:, 1:], axis=2).mean(axis=1)


def freeze_stack(stacks):
  """Zero the velocity: every frame becomes the CURRENT position."""
  f = stacks.reshape(len(stacks), NF, 2)
  return np.repeat(f[:, :1], NF, axis=1).reshape(len(stacks), GD)


def scale_velocity(stacks, frac):
  """Shrink each stack's history toward the current position by ``frac``.

  frac=1 returns the stack unchanged, frac=0 returns the frozen stack, and the
  CURRENT position (frame 0) is fixed for every frac -- which is the whole
  point: only the velocity varies along this curve.
  """
  f = stacks.reshape(len(stacks), NF, 2)
  cur = f[:, :1]
  return (cur + frac * (f - cur)).reshape(len(stacks), GD)


def cells_of(xy):
  return np.clip(np.floor(xy).astype(int), [0, 0], [8, 4])


def in_swamp(xy):
  c = cells_of(xy)
  m = np.zeros(len(xy), bool)
  for cx, cy in SWAMP_CELLS:
    m |= (c[:, 0] == cx) & (c[:, 1] == cy)
  return m


def action_grid(n=21):
  """[K, 2] grid over the action box, partitioned into DISJOINT half-sets.

  Assignment is by dominant component, so no action can be the argmax of both
  sides. Actions with |a_x| == |a_y| belong to neither and are dropped, which
  is why the two masks are returned rather than inferred by the caller.
  """
  g = np.linspace(-1.0, 1.0, n)
  a = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
  shortcut = (a[:, 0] > np.abs(a[:, 1]))              # +x dominates -> corridor
  safe = (-a[:, 1] > np.abs(a[:, 0]))                 # -y dominates -> detour
  assert not np.any(shortcut & safe), 'action masks overlap'
  return a.astype(np.float32), shortcut, safe


def discover_runs(pattern):
  """[(label, run_dir, ckpt, alpha)] sorted by alpha, from the runs on disk."""
  found = []
  for d in sorted(glob.glob(pattern)):
    ckpt = os.path.join(d, 'final.pkl')
    prov = os.path.join(d, 'arm_provenance.json')
    if not os.path.exists(ckpt):
      continue
    alpha, bank = 0.0, None
    if os.path.exists(prov):
      with open(prov) as f:
        p = json.load(f)
      alpha, bank = float(p.get('alpha', 0.0)), p.get('bank')
    found.append({'run_dir': d, 'checkpoint': ckpt, 'alpha': alpha,
                  'bank': bank, 'label': 'alpha=%g' % alpha})
  found.sort(key=lambda r: r['alpha'])
  return found


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--out-dir', default=OUT_DIR)
  ap.add_argument('--runs-glob', default=RUNS_GLOB)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--n-anchors', type=int, default=20000)
  args = ap.parse_args()
  L.select_version('f4')

  def git(*a):
    try:
      return subprocess.check_output(
          ['git'] + list(a), cwd=os.path.dirname(_HERE)).decode().strip()
    except Exception:                              # pylint: disable=broad-except
      return ''

  runs = discover_runs(args.runs_glob)
  out = {'analysis_script': 'scripts/audit_swamp_f4_failneg.py',
         'env': L.ENV, 'dataset': L.DATASET, 'runs_glob': args.runs_glob,
         'code_commit': git('log', '-1', '--format=%H', '--', 'crl', 'scripts'),
         'head_at_runtime': git('rev-parse', 'HEAD'),
         'audited_files_dirty': bool(git('status', '--porcelain', '--',
                                         'crl', 'scripts'))}
  print('=' * 100)
  print('f4 FAILURE-AWARE CRL -- CRITIC SEMANTIC AUDIT')
  print('=' * 100)
  print('  env %s' % L.ENV)
  print('  code commit %s%s' % (out['code_commit'],
                                '  (DIRTY)' if out['audited_files_dirty']
                                else ''))
  if not runs:
    raise SystemExit('no runs matched %r (need <dir>/final.pkl)'
                     % args.runs_glob)
  print('  runs discovered: %s'
        % ', '.join('%s (%s)' % (r['run_dir'], r['label']) for r in runs))

  # ---------------------------------------------------------------- data
  with np.load(L.DATASET, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
    died = np.asarray(d['entered_active_swamp']).astype(bool)
  n_eps, Lr, W = obs.shape
  assert W == 2 * GD, 'expected obs width %d, got %d' % (2 * GD, W)
  st = obs[:, :, :GD]
  s_t = st[:, :-1, :].reshape(-1, GD)
  s_n = st[:, 1:, :].reshape(-1, GD)
  a_t = act[:, :-1, :].reshape(-1, 2)
  task_goal = obs[0, 0, GD:].astype(np.float32).copy()   # tile(GOAL, NF)
  assert np.abs(obs[:, :, GD:] - task_goal).max() == 0.0, (
      'the goal half is not one constant tiled goal')

  v_n = stack_velocity(s_n)
  frozen_n = v_n <= FROZEN_TOL
  lands_swamp = in_swamp(s_n[:, :2])
  # "Safe passage": lands in a swamp cell and is still MOVING -- the f4
  # counterpart of the z "entered and z stayed 0".
  safe_pass = lands_swamp & ~frozen_n
  fatal = lands_swamp & frozen_n
  print('  transitions %s | swamp-landing %s | moving %s | frozen %s'
        % (format(len(s_t), ','), format(int(lands_swamp.sum()), ','),
           format(int(safe_pass.sum()), ','), format(int(fatal.sum()), ',')))

  rng = np.random.default_rng(args.seed)
  i_pass = np.where(safe_pass)[0]
  if len(i_pass) > args.n_anchors:
    i_pass = rng.choice(i_pass, args.n_anchors, replace=False)
  i_fatal = np.where(fatal)[0]
  if len(i_fatal) > args.n_anchors:
    i_fatal = rng.choice(i_fatal, args.n_anchors, replace=False)

  bank_path = runs[-1]['bank'] or L.BANK
  with np.load(bank_path, allow_pickle=False) as b:
    bank = np.asarray(b['goals'], np.float32)
    bank_cls = (np.asarray(b['behaviour_class'])
                if 'behaviour_class' in b.files else None)
  print('  bank %s  %s  velocity max %.2e'
        % (bank_path, bank.shape, stack_velocity(bank).max()))

  # ------------------------------------------------------------ invariants
  cfg_env = Config(env_name=L.ENV)
  env = envs_mod.make_env(L.ENV, cfg_env, seed=0)
  env.reset(); env.set_auto_resample(False)
  env.set_swamp([1, 1, 1]); oa = env._get_obs().copy()
  env.set_swamp([0, 0, 0]); oc = env._get_obs().copy()
  inv = {'obs_width': int(oa.shape[0]),
         'hidden_bits_change_obs_by': float(np.abs(oa - oc).max()),
         'bank_all_frozen': bool(stack_velocity(bank).max() <= FROZEN_TOL),
         'n_frames': NF,
         'frac_rows_frozen': float(frozen_n.mean()),
         'frac_frozen_outside_swamp': float(
             (frozen_n & ~lands_swamp).mean()),
         'frozen_outside_swamp_note':
             'the first NF-1 rows of every episode look frozen because the '
             'stack starts filled with the reset position; that is a property '
             'of any frame stack and is reported, not corrected'}
  assert inv['hidden_bits_change_obs_by'] == 0.0, 'the hidden bits leak'
  assert inv['bank_all_frozen'], 'a bank entry is not a frozen stack'
  out['invariants'] = inv
  print('  invariants: obs width %d, hidden-bit leak %.1e, frozen rows %.4f '
        '(%.4f of them outside the swamp)'
        % (inv['obs_width'], inv['hidden_bits_change_obs_by'],
           inv['frac_rows_frozen'], inv['frac_frozen_outside_swamp']))

  # real moving stacks in the corridor, for the no-synthesis pairing (3)
  cm_sel = lands_swamp & ~frozen_n
  corridor_moving = s_n[cm_sel]
  cm_xy = corridor_moving[:, :2]
  # Episode id of every flattened transition, so section 3 can say whether a
  # pair came from the SAME episode. It usually will: a dying agent freezes
  # over NF-1 steps, so its own death step is a moving stack with an identical
  # frame 0. That is the tightest possible pair, but it is NOT "a survivor
  # walking through", and reporting one as the other would be a lie.
  ep_of = np.repeat(np.arange(n_eps), Lr - 1)
  cm_ep = ep_of[cm_sel]

  a_grid, m_short, m_safe = action_grid()
  out['fork'] = {'n_actions': int(len(a_grid)),
                 'n_shortcut_ward': int(m_short.sum()),
                 'n_safe_ward': int(m_safe.sum()),
                 'masks_disjoint': True}

  # mouth stacks: REAL observed stacks whose current position is in (2,3)
  c_t = cells_of(s_t[:, :2])
  at_mouth = (c_t[:, 0] == MOUTH_CELL[0]) & (c_t[:, 1] == MOUTH_CELL[1])
  i_mouth = np.where(at_mouth)[0]
  if len(i_mouth) > 512:
    i_mouth = rng.choice(i_mouth, 512, replace=False)
  mouth_states = s_t[i_mouth]
  print('  fork: %d disjoint shortcut-ward / %d safe-ward actions, %d real '
        'mouth stacks' % (m_short.sum(), m_safe.sum(), len(mouth_states)))

  # ---------------------------------------------------------------- models
  results = {}
  for r in runs:
    arm = 'zbase' if r['alpha'] == 0.0 else 'zfail'
    cfg = L.build_cfg(arm, '', steps=1, seed=args.seed)
    envs_mod.make_env(L.ENV, cfg, seed=0)
    # NB load_nets does not pass obs_scale; for f4 the scale IS None (every
    # coordinate is already in maze units), so the audited function is exactly
    # the trained one. This is NOT true for the z envs.
    nets, state, _, step = load_nets(L.ENV, r['checkpoint'], cfg)
    res = {'run_dir': r['run_dir'], 'checkpoint': r['checkpoint'],
           'step': int(step), 'alpha': r['alpha'], 'bank': r['bank']}

    S, A = s_t[i_pass], a_t[i_pass]

    # ---- 1 BANK RESPONSE: the one quantity that moved in the 2-D null ----
    g_bank = bank[rng.integers(0, len(bank), len(S))]
    f_bank = critic_scores(nets, state, S, A, g_bank)
    res['1_bank_response'] = {'f_bank': dist(f_bank)}
    if bank_cls is not None:
      idx = rng.integers(0, len(bank), len(S))
      for c in np.unique(bank_cls):
        m = bank_cls[idx] == c
        if m.any():
          res['1_bank_response']['f_bank_' + str(c)] = dist(f_bank[m])

    # ---- 2 PRIMARY: real moving stack vs its OWN velocity-zeroed twin ----
    g_move = s_n[i_pass]
    g_freeze = freeze_stack(g_move)
    f_move = critic_scores(nets, state, S, A, g_move)
    f_freeze = critic_scores(nets, state, S, A, g_freeze)
    m2 = f_move - f_freeze
    res['2_freeze_ranking'] = {'margin': dist(m2),
                               'frac_positive': float((m2 > 0).mean()),
                               'f_move': dist(f_move),
                               'f_freeze': dist(f_freeze)}

    # ---- 3 same pairing with NOTHING synthesised ----
    # For each fatal anchor, its real frozen goal, against a real moving stack
    # drawn from the corridor whose current position is nearest to it.
    S3, A3 = s_t[i_fatal], a_t[i_fatal]
    g_dead = s_n[i_fatal]
    # Exact nearest neighbour, chunked. NOT bucketed: a bucket miss would have
    # to fall back to some arbitrary row, and silently pairing a dead state
    # with a far-away live one is exactly the failure this section exists to
    # rule out. pair_xy_gap below is the audit of that pairing.
    dead_ep = ep_of[i_fatal]
    nn = np.empty(len(g_dead), np.int64)
    nn_x = np.empty(len(g_dead), np.int64)      # cross-episode partner
    for k0 in range(0, len(g_dead), 256):
      blk = g_dead[k0:k0 + 256, :2]
      d2 = ((blk[:, None, 0] - cm_xy[None, :, 0]) ** 2
            + (blk[:, None, 1] - cm_xy[None, :, 1]) ** 2)
      nn[k0:k0 + 256] = np.argmin(d2, axis=1)
      same = dead_ep[k0:k0 + 256, None] == cm_ep[None, :]
      nn_x[k0:k0 + 256] = np.argmin(np.where(same, np.inf, d2), axis=1)
    g_alive = corridor_moving[nn]
    g_alive_x = corridor_moving[nn_x]
    pair_gap = np.linalg.norm(g_alive[:, :2] - g_dead[:, :2], axis=1)
    pair_gap_x = np.linalg.norm(g_alive_x[:, :2] - g_dead[:, :2], axis=1)
    same_ep = cm_ep[nn] == dead_ep
    f_dead = critic_scores(nets, state, S3, A3, g_dead)
    f_alive = critic_scores(nets, state, S3, A3, g_alive)
    f_alive_x = critic_scores(nets, state, S3, A3, g_alive_x)
    m3 = f_alive - f_dead
    m3x = f_alive_x - f_dead
    res['3_real_pairing'] = {
        'margin': dist(m3), 'frac_positive': float((m3 > 0).mean()),
        'f_alive': dist(f_alive), 'f_dead': dist(f_dead),
        'pair_xy_gap': dist(pair_gap),
        'pair_same_episode_frac': float(same_ep.mean()),
        'same_episode_note':
            'a same-episode partner is the anchor own death step, which is '
            'still moving because the stack takes NF-1 steps to fill with '
            'repeats. Identical frame 0, so it is the tightest pair available '
            '-- but it is the same trajectory, not a survivor passing through. '
            'The cross_episode fields below are the survivor comparison.',
        'cross_episode': {'margin': dist(m3x),
                          'frac_positive': float((m3x > 0).mean()),
                          'f_alive': dist(f_alive_x),
                          'pair_xy_gap': dist(pair_gap_x)}}

    # ---- 4 goal families, centred (calibration) ----
    g_rand = st.reshape(-1, GD)[rng.integers(0, n_eps * Lr, len(S))]
    fam = {'factual_moving_future': f_move,
           'ordinary_random': critic_scores(nets, state, S, A, g_rand),
           'velocity_zeroed_twin': f_freeze,
           'failure_bank': f_bank,
           'task_goal': critic_scores(
               nets, state, S, A, np.tile(task_goal, (len(S), 1)))}
    mu = float(np.concatenate(list(fam.values())).mean())
    res['4_goal_families'] = {k: {'raw': dist(v), 'centred': dist(v - mu)}
                              for k, v in fam.items()}
    res['4_overall_mean_logit'] = mu

    # ---- 5 VELOCITY-response curve (the f4 analogue of z's depth curve) ----
    curve = {}
    for fr in VEL_FRACTIONS:
      gv = scale_velocity(g_move, fr)
      fv = critic_scores(nets, state, S, A, gv)
      curve['%.2f' % fr] = {'f': dist(fv), 'margin_vs_full': dist(f_move - fv),
                            'mean_stack_velocity':
                                float(stack_velocity(gv).mean())}
    res['5_velocity_curve'] = curve

    # ---- 6 fork margin at the mouth, DISJOINT action masks ----
    fm = []
    for s0 in mouth_states:
      ss = np.tile(s0, (len(a_grid), 1))
      gg = np.tile(task_goal, (len(a_grid), 1))
      f = critic_scores(nets, state, ss, a_grid, gg)
      fm.append(f[m_safe].max() - f[m_short].max())
    fm = np.asarray(fm)
    res['6_fork_margin'] = {'margin_safe_minus_shortcut': dist(fm),
                            'frac_preferring_safe': float((fm > 0).mean())}

    # ---- 7 positives: how often the relabeler hands over a DEAD goal ----
    from crl.offline_audit import build_offline_buffer
    buf, _ = build_offline_buffer(L.DATASET, cfg)
    pv, pf = [], []
    for _ in range(60):
      tr = buf.sample(cfg.batch_size)
      gg = tr.observation[:, GD:]
      pv.append(stack_velocity(gg))
      pf.append(critic_scores(nets, state, tr.observation[:, :GD], tr.action,
                              gg))
    pv, pf = np.concatenate(pv), np.concatenate(pf)
    dead_pos = pv <= FROZEN_TOL
    res['7_positive_goals'] = {
        'n_sampled': int(len(pv)),
        'frac_frozen_positive': float(dead_pos.mean()),
        'critic_frozen_positive': dist(pf[dead_pos]) if dead_pos.any()
                                  else {'n': 0},
        'critic_moving_positive': dist(pf[~dead_pos]),
        'note': 'a frozen positive is the relabeler handing the critic a dead '
                'state as a REACHABLE goal; there is no death mask, by design'}

    results[r['label']] = res
    print('\n  %s (step %d, %s)' % (r['label'], step, r['run_dir']))
    print('    1 f(bank)        mean %+.4f' % f_bank.mean())
    print('    2 freeze margin  mean %+.4f  frac>0 %.4f'
          % (m2.mean(), (m2 > 0).mean()))
    print('    3 real pairing   mean %+.4f  frac>0 %.4f  (xy gap %.3f, '
          'same-episode %.3f)'
          % (m3.mean(), (m3 > 0).mean(), pair_gap.mean(), same_ep.mean()))
    print('      cross-episode  mean %+.4f  frac>0 %.4f  (xy gap %.3f)'
          % (m3x.mean(), (m3x > 0).mean(), pair_gap_x.mean()))
    print('    6 fork margin    mean %+.4f  frac safe %.4f'
          % (fm.mean(), (fm > 0).mean()))
  out['runs'] = results

  # ---------------------------------------------------------------- tables
  ks = sorted(results, key=lambda k: results[k]['alpha'])
  print('\n' + '=' * 100)
  print('ALPHA RESPONSE  (the 2-D null moved f(bank) and nothing else --')
  print('                 a real effect needs columns 2/3/6 to move too)')
  print('=' * 100)
  hdr = ('%-12s%12s%12s%10s%12s%10s%12s%10s'
         % ('alpha', 'f(bank)', 'freeze m', 'frac>0', 'realpair m', 'frac>0',
            'fork m', 'frac safe'))
  print(hdr)
  print('-' * len(hdr))
  for k in ks:
    v = results[k]
    print('%-12s%12.4f%12.4f%10.4f%12.4f%10.4f%12.4f%10.4f'
          % (k, v['1_bank_response']['f_bank']['mean'],
             v['2_freeze_ranking']['margin']['mean'],
             v['2_freeze_ranking']['frac_positive'],
             v['3_real_pairing']['margin']['mean'],
             v['3_real_pairing']['frac_positive'],
             v['6_fork_margin']['margin_safe_minus_shortcut']['mean'],
             v['6_fork_margin']['frac_preferring_safe']))
  if len(ks) >= 2:
    base, top = results[ks[0]], results[ks[-1]]
    fb = (top['1_bank_response']['f_bank']['mean']
          - base['1_bank_response']['f_bank']['mean'])
    seq = [results[k]['1_bank_response']['f_bank']['mean'] for k in ks]
    mono = all(b <= a + 1e-9 for a, b in zip(seq, seq[1:]))
    out['f_bank_delta_first_to_last'] = float(fb)
    out['f_bank_monotone_decreasing'] = bool(mono)
    print('\n  f(bank) %s -> %s : %+.4f   monotone decreasing: %s'
          % (ks[0], ks[-1], fb, mono))
    print('  READ-OUT: f(bank) moving while columns 2/3/6 do not is the '
          'signature\n  of a bank that changes only the number it targets. '
          'That is what the 2-D\n  sweep measured; it is the null, not a '
          'result.')

  print('\n' + '=' * 100)
  print('VELOCITY-RESPONSE CURVE  f(stack with velocity scaled to frac)')
  print('=' * 100)
  hdr = '%-12s' % 'alpha' + ''.join('%11s' % ('x%.2f' % f)
                                    for f in VEL_FRACTIONS)
  print(hdr)
  print('-' * len(hdr))
  for k in ks:
    row = results[k]['5_velocity_curve']
    print('%-12s' % k + ''.join('%11.4f' % row['%.2f' % f]['f']['mean']
                                for f in VEL_FRACTIONS))

  print('\n' + '=' * 100)
  print('POSITIVE-GOAL CONTAMINATION  (no death mask on the relabeler)')
  print('=' * 100)
  for k in ks:
    v = results[k]['7_positive_goals']
    print('  %-12s frozen positives %.4f   f(frozen) %+.4f   f(moving) %+.4f'
          % (k, v['frac_frozen_positive'],
             v['critic_frozen_positive'].get('mean', float('nan')),
             v['critic_moving_positive']['mean']))

  os.makedirs(args.out_dir, exist_ok=True)
  path = os.path.join(args.out_dir, 'critic_audit_f4.json')
  with open(path, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % path)
  return 0


if __name__ == '__main__':
  sys.exit(main())
