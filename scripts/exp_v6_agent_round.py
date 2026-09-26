"""AntMaze V6: one agent-update round with policy-continuation branch futures.

Round 1 (the only round here): fix the current agent -> generate futures ->
train the critic -> update the agent.  A second round would regenerate with
the updated agent; it is not part of this script's run.

  fixed agent   the d20 vanilla actor, seed 0 (chosen by the standing "seed 0"
                rule before this experiment; its hash is sealed)
  futures       the control replay's anchors, query torques and hazard-draw
                seeds verbatim (dataset d20, recorded + 2 samples from the d20
                BC walker), continuation from step 2 by the fixed agent's mode;
                oracle transitions (a diagnostic upper bound for a learned
                ETT); successes and failures kept; NCE positives from the
                generated futures
  update        3 NCE critics on the new replay (30k, uniform anchors) ->
                3 actors from the fixed agent (frozen critic, bc 0.05, BC rows
                from the same d20 data, 30k)
  control       the same initialisation, budget and data with the old
                BC-continuation replay's critics (critics_br_d20); the only
                difference is the continuation policy behind the futures
  reference     the same initialisation and budget with the recorded-data
                critics (critics_van_d20): what one more actor round from the
                same start buys without any generated futures

Checks (pre-registered in the sealed manifest):
  1. critic, development set Cdev (start states of the old held-out episodes,
     turning and north-leg states of the Cnew episodes; 5 candidates, 16
     paired draws): labels under the fixed agent's continuation for every
     critic set ("who guides the current agent better"), and labels under the
     BC continuation for the control ("did it learn its own target");
     region readout, deployed min, paired episode bootstrap;
  2. agent: the fixed agent and every updated actor evaluated on the same
     fresh development draw (300 episodes, seed 2909); success is the primary
     endpoint, detour rate explanatory, walking must be kept;
  3. beyond the control: the updated actors against the control actors.

  python scripts/exp_v6_agent_round.py seal --agent-ckpt ... --bc-ckpt ...   (V6_DATASET_STEM = the merged pool)
  python scripts/exp_v6_agent_round.py verify_replay
  python scripts/exp_v6_agent_round.py analyze --agent-ckpt ...
  python scripts/exp_v6_agent_round.py report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import build_v6_branch_replay as B  # noqa: E402
import build_v6_policy_replay as R  # noqa: E402

EXP = 'exp_agent_round'
DIAG = 'diag_round1'            # labels under the fixed agent's continuation
DIAG_BC = 'diag_round1_bc'      # the same anchors / candidates / draws, labels under the BC continuation
ROUND_TAG = 'round1'
N_EVAL_PER_STRATUM, N_DRAWS = 32, 16
CAND_SEED_BC = 121_000_007      # BC samples at the Cdev anchors: base + 7919 * anchor_id
CAND_SEED_AG = 122_000_009      # agent samples
DRAW_SEED = 121_500_000         # Cdev draws: base + 100 * index + draw
AID0 = 4_000_000
EVAL_SEED = 2909
CRITIC_SETS = (('round1', 'critics_round1', f'replay_policy_{ROUND_TAG}.npz', 'row0'),
               ('control', 'critics_br_d20', 'replay_policy_d20.npz', 'row0'),
               ('vanilla', 'critics_van_d20', 'critic_stub_d20.npz', 'row0'),          # the first readout (row-0 marginal): kept for comparison
               ('vanilla_u', 'critics_van_d20', 'critic_stub_d20.npz', 'vanilla_draw'),  # corrected: the vanilla critic's own goal marginal (episode uniform, row uniform within it)
               ('ext_ag', 'critics_ext_ag', 'replay_policy_ext_ag.npz', 'row0'),
               ('ext_bc', 'critics_ext_bc', 'replay_policy_ext_bc.npz', 'row0'))
DIAG_T0, DIAG_T0_BC = 'diag_t0', 'diag_t0_bc'   # the true t = 0 anchors (reset rows of the old held-out episodes)
N_T0 = 64
DRAW_SEED_T0 = 124_500_000     # t0 draws: base + 100 * index + draw
AID0_T0 = 5_000_000
ACTOR_SETS = (('start (fixed agent)', 'joint_van_d20', (0,)),
              ('round1 (old queries, agent continuation)', 'joint_round1', (0, 1, 2)),
              ('control (old queries, BC continuation)', 'joint_ctrl_round1', (0, 1, 2)),
              ('reference (recorded-data critics)', 'joint_vanref_round1', (0, 1, 2)),
              ('ext_ag (extended queries, agent continuation)', 'joint_ext_ag', (0, 1, 2)),
              ('ext_bc (extended queries, BC continuation)', 'joint_ext_bc', (0, 1, 2)))


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def git_info():
  try:
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    st = subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(ROOT), text=True).splitlines()
    return {'head': head, 'dirty': [l for l in st if not l.startswith('??')], 'untracked_count': sum(l.startswith('??') for l in st)}
  except Exception as e:  # noqa
    return {'error': str(e)}


def exp_dir():
  return B.OUT / EXP


# -------------------------------------------------------------------- seal
def seal(args):
  import exp_v6_episode_coverage as EC
  out = exp_dir(); out.mkdir(parents=True, exist_ok=True)
  if (out / 'manifest.json').exists() and not args.force:
    raise SystemExit('manifest exists (sealed)')
  assert 'plus5k' in str(B.DATASET), f'V6_DATASET_STEM must be the merged pool, got {B.DATASET}'
  obs, act, lengths, meta, route = R.load_dataset()
  old_held = sorted(json.loads(str(np.load(B.OUT / 'holdout_policy_r1.npz', allow_pickle=False)['meta']))['held_out_episode_ids'])
  cnew_man = json.loads((B.OUT / 'diag_cnew' / 'manifest.json').read_text(encoding='utf-8'))
  rng = np.random.default_rng(args.seed)
  # --- anchors: start states of the old held-out episodes (round-robin), the Cnew turning / north-leg anchors verbatim
  masks = R.dense_masks(obs, lengths, route, np.isin(np.arange(len(route)), old_held))
  rows = {}
  for e, t in zip(*np.nonzero(masks['start_early'])):
    rows.setdefault(int(e), []).append(int(t))
  anchors = [(e, t, 'start_early') for e, t in EC.R_select_round_robin(rows, N_EVAL_PER_STRATUM, rng)]
  anchors += [(int(a['episode']), int(a['t']), str(a['stratum'])) for a in cnew_man['anchors']]
  recs = _anchor_records(args, obs, act, anchors, AID0, DRAW_SEED)
  _seal_write(args, out, recs, 'manifest.json', DIAG, DIAG_BC, {'start_early': 'old held-out episodes: development set of every earlier diagnostic (start strata)',
                                                              'turn': 'Cnew episodes: development set since the episode-coverage experiment', 'north_leg': 'Cnew episodes: development set since the episode-coverage experiment'})


def _anchor_records(args, obs, act, anchors, aid0, draw_seed):
  # --- candidates: recorded, the agent's mode, two BC-walker samples (the replay's query distribution), one agent sample
  b_bc = R.policy_bundle(args.bc_ckpt); b_ag = R.policy_bundle(args.agent_ckpt)
  o0 = np.stack([obs[e, t] for e, t, _ in anchors])
  loc_b, sc_b = b_bc['params_fn'](o0); loc_a, sc_a = b_ag['params_fn'](o0)
  recs = []
  for idx, (e, t, s) in enumerate(anchors):
    aid = aid0 + idx
    c_bc = dict(R.candidates_for(loc_b[idx], sc_b[idx], act[e, t], aid, ['recorded', 'sample0', 'sample1'], CAND_SEED_BC))
    c_ag = dict(R.candidates_for(loc_a[idx], sc_a[idx], act[e, t], aid, ['mode', 'sample0'], CAND_SEED_AG))
    cands = {'C|recorded': c_bc['recorded'], 'C|mode': c_ag['mode'], 'C|sample0': c_bc['sample0'], 'C|sample1': c_bc['sample1'], 'C|sample2': c_ag['sample0']}
    recs.append({'index': idx, 'layer': 'C', 'anchor_id': aid, 'episode': e, 't': t, 'stratum': s,
                 'obs': o0[idx].astype(float).tolist(), 'goal_xy': o0[idx][B.STATE_DIM:B.STATE_DIM + 2].astype(float).tolist(),
                 'policy_loc': loc_a[idx].astype(float).tolist(), 'policy_scale': sc_a[idx].astype(float).tolist(), 'policy_mode': np.tanh(loc_a[idx]).astype(float).tolist(),
                 'bc_loc': loc_b[idx].astype(float).tolist(), 'bc_scale': sc_b[idx].astype(float).tolist(),
                 'candidates': {k: v.astype(float).tolist() for k, v in cands.items()}, 'original_outcomes': {},
                 'candidate_labels': {'recorded': 'the dataset torque', 'mode': "the fixed agent's mode", 'sample0': 'BC-walker sample', 'sample1': 'BC-walker sample', 'sample2': 'fixed-agent sample'},
                 'previously_examined': s, 'draw_seeds': [draw_seed + 100 * idx + r for r in range(N_DRAWS)]})
  return recs


def _seal_write(args, out, recs, man_name, diag, diag_bc, examined):
  for r in recs:
    r['previously_examined'] = examined.get(r['stratum'], r['stratum'])
  # seed-stream separation (candidate bases mod the anchor stride; draw ranges disjoint by construction)
  bases = {'r1': R.R1_SEED + 11, 'abc': 909_707_000 + 17, 'cov': 515_000_003, 'fresh': 616_000_005, 'broad': 818_000_003, 'cnew': 919_000_005, 'round_bc': CAND_SEED_BC, 'round_ag': CAND_SEED_AG}
  for a in bases:
    for b in bases:
      if a < b:
        assert (bases[a] - bases[b]) % 7919 != 0, (a, b)
  assets = {'agent_ckpt': str(args.agent_ckpt), 'agent_sha256': sha256(args.agent_ckpt), 'bc_ckpt': str(args.bc_ckpt), 'bc_sha256': sha256(args.bc_ckpt),
            'control_replay': 'replay_policy_d20.npz', 'control_replay_sha256': sha256(B.OUT / 'replay_policy_d20.npz'),
            'control_critics': {s: sha256(B.OUT / 'critics_br_d20' / f'seed_{s}' / 'final.pkl') for s in (0, 1, 2)},
            'vanilla_critics': {s: sha256(B.OUT / 'critics_van_d20' / f'seed_{s}' / 'final.pkl') for s in (0, 1, 2)},
            'dataset_d20': str(B.OUT / 'critic_stub_d20.npz'), 'dataset_d20_sha256': sha256(B.OUT / 'critic_stub_d20.npz'),
            'merged_pool': str(B.DATASET), 'merged_pool_sha256': sha256(B.DATASET),
            'scripts': {f: sha256(ROOT / f) for f in ('scripts/exp_v6_agent_round.py', 'scripts/build_v6_policy_replay.py', 'scripts/run_v6_branch_replay.py', 'scripts/diag_v6_r1_abc.py', 'scripts/probe_v6_coverage_paired_boot.py')}}
  man = {'experiment': 'agent-update round 1: policy-continuation futures from the fixed d20 vanilla actor (oracle transitions) vs the BC-continuation control',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git': git_info(), 'assets': assets,
         'fixed_agent': {'choice_rule': 'joint_van_d20/seed_0 -- the standing "seed 0" default, fixed before this experiment; not selected on any result (it is the weakest of the three d20 vanilla actors on the 909 draw)',
                         'historical_reference_909': {'success': 0.543, 'detour': 0.507, 'timeout': 0.197, 'no_hazard_success': 0.740}},
         'round1_replay': {'build': f'build_v6_policy_replay build --cont-ckpt <agent> --query-ckpt <bc> --tag {ROUND_TAG} --holdout-frac 0 --n-holdout 0 on dataset d20 (seed 0): '
                                    'the control replay\'s anchors, candidate torques and hazard-draw seeds verbatim (verified by verify_replay); continuation = the agent\'s mode from step 2; oracle transitions',
                           'diagnostic_upper_bound': 'oracle transition model; a learned ETT replaces it later; the nominal policy supplies the ETT expert advice and is not the continuation agent'},
         'training': {'critics': f'critics_{ROUND_TAG}: run_v6_branch_replay critics on the round-1 replay, 30k, seeds 0-2, uniform anchors (row 0) -- identical to critics_br_d20\'s recipe',
                      'actors': 'joint_round1 / joint_ctrl_round1 / joint_vanref_round1: actor initialised from the fixed agent (V6_JOINT_ACTOR_INIT), frozen critic, original actor loss, bc 0.05, '
                                'balanced BC rows from the d20 dataset, 30k, seeds 0-2; critic batches from the critic\'s own training data (round-1 replay / control replay / dataset)'},
         'cdev': {'anchors': len(recs), 'strata': {s: sum(r['stratum'] == s for r in recs) for s in sorted({r['stratum'] for r in recs})},
                  'episodes': {s: len({r['episode'] for r in recs if r['stratum'] == s}) for s in sorted({r['stratum'] for r in recs})},
                  'anchor_times': sorted({int(r['t']) for r in recs})[:8],
                  'candidates': 'recorded, agent mode, 2 BC samples, 1 agent sample; 16 paired draws; labels = P_goal (gamma 0.999, radius 0.5) under the continuation of the diag dir',
                  'status': 'DEVELOPMENT set (the old held-out episodes and Cnew have been analysed before); a positive result is confirmed on the reserved seeds afterwards, not here',
                  'label_sets': {diag: 'continuation = the fixed agent (every critic set is read against it: who guides the current agent better)',
                                 diag_bc: 'continuation = the d20 BC walker (the control critic\'s own law; the vanilla critics\' law is neither)'}},
         'evaluation': {'draw': f'300 natural episodes, seed {EVAL_SEED}, mean policy, p_active 0.5 -- the fixed agent and every actor on the same draw (V6_EVAL_SEED)',
                        'primary': 'success rate', 'explanatory': 'detour rate, hazard / no-hazard success, timeouts, zone deaths', 'walking': 'timeout rate <= 0.25 and no-hazard success >= 0.70 (the fixed agent: 0.197 / 0.740 on the 909 draw)'},
         'decision_rules': {'effect': 'mean over 3 seeds differs by > 2 x pooled seed s.e. with 3 / 3 seeds in the same direction; for the fixed agent (one policy) the paired comparison uses its value on the same draw',
                            'critic': f'round1 pick gain on {DIAG} above control by > 2 paired episode-bootstrap s.e., positive, 3 / 3 seeds; control on {DIAG_BC} reported separately (its own target)',
                            'agent': 'success of the round-1 actors above the fixed agent AND above the control actors (effect rule), while walking is kept; detour rate explains whether a gain is route choice',
                            'continue': 'all three met -> generate round 2 from the updated agent; otherwise stop and analyse (no mechanical iteration)'},
         'seeds': {'cdev_candidate_bases': [CAND_SEED_BC, CAND_SEED_AG], 'cdev_draw_rule': f'{DRAW_SEED} + 100 * index + draw', 'eval_seed': EVAL_SEED, 'selection_seed': args.seed,
                   'reserved_untouched': 'FRESH_TORQUE_SEED 616_000_005 / FRESH_DRAW_SEED 616_500_000 and the reserved evaluation seeds stay unused'},
         'diag_manifest': {'diagnostic': 'Cdev: start states of the old held-out episodes + the Cnew turning / north-leg anchors; labels under a stated continuation',
                           'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'continuation': {'mode': 'frozen policy, tanh(loc) mode, closed-loop from step 2', 'horizon': B.HORIZON},
                           'law': {'discount': R.GAMMA, 'radius': 0.5}, 'anchors': recs}}
  (out / man_name).write_text(json.dumps(man, indent=1, default=float), encoding='utf-8')
  for dg, ck, sh in ((diag, args.agent_ckpt, assets['agent_sha256']), (diag_bc, args.bc_ckpt, assets['bc_sha256'])):
    d = dict(man['diag_manifest']); d['provenance'] = {'reference_commit': man['git'].get('head', ''), 'continuation_ckpt': str(ck), 'continuation_sha256': sh, 'critics': {}}
    (B.OUT / dg).mkdir(parents=True, exist_ok=True)
    (B.OUT / dg / 'manifest.json').write_text(json.dumps(d, indent=1, default=float), encoding='utf-8')
  print(json.dumps({'cdev': man['cdev'], 'assets': {k: v for k, v in assets.items() if 'sha' in k and isinstance(v, str)}}, indent=1), flush=True)


def seal_t0(args):
  """The true t = 0 diagnostic: reset rows of the old held-out episodes (one anchor per episode), the same
  candidate types, draws and label sets as Cdev.  The earlier Cdev (start rows t <= 5, momentum-committed) is kept."""
  out = exp_dir()
  if (out / 'manifest_t0.json').exists() and not args.force:
    raise SystemExit('manifest_t0 exists (sealed)')
  assert 'plus5k' in str(B.DATASET), f'V6_DATASET_STEM must be the merged pool, got {B.DATASET}'
  obs, act, lengths, meta, route = R.load_dataset()
  old_held = sorted(json.loads(str(np.load(B.OUT / 'holdout_policy_r1.npz', allow_pickle=False)['meta']))['held_out_episode_ids'])
  rng = np.random.default_rng(args.seed + 1)
  eps = sorted(rng.choice(old_held, size=min(N_T0, len(old_held)), replace=False).tolist())
  anchors = [(int(e), 0, 'start_early') for e in eps]
  recs = _anchor_records(args, obs, act, anchors, AID0_T0, DRAW_SEED_T0)
  _seal_write(args, out, recs, 'manifest_t0.json', DIAG_T0, DIAG_T0_BC, {'start_early': 'old held-out episodes, reset row (t = 0) -- not examined at t = 0 before; the Cdev start anchors were t <= 5'})


# ----------------------------------------------------------- verify_replay
def verify_replay(args):
  out = exp_dir()
  new, ctl = B.OUT / f'replay_policy_{ROUND_TAG}.npz', B.OUT / 'replay_policy_d20.npz'
  res = {'round1': str(new), 'control': str(ctl)}
  with np.load(new, allow_pickle=False) as a, np.load(ctl, allow_pickle=False) as c:
    for k in ('audit_anchor_id', 'audit_episode', 'audit_anchor_time', 'audit_kind', 'audit_cand', 'audit_draw'):
      res[f'same_{k}'] = bool(a[k].shape == c[k].shape and np.array_equal(a[k], c[k]))
    res['same_row0_obs'] = bool(np.array_equal(a['obs'][:, 0], c['obs'][:, 0]))
    res['same_query_torque'] = bool(np.array_equal(a['act'][:, 0], c['act'][:, 0]))
    ma, mc = json.loads(str(a['meta'])), json.loads(str(c['meta']))
    res['continuation'] = {'round1': ma.get('continuation_ckpt'), 'control': mc.get('continuation_ckpt')}
    res['query'] = {'round1': ma.get('query_ckpt'), 'control': mc.get('query_ckpt', mc.get('continuation_ckpt'))}
    res['paths'] = {'round1': int(a['lengths'].shape[0]), 'control': int(c['lengths'].shape[0])}
    kind = a['audit_kind'].astype(str); cand = a['audit_cand'].astype(str)
    prof = {}
    for s in np.unique(kind):
      for cd in np.unique(cand[kind == s]):
        m = (kind == s) & (cand == cd); mc_ = m
        prof[f'{s}|{cd}'] = {'n': int(m.sum()),
                             'round1': {'reach': float(a['audit_success'][m].mean()), 'death': float(a['audit_failure'][m].mean()), 'p_goal': float(a['audit_p_goal'][m].mean()), 'around': float((a['audit_max_y'][m] >= B.DETOUR_Y).mean())},
                             'control': {'reach': float(c['audit_success'][mc_].mean()), 'death': float(c['audit_failure'][mc_].mean()), 'p_goal': float(c['audit_p_goal'][mc_].mean()), 'around': float((c['audit_max_y'][mc_] >= B.DETOUR_Y).mean())}}
    res['profiles'] = prof
  res['identical_inputs'] = all(v for k, v in res.items() if k.startswith('same_'))
  res['round1_replay_sha256'] = sha256(new)
  (out / 'replay_check.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
  print(json.dumps({k: v for k, v in res.items() if k != 'profiles'}, indent=1), flush=True)
  for k, v in prof.items():
    print(f'{k:28s} n {v["n"]:5d}  round1 reach {v["round1"]["reach"]:.2f} death {v["round1"]["death"]:.2f} P {v["round1"]["p_goal"]:.3f} around {v["round1"]["around"]:.2f} | '
          f'control reach {v["control"]["reach"]:.2f} death {v["control"]["death"]:.2f} P {v["control"]["p_goal"]:.3f} around {v["control"]["around"]:.2f}', flush=True)
  if not res['identical_inputs']:
    raise SystemExit('round-1 replay inputs differ from the control replay')


# ------------------------------------------------------------- verify_ext
def verify_ext(args):
  """The extended replays (tag ext_ag / ext_bc): their recorded / sample0 / sample1 paths must be the control
  replay's paths (same anchors, torques, draw seeds); the ag_* paths are new; every anchor gains the same
  candidates, so the per-stratum path masses under uniform anchors equal the control's."""
  out = exp_dir(); ctl = B.OUT / 'replay_policy_d20.npz'
  res = {}
  with np.load(ctl, allow_pickle=False) as c:
    ckey = {(int(a), str(k), int(d)): i for i, (a, k, d) in enumerate(zip(c['audit_anchor_id'], c['audit_cand'].astype(str), c['audit_draw']))}
    c_obs0 = c['obs'][:, 0]; c_act0 = c['act'][:, 0]; c_kind = c['audit_kind'].astype(str)
    c_mass = {k: float(np.mean(c_kind == k)) for k in np.unique(c_kind)}
  for tag in ('ext_ag', 'ext_bc'):
    path = B.OUT / f'replay_policy_{tag}.npz'
    if not path.exists():
      continue
    with np.load(path, allow_pickle=False) as a:
      aid = a['audit_anchor_id']; cand = a['audit_cand'].astype(str); draw = a['audit_draw']; kind = a['audit_kind'].astype(str)
      obs0 = a['obs'][:, 0]; act0 = a['act'][:, 0]
      old = np.array([not k.startswith('ag_') for k in cand])
      same_obs = same_act = 0; missing = 0
      for i in np.flatnonzero(old):
        j = ckey.get((int(aid[i]), cand[i], int(draw[i])))
        if j is None:
          missing += 1; continue
        same_obs += int(np.array_equal(obs0[i], c_obs0[j])); same_act += int(np.array_equal(act0[i], c_act0[j]))
      n_old = int(old.sum())
      mass = {k: float(np.mean(kind == k)) for k in np.unique(kind)}
      meta = json.loads(str(a['meta']))
      prof = {}
      for sset in np.unique(kind):
        for cd in np.unique(cand[kind == sset]):
          m = (kind == sset) & (cand == cd)
          prof[f'{sset}|{cd}'] = {'n': int(m.sum()), 'reach': float(a['audit_success'][m].mean()), 'death': float(a['audit_failure'][m].mean()),
                                  'p_goal': float(a['audit_p_goal'][m].mean()), 'around': float((a['audit_max_y'][m] >= B.DETOUR_Y).mean())}
      res[tag] = {'paths': int(len(aid)), 'old_paths': n_old, 'old_paths_matched_in_control': n_old - missing, 'old_same_obs0': same_obs, 'old_same_act0': same_act,
                  'new_paths': int((~old).sum()), 'control_paths': int(len(ckey)), 'identical_old_inputs': (missing == 0 and same_obs == n_old and same_act == n_old and n_old == len(ckey)),
                  'stratum_mass': mass, 'control_stratum_mass': c_mass, 'stratum_mass_equal': all(abs(mass.get(k, 0) - c_mass[k]) < 1e-9 for k in c_mass),
                  'continuation': meta.get('continuation_ckpt'), 'query': meta.get('query_ckpt'), 'extra_query': meta.get('extra_query_ckpt'), 'sha256': sha256(path), 'profiles': prof}
      print(tag, {k: v for k, v in res[tag].items() if k not in ('profiles',)}, flush=True)
      for k, v in prof.items():
        print(f'  {k:28s} n {v["n"]:5d} reach {v["reach"]:.2f} death {v["death"]:.2f} P {v["p_goal"]:.3f} around {v["around"]:.2f}', flush=True)
  (out / 'replay_check_ext.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
  bad = [t for t, r in res.items() if not (r['identical_old_inputs'] and r['stratum_mass_equal'])]
  if bad:
    raise SystemExit(f'extended replays differ from the control on the shared paths or masses: {bad}')


# ----------------------------------------------------------------- analyze
def analyze(args):
  env = {**os.environ, 'JAX_PLATFORMS': os.environ.get('JAX_PLATFORMS', 'cpu')}
  sets = [c for c in CRITIC_SETS if (B.OUT / c[1] / 'seed_0' / 'final.pkl').exists() and (B.OUT / c[2]).exists()]
  if args.sets:
    sets = [c for c in sets if c[0] in args.sets.split(',')]
  print('critic sets present:', [c[0] for c in sets], flush=True)
  diags = {'round1': (DIAG, DIAG_BC), 't0': (DIAG_T0, DIAG_T0_BC)}[args.diag]
  for diag in diags:
    for name, cdir, marg, law in sets:
      critics = [str(B.OUT / cdir / f'seed_{s}' / 'final.pkl') for s in (0, 1, 2)]
      cmd = [sys.executable, str(ROOT / 'scripts' / 'diag_v6_r1_abc.py'), 'analyze', '--cont-ckpt', str(args.agent_ckpt), '--critics', *critics, '--allow-unsealed-critics',
             '--marginal-replay', marg, '--marginal-law', law, '--tag', name, '--out', diag]
      print(' '.join(cmd), flush=True); subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    cmd = [sys.executable, str(ROOT / 'scripts' / 'probe_v6_coverage_paired_boot.py'), '--cont-ckpt', str(args.agent_ckpt),
           '--runs', *[f'{n}={d}' for n, d, _, _ in sets], '--marginals', *[f'{n}={m}@{law}' for n, _, m, law in sets],
           '--layer', 'C', '--diag-dir', diag, '--out', f'{diag}/paired_boot_C.json']
    print(' '.join(cmd), flush=True); subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


# ------------------------------------------------------------------ report
def _eval(path):
  import run_v6_branch_replay as D
  ev = json.loads(Path(path).read_text(encoding='utf-8'))
  h = D._headline(ev['summary'], ev.get('episodes')); s = ev['summary']; bl = s['by_latent']
  haz = [k for k in bl if k != 'U00']
  h['no_hazard_success'] = float(bl['U00']['success']); h['no_hazard_detours'] = int(bl['U00']['routes']['detour']['n'])
  h['hazard_success'] = float(sum(bl[k]['success_n'] for k in haz) / max(1, sum(bl[k]['n'] for k in haz)))
  h['hazard_detours'] = int(sum(bl[k]['routes']['detour']['n'] for k in haz))
  h['zone_deaths'] = (int(s['overall']['zone1_deaths']), int(s['overall']['zone2_deaths'])); h['mean_steps'] = float(s['overall']['mean_steps'])
  h['eval_seed'] = ev.get('eval_seed')
  return h


def report(args):
  out = exp_dir(); man = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
  f_ = lambda v, fmt='.3f': (format(v, fmt) if isinstance(v, (int, float)) and v is not None else '-')
  L = ['# Agent-update round 1: policy-continuation futures (fixed d20 vanilla actor) vs the BC-continuation control', '',
       f'Sealed {man["sealed_at"]}.  Oracle transitions (diagnostic upper bound).  Decision rules in `manifest.json`.', '']
  chk = out / 'replay_check.json'
  if chk.exists():
    c = json.loads(chk.read_text(encoding='utf-8'))
    L += ['## Round-1 replay vs the control replay', '', f'Identical anchors / query torques / draw seeds: **{c["identical_inputs"]}** ({c["paths"]["round1"]} paths).  Continuation: round1 = fixed agent, control = d20 BC walker.', '',
          '| stratum / candidate | n | round1 reach / death / P / around | control reach / death / P / around |', '|---|---:|---|---|']
    for k, v in c['profiles'].items():
      if k.split('|')[1] in ('recorded', 'sample0'):
        L.append(f'| {k} | {v["n"]} | {v["round1"]["reach"]:.2f} / {v["round1"]["death"]:.2f} / {v["round1"]["p_goal"]:.3f} / {v["round1"]["around"]:.2f} | '
                 f'{v["control"]["reach"]:.2f} / {v["control"]["death"]:.2f} / {v["control"]["p_goal"]:.3f} / {v["control"]["around"]:.2f} |')
    L.append('')
  # --- critic check
  for diag, title in ((DIAG, 'Cdev (start rows t <= 5 + Cnew turn / north-leg), labels under the FIXED AGENT continuation'), (DIAG_BC, 'Cdev, labels under the BC continuation (the control critic\'s own target)'),
                      (DIAG_T0, 'true t = 0 reset rows of the old held-out episodes, labels under the FIXED AGENT continuation'), (DIAG_T0_BC, 't = 0 rows, labels under the BC continuation')):
    if not (B.OUT / diag).exists():
      continue
    L += [f'## Critic check -- {title}', '']
    rows = []
    for name, cdir, _, _ in CRITIC_SETS:
      p = B.OUT / diag / f'metrics_{name}.json'
      if not p.exists():
        continue
      m = json.loads(p.read_text(encoding='utf-8'))['layers']['C']
      for s in (0, 1, 2):
        key = f'{cdir}/seed_{s} region min'
        if key in m:
          r = m[key]
          rows.append((name, s, r['pooled'], {st: m[key][st] for st in ('start_early', 'turn', 'north_leg')}))
    if rows:
      L += ['| critic set | seed | decided pairs | agreement (s.e.) | pick gain (s.e.) | start_early agreement / gain | turn | north_leg |', '|---|---|---:|---|---|---|---|---|']
      for name, s, r, st in rows:
        cell = lambda x: f'{f_(x["agreement_decided"], ".2f")} / {f_(x["pick_gain"], "+.3f")}'
        L.append(f'| {name} | {s} | {r["n_decided"]} | {f_(r["agreement_decided"])} ({f_(r["se_decided"])}) | {f_(r["pick_gain"], "+.3f")} ({f_(r["pick_gain_se"])}) | {cell(st["start_early"])} | {cell(st["turn"])} | {cell(st["north_leg"])} |')
      rnd = json.loads((B.OUT / diag / f'metrics_{rows[0][0]}.json').read_text(encoding='utf-8'))['layers']['C']['random']['pooled']
      L.append(f'| random | - | {rnd["n_decided"]} | {f_(rnd["agreement_decided"])} | {f_(rnd["pick_gain"], "+.3f")} | | | |')
      L.append(f'\nCross-fitted selector gain (signal in the outcomes themselves): {f_(rows[0][2].get("cross_fit_selector_gain"), "+.3f")}')
    pb = B.OUT / diag / 'paired_boot_C.json'
    if pb.exists():
      b = json.loads(pb.read_text(encoding='utf-8'))
      L += ['', f'Paired episode bootstrap ({b["n_decided_pairs"]} decided pairs, {b["n_episodes"]} episodes): pick gain seed-mean ' +
            ', '.join(f'{r} {b["pick_gain"][r]["seed_mean"]:+.3f} ({b["pick_gain"][r]["se"]:.3f})' for r in b['pick_gain']) + '; differences: ' +
            '; '.join(f'{k} {v["mean"]:+.3f} +- {v["se"]:.3f} (z {v["z"]:+.1f}; per seed ' + ' / '.join(f'{x:+.3f}' for x in v['per_seed'].values()) + ')' for k, v in b['pick_gain_differences'].items() if k.endswith('- control') or k.endswith('- round1'))]
    L.append('')
  # --- actors
  L += [f'## Agents on the same fresh development draw (300 episodes, seed {EVAL_SEED}, mean policy)', '',
        '| policy | seed | success | failure | timeout | detour | no-hazard success (detours) | hazard success (detours / n) | zone-1 / zone-2 deaths | mean steps |', '|---|---|---:|---:|---:|---:|---|---|---|---:|']
  agg = {}
  for label, d, seeds in ACTOR_SETS:
    for s in seeds:
      p = B.OUT / d / f'seed_{s}' / f'eval_mean_s{EVAL_SEED}.json'
      if not p.exists():
        continue
      h = _eval(p); agg.setdefault(label, []).append(h)
      L.append(f'| {label} | {s} | {h["success_rate"]:.3f} | {h["failure_rate"]:.3f} | {h["timeout_rate"]:.3f} | {h["detour_rate"]:.3f} | {h["no_hazard_success"]:.3f} ({h["no_hazard_detours"]}) | '
               f'{h["hazard_success"]:.3f} ({h["hazard_detours"]}) | {h["zone_deaths"][0]} / {h["zone_deaths"][1]} | {h["mean_steps"]:.0f} |')
  if agg:
    se = lambda v: (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    L += ['', '| policy | mean success (seed s.e.) | mean detour (seed s.e.) | mean timeout | mean no-hazard success |', '|---|---|---|---|---|']
    stats = {}
    for label, hs in agg.items():
      su = np.array([h['success_rate'] for h in hs]); de = np.array([h['detour_rate'] for h in hs]); to = np.array([h['timeout_rate'] for h in hs]); nh = np.array([h['no_hazard_success'] for h in hs])
      stats[label] = (su, de, to, nh)
      L.append(f'| {label} | {su.mean():.3f} ({se(su):.3f}) | {de.mean():.3f} ({se(de):.3f}) | {to.mean():.3f} | {nh.mean():.3f} |')
    L += ['', '## Pre-registered comparisons (success primary; detour explanatory)', '']
    def cmp(a, b):
      if a in stats and b in stats:
        for qi, qn in ((0, 'success'), (1, 'detour')):
          x, y = stats[a][qi], stats[b][qi]
          if len(y) > 1:
            pse = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / len(x)); d = x.mean() - y.mean(); same = int(np.sum(np.sign(x - y) == np.sign(d))) if d != 0 else 0
            L.append(f'- {a} minus {b}, {qn}: {d:+.3f} (pooled seed s.e. {pse:.3f}; {"> 2 s.e." if abs(d) > 2 * pse else "not > 2 s.e."}; same-direction seeds {same}/3)')
          else:   # a single fixed policy on the same draw
            d = x.mean() - y[0]; sse = se(x); same = int(np.sum(np.sign(x - y[0]) == np.sign(d))) if d != 0 else 0
            L.append(f'- {a} minus {b}, {qn}: {d:+.3f} (seed s.e. of {a} {sse:.3f}; {"> 2 s.e." if abs(d) > 2 * sse else "not > 2 s.e."}; same-direction seeds {same}/{len(x)})')
    names = [a[0] for a in ACTOR_SETS]
    for i in range(1, len(names)):
      cmp(names[i], names[0])
    for i in (1, 4, 5):
      cmp(names[i], names[2])
    cmp(names[4], names[1]); cmp(names[5], names[2]); cmp(names[4], names[5])
    L += ['', 'Walking kept (timeout <= 0.25 and no-hazard success >= 0.70), per seed: ' +
          '; '.join(f'{label}: ' + ' / '.join(('yes' if (h['timeout_rate'] <= 0.25 and h['no_hazard_success'] >= 0.70) else 'NO') for h in hs) for label, hs in agg.items())]
  (out / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'seal_t0', 'verify_replay', 'verify_ext', 'analyze', 'report'))
  ap.add_argument('--diag', default='round1', choices=('round1', 't0'), help='analyze: which diagnostic family (Cdev t <= 5 anchors, or the true t = 0 anchors)')
  ap.add_argument('--sets', default='', help='analyze: comma list of critic sets to (re)analyse (default all present)')
  ap.add_argument('--agent-ckpt', default=str(B.OUT / 'joint_van_d20' / 'seed_0' / 'final.pkl'))
  ap.add_argument('--bc-ckpt', default=str(B.OUT / 'joint_purebc_d20' / 'seed_0' / 'final.pkl'))
  ap.add_argument('--seed', type=int, default=2029)
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  {'seal': seal, 'seal_t0': seal_t0, 'verify_replay': verify_replay, 'verify_ext': verify_ext, 'analyze': analyze, 'report': report}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
