"""AntMaze V6 mainline, after the critic clip: do the current agents' futures
differ from the start agent's at the SAME training anchors, and does
switching the critic to them help?  (User's lead, 2026-09-19 evening.)

The user's frozen-model diagnostic on the clipped CF checkpoints found
that the same first torques sampled by the current policy enter the far
route 19 / 128 times when the CURRENT agent continues, 1 / 128 when the
START agent continues -- while the critic's positive futures still come
from the start agent (`branches_cf.npz`).  Two steps, in order:

  A. The check.  For each lineage s (the clipped CF final of seed s,
     `variants/critic_clip0.1/CF/seed_s/final.pkl`), regenerate the branch at
     EVERY training anchor with the same logged query torque and the same
     hazard seed (HAZARD_SEED0 + anchor_id; `generate_branches`), the lineage
     agent's mode continuing.  `profile` then compares, anchor by anchor
     (paired), the new futures with the start agent's: entered the far
     route (the env's label, y >= 6 at x < 2, after the query), completed
     it, entered a hazard zone, outcome, length, and the critic's
     discounted goal marginal (mass in the far-route legs / the goal area /
     the hazard zones under the truncated geometric law) -- anchor-weighted,
     by anchor stratum (reset rows, start region, every region, logged
     route).  This says whether the current agent's capability enters the
     TRAINING data at the logged query torques, not only the candidates of
     the user's test.

  B. The paired continuation (gated by A; gate pre-registered in the
     manifest, evaluated mechanically): from the SAME full checkpoint (the
     lineage final: params, target, both Adam states, key), +30,000 updates
     under the sealed clip recipe with the streams fast-forwarded by 30,000
     batches, arm CFold with the start-agent futures (the plain continuation
     of the run) vs arm CFnew with the lineage's own futures; evaluated on
     the development draw (seed 3909, mode).  Primary: CFnew - CFold success
     paired per episode, mean over the 3 lineages > 2 x seed s.e. and 3/3;
     also CFnew - source and CFold - source (the lineage agent itself).  No
     restart: the round-2 confound (learner restarted vs futures changed) is
     removed.  Round 2 ran before the clip, under the collapses, and is not
     evidence about this version.

Everything else -- NCE, actor objective, bc 0.05, gamma 0.999, anchors and
weights, actor stream, critic clip 0.1, evaluation -- is the sealed
critic_clip0.1 recipe.  Oracle (simulator) futures; the learned-ETT line is
separate (`exp_v6_learned_ett.py`).

  python scripts/exp_v6_clip_round.py seal
  python scripts/exp_v6_clip_round.py generate --lineage s --workers w     # JAX_PLATFORMS=cpu
  python scripts/exp_v6_clip_round.py profile                             # -> profile.json, PROFILE.md, gate
  python scripts/exp_v6_clip_round.py resume --lineage s --arm CFold|CFnew  # GPU; evaluates when done
  python scripts/exp_v6_clip_round.py report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import exp_v6_mainline_pilot as MP  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402  (maze regions; numpy only)

RECIPE = 'critic_clip0.1'
OUT = MP.OUT / 'clip_round'
ARMS = ('CFold', 'CFnew')
UPDATES = 30_000
GAMMA = MP.GAMMA
DETOUR_Y, DETOUR_MAX_X = 6.0, 2.0
ZONE_X = {1: (6.6, 9.4), 2: (14.6, 17.4)}
FAR = [REGIONS.index(n) for n in ('west_column', 'top_corridor', 'east_column')]
ZONES = [REGIONS.index(n) for n in ('zone1', 'zone2')]
GOAL = REGIONS.index('goal_area')
CONFIRM_SEED = 5909      # pre-registered (2026-09-19, before any B result): the B arms re-evaluated ONCE on a fresh draw if the primary rule is met on 3909
GATE = {'stratum': 'anchor in start region', 'quantity': 'far route completed (entered the far route after the query and the branch reached the goal), anchor-weighted share',
        'rule': 'new - old > 0 in 3/3 lineages AND the mean over the 3 lineages >= 0.02 (absolute); the start-agent share is the same file for every lineage'}


def lineage_ckpt(s):
  return MP.run_dir('CF', s, MP.variant_base(RECIPE)) / 'final.pkl'


def new_branch_path(s):
  return OUT / f'branches_cf_c_s{s}.npz'


OLD_BRANCHES = MP.OUT / 'branches_cf.npz'


# -------------------------------------------------------------------- seal
def mode_seal(args):
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / 'manifest.json'
  if p.exists() and not args.force:
    print(f'{p} exists', flush=True); return
  man = {'experiment': 'AntMaze V6 mainline, clip recipe: the current (clipped CF) agents as continuation at the same anchors -- futures profile (A), gated paired continuation (B)',
         'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'git_head': MP.git_head(), 'recipe': RECIPE, 'recipe_overrides': MP.VARIANTS[RECIPE],
         'recipe_manifest_sha256': MP.sha256(MP.variant_base(RECIPE) / 'manifest.json'), 'anchors_sha256': MP.sha256(MP.OUT / 'anchors.npz'),
         'old_branches': {'path': str(OLD_BRANCHES), 'sha256': MP.sha256(OLD_BRANCHES), 'continuation': 'start agent (joint_van_d05/seed_0)'},
         'lineage_agents': {f'seed_{s}': {'path': str(lineage_ckpt(s)), 'sha256': (MP.sha256(lineage_ckpt(s)) if lineage_ckpt(s).exists() else 'missing')} for s in MP.SEEDS},
         'A_generation': {'anchors': 'the pilot anchor set, every anchor', 'query': 'the logged torque, executed once', 'hazard_seed': 'HAZARD_SEED0 + anchor_id (identical to the old file: paired by anchor)',
                          'continuation': 'the lineage agent, mode tanh(loc), closed-loop to reach / death / horizon', 'per_lineage': [str(new_branch_path(s)) for s in MP.SEEDS]},
         'A_profile': {'paired_by': 'anchor (same state, same query torque, same hazard draw)', 'weights': 'anchor weight (the critic anchor law)',
                       'quantities': ['entered the far route after the query (y >= 6 at x < 2)', 'completed the far route (entered and reached the goal)', 'entered a hazard zone', 'outcome shares', 'branch length',
                                      'the critic goal marginal under the truncated geometric law: mass in the far-route legs / goal area / hazard zones'],
                       'strata': ['reset rows (t = 0)', 'anchor in start region', 'start region, t in [1, 30)', 'every maze region', 'logged shortcut / detour episodes x start region', 'all']},
         'B_gate': GATE,
         'B_design': {'source': 'the lineage final (full TrainingState: params, target, policy and critic Adam states incl. the clip state, key)', 'updates': UPDATES,
                      'streams': 'critic and actor streams fast-forwarded by 30,000 batches (the run continues its own sequence)',
                      'arms': {'CFold': 'the start-agent futures (branches_cf.npz): the plain continuation of the run', 'CFnew': 'the lineage futures (branches_cf_c_s<s>.npz)'},
                      'held_fixed': 'the sealed critic_clip0.1 recipe: NCE, actor loss, bc 0.05, gamma 0.999, lr, clip 0.1, batch 1024, anchors and weights, actor stream, evaluation (seed 3909, 300, mode)',
                      'no_restart': 'nothing re-initialised; the round-2 confound (restart vs futures) is absent'},
         'B_comparisons': {'primary': 'CFnew - CFold success, paired on the common episodes per lineage; rule: mean over the 3 lineages > 2 x seed s.e. and 3/3',
                           'secondary': ['detour', 'death', 'timeout', 'hazard success', 'the route ledger'], 'context': ['CFnew - source (the lineage agent)', 'CFold - source (more training alone)']},
         'no_selection': 'every lineage, both arms, the final checkpoints evaluated once; the gate is evaluated mechanically before any training',
         'status': 'oracle (simulator) futures under the disclosed optimizer-stabilisation recipe; not a learned-ETT result'}
  MP.write_json(p, man)
  print(json.dumps(man, indent=1), flush=True)


# ---------------------------------------------------------------- generate
def mode_generate(args):
  s = args.lineage
  ck = lineage_ckpt(s)
  if not ck.exists():
    raise SystemExit(f'lineage agent missing: {ck}')
  out_path = new_branch_path(s)
  if out_path.exists() and not args.force:
    print(f'{out_path} exists', flush=True); return
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  meta, _ = MP.generate_branches(anchors, ck, args.workers, out_path, limit=args.limit)
  MP.write_json(OUT / f'generation_c_s{s}.json', MP.generation_summary(out_path, anchors))
  print(json.dumps(meta, indent=1), flush=True)


# ----------------------------------------------------------------- profile
def branch_features(path):
  with np.load(path, allow_pickle=False) as d:
    xy = np.ascontiguousarray(d['obs_rows'][:, :2]).astype(np.float64)
    off, L = d['offset'].astype(np.int64), d['length'].astype(np.int64)
    oc, aid = d['outcome'].astype(str), d['anchor_id']
    meta = json.loads(str(d['meta']))
  assert np.all(aid == np.arange(len(aid)))
  K = len(L)
  anchor_of_row = np.repeat(np.arange(K), L)
  m = np.arange(len(xy)) - np.repeat(off, L)
  after = m >= 1
  far_row = (xy[:, 1] >= DETOUR_Y) & (xy[:, 0] < DETOUR_MAX_X)
  zone_row = (np.abs(xy[:, 1]) < 2.0) & (((xy[:, 0] >= ZONE_X[1][0]) & (xy[:, 0] <= ZONE_X[1][1])) | ((xy[:, 0] >= ZONE_X[2][0]) & (xy[:, 0] <= ZONE_X[2][1])))
  entered_far = np.bincount(anchor_of_row, weights=(far_row & after), minlength=K) > 0
  entered_zone = np.bincount(anchor_of_row, weights=(zone_row & after), minlength=K) > 0
  success = oc == 'success'
  w = np.where(after, GAMMA ** m, 0.0)
  z = np.bincount(anchor_of_row, weights=w, minlength=K); w = w / z[anchor_of_row]
  reg = region_of(xy)
  mass = {name: np.bincount(anchor_of_row, weights=w * np.isin(reg, idx), minlength=K) for name, idx in (('far', FAR), ('goal', [GOAL]), ('zone', ZONES))}
  return {'entered_far': entered_far, 'completed_far': entered_far & success, 'entered_zone': entered_zone, 'success': success, 'death': oc == 'death',
          'timeout': oc == 'timeout', 'length': L.astype(np.float64), 'mass_far': mass['far'], 'mass_goal': mass['goal'], 'mass_zone': mass['zone'], 'meta': meta}


QUANT = ('entered_far', 'completed_far', 'entered_zone', 'success', 'death', 'timeout', 'length', 'mass_far', 'mass_goal', 'mass_zone')


def wmean(x, w):
  return float((w * x).sum() / w.sum())


def wse(d, w):
  w = w / w.sum(); mu = (w * d).sum()
  return float(np.sqrt((w ** 2 * (d - mu) ** 2).sum()))


def mode_profile(args):
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  logged = route[anchors.episode]
  W = anchors.weight
  reg = region_of(anchors.state[:, :2])
  t = anchors.t
  strata = {'reset rows (t = 0)': t == 0, 'anchor in start region': reg == REGIONS.index('start'), 'start region, t in [1, 30)': (reg == REGIONS.index('start')) & (t >= 1) & (t < 30),
            'logged shortcut | start region': (reg == REGIONS.index('start')) & (logged == 'shortcut'), 'logged detour | start region': (reg == REGIONS.index('start')) & (logged == 'detour')}
  for r, name in enumerate(REGIONS):
    if name != 'start' and (reg == r).sum() >= 50:
      strata[f'anchor in {name}'] = reg == r
  strata['logged shortcut | all'] = logged == 'shortcut'; strata['logged detour | all'] = logged == 'detour'; strata['all'] = np.ones(anchors.n, bool)
  old = branch_features(OLD_BRANCHES)
  res = {'old_meta': old['meta'], 'lineages': {}, 'strata_n': {k: int(v.sum()) for k, v in strata.items()}, 'strata_weight': {k: float(W[v].sum()) for k, v in strata.items()}}
  gate_vals = []
  for s in MP.SEEDS:
    p = new_branch_path(s)
    if not p.exists():
      print(f'lineage {s}: no new branches yet', flush=True); continue
    new = branch_features(p)
    L = {'new_meta': new['meta'], 'strata': {}}
    for name, msk in strata.items():
      ww = W[msk]
      blk = {'n': int(msk.sum()), 'weight': float(ww.sum())}
      for q in QUANT:
        o, n = old[q][msk].astype(np.float64), new[q][msk].astype(np.float64)
        blk[q] = {'old': wmean(o, ww), 'new': wmean(n, ww), 'delta': wmean(n - o, ww), 'delta_se': wse(n - o, ww)}
      o, n = old['entered_far'][msk], new['entered_far'][msk]
      blk['switch'] = {'old_no_new_far': wmean((~o & n).astype(float), ww), 'old_far_new_no': wmean((o & ~n).astype(float), ww)}
      L['strata'][name] = blk
    res['lineages'][f'seed_{s}'] = L
    gate_vals.append(L['strata'][GATE['stratum']]['completed_far']['delta'])
  if len(gate_vals) == len(MP.SEEDS):
    res['gate'] = {**GATE, 'per_lineage_delta': gate_vals, 'mean': float(np.mean(gate_vals)), 'passed': bool(all(v > 0 for v in gate_vals) and np.mean(gate_vals) >= 0.02)}
  MP.write_json(OUT / 'profile.json', res)
  write_profile_md(res)
  if 'gate' in res:
    print('GATE', 'PASSED' if res['gate']['passed'] else 'NOT PASSED', res['gate']['per_lineage_delta'], flush=True)


def write_profile_md(res):
  L = ['# A. The current agents\' futures at the same training anchors (paired by anchor: same state, same logged query torque, same hazard draw)', '',
       'Old = the start agent\'s continuation (`branches_cf.npz`, the critic\'s positive futures of every run so far); new = the lineage agent (the clipped CF final of that seed) continuing.  '
       'Anchor-weighted shares (the critic anchor law); entered far = y >= 6 at x < 2 after the query; completed far = entered and the branch reached the goal; masses = the critic\'s goal marginal '
       '(truncated geometric law, gamma 0.999) in the far-route legs / the goal area / the hazard zones.  `profile.json`.', '']
  if 'gate' in res:
    g = res['gate']
    L += [f'**Gate for B** ({g["stratum"]}, {g["quantity"]}; {g["rule"]}): per lineage {" / ".join(f"{v:+.3f}" for v in g["per_lineage_delta"])}, mean {g["mean"]:+.3f} -> **{"PASSED" if g["passed"] else "NOT PASSED"}**.', '']
  for name in next(iter(res['lineages'].values()))['strata']:
    L += [f'## {name} (n = {res["strata_n"][name]}, weight {res["strata_weight"][name]:.3f})', '',
          '| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for lin, Lg in res['lineages'].items():
      b = Lg['strata'][name]
      f2 = lambda q: f'{b[q]["old"]:.3f} / {b[q]["new"]:.3f}'
      L.append(f'| {lin} | {f2("entered_far")} ({b["entered_far"]["delta"]:+.3f}) | {f2("completed_far")} ({b["completed_far"]["delta"]:+.3f} +- {b["completed_far"]["delta_se"]:.3f}) | {f2("entered_zone")} | {f2("success")} | {f2("death")} | {f2("timeout")} | '
               f'{b["length"]["old"]:.0f} / {b["length"]["new"]:.0f} | {f2("mass_far")} | {f2("mass_goal")} | {f2("mass_zone")} | {b["switch"]["old_no_new_far"]:.3f} / {b["switch"]["old_far_new_no"]:.3f} |')
    L.append('')
  (OUT / 'PROFILE.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


# ------------------------------------------------------------------ resume
def mode_resume(args):
  import jax
  from crl import checkpoint
  import diag_v6_training_replay as DR
  s, arm = args.lineage, args.arm
  man = MP.read_json(OUT / 'manifest.json')
  if arm == 'CFnew':
    prof = MP.read_json(OUT / 'profile.json')
    if not prof.get('gate', {}).get('passed') and not args.force:
      raise SystemExit('the gate of profile.json is not passed; CFnew is not run (use --force only on the user\'s explicit decision)')
  src = lineage_ckpt(s)
  assert MP.sha256(src) == man['lineage_agents'][f'seed_{s}']['sha256'], 'lineage checkpoint differs from the sealed hash'
  bp = OLD_BRANCHES if arm == 'CFold' else new_branch_path(s)
  out = OUT / arm / f'seed_{s}'
  if (out / 'final.pkl').exists() and not args.force:
    print(f'{out} exists', flush=True)
  else:
    out.mkdir(parents=True, exist_ok=True)
    clip = float(MP.VARIANTS[RECIPE]['critic_clip'])
    cfg, nets, update_step, critic_stream, actor_stream = DR.build(s, 'CF', bp, out / '_cfg', critic_clip=clip)
    _, q_opt = DR.build.optimizers
    step0, state = checkpoint.load_checkpoint(src)
    fresh = q_opt.init(state.q_params)
    assert jax.tree_util.tree_structure(state.q_optimizer_state) == jax.tree_util.tree_structure(fresh), 'the checkpoint critic optimizer state is not the clip -> Adam chain'
    ff = DR.fast_forward(critic_stream, actor_stream, int(step0))
    print(f'loaded {src} @ step {step0} ({arm}, futures {bp.name}); streams fast-forwarded by {step0} batches in {ff:.0f} s', flush=True)
    h0 = {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)}
    checkpoint.save_named(str(out), 'init', int(step0), state)

    def _multi(state, pair):
      state, metrics = jax.lax.scan(update_step, state, pair)
      return state, jax.tree_util.tree_map(lambda x: x.mean(), metrics)
    multi_update = jax.jit(_multi)
    G = MP.G
    n, t0, hist = 0, time.time(), []
    first = {}
    for it in range(UPDATES // G):
      cbs, ks = [], []
      for _ in range(G):
        tr, (k, m) = critic_stream.sample(); cbs.append(tr); ks.append(k)
      abs_ = [actor_stream.sample(cfg.batch_size) for _ in range(G)]
      if it == 0:
        # CFold and CFnew must see the same anchors and the same actor rows (same streams, same fast-forward) and differ only in the goals
        first = {'critic_anchor_ids': MP.arr_hash(*ks), 'critic_goals': MP.arr_hash(*[b.observation[:, MP.STATE_DIM:] for b in cbs]),
                 'actor': MP.arr_hash(*[b.observation for b in abs_], *[b.action for b in abs_])}
      state, metrics = multi_update(state, (MP._stack(cbs), MP._stack(abs_)))
      n += G
      if n % MP.LOG_EVERY == 0 or n == UPDATES:
        m = {k: float(v) for k, v in metrics.items()}
        hist.append({'update': int(step0) + n, **m})
        print(f'[{arm} s{s} upd {int(step0) + n:>6}] critic {m.get("critic_loss", 0):.4f} cat_acc {m.get("categorical_accuracy", 0):.3f} actor {m.get("actor_loss", 0):.4f} '
              f'bc_nll {m.get("bc_nll", 0):.3f} q_term {m.get("actor_q_term", 0):.3f} {n / (time.time() - t0):.1f} upd/s', flush=True)
      for ms in MP.MILESTONES:
        if n == ms:
          checkpoint.save_named(str(out), str(int(step0) + ms), int(step0) + n, state)
    checkpoint.save_named(str(out), 'final', int(step0) + n, state)
    MP.write_json(out / 'train_manifest.json', {
        'arm': arm, 'lineage': s, 'source_ckpt': str(src), 'source_ckpt_sha256': MP.sha256(src), 'source_step': int(step0), 'updates': n, 'final_step': int(step0) + n,
        'state_carried': ['policy_params', 'q_params', 'target_q_params', 'policy_optimizer_state', 'q_optimizer_state (clip -> Adam chain)', 'key'],
        'streams': {'critic_seed': MP.CRITIC_STREAM_SEED0 + s, 'actor_seed': MP.ACTOR_STREAM_SEED0 + s, 'fast_forwarded_batches': int(step0)},
        'futures': str(bp), 'branch_sha256': MP.sha256(bp), 'critic_clip': clip, 'config': MP.config_dump(cfg), 'params_at_load': h0, 'first_batches': first,
        'params_final': {'q': MP._tree_hash(state.q_params), 'policy': MP._tree_hash(state.policy_params)},
        'device': str(jax.devices()[0]), 'jax_version': jax.__version__, 'wall_seconds': time.time() - t0, 'history': hist})
    print(f'{arm} s{s}: {n} updates from step {step0} in {time.time() - t0:.0f} s', flush=True)
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = MP.EVAL['seed'], MP.EVAL['n']
  print(MP.D.evaluate_ckpt(out / 'final.pkl', out, MP.EVAL['policy']), flush=True)


# ------------------------------------------------------------------ report
def mode_report(args):
  es = MP.EVAL['seed']
  paths = {'start': MP.OUT / 'start_agent' / f'eval_mean_s{es}.json'}
  for s in MP.SEEDS:
    paths[f'source/seed_{s}'] = MP.run_dir('CF', s, MP.variant_base(RECIPE)) / f'eval_mean_s{es}.json'
    paths[f'O(clip)/seed_{s}'] = MP.run_dir('O', s, MP.variant_base(RECIPE)) / f'eval_mean_s{es}.json'
    for arm in ARMS:
      paths[f'{arm}/seed_{s}'] = OUT / arm / f'seed_{s}' / f'eval_mean_s{es}.json'
  missing = [k for k, p in paths.items() if not p.exists()]
  if missing:
    raise SystemExit(f'missing evaluations: {missing}')
  E = {k: MP._episodes(p) for k, p in paths.items()}
  raw = {k: MP.read_json(p)['episodes'] for k, p in paths.items()}
  prof = MP.read_json(OUT / 'profile.json')
  L = ['# B. Paired continuation from the same full checkpoint: the lineage futures (CFnew) vs the start-agent futures (CFold)', '',
       f'`manifest.json` (sealed before generation; gate {"PASSED" if prof["gate"]["passed"] else "NOT PASSED -- CFnew run on the user\'s decision"}).  Each lineage: the clipped CF final (30,000 updates) continued for {UPDATES:,} more updates '
       f'under the same recipe (critic clip 0.1, streams continued), critic futures = the start agent\'s (CFold) or the lineage agent\'s own (CFnew); the same {MP.EVAL["n"]} evaluation episodes (seed {es}, mode).  '
       'Rule: mean over the 3 lineages > 2 x seed s.e. and 3/3.', '', '## Per policy', '', '| policy | success | detour | death | timeout | success no hazard | success hazard | mean steps |', '|---|---:|---:|---:|---:|---:|---:|---:|']
  for k, e in E.items():
    h = MP._headline(e)
    L.append(f'| {k} | {h["success"]:.3f} | {h["detour"]:.3f} | {h["death"]:.3f} | {h["timeout"]:.3f} | {h["success_no_hazard"]:.3f} | {h["success_hazard"]:.3f} | {h["mean_steps"]:.0f} |')
  by = {a: {s: E[f'{a}/seed_{s}'] for s in MP.SEEDS} for a in ('CFold', 'CFnew', 'source', 'O(clip)')}
  comps = [('CFnew - CFold [primary]', by['CFnew'], by['CFold']), ('CFnew - source', by['CFnew'], by['source']), ('CFold - source', by['CFold'], by['source']),
           ('CFnew - O(clip)', by['CFnew'], by['O(clip)']), ('CFnew - start', by['CFnew'], E['start'])]
  res = {}
  L += ['', '## Paired differences on the common episodes (per lineage; seed mean, seed s.e., episode-bootstrap s.e.)', '']
  for key in ('success', 'detour', 'failure', 'timeout'):
    L += [f'### {key}', '', '| comparison | per lineage | mean | seed s.e. | boot s.e. | same direction | rule |', '|---|---|---:|---:|---:|---|---|']
    for cname, a, b in comps:
      r = MP.paired_block(a, b, key=key); res[f'{cname}:{key}'] = r
      per = ' / '.join(f'{v["mean"]:+.3f}' for v in r['per_seed'].values())
      rule = ('met' if r['improvement_rule_met'] else 'not met') if (key == 'success' and 'primary' in cname) else '-'
      L.append(f'| {cname} | {per} | {r["mean"]:+.3f} | {r["seed_se"]:.3f} | {r["episode_bootstrap_se_of_mean"]:.3f} | {r["seeds_same_direction"]} | {rule} |')
    L.append('')
  L += ['## Route ledger', '', *MP.ledger_table(raw), '']
  p = res['CFnew - CFold [primary]:success']; q = res['CFnew - source:success']; o = res['CFold - source:success']
  L += ['## Reading', '', f'Primary CFnew - CFold success: {p["mean"]:+.3f} (seed s.e. {p["seed_se"]:.3f}, boot {p["episode_bootstrap_se_of_mean"]:.3f}, {p["seeds_same_direction"]}) -> {"MET" if p["improvement_rule_met"] else "NOT MET"}.  '
        f'CFnew - source {q["mean"]:+.3f} ({q["seed_se"]:.3f}, {q["seeds_same_direction"]}); CFold - source {o["mean"]:+.3f} ({o["seed_se"]:.3f}, {o["seeds_same_direction"]}).  '
        'Oracle futures under the disclosed optimizer-stabilisation recipe; not a learned-ETT result.']
  MP.write_json(OUT / 'results.json', {'headlines': {k: MP._headline(e) for k, e in E.items()}, 'paired': res, 'ledger': {k: MP.route_ledger(r) for k, r in raw.items()}})
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def mode_confirm(args):
  """The B arms (CFold / CFnew x 3 lineages) and their sources once on a fresh evaluation draw (seed CONFIRM_SEED): hashes and the
  comparison sealed before any evaluation on the draw; run only after the development-draw report (the rule met there, or the
  user's explicit decision with --force)."""
  cd = OUT / f'confirm_s{CONFIRM_SEED}'; cd.mkdir(parents=True, exist_ok=True)
  res_dev = MP.read_json(OUT / 'results.json') if (OUT / 'results.json').exists() else None
  if not args.force and not (res_dev and res_dev['paired']['CFnew - CFold [primary]:success']['improvement_rule_met']):
    raise SystemExit('the primary rule is not met on the development draw (or no report yet); the confirmation is not run')
  T = [('start', MP.START_CKPT, MP.OUT / 'start_agent')]
  for s in MP.SEEDS:
    T.append((f'source/seed_{s}', lineage_ckpt(s), lineage_ckpt(s).parent))
    for arm in ARMS:
      T.append((f'{arm}/seed_{s}', OUT / arm / f'seed_{s}' / 'final.pkl', OUT / arm / f'seed_{s}'))
  man_p = cd / 'manifest.json'
  if not man_p.exists():
    missing = [lab for lab, ck, _ in T if not ck.exists()]
    if missing:
      raise SystemExit(f'confirm seal needs every checkpoint present; missing {missing}')
    from crl import checkpoint
    man = {'sealed_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'seed': CONFIRM_SEED, 'n': MP.EVAL['n'], 'policy': MP.EVAL['policy'],
           'checkpoints': {lab: {'path': str(ck), 'sha256': MP.sha256(ck), 'ckpt_step': int(checkpoint.load_checkpoint(ck)[0])} for lab, ck, _ in T},
           'comparisons': 'CFnew - CFold success (primary; rule > 2 x seed s.e. and 3/3), CFnew - source, CFold - source, CFnew - start; the same final checkpoints, evaluated once',
           'development_draws_not_re_used': [909, 2909, 3909, 4909], 'no_selection': 'nothing chosen after the result; the draw is not re-used for development'}
    MP.write_json(man_p, man)
  man = MP.read_json(man_p)
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = CONFIRM_SEED, MP.EVAL['n']
  for lab, ck, od in T:
    if args.only and lab not in args.only:
      continue
    if not ck.exists():
      print(f'skip {lab}: checkpoint not on this machine', flush=True); continue
    if (od / f'eval_mean_s{CONFIRM_SEED}.json').exists():
      continue
    assert MP.sha256(ck) == man['checkpoints'][lab]['sha256'], f'{lab}: checkpoint differs from the sealed hash'
    print(f'== confirm evaluate {lab}', flush=True)
    MP.D.evaluate_ckpt(ck, od, MP.EVAL['policy'])
  MP.D.EVAL['seed'], MP.D.EVAL['n'] = MP.EVAL['seed'], MP.EVAL['n']
  if not all((od / f'eval_mean_s{CONFIRM_SEED}.json').exists() for _, _, od in T):
    print('confirm: evaluations still missing; report skipped', flush=True); return
  E = {lab: MP._episodes(od / f'eval_mean_s{CONFIRM_SEED}.json') for lab, _, od in T}
  raw = {lab: MP.read_json(od / f'eval_mean_s{CONFIRM_SEED}.json')['episodes'] for lab, _, od in T}
  by = {a: {s: E[f'{a}/seed_{s}'] for s in MP.SEEDS} for a in ('CFold', 'CFnew', 'source')}
  comps = [('CFnew - CFold [primary]', by['CFnew'], by['CFold']), ('CFnew - source', by['CFnew'], by['source']), ('CFold - source', by['CFold'], by['source']), ('CFnew - start', by['CFnew'], E['start'])]
  res = {}
  L = [f'# B confirmed once on a fresh evaluation draw (seed {CONFIRM_SEED}; hashes sealed {man["sealed_at"]})', '', '| policy | success | detour | death | timeout | success hazard |', '|---|---:|---:|---:|---:|---:|']
  for lab, e in E.items():
    h = MP._headline(e); L.append(f'| {lab} | {h["success"]:.3f} | {h["detour"]:.3f} | {h["death"]:.3f} | {h["timeout"]:.3f} | {h["success_hazard"]:.3f} |')
  for key in ('success', 'detour', 'failure', 'timeout'):
    L += ['', f'### {key}', '', '| comparison | per lineage | mean | seed s.e. | boot s.e. | same direction | rule |', '|---|---|---:|---:|---:|---|---|']
    for cname, a, b_ in comps:
      r = MP.paired_block(a, b_, key=key); res[f'{cname}:{key}'] = r
      per = ' / '.join(f'{v["mean"]:+.3f}' for v in r['per_seed'].values())
      rule = ('met' if r['improvement_rule_met'] else 'not met') if (key == 'success' and 'primary' in cname) else '-'
      L.append(f'| {cname} | {per} | {r["mean"]:+.3f} | {r["seed_se"]:.3f} | {r["episode_bootstrap_se_of_mean"]:.3f} | {r["seeds_same_direction"]} | {rule} |')
  L += ['', '## Route ledger', '', *MP.ledger_table(raw), '']
  p = res['CFnew - CFold [primary]:success']
  L += ['## Verdict', '', f'CFnew - CFold success on the confirmation draw: {p["mean"]:+.3f} (seed s.e. {p["seed_se"]:.3f}, {p["seeds_same_direction"]}) -> {"REPRODUCED" if p["improvement_rule_met"] else "NOT REPRODUCED"}.']
  MP.write_json(cd / 'results.json', {'headlines': {k: MP._headline(e) for k, e in E.items()}, 'paired': res, 'ledger': {k: MP.route_ledger(r) for k, r in raw.items()}})
  (cd / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('seal', 'generate', 'profile', 'resume', 'report', 'confirm'))
  ap.add_argument('--only', nargs='*', default=None)
  ap.add_argument('--lineage', type=int, default=None)
  ap.add_argument('--arm', choices=ARMS, default=None)
  ap.add_argument('--workers', type=int, default=8)
  ap.add_argument('--limit', type=int, default=None, help='generate: first N anchors only (smoke)')
  ap.add_argument('--force', action='store_true')
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  if args.mode in ('generate', 'resume') and args.lineage is None:
    ap.error(f'{args.mode} needs --lineage')
  if args.mode == 'resume' and args.arm is None:
    ap.error('resume needs --arm')
  {'seal': mode_seal, 'generate': mode_generate, 'profile': mode_profile, 'resume': mode_resume, 'report': mode_report, 'confirm': mode_confirm}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
