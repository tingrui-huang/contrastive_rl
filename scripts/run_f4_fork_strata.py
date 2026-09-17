"""Step 11: fork-stratified critic batches on replay_E, two arms, gated on the critic.

Step 10b measured what the replay-E relabeling law asks of the critic at the
fork -- a DOWN-vs-RIGHT margin of +0.39 nats over all fork anchors (+0.72 at
the 550 training contexts) for a goal frame in cell (8,3) -- and what the
five E critics deliver on the same rows: mean +0.00 (point) / -0.07 (width),
seed scatter +-0.25.  Fork -> (8,3) rows are 0.7% of the critic's rows.  The
original episodes' fork anchors, whose RIGHT continuation is the sighted
teacher's (P((8,3)) 0.46 against the blind ETT's 0.12), are 25% of the RIGHT
sector's fork mass and cap the target: synthetic sources alone would ask for
+1.26.

Two arms, same sealed 30k recipe, replay_E, five critic seeds each; only the
ANCHOR distribution of the critic's batches changes (TrajectoryBuffer.
set_anchor_strata; the relabeling law, the loss and the networks are as
sealed):

  strat        every batch of 256 = 128 anchors from the buffer's own law
               + 64 fork anchors with a DOWN-sector recorded action
               + 64 fork anchors with a RIGHT-sector recorded action
               (the composition of the absorbing line's G1 experiment)
  strat_synth  the same composition, with the ORIGINAL episodes' fork-cell
               rows removed from every stratum: at the fork the critic's
               positives come from synthetic (ETT) continuations only, the
               observational rows stay everywhere else.  Row provenance
               (audit_source) selects anchors; it is never fed to a network.

Gate, fixed before training: on the buffer-law fork -> jitter-(8,3) rows of
replay_E (the rows of Step 10b, seed 777), the width-0.75 margin
Qbar(DOWN loc) - Qbar(RIGHT loc) >= +0.10 (the value of D1, whose three
actors all went DOWN).  Actors (three fresh balanced-BC actors, the Step 8
recipe on replay_E) are trained only for critics that pass, unless
--all-critics is given.

Stages: train | gate | actors | summarize | all.
Outputs: outputs/pointmaze_fork_strata_v1/.
"""
from __future__ import annotations

import argparse
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
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from run_f4_corner_coverage_fix import (   # noqa: E402
    REPLAY_E, CRL_STEPS, BC_COEF, BC_CAP, NEW_EVAL, ACTOR_SEEDS, GOAL, D_CRITICS,
    make_network, critic_metrics, sha256, write_json, run)
from audit_f4_goal_frames import frame_type, DOWN_LOC, RIGHT_LOC   # noqa: E402

E = ROOT / 'outputs' / 'pointmaze_corner_coverage_fix_v1'
OUT = Path(os.environ.get('F4_STRATA_OUT', ROOT / 'outputs' / 'pointmaze_fork_strata_v1'))
ARMS = ('strat', 'strat_synth')
COUNTS = (128, 64, 64)
CRITIC_SEEDS = (0, 1, 2, 3, 4)
GATE_WIDTH = 0.10
TITLE = '# Step 11: fork-stratified critic batches on replay_E'
FORK = (1, 3)
WIDTH = 0.75
E_CRITICS = {s: E / 'seeds' / f'seed_{s}' / 'crl' / 'E' / 'final.pkl' for s in range(5)}


def sectors(a):
  ang = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
  return {'DOWN': (ang < -67.5) & (ang > -112.5), 'RIGHT': (ang > -22.5) & (ang < 22.5)}


# ------------------------------------------------------------------ strata
def build_strata(path, arm):
  """(strata, stats) for TrajectoryBuffer.set_anchor_strata on the replay at
  ``path``.  Stratum 0 = the buffer's own law (every eligible anchor, weight
  1 / (L_e - 1)); 1 = fork anchors with a DOWN-sector action; 2 = fork anchors
  with a RIGHT-sector action.  strat_synth drops the original episodes'
  fork-cell rows from all three."""
  with np.load(path, allow_pickle=False) as d:
    obs, act, src = d['obs'], d['act'], d['audit_source']
    lengths = (d['lengths'].astype(np.int64) if 'lengths' in d.files
               else np.full(len(obs), obs.shape[1], np.int64))
  n, L = obs.shape[:2]
  e = np.repeat(np.arange(n), L - 1)
  i = np.tile(np.arange(L - 1), n)
  ok = i < lengths[e] - 1
  e, i = e[ok], i[ok]
  w_law = 1.0 / (lengths[e] - 1)
  cell = np.floor(obs[e, i, :2]).astype(np.int64)
  fork = (cell[:, 0] == FORK[0]) & (cell[:, 1] == FORK[1])
  sec = sectors(act[e, i])
  keep = np.ones(len(e), bool)
  if arm == 'strat_synth':
    keep &= ~(fork & (src[e] == 0))
  elif arm != 'strat':
    raise ValueError(arm)
  masks = [keep, keep & fork & sec['DOWN'], keep & fork & sec['RIGHT']]
  strata = [(e[m], i[m], w_law[m]) for m in masks]
  law_fork_share = float(w_law[keep & fork].sum() / w_law[keep].sum())
  stats = {
      'arm': arm, 'counts': list(COUNTS), 'stratum_rows': [int(m.sum()) for m in masks],
      'rows_dropped': int((~keep).sum()),
      'fork_share_under_law': law_fork_share,
      'fork_share_per_batch': float((COUNTS[0] * law_fork_share + COUNTS[1] + COUNTS[2]) / sum(COUNTS)),
      'fork_down_rows_by_source': {int(s): int((masks[1] & (src[e] == s)).sum()) for s in np.unique(src)},
      'fork_right_rows_by_source': {int(s): int((masks[2] & (src[e] == s)).sum()) for s in np.unique(src)},
  }
  return strata, stats


def prepare_hook(arm):
  def prepare(buffer, path):
    strata, stats = build_strata(path, arm)
    buffer.set_anchor_strata(strata, COUNTS)
    print(f'ANCHOR STRATA ({arm}): rows {stats["stratum_rows"]}, counts {COUNTS}, '
          f'fork share {stats["fork_share_under_law"]:.3f} -> {stats["fork_share_per_batch"]:.3f} '
          f'per batch, dropped {stats["rows_dropped"]}', flush=True)
    return stats
  return prepare


# ------------------------------------------------------------------- train
def crl_config(seed, ckpt_dir, steps):
  """The sealed critic recipe of arms D / E verbatim (run_f4_corner_coverage_fix.crl_config),
  with the checkpoint directory and the budget as arguments."""
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0', offline_dataset=str(REPLAY_E),
      obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
      start_index=0, end_index=-1, max_number_of_steps=int(steps),
      fail_bank_path='', fail_neg_alpha=0.0, obs_norm_mode='',
      obs_norm_z_scale=0.0, anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=BC_COEF, random_goals=0.5, entropy_coefficient=0.0,
      target_entropy=0.0, batch_size=256, repr_dim=64,
      hidden_layer_sizes=(256, 256), discount=0.95, learning_rate=3e-4,
      actor_learning_rate=3e-4, num_sgd_steps_per_step=10, num_actors=0,
      guard_abort=True, jit=True, seed=int(seed),
      eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=5_000,
      ckpt_every_steps=int(steps), ckpt_dir=str(ckpt_dir))


def critic_dir(arm, seed):
  return OUT / arm / f'seed_{seed}' / 'critic'


def critic_path(arm, seed):
  return critic_dir(arm, seed) / 'final.pkl'


def train_one(arm, seed, steps):
  from crl.train import train
  out = critic_dir(arm, seed)
  if (out / 'final.pkl').exists():
    print(f'{out} exists', flush=True)
    return
  cfg = crl_config(seed, out, steps)
  t0 = time.time()
  train(cfg, buffer_prepare=prepare_hook(arm))
  _, stats = build_strata(REPLAY_E, arm)
  write_json(out / 'run_summary.json', {
      'arm': arm, 'seed': seed, 'dataset': str(REPLAY_E), 'dataset_sha256': sha256(REPLAY_E),
      'steps': int(steps), 'batch': 256, 'anchor_strata': stats,
      'wall_seconds': time.time() - t0})


def stage_train(arms, seeds, steps, parallel):
  procs = []
  for arm in arms:
    for seed in seeds:
      if critic_path(arm, seed).exists():
        continue
      log = OUT / 'logs' / f'critic_{arm}_seed{seed}.log'
      log.parent.mkdir(parents=True, exist_ok=True)
      cmd = [sys.executable, str(Path(__file__).resolve()), '_train_one', '--arms', arm,
             '--seeds', str(seed), '--steps', str(steps)]
      f = open(log, 'w', encoding='utf-8')
      f.write(' '.join(cmd) + '\n')
      procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                     env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, critic_dir(arm, seed)))
      if len(procs) >= parallel:
        _wait(procs)
  _wait(procs)


def _wait(procs):
  for p, f, out in procs:
    rc = p.wait()
    f.close()
    if rc != 0:
      raise RuntimeError(f'run for {out} failed (rc {rc})')
  procs.clear()


# -------------------------------------------------------------------- gate
def jitter_rows(n_rows=4000, seed=777):
  """The buffer-law fork -> jitter-(8,3) rows of replay_E used in Step 10b."""
  from crl.bc_balanced import GroupBalancedBCSampler
  with np.load(REPLAY_E, allow_pickle=False) as d:
    obs, act, lengths = d['obs'], d['act'], d['lengths'].astype(np.int64)
  smp = GroupBalancedBCSampler(obs, act, lengths, 0.95, 8, cap=BC_CAP, seed=seed)
  tr, ii, jj = smp.tr, smp.ii.astype(np.int64), smp.jj.astype(np.int64)
  s_cell = np.floor(obs[tr, ii, :2]).astype(np.int64)
  g_cell = np.floor(obs[tr, jj, :2]).astype(np.int64)
  fork = (s_cell[:, 0] == FORK[0]) & (s_cell[:, 1] == FORK[1])
  g83 = (g_cell[:, 0] == 8) & (g_cell[:, 1] == 3)
  gt = np.full(len(tr), '', dtype=object)
  gt[fork & g83], _ = frame_type(obs[tr[fork & g83], jj[fork & g83], :8])
  jit = fork & g83 & ((gt == 'jitter') | (gt == 'parked'))
  w = np.where(jit, smp.w_original, 0.0)
  cdf = np.cumsum(w / w.sum())
  rng = np.random.default_rng(seed)
  pos = np.minimum(np.searchsorted(cdf, rng.random(n_rows), side='right'), len(cdf) - 1)
  return np.concatenate([obs[tr[pos], ii[pos], :8], obs[tr[pos], jj[pos], :8]], 1).astype(np.float32)


def gate_metrics(q_params, network, rows, eps):
  import jax
  import jax.numpy as jnp

  @jax.jit
  def q_point(qp, o, a):
    phi, psi = network.representation_network.apply(qp, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  @jax.jit
  def qbar(qp, o, loc):
    a = jnp.tanh(loc[None, None, :] + WIDTH * eps[None])
    a = jnp.broadcast_to(a, (o.shape[0],) + a.shape[1:])
    o_rep = jnp.repeat(o[:, None, :], eps.shape[0], 1).reshape(-1, o.shape[1])
    phi, psi = network.representation_network.apply(qp, o_rep, a.reshape(-1, 2))
    return jnp.mean(jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], -1), axis=1)

  o = jnp.asarray(rows)
  d_ = jnp.tile(jnp.array([[0.0, -1.0]], jnp.float32), (o.shape[0], 1))
  r_ = jnp.tile(jnp.array([[1.0, 0.0]], jnp.float32), (o.shape[0], 1))
  pm, wm = [], []
  for k in range(0, o.shape[0], 1024):
    sl = slice(k, k + 1024)
    pm.append(np.asarray(q_point(q_params, o[sl], d_[sl]) - q_point(q_params, o[sl], r_[sl])))
    wm.append(np.asarray(qbar(q_params, o[sl], jnp.asarray(DOWN_LOC)) - qbar(q_params, o[sl], jnp.asarray(RIGHT_LOC))))
  pm, wm = np.concatenate(pm), np.concatenate(wm)
  return {'jitter_point': float(pm.mean()), 'jitter_width': float(wm.mean()),
          'jitter_width_se': float(wm.std() / np.sqrt(len(wm))), 'rows': int(len(wm))}


def stage_gate(arms, seeds):
  import jax.numpy as jnp
  from crl import checkpoint
  network = make_network()
  rows = jitter_rows()
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  eps_root = np.random.default_rng(0).standard_normal((256, 2)).astype(np.float32)
  eps_jit = jnp.asarray(np.random.default_rng(777).standard_normal((256, 2)).astype(np.float32))
  entries = [(f'{arm} seed {s}', critic_path(arm, s), arm) for arm in arms for s in seeds]
  entries += [(f'E seed {s} (reference)', p, 'E') for s, p in E_CRITICS.items()]
  entries += [(f'D seed {s} (reference)', p, 'D') for s, p in D_CRITICS.items()]
  res = {}
  lines = ['# Gate: DOWN-vs-RIGHT margin on the buffer-law fork -> jitter-(8,3) rows of replay_E', '',
           f'{len(rows):,} rows (Step 10b rows, seed 777); point = q(DOWN) - q(RIGHT); width = '
           f'Qbar(DOWN loc) - Qbar(RIGHT loc) at width {WIDTH}; PASS = width >= +{GATE_WIDTH:.2f} '
           '(D1\'s value).  Law target on these rows: +0.37 (all fork anchors) / +0.71 (550 contexts); '
           'strat_synth raises it to about +1.26 at the fork.  16-root columns as in Step 10\'s critic check.', '',
           '| critic | jitter point | jitter width (s.e.) | gate | 16 roots d-r | Qbar DOWN-CORNER | Qbar DOWN-RIGHT |',
           '|---|---:|---:|---|---:|---:|---:|']
  for label, path, arm in entries:
    if not path.exists():
      continue
    _, st = checkpoint.load_checkpoint(path)
    m = gate_metrics(st.q_params, network, rows, eps_jit)
    m.update(critic_metrics(st.q_params, network, roots, eps_root))
    m['pass'] = bool(m['jitter_width'] >= GATE_WIDTH)
    m['arm'] = arm
    res[label] = m
    lines.append(f'| {label} | {m["jitter_point"]:+.3f} | {m["jitter_width"]:+.3f} ({m["jitter_width_se"]:.3f}) '
                 f'| {"PASS" if m["pass"] else "fail"} | {m["down_minus_right"]:+.3f} '
                 f'| {m["qbar_down_minus_corner"]:+.3f} | {m["qbar_down_minus_right"]:+.3f} |')
    print(lines[-1], flush=True)
  for arm in arms:
    n_pass = sum(m['pass'] for m in res.values() if m['arm'] == arm)
    n_all = sum(1 for m in res.values() if m['arm'] == arm)
    lines.append(f'\n{arm}: {n_pass}/{n_all} critics pass the gate.')
  write_json(OUT / 'gate.json', res)
  (OUT / 'gate.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print('\n'.join(lines[-len(arms):]), flush=True)


# ------------------------------------------------------------------ actors
def passing_critics(arms, seeds, all_critics):
  gate = json.loads((OUT / 'gate.json').read_text(encoding='utf-8')) if (OUT / 'gate.json').exists() else {}
  chosen = []
  for arm in arms:
    for s in seeds:
      m = gate.get(f'{arm} seed {s}')
      if m is None:
        continue
      if all_critics or m['pass']:
        chosen.append((arm, s))
  return chosen


def stage_actors(arms, seeds, parallel, all_critics, actor_steps, episodes):
  chosen = passing_critics(arms, seeds, all_critics)
  print(f'actors for {chosen}', flush=True)
  procs = []
  for arm, s in chosen:
    for a in ACTOR_SEEDS:
      out = OUT / arm / f'seed_{s}' / f'actor_s{a}'
      if (out / 'final.pkl').exists():
        continue
      log = OUT / 'logs' / f'actor_{arm}_seed{s}_s{a}.log'
      log.parent.mkdir(parents=True, exist_ok=True)
      cmd = [sys.executable, str(ROOT / 'scripts' / 'train_f4_actor_fixed_critic.py'),
             '--critic', str(critic_path(arm, s)), '--replay', str(REPLAY_E),
             '--actor-seed', str(a), '--log-prob', 'acme', '--random-goals', '0',
             '--bc-sampling', 'balanced', '--bc-cap', str(BC_CAP), '--out', str(out)]
      if actor_steps:
        cmd += ['--steps', str(actor_steps)]
      f = open(log, 'w', encoding='utf-8')
      f.write(' '.join(cmd) + '\n')
      procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                     env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, out))
      if len(procs) >= parallel:
        _wait(procs)
  _wait(procs)
  for arm, s in chosen:
    for pol in ('mode', 'sample'):
      target = OUT / arm / f'seed_{s}' / f'native_new_{pol}'
      if (target / 'summary.json').exists():
        continue
      cmd = [sys.executable, '-m', 'scripts.eval_pointmaze_native_routes']
      for a in ACTOR_SEEDS:
        cmd += ['--ckpt', f'{arm}{s}_a{a}={OUT / arm / f"seed_{s}" / f"actor_s{a}" / "final.pkl"}']
      cmd += ['--episodes', episodes, '--policy', pol,
              '--reset-seed-base', NEW_EVAL['reset'], '--action-seed-base', NEW_EVAL['action'],
              '--out', target]
      run(cmd, OUT / 'logs' / f'eval_{arm}_seed{s}_{pol}.log')


# --------------------------------------------------------------- summarize
def stage_summarize(arms, seeds):
  import jax.numpy as jnp
  from crl import checkpoint
  network = make_network()
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  obs = jnp.asarray(np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1))
  gate = json.loads((OUT / 'gate.json').read_text(encoding='utf-8')) if (OUT / 'gate.json').exists() else {}
  lines = [TITLE, '',
           f'Batches of 256 = {COUNTS[0]} own-law anchors + {COUNTS[1]} fork-DOWN + {COUNTS[2]} fork-RIGHT '
           '(strat); strat_synth additionally drops the original episodes\' fork-cell rows.  Sealed 30k '
           'recipe otherwise; gate = width margin >= +0.10 on the Step 10b jitter rows; three fresh '
           'balanced-BC actors per passing critic; 300 new-seed episodes.  Family = fork mode under the '
           'task goal: DOWN (y < -0.9, x < 0.6), CORNER (y < -0.9, x >= 0.6), RIGHT (otherwise).', '',
           '| arm | critic | jitter width | gate | actor | fork mode | family | mode reach | mode lower | sample lower |',
           '|---|---|---:|---|---|---|---|---:|---:|---:|']
  families = []
  for arm in arms:
    for s in seeds:
      g = gate.get(f'{arm} seed {s}', {})
      sm = OUT / arm / f'seed_{s}' / 'native_new_mode' / 'summary.json'
      ss = OUT / arm / f'seed_{s}' / 'native_new_sample' / 'summary.json'
      if not sm.exists():
        if g:
          lines.append(f'| {arm} | seed {s} | {g["jitter_width"]:+.3f} | {"PASS" if g["pass"] else "fail"} '
                       '| -- | | | | | |')
        continue
      pm = json.loads(sm.read_text(encoding='utf-8'))['policies']
      ps = json.loads(ss.read_text(encoding='utf-8'))['policies']
      for a in ACTOR_SEEDS:
        _, st = checkpoint.load_checkpoint(OUT / arm / f'seed_{s}' / f'actor_s{a}' / 'final.pkl')
        m = np.tanh(np.asarray(network.policy_network.apply(st.policy_params, obs).loc)).mean(0)
        fam = 'DOWN' if (m[1] < -0.9 and m[0] < 0.6) else ('CORNER' if m[1] < -0.9 else 'RIGHT')
        key = f'{arm}{s}_a{a}'
        families.append({'arm': arm, 'critic': s, 'actor': a, 'family': fam,
                         'mode_lower': pm[key]['lower_route']['mean'], 'mode_reach': pm[key]['reach']['mean'],
                         'sample_lower': ps[key]['lower_route']['mean']})
        lines.append(f'| {arm} | seed {s} | {g.get("jitter_width", float("nan")):+.3f} '
                     f'| {"PASS" if g.get("pass") else "fail"} | a{a} | ({m[0]:+.2f}, {m[1]:+.2f}) | {fam} '
                     f'| {pm[key]["reach"]["mean"]:.3f} | {pm[key]["lower_route"]["mean"]:.3f} '
                     f'| {ps[key]["lower_route"]["mean"]:.3f} |')
  for arm in arms:
    fam = [f for f in families if f['arm'] == arm]
    if fam:
      lines.append(f'\n{arm}: DOWN-family actors {sum(f["family"] == "DOWN" for f in fam)}/{len(fam)}; '
                   f'mode lower >= 0.9: {sum(f["mode_lower"] >= 0.9 for f in fam)}/{len(fam)}.')
  lines.append('\nReference: E series 3/15 (DOWN only under E seed 2); D series 13/18 at >= 0.81.')
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  write_json(OUT / 'results.json', {'families': families, 'gate': gate, 'counts': COUNTS})
  print('\n'.join(lines), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('train', 'gate', 'actors', 'summarize', 'all', '_train_one'))
  ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
  ap.add_argument('--seeds', type=int, nargs='+', default=list(CRITIC_SEEDS))
  ap.add_argument('--steps', type=int, default=CRL_STEPS, help='critic updates (sealed: 30k)')
  ap.add_argument('--actor-steps', type=int, default=0, help='override the actor budget (smoke only)')
  ap.add_argument('--episodes', type=int, default=NEW_EVAL['episodes'])
  ap.add_argument('--parallel', type=int, default=3)
  ap.add_argument('--all-critics', action='store_true', help='train actors on every critic, not only the passing ones')
  args = ap.parse_args(argv)
  if args.command == '_train_one':
    train_one(args.arms[0], args.seeds[0], args.steps)
    return 0
  stages = ('train', 'gate', 'actors', 'summarize') if args.command == 'all' else (args.command,)
  for st in stages:
    print(f'=== {st}', flush=True)
    if st == 'train':
      stage_train(args.arms, args.seeds, args.steps, args.parallel)
    elif st == 'gate':
      stage_gate(args.arms, args.seeds)
    elif st == 'actors':
      stage_actors(args.arms, args.seeds, args.parallel, args.all_critics, args.actor_steps, args.episodes)
    elif st == 'summarize':
      stage_summarize(args.arms, args.seeds)
  return 0


if __name__ == '__main__':
  sys.exit(main())
