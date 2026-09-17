"""Step 9: attribution control and joint training for the balanced BC rows.

Step 8 (outputs/pointmaze_fixed_dcritic_actor_bcbal_v1) showed that drawing
the BC term's rows region-balanced inside (state cell, goal cell) groups turns
three fresh actors on the frozen D1 critic into detour actors (mode lower
route 1.000 x3).  Two questions remain, and this driver answers both:

* attribution -- the same balanced BC rows on the sealed C critics (seed 2 =
  the failed sealed seed, seed 0 = the successful one), three fresh actors
  each, everything else as Step 8 (D replay, Acme log-prob, random_goals 0,
  bc 0.05, 300k actor updates).  If those actors detour too, the gain does
  not depend on the repaired critic.
* joint -- the balancing rule wired into crl.train (config.bc_sampling): the
  NCE critic and the actor's critic term keep the buffer's batches, the BC
  term draws its own balanced rows; original NCE, bc 0.05, Acme log-prob,
  random_goals 0, three training seeds, arm 'shared' (same settings, BC rows
  = the critic batch) as the control.  The sealed D-arm budget is 30,000
  updates (max_number_of_steps 30000: 600 iterations x 50 updates), which
  is a tenth of what the fixed-critic actors of Steps 2-8 received (30k
  iterations x 10 = 300k); both budgets are run, the sealed one is primary.

Evaluation: 300 NEW paired native episodes (reset seeds from 31,500,000,
innovation seeds from 31,600,000 -- never used before) as the primary
measurement, plus the sealed 200-episode protocol (29.8M / 29.9M) for
continuity; mode and sample policies; the 16 held-out fork roots for the
jointly trained critics (DOWN-vs-RIGHT margin and where the argmax over the
action square physically leads).

Stages: attribution | train | evaluate | summarize | all.  Every stage skips
work whose outputs exist.
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
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
FIX = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
DIAG = ROOT / 'outputs' / 'pointmaze_actor_fork_diagnosis_v1'
OUT = ROOT / 'outputs' / 'pointmaze_balanced_bc_joint_v1'
REPLAY_D = FIX / 'replay_D.npz'
#: the plain-CRL baseline's data: the full 0.05-rung dataset behind the sealed O arm
#: (6600 episodes; the D replay is 3300 of these plus 6600 ETT paths)
O_DATASET = (ROOT / 'artifacts' / 'f4_p30_server_30076' / 'results' / 'datasets'
             / 'swamp_windy_f4_merged_s0.npz')
D1_CRITIC = FIX / 'seeds' / 'seed_1' / 'crl' / 'D' / 'final.pkl'
C_CRITICS = {2: SEALED / 'cross_seeds' / 'seed_2' / 'crl' / 'C' / 'final.pkl',
             0: SEALED / 'crl' / 'C' / 'final.pkl'}
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
BC_COEF = 0.05
BC_CAP = 0.25
ARMS = ('shared', 'balanced')
BUDGETS = {'b30k': 30_000, 'b300k': 300_000}
SEALED_EVAL = {'episodes': 200, 'reset': 29_800_000, 'action': 29_900_000}
NEW_EVAL = {'episodes': 300, 'reset': 31_500_000, 'action': 31_600_000}
PROTOCOLS = ('mode', 'sample')
ATTR_STEPS = None            # --smoke shrinks the budgets and the episode counts


def sha256(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  Path(path).write_text(json.dumps(obj, indent=2, default=float),
                        encoding='utf-8')


def run(cmd, log):
  Path(log).parent.mkdir(parents=True, exist_ok=True)
  with open(log, 'w', encoding='utf-8') as f:
    f.write(' '.join(str(c) for c in cmd) + '\n')
    f.flush()
    p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                       cwd=str(ROOT), env={**os.environ, 'PYTHONPATH': str(ROOT)})
  if p.returncode != 0:
    raise RuntimeError(f'{cmd[:3]} failed, see {log}')


def native_eval(ckpts, protocol, spec, out, log):
  if (Path(out) / 'summary.json').exists():
    return
  cmd = [sys.executable, '-m', 'scripts.eval_pointmaze_native_routes']
  for name, path in ckpts.items():
    cmd += ['--ckpt', f'{name}={path}']
  cmd += ['--episodes', spec['episodes'], '--policy', protocol,
          '--reset-seed-base', spec['reset'], '--action-seed-base', spec['action'],
          '--out', out]
  run(cmd, log)


# ------------------------------------------------------------ attribution
def stage_attribution(actor_seeds, critic_seeds, parallel):
  procs = []
  for cs in critic_seeds:
    for a in actor_seeds:
      out = OUT / 'attribution' / f'critic_C{cs}' / f'actor_s{a}'
      if (out / 'final.pkl').exists():
        continue
      log = OUT / 'logs' / f'attribution_C{cs}_actor_s{a}.log'
      log.parent.mkdir(parents=True, exist_ok=True)
      cmd = [sys.executable, str(ROOT / 'scripts' / 'train_f4_actor_fixed_critic.py'),
             '--critic', str(C_CRITICS[cs]), '--replay', str(REPLAY_D),
             '--actor-seed', str(a), '--log-prob', 'acme', '--random-goals', '0',
             '--bc-sampling', 'balanced', '--bc-cap', str(BC_CAP), '--out', str(out)]
      if ATTR_STEPS:
        cmd += ['--steps', str(ATTR_STEPS)]
      f = open(log, 'w', encoding='utf-8')
      f.write(' '.join(cmd) + '\n')
      procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                     cwd=str(ROOT),
                                     env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, out))
      if len(procs) >= parallel:
        _wait(procs)
  _wait(procs)
  for cs in critic_seeds:
    ckpts = {f'C{cs}_a{a}': OUT / 'attribution' / f'critic_C{cs}' / f'actor_s{a}' / 'final.pkl'
             for a in actor_seeds}
    for tag, spec in (('sealed', SEALED_EVAL), ('new', NEW_EVAL)):
      for pol in PROTOCOLS:
        native_eval(ckpts, pol, spec, OUT / 'attribution' / f'native_{tag}_{pol}_C{cs}',
                    OUT / 'logs' / f'attribution_eval_{tag}_{pol}_C{cs}.log')


def _wait(procs):
  for p, f, out in procs:
    rc = p.wait()
    f.close()
    if rc != 0:
      raise RuntimeError(f'run for {out} failed (rc {rc})')
  procs.clear()


# ------------------------------------------------------------------ train
def crl_config(seed, arm, budget, ckpt_dir, dataset=REPLAY_D):
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0', offline_dataset=str(dataset),
      obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
      start_index=0, end_index=-1, max_number_of_steps=int(budget),
      fail_bank_path='', fail_neg_alpha=0.0, obs_norm_mode='',
      obs_norm_z_scale=0.0, anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=BC_COEF, random_goals=0.0, entropy_coefficient=0.0,
      target_entropy=0.0, batch_size=256, repr_dim=64,
      hidden_layer_sizes=(256, 256), discount=0.95, learning_rate=3e-4,
      actor_learning_rate=3e-4, num_sgd_steps_per_step=10, num_actors=0,
      guard_abort=True, jit=True, seed=int(seed),
      log_prob_mode='acme', bc_sampling=arm, bc_balance_cap=BC_CAP,
      eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=5_000,
      ckpt_every_steps=int(budget), ckpt_dir=str(ckpt_dir))


def _train_one(seed, arm, budget_tag, control=False):
  from crl.train import train
  if control:
    out = OUT / 'control_O' / f'seed_{seed}' / 'critic'
    dataset = O_DATASET
  else:
    out = OUT / 'joint' / budget_tag / f'seed_{seed}' / arm
    dataset = REPLAY_D
  if (out / 'final.pkl').exists():
    print(f'{out} exists', flush=True)
    return
  cfg = crl_config(seed, arm, BUDGETS[budget_tag], out, dataset=dataset)
  t0 = time.time()
  train(cfg)
  write_json(out / 'run_summary.json', {
      'seed': seed, 'arm': arm, 'budget_updates': BUDGETS[budget_tag],
      'dataset': str(dataset), 'dataset_sha256': sha256(dataset),
      'bc_coef': BC_COEF, 'bc_cap': BC_CAP if arm == 'balanced' else None,
      'log_prob_mode': 'acme', 'random_goals': 0.0,
      'wall_seconds': time.time() - t0})


def stage_train(seeds, budgets, parallel):
  # each run is its own process so parallel runs share the GPU cleanly
  procs = []
  for budget_tag in budgets:
    for seed in seeds:
      for arm in ARMS:
        out = OUT / 'joint' / budget_tag / f'seed_{seed}' / arm
        if (out / 'final.pkl').exists():
          continue
        log = OUT / 'logs' / f'train_{budget_tag}_seed{seed}_{arm}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), '_train_one',
               '--seeds', str(seed), '--arm', arm, '--budgets', budget_tag]
        if ATTR_STEPS:
          cmd.append('--smoke')
        f = open(log, 'w', encoding='utf-8')
        procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                       cwd=str(ROOT),
                                       env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, out))
        if len(procs) >= parallel:
          _wait(procs)
  _wait(procs)


# ---------------------------------------------------------------- control
def stage_control(seeds, parallel):
  """The fair baseline: plain CRL critics (NCE on the full original dataset,
  sealed 30k budget; the critic never sees the actor) frozen, then three fresh
  actors each with the SAME balanced-BC rule on the original dataset and the
  same two-stage schedule; evaluated on the new seeds.  If these detour, the
  ETT replay is not what the gain comes from."""
  procs = []
  for seed in seeds:
    out = OUT / 'control_O' / f'seed_{seed}' / 'critic'
    if (out / 'final.pkl').exists():
      continue
    log = OUT / 'logs' / f'control_O_critic_seed{seed}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), '_train_one', '--control',
           '--seeds', str(seed), '--arm', 'shared', '--budgets', 'b30k']
    if ATTR_STEPS:
      cmd.append('--smoke')
    f = open(log, 'w', encoding='utf-8')
    procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                   env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, out))
    if len(procs) >= parallel:
      _wait(procs)
  _wait(procs)
  for seed in seeds:
    critic = OUT / 'control_O' / f'seed_{seed}' / 'critic' / 'final.pkl'
    for a in seeds:
      out = OUT / 'control_O' / f'seed_{seed}' / f'actor_s{a}'
      if (out / 'final.pkl').exists():
        continue
      log = OUT / 'logs' / f'control_O_seed{seed}_actor_s{a}.log'
      cmd = [sys.executable, str(ROOT / 'scripts' / 'train_f4_actor_fixed_critic.py'),
             '--critic', str(critic), '--replay', str(O_DATASET),
             '--actor-seed', str(a), '--log-prob', 'acme', '--random-goals', '0',
             '--bc-sampling', 'balanced', '--bc-cap', str(BC_CAP), '--out', str(out)]
      if ATTR_STEPS:
        cmd += ['--steps', str(ATTR_STEPS)]
      f = open(log, 'w', encoding='utf-8')
      f.write(' '.join(cmd) + '\n')
      procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                     env={**os.environ, 'PYTHONPATH': str(ROOT)}), f, out))
      if len(procs) >= parallel:
        _wait(procs)
  _wait(procs)
  for seed in seeds:
    ckpts = {f'O{seed}_a{a}': OUT / 'control_O' / f'seed_{seed}' / f'actor_s{a}' / 'final.pkl'
             for a in seeds}
    for pol in PROTOCOLS:
      native_eval(ckpts, pol, NEW_EVAL, OUT / 'control_O' / f'native_new_{pol}_seed{seed}',
                  OUT / 'logs' / f'control_O_eval_new_{pol}_seed{seed}.log')


# --------------------------------------------------------------- evaluate
def heldout_critic(final_state, network):
  """DOWN-vs-RIGHT margin and argmax physics on the 16 held-out fork roots."""
  import jax
  import jax.numpy as jnp
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  obs = jnp.asarray(np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1))
  with np.load(DIAG / 'seed0_heldout_score_maps.npz') as maps:
    outcome_maps = maps['outcome_maps']
    axis = maps['grid_axis'].astype(np.float32)
  gx, gy = np.meshgrid(axis, axis, indexing='xy')
  grid = np.stack([gx.ravel(), gy.ravel()], 1)

  @jax.jit
  def score(q_params, o, a):
    phi, psi = network.representation_network.apply(q_params, o, a)
    return jnp.sum(phi * psi, axis=1)[:, 0]

  down = np.asarray(score(final_state.q_params, obs, jnp.asarray(np.tile([[0., -1.]], (16, 1)), jnp.float32)))
  right = np.asarray(score(final_state.q_params, obs, jnp.asarray(np.tile([[1., 0.]], (16, 1)), jnp.float32)))
  region, actions = [], []
  for i in range(16):
    o_rep = jnp.asarray(np.repeat(np.asarray(obs[i:i + 1]), len(grid), 0))
    qg = np.asarray(score(final_state.q_params, o_rep, jnp.asarray(grid)))
    k = int(np.argmax(qg))
    actions.append(grid[k])
    region.append(int(outcome_maps[i].ravel()[k]))
  region = np.asarray(region)
  return {'down_minus_right_mean': float((down - right).mean()),
          'roots_preferring_down': int((down > right).sum()),
          'argmax_mean_action': np.mean(actions, 0).tolist(),
          'argmax_region_lower': int((region == 1).sum()),
          'argmax_region_shortcut': int((region == 2).sum()),
          'argmax_region_other': int((region == 0).sum())}


def stage_evaluate(seeds, budgets):
  from crl import checkpoint, networks
  network = networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None)
  for budget_tag in budgets:
    ckpts = {}
    crit = {}
    for seed in seeds:
      for arm in ARMS:
        path = OUT / 'joint' / budget_tag / f'seed_{seed}' / arm / 'final.pkl'
        ckpts[f'{arm}_s{seed}'] = path
        _, st = checkpoint.load_checkpoint(path)
        crit[f'{arm}_s{seed}'] = heldout_critic(st, network)
    write_json(OUT / 'joint' / f'heldout_critic_{budget_tag}.json', crit)
    for tag, spec in (('new', NEW_EVAL), ('sealed', SEALED_EVAL)):
      for pol in PROTOCOLS:
        native_eval(ckpts, pol, spec, OUT / 'joint' / f'native_{tag}_{pol}_{budget_tag}',
                    OUT / 'logs' / f'joint_eval_{tag}_{pol}_{budget_tag}.log')


# -------------------------------------------------------------- summarize
def _pol(summary_path, name):
  s = json.loads(Path(summary_path).read_text(encoding='utf-8'))['policies'][name]
  return {k: s[k]['mean'] for k in ('reach', 'lower_route', 'absorbed', 'shortcut')}


def stage_summarize(seeds, budgets, critic_seeds):
  lines = ['# Balanced BC rows: attribution on the C critics and joint training', '',
           'Balanced BC rows = `GroupBalancedBCSampler` (D replay, (state cell, goal cell) '
           'groups, 8 sectors + wait, share cap 0.25); Acme log-prob, random_goals 0, '
           'bc 0.05 everywhere. New = 300 paired native episodes on reset seeds from '
           '31.5M (never used before); sealed = the 200-episode protocol of Steps 4-8.', '']
  results = {'attribution': {}, 'joint': {}}
  # attribution
  lines += ['## Attribution: balanced BC rows on the C critics (fixed critic, 300k actor updates)', '',
            '| critic | actor seed | new mode reach | new mode lower | new sample reach | new sample lower '
            '| sealed mode reach | sealed mode lower | sealed sample lower |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
  ref = OUT.parent / 'pointmaze_fixed_dcritic_actor_bcbal_v1' / 'critic_D1'
  for cs in critic_seeds:
    for a in seeds:
      name = f'C{cs}_a{a}'
      try:
        r = {f'{tag}_{pol}': _pol(OUT / 'attribution' / f'native_{tag}_{pol}_C{cs}' / 'summary.json', name)
             for tag in ('new', 'sealed') for pol in PROTOCOLS}
      except FileNotFoundError:
        continue
      results['attribution'][name] = r
      lines.append(f'| C seed {cs} | {a} | {r["new_mode"]["reach"]:.3f} | {r["new_mode"]["lower_route"]:.3f} '
                   f'| {r["new_sample"]["reach"]:.3f} | {r["new_sample"]["lower_route"]:.3f} '
                   f'| {r["sealed_mode"]["reach"]:.3f} | {r["sealed_mode"]["lower_route"]:.3f} '
                   f'| {r["sealed_sample"]["lower_route"]:.3f} |')
  for a in seeds:
    try:
      m = _pol(ref / 'native_mode_balanced' / 'summary.json', f'balanced_a{a}')
      s = _pol(ref / 'native_sample_balanced' / 'summary.json', f'balanced_a{a}')
      lines.append(f'| D1 (Step 8) | {a} | | | | | {m["reach"]:.3f} | {m["lower_route"]:.3f} | {s["lower_route"]:.3f} |')
    except FileNotFoundError:
      pass
  # joint
  for budget_tag in budgets:
    crit_path = OUT / 'joint' / f'heldout_critic_{budget_tag}.json'
    if not crit_path.exists():
      continue
    crit = json.loads(crit_path.read_text(encoding='utf-8'))
    lines += ['', f'## Joint training, {BUDGETS[budget_tag]:,} updates '
              f'({"the sealed D-arm budget" if budget_tag == "b30k" else "10x the sealed budget"})', '',
              '| arm | seed | new mode reach | new mode lower | new sample reach | new sample lower '
              '| sealed mode reach | sealed mode lower | critic d-r | roots down | argmax -> shortcut / lower |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    agg = {arm: [] for arm in ARMS}
    for arm in ARMS:
      for seed in seeds:
        name = f'{arm}_s{seed}'
        r = {f'{tag}_{pol}': _pol(OUT / 'joint' / f'native_{tag}_{pol}_{budget_tag}' / 'summary.json', name)
             for tag in ('new', 'sealed') for pol in PROTOCOLS}
        c = crit[name]
        results['joint'][f'{budget_tag}_{name}'] = {**r, 'critic': c}
        agg[arm].append(r)
        lines.append(f'| {arm} | {seed} | {r["new_mode"]["reach"]:.3f} | {r["new_mode"]["lower_route"]:.3f} '
                     f'| {r["new_sample"]["reach"]:.3f} | {r["new_sample"]["lower_route"]:.3f} '
                     f'| {r["sealed_mode"]["reach"]:.3f} | {r["sealed_mode"]["lower_route"]:.3f} '
                     f'| {c["down_minus_right_mean"]:+.3f} | {c["roots_preferring_down"]}/16 '
                     f'| {c["argmax_region_shortcut"]} / {c["argmax_region_lower"]} |')
      xs = agg[arm]
      lines.append(f'| {arm} | mean | {np.mean([x["new_mode"]["reach"] for x in xs]):.3f} '
                   f'| {np.mean([x["new_mode"]["lower_route"] for x in xs]):.3f} '
                   f'| {np.mean([x["new_sample"]["reach"] for x in xs]):.3f} '
                   f'| {np.mean([x["new_sample"]["lower_route"] for x in xs]):.3f} '
                   f'| {np.mean([x["sealed_mode"]["reach"] for x in xs]):.3f} '
                   f'| {np.mean([x["sealed_mode"]["lower_route"] for x in xs]):.3f} | | | |')
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  write_json(OUT / 'results.json', results)
  print('\n'.join(lines), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('attribution', 'train', 'evaluate',
                                      'control', 'summarize', 'all', '_train_one'))
  ap.add_argument('--control', action='store_true', help='_train_one only')
  ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
  ap.add_argument('--critic-seeds', type=int, nargs='+', default=[2, 0])
  ap.add_argument('--budgets', nargs='+', default=list(BUDGETS), choices=list(BUDGETS))
  ap.add_argument('--arm', choices=ARMS, help='_train_one only')
  ap.add_argument('--parallel', type=int, default=3, help='concurrent runs per stage')
  ap.add_argument('--smoke', action='store_true',
                  help='tiny budgets and episode counts under outputs/.../_smoke')
  args = ap.parse_args(argv)
  if args.smoke:
    global OUT, ATTR_STEPS
    OUT = OUT / '_smoke'
    ATTR_STEPS = 20
    BUDGETS.update({'b30k': 100, 'b300k': 200})
    SEALED_EVAL.update({'episodes': 4})
    NEW_EVAL.update({'episodes': 4})
  if args.command == '_train_one':
    _train_one(args.seeds[0], args.arm, args.budgets[0], control=args.control)
    return 0
  stages = (('attribution', 'train', 'evaluate', 'summarize')
            if args.command == 'all' else (args.command,))
  for st in stages:
    print(f'=== {st}', flush=True)
    if st == 'attribution':
      stage_attribution(args.seeds, args.critic_seeds, args.parallel)
    elif st == 'train':
      stage_train(args.seeds, args.budgets, args.parallel)
    elif st == 'evaluate':
      stage_evaluate(args.seeds, args.budgets)
    elif st == 'control':
      stage_control(args.seeds, args.parallel)
    elif st == 'summarize':
      stage_summarize(args.seeds, args.budgets, args.critic_seeds)
  return 0


if __name__ == '__main__':
  sys.exit(main())
