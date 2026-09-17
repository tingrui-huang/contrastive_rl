"""Step 13: joint critic + actor training on the branch replay, critic warm-started.

Steps 8-12 trained the actor against a FROZEN critic (two stages).  The
original contrastive RL learner updates critic and actor together at every
step on the same batch (crl/losses.py build_learner.update_step, the
Acme learner's schedule); Step 9 found that recipe fails from scratch
because the actor commits to RIGHT before the critic is ready.  This step
runs the original joint schedule with the critic warm-started: the critic
(and its Adam state) start from the Step 12 arm-P critic of the same seed
(30k updates on the branch replay, gate-passing), the actor starts fresh,
and both are then updated jointly for 300k steps -- the actor's budget of
the two-stage recipe, during which the critic keeps training.

Everything else is Step 12's: replay_P for the critic term and the anchor
rule (row 0 of every path), the actor's BC rows from replay_E's balanced
law (Config.bc_dataset, new: BC keeps imitating recorded actions while
the critic reads model futures), bc 0.05, Acme log-prob, random_goals 0,
the gate on the Step 10b jitter rows, the 300 new-seed episodes.
Checkpoints at 10k / 20k / 30k / 50k / 75k / 100k / 150k / 200k / 250k joint
steps are kept (critic and actor together), so the trajectory of both under
joint training can be read; seed 0 predates this and has warm + final only.

Seeds run one after another and each writes seed_<s>/summary.json when its
evaluation is done, so a watcher can report per seed.

Stages: run (prep + train + eval per seed) | summarize.
Outputs: outputs/pointmaze_joint_warm_v1/.
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
import run_f4_fork_strata as s11                                     # noqa: E402
import run_f4_branch_replay as s12                                   # noqa: E402
from run_f4_corner_coverage_fix import (                             # noqa: E402
    REPLAY_E, BC_COEF, BC_CAP, NEW_EVAL, GOAL, make_network, critic_metrics, sha256,
    write_json, run)

OUT = Path(os.environ.get('F4_JOINT_OUT', ROOT / 'outputs' / 'pointmaze_joint_warm_v1'))
REPLAY_P = s12.REPLAY_P
WARM_DIR = Path(os.environ.get('F4_WARM_DIR', ROOT / 'outputs' / 'pointmaze_branch_replay_v1' / 'P'))
JOINT_STEPS = 300_000
MILESTONES = tuple(int(m) for m in os.environ.get(
    'F4_JOINT_MILESTONES', '10000,20000,30000,50000,75000,100000,150000,200000,250000').split(','))
SEEDS = (0, 1, 2, 3, 4)


def warm_critic(seed):
  return WARM_DIR / f'seed_{seed}' / 'critic' / 'final.pkl'


def joint_dir(seed):
  return OUT / f'seed_{seed}' / 'joint'


def crl_config(seed, steps):
  from crl.config import Config
  return Config(
      env_name='point_two_route_swamp_windy_f4_v0', offline_dataset=str(REPLAY_P),
      obs_dim=8, goal_dim=8, action_dim=2, max_episode_steps=50,
      start_index=0, end_index=-1, max_number_of_steps=int(steps),
      fail_bank_path='', fail_neg_alpha=0.0, obs_norm_mode='',
      obs_norm_z_scale=0.0, anchor_cut_mode='', balanced_sampling=False,
      use_td=False, use_cpc=False, use_gcbc=False, twin_q=False,
      bc_coef=BC_COEF, random_goals=0.0, entropy_coefficient=0.0,
      target_entropy=0.0, batch_size=256, repr_dim=64,
      hidden_layer_sizes=(256, 256), discount=0.95, learning_rate=3e-4,
      actor_learning_rate=3e-4, num_sgd_steps_per_step=10, num_actors=0,
      guard_abort=True, jit=True, seed=int(seed),
      log_prob_mode='acme', bc_sampling='balanced', bc_balance_cap=BC_CAP,
      bc_dataset=str(REPLAY_E), resume=True,
      eval_every_steps=1_000_000, eval_episodes=50, log_every_steps=5_000,
      ckpt_every_steps=int(steps), ckpt_milestone_steps=tuple(m for m in MILESTONES if m < steps),
      ckpt_dir=str(joint_dir(seed)))


# -------------------------------------------------------------------- prep
def prep(seed):
  """latest.pkl = the warm critic (params, Adam state, target) + a fresh actor
  (params and Adam state initialised as build_learner.init_state does), at
  step 0, so crl.train's resume path runs the joint schedule from there."""
  import jax
  import optax
  from crl import checkpoint
  d = joint_dir(seed)
  if (d / 'latest.pkl').exists() or (d / 'final.pkl').exists():
    print(f'{d}: already prepared', flush=True)
    return
  nets = make_network()
  _, st = checkpoint.load_checkpoint(warm_critic(seed))
  key = jax.random.PRNGKey(20_000 + int(seed))
  k_pol, key = jax.random.split(key)
  policy_params = nets.policy_network.init(k_pol)
  pol_opt = optax.adam(3e-4, eps=1e-7).init(policy_params)
  new = st._replace(policy_params=policy_params, policy_optimizer_state=pol_opt, key=key)
  d.mkdir(parents=True, exist_ok=True)
  checkpoint.save_named(str(d), 'latest', 0, new)
  write_json(d / 'prep.json', {'seed': seed, 'warm_critic': str(warm_critic(seed)),
                               'warm_critic_sha256': sha256(warm_critic(seed)),
                               'actor': 'fresh init, PRNGKey(20000 + seed)', 'step': 0})
  print(f'prepared {d}', flush=True)


# ------------------------------------------------------------------- train
def train(seed, steps):
  from crl.train import train as crl_train
  d = joint_dir(seed)
  if (d / 'final.pkl').exists():
    print(f'{d}: final exists', flush=True)
    return
  cfg = crl_config(seed, steps)
  t0 = time.time()
  crl_train(cfg, buffer_prepare=s12.prepare_hook('P'))
  write_json(d / 'run_summary.json', {
      'seed': seed, 'joint_steps': int(steps), 'critic_replay': str(REPLAY_P),
      'critic_replay_sha256': sha256(REPLAY_P), 'bc_dataset': str(REPLAY_E),
      'warm_critic': str(warm_critic(seed)), 'milestones': list(MILESTONES),
      'wall_seconds': time.time() - t0})


# -------------------------------------------------------------------- eval
def evaluate(seed, episodes):
  import jax.numpy as jnp
  from crl import checkpoint
  d = joint_dir(seed)
  if (OUT / f'seed_{seed}' / 'summary.json').exists():
    print(f'seed {seed}: summary exists', flush=True)
    return
  network = make_network()
  rows = s11.jitter_rows()
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  obs = jnp.asarray(np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1))
  eps_root = np.random.default_rng(0).standard_normal((256, 2)).astype(np.float32)
  eps_jit = jnp.asarray(np.random.default_rng(777).standard_normal((256, 2)).astype(np.float32))
  critics = {'warm (0)': warm_critic(seed)}
  for ms in MILESTONES:
    p = d / f'{ms}.pkl'
    if p.exists():
      critics[f'{ms}'] = p
  critics['final'] = d / 'final.pkl'
  crit = {}
  for label, p in critics.items():
    _, st = checkpoint.load_checkpoint(p)
    m = s11.gate_metrics(st.q_params, network, rows, eps_jit)
    m.update(critic_metrics(st.q_params, network, roots, eps_root))
    m['pass'] = bool(m['jitter_width'] >= s11.GATE_WIDTH)
    if label != 'warm (0)':
      mo = np.tanh(np.asarray(network.policy_network.apply(st.policy_params, obs).loc)).mean(0)
      m['actor_fork_mode'] = mo.tolist()
      m['actor_family'] = ('DOWN' if (mo[1] < -0.9 and mo[0] < 0.6)
                           else ('CORNER' if mo[1] < -0.9 else 'RIGHT'))
    crit[label] = m
    print(f'seed {seed} critic {label}: jitter width {m["jitter_width"]:+.3f} '
          f'{"PASS" if m["pass"] else "fail"}'
          + (f'; actor mode ({m["actor_fork_mode"][0]:+.2f}, {m["actor_fork_mode"][1]:+.2f}) '
             f'{m["actor_family"]}' if 'actor_family' in m else ''), flush=True)
  native = {}
  for pol in ('mode', 'sample'):
    target = OUT / f'seed_{seed}' / f'native_new_{pol}'
    if not (target / 'summary.json').exists():
      cmd = [sys.executable, '-m', 'scripts.eval_pointmaze_native_routes',
             '--ckpt', f'J{seed}={d / "final.pkl"}', '--episodes', episodes, '--policy', pol,
             '--reset-seed-base', NEW_EVAL['reset'], '--action-seed-base', NEW_EVAL['action'],
             '--out', target]
      run(cmd, OUT / 'logs' / f'eval_seed{seed}_{pol}.log')
    pm = json.loads((target / 'summary.json').read_text(encoding='utf-8'))['policies'][f'J{seed}']
    native[pol] = {'reach': pm['reach']['mean'], 'lower': pm['lower_route']['mean']}
  summary = {'seed': seed, 'critics': crit, 'native': native}
  write_json(OUT / f'seed_{seed}' / 'summary.json', summary)
  line = (f'| {seed} | ' + ' / '.join(f'{crit[k]["jitter_width"]:+.2f}' for k in critics)
          + f' | ({crit["final"]["actor_fork_mode"][0]:+.2f}, {crit["final"]["actor_fork_mode"][1]:+.2f}) '
          f'{crit["final"]["actor_family"]} | {native["mode"]["reach"]:.3f} | {native["mode"]["lower"]:.3f} '
          f'| {native["sample"]["lower"]:.3f} |')
  with open(OUT / 'seed_lines.md', 'a', encoding='utf-8') as f:
    f.write(line + '\n')
  print(line, flush=True)


def stage_summarize(seeds):
  lines = ['# Step 13: joint training on the branch replay, critic warm-started at 30k', '',
           f'Critic warm start = Step 12 arm-P critic of the same seed (30k updates on replay_P); '
           f'actor fresh; then {JOINT_STEPS:,} joint updates (critic on replay_P row-0 anchors, actor '
           'critic term on the same batch, BC rows from replay_E balanced law, bc 0.05, Acme, rg 0). '
           'Critic column: jitter width margin at warm start / each milestone / final joint step '
           f'(milestones {list(MILESTONES)}; gate +0.10). 300 new-seed episodes.', '',
           '| seed | critic width: warm / milestones / final | actor fork mode, family | mode reach | mode lower | sample lower |',
           '|---|---|---|---:|---:|---:|']
  rows = []
  for s in seeds:
    p = OUT / f'seed_{s}' / 'summary.json'
    if not p.exists():
      lines.append(f'| {s} | pending | | | | |')
      continue
    sm = json.loads(p.read_text(encoding='utf-8'))
    rows.append(sm)
    cr = sm['critics']
    lines.append(f'| {s} | ' + ' / '.join(f'{cr[k]["jitter_width"]:+.2f}' for k in cr)
                 + f' | ({cr["final"]["actor_fork_mode"][0]:+.2f}, {cr["final"]["actor_fork_mode"][1]:+.2f}) '
                 f'{cr["final"]["actor_family"]} | {sm["native"]["mode"]["reach"]:.3f} '
                 f'| {sm["native"]["mode"]["lower"]:.3f} | {sm["native"]["sample"]["lower"]:.3f} |')
  if rows:
    n_down = sum(r['critics']['final']['actor_family'] == 'DOWN' for r in rows)
    n_ok = sum(r['native']['mode']['lower'] >= 0.9 for r in rows)
    lines += ['', f'DOWN-family actors {n_down}/{len(rows)}; mode lower >= 0.9: {n_ok}/{len(rows)}. '
              'Reference: two-stage on the same critics (Step 12 P) 15/15; joint from scratch on replay_D (Step 9) 0/6.']
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  write_json(OUT / 'results.json', {'seeds': rows})
  print('\n'.join(lines), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('command', choices=('run', 'summarize'))
  ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
  ap.add_argument('--steps', type=int, default=JOINT_STEPS)
  ap.add_argument('--episodes', type=int, default=NEW_EVAL['episodes'])
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  if args.command == 'run':
    for s in args.seeds:
      print(f'=== seed {s}', flush=True)
      prep(s)
      train(s, args.steps)
      evaluate(s, args.episodes)
      (OUT / 'logs').mkdir(parents=True, exist_ok=True)
      (OUT / 'logs' / f'MARK_SEED_{s}').write_text('done\n', encoding='utf-8')
  stage_summarize(args.seeds)
  return 0


if __name__ == '__main__':
  sys.exit(main())
