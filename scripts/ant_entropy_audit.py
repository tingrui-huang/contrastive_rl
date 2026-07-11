"""Checkpoint-only entropy-collapse audit for the AntMaze 150k run (no training).

Tests the hypothesis: entropy_coefficient=0.0 let the tanh-Gaussian policy
collapse (scale -> 0, mode -> saturated constant pose), which killed action
diversity in the replay and hence the critic's action-conditioned signal.

Per checkpoint (init 0 / early 42k / mid 84k / final 150.5k):
  A. policy-head stats on one COMMON diverse probe-observation set:
     pre-tanh loc, scale (std), mode & sampled saturation, |sample-mode|,
     entropy estimate (-E log pi), across-state mode-action covariance;
  B. 20 deterministic + 20 stochastic eval episodes (identical reset seeds
     across checkpoints): displacement, speed, static fraction, fall fraction,
     saturation, coverage, goal progress, success;
  C. deterministic vs stochastic 100-step rollouts from the SAME saved
     reference states (the 20 unique states of the probe npz);
  D. collapse timing vs metrics.json (saturation / logits_gap / actor_loss /
     goal velocity every ~10.5k steps).

No training, no env or loss modifications.
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from ant_action_validity import restore

from crl.config import Config
from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod

CKPT_DIR = 'D:/Users/trhua/Research/contrastive_rl/antmaze_umaze_s0'
NPZ = 'D:/Users/trhua/Research/contrastive_rl/artifacts/ant_action_validity/ant_action_validity_samples.npz'
OUT = 'D:/Users/trhua/Research/contrastive_rl/artifacts/ant_entropy_audit'
CKPTS = [('init', 'init.pkl'), ('early', 'early.pkl'), ('mid', 'mid.pkl'),
         ('final', 'latest.pkl')]
SAT = 0.99
N_EVAL_EPS = 20
H_STATE = 100
N_STOCH_REPS = 3
K_ENT = 16
FALL_Z = 0.3
STATIC_EPS = 1e-4


def make_policy(nets, policy_params):
  apply_ = jax.jit(lambda obs: nets.policy_network.apply(policy_params, obs))
  sample_ = jax.jit(lambda obs, key: nets.sample(
      nets.policy_network.apply(policy_params, obs), key))
  logp_ = jax.jit(lambda obs, act: nets.log_prob(
      nets.policy_network.apply(policy_params, obs), act))

  def dist(obs_b):                     # [N,D] -> loc [N,A], scale [N,A]
    p = apply_(jnp.asarray(obs_b))
    return np.asarray(p.loc), np.asarray(p.scale)

  def mode(obs):
    p = apply_(jnp.asarray(obs[None]))
    return np.asarray(jnp.tanh(p.loc)[0])

  def sample(obs, key):
    return np.asarray(sample_(jnp.asarray(obs[None]), key)[0])

  def sample_batch(obs_b, key):
    return np.asarray(sample_(jnp.asarray(obs_b), key))

  def log_prob(obs_b, act_b):
    return np.asarray(logp_(jnp.asarray(obs_b), jnp.asarray(act_b)))
  return dict(dist=dist, mode=mode, sample=sample, sample_batch=sample_batch,
              log_prob=log_prob)


def eff_rank(X):
  if len(X) < 3:
    return float('nan')
  ev = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None)
  s = ev.sum()
  return float(s * s / (np.square(ev).sum() + 1e-18)) if s > 0 else 0.0


# --------------------------------------------------------------------------- #
def reset_seeded(env, seed):
  obs_d, _ = env._env.reset(seed=seed)
  return env._flatten(obs_d)


def collect_probe_obs(env, policies, rng):
  """Common probe-observation set: random-policy states + each checkpoint's
  deterministic visitation, all with per-episode randomized start/goal."""
  obs_set, src = [], []
  for ep in range(8):                              # uniform-random policy
    obs = reset_seeded(env, 2000 + ep)
    for t in range(env.max_episode_steps):
      if t % 10 == 0:
        obs_set.append(obs.copy()); src.append('random')
      obs, _, _, _ = env.step(rng.uniform(-1, 1, env.action_dim).astype(np.float32))
  for name, pol in policies.items():               # each checkpoint's mode
    for ep in range(4):
      obs = reset_seeded(env, 3000 + ep)
      for t in range(env.max_episode_steps):
        if t % 10 == 0:
          obs_set.append(obs.copy()); src.append(name)
        obs, _, _, _ = env.step(pol['mode'](obs))
  obs_set = np.array(obs_set, np.float32)
  keep = rng.permutation(len(obs_set))[:600]
  return obs_set[keep], [src[i] for i in keep]


def policy_head_stats(pol, obs_b, key):
  loc, scale = pol['dist'](obs_b)
  mode = np.tanh(loc)
  ent, samp_sat, samp_dist = [], [], []
  for k in range(K_ENT):
    key, sk = jax.random.split(key)
    a = pol['sample_batch'](obs_b, sk)
    lp = pol['log_prob'](obs_b, a)
    ent.append(-lp)
    samp_sat.append(np.mean(np.abs(a) > SAT))
    samp_dist.append(np.linalg.norm(a - mode, axis=1))
  return {
      'loc_abs_mean': float(np.mean(np.abs(loc))),
      'loc_abs_p90': float(np.percentile(np.abs(loc), 90)),
      'scale_median': float(np.median(scale)),
      'scale_mean': float(np.mean(scale)),
      'scale_p10': float(np.percentile(scale, 10)),
      'scale_p90': float(np.percentile(scale, 90)),
      'scale_per_dim_median': np.median(scale, axis=0).tolist(),
      'mode_sat': float(np.mean(np.abs(mode) > SAT)),
      'sample_sat': float(np.mean(samp_sat)),
      'entropy_per_action_mean': float(np.mean(ent)),
      'sample_minus_mode_dist_mean': float(np.mean(samp_dist)),
      'mode_across_state_std_per_dim': mode.std(0).tolist(),
      'mode_across_state_std_mean': float(mode.std(0).mean()),
      'mode_eff_rank_across_states': eff_rank(mode),
      'mode_per_dim_mean': mode.mean(0).tolist(),
  }, scale, mode


def run_episode(env, pol, stochastic, ep_seed, key):
  u = env._env.unwrapped
  obs = reset_seeded(env, ep_seed)
  goal = obs[env.obs_dim:env.obs_dim + 2].copy()
  xy_start = obs[:2].copy()
  xy_prev = xy_start.copy()
  d0 = float(np.linalg.norm(xy_prev - goal))
  dmin, success, path_len, static, fall, sats = d0, 0.0, 0.0, 0, 0, []
  cells = set()
  for t in range(env.max_episode_steps):
    if stochastic:
      key, sk = jax.random.split(key)
      a = pol['sample'](obs, sk)
    else:
      a = pol['mode'](obs)
    sats.append(float(np.mean(np.abs(a) > SAT)))
    obs, rew, _, _ = env.step(a)
    xy = obs[:2].copy()
    step_d = float(np.linalg.norm(xy - xy_prev))
    path_len += step_d
    static += step_d < STATIC_EPS
    dmin = min(dmin, float(np.linalg.norm(xy - goal)))
    success = max(success, rew)
    fall += float(u.data.qpos[2]) < FALL_Z
    cells.add((round(xy[0] / 0.5), round(xy[1] / 0.5)))
    xy_prev = xy
  T = env.max_episode_steps
  return dict(success=success, d0=d0, dmin=dmin,
              final_dist=float(np.linalg.norm(xy_prev - goal)),
              progress=d0 - dmin,
              net_disp=float(np.linalg.norm(xy_prev - xy_start)),
              path_len=path_len, speed=path_len / T,
              static_frac=static / T, fall_frac=fall / T,
              sat=float(np.mean(sats)), cells=len(cells))


def rollout_from_state(env, u, ref, pol, stochastic, key, H=H_STATE):
  restore(u, ref['qpos'], ref['qvel'])
  u.goal = np.asarray(ref['goal'], float)
  mujoco.mj_forward(u.model, u.data)
  obs = ref['obs'].copy()
  goal = ref['goal']
  xy0 = ref['qpos'][:2].copy()
  d0 = float(np.linalg.norm(xy0 - goal))
  dmin, path_len, minz = d0, 0.0, float(ref['qpos'][2])
  xy_prev = xy0
  for t in range(H):
    if stochastic:
      key, sk = jax.random.split(key)
      a = pol['sample'](obs, sk)
    else:
      a = pol['mode'](obs)
    od = u.step(np.asarray(a, np.float32))[0]
    obs = env._flatten(od)
    xy = np.asarray(u.data.qpos[:2]).copy()
    path_len += float(np.linalg.norm(xy - xy_prev))
    dmin = min(dmin, float(np.linalg.norm(xy - goal)))
    minz = min(minz, float(u.data.qpos[2]))
    xy_prev = xy
  return dict(net_disp=float(np.linalg.norm(xy_prev - xy0)),
              path_len=path_len, progress=d0 - dmin,
              fell=bool(minz < FALL_Z))


# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt_dir', default=CKPT_DIR)
  ap.add_argument('--npz', default=NPZ)
  ap.add_argument('--out', default=OUT)
  ap.add_argument('--n_eval_eps', type=int, default=N_EVAL_EPS)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(0)

  cfg = Config(env_name='antmaze_umaze')
  env = envs_mod.make_env('antmaze_umaze', cfg, seed=11)
  u = env._env.unwrapped
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)

  policies, steps = {}, {}
  for name, fn in CKPTS:
    step, state = ckpt_mod.load_checkpoint(os.path.join(args.ckpt_dir, fn))
    policies[name] = make_policy(nets, state.policy_params)
    steps[name] = int(step)
  print('checkpoints:', steps)

  # unique saved reference states (dedup the known-redundant npz)
  d = np.load(args.npz)
  full = np.round(np.hstack([d['qpos'], d['qvel']]), 8)
  _, first_idx = np.unique(full, axis=0, return_index=True)
  refs = [dict(qpos=d['qpos'][k], qvel=d['qvel'][k], goal=d['goal'][k],
               obs=d['obs'][k]) for k in sorted(first_idx)]
  print(f'{len(refs)} unique saved reference states')

  print('collecting common probe-observation set...')
  probe_obs, probe_src = collect_probe_obs(env, policies, rng)
  from collections import Counter
  print(f'  {len(probe_obs)} obs; sources: {dict(Counter(probe_src))}')

  key = jax.random.PRNGKey(0)
  report = {'ckpt_steps': steps, 'n_probe_obs': int(len(probe_obs)),
            'probe_sources': dict(Counter(probe_src)),
            'n_unique_saved_states': len(refs),
            'per_ckpt': {}}
  scales_by_ckpt, modes_by_ckpt = {}, {}

  for name in policies:
    pol = policies[name]
    print(f'--- {name} (step {steps[name]}) ---')
    key, k1 = jax.random.split(key)
    head, scale, mode = policy_head_stats(pol, probe_obs, k1)
    scales_by_ckpt[name], modes_by_ckpt[name] = scale, mode

    evals = {}
    for stoch in (False, True):
      rows = []
      for ep in range(args.n_eval_eps):
        key, k2 = jax.random.split(key)
        rows.append(run_episode(env, pol, stoch, 5000 + ep, k2))
      evals['stochastic' if stoch else 'deterministic'] = {
          q: float(np.mean([r[q] for r in rows]))
          for q in ('success', 'dmin', 'final_dist', 'progress', 'path_len',
                    'speed', 'static_frac', 'fall_frac', 'sat', 'cells', 'd0')}

    ss = {'deterministic': [], 'stochastic': []}
    for ref in refs:
      key, k3 = jax.random.split(key)
      ss['deterministic'].append(rollout_from_state(env, u, ref, pol, False, k3))
      reps = []
      for _ in range(N_STOCH_REPS):
        key, k4 = jax.random.split(key)
        reps.append(rollout_from_state(env, u, ref, pol, True, k4))
      ss['stochastic'].append({q: float(np.mean([r[q] for r in reps]))
                               for q in ('net_disp', 'path_len', 'progress', 'fell')})
    ss_agg = {m: {q: float(np.mean([r[q] for r in ss[m]]))
                  for q in ('net_disp', 'path_len', 'progress', 'fell')}
              for m in ss}
    report['per_ckpt'][name] = {'step': steps[name], 'policy_head': head,
                                'eval_episodes': evals,
                                'saved_state_rollouts_100steps': ss_agg}
    print(f'  scale med {head["scale_median"]:.3f} | mode sat {head["mode_sat"]:.2f} | '
          f'det speed {evals["deterministic"]["speed"]:.4f} | '
          f'stoch speed {evals["stochastic"]["speed"]:.4f}')

  # ---- D: metrics.json timeline ----
  hist = json.load(open(os.path.join(args.ckpt_dir, 'metrics.json')))
  tl = {q: [e.get(q) for e in hist] for q in
        ('step', 'ant_action_saturation', 'ant_goal_velocity', 'logits_gap',
         'categorical_accuracy', 'actor_loss', 'critic_loss', 'min_dist',
         'ant_fall_fraction', 'ant_torso_height')}
  report['metrics_timeline'] = tl
  sat_arr = np.array(tl['ant_action_saturation'], float)
  step_arr = np.array(tl['step'], float)
  gap_arr = np.array(tl['logits_gap'], float)
  first_sat = step_arr[sat_arr > 0.5][0] if (sat_arr > 0.5).any() else None
  first_gap = step_arr[gap_arr > 20][0] if (gap_arr > 20).any() else None
  report['timing'] = {
      'first_eval_saturation_gt_0.5': first_sat,
      'first_eval_logits_gap_gt_20': first_gap,
      'note': 'evals every ~10.5k env steps; checkpoints at 0/42k/84k/150.5k'}

  # ---- hypothesis verdict ----
  pc = report['per_ckpt']
  init_scale = pc['init']['policy_head']['scale_median']
  fin = pc['final']
  fin_scale = fin['policy_head']['scale_median']
  h1_scale_collapse = bool(fin_scale < 0.2 and fin_scale < 0.5 * init_scale)
  h1_sat = bool(fin['policy_head']['mode_sat'] > 0.4
                and pc['init']['policy_head']['mode_sat'] < 0.1)
  det = fin['eval_episodes']['deterministic']
  sto = fin['eval_episodes']['stochastic']
  ssd = fin['saved_state_rollouts_100steps']
  h2_static_mode = bool(det['static_frac'] > 0.3 or det['speed'] < 0.005)
  h2_stoch_moves = bool(sto['path_len'] > 2 * det['path_len']
                        or ssd['stochastic']['path_len']
                        > 2 * ssd['deterministic']['path_len'])
  # TOTAL collapse: sigma ~ 0, so sampled == mode at final. The stochastic-vs-
  # deterministic equivalence is then the *signature* of collapse, not evidence
  # against it; the counterfactual is the init policy, whose entropy moved the
  # ant far more when sampled.
  fh = fin['policy_head']
  total_collapse = bool(fh['scale_median'] < 0.05
                        and fh['sample_minus_mode_dist_mean'] < 0.1)
  ie = pc['init']['eval_episodes']
  init_stoch_gain = bool(ie['stochastic']['path_len']
                         > 2 * ie['deterministic']['path_len'])
  order = ['init', 'early', 'mid', 'final']
  scale_series = [pc[n]['policy_head']['scale_median'] for n in order]
  h3_monotone = bool(all(scale_series[i + 1] <= scale_series[i] * 1.05
                         for i in range(3)))
  supported = h1_scale_collapse or h1_sat
  behavioral = ((h2_static_mode and h2_stoch_moves)
                or (total_collapse and init_stoch_gain))
  if supported and behavioral:
    verdict = 'ENTROPY_COLLAPSE_SUPPORTED'
  elif supported or behavioral:
    verdict = 'ENTROPY_COLLAPSE_PARTIALLY_SUPPORTED'
  else:
    verdict = 'ENTROPY_COLLAPSE_NOT_SUPPORTED'
  report['hypothesis'] = {
      'h1_scale_collapse': h1_scale_collapse, 'h1_mode_saturation': h1_sat,
      'h2_deterministic_static': h2_static_mode,
      'h2_stochastic_moves_2x': h2_stoch_moves,
      'h2_total_scale_collapse': total_collapse,
      'h2_init_stochastic_gain_2x': init_stoch_gain,
      'h3_scale_monotone_decreasing': h3_monotone,
      'scale_median_series': dict(zip(order, scale_series)),
      'verdict': verdict}

  json.dump(report, open(os.path.join(args.out, 'ant_entropy_audit.json'), 'w'),
            indent=2)
  _plots(args.out, report, scales_by_ckpt, modes_by_ckpt, order)
  _md(args.out, report, order)
  print('\nVERDICT:', verdict)


# --------------------------------------------------------------------------- #
def _plots(out, r, scales, modes, order):
  # 1: scale distributions per checkpoint
  fig, ax = plt.subplots(figsize=(7, 4))
  ax.boxplot([scales[n].ravel() for n in order], tick_labels=[
      f'{n}\n{r["ckpt_steps"][n] // 1000}k' for n in order], showfliers=False)
  ax.set_ylabel('policy scale (std) over probe obs x dims')
  ax.set_title('tanh-Gaussian scale by checkpoint')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'scale_by_checkpoint.png'),
                                  dpi=100); plt.close()
  # 2: mode action per-dim distribution per checkpoint
  fig, axes = plt.subplots(1, len(order), figsize=(15, 3.2), sharey=True)
  for ax, n in zip(axes, order):
    ax.boxplot([modes[n][:, i] for i in range(modes[n].shape[1])],
               showfliers=False)
    ax.axhline(0, color='k', lw=.4)
    ax.set_ylim(-1.05, 1.05); ax.set_title(f'{n}'); ax.set_xlabel('action dim')
  axes[0].set_ylabel('mode action (tanh loc)')
  fig.suptitle('mode action per dim across probe states (braced pose = tight bands at +-1)')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'mode_actions_by_checkpoint.png'),
                                  dpi=100); plt.close()
  # 3: det vs stoch behavior bars
  fig, axes = plt.subplots(1, 3, figsize=(13, 4))
  x = np.arange(len(order))
  for ax, q, title in zip(
      axes, ('path_len', 'progress', 'static_frac'),
      ('episode path length (m)', 'goal progress d0-dmin (m)', 'static fraction')):
    det = [r['per_ckpt'][n]['eval_episodes']['deterministic'][q] for n in order]
    sto = [r['per_ckpt'][n]['eval_episodes']['stochastic'][q] for n in order]
    ax.bar(x - 0.17, det, 0.34, label='deterministic')
    ax.bar(x + 0.17, sto, 0.34, label='stochastic')
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_title(title)
  axes[0].legend()
  fig.suptitle('eval episodes: mode vs sampled policy (same 20 reset seeds)')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'det_vs_stoch_eval.png'),
                                  dpi=100); plt.close()
  # 4: metrics timeline
  tl = r['metrics_timeline']
  fig, ax1 = plt.subplots(figsize=(8, 4.5))
  s = np.array(tl['step']) / 1000
  ax1.plot(s, tl['ant_action_saturation'], 'o-', color='tab:red',
           label='action saturation (eval)')
  ax1.plot(s, np.array(tl['categorical_accuracy']), 's-', color='tab:purple',
           label='categorical accuracy')
  ax1.set_xlabel('env steps (k)'); ax1.set_ylim(0, 1)
  ax1.set_ylabel('saturation / accuracy')
  ax2 = ax1.twinx()
  ax2.plot(s, tl['logits_gap'], '^-', color='tab:blue', label='logits gap')
  ax2.plot(s, tl['actor_loss'], 'v-', color='tab:green', label='actor loss')
  ax2.set_ylabel('logits gap / actor loss')
  for name in ('early', 'mid'):
    ax1.axvline(r['ckpt_steps'][name] / 1000, color='gray', ls=':', lw=1)
  h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
  ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc='center right')
  ax1.set_title('collapse timing: saturation vs critic-gap growth')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'timeline.png'), dpi=100)
  plt.close()


def _md(out, r, order):
  pc = r['per_ckpt']
  h = r['hypothesis']
  L = ['# AntMaze entropy-collapse checkpoint audit\n',
       f'**Verdict: `{h["verdict"]}`**\n',
       f'Checkpoints: ' + ', '.join(f'{n}={r["ckpt_steps"][n]}' for n in order)
       + f'; common probe set {r["n_probe_obs"]} obs '
       f'({r["probe_sources"]}); {r["n_unique_saved_states"]} unique saved states.\n',
       '## A. Policy head on the common probe set',
       '| ckpt | scale med | scale p10-p90 | |loc| mean | mode sat | sample sat | '
       '|sample-mode| | entropy est | mode std across states | mode eff rank |',
       '|---|---|---|---|---|---|---|---|---|---|']
  for n in order:
    p = pc[n]['policy_head']
    L.append(f'| {n} | {p["scale_median"]:.3f} | {p["scale_p10"]:.3f}-'
             f'{p["scale_p90"]:.3f} | {p["loc_abs_mean"]:.2f} | '
             f'{p["mode_sat"]:.3f} | {p["sample_sat"]:.3f} | '
             f'{p["sample_minus_mode_dist_mean"]:.3f} | '
             f'{p["entropy_per_action_mean"]:.2f} | '
             f'{p["mode_across_state_std_mean"]:.3f} | '
             f'{p["mode_eff_rank_across_states"]:.2f} |')
  L += ['\n## B. Eval episodes (20 eps, same reset seeds; 700 steps)',
        '| ckpt | mode | success | progress | path len | speed | static frac | '
        'fall frac | sat | cells |',
        '|---|---|---|---|---|---|---|---|---|---|']
  for n in order:
    for m in ('deterministic', 'stochastic'):
      e = pc[n]['eval_episodes'][m]
      L.append(f'| {n} | {m} | {e["success"]:.2f} | {e["progress"]:.3f} | '
               f'{e["path_len"]:.1f} | {e["speed"]:.4f} | {e["static_frac"]:.2f} | '
               f'{e["fall_frac"]:.2f} | {e["sat"]:.2f} | {e["cells"]:.1f} |')
  L += ['\n## C. 100-step rollouts from the same 20 unique saved states',
        '| ckpt | mode | net disp | path len | progress | fell |',
        '|---|---|---|---|---|---|']
  for n in order:
    for m in ('deterministic', 'stochastic'):
      e = pc[n]['saved_state_rollouts_100steps'][m]
      L.append(f'| {n} | {m} | {e["net_disp"]:.3f} | {e["path_len"]:.3f} | '
               f'{e["progress"]:.3f} | {e["fell"]:.2f} |')
  t = r['timing']
  L += ['\n## D. Collapse timing (metrics.json, eval every ~10.5k steps)',
        f'- first eval with action saturation > 0.5: step {t["first_eval_saturation_gt_0.5"]}',
        f'- first eval with logits gap > 20: step {t["first_eval_logits_gap_gt_20"]}',
        '- see timeline.png for saturation vs critic-gap growth vs actor loss.',
        '\n## Hypothesis checks',
        f'- H1 scale collapse (final med scale < 0.2 and < 0.5x init): '
        f'{h["h1_scale_collapse"]} (series {h["scale_median_series"]})',
        f'- H1 mode saturation (init <0.1 -> final >0.4): {h["h1_mode_saturation"]}',
        f'- H2 deterministic static (static_frac>0.3 or speed<0.005): '
        f'{h["h2_deterministic_static"]}',
        f'- H2 stochastic moves >=2x deterministic: {h["h2_stochastic_moves_2x"]}',
        f'- H2 TOTAL scale collapse (final sigma~0, sample==mode): '
        f'{h["h2_total_scale_collapse"]}',
        f'- H2 init policy stochastic gain >=2x (what collapse destroyed): '
        f'{h["h2_init_stochastic_gain_2x"]}',
        f'- H3 scale monotone decreasing: {h["h3_scale_monotone_decreasing"]}',
        f'\n**Verdict: `{h["verdict"]}`**']
  open(os.path.join(out, 'ant_entropy_audit.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
  main()
