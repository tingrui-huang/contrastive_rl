"""AntMaze action-validity probe (no training changes).

Uses the existing 150k checkpoint + current eval env to decide whether the
learned critic gives a VALID local action ranking for real goal progress, and
whether the actor exploits it. Exact MuJoCo clone/restore so every candidate
action is evaluated from the identical simulator state.

Gates (abort/flag honestly): (1) exact restore, (2) action effect exists,
(3) critic action-dependence, (4) progress-sign sanity. Then a within-state
action comparison over horizons 1/5/10 with stratification + bootstrap CIs.

Run:  python scripts/ant_action_validity.py --ckpt <run>/latest.pkl --out artifacts/ant_action_validity
"""
import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from crl.config import Config

HORIZONS = (1, 5, 10)
N_CAND = 130           # >=128 candidate actions per state
FALL_Z = 0.3
NEAR_GOAL = 1.0        # exclude states already within this distance (not terminal)
STAND_Z = 0.35         # exclude clearly-fallen states when sampling
SAT_THRESH = 0.99


# --------------------------------------------------------------------------- #
def load_actor_critic(env_name, ckpt, cfg):
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp, hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)
  step, state = ckpt_mod.load_checkpoint(ckpt)

  @jax.jit
  def _actor(obs):
    return nets.sample_eval(nets.policy_network.apply(state.policy_params, obs), None)

  @jax.jit
  def _critic(obs_k, acts):
    return jnp.diag(nets.q_network.apply(state.q_params, obs_k, acts))

  def actor(obs):
    return np.asarray(_actor(jnp.asarray(obs[None]))[0])

  def critic(obs, acts):                      # obs [D], acts [K,A] -> [K]
    obs_k = jnp.asarray(np.tile(obs, (len(acts), 1)))
    return np.asarray(_critic(obs_k, jnp.asarray(acts)))
  return actor, critic, step


def restore(u, qpos, qvel):
  u.data.qpos[:] = qpos
  u.data.qvel[:] = qvel
  mujoco.mj_forward(u.model, u.data)


def heading_aligned(quat, xy, goal):
  w, x, y, z = quat
  hx, hy = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)   # rotate [1,0,0]
  ha = np.arctan2(hy, hx)
  ga = np.arctan2(goal[1] - xy[1], goal[0] - xy[0])
  diff = (ha - ga + np.pi) % (2 * np.pi) - np.pi
  return abs(diff) < np.pi / 2


def saturation(a):
  return float(np.mean(np.abs(a) > SAT_THRESH))


# --------------------------------------------------------------------------- #
def sample_states(env, actor, n_target, n_eps, rng):
  u = env._env.unwrapped
  od = env.obs_dim
  states = []
  for _ in range(n_eps):
    obs = env.reset()
    for t in range(env.max_episode_steps):
      qpos = np.asarray(u.data.qpos).copy()
      goal = obs[od:od + 2].copy()
      xy = qpos[:2].copy(); zt = float(qpos[2]); d = float(np.linalg.norm(xy - goal))
      if zt > STAND_Z and d > NEAR_GOAL and t < env.max_episode_steps - 15:
        states.append(dict(
            qpos=qpos, qvel=np.asarray(u.data.qvel).copy(), goal=goal, obs=obs.copy(),
            xy=xy, z=zt, d0=d, quat=qpos[3:7].copy(),
            actor_sat=saturation(actor(obs)),
            aligned=bool(heading_aligned(qpos[3:7], xy, goal))))
      obs, _, _, _ = env.step(actor(obs))
    if len(states) >= n_target * 4:
      break
  # subsample to n_target with spread over distance (variety)
  rng.shuffle(states)
  return states[:max(n_target, 100)]


def rollout(u, ref, action, horizons=HORIZONS):
  restore(u, ref['qpos'], ref['qvel'])
  xy0, goal = ref['xy'], ref['goal']
  d0 = ref['d0']
  gdir = goal - xy0
  gdir = gdir / (np.linalg.norm(gdir) + 1e-9)
  minz = ref['z']
  out = {}
  for step in range(1, max(horizons) + 1):
    u.step(np.asarray(action, np.float32))
    minz = min(minz, float(u.data.qpos[2]))
    if step in horizons:
      xy = np.asarray(u.data.qpos[:2]).copy()
      d = float(np.linalg.norm(xy - goal))
      out[step] = dict(progress=d0 - d, d_after=d, disp=float(np.linalg.norm(xy - xy0)),
                       proj=float(np.dot(xy - xy0, gdir)), z=float(u.data.qpos[2]),
                       minz=minz, fell=bool(minz < FALL_Z))
  return out


def rollout_path(u, ref, action, n=10):
  restore(u, ref['qpos'], ref['qvel'])
  path = [ref['xy'].copy()]
  for _ in range(n):
    u.step(np.asarray(action, np.float32))
    path.append(np.asarray(u.data.qpos[:2]).copy())
  return np.array(path)


def build_candidates(actor_a, rng):
  zero = np.zeros(8, np.float32)
  uni = rng.uniform(-1, 1, (64, 8)).astype(np.float32)
  gau = np.clip(actor_a[None] + rng.normal(0, 0.3, (64, 8)), -1, 1).astype(np.float32)
  return np.vstack([actor_a[None], zero[None], uni, gau])   # idx0=actor idx1=zero


# --------------------------------------------------------------------------- #
def gates(env, actor, critic, states, rng):
  u = env._env.unwrapped
  g = {}
  # Gate 1: exact restore determinism
  ref = states[0]; a = rng.uniform(-1, 1, 8).astype(np.float32)
  r1 = rollout(u, ref, a); r2 = rollout(u, ref, a)
  d_xy = abs(r1[10]['d_after'] - r2[10]['d_after']); d_z = abs(r1[10]['z'] - r2[10]['z'])
  g['gate1_restore'] = {'d_xy': d_xy, 'd_z': d_z, 'pass': bool(d_xy < 1e-6 and d_z < 1e-6)}
  # Gate 2: action effect exists (zero vs strong action differ)
  diffs = []
  for ref in states[:8]:
    p0 = rollout(u, ref, np.zeros(8, np.float32))[5]['disp']
    p1 = rollout(u, ref, np.ones(8, np.float32) * 0.8)[5]['disp']
    diffs.append(abs(p1 - p0))
  g['gate2_action_effect'] = {'mean_disp_diff': float(np.mean(diffs)),
                              'pass': bool(np.mean(diffs) > 1e-3)}
  # Gate 3: critic action dependence (score spread across candidates)
  stds, ranges, uniq = [], [], []
  for ref in states:
    cand = build_candidates(actor(ref['obs']), rng)
    sc = critic(ref['obs'], cand)
    stds.append(float(sc.std())); ranges.append(float(sc.max() - sc.min()))
    uniq.append(int(len(np.unique(np.round(sc, 4)))))
    ref['_cand'] = cand; ref['_scores'] = sc          # cache for main loop
  med_std = float(np.median(stds))
  g['gate3_critic_dependence'] = {
      'median_score_std': med_std, 'median_score_range': float(np.median(ranges)),
      'median_unique_scores': float(np.median(uniq)),
      'action_insensitive': bool(med_std < 1e-3)}
  # Gate 4: progress-sign sanity on 5 examples
  ex = []
  for ref in states[:5]:
    r = rollout(u, ref, actor(ref['obs']))[5]
    ex.append({'d_before': ref['d0'], 'd_after': r['d_after'],
               'progress': r['progress'],
               'sign_ok': bool((r['progress'] > 0) == (r['d_after'] < ref['d0']))})
  g['gate4_progress_sign'] = {'examples': ex, 'pass': all(e['sign_ok'] for e in ex)}
  return g


# --------------------------------------------------------------------------- #
def _spearman(a, b):
  if np.std(a) == 0 or np.std(b) == 0:
    return np.nan
  ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
  return float(np.corrcoef(ra, rb)[0, 1])


def _boot_ci(x, n=2000, seed=0):
  x = np.asarray([v for v in x if np.isfinite(v)])
  if len(x) == 0:
    return (None, None)
  r = np.random.default_rng(seed)
  means = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
  return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--env_name', default='antmaze_umaze')
  ap.add_argument('--ckpt', default='D:/Users/trhua/Research/contrastive_rl/antmaze_umaze_s0/latest.pkl')
  ap.add_argument('--n_states', type=int, default=110)
  ap.add_argument('--n_eps', type=int, default=45)
  ap.add_argument('--out', default='D:/Users/trhua/Research/contrastive_rl/artifacts/ant_action_validity')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(args.seed)

  cfg = Config(env_name=args.env_name)
  env = envs_mod.make_env(args.env_name, cfg, seed=args.seed + 5)
  u = env._env.unwrapped
  actor, critic, step = load_actor_critic(args.env_name, args.ckpt, cfg)
  print(f'ckpt step {step} | sampling reference states...')
  states = sample_states(env, actor, args.n_states, args.n_eps, rng)
  print(f'  {len(states)} valid reference states')

  print('running gates...')
  G = gates(env, actor, critic, states, rng)
  for k, v in G.items():
    print(f'  {k}: pass={v.get("pass", v.get("action_insensitive"))}')
  abort = None
  if not G['gate1_restore']['pass']:
    abort = 'GATE1_RESTORE_NONDETERMINISTIC'
  elif not G['gate2_action_effect']['pass']:
    abort = 'GATE2_NO_ACTION_EFFECT'
  elif not G['gate4_progress_sign']['pass']:
    abort = 'GATE4_PROGRESS_SIGN_WRONG'
  if abort:
    json.dump({'aborted': abort, 'gates': G}, open(os.path.join(args.out, 'ant_action_validity.json'), 'w'), indent=2)
    print('ABORTED:', abort)
    return

  # ---- full within-state probe ----
  choices = ['actor', 'critic_best', 'critic_worst', 'random', 'zero']
  # per-horizon, per-choice, per-state progress + fell + critic-score
  prog = {h: {c: [] for c in choices} for h in HORIZONS}
  fell = {h: {c: [] for c in choices} for h in HORIZONS}
  csco = {c: [] for c in choices}                         # critic score of chosen action
  spear = {h: [] for h in HORIZONS}                       # within-state spearman
  cand_std, cand_prog_range = [], {h: [] for h in HORIZONS}
  pooled = {h: {'score': [], 'progress': []} for h in HORIZONS}
  score_ranges = []                                       # within-state critic score range
  strat_keys = ['near', 'far', 'standing', 'low_torso', 'low_sat', 'high_sat',
                'aligned', 'misaligned']
  strat = {s: {h: {'critic_best': [], 'random': [], 'actor': []} for h in HORIZONS} for s in strat_keys}
  d_med = np.median([s['d0'] for s in states])
  sat_med = np.median([s['actor_sat'] for s in states])

  print('probing candidates per state...')
  for si, ref in enumerate(states):
    cand = ref['_cand']; sc = ref['_scores']
    cand_std.append(float(sc.std())); score_ranges.append(float(sc.max() - sc.min()))
    idx = {'actor': 0, 'zero': 1, 'critic_best': int(np.argmax(sc)),
           'critic_worst': int(np.argmin(sc)), 'random': int(rng.integers(len(cand)))}
    # roll out ALL candidates once (records d@1/5/10)
    cand_prog = {h: np.zeros(len(cand)) for h in HORIZONS}
    cand_fell = {h: np.zeros(len(cand), bool) for h in HORIZONS}
    for ci in range(len(cand)):
      r = rollout(u, ref, cand[ci])
      for h in HORIZONS:
        cand_prog[h][ci] = r[h]['progress']; cand_fell[h][ci] = r[h]['fell']
    for h in HORIZONS:
      spear[h].append(_spearman(sc, cand_prog[h]))
      cand_prog_range[h].append(float(cand_prog[h].max() - cand_prog[h].min()))
      pooled[h]['score'].extend((sc - sc.mean()).tolist())   # within-state centered
      pooled[h]['progress'].extend(cand_prog[h].tolist())
      for c in choices:
        prog[h][c].append(float(cand_prog[h][idx[c]]))
        fell[h][c].append(bool(cand_fell[h][idx[c]]))
      # stratify (use horizon-specific progress for the 3 tracked choices)
      def add(key):
        for c in ('critic_best', 'random', 'actor'):
          strat[key][h][c].append(float(cand_prog[h][idx[c]]))
      add('near' if ref['d0'] <= d_med else 'far')
      add('standing' if ref['z'] >= 0.5 else 'low_torso')
      add('low_sat' if ref['actor_sat'] <= sat_med else 'high_sat')
      add('aligned' if ref['aligned'] else 'misaligned')
    for c in choices:
      csco[c].append(float(sc[idx[c]]))
    if si % 25 == 0:
      print(f'  {si}/{len(states)}')

  # ---- aggregate ----
  def agg_choice(h, c):
    p = np.array(prog[h][c])
    lo, hi = _boot_ci(p)
    return {'mean': float(p.mean()), 'median': float(np.median(p)),
            'ci95': [lo, hi], 'fall_fraction': float(np.mean(fell[h][c]))}
  report = {'env': args.env_name, 'ckpt': args.ckpt, 'step': int(step),
            'n_states': len(states), 'n_candidates': N_CAND, 'gates': G,
            'per_horizon': {}}
  for h in HORIZONS:
    cb = np.array(prog[h]['critic_best']); rd = np.array(prog[h]['random'])
    cw = np.array(prog[h]['critic_worst']); ac = np.array(prog[h]['actor'])
    cb_beats_rand = float(np.mean(cb > rd)); ac_beats_rand = float(np.mean(ac > rd))
    cb_beats_worst = float(np.mean(cb > cw))
    sp = np.array([v for v in spear[h] if np.isfinite(v)])
    report['per_horizon'][h] = {
        'choices': {c: agg_choice(h, c) for c in choices},
        'frac_critic_best_beats_random': cb_beats_rand,
        'frac_actor_beats_random': ac_beats_rand,
        'frac_critic_best_beats_critic_worst': cb_beats_worst,
        'ci_cb_beats_random': _boot_ci((cb > rd).astype(float)),
        'ci_cb_beats_worst': _boot_ci((cb > cw).astype(float)),
        'spearman_within_state_mean': float(sp.mean()) if len(sp) else None,
        'spearman_within_state_median': float(np.median(sp)) if len(sp) else None,
        'spearman_frac_positive': float(np.mean(sp > 0)) if len(sp) else None,
        'avg_within_state_critic_std': float(np.mean(cand_std)),
        'avg_within_state_progress_range': float(np.mean(cand_prog_range[h])),
    }
  report['actor_vs_best_critic_score'] = {
      'actor_mean': float(np.mean(csco['actor'])),
      'critic_best_mean': float(np.mean(csco['critic_best'])),
      'actor_below_best': float(np.mean(np.array(csco['actor']) < np.array(csco['critic_best'])))}
  # stratified (progress means)
  report['stratified'] = {
      s: {h: {c: (float(np.mean(strat[s][h][c])) if strat[s][h][c] else None)
              for c in ('critic_best', 'random', 'actor')}
          for h in HORIZONS} for s in strat_keys}
  report['stratum_counts'] = {s: len(strat[s][HORIZONS[0]]['critic_best']) for s in strat_keys}

  # ---- verdict (primary horizon = 5) ----
  H = 5
  ph = report['per_horizon'][H]
  insensitive = G['gate3_critic_dependence']['action_insensitive']
  cb_mean = ph['choices']['critic_best']['mean']
  prog_range = ph['avg_within_state_progress_range']
  cb_beats_rand = ph['frac_critic_best_beats_random']
  cb_beats_worst = ph['frac_critic_best_beats_critic_worst']
  ac_beats_rand = ph['frac_actor_beats_random']
  sp_pos = ph['spearman_frac_positive'] or 0.0
  # is candidate motion even controllable?
  no_control = prog_range < 0.02 and abs(cb_mean) < 0.01
  critic_valid = (cb_beats_rand > 0.60 and cb_beats_worst > 0.65 and cb_mean > 0
                  and sp_pos > 0.5)
  actor_fails = (ac_beats_rand < 0.55 and
                 report['actor_vs_best_critic_score']['actor_below_best'] > 0.6)
  if no_control:
    verdict = 'LOCOMOTION_NOT_LOCALLY_CONTROLLABLE'
  elif insensitive:
    verdict = 'CRITIC_ACTION_INSENSITIVE'
  elif critic_valid and actor_fails:
    verdict = 'ACTOR_EXTRACTION_FAILURE'
  elif critic_valid:
    verdict = 'CRITIC_ACTION_SIGNAL_VALID'
  else:
    verdict = 'CRITIC_ACTION_RANKING_INVALID'
  report['verdict'] = verdict
  report['verdict_horizon'] = H

  json.dump(report, open(os.path.join(args.out, 'ant_action_validity.json'), 'w'), indent=2)
  np.savez(os.path.join(args.out, 'ant_action_validity_samples.npz'),
           qpos=np.array([s['qpos'] for s in states]),
           qvel=np.array([s['qvel'] for s in states]),
           goal=np.array([s['goal'] for s in states]),
           obs=np.array([s['obs'] for s in states]),
           d0=np.array([s['d0'] for s in states]))
  _plots(args.out, report, pooled, score_ranges, prog, env, states, actor, critic, rng)
  _write_md(args.out, report)
  print('\nVERDICT:', verdict)
  print('saved artifacts to', args.out)


def _plots(out, report, pooled, score_ranges, prog, env, states, actor, critic, rng):
  H = 5
  # scatter: within-state-centered critic score vs progress
  plt.figure(figsize=(5, 4))
  plt.scatter(pooled[H]['score'], pooled[H]['progress'], s=4, alpha=0.2)
  plt.axhline(0, color='k', lw=.5); plt.axvline(0, color='k', lw=.5)
  plt.xlabel('critic score (within-state centered)'); plt.ylabel(f'progress @ {H} steps')
  plt.title('critic score vs true goal progress'); plt.tight_layout()
  plt.savefig(os.path.join(out, 'scatter_score_vs_progress.png'), dpi=100); plt.close()
  # bar: mean progress by choice x horizon
  choices = ['actor', 'critic_best', 'critic_worst', 'random', 'zero']
  plt.figure(figsize=(8, 4)); w = 0.25
  for hi, h in enumerate(HORIZONS):
    vals = [report['per_horizon'][h]['choices'][c]['mean'] for c in choices]
    plt.bar(np.arange(len(choices)) + hi * w, vals, w, label=f'{h} steps')
  plt.xticks(np.arange(len(choices)) + w, choices); plt.axhline(0, color='k', lw=.5)
  plt.ylabel('mean goal progress'); plt.legend(); plt.title('mean progress by action choice')
  plt.tight_layout(); plt.savefig(os.path.join(out, 'bar_progress_by_choice.png'), dpi=100); plt.close()
  # histogram: within-state critic score ranges
  plt.figure(figsize=(5, 4)); plt.hist(score_ranges, bins=30)
  plt.xlabel('within-state critic score range (max-min)'); plt.ylabel('states')
  plt.title('critic score range per state'); plt.tight_layout()
  plt.savefig(os.path.join(out, 'hist_score_ranges.png'), dpi=100); plt.close()
  # representative 10 states: 5 action-choice XY paths
  u = env._env.unwrapped
  mz = u.maze
  fig, axes = plt.subplots(2, 5, figsize=(16, 7))
  for ax, ref in zip(axes.ravel(), states[:10]):
    for r in range(len(mz.maze_map)):
      for c in range(len(mz.maze_map[0])):
        if mz.maze_map[r][c] == 1:
          x, y = mz.cell_rowcol_to_xy(np.array([r, c])); s = mz.maze_size_scaling
          ax.add_patch(plt.Rectangle((x - s / 2, y - s / 2), s, s, color='0.85', zorder=0))
    cand = ref['_cand']; sc = ref['_scores']
    idx = {'actor': 0, 'zero': 1, 'critic_best': int(np.argmax(sc)),
           'critic_worst': int(np.argmin(sc)), 'random': int(rng.integers(len(cand)))}
    col = {'actor': 'tab:blue', 'critic_best': 'tab:green', 'critic_worst': 'tab:red',
           'random': 'tab:orange', 'zero': 'gray'}
    for c in choices:
      p = rollout_path(u, ref, cand[idx[c]])
      ax.plot(p[:, 0], p[:, 1], '-', color=col[c], lw=1.2, label=c)
    ax.scatter(*ref['xy'], c='k', s=25, zorder=3)
    ax.scatter(*ref['goal'], c='red', marker='*', s=90, zorder=3)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
  axes.ravel()[0].legend(fontsize=6, loc='upper left')
  fig.suptitle('representative states: action-choice 10-step paths (star=goal)')
  fig.tight_layout(); fig.savefig(os.path.join(out, 'representative_action_paths.png'), dpi=90); plt.close()


def _write_md(out, r):
  L = []
  L.append(f'# AntMaze action-validity probe\n')
  L.append(f'**Verdict: `{r["verdict"]}`** (primary horizon = {r["verdict_horizon"]} steps)\n')
  L.append(f'checkpoint step {r["step"]}, {r["n_states"]} reference states, '
           f'{r["n_candidates"]} candidate actions/state.\n')
  L.append('## Gates')
  g = r['gates']
  L.append(f'- Gate 1 exact restore: pass={g["gate1_restore"]["pass"]} '
           f'(dXY={g["gate1_restore"]["d_xy"]:.2e}, dz={g["gate1_restore"]["d_z"]:.2e})')
  L.append(f'- Gate 2 action effect: pass={g["gate2_action_effect"]["pass"]} '
           f'(mean disp diff {g["gate2_action_effect"]["mean_disp_diff"]:.4f})')
  L.append(f'- Gate 3 critic dependence: action_insensitive='
           f'{g["gate3_critic_dependence"]["action_insensitive"]} '
           f'(median score std {g["gate3_critic_dependence"]["median_score_std"]:.4f}, '
           f'range {g["gate3_critic_dependence"]["median_score_range"]:.4f})')
  L.append(f'- Gate 4 progress sign: pass={g["gate4_progress_sign"]["pass"]}\n')
  L.append('## Per-horizon (progress = d_before - d_after, +=closer)')
  L.append('| horizon | cb>rand | cb>worst | actor>rand | spearman(med) | sp>0 frac | '
           'cb mean | rand mean | actor mean | prog range |')
  L.append('|---|---|---|---|---|---|---|---|---|---|')
  for h in HORIZONS:
    p = r['per_horizon'][h]; ch = p['choices']
    L.append(f'| {h} | {p["frac_critic_best_beats_random"]:.2f} | '
             f'{p["frac_critic_best_beats_critic_worst"]:.2f} | '
             f'{p["frac_actor_beats_random"]:.2f} | '
             f'{p["spearman_within_state_median"]} | {p["spearman_frac_positive"]} | '
             f'{ch["critic_best"]["mean"]:.4f} | {ch["random"]["mean"]:.4f} | '
             f'{ch["actor"]["mean"]:.4f} | {p["avg_within_state_progress_range"]:.4f} |')
  a = r['actor_vs_best_critic_score']
  L.append(f'\nactor critic-score mean {a["actor_mean"]:.2f} vs critic-best '
           f'{a["critic_best_mean"]:.2f} (actor below best in {a["actor_below_best"]:.2f})\n')
  L.append('## Stratified mean progress @5 (critic_best / random / actor)')
  for s, hd in r['stratified'].items():
    d = hd[5]; n = r['stratum_counts'][s]
    L.append(f'- {s} (n={n}): {d["critic_best"]} / {d["random"]} / {d["actor"]}')
  open(os.path.join(out, 'ant_action_validity.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
  main()
