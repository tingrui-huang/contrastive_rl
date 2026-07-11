"""Ant critic on-support LOCAL action ranking (no training).

Follows the OOD finding: does the 150k critic rank actions validly *within* the
behavior-supported neighborhood? Candidate sets per saved state: actor-local
Gaussian (sigma 0.01..0.20), replay-neighbor factual actions, and broad uniform
(OOD control). Support scored by kNN over a reconstructed behavior buffer (NOT
by the critic). Progress is RECEDING-HORIZON: the candidate is the first action,
then the deterministic actor resumes (1 / 3 / 5 steps).
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ant_action_validity import load_actor_critic, restore, saturation, _spearman

from crl.config import Config
from crl import envs as envs_mod

CKPT = 'D:/Users/trhua/Research/contrastive_rl/antmaze_umaze_s0/latest.pkl'
NPZ = 'D:/Users/trhua/Research/contrastive_rl/artifacts/ant_action_validity/ant_action_validity_samples.npz'
OUT = 'D:/Users/trhua/Research/contrastive_rl/artifacts/ant_critic_local_ranking'
SIGMAS = [0.01, 0.03, 0.05, 0.10, 0.20]
N_LOCAL = 128
N_UNIFORM = 128
K_NBR = 10
N_STATES = 35
FALL_Z = 0.3


def build_behavior_buffer(env, actor, n_eps=25):
  u = env._env.unwrapped
  S, A = [], []
  for _ in range(n_eps):
    obs = env.reset()
    for _ in range(env.max_episode_steps):
      a = actor(obs)
      S.append(obs[:env.obs_dim].copy()); A.append(a.copy())
      obs, _, _, _ = env.step(a)
  return np.array(S, np.float32), np.array(A, np.float32)


def receding(env, u, ref, first_action, actor, horizons=(1, 3, 5)):
  """Candidate action first, then deterministic actor resumes. u.step +
  env._flatten so we avoid the TimeLimit wrapper; goal fixed to ref goal."""
  restore(u, ref['qpos'], ref['qvel'])
  u.goal = np.asarray(ref['goal'], float)
  mujoco.mj_forward(u.model, u.data)
  goal, d0 = ref['goal'], ref['d0']
  xy0 = ref['xy']
  od = u.step(np.asarray(first_action, np.float32))[0]
  minz = float(u.data.qpos[2])
  out = {}
  for step in range(1, max(horizons) + 1):
    if step in horizons:
      xy = np.asarray(u.data.qpos[:2]).copy()
      gdir = (goal - xy0) / (np.linalg.norm(goal - xy0) + 1e-9)
      out[step] = dict(progress=float(d0 - np.linalg.norm(xy - goal)),
                       proj=float(np.dot(xy - xy0, gdir)), fell=bool(minz < FALL_Z))
    if step < max(horizons):
      a = actor(env._flatten(od))
      od = u.step(np.asarray(a, np.float32))[0]
      minz = min(minz, float(u.data.qpos[2]))
  return out


def _boot(x, n=1500):
  x = np.asarray([v for v in x if np.isfinite(v)])
  if not len(x):
    return [None, None]
  r = np.random.default_rng(0)
  m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
  return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', default=CKPT); ap.add_argument('--npz', default=NPZ)
  ap.add_argument('--out', default=OUT); ap.add_argument('--n_states', type=int, default=N_STATES)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(0)

  cfg = Config(env_name='antmaze_umaze')
  env = envs_mod.make_env('antmaze_umaze', cfg, seed=7)
  u = env._env.unwrapped
  actor, critic, step = load_actor_critic('antmaze_umaze', args.ckpt, cfg)
  d = np.load(args.npz)
  refs = [dict(qpos=d['qpos'][k], qvel=d['qvel'][k], goal=d['goal'][k],
              obs=d['obs'][k], d0=float(d['d0'][k]), xy=d['qpos'][k][:2].copy())
          for k in range(min(args.n_states, len(d['qpos'])))]
  print(f'ckpt step {step} | {len(refs)} states | building behavior buffer...')
  Sbuf, Abuf = build_behavior_buffer(env, actor)
  Smu, Ssd = Sbuf.mean(0), Sbuf.std(0) + 1e-6
  Amu, Asd = Abuf.mean(0), Abuf.std(0) + 1e-6
  Sn = (Sbuf - Smu) / Ssd
  print(f'  behavior buffer: {len(Sbuf)} transitions')

  PH = 5  # primary receding horizon
  # accumulators per candidate-set key
  sets = [f'local_s{s}' for s in SIGMAS] + ['replay_nbr', 'uniform']
  acc = {k: {'spear': [], 'cb_gt_rand': [], 'cb_gt_worst': [], 'actor_gt_rand': [],
             'cb_prog': [], 'top_dec': [], 'bot_dec': [], 'cb_dist_actor': [],
             'cb_support': [], 'cb_prog1': [], 'cb_prog3': []} for k in sets}
  replay_vs_local = []           # replay-nbr progress vs local(0.05) progress
  global_best_src = []           # source of pooled critic-best
  global_best_dist_actor, global_best_support = [], []
  nbr_state_dists = []

  for ci, ref in enumerate(refs):
    api = actor(ref['obs'])
    # kNN replay states -> neighbor actions
    sn = (ref['obs'][:cfg.obs_dim] - Smu) / Ssd
    dists = np.linalg.norm(Sn - sn[None], axis=1)
    nn = np.argsort(dists)[:K_NBR]
    nbr_actions = Abuf[nn]; nbr_state_dists.append(float(dists[nn].mean()))

    def support(a):                # kNN action support: min dist to neighbor acts
      return float(np.linalg.norm(nbr_actions - a[None], axis=1).min())

    # build candidate sets
    cand_sets = {}
    for s in SIGMAS:
      c = np.clip(api[None] + rng.normal(0, s, (N_LOCAL, 8)), -1, 1).astype(np.float32)
      cand_sets[f'local_s{s}'] = c
    cand_sets['replay_nbr'] = nbr_actions.astype(np.float32)
    cand_sets['uniform'] = rng.uniform(-1, 1, (N_UNIFORM, 8)).astype(np.float32)

    pooled_c, pooled_sc, pooled_src = [], [], []
    actor_res = receding(env, u, ref, api, actor)
    actor_prog = actor_res[PH]['progress']

    for key, C in cand_sets.items():
      sc = critic(ref['obs'], C)
      prog5 = np.zeros(len(C)); prog1 = np.zeros(len(C)); prog3 = np.zeros(len(C))
      for k in range(len(C)):
        r = receding(env, u, ref, C[k], actor)
        prog1[k] = r[1]['progress']; prog3[k] = r[3]['progress']; prog5[k] = r[5]['progress']
      ib, iw, ir = int(np.argmax(sc)), int(np.argmin(sc)), int(rng.integers(len(C)))
      acc[key]['spear'].append(_spearman(sc, prog5))
      acc[key]['cb_gt_rand'].append(prog5[ib] > prog5[ir])
      acc[key]['cb_gt_worst'].append(prog5[ib] > prog5[iw])
      acc[key]['actor_gt_rand'].append(actor_prog > prog5[ir])
      acc[key]['cb_prog'].append(prog5[ib]); acc[key]['cb_prog1'].append(prog1[ib])
      acc[key]['cb_prog3'].append(prog3[ib])
      acc[key]['cb_dist_actor'].append(float(np.linalg.norm(C[ib] - api)))
      acc[key]['cb_support'].append(support(C[ib]))
      # score deciles
      order = np.argsort(sc); nd = max(1, len(C) // 10)
      acc[key]['bot_dec'].append(float(prog5[order[:nd]].mean()))
      acc[key]['top_dec'].append(float(prog5[order[-nd:]].mean()))
      if key == 'replay_nbr':
        replay_best = prog5[ib]
      pooled_c.append(C); pooled_sc.append(sc); pooled_src += [key] * len(C)
    # replay-nbr vs local(0.05)
    loc5 = np.array(acc['local_s0.05']['cb_prog'][-1])
    replay_vs_local.append(replay_best - loc5)
    # global pooled best
    pooled_sc = np.concatenate(pooled_sc); pooled_c = np.concatenate(pooled_c)
    gb = int(np.argmax(pooled_sc))
    global_best_src.append(pooled_src[gb])
    global_best_dist_actor.append(float(np.linalg.norm(pooled_c[gb] - api)))
    global_best_support.append(support(pooled_c[gb]))
    if ci % 10 == 0:
      print(f'  {ci}/{len(refs)}')

  def summ(key):
    a = acc[key]
    sp = np.array([v for v in a['spear'] if np.isfinite(v)])
    return {
        'spearman_median': float(np.median(sp)) if len(sp) else None,
        'spearman_frac_pos': float(np.mean(sp > 0)) if len(sp) else None,
        'cb_beats_random': float(np.mean(a['cb_gt_rand'])),
        'cb_beats_random_ci': _boot(np.array(a['cb_gt_rand'], float)),
        'cb_beats_worst': float(np.mean(a['cb_gt_worst'])),
        'actor_beats_random': float(np.mean(a['actor_gt_rand'])),
        'cb_progress_mean@5': float(np.mean(a['cb_prog'])),
        'cb_progress_ci@5': _boot(a['cb_prog']),
        'cb_progress_mean@1': float(np.mean(a['cb_prog1'])),
        'cb_progress_mean@3': float(np.mean(a['cb_prog3'])),
        'top_decile_progress': float(np.mean(a['top_dec'])),
        'bot_decile_progress': float(np.mean(a['bot_dec'])),
        'cb_dist_from_actor': float(np.mean(a['cb_dist_actor'])),
        'cb_support_score': float(np.mean(a['cb_support'])),
    }
  report = {'ckpt': args.ckpt, 'step': int(step), 'n_states': len(refs),
            'primary_horizon': PH, 'behavior_buffer_size': int(len(Sbuf)),
            'mean_neighbor_state_dist': float(np.mean(nbr_state_dists)),
            'per_set': {k: summ(k) for k in sets},
            'replay_vs_local005_mean': float(np.mean(replay_vs_local)),
            'global_best_source_pct': {k: float(global_best_src.count(k) / len(global_best_src))
                                       for k in sets},
            'global_best_dist_from_actor': float(np.mean(global_best_dist_actor)),
            'global_best_support': float(np.mean(global_best_support))}

  # ---- verdict ----
  small = ['local_s0.01', 'local_s0.03', 'local_s0.05']
  def reliable_pos(keys):
    return all((report['per_set'][k]['spearman_median'] or -1) > 0.1
               and report['per_set'][k]['cb_beats_random'] > 0.55 for k in keys)
  def near_zero_or_neg(keys):
    return all((report['per_set'][k]['spearman_median'] or 0) < 0.05
               and report['per_set'][k]['cb_beats_random'] < 0.55 for k in keys)
  local_pos = reliable_pos(small) and report['per_set']['replay_nbr']['cb_beats_random'] > 0.55
  uniform_fail = report['per_set']['uniform']['cb_beats_random'] < 0.5
  local_median_sp = np.median([report['per_set'][k]['spearman_median'] or 0 for k in small])
  if local_pos and uniform_fail:
    verdict = 'ONLY_BROAD_OOD_EXTRAPOLATION_FAILURE'
  elif near_zero_or_neg(small) and uniform_fail:
    verdict = 'BOTH_LOCAL_AND_OOD_FAILURE'
  elif local_median_sp < -0.05:
    verdict = 'LOCAL_RANKING_INVERTED'
  elif reliable_pos(small):
    verdict = 'VALID_LOCAL_ACTION_RANKING'
  else:
    verdict = 'LOCAL_RANKING_WEAK'
  report['verdict'] = verdict
  json.dump(report, open(os.path.join(args.out, 'ant_local_ranking.json'), 'w'), indent=2)
  _md(args.out, report)
  print('\nVERDICT:', verdict)


def _md(out, r):
  L = [f'# Ant critic on-support local action ranking\n',
       f'**Verdict: `{r["verdict"]}`** (step {r["step"]}, {r["n_states"]} states, '
       f'receding horizon = {r["primary_horizon"]}, actor resumes)\n',
       f'behavior buffer {r["behavior_buffer_size"]} transitions; mean neighbor '
       f'state dist {r["mean_neighbor_state_dist"]:.2f}\n',
       '## Per candidate set (progress = receding-horizon @5, +=closer to goal)',
       '| set | spearman(med) | sp>0 | cb>rand | cb>worst | actor>rand | cb prog@5 | '
       'top-dec | bot-dec | cb dist_actor | cb support |',
       '|---|---|---|---|---|---|---|---|---|---|---|']
  for k, s in r['per_set'].items():
    L.append(f'| {k} | {s["spearman_median"]} | {s["spearman_frac_pos"]} | '
             f'{s["cb_beats_random"]:.2f} | {s["cb_beats_worst"]:.2f} | '
             f'{s["actor_beats_random"]:.2f} | {s["cb_progress_mean@5"]:.4f} | '
             f'{s["top_decile_progress"]:.4f} | {s["bot_decile_progress"]:.4f} | '
             f'{s["cb_dist_from_actor"]:.3f} | {s["cb_support_score"]:.3f} |')
  L.append(f'\nreplay-nbr vs local(0.05) cb progress diff: {r["replay_vs_local005_mean"]:.4f}')
  L.append(f'\npooled critic-best source %: {r["global_best_source_pct"]}')
  L.append(f'pooled critic-best mean dist from actor: {r["global_best_dist_from_actor"]:.3f}, '
           f'support {r["global_best_support"]:.3f}')
  open(os.path.join(out, 'ant_local_ranking.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
  main()
