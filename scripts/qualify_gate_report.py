"""Per-gate analysis for the near-goal/open-area Ant qualification A/B.

Given one gate checkpoint (gate_<step>.pkl), reports everything the A/B is
judged on: policy std + fraction at the actor_min_std floor, sampled-vs-mode
distance, saturation, adaptive alpha, collection-action covariance/effective
rank (proxy for what is being ADDED to replay: fresh sampled-policy rollouts,
matching training collection), moving-transition fraction + XY displacement
distribution, torso height/fall fraction, unique state/goal/episode coverage,
local immediate controllability (1 step + 2 zero-action settle, non-pooled
sigma sets), critic top-vs-bottom decile 1-step progress, deterministic eval
(near-goal success + goal velocity).

Fresh reference states are sampled from the collection rollouts under HARD
coverage gates (unique states, multiple episodes/goals, moving+stationary
representation, no duplicated resting pose). Gate failures are reported, not
silently ignored.
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from ant_action_validity import restore, _spearman
from ant_entropy_audit import make_policy, eff_rank
from ant_immediate_controllability import probe_candidate

from crl.config import Config
from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod

SAT = 0.99
FALL_Z = 0.3
MOVE_DISP = 5e-3       # per-step XY displacement (m) counting as "moving"
MOVE_SPEED = 0.05      # |v_xy| for a "moving" reference state
N_COLL_EPS = 12
N_EVAL_EPS = 10
N_REF = 30
N_CAND = 64
SIGMAS = (0.05, 0.2)
DT = 0.05              # env step = frame_skip 5 x 0.01


def build_env(seed):
  cfg = Config(env_name='antmaze_open_near')
  env = envs_mod.make_env('antmaze_open_near', cfg, seed=seed)
  return cfg, env


def build_nets(cfg):
  return networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)


def make_critic(nets, q_params):
  @jax.jit
  def _critic(obs_k, acts):
    return jnp.diag(nets.q_network.apply(q_params, obs_k, acts))
  def critic(obs, acts):
    obs_k = jnp.asarray(np.tile(obs, (len(acts), 1)))
    return np.asarray(_critic(obs_k, jnp.asarray(acts)))
  return critic


# --------------------------------------------------------------------------- #
def collect(env, pol, key, n_eps):
  """Sampled-policy rollouts, mirroring training collection."""
  u = env._env.unwrapped
  steps, states = [], []
  for ep in range(n_eps):
    obs = env.reset()
    goal = obs[env.obs_dim:env.obs_dim + 2].copy()
    xy_prev = obs[:2].copy()
    for t in range(env.max_episode_steps):
      qpos = np.asarray(u.data.qpos).copy()
      qvel = np.asarray(u.data.qvel).copy()
      key, sk = jax.random.split(key)
      a = pol['sample'](obs, sk)
      if t % 5 == 0:
        states.append(dict(qpos=qpos, qvel=qvel, goal=goal.copy(),
                           obs=obs.copy(), ep=ep,
                           speed=float(np.linalg.norm(qvel[:2])),
                           z=float(qpos[2])))
      obs, _, _, _ = env.step(a)
      xy = obs[:2].copy()
      steps.append(dict(disp=float(np.linalg.norm(xy - xy_prev)),
                        z=float(u.data.qpos[2]), a=a, ep=ep,
                        cell=(round(xy[0] / 0.5), round(xy[1] / 0.5))))
      xy_prev = xy
  return steps, states, key


def eval_episodes(env, pol, n_eps):
  u = env._env.unwrapped
  rows = []
  for ep in range(n_eps):
    obs = env.reset()
    goal = obs[env.obs_dim:env.obs_dim + 2].copy()
    xy_prev = obs[:2].copy()
    d_prev = float(np.linalg.norm(xy_prev - goal))
    d0, dmin, succ = d_prev, d_prev, 0.0
    path, static, fall, gvel, sats = 0.0, 0, 0, [], []
    for t in range(env.max_episode_steps):
      a = pol['mode'](obs)
      sats.append(float(np.mean(np.abs(a) > SAT)))
      obs, rew, _, _ = env.step(a)
      xy = obs[:2].copy()
      d = float(np.linalg.norm(xy - goal))
      gvel.append((d_prev - d) / DT)
      dmin, succ = min(dmin, d), max(succ, rew)
      sd = float(np.linalg.norm(xy - xy_prev))
      path += sd; static += sd < 1e-4
      fall += float(u.data.qpos[2]) < FALL_Z
      xy_prev, d_prev = xy, d
    T = env.max_episode_steps
    rows.append(dict(success=succ, d0=d0, dmin=dmin, progress=d0 - dmin,
                     path_len=path, speed=path / T, static_frac=static / T,
                     fall_frac=fall / T, goal_vel=float(np.mean(gvel)),
                     sat=float(np.mean(sats))))
  return {q: float(np.mean([r[q] for r in rows])) for q in rows[0]}


def select_reference_states(states, rng):
  """Dedup + stratified moving/stationary selection with hard coverage gates."""
  seen, uniq = set(), []
  for s in states:
    k = tuple(np.round(np.concatenate([s['qpos'], s['qvel']]), 6))
    if k not in seen and s['z'] > 0.3:
      seen.add(k); uniq.append(s)
  moving = [s for s in uniq if s['speed'] > MOVE_SPEED]
  station = [s for s in uniq if s['speed'] <= MOVE_SPEED]
  gates = {
      'n_unique': len(uniq), 'n_unique_ok': len(uniq) >= 60,
      'n_episodes': len({s['ep'] for s in uniq}), 'episodes_ok':
          len({s['ep'] for s in uniq}) >= 8,
      'n_goals': len({tuple(np.round(s['goal'], 3)) for s in uniq}),
      'goals_ok': len({tuple(np.round(s['goal'], 3)) for s in uniq}) >= 6,
      'moving_frac_pool': len(moving) / max(len(uniq), 1),
      'moving_ok': len(moving) >= 10, 'stationary_n_pool': len(station)}
  rng.shuffle(moving); rng.shuffle(station)
  n_mov = min(20, len(moving))
  sel = moving[:n_mov] + station[:N_REF - n_mov]
  if len(sel) < N_REF:
    sel += [s for s in moving[n_mov:]][:N_REF - len(sel)]
  gates['selected'] = len(sel)
  gates['selected_moving'] = int(sum(s['speed'] > MOVE_SPEED for s in sel))
  gates['no_duplicate_domination'] = True     # by construction (deduped)
  gates['pass'] = bool(gates['n_unique_ok'] and gates['episodes_ok']
                       and gates['goals_ok'] and gates['moving_ok']
                       and len(sel) >= 20)
  for s in sel:
    s['d0'] = float(np.linalg.norm(s['qpos'][:2] - s['goal']))
  return sel, gates


def controllability(env, u, refs, pol, critic, rng):
  out = {}
  actor_disp = []
  for sig in SIGMAS:
    per = {'std_proj': [], 'rng_prog': [], 'sp': [], 'gap': [], 'tgap': [],
           'fall3': []}
    for ref in refs:
      api = pol['mode'](ref['obs'])
      if sig == SIGMAS[0]:
        actor_disp.append(probe_candidate(u, ref, api)['disp1'])
      C = np.clip(api[None] + rng.normal(0, sig, (N_CAND, 8)), -1, 1).astype(np.float32)
      sc = critic(ref['obs'], C)
      M = [probe_candidate(u, ref, C[k]) for k in range(N_CAND)]
      proj1 = np.array([m['proj1'] for m in M])
      prog1 = np.array([m['prog1'] for m in M])
      per['std_proj'].append(proj1.std())
      per['rng_prog'].append(prog1.max() - prog1.min())
      per['sp'].append(_spearman(sc, prog1))
      nd = max(1, N_CAND // 10)
      o, ot = np.argsort(sc), np.argsort(prog1)
      per['gap'].append(prog1[o[-nd:]].mean() - prog1[o[:nd]].mean())
      per['tgap'].append(prog1[ot[-nd:]].mean() - prog1[ot[:nd]].mean())
      per['fall3'].append(np.mean([m['fall3'] for m in M]))
    sp = np.array([v for v in per['sp'] if np.isfinite(v)])
    out[f'sigma_{sig}'] = {
        'std_proj1_median': float(np.median(per['std_proj'])),
        'rng_prog1_median': float(np.median(per['rng_prog'])),
        'spearman_prog1_median': float(np.median(sp)) if len(sp) else None,
        'critic_decile_gap_mean': float(np.mean(per['gap'])),
        'true_decile_gap_mean': float(np.mean(per['tgap'])),
        'decile_usefulness': float(np.mean(per['gap'])
                                   / (np.mean(per['tgap']) + 1e-12)),
        'fall3_frac': float(np.mean(per['fall3']))}
  out['actor_disp1_median'] = float(np.median(actor_disp))
  return out


# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--out', required=True)     # gate report dir
  ap.add_argument('--tag', required=True)     # e.g. alpha0_10000
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(0)

  cfg, env_coll = build_env(seed=31)
  _, env_eval = build_env(seed=77)
  nets = build_nets(cfg)
  step, state = ckpt_mod.load_checkpoint(args.ckpt)
  pol = make_policy(nets, state.policy_params)
  critic = make_critic(nets, state.q_params)
  alpha = (float(np.exp(np.asarray(state.alpha_params)))
           if state.alpha_params is not None else None)
  key = jax.random.PRNGKey(7)

  # collection distribution (what training is adding to replay right now)
  steps, states, key = collect(env_coll, pol, key, N_COLL_EPS)
  disp = np.array([s['disp'] for s in steps])
  acts = np.array([s['a'] for s in steps])
  zs = np.array([s['z'] for s in steps])
  coll = {
      'moving_transition_frac': float(np.mean(disp > MOVE_DISP)),
      'disp_p50': float(np.percentile(disp, 50)),
      'disp_p90': float(np.percentile(disp, 90)),
      'disp_max': float(disp.max()), 'speed_mean': float(disp.mean()),
      'torso_z_mean': float(zs.mean()), 'fall_step_frac': float(np.mean(zs < FALL_Z)),
      'action_eff_rank': eff_rank(acts),
      'action_per_dim_std_mean': float(acts.std(0).mean()),
      'action_sat': float(np.mean(np.abs(acts) > SAT)),
      'unique_cells': len({s['cell'] for s in steps}),
      'n_episodes': N_COLL_EPS,
      'n_unique_goals': len({tuple(np.round(s['goal'], 3)) for s in states}),
  }

  # policy head on collection obs
  obs_b = np.array([s['obs'] for s in states], np.float32)
  obs_b = obs_b[rng.permutation(len(obs_b))[:400]]
  loc, scale = pol['dist'](obs_b)
  mode = np.tanh(loc)
  key, sk = jax.random.split(key)
  samp = pol['sample_batch'](obs_b, sk)
  head = {
      'scale_median': float(np.median(scale)),
      'scale_p10': float(np.percentile(scale, 10)),
      'scale_p90': float(np.percentile(scale, 90)),
      'log_scale_median': float(np.median(np.log(scale))),
      'frac_at_min_std': float(np.mean(scale < 1e-5)),
      'sample_vs_mode_dist': float(np.linalg.norm(samp - mode, axis=1).mean()),
      'mode_sat': float(np.mean(np.abs(mode) > SAT)),
      'sample_sat': float(np.mean(np.abs(samp) > SAT)),
      'alpha': alpha,
  }

  ev = eval_episodes(env_eval, pol, N_EVAL_EPS)

  refs, cov_gates = select_reference_states(states, rng)
  ctrl = None
  if cov_gates['pass']:
    ctrl = controllability(env_coll, env_coll._env.unwrapped, refs, pol,
                           critic, rng)
    np.savez_compressed(
        os.path.join(args.out, f'refs_{args.tag}.npz'),
        qpos=np.array([s['qpos'] for s in refs]),
        qvel=np.array([s['qvel'] for s in refs]),
        goal=np.array([s['goal'] for s in refs]),
        obs=np.array([s['obs'] for s in refs]))

  report = {'tag': args.tag, 'ckpt': args.ckpt, 'step': int(step),
            'policy_head': head, 'collection': coll, 'eval_deterministic': ev,
            'coverage_gates': cov_gates, 'controllability': ctrl}
  path = os.path.join(args.out, f'gate_{args.tag}.json')
  json.dump(report, open(path, 'w'), indent=2)
  print(json.dumps({'tag': args.tag, 'step': int(step),
                    'scale_med': head['scale_median'],
                    'frac_at_floor': head['frac_at_min_std'],
                    'alpha': alpha, 'sat': head['mode_sat'],
                    'moving_frac': coll['moving_transition_frac'],
                    'act_effrank': coll['action_eff_rank'],
                    'success': ev['success'], 'goal_vel': ev['goal_vel'],
                    'coverage_pass': cov_gates['pass']}, indent=1))
  print('saved', path)


if __name__ == '__main__':
  main()
