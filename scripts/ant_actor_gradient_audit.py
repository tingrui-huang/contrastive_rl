"""Checkpoint-only actor-gradient validity audit (no training, no changes).

On the stable 30k target_entropy=-8 checkpoints (xy / gcompact / gfull arms),
tests whether the critic's action gradient at the deterministic actor action
carries real control signal, and whether actor optimization follows it.

Per reference state (coverage-gated refs from the 30k gate reports):
  * grad_a f(s,a,g) at a_pi: norm, per-dim concentration (participation
    ratio, top-dim share), saturation relation (gradient mass pushing
    already-saturated dims outward through the clip = blocked mass),
    finite-difference validation of the gradient;
  * perturbations a' = clip(a_pi + delta * dir) for dir in {+grad_hat,
    -grad_hat, 6 random unit dirs} x delta in {0.02, 0.05, 0.1}, each executed
    from the exact restored state: critic score change, immediate XY goal
    progress, receding-horizon progress @3/@5 (actor resumes after the first
    step), displacement, fall risk (z < 0.3), torso min-z;
  * one ACTUAL actor optimizer step (verbatim actor loss incl. random_goals
    0.5 mixing, checkpoint alpha, checkpoint adam optimizer state): does the
    critic score of the new deterministic action rise, and does that predict
    real XY improvement? Plus a labeled 50-step actor-only continuation on a
    COPY of the params (critic/alpha frozen; nothing saved) to expose the
    optimization direction at measurable scale.

Verdicts: CRITIC_GRADIENT_ALIGNS_WITH_CONTROL / CRITIC_GRADIENT_SCORE_ONLY /
CRITIC_ACTION_GRADIENT_VANISHES / CRITIC_GRADIENT_DRIVES_SATURATION /
ACTOR_OPTIMIZATION_FAILURE / INCONCLUSIVE.
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import optax

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from ant_action_validity import restore, _boot_ci, _spearman

from crl.config import Config
from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod

ARMS = {
    'xy': ('antmaze_open_near', 'qual_open_near/adaptive_te8_s0/gate_30000.pkl',
           'qual_open_near/gates/refs_adaptive_te8_30000.npz'),
    'gcompact': ('antmaze_open_near_gcompact',
                 'qual_open_near/gcompact_te8_s0/gate_30000.pkl',
                 'qual_open_near/gates/refs_gcompact_te8_30000.npz'),
    'gfull': ('antmaze_open_near_gfull',
              'qual_open_near/gfull_te8_s0/gate_30000.pkl',
              'qual_open_near/gates/refs_gfull_te8_30000.npz'),
}
OUT = 'D:/Users/trhua/Research/contrastive_rl/artifacts/ant_actor_gradient_audit'
DELTAS = (0.02, 0.05, 0.1)
N_RAND = 6
FALL_Z = 0.3
SAT = 0.99
VANISH_GRAD_NORM = 0.05      # score units per action unit
HILL_STEPS = 50


def build(env_name, ckpt):
  cfg = Config(env_name=env_name)
  env = envs_mod.make_env(env_name, cfg, seed=41)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      twin_q=cfg.twin_q, use_image_obs=cfg.use_image_obs)
  step, state = ckpt_mod.load_checkpoint(ckpt)
  return cfg, env, nets, int(step), state


def make_fns(nets, state):
  @jax.jit
  def _mode(params, obs):
    return jnp.tanh(nets.policy_network.apply(params, obs).loc)

  @jax.jit
  def _f(obs, act):                       # single (obs, a) critic score
    return nets.q_network.apply(state.q_params, obs[None], act[None])[0, 0]

  grad_f = jax.jit(jax.grad(_f, argnums=1))

  def mode(params, obs):
    return np.asarray(_mode(params, jnp.asarray(obs[None]))[0])

  def f(obs, act):
    return float(_f(jnp.asarray(obs), jnp.asarray(act)))

  def gf(obs, act):
    return np.asarray(grad_f(jnp.asarray(obs), jnp.asarray(act)))
  return mode, f, gf


def set_env_goal(env, ref):
  """Point the env's flat-obs goal at the ref's goal (arm-aware)."""
  u = env._env.unwrapped
  u.goal = np.asarray(ref['goal_xy'], float)
  if hasattr(env, '_goal_vec'):
    env._goal_vec = ref['obs'][29:].copy()


def rollout(env, ref, first_action, actor_params, mode, horizons=(1, 3, 5)):
  """Exact restore; candidate for 1 step; deterministic actor resumes."""
  u = env._env.unwrapped
  restore(u, ref['qpos'], ref['qvel'])
  set_env_goal(env, ref)
  mujoco.mj_forward(u.model, u.data)
  goal, xy0, d0 = ref['goal_xy'], ref['qpos'][:2], ref['d0']
  od = u.step(np.asarray(first_action, np.float32))[0]
  minz = float(u.data.qpos[2])
  out = {}
  for h in range(1, max(horizons) + 1):
    if h in horizons:
      xy = np.asarray(u.data.qpos[:2])
      out[h] = dict(prog=float(d0 - np.linalg.norm(xy - goal)),
                    disp=float(np.linalg.norm(xy - xy0)),
                    minz=minz, fell=bool(minz < FALL_Z))
    if h < max(horizons):
      a = mode(actor_params, env._flatten(od))
      od = u.step(np.asarray(a, np.float32))[0]
      minz = min(minz, float(u.data.qpos[2]))
  return out


def actor_step(nets, cfg, state, obs_batch, key, n_steps=1):
  """Verbatim actor update (random_goals=0.5 mixing, checkpoint alpha, adam
  with the checkpoint's optimizer state). Returns new policy params."""
  alpha = (float(np.exp(np.asarray(state.alpha_params)))
           if state.alpha_params is not None else 0.0)
  obs_dim = cfg.obs_dim

  def loss_fn(policy_params, key):
    obs = jnp.asarray(obs_batch)
    s, g = obs[:, :obs_dim], obs[:, obs_dim:]
    new_obs = jnp.concatenate(
        [jnp.concatenate([s, s], 0),
         jnp.concatenate([g, jnp.roll(g, 1, axis=0)], 0)], axis=1)
    dist = nets.policy_network.apply(policy_params, new_obs)
    action = nets.sample(dist, key)
    log_prob = nets.log_prob(dist, action)
    q = nets.q_network.apply(state.q_params, new_obs, action)
    return jnp.mean(alpha * log_prob - jnp.diag(q))

  opt = optax.adam(cfg.actor_learning_rate, eps=1e-7)
  params, opt_state = state.policy_params, state.policy_optimizer_state
  vg = jax.jit(jax.value_and_grad(loss_fn))
  losses = []
  for _ in range(n_steps):
    key, sk = jax.random.split(key)
    val, grads = vg(params, sk)
    upd, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, upd)
    losses.append(float(val))
  return params, losses


# --------------------------------------------------------------------------- #
def audit_arm(arm, env_name, ckpt, refs_npz, rng):
  cfg, env, nets, step, state = build(env_name, ckpt)
  mode, f, gf = make_fns(nets, state)
  d = np.load(refs_npz)
  refs = []
  for k in range(len(d['qpos'])):
    refs.append(dict(qpos=d['qpos'][k], qvel=d['qvel'][k],
                     goal_xy=d['goal'][k], obs=d['obs'][k],
                     d0=float(np.linalg.norm(d['qpos'][k][:2] - d['goal'][k]))))
  u = env._env.unwrapped

  # gate: restore determinism
  r1 = rollout(env, refs[0], np.zeros(8, np.float32), state.policy_params, mode)
  r2 = rollout(env, refs[0], np.zeros(8, np.float32), state.policy_params, mode)
  assert abs(r1[5]['prog'] - r2[5]['prog']) < 1e-12, 'restore nondeterministic'

  G = {'gnorm': [], 'prat': [], 'topshare': [], 'blocked': [], 'sat': [],
       'fd_relerr': [], 'f0': []}
  dirs_rows = []          # per (state, dir-type, delta) physical outcomes
  for si, ref in enumerate(refs):
    api = mode(state.policy_params, ref['obs'])
    f0 = f(ref['obs'], api)
    g = gf(ref['obs'], api)
    gn = float(np.linalg.norm(g))
    ag = np.abs(g)
    G['gnorm'].append(gn)
    G['f0'].append(f0)
    G['prat'].append(float((ag.sum() ** 2) / (np.square(ag).sum() + 1e-18)))
    G['topshare'].append(float(ag.max() / (ag.sum() + 1e-18)))
    satd = np.abs(api) > SAT
    blocked = float(ag[satd & (np.sign(g) == np.sign(api))].sum()
                    / (ag.sum() + 1e-18))
    G['blocked'].append(blocked)
    G['sat'].append(float(satd.mean()))
    ghat = g / (gn + 1e-18)
    if si < 8:  # finite-difference gradient validation
      eps = 1e-3
      fd = f(ref['obs'], np.clip(api + eps * ghat, -1, 1)) - f0
      pred = eps * gn
      G['fd_relerr'].append(abs(fd - pred) / (abs(pred) + 1e-12))

    dirs = [('+grad', ghat), ('-grad', -ghat)]
    for r in range(N_RAND):
      v = rng.normal(size=8)
      dirs.append(('random', v / np.linalg.norm(v)))
    for dname, v in dirs:
      for delta in DELTAS:
        a = np.clip(api + delta * v, -1, 1).astype(np.float32)
        eff = float(np.linalg.norm(a - api))
        df = f(ref['obs'], a) - f0
        ro = rollout(env, ref, a, state.policy_params, mode)
        base = rollout(env, ref, api, state.policy_params, mode)
        dirs_rows.append(dict(
            state=si, dir=dname, delta=delta, eff_norm=eff, df=df,
            prog1=ro[1]['prog'] - base[1]['prog'],
            prog3=ro[3]['prog'] - base[3]['prog'],
            prog5=ro[5]['prog'] - base[5]['prog'],
            disp1=ro[1]['disp'], fell5=ro[5]['fell'],
            minz5=ro[5]['minz']))

  # ---- aggregate perturbation outcomes ----
  def sel(dname, delta, q):
    return np.array([r[q] for r in dirs_rows
                     if r['dir'] == dname and r['delta'] == delta])
  pert = {}
  for delta in DELTAS:
    dfp, dfm = sel('+grad', delta, 'df'), sel('-grad', delta, 'df')
    row = {'df_plus_mean': float(dfp.mean()),
           'df_plus_pos_frac': float(np.mean(dfp > 0)),
           'df_minus_mean': float(dfm.mean()),
           'eff_norm_plus': float(sel('+grad', delta, 'eff_norm').mean())}
    for h in ('prog1', 'prog3', 'prog5'):
      pp, pm = sel('+grad', delta, h), sel('-grad', delta, h)
      pr = sel('random', delta, h)
      m_pm = pp - pm
      row[f'{h}_margin_vs_minus'] = (float(m_pm.mean()), _boot_ci(m_pm))
      # random baseline: mean over the 6 random dirs per state
      pr_state = pr.reshape(len(refs), N_RAND).mean(1)
      m_pr = pp - pr_state
      row[f'{h}_margin_vs_random'] = (float(m_pr.mean()), _boot_ci(m_pr))
    row['fell5_plus'] = float(sel('+grad', delta, 'fell5').mean())
    row['fell5_random'] = float(sel('random', delta, 'fell5').mean())
    row['minz5_plus'] = float(sel('+grad', delta, 'minz5').mean())
    pert[delta] = row
  # score-vs-progress coupling across all perturbations
  alldf = np.array([r['df'] for r in dirs_rows])
  sp_df_prog1 = _spearman(alldf, np.array([r['prog1'] for r in dirs_rows]))
  sp_df_prog5 = _spearman(alldf, np.array([r['prog5'] for r in dirs_rows]))

  # ---- actual actor optimizer step (1) + 50-step probe continuation ----
  obs_batch = np.array([r['obs'] for r in refs], np.float32)
  obs_batch = np.tile(obs_batch, (9, 1))[:256]
  key = jax.random.PRNGKey(5)
  p1, _ = actor_step(nets, cfg, state, obs_batch, key, n_steps=1)
  p50, losses50 = actor_step(nets, cfg, state, obs_batch, key,
                             n_steps=HILL_STEPS)

  def eval_params(params):
    rows = []
    for ref in refs:
      a0 = mode(state.policy_params, ref['obs'])
      a1 = mode(params, ref['obs'])
      df = f(ref['obs'], a1) - f(ref['obs'], a0)
      ro = rollout(env, ref, a1, params, mode)
      base = rollout(env, ref, a0, state.policy_params, mode)
      rows.append(dict(df=df, da=float(np.linalg.norm(a1 - a0)),
                       sat=float(np.mean(np.abs(a1) > SAT)),
                       dprog1=ro[1]['prog'] - base[1]['prog'],
                       dprog5=ro[5]['prog'] - base[5]['prog']))
    dfv = np.array([r['df'] for r in rows])
    return {'df_mean': float(dfv.mean()),
            'df_pos_frac': float(np.mean(dfv > 0)),
            'da_median': float(np.median([r['da'] for r in rows])),
            'sat_mean': float(np.mean([r['sat'] for r in rows])),
            'dprog1_mean': float(np.mean([r['dprog1'] for r in rows])),
            'dprog5_mean': float(np.mean([r['dprog5'] for r in rows])),
            'dprog5_ci': _boot_ci([r['dprog5'] for r in rows]),
            'sp_df_dprog5': _spearman(
                dfv, np.array([r['dprog5'] for r in rows]))}
  step1 = eval_params(p1)
  step50 = eval_params(p50)
  sat0 = float(np.mean([np.mean(np.abs(mode(state.policy_params, r['obs']))
                                > SAT) for r in refs]))

  rep = {
      'arm': arm, 'env': env_name, 'ckpt': ckpt, 'step': step,
      'n_states': len(refs),
      'gradient': {
          'norm_median': float(np.median(G['gnorm'])),
          'norm_p10': float(np.percentile(G['gnorm'], 10)),
          'norm_p90': float(np.percentile(G['gnorm'], 90)),
          'participation_ratio_median': float(np.median(G['prat'])),
          'top_dim_share_median': float(np.median(G['topshare'])),
          'saturated_dim_frac_median': float(np.median(G['sat'])),
          'blocked_outward_mass_median': float(np.median(G['blocked'])),
          'finite_diff_rel_err_median': float(np.median(G['fd_relerr'])),
          'f0_mean': float(np.mean(G['f0']))},
      'perturbations': {str(k): v for k, v in pert.items()},
      'score_progress_coupling': {'spearman_df_prog1': sp_df_prog1,
                                  'spearman_df_prog5': sp_df_prog5},
      'actor_step_1': step1,
      'actor_steps_50_probe': {**step50, 'loss_first': losses50[0],
                               'loss_last': losses50[-1],
                               'sat_before': sat0},
  }

  # ---- verdict ----
  gr = rep['gradient']
  d05 = pert[0.05]
  score_valid = (d05['df_plus_pos_frac'] >= 0.8 and d05['df_plus_mean'] > 0
                 and gr['finite_diff_rel_err_median'] < 0.2)
  vanishes = gr['norm_median'] < VANISH_GRAD_NORM
  drives_sat = gr['blocked_outward_mass_median'] > 0.5 or (
      step50['sat_mean'] > sat0 + 0.1)
  def margin_pos(h):
    m, ci = d05[f'{h}_margin_vs_minus']
    m2, ci2 = d05[f'{h}_margin_vs_random']
    return (ci[0] is not None and ci[0] > 0 and m > 0
            and ci2[0] is not None and ci2[0] > 0)
  aligns = score_valid and (margin_pos('prog1') or margin_pos('prog5'))
  actor_ok = step1['df_pos_frac'] >= 0.7 and step1['df_mean'] > 0
  if vanishes:
    verdict = 'CRITIC_ACTION_GRADIENT_VANISHES'
  elif drives_sat and not aligns:
    verdict = 'CRITIC_GRADIENT_DRIVES_SATURATION'
  elif aligns and not actor_ok:
    verdict = 'ACTOR_OPTIMIZATION_FAILURE'
  elif aligns:
    verdict = 'CRITIC_GRADIENT_ALIGNS_WITH_CONTROL'
  elif score_valid:
    verdict = 'CRITIC_GRADIENT_SCORE_ONLY'
  else:
    verdict = 'INCONCLUSIVE'
  rep['checks'] = {'score_valid': score_valid, 'vanishes': vanishes,
                   'drives_saturation': drives_sat, 'aligns': aligns,
                   'actor_step_increases_score': actor_ok}
  rep['verdict'] = verdict
  return rep


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--out', default=OUT)
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  rng = np.random.default_rng(0)
  reports = {}
  for arm, (env_name, ckpt, refs) in ARMS.items():
    print(f'=== {arm} ===', flush=True)
    reports[arm] = audit_arm(arm, env_name, ckpt, refs, rng)
    r = reports[arm]
    print(f'  grad norm med {r["gradient"]["norm_median"]:.3f} | blocked '
          f'{r["gradient"]["blocked_outward_mass_median"]:.2f} | '
          f'df+frac@0.05 {r["perturbations"]["0.05"]["df_plus_pos_frac"]:.2f} | '
          f'prog5 margin {r["perturbations"]["0.05"]["prog5_margin_vs_minus"][0]:+.4f} | '
          f'actor-step df+ {r["actor_step_1"]["df_pos_frac"]:.2f} | '
          f'VERDICT {r["verdict"]}', flush=True)
  json.dump(reports, open(os.path.join(args.out,
            'ant_actor_gradient_audit.json'), 'w'), indent=2)
  _md(args.out, reports)
  print('\nOVERALL:', {a: r['verdict'] for a, r in reports.items()})


def _md(out, reports):
  L = ['# Actor-gradient validity audit (30k target-entropy -8 checkpoints)\n']
  for arm, r in reports.items():
    g = r['gradient']
    L += [f'\n## Arm {arm} — **`{r["verdict"]}`** (step {r["step"]}, '
          f'{r["n_states"]} states)\n',
          f'grad norm median {g["norm_median"]:.3f} (p10 {g["norm_p10"]:.3f}, '
          f'p90 {g["norm_p90"]:.3f}); participation ratio '
          f'{g["participation_ratio_median"]:.1f}/8; top-dim share '
          f'{g["top_dim_share_median"]:.2f}; saturated dims '
          f'{g["saturated_dim_frac_median"]:.2f}; blocked outward mass '
          f'{g["blocked_outward_mass_median"]:.2f}; finite-diff rel err '
          f'{g["finite_diff_rel_err_median"]:.3f}\n',
          '| delta | df+ mean | df+>0 | prog1 margin(+/-) | prog5 margin(+/-) | '
          'prog5 margin(+/rand) | fell5 + | fell5 rand |',
          '|---|---|---|---|---|---|---|---|']
    for dl, p in r['perturbations'].items():
      m1 = p['prog1_margin_vs_minus']; m5 = p['prog5_margin_vs_minus']
      mr = p['prog5_margin_vs_random']
      L.append(f'| {dl} | {p["df_plus_mean"]:.4f} | {p["df_plus_pos_frac"]:.2f} '
               f'| {m1[0]:+.5f} [{m1[1][0]:.5f},{m1[1][1]:.5f}] '
               f'| {m5[0]:+.5f} [{m5[1][0]:.5f},{m5[1][1]:.5f}] '
               f'| {mr[0]:+.5f} [{mr[1][0]:.5f},{mr[1][1]:.5f}] '
               f'| {p["fell5_plus"]:.2f} | {p["fell5_random"]:.2f} |')
    c = r['score_progress_coupling']
    a1, a50 = r['actor_step_1'], r['actor_steps_50_probe']
    L += [f'\nscore-progress coupling: sp(df,prog1)={c["spearman_df_prog1"]:.3f}, '
          f'sp(df,prog5)={c["spearman_df_prog5"]:.3f}',
          f'\nactor step x1: df_mean {a1["df_mean"]:.4f} (pos frac '
          f'{a1["df_pos_frac"]:.2f}), |da| med {a1["da_median"]:.5f}, '
          f'dprog5 {a1["dprog5_mean"]:+.5f}',
          f'actor steps x50 (probe copy): df_mean {a50["df_mean"]:.3f} (pos '
          f'{a50["df_pos_frac"]:.2f}), |da| med {a50["da_median"]:.4f}, sat '
          f'{a50["sat_before"]:.2f}->{a50["sat_mean"]:.2f}, dprog5 '
          f'{a50["dprog5_mean"]:+.5f} {a50["dprog5_ci"]}, '
          f'sp(df,dprog5) {a50["sp_df_dprog5"]}']
  L.append('\n(all rollouts from bit-exact restored states; +/-grad and '
           'random directions norm-matched before clipping; receding horizons '
           'resume the deterministic actor after the first step)')
  open(os.path.join(out, 'ant_actor_gradient_audit.md'), 'w').write(
      '\n'.join(L) + '\n')


if __name__ == '__main__':
  main()
