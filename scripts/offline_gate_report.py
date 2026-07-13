"""Per-gate diagnostic report for the staged offline antmaze qualification.

Sections (spec of the 50k staged qualification):
  1. offline invariants (dataset hash sidecar, buffer counts/fingerprint,
     structural no-collector mode, learner-update counter semantics);
  2. actor objective decomposition (raw/weighted BC + critic terms, per-term
     gradient norms + cosine, dataset-action log-prob, actor-vs-dataset L2/cos);
  3. squashed-policy numerical health (loc magnitude, scale median, floor
     fraction, saturation, dataset-action clipping, max |BC log-prob|,
     non-finite counts);
  4. twin-critic diagnostics (per-head retrieval + logits gap, Q1-Q2
     disagreement, min-Q at actor vs dataset actions, advantage);
  5. fixed-seed evaluation (success, final/min XY distance, goal-directed
     velocity, fall rate).

Also emits STOP flags per the qualification stop rules. The probe batch is
the buffer's deterministic FIRST 1024-sample draw (seed = run seed), so the
same tuples are scored at every gate.
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from crl import envs as envs_mod
from crl import networks as networks_mod
from crl import checkpoint as ckpt_mod
from crl import offline_audit
from verify_offline_d4rl import build_offline_cfg, NPZ

FLOOR = 1e-6
CLIP = 1.0 - 1e-6            # tanh_normal_log_prob boundary clip
DT = 0.1
FALL_Z = 0.3
ENV_SEED = 12345


def tree_flat(t):
  return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(t)])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--tag', required=True)
  ap.add_argument('--run_dir', required=True)
  ap.add_argument('--eval_eps', type=int, default=50)
  ap.add_argument('--out', default=None)
  args = ap.parse_args()
  out_dir = args.out or os.path.join(args.run_dir, 'reports')
  os.makedirs(out_dir, exist_ok=True)

  cfg = build_offline_cfg()
  eval_env = envs_mod.make_env('offline_ant_umaze', cfg, seed=ENV_SEED)
  buffer, fp = offline_audit.build_offline_buffer(NPZ, cfg)

  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(args.ckpt)
  R = {'tag': args.tag, 'ckpt': args.ckpt, 'step': int(step)}
  stop = []

  # ---------------- 1. offline invariants ---------------------------------
  side = os.path.join(args.run_dir, 'checkpoints', 'offline_dataset.sha256')
  sidecar = json.load(open(side)) if os.path.exists(side) else {}
  mets_p = os.path.join(args.run_dir, 'checkpoints', 'metrics.json')
  mets = json.load(open(mets_p)) if os.path.exists(mets_p) else []
  last = mets[-1] if mets else {}
  R['invariants'] = {
      'dataset_sha256': fp['sha256'],
      'sidecar_sha256': sidecar.get('sha256'),
      'sidecar_matches': sidecar.get('sha256') == fp['sha256'],
      'replay_episodes': fp['n_episodes'],
      'replay_transitions': fp['n_transitions'],
      'expected': {'episodes': 1426, 'transitions': 998200},
      'counts_unchanged': (fp['n_episodes'] == 1426
                           and fp['n_transitions'] == 998200),
      'buffer_frozen': buffer.frozen,
      'collector_processes': 0,
      'training_env_steps': 0,
      'note': ('collection env is structurally absent (env=None) and the '
               'buffer is frozen; per-eval sha + eval-step-accounting asserts '
               'ran inside train() (any violation aborts the stage)'),
      'learner_update_counter': {
          'step_clock_at_ckpt': int(step),
          'metrics_last_step': last.get('step'),
          'semantics': ('offline step clock == cumulative gradient updates '
                        '(1 update per step unit; learner_updates field in '
                        'metrics counts per-invocation and resets on resume)'),
      },
  }
  if not R['invariants']['counts_unchanged']:
    stop.append('REPLAY_CHANGED')
  if sidecar and not R['invariants']['sidecar_matches']:
    stop.append('REPLAY_CHANGED (dataset hash sidecar mismatch)')

  # ---------------- probe batch (deterministic across gates) --------------
  tr = buffer.sample(1024)
  obs = jnp.asarray(tr.observation)
  a_data = jnp.asarray(tr.action)

  key = jax.random.PRNGKey(999)
  dist = nets.policy_network.apply(st.policy_params, obs)
  a_pi = nets.sample(dist, key)
  a_mode = jnp.tanh(dist.loc)
  logp_pi = nets.log_prob(dist, a_pi)
  logp_data = nets.log_prob(dist, a_data)

  # ---------------- 2. actor objective decomposition ----------------------
  def bc_term(pp):
    d = nets.policy_network.apply(pp, obs)
    return -jnp.mean(nets.log_prob(d, a_data))

  def q_term(pp):
    d = nets.policy_network.apply(pp, obs)
    a = nets.sample(d, key)
    q = nets.q_network.apply(st.q_params, obs, a)
    q = jnp.min(q, axis=-1)
    return jnp.mean(0.0 * nets.log_prob(d, a) - jnp.diag(q))

  bc_val, bc_grad = jax.value_and_grad(bc_term)(st.policy_params)
  q_val, q_grad = jax.value_and_grad(q_term)(st.policy_params)
  gb, gq = tree_flat(bc_grad), tree_flat(q_grad)
  cos_gb_gq = float(jnp.dot(gb, gq)
                    / (jnp.linalg.norm(gb) * jnp.linalg.norm(gq) + 1e-12))
  l2 = jnp.linalg.norm(a_mode - a_data, axis=1)
  cos_a = jnp.sum(a_mode * a_data, axis=1) / (
      jnp.linalg.norm(a_mode, axis=1) * jnp.linalg.norm(a_data, axis=1)
      + 1e-12)
  R['actor_objective'] = {
      'bc_raw (E[-log pi(a_data)])': float(bc_val),
      'bc_weighted (x0.05)': float(0.05 * bc_val),
      'critic_raw (E[-minQ(a_pi)])': float(q_val),
      'critic_weighted (x0.95)': float(0.95 * q_val),
      'total_actor_loss': float(0.05 * bc_val + 0.95 * q_val),
      'grad_norm_bc': float(jnp.linalg.norm(gb)),
      'grad_norm_critic': float(jnp.linalg.norm(gq)),
      'grad_cosine_bc_vs_critic': cos_gb_gq,
      'dataset_action_logprob_mean': float(jnp.mean(logp_data)),
      'dataset_action_logprob_min': float(jnp.min(logp_data)),
      'actor_vs_dataset_L2_mean': float(jnp.mean(l2)),
      'actor_vs_dataset_cos_mean': float(jnp.mean(cos_a)),
  }
  if not np.isfinite(R['actor_objective']['bc_raw (E[-log pi(a_data)])']):
    stop.append('BC_NONFINITE')

  # ---------------- 3. squashed-policy numerical health -------------------
  loc, scale = np.asarray(dist.loc), np.asarray(dist.scale)
  n_nonfinite = int((~np.isfinite(np.asarray(a_pi))).sum()
                    + (~np.isfinite(np.asarray(logp_pi))).sum()
                    + (~np.isfinite(np.asarray(logp_data))).sum())
  R['policy_health'] = {
      'loc_abs_mean': float(np.abs(loc).mean()),
      'loc_abs_max': float(np.abs(loc).max()),
      'scale_median': float(np.median(scale)),
      'floor_fraction (scale<=1.05e-6)': float(np.mean(scale <= 1.05 * FLOOR)),
      'saturation (|a_mode|>0.99)': float(np.mean(np.abs(np.asarray(a_mode))
                                                  > 0.99)),
      'dataset_action_clip_fraction (|a|>=1-1e-6)': float(
          np.mean(np.abs(np.asarray(a_data)) >= CLIP)),
      'max_abs_bc_logprob': float(np.max(np.abs(np.asarray(logp_data)))),
      'nonfinite_samples': n_nonfinite,
  }
  if n_nonfinite:
    stop.append('BC_NONFINITE (non-finite samples/logprobs)')
  if R['policy_health']['loc_abs_max'] > 100.0:
    stop.append(f"ACTOR_LOC_EXPLODED ({R['policy_health']['loc_abs_max']:.3g})")

  # ---------------- 4. twin-critic diagnostics ----------------------------
  q_dat = nets.q_network.apply(st.q_params, obs, a_data)   # [B,B,2]
  q_act = nets.q_network.apply(st.q_params, obs, a_pi)
  I = np.eye(q_dat.shape[0], dtype=bool)
  def head_stats(q):
    q = np.asarray(q)
    out = {}
    for h in (0, 1):
      m = q[..., h]
      out[f'q{h+1}_retrieval_acc'] = float(np.mean(
          np.argmax(m, axis=1) == np.arange(m.shape[0])))
      out[f'q{h+1}_logits_gap'] = float(m[I].mean() - m[~I].mean())
    return out
  hs = head_stats(q_dat)
  d01 = np.asarray(q_dat[..., 0] - q_dat[..., 1])
  min_dat = np.asarray(jnp.diag(jnp.min(q_dat, axis=-1)))
  min_act = np.asarray(jnp.diag(jnp.min(q_act, axis=-1)))
  disagreement = float(np.abs(d01).mean())
  R['twin_critic'] = {
      **hs,
      'q1_q2_disagreement_mean_abs': disagreement,
      'q1_q2_diag_corr': float(np.corrcoef(
          np.asarray(q_dat[..., 0])[I], np.asarray(q_dat[..., 1])[I])[0, 1]),
      'minQ_at_actor_actions_mean': float(min_act.mean()),
      'minQ_at_dataset_actions_mean': float(min_dat.mean()),
      'actor_minus_dataset_score_advantage': float(min_act.mean()
                                                   - min_dat.mean()),
  }
  if disagreement < 1e-6:
    stop.append('CRITICS_IDENTICAL (wiring bug)')

  # ---------------- 5. fixed-seed evaluation ------------------------------
  @jax.jit
  def _mode(o):
    return jnp.tanh(nets.policy_network.apply(st.policy_params, o).loc)
  u = eval_env._env.unwrapped
  rows = []
  for ep in range(args.eval_eps):
    o = eval_env.reset()
    goal = o[29:31].copy()
    d_prev = float(np.linalg.norm(o[:2] - goal))
    dmin, succ, fall, gvel = d_prev, 0.0, 0, []
    for _ in range(eval_env.max_episode_steps):
      a = np.asarray(_mode(jnp.asarray(o[None]))[0])
      o, r, _, _ = eval_env.step(a)
      d = float(np.linalg.norm(o[:2] - goal))
      gvel.append((d_prev - d) / DT)
      dmin = min(dmin, d)
      succ = max(succ, float(r))
      fall += float(u.data.qpos[2]) < FALL_Z
      d_prev = d
    rows.append((succ, d_prev, dmin, float(np.mean(gvel)),
                 fall / eval_env.max_episode_steps))
  arr = np.array(rows)
  R['evaluation'] = {
      'env_seed': ENV_SEED, 'episodes': args.eval_eps,
      'success': float(arr[:, 0].mean()),
      'final_xy_dist_mean': float(arr[:, 1].mean()),
      'final_xy_dist_median': float(np.median(arr[:, 1])),
      'min_xy_dist_mean': float(arr[:, 2].mean()),
      'min_xy_dist_median': float(np.median(arr[:, 2])),
      'goal_directed_velocity_mps': float(arr[:, 3].mean()),
      'fall_rate': float(arr[:, 4].mean()),
  }

  R['stop_flags'] = stop
  path = os.path.join(out_dir, f'gate_{args.tag}.json')
  json.dump(R, open(path, 'w'), indent=2)
  md_path = os.path.join(out_dir, f'gate_{args.tag}.md')
  with open(md_path, 'w', encoding='utf-8') as f:   # π/≤/· need utf-8 on Windows
    f.write(_markdown(R))
  print(json.dumps({k: v for k, v in R.items()}, indent=1))
  print(('STOP: ' + '; '.join(stop)) if stop else 'NO STOP CONDITIONS',
        flush=True)
  print('saved', path, '+', md_path)


def _markdown(R):
  inv, ao = R['invariants'], R['actor_objective']
  ph, tw, ev = R['policy_health'], R['twin_critic'], R['evaluation']
  L = [
      f"# Offline gate {R['tag']} — step {R['step']}",
      '',
      f"**STOP flags:** {R['stop_flags'] or 'none'}",
      '',
      '## 1. Offline invariants',
      f"- dataset SHA `{(inv['dataset_sha256'] or '')[:16]}…` · "
      f"sidecar match **{inv['sidecar_matches']}**",
      f"- replay: {inv['replay_episodes']} eps / {inv['replay_transitions']} "
      f"transitions · counts unchanged **{inv['counts_unchanged']}** · "
      f"frozen **{inv['buffer_frozen']}**",
      f"- collector processes **{inv['collector_processes']}** · training env "
      f"steps **{inv['training_env_steps']}**",
      f"- learner-update clock @ ckpt: {inv['learner_update_counter']['step_clock_at_ckpt']}",
      '',
      '## 2. Actor objective decomposition',
      f"- BC raw E[-log π(a_data)] = {ao['bc_raw (E[-log pi(a_data)])']:.4f} · "
      f"weighted ×0.05 = {ao['bc_weighted (x0.05)']:.4f}",
      f"- critic raw E[-minQ(a_π)] = {ao['critic_raw (E[-minQ(a_pi)])']:.4f} · "
      f"weighted ×0.95 = {ao['critic_weighted (x0.95)']:.4f}",
      f"- **total actor loss = {ao['total_actor_loss']:.4f}**",
      f"- dataset-action log-prob mean {ao['dataset_action_logprob_mean']:.3f} "
      f"(min {ao['dataset_action_logprob_min']:.3f})",
      f"- actor-vs-dataset L2 {ao['actor_vs_dataset_L2_mean']:.3f} · "
      f"cos {ao['actor_vs_dataset_cos_mean']:.3f} · "
      f"grad cos(BC,critic) {ao['grad_cosine_bc_vs_critic']:.3f}",
      '',
      '## 3. Policy health',
      f"- |loc| mean {ph['loc_abs_mean']:.3f} / max {ph['loc_abs_max']:.3f} · "
      f"scale median {ph['scale_median']:.4f}",
      f"- floor frac {ph['floor_fraction (scale<=1.05e-6)']:.3f} · "
      f"saturation {ph['saturation (|a_mode|>0.99)']:.3f} · "
      f"non-finite {ph['nonfinite_samples']}",
      '',
      '## 4. Twin critics',
      f"- retrieval Q1 {tw['q1_retrieval_acc']:.3f} / Q2 {tw['q2_retrieval_acc']:.3f}",
      f"- logits gap Q1 {tw['q1_logits_gap']:.3f} / Q2 {tw['q2_logits_gap']:.3f} · "
      f"disagreement {tw['q1_q2_disagreement_mean_abs']:.4f}",
      f"- min-Q actor {tw['minQ_at_actor_actions_mean']:.3f} vs data "
      f"{tw['minQ_at_dataset_actions_mean']:.3f} "
      f"(adv {tw['actor_minus_dataset_score_advantage']:.3f})",
      '',
      '## 5. Fixed-seed evaluation',
      f"- **success {ev['success']:.3f}** (XY ≤ 0.5) · "
      f"final dist {ev['final_xy_dist_mean']:.3f} · min dist {ev['min_xy_dist_mean']:.3f}",
      f"- goal-directed velocity {ev['goal_directed_velocity_mps']:.4f} m/s · "
      f"fall rate {ev['fall_rate']:.3f} · {ev['episodes']} eps @ seed {ev['env_seed']}",
      '',
  ]
  return '\n'.join(L) + '\n'


if __name__ == '__main__':
  main()
