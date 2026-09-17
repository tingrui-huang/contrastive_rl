"""Does the actor's real training objective prefer the successful policy?

Earlier audits asked whether a single action (DOWN) scores above another
(RIGHT) at the fork.  This one asks the question the optimizer answers: for
WHOLE trained policies, under the exact training rule

    L(pi) = 0.95 * E_{rows ~ buffer law}[ -f(s, a, g) ],  a ~ pi(.|s, g)
          + 0.05 * E_{rows ~ balanced BC law}[ -log pi(a_data | s, g) ],

is the failed actor's L lower than the successful actor's?  Same critic, same
rows, same innovations for every actor; the actor's own sampling width; Acme
log-prob; the actual balanced BC weights (cap 0.25).  Rows are drawn from the
enumerated relabeling law (checked against the buffer in crl/bc_balanced.py),
so the critic-term rows follow the buffer's (state, future goal) law and the
BC rows follow the balanced law -- the two distributions the training loop
actually uses.

Reported per (critic, actor): critic term, BC term, total, on
  * the full training distribution;
  * the approach rows (anchor in cells (0,3) or (1,3)), drawn conditionally so
    the estimate is not swamped by parked / terminal rows;
  * the fork rows (anchor in (1,3));
  * the fork rows whose relabeled goal is in cell (8,3) -- the training pairs
    closest to the task goal;
and, separately and labelled as such, the task-goal check: the 16 held-out
roots with the canonical goal substituted (critic term only, there is no
data action for a substituted goal).

Differences between actors are paired over rows (same rows, same eps), with
standard errors.

Usage::
  python scripts/audit_f4_policy_objective.py            # D2 trio, all critics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / 'outputs' / 'pointmaze_diagonal_coverage_fix_v1'
J = ROOT / 'outputs' / 'pointmaze_balanced_bc_joint_v1'
SEALED = ROOT / 'outputs' / 'pointmaze_ett_query_coverage_20260915_v1'
OUT = J / 'policy_objective_audit'
BC = 0.05
CAP = 0.25
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)

ACTORS = {   # the D2 trio: new-seed mode lower route 0.813 / 0.007 / 0.880
    'D2 actor s0 (lower 0.81)': J / 'refreeze' / 'critic_D2' / 'actor_s0' / 'final.pkl',
    'D2 actor s1 (lower 0.01)': J / 'refreeze' / 'critic_D2' / 'actor_s1' / 'final.pkl',
    'D2 actor s2 (lower 0.88)': J / 'refreeze' / 'critic_D2' / 'actor_s2' / 'final.pkl',
}
CRITICS = {
    'D2 (their own critic)': FIX / 'seeds' / 'seed_2' / 'crl' / 'D' / 'final.pkl',
    'D1 (3/3 detour)': FIX / 'seeds' / 'seed_1' / 'crl' / 'D' / 'final.pkl',
    'D0 (3/3 detour)': FIX / 'seeds' / 'seed_0' / 'crl' / 'D' / 'final.pkl',
    'joint30k s0 (3/3)': J / 'joint' / 'b30k' / 'seed_0' / 'balanced' / 'final.pkl',
    'joint30k s1 (0/3)': J / 'joint' / 'b30k' / 'seed_1' / 'balanced' / 'final.pkl',
    'joint30k s2 (2/3)': J / 'joint' / 'b30k' / 'seed_2' / 'balanced' / 'final.pkl',
    'joint300k s0 (0/3)': J / 'joint' / 'b300k' / 'seed_0' / 'balanced' / 'final.pkl',
    'C seed 2 (0/3)': SEALED / 'cross_seeds' / 'seed_2' / 'crl' / 'C' / 'final.pkl',
}


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--rows', type=int, default=60_000, help='rows per distribution')
  ap.add_argument('--cond-rows', type=int, default=20_000, help='rows per conditional subset')
  ap.add_argument('--k', type=int, default=32, help='policy samples per row for the critic term')
  ap.add_argument('--seed', type=int, default=777)
  args = ap.parse_args(argv)
  import jax
  import jax.numpy as jnp
  from crl import checkpoint, networks
  from crl.bc_balanced import GroupBalancedBCSampler
  nets = networks.make_networks(
      8, 8, 2, repr_dim=64, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=(256, 256), actor_min_std=1e-6, twin_q=False,
      use_image_obs=False, use_layer_norm=False, obs_scale=None, log_prob_mode='acme')
  rng = np.random.default_rng(args.seed)

  with np.load(FIX / 'replay_D.npz', allow_pickle=False) as d:
    obs, act, lengths = d['obs'], d['act'], d['lengths']
  smp = GroupBalancedBCSampler(obs, act, lengths, 0.95, 8, cap=CAP, seed=args.seed)
  w_law, w_bal = smp.w_original, smp.w
  s_cell = np.floor(obs[smp.tr, smp.ii, :2]).astype(np.int64)
  g_cell = np.floor(obs[smp.tr, smp.jj, :2]).astype(np.int64)
  approach = ((s_cell[:, 0] == 0) | (s_cell[:, 0] == 1)) & (s_cell[:, 1] == 3)
  fork = (s_cell[:, 0] == 1) & (s_cell[:, 1] == 3)
  fork_g83 = fork & (g_cell[:, 0] == 8) & (g_cell[:, 1] == 3)

  def draw(w, n, mask=None):
    ww = w if mask is None else np.where(mask, w, 0.0)
    cdf = np.cumsum(ww / ww.sum())
    pos = np.minimum(np.searchsorted(cdf, rng.random(n), side='right'), len(cdf) - 1)
    tr, i, j = smp.tr[pos], smp.ii[pos].astype(np.int64), smp.jj[pos].astype(np.int64)
    o = np.concatenate([obs[tr, i, :8], obs[tr, j, :8]], 1).astype(np.float32)
    return o, act[tr, i].astype(np.float32), (s_cell[pos], g_cell[pos])

  # identical rows for every actor and critic
  sets = {
      'full': (draw(w_law, args.rows), draw(w_bal, args.rows)),
      'approach (0,3)+(1,3)': (draw(w_law, args.cond_rows, approach), draw(w_bal, args.cond_rows, approach)),
      'fork (1,3)': (draw(w_law, args.cond_rows, fork), draw(w_bal, args.cond_rows, fork)),
      'fork -> goal (8,3)': (draw(w_law, args.cond_rows, fork_g83), draw(w_bal, args.cond_rows, fork_g83)),
  }
  masses = {'approach (0,3)+(1,3)': (float(w_law[approach].sum()), float(w_bal[approach].sum())),
            'fork (1,3)': (float(w_law[fork].sum()), float(w_bal[fork].sum())),
            'fork -> goal (8,3)': (float(w_law[fork_g83].sum()), float(w_bal[fork_g83].sum()))}
  roots = np.load(ROOT / 'outputs' / 'pointmaze_matched_fork_20260914_v1'
                  / 'root_selection.npz')['state'].astype(np.float32)
  task_obs = np.concatenate([roots, np.broadcast_to(GOAL, roots.shape)], 1).astype(np.float32)
  eps = jnp.asarray(rng.standard_normal((args.k, 2)).astype(np.float32))   # shared innovations

  @jax.jit
  def critic_term(q_params, policy_params, o):
    dist = nets.policy_network.apply(policy_params, o)
    a = jnp.tanh(dist.loc[:, None, :] + dist.scale[:, None, :] * eps[None])   # [B, K, 2]
    o_rep = jnp.repeat(o[:, None, :], eps.shape[0], 1).reshape(-1, o.shape[1])
    phi, psi = nets.representation_network.apply(q_params, o_rep, a.reshape(-1, 2))
    q = jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], eps.shape[0])
    return -jnp.mean(q, axis=1)                                            # [B]

  @jax.jit
  def bc_term(policy_params, o, a_data):
    dist = nets.policy_network.apply(policy_params, o)
    return -nets.log_prob(dist, a_data)                                     # [B]

  def batched(fn, params, *arrays, bs=4096):
    out = []
    for k in range(0, len(arrays[0]), bs):
      out.append(np.asarray(fn(*params, *[jnp.asarray(x[k:k + bs]) for x in arrays])))
    return np.concatenate(out)

  actors = {name: checkpoint.load_checkpoint(p)[1].policy_params for name, p in ACTORS.items()}
  critics = {name: checkpoint.load_checkpoint(p)[1].q_params for name, p in CRITICS.items()}
  actor_names = list(actors)
  results = {'settings': vars(args), 'masses': masses, 'per_set': {}}
  lines = ['# Whole-policy objective: do the training rules prefer the successful actor?', '',
           f'{args.rows:,} rows per distribution ({args.cond_rows:,} per conditional subset), '
           f'{args.k} policy samples per row, shared innovations; BC weight {BC}, cap {CAP}. '
           'critic term = 0.95 * E[-f(s, a~pi, g)] on the buffer law; BC term = 0.05 * '
           'E[-log pi(a_data|s,g)] on the balanced law; total = sum. Lower is preferred by '
           'training. Differences are paired over rows; +-1 s.e. in brackets.', '',
           'Masses (buffer law / balanced law): ' + ', '.join(
               f'{k}: {v[0]:.4f} / {v[1]:.4f}' for k, v in masses.items()), '']
  # BC terms depend only on the actor -> compute once per set
  bc_vals = {sname: {a: batched(bc_term, (actors[a],), sets[sname][1][0], sets[sname][1][1])
                     for a in actor_names} for sname in sets}
  for sname in sets:
    (o_c, _, _), _ = sets[sname]
    lines += [f'## {sname}', '',
              '| critic | actor | critic term | BC term | total | total - total(s1 failed) |',
              '|---|---|---:|---:|---:|---|']
    results['per_set'][sname] = {}
    for cname, qp in critics.items():
      c_vals = {a: batched(critic_term, (qp, actors[a]), o_c) for a in actor_names}
      tot = {a: BC * bc_vals[sname][a].mean() + (1 - BC) * c_vals[a].mean() for a in actor_names}
      ref = actor_names[1]
      for a in actor_names:
        # paired difference of the total vs the failed actor: rows are shared per term
        dc = (1 - BC) * (c_vals[a] - c_vals[ref])
        db = BC * (bc_vals[sname][a] - bc_vals[sname][ref])
        diff = dc.mean() + db.mean()
        se = float(np.sqrt(dc.var() / len(dc) + db.var() / len(db)))
        results['per_set'][sname][f'{cname} | {a}'] = {
            'critic_term': float((1 - BC) * c_vals[a].mean()),
            'bc_term': float(BC * bc_vals[sname][a].mean()), 'total': float(tot[a]),
            'diff_vs_failed': float(diff), 'diff_se': se}
        lines.append(f'| {cname} | {a} | {(1 - BC) * c_vals[a].mean():+.4f} '
                     f'| {BC * bc_vals[sname][a].mean():+.4f} | {tot[a]:+.4f} '
                     f'| {diff:+.4f} [{se:.4f}] |')
    lines.append('')
  # task-goal check: critic term only, 16 roots with the canonical goal
  lines += ['## Task-goal check (16 held-out roots, canonical goal substituted; critic term only, '
            'no data action exists for a substituted goal)', '',
            '| critic | ' + ' | '.join(actor_names) + ' |', '|---|' + '---:|' * len(actor_names)]
  results['task_goal'] = {}
  for cname, qp in critics.items():
    vals = {a: float((1 - BC) * np.asarray(critic_term(qp, actors[a], jnp.asarray(task_obs))).mean())
            for a in actor_names}
    results['task_goal'][cname] = vals
    lines.append(f'| {cname} | ' + ' | '.join(f'{vals[a]:+.4f}' for a in actor_names) + ' |')
  # the actors' modes at the roots, for the record
  lines += ['', 'Mode at the roots (task goal): ' + '; '.join(
      f'{a}: ({m[0]:+.2f}, {m[1]:+.2f})' for a, m in (
          (a, np.tanh(np.asarray(nets.policy_network.apply(actors[a], jnp.asarray(task_obs)).loc)).mean(0))
          for a in actor_names))]
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
  (OUT / 'results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
  print('\n'.join(lines))
  return 0


if __name__ == '__main__':
  sys.exit(main())
