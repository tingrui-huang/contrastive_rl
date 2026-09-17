"""Why the RIGHT-family critics (joint 30k seed 1, joint 300k seed 0) block the actor.

Step 9f left one family unexplained: two critics that rank DOWN first on all
16 held-out roots (joint 30k balanced seed 1, +0.25; joint 300k balanced seed
0, +0.70) and prefer DOWN over the corner at the policy width, yet every fresh
balanced-BC actor trained on them rests at (+1.00, y > -0.5) and never
detours.  This audit asks, on the rows the actor is actually trained on:

A. Objective matrix.  The exact training objective
       L(pi) = 0.95 E_{buffer law}[-f(s, a~pi, g)] + 0.05 E_{balanced law}[-log pi(a_data|s, g)]
   for every two-stage actor of the DOWN family (D1 x3, joint 30k s0 x3) and
   the RIGHT family (joint 30k s1 x3, joint 300k s0 x3) under each of the four
   critics, on the full distribution and on the fork rows split by goal cell.
   If a RIGHT critic scores the DOWN actors LOWER than its own actors, the
   objective wanted DOWN and the optimizer failed; if it scores its own actors
   lower, the objective itself prefers RIGHT and part B says where.

B. Attribution.  The critic-term difference (own RIGHT actor minus a DOWN
   actor) decomposed over (anchor cell, goal cell) groups, mass-weighted, so
   the groups that pay for RIGHT are named.

C. Fork rows by goal cell.  For the replay's fork anchors, per goal cell:
   Qbar at the policy width for the DOWN loc, the RIGHT loc, the UP-RIGHT loc
   the RIGHT actors rest at, and the corner; plus the modes of one actor of
   each family under those goals.  This is the critic's fork preference on the
   training goals, not on the substituted task goal.

Outputs: outputs/pointmaze_balanced_bc_joint_v1/right_family_audit/{REPORT.md,results.json}.
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
STEP8 = ROOT / 'outputs' / 'pointmaze_fixed_dcritic_actor_bcbal_v1' / 'critic_D1' / 'balanced'
OUT = J / 'right_family_audit'
BC = 0.05
CAP = 0.25
WIDTH = 0.75
GOAL = np.tile(np.array([8.5, 3.5], np.float32), 4)
LOCS = {'DOWN (0.3,-6)': (0.3, -6.0), 'RIGHT (5,-0.1)': (5.0, -0.1),
        'UPRIGHT (4,+0.3)': (4.0, 0.3), 'CORNER (5,-5)': (5.0, -5.0)}

CRITICS = {
    'R1: joint30k s1': J / 'joint' / 'b30k' / 'seed_1' / 'balanced' / 'final.pkl',
    'R2: joint300k s0': J / 'joint' / 'b300k' / 'seed_0' / 'balanced' / 'final.pkl',
    'D-a: joint30k s0': J / 'joint' / 'b30k' / 'seed_0' / 'balanced' / 'final.pkl',
    'D-b: D1 (4080)': FIX / 'seeds' / 'seed_1' / 'crl' / 'D' / 'final.pkl',
}
ACTORS = {}
for a in range(3):
  ACTORS[f'R1 actor {a}'] = J / 'refreeze' / 'critic_b30k_s1' / f'actor_s{a}' / 'final.pkl'
for a in range(3):
  ACTORS[f'R2 actor {a}'] = J / 'refreeze' / 'critic_b300k_s0' / f'actor_s{a}' / 'final.pkl'
for a in range(3):
  ACTORS[f'D-a actor {a}'] = J / 'refreeze' / 'critic_b30k_s0' / f'actor_s{a}' / 'final.pkl'
for a in range(3):
  ACTORS[f'D-b actor {a}'] = STEP8 / f'actor_s{a}' / 'final.pkl'
FAMILY = {'R1': 'RIGHT', 'R2': 'RIGHT', 'D-a': 'DOWN', 'D-b': 'DOWN'}
OWN = {'R1: joint30k s1': 'R1', 'R2: joint300k s0': 'R2',
       'D-a: joint30k s0': 'D-a', 'D-b: D1 (4080)': 'D-b'}

MIDDLE = {(x, 3) for x in range(2, 8)}
LOWER = {(1, 2), (1, 1), (7, 2)} | {(x, 1) for x in range(2, 8)}


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--rows', type=int, default=40_000)
  ap.add_argument('--cond-rows', type=int, default=12_000)
  ap.add_argument('--k', type=int, default=16)
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
  fork = (s_cell[:, 0] == 1) & (s_cell[:, 1] == 3)
  g_key = g_cell[:, 0] * 10 + g_cell[:, 1]
  in_set = lambda cells: np.isin(g_key, [x * 10 + y for x, y in cells])
  subsets = {
      'full': None,
      'fork (1,3), all goals': fork,
      'fork -> goal (8,3)': fork & in_set({(8, 3)}),
      'fork -> middle-route goals (2..7,3)': fork & in_set(MIDDLE),
      'fork -> lower-route goals': fork & in_set(LOWER),
      'fork -> goal (1,3)': fork & in_set({(1, 3)}),
  }

  def draw(w, n, mask=None):
    ww = w if mask is None else np.where(mask, w, 0.0)
    cdf = np.cumsum(ww / ww.sum())
    pos = np.minimum(np.searchsorted(cdf, rng.random(n), side='right'), len(cdf) - 1)
    tr, i, j = smp.tr[pos], smp.ii[pos].astype(np.int64), smp.jj[pos].astype(np.int64)
    o = np.concatenate([obs[tr, i, :8], obs[tr, j, :8]], 1).astype(np.float32)
    return o, act[tr, i].astype(np.float32), pos

  sets, masses = {}, {}
  for name, mask in subsets.items():
    n = args.rows if mask is None else args.cond_rows
    sets[name] = (draw(w_law, n, mask), draw(w_bal, n, mask))
    masses[name] = (1.0, 1.0) if mask is None else (float(w_law[mask].sum()), float(w_bal[mask].sum()))
  eps = jnp.asarray(rng.standard_normal((args.k, 2)).astype(np.float32))
  eps_fixed = jnp.asarray(rng.standard_normal((256, 2)).astype(np.float32))

  @jax.jit
  def critic_term(q_params, policy_params, o):
    dist = nets.policy_network.apply(policy_params, o)
    a = jnp.tanh(dist.loc[:, None, :] + dist.scale[:, None, :] * eps[None])
    o_rep = jnp.repeat(o[:, None, :], eps.shape[0], 1).reshape(-1, o.shape[1])
    phi, psi = nets.representation_network.apply(q_params, o_rep, a.reshape(-1, 2))
    q = jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], eps.shape[0])
    return -jnp.mean(q, axis=1)

  @jax.jit
  def bc_term(policy_params, o, a_data):
    dist = nets.policy_network.apply(policy_params, o)
    return -nets.log_prob(dist, a_data)

  @jax.jit
  def qbar_at(q_params, o, loc):
    a = jnp.tanh(loc[None, None, :] + WIDTH * eps_fixed[None])          # [1, 256, 2]
    a = jnp.broadcast_to(a, (o.shape[0],) + a.shape[1:])
    o_rep = jnp.repeat(o[:, None, :], eps_fixed.shape[0], 1).reshape(-1, o.shape[1])
    phi, psi = nets.representation_network.apply(q_params, o_rep, a.reshape(-1, 2))
    return jnp.mean(jnp.sum(phi * psi, axis=1)[:, 0].reshape(o.shape[0], -1), axis=1)

  @jax.jit
  def modes(policy_params, o):
    d = nets.policy_network.apply(policy_params, o)
    return jnp.tanh(d.loc), d.loc, d.scale

  def batched(fn, params, *arrays, bs=2048):
    out = []
    for k in range(0, len(arrays[0]), bs):
      out.append(np.asarray(fn(*params, *[jnp.asarray(x[k:k + bs]) for x in arrays])))
    return np.concatenate(out)

  actors = {n: checkpoint.load_checkpoint(p)[1].policy_params for n, p in ACTORS.items()}
  critics = {n: checkpoint.load_checkpoint(p)[1].q_params for n, p in CRITICS.items()}
  anames = list(actors)
  res = {'settings': vars(args), 'masses': masses, 'A': {}, 'B': {}, 'C': {}}
  L = ['# The RIGHT family: what the training rows say', '',
       f'{args.rows:,} rows on the full distribution, {args.cond_rows:,} per conditional '
       f'subset, {args.k} policy samples per row (shared innovations); BC {BC}, cap {CAP}; '
       'critic term on the buffer law, BC term on the balanced law; lower = preferred by '
       'training.  Masses (buffer law): ' + ', '.join(
           f'{k} {v[0]:.4f}' for k, v in masses.items() if k != 'full'), '']

  # ---- A. objective matrix ------------------------------------------------
  bc_vals = {s: {a: batched(bc_term, (actors[a],), sets[s][1][0], sets[s][1][1])
                 for a in anames} for s in sets}
  L += ['## A. Objective matrix: rows = critic, columns = actor family (mean over the 3 actors)', '']
  fams = ['R1', 'R2', 'D-a', 'D-b']
  for s in sets:
    (o_c, _, _), _ = sets[s]
    L += [f'### {s}', '',
          '| critic | ' + ' | '.join(f'{f} actors ({FAMILY[f]})' for f in fams)
          + ' | own family - best DOWN family (total) | critic term only |',
          '|---|' + '---:|' * (len(fams) + 2)]
    res['A'][s] = {}
    for cn, qp in critics.items():
      c_vals = {a: batched(critic_term, (qp, actors[a]), o_c) for a in anames}
      tot = {a: BC * bc_vals[s][a].mean() + (1 - BC) * c_vals[a].mean() for a in anames}
      ct = {a: float((1 - BC) * c_vals[a].mean()) for a in anames}
      fam_tot = {f: float(np.mean([tot[a] for a in anames if a.startswith(f + ' ')])) for f in fams}
      fam_ct = {f: float(np.mean([ct[a] for a in anames if a.startswith(f + ' ')])) for f in fams}
      own = OWN[cn]
      best_down = min(fam_tot['D-a'], fam_tot['D-b'])
      best_down_ct = min(fam_ct['D-a'], fam_ct['D-b'])
      # paired s.e. of own-vs-D-a difference, actor 0 vs actor 0
      a_own, a_d = f'{own} actor 0', 'D-a actor 0'
      dc = (1 - BC) * (c_vals[a_own] - c_vals[a_d])
      db = BC * (bc_vals[s][a_own] - bc_vals[s][a_d])
      se = float(np.sqrt(dc.var() / len(dc) + db.var() / len(db)))
      res['A'][s][cn] = {'total': {a: float(tot[a]) for a in anames}, 'critic_term': ct,
                         'bc_term': {a: float(BC * bc_vals[s][a].mean()) for a in anames},
                         'family_total': fam_tot, 'family_critic_term': fam_ct,
                         'own_minus_best_down_total': fam_tot[own] - best_down,
                         'own_minus_best_down_critic': fam_ct[own] - best_down_ct,
                         'se_actor0_pair': se}
      L.append(f'| {cn} | ' + ' | '.join(f'{fam_tot[f]:+.4f}' for f in fams)
               + f' | {fam_tot[own] - best_down:+.4f} (s.e. ~{se:.4f}) '
               f'| {fam_ct[own] - best_down_ct:+.4f} |')
    L.append('')

  # ---- B. attribution by group -----------------------------------------
  L += ['## B. Which rows pay for RIGHT: critic-term difference (own actor 0 minus D-a actor 0), '
        'full distribution, by (anchor cell, goal cell) group', '',
        'contribution = share of rows in the group x mean paired difference of 0.95 * (-f); '
        'positive = the critic scores the DOWN actor better on those rows; the sum over groups '
        'is the full-distribution difference.', '']
  (o_full, _, pos_full), _ = sets['full']
  sc, gc = s_cell[pos_full], g_cell[pos_full]
  gid = (sc[:, 0] * 10 + sc[:, 1]) * 100 + gc[:, 0] * 10 + gc[:, 1]
  for cn in ('R1: joint30k s1', 'R2: joint300k s0'):
    qp = critics[cn]
    own = OWN[cn]
    d = (1 - BC) * (batched(critic_term, (qp, actors[f'{own} actor 0']), o_full)
                    - batched(critic_term, (qp, actors['D-a actor 0']), o_full))
    ug, inv = np.unique(gid, return_inverse=True)
    contrib = np.bincount(inv, weights=d, minlength=len(ug)) / len(d)
    share = np.bincount(inv, minlength=len(ug)) / len(d)
    order = np.argsort(contrib)
    rows = []
    for idx in list(order[:8]) + list(order[-8:]):
      g = int(ug[idx])
      rows.append({'anchor': (g // 1000, (g // 100) % 10), 'goal': ((g % 100) // 10, g % 10),
                   'share': float(share[idx]), 'contribution': float(contrib[idx]),
                   'mean_diff': float(contrib[idx] / share[idx])})
    fork_sum = float(contrib[[(int(g) // 1000, (int(g) // 100) % 10) == (1, 3) for g in ug]].sum())
    res['B'][cn] = {'total': float(d.mean()), 'fork_rows_sum': fork_sum, 'extremes': rows}
    L += [f'### {cn}: own actor 0 (RIGHT) minus D-a actor 0 (DOWN); total {d.mean():+.4f}, '
          f'of which fork-anchor groups {fork_sum:+.4f}', '',
          '| anchor | goal | row share | contribution | mean diff |', '|---|---|---:|---:|---:|']
    for r in rows:
      L.append(f'| {r["anchor"]} | {r["goal"]} | {r["share"]:.4f} | {r["contribution"]:+.5f} '
               f'| {r["mean_diff"]:+.3f} |')
    L.append('')

  # ---- C. fork rows by goal cell ------------------------------------------
  L += ['## C. Fork anchors (buffer law) by goal cell: Qbar at width 0.75 for fixed locs, '
        'and the actors\' modes under those goals', '']
  (o_f, _, pos_f), _ = sets['fork (1,3), all goals']
  gcf = g_cell[pos_f]
  gkey = gcf[:, 0] * 10 + gcf[:, 1]
  ug, cnt = np.unique(gkey, return_counts=True)
  keep = [(int(g), int(c)) for g, c in zip(ug, cnt) if c >= 100]
  res['C'] = {}
  for cn, qp in critics.items():
    qb = {}
    for ln, loc in LOCS.items():
      loc_j = jnp.asarray(np.array(loc, np.float32))
      vals = []
      for k in range(0, len(o_f), 512):
        vals.append(np.asarray(qbar_at(qp, jnp.asarray(o_f[k:k + 512]), loc_j)))
      qb[ln] = np.concatenate(vals)
    L += [f'### {cn}', '', '| goal cell | rows | ' + ' | '.join(
        f'{ln} - RIGHT' for ln in LOCS if not ln.startswith('RIGHT')) + ' |',
          '|---|---:|' + '---:|' * (len(LOCS) - 1)]
    res['C'][cn] = {}
    for g, c in keep:
      m = gkey == g
      cell = (g // 10, g % 10)
      diffs = {ln: float((qb[ln][m] - qb['RIGHT (5,-0.1)'][m]).mean()) for ln in LOCS
               if not ln.startswith('RIGHT')}
      res['C'][cn][str(cell)] = {'rows': int(c), **diffs}
      L.append(f'| {cell} | {c} | ' + ' | '.join(f'{v:+.3f}' for v in diffs.values()) + ' |')
    L.append('')
  L += ['### Actor modes at the fork anchors by goal cell (mean tanh(loc), x / y)', '',
        '| goal cell | rows | ' + ' | '.join(f'{f} actor 0' for f in fams) + ' |',
        '|---|---:|' + '---:|' * len(fams)]
  md = {f: np.asarray(modes(actors[f'{f} actor 0'], jnp.asarray(o_f))[0]) for f in fams}
  res['C']['modes'] = {}
  for g, c in keep:
    m = gkey == g
    cell = (g // 10, g % 10)
    res['C']['modes'][str(cell)] = {f: md[f][m].mean(0).tolist() for f in fams}
    L.append(f'| {cell} | {c} | ' + ' | '.join(
        f'({md[f][m].mean(0)[0]:+.2f}, {md[f][m].mean(0)[1]:+.2f})' for f in fams) + ' |')
  L.append('')
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / 'REPORT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  (OUT / 'results.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
  print('\n'.join(L))
  return 0


if __name__ == '__main__':
  sys.exit(main())
