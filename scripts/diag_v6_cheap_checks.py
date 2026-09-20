"""AntMaze V6 -- two cheap checks on existing data (user, 2026-09-21; no new rollouts, no training).

(a) goal_mass   Where does the critic's positive-goal law (geometric, gamma 0.999, truncated at the branch end -- exactly
                CriticStream.draw) put its mass along the EXISTING futures: at the task goal (within the 0.5 reach radius / the
                goal-area region), at stall rows (|delta state| < 1e-3), at death positions (the last 5 rows of a death branch),
                at the terminal 10 rows, in the hazard zones / far legs / elsewhere (mid-route)?  Per future source: the recorded
                continuation (arm O), the sealed CF branches (start-agent continuation), and -- at the start-region anchors --
                the stage-1 query branches (q0 logged / q1 mode / q2-5 samples / extended) per lineage.  The question: does
                "reaches more" mean "more task-goal positives"?  Anchor-weighted; by outcome too.
(b) bc_mode     On the REAL actor batches of each lineage (the actor stream fast-forwarded to the 30k checkpoint, then N
                batches), the BC term under the recipe's 'clip' log-prob (this port: actions clipped to 1 - 1e-6, density at
                atanh) vs dm-acme 0.4.0's 'acme' boundary-band rule (the original contrastive_rl actor; the successful
                PointMaze run used 'acme'): the BC NLL values, the share of boundary action components, the weighted BC
                gradients (0.05 x grad) -- norms, cosine between the two, and each against the weighted critic-term gradient.
                Small difference -> not worth a training; clearly different -> a separate numerical-implementation comparison.

  python scripts/diag_v6_cheap_checks.py goal_mass
  python scripts/diag_v6_cheap_checks.py bc_mode --batches 20
Outputs under outputs/antmaze_branch_replay_p050/exp_mainline_pilot/cheap_checks/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP  # noqa: E402
import exp_v6_clip_round as CR  # noqa: E402
import exp_v6_query_coverage as QC  # noqa: E402
from exp_v6_learned_ett import REGIONS, region_of  # noqa: E402

OUT = MP.OUT / 'cheap_checks'
GAMMA = MP.GAMMA
REACH_R = 0.5
STAT_TOL = 1e-3
DEATH_TAIL = 5
TERM_TAIL = 10
GOAL_AREA = REGIONS.index('goal_area')


# ---------------------------------------------------------------- (a) goal mass
def branch_masses(obs_rows, off, L, outcome, goal_xy_of_branch):
  """Per-branch masses of the truncated geometric law over rows m = 1..L-1.  obs_rows [R, >=31]; goal_xy_of_branch [K, 2]."""
  K = len(L)
  row_of = np.repeat(np.arange(K), L)
  m = np.arange(len(row_of)) - np.repeat(off, L)
  after = m >= 1
  w = np.where(after, GAMMA ** m, 0.0)
  z = np.bincount(row_of, weights=w, minlength=K); w = w / np.maximum(z[row_of], 1e-12)
  xy = obs_rows[:, :2].astype(np.float64)
  g = goal_xy_of_branch[row_of]
  reach = np.linalg.norm(xy - g, axis=1) <= REACH_R
  reg = region_of(xy)
  goal_area = reg == GOAL_AREA
  d = np.zeros(len(row_of), bool)
  d[1:] = np.abs(obs_rows[1:, :MP.STATE_DIM] - obs_rows[:-1, :MP.STATE_DIM]).max(axis=1) < STAT_TOL
  stall = d & after
  last = np.repeat(off + L - 1, L)
  death_pos = after & (np.repeat(outcome == 'death', L)) & (np.arange(len(row_of)) >= last - DEATH_TAIL + 1)
  terminal = after & (np.arange(len(row_of)) >= last - TERM_TAIL + 1)
  zone = np.isin(reg, CR.ZONES); far = np.isin(reg, CR.FAR)
  mid = after & ~reach & ~stall & ~death_pos
  cats = {'reach_0.5': reach & after, 'goal_area': goal_area & after, 'stall': stall, 'death_pos_last5': death_pos, 'terminal_last10': terminal,
          'zones': zone & after, 'far_legs': far & after, 'mid_route': mid}
  return {k: np.bincount(row_of, weights=w * v, minlength=K) for k, v in cats.items()}


def wm(x, w):
  return float((w * x).sum() / w.sum())


def summarise(M, W, outcome, L, label):
  out = {'label': label, 'n': int(len(L)), 'weight': float(W.sum()), 'mean_length': wm(L.astype(float), W),
         'outcome': {k: wm((outcome == k).astype(float), W) for k in ('success', 'death', 'timeout')},
         'mass': {k: wm(v, W) for k, v in M.items()}, 'mass_by_outcome': {}}
  for oc in ('success', 'death', 'timeout'):
    msk = outcome == oc
    if msk.sum() >= 20:
      out['mass_by_outcome'][oc] = {'n': int(msk.sum()), 'mean_length': wm(L[msk].astype(float), W[msk]), **{k: wm(v[msk], W[msk]) for k, v in M.items()}}
  return out


def mode_goal_mass(args):
  OUT.mkdir(parents=True, exist_ok=True)
  anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
  ids = QC.start_anchor_ids(anchors)
  W = anchors.weight
  res = {'law': f'geometric gamma {GAMMA} truncated at the branch end (CriticStream.draw); masses anchor-weighted', 'reach_radius': REACH_R, 'stat_tol': STAT_TOL, 'sources': []}
  t0 = time.time()
  # recorded continuation (arm O)
  obs, act, lengths, _ = MP.load_dataset()
  Lr = (anchors.ep_length - anchors.t).astype(np.int64)
  rows_e = np.repeat(anchors.episode, Lr); rows_t = np.repeat(anchors.t, Lr) + (np.arange(Lr.sum()) - np.repeat(np.concatenate([[0], np.cumsum(Lr)[:-1]]), Lr))
  r_obs = obs[rows_e, rows_t, :MP.OBS_W]
  off_r = np.concatenate([[0], np.cumsum(Lr)[:-1]])
  # the recorded episodes end on their reach frame (success) or at death / the horizon: label from the last row
  last_xy = obs[np.arange(len(lengths)), lengths - 1, :2]; goal_l = obs[np.arange(len(lengths)), 0, MP.STATE_DIM:MP.OBS_W]
  ep_out = np.where(np.linalg.norm(last_xy - goal_l, axis=1) <= REACH_R, 'success', np.where(lengths >= MP.HORIZON, 'timeout', 'death'))
  oc_r = ep_out[anchors.episode]
  M = branch_masses(r_obs, off_r, Lr, oc_r, anchors.goal_xy.astype(np.float64))
  res['sources'].append(summarise(M, W, oc_r, Lr, 'O recorded | all anchors'))
  res['sources'].append(summarise({k: v[ids] for k, v in M.items()}, W[ids], oc_r[ids], Lr[ids], 'O recorded | start anchors'))
  del r_obs
  print(f'recorded done {time.time() - t0:.0f} s', flush=True)
  # sealed CF branches
  with np.load(MP.OUT / 'branches_cf.npz', allow_pickle=False) as d:
    b_obs, off_b, Lb, oc_b = d['obs_rows'], d['offset'].astype(np.int64), d['length'].astype(np.int64), d['outcome'].astype(str)
  M = branch_masses(b_obs, off_b, Lb, oc_b, anchors.goal_xy.astype(np.float64))
  res['sources'].append(summarise(M, W, oc_b, Lb, 'CF sealed (start-agent continuation) | all anchors'))
  res['sources'].append(summarise({k: v[ids] for k, v in M.items()}, W[ids], oc_b[ids], Lb[ids], 'CF sealed | start anchors'))
  del b_obs
  print(f'sealed done {time.time() - t0:.0f} s', flush=True)
  # stage-1 query branches per lineage
  for s in MP.SEEDS:
    p = QC.branch_path(s)
    if not p.exists():
      continue
    with np.load(p, allow_pickle=False) as d:
      q_obs, off_q, Lq, oc_q = d['obs_rows'], d['offset'].astype(np.int64), d['length'].astype(np.int64), d['outcome'].astype(str)
      q_anchor, q_query = d['anchor_id'].astype(np.int64), d['query_id'].astype(np.int64)
    M = branch_masses(q_obs, off_q, Lq, oc_q, anchors.goal_xy[q_anchor].astype(np.float64))
    Wq = W[q_anchor]
    for name, qset in QC.SETS.items():
      sel = np.isin(q_query, qset)
      res['sources'].append(summarise({k: v[sel] for k, v in M.items()}, Wq[sel], oc_q[sel], Lq[sel], f'lineage {s} queries: {name} | start anchors'))
    del q_obs
    print(f'lineage {s} done {time.time() - t0:.0f} s', flush=True)
  MP.write_json(OUT / 'goal_mass.json', res)
  L = ['# (a) Where the critic\'s positive-goal law puts its mass along the existing futures', '',
       f'Law: geometric gamma {GAMMA} truncated at the branch end (the critic stream\'s own draw); anchor-weighted means of per-branch masses.  '
       f'reach = within {REACH_R} of the task goal; goal_area = the maze region; stall = |delta state| < {STAT_TOL}; death_pos = last {DEATH_TAIL} rows of a death branch; terminal = last {TERM_TAIL} rows; mid_route = neither reach nor stall nor death_pos.  `goal_mass.json`.', '',
       '| source | n | weight | mean length | success / death / timeout | reach 0.5 | goal_area | stall | death_pos | terminal 10 | zones | far legs | mid_route |', '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for S in res['sources']:
    o, m = S['outcome'], S['mass']
    L.append(f'| {S["label"]} | {S["n"]} | {S["weight"]:.4f} | {S["mean_length"]:.0f} | {o["success"]:.3f} / {o["death"]:.3f} / {o["timeout"]:.3f} | {m["reach_0.5"]:.4f} | {m["goal_area"]:.4f} | {m["stall"]:.3f} | {m["death_pos_last5"]:.3f} | {m["terminal_last10"]:.3f} | {m["zones"]:.3f} | {m["far_legs"]:.3f} | {m["mid_route"]:.3f} |')
  L += ['', '## By outcome (per source: mean length; reach / goal_area / stall / death_pos / terminal masses)', '', '| source | outcome | n | mean length | reach 0.5 | goal_area | stall | death_pos | terminal 10 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
  for S in res['sources']:
    for oc, b in S['mass_by_outcome'].items():
      L.append(f'| {S["label"]} | {oc} | {b["n"]} | {b["mean_length"]:.0f} | {b["reach_0.5"]:.4f} | {b["goal_area"]:.4f} | {b["stall"]:.3f} | {b["death_pos_last5"]:.3f} | {b["terminal_last10"]:.3f} |')
  (OUT / 'GOAL_MASS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


# ---------------------------------------------------------------- (b) BC log-prob mode
def mode_bc_mode(args):
  import jax
  import jax.numpy as jnp
  import optax
  from crl import checkpoint, losses as losses_mod
  from crl import networks as NW
  import diag_v6_training_replay as DR
  OUT.mkdir(parents=True, exist_ok=True)
  res = {'batches': int(args.batches), 'bc_coef': None, 'lineages': {}}
  for s in MP.SEEDS:
    cfg = MP.recipe_config(s, OUT / '_cfg'); cfg.batch_size = MP.BATCH; MP.fill_dims(cfg)
    bc = float(cfg.bc_coef); res['bc_coef'] = bc
    nets = MP.make_nets(cfg)                                     # the recipe's networks (log_prob_mode 'clip')
    assert (getattr(cfg, 'log_prob_mode', 'clip') or 'clip') == 'clip'
    src = CR.lineage_ckpt(s)
    _, st = checkpoint.load_checkpoint(src); pp, qp = st.policy_params, st.q_params
    pol_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7); q_opt = MP.critic_optimizer(cfg, float(MP.VARIANTS[CR.RECIPE]['critic_clip']))
    anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz')
    futures = MP.BranchFutures(anchors, MP.OUT / 'branches_cf.npz')
    critic_stream = MP.CriticStream(anchors, futures, cfg.batch_size, cfg.discount, MP.CRITIC_STREAM_SEED0 + s)
    actor_stream = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + s)
    DR.fast_forward(critic_stream, actor_stream, MP.UPDATES)
    key = jax.random.PRNGKey(1_000 + s)

    def nll_clip(p, Ob, A):
      return -NW.tanh_normal_log_prob(nets.policy_network.apply(p, Ob), A)

    def nll_acme(p, Ob, A):
      return -NW.tanh_normal_log_prob_acme(nets.policy_network.apply(p, Ob), A)

    def q_term(p, Ob, k):
      d = nets.policy_network.apply(p, Ob); a = nets.sample(d, k)
      q = nets.q_network.apply(qp, Ob, a); q = jnp.min(q, axis=-1) if q.ndim == 3 else q
      return (1.0 - bc) * jnp.mean(-jnp.diag(q))
    g_clip = jax.jit(jax.grad(lambda p, Ob, A: bc * jnp.mean(nll_clip(p, Ob, A))))
    g_acme = jax.jit(jax.grad(lambda p, Ob, A: bc * jnp.mean(nll_acme(p, Ob, A))))
    g_q = jax.jit(jax.grad(q_term))
    f_clip = jax.jit(nll_clip); f_acme = jax.jit(nll_acme)

    def flat(t):
      return np.asarray(jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(t)]), np.float64)
    rows = []
    for i in range(int(args.batches)):
      b = actor_stream.sample(cfg.batch_size); Ob = jnp.asarray(b.observation); A = jnp.asarray(b.action)
      key, k1 = jax.random.split(key)
      nc = np.asarray(f_clip(pp, Ob, A)); na = np.asarray(f_acme(pp, Ob, A))
      gc, ga, gq = flat(g_clip(pp, Ob, A)), flat(g_acme(pp, Ob, A)), flat(g_q(pp, Ob, k1))
      bound = np.mean(np.abs(np.asarray(b.action)) >= 0.999)
      rows.append({'nll_clip': float(nc.mean()), 'nll_acme': float(na.mean()), 'nll_diff_mean': float((nc - na).mean()), 'nll_diff_p99': float(np.percentile(np.abs(nc - na), 99)),
                   'share_boundary_components': float(bound), 'share_rows_with_boundary': float(np.mean((np.abs(np.asarray(b.action)) >= 0.999).any(axis=1))),
                   'g_bc_clip_norm': float(np.linalg.norm(gc)), 'g_bc_acme_norm': float(np.linalg.norm(ga)), 'cos_clip_acme': float(gc @ ga / (np.linalg.norm(gc) * np.linalg.norm(ga) + 1e-12)),
                   'g_q_norm': float(np.linalg.norm(gq)), 'cos_q_clip': float(gq @ gc / (np.linalg.norm(gq) * np.linalg.norm(gc) + 1e-12)), 'cos_q_acme': float(gq @ ga / (np.linalg.norm(gq) * np.linalg.norm(ga) + 1e-12)),
                   'diff_norm_over_clip_norm': float(np.linalg.norm(gc - ga) / (np.linalg.norm(gc) + 1e-12))})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res['lineages'][f'seed_{s}'] = {'ckpt': str(src), 'mean': mean, 'rows': rows}
    print(f'seed {s}: ' + json.dumps({k: round(v, 4) for k, v in mean.items()}), flush=True)
  MP.write_json(OUT / 'bc_mode.json', res)
  L = ['# (b) The BC term under the recipe\'s \'clip\' log-prob vs dm-acme 0.4.0\'s \'acme\' boundary rule, on the real actor batches', '',
       f'Each lineage\'s clipped CF final; the actor stream fast-forwarded to the 30k checkpoint, then {args.batches} real batches of {MP.BATCH} rows (means over batches).  '
       f'BC coefficient {res["bc_coef"]}; gradients are the WEIGHTED terms (bc x grad BC, (1 - bc) x grad critic term, the critic term with a fresh sampled action).  `bc_mode.json`.', '',
       '| lineage | NLL clip | NLL acme | mean diff | p99 |diff| | boundary components | rows with a boundary component | ||bc grad|| clip / acme | cos(clip, acme) | ||diff|| / ||clip|| | ||critic-term grad|| | cos(critic, bc clip) / (critic, bc acme) |',
       '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|']
  for k, v in res['lineages'].items():
    m = v['mean']
    L.append(f'| {k} | {m["nll_clip"]:.3f} | {m["nll_acme"]:.3f} | {m["nll_diff_mean"]:+.3f} | {m["nll_diff_p99"]:.3f} | {m["share_boundary_components"]:.3f} | {m["share_rows_with_boundary"]:.3f} | {m["g_bc_clip_norm"]:.4f} / {m["g_bc_acme_norm"]:.4f} | {m["cos_clip_acme"]:.4f} | {m["diff_norm_over_clip_norm"]:.4f} | {m["g_q_norm"]:.4f} | {m["cos_q_clip"]:+.4f} / {m["cos_q_acme"]:+.4f} |')
  (OUT / 'BC_MODE.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
  print('\n'.join(L), flush=True)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('mode', choices=['goal_mass', 'bc_mode'])
  ap.add_argument('--batches', type=int, default=20)
  args = ap.parse_args(argv)
  {'goal_mass': mode_goal_mass, 'bc_mode': mode_bc_mode}[args.mode](args)


if __name__ == '__main__':
  main()
