"""AntMaze V6 pilot: two frozen-model checks that decide the fix direction
(no retraining; the CF / O / start checkpoints, the dataset rows and the
recorded trajectories of diag_v6_pilot_trajectories).

  start_objective   At REAL start-region dataset rows (s, g, a_logged) -- the
                    actor's own training pairing -- compare the full actor
                    objective of the current CF policy with a policy whose mode
                    is shifted toward the critic-preferred candidate at that
                    row: the sampled mean critic score, the BC NLL of the
                    logged action, the total (0.95 / 0.05), along the straight
                    path loc(lambda) = (1 - lambda) loc + lambda loc*, lambda
                    in {0, .25, .5, .75, 1}.  Total worse at the target =
                    objective trade-off; total better but non-monotone =
                    a barrier the gradient cannot cross; better and monotone =
                    the update did not get there (budget / interference).
  midroute_update   At the CF continuation states before the stall (the 19
                    O-finishes-CF-not pairs): (a) the critic term's gradient
                    w.r.t. the action at the CF mode, projected on the
                    directions toward the O and start torques -- does the
                    critic push toward or away from the torques that keep
                    walking?; (b) the REAL parameter update: one actor step on
                    real actor-stream batches (the training seed's stream),
                    the induced change of the mode at those states projected on
                    (a_CF - a_start) (the drift direction) and on (a_O - a_CF),
                    split by the region of the batch rows that produced it
                    (start region / shortcut corridor / detour legs / other).

  python scripts/diag_v6_objective_checks.py start_objective
  python scripts/diag_v6_objective_checks.py midroute_update
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import exp_v6_mainline_pilot as MP  # noqa: E402
import diag_v6_pilot_trajectories as DT  # noqa: E402

OUT = DT.OUT
SEEDS = MP.SEEDS
STATE_DIM, OBS_W = MP.STATE_DIM, MP.OBS_W
BC = 0.05
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_MC = 64
SEED = 151_000_000


def _bundle_full(name):
  """Policy / critic handles plus the raw nets and params (for gradients)."""
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, OUT / '_cfg')
  MP.fill_dims(cfg)
  nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(DT.policies(name.split('@')[1] if '@' in name else None)[name])
  pp, qp = st.policy_params, st.q_params

  @jax.jit
  def f(o, a):
    phi, psi = nets.representation_network.apply(qp, o, a)
    return jnp.min(jnp.sum(phi * psi, axis=1), axis=1)

  @jax.jit
  def dist(o):
    d = nets.policy_network.apply(pp, o)
    return d.loc, d.scale

  @jax.jit
  def sample_from(loc, scale, key):
    eps = jax.random.normal(key, loc.shape)
    return jnp.tanh(loc + scale * eps)

  @jax.jit
  def grad_f_wrt_a(o, a):
    return jax.grad(lambda aa: jnp.sum(f(o, aa)))(a)
  return {'nets': nets, 'pp': pp, 'qp': qp, 'f': f, 'dist': dist, 'sample_from': sample_from, 'grad_f_a': grad_f_wrt_a, 'cfg': cfg}


def _dist_replace(d, loc, scale):
  return d._replace(loc=loc, scale=scale)          # crl.networks.TanhNormalParams (NamedTuple)


# --------------------------------------------------------- start_objective
def mode_start_objective(args):
  import jax
  import jax.numpy as jnp
  obs, act, lengths, _ = MP.load_dataset()
  n, L = obs.shape[:2]
  t = np.arange(L)[None, :]
  valid = t < (lengths[:, None] - 1)
  m = valid & (obs[:, :, 0] < 2.0) & (obs[:, :, 1] < 2.0) & (t <= 5)
  e, i = np.nonzero(m)
  rng = np.random.default_rng(SEED)
  sel = rng.choice(len(e), size=min(args.n_rows, len(e)), replace=False)
  O = obs[e[sel], i[sel], :OBS_W].astype(np.float32); A = act[e[sel], i[sel]].astype(np.float32)
  teach = DT._teacher_reset_torques()
  res = {'rows': int(len(sel)), 'lambdas': list(LAMBDAS), 'per_actor': {}}
  for s in SEEDS:
    for arm in ('CF', 'O'):
      name = f'{arm}_s{s}'
      b = _bundle_full(name)
      loc, scale = (np.asarray(v) for v in b['dist'](jnp.asarray(O)))
      mode = np.tanh(loc).astype(np.float32)
      # candidate set at each row: 12 own samples, the other seeds' modes (same arm), the O / start modes, the teacher reset torques
      key = jax.random.PRNGKey(SEED + s)
      keys = jax.random.split(key, 12)
      cands = {f'sample{j}': np.asarray(b['sample_from'](jnp.asarray(loc), jnp.asarray(scale), keys[j])) for j in range(12)}
      for q in SEEDS:
        if q != s:
          cands[f'mode_{arm}_s{q}'] = DT._policy_bundle(f'{arm}_s{q}')['mode_batch'](O)
      cands['start'] = DT._policy_bundle('start')['mode_batch'](O)
      for k, v in teach.items():
        cands[k] = np.repeat(v[None], len(O), 0)
      names = list(cands)
      F = np.stack([np.asarray(b['f'](jnp.asarray(O), jnp.asarray(cands[c]))) for c in names])      # [C, N]
      f_mode = np.asarray(b['f'](jnp.asarray(O), jnp.asarray(mode)))
      best_idx = F.argmax(0)
      target = np.stack([cands[names[best_idx[r]]][r] for r in range(len(O))]).astype(np.float32)
      gain = F.max(0) - f_mode
      fam = np.array([names[j].rstrip('0123456789') for j in best_idx])
      loc_star = np.arctanh(np.clip(target, -0.999, 0.999)).astype(np.float32)
      # the objective along the straight path in loc space, same scale
      curves = {'E_f': [], 'bc_nll': [], 'total': [], 'f_mode': [], 'E_f_widen': [], 'bc_nll_widen': [], 'total_widen': [], 'scale_widen': []}
      d0 = b['nets'].policy_network.apply(b['pp'], jnp.asarray(O))
      for lam in LAMBDAS:
        loc_l = (1 - lam) * loc + lam * loc_star
        # (i) the scale held at the current value; (ii) "widen as you move": per-dim sd = max(current, half the loc shift so far)
        scale_w = np.maximum(scale, 0.5 * np.abs(loc_l - loc)).astype(np.float32)
        for tag, sc in (('', scale), ('_widen', scale_w)):
          fs = []
          for j in range(N_MC):
            a_s = b['sample_from'](jnp.asarray(loc_l), jnp.asarray(sc), jax.random.fold_in(key, 1000 + j))
            fs.append(np.asarray(b['f'](jnp.asarray(O), a_s)))
          Ef = np.mean(fs, 0)
          d_l = _dist_replace(d0, jnp.asarray(loc_l), jnp.asarray(sc))
          nll = -np.asarray(b['nets'].log_prob(d_l, jnp.asarray(A)))
          curves['E_f' + tag].append(Ef); curves['bc_nll' + tag].append(nll); curves['total' + tag].append((1 - BC) * (-Ef) + BC * nll)
        curves['scale_widen'].append(scale_w.mean(1))
        curves['f_mode'].append(np.asarray(b['f'](jnp.asarray(O), jnp.tanh(jnp.asarray(loc_l)))))
      curves = {k: np.stack(v) for k, v in curves.items()}      # [lambda, N]
      tot = curves['total']
      better_at_target = tot[-1] < tot[0]
      monotone = np.all(np.diff(tot, axis=0) <= 1e-6, axis=0)
      barrier = (tot[1:-1].max(0) > tot[0] + 1e-6) & better_at_target
      strong = gain > 0.5
      summ = {'rows': int(len(O)), 'critic_gain_of_target_over_mode_mean': float(gain.mean()), 'rows_with_gain_gt_0.5': int(strong.sum()),
              'target_family_counts': {k: int((fam == k).sum()) for k in np.unique(fam)},
              'objective_at_lambda_mean': {k: [float(x) for x in v.mean(1)] for k, v in curves.items()},
              'objective_at_lambda_mean_strong_rows': {k: [float(x) for x in v[:, strong].mean(1)] for k, v in curves.items()} if strong.any() else None,
              'P_total_better_at_target': float(better_at_target.mean()), 'P_total_better_at_target_strong': float(better_at_target[strong].mean()) if strong.any() else None,
              'P_monotone_improvement_path': float((better_at_target & monotone).mean()), 'P_barrier_on_path': float(barrier.mean()),
              'P_Ef_better_at_target': float((curves['E_f'][-1] > curves['E_f'][0]).mean()), 'P_bc_worse_at_target': float((curves['bc_nll'][-1] > curves['bc_nll'][0]).mean()),
              'mean_delta_total_at_target': float((tot[-1] - tot[0]).mean()), 'mean_delta_Ef_at_target': float((curves['E_f'][-1] - curves['E_f'][0]).mean()),
              'mean_delta_bc_at_target': float((curves['bc_nll'][-1] - curves['bc_nll'][0]).mean()),
              'mean_delta_total_at_target_by_family': {k: float((tot[-1] - tot[0])[fam == k].mean()) for k in np.unique(fam)},
              'widen_path': {'P_total_better_at_target': float((curves['total_widen'][-1] < tot[0]).mean()),
                             'P_monotone_improvement': float(((curves['total_widen'][-1] < tot[0]) & np.all(np.diff(np.concatenate([tot[:1], curves['total_widen'][1:]]), axis=0) <= 1e-6, axis=0)).mean()),
                             'mean_delta_total_at_target': float((curves['total_widen'][-1] - tot[0]).mean()),
                             'mean_delta_Ef_at_target': float((curves['E_f_widen'][-1] - curves['E_f'][0]).mean()), 'mean_delta_bc_at_target': float((curves['bc_nll_widen'][-1] - curves['bc_nll'][0]).mean()),
                             'total_by_lambda_mean': [float(x) for x in np.concatenate([tot[:1], curves['total_widen'][1:]]).mean(1)], 'scale_by_lambda_mean': [float(x) for x in curves['scale_widen'].mean(1)],
                             'mean_delta_total_by_family': {k: float((curves['total_widen'][-1] - tot[0])[fam == k].mean()) for k in np.unique(fam)}},
              'mean_loc_shift_L2': float(np.linalg.norm(loc_star - loc, axis=1).mean()), 'scale_mean': float(scale.mean())}
      res['per_actor'][name] = summ
      print(name, json.dumps({k: v for k, v in summ.items() if k not in ('objective_at_lambda_mean', 'objective_at_lambda_mean_strong_rows')}), flush=True)
  MP.write_json(OUT / 'start_objective.json', res)


# --------------------------------------------------------- midroute_update
def _region_of(xy):
  x, y = xy[:, 0], xy[:, 1]
  reg = np.full(len(xy), 'other', dtype=object)
  reg[(x < 2) & (y < 2)] = 'start'
  reg[(x >= 2) & (np.abs(y) < 2.5)] = 'shortcut_corridor'
  reg[((x < 2.5) & (y >= 2)) | (y >= 5.5) | ((x >= 21.5) & (y >= 2))] = 'detour_legs'
  return reg


def mode_midroute_update(args):
  import jax
  import jax.numpy as jnp
  import optax
  from crl import losses as losses_mod
  P = MP.read_json(OUT / 'pairs.json')
  Z = np.load(OUT / 'pairs_traj.npz')
  res = {'per_seed': {}}
  for s in SEEDS:
    name = f'CF_s{s}'
    rows = [r for r in P['pairs'] if r['kind'] == 'O_finishes_CF_not' and r['reproduced'] and r['seed'] == s]
    if not rows:
      continue
    S = []
    for r in rows:
      cf_obs = Z[f's{s}_e{r["episode"]}_CF_s{s}']
      st = r['anomaly']['first_step'] if r['anomaly']['first_step'] is not None else len(cf_obs) - 1
      S.append(cf_obs[:max(2, st)][::5])
    S = np.concatenate(S).astype(np.float32)                        # the CF states before the stall
    b = _bundle_full(name)
    a_cf = DT._policy_bundle(name)['mode_batch'](S); a_o = DT._policy_bundle(f'O_s{s}')['mode_batch'](S); a_st = DT._policy_bundle('start')['mode_batch'](S)
    # (a) the critic term's action gradient at the CF mode, projected on the directions toward O / start
    g = np.asarray(b['grad_f_a'](jnp.asarray(S), jnp.asarray(a_cf)))
    def proj(g, d):
      dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
      return (g * dn).sum(1)
    pa = {'n_states': int(len(S)), 'grad_norm_mean': float(np.linalg.norm(g, axis=1).mean()),
          'proj_toward_O_mean': float(proj(g, a_o - a_cf).mean()), 'P_proj_toward_O_positive': float((proj(g, a_o - a_cf) > 0).mean()),
          'proj_toward_start_mean': float(proj(g, a_st - a_cf).mean()), 'P_proj_toward_start_positive': float((proj(g, a_st - a_cf) > 0).mean()),
          'proj_along_drift_mean (a_CF - a_start)': float(proj(g, a_cf - a_st).mean()), 'dist_CF_O': float(np.linalg.norm(a_o - a_cf, axis=1).mean()), 'dist_CF_start': float(np.linalg.norm(a_st - a_cf, axis=1).mean())}
    # reference: the same gradient at the 196 reset rows and at start-region dataset rows
    anchors, _ = MP.AnchorSet.load(MP.OUT / 'anchors.npz'); k0 = np.flatnonzero(anchors.t == 0)
    O0 = anchors.obs31[k0].astype(np.float32); a0 = DT._policy_bundle(name)['mode_batch'](O0)
    g0 = np.asarray(b['grad_f_a'](jnp.asarray(O0), jnp.asarray(a0)))
    pa['grad_norm_mean_at_reset_rows'] = float(np.linalg.norm(g0, axis=1).mean())
    # (b) the REAL parameter update: one actor step on real actor-stream batches, the induced mode change at S
    cfg = b['cfg']
    stream = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + s)
    nets, pp, qp = b['nets'], b['pp'], b['qp']
    gidx = np.asarray(cfg.goal_indices)

    def obs_to_goal(states):
      return states[:, jnp.asarray(gidx)]
    pol_opt = optax.adam(cfg.actor_learning_rate, eps=1e-7)
    q_opt = optax.adam(cfg.learning_rate, eps=1e-7)
    _, update_step = losses_mod.build_learner(nets, cfg, obs_to_goal, pol_opt, q_opt, separate_actor_batch=True)
    from crl.losses import TrainingState

    def actor_loss_only(params, tr, key):
      # the actor loss on one actor batch (critic fixed), as in crl.losses.actor_loss with alpha 0 and shared rows
      d = nets.policy_network.apply(params, tr.observation)
      a = nets.sample(d, key)
      phi, psi = nets.representation_network.apply(qp, tr.observation, a)
      q_term = -jnp.min(jnp.sum(phi * psi, axis=1), axis=1)
      nll = -nets.log_prob(d, tr.action)
      return jnp.mean(BC * nll + (1 - BC) * q_term)
    grad_fn = jax.jit(jax.grad(actor_loss_only))
    mode_fn = jax.jit(lambda params, o: jnp.tanh(nets.policy_network.apply(params, o).loc))
    base_mode = np.asarray(mode_fn(pp, jnp.asarray(S)))
    d_drift = a_cf - a_st; d_o = a_o - a_cf
    eta = cfg.actor_learning_rate
    key = jax.random.PRNGKey(SEED + 77 + s)
    acc = {}
    nb = args.n_batches
    for it in range(nb):
      tr = stream.sample(cfg.batch_size)
      key, k1 = jax.random.split(key)
      from crl.losses import Transition
      trj = Transition(*[jnp.asarray(getattr(tr, fld)) for fld in Transition._fields])
      reg = _region_of(np.asarray(tr.observation)[:, :2])
      for label, mask in (('all', np.ones(len(reg), bool)), *[(r_, reg == r_) for r_ in ('start', 'shortcut_corridor', 'detour_legs', 'other')]):
        if mask.sum() == 0:
          continue
        sub = trj._replace(**{fld: getattr(trj, fld)[jnp.asarray(mask)] for fld in trj._fields})
        gr = grad_fn(pp, sub, k1)
        # a plain SGD probe step of size eta scaled by the batch share, so that region contributions add up to the full-batch step
        share = mask.mean()
        new = jax.tree_util.tree_map(lambda p_, g_: p_ - eta * share * g_, pp, gr)
        dm = np.asarray(mode_fn(new, jnp.asarray(S))) - base_mode
        d = acc.setdefault(label, {'n': 0, 'share': [], 'dm_norm': [], 'proj_drift': [], 'proj_toward_O': []})
        d['n'] += 1; d['share'].append(float(share)); d['dm_norm'].append(float(np.linalg.norm(dm, axis=1).mean()))
        d['proj_drift'].append(float(proj(dm, d_drift).mean())); d['proj_toward_O'].append(float(proj(dm, d_o).mean()))
    pb = {k: {kk: (float(np.mean(vv)) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in acc.items()}
    res['per_seed'][name] = {'critic_action_gradient_at_prestall_states': pa, 'real_actor_step_induced_mode_change_at_prestall_states': pb,
                             'note': 'SGD probe step of size actor lr on one real actor batch (Adam preconditioning not applied); projections are mean cosine-scaled components along (a_CF - a_start) (the drift) and (a_O - a_CF)'}
    print(name, json.dumps(pa), flush=True); print(name, json.dumps(pb), flush=True)
  MP.write_json(OUT / 'midroute_update.json', res)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('mode', choices=('start_objective', 'midroute_update'))
  ap.add_argument('--n-rows', type=int, default=1024)
  ap.add_argument('--n-batches', type=int, default=16)
  args = ap.parse_args(argv)
  {'start_objective': mode_start_objective, 'midroute_update': mode_midroute_update}[args.mode](args)
  return 0


if __name__ == '__main__':
  sys.exit(main())
