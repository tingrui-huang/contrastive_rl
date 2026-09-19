"""The actor's objective at the rows it actually trains on (frozen models; no rollouts, no training).

diag_traj section 9b evaluated the actor objective at start-region dataset
rows paired with the episode's TASK goal.  The actor stream pairs every row
with a RELABELED goal -- a future state of the same logged episode drawn by
the geometric law (gamma 0.999, truncated at the episode end) -- so 9b did
not check the distribution the actor faces.  This script draws real actor
batches from the training seed's own ActorStream (same buffer, same law),
keeps the rows whose state lies in the start region (where the route is
decided) and, for the checkpoint's own actor / critic pair, reports under
the REAL goal pairing and, on the same rows, under the task goal:

  * the critic term: E_{a ~ pi}[f(s, a, g)] (N_MC samples, the actual term),
    f at the mode, f at the logged torque, f at the 4 teacher detour reset
    torques (the candidates of diag_traj section 5);
  * the "detour signal" per row: max_j f(s, a_detour_j, g) - E_pi[f] and its
    sign, by the goal's location (start / corridor before the first hazard /
    hazard corridor / after the second hazard / goal area / detour legs) and
    by the goal offset m = j - i;
  * the full objective along the straight loc path from the current policy
    toward the best detour torque (lambda 0..1; scale held and widened as in
    9b): 0.95 * (-E_pi[f]) + 0.05 * NLL(a_logged); P(total better at the
    target) and the mean delta -- real goals vs task goal, same rows;
  * the critic term's action gradient at the mode projected toward the best
    detour torque (local signal), real goals vs task goal.

  python scripts/diag_v6_actor_objective_stream.py --seeds 0 1 2 --n-batches 48 --out <json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import exp_v6_mainline_pilot as MP  # noqa: E402
import diag_v6_pilot_trajectories as DT  # noqa: E402

BC = 0.05
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_MC = 64
SEED = 161_000_000
STATE_DIM, OBS_W = MP.STATE_DIM, MP.OBS_W


def region_of(xy):
  x, y = xy[:, 0], xy[:, 1]
  reg = np.full(len(xy), 'other', dtype=object)
  reg[(x < 2) & (y < 2)] = 'start'
  reg[(x >= 2) & (np.abs(y) < 2.5)] = 'shortcut_corridor'
  reg[((x < 2.5) & (y >= 2)) | (y >= 5.5) | ((x >= 21.5) & (y >= 2))] = 'detour_legs'
  return reg


def goal_region_of(xy):
  x, y = xy[:, 0], xy[:, 1]
  reg = np.full(len(xy), 'detour_legs', dtype=object)
  on_row = np.abs(y) < 2.5
  reg[on_row & (x < 2)] = 'start'
  reg[on_row & (x >= 2) & (x < 6.6)] = 'corridor_before_hazard_1'
  reg[on_row & (x >= 6.6) & (x <= 17.4)] = 'hazard_corridor'
  reg[on_row & (x > 17.4) & (x < 22)] = 'corridor_after_hazard_2'
  reg[on_row & (x >= 22)] = 'goal_area'
  return reg


def bundle(ckpt):
  import jax
  import jax.numpy as jnp
  from crl import checkpoint
  cfg = MP.recipe_config(0, MP.OUT / 'diag_traj' / '_cfg')
  MP.fill_dims(cfg)
  nets = MP.make_nets(cfg)
  _, st = checkpoint.load_checkpoint(ckpt)
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
    return jnp.tanh(loc + scale * jax.random.normal(key, loc.shape))

  @jax.jit
  def grad_f_a(o, a):
    return jax.grad(lambda aa: jnp.sum(f(o, aa)))(a)

  def nll(loc, scale, a):
    d0 = nets.policy_network.apply(pp, jnp.zeros((loc.shape[0], OBS_W), jnp.float32))
    return -np.asarray(nets.log_prob(d0._replace(loc=jnp.asarray(loc), scale=jnp.asarray(scale)), jnp.asarray(a)))
  return {'cfg': cfg, 'nets': nets, 'f': f, 'dist': dist, 'sample_from': sample_from, 'grad_f_a': grad_f_a, 'nll': nll}


def draw_rows(cfg, seed, n_batches, route):
  """Real actor-stream rows (state, relabeled goal, logged action) with their indices, as ActorStream / TrajectoryBuffer.sample builds them."""
  from crl.replay import obs_to_goal
  stream = MP.ActorStream(cfg, MP.ACTOR_STREAM_SEED0 + seed)
  buf = stream.buffer
  rows = {'state': [], 'goal': [], 'task_goal': [], 'action': [], 'traj': [], 'i': [], 'j': []}
  for _ in range(n_batches):
    traj, i, j = buf.sampled_indices(cfg.batch_size)
    st = buf._obs[traj, i, :STATE_DIM].astype(np.float32)
    g = obs_to_goal(buf._obs[traj, j, :STATE_DIM].astype(np.float32), buf._start_index, buf._end_index, buf._goal_indices)
    rows['state'].append(st); rows['goal'].append(np.asarray(g, np.float32)); rows['task_goal'].append(buf._obs[traj, i, STATE_DIM:OBS_W].astype(np.float32))
    rows['action'].append(buf._act[traj, i].astype(np.float32)); rows['traj'].append(traj); rows['i'].append(i); rows['j'].append(j)
  R = {k: np.concatenate(v) for k, v in rows.items()}
  R['route'] = route[R['traj']]
  R['region'] = region_of(R['state'][:, :2])
  R['goal_region'] = goal_region_of(R['goal'])
  R['m'] = R['j'] - R['i']
  return R


def analyse(b, S, G, A, teach, key):
  """Per-row quantities for states S, goals G, logged actions A under one goal pairing."""
  import jax
  import jax.numpy as jnp
  O = np.concatenate([S, G], axis=1).astype(np.float32)
  loc, scale = (np.asarray(v) for v in b['dist'](jnp.asarray(O)))
  mode = np.tanh(loc).astype(np.float32)
  f_mode = np.asarray(b['f'](jnp.asarray(O), jnp.asarray(mode)))
  f_logged = np.asarray(b['f'](jnp.asarray(O), jnp.asarray(A)))
  Ef = np.mean([np.asarray(b['f'](jnp.asarray(O), b['sample_from'](jnp.asarray(loc), jnp.asarray(scale), jax.random.fold_in(key, j)))) for j in range(N_MC)], axis=0)
  names = [k for k in teach if k.startswith('teacher_detour') or k.startswith('user_detour')]
  F = np.stack([np.asarray(b['f'](jnp.asarray(O), jnp.asarray(np.repeat(teach[k][None], len(O), 0)))) for k in names])   # [4, N]
  best = F.argmax(0); f_best = F.max(0)
  target = np.stack([teach[names[best[r]]] for r in range(len(O))]).astype(np.float32)
  g = np.asarray(b['grad_f_a'](jnp.asarray(O), jnp.asarray(mode)))
  d = target - mode
  proj = (g * d).sum(1) / (np.linalg.norm(d, axis=1) + 1e-8)
  # the full objective along the loc path toward the best detour torque
  loc_star = np.arctanh(np.clip(target, -0.999, 0.999)).astype(np.float32)
  nll0 = b['nll'](loc, scale, A)
  tot = {'held': [], 'widen': []}
  for lam in LAMBDAS:
    loc_l = (1 - lam) * loc + lam * loc_star
    for tag, sc in (('held', scale), ('widen', np.maximum(scale, 0.5 * np.abs(loc_l - loc)).astype(np.float32))):
      Efl = np.mean([np.asarray(b['f'](jnp.asarray(O), b['sample_from'](jnp.asarray(loc_l), jnp.asarray(sc), jax.random.fold_in(key, 1000 + j)))) for j in range(N_MC)], axis=0)
      tot[tag].append((1 - BC) * (-Efl) + BC * b['nll'](loc_l, sc, A))
  tot = {k: np.stack(v) for k, v in tot.items()}     # [lambda, N]
  return {'f_mode': f_mode, 'f_logged': f_logged, 'E_f': Ef, 'f_best_detour': f_best, 'signal': f_best - Ef, 'signal_vs_mode': f_best - f_mode,
          'best_name': np.array([names[i] for i in best]),
          'grad_proj_toward_detour': proj, 'grad_norm': np.linalg.norm(g, axis=1), 'nll_logged': nll0,
          'delta_total_held': tot['held'][-1] - tot['held'][0], 'delta_total_widen': tot['widen'][-1] - tot['held'][0],
          'total_held_path': tot['held'], 'total_widen_path': tot['widen'], 'scale_mean': scale.mean(1)}


def summarise(Q, mask):
  m = mask
  if m.sum() == 0:
    return None
  return {'n': int(m.sum()), 'E_f': float(Q['E_f'][m].mean()), 'f_mode': float(Q['f_mode'][m].mean()), 'f_logged': float(Q['f_logged'][m].mean()),
          'f_best_detour': float(Q['f_best_detour'][m].mean()),
          'signal_mean (best detour - E_f)': float(Q['signal'][m].mean()), 'P_signal_positive': float((Q['signal'][m] > 0).mean()),
          'P_best_detour_above_logged': float((Q['f_best_detour'][m] > Q['f_logged'][m]).mean()),
          'grad_proj_toward_detour_mean': float(Q['grad_proj_toward_detour'][m].mean()), 'P_grad_proj_positive': float((Q['grad_proj_toward_detour'][m] > 0).mean()),
          'delta_total_at_target_held_mean': float(Q['delta_total_held'][m].mean()), 'P_total_better_held': float((Q['delta_total_held'][m] < 0).mean()),
          'delta_total_at_target_widen_mean': float(Q['delta_total_widen'][m].mean()), 'P_total_better_widen': float((Q['delta_total_widen'][m] < 0).mean()),
          'total_path_held_mean': [float(x) for x in Q['total_held_path'][:, m].mean(1)], 'total_path_widen_mean': [float(x) for x in Q['total_widen_path'][:, m].mean(1)],
          'nll_logged_mean': float(Q['nll_logged'][m].mean()),
          'best_candidate_share': {k: float((Q['best_name'][m] == k).mean()) for k in np.unique(Q['best_name'][m])}}


def main():
  import jax
  ap = argparse.ArgumentParser()
  ap.add_argument('--seeds', type=int, nargs='+', default=list(MP.SEEDS))
  ap.add_argument('--arms', nargs='+', default=['CF', 'O'])
  ap.add_argument('--n-batches', type=int, default=48)
  ap.add_argument('--max-rows', type=int, default=6000, help='cap on the start-region rows analysed per actor (the path analysis is the costly part)')
  ap.add_argument('--out', required=True)
  ap.add_argument('--variant', default=None, help='evaluate the checkpoints of this pre-registered variant (e.g. critic_clip0.1) instead of the base pilot')
  ap.add_argument('--candidates', default=None, help='joined.json of the user\'s frozen-model diagnostic (2026-09-19): adds, per seed, the detour candidates the '
                                                     'critic scored above the own mode (user_detour<j>) to the detour torque set')
  ap.add_argument('--max-cands', type=int, default=8)
  args = ap.parse_args()
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  teach0 = DT._teacher_reset_torques()
  user_cands = {}
  if args.candidates:
    J = json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    for c in J:
      sd = int(c['seed']); names_ = c['candidate_names']; sc_ = c['scores']; im = names_.index('mode')
      for i, nm in enumerate(names_):
        oc = c['outcomes'].get(nm, {})
        if nm != 'mode' and oc.get('route') == 'detour' and sc_[i] > sc_[im]:
          user_cands.setdefault(sd, []).append({'episode': c['episode'], 'candidate': nm, 'margin': float(sc_[i] - sc_[im]), 'action': [float(x) for x in c['actions'][i]],
                                                'success': bool(oc.get('success')), 'saved_route': c.get('saved_route')})
    for sd in user_cands:
      user_cands[sd] = sorted(user_cands[sd], key=lambda r: -r['margin'])[:args.max_cands]
  res = {'n_mc': N_MC, 'lambdas': list(LAMBDAS), 'bc': BC, 'variant': args.variant, 'per_actor': {},
         'user_candidates': {str(k): [{kk: vv for kk, vv in r.items() if kk != 'action'} for r in v] for k, v in user_cands.items()},
         'candidate_note': ('ALL candidates here are TRANSPLANTS: torques evaluated at training rows other than the state they came from, where their effect is '
                            'unverified (pose and velocity differ).  teacher_detour<j>: logged t = 0 torques of detour episodes; user_detour<j>: the user\'s '
                            'candidates that the seed\'s own critic scored above its mode at ONE evaluation reset state and whose realised continuation entered '
                            'the far route THERE (diag_v6_candidate_at_origin.py checks them at that state).  Primary reading = held scale; widened scale listed '
                            'separately.  P(total better) high = the objective prefers the moved distribution at these inputs locally (shared parameters: not a '
                            'proof of an optimisation failure); low = the tested loc path is not supported (other directions not excluded).')}
  for s in args.seeds:
    cfg = MP.recipe_config(s, MP.OUT / 'diag_traj' / '_cfg')
    cfg.batch_size = MP.BATCH
    MP.fill_dims(cfg)
    R = draw_rows(cfg, s, args.n_batches, route)
    n_all = len(R['state'])
    comp = {'rows_drawn': int(n_all), 'region_share': {k: float((R['region'] == k).mean()) for k in np.unique(R['region'])},
            'route_share': {k: float((R['route'] == k).mean()) for k in np.unique(R['route'])},
            'goal_region_share_all_rows': {k: float((R['goal_region'] == k).mean()) for k in np.unique(R['goal_region'])}}
    start = np.flatnonzero(R['region'] == 'start')
    rng = np.random.default_rng(SEED + s)
    if len(start) > args.max_rows:
      start = np.sort(rng.choice(start, size=args.max_rows, replace=False))
    S, G, TG, A = R['state'][start], R['goal'][start], R['task_goal'][start], R['action'][start]
    greg, mm, rt = R['goal_region'][start], R['m'][start], R['route'][start]
    comp['start_rows_analysed'] = int(len(start))
    comp['start_rows_goal_region_share'] = {k: float((greg == k).mean()) for k in np.unique(greg)}
    comp['start_rows_goal_offset_quantiles'] = {q: float(np.quantile(mm, float(q))) for q in ('0.1', '0.25', '0.5', '0.75', '0.9')}
    comp['start_rows_route_share'] = {k: float((rt == k).mean()) for k in np.unique(rt)}
    teach = dict(teach0)
    for j, r in enumerate(user_cands.get(s, [])):
      teach[f'user_detour{j}'] = np.asarray(r['action'], np.float32)
    for arm in args.arms:
      name = f'{arm}_s{s}' + (f'@{args.variant}' if args.variant else '')
      b = bundle(MP.run_dir(arm, s, MP.variant_base(args.variant) if args.variant else None) / 'final.pkl')
      key = jax.random.PRNGKey(SEED + 10 * s + (0 if arm == 'CF' else 1))
      out = {'composition': comp, 'pairing': {}}
      for pairing, goals in (('real_relabeled_goal', G), ('task_goal_same_rows', TG)):
        Q = analyse(b, S, goals, A, teach, key)
        blk = {'all_start_rows': summarise(Q, np.ones(len(S), bool)),
               'shortcut_episode_rows': summarise(Q, rt == 'shortcut'), 'detour_episode_rows': summarise(Q, rt == 'detour'),
               'by_goal_region': {k: summarise(Q, (greg == k) & (rt == 'shortcut')) for k in np.unique(greg)},
               'by_goal_offset': {f'm in [{lo}, {hi})': summarise(Q, (mm >= lo) & (mm < hi) & (rt == 'shortcut')) for lo, hi in ((1, 10), (10, 50), (50, 150), (150, 100000))}}
        out['pairing'][pairing] = blk
        a = blk['all_start_rows']; sc_ = blk['shortcut_episode_rows']
        print(f'{name} [{pairing}] start rows {a["n"]}: signal {a["signal_mean (best detour - E_f)"]:+.3f} (P {a["P_signal_positive"]:.2f}); '
              f'shortcut-episode rows: signal {sc_["signal_mean (best detour - E_f)"]:+.3f} (P {sc_["P_signal_positive"]:.2f}), grad proj {sc_["grad_proj_toward_detour_mean"]:+.3f} (P {sc_["P_grad_proj_positive"]:.2f}), '
              f'delta total held {sc_["delta_total_at_target_held_mean"]:+.3f} (P better {sc_["P_total_better_held"]:.2f}) widen {sc_["delta_total_at_target_widen_mean"]:+.3f} (P {sc_["P_total_better_widen"]:.2f})', flush=True)
        for k, v in blk['by_goal_region'].items():
          if v:
            print(f'    goal in {k:26s} n {v["n"]:5d}: signal {v["signal_mean (best detour - E_f)"]:+.3f} (P {v["P_signal_positive"]:.2f}) delta total widen {v["delta_total_at_target_widen_mean"]:+.3f} (P {v["P_total_better_widen"]:.2f})', flush=True)
      res['per_actor'][name] = out
  MP.write_json(Path(args.out), res)


if __name__ == '__main__':
  main()
