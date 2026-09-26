"""C1 -- the user's verified candidates at their ORIGINAL state and task goal:
why were they not adopted?  (Frozen clipped checkpoints; no rollouts.)

The user's frozen-model diagnostic (2026-09-19, `codex_clip_choice_v1/
joined.json`) found, at evaluation reset states of the clipped CF policies,
first torques that the seed's own critic scores above the policy's mode AND
whose realised continuation entered the far route.  Those candidates are
verified only there: at that state, with the task goal.  Transplanting them
to other training rows changes what they do (pose and velocity differ), so
the cross-state check lives in `diag_v6_actor_objective_stream.py` and is
labelled a transplant.  This script stays at the original state.

For every verified candidate c at reset state s (the recorded evaluation
draw; identical for every policy) with the task goal g:

  * the critic term the actor optimises: E_{a ~ pi(.|s,g)} f(s, a, g) under the
    current policy (N_MC samples), and the same expectation with the
    distribution moved to c (loc* = atanh(c), scale HELD -- the primary
    reading; the widened scale is listed separately); f at the mode, f at c;
  * how far c is from the current distribution: log pi(c | s, g), the
    per-dimension standardised offset |atanh(c) - loc| / scale;
  * the local signal: the action gradient of f at the mode projected toward c;
  * the BC pull that would resist the move: the actor never trains at s
    itself, so the K nearest dataset reset rows (29-dim state distance; the
    evaluation resets and the dataset resets share one distribution) stand in
    for it -- the NLL of their logged teacher torques under N(loc, scale)
    at the current loc and at loc*; the objective proxy
    0.95 * (-E f) + 0.05 * mean NLL along the loc path (lambda 0..1);
  * the same candidate at those K nearest rows with their REAL relabeled
    goals (the geometric law, one draw per row): E f vs the moved
    expectation -- the closest thing to "the training rows that shape this
    decision"; still a transplant (a different state), reported as such.

Readings, fixed before running: a candidate whose moved expectation (held
scale) exceeds E f at the original state and whose objective proxy also
improves is one the objective would prefer locally -- a local statement
(shared parameters: improving these inputs may cost others), not a proof
that the actor's optimisation failed; a candidate whose proxy does not
improve is not supported along the tested loc path -- other directions are
not excluded.

  JAX_PLATFORMS=cpu python scripts/diag_v6_candidate_at_origin.py --variant critic_clip0.1 --candidates /root/codex_joined.json --out <json>
"""
from __future__ import annotations

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
import diag_v6_actor_objective_stream as AO  # noqa: E402

N_MC, K_NEAR, BC = 64, 8, 0.05
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 171_000_000
STATE_DIM, OBS_W = MP.STATE_DIM, MP.OBS_W


def verified(J, seed):
  rows = []
  for c in J:
    if int(c['seed']) != seed:
      continue
    names, sc = c['candidate_names'], c['scores']; im = names.index('mode')
    for i, nm in enumerate(names):
      oc = c['outcomes'].get(nm, {})
      if nm != 'mode' and oc.get('route') == 'detour' and sc[i] > sc[im]:
        rows.append({'episode': int(c['episode']), 'candidate': nm, 'action': np.asarray(c['actions'][i], np.float32), 'user_score': float(sc[i]), 'user_mode_score': float(sc[im]),
                     'user_same_scale_qbar': float(c['same_scale_qbar'][i]), 'user_logp': float(c['own_logp'][i]), 'realised_success': bool(oc.get('success')), 'saved_route': c.get('saved_route')})
  return rows


def geom_goal(rng, obs_ep, i, L):
  n = L - 1 - i
  u = rng.random()
  m = int(np.clip(np.ceil(np.log1p(-u * (1.0 - MP.GAMMA ** n)) / np.log(MP.GAMMA)), 1, n))
  return obs_ep[i + m, :2].astype(np.float32), m


def main():
  import jax
  import jax.numpy as jnp
  ap = argparse.ArgumentParser()
  ap.add_argument('--variant', default='critic_clip0.1')
  ap.add_argument('--candidates', required=True)
  ap.add_argument('--out', required=True)
  args = ap.parse_args()
  J = json.loads(Path(args.candidates).read_text(encoding='utf-8'))
  obs, act, lengths, _ = MP.load_dataset()
  with np.load(MP.SIDECAR, allow_pickle=True) as sc:
    route = sc['route_realized'].astype(str)
  S0 = obs[:, 0, :STATE_DIM].astype(np.float32); A0 = act[:, 0].astype(np.float32)
  sd = np.maximum(S0.std(axis=0), 1e-3)
  res = {'n_mc': N_MC, 'k_near': K_NEAR, 'lambdas': list(LAMBDAS), 'variant': args.variant, 'per_seed': {}, 'rows': []}
  for s in MP.SEEDS:
    rows = verified(J, s)
    if not rows:
      continue
    T = DT.Traj(f'CF_s{s}@{args.variant}'); T0 = DT.Traj('start')
    b = AO.bundle(MP.run_dir('CF', s, MP.variant_base(args.variant)) / 'final.pkl')
    key = jax.random.PRNGKey(SEED + s)
    rng = np.random.default_rng(SEED + 100 + s)
    out_rows = []
    for r in rows:
      k = r['episode']
      o0 = T.obs(k)[0]; assert np.array_equal(o0, T0.obs(k)[0]), 'reset state differs across policies'
      st, g = o0[:STATE_DIM], o0[STATE_DIM:OBS_W]
      c = r['action']
      O = np.concatenate([st, g])[None].astype(np.float32)
      loc, scale = (np.asarray(v)[0] for v in b['dist'](jnp.asarray(O)))
      mode = np.tanh(loc).astype(np.float32)
      f_mode = float(b['f'](jnp.asarray(O), jnp.asarray(mode[None]))[0]); f_c = float(b['f'](jnp.asarray(O), jnp.asarray(c[None]))[0])
      loc_star = np.arctanh(np.clip(c, -0.999, 0.999)).astype(np.float32)
      widen = np.maximum(scale, 0.5 * np.abs(loc_star - loc)).astype(np.float32)

      def Ef(loc_, sc_, tag):
        return float(np.mean([np.asarray(b['f'](jnp.asarray(O), b['sample_from'](jnp.asarray(loc_[None]), jnp.asarray(sc_[None]), jax.random.fold_in(key, tag * 1000 + j))))[0] for j in range(N_MC)]))
      E_cur = Ef(loc, scale, 1); E_star = Ef(loc_star, scale, 2); E_star_w = Ef(loc_star, widen, 3)
      nll_c = float(b['nll'](loc[None], scale[None], c[None])[0])
      z = np.abs(loc_star - loc) / scale
      gr = np.asarray(b['grad_f_a'](jnp.asarray(O), jnp.asarray(mode[None])))[0]
      d = c - mode; proj = float((gr * d).sum() / (np.linalg.norm(d) + 1e-8))
      # the BC pull: the K nearest dataset reset rows
      dist = np.linalg.norm((S0 - st) / sd, axis=1); near = np.argsort(dist)[:K_NEAR]
      A_near = A0[near]
      path = []
      for lam in LAMBDAS:
        loc_l = ((1 - lam) * loc + lam * loc_star).astype(np.float32)
        E_l = Ef(loc_l, scale, 10 + int(lam * 4))
        nll_l = float(np.mean(b['nll'](np.repeat(loc_l[None], K_NEAR, 0), np.repeat(scale[None], K_NEAR, 0), A_near)))
        path.append({'lambda': lam, 'E_f': E_l, 'bc_nll_near': nll_l, 'total_proxy': (1 - BC) * (-E_l) + BC * nll_l})
      # the same candidate at the nearest rows with their real relabeled goals (a transplant to near-identical states)
      near_rows = []
      for e in near:
        gg, m = geom_goal(rng, obs[e], 0, int(lengths[e]))
        On = np.concatenate([S0[e], gg])[None].astype(np.float32)
        ln, scn = (np.asarray(v)[0] for v in b['dist'](jnp.asarray(On)))
        E_n = float(np.mean([np.asarray(b['f'](jnp.asarray(On), b['sample_from'](jnp.asarray(ln[None]), jnp.asarray(scn[None]), jax.random.fold_in(key, 50000 + int(e) * 7 + j))))[0] for j in range(N_MC)]))
        E_n_star = float(np.mean([np.asarray(b['f'](jnp.asarray(On), b['sample_from'](jnp.asarray(loc_star[None]), jnp.asarray(scn[None]), jax.random.fold_in(key, 60000 + int(e) * 7 + j))))[0] for j in range(N_MC)]))
        near_rows.append({'episode': int(e), 'route': str(route[e]), 'goal_offset_m': m, 'goal_xy': [float(x) for x in gg], 'E_f': E_n, 'E_f_moved_held': E_n_star,
                          'f_logged': float(b['f'](jnp.asarray(On), jnp.asarray(A0[e][None]))[0]), 'f_candidate': float(b['f'](jnp.asarray(On), jnp.asarray(c[None]))[0]),
                          'nll_logged_cur': float(b['nll'](ln[None], scn[None], A0[e][None])[0]), 'nll_logged_moved': float(b['nll'](loc_star[None], scn[None], A0[e][None])[0])})
      row = {'seed': s, 'episode': k, 'candidate': r['candidate'], 'saved_route': r['saved_route'], 'realised_success': r['realised_success'],
             'user': {'score': r['user_score'], 'mode_score': r['user_mode_score'], 'same_scale_qbar': r['user_same_scale_qbar'], 'logp': r['user_logp']},
             'task_goal': [float(x) for x in g], 'f_mode': f_mode, 'f_candidate': f_c, 'E_f_current': E_cur, 'E_f_moved_held': E_star, 'E_f_moved_widened': E_star_w,
             'delta_E_f_held': E_star - E_cur, 'logp_candidate': -nll_c, 'z_offset_max': float(z.max()), 'z_offset_mean': float(z.mean()), 'scale_mean': float(scale.mean()),
             'grad_proj_toward_candidate': proj, 'grad_norm': float(np.linalg.norm(gr)),
             'near_rows_detour_share': float(np.mean([route[e] == 'detour' for e in near])), 'near_rows_state_dist_max': float(dist[near].max()),
             'path_held': path, 'delta_total_proxy_held': path[-1]['total_proxy'] - path[0]['total_proxy'], 'delta_bc_nll_near': path[-1]['bc_nll_near'] - path[0]['bc_nll_near'],
             'near_rows_real_goals': near_rows,
             'near_rows_share_moved_above_current': float(np.mean([q['E_f_moved_held'] > q['E_f'] for q in near_rows])),
             'near_rows_mean_delta_E_f': float(np.mean([q['E_f_moved_held'] - q['E_f'] for q in near_rows]))}
      out_rows.append(row)
      print(f'seed {s} ep {k:3d} {r["candidate"]:16s} f(mode) {f_mode:+.3f} f(c) {f_c:+.3f} E_f {E_cur:+.3f} -> moved held {E_star:+.3f} (d {E_star - E_cur:+.3f}) widened {E_star_w:+.3f} | '
            f'logp(c) {-nll_c:+.1f} z_max {z.max():.1f} grad_proj {proj:+.3f} | BC nll near {path[0]["bc_nll_near"]:.2f} -> {path[-1]["bc_nll_near"]:.2f} total proxy d {row["delta_total_proxy_held"]:+.3f} | '
            f'near real goals: moved > current {row["near_rows_share_moved_above_current"]:.2f} (mean d {row["near_rows_mean_delta_E_f"]:+.3f})', flush=True)
    n = len(out_rows)
    res['per_seed'][str(s)] = {'n_candidates': n, 'moved_held_above_current': int(sum(r['delta_E_f_held'] > 0 for r in out_rows)),
                               'total_proxy_better': int(sum(r['delta_total_proxy_held'] < 0 for r in out_rows)), 'grad_proj_positive': int(sum(r['grad_proj_toward_candidate'] > 0 for r in out_rows)),
                               'f_candidate_above_f_mode': int(sum(r['f_candidate'] > r['f_mode'] for r in out_rows)),
                               'mean_delta_E_f_held': float(np.mean([r['delta_E_f_held'] for r in out_rows])), 'mean_delta_total_proxy': float(np.mean([r['delta_total_proxy_held'] for r in out_rows])),
                               'mean_delta_bc_nll_near': float(np.mean([r['delta_bc_nll_near'] for r in out_rows])), 'mean_logp_candidate': float(np.mean([r['logp_candidate'] for r in out_rows])),
                               'mean_z_offset_max': float(np.mean([r['z_offset_max'] for r in out_rows])),
                               'near_real_goals_moved_above_current_share': float(np.mean([r['near_rows_share_moved_above_current'] for r in out_rows]))}
    res['rows'] += out_rows
  MP.write_json(Path(args.out), res)
  print(json.dumps(res['per_seed'], indent=1), flush=True)


if __name__ == '__main__':
  main()
