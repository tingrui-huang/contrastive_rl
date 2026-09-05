"""Authoritative evaluation of a trained baseline on the V5 rockfall-clock
benchmark: does the u-blind agent hold at the mouth, take the detour, or
walk into a burst it cannot see coming?

Protocol (V3/V4): reset() -- ONE canonical start pose, the latent drawn by
the env at its natural Bernoulli(0.30) AND the burst start t0 ~ U{0..30}
drawn by the env's own schedule rng, horizon 400, eval seed 909, n=300.

Two policies from the same checkpoint (--mode):
  mean           deterministic tanh(mu) actor -- the repo's standard eval
  critic_select  at every step draw K=64 actions from the actor's own
                 tanh-Gaussian (plus its mode), score them with the twin-min
                 contrastive critic f(s, a, g), act on the argmax (the V3
                 critic-select probe). Training untouched.

Per-episode readouts beyond success / death / timeout / discounted:
  route (info['route']: 'shortcut' on band entry, 'detour' high in the west
  column, None if neither happened), rockfall_start / rockfall_end (the
  env's schedule, None when clear), mouth_step, band_entry_step, hesitation
  = band_entry - mouth (steps between the mouth line and the band; the
  expert spends ~11 when it goes and up to ~90 when it waits for the rocks
  to be parked), entered_while_open (active only: the ant was inside the
  band before the parking step, i.e. it did not wait long enough),
  stop_steps (steps in the approach zone [MOUTH_X, HAZARD_X0] with planar
  speed < STOP_V; a 'stand' proxy), min speed in that zone.

Why V5 needs the route readout: unlike V4, the far05 dataset shows the
learner the V3-br detour (5% of expert episodes), so a blind agent has a
third option besides walking and standing -- a safe ~225-step route that
never crosses the band. The summary therefore reports shortcut / detour /
no_route rates overall and by latent, and death | active & shortcut
separately from death | active.

Mouth diagnostic (both modes): at the first step past the mouth line, the
critic's score of the actor's mode vs the ZERO action (the expert's hold),
plus the actor's sigma there -- the direct test of 'the critic scores walk
above stand at the mouth'.

Writes <ckpt_dir>/eval_rockfall_clock_v5_<mode>.json and appends a row to
artifacts/rockfall_clock_v5/baseline_results.json. Reference numbers
(always_go / always_wait / always_detour / oracle / best_blind) come from
artifacts/rockfall_clock_v5/causal_audit.json when present.

Usage: python scripts/eval_rockfall_clock_v5_baseline.py \
           --ckpt <run>/final.pkl [--mode mean] [--n 300] [--method-label L]
           [--variant near|far05]   (default: 'far05' if the ckpt path
                                     contains 'far05', else 'near')
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

from crl import envs as envs_mod          # noqa: E402
from crl import networks as networks_mod  # noqa: E402
from crl import checkpoint as ckpt_mod    # noqa: E402
from crl import rockfall_clock_v5 as V5   # noqa: E402
from crl.tworoute_rockfall_v3 import HAZARD_X, HAZARD_HALF_Y  # noqa: E402
from verify_offline_d4rl import build_offline_cfg  # noqa: E402

OUT_ROOT = 'artifacts/rockfall_clock_v5'
ENV_NAME = 'offline_ant_umaze_rockfall_clock_v5'
VARIANTS = ('near', 'far05')
HORIZON = 400
GAMMA = 0.99
K = 64
STOP_V = 0.15       #: planar speed below which a step counts as standing
HESITATION_HOLD = 60  #: hesitation >= this = 'waited' (expert: ~90 vs ~11)
ROUTES = ('shortcut', 'detour', None)


def _yaw_deg(quat):
  q = np.asarray(quat, float)
  w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
  return float(np.degrees(np.arctan2(2 * (w * z + x * y),
                                     1 - 2 * (y * y + z * z))))


def wilson(k, n, z=1.96):
  if n == 0:
    return [None, None]
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return [round(float(c - h), 4), round(float(c + h), 4)]


def infer_variant(ckpt_path):
  """'far05' if the checkpoint path names the far05 dataset, else 'near'."""
  return 'far05' if 'far05' in ckpt_path.replace('\\', '/') else 'near'


def infer_goal_rep(ckpt_path):
  """'xy' if the checkpoint was trained on the upstream XY-goal env."""
  path = ckpt_path.replace(os.sep, '/')
  return 'xy' if '_gxy' in path else 'full'


def make_env(seed, horizon=HORIZON):
  cfg = build_offline_cfg()
  cfg.offline_dataset = ''
  cfg.eval_goal_mode = 'd4rl'
  cfg.rockfall_max_steps = horizon
  return cfg, envs_mod.make_env(ENV_NAME, cfg, seed=seed)


def build_policy(ckpt_path, k=K, horizon=HORIZON):
  """(act_mean, candidates, step). candidates(o[1,D], key) -> (acts
  [k+2, A] with the mode at index 0 and the zero action last, twin-min q
  [k+2], sigma [A])."""
  cfg, _ = make_env(1, horizon)
  nets = networks_mod.make_networks(
      obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
      repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
      repr_norm_temp=cfg.repr_norm_temp,
      hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs, use_layer_norm=cfg.use_layer_norm)
  step, st = ckpt_mod.load_checkpoint(ckpt_path)
  pp, qp = st.policy_params, st.q_params
  #: guard against evaluating a checkpoint under the wrong goal contract: a
  #: 'full' (58-wide) policy silently paired with the 31-wide XY-goal env, or
  #: the reverse, would otherwise fail deep inside jax or, worse, not at all.
  trained_width = int(pp['mlp/~/linear_0']['w'].shape[0])
  env_width = int(cfg.obs_dim) + int(cfg.goal_dim)
  assert trained_width == env_width, (
      f'checkpoint expects {trained_width}-dim observations but '
      f'{ENV_NAME} produces {env_width}; pass --goal-rep '
      f'{"xy" if trained_width < env_width else "full"}')

  @jax.jit
  def act_mean(o):
    return jnp.tanh(nets.policy_network.apply(pp, o).loc)

  @jax.jit
  def candidates(o, key):
    p = nets.policy_network.apply(pp, o)
    loc, scale = p.loc[0], p.scale[0]
    u = loc[None] + scale[None] * jax.random.normal(key, (k, loc.shape[0]))
    acts = jnp.concatenate([jnp.tanh(loc)[None], jnp.tanh(u),
                            jnp.zeros((1, loc.shape[0]))], 0)
    q = nets.q_network.apply(qp, jnp.tile(o, (k + 2, 1)), acts)
    if q.ndim == 3:
      q = jnp.min(q, axis=-1)
    return acts, jnp.diag(q), scale

  return act_mean, candidates, int(step)


def _in_zone(x, y):
  return V5.MOUTH_X <= x < HAZARD_X[0] and abs(y) < HAZARD_HALF_Y


def evaluate(act_mean, candidates, mode, n, seed, horizon=HORIZON):
  _, env = make_env(seed, horizon)
  rng = jax.random.PRNGKey(seed)
  rows = []
  for kk in range(n):
    o = env.reset()
    yaw = _yaw_deg(o[3:7])
    assert abs(yaw) <= 15.0, f'episode {kk}: non-canonical start yaw {yaw:.1f}'
    u = bool(env.privileged_rockfall_active)     # audit only
    #: the schedule is read once at reset for the row (privileged, never
    #: shown to the policy): t0 is None when the latent is clear.
    t0 = env.privileged_rockfall_start
    t_end = env.privileged_rockfall_end
    ret, info, mouth = 0.0, {}, None
    stop_steps, zone_steps, vmin = 0, 0, None
    for t in range(horizon):
      x, y = float(o[0]), float(o[1])
      in_zone = _in_zone(x, y)
      if in_zone:
        v = float(np.hypot(o[15], o[16]))
        zone_steps += 1
        stop_steps += int(v < STOP_V)
        vmin = v if vmin is None else min(vmin, v)
      need_cands = mode == 'critic_select' or (mouth is None and in_zone)
      if need_cands:
        rng, sub = jax.random.split(rng)
        acts, q, sigma = candidates(jnp.asarray(o[None]), sub)
        acts, q, sigma = np.asarray(acts), np.asarray(q), np.asarray(sigma)
        #: the zero action (index k+1) is a diagnostic candidate only: the
        #: selector may pick from the actor's mode + samples (0..k).
        j = int(np.argmax(q[:-1]))
        if mouth is None and in_zone:
          mouth = {'t': t, 'x': round(x, 3),
                   'sigma_mean': round(float(sigma.mean()), 4),
                   'q_mode': round(float(q[0]), 4),
                   'q_zero': round(float(q[-1]), 4),
                   'q_best_sample': round(float(q[:-1].max()), 4),
                   'q_samples_mean': round(float(q[1:-1].mean()), 4),
                   'mode_norm': round(float(np.linalg.norm(acts[0])), 4),
                   'chosen_norm': round(float(np.linalg.norm(acts[j])), 4)}
        a = acts[j] if mode == 'critic_select' else None
      if mode != 'critic_select':
        a = np.asarray(act_mean(jnp.asarray(o[None]))[0])
      o, r, done, info = env.step(a)
      ret += float(r)
      if done or r > 0:
        break
    steps = int(t + 1)
    success = bool(info.get('success'))
    failure = bool(info.get('failure'))
    ms, entry = info.get('mouth_step'), info.get('band_entry_step')
    hes = (entry - ms) if (ms is not None and entry is not None) else None
    rows.append({'u': u, 'start_yaw_deg': round(float(yaw), 2),
                 'success': success, 'failure': failure,
                 'timeout': bool(not success and not failure),
                 'entered_hazard': bool(info.get('entered_hazard')),
                 'route': info.get('route'),
                 'rockfall_start': t0, 'rockfall_end': t_end,
                 'rock_waves': int(info.get('rock_waves', 0)),
                 'rockfall_passed': bool(info.get('rockfall_passed')),
                 'mouth_step': ms, 'band_entry_step': entry,
                 'hesitation': hes,
                 #: active only: the band was entered before the rocks were
                 #: parked (the env parks them at the top of step t_end).
                 'entered_while_open': (bool(entry is not None
                                             and entry < t_end)
                                        if u else None),
                 'stop_steps': stop_steps, 'zone_steps': zone_steps,
                 'zone_vmin': round(vmin, 4) if vmin is not None else None,
                 'mouth': mouth,
                 'return': ret, 'steps': steps,
                 'discounted': (round(float(GAMMA ** steps), 6)
                                if success else 0.0)})
    if (kk + 1) % 50 == 0:
      print(f'  {kk + 1}/{n}', flush=True)
  return rows


def summarize(rows):
  n = len(rows)

  def m(xs, key):
    v = [x[key] for x in xs if x[key] is not None]
    return round(float(np.mean(v)), 4) if v else None

  def route_rates(xs):
    if not xs:
      return {'shortcut': None, 'detour': None, 'no_route': None}
    return {'shortcut': round(float(np.mean(
                [x['route'] == 'shortcut' for x in xs])), 4),
            'detour': round(float(np.mean(
                [x['route'] == 'detour' for x in xs])), 4),
            'no_route': round(float(np.mean(
                [x['route'] is None for x in xs])), 4)}

  def block(xs):
    hes = [x['hesitation'] for x in xs if x['hesitation'] is not None]
    return {'n': len(xs), 'success': m(xs, 'success'),
            'failure': m(xs, 'failure'),
            'timeout': m(xs, 'timeout'),
            'entered_band': m(xs, 'entered_hazard'),
            'route': route_rates(xs),
            'hesitation_n': len(hes),
            'hesitation_mean': (round(float(np.mean(hes)), 1) if hes else None),
            'hesitation_median': (round(float(np.median(hes)), 1)
                                  if hes else None),
            'waited_rate': (round(float(np.mean(
                [h >= HESITATION_HOLD for h in hes])), 4) if hes else None),
            'stop_steps_mean': m(xs, 'stop_steps'),
            'stopped_rate': (round(float(np.mean(
                [x['stop_steps'] >= 10 for x in xs])), 4) if xs else None),
            'zone_vmin_mean': m(xs, 'zone_vmin'),
            'mean_discounted': m(xs, 'discounted'),
            'mean_steps': m(xs, 'steps')}

  act = [r for r in rows if r['u']]
  clr = [r for r in rows if not r['u']]
  act_sc = [r for r in act if r['route'] == 'shortcut']
  act_dt = [r for r in act if r['route'] == 'detour']
  mouths = [r['mouth'] for r in rows if r['mouth'] is not None]
  mouth = {}
  if mouths:
    for key in ('sigma_mean', 'q_mode', 'q_zero', 'q_best_sample',
                'q_samples_mean', 'mode_norm', 'chosen_norm', 'x'):
      mouth[key] = round(float(np.mean([mm[key] for mm in mouths])), 4)
    mouth['frac_q_mode_gt_q_zero'] = round(float(np.mean(
        [mm['q_mode'] > mm['q_zero'] for mm in mouths])), 4)
    mouth['n'] = len(mouths)
  n_dead = int(sum(r['failure'] for r in rows))
  return {'overall': block(rows),
          'death_rate_ci95': wilson(n_dead, n),
          'by_latent': {'clear': block(clr), 'active': block(act)},
          'by_route': {'shortcut': block([r for r in rows
                                          if r['route'] == 'shortcut']),
                       'detour': block([r for r in rows
                                        if r['route'] == 'detour']),
                       'no_route': block([r for r in rows
                                          if r['route'] is None])},
          'death_given_active': m(act, 'failure'),
          'death_given_active_and_shortcut': m(act_sc, 'failure'),
          'detour_given_active': (round(float(len(act_dt) / len(act)), 4)
                                  if act else None),
          'entered_while_open_given_active': m(act, 'entered_while_open'),
          'mouth_diagnostic': mouth,
          'start_yaw_deg_range': [round(float(min(r['start_yaw_deg']
                                                  for r in rows)), 2),
                                  round(float(max(r['start_yaw_deg']
                                                  for r in rows)), 2)]}


def load_refs():
  """Measured references from the teacher / causal audits, if present."""
  refs = {}
  p = os.path.join(OUT_ROOT, 'teacher_audit.json')
  if os.path.exists(p):
    s = json.load(open(p))['summary']
    refs['teacher_audit'] = {
        'sighted_success': s.get('success'),
        'sighted_discounted': s.get('discounted_return'),
        'sighted_discounted_by_latent': s.get('discounted_return_by_latent'),
        'sighted_discounted_by_route': s.get('discounted_return_by_route'),
        'do_go_given_active_failure': s.get('do_go_given_active',
                                            {}).get('failure')}
  p = os.path.join(OUT_ROOT, 'causal_audit.json')
  if os.path.exists(p):
    refs['causal_audit'] = json.load(open(p)).get('reference_numbers')
  return refs


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--mode', choices=['mean', 'critic_select'], default='mean')
  ap.add_argument('--variant', choices=VARIANTS, default=None,
                  help='dataset the ckpt was trained on (default: inferred '
                       'from the ckpt path)')
  ap.add_argument('--n', type=int, default=300)
  ap.add_argument('--seed', type=int, default=909)
  ap.add_argument('--k', type=int, default=K)
  ap.add_argument('--method-label', default=None)
  ap.add_argument('--horizon', type=int, default=HORIZON)
  ap.add_argument('--out-dir', default=None)
  ap.add_argument('--results-root', default=OUT_ROOT)
  #: must match how the checkpoint was trained; inferred from its path.
  ap.add_argument('--goal-rep', choices=['full', 'xy'], default=None)
  args = ap.parse_args()
  variant = args.variant or infer_variant(args.ckpt)
  goal_rep = args.goal_rep or infer_goal_rep(args.ckpt)
  global ENV_NAME
  if goal_rep == 'xy':
    ENV_NAME = ENV_NAME + '_gxy'
  act_mean, candidates, step = build_policy(args.ckpt, args.k, args.horizon)
  print(f'ckpt {args.ckpt} @ step {step} | mode {args.mode} | '
        f'variant {variant} | goal {goal_rep} ({ENV_NAME})', flush=True)
  rows = evaluate(act_mean, candidates, args.mode, args.n, args.seed,
                  horizon=args.horizon)
  s = summarize(rows)
  print(json.dumps(s, indent=2), flush=True)
  label = args.method_label or os.path.basename(os.path.dirname(args.ckpt))
  label = f'{label}_{args.mode}'
  rec = {'label': label, 'mode': args.mode, 'variant': variant,
         'goal_rep': goal_rep,
         'env': ENV_NAME, 'ckpt': args.ckpt, 'ckpt_step': step,
         'n_eval': args.n, 'eval_seed': args.seed, 'horizon': args.horizon,
         'k': args.k, 'reference_numbers': load_refs(), 'summary': s}
  out_dir = args.out_dir or (os.path.dirname(args.ckpt) or '.')
  os.makedirs(out_dir, exist_ok=True)
  out_local = os.path.join(out_dir,
                           f'eval_rockfall_clock_v5_{args.mode}.json')
  with open(out_local, 'w') as f:
    json.dump({**rec, 'episodes': rows}, f, indent=2)
  os.makedirs(args.results_root, exist_ok=True)
  agg_path = os.path.join(args.results_root, 'baseline_results.json')
  agg = []
  if os.path.exists(agg_path):
    agg = json.load(open(agg_path))
  agg = [r for r in agg if r['label'] != label] + [rec]
  with open(agg_path, 'w') as f:
    json.dump(agg, f, indent=2)
  print('->', out_local, 'and', agg_path, flush=True)


if __name__ == '__main__':
  main()
