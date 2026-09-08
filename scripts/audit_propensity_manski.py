r"""Is the propensity-weighted Manski bound usable here, or is it vacuous?

Diagnostic only. No training, no CRL change, no dynamics change. Nothing here
writes to any existing z_v0 / z_v1 artifact.

WHY THIS RUNS BEFORE ANY LIPSCHITZ WORK
---------------------------------------
Bo, Zhang & Gursoy (Causal MAML), Appendix A Eq (11), bound the interventional
transition WITHOUT any continuity assumption:

    That(s,x,s') in [ Ttilde(s,x,s') pi_B(s,x),
                      Ttilde(s,x,s') pi_B(s,x) + (1 - pi_B(s,x)) ]

    pi_B(s,x) = \int 1_{x = f_X(s,u)} P(u) du          (their Eq (10))

so the interval WIDTH is exactly 1 - pi_B(s,x): the probability mass on which
the demonstrator did something else, and about which the data says nothing. If
that width is small at the states where the decision is made, the Manski route
needs no Lipschitz constant, no ball, and no worst-case state search -- the
entire Case B problem is moot. If it is close to 1, the bound is vacuous and
something must bridge the support gap.

THE PART THE PAPERS DO NOT COVER. Their Eq (11) assumes a DISCRETE, FINITE
action domain ("we consistently assume the action domain X and the state domain
S to be discrete and finite"), and their conclusion names continuous actions as
future work. Our actions are continuous, so pi_B(s,x) is a DENSITY: P(X=x|s)=0
pointwise and the width is trivially 1. This script therefore measures the
COARSENED propensity

    pi_B(s, B(a,h)) = P( ||X - a|| < h | S = s )

for a grid of bin radii h, which turns the action into a genuinely discrete
variable to which Eq (11) applies verbatim. Two costs are paid for that and
BOTH are measured rather than assumed away:

  (1) COARSENING BIAS. Eq (11) applies to the coarsened action only if the
      dynamics do not vary much inside the bin. Panel C measures how much the
      true sink probability actually varies across the bin, so the bias is a
      number rather than a hope.
  (2) STATE SMOOTHING. In a discrete state space pi_B(s,.) is a count. Here it
      needs a state neighbourhood of radius rho, which is itself a smoothness
      choice. So "Manski needs no smoothness assumption" is not free for us,
      and Panel A sweeps rho instead of fixing one.

A CORRECTION THIS SCRIPT IS BUILT TO SETTLE. It is tempting to reason "the
gate-aware teacher enters under an active gate 0/501 times, so pi_B ~ 0 and the
bound is vacuous". That conditions on U. Eq (10) marginalises over U, and the
teacher enters freely whenever the gate is clear, so the marginal pi_B(s,enter)
is LARGE. Panel B measures the marginal quantity the bound actually uses.

WHAT THIS BENCHMARK CAN DO THAT THE PAPERS CANNOT. We own the simulator, so the
true interventional P(sink | s, do(a)) is computable by direct intervention with
fresh U ~ P(U) and fresh action noise. Panel B therefore reports not just the
Manski interval but whether it COVERS the truth and how much slack it leaves.

Usage:
  python scripts/audit_propensity_manski.py --oracle 200      # smoke
  python scripts/audit_propensity_manski.py                   # full
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from crl import envs as envs_mod                  # noqa: E402
from crl.config import Config                     # noqa: E402

ENV = 'point_two_route_swamp_windy_z_v1'
DATASET = 'datasets/swamp_windy_z_v1_merged_s0.npz'
OUT_DIR = 'artifacts/propensity_manski'
TRACKED = ['crl/envs.py', 'scripts/audit_propensity_manski.py']

# bin radii for the coarsened action, and state-neighbourhood radii
H_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
RHO_GRID = (0.05, 0.10, 0.20, 0.40)
RHO_MAIN = 0.20

# the decision points: the holding cell (2,3) and the corridor mouth (3,3)
QUERIES = (
    ('holding_mid', (2.50, 3.50)),
    ('holding_edge', (2.90, 3.50)),
    ('mouth', (3.10, 3.50)),
    ('corridor_1', (4.10, 3.50)),
)
A_ENTER = np.array([1.0, 0.0])          # straight into the corridor
A_WAIT = np.array([0.0, 0.0])           # hold position
A_BACK = np.array([-1.0, 0.0])          # retreat


def one_step_dead(env, xy, bits, action, rng):
  """E = 1 iff the step ends inside an active swamp cell. Fresh U and xi."""
  env.set_auto_resample(False)
  env._dead = False
  env._z = 0.0
  env.state = np.asarray(xy, float).copy()
  env.set_swamp(bits)
  env._rng = rng
  env.step(np.asarray(action, np.float32))
  return bool(env._dead)


def oracle_p(env, xy, a_center, h, m, rng, active_prob):
  """True P(sink | s, do(X in B(a,h))) with X uniform on the ball.

  Uniform-on-the-ball is a CHOICE of what the coarsened intervention means; it
  is stated rather than hidden, and Panel C reports the spread across the ball
  so the choice can be re-examined.
  """
  n = 0
  for _ in range(m):
    if h <= 0:
      a = a_center
    else:
      u = rng.normal(size=2)
      u /= np.linalg.norm(u)
      a = a_center + u * h * rng.random() ** 0.5
    a = np.clip(a, -1.0, 1.0)
    bits = rng.random(3) < active_prob
    n += one_step_dead(env, xy, bits, a, np.random.default_rng(
        int(rng.integers(1 << 30))))
  return n / float(m)


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--oracle', type=int, default=4000,
                  help='Monte-Carlo draws per oracle probability')
  ap.add_argument('--panel-a-n', type=int, default=4000,
                  help='at-risk dataset transitions sampled for Panel A')
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--dataset', default=DATASET)
  ap.add_argument('--out-dir', default=OUT_DIR)
  args = ap.parse_args()

  def git(*a):
    try:
      return subprocess.check_output(['git'] + list(a),
                                     cwd=_ROOT).decode().strip()
    except Exception:                             # pylint: disable=broad-except
      return ''

  def csha(p):
    h = hashlib.sha256()
    with np.load(p, allow_pickle=False) as d:
      for k in sorted(d.files):
        x = d[k]
        h.update(k.encode())
        h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()

  os.makedirs(args.out_dir, exist_ok=True)
  out = {'analysis_script': 'scripts/audit_propensity_manski.py', 'env': ENV,
         'dataset': args.dataset, 'dataset_content_sha256': csha(args.dataset),
         'code_commit': git('log', '-1', '--format=%H', '--', *TRACKED),
         'dirty': bool(git('status', '--porcelain', '--', *TRACKED)),
         'h_grid': list(H_GRID), 'rho_grid': list(RHO_GRID),
         'config': {'oracle_draws': args.oracle, 'panel_a_n': args.panel_a_n,
                    'seed': args.seed, 'rho_main': RHO_MAIN}}

  print('=' * 100)
  print('PROPENSITY-WEIGHTED MANSKI BOUND: IS IT USABLE OR VACUOUS?')
  print('=' * 100)
  print('  dataset sha %s' % out['dataset_content_sha256'])
  print('  code commit %s  dirty %s' % (out['code_commit'][:12], out['dirty']))

  cfg = Config(env_name=ENV)
  env = envs_mod.make_env(ENV, cfg, seed=args.seed)
  active_prob = float(env.active_prob)
  print('  env active_prob %.3f   action_noise %.3f'
        % (active_prob, env._action_noise))
  out['active_prob'] = active_prob

  # ------------------------------------------------------------------ data
  with np.load(args.dataset, allow_pickle=False) as d:
    obs, act = d['obs'], d['act']
  n_ep, L, _ = obs.shape
  s = obs[:, :-1, :3].reshape(-1, 3)
  s_next = obs[:, 1:, :3].reshape(-1, 3)
  a = act[:, :-1, :].reshape(-1, 2)
  alive = s[:, 2] == 0.0
  # the one-step sink EVENT, from an alive anchor
  died = alive & (s_next[:, 2] < 0.0)
  print('\n  transitions %s | alive anchors %s | one-step sinks %s (%.4f of '
        'alive)' % (format(len(s), ','), format(int(alive.sum()), ','),
                    format(int(died.sum()), ','), died.sum() / alive.sum()))
  out['data'] = {'transitions': int(len(s)), 'alive_anchors': int(alive.sum()),
                 'one_step_sinks': int(died.sum()),
                 'sink_rate_among_alive': float(died.sum() / alive.sum())}

  sa = s[alive][:, :2]
  aa = a[alive]
  da = died[alive]
  rng = np.random.default_rng(args.seed)

  def local(xy, rho):
    """Indices of alive transitions whose anchor is within rho of xy."""
    d2 = ((sa - np.asarray(xy, float)) ** 2).sum(1)
    return np.where(d2 < rho * rho)[0]

  # -------------------------------------------------------------- Panel A
  print('\n' + '=' * 100)
  print('PANEL A -- COARSENED PROPENSITY OVER AT-RISK TRANSITIONS')
  print('=' * 100)
  print('  For each sampled at-risk anchor (x in [2,6), y in [3,4)) the action')
  print('  bin is centred on THAT transition\'s OWN action, so this is the')
  print('  propensity of what the demonstrator actually did -- the most')
  print('  favourable case. Manski width = 1 - pi_B.')
  at_risk = np.where((sa[:, 0] >= 2.0) & (sa[:, 0] < 6.0)
                     & (sa[:, 1] >= 3.0) & (sa[:, 1] < 4.0))[0]
  pick = rng.choice(at_risk, min(args.panel_a_n, len(at_risk)), False)
  print('  at-risk alive anchors %s, sampled %s'
        % (format(len(at_risk), ','), format(len(pick), ',')))
  panel_a = {}
  print('\n  %-8s%10s%10s%10s%10s%10s%10s'
        % ('rho', 'n_local', 'h', 'pi_B', 'width', 'p10 wid', 'p90 wid'))
  for rho in RHO_GRID:
    loc = [local(sa[i], rho) for i in pick]
    nloc = np.array([len(x) for x in loc], float)
    for h in H_GRID:
      pis = []
      for i, ix in zip(pick, loc):
        if len(ix) == 0:
          continue
        dd = np.linalg.norm(aa[ix] - aa[i], axis=1)
        pis.append(float((dd < h).mean()))
      pis = np.array(pis)
      w = 1.0 - pis
      panel_a['rho%g_h%g' % (rho, h)] = {
          'rho': rho, 'h': h, 'n_queries': int(len(pis)),
          'median_n_local': float(np.median(nloc)),
          'mean_pi_B': float(pis.mean()), 'median_pi_B': float(np.median(pis)),
          'mean_width': float(w.mean()), 'median_width': float(np.median(w)),
          'p10_width': float(np.percentile(w, 10)),
          'p90_width': float(np.percentile(w, 90))}
      print('  %-8.2f%10s%10.2f%10.4f%10.4f%10.4f%10.4f'
            % (rho, format(int(np.median(nloc)), ','), h, pis.mean(),
               w.mean(), np.percentile(w, 10), np.percentile(w, 90)))
  out['panel_a_coarsened_propensity'] = panel_a

  # -------------------------------------------------------------- Panel B
  print('\n' + '=' * 100)
  print('PANEL B -- THE DECISION POINT: MANSKI INTERVAL vs THE TRUE EFFECT')
  print('=' * 100)
  print('  pi_B and Ttilde are estimated from the data alone (rho = %.2f).'
        % RHO_MAIN)
  print('  The oracle column intervenes in the simulator with fresh U ~ P(U)')
  print('  and fresh action noise, so it is the ground truth the learner')
  print('  cannot see. Manski = [Tt*pi, Tt*pi + 1 - pi] on P(sink | s, do(A)).')
  panel_b = {}
  for qname, xy in QUERIES:
    for aname, a_ref in (('enter', A_ENTER), ('wait', A_WAIT),
                         ('back', A_BACK)):
      ix = local(xy, RHO_MAIN)
      if len(ix) == 0:
        continue
      print('\n  %s %s   anchor (%.2f, %.2f)  local transitions %s'
            % (qname, aname, xy[0], xy[1], format(len(ix), ',')))
      print('    %-7s%9s%9s%9s%11s%11s%11s%9s%9s'
            % ('h', 'n_bin', 'pi_B', 'Ttilde', 'manski_lo', 'manski_hi',
               'width', 'oracle', 'covers'))
      for h in H_GRID:
        dd = np.linalg.norm(aa[ix] - a_ref, axis=1)
        inb = dd < h
        pi = float(inb.mean())
        nb = int(inb.sum())
        tt = float(da[ix][inb].mean()) if nb > 0 else float('nan')
        if nb == 0:
          lo, hi = 0.0, 1.0
        else:
          lo = tt * pi
          hi = tt * pi + (1.0 - pi)
        orc = oracle_p(env, xy, a_ref, h, args.oracle, rng, active_prob)
        cov = bool(lo - 1e-9 <= orc <= hi + 1e-9)
        panel_b['%s|%s|h%g' % (qname, aname, h)] = {
            'query': qname, 'action': aname, 'h': h, 'xy': list(xy),
            'n_local': int(len(ix)), 'n_in_bin': nb, 'pi_B': pi,
            'Ttilde_sink': None if nb == 0 else tt,
            'manski_lo': lo, 'manski_hi': hi, 'width': hi - lo,
            'oracle_p_sink': orc, 'covers': cov}
        print('    %-7.2f%9s%9.4f%9s%11.4f%11.4f%11.4f%9.4f%9s'
              % (h, format(nb, ','), pi,
                 '-' if nb == 0 else '%.4f' % tt, lo, hi, hi - lo, orc,
                 'YES' if cov else 'NO'))
  out['panel_b_decision_point'] = panel_b

  # -------------------------------------------------------------- Panel C
  print('\n' + '=' * 100)
  print('PANEL C -- COARSENING BIAS: HOW MUCH DOES THE TRUTH VARY IN THE BIN?')
  print('=' * 100)
  print('  Eq (11) treats the bin as ONE action. That is only legitimate if')
  print('  the true sink probability is roughly constant across the bin. Here')
  print('  the ball around a_enter is probed at 8 directions and the SPREAD of')
  print('  the true P(sink) is reported. A large spread means the coarsened')
  print('  "action" is not one action and the tight bound is bought with bias.')
  dirs8 = [np.array([np.cos(t), np.sin(t)])
           for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
  panel_c = {}
  m_c = max(400, args.oracle // 4)
  for qname, xy in QUERIES:
    print('\n  %s  anchor (%.2f, %.2f)   oracle draws %d per point'
          % (qname, xy[0], xy[1], m_c))
    print('    %-7s%11s%11s%11s%11s' % ('h', 'p_center', 'min_edge',
                                        'max_edge', 'spread'))
    for h in H_GRID:
      pc = oracle_p(env, xy, A_ENTER, 0.0, m_c, rng, active_prob)
      edge = [oracle_p(env, xy, np.clip(A_ENTER + h * v, -1, 1), 0.0, m_c,
                       rng, active_prob) for v in dirs8]
      panel_c['%s|h%g' % (qname, h)] = {
          'query': qname, 'h': h, 'p_center': pc, 'edge': edge,
          'min_edge': float(min(edge)), 'max_edge': float(max(edge)),
          'spread': float(max(edge) - min(edge))}
      print('    %-7.2f%11.4f%11.4f%11.4f%11.4f'
            % (h, pc, min(edge), max(edge), max(edge) - min(edge)))
  out['panel_c_coarsening_bias'] = panel_c

  p = os.path.join(args.out_dir, 'propensity_manski.json')
  with open(p, 'w') as f:
    json.dump(out, f, indent=2)
  print('\nwrote %s' % p)


if __name__ == '__main__':
  main()
