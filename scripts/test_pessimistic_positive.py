"""Task section 8: unit tests for crl/pessimistic_positive.py.

Uses a SYNTHETIC worst-case table (deterministic, one distinct row per
transition) so the "positive goal equals obs_to_goal(s'_wc) exactly" assertion
is sharp and the test does not wait on the real Flow precompute. The real table
is exercised separately by the branch diagnostic once it exists.

  T1  nominal branch (B=1) is BITWISE identical to TrajectoryBuffer.sample()
      under the same base RNG state -- goal, observation, action, next_obs
  T2  worst-case branch (B=0) returns obs_to_goal(s'_wc) EXACTLY
  T3  in both branches the state/action/next_state halves are untouched;
      ONLY the goal changes
  T4  coin: rho respected (empirical rate), reproducible, [0,1] enforced,
      rho_fn required
  T5  mixed batch routes each row to the correct branch
  T6  the sampler performs NO dataset continuation, NO policy query at s'_wc,
      NO nearest-neighbour projection, NO Flow call (AST + symbol audit)
  T7  forcing a branch does not consume the coin RNG stream

Usage:  python scripts/test_pessimistic_positive.py
"""
import ast
import io
import json
import os
import sys
import tokenize

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from crl.replay import TrajectoryBuffer, obs_to_goal   # noqa: E402
from crl.pessimistic_positive import PessimisticPositiveBuffer  # noqa: E402

CLEAN = os.path.join(
    _ROOT, 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
    'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
OBS_DIM, ACT_DIM = 29, 8
OUT = os.path.join(_ROOT, 'artifacts/static_worstcase_rl')
RESULTS = []


def check(name, ok, detail=''):
  RESULTS.append({'test': name, 'passed': bool(ok), 'detail': detail})
  print('%-52s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail))
  return bool(ok)


def code_only(source):
  """Executable source: comments and docstrings stripped."""
  nc = tokenize.untokenize(
      t for t in tokenize.generate_tokens(io.StringIO(source).readline)
      if t.type != tokenize.COMMENT)
  tree = ast.parse(nc)
  for node in ast.walk(tree):
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                         ast.AsyncFunctionDef)):
      b = node.body
      if (b and isinstance(b[0], ast.Expr)
          and isinstance(b[0].value, ast.Constant)
          and isinstance(b[0].value.value, str)):
        b.pop(0)
  return ast.unparse(tree)


def build_buffer(n_eps=40):
  with np.load(CLEAN, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)[:n_eps]
    act = np.asarray(d['act'], np.float32)[:n_eps]
  E, L, W = obs.shape
  buf = TrajectoryBuffer(capacity_steps=E * L, ep_len_obs=L, full_obs_dim=W,
                         action_dim=ACT_DIM, obs_dim=OBS_DIM, start_index=0,
                         end_index=-1, discount=0.99, seed=123,
                         goal_indices=tuple(range(OBS_DIM)))
  for k in range(E):
    buf.add_episode(obs[k], act[k])
  return buf, E, L


def synthetic_table(E, L, path):
  """One distinct, recognizable s_wc per anchorable transition."""
  ei = np.repeat(np.arange(E, dtype=np.int64), L - 1)
  ti = np.tile(np.arange(L - 1, dtype=np.int64), E)
  flat = ei * L + ti
  rng = np.random.default_rng(7)
  s_wc = rng.normal(size=(len(flat), OBS_DIM)).astype(np.float32)
  s_wc[:, 0] = flat                       # exact provenance marker
  np.savez_compressed(path, flat_index=flat, episode_index=ei, time_index=ti,
                      s_wc=s_wc)
  return flat, s_wc


def main():
  buf, E, L = build_buffer()
  tmp = os.path.join(OUT, '_synthetic_table_for_tests.npz')
  os.makedirs(OUT, exist_ok=True)
  flat, s_wc = synthetic_table(E, L, tmp)

  rho_half = lambda s, g, a: np.full(len(s), 0.5)          # noqa: E731
  pb = PessimisticPositiveBuffer(buf, tmp, rho_fn=rho_half, seed=0)
  B = 512

  # ---- T1 nominal branch is bitwise identical to the baseline sampler ----
  st = buf._rng.bit_generator.state
  base_t = buf.sample(B)
  buf._rng.bit_generator.state = st                       # rewind base stream
  nom_t, aux = pb.sample(B, force_branch=1, return_aux=True)
  check('T1a_nominal_goal_bitwise_identical',
        np.array_equal(base_t.observation, nom_t.observation))
  check('T1b_nominal_action_identical',
        np.array_equal(base_t.action, nom_t.action))
  check('T1c_nominal_next_obs_identical',
        np.array_equal(base_t.next_observation, nom_t.next_observation))
  check('T1d_nominal_next_action_identical',
        np.array_equal(base_t.next_action, nom_t.next_action))

  # ---- T2 worst-case branch returns obs_to_goal(s_wc) exactly ------------
  buf._rng.bit_generator.state = st
  wc_t, wc_aux = pb.sample(B, force_branch=0, return_aux=True)
  rows = np.searchsorted(flat, wc_aux['traj'] * L + wc_aux['i'])
  want = obs_to_goal(s_wc[rows], 0, -1, tuple(range(OBS_DIM)))
  got = wc_t.observation[:, OBS_DIM:]
  check('T2a_worstcase_goal_is_exactly_obs_to_goal_swc',
        np.array_equal(got, want))
  # the provenance marker proves the lookup used the ANCHOR's flat index
  check('T2b_lookup_keyed_on_anchor_flat_index',
        np.array_equal(got[:, 0], (wc_aux['traj'] * L + wc_aux['i']
                                   ).astype(np.float32)))

  # ---- T3 only the goal changes ------------------------------------------
  check('T3a_state_half_unchanged',
        np.array_equal(base_t.observation[:, :OBS_DIM],
                       wc_t.observation[:, :OBS_DIM]))
  check('T3b_action_unchanged', np.array_equal(base_t.action, wc_t.action))
  check('T3c_next_state_half_unchanged',
        np.array_equal(base_t.next_observation[:, :OBS_DIM],
                       wc_t.next_observation[:, :OBS_DIM]))
  check('T3d_goal_actually_differs_from_nominal',
        not np.array_equal(base_t.observation[:, OBS_DIM:],
                           wc_t.observation[:, OBS_DIM:]))

  # ---- T4 coin ------------------------------------------------------------
  for r in (0.0, 0.25, 0.75, 1.0):
    pb2 = PessimisticPositiveBuffer(
        buf, tmp, rho_fn=lambda s, g, a, r=r: np.full(len(s), r), seed=5)
    hits = []
    for _ in range(20):
      _, a2 = pb2.sample(B, return_aux=True)
      hits.append(a2['nominal'].mean())
    emp = float(np.mean(hits))
    se = np.sqrt(max(r * (1 - r), 1e-12) / (B * 20))
    check('T4a_empirical_nominal_rate_matches_rho_%.2f' % r,
          abs(emp - r) < max(4 * se, 0.005), 'empirical %.4f' % emp)

  p1 = PessimisticPositiveBuffer(buf, tmp, rho_fn=rho_half, seed=99)
  p2 = PessimisticPositiveBuffer(buf, tmp, rho_fn=rho_half, seed=99)
  st2 = buf._rng.bit_generator.state
  _, a1 = p1.sample(B, return_aux=True)
  buf._rng.bit_generator.state = st2
  _, a2 = p2.sample(B, return_aux=True)
  check('T4b_coin_reproducible_same_seed',
        np.array_equal(a1['nominal'], a2['nominal']))

  bad = False
  try:
    PessimisticPositiveBuffer(
        buf, tmp, rho_fn=lambda s, g, a: np.full(len(s), 1.5)).sample(16)
  except ValueError:
    bad = True
  check('T4c_rho_outside_unit_interval_rejected', bad)
  req = False
  try:
    PessimisticPositiveBuffer(buf, tmp, rho_fn=None)
  except ValueError:
    req = True
  check('T4d_rho_fn_is_required', req)

  # ---- T5 mixed batch routes correctly ------------------------------------
  def rho_split(s, g, a):
    r = np.zeros(len(s))
    r[::2] = 1.0                       # even rows always nominal
    return r
  pm = PessimisticPositiveBuffer(buf, tmp, rho_fn=rho_split, seed=1)
  buf._rng.bit_generator.state = st
  mt, ma = pm.sample(B, return_aux=True)
  check('T5a_even_rows_nominal_odd_rows_pessimistic',
        bool(ma['nominal'][::2].all() and not ma['nominal'][1::2].any()))
  gm = mt.observation[:, OBS_DIM:]
  rows_m = np.searchsorted(flat, ma['traj'] * L + ma['i'])
  ok_nom = np.array_equal(gm[::2], base_t.observation[::2, OBS_DIM:])
  ok_wc = np.array_equal(gm[1::2],
                         obs_to_goal(s_wc[rows_m][1::2], 0, -1,
                                     tuple(range(OBS_DIM))))
  check('T5b_mixed_batch_each_row_correct_source', ok_nom and ok_wc)

  # ---- T6 forbidden operations absent -------------------------------------
  src = code_only(open(os.path.join(_ROOT,
                                    'crl/pessimistic_positive.py')).read())
  mods = set()
  for nd in ast.walk(ast.parse(open(os.path.join(
          _ROOT, 'crl/pessimistic_positive.py')).read())):
    if isinstance(nd, ast.Import):
      mods.update(a.name for a in nd.names)
    elif isinstance(nd, ast.ImportFrom):
      mods.add(nd.module or '')
  forbidden_mods = {m for m in mods if any(
      t in m for t in ('static_worstcase', 'networks', 'flow', 'policy',
                       'checkpoint', 'sklearn', 'scipy.spatial'))}
  forbidden_syms = [s for s in
                    ('policy_network', 'sample_actions', 'worst_case_next_state',
                     'KDTree', 'cKDTree', 'nearest', 'argmin', 'walk_from',
                     'while ') if s in src]
  check('T6a_no_flow_policy_or_projection_import', not forbidden_mods,
        str(sorted(forbidden_mods)))
  check('T6b_no_continuation_projection_or_policy_symbols',
        not forbidden_syms, str(forbidden_syms))
  check('T6c_pessimistic_path_is_pure_table_lookup',
        '_s_wc[rows]' in src.replace(' ', ''))

  # ---- T7 forcing does not consume the coin stream -------------------------
  pf = PessimisticPositiveBuffer(buf, tmp, rho_fn=rho_half, seed=42)
  before = pf._coin_rng.bit_generator.state['state']['state']
  buf._rng.bit_generator.state = st
  pf.sample(B, force_branch=1)
  pf.sample(B, force_branch=0)
  after = pf._coin_rng.bit_generator.state['state']['state']
  check('T7_forced_branch_does_not_consume_coin_rng', before == after)

  os.remove(tmp)
  n_fail = sum(1 for r in RESULTS if not r['passed'])
  json.dump({'n_tests': len(RESULTS), 'n_failed': n_fail, 'results': RESULTS,
             'table': 'synthetic (sharp-assertion harness)'},
            open(os.path.join(OUT, 'unit_tests_sampler.json'), 'w'), indent=2)
  print('\n%d/%d passed' % (len(RESULTS) - n_fail, len(RESULTS)))
  sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
  main()
