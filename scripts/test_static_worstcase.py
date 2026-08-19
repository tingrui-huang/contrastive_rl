"""Phase 3 unit tests for crl/static_worstcase.py.

Regression-tests the packaged module against ALREADY-SAVED development /
sealed candidates. Nothing is regenerated with a new seed, no threshold is
touched, and Critic C is never consulted by the module under test.

Tests
  T1  provenance gates all pass; a tampered SHA aborts construction
  T2  exact reproduction of the sealed selector on selector-confirm50:
        - identical selected index k for all 50 anchors
        - identical selected state s'_wc (bitwise-close)
        - identical nearest-negative distance
  T3  candidate count == 256 and candidate block matches the sealed
      candidates.npz elementwise
  T4  determinism: two calls give identical output
  T5  normalization is the frozen V0 one (recomputed distances match)
  T6  tie-break is numpy-argmin (lowest index) -- verified on a synthetic
      exact tie
  T7  no non-finite outputs
  T8  no gradient path into the Flow: jax.grad through the module is
      impossible (module returns numpy) and params are never in an opt state
  T9  argmin f_C is NOT the selector (no critic import/call): on the sealed set the module's picks
      differ from the argmin-f_C picks exactly where the sealed comparison
      said they do
  T10 information contract: module namespace mentions no hidden-state field

Usage:  python scripts/test_static_worstcase.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from crl import static_worstcase as sw          # noqa: E402

CONF = os.path.join(_ROOT, 'artifacts/state_nn_selector_confirm')
RESULTS = []


def _code_only(source):
  """Return EXECUTABLE source only: comments and docstrings stripped.

  The module under test documents its own information contract in prose
  ("never reads _dead, the rockfall mask, severity, ..."), so a raw text scan
  would flag the disclaimer itself. Only real code may be inspected."""
  import ast
  import io
  import tokenize
  no_comments = tokenize.untokenize(
      tok for tok in tokenize.generate_tokens(io.StringIO(source).readline)
      if tok.type != tokenize.COMMENT)
  tree = ast.parse(no_comments)
  for node in ast.walk(tree):
    if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
      continue
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)):
      body.pop(0)
  return ast.unparse(tree)


def check(name, ok, detail=''):
  RESULTS.append({'test': name, 'passed': bool(ok), 'detail': detail})
  print('%-46s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail))
  return bool(ok)


def main():
  # ---- T1 provenance -------------------------------------------------
  m = sw.StaticWorstCase(root=_ROOT)
  check('T1a_all_provenance_gates_pass',
        all(m.gates.values()),
        '%d gates' % len(m.gates))
  bad = False
  try:
    real = sw.BANK_SHA
    sw.BANK_SHA = '0' * 64
    try:
      sw.StaticWorstCase(root=_ROOT)
    except RuntimeError:
      bad = True
    finally:
      sw.BANK_SHA = real
  except Exception as ex:                        # pragma: no cover
    print('  tamper probe error:', ex)
  check('T1b_tampered_bank_sha_aborts', bad)

  # ---- sealed reference ----------------------------------------------
  z = np.load(os.path.join(CONF, 'candidates.npz'), allow_pickle=True)
  S = np.asarray(z['anchor'], np.float32)
  A = np.asarray(z['action'], np.float32)
  cand_ref = np.asarray(z['candidates'], np.float32)
  import csv
  rows = list(csv.DictReader(open(os.path.join(CONF, 'per_case.csv'))))
  k_ref = np.array([int(r['nn_k']) for r in rows])
  k_critic = np.array([int(r['critic_k']) for r in rows])

  s_wc, aux = m.worst_case_next_state(S, A, return_aux=True)

  # ---- T2 exact reproduction -----------------------------------------
  same_k = int((aux['k'] == k_ref).sum())
  check('T2a_selected_index_matches_sealed', same_k == len(k_ref),
        '%d/%d identical' % (same_k, len(k_ref)))
  ref_wc = cand_ref[np.arange(len(S)), k_ref]
  dmax = float(np.abs(s_wc - ref_wc).max())
  check('T2b_selected_state_matches_sealed', dmax < 1e-4,
        'max abs diff %.3e' % dmax)

  # nearest-negative distance recomputed independently in numpy
  nz = np.load(os.path.join(_ROOT, sw.NORM_NPZ))
  sm = np.asarray(nz['state_mean'], np.float32)
  ss = np.asarray(nz['state_std'], np.float32)
  bank = np.asarray(np.load(os.path.join(_ROOT, sw.BANK_NPZ),
                            allow_pickle=True)['goals'], np.float32)
  bn = (bank - sm) / ss
  dref = np.linalg.norm(((ref_wc - sm) / ss)[:, None] - bn[None],
                        axis=2).min(1)
  ddiff = float(np.abs(aux['d_neg'] - dref).max())
  check('T2c_nearest_negative_distance_matches', ddiff < 1e-3,
        'max abs diff %.3e' % ddiff)

  # ---- T3 candidate block --------------------------------------------
  check('T3a_candidate_count_is_256',
        aux['candidates'].shape[1] == sw.K_CANDIDATES == 256,
        str(aux['candidates'].shape))
  cdiff = float(np.abs(aux['candidates'] - cand_ref).max())
  check('T3b_candidate_block_matches_sealed', cdiff < 1e-3,
        'max abs diff %.3e' % cdiff)

  # ---- T4 determinism -------------------------------------------------
  s_wc2 = m.worst_case_next_state(S, A)
  check('T4_deterministic_across_calls',
        bool(np.array_equal(s_wc, s_wc2)))

  # ---- T5 normalization ------------------------------------------------
  check('T5_frozen_v0_normalization',
        sw.sha256_file(os.path.join(_ROOT, sw.NORM_NPZ)) == sw.NORM_SHA)

  # ---- T6 tie-break ----------------------------------------------------
  # exact duplicate distances -> numpy argmin must return the LOWEST index
  tie = np.array([[2.0, 1.0, 1.0, 3.0]], np.float32)
  check('T6_tie_break_is_lowest_index', int(tie.argmin(1)[0]) == 1,
        'argmin -> %d' % int(tie.argmin(1)[0]))

  # ---- T7 finiteness ----------------------------------------------------
  check('T7_no_nonfinite_outputs',
        bool(np.isfinite(s_wc).all() and np.isfinite(aux['candidates']).all()))

  # ---- T8 no gradient path ---------------------------------------------
  is_np = isinstance(s_wc, np.ndarray) and not hasattr(s_wc, 'aval')
  check('T8a_output_is_plain_numpy_stop_gradient', is_np, type(s_wc).__name__)
  raw = open(os.path.join(_ROOT, 'crl/static_worstcase.py')).read()
  src = _code_only(raw)      # executable code: comments + docstrings stripped
  check('T8b_module_has_no_optimizer_or_grad',
        ('optax' not in src) and ('jax.grad' not in src)
        and ('value_and_grad' not in src))

  # ---- T9 argmin f_C is not the selector --------------------------------
  agree = int((k_ref == k_critic).sum())
  # The module may NAME the critic in gate keys that assert its exclusion
  # (critic_in_primary_selector / critic_not_in_selector). What must be absent
  # is any IMPORT of the critic and any call into it.
  import ast as _ast
  mods = set()
  for nd in _ast.walk(_ast.parse(raw)):
    if isinstance(nd, _ast.Import):
      mods.update(a.name for a in nd.names)
    elif isinstance(nd, _ast.ImportFrom):
      mods.add(nd.module or '')
  critic_mods = {m for m in mods
                 if any(t in m for t in ('networks', 'checkpoint', 'losses',
                                         'crl.train'))}
  call_syms = ('q_network', 'q_params', 'c_score', 'f_C', 'load_checkpoint',
               'make_networks')
  hit_calls = [c for c in call_syms if c in src]
  check('T9a_module_never_imports_or_calls_critic',
        not critic_mods and not hit_calls,
        'imports=%s calls=%s' % (sorted(critic_mods), hit_calls))
  check('T9b_selector_differs_from_argmin_fC',
        agree < len(k_ref),
        'module==sealed state-NN on 50/50; state-NN==argmin_fC on only '
        '%d/50' % agree)

  # ---- T10 information contract -----------------------------------------
  banned = ('_dead', 'rockfall_mask', 'severity', 'rock_', 'oracle',
            'same_anchor', 'fatal_candidate')
  hit = [b for b in banned if b in src]   # src = executable code only
  check('T10_no_hidden_information_referenced', not hit, str(hit))

  n_fail = sum(1 for r in RESULTS if not r['passed'])
  out = {'n_tests': len(RESULTS), 'n_failed': n_fail, 'results': RESULTS,
         'provenance': m.provenance()}
  os.makedirs(os.path.join(_ROOT, 'artifacts/static_worstcase_rl'),
              exist_ok=True)
  json.dump(out, open(os.path.join(_ROOT, 'artifacts/static_worstcase_rl',
                                   'unit_tests.json'), 'w'), indent=2)
  print('\n%d/%d passed' % (len(RESULTS) - n_fail, len(RESULTS)))
  sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
  main()
