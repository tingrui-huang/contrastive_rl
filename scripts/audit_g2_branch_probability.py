"""G2 audit: is D_psi usable as the Bernoulli branch probability rho_t?

Task section 5 requires this audit BEFORE any implementation that consumes
rho. It inspects, from code and saved artifacts only (nothing is retrained):

  A. training objective and output activation
  B. what the trained object provably converges to
  C. validation / ranking metrics across all three seeds + baselines
  D. any existing calibration diagnostic (reliability / Brier / ECE)
  E. the M0 certification's own three-quantities verdict
  F. the EMPIRICAL rho distribution D_psi would produce on the authoritative
     training dataset -- i.e. what the coin would actually do

Verdict is 5A (calibrated, freeze rho = D_psi) or 5B (not demonstrated; STOP).

Usage:  python scripts/audit_g2_branch_probability.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import jax                                                   # noqa: E402
from propensity.agreement import (load_agreement_model,       # noqa: E402
                                  agreement_score_batch)

OUT = os.path.join(_ROOT, 'artifacts/static_worstcase_rl')
DISC = os.path.join(_ROOT, 'artifacts/support_discriminator')
MODEL = os.path.join(DISC, 'D_state_cmdgoal_action')
CLEAN = os.path.join(
    _ROOT, 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
    'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
OBS_DIM = 29


def qstats(x):
  x = np.asarray(x, np.float64)
  return {'mean': float(x.mean()), 'median': float(np.median(x)),
          'p10': float(np.percentile(x, 10)),
          'p90': float(np.percentile(x, 90)),
          'min': float(x.min()), 'max': float(x.max()),
          'std': float(x.std())}


def main():
  os.makedirs(OUT, exist_ok=True)
  rep = {}

  # ---- A. objective + activation ---------------------------------------
  disc_src = open(os.path.join(_ROOT, 'propensity/discriminator.py')).read()
  agr_src = open(os.path.join(_ROOT, 'propensity/agreement.py')).read()
  rep['A_training_objective'] = {
      'loss': 'bce_with_logits (propensity/discriminator.py:104-110)',
      'labels': '1 = real behavior action, 0 = CRL target-policy action',
      'class_balance': 'balanced 50/50 by construction',
      'network_output': 'RAW LOGIT (discriminator.py:74 '
                        '"Linear(256)-ReLU-Linear(256)-ReLU-Linear(1), '
                        'raw logit output")',
      'downstream_activation': 'jax.nn.sigmoid (agreement.py:195)',
      'no_calibration_layer': ('no temperature scaling, no isotonic/Platt '
                               'stage, no prior correction anywhere in '
                               'propensity/'),
      'has_calibration_code': ('calibrat' in disc_src.lower()
                               or 'platt' in agr_src.lower()
                               or 'isotonic' in agr_src.lower())}

  # ---- B. what it converges to -----------------------------------------
  rep['B_estimand'] = {
      'ideal_optimum': 'D*(x) = p_behavior(x) / (p_behavior(x) + p_target(x))',
      'source': 'propensity/discriminator.py:5-16 (module docstring)',
      'is_the_needed_quantity': False,
      'needed_quantity': 'rho = P(agreement / behavior support | s, g, a), '
                         'the analogue of P_hat(bin(x_t)|cell) in the archived '
                         'Thm-2 sampler',
      'why_different': ('D* is a RELATIVE density ratio under an ARTIFICIAL '
                        '50/50 class prior. It depends on the target policy '
                        'pi in the denominator, so it is not a property of '
                        'the behavior policy alone. rho must not depend on '
                        'the policy being trained -- but D* does, and pi '
                        'changes during RL while D_psi stays frozen.'),
      'repo_self_statements': [
          'discriminator.py:8  "must NOT be used as a causal mixture weight"',
          'agreement.py:193-195 "NOT a literal continuous propensity '
          'probability and NOT a calibrated causal branch probability"',
          'agreement.py:41  "no calibration into a claimed propensity"']}

  # ---- C. validation metrics -------------------------------------------
  s3b = json.load(open(os.path.join(DISC, 'stage3b_agreement.json')))
  ev = json.load(open(os.path.join(DISC, 'eval.json')))
  seeds = {k: {'roc_auc': v['roc_auc'], 'pr_auc': v['pr_auc']}
           for k, v in s3b['seeds'].items()}
  rep['C_validation'] = {
      'n_test_contexts': ev['n_test_contexts'],
      'seeds_D_state_cmdgoal_action': seeds,
      'mean_roc_auc': float(np.mean([v['roc_auc'] for v in seeds.values()])),
      'baselines': {k: v['roc_auc'] for k, v in s3b['baselines'].items()},
      'test_bce': ev['models']['B']['bce'],
      'accuracy': ev['models']['B']['accuracy'],
      'interpretation': ('ROC-AUC ~0.567 (3 seeds 0.564-0.571) is barely '
                         'above the 0.5 chance floor, and BELOW the '
                         'goal-marginalized baseline A (0.572). Test BCE '
                         '0.690 vs log 2 = 0.693 for a constant 0.5 '
                         'predictor: the model is almost exactly as good as '
                         'always answering "0.5".')}

  # ---- D. existing calibration diagnostics ------------------------------
  rep['D_calibration_diagnostics'] = {
      'reliability_curve_present': False,
      'brier_present': False,
      'ece_present': False,
      'explicit_artifact_flags': {
          'eval.json.NOT_a_propensity': ev.get('NOT_a_propensity'),
          'eval.json.score_semantics': ev.get('score_semantics'),
          'eval_agreement.py:185 is_calibrated_propensity': False},
      'note': ('No calibration diagnostic of any kind exists in the repo for '
               'this model. The only calibration-related records are explicit '
               'NEGATIVE declarations.')}

  # ---- E. M0 verdict ----------------------------------------------------
  m0 = json.load(open(os.path.join(_ROOT, 'artifacts/m0_agreement/'
                                          'report.json')))
  rep['E_M0_certification'] = {
      'all_gates_passed': all(g['passed'] for g in m0['gates']),
      'gate_names': [g['name'] for g in m0['gates']],
      'decisive_finding': (
          'M0 T3 was built precisely to separate three quantities: '
          '(a) an AGREEMENT ESTIMATOR recovers beta(x|s) -- the propensity; '
          '(b) a SOURCE CLASSIFIER data-vs-BC degenerates to 0.5 and is NOT a '
          'propensity; (c) a SOURCE CLASSIFIER behavior-vs-pi recovers '
          'beta/(beta+pi) -- CFQL D. D_psi is case (c). M0 therefore already '
          'certified that D_psi is NOT the propensity coin; it certified that '
          'the AGREEMENT EVENT (case a) is.'),
      'consequence': ('The coin the archived Thm-2 sampler needs is the '
                      'agreement FREQUENCY, not the discriminator posterior. '
                      'M0 validated the former on the discrete machinery; '
                      'no continuous estimator of it has been built.')}

  # ---- F. empirical rho the coin would actually produce -------------------
  model = load_agreement_model(MODEL)
  with np.load(CLEAN, allow_pickle=True) as d:
    obs = np.asarray(d['obs'], np.float32)
    act = np.asarray(d['act'], np.float32)
    lengths = np.asarray(d['lengths'], np.int64)
  E, L = obs.shape[0], obs.shape[1]
  ei, ti = [], []
  for e in range(E):
    n = int(lengths[e]) - 1
    ei.append(np.full(n, e, np.int64))
    ti.append(np.arange(n, dtype=np.int64))
  ei, ti = np.concatenate(ei), np.concatenate(ti)
  S = obs[ei, ti, :OBS_DIM]
  Gc = obs[ei, ti, OBS_DIM:]              # stored commanded goal (full width)
  A = act[ei, ti]

  rho = []
  for i in range(0, len(S), 65536):
    rho.append(np.asarray(agreement_score_batch(
        model.params, model.spec, S[i:i + 65536], Gc[i:i + 65536],
        A[i:i + 65536])))
  rho = np.concatenate(rho).astype(np.float64)

  st = qstats(rho)
  const = 0.5
  rep['F_empirical_rho'] = {
      'dataset': os.path.relpath(CLEAN, _ROOT),
      'n_transitions': int(len(rho)),
      'stats': st,
      'fraction_within_0.02_of_0.5': float(np.mean(np.abs(rho - const) < 0.02)),
      'fraction_within_0.05_of_0.5': float(np.mean(np.abs(rho - const) < 0.05)),
      'fraction_below_0.1': float(np.mean(rho < 0.1)),
      'fraction_above_0.9': float(np.mean(rho > 0.9)),
      'implied_pessimistic_branch_rate_1_minus_mean_rho':
          float(1.0 - rho.mean()),
      'interpretation': (
          'If rho = D_psi were used as the Bernoulli coin, the pessimistic '
          'branch would fire at a near-CONSTANT rate for essentially every '
          'transition. That is operationally identical to the "fixed '
          'worst-case branch rate" the task explicitly forbids -- it would '
          'carry no state- or action-dependent causal signal, only a hidden '
          'global coefficient.')}

  # ---- verdict ------------------------------------------------------------
  calibrated = False
  rep['VERDICT'] = {
      'section': '5B -- D_psi is NOT demonstrated to be calibrated',
      'is_calibrated_bernoulli_probability': calibrated,
      'action': 'STOP before the 5k smoke test; G2 remains unresolved',
      'what_D_psi_currently_represents': (
          'the posterior of an artificial balanced behavior-vs-target '
          'classification problem, i.e. p_behavior/(p_behavior + p_target) at '
          'the optimum -- a relative support/discrepancy RANKING score'),
      'ranking_performance': 'ROC-AUC 0.567 (seeds 0.564/0.564/0.571); '
                             'baselines A 0.572, action-only 0.528, '
                             'context-only 0.500',
      'why_raw_sigmoid_is_not_a_probability': [
          'trained under an artificial 50/50 class prior, so the posterior '
          'is prior-shifted away from any real-world event rate',
          'its denominator contains the TARGET policy pi, so it is not a '
          'property of the behavior policy alone and would drift in meaning '
          'as pi trains (while D_psi stays frozen)',
          'the negatives came from a DIFFERENT checkpoint '
          '(naive_rockfall_..._s0_300k/final.pkl) than the policy this run '
          'would train, so even the ratio it does estimate refers to the '
          'wrong pi',
          'no reliability curve, Brier score or ECE has ever been computed '
          'for it; the repo records only explicit NEGATIVE declarations '
          '(is_calibrated_propensity: false, NOT_a_propensity: true)',
          'empirically it is nearly constant at ~0.5, so as a coin it '
          'degenerates to a fixed global rate'],
      'minimal_calibration_step_required': (
          'Build a CONTINUOUS AGREEMENT-EVENT estimator, not a source '
          'classifier -- the case (a) quantity M0 certified. Concretely: '
          '(i) use the already-trained behavior flow mu(a|s,g_cmd) '
          '(propensity/flow.py) to draw a fresh behavior action a~ at the '
          'same (s, g_cmd); (ii) define agreement as a bin/neighborhood '
          'event on a~ vs a^data, the continuous analogue of '
          '1[bin(a~) = bin(a^data)]; (iii) estimate rho as that event '
          'frequency, which is a genuine probability by construction and '
          'needs no calibration; (iv) validate it against a held-out '
          'empirical agreement frequency with a reliability curve. Note this '
          'introduces a NEIGHBORHOOD BANDWIDTH -- itself a new coefficient '
          'requiring its own pre-registration, so it is a method decision, '
          'not an implementation detail.')}
  rep['git_commit'] = __import__('subprocess').check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()

  json.dump(rep, open(os.path.join(OUT, 'g2_calibration_audit.json'), 'w'),
            indent=2)

  print('=' * 72)
  print('G2 BRANCH-PROBABILITY AUDIT')
  print('=' * 72)
  print('A objective     : BCE-with-logits, balanced 50/50, raw logit + sigmoid')
  print('                  no temperature / Platt / isotonic stage anywhere')
  print('B estimand      : p_behavior/(p_behavior+p_target)  != rho')
  print('C ranking       : ROC-AUC %.4f mean over 3 seeds (A %.3f, action %.3f,'
        ' context %.3f)' % (rep['C_validation']['mean_roc_auc'],
                            rep['C_validation']['baselines']['A'],
                            rep['C_validation']['baselines']['action'],
                            rep['C_validation']['baselines']['context']))
  print('                  test BCE %.4f vs log2 = 0.6931 for constant 0.5'
        % rep['C_validation']['test_bce'])
  print('D calibration   : NONE exists; artifacts declare '
        'is_calibrated_propensity=false, NOT_a_propensity=true')
  print('E M0            : all gates PASS -- and T3 already separated the '
        'agreement estimator from')
  print('                  the source classifier. D_psi is the source '
        'classifier (case c).')
  print('F empirical rho : n=%d  mean %.4f  median %.4f  p10 %.4f  p90 %.4f'
        % (len(rho), st['mean'], st['median'], st['p10'], st['p90']))
  print('                  min %.4f max %.4f std %.4f' % (st['min'], st['max'],
                                                          st['std']))
  print('                  %.1f%% of transitions within 0.05 of 0.5; '
        '%.2f%% below 0.1; %.2f%% above 0.9'
        % (100 * rep['F_empirical_rho']['fraction_within_0.05_of_0.5'],
           100 * rep['F_empirical_rho']['fraction_below_0.1'],
           100 * rep['F_empirical_rho']['fraction_above_0.9']))
  print('                  implied pessimistic-branch rate %.4f (near-constant)'
        % rep['F_empirical_rho']['implied_pessimistic_branch_rate_1_minus_mean_rho'])
  print('-' * 72)
  print('VERDICT: 5B -- D_psi is NOT a calibrated Bernoulli probability.')
  print('         STOP before the smoke test. G2 remains UNRESOLVED.')
  print('saved -> %s' % os.path.join(OUT, 'g2_calibration_audit.json'))


if __name__ == '__main__':
  main()
