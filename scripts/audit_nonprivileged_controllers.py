"""Phase 1+2: audit the repository's non-privileged controllers and
pre-register the V3 diverse collection protocol.

A controller is ELIGIBLE only if its action depends exclusively on
learner-visible information. Every candidate is classified by reading the
code path that produces its action:

  eligible (learner-visible only)
    * blind teacher   collect_rockfall_v2_pilot.rollout(mode='blind') --
      base-lane walker, detour windows forced empty, base side drawn from an
      RNG independent of the mask. Already the source of D_fail^196.
    * center/coverage rollout(mode='coverage') -- rockfall_pilot.route_command
      ('center'), position-history only.
    * trained CRL policies -- tanh(loc(pi(obs58))) on the 58-dim learner
      observation [state(29), zero-padded goal(29)]. These were trained from
      obs/act alone and read nothing hidden at deployment.

  EXCLUDED (privileged)
    * sighted teacher rollout(mode='sighted') / rockfall_v2_teacher.
      active_site_windows(base_sgn, env.rockfall_mask) -- reads the hazard
      MASK to place detours. This is the confounder pathway U -> A and must
      never generate V3 factual data.
    * anything reading env.privileged_mask / privileged_severity / _dead /
      _drop_step / _hit_step, or forced-mask counterfactual replay.

The four pre-registered V3 arms are the four trained CRL policies. They are
chosen for BEHAVIOURAL DISTINCTNESS using ALREADY-PUBLISHED authoritative
diagnostics (eval_h800_resetfix/diag_final/diagnosis.json produced long
before this task), not by any post-hoc death count. The blind arm is not
re-run: its failure geometry is already represented by D_fail^196, and
repeating it would add samples without adding diversity.

Writes controller_audit.json + collection_protocol.json.

Usage:
  python scripts/audit_nonprivileged_controllers.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import litter_pilot_common as C            # noqa: E402

OUT = 'artifacts/flow_v3_diverse_failure'
EPISODES_PER_ARM = 500
N_ARMS = 4

ARMS = [
    {'arm': 'naive', 'kind': 'trained_crl_policy',
     'ckpt': 'naive_rockfall_v2_p30_h800_resetfix_s0_300k/best.pkl',
     'env_seed': 91_500_019, 'dataset_seed': 91_990_013,
     'documented': {'success': 0.515, 'hazard_exposure': 0.980,
                    'drop_rate': 0.450, 'center_fraction': 0.070}},
    {'arm': 'alpha0', 'kind': 'trained_crl_policy',
     'ckpt': 'failneg_clean_p30_h800_resetfix_a0_s0_300k/best.pkl',
     'env_seed': 92_500_019, 'dataset_seed': 92_990_013,
     'documented': {'success': 0.770, 'hazard_exposure': 0.395,
                    'drop_rate': 0.160, 'center_fraction': 0.280}},
    {'arm': 'alpha01_legacy', 'kind': 'trained_crl_policy',
     'ckpt': 'failneg_clean_p30_h800_resetfix_a01_s0_300k/best.pkl',
     'env_seed': 93_500_019, 'dataset_seed': 93_990_013,
     'documented': {'success': 0.870, 'hazard_exposure': 0.140,
                    'drop_rate': 0.035, 'center_fraction': 0.455}},
    {'arm': 'alpha01_settled', 'kind': 'trained_crl_policy',
     'ckpt': ('failneg_settledbank_a01_s0_300k/'
              'failneg_settledbank_p30_h800_resetfix_a01_s0_300k/best.pkl'),
     'env_seed': 94_500_019, 'dataset_seed': 94_990_013,
     'documented': {'success': 0.645, 'hazard_exposure': 0.770,
                    'drop_rate': 0.335, 'center_fraction': 0.260}},
]

CANDIDATES = [
    {'name': 'blind teacher',
     'location': "scripts/collect_rockfall_v2_pilot.py rollout(mode='blind')",
     'observation_inputs': ('learner obs58 + own x-history; base side from '
                            'an RNG independent of the mask'),
     'reads_mask_severity_U': False, 'reads_privileged_teacher_state': False,
     'deterministic': 'deterministic given the reset stream',
     'route_behavior': 'fixed base lane, no detours',
     'known_hazard_exposure': 1.000,
     'eligible': True,
     'used_in_v3': False,
     'reason': ('eligible, but it is exactly the behaviour that produced '
                'D_fail^196; re-running it would add samples without '
                'adding failure-anchor diversity. Its geometry is retained '
                'via D_fail^196 in the combined pool.')},
    {'name': 'center / coverage route',
     'location': ("scripts/collect_rockfall_v2_pilot.py rollout("
                  "mode='coverage') -> rockfall_pilot.route_command"),
     'observation_inputs': 'learner obs58 + own x-history',
     'reads_mask_severity_U': False, 'reads_privileged_teacher_state': False,
     'deterministic': 'deterministic given the reset stream',
     'route_behavior': 'centerline throughout',
     'known_hazard_exposure': 'approximately 0 by construction '
                              '(center route never enters a trigger band)',
     'eligible': True, 'used_in_v3': False,
     'reason': ('eligible but expected to contribute almost no settled-fatal '
                'transitions -- the center route is designed to avoid every '
                'trigger region, so it cannot broaden FAILURE geometry.')},
    {'name': 'sighted local-detour teacher',
     'location': 'scripts/rockfall_v2_teacher.py active_site_windows(...)',
     'observation_inputs': 'learner obs58 AND env.rockfall_mask',
     'reads_mask_severity_U': True, 'reads_privileged_teacher_state': True,
     'deterministic': 'deterministic', 'route_behavior': 'lane + detours',
     'known_hazard_exposure': 'low (avoids active sites)',
     'eligible': False, 'used_in_v3': False,
     'reason': 'PRIVILEGED: reads the hidden hazard mask (the U -> A '
               'confounder pathway). Excluded from all factual collection.'},
] + [{'name': 'trained CRL policy: ' + a['arm'],
      'location': a['ckpt'] + ' via tanh(loc(policy(obs58)))',
      'observation_inputs': ('58-dim learner observation only '
                             '[state(29), zero-padded goal(29)]'),
      'reads_mask_severity_U': False,
      'reads_privileged_teacher_state': False,
      'deterministic': 'deterministic eval action tanh(loc)',
      'route_behavior': ('center fraction %.3f (documented)'
                         % a['documented']['center_fraction']),
      'known_hazard_exposure': a['documented']['hazard_exposure'],
      'eligible': True, 'used_in_v3': True,
      'reason': ('eligible and behaviourally distinct; selected on '
                 'PRE-EXISTING published diagnostics, not on any death '
                 'count measured in this task')} for a in ARMS]


def main():
  os.makedirs(OUT, exist_ok=True)
  for a in ARMS:
    assert os.path.exists(a['ckpt']), 'missing checkpoint %s' % a['ckpt']
    a['ckpt_sha256'] = C.sha256_file(a['ckpt'])
    a['episodes'] = EPISODES_PER_ARM

  audit = {
      'eligibility_rule': ('action must depend only on learner-visible '
                           'information; any controller reading the rock '
                           'mask, severity, future rock schedule, hidden U, '
                           'privileged teacher state or counterfactual '
                           'information is excluded'),
      'candidates': CANDIDATES,
      'n_eligible': sum(c['eligible'] for c in CANDIDATES),
      'n_used_in_v3': sum(c['used_in_v3'] for c in CANDIDATES),
      'noise_fallback_used': False,
      'noise_fallback_note': ('not required -- four behaviourally distinct '
                              'EXISTING non-privileged controllers were '
                              'available, and the spec prefers existing '
                              'policies over synthetic action noise'),
      'git_commit': C.git_commit()}
  json.dump(audit, open(os.path.join(OUT, 'controller_audit.json'), 'w'),
            indent=2)

  proto = {
      'pre_registered': True,
      'scientific_change': 'failure-support DIVERSITY only',
      'frozen_elsewhere': {'lambda': 0.01, 'K': 256, 'R_fatal': 3.17,
                           'architecture': 'V2-SA', 'critic': 'C (frozen)'},
      'arms': ARMS, 'episodes_per_arm': EPISODES_PER_ARM,
      'n_arms': N_ARMS,
      'n_total_episodes': EPISODES_PER_ARM * N_ARMS,
      'env': {'name': 'offline_ant_umaze_rockfall', 'p_active': 0.30,
              'horizon': 800, 'reset_fix': True,
              'severity_probs': [0.80, 0.15, 0.05],
              'death_settle_substeps': 80},
      'collection_rules': ('every episode retained (successes AND failures); '
                           'no mask forcing, no severity forcing, no '
                           'failure-only filtering, no stop-at-target-deaths, '
                           'no alternate hidden worlds, no same-anchor '
                           'counterfactual replay'),
      'arm_selection_basis': ('behavioural distinctness from ALREADY-PUBLISHED '
                              'authoritative diag_final diagnostics; arms were '
                              'NOT chosen by death counts measured in this '
                              'task'),
      'blind_arm_excluded_note': ('D_fail^196 already covers the blind-lane '
                                  'failure geometry and is preserved in the '
                                  'combined pool'),
      'git_commit': C.git_commit()}
  json.dump(proto, open(os.path.join(OUT, 'collection_protocol.json'), 'w'),
            indent=2)

  print('eligible controllers: %d | used as V3 arms: %d | noise fallback: %s'
        % (audit['n_eligible'], audit['n_used_in_v3'],
           audit['noise_fallback_used']))
  print('\nEXCLUDED (privileged):')
  for c in CANDIDATES:
    if not c['eligible']:
      print('  %-34s %s' % (c['name'], c['reason'][:60]))
  print('\npre-registered arms (%d x %d = %d episodes):'
        % (N_ARMS, EPISODES_PER_ARM, N_ARMS * EPISODES_PER_ARM))
  for a in ARMS:
    d = a['documented']
    print('  %-16s hazard %.3f drop %.3f center %.3f | seed %d | %s...'
          % (a['arm'], d['hazard_exposure'], d['drop_rate'],
             d['center_fraction'], a['env_seed'], a['ckpt_sha256'][:12]))
  print('\nwrote controller_audit.json + collection_protocol.json -> %s' % OUT)


if __name__ == '__main__':
  main()
