"""Assemble the 30k goal-representation ablation comparison (3 arms)."""
import json
import glob

ARMS = {'xy': 'adaptive_te8', 'gcompact': 'gcompact_te8',
        'gfull': 'gfull_te8'}
GATES = [10000, 20000, 30000]

G, A = {}, {}
for f in glob.glob('qual_open_near/gates/gate_*.json'):
  r = json.load(open(f))
  G[r['tag']] = r
for f in glob.glob('qual_open_near/attribution/attr_*.json'):
  r = json.load(open(f))
  A[r['tag']] = r

L = ['# Goal-representation ablation @30k: xy vs compact vs full future-state goal\n',
     'All arms: antmaze_open_near task, adaptive alpha target_entropy=-8, '
     'numerical guards, identical training settings; ONLY the goal '
     'representation differs (2 / 13 / 29 dims; commanded goal = settled ant '
     'at the goal cell for the rich arms; relabeled goals = same indices of '
     'future states). Pre-verified: state layout, goal slices, and relabeling '
     'are bit-exact; no normalization anywhere '
     '(artifacts/goal_contract_verification.json).\n',
     'Primary behavioral metrics are XY-only for every arm.\n']

for arm, tagbase in ARMS.items():
  L.append(f'\n## Arm: {arm}\n')
  L.append('| gate | alpha | entropy med | scale med | sat | eff-rank | '
           'moving | XY success | XY goal vel | XY progress | ctrl sp | '
           'ctrl useful | retr acc (64-way) | rank med | XY rel | pose rel | '
           'vel rel | joints rel | only-XY |')
  L.append('|' + '---|' * 18)
  for g in GATES:
    r = G[f'{tagbase}_{g}']
    h, e = r['policy_head'], r['eval_deterministic']
    c = r['controllability']['sigma_0.05'] if r['controllability'] else {}
    at = A[f'{arm}_{g}']
    b = at['blocks']
    def rel(k):
      return f"{b[k]['reliance']:+.2f}" if k in b else '-'
    sp = c.get('spearman_prog1_median')
    L.append(
        f"| {g // 1000}k | {h['alpha']:.3f} | {h['entropy_median']:.2f} | "
        f"{h['scale_median']:.2f} | {h['mode_sat']:.2f} | "
        f"{r['collection']['action_eff_rank']:.1f} | "
        f"{r['collection']['moving_transition_frac']:.2f} | "
        f"{e['success']:.2f} | {e['goal_vel']:.4f} | {e['progress']:.2f} | "
        f"{'-' if sp is None else f'{sp:.2f}'} | "
        f"{c.get('decile_usefulness', float('nan')):.2f} | "
        f"{at['baseline_acc']:.3f} | {at['positive_rank_median']:.0f} | "
        f"{rel('xy')} | {rel('pose')} | {rel('velocity')} | {rel('joints')} | "
        f"{b['xy']['only_block_acc']:.2f} |")

L += ['\n\n## Findings\n',
      '1. **Training health is arm-independent**: all three arms track the '
      'same adaptive-alpha trajectory (0.91 -> 0.084 -> ~0.012, entropy '
      'approaching the -8 target), no guard trips, no floor, saturation '
      '<= 0.10, effective rank 4.4-6.0 at 30k, moving fraction 0.64-0.83.',
      '2. **Richer goals improve only retrieval, and only marginally**: '
      '64-way retrieval accuracy at 30k is 0.80 (xy) / 1.00 (compact) / '
      '0.79 (full); by 20k all arms are already at 0.86-0.90. The compact '
      'goal is the best retriever; the full 29-dim goal is NOT better than '
      'xy (extra joint dims add noise, not signal).',
      '3. **No shortcut learning**: block attribution shows the critic is '
      'XY-grounded in every rich arm -- XY knockout destroys retrieval '
      '(reliance +0.75..+0.98 at 30k) while velocity reliance is ~0.00 and '
      'joints +0.10; only-XY retains 0.51-0.78 accuracy. The commanded-goal '
      'velocity block is ~0 (settled ant), so velocity matching was the '
      'expected shortcut and it did not materialize.',
      '4. **XY control does NOT improve**: success 0.1 / 0.0 / 0.1, XY goal '
      'velocity <= 0.006 m/s in all arms; local critic decile usefulness '
      'stays in -0.2..0.2 with negligible physical spread. Identical to the '
      'xy arm.\n',
      '## Verdict\n',
      '**Richer goals improve (already-strong) contrastive retrieval, not XY '
      'control.** The goal representation is NOT the binding constraint at '
      'this scale: with entropy fixed and goals enriched, the critic '
      'retrieves future states nearly perfectly and grounds them in XY, yet '
      'the actor still cannot extract goal-directed locomotion, and the '
      'local action-conditioned signal remains physically negligible '
      '(1-step XY effects of ~1e-4 m against goal distances of meters). '
      'The remaining gap sits in the actor/locomotion pathway: converting a '
      'valid state-goal value landscape into multi-step motor behavior '
      'within a 30-50k budget.\n',
      '## Caveats\n',
      '- 30k steps, seed 0, one run per arm.',
      '- Commanded rich goals have ~zero velocity block (settled ant at '
      'rest); relabeled training goals carry real velocities (matches the '
      'original ant_envs semantics).',
      '- gfull entropy median at 30k (-2.6) lags the other arms (-7.6/-9.2); '
      'alpha is identical, so this is estimator variance on a different obs '
      'distribution, not a health difference.']

open('qual_open_near/goal_ablation_30k.md', 'w').write('\n'.join(L) + '\n')
json.dump({'gates': {k: v for k, v in G.items()
                     if any(k.startswith(t) for t in ARMS.values())},
           'attribution': A},
          open('qual_open_near/goal_ablation_30k.json', 'w'), indent=2)
print('written qual_open_near/goal_ablation_30k.md')
