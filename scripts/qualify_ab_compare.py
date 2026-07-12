"""Assemble the 50k A/B comparison (alpha0 vs adaptive) from the gate reports."""
import json
import glob
import os

OUT_MD = 'qual_open_near/ab_50k_comparison.md'
OUT_JSON = 'qual_open_near/ab_50k_comparison.json'
ARM_GATES = {
    'alpha0': [10000, 20000, 30000, 40000, 50000],
    'adaptive': [10000, 20000, 30000, 40000, 50000],
    'adaptive_te8': [10000, 20000, 30000],
}
ARMS = list(ARM_GATES)

G = {}
for f in glob.glob('qual_open_near/gates/gate_*.json'):
  r = json.load(open(f))
  G[r['tag']] = r
mets = {a: json.load(open(f'qual_open_near/{a}_s0/metrics.json')) for a in ARMS}


def row(tag):
  r = G[tag]
  h, c, e = r['policy_head'], r['collection'], r['eval_deterministic']
  cv, ct = r['coverage_gates'], r['controllability']
  s5 = ct['sigma_0.05'] if ct else {}
  sp = s5.get('spearman_prog1_median')
  return [f"{h['scale_median']:.3g}", f"{h['frac_at_min_std']:.2f}",
          '-' if h['alpha'] is None else f"{h['alpha']:.3f}",
          f"{h['sample_vs_mode_dist']:.3f}", f"{h['mode_sat']:.2f}",
          f"{c['action_eff_rank']:.1f}", f"{c['action_per_dim_std_mean']:.3f}",
          f"{c['moving_transition_frac']:.2f}", f"{c['disp_p90']:.4f}",
          f"{c['torso_z_mean']:.2f}", f"{c['fall_step_frac']:.2f}",
          f"{cv['n_unique']}/{cv['n_episodes']}/{cv['n_goals']}/{cv['moving_frac_pool']:.2f}",
          'PASS' if cv['pass'] else 'FAIL',
          f"{s5.get('std_proj1_median', float('nan')):.2g}",
          f"{s5.get('rng_prog1_median', float('nan')):.2g}",
          '-' if sp is None else f'{sp:.2f}',
          f"{s5.get('critic_decile_gap_mean', float('nan')):.2g}",
          f"{s5.get('decile_usefulness', float('nan')):.2f}",
          f"{e['success']:.2f}", f"{e['progress']:.2f}", f"{e['goal_vel']:.3f}",
          f"{e['speed']:.4f}", f"{e['static_frac']:.2f}", f"{e['fall_frac']:.2f}"]


COLS = ['scale med', 'frac@floor', 'alpha', '|samp-mode|', 'mode sat',
        'act effrank', 'act dim std', 'moving frac', 'disp p90', 'torso z',
        'fall step', 'cov u/ep/goal/mov', 'cov gate', 'ctrl std proj1',
        'ctrl rng prog1', 'ctrl sp', 'critic dec gap', 'dec useful',
        'success', 'progress', 'goal vel', 'speed', 'static', 'fall(eval)']

L = ['# Near-goal/open-area Ant qualification: 50k A/B (alpha=0 vs adaptive alpha)\n',
     'Task: `antmaze_open_near` = AntMaze_Open-v5 (Gymnasium-Robotics physics, '
     '2D XY goal), start cell uniform, goal same/orthogonally-adjacent cell, d0 '
     'rejection-sampled to [1.0, 4.5] m, 300-step episodes. Config identical to '
     'the 150k umaze run (binary NCE, random_goals 0.5, min_replay/random 10k, '
     '4 sgd steps/step, batch 256, seed 0) except the arm variable:',
     '- **alpha0**: entropy_coefficient = 0.0 (faithful as-shipped baseline)',
     '- **adaptive**: entropy_coefficient = None, target_entropy = 0.0 (the '
     "original repo's adaptive semantics)\n",
     'Per gate: fresh sampled-policy collection (12 eps, proxy for replay '
     'additions), deterministic eval (10 eps, fixed seeds), coverage-gated '
     'fresh reference states, immediate controllability (1 step + 2 zero-action '
     'settle, sigma=0.05, 64 candidates, 30 states).\n']
for arm in ARMS:
  L.append(f'\n## Arm: {arm}\n')
  L.append('| gate | ' + ' | '.join(COLS) + ' |')
  L.append('|' + '---|' * (len(COLS) + 1))
  for g in ARM_GATES[arm]:
    L.append(f'| {g // 1000}k | ' + ' | '.join(row(f'{arm}_{g}')) + ' |')
  L.append('\ntraining evals: ' + '; '.join(
      f"step {e['step']}: sat={e.get('ant_action_saturation', 0):.2f}"
      + (f", alpha={e['alpha']:.3f}" if 'alpha' in e else '')
      + f", actor_loss={e['actor_loss']:.3g}, logits_gap={e['logits_gap']:.1f}"
      for e in mets[arm]))

L += ['\n\n## Findings\n',
      '1. **alpha0 reproduces the instant entropy collapse on the new task**: '
      'scale median 0.80 (init) -> 0.033 at the 10.2k gate (~300 gradient '
      'batches after warmup) -> at the 1e-6 actor_min_std floor from 20k on '
      '(75% -> 99% of dims). Action effective rank ~4/8, moving-transition '
      'fraction decays 0.38 -> 0.33, saturation ~0.35.',
      '2. **Adaptive alpha with target_entropy=0.0 delays but does not prevent '
      'collapse, then destabilizes**: diversity fully preserved through 20-30k '
      '(scale 0.54-0.86, eff-rank 5.5-8.0, moving 0.70-0.97, saturation ~0) '
      'while alpha anneals 0.91 -> 0.084 -> 0.014 (policy entropy sits above '
      'the 0-nat target, so alpha decays). Once alpha ~ 0.01, near-unregularized '
      'Q-maximization explodes the actor: actor_loss -3.8e23 at the 40.8k eval, '
      'saturation 1.0, action effective rank 0.0 (a single constant clipped '
      'action), moving fraction 0.023 -- strictly worse than alpha0.',
      '3. **Neither arm learns near-goal locomotion in 50k**: success 0.00 '
      '(adaptive) vs 0.10 (alpha0, deterministic eval at 50k, d0 in [1, 4.5] m); '
      'goal velocity <= 0.012 m/s; immediate controllability at sigma=0.05 stays '
      'physically negligible at every gate (std projected disp <= 6e-4 m, '
      'max-min goal progress <= 2.6e-3 m); critic decile usefulness <= 0.16.',
      '4. **Coverage gates PASS at every gate** (316-542 unique fresh states, '
      '12 episodes, 12 goals, moving fraction 0.33-0.97): the reference-state '
      'redundancy of the earlier probes is eliminated by construction.',
      '5. Critic-side learning is arm-independent (logits gap ~33-37 by 20-30k '
      'in both arms).\n',
      '## Decision per the pre-registered rule\n',
      'The continue-beyond-50k condition -- adaptive entropy preserves action '
      'diversity AND improves locomotion-related behavior relative to alpha=0 '
      '-- is **NOT met**: the adaptive arm preserved diversity only until ~30k, '
      'then collapsed to a constant saturated action. STOPPED at 50k; no third '
      'arm launched.\n',
      '## Interpretation (for the next decision, not acted on)\n',
      'target_entropy=0.0 (the original as-shipped adaptive semantics) is the '
      'proximate cause of the adaptive failure: 0 nats is far below the initial '
      'policy entropy, so alpha monotonically anneals toward 0 and the run '
      're-enters (and numerically overshoots) the alpha=0 pathology. The repo '
      'itself exports the SAC-standard heuristic target_entropy_from_env_spec '
      '= -num_actions = -8 that lp_contrastive never calls. target_entropy=-8 '
      '(or a fixed small positive alpha, or controlled collection noise) is the '
      'natural THIRD arm -- deferred per instructions.\n',
      '## Caveats\n',
      '- Documented, unchanged in this A/B: Gymnasium-Robotics gear-150 ant '
      '(vs d4rl ctrl+-30 gear-1 RK4 dt=0.1), 2D XY goal (vs original full-29D '
      'goal obs), 1 actor (vs 4), 50k budget.',
      '- Gate snapshots land at the eval AFTER the boundary '
      '(10.2k/20.4k/30.6k/40.8k/50.1k).',
      '- adaptive_40000 and adaptive_50000 rows are bit-identical: the policy '
      'stopped changing after the blowup (constant action + fixed eval seeds).',
      '\n## Third arm: adaptive_te8 (target_entropy = -8, guards on, stopped at 30k)\n',
      'Alpha-direction sanity (artifacts/alpha_direction_sanity): the implemented '
      'loss moves alpha the correct way in the healthy regime for both targets; '
      'the entropy ESTIMATE inverts under saturation (arctanh-clip artifact), '
      'which is the feedback loop that killed target=0; exploded policies NaN '
      'the alpha optimizer (hence the guards).',
      '\nResult: entropy median on collection obs +5.0 -> +4.2 -> **-7.56** (at '
      'the -8 target) while alpha annealed 0.91 -> 0.084 -> 0.011 and STOPPED '
      'at its designed equilibrium: no explosion, zero clip-artifact fraction, '
      'no guard trip, saturation 0.10, scale 0.50, action eff-rank 4.9, moving '
      'fraction 0.82 at 30k (vs alpha0 at 30k: scale at floor, moving 0.35; vs '
      'target=0 which exploded by 40k). Diversity is trending down '
      '(eff-rank 8.0 -> 7.6 -> 4.9) but the policy is nowhere near collapse.',
      '\nHowever, per the pre-registered conditional: local critic usefulness '
      'and goal-directed locomotion remain FLAT (sigma=0.05 spearman -0.17..'
      '0.19, decile usefulness -0.17..0.15, even sigma=0.2 usefulness ~0; '
      'success 0.1, goal velocity <= 0.006 m/s, deterministic progress '
      '<= 0.33 m). STOPPED at 30k. Recommended next: goal-representation '
      'ablation -- current 2D XY goal vs a richer Ant future-state goal '
      '(closer to the original 29D goal-obs formulation).']

os.makedirs('qual_open_near', exist_ok=True)
open(OUT_MD, 'w').write('\n'.join(L) + '\n')
json.dump({'gates': G, 'decision': 'STOP_AT_50K_CONDITION_NOT_MET'},
          open(OUT_JSON, 'w'), indent=2)
print('written', OUT_MD)
