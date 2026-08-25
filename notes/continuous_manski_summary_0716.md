# Continuous Manski port: full summary (2026-07-16, R7 update 07-17)

## R7 headline result (windy env = the user's original design; original data recipe; AWR extraction; final; 3 seeds)

| arm | worst-case (all_active) | natural | safe-route share |
|---|---|---|---|
| baseline (d_pi + AWR) | 0.00 / 0.00 / 0.00 | 0.70 / 0.70 / 0.70 | 0% |
| **causal (d_lb + AWR)** | **1.00 / 0.98 / 0.30** | **1.00 / 1.00 / 0.85** | 100 / 99 / 40% |

Mean worst-case: 0.76 +/- 0.33 vs 0.00 (discrete WindyCorridor was 0.44 -> 0.56).
Baseline natural 0.70 = shortcut survival rate (~0.9^3); 3/3 seeds verdict CONFOUNDED.
Critic fork margin: **every causal seed is +2.6** (including the partially failed s2) --
the critic layer is 9/9 correct, and 100% of the residual variance sits in the AWR actor
(s2 falls into a 60/40 mixed mode).

### The full story of the extraction layer (findings 7-8)
1. (Q-max + BC) tug-of-war: in 6/6 seeds BC dominates the critic -> all take the shortcut.
2. Critic-greedy: route ordering is correct but local action resolution is insufficient,
   OOD overestimation -> aimless wandering (a continuous reproduction of the discrete
   SUMMARY item 6, "greedy-critic just spins").
3. **AWR (beta=0.5, cloning dataset actions only) = the answer the discrete version had
   already given**, 2/3 clean + 1/3 partial. Three porting-fidelity lessons: the N(s,x)
   reachable set, env resampling semantics, and the AWR actor -- every time the root cause
   was "failing to check the semantics against the discrete reference implementation".

Status: **R5 complete -- the method holds on the original benchmark, worst-case 0.00 -> 1.00.**
reachable-N seeds 1/2 reproduction in progress.

## R5 headline result (original dataset, zero changes to env or data, final checkpoint)

The decisive bug, found after the user pointed out that "backing up makes no sense":
my N(s,x) was an action-independent neighbouring-cell ball, whereas the true semantics of
the discrete `worst_case_kernel.py` is **the reachable set of THAT action under each value
of u** (enumerate the 5 wind conditions and take the argmin). After the fix
(`manski_reachable`): only actions that involve a swamp cell have a pessimistic branch
(worst = stuck/absorbing); every other action has a singleton reachable set, zero
pessimism, and no backing up -- wherever u cannot reach, the bound tightens automatically
(d_lb ~= d_pi).

| method | all_clear | all_active (worst) | natural | gap | verdict |
|---|---|---|---|---|---|
| baseline final | 1.00 | **0.00** | 0.73 | +1.00 | CONFOUNDED_SHORTCUT_BIAS |
| **causal (reachable-N) final** | **1.00** | **1.00** | **1.00** | **0.00** | NO_CLEAR_BIAS |
| always-safe oracle | 1.00 | 1.00 | 1.00 | 0.00 | -- |

The causal arm takes the safe route 100% of the time and **coincides exactly with the
oracle**; even from forced fork starts it detours 100% of the time (holding start 0.73 --
already past the fork, so it goes straight through; same footnote as before).
MC prescreen: on the original dataset under reachable-N, d_lb(safe) = 0.151 vs
d_lb(shortcut) = 0.059 (2.6x). **All three dataset re-collections were paying compensation
for the unjustified tax imposed by ball-N, and none of them was necessary.**

### WARNING -- multi-seed update (honest downgrade): the table above is 1/3 seeds

reachable-N s1/s2 collapse back to the shortcut at final; the natural curves of all three
seeds oscillate between a "detour mode" (~1.0) and a "shortcut mode" (~0.7), and s0's
perfect final simply stopped at a good moment. **Key partition diagnostic: the critic is
innocent** -- across every seed and every stage, the critic ranks f(safe) > f(short)
stably at the fork (+1.2 to +2.4 logits, never flipping from 40k onward, including the
failed s1). The oscillation comes entirely from actor extraction: the bc_coef=0.5 half of
the actor loss clones the behavioural mixture (~65% shortcut direction at the fork) and is
in permanent tug-of-war with the Q term. Countermeasure (in progress): bc_coef 0.5 -> 0.2,
applied to both arms, retrain 3 seeds x 2 arms.
The method-level conclusion is unchanged (the d_lb signal is correct and the critic learns
it stably); what remains open is the policy-extraction stability of a continuous Gaussian
actor -- this is finding 7: the "BC-critic tug-of-war", which does not exist for a discrete
argmax, is a real failure mode for a continuous actor.

## R4 (ball-N + hazard + safe40) downgraded to an ablation: seed-fragile

s0: worst 0.91 / natural 0.98; **s1 and s2 collapse back to the shortcut at final
(worst 0.00)** -- a 1.79x flip margin is not stable under training noise. A faithful
reachable-N is the pass/fail line, not a decoration. (During training s2's natural once
reached 1.0 and slid back by final -- a last-iterate instability worth recording.)

## 0. One-paragraph abstract

Port the causal contrastive RL validated on WindyCorridor (discrete) -- per-step Manski
lower-bound occupancy d_lb + reweighted NCE -- to a continuous 2D swamp PointMaze. The only
component of the training pipeline that changes is the source of the NCE positives (the
Thm-2 sampler); everything else is untouched. The first round of training failed (causal and
baseline learned the same shortcut-rushing policy); two mutually independent root causes were
diagnosed and fixed one at a time: (1) the dataset lacked coherent coverage of the safe route
and was polluted by random actions; (2) the worst-case design was weaker than the discrete
version (recoverable one-cell retreat vs instant lava death). With both fixed, the pre-training
MC prescreen shows d_lb preferring the safe route by 1.79x while d_pi still prefers the
shortcut -- the two arms are bound to diverge, and retraining is under way to confirm it.

## 1. Background and objective

- Existing result (526 slides): on MiniGrid WindyCorridor, causal contrastive raises
  forced-U worst-case success from 0.44 to 0.56 (oracle 0.92).
- Objective: the same method should hold with continuous state/action. Minimal continuous
  testbed = the self-built TwoRouteSwampMatchedEnv (continuous [x,y] + continuous 2D action,
  hidden swamp bits as the confounder u: u -> a (the teacher reads the bits to decide),
  u -> s' (active cells slow motion by x0.02), u not in obs).
- Narrative placement: PointMaze already has continuous state (previously misremembered as
  discrete); AntMaze/pixels are follow-on scaling, not a prerequisite for validating the method.

## 2. Continuous versions of the theoretical objects

| discrete object | continuous version | note |
|---|---|---|
| P(x\|s) tabular propensity | P_hat(bin\|cell): 8 direction sectors (rotated by half a width) + a stay bin, dataset counts + Laplace | continuous densities degenerate in the Manski decomposition (P(X=x\|s)=0 => the bound is vacuous), so neighbourhood-ing is necessary, not a convenience |
| N(s,x) neighbouring cells | itself + the 4 adjacent passable cell centres | covers the one-step reachable set under every u configuration (the only validity requirement) |
| argmin V | ordering by BFS distance to goal + **hazard**: the three swamp cells (static geometry) are V_lb = 0 absorbing | argmin only needs an ordering; hazard corresponds to the discrete lava -1e9, and V_lb = 0 is a valid lower bound |
| Thm-2 sampler | walk along the stored trajectory (empirical transition) (+) pessimistic teleport + re-anchor, T ~ Geom(1-gamma) | landing in a hazard terminates; p_override=1 => exactly degenerates to the no-pessimism walk (baseline arm) |
| Lipschitz assumption | Lemma 2' archived (notes/continuous_manski_lemma2prime.md) | two constants (action-side L_a, state-side L_s) enter the bound as explicit slack terms; not a blocker for experiments |

## 3. Implementation inventory

- `crl/manski.py`: binning / propensity / BFS / worst-neighbor (incl. hazard) /
  ManskiSampler.walk_from (vectorized) / ManskiPositiveBuffer (a frozen-buffer delegating
  wrapper that only overrides sample()) / build_positive_buffer.
- `scripts/fit_propensity.py`: gate G1 (beats uniform), G1b (no binning artifacts),
  G2 (decision-cell entropy higher than the downstream corridor), G3 (BFS graph valid),
  G4 (teleports point backwards).
- `scripts/manski_sampler_probe.py`: gate G5 (p_hat == 1 gives zero teleports and matches
  the replay law), G6 (teleports concentrate at the holding cell), G7 (endpoints really are
  more pessimistic), G8 (anchor-endpoint BFS correlation > 0.15, i.e. not vacuous).
- `scripts/manski_route_diagnosis.py`: **pre-training MC prescreen** (a few minutes) --
  ask the sampler directly for P(goal) of each action at the fork/holding cells; do not
  start training unless the d_lb ordering flips.
- `crl/config.py` +4 knobs; `crl/train.py` +13 lines (wrap the buffer after the audit passes).
- The rest of the training pipeline (NCE loss, networks, actor, bc, negatives, offline audit)
  is unchanged.

## 4. Experiment timeline and numbers

### R1: original dataset (safe 5%, random 20%), gamma = 0.95, matched env
- Training result: causal behaves identically to baseline (100% shortcut, worst-case 0.00,
  natural ~= 0.71/0.73), VERDICT: CONFOUNDED_SHORTCUT_BIAS (for both).
- MC diagnosis: d_lb(fork, shortcut) = 0.027 > d_lb(fork, safe) = 0.0092 (2.9x); even d_pi
  prefers the shortcut (0.59 vs 0.20). The critic faithfully learned d_lb -- the fault is in
  the signal itself.

### Diagnosis: why 5% coverage wins in discrete but not in continuous (checked against the causal-contrastive-rl code)
1. The discrete FAR expert is deterministic and has a **zero random set** => along the safe
   corridor P(a|s) = 1 per step, so 5% pays the entrance fee only once; our 20% random
   pollutes the propensity at every step (~0.2/step).
2. Wind bites at every step along NEAR (expert behaviour splits by wind at 6 lethal x
   positions); the swamp's u is visible only in the single holding cell (the teacher enters
   only when clear, so the corridor data is "sanitized" by selection bias).
3. Discrete V_lower: lava = -1e9 absorbing death; my worst-case = a recoverable one-cell
   retreat, and the re-anchor even rides the trajectory of a successful crosser => pessimism
   is toothless inside the confounded corridor.
4. Intrinsic to the continuous case: binning + noise leaks ~0.85/step, which penalizes long
   paths (the safe route).

### R2: data-side repair sweep (MC prescreen, no training)
| dataset | safe% | random% | d_lb shortcut | d_lb safe | flip? |
|---|---|---|---|---|---|
| original | 5 | 20 | 0.0270 | 0.0092 | NO (0.34x) |
| safe30 | 30 | 10 | 0.0302 | 0.0245 | NO (0.81x) |
| strong old teacher | ~19 | 20 | 0.0025 | 0.0008 | NO |
| safe40 (noise 0.1) | 40 | 5 | 0.1855 | 0.1727 | NO (0.93x) |
| strong-wait | 30 | 5 | 0.1435 | 0.1525 | YES (1.06x, too narrow) |

strong-wait training (150k) produced no detour policy (natural oscillating ~0.5) -- a 1.06x
flip does not survive NCE noise plus actor extraction.

### R3: worst-case repair (hazard V_lb; triggered by the user's challenge)
| dataset | no hazard | with hazard | conclusion |
|---|---|---|---|
| original (5%) | 0.34x | 0.44x | the coverage problem is independent; hazard cannot rescue it |
| strong-wait | 1.06x | 1.69x | the flip becomes solid |
| **safe40 + matched env** | 0.93x | **1.79x** (0.137 vs 0.077) | **best; no need to switch to a harsher env** |

Meanwhile d_pi still prefers the shortcut everywhere (safe40: 0.63 vs 0.53) => the baseline
pathology is preserved.

### An unexpected bonus about the baseline pathology (strong-wait baseline training curve)
- best@10k: detours, natural 0.88; final@150k: 100% shortcut, natural 0.52, worst 0.00.
  **The biased d_pi critic drags the policy from the BC behaviour toward the shortcut as
  training proceeds, and natural degrades monotonically** -- a direct illustration of
  "confounding actively hurts".
- Protocol lesson: **compare using final, not best** (selecting best by natural rollout
  leaks deconfounding model selection to the baseline).

### R4 (in progress): the final setting
- Env: matched (p = 0.1, the original benchmark, unmodified).
- Data: safe40 (force_safe 0.4, random 0.05, noise 0.1, 6000 episodes, frozen).
- Method: gamma = 0.95, hazard V_lb.
- Two arms: swamp_safe40_manski_s0 (causal) vs swamp_safe40_walkbase_s0 (p_override=1),
  150k steps, bc 0.5, seed 0, single variable = the pessimism switch.
- Pre-registered prediction: baseline final 100% shortcut (worst ~= 0, natural ~= 0.73);
  causal final takes the safe route (worst and natural -> 1.0, matching the always-safe oracle).

## 5. Six findings that can go into the slides

1. **Sector rotation**: putting bin boundaries on the dominant behavioural directions
   artificially halves the propensity -- an alignment problem specific to continuous
   propensity estimation that does not exist in the discrete case.
2. **Pessimism compounds with the horizon**: gamma = 0.99 (mean 100 steps) is vacuous for a
   12-step maze; gamma is the first-order knob for bound tightness (swept, chose 0.95).
3. **No coherent coverage, no certificate**: Manski can only recommend alternative routes
   that the behaviour policy actually walked properly; a random exploration set is poison for
   pessimistic methods (the discrete experiment satisfied this implicitly via zero randomness).
4. **Fidelity of the worst case**: a "soft" trap in a continuous env (recoverable slowdown)
   must be modelled explicitly as an absorbing state in V_lb (hazard), otherwise pessimism is
   toothless; a static-geometry hazard is exactly as legitimate as the discrete lava penalty.
5. **The geometry of confounding**: entry-choice confounding (swamp) gives Manski a single
   bite; path-distribution confounding (wind) gives it a bite at every step. The former is
   close to the worst case for per-step Manski -- bound strength is proportional to how
   visible the behaviour's u-dependence is in the action distribution.
6. **Offline model-selection leakage**: choosing the best checkpoint by environment rollout
   performs deconfounding on the baseline's behalf -- a strict comparison must use final or
   an offline criterion.

## 6. Open questions (thinking list)

- **Bootstrapping V_lb**: BFS + hazard is a static proxy; a formal version should switch to
  critic bootstrapping (spectral norm for Lipschitz, target network for stability), with
  hazard generalized to "low-value absorbing regions learned from V_lb".
- **The legitimacy narrative for hazard**: static geometry (known map) vs unknown
  environments -- the discrete version's docstring defence can be reused, but a reviewer may
  push for a learned V_lb.
- **Flip magnitude vs training survival**: 1.06x dies, 1.79x pending -- the critical magnitude
  in between deserves an ablation (it would explain both a success and a failure of R4).
- **Multi-seed**: if R4 succeeds, produce the formal table with 3 seeds x 2 arms.
- **Propensity conditioning**: currently cell-level; would finer conditioning (sub-cells or a
  continuous classifier) tighten the bound significantly?
- **Theory notes**: Lemma 2' (Lipschitz slack) + hazard-V_lb legitimacy + the "geometry of
  confounding" observation would make one formal discrete -> continuous discussion section.
- **Scaling**: once the method holds, the order and necessity of AntMaze (29-dim, second-order
  dynamics) and a pixel version.

## 7. Artifact index

- Data: datasets/swamp_matched_teacher_{s0,safe30_s0,safe40_s0}.npz,
  swamp_strong_waitteacher_s0.npz (all frozen + manifest).
- Tables/gates: artifacts/manski_port* (propensity_table.npz, propensity_report.json,
  probe reports, teleport heatmaps).
- Diagnosis: scripts/manski_route_diagnosis.py (--no_critic --hazard).
- Training: swamp_{manski,walkbase}_s0 (R1), swamp_strongwait_* (R2/3),
  swamp_safe40_* (R4, in progress).
- Evaluation: scripts/eval_swamp_matched_deployment.py (matched),
  eval_swamp_deployment.py (strong); compare using final.pkl.
- Theory archive: notes/continuous_manski_lemma2prime.md.
