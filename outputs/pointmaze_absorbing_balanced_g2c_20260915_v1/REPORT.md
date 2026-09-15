# G2c: key-balanced actor batches — lower-route entry passes (0.36–0.40) but navigation collapses; the balancing upweights random-policy rows

> **STATUS (2026-09-15): retracted from the method path; retained as a negative ablation.** Global key-balanced actor batches change the behaviour prior at the fork from the benchmark's rare-detour distribution to a near-balanced one and amplify random-policy rows (80% of batch rows), which is why the O control also shifts down. The G2c' proposal (128/64/64 actor batches) is withdrawn for the same reason. See `outputs/pointmaze_absorbing_faithful_actor_audit_20260915_v1/REPORT.md`.

Date: 2026-09-15. Base G1 `outputs/pointmaze_absorbing_integration_20260915_v1` (frozen critics, plan, roots, canonical goal, actor keys; hash-pinned in [config.json](config.json)). Sealed [PROTOCOL.md](PROTOCOL.md); driver [run.py](run.py); [results.json](results.json); per-actor audits `actor_audit_{arm}_balanced_seed{k}.json`; balanced rows `balanced_rows_seed{k}.npz`; checkpoints, curves, per-root arrays. Strictly offline: 6 actor runs × 1,000 updates with the sealed `.5·bc + .5·critic` objective, zero critic updates, zero environment calls; 15,554,560 model-rollout transitions; 143 s.

## Decision

Primary passes, retention fails, O control is flagged: **not a G3 candidate.**

- P physical first-step entry into (1,2): **0.357 / 0.337 / 0.402** (row seeds 0/1/2; G1 uniform 0.202) — hard pass at every seed, strong pass at one.
- Retention: away landing-bin agreement 0.798 (≥ 0.709 ✓), away-state A3 reach 0.659 (≥ 0.630 ✓), but **A1 fork reach 0.429** against the required 0.792 (G1 uniform 0.880) ✗. Under the pessimistic model P's absorbed rate falls 0.673 → 0.502 and lower-route entry rises 0.224 → 0.545, yet reach does not improve (0.232 → 0.224): the actor detours and then does not complete the detour.
- O control: O's physical (1,2) entry rises to 0.254 / 0.319 / 0.309 (uniform 0.174) with no critic support (its BC/critic gradient cosine is −0.05: the O critic term is nearly inert on these batches). The P − O contrast is +0.103 [0.096, 0.111] in first-step entry and +0.203 in A3 lower-route entry, so the critic contributes, but most of the shift is the sampling itself.

## Why: the keys are dominated by random-policy behaviour

Key-balancing over the 134 (current cell, landing cell) keys with ≥ 10 rows makes each key 0.75% of a batch. Audit with the dataset's behaviour-mode field (used here for diagnosis only, never for training):

| | uniform plan rows | key-balanced rows (seed 0) |
|---|---:|---:|
| share from random-policy episodes | 0.183 | **0.802** |
| fork-cell rows | 3.6% | 6.8% |
| fork → (1,2) rows | 0.26% | 0.76% |
| fork → (2,3) rows | 1.78% | 0.74% |
| fork rows from random episodes | | 0.867 |
| fork → (1,2) rows from random episodes | | 0.707 |

Of the 134 kept keys, 96 are random-dominated (≥ 80% random rows), 20 mixed, 18 teacher-dominated. Uniform over keys therefore means 80% random-policy imitation. In the discrete WindyCorridor data there was no random exploration (the FAR expert is deterministic), so (s, a) keys were all expert route steps and balancing only equalised route frequencies; here the same rule mostly equalises wall bumps and wandering (the fork cell alone has 9 keys, 7 of them wander/wall keys). The consequences are visible everywhere: away BC-NLL change +0.83 (uniform +0.03), mode-action L2 change 0.27 (0.08), policy scale 0.77 (0.63), A1 reach 0.43 (0.88) for P and 0.67 (0.99) for O. The discrete SUMMARY's own warning applies — "a random exploration set is poison for pessimistic methods".

## Tables

Fork roots (656 × 128 sealed innovations, canonical goal):

| actor | phys (1,2) | phys (2,3) | naive (1,2) | mask down | mean action (x, y) | Q at samples | A3 reach | A3 absorbed | A3 lower route | A1 reach | A1 lower route |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| O uniform (G1) | 0.174 | 0.731 | 0.029 | 0.072 | (0.836, −0.310) | −6.85 | 0.249 | 0.756 | 0.097 | 0.987 | 0.099 |
| P uniform (G1) | 0.202 | 0.654 | 0.042 | 0.113 | (0.732, −0.268) | −6.70 | 0.232 | 0.673 | 0.224 | 0.880 | 0.233 |
| O balanced s0 | 0.254 | 0.633 | 0.046 | 0.165 | (0.773, −0.407) | −7.04 | 0.332 | 0.684 | 0.341 | 0.668 | 0.418 |
| **P balanced s0** | **0.357** | 0.484 | 0.064 | 0.281 | (0.638, −0.516) | −6.58 | 0.224 | **0.502** | **0.545** | **0.429** | 0.615 |
| O balanced s1 | 0.319 | 0.582 | 0.043 | 0.196 | | −7.10 | 0.208 | 0.814 | 0.187 | 0.710 | 0.221 |
| P balanced s1 | 0.337 | 0.505 | 0.063 | 0.257 | | −6.59 | 0.172 | 0.621 | 0.416 | 0.481 | 0.474 |
| O balanced s2 | 0.309 | 0.597 | 0.044 | 0.198 | | −7.06 | 0.298 | 0.719 | 0.289 | 0.710 | 0.351 |
| P balanced s2 | 0.402 | 0.474 | 0.070 | 0.364 | | −6.57 | 0.244 | 0.532 | 0.514 | 0.459 | 0.607 |

(The physical landing resolves the ten-substep X-then-Y kinematics with wall rejection; the naive `floor(s + a)` bin and the direction mask are kept for continuity.) Paired over roots, P balanced s0 vs P uniform: physical (1,2) +0.155 [0.146, 0.164]; A3 lower route +0.320; A3 absorbed −0.171; A3 reach −0.008 [−0.016, −0.000]. P − O at s0: A3 absorbed −0.181, A3 lower route +0.203, **A3 reach −0.108** — P avoids the swamp more than O and reaches less.

Retention (1,024 away rows):

| actor | naive bin agreement | physical agreement | BC-NLL change vs initial | mode L2 change | scale | saturation | A3 reach from away | A3 absorbed |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| P uniform (G1) | 0.834 | 0.882 | +0.03 [−0.01, +0.08] | 0.080 | 0.634 | 0.173 | 0.680 | 0.256 |
| P balanced s0 | 0.798 | 0.852 | +0.83 [+0.72, +0.93] | 0.274 | 0.770 | 0.070 | 0.659 | 0.296 |
| O balanced s0 | 0.790 | 0.852 | +0.74 [+0.64, +0.83] | 0.218 | 0.743 | 0.070 | 0.686 | 0.300 |

Gradient cosine at the final actors on the sealed batches: P −0.63 (uniform −0.86), O −0.05 (uniform −0.53).

## What the three G2 rounds establish together

| | uniform, bc 0.5 (G1) | bc 0.05 (G2) | AWR β=5 (G2b) | key-balanced (G2c) |
|---|---:|---:|---:|---:|
| P physical (1,2) | 0.20 | — (naive 0.10) | 0.555 | 0.36 |
| P A3 lower route | 0.22 | 0.24 | 0.42 | 0.55 |
| P A3 absorbed | 0.67 | 0.77 | 0.52 | 0.50 |
| P A3 reach | 0.23 | 0.16 | 0.37 | 0.22 |
| P A1 reach | 0.88 | 0.82 | 0.84 | 0.43 |
| away agreement | 0.83 | 0.53 | 0.83 | 0.80 |
| in the paper? | yes | yes | no | sampling only |

The G1 P critic's preference is real and every actor change that increases its influence at the fork moves the policy down. Lowering BC does it by leaving the data (saturation), AWR by reweighting rows, key-balancing by changing the data mix — and only the last is paper-compatible, but its continuous-data version imports the random-policy junk.

## Recommendation

Keep the discrete fix but apply it where the rare route is decided, not to every key: **G2c′ — decision-cell stratified actor batches** with exactly the composition the sealed G1 critic stage already uses (128 ordinary rows uniform + 64 fork→(1,2) + 64 fork→(2,3), the same candidate strata and split), the same objective, critics, keys and budget, O/P symmetric, three row seeds. This equalises the two fork decisions without upweighting the 96 random-dominated keys; the ordinary 128 rows keep the 18% random share of uniform replay, so corridor-following should stay near the G1 level. Pass line unchanged: P physical (1,2) ≥ 0.30 with A1 fork reach ≥ 0.792 and the other retention checks; the multi-step A3 lower-route and reach are the numbers that matter for G3. If that also fails retention, the remaining paper-compatible lever is the critic stage (G1′ with the same stratification already there, plus more updates), not the actor objective. Nothing here was committed or pushed.
