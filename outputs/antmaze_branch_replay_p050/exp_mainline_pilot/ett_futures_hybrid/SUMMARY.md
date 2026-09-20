# The HYBRID control (user's decision after 694bc02, 2026-09-20): exact motion + the current learned risk -- does the policy gain come back?  NO (mean 0.341 vs oracle 0.449; the failure mode changes from stalls to shortcut deaths)

`scripts/exp_v6_hybrid_futures.py generate` (node3, 18 workers, 37 min) + `scripts/exp_v6_ett_futures.py --ett hybrid seal / train / evaluate / report`.
`manifest.json`, `generation_ett.json`, `REPORT.md` / `report.json`, per run `CF/seed_*/{train_manifest.json, eval_mean_s8909.json}`; the
diagnostics `table_compare.json`, `nce_future_dist.json`, `per_state_risk.json`, `paths_vs_sim.json`, `death_frame_check.json` (no training).
Oracle CF / O finals and the start agent re-used with their seed-8909 evaluations.  Same anchors (53,747), weights, context prior, future law
(reach / sampled onset / horizon 800 - t), continuation (logged torque once, then the frozen start agent's mode on the current state), CRL / actor /
BC 0.05 / critic clip 0.1 settings, five seeds, one draw (8909, 300 episodes, mode).

## Construction

Per anchor exactly the learned-ETT construction with ONE substitution: the motion is the SIMULATOR with both rockfalls forced inactive
(`env.reset(rockfall_active_1=False, rockfall_active_2=False)`, then the logged qpos / qvel restored; the simulator never kills, never
displaces).  The risk is the current learned one: the advice generator v3 (one hidden context per path from the prior, MAP hold, sampled
torque), the v4 onset head on (s, a_b, a_q, its own predicted delta, exact counters), sampled per step; the death frame is the simulator's
physics-only next state.  Nothing of the anchor's actual context is read.

## The hybrid table is the simulator table up to the death cut

* Outcome shares (anchor-weighted) 0.612 / 0.300 / 0.088 vs the simulator's 0.621 / 0.291 / 0.088, mean rows 142.5 vs 143.3; every region within
  0.03 (`generation_ett.json`).  Conditional on the sampled context P(death | u1) 0.367 vs 0.355, P(death | u2) 0.455 vs 0.453
  (`table_compare.json`).  Death-time KS 0.021 pooled (zone1 0.085, between 0.054, zone2 0.171 -- the learned risk kills earlier in zone 2,
  the same deviation the v4 futures have); path-length KS <= 0.05 in every region (v4: 0.12 pre-mouth, 0.17 west column).
* Row identity (`paths_vs_sim.json`): for every anchor whose SEALED branch had both hazards inactive (13,319, of which 5,639 paths of >= 100
  rows) the hybrid path is row-identical (|diff| <= 1e-4 on all 31 obs dims) to the sealed simulator branch -- the physics and the
  continuation are exact.  With an active hazard 30 % of the pairs differ, and (`death_frame_check.json`) of the 10,986 simulator-death
  anchors whose hybrid path is at least as long, 8,911 differ ONLY at the death frame (the rockfall's impact: max |dvel| 1.3 median /
  2.5 p90, xy 0.026 / 0.065, z -0.009), 2,071 diverge 1-8 rows before the kill (rock contact before the flag), and 608 non-death paths
  (1.1 %) are perturbed by a contact.  So the two tables differ in exactly three things: WHICH row the death cut falls on (the simulator's
  deterministic-in-context kill vs the per-step sampled learned onset), the death FRAME (impact state vs physics-only state), and the 1.1 %
  contact-perturbed survivors.
* Conditional risk at the state level (`per_state_risk.json`; anchors grouped by 1 x 1 xy cell and 40-step time bin, 222 groups with >= 8
  anchors): group death rates hybrid vs simulator corr 0.995, mean |diff| 0.012 -- inside the simulator's own split-half noise (corr 0.982,
  mean |diff| 0.016); success rates corr 0.993 (v4 0.88, rmse 0.116; top corridor 0.73 vs 0.83).
* The actual NCE positive distribution (`nce_future_dist.json`; 2 M draws of anchor by weight and m by the 0.999^m law): goal kind
  mid / success / death 0.969 / 0.019 / 0.012 vs 0.970 / 0.019 / 0.011; goal region within 0.003 everywhere; median m 31 vs 32; goal region
  by anchor region within 0.01.  Death goals sit in 'between' 0.093 vs 0.053 and zone2 0.366 vs 0.408; the death frame lies inside the
  hazard band 0.83-0.92 of the time vs the simulator's 0.89-0.95.

## Result (success / detour / death / timeout; seed 8909, the same 300 episodes)

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---|---|---|---|---|
| ETT, hybrid futures | 0.357 / 0.25 / 0.39 / 0.25 | 0.370 / 0.38 / 0.35 / 0.28 | 0.187 / 0.03 / 0.57 / 0.24 | 0.313 / 0.25 / 0.42 / 0.26 | **0.477** / 0.40 / 0.39 / 0.13 | 0.341 |
| ETT, v4 futures | 0.483 | 0.213 | 0.337 | 0.390 | 0.310 | 0.347 |
| ETT, v3 futures | 0.283 | 0.450 | 0.417 | 0.167 | 0.137 | 0.291 |
| CF (simulator futures) | 0.523 / 0.61 / 0.19 / 0.29 | 0.470 / 0.48 / 0.32 / 0.21 | 0.437 / 0.28 / 0.42 / 0.14 | 0.407 / 0.35 / 0.38 / 0.21 | 0.407 / 0.38 / 0.33 / 0.26 | 0.449 |
| O (recorded) | 0.297 | 0.287 | 0.290 | 0.300 | 0.310 | 0.297 |

Paired (success): ETT - O +0.060 / +0.083 / -0.103 / +0.013 / +0.167, mean +0.044, seed s.e. 0.044, 4 / 5 -- rule NOT met (v4 +0.050, v3
-0.006); ETT - CF -0.167 / -0.100 / -0.250 / -0.093 / +0.070, mean -0.108, s.e. 0.053, 4 / 5 below; retention of the oracle gain 0.29.

Routes (per 300): shortcut 0.56 / 0.56 / 0.91 / 0.65 / 0.56 (oracle 0.29-0.62, O 0.87-0.97), detour 0.25 / 0.38 / 0.03 / 0.25 / 0.40 (oracle
0.28-0.61).  On the shortcut with the zone-1 hazard active every arm dies 92-100 % of the time (nobody waits); the deaths follow the
shortcut share.  Timeouts 40-83 per 300 (oracle 43-86), split: entered the far route 4-41 (oracle 12-49), on the shortcut 8-62 (oracle
8-24; all past hazard 1, mostly stalled in the corridor before mouth 2), no classified route 7-41 (oracle 16-30; v4 futures 58-185).
Training metrics normal.

## Reading (the user's rule, stated before the run)

* NOT "recovers to near the oracle": 0.341 vs 0.449, 4 / 5 seeds below their oracle counterpart, the mean equal to the v4 futures' 0.347.
  So the motion error of the learned model is not what explains the v4 gap -- with the motion exact (row-identical paths) the policy is
  as far from the oracle as before.
* What the exact motion DID change: the failures before entering a route that the v4 futures produced (no-route timeouts 58-185 per 300)
  are at the oracle's level in 4 / 5 seeds (7-41) -- that part was the motion.  In its place the policies take the shortcut more often
  than the oracle's (0.56-0.91 vs 0.29-0.62) and die there, and stall in the corridor between the zones (shortcut timeouts 32-62 in three
  seeds vs 8-24).
* The user's two checks for this branch are done and both pass: the conditional risk matches the simulator's at the cell level within
  its own sampling noise, and the actual NCE positive distribution is the same to 0.003.  Therefore what remains is in the three
  differences listed above and nowhere else: the realisation of the death cut per path (sampled per step: earlier in zone 2, KS 0.17;
  deaths in the corridor 0.093 vs 0.053 of death goals; 8-17 % of death frames outside the band vs 5-11 %), the death frame without the
  impact signature, and the 1.1 % contact-perturbed survivors.  Which of these the critic reads is NOT established by this run.
* Not to do (user's rule): more motion-model training, multi-step supervision, BC tuning, the absorbing tail.
* Candidate for the user's decision: since the hybrid already IS "the sealed paths cut by the learned onset with a physics-only death
  frame", one table surgery discriminates the two remaining causes -- the sealed simulator paths with their OWN death rows but the death
  frame replaced by the physics-only state (8,911 frames available from the hybrid rows, 2,071 by one hazard-off step from the penultimate
  row; no simulation of the risk), trained with the same five-seed recipe: at the oracle's level -> the cut realisation is what the
  learned risk gets wrong; at the hybrid's level -> the impact frame is what the critic reads.  A frozen-critic readout of the oracle CF
  critic on impact vs physics-only death frames would come first (minutes, no training).
