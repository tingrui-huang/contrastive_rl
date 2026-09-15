# G3: native evaluation — the absorbing-ETT P critic makes the faithful bc = 0.2 actor take the lower route 0.60 vs 0.14 and raises native reach by +0.095 [0.020, 0.175]

Date: 2026-09-15. Sealed [PROTOCOL.md](PROTOCOL.md) (written before any environment step); driver [run.py](run.py); [checkpoint_manifest.json](checkpoint_manifest.json) (policy files with sha256, the critics they were trained against, the environment source hash, the 200 reset seeds and the innovation rule); [results.json](results.json); [per_episode.csv](per_episode.csv); [bootstrap_summary.json](bootstrap_summary.json); saved trajectories `native_{policy}.npz`. Post-run classification addendum: [reclassify.py](reclassify.py), [per_episode_regionfork.csv](per_episode_regionfork.csv), [bootstrap_summary_regionfork.json](bootstrap_summary_regionfork.json). 40,000 native steps, zero training updates, 11 s.

## Status statements

- **AWR (G2b) was withdrawn** from the method path (not in the CRL paper); it remains a diagnostic showing that support-constrained extraction can express the P critic preference.
- **G2c / G2c′ were retracted** as invalid main-path actor sampling: global key balancing changed the behaviour prior at the fork from the benchmark's rare-detour distribution and amplified random-policy trajectories.
- **G3 tests the faithful paper actor objective** `(1 − bc) E_π[Q] + bc log π(a_data | s, g)` with **bc = 0.2** on the original uniform actor rows, P versus O, with the bc = 0.5 pair (the G1 actors, byte-identical) as baseline. No reweighting, no key balancing, no new actor method; the G1 stratified critics are the critics; the uniform-anchor critic is a purity check only (endpoint +0.051, recorded in the faithful-actor audit).
- **The native environment is used** because the A1/A3 model rollouts under-estimate diagonal down-right movement: the diagonal model's projection reverts wall-endpoint moves to the current position instead of sliding (faithful-actor audit).
- **No commits** until this report is reviewed.

## Design as run

200 paired episodes; episode `i` uses `TwoRouteSwampWindyF4Env(seed = 9300000 + i)` for every policy (identical reset, actuator-noise and hidden-bit streams up to an absorbing death); the actor innovation at step `t` of episode `i` is `fold_in(PRNGKey(9400000 + t), i)` for every policy. Horizon 50, reward `1[dist < 2]`, discount 0.95 for the discounted return. All checkpoints are the saved G2 sweep actors (`actor_{P,O}_bc0p2.pkl`, `actor_{P,O}_bc0p5.pkl`; the bc 0.5 files are byte-identical to the G1 actors, verified in the manifest).

## Decision

**Under the sealed rule: provisional pass.** P_bc0.2 − O_bc0.2: reach **+0.095 [+0.020, +0.175]**, lower-route usage **+0.275 [+0.200, +0.350]**, absorbed **−0.220 [−0.295, −0.145]**; but the sealed stuck/other rate of P_bc0.2 is 0.180 > 0.10, so the strong pass is not met as sealed.

**The sealed stuck/other rate is a classification artefact, not stuck behaviour.** The sealed rule defined the fork visit as "the first step at which the agent ends in cell (1,3)". A down-right diagonal from the start cell (0.5, 3.5) passes through (1,3) and ends in (1,2) within one step, so the agent never *ends* a step in (1,3) and the sealed rule filed the episode as stuck/other. All 53 sealed stuck/other episodes across the four policies (36 / 2 / 8 / 7) reclassify as **lower route** when the fork is the start/fork region {(0,3), (1,3)}: those agents reached the lower corridor and 67% of P_bc0.2's reached the goal. Under that region definition, computed from the same saved trajectories with nothing re-run, P_bc0.2 stuck/other = 0.000 and the criteria give a **strong pass**: reach CI excludes 0, lower-route +0.455 [+0.390, +0.525] ≥ 0.15, no stuck behaviour. Both classifications are reported below; the region definition is the correct reading of the protocol's intent ("physically enters the lower-route side after the first fork") and is what the tables use unless marked "sealed".

## Native outcomes (200 paired episodes; region-fork classification; bootstrap over episodes)

| policy | reach | strict success | discounted return | undiscounted return | absorbed | lower route | shortcut | stuck/other | first departure → (1,2) | → (2,3) | hazard landings (alive) | time-to-goal \| reach | reach \| lower | reach \| shortcut | absorbed \| shortcut |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **P_bc0.2** | **0.495** | 0.485 | 5.16 [4.38, 5.93] | 19.0 | **0.385** | **0.595** | 0.405 | 0.000 | 0.495 | 0.505 | 1.31 | 11.6 | 0.64 | 0.28 | 0.72 |
| O_bc0.2 | 0.400 | 0.395 | 4.91 [4.07, 5.75] | 17.0 | 0.605 | 0.140 | 0.860 | 0.000 | 0.135 | 0.865 | 2.25 | 8.0 | 0.93 | 0.31 | 0.69 |
| P_bc0.5 | 0.435 | 0.430 | 4.15 [3.43, 4.88] | 15.8 | 0.550 | 0.305 | 0.695 | 0.000 | 0.235 | 0.765 | 2.00 | 14.2 | 0.77 | 0.29 | 0.71 |
| O_bc0.5 | 0.390 | 0.385 | 4.41 [3.64, 5.21] | 15.9 | 0.615 | 0.210 | 0.790 | 0.000 | 0.175 | 0.825 | 2.24 | 9.6 | 0.79 | 0.28 | 0.72 |

Sealed classification (fork = a step ending in (1,3)): P_bc0.2 lower 0.395 / shortcut 0.425 / stuck-other 0.180, first-fork landing (1,2) 0.255 / (2,3) 0.480 / none 0.180; O_bc0.2 0.120 / 0.870 / 0.010; P_bc0.5 0.250 / 0.710 / 0.040; O_bc0.5 0.170 / 0.795 / 0.035. Reach, return and absorbed are identical under both classifications.

Paired differences (mean, 95% CI over the 200 episode ids):

| comparison | reach | strict success | discounted return | undiscounted return | absorbed | lower route | shortcut | first departure (1,2) | hazard landings |
|---|---|---|---|---|---|---|---|---|---|
| **P_bc0.2 − O_bc0.2** (primary) | **+0.095 [+0.020, +0.175]** | +0.090 [+0.005, +0.170] | +0.24 [−0.65, +1.19] | +2.05 [−1.12, +5.35] | **−0.220 [−0.295, −0.145]** | **+0.455 [+0.390, +0.525]** | −0.455 [−0.530, −0.385] | +0.360 [+0.295, +0.420] | −0.93 [−1.17, −0.70] |
| P_bc0.5 − O_bc0.5 | +0.045 [−0.020, +0.110] | +0.045 [−0.020, +0.105] | −0.27 [−0.90, +0.36] | −0.14 [−2.42, +2.18] | −0.065 [−0.125, −0.005] | +0.095 [+0.050, +0.140] | −0.095 [−0.140, −0.050] | +0.060 [+0.020, +0.100] | −0.24 [−0.43, −0.07] |
| P_bc0.2 − P_bc0.5 | +0.060 [−0.020, +0.135] | +0.055 [−0.025, +0.130] | **+1.01 [+0.24, +1.82]** | +3.25 [+0.32, +6.25] | −0.165 [−0.240, −0.095] | +0.290 [+0.230, +0.355] | −0.290 [−0.355, −0.225] | +0.260 [+0.200, +0.325] | −0.68 [−0.93, −0.45] |
| O_bc0.2 − O_bc0.5 | +0.010 [−0.050, +0.065] | +0.010 [−0.045, +0.065] | +0.50 [−0.14, +1.15] | +1.06 [−1.25, +3.24] | −0.010 [−0.065, +0.045] | −0.070 [−0.110, −0.030] | +0.070 [+0.035, +0.110] | −0.040 [−0.070, −0.015] | +0.01 [−0.17, +0.17] |

Sealed-classification differences for the route metrics (P_bc0.2 − O_bc0.2): lower +0.275 [+0.200, +0.350], shortcut −0.445 [−0.520, −0.370], stuck/other +0.170 [+0.120, +0.225], first-fork landing (1,2) +0.145 [+0.070, +0.215].

## Reading

1. **The primary hypothesis holds natively.** With the same objective, the same bc, the same batches and keys, the same reset seeds and innovations, the only difference between P_bc0.2 and O_bc0.2 is the critic, and the P actor departs the start/fork region into (1,2) 0.495 vs 0.135, takes the lower route 0.595 vs 0.140, lands in hazard cells 1.31 vs 2.25 times per episode, is absorbed 0.385 vs 0.605, and reaches the goal 0.495 vs 0.400 (CI excludes 0). The bc = 0.5 pair shows the same direction at a smaller size (lower +0.095, absorbed −0.065, reach +0.045 n.s.), and lowering bc under the O critic moves the O actor *toward* the shortcut (lower −0.070), so the shift is the critic's, not the BC coefficient's.
2. **Discounted return is not separated** (+0.24 [−0.65, +1.19]): the detour reaches the goal later (time-to-goal 12.7 on the lower route vs 7.9 on the shortcut), so at γ = 0.95 the extra successes are discounted away; undiscounted return is +2.05 (n.s.) and P_bc0.2 − P_bc0.5 discounted is +1.01 [+0.24, +1.82].
3. **Failure case (1) applies partially.** P_bc0.2 reaches the goal on 0.64 of its lower-route episodes (absorbed 0.16 of them, mostly after wandering back to the corridor) against 0.93 for the 28 lower-route episodes of O_bc0.2 and 0.77 for P_bc0.5; its lower-route time-to-goal is 12.7 (O_bc0.2 9.4, P_bc0.5 18.7). The route decision transfers; lower-route continuation is the weaker part of the bc = 0.2 actor and the natural place for a reach gain that the 200 episodes cannot yet resolve beyond +0.095.
4. Failure cases (2), (3) and (4) do not apply: the lower-route shift is large and precise, there is no stuck behaviour under the correct classification, and P_bc0.2 differs from P_bc0.5 (lower +0.290, absorbed −0.165, discounted return +1.01).
5. Shortcut episodes are absorbed at 0.69–0.72 for every policy, consistent with a blind crossing of three cells at activation 0.30 (1 − 0.7³ = 0.66) plus lingering.

## Chain of evidence, G0 → G3

Offline: the absorbing Manski freeze channel flips the model's fork preference (G0, +4.28), the unchanged sigmoid-NCE critic learns it at matched anchors (G1, endpoint −0.088 → +1.071; sign survives without anchor stratification), and the paper actor with bc = 0.2 turns it into a physical first-step lower entry of 0.323 vs 0.131 (G2 re-audit). Native: 0.495 vs 0.135 first departure into (1,2), 0.595 vs 0.140 lower-route usage, −0.22 absorption, +0.095 reach. Everything on the P side reads only `obs`/`act`; no hidden bits, death labels, hazard labels or environment internals entered any training stage.

## Limits

One training seed for the critics and actors; 200 episodes resolve a reach difference of about ±0.08; bc = 0.2 was chosen from the G2 sweep after seeing offline (not native) numbers and is confirmed here on fresh seeds, but it is one coefficient, not a tuned optimum; the sealed route rule had the blind spot described above and was corrected post hoc with both readings reported; the freeze channel is an upper bound (offline right-route survival 0.18 vs native 0.28–0.31 reach on the shortcut), which is the intended pessimism, not a calibrated model.

## Next

G3b with bc = 0.1 (P and O), same seeds and rule set with the region-fork definition sealed in advance, to see whether the reach gain grows with the route shift or the continuation weakness dominates; and a second critic/actor training seed for the whole G1 → G3 chain. No commit until this report is reviewed.
