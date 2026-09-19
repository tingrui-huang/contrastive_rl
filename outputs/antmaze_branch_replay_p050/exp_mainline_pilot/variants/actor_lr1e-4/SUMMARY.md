# Variant actor_lr1e-4 (actor learning rate 3e-4 -> 1e-4, everything else as the pilot): walking recovered, route gain mostly lost -- not a fix

Pre-registered in `manifest.json` (sealed 2026-09-19 09:23, before training)
after the trajectory diagnostic showed the CF update losing local walking
competence the pre-update policy still has (diag_traj sections 3 and 7).
Only the actor learning rate changed; data, anchors, CF branches, batch
streams and seeds, losses (bc 0.05), critic learning rate, initialisation,
30,000 optimizer updates, gamma and the evaluation are the pilot's.  O and
CF both, three paired seeds (seed 0 on node 30043, 1 on 30125, 2 on 30016;
both arms of a seed on one GPU).  Results: `REPORT.md`, `results.json`,
`../../diag_traj/cont_variant_actor_lr1e-4.json`; checkpoints on the nodes.

## Result on the same 300 evaluation episodes (mode policy)

| policy | success | detour | death | timeout | success no hazard | success hazard |
|---|---:|---:|---:|---:|---:|---:|
| start | 0.243 | 0.003 | 0.740 | 0.017 | 0.986 | 0.000 |
| O base s0 / s1 / s2 | 0.250 / 0.263 / 0.250 | 0.02 / 0.03 / 0.00 | 0.71 / 0.72 / 0.74 | 0.04 / 0.01 / 0.01 | 0.97 / 0.99 / 1.00 | 0.01 / 0.03 / 0.00 |
| CF base s0 / s1 / s2 | 0.267 / 0.320 / 0.480 | 0.11 / 0.17 / 0.50 | 0.63 / 0.57 / 0.26 | 0.11 / 0.11 / 0.26 | 0.84 / 0.89 / 0.69 | 0.08 / 0.13 / 0.41 |
| O lr1e-4 s0 / s1 / s2 | 0.247 / 0.250 / 0.247 | 0.02 / 0.03 / 0.03 | 0.71 / 0.71 / 0.73 | 0.04 / 0.04 / 0.02 | 0.96 / 0.97 / 0.96 | 0.01 / 0.01 / 0.01 |
| CF lr1e-4 s0 / s1 / s2 | 0.270 / 0.297 / 0.297 | 0.05 / 0.09 / 0.06 | 0.71 / 0.67 / 0.69 | 0.02 / 0.03 / 0.01 | 0.99 / 0.96 / 1.00 | 0.04 / 0.08 / 0.07 |

Paired on the common episodes (per seed; seed mean, seed s.e.):

| comparison | success | detour | death | timeout |
|---|---|---|---|---|
| CF(lr1e-4) - CF(base) | +0.003 / -0.023 / -0.183; **-0.068 (0.058)**, 2/3 negative | -0.057 / -0.073 / -0.437; **-0.189 (0.124)**, 3/3 | +0.087 / +0.103 / +0.433; +0.208, 3/3 | -0.090 / -0.080 / -0.250; **-0.140 (0.055)**, 3/3 |
| CF(lr1e-4) - O(lr1e-4) | +0.023 / +0.047 / +0.050; +0.040 (0.008), 3/3, rule met | +0.030 / +0.067 / +0.037; +0.044 (0.011), 3/3 | -0.026 | -0.014 |
| CF(lr1e-4) - start | +0.027 / +0.053 / +0.053; +0.044 (0.009), 3/3, rule met | +0.066 (0.013), 3/3 | -0.048 (0.013), 3/3 | +0.003 |
| O(lr1e-4) - O(base) | -0.007 (0.003) | +0.006 | -0.007 | +0.013 |

## Criteria (pre-registered)

1. Native success CF(lr1e-4) - CF(base): -0.068 (seed s.e. 0.058, 2/3
   negative) -> **NOT MET**.  (CF(lr1e-4) is still above O(lr1e-4) and the
   start by +0.04, 3/3, > 2 seed s.e. -- the small, consistent version of
   the pilot's effect.)
2. Detour gain kept: CF(lr1e-4) - CF(base) detour -0.057 / -0.073 / -0.437,
   3/3 negative, mean -0.189.  The sealed rule ("not below base by more than
   2 seed s.e., and still above O(variant)") returns KEPT only because the
   seed s.e. (0.124) is inflated by seed 2; read plainly, **two thirds of the
   detour gain is lost** (0.26 -> 0.07 mean; seed 2 0.50 -> 0.06).  The
   residual +0.044 over O(lr1e-4) and +0.066 over the start are 3/3.
3. Continuation from the SAME detour-entrance handover states (diag_traj
   section 3, base CF's own prefix): CF(lr1e-4) reaches 0.75 / 0.67 / 0.71
   vs base CF 0.75 / 0.42 / 0.48, start 0.50 / 0.58 / 0.52, O base 0.50 /
   0.58 / 0.71 -> **RECOVERED** (>= start and > base CF on seeds 1 / 2; seed
   2 equals O's 0.71, n = 42).  From the pre-stall states 0.46 / 0.33 / 0.42
   vs base 0.50 / 0.24 / 0.45 -- unchanged (those states are bad for every
   policy).  Native timeouts fall from 0.11-0.26 to 0.01-0.03 and no-hazard
   success returns to 0.96-1.00.

## Reading (against the readings fixed in the manifest)

The variant lands closest to "walks well but no detours = not a fix": the
execution regression is gone (criterion 3, timeouts, no-hazard success) but
the route change that produced the pilot's gain shrinks to a third
(criterion 2 read plainly), and native success falls (criterion 1).  The
slower actor moves less from the start agent in the same 30,000 updates,
and BOTH the route change and the walking loss shrink with it: in this
trial they scale together with the size of the update rather than
separating.  That is direct evidence against "the update speed" as the
lever: the two opposite needs (change the reset torques, keep the mid-route
torques) are not separated by the actor learning rate.

What this does not test: a longer run at the lower rate (a different
update budget -- not pre-registered, not run), or any change to the critic
(the seed-2 scoring bias off the actor's manifold is untouched).  No
further variant was launched.
