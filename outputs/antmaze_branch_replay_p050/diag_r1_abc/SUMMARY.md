# Round-1 critic diagnostic A / B / C: summary

Reference commit 69e6572.  Frozen: the three round-1 critics (30k NCE on
`replay_policy_r1.npz`) and the continuation policy the replay was built
with (`joint_purebc/seed_0/final.pkl`); hashes in `manifest.json`.  Fresh
oracle outcomes (28,160 paths, 16 new paired draws per anchor, seeds
disjoint from every round-1 range).  Nothing trained, tuned or updated;
the pre-registered round-1 gate result stands (0/3).  Full tables:
`REPORT.md`; machine-readable: `metrics.json`; per-draw records:
`outcomes.npz`; the sealed design: `manifest.json`.

## Design

| layer | states | torques | anchors (episodes) | what changes vs A |
|---|---|---|---:|---|
| A | fit keys of the critics (5 strata x 32) | the original recorded torque + the two original policy samples | 160 (131) | nothing: only the consequences are fresh |
| B | the same anchors | three NEW samples from the frozen policy at that state (candidate seed ABC_SEED+17); same draw seeds as A | 160 (131) | the torque only |
| C | episode-held-out anchors (5 strata x 32; turning and north leg from 5 detour episodes) | their existing five candidates (recorded, mode, three samples) | 160 (67) | episode, and with it pose / goal / time |

C anchors were scored before (the round-1 gate, four draws); they remain
holdouts from critic fitting -- only their consequences are new here.  A
anchors were scored on their original two draws (`gate_policy_r1_v2_fit`).
Candidate torques in B sit where A's do (distance to the policy mode
0.52 vs 0.50, log pi 13.7 vs 13.9).  C's start-type torques are a little
closer to the mode (0.31 vs 0.45 at start_early); standardised-state
nearest-neighbour distance C->A 2.28 against A->A 2.49, so C is not far
from A in pose, but it differs in episode (and goal xy by ~0.3).

## Labels

Original records against fresh 16-draw outcomes at identical keys: on
the turning and north-leg strata the two agree (Spearman 0.82-0.90 per
key, 95 of the 106 pairs decided by both have the same sign); on the
three start-type strata the original two draws barely predict the fresh
mean (Spearman 0.12-0.36, mean |diff| 0.09-0.16), and most pairs there
are weak or tied on 16 draws (start_early A: 4 tie / 55 weak / 37 decided
of 96).  Sixteen draws decide 230 / 222 / 758 pairs in A / B / C; a
cross-fitted empirical selector (argmax on eight draws, scored on the
other eight) gains +0.061 / +0.055 / +0.078 in P_goal over the candidate
mean, so the outcomes carry usable selection signal in every layer
(most of it on the turning and north-leg strata: +0.09..+0.16).

## Result (region-integrated readout, deployed min; pooled dense strata; s.e. = episode bootstrap)

| layer | seed 0 | seed 1 | seed 2 | pick gain in P_goal (seed 0 / 1 / 2) | verdict |
|---|---|---|---|---|---|
| A: familiar state, familiar torque, fresh outcomes | **0.77** +- 0.031 (178 / 52) | **0.71** +- 0.033 | **0.70** +- 0.034 | +0.042 / +0.032 / +0.032 (s.e. 0.007) | succeeds, 3/3 |
| A, the two policy samples only | 0.77 +- 0.045 (73 pairs) | 0.71 | 0.67 | +0.025 / +0.020 / +0.016 | succeeds |
| B: familiar state, new torques | 0.55 +- 0.039 (121 / 101) | 0.51 +- 0.038 | 0.55 +- 0.032 | +0.013 / -0.000 / +0.005 (s.e. 0.006) | fails to inconclusive: at most a marginal residual |
| A x B cross pairs (one familiar, one new torque) | 0.60 +- 0.023 | 0.57 | 0.57 | - | above chance: the familiar key carries the information |
| C: held-out-episode states, existing candidates | 0.50 +- 0.021 (382 / 376) | 0.48 +- 0.027 | 0.49 +- 0.024 | -0.000 / +0.001 / -0.005 (s.e. 0.008) | fails, 3/3 |
| C, matched candidate types | 0.47 +- 0.039 | 0.48 | 0.46 | -0.006 / -0.008 / -0.003 | fails |

Layer A against the ORIGINAL labels at the same keys: 0.86 / 0.82 / 0.78;
against the FRESH labels: 0.77 / 0.71 / 0.70; of the pairs where the
critic matched the original label, 130 / 6 (seed 0) keep the sign on
fresh draws.  So the fit to the training keys is mostly a fit to their
expected consequences, not to the two realised futures: memorisation
(explanation A) accounts for at most the 0.86 -> 0.77 drop.  The critic's
pick at its own keys recovers +0.042 of the +0.061 an eight-draw
empirical selector gets.

Per stratum (seed 0 / 1 / 2, decided pairs): A start_early 0.76 / 0.76 /
0.76 (37), start_late 0.62 / 0.50 / 0.56 (32), shortcut_early 0.71 / 0.67
/ 0.62 (24), turn 0.86 / 0.74 / 0.75 (65), north_leg 0.79 / 0.76 / 0.69
(72).  B start_early 0.66 / 0.45 / 0.51 (47), start_late 0.57 / 0.52 /
0.65 (23), shortcut_early 0.52 / 0.59 / 0.72 (29), turn 0.47 / 0.49 /
0.40 (57), north_leg 0.53 / 0.55 / 0.59 (66).  C turn 0.49 / 0.44 / 0.48
(237), north_leg 0.50 / 0.52 / 0.51 (232), start strata 0.42-0.60 (88-104)
with no consistent sign across seeds.  Exact-goal and per-head readouts
tell the same story (A 0.66-0.79, B 0.48-0.58, C 0.47-0.53).

## Which boundary fails

The critic's value is specific to the (state, torque) keys it was
trained on.  At those keys it predicts fresh consequences (A succeeds,
including pairs of two policy samples, on 131 episodes); at the same
states it cannot order three new torques drawn from the same policy
distribution (B at chance to marginal, the A x B cross pairs above
chance only because one member of each pair is a trained key); at
states from other episodes it is at chance whether the candidate set is
the full five or the matched three (C fails on fresh outcomes as on the
original four draws).  Under the task's rules this is "A succeeds but B
fails": inadequate generalisation across torques at familiar states,
with the state/episode failure (C) consistent with the same limitation
-- a critic that does not interpolate across torques at a known pose has
nothing to carry to a new pose.  It is not the memorisation of realised
futures (A held up), and the label-noise reading of the earlier report
is not supported either (the fresh outcomes decide 222-758 pairs per
layer and a two-half empirical selector profits from them).

## What remains uncertain

- B's residual: seeds 0 and 2 at 0.55 (+- 0.035) leave a marginal
  above-chance signal open; seed 1 does not.  Sixteen draws decide 222
  pairs; this is not a precise zero.
- C's turning and north-leg strata rest on 5 held-out detour episodes;
  their start-type strata on 32 episodes each but with few decided pairs
  (88-104) and sign-inconsistent critics.
- Whether more torques per state would move B is not shown by this
  diagnostic; it is the intervention the pattern points at, not a result.
- Layer A's success at the start-type strata (0.62-0.76) comes with
  original labels that were nearly uninformative there (Spearman
  0.12-0.36), so what the critic learned at those keys came from more
  than the two draws (the surrounding rows of the same replay); this
  diagnostic does not identify the mechanism.

## Single next intervention best supported

A controlled comparison of TORQUE COVERAGE per state in the replay --
more policy-sampled candidates per anchor (with repeated draws), the
same anchors, the same continuation, the same NCE, 8-d torque, actor
objective and BC 0.05 -- read out with this same A / B / C design.  The
pattern (A succeeds, B fails, C fails) does not support "more episodes
alone" as the first step, and it does not support any change to the loss,
the readout or the architecture.  This is a diagnostic finding, not a
guarantee that denser torque coverage repairs the pipeline; the actor
stage stays unrun.
