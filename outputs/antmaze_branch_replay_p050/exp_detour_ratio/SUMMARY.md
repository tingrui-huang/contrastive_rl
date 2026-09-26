# Detour-demonstration share 5% vs 20%: summary

> **Status under `notes/MAINLINE_CONTRACT.md` (2026-09-19): historical diagnostic configuration.**  What this experiment compared was (i) actors initialised from a pure-BC walker, trained with a FROZEN 30k critic and displacement-balanced BC rows, whose critic-term batches were anchored at the dataset's RESET rows for the 'vanilla' arm (`row0_prepare` on the dataset copy) and at replay row 0 (arbitrary timesteps) for the 'branch' arm; (ii) branch critics trained on a replay with dense strata + every-60th anchors, the BC walker as continuation, goal hold to the horizon, two draws and sampled candidate torques.  The conclusion 'the counterfactual branch futures give a critic that is worse for the actor at both shares' is therefore a statement about THAT configuration and is superseded as a statement about the method (contract section 1c); the dataset-level facts (the 5 / 10 / 20 % share moves what recorded-future CRL learns under this recipe; the BC floor never detours) stand as facts about this recipe.  Nothing below was edited.

Question (user's proposal): does raising the share of detour demonstrations
from 5% to 20% let the pipeline work, and if so, which part -- the
demonstrations themselves (vanilla offline CRL improves too) or the
counterfactual branch futures (branch improves beyond vanilla)?  Sealed
manifest `manifest.json` (2026-09-18 11:41, before any training; dataset
hashes inside, verified against the files pulled locally); machine-readable
table `REPORT.md` (`scripts/exp_v6_detour_ratio.py report`).  d05 ran on
node 30021 (RTX 4090), d20 on node 30043 (RTX 4090 laptop); every
within-ratio comparison is same-GPU, same code, same seeds.

## Data

Nested 1,000-episode datasets drawn (seed 2028, no outcomes or scores used)
from the merged p050 pool of 6,000 episodes minus the old 100 held-out and
the 30 reserved Cnew detour episodes (pool: 246 detour / 5,622 shortcut):

| | d05 | d20 |
|---|---|---|
| episodes (detour / shortcut) | 50 / 950 | 200 / 800 (contains all of d05's 50 detour and 800 of its shortcut episodes) |
| rows | 267,481 | 286,438 |
| detour share: episodes / rows | 0.050 / 0.070 | 0.200 / 0.269 |

Same environment (p_active 0.5 / 0.5, horizon 800), same teacher, same
collector settings; the only intervention is which episodes are present.

## Recipe per ratio (identical; budgets fixed, no sweeps)

1. pure BC (bc 1.0, critic term off, 100k, balanced rows, seed 0) -> the
   ratio's BC walker;
2. BC controls: 300 natural draws (mean policy) for route choice and
   success; placed completion on the Cnew detour rows (north-leg and
   east-leg starts, 100 each, against the generation driver);
3. vanilla offline CRL critics x 3 seeds (30k, recorded futures of the
   dataset);
4. branch replay (r1 protocol: 1,500 dense anchors x {recorded, sample0,
   sample1} x 2 paired hazard draws + every-60th-row general anchors x
   {recorded, sample0} x 1 draw; continuation = this ratio's BC walker
   mode; no hold-out split) -> branch critics x 3 seeds (30k, uniform
   anchors);
5. actors: one per critic, initialised from the ratio's BC walker, frozen
   critic, bc 0.05, 30k; 300 natural draws each (seed 909, mean policy);
6. share audit at every level (`shares_{d05,d20}.json`).

## Effective detour share at every level (audit)

| level | d05 | d20 |
|---|---|---|
| dataset: episodes / rows | 0.050 / 0.070 | 0.200 / 0.269 |
| vanilla critic anchor law (episode-uniform) | 0.050 | 0.200 |
| balanced BC sampler draws (100 x 1,024; mean +- sd over batches) | 0.051 +- 0.007 | 0.201 +- 0.012 |
| branch replay: paths 18,704 / 19,378; mass on turn + north-leg strata (fixed by design) | 0.096 + 0.096 | 0.093 + 0.093 |
| branch replay: general anchors from detour episodes | 0.069 | 0.268 |
| branch replay: all paths from detour episodes | 0.235 | 0.353 |

Every level moved by the intended factor of four except the fixed dense
strata (by design) and hence the branch replay's total, which already
carried 0.235 at d05.

## BC controls (single seed)

| | d05 BC | d20 BC |
|---|---|---|
| 300 natural draws: success / failure / timeout | 0.257 / 0.743 / 0.000 | 0.260 / 0.730 / 0.010 |
| route from the start: detour / shortcut / none | 0 / 290 / 10 | 2 / 285 / 13 |
| success with a hazard active (223 episodes) | 0.000 | 0.004 |
| placed north-leg start: reach / timeout / death (driver 0.99 / 0.01 / 0) | 0.80 / 0.20 / 0 | 0.92 / 0.08 / 0 |
| placed east-leg start: reach / timeout / death (driver 0.99 / 0.01 / 0) | 0.84 / 0.16 / 0 | 0.91 / 0.09 / 0 |
| median steps to reach, placed north / east (driver 319 / 184) | 352 / 234 | 316 / 192 |

The BC floor at the fork is zero for both ratios: the mean policy takes the
shortcut even with 20% detour demonstrations (2 / 300).  What 20% buys
BC is execution: placed on the detour it completes it 0.91-0.92 instead
of 0.80-0.84, at the driver's pace.  Both BC walkers therefore can carry
a detour continuation in the replay but never choose it themselves.

## Branch replay profiles (recorded candidate; fit paths)

| stratum | d05 reach / death / went around | d20 reach / death / went around |
|---|---|---|
| start_early | 0.27 / 0.71 / 0.04 | 0.35 / 0.60 / 0.16 |
| turn | 0.57 / 0.05 / 0.68 | 0.71 / 0.02 / 0.77 |
| north_leg | 0.80 / 0.00 / 1.00 | 0.89 / 0.00 / 0.98 |
| start_late / shortcut_early | 0.26 / 0.74 / 0.00, 0.25 / 0.75 / 0.00 | (similar) |
| general | 0.60 / 0.33 / 0.06 | 0.66 / 0.27 / 0.22 |

With the d20 walker as continuation the start-state counterfactuals go
around four times as often (0.16 vs 0.04) and die less (0.60 vs 0.71):
the contrast the critic needs at the start exists in the d20 replay and
barely in the d05 one.  Generation 726 s (d05, 15 workers) / 993 s (d20,
19 workers, slower node).

## Actors (300 natural draws each, mean policy)

| ratio | critic | seed | success | timeout | route detour / shortcut / none | no-hazard success (detours) | hazard success (detours / 223) | zone-1 / zone-2 deaths | mean steps |
|---|---|---|---:|---:|---|---|---|---|---:|
| d05 | BC floor | 0 | 0.257 | 0.000 | 0 / 290 / 10 | 1.000 (0) | 0.000 (0) | 155 / 68 | 123 |
| d05 | vanilla | 0 | 0.263 | 0.063 | 20 / 260 / 20 | 0.883 (6) | 0.049 (14) | 137 / 65 | 184 |
| d05 | vanilla | 1 | 0.263 | 0.000 | 2 / 292 / 6 | 1.000 (0) | 0.009 (2) | 153 / 68 | 128 |
| d05 | vanilla | 2 | 0.260 | 0.007 | 4 / 288 / 8 | 0.987 (2) | 0.009 (2) | 155 / 65 | 133 |
| d05 | branch | 0 | 0.240 | 0.237 | 43 / 214 / 43 | 0.779 (13) | 0.054 (30) | 109 / 48 | 306 |
| d05 | branch | 1 | 0.240 | 0.090 | 6 / 266 / 28 | 0.896 (1) | 0.013 (5) | 139 / 62 | 189 |
| d05 | branch | 2 | 0.250 | 0.107 | 21 / 253 / 26 | 0.896 (4) | 0.027 (17) | 134 / 59 | 205 |
| d20 | BC floor | 0 | 0.260 | 0.010 | 2 / 285 / 13 | 1.000 (0) | 0.004 (2) | 152 / 67 | 132 |
| d20 | vanilla | 0 | 0.543 | 0.197 | 152 / 102 / 46 | 0.740 (40) | 0.475 (112) | 57 / 21 | 378 |
| d20 | vanilla | 1 | 0.723 | 0.180 | 221 / 41 / 38 | 0.883 (57) | 0.668 (164) | 24 / 5 | 419 |
| d20 | vanilla | 2 | 0.647 | 0.100 | 189 / 94 / 17 | 0.935 (54) | 0.547 (135) | 50 / 26 | 339 |
| d20 | branch | 0 | 0.243 | 0.240 | 48 / 198 / 54 | 0.779 (15) | 0.058 (33) | 107 / 48 | 303 |
| d20 | branch | 1 | 0.290 | 0.173 | 49 / 217 / 34 | 0.779 (11) | 0.121 (38) | 117 / 44 | 274 |
| d20 | branch | 2 | 0.347 | 0.163 | 90 / 187 / 23 | 0.792 (26) | 0.193 (64) | 101 / 46 | 291 |

Seed means (seed s.e.): d05 vanilla success 0.262 (0.001), detour 0.029
(0.019); d05 branch 0.243 (0.003), 0.078 (0.036); d20 vanilla 0.638
(0.052), 0.624 (0.066); d20 branch 0.293 (0.030), 0.208 (0.046).
"Route none" = no route completed (mostly timeouts wandering near the
fork, some early deaths); timeouts count as failures in success.

## Pre-registered comparisons (effect = > 2 x pooled seed s.e. and 3 / 3 seeds in the same direction)

| comparison | success | detour rate |
|---|---|---|
| vanilla d20 - d05 (demonstrations) | **+0.376** (s.e. 0.052; 3/3) | **+0.596** (0.069; 3/3) |
| branch d20 - d05 (demonstrations) | +0.050 (0.030; not > 2 s.e.; 3/3) | **+0.130** (0.058; 3/3) |
| d05 branch - vanilla (counterfactual futures) | **-0.019** (0.004; 3/3) | +0.049 (0.041; not > 2 s.e.; 3/3) |
| d20 branch - vanilla (counterfactual futures) | **-0.344** (0.060; 3/3) | **-0.417** (0.081; 3/3) |

Reading, in the terms set in advance:

- **The detour demonstrations matter.**  With 20% detour episodes,
  vanilla offline CRL alone takes the detour on 51-74% of natural draws
  and succeeds on 48-67% of hazard episodes, against 0% / 0% for its own
  BC initialisation and for every d05 run.  The effect is 3/3 seeds and
  seven times the pooled seed s.e.
- **The counterfactual branch futures add nothing here and cost a lot.**
  At d20 the branch actors reach 0.24-0.35 success (16-30% detours,
  hazard success 0.06-0.19) against vanilla's 0.54-0.72: -0.34 success,
  3/3 seeds, 5.7 x the pooled s.e.  At d05 the branch actors are below
  the BC floor (-0.019 vs vanilla, 3/3) while detouring slightly more
  (+0.05, within noise).  The branch method does respond to the
  demonstrations (detour rate +0.13, 3/3; success +0.05, 3/3 but within
  2 s.e.) -- the same direction as vanilla, one fifth of the size.
- Both branch families share a signature at either ratio: no-hazard
  success falls from 1.00 to 0.78-0.90 and 9-24% of episodes time out
  without completing a route, i.e. the branch critic pushes the actor off
  the BC behaviour even where the BC behaviour was already optimal.
  The d20 vanilla actors also lose some no-hazard episodes (0.74-0.94,
  by detouring when unnecessary and timing out), but gain far more on
  the hazard side.

**Verdict under the pre-registered rules:** the demonstrations improve
the recorded-futures method decisively (within this recipe; see the
next section for why this is not the ladder baseline); the branch method improves only weakly
with them and is far below the ordinary method at the same share.  The
"branch has an extra gain" reading is not supported; the opposite sign
is established at d20.

## What this does and does not show

- It shows that, at 20% detour share, an NCE critic on recorded
  futures is enough for the actor to learn the route choice at the fork
  **under this experiment's recipe**: p_active 0.50, gamma 0.999, a
  frozen 30k critic, the actor initialised from the BC walker and
  trained 30k at bc 0.05 with balanced BC rows.  This "vanilla" is NOT
  the frozen V6 baseline of the detour ladder (`notes/v6_detour_ladder.md`
  section 5: p_active 0.40, gamma 0.99, critic and actor trained jointly
  from scratch for 100k), which never took the detour even at the 0.30
  rung (mode policy 0.000 on three seeds; sampled at most 0.017).  Which
  of the four differences (hazard density, discount, critic schedule,
  actor initialisation / sampling) lets the demonstrations take effect
  is not isolated here; the discount is the a-priori candidate (at 0.99
  the detour's extra ~140 steps cost a factor ~4, at 0.999 ~1.15), but
  that is a prediction, not a result.  So the claim supported is narrow:
  the route choice is learnable from recorded detour futures at 20%
  share with this recipe -- not that the ladder baseline only lacked
  demonstrations.
- It shows that the branch replay as built (r1 protocol; ~19k query
  keys; every-60th-row general anchors; deterministic BC-mode
  continuation; uniform anchors; critic trained on the replay alone)
  produces a critic that is worse for the actor than the recorded-data
  critic at both shares.  It does not show why.  Untested candidate
  explanations, all post hoc: (i) supervision volume -- the branch critic
  sees 18.7k / 19.4k (s, a) query keys against 267k / 286k dataset rows,
  and the actor visits states the general stratum samples only every 60
  steps; (ii) the futures' source -- the vanilla positives at a
  detour-episode start are the teacher's recorded future (reach ~0.99),
  whereas the branch positives are the BC-mode continuation, which goes
  around from the start only 16% of the time at d20; (iii) the replay
  trains on the counterfactual futures only, discarding the recorded
  futures of the dataset.  Separating these needs a new comparison
  (e.g. branch replay plus the dataset rows, or general anchors at every
  row) and is a method change: not run.
- Limits: one BC walker per ratio (its seed is a shared input to all six
  actors of that ratio); three critic / actor seeds; 30k actor steps at
  bc 0.05 from the BC initialisation (an actor recipe chosen before, not
  tuned here); the two ratios ran on different GPUs (the cross-ratio
  vanilla effect, +0.38, is far outside any plausible hardware
  contribution; within-ratio comparisons are same-GPU); nested datasets
  share 50 detour and 800 shortcut episodes; no critic-level readout
  (A/B/C-style) was pre-registered for these critics, so the critic
  quality is inferred only through the actors; timeouts (up to 24%) are
  counted as failures.  Reserved validation seeds remain unused; nothing
  committed by this experiment beyond its own files (datasets pulled
  locally, hashes match the manifest; replays 0.84 / 0.89 GB and all
  checkpoints stay on the nodes).

Wall time: d05 61 min end to end (node 30021), d20 100 min (node 30043).
Files: `manifest.json`, `datasets.json`, `shares_{d05,d20}.json`,
`REPORT.md`; `joint_purebc_{r}/seed_0/eval_mean.json`,
`diag_walker_{r}/`, `critics_{van,br}_{r}/seed_*/branch_manifest.json`,
`joint_{van,br}_{r}/seed_*/{eval_mean.json,prep.json,branch_manifest.json}`,
logs `rt_{r}_*.log`, markers `MARK_RT_{r}_*`; datasets
`artifacts/rockfall_clock_v6/dataset/antmaze_rockfall_clock_v6_p050_{d05,d20}_{gxy,sidecar}.npz`.

## Follow-up (user's questions after the main result): density, discount, 10% share

Sealed `manifest_followup.json` (arms A, B before their runs; arm C added
before any d10 training).  Recorded-futures chain only (pure BC 100k ->
vanilla critics x3 -> actors x3 from the BC walker, 30k, bc 0.05 -> 300
draws, seed 909, mean policy); the d20 vanilla chain is the reference.
Table `REPORT_followup.md`.

| arm | dataset / p_active / gamma | BC walker success / detour | actor success (seeds 0 / 1 / 2) | detour rate | mean success (s.e.) | vs d20 vanilla, success |
|---|---|---|---|---|---|---|
| reference | d20 / 0.50 / 0.999 | 0.260 / 0.007 | 0.543 / 0.723 / 0.647 | 0.51 / 0.74 / 0.63 | 0.638 (0.052) | -- |
| A density | p040 far20 / 0.40 / 0.999 | 0.357 / 0.003 | 0.593 / 0.357 / 0.590 | 0.54 / 0.08 / 0.49 | 0.513 (0.078) | -0.124 (s.e. 0.094; not > 2 s.e.; 2/3) |
| B discount | d20 / 0.50 / 0.99 | (d20 BC reused) | 0.353 / 0.367 / 0.330 | 0.18 / 0.25 / 0.22 | 0.350 (0.011) | -0.288 (0.053; > 2 s.e.; 3/3) |
| C share 10% | d10 / 0.50 / 0.999 | 0.250 / 0.000 | 0.393 / 0.460 / 0.330 | 0.26 / 0.46 / 0.15 | 0.394 (0.038) | -0.243 (0.064; > 2 s.e.; 3/3) |
| (d05) | d05 / 0.50 / 0.999 | 0.257 / 0.000 | 0.263 / 0.263 / 0.260 | 0.07 / 0.01 / 0.01 | 0.262 (0.001) | -0.376 (0.052; > 2 s.e.; 3/3) |

Readings (rules in the manifest):

- **Density (A).**  On the ladder's own p040 far20 data this chain detours
  on two of three seeds (success 0.59, hazard success 0.51) against the
  ladder baseline's 0.357 / detour 0.000 on that density (even at the 0.30
  rung); the difference from p050 is within seed noise (2/3 seeds).  The
  hazard density is not what separates this chain from the frozen V6
  baseline; the seed that never learned the detour (seed 1: 0.08) shows
  the weaker, less stable signal at p040 that the ceiling arithmetic
  predicts (smaller target margin, same sign).
- **Discount (B).**  gamma 0.99 with everything else as in the d20 chain
  drops success from 0.64 to 0.35 and the detour rate from 0.62 to 0.22
  (3/3, > 2 s.e.), but not to zero: 0.35 / 0.22 is still above the BC
  floor (0.26 / 0.00) and above the ladder baseline (0.37 success at p040
  with 0 detours).  The discount is the largest single factor; the
  remaining gap to the ladder baseline comes from the other differences
  (frozen 30k critic vs joint 100k, BC-walker initialisation, balanced BC
  rows) -- not separated.  The pre-run prediction "A detours, B does not"
  was half right.
- **10% share (C).**  Intermediate and monotone: success 0.26 -> 0.39 ->
  0.64 and detour 0.03 -> 0.29 -> 0.62 across 5 / 10 / 20 %, with large
  seed spread at 10% (0.33-0.46 success, 0.15-0.46 detour).  The BC floor
  is zero detours at every share.

Caveats as before: one BC walker per arm, three seeds, arms on different
GPUs (A node 30021, B / C node 30043), the d10 draw nested with d05 / d20
(seed 2028), no critic-level readout.  Wall: A 61 min (BC 100k + chain), B
~25 min, C ~55 min.  Files: `manifest_followup.json`, `REPORT_followup.md`,
`joint_purebc_{p040far20,d10}`, `joint_van_{p040far20,d20_g099,d10}`,
`critics_van_{p040far20,d20_g099,d10}`, logs `fu_*.log`; dataset
`antmaze_rockfall_clock_v6_p050_d10_{gxy,sidecar}.npz` local (hashes in
`datasets.json`).  Driver: `V6_DISCOUNT` env override (default 0.999
unchanged; `vanilla_g099*` directories at 0.99).
