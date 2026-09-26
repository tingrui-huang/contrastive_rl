# Episode / state coverage experiment (NARROW vs BROAD): feasibility audit -- STOPPED before generation

Reference commits 355d273 (A/B/C diagnostic, torque-coverage arms) and
1966335 (supervised control, pretrain -> branch).  Question: does spreading
the counterfactual supervision over more INDEPENDENT original episodes
(separately reset episodes -- not new anchors, actions or hazard draws of
an existing episode) improve the critic on common held-out episodes,
especially turning and north-leg states?  Nothing was generated or trained.

## Data audited

- p050 dataset `antmaze_rockfall_clock_v6_p050_gxy.npz` (sha in
  `exp_r1_coverage/manifest.json`): 1,000 episodes, 268,112 transitions,
  teacher detour probability 0.05 -> 54 realised detour episodes, 946
  shortcut.
- Round-1 split (`holdout_policy_r1.npz` meta, `split_episodes` seed 0):
  100 held-out episodes (5 detour, 95 shortcut); training pool 900 (49
  detour, 851 shortcut).
- R replay provenance (`exp_r1_coverage/manifest.json`, `build_armR.json`,
  sha 193d3dde...): the 1,500 round-1 dense anchors (`pick_rows`, uniform
  over rows within a stratum, no episode stratification) + 4,378 general
  anchors (every 60th row of every training episode).

## Episode diversity by stratum (training pool = 900 episodes)

| stratum | training rows | training episodes | anchors in R | episodes represented in R | anchors per episode (min / median / max) | held-out rows | held-out episodes |
|---|---:|---:|---:|---:|---|---:|---:|
| start_early (x<2, y<2, t<=5) | 5,400 | 900 | 400 | 335 | 1 / 1 / 4 | 600 | 100 |
| start_late (shortcut eps, t>5) | 12,829 | 851 | 200 | 172 | 1 / 1 / 4 | 1,417 | 95 |
| shortcut_early | 22,868 | 851 | 300 | 252 | 1 / 1 / 3 | 2,529 | 95 |
| **turn (detour eps, x<2, y<2, t>5)** | 1,194 | **49** | 300 | **49** | 1 / 5 / 25 | 83 | **5** |
| **north_leg (detour eps, 2<=y<6)** | 1,781 | **49** | 300 | **49** | 2 / 6 / 11 | 178 | **5** |
| general (every 60th row) | 268k | 900 | 4,378 | 900 | ~5 | -- | 100 |

Adjacent frames of one episode are one trajectory; the counts above are
per separately reset episode.

## Findings

1. **Detour-episode diversity is exhausted.**  The turning and north-leg
   strata -- the primary endpoint -- draw on every one of the 49 training
   detour episodes already (300 anchors each, median 5-6 anchors per
   episode).  No BROAD arm can be built from p050 for these strata: there
   are no unused independent detour episodes, and the held-out side has
   5.  A NARROW arm could be made (fewer episodes, more anchors each) but
   it would only be compared against the exhausted pool, i.e. it would
   test "fewer than all" against "all", not "more independent episodes".
2. **Total (start-type) episode diversity is not exhausted** (335 / 172 /
   252 of 851-900 episodes used), so a NARROW vs BROAD contrast is
   constructible on start_early / start_late / shortcut_early.  It does
   not address the primary endpoint, and the held-out evaluation there has
   little power: on the sealed C set the start-type strata decide 104 / 97
   / 88 of 600 pairs at 16 draws (most pairs weak or tied; `diag_r1_abc`),
   the three critics were sign-inconsistent across seeds on them, and the
   start states are near the reset pose across episodes (little posture
   diversity to gain).  A 2-paired-s.e. pick-gain improvement on ~100
   decided pairs per stratum is not a realistic target.
3. **The far10-far30 datasets are p040**, not p050 (teacher detour
   probability 0.10-0.30: 111 / 165 / 216 / 263 / 306 detour episodes per
   1,000).  Their hazard density differs from the R runs, so they cannot
   substitute for a p050 pool without confounding coverage with the
   environment (and, in the ladder, with the teacher's route-coin
   composition).  Not used.

**Decision: STOP.**  The existing p050 data cannot support a meaningful
episode-diversity contrast on the strata that matter; the contrast that
is constructible (start strata) tests a different question at low power.
No manifest was sealed, no branch generated, no critic trained, no
reserved seed consumed.

## What is missing and a concrete collection plan (NOT executed; needs approval)

Missing: independent p050 detour episodes -- for training (to build a
BROAD arm with ~5x the detour episodes at the same anchor count) and for
evaluation (>= 30 held-out detour episodes instead of 5).

Plan A (composition-preserving, preferred): collect 5,000 additional p050
episodes with the collector settings of the existing dataset unchanged --
`scripts/collect_rockfall_clock_v6_dataset.py --p-active-1 0.5 --p-active-2
0.5 --teacher-detour-prob 0.05` and the same t0 ranges -- as five shards of
1,000 with seeds 607-611 (the existing dataset is seed 606), each with the
composition audit.  Expected yield: ~270 detour episodes (5.4%); training
detour pool 49 -> ~290, with ~30 detour episodes reserved as an untouched
evaluation pool.  Source behaviour mode, route proportion and environment
unchanged, so episode count is the only intervention.  Cost: the collector
is sequential, ~20 min per 1,000 episodes -> ~100 min, or ~40 min sharded
across the three nodes; storage ~100 MB per shard (gxy) + sidecars.

Plan B (cheaper, confounded): 1,000 p050 episodes at teacher detour
probability 0.30 (~300 detour episodes).  Changes the behaviour-mode
mixture (30% vs 5% detours; the teacher's route coin), so BOTH arms would
have to be drawn from the new collection and the result would not be
matched to the existing R / C1 runs.  Not recommended unless Plan A's
cost is unacceptable.

Experiment after collection (sealed before any rollout): NARROW = the
current turn / north-leg anchors (600 anchors, 49 episodes; the R
construction, seeds 0 / 1 / 2 retrained on the same hardware as BROAD);
BROAD = 600 turn / north-leg anchors over ~245 episodes (2-3 per episode,
round-robin, chosen without outcomes), every other stratum identical; the
R protocol for candidates (recorded + 2 policy samples), draws (8 per
policy torque, 2 per recorded), weights (state-stratum and recorded /
policy mass exactly R's), continuation, law; 30k NCE, random init, twin
critics, one GPU; evaluation on the new held-out detour episodes (32
anchors per stratum from >= 30 episodes, 16 fresh paired draws; ~5k
paths) plus the existing C set as a development check; pre-registered
line = BROAD's held-out pick gain above NARROW's by > 2 paired s.e. with
positive gain and consistent per-seed direction; actor stage only if
that passes.  Estimated wall time after collection: branch generation
~15 min (10,800 new paths per arm), critics ~15 min, evaluation outcomes
~10 min, analysis ~10 min; storage ~1.7 GB per arm replay.  Three
training seeds remain a limit on the training-randomness side of the
uncertainty.
