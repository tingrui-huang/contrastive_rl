# Episode coverage NARROW vs BROAD: summary

Reference commits 355d273 and 1966335.  Feasibility audit: `FEASIBILITY.md`
(the original p050 data exhausted its 49 training detour episodes; Plan A
approved).  Sealed manifest `manifest.json` (before any rollout; git
sidecar 1966335 with the uncommitted diagnostic files; script hashes
inside).  Everything ran on node 30021 (RTX 4090): both arms' critics on
the same GPU, same code, same seeds.

## Data (Plan A)

Five new p050 shards, seeds 607-611, the original collector settings
(p_active 0.5 / 0.5, teacher detour probability 0.05, t0 [10, 45] / [90,
125], horizon 800): 1,000 episodes each, no discards, composition audit
passed, success 0.997-0.999 (timeouts 0.1-0.3%), detour fractions 0.040 /
0.049 / 0.051 / 0.043 / 0.046 -> 227 new detour episodes (the original:
54).  Merged dataset `antmaze_rockfall_clock_v6_p050_plus5k` (6,000
episodes; sha in `merge_summary.json`); the original episodes keep their
indices.  30 new detour episodes reserved as the untouched evaluation
pool (Cnew); 197 join the 49 original training detour episodes -> a pool
of 246.  The old 100 held-out episodes stay held out.

## Arms (identical except for which episodes supply the turning / north-leg anchors)

| | NARROW | BROAD |
|---|---|---|
| turning anchors / episodes | 300 / 49 (median 5 per episode, max 25) | 300 / 246 (max 2 per episode) |
| north-leg anchors / episodes | 300 / 49 (median 6, max 11) | 300 / 246 (max 2) |
| everything else | the R replay verbatim | the R replay verbatim (start strata, general anchors) |
| candidates / draws / weights | recorded (2 draws, w 1) + 2 policy samples (8 draws, w 0.25) | same (new candidate and draw seed streams) |
| effective masses | dense 0.5069, recorded 0.4155, turn 0.1014, north 0.1014 | identical to four decimals |
| critics | 30k NCE, random init, seeds 0 / 1 / 2, 4090 | same |
| anchor time (turn / north) min / median / max | 6 / 17 / 99 ; 17 / 44 / 135 | 6 / 15 / 94 ; 16 / 43 / 159 |
| goal xy mean (turn / north) | (24.83, 0.77) / (24.83, 0.73) | (24.77, 0.74) / (24.78, 0.74) |

Remaining differences: BROAD's turning / north-leg episodes are mostly
absent from the general anchors (which stay on the original 900
episodes); NARROW's are present there.  Selection used no outcomes or
scores (round-robin over episodes, seeded).  BROAD's new paths: reach
0.69, death 0.03, P_goal 0.21 (the strata's usual profile).

Cost: collection ~10 min per shard (three nodes in parallel), BROAD
branches + Cnew outcomes 1,079 s (15 workers), critics 354-359 s each.

## Primary endpoint: Cnew (30 reserved detour episodes, 32 turning + 32 north-leg anchors, 5 candidates, 16 paired draws)

437 decided pairs (61 ties, 142 weak); region-integrated readout,
deployed min; s.e. = episode bootstrap; pick gain with uniform
tie-breaking; the reserved final-test seeds remain unused.

| arm | agreement seed 0 / 1 / 2 (seed-mean +- s.e.) | pick gain seed 0 / 1 / 2 (seed-mean +- s.e.) | gain vs the mode candidate |
|---|---|---|---|
| NARROW | 0.46 / 0.51 / 0.45 (0.473 +- 0.026) | +0.007 / +0.017 / -0.001 (+0.0075 +- 0.0103) | +0.047 / +0.057 / +0.039 |
| BROAD | 0.46 / 0.48 / 0.46 (0.465 +- 0.027) | -0.011 / +0.003 / -0.008 (-0.0052 +- 0.0123) | +0.029 / +0.043 / +0.032 |
| random scorer | 0.54 | -0.018 | +0.022 |

Paired BROAD - NARROW (episode bootstrap, 2,000 resamples): pick gain
-0.013 +- 0.011 (z -1.2; per seed -0.018 / -0.014 / -0.007), agreement
-0.008 +- 0.023 (z -0.3).  Per stratum: turning NARROW 0.48 / 0.50 / 0.45
vs BROAD 0.47 / 0.47 / 0.46; north leg 0.45 / 0.51 / 0.45 vs 0.46 / 0.49 /
0.46 -- all within noise of 0.5.  The outcomes themselves carry a large
usable signal here: a cross-fitted empirical selector (eight draws pick,
eight score) gains +0.164 (turning +0.155, north leg +0.173) -- neither
critic captures any of it.  The "vs mode" gains are positive for every
scorer including random because the policy mode is the worst candidate
on these states (reach 0.58 against 0.68-0.70 for the others).

**Pre-registered criterion** (BROAD above NARROW on Cnew pick gain by > 2
paired s.e., positive absolute gain, consistent per-seed direction): NOT
met -- the sign is the opposite one on all three seeds.  Actor stage not
triggered.

## Development sets (`diag_r1_abc`, membership reclassified)

- Old C (5 held-out detour episodes for turning / north leg): pooled
  NARROW 0.47 / 0.48 / 0.51, BROAD 0.50 / 0.49 / 0.52.  The turning cell
  (237 pairs, 5 episodes) favours BROAD on all seeds (0.56 / 0.61 / 0.54
  vs 0.45 / 0.44 / 0.47; pick gain +0.035 / +0.041 / +0.066 vs -0.015 /
  -0.009 / -0.013) while the north-leg cell goes the other way (0.45 /
  0.45 / 0.50 vs 0.48 / 0.50 / 0.52); the 30-episode primary set shows
  neither.  A 5-episode cell is not evidence against the primary.
- A: the diagnostic's turning / north-leg A anchors are NARROW training
  keys (0.85-0.89 / 0.75-0.85) but not BROAD keys (0.54-0.58 / 0.49-0.54,
  their episodes partly in BROAD's pool with other anchors); the start
  strata are identical keys for both arms (NARROW 0.57-0.65 / 0.50-0.59 /
  0.42-0.46, BROAD 0.49-0.59 / 0.47-0.56 / 0.42-0.46).  Not a paired
  A-layer result across arms.
- B: NARROW 0.57 / 0.62 / 0.54; BROAD 0.52 / 0.51 / 0.50 (B's turning /
  north-leg states are BROAD-unfamiliar, so B is not matched across arms
  either).  NARROW here is a fresh retrain of the R replay: its seed 1
  reads 0.62 against 0.53 for the earlier critics_armR seed 1 (different
  GPU) -- a direct measure of how much three training seeds leave
  unresolved.

## Conclusion

Spreading the turning / north-leg supervision from 49 to 246 independent
episodes at the same anchor count, budget and weights did not improve
critic generalisation to 30 new held-out detour episodes: both arms are
at chance in agreement (0.47 / 0.47) and at zero in pick gain (+0.008 /
-0.005, difference -0.013 +- 0.011), while the fresh outcomes at those
states offer a +0.16 selection gain that neither arm captures.  What
improved: nothing on the primary endpoint.  What did not: transfer to new
episodes' states, in either arm.  What remains uncertain: the old-C
turning cell (5 episodes) leans BROAD and the training-seed spread is
visible (NARROW seed 1 0.62 vs 0.53 on B across two trainings), so a
small real effect below ~0.02 pick gain cannot be excluded; a 5x
increase of detour episodes is not a limit on how far coverage could be
pushed, and the general anchors were not broadened.  This weakens the
claim that independent episode coverage is what limits this recipe at
this scale; it does not show that single-step AntMaze control is
impossible, that other representations or NCE variants would fail, that
more episodes never matter, or that a learned-ETT pipeline has
succeeded (generation is oracle).  Nothing committed.

Files: `manifest.json`, `manifest_git_sidecar.json`, `merge_summary.json`,
`build_broad.json`, logs `ec_*.log`; `diag_cnew/` (manifest, outcomes.npz,
REPORT_{narrow,broad}.md, metrics_{narrow,broad}.json, paired_boot_C.json);
`diag_r1_abc/{REPORT,metrics}_ep_{narrow,broad}.*`; critics
`critics_{narrow,broad}/seed_*/` (manifests local, checkpoints on node
30021); the BROAD replay (1.7 GB) and the merged dataset on node 30021,
the five shards local under `artifacts/rockfall_clock_v6/dataset/`.
