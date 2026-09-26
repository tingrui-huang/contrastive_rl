# The advice generator (ETT line, user's decision 2026-09-20): v3 reproduces the teacher's advice along held-out paths; gates passed; variant-B rollouts launched

`scripts/fit_v6_ett_advice.py` (node3 after node5 was wiped); `manifest.json`
(re-sealed 09:56 with the gate revision), `gen3_fold*.json`, `check3_fold*_map.json`
/ `check3_fold*.json` (MAP / sampled hold decision), `REPORT3_map.md` /
`REPORT3.md`, `gates3_map.json` / `gates3.json`.  Oracle-supervised
engineering revision: the supervision is the simulator teacher's advice
walked along every pilot branch under its redrawn timetable, and every
branch's timetable is recorded, so the hidden context is observed in
training; at generation a path draws its own context from the prior
(u ~ Bernoulli(0.5) per zone, t0_1 ~ U{10..45}, t0_2 ~ U{90..125}; the
environment's design, checked on the 1,000 logged and 53,747 redrawn
contexts) and keeps it; nothing is read from the evaluation episode's
hidden draw; no hand-forced holds.

## Three versions (all disclosed; v1 / v2 results from the node5 runs before the wipe, numbers in the plan note)

* v1: one hold head on (state, goal, clock, context, previous-row hold
  counter, previous torque); Bernoulli hold.  Teacher-forced held-out
  AUROC 0.996, but autoregressively the runs fragment (median 2 vs 14,
  P(hold | hold) 0.93 vs 0.98) and the frozen onset head gives P(death)
  0.14 vs 0.29 (AUROC 0.70).  Probe: the calibrated p at the rows where a
  hold STARTS is 0.04 (continuations 0.95, releases 0.56) -- the head had
  learnt "hold iff the previous row held".  MAP decisions: deaths 0.04.
* v2: start / continue heads with rare-event weighting, inverted at
  inference (calibrated).  Starts still 0.08: the start is not a function
  of these features.
* v3: the teacher consults its rule ONCE, at the path's first arrival at a
  zone's mouth (x in [MOUTH_X, HAZARD_X), |y| < 2), decides wait iff the
  burst overlaps the crossing window from that step, then holds until the
  burst ends wherever the ant is.  v3 adds per zone the visible-history
  features "arrived at the mouth", steps since that arrival, (arrival -
  t0), from the logged prefix and the path's own positions (maze geometry
  only).  Starts p 0.55 teacher-forced (P > 0.5: 0.59; P > 0.2: 0.84).

## v3 held-out checks (MAP hold decision; 2,000 held-out paths per fold; the sampled decision passes the same gates)

| fold | E1 in-band AUROC / ECE (true context, teacher-forced) | E2a row agreement with the real advice (true context, autoregressive) | hold share / in band / P(h|h) / run median: generated vs real | P(death) generated / real advice / realised | death AUROC gen / real advice | death-time KS |
|---|---|---:|---|---|---|---:|
| 0 | 1.000 / 0.0007 | 0.994 | 0.038 / 0.082 / 0.984 / 14 vs 0.040 / 0.085 / 0.982 / 14 | 0.165 / 0.167 / 0.292 | 0.974 / 0.991 | 0.060 |
| 1 | 1.000 / 0.0010 | 0.993 | 0.041 / 0.081 / 0.982 / 13 vs 0.042 / 0.085 / 0.981 / 14 | 0.189 / 0.194 / 0.296 | 0.965 / 0.988 | 0.062 |
| 2 | 0.999 / 0.0006 | 0.996 | 0.041 / 0.086 / 0.982 / 14 vs 0.041 / 0.088 / 0.982 / 14 | 0.187 / 0.186 / 0.293 | 0.977 / 0.991 | 0.083 |

Region hold shares (generated vs real, true context) agree within 0.003
in every hazard region; under the PRIOR context the generator never holds
in the start, the far-route legs or the goal area (v1 held 7 % in the
goal area: an unsupported (goal area, burst active) combination), and
its in-band hold share is 0.26-0.27 against the real 0.085 -- a selection
effect (the real rows are censored by the real context: a path under an
active rock dies early), reported, not gated.

Gate revision (disclosed): the first v3 fold results showed two gates were
not like for like on teacher-forced, outcome-censored paths -- a hazard
integrated along a path that died at j* cannot reach 1 by j* (a calibrated
hazard gives ~0.5 there), so the death-rate gate now compares the generated
advice with the REAL advice through the same frozen head along the same
rows (gap <= 0.03; observed <= 0.005); persistence and run length are
compared under the true context (E2a), not prior context vs real; E2b
keeps the support gate.  All gates pass in 3 / 3 folds -> the full rollout
(`diag_v6_ett_rollout.py --v3 --hist-exact --variant B`, K = 4 advice paths
per anchor, MAP hold, sampled torque, prior context per path) runs on
node3; its Step 3a table (C / A / B vs the simulator) is the test of the
advice process as a whole.  Fold 0: model death 0.301 / reach 0.615 vs
the simulator's 0.292 / 0.611 (pooled; strata in the report).

Limits: the 5 % detour-intent episodes' branch advice never holds (the
teacher's detour mode does not consult the zones) and the intent is not a
feature -- the generator holds there like a shortcut teacher; the
generator's driving torque is an MDN draw, not the teacher's gait.
