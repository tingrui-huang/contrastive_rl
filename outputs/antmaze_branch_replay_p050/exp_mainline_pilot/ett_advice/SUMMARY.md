# The advice process (ETT line, user's step 3, 2026-09-20): the memoryless nominal has NO support where the deaths are; its holds are state-driven, not context-driven

`scripts/diag_v6_ett_advice.py run` (node5, 20 s; no rollouts, no
training; the v3 models used as fixed predictors).  `manifest.json`,
`REPORT.md`, `report.json`.  Oracle-supervised engineering stage.

## What the real advice looks like (the teacher walked along every pilot branch under its redrawn timetable; 7,648,612 valid rows)

* Hold share 4.2 % of rows (8.8 % inside the two hazard bands, 3.6 %
  outside; 22 % of all holds are in-band).  By region: zone1 0.113,
  between 0.094, zone2 0.072, pre_zone1 0.053, post_zone2 0.012; the
  start, the goal area and the far-route legs 0.
* Strongly persistent: P(hold_{j+1} | hold_j) = 0.981, P(hold | no hold)
  = 0.002; 21,509 hold runs of mean 14.8 / median 14 / p90 22 steps,
  95 % of runs >= 5 steps, 1.3 % of length 1.
* 15,600 of the 15,612 onsets fall on hold rows; the per-step onset rate
  on a hold row is 0.042 (0.164 in band).

## The memoryless nominal a_b ~ MDN(s), one i.i.d. draw per row along the SAME real states (the rollout's 6,000 held-out branches, 868,101 rows)

| | real advice | nominal i.i.d. |
|---|---|---|
| hold share (all / in band / out of band) | 0.041 / 0.086 / 0.036 | 0.265 / **0.000** / 0.296 |
| P(hold given hold) / P(hold given no hold) | 0.982 / 0.002 | 0.959 / 0.014 |
| hold runs: mean / median / p90; share >= 20 | 14.9 / 14 / 22; 0.13 | 23.5 / 12 / 59; 0.35 |
| hold share by region | pre_zone1 0.05, zone1 0.11, between 0.09, zone2 0.07 | pre_zone1 0.72, zone1 0.00, between 0.38, zone2 0.00, east_column 0.11 |

Support: the nominal's exact hold probability p_h(s) is 0.000 inside the
bands; its AUROC against the real hold flag is 0.544 over all rows,
0.494 in band (chance), 0.659 at the query row and 0.453 beyond 50 steps
-- against 0.981 on the MDN's own logged validation rows.  The NLL of
the real advice under the nominal is -3.5 on real hold rows and 18.6 on
the others (27.7 beyond 50 steps; the log's own actions score -11.4):
the branch rows lie outside the nominal's support.

Reading.  In the log the teacher waits at the mouths and never drives
into an active zone, so (i) the states at which it holds are stationary
poses at the mouths, and (ii) no logged row shows a hold inside a band.
A memoryless nominal therefore learns "hold = stationary pose at the
mouth": along the branches it holds at the start agent's STALLS (the
0.72 in pre_zone1, 0.38 between -- long, persistent runs, because the
stalled state persists) and never in the bands, where the real advice
holds 8.8 % of the time and where 15,600 / 15,612 deaths happen.  Its
persistence (0.959) is an artefact of the state persisting, not a
representation of the hidden context.  So the user's point stands and is
now located: the nominal lacks support on the intervened states (in-band
rows under an active rock -- present only in the branch supervision's
310,781 off_hold rows) and its holds are not driven by a persistent
hidden context.

## The v3 onset head vs the hold counter (held-out real in-band hold rows, ~22-24k per fold; fixed models)

Realised onset rate by the real counter: 0.000 at kh 0-3, 0.09 at 4-7,
0.22 at 8-15, 0.38 at 16-31; the model's mean p_onset tracks it (0.0001
/ 0.0006 / 0.009 / 0.12 / 0.20 / 0.29 / 0.21; AUROC 0.74-0.75).  Forcing
the counter on the same rows: p_onset 0.03-0.04 at k = 0, 0.08-0.09 at k
= 5, 0.16-0.20 at k = 10, 0.28-0.35 at k = 20, then it falls (k = 50:
0.20-0.31; k = 100: 0.08-0.20 -- beyond the data, kh >= 32 is 0.5 % of
rows).  Drawing the counter from the nominal's own hold-run distribution
gives p_onset 0.17-0.23, close to the real-counter 0.18-0.22.

Reading.  A death needs a sustained in-band hold (>= 4 steps); the onset
head has learnt that.  The nominal's missing deaths (0.036 vs 0.30) are
NOT a counter-length problem (its runs are long enough) and not a
counter-offset problem: they come from the nominal never holding inside
a band.  Fixing the counter (hist_exact, running) cannot change this, as
the user said.

## What an advice process has to satisfy (the acceptance metrics, from this diagnostic; no hand-forced holds)

1. Support on the intervened states: hold probability inside the bands
   with AUROC against the real in-band hold well above 0.5 on held-out
   branches (the nominal: 0.494), and a finite NLL of the real advice on
   the branch rows.
2. Temporal coherence from a persistent hidden context, not from a
   persistent state: P(hold | hold) ~ 0.98 with runs of median ~14 /
   p90 ~22 IN BAND, and hold share by region matching the real advice
   (in band 0.09, not 0.00; pre_zone1 0.05, not 0.72).
3. The implied onset: with such advice the fixed v3 onset head must
   recover the death rate (0.29-0.30) and the death timing per stratum.

Candidate (for the user's decision, not started): a latent-context
advice model fitted on the branch supervision rows, where the hidden
context of every branch IS recorded (branch_advice.npz: u1 / u2 / t0
redrawn per anchor) -- a_b ~ A(s, z, history) with z the timetable
latent, z sampled per path from its prior at generation time (the
environment's design prior; for the offline story it would have to be
estimated from the log's mouth-waiting pattern), the hold arising from
the learned map when z says active and the ant is in the band.  It is
evaluated by 1-3 above on held-out branches before any rollout.
