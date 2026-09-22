# The critic and the consequences on one ruler, and the normalised ranking (user's plan after 555f22c; 2026-09-22; `scripts/diag_v6_critic_ruler.py`; existing checkpoints / tables / rollouts only -- no new trajectories, no training)

## 1. One ruler (`RULER.md`): the gamma-law mass of a goal region at the 64 reset states, per first action -- the simulator, the S ETT, the critic's readout

The recipe's critic is a binary NCE with one positive and B - 1 = 1023 in-batch negatives per row, so its logit estimates
f(s, a, g) = log p_gamma(g | s, a) / p(g) - log(B - 1), p(g) = the critic-training marginal (anchors by weight, the gamma law on the
table; 27 % of it lies in the goal area, 0.9 % in the start region).  The critic's mass of a region is then (B - 1) E_{g ~ p}[exp f 1{g in G}],
estimated on 8,192 marginal goals; its total over all g should be 1 (it is 0.6-1.4 for the twin-min logit at every (state, action) --
the ruler is calibrated; head 0 of the collapsed critic blows up at a few (action, goal) pairs, exp f (B - 1) up to 1e12, hidden by the
twin min).  Real = the simulator with the start agent's continuation (paired hidden draws, 64 per state), ETT = the S ETT with the same
continuation (32 per state).  The readout is the same under the other table's marginal (0.354 vs 0.358).

| region (real at the 4 actions) | real: logged / stall / prog / start | ETT | draw-1 s4 final critic (min head) | draw-1 20k | draw-2 s4 final | draw-2 20k |
|---|---|---|---|---|---|---|
| goal area | 0.040 / 0.033 / 0.034 / 0.035 | 0.033 / 0.029 / 0.032 / 0.034 | 0.061 / **0.354** / 0.038 / 0.003 | 0.059 / **0.221** / 0.074 / 0.012 | 0.022 / 0.009 / **0.052** / 0.003 | 0.031 / 0.021 / **0.090** / 0.009 |
| within 2.0 of the goal | 0.032 / 0.031 / 0.028 / 0.028 | 0.025 / 0.023 / 0.025 / 0.027 | 0.044 / **0.258** / 0.029 / 0.002 | 0.044 / 0.172 / 0.059 / 0.009 | 0.017 / 0.008 / 0.042 / 0.002 | 0.024 / 0.020 / 0.076 / 0.008 |
| within 0.5 | 0.0009 / 0.0009 / 0.0008 / 0.0009 | 0.0010 / 0.0010 / 0.0009 / 0.0009 | 0.007 / **0.047** / 0.005 / 0.001 | 0.007 / 0.031 / 0.011 / 0.002 | 0.002 / 0.002 / 0.008 / 0.001 | 0.004 / 0.004 / 0.014 / 0.002 |
| far regions | 0.008 / 0.133 / 0.110 / 0.003 | 0.000 / 0.037 / 0.022 / 0.000 | 0.001 / 0.040 / 0.009 / 0.001 | 0.002 / 0.043 / 0.017 / 0.002 | 0.001 / 0.004 / 0.009 / 0.001 | 0.002 / 0.010 / 0.023 / 0.002 |
| start region | 0.217 / 0.294 / 0.271 / 0.258 | - | 0.198 / **0.037** / 0.584 / 0.271 | 0.161 / 0.045 / 0.315 / 0.215 | 0.273 / 0.395 / 0.223 / 0.306 | 0.196 / 0.192 / 0.189 / 0.339 |

Per-state over-estimation factor critic / real (median, IQR), goal area: the collapsed critic at the stalled action **17.8x (9.2-28.9;
95 % of the states above 2x)**, already 9.5x at 20k; at the logged torque 1.8x, the progressing action 1.2x, the start agent's action
0.09x; the draw-2 critic at the stalled action 0.26x, at its own progressing action 2.2x (3.8x at 20k), at the start agent's 0.07x.
Paired contrasts stall - logged in goal-area mass: real -0.007 +- 0.011 (38 % of states positive), ETT -0.004 +- 0.003 (42 %),
draw-1 critic +0.293 +- 0.024 (100 %), 20k +0.162 (98 %), draw-2 critic -0.013 (25 %).

Agreement with the real masses over the 256 (state, action) pairs: the ETT Spearman 0.27-0.48 (goal area 0.48); every critic
|Pearson| <= 0.17, Spearman -0.14..0.14 in the goal / near regions, within-state rank correlation -0.16..0.11 -- the critics' readout carries
no information about the real conditional goal masses at these states.  Only the far region is read in the right direction by every
critic (0.33-0.41), because the saturated and the progressing actions really do send the ant onto the far route (0.13 / 0.11 vs 0.008).

Reading:
* On the same ruler, the collapsed critic's preference IS an over-estimate: it reads the stalled action's goal-area mass at 0.35 (the real
  0.033, the ETT 0.029), i.e. a density ratio of 1.3 against the marginal where the truth is 0.12; it also reads the stalled action as
  leaving the start region (0.037 vs the real 0.29).  The real and the ETT masses of the four actions are equal within noise -- the
  10-18x difference between the actions exists only in the critic.
* The ETT's conditional consequences at these specific actions are right in the goal masses (it under-states the far-route mass, 0.037
  vs 0.133, as it under-states the far entries; known).
* Both critics over-rate the action their OWN actor takes (draw 1 the saturated action 18x, draw 2 the progressing action 2.2x) and
  under-rate the actions the data executes (the start agent's 0.07-0.09x, the logged torque 0.4-1.8x).  The critic's training rows at
  these anchors carry only the logged action, so its value at any other action is an extrapolation in the action input, and the actor's
  ascent selects the action where that extrapolation is largest -- in draw 1 it lies at the action bound.

CORRECTION to stall_onset/SUMMARY.md ROUND 2 ("adding queries at these actions under the same continuation would give the critic a target
that is not lower than the logged torque's, i.e. it would not remove the preference by itself"): the target at the stalled action (0.03)
is 10x BELOW what the critic currently reads there (0.35), so a query at the stalled action would remove the over-estimate; what remains
after calibration is a near-tie between the actions (the real masses are equal), decided by BC and the residual critic differences.
Whether that makes the actor walk is not shown.

## 2. The normalised ranking (`NORMRANK.md`): per head cos, THEN the min over heads per goal, THEN the mean over the goal set; the scale matched

DECOMPOSE.md had averaged over the goals before taking the min; corrected here.  tau* per critic (0.029-0.042) is the temperature at which
the normalised score has the same within-row spread over the sampled actions as the unnormalised logit, so the BC : critic balance of the
recipe is unchanged; tau* / 2 and 2 tau* bracket it.

Mode actions at the 64 reset states, stalled minus the other action (share of states preferring the stall):

| critic | goal set | vs progressing | vs start agent | vs logged torque |
|---|---|---|---|---|
| draw-1 s4 final, unnormalised (nats) | task / relabelled / marginal | +2.14 (1.00) / +0.66 (0.73) / +1.40 (0.97) | +4.22 (1.00) / +1.54 (0.88) / +2.90 (1.00) | +2.43 (1.00) / -0.16 (0.36) / +0.74 (0.92) |
| same, normalised cos (= nats at tau*) | task / relabelled / marginal | +0.088 = +2.2 (0.88) / +0.032 = +0.8 (0.77) / +0.050 = +1.3 (0.88) | +0.048 = +1.2 (0.69) / -0.004 (0.48) / +0.022 (0.70) | +0.010 (0.44) / -0.056 = -1.4 (0.08) / -0.052 (0.17) |
| draw-2 s4 final, unnormalised | task / relabelled / marginal | -1.84 (0.09) / -1.31 (0.03) / -1.70 (0.02) | +0.92 (0.83) / +0.01 (0.47) / +0.40 (0.75) | -0.17 (0.50) / -1.44 (0.02) / -1.29 (0.03) |
| same, normalised cos | task / relabelled / marginal | **+0.027 (0.77)** / -0.002 (0.59) / -0.005 (0.58) | +0.005 (0.58) / -0.020 (0.12) / -0.012 (0.27) | -0.031 (0.25) / -0.060 (0.02) / -0.071 (0.02) |

The objective as trained (16 sampled actions per row, the actor stream's relabelled goals, 2,048 logged start rows), E Q(stalled policy) -
E Q(progressing policy) and the loss difference (negative = the stalled policy preferred):

| critic | unnormalised: E Q diff (rows preferring the stall) / loss diff | normalised at tau*: E Q diff / loss diff | at tau* / 2 | at 2 tau* |
|---|---|---|---|---|
| draw-1 final | +0.87 (0.81) / -0.66 | +0.53 (0.71) / -0.35 | loss diff -0.85 | -0.09 |
| draw-1 20k | +0.56 (0.83) / -0.37 | +0.25 (0.67) / -0.08 | -0.31 | +0.04 |
| draw-2 final | -1.03 (0.21) / +1.14 | -0.75 (0.25) / +0.88 | +1.59 | +0.51 |
| draw-2 20k | -0.53 (0.23) / +0.66 | -0.31 (0.29) / +0.46 | +0.75 | +0.30 |

Reading:
* The collapsed critic's preference for the stalled action over the PROGRESSING action does not weaken under normalisation at the matched
  scale: +2.1 -> +2.2 nats under the task goal, +0.7 -> +0.8 under the actor's relabelled goals, +1.4 -> +1.3 under the marginal (88 %
  of the states); in the sampled-action objective it shrinks from +0.87 to +0.53 (71 % of the rows) and the stalled policy's loss stays
  lower (-0.35; -0.09 even with the critic term halved).  What normalisation removes is the penalty on the LONG-phi actions -- the start
  agent's and the logged torque -- which then tie or win (vs logged under the relabelled goals: -1.4 nats, 8 %); the norm shrinkage
  of DECOMPOSE.md concerned those comparisons, not the stall-vs-progressing one.
* The recovering critic's walking preference is destroyed at the mode level: under the task goal the normalised score prefers the stalled
  action in 77 % of the states (flipped in 70 %), under the relabelled and marginal goals it is a tie (0.58-0.59); it survives only in the
  sampled-action objective (-0.75, 25 % of the rows), where the sampled actions stay near the progressing mode.
* Post-hoc normalisation of a critic trained without it is a necessary-condition check for `config.repr_norm`, not its effect; the
  condition is not met on either side -- the bad preference stays, the good one is damaged.  A fixed-budget repr_norm control is NOT
  supported by this check.

## Where the decision tree lands

* Normalisation: no clear support (2).
* The ETT wrong on these specific actions: no -- its goal masses at all four actions match the simulator's within 0.005 (its far-route
  under-statement is a separate, known limit).
* The ETT's conditional consequences right, the critic not fitting them: YES (1) -- the critic never sees any action but the logged one at
  these anchors, extrapolates in the action, and the actor's ascent finds the largest extrapolation; both critics over-rate their own
  actor's mode.  The tree's third branch -- a LOCAL calibration experiment at the actually output actions (critic rows at (s, a_actor) with
  the ETT's futures under the start continuation, at the start / reset anchors), read on this ruler and on the corrected short-update
  control -- is the branch the evidence points to.  It is the user's design call and was not launched.  Caveat from the same table: the
  over-estimate sits wherever the actor's maximiser sits (the draw-2 critic over-rates its own interior action 2.2x), so a calibration at one
  set of output actions may move the maximiser rather than remove the curse; the earlier agent-continuation query round (negative; CORRECTION on the user's review: it already ran under the
  clip-0.1 recipe -- 'pre-clip' in 4b33528 was wrong; its difference from the design below is the agent's own continuation) is the
  precedent to keep in view.

Files: `RULER.md` / `ruler_report.json` (+ `ruler_per_state.npz` on node 30027), `NORMRANK.md` / `normrank_report.json`.
