# Query coverage, stage 2: the controlled training comparison (user's decision after stage 1)

`manifest.json` (sealed before training; the two arms, the evaluation draw
and the judgement fixed in advance), `REPORT.md` / `report.json`,
`readout_s{0,1,2}.json`, per run `train_manifest.json` +
`eval_mean_s6909.json` (checkpoints on node5).  Script
`scripts/exp_v6_query_train.py`.  Status: an exploratory training
experiment added after stage 1 (whose pooled gate G1 stays NOT met);
oracle (simulator) futures under the disclosed clip recipe.

## Design

Per lineage s (all three): **control** = the start-region anchors get the
q0 (logged-torque) futures of `queries_s{s}.npz` (both hazard draws);
**extended** = the q0..q5 futures (12 per anchor) with the branch's own
first torque as the critic action; every other anchor = the sealed
mainline futures in both arms.  Both arms resume the lineage final
(30,000 updates; policy, critic, target, both Adam states with the clip
chain, key), streams fast-forwarded, +30,000 updates; NCE, actor
objective, BC 0.05, clip 0.1, the actor / BC stream and the anchor
weights unchanged (the weight is split equally over the allowed branches
by a separate RNG).  Identity check: in every lineage the two arms drew
the same anchor sequence and the same actor rows (equal hashes) and
differ only in goals and critic actions.  **continue** (reference, added
after lineages 0 / 1 came in): the same resume on the UNCHANGED sealed
futures, no queries file -- it separates "30,000 more updates" from "the
start-region futures changed".  Evaluation: every final and the lineage
agents ("current") once on a fresh paired draw (seed 6909, 300
episodes, mode).

## Result: judgement 1 not met; continued training itself does not preserve the current policy; the extended arm moves the actor toward the logged torque and the shortcut

| policy | lineage 0: success / far share / death / timeout | lineage 1 | lineage 2 |
|---|---|---|---|
| current (the lineage agent) | 0.520 / 0.59 / 0.19 / 0.29 | 0.437 / 0.44 / 0.35 / 0.21 | 0.433 / 0.31 / 0.43 / 0.14 |
| continue (sealed futures, +30k) | 0.517 / 0.54 / 0.01 / 0.48 | 0.307 / 0.43 / 0.04 / 0.65 | 0.150 / 0.27 / 0.00 / 0.85 |
| control (q0 futures, +30k) | 0.497 / 0.63 / 0.00 / 0.50 | 0.053 / 0.40 / 0.00 / 0.95 | 0.030 / 0.00 / 0.00 / 0.97 |
| extended (q0..q5 futures, +30k) | 0.267 / 0.12 / 0.48 / 0.26 | 0.233 / 0.30 / 0.14 / 0.63 | 0.180 / 0.06 / 0.32 / 0.50 |
| start agent (reference) | 0.270 / 0.01 / 0.71 / 0.02 | | |

Paired differences in success (per lineage; mean; seed s.e.; rule = mean
> 2 x s.e. and 3 / 3):

| comparison | per lineage | mean | seed s.e. | rule |
|---|---|---|---|---|
| extended - control | -0.230 / +0.180 / +0.150 | +0.033 | 0.132 | not met |
| extended - current | -0.253 / -0.203 / -0.253 | -0.237 | 0.017 | 3 / 3 WORSE |
| control - current | -0.023 / -0.383 / -0.403 | -0.270 | 0.123 | 3 / 3 worse |
| continue - current | -0.003 / -0.130 / -0.283 | -0.139 | 0.081 | |
| extended - continue | -0.250 / -0.073 / +0.030 | -0.098 | 0.084 | |

1. **Judgement 1 (extended > control and > current) is not met**: the
   extended arm is below the current policy in every lineage (-0.20 to
   -0.25) and beats control only in the two lineages where control
   collapsed.
2. **Continued training alone does not preserve the current policy**:
   the reference arm loses 0.13 / 0.28 in lineages 1 / 2 (unchanged in
   0) with the known signature -- deaths -> 0, timeouts up, the agent
   stalling near the start (100-157 no-route episodes).  The round-1
   full-state resumption showed the same (`../../round2/SUMMARY.md`);
   the clip removed the critic runaways but at 30,000 updates the
   process is still moving.  Every +30k arm therefore has to be read
   against `continue`, not only against `current`.
3. **Control (the lineage agent's own continuation at the start anchors,
   logged first torque) makes the stall far worse** (lineages 1 / 2:
   success 0.05 / 0.03, 282 / 300 episodes never leave the start region
   in lineage 2).  Changing the start-region futures' source is itself a
   large intervention.
4. **Extended moves the actor the other way**: it does not stall; it
   takes the shortcut and dies (far share 0.12 / 0.30 / 0.06 vs the
   current 0.59 / 0.44 / 0.31; deaths 0.48 / 0.14 / 0.32).  The
   in-sample read-outs say the same at the training states: from the 196
   reset anchors (both hazard draws) the actor's own rollout completes
   the far route 0.41 / 0.40 / 0.29 (current; = stage 1's "mode" row) ->
   0.05 / 0.09 / 0.02 (extended), and the extended mode's torque distance
   to the logged torque falls (2.84 -> 1.84, 1.86 -> 1.29, 1.82 -> 1.70)
   while its distance to the best-outcome query rises (0.92 -> 2.13,
   0.79 -> 1.46, 0.88 -> 1.39): the actor moved TOWARD the logged
   (shortcut) first torque.

Critic read-out (task goal; the six queries of each start anchor ranked by
the critic against the realised success of the same branches):

| lineage | critic | reset rows: within-anchor AUROC | argmax query succeeded | f(best) - f(logged) | all start anchors: AUROC |
|---|---|---|---|---|---|
| 0 | current / control / extended / continue | 0.619 / 0.562 / 0.622 / 0.574 | 0.58 / 0.53 / 0.58 / 0.55 | +1.78 / +0.71 / +1.02 / +1.14 | 0.511 / 0.517 / 0.543 / 0.511 |
| 1 | current / control / extended / continue | 0.583 / 0.572 / 0.668 / 0.571 | 0.54 / 0.55 / 0.63 / 0.57 | +1.05 / +1.13 / +0.82 / +0.77 | 0.511 / 0.526 / 0.563 / - |
| 2 | current / control / extended / continue | 0.621 / 0.583 / 0.654 / 0.594 | 0.55 / 0.56 / 0.58 / 0.53 | +0.79 / +0.38 / +0.62 / +0.88 | 0.515 / 0.525 / 0.552 / - |

The extended critic ranks the added actions' consequences somewhat better
than the current / control / continue critics (reset-row AUROC +0.00 / +0.09 / +0.03;
all start anchors +0.03 to +0.05), but only to 0.62-0.67, and its margin
for the best query under the task goal does not grow.  So the closest
pre-fixed branch is **judgement 2**: the critic learns the candidate
differences (modestly), the actor does not improve -- and it moved toward
the logged torque, i.e. the added supervision was converted into
shortcut behaviour.  A mechanism consistent with this, NOT tested here:
the actor's training rows carry relabeled goals that lie on the 95 %
shortcut logs; a critic that now knows the agent's own first step does
not reach those shortcut-corridor goals pushes the actor, on those rows,
toward the logged action (the user's earlier point that a shortcut-goal
training row asks for the shortcut).  Under the user's rule: stop adding
queries; the remaining problem is policy extraction / the objective (the
actor's goal distribution and BC), against a training process that is
itself not stable past 30,000 updates.

Limits: one evaluation draw (300 episodes), three lineages; +30k is one
budget; the read-outs are in-sample; the continue reference was added
after two lineages were seen (its numbers are not part of the pre-fixed
comparison); nothing here says what a from-scratch training with extended
queries would do.
