# Learned supervised ETT -> Contrastive RL: bounded one-seed pilot

Status: **complete**. The learned model passed the sealed preflight. Both matched fixed-final critics and actors were trained from the same seed-0 initialization with `bc_coef = 0.05`. Mode reach was unchanged; sampled reach had a +0.015 ETT-minus-original point estimate whose paired 95% interval includes zero. This pilot therefore provides no clear native reach benefit.

## What was trained

The movement module receives visible F4 `s`, recorded same-context expert advice `xb`, queried action `xq`, and action interactions; it minimizes separately averaged diagonal and off-diagonal full emitted-XY Energy Scores. The onset MLP receives the same visible conditioning plus successor displacement and minimizes natural-prevalence BCE. Hidden outcomes are targets/audit labels only, never policy, nominal, or rollout inputs. Selected step 1500; validation diagonal/off-diagonal ES 0.0105/0.1637; onset AUROC 0.985, event-minus-nonevent probability 0.430. All sealed checks passed, including action sensitivity, represented downward exits, exact F4 shifts, endpoint legality, and zero post-failure recovery.

The supervised rows are additional previously collected interventions, not just the original 6,600 observational episodes: 72 older paired-probe training episodes plus 128 later repair-training episodes (200 source-namespaced fitted episodes total); the later 32 validation episodes select/check the model. Raw episode IDs overlap between sources, which explains the 128 raw-ID count in `data_audit.json`. Older episodes 72--95 and the later 64 final episodes remained excluded and are explicitly not fresh confirmation.

## Replay and unchanged CRL

All 165,000 generated transitions came from the learned backend. Generated states fed the frozen nominal and frozen observational rollout actor at every later step; no recorded future XY, G4 onset rule, native step, oracle motion, or ground-truth death flag entered generation. The P replay is 50% complete learned trajectories and 50% a predeclared random subset of original trajectories. It therefore changes actor training contexts as well as NCE futures. All paths were retained through step 50, including absorbing tails.

The original sigmoid-NCE and discount-0.95 future sampler were unchanged. `nce_lineage.npz` records all 7,680,000 matched positives per arm as source trajectory, anchor, and future offset. The actor uses the existing objective with BC 0.05 and no AWR, ranking, failure-bank, or new regularizer.

## Fixed-final native evaluation (200 paired seeds)

| policy | O reach | ETT reach | ETT-O reach (paired 95% CI) | O lower | ETT lower | ETT-O lower (paired 95% CI) | O absorbed | ETT absorbed | O return | ETT return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mode | 0.325 | 0.325 | -0.000 [-0.000, -0.000] | 0.000 | 0.000 | -0.000 [-0.000, -0.000] | 0.675 | 0.675 | 4.28 | 4.28 |
| sample | 0.350 | 0.365 | +0.015 [-0.030, +0.060] | 0.200 | 0.135 | -0.065 [-0.100, -0.030] | 0.650 | 0.635 | 4.35 | 4.52 |

This is one training seed, so it is exploratory and says nothing about seed robustness. Evaluation outcomes were not used for fitting or checkpoint selection. No post-hoc binary policy-success threshold was introduced; the predeclared metrics and paired uncertainty are reported directly.

## Read-only task-goal critic ranking audit

After final evaluation, the frozen critics were applied to the 16 previously established coherent alive fork roots and exact `[0,-1]` down / `[1,0]` right candidates from the matched-fork study. This added no native interaction and was not used for fitting, selection, or protocol changes. Existing native continuations establish that down beats right by 6.859 return and 70.0 success percentage points, with down higher-return on all 16 roots.

O's down-minus-right critic logit is +0.015 [-0.078, +0.099] and ranks down first on 8/16 roots. P's is -1.788 [-1.821, -1.754] and ranks down first on 0/16. Thus learned replay did not repair the canonical task-goal route ranking; it confidently strengthened the wrong right/shortcut preference. This agrees with P's lower sampled lower-route rate.

## Traceable example

The trace begins at repair-training row 204 with its exact recorded `(s, xb, xq, y)` and onset label. The learned backend emitted the first successor and then generated the full continuation (state-array hash `11a9479c64faa311...`). Training update 22, batch row 44 used that trajectory at anchor 34 and future 41 (offset 7) as an NCE positive. `trace.json` gives the O and learned-replay final actor mode actions and critic values for downward, rightward, and collected-query probes at that exact generated anchor/future pair. This particular positive is on the absorbing tail and is a lineage example, not evidence about canonical task-goal route correctness; that question is answered by `critic_ranking_audit.json` above.

## What this does and does not establish

Unlike G4, which retained recorded movement and inserted rule-selected frozen tails, this pilot learns both stochastic movement and onset and uses learned emission for every synthetic successor. G4 BC-0.05 remains only a mechanism reference (reported there across three seeds); it is not a matched control here. This is supervised conditional outcome modeling plus ordinary CRL, not certified worst-case optimization. There is no retained 1-Lipschitz claim, no architecture sweep, no converged outer policy/model iteration, and no multi-seed confirmation.
