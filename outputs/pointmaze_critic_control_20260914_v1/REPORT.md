# PointMaze critic A/B control

## Answer

The preregistered result is **coverage sufficient**. The saved C1 critic began wrong on every new held-out root: teacher-signed down-minus-right logit was **-0.988** with 0/8 correct. After 400 critic-only updates on explicit, matched native fork continuations, A—ordinary sigmoid NCE only—reached **0.771 [0.536, 0.947]** and 8/8 correct. Thus improved route coverage plus continuation-policy matching was sufficient for the existing contrastive objective to learn the task-useful ordering on episode-disjoint roots.

B used the identical native data and NCE batches plus the model-return-derived ordinal loss. It reached **8.784 [8.571, 8.981]** and 8/8 correct. Its paired signed-gap increase over A was **8.012 [7.743, 8.380]**. This large margin is the direct effect of optimizing an ordinal logit gap; it is not evidence of a calibrated return estimate and is not needed to obtain the correct held-out ranking here.

Held-out native NCE loss started at 0.045806, ended at **0.036015 for A**, and **0.040132 for B**. Both improved over the start, but A fit the held-out native NCE task better than B. Critic parameter L2 movement was 2.388 for A and 3.916 for B.

## Data and teacher validity

Thirty-two new complete native episodes were run with the frozen C1 actor. Before fork outcomes, the first 24 eligible naturally reached alive fork contexts were split by episode into 16 train and 8 validation roots. A/B shared 384 complete native fork continuations and every one of their 400 NCE batches. Each batch contained all 256 train root/action/path identities once, including 64 down and 64 right rows anchored exactly at the fork; future full-F4 positives followed the original strictly-future geometric sampler.

The frozen model generated 1,536 additional continuations solely to compute labels. No direction was hard-coded. On held-out roots, model down-minus-right return was **6.267 [5.622, 7.031]** and native was **7.942 [6.614, 9.369]**; preferred direction agreed on 8/8 roots.

The old repeatedly inspected 16 roots were external reference only. They started at mean gap **-0.647**, 0/16 correct. Fixed-final A reached **0.948**, 16/16 correct; B reached **8.839**, 16/16 correct. These roots affected neither training nor the decision.

## What this establishes

The original wrong ranking is not evidence that the critic architecture cannot represent the detour preference. With direct exposure to balanced, naturally rooted native continuations generated under the final frozen actor, unchanged NCE learned the correct ordering and generalized across held-out episodes. The earlier failure is therefore localized to the training distribution/continuation mismatch at least strongly enough to be repaired by data alone in this fork-local pilot.

It does **not** establish that fork-only fine-tuning preserves arbitrary goals or that an updated actor improves full native success. B shows that an ordinal loss can force a much larger margin, but this experiment supplies no benefit of that extra margin and its held-out NCE loss is worse than A's.

## Recommended next intervention

Use the A-style change: add matched, naturally reached native fork continuations as an explicit **NCE-only** replay component while retaining the ordinary replay distribution. Do not add ranking loss yet. First verify both task ranking and general held-out NCE retention with the actor frozen; only then run a separately sealed actor-response probe and new complete native confirmation episodes.

## Integrity

Both conditions started from byte-identical C1 critic parameters and Adam state, used fixed update 400 as the final checkpoint, and performed no checkpoint search or sweep. Actor updates, model updates, and actor-training episodes were zero. One initial scoring attempt stopped before optimizer update because a single-head representation retained a `(N,1)` axis; the zero-update failure and one-line shape fix are preserved in `training_attempt_1.json` and `training_seal_amendment.json`. All data, labels, batches, budgets, and decision rules remained sealed and unchanged.
