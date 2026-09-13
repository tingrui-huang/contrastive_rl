# ETT pessimism headroom ceiling diagnostic

A2 reaches at least 10% mean reduction while both learned A1 responses remain below 3%; the bounded evidence favors an optimizer bottleneck rather than the response family as the limit.

This one-shot diagnostic uses answer-handed model oracles. Lower return is more pessimistic. Oracle results are achievable constructions under the specified frozen rules, not valid lower bounds on the true worst case.

## Setting 1: 36 held-out roots, fresh H=10

Returns are normalized discounted reward-region occupancy. The roots are the exact 36 group-balanced held-out roots associated with the learned A1 artifacts, but their stored source times are 1/3/6/10, not t=40. This resolves an inconsistency in the requested contract without selecting 36 of the separate 64-root t=40 artifact.

- a0_diagonal: mean 0.14440; p10/p25/median/p75/p90 0.00000/0.09960/0.13635/0.17504/0.25864; paired mean +0.00000, root CI [+0.00000, +0.00000], conditional-MC CI [+0.00000, +0.00000]; paired median +0.00000; signed mean change +0.00%.
- a1_s0_learned: mean 0.14412; p10/p25/median/p75/p90 0.00000/0.09960/0.13635/0.17504/0.25864; paired mean -0.00028, root CI [-0.00068, +0.00014], conditional-MC CI [-0.00080, +0.00023]; paired median +0.00000; signed mean change -0.20%.
- a1_s1_learned: mean 0.14441; p10/p25/median/p75/p90 0.00000/0.09960/0.13635/0.17504/0.25864; paired mean +0.00000, root CI [-0.00020, +0.00023], conditional-MC CI [-0.00038, +0.00039]; paired median +0.00000; signed mean change +0.00%.
- a2_lipschitz_oracle: mean 0.02030; p10/p25/median/p75/p90 0.00000/0.00000/0.00000/0.00000/0.03151; paired mean -0.12411, root CI [-0.12881, -0.12001], conditional-MC CI [-0.12741, -0.12080]; paired median -0.13635; signed mean change -85.94%.
- a3_box_oracle: mean 0.00000; p10/p25/median/p75/p90 0.00000/0.00000/0.00000/0.00000/0.00000; paired mean -0.14440, root CI [-0.15924, -0.13145], conditional-MC CI [-0.14726, -0.14155]; paired median -0.13635; signed mean change -100.00%.
- a4_postprocess_oracle: mean 0.00000; p10/p25/median/p75/p90 0.00000/0.00000/0.00000/0.00000/0.00000; paired mean -0.14440, root CI [-0.15801, -0.13204], conditional-MC CI [-0.14726, -0.14155]; paired median -0.13635; signed mean change -100.00%.

## Setting 2: 128 independent START rollouts, H=50

This is the mandatory reset-start transport comparison; it is not pooled with setting 1.

- a0_diagonal: mean 0.54737; p10/p25/median/p75/p90 0.26362/0.55330/0.62139/0.65815/0.65815; paired mean +0.00000, paired-rollout CI [+0.00000, +0.00000]; paired median +0.00000; signed mean change +0.00%.
- a1_s0_learned: mean 0.53404; p10/p25/median/p75/p90 0.17340/0.55330/0.62139/0.65815/0.65815; paired mean -0.01334, paired-rollout CI [-0.02422, -0.00478]; paired median +0.00000; signed mean change -2.44%.
- a1_s1_learned: mean 0.54461; p10/p25/median/p75/p90 0.27616/0.55330/0.62139/0.65815/0.65815; paired mean -0.00276, paired-rollout CI [-0.00807, +0.00160]; paired median +0.00000; signed mean change -0.50%.
- a2_lipschitz_oracle: mean 0.00904; p10/p25/median/p75/p90 0.00000/0.00000/0.00000/0.00000/0.00000; paired mean -0.53834, paired-rollout CI [-0.56691, -0.50631]; paired median -0.58991; signed mean change -98.35%.
- a3_box_oracle: mean 0.00000; p10/p25/median/p75/p90 0.00000/0.00000/0.00000/0.00000/0.00000; paired mean -0.54737, paired-rollout CI [-0.57392, -0.51864]; paired median -0.62139; signed mean change -100.00%.
- a4_postprocess_oracle: mean 0.00000; p10/p25/median/p75/p90 0.00000/0.00000/0.00000/0.00000/0.00000; paired mean -0.54737, paired-rollout CI [-0.57405, -0.51773]; paired median -0.62139; signed mean change -100.00%.

## Cross-setting comparison

Material disagreement by the sealed definition: True. The definition is opposite paired-mean direction or at least a five percentage-point reduction gap.

- a1_s0_learned: setting-1 R +0.20%; setting-2 R +2.44%; gap +2.24%; opposite direction False.
- a1_s1_learned: setting-1 R -0.00%; setting-2 R +0.50%; gap +0.51%; opposite direction True.
- a2_lipschitz_oracle: setting-1 R +85.94%; setting-2 R +98.35%; gap +12.41%; opposite direction False.
- a3_box_oracle: setting-1 R +100.00%; setting-2 R +100.00%; gap +0.00%; opposite direction False.
- a4_postprocess_oracle: setting-1 R +100.00%; setting-2 R +100.00%; gap +0.00%; opposite direction False.

## Tail sensitivity and mechanism

Top-5% shares use paired rollout differences ranked by absolute contribution. A signed share can exceed one under cancellation. `support` counts nonzero rollouts pointing in the mean direction; below 20 makes the paired median the predeclared primary summary.

- setting1:
  - a1_s0_learned: top-5% signed/absolute 1.00000/1.00000; support 54; tail flag False; delta norm mean/median/p90/max 0.01188/0.00677/0.02808/0.11285; box clip +7.59%; full reward-sequence changed +4.12%.
  - a1_s1_learned: top-5% signed/absolute 1.00000/1.00000; support 16; tail flag True; delta norm mean/median/p90/max 0.00548/0.00322/0.01377/0.03859; box clip +10.74%; full reward-sequence changed +1.82%.
  - a2_lipschitz_oracle: top-5% signed/absolute 0.13164/0.13158; support 1927; tail flag False; delta norm mean/median/p90/max 0.67613/0.59036/1.31255/2.15902; box clip +29.75%; full reward-sequence changed +83.72%.
  - a3_box_oracle: top-5% signed/absolute 0.12923/0.12923; support 1989; tail flag False; delta norm mean/median/p90/max 1.06978/1.04726/1.50308/2.22265; box clip +100.00%; full reward-sequence changed +86.33%.
  - a4_postprocess_oracle: top-5% signed/absolute 0.12923/0.12923; support 1989; tail flag False; delta norm mean/median/p90/max 1.12686/1.08297/1.72786/2.74643; box clip +0.00%; full reward-sequence changed +86.33%.
- setting2:
  - a1_s0_learned: top-5% signed/absolute 0.93629/0.93629; support 13; tail flag True; delta norm mean/median/p90/max 0.01102/0.00673/0.02158/0.10791; box clip +4.70%; full reward-sequence changed +10.16%.
  - a1_s1_learned: top-5% signed/absolute 1.00000/1.00000; support 4; tail flag True; delta norm mean/median/p90/max 0.00503/0.00350/0.01082/0.03891; box clip +3.28%; full reward-sequence changed +3.91%.
  - a2_lipschitz_oracle: top-5% signed/absolute 0.06686/0.06686; support 125; tail flag False; delta norm mean/median/p90/max 0.68195/0.61912/1.26105/2.01255; box clip +23.56%; full reward-sequence changed +97.66%.
  - a3_box_oracle: top-5% signed/absolute 0.06575/0.06575; support 125; tail flag False; delta norm mean/median/p90/max 1.01123/1.07930/1.21569/1.72899; box clip +100.00%; full reward-sequence changed +97.66%.
  - a4_postprocess_oracle: top-5% signed/absolute 0.06575/0.06575; support 125; tail flag False; delta norm mean/median/p90/max 1.05271/1.11080/1.23498/1.72899; box clip +0.00%; full reward-sequence changed +97.66%.

Pre-entry versus entered mechanism details, including decision counts, delta summaries, box clips, and suffix reward-sequence changes, are retained in `results.json`.

## Acceptance, cost, and limits

- Exact diagonal identity passed for every arm: True.
- A2 samplewise Lipschitz grid passed with maximum excess 2.41e-07.
- A3/A4 Lipschitz grid pass states (failure expected and retained): False/False.
- Charged model outputs: 207,360/600,000. A5 was skipped before outcomes because its exact 3,774,720-output design exceeds 1.5M.
- Frozen diagonal, nominal, actor checkpoint, normalization, and response hashes match before/after. All saved rollouts passed finite, anchor, coordinate-step, free-endpoint, exact F4-shift, and reward checks.
- There were zero updates, gradients, native steps, actor changes, or checkpoint writes. No critic or Monte Carlo value was loaded.

The comparison is conditional on one learned checkpoint pair, one diagonal model, one actor/nominal pair, the specified goal and synthetic transition postprocessing. Small oracle effects are not certified upper bounds: a different admissible construction, deeper search, or rare lower-quantile move could do more. These model trajectories do not establish physical realizability or native policy benefit.

Re-derive this report without rollouts: `python -m ett.report_pointmaze_pessimism_ceiling --out-dir artifacts/pointmaze_region_pilot/pessimism_ceiling_v1`.
