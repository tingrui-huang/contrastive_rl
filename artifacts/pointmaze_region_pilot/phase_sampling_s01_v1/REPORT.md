# Frozen PointMaze critic: input-row phase sampling

Only the training input-row sampler differs between arms. The generator, actor, nominal sampler, reward, architecture, normalization, binary NCE loss, 1:31 class weighting and Q decoding are unchanged. No ETT or actor update occurred.

Four critics completed1500 steps each. All share identical288 training trajectories and geometric positive labels. Baseline seed0 exactly reproduces the previous failed critic. Model outputs: 538,221/650,000; fresh native context collection:30,000/30,000 steps.

Early/middle/late training phases are [0,5), [5,H//2), [H//2,H). Calibration queries are their starting rows, with64 independent same-generator targets and a fixed first action. Groups describe the original root, including later generated queries. Values are normalized .05*sum(.95^t*r), not raw logits.

## Training exposures and empirical fit

- uniform_s0: early/middle/late exposures [40495, 147745, 195760], final minibatch NCE 0.120879.
- balanced_s0: early/middle/late exposures [127640, 128153, 128207], final minibatch NCE 0.124272.
- uniform_s1: early/middle/late exposures [40411, 147600, 195989], final minibatch NCE 0.119998.
- balanced_s1: early/middle/late exposures [128059, 127845, 128096], final minibatch NCE 0.124454.

Full-row empirical NCE by phase/group is in training.json. It is fit to sampled Bernoulli labels, not an exact-value error. The following seen-query calibration instead uses independent continuations at actual training state/action rows.

## Seen training queries

uniform_s0:

- early: RMSE 0.07498 (95% CI [0.06412, 0.08542]), bias 0.06329 (CI [0.0513, 0.07515]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03252.
  - approach: RMSE 0.08446, bias 0.08122; informative same-h pairs 1/66, equal 0, unresolved 65; accuracy 1.0, adequate ordering sample False.
  - transit: RMSE 0.08129, bias 0.06750; informative same-h pairs 0/45, equal 0, unresolved 45; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.05590, bias 0.04114; informative same-h pairs 12/66, equal 0, unresolved 54; accuracy 0.9166666666666666, adequate ordering sample True.
- middle: RMSE 0.07947 (95% CI [0.057, 0.09934]), bias 0.01770 (CI [-0.00691, 0.04412]), range excess 0.11543, range violation fraction 0.278, maximum MC SE 0.04196.
  - approach: RMSE 0.10107, bias 0.01216; informative same-h pairs 52/66, equal 1, unresolved 13; accuracy 0.9807692307692307, adequate ordering sample True.
  - transit: RMSE 0.08105, bias 0.02191; informative same-h pairs 23/45, equal 10, unresolved 12; accuracy 1.0, adequate ordering sample True.
  - bypass: RMSE 0.04648, bias 0.01904; informative same-h pairs 33/66, equal 0, unresolved 33; accuracy 1.0, adequate ordering sample True.
- late: RMSE 0.03748 (95% CI [0.01829, 0.05443]), bias 0.01887 (CI [0.00965, 0.03032]), range excess 0.05258, range violation fraction 0.778, maximum MC SE 0.03850.
  - approach: RMSE 0.05786, bias 0.02997; informative same-h pairs 30/66, equal 36, unresolved 0; accuracy 0.9666666666666667, adequate ordering sample True.
  - transit: RMSE 0.02058, bias 0.01189; informative same-h pairs 0/45, equal 45, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.02107, bias 0.01474; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

balanced_s0:

- early: RMSE 0.05793 (95% CI [0.051, 0.06484]), bias -0.04866 (CI [-0.05612, -0.04122]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03252.
  - approach: RMSE 0.02938, bias -0.01995; informative same-h pairs 1/66, equal 0, unresolved 65; accuracy 1.0, adequate ordering sample False.
  - transit: RMSE 0.06023, bias -0.05319; informative same-h pairs 0/45, equal 0, unresolved 45; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.07469, bias -0.07283; informative same-h pairs 12/66, equal 0, unresolved 54; accuracy 1.0, adequate ordering sample True.
- middle: RMSE 0.10049 (95% CI [0.08827, 0.11141]), bias -0.07031 (CI [-0.0917, -0.04433]), range excess 0.05711, range violation fraction 0.028, maximum MC SE 0.04196.
  - approach: RMSE 0.12660, bias -0.06380; informative same-h pairs 52/66, equal 1, unresolved 13; accuracy 0.9807692307692307, adequate ordering sample True.
  - transit: RMSE 0.08114, bias -0.06016; informative same-h pairs 23/45, equal 10, unresolved 12; accuracy 1.0, adequate ordering sample True.
  - bypass: RMSE 0.08765, bias -0.08698; informative same-h pairs 33/66, equal 0, unresolved 33; accuracy 0.9393939393939394, adequate ordering sample True.
- late: RMSE 0.03302 (95% CI [0.02001, 0.04725]), bias -0.00180 (CI [-0.01182, 0.00984]), range excess 0.05771, range violation fraction 0.389, maximum MC SE 0.03850.
  - approach: RMSE 0.04923, bias 0.00207; informative same-h pairs 30/66, equal 36, unresolved 0; accuracy 0.9666666666666667, adequate ordering sample True.
  - transit: RMSE 0.02089, bias -0.00859; informative same-h pairs 0/45, equal 45, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.02026, bias 0.00113; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

uniform_s1:

- early: RMSE 0.05687 (95% CI [0.04652, 0.06609]), bias 0.02748 (CI [0.01573, 0.03792]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03252.
  - approach: RMSE 0.07366, bias 0.06558; informative same-h pairs 1/66, equal 0, unresolved 65; accuracy 1.0, adequate ordering sample False.
  - transit: RMSE 0.06048, bias 0.03725; informative same-h pairs 0/45, equal 0, unresolved 45; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.02489, bias -0.02039; informative same-h pairs 12/66, equal 0, unresolved 54; accuracy 1.0, adequate ordering sample True.
- middle: RMSE 0.07280 (95% CI [0.05978, 0.0842]), bias -0.03928 (CI [-0.05786, -0.0182]), range excess 0.01257, range violation fraction 0.056, maximum MC SE 0.04196.
  - approach: RMSE 0.10694, bias -0.05237; informative same-h pairs 52/66, equal 1, unresolved 13; accuracy 0.9807692307692307, adequate ordering sample True.
  - transit: RMSE 0.06031, bias -0.03990; informative same-h pairs 23/45, equal 10, unresolved 12; accuracy 1.0, adequate ordering sample True.
  - bypass: RMSE 0.02879, bias -0.02558; informative same-h pairs 33/66, equal 0, unresolved 33; accuracy 0.8484848484848485, adequate ordering sample True.
- late: RMSE 0.02731 (95% CI [0.01774, 0.03724]), bias 0.00760 (CI [1e-05, 0.01673]), range excess 0.05442, range violation fraction 0.556, maximum MC SE 0.03850.
  - approach: RMSE 0.03754, bias 0.00688; informative same-h pairs 30/66, equal 36, unresolved 0; accuracy 0.9666666666666667, adequate ordering sample True.
  - transit: RMSE 0.01866, bias 0.00246; informative same-h pairs 0/45, equal 45, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.02189, bias 0.01345; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

balanced_s1:

- early: RMSE 0.04068 (95% CI [0.0326, 0.04857]), bias 0.02343 (CI [0.01334, 0.03335]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03252.
  - approach: RMSE 0.04065, bias 0.03463; informative same-h pairs 1/66, equal 0, unresolved 65; accuracy 1.0, adequate ordering sample False.
  - transit: RMSE 0.05521, bias 0.02796; informative same-h pairs 0/45, equal 0, unresolved 45; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.01629, bias 0.00771; informative same-h pairs 12/66, equal 0, unresolved 54; accuracy 1.0, adequate ordering sample True.
- middle: RMSE 0.07564 (95% CI [0.05528, 0.09517]), bias 0.00679 (CI [-0.01685, 0.03282]), range excess 0.08796, range violation fraction 0.194, maximum MC SE 0.04196.
  - approach: RMSE 0.10379, bias 0.00388; informative same-h pairs 52/66, equal 1, unresolved 13; accuracy 0.9615384615384616, adequate ordering sample True.
  - transit: RMSE 0.06887, bias -0.00743; informative same-h pairs 23/45, equal 10, unresolved 12; accuracy 1.0, adequate ordering sample True.
  - bypass: RMSE 0.04063, bias 0.02393; informative same-h pairs 33/66, equal 0, unresolved 33; accuracy 0.7878787878787878, adequate ordering sample True.
- late: RMSE 0.03791 (95% CI [0.02538, 0.05119]), bias 0.02430 (CI [0.01539, 0.03461]), range excess 0.07623, range violation fraction 0.806, maximum MC SE 0.03850.
  - approach: RMSE 0.05063, bias 0.02647; informative same-h pairs 30/66, equal 36, unresolved 0; accuracy 0.9666666666666667, adequate ordering sample True.
  - transit: RMSE 0.02439, bias 0.01808; informative same-h pairs 0/45, equal 45, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.03397, bias 0.02835; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

## Untouched final episodes

uniform_s0:

- early: RMSE 0.07166 (95% CI [0.06059, 0.08199]), bias 0.04691 (CI [0.03077, 0.06272]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03170.
  - approach: RMSE 0.08102, bias 0.07587; informative same-h pairs 0/66, equal 0, unresolved 66; accuracy None, adequate ordering sample False.
  - transit: RMSE 0.08752, bias 0.04351; informative same-h pairs 1/25, equal 0, unresolved 24; accuracy 1.0, adequate ordering sample False.
  - bypass: RMSE 0.03440, bias 0.02135; informative same-h pairs 29/66, equal 0, unresolved 37; accuracy 0.7241379310344828, adequate ordering sample True.
- middle: RMSE 0.09657 (95% CI [0.06598, 0.12494]), bias 0.03647 (CI [0.0115, 0.06368]), range excess 0.16633, range violation fraction 0.333, maximum MC SE 0.04127.
  - approach: RMSE 0.09383, bias -0.01121; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 0.9591836734693877, adequate ordering sample True.
  - transit: RMSE 0.12697, bias 0.08970; informative same-h pairs 11/25, equal 10, unresolved 4; accuracy 1.0, adequate ordering sample False.
  - bypass: RMSE 0.05525, bias 0.03093; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 1.0, adequate ordering sample True.
- late: RMSE 0.03611 (95% CI [0.02557, 0.04753]), bias 0.01601 (CI [0.0062, 0.02568]), range excess 0.06297, range violation fraction 0.611, maximum MC SE 0.02946.
  - approach: RMSE 0.05582, bias 0.03077; informative same-h pairs 29/66, equal 36, unresolved 1; accuracy 0.9655172413793104, adequate ordering sample True.
  - transit: RMSE 0.01730, bias 0.00808; informative same-h pairs 0/25, equal 25, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.02227, bias 0.00918; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

balanced_s0:

- early: RMSE 0.07059 (95% CI [0.05933, 0.08167]), bias -0.05727 (CI [-0.06747, -0.04812]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03170.
  - approach: RMSE 0.02671, bias -0.01782; informative same-h pairs 0/66, equal 0, unresolved 66; accuracy None, adequate ordering sample False.
  - transit: RMSE 0.08388, bias -0.07147; informative same-h pairs 1/25, equal 0, unresolved 24; accuracy 1.0, adequate ordering sample False.
  - bypass: RMSE 0.08485, bias -0.08254; informative same-h pairs 29/66, equal 0, unresolved 37; accuracy 0.9310344827586207, adequate ordering sample True.
- middle: RMSE 0.10699 (95% CI [0.08749, 0.12865]), bias -0.06462 (CI [-0.08809, -0.03715]), range excess 0.02073, range violation fraction 0.028, maximum MC SE 0.04127.
  - approach: RMSE 0.12987, bias -0.09692; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 0.9591836734693877, adequate ordering sample True.
  - transit: RMSE 0.10534, bias -0.02246; informative same-h pairs 11/25, equal 10, unresolved 4; accuracy 0.9090909090909091, adequate ordering sample False.
  - bypass: RMSE 0.07987, bias -0.07449; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 0.9795918367346939, adequate ordering sample True.
- late: RMSE 0.03498 (95% CI [0.02562, 0.04428]), bias 0.00106 (CI [-0.01008, 0.01163]), range excess 0.07285, range violation fraction 0.361, maximum MC SE 0.02946.
  - approach: RMSE 0.05356, bias 0.01299; informative same-h pairs 29/66, equal 36, unresolved 1; accuracy 0.9655172413793104, adequate ordering sample True.
  - transit: RMSE 0.01764, bias -0.00828; informative same-h pairs 0/25, equal 25, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.02216, bias -0.00153; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

uniform_s1:

- early: RMSE 0.05750 (95% CI [0.04567, 0.06811]), bias 0.00918 (CI [-0.00627, 0.02483]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03170.
  - approach: RMSE 0.06480, bias 0.04815; informative same-h pairs 0/66, equal 0, unresolved 66; accuracy None, adequate ordering sample False.
  - transit: RMSE 0.06936, bias 0.00145; informative same-h pairs 1/25, equal 0, unresolved 24; accuracy 1.0, adequate ordering sample False.
  - bypass: RMSE 0.03014, bias -0.02206; informative same-h pairs 29/66, equal 0, unresolved 37; accuracy 0.9310344827586207, adequate ordering sample True.
- middle: RMSE 0.08913 (95% CI [0.06524, 0.11296]), bias -0.03661 (CI [-0.05907, -0.01085]), range excess 0.03042, range violation fraction 0.056, maximum MC SE 0.04127.
  - approach: RMSE 0.11813, bias -0.07967; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 0.9387755102040817, adequate ordering sample True.
  - transit: RMSE 0.09299, bias -0.01046; informative same-h pairs 11/25, equal 10, unresolved 4; accuracy 1.0, adequate ordering sample False.
  - bypass: RMSE 0.03506, bias -0.01969; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 0.7551020408163265, adequate ordering sample True.
- late: RMSE 0.02573 (95% CI [0.01832, 0.03263]), bias 0.00767 (CI [-0.00011, 0.01551]), range excess 0.05075, range violation fraction 0.500, maximum MC SE 0.02946.
  - approach: RMSE 0.03712, bias 0.01284; informative same-h pairs 29/66, equal 36, unresolved 1; accuracy 1.0, adequate ordering sample True.
  - transit: RMSE 0.01201, bias 0.00313; informative same-h pairs 0/25, equal 25, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.02154, bias 0.00703; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

balanced_s1:

- early: RMSE 0.03517 (95% CI [0.02795, 0.04195]), bias 0.01361 (CI [0.00434, 0.02299]), range excess 0.00000, range violation fraction 0.000, maximum MC SE 0.03170.
  - approach: RMSE 0.04074, bias 0.03434; informative same-h pairs 0/66, equal 0, unresolved 66; accuracy None, adequate ordering sample False.
  - transit: RMSE 0.04091, bias 0.00608; informative same-h pairs 1/25, equal 0, unresolved 24; accuracy 1.0, adequate ordering sample False.
  - bypass: RMSE 0.01940, bias 0.00040; informative same-h pairs 29/66, equal 0, unresolved 37; accuracy 0.9310344827586207, adequate ordering sample True.
- middle: RMSE 0.07939 (95% CI [0.05409, 0.10634]), bias 0.01378 (CI [-0.00852, 0.03791]), range excess 0.04587, range violation fraction 0.306, maximum MC SE 0.04127.
  - approach: RMSE 0.08996, bias -0.03314; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 0.9387755102040817, adequate ordering sample True.
  - transit: RMSE 0.08476, bias 0.03176; informative same-h pairs 11/25, equal 10, unresolved 4; accuracy 1.0, adequate ordering sample False.
  - bypass: RMSE 0.06026, bias 0.04270; informative same-h pairs 49/66, equal 0, unresolved 17; accuracy 0.8367346938775511, adequate ordering sample True.
- late: RMSE 0.03814 (95% CI [0.02948, 0.04582]), bias 0.02224 (CI [0.01223, 0.03202]), range excess 0.08311, range violation fraction 0.778, maximum MC SE 0.02946.
  - approach: RMSE 0.05121, bias 0.02065; informative same-h pairs 29/66, equal 36, unresolved 1; accuracy 1.0, adequate ordering sample True.
  - transit: RMSE 0.02842, bias 0.02187; informative same-h pairs 0/25, equal 25, unresolved 0; accuracy None, adequate ordering sample False.
  - bypass: RMSE 0.03054, bias 0.02419; informative same-h pairs 0/66, equal 66, unresolved 0; accuracy None, adequate ordering sample False.

## Paired primary comparison

- Seed0: final early balanced-minus-uniform RMSE -0.00107, CI [-0.01854, 0.01525]; absolute-bias change 0.01036; seen-query early RMSE change -0.01705. Sampling-bottleneck criterion: False; later-harm flags {'middle': True, 'late': False}.
- Seed1: final early balanced-minus-uniform RMSE -0.02233, CI [-0.02993, -0.01398]; absolute-bias change 0.00443; seen-query early RMSE change -0.01619. Sampling-bottleneck criterion: False; later-harm flags {'middle': False, 'late': True}.

All metric/paired/ordering intervals are episode-clustered, with2000 replicates and group stratification for overall metrics. Target repeats are averaged first; raw RMSE includes target MC noise. Ranking matches exact remaining horizon within group and phase, requires a .01 and2.58-SE gap plus split-half sign agreement, and never uses scorer predictions to construct labels. Pair-bootstrap weights use endpoint episode multiplicities. Empty or underpowered cells do not establish ordering. These are descriptive intervals and approximate noise screens, not simultaneous confidence guarantees.

## Scope and stopping

The old evaluation set is reported only in old_diagnostic.json. Final contexts come from600 new native episodes generated after all four critics were frozen, using the verified teacher solely for context collection. The teacher has native privileged information to act; selection reads observable prefixes only, and critic training never reads hidden labels. Final value targets come from the frozen generator, not the logged teacher or native hidden-state continuations. The actor/nominal/diagonal model had no exposure to these new episodes, but model-side structural limitations still apply.

A change in the row distribution can support a sampling explanation only to the extent shown by both seeds and seen/fresh comparisons. It does not isolate architecture, finite-label noise or critic optimizer effects as unique causes. Do not infer useful ranking from aggregate RMSE or identify the true environment worst case. The diagonal law and samplewise Euclidean action-Lipschitz construction are unchanged exactly because no generator parameters changed.

Stop after this fixed comparison: no retuning, budget extension, actor/ETT training, loss change, additional experiment family or automatic push. Checkpoints and raw native input archives remain local. Shared labels, exposures, selected contexts, predictions, return samples, paired metrics, provenance and reports are saved.

## Reproduction

```powershell
python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback scripts.test_pointmaze_region_pilot scripts.test_pointmaze_early_pilot scripts.test_pointmaze_phase_sampling -v
python -m ett.pointmaze_phase_sampling prepare --out artifacts/pointmaze_region_pilot/phase_sampling_s01_v1
python -m ett.pointmaze_phase_sampling run --out artifacts/pointmaze_region_pilot/phase_sampling_s01_v1
```

Use a fresh output directory. Exact local input hashes are listed in preregistration.json.
