# Future-time averaged NCE: bounded ETT usefulness comparison

The comparison does not establish a clear improvement in critic usefulness for ETT updates.

Four critics completed 1,500 updates each on the identical 288 trajectories / 13,648 uniform-sampled rows. Both sampled baselines exactly reproduce prior checkpoints. Only the positive term changes. Actor, nominal policy and base ETT stayed frozen; zero ETT or actor updates.

## Signed candidate-minus-base surrogate differences

Values use the existing H*.95^t visitation weights; negative means more pessimistic. Intervals below are episode-bootstrap 95% intervals. Conditional paired-MC intervals and SEs are also retained in results.json. Direction screens require both intervals and split-half agreement.

- identity_plus: MC +0.01750, episode CI [+0.00019, +0.03896], conditional MC CI [-0.00518, +0.04017]; inconclusive.
  - sampled_s0: -0.01553, CI [-0.03582, +0.00270]; inconclusive.
  - averaged_s0: -0.01601, CI [-0.03623, +0.00176]; inconclusive.
  - sampled_s1: -0.01906, CI [-0.03829, -0.00286]; inconclusive.
  - averaged_s1: -0.01694, CI [-0.03547, -0.00141]; inconclusive.
- identity_minus: MC -0.00500, episode CI [-0.02786, +0.01734], conditional MC CI [-0.02646, +0.01646]; inconclusive.
  - sampled_s0: +0.01320, CI [-0.00427, +0.03345]; inconclusive.
  - averaged_s0: +0.01333, CI [-0.00343, +0.03316]; inconclusive.
  - sampled_s1: +0.01650, CI [+0.00098, +0.03633]; inconclusive.
  - averaged_s1: +0.01435, CI [-0.00019, +0.03275]; inconclusive.
- rotation_plus: MC -0.00190, episode CI [-0.03419, +0.02540], conditional MC CI [-0.02174, +0.01794]; inconclusive.
  - sampled_s0: -0.00873, CI [-0.01747, -0.00109]; inconclusive.
  - averaged_s0: -0.01073, CI [-0.02096, -0.00034]; inconclusive.
  - sampled_s1: -0.00631, CI [-0.01958, +0.00732]; inconclusive.
  - averaged_s1: -0.00760, CI [-0.02091, +0.00669]; inconclusive.
- rotation_minus: MC -0.00900, episode CI [-0.03614, +0.01441], conditional MC CI [-0.03472, +0.01671]; inconclusive.
  - sampled_s0: +0.00525, CI [-0.00101, +0.01219]; inconclusive.
  - averaged_s0: +0.00764, CI [-0.00193, +0.01677]; inconclusive.
  - sampled_s1: +0.00314, CI [-0.00954, +0.01499]; inconclusive.
  - averaged_s1: +0.00533, CI [-0.00794, +0.01734]; inconclusive.

## Paired error and calibration

- Seed 0: signed-difference MAE sampled 0.01808, averaged 0.01933; change +0.00125, CI [-0.00231, +0.00503].
- Seed 1: signed-difference MAE sampled 0.01865, averaged 0.01846; change -0.00019, CI [-0.00344, +0.00327].
- sampled_s0: 0 correct, 0 wrong, 0 resolved predictions among 0 distinguishable MC differences (four probes total).
- averaged_s0: 0 correct, 0 wrong, 0 resolved predictions among 0 distinguishable MC differences (four probes total).
- sampled_s1: 0 correct, 0 wrong, 0 resolved predictions among 0 distinguishable MC differences (four probes total).
- averaged_s1: 0 correct, 0 wrong, 0 resolved predictions among 0 distinguishable MC differences (four probes total).

Root calibration (normalized return Q):
- sampled_s0: RMSE 0.06619, bias +0.04242, range excess 0.00000; maximum target MC SE 0.03081.
- averaged_s0: RMSE 0.05623, bias +0.01676, range excess 0.00000; maximum target MC SE 0.03081.
- sampled_s1: RMSE 0.05814, bias +0.00921, range excess 0.00000; maximum target MC SE 0.03081.
- averaged_s1: RMSE 0.05471, bias +0.00641, range excess 0.00000; maximum target MC SE 0.03081.
- Seed 0 averaged-minus-sampled RMSE -0.00996, CI [-0.01640, -0.00299].
- Seed 1 averaged-minus-sampled RMSE -0.00343, CI [-0.01071, +0.00445].

Visitation calibration (normalized return Q):
- sampled_s0: RMSE 0.07550, bias -0.00503, range excess 0.07008; maximum target MC SE 0.02272.
- averaged_s0: RMSE 0.08099, bias -0.01363, range excess 0.04835; maximum target MC SE 0.02272.
- sampled_s1: RMSE 0.09017, bias -0.03465, range excess 0.02212; maximum target MC SE 0.02272.
- averaged_s1: RMSE 0.07787, bias -0.02387, range excess 0.03231; maximum target MC SE 0.02272.
- Seed 0 averaged-minus-sampled RMSE +0.00550, CI [-0.00581, +0.01725].
- Seed 1 averaged-minus-sampled RMSE -0.01230, CI [-0.02098, -0.00446].

## Validity and limits

All five kernels passed exact samplewise diagonal equality (maximum drift 0), geometry/history checks, and the L=1 action test (maximum observed ratio 0.02500; tolerance 2e-6). All perturbations have zero diagonal offsets; common-anchor convex projection preserves the diagonal law and the global action-Lipschitz construction. These are model-side properties, not native physical validity.

Averaging removes conditional future-time Bernoulli label variance (row mean p(1-p) 0.02170) exactly. It does not remove finite-trajectory noise, optimization error, function approximation error, coverage limitations or generator misspecification. Two initializations share one training dataset; this experiment cannot estimate training-trajectory variability.

The 36 episode roots are held out of critic training but were evaluated in the prior experiment. All visitation paths and MC targets here are fresh and excluded from training. Each candidate acts only on the first transition; every continuation uses the frozen base and actor. The primary target is the same model surrogate, not a native-environment return or a full candidate rollout. Episode intervals are descriptive, include visitation variation, and are not simultaneous guarantees. Four correlated small probes and unresolved effects do not establish general usefulness or equivalence.

Budget: 753,965/1,000,000 new model transitions; 6,000/6,000 critic updates; zero native steps. No ETT optimization, actor training, further sweep, or budget extension occurred.

Protocol/config/source/input hashes were sealed before training and evaluation. Raw continuations, queries, probabilities, checkpoints, predictions, metrics and validity checks are saved beside this report.

Reproduce with a fresh directory using `python -m ett.pointmaze_future_average prepare --out <directory>`, then `python -m ett.pointmaze_future_average run --out <directory>`. Tests: `python -m unittest scripts.test_pointmaze_future_average -v`.
