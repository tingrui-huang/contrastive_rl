# Early-context PointMaze region-NCE pilot

Stopping stage: **value_estimation**. Charged model outputs: **142,961/1,500,000**.

The frozen actor, state-goal nominal policy, shared 48-parameter convex generator, region reward and two losses are unchanged from 0059c98. Horizon is exactly 50 minus recorded root time; Q_h=(1-.95^h)*31*.5*exp(f_1), Q_0=0. The critic horizon feature is now h/50. Discounted visitation weights are H_i*.95^t.

Samplewise Euclidean action L=1 is preserved by the existing construction, with a common-x_prime Wasserstein coupling. No TV constraint, death penalty, classifier, or new geometric restriction was added. The inherited geometry and visible-only state do not guarantee native physical validity or persistent absorption.

## Predeclared gates

Coverage passed: True. Selection uses observable prefixes only, one root per episode, no resampling. Hidden labels only supplied the separate aggregate stop gate.

- train/approach: 12 roots; 12 alive at observation; times [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
- train/transit: 12 roots; 12 alive at observation; times [3, 3, 6, 3, 3, 3, 10, 3, 3, 3, 3, 3].
- train/bypass: 12 roots; 12 alive at observation; times [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3].
- validation/approach: 12 roots; 12 alive at observation; times [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
- validation/transit: 12 roots; 12 alive at observation; times [3, 3, 6, 3, 3, 6, 6, 3, 3, 10, 3, 6].
- validation/bypass: 12 roots; 12 alive at observation; times [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3].

Generator check passed: True. Mixed-outcome roots: {'approach': 3, 'transit': 2, 'bypass': 0}; recovered-after-setback paths: 393; initial root mean-value SD: 0.05899.

- initial/approach: model occupancy 0.54948, finite-horizon zero fraction 0.047, recovered-after-setback fraction 0.255.
- initial/transit: model occupancy 0.64829, finite-horizon zero fraction 0.062, recovered-after-setback fraction 0.151.
- initial/bypass: model occupancy 0.61805, finite-horizon zero fraction 0.000, recovered-after-setback fraction 0.010.
- positive_response/approach: model occupancy 0.55967, finite-horizon zero fraction 0.016, recovered-after-setback fraction 0.307.
- positive_response/transit: model occupancy 0.64038, finite-horizon zero fraction 0.005, recovered-after-setback fraction 0.271.
- positive_response/bypass: model occupancy 0.62364, finite-horizon zero fraction 0.000, recovered-after-setback fraction 0.010.
- negative_response/approach: model occupancy 0.58237, finite-horizon zero fraction 0.000, recovered-after-setback fraction 0.599.
- negative_response/transit: model occupancy 0.69142, finite-horizon zero fraction 0.000, recovered-after-setback fraction 0.375.
- negative_response/bypass: model occupancy 0.60467, finite-horizon zero fraction 0.000, recovered-after-setback fraction 0.068.

These fixed probes are not an exhaustive family search or a capacity certificate. A zero-return path is not death. Recovery labels describe realized paths only and never train the critic.

Initial critic calibration passed: False.

- root/all: RMSE 0.07950, bias 0.04974, maximum MC SE 0.07900, range excess 0.00000.
- root/approach: RMSE 0.06141, bias 0.05529, maximum MC SE 0.06834, range excess 0.00000.
- root/transit: RMSE 0.11278, bias 0.06214, maximum MC SE 0.07900, range excess 0.00000.
- root/bypass: RMSE 0.04970, bias 0.03177, maximum MC SE 0.00996, range excess 0.00000.
- midpoint/all: RMSE 0.01788, bias 0.00958, maximum MC SE 0.04395, range excess 0.03881.
- midpoint/approach: RMSE 0.01821, bias 0.00779, maximum MC SE 0.04395, range excess 0.03043.
- midpoint/transit: RMSE 0.01835, bias 0.01314, maximum MC SE 0.00000, range excess 0.03841.
- midpoint/bypass: RMSE 0.01706, bias 0.00782, maximum MC SE 0.00000, range excess 0.03881.
- last/all: RMSE 0.00185, bias 0.00048, maximum MC SE 0.00000, range excess 0.00278.
- last/approach: RMSE 0.00223, bias -0.00022, maximum MC SE 0.00000, range excess 0.00151.
- last/transit: RMSE 0.00178, bias 0.00095, maximum MC SE 0.00000, range excess 0.00278.
- last/bypass: RMSE 0.00147, bias 0.00073, maximum MC SE 0.00000, range excess 0.00262.

## Native evidence and interpretation

- approach: {'count': 12, 'already_dead': 0, 'later_native_death': 4, 'logged_goal_count': 8, 'logged_value': 0.43425164694776397}
- transit: {'count': 12, 'already_dead': 0, 'later_native_death': 0, 'logged_goal_count': 12, 'logged_value': 0.7257026226180544}
- bypass: {'count': 12, 'already_dead': 0, 'later_native_death': 0, 'logged_goal_count': 12, 'logged_value': 0.626969770771101}

Native logged goal attainment, later death and finite-horizon zero reward are separate events. Logged behavior is not the fixed continuation actor. Hidden labels did not train any scorer or select checkpoints. No new native simulator evaluation was run. Model predictions and logged hidden-context evidence are not conditional ETT ground truth.

The declared gate stopped before optimization. The remaining immediate limitation is value estimation. No later optimization result is claimed.

No budget expansion, actor training, loss change, extra experiment family, commit or push. Arrays retain every probe/evaluation trajectory; absent variable-horizon slots are explicitly NaN and never included in a return. Configurations, hashes and the pre-run protocol are saved beside this report.

## Reproduction

```powershell
python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback scripts.test_pointmaze_region_pilot scripts.test_pointmaze_early_pilot -v
python -m ett.pointmaze_early_pilot prepare --out artifacts/pointmaze_region_pilot/early_contexts_s01_v1
python -m ett.pointmaze_early_pilot run --out artifacts/pointmaze_region_pilot/early_contexts_s01_v1
```

Use a fresh output directory for reproduction; the saved attempt is protected from overwrite. Local input checkpoints/dataset must match preregistration.json.
