# PointMaze region-NCE integration pilot

This is model-continuation evidence, not native fixed-actor performance, a certified global worst case, or causal identification.

Training status: complete; 24 ETT updates. Final iterates only. Charged model outputs: 742,016/1,500,000.

The inherited rectangle generator is unchanged except for 16 trainable diagonal-head bias offsets and 32 response parameters. Actor and state-goal nominal policy are frozen. The emitted XY energy score fits the diagonal; pre-projection mixed NLL is a separate diagnostic. Samplewise Euclidean action L=1 implies a coupled Wasserstein bound, not total variation. The old geometry excludes some valid fork transitions and does not guarantee native physics.

The adapter classifies sampled future reward-region membership with binary NCE, q=(0.5,0.5), B=32, alpha=0, and remaining horizon h. Its decoded occupancy is (1-0.95^h)*31*0.5*exp(f_1), with Q_0=0. It is separate from the production full-F4 point-goal critic. Values are normalized by (1-gamma)=0.05; unnormalized task returns are 20 times larger.

## Independent same-generator calibration and final models

### initial

Model occupancy 0.219059; zero/max return fractions 0.386/0.501. Diagonal energy score 0.013117; pre-projection NLL -5.902825. Constraints passed: True; maximum numerical Lipschitz excess 0.

Mean per-coordinate final XY repeat standard deviation 0.38229; mean step distance 0.19067.

last_training_critic (gate passed: True):

- h=10: RMSE 0.03541, bias 0.01307, outside-region RMSE 0.04534610401581918, maximum target MC SE 0.03124, predicted range excess 0.05808.
- h=5: RMSE 0.01803, bias 0.00072, outside-region RMSE 0.0245554046796814, maximum target MC SE 0.02014, predicted range excess 0.01832.
- h=1: RMSE 0.00507, bias 0.00080, outside-region RMSE 0.006598620697868398, maximum target MC SE 0.00625, predicted range excess 0.01474.

independent_refit (gate passed: True):

- h=10: RMSE 0.02537, bias 0.00786, outside-region RMSE 0.031047205191971378, maximum target MC SE 0.03124, predicted range excess 0.06079.
- h=5: RMSE 0.01239, bias -0.00011, outside-region RMSE 0.015212668663450283, maximum target MC SE 0.02014, predicted range excess 0.02036.
- h=1: RMSE 0.00328, bias 0.00041, outside-region RMSE 0.0039004666009520477, maximum target MC SE 0.00625, predicted range excess 0.00891.

### diagonal_s0

Model occupancy 0.219195; zero/max return fractions 0.386/0.501. Diagonal energy score 0.013272; pre-projection NLL -5.756563. Constraints passed: True; maximum numerical Lipschitz excess 0.

Mean per-coordinate final XY repeat standard deviation 0.38511; mean step distance 0.19068.

last_training_critic (gate passed: True):

- h=10: RMSE 0.01970, bias 0.00270, outside-region RMSE 0.02732314814214911, maximum target MC SE 0.03124, predicted range excess 0.01241.
- h=5: RMSE 0.00665, bias -0.00113, outside-region RMSE 0.009385022422123043, maximum target MC SE 0.02014, predicted range excess 0.00760.
- h=1: RMSE 0.00219, bias -0.00034, outside-region RMSE 0.0031866050554597175, maximum target MC SE 0.00625, predicted range excess 0.00159.

independent_refit (gate passed: True):

- h=10: RMSE 0.02933, bias 0.01033, outside-region RMSE 0.037274721666497576, maximum target MC SE 0.03124, predicted range excess 0.06331.
- h=5: RMSE 0.01039, bias 0.00058, outside-region RMSE 0.011341897033545881, maximum target MC SE 0.02014, predicted range excess 0.02196.
- h=1: RMSE 0.00293, bias 0.00057, outside-region RMSE 0.0031694845807256596, maximum target MC SE 0.00625, predicted range excess 0.01155.

### joint_s0

Model occupancy 0.213222; zero/max return fractions 0.422/0.500. Diagonal energy score 0.015862; pre-projection NLL -4.856366. Constraints passed: True; maximum numerical Lipschitz excess 0.

Mean per-coordinate final XY repeat standard deviation 0.37117; mean step distance 0.21596.

last_training_critic (gate passed: True):

- h=10: RMSE 0.02683, bias 0.00587, outside-region RMSE 0.037599281665666576, maximum target MC SE 0.02842, predicted range excess 0.01383.
- h=5: RMSE 0.00757, bias -0.00072, outside-region RMSE 0.010494401373039692, maximum target MC SE 0.02001, predicted range excess 0.00857.
- h=1: RMSE 0.00132, bias 0.00040, outside-region RMSE 0.001648336198872496, maximum target MC SE 0.00000, predicted range excess 0.00321.

independent_refit (gate passed: True):

- h=10: RMSE 0.03803, bias 0.01655, outside-region RMSE 0.048837082416639165, maximum target MC SE 0.02842, predicted range excess 0.09651.
- h=5: RMSE 0.01284, bias 0.00057, outside-region RMSE 0.012594027702050207, maximum target MC SE 0.02001, predicted range excess 0.04025.
- h=1: RMSE 0.00305, bias 0.00120, outside-region RMSE 0.003516620435261549, maximum target MC SE 0.00000, predicted range excess 0.01100.

### diagonal_s1

Model occupancy 0.219057; zero/max return fractions 0.386/0.501. Diagonal energy score 0.013149; pre-projection NLL -5.907010. Constraints passed: True; maximum numerical Lipschitz excess 0.

Mean per-coordinate final XY repeat standard deviation 0.38224; mean step distance 0.19046.

last_training_critic (gate passed: True):

- h=10: RMSE 0.02190, bias 0.01364, outside-region RMSE 0.02733154160349715, maximum target MC SE 0.03187, predicted range excess 0.03831.
- h=5: RMSE 0.00941, bias 0.00494, outside-region RMSE 0.011559509541972261, maximum target MC SE 0.01939, predicted range excess 0.01167.
- h=1: RMSE 0.00200, bias 0.00125, outside-region RMSE 0.001956679499245074, maximum target MC SE 0.00625, predicted range excess 0.00320.

independent_refit (gate passed: True):

- h=10: RMSE 0.02580, bias 0.00882, outside-region RMSE 0.03135962191448599, maximum target MC SE 0.03187, predicted range excess 0.06227.
- h=5: RMSE 0.01238, bias -0.00010, outside-region RMSE 0.015140879289444123, maximum target MC SE 0.01939, predicted range excess 0.02036.
- h=1: RMSE 0.00328, bias 0.00037, outside-region RMSE 0.003887565962197848, maximum target MC SE 0.00625, predicted range excess 0.00860.

### joint_s1

Model occupancy 0.213690; zero/max return fractions 0.418/0.500. Diagonal energy score 0.015005; pre-projection NLL -5.498485. Constraints passed: True; maximum numerical Lipschitz excess 0.

Mean per-coordinate final XY repeat standard deviation 0.37117; mean step distance 0.20859.

last_training_critic (gate passed: True):

- h=10: RMSE 0.01921, bias 0.00805, outside-region RMSE 0.0247457287520882, maximum target MC SE 0.02835, predicted range excess 0.03715.
- h=5: RMSE 0.00774, bias 0.00233, outside-region RMSE 0.009915817367316329, maximum target MC SE 0.02014, predicted range excess 0.01554.
- h=1: RMSE 0.00286, bias 0.00109, outside-region RMSE 0.00399277387619454, maximum target MC SE 0.00504, predicted range excess 0.00315.

independent_refit (gate passed: True):

- h=10: RMSE 0.03809, bias 0.01458, outside-region RMSE 0.04812489286421279, maximum target MC SE 0.02835, predicted range excess 0.08556.
- h=5: RMSE 0.02149, bias -0.00109, outside-region RMSE 0.028036371841557977, maximum target MC SE 0.02014, predicted range excess 0.03061.
- h=1: RMSE 0.00353, bias 0.00046, outside-region RMSE 0.003273794504872226, maximum target MC SE 0.00504, predicted range excess 0.01810.

## Paired update results

joint_s0: estimator-and-update success = True.

- Versus diagonal_s0: occupancy change {'mean': -0.005973308243155706, 'ci95': [-0.008858594735199508, -0.003394325175188031], 'clusters': 64}; diagonal ES change {'mean': 0.0025898595340549946, 'ci95': [0.0020407903762217554, 0.003172289673340834], 'clusters': 311}; diagonal noninferiority True.
- Versus initial: occupancy change {'mean': -0.0058368245050818245, 'ci95': [-0.008666327612647951, -0.0032495090046331986], 'clusters': 64}; diagonal ES change {'mean': 0.002744926605373621, 'ci95': [0.002187901151325079, 0.003329423186703145], 'clusters': 311}; diagonal noninferiority True.

joint_s1: estimator-and-update success = True.

- Versus diagonal_s1: occupancy change {'mean': -0.00536733107637505, 'ci95': [-0.008019139266251606, -0.0029257604498202877], 'clusters': 64}; diagonal ES change {'mean': 0.0018557028379291296, 'ci95': [0.0015153392392318976, 0.00220226872702458], 'clusters': 311}; diagonal noninferiority True.
- Versus initial: occupancy change {'mean': -0.005368950755186511, 'ci95': [-0.00792722379093805, -0.0029800653153962335], 'clusters': 64}; diagonal ES change {'mean': 0.0018874285742640495, 'ci95': [0.0016124816478286037, 0.0021813381150979304], 'clusters': 311}; diagonal noninferiority True.

## Native-label audits and limitations

Native death timing is reconstructed only after training, using logged hidden bits and episode flags. Those fields never enter context selection, critic inputs, labels for fitting, or ETT updates. A stationary atom is not model death; the generator has no persistent death state. Alive outside-goal contexts are only recovery opportunities, not certified recoverable states.

- initial: audit-only context results: {"already_dead": {"count": 32, "model_value": 0.036854216301093964, "model_any_goal_probability": 0.228515625, "model_any_motion_probability": 0.44140625, "native_logged_behavior_value": 0.0}, "alive_outside": {"count": 0, "model_value": null, "model_any_goal_probability": null, "model_any_motion_probability": null, "native_logged_behavior_value": null}, "stationary_outside": {"count": 32, "model_value": 0.036854216301093964, "model_any_goal_probability": 0.228515625, "model_any_motion_probability": 0.44140625, "native_logged_behavior_value": 0.0}, "inside": {"count": 32, "model_value": 0.40126306674090634, "model_any_goal_probability": 1.0, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.4012630667409065}}
- diagonal_s0: audit-only context results: {"already_dead": {"count": 32, "model_value": 0.03712718377724173, "model_any_goal_probability": 0.228515625, "model_any_motion_probability": 0.44140625, "native_logged_behavior_value": 0.0}, "alive_outside": {"count": 0, "model_value": null, "model_any_goal_probability": null, "model_any_motion_probability": null, "native_logged_behavior_value": null}, "stationary_outside": {"count": 32, "model_value": 0.03712718377724173, "model_any_goal_probability": 0.228515625, "model_any_motion_probability": 0.44140625, "native_logged_behavior_value": 0.0}, "inside": {"count": 32, "model_value": 0.40126306674090634, "model_any_goal_probability": 1.0, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.4012630667409065}}
- joint_s0: audit-only context results: {"already_dead": {"count": 32, "model_value": 0.025180567290930315, "model_any_goal_probability": 0.15625, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.0}, "alive_outside": {"count": 0, "model_value": null, "model_any_goal_probability": null, "model_any_motion_probability": null, "native_logged_behavior_value": null}, "stationary_outside": {"count": 32, "model_value": 0.025180567290930315, "model_any_goal_probability": 0.15625, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.0}, "inside": {"count": 32, "model_value": 0.40126306674090634, "model_any_goal_probability": 1.0, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.4012630667409065}}
- diagonal_s1: audit-only context results: {"already_dead": {"count": 32, "model_value": 0.036850976943471045, "model_any_goal_probability": 0.228515625, "model_any_motion_probability": 0.44140625, "native_logged_behavior_value": 0.0}, "alive_outside": {"count": 0, "model_value": null, "model_any_goal_probability": null, "model_any_motion_probability": null, "native_logged_behavior_value": null}, "stationary_outside": {"count": 32, "model_value": 0.036850976943471045, "model_any_goal_probability": 0.228515625, "model_any_motion_probability": 0.44140625, "native_logged_behavior_value": 0.0}, "inside": {"count": 32, "model_value": 0.40126306674090634, "model_any_goal_probability": 1.0, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.4012630667409065}}
- joint_s1: audit-only context results: {"already_dead": {"count": 32, "model_value": 0.026116314790720938, "model_any_goal_probability": 0.1640625, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.0}, "alive_outside": {"count": 0, "model_value": null, "model_any_goal_probability": null, "model_any_motion_probability": null, "native_logged_behavior_value": null}, "stationary_outside": {"count": 32, "model_value": 0.026116314790720938, "model_any_goal_probability": 0.1640625, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.0}, "inside": {"count": 32, "model_value": 0.40126306674090634, "model_any_goal_probability": 1.0, "model_any_motion_probability": 1.0, "native_logged_behavior_value": 0.4012630667409065}}

F4 histories shift exactly and both action channels remain bounded. Complete 10-step model continuations end at original task time 50. Logged subsequent behavior returns are supplementary native evidence from a different continuation policy, not hidden-context conditional ETT ground truth. Root selection is late-task and balanced inside/outside the goal region; results do not estimate reset returns. Actor training overlaps these source episodes. Confidence intervals cluster contexts/diagonal rows by episode and pair common random streams; they do not imply identical trajectories for different kernels.

Recommendation: stop at this pilot. The paired results above determine whether an estimator-and-update effect was established; even a passing seed does not justify actor integration or a global pessimism claim.

## Reproduction

Run from the PointMaze worktree with the local input checkpoints/dataset listed and hashed in provenance.json. Checkpoints remain local; configuration, prediction arrays, audit arrays and reports are shareable.

```powershell
python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback scripts.test_pointmaze_region_pilot -v
python -m ett.pointmaze_region_pilot prepare --out artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1
python -m ett.pointmaze_region_pilot run --out artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1
python -m ett.eval_pointmaze_region_pilot --out artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1
```

Use a fresh output directory for a literal reproduction; existing attempts cannot be overwritten. PROTOCOL.md was saved and hashed before outcomes. results.json and *_evaluation.npz contain full metrics, trajectories, execution actions, separate auxiliary x_prime draws, and common-target last/refit predictions.
