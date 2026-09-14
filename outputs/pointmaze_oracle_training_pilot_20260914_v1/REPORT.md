# PointMaze oracle-motion end-to-end training pilot

## Answer

Oracle-model augmentation did not improve native return over both the starting actor and equally budgeted offline-only training in both seeds.

The comparison is exploratory and conditional on two training seeds. A is the unchanged starting actor; B is 1,000 updates of offline-only continuation; C has the identical update budget but replaces exactly 10% of batch rows with coherent paths from exact native alive motion plus the corresponding frozen repaired failure head. The native primary outcome uses 200 fresh paired episodes per final policy.

For seed 0, C-minus-A native return was +0.397 [-0.121, +0.930] and C-minus-B was -0.576 [-1.021, -0.129]. For seed 1, the corresponding differences were +1.010 [+0.309, +1.740] and +0.162 [-0.345, +0.681]. Intervals are paired episode-bootstrap 95% intervals. A point improvement over A alone is not attributed to generated data; C-versus-B is the added-data contrast.

## Native final-policy results

| Policy | Return | Failure | Reward occurrence | Strict success | First hazard | Exposure |
|---|---:|---:|---:|---:|---:|---:|
| A | 3.292 [2.591, 4.025] | 0.700 [0.635, 0.760] | 0.300 [0.240, 0.365] | 0.300 [0.240, 0.365] | 0.860 [0.810, 0.905] | 2.130 [1.920, 2.340] |
| B_s0 | 4.265 [3.494, 5.069] | 0.625 [0.555, 0.690] | 0.375 [0.310, 0.445] | 0.375 [0.310, 0.445] | 0.850 [0.795, 0.895] | 2.245 [2.020, 2.480] |
| C_s0 | 3.690 [2.960, 4.497] | 0.685 [0.615, 0.745] | 0.315 [0.255, 0.385] | 0.315 [0.255, 0.385] | 0.890 [0.845, 0.930] | 2.290 [2.090, 2.500] |
| B_s1 | 4.141 [3.334, 4.896] | 0.645 [0.580, 0.715] | 0.355 [0.285, 0.420] | 0.355 [0.285, 0.420] | 0.865 [0.815, 0.910] | 2.225 [2.020, 2.430] |
| C_s1 | 4.303 [3.475, 5.100] | 0.640 [0.575, 0.710] | 0.360 [0.290, 0.425] | 0.360 [0.290, 0.425] | 0.885 [0.840, 0.930] | 2.255 [2.055, 2.460] |

Reward occurrence uses the native radius-2 task reward. Strict success is physical distance below 0.5 and is deliberately separate. Failure is the native absorbing death flag. First hazard and exposure count actual hazardous landings while the policy is still at risk, including a fatal landing.

Seed 0 paired contrasts:

- `C_s0_minus_A`: return +0.397 [-0.121, +0.930]; failure -0.015 [-0.060, +0.035]; reward +0.015 [-0.035, +0.060]; strict success +0.015 [-0.035, +0.060]; first hazard +0.030 [-0.005, +0.065]; exposure +0.160 [+0.005, +0.325].
- `C_s0_minus_B_s0`: return -0.576 [-1.021, -0.129]; failure +0.060 [+0.020, +0.100]; reward -0.060 [-0.100, -0.020]; strict success -0.060 [-0.100, -0.020]; first hazard +0.040 [+0.010, +0.075]; exposure +0.045 [-0.080, +0.185].
- `B_s0_minus_A`: return +0.973 [+0.400, +1.550]; failure -0.075 [-0.125, -0.025]; reward +0.075 [+0.025, +0.125]; strict success +0.075 [+0.025, +0.125]; first hazard -0.010 [-0.040, +0.020]; exposure +0.115 [-0.065, +0.305].

Seed 1 paired contrasts:

- `C_s1_minus_A`: return +1.010 [+0.309, +1.740]; failure -0.060 [-0.125, +0.000]; reward +0.060 [+0.000, +0.125]; strict success +0.060 [+0.000, +0.125]; first hazard +0.025 [-0.005, +0.060]; exposure +0.125 [-0.060, +0.320].
- `C_s1_minus_B_s1`: return +0.162 [-0.345, +0.681]; failure -0.005 [-0.050, +0.040]; reward +0.005 [-0.040, +0.050]; strict success +0.005 [-0.040, +0.050]; first hazard +0.020 [+0.005, +0.040]; exposure +0.030 [-0.080, +0.140].
- `B_s1_minus_A`: return +0.849 [+0.191, +1.485]; failure -0.055 [-0.115, +0.005]; reward +0.055 [-0.005, +0.115]; strict success +0.055 [-0.005, +0.115]; first hazard +0.005 [-0.030, +0.040]; exposure +0.095 [-0.095, +0.285].

The 256 predictive paths in each model cell were not used as native sample size. Two training seeds cannot establish across-seed robustness, and a zero-crossing interval is not evidence of equivalence.

## Learning inside the frozen generated environment

Head/configuration 0:

- `A`: predicted return 3.920 [3.285, 4.590]; failure 0.664 [0.605, 0.719]; reward 0.348 [0.293, 0.406]; first hazard 0.910 [0.875, 0.941]; exposure 2.508 [2.273, 2.762].
- `B_s0`: predicted return 4.742 [4.039, 5.482]; failure 0.602 [0.543, 0.660]; reward 0.414 [0.355, 0.473]; first hazard 0.906 [0.871, 0.938]; exposure 2.578 [2.340, 2.832].
- `C_s0`: predicted return 4.520 [3.775, 5.281]; failure 0.617 [0.555, 0.676]; reward 0.391 [0.332, 0.453]; first hazard 0.930 [0.898, 0.957]; exposure 2.539 [2.320, 2.770].
- `s0/C_s0_minus_A`: predicted-return difference +0.600 [-0.022, +1.170]; failure difference -0.047 [-0.098, +0.008].
- `s0/C_s0_minus_B_s0`: predicted-return difference -0.222 [-0.652, +0.215]; failure difference +0.016 [-0.023, +0.055].

Head/configuration 1:

- `A`: predicted return 3.594 [2.961, 4.257]; failure 0.680 [0.621, 0.734]; reward 0.328 [0.270, 0.387]; first hazard 0.898 [0.859, 0.934]; exposure 2.430 [2.219, 2.652].
- `B_s1`: predicted return 4.133 [3.470, 4.824]; failure 0.641 [0.582, 0.695]; reward 0.371 [0.312, 0.430]; first hazard 0.879 [0.836, 0.918]; exposure 2.305 [2.117, 2.516].
- `C_s1`: predicted return 4.100 [3.455, 4.832]; failure 0.652 [0.594, 0.707]; reward 0.352 [0.297, 0.410]; first hazard 0.910 [0.875, 0.945]; exposure 2.367 [2.176, 2.559].
- `s1/C_s1_minus_A`: predicted-return difference +0.506 [+0.023, +0.971]; failure difference -0.027 [-0.066, +0.016].
- `s1/C_s1_minus_B_s1`: predicted-return difference -0.033 [-0.366, +0.294]; failure difference +0.012 [-0.020, +0.043].

These are actual full oracle trajectories, not decoded critic logits. The predicted return and failure estimates use the same exact native-motion helper, repaired head, nominal, support, and absorption used to generate C's replay. Any gain here shows improvement inside that frozen generated environment only. A rise here without native improvement is model exploitation or transfer mismatch evidence, not native actor improvement.

There is no evidence of a generated-environment-only advantage over offline continuation: C-minus-B predicted return was -0.222 [-0.652, +0.215] for seed 0 and -0.033 [-0.366, +0.294] for seed 1. Seed 0 is directionally worse than B both in its oracle model and, more decisively, natively. The failed pilot therefore is not the specific pattern of a rising oracle-model return that fails to transfer. It instead localizes the immediate problem to the generated-data/learner interaction or lack of marginal value at this fixed 10% mixture and budget; it does not identify a unique cause.

## Training and data-path checks

All four trained policies received exactly 1,000 joint actor/critic updates from byte-identical complete initialization within seed. B and C used identical full offline draws and batch permutations; C replaced 25,600 of 256,000 rows and refreshed 128 current-policy paths at updates 0, 250, 500, and 750. There was no critic warm-up because the complete trained critic, target critic, and optimizer states were restored.

- `B_s0`: actor/critic parameter L2 1.2812/1.3816; initial-to-final KL 0.0880; mean-action RMS/max drift 0.1114/0.7515; scale change +0.0863; generated rows 0; critic loss first/last 100 0.01886/0.01878.
- `C_s0`: actor/critic parameter L2 1.3742/1.5829; initial-to-final KL 0.0962; mean-action RMS/max drift 0.1382/0.9633; scale change +0.0518; generated rows 25,600; critic loss first/last 100 0.01892/0.01873.
- `B_s1`: actor/critic parameter L2 1.3193/1.3973; initial-to-final KL 0.0589; mean-action RMS/max drift 0.1080/0.5183; scale change +0.0432; generated rows 0; critic loss first/last 100 0.01875/0.01884.
- `C_s1`: actor/critic parameter L2 1.4573/1.6743; initial-to-final KL 0.0979; mean-action RMS/max drift 0.2263/1.5374; scale change +0.0478; generated rows 25,600; critic loss first/last 100 0.01871/0.01881.

All losses and gradients were finite, and both actor and critic parameters changed in every run. Every generated replay pair used `j>i` within its true finite path length; padding was never sampled. Absorbing tails were retained under the existing fixed-horizon objective. Failed XY stayed frozen, F4 shifted, generated reward remained zero, and no failed future observation lay in the commanded task reward radius. The internal failure flag never entered the actor's eight-dimensional F4 observation.

The bounded saved-trace inspection found one concrete adapter concern. In C0, 5,718 of 25,600 generated anchors (22.3%) were already failed before their sampled action; in C1 it was 6,891 (26.9%). Those actions have no transition consequence because XY is absorbed, yet the unchanged learner treats every sampled action as a behavior-cloning target. This did not create false commanded-goal successes, and it is not proven to cause the C-B result, but it can dilute or distort the synthetic actor update at observations whose failure status is hidden.

## Correctness and budget

The movement helper was not reimplemented or reverified by fresh simulator calls: its prior 34-call bit-exact native equivalence result and source hash were pinned and reused. New generated outcomes use no native hidden bits, native death, or native reward. Advice is sampled from the historical nominal at each generated observation; the queried action comes from that path's current actor. The repaired head alone samples onset on supported hazardous landings.

The run used 4,000 actor and 4,000 critic updates, 108,800 oracle transition slots across 2,560 paths, and 50,000 native evaluation steps. It collected zero native training episodes. Final checkpoints were selected by the fixed 1,000-update budget and hashed before any native outcome was generated. Independent saved-array verification status: `passed`.

## What this establishes and what it does not

Implementation correctness is supported by the pinned movement equivalence, replay/absorption checks, matched initialization and update accounting. Whether C learned inside the generated environment is shown by the predictive C-A/C-B rows above. Native improvement is shown only by the paired native contrasts, and additional benefit is specifically C-B.

This pilot uses supervised failure labels upstream and exact environment motion. It does not validate the original learned alive-motion block, observational identification, the original second/pessimistic loss, certified worst-case optimization, or AntMaze. It also does not establish general effectiveness over training seeds.

## Recommended next action

Do not extend or sweep this pilot, and do not move to learned motion yet. The one supported next intervention is a preregistered same-budget replay-adapter comparison that keeps the fatal incoming anchor and the full absorbed tail as eligible future goals, but makes transitions with `failed_before=true` ineligible as synthetic anchors and behavior-cloning targets. The repository's anchor-cut mechanism can express this as `cut = first_onset_time + 1` while leaving the future-goal window intact. This directly targets the observed 22-27% causally ignored-action contamination without changing the frozen transition, head, nominal, learner objective, update count, or offline control. Because it uses the generated supervised failure state, it would remain an oracle-assisted diagnostic rather than validation of the original learned ETT method.

## Reproduction

The sealed phases are `python outputs/pointmaze_oracle_training_pilot_20260914_v1/run.py prepare`, then `train`, then `evaluate`. Recompute all saved results with `verify_saved.py`; `analyze.py` regenerates the numerical base report, while `decision.json` preserves the final bounded interpretation and intervention recommendation. Historical artifacts and production modules remain unchanged.
