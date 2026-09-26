# Pretrain (recorded data) -> branch replay: critics and actors against the random-init recipe

Sealed 2026-09-18 08:37:31.  Arms: T = vanilla 100k -> 30k NCE on the R replay; C1 = random init -> 30k NCE on the R replay (existing critics_armR); C2 = vanilla 100k only.  Actors: pure-BC init, frozen critic, bc 0.05, 30k, 300-episode mean-policy evaluation.  This is an added training schedule (the loss is unchanged).

## Cost (wall seconds from the run manifests)

| arm | stage | seed 0 / 1 / 2 |
|---|---|---|
| T | critic (critics_armT) | 361 / 359 / 357 |
| T | actor (joint_T) | 491 / 488 / 488 |
| C1 | critic (critics_armR) | 631 / 639 / 627 |
| C1 | actor (joint_C1) | 651 / 648 / 557 |
| C2 | critic (critics_pre100k_only) | 815 / 808 / 810 |
| C2 | actor (joint_C2) | 651 / 650 / 552 |
| T, C2 | pretraining (vanilla_g0999_pre100k) seed 0 | 815 |
| T, C2 | pretraining (vanilla_g0999_pre100k) seed 1 | 808 |
| T, C2 | pretraining (vanilla_g0999_pre100k) seed 2 | 810 |

## Critics on the A / B / C diagnostic (region min readout; agreement among decided pairs (s.e.); pick gain)

| arm | seed | A | B | C | C matched |
|---|---|---|---|---|---|
| T | seed_0 | 0.72 (0.036) n 230; gain +0.042 | 0.53 (0.035) n 222; gain -0.004 | 0.50 (0.026) n 758; gain +0.002 | 0.51 (0.036) n 227; gain +0.004 |
| T | seed_1 | 0.70 (0.037) n 230; gain +0.036 | 0.50 (0.039) n 222; gain -0.002 | 0.49 (0.028) n 758; gain -0.001 | 0.48 (0.043) n 227; gain +0.002 |
| T | seed_2 | 0.74 (0.034) n 230; gain +0.042 | 0.52 (0.037) n 222; gain +0.004 | 0.51 (0.027) n 758; gain +0.005 | 0.52 (0.041) n 227; gain +0.009 |
| C1 | seed_0 | 0.67 (0.038) n 230; gain +0.029 | 0.58 (0.036) n 222; gain +0.010 | 0.48 (0.034) n 758; gain -0.009 | 0.41 (0.038) n 227; gain -0.018 |
| C1 | seed_1 | 0.67 (0.036) n 230; gain +0.030 | 0.53 (0.039) n 222; gain -0.000 | 0.50 (0.023) n 758; gain +0.001 | 0.46 (0.027) n 227; gain +0.003 |
| C1 | seed_2 | 0.70 (0.038) n 230; gain +0.036 | 0.55 (0.036) n 222; gain +0.009 | 0.52 (0.028) n 758; gain +0.004 | 0.46 (0.032) n 227; gain -0.002 |
| C2 | seed_0 | 0.44 (0.042) n 230; gain -0.012 | 0.45 (0.034) n 222; gain -0.004 | 0.50 (0.025) n 758; gain +0.002 | 0.46 (0.041) n 227; gain -0.005 |
| C2 | seed_1 | 0.42 (0.040) n 230; gain -0.009 | 0.44 (0.035) n 222; gain -0.008 | 0.49 (0.023) n 758; gain +0.002 | 0.50 (0.028) n 227; gain +0.004 |
| C2 | seed_2 | 0.46 (0.038) n 230; gain -0.001 | 0.45 (0.036) n 222; gain -0.003 | 0.50 (0.021) n 758; gain -0.001 | 0.47 (0.032) n 227; gain -0.008 |

Paired episode bootstrap on B (222 decided pairs, 78 episodes): T 0.518 +- 0.028; C1 0.554 +- 0.028; C2 0.449 +- 0.025.  Differences: T - C1 -0.036 +- 0.027 (z -1.4; per seed -0.054, -0.027, -0.027); T - C2 +0.069 +- 0.037 (z 1.8; per seed +0.072, +0.068, +0.068)

Paired episode bootstrap on C (758 decided pairs, 45 episodes): T 0.499 +- 0.019; C1 0.500 +- 0.022; C2 0.498 +- 0.016.  Differences: T - C1 -0.000 +- 0.016 (z -0.0; per seed +0.024, -0.009, -0.016); T - C2 +0.001 +- 0.019 (z 0.0; per seed -0.001, -0.008, +0.012)

## Deployment evaluation of the actors (300 natural draws, mean policy)

| arm | seed | success | failure | timeout | detour | shortcut | discounted | mouth 1 / 2 median | after-burst 1 / 2 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| T | 0 | 0.213 | 0.560 | 0.227 | 0.000 | 0.753 | 0.018 | 61 / 136 | 0.03 / 0.00 |
| T | 1 | 0.233 | 0.517 | 0.250 | 0.007 | 0.850 | 0.013 | 59 / 142 | 0.10 / 0.35 |
| T | 2 | 0.103 | 0.250 | 0.647 | 0.000 | 0.353 | 0.009 | 64 / 137 | 0.06 / 0.08 |
| C1 | 0 | 0.303 | 0.507 | 0.190 | 0.007 | 0.787 | 0.021 | 70 / 151 | 0.05 / 0.09 |
| C1 | 1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | - / - | - / - |
| C1 | 2 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | - / - | - / - |
| C2 | 0 | 0.287 | 0.003 | 0.710 | 0.000 | 0.293 | 0.006 | 206 / 296 | 0.96 / 1.00 |
| C2 | 1 | 0.007 | 0.000 | 0.993 | 0.000 | 0.053 | 0.000 | 188 / 335 | 0.83 / 1.00 |
| C2 | 2 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | - / - | - / - |

Reference: pure-BC walker (the actor init) success 0.253, detour 0.000; round-1 vanilla actors success 0.26, detour 0.003-0.007.

| arm | mean success (seed s.e.) | mean detour (seed s.e.) | mean discounted |
|---|---|---|---|
| T | 0.183 (0.040) | 0.002 (0.002) | 0.013 |
| C1 | 0.101 (0.101) | 0.002 (0.002) | 0.007 |
| C2 | 0.098 (0.094) | 0.000 (0.000) | 0.002 |

## Pre-registered policy-level decision

- T - C1 success +0.082 (pooled seed s.e. 0.109; not > 2 s.e.)
- T - C2 success +0.086 (pooled seed s.e. 0.103; not > 2 s.e.)
- T detour > 0.05 on 0 of 3 seeds
