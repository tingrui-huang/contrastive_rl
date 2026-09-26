# Arm CF-learned (oracle-label-trained learned ETT) vs the clipped O and CF (oracle) arms

`manifest.json` (sealed before fitting).  The critic's positive futures of arm CFL come from the cross-fitted learned marginal (`futures_learned.npz`, `CHECK.md`); everything else is the sealed critic_clip0.1 recipe.  Same 300 evaluation episodes (seed 3909, mode) as the clipped O / CF arms.  Rule: mean over the 3 paired seeds > 2 x seed s.e. and 3/3.  Status: an engineering intermediate experiment on the interface (the ETT model saw simulator labels); not the offline identification.

## Per policy

| policy | success | detour | death | timeout | success no hazard | success hazard | mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| start | 0.243 | 0.003 | 0.740 | 0.017 | 0.986 | 0.000 | 137 |
| O/seed_0 | 0.257 | 0.027 | 0.723 | 0.020 | 0.986 | 0.018 | 140 |
| O/seed_1 | 0.247 | 0.003 | 0.743 | 0.010 | 0.986 | 0.004 | 129 |
| O/seed_2 | 0.260 | 0.023 | 0.720 | 0.020 | 1.000 | 0.018 | 141 |
| CF/seed_0 | 0.457 | 0.567 | 0.193 | 0.350 | 0.676 | 0.385 | 512 |
| CF/seed_1 | 0.513 | 0.517 | 0.307 | 0.180 | 0.784 | 0.425 | 390 |
| CF/seed_2 | 0.440 | 0.360 | 0.403 | 0.157 | 0.838 | 0.310 | 349 |
| CFL/seed_0 | 0.253 | 0.010 | 0.740 | 0.007 | 1.000 | 0.009 | 129 |
| CFL/seed_1 | 0.247 | 0.013 | 0.740 | 0.013 | 0.986 | 0.004 | 132 |
| CFL/seed_2 | 0.247 | 0.000 | 0.753 | 0.000 | 1.000 | 0.000 | 121 |

## Paired differences on the common episodes (per seed; seed mean, seed s.e., episode-bootstrap s.e.)

### success

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CFL - O [primary] | -0.003 / +0.000 / -0.013 | -0.006 | 0.004 | 0.005 | 2/3 | not met |
| CFL - start [practical] | +0.010 / +0.003 / +0.003 | +0.006 | 0.002 | 0.004 | 3/3 | met |
| CFL - CF(oracle) [kept?] | -0.203 / -0.267 / -0.193 | -0.221 | 0.023 | 0.023 | 3/3 | - |
| CF(oracle) - O [reference] | +0.200 / +0.267 / +0.180 | +0.216 | 0.026 | 0.023 | 3/3 | - |

### detour

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CFL - O [primary] | -0.017 / +0.010 / -0.023 | -0.010 | 0.010 | 0.005 | 2/3 | - |
| CFL - start [practical] | +0.007 / +0.010 / -0.003 | +0.004 | 0.004 | 0.005 | 2/3 | - |
| CFL - CF(oracle) [kept?] | -0.557 / -0.503 / -0.360 | -0.473 | 0.059 | 0.021 | 3/3 | - |
| CF(oracle) - O [reference] | +0.540 / +0.513 / +0.337 | +0.463 | 0.064 | 0.021 | 3/3 | - |

### failure

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CFL - O [primary] | +0.017 / -0.003 / +0.033 | +0.016 | 0.011 | 0.005 | 2/3 | - |
| CFL - start [practical] | +0.000 / +0.000 / +0.013 | +0.004 | 0.004 | 0.008 | 1/3 | - |
| CFL - CF(oracle) [kept?] | +0.547 / +0.433 / +0.350 | +0.443 | 0.057 | 0.023 | 3/3 | - |
| CF(oracle) - O [reference] | -0.530 / -0.437 / -0.317 | -0.428 | 0.062 | 0.023 | 3/3 | - |

### timeout

| comparison | per seed | mean | seed s.e. | boot s.e. | same direction | rule |
|---|---|---:|---:|---:|---|---|
| CFL - O [primary] | -0.013 / +0.003 / -0.020 | -0.010 | 0.007 | 0.005 | 2/3 | - |
| CFL - start [practical] | -0.010 / -0.003 / -0.017 | -0.010 | 0.004 | 0.008 | 3/3 | - |
| CFL - CF(oracle) [kept?] | -0.343 / -0.167 / -0.157 | -0.222 | 0.061 | 0.014 | 3/3 | - |
| CF(oracle) - O [reference] | +0.330 / +0.170 / +0.137 | +0.212 | 0.060 | 0.014 | 3/3 | - |

## Route ledger

| policy | far route: n (share) | completed | far-route timeouts / deaths | completion | shortcut: n / success / deaths / timeouts | no route: n / deaths / timeouts | success | success if far-route timeouts rescued |
|---|---:|---:|---|---:|---|---|---:|---:|
| start | 1 (0.00) | 0 | 1 / 0 | 0.000 | 291 / 73 / 217 / 1 | 8 / 5 / 3 | 0.243 | 0.247 |
| O/seed_0 | 8 (0.03) | 5 | 3 / 0 | 0.625 | 286 / 72 / 214 / 0 | 6 / 3 / 3 | 0.257 | 0.267 |
| O/seed_1 | 1 (0.00) | 1 | 0 / 0 | 1.000 | 290 / 73 / 217 / 0 | 9 / 6 / 3 | 0.247 | 0.247 |
| O/seed_2 | 7 (0.02) | 5 | 2 / 0 | 0.714 | 284 / 73 / 211 / 0 | 9 / 5 / 4 | 0.260 | 0.267 |
| CF/seed_0 | 170 (0.57) | 111 | 59 / 0 | 0.653 | 86 / 26 / 52 / 8 | 44 / 6 / 38 | 0.457 | 0.653 |
| CF/seed_1 | 155 (0.52) | 122 | 33 / 0 | 0.787 | 123 / 32 / 89 / 2 | 22 / 3 / 19 | 0.513 | 0.623 |
| CF/seed_2 | 108 (0.36) | 87 | 21 / 0 | 0.806 | 166 / 45 / 113 / 8 | 26 / 8 / 18 | 0.440 | 0.510 |
| CFL/seed_0 | 3 (0.01) | 2 | 1 / 0 | 0.667 | 284 / 74 / 210 / 0 | 13 / 12 / 1 | 0.253 | 0.257 |
| CFL/seed_1 | 4 (0.01) | 1 | 3 / 0 | 0.250 | 286 / 73 / 212 / 1 | 10 / 10 / 0 | 0.247 | 0.257 |
| CFL/seed_2 | 0 (0.00) | 0 | 0 / 0 | nan | 289 / 74 / 215 / 0 | 11 / 11 / 0 | 0.247 | 0.247 |

## Reading

Primary CFL - O success: -0.006 (seed s.e. 0.004, boot 0.005, 2/3) -> NOT MET; practical CFL - start +0.006 (0.002, 3/3) -> MET.  Against the oracle arm: CFL - CF -0.221 (seed s.e. 0.023, 3/3); the oracle gain on these episodes is CF - O +0.216 (0.026).

ETT model checks (`CHECK.md`): held-out marginal NLL 3.678 (position-only reference 3.630), region L1 0.497, death AUC 0.871, far-route mass 0.040 vs oracle 0.040, zone mass 0.179 vs 0.175.

Oracle-label-trained learned ETT under the disclosed optimizer-stabilisation recipe; not the original learner, not the offline identification.
