# The critic and the consequences on the same ruler: gamma-law region masses at the 64 reset states, per first action

Real = the simulator (paired hidden draws, the start agent's continuation; {'logged': 64, 'prog': 64, 'stall': 64, 'start': 64} draws per state); ETT = the S ETT with the start continuation ({'logged': 32, 'prog': 32, 'stall': 32, 'start': 32} draws per state); critic = (B - 1) E_g[exp f(s, a, g) 1{g in G}] over 8192 goals of the critic's own training marginal (B = 1024; 'total' = the same over all g, 1 for a calibrated critic; 'share' = mass / total).  Regions: within 0.5 / 2.0 of the task goal, the goal area, the far regions, the start region.  The marginal's own region masses: reach0.5 0.0395, near2.0 0.2138, goal_area 0.2711, far 0.0374, start_region 0.0090 (draw 1).

## reach0.5

| first action | real mass | ETT mass | d1_final: min-head mass / share / total | d1_20k: min-head mass / share / total | d2_final: min-head mass / share / total | d2_20k: min-head mass / share / total |
|---|---:|---:|---|---|---|---|
| logged | 0.0009 | 0.0010 | 0.0066 / 0.0074 / 0.88 | 0.0068 / 0.0081 / 0.85 | 0.0023 / 0.0026 / 0.93 | 0.0039 / 0.0046 / 0.90 |
| stall | 0.0009 | 0.0010 | 0.0466 / 0.0305 / 1.42 | 0.0313 / 0.0419 / 0.74 | 0.0016 / 0.0051 / 0.77 | 0.0041 / 0.0110 / 0.61 |
| prog | 0.0008 | 0.0009 | 0.0051 / 0.0069 / 0.96 | 0.0105 / 0.0126 / 0.80 | 0.0080 / 0.0135 / 0.62 | 0.0142 / 0.0192 / 0.76 |
| start | 0.0009 | 0.0009 | 0.0006 / 0.0007 / 0.85 | 0.0019 / 0.0028 / 0.67 | 0.0005 / 0.0008 / 0.83 | 0.0016 / 0.0019 / 0.96 |

| paired contrast | real | ETT | d1_final: min-head mass (share pos) / share | d1_20k: min-head mass (share pos) / share | d2_final: min-head mass (share pos) / share | d2_20k: min-head mass (share pos) / share |
|---|---|---|---|---|---|---|
| stall - logged | -0.0000 +- 0.0001 (0.23) | +0.0000 +- 0.0001 (0.34) | +0.0401 +- 0.0044 (1.00) / +0.0231 | +0.0244 +- 0.0015 (1.00) / +0.0339 | -0.0007 +- 0.0004 (0.47) / +0.0025 | +0.0002 +- 0.0005 (0.53) / +0.0064 |
| stall - prog | +0.0001 +- 0.0001 (0.38) | +0.0001 +- 0.0001 (0.39) | +0.0415 +- 0.0048 (1.00) / +0.0235 | +0.0208 +- 0.0014 (1.00) / +0.0294 | -0.0064 +- 0.0008 (0.06) / -0.0084 | -0.0101 +- 0.0008 (0.05) / -0.0082 |

## near2.0

| first action | real mass | ETT mass | d1_final: min-head mass / share / total | d1_20k: min-head mass / share / total | d2_final: min-head mass / share / total | d2_20k: min-head mass / share / total |
|---|---:|---:|---|---|---|---|
| logged | 0.0321 | 0.0250 | 0.0440 / 0.0487 / 0.88 | 0.0436 / 0.0507 / 0.85 | 0.0165 / 0.0189 / 0.93 | 0.0244 / 0.0286 / 0.90 |
| stall | 0.0311 | 0.0230 | 0.2581 / 0.1781 / 1.42 | 0.1721 / 0.2330 / 0.74 | 0.0079 / 0.0264 / 0.77 | 0.0200 / 0.0549 / 0.61 |
| prog | 0.0279 | 0.0247 | 0.0292 / 0.0395 / 0.96 | 0.0586 / 0.0709 / 0.80 | 0.0415 / 0.0702 / 0.62 | 0.0757 / 0.1014 / 0.76 |
| start | 0.0278 | 0.0269 | 0.0024 / 0.0029 / 0.85 | 0.0093 / 0.0143 / 0.67 | 0.0023 / 0.0042 / 0.83 | 0.0080 / 0.0099 / 0.96 |

| paired contrast | real | ETT | d1_final: min-head mass (share pos) / share | d1_20k: min-head mass (share pos) / share | d2_final: min-head mass (share pos) / share | d2_20k: min-head mass (share pos) / share |
|---|---|---|---|---|---|---|
| stall - logged | -0.0010 +- 0.0115 (0.38) | -0.0020 +- 0.0031 (0.42) | +0.2141 +- 0.0171 (1.00) / +0.1294 | +0.1285 +- 0.0069 (0.98) / +0.1823 | -0.0086 +- 0.0023 (0.30) / +0.0074 | -0.0044 +- 0.0021 (0.42) / +0.0263 |
| stall - prog | +0.0032 +- 0.0103 (0.41) | -0.0017 +- 0.0041 (0.42) | +0.2289 +- 0.0178 (1.00) / +0.1386 | +0.1136 +- 0.0058 (1.00) / +0.1621 | -0.0336 +- 0.0028 (0.03) / -0.0439 | -0.0557 +- 0.0030 (0.00) / -0.0465 |

## goal_area

| first action | real mass | ETT mass | d1_final: min-head mass / share / total | d1_20k: min-head mass / share / total | d2_final: min-head mass / share / total | d2_20k: min-head mass / share / total |
|---|---:|---:|---|---|---|---|
| logged | 0.0401 | 0.0328 | 0.0610 / 0.0674 / 0.88 | 0.0591 / 0.0686 / 0.85 | 0.0219 / 0.0251 / 0.93 | 0.0312 / 0.0364 / 0.90 |
| stall | 0.0333 | 0.0289 | 0.3538 / 0.2413 / 1.42 | 0.2212 / 0.2996 / 0.74 | 0.0088 / 0.0286 / 0.77 | 0.0212 / 0.0566 / 0.61 |
| prog | 0.0340 | 0.0321 | 0.0381 / 0.0520 / 0.96 | 0.0741 / 0.0902 / 0.80 | 0.0524 / 0.0876 / 0.62 | 0.0903 / 0.1197 / 0.76 |
| start | 0.0352 | 0.0344 | 0.0028 / 0.0034 / 0.85 | 0.0115 / 0.0176 / 0.67 | 0.0025 / 0.0046 / 0.83 | 0.0089 / 0.0109 / 0.96 |

| paired contrast | real | ETT | d1_final: min-head mass (share pos) / share | d1_20k: min-head mass (share pos) / share | d2_final: min-head mass (share pos) / share | d2_20k: min-head mass (share pos) / share |
|---|---|---|---|---|---|---|
| stall - logged | -0.0068 +- 0.0111 (0.38) | -0.0039 +- 0.0033 (0.42) | +0.2927 +- 0.0239 (1.00) / +0.1739 | +0.1622 +- 0.0085 (0.98) / +0.2311 | -0.0131 +- 0.0031 (0.25) / +0.0035 | -0.0100 +- 0.0024 (0.33) / +0.0202 |
| stall - prog | -0.0007 +- 0.0083 (0.42) | -0.0031 +- 0.0046 (0.44) | +0.3156 +- 0.0244 (1.00) / +0.1893 | +0.1471 +- 0.0071 (1.00) / +0.2094 | -0.0436 +- 0.0036 (0.03) / -0.0589 | -0.0690 +- 0.0036 (0.00) / -0.0631 |

## far

| first action | real mass | ETT mass | d1_final: min-head mass / share / total | d1_20k: min-head mass / share / total | d2_final: min-head mass / share / total | d2_20k: min-head mass / share / total |
|---|---:|---:|---|---|---|---|
| logged | 0.0082 | 0.0000 | 0.0013 / 0.0018 / 0.88 | 0.0019 / 0.0024 / 0.85 | 0.0007 / 0.0010 / 0.93 | 0.0018 / 0.0025 / 0.90 |
| stall | 0.1329 | 0.0370 | 0.0396 / 0.0442 / 1.42 | 0.0430 / 0.0581 / 0.74 | 0.0039 / 0.0157 / 0.77 | 0.0101 / 0.0342 / 0.61 |
| prog | 0.1095 | 0.0220 | 0.0085 / 0.0099 / 0.96 | 0.0170 / 0.0188 / 0.80 | 0.0093 / 0.0169 / 0.62 | 0.0225 / 0.0319 / 0.76 |
| start | 0.0027 | 0.0000 | 0.0008 / 0.0011 / 0.85 | 0.0020 / 0.0033 / 0.67 | 0.0008 / 0.0018 / 0.83 | 0.0024 / 0.0034 / 0.96 |

| paired contrast | real | ETT | d1_final: min-head mass (share pos) / share | d1_20k: min-head mass (share pos) / share | d2_final: min-head mass (share pos) / share | d2_20k: min-head mass (share pos) / share |
|---|---|---|---|---|---|---|
| stall - logged | +0.1246 +- 0.0408 (0.17) | +0.0370 +- 0.0211 (0.06) | +0.0383 +- 0.0083 (0.98) / +0.0424 | +0.0411 +- 0.0087 (0.98) / +0.0558 | +0.0032 +- 0.0009 (0.78) / +0.0147 | +0.0083 +- 0.0018 (0.88) / +0.0317 |
| stall - prog | +0.0234 +- 0.0543 (0.16) | +0.0149 +- 0.0211 (0.06) | +0.0311 +- 0.0079 (0.77) / +0.0344 | +0.0260 +- 0.0081 (0.59) / +0.0393 | -0.0054 +- 0.0012 (0.17) / -0.0011 | -0.0124 +- 0.0021 (0.16) / +0.0023 |

## start_region

| first action | real mass | ETT mass | d1_final: min-head mass / share / total | d1_20k: min-head mass / share / total | d2_final: min-head mass / share / total | d2_20k: min-head mass / share / total |
|---|---:|---:|---|---|---|---|
| logged | 0.2174 | - | 0.1982 / 0.2316 / 0.88 | 0.1608 / 0.1894 / 0.85 | 0.2730 / 0.2908 / 0.93 | 0.1963 / 0.2183 / 0.90 |
| stall | 0.2936 | - | 0.0365 / 0.0380 / 1.42 | 0.0446 / 0.0611 / 0.74 | 0.3945 / 0.5125 / 0.77 | 0.1923 / 0.2869 / 0.61 |
| prog | 0.2711 | - | 0.5836 / 0.4701 / 0.96 | 0.3149 / 0.3792 / 0.80 | 0.2225 / 0.3589 / 0.62 | 0.1888 / 0.2495 / 0.76 |
| start | 0.2575 | - | 0.2708 / 0.3196 / 0.85 | 0.2149 / 0.3318 / 0.67 | 0.3062 / 0.3795 / 0.83 | 0.3387 / 0.3580 / 0.96 |

| paired contrast | real | ETT | d1_final: min-head mass (share pos) / share | d1_20k: min-head mass (share pos) / share | d2_final: min-head mass (share pos) / share | d2_20k: min-head mass (share pos) / share |
|---|---|---|---|---|---|---|
| stall - logged | +0.0762 +- 0.0325 (0.77) | - | -0.1616 +- 0.0089 (0.00) / -0.1936 | -0.1163 +- 0.0061 (0.03) / -0.1283 | +0.1215 +- 0.0287 (0.62) / +0.2217 | -0.0040 +- 0.0159 (0.39) / +0.0687 |
| stall - prog | +0.0225 +- 0.0376 (0.58) | - | -0.5471 +- 0.1993 (0.03) / -0.4321 | -0.2703 +- 0.0184 (0.03) / -0.3181 | +0.1720 +- 0.0350 (0.66) / +0.1536 | +0.0035 +- 0.0208 (0.41) / +0.0374 |

## Per-head totals (calibration; mean over states)

| critic | action | h0 total | h1 total | min total | max-weight share (min) |
|---|---|---:|---:|---:|---:|
| d1_final | logged | 85534240.73 | 0.94 | 0.88 | 0.008 |
| d1_final | stall | 2.34 | 1.63 | 1.42 | 0.008 |
| d1_final | prog | 4866948967586.06 | 1.15 | 0.96 | 0.078 |
| d1_final | start | 1.39 | 0.89 | 0.85 | 0.016 |
| d1_20k | logged | 0.92 | 0.97 | 0.85 | 0.006 |
| d1_20k | stall | 1.02 | 0.91 | 0.74 | 0.008 |
| d1_20k | prog | 0.93 | 1.13 | 0.80 | 0.038 |
| d1_20k | start | 1.02 | 0.68 | 0.67 | 0.020 |
| d2_final | logged | 1.10 | 1.01 | 0.93 | 0.010 |
| d2_final | stall | 1.00 | 1.24 | 0.77 | 0.034 |
| d2_final | prog | 1.02 | 0.87 | 0.62 | 0.037 |
| d2_final | start | 1.08 | 1.13 | 0.83 | 0.013 |
| d2_20k | logged | 0.97 | 1.05 | 0.90 | 0.006 |
| d2_20k | stall | 0.72 | 0.76 | 0.61 | 0.014 |
| d2_20k | prog | 0.87 | 1.00 | 0.76 | 0.017 |
| d2_20k | start | 1.31 | 1.09 | 0.96 | 0.014 |

## Agreement with the real masses (256 (state, action) pairs; the per-state stall - logged contrast)

| region | reader | Pearson (pairs) | Spearman (pairs) | Spearman within state (mean over states, 4 actions) | contrast Pearson | contrast sign agreement |
|---|---|---:|---:|---:|---:|---:|
| reach0.5 | ETT | 0.10 | 0.37 | | | |
| reach0.5 | d1_final|min | -0.04 | 0.14 | -0.07 | -0.16 | 0.23 |
| reach0.5 | d1_final|h0 | -0.01 | 0.10 | -0.11 | -0.06 | 0.23 |
| reach0.5 | d1_20k|min | -0.07 | 0.12 | -0.10 | -0.22 | 0.23 |
| reach0.5 | d1_20k|h0 | -0.03 | 0.10 | -0.16 | 0.00 | 0.22 |
| reach0.5 | d2_final|min | 0.05 | 0.05 | 0.06 | 0.16 | 0.53 |
| reach0.5 | d2_final|h0 | 0.02 | 0.07 | 0.07 | 0.04 | 0.58 |
| reach0.5 | d2_20k|min | -0.02 | 0.02 | 0.04 | 0.09 | 0.50 |
| reach0.5 | d2_20k|h0 | -0.00 | 0.10 | 0.11 | 0.16 | 0.59 |
| near2.0 | ETT | 0.29 | 0.27 | | | |
| near2.0 | d1_final|min | -0.01 | -0.05 | -0.09 | -0.06 | 0.38 |
| near2.0 | d1_final|h0 | 0.06 | 0.01 | -0.09 | 0.06 | 0.38 |
| near2.0 | d1_20k|min | 0.13 | 0.01 | -0.09 | 0.13 | 0.39 |
| near2.0 | d1_20k|h0 | 0.17 | 0.04 | -0.11 | 0.18 | 0.39 |
| near2.0 | d2_final|min | 0.08 | 0.13 | -0.05 | 0.16 | 0.55 |
| near2.0 | d2_final|h0 | 0.10 | 0.10 | 0.00 | 0.44 | 0.58 |
| near2.0 | d2_20k|min | 0.13 | 0.11 | -0.02 | 0.45 | 0.48 |
| near2.0 | d2_20k|h0 | 0.11 | 0.08 | 0.02 | 0.48 | 0.61 |
| goal_area | ETT | 0.25 | 0.48 | | | |
| goal_area | d1_final|min | -0.04 | -0.14 | -0.16 | 0.01 | 0.38 |
| goal_area | d1_final|h0 | 0.00 | -0.13 | -0.16 | 0.09 | 0.38 |
| goal_area | d1_20k|min | 0.06 | -0.12 | -0.14 | 0.14 | 0.39 |
| goal_area | d1_20k|h0 | 0.08 | -0.12 | -0.14 | 0.16 | 0.39 |
| goal_area | d2_final|min | 0.07 | 0.08 | -0.03 | 0.07 | 0.47 |
| goal_area | d2_final|h0 | 0.06 | 0.05 | -0.02 | 0.30 | 0.47 |
| goal_area | d2_20k|min | 0.07 | 0.04 | -0.02 | 0.30 | 0.52 |
| goal_area | d2_20k|h0 | 0.06 | 0.02 | -0.02 | 0.30 | 0.48 |
| far | ETT | 0.26 | 0.30 | | | |
| far | d1_final|min | 0.33 | 0.36 | 0.58 | 0.34 | 0.19 |
| far | d1_final|h0 | 0.41 | 0.37 | 0.63 | 0.34 | 0.19 |
| far | d1_20k|min | 0.36 | 0.37 | 0.60 | 0.34 | 0.19 |
| far | d1_20k|h0 | 0.37 | 0.38 | 0.60 | 0.34 | 0.19 |
| far | d2_final|min | 0.35 | 0.34 | 0.56 | 0.45 | 0.19 |
| far | d2_final|h0 | 0.36 | 0.36 | 0.61 | 0.45 | 0.19 |
| far | d2_20k|min | 0.34 | 0.35 | 0.56 | 0.40 | 0.19 |
| far | d2_20k|h0 | 0.39 | 0.37 | 0.64 | 0.44 | 0.19 |
| start_region | d1_final|min | -0.06 | 0.04 | -0.05 | -0.02 | 0.23 |
| start_region | d1_final|h0 | -0.08 | -0.00 | -0.04 | -0.07 | 0.25 |
| start_region | d1_20k|min | 0.02 | -0.01 | -0.07 | -0.05 | 0.23 |
| start_region | d1_20k|h0 | 0.02 | 0.11 | 0.14 | -0.13 | 0.33 |
| start_region | d2_final|min | 0.08 | 0.17 | 0.16 | 0.07 | 0.67 |
| start_region | d2_final|h0 | 0.09 | 0.19 | 0.13 | 0.12 | 0.62 |
| start_region | d2_20k|min | 0.06 | 0.25 | 0.22 | 0.16 | 0.53 |
| start_region | d2_20k|h0 | 0.04 | 0.29 | 0.26 | 0.21 | 0.52 |

## Over-estimation factor: critic (min-head) mass / real mass per (state, action)

| region | critic | action | median | IQR | share > 2x | share < 0.5x | MC half-split error / mass (median) |
|---|---|---|---:|---|---:|---:|---:|
| goal_area | d1_final | logged | 1.76 | 0.82 - 3.40 | 0.44 | 0.11 | 0.04 |
| goal_area | d1_final | stall | 17.77 | 9.21 - 28.85 | 0.95 | 0.00 | 0.03 |
| goal_area | d1_final | prog | 1.18 | 0.67 - 2.37 | 0.31 | 0.20 | 0.02 |
| goal_area | d1_final | start | 0.09 | 0.05 - 0.14 | 0.00 | 0.97 | 0.01 |
| goal_area | d1_20k | logged | 1.83 | 1.29 - 2.73 | 0.42 | 0.08 | 0.04 |
| goal_area | d1_20k | stall | 9.54 | 7.54 - 15.74 | 0.95 | 0.00 | 0.03 |
| goal_area | d1_20k | prog | 2.53 | 1.55 - 4.26 | 0.66 | 0.06 | 0.02 |
| goal_area | d1_20k | start | 0.42 | 0.31 - 0.58 | 0.02 | 0.66 | 0.02 |
| goal_area | d2_final | logged | 0.38 | 0.16 - 0.83 | 0.11 | 0.61 | 0.04 |
| goal_area | d2_final | stall | 0.26 | 0.12 - 0.78 | 0.22 | 0.61 | 0.02 |
| goal_area | d2_final | prog | 2.19 | 1.61 - 3.46 | 0.55 | 0.08 | 0.03 |
| goal_area | d2_final | start | 0.07 | 0.05 - 0.12 | 0.00 | 0.98 | 0.01 |
| goal_area | d2_20k | logged | 0.94 | 0.45 - 1.35 | 0.14 | 0.30 | 0.05 |
| goal_area | d2_20k | stall | 0.72 | 0.45 - 1.82 | 0.23 | 0.28 | 0.02 |
| goal_area | d2_20k | prog | 3.83 | 2.81 - 5.21 | 0.88 | 0.05 | 0.03 |
| goal_area | d2_20k | start | 0.30 | 0.19 - 0.37 | 0.02 | 0.86 | 0.02 |
| near2.0 | d1_final | logged | 1.71 | 0.87 - 3.59 | 0.47 | 0.12 | 0.04 |
| near2.0 | d1_final | stall | 17.74 | 7.28 - 30.50 | 0.92 | 0.02 | 0.03 |
| near2.0 | d1_final | prog | 1.26 | 0.68 - 2.55 | 0.34 | 0.16 | 0.02 |
| near2.0 | d1_final | start | 0.11 | 0.06 - 0.19 | 0.00 | 0.95 | 0.01 |
| near2.0 | d1_20k | logged | 2.02 | 1.24 - 3.03 | 0.50 | 0.08 | 0.04 |
| near2.0 | d1_20k | stall | 10.01 | 6.80 - 14.43 | 0.94 | 0.02 | 0.03 |
| near2.0 | d1_20k | prog | 2.87 | 1.81 - 4.77 | 0.70 | 0.03 | 0.02 |
| near2.0 | d1_20k | start | 0.49 | 0.34 - 0.70 | 0.02 | 0.55 | 0.02 |
| near2.0 | d2_final | logged | 0.43 | 0.18 - 0.92 | 0.12 | 0.61 | 0.04 |
| near2.0 | d2_final | stall | 0.28 | 0.15 - 0.77 | 0.16 | 0.61 | 0.02 |
| near2.0 | d2_final | prog | 2.52 | 1.53 - 4.50 | 0.72 | 0.06 | 0.03 |
| near2.0 | d2_final | start | 0.09 | 0.06 - 0.15 | 0.00 | 0.98 | 0.01 |
| near2.0 | d2_20k | logged | 1.04 | 0.50 - 1.66 | 0.19 | 0.25 | 0.05 |
| near2.0 | d2_20k | stall | 0.93 | 0.54 - 1.64 | 0.22 | 0.23 | 0.02 |
| near2.0 | d2_20k | prog | 4.37 | 3.27 - 6.59 | 0.91 | 0.03 | 0.03 |
| near2.0 | d2_20k | start | 0.37 | 0.26 - 0.49 | 0.02 | 0.77 | 0.02 |
| reach0.5 | d1_final | logged | 4.46 | 1.79 - 11.66 | 0.72 | 0.08 | 0.04 |
| reach0.5 | d1_final | stall | 55.51 | 26.63 - 93.91 | 0.98 | 0.00 | 0.03 |
| reach0.5 | d1_final | prog | 4.66 | 2.72 - 9.29 | 0.81 | 0.03 | 0.02 |
| reach0.5 | d1_final | start | 0.48 | 0.31 - 0.73 | 0.11 | 0.50 | 0.01 |
| reach0.5 | d1_20k | logged | 5.20 | 2.84 - 11.14 | 0.86 | 0.03 | 0.04 |
| reach0.5 | d1_20k | stall | 36.01 | 21.11 - 59.41 | 1.00 | 0.00 | 0.03 |
| reach0.5 | d1_20k | prog | 10.37 | 6.39 - 20.40 | 0.95 | 0.00 | 0.02 |
| reach0.5 | d1_20k | start | 1.92 | 1.33 - 2.44 | 0.41 | 0.02 | 0.02 |
| reach0.5 | d2_final | logged | 1.27 | 0.67 - 3.17 | 0.38 | 0.16 | 0.04 |
| reach0.5 | d2_final | stall | 1.33 | 0.47 - 4.36 | 0.44 | 0.27 | 0.02 |
| reach0.5 | d2_final | prog | 8.49 | 4.94 - 16.91 | 0.97 | 0.00 | 0.03 |
| reach0.5 | d2_final | start | 0.39 | 0.24 - 0.70 | 0.06 | 0.55 | 0.01 |
| reach0.5 | d2_20k | logged | 3.04 | 1.56 - 5.73 | 0.64 | 0.06 | 0.05 |
| reach0.5 | d2_20k | stall | 4.02 | 1.95 - 9.19 | 0.73 | 0.03 | 0.02 |
| reach0.5 | d2_20k | prog | 15.60 | 10.86 - 23.86 | 1.00 | 0.00 | 0.03 |
| reach0.5 | d2_20k | start | 1.54 | 0.98 - 2.13 | 0.28 | 0.03 | 0.02 |
