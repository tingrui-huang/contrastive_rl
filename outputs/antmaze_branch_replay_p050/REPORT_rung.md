# antmaze_branch_replay_p050: every evaluated policy, natural draws at the rung density

| policy | n | success | failure | timeout | detour | shortcut | discounted (g 0.99) | mouth 1 median step / after-burst share | detour episodes: success / timeout | shortcut episodes: failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| vanilla g0.999 (recorded data), seed 0, mean | 300 | 0.260 | 0.730 | 0.010 | **0.003** | 0.970 | 0.028 | 51.000 / 0.000 | 0.00 / 1.00 | 0.73 |
| vanilla g0.999 (recorded data), seed 1, mean | 300 | 0.260 | 0.737 | 0.003 | **0.007** | 0.960 | 0.028 | 51.000 / 0.000 | 1.00 / 0.00 | 0.74 |
| vanilla g0.999 (recorded data), seed 2, mean | 300 | 0.260 | 0.733 | 0.007 | **0.007** | 0.967 | 0.028 | 51.000 / 0.000 | 0.50 / 0.50 | 0.73 |
| vanilla g0.999 (recorded data), seed 0, sample | 300 | 0.260 | 0.737 | 0.003 | **0.003** | 0.960 | 0.027 | 52.000 / 0.007 | 1.00 / 0.00 | 0.73 |
| vanilla g0.999 (recorded data), seed 1, sample | 300 | 0.260 | 0.733 | 0.007 | **0.017** | 0.960 | 0.027 | 52.000 / 0.000 | 0.60 / 0.40 | 0.74 |
| vanilla g0.999 (recorded data), seed 2, sample | 300 | 0.260 | 0.723 | 0.017 | **0.003** | 0.970 | 0.027 | 52.000 / 0.000 | 0.00 / 1.00 | 0.73 |
| joint 100k on the 100k branch critic, seed 0, mean | 300 | 0.603 | 0.027 | 0.370 | **0.003** | 0.790 | 0.019 | 156.000 / 0.926 | 0.00 / 1.00 | 0.03 |
| joint 100k on the 100k branch critic, seed 1, mean | 300 | 0.453 | 0.033 | 0.513 | **0.017** | 0.543 | 0.015 | 143.000 / 0.837 | 0.00 / 1.00 | 0.06 |
| joint 100k on the 100k branch critic, seed 2, mean | 300 | 0.000 | 0.000 | 1.000 | **0.000** | 0.000 | 0.000 | -- / -- | -- | -- |
| joint 100k on the 100k branch critic, seed 3, mean | 300 | 0.000 | 0.000 | 1.000 | **0.000** | 0.000 | 0.000 | -- / -- | -- | -- |
| joint 100k on the 100k branch critic, seed 4, mean | 300 | 0.093 | 0.027 | 0.880 | **0.007** | 0.160 | 0.004 | 122.000 / 0.667 | 0.00 / 1.00 | 0.17 |
| joint 100k on the 100k branch critic, seed 0, sample | 300 | 0.337 | 0.000 | 0.663 | **0.250** | 0.477 | 0.003 | 285.000 / 1.000 | 0.00 / 1.00 | 0.00 |
| joint 100k on the 100k branch critic, seed 1, sample | 300 | 0.450 | 0.023 | 0.527 | **0.050** | 0.563 | 0.013 | 158.000 / 0.926 | 0.00 / 1.00 | 0.04 |
| joint 100k on the 100k branch critic, seed 2, sample | 300 | 0.000 | 0.000 | 1.000 | **0.077** | 0.040 | 0.000 | 409.500 / 1.000 | 0.00 / 1.00 | 0.00 |
| joint 100k on the 100k branch critic, seed 3, sample | 300 | 0.000 | 0.000 | 1.000 | **0.243** | 0.140 | 0.000 | 354.500 / 1.000 | 0.00 / 1.00 | 0.00 |
| joint 100k on the 100k branch critic, seed 4, sample | 300 | 0.107 | 0.003 | 0.890 | **0.120** | 0.187 | 0.003 | 186.500 / 0.917 | 0.00 / 1.00 | 0.02 |
| frozen 30k critic, actor bc 0.5, seed 0, mean | 300 | 0.300 | 0.473 | 0.227 | **0.003** | 0.750 | 0.023 | 65.000 / 0.116 | 0.00 / 1.00 | 0.60 |
| frozen 30k critic, actor bc 0.5, seed 0, sample | 300 | 0.417 | 0.467 | 0.117 | **0.037** | 0.877 | 0.026 | 70.000 / 0.248 | 0.09 / 0.91 | 0.51 |
| pure BC (bc 1.0), seed 0, mean | 300 | 0.253 | 0.743 | 0.003 | **0.000** | 0.977 | 0.026 | 52.000 / 0.000 | -- | 0.74 |
| pure BC (bc 1.0), seed 0, sample | 300 | 0.260 | 0.740 | 0.000 | **0.000** | 0.963 | 0.025 | 52.000 / 0.007 | -- | 0.73 |
| rank128 | critic critics_30k/seed_0 | 100 | 0.440 | 0.260 | 0.300 | **0.080** | 0.760 | 0.019 | 95.000 / 0.447 | 0.00 / 1.00 | 0.33 |
| rank32 | critic critics_30k/seed_0 | 100 | 0.410 | 0.340 | 0.250 | **0.120** | 0.770 | 0.021 | 81.000 / 0.400 | 0.00 / 1.00 | 0.42 |
| rank32_tau1 | critic critics_30k/seed_0 | 300 | 0.253 | 0.710 | 0.037 | **0.023** | 0.940 | 0.023 | 52.000 / 0.014 | 0.14 / 0.86 | 0.72 |
| rank32_tau1 | critic critics_30k/seed_1 | 300 | 0.257 | 0.693 | 0.050 | **0.037** | 0.890 | 0.022 | 52.000 / 0.000 | 0.27 / 0.73 | 0.72 |
| rank32_tau1 | critic critics_30k/seed_2 | 300 | 0.270 | 0.700 | 0.030 | **0.027** | 0.950 | 0.024 | 52.000 / 0.014 | 0.12 / 0.88 | 0.71 |
| rank32_tau1 | critic critics_30k/seed_3 | 300 | 0.263 | 0.707 | 0.030 | **0.030** | 0.930 | 0.024 | 52.000 / 0.014 | 0.11 / 0.89 | 0.72 |
| rank32_tau1 | critic critics_30k/seed_4 | 300 | 0.247 | 0.730 | 0.023 | **0.013** | 0.933 | 0.023 | 52.000 / 0.000 | 0.50 / 0.50 | 0.74 |
| rank32_tau1_knn60000 | critic critics_30k/seed_0 | 100 | 0.060 | 0.120 | 0.820 | **0.030** | 0.550 | 0.001 | 80.000 / 0.130 | 0.00 / 1.00 | 0.20 |
| rank32_tau0.3 | critic critics_30k/seed_0 | 100 | 0.330 | 0.460 | 0.210 | **0.030** | 0.830 | 0.023 | 66.000 / 0.190 | 0.00 / 1.00 | 0.55 |
| rank8_tau1 | critic critics_30k/seed_0 | 300 | 0.253 | 0.727 | 0.020 | **0.017** | 0.947 | 0.024 | 52.000 / 0.000 | 0.20 / 0.80 | 0.74 |
| segrank128_L25 | critic critics_30k/seed_0 | 100 | 0.360 | 0.290 | 0.350 | **0.340** | 0.450 | 0.017 | 56.000 / 0.048 | 0.59 / 0.41 | 0.62 |
| segrank32_L25 | critic critics_30k/seed_0 | 100 | 0.380 | 0.230 | 0.390 | **0.350** | 0.370 | 0.017 | 56.000 / 0.000 | 0.69 / 0.31 | 0.59 |
| segrank64_L25 | critic critics_30k/seed_0 | 100 | 0.390 | 0.310 | 0.300 | **0.400** | 0.470 | 0.017 | 55.000 / 0.000 | 0.57 / 0.42 | 0.66 |
| segrank64_L25 | critic critics_30k/seed_1 | 100 | 0.450 | 0.240 | 0.310 | **0.520** | 0.330 | 0.014 | 55.000 / 0.118 | 0.67 / 0.33 | 0.67 |
| segrank64_L25 | critic critics_30k/seed_2 | 100 | 0.420 | 0.320 | 0.260 | **0.360** | 0.470 | 0.018 | 56.000 / 0.107 | 0.72 / 0.28 | 0.64 |
| segrank64_L25 | critic critics_30k/seed_3 | 100 | 0.430 | 0.240 | 0.330 | **0.430** | 0.380 | 0.018 | 56.000 / 0.118 | 0.67 / 0.33 | 0.61 |
| segrank64_L25 | critic critics_30k/seed_4 | 100 | 0.340 | 0.380 | 0.280 | **0.260** | 0.520 | 0.018 | 57.000 / 0.000 | 0.62 / 0.38 | 0.65 |
| segrank64_L25 | critic critics_30k/seed_0 | 300 | 0.407 | 0.280 | 0.313 | **0.443** | 0.423 | 0.016 | 55.000 / 0.016 | 0.60 / 0.40 | 0.65 |
| segrank64_L25 | critic critics_30k/seed_1 | 300 | 0.390 | 0.290 | 0.320 | **0.457** | 0.383 | 0.013 | 56.000 / 0.103 | 0.63 / 0.37 | 0.69 |
| segrank64_L25 | critic critics_30k/seed_2 | 300 | 0.373 | 0.320 | 0.307 | **0.360** | 0.457 | 0.016 | 56.000 / 0.052 | 0.64 / 0.36 | 0.68 |
| segrank64_L25 | critic critics_30k/seed_3 | 300 | 0.467 | 0.237 | 0.297 | **0.473** | 0.383 | 0.018 | 56.000 / 0.107 | 0.69 / 0.31 | 0.61 |
| segrank64_L25 | critic critics_30k/seed_4 | 300 | 0.377 | 0.313 | 0.310 | **0.330** | 0.447 | 0.017 | 56.000 / 0.071 | 0.68 / 0.32 | 0.65 |
| segrank64_L25_random | no critic (uniform pick) | 300 | 0.240 | 0.557 | 0.203 | **0.050** | 0.767 | 0.020 | 55.000 / 0.035 | 0.47 / 0.53 | 0.71 |
| segrank64_L25 | critic vanilla_g0999/seed_0 | 300 | 0.397 | 0.270 | 0.333 | **0.383** | 0.437 | 0.017 | 60.000 / 0.111 | 0.58 / 0.42 | 0.58 |
| segrank64_L25 | critic vanilla_g0999/seed_1 | 300 | 0.407 | 0.250 | 0.343 | **0.473** | 0.377 | 0.014 | 56.000 / 0.164 | 0.61 / 0.39 | 0.64 |
| segrank64_L25 | critic vanilla_g0999/seed_2 | 300 | 0.427 | 0.207 | 0.367 | **0.463** | 0.350 | 0.014 | 63.000 / 0.193 | 0.61 / 0.39 | 0.56 |

mode = tanh(loc), the repo and the paper's evaluation convention; sample = tanh(loc + scale eps), secondary.  rank policies: K proposals scored by the critic, argmax (per-step: K tanh-normal samples of the BC actor; segrank: K recorded 25-step torque segments at the start, run open-loop, BC mode elsewhere).  The rockfall clocks run from the reset (zone 1 closes by step 117): an after-burst share near one means the shortcut survives by lateness.
