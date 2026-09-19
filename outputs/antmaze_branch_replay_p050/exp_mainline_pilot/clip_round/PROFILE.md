# A. The current agents' futures at the same training anchors (paired by anchor: same state, same logged query torque, same hazard draw)

Old = the start agent's continuation (`branches_cf.npz`, the critic's positive futures of every run so far); new = the lineage agent (the clipped CF final of that seed) continuing.  Anchor-weighted shares (the critic anchor law); entered far = y >= 6 at x < 2 after the query; completed far = entered and the branch reached the goal; masses = the critic's goal marginal (truncated geometric law, gamma 0.999) in the far-route legs / the goal area / the hazard zones.  `profile.json`.

**Gate for B** (anchor in start region, far route completed (entered the far route after the query and the branch reached the goal), anchor-weighted share; new - old > 0 in 3/3 lineages AND the mean over the 3 lineages >= 0.02 (absolute); the start-agent share is the same file for every lineage): per lineage +0.002 / +0.010 / +0.009, mean +0.007 -> **NOT PASSED**.

## reset rows (t = 0) (n = 196, weight 0.004)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.013 / 0.066 (+0.053) | 0.009 / 0.035 (+0.026 +- 0.011) | 0.956 / 0.903 | 0.286 / 0.189 | 0.674 / 0.634 | 0.040 / 0.176 | 160 / 262 | 0.010 / 0.055 | 0.057 / 0.016 | 0.139 / 0.144 | 0.053 / 0.000 |
| seed_1 | 0.013 / 0.066 (+0.053) | 0.009 / 0.026 (+0.018 +- 0.011) | 0.956 / 0.907 | 0.286 / 0.172 | 0.674 / 0.586 | 0.040 / 0.242 | 160 / 296 | 0.010 / 0.061 | 0.057 / 0.015 | 0.139 / 0.174 | 0.053 / 0.000 |
| seed_2 | 0.013 / 0.035 (+0.022) | 0.009 / 0.022 (+0.013 +- 0.010) | 0.956 / 0.890 | 0.286 / 0.220 | 0.674 / 0.608 | 0.040 / 0.172 | 160 / 259 | 0.010 / 0.028 | 0.057 / 0.024 | 0.139 / 0.155 | 0.022 / 0.000 |

## anchor in start region (n = 4276, weight 0.079)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.032 / 0.041 (+0.009) | 0.025 / 0.026 (+0.002 +- 0.002) | 0.937 / 0.919 | 0.243 / 0.172 | 0.728 / 0.703 | 0.029 / 0.124 | 138 / 206 | 0.028 / 0.041 | 0.038 / 0.015 | 0.144 / 0.153 | 0.014 / 0.005 |
| seed_1 | 0.032 / 0.047 (+0.016) | 0.025 / 0.035 (+0.010 +- 0.002) | 0.937 / 0.897 | 0.243 / 0.174 | 0.728 / 0.664 | 0.029 / 0.162 | 138 / 230 | 0.028 / 0.043 | 0.038 / 0.016 | 0.144 / 0.159 | 0.018 / 0.002 |
| seed_2 | 0.032 / 0.040 (+0.008) | 0.025 / 0.033 (+0.009 +- 0.002) | 0.937 / 0.898 | 0.243 / 0.209 | 0.728 / 0.686 | 0.029 / 0.105 | 138 / 202 | 0.028 / 0.039 | 0.038 / 0.019 | 0.144 / 0.152 | 0.012 / 0.003 |

## start region, t in [1, 30) (n = 4037, weight 0.075)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.031 / 0.032 (+0.000) | 0.024 / 0.020 (-0.004 +- 0.002) | 0.944 / 0.928 | 0.240 / 0.167 | 0.739 / 0.714 | 0.021 / 0.119 | 132 / 199 | 0.028 / 0.034 | 0.037 / 0.015 | 0.145 / 0.155 | 0.006 / 0.006 |
| seed_1 | 0.031 / 0.038 (+0.007) | 0.024 / 0.029 (+0.005 +- 0.002) | 0.944 / 0.905 | 0.240 / 0.169 | 0.739 / 0.675 | 0.021 / 0.156 | 132 / 223 | 0.028 / 0.034 | 0.037 / 0.015 | 0.145 / 0.160 | 0.009 / 0.002 |
| seed_2 | 0.031 / 0.035 (+0.004) | 0.024 / 0.029 (+0.005 +- 0.002) | 0.944 / 0.905 | 0.240 / 0.204 | 0.739 / 0.697 | 0.021 / 0.099 | 132 / 195 | 0.028 / 0.034 | 0.037 / 0.018 | 0.145 / 0.152 | 0.007 / 0.003 |

## logged shortcut | start region (n = 4071, weight 0.076)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.003 (+0.003) | 0.000 / 0.001 (+0.001 +- 0.001) | 0.976 / 0.961 | 0.226 / 0.153 | 0.761 / 0.737 | 0.013 / 0.110 | 121 / 186 | 0.000 / 0.003 | 0.036 / 0.014 | 0.150 / 0.160 | 0.003 / 0.000 |
| seed_1 | 0.000 / 0.006 (+0.006) | 0.000 / 0.003 (+0.003 +- 0.001) | 0.976 / 0.938 | 0.226 / 0.148 | 0.761 / 0.695 | 0.013 / 0.157 | 121 / 215 | 0.000 / 0.006 | 0.036 / 0.013 | 0.150 / 0.166 | 0.006 / 0.000 |
| seed_2 | 0.000 / 0.004 (+0.004) | 0.000 / 0.003 (+0.003 +- 0.001) | 0.976 / 0.936 | 0.226 / 0.185 | 0.761 / 0.718 | 0.013 / 0.097 | 121 / 186 | 0.000 / 0.003 | 0.036 / 0.017 | 0.150 / 0.158 | 0.004 / 0.000 |

## logged detour | start region (n = 205, weight 0.004)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.688 / 0.819 (+0.131) | 0.529 / 0.538 (+0.009 +- 0.052) | 0.127 / 0.059 | 0.593 / 0.566 | 0.063 / 0.018 | 0.344 / 0.416 | 490 / 602 | 0.612 / 0.830 | 0.063 / 0.039 | 0.020 / 0.009 | 0.244 / 0.113 |
| seed_1 | 0.688 / 0.891 (+0.204) | 0.529 / 0.692 (+0.163 +- 0.049) | 0.127 / 0.063 | 0.593 / 0.710 | 0.063 / 0.036 | 0.344 / 0.253 | 490 / 537 | 0.612 / 0.806 | 0.063 / 0.065 | 0.020 / 0.008 | 0.249 / 0.045 |
| seed_2 | 0.688 / 0.787 (+0.100) | 0.529 / 0.665 (+0.136 +- 0.043) | 0.127 / 0.100 | 0.593 / 0.715 | 0.063 / 0.023 | 0.344 / 0.262 | 490 / 527 | 0.612 / 0.765 | 0.063 / 0.051 | 0.020 / 0.021 | 0.167 / 0.068 |

## anchor in pre_zone1 (n = 12541, weight 0.234)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.822 / 0.961 | 0.312 / 0.249 | 0.505 / 0.569 | 0.183 / 0.182 | 226 / 217 | 0.000 / 0.001 | 0.055 / 0.028 | 0.212 / 0.267 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.822 / 0.955 | 0.312 / 0.255 | 0.505 / 0.526 | 0.183 / 0.219 | 226 / 240 | 0.000 / 0.000 | 0.055 / 0.027 | 0.212 / 0.272 | 0.000 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.822 / 0.956 | 0.312 / 0.338 | 0.505 / 0.517 | 0.183 / 0.145 | 226 / 210 | 0.000 / 0.000 | 0.055 / 0.034 | 0.212 / 0.264 | 0.000 / 0.000 |

## anchor in zone1 (n = 4946, weight 0.092)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.999 / 0.994 | 0.531 / 0.381 | 0.446 / 0.385 | 0.023 / 0.235 | 117 / 248 | 0.000 / 0.001 | 0.111 / 0.049 | 0.365 / 0.397 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.999 / 0.992 | 0.531 / 0.375 | 0.446 / 0.366 | 0.023 / 0.259 | 117 / 258 | 0.000 / 0.000 | 0.111 / 0.048 | 0.365 / 0.375 | 0.000 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.999 / 0.994 | 0.531 / 0.447 | 0.446 / 0.375 | 0.023 / 0.177 | 117 / 218 | 0.000 / 0.000 | 0.111 / 0.058 | 0.365 / 0.392 | 0.000 / 0.000 |

## anchor in between (n = 12326, weight 0.229)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.924 / 0.887 | 0.637 / 0.454 | 0.273 / 0.284 | 0.091 / 0.262 | 147 / 244 | 0.000 / 0.002 | 0.170 / 0.081 | 0.194 / 0.251 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.924 / 0.880 | 0.637 / 0.479 | 0.273 / 0.275 | 0.091 / 0.246 | 147 / 231 | 0.000 / 0.001 | 0.170 / 0.086 | 0.194 / 0.262 | 0.000 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.924 / 0.899 | 0.637 / 0.522 | 0.273 / 0.275 | 0.091 / 0.203 | 147 / 210 | 0.000 / 0.000 | 0.170 / 0.092 | 0.194 / 0.252 | 0.000 / 0.000 |

## anchor in zone2 (n = 4946, weight 0.092)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.955 / 0.955 | 0.835 / 0.668 | 0.123 / 0.130 | 0.042 / 0.203 | 97 / 190 | 0.000 / 0.005 | 0.319 / 0.184 | 0.232 / 0.237 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.955 / 0.955 | 0.835 / 0.685 | 0.123 / 0.132 | 0.042 / 0.183 | 97 / 175 | 0.000 / 0.003 | 0.319 / 0.192 | 0.232 / 0.248 | 0.000 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.955 / 0.955 | 0.835 / 0.707 | 0.123 / 0.133 | 0.042 / 0.160 | 97 / 162 | 0.000 / 0.002 | 0.319 / 0.192 | 0.232 / 0.255 | 0.000 / 0.000 |

## anchor in post_zone2 (n = 8069, weight 0.152)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.955 / 0.943 | 0.000 / 0.000 | 0.045 / 0.057 | 75 / 82 | 0.000 / 0.011 | 0.623 / 0.526 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.955 / 0.927 | 0.000 / 0.000 | 0.045 / 0.072 | 75 / 89 | 0.000 / 0.008 | 0.623 / 0.525 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.955 / 0.927 | 0.000 / 0.000 | 0.045 / 0.073 | 75 / 89 | 0.000 / 0.004 | 0.623 / 0.516 | 0.000 / 0.000 | 0.000 / 0.000 |

## anchor in goal_area (n = 4140, weight 0.076)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.977 / 0.990 | 0.000 / 0.000 | 0.023 / 0.010 | 22 / 17 | 0.000 / 0.003 | 1.000 / 0.997 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.977 / 0.987 | 0.000 / 0.000 | 0.023 / 0.013 | 22 / 18 | 0.000 / 0.005 | 1.000 / 0.995 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.977 / 0.979 | 0.000 / 0.000 | 0.023 / 0.021 | 22 / 22 | 0.000 / 0.002 | 1.000 / 0.998 | 0.000 / 0.000 | 0.000 / 0.000 |

## anchor in west_column (n = 299, weight 0.005)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 1.000 / 1.000 (+0.000) | 0.784 / 0.757 (-0.027 +- 0.035) | 0.000 / 0.000 | 0.784 / 0.757 | 0.000 / 0.000 | 0.216 / 0.243 | 446 / 546 | 0.907 / 0.936 | 0.093 / 0.064 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_1 | 1.000 / 1.000 (+0.000) | 0.784 / 0.833 (+0.049 +- 0.033) | 0.000 / 0.000 | 0.784 / 0.833 | 0.000 / 0.000 | 0.216 / 0.167 | 446 / 485 | 0.907 / 0.928 | 0.093 / 0.072 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_2 | 1.000 / 0.997 (-0.003) | 0.784 / 0.903 (+0.119 +- 0.029) | 0.000 / 0.000 | 0.784 / 0.903 | 0.000 / 0.000 | 0.216 / 0.097 | 446 / 467 | 0.907 / 0.910 | 0.093 / 0.090 | 0.000 / 0.000 | 0.000 / 0.003 |

## anchor in top_corridor (n = 1601, weight 0.029)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.159 / 0.159 (+0.000) | 0.135 / 0.140 (+0.005 +- 0.005) | 0.000 / 0.000 | 0.818 / 0.892 | 0.000 / 0.000 | 0.182 / 0.108 | 313 / 383 | 0.873 / 0.897 | 0.127 / 0.103 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_1 | 0.159 / 0.159 (+0.000) | 0.135 / 0.135 (+0.000 +- 0.005) | 0.000 / 0.000 | 0.818 / 0.850 | 0.000 / 0.000 | 0.182 / 0.150 | 313 / 369 | 0.873 / 0.878 | 0.127 / 0.122 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_2 | 0.159 / 0.159 (+0.000) | 0.135 / 0.147 (+0.013 +- 0.004) | 0.000 / 0.000 | 0.818 / 0.920 | 0.000 / 0.000 | 0.182 / 0.080 | 313 / 328 | 0.873 / 0.872 | 0.127 / 0.128 | 0.000 / 0.000 | 0.000 / 0.000 |

## anchor in east_column (n = 603, weight 0.011)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.924 / 0.961 | 0.000 / 0.000 | 0.076 / 0.039 | 95 / 120 | 0.678 / 0.690 | 0.322 / 0.310 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.924 / 0.955 | 0.000 / 0.000 | 0.076 / 0.045 | 95 / 112 | 0.678 / 0.687 | 0.322 / 0.313 | 0.000 / 0.000 | 0.000 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.000 / 0.000 | 0.924 / 0.946 | 0.000 / 0.000 | 0.076 / 0.054 | 95 / 104 | 0.678 / 0.680 | 0.322 / 0.320 | 0.000 / 0.000 | 0.000 / 0.000 |

## logged shortcut | all (n = 50983, weight 0.950)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.693 / 0.716 | 0.611 / 0.514 | 0.306 / 0.317 | 0.083 / 0.169 | 135 / 184 | 0.000 / 0.004 | 0.278 / 0.213 | 0.169 / 0.201 | 0.000 / 0.000 |
| seed_1 | 0.000 / 0.001 (+0.001) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.693 / 0.711 | 0.611 / 0.519 | 0.306 / 0.300 | 0.083 / 0.181 | 135 / 190 | 0.000 / 0.003 | 0.278 / 0.214 | 0.169 / 0.204 | 0.001 / 0.000 |
| seed_2 | 0.000 / 0.000 (+0.000) | 0.000 / 0.000 (+0.000 +- 0.000) | 0.693 / 0.716 | 0.611 / 0.561 | 0.306 / 0.300 | 0.083 / 0.138 | 135 / 170 | 0.000 / 0.001 | 0.278 / 0.217 | 0.169 / 0.201 | 0.000 / 0.000 |

## logged detour | all (n = 2764, weight 0.050)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.253 / 0.263 (+0.010) | 0.204 / 0.204 (+0.001 +- 0.006) | 0.009 / 0.004 | 0.824 / 0.870 | 0.005 / 0.001 | 0.171 / 0.129 | 288 / 353 | 0.797 / 0.833 | 0.179 / 0.158 | 0.001 / 0.001 | 0.018 / 0.008 |
| seed_1 | 0.253 / 0.268 (+0.015) | 0.204 / 0.221 (+0.017 +- 0.006) | 0.009 / 0.005 | 0.824 / 0.863 | 0.005 / 0.003 | 0.171 / 0.135 | 288 / 331 | 0.797 / 0.818 | 0.179 / 0.172 | 0.001 / 0.001 | 0.018 / 0.003 |
| seed_2 | 0.253 / 0.260 (+0.007) | 0.204 / 0.234 (+0.031 +- 0.005) | 0.009 / 0.007 | 0.824 / 0.910 | 0.005 / 0.002 | 0.171 / 0.088 | 288 / 303 | 0.797 / 0.808 | 0.179 / 0.178 | 0.001 / 0.002 | 0.012 / 0.005 |

## all (n = 53747, weight 1.000)

| lineage | entered far old / new (delta) | completed far old / new (delta +- se) | entered zone old / new | success old / new | death old / new | timeout old / new | length old / new | mass far old / new | mass goal old / new | mass zone old / new | switched to far / away |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed_0 | 0.013 / 0.013 (+0.001) | 0.010 / 0.010 (+0.000 +- 0.000) | 0.659 / 0.681 | 0.621 / 0.531 | 0.291 / 0.302 | 0.088 / 0.167 | 143 / 192 | 0.040 / 0.045 | 0.273 / 0.210 | 0.161 / 0.191 | 0.001 / 0.000 |
| seed_1 | 0.013 / 0.014 (+0.001) | 0.010 / 0.011 (+0.001 +- 0.000) | 0.659 / 0.676 | 0.621 / 0.536 | 0.291 / 0.285 | 0.088 / 0.179 | 143 / 197 | 0.040 / 0.043 | 0.273 / 0.212 | 0.161 / 0.194 | 0.001 / 0.000 |
| seed_2 | 0.013 / 0.013 (+0.001) | 0.010 / 0.012 (+0.002 +- 0.000) | 0.659 / 0.681 | 0.621 / 0.579 | 0.291 / 0.285 | 0.088 / 0.136 | 143 / 176 | 0.040 / 0.042 | 0.273 / 0.215 | 0.161 / 0.191 | 0.001 / 0.000 |

