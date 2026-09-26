# Does the motion / stationary model turn the advice a_b into motion?  (fixed v3 models; held-out rows / anchors per fold)

Read-out thresholds (stated in advance): {"A_ab_shift_vs_aq_shift": 0.3, "A_stat_flip_rate": 0.05, "B_outcome_gap_gen_vs_diag": 0.05}; deviation tolerance 0.25.

## A. One-step sensitivity to a_b with (s, a_q) fixed (xy shift of the predicted next state; median / p90), against the a_q swap and the model error

| fold | stratum | rows | real displ. xy med | model error xy med / p90 | stat on (base) / real stationary | generator draw: xy shift med / p90, stat flip | zero hold | diagonal a_q | other row a_b | **a_q swap** (executed torque): xy shift med / p90, stat flip |
|---|---|---:|---:|---|---|---|---|---|---|---|
| fold0 | all | 40000 | 0.1048 | 0.0013 / 0.0063 | 0.274 / 0.274 | 0.0002 / 0.0017, 0.005 | 0.0011 / 0.0035, 0.072 | 0.0001 / 0.0020, 0.049 | 0.0017 / 0.0052, 0.133 | **0.0108 / 0.0262, 0.221** |
| fold0 | start | 26750 | 0.1054 | 0.0019 / 0.0099 | 0.294 / 0.352 | 0.0004 / 0.0026, 0.024 | 0.0017 / 0.0046, 0.294 | 0.0001 / 0.0030, 0.034 | 0.0023 / 0.0061, 0.156 | **0.0104 / 0.0238, 0.193** |
| fold0 | pre_zone1 | 40000 | 0.0002 | 0.0002 / 0.0024 | 0.496 / 0.497 | 0.0000 / 0.0004, 0.010 | 0.0004 / 0.0017, 0.177 | 0.0001 / 0.0010, 0.151 | 0.0006 / 0.0027, 0.204 | **0.0004 / 0.0190, 0.190** |
| fold0 | band | 40000 | 0.1250 | 0.0023 / 0.0057 | 0.000 / 0.001 | 0.0004 / 0.0011, 0.000 | 0.0017 / 0.0033, 0.001 | 0.0001 / 0.0006, 0.001 | 0.0022 / 0.0049, 0.001 | **0.0109 / 0.0260, 0.000** |
| fold0 | far_legs_moving | 40000 | 0.1205 | 0.0046 / 0.0174 | 0.003 / 0.000 | 0.0014 / 0.0053, 0.000 | 0.0027 / 0.0067, 0.003 | 0.0015 / 0.0053, 0.002 | 0.0034 / 0.0083, 0.002 | **0.0132 / 0.0299, 0.003** |
| fold0 | stationary_real | 40000 | 0.0000 | 0.0000 / 0.0001 | 0.966 / 1.000 | 0.0000 / 0.0000, 0.012 | 0.0000 / 0.0008, 0.218 | 0.0000 / 0.0006, 0.139 | 0.0000 / 0.0024, 0.426 | **0.0032 / 0.0151, 0.519** |
| fold0 | hold_rows | 40000 | 0.1184 | 0.0026 / 0.0071 | 0.002 / 0.002 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0015 / 0.0033, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0103 / 0.0246, 0.002** |
| fold1 | all | 40000 | 0.1047 | 0.0014 / 0.0062 | 0.265 / 0.267 | 0.0003 / 0.0017, 0.012 | 0.0011 / 0.0037, 0.036 | 0.0001 / 0.0020, 0.035 | 0.0017 / 0.0053, 0.129 | **0.0103 / 0.0244, 0.201** |
| fold1 | start | 30022 | 0.0736 | 0.0018 / 0.0070 | 0.258 / 0.336 | 0.0004 / 0.0020, 0.026 | 0.0018 / 0.0036, 0.258 | 0.0002 / 0.0025, 0.062 | 0.0022 / 0.0056, 0.152 | **0.0099 / 0.0229, 0.141** |
| fold1 | pre_zone1 | 40000 | 0.0002 | 0.0003 / 0.0026 | 0.471 / 0.479 | 0.0000 / 0.0006, 0.027 | 0.0002 / 0.0018, 0.086 | 0.0001 / 0.0011, 0.088 | 0.0007 / 0.0030, 0.200 | **0.0002 / 0.0175, 0.133** |
| fold1 | band | 40000 | 0.1248 | 0.0024 / 0.0058 | 0.002 / 0.002 | 0.0004 / 0.0011, 0.000 | 0.0018 / 0.0035, 0.000 | 0.0001 / 0.0007, 0.000 | 0.0022 / 0.0050, 0.000 | **0.0105 / 0.0250, 0.001** |
| fold1 | far_legs_moving | 40000 | 0.1213 | 0.0044 / 0.0165 | 0.001 / 0.000 | 0.0015 / 0.0054, 0.000 | 0.0026 / 0.0064, 0.001 | 0.0014 / 0.0053, 0.001 | 0.0033 / 0.0081, 0.001 | **0.0123 / 0.0285, 0.002** |
| fold1 | stationary_real | 40000 | 0.0000 | 0.0000 / 0.0001 | 0.954 / 1.000 | 0.0000 / 0.0000, 0.032 | 0.0000 / 0.0003, 0.088 | 0.0000 / 0.0003, 0.088 | 0.0001 / 0.0022, 0.470 | **0.0000 / 0.0095, 0.437** |
| fold1 | hold_rows | 40000 | 0.1177 | 0.0028 / 0.0075 | 0.003 / 0.002 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0015 / 0.0035, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0104 / 0.0235, 0.003** |
| fold2 | all | 40000 | 0.1070 | 0.0014 / 0.0063 | 0.249 / 0.257 | 0.0002 / 0.0018, 0.005 | 0.0012 / 0.0039, 0.085 | 0.0002 / 0.0021, 0.056 | 0.0018 / 0.0053, 0.135 | **0.0106 / 0.0260, 0.203** |
| fold2 | start | 16602 | 0.1232 | 0.0027 / 0.0080 | 0.045 / 0.045 | 0.0006 / 0.0024, 0.001 | 0.0023 / 0.0048, 0.045 | 0.0001 / 0.0006, 0.045 | 0.0026 / 0.0060, 0.028 | **0.0113 / 0.0271, 0.043** |
| fold2 | pre_zone1 | 40000 | 0.0002 | 0.0002 / 0.0025 | 0.470 / 0.496 | 0.0000 / 0.0005, 0.011 | 0.0003 / 0.0019, 0.132 | 0.0002 / 0.0011, 0.140 | 0.0005 / 0.0027, 0.190 | **0.0003 / 0.0191, 0.157** |
| fold2 | band | 40000 | 0.1250 | 0.0023 / 0.0057 | 0.004 / 0.004 | 0.0004 / 0.0011, 0.000 | 0.0017 / 0.0037, 0.004 | 0.0001 / 0.0006, 0.004 | 0.0021 / 0.0048, 0.003 | **0.0110 / 0.0268, 0.004** |
| fold2 | far_legs_moving | 40000 | 0.1212 | 0.0044 / 0.0162 | 0.003 / 0.000 | 0.0014 / 0.0052, 0.000 | 0.0027 / 0.0064, 0.002 | 0.0014 / 0.0053, 0.003 | 0.0034 / 0.0082, 0.002 | **0.0124 / 0.0288, 0.003** |
| fold2 | stationary_real | 40000 | 0.0000 | 0.0000 / 0.0001 | 0.951 / 1.000 | 0.0000 / 0.0000, 0.014 | 0.0000 / 0.0010, 0.301 | 0.0000 / 0.0007, 0.186 | 0.0003 / 0.0029, 0.503 | **0.0016 / 0.0168, 0.525** |
| fold2 | hold_rows | 40000 | 0.1179 | 0.0026 / 0.0073 | 0.002 / 0.002 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0015 / 0.0037, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0106 / 0.0252, 0.003** |

## B. Closed-loop model paths from held-out anchors vs the simulator branch of the same anchor

| fold | group | n | sim success / timeout / death (rows) | advice | model reach / p_death / rows | stat-gate share of steps | deviated share | first deviation step med / p90 | kind (share) |
|---|---|---:|---|---|---|---:|---:|---|---|
| fold0 | far_legs | 905 | 0.859 / 0.141 / 0.000 (271) | gen | 0.678 / 0.000 / 324 | 0.194 | 0.943 | 28.000 / 57.000 | {"drift": 1.0} |
| fold0 | far_legs | 905 | 0.859 / 0.141 / 0.000 (271) | diag | 0.705 / 0.000 / 306 | 0.242 | 0.939 | 29.000 / 58.000 | {"drift": 1.0, "sim_stalled": 0.0} |
| fold0 | far_legs | 905 | 0.859 / 0.141 / 0.000 (271) | real_seq | 0.709 / 0.000 / 312 | 0.107 | 0.937 | 29.000 / 65.000 | {"drift": 1.0, "sim_stalled": 0.0} |
| fold0 | far_legs | | teacher-forced one-step along the sim paths (244054 rows): xy error med 0.0035 / p90 0.0159; false-stationary at moving rows 0.0035; missed stationary 0.045; moving share 0.808 | | | | | | |
| fold0 | start_sim_far_success | 49 | 1.000 / 0.000 / 0.000 (388) | gen | 0.755 / 0.126 / 444 | 0.096 | 1.000 | 11.000 / 26.400 | {"drift": 1.0} |
| fold0 | start_sim_far_success | 49 | 1.000 / 0.000 / 0.000 (388) | diag | 0.571 / 0.000 / 534 | 0.247 | 1.000 | 11.000 / 32.200 | {"drift": 1.0} |
| fold0 | start_sim_far_success | 49 | 1.000 / 0.000 / 0.000 (388) | real_seq | 0.694 / 0.100 / 477 | 0.064 | 1.000 | 10.000 / 31.800 | {"drift": 1.0} |
| fold0 | start_sim_far_success | | teacher-forced one-step along the sim paths (18952 rows): xy error med 0.0050 / p90 0.0173; false-stationary at moving rows 0.0000; missed stationary nan; moving share 1.000 | | | | | | |
| fold0 | pre_zone1_sim_timeout | 780 | 0.000 / 1.000 / 0.000 (725) | gen | 0.865 / 0.147 / 332 | 0.117 | 0.996 | 54.000 / 151.000 | {"drift": 0.48, "model_stalled": 0.0, "sim_stalled": 0.52} |
| fold0 | pre_zone1_sim_timeout | 780 | 0.000 / 1.000 / 0.000 (725) | diag | 0.685 / 0.002 / 407 | 0.285 | 0.926 | 63.500 / 142.700 | {"drift": 0.43, "model_stalled": 0.0, "sim_stalled": 0.56, "stat_gate_stalle": 0.02} |
| fold0 | pre_zone1_sim_timeout | 780 | 0.000 / 1.000 / 0.000 (725) | real_seq | 0.835 / 0.518 / 342 | 0.139 | 0.962 | 57.500 / 138.000 | {"drift": 0.46, "model_stalled": 0.01, "sim_stalled": 0.53, "stat_gate_stalle": 0.0} |
| fold0 | pre_zone1_sim_timeout | | teacher-forced one-step along the sim paths (564366 rows): xy error med 0.0001 / p90 0.0007; false-stationary at moving rows 0.0706; missed stationary 0.039; moving share 0.367 | | | | | | |
| fold0 | pre_zone1_sim_success | 1329 | 1.000 / 0.000 / 0.000 (208) | gen | 0.895 / 0.074 / 252 | 0.085 | 0.988 | 47.000 / 104.800 | {"drift": 0.97, "model_stalled": 0.0, "sim_stalled": 0.02, "stat_gate_stalle": 0.0} |
| fold0 | pre_zone1_sim_success | 1329 | 1.000 / 0.000 / 0.000 (208) | diag | 0.777 / 0.001 / 313 | 0.171 | 0.993 | 46.000 / 110.000 | {"drift": 0.95, "model_stalled": 0.0, "sim_stalled": 0.02, "stat_gate_stalle": 0.02} |
| fold0 | pre_zone1_sim_success | 1329 | 1.000 / 0.000 / 0.000 (208) | real_seq | 0.883 / 0.617 / 264 | 0.027 | 0.985 | 50.000 / 111.000 | {"drift": 0.98, "model_stalled": 0.0, "sim_stalled": 0.02} |
| fold0 | pre_zone1_sim_success | | teacher-forced one-step along the sim paths (274555 rows): xy error med 0.0021 / p90 0.0059; false-stationary at moving rows 0.0013; missed stationary 0.461; moving share 0.993 | | | | | | |
| fold1 | far_legs | 689 | 0.852 / 0.148 / 0.000 (268) | gen | 0.864 / 0.001 / 244 | 0.066 | 0.949 | 27.000 / 55.700 | {"drift": 1.0} |
| fold1 | far_legs | 689 | 0.852 / 0.148 / 0.000 (268) | diag | 0.833 / 0.000 / 251 | 0.082 | 0.945 | 28.000 / 54.000 | {"drift": 1.0} |
| fold1 | far_legs | 689 | 0.852 / 0.148 / 0.000 (268) | real_seq | 0.903 / 0.003 / 238 | 0.051 | 0.945 | 28.000 / 50.000 | {"drift": 1.0} |
| fold1 | far_legs | | teacher-forced one-step along the sim paths (183981 rows): xy error med 0.0034 / p90 0.0152; false-stationary at moving rows 0.0032; missed stationary 0.048; moving share 0.797 | | | | | | |
| fold1 | start_sim_far_success | 26 | too few anchors | | | | | | |
| fold1 | pre_zone1_sim_timeout | 759 | 0.000 / 1.000 / 0.000 (726) | gen | 0.891 / 0.160 / 307 | 0.038 | 1.000 | 61.000 / 143.000 | {"drift": 0.49, "model_stalled": 0.0, "sim_stalled": 0.51} |
| fold1 | pre_zone1_sim_timeout | 759 | 0.000 / 1.000 / 0.000 (726) | diag | 0.701 / 0.041 / 403 | 0.327 | 0.867 | 66.000 / 160.300 | {"drift": 0.47, "sim_stalled": 0.5, "stat_gate_stalle": 0.03} |
| fold1 | pre_zone1_sim_timeout | 759 | 0.000 / 1.000 / 0.000 (726) | real_seq | 0.918 / 0.416 / 312 | 0.060 | 0.982 | 67.000 / 137.800 | {"drift": 0.47, "sim_stalled": 0.53} |
| fold1 | pre_zone1_sim_timeout | | teacher-forced one-step along the sim paths (550450 rows): xy error med 0.0001 / p90 0.0009; false-stationary at moving rows 0.0662; missed stationary 0.055; moving share 0.378 | | | | | | |
| fold1 | pre_zone1_sim_success | 1311 | 1.000 / 0.000 / 0.000 (225) | gen | 0.937 / 0.095 / 235 | 0.022 | 0.998 | 41.000 / 86.000 | {"drift": 0.95, "sim_stalled": 0.05} |
| fold1 | pre_zone1_sim_success | 1311 | 1.000 / 0.000 / 0.000 (225) | diag | 0.872 / 0.016 / 266 | 0.117 | 0.997 | 43.000 / 101.000 | {"drift": 0.93, "sim_stalled": 0.04, "stat_gate_stalle": 0.03} |
| fold1 | pre_zone1_sim_success | 1311 | 1.000 / 0.000 / 0.000 (225) | real_seq | 0.939 / 0.663 / 240 | 0.012 | 1.000 | 42.000 / 86.000 | {"drift": 0.93, "model_stalled": 0.0, "sim_stalled": 0.07, "stat_gate_stalle": 0.0} |
| fold1 | pre_zone1_sim_success | | teacher-forced one-step along the sim paths (293505 rows): xy error med 0.0020 / p90 0.0057; false-stationary at moving rows 0.0054; missed stationary 0.256; moving share 0.971 | | | | | | |
| fold2 | far_legs | 909 | 0.817 / 0.183 / 0.000 (287) | gen | 0.735 / 0.001 / 303 | 0.170 | 0.955 | 30.000 / 65.000 | {"drift": 1.0} |
| fold2 | far_legs | 909 | 0.817 / 0.183 / 0.000 (287) | diag | 0.690 / 0.000 / 311 | 0.333 | 0.957 | 29.000 / 58.100 | {"drift": 1.0} |
| fold2 | far_legs | 909 | 0.817 / 0.183 / 0.000 (287) | real_seq | 0.774 / 0.034 / 287 | 0.192 | 0.948 | 29.000 / 57.900 | {"drift": 1.0} |
| fold2 | far_legs | | teacher-forced one-step along the sim paths (260400 rows): xy error med 0.0031 / p90 0.0153; false-stationary at moving rows 0.0055; missed stationary 0.023; moving share 0.774 | | | | | | |
| fold2 | start_sim_far_success | 34 | 1.000 / 0.000 / 0.000 (394) | gen | 0.824 / 0.092 / 428 | 0.122 | 1.000 | 11.500 / 27.200 | {"drift": 1.0} |
| fold2 | start_sim_far_success | 34 | 1.000 / 0.000 / 0.000 (394) | diag | 0.735 / 0.000 / 461 | 0.228 | 1.000 | 13.000 / 45.200 | {"drift": 1.0} |
| fold2 | start_sim_far_success | 34 | 1.000 / 0.000 / 0.000 (394) | real_seq | 0.676 / 0.158 / 503 | 0.129 | 1.000 | 10.000 / 38.700 | {"drift": 1.0} |
| fold2 | start_sim_far_success | | teacher-forced one-step along the sim paths (13351 rows): xy error med 0.0048 / p90 0.0165; false-stationary at moving rows 0.0000; missed stationary nan; moving share 1.000 | | | | | | |
| fold2 | pre_zone1_sim_timeout | 768 | 0.000 / 1.000 / 0.000 (726) | gen | 0.805 / 0.173 / 347 | 0.170 | 0.995 | 46.000 / 179.000 | {"drift": 0.53, "sim_stalled": 0.46, "stat_gate_stalle": 0.0} |
| fold2 | pre_zone1_sim_timeout | 768 | 0.000 / 1.000 / 0.000 (726) | diag | 0.697 / 0.026 / 384 | 0.381 | 0.878 | 46.000 / 103.000 | {"drift": 0.56, "model_stalled": 0.0, "sim_stalled": 0.42, "stat_gate_stalle": 0.02} |
| fold2 | pre_zone1_sim_timeout | 768 | 0.000 / 1.000 / 0.000 (726) | real_seq | 0.695 / 0.462 / 386 | 0.213 | 0.982 | 45.000 / 109.000 | {"drift": 0.55, "model_stalled": 0.0, "sim_stalled": 0.45, "stat_gate_stalle": 0.0} |
| fold2 | pre_zone1_sim_timeout | | teacher-forced one-step along the sim paths (556926 rows): xy error med 0.0001 / p90 0.0008; false-stationary at moving rows 0.0370; missed stationary 0.068; moving share 0.362 | | | | | | |
| fold2 | pre_zone1_sim_success | 1279 | 1.000 / 0.000 / 0.000 (209) | gen | 0.932 / 0.078 / 233 | 0.087 | 0.983 | 46.000 / 112.000 | {"drift": 0.97, "model_stalled": 0.0, "sim_stalled": 0.03, "stat_gate_stalle": 0.0} |
| fold2 | pre_zone1_sim_success | 1279 | 1.000 / 0.000 / 0.000 (209) | diag | 0.891 / 0.002 / 251 | 0.151 | 0.994 | 47.000 / 105.000 | {"drift": 0.96, "sim_stalled": 0.03, "stat_gate_stalle": 0.01} |
| fold2 | pre_zone1_sim_success | 1279 | 1.000 / 0.000 / 0.000 (209) | real_seq | 0.932 / 0.798 / 236 | 0.054 | 0.998 | 42.000 / 94.000 | {"drift": 0.97, "sim_stalled": 0.03} |
| fold2 | pre_zone1_sim_success | | teacher-forced one-step along the sim paths (265423 rows): xy error med 0.0021 / p90 0.0059; false-stationary at moving rows 0.0010; missed stationary 0.446; moving share 0.994 | | | | | | |
