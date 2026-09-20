# Does the motion / stationary model turn the advice a_b into motion?  (fixed v3 models; held-out rows / anchors per fold)

Read-out thresholds (stated in advance): {"A_ab_shift_vs_aq_shift": 0.3, "A_stat_flip_rate": 0.05, "B_outcome_gap_gen_vs_diag": 0.05}; deviation tolerance 0.25.

## A. One-step sensitivity to a_b with (s, a_q) fixed (xy shift of the predicted next state; median / p90), against the a_q swap and the model error

| fold | stratum | rows | real displ. xy med | model error xy med / p90 | stat on (base) / real stationary | generator draw: xy shift med / p90, stat flip | zero hold | diagonal a_q | other row a_b | **a_q swap** (executed torque): xy shift med / p90, stat flip |
|---|---|---:|---:|---|---|---|---|---|---|---|
| fold0 | all | 40000 | 0.1048 | 0.0013 / 0.0061 | 0.274 / 0.274 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0108 / 0.0254, 0.221** |
| fold0 | start | 26750 | 0.1054 | 0.0018 / 0.0095 | 0.307 / 0.352 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0103 / 0.0238, 0.211** |
| fold0 | pre_zone1 | 40000 | 0.0002 | 0.0001 / 0.0023 | 0.499 / 0.497 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0004 / 0.0189, 0.195** |
| fold0 | band | 40000 | 0.1250 | 0.0022 / 0.0057 | 0.001 / 0.001 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0104 / 0.0244, 0.001** |
| fold0 | far_legs_moving | 40000 | 0.1205 | 0.0044 / 0.0171 | 0.003 / 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0126 / 0.0278, 0.003** |
| fold0 | stationary_real | 40000 | 0.0000 | 0.0000 / 0.0001 | 0.974 / 1.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0017 / 0.0167, 0.525** |
| fold0 | hold_rows | 40000 | 0.1184 | 0.0020 / 0.0066 | 0.002 / 0.002 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0106 / 0.0249, 0.002** |
| fold1 | all | 40000 | 0.1047 | 0.0014 / 0.0061 | 0.260 / 0.267 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0107 / 0.0247, 0.209** |
| fold1 | start | 30022 | 0.0736 | 0.0015 / 0.0070 | 0.248 / 0.336 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0099 / 0.0222, 0.172** |
| fold1 | pre_zone1 | 40000 | 0.0002 | 0.0002 / 0.0026 | 0.458 / 0.479 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0002 / 0.0181, 0.125** |
| fold1 | band | 40000 | 0.1248 | 0.0025 / 0.0059 | 0.001 / 0.002 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0100 / 0.0247, 0.001** |
| fold1 | far_legs_moving | 40000 | 0.1213 | 0.0043 / 0.0167 | 0.002 / 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0119 / 0.0275, 0.002** |
| fold1 | stationary_real | 40000 | 0.0000 | 0.0000 / 0.0001 | 0.946 / 1.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0029 / 0.0179, 0.496** |
| fold1 | hold_rows | 40000 | 0.1177 | 0.0023 / 0.0067 | 0.003 / 0.002 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0101 / 0.0248, 0.003** |
| fold2 | all | 40000 | 0.1070 | 0.0014 / 0.0064 | 0.256 / 0.257 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0114 / 0.0267, 0.202** |
| fold2 | start | 16602 | 0.1232 | 0.0027 / 0.0080 | 0.046 / 0.045 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0110 / 0.0258, 0.043** |
| fold2 | pre_zone1 | 40000 | 0.0002 | 0.0001 / 0.0024 | 0.491 / 0.496 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0002 / 0.0203, 0.141** |
| fold2 | band | 40000 | 0.1250 | 0.0023 / 0.0057 | 0.004 / 0.004 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0106 / 0.0260, 0.004** |
| fold2 | far_legs_moving | 40000 | 0.1212 | 0.0043 / 0.0160 | 0.003 / 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0119 / 0.0282, 0.002** |
| fold2 | stationary_real | 40000 | 0.0000 | 0.0000 / 0.0001 | 0.969 / 1.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0029 / 0.0191, 0.502** |
| fold2 | hold_rows | 40000 | 0.1179 | 0.0021 / 0.0065 | 0.003 / 0.002 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | 0.0000 / 0.0000, 0.000 | **0.0107 / 0.0262, 0.003** |

## B. Closed-loop model paths from held-out anchors vs the simulator branch of the same anchor

| fold | group | n | sim success / timeout / death (rows) | advice | model reach / p_death / rows | stat-gate share of steps | deviated share | first deviation step med / p90 | kind (share) |
|---|---|---:|---|---|---|---:|---:|---|---|
| fold0 | far_legs | 905 | 0.859 / 0.141 / 0.000 (271) | gen | 0.718 / 0.000 / 318 | 0.106 | 0.938 | 29.000 / 60.000 | {"drift": 1.0} |
| fold0 | far_legs | 905 | 0.859 / 0.141 / 0.000 (271) | diag | 0.718 / 0.000 / 318 | 0.106 | 0.938 | 29.000 / 60.000 | {"drift": 1.0} |
| fold0 | far_legs | 905 | 0.859 / 0.141 / 0.000 (271) | real_seq | 0.718 / 0.001 / 318 | 0.106 | 0.938 | 29.000 / 60.000 | {"drift": 1.0} |
| fold0 | far_legs | | teacher-forced one-step along the sim paths (244054 rows): xy error med 0.0034 / p90 0.0158; false-stationary at moving rows 0.0032; missed stationary 0.022; moving share 0.808 | | | | | | |
| fold0 | start_sim_far_success | 49 | 1.000 / 0.000 / 0.000 (388) | gen | 0.755 / 0.014 / 474 | 0.138 | 1.000 | 10.000 / 26.400 | {"drift": 1.0} |
| fold0 | start_sim_far_success | 49 | 1.000 / 0.000 / 0.000 (388) | diag | 0.755 / 0.000 / 474 | 0.138 | 1.000 | 10.000 / 26.400 | {"drift": 1.0} |
| fold0 | start_sim_far_success | 49 | 1.000 / 0.000 / 0.000 (388) | real_seq | 0.755 / 0.031 / 474 | 0.138 | 1.000 | 10.000 / 26.400 | {"drift": 1.0} |
| fold0 | start_sim_far_success | | teacher-forced one-step along the sim paths (18952 rows): xy error med 0.0047 / p90 0.0172; false-stationary at moving rows 0.0000; missed stationary nan; moving share 1.000 | | | | | | |
| fold0 | pre_zone1_sim_timeout | 780 | 0.000 / 1.000 / 0.000 (725) | gen | 0.833 / 0.141 / 347 | 0.232 | 0.949 | 62.500 / 138.000 | {"drift": 0.48, "model_stalled": 0.0, "sim_stalled": 0.51, "stat_gate_stalle": 0.01} |
| fold0 | pre_zone1_sim_timeout | 780 | 0.000 / 1.000 / 0.000 (725) | diag | 0.833 / 0.014 / 347 | 0.232 | 0.949 | 62.500 / 138.000 | {"drift": 0.48, "model_stalled": 0.0, "sim_stalled": 0.51, "stat_gate_stalle": 0.01} |
| fold0 | pre_zone1_sim_timeout | 780 | 0.000 / 1.000 / 0.000 (725) | real_seq | 0.833 / 0.345 / 347 | 0.232 | 0.949 | 62.500 / 138.000 | {"drift": 0.48, "model_stalled": 0.0, "sim_stalled": 0.51, "stat_gate_stalle": 0.01} |
| fold0 | pre_zone1_sim_timeout | | teacher-forced one-step along the sim paths (564366 rows): xy error med 0.0001 / p90 0.0007; false-stationary at moving rows 0.0599; missed stationary 0.030; moving share 0.367 | | | | | | |
| fold0 | pre_zone1_sim_success | 1329 | 1.000 / 0.000 / 0.000 (208) | gen | 0.908 / 0.067 / 253 | 0.129 | 0.987 | 52.000 / 115.900 | {"drift": 0.95, "model_stalled": 0.0, "sim_stalled": 0.03, "stat_gate_stalle": 0.02} |
| fold0 | pre_zone1_sim_success | 1329 | 1.000 / 0.000 / 0.000 (208) | diag | 0.908 / 0.002 / 253 | 0.129 | 0.987 | 52.000 / 115.900 | {"drift": 0.95, "model_stalled": 0.0, "sim_stalled": 0.03, "stat_gate_stalle": 0.02} |
| fold0 | pre_zone1_sim_success | 1329 | 1.000 / 0.000 / 0.000 (208) | real_seq | 0.908 / 0.592 / 253 | 0.129 | 0.987 | 52.000 / 115.900 | {"drift": 0.95, "model_stalled": 0.0, "sim_stalled": 0.03, "stat_gate_stalle": 0.02} |
| fold0 | pre_zone1_sim_success | | teacher-forced one-step along the sim paths (274555 rows): xy error med 0.0021 / p90 0.0059; false-stationary at moving rows 0.0013; missed stationary 0.240; moving share 0.993 | | | | | | |
| fold1 | far_legs | 689 | 0.852 / 0.148 / 0.000 (268) | gen | 0.939 / 0.000 / 228 | 0.005 | 0.946 | 29.000 / 59.000 | {"drift": 1.0, "sim_stalled": 0.0} |
| fold1 | far_legs | 689 | 0.852 / 0.148 / 0.000 (268) | diag | 0.939 / 0.000 / 228 | 0.005 | 0.946 | 29.000 / 59.000 | {"drift": 1.0, "sim_stalled": 0.0} |
| fold1 | far_legs | 689 | 0.852 / 0.148 / 0.000 (268) | real_seq | 0.939 / 0.004 / 228 | 0.005 | 0.946 | 29.000 / 59.000 | {"drift": 1.0, "sim_stalled": 0.0} |
| fold1 | far_legs | | teacher-forced one-step along the sim paths (183981 rows): xy error med 0.0034 / p90 0.0154; false-stationary at moving rows 0.0029; missed stationary 0.029; moving share 0.797 | | | | | | |
| fold1 | start_sim_far_success | 26 | too few anchors | | | | | | |
| fold1 | pre_zone1_sim_timeout | 759 | 0.000 / 1.000 / 0.000 (726) | gen | 0.742 / 0.136 / 405 | 0.307 | 0.859 | 71.000 / 244.000 | {"drift": 0.47, "model_stalled": 0.0, "sim_stalled": 0.51, "stat_gate_stalle": 0.02} |
| fold1 | pre_zone1_sim_timeout | 759 | 0.000 / 1.000 / 0.000 (726) | diag | 0.742 / 0.000 / 405 | 0.307 | 0.859 | 71.000 / 244.000 | {"drift": 0.47, "model_stalled": 0.0, "sim_stalled": 0.51, "stat_gate_stalle": 0.02} |
| fold1 | pre_zone1_sim_timeout | 759 | 0.000 / 1.000 / 0.000 (726) | real_seq | 0.742 / 0.391 / 405 | 0.307 | 0.859 | 71.000 / 244.000 | {"drift": 0.47, "model_stalled": 0.0, "sim_stalled": 0.51, "stat_gate_stalle": 0.02} |
| fold1 | pre_zone1_sim_timeout | | teacher-forced one-step along the sim paths (550450 rows): xy error med 0.0001 / p90 0.0007; false-stationary at moving rows 0.0395; missed stationary 0.065; moving share 0.378 | | | | | | |
| fold1 | pre_zone1_sim_success | 1311 | 1.000 / 0.000 / 0.000 (225) | gen | 0.944 / 0.080 / 232 | 0.115 | 0.999 | 41.000 / 93.000 | {"drift": 0.93, "model_stalled": 0.0, "sim_stalled": 0.06, "stat_gate_stalle": 0.01} |
| fold1 | pre_zone1_sim_success | 1311 | 1.000 / 0.000 / 0.000 (225) | diag | 0.944 / 0.001 / 232 | 0.115 | 0.999 | 41.000 / 93.000 | {"drift": 0.93, "model_stalled": 0.0, "sim_stalled": 0.06, "stat_gate_stalle": 0.01} |
| fold1 | pre_zone1_sim_success | 1311 | 1.000 / 0.000 / 0.000 (225) | real_seq | 0.944 / 0.725 / 232 | 0.115 | 0.999 | 41.000 / 93.000 | {"drift": 0.93, "model_stalled": 0.0, "sim_stalled": 0.06, "stat_gate_stalle": 0.01} |
| fold1 | pre_zone1_sim_success | | teacher-forced one-step along the sim paths (293505 rows): xy error med 0.0022 / p90 0.0058; false-stationary at moving rows 0.0032; missed stationary 0.193; moving share 0.971 | | | | | | |
| fold2 | far_legs | 909 | 0.817 / 0.183 / 0.000 (287) | gen | 0.722 / 0.003 / 311 | 0.158 | 0.954 | 28.000 / 60.000 | {"drift": 1.0} |
| fold2 | far_legs | 909 | 0.817 / 0.183 / 0.000 (287) | diag | 0.722 / 0.000 / 311 | 0.158 | 0.954 | 28.000 / 60.000 | {"drift": 1.0} |
| fold2 | far_legs | 909 | 0.817 / 0.183 / 0.000 (287) | real_seq | 0.722 / 0.068 / 311 | 0.158 | 0.954 | 28.000 / 60.000 | {"drift": 1.0} |
| fold2 | far_legs | | teacher-forced one-step along the sim paths (260400 rows): xy error med 0.0031 / p90 0.0151; false-stationary at moving rows 0.0051; missed stationary 0.014; moving share 0.774 | | | | | | |
| fold2 | start_sim_far_success | 34 | 1.000 / 0.000 / 0.000 (394) | gen | 0.882 / 0.094 / 396 | 0.029 | 1.000 | 9.000 / 21.700 | {"drift": 1.0} |
| fold2 | start_sim_far_success | 34 | 1.000 / 0.000 / 0.000 (394) | diag | 0.882 / 0.000 / 396 | 0.029 | 1.000 | 9.000 / 21.700 | {"drift": 1.0} |
| fold2 | start_sim_far_success | 34 | 1.000 / 0.000 / 0.000 (394) | real_seq | 0.882 / 0.303 / 396 | 0.029 | 1.000 | 9.000 / 21.700 | {"drift": 1.0} |
| fold2 | start_sim_far_success | | teacher-forced one-step along the sim paths (13351 rows): xy error med 0.0046 / p90 0.0163; false-stationary at moving rows 0.0000; missed stationary nan; moving share 1.000 | | | | | | |
| fold2 | pre_zone1_sim_timeout | 768 | 0.000 / 1.000 / 0.000 (726) | gen | 0.706 / 0.117 / 383 | 0.435 | 0.850 | 51.000 / 115.000 | {"drift": 0.49, "model_stalled": 0.0, "sim_stalled": 0.48, "stat_gate_stalle": 0.02} |
| fold2 | pre_zone1_sim_timeout | 768 | 0.000 / 1.000 / 0.000 (726) | diag | 0.706 / 0.012 / 383 | 0.435 | 0.850 | 51.000 / 115.000 | {"drift": 0.49, "model_stalled": 0.0, "sim_stalled": 0.48, "stat_gate_stalle": 0.02} |
| fold2 | pre_zone1_sim_timeout | 768 | 0.000 / 1.000 / 0.000 (726) | real_seq | 0.706 / 0.380 / 383 | 0.435 | 0.850 | 51.000 / 115.000 | {"drift": 0.49, "model_stalled": 0.0, "sim_stalled": 0.48, "stat_gate_stalle": 0.02} |
| fold2 | pre_zone1_sim_timeout | | teacher-forced one-step along the sim paths (556926 rows): xy error med 0.0001 / p90 0.0006; false-stationary at moving rows 0.0505; missed stationary 0.037; moving share 0.362 | | | | | | |
| fold2 | pre_zone1_sim_success | 1279 | 1.000 / 0.000 / 0.000 (209) | gen | 0.949 / 0.083 / 223 | 0.105 | 0.994 | 45.000 / 110.000 | {"drift": 0.96, "model_stalled": 0.0, "sim_stalled": 0.03, "stat_gate_stalle": 0.01} |
| fold2 | pre_zone1_sim_success | 1279 | 1.000 / 0.000 / 0.000 (209) | diag | 0.949 / 0.001 / 223 | 0.105 | 0.994 | 45.000 / 110.000 | {"drift": 0.96, "model_stalled": 0.0, "sim_stalled": 0.03, "stat_gate_stalle": 0.01} |
| fold2 | pre_zone1_sim_success | 1279 | 1.000 / 0.000 / 0.000 (209) | real_seq | 0.949 / 0.778 / 223 | 0.105 | 0.994 | 45.000 / 110.000 | {"drift": 0.96, "model_stalled": 0.0, "sim_stalled": 0.03, "stat_gate_stalle": 0.01} |
| fold2 | pre_zone1_sim_success | | teacher-forced one-step along the sim paths (265423 rows): xy error med 0.0022 / p90 0.0060; false-stationary at moving rows 0.0014; missed stationary 0.276; moving share 0.994 | | | | | | |
