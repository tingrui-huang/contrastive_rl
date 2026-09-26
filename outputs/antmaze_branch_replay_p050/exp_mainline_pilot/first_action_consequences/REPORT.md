# The consequences of one first action at the reset states, then the frozen start agent (paired hidden draws): simulator vs the S ETT

Per candidate first action: the mean over states of the per-state draw means; paired contrasts vs the logged torque and vs the start agent's action (mean +- s.e. over states; share of states where the candidate is higher).  Masses = the gamma-law (0.999) masses of the path rows: reach 0.5 / near 2.0 / goal area / death frame -- what the critic's sampling law provides.

## sim (repeated_draws/rollouts_start_reset_stall.npz; 64 states; draws per state {np.str_('logged'): 64, np.str_('prog'): 64, np.str_('stall'): 64, np.str_('start'): 64})

| first action | success | death | timeout | far entry | reach0.5 | near2.0 | goal_area | death_frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.235 | 0.737 | 0.027 | 0.016 | 0.0009 | 0.0321 | 0.0401 | 0.0089 |
| prog | 0.239 | 0.574 | 0.187 | 0.156 | 0.0008 | 0.0279 | 0.0340 | 0.0069 |
| stall | 0.271 | 0.527 | 0.202 | 0.172 | 0.0009 | 0.0311 | 0.0333 | 0.0061 |
| start | 0.233 | 0.738 | 0.029 | 0.020 | 0.0009 | 0.0278 | 0.0352 | 0.0085 |

| paired vs logged: candidate | success | timeout | goal_area mass | near2.0 mass | death_frame mass |
|---|---|---|---|---|---|
| prog | +0.003 +- 0.025 (0.11) | +0.160 +- 0.050 (0.25) | -0.006 +- 0.009 (0.44) | -0.004 +- 0.009 (0.42) | -0.002 +- 0.001 (0.23) |
| stall | +0.035 +- 0.032 (0.14) | +0.175 +- 0.054 (0.22) | -0.007 +- 0.011 (0.38) | -0.001 +- 0.011 (0.38) | -0.003 +- 0.001 (0.12) |
| start | -0.003 +- 0.008 (0.03) | +0.002 +- 0.022 (0.05) | -0.005 +- 0.008 (0.44) | -0.004 +- 0.008 (0.39) | -0.000 +- 0.000 (0.09) |

| paired vs start: candidate | success | timeout | goal_area mass | near2.0 mass | death_frame mass |
|---|---|---|---|---|---|
| logged | +0.003 +- 0.008 (0.06) | -0.002 +- 0.022 (0.03) | +0.005 +- 0.008 (0.56) | +0.004 +- 0.008 (0.59) | +0.000 +- 0.000 (0.91) |
| prog | +0.006 +- 0.024 (0.09) | +0.158 +- 0.048 (0.23) | -0.001 +- 0.007 (0.44) | +0.000 +- 0.007 (0.41) | -0.002 +- 0.001 (0.59) |
| stall | +0.038 +- 0.034 (0.12) | +0.173 +- 0.052 (0.22) | -0.002 +- 0.007 (0.36) | +0.003 +- 0.008 (0.44) | -0.002 +- 0.000 (0.30) |

## model (ett_crossover/v4s20_reset_stall/model_rollouts.npz; 64 states; draws per state {np.str_('logged'): 32, np.str_('prog'): 32, np.str_('stall'): 32, np.str_('start'): 32})

| first action | success | death | timeout | far entry | reach0.5 | near2.0 | goal_area | death_frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logged | 0.248 | 0.752 | 0.000 | 0.000 | 0.0010 | 0.0250 | 0.0328 | 0.0092 |
| prog | 0.250 | 0.684 | 0.066 | 0.031 | 0.0009 | 0.0247 | 0.0321 | 0.0084 |
| stall | 0.266 | 0.685 | 0.050 | 0.050 | 0.0010 | 0.0230 | 0.0289 | 0.0083 |
| start | 0.232 | 0.740 | 0.027 | 0.000 | 0.0009 | 0.0269 | 0.0344 | 0.0090 |

| paired vs logged: candidate | success | timeout | goal_area mass | near2.0 mass | death_frame mass |
|---|---|---|---|---|---|
| prog | +0.002 +- 0.019 (0.05) | +0.066 +- 0.030 (0.08) | -0.001 +- 0.003 (0.47) | -0.000 +- 0.003 (0.48) | -0.001 +- 0.000 (0.33) |
| stall | +0.018 +- 0.021 (0.09) | +0.050 +- 0.026 (0.06) | -0.004 +- 0.003 (0.42) | -0.002 +- 0.003 (0.42) | -0.001 +- 0.000 (0.23) |
| start | -0.016 +- 0.007 (0.05) | +0.027 +- 0.017 (0.06) | +0.002 +- 0.004 (0.50) | +0.002 +- 0.004 (0.48) | -0.000 +- 0.000 (0.19) |

| paired vs start: candidate | success | timeout | goal_area mass | near2.0 mass | death_frame mass |
|---|---|---|---|---|---|
| logged | +0.016 +- 0.007 (0.12) | -0.027 +- 0.017 (0.00) | -0.002 +- 0.004 (0.50) | -0.002 +- 0.004 (0.52) | +0.000 +- 0.000 (0.81) |
| prog | +0.018 +- 0.020 (0.16) | +0.039 +- 0.036 (0.08) | -0.002 +- 0.005 (0.47) | -0.002 +- 0.005 (0.50) | -0.001 +- 0.000 (0.58) |
| stall | +0.033 +- 0.022 (0.16) | +0.022 +- 0.032 (0.06) | -0.006 +- 0.004 (0.45) | -0.004 +- 0.004 (0.50) | -0.001 +- 0.000 (0.48) |

## Per-state agreement of the candidate-minus-logged contrast, simulator vs model

| candidate | key | corr | sign agreement |
|---|---|---:|---:|
| prog | success | -0.18 | 0.69 |
| prog | goal_area | 0.12 | 0.44 |
| prog | near2.0 | 0.27 | 0.41 |
| stall | success | 0.07 | 0.56 |
| stall | goal_area | -0.04 | 0.45 |
| stall | near2.0 | -0.11 | 0.50 |
| start | success | 0.21 | 0.78 |
| start | goal_area | -0.06 | 0.38 |
| start | near2.0 | -0.06 | 0.47 |
