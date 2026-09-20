# The advice generator: held-out checks (E1 true context / E2a true context autoregressive / E2b prior context autoregressive)
Generator version 3; hold decision rule: sample.

Sealed 2026-09-20 09:56:37.  Gates: {"E1_in_band_auroc_true_context": 0.9, "E1_ece": 0.05, "E1_region_hold_share_abs_gap": 0.02, "E2a_p_hold_given_hold_abs_gap": 0.05, "E2a_run_median_rel_gap": 0.3, "E2a_death_rate_vs_real_advice_abs_gap": 0.03, "E2a_death_auroc": 0.9, "E2a_death_time_ks": 0.15, "E2b_never_hold_regions_max_share": 0.01}.

| fold | rows | E1 AUROC in band | E1 ECE | E1 torque NLL (non-hold) | E2a P(death) generated / real advice / realised | E2a death AUROC gen / real advice | E2a death-time KS | E2a hold-row agreement | E2a P(h|h) gen / real | E2a run median gen / real | E2b in-band hold gen / real | E2b P(death) | gates |
|---|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|---|
| 0 | 600000 | 1.000 | 0.0007 | -13.14 | 0.167 / 0.167 / 0.292 | 0.976 / 0.991 | 0.049 | 0.995 | 0.974 / 0.982 | 12 / 14 | 0.269 / 0.085 | 0.207 | PASS |
| 1 | 600000 | 1.000 | 0.0010 | -10.99 | 0.190 / 0.194 / 0.296 | 0.966 / 0.988 | 0.057 | 0.993 | 0.968 / 0.981 | 12 / 14 | 0.266 / 0.085 | 0.234 | PASS |
| 2 | 600000 | 0.999 | 0.0006 | -12.97 | 0.184 / 0.186 / 0.293 | 0.976 / 0.991 | 0.081 | 0.995 | 0.971 / 0.982 | 12 / 14 | 0.263 / 0.088 | 0.220 | PASS |

## Hold share by region (E1 pred / real; E2b gen / real)

| fold | pre_zone1 | zone1 | between | zone2 |
|---|---|---|---|---|
| 0 | 0.052 / 0.052; 0.048 / 0.047 | 0.112 / 0.114; 0.332 / 0.104 | 0.087 / 0.088; 0.108 / 0.084 | 0.071 / 0.072; 0.229 / 0.073 |
| 1 | 0.053 / 0.054; 0.058 / 0.059 | 0.119 / 0.122; 0.322 / 0.117 | 0.097 / 0.099; 0.116 / 0.091 | 0.074 / 0.075; 0.232 / 0.066 |
| 2 | 0.052 / 0.053; 0.052 / 0.051 | 0.102 / 0.104; 0.321 / 0.117 | 0.096 / 0.098; 0.124 / 0.095 | 0.070 / 0.071; 0.226 / 0.071 |

All gates in every fold: **True** -> the full rollout (variant B) runs.
