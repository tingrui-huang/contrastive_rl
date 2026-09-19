# One-step ETT v2 preflight (held-out fold rows, cross-fitted)

`check.json`; gates in `manifest.json`.  **ALL GATES PASSED**

| gate | pass |
|---|---|
| G1_xy_median | yes |
| G1_xy_p99 | yes |
| G1_delta_rmse | yes |
| G2_xy50 | yes |
| G2_xy10 | yes |
| G3_auroc | yes |
| G4_ratio | yes |
| G5_calibration | yes |
| G6_advice | yes |
| G7_stat_auroc | yes |
| G7_stat_accuracy | yes |
| G7_atom_xy | yes |
| param_delta | yes |

## Pooled

```
{
 "xy_err_median_moving": 0.0021919889841228724,
 "xy_err_p99_moving": 0.03250768408179283,
 "delta_rmse_moving": 0.16087006032466888,
 "xy10_median": 0.05594766139984131,
 "xy50_median": 0.4839707612991333,
 "onset_auroc": 0.9969378113746643,
 "ratio_onsets_vs_non": 170.30023193359375,
 "calibration_mean_over_prevalence": 1.1094026184566168,
 "advice_ratio_in_band": 647.8822631835938,
 "stat_gate_auroc": 0.9973297119140625,
 "stat_gate_accuracy": 0.9810177777777778,
 "atom_xy_err_median_stationary": 5.092622927804769e-07
}
```

## Per fold

| fold | held-out rows | stationary share | xy err moving median / p90 / p99 | delta RMSE moving | atom xy err stationary median / p99 | stat gate AUROC / acc / fires | 10 / 50-step xy | onset AUROC | P on onsets / non | best step |
|---|---:|---:|---|---:|---|---|---|---:|---|---:|
| 0 | 2553073 | 0.272 | 0.0022 / 0.0077 / 0.0332 | 0.166 | 0.00000 / 0.0012 | 0.9971 / 0.9816 / 0.272 | 0.054 / 0.425 | 0.997 | 0.2651 / 0.00148 | 39000 |
| 1 | 2565460 | 0.268 | 0.0022 / 0.0075 / 0.0324 | 0.157 | 0.00000 / 0.0018 | 0.9971 / 0.9799 / 0.264 | 0.058 / 0.537 | 0.997 | 0.3108 / 0.00193 | 40000 |
| 2 | 2530079 | 0.261 | 0.0022 / 0.0078 / 0.0322 | 0.160 | 0.00000 / 0.0007 | 0.9979 / 0.9815 / 0.254 | 0.056 / 0.487 | 0.997 | 0.2905 / 0.00168 | 40000 |
