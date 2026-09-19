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
 "xy_err_median_moving": 0.002769267652183771,
 "xy_err_p99_moving": 0.03397765010595322,
 "delta_rmse_moving": 0.17145398259162903,
 "xy10_median": 0.06495242565870285,
 "xy50_median": 0.4472200870513916,
 "onset_auroc": 0.9964258670806885,
 "ratio_onsets_vs_non": 167.22555541992188,
 "calibration_mean_over_prevalence": 0.911298316359995,
 "advice_ratio_in_band": 633.6935424804688,
 "stat_gate_auroc": 0.9951786994934082,
 "stat_gate_accuracy": 0.9702755555555556,
 "atom_xy_err_median_stationary": 5.364418029785156e-07
}
```

## Per fold

| fold | held-out rows | stationary share | xy err moving median / p90 / p99 | delta RMSE moving | atom xy err stationary median / p99 | stat gate AUROC / acc / fires | 10 / 50-step xy | onset AUROC | P on onsets / non | best step |
|---|---:|---:|---|---:|---|---|---|---:|---|---:|
| 0 | 2553073 | 0.272 | 0.0026 / 0.0081 / 0.0326 | 0.159 | 0.00000 / 0.0022 | 0.9968 / 0.9808 / 0.276 | 0.063 / 0.407 | 0.997 | 0.2812 / 0.00157 | 37000 |
| 1 | 2565460 | 0.268 | 0.0033 / 0.0099 / 0.0370 | 0.196 | 0.00000 / 0.0040 | 0.9884 / 0.9467 / 0.261 | 0.074 / 0.548 | 0.995 | 0.1573 / 0.00118 | 4000 |
| 2 | 2530079 | 0.261 | 0.0025 / 0.0079 / 0.0321 | 0.156 | 0.00000 / 0.0021 | 0.9978 / 0.9833 / 0.260 | 0.057 / 0.398 | 0.997 | 0.2691 / 0.00146 | 39000 |
