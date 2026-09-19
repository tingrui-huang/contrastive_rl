# One-step ETT preflight (held-out fold rows, cross-fitted)

`check.json`; gates from `manifest.json`.  **ALL GATES PASSED**

| gate | threshold | pooled value | pass |
|---|---|---|---|
| G1_xy_median | 0.01 | see pooled | yes |
| G1_xy_p99 | 0.1 | see pooled | yes |
| G1_delta_rmse | 0.3 | see pooled | yes |
| G2_xy50 | 1.0 | see pooled | yes |
| G2_xy10 | 0.2 | see pooled | yes |
| G3_auroc | 0.85 | see pooled | yes |
| G4_ratio | 5.0 | see pooled | yes |
| G5_calibration | (0.5, 2.0) | see pooled | yes |
| G6_advice |  | see pooled | yes |
| param_delta |  | see pooled | yes |

## Pooled

```
{
 "xy_err_median": 0.0016130420845001936,
 "xy_err_p99": 0.027449607849121094,
 "delta_rmse": 0.1565990447998047,
 "xy10_median": 0.05951214209198952,
 "xy50_median": 0.46919047832489014,
 "onset_auroc": 0.9959830045700073,
 "ratio_onsets_vs_non": 139.0336456298828,
 "calibration_mean_over_prevalence": 1.0501485579933925,
 "advice_ratio_in_band": 13232.310546875,
 "n_band_off_hold": 20234,
 "n_band_drive": 84261
}
```

## Per fold

| fold | held-out rows | xy err median / p90 / p99 | delta RMSE | xy err diag / off rows | 10-step / 50-step xy (median) | onset AUROC | P(onset) on onsets / non | prevalence | best step |
|---|---:|---|---:|---|---|---:|---|---:|---:|
| 0 | 2553073 | 0.0016 / 0.0066 / 0.0279 | 0.161 | 0.0026 / 0.0016 | 0.056 / 0.411 | 0.996 | 0.2313 / 0.00187 | 0.00202 | 39000 |
| 1 | 2565460 | 0.0016 / 0.0063 / 0.0269 | 0.151 | 0.0024 / 0.0016 | 0.060 / 0.498 | 0.996 | 0.2509 / 0.00185 | 0.00210 | 40000 |
| 2 | 2530079 | 0.0016 / 0.0066 / 0.0276 | 0.158 | 0.0025 / 0.0016 | 0.062 / 0.516 | 0.996 | 0.2164 / 0.00131 | 0.00200 | 40000 |
