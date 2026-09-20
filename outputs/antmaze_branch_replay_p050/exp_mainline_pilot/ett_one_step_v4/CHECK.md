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
 "xy_err_median_moving": 0.0022009157110005617,
 "xy_err_p99_moving": 0.03196408599615097,
 "delta_rmse_moving": 0.1609470546245575,
 "xy10_median": 0.054157331585884094,
 "xy50_median": 0.45691433548927307,
 "onset_auroc": 0.9969075322151184,
 "ratio_onsets_vs_non": 171.67674255371094,
 "calibration_mean_over_prevalence": 1.1014972590507273,
 "advice_ratio_in_band": 622.2979736328125,
 "stat_gate_auroc": 0.99771648645401,
 "stat_gate_accuracy": 0.9836444444444444,
 "atom_xy_err_median_stationary": 5.298300607137207e-07
}
```

## Per fold

| fold | held-out rows | stationary share | xy err moving median / p90 / p99 | delta RMSE moving | atom xy err stationary median / p99 | stat gate AUROC / acc / fires | 10 / 50-step xy | onset AUROC | P on onsets / non | best step |
|---|---:|---:|---|---:|---|---|---|---:|---|---:|
| 0 | 2553073 | 0.272 | 0.0021 / 0.0077 / 0.0326 | 0.166 | 0.00000 / 0.0021 | 0.9976 / 0.9854 / 0.273 | 0.053 / 0.404 | 0.997 | 0.2607 / 0.00144 | 39000 |
| 1 | 2565460 | 0.268 | 0.0023 / 0.0074 / 0.0315 | 0.156 | 0.00000 / 0.0020 | 0.9976 / 0.9804 / 0.259 | 0.055 / 0.477 | 0.997 | 0.3082 / 0.00194 | 40000 |
| 2 | 2530079 | 0.261 | 0.0022 / 0.0077 / 0.0319 | 0.161 | 0.00000 / 0.0007 | 0.9979 / 0.9851 / 0.260 | 0.055 / 0.494 | 0.997 | 0.2951 / 0.00166 | 40000 |
