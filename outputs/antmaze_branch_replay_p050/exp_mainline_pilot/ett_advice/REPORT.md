# The advice process: temporal coherence and support of the memoryless nominal; the onset head vs the hold history

Sealed 2026-09-20 03:33:21; `manifest.json`, `report.json`.  No rollouts, no training.

## A / B: hold structure of the real advice vs the memoryless nominal drawn i.i.d. along the same real states

| source | rows | hold share | in band | out of band | share of holds in band | P(hold|hold) | P(hold|no hold) | runs | mean | median | p90 | len 1 | len >= 5 | len >= 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real advice, all 53,747 branches, valid rows (teacher-forced along each branch) | 7648612 | 0.042 | 0.088 | 0.036 | 0.216 | 0.981 | 0.002 | 21509 | 14.828 | 14.000 | 22.000 | 0.013 | 0.948 | 0.131 |
| real advice, sampled branches | 868101 | 0.041 | 0.086 | 0.036 | 0.216 | 0.982 | 0.002 | 2383 | 14.949 | 14.000 | 22.000 | 0.014 | 0.948 | 0.131 |
| memoryless nominal, one i.i.d. draw per row along the real states | 868101 | 0.265 | 0.000 | 0.296 | 0.000 | 0.959 | 0.014 | 9790 | 23.530 | 12.000 | 59.000 | 0.111 | 0.723 | 0.354 |

Real advice, all branches: onset rows 15612, of which on hold rows 15600; onset rate per hold row 0.0419 (in band 0.1638).  Class share {"diagonal": 0.0058, "invalid": 0.007, "off_hold": 0.041, "off_other": 0.9457, "off_release": 0.0006}.

Hold share by region (real, all branches): {"start": 0.0, "pre_zone1": 0.053, "zone1": 0.113, "between": 0.094, "zone2": 0.072, "post_zone2": 0.012, "goal_area": 0.0, "west_column": 0.0, "top_corridor": 0.0, "east_column": 0.0}

Hold share by region (nominal i.i.d., sample): {"start": 0.0, "pre_zone1": 0.716, "zone1": 0.0, "between": 0.379, "zone2": 0.001, "post_zone2": 0.0, "goal_area": 0.001, "west_column": 0.0, "top_corridor": 0.0, "east_column": 0.112}

## B: support of the nominal on the intervened states

Nominal exact hold probability: mean 0.265 (in band 0.000); on real hold rows 0.209 vs real no-hold rows 0.268.

| AUROC of p_h(s) vs the real hold | all | in_band | out_of_band | j_0 | j_1_10 | j_11_50 | j_gt_50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| | 0.544 | 0.494 | 0.562 | 0.659 | 0.635 | 0.596 | 0.453 |

Logged reference (the MDN's 13300 validation rows): NLL of the logged action -11.38; hold share 0.132; AUROC p_h vs logged hold 0.981.

| NLL of the REAL advice under the nominal | all | real_hold_rows | real_nohold_rows | j_0 | j_1_10 | j_11_50 | j_gt_50 | class diagonal | class off_hold | class off_other | class off_release |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | 17.71 | -3.52 | 18.61 | -8.55 | -7.69 | -5.85 | 27.70 | -10.51 | -3.26 | 18.78 | -2.91 |

Counter at hold rows (run position, 1 = first hold of a run), share of hold rows: real vs nominal

| run position | 1 | 2 | 3-4 | 5-9 | 10-19 | 20-49 | >= 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| real | 0.067 | 0.066 | 0.129 | 0.305 | 0.322 | 0.107 | 0.004 |
| nominal | 0.042 | 0.038 | 0.068 | 0.137 | 0.191 | 0.284 | 0.239 |

## C: the v3 onset head vs the hold counter (held-out real in-band hold rows; fixed models)

| fold | rows | realised onset rate | p_onset (real counter) | AUROC | k = 0 | k = 1 | k = 2 | k = 3 | k = 5 | k = 10 | k = 20 | k = 50 | k = 100 | p_onset (nominal counter draw) | kh real mean / median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fold0 | 22561 | 0.1991 | 0.1779 | 0.753 | 0.0288 | 0.0355 | 0.0433 | 0.0527 | 0.0765 | 0.1628 | 0.2777 | 0.2024 | 0.0767 | 0.1667 | 11.1 / 12 |
| fold1 | 24441 | 0.1947 | 0.2167 | 0.750 | 0.0369 | 0.0449 | 0.0544 | 0.0656 | 0.0940 | 0.1972 | 0.3468 | 0.3080 | 0.2008 | 0.2336 | 11.1 / 12 |
| fold2 | 21961 | 0.2016 | 0.2032 | 0.740 | 0.0360 | 0.0437 | 0.0528 | 0.0635 | 0.0906 | 0.1843 | 0.3174 | 0.2733 | 0.1452 | 0.2083 | 11.3 / 12 |

Realised onset rate / mean p_onset (rows) by real counter bin:

| fold | kh_0_0 | kh_1_1 | kh_2_3 | kh_4_7 | kh_8_15 | kh_16_31 | kh_32_63 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fold0 | 0.0000 / 0.0001 (776) | 0.0000 / 0.0006 (757) | 0.0007 / 0.0087 (1488) | 0.0908 / 0.1177 (2610) | 0.2217 / 0.2025 (13819) | 0.3876 / 0.2899 (3003) | 0.2500 / 0.2127 (108) |
| fold1 | 0.0000 / 0.0003 (804) | 0.0000 / 0.0015 (828) | 0.0000 / 0.0139 (1646) | 0.0819 / 0.1419 (2967) | 0.2192 / 0.2480 (14779) | 0.3777 / 0.3479 (3278) | 0.2734 / 0.3330 (139) |
| fold2 | 0.0000 / 0.0003 (706) | 0.0000 / 0.0010 (692) | 0.0000 / 0.0123 (1366) | 0.0887 / 0.1489 (2458) | 0.2234 / 0.2247 (13615) | 0.3794 / 0.3284 (2984) | 0.2643 / 0.2756 (140) |
