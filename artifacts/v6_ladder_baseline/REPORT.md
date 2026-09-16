# V6 ladder baseline: vanilla CRL across teacher-detour rungs

Mean over seeds; n natural-draw episodes per policy at one fixed reset seed; mean = deterministic tanh(loc), sample = tanh-normal samples with a fixed action seed.

| rung | seeds | policy | success | failure | timeout | shortcut | detour |
|---|---|---|---:|---:|---:|---:|---:|
| 0.05 | 3 | mean | 0.370 | 0.630 | 0.000 | 0.966 | 0.000 |
| 0.05 | 3 | sample | 0.370 | 0.629 | 0.001 | 0.979 | 0.001 |
| 0.3 | 3 | mean | 0.370 | 0.630 | 0.000 | 0.976 | 0.000 |
| 0.3 | 3 | sample | 0.373 | 0.622 | 0.004 | 0.974 | 0.006 |

Per seed:

| rung | seed | policy | success | failure | timeout | shortcut | detour |
|---|---|---|---:|---:|---:|---:|---:|
| 0.05 | 0 | mean | 0.370 | 0.630 | 0.000 | 0.973 | 0.000 |
| 0.05 | 0 | sample | 0.373 | 0.627 | 0.000 | 0.977 | 0.003 |
| 0.05 | 1 | mean | 0.370 | 0.630 | 0.000 | 0.960 | 0.000 |
| 0.05 | 1 | sample | 0.370 | 0.630 | 0.000 | 0.977 | 0.000 |
| 0.05 | 2 | mean | 0.370 | 0.630 | 0.000 | 0.963 | 0.000 |
| 0.05 | 2 | sample | 0.367 | 0.630 | 0.003 | 0.983 | 0.000 |
| 0.3 | 0 | mean | 0.370 | 0.630 | 0.000 | 0.963 | 0.000 |
| 0.3 | 0 | sample | 0.373 | 0.617 | 0.010 | 0.973 | 0.017 |
| 0.3 | 1 | mean | 0.370 | 0.630 | 0.000 | 0.977 | 0.000 |
| 0.3 | 1 | sample | 0.370 | 0.630 | 0.000 | 0.977 | 0.000 |
| 0.3 | 2 | mean | 0.370 | 0.630 | 0.000 | 0.987 | 0.000 |
| 0.3 | 2 | sample | 0.377 | 0.620 | 0.003 | 0.973 | 0.000 |
