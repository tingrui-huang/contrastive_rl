# Short actor-only updates from a20k (5000 updates; the recipe actor loss; the SAME actor batches and keys in every condition; the critic frozen at every update, hash-verified); then rolled from the 128 start states

| critic | left start at 100 | xy disp median at 100 | torque saturation | policy param delta | actor loss / bc nll / q term at the end | first actor batch identical |
|---|---:|---:|---:|---:|---|---|
| plain | 0.40 | 0.71 | 0.46 | 17.81 | 5.584 / -12.56 / 6.538 | True |
| control | 0.47 | 0.87 | 0.54 | 16.09 | 5.512 / -13.86 / 6.531 | True |
| query | 0.80 | 10.49 | 0.18 | 17.17 | 5.556 / -12.97 / 6.531 | True |
| d1_final | 0.38 | 0.58 | 0.55 | 17.46 | 5.546 / -13.70 / 6.559 | True |
| d2_final | 0.80 | 10.42 | 0.18 | 18.20 | 5.625 / -13.41 / 6.627 | True |

## The actor objective as trained on 2048 valid logged rows of the start region (1000 reset rows + start_early rows; relabelled goals; 16 sampled actions per row with common random numbers)

| policy | BC NLL | E Q under plain / loss | E Q under control / loss | E Q under query / loss | E Q under d1_final / loss | E Q under d2_final / loss |
|---|---:|---|---|---|---|---|
| a20k | -11.75 | -6.987 / 6.050 | -6.934 / 6.000 | -7.091 / 6.149 | -6.975 / 6.038 | -7.704 / 6.732 |
| after_plain | -7.23 | -6.522 / 5.834 | -6.710 / 6.013 | -6.862 / 6.158 | -6.514 / 5.827 | -8.199 / 7.427 |
| after_control | -11.61 | -6.945 / 6.017 | -6.842 / 5.919 | -7.060 / 6.127 | -6.926 / 5.999 | -7.649 / 6.686 |
| after_query | -12.62 | -7.025 / 6.043 | -6.983 / 6.003 | -6.982 / 6.002 | -7.018 / 6.036 | -7.668 / 6.654 |
| after_d1_final | -8.27 | -6.579 / 5.836 | -6.726 / 5.976 | -6.892 / 6.134 | -6.554 / 5.813 | -8.110 / 7.291 |
| after_d2_final | -10.05 | -7.369 / 6.498 | -7.224 / 6.360 | -7.288 / 6.421 | -7.400 / 6.528 | -6.924 / 6.075 |
| stall | -8.53 | -6.600 / 5.843 | -6.735 / 5.972 | -6.905 / 6.133 | -6.581 / 5.826 | -8.123 / 7.291 |
| prog | -11.72 | -7.458 / 6.499 | -7.241 / 6.293 | -7.311 / 6.359 | -7.446 / 6.488 | -7.092 / 6.152 |

E Q(stall) - E Q(prog) under plain: all rows +0.859 (stall higher in 0.82); reset rows +1.338 (0.95)

E Q(stall) - E Q(prog) under control: all rows +0.505 (stall higher in 0.82); reset rows +0.737 (0.93)

E Q(stall) - E Q(prog) under query: all rows +0.406 (stall higher in 0.80); reset rows +0.608 (0.93)

E Q(stall) - E Q(prog) under d1_final: all rows +0.865 (stall higher in 0.81); reset rows +1.366 (0.94)

E Q(stall) - E Q(prog) under d2_final: all rows -1.031 (stall higher in 0.21); reset rows -1.826 (0.08)
