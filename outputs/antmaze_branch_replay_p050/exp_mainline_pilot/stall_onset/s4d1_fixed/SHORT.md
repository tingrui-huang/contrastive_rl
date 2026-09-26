# Short actor-only updates from 20k (5000 updates; the recipe actor loss; the SAME actor batches and keys in every condition; the critic frozen at every update, hash-verified); then rolled from the 128 start states

| critic | left start at 100 | xy disp median at 100 | torque saturation | policy param delta | actor loss / bc nll / q term at the end | first actor batch identical |
|---|---:|---:|---:|---:|---|---|
| own20k | 0.49 | 0.95 | 0.48 | 15.11 | 5.333 / -14.62 / 6.383 | True |
| ownfinal | 0.43 | 0.74 | 0.49 | 17.55 | 5.580 / -13.05 / 6.561 | True |
| other_d2_20k | 0.86 | 7.42 | 0.20 | 15.80 | 5.329 / -14.09 / 6.351 | True |
| other_d2_final | 0.83 | 10.36 | 0.16 | 18.19 | 5.621 / -13.54 / 6.630 | True |

## The actor objective as trained on 2048 valid logged rows of the start region (1000 reset rows + start_early rows; relabelled goals; 16 sampled actions per row with common random numbers)

| policy | BC NLL | E Q under own20k / loss | E Q under ownfinal / loss | E Q under other_d2_20k / loss | E Q under other_d2_final / loss |
|---|---:|---|---|---|---|
| 20k | -11.75 | -6.784 / 5.858 | -6.975 / 6.038 | -7.145 / 6.200 | -7.704 / 6.732 |
| after_own20k | -11.46 | -6.728 / 5.818 | -6.917 / 5.997 | -7.144 / 6.214 | -7.732 / 6.772 |
| after_ownfinal | -7.84 | -6.638 / 5.914 | -6.528 / 5.810 | -7.410 / 6.648 | -8.124 / 7.326 |
| after_other_d2_20k | -13.62 | -7.042 / 6.009 | -7.283 / 6.238 | -6.947 / 5.918 | -7.348 / 6.299 |
| after_other_d2_final | -10.22 | -7.284 / 6.409 | -7.417 / 6.536 | -6.875 / 6.020 | -6.924 / 6.067 |
| stall | -8.53 | -6.626 / 5.868 | -6.581 / 5.826 | -7.387 / 6.591 | -8.123 / 7.291 |
| prog | -11.72 | -7.183 / 6.238 | -7.446 / 6.488 | -6.860 / 5.931 | -7.092 / 6.152 |

E Q(stall) - E Q(prog) under own20k: all rows +0.557 (stall higher in 0.83); reset rows +0.837 (0.94)

E Q(stall) - E Q(prog) under ownfinal: all rows +0.865 (stall higher in 0.81); reset rows +1.366 (0.94)

E Q(stall) - E Q(prog) under other_d2_20k: all rows -0.527 (stall higher in 0.23); reset rows -0.912 (0.13)

E Q(stall) - E Q(prog) under other_d2_final: all rows -1.031 (stall higher in 0.21); reset rows -1.826 (0.08)
