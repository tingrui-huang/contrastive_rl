# Balanced BC rows: attribution on the C critics and joint training

Balanced BC rows = `GroupBalancedBCSampler` (D replay, (state cell, goal cell) groups, 8 sectors + wait, share cap 0.25); Acme log-prob, random_goals 0, bc 0.05 everywhere. New = 300 paired native episodes on reset seeds from 31.5M (never used before); sealed = the 200-episode protocol of Steps 4-8.

## Attribution: balanced BC rows on the C critics (fixed critic, 300k actor updates)

| critic | actor seed | new mode reach | new mode lower | new sample reach | new sample lower | sealed mode reach | sealed mode lower | sealed sample lower |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C seed 2 | 0 | 0.363 | 0.000 | 0.610 | 0.423 | 0.305 | 0.000 | 0.445 |
| C seed 2 | 1 | 0.363 | 0.000 | 0.600 | 0.403 | 0.305 | 0.000 | 0.405 |
| C seed 2 | 2 | 0.363 | 0.000 | 0.597 | 0.403 | 0.305 | 0.000 | 0.435 |
| C seed 0 | 0 | 0.503 | 0.223 | 0.637 | 0.450 | 0.480 | 0.250 | 0.460 |
| C seed 0 | 1 | 0.363 | 0.000 | 0.580 | 0.357 | 0.305 | 0.000 | 0.350 |
| C seed 0 | 2 | 0.523 | 0.263 | 0.613 | 0.423 | 0.510 | 0.300 | 0.465 |
| D1 (Step 8) | 0 | | | | | 1.000 | 1.000 | 0.885 |
| D1 (Step 8) | 1 | | | | | 1.000 | 1.000 | 0.900 |
| D1 (Step 8) | 2 | | | | | 1.000 | 1.000 | 0.875 |

## Joint training, 30,000 updates (the sealed D-arm budget)

| arm | seed | new mode reach | new mode lower | new sample reach | new sample lower | sealed mode reach | sealed mode lower | critic d-r | roots down | argmax -> shortcut / lower |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| shared | 0 | 0.363 | 0.000 | 0.443 | 0.163 | 0.305 | 0.000 | +0.204 | 13/16 | 3 / 13 |
| shared | 1 | 0.363 | 0.000 | 0.400 | 0.093 | 0.305 | 0.000 | +0.247 | 16/16 | 2 / 14 |
| shared | 2 | 1.000 | 1.000 | 0.840 | 0.743 | 1.000 | 1.000 | +0.173 | 16/16 | 0 / 16 |
| shared | mean | 0.576 | 0.333 | 0.561 | 0.333 | 0.537 | 0.333 | | | |
| balanced | 0 | 0.363 | 0.000 | 0.510 | 0.270 | 0.305 | 0.000 | +0.277 | 16/16 | 0 / 16 |
| balanced | 1 | 0.363 | 0.000 | 0.443 | 0.140 | 0.305 | 0.000 | +0.247 | 16/16 | 2 / 14 |
| balanced | 2 | 0.620 | 0.377 | 0.737 | 0.597 | 0.580 | 0.415 | +0.171 | 16/16 | 0 / 16 |
| balanced | mean | 0.449 | 0.126 | 0.563 | 0.336 | 0.397 | 0.138 | | | |

## Joint training, 300,000 updates (10x the sealed budget)

| arm | seed | new mode reach | new mode lower | new sample reach | new sample lower | sealed mode reach | sealed mode lower | critic d-r | roots down | argmax -> shortcut / lower |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| shared | 0 | 0.363 | 0.000 | 0.440 | 0.153 | 0.305 | 0.000 | +0.892 | 16/16 | 0 / 16 |
| shared | 1 | 0.363 | 0.000 | 0.353 | 0.027 | 0.305 | 0.000 | +1.071 | 16/16 | 1 / 15 |
| shared | 2 | 0.363 | 0.000 | 0.407 | 0.100 | 0.305 | 0.000 | +0.643 | 15/16 | 3 / 13 |
| shared | mean | 0.363 | 0.000 | 0.400 | 0.093 | 0.305 | 0.000 | | | |
| balanced | 0 | 0.363 | 0.000 | 0.353 | 0.050 | 0.305 | 0.000 | +0.697 | 16/16 | 0 / 16 |
| balanced | 1 | 0.363 | 0.000 | 0.383 | 0.057 | 0.305 | 0.000 | +0.755 | 16/16 | 0 / 16 |
| balanced | 2 | 0.363 | 0.000 | 0.377 | 0.067 | 0.305 | 0.000 | +1.064 | 16/16 | 0 / 16 |
| balanced | mean | 0.363 | 0.000 | 0.371 | 0.058 | 0.305 | 0.000 | | | |
