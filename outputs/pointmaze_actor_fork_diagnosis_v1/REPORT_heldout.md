# Step 1: where the query-coverage actor sits at the fork

Read-only on the sealed arm-C checkpoints, 16 held-out fork roots, canonical goal (8.5, 3.5). Seed 1 (the 100% seed) has no checkpoint on disk, so the successful seed here is seed 0.

| seed | mode sector (down/right/other) | argmax sector | roots q_down>q_right | roots q_down>q_mode | trapped-by-ridge roots | mean q mode / down / right / grid-max | samples down / right | scale |
|---|---|---|---:|---:|---:|---|---|---:|
| 0 | 0/0/16 | 0/0/16 | 4/16 | 13/16 | 0/16 | -6.365 / -6.348 / -6.249 / -5.998 | 0.02 / 0.05 | 1.531 |
| 2 | 0/0/16 | 0/0/16 | 16/16 | 16/16 | 6/16 | -6.367 / -6.041 / -6.246 / -5.646 | 0.00 / 0.17 | 1.652 |

## What the actions physically do (noise-free kinematics from each root, 8 steps)

| seed | mode outcome (lower/shortcut/other) | q-argmax region | roots where max q in lower region > max q in shortcut region | samples lower / shortcut |
|---|---|---|---:|---|
| 0 | 7/9/0 | 7/9/0 | 7/16 | 0.42 / 0.58 |
| 2 | 5/11/0 | 3/13/0 | 3/16 | 0.26 / 0.74 |

## Pulls at the mode (mean over roots; projection on DOWN (0,-1) and RIGHT (1,0))

| seed | grad_a q . down | grad_a q . right | BC pull . down | BC pull . right |
|---|---:|---:|---:|---:|
| 0 | -1.111 | -0.114 | -0.730 | -0.419 |
| 2 | -1.034 | -0.987 | -0.478 | -0.422 |

## One actual SGD step on an actor copy: movement of tanh(loc) at the 16 roots

critic = only the critic term, bc = only the BC term, full = 0.95 critic + 0.05 bc. "training batches" = 10 x 256 relabeled transitions from the C replay (the real training signal, acting on the roots through shared parameters); "local" = the roots themselves (critic) and the replay actions within 0.15 of them (bc). Values are mean displacement projected on DOWN / RIGHT and the mean change of the critic score at the new mode.

| seed | batch | term | lr | |grad| | d.down | d.right | |d| | dq(mode) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 0 | training_batches | critic | 0.0003 | 2.075 | -0.0001 | +0.0000 | 0.0001 | +0.0001 |
| 0 | training_batches | critic | 0.003 | 2.075 | -0.0014 | +0.0001 | 0.0014 | +0.0014 |
| 0 | training_batches | bc | 0.0003 | 25.734 | +0.0024 | -0.0000 | 0.0024 | -0.0026 |
| 0 | training_batches | bc | 0.003 | 25.734 | +0.0132 | -0.0001 | 0.0132 | -0.0142 |
| 0 | training_batches | full | 0.0003 | 1.343 | +0.0000 | +0.0000 | 0.0000 | -0.0000 |
| 0 | training_batches | full | 0.003 | 1.343 | +0.0000 | +0.0001 | 0.0002 | -0.0000 |
| 0 | local_root_batch | critic | 0.0003 | 8.734 | -0.0052 | +0.0002 | 0.0052 | +0.0055 |
| 0 | local_root_batch | critic | 0.003 | 8.734 | -0.1204 | +0.0014 | 0.1205 | +0.1229 |
| 0 | local_root_batch | bc | 0.0003 | 96.002 | +0.0220 | -0.0019 | 0.0225 | -0.0234 |
| 0 | local_root_batch | bc | 0.003 | 96.002 | +0.0245 | -0.1178 | 0.1252 | -0.0132 |
| 0 | local_root_batch | full | 0.0003 | 4.848 | -0.0003 | +0.0001 | 0.0003 | +0.0003 |
| 0 | local_root_batch | full | 0.003 | 4.848 | -0.0034 | +0.0006 | 0.0036 | +0.0035 |
| 2 | training_batches | critic | 0.0003 | 2.067 | +0.0021 | -0.0000 | 0.0021 | -0.0021 |
| 2 | training_batches | critic | 0.003 | 2.067 | +0.0202 | -0.0000 | 0.0202 | -0.0210 |
| 2 | training_batches | bc | 0.0003 | 83.508 | -0.0758 | +0.0000 | 0.0758 | +0.0763 |
| 2 | training_batches | bc | 0.003 | 83.508 | -1.0263 | +0.0001 | 1.0263 | -0.2342 |
| 2 | training_batches | full | 0.0003 | 2.682 | -0.0020 | +0.0000 | 0.0020 | +0.0020 |
| 2 | training_batches | full | 0.003 | 2.682 | -0.0194 | +0.0000 | 0.0194 | +0.0199 |
| 2 | local_root_batch | critic | 0.0003 | 8.032 | -0.0228 | -0.0000 | 0.0228 | +0.0234 |
| 2 | local_root_batch | critic | 0.003 | 8.032 | -0.1862 | -0.0000 | 0.1862 | +0.1715 |
| 2 | local_root_batch | bc | 0.0003 | 156.961 | +0.2470 | -0.0001 | 0.2470 | -0.2599 |
| 2 | local_root_batch | bc | 0.003 | 156.961 | +0.2760 | -0.0147 | 0.2764 | -0.2765 |
| 2 | local_root_batch | full | 0.0003 | 2.881 | +0.0122 | -0.0000 | 0.0122 | -0.0127 |
| 2 | local_root_batch | full | 0.003 | 2.881 | +0.1019 | -0.0001 | 0.1019 | -0.1071 |
