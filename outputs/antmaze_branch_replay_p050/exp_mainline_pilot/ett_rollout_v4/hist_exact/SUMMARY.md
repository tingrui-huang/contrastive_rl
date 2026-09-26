# v4 rollouts (motion / stationary on (s, a_q) only; exact counters), C and B, 2026-09-20

`diag_v6_ett_rollout.py roll --v4 --hist-exact --variant C|B` (node3; the v4 one-step models, the advice generator v3 for B).
`manifest.json`, `REPORT.md`, `report.json`; pickles on node3.

| variant | death sim / model | reach | timeout | KS death time | KS death x | KS reach time | AUROC | failing strata |
|---|---|---|---|---:|---:|---:|---:|---|
| C v3 (reference) | 0.294 / 0.303 | 0.615 / 0.637 | 0.091 / 0.059 | 0.027 | 0.064 | 0.035 | 0.995 | pre_zone1 (reach, timeout), between (death x), zone2 (death time, x) |
| **C v4** | 0.294 / 0.304 | 0.615 / 0.649 | 0.091 / 0.047 | 0.029 | 0.056 | 0.037 | 0.995 | pre_zone1 (reach, timeout, reach time), between (reach, timeout), zone2 (death time, x) |
| B v3 | 0.294 / 0.312 | 0.615 / 0.627 | 0.091 / 0.061 | 0.028 | 0.050 | 0.035 | 0.796 | pre_zone1 (reach, timeout), zone2 (death time, x) |
| **B v4** | 0.294 / 0.310 | 0.615 / 0.640 | 0.091 / 0.050 | 0.025 | 0.064 | 0.036 | 0.803 | pre_zone1 (reach, timeout, reach time), between (timeout), zone2 (death time, x) |

Strata (B v4, sim / model): start death 0.724 / 0.718, reach 0.248 / 0.263; pre_zone1 death 0.503 / 0.534, reach 0.319 /
0.402, timeout 0.178 / 0.065; between death 0.283 / 0.308, timeout 0.102 / 0.047; zone2 death 0.105 / 0.110, death-time KS
0.197; post_zone2 / goal_area pass.

Reading.  Removing the advice from the motion / stationary inputs leaves the pooled picture where it was (deaths, timing,
positions within 0.01-0.02 of v3) and does not repair the long-path gaps: the model still under-stalls before the mouth
(pre_zone1 timeouts 0.065 vs 0.178) and in the corridor between the zones (0.047 vs 0.102), and the zone-2 death timing is
unchanged (KS 0.20-0.24).  The v4 gain shows on the far route (held-out far-leg reach 0.72 / 0.94 / 0.72 vs v3 0.68 / 0.86 /
0.74, ett_motion_ab_v4) -- the rollout's 6,000 anchors hold few far-leg anchors, so it barely registers here.  The advice
dependence was real but not what drives the under-stall; that is the motion model's own long-path error.
