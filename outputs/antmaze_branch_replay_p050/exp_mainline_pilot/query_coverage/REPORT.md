# Query coverage, stage 1: logged-query vs extended-query futures at the start / early-decision anchors (generation only)

`manifest.json` (sealed before generation; gates pre-registered), `report.json`.  States = every pilot anchor in the start region (4276 anchors, weight 0.079, t median 10); queries q0 logged torque / q1 lineage-agent mode / q2..q5 four pinned samples; the same lineage agent continues every query (mode, closed-loop); two paired hazard draws per anchor; nothing filtered.  Anchor-weighted shares; "extended" spreads each anchor's weight equally over its six queries; deltas are paired by anchor (bootstrap 95 % CI).  Entered far = y >= 6 at x < 2 after the query; completed far = entered and reached the goal; masses = the critic goal marginal (truncated geometric, gamma 0.999).

**Gate for stage 2** (G1, G2 and G3 in every lineage): **NOT PASSED** (1 / 3 lineages reported).

## seed_0 (n = 4276 anchors; policy scale median 0.137; restore max diff 0.0e+00)

G1 completed-far gain (extended - logged) +0.0154 CI [+0.0116, +0.0194] -> FAIL; G2 goal mass extended / logged 0.017 / 0.016 -> pass; G3 distinct anchors with an added far-complete future draw 0 / 1: 377 / 378 -> pass.

| query set | entered far | completed far | entered zone | success | death | timeout | length | mass far | mass goal | mass zone |
|---|---|---|---|---|---|---|---|---|---|---|
| logged | 0.041 | 0.026 | 0.920 | 0.180 | 0.692 | 0.128 | 209 | 0.041 | 0.016 | 0.155 |
| mode | 0.070 | 0.046 | 0.889 | 0.195 | 0.667 | 0.138 | 224 | 0.067 | 0.016 | 0.147 |
| samples | 0.066 | 0.044 | 0.891 | 0.195 | 0.670 | 0.135 | 221 | 0.063 | 0.017 | 0.147 |
| added | 0.067 | 0.045 | 0.890 | 0.195 | 0.669 | 0.136 | 222 | 0.064 | 0.017 | 0.147 |
| extended | 0.062 | 0.041 | 0.895 | 0.193 | 0.673 | 0.134 | 220 | 0.060 | 0.017 | 0.149 |

| single query | entered far | completed far | success | death | timeout | mass goal |
|---|---|---|---|---|---|---|
| logged | 0.041 | 0.026 | 0.180 | 0.692 | 0.128 | 0.016 |
| mode | 0.070 | 0.046 | 0.195 | 0.667 | 0.138 | 0.016 |
| sample0 | 0.067 | 0.048 | 0.199 | 0.668 | 0.132 | 0.017 |
| sample1 | 0.065 | 0.041 | 0.193 | 0.671 | 0.136 | 0.017 |
| sample2 | 0.066 | 0.045 | 0.196 | 0.670 | 0.134 | 0.017 |
| sample3 | 0.066 | 0.043 | 0.193 | 0.670 | 0.137 | 0.017 |

| paired delta vs logged | entered far | completed far | success | death | timeout | mass goal | mass far |
|---|---|---|---|---|---|---|---|
| extended_minus_logged | +0.0216 [+0.0173, +0.0257] | +0.0154 [+0.0116, +0.0194] | +0.0127 [+0.0078, +0.0176] | -0.0188 [-0.0231, -0.0146] | +0.0061 [+0.0007, +0.0114] | +0.0004 [-0.0004, +0.0012] | +0.0189 [+0.0149, +0.0227] |
| added_minus_logged | +0.0259 [+0.0203, +0.0314] | +0.0185 [+0.0141, +0.0233] | +0.0153 [+0.0094, +0.0215] | -0.0226 [-0.0281, -0.0176] | +0.0073 [+0.0011, +0.0136] | +0.0005 [-0.0005, +0.0014] | +0.0227 [+0.0181, +0.0272] |
| mode_minus_logged | +0.0290 [+0.0226, +0.0363] | +0.0195 [+0.0136, +0.0256] | +0.0145 [+0.0070, +0.0222] | -0.0248 [-0.0309, -0.0188] | +0.0103 [+0.0020, +0.0191] | +0.0001 [-0.0011, +0.0013] | +0.0257 [+0.0200, +0.0312] |
| samples_minus_logged | +0.0251 [+0.0200, +0.0304] | +0.0183 [+0.0135, +0.0229] | +0.0155 [+0.0095, +0.0216] | -0.0220 [-0.0272, -0.0170] | +0.0066 [+0.0001, +0.0132] | +0.0006 [-0.0004, +0.0015] | +0.0219 [+0.0174, +0.0266] |

| stratum | n | weight | completed far logged / mode / samples / added / extended | extended - logged (CI) | success logged / extended | mass goal logged / extended |
|---|---|---|---|---|---|---|
| all start anchors | 4276 | 0.0794 | 0.026 / 0.046 / 0.044 / 0.045 / 0.041 | +0.0154 [+0.0118, +0.0193] | 0.180 / 0.193 | 0.016 / 0.017 |
| reset rows (t = 0) | 196 | 0.0038 | 0.037 / 0.423 / 0.379 / 0.388 / 0.330 | +0.2922 [+0.2536, +0.3291] | 0.198 / 0.425 | 0.016 / 0.026 |
| t in [1, 30) | 4037 | 0.0749 | 0.020 / 0.021 / 0.023 / 0.023 / 0.022 | +0.0018 [-0.0008, +0.0044] | 0.175 / 0.177 | 0.016 / 0.016 |
| logged shortcut episodes | 4071 | 0.0758 | 0.002 / 0.022 / 0.022 / 0.022 / 0.018 | +0.0169 [+0.0140, +0.0201] | 0.162 / 0.175 | 0.015 / 0.016 |
| logged detour episodes | 205 | 0.0037 | 0.529 / 0.523 / 0.510 / 0.513 / 0.515 | -0.0140 [-0.0693, +0.0421] | 0.557 / 0.553 | 0.043 / 0.040 |

Reliability: far-complete rate per (anchor, query) draw 0 / draw 1 = 0.0419 / 0.0423; agreement of the far-complete indicator across draws 0.995; Jaccard of the far-complete sets 0.882; distinct anchors with a logged far-complete future draw 0 / 1: 120 / 118.

