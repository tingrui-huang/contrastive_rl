# Query coverage, stage 1: logged-query vs extended-query futures at the start / early-decision anchors (generation only)

`manifest.json` (sealed before generation; gates pre-registered), `report.json`.  States = every pilot anchor in the start region (4276 anchors, weight 0.079, t median 10); queries q0 logged torque / q1 lineage-agent mode / q2..q5 four pinned samples; the same lineage agent continues every query (mode, closed-loop); two paired hazard draws per anchor; nothing filtered.  Anchor-weighted shares; "extended" spreads each anchor's weight equally over its six queries; deltas are paired by anchor (bootstrap 95 % CI).  Entered far = y >= 6 at x < 2 after the query; completed far = entered and reached the goal; masses = the critic goal marginal (truncated geometric, gamma 0.999).

**Gate for stage 2** (G1, G2 and G3 in every lineage): **NOT PASSED** (3 / 3 lineages reported).

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

| stratum | draw | mode not far-complete | of which another query far-complete | of which a sample far-complete | mode not success | of which another query success |
|---|---|---|---|---|---|---|
| reset rows | 0 | 110 | 70 | 70 | 92 | 62 |
| reset rows | 1 | 113 | 74 | 73 | 99 | 71 |
| all start anchors | 0 | 4079 | 186 | 180 | 3483 | 440 |
| all start anchors | 1 | 4078 | 188 | 180 | 3392 | 450 |

Reliability: far-complete rate per (anchor, query) draw 0 / draw 1 = 0.0419 / 0.0423; agreement of the far-complete indicator across draws 0.995; Jaccard of the far-complete sets 0.882; distinct anchors with a logged far-complete future draw 0 / 1: 120 / 118.

## seed_1 (n = 4276 anchors; policy scale median 0.121; restore max diff 0.0e+00)

G1 completed-far gain (extended - logged) +0.0164 CI [+0.0122, +0.0205] -> FAIL; G2 goal mass extended / logged 0.017 / 0.017 -> pass; G3 distinct anchors with an added far-complete future draw 0 / 1: 389 / 389 -> pass.

| query set | entered far | completed far | entered zone | success | death | timeout | length | mass far | mass goal | mass zone |
|---|---|---|---|---|---|---|---|---|---|---|
| logged | 0.047 | 0.035 | 0.899 | 0.185 | 0.653 | 0.162 | 231 | 0.043 | 0.017 | 0.161 |
| mode | 0.073 | 0.057 | 0.873 | 0.208 | 0.638 | 0.155 | 236 | 0.066 | 0.017 | 0.156 |
| samples | 0.069 | 0.054 | 0.878 | 0.200 | 0.638 | 0.162 | 239 | 0.062 | 0.017 | 0.156 |
| added | 0.070 | 0.054 | 0.877 | 0.202 | 0.638 | 0.160 | 238 | 0.063 | 0.017 | 0.156 |
| extended | 0.066 | 0.051 | 0.880 | 0.199 | 0.640 | 0.161 | 237 | 0.059 | 0.017 | 0.157 |

| single query | entered far | completed far | success | death | timeout | mass goal |
|---|---|---|---|---|---|---|
| logged | 0.047 | 0.035 | 0.185 | 0.653 | 0.162 | 0.017 |
| mode | 0.073 | 0.057 | 0.208 | 0.638 | 0.155 | 0.017 |
| sample0 | 0.066 | 0.053 | 0.201 | 0.638 | 0.161 | 0.017 |
| sample1 | 0.070 | 0.054 | 0.202 | 0.640 | 0.158 | 0.017 |
| sample2 | 0.068 | 0.054 | 0.200 | 0.638 | 0.162 | 0.017 |
| sample3 | 0.071 | 0.054 | 0.198 | 0.636 | 0.166 | 0.017 |

| paired delta vs logged | entered far | completed far | success | death | timeout | mass goal | mass far |
|---|---|---|---|---|---|---|---|
| extended_minus_logged | +0.0187 [+0.0145, +0.0233] | +0.0164 [+0.0122, +0.0205] | +0.0137 [+0.0087, +0.0186] | -0.0124 [-0.0168, -0.0081] | -0.0013 [-0.0070, +0.0042] | +0.0002 [-0.0004, +0.0009] | +0.0167 [+0.0127, +0.0206] |
| added_minus_logged | +0.0225 [+0.0172, +0.0280] | +0.0196 [+0.0148, +0.0251] | +0.0164 [+0.0107, +0.0224] | -0.0149 [-0.0203, -0.0093] | -0.0015 [-0.0081, +0.0049] | +0.0003 [-0.0005, +0.0010] | +0.0200 [+0.0153, +0.0247] |
| mode_minus_logged | +0.0257 [+0.0191, +0.0327] | +0.0227 [+0.0163, +0.0293] | +0.0223 [+0.0145, +0.0299] | -0.0153 [-0.0217, -0.0088] | -0.0070 [-0.0161, +0.0014] | +0.0007 [-0.0003, +0.0017] | +0.0232 [+0.0176, +0.0289] |
| samples_minus_logged | +0.0217 [+0.0163, +0.0272] | +0.0189 [+0.0139, +0.0238] | +0.0149 [+0.0091, +0.0208] | -0.0148 [-0.0199, -0.0097] | -0.0001 [-0.0067, +0.0067] | +0.0002 [-0.0006, +0.0010] | +0.0192 [+0.0143, +0.0240] |

| stratum | n | weight | completed far logged / mode / samples / added / extended | extended - logged (CI) | success logged / extended | mass goal logged / extended |
|---|---|---|---|---|---|---|
| all start anchors | 4276 | 0.0794 | 0.035 / 0.057 / 0.054 / 0.054 / 0.051 | +0.0164 [+0.0121, +0.0207] | 0.185 / 0.199 | 0.017 / 0.017 |
| reset rows (t = 0) | 196 | 0.0038 | 0.031 / 0.390 / 0.347 / 0.356 / 0.301 | +0.2706 [+0.2325, +0.3086] | 0.176 / 0.400 | 0.014 / 0.027 |
| t in [1, 30) | 4037 | 0.0749 | 0.029 / 0.035 / 0.033 / 0.033 / 0.033 | +0.0038 [+0.0007, +0.0071] | 0.181 / 0.184 | 0.016 / 0.016 |
| logged shortcut episodes | 4071 | 0.0758 | 0.003 / 0.026 / 0.022 / 0.023 / 0.020 | +0.0164 [+0.0132, +0.0196] | 0.160 / 0.173 | 0.014 / 0.015 |
| logged detour episodes | 205 | 0.0037 | 0.683 / 0.713 / 0.701 / 0.703 / 0.700 | +0.0166 [-0.0463, +0.0777] | 0.699 / 0.734 | 0.063 / 0.059 |

| stratum | draw | mode not far-complete | of which another query far-complete | of which a sample far-complete | mode not success | of which another query success |
|---|---|---|---|---|---|---|
| reset rows | 0 | 122 | 67 | 67 | 105 | 58 |
| reset rows | 1 | 121 | 66 | 66 | 98 | 57 |
| all start anchors | 0 | 4029 | 150 | 142 | 3442 | 283 |
| all start anchors | 1 | 4031 | 151 | 144 | 3335 | 292 |

Reliability: far-complete rate per (anchor, query) draw 0 / draw 1 = 0.0523 / 0.0516; agreement of the far-complete indicator across draws 0.992; Jaccard of the far-complete sets 0.856; distinct anchors with a logged far-complete future draw 0 / 1: 159 / 158.

## seed_2 (n = 4276 anchors; policy scale median 0.164; restore max diff 0.0e+00)

G1 completed-far gain (extended - logged) +0.0097 CI [+0.0061, +0.0133] -> FAIL; G2 goal mass extended / logged 0.019 / 0.019 -> pass; G3 distinct anchors with an added far-complete future draw 0 / 1: 367 / 363 -> pass.

| query set | entered far | completed far | entered zone | success | death | timeout | length | mass far | mass goal | mass zone |
|---|---|---|---|---|---|---|---|---|---|---|
| logged | 0.040 | 0.033 | 0.904 | 0.220 | 0.678 | 0.102 | 202 | 0.039 | 0.019 | 0.153 |
| mode | 0.056 | 0.048 | 0.887 | 0.235 | 0.663 | 0.102 | 209 | 0.053 | 0.019 | 0.150 |
| samples | 0.053 | 0.044 | 0.891 | 0.230 | 0.668 | 0.102 | 207 | 0.051 | 0.019 | 0.150 |
| added | 0.054 | 0.045 | 0.890 | 0.231 | 0.667 | 0.102 | 207 | 0.052 | 0.019 | 0.150 |
| extended | 0.052 | 0.043 | 0.892 | 0.229 | 0.669 | 0.102 | 206 | 0.050 | 0.019 | 0.151 |

| single query | entered far | completed far | success | death | timeout | mass goal |
|---|---|---|---|---|---|---|
| logged | 0.040 | 0.033 | 0.220 | 0.678 | 0.102 | 0.019 |
| mode | 0.056 | 0.048 | 0.235 | 0.663 | 0.102 | 0.019 |
| sample0 | 0.052 | 0.044 | 0.229 | 0.670 | 0.102 | 0.019 |
| sample1 | 0.057 | 0.047 | 0.233 | 0.665 | 0.101 | 0.019 |
| sample2 | 0.051 | 0.043 | 0.228 | 0.670 | 0.102 | 0.019 |
| sample3 | 0.054 | 0.043 | 0.229 | 0.668 | 0.103 | 0.019 |

| paired delta vs logged | entered far | completed far | success | death | timeout | mass goal | mass far |
|---|---|---|---|---|---|---|---|
| extended_minus_logged | +0.0115 [+0.0080, +0.0150] | +0.0097 [+0.0061, +0.0133] | +0.0090 [+0.0036, +0.0145] | -0.0091 [-0.0134, -0.0048] | +0.0001 [-0.0055, +0.0054] | -0.0001 [-0.0010, +0.0007] | +0.0110 [+0.0079, +0.0141] |
| added_minus_logged | +0.0138 [+0.0096, +0.0183] | +0.0116 [+0.0073, +0.0162] | +0.0108 [+0.0047, +0.0174] | -0.0109 [-0.0163, -0.0057] | +0.0001 [-0.0068, +0.0064] | -0.0001 [-0.0011, +0.0009] | +0.0131 [+0.0094, +0.0169] |
| mode_minus_logged | +0.0162 [+0.0103, +0.0223] | +0.0147 [+0.0089, +0.0208] | +0.0153 [+0.0069, +0.0236] | -0.0154 [-0.0222, -0.0089] | +0.0001 [-0.0082, +0.0080] | +0.0002 [-0.0011, +0.0015] | +0.0149 [+0.0102, +0.0200] |
| samples_minus_logged | +0.0132 [+0.0089, +0.0175] | +0.0108 [+0.0064, +0.0151] | +0.0097 [+0.0033, +0.0162] | -0.0098 [-0.0152, -0.0044] | +0.0001 [-0.0067, +0.0065] | -0.0002 [-0.0012, +0.0008] | +0.0127 [+0.0091, +0.0164] |

| stratum | n | weight | completed far logged / mode / samples / added / extended | extended - logged (CI) | success logged / extended | mass goal logged / extended |
|---|---|---|---|---|---|---|
| all start anchors | 4276 | 0.0794 | 0.033 / 0.048 / 0.044 / 0.045 / 0.043 | +0.0097 [+0.0058, +0.0134] | 0.220 / 0.229 | 0.019 / 0.019 |
| reset rows (t = 0) | 196 | 0.0038 | 0.026 / 0.302 / 0.235 / 0.248 / 0.211 | +0.1847 [+0.1455, +0.2236] | 0.205 / 0.374 | 0.020 / 0.023 |
| t in [1, 30) | 4037 | 0.0749 | 0.029 / 0.031 / 0.030 / 0.030 / 0.030 | +0.0013 [-0.0015, +0.0041] | 0.216 / 0.218 | 0.019 / 0.018 |
| logged shortcut episodes | 4071 | 0.0758 | 0.003 / 0.020 / 0.016 / 0.017 / 0.015 | +0.0116 [+0.0091, +0.0144] | 0.196 / 0.207 | 0.017 / 0.017 |
| logged detour episodes | 205 | 0.0037 | 0.654 / 0.615 / 0.619 / 0.618 / 0.624 | -0.0298 [-0.0826, +0.0264] | 0.708 / 0.682 | 0.058 / 0.057 |

| stratum | draw | mode not far-complete | of which another query far-complete | of which a sample far-complete | mode not success | of which another query success |
|---|---|---|---|---|---|---|
| reset rows | 0 | 134 | 40 | 39 | 105 | 38 |
| reset rows | 1 | 137 | 43 | 42 | 103 | 41 |
| all start anchors | 0 | 4064 | 165 | 155 | 3323 | 478 |
| all start anchors | 1 | 4072 | 173 | 159 | 3215 | 490 |

Reliability: far-complete rate per (anchor, query) draw 0 / draw 1 = 0.0440 / 0.0438; agreement of the far-complete indicator across draws 0.995; Jaccard of the far-complete sets 0.892; distinct anchors with a logged far-complete future draw 0 / 1: 144 / 145.

