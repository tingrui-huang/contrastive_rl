# Gate: start-region margin north - east on recorded anchors (t <= 5)

451 north-moving and 3621 east-moving recorded rows (next-frame XY displacement), each scored with its own recorded goal; critic score = min over the twin heads.  Law target on the replay's query paths (gamma 0.999): r0.5 +0.60, r1.0 +0.62.  PASS = paired same-state margin >= +0.15 (300 states x 200 north / 200 east recorded torques).

| critic | mean f north | mean f east | row margin | s.e. | paired margin (same state) | s.e. | states > 0 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| branch critic seed 0 | -22.168 | -24.935 | +2.767 | 0.421 | **+0.165** | 0.104 | 0.54 | PASS |
| branch critic seed 1 | -21.623 | -24.507 | +2.884 | 0.429 | **+0.276** | 0.103 | 0.57 | PASS |
| branch critic seed 2 | -23.161 | -24.959 | +1.797 | 0.452 | **-0.261** | 0.103 | 0.46 | fail |
| branch critic seed 3 | -22.414 | -24.793 | +2.379 | 0.422 | **-0.056** | 0.108 | 0.46 | fail |
| branch critic seed 4 | -21.128 | -23.255 | +2.127 | 0.387 | **-0.164** | 0.102 | 0.47 | fail |
