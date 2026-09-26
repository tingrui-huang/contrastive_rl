# Gate: start-region margin north - east on recorded anchors (t <= 5)

451 north-moving and 3621 east-moving recorded rows (next-frame XY displacement), each scored with its own recorded goal; critic score = min over the twin heads.  Law target on the replay's query paths (gamma 0.999): r0.5 +0.60, r1.0 +0.62.  PASS = paired same-state margin >= +0.15 (300 states x 200 north / 200 east recorded torques).

| critic | mean f north | mean f east | row margin | s.e. | paired margin (same state) | s.e. | states > 0 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| branch critic seed 0 | -12.812 | -14.138 | +1.325 | 0.197 | **+0.107** | 0.061 | 0.54 | fail |
| branch critic seed 1 | -12.552 | -14.776 | +2.224 | 0.196 | **+0.255** | 0.057 | 0.63 | PASS |
| branch critic seed 2 | -13.068 | -14.553 | +1.484 | 0.200 | **+0.204** | 0.055 | 0.60 | PASS |
| branch critic seed 3 | -12.836 | -14.192 | +1.356 | 0.204 | **+0.107** | 0.062 | 0.53 | fail |
| branch critic seed 4 | -12.386 | -13.886 | +1.500 | 0.187 | **+0.077** | 0.062 | 0.54 | fail |
