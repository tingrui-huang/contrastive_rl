# Gate: start-region margin north - east on recorded anchors (t <= 5)

451 north-moving and 3621 east-moving recorded rows (next-frame XY displacement), each scored with its own recorded goal; critic score = min over the twin heads.  Law target on the replay's query paths (gamma 0.999): r0.5 +0.23, r1.0 +0.24.  PASS = paired same-state margin >= +0.15 (300 states x 200 north / 200 east recorded torques).

| critic | mean f north | mean f east | row margin | s.e. | paired margin (same state) | s.e. | states > 0 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| branch critic seed 0 | -20.311 | -22.043 | +1.733 | 0.385 | **-0.529** | 0.110 | 0.36 | fail |
| branch critic seed 1 | -22.690 | -23.412 | +0.722 | 0.464 | **-0.649** | 0.112 | 0.37 | fail |
| branch critic seed 2 | -24.856 | -25.537 | +0.681 | 0.536 | **-0.588** | 0.131 | 0.41 | fail |
| branch critic seed 3 | -22.726 | -24.609 | +1.883 | 0.561 | **-0.048** | 0.122 | 0.48 | fail |
| branch critic seed 4 | -21.697 | -22.475 | +0.778 | 0.412 | **-0.287** | 0.095 | 0.42 | fail |
