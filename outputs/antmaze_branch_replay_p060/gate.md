# Gate: start-region margin north - east on recorded anchors (t <= 5)

451 north-moving and 3621 east-moving recorded rows (next-frame XY displacement), each scored with its own recorded goal; critic score = min over the twin heads.  Law target on the replay's query paths (gamma 0.999): r0.5 +1.02, r1.0 +1.05.  PASS = paired same-state margin >= +0.15 (300 states x 200 north / 200 east recorded torques).

| critic | mean f north | mean f east | row margin | s.e. | paired margin (same state) | s.e. | states > 0 | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| branch critic seed 0 | -21.665 | -23.810 | +2.144 | 0.381 | **-0.656** | 0.092 | 0.37 | fail |
| branch critic seed 1 | -22.811 | -24.422 | +1.610 | 0.382 | **-0.085** | 0.103 | 0.44 | fail |
| branch critic seed 2 | -23.745 | -25.858 | +2.113 | 0.481 | **+0.130** | 0.109 | 0.52 | fail |
| branch critic seed 3 | -23.801 | -26.153 | +2.352 | 0.474 | **-0.022** | 0.107 | 0.50 | fail |
| branch critic seed 4 | -23.122 | -25.935 | +2.813 | 0.456 | **-0.142** | 0.126 | 0.44 | fail |
