# V6 two-rockfall: the teacher-detour dataset ladder

Six demonstrator datasets for the long two-rockfall AntMaze V6 benchmark
(p_active 0.40 / 0.40) that differ in exactly one number: the probability
that the teacher takes the long safe perimeter route instead of the straight
shortcut through both hazard zones. Collaborators pick a rung with one
parameter; everything else (benchmark, seed, failure bank, training recipe,
evaluation) is unchanged.

```
rung   0.05   0.10   0.15   0.20   0.25   0.30
```

---

## 1. How the 5% was made, and why the ladder is a controlled comparison

The route is a **per-episode coin**, not a fixed set of far-route
trajectories. [`scripts/collect_rockfall_clock_v6_dataset.py`](../scripts/collect_rockfall_clock_v6_dataset.py)
draws, at the start of each of the 1000 episodes,

```python
take_detour = bool(route_rng.random() < args.teacher_detour_prob)
```

from `route_rng = default_rng(seed + ROUTE_SEED_OFFSET)`, a stream separate
from the environment's six reset streams (U1, U2, t0_1, t0_2, two jitter
streams) and from the Ant reset-pose / goal stream. Every RNG draw in an
episode happens at reset with a fixed draw count, and the teacher is
deterministic (frozen walker, latched turns, no noise). Consequently, for the
same collection seed:

* **nested** — the uniform draw for episode *k* is the same number on every
  rung, so the detour set at 0.05 is a subset of the set at 0.10, which is a
  subset of the set at 0.15, and so on;
* **shared shortcut episodes are byte-identical** — an episode that is a
  shortcut on two rungs has identical observations, actions, length and goal
  on both;
* **the environment never sees the coin** — U1/U2/t0/jitter/reset pose are
  identical episode-for-episode across all six rungs.

Raising the probability therefore does one thing only: it promotes more
episodes from shortcut to detour. Each promoted episode is a fresh rollout of
the same reset (same hazards, same clocks, same initial pose) along the other
route. The total stays 1000, so the shortcut count — the episodes that carry
the go/wait decisions at the two mouths — falls from ~950 to ~700 across the
ladder. That is the intended reading of "x% of the dataset is the long
route"; a design that holds the shortcut count fixed and appends detours was
considered and not chosen.

The failure bank is rung-independent. `collect_v6_failure_candidates.py`
disables the route coin (every candidate episode is a shortcut with a
deliberate or noisy `go` into an active zone), so the composed
60/20/20 bank `v6_failure_bank_r60_z20_z20_p040.npz` serves every rung
unchanged.

## 2. Files

All in `artifacts/rockfall_clock_v6/dataset/`, seed 606, 1000 episodes,
horizon 800, p_active 0.40/0.40, t0 ranges [10,45] / [90,125]. The 0.05 rung
is the original `_p040` file (unchanged bytes, so every existing run manifest,
hash and bank provenance stays valid); the others carry `_far{pp}`.

| rung | stem | detour episodes | shortcut | teacher success |
|---|---|---|---|---|
| 0.05 | `antmaze_rockfall_clock_v6_p040` | 54 | 946 | 0.999 |
| 0.10 | `antmaze_rockfall_clock_v6_p040_far10` | 111 | 889 | 0.997 |
| 0.15 | `antmaze_rockfall_clock_v6_p040_far15` | 165 | 835 | 0.996 |
| 0.20 | `antmaze_rockfall_clock_v6_p040_far20` | 217 | 783 | 0.994 |
| 0.25 | `antmaze_rockfall_clock_v6_p040_far25` | 267 | 733 | 0.990 |
| 0.30 | `antmaze_rockfall_clock_v6_p040_far30` | 310 | 690 | 0.990 |

Every non-success on every rung is a **detour timeout**, never a failure:
1, 3, 4, 6, 10, 10 episodes (1.9-3.7% of the detour episodes on each rung).
They are a property of the frozen detour controller, already present on the
0.05 rung (1/54), and they split into two stall modes: the Ant never
completes the first turn north (x ~ 2) or wanders on the final south leg
(x ~ 24-25, y between -1.4 and 4.6) without settling at the goal. The
collector keeps every natural draw by design, so they stay in the data as on
the 0.05 rung; the composition gates `teacher_success_above_0p90` and
`kept_teacher_failures_zero` pass on every rung.

Each stem has four files: `.npz` (58-column `state(29) | goal(29)`),
`_gxy.npz` (31-column `state(29) | goal_xy(2)`, the training default),
`_sidecar.npz` (privileged per-episode metadata: route, U1/U2, t0, decisions)
and `_composition_audit.json` (the collector's 17 gates, all PASS on every
rung).

`detour_ladder_audit.json` in the same directory is the cross-rung audit
written by [`scripts/audit_v6_detour_ladder.py`](../scripts/audit_v6_detour_ladder.py):
per rung, the composition audit passed, the configured probability sits
inside the realized Wilson99 interval, the SHA-256s match the audit, and the
`_gxy` file is a column selection of the raw file; per pair of rungs, the
nesting, the byte-identity of shared episodes, and the identity of the
environment draws. All 6 rungs and all 15 pairs PASS. Between 0.05 and 0.30, 690 shortcut episodes are shared byte-for-byte and 256 episodes were promoted to the detour; no episode was ever demoted.

A byte-identity check of the 0.05 rung itself: re-collecting
`--teacher-detour-prob 0.05` at seed 606 reproduced the committed `_p040`
learner files exactly (`obs`, `act`, `lengths`, `eval_goals` of both learner files and every sidecar column identical, NaN padding aside; only the file names inside the JSON meta differ). The re-collection is not committed.

## 3. Picking a rung

One flag, checked against the dataset's own metadata so a rung/dataset
mismatch is refused rather than silently trained:

```bash
# alpha sweep on the 15% rung (gates first, then the sweep)
DETOUR=0.15 bash scripts/run_v6_failneg_sweep.sh check
DETOUR=0.15 SEEDS="0 1 2" bash scripts/run_v6_failneg_sweep.sh run

# a single arm
python scripts/run_v6_failneg.py --alpha 0.0 --seed 0 --teacher-detour-prob 0.15

# the vanilla baseline trainer directly
python scripts/train_rockfall_clock_v6_baseline.py --steps 100000 --seed 0 \
    --teacher-detour-prob 0.15
```

* `--teacher-detour-prob` (default 0.05) resolves the dataset:
  `_p040` for 0.05, `_p040_far{pp}` otherwise. A value with no collected
  rung fails with the exact collector command to produce it.
* Run directories and sweep log directories get a `_far{pp}` suffix on every
  rung except 0.05, so rungs never overwrite each other:
  `artifacts/v6_failneg/runs/v6fn_a0_s0_100k_gxy_p0.4-0.4_h800_far15/`,
  `logs/v6_failneg_sweep_far15/`.
* `failneg_arm.json` and `benchmark_config.json` record the rung; the sweep's
  evaluation and results table only touch runs of the rung it was launched
  with.
* `eval_rockfall_clock_v6_baseline.py` reads the rung from the checkpoint's
  manifest (the evaluation environment does not depend on it) and records it
  in the result json. `audit_v6_failneg_critic.py --teacher-detour-prob`
  selects the matching dataset for the post-training critic audit.

Adding a rung outside the six (say 0.40): run the collector command the
trainer prints, then `python scripts/audit_v6_detour_ladder.py --rungs 0.05
0.4` to confirm it is nested and shares the shortcut episodes.

## 4. What this does not change

Nothing in `crl/` was touched. The teacher, the environment, the failure
bank, the vanilla recipe and its guard, the evaluation protocol and the
`_p040` bytes are as they were at the previous commit; the 0.05 rung with no
flag reproduces the previous run names and manifests exactly.
