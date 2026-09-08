# V5 rockfall-clock: composed failure bank + failure-negative alpha sweep

Port of the PointMaze f4 design (`48e90db` composed bank, `85c0904` critic
audit) onto AntMaze V5 (`dc6690a`: own-clock rockfall, far05 dataset, XY goal).
Implementation and small-scale validation only — no production sweep has been
run.

Everything below is measured on this machine unless a line says otherwise.

---

## 1. Representation: what survives the projection into goal space

`crl/losses.py` implements failure-aware negatives by splicing bank entries
into the **goal half** of the observation, so a bank entry has to be a point in
goal space. Under the frozen V5 default that space is the torso XY pair.

f4 got away with this because its goal *is* the four-frame position stack and a
dead agent is frozen in it. V5 does not: the ant is killed by a **rule** (a
flagged dropped-rock contact) at a point a safe crossing also walks through.

`scripts/probe_v5_failure_representation.py`, 200 real deaths (do(go) under an
active latent) against the expert's own states. `AUC` is five-fold held-out
Fisher-LDA separating a death state from safe band crossings **matched on XY**
(fit and scored episodes do not overlap); `gap_*` are median
nearest-neighbour distances in units of the data's own per-column sigma;
`goal_off` is how far the commanded task goal sits from the nearest real state.

| goal columns | dim | AUC | gap_band | gap_hold | goal_off |
|---|---|---|---|---|---|
| xy — the frozen V5 default | 2 | **0.440** | 0.004 | 0.435 | 0.017 |
| xy+z | 3 | 0.657 | 0.034 | 0.500 | 4.870 |
| xy+linvel | 5 | 0.838 | 0.602 | 0.704 | 0.787 |
| xy+angvel | 5 | 0.601 | 0.300 | 0.841 | 0.261 |
| **xy+linvel+angvel (XYV)** | **8** | **0.879** | **1.316** | **1.515** | **0.815** |
| xy+z+linvel+angvel | 9 | 0.889 | 1.461 | 1.617 | 5.281 |
| xy+all velocity (incl. joints) | 16 | 0.891 | 2.062 | 2.590 | 1.039 |
| xy+full 27 (the padded goal) | 29 | 0.909 | 2.507 | 3.163 | 16.533 |

Three things this decides:

* **XY alone is chance.** 0.440 is not "weak", it is "the representation cannot
  tell a death from a safe crossing at all". A bank in that space can only say
  *the corridor is bad*; it cannot say *this way of being in the corridor is
  bad*. Projected onto XY, a death state IS the position of a normal passage.
* **`z` and the full padded goal are out.** `goal_off` is the "XY + 27 zeros"
  pathology as a number. Zero velocity is a state the ant really occupies (the
  expert's zero-torque hold); `z = 0` is an ant underground and
  `quat = (0,0,0,0)` is not a rotation. That is why XYV commands
  `[gx, gy, 0, 0, 0, 0, 0, 0]` and stops there.
* **The task goal keeps its meaning.** Reward is still
  `dist(xy, goal_xy) <= 0.5`, XY are still the first two goal columns, and the
  extra six say "at rest". Nothing about the V5 task changed.

Chosen: `GOAL_INDICES_XYV = (0, 1, 15, 16, 17, 18, 19, 20)` — torso XY plus
`qvel[0:6]` (linear and angular velocity). Env id
`offline_ant_umaze_rockfall_clock_v5_gxyv`, obs width 37 = state 29 + goal 8.

The frozen XY contract is **kept as a parallel arm** (`--goal-rep xy`), and
each representation carries its own alpha = 0 control. The original V5 baseline
(`scripts/train_rockfall_clock_v5_baseline.py`) is untouched.

### Extraction moment: settle 0, and why not more

`crl/rockfall_ant.py` (the V1/V2 rockfall family) has an opt-in
`death_settle_substeps` that lets physics run after a fatal contact so the
death develops into the observation. Measured for V5 as a read-only probe (the
env is not modified; the raw sim is stepped with zero ctrl after the env has
already declared the death), in the XYV columns:

| settle | AUC | gap_band | gap_hold | mean \|v\| | mean \|w\| |
|---|---|---|---|---|---|
| **0 (what V5 returns)** | 0.879 | 1.316 | **1.515** | 1.117 | 1.742 |
| 10 | 0.998 | 1.677 | 0.931 | 0.639 | 0.795 |
| 20 | 1.000 | 1.829 | 0.687 | 0.387 | 0.511 |
| 40 | 1.000 | 1.889 | 0.528 | 0.153 | 0.198 |
| 80 | 1.000 | 1.863 | 0.468 | 0.058 | 0.070 |
| 160 | 1.000 | 1.841 | 0.453 | 0.020 | 0.027 |
| *reference: expert HOLD* | | | | 0.076 | 0.084 |
| *reference: band crossing* | | | | 1.385 | 1.280 |

Settling buys separation from crossings and spends it against the hold: a
settled dead ant converges onto the motionless ant, which is exactly the safe
blind behaviour this benchmark is trying to elicit. Settle 0 is used — it also
happens to be the only choice that costs no change to the env at all.

Supporting numbers from the same probe: expert band crossings have planar speed
p1 = 0.900 and **0.0000** of them are below 0.3; the hold has median 0.001,
p95 = 0.333; deaths have median 0.899. The top separating columns at settle 0
are `vx` (d' 1.40), `qy` 0.58, `z` 0.55, `wy` 0.41 — the rock impact is a
deceleration-and-rotation jolt, not a change of place.

---

## 2. Failure sources

**The existing V5 data has no failures.** Audited first:
`antmaze_rockfall_clock_v5_far05.npz`, 400 episodes, 43 181 transitions,
`n_deaths = 0`, `success = 397/400` — the sighted expert never dies and the
collector asserts it. New collection was therefore necessary.

`scripts/collect_v5_failure_episodes.py` drives the **unmodified** env with
five behaviour policies. The rockfall keeps running on its own clock (`t0` is
drawn by the env's rng every reset); it is never forced always-on and never
triggered by the ant's arrival. The latent is forced active — deaths only exist
under an active latent — and that is recorded in the manifest as *generation*
using a privileged field, not merely selection.

Measured at 400 kept episodes per arm (`--all --seed 808`; independent actual
arm seeds are recorded below):

| arm | what it is | keep rate | death in band | mean length | schedule read |
|---|---|---|---|---|---|
| `random` | uniform torque `U[-1,1]^8` | **0 / 60** | — | — | no (structurally) |
| `noisy` | walking controller + Gaussian action noise (σ 0.3), blind | 400/400 | 0.950 | 31.8 | no (structurally) |
| `blind` | walking controller, no noise, timetable never read | 400/400 | 0.963 | 30.6 | no (structurally) |
| `deliberate` | sighted expert read the timetable, its rule said WAIT, overridden to GO | 400/401 | 0.950 | 30.9 | **yes** |
| `mistimed` | sighted expert waited, hold cut short by 30 steps, entered while the burst was open | 400/400 | 0.950 | 71.5 | **yes** |

* **Uniform random torque is task-irrelevant here and is reported as such.**
  0 failures in 60 episodes, all timeouts, mean furthest torso x = **0.93**
  against a band that starts at x = 2.6. The ant never leaves the start cell.
  The arm is kept in the code and writes a `..._EMPTY.json` recording the
  numbers, so the "random" 60% is served by the **noisy-controller** arm and is
  called that everywhere. It is not passed off as uniform random.
* **`schedule_read` is structural, not a promise.** The blind arms are handed a
  `RedactedSchedule` that raises on `active` / `start` / `end`. They cannot
  consult the timetable; the field records what happened.
* **The three ways of entering are distinguished by mechanism, not by name.**
  `blind` never read it; `deliberate` read it, obtained WAIT, and was
  overridden (sidecar shows `teacher_decision='wait'`, `schedule_read=True`,
  `intervention='override_wait_to_go'`, `hold_steps=1` — the single step at the
  mouth before the override); `mistimed` read it and actually held (`hold_steps`
  median 42, range 25–58; death at step median 72, range 53–95, against
  `deliberate`'s median 31) before releasing early. `noisy` is execution
  noise on a blind walk.
* **Only real deaths are kept.** The keep rule is the env's own
  `info['failure']` (a flagged dropped-rock contact) plus "in the corridor row".
  An intervention that does not kill is dropped and counted. V5 has no
  fall-death, so an ant that merely falls over runs to the horizon and is
  recorded as `timeout`; the drop counts are in the manifest
  (`deliberate` dropped exactly 1, reason `not_a_rock_death:timeout`).

Representative failure cases — episode 0 of each arm, read from
the sidecars. `x, y, |v|, |w|` are the death state the bank would store:

| arm | ep | death step | t0 | mouth | band entry | held | teacher said | read schedule | intervention | x | y | \|v\| | \|w\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `noisy` (seed 945) | 0 | 32 | 0 | 17 | 27 | 0 | — | no | `action_noise` | 2.78 | -0.01 | 0.77 | 0.80 |
| `blind` (seed 1082) | 0 | 29 | 24 | 15 | 26 | 0 | — | no | `none` | 2.95 | -0.13 | 1.37 | 3.91 |
| `deliberate` (seed 1219) | 0 | 26 | 4 | 14 | 25 | 1 | `wait` | **yes** | `override_wait_to_go` | 2.78 | 0.07 | 1.14 | 1.03 |
| `mistimed` (seed 1356) | 0 | 69 | 15 | 20 | 68 | 37 | `wait` | **yes** | `early_release` | 2.81 | -0.13 | 1.57 | 0.98 |

All four die at essentially the same place (x ~2.8, y ~0) and differ in *how*
they got there and how they were moving — which is exactly why the XY-only goal
cannot tell them from a safe crossing. The `deliberate` row shows the pattern:
`hold_steps = 1` is the single step at the mouth where the sighted rule ran and
returned WAIT before the override took over. `mistimed` really waited (37 steps)
and died at step 69 instead of 26.

Under the default 60/40 compose only `noisy` and `deliberate` enter the bank;
`blind` and `mistimed` stay in the pool and are available to other mixtures.

---

## 3. Bank

`scripts/make_v5_failure_bank.py --compose noisy=0.6,deliberate=0.4
--max-bank 256 --seed 0`

* pool 1600 (4 arms x 400); `random` pool 0.
* kept **154 noisy (0.602) + 102 deliberate (0.398)** = 256.
* one entry per failed episode — the observation V5 itself returned on the
  fatal step, asserted to be the last valid row (`death_row == lengths - 1`),
  so no episode can gain weight from repeated post-death frames.
* stratified **without replacement**, largest-remainder rounding; a short pool
  is a hard error, never a top-up from another class and never a duplicate.
* stored as the full **29-dim learner state**; `crl/train.py` slices it with
  `config.goal_indices`, so one bank file serves both goal contracts and the
  XY / XYV comparison is exact.
* 0.945 of entries inside the band, mean |v| 1.168.
* content sha256 `3a0a671d2f001edd7a89a9e37e6973677c83f9d5eaafc0c4b079eeb29a0ff39a`.
* `selection_uses_privileged_fields`:
  `source_arm, teacher_decision, schedule_read, intervention, outcome` — class
  selection only. The stored vectors carry no provenance; the manifest records
  per entry the source file, episode id, death step, schedule, mouth/band steps
  and collection seed.

**60/40 and alpha are independent knobs.** 60/40 is the composition *inside*
the bank (`--compose`); alpha is the weight the bank carries in the critic's
negative term (`--alpha`). Both are set explicitly and separately.

---

## 4. Training and sweep

`crl/losses.py` is used unchanged — the loss-level mixture

```
L(alpha) = S_pos/B^2 + (1-alpha) * S_neg/B^2 + alpha * (B-1)/B * E_fail[BCE]
```

which keeps the positive term and the total negative mass at their original
weights. It is a mixture, not an extra penalty added on top.

* recipe: the frozen offline AntMaze one (`verify_offline_d4rl.build_offline_cfg`
  — bc 0.05, twin-min, entropy 0, **batch 1024**, repr 16, hidden (1024,1024),
  discount 0.99, 4 sgd steps/step) with the V5 env, far05 dataset, horizon 400.
  PointMaze's batch 256 is not carried over; the bank cap here is
  batch_size = 1024 and the bank is 256.
* `--arm base` (alpha 0) loads **no bank at all**, so the failure branch is
  skipped entirely.
* the alpha = 0 XY arm was checked **field for field against
  `scripts/train_rockfall_clock_v5_baseline.py --variant far05 --method crl
  --goal-rep xy`** and is identical apart from `ckpt_dir`.
* far05's 27 detour episodes are asserted present by the gate; no failure
  trajectory is removed, future-goal relabeling is untouched, the training set
  is not swapped.
* the failure **episodes do not enter the positive training set** by default.
  That choice is recorded either way in `arm_provenance.json`
  (`failure_episodes_in_training_set`); `--train-npz` overrides the training
  file and must be held fixed across a sweep.
* all alphas share the same frozen bank, dataset, budget and paired seeds; run
  directories carry the alpha (`v5fn_xyv_a3_s0`) so no arm overwrites another.
* entry points: `check` / `smoke` / `run` in the sweep driver, and
  `--diff / --check-only / --smoke / --run / --resume / --alpha / --seed /
  --bank / --goal-rep` in the launcher.

---

## 5. Verification (all run here, CPU)

`python scripts/verify_v5_failneg.py` — **9/9 PASS**

| check | result |
|---|---|
| V1 bank composition | n=256, noisy 0.602 / deliberate 0.398 vs requested 0.60/0.40 |
| V1b real failures only | extraction "settle 0", every entry a rock death |
| V2 no duplicate episodes | 256 unique (seed, arm, episode); 256 unique vectors |
| V3 dimensions | stored (256, 29) -> (256, 8) under xyv; 256 <= batch 1024 |
| V4 normalisation | bank row 0 byte-identical to the raw death row; no obs_norm |
| V5 no privileged input | training npz keys = obs/act/lengths/eval_goals/meta, width 37; failure npz leaks none; `train.py` reads `goals` only |
| V6 alpha = 0 equivalence | bank loaded at alpha 0 vs no bank: **max \|dq\| 0, max \|dpi\| 0** after 3 updates |
| V7 alpha > 0 is live | alpha 0.3 moves params (max \|dq\| 1.8e-3); pos 0.00462 + ord 0.26374 + fail 0.07650 = reported 0.34486; alpha 1e-8 vs baseline loss gap 0 |
| V8 bank/eval isolation | 800 held-out failure episodes (base seed 909), 0 shared identities or death states with the bank (base seed 808) |

Also run end to end:

* `scripts/make_v5_gxyv_dataset.py` — 58 -> 37 columns, states/actions/lengths
  bitwise identical to the 58- and 31-column files.
* a 60-step smoke train at alpha 0.3 on the gxyv dataset: offline audit gates
  **G1–G8 all PASS**, `FAILURE-NEGATIVE BANK: ... (256 states -> goal dim 8),
  alpha=0.3`, checkpoint written.
* the provenance gate for both arms and both goal reps.

### What the diagnosis measures

`scripts/audit_v5_failneg.py` asks the three f4 questions in V5's terms, and
deliberately does **not** port f4's velocity-response curve or its 21x21
two-dimensional action grid (the ant's action space is 8-D and its failure
mechanism is different):

1. `f(s, a, g_bank)` overall and per bank class.
2. **real pairing, primary**: held-out failure states (base seed 909, never in the
   bank — asserted) against the training set's own XY-nearest band crossings.
   Different episodes from different collection runs, nothing synthesised.
3. **the hold must survive**: `f(g_hold)` on real mouth-hold states, and
   `f(hold) - f(dead)`. If the bank has taught the critic that "slow" is death,
   this is where it shows.
4. goal families, centred (factual future, random state, real crossing, real
   hold, held-out failure, bank, task goal).
5. one clearly-labelled **synthetic** diagnostic: a real crossing with its six
   velocity columns zeroed ("the same ant, stopped here").
6. behaviour at the mouth: the critic's ranking of the **actor's own action**
   against the zero action (the expert's hold), plus the commanded action norm.
7. positives: nearest-neighbour distance from relabeled positive goals to the
   bank, in sigma units.

Deployment readouts come from the existing
`scripts/eval_rockfall_clock_v5_baseline.py` (extended to accept
`--goal-rep xyv`), which already reports success, death, **timeout**, route
shortcut/detour/none, band entry, hesitation, `waited_rate`, `stop_steps` and
discounted return — so "it stopped moving" cannot be read as an improvement.

---

## 6. Commands

```bash
# 0. datasets (the 58-column far05 set already exists; these are column selections)
python scripts/make_v5_gxy_dataset.py
python scripts/make_v5_gxyv_dataset.py

# 1. the representation evidence
python scripts/probe_v5_failure_representation.py --n-death 300 --n-normal 300

# 2. failure episodes: the bank set (808) and an ISOLATED held-out set (909)
python scripts/collect_v5_failure_episodes.py --all --episodes 400 --seed 808
python scripts/collect_v5_failure_episodes.py --all --episodes 200 --seed 909 \
    --out-dir artifacts/v5_failneg/failures_heldout --name v5_failures_heldout

# 3. the composed 60/40 bank
python scripts/make_v5_failure_bank.py --compose noisy=0.6,deliberate=0.4 \
    --max-bank 256 --seed 0

# 4. verify before spending GPU time
python scripts/verify_v5_failneg.py

# 5. gates, then a smoke of every arm, then the sweep
bash scripts/run_v5_failneg_sweep.sh check
bash scripts/run_v5_failneg_sweep.sh smoke
SEEDS="0 1 2" bash scripts/run_v5_failneg_sweep.sh run

# the XY control sweep (same bank, the frozen V5 goal contract)
SEEDS="0 1 2" GOAL_REP=xy bash scripts/run_v5_failneg_sweep.sh run

# single arms
python scripts/run_v5_failneg.py --goal-rep xyv --arm base --seed 0 --run
python scripts/run_v5_failneg.py --goal-rep xyv --arm fail --alpha 0.3 --seed 0 \
    --bank artifacts/v5_failneg/bank/v5_failure_bank_n60d40.npz --run
python scripts/run_v5_failneg.py --goal-rep xyv --arm fail --alpha 0.3 --seed 0 --run --resume

# eval + audit (the sweep driver runs both, these are the standalone forms)
python scripts/eval_rockfall_clock_v5_baseline.py --ckpt v5fn_xyv_a3_s0/final.pkl \
    --goal-rep xyv --variant far05 --mode mean --n 300
python scripts/audit_v5_failneg.py --goal-rep xyv --runs-glob 'v5fn_xyv_*_s*'
```

Alphas swept: 0 (the `base` arm), 0.1, 0.2, 0.3, 0.4, 0.5 — `ALPHAS` in the
driver. The frozen XY/XYV dataset content hashes and bank content hash are
pinned in the driver; preflight recomputes and checks them before every arm.

---

## 7. Not run

* **No production training.** No 100k-step run at any alpha, in either goal
  representation. This machine's JAX is CPU-only; batch 1024 through a
  1024x1024 MLP is not a sensible place to run the sweep, and the driver
  refuses `run` on a CPU backend unless `FORCE_CPU=1`.
* Consequently: no production deployment or critic alpha-response numbers.
  `scripts/audit_v5_failneg.py` was exercised on the paired 400-step smoke
  checkpoints (300 paired anchors/probes): alpha 0 vs 0.3 gave mean
  alive-minus-dead margins 0.525 vs 1.571 and positive-margin fractions 0.447
  vs 0.807.  This confirms
  the diagnostic and representation are live, but 400 CPU updates are far too
  small for those values to support a scientific conclusion. These legacy
  smoke checkpoints predate the corrected metadata/seed provenance hashes;
  their learner tensors and selected bank goal vectors are unchanged, and the
  audit records the explicit provenance override. Production runs do not use
  that override and require exact dataset/bank hashes.
* `--train-npz` (failure episodes merged into the positive set) is implemented
  and recorded but no merged dataset has been built.
