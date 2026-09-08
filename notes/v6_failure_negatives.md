# V6 two-rockfall: composed failure bank + failure-negative alpha sweep

Failure-aware negative sampling on the long rectangular two-rockfall AntMaze V6
benchmark, reusing the production AntMaze failure-aware CRL pipeline and the
PointMaze composed-bank rule.

```
LEVEL 1   q_alpha(g) = (1-alpha) q_normal(g) + alpha q_fail(g)     per run
LEVEL 2   q_fail     = 0.60 q_random + 0.40 q_deliberate           in the bank
LEVEL 3   q_deliberate = 0.50 q_zone1 + 0.50 q_zone2               exact
```

Everything below was run on this machine (CPU) unless a line says otherwise.
**No alpha training has been run.** That step needs the GPU.

---

## 1. What was reused, and what is new

**Reused unchanged** — no second implementation of anything:

| piece | where |
|---|---|
| `q_alpha` mixture, failure-negative NCE, metrics | `crl/losses.py` (untouched) |
| bank loading + goal projection | `crl/train.py` (untouched) |
| `fail_bank_path`, `fail_neg_alpha` | `crl/config.py` (untouched) |
| V6 config, benchmark contract, vanilla-CRL guard, manifest | `scripts/train_rockfall_clock_v6_baseline.py` (untouched, imported) |
| U00/U10/U01/U11 evaluation | `scripts/eval_rockfall_clock_v6_baseline.py` (untouched) |
| settled post-fatal state convention (`SETTLE_N = 80`) | `scripts/rebuild_failure_bank_settled.py`, `crl/rockfall_ant.py` |
| composed-bank rule, privileged provenance, hard composition assertions | PointMaze `scripts/make_swamp_f4_failure_bank.py` |

**Modified** — two minimal, opt-in interfaces, both inert when unused:

* `scripts/rockfall_clock_v6_teacher.py` — normal vs executed decision, plus a
  zone-specific deliberate override (§2 below).
* `crl/rockfall_clock_v6.py` — `death_settle_substeps`, default 0.
* `crl/envs.py` — one line passing `rockfall_death_settle_substeps` through.

**New**: `collect_v6_failure_candidates.py`, `make_v6_failure_bank.py`,
`audit_v6_failure_bank.py`, `run_v6_failneg.py`, `run_v6_failneg_sweep.sh`,
`regress_v6_teacher_patch.py`.

---

## 2. The V6 teacher compatibility issue, and the patch

**The issue.** The pre-patch teacher wrote the executed decision over the
normal one:

```python
if force_go or not zschedule['active']:
    self._decision[zone] = 'go'      # the sighted rule is never evaluated
```

so a forced-go episode was indistinguishable in provenance from a genuinely
clear one. And the V5 workaround does not transfer: the hold branch keys off
`_decision[zone] == 'wait'`, so latching WAIT first and switching intent later
leaves the ant standing there.

**The patch.** Three quantities where there was one:

* `_normal_decision[zone]` — what the privileged rule says. `None` means the
  rule was never consulted, which is what the blind `intent='go'` path now
  records instead of a `'go'` it never decided.
* `_decision[zone]` — the **executed** decision. `decisions` keeps this
  meaning, so every existing reader is untouched.
* `_deliberate_override[zone]` — true only where a `'wait'` was overridden.

The hold branch is keyed on the executed decision, so an overridden zone walks
in while its normal decision still records `'wait'`. `fresh()` gains
`deliberate_override_zone=None|1|2`; new properties `normal_decisions`,
`executed_decisions`, `deliberate_overrides`, `deliberate_override_zone`.

One further line: the timetable is no longer even *indexed* on the `force_go`
path, so a blind collector handed a redacted schedule is structurally blind
rather than blind by promise.

**Regression** (`scripts/regress_v6_teacher_patch.py`, all PASS):

| check | result |
|---|---|
| R1 replay the frozen dataset, 40 episodes at seed 606 | `obs`, `act`, `lengths`, `eval_goals` **bitwise identical**; all 22 behaviour sidecar columns identical |
| R2 override off | executed == normal for every zone decision, no override fired |
| R3 blind intent | executed all `'go'` as before, normal all `None` |

R1 is the real evidence: identical actions imply identical trajectories.

---

## 3. Failure candidates

The benchmark is untouched — maze, rocks, both coins, both clocks, p_active,
horizon. Only the behaviour policy changes. Clocks are **never** forced; the
targeted zone's latent **is** forced active (a disarmed zone can never make its
rule say `'wait'`), recorded as generation-by-privileged-field. The 5% teacher
detour coin is disabled **in this collector only**.

300 kept per arm, seed 707 + 137·i (`artifacts/v6_failneg/candidates/`):

| arm | keep rate | zones | mean length | schedule read |
|---|---|---|---|---|
| `random` — uniform torque | **0 / 60** | — | — | no (structurally) |
| `noisy` — walker + N(0, 0.3) noise, blind | 300/499 = 0.601 | Z1 193, Z2 107 | 94.4 | no (structurally) |
| `deliberate_z1` | 300/300 = 1.000 | Z1 300 | 65.2 | yes |
| `deliberate_z2` | 300/402 = 0.746 | Z2 300 | 145.4 | yes |

* **Uniform torque produces nothing here and is reported as such**: 0 failures
  in 60 episodes, all timeouts, mean furthest torso x **1.44** against a first
  band at x = 6.6. A `..._EMPTY.json` records it. The random/noisy 60% is
  therefore served by the noisy-controller arm and is named that everywhere.
* `deliberate_z2` mean length 145 vs `deliberate_z1` 65: it crosses zone 1
  first. Its only drop reason is `not_a_rock_death:success` (102) — it entered
  zone 2 and lived; those are dropped, never relabelled.

A held-out pool (seed 2024, 120/arm) is in
`artifacts/v6_failneg/candidates_heldout/`.

**Settled failure state.** `--settle 80`, the existing AntMaze convention: at
the flagged fatal contact the actor loses control, ctrl is zeroed and physics
advances 80 substeps inside the same env step. One entry per failed episode,
asserted to be the last valid row. Measured effect (audit E, episode 0, settle
0 vs 80): the 70 pre-fatal rows and all actions are **identical**, the fatal
row differs by max 3.74, and |v| goes 1.772 → 0.006.

---

## 4. The bank

`python scripts/make_v6_failure_bank.py --n-bank 250 --seed 0`

```
N_bank 250 = 150 random/noisy (0.60) + 50 deliberate Z1 (0.20) + 50 deliberate Z2 (0.20)
random by zone: Z1 101, Z2 49, ambiguous 0
settle 80 | stored as 29-dim learner states | sliced by config.goal_indices at load
sha256  fa58d571ff0670e80d4f01e4b9d703c43b70ad5ed957705a5805f9874e008acc
```

Stratified **without replacement**; a short pool is a hard error — never a
top-up from another class, never a duplicate, never a silently reduced N_bank
or moved 60/40. Every deliberate entry is re-verified from provenance *twice*:
once over the whole pool before selection, once over the selection.

Manifest: `artifacts/v6_failneg/bank/v6_failure_bank_r60_z20_z20_manifest.json`
(env version, git commit, dataset path + sha, candidate pool paths + hashes,
counts, collector seeds, p_active, goal dim, settle settings, composition seed,
override API, and the per-entry provenance of all 250).

---

## 5. Pre-training audit — 9/9 PASS

`python scripts/audit_v6_failure_bank.py`

| | check | result |
|---|---|---|
| A | normal V6 unchanged | executed == normal everywhere, no override fired |
| B | deliberate Z1 | 50 entries, all normal `wait` → executed `go` → real Z1 death |
| C | deliberate Z2 | 50 entries, all survived Z1, normal `wait` at Z2 → executed `go` → real Z2 death; 8/50 had U1 active and all 8 actually held before crossing |
| D | random source | arm `noisy`, blind by construction, zones {Z1 101, Z2 49, ambiguous 0}; uniform-torque emptiness recorded |
| E | settled state | prefix and actions identical, fatal row max‖d‖ 3.74, ‖v‖ 1.772 → 0.006 |
| F | learner visibility | critic sees `[250, 2] = state[:, [0, 1]]`; 9 provenance fields searched numerically, none present |
| G | composition | 150 / 50 / 50 of 250, 250 unique (file, episode), 250 unique vectors |
| H | alpha = 0 | bank at alpha 0 vs no bank: **max ‖dq‖ 0, max ‖dπ‖ 0** after 3 updates |
| H | alpha live | alpha 0.3 moves params; pos 0.00355 + ord 0.37594 + fail 0.01836 = reported 0.39785 |

### Reported, not gated: what a 2-dim XY goal can express

V6 CRL's headline contract is `_gxy`, goal = torso XY. Bank failure states vs
the training set's own crossings of the **same band**, matched on XY:

| zone | failures | crossings | xy gap | AUC in goal columns `[0,1]` | AUC in the full 29-dim state |
|---|---|---|---|---|---|
| 1 | 151 | 23736 | 0.039 | **0.507** | 1.000 |
| 2 | 99 | 23736 | 0.054 | **0.506** | 1.000 |

0.5 is chance. Mechanically: in the goal columns the critic actually sees, a
bank entry names a *place inside a band*, not a way of dying — the same
measurement the V5 line produced (0.502). The full state separates perfectly,
so the information exists in the observation and is dropped by the projection.
The spec fixes the goal representation to V6 CRL's, so this is recorded as a
property of the run, not changed.

---

## 6. Alpha sweep

One launcher for every alpha. It imports the baseline for the config, the
contract, the manifest and the vanilla guard, sets `fail_neg_alpha` and
`fail_bank_path`, and touches nothing else. The guard is re-run on each arm's
config with those two fields reset to vanilla — if it passes, the arm differs
from the frozen V6 recipe in `fail_neg_alpha` (plus the bank path it needs) and
in nothing else. **All 6 gates pass.**

```bash
# gates only (no training)
bash scripts/run_v6_failneg_sweep.sh check

# the sweep -- GPU
BANK_SHA=fa58d571ff0670e80d4f01e4b9d703c43b70ad5ed957705a5805f9874e008acc \
SEEDS="0 1 2" bash scripts/run_v6_failneg_sweep.sh run

# single arms
python scripts/run_v6_failneg.py --alpha 0.0 --seed 0 --steps 100000
python scripts/run_v6_failneg.py --alpha 0.3 --seed 0 --steps 100000

# evaluation (identical protocol for every alpha)
python scripts/eval_rockfall_clock_v6_baseline.py \
    --ckpt artifacts/v6_failneg/runs/<run>/final.pkl --n 300
```

alphas 0.0, 0.1, 0.2, 0.3, 0.4, 0.5. `--steps` must be divisible by the
horizon (800), so 100000 and 300000 are both valid. Every arm shares the env,
dataset, bank, steps, batch, lr, architectures, BC coefficient, optimizer,
positive and ordinary-negative sampling, seed protocol and eval protocol.

The failure episodes are **negative-bank data only**: the training set is the
frozen `antmaze_rockfall_clock_v6_gxy.npz`
(sha `8f82beaf…`, 1000 episodes, 259 171 transitions, `failure 0.0`), and
`failure_episodes_in_training_set: false` is recorded in every run's
`failneg_arm.json`.

---

## 6b. The p_active 0.40 round

The benchmark constant moved from 0.35 to 0.40 (`crl/rockfall_clock_v6.py`).
This was decided while `artifacts/rockfall_clock_v6/runs` was still EMPTY --
no V6 training run of any kind existed -- so nothing was tuned against a
result. The 0.35 artifacts are kept byte-for-byte; every default now describes
the 0.40 benchmark and reproducing 0.35 needs the values passed explicitly
(`--npz <0.35 file> --p-active-1 0.35 --p-active-2 0.35`), which
`_dataset_contract` enforces rather than silently mistrains.

The two rounds are SEPARATE EXPERIMENTS, not points on one curve: different
datasets, different banks, different alpha grids and different seed counts.

Dataset `antmaze_rockfall_clock_v6_p040{,_gxy,_sidecar}.npz`, seed 606, 1000
episodes, 262,493 transitions:

| | p 0.35 | p 0.40 |
|---|---|---|
| u1 / u2 measured | 0.345 / 0.364 | 0.389 / 0.414 |
| U00 / U10 / U01 / U11 | 421 / 215 / 234 / 130 | 363 / 223 / 248 / 166 |
| U1 independent of U2 | p 0.54 | p 0.51 |
| detour | 0.054, indep. p 0.30 | 0.054, indep. p 0.44 |
| expert failure | 0.000 | 0.000 |
| shortcut waits z1 / z2 / both | 0.344 / 0.318 / 0.088 | 0.389 / 0.359 / 0.115 |

Candidates at 0.40 (seed 707+137i, settle 80): uniform torque again **0 of 60**
(mean furthest x 1.45 vs band at 6.6); noisy 300/435 = 0.690 (zones 190/110);
deliberate_z1 300/300; deliberate_z2 300/418 = 0.718.

Bank `v6_failure_bank_r60_z20_z20_p040.npz`, N=250 = 150 / 50 / 50, random by
zone {Z1 97, Z2 53, ambiguous 0},
sha256 `09149346639f351f28efe3a8082292d6ba143ca840f6c2bc4c1e19ad2f90a301`.

Audit **9/9 PASS**, same shape as the 0.35 round; 10 of the 50 banked zone-2
entries had U1 armed and held at zone 1 before going in. Goal expressiveness
(reported, not gated): AUC **0.505 / 0.500** in the goal columns against
**1.000** in the full 29-dim state.

---

## 7. Not run, and one deviation

* **No alpha training.** This machine's JAX is CPU-only; the sweep driver
  refuses `run` on a CPU backend without `FORCE_CPU=1`. No per-alpha success /
  failure / timeout, no U00/U10/U01/U11 breakdown, no alpha response. An
  alpha = 0.3 smoke was started on CPU purely to exercise the training path:
  offline gates G1–G8 PASS and the bank loads as
  `250 states -> goal dim 2, alpha=0.3`.
* **p_active.** The task text said 0.37; the code and the frozen dataset were
  at 0.35, and `_dataset_contract` refuses a CLI/dataset mismatch, so 0.37
  would have meant regenerating the main dataset. The 0.35 round was run as
  frozen, and the benchmark was subsequently moved to **0.40** on request (see
  section 6b) with a freshly collected dataset, candidate pools and bank. The
  deliberate collectors force the targeted latent, so the deliberate half of a
  bank does not depend on p_active at all; only the noisy arm's yield and zone
  split move with it.
