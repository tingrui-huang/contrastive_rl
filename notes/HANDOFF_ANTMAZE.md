# AntMaze rockfall benchmark — handoff

Everything a new person needs to run this line without picking the wrong version.
Read section 1 and section 3 before touching anything.

Last verified: commit `8c075fa`, branch `feature/continuous-action-agreement`.

---

## 1. The 30-second version

| what | value |
|---|---|
| env id | `offline_ant_umaze_rockfall` |
| variant | `local_detour_v2.1_sev0.80_p30_h800_resetfix_v1` |
| p_active | **0.30** |
| severity | **(0.80 severe, 0.15 impaired, 0.05 mild)** — v2.1 |
| horizon | **800** for this research line (see the H700 trap in §3) |
| reset | **resetfix_v1** (`--reset-fix`) |
| training dataset | `artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz`, sha `6bec8a52…` |
| budget / seed | 300k updates, seed 0 |
| trainer | `scripts/naive_rockfall_v2_crl.py` |
| eval | `scripts/diagnose_naive_rockfall.py --v2 --p-active 0.3 --reset-fix --horizon 800`, n=200, fixed seed bank |

**None of the above are the module defaults.** They are all instance-level overrides. See §3.

---

## 2. Version inventory — what exists in this repo

### 2.1 Environment knobs (`crl/rockfall_ant.py`, module is FROZEN)

Five independent axes. The module default is the *pilot* setting on every one of them:

| knob | module default | authoritative for this line | how to override |
|---|---|---|---|
| `P_ACTIVE` | 0.2 | **0.30** | `cfg.rockfall_p_active` / `--p-active 0.3` |
| `SEVERITY_PROBS` | (0.55, 0.30, 0.15) | **(0.80, 0.15, 0.05)** | `cfg.rockfall_severity` (`rockfall_v2_teacher.SEVERITY_V2`) |
| `max_episode_steps` | 700 | **800** | `cfg.rockfall_max_steps` + `cfg.max_episode_steps` / `--horizon 800` |
| reset | legacy | **resetfix_v1** | `cfg.rockfall_reset_fix=True` / `--reset-fix` |
| `death_settle_substeps` | 0 | **80, but only for building the failure bank** | ctor param |

`death_settle_substeps=0` is byte-identical legacy behaviour and is what training/eval use.
N=80 is used **only** by `scripts/rebuild_failure_bank_settled.py` to make fatal contact
observable in the 29-dim obs. Do not enable it for training or evaluation.

### 2.2 Datasets (8 directories)

| directory | what | status |
|---|---|---|
| `artifacts/rockfall_dataset/{pilot,full,oracle}` | v1 rockfall | superseded |
| `artifacts/rockfall_v2_dataset/pilot` | v2 pilot, p=0.2 | superseded |
| `artifacts/rockfall_v2_dataset/pilot_sev055_ablation` | severity ablation | ablation only |
| `artifacts/rockfall_v2_dataset_p30` | v2, p=0.30, H700, legacy reset | superseded |
| `artifacts/rockfall_v2_dataset_p50` | v2, p=0.50 | ablation only |
| `artifacts/rockfall_v2_p30_h800` | H800, **legacy reset** | superseded by resetfix |
| `artifacts/rockfall_v2_p30_h700_resetfix` | H700 + resetfix, sha `ff5a8136…` | the FORMAL benchmark |
| **`artifacts/rockfall_v2_p30_h800_resetfix`** | H800 + resetfix | **this line** |

Inside `rockfall_v2_p30_h800_resetfix` there are **two** datasets and they are not
interchangeable:

| file | sha | episodes | use |
|---|---|--:|---|
| `pilot/…_pilot.npz` | `08bdc44b…` | 300 (incl. deaths) | the FULL set; used by the old naive baseline and by the D_psi discriminator |
| **`failure_split/…_pilot_clean.npz`** | **`6bec8a52…`** | **284** | **all failure-negative and worst-case runs** |

The clean split is the full set with rockfall-death episodes removed (those deaths become
the failure bank). Comparing a run on `08bdc44b` against a run on `6bec8a52` is a
**dataset-confounded comparison** — see §3.

### 2.3 Failure banks (3)

| file | sha | n | status |
|---|---|--:|---|
| `artifacts/rockfall_v2_p30_h800_resetfix/failure_split/failure_bank.npz` | `8d35b76a…` | 16 | **LEGACY** — pre-physics-patch, frozen last obs |
| **`artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz`** | **`8c502024…`** | **16** | **authoritative** (N=80 physically settled) |
| `artifacts/flow_v3_diverse_failure/failure_pool_diversity_audit/failure_pool_diverse.npz` | `f257e0f0…` | 603 | Flow V3 **training pool only** — never a critic bank |

The two 16-state banks cover the *same* 16 pilot deaths; they differ only in whether the
recorded terminal state settled physically. **They produce very different training
outcomes** (§5) — always record which one a run used.

### 2.4 Flow generator (5 generations)

| version | change | verdict |
|---|---|---|
| V0 | `q(ds\|s)`, state only | no fatal coverage |
| V0.5 | `q(ds\|s,a)` | interface/numerical check |
| V1 | source mixture `(1−β)·D_good + β·D_bad`, β ∈ {.05,.10,.15,.20} (8 runs in `flow_v1_sweep`) | **FAILED** — fatal transitions are diluted (r_fatal=7.8e-4) |
| V2 | failure-local `(1−λ)·D_good + λ·D_fail(196)` | works on training support, poor generalization |
| **V3** | λ=0.01, diverse 603-state failure pool from 4 non-privileged arms | **authoritative**, sha `7b0ac9c8…` |

Frozen sampling constants: K=256, 50-step Euler, `PRNGKey(11)`.
Normalization: `artifacts/flow_v0_clean/norm_stats.npz`, sha `262daa47…` (never recomputed).

### 2.5 Worst-case selector

Frozen in `artifacts/state_nn_selector_confirm/selector_freeze.json`:

```
d_neg(s') = min_{g in D_C^-(16)} || norm(s') - norm(g) ||_2
s'_wc     = argmin_{k<=256} d_neg(s'_k)      # ties -> lowest index
```

Normalized 29-dim **observable state space**, Euclidean L2, the 16-state bank, no Critic C.
Precomputed for all 227,200 clean transitions:
`artifacts/static_worstcase_rl/worstcase_table.npz`, sha `d059db0c…`.

### 2.6 Completed 300k runs (13)

Not all are comparable. Only the `*_h800_resetfix*` ones on the same dataset are.

| run | dataset | bank | comparable? |
|---|---|---|---|
| `naive_rockfall_v2_s0_300k`, `_p30_`, `_p50_`, `_p30_h800_`, `naive_rockfall_full_` | various | — | no, different env variants |
| `naive_rockfall_v2_p30_h700_resetfix_s0_300k` | H700 | — | no, different horizon |
| `naive_rockfall_v2_p30_h800_resetfix_s0_300k` | **full `08bdc44b`** | — | dataset-confounded |
| `failneg_clean_p30_h800_resetfix_a{0,005,01,02}_s0_300k` | clean | **legacy** | yes (α sweep) |
| `failneg_settledbank_a01_s0_300k` | clean | **settled** | yes |
| `wc_fixedp10_h800_a01_s0_300k` | clean | settled | yes (four-arm) |
| `wc_fixedp50…`, `wc_dpsi_surrogate…`, `blind_crl_clean…` | clean | settled | yes (four-arm) |

---

## 3. The four traps

1. **Module defaults are pilot values, not the experiment.** Reading
   `crl/rockfall_ant.py` and assuming P_ACTIVE=0.2 / H=700 / severity (0.55,0.30,0.15)
   is wrong for every run in this line. Always pass the overrides.
2. **H=700 is the formal benchmark; H=800 is a horizon-stress variant.** The verdict
   recorded in `notes/p30_h800_briefing.md` keeps the benchmark at H700. This research
   line nevertheless uses H800 data, because it only needs behavioural data and does not
   depend on the benchmark verdict. Do not mix H700 and H800 numbers in one table.
3. **Full vs clean dataset.** `08bdc44b` (300 eps) vs `6bec8a52` (284 eps). The old naive
   baseline used the full set; everything in the failure-negative / worst-case line uses
   the clean split. A cross-dataset comparison is confounded and has already caused one
   wrong conclusion (§5).
4. **Legacy vs settled bank.** Both are 16 states over the same deaths, and the run scripts
   pin the settled sha and *explicitly refuse* the legacy sha. But the older
   `failneg_clean_*` sweep used the legacy bank — do not put those numbers in the same
   column as settled-bank runs without saying so.

---

## 4. How to run

```bash
python scripts/naive_rockfall_v2_crl.py \
    --npz artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz \
    --steps 300000 --seed 0 --ckpt-dir ./<RUN_ID> \
    --p-active 0.3 --horizon 800 --reset-fix \
    --fail-bank artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz \
    --fail-neg-alpha 0.1
```

Add the worst-case branch with
`--wc-positive --wc-table artifacts/static_worstcase_rl/worstcase_table.npz
--wc-table-sha256 d059db0c… --wc-rho-mode fixed --wc-p-wc 0.10 --wc-coin-seed 0`.

Ready-made launchers with hard SHA gates:
`scripts/run_wc_fixedp10_h800_a01_s0_300k.sh`, `…fixedp50…`, `…dpsi_surrogate…`,
`scripts/run_blind_crl_clean_h800_a00_s0_300k.sh`.
Multi-server instructions: `artifacts/four_arm_wc_run/FOUR_SERVER_RUNBOOK.md`.

Environment on a fresh GPU node: `bash scripts/failneg_h800_node_setup.sh`
(builds `~/crlenv`; do **not** install the 2022 `requirements.txt`, it is the dead Acme
pin set). Activate with `source ~/crlenv/bin/activate` — the launchers call `python`.

Runtime: ~0.0117 s/update on an RTX 3090 → roughly 1.5–2.5 h per 300k run including
in-training evals and the two final 200-episode diagnoses.

---

## 5. Known variance — read before drawing conclusions

**Same config + same seed does not reproduce across machines.** Two runs with byte-identical
configs (verified: the only differences are the new, disabled `wc_*` fields):

| run | final | best |
|---|--:|--:|
| `failneg_clean_…_a0_s0_300k` (α=0, clean) | 0.770 | 0.785 |
| `blind_crl_clean_h800_a00_s0_300k` (α=0, clean) | 0.715 | 0.755 |

Δ = 0.055 / 0.030. Most likely cause is GPU/XLA floating-point nondeterminism compounding
over 300k updates. Practical consequence: **treat single-seed differences below ~0.06 as
noise.** At n=200 the binomial standard error alone is ~0.03.

**Bank version is a large lever.** Same α, same dataset, same seed:

| bank | final | best |
|---|--:|--:|
| legacy `8d35b76a…` | **0.870** | 0.745 |
| settled `8c502024…` | 0.645 | 0.660 |

A 0.225 gap at final — larger than any method effect measured so far. This is unexplained
and is the single most important open question in this line.

---

## 6. All comparable results (n=200, fixed eval bank, masks 48/45/53/54)

| run | dataset | α | bank | p_wc | final | best |
|---|---|--:|---|--:|--:|--:|
| naive (blind) | full `08bdc44b` | 0 | — | — | 0.515 | 0.520 |
| blind (α=0) | clean | 0 | — | — | 0.770 | 0.785 |
| blind (α=0), rerun | clean | 0 | — | — | 0.715 | 0.755 |
| fail-neg α=0.05 | clean | 0.05 | legacy | — | 0.825 | 0.720 |
| **fail-neg α=0.1** | clean | 0.1 | legacy | — | **0.870** | 0.745 |
| fail-neg α=0.2 | clean | 0.2 | legacy | — | 0.660 | 0.775 |
| fail-neg α=0.1 | clean | 0.1 | settled | — | 0.645 | 0.660 |
| WC fixed 0.10 | clean | 0.1 | settled | 0.10 | 0.820 | **0.845** |
| WC fixed 0.50 | clean | 0.1 | settled | 0.50 | 0.575 | 0.450 |
| WC D_psi surrogate | clean | 0.1 | settled | ~0.495 | 0.440 | 0.630 |

Every number is single-seed. Given §5, only differences well above ~0.06 should be
interpreted, and only within a fixed bank version.

---

## 7. Do not touch

Rockfall module defaults (P_ACTIVE 0.2 / severity (0.55,0.30,0.15) / H700), global-route v1
data and docs, legacy datasets and checkpoints, the walker
(`artifacts/walker/phase1/walker_best.pkl`, sha `70b0a460…`) and the base policy
(`offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl`, sha `6bece3e3…`).

**Those last two are NOT in git** and never have been. They are needed only by data
collection / rendering / `authoritative_eval.py` — *not* by training or by
`diagnose_naive_rockfall.py`. A new machine can train and evaluate without them; it cannot
collect new datasets or render GIFs without them.

New variants get a new directory and a new manifest; nothing in place is edited.
