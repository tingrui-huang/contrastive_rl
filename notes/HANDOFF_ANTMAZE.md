# AntMaze rockfall — handoff

What the current configuration is, what is deprecated, and a short description of the
worst-case branch. No experimental results here — the worst-case branch is being redesigned.

Verified at commit `8c075fa`, branch `feature/continuous-action-agreement`.

---

## 1. Current configuration

Variant name: `local_detour_v2.1_sev0.80_p30_h800_resetfix_v1`

| item | value |
|---|---|
| env id | `offline_ant_umaze_rockfall` |
| p_active | 0.30 |
| severity | (0.80 severe, 0.15 impaired, 0.05 mild) — v2.1 |
| horizon | 800 |
| reset | resetfix_v1 |
| obs / action | 29-dim proprioception (hazard mask NOT visible) / 8-dim |
| dataset | `artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz` |
| dataset sha | `6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852` |
| failure bank | `artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz`, sha `8c502024…`, 16 states |
| budget / seed | 300k updates, seed 0 |
| trainer | `scripts/naive_rockfall_v2_crl.py` |
| eval | `scripts/diagnose_naive_rockfall.py --v2 --p-active 0.3 --reset-fix --horizon 800` (n=200, fixed seed bank) |

Run command:

```bash
python scripts/naive_rockfall_v2_crl.py \
    --npz artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz \
    --steps 300000 --seed 0 --ckpt-dir ./<RUN_ID> \
    --p-active 0.3 --horizon 800 --reset-fix \
    --fail-bank artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz \
    --fail-neg-alpha 0.1
```

Fresh GPU node setup: `bash scripts/failneg_h800_node_setup.sh` (creates `~/crlenv`).
Activate it before running — the launcher scripts call `python`.
Do **not** install the 2022 `requirements.txt`; it is the dead Acme pin set.

---

## 2. IMPORTANT: module defaults are not the experiment

`crl/rockfall_ant.py` is frozen and its defaults are the **old pilot** values. Every
experiment overrides them per instance. Reading the module and assuming the defaults is
the single most common mistake here.

| knob | module default (do not use) | current value | override |
|---|---|---|---|
| `P_ACTIVE` | 0.2 | **0.30** | `--p-active 0.3` |
| `SEVERITY_PROBS` | (0.55, 0.30, 0.15) | **(0.80, 0.15, 0.05)** | `cfg.rockfall_severity` |
| `max_episode_steps` | 700 | **800** | `--horizon 800` |
| reset | legacy | **resetfix_v1** | `--reset-fix` |
| `death_settle_substeps` | 0 | **0 for training/eval** | see §4 |

---

## 3. Deprecated — do not use

**Datasets**

| directory | why not |
|---|---|
| `artifacts/rockfall_dataset/*` | v1 rockfall, superseded |
| `artifacts/rockfall_v2_dataset/pilot` | p=0.2 |
| `artifacts/rockfall_v2_dataset/pilot_sev055_ablation` | severity ablation only |
| `artifacts/rockfall_v2_dataset_p30` | H700 + legacy reset |
| `artifacts/rockfall_v2_dataset_p50` | p=0.50 ablation only |
| `artifacts/rockfall_v2_p30_h800` | legacy reset |
| `artifacts/rockfall_v2_p30_h700_resetfix` | H700 — this is the **formal benchmark**, but not what this line trains on |

**Two datasets live side by side in `rockfall_v2_p30_h800_resetfix` — they are not
interchangeable:**

| file | sha | episodes | use |
|---|---|--:|---|
| `pilot/…_pilot.npz` | `08bdc44b…` | 300 | FULL set (contains deaths). Only for the D_psi discriminator and old naive runs. |
| `failure_split/…_pilot_clean.npz` | `6bec8a52…` | 284 | **current** — deaths removed into the failure bank |

**Failure banks**

| file | sha | status |
|---|---|---|
| `…/failure_split/failure_bank.npz` | `8d35b76a…` | **DEPRECATED** — pre-physics-patch, frozen terminal obs |
| `artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz` | `8c502024…` | **current** |
| `…/failure_pool_diversity_audit/failure_pool_diverse.npz` | `f257e0f0…` | 603 states, **Flow training pool only** — never a critic bank |

Both 16-state banks cover the same 16 pilot deaths; they differ only in whether the
terminal state was allowed to settle physically. Never mix runs that used different banks.

**Flow generator versions**

| version | status |
|---|---|
| V0 (state only), V0.5 (+action) | superseded |
| V1 (source mixture, β sweep, 8 runs) | **failed** — fatal transitions too diluted |
| V2 (failure-local λ) | superseded by V3 |
| **V3** (`artifacts/flow_v3_diverse_failure/flow_v3/flow_v3.pkl`, sha `7b0ac9c8…`) | **current** |

**Old run directories** — `naive_rockfall_*`, `failneg_clean_*` are historical and use
either a different env variant, the full dataset, or the deprecated bank.

---

## 4. `death_settle_substeps` — special case

Fatal rock contact was originally invisible in the 29-dim obs (the env froze the last
observation). The patch lets MuJoCo settle the body for N steps after control ends, so the
terminal state is physically real.

- **Training and evaluation use `death_settle_substeps=0`** — byte-identical to legacy.
- **N=80 is used only** by `scripts/rebuild_failure_bank_settled.py` to build the settled
  failure bank.

Do not enable it for training.

---

## 5. Worst-case branch — how it currently works (being redesigned)

Two independent modifications to the sampler; the loss, architecture and optimizer are
untouched.

**(a) Failure-aware negatives** — 16 settled failure states act as extra negative goals:

```
L = L_pos + (1-alpha) * L_neg_batch + alpha * L_neg_fail      alpha = 0.1
```

`L_neg_fail` is the exact uniform expectation over the 16-state bank (all scored against
every anchor, no sampling). alpha = 0 reduces exactly to the baseline.

**(b) Pessimistic positive goal** — with probability `1 - rho`, replace the normal
future-state positive with a generated worst-case successor:

```
B ~ Bernoulli(rho)
g+ = normal future goal        if B = 1
   = obs_to_goal(s'_wc)        if B = 0
```

`s'_wc` comes from a three-step frozen pipeline, conditioned on the **logged** dataset
action `a_t^data`:

1. **Generate** — V3 Flow draws 256 candidate next states, 50-step Euler.
2. **Score** — distance of each candidate to the 16 failure states, Euclidean L2 in the
   frozen normalized 29-dim state space.
3. **Select** — take the argmin (most failure-like); ties go to the lowest index.

Critic C never enters the selection. `s'_wc` is treated as **absorbing**: it is returned
directly as the goal, with no trajectory continuation, no projection to a dataset state,
and no policy query. Because everything is frozen and the dataset is fixed, the whole map
is precomputed once into a static table (`artifacts/static_worstcase_rl/worstcase_table.npz`,
sha `d059db0c…`, 227,200 rows), so training does a lookup and never invokes the Flow.

Note: `rho` is a **required injected argument** — the module deliberately does not pick or
default it. Whatever replaces this design has to answer the same question: what decides
when the pessimistic branch fires.

Status: **the whole worst-case branch is being redesigned.** Treat the above as a
description of the existing code, not as a recommended method.

---

## 6. Do not touch

Rockfall module defaults, global-route v1 data and docs, legacy datasets and checkpoints,
the walker (`artifacts/walker/phase1/walker_best.pkl`, sha `70b0a460…`) and the base policy
(`offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl`, sha `6bece3e3…`).

**Those last two are not in git and never have been.** They are needed only for data
collection, GIF rendering, and `authoritative_eval.py` — **not** for training and not for
`diagnose_naive_rockfall.py`. A new machine can train and evaluate without them; it cannot
collect a new dataset without them.

New variants get a new directory and a new manifest; nothing existing is edited in place.
