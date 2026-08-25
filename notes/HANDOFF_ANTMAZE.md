# AntMaze rockfall — handoff

What the current configuration is, what is deprecated, and how positives/negatives are
sampled. No experimental results — the worst-case branch is being redesigned.

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

**Nothing errors if you forget them.** The run completes normally -- you will just have
silently trained on the old, easier pilot benchmark (sparser hazards, lower lethality,
shorter horizon), and the numbers will not be comparable to anything. Always pass the
overrides explicitly instead of relying on what the module says.

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

## 5. Sampling: what the positives and negatives are

### Negative samples (current, in use)

Per anchor `(s_i, a_i)` the critic sees **two kinds** of negative goal:

| kind | what it is | count per anchor |
|---|---|--:|
| ordinary (in-batch) | goals of the *other* rows in the batch, i.e. achieved future states from **different trajectories** — the off-diagonal of the B x B logit matrix | B - 1 = 1023 |
| failure | the **16 settled failure states** (`failure_bank_settled.npz`, sha `8c502024...`), the same 16 for every anchor | 16 |

They are combined at the **loss** level, not by resampling:

```
L = L_pos + (1 - alpha) * L_neg_ordinary + alpha * L_neg_failure     alpha = 0.1

L_pos          = sum_i    BCE(f(s_i,a_i,g_i), 1) / B^2        # diagonal
L_neg_ordinary = sum_i!=j BCE(f(s_i,a_i,g_j), 0) / B^2        # off-diagonal
L_neg_failure  = (B-1)/B * mean_{i,b} BCE(f(s_i,a_i,g^fail_b), 0)
```

Three properties worth knowing:

- `L_neg_failure` is the **exact uniform expectation** over the bank — all 16 states are
  scored against every anchor, nothing is sampled, so no extra variance enters.
- The positive term keeps its original coefficient and the total negative mass is
  preserved; `alpha = 0` reduces algebraically to the untouched baseline.
- Failure goals are always labelled negative for ordinary anchors. Batch size is 1024, so
  the bank is a small fraction of the negatives by count but carries weight `alpha`.

### Positive samples

Baseline: a future state from the *same* episode, `j ~ Categorical(gamma^(j-i))`, `j > i`.

### Worst-case branch (being redesigned — short version)

With probability `1 - rho` the positive goal is replaced by a generated worst-case
successor `s'_wc` instead of the normal future state:

1. V3 Flow draws 256 candidate next states from `(s_t, a_t^data)`, 50-step Euler.
2. Each candidate is scored by Euclidean L2 distance to the 16 failure states, in the
   frozen normalized 29-dim state space.
3. Take the argmin (most failure-like). Critic C is not involved.

`s'_wc` is treated as absorbing — returned directly as the goal, with no trajectory
continuation, no projection to a dataset state, and no policy query. Everything is frozen,
so the whole map is precomputed into a static table
(`artifacts/static_worstcase_rl/worstcase_table.npz`, sha `d059db0c...`, 227,200 rows) and
training only does a lookup.

`rho` is a required injected argument; the code deliberately does not pick one. Any
replacement has to answer the same question: what decides when the branch fires.

**This whole branch is being redesigned. The above describes existing code, not a
recommended method.** The negative-sample setup above is independent of it and stays.

## 6. Do not touch

Rockfall module defaults, global-route v1 data and docs, legacy datasets and checkpoints,
the walker (`artifacts/walker/phase1/walker_best.pkl`, sha `70b0a460…`) and the base policy
(`offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl`, sha `6bece3e3…`).

**Those last two are not in git and never have been.** They are needed only for data
collection, GIF rendering, and `authoritative_eval.py` — **not** for training and not for
`diagnose_naive_rockfall.py`. A new machine can train and evaluate without them; it cannot
collect a new dataset without them.

New variants get a new directory and a new manifest; nothing existing is edited in place.
