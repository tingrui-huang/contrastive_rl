# Four-server runbook — worst-case integration ablation (300k each)

**Do not launch until the per-server setup below reports the exact SHAs.**
Each server runs exactly ONE arm. Nothing here starts automatically.

Scientific status: these are **mechanism / integration ablations**, not final
causal-method validation. Arm 3 does **not** validate `D_psi` as a propensity;
arms 1–2 have **no** causal propensity interpretation; one seed is not
statistical validation.

---

## The four arms

| server | run id | α_fail | WC branch | ρ mechanism | needs bundle |
|---|---|--:|:--:|---|:--:|
| 1 | `wc_fixedp10_h800_a01_s0_300k` | 0.1 | ✓ | fixed p_wc=0.10 (ρ=0.90) | yes |
| 2 | `wc_fixedp50_h800_a01_s0_300k` | 0.1 | ✓ | fixed p_wc=0.50 (ρ=0.50) | yes |
| 3 | `wc_dpsi_surrogate_h800_a01_s0_300k` | 0.1 | ✓ | ρ = D_psi **surrogate, NOT calibrated** | yes |
| 4 | `blind_crl_clean_h800_a00_s0_300k` | 0.0 | ✗ | none | **no** |

**Key matched comparison: server 2 vs server 3.** Measured smoke doses are
0.5002 and 0.4954 — equal within 0.005 — so the pair isolates whether `D_psi`
contributes useful state/action dependence *beyond its mean branch rate*.

Non-rerun references (already on disk, same protocol):

| reference | final | best |
|---|---|---|
| fail-neg only α=0.1, clean | succ 0.645, hazard 0.770, drop 0.335, center 0.260, both-sides 0.400 | step 208000: succ 0.660, hazard 0.615, drop 0.280, center 0.340, both-sides 0.578 |
| blind CRL, **FULL** data (secondary, not dataset-matched) | succ 0.515, hazard 0.980, drop 0.450, center 0.070, both-sides 0.200 | step 239200: succ 0.520, hazard 0.920, drop 0.430, center 0.110, both-sides 0.178 |

Compare **final-to-final and best-to-best**. Never compare a historical best
against a new-run final.

---

## Pinned artifacts

| artifact | sha256 |
|---|---|
| commit | `aae3245` (or later on `feature/continuous-action-agreement`) |
| clean dataset | `6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852` |
| worst-case table (227,200 rows) | `d059db0cdae4cd528cdb68e6c0caf7439cc5d66429a39b7b883ecf15d974ac4b` |
| failure bank (16 states) | `8c50202403317e8adc67670be716fffe5c3b3cb891017bfbe6eb070acd61d0ce` |
| Flow V3 | `7b0ac9c80afa8713155d30ee4b784b52bd6d5b58fc57ff73b6b900a11224803c` |
| V0 normalization | `262daa472316773b441e0dfed897275ffac13e10966728d39d4f9e23ffe8d4ca` |
| bundle tarball | `8e38aef220aac881ac43085fcbc95022e751d371da714f2606fd32b55a2d6e64` |

Servers 1–3 **abort** if the table sha differs, so all three provably consume
the same artifact. The table is built once and shipped — never regenerated
per server.

---

## Per-server setup

### 0. repo

```bash
cd ~ && if [ -d contrastive_rl/.git ]; then cd contrastive_rl && git fetch origin && git checkout feature/continuous-action-agreement && git pull; else git clone -b feature/continuous-action-agreement https://github.com/tingrui-huang/contrastive_rl.git && cd contrastive_rl; fi && git log --oneline -1
```

### 1. environment (fresh node, no python)

```bash
command -v sudo >/dev/null || { printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo && chmod +x /usr/local/bin/sudo; }
apt-get update && apt-get install -y git python3 python3-venv python3-dev libegl1 libgl1
cd ~/contrastive_rl && PYTHON=python3 bash scripts/failneg_h800_node_setup.sh
```

Builds `~/crlenv` with the validated pin set (`jax[cuda12]==0.10.2`,
`dm-haiku==0.0.16`, `optax==0.2.8`, `mujoco==3.10.0`, `numpy==2.4.4`) and
asserts `jax.default_backend() == 'gpu'`. Do **not** install the 2022
`requirements.txt` — it is the dead Acme pin set.

**Every new shell must activate the venv** (the launchers call `python`):

```bash
source ~/crlenv/bin/activate
```

### 2. bundle — servers 1–3 only

From the local workstation:

```powershell
scp -i "$HOME\.ssh\id_rsa" -P <PORT> "D:\Users\trhua\Research\contrastive_rl\artifacts\four_arm_wc_run\common_wc_bundle.tar.gz" root@<HOST>:/root/contrastive_rl/
```

On the server:

```bash
cd ~/contrastive_rl && sha256sum common_wc_bundle.tar.gz
tar xzf common_wc_bundle.tar.gz && cp -r common_wc_bundle/artifacts/. artifacts/
sha256sum artifacts/static_worstcase_rl/worstcase_table.npz artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz
```

Must print `8e38aef2…`, then `d059db0c…` and `6bec8a52…`.

### 3. GPU check

```bash
cd ~/contrastive_rl && source ~/crlenv/bin/activate && python -c "import jax;print(jax.default_backend());print(jax.devices())"
```

### 4. smoke (optional per server; already passed once)

```bash
cd ~/contrastive_rl && source ~/crlenv/bin/activate && mkdir -p logs && for a in fixed10 fixed50 dpsi blind; do echo "===== $a ====="; python scripts/smoke_four_arm.py --arm $a --updates 2000 || echo "SMOKE_FAILED:$a"; done 2>&1 | tee logs/four_arm_smokes_gpu.log
```

Recorded result: fixed10 **13/13**, fixed50 **13/13**, dpsi **12/12**, blind
**6/6**; realized worst-case rates 0.1002 / 0.5002 / 0.4954; 0.0117 s/update.

---

## Launch

### Server 1 — fixed p_wc = 0.10

```bash
cd ~/contrastive_rl && source ~/crlenv/bin/activate && mkdir -p logs && nohup bash scripts/run_wc_fixedp10_h800_a01_s0_300k.sh > logs/wc_fixedp10_h800_a01_s0_300k.log 2>&1 & echo $!
```

```bash
tail -f ~/contrastive_rl/logs/wc_fixedp10_h800_a01_s0_300k.log
```

```bash
ps -ef | grep wc_fixedp10_h800_a01_s0_300k | grep -v grep
```

Output: `~/contrastive_rl/wc_fixedp10_h800_a01_s0_300k/`

### Server 2 — fixed p_wc = 0.50 (dose-matched control)

```bash
cd ~/contrastive_rl && source ~/crlenv/bin/activate && mkdir -p logs && nohup bash scripts/run_wc_fixedp50_h800_a01_s0_300k.sh > logs/wc_fixedp50_h800_a01_s0_300k.log 2>&1 & echo $!
```

```bash
tail -f ~/contrastive_rl/logs/wc_fixedp50_h800_a01_s0_300k.log
```

```bash
ps -ef | grep wc_fixedp50_h800_a01_s0_300k | grep -v grep
```

Output: `~/contrastive_rl/wc_fixedp50_h800_a01_s0_300k/`

### Server 3 — D_psi surrogate

```bash
cd ~/contrastive_rl && source ~/crlenv/bin/activate && mkdir -p logs && nohup bash scripts/run_wc_dpsi_surrogate_h800_a01_s0_300k.sh > logs/wc_dpsi_surrogate_h800_a01_s0_300k.log 2>&1 & echo $!
```

```bash
tail -f ~/contrastive_rl/logs/wc_dpsi_surrogate_h800_a01_s0_300k.log
```

```bash
ps -ef | grep wc_dpsi_surrogate_h800_a01_s0_300k | grep -v grep
```

Output: `~/contrastive_rl/wc_dpsi_surrogate_h800_a01_s0_300k/`

### Server 4 — pure blind CRL (no bundle needed)

```bash
cd ~/contrastive_rl && source ~/crlenv/bin/activate && mkdir -p logs && nohup bash scripts/run_blind_crl_clean_h800_a00_s0_300k.sh > logs/blind_crl_clean_h800_a00_s0_300k.log 2>&1 & echo $!
```

```bash
tail -f ~/contrastive_rl/logs/blind_crl_clean_h800_a00_s0_300k.log
```

```bash
ps -ef | grep blind_crl_clean_h800_a00_s0_300k | grep -v grep
```

Output: `~/contrastive_rl/blind_crl_clean_h800_a00_s0_300k/`

---

## What each log prints at startup

`ARM_NAME`, `GIT_SHA`, `DATASET_SHA`, `SEED`, `ALPHA_FAIL`, `PROPENSITY_TYPE`,
`WC_TABLE_SHA` + `WC_TABLE_ROWS` + `BANK_SHA` (worst-case arms), GPU backend
and devices, `STEPS`, `OUTPUT DIR`. Then the offline audit G1–G8.

Hard aborts: dataset sha mismatch, table sha mismatch, bank sha mismatch, no
GPU backend, or uncommitted changes under `crl/` or `scripts/`.

During training, every log line carries: step, critic loss, actor loss,
`cat_acc`, `pos` / `neg_ord` / `fail_neg` loss components, and for worst-case
arms `wc_rate`, `nominal`/`wc` counts, `E[rho]`, `E[p_wc]`.

**Expected and normal:** `fail_neg=0.0000`. The critic drives failure-goal
logits below −30 within ~10k steps, so `BCE(logit, 0)` saturates to zero. The
historical α=0.1 run behaved identically for its whole 300k (logits_fail_neg
−31.9 → −87.4). The term is still wired: the loss-decomposition identity and
`fail_bank_size = 16` are asserted every smoke.

Do **not** change coefficients based on these logs.

## Runtime

~0.0117 s/update measured on an RTX 3090. 300k updates plus 30 in-training
evals plus the final/best 200-episode diagnoses ≈ **3–5 h per server**, all
four in parallel.

## Evaluation

Each launcher runs the authoritative protocol itself on **both** `final.pkl`
and `best.pkl`:

```bash
python scripts/diagnose_naive_rockfall.py --v2 --p-active 0.3 --reset-fix --horizon 800 --ckpt <DIR>/<ckpt>.pkl --out-dir <DIR>/eval_h800_resetfix/diag_<ckpt>
```

n=200, fixed eval seed bank (identical mask-pattern counts 48/45/53/54 across
every run). Do not evaluate arms with independently sampled conditions.

Collect afterwards:

```bash
for d in wc_fixedp10_h800_a01_s0_300k wc_fixedp50_h800_a01_s0_300k wc_dpsi_surrogate_h800_a01_s0_300k blind_crl_clean_h800_a00_s0_300k; do for c in final best; do echo "== $d $c"; python -c "import json;j=json.load(open('$d/eval_h800_resetfix/diag_$c/diagnosis.json'));b=j['naive_success_by_mask_pattern'];print('succ %.3f hazard %.3f drop %.3f center %.3f both %.3f'%(j['naive_success'],j['naive_hazard_exposure_rate'],j['naive_drop_rate'],j['naive_center_fraction'],b['both_sides']['success']))"; done; done
```

## Interpretation (decided in advance)

- fixed-rate works but both learned arms fail → integration is functional,
  propensity is the suspect.
- all three worst-case arms beat fail-neg-only → branch is robust to
  propensity choice.
- all three degrade → inspect the `s'_wc → g+` pessimistic-positive semantics
  before blaming propensity.
- `D_psi` bad while fixed arms fine → consistent with the calibration audit.
- blind poor but fail-neg-only strong → evidence for the negative-sampling
  component, **not** the worst-case component.

Server 2 vs 3 is the decisive contrast: equal dose, only state/action
dependence differs.
