# SSH runbook — windy-swamp α sweep (PointMaze failure negatives)

Branch: `feature/pointmaze-causal-transition`
Sweep launcher: `scripts/run_swamp_windy_sweep.sh`
Per-run launcher: `scripts/run_swamp_windy_failneg.py`

Pipeline under test: the original 6000 episodes retained whole (fixed length,
as in the original CRL) → anchors by **scheme C** → ordinary in-batch negatives
unchanged → **plus** failure-state negatives.

| arm | anchor cut | α | what it isolates |
|---|---|---|---|
| `baseline` | off | 0 | the current pipeline (reference) |
| `anchorcut` | on | 0 | scheme C alone |
| `failneg` | on | 0.05 / 0.1 / 0.2 | scheme C + failure negatives |

Default sweep = 5 arms × 3 seeds = **15 runs**.

---

## 0. Sync plan — read this first, it differs from the AntMaze runbook

`git pull` is **not** sufficient. Two things are not in git:

| item | status | action |
|---|---|---|
| `datasets/swamp_windy_teacher_s0.npz` (8.4 MB) | **gitignored** (`datasets/` is in `.gitignore`) | **regenerate on the node** — verified bit-identical from seed 0, sha gate confirms |
| `artifacts/swamp_windy_failure_bank/failure_bank.npz` (6 KB) | tracked | `git pull` |
| `scripts/collect_swamp_windy.py`, `scripts/eval_swamp_windy_deployment.py` | were only on `manski-port-archive`; now committed to this branch | `git pull` |

Do **not** scp the dataset — regeneration is deterministic and self-verifying
against sha `dfdbbaf7…`, which is strictly stronger than copying a file.

---

## 1. Node bootstrap (bare metal: no python, no sudo, you are root)

```bash
head -2 /etc/os-release; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
command -v sudo >/dev/null || { printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo && chmod +x /usr/local/bin/sudo; }
apt-get update && apt-get install -y git python3 python3-venv python3-dev
```

`libegl1 libgl1` are **not needed** for this run — the windy swamp is a pure
numpy point env, no MuJoCo rendering. (Harmless to install if you prefer to
match the AntMaze node exactly.)

```bash
git clone https://github.com/tingrui-huang/contrastive_rl.git ~/contrastive_rl
cd ~/contrastive_rl
git checkout feature/pointmaze-causal-transition
git rev-parse HEAD
PYTHON=python3 bash scripts/failneg_h800_node_setup.sh 2>&1 | tail -20
```

That script builds `~/crlenv` with the validated set (`jax[cuda12]==0.10.2`,
`dm-haiku==0.0.16`, `optax==0.2.8`, `mujoco==3.10.0`, `numpy==2.4.4`) and
**asserts `jax.default_backend() == 'gpu'`**. Do NOT install
`requirements.txt` — it is the dead 2022 Acme pin set.

Note it installs **no scipy**; nothing in this sweep requires it
(`verify_anchor_cut.py` falls back to a numpy scan).

### Every new shell

```bash
cd ~/contrastive_rl && source ~/crlenv/bin/activate
python -c "import jax;print(jax.default_backend(), jax.devices())"
```

Expect `gpu [CudaDevice(id=0)]`. The launchers call `python`, which only
exists after activation. If it reports `cpu`, paste the install tail — do not
downgrade the driver.

---

## 2. Regenerate the dataset (~4 min, one time)

```bash
python -m scripts.collect_swamp_windy --episodes 6000 --random_frac 0.2 \
    --force_safe_prob 0.05 --teacher_noise 0.15 --seed 0 \
    --out datasets/swamp_windy_teacher_s0.npz
sha256sum datasets/swamp_windy_teacher_s0.npz
# MUST be dfdbbaf7b6a62754f8c865257cea4f3d271ba524f4510c8459ca6fc901e1bfee
```

The bank (`artifacts/swamp_windy_failure_bank/failure_bank.npz`, sha
`71922996…`, 514×2) comes with the repo; the sweep rebuilds it automatically
if absent and refuses to start on any sha mismatch.

---

## 3. Gate check — no training, ~2 min

```bash
bash scripts/run_swamp_windy_sweep.sh check
```

Runs the 9-gate mechanism verification plus a provenance gate per arm. The
gates that matter:

* `V1_default_rng_identical` — with the anchor cut off the buffer consumes the
  RNG exactly as before (the AntMaze line is unaffected by these changes);
* `V3_cut_and_full_future` — anchors never exceed the cut, **and** goals still
  come from at/past it (proof the future window was not truncated);
* `V8a_loss_decomposition` — `pos + (1−α)·ord + α·fail == critic_loss`;
* `V8b_alpha0_identical` — α=0 with a bank present is byte-identical to baseline.

`V7` is **informational, not a gate**: it reports the bank-vs-positive overlap
(≈6.7% at r=0.02, and 71.7% for anchors inside dead episodes). This is expected
— the dataset is deliberately not split — and is the known positive/negative
tension. It is printed on every run so it stays visible.

---

## 4. Smoke — 2 000 steps per arm

```bash
SEEDS="0" ALPHAS="0.1" JOBS=3 bash scripts/run_swamp_windy_sweep.sh smoke
```

Confirms concurrent launch, GPU consumption, bank loading, and checkpointing.
Scratch dirs end in `_smoke/` and are ignorable.

---

## 5. Full sweep

Always launch **detached** — this host resets long-held SSH connections:

```bash
mkdir -p logs
setsid nohup env SEEDS="0 1 2" ALPHAS="0.05 0.1 0.2" \
  bash -c "source ~/crlenv/bin/activate && exec bash scripts/run_swamp_windy_sweep.sh run" \
  > logs/sweep.log 2>&1 < /dev/null &
```

**Runs go sequentially, and that is deliberate.** Measured on this 3090:

| setting | steps/s | 150k run |
|---|---|---|
| 1 process, G=1 | 114 | 21.9 min |
| 1 process, **G=10** | **253** | **9.9 min** |
| 5 concurrent, G=1 | ~22 each (~110 total) | — |

The model is tiny, so a step is dominated by dispatch and host sync, not GPU
compute — utilisation sits near 4%. Extra processes just contend for the single
CUDA context, so `JOBS=5` was *slower in aggregate* than one process. The real
lever is `num_sgd_steps_per_step` (`SGD_STEPS_PER_STEP = 10` in the launcher),
which batches 10 updates into one `jax.lax.scan` dispatch. It does **not**
change the math: the buffer is frozen and its RNG is independent of the learner,
so the 10 batches are exactly the ones G=1 would have drawn and `scan` applies
the same updates in the same order. It must divide `max_episode_steps` (50) or
`train.py` silently drops updates.

**Expected wall-clock**: ~10 min per run → **5 arms × 3 seeds ≈ 2.5 h**;
one seed ≈ 50 min. At $0.12/hr that is well under a dollar. Deployment eval is
seconds per run (pure numpy point env).

The sweep refuses to start on a CPU backend (override with `FORCE_CPU=1`).

---

## 6. Monitoring

```bash
tail -f logs/swamp_windy_sweep.log
ls logs/swamp_windy_sweep/                       # one log per run
nvidia-smi                                       # all processes on one GPU
grep -h "\[step" logs/swamp_windy_sweep/*.log | tail -20
grep -c "DONE" logs/swamp_windy_sweep/*.log
```

Per-run provenance is written to `<run_dir>/arm_provenance.json` (git commit,
dataset sha, bank sha, α, anchor-cut mode, batch size).

---

## 7. Results

The sweep prints a table at the end and writes one report per run:

```
artifacts/swamp_windy_<arm>_s<seed>/deployment_report.json
```

The pre-registered metric is **`worst_case` = success under `all_active`**
(bits frozen `[1,1,1]`, corridor entry = instant death), read against `entry`
(shortcut-taking rate). The reference to beat is the prior naive result in
`artifacts/windy_manski_s0_deployment/`: `CONFOUNDED_SHORTCUT_BIAS`,
entry 1.0, worst_case 0.0, while the always-safe policy scores 1.0 everywhere.

Note that prior run's hyperparameters are not recorded anywhere in the repo, so
**compare against the `baseline` arm from this sweep**, not against that number.

---

## 8. Bring it back

```bash
# from the workstation
scp -r <node>:~/contrastive_rl/artifacts/swamp_windy_* \
       D:/Users/trhua/Research/contrastive_rl/artifacts/
scp -r <node>:~/contrastive_rl/logs/swamp_windy_sweep \
       D:/Users/trhua/Research/contrastive_rl/logs/
```

Checkpoints (`swamp_windy_*_s*/`) are only needed if you want to re-evaluate
locally; the deployment reports carry the results.
