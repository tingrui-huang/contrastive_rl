# SSH runbook — settled-bank α=0.1 production run

Run id: `failneg_settledbank_p30_h800_resetfix_a01_s0_300k`
Launcher: `scripts/run_failneg_settledbank_h800.sh` (same trainer/recipe/eval
as `scripts/run_failneg_h800.sh`; only the failure bank differs, and a hard
provenance gate refuses the legacy bank, a wrong clean-dataset sha, a bank
size ≠ 16, or α ≠ 0.1).

**Sync plan: `git pull` is sufficient.** Everything the run needs is tracked
in git on `feature/continuous-action-agreement`: the settled bank
(`artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz`, 12 KB),
its manifest, the clean dataset (already in the repo since Part 1 — do not
re-upload; its sha is verified below), the death-settle env patch, the
trainer, and this launcher. Nothing must be copied manually.

## 1. Sync the repo

```bash
cd ~/contrastive_rl
git fetch
git checkout feature/continuous-action-agreement
git pull
git rev-parse HEAD
test -f scripts/run_failneg_settledbank_h800.sh && echo LAUNCHER PRESENT
```

## 2. Activate the environment and verify the GPU

The node env is the venv from `scripts/failneg_h800_node_setup.sh`
(adjust the path if yours differs):

```bash
source ~/crlenv/bin/activate
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Expected: `gpu [CudaDevice(id=0)]` (any CUDA device). The launcher aborts if
the backend is not GPU.

## 3. Verify the authoritative hashes (standalone)

```bash
sha256sum artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz
# must be: 6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852
sha256sum artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz
# must be: 8c50202403317e8adc67670be716fffe5c3b3cb891017bfbe6eb070acd61d0ce
# (legacy bank 8d35b76a... is REFUSED by the launcher's gate)
```

The launcher re-verifies both hashes (plus bank size 16×29, α=0.1, bank
manifest checks, and the 40-death held-out list) before every start.

## 4. Smoke test (~2 min, no checkpoints, production dir untouched)

```bash
bash scripts/run_failneg_settledbank_h800.sh smoke
```

Expected output ends with a line like:

```
SMOKE OK | backend gpu | fail_bank_size 16.0 | logits_fail_neg ... | L = pos + 0.9*ord + 0.1*fail verified | ... updates/s
```

This confirms: settled bank reaches the failure-negative loss, α=0.1, exact
loss decomposition holds, GPU in use. Scratch dir:
`failneg_settledbank_p30_h800_resetfix_a01_s0_300k_smoke/` (ignorable).

## 5. Full production launch (α=0.1 only; no sweep, no α=0 rerun)

```bash
mkdir -p logs
nohup bash scripts/run_failneg_settledbank_h800.sh run \
  > logs/failneg_settledbank_a01.log 2>&1 &
echo $!
```

Note the printed PID. At the previous runs' ~32 updates/s this is roughly
2.5–3 h of training plus ~30–40 min for the two post-train diagnosis evals
(the launcher runs them automatically, same protocol as A/B: `--v2
--p-active 0.30 --reset-fix --horizon 800`, final + best, n=200).

After a disconnect/interruption:

```bash
nohup bash scripts/run_failneg_settledbank_h800.sh run --resume \
  >> logs/failneg_settledbank_a01.log 2>&1 &
```

## 6. Monitoring

```bash
tail -f logs/failneg_settledbank_a01.log
```

```bash
ps -p <PID> -o pid,etime,cmd          # process alive?
```

```bash
nvidia-smi                             # GPU utilization
```

```bash
ls -lt failneg_settledbank_p30_h800_resetfix_a01_s0_300k/ | head   # checkpoints
grep EVAL logs/failneg_settledbank_a01.log | tail -5               # eval curve
```

Completion:

```bash
grep -c "DONE failneg_settledbank" logs/failneg_settledbank_a01.log   # 1 = done
```

## 7. Expected outputs

```
failneg_settledbank_p30_h800_resetfix_a01_s0_300k/
├── init.pkl early.pkl mid.pkl final.pkl best.pkl latest.pkl
├── metrics.json          # per-eval metrics incl. logits_fail_neg trajectory
├── train.log  diagnose.log  tb/
└── eval_h800_resetfix/
    ├── diag_best/diagnosis.json
    └── diag_final/diagnosis.json
```

## 8. After the run: bring it back and compare

Copy the whole run directory to the workstation repo root (same place the
A/B runs live), e.g. from the workstation:

```bash
scp -r <node>:~/contrastive_rl/failneg_settledbank_p30_h800_resetfix_a01_s0_300k D:/Users/trhua/Research/contrastive_rl/
```

then produce the A/B/C table (historical protocol only — the optional
`death_settle_substeps=80` consistency re-eval is a separate later pass and
must not be mixed into this table):

```bash
python scripts/compare_settled_bank_run.py
```
