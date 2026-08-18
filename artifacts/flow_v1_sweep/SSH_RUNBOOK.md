# SSH runbook — V1 Flow sweep (8 runs, two servers)

Code + both datasets are committed on `feature/continuous-action-agreement`
at **`382b87a`**, so the sync is a plain `git pull` — nothing to copy by hand.

| | |
|---|---|
| clean dataset | `artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz` |
| clean sha256 | `6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852` |
| bad-demo dataset | `artifacts/bad_demo_fixed/bad_demo_blind_p30_h800_settle80.npz` (33 MB) |
| bad-demo sha256 | `cfa948fe491e1461e43ba82f0d0ab335cf8b11a4c849d098f3fc56fd246ff4f0` |
| normalization | `artifacts/flow_v0_clean/norm_stats.npz` (frozen V0 stats) |
| split | `artifacts/flow_v0_clean/split_manifest.json` (frozen V0 episodes) |
| bad-demo env seed / dataset seed | `82500019` / `82990013` |

`scripts/train_flow_v1.py` prints and hard-checks provenance before every run
(git commit, run id, family, β, both dataset shas, normalization source, seed,
step budget, JAX device) and **aborts** on any sha mismatch or if the bad-demo
manifest is not `death_settle_substeps=80`.

## 1. Sync (both servers)

```bash
cd ~/contrastive_rl && git fetch && git checkout feature/continuous-action-agreement && git pull && git rev-parse HEAD
```

```bash
source ~/crlenv/bin/activate && python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## 2. Verify the two dataset hashes (both servers)

```bash
sha256sum artifacts/rockfall_v2_p30_h800_resetfix/failure_split/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz artifacts/bad_demo_fixed/bad_demo_blind_p30_h800_settle80.npz
```

Expect `6bec8a52…` and `cfa948fe…` respectively.

## 3. Launch

**Server A** — β ∈ {0.05, 0.10}, both families (4 runs, sequential):

```bash
mkdir -p logs && nohup bash scripts/run_flow_v1_server_a.sh > logs/flow_v1_serverA_driver.log 2>&1 & echo $!
```

**Server B** — β ∈ {0.15, 0.20}, both families (4 runs, sequential):

```bash
mkdir -p logs && nohup bash scripts/run_flow_v1_server_b.sh > logs/flow_v1_serverB_driver.log 2>&1 & echo $!
```

Each run is 20k updates at batch 1024 (~6 min on CPU, less on GPU), so a
server finishes its four runs in well under an hour.

Single run, if you need to redo one:

```bash
python scripts/train_flow_v1.py --family SA --beta 0.15
```

## 4. Monitor

```bash
tail -f logs/flow_v1_serverA_driver.log
```

```bash
ls -d artifacts/flow_v1_sweep/V1-*/ && grep -c "^final:" logs/flow_v1_server*.log
```

## 5. Expected outputs

```
artifacts/flow_v1_sweep/
├── V1-S-b005/   flow_v1.pkl  train_log.json
├── V1-SA-b005/  …
├── V1-S-b010/   V1-SA-b010/
├── V1-S-b015/   V1-SA-b015/
└── V1-S-b020/   V1-SA-b020/
```

## 6. Aggregate (either machine, after all 8 exist)

```bash
python scripts/eval_flow_v1_dev16.py
```

Writes `artifacts/flow_v1_dev16/{dev16_summary.json,dev16_table.csv,v1_dev16.png}`
and applies the pre-frozen gates: median `d_fatal@256 ≤ 3.17`,
`FatalCoverage@256 ≥ 8/16`, and ≤20% degradation vs the family baseline
(V1-S vs V0, V1-SA vs V0.5). Runs failing at K=256 but meeting coverage at
K=2048 are reported as **WEAK**, not passing.

The 39 sealed same-anchor cases and the 40 fresh death stream are not touched
by any command here.
