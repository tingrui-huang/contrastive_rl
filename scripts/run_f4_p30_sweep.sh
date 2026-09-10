#!/usr/bin/env bash
# F4 p=0.3: alpha 0, 0.1, 0.3 x learner seed 0, 1; 150k updates each.
# Data and composed bank must already exist. No collection happens here.
# Usage: PY=/path/to/python bash scripts/run_f4_p30_sweep.sh [run|check|smoke]
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-run}"
PY="${PY:-python}"
SEEDS="${SEEDS:-0 1}"
ALPHAS="${ALPHAS:-0.1 0.3}"
JOBS="${JOBS:-1}"
EPISODES="${EPISODES:-100}"
LOGDIR="${LOGDIR:-logs/f4_p30_sweep}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
RUN_ROOT="${RUN_ROOT:-runs/f4_p30_sweep/$RUN_ID}"
EVAL_ROOT="${EVAL_ROOT:-artifacts/f4_p30_sweep/$RUN_ID}"
DATASET="datasets/swamp_windy_f4_merged_s0.npz"
BANK="${BANK:-artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz}"
ENV_NAME="point_two_route_swamp_windy_f4_v0"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGDIR/$RUN_ID" "$RUN_ROOT" "$EVAL_ROOT"
STATUS="$LOGDIR/$RUN_ID/status.tsv"
CURRENT="preflight"
write_status() {
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$CURRENT" "$1" | tee -a "$STATUS"
}
on_error() {
  local rc=$?
  trap - ERR
  write_status "FAILED exit=$rc line=$1"
  exit "$rc"
}
trap 'on_error "$LINENO"' ERR
trap 'write_status "INTERRUPTED signal=TERM"; exit 143' TERM
trap 'write_status "INTERRUPTED signal=INT"; exit 130' INT
die() { write_status "FAILED $*"; exit 1; }

[[ "$MODE" == run || "$MODE" == check || "$MODE" == smoke ]] || die "mode must be run, check, or smoke"
[[ "$JOBS" == 1 ]] || die "this sweep runs sequentially; set JOBS=1"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "EPISODES must be a positive integer"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID must contain only letters, digits, dot, underscore, or hyphen"
exec 9>"$LOGDIR/.sweep.lock"
flock -n 9 || die "another F4 p=0.3 sweep holds $LOGDIR/.sweep.lock"
[[ -f "$DATASET" ]] || die "missing dataset: $DATASET"
[[ -f "$BANK" ]] || die "missing composed bank: $BANK"
read -r -a SEED_LIST <<< "$SEEDS"
read -r -a ALPHA_LIST <<< "$ALPHAS"
[[ ${#SEED_LIST[@]} -gt 0 && ${#ALPHA_LIST[@]} -gt 0 ]] || die "SEEDS and ALPHAS cannot be empty"
for seed in "${SEED_LIST[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || die "invalid seed: $seed"
done
write_status "START mode=$MODE jobs=$JOBS run_root=$RUN_ROOT"

# Reuse the launcher's provenance gate instead of pinning obsolete p=0.1
# artifact hashes. Record the hashes of the actual p=0.3 arrays and git HEAD.
"$PY" - "$DATASET" "$BANK" "$RUN_ROOT" "$MODE" "$SEEDS" "$ALPHAS" "$EPISODES" <<'PYEOF' 2>&1 | tee "$LOGDIR/$RUN_ID/preflight.log"
import collections
import json
import pathlib
import subprocess
import sys

import jax
import numpy as np
from scripts import run_swamp_windy_z_failneg as launcher

dataset, bank, run_root, mode, seeds, alphas, episodes = sys.argv[1:]
seed_values = [int(s) for s in seeds.split()]
alpha_values = [float(a) for a in alphas.split()]
assert len(set(seed_values)) == len(seed_values), 'duplicate learner seeds'
assert len(set(alpha_values)) == len(alpha_values), 'duplicate alphas'
assert all(np.isfinite(a) and 0 < a < 1 for a in alpha_values), 'ALPHAS must be in (0, 1); baseline is always included'
devices = jax.devices()
assert jax.default_backend() == 'gpu', f'GPU required; found {devices}'
assert all(device.platform == 'gpu' for device in devices), devices
assert float((jax.numpy.ones((2, 2)) @ jax.numpy.ones((2, 2))).block_until_ready()[0, 0]) == 2
print('GPU execution passed:', devices, flush=True)
launcher.select_version('f4')
assert dataset == launcher.DATASET, 'dataset must match launcher registry'
launcher.select_bank(bank)
prov = launcher.gate('zfail', seed_values[0])
assert np.isclose(prov['per_cell_swamp_prob'], 0.3, rtol=0, atol=1e-8), 'p=0.3 required'
with np.load(dataset, allow_pickle=False) as data:
    assert data['obs'].ndim == 3 and data['obs'].shape[2] == 16, 'F4 requires observation width 16'
with np.load(bank, allow_pickle=False) as data:
    goals = data['goals']
    meta = json.loads(str(data['meta']))
    assert goals.ndim == 2 and goals.shape[1] == 8 and 0 < len(goals) <= 256, 'invalid F4 bank shape'
    assert np.isfinite(goals).all(), 'nonfinite bank goals'
    frames = goals.reshape(len(goals), 4, 2)
    assert np.array_equal(frames, np.broadcast_to(frames[:, :1], frames.shape)), 'bank must contain frozen frame stacks'
    mix = collections.Counter(data['behaviour_class'].tolist())
    assert meta.get('composed') is True, 'composed bank required'
    assert set(mix) == {'random', 'deliberate'}, f'unexpected bank classes: {mix}'
    assert abs(mix['random'] / len(goals) - 0.6) <= 1 / len(goals), f'expected r60d40 bank: {mix}'
    print('Composed bank:', dict(mix), flush=True)
for seed in seed_values:
    for alpha in alpha_values:
        launcher.select_alpha(alpha)
        launcher.config_diff(seed)
prov.update({
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
    'mode': mode, 'learner_seeds': seed_values, 'alphas': [0.0] + alpha_values,
    'steps_per_run': launcher.STEPS if mode == 'run' else (2000 if mode == 'smoke' else 0),
    'eval_episodes_per_condition': int(episodes), 'jax_devices': [str(d) for d in devices],
})
path = pathlib.Path(run_root) / 'sweep_provenance.json'
assert not path.exists(), f'sweep already exists: {path}; choose a fresh RUN_ID'
path.write_text(json.dumps(prov, indent=2) + '\n')
print('Wrote', path, flush=True)
PYEOF
write_status "PREFLIGHT_PASSED"

run_one() {
  local arm="$1" alpha="$2" seed="$3"
  local alpha_label="${alpha//./p}"
  local tag="alpha${alpha_label}_seed${seed}"
  local run_dir="$RUN_ROOT/$tag"
  local train_log="$LOGDIR/$RUN_ID/${tag}_train.log"
  local eval_log="$LOGDIR/$RUN_ID/${tag}_eval.log"
  local -a cmd=("$PY" scripts/run_swamp_windy_z_failneg.py --version f4 --arm "$arm" --seed "$seed" --ckpt-dir "$run_dir")
  if [[ "$arm" == zfail ]]; then
    cmd+=(--alpha "$alpha" --bank "$BANK")
  fi
  CURRENT="$tag"
  if [[ "$MODE" == check ]]; then
    cmd+=(--check-only)
    write_status "CHECK_STARTED log=$train_log"
    "${cmd[@]}" > "$train_log" 2>&1
    write_status "CHECK_PASSED"
    return
  fi
  [[ ! -e "$run_dir" ]] || die "run directory exists: $run_dir"
  if [[ "$MODE" == smoke ]]; then cmd+=(--smoke); else cmd+=(--run); fi
  write_status "TRAIN_STARTED log=$train_log"
  "${cmd[@]}" > "$train_log" 2>&1
  [[ -f "$run_dir/final.pkl" ]] || die "training exited without final.pkl: $run_dir"
  write_status "TRAIN_FINISHED"
  write_status "EVAL_STARTED episodes=$EPISODES log=$eval_log"
  "$PY" -m scripts.eval_swamp_windy_z_deployment \
    --ckpt "$run_dir/final.pkl" --env "$ENV_NAME" \
    --out "$EVAL_ROOT/$tag" --episodes "$EPISODES" --seed 123 \
    > "$eval_log" 2>&1
  [[ -f "$EVAL_ROOT/$tag/deployment_report.json" ]] || die "evaluation report missing: $tag"
  write_status "COMPLETE report=$EVAL_ROOT/$tag/deployment_report.json"
}

for seed in "${SEED_LIST[@]}"; do
  run_one zbase 0 "$seed"
  for alpha in "${ALPHA_LIST[@]}"; do
    run_one zfail "$alpha" "$seed"
  done
done
CURRENT="sweep"
write_status "COMPLETE mode=$MODE total=$((${#SEED_LIST[@]} * (${#ALPHA_LIST[@]} + 1)))"
