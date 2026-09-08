#!/usr/bin/env bash
# Alpha sweep for the V6 two-rockfall failure-negative line. GPU node.
#
#   q_alpha(g) = (1 - alpha) q_normal(g) + alpha q_fail(g)
#
# Arms (each x N seeds):
#   alpha 0.0                    the vanilla control -- no bank is loaded at all
#   alpha 0.1 0.2 0.3 0.4 0.5    the composed bank in the NEGATIVE distribution
#
# The bank is 60% random/noisy + 40% deliberate, and the deliberate half is
# exactly 50/50 across the two hazard zones (scripts/make_v6_failure_bank.py).
# Those fractions live INSIDE the bank; alpha is the weight the bank carries in
# the negative term. Independent knobs, both set explicitly.
#
# Every arm calls the SAME launcher, which reuses
# scripts/train_rockfall_clock_v6_baseline.py for the config, the benchmark
# contract and the vanilla-CRL guard. No training script is duplicated per
# alpha, and the guard proves each arm differs from the vanilla baseline in
# fail_neg_alpha (plus the bank path it needs) and nothing else.
#
# Usage:
#   bash scripts/run_v6_failneg_sweep.sh check     # gates only, no training
#   bash scripts/run_v6_failneg_sweep.sh smoke     # tiny run, every arm
#   bash scripts/run_v6_failneg_sweep.sh run       # the sweep
#   SEEDS="0 1 2" bash scripts/run_v6_failneg_sweep.sh run
set -uo pipefail

MODE="${1:-check}"
SEEDS="${SEEDS:-0}"
ALPHAS="${ALPHAS:-0.0 0.1 0.2 0.3 0.4 0.5}"
STEPS="${STEPS:-100000}"
SMOKE_STEPS="${SMOKE_STEPS:-800}"
BANK="${BANK:-artifacts/v6_failneg/bank/v6_failure_bank_r60_z20_z20.npz}"
ENV_NAME="${ENV_NAME:-offline_antmaze_rockfall_clock_v6_gxy}"
LOGDIR="${LOGDIR:-logs/v6_failneg_sweep}"
PY="${PY:-python}"
EPISODES="${EPISODES:-300}"
RUN_ROOT="artifacts/v6_failneg/runs"
# Pin the artifact this sweep was defined against. The preflight prints it;
# fill it in before a production sweep so a rebuilt bank cannot silently change
# what is being compared.
BANK_SHA="${BANK_SHA:-}"

JOBS="${JOBS:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "$LOGDIR"
banner() { echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

banner "PREFLIGHT"
if [ ! -f "$BANK" ]; then
  echo "MISSING $BANK"
  echo "build it with:"
  echo "  $PY scripts/collect_v6_failure_candidates.py --all --episodes 300"
  echo "  $PY scripts/make_v6_failure_bank.py --n-bank 250"
  exit 1
fi
got=$($PY -c "
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
        h.update(block)
print(h.hexdigest())
" "$BANK")
if [ -n "$BANK_SHA" ] && [ "$got" != "$BANK_SHA" ]; then
  echo "bank sha mismatch -- this is not the bank the sweep was defined against"
  echo "  expected $BANK_SHA"; echo "  found    $got"; exit 1
fi
echo "bank     : $BANK"
echo "bank sha : $got"
echo "pin BANK_SHA=$got before a production sweep"
$PY -c "
import json, sys
import numpy as np
b = np.load(sys.argv[1], allow_pickle=False)
m = json.loads(str(b['meta']))
print('compose  : random %d / delib Z1 %d / Z2 %d  of %d'
      % (m['n_random'], m['n_deliberate_zone1'], m['n_deliberate_zone2'], m['n_bank']))
print('settle   : %d substeps' % m['settled_state_extraction']['death_settle_substeps'])
print('p_active : %g / %g' % (m['p_active_1'], m['p_active_2']))
assert m['n_random'] / m['n_bank'] == 0.60
assert m['n_deliberate_zone1'] == m['n_deliberate_zone2']
assert (m['n_deliberate_zone1'] + m['n_deliberate_zone2']) / m['n_bank'] == 0.40
print('composition assertions: PASS')
" "$BANK" || exit 1

backend=$($PY -c "import jax;print(jax.default_backend())" 2>/dev/null)
echo "jax      : $backend"
$PY -c "import jax;print('devices  :',jax.devices())"
if [ "$MODE" = "run" ] && [ "$backend" != "gpu" ] && [ "$backend" != "cuda" ]; then
  echo "REFUSING a full sweep on backend '$backend' (batch 1024 through a"
  echo "1024x1024 MLP on CPU is hours per arm). Set FORCE_CPU=1 to override."
  [ "${FORCE_CPU:-0}" = "1" ] || exit 1
fi

ARMS=()
for s in $SEEDS; do for a in $ALPHAS; do ARMS+=("$a|$s"); done; done

banner "SWEEP PLAN  (${#ARMS[@]} runs, mode=$MODE, jobs=$JOBS, steps=$STEPS)"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r alpha seed <<< "$spec"
  printf '  alpha=%-5s seed=%s\n' "$alpha" "$seed"
done

argv_for() {
  local alpha="$1" seed="$2" steps="$3"
  echo "--alpha $alpha --seed $seed --steps $steps --bank $BANK --env-name $ENV_NAME"
}

if [ "$MODE" = "check" ]; then
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r alpha seed <<< "$spec"
    # shellcheck disable=SC2046
    $PY scripts/run_v6_failneg.py $(argv_for "$alpha" "$seed" "$STEPS") \
        --check-only > /dev/null || { echo "GATE FAILED: alpha=$alpha seed=$seed"; exit 1; }
  done
  echo; echo "ALL ${#ARMS[@]} GATES PASS (no training performed)"
  exit 0
fi

FLAG=""; STEPS_USED="$STEPS"
if [ "$MODE" = "smoke" ]; then FLAG="--smoke --smoke-steps $SMOKE_STEPS"; STEPS_USED="$SMOKE_STEPS"; fi

banner "LAUNCHING"
pids=(); names=()
for spec in "${ARMS[@]}"; do
  IFS='|' read -r alpha seed <<< "$spec"
  tag="alpha${alpha}_s${seed}"
  log="$LOGDIR/${tag}.log"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  # shellcheck disable=SC2046
  $PY scripts/run_v6_failneg.py $(argv_for "$alpha" "$seed" "$STEPS_USED") \
      $FLAG > "$log" 2>&1 &
  pids+=($!); names+=("$tag")
  echo "  [$!] $tag -> $log"
  sleep 2
done

echo; echo "waiting for ${#pids[@]} runs..."
fail=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "  OK   ${names[$k]}"
  else echo "  FAIL ${names[$k]}  (see $LOGDIR/${names[$k]}.log)"; fail=$((fail+1)); fi
done
banner "TRAINING COMPLETE  ($((${#pids[@]}-fail))/${#pids[@]} ok)"
[ "$MODE" = "smoke" ] && exit $fail

banner "EVALUATION  (n=$EPISODES, identical protocol for every alpha)"
for d in "$RUN_ROOT"/v6fn_*; do
  [ -f "$d/final.pkl" ] || { echo "  skip $d (no final.pkl)"; continue; }
  $PY scripts/eval_rockfall_clock_v6_baseline.py --ckpt "$d/final.pkl" \
      --n "$EPISODES" > "$LOGDIR/$(basename "$d")_eval.log" 2>&1 \
    && echo "  evaluated $(basename "$d")" \
    || echo "  EVAL FAILED $(basename "$d") (see $LOGDIR/)"
done

banner "RESULTS"
$PY - "$RUN_ROOT" <<'PYEOF'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(root, 'v6fn_*'))):
    arm_path = os.path.join(d, 'failneg_arm.json')
    if not os.path.exists(arm_path):
        continue
    arm = json.load(open(arm_path))
    ev = sorted(glob.glob(os.path.join(d, 'eval_*.json')))
    s = json.load(open(ev[-1])).get('summary', {}) if ev else {}
    o = s.get('overall', s)
    bl = s.get('by_latent', {})
    rows.append((os.path.basename(d), arm.get('fail_neg_alpha'),
                 arm.get('seed'), o.get('success'), o.get('failure'),
                 o.get('timeout'),
                 {k: (v or {}).get('success') for k, v in bl.items()}))
if rows:
    print('%-44s%7s%6s%9s%9s%9s  by-latent success'
          % ('run', 'alpha', 'seed', 'success', 'failure', 'timeout'))
    print('-' * 118)
    f = lambda v: ('%.3f' % v) if isinstance(v, (int, float)) else '   -  '
    for r in rows:
        print('%-44s%7s%6s%9s%9s%9s  %s'
              % (r[0], f(r[1]), r[2], f(r[3]), f(r[4]), f(r[5]), r[6]))
    print()
    print('success/failure alone are not the readout: an agent that stops')
    print('moving has failure 0 too. Read them with timeout and the route /')
    print('waiting columns in each eval json.')
else:
    print('no runs found under', root)
PYEOF
echo
echo "logs: $LOGDIR"
exit $fail
