#!/usr/bin/env bash
# Alpha sweep for the FRAME-STACKED (f4) failure-negative line, for a GPU node.
#
# Arms (each x N seeds):
#   zbase            alpha 0            -- the reference, no bank at all
#   zfail alpha in   {0.1 .. 0.5}       -- composed bank, mixture weight swept
#
# The bank is the COMPOSED one: 60% random-noise deaths + 40% expert
# "deliberate" deaths (moved into a cell whose bit was ACTIVE in the very bits
# the teacher reads before acting, with a clear alternative available). Built
# by scripts/make_swamp_f4_failure_bank.py --compose. Its class selection reads
# teacher_mode and swamp_bits -- god-view fields -- which is recorded in the
# bank manifest; the stored goal vectors themselves stay learner-visible
# frozen frame stacks.
#
# f4 has NO obs normalisation (every coordinate is already in maze units), so
# the z-scaling path is inert here.
#
# Usage:
#   bash scripts/run_f4_failneg_sweep.sh check      # gates only, no training
#   bash scripts/run_f4_failneg_sweep.sh smoke      # 2k-step smoke, every arm
#   bash scripts/run_f4_failneg_sweep.sh run        # full sweep
#   SEEDS="0 1 2" bash scripts/run_f4_failneg_sweep.sh run
set -uo pipefail

MODE="${1:-check}"
SEEDS="${SEEDS:-0}"
ALPHAS="${ALPHAS:-0.1 0.2 0.3 0.4 0.5}"
WITH_BASELINE="${WITH_BASELINE:-1}"
LOGDIR="${LOGDIR:-logs/f4_failneg_sweep}"
PY="${PY:-python}"
EPISODES="${EPISODES:-100}"

DATASET="datasets/swamp_windy_f4_merged_s0.npz"
DATASET_CONTENT_SHA="22185598cbcc4fd12b461af67cfddcf00b471be9acd6c6ea520c6e88ceab118e"
BANK="artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz"
BANK_CONTENT_SHA="688a2956697d62c0a2c8236aefc9fa7528ea6ac7ed930c8c07395769e2863d83"
COMPOSE="random=0.6,deliberate=0.4"
ENV_NAME="point_two_route_swamp_windy_f4_v0"

# SEQUENTIAL BY DEFAULT, and this is a measured choice, not caution: on a 3090
# one process runs this workload at ~253 steps/s while five concurrent ones
# managed ~22 steps/s each (~110 aggregate). The model is dispatch-bound --
# GPU utilisation sits near 4% -- so extra processes only fight over the one
# CUDA context and the whole sweep gets ~2.3x SLOWER. Raise JOBS only if you
# have measured otherwise on your node.
JOBS="${JOBS:-1}"
# Many small processes on one GPU: JAX would otherwise preallocate 75% each.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "$LOGDIR"
banner() { echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

content_sha() {  # sha256 over the npz ARRAY CONTENTS (zip metadata ignored)
  $PY -c "
import hashlib, sys, numpy as np
h = hashlib.sha256()
with np.load(sys.argv[1], allow_pickle=False) as d:
    for k in sorted(d.files):
        a = d[k]
        h.update(k.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
print(h.hexdigest())
" "$1"
}

# ----------------------------------------------------------- preflight gate
banner "PREFLIGHT"
if [ ! -f "$DATASET" ]; then
  echo "MISSING $DATASET"
  echo "datasets/ is gitignored -- regenerate it on this node with:"
  echo "  $PY -m scripts.collect_swamp_windy_f4 --mode teacher --episodes 6000 \\"
  echo "      --random_frac 0.2 --force_safe_prob 0.05 --teacher_noise 0.15 --seed 0 \\"
  echo "      --out datasets/swamp_windy_f4_teacher_s0.npz"
  echo "  $PY -m scripts.collect_swamp_windy_f4 --mode baddemo --episodes 600 --seed 0 \\"
  echo "      --out datasets/swamp_windy_f4_baddemo_s0.npz"
  echo "  $PY -m scripts.merge_swamp_windy_baddemo \\"
  echo "      --main datasets/swamp_windy_f4_teacher_s0.npz \\"
  echo "      --bad  datasets/swamp_windy_f4_baddemo_s0.npz --out $DATASET"
  exit 1
fi
got=$(content_sha "$DATASET")
[ "$got" = "$DATASET_CONTENT_SHA" ] || { echo "dataset CONTENT mismatch"; echo "  expected $DATASET_CONTENT_SHA"; echo "  found    $got"; exit 1; }
echo "dataset OK  content $got"

if [ ! -f "$BANK" ]; then
  echo "building the composed failure bank ($COMPOSE)..."
  $PY scripts/make_swamp_f4_failure_bank.py --compose "$COMPOSE" \
      --max-bank 256 --seed 0 --out-name "$(basename "$BANK")" || exit 1
fi
got=$(content_sha "$BANK")
[ "$got" = "$BANK_CONTENT_SHA" ] || { echo "bank CONTENT mismatch -- the composed bank is not the one this sweep was defined against"; echo "  expected $BANK_CONTENT_SHA"; echo "  found    $got"; exit 1; }
echo "bank OK     content $got"
$PY -c "
import numpy as np, collections, sys
b = np.load(sys.argv[1], allow_pickle=False)
g, c = b['goals'], b['behaviour_class']
fr = g.reshape(len(g), 4, 2)
assert np.abs(fr - fr[:, :1]).max() == 0.0, 'a bank entry is not a frozen stack'
print('bank mix    : %s  (n=%d, goal dim %d)'
      % (dict(collections.Counter(c.tolist())), len(g), g.shape[1]))
" "$BANK" || exit 1

backend=$($PY -c "import jax;print(jax.default_backend())" 2>/dev/null)
echo "jax backend : $backend"
$PY -c "import jax;print('devices    :',jax.devices())"
if [ "$MODE" = "run" ] && [ "$backend" != "gpu" ] && [ "$backend" != "cuda" ]; then
  echo "REFUSING a full sweep on backend '$backend' (a 150k run is ~70 min/arm on CPU)."
  echo "Set FORCE_CPU=1 to override."
  [ "${FORCE_CPU:-0}" = "1" ] || exit 1
fi

# ------------------------------------------------------------- arm listing
ARMS=()
for s in $SEEDS; do
  [ "$WITH_BASELINE" = "1" ] && ARMS+=("zbase|0|$s")
  for a in $ALPHAS; do ARMS+=("zfail|$a|$s"); done
done

banner "SWEEP PLAN  (${#ARMS[@]} runs, mode=$MODE, jobs=$JOBS)"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm alpha seed <<< "$spec"
  printf '  %-6s alpha=%-4s seed=%s\n' "$arm" "$alpha" "$seed"
done

argv_for() {   # arm alpha seed -> the launcher flags for that run
  local arm="$1" alpha="$2" seed="$3"
  if [ "$arm" = "zbase" ]; then
    echo "--version f4 --arm zbase --seed $seed"
  else
    echo "--version f4 --arm zfail --alpha $alpha --bank $BANK --seed $seed"
  fi
}
tag_for() {    # must match the launcher tag EXACTLY, or the eval step looks in
               # a directory that does not exist
  local arm="$1" alpha="$2" seed="$3"
  if [ "$arm" = "zbase" ]; then echo "swamp_windy_f4_zbase_s${seed}"
  else echo "swamp_windy_f4_zfail_a${alpha//0./}_s${seed}"; fi
}

if [ "$MODE" = "check" ]; then
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm alpha seed <<< "$spec"
    # shellcheck disable=SC2046
    $PY scripts/run_swamp_windy_z_failneg.py $(argv_for "$arm" "$alpha" "$seed") \
        --check-only >/dev/null || { echo "GATE FAILED: $spec"; exit 1; }
  done
  echo; echo "ALL ${#ARMS[@]} GATES PASS (no training performed)"
  exit 0
fi

FLAG="--run"; [ "$MODE" = "smoke" ] && FLAG="--smoke"

# ------------------------------------------------------------------- launch
banner "LAUNCHING"
pids=(); names=()
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm alpha seed <<< "$spec"
  tag=$(tag_for "$arm" "$alpha" "$seed")
  [ "$MODE" = "smoke" ] && tag="${tag}_smoke"
  log="$LOGDIR/${tag}.log"
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  # shellcheck disable=SC2046
  $PY scripts/run_swamp_windy_z_failneg.py $(argv_for "$arm" "$alpha" "$seed") \
      $FLAG > "$log" 2>&1 &
  pids+=($!); names+=("$tag")
  echo "  [$!] $tag -> $log"
  sleep 2                       # stagger XLA compilation
done

echo; echo "waiting for ${#pids[@]} runs..."
fail=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "  OK   ${names[$k]}"
  else echo "  FAIL ${names[$k]}  (see $LOGDIR/${names[$k]}.log)"; fail=$((fail+1)); fi
done
banner "TRAINING COMPLETE  ($((${#pids[@]}-fail))/${#pids[@]} ok)"
[ "$MODE" = "smoke" ] && exit $fail

# ------------------------------------------------------------------- eval
banner "DEPLOYMENT EVAL"
for k in "${!names[@]}"; do
  d="${names[$k]}"
  [ -f "$d/final.pkl" ] || { echo "  skip $d (no final.pkl)"; continue; }
  $PY -m scripts.eval_swamp_windy_z_deployment --ckpt "$d/final.pkl" \
      --env "$ENV_NAME" --out "artifacts/$d" --episodes "$EPISODES" \
      > "$LOGDIR/${d}_eval.log" 2>&1 \
    && echo "  evaluated $d" || echo "  EVAL FAILED $d (see $LOGDIR/${d}_eval.log)"
done

banner "RESULTS  (worst_case = success under all_active)"
$PY - <<'PYEOF'
import glob, json, os
rows = []
for p in sorted(glob.glob('artifacts/swamp_windy_f4_z*/deployment_report.json')):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    L = d.get('learner', {})
    rows.append((os.path.basename(os.path.dirname(p)),
                 L.get('all_clear', {}).get('success'),
                 L.get('all_active', {}).get('success'),
                 L.get('natural', {}).get('success'),
                 L.get('natural', {}).get('entry'),
                 L.get('natural', {}).get('died'),
                 d.get('verdict')))
if rows:
    hdr = '%-36s%7s%8s%7s%7s%7s  verdict' % (
        'run', 'clear', 'ACTIVE', 'nat', 'entry', 'died')
    print(hdr)
    print('-' * 98)
    for r in rows:
        f = lambda v: ('%.2f' % v) if isinstance(v, (int, float)) else '  - '
        print('%-36s%7s%8s%7s%7s%7s  %s'
              % (r[0], f(r[1]), f(r[2]), f(r[3]), f(r[4]), f(r[5]), r[6]))
else:
    print('no deployment reports found')
PYEOF
echo
echo "logs: $LOGDIR"
exit $fail
