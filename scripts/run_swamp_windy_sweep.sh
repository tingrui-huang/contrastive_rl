#!/usr/bin/env bash
# Alpha sweep for the windy-swamp failure-negative line, for an SSH GPU node.
#
# Arms (each x N seeds):
#   baseline    anchor cut OFF, alpha 0     -- the current pipeline
#   anchorcut   anchor cut ON,  alpha 0     -- scheme C alone
#   failneg     anchor cut ON,  alpha in {0.05, 0.1, 0.2}
#
# The model is tiny (hidden 256x256, repr 16, obs 2) and needs well under 1 GB
# of VRAM, so the runs are launched CONCURRENTLY on one GPU with JAX
# preallocation disabled. --jobs controls how many run at once.
#
# Usage:
#   bash scripts/run_swamp_windy_sweep.sh check          # gates only, no training
#   bash scripts/run_swamp_windy_sweep.sh smoke          # 2k-step smoke, every arm
#   bash scripts/run_swamp_windy_sweep.sh run            # full sweep
#   SEEDS="0 1 2" JOBS=5 bash scripts/run_swamp_windy_sweep.sh run
set -uo pipefail

MODE="${1:-check}"
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0.05 0.1 0.2}"
JOBS="${JOBS:-5}"
LOGDIR="${LOGDIR:-logs/swamp_windy_sweep}"
PY="${PY:-python}"

DATASET="datasets/swamp_windy_teacher_s0.npz"
BANK="artifacts/swamp_windy_failure_bank/failure_bank.npz"
# Content hashes, NOT file hashes: an .npz is a zip that embeds timestamps, so a
# regenerated dataset has different file bytes but identical arrays. datasets/
# is gitignored and therefore always regenerated on a fresh node.
DATASET_CONTENT_SHA="fd41c45cdb72749fb3b5a071c6f65a3003ec3117af630222f6726bfab7ea7952"
BANK_CONTENT_SHA="009edb4b529447f00e7ac59b088a6d9c9501084236df2aba16dfd43dda6f19a3"

# Many small processes on one GPU: JAX would otherwise preallocate 75% each.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.10}"
# Each process runs its own numpy sampler; do not let BLAS oversubscribe.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "$LOGDIR"

banner() { echo; echo "=============================================================="; echo "$1"; echo "=============================================================="; }

content_sha() {  # sha256 over the npz ARRAY CONTENTS (zip metadata ignored)
  $PY -c "
import hashlib, sys, numpy as np
h = hashlib.sha256()
with np.load(sys.argv[1], allow_pickle=True) as d:
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
  echo "  $PY -m scripts.collect_swamp_windy --episodes 6000 --random_frac 0.2 \\"
  echo "      --force_safe_prob 0.05 --teacher_noise 0.15 --seed 0 --out $DATASET"
  exit 1
fi
got=$(content_sha "$DATASET")
[ "$got" = "$DATASET_CONTENT_SHA" ] || { echo "dataset CONTENT mismatch (arrays differ, not just the zip container)"; echo "  expected $DATASET_CONTENT_SHA"; echo "  found    $got"; exit 1; }
echo "dataset OK  content $got"

if [ ! -f "$BANK" ]; then
  echo "building failure bank..."
  $PY scripts/make_swamp_failure_bank.py || exit 1
fi
got=$(content_sha "$BANK")
[ "$got" = "$BANK_CONTENT_SHA" ] || { echo "bank CONTENT mismatch"; echo "  expected $BANK_CONTENT_SHA"; echo "  found    $got"; exit 1; }
echo "bank OK     content $got"

backend=$($PY -c "import jax;print(jax.default_backend())" 2>/dev/null)
echo "jax backend : $backend"
$PY -c "import jax;print('devices    :',jax.devices())"
if [ "$MODE" = "run" ] && [ "$backend" != "gpu" ] && [ "$backend" != "cuda" ]; then
  echo "REFUSING a full sweep on backend '$backend' -- this is a ~6 h/arm job on CPU."
  echo "Set FORCE_CPU=1 to override."
  [ "${FORCE_CPU:-0}" = "1" ] || exit 1
fi

banner "MECHANISM VERIFICATION (scheme C + bank)"
$PY scripts/verify_anchor_cut.py || { echo "verification FAILED -- refusing to sweep"; exit 1; }

# ------------------------------------------------------------- arm listing
ARMS=()
for s in $SEEDS; do
  ARMS+=("baseline|0|$s")
  ARMS+=("anchorcut|0|$s")
  for a in $ALPHAS; do ARMS+=("failneg|$a|$s"); done
done

banner "SWEEP PLAN  (${#ARMS[@]} runs, mode=$MODE, jobs=$JOBS)"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm alpha seed <<< "$spec"
  printf '  %-10s alpha=%-5s seed=%s\n' "$arm" "$alpha" "$seed"
done

if [ "$MODE" = "check" ]; then
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm alpha seed <<< "$spec"
    $PY scripts/run_swamp_windy_failneg.py --arm "$arm" --alpha "$alpha" \
        --seed "$seed" --check-only >/dev/null || { echo "GATE FAILED: $spec"; exit 1; }
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
  tag="$arm"
  [ "$arm" = "failneg" ] && tag="failneg_a${alpha//./}"
  tag="swamp_windy_${tag}_s${seed}"
  log="$LOGDIR/${tag}.log"

  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done

  $PY scripts/run_swamp_windy_failneg.py --arm "$arm" --alpha "$alpha" \
      --seed "$seed" $FLAG > "$log" 2>&1 &
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

banner "SWEEP COMPLETE  ($((${#pids[@]}-fail))/${#pids[@]} ok)"
[ "$MODE" = "smoke" ] && exit $fail

# ------------------------------------------------------------------- eval
banner "DEPLOYMENT EVAL"
for k in "${!names[@]}"; do
  d="${names[$k]}"
  [ -f "$d/final.pkl" ] || { echo "  skip $d (no final.pkl)"; continue; }
  $PY -m scripts.eval_swamp_windy_deployment --ckpt "$d/final.pkl" \
      --out "artifacts/$d" --episodes 100 > "$LOGDIR/${d}_eval.log" 2>&1 \
    && echo "  evaluated $d" || echo "  EVAL FAILED $d"
done

banner "RESULTS  (worst_case = success under all_active)"
$PY - <<'EOF'
import glob, json, os
rows = []
for p in sorted(glob.glob('artifacts/swamp_windy_*/deployment_report.json')):
    try: d = json.load(open(p))
    except Exception: continue
    L = d.get('learner', {})
    rows.append((os.path.basename(os.path.dirname(p)),
                 L.get('all_clear', {}).get('success'),
                 L.get('all_active', {}).get('success'),
                 L.get('natural', {}).get('success'),
                 L.get('natural', {}).get('entry'),
                 L.get('natural', {}).get('died'),
                 d.get('verdict')))
if rows:
    print(f'{"run":<34}{"clear":>7}{"ACTIVE":>8}{"nat":>7}{"entry":>7}{"died":>7}  verdict')
    print('-' * 96)
    for r in rows:
        f = lambda v: f'{v:.2f}' if isinstance(v, (int, float)) else '  - '
        print(f'{r[0]:<34}{f(r[1]):>7}{f(r[2]):>8}{f(r[3]):>7}{f(r[4]):>7}'
              f'{f(r[5]):>7}  {r[6]}')
else:
    print('no deployment reports found')
EOF
echo
echo "logs: $LOGDIR"
exit $fail
