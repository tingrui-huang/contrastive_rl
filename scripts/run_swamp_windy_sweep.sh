#!/usr/bin/env bash
# Alpha sweep for the windy-swamp failure-negative line, for an SSH GPU node.
#
# Arms (each x N seeds):
#   baseline    anchor cut OFF, alpha 0     -- the current pipeline
#   anchorcut   anchor cut ON,  alpha 0     -- scheme C alone
#   failneg     anchor cut ON,  alpha in {0.05, 0.1, 0.2}
#
# Runs go SEQUENTIALLY by default (see JOBS below -- concurrency measured
# WORSE on this workload). A 150k-step run takes ~10 min on an RTX 3090, so
# 5 arms x 3 seeds is roughly 2.5 h.
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
ARMSET="${ARMSET:-default}"
# Registered dataset name, not a path: main | merged_s0 | merged_s1 | merged_s2.
# merged_* add the 600 bad-demonstrator episodes.
DATASET="${DATASET:-main}"
# SEQUENTIAL by default. Measured on a 3090: one process runs at 253 steps/s
# (G=10), but five concurrent processes managed only ~22 steps/s each (~110
# aggregate). The model is dispatch-bound, not compute-bound -- GPU utilisation
# sits near 4% -- so extra processes only contend for the one CUDA context.
JOBS="${JOBS:-1}"
LOGDIR="${LOGDIR:-logs/swamp_windy_sweep}"
PY="${PY:-python}"

BANK="artifacts/swamp_windy_failure_bank/failure_bank.npz"
# Content hashes, NOT file hashes: an .npz is a zip that embeds timestamps, so a
# regenerated dataset has different file bytes but identical arrays. datasets/
# is gitignored and therefore always regenerated on a fresh node.
BANK_CONTENT_SHA="b680aab6b224ec5b1243058a54c678d5ab8897935106b84a4addab5429fa5381"

# The dataset path and its expected content hash come from the launcher's
# DATASETS registry rather than being repeated here -- two copies of a hash is
# exactly the pair that drifts. The launcher re-checks it anyway; this
# preflight exists to fail in one second instead of after the first XLA
# compile. (No f-string: the node runs python 3.11, which predates PEP 701 and
# rejects nested quotes inside one.)
read -r DATASET DATASET_CONTENT_SHA DATASET_REGEN <<EOF
$($PY -c "
import sys
sys.path.insert(0, 'scripts')
from run_swamp_windy_failneg import DATASETS
name = sys.argv[1]
if name not in DATASETS:
    sys.exit('unknown DATASET=' + name + '; registered: ' + repr(sorted(DATASETS)))
d = DATASETS[name]
print(d['path'], d['content_sha'], name)
" "$DATASET")
EOF
[ -n "${DATASET:-}" ] || { echo "could not resolve DATASET (see above)"; exit 1; }
echo "dataset name: $DATASET_REGEN  ->  $DATASET"

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
  echo "      --force_safe_prob 0.05 --teacher_noise 0.15 --seed 0 \\"
  echo "      --out datasets/swamp_windy_teacher_s0.npz"
  case "$DATASET_REGEN" in
    merged_s*)
      n="${DATASET_REGEN#merged_s}"
      echo "  # then, for $DATASET_REGEN:"
      echo "  $PY -m scripts.collect_swamp_windy_baddemo --episodes 600 --seed $n \\"
      echo "      --out datasets/swamp_windy_baddemo_s${n}.npz"
      echo "  $PY -m scripts.merge_swamp_windy_baddemo \\"
      echo "      --bad datasets/swamp_windy_baddemo_s${n}.npz --out $DATASET"
      ;;
  esac
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
# ARMSET: which family of arms to run.
#   control  baseline only                              -- the reference check
#   default  baseline / anchorcut / failneg(alpha)      -- uniform-row anchors
#   balanced balanced / balancedfail(alpha)             -- + balanced (s,a)
#   all      default + balanced
case "$ARMSET" in
  control|default|balanced|all) ;;
  *) echo "unknown ARMSET=$ARMSET (expected control|default|balanced|all)"; exit 1 ;;
esac
ARMS=()
for s in $SEEDS; do
  if [ "$ARMSET" = "control" ]; then
    ARMS+=("baseline|0|$s")
  fi
  if [ "$ARMSET" = "default" ] || [ "$ARMSET" = "all" ]; then
    ARMS+=("baseline|0|$s")
    ARMS+=("anchorcut|0|$s")
    for a in $ALPHAS; do ARMS+=("failneg|$a|$s"); done
  fi
  if [ "$ARMSET" = "balanced" ] || [ "$ARMSET" = "all" ]; then
    ARMS+=("balanced|0|$s")
    for a in $ALPHAS; do ARMS+=("balancedfail|$a|$s"); done
  fi
done

banner "SWEEP PLAN  (${#ARMS[@]} runs, mode=$MODE, jobs=$JOBS, dataset=$DATASET_REGEN)"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm alpha seed <<< "$spec"
  printf '  %-10s alpha=%-5s seed=%s\n' "$arm" "$alpha" "$seed"
done

if [ "$MODE" = "check" ]; then
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm alpha seed <<< "$spec"
    $PY scripts/run_swamp_windy_failneg.py --arm "$arm" --alpha "$alpha" \
        --seed "$seed" --dataset "$DATASET_REGEN" --check-only >/dev/null \
      || { echo "GATE FAILED: $spec"; exit 1; }
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
  # Must match run_swamp_windy_failneg.py's own tag exactly, or the eval step
  # looks in a directory that does not exist. Every bank-carrying arm needs the
  # alpha suffix, otherwise its alpha values collide into one directory.
  tag="$arm"
  case "$arm" in
    failneg|balancedfail) tag="${arm}_a${alpha//./}" ;;
  esac
  # Dataset suffix, matching the launcher: main -> nothing (the 27 existing run
  # directories keep their names), merged_s<n> -> _bd<n>.
  dstag=""
  case "$DATASET_REGEN" in
    merged_s*) dstag="_bd${DATASET_REGEN#merged_s}" ;;
  esac
  tag="swamp_windy_${tag}${dstag}_s${seed}"
  log="$LOGDIR/${tag}.log"

  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done

  $PY scripts/run_swamp_windy_failneg.py --arm "$arm" --alpha "$alpha" \
      --seed "$seed" --dataset "$DATASET_REGEN" $FLAG > "$log" 2>&1 &
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
