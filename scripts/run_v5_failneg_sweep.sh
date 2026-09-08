#!/usr/bin/env bash
# Alpha sweep for the V5 rockfall-clock failure-negative line, for a GPU node.
#
# Arms (each x N seeds):
#   base            alpha 0            -- the reference, no bank is loaded
#   fail alpha in   {0.1 .. 0.5}       -- the composed bank, mixture weight swept
#
# The bank is the COMPOSED one: 60% noisy-controller deaths + 40% "deliberate"
# deaths (the sighted expert read the timetable, its rule said WAIT, and it was
# overridden to GO). Built by scripts/make_v5_failure_bank.py --compose. The
# class selection reads sidecar fields the learner never sees (source_arm,
# teacher_decision, schedule_read, intervention), which is recorded in the bank
# manifest; the stored vectors are plain 29-dim learner states.
#
# NOTE ON "60/40": that is the composition INSIDE the bank. Alpha is the weight
# the bank carries in the critic's negative term. They are independent knobs and
# both are set here explicitly.
#
# GOAL_REP picks the goal contract, and both values get their own alpha = 0 arm:
#   xyv (default)  XY + the six torso velocity columns -- the representation in
#                  which a death state is distinguishable from a safe crossing
#   xy             the frozen V5 default; a death and a crossing share their XY
#                  exactly (5-fold episode-grouped AUC 0.440), so this arm measures
#                  what a
#                  positional-only bank does
#
# Usage:
#   bash scripts/run_v5_failneg_sweep.sh check      # gates only, no training
#   bash scripts/run_v5_failneg_sweep.sh smoke      # 2k-step smoke, every arm
#   bash scripts/run_v5_failneg_sweep.sh run        # full sweep
#   SEEDS="0 1 2" bash scripts/run_v5_failneg_sweep.sh run
#   GOAL_REP=xy bash scripts/run_v5_failneg_sweep.sh run
set -uo pipefail

MODE="${1:-check}"
SEEDS="${SEEDS:-0}"
ALPHAS="${ALPHAS:-0.1 0.2 0.3 0.4 0.5}"
WITH_BASELINE="${WITH_BASELINE:-1}"
GOAL_REP="${GOAL_REP:-xyv}"
STEPS="${STEPS:-100000}"
SMOKE_STEPS="${SMOKE_STEPS:-2000}"
LOGDIR="${LOGDIR:-logs/v5_failneg_sweep_${GOAL_REP}}"
PY="${PY:-python}"
EPISODES="${EPISODES:-300}"
EVAL_MODE="${EVAL_MODE:-mean}"

DATASET_DIR="artifacts/rockfall_clock_v5/dataset"
case "$GOAL_REP" in
  xyv)
    DATASET="$DATASET_DIR/antmaze_rockfall_clock_v5_far05_gxyv.npz"
    DEFAULT_DATASET_CONTENT_SHA="f8a0dc3580fb46afb75c1d1534bf0daf7d3db26cce69b803888d76946f65e156"
    ;;
  xy)
    DATASET="$DATASET_DIR/antmaze_rockfall_clock_v5_far05_gxy.npz"
    DEFAULT_DATASET_CONTENT_SHA="6c80a581a1a6fdffdbd35fa4f7354a31fdfb97c0962ad24b40e0c197e93ea9ea"
    ;;
  *)   echo "GOAL_REP must be xyv or xy, got '$GOAL_REP'"; exit 1 ;;
esac
BANK="${BANK:-artifacts/v5_failneg/bank/v5_failure_bank_n60d40.npz}"
COMPOSE="${COMPOSE:-noisy=0.6,deliberate=0.4}"
MAX_BANK="${MAX_BANK:-256}"
# Pin the exact artifacts this sweep was defined against.  Environment
# overrides remain available for an intentionally different frozen dataset or
# bank, but production defaults can never silently drift to a rebuild.
DATASET_CONTENT_SHA="${DATASET_CONTENT_SHA:-$DEFAULT_DATASET_CONTENT_SHA}"
BANK_CONTENT_SHA="${BANK_CONTENT_SHA:-3a0a671d2f001edd7a89a9e37e6973677c83f9d5eaafc0c4b079eeb29a0ff39a}"

# SEQUENTIAL BY DEFAULT. The f4 sweep measured a 2.3x SLOWDOWN from running
# five of these concurrently on one 3090 (dispatch-bound workload, ~4% GPU
# utilisation, one CUDA context to fight over). The AntMaze recipe is heavier
# per step (batch 1024, hidden 1024x1024) so it may parallelise better -- but
# raise JOBS only after measuring it on your node, not on principle.
JOBS="${JOBS:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
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
banner "PREFLIGHT  (goal rep $GOAL_REP)"
if [ ! -f "$DATASET" ]; then
  echo "MISSING $DATASET"
  echo "regenerate it on this node with:"
  echo "  $PY scripts/collect_rockfall_clock_v5_dataset.py --p-far 0.05"
  echo "  $PY scripts/make_v5_gxy_dataset.py     # goal rep xy"
  echo "  $PY scripts/make_v5_gxyv_dataset.py    # goal rep xyv"
  exit 1
fi
got=$(content_sha "$DATASET")
if [ -n "$DATASET_CONTENT_SHA" ] && [ "$got" != "$DATASET_CONTENT_SHA" ]; then
  echo "dataset CONTENT mismatch"; echo "  expected $DATASET_CONTENT_SHA"; echo "  found    $got"; exit 1
fi
echo "dataset OK  $DATASET"
echo "            content $got"

if [ ! -f "$BANK" ]; then
  echo "building the composed failure bank ($COMPOSE, n=$MAX_BANK)..."
  $PY scripts/make_v5_failure_bank.py --compose "$COMPOSE" \
      --max-bank "$MAX_BANK" --seed 0 --out-name "$(basename "$BANK")" || exit 1
fi
gotb=$(content_sha "$BANK")
if [ -n "$BANK_CONTENT_SHA" ] && [ "$gotb" != "$BANK_CONTENT_SHA" ]; then
  echo "bank CONTENT mismatch -- this is not the bank the sweep was defined against"
  echo "  expected $BANK_CONTENT_SHA"; echo "  found    $gotb"; exit 1
fi
echo "bank OK     $BANK"
echo "            content $gotb"
$PY -c "
import collections, json, sys
import numpy as np
b = np.load(sys.argv[1], allow_pickle=False)
g, c = b['goals'], b['source_arm']
m = json.loads(str(b['meta']))
assert g.shape[1] == 29, 'bank must store 29-dim learner states, got %d' % g.shape[1]
assert len(set(zip(b['source_file'].tolist(), b['episode_id'].tolist()))) == len(g), \
    'an episode appears twice in the bank'
print('bank mix    : %s  (n=%d, stored dim %d)'
      % (dict(collections.Counter(c.tolist())), len(g), g.shape[1]))
print('bank compose: %s   extraction: %s' % (m.get('compose'), m['extraction_moment'].split(' -- ')[0]))
print('privileged  : %s' % m.get('selection_uses_privileged_fields'))
" "$BANK" || exit 1
echo "pin these in the script header before a production sweep:"
echo "  DATASET_CONTENT_SHA=$got"
echo "  BANK_CONTENT_SHA=$gotb"

backend=$($PY -c "import jax;print(jax.default_backend())" 2>/dev/null)
echo "jax backend : $backend"
$PY -c "import jax;print('devices     :',jax.devices())"
if [ "$MODE" = "run" ] && [ "$backend" != "gpu" ] && [ "$backend" != "cuda" ]; then
  echo "REFUSING a full sweep on backend '$backend' (batch 1024 x 1024x1024 MLP on CPU is hours per arm)."
  echo "Set FORCE_CPU=1 to override."
  [ "${FORCE_CPU:-0}" = "1" ] || exit 1
fi

# ------------------------------------------------------------- arm listing
ARMS=()
for s in $SEEDS; do
  [ "$WITH_BASELINE" = "1" ] && ARMS+=("base|0|$s")
  for a in $ALPHAS; do ARMS+=("fail|$a|$s"); done
done

banner "SWEEP PLAN  (${#ARMS[@]} runs, mode=$MODE, jobs=$JOBS, goal=$GOAL_REP)"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm alpha seed <<< "$spec"
  printf '  %-5s alpha=%-4s seed=%s\n' "$arm" "$alpha" "$seed"
done

argv_for() {   # arm alpha seed -> launcher flags
  local arm="$1" alpha="$2" seed="$3"
  if [ "$arm" = "base" ]; then
    echo "--goal-rep $GOAL_REP --arm base --seed $seed --steps $STEPS"
  else
    echo "--goal-rep $GOAL_REP --arm fail --alpha $alpha --bank $BANK --seed $seed --steps $STEPS"
  fi
}
tag_for() {    # must match the launcher's run_id EXACTLY or the eval step
               # looks in a directory that does not exist
  local arm="$1" alpha="$2" seed="$3"
  if [ "$arm" = "base" ]; then echo "v5fn_${GOAL_REP}_base_s${seed}"
  else echo "v5fn_${GOAL_REP}_a${alpha//0./}_s${seed}"; fi
}

if [ "$MODE" = "check" ]; then
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm alpha seed <<< "$spec"
    # shellcheck disable=SC2046
    $PY scripts/run_v5_failneg.py $(argv_for "$arm" "$alpha" "$seed") \
        --check-only >/dev/null || { echo "GATE FAILED: $spec"; exit 1; }
  done
  echo; echo "ALL ${#ARMS[@]} GATES PASS (no training performed)"
  exit 0
fi

FLAG="--run"; [ "$MODE" = "smoke" ] && FLAG="--smoke"
EXPECTED_STEPS="$STEPS"; [ "$MODE" = "smoke" ] && EXPECTED_STEPS="$SMOKE_STEPS"

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
  $PY scripts/run_v5_failneg.py $(argv_for "$arm" "$alpha" "$seed") \
      $FLAG --smoke-steps "$SMOKE_STEPS" > "$log" 2>&1 &
  pids+=($!); names+=("$tag")
  echo "  [$!] $tag -> $log"
  sleep 2                       # stagger XLA compilation
done

echo; echo "waiting for ${#pids[@]} runs..."
fail=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}" && $PY -c \
      "from crl.checkpoint import load_checkpoint; import sys; step,_=load_checkpoint(sys.argv[1]); assert step == int(sys.argv[2]), (step, int(sys.argv[2]))" \
      "${names[$k]}/final.pkl" "$EXPECTED_STEPS"; then
    echo "  OK   ${names[$k]} (verified final step $EXPECTED_STEPS)"
  else
    echo "  FAIL ${names[$k]} (process error, guard abort, missing checkpoint, or wrong final step; see $LOGDIR/${names[$k]}.log)"
    fail=$((fail+1))
  fi
done
banner "TRAINING COMPLETE  ($((${#pids[@]}-fail))/${#pids[@]} ok)"
[ "$MODE" = "smoke" ] && exit $fail

# ------------------------------------------------------------------- eval
banner "DEPLOYMENT EVAL  (n=$EPISODES, mode=$EVAL_MODE)"
post_fail=0
for k in "${!names[@]}"; do
  d="${names[$k]}"
  [ -f "$d/final.pkl" ] || { echo "  skip $d (no final.pkl)"; continue; }
  $PY scripts/eval_rockfall_clock_v5_baseline.py --ckpt "$d/final.pkl" \
      --goal-rep "$GOAL_REP" --variant far05 --mode "$EVAL_MODE" \
      --n "$EPISODES" --method-label "$d" \
      > "$LOGDIR/${d}_eval.log" 2>&1 \
    && echo "  evaluated $d" \
    || { echo "  EVAL FAILED $d (see $LOGDIR/${d}_eval.log)"; post_fail=$((post_fail+1)); }
done

# ------------------------------------------------------------------ audit
banner "CRITIC-SEMANTIC AUDIT"
$PY scripts/audit_v5_failneg.py --goal-rep "$GOAL_REP" \
    --runs-glob "v5fn_${GOAL_REP}_*_s*" --bank "$BANK" \
    > "$LOGDIR/audit.log" 2>&1 \
  && echo "  audit written (see $LOGDIR/audit.log)" \
  || { echo "  AUDIT FAILED (see $LOGDIR/audit.log)"; post_fail=$((post_fail+1)); }

banner "RESULTS"
$PY - "$GOAL_REP" <<'PYEOF'
import glob, json, os, sys
rep = sys.argv[1]
rows = []
for p in sorted(glob.glob(f'v5fn_{rep}_*_s*/eval_rockfall_clock_v5_*.json')):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    s = d.get('summary', {})
    o = s.get('overall', {})
    r = o.get('route', {})
    prov = {}
    pp = os.path.join(os.path.dirname(p), 'arm_provenance.json')
    if os.path.exists(pp):
        prov = json.load(open(pp))
    rows.append((os.path.basename(os.path.dirname(p)), prov.get('alpha'),
                 o.get('success'), o.get('failure'), o.get('timeout'),
                 s.get('death_given_active'), r.get('shortcut'),
                 r.get('detour'), o.get('entered_band'), o.get('waited_rate'),
                 o.get('mean_discounted')))
if rows:
    hdr = ('%-24s%6s%8s%7s%8s%9s%9s%8s%8s%8s%7s'
           % ('run', 'alpha', 'succ', 'death', 'timeo', 'd|act', 'shortcut',
              'detour', 'band', 'waited', 'disc'))
    print(hdr); print('-' * len(hdr))
    f = lambda v: ('%.3f' % v) if isinstance(v, (int, float)) else '   -  '
    for x in rows:
        print('%-24s%6s%8s%7s%8s%9s%9s%8s%8s%8s%7s'
              % (x[0], f(x[1]), f(x[2]), f(x[3]), f(x[4]), f(x[5]), f(x[6]),
                 f(x[7]), f(x[8]), f(x[9]), f(x[10])))
    print()
    print('success and death alone are NOT the readout: an ant that never')
    print('moves has death 0 too. Read them with timeout, shortcut, detour,')
    print('band entry and waited.')
else:
    print('no eval reports found')
PYEOF
echo
echo "logs: $LOGDIR"
exit $((fail+post_fail))
