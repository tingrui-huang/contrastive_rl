#!/usr/bin/env bash
# G4 sweep: from-scratch offline CRL on PointMaze F4, O (recorded) vs P
# (absorbing-freeze) dataset, same recipe, same bc, several seeds.
#
#   bash scripts/run_pointmaze_absorbing_g4_sweep.sh check   # gates only
#   bash scripts/run_pointmaze_absorbing_g4_sweep.sh smoke   # 2k steps per run
#   bash scripts/run_pointmaze_absorbing_g4_sweep.sh run     # 150k steps per run
#
# Env: SEEDS="0 1 2"  BCS="0.2"  ARMS="O P"  JOBS=3  LOGDIR=logs/pointmaze_absorbing_g4
# Every run is launched detached (nohup) so a dropped SSH session does not
# kill it; progress is in $LOGDIR/<tag>.log and runs/pointmaze_absorbing_g4/<tag>/.
set -uo pipefail
MODE="${1:-check}"
SEEDS="${SEEDS:-0 1 2}"
BCS="${BCS:-0.2}"
ARMS="${ARMS:-O P}"
JOBS="${JOBS:-3}"
LOGDIR="${LOGDIR:-logs/pointmaze_absorbing_g4}"
PY="${PY:-python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$LOGDIR"

tag_of() { echo "g4_$1_bc$(echo "$2" | tr -d '.' | sed 's/^0/0p/')_s$3"; }

if [ "$MODE" = "check" ]; then
  for bc in $BCS; do
    $PY -m scripts.run_pointmaze_absorbing_g4 --arm O --bc "$bc" --seed 0 --check-only || exit 1
    $PY -m scripts.run_pointmaze_absorbing_g4 --arm P --bc "$bc" --seed 0 --check-only || exit 1
  done
  echo "CHECK PASSED"; exit 0
fi

FLAG="--run"; [ "$MODE" = "smoke" ] && FLAG="--smoke"
running=0
for seed in $SEEDS; do
  for bc in $BCS; do
    for arm in $ARMS; do
      tag=$(tag_of "$arm" "$bc" "$seed")
      [ "$MODE" = "smoke" ] && tag="${tag}_smoke"
      echo "launch $tag"
      nohup $PY -m scripts.run_pointmaze_absorbing_g4 --arm "$arm" --bc "$bc" --seed "$seed" $FLAG \
        > "$LOGDIR/$tag.log" 2>&1 &
      running=$((running + 1))
      if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
    done
  done
done
wait
echo "ALL LAUNCHED RUNS FINISHED"
