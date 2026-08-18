#!/usr/bin/env bash
# Server A: V1 Flow sweep, beta in {0.05, 0.10}, both families (4 runs).
# Sequential inside the server; Server B runs beta in {0.15, 0.20} in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p logs
for beta in 0.05 0.10; do
  for fam in S SA; do
    echo "=== V1-${fam}-beta${beta} ==="
    python scripts/train_flow_v1.py --family "$fam" --beta "$beta" \
      2>&1 | tee -a "logs/flow_v1_serverA.log"
  done
done
echo "SERVER A DONE"
