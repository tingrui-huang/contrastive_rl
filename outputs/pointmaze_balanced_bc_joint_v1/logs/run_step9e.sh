#!/usr/bin/env bash
# Step 9e: fair plain-CRL control (O critics on the original data + the same balanced-BC two-stage actor)
cd ~/contrastive_rl
until grep -q "STEP9D_DONE" step9d.out 2>/dev/null; do sleep 30; done
export PYTHONPATH=$PWD XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
~/crlenv/bin/python scripts/run_f4_balanced_bc_joint.py control --parallel 3 || echo "CONTROL FAILED"
echo STEP9E_DONE
