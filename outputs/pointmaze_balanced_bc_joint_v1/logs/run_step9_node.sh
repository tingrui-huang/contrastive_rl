#!/usr/bin/env bash
cd ~/contrastive_rl
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
P=~/crlenv/bin/python
$P -c "import jax, jax.numpy as jnp; x=jnp.ones((2048,2048)); print('gpu matmul ok', float((x@x).sum()), jax.devices())" || { echo GPU_TEST_FAILED; exit 1; }
$P scripts/run_f4_balanced_bc_joint.py attribution --parallel 3 || echo "ATTRIBUTION FAILED"
$P scripts/run_f4_balanced_bc_joint.py train --budgets b30k --parallel 3 || echo "TRAIN b30k FAILED"
$P scripts/run_f4_balanced_bc_joint.py evaluate --budgets b30k || echo "EVAL b30k FAILED"
$P scripts/run_f4_balanced_bc_joint.py train --budgets b300k --parallel 3 || echo "TRAIN b300k FAILED"
$P scripts/run_f4_balanced_bc_joint.py evaluate --budgets b300k || echo "EVAL b300k FAILED"
$P scripts/run_f4_balanced_bc_joint.py summarize || echo "SUMMARIZE FAILED"
echo STEP9_DONE
