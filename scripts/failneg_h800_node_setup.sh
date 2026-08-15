#!/usr/bin/env bash
# Environment setup for the Part 1 failure-negative runs on a FRESH
# Ubuntu + NVIDIA GPU node (tested version set = the workstation-validated
# environment; the 2022 requirements.txt is the dead Acme pin set -- do NOT
# install it, the crl/ port does not use it).
#
# Usage (on the node):
#   bash scripts/failneg_h800_node_setup.sh
#
# Prereqs: NVIDIA driver already installed (nvidia-smi works). CUDA toolkit
# is NOT required -- jax[cuda12] wheels bundle their own CUDA libraries.
set -euo pipefail

PYTHON=${PYTHON:-python3.12}          # 3.11-3.14 all fine for these pins
VENV=${VENV:-$HOME/crlenv}

# --- system packages (venv module, EGL for headless MuJoCo rendering) --------
sudo apt-get update
sudo apt-get install -y git ${PYTHON}-venv ${PYTHON}-dev libegl1 libgl1

# --- virtualenv ---------------------------------------------------------------
$PYTHON -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip

# --- python packages (mirror the validated workstation env) -------------------
# jax[cuda12] pulls the matching CUDA-enabled jaxlib + bundled CUDA runtime.
pip install "jax[cuda12]==0.10.2"
pip install dm-haiku==0.0.16 optax==0.2.8 mujoco==3.10.0 "numpy==2.4.4" \
            etils==1.14.0 jmp==0.0.4 tabulate==0.10.0 \
            imageio imageio-ffmpeg tensorboardX   # last three optional (gifs/TB)

# --- verify -------------------------------------------------------------------
python - <<'EOF'
import jax, mujoco, haiku, optax, numpy
print('jax', jax.__version__, '| backend:', jax.default_backend(),
      '| devices:', jax.devices())
print('mujoco', mujoco.__version__, '| haiku', haiku.__version__,
      '| optax', optax.__version__, '| numpy', numpy.__version__)
assert jax.default_backend() == 'gpu', 'NO GPU BACKEND -- check driver/wheels'
EOF

echo
echo "OK. Next:"
echo "  git clone <repo-url> && cd contrastive_rl && git checkout <pinned-sha>"
echo "  source $VENV/bin/activate"
echo "  python scripts/sanity_fail_negatives.py          # pre-launch gate"
echo "  bash scripts/run_failneg_h800.sh all             # the 4 runs"
