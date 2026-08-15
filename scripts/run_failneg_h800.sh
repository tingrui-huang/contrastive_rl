#!/usr/bin/env bash
# Part 1 failure-aware negative sampling -- direct launcher (SSH GPU node).
#
# One run per alpha on the H=800 corrected-reset protocol (p_active=0.30,
# v2.1 severity, reset-fix), D_clean dataset, 300k steps, seed 0. The ONLY
# thing that varies across runs is --fail-neg-alpha. Each run gets its own
# directory (checkpoints + metrics.json + train.log + post-train diagnosis).
#
# Usage:
#   bash scripts/run_failneg_h800.sh all                # a=0, 0.1, 0.3, 0.5 sequentially
#   bash scripts/run_failneg_h800.sh 0.3                # one alpha
#   bash scripts/run_failneg_h800.sh 0.3 --resume       # resume after interruption
#   SEED=1 bash scripts/run_failneg_h800.sh all         # different seed
#   STEPS=300000 RUNS_ROOT=. ...                        # overrides (defaults shown)
#
# Run inside tmux/screen or under nohup, e.g.:
#   nohup bash scripts/run_failneg_h800.sh all > failneg_all.log 2>&1 &
#
# Experiments:
#   A  alpha=0    clean baseline (byte-identical original CRL on D_clean)
#   B  alpha>0    failure-aware negatives (the experiment)
#   C  (control)  NOT launched here -- reuse the existing full-data run
#                 naive_rockfall_v2_p30_h800_resetfix_s0_300k (normal CRL on
#                 D_original, same recipe/eval).
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-0}
STEPS=${STEPS:-300000}
RUNS_ROOT=${RUNS_ROOT:-.}            # '.' = repo root, matching existing runs
P_ACTIVE=0.3
HORIZON=800

SPLIT=artifacts/rockfall_v2_p30_h800_resetfix/failure_split
CLEAN_NPZ=$SPLIT/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz
BANK_NPZ=$SPLIT/failure_bank.npz
MANIFEST=$SPLIT/failure_split_manifest.json

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=${MUJOCO_GL:-egl}

# --- integrity gate: dataset + bank must match the committed manifest ---------
python - "$CLEAN_NPZ" "$BANK_NPZ" "$MANIFEST" <<'EOF'
import hashlib, json, sys
clean, bank, manifest = sys.argv[1:4]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
man = json.load(open(manifest))
assert sha(clean) == man['clean']['sha256'], 'clean npz sha mismatch'
assert sha(bank) == man['failure_bank']['sha256'], 'failure bank sha mismatch'
assert man['clean']['n_episodes'] == 284 and man['rockfail']['n_episodes'] == 16
print('failure split verified: clean sha %s..., bank sha %s..., 284+16 eps'
      % (man['clean']['sha256'][:12], man['failure_bank']['sha256'][:12]))
EOF

tag_of() {  # 0->a0  0.1->a01  0.3->a03  0.5->a05
  echo "a$(echo "$1" | tr -d '.' | sed 's/0*$//; s/^$/0/')"
}

run_one() {
  local alpha=$1; shift
  local resume_flag=${1:-}
  local tag; tag=$(tag_of "$alpha")
  local run_id="failneg_clean_p30_h800_resetfix_${tag}_s${SEED}_$((STEPS/1000))k"
  local dir="$RUNS_ROOT/$run_id"
  mkdir -p "$dir"

  local fail_args=()
  if [ "$alpha" != "0" ] && [ "$alpha" != "0.0" ]; then
    fail_args=(--fail-bank "$BANK_NPZ" --fail-neg-alpha "$alpha")
  fi

  echo "=============================================================="
  echo "RUN $run_id  (alpha=$alpha seed=$SEED steps=$STEPS)"
  echo "=============================================================="
  # train (G1-G8 offline audit runs first and aborts on any failure)
  python scripts/naive_rockfall_v2_crl.py \
      --npz "$CLEAN_NPZ" \
      --steps "$STEPS" --seed "$SEED" \
      --ckpt-dir "$dir" \
      --p-active "$P_ACTIVE" --horizon "$HORIZON" --reset-fix \
      "${fail_args[@]}" \
      ${resume_flag:+--resume} \
      2>&1 | tee -a "$dir/train.log"

  # post-train behavioural diagnosis (cap-800, corrected reset), final + best
  for ckpt in final best; do
    python scripts/diagnose_naive_rockfall.py \
        --v2 --p-active "$P_ACTIVE" --reset-fix --horizon "$HORIZON" \
        --ckpt "$dir/$ckpt.pkl" \
        --out-dir "$dir/eval_h800_resetfix/diag_$ckpt" \
        2>&1 | tee -a "$dir/diagnose.log"
  done
  echo "DONE $run_id"
}

case "${1:-}" in
  all)
    for a in 0 0.1 0.3 0.5; do run_one "$a"; done ;;
  0|0.0|0.1|0.3|0.5)
    run_one "$1" "${2:-}" ;;
  *)
    echo "usage: $0 all | <alpha in {0,0.1,0.3,0.5}> [--resume]" >&2
    exit 1 ;;
esac
