#!/usr/bin/env bash
# SETTLED-BANK alpha=0.1 run -- direct launcher (SSH GPU node).
#
# Identical to scripts/run_failneg_h800.sh (same trainer, dataset, recipe,
# seed, steps, eval protocol) except:
#   * failure bank -> artifacts/settled_failure_bank_alpha01/
#     failure_bank_settled.npz (16 N=80 physically-settled fatal states of
#     the SAME 16 pilot death episodes; scripts/rebuild_failure_bank_settled.py)
#   * alpha is FIXED at 0.1 (no sweep; the a0/a03/a05 arms are not rerun)
#   * hard provenance gate: refuses to start unless the clean-dataset sha,
#     settled-bank sha, bank size and alpha are exactly the authoritative
#     values -- and refuses the LEGACY bank sha explicitly, so this launcher
#     cannot silently reproduce the old experiment.
#
# Usage (on the node, venv active):
#   bash scripts/run_failneg_settledbank_h800.sh smoke          # gate + 5-update smoke, no checkpoints
#   bash scripts/run_failneg_settledbank_h800.sh run            # the production 300k run
#   bash scripts/run_failneg_settledbank_h800.sh run --resume   # resume after interruption
#   STEPS=300000 SEED=0 RUNS_ROOT=. ...                         # overrides (defaults shown)
set -euo pipefail
cd "$(dirname "$0")/.."

SEED=${SEED:-0}
STEPS=${STEPS:-300000}
RUNS_ROOT=${RUNS_ROOT:-.}            # '.' = repo root, matching existing runs
P_ACTIVE=0.3
HORIZON=800
ALPHA=0.1

SPLIT=artifacts/rockfall_v2_p30_h800_resetfix/failure_split
CLEAN_NPZ=$SPLIT/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz
BANK_NPZ=artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz
BANK_MANIFEST=artifacts/settled_failure_bank_alpha01/bank_manifest.json

# authoritative shas (also cross-checked against the committed manifests)
CLEAN_SHA_PIN=6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852
BANK_SHA_PIN=8c50202403317e8adc67670be716fffe5c3b3cb891017bfbe6eb070acd61d0ce
LEGACY_BANK_SHA=8d35b76ada59199e6ba22250a02bbdda931ff885beb08c2d8d146ecd14b41481

RUN_ID="failneg_settledbank_p30_h800_resetfix_a01_s${SEED}_$((STEPS/1000))k"
DIR="$RUNS_ROOT/$RUN_ID"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=${MUJOCO_GL:-egl}

# --- hard provenance gate ------------------------------------------------------
CLEAN_NPZ="$CLEAN_NPZ" BANK_NPZ="$BANK_NPZ" BANK_MANIFEST="$BANK_MANIFEST" \
CLEAN_SHA_PIN="$CLEAN_SHA_PIN" BANK_SHA_PIN="$BANK_SHA_PIN" \
LEGACY_BANK_SHA="$LEGACY_BANK_SHA" RUN_ID="$RUN_ID" ALPHA="$ALPHA" \
SEED="$SEED" STEPS="$STEPS" ALLOW_CPU="${ALLOW_CPU:-0}" \
python - <<'EOF'
import hashlib, json, os, subprocess
import numpy as np
import jax

env = os.environ
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                        text=True).stdout.strip()
clean_sha, bank_sha = sha(env['CLEAN_NPZ']), sha(env['BANK_NPZ'])
with np.load(env['BANK_NPZ']) as b:
    bank_n, bank_dim = b['goals'].shape
man = json.load(open(env['BANK_MANIFEST']))

print('=' * 66)
print('SETTLED-BANK PROVENANCE GATE')
print('  git commit        :', commit)
print('  run id            :', env['RUN_ID'])
print('  fail_neg_alpha    :', env['ALPHA'])
print('  failure bank      :', env['BANK_NPZ'])
print('  failure bank sha  :', bank_sha)
print('  failure bank size :', bank_n, 'x', bank_dim)
print('  clean dataset     :', env['CLEAN_NPZ'])
print('  clean dataset sha :', clean_sha)
print('  seed              :', env['SEED'])
print('  total updates     :', env['STEPS'])
print('  jax backend       :', jax.default_backend(), '|', jax.devices())
print('=' * 66)

assert clean_sha == env['CLEAN_SHA_PIN'], (
    'ABORT: clean dataset sha is not the authoritative sha')
assert bank_sha != env['LEGACY_BANK_SHA'], (
    'ABORT: this is the LEGACY failure bank -- refusing to run')
assert bank_sha == env['BANK_SHA_PIN'], (
    'ABORT: failure bank sha is not the settled-bank sha')
assert (bank_n, bank_dim) == (16, 29), 'ABORT: bank must be 16 x 29'
assert float(env['ALPHA']) == 0.1, 'ABORT: alpha must be 0.1'
assert man['bank']['sha256'] == bank_sha, 'ABORT: bank manifest sha drift'
assert all(man['checks'].values()), 'ABORT: bank validation not clean'
assert man['heldout_fresh_deaths']['n'] == 40, 'ABORT: held-out list missing'
if env['ALLOW_CPU'] != '1':
    assert jax.default_backend() == 'gpu', (
        'ABORT: no GPU backend (set ALLOW_CPU=1 only for a CPU smoke)')
print('PROVENANCE GATE PASSED')
EOF

case "${1:-}" in
  smoke)
    # 5-update in-process smoke: settled bank reaches the fail-neg loss,
    # metrics carry fail_bank_size=16, loss identity holds. No checkpoints;
    # scratch dir is separate from the production run dir.
    SMOKE_DIR="$RUNS_ROOT/${RUN_ID}_smoke"
    mkdir -p "$SMOKE_DIR"
    CLEAN_NPZ="$CLEAN_NPZ" BANK_NPZ="$BANK_NPZ" SEED="$SEED" \
    python - <<'EOF'
import os, time
import numpy as np, jax, optax
import sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
from crl import envs as envs_mod, networks as networks_mod, losses as losses_mod
from crl.replay import TrajectoryBuffer, obs_to_goal
from verify_offline_d4rl import build_offline_cfg
from rockfall_v2_teacher import SEVERITY_V2

env = os.environ
cfg = build_offline_cfg(max_steps=300000, ckpt_dir='')
cfg.env_name = 'offline_ant_umaze_rockfall'
cfg.offline_dataset = env['CLEAN_NPZ']
cfg.eval_goal_mode = 'd4rl'; cfg.rockfall_severity = SEVERITY_V2
cfg.rockfall_p_active = 0.3; cfg.rockfall_max_steps = 800
cfg.max_episode_steps = 800; cfg.rockfall_reset_fix = True
cfg.fail_bank_path = env['BANK_NPZ']; cfg.fail_neg_alpha = 0.1
cfg.seed = int(env['SEED'])
envs_mod.make_env(cfg.env_name, cfg, seed=cfg.seed)
with np.load(env['BANK_NPZ']) as fb:
    bank = np.asarray(fb['goals'], np.float32)
fail_bank = obs_to_goal(bank, cfg.start_index, cfg.end_index, cfg.goal_indices)
nets = networks_mod.make_networks(
    obs_dim=cfg.obs_dim, goal_dim=cfg.goal_dim, action_dim=cfg.action_dim,
    repr_dim=int(cfg.repr_dim), repr_norm=cfg.repr_norm,
    repr_norm_temp=cfg.repr_norm_temp,
    hidden_layer_sizes=cfg.hidden_layer_sizes, twin_q=cfg.twin_q,
    use_image_obs=False, use_layer_norm=cfg.use_layer_norm)
init_state, update_step = losses_mod.build_learner(
    nets, cfg, obs_to_goal, optax.adam(cfg.actor_learning_rate),
    optax.adam(cfg.learning_rate), fail_bank=fail_bank)
with np.load(env['CLEAN_NPZ'], allow_pickle=True) as d:
    obs, act = d['obs'], d['act']
buf = TrajectoryBuffer(capacity_steps=obs.shape[0] * obs.shape[1],
                       ep_len_obs=obs.shape[1], full_obs_dim=58, action_dim=8,
                       obs_dim=29, start_index=0, end_index=-1, discount=0.99,
                       seed=cfg.seed, goal_indices=tuple(range(29)))
for k in range(obs.shape[0]):
    buf.add_episode(obs[k], act[k])
state = init_state(jax.random.PRNGKey(cfg.seed))
state, m = update_step(state, buf.sample(cfg.batch_size))   # compile
m = {k: float(v) for k, v in m.items()}
assert m['fail_bank_size'] == 16 and abs(m['fail_neg_alpha'] - 0.1) < 1e-6
lhs = m['critic_pos_term'] + m['critic_neg_ord_term'] + m['critic_neg_fail_term']
assert abs(lhs - m['critic_loss']) < 1e-5, 'loss identity violated'
t0 = time.time()
for _ in range(5):
    state, m = update_step(state, buf.sample(cfg.batch_size))
jax.block_until_ready(state.q_params)
print('SMOKE OK | backend', jax.default_backend(),
      '| fail_bank_size', m['fail_bank_size'],
      '| logits_fail_neg %.3f' % m['logits_fail_neg'],
      '| L = pos + 0.9*ord + 0.1*fail verified',
      '| %.1f updates/s' % (5 / (time.time() - t0)))
EOF
    echo "SMOKE COMPLETE (scratch: $SMOKE_DIR; production dir untouched)"
    ;;

  run)
    resume_flag=${2:-}
    mkdir -p "$DIR"
    echo "=============================================================="
    echo "RUN $RUN_ID  (alpha=$ALPHA seed=$SEED steps=$STEPS)"
    echo "  bank: $BANK_NPZ (SETTLED, N=80)"
    echo "=============================================================="
    # train (G1-G8 offline audit runs first and aborts on any failure)
    python scripts/naive_rockfall_v2_crl.py \
        --npz "$CLEAN_NPZ" \
        --steps "$STEPS" --seed "$SEED" \
        --ckpt-dir "$DIR" \
        --p-active "$P_ACTIVE" --horizon "$HORIZON" --reset-fix \
        --fail-bank "$BANK_NPZ" --fail-neg-alpha "$ALPHA" \
        ${resume_flag:+--resume} \
        2>&1 | tee -a "$DIR/train.log"

    # post-train behavioural diagnosis (cap-800, corrected reset), final+best
    for ckpt in final best; do
      python scripts/diagnose_naive_rockfall.py \
          --v2 --p-active "$P_ACTIVE" --reset-fix --horizon "$HORIZON" \
          --ckpt "$DIR/$ckpt.pkl" \
          --out-dir "$DIR/eval_h800_resetfix/diag_$ckpt" \
          2>&1 | tee -a "$DIR/diagnose.log"
    done
    echo "DONE $RUN_ID"
    ;;

  *)
    echo "usage: $0 smoke | run [--resume]" >&2
    exit 1 ;;
esac
