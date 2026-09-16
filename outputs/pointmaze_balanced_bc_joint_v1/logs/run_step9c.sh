#!/usr/bin/env bash
# Step 9c: are the jointly trained critics good enough when the actor starts fresh?
# Freeze the final critic of joint balanced seed 0 at 300k (d-r +0.70, 16/16) and at 30k
# (d-r +0.28, 16/16); train 3 fresh actors each with balanced BC rows (Step 8 procedure).
cd ~/contrastive_rl
export PYTHONPATH=$PWD XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
P=~/crlenv/bin/python
F=outputs/pointmaze_diagonal_coverage_fix_v1
J=outputs/pointmaze_balanced_bc_joint_v1/joint
O=outputs/pointmaze_balanced_bc_joint_v1/refreeze
mkdir -p $O/logs
for b in b300k b30k; do
  for a in 0 1 2; do
    $P scripts/train_f4_actor_fixed_critic.py --critic $J/$b/seed_0/balanced/final.pkl --replay $F/replay_D.npz --actor-seed $a --log-prob acme --random-goals 0 --bc-sampling balanced --bc-cap 0.25 --out $O/critic_${b}_s0/actor_s$a > $O/logs/${b}_actor_s$a.log 2>&1 &
  done
  wait
done
for b in b300k b30k; do
  args=()
  for a in 0 1 2; do args+=(--ckpt "${b}_a${a}=$O/critic_${b}_s0/actor_s$a/final.pkl"); done
  for pol in mode sample; do
    $P -m scripts.eval_pointmaze_native_routes "${args[@]}" --episodes 300 --policy $pol --reset-seed-base 31500000 --action-seed-base 31600000 --out "$O/native_new_${pol}_$b" > "$O/logs/eval_new_${pol}_$b.log" 2>&1 || echo "EVAL FAILED new $pol $b"
    $P -m scripts.eval_pointmaze_native_routes "${args[@]}" --episodes 200 --policy $pol --reset-seed-base 29800000 --action-seed-base 29900000 --out "$O/native_sealed_${pol}_$b" > "$O/logs/eval_sealed_${pol}_$b.log" 2>&1 || echo "EVAL FAILED sealed $pol $b"
  done
done
echo STEP9C_DONE
