#!/usr/bin/env bash
# Step 9d: critic-seed stability of the two-stage recipe. Freeze every other 30k critic that
# ranks DOWN first (D0, D2 from the diag-fix node; joint-30k balanced seeds 1, 2 from this node)
# and train 3 fresh balanced-BC actors on each (Step 8 procedure); evaluate on the new seeds.
cd ~/contrastive_rl
export PYTHONPATH=$PWD XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
P=~/crlenv/bin/python
F=outputs/pointmaze_diagonal_coverage_fix_v1
J=outputs/pointmaze_balanced_bc_joint_v1/joint
O=outputs/pointmaze_balanced_bc_joint_v1/refreeze
mkdir -p $O/logs
declare -A C=( [D0]=$F/seeds/seed_0/crl/D/final.pkl [D2]=$F/seeds/seed_2/crl/D/final.pkl [b30k_s1]=$J/b30k/seed_1/balanced/final.pkl [b30k_s2]=$J/b30k/seed_2/balanced/final.pkl )
for k in D0 D2 b30k_s1 b30k_s2; do
  for a in 0 1 2; do
    $P scripts/train_f4_actor_fixed_critic.py --critic ${C[$k]} --replay $F/replay_D.npz --actor-seed $a --log-prob acme --random-goals 0 --bc-sampling balanced --bc-cap 0.25 --out $O/critic_$k/actor_s$a > $O/logs/${k}_actor_s$a.log 2>&1 &
  done
  wait
done
for k in D0 D2 b30k_s1 b30k_s2; do
  args=()
  for a in 0 1 2; do args+=(--ckpt "${k}_a${a}=$O/critic_$k/actor_s$a/final.pkl"); done
  for pol in mode sample; do
    $P -m scripts.eval_pointmaze_native_routes "${args[@]}" --episodes 300 --policy $pol --reset-seed-base 31500000 --action-seed-base 31600000 --out "$O/native_new_${pol}_$k" > "$O/logs/eval_new_${pol}_$k.log" 2>&1 || echo "EVAL FAILED new $pol $k"
  done
done
echo STEP9D_DONE
