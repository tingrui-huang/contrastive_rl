#!/usr/bin/env bash
# Step 8: D1 critic fixed, D replay, Acme log-prob, random_goals 0, bc 0.05 --
# BC rows region-balanced within (state cell, goal cell) groups (cap 0.25),
# plus the 'independent' control (second BC batch, original law). 3 fresh actors each.
cd ~/contrastive_rl
export PYTHONPATH=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.14
F=outputs/pointmaze_diagonal_coverage_fix_v1
O=outputs/pointmaze_fixed_dcritic_actor_bcbal_v1
mkdir -p $O/logs
for arm in balanced independent; do
  for a in 0 1 2; do
    ~/crlenv/bin/python scripts/train_f4_actor_fixed_critic.py --critic $F/seeds/seed_1/crl/D/final.pkl --replay $F/replay_D.npz --actor-seed $a --log-prob acme --random-goals 0 --bc-sampling $arm --bc-cap 0.25 --out $O/critic_D1/$arm/actor_s$a > $O/logs/critic_D1_${arm}_actor_s$a.log 2>&1 &
  done
done
wait
for arm in balanced independent; do
  args=()
  for a in 0 1 2; do args+=(--ckpt "${arm}_a${a}=$O/critic_D1/$arm/actor_s$a/final.pkl"); done
  for pol in mode sample; do
    ~/crlenv/bin/python -m scripts.eval_pointmaze_native_routes "${args[@]}" --episodes 200 --policy $pol --reset-seed-base 29800000 --action-seed-base 29900000 --out "$O/critic_D1/native_${pol}_$arm" > "$O/logs/eval_${arm}_$pol.log" 2>&1 || echo "EVAL FAILED $arm $pol"
  done
done
echo STEP8_DONE
