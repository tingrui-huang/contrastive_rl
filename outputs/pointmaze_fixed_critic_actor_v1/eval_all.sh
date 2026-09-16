#!/usr/bin/env bash
# evaluate the six fixed-critic actors: mode and sample, sealed reset/action seeds
cd "$(dirname "$0")/../.."
O=outputs/pointmaze_fixed_critic_actor_v1
for cs in 0 2; do
  args=()
  for a in 0 1 2; do args+=(--ckpt "c${cs}a${a}=$O/critic_s$cs/actor_s$a/final.pkl"); done
  for pol in mode sample; do
    python -m scripts.eval_pointmaze_native_routes "${args[@]}" --episodes 200 --policy $pol \
      --reset-seed-base 29800000 --action-seed-base 29900000 --out "$O/critic_s$cs/native_$pol" > "$O/logs/eval_critic_s${cs}_$pol.log" 2>&1 || echo "EVAL FAILED critic_s$cs $pol"
  done
done
echo EVAL_DONE
