# PointMaze critic A/B control protocol

This is a bounded critic-only experiment on the frozen C1 PointMaze actor and frozen repaired model. It starts from commit `b3d1423eb0064b63cce63ef5b8194d0f74a18449`. No actor, transition model, emitter, failure head, nominal model, representation architecture, historical artifact, or AntMaze component is modified.

## New roots and episode split

Run 32 new complete 50-step native episodes using the frozen stochastic C1 actor and preregistered reset/action streams. In ascending episode order, retain the first 24 episodes that naturally reach a state with XY in `[1,2) x [3,4)`, are alive, have no earlier hazard landing, have at least three steps left, and admit the fixed zero-noise exits `down=[0,-1] -> cell (1,2)` and `right=[1,0] -> cell (2,3)`. Take the first qualifying time within an episode.

Before any fork outcome is generated, partition the 24 selected episode roots with the fixed split seed into 16 train roots and 8 validation roots. The 16 roots used repeatedly by the preceding matched-fork diagnostic are external reference roots only. They cannot enter training, validation, tuning, or checkpoint selection.

## Common branch data

From each naturally reached root, restore the exact native state and hidden swamp bits. For each fixed first action, generate 8 complete native continuations to the remaining horizon and then resume the same frozen C1 actor. Pair actor Gaussian innovations, action noise, and post-step native swamp-bit draws across the two actions. Save full F4 states, physical states, actions, hidden audit fields, rewards, failure, and discounted return. These native paths are the sole new NCE training/validation trajectories and are identical for A and B.

Separately generate 32 frozen-model continuations per root/action with paired actor innovations, action noise, nominal keys, and model-onset uniforms. Save full states and discounted returns. Per-root ranking labels are `sign(mean model return(down) - mean model return(right))`; no action direction is hard-coded. Native return differences audit whether the frozen-model teacher has the same direction.

## Shared NCE batches

Precompute and seal every training and validation relabel tuple before critic updates. Each 256-row training batch uses each of the 256 train root/action/path identities once. Exactly 64 down and 64 right paths use the fork state at local time zero as anchor; the remaining 128 paths use a later anchor. Each goal is a strictly later full-F4 state from the same native continuation, sampled with probability proportional to `0.95**(future-anchor)`. Rows are then shuffled. This is the existing full `256 x 256` sigmoid-NCE objective: diagonal pairs are positive and every off-diagonal pair is negative.

Validation batches use only the 128 paths from the eight held-out episodes. They are never optimized. A fixed held-out native-NCE loss is reported for the initial critic and both final critics.

## Conditions and fixed budget

Both conditions start from byte-identical C1 critic parameters and Adam state, consume exactly the same sealed NCE batches in the same order, and receive 400 critic updates at learning rate `3e-4`. The actor and model remain frozen.

- A: `L = L_NCE`.
- B: `L = L_NCE + L_rank`, with fixed weight 1.0 and `L_rank = mean(softplus(-y * (q(s,down,G*) - q(s,right,G*))))`, where `y` is the model-return-derived sign for that training episode and `G*` is the canonical stationary task F4.

The ranking term teaches only the preferred action ordering. It does not regress contrastive logits to returns. There is one fixed weight, update budget, split, and final checkpoint; no sweep or best-checkpoint selection is permitted.

## Primary readout and interpretation

At the fixed final update, report teacher-signed critic gaps and ranking accuracy on train roots, held-out new roots, and the old 16 reference roots. Bootstrap episodes, not paths. Also report held-out native NCE loss, native/model teacher agreement, parameter movement, and paired B-minus-A differences.

The declared result is:

- coverage sufficient if A has a clearly positive held-out signed gap;
- ranking adds missing signal if A is not clearly positive, B is clearly positive, and held-out B-minus-A is clearly positive;
- neither resolves if neither critic is clearly positive;
- otherwise inconclusive.

This pilot localizes critic supervision only. It cannot establish that a subsequently trained actor improves complete native success. Any selected training modification still requires new complete confirmation episodes.
