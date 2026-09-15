# Faithful-actor consistency audit (no training) + uniform-anchor critic purity check

Sealed before execution. Inputs: G1 `outputs/pointmaze_absorbing_integration_20260915_v1`
(frozen critics, plan, 656 fork roots, sealed `fork_eps`, canonical goal) and
the saved G2 bc-sweep actors `outputs/pointmaze_absorbing_actor_sweep_20260915_v1`
(bc = 0.5 reference, 0.2, 0.1, 0.05; O and P). Strictly offline: no actor or
critic training in Part A; Part B trains two critics (400 updates each) from
the sealed initial checkpoint; zero environment calls.

Decision context: G2c / G2c' (global key-balanced actor batches) are retracted
from the method path as a negative ablation (they change the behaviour prior at
the fork and amplify random-policy rows). The faithful candidate is the paper
objective on the original actor rows with bc = 0.2, whose physical first-step
entry into (1,2) is 0.323 (O 0.131) while the multi-step A3 lower-route stays
about 0.22. Part A explains that gap before anything else is run.

## Part A — first-step route classification and continuation, per checkpoint

For each of the 656 fork roots and the 128 sealed `fork_eps` innovations, the
actor's sample `a` at the canonical goal is classified four ways:

1. naive `floor(clip(s + a))`;
2. physical substep landing (ten substeps, X then Y, blocked coordinate
   updates rejected, box bounds and map only, no noise);
3. A1 first transition: the eligible diagonal queried at `(s, a, a)`, one draw
   per sample, projected as in the model (includes its stationary atom);
4. A3 first transition: identical XY to A1 under the same key, plus the
   Manski onset flag (A3 differs from A1 only through the frozen flag).

Reported: the joint table physical x model cell; among samples with physical
(1,2), the model cell distribution; among samples whose naive cell is a wall
cell and whose physical cell is (1,2) (down-right wall diagonals), whether the
model slides into (1,2), stays in (1,3), or lands elsewhere; per-root
disagreement (fraction of samples where the model cell differs from the
physical cell, and P(model (1,2) | physical (1,2))). The eligible Kernel at
zero offsets is also queried once with a nominal draw to document its
action-blindness.

Continuation: A3 rollouts from the fork roots (32 paths per root, horizon 49
masked to the remaining length, common keys with G2/G2b/G2c) with full traces.
For every path: first-step model cell; whether and when y < 2 is reached;
absorbed; for paths whose first cell is (1,2), the outcome class — continued
(reaches y < 2 before revisiting (1,3)/(2,3)), returned (revisits (1,3) or
(2,3) before y < 2), stuck (stays in (1,2) to the end without reaching y < 2),
absorbed. The actor's action at the first model state inside (1,2) of every
such path is classified by its physical landing (down (1,1), up (1,3), other);
the recorded actions at (1,2) states in the training partition are classified
the same way for comparison. A1 rollouts (16 paths) give reach and lower-route
under the optimistic model.

## Part B — uniform-anchor critic purity check

Same as the G1 critic stage except the training pool: 4,096 anchors drawn
uniformly (seed 0, without replacement) from the training partition rows,
instead of the 2,048/1,024/1,024 ordinary/down/right pool; 400 batches of 256
pool anchors drawn uniformly with replacement, future offsets by the sealed
geometric law. P positives are A3 continuations of those anchors (recorded
first action, then the original frozen actor, as in G1); O positives are the
recorded futures. Both critics start from the sealed initial checkpoint with
the actor frozen; unchanged sigmoid NCE, learning rate, Adam state, 400
updates. Read the endpoint and neighbourhood down-minus-right gaps at the 656
roots. This is a purity check on the G1 anchor stratification, not a
replacement for G1.

## What counts as an explanation

The gap is attributed, in order, to (a) classification (physical vs model
first cell disagree), (b) diagonal-motion mismatch (the model does not slide
wall diagonals into (1,2)), (c) continuation failure (the model enters (1,2)
but paths return, stall or are absorbed), using the tables above. No
threshold decides anything here; the output is the attribution and the
numbers. Budget: rollout cap 30,000,000 model transitions; Part B model
output cap 480,000 (the sealed G1 cap). No commit or push.
