# Why the query-coverage actor is seed-sensitive: fork diagnosis

Two read-only / actor-only experiments on the sealed fixed-ETT query-coverage
checkpoints (`outputs/pointmaze_ett_query_coverage_20260915_v1`, arm C):

* **Step 1** — at the same 16 held-out fork roots and the canonical goal
  (8.5, 3.5): where the actor's deterministic and sampled actions sit, what the
  critic scores over the whole 2-D action square, which way the critic and the
  BC term push, and how one actual SGD step of each term moves the actor.
  [`scripts/diagnose_f4_actor_fork.py`](../scripts/diagnose_f4_actor_fork.py),
  outputs in `outputs/pointmaze_actor_fork_diagnosis_v1/`.
* **Step 2** — freeze each critic and train three fresh actors on it with the
  unchanged actor objective, identical replay, initializations, batch order
  and budget. [`scripts/train_f4_actor_fixed_critic.py`](../scripts/train_f4_actor_fixed_critic.py),
  outputs in `outputs/pointmaze_fixed_critic_actor_v1/`.

The seed-1 C checkpoint (the 100% seed) is not on disk: only a truncated B
copy and the JSONs came back from the remote node, which no longer exists.
The successful seed used here is therefore seed 0 (mode success 0.845, lower
route 0.78); the failed one is seed 2 (0.305 / 0.00).

## Step 1

### Neither actor outputs "down": both are saturated diagonal corners

| | seed 0 (successful) | seed 2 (failed) |
|---|---|---|
| mode action, 16 roots | (+1.00, -0.99) | (+1.00, -0.75) |
| pre-tanh loc, mean over roots | x +3.3, y -2.4 | x +5.1, y -0.92 |
| tanh slope at the mode (how far loc can still move the action) | x 0.007, y 0.048 | x 0.0002, y 0.48 |
| policy scale (pre-tanh) | 1.6 / 1.4 | 2.2 / 1.1 |
| sampled actions in the DOWN / RIGHT sector | 2% / 5% | 0% / 17% |
| noise-free physics of the mode from the 16 roots: lower / shortcut | 7 / 9 | 5 / 11 |

The mode is "right and down at full throttle" on every root for both seeds;
the seeds differ only in how saturated the y component is. Under the
environment's axis-by-axis sub-stepping the route is a race between the two
components: a fully saturated y drops the point into the lower corridor before
x carries it into the shortcut, a y of -0.75 usually does not. Seed 0's 78%
lower-route rate is the fraction of resets that win this race, and its actor
is frozen there because both components are deep in the tanh tail (gradient
through tanh ~ 0.007 in x); seed 2's y is still in the linear range and is
still being pushed. With a scale of 1-2 in pre-tanh space the *sampled* policy
covers the whole action square almost uniformly, which is why mode and
sample evaluations disagree so much.

### The critic's best region is a diagonal that physically leads into the shortcut

| | seed 0 | seed 2 |
|---|---|---|
| roots with q(DOWN) > q(RIGHT) | 4/16 | 16/16 |
| roots with q(DOWN) > q(mode) | 13/16 | 16/16 |
| argmax of q over the action square | ~(+0.33, -0.30) | ~(+0.47, -0.30) |
| physics of the argmax action: shortcut / lower | 9 / 7 | **13 / 3** |
| roots where max q inside the lower-corridor action set > max q inside the shortcut set | 7/16 | 3/16 |
| q at mode / DOWN / RIGHT / global max | -6.37 / -6.35 / -6.25 / -6.00 | -6.37 / -6.04 / -6.25 / **-5.65** |

Seed 2's critic passes the two-point gate (DOWN above RIGHT on every root),
but the maximum of its score over the whole square is an interior diagonal
action 0.4 above q(DOWN), and from 13 of 16 roots that action enters the
shortcut corridor. The two-point comparison hides this; the actor optimizes
over the whole square.

### Pulls and one real step

At the mode, grad_a q points up-left for both seeds (seed 2: (-0.93, +1.06),
projection on DOWN -1.03): up the slope towards the interior maximum, not
towards (0, -1). One SGD step (lr 3e-4) on an actor copy with real training
batches (10 x 256 from the C replay), displacement of the mode at the roots
projected on DOWN (positive = more downward):

| seed | critic term only | BC term only | full 0.95/0.05 | grad norms critic / BC |
|---|---:|---:|---:|---|
| 0 | -0.0001 | +0.0024 | 0.0000 | 2.1 / 25.7 |
| 2 | +0.0021 | **-0.0758** | -0.0020 | 2.1 / **83.5** |

For seed 2 the BC term, even at weight 0.05, dominates the full gradient and
pushes the mode upward (away from DOWN); the critic term alone would move it
slightly down. For seed 0 nothing moves: saturation. The BC gradient is large
because the tanh-normal log-prob maps clipped boundary actions to
atanh(1 - 1e-6) ~ 7.25 in pre-tanh space; the replay has many such actions
(about 10% of teacher x components are clipped at +/-1).

### Reading

1. The critic's full-square score landscape prefers a diagonal interior
   region that physically leads into the shortcut, on most roots for both
   seeds; the DOWN-vs-RIGHT gate does not see this.
2. Both actors are saturated corner policies; which seed "succeeds" is decided
   by whether its corner wins the axis race, not by a learned route choice.
3. In the actual training gradient at the fork the BC term dominates and
   pushes towards the shortcut; the critic term is small.

## Step 2

Fixed critic, fresh actor, three actor seeds per critic; same replay,
initializations, batch order and 30k x 10 budget; BC 0.05; 200 native
episodes at the sealed reset/action seeds (run on an RTX 4080 node, ~4 min
per run; the actor-seed index fixes the initialization, the buffer's batch
order and the reparameterization keys, so actor seed a sees byte-identical
inputs under both critics).

| critic | actor seed | mode reach | mode lower | sample reach | sample lower | mode action at the 16 roots | physics of the mode: lower / shortcut |
|---|---|---:|---:|---:|---:|---|---|
| seed 0 | 0 | 0.765 | 0.675 | 0.550 | 0.395 | (+1.00, -0.96) | 7 / 9 |
| seed 0 | 1 | 0.775 | 0.690 | 0.520 | 0.370 | (+1.00, -0.94) | 7 / 9 |
| seed 0 | 2 | 0.775 | 0.685 | 0.500 | 0.380 | (+1.00, -0.95) | 7 / 9 |
| seed 0 | mean | **0.772** | 0.683 | 0.523 | 0.382 | | |
| seed 2 | 0 | 0.310 | 0.010 | 0.530 | 0.330 | (+1.00, -0.85) | 6 / 10 |
| seed 2 | 1 | 0.305 | 0.000 | 0.495 | 0.295 | (+1.00, -0.79) | 6 / 10 |
| seed 2 | 2 | 0.610 | 0.455 | 0.545 | 0.360 | (+1.00, -0.91) | 7 / 9 |
| seed 2 | mean | **0.408** | 0.155 | 0.523 | 0.328 | | |

The original arms for reference: seed 0 actor 0.845 / 0.78, seed 2 actor
0.305 / 0.00 (mode reach / lower).

### Reading

* **The successful seed's critic brings out the detour actor every time**:
  three fresh actors land within 0.01 of each other (0.765-0.775 mode
  reach, 0.68-0.69 lower route), all at the saturated corner (+1.00, -0.95).
* **The failed seed's critic does not**: two of three fresh actors reproduce
  the failure exactly (0.31 / 0.00), one reaches 0.61 / 0.455. Actor-seed
  variance appears only under this critic.
* All six new actors converge to the same saturated corner family
  (+1.00, y) as the original arms; they differ only in y: -0.94..-0.96 under
  critic 0, -0.79 / -0.85 / -0.91 under critic 2. The environment's axis race
  turns that difference into all-or-nothing route outcomes (y = -0.91 wins
  the race from 7/16 roots, -0.79 from 6/16, and over the 200 native resets
  0.455 vs 0.00 lower-route use). Every critic still scores DOWN above the
  corner the actor converged to (q(mode) -6.33..-6.57 vs q(DOWN) -6.35 /
  -6.04); the actor never reaches (0, -1) because the BC term pulls towards
  the replay's rightward actions and the critic's own maximum is the interior
  diagonal.
* Sampled behaviour is the same for all six (0.50-0.55 reach, 0.30-0.40
  lower): the scale is 1-2 pre-tanh, so samples spread over the square
  regardless of the critic.

So, in the terms of the decision table: this is the "only the successful
seed's critic brings out the detour actor" case, with a residual actor-seed
sensitivity that is itself a consequence of the critic landscape -- the
failed critic pushes y less far down, and the physics race sits right at that
threshold. The critic's full-action-square scoring (Step 1: its maximum lies
in the shortcut region on 13/16 roots) is the primary suspect, not the actor
optimizer by itself; the two-point DOWN-vs-RIGHT gate is not an adequate
critic criterion, and the saturated-corner policy family means "success" is
not a learned route choice on either seed.
