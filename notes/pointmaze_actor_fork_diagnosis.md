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

**Budget note (added with Step 9).** The sealed joint recipe runs
`max_number_of_steps` 30000 with 50 updates per 50-step iteration, i.e.
30,000 gradient updates for critic and actor (the D-arm checkpoint's Adam
count is 30,000). `train_f4_actor_fixed_critic.py` runs 30,000 iterations
x 10 = 300,000 actor updates. Earlier text in this file called both
"30k x 10"; the figures have been corrected in place. Steps 2-8 are
internally consistent (all 300k), but their actors received ten times the
sealed actor budget, which is why Step 9 runs the joint training at both.

**RNG note (Step 8).** The Step 8 runs called the sampler's law check on the
training buffer before training, which advanced the buffer's stream by 200k
draws, so their critic-term batch order was not byte-identical to Step 7's
(the `independent` control, which shares that shift, reproduces Step 7 and
carries the comparison). `crl/bc_balanced.py` now restores the RNG state.

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
initializations, batch order and 300k actor updates (30k iterations x 10;
see the budget note below); BC 0.05; 200 native
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

## Step 3a: consequence check of the six fork actions

[`scripts/audit_f4_action_consequences.py`](../scripts/audit_f4_action_consequences.py),
outputs in `outputs/pointmaze_action_consequences_v1/`. At each of the 16
held-out roots: DOWN, RIGHT, the two actors' modes, the two critics' argmax
actions; both critics score all six; each action is executed from the root
followed by the sealed observational continuation actor (sampled, paired
keys) in the native environment (64 paired replicates, evaluation only) and
in the fixed ETT with the sealed generation rule.

| first action | mean action | q s0 (rank) | q s2 (rank) | native reach | disc. ret | lower | absorbed | ETT reach | disc. ret | lower | absorbed |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DOWN | (0, -1) | -6.348 (4.9) | -6.041 (2.9) | **0.950** | **9.91** | 1.000 | 0.041 | **0.825** | **7.30** | 1.000 | 0.130 |
| RIGHT | (1, 0) | -6.249 (4.1) | -6.246 (4.2) | 0.305 | 3.54 | 0.056 | 0.699 | 0.267 | 3.25 | 0.040 | 0.738 |
| actor s0 mode | (+1.00, -0.98) | -6.365 (5.6) | -6.629 (6.0) | 0.589 | 6.29 | 0.499 | 0.410 | 0.238 | 2.34 | 0.157 | 0.757 |
| actor s2 mode | (+1.00, -0.72) | -6.137 (3.4) | -6.367 (4.8) | 0.508 | 5.64 | 0.385 | 0.493 | 0.251 | 2.64 | 0.135 | 0.742 |
| critic s0 argmax | (+0.52, -0.32) | **-5.998 (1.0)** | -5.819 (2.1) | 0.493 | 4.87 | 0.469 | 0.507 | 0.319 | 2.95 | 0.261 | 0.673 |
| critic s2 argmax | (+0.49, -0.32) | -6.029 (2.0) | **-5.646 (1.0)** | 0.411 | 4.14 | 0.348 | 0.586 | 0.283 | 2.86 | 0.307 | 0.703 |

Paired per root, the failed critic's argmax is worse than DOWN on 16/16
roots (reach -0.54, discounted return -5.8, absorbed +0.55); the same holds
for the successful critic's argmax and in the ETT (-0.54 reach, 0/16). The
ETT therefore already ranks DOWN >> diagonal ~ RIGHT; it over-predicts
absorption for the diagonal (0.70 vs 0.59) and under-predicts the corner
actions (0.24 vs 0.55), but the preference order matches the environment.

Did the diagonal ever enter NCE?  C-replay transitions within 0.3 of a root
with an action within 0.25 of the action: DOWN 1182 (1090 covered
first-step queries), RIGHT 2564 (1090), critic-argmax actions 105-194 with
**zero** covered first-step queries -- only sparse teacher transitions and
later synthetic steps, whose episodes reach at 0.24-0.28. Verdict: case 2 of
the plan -- the real consequence is bad, the ETT knows, but the region was
never queried, and the critic interpolates a high score between the DOWN
and RIGHT clusters.

## Step 3b: the fix -- diagonal query coverage (arm D)

[`scripts/run_f4_diagonal_coverage_fix.py`](../scripts/run_f4_diagonal_coverage_fix.py),
outputs in `outputs/pointmaze_diagonal_coverage_fix_v1/`. At the same 550
training contexts the sealed generator used (not the 16 evaluation roots),
six diagonal first queries between DOWN and RIGHT -- (0.5,-0.3),
(0.35,-0.45), (0.65,-0.2), (0.5,-0.5), (0.3,-0.3), (0.7,-0.4) -- are
generated with the fixed ETT / nominal / continuation actor exactly as arm
C's queries were; all 3300 paths (reach 0.28, absorbed 0.70) are appended
to the C replay (9900 episodes). Same CRL loss, bc 0.05, the sealed 30k
updates, identical
initialization per seed, 200 native episodes at the sealed seeds. Run on an
RTX 4080 node.

| seed | arm | mode reach | mode lower | sample reach | sample lower | critic d-r (roots down) | critic argmax -> shortcut / lower | actor mode at the roots |
|---|---|---:|---:|---:|---:|---|---|---|
| 0 | C | 1.000 | 1.000 | 0.755 | 0.685 | +0.025 (13/16) | 11 / 5 | (+1.00, -1.00) |
| 0 | D | 0.860 | 0.805 | 0.555 | 0.445 | +0.114 (14/16) | **3 / 13** | (+1.00, -0.72) |
| 1 | C | 0.320 | 0.020 | 0.535 | 0.380 | +0.042 (13/16) | 13 / 3 | (+1.00, -1.00) |
| 1 | D | 0.305 | 0.000 | 0.380 | 0.145 | +0.359 (16/16) | **0 / 16** | (+1.00, -0.17) |
| 2 | C | 1.000 | 0.995 | 0.640 | 0.575 | -0.221 (0/16) | 15 / 1 | (+1.00, -0.96) |
| 2 | D | 1.000 | 1.000 | 0.845 | 0.825 | +0.168 (13/16) | **3 / 13** | (+1.00, -1.00) |
| mean | C | 0.773 | 0.672 | 0.643 | 0.547 | -0.051 | 13.0 / 3.0 | |
| mean | D | 0.722 | 0.602 | 0.593 | 0.472 | +0.214 | 2.0 / 14.0 | |

**The fix repairs the critic and leaves the actor where it was.** With the
diagonal covered, every D critic's best action over the square physically
leads into the lower corridor on 13-16 of 16 roots (C: 3), its mean argmax
moves from the diagonal to (+0.06..+0.31, -0.78..-1.00), and it scores DOWN
0.4-0.6 above the corner the actor ends at (q(DOWN) -5.85/-5.89/-5.94 vs
q(mode) -6.46/-6.32/-6.08). Yet the D actors converge to the same saturated
corner family (+1.00, y) with x at loc +3.6..+5.4, and seed 1 -- whose
critic now ranks (+0.06, -1.00) first on 16/16 roots -- ends at
(+1.00, -0.17), i.e. straight into the shortcut. Mode success is unchanged
in mean (0.773 vs 0.722) and seed 1 fails in both arms.

So the chain has two breaks and the first is now confirmed and closed:

1. **critic**: the diagonal region between the two covered clusters was
   never queried and the critic scored it highest although its consequence
   is bad and the ETT knew it. Covering it fixes the critic's action ranking
   (this is a sampling fix; loss unchanged).
2. **actor**: with a correct critic the actor still does not follow it. The
   x component saturates at +1 (pre-tanh loc 3.6-5.4, tanh slope < 1e-3)
   under the BC term, whose gradient on boundary-clipped replay actions is
   40x the critic term's at the fork; once saturated the critic cannot move
   it, and whether the episode ends on the lower route is decided by how far
   y was pushed before that -- the seed lottery. The next fix belongs on the
   actor side (the BC term on clipped actions / the tanh-normal
   parameterization / the action distribution), not on the critic or the
   ETT.

## Step 4: fixed D critics, fresh actors, actor-only training

Each of the three repaired D critics frozen; three fresh actors per critic;
D replay; the unchanged actor loss, bc 0.05, 300k actor updates; 200 native episodes
at the sealed seeds ([`scripts/train_f4_actor_fixed_critic.py`](../scripts/train_f4_actor_fixed_critic.py)
with `--replay replay_D.npz`; outputs in `outputs/pointmaze_fixed_dcritic_actor_v1/`).

| critic | actor seed | mode reach | mode lower | sample reach | sample lower | actor mode at roots | pre-tanh |loc| x, y |
|---|---|---:|---:|---:|---:|---|---|
| D0 (argmax -> lower 13/16) | 0 / 1 / 2 | 0.305 / 0.305 / 0.305 | 0.000 | 0.46 / 0.49 / 0.47 | 0.24 / 0.29 / 0.25 | (+1.00, -0.69 / -0.45 / -0.74) | 3.9-4.3, 0.6-1.1 |
| D1 (argmax (+0.06, -1.00), 16/16) | 0 / 1 / 2 | 0.305 / 0.305 / 0.305 | 0.000 | 0.38 / 0.47 / 0.44 | 0.12 / 0.26 / 0.21 | (+1.00, -0.30 / -0.51 / -0.37) | 4.0-4.7, 0.3-0.8 |
| D2 (argmax -> lower 13/16) | 0 / 1 / 2 | 0.845 / 0.770 / 0.805 | 0.78 / 0.68 / 0.73 | 0.69 / 0.66 / 0.67 | 0.59 / 0.53 / 0.56 | (+1.00, -1.00) x3 | 3.3-3.8, 4.9-5.5 |

Whether fresh actors take the detour is decided by which critic they are
trained against, not by the actor seed (within a critic the three actors are
near-identical). The critic whose best action is exactly DOWN on every root
(D1) does not bring out the detour at all: its three actors sit at
(+1.00, -0.3..-0.5) and enter the shortcut. D2's actors succeed only as the
usual saturated corner (+1.00, -1.00) with y at loc -5, i.e. by winning the
axis race, not by learning (0, -1). All nine actors saturate x at +1 (loc
3.3-4.7). Under this loss the actor's fork action is essentially the BC mean
action plus whatever offset the critic gradient can add before x saturates;
a correct critic ranking is not sufficient. The remaining break is the actor
objective -- the BC term's dominance (boundary-clipped replay actions through
atanh) and tanh saturation -- not the data and not the critic.

## Step 5: D1 vs D2 under the objective the actor actually optimizes

[`scripts/compare_f4_actor_objective.py`](../scripts/compare_f4_actor_objective.py),
outputs in `outputs/pointmaze_actor_objective_D1_vs_D2/` (landscape figures
for roots 0, 7, 14). The actor minimizes
L(loc) = 0.95 E_{a~tanh N(loc,scale)}[-q(s,a)] + 0.05 E_{a_data}[-log pi(a_data|loc,scale)],
so the relevant critic quantity is Qbar(loc), the sampled-action mean score at
the policy's own width, and the BC term is evaluated with the exact
tanh-normal log-prob on the D-replay actions taken within 0.15 of each root.

| critic | raw argmax region (lower/shortcut/other) | Qbar argmax region at the actor's scale | at scale 0.3 | total-objective argmin region | fresh actor mode |
|---|---|---|---|---|---|
| D1 | 16/0/0 | **6/10/0** | 16/0/0 | **2/14/0** | (+1.00, -0.30) |
| D2 | 13/3/0 | 7/9/0 | 16/0/0 | 7/9/0 | (+1.00, -1.00) |

At the fresh actors' loc (mean over roots), the LOSS gradient dL/dloc_y of
the critic term is -0.035 for D1 and +0.027 for D2, of the BC term +0.027 for
D1 and -0.039 for D2; descent moves loc_y by -lr * dL/dloc_y, so under D1 the
critic term pushes the actor UP (away from DOWN) and the BC term pushes it
DOWN, and under D2 the roles are reversed; the totals are ~0. In this local,
fixed-scale, loc-only picture (BC aggregated over an XY neighbourhood, no F4
history / goal conditioning, shared parameters and the learned scale
ignored) both actors sit near a minimum of the local objective -- D1's at the
right plateau (+1, -0.3..-0.5), D2's at the corner (+1, -1), neither at
DOWN. That supports an objective conflict; it does not prove the actors are
at the minimum of the full training objective.

Mechanism (scale table in the REPORT): 13% of the x components and 4% of the
y components of the replay actions near the fork are boundary-clipped
(atanh ~ +/-7.25); the BC NLL is 97-170 nats at scale 0.3, ~1-2 at scale
1.5-2, and still decreasing at 3, so the observed width 1.5-2 is a trade-off
between BC and the sampled-average critic term rather than something BC sets
alone, and the boundary handling is one suspect for BC's preference for
width alongside action diversity, mixed behaviour and the single-Gaussian
form (not separated here). With that width,
D1's narrow DOWN peak (the x ~ 0 column of the action square) averages to
only ~0.07 above the broad right plateau, worth 0.07 in the objective, while
BC still charges the DOWN loc ~3 nats more than an x ~ +4 loc (0.15 in the
objective) because 13% of the data x's are +1. The objective therefore
prefers the plateau, i.e. the shortcut. D2's critic puts its high region in
the bottom-right corner, where BC's x = +1 preference is also satisfied, so
both terms agree and the actor saturates at (+1, -1); its detours come from
the axis race, not from choosing (0, -1). At a sharp scale (0.3) both
critics' Qbar argmax would be the lower route on 16/16 roots.

Chain summary: the coverage fix made the critic's argmax DOWN, but the
objective formed by the BC term, the policy width and the critic's scores
locally prefers a policy that performs worse; a narrow DOWN optimum cannot
win the sampled-average objective against a broad plateau, and only a critic
that happens to score the whole bottom-right high (D2) produces a detour
actor, via saturation. The boundary handling of the BC log-prob is a
concrete suspect but has not been isolated; Step 6 tests it directly by
restoring the original Acme 0.4.0 rule.

## Step 6: restore the original Acme 0.4.0 boundary log-prob (D1 fixed, fresh actors)

The port's `tanh_normal_log_prob` clips boundary actions to 1 - 1e-6 and
scores them at the density of atanh(1 - 1e-6) ~ 7.25; the original
contrastive_rl actor (dm-acme 0.4.0 `TanhTransformedDistribution`,
threshold 0.999) scores an action in the band [0.999, 1] with the average
density of the whole tail, log P(x >= atanh 0.999) - log 0.001, symmetric on
the left. `crl.networks.tanh_normal_log_prob_acme` implements the Acme rule
and `make_networks(log_prob_mode='acme')` selects it (default 'clip', so
nothing else changes). [`scripts/test_tanh_log_prob_acme.py`](../scripts/test_tanh_log_prob_acme.py)
checks it against an independent float64 reference: interior values (1e-4),
values at +/-1, +/-0.9995 and the threshold (rel 1e-6), band value = the
quadrature average density over [0.999, 1], interior + bands integrate to
one, and d/dloc, d/dscale against central finite differences (rel 3e-5).
For loc 0 the log-prob of action +1 is -14.1 (clip) vs -2.6 (Acme) at scale
1, and -278 vs -77 at scale 0.3.

Same protocol as Step 4 for critic D1 (D replay, three fresh actors with
the same initializations, batch order, Adam and 300k actor updates, bc 0.05,
scale learned), the only change being the BC log-prob rule:

| log-prob | actor seed | mode reach | mode lower | sample reach | sample lower | mode at roots | pre-tanh |loc| x, y | scale x, y |
|---|---|---:|---:|---:|---:|---|---|---|
| clip (Step 4) | 0 / 1 / 2 | 0.305 x3 | 0.000 x3 | 0.375 / 0.470 / 0.440 | 0.115 / 0.255 / 0.205 | (+1.00, -0.30 / -0.51 / -0.37) | 4.0-4.7, 0.3-0.8 | 2.1-2.2, 1.1-1.3 |
| acme (Step 6) | 0 / 1 / 2 | 0.305 x3 | 0.000 x3 | 0.345 / 0.355 / 0.345 | 0.085 / 0.080 / 0.095 | (+1.00, -0.20 / -0.19 / -0.09) | 3.7-3.8, 0.1-0.2 | 1.6-1.7, 0.7-0.8 |

No improvement; the sampled policy is slightly worse. The Acme rule does
change the policy shape as predicted -- the scale shrinks from (2.2, 1.1) to
(1.65, 0.75) and loc_x settles at 3.7-3.8 (the tail band is covered once
loc_x >= atanh 0.999) instead of 4-4.7 -- so the clip rule was inflating
the width, but the narrower policy tracks the BC mean action (data y ~ -0.24
near the fork) even more tightly and the critic term still does not move y.
Under the plan's own criterion, the boundary implementation difference is
not the cause of D1's failure. What remains is the conflict between the BC
term on the data's mixed, x = +1-heavy behaviour and the critic, and the
single-Gaussian form on a multimodal action set; neither is separated yet.

## Step 7: original offline goal pairing (random_goals 0) on top of Step 6

`train_f4_actor_fixed_critic.py --random-goals 0` replaces the hard-coded
random_goals 0.5 branch (batch doubled, second half with rolled goals) by
crl/losses.py's random_goals 0 branch (each state with its own relabeled
future goal only); Acme log-prob, D1 critic, D replay, bc 0.05, the same three
initializations, batch order, Adam and 300k actor updates.

| arm | seed | mode reach | mode lower | sample reach | sample lower | mode at roots |
|---|---|---:|---:|---:|---:|---|
| acme, rg 0.5 (Step 6) | 0/1/2 | 0.305 x3 | 0.000 x3 | 0.345 / 0.355 / 0.345 | 0.085 / 0.080 / 0.095 | (+1.00, -0.20 / -0.19 / -0.09) |
| acme, rg 0 (Step 7) | 0/1/2 | 0.305 x3 | 0.000 x3 | 0.340 / 0.325 / 0.345 | 0.065 / 0.050 / 0.065 | (+1.00, -0.12 / -0.10 / -0.24) |

No improvement. Goal sensitivity ([`scripts/audit_f4_actor_goal_sensitivity.py`](../scripts/audit_f4_actor_goal_sensitivity.py),
`outputs/pointmaze_actor_goal_sensitivity_v1/`): every fixed-critic actor
(Steps 4, 6, 7) is strongly goal-conditioned at the fork -- canonical goal
(8.5, 3.5): (+1.00, -0.1..-0.3); a goal inside the lower corridor (1.5, 1.0):
(-1.00, -1.00); a goal inside the shortcut corridor: (+1.00, -0.4..-0.6);
the start cell: (-1.00, +1.00); std 0.6-0.7 per component over 64 replay
goals. The actor has learned "move towards the goal": with hindsight
relabeling the data actions are aligned with the direction to the relabeled
goal, the majority of fork data (shortcut teacher episodes) pairs a rightward
action with goals to the right, and the single-Gaussian BC averages to that
side; the detour requires moving against the goal direction at the fork,
which the BC term penalizes and the correct critic does not overcome at
0.95. Neither the boundary rule nor the goal pairing changes this.

## Step 8: balanced BC rows inside (state, goal) groups

Balanced sampling had been tried twice before on PointMaze and both times on
*every* term at once: the Aug-29 windy-swamp sweep balanced the shared replay
over global (state cell, action sector) buckets for critic and actor (commit
434cb0b; all nine arms a null, 8d3fc0d), and G2c balanced the whole actor
batch over (cell, landing cell) keys (`outputs/pointmaze_absorbing_balanced_g2c_20260915_v1`;
retracted because 96 of 134 keys are random-walker keys, so the balanced
batches were 80% random rows and navigation collapsed). The BC term on its
own had never been balanced.

`train_f4_actor_fixed_critic.py --bc-sampling balanced` (class
`GroupBalancedBCSampler`) keeps everything of Step 7 -- D1 critic frozen, D
replay, Acme log-prob, random_goals 0, bc 0.05, the same three
initializations, the same critic-term batch stream up to the law-check shift
described in the RNG note above, and 300k actor updates -- and draws the BC term's 256 rows per update
from a second stream instead:

* every (episode, anchor i, future j) triple of the replay is enumerated with
  the weight the buffer's variable-length sampler gives it, `1/N * 1/(L_e-1) *
  gamma^(j-i) / sum_k gamma^k` (12,292,500 triples; a 200k-draw check against
  the buffer agrees to a max bucket-share deviation of 6e-4);
* group = (floor cell of the anchor's newest frame, floor cell of the
  relabeled goal's newest frame), 282 groups; region = the recorded action's
  angle in 8 sectors rotated half a width (cardinals and diagonals at bin
  centres) plus a wait bucket for |a| < 0.1, 2,211 non-empty buckets;
* inside a group a region's share is ceiled at 0.25 and the group is
  renormalised, so the (state, goal) marginal of the BC rows is exactly the
  original one (max deviation 1e-12) and only the conditional over action
  regions is flattened; regions under the ceiling keep their natural relative
  frequency, which is what keeps the random-walker regions from being
  amplified (max multiplier 3.25, effective sample size 5.08M -> 4.89M).
  The cap was fixed at 0.25 after tabulating the fork composition for caps
  0.2 / 0.25 / 0.33 / 0.5 and strict uniform; only 0.25 was trained.

No action label, no trajectory and no coefficient changes. A control arm
`--bc-sampling independent` draws the BC rows from the same second stream
with the original weights, to separate "a separate BC batch" from "a balanced
BC batch".

At the fork group that the task goal queries, (1,3) -> goal cell (8,3), the
BC rows change from R 0.55 / D 0.20 / DR 0.17 / other 0.08 to R 0.36 / D 0.29
/ DR 0.24 / other 0.11; the goal-(2,3), (3,3), (4,3) groups move from R
0.40-0.54 to 0.32-0.37 with DR at 0.32-0.37.

| arm | seed | mode reach | mode lower | sample reach | sample lower | mode at roots (task goal) | scale | bc_nll |
|---|---|---:|---:|---:|---:|---|---:|---:|
| shared BC rows (Step 7) | 0/1/2 | 0.305 x3 | 0.000 x3 | 0.340 / 0.325 / 0.345 | 0.065 / 0.050 / 0.065 | (+1.00, -0.12 / -0.10 / -0.24) | 0.81 / 0.82 / 0.81 | 1.21 / 1.03 / 1.18 |
| independent BC rows | 0/1/2 | 0.305 x3 | 0.000 x3 | 0.340 / 0.345 / 0.335 | 0.055 / 0.060 / 0.065 | (+1.00, -0.12 / -0.10 / -0.20) | 0.82 / 0.85 / 0.80 | 1.12 / 1.02 / 1.05 |
| **balanced BC rows** | 0/1/2 | **1.000 x3** | **1.000 x3** | 0.930 / 0.930 / 0.910 | 0.885 / 0.900 / 0.875 | (+0.31, -1.00) / (+0.26, -1.00) / (+0.32, -1.00) | 0.80 / 0.83 / 0.84 | 1.38 / 1.38 / 1.28 |

(`outputs/pointmaze_fixed_dcritic_actor_bcbal_v1/`; 200 paired native
episodes with the reset and innovation seeds of Steps 4-7.)

All three balanced actors take the lower route on every mode episode and
reach the goal on every one; with sampled actions they detour 0.875-0.900
and reach 0.91-0.93 (the residual 10-12% are shortcut samples; 7-9% of
all episodes end absorbed in the swamp). The independent control reproduces Step 7 to within
one episode, so the whole effect is the reweighting, not the decoupled batch.
Under the task goal the mode moves from a saturated +x with a shallow -y to
a saturated -y with x +0.3 -- "down" through the gap rather than the corner
race of Step 4 -- while the lower-corridor goal still gives (-1, -1), the
shortcut-corridor goal (+1, -0.4) and the start (-1, +0.9)
(`outputs/pointmaze_actor_goal_sensitivity_v1/REPORT.md`): the actor is still
goal-conditioned, it has stopped averaging to the majority side at the fork.
Policy scale and |loc| are unchanged (0.80-0.84, 1.6-1.7); the BC NLL is
higher by 0.25-0.35 because the balanced rows are harder to fit with one
Gaussian, as expected.

What this does and does not show. With a critic that ranks DOWN first at the
fork (D1), flattening the BC term's route imbalance at 0.05 is enough for the
critic term to decide; this is consistent with Step 5, where the narrow DOWN
peak lost to the broad right plateau by a margin the BC pull supplied. It does
not show what the balanced BC does with a critic that prefers the shortcut
(the C / O critics), nor what happens when the critic is trained jointly
rather than frozen -- the critic batches were deliberately left alone here.
Those are the two attribution controls to run next; nothing else was changed
or tuned.

## Step 9: attribution on the C critics, joint training, new evaluation seeds

Driver [`scripts/run_f4_balanced_bc_joint.py`](../scripts/run_f4_balanced_bc_joint.py);
the balancing rule is now a learner option (`crl/bc_balanced.py`,
`Config.bc_sampling`, `Config.log_prob_mode`); outputs in
`outputs/pointmaze_balanced_bc_joint_v1/` (REPORT.md, results.json,
failure_audit/). Everything below uses Acme log-prob, random_goals 0, bc
0.05, the D replay and cap 0.25. "New" = 300 paired native episodes on reset
seeds from 31.5M / innovation seeds from 31.6M, never used before; "sealed" =
the 200-episode protocol of Steps 4-8. All runs on one RTX 5060 Ti node
(a first attempt with six parallel processes exceeded the pod's 15 GB and
restarted it; everything was redone with three).

### 9a. Attribution: the same balanced BC rows on the sealed C critics

Fixed critic, three fresh actors, 300k actor updates -- Step 8 with the
critic swapped.

| critic | mode lower, new (seeds 0/1/2) | mode reach, new | sample lower, new | mode lower, sealed |
|---|---|---|---|---|
| C seed 2 (the failed sealed seed) | 0.000 / 0.000 / 0.000 | 0.363 x3 | 0.42 / 0.40 / 0.40 | 0.000 x3 |
| C seed 0 | 0.223 / 0.000 / 0.263 | 0.50 / 0.36 / 0.52 | 0.45 / 0.36 / 0.42 | 0.25 / 0.00 / 0.30 |
| D1 (Step 8) | -- | -- | -- | 1.000 x3 |

The balanced BC rows do not turn a shortcut-preferring critic into a detour
actor. The gain of Step 8 depends on the repaired critic; the balancing only
removes the BC pull that a correct critic could not overcome.

### 9b. Joint training: critic and actor together, shared vs balanced BC rows

`crl.train` on the D replay, NCE critic unchanged, actor's critic term on the
buffer batch, BC term on its own rows (`bc_sampling` shared / balanced),
three training seeds, sealed budget (30k updates) and 10x (300k). Because
the NCE critic does not depend on the actor, the two arms' critics are the
same training up to GPU floating-point non-determinism; only the actor
differs.

| budget | arm | mode lower, new (seeds 0/1/2) | sample lower, new | critic on the 16 roots: DOWN-RIGHT, roots down, argmax -> shortcut/lower |
|---|---|---|---|---|
| 30k | shared | 0.000 / 0.000 / 1.000 | 0.16 / 0.09 / 0.74 | +0.20 13/16 3/13; +0.25 16/16 2/14; +0.17 16/16 0/16 |
| 30k | balanced | 0.000 / 0.000 / 0.377 | 0.27 / 0.14 / 0.60 | +0.28 16/16 0/16; +0.25 16/16 2/14; +0.17 16/16 0/16 |
| 300k | shared | 0.000 / 0.000 / 0.000 | 0.15 / 0.03 / 0.10 | +0.89 16/16 0/16; +1.07 16/16 1/15; +0.64 15/16 3/13 |
| 300k | balanced | 0.000 / 0.000 / 0.000 | 0.05 / 0.06 / 0.07 | +0.70 16/16 0/16; +0.76 16/16 0/16; +1.06 16/16 0/16 |

Sealed-seed numbers are the same (0.305 / 0.000 everywhere except 30k seed
2). Joint training does not produce a stable detour at either budget, with
or without the balanced rows, although every 300k critic ranks DOWN first on
the roots by a wider margin than D1.

The seed-1 30k critic is the same training as D1 (same replay, seed, batch
stream) and on the RTX 4080 it ranked DOWN 16/16 with +0.36; on this node
it is 16/16 with +0.25 and its argmax leads to the shortcut on 2 roots. The
fork ranking of a 30k critic is sensitive to floating-point differences
between GPUs.

### 9c. Where the joint runs fail: the actor, not (only) the critic

Freezing the jointly trained critics and training fresh balanced-BC actors on
them (the Step 8 procedure; `refreeze/`):

| frozen critic | mode lower, new (3 fresh actors) | mode reach, new | sample lower, new | mode lower, sealed |
|---|---|---|---|---|
| joint 30k balanced seed 0 (its own joint actor: 0.000) | **1.000 / 1.000 / 1.000** | 1.000 x3 | 0.87 / 0.84 / 0.89 | 1.000 x3 |
| joint 300k balanced seed 0 (its own joint actor: 0.000) | 0.000 / 0.000 / 0.000 | 0.363 x3 | 0.02 / 0.05 / 0.06 | 0.000 x3 |

So the sealed-budget critic is usable: with the actor started fresh against
the finished critic it detours on every mode episode of the new seeds, as
D1 did in Step 8. Trained jointly, the same critic's actor never detours.
`failure_audit/REPORT.md` shows why: at the fork the joint actors sit at
pre-tanh loc x = +1.8..+5.2 (mode x = 0.94..1.00, tanh slope 0.000-0.11)
while their critics score DOWN 0.2-0.8 above the actor's own mode -- the
actor committed to RIGHT while the critic was still immature (BC majority
plus a noisy critic term) and a saturated tanh cannot be pulled back. The
two-stage actors rest at (+0.2..+0.4, -1.00) with slope ~0.9 in x.

The 300k critic is different: it fails even frozen, its fresh actors settle
at (+1.00, +0.2..+0.3). Its fork-root ranking is the strongest of all, so
the 16-root metric does not capture what makes a critic usable. A crude
look along the approach path (Qbar(DOWN) - Qbar(RIGHT) at policy width over
x 0.5-1.45, y 3.2-3.8, moving and still; `failure_audit` part B) does not
separate the cases either: D1 prefers DOWN in 13/30 cells, the working 30k
seed-0 critic in 3/30, the failing 300k one in 1/30, C seed 2 in 1/30. What
property of the 300k critic blocks the actor is open.

### What Step 9 establishes

1. The balanced BC rows need a critic that already prefers the detour
   (9a) -- they are not a substitute for the critic repair.
2. With such a critic frozen and the actor trained fresh, the detour is
   stable on new evaluation seeds: 6 of 6 actors on two different 30k critics
   (D1 on the 4080, joint seed 0 on the 5060 Ti) at mode 1.000, sample
   0.84-0.90, on 300 unseen resets each.
3. Training critic and actor together does not work at either budget: the
   actor saturates towards the shortcut before the critic is ready (30k), and
   at 300k the critic itself stops being usable for the actor for a reason
   not yet identified.
4. The fork ranking of a 30k critic on 16 roots is not reproducible across
   GPUs and is not sufficient as a usability criterion.

### 9d. Critic-seed stability of the two-stage recipe

Every other 30k critic frozen, three fresh balanced-BC actors each
(`refreeze/`; new seeds, 300 episodes):

| frozen critic | node | 16-root ranking (d-r, roots down, argmax -> shortcut) | mode lower (3 actors) |
|---|---|---|---|
| D1 (Step 8) | RTX 4080 | +0.36, 16/16, 0 | 1.000 / 1.000 / 1.000 |
| D0 | RTX 4080 | +0.11, 14/16, 3 | 1.000 / 1.000 / 1.000 |
| D2 | RTX 4080 | +0.17, 13/16, 3 | 0.813 / **0.007** / 0.880 |
| joint 30k balanced seed 0 | RTX 5060 Ti | +0.28, 16/16, 0 | 1.000 / 1.000 / 1.000 |
| joint 30k balanced seed 1 (same training as D1) | RTX 5060 Ti | +0.25, 16/16, 2 | **0.000 / 0.000 / 0.000** |
| joint 30k balanced seed 2 | RTX 5060 Ti | +0.17, 16/16, 0 | 0.880 / 1.000 / **0.000** |

13 of 18. Actor-seed variance reappears under the marginal critics (D2,
joint seed 2), and the critic side is fragile: seed 1 trained on the 4080
(D1) gives 3/3, the same training on the 5060 Ti gives 0/3. The 16-root
ranking does not predict the outcome (D0 ranks worst among the successes).

### 9e. The fair plain-CRL control

Plain CRL critics (NCE on the full 6600-episode 0.05-rung dataset, sealed
30k budget, no ETT data) frozen, then the same balanced-BC two-stage actors
on the original dataset (`control_O/`, driver stage `control`):

| O critic seed | q(DOWN) - q(RIGHT) at the roots | mode lower (3 actors) | sample lower |
|---|---|---|---|
| 0 | +0.34 | 0.763 / 0.547 / 0.633 | 0.51 / 0.44 / 0.45 |
| 1 | -0.49 | 0.000 / 0.000 / 0.000 | 0.01 x3 |
| 2 | -0.04 | 0.000 / 0.000 / 0.000 | 0.02 / 0.09 / 0.03 |

So plain CRL with the same actor treatment is not zero: one critic seed of
three prefers DOWN at the fork and its actors detour 0.55-0.76 (all of them
corner policies, see 9f). Against the ETT-D critics (13/18 actors at
>= 0.81, 9/18 at 1.000) the ETT data still makes the difference, but the
earlier statement "O never detours" was a statement about the joint recipe,
not about O critics.

### 9f. Whole-policy objective: does training prefer the failed actor?

[`scripts/audit_f4_policy_objective.py`](../scripts/audit_f4_policy_objective.py)
(`policy_objective_audit/`): for the D2 trio (0.81 / 0.01 / 0.88), the
exact training objective 0.95 E[-f(s, a~pi, g)] + 0.05 E[-log pi(a_data|s,g)]
on identical rows -- critic rows from the buffer's relabeling law, BC rows
from the balanced law, the actor's own width, Acme log-prob, shared
innovations -- on the full distribution and conditionally on the approach,
fork and fork -> goal-(8,3) rows; then the same policies under every other
critic.

Under their own critic the three totals differ by at most 0.3% of their
value and the sign flips across subsets: on the full distribution the two
successful actors are lower by 0.002-0.003 (s.e. 0.0003), on the fork rows
the failed actor is lower by 0.002-0.007 (s.e. 0.001). No other critic
separates them either (differences -0.04..+0.03 on totals of 2.4-3.1).
The three policies are the same policy: their fork modes under the task
goal are (+0.95, -1.00), (+0.94, -1.00), (+0.96, -1.00). The route is
decided by the axis-by-axis substep race at the action boundary, and at x
0.94-0.96 that race flips between resets. The training objective does not
reward the failure; it is flat across a family of saturated policies whose
outcomes the environment separates.

The same classification over every two-stage actor
(`policy_objective_audit/two_stage_actor_modes.json`):

| family (fork mode under the task goal) | actors | mode lower |
|---|---|---|
| DOWN: y = -1.00, x 0.2-0.4 | D1 x3, joint s0 x3 | 1.000 x6 |
| CORNER: y = -1.00, x 0.65-1.00 | D0 (x 0.65-0.69), joint s2 (0.79-0.91), D2 (0.94-0.96), O seed 0 (1.00) | 1.000 x3; 0.88/1.00/0.00; 0.81/0.01/0.88; 0.76/0.55/0.63 |
| RIGHT: y > -0.5 | joint s1, joint 300k, O seeds 1-2, C | 0.000 x15 |

Within the CORNER family the detour rate is a steep function of x: 1.000 at
x <= 0.7, all-or-nothing at 0.8-0.96, 0.55-0.76 at 1.00. So there are two
kinds of "success": the DOWN family (robust across actor seeds and evaluation
seeds) and the CORNER family (a physics race that actor seeds and resets
decide). Only D1 and joint seed 0 produce the DOWN family. What separates
those two critics from the CORNER-producing ones is the next question; the
16-root DOWN-vs-RIGHT margin does not (D0 +0.11 -> corner at x 0.67 and
1.000, D2 +0.17 -> corner at 0.95). A smoothed DOWN-vs-CORNER preference at
the policy width (Qbar at loc (0.3, -6) minus Qbar at (5, -5)) fits the
DOWN/CORNER split of the D critics (D1 +0.54, joint s0 +0.59 vs D0 +0.23,
joint s2 +0.28, D2 -0.20) but not the RIGHT family (joint s1 +0.31, joint
300k +0.73), which fails for a reason this audit does not reach.

### 9g. The RIGHT family: the probe measured a goal frame the actor never trains on

Three read-only audits on the two unexplained critics (joint 30k balanced
seed 1 = R1, joint 300k balanced seed 0 = R2) against two DOWN-family
critics (joint 30k seed 0 = D-a, D1 = D-b), their fresh balanced-BC actors
(3 each), and the training rows themselves:
[`scripts/audit_f4_right_family.py`](../scripts/audit_f4_right_family.py),
[`scripts/audit_f4_goal_frames.py`](../scripts/audit_f4_goal_frames.py),
[`scripts/audit_f4_jitter_margin_all_critics.py`](../scripts/audit_f4_jitter_margin_all_critics.py);
outputs in `outputs/pointmaze_balanced_bc_joint_v1/right_family_audit/`
(CPU, this machine).

**1. Not the optimizer.** The exact training objective (0.95 critic term on
the buffer law + 0.05 balanced BC term, 40k shared rows, 16 samples per row)
is LOWER for each RIGHT critic's own actors than for the DOWN actors: R1
-0.089 on the full distribution, -0.030 on the fork rows, -0.012 (s.e.
0.005) on the fork -> goal-(8,3) rows; R2 -0.489 / -0.162 / -0.243.  Under
the two DOWN critics the DOWN actors are the ones preferred.  Training
found what these critics ask for.

**2. The actor never sees the canonical goal.** The task goal
tile((8.5, 3.5), 4) is a stationary stack.  Of the replay's fork -> (8,3)
rows (mass 0.0069, 13% of the fork rows), 83% carry a goal frame that
jitters inside the goal cell (median frame-to-frame displacement 0.22 --
the teacher and the continuation actor keep moving around the goal), 14%
an arrival frame along the middle route, 3% an arrival frame from below,
and 3% a frame with displacement < 0.05.  All twelve actors output the
same mode on the canonical goal as on the jitter frames (D2 table of
`goal_frames_REPORT.md`), so the task-goal action is the jitter-frame
action.  The route-identified arrival frames are handled correctly by every
critic and every actor (arrival-middle -> RIGHT, arrival-below -> DOWN).

**3. What splits the families is the policy-width margin on the jitter
frames.** On those rows, Qbar(DOWN loc) - Qbar(RIGHT loc) at width 0.75:

| critic | family | 16 roots x canonical, point / width | jitter frames, point / width |
|---|---|---:|---:|
| D1 (4080) | DOWN | +0.36 / +0.17 | +0.24 / **+0.11** |
| joint 30k s0 | DOWN | +0.28 / +0.14 | +0.14 / **+0.05** |
| D0 | CORNER x0.67 | +0.11 / +0.03 | +0.03 / **-0.01** |
| joint 30k s2 | CORNER x0.85 | +0.17 / +0.15 | +0.03 / **+0.06** |
| D2 | CORNER x0.95 | +0.17 / +0.25 | +0.13 / **+0.21** |
| O seed 0 | CORNER x1.00 | +0.34 / +0.22 | +0.35 / **+0.22** |
| joint 30k s1 (R1) | RIGHT | +0.25 / +0.01 | +0.12 / **-0.05** |
| joint 300k s0 (R2) | RIGHT | +0.71 / +0.20 | -0.07 / **-0.28** |
| O seed 1 | RIGHT | -0.49 / -0.36 | -0.45 / **-0.35** |
| O seed 2 | RIGHT | -0.04 / +0.11 | -0.18 / **+0.03** |
| C seed 2 | RIGHT | +0.20 / +0.02 | -0.11 / **-0.16** |
| C seed 0 | RIGHT (9a) | -0.10 / -0.11 | -0.27 / **-0.20** |

Every RIGHT critic has a jitter-frame width margin <= +0.03, every DOWN and
CORNER critic >= +0.05 except D0 (-0.01, whose actors sit at x 0.67 and win
the race).  The 16-root canonical point margin -- the criterion used since
Step 1 -- ranks R1 (+0.25) and R2 (+0.71) among the best critics.  Two
different reasons, both visible in the cross probes (`goal_frames_REPORT.md`
D2): for R1 the +0.25 is a narrow peak, the width margin at the very same
roots and goal is +0.01; for R2 the DOWN preference is real at the policy
width but lives on the stationary goal frame only (+0.71 / +0.20 on the
canonical goal, +0.03 / -0.24 with the replay's jitter frames substituted at
the same 16 roots).  R2's margin falls monotonically with the goal frame's
displacement (+0.14 at < 0.05, +0.10, -0.03, -0.29 at >= 0.3) while the
DOWN critics are flat across the bins (+0.06 / +0.12); the 300k critic
reads the goal frame's velocity as a route cue, and among the jitter frames
the moving ones come mostly from shortcut survivors (recorded fork actions
on jitter-goal rows: R 0.55 / D 0.20 / DR 0.17).

**4. Why the margin is this small.** The data's own discounted answer at
the fork is weak: under the relabeling law P(goal cell (8,3) | fork, D) =
0.25 against 0.19 for R (original episodes alone 0.14 vs 0.46, the C
queries 0.32 vs 0.12, the diagonal queries 0.10 vs 0.07).  The lower route
is longer, gamma 0.95 discounts it, the DOWN continuation paths reach
0.3-0.8, and the 1200 random-walker episodes supply DOWN actions at the
fork that go nowhere.  A target of a quarter of a nat is inside the seed
and GPU noise of a 30k critic and inside what smoothing at the policy
width flips.

So: the actor at the task goal follows the critic's policy-width preference
on the replay's jittering (8,3) goal frames; the 16-root canonical-goal
point probe measures an extrapolation the actor is never trained on and
does not predict the family (10 of 12 by sign, 11 of 12 with a +0.04
threshold on the width margin).  The DOWN-vs-CORNER split within the
positive side is not explained by this margin (D2 +0.21, O0 +0.22 are
corner critics) and remains the 9f question.

## Step 10: corner / edge query coverage (arm E) -- a net regression

[`scripts/run_f4_corner_coverage_fix.py`](../scripts/run_f4_corner_coverage_fix.py),
`outputs/pointmaze_corner_coverage_fix_v1/`. Six bottom-edge / right-edge
first queries were audited against the native physics at the 16 held-out
roots (64 paired replicates, `audit/REPORT.md`): natively the bottom edge is
a smooth ramp (lower route 1.00 at (0.3,-1), 0.85 at (0.6,-1), 0.55 at
(0.85,-1), 0.46 at the corner, 0.21 at (1,-0.6)); the fixed ETT gets
(0.3,-1) right (lower 1.00), is direction-right but over-pessimistic on the
corner (lower 0.15, absorbed 0.77 vs native 0.46 / 0.46 -- the Step 3 kind
of bias), and flattens everything with x >= 0.6 to RIGHT-like outcomes
((0.6,-1): native lower 0.85, ETT 0.16). By the user's decision only the
two queries the ETT gets right were generated: replay_E = replay_D + 550 x
(0.3,-1) paths (reach 0.76, lower 0.91) + 550 corner paths (reach 0.23,
lower 0.07), 11,000 episodes.

Five 30k critics (sealed recipe), three fresh balanced-BC actors each,
300 new-seed episodes:

| E critic | d-r at the roots | Qbar DOWN-CORNER | actor family (fork mode) | mode lower |
|---|---|---|---|---|
| seed 0 | -0.10, 0/16 | +0.43 | RIGHT x3 (+0.9, -0.65) | 0 / 0 / 0 |
| seed 1 | +0.02, 11/16 | +0.56 | RIGHT x3 (+1.0, +0.2) | 0 / 0 / 0 |
| seed 2 | +0.35, 16/16 | +0.60 | DOWN x3 (+0.3, -1.0) | 1.000 x3 |
| seed 3 | -0.29, 0/16 | +0.35 | RIGHT x3 (+1.0, 0.0) | 0 / 0 / 0 |
| seed 4 | +0.10, 14/16 | +0.68 | RIGHT x3 (+1.0, -0.35) | 0 / 0 / 0 |

3/15 against the D series' 13/18. The corner data did exactly what it was
added for -- every E critic separates the gentle-down action from the corner
(+0.35..+0.68 at the policy width, 16/16 roots; no CORNER-family actor
remains) -- and the DOWN-vs-RIGHT separation got worse: mean raw margin at
the roots +0.02 (D series +0.21; the same recipe on this node +0.23). The
loss is not a generalization gap: at the 550 training contexts themselves,
where 550 added paths say (0.3,-1) reaches the goal 0.76 and RIGHT 0.25,
three of five E critics score RIGHT at or above the gentle-down action
(E3: -5.86 vs -6.21). The NCE critic is not fitting the action dependence
of the fork rows it is given; adding rows changed which failure the seeds
fall into (CORNER -> RIGHT), not the fraction that fail. Whether this is
the fork rows' small share of the batches (5% of anchors, 0.7% for
fork -> goal cell), the 30k budget, or the dot-product critic's capacity for
action discrimination at one state, is the open question; the stratified
critic batches of the earlier absorbing line (G1: 128 ordinary + 64 + 64
fork rows) are the obvious next probe, and were not run here.
