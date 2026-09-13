# Bounded native policy-improvement experiment

Predeclared on 817e429 before any new fitting or native outcome inspection.
One experiment, two paired learner/model seeds 0/1. Freeze every transition
parameter and the state-goal nominal policy. No ETT update, sweep, extra seed,
budget extension, checkpoint selection, or automatic push.

## A/B/C and the observational model

A is the exact unchanged original actor checkpoint used in 817e429, shared
between seeds. B_s starts from A and uses final diagonal_s{s} from 817e429,
with BOTH action channels equal to the executed action: sample(theta,s,a,a).
There is NO independent nominal action in B. C_s starts from the identical A
and uses final critic_s{s}, with an independent frozen nominal x_prime and
sample(theta,s,a,x_prime). B thus uses the fitted observational conditional,
not the action-invariant independently marginalized diagonal-only kernel.
The model's original rectangle geometry and projection are unchanged. C's
inherited L=1 action bound conditions on fixed x_prime; do not infer that bound
for B's joint action-dependent diagonal conditioning.

Use the original full actor network and tanh-Gaussian sampling in training,
native evaluation, and cross-model evaluation, with no extra exploration or
deterministic-evaluation switch. All actors receive only visible F4 and the
fixed commanded goal. Preserve task reward 1[distance(next_XY,goal)<2], gamma=.95,
and the original 50-step external horizon. Native success is any distance<.5;
native failure is the absorbing-death flag, read for evaluation only. Failure
and failure-to-succeed are distinct. No hidden labels enter training.

## Paired bounded training

Four rounds per trained actor, identical B/C schedules and initialization.
Each round generates eight model paths from each of the existing 36 train
roots (actual remaining horizon 50-root_time), plus 12 copies of the public
reset state START with horizon 50. Reset roots supply 25% of root draws. Keys
are paired across B/C within seed; model paths naturally differ between models.

Initialize paired region critics from seed 130000000+s, not old hindsight
critics. Fit each model's initial critic for 1,500 Adam steps before its first
actor update; subsequent rounds refresh it for 400 steps with persistent Adam
state. Batch 256, lr=.003, uniform transition rows. Analytically average the
positive NCE term over all realized future times with gamma-geometric weights.
Retain binary architecture, normalization, 1:31 class weights, negative term,
horizon feature h/50, unclipped Q_h=(1-.95^h)*31*.5*exp(f1), and Q_0=0.
The critic estimates the current actor under that round's fixed transition model.

Freeze the critic and collected paths during 25 actor-update attempts per
round. Sample 256 paths uniformly then one within-path time uniformly. Maximize
mean[H*.95^t * Q_{H-t}(s, a_pi)] using four reparameterized tanh-Gaussian actions
per state (two independent Gaussian draws and their negatives). Copy the region
MLP/embedding weights to an equivalent float32 JAX forward implementation;
differentiate decoded Q through action and the original actor, not through the
no-gradient predict interface. Verify values/action gradients against Torch,
and the actor gradient against central differences. Critic parameters, states,
weights, transition model and nominal policy are not differentiation targets.
Do not use full-F4 hindsight goals, the old CRL actor loss, or native targets.

Actor Adam is reset identically for B/C (lr=1e-5, global gradient norm clip=1).
Propose one step, then accept only if all declared constraints pass; otherwise
retain actor AND optimizer state with no line search or retry. On a fixed bank
of 512 train-fitting states +36 train roots +64 public START copies, and on
the current 256 actor-update states, require mean KL(initial||candidate)<=.02,
RMS deterministic-mean-action L2 drift from initial<=.05, and maximum drift
<=.15. Also require mean KL(prestep||candidate)<=.002 on update states. KL is
the exact diagonal-Gaussian KL before the invertible tanh transform. These are
empirical state-bank constraints, not global policy guarantees. Log attempted
and accepted steps and frozen-critic Q changes; native outcomes never select.

## Independent final evaluation and budget

After all four trained actors are sealed, evaluate A, B0, C0, B1, C1 for 50
native steps on exactly 256 fresh reset seeds (140000000+episode). Stochastic
action keys (141000000+t) are shared across all actors with the same batch
shape, independent of environment seeds. Continue all 50 steps, including
absorbed trajectories. No native outcome is observed during training.

Cross-evaluate all FIVE actors on all FOUR frozen models (obs0, pess0, obs1,
pess1), 128 independent full 50-step model rollouts from START per cell, with
common streams (142000000). Each model acts throughout. Report actual model
rollout returns, not only predicted critic values; compare each actor to A
on each model to reveal model-specific improvements or deterioration. No MC
reference ETT, new model, or extra ablation is trained.

Budgets: exactly 400 actor-update attempts, at most 400 accepted; exactly
10,800 NCE optimizer steps across four actors; at most 500,000 newly emitted
model transitions (training 295,168, cross-evaluation 128,000; remaining capacity
is not used to extend training); exactly 64,000 native evaluation steps. Test
fixtures are separate correctness checks, never experimental training data.
No calibration gate or outcome-based early stopping. A numerical failure stops
the attempt and is preserved rather than silently retuned.

## Analysis fixed before outcomes

Primary native metric: raw discounted task return sum_t .95^t*r_{t+1}.
Report normalized .05*return when comparing to critic Q. Also report success
and absorbing failure rates. For each seed report C-minus-A, C-minus-B and
B-minus-A means and 95% paired episode-bootstrap intervals (2,000 replicates,
seed 143000000), with discordant counts for binary endpoints. C-versus-A and
C-versus-B are the prespecified primary contrasts. Positive return/success and
negative failure are favorable. Zero-crossing intervals are inconclusive;
all-zero paired differences are observed ties, not evidence of equivalence.

Cross-model intervals use paired simulated episodes with the same bootstrap
rule. Report own-model improvement and the full cross-model comparison against
A. State clearly when model gains fail to transfer natively or depend on the
training model. Evidence favoring C requires native return improvement over
both A and B in both seeds, with no resolved worsening of success/failure;
otherwise describe mixed or inconclusive evidence. Intervals are descriptive,
not simultaneous or population-over-training-seeds guarantees.

Report fixed-bank mean KL, deterministic-action RMS/max changes, parameter
change norm, scale changes, attempted/accepted updates, simulation cost, and
the inherited validation diagonal-energy degradation from 817e429. B/C inherit
different diagonal heads as well as response maps: this experiment does not
isolate the response term. F4 partial observability and generator error remain.
The ETT was optimized against A and is NOT asserted worst-case for updated B/C.
Persist raw rollouts, native audit arrays, model routing, checkpoints, hashes,
gradient checks, costs and a concise report in a fresh output directory.
