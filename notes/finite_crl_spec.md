# Finite-state Contrastive RL pessimistic-loss protocol

This is an isolated loss-validation experiment, not PointMaze or an estimate of
the true environment's worst case. The protocol is saved before collection.
No actor, encoder, production ETT, or existing readout is trained or modified.

## Benchmark and feasible family

States are S (start), M (middle), G (commanded goal), and D (death). Actions and
natural actions are {0,1}; actor pi(1|s,G)=3/4 and b(1|s,G)=1/4 for every state.
Both remain fixed. Fresh natural actions are independent auxiliary draws, not
executed actions. All interventions use the SAME shared kernel marginalized
over b. Initial state is S, horizon H=4, gamma=9/10. The reward after a transition
is (1-gamma)1{s_next=G}. G and D absorb; neither ends the original four-step
episode. A goal at step 2 receives rewards at steps 2,3,4.

For s=S,M, p_s(x,x_prime)=d_s+v_s(x-x_prime). Successor is M from S, or G from M,
with probability p_s; otherwise D. Observed diagonal laws have success masses
m_S=7/10 and m_M=4/5 at BOTH actions (equivalently 100 exact-count observations
per state/action). They are declared benchmark distributions, not estimates
from PointMaze. Optimize all four real parameters d_S,v_S,d_M,v_M jointly.
The feasible family is d_s in [m_s-1/50,m_s+1/50], v_s in [-3/20,3/20]. These
intervals imply every cell is a valid probability; no extra clipping is used.

Diagonal error is maximum total variation (TV) over all diagonal cells; tolerance
is 1/50. The full action constraint is
TV(T(.|s,x1,x_prime),T(.|s,x2,x_prime)) <= (3/20)|x1-x2|,
for EVERY s,x_prime,x1,x2. State ground metric is discrete (distance 1 for
distinct states); TV equals Wasserstein-1 under this metric. This is a
DISTRIBUTIONAL constraint, not samplewise. Under a shared uniform inverse CDF,
individual draws can differ by 1 even when the TV bound is 3/20. No bound in
x_prime is asserted. |v_s| is exactly the nontrivial pair's TV/action ratio;
absorbing rows have ratio zero. Every iterate is projected onto the parameter
box, so feasibility is enforced, not purchased with a third penalty.

The affine response couples both off-diagonal cells to each other and the
diagonal. It is a four-parameter restricted family, not all TV-feasible kernels.
The complete Markov state explicitly includes death; no hidden-state inference
or causal identification problem is represented in this benchmark.

## Exactly two training losses and alternating schedule

Minimize mean Bernoulli diagonal NLL plus 4 times the pessimistic-return
gradient surrogate (weight zero for the diagonal control). The NLL is averaged
over states/actions using the exact declared observation frequencies. d receives
both derivatives. v receives the pessimistic derivative and zero diagonal
derivative because the affine action difference vanishes on the diagonal.
The diagonal remains trainable and is not an immutable fitted backbone.

Use three seeds 0/1/2, paired identical initializations and random streams across
control/joint. d starts at m+Uniform[-.015,.015], v at Uniform[-.03,.03]. Run 48
refresh/update rounds, one projected SGD update per refresh, learning rate .12.
Select only the final iterate. No oracle, exact model values, monitoring-based
selection, hyperparameter adjustment, or early stopping enters training.

Each refresh freezes ETT and generates 512 independent complete continuations
per (remaining h=1..4, state, first action) query. First actions are fixed by the
query; every subsequent action uses pi. This stratified coverage provides all
successor queries, separately from 2,048 on-policy S-root paths used to estimate
visitation d_t(s). Keep h explicit. One positive per query continuation is
s_{k}, with k=1..h drawn with probability (1-gamma)gamma^(k-1)/(1-gamma^h).
The achieved-state contrastive label does not change the commanded goal G or
the actor. MC positives, not Bellman targets or exact dynamic programming,
train the critic.

Binary NCE uses B=32, one positive and B-1 independent negatives per row in
expectation, matching the repository's mean B-by-B sigmoid cross entropy.
Unlike reusing other rows' positives as negatives, integrate a known fixed
q_alpha=(1-alpha)Uniform(S,M,G,D)+alpha*delta_D exactly. All states retain
support. This changes the negative sampler intentionally, not the class-prior
weight. Minimize the empirical tabular NCE objective analytically:
exp(f_h)=p_hat_h/((B-1)q_alpha). Zero-count cells use the extended MLE f=-infinity
(stored as finite zero odds), without pseudocounts or clipping.

The stationary equation p_hat_h sigmoid(-f)=(B-1)q_alpha sigmoid(f) gives
Q_hat_h=(1-gamma^h)(B-1)q_alpha*exp(f_h). This estimates
E[sum_{k=1}^h (1-gamma)gamma^(k-1)1{s_k=g}]. It is NOT valid without the class
prior factor, with unaccounted empirical negatives, or for an arbitrary raw
score. Q_0=0. Use only alpha=0 for ETT training. On the same final MC counts,
alpha=.5 is a controlled calibration comparison: correct q should cancel the
logit change; using q_0 to decode alpha=.5 is an explicitly incorrect control.

Freeze critic parameters and visitation during each ETT update. Sum
gamma^t d_t(s) pi(a|s) b(x_prime|s) T_theta(y|s,a,x_prime) times
[(1-gamma)1{y=G}+gamma sum_a' pi(a'|y) Q_hat_(H-t-1)(y,a',G)].
Enumerate ALL categorical choices in this one-step expression. Its derivative
is the Rao-Blackwellized categorical score-function estimator. Do not backprop
through sampled states, ignore categorical mass derivatives, or average logits.
The visitation-weighted surrogate's VALUE is not J: with exact Q its gradient
at the generating kernel equals grad J by the finite-horizon likelihood-ratio
identity. Fresh visitation and critic are collected before the next update.

## Bounded validation, independent of training

Training seeds: base 82000000+seed for initialization;
base+10000+1000*seed+round for refreshes. Final evaluation base+100000+seed,
paired across arms, uses 2,048 new positives per (h,s,a) and 16,384 S-root paths.
Hard caps are 15,000,000 training and 1,500,000 evaluation transitions across
all six runs. Tests use separate small bounded draws. No expansion or retries
selected by outcomes. Frozen artifacts and source hashes are saved before run.

After all six final checkpoints are sealed, a separate evaluator computes
exact dynamic-programming Q, exact objective and probabilities, and a rational
analytic feasible-family certificate. It reports critic RMSE/max error at all
queries and goal-only queries, on-policy gradient error, each iterate's
feasibility, final objective gap, goal/death probability, and independent MC
intervals. Across seeds report individual values, not unsupported significance.
No clipping of a negative gap is allowed. The oracle may not be imported by
the training module or used to choose checkpoints.

Certificate: intervention success is pbar_s=d_s+(1/2)v_s. J is
gamma(1-gamma^3)*pbar_S*pbar_M, strictly increasing in both pbar_s on the whole
positive feasible box. Hence its global minimum is attained simultaneously
at d=(.68,.78), v=(-.15,-.15), giving pbar=(.605,.705). Rational arithmetic supplies
the exact optimum, not a sampled minimum or independently incompatible cells.
For diagonal NLL+4J, each d derivative is at least
-EPS/[2*min_box d(1-d)] + 4*gamma(1-gamma^3)*min_other pbar > 0;
each v derivative is positive. Thus the same feasible corner also globally
minimizes the joint objective. NLL's logarithmic constant is evaluated
numerically after the exact monotonicity certificate. Diagonal-only NLL has
d=m and arbitrary v; its unresolved off response is a representational choice.

Focused checks: exhaustive action pairs and absorption, joint b marginalization,
reward timing/horizon, replay and first-action alignment, geometric/NCE scaling
and stationarity, alpha correction, exact-gradient versus finite difference,
two live parameter derivatives, projection and rational certificate witness.
Stop after reporting; no PointMaze training, actor update, alpha sweep, ETT
integration, or push is authorized by this experiment.

Background: contrastive/learning.py MC binary branch; existing shared-design,
hard-rollout-return, and frozen-return-readout reports. The discounted-occupancy
interpretation follows Eysenbach et al., Contrastive Learning as Goal-Conditioned
Reinforcement Learning (2022), https://arxiv.org/abs/2206.07568. The finite-horizon,
fixed-negative, B-1 correction above is explicitly derived for THIS protocol.
