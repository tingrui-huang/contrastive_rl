# Three-arm ETT update reference experiment

Declared on f845b23 before new training or outcome generation. This is one
bounded model-internal experiment, not actor training or worst-case recovery.
No historical calibration or generator-expressiveness gate blocks this run.
Only numerical/geometry failures and the existing diagonal-fit guard can halt
or reject an update. No tuning, extra seeds, sweeps or budget extensions.

## Arms and fixed updates

Seeds 0 and 1, three proposed updates per arm, all theta=zeros(48) initially.
Arms: diagonal fitting only; diagonal plus averaged-NCE surrogate; diagonal
plus MC surrogate. The first 16 coordinates offset the existing diagonal
distribution head; the last 32 are its existing convex action-response maps.
Keep rectangle geometry, L=1 construction, actor and state-goal nominal policy,
region reward, gamma=.95, H=50-root_time, and Q decoding unchanged.

Each update uses four paired Gaussian directions in 48 dimensions. Central
finite differences use coordinate scales .01 (diagonal) and .1 (response).
Compute the gradient as mean[(loss(theta+sigma*u)-loss(theta-sigma*u))*u/(2*sigma)].
The energy-score diagonal gradient's response coordinates are exactly zero.
Sum diagonal and surrogate gradients with lambda_off=1 for the pessimistic
arms. Multiply by learning rates .01/.5 and clip each block's step norm to
.03/.1, respectively, then subtract. Identical directions, diagonal row draws,
and random keys are paired across arms within seed/update. No line search.

Diagonal finite differences use 128 uniformly sampled rows from the existing
512-row train fitting set and eight draws per row. The existing accept/reject
guard uses train rows 0:256, 16 draws, and one immutable random stream: accept
iff candidate mean emitted-XY energy score <= initialization score + .02.
Otherwise retain the previous theta. Signed probes may lie outside this fit
guard; accepted iterates may not. The geometry/Lipschitz construction applies
to all sampled probes as well as accepted kernels.

## Critic, visitation and MC alignment

Load the corresponding averaged_s0/s1 critic AND Adam state from the f845b23
experiment. Before every critic-arm update, generate eight current-ETT paths
per existing train root (36 roots), then refresh for exactly 400 Adam steps,
batch 256, lr=.003, uniform rows. Integrate the positive NCE term over every
available future time with truncated-geometric weights; keep the negative
term, class weights, architecture, normalization and unclipped Q decode.
No evaluation target enters this refresh. Freeze critic and optimizer during
all eight signed probes and the proposed ETT update; no final critic refit.

Each arm collects the same number of current-ETT paths using paired random
streams (also collected for the diagonal control for an auditable paired
visitation baseline). Freeze these paths during each update. Select 32 queries
by uniformly choosing a path, then uniformly choosing time within that path.
Keep its recorded action, h=H-t and weight H*.95^t. The two pessimistic arms
use identical selection identities and keys; their states may diverge after
their kernels diverge. Initial paths/queries must be identical across all arms.

For every signed candidate, draw four first successors per query and four
frozen-actor actions per successor. The NCE estimate is
mean[H*.95^t*(.05*r(s')+.95*Q_{h-1}(s',a'))]. For MC replace ONLY Q with
.05*sum_{j=0}^{h-2} .95^j*r(s_{j+1}), using one independent continuation per
actor action, all under that arm's frozen PRE-UPDATE theta. Candidate theta
acts ONLY on the proposed first transition. h=1 has exactly zero continuation.
First-successor, actor, and continuation streams are common across directions
and signs; keys are disjoint from trajectory collection and critic fitting.
This is a noisy model-internal reference, not a native-environment oracle.
The MC arm has no critic and does not learn from its continuation targets.

## Evaluation and hard budgets

After all six final kernels are sealed, evaluate them plus initialization with
64 independent full rollouts at each of the prior final 36 roots (12/group).
Use paired streams across ALL kernels; each evaluated theta acts at EVERY
transition. These are held-out roots for training but previously inspected
evaluation contexts. New return samples are independent of all update streams.
Report normalized discounted occupancy .05*sum .95^t*r. Zero return is not death.

Evaluate diagonal fitting independently on all 512 validation rows with 32
draws and paired streams. Report final-minus-initial energy degradation,
episode-clustered uncertainty, and whether the empirical validation degradation
exceeds .02 (a diagnostic, not a new selection gate). Re-evaluate final train
guards on the original stream. Check diagonal identity against the same
updated diagonal-head law with response coordinates zero, not against the
initial law, since diagonal heads are now trainable. Audit geometry/history and
common-x_prime samplewise action Lipschitz validity (2e-6 tolerance), with the
existing fixed action-grid test and bounded component-matrix construction.

Hard cap: 3,000,000 newly emitted model transitions, zero native simulation,
18 total ETT update attempts, 2,400 critic-refresh optimizer steps. The six
current-path collections per iteration cost <=254,016 total. MC signed
continuations cost <=1,179,648; all signed first transitions <=12,288;
diagonal signed probes <=147,456; initial/proposal/final train guards <=102,400;
final full rollouts <=790,272; validation diagonal <=114,688; validity <=20,000.
These conservative bounds sum to <2.73 million. Charge all emissions before
sampling, with arm/stage separated. Reused historical critic training and
checkpoints incur no new simulation; list their provenance separately.

## Predeclared analysis

Primary contrasts for each seed: critic-minus-diagonal, MC-minus-diagonal,
MC-minus-critic, and each final arm minus initialization. Lower return is the
pessimistic direction. Report signed means, conditional paired-MC standard
errors/95% normal intervals, and 2,000-replicate episode bootstrap intervals
stratified by root group. Repeats are averaged before the episode bootstrap;
intervals describe noisy episode means, not training-seed population uncertainty
or simultaneous guarantees. Also report group means, update acceptances,
diagonal/surrogate gradient and clipped step norms, saturation, fit degradation,
constraint tests, and simulation cost by arm and purpose. Do not select seeds
or checkpoints using these metrics.

A resolved improvement must have BOTH return intervals below zero. Evidence
favoring a critic bottleneck requires MC to improve relative to BOTH diagonal
and critic arms in BOTH seeds, with valid accepted kernels and no resolved
excess of the existing .02 diagonal allowance on validation. A missing critic
improvement alone is not evidence that the critic blocks optimization.
If both pessimistic arms improve similarly, a critic bottleneck is not
established; similar estimates are not proof of equivalence. If both fail to
show resolved improvement, consider limited update count, finite-difference
noise, one-step surrogate/local approximation, step caps, shared geometry or
fit constraints; do not attribute failure uniquely to critic estimation.
Resolved worsening in both arms supports a shared optimization problem in this
budget, not a global impossibility claim. Report unresolved cases as inconclusive.

Additional descriptive comparison: the first update uses identical theta,
visitation, candidates and successor streams; compare its four signed surrogate
differences and proposed steps between critic and MC. Later differences also
reflect different current kernels and cannot isolate estimator error. No extra
MC probes or calibration runs are permitted after seeing results.

Preserve prior artifacts; save protocol, source/input hashes, per-step paths,
queries, signed values, proposed/accepted parameters, refreshed critics, raw
final rewards, constraints, ledger and concise report in a fresh directory.
Stop after this experiment; no automatic push, actor training, extra sweep or
extension follows the outcome.
