# Bounded future-time averaging comparison

Declared before fitting or generating new outcomes on branch head 3311150.
No ETT optimization, actor training, nominal fitting, checkpoint selection,
additional seeds, perturbations, sweeps, or budget extensions are permitted.

## Training

Reuse the exact 288 model trajectories (13,648 rows) and sampled labels from
phase_sampling_s01_v1. Both arms use the existing uniform row sampler, paired
initialization seeds 98000000/98000001, and identical minibatch indices: 1,500
Adam updates, batch 256, learning rate .003. The sampled arm is the original
one sampled truncated-geometric future label per row, fixed across epochs.
The averaged arm substitutes p*softplus(-f1)+(1-p)*softplus(-f0), where
p=sum_{k=1}^h .95^(k-1)*r[t+k]/sum_{k=1}^h .95^(k-1). The negative term remains
31*mean_g softplus(f_g); the entire loss is divided by 32. Architecture,
normalization, reward, gamma, horizon feature h/50, and unclipped decoding
Q_h=(1-.95^h)*31*.5*exp(f1), Q_0=0, remain unchanged. Baseline hashes must
reproduce both prior uniform critics before new evaluation is inspected.

## Fixed evaluation and budget

Reuse the prior final_contexts.npz 36 episode roots (12 per observable group).
These roots were held out of training but their prior evaluation has been
inspected; they are NOT a pristine new population holdout. No old value targets
are loaded. Generate one fresh base-ETT visitation trajectory per root and
select one uniform time per trajectory with seed 117000001. Each context uses
its recorded action, remaining horizon H-t, and visitation weight H*.95^t.
This is uniform-root / uniform-within-path-time Monte Carlo of the existing
discounted visitation surrogate, not uniform transitions or normalized weights.

The frozen base is theta=zeros(48). Four predetermined candidate offsets have
zero first 16 diagonal entries. Their last 32 entries are +/-.1 times the
unit-Frobenius vector of eight repeated identity or 90-degree rotation matrices.
Each total parameter norm is .1. Evaluate candidate-minus-base differences
only, no ranking-based candidate selection or subsequent update.

For each of the 36 visitation queries and each of base plus four candidates,
draw 128 first successors. Hold contexts/actions/x_prime/noise streams matched
across candidates and both critics. Predict .05*r(s')+.95*Q_{H-t-1}(s',a'),
using one frozen actor action per draw (128 total). Independently compute the
same target using an entire continuation under ZERO base theta, holding this
first continuation action a' fixed. Candidate theta applies ONLY to the first
transition. Continuation random streams are paired across candidates, with
fresh keys disjoint from training and visitation. The target is independently
computed from rewards, sharing randomness with the prediction for precision;
it is never provided to a fitter. No native simulator ground truth is claimed.

Calibration: each of the same 36 roots, actual H=50-root_time, one fresh frozen
actor action, and 64 independent base continuations. Report RMSE, bias, range
violations, MC SE, and paired averaged-minus-sampled RMSE. Also report
visitation-query calibration against the independent base continuation values.

Hard caps: 6,000 total critic optimizer steps; 1,000,000 newly emitted model
transitions including visitation, calibration, surrogate continuations, and
validity audits; zero native steps. Worst-case surrogate cost is
36*128*49*5=1,128,960, so the selected query lengths must be checked against
the cap BEFORE any model evaluation. If the fixed selection exceeds the cap,
stop without resampling or reducing repeats. Calibration <=112,896;
visitation <=1,764; validity audits <=25,000. Typical total is about 700,000.
No evaluation targets are used to choose the time indices. All artifacts and
frozen checkpoints are hashed; prior artifacts are read-only.

## Analysis declared before outcomes

Report all four signed surrogate differences, each critic seed separately,
in the existing unnormalized visitation-weighted surrogate units. Lower is
the pessimistic direction. For each estimate report a conditional-on-queries
Monte Carlo 95% normal interval using paired repeat differences, and a
descriptive 95% episode-bootstrap interval (2,000 replicates, stratified by
the three root groups). The latter resamples noisy episode means and describes
root/visitation variation; it is not a simultaneous or training-seed interval.

An MC difference is distinguishable only if both intervals exclude zero,
abs(mean)>.001, and repeat-half mean differences agree in sign. Score direction
only on these differences; critic predictions whose own two intervals do not
resolve a sign are inconclusive, not successes. Report correct/resolved/eligible
counts for each seed. Four correlated probes do not support a broad usefulness
claim. Primary continuous metric: mean absolute error of the four predicted
signed differences versus MC, with paired averaged-minus-sampled change and
episode-bootstrap interval. Improvement requires a negative interval for both
paired seeds without losing a correct resolved direction; otherwise no clear
improvement. No tuning follows any outcome.

Audit samplewise diagonal equality against base on matched draws, emitted
geometry/history validity, and common-anchor action Lipschitz differences on
16 fixed random action pairs at the 36 roots, four draws per pair. The existing
matrix normalization bounds each component operator norm by one; convex gates
and nonexpansive action-independent rectangle projection preserve L<=1.
Zero diagonal offsets preserve the entire diagonal law, not just the tested
samples. Report numerical tolerances (2e-6), measured maximum ratio/excess,
and component norms. These properties do not establish native physical validity
or latent-state absorption.

Analytic averaging eliminates ONLY the conditional future-time label draw on
each realized suffix. Quantify mean p(1-p) and conditional positive-loss
variance. Shared finite trajectories still have rollout noise; function
approximation, optimizer error, coverage, and model misspecification remain.
Two initialization seeds on one trajectory dataset cannot estimate variability
over independently collected training datasets. Unresolved effects are
inconclusive, never evidence of equality. Stop after reporting this comparison.
