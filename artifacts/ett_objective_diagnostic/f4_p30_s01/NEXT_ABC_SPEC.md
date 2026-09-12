# Later A/B/C experiment specification (NOT RUN)

This is a bounded exploratory protocol for a separately authorized follow-up.
The present decision is to reconsider conditional references and target-horizon
compatibility first; the current diagnostic does not validate either objective.
If an A/B/C comparison is nevertheless authorized, preregister this protocol
without repeatedly tuning it on the 91 frozen histories.

1. Freeze the same nominal policy and actor; retain the same expert-source
   episode split, commanded goals, 233 training-only references and fixed
   normalization. Use the original diagonal initialization for every arm.
   No hidden input, new bank, death reward or critic objective is introduced.
2. A: use a plain 128/128 conditional zero-inflated three-Gaussian transition
   network with input [s,x,x_prime,g], only newest XY stochastic, and train only
   the existing mixed-measure diagonal NLL. Off-diagonal sampling evaluates
   these ordinary concatenated inputs. There is no equality flag or residual
   multiplied by action difference.
3. B: exactly the same plain network and initialization as A; train diagonal
   NLL plus lambda times the hard full-F4 nearest-set cost. There is no equality
   flag, explicit zero-on-equality residual, or enforced diagonal identity.
   The network must learn empirical diagonal preservation from NLL alone.
4. C: the current guarded architecture, initialized from the same diagonal
   model, with its zero-initialized 64/64 residual. Train the same NLL plus the
   same lambda/set cost as B. Retain L=0.25 in raw XY/action Euclidean units.
   C forces equality with its OWN CURRENT diagonal sampling law at x=x_prime;
   because its base parameters are updated, it does not freeze the initial
   observational distribution. This distinction must be tested empirically.
5. All arms retain exact history shifts and the existing endpoint/cap geometry.
   A and B have no guaranteed anchored action-change bound: measure violations
   against their own shared-randomness diagonal draws rather than silently
   projecting them into C's feasible set. C enforces its bound with the guard
   and fallback. Endpoint/segment checks are not a dynamics proof.
6. Fixed budget: seed 0 and 1, 500 updates each, batch 64, K=8 for pessimistic
   training; K=8/16 for evaluation. Use Adam at 1e-5 for all trainable parameters
   in all arms, without weight decay or an extra soft constraint. Choose one
   common lambda for B/C using four TRAINING batches (seeds 300..303): match a
   weighted pessimistic/diagonal gradient norm ratio of 0.01 using the median
   over both architectures, clipped to [1e-4,1]. Record the actual objective and
   gradient scales before training; do not select lambda on evaluation outcomes.
   Keep the final fixed-budget checkpoint primary. Run finite-gradient and
   parameter-change checks first; record discrete-gate gradient limitations
   consistently in B/C rather than changing estimators between arms.
7. Measure empirical diagonal retention against initialization AND A with
   episode-paired NLL changes, mixed atom calibration, position/energy errors,
   and visible frozen/moving/region strata. Report uncertainty, not just an
   overall average dominated by common near-goal rows. A provisional engineering
   guardrail is no more than 0.05 nats extra NLL deterioration versus A; it is
   not a theorem or a causal criterion. Report off-diagonal set/MMD terms,
   stationarity drift, diversity/collapse and geometry/bound diagnostics even
   if the diagonal guardrail passes.
8. Treat the current saved evaluation subset as development cases. Use newly
   held-out expert-source episodes for a later independent diagonal assessment;
   obtaining them is separate future work. Do not claim counterfactual accuracy
   without a verified simulator state-restoration/randomness protocol or valid
   off-diagonal outcome data. Observed stationarity alone is not an absorbing
   death label.

A versus B measures how this objective affects a plain network's empirical
diagonal fit and sampled outcomes. B versus C measures the combined influence
of anchoring, the hard action bound and added residual capacity. It does NOT
isolate an equality label, prove that structure alone preserves a pretrained
diagonal density, or separate these architectural factors causally. Different
optimization difficulty and C's extra parameters must be reported. None of
the arms can establish the true counterfactual kernel, a worst-case Q function,
or genuine failure recognition merely by decreasing the objective.

No A/B/C model code, optimizer run, or command has been launched in this task.
