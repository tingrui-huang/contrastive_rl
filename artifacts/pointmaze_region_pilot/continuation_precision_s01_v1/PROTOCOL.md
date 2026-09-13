# Fixed-model continuation diagnostic, declared before sampling

Base: 1d6deef. Exactly two candidate-minus-reference comparisons, in order:
s1_k0_u24 minus s1_k0_u0, and s0_k2_u24 minus s0_k2_u0, from
response_search_s01_v1/candidate_pool.npz. Use checks/s1_k0_u1/critic.pt and
checks/s0_k2_u1/critic.pt respectively: the averaged-positive critic fitted on
the corresponding reference before its first update. Verify saved parameter
hash, preupdate theta, fitted-path probability hash, and normalization provenance.
Freeze everything. No optimizer, fitting, native samples, or candidate selection.

## Estimand and sampling

Keep the original 36 training roots, with their approach/transit/bypass labels
and remaining horizons H=50-root_time. The original target selects a root
uniformly, a stochastic reference path, then a time uniformly from 0,...,H-1.
Its integrand remains H*.95^t*(.05*r(next)+.95*Q_{H-t-1}(next,next_action)).
Q decoding remains (1-.95^h)*31*.5*exp(positive_logit), with h=0 returning zero;
state normalization is the saved early_contexts_s01_v1 mean/std. No clipping.

For EACH root use four fixed relative-time strata: {0}, {1,2,3}, {4,...,9},
{10,...,H-1}. Allocate 16 independent reference paths to EACH root/time cell,
one query per path, and choose its time uniformly within its stratum using a
separate fixed RNG. Thus 2,304 queries per pair, 576 at time zero (all roots
are pre-goal) and another 576 at times 1-3. Sample all reference paths and times
without consulting critic scores or candidate differences. Root-group and
time strata are predeclared; never-entered-goal and previously-entered-goal
subgroups use only the reference prefix through the query state. The latter
are conditional analyses, not a filter or a reason to collect more samples.

Use eight independent paired first-successor groups per query, each with two
frozen-actor next-action / continuation draws. Pair natural-action, diagonal,
next-actor noise and continuation streams across the models. The natural
action is independently sampled from the frozen nominal policy. Critic and
MC consume exactly the SAME saved successors and next actions for each side.
BOTH continuations use the reference u0 ETT at every subsequent step, overriding
the first continuation action with the saved next action. Use remaining horizon
H-t-1, original binary region reward, .95 discount and .05 return normalization.
No death/goal termination is added; zero-mask exhausted horizons as before.

Each query's target mass is a_i=|time_bin_i|/(36*H_i*16). Overall means are
sum_i a_i*integrand_i, NOT an unweighted average of oversampled early queries.
Report time, root-group-by-time, and pre-entry strata conditionally by dividing
their weighted sum by their target mass. Estimated pre-entry mass is reported.
The original H*.95^t integrand weight is separate from this design correction.

## Fixed budget, precision and analysis

Practical tolerance epsilon=.01 in weighted, normalized one-step surrogate units;
this is a declared scale of interest, not a conversion of published full returns.
Target CI half-width <=.005 for overall MC difference and MC-minus-critic error.
Budget: 4,608 queries, 8 successors and 2 next actions each, 73,728 paired
continuations total. Charge both sides, including all 48 padded continuation
slots, all reference-path transitions, first transitions and 256 diagonal probes:
7,370,240 computed transitions, hard cap 7,500,000. Fixed seeds 171000000 for
queries (pair stride 1000000), 172000000 for time draws (same pair stride),
175000000 for comparisons (pair stride 1000000, batch stride 100), and
178000000 for diagonal probes. Batches of 256 queries. No interim analysis.
Stop after these two pairs and one final analysis; no precision-based extensions.

Average the 16 draws WITHIN a query first. For paired critic, MC and
MC-minus-critic differences, use stratified independent-query variance:
sum_cells mass_cell^2 * sample_variance(query_influence)/16. For conditional
ratios use flag*(value-conditional_mean)/estimated_mass as the influence.
Use Welch-Satterthwaite t intervals (cell df=15). Main intervals are 97.5%
per pair, Bonferroni 95% across the two pairs separately for each metric;
no joint familywise claim across metrics. Conditional intervals are descriptive
95%, with no multiplicity correction. Roots are fixed: inference includes fresh
path/action/transition randomness, not generalization to a new root population.
Shared successors make the two next-action draws a group; do not pretend all
continuations are independent queries. Separately report conditional Monte Carlo
SE from the eight successor groups, Kish effective target/design and visitation
query counts, unique state and state/action counts, and paired reward-sequence
changes (first reward OR any active continuation reward), with weighted rates.
Zero observed variance is flagged, not treated as proof of equivalence.

Classify each overall comparison in this priority order:
1. Meaningful critic error: discrepancy interval lies wholly outside [-.01,.01].
2. Agreement at a meaningful effect: discrepancy interval lies inside tolerance,
   and critic and MC intervals lie beyond .01 in the same direction.
3. Precisely small effect: critic, MC and discrepancy intervals all lie within
   tolerance, precision targets met, and no zero-variance diagnostic limitation.
4. Otherwise insufficient precision to resolve these categories. Report achieved
   half-widths separately, including when statistically nonzero effects are small.

Reuse published final-minus-start full-rollout results solely as context. They
apply each candidate at every step on held-out roots and are a different quantity.
Only discrepancy against matched reference-continuation MC diagnoses this
critic's contrast error. Large u24/u0 moves test continuation generalization;
they do not retrospectively validate every update. End with a compact research
decision, without claiming global worst-case recovery, universal action
equivalence or native-policy benefit. Preserve all previous files.
