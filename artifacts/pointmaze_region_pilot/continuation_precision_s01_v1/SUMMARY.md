# Continuation generalization error is supported for one fixed comparison

Starting from 1d6deef, the two predeclared endpoint comparisons were completed
without training or changing the actor, critic, nominal policy or ETT. The
saved reference-model critics and their state normalization passed provenance
checks. Existing experiments and checkpoints were preserved.

**s0_k2_u24 versus s0_k2_u0 shows a meaningful critic contrast error.** The
weighted one-step critic difference is **−0.02242**, versus matched MC
**+0.01710**. MC-minus-critic is **+0.03953**, with a 97.5% interval
**[+0.01468, +0.06437]**, wholly beyond the predeclared **0.01** tolerance.
MC's own interval includes zero, so this establishes a magnitude discrepancy,
not a confidently reversed MC direction. The pre-goal conditional discrepancy
is +0.19100 [ +0.09079, +0.29121 ]; that interval is descriptive 95%.

**s1_k0_u24 versus s1_k0_u0 remains unresolved.** Critic difference is
**−0.04663**, MC **−0.02162**, and discrepancy **+0.02500**
**[−0.00301, +0.05301]**. This is neither demonstrated accurate agreement nor
evidence of a precisely small effect. Neither comparison satisfies the declared
agreement-at-meaningful-effect or precisely-small-effect category.

Neither pair reached the predeclared 0.005 half-width target. MC/discrepancy
half-widths were **0.03231/0.02801** for s1_k0 and **0.01950/0.02485** for s0_k2.
These intervals include fresh-query uncertainty, grouping repeated draws within
each query. Main intervals correct for two comparisons separately per metric;
conditional strata have unadjusted descriptive intervals. Inference is conditional
on the 36 fixed roots, rather than a new population of roots.

The fixed allocation used **2,304 fresh paths/queries per pair**, eight paired
successors and two next-action/continuation draws per successor. Of 4,608 queries,
**3,026** were before first goal entry; only **1,580** were currently inside the
reward region. **9,569/73,728** paired reward sequences changed. This replaces
the previous mostly post-entry design with explicit early-time coverage. Design
weights recover the original target: each pair has Kish effective target-query
count **896.3**, or **1,190.0 / 1,162.7** after including visitation weights.
The report and results.json give all root/time strata, prefix subgroups, rates,
uncertainty and unique-state counts.

Critic and MC used identical saved successors and next actions on each side;
both MC branches continued under the frozen reference u0 model. Original reward,
discount, horizon masks, Q decoding and H*gamma^t visitation weights were retained.
Published full-model differences remain useful context: **−0.01121** for s1_k0
and **+0.00691** for s0_k2. Those apply the candidate every step on different roots;
they are not the one-step surrogate estimand and cannot alone diagnose critic error.

**Decision:** prioritize continuation-value generalization and calibration before
spending more on response optimization. The evidence identifies a real contrast
error for one large candidate move; it does not establish that this is the dominant
bottleneck for every update or baseline. No new calibration, precision round,
optimizer sweep or full-model evaluation was launched. These two comparisons
cannot establish global worst-case recovery, universal action equivalence or
native-policy benefit.

Cost: **7,370,240 / 7,500,000** computed transitions, including padded slots;
zero parameter updates and zero native steps. Diagonal identity and frozen hashes
passed. Three analytic/provenance tests passed; the final audit re-decoded all Q
values and reconstructed all masked MC returns and paired surrogate values.

Details: PROTOCOL.md, provenance.json, REPORT.md, results.json, audit.json,
and per-pair queries.npz, reference_paths.npz and batch*.npz.
