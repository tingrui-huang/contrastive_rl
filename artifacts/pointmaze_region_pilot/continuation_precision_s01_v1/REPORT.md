# Fixed-model continuation-value diagnostic

Prioritize continuation-value generalization and calibration before spending more on response optimization: at least one fixed reference critic has a practically meaningful contrast error against matched MC.

Candidate minus reference throughout. Overall intervals are 97.5% per pair (95% across the two pairs separately for each metric). Time/prefix intervals below are descriptive 95%. Units are the original weighted, normalized one-step surrogate; negative means more pessimistic. Tolerance is ±0.01; target half-width is 0.005. Root/time design weights recover the original uniform-root, uniform-time visitation target.

## s1_k0_u24 minus s1_k0_u0

**insufficient precision to resolve the declared categories.**
Critic -0.04663 [-0.08685, -0.00640]; MC -0.02162 [-0.05394, +0.01069]; MC-minus-critic +0.02500 [-0.00301, +0.05301].
MC / discrepancy half-widths: 0.03231 / 0.02801; precision target met: False.
Conditional successor-MC SEs: 0.00901 / 0.00792. These exclude fresh-query uncertainty; main intervals include it.
2304 independent paths/queries on 36 fixed roots; 1764 unique states, 2304 unique state/actions. 1505 queries before first goal entry; 798 currently inside the reward region. Kish effective query counts: design 896.3, design × visitation 1190.0.
Changed reward sequences: 5160/36864 paired continuations (first or later reward); first rewards 78/18432, later sequences 5055/36864. Original-target weighted sequence-change rate 4.956%. Changed full-F4 successors: 18430/18432.

- time_0: n=576, target mass=0.0211; critic -0.00860 [-0.06630, +0.04910]; MC -0.13556 [-0.25368, -0.01743]; discrepancy -0.12696 [-0.22917, -0.02475]; weighted sequence-change rate 20.294%.
- time_1-3: n=576, target mass=0.0634; critic -0.07807 [-0.20513, +0.04899]; MC +0.01354 [-0.14081, +0.16789]; discrepancy +0.09161 [-0.01577, +0.19899]; weighted sequence-change rate 21.993%.
- time_4-9: n=576, target mass=0.1268; critic -0.17436 [-0.30844, -0.04027]; MC -0.10182 [-0.22685, +0.02321]; discrepancy +0.07253 [-0.01144, +0.15651]; weighted sequence-change rate 11.576%.
- time_10+: n=576, target mass=0.7887; critic -0.02458 [-0.06257, +0.01340]; MC -0.00851 [-0.03549, +0.01847]; discrepancy +0.01608 [-0.01056, +0.04271]; weighted sequence-change rate 2.112%.
- pre_entry: n=1505, target mass=0.2113; critic -0.14617 [-0.25417, -0.03817]; MC -0.05659 [-0.17864, +0.06547]; discrepancy +0.08958 [+0.01395, +0.16521]; weighted sequence-change rate 22.171%.
- entered: n=799, target mass=0.7887; critic -0.01996 [-0.05466, +0.01473]; MC -0.01226 [-0.02744, +0.00292]; discrepancy +0.00770 [-0.01614, +0.03155]; weighted sequence-change rate 0.346%.

Published full-model return difference: -0.01121, root 95% CI [-0.01812, -0.00380]. This reuses independent held-out-root rollouts with each ETT applied every step. It is context, not a matched one-step target or evidence of critic error by itself.

## s0_k2_u24 minus s0_k2_u0

**meaningful critic error.**
Critic -0.02242 [-0.03965, -0.00520]; MC +0.01710 [-0.00240, +0.03661]; MC-minus-critic +0.03953 [+0.01468, +0.06437].
MC / discrepancy half-widths: 0.01950 / 0.02485; precision target met: False.
Conditional successor-MC SEs: 0.00703 / 0.00711. These exclude fresh-query uncertainty; main intervals include it.
2304 independent paths/queries on 36 fixed roots; 1764 unique states, 2304 unique state/actions. 1521 queries before first goal entry; 782 currently inside the reward region. Kish effective query counts: design 896.3, design × visitation 1162.7.
Changed reward sequences: 4409/36864 paired continuations (first or later reward); first rewards 70/18432, later sequences 4302/36864. Original-target weighted sequence-change rate 4.452%. Changed full-F4 successors: 18427/18432.

- time_0: n=576, target mass=0.0211; critic +0.04448 [+0.00147, +0.08749]; MC -0.03893 [-0.12252, +0.04466]; discrepancy -0.08341 [-0.16304, -0.00378]; weighted sequence-change rate 16.714%.
- time_1-3: n=576, target mass=0.0634; critic +0.01602 [-0.03649, +0.06853]; MC -0.02426 [-0.13208, +0.08356]; discrepancy -0.04028 [-0.13616, +0.05560]; weighted sequence-change rate 17.173%.
- time_4-9: n=576, target mass=0.1268; critic -0.12398 [-0.18948, -0.05849]; MC +0.04358 [-0.03722, +0.12439]; discrepancy +0.16757 [+0.07640, +0.25874]; weighted sequence-change rate 12.060%.
- time_10+: n=576, target mass=0.7887; critic -0.01098 [-0.02639, +0.00442]; MC +0.01767 [+0.00276, +0.03258]; discrepancy +0.02865 [+0.00663, +0.05068]; weighted sequence-change rate 1.878%.
- pre_entry: n=1521, target mass=0.2058; critic -0.11948 [-0.19041, -0.04854]; MC +0.07152 [-0.00944, +0.15249]; discrepancy +0.19100 [+0.09079, +0.29121]; weighted sequence-change rate 21.356%.
- entered: n=783, target mass=0.7942; critic +0.00273 [+0.00072, +0.00473]; MC +0.00300 [+0.00017, +0.00583]; discrepancy +0.00028 [-0.00337, +0.00392]; weighted sequence-change rate 0.072%.

Published full-model return difference: +0.00691, root 95% CI [+0.00199, +0.01197]. This reuses independent held-out-root rollouts with each ETT applied every step. It is context, not a matched one-step target or evidence of critic error by itself.

## Scope, provenance and verification

Each reference critic was loaded from its saved _u1 checkpoint, after its 1,000-step reference-model fit and before response update 1. Checkpoint bytes, parameter hashes, preupdate theta, fitted-path probability hashes, and saved mean/std provenance were verified. The earlier reporting-only source correction is explicitly verified. See provenance.json for exact paths and hashes. Both MC branches use reference u0 after their respective first successors; critic and MC share successor/next-action arrays. The final audit re-decodes all saved Q values and reconstructs masked MC returns and visitation-weighted objectives.

Computed transitions: 7,370,240/7,500,000, including padded slots. Zero actor, critic, nominal or ETT updates and zero native steps. Diagonal offsets and paired diagonal samples remain exact; response constraints and frozen parameter hashes pass. Root-group-by-time conditional results, estimated pre-entry masses, effective counts and all uncertainty details are in results.json.

Intervals condition on these 36 existing roots; they do not cover root-population generalization. Repeated time-zero states have independently sampled actions, and successor/actor draws remain grouped within queries. Conditional zero-variance strata do not prove equivalence, particularly with rare reward changes. These two large endpoint contrasts test continuation generalization, not every historical optimization update. They cannot establish global worst-case recovery, universal action equivalence or native-policy benefit. No new full-model evaluation, precision extension, or sweep was run.
