# Response-only ETT bottleneck diagnosis

The bounded search remains inconclusive about a dominant response-optimization versus continuation-estimation bottleneck.

Frozen baselines are critic_s0/critic_s1 from 817e429. Only 32 response parameters change: all base diagonal weights and 16 head offsets, actor and nominal policy remain fixed. Three starts per baseline, 24 updates each; same averaged-positive region critic and pessimistic surrogate. No diagonal-fit rejection.

## Independent full-rollout results

Normalized discounted return (.05 times raw return), lower is more pessimistic. Each selected model was chosen using 16 separate paths per training root. Final results use 64 new paths per held-out root, candidate ETT at every step. Public-reset results use 128 separate paths.

- Baseline 0: 0.59177; selected s0_k0_u0: 0.59177; difference +0.00000; root 95% CI [+0.00000, +0.00000]; MC 95% CI [+0.00000, +0.00000].
  Public START: 0.52080 -> 0.52080; delta +0.00000, paired 95% CI [0. 0.].
- Baseline 1: 0.60129; selected s1_k2_u0: 0.59335; difference -0.00794; root 95% CI [-0.01299, -0.00243]; MC 95% CI [-0.01336, -0.00251].
  Public START: 0.50532 -> 0.48411; delta -0.02121, paired 95% CI [-0.05428568  0.0121442 ].

## Critic versus matched MC

9/18 raw opposite signs; 0/18 resolved opposite signs under both declared intervals. Each pair shares query states/actions, candidate successors and next actions between critic and MC; MC continues under the frozen pre-update ETT. These are one-step local comparisons, not full-kernel evaluations.

- s0_k0_u1: critic +0.00767; root 95% CI [-0.00429, +0.02762]; MC 95% CI [-0.00020, +0.01554]; MC +0.00100; root 95% CI [-0.00696, +0.01097]; MC 95% CI [-0.00785, +0.00985]; MC-minus-critic -0.00667; root 95% CI [-0.02743, +0.01143]; MC 95% CI [-0.01836, +0.00502].
- s0_k0_u12: critic +0.00155; root 95% CI [-0.00248, +0.00790]; MC 95% CI [-0.00135, +0.00445]; MC -0.02560; root 95% CI [-0.07822, +0.00178]; MC 95% CI [-0.07224, +0.02105]; MC-minus-critic -0.02715; root 95% CI [-0.07900, -0.00042]; MC 95% CI [-0.07385, +0.01956].
- s0_k0_u24: critic -0.00075; root 95% CI [-0.00179, -0.00010]; MC 95% CI [-0.00200, +0.00049]; MC +0.01619; root 95% CI [-0.00419, +0.05330]; MC 95% CI [-0.01336, +0.04574]; MC-minus-critic +0.01695; root 95% CI [-0.00347, +0.05412]; MC 95% CI [-0.01291, +0.04681].
- s0_k1_u1: critic -0.00693; root 95% CI [-0.02389, +0.00182]; MC 95% CI [-0.01002, -0.00383]; MC -0.02440; root 95% CI [-0.08364, +0.00000]; MC 95% CI [-0.07744, +0.02865]; MC-minus-critic -0.01747; root 95% CI [-0.05949, +0.00260]; MC 95% CI [-0.07063, +0.03570].
- s0_k1_u12: critic -0.00112; root 95% CI [-0.00318, +0.00004]; MC 95% CI [-0.00150, -0.00074]; MC +0.00369; root 95% CI [-0.00335, +0.01637]; MC 95% CI [-0.00776, +0.01514]; MC-minus-critic +0.00481; root 95% CI [-0.00278, +0.01946]; MC 95% CI [-0.00671, +0.01634].
- s0_k1_u24: critic -0.00087; root 95% CI [-0.00210, +0.00017]; MC 95% CI [-0.00169, -0.00005]; MC +0.00141; root 95% CI [-0.00159, +0.00657]; MC 95% CI [-0.00281, +0.00563]; MC-minus-critic +0.00228; root 95% CI [-0.00136, +0.00799]; MC 95% CI [-0.00189, +0.00646].
- s0_k2_u1: critic +0.00790; root 95% CI [-0.00043, +0.02215]; MC 95% CI [+0.00507, +0.01073]; MC +0.01913; root 95% CI [-0.00031, +0.04582]; MC 95% CI [-0.00244, +0.04071]; MC-minus-critic +0.01123; root 95% CI [-0.00230, +0.02814]; MC 95% CI [-0.01080, +0.03327].
- s0_k2_u12: critic +0.00218; root 95% CI [-0.00285, +0.00732]; MC 95% CI [+0.00094, +0.00342]; MC -0.00303; root 95% CI [-0.01185, +0.00273]; MC 95% CI [-0.00846, +0.00241]; MC-minus-critic -0.00521; root 95% CI [-0.01368, +0.00282]; MC 95% CI [-0.01082, +0.00041].
- s0_k2_u24: critic -0.00205; root 95% CI [-0.00773, +0.00085]; MC 95% CI [-0.00338, -0.00072]; MC +0.00178; root 95% CI [+0.00000, +0.00610]; MC 95% CI [-0.00171, +0.00527]; MC-minus-critic +0.00383; root 95% CI [-0.00085, +0.01388]; MC 95% CI [-0.00001, +0.00767].
- s1_k0_u1: critic +0.00620; root 95% CI [-0.00726, +0.02400]; MC 95% CI [+0.00191, +0.01049]; MC -0.06464; root 95% CI [-0.15269, +0.00000]; MC 95% CI [-0.15110, +0.02181]; MC-minus-critic -0.07084; root 95% CI [-0.16132, -0.00262]; MC 95% CI [-0.15679, +0.01510].
- s1_k0_u12: critic -0.00754; root 95% CI [-0.01783, +0.00040]; MC 95% CI [-0.01007, -0.00501]; MC -0.00116; root 95% CI [-0.00274, +0.00000]; MC 95% CI [-0.00732, +0.00501]; MC-minus-critic +0.00638; root 95% CI [-0.00046, +0.01543]; MC 95% CI [-0.00099, +0.01376].
- s1_k0_u24: critic -0.01855; root 95% CI [-0.05852, +0.00225]; MC 95% CI [-0.02276, -0.01433]; MC -0.00499; root 95% CI [-0.02348, +0.00985]; MC 95% CI [-0.02426, +0.01428]; MC-minus-critic +0.01356; root 95% CI [-0.01683, +0.05447]; MC 95% CI [-0.00593, +0.03304].
- s1_k1_u1: critic -0.01970; root 95% CI [-0.05932, +0.00327]; MC 95% CI [-0.02894, -0.01047]; MC +0.01192; root 95% CI [-0.01161, +0.03978]; MC 95% CI [-0.01238, +0.03622]; MC-minus-critic +0.03162; root 95% CI [-0.00373, +0.09489]; MC 95% CI [+0.00587, +0.05737].
- s1_k1_u12: critic +0.00602; root 95% CI [+0.00004, +0.01489]; MC 95% CI [+0.00267, +0.00937]; MC +0.00607; root 95% CI [+0.00000, +0.01308]; MC 95% CI [-0.00124, +0.01338]; MC-minus-critic +0.00005; root 95% CI [-0.00496, +0.00530]; MC 95% CI [-0.00695, +0.00705].
- s1_k1_u24: critic +0.00270; root 95% CI [+0.00007, +0.00780]; MC 95% CI [-0.00090, +0.00631]; MC +0.00009; root 95% CI [-0.00517, +0.00497]; MC 95% CI [-0.00467, +0.00484]; MC-minus-critic -0.00261; root 95% CI [-0.00713, -0.00007]; MC 95% CI [-0.00692, +0.00169].
- s1_k2_u1: critic +0.00517; root 95% CI [+0.00068, +0.01033]; MC 95% CI [+0.00300, +0.00733]; MC -0.00221; root 95% CI [-0.00684, +0.00000]; MC 95% CI [-0.00654, +0.00212]; MC-minus-critic -0.00737; root 95% CI [-0.01533, -0.00069]; MC 95% CI [-0.01305, -0.00170].
- s1_k2_u12: critic +0.00051; root 95% CI [-0.00008, +0.00130]; MC 95% CI [-0.00021, +0.00123]; MC +0.00434; root 95% CI [+0.00000, +0.01489]; MC 95% CI [-0.00417, +0.01286]; MC-minus-critic +0.00383; root 95% CI [-0.00102, +0.01435]; MC 95% CI [-0.00480, +0.01246].
- s1_k2_u24: critic -0.02207; root 95% CI [-0.06709, +0.00489]; MC 95% CI [-0.03842, -0.00572]; MC -0.00469; root 95% CI [-0.03207, +0.02460]; MC 95% CI [-0.03261, +0.02323]; MC-minus-critic +0.01738; root 95% CI [-0.01116, +0.05151]; MC 95% CI [-0.00607, +0.04084].

## Local improvement versus full use

- s0_k0: last-step full return change +0.00077; root 95% CI [-0.00064, +0.00210]; MC 95% CI [-0.00094, +0.00249]; final-minus-start +0.00506; root 95% CI [-0.00025, +0.01034]; MC 95% CI [-0.00083, +0.01096]; response parameter L2 change 1.7568.
- s0_k1: last-step full return change -0.00100; root 95% CI [-0.00176, -0.00016]; MC 95% CI [-0.00199, -0.00000]; final-minus-start +0.00120; root 95% CI [-0.00325, +0.00581]; MC 95% CI [-0.00424, +0.00663]; response parameter L2 change 1.6603.
- s0_k2: last-step full return change -0.00023; root 95% CI [-0.00111, +0.00065]; MC 95% CI [-0.00151, +0.00105]; final-minus-start +0.00691; root 95% CI [+0.00199, +0.01197]; MC 95% CI [+0.00181, +0.01201]; response parameter L2 change 1.6081.
- s1_k0: last-step full return change -0.00061; root 95% CI [-0.00317, +0.00179]; MC 95% CI [-0.00277, +0.00155]; final-minus-start -0.01121; root 95% CI [-0.01812, -0.00380]; MC 95% CI [-0.01879, -0.00363]; response parameter L2 change 1.5460.
- s1_k1: last-step full return change -0.00138; root 95% CI [-0.00315, +0.00036]; MC 95% CI [-0.00340, +0.00063]; final-minus-start -0.00285; root 95% CI [-0.00773, +0.00213]; MC 95% CI [-0.00968, +0.00398]; response parameter L2 change 1.3917.
- s1_k2: last-step full return change -0.00007; root 95% CI [-0.00274, +0.00252]; MC 95% CI [-0.00278, +0.00264]; final-minus-start -0.00187; root 95% CI [-0.00715, +0.00384]; MC 95% CI [-0.00887, +0.00513]; response parameter L2 change 1.6474.

0/6 final updates have resolved local MC decreases without resolved full-rollout decreases. Lack of resolution is not proof of no effect; inspect both intervals above.

## Verification and cost

5,687,360/6,000,000 computed model transitions (including masked MC padding), 144 response updates, 47,400 critic steps. Zero native steps or actor/nominal updates. Raw MC targets, paired intervals, optimizer proposals, selection means, and full returns were recomputed from saved artifacts. All 16 diagonal offsets remain exact, every proposal produces identical paired diagonal samples, and all final geometry/history/Lipschitz checks passed. No minimum action sensitivity, forced transition, reward change or new geometry was introduced.

## Decision and scope

Next prioritize a separately bounded replication of the largest unresolved matched comparisons and full-rollout differences, rather than an architecture sweep.

Local estimation error, response exploration and local-to-full mismatch can coexist. Fresh randomness on previously inspected held-out roots does not establish generalization to a new root distribution. Intervals are descriptive and not simultaneous across the 18 checks. Selection samples never enter final evaluation. Any recovered pessimism is relative to these baselines and this fixed family; neither global worst-case recovery nor native-policy benefit is established. The actor remains fixed, and no native environment evaluation was performed. No further search or experiment followed.

Protocol: PROTOCOL.md. Complete results: results.json. Raw comparisons: checks/*/matched_mc.npz. Full models: candidate_pool.npz and pre_final.npz. Independent paths: evaluation/. No existing artifact was overwritten.
