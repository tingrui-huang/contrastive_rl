# Fixed-ETT query-coverage intervention

Sealed before generating either candidate replay or training either candidate critic. Source branch `feature/pointmaze-causal-transition`, commit `d5e2da05f4a3f7174bc9f6130f7ad929f1b50823`. This is one bounded seed and tests sufficiency of query coverage at the fixed budget; it is not a general causal-identification or seed-robustness claim.

## Question and one changed factor

The fixed learned ETT can produce better down-first continuations on the previously inspected 16-root model probe, while the saved P critic ranks right on all 16 roots. Test whether explicitly exposing the unchanged CRL learner to the ETT's down/right consequences at fork roots is sufficient to learn the held-out task-goal ordering.

Freeze the selected ETT checkpoint, nominal model, observational continuation actor, original-data half, learner architecture, original sigmoid-NCE, geometric discount-0.95 future sampler, actor objective, BC coefficient 0.05, optimizer, initialization seed, budgets, and final-iterate rule. The only experimental difference is first-step `xq`: arm B samples it from the frozen observational actor; arm C assigns a sealed balanced six-action down/right neighborhood. Both arms use identical root/context schedules and the same nominal and random streams. All later actions come from the same frozen observational actor evaluated on the arm's generated current state.

This intervention changes the actual synthetic trajectories, their later states, and consequently both critic and actor training contexts. Arm C's BC term also sees its covered first-step actions. Therefore actor behavior cannot be attributed to the critic alone; the primary critic ranking is the cleanest answer to the stated question.

## Roots, actions, and complete continuations

Construction roots come only from the already frozen supervised training partition: `time == 1`, current XY in cell `(1,3)`, then exact-F4 deduplication. The expected result is 198 unique roots. The separately saved 16 coherent roots in `pointmaze_matched_fork_20260914_v1/root_selection.npz` are evaluation-only and must have zero exact-F4 overlap with construction roots.

Every construction root receives two contexts. A seed-2026091512 permutation selects 154 of 198 roots for a third, yielding 550 contexts. Each context samples one nominal `xb`, shared across its six paths. Arm C's actions, in sealed order, are:

- down neighborhood: `[-0.01,-0.99995]`, `[0,-1]`, `[0.01,-0.99995]`;
- right neighborhood: `[0.99995,-0.01]`, `[1,0]`, `[0.99995,0.01]`.

The narrow approximately 0.57-degree neighborhoods are fixed before generation because all three down actions then have noiseless endpoints in cell `(1,2)` and all three right actions in `(2,3)` for every eligible training root, including roots close to a cell edge.

Arm B has six matched paths per context but independently samples each first query from the frozen actor. Transition innovations are paired by array position across B/C. These roots occur at task time 1, so each path generates the full 49 remaining transitions. Storage stays compatible with the 51-row replay shape: synthetic rows 0--49 are valid, row 50 duplicates row 49 and is excluded structurally by `lengths=50`. All predicted deaths, out-of-hazard onsets, stationary corrections, failures, and successes remain; no trajectory is selected or rejected by outcome. Persistent failure and exact F4 shifts remain structural.

Each replay has exactly 6,600 episodes: the exact 3,300 original episodes already retained in the d5e2da0 P replay plus its arm's 3,300 new synthetic paths. Both therefore have identical episode counts, valid-length patterns, original half, and sampling-index stream. The replay sampler stays ordinary and uniform; no bucket balancing, new loss, route weighting, or hand-written failure correction is enabled.

## Fixed CRL

Independently instantiate B and C from hash-verified identical seed-0 learner state. Run 30,000 learner updates per arm, batch 256, ten updates per scan, representation 64, hidden layers 256-256, discount 0.95, random goals 0.5, entropy 0, BC 0.05, learning rates 3e-4. No TD, CPC, GCBC, twin Q, AWR, ranking term, failure bank, anchor cut, or balanced replay. No native evaluation during training; use only fixed final checkpoints.

Materialize the exact shared 30,000 x 256 `(trajectory, anchor, future)` sampler stream after reproducing offline-audit RNG consumption. Report how often covered step-0 anchors and their task-near futures actually enter NCE. This is a distribution audit, not another training knob.

## Held-out decision and secondary outcomes

At the 16 evaluation-only coherent alive fork roots, evaluate each fixed-final critic on exact down `[0,-1]` and right `[1,0]` under the canonical stationary F4 goal. Bootstrap roots (5,000 replicates, seed 2026091513). Declare `QUERY_COVERAGE_SUFFICIENT_FOR_HELDOUT_CRITIC_RANKING_AT_FIXED_BUDGET` only if all three sealed gates pass:

1. arm C mean down-minus-right logit has root-bootstrap 95% lower endpoint above zero;
2. paired C-minus-B change in that margin has 95% lower endpoint above zero;
3. C ranks down above right on at least 12 of 16 roots.

Otherwise declare only that this coverage intervention was not sufficient at this budget. If B alone becomes positive, report that the common fork-root distribution rather than the sealed query contrast may be sufficient; do not relabel it a C-specific success.

Secondary read-only diagnostics: actor action distribution on the same held-out roots; NCE counts by candidate and future distance to the canonical full-F4 goal; generated route, reach, absorption, out-of-hazard onset, and near-stop nominal counts. Finally evaluate both fixed actors under mode and sampled actions on the same 200 fresh native reset seeds. These secondary outcomes never select checkpoints or extend training.

The existing post-hoc model probe is supporting motivation only. It used no native interaction, and its historical native reference used a different continuation actor. This experiment does not treat model success as real-environment success and does not repair the known onset extrapolation errors.
