# The averaged critic enables bounded pessimistic ETT updates

**Critic estimation did not block pessimistic optimization in this experiment.**
Both the averaged-NCE and MC arms lowered independent full-rollout returns
relative to diagonal-only and initialization in both seeds. Replacing the
critic with MC did not improve the result.

All arms started from the same 48-parameter ETT and made three paired update
attempts per seed. Actor and nominal policy were frozen. The critic refreshed
on its current ETT before each update; MC continued under its frozen pre-update
ETT after the candidate first transition. Final evaluation applied each final
kernel throughout, using 64 independent rollouts at each of 36 held-out roots.

Paired changes versus diagonal-only, with 95% episode-bootstrap intervals
(negative means more pessimistic):

- Seed 0: critic **−.02050 [−.02414, −.01687]**; MC **−.01686 [−.01901, −.01475]**.
- Seed 1: critic **−.01316 [−.01620, −.01000]**; MC **−.01158 [−.01447, −.00871]**.

Conditional paired-MC intervals also exclude zero for all four reductions.
MC-minus-critic is +.00363/+.00159 in seeds 0/1: the critic yields slightly
lower returns under both declared interval checks. Seed 1's conditional-MC
lower bound is only +.00002; this is not a general superiority claim.

**Shared limits were active.** Both pessimistic arms hit both block step caps
on every proposal and rejected their last two seed-1 proposals. Overall,
14/18 proposals were accepted. Validation diagonal degradation ranged from
.00470 to .01608 for pessimistic kernels, with intervals below the .02
allowance. All final train guards, geometry/history checks, updated-diagonal
identity checks, and L=1 action-Lipschitz tests passed. Return reductions
accompany allowed fitting deterioration; its contribution was not isolated.

Used **1,954,528/3,000,000** model transitions and 2,400 critic-refresh steps.
The 22 tests and stored-array audit passed. These are model-internal results
on previously inspected held-out roots, with fresh evaluation noise. They do
not establish native worst cases, estimator equivalence, or global convergence.
No actor training, extra sweep, extension, commit or push followed.

See [REPORT.md](REPORT.md) for all initialization contrasts, intervals,
acceptances and separate simulation costs, and [PROTOCOL.md](PROTOCOL.md)
for the analysis sealed before the run.
