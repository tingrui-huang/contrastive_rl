# G3 baselines: vanilla CRL without the absorbing ETT, natively, on the G3 seeds

Sealed before execution. Purpose: put the plain CRL baselines next to the G3
policies on the same 200 paired native episodes (reset seeds 9300000 + i,
actor innovation fold_in(PRNGKey(9400000 + t), i), horizon 50, region-fork
route classification of the faithful-actor audit / G3 addendum).

Policies:

1. `vanilla_bc0.5` — the original offline CRL checkpoint as trained
   (`alpha0_seed0/final.pkl`, 150,000 steps, bc_coef 0.5, no failure
   negatives, no ETT, no continuation). Evaluated as is.
2. `vanilla_critic_bc0.2` — the same checkpoint's critic left untouched, its
   actor continued for the sealed 1,000 updates at bc = 0.2 on the sealed
   uniform actor batches and keys (exactly the G2 actor stage, but with the
   original critic instead of the O or P critic). New, one training run of
   the actor only.
3. `O_bc0.2` — G3's baseline pair member (recorded-future critic continuation
   + bc 0.2 actor); trajectories reused from G3 (same seeds).
4. `P_bc0.2` — the method (absorbing-ETT critic + bc 0.2 actor); reused from G3.
5. `P_bc0.5`, `O_bc0.5` — reused from G3.

Metrics: reach (native reward > 0), strict success (min distance < 0.5),
discounted and undiscounted return, absorbed, lower-route / shortcut /
stuck-other usage, first departure from the start/fork region into (1,2) or
(2,3), time-to-goal given reach. Paired bootstrap (2,000 replicates, seed 0)
over the 200 episode ids for every policy minus `P_bc0.2` and for
`vanilla_critic_bc0.2` minus `vanilla_bc0.5`.

No environment call is made for the reused policies. No commit until
reviewed.
