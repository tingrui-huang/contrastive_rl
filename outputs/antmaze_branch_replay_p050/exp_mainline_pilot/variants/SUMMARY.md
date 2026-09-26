# Pre-registered single-change variants of the pilot: four reference points, none a fix

Each variant changes exactly one thing against the pilot (same data, anchors,
CF branches, streams and seeds, losses unless stated, critic lr, initialisation
from the start agent, 30,000 optimizer updates, evaluation on the same 300
episodes, seed 3909, mode policy); O and CF both, three paired seeds; sealed
manifest with criteria before training (`<variant>/manifest.json`); results
in `<variant>/REPORT.md`, `results.json`; continuation from the base CF
policy's own detour-entrance handover states in
`../diag_traj/cont_variant_<variant>.json`.  Status (user, 2026-09-19):
reference points only; "rebalance BC / protect walking by region" are
withdrawn as default fix proposals.

| variant | change | CF success s0 / s1 / s2 | CF detour | CF timeout | O success | continuation from the entrance, CF (s0 / s1 / s2) | criterion 1 (success vs base CF) | route gain vs O(variant) |
|---|---|---|---|---|---|---|---|---|
| base pilot | -- | 0.267 / 0.320 / 0.480 | 0.11 / 0.17 / 0.50 | 0.11 / 0.11 / 0.26 | 0.250 / 0.263 / 0.250 | 0.75 / 0.42 / 0.48 | -- | +0.10 (3/3) |
| actor_lr1e-4 | actor lr 3e-4 -> 1e-4 | 0.270 / 0.297 / 0.297 | 0.05 / 0.09 / 0.06 | 0.02 / 0.03 / 0.01 | 0.247 / 0.250 / 0.247 | 0.75 / 0.67 / 0.71 (recovered) | -0.068, NOT MET | +0.04 (3/3) |
| bc0.02 | BC 0.05 -> 0.02 | **0.000 / 0.000 / 0.000** | 0.12 / 0.40 / 0.02 (turns, never arrives) | 1.00 / 1.00 / 1.00 | 0.203 / 0.240 / 0.223 | 0 / 0 / 0 | -0.356, NOT MET | -- (no walking) |
| bc0 | BC 0 | **0.000 / 0.000 / 0.000** | 0.00 / 0.00 / 0.01 | 1.00 / 1.00 / 1.00 | **0.000 / 0.000 / 0.000** | 0 / 0 / 0 | -0.356, NOT MET | -- (no walking, either arm) |
| anchor_start0.5 | + 0.5 * \|\|tanh(loc) - tanh(loc_start)\|\|^2 (bc 0.05 kept) | 0.247 / 0.290 / 0.257 | 0.00 / 0.09 / 0.03 | 0.00 / 0.04 / 0.01 | 0.260 / 0.273 / 0.257 | 0.75 / 0.67 / 0.62 (recovered) | -0.091, NOT MET | +0.00 (1/3): gone |
| **critic_clip0.1** | clip_by_global_norm(0.1) on the critic gradient before Adam, both arms | **0.457 / 0.513 / 0.440** | 0.57 / 0.52 / 0.36 | 0.35 / 0.18 / 0.16 | 0.257 / 0.247 / 0.260 | 0.75 / 0.75 / 0.67 (>= start) | +0.114 (2/3) | **+0.216 (0.026, 3/3): MET** |

Readings (against the readings fixed in each manifest):

- **actor_lr1e-4**: walking recovered, route gain shrinks to a third; the
  route change and the walking loss scale together with the update size.
- **bc0.02 / bc0**: BC below 0.05 removes walking altogether -- the actor
  follows the critic term off the walkable torque manifold (all 300
  episodes time out, no deaths because the hazards are never reached; with
  bc 0 the O arm does the same).  bc0.02 still turns north in 12-40 % of
  episodes, so the route preference is there, but no walking is left to
  carry it.  BC is what keeps the actor on the manifold; at 0.05 there is
  no slack below.
- **anchor_start0.5**: the trust region to the start policy holds the
  mid-route torques (continuation recovered to 0.62-0.75, timeouts 0-4 %)
  but at this coefficient it also holds the reset torques: the detour rate
  returns to the O level (+0.00 over O(variant)) and CF - start is +0.02.
  A smaller coefficient would sit between this and the base -- the same
  one-knob trade-off the learning rate showed.

Together with diag_traj sections 9b-9c (the full actor objective is at a
tie at the reset rows; the mid-route regression is interference from the
majority rows): the four variants move along one axis -- how far the actor
is allowed to leave the start policy -- and on that axis the route gain and
the walking loss come together.  None separates them, and none of them is
proposed as a fix.  Checkpoints stay on the nodes.

**critic_clip0.1 (2026-09-19, later; `critic_clip0.1/SUMMARY.md`)** is not
a one-axis variant of the actor: it removes the critic runaways
(`../diag_replay/spike/SUMMARY.md`) in both arms.  No spike in any of the
six runs; O unchanged (= start); CF - O success +0.216 (seed s.e. 0.026,
3/3), the mainline rule met; detour 0.36-0.57, deaths 0.19-0.40, hazard
success 0.31-0.43; timeouts 0.16-0.35 remain.  Route ledger (corrected
2026-09-19): the far route is taken in 0.57 / 0.52 / 0.36 of the episodes
and completed in 0.65 / 0.79 / 0.81 of those (pooled 0.74; every
far-route loss a timeout, none a death); the rest are shortcut deaths and
no-route timeouts.  Disclosed as an optimizer-stabilisation change.
