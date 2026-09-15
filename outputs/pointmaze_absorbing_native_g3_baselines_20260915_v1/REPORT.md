# Vanilla CRL baselines next to the G3 policies (same 200 paired native episodes)

Date: 2026-09-15. Sealed [PROTOCOL.md](PROTOCOL.md); driver [run.py](run.py); [checkpoint_manifest.json](checkpoint_manifest.json); [results.json](results.json); [per_episode_regionfork.csv](per_episode_regionfork.csv); new trajectories `native_vanilla_bc0.5.npz`, `native_vanilla_critic_bc0.2.npz`; the G3 trajectories are reused (hashes in the manifest). One actor training run (1,000 updates, original critic untouched), 20,000 new native steps, zero critic updates.

Same seeds, noise/bit streams and actor innovations as G3 (reset `9300000 + i`, innovation `fold_in(PRNGKey(9400000 + t), i)`); route classification uses the region-fork rule (first departure from the start/fork region {(0,3), (1,3)} into (1,2) = lower route, hazard-corridor landing before that = shortcut).

## Table

| policy | what it is | reach (success) | strict success | absorbed | lower route | shortcut | first departure → (1,2) | discounted return | undiscounted return | time-to-goal \| reach |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_bc0.5 | original offline CRL checkpoint as trained (150k steps, bc 0.5) | 0.330 | 0.325 | 0.675 | 0.155 | 0.845 | 0.135 | 3.71 | 13.4 | 9.9 |
| vanilla_critic_bc0.2 | original critic untouched + 1,000 actor updates at bc 0.2 | 0.350 | 0.345 | 0.655 | 0.085 | 0.915 | 0.075 | 4.33 | 14.9 | 7.9 |
| O_bc0.5 | recorded-future critic continuation + bc 0.5 actor (G1 O actor) | 0.390 | 0.385 | 0.615 | 0.210 | 0.790 | 0.175 | 4.41 | 15.9 | 9.6 |
| O_bc0.2 | recorded-future critic continuation + bc 0.2 actor | 0.400 | 0.395 | 0.605 | 0.140 | 0.860 | 0.135 | 4.91 | 17.0 | 8.0 |
| P_bc0.5 | **absorbing-ETT critic** + bc 0.5 actor (G1 P actor) | 0.435 | 0.430 | 0.550 | 0.305 | 0.695 | 0.235 | 4.15 | 15.8 | 14.2 |
| **P_bc0.2** | **absorbing-ETT critic** + bc 0.2 actor | **0.495** | **0.485** | **0.385** | **0.595** | **0.405** | **0.495** | **5.16** | **19.0** | 11.6 |

Paired differences relative to P_bc0.2 (mean, 95% CI over the 200 episode ids):

| baseline − P_bc0.2 | reach | absorbed | lower route | shortcut | discounted return |
|---|---|---|---|---|---|
| vanilla_bc0.5 | −0.165 [−0.245, −0.090] | +0.290 [+0.220, +0.365] | −0.440 [−0.505, −0.370] | +0.440 [+0.370, +0.510] | −1.45 [−2.32, −0.57] |
| vanilla_critic_bc0.2 | −0.145 [−0.230, −0.065] | +0.270 [+0.190, +0.350] | −0.510 [−0.580, −0.440] | +0.510 [+0.440, +0.585] | −0.83 [−1.81, +0.06] |
| O_bc0.5 | −0.105 [−0.185, −0.025] | +0.230 [+0.160, +0.300] | −0.385 [−0.450, −0.320] | +0.385 [+0.315, +0.455] | −0.74 [−1.63, +0.16] |
| O_bc0.2 | −0.095 [−0.175, −0.020] | +0.220 [+0.145, +0.295] | −0.455 [−0.525, −0.390] | +0.455 [+0.385, +0.530] | −0.25 [−1.19, +0.65] |
| P_bc0.5 | −0.060 [−0.135, +0.020] | +0.165 [+0.095, +0.240] | −0.290 [−0.355, −0.230] | +0.290 [+0.225, +0.355] | −1.01 [−1.82, −0.24] |

Baselines among themselves: vanilla_critic_bc0.2 − vanilla_bc0.5: reach +0.02 [−0.05, +0.08], lower route −0.07 [−0.11, −0.03], shortcut +0.07 — lowering BC under the vanilla critic pushes the actor further onto the shortcut. O_bc0.2 − vanilla_critic_bc0.2: reach +0.05 [+0.02, +0.09], lower +0.055, absorbed −0.05 — the 400 recorded-future critic updates help a little on their own.

## Reading

- Plain CRL at bc = 0.2 without the absorbing ETT (`vanilla_critic_bc0.2`) takes the shortcut 91.5% of the time, is absorbed 65.5% of the time and reaches the goal 35.0%. The same actor stage on the absorbing-ETT critic (`P_bc0.2`) takes the lower route 59.5%, is absorbed 38.5% and reaches the goal 49.5%: +14.5 points reach [+6.5, +23.0], −27 points absorption, +51 points lower-route usage.
- Every baseline critic (vanilla, vanilla + bc 0.2, O at either bc) leaves the actor on the shortcut at 79–92%; only the P critic moves it, and it does so more when the BC weight is lower (P_bc0.5 30.5% → P_bc0.2 59.5%), whereas lowering BC under any non-pessimistic critic moves the actor *toward* the shortcut.
- The remaining gap to a full success is on the lower route itself: P_bc0.2 reaches the goal on 64% of its lower-route episodes (time-to-goal 12.7) and on 28% of its shortcut episodes; the 40.5% of episodes that still take the shortcut are absorbed at 72%.

## Route definitions, strict and loose

"Lower route" above is the first departure from the start/fork region into (1,2) before any hazard landing; an episode that dips into (1,2) and then returns to the corridor and crosses the swamp still counts as lower. The strict definition is reaching the lower corridor proper (y < 2):

| policy | lower (loose) | reached y < 2 (strict) | lower episodes with a later hazard landing | reach \| y < 2 |
|---|---:|---:|---:|---:|
| vanilla_bc0.5 | 0.155 | 0.115 | 0.355 | 0.957 |
| vanilla_critic_bc0.2 | 0.085 | 0.055 | 0.412 | 0.909 |
| O_bc0.5 | 0.210 | 0.160 | 0.262 | 1.000 |
| O_bc0.2 | 0.140 | 0.120 | 0.143 | 1.000 |
| P_bc0.5 | 0.305 | 0.250 | 0.262 | 0.880 |
| P_bc0.2 | 0.595 | 0.465 | 0.252 | 0.710 |

Under the strict definition P_bc0.2 uses the far route 46.5% of the time against 5.5% for plain CRL at the same bc and 11.5% for the checkpoint as trained. The baselines' far-route share is not new: the 09-14 native evaluations of the same original actor reported first-hazard entry 0.86 with failure 0.70 and reward 0.30 and no surviving unrewarded episode, i.e. about 14% of episodes reached the goal without ever landing in the swamp (oracle training pilot, supervised repair, matched failure audit). It comes from the 5% forced-safe teacher episodes and the random-policy down moves in the data, expressed through a stochastic tanh-Gaussian actor whose down-right diagonals slide into (1,2). The earlier 2-D swamp environment and the discrete corridor, where the baseline took the shortcut 100% of the time, used a different observation and a different actor.

No commit until reviewed.
