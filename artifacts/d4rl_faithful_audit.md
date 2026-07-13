# Original-vs-current audit and the D4RL-faithful reproduction branch

Sources: vendored `crl/assets/d4rl_ant.xml` (verbatim from
Farama-Foundation/D4RL `d4rl/locomotion/assets/ant.xml`), D4RL
`locomotion/maze_env.py` + `locomotion/ant.py` (fetched), and this repo's
original-code snapshot (`ant_envs.py`, `contrastive/*`, `lp_contrastive.py`).

| item | original (as shipped, 2022) | current Gymnasium branch | d4rl-faithful branch (`d4rl_ant_umaze_gfull`) |
|---|---|---|---|
| ant XML | d4rl locomotion ant.xml | gymnasium-robotics ant.xml | **verbatim vendored d4rl XML** |
| actuator gear / ctrlrange | gear 1, ctrl ±30 (max torque 30) | gear 150, ctrl ±1 (max torque 150) | **gear 1, ctrl ±30** |
| action scaling | acme CanonicalSpecWrapper: [-1,1] → ±30 | native [-1,1] | **ctrl = 30·clip(a,−1,1)** (verified) |
| integrator / timestep | RK4 / 0.02 | Euler / 0.01 | **RK4 / 0.02** (verified) |
| frame skip → dt/env-step | 5 → **0.1 s** | 5 → 0.05 s | **5 → 0.1 s** (verified) |
| termination | done always False; StepLimit 700 | fixed 700 (umaze) / 300 (open-near) | **False; 700 steps** |
| observation layout | expose_all_qpos: [qpos(15), qvel(14)] = 29 | identical (verified bit-exact) | identical (verified bit-exact) |
| goal slice | start 0 / end −1 ⇒ FULL 29-dim state | 2-dim xy (open-near); 13/29 in ablation | **goal_indices = range(29)** |
| goal construction | ant teleported to goal xy, settled 50 zero-action steps, full obs snapshot; reward target = settled xy | commanded xy (near); settled snapshot in rich arms (qvel zeroed) | **settled snapshot, post-reset qvel kept, reward target = settled xy** |
| future relabeling | geometric discount^(j−i), full-state goals, cross-traj negatives | same mechanism, sliced goals | same mechanism, full-state goals (bit-exact gate) |
| horizon sampling | Reverb flatten_fn categorical | Gumbel-max equivalent (audited 21/21) | same |
| reset distribution | non_zero_reset: 50 % weighted free cell / 50 % goal sampler; goal from G cells ±0.25·S noise | open-near: cell-targeted, d0∈[1,4.5] | **50 % uniform open cell / 50 % goal sampler, G cells ±1.0 noise** (uniform instead of accessibility-weighted — documented approximation) |
| maze | U_MAZE 5×5, scaling 4.0, origin at R | AntMaze_Open / UMaze v5, scaling 4 | **U_MAZE 5×5, scaling 4.0, origin at R** (wall count gate) |
| actors | 4 async | 1 | 1 (documented divergence) |
| replay | Reverb episodes, 1M cap | numpy trajectory ring, 1M cap | same as current |
| entropy | α = 0 fixed (state-based envs) | α=0 collapses; adaptive target −8 verified stable | adaptive target −8 + guards (nominal); α=0 faithful variant possible later |
| training budget | 1M env steps | 30–50k qualifications | 30k short qualification |

## Pre-training gates (artifacts/d4rl_ant_verification.json — ALL PASS)

- **Physics**: timestep 0.02, RK4, 8 actuators ±30 gear 1, dt 0.1, hip range ±30°, wall count = U_MAZE, floor present, ctrl = 30·clip(a) verified numerically.
- **Observation**: flat obs 58; state ≡ [qpos, qvel] bit-exact.
- **Goal**: goal half ≡ settled 29-dim snapshot bit-exact; reward target = settled xy.
- **Relabel**: 256/256 sampled goals are bit-exact strict-future full states.
- **Action sensitivity**: 64 uniform 1-step actions from a reset state — XY spread std **0.0087 m (d4rl) vs 0.0038 m (gymnasium)**, 2.3×; max disp 0.035 vs 0.016 m. The faithful physics carries visibly more action-consequence signal per env step.
- **Coverage**: 840 unique states / 87 cells / 6 goals / 0.99 moving fraction over 6 random episodes.
