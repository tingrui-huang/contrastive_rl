# G4: from-scratch offline CRL on the absorbing-freeze dataset (P) vs the recorded dataset (O), bc = 0.2, three seeds, GPU

Date: 2026-09-15. Server: RTX 3060 Ti, Python 3.12, `jax[cuda12]==0.10.2`, repo at `7e1837d`. Launcher `scripts/run_pointmaze_absorbing_g4.py` via `scripts/run_pointmaze_absorbing_g4_sweep.sh` (three concurrent detached runs, ~10 min each at ~270 learner steps/s). Datasets: O = `swamp_windy_f4_merged_s0.npz` (content sha `ad8b4470…`), P = `swamp_windy_f4_absorbing_s0.npz` (content sha `6e046bdc…`, built by `scripts/build_pointmaze_absorbing_dataset.py`, seed 0; [manifest](datasets/swamp_windy_f4_absorbing_s0.npz.manifest.json)). Every run: the established windy-F4 recipe of the original `alpha0_seed0` checkpoint — 150,000 learner steps × 10 SGD steps, batch 256, repr 64, (256, 256), discount 0.95, random_goals 0.5, entropy 0, alpha 0, no bank — with **bc = 0.2** for both arms; `--diff` asserts the arms differ only in `offline_dataset`. Per-run [logs](logs/pointmaze_absorbing_g4/), [provenance](runs/pointmaze_absorbing_g4/), `final.pkl` / `best.pkl` checkpoints and `metrics.json` are included. Native evaluation: `scripts/eval_pointmaze_native_routes.py`, 200 paired seeds (`9300000 + i`), mode and sampled protocols, region-fork route classification ([results](results/pointmaze_absorbing_g4/)).

## What the P dataset is

The absorbing-freeze ETT applied along the recorded trajectories: at each alive step the recorded action is executed, a frozen-nominal draw is the behaviour alternative, and the Manski onset (landing-bin disagreement with the recorded landing in the data-discovered support {(3,3), (4,3), (5,3)}) freezes the trajectory from that landing on, exactly as the environment freezes an absorbed agent; the XY channel is the recorded outcome itself, only the freeze channel is modelled. Seed 0 adds an onset to 4,631 of 6,600 episodes (mean onset step 5.7; onset landings (3,3) 2,774 / (4,3) 1,391 / (5,3) 466). Audit-only breakdown by behaviour label (never seen by the learner): forced-safe 0%, random 46%, immediate-shortcut 70%, wait-shortcut 89% — the teacher's clear-cell crossings and in-swamp waits are exactly what a blind agent cannot reproduce, and the bound freezes them. All eight offline audit gates pass on both datasets.

## Training curves (repo greedy evaluation, 50 episodes every 10k steps)

| run | 10k … 150k success | mean | evals ≥ 0.8 | final | best |
|---|---|---:|---:|---:|---:|
| O s0 | 0.32 0.26 0.38 0.26 0.24 0.40 0.36 0.32 0.42 0.36 0.34 0.30 0.26 0.42 0.26 | 0.33 | 0/15 | 0.26 | 0.42 |
| O s1 | 0.34 0.36 0.30 0.30 0.42 0.48 0.36 0.40 0.44 0.34 0.36 0.26 0.42 0.40 0.36 | 0.37 | 0/15 | 0.36 | 0.48 |
| O s2 | 0.48 0.58 0.32 0.30 0.36 0.36 0.38 0.34 0.36 0.28 0.24 0.20 0.52 0.42 0.20 | 0.36 | 0/15 | 0.20 | 0.58 |
| **P s0** | **1.00 1.00 1.00 1.00** 0.42 0.84 0.38 0.36 0.44 0.86 0.42 0.64 0.82 0.40 **0.88** | 0.70 | 8/15 | 0.88 | 1.00 |
| **P s1** | **1.00** 0.40 **1.00 1.00** 0.36 0.32 0.44 0.30 0.32 0.84 **1.00** 0.38 0.34 0.40 0.28 | 0.56 | 5/15 | 0.28 | 1.00 |
| **P s2** | **1.00 1.00** 0.32 0.34 0.48 0.34 **1.00** 0.46 0.40 0.40 0.26 0.34 0.32 0.20 0.24 | 0.47 | 3/15 | 0.24 | 1.00 |

Every P seed reaches the detour mode (≈ 1.0) repeatedly and no O seed ever exceeds 0.58. The P runs oscillate between a detour mode and a shortcut mode (≈ 0.35, the O level) — the same phenomenon the continuous Manski port recorded in R5 ("the natural curves of all three seeds oscillate between a detour mode (~1.0) and a shortcut mode (~0.7); s0's perfect final simply stopped at a good moment"). One of three P seeds ends in the detour mode.

## Native evaluation of the final checkpoints (200 paired seeds)

Mode action (the repo's convention):

| final checkpoint | reach | absorbed | lower route | shortcut | discounted return | P − O (paired) |
|---|---:|---:|---:|---:|---:|---|
| O s0 / s1 / s2 | 0.375 / 0.375 / 0.375 | 0.625 | 0.000 | 1.000 | 4.94 | |
| **P s0** | **0.860** | **0.140** | **0.755** | 0.245 | **10.24** | reach +0.485 [+0.415, +0.555], lower +0.755, return +5.30 |
| P s1 | 0.375 | 0.625 | 0.000 | 1.000 | 4.94 | 0 (identical trajectories) |
| P s2 | 0.375 | 0.625 | 0.000 | 1.000 | 4.94 | 0 |

Sampled action (secondary protocol, innovations paired across policies):

| final checkpoint | reach | absorbed | lower route | y < 2 | shortcut | disc. return | P − O reach | P − O lower | P − O absorbed |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| O s0 / s1 / s2 | 0.355 / 0.375 / 0.345 | 0.66 / 0.63 / 0.66 | 0.06 / 0.09 / 0.03 | 0.05 / 0.08 / 0.02 | 0.94 / 0.91 / 0.97 | 4.27 / 4.66 / 4.35 | | | |
| P s0 | 0.620 | 0.380 | 0.520 | 0.440 | 0.480 | 7.28 | +0.265 [+0.195, +0.340] | +0.460 [+0.385, +0.530] | −0.280 |
| P s1 | 0.540 | 0.470 | 0.300 | 0.265 | 0.700 | 6.46 | +0.165 [+0.100, +0.235] | +0.210 [+0.145, +0.270] | −0.160 |
| P s2 | 0.480 | 0.520 | 0.210 | 0.165 | 0.790 | 5.87 | +0.135 [+0.075, +0.195] | +0.180 [+0.130, +0.235] | −0.140 |

Under sampling every P final beats its O twin on reach, absorption and lower-route usage with intervals excluding zero — including the two seeds whose *mode* sits on the shortcut side. `best.pkl` (selected by the in-training greedy evaluation, which leaks a deconfounding model selection and is reported only for completeness): P 1.000 / 1.000 / 1.000 reach and 1.000 lower route for all three seeds under the mode protocol, O 0.375 for all three; under sampling P best 0.695 / 0.650 / 0.660, O best 0.345 / 0.365 / 0.450.

## Why the mode is bistable

Mode actions of the P checkpoints at the start state and at the t = 1 state (x ≈ 1.5), and the physical cell after the t = 1 move as x₁ varies from 1.47 to 1.52 (the range the 0.01 actuator noise produces):

| checkpoint | mode a at t=0 | mode a at t=1 | landing cell for x₁ = 1.47 … 1.52 |
|---|---|---|---|
| P s0 final | (+1.000, −0.043) | (+0.999, **−0.999**) | (1,2) (1,2) (1,2) (1,2) (2,3) (2,3) |
| P s0 best | (+1.000, −0.169) | (+1.000, −0.957) | (1,2) × 6 |
| P s1 final | (+1.000, +0.040) | (+1.000, **−0.741**) | (2,3) × 6 |
| P s1 best | (+1.000, −0.053) | (+0.969, −0.967) | (1,2) × 5, (2,3) |
| P s2 final | (+1.000, +0.047) | (+0.999, **−0.525**) | (2,3) × 6 |
| P s2 best | (+1.000, −0.269) | (+0.996, −0.837) | (1,2) × 6 |

The x-component is saturated at +1 in every checkpoint (the BC term pins it to the data's fork actions, ≈ (1, 0)); what the critic moves is the y-component, between −0.5 and −1.0. Under the X-then-Y substep rule a move (+1, −a_y) from x ≈ 1.5 slides into (1,2) only if y drops below 3 before x crosses 2.0, i.e. roughly a_y ≤ −0.95: the detour mode is "full speed right *and* full speed down, let the wall sort it out", not a clean downward action, and it sits on a geometric knife-edge. That is the whole oscillation: the y-component drifts across −0.95 during training, and the greedy evaluation flips between ≈ 1.0 and ≈ 0.35.

## Reading

1. The from-scratch result confirms the mechanism end to end with nothing reused from the earlier checkpoint: the pessimistic dataset makes vanilla offline CRL find the detour (all three P seeds reach 1.0 during training; O never does), and every P final beats its O twin under the sampled protocol.
2. The deterministic outcome is unstable: one of three P finals is in the detour mode (0.86), two ended in the shortcut mode. This is the R5 finding again — "the d_lb signal is correct and the critic learns it stably; what remains open is the policy-extraction stability of a continuous Gaussian actor".
3. The instability has a concrete geometric cause here — the saturated x-component and the wall-slide knife-edge — which suggests the faithful lever to test next: a lower BC weight from scratch (0.1, 0.05), which in G2 moved the fork action further down; and reporting the fraction of training spent in the detour mode (8/15, 5/15, 3/15 for P; 0/15 for O) alongside the final.
4. AWR (the R7 remedy) and key balancing remain excluded from the method path.

## Addendum: bc = 0.1 and bc = 0.05 from scratch (same recipe, six runs each)

Launched with the same sweep script on the same node (`logs/g4_sweep_bc01_bc005.log`); evaluated exactly as above (`results/pointmaze_absorbing_g4_bc0p1/`, `results/pointmaze_absorbing_g4_bc0p05/`).

Training curves (greedy, every 10k steps):

| run | evals ≥ 0.8 | mean | final | run | evals ≥ 0.8 | mean | final |
|---|---:|---:|---:|---|---:|---:|---:|
| O bc0.1 s0 / s1 / s2 | 0 / 0 / 0 | 0.33 / 0.37 / 0.41 | 0.26 / 0.36 / 0.40 | O bc0.05 s0 / s1 / s2 | 0 / 0 / 0 | 0.31 / 0.32 / 0.36 | 0.30 / 0.30 / 0.20 |
| **P bc0.1** s0 / s1 / s2 | **14** / 7 / 6 | 0.98 / 0.68 / 0.59 | 0.74 / 0.90 / 0.34 | **P bc0.05** s0 / s1 / s2 | **13 / 9 / 10** | 0.91 / 0.76 / 0.76 | **0.80 / 0.84 / 1.00** |

(For reference bc 0.2: P 8 / 5 / 3 evals ≥ 0.8, finals 0.88 / 0.28 / 0.24.) Time in the detour mode grows as the BC weight falls; at bc = 0.05 all three P seeds end in it. No O run at any bc ever exceeds 0.58.

Native evaluation of the finals, 200 paired seeds, mode protocol:

| final | reach | absorbed | lower route | P − O reach [95% CI] | P − O lower route |
|---|---:|---:|---:|---|---|
| O bc0.1 s0/s1/s2 | 0.375 | 0.625 | 0.000 | | |
| P bc0.1 s0 | 0.825 | 0.175 | 0.720 | +0.450 [+0.380, +0.520] | +0.720 |
| P bc0.1 s1 | 0.860 | 0.140 | 0.755 | +0.485 [+0.415, +0.555] | +0.755 |
| P bc0.1 s2 | 0.375 | 0.625 | 0.000 | 0 | 0 |
| O bc0.05 s0/s1/s2 | 0.375 | 0.625 | 0.000 | | |
| **P bc0.05 s0** | **0.795** | 0.120 | 0.835 | +0.420 [+0.325, +0.505] | +0.835 |
| **P bc0.05 s1** | **0.860** | 0.140 | 0.805 | +0.485 [+0.395, +0.565] | +0.805 |
| **P bc0.05 s2** | **1.000** | 0.000 | 1.000 | +0.625 [+0.555, +0.695] | +1.000 |

Sampled protocol (secondary): P bc0.1 finals reach 0.615 / 0.690 / 0.470 vs O 0.345 / 0.370 / 0.335 (lower route 0.44 / 0.52 / 0.29 vs 0.05 / 0.04 / 0.04); P bc0.05 finals reach **0.825 / 0.780 / 0.715** vs O 0.375 / 0.370 / 0.375 (lower route 0.80 / 0.76 / 0.61 vs 0.02 / 0.05 / 0.01); every P − O interval excludes zero.

Mode actions of the P finals at t = 1 (x ≈ 1.5) and the landing cell as x₁ varies over 1.47–1.52:

| P final | a at t=1 | landing for x₁ = 1.47 … 1.52 |
|---|---|---|
| bc0.2 s0 / s1 / s2 | (+1.00, −1.00) / (+1.00, −0.74) / (+1.00, −0.52) | (1,2)×4 (2,3)×2 / (2,3)×6 / (2,3)×6 |
| bc0.1 s0 / s1 / s2 | (+1.00, −0.95) / (+1.00, −1.00) / (+1.00, −0.80) | (1,2)×4 (2,3)×2 / (1,2)×4 (2,3)×2 / (2,3)×6 |
| bc0.05 s0 / s1 / s2 | (+1.00, −1.00) / (+1.00, −0.99) / (+1.00, −0.95) | (1,2)×4 (2,3)×2 / (1,2)×4 (2,3)×2 / (1,2)×6 |

The x-component stays saturated at +1 at every BC weight; what a lower BC weight buys is a y-component that sits reliably at −1 rather than drifting between −0.5 and −1, so the detour mode becomes the stable attractor (all three bc = 0.05 finals). The fork decision is still the wall-slide (+1, −1) with the actuator noise deciding about 20% of episodes, which is why the mode-protocol reach at bc = 0.05 is 0.80–0.86 rather than 1.0 on two seeds and 1.0 on the seed whose t = 1 position sits lower.

Reading: with bc = 0.05 the from-scratch method is stable across seeds under the repo's own evaluation — reach 0.80 / 0.86 / 1.00 against 0.375 for every baseline run — and the remaining shortfall is the geometric knife-edge of a saturated x-component, not the route preference. A cleaner downward action would need the actor to unpin a_x from the data's (1, 0), which the BC term prevents at every tested weight.

No commit until reviewed.
