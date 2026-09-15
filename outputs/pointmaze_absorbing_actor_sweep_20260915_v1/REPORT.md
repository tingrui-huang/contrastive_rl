# G2: BC-coefficient sweep on the frozen G1 critics — no coefficient flips the P actor; recommend AWR (G2b)

Date: 2026-09-15. Base G1 `outputs/pointmaze_absorbing_integration_20260915_v1` (critics, plan, batches, keys, roots hash-pinned in [config.json](config.json)). Sealed [PROTOCOL.md](PROTOCOL.md); driver [run.py](run.py); numbers [results.json](results.json), per-actor audits `actor_audit_{arm}_{coef}.json`, curves `learning_curves.npz`, per-root arrays `per_root.npz`, addendum [addendum_fork_landing_bins.json](addendum_fork_landing_bins.json). Strictly offline: 10 actor runs of 1,000 updates, zero critic updates, zero environment calls; 19,443,200 model-rollout transitions under the 30,000,000 cap; 173 s.

## Decision

**Primary criterion not met at any coefficient.** The P actor's canonical-goal down probability on the 656 held-out first-fork roots rises monotonically as BC is reduced — 0.113 (0.5) → 0.201 (0.2) → 0.285 (0.1) → 0.411 (0.05) — but never reaches 0.50, and down/right peaks at 0.74 (threshold 1.5). Retention holds only at 0.5 and 0.2 (away landing-bin agreement 0.834 → 0.799, then 0.636 at 0.1 and 0.527 at 0.05). `bc = 0` is degenerate: both actors saturate into corner actions and the P actor's model rollouts reach the goal 0.4% of the time.

Per the sealed rule: **no coefficient is selected; stop and run AWR extraction as G2b.** The sweep isolates two separate obstacles: the BC/critic conflict is real and lowering BC does move the fork action toward the critic's preference, but the tanh-Gaussian reparameterized actor cannot realize the critic's preferred action even when BC is nearly removed — it saturates instead of turning down — and the physical lower-route entry in rollouts never exceeds 0.24.

Reproduction check: at `bc = 0.5` both actor checkpoints are byte-identical to G1's (`policy_sha256_equal_to_G1` true, L2 delta 0.0).

## Fork-root action distribution (656 roots × 128 sealed innovations, canonical goal)

| bc | arm | down | right | other | saturated | mean action (x, y) | P − O down [95% CI] | P − O right |
|---:|---|---:|---:|---:|---:|---|---|---|
| 0.5 | O | 0.072 | 0.851 | 0.077 | 0.707 | (0.836, −0.310) | | |
| 0.5 | P | 0.113 | 0.757 | 0.130 | 0.625 | (0.732, −0.268) | +0.041 [0.039, 0.043] | −0.094 |
| 0.2 | O | 0.036 | 0.928 | 0.036 | 0.802 | | | |
| 0.2 | P | 0.201 | 0.693 | 0.106 | 0.624 | (0.687, −0.554) | +0.165 [0.160, 0.169] | −0.234 |
| 0.1 | O | 0.016 | 0.968 | 0.017 | 0.848 | | | |
| 0.1 | P | 0.285 | 0.644 | 0.071 | 0.673 | (0.689, −0.728) | +0.269 [0.262, 0.276] | −0.324 |
| 0.05 | O | 0.006 | 0.983 | 0.012 | 0.896 | | | |
| 0.05 | P | **0.411** | 0.556 | 0.034 | 0.731 | (0.717, −0.863) | +0.405 [0.396, 0.413] | −0.427 |
| 0.0 | O | 0.000 | 1.000 | 0.000 | 0.997 | (1.0, ·) | | |
| 0.0 | P | 0.001 | 0.999 | 0.000 | 0.995 | (1.000, −0.978) | +0.001 | −0.001 |

The O actor moves the opposite way (toward right, 0.851 → 1.000) as BC is reduced, consistent with its own critic (endpoint −0.305); no spurious O down shift at any coefficient. Saturation (any component beyond 0.99) is already 0.6–0.7 at 0.5 and 0.995–0.997 at 0.

**What "down" means physically.** The sealed direction mask (`a_y ≤ −0.5, |a_y| ≥ |a_x|`) counts diagonals such as (0.72, −0.86); from the fork cell those aim into the wall below the holding cell and end up going right. Classifying the same samples by the visible landing cell of `s + a` (addendum):

| bc | P samples landing in (1,2) = lower entry | in (2,3) = holding/right | Q_P at samples | Q_P at (1,2)-samples | Q_P at (2,3)-samples |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.042 | 0.267 | −6.70 | −6.36 | −6.77 |
| 0.2 | 0.067 | 0.190 | −6.54 | −6.20 | −6.79 |
| 0.1 | 0.081 | 0.125 | −6.46 | −6.15 | −6.83 |
| 0.05 | 0.097 | 0.069 | −6.40 | −6.10 | −6.93 |
| 0.0 | 0.000 | 0.012 | −6.46 | — | −7.52 |

(The remaining mass lands in wall cells, i.e. blocked diagonals.) True lower-route entry at the first step grows only from 4% to 10%; the mask's 41% is mostly wall-directed diagonals whose x stays near 1 (P x-quantiles at 0.05: 10% −0.31, 25% 0.81, median 0.99). The P critic itself prefers the true-down samples by 0.5–0.8 over the right samples at every coefficient, and its canonical probes are Q(down) = −5.87 vs Q(right) = −6.94; the actor's samples never get above −6.40.

## Model rollouts from the fork roots (canonical goal, common keys)

| bc | arm | A3 reach | A3 absorbed | A3 lower-route | A3 entered support | A1 reach (no freeze) | A1 lower-route |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.5 | O | 0.249 | 0.756 | 0.097 | | 0.987 | |
| 0.5 | P | 0.232 | 0.673 | 0.224 | | 0.880 | |
| 0.2 | O | 0.227 | 0.779 | 0.019 | | 0.997 | |
| 0.2 | P | 0.245 | 0.653 | 0.221 | | 0.881 | |
| 0.1 | O | 0.251 | 0.751 | 0.005 | | 0.998 | |
| 0.1 | P | 0.163 | 0.765 | 0.196 | | 0.897 | |
| 0.05 | O | 0.267 | 0.736 | 0.002 | | 0.998 | |
| 0.05 | P | 0.160 | 0.766 | 0.235 | | 0.817 | |
| 0.0 | O | 0.315 | 0.682 | 0.000 | | 0.996 | |
| 0.0 | P | 0.004 | 0.063 | 0.004 | | 0.004 | |

(A3 = `manski_absorbing`, 32 paths per root; A1 = `diagonal_motion`, 16 paths. Full per-metric intervals, including `entered_support` and `manski_onset`, are in `results.json`.) Under the pessimistic model the P actor's lower-route entry stays at 0.20–0.24 for every coefficient and its reach falls below 0.5's value at 0.1 and 0.05; paired P − O A3 reach is −0.017 (0.5), +0.018 (0.2), −0.087 (0.1), −0.107 (0.05). Under the optimistic model the P actor's reach declines from 0.88 to 0.82 as BC is reduced — the extracted policy navigates worse, it does not detour more.

## Non-fork retention (1,024 sealed away rows, P arm)

| bc | BC-NLL of recorded action | change vs initial [95% CI] | mode L2 change vs initial | landing-bin agreement (initial actor 0.832) | mode OOD fraction | A3 reach from away states | A3 absorbed |
|---:|---:|---|---|---:|---:|---:|---:|
| 0.5 | −1.150 | +0.03 [−0.01, +0.08] | 0.080 | 0.834 | 0.000 | 0.680 | 0.256 |
| 0.2 | −0.655 | +0.52 [+0.46, +0.59] | 0.242 | 0.799 | 0.000 | 0.684 | 0.251 |
| 0.1 | +0.348 | +1.53 [+1.38, +1.66] | 0.486 | 0.636 | 0.000 | 0.667 | 0.280 |
| 0.05 | +1.422 | +2.60 [+2.42, +2.77] | 0.656 | 0.527 | 0.000 | 0.654 | 0.286 |
| 0.0 | 2.1 × 10⁵ | | 0.724 | 0.428 | 0.000 | 0.657 | 0.250 |

Retention passes the sealed rule (agreement ≥ 0.85 × 0.834 = 0.709 and away reach ≥ 0.630) at 0.5 and 0.2 only. The OOD proxy (actor mode landing in a (cell → landing cell) pair never seen in training) stays 0 for every coefficient, so the degradation is drift within the observed support, not leaving it; the away-state rollout reach under A3 is nearly flat (0.65–0.68) because most away states are past the fork. The O arm degrades similarly (agreement 0.837 → 0.713).

## Gradient components on the 4 sealed batches (final actors)

| bc | arm | BC raw ‖g‖ | critic raw ‖g‖ | cosine | coef-weighted BC ‖g‖ | coef-weighted critic ‖g‖ | tail sample entropy | tail saturation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | O | 9.6 | 10.8 | −0.53 | 4.8 | 5.4 | −1.80 | |
| 0.5 | P | 13.4 | 17.2 | −0.86 | 6.7 | 8.6 | −1.47 | |
| 0.2 | P | 35.5 | 11.6 | −0.89 | 7.1 | 9.3 | −2.08 | |
| 0.1 | P | 57.6 | 8.0 | −0.89 | 5.8 | 7.2 | −2.60 | |
| 0.05 | P | 81.6 | 6.2 | −0.82 | 4.1 | 5.9 | −3.18 | |
| 0.0 | P | 7.2 × 10⁶ | 4.4 | −0.01 | 0 | 4.4 | +4.94 | |

(Tail values are means over the last 100 updates; the full nine-column curves are saved.) The BC and critic gradients stay nearly opposite for P at every positive coefficient (cosine −0.82 to −0.89), and the BC gradient norm grows as the actor drifts from the data. At 0 the policy scale collapses (away-row mean scale 0.037), the entropy diagnostic goes positive and the BC-NLL diagnostic explodes: the pure-critic actor is a deterministic saturated corner policy whose critic value (−6.46) is *lower* than at 0.05 (−6.40) — the reparameterized tanh gradient vanishes at saturation before reaching the critic's optimum.

## Interpretation

1. **The BC coefficient is not the whole actor bottleneck.** Removing BC does not produce the critic's preferred action: at 0.05 the actor's samples score −6.40 against the critic's down probe −5.87, only 10% of first steps physically enter the lower route, and at 0 the actor saturates away from the optimum. This is the continuous-actor extraction failure the archived Manski port recorded as finding 7 (Q-max + BC tug-of-war; critic-greedy spins), here with the additional symptom that "down" by the direction mask is mostly a wall-directed diagonal.
2. **The conflict is real and directional.** P−O down probability grows to +0.41 and the O actor never shows a spurious down shift, so the critic signal from G1 is what the actor is responding to.
3. **Retention collapses below 0.2**, so any coefficient that moves the fork action materially also costs non-fork behaviour.

## Recommendation: G2b, AWR extraction

Keep both G1 critics frozen and extract the actor by advantage-weighted regression on **recorded** actions only (`w = exp(β (Q(s, a_rec) − V(s)))` with V from the actor's own samples or a mean over recorded actions, β to be preregistered, the same 1,000 batches and keys, applied identically to O and P). AWR cannot leave the recorded action support, so saturation and OOD diagonals are excluded by construction, and the recorded lower-route actions exist in the data (47.9% of the recorded down-anchor actions in the training pool land in (1,2)). Read the same fork-root landing-bin distribution and the same A3/A1 rollouts; the pass line remains lower-route entry and reach, not the direction mask. Nothing here was committed or pushed.
