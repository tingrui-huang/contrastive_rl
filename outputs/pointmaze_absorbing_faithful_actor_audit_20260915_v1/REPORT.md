# Faithful-actor audit: the one-step/multi-step gap is a diagonal-motion mismatch in the model, not a continuation failure; uniform-anchor P critic stays positive

Date: 2026-09-15. Inputs: G1 critics/plan/roots, G2 actor checkpoints (bc 0.5 reference, 0.2, 0.1, 0.05; O and P). Sealed [PROTOCOL.md](PROTOCOL.md); driver [run.py](run.py); [results.json](results.json); per-actor audits `audit_{arm}_{bc}.json`; per-root arrays `per_root.npz`; Part B cache and critics in the directory. Part A: no training, 12,343,296 model-rollout transitions. Part B: 104,808 model outputs for the uniform-pool cache, 800 critic updates, zero actor updates. Zero environment calls. 115 s.

## Decision context

G2c and G2c′ are retracted from the method path (negative ablation: global key balancing changes the fork prior and imports random-policy rows). The faithful candidate is the paper objective on the original actor rows with **bc = 0.2**: physical first-step entry into (1,2) **0.323** (O 0.131), away physical agreement 0.853, A1 reach 0.881. This audit explains why its multi-step A3 lower-route entry stayed at 0.22.

## Answer

**(b) Diagonal-motion mismatch.** The actors' down moves at the fork are mostly down-right diagonals such as (0.80, −0.89). Physically (ten substeps, X then Y, blocked coordinate updates rejected) the X update is refused once the point is below y = 3 and the move slides into (1,2), displacement ≈ 1.05. The eligible diagonal model's projection (`_project_samples`) instead checks the naive endpoint cell, finds the wall cell (2,2), and **reverts the whole move to the current position**: model displacement 0.004–0.007, landing "(1,3) stay" in 99.5–99.6% of these samples. So the model's first transition enters (1,2) for only 0.065 of the bc = 0.2 samples against 0.323 physically, and the disagreement grows with the actor's diagonal share (bc 0.05: 0.093 vs 0.520).

Not (a) classification: physical and model cells are computed on the same samples; the model simply does not move them. Not (c) continuation: once the model does enter (1,2), 82–89% of paths continue to y < 2 before revisiting the fork or the holding cell, and the actor's own action at the first (1,2) state physically heads down 47–54% (recorded actions at (1,2) in the training data: 47.5%). The multi-step A3 lower-route number is therefore a downward-biased proxy for any actor that emits diagonals; the physical first-step entry times the observed continuation rate (0.885) is the better offline estimate — 0.29 at bc 0.2, 0.36 at 0.1, 0.46 at 0.05 — and G3 native episodes are the real measurement.

## Part A tables (656 roots × 128 sealed innovations; P arm unless stated)

First-step classification of the same samples:

| bc | naive (1,2) | naive wall | physical (1,2) | physical (2,3) | model A1/A3 (1,2) | model "(1,3) stay" | model (2,3) | Kernel θ=0 (2,3) | P(model (1,2) \| physical (1,2)) | per-root disagreement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.042 | 0.610 | 0.202 | 0.654 | 0.041 | 0.646 | 0.273 | 0.880 | 0.37 | 0.58 |
| 0.2 | 0.067 | 0.672 | 0.323 | 0.543 | 0.065 | 0.704 | 0.196 | 0.880 | 0.37 | 0.66 |
| 0.1 | 0.081 | 0.746 | 0.420 | 0.470 | 0.078 | 0.767 | 0.132 | 0.880 | 0.35 | 0.74 |
| 0.05 | 0.097 | 0.810 | 0.520 | 0.407 | 0.093 | 0.818 | 0.079 | 0.880 | 0.35 | 0.80 |
| 0.2 (O) | 0.016 | 0.513 | 0.131 | 0.817 | 0.016 | 0.540 | 0.432 | 0.880 | 0.31 | 0.51 |

The model's first-step (1,2) share equals the naive (1,2) share to within 0.003 at every bc: the model enters (1,2) only when the naive endpoint is already (1,2). Among samples physically in (1,2), the model puts 80–82% at "(1,3) stay" and 18–20% in (1,2). Among samples physically in (2,3) the model also stays 58–81% of the time — the same rule hits right-up diagonals into (2,4). The eligible Kernel at zero offsets lands in (2,3) 88% regardless of the action (action-blind, as in G0). The first-step atom fraction is 0.001–0.002 and the Manski onset never fires at the fork (no support cell adjacent), so freezing plays no role in the first step.

Down-right wall diagonals specifically (naive cell is a wall, physical landing (1,2)):

| bc | share of samples | mean action | model displacement | physical displacement | model cell |
|---:|---:|---|---:|---:|---|
| 0.5 | 0.160 | (0.83, −0.86) | 0.007 | 1.03 | (1,3) stay 0.993 |
| 0.2 | 0.255 | (0.80, −0.89) | 0.004 | 1.06 | (1,3) stay 0.996 |
| 0.1 | 0.339 | (0.81, −0.92) | 0.005 | 1.08 | (1,3) stay 0.995 |
| 0.05 | 0.423 | (0.82, −0.94) | 0.005 | 1.10 | (1,3) stay 0.996 |

Continuation (A3, 32 paths per root, common keys):

| bc | A3 first-step (1,2) | A3 ever in (1,2) | A3 lower route | lower \| first (1,2) | lower \| not first (1,2) | absorbed \| first (1,2) | A3 absorbed | A3 reach | A1 first (1,2) | A1 lower | A1 reach |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.041 | 0.345 | 0.224 | 0.82 | 0.20 | 0.19 | 0.673 | 0.232 | 0.044 | 0.233 | 0.880 |
| 0.2 | 0.064 | 0.363 | 0.221 | 0.89 | 0.18 | 0.11 | 0.653 | 0.245 | 0.066 | 0.224 | 0.881 |
| 0.1 | 0.077 | 0.357 | 0.196 | 0.86 | 0.14 | 0.17 | 0.765 | 0.163 | 0.079 | 0.197 | 0.897 |
| 0.05 | 0.092 | 0.361 | 0.235 | 0.88 | 0.17 | 0.18 | 0.766 | 0.160 | 0.091 | 0.229 | 0.817 |
| 0.2 (O) | 0.015 | 0.048 | 0.019 | 0.61 | 0.01 | 0.30 | 0.779 | 0.227 | 0.016 | 0.020 | 0.997 |

Outcome classes of paths whose first model cell is (1,2) (bc 0.2, n = 1,334): continued to y < 2 before revisiting the fork/holding cell 0.877; returned 0.031; returned then absorbed 0.091; stuck 0.001. The actor sampled at the first (1,2) state of those paths lands physically in (1,2)/(1,1) 0.54 (continuing down), (1,3) 0.03, (2,3) 0.11. "Ever in (1,2)" is 0.36 for every P actor because the model's later steps (actor keeps trying diagonals; eventually one has a free naive endpoint) get the paths in, which is why the multi-step number is flat in bc while the first-step physical number rises.

## Part B — uniform-anchor critic purity check

Pool: 4,096 uniformly drawn training rows (170 fork anchors, 19 down, 94 right) instead of the 2,048/1,024/1,024 stratified pool; 400 batches × 256 uniform pool anchors; A3 continuations for P (recorded first action, then the original actor), recorded futures for O; same initial checkpoint, NCE, Adam, budget.

| critic | endpoint down − right [95% CI] | neighborhood | G1 stratified |
|---|---|---:|---:|
| initial | −0.603 [−0.669, −0.542] | −0.504 | −0.603 |
| O uniform anchors | −0.360 [−0.426, −0.296] | −0.250 | −0.305 |
| **P uniform anchors** | **+0.051 [+0.023, +0.073]** | +0.074 | +1.071 |
| paired P − O | **+0.411 [+0.361, +0.459]** | | +1.376 |

The sign flip survives without the anchor stratification (interval excludes zero), but the margin is 20× smaller: with 19 down anchors among 4,096 the critic sees the detour's futures rarely. That is the critic-side analogue of the discrete pathology #5 (rare-route undertraining of action values), and it concerns estimation, not the imitated action prior; the O critic under the same pool still prefers right. G1's stratification is therefore a variance device for the critic, not a change of the benchmark, but the purity check shows the effect size depends on it.

## Consequences

1. The offline A3/A1 rollout metrics undercount lower-route entry for diagonal-emitting actors by a factor of ~5 at bc 0.2. Reports G2, G2b and G2c that used "A3 lower route" as a deployment proxy should be read with the physical first-step entry instead; their O/P contrasts remain valid in direction.
2. The same blocked-endpoint rule affects the P caches wherever a recorded first action is a wall diagonal (random-policy rows, 18% of anchors); the teacher's actions are mostly axis-aligned, so G0/G1 are affected only marginally. A sliding projection (per-axis rejection, as the physics) would be a model-family change and is not made here.
3. For bc = 0.2 the faithful offline estimate of lower-route entry is ≈ 0.29 (0.323 × 0.885) against O ≈ 0.08 (0.131 × 0.61); native G3 episodes measure this directly and do not depend on the model's projection.

## Recommendation

Proceed to G3 with bc = 0.2 as the primary faithful candidate (bc = 0.1 as a secondary if the preregistration allows two P actors): 200 paired native episodes per policy under the original frozen-actor harness of the oracle pilot — P bc 0.2, O bc 0.2, and the G1 bc 0.5 pair as baselines — reading native route entry, success, failure and return. Retain the G1 stratified critic; the uniform-anchor result is recorded as a positive purity check. Nothing here was committed or pushed.
