# Formal H=800 vs existing H=700 — p_active=0.30 local-detour benchmark

Formal experiment (NOT eval-only): fresh H=800 dataset (300 ep, 270/0/30, same
collector seeds as the H=700 p30 pilot so only the horizon differs; sha 45c4db1d,
V1-V10 + G1-G8 ALL PASS) -> fresh naive CRL 300k @ H=800 (seed 0, same config as
the H=700 p30 run, only horizon 700->800) -> authoritative reconcile + diagnosis
@ H=800. Provenance: run naive_rockfall_v2_p30_h800_s0_300k, dataset sha 45c4db1d,
git 197fac9, GPU. Both columns are reconcile at each model's NATIVE cap.

## Direct comparison

| Metric | H=700 existing | H=800 newly trained |
|---|---:|---:|
| Teacher natural success | 0.867 | 0.867 |
| Center natural success | 0.796 | **0.898** |
| Blind natural success | 0.530 | 0.530 |
| Naive natural success | 0.706 | 0.669 |
| Naive balanced macro | 0.708 | 0.675 |
| Naive worst-mask | 0.63 | 0.36 |
| Naive median success steps | 582 | 365 |
| Naive center usage | 0.375 | 0.235 |
| Naive hazard exposure | 0.295 | 0.615 |
| Naive drop rate | 0.105 | 0.25 |
| teacher - center (value of mask) | **+0.071** | **-0.031** |
| teacher - naive (confounder gap) | +0.161 | +0.198 |

Naive natural is the analytic-weighted metric (0.706 / 0.669); pooled 0.755 / 0.665;
best.pkl for reference: analytic 0.672, macro 0.678, worst 0.39. Scripted anchors
use their own control law at H=800.

## The decisive, reliable finding: the ordering inverts at H=800

- **Center is horizon-rescuable** (0.796 -> 0.898): it is slow (median ~655 steps,
  near the 700 boundary) but hazard-free (0 rockfall), so the extra 100 steps let
  ~10 pts of timeout episodes finish -- exactly as the center-failure diagnosis
  predicted (mud is a SLOWDOWN cost).
- **Teacher is horizon-flat** (0.867 = 0.867): it is fast (median 212) and its
  ceiling is rockfall/fall-limited, so more time rescues nothing.
- Therefore at H=800 the **mask-BLIND center controller (0.898) OVERTAKES the
  sighted teacher (0.867)**: teacher - center goes +0.071 -> -0.031. This VIOLATES
  the criterion "sighted teacher remains better than center." Blind is also flat
  (0.530; rockfall-killed, not timeout-limited).

This is the robust result -- it is a controller-level property (deterministic
horizon rescue of a fixed scripted policy vs a fixed fast teacher), not training
noise.

## Guardrail checks (as requested)

- **Does H=800 merely look better because center rises? Yes -- and that is the
  problem, not a win.** The center rise is what inverts the ordering.
- **Does naive learn the slow center route MORE strongly? NO -- the opposite.**
  Center usage 0.375 -> 0.235; the fresh H=800 naive took the fast side lanes more
  (exposure 0.295 -> 0.615, drops 0.105 -> 0.25, median steps 582 -> 365: faster
  when it works, but riskier). The "universal center-routing" concern does not
  intensify at H=800.
- **Does the natural teacher-naive gap survive? Yes** (+0.161 -> +0.198). Caveat:
  the naive delta (0.706 -> 0.669) is a SINGLE seed and within training variance
  (both 30-ep monitoring curves peaked ~0.77); do not over-read it. The durable
  finding is the center/teacher inversion.
- **Leakage / trigger-gaming: clean at H=800** (paired mask-flip 30/30; trigger
  avoidance 0.09 final / 0.075 best, well under the 0.30 line).

## Recommendation: do NOT adopt H=800

H=800 fails the "teacher > center" requirement: the extra horizon lets a
mask-independent safe-slow policy beat the privileged teacher on natural success,
undermining the confounder premise. **Keep H=700**, which preserves the intended
ordering teacher 0.867 > center 0.796 > naive 0.706 > blind 0.530, with the mask
still worth +0.071 over always-center. This formal, fully-trained experiment
confirms the earlier eval-only horizon-sweep prediction.

If the center's H=700 sub-0.80 (a bounded mud-timeout artifact, ~13% rescuable)
is felt to understate the robust baseline, the fix is NOT a longer horizon (it
inverts the ordering) but a steps-to-success / efficiency co-metric, on which the
teacher dominates at every horizon (median 212 vs center ~655).

All H=700 and p_active-sweep artifacts are unchanged; H=800 lives in its own
namespace (artifacts/rockfall_v2_p30_h800/, run naive_rockfall_v2_p30_h800_s0_300k,
artifacts/rockfall_reconcile/p30_h800_{final,best}.json).
