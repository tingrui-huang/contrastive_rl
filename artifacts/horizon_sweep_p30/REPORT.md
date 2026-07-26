# Horizon sweep — p_active=0.30 local-detour (diagnosis only)

H ∈ {700, 800, 900, 1000}. No frozen defaults changed; no dataset/checkpoint/
report overwritten. Horizon = env-instance override + loop bound. Single pass to
H=1000 per episode; all four horizons derived by thresholding the first-success
step (every controller's action depends only on (obs,t), so this is exact).
n=500 natural paired masks+seeds, seed 20260726, p=0.30, severity 0.80/0.15/0.05.

## Part A — scripted controllers (natural-distribution success)

| policy | H700 | H800 | H900 | H1000 | median steps→succ | rockfall hit-eps |
|---|---|---|---|---|---|---|
| sighted teacher | 0.892 | 0.892 | 0.892 | 0.892 | 212 | 29 |
| fixed-center | 0.792 | 0.902 | 0.918 | 0.918 | ~655 | **0** |
| blind-side | 0.550 | 0.552 | 0.552 | 0.552 | 184 | 227 |

Derived orderings:

| | H700 | H800 | H900 | H1000 |
|---|---|---|---|---|
| teacher − center | **+0.100** | −0.010 | −0.026 | −0.026 |
| teacher − blind | +0.342 | +0.340 | +0.340 | +0.340 |
| center rescued after 700 | 0 | 0.110 | 0.126 | 0.126 |

Failure detail (H700 → H1000):
- **teacher**: horizon-FLAT. Fast (median 212, mean 236). Its 0.892 ceiling is set
  by rockfall (20 dead, 29 hit) + 27 falls — **not timeout** (0 rescued by more
  time). trigger-eps 429 but only 29 hits: the local detour works, residual hits
  are the few it can't avoid.
- **center**: horizon-RESCUABLE. 104 timeouts at H700 → 41 at H900 (plateau; H1000
  identical). **ZERO rockfall interaction at every horizon** (trigger/drop/hit all
  0). Median step-to-success ~655 — it barely finishes inside 700. Residual at
  plateau = 35 falls (controller ceiling, horizon-independent) + ~6 still-slow.
- **blind**: horizon-FLAT. Fast (median 184) but 178/500 **dead** from rockfall
  (227 drop/hit eps) — its failures are deaths, not timeouts, so horizon does
  nothing (0.2% rescued).

## Part B — existing naive final.pkl (EVALUATION-ONLY horizon sensitivity)

**Not a trained baseline for H>700.** The H=700-trained checkpoint is Markovian, so
evaluating it at longer horizons only rescues slow-but-progressing episodes; a
formal H>700 baseline requires retraining. Behavioural metrics
(route/exposure/drop, and leakage 30/30 + no trigger-gaming from the committed
v2_p30_final diagnosis) are horizon-invariant (set during the pre-goal traversal).

| metric | H700 | H800 | H900 | H1000 |
|---|---|---|---|---|
| natural success | 0.770 | 0.810 | 0.816 | 0.818 |
| balanced macro | 0.750 | 0.796 | 0.804 | 0.804 |
| worst-mask | 0.667 | 0.683 | 0.700 | 0.700 |
| median steps→succ | 590 | 599 | 600 | 600 |
| rescued after 700 | 0 | 0.040 | 0.046 | 0.048 |
| center/left/right route | 0.352 / 0.618 / 0.03 (horizon-invariant) | | | |
| hazard exposure / drop | 0.316 / 0.106 (horizon-invariant) | | | |

The naive also gains ~5 pts from longer eval (its center-routing episodes are slow
through the mud too), plateauing at H900.

## The decisive finding: criteria 1 and 2 are mutually exclusive here

The five criteria cannot be jointly satisfied in [700, 1000]:

- **"center reliably high"** needs H ≥ 800 (0.792 → 0.902 → 0.918).
- **"teacher remains better than center"** holds ONLY at H700 (+0.100). At H800
  they are tied (0.892 vs 0.902, within noise); at H900+ **center exceeds the
  teacher** (0.918 vs 0.892).

Mechanism: the **teacher's ceiling is rockfall-limited and horizon-insensitive**
(it's fast, capped by the hits/falls it can't avoid), while the **center's ceiling
is timeout-limited and horizon-sensitive** (slow but hazard-free). So any horizon
long enough to un-cripple the center lets the mask-BLIND safe-slow center match or
beat the mask-READING teacher on success rate.

The other three criteria hold at all horizons: teacher − blind ≈ +0.34; center is
~3× slower than the fast side route (median ~655 vs ~200); the center rescue
saturates at H900 (H1000 adds nothing), so no excessively-long timeout is needed.

## Recommendation

**Do not increase the horizon for p=0.30.** The +0.100 teacher-over-center gap at
H700 is largely a horizon artifact: it comes from the center being timeout-truncated
by the mud, not from genuine mask-information value. Extending the horizon to make
the center "reliably high" removes that artifact and thereby **inverts the required
ordering** — a mask-independent safe-slow policy matches/beats the privileged
teacher, which undercuts the confounder premise.

- If **success rate is the sole primary metric**: keep **H = 700** — the only
  horizon with the clean teacher > center > blind ordering (0.892 / 0.792 / 0.550).
  The center's sub-0.80 is a bounded, understood mud-timeout artifact (~13% of its
  episodes are horizon-rescuable), not a bug, and it does not affect the naive
  vs teacher comparison, which is what the benchmark reports.
- If the center's H700 timeout-suppression is considered unacceptable, the fix is
  **not** a longer horizon but either (a) reducing the center's mud SPEED cost
  (a protocol change — separate decision), or (b) adding a **steps-to-success /
  efficiency co-metric**, on which the teacher dominates at every horizon
  (median 212 vs ~655) and the ordering is preserved.

Either way, this argues the p=0.30 confounder value on raw success is thin once the
center is not horizon-crippled — consistent with the earlier finding that
teacher − center shrinks with density. If p=0.30 is adopted, the clean confounder
signal to headline is **teacher − blind on the shared fast lane** (+0.34, robust at
all horizons), not teacher − center.
