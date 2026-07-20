# Stage 3A litter pilot audit

**Status: `COVERAGE_DECISION_REQUIRED`**

- episodes: 200  states: 138813  transitions: 138613
- npz sha256: `5d45416c9f2497ef01ebd9764e1e85e4d3c9f7f211bd707ed8e8dd1e36adc987`
- sidecar sha256: `da5507e98adf44c385565429c851dfe6033a60e13a6148332b5b29260980c7a9`
- teacher blind speed used: 0.6 (frozen code; manifest slow_v documents the gate arm)

## Audit gates
- **A1 frozen integrity**: PASS
- **A2 shape/count**: PASS
- **A3 leakage**: PASS
- **A4 boundary/relabel**: PASS
- **A5 RNG & U balance**: PASS
- **A6 teacher compliance / U->A**: PASS
- **A7 local U->S'**: PASS
- **A8 pre-contact hiddenness**: PASS
- **A9 robust-policy coverage**: FAIL

## Key numbers
- U balance: u1=103 u0=97 (frac_u1=0.515)
- success by U: {'u1': 0.8737864077669902, 'u0': 0.9175257731958762}; collapse by U: {'u1': 0.02912621359223301, 'u0': 0.0}
- A6 sighted compliance 1.000; lane separation 2.095; blind invariance 0.0e+00
- A7 U->S' near/far median L2: 2.825 / 0.000; near collapse frac 0.70
- A8 environmental hiddenness (blind-only): acc 0.518, p=0.460 (+ A3 exact identity) -> U not in observation
- A8 confounding strength (full dataset): acc 0.963 (AUC 0.997), p=0.000 -- EXPECTED U->A->S confounding, not leakage (see A3)
- A9 coverage fraction 0.255, nn state dist med 3.097, center-slow fraction 0.129, sustained center-slow eps 11

## A1 documentation discrepancies (non-blocking)
- `commands.slow_v`: manifest slow_v documents the GATE middle_slow arm (--slow-v 0.8); the frozen QUALIFIED teacher blind policy uses WG.SLOW_V=probe.V_SLOW=0.6. Data generation uses the code value. Manifest should split gate_middle_slow_v from teacher_blind_v before full collection.

## A8 interpretation
- Environmental hiddenness holds: the observation carries no U label (A3 exact identity; blind-only probe not significant). The HIGH full-dataset predictability (acc=0.963) is the confounding pathway U->A->S (sighted teacher steers by U pre-contact), which is the intended mechanism, not leakage.

## COVERAGE_DECISION_REQUIRED -- proposed data mixture (NOT applied)
- At epsilon=0.05 the pilot has 11 episodes with sustained center-slow behaviour; reference-bank coverage fraction is 0.255 (< 0.5 gate) and center-slow zone transitions are 0.129 of zone transitions -- thin support for the U-invariant robust policy.
- Proposal (requires explicit approval before full collection): keep the sighted/blind confounding structure at epsilon=0.05, and ADD a separate frozen middle_slow COVERAGE component of ~10-15% of episodes (both U, using the exact frozen middle_slow controller incl. the unstick heuristic). This lifts safe-behaviour support without changing the epsilon confounding ratio or any frozen constant.
- Alternative: raise epsilon, but that changes the confounding semantics (blind rate) and would require re-running teacher qualification -- NOT recommended.
- Do NOT implement either without approval.
