# Stage 3A litter pilot audit

**Status: `PILOT_PASS_READY_FOR_FULL_COLLECTION`**

- episodes: 200  states: 134965  transitions: 134765
- npz sha256: `deb46545169d8f3116415253dbf45f423bdcbd9e9dfd6eb043ab61694bf26685`
- sidecar sha256: `8cdb965c7475563d64e140f31ac30412f5aaff7a44e28f52470737c93370f1f5`
- mixture: {'sighted': 170, 'blind': 10, 'coverage': 20} (sighted clean-fast / blind eps=0.05 @v=0.6 / coverage @v=0.8)

## Audit gates
- **A1 frozen integrity**: PASS
- **A2 shape/count**: PASS
- **A3 leakage**: PASS
- **A4 boundary/relabel**: PASS
- **A5 RNG & U balance**: PASS
- **A6 teacher compliance / U->A**: PASS
- **A7 local U->S'**: PASS
- **A8 pre-contact hiddenness**: PASS
- **A9 robust-policy coverage**: PASS

## Key numbers
- U balance: u1=101 u0=99 (frac_u1=0.505)
- success by U: {'u1': 0.8415841584158416, 'u0': 0.898989898989899}; collapse by U: {'u1': 0.07920792079207921, 'u0': 0.030303030303030304}
- A6 sighted compliance 0.984; lane separation 1.983; blind invariance 0.0e+00
- A7 U->S' near/far median L2: 2.825 / 0.000; near collapse frac 0.70
- A8 environmental hiddenness (U-independent-only): acc 0.533, p=0.760 (+ A3 exact identity) -> U not in observation
- A8 confounding strength (full dataset): acc 0.915 (AUC 0.984), p=0.000 -- EXPECTED U->A->S confounding, not leakage (see A3)
- A9 coverage fraction 0.396 (raw near-complete gate >=0.5: False), nn state dist med 3.094, center-slow fraction 0.344, sustained center-slow eps 25; meaningful support: True

## A9 coverage -- resolved by approved data mixture
- The user-approved 85/5/10 mixture (coverage component 10% of episodes, frozen middle_slow @v=0.8) provides SUBSTANTIAL direct robust-behaviour support: 25 episodes with sustained center-slow behaviour, 34.4% of zone transitions center-slow, coverage fraction 0.396.
- vs the earlier epsilon-only pilot (coverage 0.255, center-slow 0.129, 11 sustained eps): all three roughly doubled.
- Coverage fraction is below the strict near-complete 0.5 mark but the acceptance criterion is met via the explicit approved data-mixture resolution: the pilot now contains direct, substantial state-action support for the U-invariant robust (middle-slow) strategy.

## A8 interpretation
- Environmental hiddenness holds: the observation carries no U label (A3 exact identity; U-independent-only probe not significant). The HIGH full-dataset predictability (acc=0.915) is the confounding pathway U->A->S (sighted teacher steers by U pre-contact), which is the intended mechanism, not leakage.
