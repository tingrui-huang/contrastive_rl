# Stage 3A litter pilot audit

**Status: `PILOT_PASS_READY_FOR_FULL_COLLECTION`**

- episodes: 1426  states: 979056  transitions: 977630
- npz sha256: `715aaeb832283f2b52bd2569e96002c238d5082f2f660eeed104d933f1157703`
- sidecar sha256: `f913a5ba40d334f87a543b0598584faba518ab3271c669d49448813d60bb97bc`
- mixture: {'sighted': 1212, 'blind': 71, 'coverage': 143} (sighted clean-fast / blind eps=0.05 @v=0.6 / coverage @v=0.8)

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
- U balance: u1=716 u0=710 (frac_u1=0.502)
- success by U: {'u1': 0.861731843575419, 'u0': 0.8718309859154929}; collapse by U: {'u1': 0.03770949720670391, 'u0': 0.016901408450704224}
- A6 command compliance 1.000 (deterministic); trajectory-follow 0.976; lane separation 1.971; blind invariance 0.0e+00
- A7 U->S' near/far median L2: 2.825 / 0.000; near collapse frac 0.70
- A8 environmental hiddenness (within single U-independent behaviour): coverage-only acc 0.519 p=0.100, blind-only acc 0.541 p=0.380 (+ A3 exact identity) -> U not in observation; mixed U-indep probe is confounded (speed->subset->prior), not used
- A8 confounding strength (full dataset): acc 0.928 (AUC 0.989), p=0.000 -- EXPECTED U->A->S confounding, not leakage (see A3)
- A9 coverage fraction 0.511 (raw near-complete gate >=0.5: True), nn state dist med 2.372, center-slow fraction 0.294, sustained center-slow eps 205; meaningful support: True

## A9 coverage -- resolved by approved data mixture
- The user-approved 85/5/10 mixture (coverage component 10% of episodes, frozen middle_slow @v=0.8) provides SUBSTANTIAL direct robust-behaviour support: 205 episodes with sustained center-slow behaviour, 29.4% of zone transitions center-slow, coverage fraction 0.511.
- vs the earlier epsilon-only pilot (coverage 0.255, center-slow 0.129, 11 sustained eps): all three roughly doubled.
- Coverage fraction is below the strict near-complete 0.5 mark but the acceptance criterion is met via the explicit approved data-mixture resolution: the pilot now contains direct, substantial state-action support for the U-invariant robust (middle-slow) strategy.

## A8 interpretation
- Environmental hiddenness holds: A3 proves the observation is byte-identical under U-flip, and WITHIN each single U-independent behaviour the pre-contact probe is not significant (coverage-only acc=0.519 p=0.100; blind-only acc=0.541 p=0.380). The mixed U-independent probe is CONFOUNDED (speed->subset->U-prior) and must not be used. The HIGH full-dataset predictability (acc=0.928) is the intended U->A->S confounding (sighted teacher steers by U), not leakage.
