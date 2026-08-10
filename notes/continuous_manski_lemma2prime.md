# [ARCHIVED THEORY] Lemma 2′ — Continuous per-step Manski bound

Status: archived 2026-07-15, not blocking experiments. Dig out when writing the
paper's continuous-extension section or when Junzhe asks about the Lipschitz
assumptions.

## Where Lipschitz enters (exactly two places)

**(A1) transition Lipschitz in ACTION** — makes the action-neighborhood
propensity legal. Continuous actions have P(X=x|s)=0, so the mixing weight
must become P(X ∈ B_h(x) | s). Samples in the bin used x′ ≈ x, not x; they
can speak for do(x) only if

    | E[V(S′)|s,do(x)] − E[V(S′)|s,do(x′)] |  ≤  L_a · ‖x−x′‖.

**(A2) value Lipschitz in STATE** — makes the finite-grid argmin legal.
N(s,x) is uncountable in continuous state; taking min over cell centers
(grid pitch δ) instead of inf over the set costs at most L_s·δ:

    inf_{continuous N} V̲  ≥  min_{grid N} V̲ − L_s·δ.

Same assumption also licenses re-anchoring ("any dataset state inside the
cell counts as that state") in the Thm-2 sampler.

## Lemma 2′ (statement)

Under (A1)(A2), for any bin radius h and grid pitch δ:

    E[V(S′)|s,do(x)] ≥   P(B_h(x)|s) · ( E_obs[V | s, X∈B_h(x)] − L_a·h )
                       + (1−P(B_h(x)|s)) · ( min_{grid N(s,x)} V̲ − L_s·δ )

Properties:
1. Valid (a true lower bound) for EVERY h, δ — slack is subtracted, so
   Thm 1's validity proof structure is unchanged; Lemma 1 contraction
   unaffected (slacks are constants).
2. h is an explicit tightness knob: h→0 ⇒ slack→0 but P(B_h)→0 (bound
   degenerates to global min); the DISCRETE case is the special case where
   h equals one action atom.
3. In the 2D swamp env both constants are computable, not assumed:
   dynamics Δs = dt·a·factor give ‖∂s′/∂x‖ ≤ 1 so L_a ≤ L_s; the BFS
   proxy V̲ has an explicit Lipschitz constant. The SAME constants give the
   fine-grid VI oracle its discretization-error bound (≈ L·δ/(1−γ)) —
   one assumption pair, two uses.

## Proof sketch

Follow the original Lemma 2 proof; insert two triangle inequalities:
one over the action bin (A1) when replacing do(x) with observed x′∈B_h(x),
one over the cell (A2) when replacing the continuous inf with the grid min.

## When the critic replaces the BFS proxy

(A2) becomes a requirement on the learned V̲ — enforceable by architecture
(spectral norm / weight clipping), turning the assumption into a design
parameter. Complements target-network stabilization for the critic–sampler
coupling.

## One-sentence version (for talks)

"Continuity costs two Lipschitz assumptions — transition-in-action and
value-in-state — each entering the per-step Manski bound as an explicit
slack term; the bound stays valid at every discretization granularity, the
discrete case is a special case, and on our 2D benchmark both constants are
computable and double as the fine-grid-VI oracle's error guarantee."
