# Reproduce the deterministic gate

From the `feature/pointmaze-causal-transition` checkout based on
`c5810d2f08a966c202189347a510fac8a6d0cd31`, use the existing local Python runtime
and frozen artifacts recorded in `provenance.json`:

```powershell
python -m scripts.audit_ett_death_admissibility --out-dir artifacts/ett_death_admissibility/frozen_f4_gate_reproduction
```

Use a new output directory to preserve this run. The script checks the original
153 input hashes and the predeclared SPEC before computing and checks input
hashes again afterwards. Missing or changed frozen inputs fail the analysis;
there is no reconstruction, retraining, simulator call, random sampler call,
policy evaluation, or optimization fallback. JAX is used for deterministic
network and emitter evaluations only. `provenance.json` belongs to the original
gate and is read from `frozen_f4_gate_v1` even with a different output directory.

The script uses the current frozen diagonal distribution's actual normalization,
stationary logits, Gaussian component means/scales, and geometry guard. The
{-1,0,1} component noise panel is fixed, not sampled or probability weighted.
The five coefficient arrays are zero plus the four final geometry-comparison
checkpoints; each is tested in both modes as a component comparison. No array
is modified. Exact F4 shifts and diagonal outputs are asserted, as are selected
set validity and the existing matrix norm bounds. The hazard-entry example uses
one analytically chosen rank-one matrix, fixed before its nine-action check.

`conditions.json` contains only visible state/history, execution action, nominal
action, and commanded goal. `native_validation_only.json` contains the separately
saved native failure flags' interpretation and source indices. Other outputs
retain deterministic support anchors, raw displacement, emitted F4 states,
reference rectangles, coefficients' response summaries, and verification counts.
`DERIVATIONS.md` contains the proof; neither the finite component panel nor the
example's finite action grid replaces that proof.

Publication packaging adds local Git attributes for this artifact directory and
the analysis script to preserve hashed bytes across Windows checkouts. Existing
root attributes and all recorded frozen input hashes remain unchanged.
