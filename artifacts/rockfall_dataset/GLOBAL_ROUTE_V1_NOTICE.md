# Rockfall dataset — `global-route v1` (preserved diagnostic)

**Specification note (2026-07-21).** The sighted teacher used to collect the
pilot and full datasets here is a **global-route avoider**: at episode start it
reads the full privileged mask and commits to one whole-corridor route — the
clear side if exactly one side has active sites, `center` if both do, a balanced
side if both are clear (`scripts/rockfall_pilot.py::teacher_route`). It makes **no
local per-site detours**; the control law holds a constant lane `y_ref`.

This is a **deviation** from the intended primary specification, which is
**local privileged avoidance**: a mask-independent balanced base side lane, with a
**local inward detour before each active site** and a return to the base lane
(straight at inactive sites). Verified from the frozen trajectories: sighted
side-route episodes traverse **0** active sites on their lane, and blind episodes
walking into active sites show identical closest-approach (0.906 active vs 0.922
inactive |y| — pure gait wobble, no detour).

**Status:** these `global-route v1` artifacts are **preserved as a documented
diagnostic** and are NOT overwritten. The intended `local-detour` variant is
built separately (`scripts/rockfall_v2_teacher.py`, `rockfall_v2` dataset dirs).
Do not run the causal-method experiments on this v1 full dataset as though it
matched the local-detour spec.

Preserved here:
- `full/` — 1400-ep global-route dataset (971,516 transitions) + R1–R13 audit.
- `pilot/` — 300-ep global-route pilot + audit.
- `oracle/` — center-only / reweight diagnostic subsets of the v1 full set.
- run `naive_rockfall_full_s0_300k` (diagnosis in
  `artifacts/naive_rockfall_diagnosis/full_best,full_final`).

A severity sweep (`scripts/rockfall_severity_sweep.py`) is evaluation-only
sensitivity analysis; changing severity changes the transition kernel, so any
adopted severity change would require re-collection + retraining, not just eval.
