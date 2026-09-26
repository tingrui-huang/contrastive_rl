# Arm CF-learned (oracle-label-trained learned ETT, cross-fitted): NEGATIVE -- CFL = O = start; none of the oracle gain is kept

> **Identity correction (2026-09-19, after the user's review).**  The model
> trained here is NOT the project's ETT.  The ETT of the PointMaze pipeline
> (`feature/pointmaze-causal-transition`: `ett/diagonal_transition.py`,
> `outputs/pointmaze_learned_ett_crl_20260915_v1/PROTOCOL.md`,
> `scripts/run_f4_ladder_ett.py`) is a ONE-STEP model `F(s, a_b, a_q) -> s'`
> with a separate irreversible-failure (onset) head, fitted on same-context
> tuples `(s, a_b, a_q, s')` (diagonal `a_q = a_b` from the log, off-diagonal
> from same-context interventions), rolled out step by step with a nominal
> model of the expert advice `a_b ~ P(a_b | s)` and the current agent
> choosing `a_q`, with absorbing tails after onset.  What this directory
> trains is a per-anchor classifier of the discounted future-goal cell,
> `p_gamma(g | s, a_logged)`, under the fixed start agent: no `a_b`, no
> diagonal / off-diagonal structure, no next state, no step-wise rollout, and
> the death head never constrains a trajectory.  It tests only whether a
> learned sampling backend can replace the simulator branches in arm CF; its
> failure is not evidence about the one-step ETT.  Wherever this directory or
> the notes say "learned ETT", read "future-goal marginal model".


Status: an engineering intermediate experiment on the INTERFACE (the ETT
model was supervised on the pilot's simulator branches); not the offline
identification.  `manifest.json` sealed before fitting; `CHECK.md` /
`check.json` (model checks), `REPORT.md` / `results.json` (the arm).

## The model (`CHECK.md`)

* Held-out marginal NLL (exact expectation over the oracle path, the same
  anchors, the same output head): state + torque model 3.678, position-only
  reference 3.630; start region 4.50 vs 4.36; reset rows 4.59 vs 4.45; the
  10-step position head 1.77 vs 1.04.  Training curves: the full model's
  training NLL falls 3.53 -> 3.17 while its independent validation NLL
  rises 3.67 -> 4.09 (early stopping picks 1,000-3,000 steps); the
  position-only model stays flat (3.61 -> 3.50 / 3.65 -> 3.60).  With the
  same outputs and the same evaluation anchors, the extra inputs make the
  model generalise worse -- overfitting; "memorising anchors" as the
  mechanism remains a hypothesis (one branch per anchor does not by itself
  force it).
* Aggregate masses match the oracle (far route 0.040 / 0.040, goal area
  0.269 / 0.273, hazard zones 0.179 / 0.175); region L1 ~0.50; death AUC
  0.87 (0.58 at the start region, where the hazard draw decides).
* The action-conditional at the start region is KEPT in the aggregate:
  logged-detour-torque anchors far-route mass 0.487 (oracle 0.612) vs
  logged-shortcut-torque anchors 0.002 (0.000); AUC 0.98 (oracle 0.86);
  goal-area mass 0.090 vs 0.042 (oracle 0.063 vs 0.036).  At the RESET rows
  it is not: the 4 detour-torque reset anchors get far mass 0.093 (oracle
  0.569), goal mass 0.052 (0.145).

## The arm (`REPORT.md`; the sealed critic_clip0.1 recipe, only the critic's positive goal replaced by a draw from the learned marginal)

| policy | success | detour | death | timeout |
|---|---|---|---|---|
| start | 0.243 | 0.003 | 0.740 | 0.017 |
| O clip s0 / s1 / s2 | 0.257 / 0.247 / 0.260 | 0.03 / 0.00 / 0.02 | 0.72 / 0.74 / 0.72 | 0.02 / 0.01 / 0.02 |
| CF clip (oracle) | 0.457 / 0.513 / 0.440 | 0.57 / 0.52 / 0.36 | 0.19 / 0.31 / 0.40 | 0.35 / 0.18 / 0.16 |
| **CFL** | **0.253 / 0.247 / 0.247** | 0.01 / 0.01 / 0.00 | 0.74 / 0.74 / 0.75 | 0.01 / 0.01 / 0.00 |

CFL - O success -0.006 (seed s.e. 0.004, 2/3) NOT MET; CFL - start +0.006
(0.002, 3/3); CFL - CF(oracle) -0.221 (0.023, 3/3) -- the whole oracle gain
(+0.216) is lost.  Training: no spikes (clip); critic loss 0.0069 at 30k,
in-batch categorical accuracy 0.008-0.016 (O 0.013, CF oracle 0.026): the
learned positives are less identifiable from (s, a) than the recorded ones.
The actor never leaves the shortcut (far route 3 / 4 / 0 of 300).

## Reading

* The interface as built -- a per-anchor categorical over 0.5 cells,
  cross-fitted, sampled by the stream's future uniform -- does not carry
  the oracle gain, although the aggregate action contrast survives at the
  start region.  Candidate causes, not separated here: (a) the blur (cell
  + jitter + the model's own spread: lower in-batch accuracy, a smoother
  critic); (b) the reset rows, the states the evaluation actually starts
  from, where the model assigns the detour torque almost no far-route mass
  (n = 4 anchors); (c) the model's poorer (s, a)-conditional generalisation
  (worse than position-only).  The next diagnostic is the critic readout at
  the evaluation reset states (detour vs shortcut torques, task goal) for
  the CFL critics against the oracle-CF critics -- not run yet.
* For the learned-ETT line the lesson is structural: what the critic needs
  is the EFFECT of the first action, and a marginal model fitted on one
  branch per anchor loses it where it matters.  The user's separation --
  the ETT fixed as an environment-transition model, the continuation agent
  free -- points to a one-step dynamics + hazard / death model rolled out
  with the agent (267k logged transitions + 7.7 M branch rows of
  supervision for the dynamics; only the death / hazard part needs the
  counterfactual labels).  Design, not a result.
