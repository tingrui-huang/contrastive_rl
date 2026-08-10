# Continuous-action agreement surrogate — theoretical status

Status: **finalized (Stage 3B)**. This note records what the propensity-side
object is and, more importantly, what it is not. It does not claim a theorem.

## 1. Discrete causal motivation

In the discrete causal formulation, exact action agreement decides how much
mass a state–action pair contributes to the observational (nominal) branch
versus the pessimistic branch. The deciding quantity is the propensity mass

    P(A = a | S = s)

together with the hard agreement indicator `1[a = a_obs]`.

## 2. Why the discrete object does not transfer

In a continuous action space the exact point probability is zero,

    P(A = a | S = s) = 0   for every a,

so the discrete propensity term has no direct continuous analogue and the hard
agreement indicator is almost surely zero. We do **not** attempt to reconstruct
it. This is the same practical step CFQL takes.

## 3. Practical continuous approximation

We replace the hard agreement indicator with a **learned soft agreement score**

    D_psi(s, g_cmd, a) ∈ [0, 1]

    high D  — the queried action agrees with / resembles actions supported by
              the observational behavior policy in this pre-action context
    low  D  — the queried action disagrees with it

`D` is trained as a balanced binary classifier. Under an artificial 50/50 class
prior an ideal classifier recovers

    D*(x) = p_behavior(x) / (p_behavior(x) + p_target(x)),

a **relative** agreement/discrepancy score.

## 4. Training

    positive:  (s_i, g_cmd_i, a_real_i)  ->  1
    negative:  (s_i, g_cmd_i, a_crl_i)   ->  0

    a_real_i    real behavior action stored in the frozen offline dataset
    a_crl_i     ~ pi(. | s_i, g_query_i), frozen CRL actor
    g_query_i   future achieved-state goal, sampled with the existing CRL
                replay law (P(j) ∝ discount^(j-t) over j ∈ [t+1, L_e-1])

Loss: binary cross-entropy with logits, exactly balanced batches.

## 5. `g_cmd` vs `g_query` — the load-bearing distinction

    g_cmd    the actual PRE-ACTION commanded goal from the offline data. Drawn
             at env.reset(), constant within the episode, part of the context
             the behavior teacher acted on.   **INPUT TO D.**

    g_query  a hindsight / future achieved state, used by CRL replay to index
             which goal-conditioned value function is queried. It is a
             DESCENDANT of the action.   **NOT AN INPUT TO D.**

`g_query` only selects *which action the CRL actor proposes*. Conditioning the
behavior-support model on it would turn `D` into a hindsight posterior
`p(a_t | s_t, s_{t+k})` — an inverse-dynamics-flavoured object that carries
outcome information — rather than a behavior-support quantity.

General formulation: `D(s, g_cmd, a)`. In the current rockfall dataset only
`g_cmd[0:2]` is live (the other 27 dimensions are identically zero), so the
implementation feeds `g_cmd[:2]`. That is an environment-specific narrowing
recorded in the run metadata, **not** a redefinition of the method.

## 6. Difference from CFQL

CFQL uses BC-Flow samples as positives because its FQL backbone already
contains a behavior generator. The Contrastive RL backbone does not require
one, so we use the **real offline behavior actions directly**.

This is not merely a simplification. The Stage-2.5 boundary-shortcut audit
measured, with boundary/saturation features alone:

    AUC(BC-Flow vs CRL)  ≈ 0.97
    AUC(BC-Flow vs real) ≈ 0.96
    AUC(real   vs CRL)   ≈ 0.52   (chance)

A generative positive would have handed any discriminator a near-perfect
source-identification shortcut that has nothing to do with behavior support.
Real positives remove it at the source.

## 7. What D is NOT

* **not** `P(A = a | s, g_cmd)` — no continuous propensity is claimed;
* **not** a calibrated causal branch probability. `sigmoid(logit)` is the
  posterior of an *artificial* balanced classification problem;
* **not** a density or an explicitly estimated density ratio;
* **not** a distance-to-behavior-manifold estimator — it is trained only to
  separate behavior actions from target actions, so monotone decay away from
  the data is not something it is fit to provide and is not required of it;
* **no** action-neighborhood bandwidth `h` is used anywhere;
* **no** uniform reference distribution is used anywhere.

Calibrating the score and then renaming it a propensity would be invalid.

## 8. Empirical status (rockfall p30 h800, 30 held-out episodes)

Held-out ROC-AUC ≈ 0.564–0.572 across three seeds. This is modest **by
construction**: the CRL actor is BC-regularized (`bc_coef = 0.05`) on this very
dataset, so the target policy is already close to the behavior policy. Modest
AUC therefore indicates strong behavior/target agreement, not a broken
estimator, and is not grounds for changing the formulation.

Controls: context-only classifier at exactly 0.5000 (BCE = log 2), confirming
the positive/negative pairing is exact and leakage-free; action-only at 0.529;
correct-context real actions score ≈ +0.08 above mismatched contexts (win rate
0.64–0.67), so the model genuinely uses context.

Known limitation, carried forward: per-example scores are **seed-sensitive**
(pairwise Pearson 0.53–0.62 across seeds, decile overlap 0.15–0.43). Estimator
variance — not the modeling definition — is the open issue for downstream use.

## 9. Open questions before causal integration

1. Which functional the bound actually needs — `mu` on an absolute scale, or
   the relative `mu/(mu+pi)` that this classifier estimates. `scripts/
   m0_agreement_equivalence.py` notes CFQL's `D` is a discrepancy score, not
   `beta`.
2. Whether the seed variance above is small enough for the quantity that
   enters the bound, particularly in the low-score tail where a pessimism
   weight would bite.
3. `crl/replay.py` currently discards `g_cmd`, so the learner cannot yet see
   the conditioning variable `D` needs (see the Stage-3B integration report).
