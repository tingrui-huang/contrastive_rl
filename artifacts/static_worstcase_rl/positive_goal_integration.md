# Worst-case state as a future-positive goal: integration report

Task sections 1-14. **G2 FAILED -> no 5k smoke test was run, no production
command was prepared.** Everything independent of the branch probability was
completed.

`crl/losses.py`, `crl/config.py`, `crl/train.py` and `crl/replay.py` are
**untouched**. No training of any kind was started.

---

## 1. D_psi calibration verdict -- **5B: NOT CALIBRATED. STOP.**

Full evidence in `g2_calibration_audit.json`.

### What D_psi actually is

| aspect | finding | source |
|---|---|---|
| training loss | `bce_with_logits`, labels 1 = real behavior action, 0 = CRL target action, **balanced 50/50 by construction** | `propensity/discriminator.py:104-110` |
| network output | **raw logit** ("Linear(256)-ReLU-Linear(256)-ReLU-Linear(1), raw logit output") | `propensity/discriminator.py:74` |
| downstream activation | plain `jax.nn.sigmoid` | `propensity/agreement.py:195` |
| calibration layer | **none** -- no temperature scaling, no Platt, no isotonic, no prior correction anywhere in `propensity/` | audit |
| ideal optimum | `D*(x) = p_behavior(x) / (p_behavior(x) + p_target(x))` | `propensity/discriminator.py:5-16` |

The estimand is a **relative density ratio under an artificial 50/50 class
prior**, not `rho`. Two independent reasons it cannot be the coin:

1. **Prior shift.** The 50/50 training prior is artificial, so the posterior is
   shifted away from any real-world event rate.
2. **Policy dependence.** `p_target` sits in the denominator, so `D*` is not a
   property of the behavior policy alone. `rho` must not depend on the policy
   being trained -- but `D*` does, and `pi` changes throughout RL while
   `D_psi` stays frozen. Worse, the negatives were generated from
   `naive_rockfall_v2_p30_h800_resetfix_s0_300k/final.pkl` -- a **different
   checkpoint** from the policy this run would train, so even the ratio it does
   estimate refers to the wrong `pi`.

### The repository already says so, in three places

* `discriminator.py:8` -- *"must NOT be used as a causal mixture weight"*
* `agreement.py:193-195` -- *"NOT a literal continuous propensity probability
  and NOT a calibrated causal branch probability"*
* `agreement.py:41` -- *"no calibration into a claimed propensity"*

and records it as machine-readable flags:
`eval.json -> "NOT_a_propensity": true`,
`eval_agreement.py:185 -> 'is_calibrated_propensity': False`.

### Ranking performance (what it CAN do)

| model | ROC-AUC |
|---|--:|
| `D(s, g_cmd, a)` seed 0 / 1 / 2 | 0.5672 / 0.5643 / 0.5715 |
| **mean** | **0.5676** |
| baseline A (goal-marginalized `D(s,a)`) | 0.5724 |
| baseline action-only | 0.5285 |
| baseline context-only (must be chance) | 0.5000 |

n = 8192 held-out contexts, 30 held-out episodes. Test BCE **0.6903** versus
**log 2 = 0.6931** for a constant-0.5 predictor: the model is almost exactly as
good as always answering "0.5". It is a weak ranking signal, above chance but
below the goal-marginalized baseline.

### No calibration diagnostic has ever been computed

No reliability curve, no Brier score, no ECE exists for this model anywhere in
the repo. The only calibration-related records are the explicit **negative**
declarations above.

### M0 already certified the distinction

`scripts/m0_agreement_equivalence.py` (all gates PASS,
`artifacts/m0_agreement/report.json`) was built precisely to separate three
quantities. Its T3:

* **(a) agreement estimator** -> recovers `beta(x|s)` -- *this is the
  propensity*;
* **(b) source classifier data-vs-BC** -> degenerates to 0.5, *NOT a
  propensity*;
* **(c) source classifier behavior-vs-pi** -> recovers `beta/(beta+pi)` --
  CFQL's `D`.

**`D_psi` is case (c).** M0 therefore already certified that `D_psi` is not the
coin, and that the **agreement EVENT frequency** (case a) is. This is the exact
confusion M0 was written to catch.

### What the coin would actually do -- empirical rho on the training dataset

Evaluated over all **227,200** transitions of the authoritative clean dataset:

| statistic | value |
|---|--:|
| mean | 0.5050 |
| median | 0.4992 |
| p10 / p90 | 0.3930 / 0.6181 |
| min / max | 0.0196 / 0.9818 |
| std | 0.0986 |
| within 0.05 of 0.5 | **47.1 %** |
| below 0.1 | 0.01 % |
| above 0.9 | 0.76 % |
| implied pessimistic-branch rate `1 - E[rho]` | **0.4950** |

The pessimistic branch would fire at a near-**constant ~49.5 %** for
essentially every transition. That is operationally identical to the *"fixed
worst-case branch rate"* the task explicitly forbids: it would carry almost no
state- or action-dependent causal signal, only a hidden global coefficient.

### Minimal calibration step required

Build a **continuous agreement-EVENT estimator** -- the case-(a) quantity M0
certified -- not a source classifier:

1. use the already-trained behavior flow `mu(a | s, g_cmd)`
   (`propensity/flow.py`) to draw a fresh behavior action `a~` at the same
   `(s, g_cmd)`;
2. define agreement as a neighborhood event on `a~` vs `a^data` -- the
   continuous analogue of `1[bin(a~) = bin(a^data)]`;
3. estimate `rho` as that event frequency, which is a genuine probability by
   construction and needs no post-hoc calibration;
4. validate against a held-out empirical agreement frequency with a reliability
   curve.

**Caveat that makes this a method decision, not an implementation detail:**
step 2 introduces a **neighborhood bandwidth**, itself a new coefficient
requiring its own pre-registration. The archived discrete sampler hid this
choice inside `action_bins(n_sectors=8, zero_thresh=0.15)`.

---

## 2. Exact branch equation (implemented, coin left open)

```
B_t ~ Bernoulli(rho_t)                      rho_t INJECTED, not chosen here

g+_t =  obs_to_goal(s_j),  j ~ Cat(gamma^(j-t)), j > t, same episode   if B_t = 1
        obs_to_goal(s'_wc,t)                                           if B_t = 0

s'_wc,t = table[flat(e, t)]        precomputed, absorbing: s'_wc -> s'_wc -> ...
```

Nothing else in the objective changes:

```
L_critic = pos_term + (1-alpha)*neg_ord + alpha*neg_fail        alpha = 0.1
L_actor  = bc*(-log pi(a^data|s,g)) + (1-bc)*(alpha_ent*log pi - min_m f^m)
```

No new coefficient, no extra TD term, no actor change, no change to Critic C,
the failure-negative weight, or ordinary negative sampling.

---

## 3. Exact modified files / functions

| file | status |
|---|---|
| `crl/static_worstcase.py` | extended: optional `key=` argument on `worst_case_next_state` (default `None` reproduces the sealed `PRNGKey(11)` draw bit-for-bit) |
| `crl/pessimistic_positive.py` | **NEW** -- `PessimisticPositiveBuffer`, the branch sampler |
| `scripts/precompute_worstcase_table.py` | **NEW** -- section 2 static table |
| `scripts/audit_g2_branch_probability.py` | **NEW** -- section 5 audit |
| `scripts/test_pessimistic_positive.py` | **NEW** -- section 8 tests |
| `scripts/diagnose_pair_dependence.py` | **NEW** -- section 11 |
| `scripts/diagnose_positive_goal_distribution.py` | **NEW** -- section 9 |
| `crl/losses.py`, `crl/config.py`, `crl/train.py`, `crl/replay.py` | **UNTOUCHED** |

Nothing in `crl/train.py` constructs `PessimisticPositiveBuffer`: the sampler
is not wired into any training path, because that would require a coin.

---

## 4. Positive-goal sampler: before / after

```python
# BEFORE -- crl/replay.py:207 TrajectoryBuffer.sample (UNCHANGED, still the
# authoritative path)
traj, i, j = self._draw_indices(batch_size)
goal_state = self._obs[traj, j, :obs_dim]
goal       = obs_to_goal(goal_state, ...)

# AFTER -- crl/pessimistic_positive.py PessimisticPositiveBuffer.sample
traj, i, j = base._draw_indices(batch_size)          # IDENTICAL anchor+future law
goal_state = base._obs[traj, j, :obs_dim]            # nominal, unchanged
rho        = rho_fn(state, g_cmd, action)            # INJECTED, required arg
nominal    = coin_rng.random(batch) < rho            # SEPARATE rng stream
rows                 = row_map[traj*L + i]           # anchor's flat index
goal_state[~nominal] = s_wc[rows]                    # pure table lookup
goal       = obs_to_goal(goal_state, ...)
```

Design points that matter:

* the coin uses its **own RNG stream**, so with `B=1` the sampler is *bitwise*
  identical to the baseline under the same base RNG state (test T1);
* the pessimistic path is a **pure table lookup** -- no walk continuation, no
  policy query at `s'_wc`, no nearest-neighbour projection, no second Flow call
  (test T6, enforced by an AST/symbol audit);
* only the **goal** changes; state, action, next_state and next_action are
  untouched in both branches (test T3).

### Section 4 semantic note (respected)

"Positive" means *drawn from the future occupancy of this `(s,a)`*, not
*desirable*. A fatal-looking `s'_wc` is a legitimate positive for a risky
anchor under the pessimistic world. The failure-negative mechanism is
**retained unchanged** at `alpha = 0.1`; see section 11 below for the empirical
check that this is not self-contradictory.

---

## 5. Unit-test results

**`crl/static_worstcase.py`: 16/16 PASS** (`unit_tests.json`) -- unchanged by
the `key=` addition; still reproduces the sealed selector on **50/50** anchors.

**`crl/pessimistic_positive.py`: 23/23 PASS** (`unit_tests_sampler.json`).
A synthetic table with one distinct, index-marked `s_wc` per transition makes
the assertions exact.

| test | result |
|---|---|
| T1a-d nominal branch **bitwise identical** to `TrajectoryBuffer.sample` (goal, action, next_obs, next_action) | PASS |
| T2a worst-case goal is **exactly** `obs_to_goal(s'_wc)` | PASS |
| T2b lookup keyed on the **anchor's** flat index (proved by an index marker) | PASS |
| T3a-d only the goal changes; it does change | PASS |
| T4a empirical nominal rate matches rho = 0.00 / 0.25 / 0.75 / 1.00 (0.0000 / 0.2538 / 0.7512 / 1.0000) | PASS |
| T4b coin reproducible under the same seed | PASS |
| T4c rho outside [0,1] rejected | PASS |
| T4d `rho_fn` is a **required** argument | PASS |
| T5a-b mixed batch routes every row to the correct source | PASS |
| T6a-c no Flow/policy/projection import, no continuation or projection symbols, pessimistic path is a pure lookup | PASS |
| T7 forcing a branch does not consume the coin RNG | PASS |

---

## 7. Section 11 -- pair-dependence, NOT global contradiction

`pair_dependence.json`, 512 dataset anchors x the 16 failure negatives, frozen
Critic C (read-only, never in worst-case selection).

**Q2 variance decomposition of `f_C(s_i, a_i, g^f_b)`:**

| component | share |
|---|--:|
| goal main effect (the "global score" part) | 0.166 |
| anchor main effect | 0.680 |
| anchor x goal interaction | 0.155 |
| **pair-dependent total (anchor + interaction)** | **0.834** |

**Q3 -- the decisive test.** The *same 16 failure goals*, scored by different
anchors, split by whether the anchor's own frozen `s'_wc` lands near the bank:

| anchor group | mean `f_C` on the 16 failure goals |
|---|--:|
| risky (`d_neg <= p25`) | **-68.97** |
| safe (`d_neg >= p75`) | **-100.39** |
| difference | **+31.42**, bootstrap CI95 **[23.90, 38.85]** |

**Q4.** `f_C(s_i, a_i, own s'_wc)` mean **-4.61** versus bank goals paired with
unrelated anchors mean **-90.97**.

**Conclusion.** Only 16.6 % of the score variance is a per-goal global effect;
the critic resolves the *same* goal by tens of logits depending on the anchor,
in the expected direction (risky anchors rate failure goals higher). So the two
roles -- pessimistic positive for a risky anchor, failure negative for
unrelated anchors -- are **not globally contradictory**.

*Caveat:* measured on a Critic C trained **without** any pessimistic-positive
branch. It shows the architecture resolves goals pair-dependently; it does not
predict what training *with* the branch would do.

---

## 8. Runtime / profile

Frozen K = 256 and 50 Euler steps, never reduced. Local CPU: ~12.7k
candidates/s, **20.1 ms/anchor** batched, <1 GB at 256 anchors
(`profile.json`).

Online generation would cost 82 s per learner step (6854 h for 300k) --
infeasible. **It never runs online.** With `a^data` conditioning and everything
frozen, the map is a static table: **227,200 transitions**, built once, ~26 MB.
RL then carries **zero** Flow cost per update at any branch rate.

### Table noise convention (one new, unavoidable convention -- recorded)

The sealed convention is a single `normal(PRNGKey(11), (n*K, 29))` draw per
call. Reusing one key across chunks would give transitions `chunk` apart
**identical x0 noise**, so each chunk uses
`jax.random.fold_in(PRNGKey(11), chunk_index)` with `chunk = 256`. The frozen
root seed is preserved and **no selector constant changes** (K, 50-step Euler,
16-state bank, V0 normalization, Euclidean L2, lowest-index tie-break). The
chunk size is recorded in `worstcase_table_manifest.json`; the table is
reproducible only with it. `worst_case_next_state` still defaults to the exact
sealed draw, so the 50/50 sealed reproduction test is unaffected.

---

## 9 / 10 / 14. BLOCKED items

| section | status |
|---|---|
| 9 branch-rate diagnostics | **PART 1 only** (rho-independent goal populations). PART 2 -- mean/median/p10/p90 rho, empirical branch rates, fraction of positives from `s'_wc` -- is **not reported**: it requires a coin that does not exist. |
| 10 5k smoke test | **NOT RUN.** Section 5B: "STOP before RL training"; the hard stop repeats "stop before the smoke test". |
| 14 SSH runbook + production command | **NOT PREPARED.** A production command cannot be written for an objective whose branch probability is undefined. |

---

## 12. Status wording

This is **approximate support-restricted pessimistic occupancy sampling**. It
is **not** a certified Manski lower bound: V3 coverage on the sealed set is
31/50 = 0.620, so `N_Flow(s,a) contains supp P(s'|s,a)` is **not** established.
No theorem claim was modified.

---

## 9'. Unresolved issues

1. **G2 (blocking).** No calibrated `rho`. Needs the agreement-event estimator
   above -- including a pre-registered neighborhood bandwidth.
2. **D_psi provenance mismatch.** Its negatives came from the *naive* rockfall
   checkpoint and the *full* dataset (sha `08bdc44b...`), while the run to be
   trained uses the *clean* split (sha `6bec8a52...`) and the settled-bank
   policy. Even as a ranking score it refers to a different `pi`.
3. **Coverage (recorded, not blocking).** 62 % fatal coverage means the
   pessimistic branch sometimes returns a `s'_wc` that is not failure-like;
   under an absorbing rule that goal is committed to with no fallback.
4. **Absorbing semantics interact with `gamma`.** Once the branch fires the
   geometric future collapses to a point mass, so the pessimistic positive
   carries no `gamma` weighting while the nominal one does. This is a direct
   consequence of the section-3 definition and is recorded, not altered.
