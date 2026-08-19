# Integration audit: where can a Flow-selected worst-case next state enter CRL?

Phase 0-1 of the static worst-case RL integration task. **Read-only audit.**
Nothing in `crl/` was modified. No RL training was started.

Repo state: branch `feature/continuous-action-agreement`; commit recorded in
`audit_provenance.json` next to this file.

---

## 0A. The exact objective implemented by the authoritative training path

Entry point of the authoritative rockfall runs:

`scripts/run_failneg_settledbank_h800.sh` -> `scripts/naive_rockfall_v2_crl.py:76`
-> `crl/train.py:train()` -> `crl/losses.py:build_learner()`.

Config = `scripts/verify_offline_d4rl.py:64 build_offline_cfg()` overridden by
`scripts/naive_rockfall_v2_crl.py`:

| knob | value | source |
|---|---|---|
| `use_td` / `use_cpc` / `use_gcbc` | False / False / False | `verify_offline_d4rl.py:73-74` |
| `twin_q` | True | `verify_offline_d4rl.py:74` |
| `bc_coef` | 0.05 | `verify_offline_d4rl.py:75` |
| `random_goals` | 0.0 | `verify_offline_d4rl.py:76` |
| `entropy_coefficient` | 0.0 (fixed, non-adaptive) | `verify_offline_d4rl.py:77` |
| `batch_size` / `repr_dim` / `hidden` | 1024 / 16 / (1024,1024) | `verify_offline_d4rl.py:79` |
| `discount` | 0.99 | `verify_offline_d4rl.py:80` |
| `num_sgd_steps_per_step` | 4 | `verify_offline_d4rl.py:82` |
| `fail_neg_alpha` | 0.1 | `run_failneg_settledbank_h800.sh:33` |
| `fail_bank_path` | `artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz` | ibid. |
| steps / seed | 300000 / 0 | ibid. |

### Data law (`crl/replay.py:177 _draw_indices`, `crl/replay.py:207 sample`)

```
traj ~ U{0..E-1};  i ~ U{0..L-2};  j ~ Cat( p(j) prop. gamma^(j-i) ), j > i, SAME episode
g                = obs_to_goal(s_j)        # goal_indices = range(29): full achieved state
observation      = concat(s_i,   g)
next_observation = concat(s_i+1, g)
action           = a_i^data ;  next_action = a_{i+1}^data
reward = 0, discount = gamma               # both unread by the losses
```

### Critic loss (`crl/losses.py:105-252`)

Logits, twin `m in {1,2}` (`crl/networks.py`):

```
L^m_ij = f^m(s_i, a_i, g_j) = phi^m(s_i, a_i)^T psi^m(g_j)
```

Base elementwise BCE matrix (`losses.py:167-176`), `I` = identity:

```
l_ij = mean_m BCE( L^m_ij , I_ij )
```

Failure-aware mixture (`losses.py:204-221`), `alpha = 0.1`, bank
`Gf = {g^f_1 .. g^f_16}` (settled fatal states in goal coords):

```
pos_term = sum_i l_ii / B^2
neg_ord  = sum_{i != j} l_ij / B^2
neg_fail = (B-1)/B * mean_{i,b} BCE( f(s_i, a_i, g^f_b) , 0 )

L_critic = pos_term + (1 - alpha) * neg_ord + alpha * neg_fail          (losses.py:221)
```

At `alpha = 0` this reduces algebraically to `mean(l)` (`losses.py:235`).

### Actor loss (`crl/losses.py:255-334`), `random_goals = 0.0` -> `losses.py:265-268`

```
new_state = s ; new_goal = g ; orig_action = a^data
a ~ pi(.|s,g) ;  q = min_m f^m(s, a, g)          (diagonal, losses.py:289-290)
q_term  = alpha_ent * log pi(a|s,g) - q          with alpha_ent = 0
L_actor = bc * ( -log pi(a^data|s,g) ) + (1 - bc) * q_term              (losses.py:317)
```

### Alpha loss

Inactive: `entropy_coefficient = 0.0` is not None, so
`adaptive_entropy_coefficient = False` (`losses.py:53`).

### Target network

`target_q_params` is Polyak-updated (`losses.py:372`) but read **only** inside
the `config.use_td` branch (`losses.py:138`). With `use_td = False` the target
critic **never enters the objective**.

### >>> CRITICAL FINDING <<<

`transitions.next_observation` and `transitions.next_action` are populated by
the buffer but read **only** in the `use_td` branch (`losses.py:107-164`). In
the authoritative rockfall configuration (`use_td = False`) **the next state
does not enter the objective at all.** There is no transition slot, no Bellman
backup and no target-critic input for `s'_wc` to occupy.

---

## 0B. What a "worst-case next state" means to the existing code

Repo-wide search (`worst_case`, `worst`, `next_state`, `transition`, `causal`,
`occupancy`, `min`, `failure`, `fail_neg`) over `crl/`, `scripts/`,
`propensity/`:

**On this branch there is no worst-case branch anywhere in the RL path.**
`crl/losses.py`, `crl/train.py`, `crl/replay.py` and `crl/config.py` contain no
worst-case symbol, field or code path.

The only worst-case implementation that has ever existed in this repo is the
**archived continuous-Manski d_lb positive-goal sampler**:

* source: `manski-port-archive:crl/manski.py`; verbatim copy at
  `scripts/manski_archive_c1368c7.py` (header: *"DO NOT EDIT"*)
* wiring: `manski-port-archive:crl/train.py:324-336`, which **wraps the frozen
  buffer** with `manski.build_positive_buffer(...)`
* config surface: `manski-port-archive:crl/config.py:85-98`
  (`manski_positives`, `manski_table`, `manski_p_override`, `manski_hazard`,
  `manski_reachable`)

Answering the audit questions exactly:

| question | answer |
|---|---|
| expects `s'_wc` as a single next state? | **No.** It expects a **dataset flat index** (`walk_from`'s `cur`, `manski_archive:222-266`). |
| expects a value `V(s'_wc)`? | **No.** Values appear only as a *ranking* (`bfs_dist_map`) used to pick which neighbor is worst. |
| modifies a contrastive positive/negative goal? | **Yes -- the POSITIVE goal.** `ManskiPositiveBuffer.sample` (`manski_archive:311-332`) replaces the same-trajectory truncated-geometric future goal with the endpoint of the Thm-2 d_lb walk. Negatives are untouched (still cross-batch off-diagonals). |
| appears inside a Bellman/occupancy recursion? | **Occupancy, not Bellman.** It is a per-step teleport inside the geometric occupancy walk that defines `d_lb`. |
| alters a target critic? | **No.** |
| does a usable *continuous* insertion point currently exist? | **No.** See the gate below. |

Archived walk semantics (`manski_archive:132-266`), per step:

```
stop w.p. (1 - gamma)
else w.p. P_hat( bin(x_t) | cell(s_t) ):  cur += 1                  # empirical transition
else                                   :  cur <- random dataset index in
                                          argmax_{c in N(s,x)} BFS_dist(c)   # PESSIMISTIC TELEPORT
```

The pessimistic teleport target is the semantic slot `s'_wc` would occupy.

---

## 0C. Action source -- RESOLVED, unambiguously

**Option 1: the factual offline dataset action `a_t^data`.**

Evidence, all from the existing method (not convenience):

1. `manski_archive:137-139` -- *"advance one step along the stored trajectory
   (the empirical P_obs transition; **x_t is the logged behavior action, which
   IS the policy being evaluated**)"*.
2. The branch coin is `P_hat(bin(x_t) | cell(s_t))` with
   `x_t = self._act[cur]` (`manski_archive:171-173`), i.e. the **logged**
   action's propensity. Thm-2 bounds the occupancy of the *behavior* policy.
3. On teleport it re-anchors to a dataset point whose *logged* action is the
   fresh draw -- `manski_archive:141-143`, *"matching X_{t+1} ~ pi(.|S_{t+1})
   in Thm 2"*.
4. The continuous replacement coin conditions on the logged action too:
   `propensity/agreement.py` -- *"exact action agreement 1[a = a_obs] decides
   how much mass goes to the observational (nominal) branch versus the
   pessimistic branch"*; certified equivalent to the table coin by
   `scripts/m0_agreement_equivalence.py` (all gates PASS,
   `artifacts/m0_agreement/report.json`).
5. Consistency check: the frozen V3 Flow was itself trained conditioned on
   `(s, a)` drawn from the offline dataset (`scripts/train_flow_v3.py`), so
   `a^data` is also the in-distribution conditioning for the generator.

No ambiguity. **The Flow must condition on `a_t^data`.**

---

## 0D. Can the Flow remain frozen? -- YES

The archived insertion point lives inside `buffer.sample()`, i.e. in **numpy,
outside the JAX gradient graph** entirely (`crl/train.py:sample_G` moves the
numpy Transition to device arrays only afterwards). Gradients from
`critic_grad` / `actor_grad` (`losses.py:337-338`) cannot reach the generator,
the negative bank or the selector by construction.

No joint optimization is required or possible at this insertion point. `s'_wc`
is a stop-gradient generated quantity. **This requirement is satisfiable.**

---

## HARD AUDIT GATE: **FAILED -- STOP**

The *semantic role* is identifiable (pessimistic teleport target in the
positive-goal occupancy walk), but `s'_wc` is **not type-compatible** with that
slot, and the rule deciding *when the branch fires* does not exist for this
environment. Both gaps require inventing method.

### G1. The occupancy walk cannot continue from a synthetic state

`walk_from` carries a **dataset flat index**. At a teleport destination the walk
needs (a) a *logged action* for the next step's propensity coin and (b) a
*successor state* for the empirical branch. A Flow-generated `s'_wc` in R^29 has
**neither**. Continuing requires one of:

* **terminate the walk at `s'_wc`** and use `obs_to_goal(s'_wc)` as the positive
  goal -- a new stopping rule that changes the definition of the occupancy;
* **project `s'_wc` to its nearest dataset state** and re-anchor -- a new
  projection operator plus a new metric choice;
* **iterate**: query `pi` at `s'_wc` and re-run the Flow -- this is exactly the
  iterative Proposition-2 optimization the task forbids.

All three are new method. None is implied by existing code.

### G2. The branch-firing weight is not wired for this env

The archive's coin is `P_hat(bin(x)|cell)`, a 2-D grid x 9-action-bin table
(`manski_archive:62-69`). It does not exist for the 29-D Ant / 8-D continuous
action.

The repo's *intended* continuous replacement exists and is trained:
`propensity.agreement.D_psi(s, g_cmd, a)` in [0,1], checkpoints at
`artifacts/support_discriminator/D_state_cmdgoal_action{,_seed1,_seed2}`,
described as "integration-ready", and certified coin-equivalent by M0.

But `propensity/__init__.py:16-19` states verbatim:

```
Stage 1  offline dataset interface     <- propensity.dataset (this commit)
Stage 2  behavior flow model           (not implemented)
Stage 3  discriminator + diagnostics   (not implemented)
Stage 4  integration into causal CRL   (not implemented)
```

Stages 2-3 have since been implemented; **Stage 4 has not.** Deciding how `D`
becomes the coin -- raw `D`, a calibration of `D`, a threshold on `D`, or a
fixed rate -- **is a new coefficient**, which the task explicitly forbids
inventing.

### G3. Manski validity does not hold for the Flow candidate set (recorded, not blocking)

Thm-1 validity requires the candidate set `N(s,x)` to be a **superset** of the
true one-step reachable set; the archived `neighborhood()` was constructed to
guarantee that. The sealed measurement gives V3 fatal coverage
**31/50 = 0.620** (`artifacts/state_nn_selector_confirm/summary.json`), so the
Flow candidate set is demonstrably **not** a superset. Any resulting `d_lb`
would be a *heuristic pessimistic occupancy*, not a certified lower bound. This
does not forbid an empirical run, but it does forbid the lower-bound claim.

### Compounding structural point

Because `use_td = False`, `next_observation` is unread (0A). There is no
alternative "transition slot" to fall back on: the positive-goal law is the
**only** existing route by which a next state can influence this objective, and
that route is blocked by G1 + G2.

---

## The unresolved design choice, stated for decision

> **When the pessimistic branch fires, what does the occupancy walk do with a
> synthetic `s'_wc` that has no logged action and no successor -- and what
> decides how often it fires?**

Two coupled sub-decisions, both currently outside the repo:

1. **Continuation rule** (G1): stop-at-`s'_wc` / project-to-dataset / iterate.
2. **Branch weight** (G2): how `propensity.agreement.D_psi(s, g_cmd, a^data)`
   maps to the Bernoulli coin -- this is Stage 4, unimplemented.

Per the task's hard restrictions ("do not invent ... a new loss coefficient",
"do not invent a new RL objective"), **no production run and no loss edit
happens until these are decided by the user.**

---

## What was built anyway (method-neutral, zero objective commitment)

Phases 2-4 were executed because they commit to **nothing** about the RL
objective and are fully specified by already-sealed constants:

* `crl/static_worstcase.py` -- the frozen `(s, a) -> s'_wc` module, all
  provenance SHAs gated at construction, batched.
* `scripts/test_static_worstcase.py` -- regression tests against the sealed
  development candidates.
* `artifacts/static_worstcase_rl/profile.json` -- throughput / memory.

**Not done, deliberately:** no change to `crl/losses.py`, `crl/config.py`,
`crl/train.py` or `crl/replay.py`; no RL smoke (Phase 5); no production run
(Phase 6); no SSH runbook (Phase 7) -- a runbook cannot be written for a launch
command whose objective is undecided.

### Phase 3 results -- 16/16 PASS (`unit_tests.json`)

The packaged module reproduces the sealed selector **exactly**: identical
selected index `k` on **50/50** selector-confirm50 anchors; selected state
matches to 9.5e-07; nearest-negative distance to 4.8e-07; candidate block to
3.6e-06. 19 provenance gates pass and a tampered bank SHA aborts construction.
Verified additionally: K == 256, determinism across calls, lowest-index
tie-break, no non-finite outputs, plain-numpy (stop-gradient) return, no
optimizer/`jax.grad` in the module, **no import of or call into Critic C**, and
no reference to `_dead` / rock mask / severity / oracle pairing in executable
code.

### Phase 4 results (`profile.json`) -- throughput is NOT a blocker

Local CPU backend, K=256 and 50 Euler steps never reduced:

| anchors | candidates | steady state | cand/s | ms/anchor | RSS |
|--:|--:|--:|--:|--:|--:|
| 1 | 256 | 0.046 s | 5561 | 46.03 | 252 MB |
| 16 | 4096 | 0.329 s | 12466 | 20.54 | 358 MB |
| 64 | 16384 | 1.310 s | 12506 | 20.47 | 485 MB |
| 128 | 32768 | 2.589 s | 12659 | 20.22 | 652 MB |
| 256 | 65536 | 5.140 s | 12750 | **20.08** | 960 MB |

Batching saturates at ~12.7k candidates/s (~20 ms/anchor); memory grows
linearly and stays under 1 GB at 256 anchors. Naive per-update generation would
cost 82 s per learner step -> **6854 h for 300k** -- infeasible.

**But it never has to run online.** 0C resolved the conditioning action to
`a^data`; the Flow, bank, normalization and dataset are all frozen. Therefore

    (s_t^data, a_t^data) -> s'_wc

is a **static table over the frozen dataset**. The clean rockfall dataset has
227,200 transitions, so caching every `s'_wc` costs **1.27 h once on this CPU**
(~1.5 min on a GPU) and occupies **26 MB**. RL training then carries **zero**
Flow cost per update, at any branch-firing rate.

Compute is therefore not a constraint on any of the candidate designs in G1/G2.
The blocker is purely semantic.

## Before/after pseudocode (the integration that is BLOCKED)

```python
# BEFORE -- crl/replay.py:207 TrajectoryBuffer.sample (authoritative today)
traj, i, j = draw(gamma)                        # j > i, SAME episode
g          = obs_to_goal(obs[traj, j])
return Transition(observation=concat(obs[traj, i], g),
                  action=act[traj, i], ...)

# AFTER -- the archived worst-case semantics, transposed to continuous
traj, i = draw_anchor()
cur     = flat(traj, i)
while alive:
    alive &= (u() < gamma)                      # geometric stopping
    p = ???                                     # <-- G2 UNRESOLVED (branch weight)
    if u() < p:
        cur += 1                                # empirical branch
    else:
        s_wc = worst_case_next_state(s[cur], a_data[cur])   # FROZEN, built
        cur  = ???                              # <-- G1 UNRESOLVED (continuation)
g = obs_to_goal(state_at(cur))
return Transition(observation=concat(s_anchor, g),
                  action=a_data_anchor, ...)
```

The two `???` are exactly the decisions that must come from the user.
