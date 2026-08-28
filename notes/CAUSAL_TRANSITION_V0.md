# CAUSAL TRANSITION — V0 FROZEN STARTING POINT

Branch: `feature/causal-transition-v0`, forked from `527c702` (the last commit
before the Flow / worst-case line begins at `9c75921`).

This note is the authoritative description of the V0 base. Everything below was
verified against the repository and the on-disk artifacts, not copied from a
plan. Launcher: `scripts/run_causal_transition_v0.py`.

---

## Environment

| | |
|---|---|
| benchmark | `offline_ant_umaze_rockfall` (`crl/rockfall_ant.py`) |
| protocol version | `local_detour_v2.1_sev0.80` |
| severity (eval lethality) | `(0.80, 0.15, 0.05)` = `rockfall_v2_teacher.SEVERITY_V2` |
| `p_active` (hazard-mask density) | **0.30** |
| horizon `H` | **800** |
| reset fix | **enabled** (`resetfix_v1`, canonical episode-independent full reset) |
| `death_settle_substeps` | **0** for training/eval (see note below) |
| obs / goal / action dims | 29 / 29 / 8 (58-wide stored rows) |

Recipe (`scripts/verify_offline_d4rl.py::build_offline_cfg`, unchanged):
binary NCE, `twin_q=True` (actor uses min), `bc_coef=0.05`, `random_goals=0.0`,
`entropy_coefficient=0.0` (fixed alpha), `batch_size=1024`, `repr_dim=16`,
`hidden_layer_sizes=(1024, 1024)`, `discount=0.99`, lr `3e-4`,
`num_sgd_steps_per_step=4`, `guard_abort=True`, `num_actors=0` (offline).

### Rockfall overrides are now declared `Config` fields

Previously `rockfall_severity` / `rockfall_p_active` / `rockfall_max_steps` /
`rockfall_death_settle_substeps` / `rockfall_reset_fix` were set as **ad-hoc
attributes** on the config instance and read back in `crl/envs.py` with
`getattr(config, 'rockfall_*', <default>)`. Two consequences: a dropped or
misspelled assignment degraded **silently** to the older, easier setting instead
of raising, and the values never appeared in the startup `Config(...)` banner,
so a run's log did not record which benchmark it actually ran.

They are now **declared fields on `crl.config.Config`** with defaults
`None / None / None / None / False` — exactly the fallbacks `make_env` already
used. Verified: an unset `Config` still produces the legacy env
`severity=(0.55, 0.30, 0.15)`, `p_active=0.20`, `H=700`,
`death_settle_substeps=0`, `full_reset=False`, so every pre-existing v1 /
p=0.20 / H=700 / legacy-reset run stays byte-identical.

Audited before the change: these five were the **only** undeclared attributes
assigned on `Config` anywhere in `crl/`, `scripts/` or `propensity/`.

This does not make typos impossible (a plain dataclass still accepts unknown
attribute assignment), which is why `run_causal_transition_v0.py` keeps its
post-construction `assert_env_matches()`. A strict `__setattr__` was considered
and **rejected**: Colab notebooks outside this repo also build `Config` objects,
and a repo-wide behavioural change to the config class could break them
silently.

**`death_settle_substeps` disambiguation.** The number 80 appears in the bank's
provenance as `death_settle_substeps=80`. That is a **bank-construction**
parameter (80 extra MuJoCo substeps run inside the fatal transition so the
recorded post-fatal state is physically settled). It is **not** the bank size
and it is **not** applied to the training/eval env, which stays at the frozen
default 0 so V0 remains byte-comparable with the existing H800 runs.

---

## Data — clean training dataset

```
artifacts/rockfall_v2_p30_h800_resetfix/failure_split/
    antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz
```

SHA-256 `6bec8a52e771569c4edc14ff0c7319df4322fe6d74e6e69a6a7074fc76be1852`
— **verified on disk.**

| | |
|---|---|
| shape | `obs [284, 801, 58]`, `act [284, 801, 8]`, `eval_goals [284, 2]` |
| episodes | 284 |
| transitions | 227,200 |
| source pilot | `..._resetfix_pilot.npz`, sha `08bdc44b…`, 300 eps / 227,764 transitions |

**What was removed:** the 16 episodes whose sidecar `dead` flag is true (severe
rock contact → burial), 564 transitions total. Episode ids
`1, 66, 109, 116, 121, 187, 205, 210, 223, 238, 241, 247, 252, 276, 292, 294`.
They are preserved separately in `..._pilot_rockfail.npz`.

One **impaired but not dead** episode (id 63) is deliberately **kept** in the
clean set; the split rule is death, not impairment.

---

## Failure bank

```
artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz
```

SHA-256 `8c50202403317e8adc67670be716fffe5c3b3cb891017bfbe6eb070acd61d0ce`
— **verified on disk.**

| | |
|---|---|
| size | **16** states × 29 dims (`goals`) |
| source episodes | exactly the 16 removed death episodes above |
| construction | `obs[e, collapse_step + 1, :29]` from the patched env |
| settle | `death_settle_substeps = 80` |
| manifest | `artifacts/settled_failure_bank_alpha01/bank_manifest.json`, 12/12 checks true |

**"Settled" means:** at the fatal contact the actor's control is zeroed and 80
extra MuJoCo substeps of gravity / rock / contact / mud physics are integrated
*inside* the fatal transition, so the banked state is the physically collapsed
pose rather than the healthy-looking frozen pose captured at the instant of
contact.

**Legacy bank is refused.** The pre-patch bank
(`.../failure_split/failure_bank.npz`, sha `8d35b76a…`) is preserved unchanged
but the V0 launcher aborts explicitly if it sees that SHA, so V0 cannot
silently reproduce the superseded experiment.

---

## CRL sampling (unchanged from `527c702`, verified in `crl/losses.py`)

Semantics of the failure-aware negative mixture
`q_alpha = (1 - alpha) * p_clean + alpha * q_fail`, implemented as a **loss-level**
mixture that preserves the original positive/negative weighting:

```
L(alpha) = pos_term                         # positive term UNCHANGED
         + (1 - alpha) * neg_ordinary       # in-batch off-diagonal negatives
         + alpha       * neg_failure        # exact uniform expectation over the bank
```

* **Positive sampling is untouched** — the original future-achieved-state goal
  law, `P(j) ∝ discount^(j-t)`, straight out of `crl/replay.py`.
* **Ordinary in-batch negatives remain** — the `B × B` off-diagonal elements.
* **Failure-bank states are EXTRA negative goals**, never positives. Every bank
  state is scored against every in-batch anchor `(s_i, a_i)` and averaged
  exactly (all 16 scored, uniform `q_fail`), so no sampling noise enters.
* **Failure states are always labelled negative** for ordinary anchors
  (`labels = zeros`).
* `alpha = 0.1` for the V0 failure-aware arm.
* `alpha = 0` skips the failure branch entirely (`fail_enabled = False`), and the
  loss reduces algebraically to `jnp.mean(loss)` — **byte-identical to the
  baseline**, gradients included. The V0 launcher additionally leaves
  `fail_bank_path` empty on that arm.

Guard rails already present: failure-aware negatives are refused for
`use_td` / `use_cpc` / `use_gcbc`; `alpha` must be in `(0, 1)`; `alpha > 0`
without a bank raises; bank size must be `<= batch_size`.

Auditable metrics emitted per update: `critic_pos_term`,
`critic_neg_ord_term`, `critic_neg_fail_term`, `critic_neg_ord_raw`,
`critic_neg_fail_raw`, `fail_neg_alpha`, `fail_bank_size`, `logits_fail_neg`.
The identity `pos + ord + fail == critic_loss` is asserted by the smoke test.

---

## Agreement module

`propensity/` is **retained in full** — `agreement.py`, `discriminator.py`,
`dataset.py`, `crl_policy_adapter.py`, the eval scripts, and
`scripts/m0_agreement_equivalence.py`. Theory note:
`notes/continuous_action_agreement.md`.

**It is infrastructure only at V0.** `D_psi(s, g_cmd, a)` is **NOT** used to
route the diagonal / off-diagonal training branches of the new transition
model, and is not wired into training at all on this branch.

Two carried-forward caveats, both recorded in the agreement note:

1. `D_psi` is **not** a calibrated propensity or a calibrated Bernoulli branch
   probability. It is the posterior of an artificial balanced classification
   problem. Calibrating it and renaming it a propensity would be invalid.
2. Per-example scores are **seed-sensitive** (pairwise Pearson 0.53–0.62 across
   seeds). Estimator variance is the open issue for any downstream use.

Note: `propensity/` also contains `flow.py` / `train_flow.py` / `eval_flow.py` /
`sweep_flow_steps.py`. This is the **BC-Flow behavior generator used by the
agreement audit** — a different object from the Flow candidate generator of the
deprecated worst-case line (`scripts/train_flow_v*.py`, which do not exist on
this branch). It came along with the agreement module and is currently unused;
per the agreement note §6 the discriminator deliberately uses **real** offline
actions as positives, not BC-Flow samples, because BC-Flow positives handed the
discriminator a near-perfect source-identification shortcut
(AUC 0.97 BC-Flow-vs-CRL, 0.96 BC-Flow-vs-real, 0.52 real-vs-CRL).

---

## New causal transition redesign

**NOT IMPLEMENTED YET.** Nothing in this branch implements it, and this task
deliberately did not start it.

Planned model concept:

```
F_theta(s, x, x_prime) -> s_next
```

with two future training cases:

1. **diagonal** — `x = x_prime`
   observational / consistency objective (identified from the offline data).

2. **off-diagonal** — `x != x_prime`
   pessimistic / worst-case objective (not identified; this is where the
   Manski-style lower bound enters).

These will eventually be constrained by a **Lipschitz condition**. The two
places Lipschitz enters the bound are written up in
`notes/continuous_manski_lemma2prime.md` (A1 transition-in-action `L_a`,
A2 value-in-state `L_s`), which is currently archived, not implemented.

**Do not implement these losses without a fresh decision.** Open questions
recorded before any implementation starts:

* whether the accumulated slack `(L_a·h + L_s·δ) / (1 - γ)` is small enough to
  be non-vacuous on this benchmark — arithmetic, not engineering, and it gates
  the whole Lipschitz direction;
* whether the off-diagonal target should use the curated failure bank or
  ordinary in-batch negatives (these are **not** interchangeable: ordinary
  negatives are random states, so "most similar to a negative" means *generic*,
  not *worst*);
* how `lambda` (the pessimism weight) is chosen and justified;
* whether the critic must also be Lipschitz-certified (A2), which interacts
  with the NCE logit dynamic range.

### Deprecated — do NOT reuse

The following are **deprecated for this redesign**. They remain in git history
on `feature/continuous-action-agreement` as historical experiments and must not
be resurrected into the new causal-transition implementation:

* Flow candidate generation (`scripts/train_flow_v0…v3.py`, V0.5/V1/V2/V3 arms)
* `crl/static_worstcase.py`
* `crl/pessimistic_positive.py`
* `wc_positive`, `wc_table`, `wc_table_sha256`, `wc_rho_mode`, `wc_p_wc`,
  `wc_dpsi_model`, `wc_coin_seed`, `arm_name`
* the precomputed worst-case **positive replacement** (a static table supplying
  `s'_wc`)
* Bernoulli / `rho` switching between nominal and pessimistic **positives**

None of these files or config fields exist on this branch, and
`scripts/run_causal_transition_v0.py` imports none of them.

The specific idea being retired: **replacing contrastive positive goals with
worst-case generated states**. The redesign models the transition instead.

---

## Launcher

`scripts/run_causal_transition_v0.py` — the only sanctioned V0 entry point.

```bash
python scripts/run_causal_transition_v0.py --check-only            # gate + env assertions
python scripts/run_causal_transition_v0.py --smoke --allow-cpu     # + 5 learner updates
python scripts/run_causal_transition_v0.py --run                   # 300k production (GPU)
python scripts/run_causal_transition_v0.py --run --alpha 0         # baseline arm
```

Why it exists: `scripts/naive_rockfall_v2_crl.py` defaults to the **older,
easier** setting for every authoritative knob when a flag is omitted —
`--npz` → the pre-H800 v2 pilot, `--p-active` → env default 0.20,
`--horizon` → env default 700, `--reset-fix` → off, `--fail-bank` → none.

The V0 launcher closes that hole:

* the authoritative configuration is frozen in module constants — there is no
  flag to change the dataset, density, horizon, reset-fix or severity;
* `--alpha` accepts only the two pre-registered arms, `0.1` and `0.0`;
* a hard provenance gate runs first: SHA-256 of both artifacts, bank shape
  `(16, 29)`, `bank_manifest.json` checks all true, held-out fresh-death list
  present, explicit refusal of the legacy bank SHA, GPU backend required unless
  `--allow-cpu`;
* `assert_env_matches()` constructs the env and **asserts every override
  actually landed** (`p_active`, `severity_probs`, `max_episode_steps`,
  `full_reset`, `death_settle_substeps`, obs/action dims) — this is the check
  the `getattr` path cannot perform for itself;
* `SEVERITY_V2` is imported from `rockfall_v2_teacher` and cross-checked against
  the V0 literal, so drift on either side aborts.
