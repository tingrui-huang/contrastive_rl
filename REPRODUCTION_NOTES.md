# FetchPush reproduction notes (contrastive RL port)

Status of reproducing the 2022 contrastive-RL results after porting the repo off
the dead Acme/Launchpad/Reverb + `gym.envs.robotics` + `mujoco-py` stack onto a
single-process modern-JAX loop (`crl/`). **No causal modifications** are involved
in anything below; this is baseline reproduction only.

## TL;DR
- **FetchReach validates the port.** Binary-NCE contrastive RL solves it
  (mean-last-5 success **0.94**, final object/gripper distance 0.036), so the
  objective, relabeling, networks, and training loop are correct.
- **State FetchPush is not reproduced.** Binary NCE plateaus at ~**0.05–0.11**
  success. The critic *learns* (high categorical accuracy, large logits gap) but
  the **policy never learns goal-directed pushing**.
- We ran **four controlled, bounded ablations** toward the original regime.
  **None closes the gap.** The remaining unmatched differences are expensive or
  impossible to test locally (1M steps, multi-actor data diversity, the image
  env), so we stop short of a bug claim: this is a **characterized reproduction
  gap in a materially different / larger-budget setup**, not a port defect.

---

## FetchReach — port validated
Binary NCE (`use_td=False, twin_q=False`, `random_goals=0.5`, entropy 0.0), 300k:

| success (mean last 5) | final_dist | min_dist | logits_gap | cat_acc | NaN |
|---|---|---|---|---|---|
| **0.94** (max 1.0) | 0.557 → **0.036** | 0.032 | +348 | 0.91 | clean |

All standard sanity signals pass; the pipeline is correct.

## FetchPush — baseline failure and root cause
Vanilla state FetchPush, same NCE config, 300k:

- success **~0.07** (mean last 5); `final_dist` **rises** 0.17 → 0.27; min_dist
  never < 0.05 (success threshold). Critic learns: logits_gap +440, cat_acc 0.78.
- **Diagnosis (trained-policy rollouts vs random):** the trained policy makes
  contact but its object motion is **undirected**, only **~5–16%** of episodes
  end with the object closer to the goal, and it often **knocks the object off
  the table**. The relabeled **positive goals are degenerate**: with an
  object-only goal, most positives are "goal = the object's *current* position"
  (median object positive-spread ≈ 0), so the critic is rewarded for the object
  **staying put** rather than reaching a distant target. This is a
  self-reinforcing loop: degenerate positives → critic rewards a static object →
  actor stops disturbing it → data becomes even more static.

---

## Tested differences (four controlled ablations) — none closes the gap

| # | Ablation | What changed | Result | Verdict |
|---|---|---|---|---|
| 1 | **start-at-object reset** (`fetch_push_start_at_obj`, ports `FetchPushImage._move_hand_to_obj`) | gripper begins within 0.06 of the object; random-policy contact 23%→90%, displacement 3× | success stayed flat ~0.05; the **trained** policy *collapses* the coverage (displacement 0.044 < random 0.107; positives 2.4% < random) and disengages | **ruled out** — contact/initial coverage is not the bottleneck |
| 2 | **adaptive entropy** (`entropy_coefficient=None, target=0`) | matches original image config | learned temperature **α collapsed 1.0→0.02** → effectively deterministic → **no effect** (success 0.04) | **ruled out** — entropy is not the missing factor |
| 3 | **higher update ratio** (4× local, **16× Colab**) | more SGD updates per env step, toward original's ~64 | success **flat 0.06**, distance rising, but critic **sharper** (gap 756, cat-acc 0.84); more updates only sharpen the *degenerate* critic | **ruled out** up to 16× |
| 4 | **full-state goal slice** (`fetch_push_original_style`, literal as-shipped `start=0,end=-1`) | relabel goal = full future state (25-dim) instead of object-only (3-dim) | object positive-spread 2–7% → **40%**, `logits_gap` 200–750 → **55**, displacement 0.04 → **0.44**, contact **98%** — but movement **toward goal still 5%**; success ~**0.11** (noisy 0–0.20); **shoves the object off the table** | **weakened** — changes the *failure mode* (freeze ↔ shove) but **not the outcome** |

**Key cross-cutting observation:** the bottleneck is **goal-directed control** —
learning to push the object toward the *specific* target. That survives every
ablation. The goal-slice (#4) is the most consequential: it fixes the diagnosed
degeneracy and un-freezes the policy, but the policy then shoves the object
undirected (often off the table), so success does not improve materially.

### Goal-slice detail (the most consequential finding)
- **Object-only goal** (`[3:6]`, our `fetch_push`): critic sharpens but the
  policy **freezes / disengages** or pushes poorly; positives are degenerate.
- **Full-state goal** (`[0:-1]`, `fetch_push_original_style`): positive spread
  improves and object movement increases sharply, but movement is **not
  goal-directed** and often **pushes the object off the table**.
- **Therefore:** the goal slice **affects the failure mode but is not sufficient
  to reproduce Push success.**
- The slice is **ambiguous in the original source**: `config.start_index=0,
  end_index=-1` (config.py:78-79) are never overridden in `lp_contrastive.py`
  (→ full-state), yet the comment "should be overwritten, based on each
  environment" (config.py:75) and the object slice hardcoded in
  `fetch_envs.FetchPushEnv.observation` (fetch_envs.py:89-90) imply object-only.

---

## Remaining unmatched differences (untested)

| Difference | Original | Our port | Testable locally? |
|---|---|---|---|
| **Total steps** | 1,000,000 (lp_contrastive.py:105) | 100k–300k | no (compute) |
| **Multi-actor + Reverb data diversity** | 4–10 parallel actors + rate limiter | single process | no (architecture) |
| **Image env / fixed object** | `fetch_push_image`: 64×64 obs, object pinned at [1.15,0.75] (fetch_envs.py:234) | state obs, randomized object | not attempted |
| **Old gym / mujoco-py dynamics** | gym FetchPush-v1 + mujoco-py | gymnasium FetchPush-v4 + `mujoco` | no (can't install dead stack) |
| **Exact original Push config ambiguity** | goal slice + state-vs-image not recoverable from source | assumed state + object-slice | unresolvable from source |

---

## Conservative conclusion
1. **The port is faithful** — FetchReach reproduces, and the objective /
   relabeling / networks are verified identical (see `crl/audit.py`, 21/21).
2. **State FetchPush is not reproduced at accessible compute.** The failure is
   mechanistically understood (degenerate/undirected goal-reaching), and it is
   **not closed by any cheap, source-aligned ablation** (reset, entropy, update
   ratio ≤16×, goal slice).
3. **This is a characterized reproduction gap, not a port bug.** The most
   defensible explanation is a combination of (a) a **materially different or
   larger-budget original setup** — 1M steps, multi-actor data diversity, and
   possibly the image env with a fixed object — and (b) the intrinsic difficulty
   of goal-directed manipulation for binary-NCE-from-scratch at this scale.
4. **It says nothing for or against the contrastive method itself** on state
   FetchPush; it bounds what our current single-process, ≤300k-step setup can
   show.

### Reproducible assets in this repo
- Envs: `fetch_push`, `fetch_push_start_at_obj`, `fetch_push_original_style`
  (`crl/envs.py`).
- Tools: `crl/diagnose_push.py` (object displacement / contact / positive-spread
  / actor diagnostics), `crl/repro_report.py` (curves + GIFs + audit),
  `crl/audit.py` (semantics audit).
- Notebooks: `notebooks/colab_contrastive_rl.ipynb`,
  `notebooks/fetchpush_regime_repro.ipynb`.
