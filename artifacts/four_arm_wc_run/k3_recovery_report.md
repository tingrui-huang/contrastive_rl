# Arm A: recovery of the historical "k=3" propensity — **NOT RECOVERABLE**

Per the task instruction:

> If the exact historical k=3 method cannot be uniquely recovered: STOP Arm A
> preparation and report the ambiguity. Do not invent a new k=3 estimator.

**Arm A was not prepared. No launch script was generated. No k=3 estimator was
invented.**

---

## Search performed

Exhaustive, over the working tree *and* all git history/branches:

| where | how |
|---|---|
| all `.py` / `.sh` / `.md` / `.json` in the tree | `grep -rn "k=3\|k = 3\|K = 3\|K=3\|k3\|knn\|kNN\|k_neighbors\|n_neighbors\|neighbor"` |
| git commit messages, all branches | `git log --all --grep=` for `k=3`, `k3`, `knn`, `propensity` (case-insensitive) |
| git content history (pickaxe), all branches | `git log --all -S"k=3"`, `-S"k3"` over `*.py`, `*.md` |
| filenames | `find . -iname "*k3*" -o -iname "*knn*"` |
| artifact directories | `artifacts/` scan for `k3` / `knn` / `propens` / `agree` |
| the user's suggested terms | `propensity`, `neighborhood`, `agreement`, `h=0.3`, `calibration`, `positive rate` |

## What exists

### 1. The `propensity_net/` project — this is the **h = 0.3** work, not k=3

A genuinely *calibrated* neighborhood propensity net `w(s, a_q)`:

* `propensity_net/make_pairs.py` — `z = 1 iff ||a_data - a_q||_2 <= h`, **h = 0.3**
* `propensity_net/train_w.py` — supervised net
* `propensity_net/calibrate.py` — MC ground truth = *"among the **k = 500**
  nearest dataset transitions to s (Euclidean in the full 4-dim obs space), the
  fraction with `||a_i - a_q|| <= h = 0.3`"*

Gates (all passed): positive rate **0.1334**, val **AUC 0.9145**, calibration
**MAD 0.0166**, mean bias **-0.0037**, not one-sided.

**Why it cannot be Arm A:**

1. **It is the h=0.3 method.** The task explicitly warns: *"Do not conflate k=3
   with the separate h=0.3 result unless the code/artifacts prove they are the
   same method."* Nothing proves that; the code proves the opposite — the only
   `k` in it is **k = 500**, the MC neighbor count for *validating* the net, not
   a model parameter.
2. **Wrong environment.** It is trained on `datasets/swamp_windy_teacher_s0.npz`
   — the 2-D swamp PointMaze, `obs` shape `(6000, 51, 4)`, `act` shape
   `(6000, 51, 2)`. The rockfall AntMaze is 29-D state / 8-D action. The
   checkpoint `w_net.pt` cannot even be applied to this experiment's inputs.
3. Applying it here would require **retraining on the rockfall dataset and
   re-choosing the bandwidth** — i.e. inventing a new estimator, explicitly
   forbidden.

### 2. The only literal `k=3` in the repository is **not a propensity at all**

Three occurrences, all the same thing:

* `scripts/qualify_two_route_gate.py:214` — `def gate4_u_to_next_state(env, rng, per_cell=40, k=3)`
* `scripts/qualify_two_route_swamp.py:380` — `def gate5_u_to_next_state(env, rng, per_cell=30, k=3)`
* `scripts/qualify_two_route_swamp_matched.py:244` — `def g9_u_to_next_state(env, rng, per_cell=40, k=3)`

Reading the bodies: `k` is the **number of environment steps** taken from a
cloned state (`for _ in range(k): env.step(action)`), used in a structural
qualification gate that measures whether the hidden variable U changes `S'`. It
is reported as `k_steps=3`. It is:

* not a probability, not a propensity, not a kNN count;
* not a learned model — pure env stepping;
* on the 2-D two-route PointMaze, not the rockfall AntMaze;
* an **environment-qualification** gate, not part of any RL objective.

### 3. Other near-misses, all excluded

| candidate | why not |
|---|---|
| archived Manski `action_bins(n_sectors=8, zero_thresh=0.15)` | 8 sectors + a stay bin; no 3, and it is a 2-D grid propensity table |
| `propensity/agreement.py` `D_psi` | that is Arm B's surrogate; no `k` parameter at all — the module states *"no action-neighborhood bandwidth h is used anywhere"* |
| `crl/rockfall_ant.py:140` `k=35` | rock-trigger tuning comment, unrelated |
| `scripts/ant_critic_local_ranking.py` kNN support | a critic diagnostic over a reconstructed behavior buffer, not a propensity; no k=3 |

---

## Conclusion

There is **no historical k=3 propensity implementation** in this repository, on
any branch, in any artifact, or anywhere in git history. The nearest real
propensity work is the `propensity_net` **h=0.3 / k=500-validation** project on
a *different environment and a different observation/action space*.

Recovering Arm A therefore requires information the repository does not contain.

## What is needed from the user

One of:

1. **Point to the k=3 artifact** — a path, commit SHA, branch, or external
   location. If it lives outside this repo (another machine, a Colab drive), it
   can be imported and used verbatim.
2. **State the definition** — what `k=3` is a count of (nearest neighbors?
   action bins? env steps?), its inputs, its output range, and whether the
   output is `rho`, `p_wc`, a support measure, or something else.
3. **Confirm k=3 *is* the `propensity_net` h=0.3 method** — in which case the
   blocker becomes concrete and different: it must be retrained on the 29-D/8-D
   rockfall dataset, which is a new model and needs its own pre-registration
   (bandwidth choice included).
4. **Drop Arm A** and run the three prepared arms (B, C, D).

Arms **B**, **C** and **D** are fully prepared, smoke-tested and launch-ready
regardless of how Arm A is resolved.
