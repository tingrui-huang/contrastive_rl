# Phase-balanced critic sampling: partial benefit, no consistent calibration fix

**Increasing early-row exposure improves early seen-query fit in both seeds,
but does not reliably fix calibration on untouched episodes.** One seed has
a clear final early RMSE improvement; the other mainly changes overestimation
into underestimation and transfers error between groups. Difficult early
approach/transit ordering remains underpowered. No ETT or actor update was run.

## Controlled comparison and untouched evaluation

From `23da83f`, both arms and both seeds use the exact same **288 trajectories
and 13,648 sampled positive labels**, paired critic initializations, and
**1,500 Adam steps per critic**. Uniform seed0 exactly reproduces the preceding
critic's parameters and old diagnostic predictions. Nothing changed in the
generator, actor, state-goal nominal sampler, normalization, reward, discount,
architecture, binary NCE loss, 1:31 class weighting, q=(.5,.5), or Q decoding.

The sole comparison is the row sampler. Phases are the first5 transitions,
then t=5 through H//2-1, then the remaining half. The alternative uniformly
chooses a trajectory, a phase, and a row within that phase. Early exposure
rises from about **10.5% to33.3%** of minibatch rows, approximately3.2x. No
reward, future-outcome or hidden-label balancing; labels are never resampled.
The conditional calibration stays `Q_h=(1-.95^h)*31*.5*exp(f_1)`, `Q_0=0`.

After all four checkpoints were frozen, exactly **600 new native episodes**
were collected with the established noisy teacher solely to supply valid F4
contexts. The observable preregistered selector obtained12 distinct episodes
per approach/transit/bypass group. These episodes did not exist during model
or critic training. Raw native archives remain local. The teacher's privileged
information affects its collection actions, but is never a critic input or
training label; selection reads only observable histories.

Evaluation queries are phase starts: t=0,5,H//2 relative to each root. Each
uses the exact remaining horizon and64 independent continuations from the
same frozen generator, with the recorded first action held fixed. All critics
share the queries and targets. Seen training queries use actual training
state/action rows; fresh queries use a new model path per new native root.
The old evaluation set is only an explicitly inspected diagnostic archive.

## Calibration: seen and fresh results differ by seed

Values and errors use normalized occupancy `.05*sum(.95^t*reward)`.
Paired intervals use2000 episode-cluster bootstrap replicates; overall metrics
stratify by root group. Repeats are averaged first.

- **Seen early queries, seed0:** RMSE **.07498 -> .05793**, change **-.01705**,
  95% CI **[-.03296,-.00117]**.
- **Seen early queries, seed1:** RMSE **.05687 -> .04068**, change **-.01619**,
  CI **[-.02512,-.00618]**.
- **Fresh early queries, seed0:** RMSE **.07166 -> .07059**, change **-.00107**,
  CI **[-.01854,+.01525]**. Bias changes **+.04691 -> -.05727**; its absolute
  magnitude worsens. Approach RMSE improves **.0810 -> .0267**, while bypass
  worsens **.0344 -> .0848**. There is no reliable overall improvement.
- **Fresh early queries, seed1:** RMSE **.05750 -> .03517**, change **-.02233**,
  CI **[-.02993,-.01398]**. All three groups' early RMSE improves. Bias moves
  **+.00918 -> +.01361**, a small worsening despite the RMSE improvement.

The strict preregistered sampling-bottleneck criterion is unmet in both seeds:
it required significant final early RMSE improvement, reduced absolute bias
and corresponding seen-query improvement, consistently across seeds. This
does **not** erase seed1's clear RMSE benefit; it limits the robustness claim.

There are phase tradeoffs. Seed0 final middle RMSE rises **.09657 -> .10699**;
seed1 final late RMSE rises **.02573 -> .03813**. Those trigger the registered
later-harm flags. Seed0 late and seed1 middle instead improve. The alternative
does not uniformly improve the continuation estimator.

No early prediction exceeds the legal occupancy range. Later violations
remain: maximum excess for seed0 uniform/balanced is **.16633/.02073** in the
middle and **.06297/.07285** late; seed1 is **.03042/.04587** in the middle and
**.05075/.08311** late. Values were not clipped. Mean/maximum target MC SE on
fresh queries is **.01924/.03170 early**, **.01325/.04127 middle**, and
**.00168/.02946 late**. Raw RMSE includes this target noise. Full phase/group
RMSE, bias, range fractions, MC SE and intervals are in the calibration JSONs.

## Ordering is a separate question

Only within-group, same-remaining-horizon episode pairs are eligible. A label
must exceed both .01 and2.58 pooled MC SE, and agree in the two32-repeat halves.
Scorer predictions never select pairs. Minimum support is10 informative pairs
spanning8 episodes. Bootstrap pair weights cluster their endpoint episodes.

- Fresh **early approach:** **0 informative /66** same-h pairs, all noise-
  unresolved. No ranking conclusion.
- Fresh **early transit:** **1/25**, with24 unresolved and only2 contributing
  episodes. Its observed100% accuracy is explicitly insufficient evidence.
- Fresh **early bypass:** **29/66**, spanning12 episodes;37 unresolved.
  Seed0 accuracy is **.724 -> .931**, but the paired accuracy-change CI is
  **[-.223,+.647]**. Seed1 is **.931 -> .931**, CI **[-.167,+.125]**. Neither
  establishes a ranking advantage for phase balancing.
- Middle approach and bypass, and late approach, supply more informative
  pairs; they do not show a consistent alternative advantage. Late approach
  also has36 equal-return pairs. Every cell's equal/unresolved/informative
  counts and clustered accuracy intervals are retained, including inadequate
  cells. These noise screens are approximate, not simultaneous guarantees.

## Interpretation and stopping

The comparison supports **sensitivity to early-row allocation**, including
improved seen-query early RMSE in both seeds and fresh-episode improvement in
seed1. It does not establish insufficient early sampling as the sole cause
or show that this reweighting fixes it. Shared-network calibration shifts
across groups and phases remain substantial and seed dependent.

This is not clean evidence of a generalization-only problem: uniform early
RMSE is similar on seen and fresh queries in each seed, and balanced seed1
is better on fresh than seen queries. Seed0 has a seen-to-fresh gap, but it
is not consistent across seeds. Training empirical early NCE is almost
unchanged for seed0 (.134162 -> .134174) and slightly lower for seed1
(.134092 -> .133876); those small loss changes coexist with larger Q changes.
Finite-label fitting, limited critic optimization and function approximation
remain plausible contributors; this single controlled comparison cannot
separate them. The targets are from the same visible-state generator, so
native hidden-state uncertainty is not itself an explanation of this
model-internal calibration failure.

**Stop here.** Do not infer useful early hazard ordering, verified death,
native task values, causal identification, or a globally optimal worst case.
Generator parameters never changed, so diagonal degradation is zero and the
existing samplewise Euclidean action-Lipschitz construction is unchanged.
No reward/loss change, ETT optimization, actor training, extra family or
budget extension occurred.

Used **538,221/650,000 model outputs**, **30,000/30,000 native steps**, and
**6,000 optimizer steps**. All **33 regression/component tests** passed. A
read-only post-run audit replayed labels, exposures, checkpoint hashes,
selection, queries, predictions and metrics without adding outcomes.

## Reproduction and artifacts

`REPORT.md` contains full metric listings and test commands. From the PointMaze
worktree with the frozen local inputs in `preregistration.json`:

```powershell
python -m ett.pointmaze_phase_sampling prepare --out artifacts/pointmaze_region_pilot/phase_sampling_s01_v1
python -m ett.pointmaze_phase_sampling run --out artifacts/pointmaze_region_pilot/phase_sampling_s01_v1
python -m ett.check_pointmaze_phase_sampling --out artifacts/pointmaze_region_pilot/phase_sampling_s01_v1
```

Use a fresh output directory for reproduction. Shared paths/labels, phase
exposures, final selected contexts, predictions,64-repeat return arrays,
paired differences and ordering intervals are saved. Local checkpoint/raw
input files are ignored. `PROTOCOL.md` and hashes predate outcomes; the
post-run checker is explicitly an integrity audit, not a retrospective gate.
No commit or push has been performed.
