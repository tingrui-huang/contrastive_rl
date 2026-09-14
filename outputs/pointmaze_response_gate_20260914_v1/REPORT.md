# Controlled query-independent response gate for PointMaze ETT

Date: 2026-09-14. Starting commit: `4de08e7ff35ebb9e5e02d97cc41f334b926c1371`. Parent models: the two supervised B checkpoints produced in `54b36620ddd6818ad91679ec5f587ff9f59e2474`.

## Decision

**Do not advance this gate as a validated model change.** It produced a strong, coherent correction in seed 1 but did not replicate in seed 0. In seed 1 it nearly switched off response on already-dead atom draws, preserved response on almost every alive-moving draw, improved the main one-step strata, and moved task-return predictions closer to native returns. In seed 0 it stayed essentially at `h=1`, gave no already-dead energy-score improvement beyond equal-budget continuation, worsened alive and fatal-onset scores, and slightly worsened return calibration.

This is evidence that response conditioning can be a consequential lever in this model, not evidence that it is a stable bottleneck or that this 21-parameter gate solves the problem. Both candidates still overpredict native task return by more than six discounted-return units. The experiment establishes neither actor benefit, reliable death identification, architectural impossibility, nor successful worst-case optimization.

## Controlled intervention and verification

The candidate changes only

`project(anchor + M(s) @ (xq - xb))`

to

`project(anchor + h * M(s) @ (xq - xb))`, where `h = clip(1 + w @ features, 0, 1)`.

The fixed 21 features are eight training-standardized F4 coordinates, raw `xb` (2), raw anchor-minus-current XY (2), the raw atom flag (1), and the original eight gate weights. Scaling was estimated only on original training episodes 0--71. The gate receives no direct `xq`, outcome, reward, death label, privileged state, or future observation. It starts at `w=0`, exactly reproducing each parent B model. Raw atoms remain mixture components, not death labels.

For fixed state, `xb`, anchor, atom, and coupled randomness, `h` is independent of `xq` and is bounded by one. Multiplying the normalized response by `h`, followed by the unchanged non-expansive projection, therefore preserves the original coupled action-Lipschitz bound. Saved checks confirmed exact initialization equivalence, diagonal identity, F4 shift, coupled anchor/atom draws, query-independent `h` under coupling, native sample legality, and maximum observed coupled-action ratio at most 0.858. Both candidate gates received nonzero gate gradients and parameter updates on all 120 continuation steps; their final weight norms were 3.583 and 4.427.

Each parent seed supplied two exact-checkpoint continuations: original control and gated candidate. Both received 120 updates, the same original tuples, separately averaged diagonal/off-diagonal full energy-score losses with lambda one, inherited Adam state and age, equal common-parameter budgets, shared common perturbations, and common model randomness. Only the candidate had the predeclared 21 gate directions. Final iterates were frozen before evaluation; there was no sweep, tuning, or checkpoint selection.

## New complete-episode one-step evaluation

The evaluation set contains 48 new complete 50-step prefix episodes and 9,600 same-snapshot rows: 24 teacher-prefix and 24 blind-prefix episodes, four queries per context. The original exact action-equality diagonal definition, `1e-7` stationarity threshold, inclusive native bounds, clipped floor-cell legality, and all outcomes were preserved. Scores below are the full 64-sample energy-score U-statistic, including every sample-pair term. `P/C/G` means parent/equal-budget control/gated candidate. RMSE is the XY error of the emitted sample mean. Stationarity is the emitted-sample fraction.

The fixed strata contain: diagonal 2,527 rows/48 episodes (32.45% of actual outcomes stationary); all off-diagonal 7,073/48 (29.41% actual stationary); already dead 2,080/16 (100% actual stationary); alive outside goal 931/48 (0% actual stationary); alive moving 4,993/48 (0% actual stationary); and fatal onset 90/40 (0% actual stationary). There are zero alive-stationary rows and zero alive stationary-diagonal/moving-off-diagonal pairs; these remain explicitly absent.

Seed 0, `P/C/G`:

- Diagonal: ES `0.02711 / 0.02705 / 0.02709`; RMSE `0.14178 / 0.13395 / 0.13707`; stationary `27.22% / 27.80% / 27.51%`; candidate mean `h=0.99928`.
- All off-diagonal: ES `0.14692 / 0.14391 / 0.14487`; RMSE `0.22005 / 0.21237 / 0.21644`; stationary `0 / 0 / 0`; `h=0.99923`.
- Already dead: ES `0.11282 / 0.10543 / 0.10470`; RMSE `0.24273 / 0.22631 / 0.23212`; stationary `0 / 0 / 0`; `h=1.00000`.
- Alive outside goal: ES `0.23730 / 0.22443 / 0.23291`; RMSE `0.29656 / 0.27817 / 0.29059`; stationary `0 / 0 / 0`; `h=0.99671`.
- Alive moving: ES `0.16112 / 0.15994 / 0.16160`; RMSE `0.20989 / 0.20628 / 0.20956`; stationary `0 / 0 / 0`; `h=0.99891`.
- Fatal onset: ES `0.22771 / 0.22891 / 0.25652`; RMSE `0.26221 / 0.25750 / 0.31449`; stationary `0 / 0 / 0`; `h=0.98597`.

Seed 1, `P/C/G`:

- Diagonal: ES `0.02743 / 0.02635 / 0.02754`; RMSE `0.13939 / 0.13563 / 0.13628`; stationary `27.24% / 27.48% / 27.47%`; candidate mean `h=0.72902`. The low diagonal gate value has no direct response effect because `xq=xb`.
- All off-diagonal: ES `0.14350 / 0.14378 / 0.11499`; RMSE `0.21723 / 0.21613 / 0.18730`; stationary `0 / 0 / 23.09%`; `h=0.75347`.
- Already dead: ES `0.10183 / 0.09644 / 0.04977`; RMSE `0.23839 / 0.22300 / 0.19155`; stationary `0 / 0 / 78.16%`; `h=0.16867`.
- Alive outside goal: ES `0.22135 / 0.24041 / 0.19391`; RMSE `0.27565 / 0.29865 / 0.24276`; stationary `0 / 0 / 0.54%`; `h=0.99251`.
- Alive moving: ES `0.16086 / 0.16349 / 0.14215`; RMSE `0.20778 / 0.21320 / 0.18550`; stationary `0 / 0 / 0.14%`; `h=0.99708`.
- Fatal onset: ES `0.21686 / 0.26335 / 0.21894`; RMSE `0.24794 / 0.29308 / 0.25164`; stationary `0 / 0 / 0.63%`; `h=0.99078`.

Every emitted sample and every emitted sample mean was native-legal in every displayed model/stratum; illegal-sample and illegal-mean fractions are therefore separately zero. Mean Euclidean errors, displacement, atom fractions, gate quantiles, and all parent/control/candidate metrics are saved in [one_step_metrics.json](one_step_metrics.json).

Paired whole-episode bootstrap results make the seed disagreement explicit. Differences are candidate minus equally trained control; negative ES/RMSE is better.

- Seed 0 already dead: ES `-0.00074`, 95% interval `[-0.00275, 0.00160]`; RMSE `+0.00582 [0.00198, 0.00939]`. Because `h=1` on every already-dead sample, this is not an observed response-suppression gain. Ordinary control continuation itself improved dead ES versus parent by `-0.00739 [-0.00931, -0.00575]`.
- Seed 1 already dead: ES `-0.04667 [-0.05733, -0.03520]`; RMSE `-0.03144 [-0.03600, -0.02793]`. The candidate's mean gate was 0.0094 on atom draws and 1.0000 on non-atoms in this stratum, producing 78.16% stationary samples. Control continuation improved dead ES versus parent by `-0.00539 [-0.00734, -0.00339]` before this additional candidate gain.
- Seed 0 all off-diagonal/alive moving/fatal onset ES differences were respectively `+0.00095 [0.00015, 0.00180]`, `+0.00166 [0.00105, 0.00244]`, and `+0.02762 [0.00992, 0.05702]`: small aggregate damage and material fatal-onset damage.
- Seed 1 corresponding differences were `-0.02879 [-0.03380, -0.02424]`, `-0.02134 [-0.02269, -0.02031]`, and `-0.04440 [-0.05125, -0.03830]`. Its mean gate stayed at 0.9971 on alive-moving and 0.9908 on fatal-onset samples; atom draws were only 0.24% and 1.08% of those samples.
- Diagonal ES was statistically unchanged in seed 0 (`+0.00003 [-0.00081, 0.00100]`) and worsened slightly in seed 1 (`+0.00120 [0.00099, 0.00140]`).

The seed-1 pattern shows that the available observed/anchor/atom features can distinguish many response requirements: atom response was suppressed strongly in already-dead contexts while non-atom and nearly all alive response was retained. Seed 0 learned no such usable partition even though its gate updated on every step. This overlap/separation evidence concerns this response component only. It does not prove that the entire model family can or cannot fit the conditional law, and common diagonal-head/response parameters co-adapted with the candidate gate.

## Frozen-policy task-return calibration

The separate return evaluation used 48 fresh native 50-step episodes under one frozen stochastic actor. Each model generated 64 paths from each matching initial observation, with the same frozen actor and historical nominal checkpoint operating on that model's own generated states. No native death flags or absorbing wrapper were injected. The native discounted return was **2.8427**, episode-bootstrap interval `[1.4990, 4.2931]`; 27.08% of native episodes had positive return.

- Seed 0 parent/control/candidate predicted returns were `9.9370 / 9.5769 / 9.7106`, with biases above native of `7.0943 / 6.7343 / 6.8680`. Candidate minus control increased predicted return by `+0.1337 [0.0771, 0.1896]`; absolute return error changed by `+0.0567 [-0.0091, 0.1245]`, return RMSE worsened by `+0.0995 [0.0453, 0.1543]`, and reward-time MAE changed by `+0.00414 [-0.00061, 0.00884]`. Its rollout gate mean was 0.99970 and only 0.155% of gates were below one.
- Seed 1 parent/control/candidate predicted returns were `10.0045 / 9.8200 / 8.8975`, with biases of `7.1618 / 6.9773 / 6.0548`. Here the lower candidate return is a real calibration improvement: absolute return error changed by `-0.5501 [-0.7774, -0.3016]`, return RMSE by `-0.7079 [-0.8620, -0.5463]`, and reward-time MAE by `-0.03893 [-0.06358, -0.01277]`. Its rollout gate mean was 0.81922, 19.07% of gates were below one, and generated-step stationarity rose to 16.91%.

Thus the local seed-1 correction reaches the rollout law, but it does not replicate. Even the better candidate predicts 8.90 versus 2.84 native return, and its positive-return path fraction is 83.79% versus 27.08% positive native episodes. The main end-to-end calibration problem remains unresolved.

These are rollout-law comparisons, not matched latent conditional ETT trajectories. The nominal checkpoint was trained on its historical population, while the one-step collection deliberately balances teacher and blind prefixes. Model paths also condition the actor and nominal on generated states after divergence. Those population and compounding-state differences limit causal attribution to any one transition component.

## What is established, unresolved, and next

The experiment establishes that one fitted gate can learn a nearly atom-specific response correction and that such a correction can affect long-horizon return predictions while retaining most alive action response. It also establishes severe optimizer/seed instability: the other fitted gate remains clipped near one and provides neither the dead correction nor return benefit. Because the shared diagonal-head and response parameters were trainable and co-adapted, the saved comparison does not isolate how much of seed 1's gain is caused directly by `h` rather than by its altered common-parameter trajectory. Saved emitted arrays cannot reverse projection to recover the missing anchors and answer that counterfactual.

**Single next action:** run one predeclared exact-checkpoint gate ablation in a fresh output directory: replay each final candidate with identical randomness once with learned `h` and once with `h` forced to one, holding its trained 48 common parameters fixed, for both one-step and fixed-policy rollouts. This directly separates the learned multiplier from co-adaptation and tests whether the seed-1 return improvement is truly a response-gate effect. It requires exact model replay; it cannot be reconstructed from saved XY arrays. Treat it as exploratory isolation, not selection evidence. Any resulting model modification still requires a new set of complete confirmation episodes rather than reusing these 48.

## Reproducibility and accounting

The sealed configuration and seeds are in [config.json](config.json) and [PROTOCOL.md](PROTOCOL.md); implementation is in [run.py](run.py) and the isolated paired collector in [collect_evaluation.py](collect_evaluation.py). Final checkpoints, full one-step arrays, paired tuples, native episodes, rollout arrays, bootstrap contrasts, and numerical summaries are all preserved in this directory.

[verify_saved.py](verify_saved.py) independently reconstructs all 480 optimizer updates from saved signed losses, all one-step full U-statistics and group masks, primary episode-bootstrap intervals, native/model rewards and returns, return contrasts, legality, F4 shifts, hashes, and budgets without any model, actor, nominal, or environment call. It passes and writes [verification.json](verification.json).

The completed accounting is **20,657,152 model successors** under the 22,000,000 cap and **14,400 native steps** under the 16,000 cap. The model count includes 307,200 explicitly recorded overhead from two failed float-reduction checks before return arrays were saved; training, one-step sampling, and native collection were not repeated. There were 480 model updates total and zero actor, critic, or nominal-policy updates. Reporting/verification failures and resumptions remain in the historical `failure.json` and `resume*.json` records. Dependency hashes remained unchanged. The experiment itself made no commit or push; repository delivery was performed only after the user's later explicit request.
