# PointMaze frozen-critic continuation-score diagnostic

## Decision

The frozen raw contrastive continuation score separates obvious failure histories globally, but it is systematically inverted on the closest authentic dead/alive matches and is not suitable as the pessimistic term in off-diagonal ETT training.

The tested quantity is a surrogate ranking score. It is not a calibrated value, success probability, death probability, or worst-case Q. No ETT, nominal policy, actor, or critic parameter was trained or modified.

## Evaluated score and checkpoint

The primary checkpoint is `alpha0p3_seed0` at step 150000 (failure-negative alpha `0.3`, learner seed `0`). Its SHA-256 is `bf19d9e783ad689e1771d17ee5e7a5725bfb33a142eaa6b2458ae8aea96cda9c`.

The diagnostic evaluates

$$C(s,g_{task}) = \mathbb{E}_{a \sim \pi_{frozen}(\cdot|s,g_{task})}[f_{frozen}(s,a,g_{task})].$$

`f` is the raw aligned contrastive logit. No sigmoid, exponential, temperature, or value conversion is applied. The training actor critic term minimizes `-diag(f)`; these runs have `entropy_coefficient=0`, `bc_coef=0.5`, and one critic head. The reported expectation uses 128 nested samples; 8 samples and the actor mode are stability checks.

Contrastive positives are future achieved F4 goals from the same episode. Ordinary negatives are off-diagonal goals from other sampled episodes. Failure-negative training places established frozen bank states only in the GOAL slot. Actor training uses hindsight goals (including the `random_goals=0.5` batch roll); this diagnostic keeps the commanded task goal fixed, as deployment does.

## Evaluation population and timing

Evaluation uses 23,950 p=0.30 expert-positive validation transitions from 479 source episodes. The split is useful for comparison with the nominal/ETT diagnostics, but it is in-sample for these critics: their offline CRL training used all 6,600 source episodes.

Death is reconstructed with the verified timing: the fatal-entry state is alive before its action; death age 0 is the first observation after fatal contact; the F4 history becomes fully established at age 3. Hidden swamp bits and death fields are used only for grouping.

## Primary results

Using negative continuation score as an uncalibrated death ranking gives ROC-AUC `0.98936` and average precision `0.96631` at death prevalence `0.07791`. The episode-bootstrap intervals are ROC-AUC `0.98750` to `0.99103` and AP `0.95450` to `0.97427`.

Mean continuation score is `-5.0774` for alive moving states and `-5.1831` for alive waiting/near-stationary states. Death-age means are `-5.4628`, `-6.7966`, `-8.1610`, and `-11.2922` for ages 0, 1, 2, and at least 3. The early-age pattern must be read with the prior F4 observability result: recently dead histories still resemble moving histories.

There are 1,866 cross-episode matched dead/alive pairs constrained to the same maze region and distance-to-goal band. Dead scores are lower in `86.98%` of pairs; mean dead-minus-alive score is `-4.0538`. Median raw full-history F4 distance is `0.9351`. Matching narrows observed geometry and does not remove confounding.

The tighter raw-F4 subsets contain 284 pairs at distance at most 0.5 and 976 pairs at distance at most 1.0; their dead-below-alive rates are `29.93%` and `75.10%`. These radius checks expose how the conclusion changes as observed-history similarity is tightened. A rate below 50% on the adequately sized 0.5 subset is systematic inversion and is the decisive failed criterion.

Across all six frozen checkpoints, small-budget death ROC-AUC ranges from `0.98638` to `0.99108`, AP from `0.94922` to `0.97230`, and matched dead-below-alive frequency from `82.48%` to `88.91%`. On the tight F4-distance-at-most-0.5 subset, every checkpoint is inverted: the range is only `15.85%` to `30.63%`. The alpha-specific seed means are stored in `metrics.json`. Failure negative training does not consistently improve this STATE-slot continuation ranking over alpha 0, so the observed signal cannot be attributed uniquely to the failure bank.

Alive safe-route states have mean score `-5.6751`. The report records their frequency below the established-death median and below the alive-moving lower decile, so a sparse but legitimate detour is not silently treated as failure.

## State slot versus goal slot

For authentic alive-moving anchors with recorded actions, the mean failure-bank GOAL score is below the commanded task-goal score for `100.00%` of anchors. Supplying the same bank observations in the STATE slot and pursuing the commanded goal gives mean continuation score `-11.6811`. These are different queries; the first is directly trained as a negative, while the second is the proposed ETT objective and must be validated empirically.

## Stability and gradients

The 8-versus-128 sample scores have Spearman correlation `0.984663`, Pearson correlation `0.996285`, and mean absolute difference `0.11504`. Matched-pair score direction agrees in `99.25%` of cases. Results for alpha 0, 0.1, and 0.3 with seeds 0 and 1 are in `metrics.json`. Every checkpoint has `twin_q=False`, so no head reduction comparison exists.

Direct critic-state gradients and total gradients through the frozen actor are finite (`True` / `True`). The actor path changes the state gradient in `100.00%` of checked authentic states. Frozen parameters do not detach state inputs. This does not validate rankings or gradients on arbitrary generated OOD states.

## Concrete authentic states

### Highest-scoring failure states

- Episode 1811, t=9, death age 0, region `swamp_corridor`, score `-4.9596`; frames newest-first `[(5.4328,3.5544), (4.4738,3.4350), (4.0160,3.3580), (3.0170,3.5405)]`; commanded goal `(8.5000, 3.5000)` tiled over four frames.
- Episode 3878, t=8, death age 0, region `swamp_corridor`, score `-5.0201`; frames newest-first `[(5.9014,3.5157), (5.0871,3.4010), (4.0871,3.2102), (3.1345,3.2901)]`; commanded goal `(8.5000, 3.5000)` tiled over four frames.
- Episode 1236, t=6, death age 0, region `swamp_corridor`, score `-5.0626`; frames newest-first `[(5.3010,3.4877), (4.4504,3.3736), (4.4410,3.3861), (3.4410,3.4845)]`; commanded goal `(8.5000, 3.5000)` tiled over four frames.

### Lowest-scoring alive states

- Episode 4760, t=8, death age -1, region `swamp_corridor`, score `-15.9146`; frames newest-first `[(3.4405,3.9999), (3.4476,3.9991), (3.4533,3.9997), (3.4532,3.9880)]`; commanded goal `(8.5000, 3.5000)` tiled over four frames.
- Episode 1785, t=7, death age -1, region `swamp_corridor`, score `-13.9813`; frames newest-first `[(4.3017,3.0891), (4.3007,3.1050), (4.2836,3.1133), (4.2846,3.1181)]`; commanded goal `(8.5000, 3.5000)` tiled over four frames.
- Episode 3574, t=8, death age -1, region `swamp_corridor`, score `-11.3694`; frames newest-first `[(4.8689,3.9762), (4.8602,3.9850), (5.0229,3.7519), (4.1857,3.2257)]`; commanded goal `(8.5000, 3.5000)` tiled over four frames.

## Limitations

- The critic evaluation is in-sample at the episode level.
- Failure labels and matching diagnose ranking; they do not calibrate the raw logit.
- Safe-route support is sparse, and matching cannot remove hidden swamp or behavior-policy confounding.
- Authentic offline states do not test arbitrary ETT-generated states.
- This diagnostic neither recovers nor estimates worst-case Q.

## Reproduction

```bash
python scripts/diagnose_f4_critic_continuation.py \
  --dataset artifacts/f4_p30_server_30076/results/datasets/swamp_windy_f4_merged_s0.npz \
  --runs-root artifacts/f4_p30_server_30076/results/runs/f4_p30_sweep/p30_a0_a01_a03_s0_s1 \
  --failure-bank artifacts/f4_p30_server_30076/results/artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz \
  --out-dir artifacts/critic_continuation/f4_p30_alpha_s01 \
  --small-samples 8 --large-samples 128 \
  --gradient-samples 8 --bootstrap-replicates 1000 --seed 2301
```

Do not start off-diagonal training with this score. First resolve why the goal-slot failure-negative construction does not produce reliable state-slot ordering on the closest authentic histories, then test that resolution on critic-heldout episodes.

This task introduced no Lipschitz penalty, bank-distance objective, death classifier, or new bank.
