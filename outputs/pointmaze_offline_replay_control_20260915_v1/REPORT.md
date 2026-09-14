# Strictly offline PointMaze replay-control report

## Result

R did not improve the down-versus-right endpoint preference relative to F. This is an offline sampling/actor-response diagnostic, not evidence of route entry, native success, or recovered intervention Q.

Both arms began from the eligible original step-150,000 zbase checkpoint and used only `obs` and `act` from the fixed 6,600-episode dataset. F used ordinary replay; R replaced half of every critic batch with 64 recorded-down and 64 recorded-right fork anchors. Each arm then ran the required 1,000 actor updates on byte-identical ordinary batches and Gaussian innovations.

## Coverage and training

The preserved split contains 5,940 continuation-training and 660 held-out episodes. The initial checkpoint had already seen all 6,600, so this is a continuation-held-out comparison, not an independently unseen test. Training support was 1,344 down anchors from 882 episodes and 5,781 right anchors from 5,325 episodes. R drew 25,600 rows per direction; it touched 1,344 distinct down and 5,707 distinct right anchors, with maximum multiplicities 39 and 15. Ordinary replay remained 128/256 rows in every R batch.

All 400 critic updates and all 1,000 actor updates completed in both arms. Actor parameter L2 changes were 1.4307 (F) and 1.4785 (R). Mean raw actor-gradient versus optimizer-applied update norms were 5.6778/0.014951 and 5.6079/0.015018; raw gradients therefore must not be read as applied step sizes.

## Critic

On 8,192 common ordinary held-out rows, final NCE loss was 0.019386 for F and 0.019817 for R; paired R-minus-F across fixed batches was 0.000431 [0.000359, 0.000490]. On 8,192 balanced fork rows, losses were 0.029991 and 0.022681; paired difference -0.007309 [-0.007509, -0.007125].

Across 656 held-out episodes, canonical-task endpoint down-minus-right logits were -0.6029 [-0.6647, -0.5415] initially, -0.4911 [-0.5621, -0.4241] after F, and -0.4973 [-0.5636, -0.4383] after R. The paired R-minus-F change was -0.0062 [-0.0239, 0.0120]. Four-point endpoint neighborhoods gave -0.4095 [-0.4763, -0.3528] (F) and -0.4135 [-0.4726, -0.3625] (R). These are preference probes, not accuracy labels.

At actual stochastic samples from the final canonical-goal actors, the local maximizing-gradient projection along down-minus-right was -0.4606 [-0.5351, -0.3915] for F and -0.3569 [-0.4313, -0.2880] for R. Endpoint rankings and these local derivatives are distinct objects.

## Actor response and retention

Under the canonical goal, the original actor sampled the recorded-down stratum with probability 0.0494 and recorded-right with 0.8460. Final F was 0.0649/0.8448; final R was 0.0624/0.8632. Paired down-probability changes were 0.0154 [0.0139, 0.0171] (F) and 0.0130 [0.0113, 0.0148] (R). These probabilities classify continuous samples only; they do not establish lower-route entry.

The continuous canonical-goal action means `(x,y)` were `(0.8058,-0.1560)` initially, `(0.8202,-0.2504)` for F, and `(0.8467,-0.3125)` for R; component standard deviations were `(0.4646,0.5594)`, `(0.4479,0.5937)`, and `(0.4103,0.5661)`. Thus R's more-negative mean y did not represent a route-preference repair: its x mean and recorded-right probability also increased, and its recorded-down probability remained below F.

With ordinary same-episode future goals at the same fork states, down/right probabilities were 0.0656/0.7900 (F) and 0.0643/0.7955 (R). On 1,024 non-fork held-out ordinary-goal rows, mode-action L2 changes were 0.0815 [0.0739, 0.0893] and 0.0899 [0.0826, 0.0971]; BC-NLL changes were 0.0544 [0.0283, 0.0826] and 0.0705 [0.0388, 0.1045].

For those ordinary future goals at fork states, continuous action means were `(0.7331,-0.1006)` for F and `(0.7368,-0.1300)` for R, with standard deviations `(0.5576,0.5806)` and `(0.5582,0.5750)`. Overall, R bought a large fork-NCE fit improvement while slightly worsening ordinary NCE, moving farther away from the initial non-fork policy, and degrading non-fork BC fit more than F; it did not buy the desired endpoint sign or a more-downward actor classification.

On four common fixed actor batches, final BC/critic gradient cosines were -0.5328 (F) and -0.6669 (R). Weighted BC and critic gradient norms were 5.1425/4.4126 and 4.8672/5.1898.

## What this establishes

The experiment isolates whether changing exposure to already-recorded observable fork/action strata, while retaining ordinary replay and the unchanged all-pairs NCE objective, changes the critic probes and propagates through the original actor objective. It does not remove behavior-policy confounding, add matched counterfactuals, or test current-policy continuations. A failure would therefore mean this particular reweighting was insufficient, not that contrastive learning or the model family is incapable.

The old matched-fork and critic-control reports were hypothesis context only. Their trajectories, labels, critics, and optimizer states were excluded. No model rollout or environment interaction occurred, and no historical artifact was modified.

## Recommended next intervention

Do not increase the replay weight blindly. The next justified intervention is one matched-continuation ETT experiment whose transition model is trained only from the eligible original offline data, whose roots also come from that data, and which uses neither a true-dynamics helper nor hidden fields. It should compare ordinary continuation against balanced current-policy fork continuations through the same required actor stage, then—only if a modification is selected—seek confirmation with newly preregistered complete native episodes. This directly tests the continuation-mismatch hypothesis that static behavior-replay balancing could not resolve.

Numerical arrays, exact hashes, sealed indices, curves, and fixed checkpoints accompany this report.
