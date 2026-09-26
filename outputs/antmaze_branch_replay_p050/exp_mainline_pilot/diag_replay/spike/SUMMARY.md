# The training collapses are critic runaways: one update at a time, with a frozen-critic control

User's lead (2026-09-19): the training logs show isolated spikes of the
critic loss / mean logit at fixed update counts, and the walking probe falls
to zero right after them.  Verified in every history (`train_manifest.json`
of the pilot, the round-2 arms, the variants, the replays):

| run | update | critic loss | mean positive logit | actor q-term | actor scale median | saturation |
|---|---:|---:|---:|---:|---:|---:|
| CF s0: original, replay, bc0 variant (three runs) | 4,500 | 0.133 | -136 | 267 | 3.0 | 0.78 |
| CF s1: original, replay, CFold2, bc0 (four runs) | 2,500 | 0.120 | -122 | 241 | 1.6 | 0.51 |
| CF s2: original / replay | 6,500 | 0.362 / 0.089 | -370 / -91 | -- / 178 | 0.63 / 1.1 | 0.32 / 0.48 |
| O s1: original, round-2 O (two runs) | 8,000 | 0.210 | -214 | -- | 1.7 | 0.72 |
| O s2 (round 2), anchor variant O s2 | 6,500 | 0.119 | -122 | | | |

The values are bit-identical across runs that share the critic stream and
differ only in the actor (bc 0 vs 0.05) or in the initialisation of the
actor (CFold2): the event is set by the critic batch sequence, not by the
actor.  O and CF of the same seed share the anchors and the actor batches
and spike at different updates: the goal rows are involved.  0.133 = 136 /
1024: in the spiking batch all B x B logits sit at one large negative value
(the positives' BCE / 1024), i.e. the critic's output collapsed globally.
The learner has no representation normalisation and no layer norm
(`repr_norm False`, `use_layer_norm False`), Adam 3e-4 with eps 1e-7.

## Reproduction (`scripts/diag_v6_spike_trace.py trace`)

From the replay checkpoints before each seed's first spike (`upd_4000.pkl`,
`upd_2000.pkl`, `upd_6000.pkl`), the pilot's own `update_step` applied one
update at a time with the streams' own batch sequence, 1,000 updates, per
update: the training-batch metrics, the critic's and the actor's parameter
update norms (the real Adam steps), the critic on a FIXED batch drawn once
before the window (positive / negative logits, its loss, |phi|, |psi|), the
critic at the 192 logged-shortcut reset rows, the actor on a fixed actor
batch (scale, |loc|, saturation, E_f, the critic-term action gradient), the
content of the critic batch; a checkpoint every 50 updates probed by the
route probe and the entrance-walking probe of `diag_v6_training_replay.py`.
Seeds 0 and 1 reproduce the spike at 4,407 / 2,381; seed 2 does not (two
attempts from `upd_6000`, per-update loop and the original scan-of-4 loop:
critic loss 0.0068-0.0077 throughout) -- its event is numerically
knife-edge (its magnitude already differed between the original and the
replay), so the seed-2 window is left out.

## What happens, in order (seeds 0 and 1)

| | seed 0 | seed 1 |
|---|---|---|
| critic gradient norm before (median / p95) | 0.017 / 0.055 | 0.031 / 0.087 |
| first impulse | 4,407: **0.916**, parameter update 0.217 (usual 0.016) | 2,375: 0.398; **2,381: 2.53**, update 0.232 |
| fixed-batch logits (pos / neg move together) | -6.3 -> -13.5 (4,407) -> -19.2 -> -21 | -5.8 -> -21 (2,381) -> -36 -> -57 (2,385) |
| the ~10 updates after | gradient 0.07-0.13, update still 0.17-0.25 | gradient 0.15-0.19, update 0.15-0.19 |
| representation norms on the fixed batch | \|phi\| 3.4 -> 4.5 (4,413) -> 16 (4,437); \|psi\| 5.3 -> 7.9 -> 18 | \|phi\| 4.7 -> 5.4 (2,391) -> 21 (2,431); \|psi\| 7.3 -> 11 -> 33 |
| second impulse / runaway | 4,415: gradient 18, fixed -58; 4,428: gradient 258, loss 2.5; minimum -929 at 4,438 | 2,396: gradient 87, loss 0.64; 2,418: gradient 600; minimum -2,422 at 2,431 |
| actor q-term | 6.6 -> 14.7 (4,408) -> 110 (4,417) -> 928 (4,437) | 6.1 -> 21.6 (2,382) -> 55 (2,387) -> 2,455 (2,431) |
| actor parameter update norm | 0.053 (usual) -> 0.17 (4,422) -> 0.44 (4,437) | 0.052 -> 0.10 (2,396) -> 0.38 (2,411) |
| actor scale median / saturation | 0.063 until 4,417; 0.108 / 0.04 (4,422); 0.34 / 0.08 (4,427); 0.95 / 0.44 (4,437); peak 4.3 (4,473) | 0.063 until 2,396; 0.109 (2,401); 0.95 / 0.26 (2,411); 1.6 / 0.59 (2,431); peak 4.7 (2,770) |
| entrance-walking probe (n = 45 / 60) | 0.47-0.67 through 4,400; **0.00 from 4,450 to 5,000** | 0.65 (2,000), 0.22-0.43 (2,100-2,350); 0.03 (2,400); **0.00 from 2,450 to 3,000** |
| reset policy reaches hazard 1 within 100 steps | 0.37-0.44 -> 0.00 from 4,450 (the actor no longer walks) | 0.51 -> 0.00 from 2,450 |
| critic recovery (fixed logits above -10 for good) | 4,807; at 5,000: -6.7 with \|phi\| 8.3, \|psi\| 10.5 (start 3.4, 5.3) | 2,945; at 3,000: -8.0 with \|phi\| 11.3, \|psi\| 12.9 (start 4.7, 7.3) |
| actor at the window end | scale 0.36 (start 0.064), saturation 0.03, walking 0.00 | scale 1.54, saturation 0.38, walking 0.00 |

Order: one large critic gradient on an ordinary batch -> Adam keeps taking
x10-15 steps in the same direction for 10+ updates although the gradient is
back to x3-5 (momentum, lagging second moment) -> all logits shift down
together (fixed batch, reset rows, training batch) and the representation
norms grow -> further impulses and a runaway to logits of -900 to -2,400 ->
the actor's critic term explodes and, 12-20 updates after the first
impulse, the actor's scale and saturation blow up -> walking is gone.  The
critic returns to normal logits within ~400-560 updates but with
representation norms x2; the actor's distribution is still inflated at the
window end (scale 0.36 / 1.54 vs 0.064) and the walking probe is still zero.

## The triggering batch and the per-row gradient (`rows_4407.json`, `rows_2381.json`)

The batch content is ordinary: input maxima, goal locations (goal area 30 %
/ 28 %, hazard corridor 33 % / 34 %), anchor times, future offsets and
branch outcomes are all within the window's range (|z| < 3.1).  Per row,
the gradient of the loss with respect to the representations is x12-20 the
previous updates in total (dL/dphi 1.0e-3 vs 6-9e-5; 2.5e-3 vs 1.2e-4) and
is spread over many rows (the five largest rows carry 35-59 % of its
squared norm), with representation norms no larger than before (|phi| max
10 / 15, |psi| max 22).  The largest rows are anchors in the goal area
(x 22-25) paired with goal-area goals 1-77 steps ahead -- the end of
successful trajectories, where many (s, a) - g pairs of the batch are
near-duplicates that the binary-NCE loss must still separate.  So the
impulse is a batch-composition effect at the goal area, not an outlier row
or an input-scale problem; why it is x30 rather than x2 (the critic's
conditioning at that moment) is not pinned down.

## Frozen-critic control (same checkpoint, same batches, same actor Adam state; the critic restored after every update)

| | seed 0 | seed 1 |
|---|---|---|
| critic gradient norm max over the window | 0.039 | 0.039 |
| fixed-batch positive logit min | -6.3 | -6.5 |
| actor scale median (max / end) | 0.103 / 0.073 | 0.087 / 0.064 |
| actor saturation max | 0.041 | 0.045 |
| entrance-walking probe | 0.36-0.69 throughout (0.56 at 5,000) | 0.48-0.73 throughout (0.58 at 3,000) |
| turned north | 0.09-0.21 throughout | 0.00 |

With the critic held at the window's start the actor does not collapse:
the whole walking loss of the window -- including seed 1's pre-spike drift
from 0.65 to 0.2-0.4, which is also absent with the frozen critic -- is
produced by the critic updates.

## Reading, against the decision rule set before the run

* "Does the score on fixed data also jump?"  Yes, at the same update as the
  training batch, by the same amount: a global change of the critic, not a
  batch-specific score.
* "Which changes first: the critic's action gradient, the representation
  size, the Adam step, the actor?"  The critic's parameter step (one
  gradient impulse, then Adam momentum), then the representation norms,
  then the actor -- by 12-20 updates.
* "Does the actor still collapse with the critic frozen?"  No.  The
  decision rule's first branch applies: the critic update and the signal
  it passes to the actor are what to examine.

The instability is a property of the shared learner (binary NCE, no
normalisation, Adam 3e-4 / eps 1e-7, batch 1024) under the shared data law
(gamma 0.999 relabeling that puts ~30 % of goals at the goal area): the O
arm has it too (s1 at 8,000; round-2 O s2 at 6,500), the CF arm has it in
every seed within the first 7k updates (4.5k / 2.5k / 6.5k).  The pilot's
30k evaluations therefore compare actors that went through different
numbers of such collapses at different times; the route and walking
fluctuations of the replay (diag_replay/SUMMARY.md) include the aftermath
of these events.  Nothing here changes the method; a remedy would be a
training-stability safeguard applied to both arms identically (the user's
call: e.g. gradient-norm clipping on the critic, a representation or layer
norm, a smaller critic step) and is not run.

Files: `CF_s{0,1}/{normal,frozen_critic}/{trace,probe}.json`,
`CF_s{0,1}/normal/rows_*.json`, `CF_s{0,1}/REPORT.md`; seed 2's non-
reproducing window in `CF_s2/normal/trace.json` and `CF_s2/scan_check/`.
Checkpoints stay on the nodes.
