# Pretrain on the recorded data -> continue NCE on the branch replay: summary

Sealed `exp_pretrain_branch/manifest.json` (2026-09-18 08:37 node time,
before any run; git sidecar: 355d273, 5 dirty tracked files).  Nodes: T
critics and actors on an RTX 4090 (node 30021); C1 and C2 actors on an RTX
4090 laptop (node 30043); C1 critics are the existing `critics_armR` (4090
laptop).  Everything else identical across arms: the R replay with its
anchor weights, the recorded p050 dataset for the BC rows, the pure-BC
walker as the actor init, frozen critic, the original actor loss with bc
0.05, 30k actor updates, 300 natural draws (seed 909) with the deployment
(mean) policy, seeds 0 / 1 / 2.  Full tables: `REPORT.md`; critic tables
`diag_r1_abc/REPORT_armT.md`, `REPORT_pre100k.md`, paired bootstraps
`diag_r1_abc/paired_boot_pretrain_{B,C}.json`.

## Arms

| arm | critic | actor |
|---|---|---|
| T (treatment) | vanilla CRL 100k on the full recorded data (gamma 0.999) -> the same NCE continued 30k on the R replay (counterfactual futures only, no ordering constraint, anchor weights) | pure-BC init, frozen T critic |
| C1 (existing recipe) | random init -> 30k NCE on the R replay | pure-BC init, frozen C1 critic |
| C2 (pretrain only) | vanilla CRL 100k on the recorded data | pure-BC init, frozen C2 critic |

Cost (wall seconds per seed): pretraining 808-815 (4090, three in
parallel); T branch stage 357-361; T actor 488-491; C1 critic 627-639 and
actor 557-651 (4090 laptop); C2 actor 552-651.  The added schedule costs
one 100k vanilla run per seed on top of the existing recipe (about 2.3x
the branch critic's own 30k in wall time on the same GPU).

## Critic level (A / B / C diagnostic, region-integrated readout, agreement among decided pairs; seeds 0 / 1 / 2)

| | A (230 pairs, 131 eps) | B (222 pairs, 78 eps) | C (758 pairs, 45 eps) |
|---|---|---|---|
| T | 0.72 / 0.70 / 0.74; gain +.042 / +.036 / +.042 | 0.53 / 0.50 / 0.52; gain -.004 / -.002 / +.004 | 0.50 / 0.49 / 0.51; ~0 |
| C1 | 0.67 / 0.67 / 0.70; +.029 / +.030 / +.036 | 0.58 / 0.53 / 0.55; +.010 / .000 / +.009 | 0.48 / 0.50 / 0.52; ~0 |
| C2 | 0.44 / 0.42 / 0.46; -.012 / -.009 / -.001 | 0.45 / 0.44 / 0.45; -.004 / -.008 / -.003 | 0.50 / 0.49 / 0.50; ~0 |

Paired episode bootstrap (seed-averaged): B: T 0.518 +- 0.028, C1 0.554
+- 0.028, C2 0.449 +- 0.025; T - C1 = -0.036 +- 0.027 (z -1.4, every seed
negative: -0.054 / -0.027 / -0.027); T - C2 = +0.069 +- 0.037 (z 1.8).  C:
T 0.499, C1 0.500, C2 0.498; all differences within 0.02 (z 0.0).

Reading.  Pretraining changed nothing at B or C: the T critics rank
unseen torques at familiar states no better than the random-init recipe
(if anything slightly worse, consistently across seeds but not beyond
two s.e.) and are at chance on held-out episodes.  At A they are a little
better than C1 (0.72 vs 0.68), i.e. the branch stage still fits its keys.
The pretrain-only critic (C2) is BELOW chance at A and B (0.42-0.46):
its ordering of counterfactual outcomes at the training keys is
reversed, which is what an expert-confounded value looks like on this
benchmark (recorded futures are the sighted teacher's), and the branch
stage undoes that (T at A 0.72) without adding transfer.  The critic-level
decision rule (improvement over C1 on B or C by 2 paired s.e. with
consistent seed direction and positive pick gain) is not met.

## Policy level (300 natural draws, mean policy)

| arm | seed | success | failure | timeout | detour | discounted |
|---|---|---:|---:|---:|---:|---:|
| T | 0 / 1 / 2 | 0.213 / 0.233 / 0.103 | 0.560 / 0.517 / 0.250 | 0.227 / 0.250 / 0.647 | 0.000 / 0.007 / 0.000 | 0.018 / 0.013 / 0.009 |
| C1 | 0 / 1 / 2 | 0.303 / 0.000 / 0.000 | 0.507 / 0 / 0 | 0.190 / 1.000 / 1.000 | 0.007 / 0 / 0 | 0.021 / 0 / 0 |
| C2 | 0 / 1 / 2 | 0.287 / 0.007 / 0.000 | 0.003 / 0 / 0 | 0.710 / 0.993 / 1.000 | 0 / 0 / 0 | 0.006 / 0 / 0 |
| reference | pure-BC init | 0.253 | 0.743 | 0.003 | 0.000 | -- |

Means over seeds: success T 0.183 (s.e. 0.040), C1 0.101 (0.101), C2
0.098 (0.094); detour 0.002 / 0.002 / 0.000.  Pre-registered decision: T -
C1 = +0.082 with pooled seed s.e. 0.109 and T - C2 = +0.086 with 0.103 --
neither beyond 2 s.e.; T's detour rate exceeds 0.05 on 0 of 3 seeds.  Not
met.

Reading.  Starting the actor from a walker that succeeds 0.253 and never
detours, the critic term at bc 0.05 made every arm worse or broke it: T
keeps walking on all three seeds but succeeds less than its init (0.10-
0.23) and detours 0-0.007; C1 and C2 stop walking entirely on two of
three seeds (timeout 1.0) and the surviving C1 seed reproduces the init
(0.303, detour 0.007); the C2 survivor is a slow shortcut (success 0.287
with mouth-1 median step 206 and 0.96 of active-zone shortcuts arriving
after the burst -- the lateness loophole).  No arm produced detours.
The round-1 off-manifold behaviour of the actor under a frozen critic at
bc 0.05 recurs from a BC init: the actor needs no rebuilding of walking
for the critic term to push it off the data, and the pretrained critic
does not change that in kind (it only leaves the walker less destroyed).

## What this run establishes and what it does not

- Establishes: with the loss, actor objective, BC weight and budgets
  fixed, inserting a 100k vanilla pretraining before the branch stage
  does not improve the critic's transfer to unseen torques (B) or
  held-out episodes (C) -- T is at chance on both like C1 -- and does not
  improve, let alone rescue, the deployed policy; the pretrain-only
  critic is anti-aligned with the counterfactual outcomes at the training
  keys.  The proposal's premise (the recorded trajectories supply a
  pose / torque representation that the branch stage can then correct)
  is not borne out at this budget: whatever representation 100k of
  vanilla NCE builds does not carry the branch-stage value beyond the
  branch keys.
- Does not establish: that no pretraining schedule can help (one budget,
  one recipe, the critic's own encoders as the representation); anything
  about a different actor coupling (bc 0.05 with a frozen critic is the
  pre-registered choice, and it breaks the walker in most seeds for every
  critic); the C-level comparison rests on 45 held-out episodes and the
  policy-level one on 3 seeds with seed s.e. ~0.1.
- Caveats: T and C1 / C2 actors trained on different GPUs; the T chain was
  relaunched once after a first attempt ran on CPU by mistake (killed
  before any checkpoint, no result reused); the C2 critic's region
  readout uses the recorded dataset's goal marginal (4,000 frames), not
  the replay's.

Not committed; no reserved seeds used; the actor stage was part of the
pre-registered design here (the user's step 3) and is reported as such.
