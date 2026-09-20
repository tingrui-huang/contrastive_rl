# The pairing check (user's plan after multi_futures, 2026-09-21; no training): at the reset / early-fork states the single-table critics learned their own draw's (state, action) -> future pairing, not a repeatable consequence -- there is none per anchor in the data there

`scripts/diag_v6_pairing.py run` (node3, 4.6 min): `data.json`, `critic.json`, `actor.json`, `REPORT.md`.  Inputs: the two oracle tables (T1 =
sealed draw, T2 = draw 2), the 15 critics / actors (CF, CF2, MF x seeds 0-4), the start agent and the O actors, the d05 dataset.  Strata by
the anchor's state: reset (t = 0, n 196), start_early (start region, t 1-40, n 4,049), west_entry (west column y < 4, n 153; all from detour
episodes), pre_zone1_early (x < 4.5, n 4,479), pre_mouth (x 4.5-6.6, n 8,062), zone1, between, far_legs, all.  The anchors' source-episode
route: 5 % detour (d05).  Repr-based logits checked against the q_network's pairwise logits (max diff 0).

## Layer 1 -- the data

* Per-anchor repeatability of the consequence across the two draws (corr of the branch SUCCESS indicator, T1 vs T2): reset -0.06,
  start_early +0.11, pre_zone1_early -0.01; pre_mouth 0.49, zone1 0.34, between 0.36; far legs ~1 (no hazard).  Outcome agreement at
  the fork 0.55-0.65.  At the states where the route is decided, one draw says nothing about the next: the consequence is fixed by
  the hazard draw 50-100 steps later, not by (state, action).
* The population-level detour signal exists and is the same in both tables: at start_early the anchors whose logged torque comes from
  a detour episode (170) carry goal-area mass 0.066 / 0.073 vs 0.035 / 0.034 for the shortcut ones (+0.030 / +0.039), near-goal (2.0)
  mass 0.081 / 0.089 vs 0.028 / 0.026, death-frame mass 0.001 vs 0.011, far-region mass +0.71 in both.  (Over all anchors the detour
  ones carry LESS goal-area mass, 0.179 vs 0.278: the far route is long under the geometric law.)  This is a cross-state comparison
  (the detour anchors sit north of the start), reported as the data's signal only.
* The actor's own training goals (the recorded-episode law on the same rows): goal-area mass 0.075 at reset / 0.077 at start_early;
  the relabelled goal lies on the far route for 0.87-0.93 of the detour rows and 0.00 of the shortcut rows.  Reach-radius (0.5) mass
  is 0.001-0.005 everywhere under both laws.

## Layer 2 -- the critics (15, same inputs)

* NCE fit on identical batches: every family fits T1, T2 and the mixture equally (loss 0.0065-0.0066, categorical accuracy
  0.025-0.027).  The batch loss does not see the pairing.
* The pairing test (the critic's logit at its anchor's own (s, a, task goal) vs the realised consequence; seed mean of five):

  | stratum | data corr success T1,T2 | CF: own draw / other draw | CF2: own / other | MF: T1 / T2 |
  |---|---:|---|---|---|
  | reset | -0.06 | 0.242 / -0.033 | 0.446 / -0.148 | -0.115 / 0.220 |
  | start_early | +0.11 | 0.415 / +0.010 | 0.413 / -0.008 | 0.160 / 0.173 |
  | pre_zone1_early | -0.01 | 0.493 / -0.007 | 0.467 / -0.009 | 0.158 / 0.151 |
  | pre_mouth | +0.49 | 0.408 / 0.270 | 0.424 / 0.266 | 0.306 / 0.320 |
  | zone1 | +0.34 | 0.305 / 0.144 | 0.304 / 0.121 | 0.142 / 0.164 |
  | between | +0.36 | 0.210 / 0.138 | 0.185 / 0.142 | 0.152 / 0.143 |

  (the same pattern with the goal-area mass and the reach mass as targets, `critic.json`).  At the reset / early-fork states each
  single-table critic tracks its OWN draw's outcomes at 0.24-0.49 and the other draw's at 0.00 -- as much cross-draw structure as the
  data has (none).  Where a repeatable component exists (pre-mouth and beyond) the critics track the other draw too (0.12-0.27).
  MF tracks each draw at 0.15-0.17 at the fork; with both draws in its training set this cannot be separated from half-memorisation
  (a third draw would be needed).
* Normalised region read-out (softmax over a common goal set from both tables; per-anchor corr of the predicted goal-area mass with
  the true one): reset CF 0.64 (T1) / 0.34 (T2), CF2 0.29 / 0.60, MF 0.32 / 0.39; start_early CF 0.51 / 0.21, CF2 0.25 / 0.44, MF 0.32 /
  0.36; pre_zone1_early CF 0.62 / 0.10, CF2 0.12 / 0.60, MF 0.34 / 0.29.  Same picture under a per-critic normalisation.

## Layer 3 -- the actor (same state, same goal; candidates = the logged torque and the modes of pi_CF / pi_CF2 / pi_MF / pi_start / pi_O at THAT state; objective = crl.losses.actor_loss with twin-min, 16 samples, bc 0.05; seed-matched pairs)

* Reset states, task goal.  Under the CF critic: mean Q_min pi_CF -8.12, pi_CF2 -8.70, pi_MF -9.13, logged -9.71, pi_start -9.67;
  pi_CF's action is the best candidate in 54 % of the states, above pi_start in 92 %, above pi_MF in 86 %; the objective of pi_MF
  minus pi_CF: critic term +0.90 (per seed +1.04 / +1.19 / +0.85 / +0.93 / +0.50), BC term -3.75 nats (x 0.05 = -0.19), total +0.67 --
  the CF actor's far-going action is worth 0.9 nats to its own critic, five times the BC cost.
* Under the MF critic: pi_CF -8.92, pi_CF2 -8.29, pi_MF -8.93, pi_start -9.80; pi_CF above pi_start in 92 % of the states but above
  pi_MF in 54 %; the critic term of pi_MF minus pi_CF +0.04 (per seed -0.52 / +0.39 / +0.29 / +0.15 / -0.10 -- sign seed-dependent),
  BC -3.75 nats, total -0.15: the MF critic is indifferent between the far-going action and the MF actor's own action, and the BC
  term decides for the latter.  The same at start_early / pre_zone1_early / pre_mouth (critic-term differences +0.06 / +0.06 / -0.06
  under MF vs +0.31 / +0.16 / +0.31 under CF) and at the relabelled training goals (MF +0.01 / +0.02 / -0.10; CF +0.46 / +0.18 / +0.18).
* So the far-route preference of the CF actor rests on a ~0.9-nat margin that its single-draw critic assigns to the far-going action at
  the reset states; under the two-draw critic that margin is ~0 with a seed-dependent sign, and the MF actor settles where BC is
  cheapest -- the start agent's behaviour.

## Reading against the user's decision tree

* "The original good critic holds only on its own future draw": YES at the reset and early-fork states -- own-draw correlation 0.24-0.49,
  other-draw ~0, equal to the data's cross-draw correlation there (~0).  The 0.449 is therefore not a reliable oracle benchmark; the
  supervision at the decision states has to be stabilised before any learned-ETT comparison is meaningful.
* "The two consequences of the key anchors diverge strongly": YES -- at the fork the per-anchor success indicator is uncorrelated across
  draws (agreement 0.55-0.65).  The population-level advantage of the far-going anchors is small in the geometric mass (+0.03-0.04
  goal-area on a base of 0.035; death-frame 0.001 vs 0.011) and is the same in both tables, i.e. it is real but buried under the
  per-anchor draw noise that a single future cannot average.
* "The averaged target has a signal the MF critic did not fit": partly -- MF's per-draw tracking at the fork (0.15-0.17) cannot be
  separated from half-memorisation with two draws; its same-state margin for the far-going action is ~0.
* "The MF critic keeps the preference and the actor refuses it": NO in the strict sense -- the MF critic prefers the far-going action
  over the start agent's (92 %) but not over the MF actor's own action (54 %); the objective is decided by BC only because the critic
  term is flat between those two.
* Consequence (the user's pre-stated remedy for this branch): repeated paired hazard draws at the pre-selected decision states (reset /
  start_early / pre_zone1_early anchors; e.g. 32 draws per candidate action with the frozen start-agent continuation) to measure the
  true average advantage of the far-going vs the straight candidate at the SAME state -- no full tables, no five-seed training.
* Caveats: the pairing test's "vs avg" columns are not out-of-sample for any critic (each has at least one of the two draws in its
  training set); the cross-anchor "detour signal" in the data compares different states and is not used for the action reading; the
  reach-radius masses are 0.001-0.005 everywhere and the goal-area / success targets carry the result.
