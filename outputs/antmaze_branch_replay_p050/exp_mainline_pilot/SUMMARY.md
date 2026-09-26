# Mainline pilot (O vs CF-oracle positive futures, matched offline CRL): run complete -- CF above O in 3/3 seeds on every metric, pre-registered success rule NOT met (seed variability)

Contract `notes/MAINLINE_CONTRACT.md`.  Script `scripts/exp_v6_mainline_pilot.py`.
Sealed manifest `manifest.json` (2026-09-18 23:54 node time, before generation and
training; `manifest_blocked_20260919.json` is the earlier seal made while the
start checkpoint was unavailable).  Checks `AUDIT.md` / `audit.json` (pre- and
post-training, all PASS).  Machine-readable result `REPORT.md` / `results.json`.
Toy code-path test `_smoke/`.

## Start agent: re-derived (user decision, option b)

The contract's start agent `joint_van_d05/seed_0/final.pkl` existed only on
node 30021, which died on 2026-09-18 (no copy anywhere; see
`manifest_blocked_20260919.json`).  The user chose to re-derive it: the
exp_detour_ratio recipe and seed 0, unchanged (pure BC 100k -> vanilla critic
30k -> frozen-critic actor 30k, all on d05; `node_rederive_d05.sh`, 43 min on
the 4090L).  Record: `joint_van_d05/seed_0/REDERIVED.json` (in the manifest
under `start_agent.rederivation`); the lost run's pulled JSONs are kept under
`*/seed_0/_lost_20260918/`.  Not byte-identical to the lost file (GPU
nondeterminism).  Sanity: pure-BC walker 909-draw success 0.253 / detour 0
(lost record 0.257 / 0); the agent 0.247 / detour 1/300 / timeout 0.013 (lost
record 0.263 / 20/300 / 0.063) -- the lost seed 0 was the most detour-prone of
its three siblings (20 / 2 / 4 per 300); the re-derived one sits with the
other two.  The demonstration share in the data is unchanged (50 / 1,000
episodes; anchor mass on detour episodes 0.0497).

## What ran

- Anchors: 60,000 draws of the offline law -> 53,747 unique logged rows (196
  at t = 0), weights = multiplicity / 60,000; one file for both arms.
- CF branches: 53,747 (one per anchor; logged torque once, then the start
  agent's mode to the first reach frame / death / horizon; hazards redrawn;
  no goal hold; nothing filtered): success 0.621 / death 0.290 / timeout
  0.088; by anchor time: t = 0 success 0.30 / death 0.66; t in [6, 50)
  0.25 / 0.73; t in [50, 150) 0.48 / 0.37; t >= 150 0.91 / 0.03.  7.70 M
  rows, 18 min on 18 workers.  (Arm O's recorded futures are the sighted
  teacher's: every episode a success.)
- Training: 30,000 optimizer updates per run (counted), joint critic +
  actor, V6 recipe at gamma 0.999, batch 1024, fresh paired critics, actors
  from the start agent, actor batches = the buffer's own law over the whole
  d05 file (random_goals 0, bc 0.05 on the same rows).  Seed 0 on node 30043
  (4090L), seed 1 on 30125 (4080S), seed 2 on 30016 (5090L); both arms of a
  seed on the same GPU, 390-550 s each.  First-batch hashes identical across
  arms for every seed; critic and actor parameters moved in every run.
- Evaluation: 300 natural draws, env seed 3909, mode policy; episode
  identities verified identical across the seven policies.

## Result (`REPORT.md`)

| policy | success | detour | death | timeout | success no hazard (74) | success hazard (226) | zone-1 / 2 deaths |
|---|---:|---:|---:|---:|---:|---:|---|
| start (re-derived d05 agent) | 0.243 | 0.003 | 0.740 | 0.017 | 0.986 | 0.000 | 152 / 70 |
| O seed 0 / 1 / 2 | 0.250 / 0.263 / 0.250 | 0.023 / 0.030 / 0.003 | 0.710 / 0.723 / 0.740 | 0.040 / 0.013 / 0.010 | 0.973 / 0.986 / 1.000 | 0.013 / 0.027 / 0.004 | ~150 / ~68 |
| CF seed 0 / 1 / 2 | 0.267 / 0.320 / **0.480** | 0.107 / 0.167 / **0.500** | 0.627 / 0.567 / **0.260** | 0.107 / 0.113 / 0.260 | 0.838 / 0.892 / 0.689 | 0.080 / 0.133 / **0.412** | 134/54, 118/52, 53/25 |

Seed means (seed s.e.): O success 0.254 (0.004), detour 0.019, death 0.724,
hazard success 0.015; CF success 0.356 (0.064), detour 0.258 (0.122), death
0.484 (0.114), timeout 0.160 (0.050), hazard success 0.208 (0.103).

Paired on the common episodes (per seed, episode s.e. in brackets; then the
seed mean, seed s.e., episode-paired bootstrap s.e. with seeds fixed):

| comparison | success | detour | death | timeout |
|---|---|---|---|---|
| CF - O (primary) | +0.017 (0.017) / +0.057 (0.020) / +0.230 (0.034); **+0.101, seed s.e. 0.065, boot 0.017, 3/3** | +0.083 / +0.137 / +0.497; +0.239 (0.130), 3/3 | -0.083 / -0.157 / -0.480; -0.240 (0.122), 3/3 | +0.067 / +0.100 / +0.250; +0.139 (0.056), 3/3 |
| CF - start (practical) | +0.023 / +0.077 / +0.237; +0.112 (0.064), 3/3 | +0.254 (0.122), 3/3 | -0.256 (0.114), 3/3 | +0.143 (0.050), 3/3 |
| O - start | +0.007 / +0.020 / +0.007; +0.011 (0.004), 3/3 | +0.016 (0.008), 2/3 | -0.016 (0.009), 2/3 | +0.004, 1/3 |

Pre-registered rule (mean over the 3 paired seeds > 2 x seed s.e. AND 3/3):
**primary CF - O success: NOT met** (+0.101 vs the 0.130 the rule needs; 3/3
positive); **practical CF - start: NOT met** (+0.112 vs 0.128; 3/3).  Within
each seed the paired episode difference is > 2 episode s.e. for seeds 1 and
2 and not for seed 0 (+0.017 +- 0.017).

By realised route (evaluation episodes): CF actors take the detour in
32 / 50 / 150 of 300 episodes (O: 7 / 9 / 1; start: 1) and complete it
0.62 / 0.76 / 0.75 of the time (the rest time out); on the shortcut they die
at the O rate (0.68-0.72).  Under an active hazard CF succeeds 0.08 / 0.13 /
0.41 (O 0.00-0.03) because it is on the detour; without a hazard CF loses
0.10-0.30 of the near-perfect O success to detour timeouts (median successful
episode 234 / 236 / 380 steps vs 224).

## Reading under the contract

- Primary: not established.  CF - O success is +0.10 with a seed s.e. of
  0.065 -- three seeds, one of them (+0.23) far larger than the other two
  (+0.02, +0.06).  The rule was written for exactly this case and it says
  "not met"; this is not a negative-effect finding either (3/3 positive,
  episode-level uncertainty 0.017).
- Direction and mechanism, consistent 3/3 on every metric: with the
  counterfactual futures the same procedure moves the agent onto the detour
  (detour rate 0.3 % -> 11-50 %), roughly halves the deaths in the strongest
  seed, and raises hazard-active success from ~0 to 0.08-0.41, at the price
  of slower walking and detour timeouts (timeout 0.02 -> 0.11-0.26).  With
  the recorded futures the identical procedure changes almost nothing
  (O - start +0.011): the sighted teacher's futures say every anchor
  succeeds, so the critic has no reason to prefer one first torque over
  another at the fork.
- Practical check: CF above the start in 3/3 (+0.02 / +0.08 / +0.24), not
  established by the rule for the same reason.
- What this establishes: under a matched offline CRL procedure (same
  anchors and weights, same actor stream, same losses, same initialisation,
  same update count), the source of the critic's positive futures is what
  decides whether the actor learns the route: oracle counterfactual futures
  do, observational ones do not, in this data.  What it does not establish:
  the size of the success gain (three seeds; the pre-registered rule is
  not met), why seed 2 responds three times more than seeds 0-1 (not
  investigated -- stopping rule), and anything about a learned ETT (the
  simulator produced the futures).  It is oracle evidence for the sampling
  design; the remaining methodological step is to replace the oracle with
  an offline-learned ETT.  No further diagnostics or configurations were
  launched.

## How this differs from the historical diagnostic runs

- Anchors: logged rows under the recipe's own law with equal weights in
  both arms (not dense strata + every-60th rows, not replay row 0, not the
  dataset's reset rows).
- Intervention: only the positive futures (recorded vs generated after the
  logged torque by the fixed agent); no candidate queries, repeated draws,
  goal hold or filtering.
- Actor: whole dataset, standard pairing, joint training with a fresh
  paired critic (not a frozen critic, reset-only anchors, balanced BC, or a
  pure-BC / d20 initialisation).  d20 enters nowhere.
- Comparison: CF vs O paired on common episodes with the practical check
  separate; evaluation uncertainty and seed variability reported separately.
- Under those historical configurations the branch replays made the actor
  worse; under this matched procedure the counterfactual arm is the one
  that moves.

## Infrastructure notes

Helper nodes 30125 / 30016 were bootstrapped with the pinned environment
(`node_setup_30108.sh`); the orchestrator (`node_mainline.sh` in the session
scratchpad) had two timing bugs -- the launch ssh blocked until the remote
job finished (missing stdin redirect), and the completion poll used a
relative path -- which delayed node 30043's own seed by ~15 min and needed a
manual marker; neither touched any computation.  Checkpoints (`O/*/`,
`CF/*/`, the re-derived d05 chain), `branches_cf.npz` (984 MB) and
`anchors.npz` stay on node 30043 (`anchors.npz` is deterministic in the seed;
its hash is in the manifest).
