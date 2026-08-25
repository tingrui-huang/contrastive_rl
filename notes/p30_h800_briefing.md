# p30 / H800 rockfall benchmark: full setup summary (new-session briefing)

Written for a freshly opened Claude window / future self. Every number is checked directly
against the code, the manifests and the committed reports (as of commit `44edf50`,
2026-07-28). Full variant name:
**`local_detour_v2.1_sev0.80_p30_h800_resetfix_v1`**.

## 1. Environment (crl/rockfall_ant.py, module FROZEN)

`offline_ant_umaze_rockfall`: D4RL AntMaze U-maze base, learner obs is 58-dim
proprioception (the mask is not visible), 8-dim action, goals follow the d4rl eval protocol.

- 4 rockfall sites `ROCKFALL_SITES`: left_1 (x=3.0, y=+1), left_2 (4.3, +1),
  right_1 (3.6, -1), right_2 (4.9, -1); each activates independently Bernoulli, fixed
  per episode, and is written only to the sidecar.
- Trigger: |x - sx| <= 0.6 and |y| in [1.0, 2.0] and vx > 0.1, with a 3-step dwell
  (TRIG_DWELL).
- Severity (v2.1 protocol): **0.80 severe (absorbing collapse) / 0.15 impaired (2 legs,
  gear x0.15, damping x3.5) / 0.05 mild (impulse)**.
- Center route: U-independent mud drag MUD_DRAG = 40 for x in (2.4, 5.6), applied via
  qfrc_applied.
- **Module defaults are untouched**: P_ACTIVE = 0.2, SEVERITY = (0.55, 0.30, 0.15),
  max_episode_steps = 700. p30 / severity / H are all **instance-level config overrides**
  (crl/envs.py: `rockfall_p_active`, `rockfall_severity`, horizon override; on the teacher
  side via `rockfall_v2_teacher.apply_v2_config`).

## 2. p_active = 0.30

- config `rockfall_p_active=0.30` -> expected mask spectrum: all_clear .2401 /
  left_only .2499 / right_only .2499 / **both_sides .2601** (compare p=0.2, where
  all_clear ~= .41). Prescreen measures a per-site activation frequency of ~ .31/.31/.31/.31.
- Motivation: at p=0.2 all_clear dominates and the worst-mask signal is sparse; p=0.3 pulls
  the four patterns close to uniform. (p=0.5 was also swept, commits `4867763` / `92a8302`.)

## 3. Reset fix (resetfix_v1)

- Bug: the legacy reset only rewrote qpos/qvel, so MuJoCo warmstart/qacc/qfrc/contacts leaked
  across episodes -> episodes were not independent.
- Fix: `_RockSim.reset_model` now calls `mj_resetData` first when `full_reset=True`
  (clearing warmstart/qacc/qfrc/act/ctrl/time/contacts) and then sets the initial state
  (INIT_QPOS + U(-0.1, 0.1) jitter, qvel = 0.1 * N).
- Correctness gates A-E: scripts/test_reset_independence.py, PASS for both cap-700 and
  cap-800 (artifacts/reset_fix/reset_tests*.json).
- Quantified impact (PART 2, N=500, same bank and same model): 3.6-9.4% of episodes flip but
  it is **statistically unbiased**, pooled change <= 0.012 -> the legacy conclusions hold
  approximately; the fix buys reproducibility and does not change any verdict.
  (artifacts/reset_fix/legacy_vs_corrected_n500.json)

## 4. H=800 dataset (the training data for the current propensity line)

`artifacts/rockfall_v2_p30_h800_resetfix/pilot/antmaze_rockfall_v2_p30_h800_resetfix_pilot.npz`

- sha256 `08bdc44b...`; sidecar `a0362b7c...`; manifest in the same directory.
- 300 episodes (90/0/10 = 270 sighted local-detour / 0 blind / 30 center),
  obs [300, 801, 58], act [300, 801, 8], **227,764** transitions, episode lengths 26-801.
- Teacher: v2 local-detour (DETOUR_Y .72 / PRE 1.1 / POST .8 / LANE 1.1), base side balanced
  and independent of the mask; walker `artifacts/walker/phase1/walker_best.pkl`
  (sha 70b0a460...), base policy
  `offline_umaze_bc005_twinmin_s0_50k/checkpoints/best.pkl` (sha 6bece3e3...).
- Seeds: env 52400019 / dataset rng 51990013 / side rng 51990014.
- Collection commit `44aec04` (frozen rockfall code `225b2b0`); V1-V10 + G1-G8 all pass.
- H700 twin dataset: `artifacts/rockfall_v2_p30_h700_resetfix/pilot/`
  (sha `ff5a8136...`).

## 5. Authoritative evaluation conclusions (PART 6/7/8, commit `d7e535f`)

N=1000 natural paired bank, corrected reset, each policy at its native cap,
primary metric = pooled success. Report: `artifacts/reports/p30_h800_resetfix_REPORT.md`.

| | H700 | H800 |
|---|---|---|
| teacher | 0.868 | 0.870 |
| **center** | 0.800 | **0.899 (overtakes the teacher)** |
| blind | 0.500 | 0.503 |
| fresh naive final/best | 0.615 / 0.704 | 0.492 / 0.542 |
| naive worst-mask (final) | 0.306 | 0.106 |
| naive center usage | 0.32 | 0.07 |

- **The H800 center route overtaking the teacher (-0.029) still holds after the reset fix** --
  it is a genuine controller-level property (the center route saves 99 extra episodes in the
  701-800 step window; naive gains only +3, so it is not merely timeout-limited). The
  teacher/center ordering is normal at H700 (+0.068).
- At H800 the naive policy essentially abandons the center route (0.32 -> 0.07) and rushes the
  side lanes (exposure .97).
- **Verdict: the formal benchmark stays at H=700**; H800 is retained as a horizon-stress
  variant. (Note: the current propensity / behavior-support module uses the **H800** pilot
  data -- it only needs behavioural data and does not depend on the benchmark verdict.)

## 6. Current work on this data (propensity/ module, commit `44edf50`)

- Stage 1 loader/audit: context c = concat(s, g_cmd), 58-dim (29+29, the **pre-action
  commanded goal**, not the contrastive relabel); episode-level 9/1 split
  (artifacts/propensity_stage1/...json).
- Stage 2 behaviour flow model mu(a|c): conditional flow matching, MLP 256x3, lr 3e-4,
  batch 256; runs: s0 50k (val MSE 3.212) + cont150k; sampling with 10 Euler steps and an
  action clip to [-1, 1] after every step (annotated "matches the official CFQL
  implementation, PROTOTYPE, unvalidated"). Checkpoints: artifacts/propensity_flow/...
- Stage 2.5 audit: **flow-generated actions carry a boundary-artifact signature with
  ~0.97 AUC** (real actions ~0.52) -> the Stage 3A discriminator uses only real actions as
  the positive class.
- Stage 3A support discriminator D(s, g_cmd, a): positives = real behaviour actions,
  negatives = actions from a naive CRL checkpoint (pi draws g_query via the replay law, and
  g_query is not an input); **explicitly labelled a relative support/discrepancy score, NOT a
  propensity and NOT a causal mixture weight** (consistent with the M0 T3 conclusion: source
  classification != propensity).
- Related: scripts/m0_agreement_equivalence.py (M0, 13/13 gates, commit `e7e2899`) certifies
  on the swamp testbed that the agreement event is equivalent to the propensity table.

## 7. Things that must not be touched (standing rule)

Rockfall module defaults (P_ACTIVE 0.2 / severity 0.55, 0.30, 0.15 / H700), global-route v1
data and documentation, legacy datasets and checkpoints, the walker, and the base policy.
p30 / H800 / severity all go through instance-level overrides; a new variant gets a new
directory and a new manifest.

## 8. Known caveats

- All naive numbers are single-seed; treat differences of that magnitude as noise. The
  scripted-anchor differences are real.
- Always compare using the final checkpoint (best leaks model selection to the baseline).
- The H800 naive 300k run was done on Colab (run_p30_h800_resetfix.ipynb, pinned `14335a6`);
  the authoritative evaluation runs on the workstation after download via
  scripts/authoritative_eval.py.
