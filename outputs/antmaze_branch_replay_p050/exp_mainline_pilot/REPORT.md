# Mainline pilot: observational (O) vs counterfactual-oracle (CF) positive futures

Contract `notes/MAINLINE_CONTRACT.md`; manifest `manifest.json` (sealed 2026-09-19 00:32:52); checks `AUDIT.md`.  Evaluation: 300 natural draws, seed 3909, mode policy, the same episodes for every policy.  Uncertainty: per-seed episode s.e. (evaluation), seed s.e. over 3 paired training seeds (training variability), episode-paired bootstrap of the seed mean (evaluation uncertainty with the seeds fixed).  Rule: CF - O mean over the 3 paired seeds > 2 x seed s.e. and 3/3 seeds in the same direction.

## Status

No evaluation exists.  Start checkpoint available: False (D:\Users\trhua\Research\contrastive_rl\outputs\antmaze_branch_replay_p050\joint_van_d05\seed_0\final.pkl).  See the blocker section of the SUMMARY.
