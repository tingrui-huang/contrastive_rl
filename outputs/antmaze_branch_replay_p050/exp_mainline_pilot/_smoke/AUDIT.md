# Pre-training and post-hoc checks (mainline pilot)

SMOKE RUN: toy scale, stand-in continuation, no checkpoint -- a code-path test, not a result.
| check | value |
|---|---|
| smoke | True |
| dataset_sha_matches | PASS |
| sidecar_sha_matches | PASS |
| no_held_out_or_cnew_episode_in_training | PASS |
| anchors_valid_rows | PASS |
| anchor_weight_sum | 0.9999999999999999 |
| anchor_roots_match_dataset | PASS |
| anchor_law.draws_per_episode_mean_sd | [0.048, 0.2137662274541982] |
| anchor_law.expected_sd_if_uniform | 0.21908902300206645 |
| anchor_law.relative_row_position_mean (0.5 if uniform) | 0.4845535772282033 |
| anchor_law.share_t0 | 0.0 |
| anchor_law.expected_share_t0 | 0.003806691489056454 |
| branches_available | True |
| critic_stream_recorded_first_batches | 2 batches: 885e20833e25f2d8/1eae3f2a8bfe658f; c4445b270bb149fe/e1c559086c996956 ... |
| critic_anchor_sequence_identical_across_arms | PASS |
| critic_goals_differ_across_arms | True |
| branch_roots_keep_timestep | PASS |
| query_executed_exactly_once | PASS |
| branch_first_action_is_logged | PASS |
| branch_lengths_within_horizon | PASS |
| branch_restore_maxdiff_max | 0.0 |
| branch_outcomes.death | 3 |
| branch_outcomes.success | 2 |
| branch_outcomes.timeout | 43 |
| branch_all_outcomes_kept | PASS |
| actor_streams.seed_0.identical_across_instances | PASS |
| actor_streams.seed_0.reset_row_share | 0.001953125 |
| actor_streams.seed_0.non_reset_row_share | 0.998046875 |
| actor_streams.seed_0.future_crosses_episode_boundary | PASS |
| actor_streams.seed_0.sampler | TrajectoryBuffer variable-length law |
| actor_streams.seed_0.law_check | PASS |
| offline_static_audit.all_pass | PASS |
| offline_static_audit.gates.G1_FINGERPRINT | PASS |
| offline_static_audit.gates.G2_KEY_SEPARATION | PASS |
| offline_static_audit.gates.G3_SHAPES_DIMS | PASS |
| offline_static_audit.gates.G4_DTYPES_FINITE | PASS |
| offline_static_audit.gates.G5_EP_LENGTHS | PASS |
| offline_static_audit.gates.G6_NO_AUDIT_LEAK | PASS |
| offline_static_audit.gates.G7_RELABEL_BOUNDS | PASS |
| offline_static_audit.gates.G8_FROZEN_BUFFER | PASS |
| start_checkpoint.path | D:\Users\trhua\Research\contrastive_rl\outputs\antmaze_branch_replay_p050\joint_van_d05\seed_0\final.pkl |
| start_checkpoint.available | False |
| start_checkpoint.sha256 | UNAVAILABLE |
| post_training.O/seed_0.critic_changed | PASS |
| post_training.O/seed_0.actor_changed | PASS |
| post_training.O/seed_0.optimizer_updates | 8 |
| post_training.O/seed_0.actor_init_is_start_agent | FAIL |
| post_training.O/seed_0.first_critic_anchor_hash | 212a99c02db189c5 |
| post_training.O/seed_0.first_actor_hash | 9344aad72b933833 |
| post_training.CF/seed_0.critic_changed | PASS |
| post_training.CF/seed_0.actor_changed | PASS |
| post_training.CF/seed_0.optimizer_updates | 8 |
| post_training.CF/seed_0.actor_init_is_start_agent | FAIL |
| post_training.CF/seed_0.first_critic_anchor_hash | 212a99c02db189c5 |
| post_training.CF/seed_0.first_actor_hash | 9344aad72b933833 |
| paired_streams_identical_across_arms.seed_0 | PASS |
