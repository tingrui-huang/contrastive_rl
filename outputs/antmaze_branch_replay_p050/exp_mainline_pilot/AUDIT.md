# Pre-training and post-hoc checks (mainline pilot)

| check | value |
|---|---|
| smoke | False |
| dataset_sha_matches | PASS |
| sidecar_sha_matches | PASS |
| no_held_out_or_cnew_episode_in_training | PASS |
| anchors_deterministic | PASS |
| anchors_valid_rows | PASS |
| anchor_weight_sum | 1.0000000000000002 |
| anchor_roots_match_dataset | PASS |
| anchor_law.draws_per_episode_mean_sd | [60.0, 7.874007874011811] |
| anchor_law.expected_sd_if_uniform | 7.745966692414834 |
| anchor_law.relative_row_position_mean (0.5 if uniform) | 0.4982181194926541 |
| anchor_law.share_t0 | 0.003783333333333334 |
| anchor_law.expected_share_t0 | 0.003806691489056454 |
| branches_available | False |
| critic_stream_recorded_first_batches | 5 batches: 8a2cde22818624b9/509c15fca79bbba4; cf879f96d936572e/f0a1ba803a34fc51; ca416aa436816d18/b2fc601cc9331c03 ... |
| actor_streams.seed_0.identical_across_instances | PASS |
| actor_streams.seed_0.reset_row_share | 0.003125 |
| actor_streams.seed_0.non_reset_row_share | 0.996875 |
| actor_streams.seed_0.future_crosses_episode_boundary | PASS |
| actor_streams.seed_0.sampler | TrajectoryBuffer variable-length law |
| actor_streams.seed_0.law_check | PASS |
| actor_streams.seed_1.identical_across_instances | PASS |
| actor_streams.seed_1.reset_row_share | 0.0044921875 |
| actor_streams.seed_1.non_reset_row_share | 0.9955078125 |
| actor_streams.seed_1.future_crosses_episode_boundary | PASS |
| actor_streams.seed_1.sampler | TrajectoryBuffer variable-length law |
| actor_streams.seed_1.law_check | PASS |
| actor_streams.seed_2.identical_across_instances | PASS |
| actor_streams.seed_2.reset_row_share | 0.0037109375 |
| actor_streams.seed_2.non_reset_row_share | 0.9962890625 |
| actor_streams.seed_2.future_crosses_episode_boundary | PASS |
| actor_streams.seed_2.sampler | TrajectoryBuffer variable-length law |
| actor_streams.seed_2.law_check | PASS |
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
| manifest_mentions_no_d20_artifact | PASS |
