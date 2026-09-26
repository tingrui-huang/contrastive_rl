# The three-layer acceptance of the CF-motion revision (test split of the CF-controlled rollouts; no training)

Models {'v4': '/root/contrastive_rl/outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_one_step_v4', 'v4a': '/root/contrastive_rl/outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_one_step_v4a', 'v4b': '/root/contrastive_rl/outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_one_step_v4b'}; test rollouts 174, rows 49542.

## L1 -- teacher forcing (real state every step): one-step xy error (median / p90) by row type and phase; the stationary gate; slow rows

| model | moving | slow | static | turn30 | far | corridor | zone | gate fires on static / slow / moving | slow rows: real speed / pred speed / pred frozen share | static rows pred speed |
|---|---|---|---|---|---|---|---|---|---|---|
| v4 | 0.0070 / 0.0264 (n 26346) | 0.0248 / 0.0332 (n 3971) | 0.0000 / 0.0000 (n 19225) | 0.0140 / 0.0312 (n 1080) | 0.0113 / 0.0313 (n 10568) | 0.0017 / 0.0195 (n 26399) | 0.0000 / 0.0139 (n 11495) | 0.982 / 0.018 / 0.000 | 0.0082 / 0.0250 / 0.02 | 0.0000 |
| v4a | 0.0059 / 0.0209 (n 26346) | 0.0023 / 0.0189 (n 3971) | 0.0000 / 0.0000 (n 19225) | 0.0113 / 0.0260 (n 1080) | 0.0046 / 0.0207 (n 10568) | 0.0016 / 0.0136 (n 26399) | 0.0000 / 0.0112 (n 11495) | 0.982 / 0.014 / 0.000 | 0.0082 / 0.0095 / 0.01 | 0.0000 |
| v4b | 0.0065 / 0.0223 (n 26346) | 0.0029 / 0.0193 (n 3971) | 0.0000 / 0.0000 (n 19225) | 0.0122 / 0.0284 (n 1080) | 0.0051 / 0.0218 (n 10568) | 0.0018 / 0.0147 (n 26399) | 0.0000 / 0.0118 (n 11495) | 0.982 / 0.017 / 0.000 | 0.0082 / 0.0096 / 0.02 | 0.0000 |

## L2 -- open loop (real action sequence, model rolled forward; onset off): xy error median / p90 at steps; stall segments

| model | step 5 | step 10 | step 20 | step 30 | step 50 | step 100 | pose err median at 30 / 100 | vel err median at 30 / 100 | stall segments (n, len): real disp / model disp median; share model disp > 0.5 / > 2.0 (real > 0.5) |
|---|---|---|---|---|---|---|---|---|---|
| v4 | 0.040 / 0.137 | 0.105 / 0.350 | 0.224 / 0.709 | 0.400 / 1.218 | 0.810 / 2.472 | 2.044 / 6.759 | 0.039 / 0.066 | 0.375 / 0.444 | (8, 26): 0.05 / 0.77; 0.75 / 0.25 (0.00) |
| v4a | 0.037 / 0.120 | 0.099 / 0.266 | 0.192 / 0.497 | 0.352 / 0.787 | 0.649 / 1.830 | 1.594 / 5.684 | 0.031 / 0.076 | 0.270 / 0.422 | (8, 26): 0.05 / 1.04; 0.75 / 0.00 (0.00) |
| v4b | 0.042 / 0.119 | 0.090 / 0.278 | 0.158 / 0.525 | 0.219 / 0.667 | 0.429 / 1.281 | 1.039 / 2.194 | 0.027 / 0.042 | 0.220 / 0.303 | (8, 26): 0.05 / 0.44; 0.25 / 0.00 (0.00) |

## L2b -- restart at the real stall-segment entry (and 20 rows before it), recorded actions through the segment

| model | restart | n | seg len median | real disp median | model disp median / p90 | share model disp > 0.5 / < 0.2 | xy error at the entry (median) | gate share median | real speed median |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| v4 | restart_0_before_entry | 64 | 496 | 0.024 | 3.225 / 1261.163 | 0.94 / 0.03 | 0.000 | 0.00 | 0.0000 |
| v4 | restart_20_before_entry | 64 | 496 | 0.024 | 2.957 / 39.037 | 0.97 / 0.02 | 1.047 | 0.00 | 0.0000 |
| v4a | restart_0_before_entry | 64 | 496 | 0.024 | 0.090 / 4.362 | 0.25 / 0.72 | 0.000 | 0.00 | 0.0000 |
| v4a | restart_20_before_entry | 64 | 496 | 0.024 | 0.162 / 4.870 | 0.36 / 0.55 | 0.218 | 0.00 | 0.0000 |
| v4b | restart_0_before_entry | 64 | 496 | 0.024 | 0.171 / 3.245 | 0.39 / 0.55 | 0.000 | 0.00 | 0.0000 |
| v4b | restart_20_before_entry | 64 | 496 | 0.024 | 0.266 / 4.519 | 0.41 / 0.34 | 0.186 | 0.00 | 0.0000 |

Split / episode overlap: {"train_episodes": 307, "test_episodes": 66, "test_episodes_also_in_train": 0, "val_episodes_also_in_train": 0, "note": "the split is by start anchor; anchors of one source episode can fall into different splits (their CF rollouts are different trajectories from different states)"}


## L3 -- closed loop (CF actor on the model state; onset off) vs the real hazard-free rollouts, both inside the first 400 steps

| model | n seq | heading north at 30: real / model / agreement | far entry by 100: real / model / agreement | reach by 400: real / model / agreement | stalled in the last 100: real / model / agreement |
|---|---:|---|---|---|---|
| v4 | 38 | 0.13 / 0.16 / 0.97 | 0.16 / 0.18 / 0.97 | 0.42 / 0.87 / 0.55 | 0.47 / 0.00 / 0.53 |
| v4a | 38 | 0.13 / 0.18 / 0.95 | 0.16 / 0.18 / 0.97 | 0.42 / 0.87 / 0.55 | 0.47 / 0.03 / 0.55 |
| v4b | 38 | 0.13 / 0.13 / 0.95 | 0.16 / 0.13 / 0.97 | 0.42 / 0.82 / 0.55 | 0.47 / 0.05 / 0.47 |

## The original validation at the selected step (no-regression on the start-agent / logged rows) and the CF val

| model | fold | step | val mse diag / off | onset bce / auroc | stat acc | CF val mse moving / all | CF val stat acc |
|---|---|---:|---|---|---:|---|---:|
| v4 | fold0 | 39000 | 0.0129 / 0.0332 | 0.0043 / 0.9978 | 0.9874 | - / - | - |
| v4 | fold1 | 40000 | 0.0137 / 0.0333 | 0.0050 / 0.9973 | 0.9828 | - / - | - |
| v4 | fold2 | 40000 | 0.0097 / 0.0281 | 0.0049 / 0.9973 | 0.9881 | - / - | - |
| v4a | fold0 | 10000 | 0.0127 / 0.0332 | 0.0043 / 0.9978 | 0.9873 | 0.1187 / 0.0830 | 0.9925 |
| v4a | fold1 | 9000 | 0.0132 / 0.0335 | 0.0050 / 0.9973 | 0.9850 | 0.1133 / 0.0843 | 0.9904 |
| v4a | fold2 | 10000 | 0.0096 / 0.0283 | 0.0049 / 0.9973 | 0.9890 | 0.1142 / 0.0799 | 0.9778 |
| v4b | fold0 | 10000 | 0.0152 / 0.0362 | 0.0043 / 0.9978 | 0.9876 | 0.1264 / 0.0881 | 0.9921 |
| v4b | fold1 | 10000 | 0.0158 / 0.0359 | 0.0050 / 0.9973 | 0.9869 | 0.1186 / 0.0856 | 0.9921 |
| v4b | fold2 | 10000 | 0.0117 / 0.0305 | 0.0049 / 0.9973 | 0.9878 | 0.1212 / 0.0847 | 0.9898 |
