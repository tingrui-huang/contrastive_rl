# The three-layer acceptance of the CF-motion revision (test split of the CF-controlled rollouts; no training)

Models {'v4': '/root/contrastive_rl/outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_one_step_v4', 'v4c': '/root/contrastive_rl/outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_one_step_v4c', 'v4r': '/root/contrastive_rl/outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_one_step_v4r'}; test rollouts 174, rows 39416.

## L1 -- teacher forcing (real state every step): one-step xy error (median / p90) by row type and phase; the stationary gate; slow rows

| model | moving | slow | static | turn30 | far | corridor | zone | gate fires on static / slow / moving | slow rows: real speed / pred speed / pred frozen share | static rows pred speed |
|---|---|---|---|---|---|---|---|---|---|---|
| v4 | 0.0063 / 0.0236 (n 21780) | 0.0206 / 0.0293 (n 8083) | 0.0000 / 0.0000 (n 9553) | 0.0147 / 0.0340 (n 840) | 0.0187 / 0.0312 (n 12116) | 0.0031 / 0.0138 (n 20006) | 0.0015 / 0.0143 (n 6454) | 0.970 / 0.092 / 0.000 | 0.0034 / 0.0187 / 0.09 | 0.0000 |
| v4c | 0.0063 / 0.0238 (n 21780) | 0.0218 / 0.0308 (n 8083) | 0.0000 / 0.0000 (n 9553) | 0.0156 / 0.0345 (n 840) | 0.0191 / 0.0321 (n 12116) | 0.0031 / 0.0136 (n 20006) | 0.0014 / 0.0145 (n 6454) | 0.970 / 0.066 / 0.000 | 0.0034 / 0.0203 / 0.07 | 0.0000 |
| v4r | 0.0053 / 0.0181 (n 21780) | 0.0028 / 0.0081 (n 8083) | 0.0000 / 0.0000 (n 9553) | 0.0119 / 0.0281 (n 840) | 0.0031 / 0.0180 (n 12116) | 0.0028 / 0.0110 (n 20006) | 0.0014 / 0.0112 (n 6454) | 0.970 / 0.002 / 0.000 | 0.0034 / 0.0042 / 0.00 | 0.0000 |

## L2 -- open loop (real action sequence, model rolled forward; onset off): xy error median / p90 at steps; stall segments

| model | step 5 | step 10 | step 20 | step 30 | step 50 | step 100 | pose err median at 30 / 100 | vel err median at 30 / 100 | stall segments (n, len): real disp / model disp median; share model disp > 0.5 / > 2.0 (real > 0.5) |
|---|---|---|---|---|---|---|---|---|---|
| v4 | 0.039 / 0.118 | 0.098 / 0.322 | 0.199 / 0.699 | 0.366 / 1.039 | 0.638 / 2.144 | 2.099 / 5.150 | 0.037 / 0.077 | 0.299 / 0.452 | (14, 26): 0.05 / 0.43; 0.43 / 0.14 (0.00) |
| v4c | 0.036 / 0.096 | 0.096 / 0.296 | 0.220 / 0.676 | 0.400 / 0.971 | 0.861 / 2.131 | 2.245 / 5.703 | 0.036 / 0.076 | 0.280 / 0.414 | (14, 26): 0.05 / 0.47; 0.43 / 0.14 (0.00) |
| v4r | 0.035 / 0.105 | 0.084 / 0.292 | 0.236 / 0.598 | 0.427 / 0.883 | 0.814 / 1.935 | 2.138 / 5.165 | 0.036 / 0.081 | 0.266 / 0.398 | (14, 26): 0.05 / 0.67; 0.71 / 0.00 (0.00) |

## L3 -- closed loop (CF actor on the model state; onset off) vs the real hazard-free rollouts

| model | n seq | heading north at 30: real / model / agreement | far entry by 100: real / model / agreement | reach by 400: real / model / agreement | stalled in the last 100: real / model / agreement |
|---|---:|---|---|---|---|
| v4 | 31 | 0.06 / 0.10 / 0.97 | 0.10 / 0.10 / 1.00 | 0.61 / 0.97 / 0.65 | 0.39 / 0.00 / 0.61 |
| v4c | 31 | 0.06 / 0.03 / 0.97 | 0.10 / 0.03 / 0.94 | 0.61 / 0.97 / 0.65 | 0.39 / 0.03 / 0.65 |
| v4r | 31 | 0.06 / 0.06 / 0.94 | 0.10 / 0.06 / 0.97 | 0.61 / 0.94 / 0.61 | 0.39 / 0.00 / 0.61 |

## The original validation at the selected step (no-regression on the start-agent / logged rows) and the CF val

| model | fold | step | val mse diag / off | onset bce / auroc | stat acc | CF val mse moving / all | CF val stat acc |
|---|---|---:|---|---|---:|---|---:|
| v4 | fold0 | 39000 | 0.0129 / 0.0332 | 0.0043 / 0.9978 | 0.9874 | - / - | - |
| v4 | fold1 | 40000 | 0.0137 / 0.0333 | 0.0050 / 0.9973 | 0.9828 | - / - | - |
| v4 | fold2 | 40000 | 0.0097 / 0.0281 | 0.0049 / 0.9973 | 0.9881 | - / - | - |
| v4c | fold0 | 9000 | 0.0127 / 0.0325 | 0.0043 / 0.9978 | 0.9868 | 0.1627 / 0.1349 | 0.9827 |
| v4c | fold1 | 10000 | 0.0131 / 0.0322 | 0.0050 / 0.9973 | 0.9833 | 0.1492 / 0.1229 | 0.9884 |
| v4c | fold2 | 6000 | 0.0094 / 0.0274 | 0.0049 / 0.9973 | 0.9878 | 0.1459 / 0.1099 | 0.9873 |
| v4r | fold0 | 9000 | 0.0129 / 0.0335 | 0.0043 / 0.9978 | 0.9887 | 0.0996 / 0.0734 | 0.9903 |
| v4r | fold1 | 9000 | 0.0135 / 0.0330 | 0.0050 / 0.9973 | 0.9878 | 0.0945 / 0.0704 | 0.9903 |
| v4r | fold2 | 9000 | 0.0096 / 0.0280 | 0.0049 / 0.9973 | 0.9888 | 0.0966 / 0.0713 | 0.9861 |
