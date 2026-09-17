# Detour-corridor completion: generation walker vs deployment BC walker (paired hazards, same placed rows)

BC actor: /root/contrastive_rl/outputs/antmaze_branch_replay_p050/joint_purebc/seed_0/final.pkl

| leg | walker | n | reach | death | timeout | reach step (median) | P_goal |
|---|---|---:|---:|---:|---:|---:|---:|
| north | generation (frozen walker + blind driver) | 100 | 0.95 | 0.00 | 0.05 | 322.0 | 0.434 |
| north | deployment (BC mode) | 100 | 0.83 | 0.00 | 0.17 | 380.0 | 0.256 |
| east | generation (frozen walker + blind driver) | 100 | 0.98 | 0.00 | 0.02 | 183.5 | 0.604 |
| east | deployment (BC mode) | 100 | 0.92 | 0.00 | 0.08 | 235.5 | 0.402 |
