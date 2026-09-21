# The critics' twin-min logit along the action path forward -> stall at the 64 reset states, by goal set

Path a(l) = (1 - l) a_forward + l a_stall, l = 0 .. 1 in 11 steps; a_stall = the stalled policy's mode (outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_futures_v4s20/CF/seed_4/final.pkl), a_forward = the progressing policy's mode (outputs/antmaze_branch_replay_p050/exp_mainline_pilot/ett_futures_v4s20_draw2/CF/seed_4/final.pkl), the start agent's mode, or the logged first torque.  Goal sets per state: the task goal; 32 goals within 2.0 of it; 16 relabelled future goals of the anchor's logged episode (the actor stream's law); 256 goals from each table's critic-training marginal (anchors by weight, the gamma law).  Bump = Q(l = 1) - Q(l = 0) per state, mean over states; 'stall higher' = the share of states with a positive bump.

| action path | critic | goal set | Q at l = 0 / 0.5 / 1 (mean) | bump mean +- se | stall higher (share) | argmax l mean | argmax at the stall end / forward end / interior |
|---|---|---|---|---|---:|---:|---|
| fwd(d2 final) -> stall(d1 final) | d1_s4 | task | -9.13 / -7.89 / -7.00 | +2.139 +- 0.125 | 1.00 | 0.98 | 0.95 / 0.00 / 0.05 |
| fwd(d2 final) -> stall(d1 final) | d1_s4 | near2.0 | -9.39 / -8.16 / -7.23 | +2.161 +- 0.101 | 1.00 | 0.99 | 0.98 / 0.00 / 0.02 |
| fwd(d2 final) -> stall(d1 final) | d1_s4 | relabelled (actor stream) | -7.79 / -7.19 / -7.31 | +0.481 +- 0.131 | 0.66 | 0.61 | 0.33 / 0.03 / 0.64 |
| fwd(d2 final) -> stall(d1 final) | d1_s4 | marginal draw1 (critic training) | -8.56 / -7.58 / -7.26 | +1.298 +- 0.087 | 0.95 | 0.89 | 0.73 / 0.00 / 0.27 |
| fwd(d2 final) -> stall(d1 final) | d1_s4 | marginal draw2 (critic training) | -8.64 / -7.57 / -7.14 | +1.509 +- 0.086 | 0.98 | 0.92 | 0.81 / 0.00 / 0.19 |
| fwd(d2 final) -> stall(d1 final) | d2_s4 | task | -8.59 / -9.44 / -10.43 | -1.844 +- 0.169 | 0.09 | 0.12 | 0.06 / 0.78 / 0.16 |
| fwd(d2 final) -> stall(d1 final) | d2_s4 | near2.0 | -8.85 / -9.75 / -10.83 | -1.979 +- 0.158 | 0.05 | 0.06 | 0.00 / 0.86 / 0.14 |
| fwd(d2 final) -> stall(d1 final) | d2_s4 | relabelled (actor stream) | -7.61 / -7.94 / -8.89 | -1.278 +- 0.088 | 0.02 | 0.17 | 0.02 / 0.56 / 0.42 |
| fwd(d2 final) -> stall(d1 final) | d2_s4 | marginal draw1 (critic training) | -8.44 / -9.10 / -10.34 | -1.900 +- 0.093 | 0.00 | 0.08 | 0.00 / 0.73 / 0.27 |
| fwd(d2 final) -> stall(d1 final) | d2_s4 | marginal draw2 (critic training) | -8.27 / -8.92 / -10.10 | -1.834 +- 0.096 | 0.02 | 0.10 | 0.00 / 0.69 / 0.31 |
| fwd(d2 final) -> stall(d1 final) | d1_s4_20k | task | -8.34 / -7.63 / -7.14 | +1.205 +- 0.066 | 1.00 | 0.99 | 0.98 / 0.00 / 0.02 |
| fwd(d2 final) -> stall(d1 final) | d1_s4_20k | near2.0 | -8.48 / -7.75 / -7.29 | +1.184 +- 0.073 | 0.98 | 0.97 | 0.92 / 0.00 / 0.08 |
| fwd(d2 final) -> stall(d1 final) | d1_s4_20k | relabelled (actor stream) | -7.50 / -7.29 / -7.51 | -0.014 +- 0.109 | 0.45 | 0.49 | 0.27 / 0.25 / 0.48 |
| fwd(d2 final) -> stall(d1 final) | d1_s4_20k | marginal draw1 (critic training) | -8.09 / -7.62 / -7.53 | +0.563 +- 0.058 | 0.89 | 0.77 | 0.45 / 0.03 / 0.52 |
| fwd(d2 final) -> stall(d1 final) | d1_s4_20k | marginal draw2 (critic training) | -8.20 / -7.66 / -7.49 | +0.712 +- 0.054 | 0.95 | 0.87 | 0.69 / 0.03 / 0.28 |
| start agent -> stall | d1_s4 | task | -11.22 / -8.86 / -7.00 | +4.221 +- 0.115 | 1.00 | 0.97 | 0.97 / 0.00 / 0.03 |
| start agent -> stall | d1_s4 | near2.0 | -11.67 / -9.14 / -7.23 | +4.436 +- 0.071 | 1.00 | 0.96 | 0.94 / 0.00 / 0.06 |
| start agent -> stall | d1_s4 | relabelled (actor stream) | -8.64 / -7.83 / -7.31 | +1.336 +- 0.155 | 0.89 | 0.78 | 0.42 / 0.06 / 0.52 |
| start agent -> stall | d1_s4 | marginal draw1 (critic training) | -10.21 / -8.59 / -7.26 | +2.953 +- 0.064 | 1.00 | 0.94 | 0.88 / 0.00 / 0.12 |
| start agent -> stall | d1_s4 | marginal draw2 (critic training) | -10.32 / -8.58 / -7.14 | +3.181 +- 0.068 | 1.00 | 0.94 | 0.89 / 0.00 / 0.11 |
| start agent -> stall | d2_s4 | task | -11.36 / -9.35 / -10.43 | +0.920 +- 0.108 | 0.83 | 0.62 | 0.06 / 0.00 / 0.94 |
| start agent -> stall | d2_s4 | near2.0 | -11.77 / -9.65 / -10.83 | +0.940 +- 0.112 | 0.80 | 0.61 | 0.08 / 0.00 / 0.92 |
| start agent -> stall | d2_s4 | relabelled (actor stream) | -8.87 / -7.89 / -8.89 | -0.022 +- 0.093 | 0.47 | 0.52 | 0.02 / 0.08 / 0.91 |
| start agent -> stall | d2_s4 | marginal draw1 (critic training) | -10.83 / -9.13 / -10.34 | +0.492 +- 0.084 | 0.80 | 0.55 | 0.00 / 0.02 / 0.98 |
| start agent -> stall | d2_s4 | marginal draw2 (critic training) | -10.56 / -8.90 / -10.10 | +0.457 +- 0.093 | 0.75 | 0.55 | 0.00 / 0.02 / 0.98 |
| start agent -> stall | d1_s4_20k | task | -9.94 / -8.14 / -7.14 | +2.805 +- 0.046 | 1.00 | 0.95 | 0.86 / 0.00 / 0.14 |
| start agent -> stall | d1_s4_20k | near2.0 | -10.15 / -8.29 / -7.29 | +2.857 +- 0.041 | 1.00 | 0.94 | 0.81 / 0.00 / 0.19 |
| start agent -> stall | d1_s4_20k | relabelled (actor stream) | -8.05 / -7.57 / -7.51 | +0.538 +- 0.103 | 0.78 | 0.70 | 0.27 / 0.09 / 0.64 |
| start agent -> stall | d1_s4_20k | marginal draw1 (critic training) | -9.16 / -8.10 / -7.53 | +1.632 +- 0.032 | 1.00 | 0.89 | 0.62 / 0.00 / 0.38 |
| start agent -> stall | d1_s4_20k | marginal draw2 (critic training) | -9.22 / -8.11 / -7.49 | +1.735 +- 0.033 | 1.00 | 0.90 | 0.66 / 0.00 / 0.34 |
| logged torque -> stall | d1_s4 | task | -9.42 / -8.05 / -7.00 | +2.425 +- 0.106 | 1.00 | 0.95 | 0.84 / 0.00 / 0.16 |
| logged torque -> stall | d1_s4 | near2.0 | -9.84 / -8.37 / -7.23 | +2.605 +- 0.101 | 0.98 | 0.97 | 0.94 / 0.00 / 0.06 |
| logged torque -> stall | d1_s4 | relabelled (actor stream) | -6.99 / -6.70 / -7.31 | -0.315 +- 0.119 | 0.36 | 0.48 | 0.11 / 0.16 / 0.73 |
| logged torque -> stall | d1_s4 | marginal draw1 (critic training) | -8.08 / -7.30 / -7.26 | +0.818 +- 0.073 | 0.92 | 0.73 | 0.28 / 0.00 / 0.72 |
| logged torque -> stall | d1_s4 | marginal draw2 (critic training) | -8.07 / -7.23 / -7.14 | +0.930 +- 0.073 | 0.94 | 0.77 | 0.31 / 0.00 / 0.69 |
| logged torque -> stall | d2_s4 | task | -10.26 / -10.24 / -10.43 | -0.173 +- 0.169 | 0.50 | 0.50 | 0.36 / 0.36 / 0.28 |
| logged torque -> stall | d2_s4 | near2.0 | -10.55 / -10.56 / -10.83 | -0.283 +- 0.151 | 0.47 | 0.43 | 0.25 / 0.39 / 0.36 |
| logged torque -> stall | d2_s4 | relabelled (actor stream) | -7.46 / -8.01 / -8.89 | -1.425 +- 0.096 | 0.03 | 0.10 | 0.00 / 0.78 / 0.22 |
| logged torque -> stall | d2_s4 | marginal draw1 (critic training) | -9.17 / -9.56 / -10.34 | -1.172 +- 0.093 | 0.09 | 0.16 | 0.02 / 0.67 / 0.31 |
| logged torque -> stall | d2_s4 | marginal draw2 (critic training) | -8.84 / -9.35 / -10.10 | -1.261 +- 0.094 | 0.05 | 0.15 | 0.02 / 0.72 / 0.27 |
| logged torque -> stall | d1_s4_20k | task | -9.14 / -8.09 / -7.14 | +2.002 +- 0.109 | 1.00 | 0.97 | 0.88 / 0.00 / 0.12 |
| logged torque -> stall | d1_s4_20k | near2.0 | -9.27 / -8.21 / -7.29 | +1.980 +- 0.093 | 0.98 | 0.97 | 0.91 / 0.02 / 0.08 |
| logged torque -> stall | d1_s4_20k | relabelled (actor stream) | -7.00 / -6.83 / -7.51 | -0.515 +- 0.098 | 0.23 | 0.44 | 0.08 / 0.08 / 0.84 |
| logged torque -> stall | d1_s4_20k | marginal draw1 (critic training) | -8.00 / -7.47 / -7.53 | +0.475 +- 0.039 | 0.95 | 0.69 | 0.27 / 0.00 / 0.73 |
| logged torque -> stall | d1_s4_20k | marginal draw2 (critic training) | -8.03 / -7.48 / -7.49 | +0.547 +- 0.040 | 0.95 | 0.75 | 0.34 / 0.00 / 0.66 |

Mean action gaps: fwd(d2 final) -> stall(d1 final): 0.635, start agent -> stall: 1.347, logged torque -> stall: 0.770
