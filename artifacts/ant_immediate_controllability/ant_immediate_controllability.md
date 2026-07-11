# Ant immediate local controllability probe

**Verdict: `LOCAL_ACTIONS_PHYSICALLY_INDISTINGUISHABLE`** (ckpt step 150500, 110 states, 128 candidates/sigma)

Protocol: restore exact state; candidate 1 step; +2 zero-action steps (persistence); actor never resumes.

## Gates
- gateA_restore: pass=True
- gateB_obs_consistency: pass=True
- gateC_repr_consistency: pass=True
- gateD_orientation: pass=True
- orientation: validated quaternion (matches xmat)
- max |ctrl - clipped action| across all rollouts: 0.00e+00
- contact data available: True

Actor 1-step displacement median: 0 m; actor prog1 mean 0.002214 m; actor fall@3 0.00

Jacobian proxy |d proj1 / d a|: median norm 0.000483 m/unit (p10 0.000483, p90 0.001661)

## Physical spread per candidate set (within-state, 1 step)
| set | std proj1 | rng prog1 | std dvxy | std djvel | rng prog3 | fall@3 | label |
|---|---|---|---|---|---|---|---|
| local_s0.01 | 3.804e-06 | 1.997e-05 | 5.693e-05 | 0.0002171 | 2.551e-05 | 0.00 | negligible |
| local_s0.03 | 1.127e-05 | 5.992e-05 | 0.0001663 | 0.0006463 | 7.065e-05 | 0.00 | negligible |
| local_s0.05 | 1.828e-05 | 9.79e-05 | 0.0002766 | 0.001088 | 0.0001331 | 0.00 | negligible |
| local_s0.1 | 3.638e-05 | 0.0001914 | 0.000554 | 0.002204 | 0.0002518 | 0.00 | negligible |
| local_s0.2 | 0.0001503 | 0.001471 | 0.006907 | 0.07535 | 0.005648 | 0.00 | negligible |
| replay_nbr | 0.002344 | 0.006139 | 0.1765 | 1.19 | 0.02307 | 0.00 | - |
| uniform | 0.005333 | 0.02995 | 0.3177 | 2.549 | 0.1649 | 0.00 | - |

Thresholds: negligible if std proj1 < 0.002 and rng prog1 < 0.01; meaningful if std proj1 >= 0.005 or rng prog1 >= 0.02 (m).

## Critic ranking per set (immediate physics)
| set | sp(prog1) med | frac>0 | sp(proj1) | sp(dvxy) | sp(proj3) | critic dec gap | true dec gap | usefulness | cb prog1 | cw prog1 | rand | actor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local_s0.01 | 0.257 | 0.94 | 0.257 | 0.249 | -0.011 | 5.482e-06 [2.3e-06,9.066e-06] | 3.197e-05 | 0.17 | 0.002214 | 0.002201 | 0.002212 | 0.002214 |
| local_s0.03 | 0.297 | 0.94 | 0.297 | 0.286 | 0.073 | 1.372e-05 [-1.41e-06,2.801e-05] | 8.89e-05 | 0.15 | 0.002219 | 0.002184 | 0.002206 | 0.002214 |
| local_s0.05 | 0.311 | 0.94 | 0.309 | 0.301 | 0.106 | 2.448e-05 [5.867e-06,4.188e-05] | 0.0001474 | 0.17 | 0.002205 | 0.002164 | 0.002207 | 0.002214 |
| local_s0.1 | 0.321 | 0.94 | 0.321 | 0.310 | 0.094 | 4.888e-05 [2.632e-05,7.638e-05] | 0.0002765 | 0.18 | 0.002227 | 0.002136 | 0.002182 | 0.002214 |
| local_s0.2 | 0.353 | 0.94 | 0.353 | 0.350 | 0.115 | 0.0002035 [0.0001558,0.0002592] | 0.0006976 | 0.29 | 0.00249 | 0.002077 | 0.002197 | 0.002214 |
| replay_nbr | -0.624 | 0.04 | -0.624 | -0.697 | -0.564 | -0.004379 [-0.004785,-0.003996] | 0.006521 | -0.67 | -0.003012 | 0.001367 | -0.0005493 | 0.002214 |
| uniform | -0.021 | 0.43 | -0.020 | 0.004 | 0.027 | -0.0006418 [-0.001171,-6.013e-05] | 0.01959 | -0.03 | -0.003211 | -0.001094 | -0.002122 | 0.002214 |

## Candidate diversity / clipping
| set | mean dist actor | per-dim std | clip frac | raw clip frac | eff rank | mean score |
|---|---|---|---|---|---|---|
| local_s0.01 | 0.024 | 0.0081 | 0.229 | 0.229 | 6.39 | -14.37 |
| local_s0.03 | 0.067 | 0.0228 | 0.284 | 0.284 | 6.24 | -14.37 |
| local_s0.05 | 0.110 | 0.0369 | 0.307 | 0.307 | 6.15 | -14.37 |
| local_s0.1 | 0.214 | 0.0713 | 0.337 | 0.337 | 6.13 | -14.36 |
| local_s0.2 | 0.423 | 0.1396 | 0.357 | 0.357 | 5.98 | -14.40 |
| replay_nbr | 3.007 | 0.3149 | 0.125 | None | 1.18 | -15.22 |
| uniform | 2.950 | 0.5755 | 0.000 | None | 7.57 | -12.81 |

Replay neighbors: k=10, mean obs dist 4.77, action std 0.327, dist from actor 2.98

## Score-scale artifacts (Control 4; repr_norm=False, score = raw dot)
| set | phi-norm std | phi-norm mean | cos std | sp(dot,cos) | sp(cos,prog1) | sp(dot,prog1) |
|---|---|---|---|---|---|---|
| local_s0.01 | 0.017 | 11.33 | 0.0004 | 0.751 | 0.141 | 0.257 |
| local_s0.03 | 0.047 | 11.32 | 0.0010 | 0.781 | 0.265 | 0.297 |
| local_s0.05 | 0.074 | 11.31 | 0.0017 | 0.795 | 0.296 | 0.311 |
| local_s0.1 | 0.139 | 11.27 | 0.0033 | 0.805 | 0.304 | 0.321 |
| local_s0.2 | 0.261 | 11.23 | 0.0065 | 0.822 | 0.312 | 0.353 |
| replay_nbr | 0.671 | 9.28 | 0.0545 | 0.988 | -0.648 | -0.624 |
| uniform | 0.816 | 10.34 | 0.0236 | 0.773 | -0.035 | -0.021 |

psi(g) norm mean: 6.02

## Persistence (1 candidate step + 2 zero steps)
| set | sp(proj1,proj3) med | sign persistence (top |proj1| decile) |
|---|---|---|
| local_s0.01 | 0.197 | 1.00 |
| local_s0.03 | 0.130 | 1.00 |
| local_s0.05 | 0.133 | 1.00 |
| local_s0.1 | 0.101 | 1.00 |
| local_s0.2 | 0.138 | 0.92 |
| replay_nbr | 0.939 | 1.00 |
| uniform | 0.925 | 1.00 |

## Control 1: action distance vs effect (local sets)
| set | dist | std proj1 | fall@3 | sp(dist,score) | sp(dist,|dproj|) |
|---|---|---|---|---|---|
| local_s0.01 | 0.024 | 3.804e-06 | 0.00 | -0.050 | 0.370 |
| local_s0.03 | 0.067 | 1.127e-05 | 0.00 | -0.025 | 0.356 |
| local_s0.05 | 0.110 | 1.828e-05 | 0.00 | 0.022 | 0.359 |
| local_s0.1 | 0.214 | 3.638e-05 | 0.00 | -0.010 | 0.389 |
| local_s0.2 | 0.423 | 0.0001503 | 0.00 | -0.053 | 0.410 |

## Control 3: state-regime split (at local sigma=0.05)
| regime | n | std proj1 | rng prog1 | sp(prog1) med | critic gap [ci] | true gap |
|---|---|---|---|---|---|---|
| standing_moving | 55 | 1.895e-05 | 0.0001012 | 0.322 | 3.007e-05 [-5.772e-06,6.434e-05] | 0.00023 |
| standing_stationary | 55 | 1.803e-05 | 9.007e-05 | 0.303 | 1.889e-05 [1.719e-05,2.061e-05] | 6.476e-05 |
| low_torso | 0 | - | - | - | - | - |
| falling_vz | 8 | 8.411e-05 | 0.0004616 | 0.113 | -5.483e-05 [-0.0002217,7.232e-05] | 0.0004957 |
| near_goal | 93 | 1.805e-05 | 9.354e-05 | 0.313 | 2.665e-05 [1.773e-05,4.008e-05] | 8.173e-05 |
| far_goal | 17 | 8.394e-05 | 0.0003879 | 0.244 | 1.259e-05 [-8.793e-05,9.747e-05] | 0.0005064 |
| high_sat | 6 | 0.0001947 | 0.001183 | 0.323 | 0.0001208 [3.356e-05,0.000224] | 0.0006291 |
| low_sat | 104 | 1.808e-05 | 9.59e-05 | 0.309 | 1.892e-05 [4.656e-07,3.698e-05] | 0.0001196 |
| fast_spin | 55 | 1.895e-05 | 0.0001012 | 0.322 | 3.016e-05 [-4.729e-06,6.428e-05] | 0.0002299 |
| goal_aligned_heading | 0 | - | - | - | - | - |
| goal_misaligned_heading | 110 | 1.828e-05 | 9.79e-05 | 0.311 | 2.448e-05 [5.867e-06,4.188e-05] | 0.0001474 |
| moving_toward_goal | 6 | 0.0002478 | 0.001382 | 0.175 | 0.0001194 [3.269e-05,0.0002279] | 0.0009616 |

Regimes passing the validity bar: none

Primary ranking set: `local_s0.05` -> {"valid": false, "fails": false, "spearman": 0.3114070225233474, "frac_pos": 0.9363636363636364, "gap": 2.448021391842744e-05, "gap_ci": [5.866630256883474e-06, 4.1877584613968576e-05], "usefulness": 0.16612126207264039}

**Verdict: `LOCAL_ACTIONS_PHYSICALLY_INDISTINGUISHABLE`**

## State-coverage audit (post-hoc, honest disclosure)
Only **20 unique** (qpos,qvel) states among the 110 saved states; **91/110 are bit-identical copies of one resting pose** (the ant settles into static equilibrium; |v_xy| ~ 1e-15) and all states share **one goal**. Bootstrap CIs above are therefore overconfident and the moving/stationary/heading regime splits are degenerate. Key stats recomputed on the deduplicated subsets:

| set | std proj1 (uniq) | rng prog1 (uniq) | sp (uniq) | usefulness (uniq) | std proj1 (moving) | rng prog1 (moving) | sp (moving) |
|---|---|---|---|---|---|---|---|
| local_s0.01 | 2.3e-05 | 0.000124 | 0.213 | 0.12 | 3.03e-05 | 0.000169 | 0.174 |
| local_s0.03 | 6.5e-05 | 0.000306 | 0.222 | 0.07 | 9.33e-05 | 0.000436 | 0.153 |
| local_s0.05 | 8.41e-05 | 0.000413 | 0.298 | 0.10 | 0.000127 | 0.00076 | 0.175 |
| local_s0.1 | 0.00019 | 0.00112 | 0.156 | 0.08 | 0.000279 | 0.00139 | 0.075 |
| local_s0.2 | 0.000357 | 0.00215 | 0.182 | 0.12 | 0.000529 | 0.00216 | 0.106 |
| replay_nbr | 0.00234 | 0.00707 | -0.491 | -0.40 | 0.00275 | 0.00856 | -0.212 |
| uniform | 0.00537 | 0.027 | 0.013 | 0.03 | 0.00537 | 0.026 | 0.080 |

(unique n=20, moving n=16 with |v_xy|>0.05)

The verdict is **unchanged** on the deduplicated subsets: every local sigma stays >=10x below the meaningful-spread thresholds while uniform actions exceed them, and the critic decile gap remains ~0.1 of the (already micron-scale) true physical range. The same saved-state file fed the earlier forensic-audit and local-ranking probes, so their nominal state counts carry the same redundancy.
