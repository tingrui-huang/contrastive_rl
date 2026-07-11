# AntMaze action-validity probe

**Verdict: `CRITIC_ACTION_RANKING_INVALID`** (primary horizon = 5 steps)

checkpoint step 150500, 110 reference states, 130 candidate actions/state.

## Gates
- Gate 1 exact restore: pass=True (dXY=8.88e-16, dz=1.11e-16)
- Gate 2 action effect: pass=True (mean disp diff 0.1265)
- Gate 3 critic dependence: action_insensitive=False (median score std 1.5249, range 7.5293)
- Gate 4 progress sign: pass=True

## Per-horizon (progress = d_before - d_after, +=closer)
| horizon | cb>rand | cb>worst | actor>rand | spearman(med) | sp>0 frac | cb mean | rand mean | actor mean | prog range |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.26 | 0.23 | 0.69 | -0.18343006186073824 | 0.11818181818181818 | -0.0036 | 0.0005 | 0.0022 | 0.0269 |
| 5 | 0.24 | 0.38 | 0.65 | -0.1170410629668574 | 0.16363636363636364 | -0.0616 | -0.0098 | 0.0126 | 0.3630 |
| 10 | 0.35 | 0.43 | 0.68 | -0.09650548280052984 | 0.2 | -0.0895 | -0.0218 | 0.0172 | 0.6901 |

actor critic-score mean -13.83 vs critic-best -9.50 (actor below best in 1.00)

## Stratified mean progress @5 (critic_best / random / actor)
- near (n=93): -0.08551993864858047 / -0.029174198339372986 / -0.0026235630856099087
- far (n=17): 0.06951522313378397 / 0.09616529227817716 / 0.09611941404894485
- standing (n=110): -0.06155995910039687 / -0.009803549789387965 / 0.01263671519882128
- low_torso (n=0): None / None / None
- low_sat (n=104): -0.07158610349670481 / -0.02070961494480585 / 0.004880866230930096
- high_sat (n=6): 0.11222654376894074 / 0.17923491290452206 / 0.14707143064226846
- aligned (n=0): None / None / None
- misaligned (n=110): -0.06155995910039687 / -0.009803549789387965 / 0.01263671519882128
