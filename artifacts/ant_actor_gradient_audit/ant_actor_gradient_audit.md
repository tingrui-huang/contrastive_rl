# Actor-gradient validity audit (30k target-entropy -8 checkpoints)


## Arm xy ¡ª **`CRITIC_GRADIENT_SCORE_ONLY`** (step 30000, 30 states)

grad norm median 0.733 (p10 0.462, p90 1.154); participation ratio 5.3/8; top-dim share 0.28; saturated dims 0.12; blocked outward mass 0.00; finite-diff rel err 0.002

| delta | df+ mean | df+>0 | prog1 margin(+/-) | prog5 margin(+/-) | prog5 margin(+/rand) | fell5 + | fell5 rand |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.0138 | 1.00 | +0.00000 [-0.00003,0.00004] | +0.00007 [-0.00074,0.00078] | +0.00016 [-0.00016,0.00048] | 0.00 | 0.00 |
| 0.05 | 0.0325 | 1.00 | +0.00002 [-0.00006,0.00011] | +0.00073 [-0.00087,0.00246] | +0.00060 [0.00001,0.00128] | 0.00 | 0.00 |
| 0.1 | 0.0621 | 1.00 | +0.00010 [-0.00008,0.00030] | +0.00186 [-0.00028,0.00480] | +0.00046 [-0.00025,0.00119] | 0.00 | 0.00 |

score-progress coupling: sp(df,prog1)=0.065, sp(df,prog5)=0.049

actor step x1: df_mean 0.0827 (pos frac 0.83), |da| med 0.21480, dprog5 +0.00251
actor steps x50 (probe copy): df_mean 0.545 (pos 1.00), |da| med 1.7978, sat 0.10->0.07, dprog5 -0.00586 (-0.030536914448728948, 0.02065581254362836), sp(df,dprog5) -0.04382647385984427

## Arm gcompact ¡ª **`CRITIC_GRADIENT_SCORE_ONLY`** (step 30000, 30 states)

grad norm median 0.623 (p10 0.497, p90 1.055); participation ratio 5.4/8; top-dim share 0.27; saturated dims 0.12; blocked outward mass 0.14; finite-diff rel err 0.005

| delta | df+ mean | df+>0 | prog1 margin(+/-) | prog5 margin(+/-) | prog5 margin(+/rand) | fell5 + | fell5 rand |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.0119 | 1.00 | +0.00000 [-0.00001,0.00002] | -0.00040 [-0.00088,-0.00000] | -0.00011 [-0.00070,0.00040] | 0.03 | 0.03 |
| 0.05 | 0.0266 | 1.00 | +0.00001 [-0.00003,0.00004] | -0.00028 [-0.00139,0.00062] | -0.00001 [-0.00075,0.00056] | 0.03 | 0.03 |
| 0.1 | 0.0456 | 1.00 | +0.00004 [-0.00005,0.00013] | -0.00164 [-0.00428,0.00039] | -0.00094 [-0.00316,0.00069] | 0.03 | 0.03 |

score-progress coupling: sp(df,prog1)=0.045, sp(df,prog5)=-0.192

actor step x1: df_mean 0.0353 (pos frac 0.97), |da| med 0.15222, dprog5 -0.00126
actor steps x50 (probe copy): df_mean 0.385 (pos 1.00), |da| med 1.2512, sat 0.13->0.05, dprog5 -0.02184 (-0.04071746421326408, -0.003917569375838157), sp(df,dprog5) -0.4122358175750834

## Arm gfull ¡ª **`CRITIC_GRADIENT_SCORE_ONLY`** (step 30000, 30 states)

grad norm median 0.697 (p10 0.485, p90 1.171); participation ratio 5.1/8; top-dim share 0.29; saturated dims 0.00; blocked outward mass 0.00; finite-diff rel err 0.003

| delta | df+ mean | df+>0 | prog1 margin(+/-) | prog5 margin(+/-) | prog5 margin(+/rand) | fell5 + | fell5 rand |
|---|---|---|---|---|---|---|---|
| 0.02 | 0.0141 | 1.00 | +0.00001 [-0.00004,0.00005] | -0.00095 [-0.00254,0.00029] | -0.00066 [-0.00194,0.00027] | 0.00 | 0.00 |
| 0.05 | 0.0331 | 1.00 | +0.00001 [-0.00011,0.00012] | -0.00174 [-0.00351,-0.00030] | -0.00027 [-0.00088,0.00027] | 0.00 | 0.00 |
| 0.1 | 0.0624 | 1.00 | +0.00005 [-0.00019,0.00029] | -0.00052 [-0.00201,0.00094] | -0.00031 [-0.00162,0.00097] | 0.00 | 0.00 |

score-progress coupling: sp(df,prog1)=0.027, sp(df,prog5)=0.002

actor step x1: df_mean 0.0654 (pos frac 0.80), |da| med 0.22782, dprog5 +0.00080
actor steps x50 (probe copy): df_mean 0.517 (pos 1.00), |da| med 1.7584, sat 0.01->0.03, dprog5 -0.01362 (-0.0484819165881294, 0.02020712096257593), sp(df,dprog5) -0.03136818687430478

(all rollouts from bit-exact restored states; +/-grad and random directions norm-matched before clipping; receding horizons resume the deterministic actor after the first step)
