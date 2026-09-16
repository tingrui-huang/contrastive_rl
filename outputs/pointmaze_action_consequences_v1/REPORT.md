# Consequence check of the six fork actions

16 held-out roots, 64 paired replicates per (root, action), continuation = the sealed observational rollout actor (sampled, paired keys), horizon 49. Native = real environment (evaluation only); ETT = the fixed learned model with the sealed generation rule. Scores are phi.psi logits of the two critics (seed 0 = successful arm, seed 2 = failed arm); rank = mean rank of the action among the six (1 = best).

| action | mean action | q s0 (rank) | q s2 (rank) | NATIVE reach | strict | disc.ret | lower | shortcut | absorbed | ETT reach | strict | disc.ret | lower | shortcut | absorbed |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| down | (+0.00,-1.00) | -6.348 (4.9) | -6.041 (2.9) | 0.950 | 0.943 | 9.91 | 1.000 | 0.000 | 0.041 | 0.825 | 0.813 | 7.30 | 1.000 | 0.000 | 0.130 |
| right | (+1.00,+0.00) | -6.249 (4.1) | -6.246 (4.2) | 0.305 | 0.301 | 3.54 | 0.056 | 0.944 | 0.699 | 0.267 | 0.262 | 3.25 | 0.040 | 0.866 | 0.738 |
| actor_s0 | (+1.00,-0.98) | -6.365 (5.6) | -6.629 (6.0) | 0.589 | 0.580 | 6.29 | 0.499 | 0.501 | 0.410 | 0.238 | 0.235 | 2.34 | 0.157 | 0.630 | 0.757 |
| actor_s2 | (+1.00,-0.72) | -6.137 (3.4) | -6.367 (4.8) | 0.508 | 0.501 | 5.64 | 0.385 | 0.615 | 0.493 | 0.251 | 0.246 | 2.64 | 0.135 | 0.691 | 0.742 |
| argmax_s0 | (+0.52,-0.32) | -5.998 (1.0) | -5.819 (2.1) | 0.493 | 0.489 | 4.87 | 0.469 | 0.531 | 0.507 | 0.319 | 0.307 | 2.95 | 0.261 | 0.590 | 0.673 |
| argmax_s2 | (+0.49,-0.32) | -6.029 (2.0) | -5.646 (1.0) | 0.411 | 0.406 | 4.14 | 0.348 | 0.652 | 0.586 | 0.283 | 0.275 | 2.86 | 0.307 | 0.566 | 0.703 |

## Paired per-root comparisons (mean difference; roots where the first action is better / worse)

| comparison | metric | native | ett |
|---|---|---|---|
| argmax_s2_vs_down | reach | -0.539 (0/16) | -0.542 (0/16) |
| argmax_s2_vs_down | discounted_return | -5.771 (0/16) | -4.447 (0/16) |
| argmax_s2_vs_down | lower | -0.652 (0/14) | -0.693 (0/15) |
| argmax_s2_vs_down | absorbed | +0.545 (16/0) | +0.573 (16/0) |
| argmax_s0_vs_down | reach | -0.457 (0/16) | -0.506 (0/16) |
| argmax_s0_vs_down | discounted_return | -5.045 (0/16) | -4.355 (0/16) |
| argmax_s0_vs_down | lower | -0.531 (0/15) | -0.739 (0/16) |
| argmax_s0_vs_down | absorbed | +0.466 (16/0) | +0.543 (16/0) |
| actor_s2_vs_actor_s0 | reach | -0.081 (1/6) | +0.013 (9/5) |
| actor_s2_vs_actor_s0 | discounted_return | -0.652 (10/5) | +0.298 (9/7) |
| actor_s2_vs_actor_s0 | lower | -0.114 (3/5) | -0.022 (2/8) |
| actor_s2_vs_actor_s0 | absorbed | +0.083 (7/1) | -0.015 (6/9) |

## Did these actions enter NCE? C-replay transitions within 0.3 of the root with an action within 0.25 of the action (mean over roots)

| action | transitions | synthetic first-step queries | synthetic later steps | original transitions | reach fraction of those episodes |
|---|---:|---:|---:|---:|---:|
| down | 1182 | 1090 | 20 | 72 | 0.813 |
| right | 2564 | 1090 | 205 | 1269 | 0.555 |
| actor_s0 | 37 | 0 | 24 | 13 | 0.121 |
| actor_s2 | 109 | 0 | 72 | 38 | 0.270 |
| argmax_s0 | 194 | 0 | 57 | 137 | 0.283 |
| argmax_s2 | 105 | 0 | 46 | 58 | 0.235 |
