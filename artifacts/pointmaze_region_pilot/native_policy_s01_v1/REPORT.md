# Frozen pessimistic ETT: native policy-improvement test

The experiment does not establish a consistent native benefit from training against the pessimistic ETT.

A is unchanged. B uses the final diagonal-fit head with both channels equal to the executed action; C uses the final critic-driven ETT with independent frozen nominal actions. Four training rounds per B/C actor alternate current-policy rollouts, averaged-NCE fitting and 25 guarded decoded-Q actor proposals. All actors use the same stochastic tanh-Gaussian execution convention, with paired noise. No native outcome or full-F4 hindsight actor objective enters training.

## Native outcomes: 256 fresh paired reset seeds

Return is sum(.95^t*r), not the normalized critic Q (.05 times return). Success uses distance<.5; failure is absorbing death, not merely lack of success. Intervals are 95% paired episode bootstrap.

- A: return 3.3252, success 0.297, failure 0.703.
- B_s0: return 3.3587, success 0.293, failure 0.707.
- C_s0: return 3.4357, success 0.305, failure 0.695.
- B_s1: return 3.4049, success 0.301, failure 0.699.
- C_s1: return 3.3462, success 0.297, failure 0.703.
- C_s0_minus_A: returns +0.1106 [-0.0596, +0.2862]; success +0.0078 [-0.0117, +0.0273]; failure -0.0078 [-0.0273, +0.0117].
- C_s0_minus_B: returns +0.0771 [-0.0348, +0.2240]; success +0.0117 [-0.0039, +0.0312]; failure -0.0117 [-0.0312, +0.0039].
- B_s0_minus_A: returns +0.0335 [-0.1474, +0.2167]; success -0.0039 [-0.0234, +0.0156]; failure +0.0039 [-0.0156, +0.0234].
- C_s1_minus_A: returns +0.0211 [-0.0830, +0.1410]; success +0.0000 [-0.0117, +0.0117]; failure +0.0000 [-0.0117, +0.0117].
- C_s1_minus_B: returns -0.0587 [-0.2076, +0.0917]; success -0.0039 [-0.0195, +0.0117]; failure +0.0039 [-0.0117, +0.0195].
- B_s1_minus_A: returns +0.0798 [-0.0897, +0.2619]; success +0.0039 [-0.0156, +0.0234]; failure -0.0039 [-0.0234, +0.0156].

## Independent cross-model improvement versus A

Each cell uses 128 paired full rollouts from START, with the model acting at every transition. Rows below give actor return differences versus A on each frozen model. These are model outputs, not native outcomes.

- obs_s0: B_s0 +0.1813 [-0.1745, +0.5484]; C_s0 +0.0248 [-0.3387, +0.3760]; B_s1 +0.2322 [-0.1439, +0.6068]; C_s1 -0.1857 [-0.4905, +0.0894].
- pess_s0: B_s0 -0.0251 [-0.0754, +0.0000]; C_s0 -0.0251 [-0.0754, +0.0000]; B_s1 +0.0390 [-0.0754, +0.1925]; C_s1 -0.0251 [-0.0754, +0.0000].
- obs_s1: B_s0 -0.0151 [-0.4017, +0.3775]; C_s0 +0.3043 [-0.0053, +0.6244]; B_s1 +0.2518 [-0.1326, +0.6325]; C_s1 +0.0707 [-0.1355, +0.3064].
- pess_s1: B_s0 -0.0035 [-0.0105, +0.0000]; C_s0 -0.0035 [-0.0105, +0.0000]; B_s1 -0.0035 [-0.0105, +0.0000]; C_s1 -0.0035 [-0.0105, +0.0000].

## Policy changes, constraints and cost

KL/drift below use the fixed training state bank. They are empirical constraints, not global bounds. Every proposed update also checks current-rollout states and a .002 one-step KL limit.

- B_s0: 9/100 accepted; initial-to-final KL 0.008849, mean-action RMS/max L2 0.02328/0.14857, mean scale change +0.00305, parameter L2 change 0.01211.
- C_s0: 10/100 accepted; initial-to-final KL 0.015109, mean-action RMS/max L2 0.02555/0.09673, mean scale change +0.00408, parameter L2 change 0.01284.
- B_s1: 8/100 accepted; initial-to-final KL 0.005017, mean-action RMS/max L2 0.01866/0.13579, mean scale change +0.00393, parameter L2 change 0.01076.
- C_s1: 10/100 accepted; initial-to-final KL 0.003079, mean-action RMS/max L2 0.01794/0.14456, mean scale change +0.00764, parameter L2 change 0.01132.
- Inherited seed 0 validation diagonal-energy degradation: B +0.000181, C +0.014969; the prior .02 allowance and action-Lipschitz tests passed for the underlying ETTs.
- Inherited seed 1 validation diagonal-energy degradation: B +0.000249, C +0.008099; the prior .02 allowance and action-Lipschitz tests passed for the underlying ETTs.

Budgets: 423,168/500,000 model transitions; 64,000/64,000 native steps; 10,800 NCE steps and 400 actor proposals. Transition models and nominal policy remain unchanged. All five actors are sealed before any native or cross-model evaluation; final checkpoints only.

## Limits

All primary native return, success and failure intervals include zero. Seed 0 favors C in its point estimates, while seed 1's C-versus-B return estimate is negative. This is unresolved evidence, not equivalence or a demonstrated native improvement.

Own-model full-rollout changes are also unresolved: B gains +.1813/+.2518 in seeds 0/1, while C changes by -.0251/-.0035 with upper interval endpoints at zero. The pessimistic-model evaluations contain many exactly tied realized returns despite changed actions. These finite-sample ties do not establish policy equivalence. Accepted proposals raise the frozen critic objective on their training minibatches on average, but that does not establish improved independent model rollout return.

The models also remain markedly different from the native environment: A's raw mean return is 3.3252 natively, versus 8.0747/7.8029 in the observational models and 10.3148/10.1756 in the pessimistic models. These descriptive means expose an inherited transfer limitation under the matched execution convention; model targets are not native ground truth. Only 37/400 actor proposals were accepted, so the fixed policy constraints substantially limited adaptation. This run cannot separate limited policy movement, critic approximation and transition-model error as unique causes of the unresolved outcome.

Cross-model gains can reflect model-specific approximation or exploitation, and do not establish native improvement. B and C inherit different fitted diagonal heads as well as response maps. Their permitted diagonal-fit degradation, partial-observation F4 state and generator approximation remain limitations. The action-Lipschitz construction for C conditions on x_prime; it is not a bound on B when both channels vary.

An ETT trained against A is not asserted to remain worst-case for an updated actor. Two paired training seeds and descriptive, non-simultaneous intervals do not establish universal benefit or equivalence. Intervals crossing zero are inconclusive; observed ties do not prove equivalence. No ETT retraining, extra sweeps, automatic extensions or push followed.

Reproduce with a fresh directory: `python -m ett.pointmaze_native_policy prepare --out <dir>` then `python -m ett.pointmaze_native_policy run --out <dir>`. Read-only audit: `python -m ett.check_pointmaze_native_policy --out <dir>`.
