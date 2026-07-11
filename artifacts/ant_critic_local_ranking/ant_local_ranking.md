# Ant critic on-support local action ranking

**Verdict: `LOCAL_RANKING_WEAK`** (step 150500, 35 states, receding horizon = 5, actor resumes)

behavior buffer 17500 transitions; mean neighbor state dist 4.88

## Per candidate set (progress = receding-horizon @5, +=closer to goal)
| set | spearman(med) | sp>0 | cb>rand | cb>worst | actor>rand | cb prog@5 | top-dec | bot-dec | cb dist_actor | cb support |
|---|---|---|---|---|---|---|---|---|---|---|
| local_s0.01 | 0.2734560977842886 | 0.9142857142857143 | 0.63 | 0.69 | 0.91 | 0.0039 | 0.0045 | 0.0045 | 0.033 | 2.522 |
| local_s0.03 | 0.24579403955319537 | 0.9428571428571428 | 0.54 | 0.74 | 0.74 | 0.0046 | 0.0044 | 0.0043 | 0.098 | 2.489 |
| local_s0.05 | 0.1777948177989379 | 0.9142857142857143 | 0.69 | 0.77 | 0.74 | 0.0047 | 0.0044 | 0.0042 | 0.156 | 2.460 |
| local_s0.1 | 0.161737776963926 | 0.9714285714285714 | 0.69 | 0.60 | 0.77 | 0.0053 | 0.0047 | 0.0040 | 0.324 | 2.360 |
| local_s0.2 | 0.11570110785570409 | 0.9142857142857143 | 0.69 | 0.66 | 0.74 | 0.0059 | 0.0048 | 0.0038 | 0.608 | 2.205 |
| replay_nbr | 0.8303030303030302 | 0.8285714285714286 | 0.51 | 0.83 | 0.57 | 0.0131 | 0.0131 | -0.0171 | 2.655 | 0.000 |
| uniform | 0.2360888573521333 | 0.9428571428571428 | 0.60 | 0.77 | 0.43 | 0.0388 | 0.0461 | -0.0028 | 3.092 | 2.155 |

replay-nbr vs local(0.05) cb progress diff: 0.0085

pooled critic-best source %: {'local_s0.01': 0.0, 'local_s0.03': 0.0, 'local_s0.05': 0.0, 'local_s0.1': 0.0, 'local_s0.2': 0.02857142857142857, 'replay_nbr': 0.17142857142857143, 'uniform': 0.8}
pooled critic-best mean dist from actor: 3.013, support 1.869
