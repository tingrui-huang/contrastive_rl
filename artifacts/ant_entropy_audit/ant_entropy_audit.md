# AntMaze entropy-collapse checkpoint audit

**Verdict: `ENTROPY_COLLAPSE_SUPPORTED`**

Checkpoints: init=0, early=42000, mid=84000, final=150500; common probe set 600 obs ({'random': 202, 'mid': 90, 'init': 104, 'final': 112, 'early': 92}); 20 unique saved states.

## A. Policy head on the common probe set
| ckpt | scale med | scale p10-p90 | |loc| mean | mode sat | sample sat | |sample-mode| | entropy est | mode std across states | mode eff rank |
|---|---|---|---|---|---|---|---|---|---|
| init | 0.800 | 0.333-1.524 | 0.91 | 0.028 | 0.083 | 1.329 | -2.29 | 0.377 | 3.76 |
| early | 0.000 | 0.000-0.000 | 6.03 | 0.648 | 0.648 | 0.000 | 71112645935104.00 | 0.701 | 4.04 |
| mid | 0.000 | 0.000-0.000 | 5.37 | 0.670 | 0.670 | 0.000 | 41152766017536.00 | 0.774 | 5.20 |
| final | 0.000 | 0.000-0.000 | 4.64 | 0.584 | 0.584 | 0.000 | 30388193853440.00 | 0.779 | 5.81 |

## B. Eval episodes (20 eps, same reset seeds; 700 steps)
| ckpt | mode | success | progress | path len | speed | static frac | fall frac | sat | cells |
|---|---|---|---|---|---|---|---|---|---|
| init | deterministic | 0.00 | 0.197 | 0.9 | 0.0013 | 0.90 | 0.00 | 0.00 | 2.2 |
| init | stochastic | 0.00 | 1.531 | 20.0 | 0.0286 | 0.00 | 0.72 | 0.11 | 18.1 |
| early | deterministic | 0.00 | 0.244 | 1.8 | 0.0026 | 0.79 | 0.05 | 0.69 | 4.7 |
| early | stochastic | 0.00 | 0.245 | 1.8 | 0.0026 | 0.83 | 0.05 | 0.69 | 4.5 |
| mid | deterministic | 0.00 | 0.502 | 3.7 | 0.0052 | 0.73 | 0.18 | 0.61 | 6.1 |
| mid | stochastic | 0.05 | 0.771 | 6.4 | 0.0092 | 0.55 | 0.32 | 0.60 | 8.2 |
| final | deterministic | 0.05 | 0.651 | 6.2 | 0.0089 | 0.63 | 0.41 | 0.54 | 9.1 |
| final | stochastic | 0.00 | 0.640 | 5.9 | 0.0084 | 0.65 | 0.48 | 0.56 | 8.9 |

## C. 100-step rollouts from the same 20 unique saved states
| ckpt | mode | net disp | path len | progress | fell |
|---|---|---|---|---|---|
| init | deterministic | 1.233 | 3.222 | 0.511 | 0.25 |
| init | stochastic | 2.099 | 5.845 | 0.838 | 0.57 |
| early | deterministic | 1.196 | 2.942 | 0.442 | 0.30 |
| early | stochastic | 0.946 | 2.757 | 0.400 | 0.25 |
| mid | deterministic | 0.948 | 2.577 | 0.423 | 0.10 |
| mid | stochastic | 1.011 | 2.510 | 0.340 | 0.23 |
| final | deterministic | 1.012 | 1.800 | 0.673 | 0.00 |
| final | stochastic | 0.896 | 1.848 | 0.690 | 0.12 |

## D. Collapse timing (metrics.json, eval every ~10.5k steps)
- first eval with action saturation > 0.5: step 21000.0
- first eval with logits gap > 20: step 21000.0
- see timeline.png for saturation vs critic-gap growth vs actor loss.

## Hypothesis checks
- H1 scale collapse (final med scale < 0.2 and < 0.5x init): True (series {'init': 0.7998886704444885, 'early': 9.999999974752427e-07, 'mid': 9.999999974752427e-07, 'final': 9.999999974752427e-07})
- H1 mode saturation (init <0.1 -> final >0.4): True
- H2 deterministic static (static_frac>0.3 or speed<0.005): True
- H2 stochastic moves >=2x deterministic: False
- H2 TOTAL scale collapse (final sigma~0, sample==mode): True
- H2 init policy stochastic gain >=2x (what collapse destroyed): True
- H3 scale monotone decreasing: True

**Verdict: `ENTROPY_COLLAPSE_SUPPORTED`**

## Caveats
- The `entropy est` values ~1e13 at early/mid/final are a numerical artifact (log-prob of clipped tanh samples at the sigma=1e-6 floor), not real entropies; the scale series is the evidence.
- Timing resolution is the eval cadence (~10.5k steps): collapse (saturation>0.5) and critic logits-gap>20 both first appear at step 21000 -- collapse completed within ~10k steps of learning onset; finer ordering is not resolvable.
- stochastic==deterministic at early/mid/final is the signature of total collapse (sample-minus-mode distance 0.000), not counter-evidence.
