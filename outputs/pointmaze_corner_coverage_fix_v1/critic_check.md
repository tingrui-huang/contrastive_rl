# Critic check: DOWN vs RIGHT (16 roots) and the smoothed DOWN vs CORNER preference

Qbar = policy-width (0.75) average of the critic over tanh(loc + 0.75 eps); DOWN loc (0.3, -6), CORNER loc (5, -5), RIGHT loc (5, -0.1). D series reference: only D1 (+0.54 here) produced DOWN-family actors; D0 (+0.23) and D2 (-0.20) produced CORNER.

| critic | d-r raw | roots down | q(0.3,-1) - q(1,-1) raw | Qbar DOWN - CORNER | roots | Qbar DOWN - RIGHT |
|---|---:|---:|---:|---:|---:|---:|
| E seed 0 | -0.099 | 0/16 | +0.906 | +0.434 | 16/16 | -0.156 |
| E seed 1 | +0.024 | 11/16 | +1.053 | +0.561 | 16/16 | -0.263 |
| E seed 2 | +0.346 | 16/16 | +1.024 | +0.597 | 16/16 | +0.238 |
| E seed 3 | -0.286 | 0/16 | +0.623 | +0.351 | 16/16 | -0.181 |
| E seed 4 | +0.099 | 14/16 | +1.210 | +0.680 | 16/16 | +0.009 |
| D seed 0 (reference) | +0.114 | 14/16 | +0.617 | +0.267 | 16/16 | +0.028 |
| D seed 1 (reference) | +0.359 | 16/16 | +0.783 | +0.529 | 16/16 | +0.199 |
| D seed 2 (reference) | +0.168 | 13/16 | +0.144 | -0.133 | 0/16 | +0.233 |
