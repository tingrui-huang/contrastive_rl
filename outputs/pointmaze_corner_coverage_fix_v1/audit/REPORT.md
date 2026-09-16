# Corner-query consequence audit: native vs fixed ETT, 16 held-out roots

64 paired replicates per (root, action); continuation = the sealed rollout actor with paired keys; means over roots.

| action | native reach | native strict_success | native discounted_return | native lower | native shortcut | native absorbed | ETT reach | ETT strict_success | ETT discounted_return | ETT lower | ETT shortcut | ETT absorbed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_x030 [0.30000001192092896, -1.0] | 0.977 | 0.962 | 10.281 | 1.000 | 0.000 | 0.038 | 0.728 | 0.697 | 5.825 | 1.000 | 0.000 | 0.184 |
| edge_x060 [0.6000000238418579, -1.0] | 0.852 | 0.833 | 9.010 | 0.854 | 0.146 | 0.167 | 0.232 | 0.228 | 2.303 | 0.160 | 0.630 | 0.766 |
| edge_x085 [0.8500000238418579, -1.0] | 0.617 | 0.605 | 6.605 | 0.553 | 0.447 | 0.395 | 0.214 | 0.210 | 2.155 | 0.155 | 0.604 | 0.785 |
| corner [1.0, -1.0] | 0.550 | 0.539 | 5.958 | 0.456 | 0.544 | 0.461 | 0.235 | 0.229 | 2.378 | 0.145 | 0.610 | 0.766 |
| right_y085 [1.0, -0.8500000238418579] | 0.480 | 0.471 | 5.393 | 0.365 | 0.635 | 0.529 | 0.220 | 0.209 | 2.162 | 0.141 | 0.640 | 0.782 |
| right_y060 [1.0, -0.6000000238418579] | 0.360 | 0.351 | 4.178 | 0.210 | 0.790 | 0.649 | 0.232 | 0.228 | 2.495 | 0.126 | 0.679 | 0.764 |
| down [0.0, -1.0] | 0.986 | 0.972 | 10.267 | 1.000 | 0.000 | 0.028 | 0.771 | 0.746 | 6.516 | 1.000 | 0.000 | 0.146 |
| right [1.0, 0.0] | 0.255 | 0.236 | 3.079 | 0.015 | 0.985 | 0.761 | 0.233 | 0.225 | 2.860 | 0.021 | 0.880 | 0.769 |

reach ordering, native: ['down', 'edge_x030', 'edge_x060', 'edge_x085', 'corner', 'right_y085', 'right_y060', 'right']
reach ordering, ETT:    ['down', 'edge_x030', 'corner', 'right', 'edge_x060', 'right_y060', 'right_y085', 'edge_x085']
correlation of per-action reach (native vs ETT): 0.768
