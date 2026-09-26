# The critic score at the 64 reset states split into |phi(s, a)|, |psi(g)| and cos(phi, psi) (mean over the twin heads and states)

| critic | goal set | action | mean abs phi | mean abs psi | mean cos | twin-min logit |
|---|---|---|---:|---:|---:|---:|
| d1_s4 | task | stall | 1.284 | 6.813 | -0.773 | -6.997 |
| d1_s4 | task | fwd | 1.510 | 6.813 | -0.856 | -9.135 |
| d1_s4 | task | start | 1.929 | 6.813 | -0.832 | -11.217 |
| d1_s4 | task | logged | 1.756 | 6.813 | -0.784 | -9.422 |
| d1_s4 | marginal draw1 | stall | 1.284 | 15.945 | -0.487 | -7.034 |
| d1_s4 | marginal draw1 | fwd | 1.510 | 15.945 | -0.545 | -8.567 |
| d1_s4 | marginal draw1 | start | 1.929 | 15.945 | -0.528 | -10.417 |
| d1_s4 | marginal draw1 | logged | 1.756 | 15.945 | -0.451 | -8.087 |
| d2_s4 | task | stall | 2.065 | 6.061 | -0.810 | -10.434 |
| d2_s4 | task | fwd | 1.596 | 6.061 | -0.847 | -8.591 |
| d2_s4 | task | start | 2.235 | 6.061 | -0.823 | -11.355 |
| d2_s4 | task | logged | 2.104 | 6.061 | -0.787 | -10.261 |
| d2_s4 | marginal draw1 | stall | 2.065 | 15.949 | -0.507 | -9.990 |
| d2_s4 | marginal draw1 | fwd | 1.596 | 15.949 | -0.512 | -8.241 |
| d2_s4 | marginal draw1 | start | 2.235 | 15.949 | -0.504 | -10.543 |
| d2_s4 | marginal draw1 | logged | 2.104 | 15.949 | -0.449 | -8.866 |
| d1_s4 | task | **stall vs fwd** | ratio 0.850 | | diff +0.083 | diff +2.138 |
| d1_s4 | marginal draw1 | **stall vs fwd** | ratio 0.850 | | diff +0.057 | diff +1.532 |
| d2_s4 | task | **stall vs fwd** | ratio 1.294 | | diff +0.037 | diff -1.843 |
| d2_s4 | marginal draw1 | **stall vs fwd** | ratio 1.294 | | diff +0.005 | diff -1.749 |
