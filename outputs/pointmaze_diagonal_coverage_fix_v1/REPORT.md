# Diagonal query coverage fix: C vs D

D = C replay + 3300 ETT paths from the 550 training contexts with six diagonal first queries [[0.5, -0.3], [0.35, -0.45], [0.65, -0.2], [0.5, -0.5], [0.3, -0.3], [0.7, -0.4]]; those paths: reach 0.281, absorbed 0.703, lower 0.127. Same CRL loss, bc 0.05, 30k x 10, identical initialization per seed; 200 native episodes at the sealed seeds; critic columns on the 16 held-out roots (argmax = where the critic's best action over the square physically leads: shortcut / lower roots out of 16).

| seed | arm | mode reach | mode lower | sample reach | sample lower | critic d-r | roots down | argmax -> shortcut / lower | argmax action |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | C | 1.000 | 1.000 | 0.755 | 0.685 | +0.025 | 13/16 | 11 / 5 | (+1.00,-0.86) |
| 0 | D | 0.860 | 0.805 | 0.555 | 0.445 | +0.114 | 14/16 | 3 / 13 | (+0.31,-0.78) |
| 1 | C | 0.320 | 0.020 | 0.535 | 0.380 | +0.042 | 13/16 | 13 / 3 | (+0.61,-0.48) |
| 1 | D | 0.305 | 0.000 | 0.380 | 0.145 | +0.359 | 16/16 | 0 / 16 | (+0.06,-1.00) |
| 2 | C | 1.000 | 0.995 | 0.640 | 0.575 | -0.221 | 0/16 | 15 / 1 | (+0.85,-0.50) |
| 2 | D | 1.000 | 1.000 | 0.845 | 0.825 | +0.168 | 13/16 | 3 / 13 | (+0.26,-0.84) |
| mean | C | 0.773 | 0.672 | 0.643 | 0.547 | -0.051 | | 13.0 / 3.0 | |
| mean | D | 0.722 | 0.602 | 0.593 | 0.472 | +0.214 | | 2.0 / 14.0 | |
