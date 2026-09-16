# D1 vs D2: the objective the actor actually optimizes at the fork

L(loc) = (1-bc) * E_(a~tanh N(loc, scale))[-q(s,a)] + bc * E_(a_data)[-log pi(a_data|loc,scale)], bc = 0.05. Qbar = the sampled-action mean score, on a pre-tanh loc grid [-6, 6]^2 with 128 innovations, at the fresh actor's own scale at the root (actor seed 0); BC = the tanh-normal NLL of the D-replay actions taken within 0.15 of the root (thousands per root, boundary-clipped ones included); region = where the corresponding mode action physically leads (Step 1 physics map).

| critic | raw argmax region (lower/shortcut/other) | Qbar argmax region @actor scale | Qbar argmax region @scale 0.3 | BC argmin mode (mean) | total argmin region | actor mode (mean) | fresh actors: mode y (3 seeds, root mean) |
|---|---|---|---|---|---|---|---|
| D1 | 16/0/0 | 6/10/0 | 16/0/0 | (+0.95,-0.72) | 2/14/0 | (+1.00,-0.30) | -0.30 / -0.51 / -0.37 |
| D2 | 13/3/0 | 7/9/0 | 16/0/0 | (+0.95,-0.72) | 7/9/0 | (+1.00,-1.00) | -1.00 / -1.00 / -1.00 |

## Values at the fresh actor's loc (mean over roots)

| critic | Qbar(actor) | Qbar max | Qbar at loc (0,-5) (= DOWN) | q(DOWN) raw | BC(actor) | BC min | d/dloc_y of (1-bc)(-Qbar) | d/dloc_y of bc*BC | total d/dloc_y | d/dloc_x critic / bc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1 | -6.626 | -6.561 | -6.675 | -5.894 | 1.268 | 0.323 | -0.0353 | +0.0271 | -0.0082 | -0.0232 / +0.0285 |
| D2 | -6.113 | -6.081 | -6.507 | -5.937 | 1.708 | -0.137 | +0.0274 | -0.0386 | -0.0112 | -0.0152 / +0.0227 |

Sign convention: loc_y more negative = more DOWN. A negative d/dloc_y means the term pushes loc_y down (towards the detour); the actor rests where the total is ~0 (or where tanh saturation kills the critic gradient).

## Why the scale is wide, and what that does to Qbar (mean over the 16 roots)

12,722 D-replay actions within 0.15 of the roots; 13.1% of their x components
and 4.4% of their y components are clipped to +/-1, i.e. atanh ~ +/-7.25 in
pre-tanh space. The BC term as a function of the policy scale (same scale on
both axes), at the fresh actor's mean loc and at the DOWN loc (0, -5):

| scale | D1: BC@actor | BC@DOWN | Qbar@DOWN | Qbar@actor | total@DOWN | total@actor | D2: BC@actor | BC@DOWN | Qbar@DOWN | Qbar@actor | total@DOWN | total@actor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 96.9 | 169.4 | -6.03 | -6.43 | 14.20 | 10.95 | 155.5 | 169.4 | -6.09 | -6.07 | 14.26 | 13.54 |
| 1.0 | 6.24 | 12.77 | -6.44 | -6.60 | 6.76 | 6.58 | 11.51 | 12.77 | -6.40 | -6.07 | 6.72 | 6.34 |
| 1.5 | 1.94 | 4.84 | -6.61 | -6.68 | 6.52 | 6.45 | 4.28 | 4.84 | -6.52 | -6.08 | 6.44 | 5.99 |
| 2.0 | 0.72 | 2.36 | -6.72 | -6.75 | 6.50 | 6.45 | 2.04 | 2.36 | -6.61 | -6.11 | 6.40 | 5.91 |
| 3.0 | 0.26 | 0.98 | -6.88 | -6.87 | 6.59 | 6.54 | 0.84 | 0.98 | -6.75 | -6.27 | 6.46 | 6.00 |

The boundary-clipped actions make the BC NLL explode at any scale below ~1
(97-170 nats at 0.3), so the BC term alone drives the scale to 1.5-2 where it
is cheap; there it still charges the DOWN loc (x = 0) about 3 nats more than
a loc with x ~ +4 (D1: 4.8 vs 1.9), because 13% of the data x's sit at +1.
At that scale D1's sampled-average critic advantage of DOWN over the actor's
plateau is only ~0.07 (-6.61 vs -6.68), worth 0.95 * 0.07 = 0.07 in the
objective, while the BC penalty of DOWN is 0.05 * 2.9 = 0.15: the objective
prefers the plateau (total 6.45 < 6.52), which is exactly where the fresh
actors sit. Under D2 the critic's high region is the bottom-right corner,
which also satisfies BC's x = +1 preference (Qbar@actor -6.08 vs -6.52 at
DOWN; total 5.99 vs 6.44): both terms agree, the actor goes to (+1, -1), and
the axis race does the rest.

## Per root

| critic | root | raw argmax | Qbar argmax mode (region) | BC argmin mode | total argmin mode (region) | actor mode | grad_y critic | grad_y bc |
|---|---|---|---|---|---|---|---:|---:|
| D1 | 0 | (+0.07,-1.00) | (+0.00,-1.00) (lower) | (+0.96,-0.64) | (+1.00,-0.46) (shortcut) | (+1.00,-0.33) | -0.0327 | +0.0146 |
| D1 | 1 | (+0.07,-1.00) | (+0.00,-1.00) (lower) | (+0.96,-0.64) | (+1.00,-0.46) (shortcut) | (+1.00,-0.33) | -0.0408 | +0.0163 |
| D1 | 2 | (+0.10,-1.00) | (+0.00,-1.00) (lower) | (+0.94,-0.46) | (+1.00,-0.46) (lower) | (+1.00,-0.30) | -0.0198 | +0.0118 |
| D1 | 3 | (+0.07,-1.00) | (+1.00,-0.24) (shortcut) | (+0.98,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.30) | -0.0354 | +0.0311 |
| D1 | 4 | (+0.07,-1.00) | (+0.00,-1.00) (lower) | (+0.96,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.32) | -0.0159 | +0.0241 |
| D1 | 5 | (+0.07,-1.00) | (+1.00,-0.24) (shortcut) | (+0.98,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.30) | -0.0380 | +0.0315 |
| D1 | 6 | (+0.07,-1.00) | (+0.00,-1.00) (lower) | (+0.96,-0.64) | (+1.00,-0.46) (shortcut) | (+1.00,-0.34) | -0.0337 | +0.0121 |
| D1 | 7 | (+0.00,-1.00) | (+1.00,-0.24) (shortcut) | (+0.96,-0.85) | (+1.00,-0.46) (shortcut) | (+1.00,-0.28) | -0.0264 | +0.0427 |
| D1 | 8 | (+0.10,-1.00) | (+1.00,-0.24) (shortcut) | (+0.98,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.28) | -0.0541 | +0.0312 |
| D1 | 9 | (+0.10,-1.00) | (+1.00,-0.24) (shortcut) | (+0.98,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.29) | -0.0338 | +0.0283 |
| D1 | 10 | (+0.07,-1.00) | (+1.00,-0.24) (shortcut) | (+0.98,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.27) | -0.0437 | +0.0383 |
| D1 | 11 | (+0.00,-1.00) | (+1.00,-0.24) (shortcut) | (+0.94,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.28) | -0.0431 | +0.0389 |
| D1 | 12 | (+0.03,-1.00) | (+1.00,-0.24) (shortcut) | (+0.98,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.29) | -0.0583 | +0.0349 |
| D1 | 13 | (+0.00,-1.00) | (+1.00,-0.24) (shortcut) | (+0.94,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.28) | -0.0315 | +0.0375 |
| D1 | 14 | (+0.07,-1.00) | (+0.00,-1.00) (lower) | (+0.76,-0.64) | (+0.00,-0.99) (lower) | (+1.00,-0.35) | -0.0243 | +0.0103 |
| D1 | 15 | (+0.07,-1.00) | (+1.00,-0.24) (shortcut) | (+0.98,-0.76) | (+1.00,-0.46) (shortcut) | (+1.00,-0.30) | -0.0338 | +0.0298 |
| D2 | 0 | (+0.03,-1.00) | (+1.00,-1.00) (lower) | (+0.96,-0.64) | (+1.00,-1.00) (lower) | (+0.99,-1.00) | +0.0307 | -0.0383 |
| D2 | 1 | (+0.07,-1.00) | (+1.00,-1.00) (lower) | (+0.96,-0.64) | (+1.00,-1.00) (lower) | (+0.99,-1.00) | +0.0260 | -0.0382 |
| D2 | 2 | (+0.07,-1.00) | (+1.00,-1.00) (lower) | (+0.94,-0.46) | (+0.99,-1.00) (lower) | (+0.99,-1.00) | +0.0244 | -0.0357 |
| D2 | 3 | (+0.07,-1.00) | (+1.00,-1.00) (shortcut) | (+0.98,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0269 | -0.0413 |
| D2 | 4 | (+0.07,-1.00) | (+1.00,-1.00) (lower) | (+0.96,-0.76) | (+1.00,-1.00) (lower) | (+1.00,-1.00) | +0.0335 | -0.0374 |
| D2 | 5 | (+0.07,-1.00) | (+1.00,-1.00) (shortcut) | (+0.98,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0274 | -0.0416 |
| D2 | 6 | (+0.07,-1.00) | (+1.00,-1.00) (lower) | (+0.96,-0.64) | (+1.00,-1.00) (lower) | (+0.99,-1.00) | +0.0242 | -0.0378 |
| D2 | 7 | (+1.00,-0.13) | (+1.00,-1.00) (shortcut) | (+0.96,-0.85) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0299 | -0.0379 |
| D2 | 8 | (+0.03,-1.00) | (+1.00,-1.00) (shortcut) | (+0.98,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0277 | -0.0389 |
| D2 | 9 | (+0.07,-1.00) | (+1.00,-1.00) (lower) | (+0.98,-0.76) | (+1.00,-1.00) (lower) | (+1.00,-1.00) | +0.0257 | -0.0374 |
| D2 | 10 | (+0.13,-1.00) | (+1.00,-1.00) (shortcut) | (+0.98,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0234 | -0.0407 |
| D2 | 11 | (+1.00,-0.13) | (+1.00,-1.00) (shortcut) | (+0.94,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0400 | -0.0388 |
| D2 | 12 | (+0.20,-1.00) | (+1.00,-1.00) (shortcut) | (+0.98,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0249 | -0.0413 |
| D2 | 13 | (+1.00,-0.13) | (+1.00,-1.00) (shortcut) | (+0.94,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0341 | -0.0391 |
| D2 | 14 | (+0.13,-1.00) | (+0.99,-1.00) (lower) | (+0.76,-0.64) | (+0.94,-1.00) (lower) | (+0.98,-1.00) | +0.0162 | -0.0325 |
| D2 | 15 | (+0.07,-1.00) | (+1.00,-1.00) (shortcut) | (+0.98,-0.76) | (+1.00,-1.00) (shortcut) | (+1.00,-1.00) | +0.0237 | -0.0403 |
