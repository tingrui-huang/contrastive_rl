# (b) The BC term under the recipe's 'clip' log-prob vs dm-acme 0.4.0's 'acme' boundary rule, on the real actor batches

Each lineage's clipped CF final; the actor stream fast-forwarded to the 30k checkpoint, then 20 real batches of 1024 rows (means over batches).  BC coefficient 0.05; gradients are the WEIGHTED terms (bc x grad BC, (1 - bc) x grad critic term, the critic term with a fresh sampled action).  `bc_mode.json`.

| lineage | NLL clip | NLL acme | mean diff | p99 |diff| | boundary components | rows with a boundary component | ||bc grad|| clip / acme | cos(clip, acme) | ||diff|| / ||clip|| | ||critic-term grad|| | cos(critic, bc clip) / (critic, bc acme) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| seed_0 | -13.335 | -13.191 | -0.144 | 4.217 | 0.016 | 0.106 | 12.0437 / 11.9624 | 0.9994 | 0.0351 | 2.4598 | -0.0883 / -0.0873 |
| seed_1 | -13.687 | -13.552 | -0.135 | 4.128 | 0.016 | 0.106 | 10.7843 / 10.6067 | 0.9986 | 0.0524 | 2.9090 | -0.2609 / -0.2644 |
| seed_2 | -13.113 | -12.978 | -0.136 | 4.044 | 0.015 | 0.107 | 43.1038 / 43.0864 | 1.0000 | 0.0073 | 2.5692 | +0.0218 / +0.0220 |
