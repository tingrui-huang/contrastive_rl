# Short actor-only updates from 20k (5000 updates, the recipe actor loss and stream, critic frozen) under each critic; then rolled from the 128 start states

| critic | left start at 100 | xy disp median at 100 | torque saturation | policy param delta | actor loss / bc nll / q term at the end | saturation on training batches at the end |
|---|---:|---:|---:|---:|---|---:|
| own20k | 0.60 | 2.68 | 0.40 | 13.60 | 5.303 / -15.95 / 6.421 | 0.051 |
| ownfinal | 0.60 | 6.83 | 0.35 | 15.24 | 5.534 / -15.05 / 6.618 | 0.054 |
| other_d2_20k | 1.00 | 10.17 | 0.08 | 14.47 | 5.321 / -15.67 / 6.425 | 0.049 |
| other_d2_final | 0.98 | 10.61 | 0.08 | 16.87 | 5.471 / -14.88 / 6.542 | 0.057 |
