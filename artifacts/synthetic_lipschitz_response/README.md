# Full action-Lipschitz synthetic follow-up

This isolated follow-up uses a conditional continuous linear spline with bounded
execution-action slopes. The shared network learns the diagonal value through
the diagonal loss; no architectural equality branch or third loss is used.

[l1_s012/REPORT.md](l1_s012/REPORT.md) contains the authoritative L=1,
three-seed, 2,500-step comparison with the two previously trained unconstrained
joint-fitting models. `smoke/` is a separate five-step execution check.

Original baseline artifacts are read and verified without modification. New
`.pt` checkpoints stay local and Git-ignored. Configuration, metrics, evaluation
arrays, plots and English reports are prepared for version control.
