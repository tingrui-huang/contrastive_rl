# Synthetic shared-response diagnostic

This experiment is isolated from PointMaze training and the existing ETT method.
It fits a known scalar response with an ordinary shared ReLU network, comparing
joint diagonal/off-diagonal fitting, a diagonal-only control, and a signed
difference input ablation. The prescribed constants are L=0.25,1,4.

The authoritative fixed-budget run is
[l025_1_4_s012/REPORT.md](l025_1_4_s012/REPORT.md): three seeds, three arms,
three scales, 2,500 steps per run. `smoke/` is a separate five-step execution
check and is not a scientific comparison. No checkpoint is selected using
evaluation outcomes, and normalized predictions coincide across scales.

Code, configuration, metrics, evaluation arrays and English plots/report are
intended for version control. Final `.pt` checkpoints are local and Git-ignored.
Reproduction commands and numerical checks are documented in the report.
