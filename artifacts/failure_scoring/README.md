# Scope-B failure-scoring feasibility

The completed comparison is [scope_b_f4_p30_s01/REPORT.md](scope_b_f4_p30_s01/REPORT.md).
Every result in this directory belongs to a supervised feasibility probe using
privileged simulator audit labels. It is not an offline-compatible training
objective or a worst-case value function.

The comparison fits current-XY, successive-displacement and full-F4 scorers for
2,000 updates with seeds 0 and 1. The `scope_b_three_update` directory records
the preceding three-update numerical check. Existing ETT, nominal policy, actor,
critic and failure-bank artifacts were preserved.

Checkpoints named `supervised_probe_*.pkl` remain local and are ignored by Git.
Evaluation NPZ files contain visible observations, scores and explicitly named
audit labels; those labels are supervised targets/evaluation metadata, never
model inputs. No ETT optimization or A/B/C comparison was launched.

The user separately authorized publication of the code, configuration, metrics
and evaluation artifacts. Checkpoints remain local and are excluded from publication.
