# Rockfall datasets — Colab delivery

The full learner NPZs exceed GitHub's 100 MB limit, so they are delivered via
Google Drive (their sidecars, manifests, audit reports and `.sha256` files are
tracked in git for provenance). Upload the three NPZs below to:

```
/content/drive/MyDrive/contrastive_rl_datasets/rockfall/
```

| dataset | file | transitions | sha256 (prefix) | purpose |
|---|---|---|---|---|
| full | `antmaze_rockfall_full.npz` | 971,516 | `d9722c9142…` | main naive full-data CRL (≥300k) |
| center-only (oracle) | `antmaze_rockfall_full_center_only.npz` | 203,700 | `7deba09d43…` | is center behaviour learnable |
| reweight c50 (oracle) | `antmaze_rockfall_full_reweight_c50.npz` | 1,582,616 | `47cf12eba3…` | can weighting push CRL to center |

**Provenance** (see `full/pilot_manifest.json`, `oracle/*_manifest.json`):
- frozen env commit `225b2b0`; full-data collected with `collect_rockfall_pilot.py`
  at env_seed `90100019`, dataset_seed `88140077`, mixture 1190/70/140 (85/5/10).
- The two oracle sets are DIAGNOSTIC ONLY — they use the privileged sidecar
  `route` label to select/upweight episodes. They are not fair baselines.

Learner contract for every NPZ: keys `obs`,`act`,`eval_goals`,`lengths`,`meta`
only (58-dim obs, no mask/route/mode). Full audit (R1–R13) ALL PASS —
`full/audit_report.json`.
