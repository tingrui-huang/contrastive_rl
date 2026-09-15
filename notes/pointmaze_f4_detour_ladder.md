# PointMaze F4: the teacher-detour dataset ladder

Six F4 (p = 0.3, frame-stacked) datasets that differ in exactly one number:
`force_safe_prob`, the probability that a teacher episode takes the
always-safe lower route instead of heading for the shortcut. One flag picks a
rung; the benchmark, seed, composed failure bank, recipe and evaluation are
unchanged. The AntMaze counterpart is `notes/v6_detour_ladder.md` on
`feature/antmaze-causal-transition`.

```
rung   0.05   0.10   0.15   0.20   0.25   0.30      (fraction of TEACHER episodes)
```

---

## 1. How the 5% is made here

[`scripts/collect_swamp_windy.py`](../scripts/collect_swamp_windy.py)
(`make_windy_teacher`) draws one coin per teacher episode at its first step,

```python
memo['force_safe'] = bool(rng.random() < force_safe_prob)
```

and a forced-safe episode follows the BFS oracle on swamp-blocked walls for
the whole episode; every other teacher episode heads for the shortcut with the
per-step reactive dodge. The F4 recipe collects 6000 episodes of which the
first 1200 (`random_frac 0.2`) are uniform-random actions and the remaining
4800 are teacher episodes; 600 bad-demonstrator episodes are appended by
`merge_swamp_windy_baddemo.py`. So `force_safe_prob` is a fraction of the
4800 teacher episodes: 0.05 is ~257 safe-route episodes = 3.9% of the 6600.

**What the rungs share, and what they do not.** The collector draws the
coin, the random-episode actions and the per-step teacher noise from ONE RNG
stream, and the environment draws its wind bits and action noise from one
stream whose consumption depends on when an episode dies. Consequently:

* the 1200 random episodes (all drawn before any coin) and the 600 bad-demo
  episodes are byte-identical on every rung;
* the 4800 teacher episodes diverge at the very first one (episode 1200 or
  1201 on every pair): the rungs are an independent re-roll of the same
  teacher process, **not nested and not paired**, unlike the AntMaze V6
  ladder where the route coin has its own stream.

Making them nested would need a stream-separated collector and therefore a
NEW 0.05 dataset, i.e. a ladder disconnected from every existing p=0.3 result
(the alpha sweep, G4, the ETT line). Keeping the historical file as the 0.05
rung was judged more useful; the audit records the re-roll honestly.

## 2. Files

`datasets/` is gitignored on this branch, so the repository carries the
**manifest** (`artifacts/swamp_windy_f4_detour_ladder/ladder_manifest.json`,
every rung's content sha and composition) and the **audit**
(`ladder_audit.json`); any node rebuilds a rung deterministically with
`python scripts/build_f4_detour_ladder.py build --rungs 0.15` (~3 min per
rung on one core) and the launcher refuses a rung whose content sha does not
match the manifest. The same seed reproduces the same content sha on Windows
and on the Linux GPU node (checked, section 4).

| rung | merged file | safe-route eps | of all 6600 | died | reached@0.5 |
|---|---|---|---|---|---|
| 0.05 | `swamp_windy_f4_merged_s0.npz` (the existing file) | 257 | 3.9% | 0.220 | 0.717 |
| 0.10 | `swamp_windy_f4_far10_merged_s0.npz` | 490 | 7.4% | 0.213 | 0.725 |
| 0.15 | `swamp_windy_f4_far15_merged_s0.npz` | 738 | 11.2% | 0.211 | 0.727 |
| 0.20 | `swamp_windy_f4_far20_merged_s0.npz` | 983 | 14.9% | 0.209 | 0.729 |
| 0.25 | `swamp_windy_f4_far25_merged_s0.npz` | 1244 | 18.8% | 0.198 | 0.741 |
| 0.30 | `swamp_windy_f4_far30_merged_s0.npz` | 1461 | 22.1% | 0.201 | 0.738 |

Forced-safe episodes never die (died rate 0.0 on every rung), so the overall
death rate falls slowly along the ladder; the configured probability sits
inside the realized Wilson99 interval on every rung. Audit gates: per rung 8/8,
per pair 3/3 (random block identical, bad-demo block identical, safe-route
count increases), 6 rungs and 15 pairs PASS.

**The composed bank.** `artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz`
is held fixed across rungs (the same choice as on AntMaze: one variable). The
launcher's provenance gate normally requires the bank to have been built from
the selected dataset; with a rung selected it also accepts the ladder's
canonical bank -- the one built from the 0.05 rung -- and prints which case
applied. A per-rung bank remains possible by rebuilding with
`make_swamp_f4_failure_bank.py --npz <rung>` and passing `--bank`.

Note that the tracked bank file was **stale** before this change: it had been
built from the obsolete p=0.1 dataset (source content sha `22185598...`), so on
a fresh clone the zfail gate failed against the current p=0.3 dataset
(`ad8b4470...`). It is now rebuilt from the current 0.05 rung with the same
command (`--compose random=0.6,deliberate=0.4 --max-bank 256 --seed 0`) and is
content-identical to the bank the p=0.3 server sweep actually used
(`022f2d0d...`).

## 3. Picking a rung

```bash
# the p=0.3 alpha sweep on the 15% rung (builds the rung if missing)
FORCE_SAFE=0.15 SEEDS="0 1 2" bash scripts/run_f4_p30_sweep.sh run

# a single arm
python scripts/run_swamp_windy_z_failneg.py --version f4 --arm zfail --alpha 0.3 \
    --bank artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz \
    --force-safe-prob 0.15 --run
```

* `--force-safe-prob` (f4 only) resolves the dataset, pins its content sha
  against the manifest, and suffixes the run tag: `swamp_windy_f4_far15_zfail_a3_s0`.
  Without the flag nothing changes (registry dataset, old tag, old gate).
* `FORCE_SAFE` on the p=0.3 sweep suffixes `RUN_ID` (hence the run root,
  eval root and log dir) with `_far15`, passes the flag to every arm, and
  records `force_safe_prob` in `sweep_provenance.json`; `arm_provenance.json`
  records it per run together with the bank's source sha.
* A rung outside the six is refused (`rung 0.12 is not on the ladder`); the
  flag on `--version v0/v1` is refused.

Not wired: the older `run_f4_failneg_sweep.sh` (pinned to the obsolete p=0.1
hashes, already dead), `run_pointmaze_absorbing_g4.py` (G4 is retired) and the
ETT integration under `ett/` (its nominal policy, diagonal model and initial
checkpoint are all trained on the 0.05 rung; retargeting it is a separate job).

## 4. Verification

* Gate on the 0.15 and 0.30 rungs (zbase and zfail), on 0.05 through the
  ladder, and on the legacy no-flag path: PASS; the two error paths refuse.
* 2000-step smoke of `zfail alpha 0.3` on the 0.15 rung (CPU), then
  `eval_swamp_windy_z_deployment` on the checkpoint: OK. The trainer's own
  `offline_dataset.sha256` fingerprint equals the file sha of
  `swamp_windy_f4_far15_merged_s0.npz`, not the 0.05 file's.
* GPU node (RTX 3060 Ti, `~/contrastive_rl` at this commit): the five rungs
  rebuilt from scratch reproduce the manifest's content shas exactly
  (Windows and Linux agree), the ladder audit passes there,
  `FORCE_SAFE=0.15 ALPHAS=0.3 bash scripts/run_f4_p30_sweep.sh check` passes
  both arms with the bank accepted as the canonical one, and
  `FORCE_SAFE=0.30 ALPHAS=0.3 EPISODES=20 ... smoke` trains and evaluates both
  arms; each arm's `offline_dataset.sha256` equals the far30 file's sha and
  `arm_provenance.json` records `force_safe_prob 0.3`. The check/smoke run
  directories were removed afterwards; the rung datasets stay in
  `~/contrastive_rl/datasets/` on that node.
