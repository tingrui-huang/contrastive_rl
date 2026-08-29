"""Merge the bad-demonstrator episodes into the frozen windy-swamp dataset.

crl.offline_audit.build_offline_buffer takes ONE path, so the bad demonstrator
has to be concatenated rather than loaded alongside. Layout is identical by
construction (scripts/collect_swamp_windy_baddemo.py mirrors
scripts/collect_swamp_windy.py), so this is a pure episode-axis concatenation
of every array; nothing is reshaped and no column is added.

ORDER: main episodes keep indices [0, n_main), the bad demonstrator occupies
[n_main, n_main + n_bad). Appending rather than interleaving keeps every
existing episode index stable, so anchor cuts, balanced buckets and the failure
bank's recorded episode_ids all still point at the same trajectories.

WHY teacher_mode MATTERS HERE. The merged file carries code 4 = bad_demo on
exactly the appended episodes, so every downstream audit can select or exclude
them without re-deriving anything. That code is an AUDIT field: it has a
per-episode leading dimension and lives outside obs/act, which is what gate G6
(NO_AUDIT_LEAK) checks. The learner observation stays [x, y, gx, gy]; the
confounder is still not exposed.

The bad demonstrator is POSITIVE-side data. It is not a psi-side negative bank
and this script does not build one -- see the docstring of
scripts/collect_swamp_windy_baddemo.py for why that distinction is the whole
point.

Run:
  python -m scripts.merge_swamp_windy_baddemo \
      --main datasets/swamp_windy_teacher_s0.npz \
      --bad  datasets/swamp_windy_baddemo_s0.npz \
      --out  datasets/swamp_windy_merged_s0.npz
"""
import argparse
import hashlib
import json
import os

import numpy as np

# Episode-axis arrays that must be concatenated. Everything else in the file is
# scalar metadata.
EP_KEYS = ('obs', 'act', 'swamp_bits', 'route_label', 'teacher_mode',
           'force_safe', 'wait_count', 'entered_active_swamp')
BAD_DEMO_CODE = 4


def sha256_file(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def content_sha(path):
  """sha256 over the ARRAY CONTENTS, ignoring zip container metadata.

  EVERY array is hashed, ``meta`` included -- the same single convention as
  scripts/run_swamp_windy_sweep.sh and run_swamp_windy_failneg.py, so a merged
  dataset can be pinned by the existing provenance gate with no special case.

  That only works because the meta this script writes is itself reproducible:
  it records source BASENAMES and source CONTENT hashes, never absolute paths
  and never source FILE hashes. Both of those would vary across machines (a
  different checkout directory; zip timestamps), which would make a correct
  regeneration look like a content mismatch. Non-reproducible provenance --
  absolute paths, file shas -- goes in the sidecar .manifest.json instead,
  which is not part of the npz and therefore not hashed.
  """
  h = hashlib.sha256()
  with np.load(path, allow_pickle=False) as d:
    for k in sorted(d.files):
      a = d[k]
      h.update(k.encode())
      h.update(str(a.dtype).encode())
      h.update(str(a.shape).encode())
      h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def merge(main_path, bad_path, out_path, force=False):
  if os.path.exists(out_path) and not force:
    raise SystemExit(f'REFUSING to overwrite {out_path} (use --force).')

  M = np.load(main_path, allow_pickle=False)
  B = np.load(bad_path, allow_pickle=False)

  mk, bk = set(M.files), set(B.files)
  if mk != bk:
    raise SystemExit(f'key sets differ.\n  only in main: {sorted(mk - bk)}'
                     f'\n  only in bad : {sorted(bk - mk)}')
  missing = [k for k in EP_KEYS if k not in mk]
  if missing:
    raise SystemExit(f'missing episode-axis arrays: {missing}')

  n_main, n_bad = M['obs'].shape[0], B['obs'].shape[0]
  # Trailing dims must agree exactly or the concatenation would silently
  # broadcast a differently-shaped episode into the buffer.
  for k in EP_KEYS:
    a, b = M[k], B[k]
    if a.shape[1:] != b.shape[1:]:
      raise SystemExit(f'{k}: trailing shape {a.shape[1:]} vs {b.shape[1:]}')
    if a.dtype != b.dtype:
      raise SystemExit(f'{k}: dtype {a.dtype} vs {b.dtype}')
    if a.shape[0] != (n_main if k != 'x' else 0) or b.shape[0] != n_bad:
      raise SystemExit(f'{k}: leading dim {a.shape[0]}/{b.shape[0]} != '
                       f'{n_main}/{n_bad}')

  if not np.all(np.asarray(B['teacher_mode']) == BAD_DEMO_CODE):
    raise SystemExit('bad-demo file has episodes whose teacher_mode != '
                     f'{BAD_DEMO_CODE}; refusing to merge an unlabelled file.')
  if BAD_DEMO_CODE in set(np.unique(np.asarray(M['teacher_mode'])).tolist()):
    raise SystemExit(f'main file already uses teacher_mode {BAD_DEMO_CODE}; '
                     'the merged file could not be un-merged.')

  out = {k: np.concatenate([M[k], B[k]], axis=0) for k in EP_KEYS}

  m_meta = json.loads(str(M['meta'])) if 'meta' in mk else {}
  b_meta = json.loads(str(B['meta'])) if 'meta' in bk else {}
  died = np.asarray(out['entered_active_swamp']).astype(bool)
  mode = np.asarray(out['teacher_mode'])
  meta = dict(m_meta)
  meta.update({
      'setting': 'windy_lethal_merged_with_bad_demonstrator',
      'episodes': int(n_main + n_bad),
      'n_transitions': int((n_main + n_bad) * (out['obs'].shape[1] - 1)),
      'merge': {
          # basenames + CONTENT hashes only: see content_sha() for why absolute
          # paths and file shas must not enter the npz meta.
          'main_name': os.path.basename(main_path),
          'main_content_sha256': content_sha(main_path),
          'main_episodes': int(n_main), 'main_index_range': [0, int(n_main)],
          'bad_name': os.path.basename(bad_path),
          'bad_content_sha256': content_sha(bad_path),
          'bad_episodes': int(n_bad),
          'bad_index_range': [int(n_main), int(n_main + n_bad)],
          'bad_demo_teacher_mode_code': BAD_DEMO_CODE,
          'bad_demo_meta': b_meta,
      },
      'teacher_mode_frequencies': {
          str(int(m)): int((mode == m).sum()) for m in np.unique(mode)},
      'died_rate_overall': float(died.mean()),
      'died_rate_main': float(died[:n_main].mean()),
      'died_rate_bad_demo': float(died[n_main:].mean()),
      'note': 'bad-demo episodes are POSITIVE-side support for the '
              '(forward | U active) branch; obs is still [x,y,gx,gy] and the '
              'confounder is not exposed',
  })

  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  np.savez(out_path, meta=np.array(json.dumps(meta)), **out)
  f_sha, c_sha = sha256_file(out_path), content_sha(out_path)
  # The sidecar holds everything that is NOT reproducible across machines.
  manifest = dict(path=os.path.abspath(out_path), sha256=f_sha,
                  content_sha256=c_sha,
                  main_path=os.path.abspath(main_path),
                  main_file_sha256=sha256_file(main_path),
                  bad_path=os.path.abspath(bad_path),
                  bad_file_sha256=sha256_file(bad_path),
                  size_bytes=int(os.path.getsize(out_path)),
                  obs_shape=list(out['obs'].shape),
                  act_shape=list(out['act'].shape), frozen=True, meta=meta)
  with open(out_path + '.manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
  try:
    os.chmod(out_path, 0o444)
  except OSError:
    pass

  print(f'merged -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)')
  print(f'  episodes   {n_main} main + {n_bad} bad_demo = {n_main + n_bad}')
  print(f'  obs        {out["obs"].shape}   act {out["act"].shape}')
  print(f'  died       overall {died.mean():.3f}  '
        f'(main {died[:n_main].mean():.3f}, bad {died[n_main:].mean():.3f})')
  print(f'  file sha   {f_sha}')
  print(f'  content sha{c_sha}')
  return out_path, c_sha


def audit(path):
  """Run the same static gates the launcher runs, on the merged file."""
  from crl.config import Config
  from crl import envs as envs_mod
  from crl.offline_audit import run_static_audit

  cfg = Config(env_name='point_two_route_swamp_windy_v0')
  envs_mod.make_env(cfg.env_name, cfg, seed=0)     # fills obs/goal/action dims
  cfg.dataset_path = path
  _, gates, _ = run_static_audit(path, cfg)
  print('\nSTATIC AUDIT')
  for k in sorted(gates):
    print(f'  {"PASS" if gates[k] else "FAIL"}  {k}')
  bad = [k for k, v in gates.items() if not v]
  print(f'\n{len(gates) - len(bad)}/{len(gates)} gates pass'
        + (f'  -- FAILED: {bad}' if bad else ''))
  return not bad


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument('--main', default='datasets/swamp_windy_teacher_s0.npz')
  p.add_argument('--bad', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--force', action='store_true')
  p.add_argument('--no-audit', action='store_true')
  a = p.parse_args()
  if a.force and os.path.exists(a.out):
    try:
      os.chmod(a.out, 0o644)
    except OSError:
      pass
  path, _ = merge(a.main, a.bad, a.out, force=a.force)
  if not a.no_audit:
    ok = audit(path)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
  main()
