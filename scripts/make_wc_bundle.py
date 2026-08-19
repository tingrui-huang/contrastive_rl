"""Package the untracked artifacts arms B/C need into one transferable bundle.

Arms B and C must consume the SAME static worst-case table, so it is built once
here and shipped -- never regenerated per server. Arm D needs none of this.

Produces:
  artifacts/four_arm_wc_run/common_wc_bundle/      (staged tree)
  artifacts/four_arm_wc_run/common_wc_bundle.tar.gz
  artifacts/four_arm_wc_run/common_wc_bundle.sha256
  artifacts/four_arm_wc_run/common_wc_bundle_manifest.json

Usage:  python scripts/make_wc_bundle.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

OUT = os.path.join(_ROOT, 'artifacts/four_arm_wc_run')
STAGE = os.path.join(OUT, 'common_wc_bundle')
TARBALL = os.path.join(OUT, 'common_wc_bundle.tar.gz')

# repo-relative paths that every worst-case arm needs at runtime
ITEMS = [
    # the shared static table (the whole point of the bundle)
    'artifacts/static_worstcase_rl/worstcase_table.npz',
    'artifacts/static_worstcase_rl/worstcase_table_manifest.json',
    # frozen selector provenance -- StaticWorstCase gates these SHAs on import
    'artifacts/flow_v3_diverse_failure/flow_v3/flow_v3.pkl',
    'artifacts/flow_v0_clean/norm_stats.npz',
    'artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz',
    'artifacts/settled_failure_bank_alpha01/bank_manifest.json',
    'artifacts/state_nn_selector_confirm/selector_freeze.json',
    # the training dataset
    ('artifacts/rockfall_v2_p30_h800_resetfix/failure_split/'
     'antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz'),
]
# Arm B only: the D_psi surrogate model directory
DPSI_DIR = 'artifacts/support_discriminator/D_state_cmdgoal_action'


def sha256_file(p):
  h = hashlib.sha256()
  with open(p, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def main():
  if os.path.isdir(STAGE):
    shutil.rmtree(STAGE)
  os.makedirs(STAGE)

  entries = []
  paths = list(ITEMS)
  d = os.path.join(_ROOT, DPSI_DIR)
  if os.path.isdir(d):
    for n in sorted(os.listdir(d)):
      if os.path.isfile(os.path.join(d, n)):
        paths.append(DPSI_DIR + '/' + n)

  total = 0
  for rel in paths:
    src = os.path.join(_ROOT, rel)
    if not os.path.exists(src):
      raise FileNotFoundError('bundle item missing: %s' % rel)
    dst = os.path.join(STAGE, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    size = os.path.getsize(src)
    total += size
    entries.append({'path': rel, 'sha256': sha256_file(src),
                    'bytes': size})
    print('  + %-84s %8.2f MB' % (rel, size / 1e6))

  man = {
      'purpose': ('shared runtime artifacts for the worst-case arms; the '
                  'static table is built ONCE and shipped so every arm '
                  'provably consumes the same one'),
      'needed_by': {'ArmB_dpsi': 'all items', 'ArmC_fixed': 'all except D_psi',
                    'ArmD_blind': 'NONE (only the dataset, already listed)'},
      'n_files': len(entries), 'total_bytes': total,
      'files': entries,
      'git_commit': subprocess.check_output(
          ['git', 'rev-parse', 'HEAD'], cwd=_ROOT).decode().strip()}
  json.dump(man, open(os.path.join(STAGE, 'BUNDLE_MANIFEST.json'), 'w'),
            indent=2)
  json.dump(man, open(os.path.join(OUT, 'common_wc_bundle_manifest.json'),
                      'w'), indent=2)

  with tarfile.open(TARBALL, 'w:gz') as tf:
    tf.add(STAGE, arcname='common_wc_bundle')
  bsha = sha256_file(TARBALL)
  with open(os.path.join(OUT, 'common_wc_bundle.sha256'), 'w',
            newline='\n') as f:
    f.write('%s  common_wc_bundle.tar.gz\n' % bsha)

  print('\n%d files, %.1f MB staged' % (len(entries), total / 1e6))
  print('tarball : %s (%.1f MB)' % (TARBALL, os.path.getsize(TARBALL) / 1e6))
  print('sha256  : %s' % bsha)


if __name__ == '__main__':
  main()
