"""Generate the p_active-sweep Colab training notebooks from the reference
naive_rockfall_v2_crl.ipynb. One notebook per condition, pinned to a frozen
commit, pointed at that condition's committed dataset (sha-verified), with
--p-active threaded into the training launch.

Run AFTER the sweep datasets are collected+audited+committed, passing the
pinned commit. Reads each dataset's real sha256 from disk.

Usage:
  python scripts/make_sweep_notebooks.py --commit <sha> \
      --conditions p30:0.30:artifacts/rockfall_v2_dataset_p30/pilot/antmaze_rockfall_v2_p30_pilot.npz \
                   p50:0.50:artifacts/rockfall_v2_dataset_p50/pilot/antmaze_rockfall_v2_p50_pilot.npz
"""
import argparse
import copy
import hashlib
import json
import os
import sys

REF = 'notebooks/naive_rockfall_v2_crl.ipynb'


def sha256(path, chunk=1 << 20):
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for b in iter(lambda: f.read(chunk), b''):
      h.update(b)
  return h.hexdigest()


def edit_config_cell(src, tag, p_active, relpath, sha, commit):
  out = []
  for ln in src:
    if ln.startswith('COMMIT '):
      ln = f"COMMIT   = '{commit}'          # p_active={p_active} sweep frozen commit\n"
    elif ln.startswith('DATASET_REPO_RELPATH'):
      ln = f"DATASET_REPO_RELPATH = '{relpath}'\n"
    elif ln.startswith('DATASET_SHA256'):
      ln = f"DATASET_SHA256       = '{sha}'\n"
    elif ln.startswith('RUN_ID '):
      ln = ("RUN_ID            = "
            f"f'naive_rockfall_v2_{tag}_s{{SEED}}_{{MAX_UPDATES//1000}}k'\n")
    out.append(ln)
    if ln.startswith('REQUIRE_GPU'):
      out.append(f"P_ACTIVE    = {p_active}          "
                 "# mask density for THIS sweep condition\n")
  return out


def edit_launch_cell(src):
  out = []
  for ln in src:
    out.append(ln)
    if "'--seed', str(SEED), '--ckpt-dir', LOCAL_RUN_DIR]" in ln:
      # append p_active to the arg list (same indentation as the list literal)
      indent = ln[:len(ln) - len(ln.lstrip())]
      out[-1] = ln.replace(
          "'--seed', str(SEED), '--ckpt-dir', LOCAL_RUN_DIR]",
          "'--seed', str(SEED), '--ckpt-dir', LOCAL_RUN_DIR,\n"
          f"{indent}       '--p-active', str(P_ACTIVE)]")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--commit', required=True)
  ap.add_argument('--conditions', nargs='+', required=True,
                  help='tag:p_active:npz_relpath triples')
  args = ap.parse_args()
  ref = json.load(open(REF))
  for spec in args.conditions:
    tag, p_str, relpath = spec.split(':', 2)
    p_active = float(p_str)
    if not os.path.exists(relpath):
      sys.exit(f'dataset missing: {relpath}')
    sha = sha256(relpath)
    nb = copy.deepcopy(ref)
    for c in nb['cells']:
      src = c['source']
      if c['cell_type'] == 'code' and any('COMMIT ' in l for l in src):
        c['source'] = edit_config_cell(src, tag, p_active, relpath, sha,
                                        args.commit)
      elif c['cell_type'] == 'code' and any(
          "naive_rockfall_v2_crl.py'" in l for l in src):
        c['source'] = edit_launch_cell(src)
      elif c['cell_type'] == 'markdown' and src and src[0].startswith('# '):
        c['source'][0] = (f'# Naive offline CRL 300k -- rockfall v2.1 '
                          f'local-detour, p_active={p_active} '
                          f'({tag}) sweep (Colab GPU)\n')
    out = f'notebooks/naive_rockfall_v2_{tag}_crl.ipynb'
    json.dump(nb, open(out, 'w'), indent=1)
    print(f'wrote {out} | p_active={p_active} sha={sha[:16]}... commit={args.commit}')


if __name__ == '__main__':
  main()
