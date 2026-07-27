"""Generate run_p30_h800_resetfix.ipynb from the corrected-reset H=700 notebook.
Same reset-fix implementation/commit lineage; ONLY the horizon changes 700->800.
Adds a startup reset-correctness gate at cap-800 and the H=800 audit-config fix.
Self-contained + resumable, distinct H=800 namespace.

Usage:
  python scripts/make_h800_resetfix_notebook.py --commit <sha> \
      --npz artifacts/rockfall_v2_p30_h800_resetfix/pilot/antmaze_rockfall_v2_p30_h800_resetfix_pilot.npz
"""
import argparse
import hashlib
import json

BASE = 'notebooks/naive_rockfall_v2_p30_h700_resetfix_crl.ipynb'
OUT = 'notebooks/run_p30_h800_resetfix.ipynb'


def sha256(p):
  h = hashlib.sha256()
  with open(p, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def code(t):
  return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
          'outputs': [], 'source': [t]}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--commit', required=True)
  ap.add_argument('--npz', required=True)
  args = ap.parse_args()
  sha = sha256(args.npz)
  nb = json.load(open(BASE))
  gpu_idx = None

  for i, c in enumerate(nb['cells']):
    src = c['source']
    if c['cell_type'] == 'markdown' and src and src[0].startswith('# '):
      src[0] = ('# CORRECTED-RESET (resetfix_v1) formal H=800 naive CRL 300k -- '
                'rockfall v2.1 local-detour, p_active=0.30 (Colab GPU)\n')
    if c['cell_type'] != 'code':
      continue
    if any('REQUIRE_GPU' in l for l in src):
      gpu_idx = i
    if any('COMMIT ' in l for l in src):
      out = []
      for ln in src:
        if ln.startswith('COMMIT '):
          ln = f"COMMIT   = '{args.commit}'          # corrected-reset H=800 commit (same reset-fix lineage as H=700)\n"
        elif ln.startswith('DATASET_REPO_RELPATH'):
          ln = f"DATASET_REPO_RELPATH = '{args.npz}'\n"
        elif ln.startswith('DATASET_SHA256'):
          ln = f"DATASET_SHA256       = '{sha}'\n"
        elif ln.startswith('RUN_ID '):
          ln = ("RUN_ID            = "
                "f'naive_rockfall_v2_p30_h800_resetfix_s{SEED}_{MAX_UPDATES//1000}k'\n")
        out.append(ln)
        if ln.startswith('RESET_FIX'):
          out.append("HORIZON     = 800         # ONLY change vs corrected H=700\n")
      c['source'] = out
    elif any('run_static_audit' in l for l in src):
      out = []
      for ln in src:
        if ln.lstrip().startswith('passed, gates, rep = offline_audit.run_static_audit'):
          out.append('_c.max_episode_steps = HORIZON   # H=800: audit the 801 contract\n')
        out.append(ln)
      c['source'] = out
    elif any("naive_rockfall_v2_crl.py'" in l for l in src):
      c['source'] = [l.replace(
          "(['--reset-fix'] if RESET_FIX else [])",
          "(['--reset-fix'] if RESET_FIX else []) + ['--horizon', str(HORIZON)]")
          for l in src]
    elif any('authoritative_eval.py' in l for l in src):
      c['source'] = [l.replace("'--horizon','700'", "'--horizon','800'")
                     .replace("'p30_h700_resetfix'", "'p30_h800_resetfix'")
                     .replace("eval_resetfix", "eval_h800_resetfix") for l in src]
    elif any('diagnose_naive_rockfall.py' in l for l in src):
      c['source'] = [l.replace("'--reset-fix',",
                               "'--reset-fix','--horizon','800',") for l in src]

  # Startup reset-correctness gate at cap-800 (right after GPU verification).
  gate = code(
      "# 5b. Reset-fix provenance + correctness gate at cap-800 (STOP if any fail).\n"
      "os.chdir('/content/'+REPO)\n"
      "import subprocess\n"
      "commit = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()\n"
      "print('checked-out commit', commit)\n"
      "r = subprocess.run([sys.executable,'scripts/test_reset_independence.py',\n"
      "    '--horizon','800','--out', f'{RUN_DRIVE_DIR}/reset_tests_h800.json'],\n"
      "    capture_output=True, text=True)\n"
      "print(r.stdout[-1500:]); print(r.stderr[-600:] if r.returncode else '')\n"
      "import json as _j\n"
      "_rt = _j.load(open(f'{RUN_DRIVE_DIR}/reset_tests_h800.json'))\n"
      "assert _rt['reset_fix_version']=='resetfix_v1', 'reset-fix version mismatch'\n"
      "assert _rt['ALL_CORRECTED_PASS'], 'RESET TESTS FAILED -- stop before collection'\n"
      "print('reset gate PASS | version', _rt['reset_fix_version'], '| cap', _rt['cap'])\n")
  if gpu_idx is not None:
    nb['cells'].insert(gpu_idx + 1, gate)
  else:
    nb['cells'].append(gate)

  json.dump(nb, open(OUT, 'w'), indent=1)
  print(f'wrote {OUT} | npz sha {sha[:16]} | commit {args.commit}')


if __name__ == '__main__':
  main()
