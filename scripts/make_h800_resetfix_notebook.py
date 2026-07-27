"""Generate run_p30_h800_resetfix.ipynb from the corrected-reset H=700 notebook.
Same reset-fix implementation/commit lineage; ONLY the horizon changes 700->800.

COLAB CONSTRAINT: the scripted walker/base controllers are NOT committed and the
d4rl npz is absent off the workstation, so load_controllers-dependent code (the
full reset tests + the teacher/center/blind anchors) CANNOT run on Colab. The
notebook therefore (a) VERIFIES the committed reset-fix + its workstation-run
correctness report instead of re-running it, and (b) runs only the naive-policy
diagnosis on Colab. The authoritative N=1000 pooled anchor evaluation runs on the
workstation after download.

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


def md(t):
  return {'cell_type': 'markdown', 'metadata': {}, 'source': [t]}


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
  cells = []

  for i, c in enumerate(nb['cells']):
    src = c['source']
    if c['cell_type'] == 'markdown' and src and src[0].startswith('# '):
      src[0] = ('# CORRECTED-RESET (resetfix_v1) formal H=800 naive CRL 300k -- '
                'rockfall v2.1 local-detour, p_active=0.30 (Colab GPU)\n')
    # DROP the workstation-only authoritative anchor-eval cell (needs walker/base).
    if c['cell_type'] == 'code' and any('authoritative_eval.py' in l for l in src):
      continue
    if c['cell_type'] == 'markdown' and any('Authoritative corrected-reset' in l
                                            for l in src):
      c['source'] = [
          '## 12. Colab evaluation = naive-policy diagnosis only, then package\n'
          'The teacher/center/blind anchors and the authoritative N=1000 pooled '
          'table need the scripted walker/base controllers (not committed) and so '
          'run on the WORKSTATION after download. On Colab we run the naive '
          'diagnosis (route/exposure/drop/leakage/gaming) at cap-800 under the '
          'corrected reset, then package the checkpoints for the workstation eval.']
    if c['cell_type'] == 'code' and any('REQUIRE_GPU' in l for l in src):
      gpu_idx = len(cells)
    if c['cell_type'] == 'code' and any('COMMIT ' in l for l in src):
      out = []
      for ln in src:
        if ln.startswith('COMMIT '):
          ln = f"COMMIT   = '{args.commit}'          # corrected-reset H=800 (same reset-fix lineage as H=700)\n"
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
    elif c['cell_type'] == 'code' and any('run_static_audit' in l for l in src):
      out = []
      for ln in src:
        if ln.lstrip().startswith('passed, gates, rep = offline_audit.run_static_audit'):
          out.append('_c.max_episode_steps = HORIZON   # H=800: audit the 801 contract\n')
        out.append(ln)
      c['source'] = out
    elif c['cell_type'] == 'code' and any("naive_rockfall_v2_crl.py'" in l for l in src):
      c['source'] = [l.replace(
          "(['--reset-fix'] if RESET_FIX else [])",
          "(['--reset-fix'] if RESET_FIX else []) + ['--horizon', str(HORIZON)]")
          for l in src]
    elif c['cell_type'] == 'code' and any('diagnose_naive_rockfall.py' in l for l in src):
      # this cell used RES/os defined by the (dropped) anchor-eval cell -> define here
      setup = ["import subprocess, sys, os\n",
               "os.chdir('/content/'+REPO)\n",
               "RES = f'{RUN_DRIVE_DIR}/eval_h800_resetfix'\n",
               "os.makedirs(RES, exist_ok=True)\n"]
      c['source'] = setup + [l.replace("'--reset-fix',",
                                       "'--reset-fix','--horizon','800',")
                             for l in src]
    cells.append(c)
  nb['cells'] = cells

  # Lightweight reset-fix verification (Colab-safe: no walker/base, no d4rl,
  # no naive ckpt). Confirms the deployed commit carries the validated fix.
  verify = code(
      "# 5b. Reset-fix provenance verification (Colab-safe -- no re-run).\n"
      "# The full reset correctness tests (A-E) need the scripted controllers +\n"
      "# a naive checkpoint that are not on Colab; they were RUN + COMMITTED on the\n"
      "# workstation at cap-700 AND cap-800. Here we verify the deployed commit\n"
      "# carries the exact validated fix.\n"
      "os.chdir('/content/'+REPO)\n"
      "import subprocess, json as _j\n"
      "commit = subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()\n"
      "print('checked-out commit', commit)\n"
      "src = open('crl/rockfall_ant.py').read() + open('crl/d4rl_ant.py').read()\n"
      "assert 'mj_resetData' in src, 'reset fix (mj_resetData) NOT present in checkout'\n"
      "rt = _j.load(open('artifacts/reset_fix/reset_tests_h800.json'))\n"
      "assert rt['reset_fix_version']=='resetfix_v1', 'reset-fix version mismatch'\n"
      "assert rt['ALL_CORRECTED_PASS'], 'committed reset tests did NOT all pass'\n"
      "assert rt['cap']==800, 'committed reset tests not at cap-800'\n"
      "print('reset-fix VERIFIED: version', rt['reset_fix_version'],\n"
      "      '| cap', rt['cap'], '| A-E all pass (workstation-run, committed)')\n")
  nb['cells'].insert((gpu_idx + 1) if gpu_idx is not None else len(nb['cells']),
                     verify)

  json.dump(nb, open(OUT, 'w'), indent=1)
  print(f'wrote {OUT} | npz sha {sha[:16]} | commit {args.commit}')


if __name__ == '__main__':
  main()
