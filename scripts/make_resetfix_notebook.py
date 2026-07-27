"""Generate the corrected-reset (resetfix_v1) H=700 p30 Colab notebook from the
p30 notebook. Trains with --reset-fix on the corrected dataset, then runs the
authoritative N=1000 pooled-success evaluation + naive diagnosis, both under the
corrected reset, and packages to Drive. Self-contained + resumable.

Usage:
  python scripts/make_resetfix_notebook.py --commit <sha> \
      --npz artifacts/rockfall_v2_p30_h700_resetfix/pilot/antmaze_rockfall_v2_p30_h700_resetfix_pilot.npz
"""
import argparse
import hashlib
import json

BASE = 'notebooks/naive_rockfall_v2_p30_crl.ipynb'
OUT = 'notebooks/naive_rockfall_v2_p30_h700_resetfix_crl.ipynb'


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

  for c in nb['cells']:
    src = c['source']
    if c['cell_type'] == 'markdown' and src and src[0].startswith('# '):
      src[0] = ('# CORRECTED-RESET (resetfix_v1) naive CRL 300k -- rockfall '
                'v2.1 local-detour, p_active=0.30, H=700 (Colab GPU)\n')
    if c['cell_type'] != 'code':
      continue
    if any('COMMIT ' in l for l in src):
      out = []
      for ln in src:
        if ln.startswith('COMMIT '):
          ln = f"COMMIT   = '{args.commit}'          # corrected-reset experiment commit\n"
        elif ln.startswith('DATASET_REPO_RELPATH'):
          ln = f"DATASET_REPO_RELPATH = '{args.npz}'\n"
        elif ln.startswith('DATASET_SHA256'):
          ln = f"DATASET_SHA256       = '{sha}'\n"
        elif ln.startswith('RUN_ID '):
          ln = ("RUN_ID            = "
                "f'naive_rockfall_v2_p30_h700_resetfix_s{SEED}_{MAX_UPDATES//1000}k'\n")
        out.append(ln)
        if ln.startswith('P_ACTIVE'):
          out.append("RESET_FIX   = True        # canonical episode-independent "
                     "reset (resetfix_v1)\n")
      c['source'] = out
    elif any("naive_rockfall_v2_crl.py'" in l for l in src):
      out = []
      for ln in src:
        out.append(ln)
        if "'--p-active', str(P_ACTIVE)]" in ln:
          indent = ln[:len(ln) - len(ln.lstrip())]
          out[-1] = ln.replace(
              "'--p-active', str(P_ACTIVE)]",
              "'--p-active', str(P_ACTIVE)] + (['--reset-fix'] if RESET_FIX else [])")
      c['source'] = out

  # drop the old light eval cell(s)
  while nb['cells'] and nb['cells'][-1]['cell_type'] == 'code' and \
          any('diagnose_naive_rockfall' in l or 'Behavioural eval' in l
              for l in nb['cells'][-1]['source']):
    nb['cells'].pop()

  nb['cells'].append(md(
      '## 12. Authoritative corrected-reset evaluation (N=1000 pooled) + package\n'
      'PRIMARY metric = natural sample-POOLED success (successes/1000) under the '
      'corrected reset at true cap=700. Frozen bank + all policies. Naive '
      'diagnosis adds route/exposure/drop/leakage/gaming.'))
  nb['cells'].append(code(
      "# 12a. Authoritative N=1000 pooled eval (teacher/center/blind + fresh naive)\n"
      "import subprocess, sys, os\n"
      "os.chdir('/content/'+REPO)\n"
      "RES = f'{RUN_DRIVE_DIR}/eval_resetfix'\n"
      "os.makedirs(RES, exist_ok=True)\n"
      "r = subprocess.run([sys.executable,'scripts/authoritative_eval.py',\n"
      "    '--naive-final', f'{LOCAL_RUN_DIR}/final.pkl',\n"
      "    '--naive-best', f'{LOCAL_RUN_DIR}/best.pkl',\n"
      "    '--horizon','700','--n','1000','--tag','p30_h700_resetfix',\n"
      "    '--out-dir', RES], capture_output=True, text=True)\n"
      "print(r.stdout[-2500:]); print(r.stderr[-800:] if r.returncode else '')\n"))
  nb['cells'].append(code(
      "# 12b. Naive behavioural diagnosis under corrected reset (final + best)\n"
      "for tag in ('final','best'):\n"
      "    r = subprocess.run([sys.executable,'scripts/diagnose_naive_rockfall.py',\n"
      "        '--v2','--p-active','0.30','--reset-fix','--ckpt', f'{LOCAL_RUN_DIR}/{tag}.pkl',\n"
      "        '--out-dir', f'{RES}/diag_{tag}'], capture_output=True, text=True)\n"
      "    print('===',tag,'==='); print(r.stdout[-1200:]); print(r.stderr[-500:] if r.returncode else '')\n"))
  nb['cells'].append(code(
      "# 12c. Package checkpoints + metrics + eval for download\n"
      "import shutil, glob\n"
      "pkg = f'/content/{RUN_ID}_bundle'; os.makedirs(pkg, exist_ok=True)\n"
      "for f in glob.glob(f'{LOCAL_RUN_DIR}/*.pkl')+glob.glob(f'{LOCAL_RUN_DIR}/*.json')+glob.glob(f'{LOCAL_RUN_DIR}/*.sha256'):\n"
      "    shutil.copy2(f, pkg)\n"
      "shutil.copytree(RES, f'{pkg}/eval_resetfix', dirs_exist_ok=True)\n"
      "zp = shutil.make_archive(f'{RUN_DRIVE_DIR}/{RUN_ID}_bundle','zip', pkg)\n"
      "print('packaged ->', zp)\n"))

  json.dump(nb, open(OUT, 'w'), indent=1)
  print(f'wrote {OUT} | npz sha {sha[:16]} | commit {args.commit}')


if __name__ == '__main__':
  main()
