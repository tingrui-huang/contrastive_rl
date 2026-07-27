"""Generate the formal H=800 p_active=0.30 Colab notebook from the p30 (H=700)
notebook. Sets the H=800 dataset + horizon, and adds authoritative-evaluation
and packaging cells (reconcile all-policies + naive behavioural diagnosis, both
at H=800), so the notebook is self-contained: collect(shipped)->train->eval->pack.

Usage:
  python scripts/make_h800_notebook.py --commit <sha> \
      --npz artifacts/rockfall_v2_p30_h800/pilot/antmaze_rockfall_v2_p30_h800_pilot.npz
"""
import argparse
import copy
import hashlib
import json
import os

BASE = 'notebooks/naive_rockfall_v2_p30_crl.ipynb'
OUT = 'notebooks/naive_rockfall_v2_p30_h800_crl.ipynb'


def sha256(p):
  h = hashlib.sha256()
  with open(p, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''):
      h.update(b)
  return h.hexdigest()


def md(text):
  return {'cell_type': 'markdown', 'metadata': {}, 'source': [text]}


def code(text):
  return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
          'outputs': [], 'source': [text]}


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
      src[0] = ('# Formal H=800 naive CRL (300k) -- rockfall v2.1 local-detour, '
                'p_active=0.30 (Colab GPU)\\n')
    if c['cell_type'] != 'code':
      continue
    if any('COMMIT ' in l for l in src):
      out = []
      for ln in src:
        if ln.startswith('COMMIT '):
          ln = f"COMMIT   = '{args.commit}'          # H=800 experiment frozen commit\n"
        elif ln.startswith('DATASET_REPO_RELPATH'):
          ln = f"DATASET_REPO_RELPATH = '{args.npz}'\n"
        elif ln.startswith('DATASET_SHA256'):
          ln = f"DATASET_SHA256       = '{sha}'\n"
        elif ln.startswith('RUN_ID '):
          ln = ("RUN_ID            = "
                "f'naive_rockfall_v2_p30_h800_s{SEED}_{MAX_UPDATES//1000}k'\n")
        out.append(ln)
        if ln.startswith('P_ACTIVE'):
          out.append("HORIZON     = 800         # H=800 experiment "
                     "(H=700 default is untouched)\n")
      c['source'] = out
    elif any('run_static_audit' in l for l in src):
      # G1-G8 static audit must validate against the H=800 contract (obs len
      # 801, ep-lengths <= 800), else G3/G5 fail. make_env RESETS
      # _c.max_episode_steps to the env default (700), so we set it AFTER
      # make_env, right before run_static_audit reads config.max_episode_steps+1.
      out = []
      for ln in src:
        if ln.lstrip().startswith('passed, gates, rep = offline_audit.run_static_audit'):
          out.append('_c.max_episode_steps = HORIZON   '
                     '# H=800: audit against the 801-long contract\n')
        out.append(ln)
      c['source'] = out
    elif any("naive_rockfall_v2_crl.py'" in l for l in src):
      out = []
      for ln in src:
        out.append(ln)
        if "'--seed', str(SEED), '--ckpt-dir', LOCAL_RUN_DIR," in ln:
          indent = ln[:len(ln) - len(ln.lstrip())]
          out[-1] = ln.replace(
              "'--p-active', str(P_ACTIVE)]",
              "'--p-active', str(P_ACTIVE),\n"
              f"{indent}       '--horizon', str(HORIZON)]")
      c['source'] = out

  # Replace the old light behavioural-eval cell (last code cell) with the
  # authoritative H=800 evaluation + packaging cells.
  while nb['cells'] and nb['cells'][-1]['cell_type'] == 'code' and \
          any('diagnose_naive_rockfall' in l or 'Behavioural eval'
              in l for l in nb['cells'][-1]['source']):
    nb['cells'].pop()

  nb['cells'].append(md(
      '## 12. Authoritative evaluation at H=800 (final + best) and packaging\n'
      'Reconcile harness scores naive + teacher + center + blind at H=800; the '
      'naive behavioural diagnosis adds route/exposure/drop/leakage/gaming. '
      'These are the authoritative results (training-eval was monitoring only).'))
  nb['cells'].append(code(
      "# 12a. Authoritative reconcile eval (all policies) at H=800 -- final + best\n"
      "import subprocess, sys, os\n"
      "os.chdir('/content/'+REPO)\n"
      "RES = f'{RUN_DRIVE_DIR}/eval_h800'\n"
      "os.makedirs(RES, exist_ok=True)\n"
      "for tag in ('final','best'):\n"
      "    ck = f'{LOCAL_RUN_DIR}/{tag}.pkl'\n"
      "    print('=== reconcile', tag, '===', flush=True)\n"
      "    r = subprocess.run([sys.executable,'scripts/reconcile_rockfall_eval.py',\n"
      "        '--naive-ckpt', ck, '--p-active','0.30','--horizon','800',\n"
      "        '--k','100','--n-nat','200',\n"
      "        '--out', f'{RES}/reconcile_{tag}.json'], capture_output=True, text=True)\n"
      "    print(r.stdout[-1500:]); print(r.stderr[-600:] if r.returncode else '')\n"))
  nb['cells'].append(code(
      "# 12b. Naive behavioural diagnosis at H=800 (route/exposure/drop/leakage/gaming)\n"
      "for tag in ('final','best'):\n"
      "    ck = f'{LOCAL_RUN_DIR}/{tag}.pkl'\n"
      "    print('=== diagnose', tag, '===', flush=True)\n"
      "    r = subprocess.run([sys.executable,'scripts/diagnose_naive_rockfall.py',\n"
      "        '--v2','--p-active','0.30','--horizon','800','--ckpt', ck,\n"
      "        '--out-dir', f'{RES}/diag_{tag}'], capture_output=True, text=True)\n"
      "    print(r.stdout[-1200:]); print(r.stderr[-600:] if r.returncode else '')\n"))
  nb['cells'].append(code(
      "# 12c. Package final artifacts for download (checkpoints + metrics + eval)\n"
      "import shutil\n"
      "pkg = f'/content/{RUN_ID}_bundle'\n"
      "os.makedirs(pkg, exist_ok=True)\n"
      "for f in glob.glob(f'{LOCAL_RUN_DIR}/*.pkl')+glob.glob(f'{LOCAL_RUN_DIR}/*.json')+glob.glob(f'{LOCAL_RUN_DIR}/*.sha256'):\n"
      "    shutil.copy2(f, pkg)\n"
      "shutil.copytree(RES, f'{pkg}/eval_h800', dirs_exist_ok=True)\n"
      "zp = shutil.make_archive(f'{RUN_DRIVE_DIR}/{RUN_ID}_bundle','zip', pkg)\n"
      "print('packaged ->', zp)\n"
      "print('Download from Drive:', f'{RUN_DRIVE_DIR}/{RUN_ID}_bundle.zip')\n"))

  json.dump(nb, open(OUT, 'w'), indent=1)
  print(f'wrote {OUT} | npz sha {sha[:16]} | commit {args.commit}')


if __name__ == '__main__':
  main()
