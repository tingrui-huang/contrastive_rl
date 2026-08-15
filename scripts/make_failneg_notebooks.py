"""Generate the failure-aware-negative-sampling Colab notebooks (Part 1).

Derives one notebook per alpha in {0, 0.1, 0.3, 0.5} from the authoritative
corrected-reset H=800 notebook (notebooks/run_p30_h800_resetfix.ipynb),
changing ONLY:

  * dataset -> D_clean (284 eps; the 16 rockfall-death episodes removed),
  * failure-negative flags --fail-bank/--fail-neg-alpha (alpha>0 runs),
  * run id, title, and a bank/manifest sha-verification cell.

Everything else (recipe, audit gates, eval cadence, reset-fix provenance
check, Drive mirroring, cap-800 diagnosis) is inherited verbatim, so the runs
stay comparable with naive_rockfall_v2_p30_h800_resetfix_s0_300k (= control C
on D_original).

Usage:
  python scripts/make_failneg_notebooks.py --commit <sha>   # pin (recommended)
  python scripts/make_failneg_notebooks.py                  # track branch head

Note: the failure_split artifacts + this code must be COMMITTED (and pushed)
at the referenced commit/branch -- the notebooks verify shas after cloning.
"""
import argparse
import copy
import json
import subprocess

BASE = 'notebooks/run_p30_h800_resetfix.ipynb'
OUT_TMPL = 'notebooks/run_failneg_p30_h800_resetfix_a{tag}.ipynb'

SPLIT_RELDIR = 'artifacts/rockfall_v2_p30_h800_resetfix/failure_split'
CLEAN_RELPATH = (SPLIT_RELDIR +
                 '/antmaze_rockfall_v2_p30_h800_resetfix_pilot_clean.npz')
BANK_RELPATH = SPLIT_RELDIR + '/failure_bank.npz'
MANIFEST_RELPATH = SPLIT_RELDIR + '/failure_split_manifest.json'

MAN = json.load(open(MANIFEST_RELPATH))
CLEAN_SHA = MAN['clean']['sha256']
BANK_SHA = MAN['failure_bank']['sha256']
SRC_SHA = MAN['source_sha256']

ALPHAS = (0.0, 0.1, 0.3, 0.5)

# the exact tail of the base notebook's training-command construction; the
# failure-negative flags are appended right after it.
LAUNCH_ANCHOR = "+ ['--horizon', str(HORIZON)]"


def tag_of(alpha):
  return str(alpha).replace('.', '').rstrip('0') or '0'   # 0.1->01, 0.0->0


def make_config_cell(alpha, branch, commit):
  tag = tag_of(alpha)
  return f'''# 1. Configuration. Everything tunable lives here.
import os

SEED        = 0
MAX_UPDATES = 300_000
RESUME      = False           # set True after a Colab disconnect (restores latest.pkl)
REQUIRE_GPU = True
P_ACTIVE    = 0.3          # mask density (must match the dataset's collection density)
RESET_FIX   = True        # canonical episode-independent reset (resetfix_v1)
HORIZON     = 800         # H=800 protocol (per Part 1 experiment decision)

# --- failure-aware negative sampling (THE experimental variable) ---------------
ALPHA       = {alpha}          # mixture weight on the critic NEGATIVE distribution
                             # 0 => clean baseline (experiment A); >0 => experiment B

# --- repo (pinned to the failure-split commit) ---------------------------------
REPO     = 'contrastive_rl'
REPO_URL = 'https://github.com/tingrui-huang/contrastive_rl.git'
BRANCH   = '{branch}'
COMMIT   = '{commit}'

# --- dataset: D_clean = D_original minus the 16 rockfall-death episodes --------
# (ships WITH the repo; the failure bank = 16 terminal buried poses of the
#  removed episodes, used ONLY as biased negatives, never as anchors/positives)
ENV_NAME             = 'offline_ant_umaze_rockfall'
DATASET_REPO_RELPATH = '{CLEAN_RELPATH}'
DATASET_SHA256       = '{CLEAN_SHA}'
BANK_REPO_RELPATH    = '{BANK_RELPATH}'
BANK_SHA256          = '{BANK_SHA}'
MANIFEST_RELPATH     = '{MANIFEST_RELPATH}'
SOURCE_DATASET_SHA256 = '{SRC_SHA}'  # the authoritative D_original this was split from

# --- run dirs --------------------------------------------------------------------
RUN_ID            = f'failneg_clean_p30_h800_resetfix_a{tag}_s{{SEED}}_{{MAX_UPDATES//1000}}k'
LOCAL_RUN_DIR     = f'/content/runs/{{RUN_ID}}'
RUN_DRIVE_DIR     = f'/content/drive/MyDrive/contrastive_rl_runs/{{RUN_ID}}'
LOCAL_DATASET_PATH = f'/content/{{REPO}}/{{DATASET_REPO_RELPATH}}'
LOCAL_BANK_PATH    = f'/content/{{REPO}}/{{BANK_REPO_RELPATH}}'
print(RUN_ID, '| alpha =', ALPHA)
'''


BANK_VERIFY_CELL = '''# 6b. Verify the failure bank + split manifest (sha256 + provenance).
import json as _j
if ALPHA > 0:
    got = sha256(f'/content/{REPO}/{BANK_REPO_RELPATH}')
    if got != BANK_SHA256:
        raise SystemExit('failure-bank sha mismatch: got ' + got)
man = _j.load(open(f'/content/{REPO}/{MANIFEST_RELPATH}'))
assert man['source_sha256'] == SOURCE_DATASET_SHA256, 'split source mismatch'
assert man['clean']['sha256'] == DATASET_SHA256, 'clean npz sha mismatch'
assert man['clean']['n_episodes'] == 284 and man['rockfail']['n_episodes'] == 16
assert man['failure_bank']['n_states'] == 16
print('failure split verified: clean 284 eps / rockfail 16 eps / bank 16 states')
print('split of source', man['source_sha256'][:16] + '...')
'''

TITLE = '''# FAILURE-AWARE NEGATIVES (Part 1) -- clean-data CRL, alpha={alpha} (Colab GPU)

Faithful offline CRL (corrected reset, H=800, p_active=0.30, v2.1 severity
0.80/0.15/0.05) on **D_clean** = the authoritative 300-episode dataset MINUS
the 16 rockfall-death episodes. The ONLY algorithmic change vs the baseline
recipe is the critic NEGATIVE distribution:

    q_alpha(g) = (1-alpha) * p_clean(g) + alpha * q_fail(g)

implemented as a loss-level mixture that PRESERVES the original
positive/negative weighting exactly:

    L(alpha) = L_pos + (1-alpha) * L_ordinary-neg + alpha * L_failure-neg

where L_failure-neg is the exact uniform expectation over the 16-state
failure bank (terminal buried poses of the removed episodes; all states
scored each step, no sampling noise). The positive term and the total
negative mass keep their original coefficients; alpha moves ONLY the mixture
inside the negative term. alpha=0 is the byte-identical clean baseline
(experiment A). Control C = `naive_rockfall_v2_p30_h800_resetfix_s0_300k`
(normal CRL on D_original).

Anchors/positives come ONLY from D_clean; the bank states can never be
positives (verified: scripts/sanity_fail_negatives.py, all gates PASS).
'''


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--commit', default='',
                  help='pin the repo to this sha (empty: track origin/BRANCH)')
  ap.add_argument('--branch', default=None)
  args = ap.parse_args()
  branch = args.branch or subprocess.run(
      ['git', 'branch', '--show-current'], capture_output=True,
      text=True).stdout.strip()

  base = json.load(open(BASE))
  outs = []
  for alpha in ALPHAS:
    nb = copy.deepcopy(base)
    cells = nb['cells']

    # title
    assert cells[0]['cell_type'] == 'markdown'
    cells[0]['source'] = TITLE.format(alpha=alpha)

    # config cell = first code cell
    ci = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code')
    cells[ci]['source'] = make_config_cell(alpha, branch, args.commit)

    # dataset markdown (## 6): correct the episode count text
    for c in cells:
      if c['cell_type'] == 'markdown' and '300-episode pilot npz' in ''.join(
          c['source']):
        c['source'] = ('## 6. Dataset: verify the repo-shipped CLEAN npz '
                       '(sha256)\n\nD_clean (284 episodes) + failure bank are '
                       'committed at the pinned commit -- nothing to upload.\n')

    # insert bank verification right after the dataset-verify code cell
    vi = next(i for i, c in enumerate(cells)
              if c['cell_type'] == 'code'
              and 'dataset sha mismatch' in ''.join(c['source']))
    cells.insert(vi + 1, {'cell_type': 'code', 'execution_count': None,
                          'metadata': {}, 'outputs': [],
                          'source': BANK_VERIFY_CELL})

    # launch cell: add the failure-negative flags
    li = next(i for i, c in enumerate(cells)
              if c['cell_type'] == 'code'
              and 'cmd = [' in ''.join(c['source'])
              and 'naive_rockfall_v2_crl.py' in ''.join(c['source']))
    src = ''.join(cells[li]['source'])
    assert LAUNCH_ANCHOR in src, 'launch cell changed upstream; fix the anchor'
    src = src.replace(
        LAUNCH_ANCHOR,
        LAUNCH_ANCHOR + (" + (['--fail-bank', LOCAL_BANK_PATH,"
                         " '--fail-neg-alpha', str(ALPHA)]"
                         " if ALPHA > 0 else [])"))
    cells[li]['source'] = src

    out = OUT_TMPL.format(tag=tag_of(alpha))
    with open(out, 'w') as f:
      json.dump(nb, f, indent=1)
    outs.append(out)
    print('wrote', out)

  print('\nRun on Colab (any order):')
  for o in outs:
    print(' ', o)
  print('control C = existing naive_rockfall_v2_p30_h800_resetfix_s0_300k')


if __name__ == '__main__':
  main()
