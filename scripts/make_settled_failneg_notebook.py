"""Generate the SETTLED-BANK alpha=0.1 Colab notebook (failure-bank rebuild).

Derives notebooks/run_failneg_settledbank_a01.ipynb from the authoritative
Part-1 notebook (notebooks/run_failneg_p30_h800_resetfix_a01.ipynb), changing
ONLY:

  * failure bank -> artifacts/settled_failure_bank_alpha01/
    failure_bank_settled.npz (16 N=80 physically-settled fatal states of the
    SAME 16 pilot death episodes; scripts/rebuild_failure_bank_settled.py);
  * run id / title;
  * the bank-verification cell (settled-bank sha + rebuild-manifest checks;
    still verifies the UNCHANGED clean npz sha and split provenance).

Everything else -- clean dataset, recipe, seed, steps, audit gates, eval
cadence, launch command shape, the diag_final/diag_best evaluation protocol --
is inherited verbatim, so the run is directly comparable to
failneg_clean_p30_h800_resetfix_a01_s0_300k (legacy bank) and the a0 baseline.

Usage: python scripts/make_settled_failneg_notebook.py
       # requires the settled bank + manifest to be COMMITTED (the notebook
       # verifies shas after cloning branch head).
"""
import copy
import json

BASE = 'notebooks/run_failneg_p30_h800_resetfix_a01.ipynb'
OUT = 'notebooks/run_failneg_settledbank_a01.ipynb'

NEW_BANK_RELPATH = 'artifacts/settled_failure_bank_alpha01/failure_bank_settled.npz'
NEW_MANIFEST_RELPATH = 'artifacts/settled_failure_bank_alpha01/bank_manifest.json'
MAN = json.load(open(NEW_MANIFEST_RELPATH))
NEW_BANK_SHA = MAN['bank']['sha256']
OLD_BANK_SHA = MAN['old_bank']['sha256']
CLEAN_SHA = MAN['clean_npz']['sha256']

TITLE = '''# SETTLED FAILURE BANK -- clean-data CRL, alpha=0.1 (Colab GPU)

Identical to the Part-1 alpha=0.1 failure-negative run
(`failneg_clean_p30_h800_resetfix_a01_s0_300k`) in every respect -- clean
dataset, recipe, seed, steps, eval protocol -- EXCEPT the failure bank:

    legacy bank (16 healthy-looking frozen death poses)
        -> settled bank (16 N=80 physically-settled fatal observations)

The settled states come from the SAME 16 authoritative pilot death episodes,
re-collected under the death-settle physics patch (actor control zeroed at
the fatal contact, 80 extra MuJoCo substeps inside the fatal transition; see
scripts/rebuild_failure_bank_settled.py + bank_manifest.json: pre-contact
prefixes reproduce the pilot bitwise). The 40 fresh diagnostic deaths are
HELD OUT and never touch this run.

Loss (unchanged): L = L_pos + (1-alpha) L_ordinary-neg + alpha L_failure-neg,
alpha = 0.1, exact uniform expectation over the 16-state bank.
'''

VERIFY_CELL = '''# 6b. Verify the SETTLED failure bank + rebuild manifest + split provenance.
import json as _j
got = sha256(f'/content/{REPO}/{BANK_REPO_RELPATH}')
if got != BANK_SHA256:
    raise SystemExit('settled failure-bank sha mismatch: got ' + got)
bman = _j.load(open(f'/content/{REPO}/{NEW_MANIFEST_RELPATH}'))
assert all(bman['checks'].values()), ('bank rebuild validation not clean: '
    + str({k: v for k, v in bman['checks'].items() if not v}))
assert bman['bank']['sha256'] == BANK_SHA256
assert bman['bank']['n_states'] == 16 and bman['bank']['state_dim'] == 29
assert bman['bank']['death_settle_substeps'] == 80
assert bman['old_bank']['sha256'] == LEGACY_BANK_SHA256, 'legacy bank drifted'
assert bman['clean_npz']['sha256'] == DATASET_SHA256, 'clean npz sha mismatch'
assert bman['heldout_fresh_deaths']['n'] == 40, 'held-out list missing'
man = _j.load(open(f'/content/{REPO}/{MANIFEST_RELPATH}'))
assert man['clean']['sha256'] == DATASET_SHA256, 'split clean sha mismatch'
assert man['source_sha256'] == SOURCE_DATASET_SHA256, 'split source mismatch'
assert man['clean']['n_episodes'] == 284 and man['rockfail']['n_episodes'] == 16
assert bman['per_state_provenance'][0].keys() >= {'episode_id',
    'prefix_bitwise_ok'}
assert all(r['prefix_bitwise_ok'] and r['settled_matches_sweep_trace']
           and r['differs_from_legacy'] for r in bman['per_state_provenance'])
print('settled bank verified: 16 N=80 states of the original pilot deaths; '
      'clean npz + legacy bank untouched; 40 fresh deaths held out')
'''


def main():
  nb = copy.deepcopy(json.load(open(BASE)))
  cells = nb['cells']
  assert cells[0]['cell_type'] == 'markdown'
  cells[0]['source'] = TITLE

  # config cell: swap the bank, tag the run id, add the extra shas.
  ci = next(i for i, c in enumerate(cells) if c['cell_type'] == 'code')
  src = ''.join(cells[ci]['source'])
  old_bank_line = next(l for l in src.split('\n')
                       if l.startswith('BANK_REPO_RELPATH'))
  old_sha_line = next(l for l in src.split('\n')
                      if l.startswith('BANK_SHA256'))
  src = src.replace(old_bank_line,
                    f"BANK_REPO_RELPATH    = '{NEW_BANK_RELPATH}'")
  src = src.replace(
      old_sha_line,
      f"BANK_SHA256          = '{NEW_BANK_SHA}'\n"
      f"LEGACY_BANK_SHA256   = '{OLD_BANK_SHA}'\n"
      f"NEW_MANIFEST_RELPATH = '{NEW_MANIFEST_RELPATH}'")
  src = src.replace(
      "RUN_ID            = f'failneg_clean_p30_h800_resetfix_a01_",
      "RUN_ID            = f'failneg_settledbank_p30_h800_resetfix_a01_")
  assert NEW_BANK_RELPATH in src and 'settledbank' in src
  cells[ci]['source'] = src

  # replace the bank-verification cell (6b).
  vi = next(i for i, c in enumerate(cells)
            if c['cell_type'] == 'code'
            and 'failure bank' in ''.join(c['source'])
            and 'sha mismatch' in ''.join(c['source']))
  cells[vi]['source'] = VERIFY_CELL

  with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
  print('wrote', OUT)
  print('bank :', NEW_BANK_RELPATH, NEW_BANK_SHA[:16] + '...')
  print('clean:', CLEAN_SHA[:16] + '... (unchanged)')


if __name__ == '__main__':
  main()
