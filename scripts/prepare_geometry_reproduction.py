"""Explicitly reuse a saved specification's streams for an exact reproduction.

Prepares a separate directory only; never samples or starts training. Ordinary
new experiments still use the runner's fresh-namespace check.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
from ett.run_geometry_comparison import CONFIG,verify,read,write,sha


def prepare(source,root):
    verify(source)
    if root.exists():raise ValueError('reproduction needs a fresh output directory')
    root.mkdir(parents=True);(root/'cache').mkdir();(root/'checkpoints').mkdir()
    for name in ['config.json','SPEC.md','fit_contexts.npz','seed_schedule.json']:
        shutil.copyfile(source/name,root/name)
    provenance=read(source/'provenance.json')
    provenance.update(exact_reproduction_of=source.as_posix(),
        source_experiment_starting_commit=provenance['starting_commit'],
        starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        reproduction_preparer_sha256=sha(__file__))
    write(root/'provenance.json',provenance)
    write(root/'seed_audit.json',dict(intentional_seed_reuse=True,
        reason='Explicit exact reproduction of the saved configuration, not fresh independent evaluation.',
        source_experiment=source.as_posix(),source_schedule_sha256=sha(source/'seed_schedule.json')))
    write(root/'prepared.json',{p:sha(root/p) for p in ['SPEC.md','config.json','seed_schedule.json','fit_contexts.npz','provenance.json']})
    write(root/'transition_ledger.json',dict(hard_cap=CONFIG['hard_cap'],planned=CONFIG['planned_transitions'],charged=0,entries=[]))
    print('Prepared exact reproduction; no sampling or training has run.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();prepare(args.from_run,args.out_dir)
