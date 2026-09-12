"""Verify completed rollout artifacts and replay saved checkpoints under fixed keys."""
import argparse
import ast
import json
from pathlib import Path
import subprocess

import jax
import numpy as np

from ett.anchored_transition import load_anchored
from ett.rollout_return import GOAL,START,DISCOUNT,TaskRollout
from ett.run_rollout_return import SOURCE,read
from ett.run_distribution_matching import frozen_actor,write_json
from propensity.nominal_policy import load_nominal_policy
from scripts.make_swamp_f4_failure_bank import file_sha


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir',required=True)
  args=parser.parse_args();out=Path(args.run_dir)
  config=read(out/'config.json');search=read(out/'search_config.json')
  if config['status']!='complete':raise ValueError('run is incomplete')
  for path,expected in config['input_file_sha256'].items():
    if file_sha(path)!=expected:raise RuntimeError(f'preserved input changed: {path}')
  sources={**config['source_file_sha256'],'ett/search_rollout_return.py':search['source_sha256']}
  for path,expected in sources.items():
    if file_sha(path)!=expected:raise RuntimeError(f'experiment source changed: {path}')
  added_sources=[*sources,'ett/report_rollout_return.py',__file__]
  for path in added_sources:
    source=Path(path).read_text(encoding='utf-8');ast.parse(source)
    if any('\u4e00'<=c<='\u9fff' for c in source):raise RuntimeError(f'non-English source: {path}')
  def reject_nonfinite(value):raise ValueError(f'non-finite JSON constant: {value}')
  json_files=list(out.rglob('*.json'))
  for path in json_files:
    json.loads(path.read_text(encoding='utf-8'),parse_constant=reject_nonfinite)
  nominal=load_nominal_policy(Path(config['paths']['nominal']).parent)
  actor,_=frozen_actor(config['paths']['actor'],read(SOURCE/'config.json')['references']['dataset_content_sha256'])
  evaluation=read(out/'reset_evaluation.json');checks={}
  for name,metrics in evaluation['models'].items():
    checkpoint=out/f'{name}.pkl'
    if file_sha(checkpoint)!=metrics['checkpoint_sha256']:raise RuntimeError('checkpoint hash changed')
    ignored=subprocess.run(['git','check-ignore','--quiet',str(checkpoint)],check=False).returncode==0
    if not ignored:raise RuntimeError(f'checkpoint must remain local and ignored: {checkpoint}')
    model=load_anchored(checkpoint)
    # The checkpoint already includes the learned residual; zero adds nothing.
    replay=TaskRollout(model,nominal,actor).run(np.zeros(6,np.float32),START[None],GOAL[None],
        jax.random.PRNGKey(evaluation['evaluation_keys'][0]),evaluation['replicates_per_key'])
    with np.load(out/f'{name}_trajectories.npz',allow_pickle=False) as saved:
      for key in saved.files:
        if not np.array_equal(replay[key],saved[key][:,:evaluation['replicates_per_key']]):
          raise RuntimeError(f'checkpoint replay mismatch: {name}/{key}')
    checks[name]={'loaded_checkpoint_replay_exact':True,'replayed_paths':evaluation['replicates_per_key'],
        'checkpoint_git_ignored':ignored}
  largest_error=0.;saved_files=list(out.glob('*_trajectories.npz'))
  for path in saved_files:
    with np.load(path,allow_pickle=False) as data:
      for key in data.files:
        if not np.isfinite(data[key]).all():raise RuntimeError(f'non-finite trajectory: {path}/{key}')
      rewards=data['reward'];active=data['active'];before=data['states'][...,:-1,:];after=data['states'][...,1:,:]
      if not np.array_equal(after[...,2:][active],before[...,:6][active]):raise RuntimeError('invalid F4 shift')
      if not np.all(data['commanded_goal']==GOAL):raise RuntimeError('goal changed')
      if not np.all(np.abs(data['action'])<=1):raise RuntimeError('action outside bounds')
      expected=(rewards*np.power(DISCOUNT,np.arange(rewards.shape[-1]))).sum(-1)
      error=float(np.abs(expected-data['return']).max());largest_error=max(largest_error,error)
      if error>1e-5:raise RuntimeError('discounted return mismatch')
      if np.any(rewards[~active]!=0):raise RuntimeError('inactive transition rewarded')
  result={'status':'passed','command':f'python -m scripts.check_rollout_return_artifacts --run-dir {out.as_posix()}',
      'source_sha256':file_sha(__file__),'all_previous_inputs_unchanged':True,'experiment_source_hashes_match':True,
      'source_ast_and_english_checks_passed':True,'finite_json_files_checked':len(json_files),
      'saved_trajectory_files_checked':len(saved_files),'maximum_return_recompute_roundoff':largest_error,
      'saved_checkpoint_checks':checks,'scope':'artifact verification and exact replay, no additional optimization or independent evaluation'}
  write_json(out/'verification.json',result)
  print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
