"""Verify saved zero-bias checkpoints, objective records and rollout contracts."""
import argparse
from pathlib import Path
import subprocess

import jax
import numpy as np

from ett.anchored_transition import load_anchored
from ett.bias_ablation import check_subspace,load_arrays,require_saved_equal
from ett.rollout_return import TaskRollout,START,GOAL,task_reward
from ett.run_rollout_return import read,setup
from ett.run_distribution_matching import write_json
from scripts.make_swamp_f4_failure_bank import file_sha


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);args=p.parse_args();out=Path(args.run_dir)
  config=read(out/'config.json');history=read(out/'optimization_history.json');evaluation=read(out/'evaluation.json')
  if config['status']!='complete':raise RuntimeError('run incomplete')
  for path,expected in {**config['input_file_sha256'],**config['source_sha256']}.items():
    if file_sha(path)!=expected:raise RuntimeError(f'preserved input/source changed: {path}')
  original,nominal,actor,dataset,_=setup(out);checks={}
  validation=dataset.arrays('validation').take(np.arange(64));key=jax.random.PRNGKey(993)
  diagonal_reference=original.sample(validation.state,validation.action,validation.action,key,4,validation.goal)
  for name,steps in history.items():
    current=load_anchored(out/f'{name}.pkl');weights=np.array(steps[-1]['weights'],np.float32)
    check_subspace(original.params,current.params,weights)
    changed=np.argwhere(np.asarray(current.params['residual']['linear']['w'])!=np.asarray(original.params['residual']['linear']['w']))
    if any(row not in (0,32) for row,col in changed):raise RuntimeError('unselected weight changed')
    actual=current.sample(validation.state,validation.action,validation.action,key,4,validation.goal)
    if not np.array_equal(actual,diagonal_reference):raise RuntimeError('diagonal sample identity failed')
    replay=TaskRollout(current,nominal,actor).run(np.zeros(6,np.float32),START[None],GOAL[None],jax.random.PRNGKey(710001),128)
    saved=load_arrays(out/f'{name}_trajectories.npz')
    for k in saved:
      if not np.array_equal(replay[k],saved[k][:,:128]):raise RuntimeError('loaded checkpoint reproduction failed')
    for step in steps[1:]:
      if step['diagonal_loss']!=config['diagonal_train_nll_constant']:raise RuntimeError('diagonal loss moved')
      if abs(step['two_term_loss']-(step['diagonal_loss']+step['off_loss']))>2e-6:raise RuntimeError('objective sign/terms mismatch')
      if step['bias']!=[0.,0.]:raise RuntimeError('bias moved in search')
    if subprocess.run(['git','check-ignore','--quiet',str(out/f'{name}.pkl')]).returncode!=0:raise RuntimeError('checkpoint is not ignored')
    checks[name]={'exact_zero_bias':True,'changed_weight_indices':changed,'exact_diagonal_sample_identity':True,
        'loaded_checkpoint_128_path_replay_exact':True,'git_ignored':True,'objective_records_valid':True}
  precision={}
  for path in out.glob('*_trajectories.npz'):
    d=load_arrays(path);active=d['active'];before=d['states'][...,:-1,:];after=d['states'][...,1:,:]
    if not all(np.isfinite(v).all() for v in d.values()):raise RuntimeError('nonfinite rollout')
    if not np.array_equal(after[...,2:][active],before[...,:6][active]):raise RuntimeError('F4 shift error')
    if not np.all(d['commanded_goal']==GOAL) or not np.all(np.abs(d['action'])<=1):raise RuntimeError('goal/action contract changed')
    if not np.all(d['reward'][~active]==0):raise RuntimeError('inactive steps rewarded')
    if 'continuation' in path.name:
      remaining=50-np.array(evaluation['continuation_contexts']['timestep'])
      if not np.array_equal(active.sum(-1),np.broadcast_to(remaining[:,None],active.shape[:-1])):raise RuntimeError('original horizon exceeded')
    reward=np.asarray(task_reward(after,d['commanded_goal'][...,None,:]))*active
    if not np.array_equal(reward,d['reward']):raise RuntimeError('hard reward convention changed')
    high=(np.linalg.norm(after[...,:2].astype(np.float64)-GOAL[:2],axis=-1)<2)&active
    precision[path.name]={'float32_vs_float64_indicator_mismatches':int(np.count_nonzero(high!=reward))}
  write_json(out/'verification.json',{'status':'passed','checkpoint_checks':checks,'reward_precision':precision,
      'all_previous_inputs_unchanged':True,'remaining_horizon_and_fixed_goal_verified':True,
      'source_sha256':file_sha(__file__),'command':f'python -m scripts.check_bias_ablation --run-dir {out.as_posix()}'})
  print('saved checkpoints, objective sign, frozen parameters and rollout contracts verified',flush=True)


if __name__=='__main__':main()
