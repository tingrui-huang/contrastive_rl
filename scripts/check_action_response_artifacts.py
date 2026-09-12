"""Check saved diagnostic contracts, native replay and preserved input hashes."""
import argparse
from pathlib import Path

import numpy as np

from ett.action_reference import restore
from ett.eval_diagonal_transition import _open_endpoint
from ett.rollout_return import GOAL
from ett.run_rollout_return import read
from ett.run_distribution_matching import write_json
from scripts.make_swamp_f4_failure_bank import file_sha


def load(path):
  with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}


def check_states(states):
  if not np.isfinite(states).all():raise RuntimeError('non-finite state')
  if not np.array_equal(states[...,1:,2:],states[...,:-1,:6]):raise RuntimeError('incorrect history shift')
  if not _open_endpoint(states[...,:2]).all():raise RuntimeError('invalid endpoint')
  if np.any(np.abs(np.diff(states[...,:2],axis=-2))>1+1e-6):raise RuntimeError('coordinate cap violation')


def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True);args=parser.parse_args();out=Path(args.run_dir)
  config=read(out/'config.json')
  for path,expected in {**config['input_sha256'],**config['source_sha256']}.items():
    if file_sha(path)!=expected:raise RuntimeError(f'preserved dependency changed: {path}')
  for path in out.glob('*.npz'):
    if not all(np.isfinite(v).all() for v in load(path).values()):raise RuntimeError(f'non-finite artifact: {path}')
  for kind in ('model','teacher'):
    control=load(out/f'control_{kind}_probes.npz')
    if not np.all(control['next_state']==control['next_state'][:,:1]):raise RuntimeError('control depends on execution action')
    for name in ('control','return_seed0','return_seed1'):
      r=load(out/f'{name}_{kind}_probes.npz')
      if not np.array_equal(r['anchor_xy'],control['anchor_xy']):raise RuntimeError('diagonal anchor changed across models')
      if not np.all(r['goal']==GOAL):raise RuntimeError('goal changed')
      if not np.all(np.abs(r['execution_action'])<=1) or not np.all(np.abs(r['observational_action'])<=1):raise RuntimeError('invalid action')
      if not np.array_equal(r['execution_action'][:,-1],r['observational_action']):raise RuntimeError('identity action mismatch')
      if not np.array_equal(r['next_state'][:,-1,:,:2],r['anchor_xy'][:,-1]):raise RuntimeError('identity branch mismatch')
      if not np.array_equal(r['next_state'][...,2:],np.broadcast_to(r['state'][:,None,None,:6],r['next_state'][...,2:].shape)):
        raise RuntimeError('one-step shift error')
      if np.any(r['emitted_change']>r['radius']+1e-7):raise RuntimeError('anchor bound violation')
  audit=read(out/'simulator_restoration_audit.json')
  for row in audit['contexts']:
    env=restore(row['restoration']);obs,reward,done,_=env.step(row['observational_action'])
    if not np.array_equal(obs[:8],row['observed_next_state']) or reward!=row['observed_reward'] or done:
      raise RuntimeError('JSON-restored simulator replay mismatch')
    if env._rng.bit_generator.state!=row['environment_rng_after']:raise RuntimeError('RNG replay mismatch')
  selection=read(out/'continuation_contexts.json')
  if np.any(np.array(selection['timestep'])+5>50):raise RuntimeError('original horizon exceeded')
  for name in ('control','return_seed0','return_seed1','simulator'):
    r=load(out/f'{name}_continuation.npz');check_states(r['states'])
  verification={'status':'passed','source_sha256':file_sha(__file__),
      'command':f'python -m scripts.check_action_response_artifacts --run-dir {out.as_posix()}',
      'all_old_artifacts_and_run_sources_unchanged':True,'all_npz_values_finite':True,
      'fixed_randomness_control_action_invariance_exact':True,'common_diagonal_anchor_across_models_exact':True,
      'original_horizon_respected':True,'f4_shift_endpoints_action_and_displacement_checks_passed':True,
      'native_replay_from_saved_json_exact_context_count':len(audit['contexts']),
      'model_step_input_scope':'visible F4, fixed goal, execution action and observational action only',
      'continuation_contexts':len(selection['timestep']),'no_training_or_model_resampling':True}
  write_json(out/'verification.json',verification);print(verification,flush=True)


if __name__=='__main__':main()
