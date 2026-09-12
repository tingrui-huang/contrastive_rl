"""Replay published execution actions and frozen scores after continuation."""
import argparse
from pathlib import Path
import pickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ett.run_fixed_actor_continuation import FrozenModels, PROTOCOL, verify_run
from ett.fixed_actor_continuation import TimedSimulator
from ett.run_return_readout import read, write
from ett.return_readout import make_pairs
from scripts.make_swamp_f4_failure_bank import file_sha


def main(root):
    verify_run(root)
    for name,digest in read(root/'collection.json')['file_sha256'].items(): assert file_sha(root/name)==digest,name
    c=dict(np.load(root/'contexts.npz'));data=dict(np.load(root/'continuations.npz'))
    assert file_sha(root/'continuations.npz')==read(root/'completion.json')['continuation_sha256']
    with (root/'simulator_snapshots.pkl').open('rb') as stream: snapshots=pickle.load(stream)
    n,r,h=data['rewards'].shape;assert (n,r,h)==(768,24,20)
    assert np.isfinite(data['actions']).all() and np.max(np.abs(data['actions']))<=1
    np.testing.assert_array_equal(data['returns'],data['rewards']@(.95**np.arange(20)))
    actual=(np.linalg.norm(data['xy'][:,:,1:].astype(float)-[8.5,3.5],axis=-1)<2).astype(float)
    np.testing.assert_array_equal(actual,data['rewards'])
    models=FrozenModels()
    np.testing.assert_array_equal(models.actions(c['observation'],PROTOCOL['seeds']['first_action']),c['first_action'])
    outputs,_=models.predictions(c['observation'],c['first_action'])
    with np.load(root/'predictions_before_outcomes.npz') as stored:
        for key,value in outputs.items():np.testing.assert_array_equal(value,stored[key])
    # Independently reconstruct actor observations from initial F4 and saved XY.
    for lo in range(0,n,64):
        hi=min(lo+64,n);obs=np.repeat(c['observation'][lo:hi],r,axis=0)
        for t in range(20):
            expected=(np.repeat(c['first_action'][lo:hi],r,axis=0) if t==0 else
                      models.actions(obs,PROTOCOL['seeds']['continuation_actor_base']+(lo//64)*20+t))
            np.testing.assert_array_equal(expected,data['actions'][lo:hi,:,t].reshape(-1,2))
            obs=np.concatenate([data['xy'][lo:hi,:,t+1].reshape(-1,2),obs[:,:6],obs[:,8:]],axis=1)
    steps=0
    collection=dict(np.load(root/'collection_visible.npz'))
    for i in np.linspace(0,n-1,8,dtype=int):
        clone=TimedSimulator.restore(snapshots[i]);parent,t=c['parent'][i],c['time'][i]
        np.testing.assert_array_equal(clone.observation(),collection['observations'][parent,t])
        assert clone.elapsed==t
        if t<30:
            observation,_,_=clone.step(collection['actions'][parent,t]);steps+=1
            np.testing.assert_array_equal(observation,collection['observations'][parent,t+1])
        for repeat in [0,1]:
            clone=TimedSimulator.restore(snapshots[i],PROTOCOL['seeds']['continuation_environment_base']+i*r+repeat)
            for k in range(20):
                observation,reward,truncated=clone.step(data['actions'][i,repeat,k]);steps+=1
                np.testing.assert_array_equal(observation[:2],data['xy'][i,repeat,k+1])
                assert reward==data['rewards'][i,repeat,k]
                assert truncated==(t+k+1==50)
    with np.load(root/'pairs.npz') as pairs:
        for key in ['overall','matched']:
            again=make_pairs(c['parent'],c['time'],c['observation'][:,:8],c['observation'][:,8:],c['source'],
                             seed=PROTOCOL['seeds']['matching'],matched=key=='matched')
            np.testing.assert_array_equal(again,pairs[key])
    verify_run(root)
    assert steps+283<PROTOCOL['maximum_check_steps']
    write(root/'verification.json',dict(status='passed',all_frozen_scores_replayed_exactly=True,
        all_future_actor_actions_replayed_exactly=True,selected_native_trajectories_replayed_exactly=True,
        native_replay_steps=steps,focused_unit_tests=6,unit_test_environment_steps=283,
        exact_return_indexing=True,full_horizon_and_time_limit=True,reconstructed_visible_reward_exact=True,
        pair_selection_reproduced=True,original_input_hashes_unchanged=True,
        total_main_and_check_environment_steps=read(root/'completion.json')['environment_steps']+steps+283,
        source_sha256=file_sha(__file__)))
    print('All frozen scores, actor actions, native replays, return targets and pair definitions verified.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): main(Path(parser.parse_args().run_dir))
