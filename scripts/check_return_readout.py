"""Replay saved readouts and audit visible reward boundaries and episode pairing."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from crl.checkpoint import load_checkpoint
from ett.rollout_return import task_reward
from ett.run_return_readout import read,write,load_data,extract,RUNS,CONFIG
from ett.return_readout import ridge_predict,regression_metrics,make_pairs,pair_metrics
from scripts.make_swamp_f4_failure_bank import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();root=Path(args.run_dir);torch.set_num_threads(1)
    provenance=read(root/'provenance.json');completion=read(root/'completion.json')
    for p,h in {**provenance['input_sha256'],**provenance['source_sha256']}.items(): assert file_sha(p)==h,p
    for p,h in completion['readout_files'].items(): assert file_sha(root/'readouts'/p)==h,p
    obs,act,ep,t,y,parts,masks,source,bank=load_data();test=masks['test']
    assert not np.isin(parts['test'],bank).any() and not np.isin(parts['selection'],bank).any()
    # Independent NumPy native-coordinate reward reconstruction and roundoff margin.
    distance=np.linalg.norm(obs[:,:,:2].astype(float)-obs[:,:,8:10].astype(float),axis=-1)
    reward=(distance<2).astype(float)
    np.testing.assert_array_equal(reward,np.asarray(task_reward(obs[:,:,:8],obs[:,:,8:])))
    margin=float(np.min(np.abs(distance-2)))
    assert margin>1e-6, 'inspect reward-boundary rounding ambiguity before claiming exact reconstruction'
    manual=np.array([reward[e,tt+1:tt+21]@(.95**np.arange(20)) for e,tt in zip(ep,t)])
    np.testing.assert_array_equal(manual,y)
    with np.load(root/'evaluation.npz') as saved: arrays={k:saved[k] for k in saved.files}
    np.testing.assert_array_equal(arrays['episode_id'],ep[test]);np.testing.assert_array_equal(arrays['target'],y[test])
    observation,action=obs[ep[test],t[test]],act[ep[test],t[test]]
    raw_x=np.concatenate([observation[:,:8],action,observation[:,8:]],1).astype(float)
    deviations={}
    for name in CONFIG['checkpoints']:
        _,state=load_checkpoint(RUNS/name/'final.pkl')
        phi,raw,_=extract(state.q_params,observation,action)
        with np.load(root/'readouts'/f'{name}_ridge.npz') as archive: model={k:archive[k] for k in archive.files}
        pred=ridge_predict(model,phi.astype(float))
        np.testing.assert_allclose(pred,arrays[name+'_ridge'],rtol=1e-5,atol=2e-4)
        np.testing.assert_allclose(raw,arrays[name+'_raw'],rtol=1e-5,atol=2e-4)
        deviations[name]=float(np.max(np.abs(pred-arrays[name+'_ridge'])))
    with np.load(root/'readouts'/'raw_input_ridge.npz') as archive: model={k:archive[k] for k in archive.files}
    np.testing.assert_allclose(ridge_predict(model,raw_x),arrays['raw_input_ridge'],atol=1e-12)
    for seed in [110,111]:
        # Locally created, hash-verified checkpoint; no external pickle source.
        saved=torch.load(root/'readouts'/f'raw_mlp_s{seed}.pt',weights_only=False)
        model=torch.nn.Sequential(torch.nn.Linear(18,32),torch.nn.ReLU(),torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
        model.load_state_dict(saved['state_dict'])
        with torch.no_grad(): pred=model(torch.tensor((raw_x-saved['mean'])/saved['std'],dtype=torch.float32))[:,0].numpy()*saved['target_std']+saved['target_mean']
        np.testing.assert_allclose(pred,arrays[f'raw_mlp_s{seed}'],rtol=1e-5,atol=2e-5)
    with np.load(root/'pair_indices.npz') as saved: pairs={k:saved[k] for k in saved.files}
    metrics=read(root/'metrics.json')
    for key,matched in [('overall',False),('matched',True)]:
        again=make_pairs(ep[test],t[test],observation[:,:8],observation[:,8:],source[test],matched=matched)
        np.testing.assert_array_equal(again,pairs[key])
        a,b=again.T
        assert len(np.unique(ep[test][again]))==again.size
        if matched:
            assert np.all(source[test][a]==source[test][b]) and np.all(np.abs(t[test][a]-t[test][b])<=3)
            assert np.all(np.linalg.norm(observation[a,:2]-observation[b,:2],axis=1)<=.25)
        for name in metrics:
            assert pair_metrics(y[test],arrays[name],pairs[key])==metrics[name]['pairwise'][key]
    write(root/'verification.json',dict(status='passed',checkpoint_replay=True,
        maximum_ridge_replay_differences=deviations,stored_metrics_and_pairings_reproduced=True,
        reward_float32_float64_agreement=True,min_distance_to_reward_boundary=margin,
        complete_return_indexing=True,episode_and_bank_exclusion=True,upstream_files_unchanged=True,
        hidden_audit_fields_loaded=False,unit_tests_passed=6,source_sha256=file_sha(__file__)))
    print('Readout replay, visible reward reconstruction, return indexing and pair checks passed.')


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
