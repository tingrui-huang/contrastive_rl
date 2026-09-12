"""Bounded independent fixed-actor evaluation of existing frozen readouts."""
import argparse
import copy
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from crl.checkpoint import load_checkpoint
from ett.run_distribution_matching import frozen_actor
from ett.run_return_readout import read, write, extract, load_data, RUNS, DATA, BANK, NAMES
from ett.return_readout import ridge_predict, make_pairs
from ett.fixed_actor_continuation import TimedSimulator, observable_strata, select_contexts, coverage_action
from scripts.make_swamp_f4_failure_bank import file_sha

PRIOR = Path('artifacts/return_readout/f4_h20_g095_v1')
SOURCES = ['fixed_actor', 'blind_shortcut', 'visible_detour', 'uniform']
PROTOCOL = dict(
    base_commit='5d454cbb3782a86eee40a6cf2664fd9276fe992e',
    environment='point_two_route_swamp_windy_f4_v0', active_prob=.3,
    goal=[8.5,3.5]*4, horizon=20, discount=.95, episode_horizon=50,
    actor='alpha0_seed0', actor_execution='stochastic tanh Gaussian, unchanged historical frozen_actor helper',
    actor_reason='historical default actor in run_distribution_matching and run_rollout_return; chosen before new outcomes',
    first_action='one draw per context held fixed across all repeats; every scorer receives that exact action',
    parents=384, collection_steps=30, candidate_times=[4,8,12,16,20,24,28,30],
    sources=SOURCES, source_assignment='parent modulo 4; 96 parents each',
    coverage='visible waypoint shortcut/detour or uniform; speed .55/.9 by (parent//4)%2, Gaussian .12, pauses t=5,6,13; fixed actor source stochastic',
    selection='2 candidates per parent without replacement, inverse global (region,progress,motion) count, 4x weight outside strict radius-2 goal',
    regions=['before x<3','swamp 3<=x<6 and 3<=y<4','after x>=6','detour 3<=x<6 outside corridor'],
    progress=['distance>=6','2<=distance<6','distance<2'], motion=['F4 RMS step <.02','.02<=RMS<.35','RMS>=.35'],
    max_contexts=768, repeats=24, continuation_batch_contexts=64,
    maximum_collection_steps=11520, maximum_continuation_steps=368640,
    maximum_check_steps=2048, maximum_total_environment_steps=382208,
    seeds=dict(environment_base=12000000, coverage_base=12001000, selection=12002001,
               collector_actor_base=12003000, first_action=12004000,
               continuation_environment_base=13000000, continuation_actor_base=14000000,
               matching=12005000, bootstrap=12006000, support=12007000, checks=15000000),
    restoration='deep copy full native environment plus external elapsed counter; fresh future RNG retains initial bits/absorption/history/time',
    repeats_interpretation='future randomness conditional on one full simulator snapshot, not hidden posterior draws',
    pairing=dict(algorithm='existing make_pairs, one random anchor per parent; greedily match unused distinct parent; each parent at most once',
                 xy=.25, goal_distance=.25, time=3, exact=['source','goal','XY cell'],
                 outcome_blind=True, return_tie_tolerance=1e-10,
                 reliable_order='abs(mean difference) > max(0.5, t_0.975_Welch * independent-repeat SE); diagnostic, not simultaneous testing',
                 prediction_ties='half credit', tolerances_never_adapted=True),
    statistics=dict(bootstrap_replicates=500, aggregate='parent-cluster bootstrap, both contexts together; within each context resample its 24 repeats',
                    pairs='resample disjoint parent-pair blocks; independently resample repeats, report fixed reliable subset and order-flip sensitivity',
                    comparisons='paired same resamples for alpha .3 minus 0, phi versus own scalar, phi versus raw baselines',
                    spearman='descriptive and parent-cluster nested interval', confidence=.95),
    support=dict(inputs='8 observed F4 + recorded first action 2 + constant goal 8',
                 scaling='saved raw ridge training mean/std; never refit on new contexts',
                 reference_rows=16384, calibration_rows=2048, threshold='95th percentile calibration nearest-reference distance',
                 diagnostics=['coordinate training range violations','action saturation','F4 RMS motion','standardized nearest-training-reference distance']),
    stopping='one collection, one evaluation; no outcome-dependent budget changes, no fitting or ETT',
)


def verify_prior():
    hashes = {}
    provenance = read(PRIOR/'provenance.json')
    for path, digest in {**provenance['input_sha256'], **provenance['source_sha256']}.items():
        assert file_sha(path)==digest, path
        hashes[path] = digest
    for manifest_name in ['completion.json','report_manifest.json','local_checks.json']:
        manifest = read(PRIOR/manifest_name)
        entries = manifest.get('readout_files', manifest.get('files', {}))
        prefix = PRIOR/'readouts' if manifest_name=='completion.json' else PRIOR
        for name, digest in entries.items():
            path = prefix/name
            if not path.exists():
                raise FileNotFoundError(f'Missing frozen artifact {path}; reproduce original protocol in a fresh directory and verify original predictions first')
            assert file_sha(path)==digest, path
            hashes[path.as_posix()] = digest
        if manifest_name=='local_checks.json':
            for path,digest in manifest['source_sha256'].items():
                assert file_sha(path)==digest, path
                hashes[path] = digest
    for path in PRIOR.glob('*.json'):
        hashes[path.as_posix()] = file_sha(path)
    assert read(PRIOR/'verification.json')['status']=='passed'
    return hashes


class FrozenModels:
    def __init__(self):
        self.hashes = verify_prior()
        self.actor, self.actor_info = frozen_actor(str(RUNS/'alpha0_seed0/final.pkl'),
            read(RUNS/'alpha0_seed0/arm_provenance.json')['dataset_content_sha256'])

    def actions(self, obs, seed):
        return np.asarray(self.actor(jnp.asarray(obs[:,:8]), jnp.asarray(obs[:,8:]), jax.random.PRNGKey(seed)))

    def predictions(self, obs, actions):
        raw = np.concatenate([obs[:,:8],actions,obs[:,8:]],axis=1).astype(float)
        outputs, features = {}, {}
        for name in NAMES:
            _, state = load_checkpoint(RUNS/name/'final.pkl')
            phi, score, detail = extract(state.q_params, obs, actions)
            with np.load(PRIOR/'readouts'/f'{name}_ridge.npz') as z:
                outputs[name+'_ridge'] = ridge_predict(dict(z), phi.astype(float))
            outputs[name+'_raw'] = score
            features[name] = detail
        with np.load(PRIOR/'readouts/raw_input_ridge.npz') as z:
            outputs['raw_input_ridge'] = ridge_predict(dict(z), raw)
        for seed in [110,111]:
            saved = torch.load(PRIOR/'readouts'/f'raw_mlp_s{seed}.pt', weights_only=False)
            model = torch.nn.Sequential(torch.nn.Linear(18,32),torch.nn.ReLU(),torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
            model.load_state_dict(saved['state_dict']); model.eval()
            for p in model.parameters(): p.requires_grad_(False)
            with torch.no_grad():
                pred = model(torch.tensor((raw-saved['mean'])/saved['std'], dtype=torch.float32))[:,0].numpy()
            outputs[f'raw_mlp_s{seed}'] = pred*saved['target_std']+saved['target_mean']
        with np.load(PRIOR/'readouts/raw_input_ridge.npz') as z:
            outputs['constant_training_mean'] = np.full(len(obs),float(z['intercept']))
        assert all(np.isfinite(x).all() for x in outputs.values())
        return outputs, features


def prepare(root):
    if root.exists(): raise ValueError('Use a fresh output directory')
    hashes = verify_prior()
    for path in ['ett/fixed_actor_continuation.py','ett/run_fixed_actor_continuation.py',
                 'ett/analyze_fixed_actor_continuation.py','scripts/test_fixed_actor_continuation.py',
                 'ett/run_distribution_matching.py','ett/action_reference.py']:
        hashes[path] = file_sha(path)
    root.mkdir(parents=True)
    write(root/'protocol.json',PROTOCOL)
    write(root/'provenance.json',dict(input_sha256=hashes,
        actor_checkpoint_sha256=file_sha(RUNS/'alpha0_seed0/final.pkl'),
        prior_report=PRIOR.as_posix(),bank='original 256 references; alpha weights negative loss, not bank sampling fraction',
        pretrained_models_unchanged=True))
    write(root/'prepared.json',dict(protocol_sha256=file_sha(root/'protocol.json'),prepared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('Saved pre-outcome protocol; hard environment-step cap:',PROTOCOL['maximum_total_environment_steps'],flush=True)


def verify_run(root):
    assert read(root/'protocol.json')==PROTOCOL
    assert file_sha(root/'protocol.json')==read(root/'prepared.json')['protocol_sha256']
    for path,digest in read(root/'provenance.json')['input_sha256'].items():
        assert file_sha(path)==digest, path


def support_diagnostics(obs, action):
    old_obs, old_act, ep, t, _, _, masks, _, _ = load_data()
    ids = np.flatnonzero(masks['train'])
    train = np.concatenate([old_obs[ep[ids],t[ids],:8],old_act[ep[ids],t[ids]],old_obs[ep[ids],t[ids],8:]],axis=1).astype(float)
    query = np.concatenate([obs[:,:8],action,obs[:,8:]],axis=1).astype(float)
    with np.load(PRIOR/'readouts/raw_input_ridge.npz') as z: mean,std=z['mean'],z['std']
    perm = np.random.default_rng(PROTOCOL['seeds']['support']).permutation(len(train))
    reference, calibration = perm[:16384], perm[16384:18432]
    tree = cKDTree((train[reference]-mean)/std)
    distance = tree.query((query-mean)/std)[0]
    calibration_distance = tree.query((train[calibration]-mean)/std)[0]
    threshold = float(np.quantile(calibration_distance,.95))
    motion = lambda x: np.sqrt(np.mean(np.sum(np.diff(x[:,:8].reshape(-1,4,2),axis=1)**2,axis=2),axis=1))
    report = dict(reference_rows=len(reference), calibration_rows=len(calibration),
        threshold=threshold, new_nn_quantiles=np.quantile(distance,[0,.25,.5,.75,.95,1]),
        old_nn_quantiles=np.quantile(calibration_distance,[0,.25,.5,.75,.95,1]),
        outside_reference_threshold_fraction=float(np.mean(distance>threshold)),
        coordinate_outside_training_range_fraction=np.mean((query<train.min(0))|(query>train.max(0)),axis=0),
        new_action_saturation_fraction=float(np.mean(np.any(np.abs(action)>.99,axis=1))),
        old_action_saturation_fraction=float(np.mean(np.any(np.abs(train[:,8:10])>.99,axis=1))),
        new_motion_quantiles=np.quantile(motion(query),[0,.25,.5,.75,.95,1]),
        old_motion_quantiles=np.quantile(motion(train),[0,.25,.5,.75,.95,1]),
        scope='observable raw-input support only; proximity does not establish hidden-state or continuation-policy support')
    return report, distance, threshold


def collect(root):
    verify_run(root)
    if (root/'contexts.npz').exists(): raise ValueError('contexts already collected')
    models = FrozenModels(); seeds = PROTOCOL['seeds']; n=PROTOCOL['parents']
    simulators = [TimedSimulator(seeds['environment_base']+i) for i in range(n)]
    randoms = [np.random.default_rng(seeds['coverage_base']+i) for i in range(n)]
    memos = [dict(speed=[.55,.9][(i//4)%2]) for i in range(n)]
    source=np.arange(n)%4; actor_ids=np.flatnonzero(source==0)
    histories=[np.array([e.observation() for e in simulators])]; actions=[]; snapshots=[]; records=[]
    for t in range(31):
        obs=histories[-1]
        if t in PROTOCOL['candidate_times']:
            for i,env in enumerate(simulators):
                snapshots.append(env.snapshot());records.append((i,t))
        if t==30: break
        action=np.zeros((n,2),np.float32)
        action[actor_ids]=models.actions(obs[actor_ids],seeds['collector_actor_base']+t)
        for i in np.flatnonzero(source!=0):
            action[i]=coverage_action(source[i],obs[i],t,randoms[i],memos[i])
        histories.append(np.array([env.step(a)[0] for env,a in zip(simulators,action)]));actions.append(action)
    history=np.stack(histories,axis=1); records=np.asarray(records)
    candidates=history[records[:,0],records[:,1]]
    selected=select_contexts(candidates,records[:,0],seeds['selection'])
    parent,t=records[selected].T;obs=candidates[selected];src=source[parent]
    first=models.actions(obs,seeds['first_action'])
    pairs={name:make_pairs(parent,t,obs[:,:8],obs[:,8:],src,seed=seeds['matching'],matched=matched)
           for name,matched in [('overall',False),('matched',True)]}
    np.savez_compressed(root/'pairs.npz',**pairs)
    predictions,features=models.predictions(obs,first)
    np.savez_compressed(root/'predictions_before_outcomes.npz',**predictions)
    write(root/'feature_verification.json',features)
    support,nn,threshold=support_diagnostics(obs,first)
    write(root/'support_before_outcomes.json',support)
    np.savez_compressed(root/'contexts.npz',observation=obs,first_action=first,parent=parent,time=t,source=src,
                        support_distance=nn,support_outside=nn>threshold,**observable_strata(obs))
    np.savez_compressed(root/'collection_visible.npz',observations=history,actions=np.stack(actions,axis=1),source=source)
    # Snapshots are local and ignored; no hidden state is published with models.
    with (root/'simulator_snapshots.pkl').open('wb') as stream: pickle.dump([snapshots[i] for i in selected],stream)
    write(root/'collection.json',dict(contexts=len(obs),parents=n,environment_steps=n*30,
        pairs={k:len(v) for k,v in pairs.items()},source_counts={s:int(np.sum(src==i)) for i,s in enumerate(SOURCES)},
        strata={k:{str(i):int(np.sum(v==i)) for i in np.unique(v)} for k,v in observable_strata(obs).items()},
        hidden_fields_used_in_selection_or_scoring=False,selected_contexts_per_parent=2,
        file_sha256={p.name:file_sha(p) for p in root.glob('*') if p.is_file()},
        complete=True))
    print('Context selection and pairing fixed before continuation:',len(obs),{k:len(v) for k,v in pairs.items()},flush=True)


def run(root):
    verify_run(root)
    if (root/'continuations.npz').exists(): raise ValueError('continuations already exist')
    for name,digest in read(root/'collection.json')['file_sha256'].items(): assert file_sha(root/name)==digest,name
    models=FrozenModels();c=dict(np.load(root/'contexts.npz'));n=len(c['parent']);r=PROTOCOL['repeats'];seeds=PROTOCOL['seeds']
    assert n<=PROTOCOL['max_contexts']
    with (root/'simulator_snapshots.pkl').open('rb') as stream: snapshots=pickle.load(stream)
    xy=np.empty((n,r,21,2),np.float32);actions=np.empty((n,r,20,2),np.float32);rewards=np.empty((n,r,20),np.float64)
    start=time.monotonic();steps=0
    for lo in range(0,n,64):
        hi=min(lo+64,n)
        envs=[TimedSimulator.restore(snapshots[i],seeds['continuation_environment_base']+i*r+j)
              for i in range(lo,hi) for j in range(r)]
        for e in envs: e.require_horizon(20)
        obs=np.array([e.observation() for e in envs]);expected=np.repeat(c['observation'][lo:hi],r,axis=0)
        np.testing.assert_array_equal(obs,expected)
        xy[lo:hi,:,0]=obs[:,:2].reshape(hi-lo,r,2)
        for t in range(20):
            action=(np.repeat(c['first_action'][lo:hi],r,axis=0) if t==0 else
                    models.actions(obs,seeds['continuation_actor_base']+(lo//64)*20+t))
            outputs=[e.step(a) for e,a in zip(envs,action)]
            obs=np.array([o[0] for o in outputs]);reward=np.array([o[1] for o in outputs])
            assert all(o[2]==(c['time'][lo+i//r]+t+1==50) for i,o in enumerate(outputs))
            xy[lo:hi,:,t+1]=obs[:,:2].reshape(hi-lo,r,2)
            actions[lo:hi,:,t]=action.reshape(hi-lo,r,2);rewards[lo:hi,:,t]=reward.reshape(hi-lo,r)
            steps+=len(envs)
        print(f'Continuation contexts {hi}/{n}; steps {steps}; seconds {time.monotonic()-start:.1f}',flush=True)
    np.testing.assert_array_equal(actions[:,:,0],np.repeat(c['first_action'][:,None],r,axis=1))
    assert steps<=PROTOCOL['maximum_continuation_steps']
    returns=rewards@(.95**np.arange(20))
    from ett.rollout_return import task_reward
    audited=np.asarray(task_reward(xy[:,:,1:],np.broadcast_to(np.array(PROTOCOL['goal'],np.float32),(n,r,20,8))))
    np.testing.assert_array_equal(rewards,audited)
    np.savez_compressed(root/'continuations.npz',xy=xy,actions=actions,rewards=rewards,returns=returns)
    verify_run(root)
    write(root/'completion.json',dict(complete=True,environment_steps=steps+PROTOCOL['maximum_collection_steps'],
        continuation_steps=steps,elapsed_continuation_seconds=time.monotonic()-start,
        first_action_alignment_exact=True,goal_and_f4_checked_every_step=True,
        actual_and_audited_rewards_equal=True,full_20_step_continuations=True,
        inputs_unchanged=True,protocol_sha256=file_sha(root/'protocol.json'),
        continuation_sha256=file_sha(root/'continuations.npz')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--phase',choices=['prepare','collect','run'],required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        {'prepare':prepare,'collect':collect,'run':run}[args.phase](Path(args.out_dir))


if __name__=='__main__': main()
