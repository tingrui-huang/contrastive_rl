"""Bounded logged-return probes of frozen PointMaze contrastive features."""
import argparse
import copy
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from crl.checkpoint import load_checkpoint
from crl.networks import make_networks
from ett.return_readout import (return_windows, episode_split, ridge_fit, ridge_predict,
    standardize_fit, regression_metrics, make_pairs, pair_metrics, pair_values, error_intervals)
from scripts.make_swamp_f4_failure_bank import file_sha


DATA = Path('artifacts/f4_p30_server_30076/results/datasets/swamp_windy_f4_merged_s0.npz')
RUNS = Path('artifacts/f4_p30_server_30076/results/runs/f4_p30_sweep/p30_a0_a01_a03_s0_s1')
BANK = Path('artifacts/f4_p30_server_30076/results/artifacts/swamp_windy_f4_failure_bank/failure_bank_f4_r60d40.npz')
OLD = Path('artifacts/critic_continuation/f4_p30_alpha_s01/config.json')
NAMES = ['alpha0_seed0','alpha0p3_seed0','alpha0_seed1','alpha0p3_seed1']
CONFIG = dict(horizon=20, discount=.95, checkpoints=NAMES, ridge_grid=[.0001,.01,1.,100.],
    ridge_objective='mean MSE + lambda*||w||^2 on train-standardized features; unpenalized intercept',
    features='all phi heads concatenated; psi verified constant, not concatenated',
    raw_input='8 observed state coordinates + 2 recorded actions + 8 constant commanded goal coordinates',
    mlp=dict(hidden=[32,32], steps=1000, batch_size=512, learning_rate=.003,
        seeds=[110,111], batch_seed=95124, eval_every=200, selection='minimum selection MSE; no test selection'),
    matching=dict(seed=95121, newest_xy=.25, goal_distance=.25, time=3,
        exact=['source membership','commanded goal','XY cell'],
        anchors='one random eligible window per episode; score/return blind',
        reuse='each episode at most once in either role per pair set',
        ties='equal returns within 1e-10 excluded and counted; equal predictions half credit'),
    split='historical source 10% test, complement 10% selection using 9201; exclude original bank episodes from selection/test',
    comparison='alpha 0 versus historically primary alpha 0.3, seeds 0/1; no alpha selection',
    targets='visible fixed-task reward only; no hindsight, masks, death labels or failure scores',
    bootstrap=dict(pair_replicates=1000,error_episode_replicates=500),
    publication='readout weights local; configs, normalization, metrics and heldout predictions publishable')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    def default(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v))
    Path(path).write_text(json.dumps(value,indent=2,default=default,allow_nan=False)+'\n',encoding='utf-8')


def load_data():
    expected=read(OLD)['dataset']
    manifest=read(str(DATA)+'.manifest.json')
    assert file_sha(DATA)==manifest['sha256']=='83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c'
    with np.load(DATA,allow_pickle=False) as d:
        # No audit-only field is read. Metadata is used only for provenance.
        obs,act,meta=d['obs'],d['act'],json.loads(str(d['meta']))
        lengths=d['lengths'] if 'lengths' in d else np.full(len(obs),obs.shape[1])
    assert meta['per_cell_swamp_prob']==.3 and meta['env_name']=='point_two_route_swamp_windy_f4_v0'
    assert np.all(act[:,-1]==0) and np.isfinite(obs).all() and np.isfinite(act).all()
    assert np.array_equal(obs[:,1:,2:8],obs[:,:-1,:6])
    with np.load(BANK,allow_pickle=False) as bank:
        bank_ep,bank_row,bank_vectors=bank['episode_id'],bank['failure_row'],bank['goals']
    assert np.array_equal(bank_vectors,obs[bank_ep,bank_row,:8])
    ep,t,y,rewards=return_windows(obs,act,lengths)
    parts=episode_split(len(obs),bank_ep)
    masks={k:np.isin(ep,ids) for k,ids in parts.items()}
    signatures={}; duplicate=0
    for part,ids in parts.items():
        for e in ids:
            sig=obs[e,:lengths[e]].tobytes()+act[e,:lengths[e]-1].tobytes()
            if sig in signatures and signatures[sig]!=part: duplicate+=1
            signatures[sig]=part
    assert duplicate==0
    source=np.where(ep<1200,0,np.where(ep<6000,1,2))
    return obs,act,ep,t,y,parts,masks,source,bank_ep


def prepare(root):
    if root.exists(): raise ValueError('use a fresh output directory')
    obs,act,ep,t,y,parts,masks,source,bank_ep=load_data()
    historical=read(OLD)
    records={}; input_hashes={p.as_posix():file_sha(p) for p in [DATA,BANK,OLD,Path('crl/losses.py'),Path('crl/envs.py'),Path('ett/rollout_return.py')]}
    old_runs={v['name']:v for v in historical['checkpoints']['all']}
    for name in NAMES:
        path=RUNS/name/'final.pkl'; provenance=read(RUNS/name/'arm_provenance.json')
        assert file_sha(path)==old_runs[name]['checkpoint_sha256']
        assert provenance['dataset_content_sha256']==read(str(DATA)+'.manifest.json')['content_sha256']
        assert provenance['steps']==150000 and provenance['batch_size']==256
        assert provenance['obs_norm_mode']=='' and provenance['per_cell_swamp_prob']==.3
        if provenance['alpha']>0:
            assert provenance['bank_content_sha256']=='022f2d0d52cf6e0147def46cd800ae2da65e4b5592090397f606f051f6e2c7e9'
        records[name]=dict(checkpoint_sha256=file_sha(path),provenance=provenance)
        input_hashes[path.as_posix()]=file_sha(path)
        input_hashes[(RUNS/name/'arm_provenance.json').as_posix()]=file_sha(RUNS/name/'arm_provenance.json')
    matched_fields=['code_commit','seed','dataset_content_sha256','steps','batch_size','obs_norm_mode','per_cell_swamp_prob']
    for seed in [0,1]:
        a,b=[records[f'{arm}_seed{seed}']['provenance'] for arm in ['alpha0','alpha0p3']]
        assert all(a[k]==b[k] for k in matched_fields)
    root.mkdir(parents=True); (root/'readouts').mkdir()
    write(root/'config.json',CONFIG)
    write(root/'provenance.json',dict(checkpoints=records,input_sha256=input_hashes,
        architecture=historical['checkpoints']['architecture'],shared_training_config=historical['checkpoints']['shared_training_config'],
        matched_fields=matched_fields,initialization='same recorded checkpoint seeds; original initial checkpoints and batch hashes unavailable',
        upstream_reuse='all source episodes used by encoders and actors; source holdout previously inspected by multiple audits; held out only from readouts',
        bank_provenance='original 256-vector random60/deliberate40 bank curated using teacher_mode/swamp_bits; no bank distance/labels used here',
        episode_ids=parts,bank_episode_overlap={k:int(np.isin(ids,bank_ep).sum()) for k,ids in parts.items()},
        source_sha256={p:file_sha(p) for p in ['crl/networks.py','ett/return_readout.py','ett/run_return_readout.py']}))
    population={}
    for k,mask in masks.items():
        values=y[mask]
        population[k]=dict(episodes=len(parts[k]),windows=int(mask.sum()),return_mean=float(values.mean()),return_std=float(values.std()),
            quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,1]),zero_fraction=float(np.mean(values==0)),
            maximal_fraction=float(np.mean(np.isclose(values,(1-.95**20)/.05))),
            by_source={str(s):dict(rows=int(np.sum(mask&(source==s))),mean=float(y[mask&(source==s)].mean())) for s in range(3)})
    write(root/'population_before_training.json',dict(partitions=population,eligible_windows=len(y),
        available_transitions=len(obs)*50,complete_future_fraction=len(y)/(len(obs)*50),
        times=[int(t.min()),int(t.max())],discarded_incomplete_windows=len(obs)*19,
        excluded_bank_selection_episodes=594-len(parts['selection']),excluded_bank_test_episodes=660-len(parts['test']),
        duplicate_trajectories_crossing_readout_splits=0,reward_reconstructed=True,reward_radius=2.,goal=[8.5,3.5],
        hidden_fields_loaded=False,target_variation_sufficient=bool(all(y[m].std()>.1 for m in masks.values()))))
    test=masks['test']; pairs={}
    for name,matched in [('overall',False),('matched',True)]:
        pairs[name]=make_pairs(ep[test],t[test],obs[ep[test],t[test],:8],obs[ep[test],t[test],8:],source[test],matched=matched)
    np.savez_compressed(root/'pair_indices.npz',**pairs)
    write(root/'pairing_before_training.json',{k:dict(pairs=len(v),covered_episodes=2*len(v),total_test_episodes=len(parts['test']),
        equal_return_pairs=int(np.sum(np.abs(y[test][v[:,0]]-y[test][v[:,1]])<=1e-10))) for k,v in pairs.items()})
    print('Prepared fixed protocol and return distribution:',json.dumps(population,default=lambda a:a.tolist()),flush=True)


def extract(q_params,observation,action):
    network=make_networks(8,8,2)
    fn=jax.jit(network.representation_network.apply)
    phi=[];raw=[];max_error=0.;psi_reference=None
    for start in range(0,len(observation),4096):
        o,a=observation[start:start+4096],action[start:start+4096]
        p,g=map(np.asarray,fn(q_params,jnp.asarray(o),jnp.asarray(a)))
        assert p.shape[1:]==(64,1)
        if psi_reference is None: psi_reference=g[0]
        np.testing.assert_allclose(g,np.broadcast_to(psi_reference,g.shape),rtol=0,atol=0)
        aligned=np.sum(p*g,axis=1)[:,0]
        # Full BxB output check on each chunk, including preprocessing and head axes.
        count=min(8,len(o)); original=np.asarray(network.q_network.apply(q_params,o[:count],a[:count]))
        reproduced=np.einsum('idh,jdh->ijh',p[:count],g[:count])[...,0]
        max_error=max(max_error,float(np.max(np.abs(original-reproduced))))
        np.testing.assert_allclose(original,reproduced,rtol=2e-5,atol=2e-5)
        phi.append(p.reshape(len(p),-1));raw.append(aligned)
    return np.concatenate(phi),np.concatenate(raw),dict(max_dot_product_error=max_error,phi_dim=64,heads=1,
        psi_constant=True,psi=psi_reference.tolist(),repr_norm=False,temperature_applied=False,obs_scale=None)


def fit_mlp(x,y,masks,seed,root):
    settings=CONFIG['mlp']; torch.manual_seed(seed)
    mean,std=standardize_fit(x[masks['train']]); z=((x-mean)/std).astype(np.float32)
    ym,ys=float(y[masks['train']].mean()),float(y[masks['train']].std())
    train_x=torch.from_numpy(z[masks['train']]);train_y=torch.tensor((y[masks['train']]-ym)/ys,dtype=torch.float32)
    val_x=torch.from_numpy(z[masks['selection']]); val_y=y[masks['selection']]
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    rng=np.random.default_rng(settings['batch_seed']);history=[];best=None;best_loss=float('inf');best_step=None
    for step in range(settings['steps']+1):
        if step%settings['eval_every']==0:
            with torch.no_grad(): v=model(val_x)[:,0].numpy()*ys+ym
            loss=float(np.mean((v-val_y)**2)); history.append(dict(step=step,selection_mse=loss))
            if loss<best_loss: best_loss=loss;best=copy.deepcopy(model.state_dict());best_step=step
        if step==settings['steps']: break
        idx=rng.integers(len(train_x),size=settings['batch_size']);optimizer.zero_grad()
        loss=(model(train_x[idx])[:,0]-train_y[idx]).square().mean();loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    model.load_state_dict(best)
    with torch.no_grad(): prediction=np.concatenate([model(torch.from_numpy(v))[:,0].numpy() for v in np.array_split(z,32)])*ys+ym
    torch.save(dict(state_dict=best,mean=mean,std=std,target_mean=ym,target_std=ys),root/'readouts'/f'raw_mlp_s{seed}.pt')
    return prediction,dict(seed=seed,history=history,selected_step=best_step,parameter_count=sum(p.numel() for p in model.parameters())),dict(mean=mean,std=std,target_mean=ym,target_std=ys)


def run(root):
    assert read(root/'config.json')==CONFIG
    if (root/'completion.json').exists(): raise ValueError('completed run is immutable')
    assert read(root/'population_before_training.json')['target_variation_sufficient']
    provenance=read(root/'provenance.json')
    for p,h in {**provenance['input_sha256'],**provenance['source_sha256']}.items(): assert file_sha(p)==h,p
    obs,act,ep,t,y,parts,masks,source,_=load_data()
    observation,action=obs[ep,t],act[ep,t]
    x=np.concatenate([observation[:,:8],action,observation[:,8:]],1).astype(np.float64)
    train,val,test=[masks[k] for k in ['train','selection','test']]
    predictions={};fits={};normalization={};features={};start=time.perf_counter()
    for name in NAMES:
        step,state=load_checkpoint(RUNS/name/'final.pkl');assert step==150000
        phi,raw,audit=extract(state.q_params,observation,action)
        model,fit=ridge_fit(phi[train].astype(float),y[train],phi[val].astype(float),y[val],CONFIG['ridge_grid'])
        predictions[name+'_ridge']=ridge_predict(model,phi.astype(float));predictions[name+'_raw']=raw
        fits[name+'_ridge']=fit;normalization[name+'_ridge']={k:v for k,v in model.items() if k in ['mean','std']};features[name]=audit
        np.savez_compressed(root/'readouts'/f'{name}_ridge.npz',**model)
        print(f'{name}: feature extraction verified; ridge lambda={fit["selected_penalty"]}',flush=True)
    model,fit=ridge_fit(x[train],y[train],x[val],y[val],CONFIG['ridge_grid'])
    predictions['raw_input_ridge']=ridge_predict(model,x);fits['raw_input_ridge']=fit
    normalization['raw_input_ridge']={k:model[k] for k in ['mean','std']}
    np.savez_compressed(root/'readouts'/'raw_input_ridge.npz',**model)
    for seed in CONFIG['mlp']['seeds']:
        name=f'raw_mlp_s{seed}'
        predictions[name],fits[name],normalization[name]=fit_mlp(x,y,masks,seed,root)
        print(f'{name}: selected step {fits[name]["selected_step"]}',flush=True)
    predictions['constant_mean']=np.full(len(y),y[train].mean())
    predictions['constant_median']=np.full(len(y),np.median(y[train]))
    write(root/'readout_selection.json',fits);write(root/'normalization.json',normalization);write(root/'feature_verification.json',features)
    with np.load(root/'pair_indices.npz') as pairfile: pairs={k:pairfile[k] for k in pairfile.files}
    metrics={};target=y[test];test_ep=ep[test]
    ranges={'zero':target==0,'low':(target>0)&(target<5),'middle':(target>=5)&(target<10),'high':target>=10}
    for name,pred in predictions.items():
        pred=pred[test];calibrated=not name.endswith('_raw')
        assert np.isfinite(pred).all()
        result=regression_metrics(target,pred,calibrated)
        result.update(pairwise={k:pair_metrics(target,pred,v) for k,v in pairs.items()},
            by_source={str(s):regression_metrics(target[source[test]==s],pred[source[test]==s],calibrated) for s in range(3)},
            by_return={k:regression_metrics(target[mask],pred[mask],calibrated) for k,mask in ranges.items()})
        if calibrated: result['error_intervals']=error_intervals(target,pred,test_ep)
        metrics[name]=result
    differences={}
    for seed in [0,1]:
        for label,a,b in [('failure_negative_effect',f'alpha0p3_seed{seed}_ridge',f'alpha0_seed{seed}_ridge'),
                          ('readout_vs_scalar',f'alpha0p3_seed{seed}_ridge',f'alpha0p3_seed{seed}_raw'),
                          ('features_vs_raw_ridge',f'alpha0p3_seed{seed}_ridge','raw_input_ridge')]:
            va,valid=pair_values(target,predictions[a][test],pairs['matched']);vb,_=pair_values(target,predictions[b][test],pairs['matched'])
            delta=va-vb;rng=np.random.default_rng(95122)
            samples=delta[rng.integers(len(delta),size=(1000,len(delta)))].mean(1)
            differences[f'{label}_s{seed}']=dict(pairs=len(delta),accuracy_difference=float(delta.mean()),interval95=np.quantile(samples,[.025,.975]).tolist())
    write(root/'metrics.json',metrics);write(root/'paired_comparisons.json',differences)
    np.savez_compressed(root/'evaluation.npz',episode_id=test_ep,timestep=t[test],target=target,source=source[test],
        **{k:v[test] for k,v in predictions.items()})
    assert all(file_sha(p)==h for p,h in provenance['input_sha256'].items())
    write(root/'completion.json',dict(status='complete',input_models_and_data_unchanged=True,
        elapsed_seconds=time.perf_counter()-start,readout_files={p.name:file_sha(p) for p in (root/'readouts').iterdir()},
        test_predictions_sha256=file_sha(root/'evaluation.npz'),only_readouts_trained=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--run-prepared',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if not args.run_prepared: prepare(Path(args.out_dir))
        if not args.prepare_only: run(Path(args.out_dir))


if __name__=='__main__': main()
