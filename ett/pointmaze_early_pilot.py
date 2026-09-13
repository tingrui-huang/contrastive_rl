"""Preregistered observable early contexts and variable-horizon region NCE.

Prior pilot and finite-state implementations are imported without modification.
Hidden audit labels live in the separate evaluation module, never in training.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import jax
import numpy as np
import torch

from ett import pointmaze_region_pilot as old
from ett.finite_crl import write, sha

GROUPS = ('approach', 'transit', 'bypass')
TIMES = (1, 3, 6, 10)
TASK_HORIZON = 50
CONFIG = dict(base='0059c98', times=list(TIMES), groups=list(GROUPS), roots_per_group=12,
    selection_order=['bypass','transit','approach'], displacement_min=.05,
    alive_per_group_min=9, alive_total_min=30, generator_repeats=16,
    mixed_roots_per_approach_transit_min=2, each_outcome_count_min=2,
    recovered_paths_min=8, initial_value_std_min=.025,
    probe_response=.35, setback_distance=.25, setback_stationary_steps=3,
    recovery_first_goal_min=5, seeds=[0,1], arms=['diagonal','joint'], updates=6,
    rollout_repeats=8, eval_repeats=16, critic_initial_steps=1500,
    critic_refresh_steps=400, critic_final_steps=1000, critic_lr=.003,
    critic_batch=256, critic_width=64, embedding=16, horizon_scale=50,
    discount=.95, nce_B=32, alpha=0., q_negative=[.5,.5],
    calibration_rmse=.06, calibration_bias=.025, group_rmse=.07, group_bias=.035,
    directions=4, sigma_diagonal=.01, sigma_response=.1,
    rate_diagonal=.01, rate_response=.5, cap_diagonal=.03, cap_response=.1,
    lambda_off=1., diagonal_tolerance=.02, bound=1., geometry='rectangle',
    constraint_tolerance=2e-6, model_output_cap=1500000,
    root_seed=96000000, generator_seed=97000000, critic_seed=98000000,
    initial_seed=99000000, update_seed=100000000, evaluation_seed=101000000,
    bootstrap_seed=102000000, bootstrap_repeats=2000, final_iterates_only=True)


def group_mask(state, group):
    x,y = state[...,0],state[...,1]
    moving = np.linalg.norm(state[...,:2]-state[...,2:4],axis=-1) > .05
    outside = np.asarray(old.task_reward(state,np.broadcast_to(old.GOAL,state.shape))) == 0
    position = ((x>=1)&(x<3)&(y>=3)&(y<4) if group=='approach' else
                (x>=3)&(x<6)&(y>=3)&(y<4) if group=='transit' else
                (x>=1)&(x<6)&(y>=1)&(y<3))
    return moving & outside & position


def select_roots(obs, episodes, seed):
    """No actions, future returns or audit arrays accepted by this function."""
    rng = np.random.default_rng(seed); used=set(); rows=[]; counts={}
    for group in CONFIG['selection_order']:
        selected=[]; eligible_count=0
        for episode in rng.permutation(episodes):
            if int(episode) in used: continue
            for t in TIMES:
                s=obs[episode,t,:8]
                if not group_mask(s,group): continue
                prefix=obs[episode,:t+1,:8]
                if np.asarray(old.task_reward(prefix,np.broadcast_to(old.GOAL,prefix.shape))).any(): continue
                eligible_count+=1
                if len(selected)<12:
                    selected.append((int(episode),t,GROUPS.index(group)))
                    used.add(int(episode))
                break
        rows.extend(selected); counts[group]=dict(eligible_unused_episodes=eligible_count, selected=len(selected))
    rows.sort(key=lambda r:r[2])
    return np.asarray(rows,dtype=np.int64).reshape(-1,3),counts


class Critic(old.RegionCritic):
    @torch.no_grad()
    def predict(self,state,action,h,mean,std):
        f=self(torch.from_numpy(features(state,action,h,mean,std))).numpy()
        if not np.isfinite(f).all() or f.max()>600: raise FloatingPointError('Unsafe logits; no clipping.')
        q=(1-.95**np.asarray(h))*31*.5*np.exp(f[...,1])
        assert np.isfinite(q).all()
        return np.where(np.asarray(h)==0,0.,q)


def features(state,action,h,mean,std):
    return np.concatenate([(np.asarray(state)-mean)/std,np.asarray(action),np.asarray(h)[...,None]/50],-1).astype(np.float64)


def collect(engine,theta,roots,times,seed,ledger,purpose,repeats=8,first=None):
    """Group by actual remaining horizon, preserving root/repeat identities."""
    records=[]
    for t in sorted(np.unique(times)):
        ids=np.flatnonzero(times==t); h=50-int(t)
        s=np.repeat(roots[ids],repeats,0)
        a=None if first is None else np.repeat(first[ids],repeats,0)
        ledger.add(len(s)*h,purpose+f'_h{h}')
        record=engine.rollout(theta,s,jax.random.PRNGKey(seed+h),h,a)
        record['root']=np.repeat(ids,repeats)
        records.append(record)
    return records


def positives(records,seed,mean,std):
    xs=[]; labels=[]; rng=np.random.default_rng(seed)
    for r in records:
        n,h=r['action'].shape[:2]; y=np.empty((n,h),np.int64)
        for t in range(h):
            w=.95**np.arange(h-t)
            offset=rng.choice(h-t,n,p=w/w.sum())
            y[:,t]=r['reward'][np.arange(n),t+offset]
        hs=np.broadcast_to(np.arange(h,0,-1),(n,h))
        xs.append(features(r['states'][:,:-1],r['action'],hs,mean,std).reshape(-1,11));labels.append(y.ravel())
    return np.concatenate(xs),np.concatenate(labels)


def surrogate(engine,theta,critic,records,mean,std,seed,ledger):
    rng=np.random.default_rng(seed)
    # Uniform paths, then uniform within-path time: weight is H_i * gamma^t.
    paths=[(r,i) for r in records for i in range(len(r['states']))]
    selected=rng.integers(len(paths),size=128)
    states=[];actions=[];lengths=[];times=[]
    for index in selected:
        r,i=paths[index];h=r['action'].shape[1];t=int(rng.integers(h))
        states.append(r['states'][i,t]);actions.append(r['action'][i,t]);lengths.append(h);times.append(t)
    s=np.repeat(np.asarray(states),4,0);a=np.repeat(np.asarray(actions),4,0)
    xp=engine.nominal.sample(s,jax.random.PRNGKey(seed+1),1,goal=np.broadcast_to(old.GOAL,s.shape))
    ledger.add(len(s),'signed_variable_horizon_successors')
    y,d=engine.sample(theta,s,a,xp,jax.random.PRNGKey(seed+2),1)
    old.validate_selected_set(s,y,d);ns=np.asarray(y[:,0]);repeated=np.repeat(ns,4,0)
    next_a=np.asarray(engine.actor_jit(repeated,jax.random.PRNGKey(seed+3)))
    remaining=np.asarray(lengths)-np.asarray(times)-1
    q=critic.predict(repeated,next_a,np.repeat(remaining,16),mean,std).reshape(128,4,4).mean(-1)
    reward=np.asarray(old.task_reward(ns,np.broadcast_to(old.GOAL,ns.shape))).reshape(128,4)
    return float(np.mean(np.asarray(lengths)*.95**np.asarray(times)*(.05*reward+.95*q).mean(1)))


def prepare(out):
    if out.exists(): raise ValueError('Use a fresh output directory.')
    paths=old.inputs()
    source=['ett/pointmaze_early_pilot.py','ett/eval_pointmaze_early_pilot.py',
        'scripts/test_pointmaze_early_pilot.py','notes/pointmaze_early_pilot.md']
    previous=json.loads(Path('artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1/provenance.json').read_text())
    old.verify(Path('artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1'))
    hashes={**previous['inputs'],**{p:sha(p) for p in source}}
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    write(out/'config.json',CONFIG)
    (out/'PROTOCOL.md').write_bytes(Path('notes/pointmaze_early_pilot.md').read_bytes())
    # Seal protocol BEFORE looking at even observable coverage counts.
    write(out/'preregistration.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inputs=hashes,config_sha256=sha(out/'config.json'),protocol_sha256=sha(out/'PROTOCOL.md'),
        dataset=str(paths['dataset']),phase='before root selection'))
    with np.load(paths['dataset']) as z: obs=z['obs']
    assert obs.shape==(6600,51,16) and np.all(obs[:,:,8:]==old.GOAL)
    np.testing.assert_array_equal(obs[:,1:,2:8],obs[:,:-1,:6])
    held=np.random.default_rng(0).permutation(6600)[:660];teacher=np.arange(1200,6000)
    saved={};coverage={}
    for i,(part,ids) in enumerate([('train',np.setdiff1d(teacher,held)),('validation',np.intersect1d(teacher,held))]):
        rows,count=select_roots(obs,ids,CONFIG['root_seed']+i)
        saved[part+'_indices']=rows;saved[part+'_roots']=obs[rows[:,0],rows[:,1],:8]
        coverage[part]=count
    with np.load('artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1/contexts.npz') as z:
        saved.update({k:z[k] for k in z.files if k in ['mean','std'] or any(k==p+'_'+f for p in ['train','validation'] for f in ['state','action','target','goal','episode','time'])})
    np.savez_compressed(out/'contexts.npz',**saved)
    write(out/'selection.json',dict(groups=coverage,passed=all(c['selected']==12 for p in coverage.values() for c in p.values()),
        contexts_sha256=sha(out/'contexts.npz'),hidden_fields_read=False))


def verify(out):
    p=json.loads((out/'preregistration.json').read_text())
    assert json.loads((out/'config.json').read_text())==CONFIG
    assert sha(out/'config.json')==p['config_sha256'] and sha(out/'PROTOCOL.md')==p['protocol_sha256']
    for path,digest in p['inputs'].items(): assert sha(path)==digest,path
    assert sha(out/'contexts.npz')==json.loads((out/'selection.json').read_text())['contexts_sha256']


def train(engine,out,data,ledger,critic,opt):
    mean,std=data['mean'],data['std'];roots=data['train_roots'];times=data['train_indices'][:,1]
    initial_es=float(old.diagonal(engine,np.zeros(48,np.float32),data,'train',np.arange(256),16,
        CONFIG['initial_seed']+5000,ledger,'initial_guard').mean())
    initial_model,initial_opt=copy.deepcopy(critic.state_dict()),copy.deepcopy(opt.state_dict())
    records=[]
    for seed in CONFIG['seeds']:
        for arm in CONFIG['arms']:
            theta=np.zeros(48,np.float32);history=[];snapshots=[]
            critic=Critic(CONFIG['critic_seed']);critic.load_state_dict(initial_model)
            opt=torch.optim.Adam(critic.parameters(),lr=.003);opt.load_state_dict(copy.deepcopy(initial_opt))
            for it in range(6):
                key=CONFIG['update_seed']+100000*seed+1000*it
                paths=collect(engine,theta,roots,times,key,ledger,'refresh_training')
                loss=old.fit(critic,opt,positives(paths,key+100,mean,std),400,key+101)
                snapshots.append(theta.copy());before=old.parameter_sha(critic)
                rng=np.random.default_rng(key+102);dirs=rng.normal(size=(4,48));idx=rng.integers(512,size=128)
                sigma=np.r_[np.full(16,.01),np.full(32,.1)];signed=[]
                for direction in dirs:
                    pair=[]
                    for sign in [1,-1]:
                        trial=(theta+sign*sigma*direction).astype(np.float32)
                        es=float(old.diagonal(engine,trial,data,'train',idx,8,key+300,ledger,'signed_diagonal').mean())
                        off=surrogate(engine,trial,critic,paths,mean,std,key+200,ledger)
                        pair.append([es,off])
                    signed.append(pair)
                signed=np.asarray(signed)
                components=np.einsum('dt,di->ti',signed[:,0]-signed[:,1],dirs)/(2*sigma*4)
                components[0,16:]=0
                gradient=components[0]+(components[1] if arm=='joint' else 0.)
                step=gradient*np.r_[np.full(16,.01),np.full(32,.5)]
                for sl,cap in [(slice(0,16),.03),(slice(16,48),.1)]:
                    step[sl]*=min(1.,cap/max(np.linalg.norm(step[sl]),1e-12))
                proposed=(theta-step).astype(np.float32)
                es=float(old.diagonal(engine,proposed,data,'train',np.arange(256),16,CONFIG['initial_seed']+5000,ledger,'candidate_guard').mean())
                accepted=es<=initial_es+.02
                if accepted:theta=proposed
                assert old.parameter_sha(critic)==before and np.isfinite(theta).all()
                history.append(dict(iteration=it,nce=loss,signed_terms=signed,diagonal_gradient=components[0],
                    off_gradient=components[1],candidate_es=es,accepted=bool(accepted)))
            name=f'{arm}_s{seed}'
            np.savez_compressed(out/'checkpoints'/(name+'.npz'),theta=theta,snapshots=snapshots)
            torch.save(dict(model=critic.state_dict(),optimizer=opt.state_dict()),out/'checkpoints'/(name+'.pt'))
            records.append(dict(name=name,seed=seed,arm=arm,history=history,theta_sha256=sha(out/'checkpoints'/(name+'.npz')),
                critic_sha256=sha(out/'checkpoints'/(name+'.pt'))))
            print(name+': six updates complete',flush=True)
    write(out/'training.json',dict(status='complete',initial_train_es=initial_es,records=records,updates=24))


def run(out):
    from ett import eval_pointmaze_early_pilot as evaluation
    verify(out)
    if (out/'started.json').exists():raise ValueError('Preserve the existing attempt.')
    write(out/'started.json',dict(status='started'))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=dict(np.load(out/'contexts.npz'));ledger=old.Ledger(out)
    ledger.data['cap']=CONFIG['model_output_cap'];ledger.add(0,'begin')
    if not evaluation.coverage(out,data):
        evaluation.finish(out,data,ledger,'coverage');return
    engine=old.Kernel();theta=np.zeros(48,np.float32)
    if not evaluation.generator_gate(engine,out,data,ledger):
        evaluation.finish(out,data,ledger,'generator_expressiveness');return
    critic=Critic(CONFIG['critic_seed']);opt=torch.optim.Adam(critic.parameters(),lr=.003)
    paths=collect(engine,theta,data['train_roots'],data['train_indices'][:,1],CONFIG['initial_seed'],ledger,'initial_critic_training')
    loss=old.fit(critic,opt,positives(paths,CONFIG['initial_seed']+100,data['mean'],data['std']),1500,CONFIG['critic_seed'])
    metrics,arrays=evaluation.calibrate(engine,theta,critic,data,CONFIG['initial_seed']+10000,ledger,'preflight')
    write(out/'preflight.json',dict(**metrics,nce_loss=loss));np.savez_compressed(out/'preflight.npz',**arrays)
    np.savez(out/'checkpoints'/'initial.npz',theta=theta)
    torch.save(dict(model=critic.state_dict(),optimizer=opt.state_dict()),out/'checkpoints'/'initial.pt')
    if not metrics['passed']:
        evaluation.finish(out,data,ledger,'value_estimation');return
    train(engine,out,data,ledger,critic,opt)
    evaluation.final_evaluation(engine,out,data,ledger)
    evaluation.finish(out,data,ledger,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=['prepare','run']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();(prepare if args.phase=='prepare' else run)(args.out)
