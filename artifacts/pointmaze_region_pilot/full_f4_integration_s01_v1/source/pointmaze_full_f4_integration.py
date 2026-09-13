"""One bounded integration using the original full-F4 CRL + BC learner."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import jax
import jax.numpy as j
import numpy as np
import optax

from crl import checkpoint
from ett import run_policy_improvement as original
from ett import pointmaze_native_policy as previous
from ett.policy_improvement import SegmentReplay, replay, mixed_batch, tree_sha
from ett.finite_crl import write, sha
from ett.rollout_return import START, GOAL

SPEC=Path('notes/pointmaze_full_f4_integration.md')
CONFIG=dict(seeds=[0,1], arms=['O','B','C'], updates=1000, refresh=250,
    roots_per_time=32, root_times=[0,10,25,40], synthetic_fraction=.1,
    native_episodes=256, model_episodes=128, horizon=50, gamma=.95,
    train_seed=154000000, rollout_seed=155000000, native_reset_seed=150000000,
    native_action_seed=151000000, model_seed=152000000, bootstrap_seed=153000000,
    model_cap=243200, native_cap=89600, joint_updates=6000)

def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def learner():
    cfg,net,step=original.learner_setup()
    cfg.max_number_of_steps=CONFIG['updates']
    return cfg,net,step

def prepare(out):
    if out.exists(): raise ValueError('Fresh output directory required')
    cfg,net,_=learner(); paths=original.inputs()
    step,initial=checkpoint.load_checkpoint(paths['actor']); assert step==150000
    obs,act,ids,heldout=original.dataset()
    # Verify frozen final kernels against their prior completion manifest.
    manifest=json.loads((previous.PREVIOUS/'manifest.json').read_text())
    assert sha(previous.PREVIOUS/'final_kernels.npz')==manifest['final_kernels.npz']
    sources=['ett/pointmaze_full_f4_integration.py','ett/run_policy_improvement.py',
        'ett/policy_improvement.py','ett/pointmaze_native_policy.py','crl/losses.py',
        'crl/networks.py','crl/replay.py','crl/config.py','crl/checkpoint.py','crl/envs.py',
        'ett/fixed_actor_continuation.py','ett/pointmaze_region_pilot.py',
        'ett/convex_action_transition.py','ett/diagonal_transition.py',
        'propensity/nominal_policy.py','scripts/run_swamp_windy_z_failneg.py',
        'scripts/test_pointmaze_full_f4_integration.py',str(SPEC)]
    inputs=list(paths.values())+[previous.PREVIOUS/'final_kernels.npz',
        paths['actor'].with_name('arm_provenance.json')]
    out.mkdir(parents=True); (out/'checkpoints').mkdir(); (out/'rounds').mkdir()
    write(out/'config.json',CONFIG); write(out/'learner_config.json',dataclasses.asdict(cfg))
    (out/'PROTOCOL.md').write_bytes(SPEC.read_bytes())
    np.savez_compressed(out/'partition.npz',train=ids,heldout=heldout)
    bank=obs.reshape(-1,16)[np.random.default_rng(156000000).choice(obs.shape[0]*51,1024,False),:8]
    np.savez_compressed(out/'policy_bank.npz',states=bank)
    write(out/'preregistration.json',dict(utc=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={str(p):sha(p) for p in sources+inputs},
        sealed={f:sha(out/f) for f in ['config.json','learner_config.json','PROTOCOL.md','partition.npz','policy_bank.npz']},
        initial_state_sha256=tree_sha(initial),initial_actor_sha256=tree_sha(initial.policy_params),
        initialization='full checkpoint and optimizer restore; only learner RNG changes',
        device=[str(d) for d in jax.devices()],python=platform.python_version()))
    print('Protocol, budget, original learner configuration and input hashes sealed.',flush=True)

def verify(out):
    p=json.loads((out/'preregistration.json').read_text())
    assert json.loads((out/'config.json').read_text())==CONFIG
    for f,h in p['sources'].items(): assert sha(f)==h,f
    for f,h in p['sealed'].items(): assert sha(out/f)==h,f

class Ledger:
    def __init__(self,out): self.out=out; self.data=dict(charged=0,cap=CONFIG['model_cap'],entries=[])
    def add(self,n,purpose):
        assert self.data['charged']+n<=self.data['cap']
        self.data['charged']+=n; self.data['entries'].append(dict(purpose=purpose,outputs=n))
        write(self.out/'model_ledger.json',self.data)

def synthetic_replay(records,seed):
    buffer=SegmentReplay(128*51,51,16,2,8,0,-1,.95,seed)
    for record in records:
        for states,actions in zip(record['states'],record['action']):
            length=len(states)
            observation=np.full((51,16),np.nan,np.float32)
            action=np.full((51,2),np.nan,np.float32)
            observation[:length]=np.concatenate([states,np.broadcast_to(GOAL,states.shape)],-1)
            action[:length-1]=actions
            buffer.add_episode(observation,action,length=length)
    assert buffer._num_eps==128 and len(buffer)==4000
    buffer.freeze(); return buffer

def train(out):
    verify(out)
    if (out/'started.json').exists(): raise ValueError('Existing training attempt protected')
    write(out/'started.json',dict(utc=stamp()))
    cfg,net,update=learner(); _,initial=checkpoint.load_checkpoint(original.inputs()['actor'])
    obs,act,ids,_=original.dataset(); engine=previous.old.Kernel()
    models=previous.models(); frozen=tree_sha((engine.base.params,engine.nominal.params,models))
    generator=previous.FrozenRollouts(engine,net); ledger=Ledger(out)
    bank=np.load(out/'policy_bank.npz')['states']
    change=jax.jit(lambda p:previous.policy_change(net,initial.policy_params,p,bank))
    checkpoint.save_named(str(out/'checkpoints'),'I',150000,initial)
    runs={}
    for seed in CONFIG['seeds']:
        starts=[]; pairing=[]
        for arm in CONFIG['arms']:
            name=f'{arm}_s{seed}'; base=CONFIG['train_seed']+seed
            state=initial._replace(key=jax.random.PRNGKey(base)); starts.append(tree_sha(state))
            offline=replay(obs,act,base+10); order=np.random.default_rng(base+20)
            roots_rng=np.random.default_rng(CONFIG['rollout_seed']+seed)
            synthetic=None; curves=[]; counts=0; pair_hash=hashlib.sha256(); off_hash=hashlib.sha256()
            generation=[]; audit_rows=[]; started=time.monotonic()
            for u in range(CONFIG['updates']):
                if arm!='O' and u%CONFIG['refresh']==0:
                    folder=out/'rounds'/f'{name}_u{u}'; folder.mkdir()
                    theta,observational=models[f'obs_s{seed}' if arm=='B' else f'pess_s{seed}']
                    records=[]
                    for t in CONFIG['root_times']:
                        e=roots_rng.integers(len(obs),size=32); h=50-t
                        key=CONFIG['rollout_seed']+10000*seed+100*u+t
                        ledger.add(32*h,name+f'/refresh{u}/h{h}')
                        r=generator.rollout(state.policy_params,theta,observational,obs[e,t,:8],jax.random.PRNGKey(key),h)
                        r.update(episode=ids[e],root_time=np.full(32,t))
                        np.savez_compressed(folder/f'h{h}.npz',**r); records.append(r)
                        pair_hash.update(ids[e].tobytes())
                    synthetic=synthetic_replay(records,base+100+u)
                    generation.append(dict(update=u,actor_sha256=tree_sha(state.policy_params),transitions=4000))
                batch,audit=mixed_batch(offline,synthetic,u,order); counts+=audit['synthetic_count']
                for v in audit['offline'].values(): off_hash.update(v.tobytes())
                off_hash.update(audit['permutation'].tobytes())
                if synthetic is not None:
                    ix=audit['synthetic']; assert np.all(ix['future']<synthetic.lengths[ix['episode']])
                    for v in ix.values(): pair_hash.update(v.tobytes())
                compact={f'off_{k}':v for k,v in audit['offline'].items()}
                compact.update(permutation=audit['permutation'],source=audit['source'])
                for k in ['episode','time','future']:
                    v=np.full(26,-1,np.int32)
                    if synthetic is not None: v[:audit['synthetic_count']]=audit['synthetic'][k]
                    compact['syn_'+k]=v
                audit_rows.append(compact)
                state,metrics=update(state,jax.tree.map(j.asarray,batch))
                row={k:float(v) for k,v in metrics.items()}
                assert all(np.isfinite(v) for v in row.values()),(name,u,row)
                curves.append(row)
                if (u+1)%100==0:
                    print(name,u+1,'/1000; actor loss',round(row['actor_loss'],5),'elapsed',round(time.monotonic()-started,1),flush=True)
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(state))
            assert counts==(0 if arm=='O' else 25600)
            checkpoint.save_named(str(out/'checkpoints'),name+'_final',151000,state)
            np.savez_compressed(out/f'{name}_batch_audit.npz',**{k:np.stack([r[k] for r in audit_rows]) for k in audit_rows[0]})
            entry=dict(updates=1000,synthetic_rows=counts,total_rows=256000,refreshes=generation,
                initial_state_sha256=starts[-1],final_state_sha256=tree_sha(state),
                final_actor_sha256=tree_sha(state.policy_params),final_critic_sha256=tree_sha(state.q_params),
                policy_change=np.asarray(change(state.policy_params)),
                parameter_l2=float(optax.global_norm(jax.tree.map(lambda a,b:a-b,state.policy_params,initial.policy_params))),
                critic_parameter_l2=float(optax.global_norm(jax.tree.map(lambda a,b:a-b,state.q_params,initial.q_params))),
                offline_draw_sha256=off_hash.hexdigest(),synthetic_pairing_sha256=pair_hash.hexdigest(),
                seconds=time.monotonic()-started,metrics=curves,
                checkpoint_sha256=sha(out/'checkpoints'/f'{name}_final.pkl'))
            runs[name]=entry; write(out/'training_progress.json',runs)
            assert entry['parameter_l2']>0 and entry['critic_parameter_l2']>0
            if arm!='O': pairing.append(pair_hash.hexdigest())
        assert len(set(starts))==1 and len(set(pairing))==1
        assert len({runs[f'{a}_s{seed}']['offline_draw_sha256'] for a in CONFIG['arms']})==1
    assert frozen==tree_sha((engine.base.params,engine.nominal.params,models))
    assert ledger.data['charged']==64000
    assert tree_sha(initial)==json.loads((out/'preregistration.json').read_text())['initial_state_sha256']
    write(out/'training.json',dict(utc=stamp(),runs=runs,frozen_sha256=frozen,joint_updates=6000,
        model_transitions=64000,initial_actor_unchanged=True))
    verify(out)

def paired(a,b):
    n=len(a['returns']); rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    idx=rng.integers(n,size=(2000,n)); result={}
    for k in a:
        d=a[k]-b[k]; row=dict(difference=float(d.mean()),ci95=np.quantile(d[idx].mean(1),[.025,.975]),all_ties=bool(np.all(d==0)))
        if k!='returns': row.update(a_one_b_zero=int(np.sum((a[k]==1)&(b[k]==0))),a_zero_b_one=int(np.sum((a[k]==0)&(b[k]==1))))
        result[k]=row
    return result

def evaluate(out):
    # Native simulator is imported only in this phase, after final checkpoints.
    from ett.fixed_actor_continuation import TimedSimulator
    verify(out); training=json.loads((out/'training.json').read_text())
    if (out/'evaluation_started.json').exists(): raise ValueError('Existing evaluation protected')
    _,net,_=learner(); actors={'I':checkpoint.load_checkpoint(out/'checkpoints'/'I.pkl')[1].policy_params}
    for name,row in training['runs'].items():
        p=out/'checkpoints'/f'{name}_final.pkl'; assert sha(p)==row['checkpoint_sha256']
        actors[name]=checkpoint.load_checkpoint(p)[1].policy_params
    hashes={k:tree_sha(v) for k,v in actors.items()}
    write(out/'evaluation_started.json',dict(utc=stamp(),training_sha256=sha(out/'training.json'),actors=hashes))
    sample=jax.jit(lambda p,s,k:previous.sample_actor(net,p,s,k))
    native={}; steps=0
    for name,params in actors.items():
        seeds=np.arange(256)+CONFIG['native_reset_seed']; sims=[TimedSimulator(int(s)) for s in seeds]
        states=[np.stack([s.observation()[:8] for s in sims])]; actions=[]; rewards=[]; failures=[]
        np.testing.assert_array_equal(states[0],np.tile(START,(256,1)))
        steps+=256*50; assert steps<=CONFIG['native_cap']
        write(out/'native_ledger.json',dict(charged=steps,cap=CONFIG['native_cap'],actor=name))
        for t in range(50):
            a=np.asarray(sample(params,states[-1],jax.random.PRNGKey(CONFIG['native_action_seed']+t)))
            v=[s.step(x) for s,x in zip(sims,a)]
            states.append(np.stack([x[0][:8] for x in v])); actions.append(a)
            rewards.append([x[1] for x in v]); failures.append([s.env.dead for s in sims])
        r=dict(states=np.stack(states,1),action=np.stack(actions,1),reward=np.array(rewards).T,
            failure=np.array(failures).T,reset_seed=seeds)
        np.savez_compressed(out/f'{name}_native.npz',**r)
        native[name]=dict(returns=r['reward']@.95**np.arange(50),
            success=(np.linalg.norm(r['states'][...,:2]-GOAL[:2],axis=-1).min(1)<.5).astype(float),
            failure=r['failure'][:,-1].astype(float))
        print(name+': 256 fresh paired native episodes complete.',flush=True)
    engine=previous.old.Kernel(); generator=previous.FrozenRollouts(engine,net); models=previous.models()
    assert tree_sha((engine.base.params,engine.nominal.params,models))==training['frozen_sha256']
    ledger=Ledger(out); ledger.data=json.loads((out/'model_ledger.json').read_text()); model={}
    for m,(theta,observational) in models.items():
        model[m]={}
        for name,params in actors.items():
            ledger.add(128*50,'eval/'+m+'/'+name)
            r=generator.rollout(params,theta,observational,np.tile(START,(128,1)),jax.random.PRNGKey(CONFIG['model_seed']),50)
            np.savez_compressed(out/f'{m}_{name}_model.npz',**r)
            returns=r['reward']@.95**np.arange(50)
            model[m][name]=dict(returns=returns,success=(np.linalg.norm(r['states'][...,:2]-GOAL[:2],axis=-1).min(1)<.5).astype(float),zero_return=(returns==0).astype(float))
        print(m+': independent rollouts for all seven actors complete.',flush=True)
    summary=lambda arrays:{a:{k:float(v.mean()) for k,v in row.items()} for a,row in arrays.items()}
    contrasts={}; evidence=[]
    for seed in [0,1]:
        for a,b in [('C','O'),('C','B'),('B','O'),('O','I'),('B','I'),('C','I')]:
            left=f'{a}_s{seed}'; right='I' if b=='I' else f'{b}_s{seed}'
            contrasts[left+'_minus_'+right]=paired(native[left],native[right])
        for b in ['O','B']:
            row=contrasts[f'C_s{seed}_minus_{b}_s{seed}']
            evidence.append(row['returns']['ci95'][0]>0 and row['success']['ci95'][1]>=0 and row['failure']['ci95'][0]<=0)
    pooled={}
    for a,b in [('C','O'),('C','B'),('B','O')]:
        avg=lambda arm:{k:np.mean([native[f'{arm}_s{s}'][k] for s in [0,1]],axis=0) for k in native['I']}
        pooled[a+'_minus_'+b]=paired(avg(a),avg(b))
        for row in pooled[a+'_minus_'+b].values():
            row.pop('a_one_b_zero',None); row.pop('a_zero_b_one',None)
    result=dict(native=summary(native),native_contrasts=contrasts,conditional_two_seed_mean=pooled,
        model={m:summary(v) for m,v in model.items()},
        model_vs_I={m:{a:paired(v,rows['I']) for a,v in rows.items() if a!='I'} for m,rows in model.items()},
        model_primary={m:{f'C_s{s}_minus_{b}_s{s}':paired(rows[f'C_s{s}'],rows[f'{b}_s{s}']) for s in [0,1] for b in ['O','B']} for m,rows in model.items()},
        consistent_added_benefit=bool(all(evidence)))
    write(out/'results.json',result)
    assert steps==89600 and ledger.data['charged']==243200
    assert tree_sha((engine.base.params,engine.nominal.params,models))==training['frozen_sha256']
    assert {k:tree_sha(v) for k,v in actors.items()}==hashes
    verify(out)
    write(out/'completion.json',dict(utc=stamp(),joint_updates=6000,model_transitions=243200,native_steps=89600,
        frozen_verified=True,final_only=True,native_training_or_selection=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__); p.add_argument('phase',choices=['prepare','train','evaluate'])
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    globals()[a.phase](a.out)
