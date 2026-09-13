"""Fixed-budget response-only optimization and matched continuation diagnosis."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import jax
import jax.numpy as j
import numpy as np
import torch
from ett import pointmaze_update_reference as prior
from ett import eval_pointmaze_update_reference as evaluation
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett import pointmaze_phase_sampling as phase
from ett import pointmaze_future_average as average
from ett.policy_improvement import tree_sha
from ett.finite_crl import write,sha
from ett.rollout_return import START,GOAL

BASE=Path('artifacts/pointmaze_region_pilot/update_reference_s01_v1')
SPEC=Path('notes/pointmaze_response_search.md')
CONFIG=dict(base='a439486',seeds=[0,1],starts=3,updates=24,directions=8,sigma=.1,
    response_lr=.05,response_step_cap=.25,betas=[.9,.999],adam_eps=1e-8,
    query_count=32,successors=4,actor_draws=4,paths_per_root=4,
    first_refresh=1000,refresh=300,critic_steps=47400,
    audit_updates=[0,11,23],audit_successors=8,audit_actors=4,
    pool_updates=[0,12,24],selection_repeats=16,final_repeats=64,reset_repeats=128,
    init_seed=160000000,search_seed=161000000,mc_seed=162000000,
    selection_seed=163000000,final_seed=164000000,reset_seed=165000000,
    bootstrap_seed=166000000,diagonal_seed=167000000,
    gamma=.95,geometry='rectangle',bound=1.,model_cap=6000000,
    native_steps=0,actor_updates=0)

def read(path): return json.loads(Path(path).read_text())

class Ledger(old.Ledger):
    def __init__(self,out,restore=False):
        super().__init__(out);self.data['cap']=CONFIG['model_cap']
        if restore:self.data=read(out/'ledger.json')

def project_response(x):
    x=np.asarray(x).reshape(8,2,2)
    return (x/np.maximum(1,np.linalg.norm(x,axis=(1,2),keepdims=True))).astype(np.float32).ravel()

def starts(base):
    result=[base.copy()]
    for k in [1,2]:
        r=np.random.default_rng(CONFIG['init_seed']+k).normal(size=(8,2,2))
        r=.5*r/np.linalg.norm(r,axis=(1,2),keepdims=True)
        result.append(np.r_[base[:16],r.ravel()].astype(np.float32))
    return result

def response_step(theta,directions,signed,m,v,t):
    gradient=np.mean((signed[:,0]-signed[:,1])[:,None]*directions,axis=0)/(2*CONFIG['sigma'])
    m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
    delta=CONFIG['response_lr']*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+CONFIG['adam_eps'])
    delta*=min(1,CONFIG['response_step_cap']/max(np.linalg.norm(delta),1e-15))
    candidate=np.r_[theta[:16],project_response(theta[16:]-delta)].astype(np.float32)
    return candidate,m,v,dict(gradient=gradient,step=delta,actual_step_norm=float(np.linalg.norm(candidate[16:]-theta[16:])))

def frozen_hash(engine): return tree_sha((engine.base.params,engine.nominal.params,engine.actor_info))

def prepare(out):
    if out.exists():raise ValueError('Fresh output directory required')
    assert subprocess.check_output(['git','rev-parse','--short','HEAD'],text=True).strip()=='a439486'
    manifest=read(BASE/'manifest.json')
    inputs=[BASE/'final_kernels.npz']+[BASE/'checkpoints'/f'critic_s{s}_u2.pt' for s in [0,1]]
    assert sha(inputs[0])==manifest['final_kernels.npz']
    previous_training=read(BASE/'training.json')
    for s,f in enumerate(inputs[1:]):
        assert sha(f)==previous_training['histories'][f'critic_s{s}'][2]['checkpoint_sha256']
    inputs+=list(old.inputs().values())+[phase.PREVIOUS/'contexts.npz',average.PREVIOUS/'final_contexts.npz']
    sources=[str(SPEC),'ett/pointmaze_response_search.py','ett/report_response_search.py',
        'scripts/test_response_search.py','ett/pointmaze_update_reference.py','ett/eval_pointmaze_update_reference.py',
        'ett/pointmaze_region_pilot.py','ett/pointmaze_early_pilot.py','ett/pointmaze_future_average.py',
        'ett/convex_action_transition.py','ett/diagonal_transition.py','propensity/nominal_policy.py']
    inherited=read(BASE/'preregistration.json')['sources']
    out.mkdir(parents=True);(out/'checks').mkdir();(out/'checkpoints').mkdir()
    write(out/'config.json',CONFIG);(out/'PROTOCOL.md').write_bytes(SPEC.read_bytes())
    kernels=dict(np.load(BASE/'final_kernels.npz'))
    np.savez_compressed(out/'starts.npz',**{f's{s}_k{k}_u0':v for s in [0,1] for k,v in enumerate(starts(kernels[f'critic_s{s}']))})
    write(out/'preregistration.json',dict(utc=average.timestamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={**inherited,**{str(f):sha(f) for f in sources+inputs}},
        sealed={f:sha(out/f) for f in ['config.json','PROTOCOL.md','starts.npz']}))
    print('Response-only protocol, starts, input hashes and fixed budget sealed.',flush=True)

def verify(out):
    p=read(out/'preregistration.json');assert read(out/'config.json')==CONFIG
    for f,h in p['sources'].items():assert sha(f)==h,f
    for f,h in p['sealed'].items():assert sha(out/f)==h,f

class Continuation:
    """Frozen pre-update MC, actual horizons masked; charge all computed slots."""
    def __init__(self,engine):
        def run(theta,states,first,horizon,key):
            def step(s,tk):
                t,key=tk;bk,ak,sk=jax.random.split(key,3)
                xp=engine.nominal.sample(s,bk,1,goal=j.broadcast_to(j.asarray(GOAL),s.shape))
                a=engine.actor_jit(s,ak);a=j.where(t==0,first,a)
                y,d=engine._sample(theta,s,a,xp,sk,1);ns=y[:,0]
                active=t<horizon
                reward=j.where(active,old.task_reward(ns,j.broadcast_to(j.asarray(GOAL),ns.shape)),0.)
                return j.where(active[:,None],ns,s),(reward,d['box_valid'][:,0])
            _,(reward,valid)=jax.lax.scan(step,states,(j.arange(48),jax.random.split(key,48)))
            return reward.T,valid.T
        self.run=jax.jit(run)

def first_samples(engine,theta,queries,critic,context,seed,draws,actors,ledger,purpose):
    n=len(queries['h']);s=np.repeat(queries['state'],draws,0);a=np.repeat(queries['action'],draws,0)
    xp=engine.nominal.sample(s,jax.random.PRNGKey(seed),1,goal=np.broadcast_to(GOAL,s.shape))
    ledger.add(len(s),purpose)
    y,d=engine.sample(theta,s,a,xp,jax.random.PRNGKey(seed+1),1);old.validate_selected_set(s,y,d)
    ns=np.asarray(y[:,0]);repeat=np.repeat(ns,actors,0)
    next_a=np.asarray(engine.actor_jit(repeat,jax.random.PRNGKey(seed+2)))
    h=np.repeat(queries['h']-1,draws*actors)
    q=critic.predict(repeat,next_a,h,context['mean'],context['std']).reshape(n,draws,actors)
    r=np.asarray(old.task_reward(ns,np.broadcast_to(GOAL,ns.shape))).reshape(n,draws)
    integrand=queries['weight'][:,None,None]*(.05*r[:,:,None]+.95*q)
    return dict(successor=ns.reshape(n,draws,8),next_action=next_a.reshape(n,draws,actors,2),
        x_prime=np.asarray(xp).reshape(n,draws,2),reward=r,q=q,critic=integrand)

def uncertainty(samples,roots):
    """Rows=queries; columns=independent successor groups, actor draws averaged."""
    x=np.asarray(samples);means=x.mean(1);root_ids,inverse=np.unique(roots,return_inverse=True)
    sums=np.bincount(inverse,weights=means);counts=np.bincount(inverse)
    idx=np.random.default_rng(CONFIG['bootstrap_seed']).integers(len(sums),size=(2000,len(sums)))
    value=float(x.mean());se=float(np.sqrt(np.sum(x.var(1,ddof=1)/x.shape[1]))/len(x))
    return dict(mean=value,root_ci95=np.quantile(sums[idx].sum(1)/counts[idx].sum(1),[.025,.975]),
        conditional_mc_ci95=[value-1.96*se,value+1.96*se],roots=len(root_ids))

def matched_check(engine,mc,pre,proposed,records,critic,context,seed,ledger,folder):
    queries=prior.visitation(records,seed,32);raw={}
    for label,theta in [('before',pre),('proposed',proposed)]:
        r=first_samples(engine,theta,queries,critic,context,seed+10,8,4,ledger,'matched/first')
        s=np.repeat(r['successor'].reshape(-1,8),4,0);a=r['next_action'].reshape(-1,2)
        h=np.repeat(queries['h']-1,32);assert h.max()<=48 and h.min()>=0
        ledger.add(len(s)*48,'matched/continuation_including_padding')
        rewards,valid=mc.run(pre,s,a,h,jax.random.PRNGKey(seed+20))
        rewards=np.asarray(rewards);assert np.asarray(valid).all()
        assert np.all(rewards[np.arange(48)[None,:]>=h[:,None]]==0)
        q=(.05*rewards@.95**np.arange(48)).reshape(32,8,4)
        r.update(mc=queries['weight'][:,None,None]*(.05*r['reward'][:,:,None]+.95*q),
            continuation_rewards=rewards.reshape(32,8,4,48),remaining=h.reshape(32,8,4),mc_q=q)
        raw[label]=r
    np.testing.assert_array_equal(raw['before']['x_prime'],raw['proposed']['x_prime'])
    c=(raw['proposed']['critic']-raw['before']['critic']).mean(-1)
    m=(raw['proposed']['mc']-raw['before']['mc']).mean(-1)
    result={k:uncertainty(v,queries['root']) for k,v in [('critic',c),('mc',m),('mc_minus_critic',m-c)]}
    np.savez_compressed(folder/'matched_mc.npz',**{label+'_'+k:v for label,r in raw.items() for k,v in r.items()},
        **{'query_'+k:v for k,v in queries.items()},preupdate=pre,proposed=proposed)
    write(folder/'matched_mc.json',result);return result

def diagonal_probe(engine,theta,context,key,ledger):
    s=context['validation_state'][:16];a=context['validation_action'][:16]
    ledger.add(64,'constraint/diagonal_identity')
    y,d=engine.sample(theta,s,a,a,jax.random.PRNGKey(key),4)
    old.validate_selected_set(s,y,d);return np.asarray(y)

def search(out):
    verify(out)
    if (out/'started.json').exists():raise ValueError('Existing experiment protected')
    write(out/'started.json',dict(utc=average.timestamp()))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    engine=old.Kernel();frozen=frozen_hash(engine);mc=Continuation(engine)
    context=dict(np.load(phase.PREVIOUS/'contexts.npz'));ledger=Ledger(out)
    initial=dict(np.load(out/'starts.npz'));pool=initial.copy();histories={};pre_final={}
    for seed in [0,1]:
        base=initial[f's{seed}_k0_u0'];diagonal=diagonal_probe(engine,base,context,CONFIG['diagonal_seed'],ledger)
        saved=torch.load(BASE/'checkpoints'/f'critic_s{seed}_u2.pt',weights_only=True)
        for start in range(3):
            name=f's{seed}_k{start}';theta=initial[name+'_u0'].copy();m=np.zeros(32);v=np.zeros(32);history=[]
            np.testing.assert_array_equal(diagonal_probe(engine,theta,context,CONFIG['diagonal_seed'],ledger),diagonal)
            critic=early.Critic(0);critic.load_state_dict(saved['model'])
            optimizer=torch.optim.Adam(critic.parameters(),lr=.003);optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
            for u in range(24):
                key=CONFIG['search_seed']+seed*1000000+u*1000 # paired streams across starts
                pre=theta.copy()
                if u==23:pre_final[name+'_u23']=pre.copy()
                records=early.collect(engine,pre,context['train_roots'],context['train_indices'][:,1],key,ledger,'search/'+name+'/paths',4)
                fit=prior.refresh(critic,optimizer,records,context,key+100,1000 if u==0 else 300)
                critic_hash=old.parameter_sha(critic);optimizer_hash=tree_sha(optimizer.state_dict())
                queries=prior.visitation(records,key+101,32)
                directions=np.random.default_rng(key+102).normal(size=(8,32));signed=np.empty((8,2))
                for i,direction in enumerate(directions):
                    for k,sign in enumerate([1,-1]):
                        candidate=np.r_[pre[:16],project_response(pre[16:]+sign*.1*direction)].astype(np.float32)
                        signed[i,k]=first_samples(engine,candidate,queries,critic,context,key+200,4,4,ledger,'search/signed')['critic'].mean()
                theta,m,v,info=response_step(pre,directions,signed,m,v,u+1)
                local=[]
                for candidate in [pre,theta]:
                    local.append(float(first_samples(engine,candidate,queries,critic,context,key+200,4,4,ledger,'search/local')['critic'].mean()))
                assert np.isfinite(theta).all();np.testing.assert_array_equal(theta[:16],base[:16])
                np.testing.assert_array_equal(diagonal_probe(engine,theta,context,CONFIG['diagonal_seed'],ledger),diagonal)
                row=dict(update=u+1,theta=theta,preupdate=pre,fit=fit,local_before=local[0],local_proposed=local[1],
                    critic_sha256=critic_hash,directions=directions,signed=signed,**info)
                if u in CONFIG['audit_updates']:
                    folder=out/'checks'/f'{name}_u{u+1}';folder.mkdir()
                    row['matched']=matched_check(engine,mc,pre,theta,records,critic,context,
                        CONFIG['mc_seed']+seed*1000000+start*100000+u*1000,ledger,folder)
                    torch.save(dict(model=critic.state_dict()),folder/'critic.pt')
                    phase.save_records(folder/'visitation.npz',records)
                assert old.parameter_sha(critic)==critic_hash and tree_sha(optimizer.state_dict())==optimizer_hash
                history.append(row)
                if u+1 in CONFIG['pool_updates']:pool[name+f'_u{u+1}']=theta.copy()
                print(name,f'update {u+1}/24; diagonal exact; local delta {local[1]-local[0]:+.5f}',flush=True)
            histories[name]=history;write(out/'search_progress.json',histories)
    assert frozen_hash(engine)==frozen
    np.savez_compressed(out/'candidate_pool.npz',**pool)
    np.savez_compressed(out/'pre_final.npz',**pre_final)
    write(out/'search.json',dict(utc=average.timestamp(),histories=histories,frozen_sha256=frozen,
        candidate_pool_sha256=sha(out/'candidate_pool.npz'),pre_final_sha256=sha(out/'pre_final.npz'),updates=144,critic_steps=47400))
    verify(out)

def full_returns(engine,theta,roots,indices,repeats,seed,ledger,purpose,path):
    records=early.collect(engine,theta,roots,indices[:,1],seed,ledger,purpose,repeats)
    phase.save_records(path,records)
    values,_=evaluation.pack_returns(records,len(roots),repeats)
    return values

def select(out):
    verify(out)
    if (out/'selection.json').exists():raise ValueError('Selection protected')
    search=read(out/'search.json');assert sha(out/'candidate_pool.npz')==search['candidate_pool_sha256']
    engine=old.Kernel();assert frozen_hash(engine)==search['frozen_sha256'];ledger=Ledger(out,True)
    context=dict(np.load(phase.PREVIOUS/'contexts.npz'));pool=dict(np.load(out/'candidate_pool.npz'))
    scores={};(out/'selection').mkdir()
    for name,theta in sorted(pool.items()):
        values=full_returns(engine,theta,context['train_roots'],context['train_indices'],16,CONFIG['selection_seed'],ledger,
            'selection/'+name,out/'selection'/f'{name}.npz');scores[name]=float(values.mean())
    winners={str(s):min((n for n in scores if n.startswith(f's{s}_')),key=lambda n:(scores[n],n)) for s in [0,1]}
    write(out/'selection.json',dict(utc=average.timestamp(),scores=scores,winners=winners,
        rule='minimum fresh full-rollout mean on training roots; lexicographic ties',candidate_pool_sha256=sha(out/'candidate_pool.npz')))
    print('Selection sealed:',winners,flush=True)

def evaluate(out):
    verify(out)
    if (out/'evaluation_started.json').exists():raise ValueError('Existing evaluation protected')
    selection=read(out/'selection.json');pool=dict(np.load(out/'candidate_pool.npz'));search=read(out/'search.json')
    assert sha(out/'candidate_pool.npz')==selection['candidate_pool_sha256']
    assert sha(out/'pre_final.npz')==search['pre_final_sha256']
    pool.update(dict(np.load(out/'pre_final.npz')))
    names=sorted({f's{s}_k{k}_u{u}' for s in [0,1] for k in range(3) for u in [0,23,24]}|set(selection['winners'].values()))
    write(out/'evaluation_started.json',dict(utc=average.timestamp(),selection_sha256=sha(out/'selection.json'),models=names))
    engine=old.Kernel();assert frozen_hash(engine)==search['frozen_sha256'];ledger=Ledger(out,True)
    context=dict(np.load(phase.PREVIOUS/'contexts.npz'));final=dict(np.load(average.PREVIOUS/'final_contexts.npz'))
    values={};reset={};constraints={};(out/'evaluation').mkdir()
    for name in names:
        theta=pool[name];seed=int(name[1]);base=pool[f's{seed}_k0_u0']
        np.testing.assert_array_equal(theta[:16],base[:16])
        values[name]=full_returns(engine,theta,final['roots'],final['indices'],64,CONFIG['final_seed'],ledger,
            'evaluation/'+name,out/'evaluation'/f'{name}.npz')
        ledger.add(128*50,'evaluation/reset/'+name)
        r=engine.rollout(theta,np.tile(START,(128,1)),jax.random.PRNGKey(CONFIG['reset_seed']),50)
        np.savez_compressed(out/'evaluation'/f'{name}_reset.npz',**r)
        reset[name]=.05*r['reward']@.95**np.arange(50)
        constraints[name]=evaluation.constraints(engine,theta,context,ledger)
        print(name+': independent full-kernel evaluation and constraints complete.',flush=True)
    assert frozen_hash(engine)==search['frozen_sha256'];verify(out)
    np.savez_compressed(out/'evaluation_returns.npz',**values,**{k+'_reset':v for k,v in reset.items()})
    write(out/'completion.json',dict(utc=average.timestamp(),computed_transitions=ledger.data['charged'],cap=CONFIG['model_cap'],
        response_updates=144,critic_steps=47400,actor_updates=0,nominal_updates=0,native_steps=0,
        frozen_verified=True,diagonal_bitwise_equal=True,constraints=constraints))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=['prepare','search','select','evaluate'])
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();globals()[args.phase](args.out)
