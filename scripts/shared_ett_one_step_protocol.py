"""Opt-in reference protocol for the shared ETT design. No rollout or death model.

prepare/check are deterministic; run explicitly trains NEW isolated generators.
The run phase is supplied for reproducibility, not executed by the design audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

OLD = Path('artifacts/ett_rollout_return/residual6_s01/config.json')
CONFIG = dict(seeds=[0, 1], hidden=32, response_width=8, bound=1.,
    warmup_updates=128, joint_updates=256, directions=4, batch=16, draws=4,
    sigma=.02, learning_rate=.05, update_cap=.1, weights=[0., .01],
    evaluation_draws=64, bootstrap=2000, model_output_cap=1500000,
    seed_base=73000000, diagonal_noninferiority=.02,
    minimum_reward_decrease=.02, constraint_tolerance=2e-6,
    region=[5.5, 5.75, 3.125, 3.875], final_iterate_only=True,
    execution_distribution='uniform [-1,1]^2', natural_action='recorded action',
    context_distribution='uniform eligible episode, then uniform eligible row',
    objective='emitted XY energy score + weight * expected one-step task reward')
SHAPES = [(20,32),(32,),(32,32),(32,),(32,11),(11,),(8,2),(2,8)]
SIZE = sum(int(np.prod(s)) for s in SHAPES)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v))
    Path(p).write_text(json.dumps(x, indent=2, default=convert)+'\n', encoding='utf-8', newline='\n')


def prepare(out):
    if out.exists(): raise ValueError('use a fresh output directory')
    old = json.loads(OLD.read_text())
    hashes = {p: digest(p) for p in old['paths'].values()}
    for p, h in hashes.items():
        assert old['input_file_sha256'][p] == h, p
    with np.load(old['paths']['dataset'], allow_pickle=False) as z:
        obs, act = z['obs'], z['act']  # No labels, masks, or hidden arrays.
    assert obs.shape == (6600,51,16)
    assert np.array_equal(obs[:,1:,2:8], obs[:,:-1,:6])
    goal = np.tile([8.5,3.5],4)
    assert np.all(obs[:,:,8:] == goal)
    held = np.random.default_rng(0).permutation(6600)[:660]
    source = np.arange(1200,6000)
    parts = {}; summary = {}
    for name, episodes in [('train',np.setdiff1d(source,held)), ('test',np.intersect1d(source,held))]:
        q = obs[episodes,:50,:2]
        mask = ((q[...,0]>=5.5)&(q[...,0]<=5.75)&(q[...,1]>=3.125)&(q[...,1]<=3.875))
        ei, t = np.where(mask); ep = episodes[ei]
        s, a, y = obs[ep,t,:8], act[ep,t], obs[ep,t+1,:8]
        lo=np.maximum(s[:,:2]-1, [0,3]);hi=np.minimum(s[:,:2]+1,[9,np.nextafter(np.float32(4),np.float32(-np.inf))])
        assert np.all(y[:,:2]>=lo-2e-6) and np.all(y[:,:2]<=hi+2e-6)
        parts.update({name+'_'+k:v for k,v in dict(state=s,action=a,target=y,episode=ep,time=t).items()})
        reward=np.linalg.norm(y[:,:2]-[8.5,3.5],axis=-1)<2
        summary[name]=dict(episodes=len(np.unique(ep)),rows=len(ep),recorded_reward_one=int(reward.sum()),
                          recorded_reward_zero=int((~reward).sum()))
    assert summary['train']['episodes']>=100 and summary['test']['episodes']>=20
    out.mkdir(parents=True)
    write(out/'config.json', CONFIG)
    np.savez_compressed(out/'contexts.npz', **parts)
    sources=['scripts/shared_ett_one_step_protocol.py','ett/rollout_return.py','crl/envs.py',str(OLD)]
    hashes.update({p:digest(p) for p in sources})
    write(out/'preparation.json', dict(starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        input_sha256=hashes,config_sha256=digest(out/'config.json'),contexts_sha256=digest(out/'contexts.npz'),
        support=summary,parameter_count=SIZE,training_run=False,new_model_outputs=0,new_native_transitions=0,
        original_diagonal='archived unchanged; new generator fits observations, not its sampled approximation',
        holdout='historically inspected source episode holdout; independent of new fitting, not an untouched test'))
    print(json.dumps(summary,indent=2))


def checked_inputs(out):
    p=json.loads((out/'preparation.json').read_text())
    for f,h in p['input_sha256'].items(): assert digest(f)==h,f
    assert digest(out/'config.json')==p['config_sha256']
    assert digest(out/'contexts.npz')==p['contexts_sha256']
    assert json.loads((out/'config.json').read_text())==CONFIG
    return dict(np.load(out/'contexts.npz',allow_pickle=False))


def generator(theta,s,x,xp,z,v,mean,std):
    import jax.numpy as j
    blocks=[];offset=0
    for shape in SHAPES:
        n=int(np.prod(shape));blocks.append(theta[offset:offset+n].reshape(shape));offset+=n
    w1,b1,w2,b2,wo,bo,ur,vr=blocks
    g=j.broadcast_to(j.tile(j.array([8.5,3.5]),4),s.shape)
    context=(j.concatenate([s,xp,g],axis=-1)-mean)/std
    context=j.broadcast_to(context[:,None],z.shape[:2]+(18,))
    h=j.tanh(j.concatenate([context,z],axis=-1)@w1+b1)
    h=j.tanh(h@w2+b2);heads=h@wo+bo
    atom=v<1/(1+j.exp(-heads[...,0]))
    displacement=j.where(atom[...,None],0.,2*j.tanh(heads[...,1:3]))
    k=heads[...,3:11]
    u=ur/j.maximum(1.,j.linalg.norm(ur));vmat=vr/j.maximum(1.,j.linalg.norm(vr))
    d=(x-xp)@u.T
    response=(j.maximum(d[:,None]+k,0)-j.maximum(k,0))@vmat.T
    q=s[:,None,:2]
    lo=j.maximum(q-1.,j.array([0.,3.]))
    hi=j.minimum(q+1.,j.array([9.,float(np.nextafter(np.float32(4),np.float32(-np.inf)))]))
    xy=j.clip(q+displacement+response,lo,hi)
    return j.concatenate([xy,j.broadcast_to(s[:,None,:6],xy.shape[:2]+(6,))],axis=-1)


def check(out):
    """Deterministic full-function witness, controls, and implementation checks."""
    import jax.numpy as j
    data=checked_inputs(out)
    theta=np.zeros(SIZE,np.float32)
    offsets=np.cumsum([0]+[int(np.prod(s)) for s in SHAPES])
    # A=.875 to the right; fixed uniforms select the moving branch. No fitted target.
    theta[offsets[5]+1]=np.arctanh(.875/2)
    u=np.zeros((8,2),np.float32);u[0,0]=1/np.sqrt(2);u[1,0]=-1/np.sqrt(2)
    v=np.zeros((2,8),np.float32);v[0,:2]=-1/np.sqrt(2)
    theta[offsets[6]:offsets[7]]=u.ravel();theta[offsets[7]:]=v.ravel()
    actions=np.array([(a,b) for a in [-1.,-.5,0.,.5,1.] for b in [-1.,0.,1.]],np.float32)
    s=np.tile([5.75,3.5],(len(actions),4)).astype(np.float32)
    xp=np.zeros_like(actions);z=np.zeros((len(s),1,2),np.float32);uniform=np.full((len(s),1),.75,np.float32)
    y=np.asarray(generator(j.array(theta),s,actions,xp,z,uniform,np.zeros(18),np.ones(18)))
    expected_x=6.625-.5*np.abs(actions[:,0])
    np.testing.assert_allclose(y[:,0,0],expected_x,atol=2e-6,rtol=0)
    np.testing.assert_array_equal(y[:,0,2:],s[:,:6])
    excess=np.linalg.norm(y[:,None,0,:2]-y[None,:,0,:2],axis=-1)-np.linalg.norm(actions[:,None]-actions[None],axis=-1)
    assert excess.max()<=2e-6
    assert (np.linalg.norm(y[7,0,:2]-[8.5,3.5])<2) # x=(0,0)
    assert not (np.linalg.norm(y[10,0,:2]-[8.5,3.5])<2) # x=(.5,0)
    # Geometry controls: all-zero at X=3.5, all-one at X=8, for every legal output.
    for q,expected in [([3.5,3.5],False),([8.,3.5],True)]:
        ss=np.tile(q,(len(actions),4)).astype(np.float32)
        yy=np.asarray(generator(j.array(theta),ss,actions,xp,z,uniform,np.zeros(18),np.ones(18)))
        assert np.all((np.linalg.norm(yy[:,0,:2]-[8.5,3.5],axis=-1)<2)==expected)
    write(out/'design_checks.json',dict(training_run=False,stochastic_draws=0,
        deterministic_generator_outputs=45,witness='R_X=-abs(x_X-x_prime_X)/2; diagonal reward 1, off reward 0 at d_X=.5',
        continuum_proof='fixed convex projection; ||V||op ||U||op <= 1; ReLU nonexpansive',
        finite_pair_max_excess=float(excess.max()),controls_pass=True,F4_pass=True,
        parameter_count=SIZE,training_runtime_unexecuted=True))
    print('Deterministic design checks passed; training was not run.')


def run(out):
    """Explicit opt-in future experiment. Final iterates; no trajectories."""
    import jax
    import jax.numpy as j
    from ett.convex_action_transition import energy_score
    from ett.rollout_return import antithetic_gradient,descent_step
    data=checked_inputs(out)
    if (out/'RUN_STARTED.json').exists():raise ValueError('preserve previous attempt; use a fresh directory')
    write(out/'RUN_STARTED.json',dict(config=CONFIG,status='started'))
    ep=data['train_episode'];unique=np.unique(ep);groups=[np.flatnonzero(ep==e) for e in unique]
    rowweight=np.array([1/(len(unique)*np.sum(ep==e)) for e in ep])
    c=np.concatenate([data['train_state'],data['train_action'],np.tile(np.tile([8.5,3.5],4),(len(ep),1))],axis=1)
    mean=(c*rowweight[:,None]).sum(0);std=np.sqrt(((c-mean)**2*rowweight[:,None]).sum(0));std=np.maximum(std,.001)
    np.savez(out/'normalization.npz',mean=mean,std=std)
    count=0
    def add(n):
        nonlocal count
        count+=n
        if count>CONFIG['model_output_cap']:raise RuntimeError('predeclared model-output cap exceeded')
    @jax.jit
    def terms(theta,s,a,target,x,z,v):
        diag=generator(theta,s,a,a,z,v,mean,std)
        off=generator(theta,s,x,a,z,v,mean,std)
        return energy_score(diag[...,:2],target[...,:2]).mean(),(j.linalg.norm(off[...,:2]-j.array([8.5,3.5]),axis=-1)<2).mean()
    def batch(seed,iteration,phase):
        rng=np.random.default_rng(CONFIG['seed_base']+phase+seed*100000+iteration)
        idx=np.array([rng.choice(groups[i]) for i in rng.integers(len(groups),size=16)])
        return (data['train_state'][idx],data['train_action'][idx],data['train_target'][idx],
                rng.uniform(-1,1,(16,2)).astype(np.float32),rng.normal(size=(16,4,2)).astype(np.float32),
                rng.uniform(size=(16,4)).astype(np.float32))
    checkpoints={};history=[]
    for seed in CONFIG['seeds']:
        rng=np.random.default_rng(CONFIG['seed_base']+seed)
        theta=np.concatenate([rng.normal(0,.1,int(np.prod(shape))) if len(shape)>1 else np.zeros(shape) for shape in SHAPES]).astype(np.float32)
        checkpoints[f's{seed}_initial']=theta.copy()
        # Matched streams across arms; seed/phase namespaces separate warmup and joint.
        def optimize(start,weight,updates,phase,label):
            t=start.copy()
            for it in range(updates):
                args=batch(seed,it,phase)
                dr=np.random.default_rng(CONFIG['seed_base']+5000000+phase+seed*100000+it)
                directions=dr.normal(size=(4,SIZE));plus=[];minus=[];query_terms=[]
                for direction in directions:
                    values=[]
                    for sign in [1,-1]:
                        di,off=map(float,terms(j.asarray(t+sign*.02*direction,dtype=j.float32),*args));add(128)
                        query_terms.append([di,off])
                        values.append(di+weight*off)
                    plus.append(values[0]);minus.append(values[1])
                grad=antithetic_gradient(plus,minus,directions,.02,1.)
                t=np.asarray(descent_step(t,grad,.05,.1),np.float32)
                if not np.isfinite(t).all():raise ValueError('nonfinite iterate; preserve attempt')
                history.append(dict(seed=seed,arm=label,update=it,positive=plus,negative=minus,
                    diagonal_and_reward_by_signed_query=query_terms,parameter_norm=float(np.linalg.norm(t))))
            return t
        warm=optimize(theta,0.,128,1000000,'warmup');checkpoints[f's{seed}_warm']=warm.copy()
        for name,weight in [('diagonal',0.),('joint',.01)]:
            checkpoints[f's{seed}_{name}']=optimize(warm,weight,256,2000000,name)
        # Preserve finished seed before starting the next one.
        for name,t in checkpoints.items():np.savez(out/(name+'.npz'),theta=t,model='shared_one_step_v1',bound=1.)
        write(out/'training_history.json',history)
    # Independent noise, paired across all checkpoints; no checkpoint selection.
    rng=np.random.default_rng(CONFIG['seed_base']+9000000)
    s,a,target=[data['test_'+f] for f in ['state','action','target']];n=len(s)
    x=rng.uniform(-1,1,(n,2)).astype(np.float32)
    z=rng.normal(size=(n,64,2)).astype(np.float32);v=rng.uniform(size=(n,64)).astype(np.float32)
    saved=dict(state=s,natural_action=a,execution_action=x,noise=z,uniform=v,episode=data['test_episode'],target=target)
    for name,t in checkpoints.items():
        diag=np.asarray(generator(j.asarray(t),s,a,a,z,v,mean,std));off=np.asarray(generator(j.asarray(t),s,x,a,z,v,mean,std));add(2*n*64)
        saved[name+'_diagonal']=diag;saved[name+'_off']=off
        saved[name+'_energy']=np.asarray(energy_score(j.asarray(diag[...,:2]),j.asarray(target[...,:2])))
    np.savez_compressed(out/'independent_evaluation.npz',**saved)
    checked_inputs(out)
    write(out/'transition_ledger.json',dict(model_outputs=count,cap=CONFIG['model_output_cap'],native_transitions=0,
        completed=True,acceptance='apply separate fit, continuous-proof/finite-constraint and paired-pessimism criteria in PROTOCOL.md; completion is not a passing result'))
    print('Future experiment completed; apply the predeclared independent acceptance checks.')


def assess(out):
    """Independent saved-output assessment; optional fixed component checks only."""
    import jax.numpy as j
    data=checked_inputs(out)
    with np.load(out/'independent_evaluation.npz') as f: e=dict(f)
    with np.load(out/'normalization.npz') as f: norm=dict(f)
    ep=e['episode'];ids=np.unique(ep)
    average=lambda v:np.array([v[ep==i].mean() for i in ids])
    rng=np.random.default_rng(CONFIG['seed_base']+10000000)
    draws=rng.integers(len(ids),size=(2000,len(ids)))
    def interval(rows):
        values=average(rows)
        return dict(mean=float(values.mean()),ci95=np.quantile(values[draws].mean(1),[.025,.975]))
    reward=lambda y:(np.linalg.norm(y[...,:2]-[8.5,3.5],axis=-1)<2).mean(1)
    summary={};count=0;violations=0;max_excess=0.
    for seed in CONFIG['seeds']:
        control=f's{seed}_diagonal';joint=f's{seed}_joint';initial=f's{seed}_initial'
        drop=interval(reward(e[joint+'_off'])-reward(e[control+'_off']))
        response=interval((reward(e[joint+'_off'])-reward(e[joint+'_diagonal']))-
                          (reward(e[control+'_off'])-reward(e[control+'_diagonal'])))
        fit=interval(e[joint+'_energy']-e[control+'_energy'])
        learned=interval(e[control+'_energy']-e[initial+'_energy'])
        summary[str(seed)]=dict(diagonal_joint_minus_control=fit,control_fit_minus_initial=learned,
            off_reward_joint_minus_control=drop,response_contrast=response,
            diagonal_fit_pass=bool(fit['ci95'][1]<=.02 and learned['ci95'][1]<0),
            pessimistic_selection_pass=bool(drop['ci95'][1]<-.02 and response['ci95'][1]<0))
    # Same fixed conditioning, noise and uniform for every compared action.
    s=e['state'];a=e['natural_action'];z=e['noise'][:,:4];v=e['uniform'][:,:4]
    grid=[np.tile([x,y],(len(s),1)).astype(np.float32) for x in [-1,0,1] for y in [-1,0,1]]
    grid += [a,e['execution_action'],np.clip(e['execution_action']+1e-4,-1,1)]
    for seed in CONFIG['seeds']:
        for arm in ['initial','warm','diagonal','joint']:
            with np.load(out/f's{seed}_{arm}.npz') as f: theta=f['theta']
            outputs=[]
            for x in grid:
                yy=np.asarray(generator(j.asarray(theta),s,x,a,z,v,norm['mean'],norm['std']));count+=len(s)*4
                np.testing.assert_array_equal(yy[...,2:],np.broadcast_to(s[:,None,:6],yy[...,2:].shape))
                lo=np.maximum(s[:,None,:2]-1,[0,3]);hi=np.minimum(s[:,None,:2]+1,[9,np.nextafter(np.float32(4),np.float32(-np.inf))])
                violations+=int(np.sum(np.any((yy[...,:2]<lo-2e-6)|(yy[...,:2]>hi+2e-6),axis=-1)))
                assert np.isfinite(yy).all();outputs.append(yy)
            for i in range(len(grid)):
                for k in range(i):
                    excess=np.linalg.norm(outputs[i][...,:2]-outputs[k][...,:2],axis=-1)-np.linalg.norm(grid[i]-grid[k],axis=-1)[:,None]
                    max_excess=max(max_excess,float(excess.max()));violations+=int(np.sum(excess>2e-6))
    ledger=json.loads((out/'transition_ledger.json').read_text())
    assert ledger['model_outputs']+count+45<=CONFIG['model_output_cap']
    write(out/'assessment.json',dict(seeds=summary,geometry_or_action_pair_violations=violations,
        maximum_action_pair_excess=max_excess,total_outputs_including_assessment_and_design_check=ledger['model_outputs']+count+45,
        all_tests_pass=bool(violations==0 and all(v['diagonal_fit_pass'] and v['pessimistic_selection_pass'] for v in summary.values())),
        bootstrap_unit='source episode; optimizer seeds separate; paired generator noise',
        death_rate=None,native_or_downstream_result=None))
    print('Independent assessment saved; finite checks do not replace the analytic proof.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','check','run','assess'])
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();globals()[args.phase](args.out_dir)
