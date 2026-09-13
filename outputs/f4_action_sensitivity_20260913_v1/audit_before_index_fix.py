"""Bounded read-only current F4 sensitivity audit. Run from repository root."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import copy
import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
sys.path.insert(0, str(ROOT))
from crl.envs import TwoRouteSwampWindyF4Env
from ett.action_reference import snapshot, restore, natural_action
from scripts.collect_swamp_windy import make_windy_teacher

EPS = [.1, .03, .01, .003, .001]
STRATA = ['free', 'wall_near', 'pre_hazard', 'hazard_interior', 'hazard_boundary', 'dead', 'alive_waiting']
CAPS = dict(collection=5600, setup=1024, pairs=9600, validation=512, continuation=512,
            model=60000, jacobian=512)

def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    return v

def write(name, obj):
    (OUT/name).write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+'\n', encoding='utf-8')

def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

COUNTS = {k:0 for k in CAPS}
def charge(kind,n=1):
    if COUNTS[kind]+n>CAPS[kind]: raise RuntimeError(('budget exhausted',kind,COUNTS))
    COUNTS[kind]+=n

def progress(stage):
    write('costs.json',dict(stage=stage,counts=COUNTS,caps=CAPS,
          native_total=sum(COUNTS[k] for k in ['collection','setup','pairs','validation','continuation'])))
    print(stage,COUNTS,flush=True)

class Tape:
    def __init__(self,n,u): self.n=np.array(n); self.u=np.array(u); self.normal_calls=0; self.uniform_calls=0
    def normal(self,loc,scale,size):
        assert tuple(np.atleast_1d(size))==(2,)
        self.normal_calls+=1
        return loc+scale*self.n.copy()
    def random(self,size):
        assert size==3
        self.uniform_calls+=1
        return self.u.copy()

def step_record(rec,action,n,u,kind,instrument=True):
    env=restore(rec); tape=Tape(n,u); env._rng=tape
    signature=[]
    if instrument:
        original=env._is_blocked
        def blocked(p):
            b=original(p); signature.append(bool(b)); return b
        env._is_blocked=blocked
    charge(kind)
    obs,reward,done,_=env.step(action)
    assert not done
    assert np.array_equal(obs[2:8],np.asarray(rec['frames'],np.float32).reshape(-1)[:6])
    result=dict(obs=obs[:8],physical=env.state.copy(),dead=env.dead,bits=env.swamp_bits,
                reward=reward,signature=signature,normal_calls=tape.normal_calls,
                uniform_calls=tape.uniform_calls)
    # The artificial tape has no RNG state; use the original snapshot RNG for storage only.
    env._rng=np.random.default_rng(0)
    result['after']=snapshot(env)
    return result

def margin(s):
    xy=s[:2]; walls=TwoRouteSwampWindyF4Env.SWAMP_CELLS # geometry below is actual env walls
    del walls
    lo=np.argwhere(GEOMETRY==1)
    return float(np.linalg.norm(np.maximum(np.maximum(lo-xy,xy-(lo+1)),0),axis=1).min())

GEOMETRY=TwoRouteSwampWindyF4Env()._walls.copy()

def tags(row):
    p=row['snapshot']['state']; dead=row['snapshot']['absorbing']; x,y=p
    if dead:return ['dead']
    out=[]; hazard=3<=x<6 and 3<=y<4
    if margin(p)>.2 and not hazard:out.append('free')
    if margin(p)<.12:out.append('wall_near')
    if 2<=x<3 and 3<=y<4:out.append('pre_hazard')
    if hazard:out.append('hazard_interior')
    if (2.8<=x<=6.2 and 2.8<=y<=4.2 and
        (min(abs(x-k) for k in [3,4,5,6])<.15 or min(abs(y-3),abs(y-4))<.15)):
        out.append('hazard_boundary')
    if np.linalg.norm(row['action'])<.02:out.append('alive_waiting')
    return out

def collect():
    env=TwoRouteSwampWindyF4Env(seed=26091301,active_prob=.3,action_noise=.01)
    rng=np.random.default_rng(26091302); teacher=make_windy_teacher(env,rng,.05); rows=[]
    for ep in range(112):
        env.reset(); memo={}
        for t in range(50):
            rec=snapshot(env)
            action=natural_action(teacher,env,rng,memo)[0] if ep<80 else rng.uniform(-1,1,2)
            if t<=45:
                rows.append(dict(snapshot=rec,action=action,episode=ep,time=t,
                                 behavior='teacher' if ep<80 else 'uniform'))
            charge('collection'); env.step(action)
    selected=[]; selection={}; rng=np.random.default_rng(26091303)
    all_tags=[tags(r) for r in rows]
    for label in STRATA:
        ids=[i for i,t in enumerate(all_tags) if label in t]
        chosen=rng.choice(ids,min(24,len(ids)),replace=False) if ids else []
        selection[label]=dict(eligible=len(ids),selected=len(chosen))
        for idx in chosen:
            r=copy.deepcopy(rows[int(idx)]); r.update(stratum=label,source='natural',id=len(selected))
            selected.append(r)
    write('selection.json',selection)
    return selected

def target_contexts(start_id):
    settings=[('free',[.5,3.5],[.4,0],False),
              ('upper_wall',[.5,3.95],[.4,.5],False),
              ('corner',[1.95,2.95],[.8,.5],False),
              ('pre_hazard',[2.95,3.5],[.05,0],False),
              ('hazard_interior',[3.5,3.5],[.3,0],False),
              ('entry_boundary',[2.5,3.5],[.5,0],False),
              ('dead',[3.5,3.5],[.3,0],True),
              ('alive_waiting',[2.5,3.5],[0,0],False),
              ('action_clipping',[.5,3.5],[1,1],False)]
    rows=[]
    for label,dest,center,make_dead in settings:
        env=TwoRouteSwampWindyF4Env(seed=26091310,active_prob=.3,action_noise=.01)
        # Reset mask is an explicit controlled setup intervention; all subsequent positions arise by env.step.
        env.set_swamp([False]*3); t=0
        waypoints=[[1.5,3.5],[1.5,2.95],dest] if label=='corner' else [dest]
        for waypoint in waypoints:
            while np.linalg.norm(env.state-waypoint)>1e-10:
                action=np.clip(np.array(waypoint)-env.state,-1,1)
                final=np.linalg.norm(env.state+action-waypoint)<1e-10 and waypoint==waypoints[-1]
                # If requested, activate mask on the preceding reachable step, before entry.
                u=np.array([.1,.1,.1]) if make_dead and env.state[0]>=1.5 else np.array([.9]*3)
                if final and not make_dead:u=np.array([.1,.9,.9])
                env._rng=Tape([0,0],u); charge('setup'); env.step(action); t+=1
                if t>12:raise RuntimeError(('unreachable target',label,env.state))
        assert env.dead==make_dead,(label,env.dead)
        env._rng=np.random.default_rng(26091311)
        for k in range(8):
            rows.append(dict(snapshot=snapshot(env),action=np.array(center),episode=f'target_{label}',
                             time=t,behavior='controlled_native_prefix',stratum=label,source='targeted',
                             id=start_id+len(rows),target_tape=k))
    return rows

def directions(rng):
    d=rng.normal(size=(2,2));d/=np.linalg.norm(d,axis=1)[:,None]
    return np.concatenate([np.eye(2),d])

def stats(v):
    v=np.array(v,float)
    if not len(v):return dict(count=0)
    return dict(count=len(v),**{k:float(np.percentile(v,p)) for k,p in
        [('median',50),('p90',90),('p95',95),('p99',99),('p99_9',99.9)]},
        max=float(v.max()),**{f'above_{b}':float(np.mean(v>b+.002)) for b in [1,1.05,1.25,2,5]})

def native_pairs(contexts):
    results=[]; continuation_candidates=[]; validation=[]
    for row in contexts:
        rng=np.random.default_rng((26091304 if row['source']=='natural' else 26091305)+row['id'])
        normal=rng.normal(size=2);uniform=rng.random(3);ds=directions(rng)
        center=np.array(row['action'],float)
        if row['source']=='targeted' and row['stratum']!='action_clipping':center-=.01*normal
        first_switch=None
        for eps in EPS:
            for di,d in enumerate(ds):
                aa=np.clip(center-eps*d/2,-1,1);bb=np.clip(center+eps*d/2,-1,1)
                ae=np.clip(aa+.01*normal,-1,1);be=np.clip(bb+.01*normal,-1,1)
                den=np.linalg.norm(be-ae);requested_den=np.linalg.norm(bb-aa)
                a=step_record(row['snapshot'],aa,normal,uniform,'pairs')
                b=step_record(row['snapshot'],bb,normal,uniform,'pairs')
                assert np.array_equal(a['bits'],b['bits'])
                dv=a['obs'].astype(float)-b['obs'].astype(float)
                np.testing.assert_array_equal(dv[2:],np.zeros(6))
                assert np.linalg.norm(dv)==np.linalg.norm(dv[:2])
                delta64=np.linalg.norm(a['physical']-b['physical']);delta32=np.linalg.norm(dv)
                switched=a['dead']!=b['dead']
                r=dict(context=row['id'],source=row['source'],stratum=row['stratum'],episode=row['episode'],
                       time=row['time'],epsilon=eps,direction=di,action_a=aa,action_b=bb,effective_a=ae,effective_b=be,
                       denominator=den,requested_denominator=requested_den,zero_denominator=den==0,
                       ratio=None if den==0 else delta32/den,physical_ratio=None if den==0 else delta64/den,
                       requested_ratio=None if requested_den==0 else delta64/requested_den,
                       roundoff_allowance=None if den==0 else 2e-6/den,
                       visible_physical_difference=abs(delta64-delta32),death_switch=switched,
                       collision_change=a['signature']!=b['signature'],
                       pre_state=row['snapshot']['state'],pre_f4=np.array(row['snapshot']['frames']).reshape(-1),
                       pre_dead=row['snapshot']['absorbing'],pre_bits=row['snapshot']['bits'],
                       normal=normal,uniform=uniform,after_a={k:v for k,v in a.items() if k!='after'},
                       after_b={k:v for k,v in b.items() if k!='after'})
                results.append(r)
                if switched and den>0 and row['time']<=46 and first_switch is None:
                    first_switch=(r,a['after'],b['after'])
                if eps==.1 and di==0 and len(validation)<32:
                    same=step_record(row['snapshot'],aa,normal,uniform,'validation')
                    plain=step_record(row['snapshot'],aa,normal,uniform,'validation',False)
                    for other in [same,plain]:
                        assert np.array_equal(a['obs'],other['obs']) and a['dead']==other['dead']
                        assert np.array_equal(a['physical'],other['physical'])
                    validation.append(dict(id=row['id'],identical_key_replay=True,instrumentation_exact=True))
        if first_switch is not None and len(continuation_candidates)<64:
            continuation_candidates.append(first_switch)
    # Native stored RNG replay and active-noise tests are explicit extra calls.
    original=contexts[0]['snapshot']; e1=restore(original);e2=restore(original)
    charge('validation',2);o1=e1.step([.2,.1]);o2=e2.step([.2,.1]);assert np.array_equal(o1[0],o2[0])
    free=next(c for c in contexts if c['source']=='targeted' and c['stratum']=='free')
    z=step_record(free['snapshot'],[.2,0],[0,0],[.9]*3,'validation')
    n=step_record(free['snapshot'],[.2,0],[1,0],[.9]*3,'validation')
    np.testing.assert_allclose(n['physical']-z['physical'],[.01,0],atol=1e-12)
    write('coupling_checks.json',dict(repeats=validation,original_rng_replay=True,active_noise_displacement=.01,
                                    same_post_masks_all_pairs=True,history_shift_all_pairs=True))
    with (OUT/'native_pairs.pkl').open('wb') as f:pickle.dump(results,f)
    write('native_witnesses.json',sorted([r for r in results if r['ratio'] is not None],
                                      key=lambda r:r['physical_ratio'],reverse=True)[:30])
    summaries=[]
    for source,label in dict.fromkeys((r['source'],r['stratum']) for r in results):
        for eps in EPS:
            group=[r for r in results if (r['source'],r['stratum'],r['epsilon'])==(source,label,eps)]
            for event in ['all','preserving','death_switching','collision_changed','collision_unchanged']:
                sub=[r for r in group if event=='all' or
                     (event=='preserving' and not r['death_switch']) or
                     (event=='death_switching' and r['death_switch']) or
                     (event=='collision_changed' and r['collision_change']) or
                     (event=='collision_unchanged' and not r['collision_change'])]
                if not sub:continue
                summaries.append(dict(source=source,stratum=label,epsilon=eps,event=event,
                                      zero_denominators=sum(r['zero_denominator'] for r in sub),
                                      visible=stats([r['ratio'] for r in sub if r['ratio'] is not None]),
                                      physical=stats([r['physical_ratio'] for r in sub if r['ratio'] is not None])))
    write('native_summary.json',summaries)
    return continuation_candidates

def continuations(candidates):
    results=[]
    for i,(pair,ra,rb) in enumerate(candidates):
        sep=[dict(step=1,f4_ratio=pair['ratio'],xy_ratio=pair['ratio'])]
        rng=np.random.default_rng(26091306+i)
        for k in range(2,5):
            n=rng.normal(size=2);u=rng.random(3)
            a=step_record(ra,[1,0],n,u,'continuation');b=step_record(rb,[1,0],n,u,'continuation')
            if k in [2,4]:
                dv=a['obs'].astype(float)-b['obs'].astype(float)
                sep.append(dict(step=k,f4_ratio=np.linalg.norm(dv)/pair['denominator'],
                                xy_ratio=np.linalg.norm(dv[:2])/pair['denominator']))
            ra=a['after'];rb=b['after']
        results.append(dict(context=pair['context'],source=pair['source'],stratum=pair['stratum'],
                            epsilon=pair['epsilon'],direction=pair['direction'],ratios=sep))
    write('continuation.json',results)

def model_panel(contexts):
    # Round-robin natural strata first, then targeted strata, independent of paired outcomes.
    groups={}
    for r in contexts:groups.setdefault((r['source'],r['stratum']),[]).append(r)
    panel=[]
    for i in range(24):
        for rs in groups.values():
            if i<len(rs):panel.append(rs[i])
            if len(panel)==64:return panel
    return panel

def learned(contexts):
    import jax
    import jax.numpy as j
    from ett.diagonal_transition import load_diagonal_transition
    from ett.pointmaze_norm_comparison import ModeKernel, matrices_mode
    from ett.convex_action_transition import gates
    goal=np.tile([8.5,3.5],4).astype(np.float32)
    states=np.array([np.array(r['snapshot']['frames']).reshape(-1) for r in contexts],np.float32)
    actions=np.array([r['action'] for r in contexts],np.float32)
    detpath=ROOT/'artifacts/ett_diagonal/f4_p30_expert_deterministic_s0'
    det=load_diagonal_transition(str(detpath))
    def f(s,a):
        ctx=j.concatenate([s,a,a,j.asarray(goal)])
        return s[:2]+det._predict(det.params,ctx[None])[0]
    jf=jax.jit(jax.vmap(jax.jacfwd(f,1)))
    count=min(256,len(states));charge('jacobian',count);charge('model',count)
    jac=np.asarray(jf(j.asarray(states[:count]),j.asarray(actions[:count])))
    norms=np.linalg.svd(jac,compute_uv=False)[:,0]
    fd=[];batch_f=jax.jit(jax.vmap(f))
    for h in [.003,.001]:
        for axis in range(2):
            ap=actions[:64].copy();am=ap.copy();ap[:,axis]=np.minimum(1,ap[:,axis]+h);am[:,axis]=np.maximum(-1,am[:,axis]-h)
            charge('model',128)
            values=(np.asarray(batch_f(states[:64],ap))-np.asarray(batch_f(states[:64],am)))/(ap[:,axis]-am[:,axis])[:,None]
            fd.append(dict(h=h,axis=axis,column_error=stats(np.linalg.norm(values-jac[:64,:,axis],axis=1))))
    write('deterministic.json',dict(object='raw unprojected next-XY deterministic prediction; both action slots varied',
          jacobian=stats(norms),by_stratum={str(k):stats([norms[i] for i,r in enumerate(contexts[:count]) if (r['source'],r['stratum'])==k])
            for k in dict.fromkeys((r['source'],r['stratum']) for r in contexts[:count])},finite_difference=fd))
    np.savez_compressed(OUT/'jacobians.npz',states=states[:count],actions=actions[:count],jacobian=jac,norm=norms)
    progress('deterministic_complete')
    panel=model_panel(contexts);ss=np.array([np.array(r['snapshot']['frames']).reshape(-1) for r in panel],np.float32)
    modekernels={mode:ModeKernel(mode) for mode in ['frobenius','spectral']}
    xp=np.asarray(modekernels['frobenius'].nominal.sample(ss,jax.random.PRNGKey(26091320),1,goal=np.tile(goal,(len(ss),1))))
    rng=np.random.default_rng(26091321)
    dd=np.stack([directions(rng) for _ in panel])
    indexes=np.repeat(np.arange(len(panel)),20);es=np.tile(np.repeat(EPS,4),len(panel));dirs=np.tile(np.arange(4),len(panel))
    s=ss[indexes];xps=xp[indexes];delta=es[:,None]*dd[indexes,dirs]/2
    np.savez_compressed(OUT/'model_panel.npz',state=ss,x_prime=xp,context_ids=[r['id'] for r in panel])
    params=np.load(ROOT/'outputs/ett_frobenius_spectral_v1/final_parameters.npz')
    summary=[];records={};matrix_rows=[]
    for name in params.files:
        theta=params[name]
        for mode,kernel in modekernels.items():
            effective=np.einsum('bj,jkl->bkl',np.asarray(gates(ss)),np.asarray(matrices_mode(theta[16:],mode)))
            raw=theta[16:].reshape(8,2,2);rawop=np.linalg.svd(raw,compute_uv=False)[:,0];rawfro=np.linalg.norm(raw,axis=(1,2))
            matrix_rows.append(dict(parameters=name,emitter=mode,raw_operator=rawop,raw_frobenius=rawfro,
                                   normalization_active=int(np.sum((rawfro if mode=='frobenius' else rawop)>1)),
                                   effective_operator=stats(np.linalg.svd(effective,compute_uv=False)[:,0]),
                                   effective_frobenius=stats(np.linalg.norm(effective,axis=(1,2)))))
            for condition in ['diagonal_center_fixed_xprime','off_diagonal_fixed_xprime']:
                centers=xps if condition.startswith('diagonal') else np.array([panel[i]['action'] for i in indexes])
                aa=np.clip(centers-delta,-1,1).astype(np.float32);bb=np.clip(centers+delta,-1,1).astype(np.float32)
                den=np.linalg.norm(bb.astype(float)-aa.astype(float),axis=1)
                key=jax.random.PRNGKey(26091322)
                charge('model',len(s)*2)
                ya,da=kernel.sample(j.asarray(theta),s,aa,xps,key,1)
                yb,db=kernel.sample(j.asarray(theta),s,bb,xps,key,1)
                ya,yb=np.asarray(ya)[:,0],np.asarray(yb)[:,0]
                assert np.array_equal(da['anchor_xy'],db['anchor_xy'])
                assert np.array_equal(da['stationary_atom'],db['stationary_atom'])
                assert np.array_equal(ya[:,2:],yb[:,2:]) and np.isfinite(ya).all() and np.isfinite(yb).all()
                valid=den>0;ratio=np.linalg.norm(ya.astype(float)-yb.astype(float),axis=1)[valid]/den[valid]
                code=name+'__'+mode+'__'+condition
                records[code+'__ratio']=ratio;records[code+'__epsilon']=es[valid]
                records[code+'__context']=indexes[valid]
                for eps in EPS:
                    keep=es[valid]==eps
                    summary.append(dict(parameters=name,emitter=mode,condition=condition,epsilon=eps,
                                        ratios=stats(ratio[keep]),zero_denominators=int(np.sum(~valid & (es==eps))),
                                        projection_frequency=float((np.asarray(da['projection_corrected'])[es==eps].mean()+np.asarray(db['projection_corrected'])[es==eps].mean())/2),
                                        atom_frequency=float(np.asarray(da['stationary_atom'])[es==eps].mean())))
                # Same complete batch/key replay plus diagonal identity on 8 fresh single-point queries.
                charge('model',16)
                qa,qd=kernel.sample(j.asarray(theta),s[:8],aa[:8],xps[:8],key,1)
                qb,qbd=kernel.sample(j.asarray(theta),s[:8],aa[:8],xps[:8],key,1)
                assert np.array_equal(qa,qb) and np.array_equal(qd['stationary_atom'],qbd['stationary_atom'])
            charge('model',8)
            yd,detail=kernel.sample(j.asarray(theta),ss[:8],xp[:8],xp[:8],jax.random.PRNGKey(26091323),1)
            np.testing.assert_array_equal(np.asarray(yd)[...,:2],np.asarray(detail['anchor_xy']))
        # Observational diagonal: original raw diagonal law; response vanishes in either norm mode.
        aa=np.clip(xps-delta,-1,1).astype(np.float32);bb=np.clip(xps+delta,-1,1).astype(np.float32)
        key=jax.random.PRNGKey(26091324);charge('model',2*len(s))
        ya,da=modekernels['frobenius'].sample(j.asarray(theta),s,aa,aa,key,1)
        yb,db=modekernels['frobenius'].sample(j.asarray(theta),s,bb,bb,key,1)
        den=np.linalg.norm(bb.astype(float)-aa.astype(float),axis=1);valid=den>0
        ratio=np.linalg.norm(np.asarray(ya)[:,0].astype(float)-np.asarray(yb)[:,0].astype(float),axis=1)[valid]/den[valid]
        code=name+'__observational_diagonal';records[code+'__ratio']=ratio;records[code+'__epsilon']=es[valid]
        records[code+'__context']=indexes[valid]
        for eps in EPS:
            summary.append(dict(parameters=name,emitter='diagonal',condition='both_action_slots_vary',epsilon=eps,
                 ratios=stats(ratio[es[valid]==eps]),zero_denominators=int(np.sum(~valid&(es==eps))),
                 atom_switch_fraction=float(np.mean(np.asarray(da['stationary_atom'])[es==eps]!=np.asarray(db['stationary_atom'])[es==eps]))))
        progress('ett_'+name)
    np.savez_compressed(OUT/'model_ratios.npz',**records)
    write('model_summary.json',summary);write('matrices.json',matrix_rows)
    write('model_checks.json',dict(identical_key_samples_equal=True,fixed_xprime_anchor_and_atom_equal=True,
                                 diagonal_identity=True,history_equal=True,parameter_sources=list(params.files)))

def main():
    assert not (OUT/'costs.json').exists(),'fresh run only'
    paths=['crl/envs.py','ett/action_reference.py','scripts/collect_swamp_windy.py',
           'ett/pointmaze_norm_comparison.py','ett/pointmaze_region_pilot.py','ett/convex_action_transition.py',
           'ett/diagonal_transition.py','propensity/nominal_policy.py',
           'outputs/ett_frobenius_spectral_v1/final_parameters.npz',
           'artifacts/ett_diagonal/f4_p30_expert_deterministic_s0/best.pkl',
           'artifacts/ett_diagonal/f4_p30_expert_deterministic_s0/config.json']
    cfg=json.loads((ROOT/'artifacts/ett_rollout_return/residual6_s01/config.json').read_text())
    paths+=list(cfg['paths'][k] for k in ['dataset','nominal','actor','control'])
    hashes={p:sha(ROOT/p) for p in paths}
    write('provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
          source_sha256=hashes,protocol_sha256=sha(OUT/'PROTOCOL.md'),audit_source_sha256=sha(__file__),
          environment=dict(active_prob=.3,action_noise=.01,horizon=50,discount=.95,dt=.1,substeps=10),
          python=sys.version,numpy=np.__version__,start=time.time()))
    progress('started')
    natural=collect();targets=target_contexts(len(natural));contexts=natural+targets
    with (OUT/'contexts.pkl').open('wb') as f:pickle.dump(contexts,f)
    write('contexts_index.json',[{k:v for k,v in r.items() if k!='snapshot'} for r in contexts])
    progress('contexts_complete')
    candidates=native_pairs(contexts);progress('native_pairs_complete')
    continuations(candidates);progress('continuation_complete')
    learned(contexts)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    progress('complete')
    write('completion.json',dict(status='complete',source_hashes_unchanged=True,costs=COUNTS,training_updates=0))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        progress('failed');write('failure.json',dict(error=repr(error)));raise
