"""Supervised emitted-energy-score comparison of the existing random ETT."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
from pathlib import Path
import sys,json,hashlib,subprocess,time
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as j
from ett.pointmaze_region_pilot import Kernel
from ett.convex_action_transition import energy_score,matrices
from ett.diagonal_transition import POINTMAZE_WALLS

SOURCE=Path('C:/Users/trhua/Documents/Codex/2026-09-08/f/outputs/supervised_ett_native_probe_v1')
DATA=SOURCE/'paired_transitions.npz'
CAP=13546800
COUNTS={'training':0,'evaluation':0,'checks':0,'rollout':0}
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
def write(name,v):(OUT/name).write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n',encoding='utf-8')
def cost(kind,n):
    assert sum(COUNTS.values())+n<=CAP,'hard cap exhausted'
    COUNTS[kind]+=int(n)
def progress(stage):
    write('costs.json',dict(stage=stage,counts=COUNTS,total=sum(COUNTS.values()),cap=CAP))
    print(stage,COUNTS,flush=True)
def numpy_es(y,target):
    k=y.shape[1]
    return np.linalg.norm(y-target[:,None],axis=-1).mean(1)-np.linalg.norm(y[:,:,None]-y[:,None,:],axis=-1).sum((1,2))/(2*k*(k-1))
def legal(y):
    floor=np.floor(y).astype(int)
    bounded=(y[...,0]>=0)&(y[...,0]<9)&(y[...,1]>=0)&(y[...,1]<5)
    cell=POINTMAZE_WALLS[np.clip(floor[...,0],0,8),np.clip(floor[...,1],0,4)]
    return bounded&(cell==0)

def prepare():
    assert not (OUT/'costs.json').exists(),'fresh output required'
    d=dict(np.load(DATA,allow_pickle=False));eq=np.all(d['xb']==d['xq'],axis=1)
    assert d['s'].shape==(19200,8) and np.array_equal(d['s'][:,:6],d['y'][:,2:])
    assert np.array_equal(np.unique(d['episode']),np.arange(96))
    assert np.isfinite(d['s']).all() and np.isfinite(d['y']).all()
    assert np.max(np.abs(d['xb']))<=1 and np.max(np.abs(d['xq']))<=1
    for ep in range(96):
        for t in range(50):
            ids=np.flatnonzero((d['episode']==ep)&(d['time']==t))
            assert len(ids)==4 and set(d['query'][ids])=={0,1,2,3}
            assert np.all(d['s'][ids]==d['s'][ids[0]]) and np.all(d['xb'][ids]==d['xb'][ids[0]])
    train=d['episode']<72;dev=~train
    np.savez_compressed(OUT/'split.npz',train_episodes=np.arange(72),dev_episodes=np.arange(72,96),
                        diagonal_train=np.flatnonzero(train&eq),off_train=np.flatnonzero(train&~eq),dev_rows=np.flatnonzero(dev))
    rr=[]
    for ep in range(72,96):
        for t in [0,5,10,20,35,46]:
            idx=[]
            for k in range(4):
                ids=np.flatnonzero((d['episode']==ep)&(d['time']==t+k)&(d['query']==ep%2))
                assert len(ids)==1;idx.append(ids[0])
            assert np.array_equal(d['y'][idx[:-1]],d['s'][idx[1:]])
            rr.append(idx)
    rr=np.array(rr)
    np.savez_compressed(OUT/'sealed_sequences.npz',rows=rr,states=d['s'][rr],targets=d['y'][rr],
                        actions=d['xq'][rr],native_advice=d['xb'][rr],episode=d['episode'][rr],
                        time=d['time'][rr],dead_before=d['dead_before'][rr])
    config=json.loads((ROOT/'artifacts/ett_rollout_return/residual6_s01/config.json').read_text())
    paths=[DATA,SOURCE/'config.json',SOURCE/'provenance.json',
           SOURCE.parent.parent/'work/supervised-ett-audit/pointmaze_supervised_probe.py']
    paths += [ROOT/config['paths'][k] for k in ['control','nominal','actor','dataset']]
    paths += [ROOT/p for p in ['ett/pointmaze_region_pilot.py','ett/convex_action_transition.py',
                 'ett/diagonal_transition.py','ett/anchored_transition.py','ett/run_convex_adversarial.py',
                 'propensity/nominal_policy.py','crl/envs.py']]
    for k in ['control','nominal']:
        assert sha(ROOT/config['paths'][k])==config['input_file_sha256'][config['paths'][k]]
    write('provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
          sources={str(p):sha(p) for p in paths},protocol_sha256=sha(OUT/'PROTOCOL.md'),driver_sha256=sha(__file__),
          sequence_sha256=sha(OUT/'sealed_sequences.npz'),numpy=np.__version__,jax=jax.__version__,
          units='maze distance; raw action coordinates; original F4',prepared_time=time.time()))
    write('data_audit.json',dict(rows=len(eq),train_diagonal=int(np.sum(train&eq)),train_off=int(np.sum(train&~eq)),
        dev_diagonal=int(np.sum(dev&eq)),dev_off=int(np.sum(dev&~eq)),
        strict_equality_definition=True,episode_split_preserved=True,all_outcomes_retained=True,
        evaluation_labels_not_used_in_training=True,development_evaluation_reused=True,
        compatible_diagonal=config['paths']['control'],compatible_nominal=config['paths']['nominal']))
    return d,eq

def train(kernel,d,eq):
    train=d['episode']<72;di=np.flatnonzero(train&eq);off=np.flatnonzero(train&~eq)
    # This function sees only state, recorded advice/executed action, and actual successor XY.
    def loss(theta,s,xb,xq,target,key):
        y,_=kernel._sample(theta,s,xq,xb,key,8)
        return energy_score(y[...,:2],target[:,:2]).mean()
    batch_loss=jax.jit(jax.vmap(loss,in_axes=(0,None,None,None,None,None)))
    sigma=np.r_[np.full(16,.01),np.full(32,.1)]
    rate=np.r_[np.full(16,.01),np.full(32,.05)]
    finals={'initial':np.zeros(48,np.float32)}
    np.savez(OUT/'initial.npz',theta=finals['initial'])
    for seed in [0,1]:
        for arm in ['A','B']:
            theta=np.zeros(48,np.float32);m=np.zeros(48);v=np.zeros(48);history=[];snap=[]
            for step in range(120):
                rng=np.random.default_rng(202609140+seed*100000+step*10)
                ids_d=rng.choice(di,128);ids_o=rng.choice(off,128);dirs=rng.normal(size=(8,48))
                if arm=='A':dirs[:,16:]=0
                trials=np.stack([theta+sigma*dirs,theta-sigma*dirs],1).reshape(16,48).astype(np.float32)
                terms=[]
                for term,idx in enumerate([ids_d]+([ids_o] if arm=='B' else [])):
                    cost('training',16*128*8)
                    key=jax.random.PRNGKey(314000000+seed*100000+step*10+term)
                    values=np.asarray(batch_loss(trials,*[j.asarray(d[k][idx]) for k in ['s','xb','xq','y']],key)).reshape(8,2)
                    assert np.isfinite(values).all()
                    terms.append(values)
                components=np.stack([np.mean((q[:,0]-q[:,1])[:,None]*dirs,axis=0)/(2*sigma) for q in terms])
                components[0,16:]=0
                gradient=components.sum(0)
                m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
                update=rate*(m/(1-.9**(step+1)))/(np.sqrt(v/(1-.999**(step+1)))+1e-8)
                for sl,cap in [(slice(0,16),.03),(slice(16,48),.1)]:
                    update[sl]*=min(1.,cap/max(np.linalg.norm(update[sl]),1e-12))
                theta=(theta-update).astype(np.float32)
                if arm=='A':assert np.array_equal(theta[16:],np.zeros(32))
                snap.append(theta.copy());history.append(dict(step=step+1,signed_terms=terms,
                    diagonal_gradient_norm=float(np.linalg.norm(components[0])),
                    off_gradient_norm=0 if arm=='A' else float(np.linalg.norm(components[1])),
                    head_update_norm=float(np.linalg.norm(update[:16])),response_update_norm=float(np.linalg.norm(update[16:]))))
                if (step+1)%30==0:progress(f'{arm}_s{seed}_u{step+1}')
            name=f'{arm}_s{seed}';finals[name]=theta
            np.savez_compressed(OUT/(name+'.npz'),theta=theta,trajectory=np.array(snap),adam_m=m,adam_v=v)
            write(name+'_training.json',history)
    return finals

def check(kernel,theta,d):
    ids=np.arange(128);s,xb=[d[k][ids] for k in ['s','xb']]
    key=jax.random.PRNGKey(414000000)
    cost('checks',128*8);a,detail=kernel.sample(theta,s,xb,xb,key,8);a=np.asarray(a)
    np.testing.assert_array_equal(a[...,:2],np.asarray(detail['anchor_xy']))
    target=d['y'][ids,:2];v=np.asarray(energy_score(j.asarray(a[...,:2]),j.asarray(target)))
    np.testing.assert_allclose(v,numpy_es(a[...,:2].astype(float),target.astype(float)),atol=2e-6)
    rng=np.random.default_rng(414000001);x=rng.uniform(-1,1,(128,2)).astype(np.float32);z=rng.uniform(-1,1,(128,2)).astype(np.float32)
    cost('checks',2*128*8)
    p,pd=kernel.sample(theta,s,x,xb,key,8);q,qd=kernel.sample(theta,s,z,xb,key,8)
    np.testing.assert_array_equal(pd['anchor_xy'],qd['anchor_xy'])
    np.testing.assert_array_equal(pd['stationary_atom'],qd['stationary_atom'])
    ratio=np.linalg.norm(np.asarray(p)[...,:2]-np.asarray(q)[...,:2],axis=-1)/np.linalg.norm(x-z,axis=-1)[:,None]
    assert ratio.max()<=1+2e-6
    return dict(diagonal_identity=True,energy_numpy_agreement=True,coupled_anchors_atoms=True,action_ratio_max=float(ratio.max()),
                head_offset_norm=float(np.linalg.norm(theta[:16])),response_norm=float(np.linalg.norm(theta[16:])))

def evaluate(kernel,d,eq,finals):
    ids=np.flatnonzero(d['episode']>=72);checks={}
    for name,theta in finals.items():
        checks[name]=check(kernel,theta,d)
        outputs=[];atom=[];project=[]
        for start in range(0,len(ids),128):
            idx=ids[start:start+128];cost('evaluation',len(idx)*64)
            y,details=kernel.sample(theta,*[d[k][idx] for k in ['s','xq','xb']],jax.random.PRNGKey(514000000+start),64)
            y=np.asarray(y)
            assert np.isfinite(y).all()
            np.testing.assert_array_equal(y[:,:,2:],np.broadcast_to(d['s'][idx,None,:6],y[:,:,2:].shape))
            outputs.append(y[:,:,:2]);atom.append(np.asarray(details['stationary_atom']));project.append(np.asarray(details['projection_corrected']))
        y=np.concatenate(outputs);mean=y.mean(1)
        es=numpy_es(y.astype(float),d['y'][ids,:2].astype(float))
        sq=np.sum((mean-d['y'][ids,:2])**2,axis=-1)
        np.savez_compressed(OUT/(name+'_evaluation.npz'),rows=ids,samples_xy=y,es=es,mean=mean,squared_error=sq,
             legal=legal(y),mean_legal=legal(mean),atom=np.concatenate(atom),projection=np.concatenate(project),
             stationary=np.linalg.norm(y-d['s'][ids,None,:2],axis=-1)<=1e-7)
        progress(name+'_evaluated')
    write('checks.json',checks)

def bootstrap(left,right,episodes,mask,rmse=False):
    # Resample full original development episodes; missing strata contribute zero weight.
    ep=np.arange(72,96);n=np.array([np.sum(mask&(episodes==e)) for e in ep])
    a=np.array([left[mask&(episodes==e)].sum() for e in ep]);b=np.array([right[mask&(episodes==e)].sum() for e in ep])
    rng=np.random.default_rng(614000000);w=rng.multinomial(24,np.full(24,1/24),size=2000);den=w@n;valid=den>0
    av=(w[valid]@a)/den[valid];bv=(w[valid]@b)/den[valid]
    trans=np.sqrt if rmse else lambda x:x
    return dict(mean=float(trans(a.sum()/n.sum())-trans(b.sum()/n.sum())),
                ci95=np.quantile(trans(av)-trans(bv),[.025,.975]).tolist(),
                contributing_episodes=int(np.sum(n>0)),empty_bootstrap_replicates=int(np.sum(~valid)))

def summarize(d,eq,finals):
    ids=np.flatnonzero(d['episode']>=72);ep=d['episode'][ids];alive=~d['dead_before'][ids]
    event={'all':np.ones(len(ids),bool),'alive_outside_goal':alive&(np.linalg.norm(d['s'][ids,:2]-[8.5,3.5],axis=1)>=2),
           'ordinary_motion':alive&~d['dead_after'][ids]&(np.linalg.norm(d['y'][ids,:2]-d['s'][ids,:2],axis=1)>1e-7),
           'fatal_onset':alive&d['dead_after'][ids],'already_dead':~alive,
           'teacher_prefix':d['prefix_mode'][ids]==0,'blind_prefix':d['prefix_mode'][ids]==1}
    groups={side+'__'+k:(eq[ids] if side=='diagonal' else ~eq[ids])&v for side in ['diagonal','off'] for k,v in event.items()}
    scores={name:dict(np.load(OUT/(name+'_evaluation.npz'))) for name in finals}
    metrics={};contrasts={}
    for name,s in scores.items():
        metrics[name]={}
        for label,mask in groups.items():
            if not mask.any():metrics[name][label]={'rows':0};continue
            metrics[name][label]=dict(rows=int(mask.sum()),episodes=len(np.unique(ep[mask])),es=float(s['es'][mask].mean()),
                 mean_xy_vector_rmse=float(np.sqrt(s['squared_error'][mask].mean())),illegal_sample_fraction=float(1-s['legal'][mask].mean()),
                 illegal_mean_fraction=float(1-s['mean_legal'][mask].mean()),atom_fraction=float(s['atom'][mask].mean()),
                 stationary_fraction=float(s['stationary'][mask].mean()),projection_fraction=float(s['projection'][mask].mean()),
                 mean_step_distance=float(np.linalg.norm(s['samples_xy'][mask]-d['s'][ids[mask],None,:2],axis=-1).mean()))
    gates={}
    for seed in [0,1]:
        a=scores[f'A_s{seed}'];b=scores[f'B_s{seed}'];initial=scores['initial'];c={}
        for label,mask in groups.items():
            if not mask.any():continue
            c[label]=dict(es=bootstrap(b['es'],a['es'],ep,mask),rmse=bootstrap(b['squared_error'],a['squared_error'],ep,mask,True))
        diag=groups['diagonal__all'];off=groups['off__all']
        versus_initial=bootstrap(b['es'],initial['es'],ep,diag)
        ae=float(a['es'][off].mean());diff=c['off__all']['es'];dg=c['diagonal__all']['es']
        passed=(diff['mean']<=-.02 and ae>0 and -diff['mean']>=.1*ae and diff['ci95'][1]<0
            and dg['ci95'][1]<=.02 and versus_initial['ci95'][1]<=.02
            and c['off__all']['rmse']['mean']<=0 and b['legal'].all() and a['legal'].all())
        gates[str(seed)]=dict(passed=bool(passed),B_minus_initial_diagonal_es=versus_initial,
              relative_off_es_reduction=-diff['mean']/ae if ae>0 else None)
        contrasts[str(seed)]=c
    write('metrics.json',metrics);write('contrasts.json',contrasts);write('gate.json',dict(seeds=gates,passed=all(v['passed'] for v in gates.values())))
    return all(v['passed'] for v in gates.values())

def rollout(kernel,finals):
    data=dict(np.load(OUT/'sealed_sequences.npz'));goals=np.tile(np.array([8.5,3.5]*4,np.float32),(144*64,1))
    for name,theta in finals.items():
        state=np.repeat(data['states'][:,0],64,axis=0);states=[state.copy()]
        for t in range(4):
            xp=kernel.nominal.sample(state,jax.random.PRNGKey(714000000+t),1,goal=goals)
            action=np.repeat(data['actions'][:,t],64,axis=0)
            cost('rollout',len(state));y,_=kernel.sample(theta,state,action,xp,jax.random.PRNGKey(714000100+t),1)
            state=np.asarray(y)[:,0];assert np.isfinite(state).all();states.append(state.copy())
        arr=np.stack(states,1).reshape(144,64,5,8)
        np.savez_compressed(OUT/(name+'_rollout.npz'),states=arr)
    write('rollout_status.json',dict(ran=True,law='historical K5 nominal on each generated F4; fixed recorded executed actions',
           conditional_accuracy_claim=False,reason='Native shadow-teacher advice law differs after branch divergence'))

def main():
    d,eq=prepare();progress('prepared');kernel=Kernel()
    finals=train(kernel,d,eq);assert COUNTS['training']==11796480
    evaluate(kernel,d,eq,finals)
    passed=summarize(d,eq,finals)
    if passed:rollout(kernel,finals)
    else:write('rollout_status.json',dict(ran=False,reason='Predeclared one-step distribution/preservation gate did not pass in both seeds'))
    p=json.loads((OUT/'provenance.json').read_text())
    assert all(sha(path)==h for path,h in p['sources'].items())
    assert sha(OUT/'PROTOCOL.md')==p['protocol_sha256']
    progress('complete');write('completion.json',dict(status='complete',costs=COUNTS,source_hashes_unchanged=True,
        updates=480,actor_updates=0,critic_updates=0,nominal_updates=0,native_steps=0,gate_passed=passed))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        progress('failed');write('failure.json',dict(error=repr(exc)));raise
