"""Independent saved-array verification and descriptive rollout summaries; no sampling."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def write(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
provenance=read('provenance.json')
assert all(sha(p)==h for p,h in provenance['sources'].items())
assert sha(OUT/'PROTOCOL.md')==provenance['protocol_sha256']
assert sha(OUT/'run_before_legality_fix.py')==provenance['driver_sha256']
source=next(p for p in provenance['sources'] if p.endswith('paired_transitions.npz'))
d=dict(np.load(source));split=dict(np.load(OUT/'split.npz'));rows=split['dev_rows'];eq=np.all(d['xb'][rows]==d['xq'][rows],axis=1)
metrics=read('metrics.json');verified={}
for name in metrics:
    z=dict(np.load(OUT/(name+'_evaluation.npz')))
    np.testing.assert_array_equal(z['rows'],rows)
    old=dict(np.load(OUT/'before_legality_correction'/(name+'_evaluation.npz')))
    for key in ['samples_xy','mean','es','squared_error','atom','projection','stationary']:
        np.testing.assert_array_equal(z[key],old[key])
    score=[]
    for start in range(0,len(rows),64):
        y=z['samples_xy'][start:start+64].astype(float);target=d['y'][rows[start:start+64],:2]
        first=np.sqrt(np.sum((y-target[:,None])**2,axis=-1)).mean(1)
        second=np.zeros(len(y))
        for k in range(64):second+=np.linalg.norm(y[:,k,None]-y[:,k+1:],axis=-1).sum(1)
        score.extend(first-second/(64*63))
    np.testing.assert_allclose(score,z['es'],atol=1e-12)
    assert z['legal'].all()
    verified[name]={'score_reconstructed':True,'all_emitted_positions_native_legal':True,'original_samples_unchanged':True}

sigma=np.r_[np.full(16,.01),np.full(32,.1)];rates=np.r_[np.full(16,.01),np.full(32,.05)]
updates={}
for seed in [0,1]:
    for arm in ['A','B']:
        name=f'{arm}_s{seed}';history=read(name+'_training.json');theta=np.zeros(48,np.float32);m=np.zeros(48);v=np.zeros(48)
        ck=dict(np.load(OUT/(name+'.npz')))
        for i,h in enumerate(history):
            rng=np.random.default_rng(202609140+seed*100000+i*10)
            rng.choice(split['diagonal_train'],128);rng.choice(split['off_train'],128)
            dirs=rng.normal(size=(8,48))
            if arm=='A':dirs[:,16:]=0
            terms=np.asarray(h['signed_terms'])
            components=np.stack([np.mean((t[:,0]-t[:,1])[:,None]*dirs,axis=0)/(2*sigma) for t in terms])
            components[0,16:]=0;g=components.sum(0)
            m=.9*m+.1*g;v=.999*v+.001*g*g
            u=rates*(m/(1-.9**(i+1)))/(np.sqrt(v/(1-.999**(i+1)))+1e-8)
            for sl,cap in [(slice(0,16),.03),(slice(16,48),.1)]:u[sl]*=min(1.,cap/max(np.linalg.norm(u[sl]),1e-12))
            theta=(theta-u).astype(np.float32)
            np.testing.assert_allclose(theta,ck['trajectory'][i],atol=2e-7,rtol=2e-7)
        np.testing.assert_allclose(theta,ck['theta'],atol=2e-7,rtol=2e-7)
        updates[name]={'updates':len(history),'head_changed':bool(np.linalg.norm(theta[:16])>0),
                       'response_changed':bool(np.linalg.norm(theta[16:])>0),
                       'nonzero_diagonal_gradient_steps':sum(h['diagonal_gradient_norm']>0 for h in history),
                       'nonzero_off_gradient_steps':sum(h['off_gradient_norm']>0 for h in history)}

seq=dict(np.load(OUT/'sealed_sequences.npz'));reference=np.concatenate([seq['states'][:,:1],seq['targets']],axis=1)
np.testing.assert_array_equal(reference[:,:-1],seq['states'])
groups={'all':np.ones(144,bool),'teacher_prefix':seq['episode'][:,0]%2==0,'blind_prefix':seq['episode'][:,0]%2==1,
        'already_dead_at_root':seq['dead_before'][:,0],
        'alive_outside_at_root':~seq['dead_before'][:,0]&(np.linalg.norm(seq['states'][:,0,:2]-[8.5,3.5],axis=1)>=2)}
roll={}
def describe(arr,mask):
    # arr: roots, replicates, time, F4. Position distribution is relative to each root.
    y=arr[mask];step=np.linalg.norm(np.diff(y[...,:2],axis=2),axis=-1)
    relative=y[:,:,1:,:2]-y[:,:,:1,:2]
    means=relative.mean((0,1));centered=relative-means[None,None]
    covariance=np.einsum('nrti,nrtj->tij',centered,centered)/(len(y)*y.shape[1])
    return dict(roots=int(mask.sum()),mean_step_distance=step.mean((0,1)).tolist(),
         stationary_step_fraction=(step<=1e-7).mean((0,1)).tolist(),
         near_stationary_step_fraction=(step<=.01).mean((0,1)).tolist(),
         any_motion_path_fraction=float((step>1e-7).any(-1).mean()),
         mean_relative_xy=means.tolist(),relative_xy_covariance=covariance.tolist())
roll['native_recorded']={k:describe(reference[:,None],mask) for k,mask in groups.items() if mask.any()}
for name in metrics:
    arr=np.load(OUT/(name+'_rollout.npz'))['states']
    np.testing.assert_array_equal(arr[:,:,1:,2:],arr[:,:,:-1,:6])
    roll[name]={k:describe(arr,mask) for k,mask in groups.items() if mask.any()}
write('rollout_descriptive.json',dict(scope='Separate laws: native recorded shadow-teacher prefixes versus generated historical-nominal continuations. Not conditional ETT accuracy.',models=roll))
flat=[]
for name,groups_m in metrics.items():
    for group,row in groups_m.items():flat.append(dict(model=name,group=group,**row))
with (OUT/'one_step_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)));w.writeheader();w.writerows(flat)

fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for seed,ax in enumerate(axes):
    labels=['All off','Alive/outside','Fatal onset','Already dead']
    keys=['off__all','off__alive_outside_goal','off__fatal_onset','off__already_dead']
    x=np.arange(4)
    ax.bar(x-.17,[metrics[f'A_s{seed}'][k]['es'] for k in keys],.34,label='A: diagonal only',color='#737b84')
    ax.bar(x+.17,[metrics[f'B_s{seed}'][k]['es'] for k in keys],.34,label='B: diagonal + off',color='#3274a1')
    ax.set_xticks(x,labels,rotation=15);ax.set(ylabel='Emitted XY energy score (maze units)',title=f'Training seed {seed}')
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
fig.suptitle('Original stochastic ETT · reused 24-episode development split')
fig.savefig(OUT/'one_step_scores.png',dpi=180);plt.close(fig)
counts=read('completion.json')['costs'];assert counts==dict(training=11796480,evaluation=1536000,checks=15360,rollout=184320)
write('verification.json',dict(status='pass',source_and_protocol_hashes_unchanged=True,
    score_checks=verified,optimizer_reconstruction=updates,rollout_f4_shift=True,
    original_split_preserved=True,model_successors=sum(counts.values()),native_steps=0,
    extra_model_calls=0,legality_fix_preserves_all_samples=True))
print(json.dumps({'status':'pass','model_successors':sum(counts.values()),'rollout_dead_roots':int(groups['already_dead_at_root'].sum())}))
