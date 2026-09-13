"""Critic-only comparison: uniform rows versus within-trajectory phase balance."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write,sha

PREVIOUS=Path('artifacts/pointmaze_region_pilot/early_contexts_s01_v1')
PHASES=('early','middle','late')
CONFIG=dict(base='23da83f',arms=['uniform','balanced'],seeds=[0,1],critic_seed=98000000,
    steps=1500,batch=256,lr=.003,training_paths_per_root=8,eval_repeats=64,
    phase_boundaries='[0,5), [5,H//2), [H//2,H)',query_times='0,5,H//2',
    nce_weights=[1,31],q=[.5,.5],alpha=0.,discount=.95,horizon_scale=50,
    model_output_cap=650000,native_episodes=600,native_step_cap=30000,
    native_seed=103000000,final_selection_seed=103000001,
    training_collection_seed=99000000,positive_seed=99000100,
    train_evaluation_seed=104001000,final_query_seed=105000000,final_evaluation_seed=105001000,
    bootstrap_seed=106000000,bootstrap_repeats=2000,
    ranking_min_difference=.01,ranking_z=2.58,ranking_min_pairs=10,ranking_min_episodes=8,
    final_iterates_only=True)


def phase_bounds(lengths):
    lengths=np.asarray(lengths,int)
    assert np.all(lengths>10)
    lower=np.stack([np.zeros_like(lengths),np.full_like(lengths,5),lengths//2],-1)
    upper=np.stack([np.full_like(lengths,5),lengths//2,lengths],-1)
    return lower,upper


def training_rows(records,root_groups,mean,std):
    x,labels=early.positives(records,CONFIG['positive_seed'],mean,std)
    lengths=[];roots=[]
    for r in records:
        lengths.extend([r['action'].shape[1]]*len(r['root']));roots.extend(r['root'])
    lengths=np.asarray(lengths,int);roots=np.asarray(roots,int)
    offsets=np.r_[0,np.cumsum(lengths)[:-1]]
    path=np.repeat(np.arange(len(lengths)),lengths)
    t=np.concatenate([np.arange(h) for h in lengths])
    phase=np.where(t<5,0,np.where(t<lengths[path]//2,1,2))
    return dict(x=x,label=labels,lengths=lengths,offsets=offsets,path=path,
        root=roots[path],group=np.asarray(root_groups)[roots[path]],phase=phase,t=t)


def sample_rows(rng,data,arm,size=256):
    if arm=='uniform':return rng.integers(len(data['label']),size=size)
    if arm!='balanced':raise ValueError(arm)
    path=rng.integers(len(data['lengths']),size=size);phase=rng.integers(3,size=size)
    lower,upper=phase_bounds(data['lengths'][path]);lo=lower[np.arange(size),phase];hi=upper[np.arange(size),phase]
    t=lo+np.floor(rng.random(size)*(hi-lo)).astype(int)
    return data['offsets'][path]+t


def fit_rows(critic,data,arm,seed,steps=1500):
    x,label=torch.from_numpy(data['x']),torch.from_numpy(data['label'])
    optimizer=torch.optim.Adam(critic.parameters(),lr=.003)
    rng=np.random.default_rng(seed);counts=np.zeros((3,3),np.int64)
    for param in critic.parameters():param.requires_grad_(True)
    for _ in range(steps):
        idx=sample_rows(rng,data,arm)
        counts+=np.bincount(data['phase'][idx]*3+data['group'][idx],minlength=9).reshape(3,3)
        logits=critic(x[idx])
        positive=F.softplus(-logits).gather(1,label[idx,None]).mean()
        negative=31*F.softplus(logits).mean()
        loss=(positive+negative)/32
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    for param in critic.parameters():param.requires_grad_(False)
    return optimizer,dict(final_minibatch_nce=float(loss.detach()),row_exposures=counts,steps=steps)


@torch.no_grad()
def empirical_fit(critic,data):
    f=critic(torch.from_numpy(data['x']))
    losses=((F.softplus(-f).gather(1,torch.from_numpy(data['label'])[:,None])[:,0]+31*F.softplus(f).mean(-1))/32).numpy()
    rows=[]
    for phase_id,phase in enumerate(PHASES):
        for group_id,group in [(-1,'all'),*enumerate(early.GROUPS)]:
            mask=(data['phase']==phase_id)&((data['group']==group_id) if group_id>=0 else True)
            rows.append(dict(phase=phase,group=group,rows=int(mask.sum()),empirical_nce=float(losses[mask].mean()),
                label_one_fraction=float(data['label'][mask].mean())))
    return rows


def save_records(path,records):
    np.savez_compressed(path,**{f'b{i}_{key}':value for i,r in enumerate(records) for key,value in r.items()})


def load_records(path):
    with np.load(path) as z:
        buckets=sorted({int(k.split('_')[0][1:]) for k in z.files})
        return [{k.split('_',1)[1]:z[k] for k in z.files if k.startswith(f'b{i}_')} for i in buckets]


def prepare(out):
    if out.exists():raise ValueError('Fresh output directory required.')
    early.verify(PREVIOUS)
    sources=['ett/pointmaze_phase_sampling.py','ett/eval_pointmaze_phase_sampling.py',
        'scripts/test_pointmaze_phase_sampling.py','notes/pointmaze_phase_sampling.md',
        'scripts/collect_swamp_windy_f4.py','scripts/collect_swamp_windy.py','crl/envs.py','crl/config.py']
    prior=json.loads((PREVIOUS/'preregistration.json').read_text())
    hashes={**prior['inputs'],**{s:sha(s) for s in sources}}
    for s in ['contexts.npz','preflight.npz','checkpoints/initial.pt']:
        hashes[str(PREVIOUS/s)]=sha(PREVIOUS/s)
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'local_inputs').mkdir()
    write(out/'config.json',CONFIG);(out/'PROTOCOL.md').write_bytes(Path('notes/pointmaze_phase_sampling.md').read_bytes())
    write(out/'preregistration.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=hashes,config_sha256=sha(out/'config.json'),protocol_sha256=sha(out/'PROTOCOL.md'),
        final_episode_policy='New native episodes created after all critic checkpoints are sealed.'))


def verify(out):
    declaration=json.loads((out/'preregistration.json').read_text())
    assert json.loads((out/'config.json').read_text())==CONFIG
    assert sha(out/'config.json')==declaration['config_sha256']
    assert sha(out/'PROTOCOL.md')==declaration['protocol_sha256']
    for name,digest in declaration['sources'].items():assert sha(name)==digest,name


def run(out):
    from ett import eval_pointmaze_phase_sampling as evaluation
    verify(out)
    if (out/'started.json').exists():raise ValueError('Existing attempt is protected.')
    write(out/'started.json',dict(status='started'))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ledger=old.Ledger(out);ledger.data['cap']=CONFIG['model_output_cap'];ledger.add(0,'begin')
    engine=old.Kernel();context=dict(np.load(PREVIOUS/'contexts.npz'));theta=np.zeros(48,np.float32)
    paths=early.collect(engine,theta,context['train_roots'],context['train_indices'][:,1],CONFIG['training_collection_seed'],ledger,'shared_training')
    save_records(out/'shared_training_paths.npz',paths)
    data=training_rows(paths,context['train_indices'][:,2],context['mean'],context['std'])
    np.savez_compressed(out/'shared_training_rows.npz',**data)
    models={};fits={};checkpoint_hashes={}
    for seed in [0,1]:
        paired_hash=None
        for arm in CONFIG['arms']:
            name=f'{arm}_s{seed}';critic=early.Critic(CONFIG['critic_seed']+seed)
            initial=old.parameter_sha(critic)
            if paired_hash is None:paired_hash=initial
            else:assert initial==paired_hash
            opt,info=fit_rows(critic,data,arm,CONFIG['critic_seed']+seed)
            info.update(initial_parameters_sha256=initial,final_parameters_sha256=old.parameter_sha(critic),empirical_fit=empirical_fit(critic,data))
            torch.save(dict(model=critic.state_dict(),optimizer=opt.state_dict()),out/'checkpoints'/(name+'.pt'))
            checkpoint_hashes[name]=sha(out/'checkpoints'/(name+'.pt'));models[name]=critic;fits[name]=info
            print(name+': fixed 1500-step critic fit complete',flush=True)
    previous=early.Critic(0);previous.load_state_dict(torch.load(PREVIOUS/'checkpoints'/'initial.pt',weights_only=True)['model'])
    reproduced=old.parameter_sha(previous)==old.parameter_sha(models['uniform_s0'])
    write(out/'training.json',dict(models=fits,shared_rows_sha256=sha(out/'shared_training_rows.npz'),
        shared_paths_sha256=sha(out/'shared_training_paths.npz'),baseline_seed0_reproduces_previous=reproduced,
        checkpoint_hashes=checkpoint_hashes,total_optimizer_steps=6000,ett_updates=0,actor_updates=0))
    if not reproduced:raise RuntimeError('Uniform baseline replay mismatch; do not inspect final evaluation.')
    evaluation.evaluate(engine,out,context,paths,models,ledger)
    verify(out)
    for name,digest in checkpoint_hashes.items():assert sha(out/'checkpoints'/(name+'.pt'))==digest


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=['prepare','run']);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();(prepare if args.phase=='prepare' else run)(args.out)
