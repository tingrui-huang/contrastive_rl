"""Fixed-model, stratified paired continuation diagnostic. No training entrypoint."""
import argparse
from pathlib import Path
import subprocess
import jax
import numpy as np
import torch
from ett import pointmaze_response_search as p
from ett.finite_crl import write, sha

BASE=Path('artifacts/pointmaze_region_pilot/response_search_s01_v1')
SPEC=Path('notes/pointmaze_continuation_precision.md')
CONFIG=dict(base='1d6deef',pairs=['s1_k0','s0_k2'],per_cell=16,bins=['0','1-3','4-9','10+'],
    successors=8,actors=2,batch=256,tolerance=.01,precision=.005,cap=7500000,
    expected=7370240,query_seed=171000000,time_seed=172000000,
    comparison_seed=175000000,diagonal_seed=178000000,updates=0,native_steps=0)

def bounds(h): return [(0,1),(1,4),(4,10),(10,h)]

def load_critic(name):
    critic=p.early.Critic(0)
    critic.load_state_dict(torch.load(BASE/'checks'/f'{name}_u1'/'critic.pt',weights_only=True)['model'])
    critic.eval()
    for param in critic.parameters(): param.requires_grad_(False)
    return critic

def provenance():
    manifest={k.replace('\\','/'):v for k,v in p.read(BASE/'manifest.json').items()}
    inherited=p.read(BASE/'preregistration.json')['sources']; sources={}; details={}
    # The previous reporting-only correction is documented and never relaxed silently.
    for f,h in inherited.items():
        if f=='ett/report_response_search.py' and sha(f)!=h:
            fix=p.read(BASE/'reporting_fix.json')
            assert sha(BASE/'sealed_report_source.py')==h==fix['original_sha256']
            assert sha(f)==fix['corrected_sha256']
        else: assert sha(f)==h,f
        sources[f]=sha(f)
    files=['candidate_pool.npz','search.json','results.json','starts.npz']
    for name in CONFIG['pairs']:
        files += [f'checks/{name}_u1/{f}' for f in ['critic.pt','matched_mc.npz','visitation.npz']]
    for f in files:
        assert sha(BASE/f)==manifest[f],f
        sources[str(BASE/f)]=sha(BASE/f)
    pool=dict(np.load(BASE/'candidate_pool.npz')); history=p.read(BASE/'search.json')['histories']
    ctx=dict(np.load(p.phase.PREVIOUS/'contexts.npz'))
    for name in CONFIG['pairs']:
        row=history[name][0]; critic=load_critic(name)
        assert row['update']==1 and row['fit']['steps']==1000
        assert p.old.parameter_sha(critic)==row['critic_sha256']
        np.testing.assert_array_equal(row['preupdate'],pool[name+'_u0'])
        saved=np.load(BASE/'checks'/f'{name}_u1'/'matched_mc.npz')
        np.testing.assert_array_equal(saved['preupdate'],pool[name+'_u0'])
        records=p.phase.load_records(BASE/'checks'/f'{name}_u1'/'visitation.npz')
        assert p.prior.array_sha(p.average.future_probabilities(records))==row['fit']['probabilities_sha256']
        details[name]=dict(critic_checkpoint=str(BASE/'checks'/f'{name}_u1'/'critic.pt'),
            checkpoint_sha256=sha(BASE/'checks'/f'{name}_u1'/'critic.pt'),
            parameter_sha256=row['critic_sha256'],reference=name+'_u0',fit=row['fit'],
            normalization_path=str(p.phase.PREVIOUS/'contexts.npz'),
            normalization_file_sha256=sha(p.phase.PREVIOUS/'contexts.npz'),
            mean_sha256=p.prior.array_sha(ctx['mean']),std_sha256=p.prior.array_sha(ctx['std']),
            normalization='same saved mean/std used by reference fit; no recomputation')
    return sources,details

def prepare(out):
    if out.exists(): raise ValueError('Fresh output directory required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert head.startswith(CONFIG['base'])
    sources,details=provenance()
    for f in [SPEC,Path(__file__).relative_to(Path.cwd()),Path('ett/report_continuation_precision.py'),
              Path('scripts/test_continuation_precision.py')]: sources[str(f)]=sha(f)
    # Include transitive source files without changing their previous versions.
    for folder in ['ett','propensity']:
        for f in Path(folder).glob('*.py'): sources[str(f)]=sha(f)
    out.mkdir(parents=True)
    write(out/'config.json',CONFIG);(out/'PROTOCOL.md').write_bytes(SPEC.read_bytes())
    write(out/'provenance.json',details)
    write(out/'preregistration.json',dict(utc=p.average.timestamp(),head=head,sources=sources,
        sealed={f:sha(out/f) for f in ['config.json','PROTOCOL.md','provenance.json']}))
    print('Protocol, provenance, code and 7.5M cap sealed before sampling.',flush=True)

def verify(out):
    reg=p.read(out/'preregistration.json');assert p.read(out/'config.json')==CONFIG
    for f,h in reg['sources'].items(): assert sha(f)==h,f
    for f,h in reg['sealed'].items(): assert sha(out/f)==h,f

def queries(engine,theta,ctx,pair,ledger,folder):
    records=p.early.collect(engine,theta,ctx['train_roots'],ctx['train_indices'][:,1],
        CONFIG['query_seed']+pair*1000000,ledger,'reference_queries',64)
    p.phase.save_records(folder/'reference_paths.npz',records)
    rng=np.random.default_rng(CONFIG['time_seed']+pair*1000000); rows=[]
    for r in records:
        H=r['action'].shape[1]
        for root in np.unique(r['root']):
            paths=np.flatnonzero(r['root']==root);assert len(paths)==64
            for b,(lo,hi) in enumerate(bounds(H)):
                for i in paths[b*16:(b+1)*16]:
                    t=int(rng.integers(lo,hi))
                    rows.append(dict(state=r['states'][i,t],action=r['action'][i,t],root=root,
                        group=ctx['train_indices'][root,2],bin=b,cell=root*4+b,
                        length=H,t=t,h=H-t,weight=H*.95**t,mass=(hi-lo)/(36*H*16),
                        pre_entry=not bool(r['reward'][i,:t].any()),
                        in_region=bool(p.old.task_reward(r['states'][i,t],p.GOAL))))
    q={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    assert len(q['h'])==2304 and np.all(q['pre_entry'][q['bin']==0])
    np.testing.assert_allclose(q['mass'].sum(),1)
    assert np.all(np.bincount(q['cell'])==16)
    np.savez_compressed(folder/'queries.npz',**q)
    return q

def sample(out):
    verify(out)
    if (out/'started.json').exists(): raise ValueError('Completed or started run protected')
    write(out/'started.json',dict(utc=p.average.timestamp()))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    engine=p.old.Kernel(); frozen=p.frozen_hash(engine)
    assert frozen==p.read(BASE/'search.json')['frozen_sha256']
    ctx=dict(np.load(p.phase.PREVIOUS/'contexts.npz'));pool=dict(np.load(BASE/'candidate_pool.npz'))
    assert not np.asarray(p.old.task_reward(ctx['train_roots'],np.broadcast_to(p.GOAL,ctx['train_roots'].shape))).any()
    ledger=p.old.Ledger(out);ledger.data['cap']=CONFIG['cap'];mc=p.Continuation(engine)
    for pair,name in enumerate(CONFIG['pairs']):
        folder=out/name;folder.mkdir();ref=pool[name+'_u0'];cand=pool[name+'_u24']
        before=p.tree_sha((ref,cand));critic=load_critic(name);critic_sha=p.old.parameter_sha(critic)
        np.testing.assert_array_equal(ref[:16],cand[:16])
        assert max(np.linalg.norm(x[16:].reshape(8,2,2),axis=(1,2)).max() for x in [ref,cand])<=1.000002
        probes=[p.diagonal_probe(engine,x,ctx,CONFIG['diagonal_seed'],ledger) for x in [ref,cand]]
        np.testing.assert_array_equal(*probes)
        q=queries(engine,ref,ctx,pair,ledger,folder)
        for batch,start in enumerate(range(0,len(q['h']),256)):
            part={k:v[start:start+256] for k,v in q.items()};n=len(part['h']); raw={}
            seed=CONFIG['comparison_seed']+pair*1000000+batch*100
            for label,theta in [('reference',ref),('candidate',cand)]:
                r=p.first_samples(engine,theta,part,critic,ctx,seed,8,2,ledger,'paired_first')
                s=np.repeat(r['successor'].reshape(-1,8),2,0);a=r['next_action'].reshape(-1,2)
                h=np.repeat(part['h']-1,16);assert h.min()>=0 and h.max()<=48
                ledger.add(len(s)*48,'paired_reference_continuations_including_padding')
                rewards,valid=mc.run(ref,s,a,h,jax.random.PRNGKey(seed+20))
                rewards=np.asarray(rewards);assert np.asarray(valid).all()
                assert not rewards[np.arange(48)[None,:]>=h[:,None]].any()
                mc_q=(.05*rewards@.95**np.arange(48)).reshape(n,8,2)
                r.update(mc_q=mc_q,continuation_rewards=rewards.reshape(n,8,2,48),
                    mc=part['weight'][:,None,None]*(.05*r['reward'][:,:,None]+.95*mc_q))
                raw[label]=r
            np.testing.assert_array_equal(raw['reference']['x_prime'],raw['candidate']['x_prime'])
            np.savez_compressed(folder/f'batch{batch}.npz',
                **{label+'_'+k:v for label,r in raw.items() for k,v in r.items()},
                reference_theta=ref,candidate_theta=cand,continuation_theta=ref,
                query_index=np.arange(start,start+n))
            print(name,f'batch {batch+1}/9 saved; no outcome analysis',flush=True)
        assert p.old.parameter_sha(critic)==critic_sha and p.tree_sha((ref,cand))==before
    assert frozen==p.frozen_hash(engine);assert ledger.data['charged']==CONFIG['expected']
    verify(out)
    write(out/'completion.json',dict(utc=p.average.timestamp(),computed_transitions=ledger.data['charged'],
        cap=CONFIG['cap'],actor_updates=0,critic_updates=0,nominal_updates=0,ett_updates=0,native_steps=0,
        frozen_verified=True,diagonal_bitwise_equal=True,
        samples={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*.npz'))}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('phase',choices=['prepare','sample'])
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();globals()[args.phase](args.out)
