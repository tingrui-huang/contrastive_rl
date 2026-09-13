"""Bounded, resumable two-mode fixed-actor ETT experiment; no CRL training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from ett.convex_action_transition import (ConvexActionTransition, ConvexRollout, energy_score,
    matrices, save_convex_checkpoint, load_convex_checkpoint, validate_selected_set)
from ett.run_convex_adversarial import inputs, setup, reset_run
from ett.rollout_return import START, GOAL, antithetic_gradient, descent_step

SPEC = Path('notes/ett_geometry_comparison_spec.md')
OLD = Path('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1')
CONFIG = dict(modes=['rectangle', 'fork_segment'], optimizer_seeds=[0, 1], updates=16,
    directions=8, paths=32, sigma=.1, learning_rate=1., max_step=.2, horizon=50, discount=.95,
    bound=1., dimension=32, initialization='exactly zero; no warm start',
    weight=float(1/np.sum(.95**np.arange(50))), fit_rows_per_partition=1024, fit_draws=16,
    evaluation_seeds=list(range(64000000,64000004)), evaluation_paths=128,
    component_contexts=64, component_draws=8, bootstrap_replicates=2000,
    seed_bases=dict(directions=61000000,search=62000000,monitor=63000000,diagonal=65000000,
                    component_nominal=65000100,component_anchor=65000101,smoke=66000000,
                    bootstrap=66000100,probe=66000200),
    representative_paths=dict(evaluation_seed=64000000,path_indices=[0,1,2,3]),
    transition_budget=dict(optimization=1638400,monitoring=108800,evaluation=128000,
        initialization_rollouts=800,diagonal=196608,replay=25600,component_anchors=512),
    planned_transitions=2098720,hard_cap=2500000,tolerance=2e-6,
    checkpoint_selection='final iterate only', diagonal_term='constant emitted energy fitting score',
    no_native_or_downstream_training=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x))
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=convert)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def tree_sha(tree):
    digest=hashlib.sha256()
    for item in jax.tree.leaves(tree):
        array=np.ascontiguousarray(item)
        digest.update(str((array.shape,array.dtype)).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def schedule():
    return dict(directions={str(s):61000000+s for s in [0,1]},
        optimization={str(s):[[62000000+s*10000+it*8+i for i in range(8)] for it in range(16)] for s in [0,1]},
        monitoring={str(s):[63000000+s*100+it for it in range(17)] for s in [0,1]},
        evaluation=CONFIG['evaluation_seeds'],diagonal=[65000000,65000001],
        component=[65000100,65000101],smoke=66000000,bootstrap=66000100,probe=66000200)


def prepare(root):
    if root.exists(): raise ValueError('use a fresh experiment directory')
    paths=inputs()  # Checks exact historical checkpoint hashes; missing files fail.
    assert sum(CONFIG['transition_budget'].values())==CONFIG['planned_transitions']
    # Search exact integer namespace bases, avoiding matches inside float fractions/hashes.
    search=subprocess.run(['rg','-n',r'\b(61000000|62000000|63000000|64000000|65000000|66000000)\b',
        'artifacts','ett','scripts','notes','-g','*.json','-g','*.py','-g','SPEC.md'],capture_output=True,text=True)
    current=('ett/run_geometry_comparison.py:','ett/eval_geometry_comparison.py:',
             'artifacts/ett_geometry_comparison/implementation_attempts/')
    hits=[line for line in search.stdout.splitlines() if not line.replace('\\','/').startswith(current)]
    if hits: raise ValueError(f'seed namespace already referenced: {hits[:10]}')
    root.mkdir(parents=True);(root/'checkpoints').mkdir();(root/'cache').mkdir()
    shutil.copyfile(SPEC,root/'SPEC.md')
    write(root/'config.json',CONFIG);write(root/'seed_schedule.json',schedule())
    write(root/'seed_audit.json',dict(query='exact proposed namespace bases in existing JSON/Python/SPEC artifacts',
        prior_matches=hits,scope='repository artifacts, ett, scripts, notes; current new comparison sources/failed preparation snapshot excluded',
        phases_disjoint=True))
    hashes={p.as_posix():sha(p) for p in paths.values()}
    hashes.update({p.as_posix():sha(p) for p in [paths['nominal'].with_name('config.json'),
        paths['actor'].with_name('arm_provenance.json'),OLD/'fit_contexts.npz']})
    for directory in [OLD,Path('artifacts/ett_structure_audit/fork_saved_s01_v1'),
                      Path('artifacts/ett_convex_set_component/fork_segment_v1')]:
        hashes.update({p.as_posix():sha(p) for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    sources=['ett/convex_action_transition.py','ett/run_convex_adversarial.py','ett/eval_convex_adversarial.py',
        'ett/run_geometry_comparison.py','ett/eval_geometry_comparison.py','ett/diagonal_transition.py',
        'ett/anchored_transition.py','ett/run_distribution_matching.py','ett/rollout_return.py',
        'propensity/nominal_policy.py','scripts/test_geometry_comparison.py',str(SPEC)]
    write(root/'provenance.json',dict(starting_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        requested_commit='be1b3c520ce91ea4a7e9c4d9f4a7fa0941e86345',
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        subsequent_commits=[],input_sha256=hashes,source_sha256={p:sha(p) for p in sources},
        paths={k:p.as_posix() for k,p in paths.items()},
        source_changes='mode-aware checkpoint API, rollout set diagnostics, selected-set validation; isolated comparison runner/evaluator',
        prior_source_sha256=read('artifacts/ett_convex_set_component/fork_segment_v1/final_audit.json')['output_and_source_sha256']))
    with np.load(OLD/'fit_contexts.npz') as z:
        np.savez_compressed(root/'fit_contexts.npz',**{k:z[k][:1024] for k in z.files})
    write(root/'prepared.json',{p:sha(root/p) for p in ['SPEC.md','config.json','seed_schedule.json','fit_contexts.npz','provenance.json']})
    write(root/'transition_ledger.json',dict(hard_cap=CONFIG['hard_cap'],planned=CONFIG['planned_transitions'],
                                           charged=0,entries=[]))
    print('Prepared fixed configuration, fresh schedules, inputs and 2,098,720-transition plan.',flush=True)


def verify(root):
    assert read(root/'config.json')==CONFIG
    for p,h in read(root/'prepared.json').items(): assert sha(root/p)==h,p
    provenance=read(root/'provenance.json')
    for p,h in {**provenance['input_sha256'],**provenance['source_sha256']}.items(): assert sha(p)==h,p


class Ledger:
    """Charge attempts before calls; cache completion and never hide failed draws."""
    def __init__(self,root): self.root=root;self.data=read(root/'transition_ledger.json')
    def call(self,name,category,count,signature,fn):
        entries=[e for e in self.data['entries'] if e['name']==name]
        completed=[e for e in entries if e['status']=='complete']
        if completed:
            e=completed[-1];assert e['signature']==signature
            path=self.root/e['cache'];assert sha(path)==e['sha256']
            return dict(np.load(path,allow_pickle=False))
        if self.data['charged']+count>self.data['hard_cap']: raise RuntimeError('transition cap exhausted')
        e=dict(name=name,category=category,transitions=count,signature=signature,status='reserved',attempt=len(entries)+1)
        self.data['entries'].append(e);self.data['charged']+=count
        write(self.root/'transition_ledger.json',self.data)
        try:
            result=jax.tree.map(np.asarray,fn())
            cache=Path('cache')/f'{name}_attempt{e["attempt"]}.npz'
            np.savez_compressed(self.root/cache,**result)
            e.update(status='complete',cache=cache.as_posix(),sha256=sha(self.root/cache))
            write(self.root/'transition_ledger.json',self.data)
            return result
        except BaseException:
            e.update(status='failed',error=traceback.format_exc())
            write(self.root/'transition_ledger.json',self.data)
            raise


def check_record(record):
    state=record['states'][:,:-1].reshape(-1,8);out=record['states'][:,1:].reshape(-1,1,8)
    mapping=dict(box_valid='valid',geometry_kind='geometry_kind',box_low='box_low',box_high='box_high',
        segment_start='segment_start',segment_end='segment_end',segment_fraction='segment_fraction')
    d={key:record[field].reshape((len(state),1)+record[field].shape[2:]) for key,field in mapping.items()}
    validate_selected_set(state,out,d,CONFIG['tolerance'])
    np.testing.assert_array_equal(record['goal'],np.broadcast_to(GOAL,record['goal'].shape))
    assert np.max(np.abs(record['action']))<=1 and np.max(np.abs(record['aux_x_prime']))<=1
    reward=(np.linalg.norm(record['states'][:,1:,:2]-GOAL[:2],axis=-1)<2).astype(np.float32)
    np.testing.assert_array_equal(record['reward'],reward)
    np.testing.assert_allclose(record['return'],reward@(.95**np.arange(50)),atol=5e-6,rtol=0)


def rollout(ledger,engine,theta,key,paths,name,category,keep=False):
    def compute():
        record=reset_run(engine,theta,key,paths);check_record(record)
        return record if keep else {'return':record['return']}
    signature=dict(theta=tree_sha(np.asarray(theta)),mode=engine.model.geometry_mode,key=int(key),paths=paths,keep=keep)
    return ledger.call(name,category,paths*50,signature,compute)


def frozen_fingerprints(model,nominal,actor):
    base=model.diagonal
    result=dict(diagonal=tree_sha(base.params),nominal=tree_sha(nominal.params),
        diagonal_normalization=tree_sha([base.context_mean,base.context_std,base.delta_mean,base.delta_std]),
        nominal_normalization=tree_sha([nominal.context_mean,nominal.context_std]),
        actor_probe=tree_sha(actor(START[None],GOAL[None],jax.random.PRNGKey(66000200))),
        nominal_probe=tree_sha(nominal.sample(START[None],jax.random.PRNGKey(66000200),8,goal=GOAL[None])))
    closure=dict(zip(actor.__wrapped__.__code__.co_freevars,actor.__wrapped__.__closure__))
    result['frozen_actor_and_critic_state']=tree_sha(closure['state'].cell_contents)
    return result


def diagonal(ledger,model,name,reference=None):
    data=dict(np.load(ledger.root/'fit_contexts.npz'));report={};saved={}
    for number,part in enumerate(['train','validation']):
        s,a,g,y=[data[part+'_'+f] for f in ['state','action','goal','target']]
        def compute():
            samples,d=model._sample(model.theta,s,a,a,g,jax.random.PRNGKey(65000000+number),16)
            validate_selected_set(s,samples,jax.tree.map(np.asarray,d))
            np.testing.assert_array_equal(samples[...,:2],d['anchor_xy'])
            return dict(samples=samples,energy=energy_score(samples[...,:2],jnp.asarray(y[:,:2])))
        record=ledger.call(f'diagonal_{name}_{part}','diagonal',len(s)*16,
                          dict(mode=model.geometry_mode,theta=tree_sha(model.theta),key=65000000+number),compute)
        if reference is not None:
            np.testing.assert_array_equal(record['samples'],reference[part+'_samples'])
            np.testing.assert_array_equal(record['energy'],reference[part+'_energy'])
        report[part]=dict(energy_mean=float(record['energy'].mean()),max_coupled_drift=0.,
            exact_anchor_identity=True,energy_unchanged=True,
            prediction_error_quantiles=np.quantile(np.linalg.norm(record['samples'][...,:2].mean(1)-y[:,:2],axis=-1),[.5,.95,.99,1]))
        saved.update({part+'_'+k:v for k,v in record.items()})
    return report,saved


def initialize(root,ledger,models,engines):
    if (root/'initialization.json').exists():
        return read(root/'initialization.json'),dict(np.load(root/'diagonal_initial.npz'))
    report,reference=diagonal(ledger,models['rectangle'],'zero_rectangle')
    other,_=diagonal(ledger,models['fork_segment'],'zero_fork_segment',reference)
    records={mode:rollout(ledger,engines[mode],np.zeros(32,np.float32),66000000,8,
                       f'smoke_{mode}','initialization_rollouts',True) for mode in CONFIG['modes']}
    for field in ['states','action','aux_x_prime','anchor','reward','return']:
        np.testing.assert_array_equal(records['rectangle'][field],records['fork_segment'][field])
    np.savez_compressed(root/'diagonal_initial.npz',**reference)
    result=dict(diagonal=report,other_zero_mode=other,matched_initial_paths=8,
                exact_initial_rollout_equality=True,diagonal_score=report['train']['energy_mean'])
    write(root/'initialization.json',result)
    return result,reference


def train(root):
    verify(root);ledger=Ledger(root)
    base,nominal,actor,_,info=setup()
    models={mode:ConvexActionTransition(base.diagonal,geometry_mode=mode) for mode in CONFIG['modes']}
    engines={mode:ConvexRollout(model,nominal,actor) for mode,model in models.items()}
    fingerprints=frozen_fingerprints(base,nominal,actor)
    if (root/'frozen_before.json').exists(): assert read(root/'frozen_before.json')==fingerprints
    else: write(root/'frozen_before.json',fingerprints)
    initial,_=initialize(root,ledger,models,engines);diag=initial['diagonal_score']
    completed={};start=time.monotonic()
    for seed in CONFIG['optimizer_seeds']:
        directions=np.random.default_rng(61000000+seed).normal(size=(16,8,32))
        np.savez_compressed(root/f'directions_s{seed}.npz',directions=directions)
        for mode in CONFIG['modes']:
            name=f'{mode}_s{seed}';theta=np.zeros(32,np.float32);history=[];theta_history=[theta.copy()]
            for iteration in range(17):
                monitor=rollout(ledger,engines[mode],theta,63000000+seed*100+iteration,32,
                                f'monitor_{name}_{iteration}','monitoring')
                row=dict(iteration=iteration,monitor_return=float(monitor['return'].mean()),
                         L_diag=diag,weighted_J=CONFIG['weight']*float(monitor['return'].mean()),theta_norm=float(np.linalg.norm(theta)))
                if iteration<16:
                    plus=[];minus=[]
                    for i,u in enumerate(directions[iteration]):
                        key=62000000+seed*10000+iteration*8+i
                        for sign,values in [(1,plus),(-1,minus)]:
                            result=rollout(ledger,engines[mode],theta+sign*.1*u,key,32,
                                f'query_{name}_{iteration}_{i}_{"plus" if sign==1 else "minus"}','optimization')
                            values.append(diag+CONFIG['weight']*float(result['return'].mean()))
                    grad=antithetic_gradient(np.array(plus),np.array(minus),directions[iteration],.1,1.)
                    next_theta=descent_step(theta,grad,1.,.2)
                    assert np.isfinite(next_theta).all()
                    assert np.linalg.norm(np.asarray(matrices(next_theta)),axis=(1,2)).max()<=1+2e-6
                    row.update(plus=plus,minus=minus,gradient_norm=float(np.linalg.norm(grad)),
                               update_norm=float(np.linalg.norm(next_theta-theta)))
                    theta=next_theta;theta_history.append(theta.copy())
                history.append(row)
                write(root/f'progress_{name}.json',dict(mode=mode,seed=seed,theta=theta,history=history))
            checkpoint=root/'checkpoints'/f'{name}.npz'
            if not checkpoint.exists(): save_convex_checkpoint(checkpoint,models[mode],theta)
            reloaded=load_convex_checkpoint(base.diagonal,checkpoint,require_geometry=True)
            assert reloaded.geometry_mode==mode
            np.testing.assert_array_equal(reloaded.theta,np.asarray(theta,np.float32))
            np.savez_compressed(root/f'theta_history_{name}.npz',theta=np.asarray(theta_history))
            completed[name]=dict(mode=mode,seed=seed,history=history,checkpoint_sha256=sha(checkpoint))
            print(name,'finished 16 updates; monitor',history[0]['monitor_return'],'->',history[-1]['monitor_return'],flush=True)
    assert frozen_fingerprints(base,nominal,actor)==fingerprints
    verify(root)
    write(root/'training.json',dict(status='complete',models=completed,diagonal_score=diag,
          frozen_unchanged=True,actor=info,elapsed_seconds=time.monotonic()-start))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','train'],required=True)
    args=parser.parse_args()
    try: {'prepare':prepare,'train':train}[args.phase](args.out_dir)
    except BaseException:
        if args.out_dir.exists():
            path=args.out_dir/f'failed_{args.phase}_{time.time_ns()}.json'
            write(path,dict(error=traceback.format_exc(),source_sha256=sha(__file__)))
        raise


if __name__=='__main__': main()
