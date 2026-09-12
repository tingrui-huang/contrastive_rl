"""One fixed-budget ETT return-minimization comparison with frozen policies."""
import argparse
from pathlib import Path
import shutil
import time
import jax
import jax.numpy as jnp
import numpy as np

from ett.anchored_transition import load_anchored
from ett.convex_action_transition import ConvexActionTransition,ConvexRollout,emit,energy_score,matrices,objective
from ett.rollout_return import START,GOAL,antithetic_gradient,descent_step
from ett.run_distribution_matching import frozen_actor
from ett.run_return_readout import read,write
from propensity.nominal_policy import load_nominal_policy
from scripts.make_swamp_f4_failure_bank import file_sha

SPEC=Path('notes/ett_adversarial_fixed_actor_spec.md')
OLD=Path('artifacts/ett_rollout_return/residual6_s01/config.json')
CONFIG=dict(seeds=[0,1],iterations=16,directions=8,paths=32,sigma=.1,learning_rate=1.,max_step=.2,
    dimension=32,bound=1.,horizon=50,discount=.95,weights=[0.,float(1/np.sum(.95**np.arange(50)))],
    signal_paths=256,monitor_paths=32,evaluation_seeds=[18010000,18010001,18010002,18010003],evaluation_paths=128,
    fit_rows_per_partition=2048,fit_samples=32,action_contexts=128,action_samples=8,arbitrary_pairs=16,
    seed_bases=dict(fit=18000000,fit_noise=18000001,signal=18000002,search=18100000,monitor=18200000,
                    directions=18300000,action=18400000,bootstrap=18500000),
    diagonal_max_sample_drift=2e-6,diagonal_max_energy_increase=1e-6,
    numerical_constraint_tolerance=2e-6,primary_transition_budget=1862400,
    additional_transition_budget=1000000,total_transition_cap=2862400,
    selection='final fixed iterate; no evaluation/monitor selection',
    input_arrays=['obs','act','meta'],simulator_or_readout_targets_used=False,
    diagonal_term='emitted XY energy fitting score; frozen constant; no shared-network diagonal learning')

REPORTS=[
    'artifacts/ett_rollout_return/residual6_s01/REPORT.md',
    'artifacts/ett_action_response/f4_return_s01_horizon50/REPORT.md',
    'artifacts/synthetic_shared_response/l025_1_4_s012/REPORT.md',
    'artifacts/synthetic_lipschitz_response/l1_s012/REPORT.md',
    'artifacts/synthetic_failure_bank/bank_m3_lambda01_s012/REPORT.md',
    'artifacts/synthetic_diagonal_budget/lambda0004_s012/REPORT.md',
    'artifacts/return_readout/f4_h20_g095_v1/REPORT.md',
    'artifacts/fixed_actor_continuation/f4_h20_g095_v1/SUMMARY.md']


def inputs():
    old=read(OLD);paths={k:Path(old['paths'][k]) for k in ['dataset','nominal','actor','control']}
    for p in paths.values():
        if not p.exists():raise FileNotFoundError(p)
        assert file_sha(p)==old['input_file_sha256'][p.as_posix()],p
    return paths


def setup():
    paths=inputs();old=load_anchored(paths['control'])
    nominal=load_nominal_policy(str(paths['nominal'].parent))
    assert nominal.spec.conditioning=='state_goal'
    actor,info=frozen_actor(str(paths['actor']),read(paths['actor'].with_name('arm_provenance.json'))['dataset_content_sha256'])
    model=ConvexActionTransition(old.diagonal,bound=CONFIG['bound'])
    return model,nominal,actor,ConvexRollout(model,nominal,actor),info


def fit_data():
    paths=inputs()
    with np.load(paths['dataset'],allow_pickle=False) as z:
        obs,act,meta=z['obs'],z['act'],read_json_scalar(z['meta'])
    assert meta['per_cell_swamp_prob']==.3 and obs.shape==(6600,51,16)
    assert np.all(obs[:,:,8:]==GOAL) and np.array_equal(obs[:,1:,2:8],obs[:,:-1,:6])
    validation=np.random.default_rng(0).permutation(6600)[:660]
    # Established teacher-source membership; no teacher_mode or hidden labels.
    ids=np.arange(1200,6000);rng=np.random.default_rng(CONFIG['seed_bases']['fit'])
    output={}
    for name,ep in [('train',np.setdiff1d(ids,validation)),('validation',np.intersect1d(ids,validation))]:
        rows=rng.choice(len(ep)*50,CONFIG['fit_rows_per_partition'],replace=False)
        e,t=ep[rows//50],rows%50
        output[name]=dict(state=obs[e,t,:8],goal=obs[e,t,8:],action=act[e,t],target=obs[e,t+1,:8],episode=e,time=t)
    return output


def read_json_scalar(value):
    import json
    return json.loads(str(value))


def prepare(root):
    if root.exists():raise ValueError('use a fresh output directory')
    paths=inputs();root.mkdir(parents=True);(root/'checkpoints').mkdir()
    shutil.copyfile(SPEC,root/'SPEC.md');write(root/'config.json',CONFIG)
    sources=['ett/convex_action_transition.py','ett/run_convex_adversarial.py','ett/eval_convex_adversarial.py',
             'scripts/test_convex_action_transition.py','ett/diagonal_transition.py','ett/anchored_transition.py',
             'ett/run_distribution_matching.py','ett/rollout_return.py','propensity/nominal_policy.py',str(SPEC)]
    hashes={str(p).replace('\\','/'):file_sha(p) for p in list(paths.values())+[OLD]+list(map(Path,REPORTS+sources))}
    write(root/'provenance.json',dict(base_commit='f1f24b3f22a931f2fedb25af7e97e759e36e11df',input_sha256=hashes,
        paths={k:v.as_posix() for k,v in paths.items()},rambo_url='https://arxiv.org/html/2204.12581v3#S5',
        forbidden_data='simulator continuation arrays, readout weights, hidden labels, U and bank distances are not loaded',
        nominal='existing noisy teacher-source behavior model, including failures; state_goal conditioning',
        diagonal='entire old guarded-control diagonal sampling law frozen; pre-projection NLL is diagnostic only'))
    data=fit_data();np.savez_compressed(root/'fit_contexts.npz',**{k+'_'+f:v for k,part in data.items() for f,v in part.items()})
    write(root/'prepared.json',dict(config_sha256=file_sha(root/'config.json'),spec_sha256=file_sha(root/'SPEC.md'),
        fit_contexts_sha256=file_sha(root/'fit_contexts.npz'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('Specification, budgets, hashes and fit contexts saved before training.',flush=True)


def verify(root):
    assert read(root/'config.json')==CONFIG
    manifest=read(root/'prepared.json')
    for name,key in [('config.json','config_sha256'),('SPEC.md','spec_sha256'),('fit_contexts.npz','fit_contexts_sha256')]:
        assert file_sha(root/name)==manifest[key],name
    for path,digest in read(root/'provenance.json')['input_sha256'].items():assert file_sha(path)==digest,path


def reset_run(engine,theta,seed,paths):
    return engine.run(theta,np.tile(START,(paths,1)),np.tile(GOAL,(paths,1)),jax.random.PRNGKey(seed))


def fit_before(root,model):
    data=dict(np.load(root/'fit_contexts.npz'));saved={};report={}
    for number,name in enumerate(['train','validation']):
        s,a,g,y=[data[name+'_'+field] for field in ['state','action','goal','target']]
        samples,detail=model._sample(model.theta,s,a,a,g,jax.random.PRNGKey(CONFIG['seed_bases']['fit_noise']+number),32)
        samples=np.asarray(samples);score=np.asarray(energy_score(jnp.asarray(samples[:,:,:2]),jnp.asarray(y[:,:2])))
        np.testing.assert_array_equal(samples[:,:,2:],np.broadcast_to(s[:,None,:6],samples[:,:,2:].shape))
        pred=samples[:,:,:2].mean(1);errors=np.linalg.norm(pred-y[:,:2],axis=1)
        # Explicitly labelled pre-projection score, not emitted likelihood.
        context=np.concatenate([s,a,a,g],axis=1)
        nll=-np.asarray(model.diagonal._log_prob(model.diagonal.params,context,y[:,:2]-s[:,:2]))
        report[name]=dict(energy_mean=float(score.mean()),energy_quantiles=np.quantile(score,[.5,.95,.99,1]),
            predictive_mean_l2_quantiles=np.quantile(errors,[.5,.95,.99,1]),predictive_mean_l2_mean=float(errors.mean()),
            unprojected_mixed_nll_mean=float(nll.mean()),nll_is_not_emitted_density=True)
        saved[name+'_samples']=samples;saved[name+'_energy']=score
    np.savez_compressed(root/'diagonal_before.npz',**saved);write(root/'diagonal_before.json',report)
    return report['train']['energy_mean']


def train(root):
    verify(root)
    if (root/'training.json').exists():raise ValueError('training already attempted')
    model,nominal,actor,engine,actor_info=setup();theta0=np.zeros(32,np.float32)
    before=jax.tree.map(lambda x:np.array(x),model.diagonal.params)
    nominal_before=jax.tree.map(lambda x:np.array(x),nominal.params)
    probe_key=jax.random.PRNGKey(19900001)
    actor_before=np.asarray(actor(START[None],GOAL[None],probe_key))
    nominal_probe=np.asarray(nominal.sample(START[None],probe_key,8,goal=GOAL[None]))
    signal=reset_run(engine,theta0,CONFIG['seed_bases']['signal'],256)
    nonzero=float(np.mean(signal['return']>0));std=float(signal['return'].std())
    write(root/'signal.json',dict(mean=float(signal['return'].mean()),std=std,nonzero_fraction=nonzero,
                               threshold_nonzero=.05,threshold_std=.1,passed=nonzero>=.05 and std>=.1))
    if nonzero<.05 or std<.1:
        write(root/'training.json',dict(status='stopped_signal',updates=0));return
    diag=fit_before(root,model);start=time.monotonic();records={};steps=256*50
    for seed in CONFIG['seeds']:
        for arm,weight in zip(['control','adversarial'],CONFIG['weights']):
            name=f's{seed}_{arm}';theta=theta0.copy();rng=np.random.default_rng(CONFIG['seed_bases']['directions']+seed)
            history=[]
            for iteration in range(CONFIG['iterations']+1):
                monitor=reset_run(engine,theta,CONFIG['seed_bases']['monitor']+seed*100+iteration,32);steps+=32*50
                history.append(dict(iteration=iteration,return_mean=float(monitor['return'].mean()),
                    objective=float(objective(diag,monitor['return'],weight)),theta_norm=float(np.linalg.norm(theta))))
                if iteration==CONFIG['iterations']:break
                directions=rng.normal(size=(8,32));plus=[];minus=[]
                for i,u in enumerate(directions):
                    key=CONFIG['seed_bases']['search']+seed*10000+iteration*8+i
                    p=reset_run(engine,theta+CONFIG['sigma']*u,key,32)['return']
                    m=reset_run(engine,theta-CONFIG['sigma']*u,key,32)['return'];steps+=2*32*50
                    plus.append(float(objective(diag,p,weight)));minus.append(float(objective(diag,m,weight)))
                gradient=antithetic_gradient(np.array(plus),np.array(minus),directions,CONFIG['sigma'],1.)
                theta=descent_step(theta,gradient,CONFIG['learning_rate'],CONFIG['max_step'])
                assert np.isfinite(theta).all() and np.linalg.norm(np.asarray(matrices(theta)),axis=(1,2)).max()<=1+2e-6
                history[-1].update(gradient_norm=float(np.linalg.norm(gradient)),paired_objectives=list(zip(plus,minus)))
            if arm=='control':np.testing.assert_array_equal(theta,theta0)
            np.savez(root/'checkpoints'/f'{name}.npz',theta=theta,bound=1.)
            records[name]=dict(weight=weight,seed=seed,history=history,
                checkpoint_sha256=file_sha(root/'checkpoints'/f'{name}.npz'))
            print(name,'monitor',history[0]['return_mean'],'->',history[-1]['return_mean'],flush=True)
    for a,b in zip(jax.tree.leaves(before),jax.tree.leaves(model.diagonal.params)):np.testing.assert_array_equal(a,b)
    for a,b in zip(jax.tree.leaves(nominal_before),jax.tree.leaves(nominal.params)):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(actor_before,np.asarray(actor(START[None],GOAL[None],probe_key)))
    np.testing.assert_array_equal(nominal_probe,np.asarray(nominal.sample(START[None],probe_key,8,goal=GOAL[None])))
    verify(root)
    write(root/'training.json',dict(status='complete',arms=records,elapsed_seconds=time.monotonic()-start,
        primary_model_transitions_so_far=steps,diagonal_single_step_samples=4096*32,
        diagonal_and_policies_frozen=True,actor=actor_info,diagonal_score_constant=diag))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',required=True)
    parser.add_argument('--phase',choices=['prepare','train'],required=True);args=parser.parse_args()
    {'prepare':prepare,'train':train}[args.phase](Path(args.out_dir))


if __name__=='__main__':main()
