"""Final-only native outcomes and cross-model rollouts; never training inputs."""
import json
import jax
import numpy as np

from ett import pointmaze_native_policy as p
from ett import pointmaze_future_average as average
from ett.fixed_actor_continuation import TimedSimulator
from ett.finite_crl import write,sha
from ett.policy_improvement import tree_sha


def metrics(record,native=False):
    rewards=record['reward']; positions=record.get('position',record['states'][...,:2])
    result=dict(returns=rewards@.95**np.arange(50),
        success=(np.linalg.norm(positions-p.GOAL[:2],axis=-1).min(1)<.5).astype(float))
    if native: result['failure']=record['failure'][:,-1].astype(float)
    else: result['zero_return']=(result['returns']==0).astype(float)
    return result


def paired(a,b):
    rng=np.random.default_rng(p.CONFIG['bootstrap_seed']); n=len(a['returns'])
    index=rng.integers(n,size=(2000,n)); result={}
    for key in a:
        difference=a[key]-b[key]
        row=dict(difference=float(difference.mean()),ci95=np.quantile(difference[index].mean(1),[.025,.975]),
            observed_all_ties=bool(np.all(difference==0)))
        if key!='returns': row.update(a_one_b_zero=int(np.sum((a[key]==1)&(b[key]==0))),a_zero_b_one=int(np.sum((a[key]==0)&(b[key]==1))))
        result[key]=row
    return result


def native(params,sample,reset_seeds):
    sims=[TimedSimulator(int(seed)) for seed in reset_seeds]
    states=[np.stack([sim.observation()[:8] for sim in sims])]
    np.testing.assert_array_equal(states[0],np.tile(p.START,(len(sims),1)))
    positions=[np.stack([sim.env.state.copy() for sim in sims])]
    actions=[]; rewards=[]; failures=[]
    for t in range(50):
        action=np.asarray(sample(params,states[-1],jax.random.PRNGKey(p.CONFIG['native_action_seed']+t)))
        values=[sim.step(a) for sim,a in zip(sims,action)]
        states.append(np.stack([v[0][:8] for v in values])); rewards.append([v[1] for v in values]); actions.append(action)
        positions.append(np.stack([sim.env.state.copy() for sim in sims]))
        failures.append([sim.env.dead for sim in sims])
    assert all(sim.elapsed==50 for sim in sims)
    return dict(states=np.stack(states,1),position=np.stack(positions,1),action=np.stack(actions,1),
        reward=np.asarray(rewards,float).T,failure=np.asarray(failures,bool).T,reset_seed=reset_seeds)


def analyze(native_arrays,model_arrays):
    summaries={name:{key:float(v.mean()) for key,v in data.items()} for name,data in native_arrays.items()}
    comparisons={}; evidence=[]
    for seed in [0,1]:
        c=f'C_s{seed}'; b=f'B_s{seed}'
        for label,a,ref in [(f'{c}_minus_A',c,'A'),(f'{c}_minus_B',c,b),(f'{b}_minus_A',b,'A')]:
            comparisons[label]=paired(native_arrays[a],native_arrays[ref])
        endpoints=[comparisons[f'{c}_minus_A'],comparisons[f'{c}_minus_B']]
        evidence.append(all(row['returns']['ci95'][0]>0 and row['success']['ci95'][1]>=0 and row['failure']['ci95'][0]<=0 for row in endpoints))
    model_summaries={m:{a:{key:float(value.mean()) for key,value in data.items()} for a,data in actors.items()} for m,actors in model_arrays.items()}
    model_comparisons={m:{a:paired(data,actors['A']) for a,data in actors.items() if a!='A'} for m,actors in model_arrays.items()}
    return dict(native=summaries,native_comparisons=comparisons,model=model_summaries,
        model_vs_A=model_comparisons,consistent_native_C_benefit=all(evidence))


def report(out,result,training,ledger):
    ci=lambda x:'['+', '.join(f'{v:+.4f}' for v in x)+']'
    inherited=json.loads((p.PREVIOUS/'results.json').read_text())['models']
    title=('C improves native return over both A and B in both seeds under the predeclared rule.' if result['consistent_native_C_benefit'] else
        'The experiment does not establish a consistent native benefit from training against the pessimistic ETT.')
    lines=['# Frozen pessimistic ETT: native policy-improvement test','',title,'',
        'A is unchanged. B uses the final diagonal-fit head with both channels equal to the executed action; '
        'C uses the final critic-driven ETT with independent frozen nominal actions. Four training rounds per '
        'B/C actor alternate current-policy rollouts, averaged-NCE fitting and 25 guarded decoded-Q actor proposals. '
        'All actors use the same stochastic tanh-Gaussian execution convention, with paired noise. '
        'No native outcome or full-F4 hindsight actor objective enters training.','',
        '## Native outcomes: 256 fresh paired reset seeds','',
        'Return is sum(.95^t*r), not the normalized critic Q (.05 times return). Success uses distance<.5; '
        'failure is absorbing death, not merely lack of success. Intervals are 95% paired episode bootstrap.','']
    for name,row in result['native'].items():
        lines.append(f"- {name}: return {row['returns']:.4f}, success {row['success']:.3f}, failure {row['failure']:.3f}.")
    for label,rows in result['native_comparisons'].items():
        parts=[f"{k} {r['difference']:+.4f} {ci(r['ci95'])}" for k,r in rows.items()]
        lines.append('- '+label+': '+ '; '.join(parts)+'.')
    lines+=['','## Independent cross-model improvement versus A','',
        'Each cell uses 128 paired full rollouts from START, with the model acting at every transition. '
        'Rows below give actor return differences versus A on each frozen model. These are model outputs, not native outcomes.','']
    for model,actors in result['model_vs_A'].items():
        parts=[f"{a} {r['returns']['difference']:+.4f} {ci(r['returns']['ci95'])}" for a,r in actors.items()]
        lines.append('- '+model+': '+'; '.join(parts)+'.')
    lines+=['','## Policy changes, constraints and cost','',
        'KL/drift below use the fixed training state bank. They are empirical constraints, not global bounds. '
        'Every proposed update also checks current-rollout states and a .002 one-step KL limit.','']
    for name,row in training['actors'].items():
        steps=[s for r in row['rounds'] for s in r['steps']]; change=row['final_bank_change']
        lines.append(f"- {name}: {sum(s['accepted'] for s in steps)}/100 accepted; initial-to-final KL {change[0]:.6f}, "
            f"mean-action RMS/max L2 {change[1]:.5f}/{change[2]:.5f}, mean scale change {change[3]:+.5f}, "
            f"parameter L2 change {row['parameter_change_norm']:.5f}.")
    for seed in [0,1]:
        b=inherited[f'diagonal_s{seed}']['diagonal_degradation']; c=inherited[f'critic_s{seed}']['diagonal_degradation']
        lines.append(f"- Inherited seed {seed} validation diagonal-energy degradation: B {b['mean']:+.6f}, C {c['mean']:+.6f}; "
            'the prior .02 allowance and action-Lipschitz tests passed for the underlying ETTs.')
    lines+=['',f"Budgets: {ledger.data['charged']:,}/500,000 model transitions; 64,000/64,000 native steps; "
        '10,800 NCE steps and 400 actor proposals. Transition models and nominal policy remain unchanged. '
        'All five actors are sealed before any native or cross-model evaluation; final checkpoints only.','',
        '## Limits','',
        'Cross-model gains can reflect model-specific approximation or exploitation, and do not establish native '
        'improvement. B and C inherit different fitted diagonal heads as well as response maps. Their permitted '
        'diagonal-fit degradation, partial-observation F4 state and generator approximation remain limitations. '
        'The action-Lipschitz construction for C conditions on x_prime; it is not a bound on B when both channels vary.', '',
        'An ETT trained against A is not asserted to remain worst-case for an updated actor. Two paired training '
        'seeds and descriptive, non-simultaneous intervals do not establish universal benefit or equivalence. '
        'Intervals crossing zero are inconclusive; observed ties do not prove equivalence. '
        'No ETT retraining, extra sweeps, automatic extensions or push followed.','',
        'Reproduce with a fresh directory: `python -m ett.pointmaze_native_policy prepare --out <dir>` then '
        '`python -m ett.pointmaze_native_policy run --out <dir>`. Read-only audit: '
        '`python -m ett.check_pointmaze_native_policy --out <dir>`.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def evaluate(out,net,initial,actors,generator,ledger,context):
    training=json.loads((out/'training.json').read_text()); hashes={a:tree_sha(params) for a,params in actors.items()}
    write(out/'evaluation_started.json',dict(utc=average.timestamp(),training_sha256=sha(out/'training.json'),
        final_actor_hashes=hashes))
    sample=jax.jit(lambda params,state,key:p.sample_actor(net,params,state,key))
    reset_seeds=np.arange(256)+p.CONFIG['native_reset_seed']; native_arrays={}; steps=0
    for name,params in actors.items():
        steps+=256*50; assert steps<=p.CONFIG['native_cap']
        write(out/'native_ledger.json',dict(charged=steps,cap=64000,current_actor=name))
        r=native(params,sample,reset_seeds); native_arrays[name]=metrics(r,True)
        np.savez_compressed(out/f'{name}_native.npz',**{k:v for k,v in r.items() if k!='failure'})
        np.savez_compressed(out/f'{name}_native_audit.npz',failure=r['failure'])
        print(name+': 256 paired native episodes complete.',flush=True)
    model_arrays={}
    for model,(theta,observational) in p.models().items():
        model_arrays[model]={}
        for name,params in actors.items():
            ledger.scope='cross_'+model+'_'+name; ledger.add(128*50,'full_rollouts')
            r=generator.rollout(params,theta,observational,np.tile(p.START,(128,1)),
                jax.random.PRNGKey(p.CONFIG['model_evaluation_seed']),50)
            np.savez_compressed(out/f'{model}_{name}_model.npz',**r)
            model_arrays[model][name]=metrics(r)
        print(model+': all five final actors cross-evaluated.',flush=True)
    result=analyze(native_arrays,model_arrays); write(out/'results.json',result)
    assert steps==64000 and ledger.data['charged']==423168
    for name,params in actors.items(): assert tree_sha(params)==hashes[name]
    report(out,result,training,ledger)
