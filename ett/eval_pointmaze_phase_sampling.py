"""Independent calibration and noise-screened ordering for frozen critics."""
import json
from pathlib import Path
import jax
import numpy as np

from ett import pointmaze_phase_sampling as p
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write,sha


def bootstrap_indices(groups):
    rng=np.random.default_rng(p.CONFIG['bootstrap_seed']);columns=[]
    for g in np.unique(groups):
        ids=np.flatnonzero(groups==g);columns.append(rng.choice(ids,(2000,len(ids)),replace=True))
    return np.concatenate(columns,axis=1)


def calibration(prediction,returns,h,groups):
    target=returns.mean(1);error=prediction-target;idx=bootstrap_indices(groups)
    rmse=np.sqrt(np.mean(error[idx]**2,axis=1));bias=error[idx].mean(1)
    se=returns.std(1,ddof=1)/np.sqrt(returns.shape[1]);excess=np.maximum(prediction-(1-.95**h),-prediction)
    return dict(count=len(prediction),rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),
        rmse_ci95=np.quantile(rmse,[.025,.975]),bias_ci95=np.quantile(bias,[.025,.975]),
        range_violation_fraction=float(np.mean(excess>1e-6)),range_excess=float(max(0.,excess.max())),
        mc_se_mean=float(se.mean()),mc_se_max=float(se.max()),target_mean=float(target.mean()),
        target_noise_variance_mean=float(np.mean(se**2)))


def paired_calibration(alternative,baseline,returns,groups):
    target=returns.mean(1);a=alternative-target;b=baseline-target;idx=bootstrap_indices(groups)
    rmse=np.sqrt(np.mean(a[idx]**2,axis=1))-np.sqrt(np.mean(b[idx]**2,axis=1))
    bias=(a-b)[idx].mean(1)
    return dict(rmse_change=float(np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b))),
        rmse_change_ci95=np.quantile(rmse,[.025,.975]),bias_change=float((a-b).mean()),
        bias_change_ci95=np.quantile(bias,[.025,.975]),
        absolute_bias_change=float(abs(a.mean())-abs(b.mean())))


def ordering_pairs(returns,h):
    n=len(h);i,j=np.triu_indices(n,1);same=h[i]==h[j];i,j=i[same],j[same]
    mean=returns.mean(1);se=returns.std(1,ddof=1)/np.sqrt(returns.shape[1]);delta=mean[i]-mean[j]
    first=returns[:,:32].mean(1);second=returns[:,32:].mean(1)
    equal=np.abs(delta)<=1e-12
    informative=(np.abs(delta)>np.maximum(.01,2.58*np.sqrt(se[i]**2+se[j]**2)))
    informative &= (np.sign(first[i]-first[j])==np.sign(delta))&(np.sign(second[i]-second[j])==np.sign(delta))
    used=np.unique(np.r_[i[informative],j[informative]])
    return dict(i=i[informative],j=j[informative],label=np.sign(delta[informative]),
        all_same_h_pairs=len(i),equal_pairs=int(equal.sum()),unresolved_pairs=int((~equal&~informative).sum()),
        informative_pairs=int(informative.sum()),informative_episodes=len(used),
        adequate=bool(informative.sum()>=10 and len(used)>=8))


def ordering(prediction,pairs,other=None):
    i,j,label=pairs['i'],pairs['j'],pairs['label'];public={k:v for k,v in pairs.items() if k not in ['i','j','label']}
    if not len(i):return dict(**public,accuracy=None,ci95=None,paired_accuracy_change=None,paired_change_ci95=None)
    def credit(pred):
        difference=pred[i]-pred[j]
        return np.where(np.abs(difference)<=1e-12,.5,(np.sign(difference)==label).astype(float))
    correct=credit(prediction);n=len(prediction)
    draws=bootstrap_indices(np.zeros(n,int))
    counts=np.zeros((2000,n),int)
    for k in range(n):counts[:,k]=np.sum(draws==k,axis=1)
    weights=counts[:,i]*counts[:,j];denominator=weights.sum(1);valid=denominator>0
    boot=weights[valid]@correct/denominator[valid]
    result=dict(**public,accuracy=float(correct.mean()),ci95=np.quantile(boot,[.025,.975]),
        bootstrap_nonempty=int(valid.sum()),paired_accuracy_change=None,paired_change_ci95=None)
    if other is not None:
        delta=correct-credit(other);result.update(paired_accuracy_change=float(delta.mean()),
            paired_change_ci95=np.quantile(weights[valid]@delta/denominator[valid],[.025,.975]))
    return result


def query_records(records,n):
    result={}
    for phase_id,phase in enumerate(p.PHASES):
        states=np.empty((n,8),np.float32);actions=np.empty((n,2),np.float32);remaining=np.empty(n,int)
        for r in records:
            length=r['action'].shape[1];t=[0,5,length//2][phase_id]
            for root in np.unique(r['root']):
                k=np.flatnonzero(r['root']==root)[0]
                states[root]=r['states'][k,t];actions[root]=r['action'][k,t];remaining[root]=length-t
        result.update({phase+'_'+k:v for k,v in dict(state=states,action=actions,h=remaining).items()})
    return result


def target_continuations(engine,queries,seed,ledger,purpose):
    arrays=dict(queries);theta=np.zeros(48,np.float32)
    for phase_id,phase in enumerate(p.PHASES):
        state,action,h=[queries[phase+'_'+k] for k in ['state','action','h']]
        records=early.collect(engine,theta,state,50-h,seed+100*phase_id,ledger,purpose+'_'+phase,64,action)
        values=np.zeros((len(state),64))
        for r in records:
            length=r['action'].shape[1]
            np.testing.assert_array_equal(r['action'][:,0],action[r['root']])
            reward=np.asarray(old.task_reward(r['states'][:,1:],np.broadcast_to(old.GOAL,r['states'][:,1:].shape)))
            np.testing.assert_array_equal(reward,r['reward'])
            for root in np.unique(r['root']):
                selected=r['root']==root
                values[root]=.05*r['reward'][selected] @ .95**np.arange(length)
        arrays[phase+'_returns']=values
    return arrays


def collect_final_contexts(out):
    # Called only after the four final critic checkpoints have been sealed.
    from scripts.collect_swamp_windy_f4 import collect
    source=out/'local_inputs'/'fresh_context_source.npz'
    write(out/'native_collection_started.json',dict(seed=103000000,episodes=600,charged_native_steps=30000))
    collect('teacher',600,0.,.05,.15,103000000,str(source),False)
    with np.load(source) as z:obs=z['obs']  # No audit arrays or labels read.
    assert obs.shape==(600,51,16) and np.all(obs[:,:,8:]==old.GOAL)
    np.testing.assert_array_equal(obs[:,1:,2:8],obs[:,:-1,:6])
    indices,coverage=early.select_roots(obs,np.arange(600),103000001)
    roots=obs[indices[:,0],indices[:,1],:8]
    np.savez_compressed(out/'final_contexts.npz',roots=roots,indices=indices)
    write(out/'final_selection.json',dict(coverage=coverage,passed=all(r['selected']==12 for r in coverage.values()),
        native_source_sha256=sha(source),selected_contexts_sha256=sha(out/'final_contexts.npz'),
        namespace='fresh_native_103000000',hidden_fields_read=False,
        collection_after_training_sha256=sha(out/'training.json')))
    return roots,indices,all(r['selected']==12 for r in coverage.values())


def cohort_results(arrays,groups,models,mean,std):
    result={};differences={}
    for name,critic in models.items():
        result[name]={}
        for phase in p.PHASES:
            state,action,h,returns=[arrays[phase+'_'+k] for k in ['state','action','h','returns']]
            pred=critic.predict(state,action,h,mean,std);arrays[phase+'_'+name+'_prediction']=pred
            result[name][phase]={}
            for gi,group in [(-1,'all'),*enumerate(early.GROUPS)]:
                mask=np.ones(len(groups),bool) if gi<0 else groups==gi
                row=calibration(pred[mask],returns[mask],h[mask],groups[mask])
                if gi>=0:row['ordering']=ordering(pred[mask],ordering_pairs(returns[mask],h[mask]))
                result[name][phase][group]=row
    for seed in [0,1]:
        baseline=f'uniform_s{seed}';alternative=f'balanced_s{seed}';differences[str(seed)]={}
        for phase in p.PHASES:
            a=arrays[phase+'_'+alternative+'_prediction'];b=arrays[phase+'_'+baseline+'_prediction']
            returns=arrays[phase+'_returns'];h=arrays[phase+'_h'];differences[str(seed)][phase]={}
            for gi,g in [(-1,'all'),*enumerate(early.GROUPS)]:
                mask=np.ones(len(groups),bool) if gi<0 else groups==gi
                row=paired_calibration(a[mask],b[mask],returns[mask],groups[mask])
                if gi>=0:row['ordering']=ordering(a[mask],ordering_pairs(returns[mask],h[mask]),b[mask])
                differences[str(seed)][phase][g]=row
    return dict(models=result,paired=differences)


def old_diagnostic(models,context):
    old_arrays=dict(np.load(p.PREVIOUS/'preflight.npz'));result={}
    for name,critic in models.items():
        result[name]={}
        for phase in ['root','midpoint','last']:
            s,a,h,returns=[old_arrays[phase+'_'+k] for k in ['state','action','h','returns']]
            prediction=critic.predict(s,a,h,context['mean'],context['std'])
            if name=='uniform_s0':np.testing.assert_array_equal(prediction,old_arrays[phase+'_prediction'])
            result[name][phase]=calibration(prediction,returns,h,context['validation_indices'][:,2])
    return dict(note='Previously inspected diagnostic targets; not final evaluation. Original phase definitions retained.',models=result)


def evaluate(engine,out,context,training_paths,models,ledger):
    before={name:old.parameter_sha(model) for name,model in models.items()}
    diagnostic=old_diagnostic(models,context);write(out/'old_diagnostic.json',diagnostic)
    queries=query_records(training_paths,36)
    train=target_continuations(engine,queries,p.CONFIG['train_evaluation_seed'],ledger,'seen_training_queries')
    train['episode']=context['train_indices'][:,0];train['group']=context['train_indices'][:,2]
    seen=cohort_results(train,train['group'],models,context['mean'],context['std'])
    np.savez_compressed(out/'training_calibration.npz',**train);write(out/'training_calibration.json',seen)
    roots,indices,covered=collect_final_contexts(out)
    if not covered:
        write(out/'completion.json',dict(status='stopped_final_coverage',model_outputs=ledger.data['charged'],native_steps=30000))
        (out/'REPORT.md').write_text('# Final coverage insufficient\n\nTraining completed with fixed budgets, but the fixed600-episode final collection did not supply12 observable roots per group. No replacement collection or final tuning occurred. See final_selection.json and training_calibration.json.\n',encoding='utf-8')
        return
    records=early.collect(engine,np.zeros(48,np.float32),roots,indices[:,1],p.CONFIG['final_query_seed'],ledger,'fresh_final_query_paths',1)
    p.save_records(out/'final_query_paths.npz',records)
    final=target_continuations(engine,query_records(records,36),p.CONFIG['final_evaluation_seed'],ledger,'untouched_final_queries')
    final['episode']=indices[:,0];final['group']=indices[:,2]
    fresh=cohort_results(final,final['group'],models,context['mean'],context['std'])
    np.savez_compressed(out/'final_calibration.npz',**final);write(out/'final_calibration.json',fresh)
    for name,model in models.items():assert old.parameter_sha(model)==before[name]
    primary=[]
    for seed in [0,1]:
        a=f'balanced_s{seed}';b=f'uniform_s{seed}';row=fresh['paired'][str(seed)]['early']['all']
        train_row=seen['paired'][str(seed)]['early']['all']
        supports=(row['rmse_change_ci95'][1]<0 and row['absolute_bias_change']<0 and train_row['rmse_change']<0)
        primary.append(dict(seed=seed,supports_sampling_bottleneck=bool(supports),final_early=row,
            seen_early=train_row,later_harm_flag={phase:bool(fresh['paired'][str(seed)][phase]['all']['rmse_change_ci95'][1]>.01) for phase in ['middle','late']}))
    write(out/'conclusion.json',dict(primary=primary,both_seeds_support=all(r['supports_sampling_bottleneck'] for r in primary),
        warning='Calibration improvement alone does not establish useful within-group ordering. No ETT optimization or causal identification.'))
    write(out/'completion.json',dict(status='complete',model_outputs=ledger.data['charged'],native_steps=30000,
        optimizer_steps=6000,ett_updates=0,actor_updates=0))
    report(out,seen,fresh,primary,ledger)
    write(out/'manifest.json',{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'})


def report(out,seen,fresh,primary,ledger):
    training=json.loads((out/'training.json').read_text())
    lines=['# Frozen PointMaze critic: input-row phase sampling','',
        'Only the training input-row sampler differs between arms. The generator, actor, nominal sampler, reward, architecture, normalization, binary NCE loss, 1:31 class weighting and Q decoding are unchanged. No ETT or actor update occurred.','',
        f'Four critics completed1500 steps each. All share identical288 training trajectories and geometric positive labels. Baseline seed0 exactly reproduces the previous failed critic. Model outputs: {ledger.data["charged"]:,}/650,000; fresh native context collection:30,000/30,000 steps.','',
        'Early/middle/late training phases are [0,5), [5,H//2), [H//2,H). Calibration queries are their starting rows, with64 independent same-generator targets and a fixed first action. Groups describe the original root, including later generated queries. Values are normalized .05*sum(.95^t*r), not raw logits.','',
        '## Training exposures and empirical fit','']
    for name,r in training['models'].items():
        counts=np.asarray(r['row_exposures']);lines += [f'- {name}: early/middle/late exposures {counts.sum(1).tolist()}, final minibatch NCE {r["final_minibatch_nce"]:.6f}.']
    lines += ['','Full-row empirical NCE by phase/group is in training.json. It is fit to sampled Bernoulli labels, not an exact-value error. The following seen-query calibration instead uses independent continuations at actual training state/action rows.','']
    for label,result in [('Seen training queries',seen),('Untouched final episodes',fresh)]:
        lines += ['## '+label,'']
        for name,phases in result['models'].items():
            lines += [name+':','']
            for phase,groups in phases.items():
                r=groups['all'];lines += [f'- {phase}: RMSE {r["rmse"]:.5f} (95% CI {np.asarray(r["rmse_ci95"]).round(5).tolist()}), bias {r["bias"]:.5f} (CI {np.asarray(r["bias_ci95"]).round(5).tolist()}), range excess {r["range_excess"]:.5f}, range violation fraction {r["range_violation_fraction"]:.3f}, maximum MC SE {r["mc_se_max"]:.5f}.']
                for g in early.GROUPS:
                    c=groups[g];order=c['ordering'];lines += [f'  - {g}: RMSE {c["rmse"]:.5f}, bias {c["bias"]:.5f}; informative same-h pairs {order["informative_pairs"]}/{order["all_same_h_pairs"]}, equal {order["equal_pairs"]}, unresolved {order["unresolved_pairs"]}; accuracy {order["accuracy"]}, adequate ordering sample {order["adequate"]}.']
            lines += ['']
    lines += ['## Paired primary comparison','']
    for row in primary:
        r=row['final_early'];t=row['seen_early']
        lines += [f'- Seed{row["seed"]}: final early balanced-minus-uniform RMSE {r["rmse_change"]:.5f}, CI {np.asarray(r["rmse_change_ci95"]).round(5).tolist()}; absolute-bias change {r["absolute_bias_change"]:.5f}; seen-query early RMSE change {t["rmse_change"]:.5f}. Sampling-bottleneck criterion: {row["supports_sampling_bottleneck"]}; later-harm flags {row["later_harm_flag"]}.']
    lines += ['','All metric/paired/ordering intervals are episode-clustered, with2000 replicates and group stratification for overall metrics. Target repeats are averaged first; raw RMSE includes target MC noise. Ranking matches exact remaining horizon within group and phase, requires a .01 and2.58-SE gap plus split-half sign agreement, and never uses scorer predictions to construct labels. Pair-bootstrap weights use endpoint episode multiplicities. Empty or underpowered cells do not establish ordering. These are descriptive intervals and approximate noise screens, not simultaneous confidence guarantees.','',
        '## Scope and stopping','',
        'The old evaluation set is reported only in old_diagnostic.json. Final contexts come from600 new native episodes generated after all four critics were frozen, using the verified teacher solely for context collection. The teacher has native privileged information to act; selection reads observable prefixes only, and critic training never reads hidden labels. Final value targets come from the frozen generator, not the logged teacher or native hidden-state continuations. The actor/nominal/diagonal model had no exposure to these new episodes, but model-side structural limitations still apply.','',
        'A change in the row distribution can support a sampling explanation only to the extent shown by both seeds and seen/fresh comparisons. It does not isolate architecture, finite-label noise or critic optimizer effects as unique causes. Do not infer useful ranking from aggregate RMSE or identify the true environment worst case. The diagonal law and samplewise Euclidean action-Lipschitz construction are unchanged exactly because no generator parameters changed.','',
        'Stop after this fixed comparison: no retuning, budget extension, actor/ETT training, loss change, additional experiment family or automatic push. Checkpoints and raw native input archives remain local. Shared labels, exposures, selected contexts, predictions, return samples, paired metrics, provenance and reports are saved.','',
        '## Reproduction','', '```powershell',
        'python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback scripts.test_pointmaze_region_pilot scripts.test_pointmaze_early_pilot scripts.test_pointmaze_phase_sampling -v',
        'python -m ett.pointmaze_phase_sampling prepare --out artifacts/pointmaze_region_pilot/phase_sampling_s01_v1',
        'python -m ett.pointmaze_phase_sampling run --out artifacts/pointmaze_region_pilot/phase_sampling_s01_v1',
        '```','', 'Use a fresh output directory. Exact local input hashes are listed in preregistration.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
