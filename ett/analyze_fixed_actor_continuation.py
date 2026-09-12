"""Predeclared clustered analysis of fixed-actor continuations; no fitting."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ett.run_fixed_actor_continuation import PROTOCOL, PRIOR, SOURCES, verify_run
from ett.run_return_readout import read, write, NAMES
from ett.return_readout import regression_metrics
from ett.fixed_actor_continuation import return_summary
from scripts.make_swamp_f4_failure_bank import file_sha


def reliable_pairs(returns, pairs):
    a,b=pairs.T
    mean=returns.mean(1);v=returns.var(1,ddof=1)/returns.shape[1]
    se=np.sqrt(v[a]+v[b]);den=(v[a]**2+v[b]**2)/(returns.shape[1]-1)
    df=np.divide((v[a]+v[b])**2,den,out=np.full(len(a),np.inf),where=den>0)
    margin=np.maximum(.5,student_t.ppf(.975,df)*se)
    delta=mean[a]-mean[b]
    return np.abs(delta)>margin, np.abs(delta)>1e-10, margin


def credit(delta, change):
    return (delta*change>0).astype(float)+.5*((change==0)|(delta==0))


def interval(values):
    return np.quantile(values,[.025,.975]).tolist() if len(values) else None


def analyze(root):
    verify_run(root)
    completion=read(root/'completion.json')
    assert completion['complete'] and file_sha(root/'continuations.npz')==completion['continuation_sha256']
    c=dict(np.load(root/'contexts.npz'));samples=dict(np.load(root/'continuations.npz'))
    pred=dict(np.load(root/'predictions_before_outcomes.npz'));pairsets=dict(np.load(root/'pairs.npz'))
    returns=samples['returns'];target=returns.mean(1);n,r=returns.shape;B=PROTOCOL['statistics']['bootstrap_replicates']
    rng=np.random.default_rng(PROTOCOL['seeds']['bootstrap'])
    nested=returns[np.arange(n)[None,:,None],rng.integers(r,size=(B,n,r))].mean(2)
    parents=np.unique(c['parent']);groups=np.array([np.flatnonzero(c['parent']==p) for p in parents])
    assert groups.shape==(len(parents),2)
    draw=groups[rng.integers(len(parents),size=(B,len(parents)))].reshape(B,-1)
    boot_target=np.take_along_axis(nested,draw,axis=1)
    masks={'all':np.ones(n,bool),'outside_goal':c['progress']!=2,'inside_goal':c['progress']==2,
           'within_observable_support':~c['support_outside'],'outside_observable_support':c['support_outside']}
    for key,labels in [('source',SOURCES),('region',['before','swamp','after','detour']),
                       ('progress',['far','approaching','inside']),('motion',['stationary','slow','moving'])]:
        for i,label in enumerate(labels): masks[f'{key}_{label}']=c[key]==i
    population={k:dict(contexts=int(m.sum()),parents=len(np.unique(c['parent'][m])),
                       mean_returns=return_summary(target[m]),individual_returns=return_summary(returns[m]))
                for k,m in masks.items() if m.any()}
    write(root/'return_population.json',population)
    ci=np.quantile(nested,[.025,.975],axis=0)
    np.savez_compressed(root/'context_return_estimates.npz',mean=target,se=returns.std(1,ddof=1)/np.sqrt(r),interval95=ci)
    metrics={};boot_errors={}
    for name,p in pred.items():
        calibrated=not name.endswith('_raw')
        row=regression_metrics(target,p,calibrated=calibrated)
        row['by_stratum']={k:regression_metrics(target[m],p[m],calibrated=calibrated) for k,m in masks.items()}
        ranks=[]
        for j in range(B):
            ranks.append(float(spearmanr(boot_target[j],p[draw[j]]).statistic) if np.ptp(p[draw[j]])>0 else 0.)
        row['spearman_interval95']=interval(ranks) if np.ptp(p)>0 else None
        if calibrated:
            difference=p[draw]-boot_target
            mae=np.mean(np.abs(difference),axis=1);rmse=np.sqrt(np.mean(difference**2,axis=1))
            boot_errors[name]=rmse
            row.update(mae_interval95=interval(mae),rmse_interval95=interval(rmse),
                       mean_prediction=float(p.mean()),bias=float(np.mean(p-target)))
        row['pairwise']={};metrics[name]=row
    pairing={};boot_pairs={};point_pairs={}
    for key,pairs in pairsets.items():
        a,b=pairs.T;reliable,unequal,margin=reliable_pairs(returns,pairs)
        detail=dict(pairs=len(pairs),covered_parents=len(np.unique(c['parent'][pairs])),
                    equal_mean_pairs=int((~unequal).sum()),unequal_mean_pairs=int(unequal.sum()),
                    reliable_pairs=int(reliable.sum()),uncertain_unequal_pairs=int((unequal&~reliable).sum()),
                    reliable_margin_quantiles=np.quantile(margin,[0,.5,1]).tolist(),
                    action_l2_difference_quantiles=np.quantile(np.linalg.norm(c['first_action'][a]-c['first_action'][b],axis=1),[0,.5,.95,1]).tolist(),
                    old_frame_l2_difference_quantiles=np.quantile(np.linalg.norm(c['observation'][a,2:8]-c['observation'][b,2:8],axis=1),[0,.5,.95,1]).tolist(),
                    xy_difference_max=float(np.max(np.linalg.norm(c['observation'][a,:2]-c['observation'][b,:2],axis=1))),
                    time_difference_max=int(np.max(np.abs(c['time'][a]-c['time'][b]))))
        assert len(np.unique(c['parent'][pairs]))==2*len(pairs)
        pairing[key]=detail
        np.savez_compressed(root/f'{key}_pair_labels.npz',pairs=pairs,reliable=reliable,unequal=unequal,margin=margin)
        for subset,valid in [('unequal',unequal),('reliable',reliable)]:
            aa,bb=a[valid],b[valid];K=len(aa)
            if K:
                pairdraw=rng.integers(K,size=(B,K))
                deltas=np.take_along_axis(nested[:,aa]-nested[:,bb],pairdraw,axis=1)
                signs=target[aa]-target[bb]
                flip=float(np.mean(np.sign(nested[:,aa]-nested[:,bb])!=np.sign(signs)))
            else: flip=None
            detail[subset+'_nested_order_flip_fraction']=flip
            for name,p in pred.items():
                val=credit(target[aa]-target[bb],p[aa]-p[bb]) if K else np.array([])
                boot=(credit(deltas,(p[aa]-p[bb])[pairdraw]).mean(1) if K else np.array([]))
                metrics[name]['pairwise'][key+'_'+subset]=dict(count=K,accuracy=float(val.mean()) if K else None,interval95=interval(boot))
                boot_pairs[key,subset,name]=boot;point_pairs[key,subset,name]=float(val.mean()) if K else None
    comparisons={}
    contrasts=[]
    for seed in [0,1]:
        a=f'alpha0p3_seed{seed}_ridge';b=f'alpha0_seed{seed}_ridge'
        contrasts.append((f'negative_minus_no_negative_s{seed}',a,b))
        for base in ['raw_input_ridge','raw_mlp_s110','raw_mlp_s111',f'alpha0p3_seed{seed}_raw']:
            contrasts.append((f'{a}_minus_{base}',a,base))
    for label,a,b in contrasts:
        record={}
        if a in boot_errors and b in boot_errors:
            record['rmse_difference']=dict(value=metrics[a]['rmse']-metrics[b]['rmse'],interval95=interval(boot_errors[a]-boot_errors[b]))
        for key in pairsets:
            for subset in ['unequal','reliable']:
                left,right=point_pairs[key,subset,a],point_pairs[key,subset,b]
                record[key+'_'+subset]=dict(accuracy_difference=left-right if left is not None else None,
                    interval95=interval(boot_pairs[key,subset,a]-boot_pairs[key,subset,b]))
        comparisons[label]=record
    write(root/'metrics.json',metrics);write(root/'pairing_results.json',pairing);write(root/'paired_comparisons.json',comparisons)
    make_report(root,c,samples,pred,metrics,population,pairing,comparisons)


def make_report(root,c,samples,pred,metrics,population,pairing,comparisons):
    labels=['Alpha 0\nseed 0','Alpha 0\nseed 1','Alpha .3\nseed 0','Alpha .3\nseed 1','Raw ridge','Raw MLP\n110','Raw MLP\n111']
    names=['alpha0_seed0_ridge','alpha0_seed1_ridge','alpha0p3_seed0_ridge','alpha0p3_seed1_ridge','raw_input_ridge','raw_mlp_s110','raw_mlp_s111']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,['rmse','mae']):
        values=np.array([metrics[n][key] for n in names]);ci=np.array([metrics[n][key+'_interval95'] for n in names])
        ax.bar(range(7),values)
        # Percentile nested intervals need not contain the point estimate.
        ax.vlines(range(7),ci[:,0],ci[:,1],color='black')
        ax.axhline(metrics['constant_training_mean'][key],ls='--',color='gray',label='Original training mean')
        ax.set(xticks=range(7),xticklabels=labels,ylabel=key.upper()+' (return units)',title='Fresh fixed-actor continuation')
        ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
    fig.savefig(root/'prediction_errors.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,key in zip(axes,['matched_unequal','matched_reliable']):
        for i,n in enumerate(names):
            record=metrics[n]['pairwise'][key]
            if record['accuracy'] is not None:
                ax.plot(i,record['accuracy'],'o');ax.vlines(i,*record['interval95'])
        ax.axhline(.5,color='gray',ls='--');ax.set(xticks=range(7),xticklabels=labels,ylim=(0,1),ylabel='Ordering accuracy',title=f'{key}: {metrics[names[0]]["pairwise"][key]["count"]} pairs')
        ax.tick_params(axis='x',labelsize=8)
    fig.savefig(root/'matched_ordering.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].hist(samples['returns'].mean(1),bins=20);axes[0].set(xlabel='Mean of 24 actual continuation returns',ylabel='Contexts')
    axes[1].scatter(c['observation'][:,0],c['observation'][:,1],c=samples['returns'].mean(1),s=12,vmin=0,vmax=12.83)
    axes[1].set(xlabel='X',ylabel='Y',title='Fresh naturally reached contexts');fig.savefig(root/'context_population.png',dpi=150);plt.close(fig)
    # Representatives fixed by context index, never by scores or return ranks.
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    ids=np.linspace(0,len(c['parent'])-1,8,dtype=int)
    for ax,i in zip(axes.flat,ids):
        for path in samples['xy'][i,:4]:ax.plot(path[:,0],path[:,1],alpha=.7)
        ax.scatter(*c['observation'][i,:2],marker='x',c='black');ax.scatter(8.5,3.5,marker='*',c='orange')
        ax.set(xlim=(0,9),ylim=(0,5),title=f'Context {i}, {SOURCES[c["source"][i]]}',xlabel='X',ylabel='Y')
    fig.savefig(root/'representative_continuations.png',dpi=150);plt.close(fig)
    p=population['all'];matched=pairing['matched'];support=read(root/'support_before_outcomes.json')
    rows=['# Independent fixed-actor continuation evaluation','',
        'This is a single bounded transfer evaluation. All encoders, actors and readouts remain frozen; no new targets fit any model. The evaluated returns come from the actual native PointMaze simulator, not an ETT rollout.','',
        '## Protocol and provenance','',
        'The protocol was saved before collecting any new trajectories. The historical default alpha0_seed0 final actor (150,000 updates) was chosen because it was already designated by the distribution-matching/rollout experiments, not because of new returns. It uses the unchanged stochastic tanh-Gaussian sampling helper. Every context has one first action sampled once, scored by all models, then held fixed across 24 continuations. Later actor draws and future environment RNG streams vary. Goal (8.5,3.5), horizon 20 and discount 0.95 are fixed.','',
        'All four original contrastive checkpoints (alpha 0/.3, seeds 0/1), five ridge readouts, two raw MLP readouts, normalization and prior predictions were hash-verified. The original readout replay check was run before collection. Feature extraction again verifies phi/psi against the native critic. Alpha is the negative-loss mixture weight; the original failure_bank_f4_r60d40.npz has 256 references. There is no alpha .1 or .5 arm. Exact hashes are in provenance.json.','',
        '384 independent reset seeds produce 30-step natural prefixes (96 parents each: historical actor, blind shortcut waypoints, visible detour waypoints, uniform actions). Coverage controllers use visible positions only, with declared speeds/noise/pauses. Candidates at t=4,8,12,16,20,24,28,30 are sampled two per parent using inverse observable-stratum frequencies and 4x outside-goal weight. No scorer, future return, hidden death flag or mask affects selection. Complete F4 histories come from actual environment steps, with no teleportation.','',
        'The native environment delegates its 50-step limit to the caller. TimedSimulator stores the external elapsed time and deep-copies the entire native environment, including geometry, all configuration, frame history, persistent absorption, current bits and RNG. Restoring with a new future RNG does not call reset or resample current bits. Bits redraw naturally at each subsequent step; absorption persists. Death is an absorbing zero-reward state with done=False, not an early time-limit truncation. Repeats are conditional on the SAME full snapshot, not hidden-state posterior samples.','',
        'Every continuation completes exactly 20 steps without crossing t=50. Actual reward is checked every step against visible fixed-goal reconstruction and against the audited implementation after collection. That equivalence holds only for this goal: lethal positions cannot lie within its radius-2 reward region. Action bounds, fixed first-action execution, F4 shifts and fixed goal are asserted. Tests check exact replay (including RNG), changed future RNG with unchanged persistent initial state, absorbing behavior, horizon boundaries and indexing.','',
        f'Budget: 11,520 collection steps plus 368,640 continuation steps; at most 2,048 focused-check steps, total cap 382,208. Collection stops at the declared budget regardless of unequal-pair counts. Actual main-run steps: {read(root/"completion.json")["environment_steps"]:,}. Native simulation is used because no reusable vectorized PointMaze simulator was found; actor inference is batched.','',
        '## Population and applicability','',
        f'Mean context return: {p["mean_returns"]["mean"]:.4f}, standard deviation {p["mean_returns"]["std"]:.4f}. Context-mean zero/max fractions: {p["mean_returns"]["zero_fraction"]:.2%}/{p["mean_returns"]["maximum_fraction"]:.2%}. Individual continuation zero/max fractions: {p["individual_returns"]["zero_fraction"]:.2%}/{p["individual_returns"]["maximum_fraction"]:.2%}. Full predeclared region, progress, motion, source and support strata are in return_population.json and metrics.json.','',
        '![Population](context_population.png)','',
        f'Observable support uses the original raw-ridge training normalization, 16,384 fixed training reference rows and 2,048 disjoint calibration rows. {support["outside_reference_threshold_fraction"]:.2%} of new inputs exceed the old calibration 95th-percentile nearest-reference distance ({support["threshold"]:.4f}). Coordinate-range violations, action saturation and F4 motion distributions are reported in support_before_outcomes.json. No support statistic changes selection or fitting. Proximity in visible inputs cannot establish hidden-state or continuation-policy support.','',
        'Fresh RNG trajectories were never used to train the original encoder or readouts. They can revisit the same visible states, which is expected in this small maze. The new parent-policy mixture differs from the logged mixture, and continuation always uses the single fixed actor. Transfer errors therefore combine continuation-policy and context-distribution shifts; they do not isolate either mechanism.','',
        '## Frozen prediction and ranking results','']
    for n in names:
        m=metrics[n];v=m['pairwise']['matched_reliable']
        rows.append(f'- {n}: MAE {m["mae"]:.4f}, RMSE {m["rmse"]:.4f} (95% interval {m["rmse_interval95"]}), Spearman {m["spearman"]:.4f}; reliable matched ordering {v["accuracy"]}, interval {v["interval95"]}.')
    rows+=['',f'Original training-mean baseline RMSE: {metrics["constant_training_mean"]["rmse"]:.4f}. Raw scalar logits are ranking-only, with null MAE/RMSE; no calibration or clipping is introduced. All raw-scalar, unmatched ordering and conditional results are in metrics.json.','',
        '![Errors](prediction_errors.png)','',
        f'Matching is fixed before outcomes: exact source, goal and XY cell, XY distance <=.25, goal-distance gap <=.25, time gap <=3. A random anchor per parent may match another candidate of an unused parent; each parent occurs in at most one pair. There are {matched["pairs"]} matched pairs, {matched["equal_mean_pairs"]} equal estimated-mean pairs, {matched["unequal_mean_pairs"]} unequal pairs and only {matched["reliable_pairs"]} meeting the predeclared uncertainty-and-effect-size rule.','',
        'A reliable label requires |mean difference| > max(0.5, Welch 97.5% t critical value times repeat SE). Exact equal means within 1e-10 are excluded, prediction ties receive half credit. These labels are provisional finite-repeat estimates, not simultaneous significance tests or proof that rare outcomes cannot occur. Twenty-four identical outcomes can underestimate uncertainty about rare events. All unequal-mean and conservative-subset results are shown separately; no tolerance changed after outcomes.','',
        f'Remaining matched first-action L2 gaps (min/median/95%/max): {matched["action_l2_difference_quantiles"]}; old F4-frame L2 gaps: {matched["old_frame_l2_difference_quantiles"]}. Actions/history are intentionally not equated; these are comparable visible contexts, not identical hidden states.','',
        '500 nested bootstrap replicates resample whole parents with both contexts together and repeats within each context. Paired model comparisons reuse the same resamples. Pair intervals resample disjoint two-parent blocks and continuation repeats, conditional on the fixed observed reliable subset; the subset is not adaptively reselected. Nested ordering-flip rates are reported. This conditions on trained models, the chosen actor and fixed matching, and does not capture checkpoint-training or matching uncertainty. MAE/RMSE compare predictions to Monte Carlo means, so residual target noise remains.','',
        '![Matched ordering](matched_ordering.png)','',
        'Paired alpha=.3 minus alpha=0 comparisons (positive ordering difference favors .3; negative RMSE difference favors .3):','']
    for seed in [0,1]: rows.append(f'- Seed {seed}: {comparisons[f"negative_minus_no_negative_s{seed}"]}')
    rows+=['','## Answers and stopping decision','',
        'The final interpretation must distinguish broad return prediction from comparable-context ordering, and transfer from the original logged-policy task. The numeric summaries above and paired_comparisons.json are the basis for the accompanying conclusion; no model is selected from these outcomes.','',
        'This evaluation does not establish conditional ETT ground truth, intervention identification, robust/worst-case Q or safe optimization over generated states. A snapshot fixes latent state instead of integrating the latent posterior given the observation. A new alpha=.5 arm would require matched training and evaluation and is not run here.','',
        '![Preselected representatives](representative_continuations.png)','',
        '## Reproduction','',
        'From the PointMaze worktree, use the same Python environment as the prior readout report. The commands below use a fresh output directory; checkpoints and simulator snapshots remain local. Configurations, observable evaluation arrays, metrics, plots and reports are publishable.','',
        '```bash','python -m scripts.check_return_readout --run-dir artifacts/return_readout/f4_h20_g095_v1',
        'python -m unittest scripts.test_fixed_actor_continuation',
        'python -m ett.run_fixed_actor_continuation --out-dir artifacts/fixed_actor_continuation/f4_h20_g095_v1 --phase prepare',
        'python -m ett.run_fixed_actor_continuation --out-dir artifacts/fixed_actor_continuation/f4_h20_g095_v1 --phase collect',
        'python -m ett.run_fixed_actor_continuation --out-dir artifacts/fixed_actor_continuation/f4_h20_g095_v1 --phase run',
        'python -m ett.analyze_fixed_actor_continuation --run-dir artifacts/fixed_actor_continuation/f4_h20_g095_v1','```','']
    (root/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')
    write(root/'analysis_manifest.json',dict(source_sha256=file_sha(__file__),files={p.name:file_sha(p) for p in root.iterdir() if p.suffix in ('.json','.npz','.png','.md') and p.name!='analysis_manifest.json'}))
    print('Analysis complete; matched pairing:',pairing['matched'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    analyze(Path(parser.parse_args().run_dir))
