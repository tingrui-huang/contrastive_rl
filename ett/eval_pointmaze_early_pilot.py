"""Coverage audits and independent variable-horizon model evaluation."""
import json
from pathlib import Path
import jax
import numpy as np
import torch

from ett import pointmaze_early_pilot as p
from ett import pointmaze_region_pilot as old
from ett.eval_pointmaze_region_pilot import constraints
from ett.finite_crl import write,sha


def native_audit(dataset,indices,include_future=False):
    """Read hidden evidence separately; never return it to ETT/critic fitting."""
    episodes=indices[:,0];times=indices[:,1]
    with np.load(dataset) as z:
        obs=z['obs'][episodes];bits=z['swamp_bits'][episodes];died=z['entered_active_swamp'][episodes].astype(bool)
    death=np.full(len(episodes),-1,int);cells=((3,3),(4,3),(5,3))
    for e in range(len(episodes)):
        if not died[e]:continue
        for t in range(50):
            cell=tuple(np.floor(obs[e,t+1,:2]).astype(int))
            if cell in cells and bits[e,t,cells.index(cell)]:death[e]=t+1;break
        assert death[e]>0
    dead=(death>=0)&(death<=times)
    audit=dict(dead_when_observed=dead,death_row=death)
    if include_future:
        values=[];hits=[]
        for i,t in enumerate(times):
            s=obs[i,t+1:51,:8];r=np.asarray(old.task_reward(s,np.broadcast_to(old.GOAL,s.shape)))
            values.append(float(.05*r @ .95**np.arange(50-t)));hits.append(bool(r.any()))
            if dead[i]:assert not r.any()
        audit.update(native_logged_value=np.asarray(values),native_logged_goal=np.asarray(hits),
            native_dies_after_root=(death>times))
    return audit


def coverage(out,data):
    selection=json.loads((out/'selection.json').read_text())
    rows={};passed=selection['passed'];dataset=json.loads((out/'preregistration.json').read_text())['dataset']
    assert not np.intersect1d(data['train_indices'][:,0],data['validation_indices'][:,0]).size
    arrays={}
    for part in ['train','validation']:
        indices=data[part+'_indices'];roots=data[part+'_roots']
        assert len(np.unique(indices[:,0]))==len(indices)
        audit=native_audit(dataset,indices)
        arrays.update({part+'_'+k:v for k,v in audit.items()})
        rows[part]={}
        for gi,group in enumerate(p.GROUPS):
            selected=indices[:,2]==gi;n=int(selected.sum());alive=int((~audit['dead_when_observed'][selected]).sum())
            assert p.group_mask(roots[selected],group).all()
            rows[part][group]=dict(count=n,alive_at_root=alive,already_dead=n-alive,
                times=indices[selected,1].tolist(),episodes=indices[selected,0].tolist(),
                mean_current_step=float(np.linalg.norm(roots[selected,:2]-roots[selected,2:4],axis=1).mean()) if n else None)
            passed &= n==12 and alive>=9
        passed &= (~audit['dead_when_observed']).sum()>=30
    np.savez_compressed(out/'coverage_audit_only.npz',**arrays)
    write(out/'coverage.json',dict(passed=bool(passed),groups=rows,
        selection_sha256=sha(out/'selection.json'),hidden_labels_used_for_resampling=False,
        policy='Aggregate stop-only gate; no individual exclusions or training labels.'))
    print('Coverage gate: '+str(bool(passed)),flush=True)
    return bool(passed)


def path_statistics(records,n,repeats):
    values=np.zeros((n,repeats));recovered=np.zeros((n,repeats),bool);motion=np.zeros((n,repeats),bool)
    # NaN marks absent time slots; rewards/returns never use padded slots.
    states=np.full((n,repeats,50,8),np.nan,np.float32)
    actions=np.full((n,repeats,49,2),np.nan,np.float32);xp=actions.copy()
    horizons=np.zeros(n,int)
    for r in records:
        h=r['action'].shape[1]
        for root in np.unique(r['root']):
            ids=np.flatnonzero(r['root']==root)
            assert len(ids)==repeats
            paths=r['states'][ids];reward=r['reward'][ids]
            values[root]=.05*reward @ .95**np.arange(h)
            states[root,:,:h+1]=paths;actions[root,:,:h]=r['action'][ids];xp[root,:,:h]=r['x_prime'][ids];horizons[root]=h
            step=np.linalg.norm(np.diff(paths[...,:2],axis=1),axis=-1);motion[root]=np.any(step>1e-6,axis=1)
            distance=np.linalg.norm(paths[...,:2]-old.GOAL[:2],axis=-1)
            for k in range(repeats):
                hit=np.flatnonzero(reward[k]>0)
                if not len(hit):continue
                first=int(hit[0])+1
                if first<5:continue
                dist=distance[k,:first]
                setback=np.any(dist-np.minimum.accumulate(dist)>=.25)
                still=step[k,:first-1]<=1e-6
                waited=len(still)>=3 and np.any(np.convolve(still.astype(int),np.ones(3,int),'valid')==3)
                recovered[root,k]=bool(setback or waited)
    return dict(returns=values,recovered=recovered,any_motion=motion,states=states,
        executed_actions=actions,auxiliary_x_prime=xp,horizons=horizons)


def summarize_paths(arrays,groups):
    result={}
    for gi,name in enumerate(p.GROUPS):
        mask=groups==gi;v=arrays['returns'][mask];h=arrays['horizons'][mask]
        result[name]=dict(count=int(mask.sum()),model_value=float(v.mean()) if len(v) else None,
            zero_return_fraction=float(np.mean(v==0)) if len(v) else None,
            goal_reaching_fraction=float(np.mean(v>0)) if len(v) else None,
            recovered_after_setback_fraction=float(arrays['recovered'][mask].mean()) if len(v) else None,
            any_motion_fraction=float(arrays['any_motion'][mask].mean()) if len(v) else None,
            maximum_possible_occupancy_mean=float(np.mean(1-.95**h)) if len(v) else None)
    return result


def generator_gate(engine,out,data,ledger):
    roots=data['validation_roots'];indices=data['validation_indices'];n=len(roots)
    probe_arrays=[];rows={};diagonal_reference=None
    for name,scale in [('initial',0.),('positive_response',.35),('negative_response',-.35)]:
        theta=np.r_[np.zeros(16),np.tile((scale*np.eye(2)).ravel(),8)].astype(np.float32)
        records=p.collect(engine,theta,roots,indices[:,1],p.CONFIG['generator_seed'],ledger,'generator_'+name,16)
        arrays=path_statistics(records,n,16);probe_arrays.append(arrays)
        np.savez_compressed(out/('generator_'+name+'.npz'),**arrays)
        check=constraints(engine,theta,data,ledger)
        s=data['validation_state'][:16];a=data['validation_action'][:16]
        ledger.add(128,'probe_exact_diagonal_invariance')
        y,_=engine.sample(theta,s,a,a,jax.random.PRNGKey(p.CONFIG['generator_seed']+100),8)
        if diagonal_reference is None:diagonal_reference=np.asarray(y)
        else:np.testing.assert_array_equal(diagonal_reference,y)
        rows[name]=dict(groups=summarize_paths(arrays,indices[:,2]),constraints=check,
            recovered_paths=int(arrays['recovered'].sum()))
    mixed=np.zeros(n,bool)
    for a in probe_arrays:
        hits=(a['returns']>0).sum(1)
        mixed |= (hits>=2)&(hits<=14)
    mixed_counts={g:int(mixed[indices[:,2]==i].sum()) for i,g in enumerate(p.GROUPS)}
    recovered=sum(int(a['recovered'].sum()) for a in probe_arrays)
    std=float(probe_arrays[0]['returns'].mean(1).std())
    passed=(mixed_counts['approach']>=2 and mixed_counts['transit']>=2 and recovered>=8 and std>=.025
        and all(r['constraints']['passed'] for r in rows.values()))
    write(out/'generator_gate.json',dict(passed=passed,probes=rows,mixed_roots=mixed_counts,
        recovered_path_count=recovered,initial_between_context_std=std,
        interpretation='Finite probe evidence only; no full-family capacity certificate. Zero return is not death.'))
    print('Generator gate: '+str(passed),flush=True)
    return passed


def calibration_metrics(critic,arrays,mean,std):
    rows=[]
    for phase in ['root','midpoint','last']:
        s,a,h,groups,values=[arrays[phase+'_'+k] for k in ['state','action','h','group','returns']]
        prediction=critic.predict(s,a,h,mean,std);arrays[phase+'_prediction']=prediction
        for gi,name in [(-1,'all'),*enumerate(p.GROUPS)]:
            mask=np.ones(len(s),bool) if gi==-1 else groups==gi
            delta=prediction[mask]-values[mask].mean(1)
            rows.append(dict(phase=phase,group=name,count=int(mask.sum()),rmse=float(np.sqrt(np.mean(delta**2))),
                bias=float(delta.mean()),target_mean=float(values[mask].mean()),
                max_target_mc_se=float((values[mask].std(1,ddof=1)/np.sqrt(values.shape[1])).max()),
                range_excess=float(max(0.,(prediction[mask]-(1-.95**h[mask])).max()))))
    passed=all(r['rmse']<=(.06 if r['group']=='all' else .07) and abs(r['bias'])<=(.025 if r['group']=='all' else .035) for r in rows)
    return dict(passed=passed,rows=rows)


def calibrate(engine,theta,critic,data,seed,ledger,purpose):
    roots=data['validation_roots'];indices=data['validation_indices'];n=len(roots);times=indices[:,1]
    query_paths=p.collect(engine,theta,roots,times,seed,ledger,purpose+'_query',1)
    arrays={}
    for phase_id,phase in enumerate(['root','midpoint','last']):
        state=np.empty_like(roots);remaining=np.empty(n,int)
        for r in query_paths:
            h=r['action'].shape[1];t=0 if phase=='root' else h//2 if phase=='midpoint' else h-1
            ids=r['root'];state[ids]=r['states'][:,t];remaining[ids]=h-t
        action=np.asarray(engine.actor_jit(state,jax.random.PRNGKey(seed+1000+phase_id)))
        paths=p.collect(engine,theta,state,50-remaining,seed+2000+100*phase_id,ledger,purpose+'_'+phase,16,action)
        packed=path_statistics(paths,n,16)
        arrays.update({phase+'_'+k:v for k,v in dict(state=state,action=action,h=remaining,group=indices[:,2],returns=packed['returns']).items()})
        if phase=='root':arrays.update(packed)
    metrics=calibration_metrics(critic,arrays,data['mean'],data['std'])
    return metrics,arrays


def interval(difference,episodes,groups=None):
    rng=np.random.default_rng(p.CONFIG['bootstrap_seed']);difference=np.asarray(difference)
    if groups is not None:
        assert len(np.unique(episodes))==len(episodes)
        samples=[]
        for gi in np.unique(groups):
            ids=np.flatnonzero(groups==gi);samples.append(difference[rng.choice(ids,(2000,len(ids)),replace=True)])
        boot=np.concatenate(samples,1).mean(1)
    else:
        ids,inverse=np.unique(episodes,return_inverse=True)
        sums=np.bincount(inverse,weights=difference);counts=np.bincount(inverse)
        draws=rng.integers(len(ids),size=(2000,len(ids)))
        boot=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(difference.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),episodes=int(len(np.unique(episodes))))


def final_evaluation(engine,out,data,ledger):
    training=json.loads((out/'training.json').read_text());results={};saved={}
    for name in ['initial']+[r['name'] for r in training['records']]:
        theta=np.load(out/'checkpoints'/(name+'.npz'))['theta']
        critic=p.Critic(0);critic.load_state_dict(torch.load(out/'checkpoints'/(name+'.pt'),weights_only=True)['model'])
        metrics,arrays=calibrate(engine,theta,critic,data,p.CONFIG['evaluation_seed'],ledger,name+'_evaluation')
        fresh=p.Critic(p.CONFIG['critic_seed']+100);opt=torch.optim.Adam(fresh.parameters(),lr=.003)
        records=p.collect(engine,theta,data['train_roots'],data['train_indices'][:,1],p.CONFIG['evaluation_seed']+10000,ledger,'final_refit')
        loss=old.fit(fresh,opt,p.positives(records,p.CONFIG['evaluation_seed']+11000,data['mean'],data['std']),1000,p.CONFIG['evaluation_seed']+11001)
        refit_arrays=dict(arrays);refit=calibration_metrics(fresh,refit_arrays,data['mean'],data['std'])
        for phase in ['root','midpoint','last']:arrays[phase+'_refit_prediction']=refit_arrays[phase+'_prediction']
        torch.save(dict(model=fresh.state_dict()),out/'checkpoints'/(name+'_refit.pt'))
        es=old.diagonal(engine,theta,data,'validation',np.arange(512),32,p.CONFIG['evaluation_seed']+12000,ledger,'final_diagonal')
        nll=np.asarray(engine.nll(theta,data['validation_state'],data['validation_action'],data['validation_target']))
        assert np.isfinite(nll).all() and np.isfinite(es).all()
        arrays.update(diagonal_es=es,preprojection_nll=nll)
        np.savez_compressed(out/(name+'_evaluation.npz'),**arrays);saved[name]=arrays
        results[name]=dict(model_value=float(arrays['returns'].mean()),groups=summarize_paths(arrays,data['validation_indices'][:,2]),
            last_critic=metrics,independent_refit=refit,refit_nce=loss,diagonal_es=float(es.mean()),nll=float(nll.mean()),
            constraints=constraints(engine,theta,data,ledger),theta_sha256=sha(out/'checkpoints'/(name+'.npz')),
            last_critic_sha256=sha(out/'checkpoints'/(name+'.pt')),refit_sha256=sha(out/'checkpoints'/(name+'_refit.pt')))
        print(name+': independent evaluation complete',flush=True)
    comparisons={};indices=data['validation_indices']
    for seed in [0,1]:
        joint=f'joint_s{seed}';control=f'diagonal_s{seed}';comparisons[joint]={}
        for ref in [control,'initial']:
            a,b=saved[joint],saved[ref]
            np.testing.assert_array_equal(a['root_action'],b['root_action'])
            diff=(a['returns']-b['returns']).mean(1)
            groups={g:interval(diff[indices[:,2]==i],indices[indices[:,2]==i,0]) for i,g in enumerate(p.GROUPS)}
            overall=interval(diff,indices[:,0],indices[:,2])
            es=interval(a['diagonal_es']-b['diagonal_es'],data['validation_episode'])
            comparisons[joint][ref]=dict(overall=overall,groups=groups,diagonal=es,diagonal_passed=es['ci95'][1]<=.02)
        comparisons[joint]['effect_established']=bool(results[joint]['last_critic']['passed'] and results[joint]['independent_refit']['passed']
            and results[joint]['constraints']['passed'] and comparisons[joint][control]['overall']['ci95'][1]<0
            and all(comparisons[joint][r]['diagonal_passed'] for r in [control,'initial']))
    write(out/'results.json',dict(models=results,comparisons=comparisons))


def finish(out,data,ledger,stage):
    p.verify(out)
    if stage!='complete':write(out/'training.json',dict(status='stopped_'+stage,updates=0,records=[]))
    dataset=json.loads((out/'preregistration.json').read_text())['dataset']
    audit=native_audit(dataset,data['validation_indices'],include_future=True)
    np.savez_compressed(out/'native_evidence_audit_only.npz',**audit)
    native={}
    for i,g in enumerate(p.GROUPS):
        selected=data['validation_indices'][:,2]==i
        native[g]=dict(count=int(selected.sum()),already_dead=int(audit['dead_when_observed'][selected].sum()),
            later_native_death=int(audit['native_dies_after_root'][selected].sum()),
            logged_goal_count=int(audit['native_logged_goal'][selected].sum()),
            logged_value=float(audit['native_logged_value'][selected].mean()) if selected.any() else None)
    write(out/'native_evidence.json',dict(groups=native,interpretation='Logged continuation policy differs from fixed actor; not conditional ETT ground truth.'))
    lines=['# Early-context PointMaze region-NCE pilot','',f'Stopping stage: **{stage}**. Charged model outputs: **{ledger.data["charged"]:,}/{ledger.data["cap"]:,}**.','',
        'The frozen actor, state-goal nominal policy, shared 48-parameter convex generator, region reward and two losses are unchanged from 0059c98. Horizon is exactly 50 minus recorded root time; Q_h=(1-.95^h)*31*.5*exp(f_1), Q_0=0. The critic horizon feature is now h/50. Discounted visitation weights are H_i*.95^t.','',
        'Samplewise Euclidean action L=1 is preserved by the existing construction, with a common-x_prime Wasserstein coupling. No TV constraint, death penalty, classifier, or new geometric restriction was added. The inherited geometry and visible-only state do not guarantee native physical validity or persistent absorption.','',
        '## Predeclared gates','']
    coverage=json.loads((out/'coverage.json').read_text());lines += [f'Coverage passed: {coverage["passed"]}. Selection uses observable prefixes only, one root per episode, no resampling. Hidden labels only supplied the separate aggregate stop gate.','']
    for part,groups in coverage['groups'].items():
        for g,r in groups.items():lines += [f'- {part}/{g}: {r["count"]} roots; {r["alive_at_root"]} alive at observation; times {r["times"]}.']
    if (out/'generator_gate.json').exists():
        gate=json.loads((out/'generator_gate.json').read_text());lines += ['',f'Generator check passed: {gate["passed"]}. Mixed-outcome roots: {gate["mixed_roots"]}; recovered-after-setback paths: {gate["recovered_path_count"]}; initial root mean-value SD: {gate["initial_between_context_std"]:.5f}.','']
        for probe,r in gate['probes'].items():
            for g,row in r['groups'].items():lines += [f'- {probe}/{g}: model occupancy {row["model_value"]:.5f}, finite-horizon zero fraction {row["zero_return_fraction"]:.3f}, recovered-after-setback fraction {row["recovered_after_setback_fraction"]:.3f}.']
        lines += ['','These fixed probes are not an exhaustive family search or a capacity certificate. A zero-return path is not death. Recovery labels describe realized paths only and never train the critic.']
    if (out/'preflight.json').exists():
        pre=json.loads((out/'preflight.json').read_text());lines += ['',f'Initial critic calibration passed: {pre["passed"]}.','']
        for r in pre['rows']:lines += [f'- {r["phase"]}/{r["group"]}: RMSE {r["rmse"]:.5f}, bias {r["bias"]:.5f}, maximum MC SE {r["max_target_mc_se"]:.5f}, range excess {r["range_excess"]:.5f}.']
    if stage=='complete':
        results=json.loads((out/'results.json').read_text());lines += ['','## Independent final models','']
        for name,r in results['models'].items():
            lines += [f'- {name}: value {r["model_value"]:.6f}, diagonal ES {r["diagonal_es"]:.6f}, pre-projection NLL {r["nll"]:.6f}, constraints {r["constraints"]["passed"]}, last/refit calibration pass {r["last_critic"]["passed"]}/{r["independent_refit"]["passed"]}.']
            for g,row in r['groups'].items():lines += [f'  - {g}: value {row["model_value"]:.6f}, zero fraction {row["zero_return_fraction"]:.3f}, recovered fraction {row["recovered_after_setback_fraction"]:.3f}.']
        lines += ['','## Paired final differences','']
        for name,c in results['comparisons'].items():
            lines += [f'{name}: effect established under all declared gates = {c["effect_established"]}.','']
            for ref,row in c.items():
                if ref=='effect_established':continue
                lines += [f'- Versus {ref}: overall {row["overall"]}; diagonal ES {row["diagonal"]}; per-group differences {row["groups"]}.']
        lines += ['','All intervals use 2000 paired episode-cluster bootstrap replicates; overall return comparisons stratify by visible group. Repeats are averaged first. Group intervals are descriptive. Final refits use new Monte Carlo labels on the same frozen kernel and identical evaluation targets; they are never used for ETT updates.']
    lines += ['','## Native evidence and interpretation','']
    for g,row in native.items():lines += [f'- {g}: {row}']
    lines += ['','Native logged goal attainment, later death and finite-horizon zero reward are separate events. Logged behavior is not the fixed continuation actor. Hidden labels did not train any scorer or select checkpoints. No new native simulator evaluation was run. Model predictions and logged hidden-context evidence are not conditional ETT ground truth.','',
        ('The declared gate stopped before optimization. The remaining immediate limitation is '+stage.replace('_',' ')+'. No later optimization result is claimed.' if stage!='complete' else 'Interpret optimization effects using both the calibration/feasibility gates and group differences above. Lower model return does not establish death, globally optimal pessimism, or causal identification.'), '',
        'No budget expansion, actor training, loss change, extra experiment family, commit or push. Arrays retain every probe/evaluation trajectory; absent variable-horizon slots are explicitly NaN and never included in a return. Configurations, hashes and the pre-run protocol are saved beside this report.','',
        '## Reproduction','', '```powershell',
        'python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback scripts.test_pointmaze_region_pilot scripts.test_pointmaze_early_pilot -v',
        'python -m ett.pointmaze_early_pilot prepare --out artifacts/pointmaze_region_pilot/early_contexts_s01_v1',
        'python -m ett.pointmaze_early_pilot run --out artifacts/pointmaze_region_pilot/early_contexts_s01_v1',
        '```','', 'Use a fresh output directory for reproduction; the saved attempt is protected from overwrite. Local input checkpoints/dataset must match preregistration.json.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'completion.json',dict(stage=stage,charged=ledger.data['charged'],updates=24 if stage=='complete' else 0))
    write(out/'manifest.json',{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='manifest.json'})
    print('Finished: '+stage,flush=True)
