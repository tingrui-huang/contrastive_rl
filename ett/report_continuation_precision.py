"""Single final analysis of the fixed-model diagnostic, using saved draws only."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import t as student
from ett import pointmaze_continuation_precision as d
from ett.finite_crl import write,sha

def estimate(x,q,mask=None,level=.95):
    """Stratified query variance; ratio influence for conditional subpopulations."""
    x=np.asarray(x);mask=np.ones(len(x),bool) if mask is None else np.asarray(mask)
    a=q['mass'];den=float(a[mask].sum())
    if den==0:return None
    y=x.mean(1) if x.ndim==2 else x
    value=float(np.sum(a*mask*y)/den);influence=mask*(y-value)/den
    parts=[];degrees=[]
    for cell in np.unique(q['cell']):
        ids=q['cell']==cell;n=int(ids.sum())
        parts.append(a[ids].sum()**2*influence[ids].var(ddof=1)/n);degrees.append(n-1)
    parts=np.array(parts);variance=float(parts.sum())
    df=float(variance**2/np.sum(parts**2/np.array(degrees))) if variance>0 else None
    half=float(student.ppf((1+level)/2,df)*np.sqrt(variance)) if variance>0 else 0.
    result=dict(mean=value,ci=[value-half,value+half],half_width=half,level=level,
        se=float(np.sqrt(variance)),df=df,target_mass=den,queries=int(mask.sum()),
        zero_variance=variance==0,
        effective_design_queries=float(den**2/np.sum((a*mask)**2)),
        effective_visitation_queries=float(np.sum(a*mask*q['weight'])**2/np.sum((a*mask*q['weight'])**2)))
    if x.ndim==2:
        result['conditional_successor_mc_se']=float(np.sqrt(np.sum((a*mask/den)**2*x.var(1,ddof=1)/x.shape[1])))
    return result

def classify(rows):
    eps=d.CONFIG['tolerance'];target=d.CONFIG['precision']
    c,m,e=[rows[k] for k in ['critic','mc','mc_minus_critic']]
    within=lambda r:r['ci'][0]>=-eps and r['ci'][1]<=eps
    direction=lambda r:-1 if r['ci'][1]<-eps else 1 if r['ci'][0]>eps else 0
    if direction(e):return 'meaningful critic error'
    if within(e) and direction(c)!=0 and direction(c)==direction(m):return 'agreement at a meaningful effect'
    if all(within(r) and not r['zero_variance'] for r in [c,m,e]) and max(m['half_width'],e['half_width'])<=target:
        return 'precisely small effect within tolerance'
    return 'insufficient precision to resolve the declared categories'

def analyze_pair(out,name):
    folder=out/name;q=dict(np.load(folder/'queries.npz'));xs={k:[] for k in ['critic','mc','mc_minus_critic','sequence_changed']}
    critic=d.load_critic(name);ctx=dict(np.load(d.p.phase.PREVIOUS/'contexts.npz'))
    first_changes=0;continuation_changes=0;count=0;successor_changes=0
    for batch in range(9):
        r=dict(np.load(folder/f'batch{batch}.npz'));idx=r['query_index'];n=len(idx)
        np.testing.assert_array_equal(idx,np.arange(batch*256,(batch+1)*256))
        np.testing.assert_array_equal(r['continuation_theta'],r['reference_theta'])
        np.testing.assert_array_equal(r['reference_theta'][:16],r['candidate_theta'][:16])
        np.testing.assert_array_equal(r['reference_x_prime'],r['candidate_x_prime'])
        for label in ['reference','candidate']:
            rewards=r[label+'_continuation_rewards'];h=q['h'][idx]-1
            assert not rewards[np.broadcast_to(np.arange(48)>=h[:,None,None,None],rewards.shape)].any()
            mcq=.05*np.sum(rewards*.95**np.arange(48),axis=-1)
            np.testing.assert_allclose(mcq,r[label+'_mc_q'],atol=1e-12)
            decoded=critic.predict(np.repeat(r[label+'_successor'].reshape(-1,8),2,0),
                r[label+'_next_action'].reshape(-1,2),np.repeat(h,16),ctx['mean'],ctx['std']).reshape(n,8,2)
            np.testing.assert_allclose(decoded,r[label+'_q'],atol=1e-12)
            for key,values in [('mc',mcq),('critic',decoded)]:
                val=q['weight'][idx,None,None]*(.05*r[label+'_reward'][:,:,None]+.95*values)
                np.testing.assert_allclose(val,r[label+'_'+key],atol=1e-12)
        c=(r['candidate_critic']-r['reference_critic']).mean(-1)
        m=(r['candidate_mc']-r['reference_mc']).mean(-1)
        first=r['candidate_reward']!=r['reference_reward']
        later=(r['candidate_continuation_rewards']!=r['reference_continuation_rewards']).any(-1)
        changed=first[:,:,None]|later
        for key,val in [('critic',c),('mc',m),('mc_minus_critic',m-c),('sequence_changed',changed.mean(-1))]:xs[key].append(val)
        first_changes+=int(first.sum());continuation_changes+=int(later.sum());count+=int(changed.sum())
        successor_changes+=int((r['candidate_successor']!=r['reference_successor']).any(-1).sum())
    xs={k:np.concatenate(v) for k,v in xs.items()}
    overall={k:estimate(v,q,level=.975) for k,v in xs.items()}
    conditional={}
    masks={**{f'time_{b}':q['bin']==i for i,b in enumerate(d.CONFIG['bins'])},
        'pre_entry':q['pre_entry'],'entered':~q['pre_entry']}
    masks.update({f'{g}/time_{b}':(q['group']==i)&(q['bin']==j)
        for i,g in enumerate(d.p.early.GROUPS) for j,b in enumerate(d.CONFIG['bins'])})
    for label,mask in masks.items():
        if mask.any():conditional[label]={k:estimate(v,q,mask) for k,v in xs.items()}
    old=d.p.read(d.BASE/'results.json')['terminal'][name]['final_minus_start']
    result=dict(classification=classify(overall),overall=overall,conditional=conditional,
        precision_target_met=all(overall[k]['half_width']<=.005 for k in ['mc','mc_minus_critic']),
        counts=dict(independent_reference_paths=len(q['h']),queries=len(q['h']),fixed_roots=len(np.unique(q['root'])),
            unique_states=len(np.unique(q['state'],axis=0)),
            unique_state_actions=len(np.unique(np.c_[q['state'],q['action']],axis=0)),
            pre_entry=int(q['pre_entry'].sum()),in_region=int(q['in_region'].sum()),
            successor_pairs=len(q['h'])*8,changed_successors=successor_changes,
            paired_continuations=len(q['h'])*16,changed_sequences=count,
            changed_first_rewards=first_changes,changed_continuation_sequences=continuation_changes),
        published_full_rollout_difference=old)
    write(folder/'analysis.json',result)
    return result

def interval(r):return f"{r['mean']:+.5f} [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}]"

def run(out):
    d.verify(out)
    if (out/'results.json').exists():raise ValueError('Final analysis already completed')
    done=d.p.read(out/'completion.json');assert done['computed_transitions']==d.CONFIG['expected']
    for f,h in done['samples'].items():assert sha(out/f)==h,f
    d.torch.set_num_threads(1)
    results={name:analyze_pair(out,name) for name in d.CONFIG['pairs']}
    categories=[r['classification'] for r in results.values()]
    if 'meaningful critic error' in categories:
        decision='Prioritize continuation-value generalization and calibration before spending more on response optimization: at least one fixed reference critic has a practically meaningful contrast error against matched MC.'
    elif all(c=='precisely small effect within tolerance' for c in categories):
        decision='Prioritize the gap between one-step surrogate changes and repeated full-model use. Both tested local contrasts are precisely small at the declared scale; more optimizer budget is not justified by these diagnostics alone.'
    elif 'agreement at a meaningful effect' in categories:
        decision='For the resolved comparison, continuation accuracy is supported at a meaningful effect size. Prioritize the local-to-full objective relationship before expanding response optimization; the other comparison retains its reported uncertainty.'
    else:
        decision='The fixed budget does not resolve the declared continuation-accuracy categories. Neither another optimization sweep nor critic retraining is justified as the dominant remedy by these data; first examine the saved stratum contributions and objective alignment.'
    lines=['# Fixed-model continuation-value diagnostic','',decision,'',
        'Candidate minus reference throughout. Overall intervals are 97.5% per pair (95% across the two pairs separately for each metric). '
        'Time/prefix intervals below are descriptive 95%. Units are the original weighted, normalized one-step surrogate; negative means more pessimistic. '
        'Tolerance is ±0.01; target half-width is 0.005. Root/time design weights recover the original uniform-root, uniform-time visitation target.','']
    for name,r in results.items():
        v=r['overall'];counts=r['counts']
        lines += [f'## {name}_u24 minus {name}_u0','',f"**{r['classification']}.**",
            f"Critic {interval(v['critic'])}; MC {interval(v['mc'])}; MC-minus-critic {interval(v['mc_minus_critic'])}.",
            f"MC / discrepancy half-widths: {v['mc']['half_width']:.5f} / {v['mc_minus_critic']['half_width']:.5f}; precision target met: {r['precision_target_met']}.",
            f"Conditional successor-MC SEs: {v['mc']['conditional_successor_mc_se']:.5f} / {v['mc_minus_critic']['conditional_successor_mc_se']:.5f}. These exclude fresh-query uncertainty; main intervals include it.",
            f"{counts['queries']} independent paths/queries on 36 fixed roots; {counts['unique_states']} unique states, {counts['unique_state_actions']} unique state/actions. "
            f"{counts['pre_entry']} queries before first goal entry; {counts['in_region']} currently inside the reward region. "
            f"Kish effective query counts: design {v['mc']['effective_design_queries']:.1f}, design × visitation {v['mc']['effective_visitation_queries']:.1f}.",
            f"Changed reward sequences: {counts['changed_sequences']}/{counts['paired_continuations']} paired continuations (first or later reward); "
            f"first rewards {counts['changed_first_rewards']}/{counts['successor_pairs']}, later sequences {counts['changed_continuation_sequences']}/{counts['paired_continuations']}. "
            f"Original-target weighted sequence-change rate {100*v['sequence_changed']['mean']:.3f}%. "
            f"Changed full-F4 successors: {counts['changed_successors']}/{counts['successor_pairs']}.",'']
        for label in [f'time_{b}' for b in d.CONFIG['bins']]+['pre_entry','entered']:
            if label not in r['conditional']:continue
            s=r['conditional'][label];m=s['mc']
            lines.append(f"- {label}: n={m['queries']}, target mass={m['target_mass']:.4f}; critic {interval(s['critic'])}; MC {interval(m)}; discrepancy {interval(s['mc_minus_critic'])}; weighted sequence-change rate {100*s['sequence_changed']['mean']:.3f}%.")
        full=r['published_full_rollout_difference']
        lines+=['',f"Published full-model return difference: {full['mean']:+.5f}, root 95% CI [{full['root_ci95'][0]:+.5f}, {full['root_ci95'][1]:+.5f}]. "
            'This reuses independent held-out-root rollouts with each ETT applied every step. It is context, not a matched one-step target or evidence of critic error by itself.','']
    lines += ['## Scope, provenance and verification','',
        'Each reference critic was loaded from its saved _u1 checkpoint, after its 1,000-step reference-model fit and before response update 1. '
        'Checkpoint bytes, parameter hashes, preupdate theta, fitted-path probability hashes, and saved mean/std provenance were verified. '
        'The earlier reporting-only source correction is explicitly verified. See provenance.json for exact paths and hashes. '
        'Both MC branches use reference u0 after their respective first successors; critic and MC share successor/next-action arrays. '
        'The final audit re-decodes all saved Q values and reconstructs masked MC returns and visitation-weighted objectives.','',
        f"Computed transitions: {done['computed_transitions']:,}/{done['cap']:,}, including padded slots. Zero actor, critic, nominal or ETT updates and zero native steps. "
        'Diagonal offsets and paired diagonal samples remain exact; response constraints and frozen parameter hashes pass. '
        'Root-group-by-time conditional results, estimated pre-entry masses, effective counts and all uncertainty details are in results.json.','',
        'Intervals condition on these 36 existing roots; they do not cover root-population generalization. Repeated time-zero states have independently sampled actions, '
        'and successor/actor draws remain grouped within queries. Conditional zero-variance strata do not prove equivalence, particularly with rare reward changes. '
        'These two large endpoint contrasts test continuation generalization, not every historical optimization update. '
        'They cannot establish global worst-case recovery, universal action equivalence or native-policy benefit. No new full-model evaluation, precision extension, or sweep was run.','']
    write(out/'results.json',dict(comparisons=results,decision=decision))
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'audit.json',dict(passed=True,source_and_sample_hashes_verified=True,all_q_decoding_recomputed=True,
        all_mc_and_surrogate_values_recomputed=True,remaining_horizon_masks_verified=True,
        continuation_reference_theta_verified=True,diagonal_and_frozen_hashes_verified=True))
    write(out/'manifest.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='manifest.json'})
    print('\n'.join(lines[:22]),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
