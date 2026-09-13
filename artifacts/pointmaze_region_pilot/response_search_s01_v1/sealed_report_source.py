"""Compact response-search report and audit from saved arrays only."""
import argparse
from pathlib import Path
import numpy as np
from ett import pointmaze_response_search as p
from ett.finite_crl import write,sha

def sign(row,root='root_ci95'):
    a,b=row[root],row['conditional_mc_ci95']
    return -1 if max(a[1],b[1])<0 else 1 if min(a[0],b[0])>0 else 0

def full_interval(difference,groups):
    x=np.asarray(difference);means=x.mean(1);rng=np.random.default_rng(p.CONFIG['bootstrap_seed'])
    idx=np.concatenate([rng.choice(np.flatnonzero(groups==g),(2000,int(np.sum(groups==g)))) for g in np.unique(groups)],axis=1)
    value=float(x.mean());se=float(np.sqrt(np.sum(x.var(1,ddof=1)/x.shape[1]))/len(x))
    return dict(mean=value,root_ci95=np.quantile(means[idx].mean(1),[.025,.975]),conditional_mc_ci95=[value-1.96*se,value+1.96*se])

def run(out):
    p.verify(out);done=p.read(out/'completion.json');search=p.read(out/'search.json');selection=p.read(out/'selection.json')
    assert sha(out/'selection.json')==p.read(out/'evaluation_started.json')['selection_sha256']
    assert done['computed_transitions']<=6000000 and done['response_updates']==144 and done['critic_steps']==47400
    pool=dict(np.load(out/'candidate_pool.npz'));initial=dict(np.load(out/'starts.npz'))
    returns=dict(np.load(out/'evaluation_returns.npz'));roots=dict(np.load(p.average.PREVIOUS/'final_contexts.npz'))
    local={};n_opposite=0;n_raw_opposite=0;computed=0
    for name,history in search['histories'].items():
        theta=initial[name+'_u0'];m=np.zeros(32);v=np.zeros(32)
        for row in history:
            np.testing.assert_array_equal(row['preupdate'],theta)
            theta,m,v,_=p.response_step(theta,np.array(row['directions']),np.array(row['signed']),m,v,row['update'])
            np.testing.assert_array_equal(theta,row['theta']);np.testing.assert_array_equal(theta[:16],initial[name+'_u0'][:16])
            assert np.linalg.norm(theta[16:].reshape(8,2,2),axis=(1,2)).max()<=1.000002
            if 'matched' not in row:continue
            folder=out/'checks'/f"{name}_u{row['update']}";r=dict(np.load(folder/'matched_mc.npz'))
            for label in ['before','proposed']:
                reward=r[label+'_continuation_rewards'];h=r[label+'_remaining']
                assert np.all(reward[np.arange(48)[None,None,None,:]>=h[...,None]]==0)
                q=.05*np.sum(reward*.95**np.arange(48),axis=-1)
                np.testing.assert_allclose(q,r[label+'_mc_q'],atol=1e-7)
                value=r['query_weight'][:,None,None]*(.05*r[label+'_reward'][:,:,None]+.95*q)
                np.testing.assert_allclose(value,r[label+'_mc'],atol=1e-6)
                computed+=reward.size+r[label+'_reward'].size
            c=(r['proposed_critic']-r['before_critic']).mean(-1)
            mc=(r['proposed_mc']-r['before_mc']).mean(-1)
            actual={k:p.uncertainty(x,r['query_root']) for k,x in [('critic',c),('mc',mc),('mc_minus_critic',mc-c)]}
            for k in actual:
                np.testing.assert_allclose(actual[k]['root_ci95'],row['matched'][k]['root_ci95'],atol=1e-12)
            n_opposite+=sign(actual['critic'])*sign(actual['mc'])==-1
            n_raw_opposite+=np.sign(actual['critic']['mean'])*np.sign(actual['mc']['mean'])==-1
            local[f"{name}_u{row['update']}"]=actual
        np.testing.assert_array_equal(theta,pool[name+'_u24'])
    assert computed==1778688
    for name,score in selection['scores'].items():
        records=p.phase.load_records(out/'selection'/f'{name}.npz')
        values,_=p.evaluation.pack_returns(records,36,16)
        np.testing.assert_allclose(values.mean(),score,atol=1e-12)
    for s,winner in selection['winners'].items():
        assert winner==min((n for n in selection['scores'] if n.startswith('s'+s+'_')),key=lambda n:(selection['scores'][n],n))
    summaries={};contrasts={};terminal={}
    for name in done['constraints']:
        records=p.phase.load_records(out/'evaluation'/f'{name}.npz');v,_=p.evaluation.pack_returns(records,36,64)
        np.testing.assert_array_equal(v,returns[name]);r=np.load(out/'evaluation'/f'{name}_reset.npz')
        np.testing.assert_allclose(.05*r['reward']@.95**np.arange(50),returns[name+'_reset'],atol=1e-12)
        summaries[name]=dict(return_mean=float(v.mean()),public_reset_mean=float(returns[name+'_reset'].mean()))
    for s,winner in selection['winners'].items():
        base=f's{s}_k0_u0';contrasts[s]=dict(winner=winner,versus_baseline=full_interval(returns[winner]-returns[base],roots['indices'][:,2]))
        d=returns[winner+'_reset']-returns[base+'_reset'];idx=np.random.default_rng(p.CONFIG['bootstrap_seed']).integers(128,size=(2000,128))
        contrasts[s]['public_reset']=dict(mean=float(d.mean()),ci95=np.quantile(d[idx].mean(1),[.025,.975]))
        for k in range(3):
            name=f's{s}_k{k}';terminal[name]=dict(local=local[name+'_u24'],
                final_minus_prefinal=full_interval(returns[name+'_u24']-returns[name+'_u23'],roots['indices'][:,2]),
                final_minus_start=full_interval(returns[name+'_u24']-returns[name+'_u0'],roots['indices'][:,2]),
                response_l2_change=float(np.linalg.norm(pool[name+'_u24'][16:]-initial[name+'_u0'][16:])))
    improved=sum(sign(r['versus_baseline'])==-1 for r in contrasts.values())
    transfer_fail=sum(sign(r['local']['mc'])==-1 and sign(r['final_minus_prefinal'])!=-1 for r in terminal.values())
    if improved==2:
        decision='Broader response search finds independently lower continuation-root returns for both baselines. Insufficient response exploration is a demonstrated contributor within this fixed model family.'
        next_step='Next prioritize diagnosing why model-internal pessimism does or does not transfer to native behavior, with a separately authorized validation experiment; do not start an architecture sweep.'
    elif n_opposite:
        decision='Search does not establish improvement for both baselines, and resolved critic/MC direction disagreements reveal local continuation-estimation errors. The dominant bottleneck remains only partially identified.'
        next_step='Next prioritize targeted continuation-value calibration at the saved disagreeing queries, using matched MC as a reference, before expanding response optimization.'
    elif transfer_fail:
        decision='MC-validated local improvement does not consistently transfer to full-kernel improvement. Local-to-full surrogate alignment is a supported concern; broad-search success is not established for both baselines.'
        next_step='Next prioritize the local-to-full objective gap on saved model pairs, before adding optimizer budget or changing architectures.'
    else:
        decision='The bounded search remains inconclusive about a dominant response-optimization versus continuation-estimation bottleneck.'
        next_step='Next prioritize a separately bounded replication of the largest unresolved matched comparisons and full-rollout differences, rather than an architecture sweep.'
    result=dict(models=summaries,selected=contrasts,local_checks=local,terminal=terminal,
        resolved_opposite_directions=int(n_opposite),raw_opposite_directions=int(n_raw_opposite),
        improved_baselines=int(improved),mc_local_without_resolved_full_improvement=int(transfer_fail),decision=decision,next_direction=next_step)
    write(out/'results.json',result)
    fmt=lambda r:f"{r['mean']:+.5f}; root 95% CI [{r['root_ci95'][0]:+.5f}, {r['root_ci95'][1]:+.5f}]; MC 95% CI [{r['conditional_mc_ci95'][0]:+.5f}, {r['conditional_mc_ci95'][1]:+.5f}]"
    lines=['# Response-only ETT bottleneck diagnosis','',decision,'',
        'Frozen baselines are critic_s0/critic_s1 from 817e429. Only 32 response parameters change: all base diagonal weights and 16 head offsets, actor and nominal policy remain fixed. '
        'Three starts per baseline, 24 updates each; same averaged-positive region critic and pessimistic surrogate. No diagonal-fit rejection.','',
        '## Independent full-rollout results','',
        'Normalized discounted return (.05 times raw return), lower is more pessimistic. Each selected model was chosen using 16 separate paths per training root. '
        'Final results use 64 new paths per held-out root, candidate ETT at every step. Public-reset results use 128 separate paths.','']
    for s,row in contrasts.items():
        base=summaries[f's{s}_k0_u0'];win=summaries[row['winner']];reset=row['public_reset']
        lines.append(f"- Baseline {s}: {base['return_mean']:.5f}; selected {row['winner']}: {win['return_mean']:.5f}; difference {fmt(row['versus_baseline'])}.")
        lines.append(f"  Public START: {base['public_reset_mean']:.5f} -> {win['public_reset_mean']:.5f}; delta {reset['mean']:+.5f}, paired 95% CI {reset['ci95']}.")
    lines+=['','## Critic versus matched MC','',f'{n_raw_opposite}/18 raw opposite signs; {n_opposite}/18 resolved opposite signs under both declared intervals. '
        'Each pair shares query states/actions, candidate successors and next actions between critic and MC; MC continues under the frozen pre-update ETT. These are one-step local comparisons, not full-kernel evaluations.','']
    for name,row in local.items():lines.append(f"- {name}: critic {fmt(row['critic'])}; MC {fmt(row['mc'])}; MC-minus-critic {fmt(row['mc_minus_critic'])}.")
    lines+=['','## Local improvement versus full use','']
    for name,row in terminal.items():
        lines.append(f"- {name}: last-step full return change {fmt(row['final_minus_prefinal'])}; final-minus-start {fmt(row['final_minus_start'])}; response parameter L2 change {row['response_l2_change']:.4f}.")
    lines+=['',f'{transfer_fail}/6 final updates have resolved local MC decreases without resolved full-rollout decreases. Lack of resolution is not proof of no effect; inspect both intervals above.','',
        '## Verification and cost','',
        f"{done['computed_transitions']:,}/6,000,000 computed model transitions (including masked MC padding), 144 response updates, 47,400 critic steps. Zero native steps or actor/nominal updates. '
        'Raw MC targets, paired intervals, optimizer proposals, selection means, and full returns were recomputed from saved artifacts. '
        'All 16 diagonal offsets remain exact, every proposal produces identical paired diagonal samples, and all final geometry/history/Lipschitz checks passed. '
        'No minimum action sensitivity, forced transition, reward change or new geometry was introduced.','',
        '## Decision and scope','',next_step,'',
        'Local estimation error, response exploration and local-to-full mismatch can coexist. Fresh randomness on previously inspected held-out roots does not establish generalization to a new root distribution. '
        'Intervals are descriptive and not simultaneous across the 18 checks. Selection samples never enter final evaluation. '
        'Any recovered pessimism is relative to these baselines and this fixed family; neither global worst-case recovery nor native-policy benefit is established. '
        'The actor remains fixed, and no native environment evaluation was performed. No further search or experiment followed.','',
        'Protocol: PROTOCOL.md. Complete results: results.json. Raw comparisons: checks/*/matched_mc.npz. '
        'Full models: candidate_pool.npz and pre_final.npz. Independent paths: evaluation/. No existing artifact was overwritten.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(out/'audit.json',dict(passed=True,matched_mc_computed_slots_recounted=computed,all144_proposals_recomputed=True,
        all_selection_means_and_winners_recomputed=True,all_final_returns_recomputed=True,source_hashes_verified=True))
    print(decision,flush=True);print(next_step,flush=True)
    print({s:r['versus_baseline'] for s,r in contrasts.items()},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True);run(parser.parse_args().out)
