"""Read-only artifact audit and report; emits no model/native transitions."""
import argparse
import json
from pathlib import Path
import numpy as np
from crl import checkpoint
from ett import pointmaze_full_f4_integration as p
from ett.policy_improvement import tree_sha
from ett.finite_crl import write,sha

def metrics(r,native):
    returns=np.sum(r['reward']*np.power(.95,np.arange(50))[None,:],axis=1)
    result=dict(returns=returns,success=(np.min(np.linalg.norm(r['states'][...,:2]-p.GOAL[:2],axis=-1),axis=1)<.5).astype(float))
    result['failure' if native else 'zero_return']=r['failure'][:,-1].astype(float) if native else (returns==0).astype(float)
    return result

def run(out):
    p.verify(out); results=json.loads((out/'results.json').read_text())
    training=json.loads((out/'training.json').read_text()); completion=json.loads((out/'completion.json').read_text())
    assert completion['joint_updates']==6000 and completion['native_steps']==89600
    assert completion['model_transitions']==243200
    initial=checkpoint.load_checkpoint(out/'checkpoints'/'I.pkl')[1]
    assert tree_sha(initial)==json.loads((out/'preregistration.json').read_text())['initial_state_sha256']
    native={}; model={}; episodes=0; simulated=0; audit_checks=[]
    for name,row in results['native'].items():
        r=dict(np.load(out/f'{name}_native.npz')); native[name]=metrics(r,True); episodes+=len(r['reward'])
        np.testing.assert_array_equal(r['reset_seed'],np.arange(256)+150000000)
        np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
        assert np.all(np.diff(r['failure'].astype(int),axis=1)>=0)
        assert np.max(np.abs(r['action']))<=1 and r['reward'].shape==(256,50)
        for k,v in row.items(): np.testing.assert_allclose(native[name][k].mean(),v,atol=1e-12)
    for label,row in results['native_contrasts'].items():
        a,b=label.split('_minus_'); actual=p.paired(native[a],native[b])
        for k in row:
            np.testing.assert_allclose(actual[k]['ci95'],row[k]['ci95'],atol=1e-12)
            np.testing.assert_allclose(actual[k]['difference'],row[k]['difference'],atol=1e-12)
    for m,rows in results['model'].items():
        model[m]={}
        for name,row in rows.items():
            r=dict(np.load(out/f'{m}_{name}_model.npz')); simulated+=r['action'].shape[0]*50
            model[m][name]=metrics(r,False)
            np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
            assert r['valid'].all()
            if m.startswith('obs'): np.testing.assert_array_equal(r['action'],r['x_prime'])
            else: assert np.mean(np.abs(r['action']-r['x_prime']))>0
            for k,v in row.items(): np.testing.assert_allclose(model[m][name][k].mean(),v,atol=1e-12)
        for label,row in results['model_primary'][m].items():
            a,b=label.split('_minus_'); actual=p.paired(model[m][a],model[m][b])
            for k in row: np.testing.assert_allclose(actual[k]['ci95'],row[k]['ci95'],atol=1e-12)
    for seed in [0,1]:
        audits={a:dict(np.load(out/f'{a}_s{seed}_batch_audit.npz')) for a in ['O','B','C']}
        for field in ['off_episode','off_time','off_future','permutation']:
            np.testing.assert_array_equal(audits['O'][field],audits['B'][field])
            np.testing.assert_array_equal(audits['O'][field],audits['C'][field])
        for field in ['syn_episode','syn_time','syn_future','source']:
            np.testing.assert_array_equal(audits['B'][field],audits['C'][field])
        assert audits['O']['source'].sum()==0
        for arm in ['B','C']:
            assert audits[arm]['source'].sum()==25600
            for u in [0,250,500,750]:
                records=[]
                for h in [50,40,25,10]:
                    r=dict(np.load(out/'rounds'/f'{arm}_s{seed}_u{u}'/f'h{h}.npz'))
                    assert r['states'].shape==(32,h+1,8)
                    assert np.all(r['root_time']+h==50)
                    if arm=='B': np.testing.assert_array_equal(r['action'],r['x_prime'])
                    else: assert np.mean(np.abs(r['action']-r['x_prime']))>0
                    np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
                    simulated+=32*h; records.append(r)
                lengths=np.repeat([51,41,26,11],32)
                e=audits[arm]['syn_episode'][u:u+250]; t=audits[arm]['syn_time'][u:u+250]; f=audits[arm]['syn_future'][u:u+250]
                mask=e>=0
                assert np.all(f[mask]>t[mask]) and np.all(f[mask]<lengths[e[mask]])
        for arm in ['O','B','C']:
            name=f'{arm}_s{seed}'; state=checkpoint.load_checkpoint(out/'checkpoints'/f'{name}_final.pkl')[1]
            assert tree_sha(state)==training['runs'][name]['final_state_sha256']
    assert episodes*50==89600 and simulated==243200
    audit=dict(utc=p.stamp(),passed=True,native_steps_recounted=episodes*50,model_steps_recounted=simulated,
        raw_metrics_and_primary_intervals_recomputed=True,all_final_state_hashes_verified=True,
        original_full_state_preserved=True,paired_offline_indices_permutations_and_synthetic_goal_indices=True,
        no_padded_future_goals=True,observational_equal_channels=True,pessimistic_distinct_natural_actions=True,
        train_eval_separation='all final checkpoints sealed in evaluation_started.json before native outcomes',
        correctness_tests='4 integration/routing/initial-actor tests and 2 original replay/mixing tests passed before prepare')
    write(out/'audit.json',audit)
    ci=lambda row:f"{row['difference']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
    title=('Pessimistic augmentation met the predeclared native-benefit criterion.' if results['consistent_added_benefit'] else
           'Pessimistic ETT augmentation did not establish added native benefit over both offline continuation and observational augmentation.')
    lines=['# Original full-F4 Contrastive RL + BC integration','',title,'',
        'One fixed experiment: seeds 0 and 1; 1,000 updates per trained actor from the same original step-150,000 full checkpoint. '
        'I is unchanged; O is offline continuation; B is observational augmentation; C is pessimistic ETT augmentation. '
        'Models are the final diagonal/critic-driven ETTs from 817e429. All six actors actually changed.','',
        'Seed 0 favors C over B in return (+0.4103, 95% CI [0.0050, 0.8479]), success '
        '(+4.30 percentage points) and failure (-4.30 points). Seed 1 does not resolve any of these advantages. '
        'Neither seed resolves C over O. The conditional two-seed mean C-O return difference is +0.0343 '
        '[-0.2145, 0.2680], and C-B is +0.1703 [-0.1143, 0.4877]. These results are insufficient to '
        'claim benefit beyond both controls; they do not establish equivalence or that longer training would behave the same.','',
        '## Native outcomes','',
        '256 fresh paired native resets per actor, horizon 50, stochastic tanh-Gaussian execution for every actor. '
        'Return = sum(.95^t * reward), where reward uses next-XY distance <2. Success = any distance <.5; failure = absorbing death. '
        'A failure-to-succeed is not counted as an absorbing failure.','']
    for name,row in results['native'].items():
        lines.append(f"- {name}: discounted return {row['returns']:.4f}; success {row['success']:.4%}; failure {row['failure']:.4%}.")
    lines+=['','## Paired native differences','',
        'Differences are left minus right. Brackets are 95% paired episode-bootstrap intervals (2,000 replicates). '
        'Return/success increases and failure decreases are favorable. The C-O and C-B contrasts are primary.','']
    for label,row in results['native_contrasts'].items():
        lines.append('- '+label+': '+ '; '.join(k+' '+ci(v) for k,v in row.items())+'.')
    lines+=['','Binary discordant counts are retained in results.json. These intervals condition on each trained actor; '
        'two seeds do not estimate broad training-seed uncertainty. No multiplicity correction was applied.','',
        'Descriptive paired mean across the two fixed seeds (joint resampling of shared episode indices):','']
    for label,row in results['conditional_two_seed_mean'].items():
        lines.append('- '+label+': '+ '; '.join(k+' '+ci(v) for k,v in row.items())+'.')
    lines+=['','## Actual policy changes','',
        'On a fixed bank of 1,024 offline F4 states with the commanded goal: KL(initial Gaussian || final Gaussian), '
        'RMS/max L2 distance between tanh means, average Gaussian-scale change, and parameter L2. '
        'These were measured after training and never used as update rejection criteria.','']
    for name,row in training['runs'].items():
        c=row['policy_change']
        lines.append(f"- {name}: KL {c[0]:.6f}; action RMS/max {c[1]:.6f}/{c[2]:.6f}; scale change {c[3]:+.6f}; "
                     f"actor parameter L2 {row['parameter_l2']:.5f}; critic parameter L2 {row['critic_parameter_l2']:.5f}; 1,000 joint updates.")
    lines+=['','## Independent frozen-model rollouts','',
        'Each cell contains 128 independent 50-step rollouts from START, paired across actors, with the specified model acting at every step. '
        'These are actual simulated trajectories, not critic predictions. Full paired differences versus I and primary C-O/C-B contrasts '
        'are in results.json. The model has no validated absorbing-death label: zero return is reported separately and is not a failure label.','',
        'The unchanged actor scores 8.24-10.00 under the four models versus 3.32 natively. Thus model rollout scores are '
        'poor estimates of native performance here. Observationally augmented B improves versus I on its own model '
        'in both seeds (+1.3222 [0.7373, 1.9353] and +1.1258 [0.5055, 1.7509]); this does not transfer into '
        'a resolved native improvement. C does not show resolved own-model improvement: -0.1012 [-0.3557, 0.0946] '
        'and -0.0291 [-0.2046, 0.1167]. The full cross-model matrix below avoids evaluating only on each arm\'s training model.','']
    for m,rows in results['model'].items():
        lines.append('### '+m+'\n')
        for name,row in rows.items():
            lines.append(f"- {name}: return {row['returns']:.4f}; geometric success {row['success']:.4%}; zero return {row['zero_return']:.4%}.")
        for label,row in results['model_primary'][m].items(): lines.append('- '+label+' return '+ci(row['returns'])+'.')
        lines.append('')
    lines+=['## Learner and synthetic-action BC treatment','',
        'The driver directly calls the existing original full-F4 learner. It restores actor, critic, target critic and Adam states. '
        'Binary NCE, full-F4 achieved-goal relabeling, actor diagonal-logit maximization, BC=.5, random_goals=.5, '
        'batch=256, both learning rates=.0003 and Adam eps=1e-7 remain unchanged. '
        'There is no region-Q update, cumulative KL/action-drift rejection, or new gradient clipping.','',
        'Every synthetic executed action enters the same BC term as an offline action, duplicated for the hindsight goal and '
        'rolled random goal by the existing actor loss. Natural nominal actions are auxiliary model inputs only. '
        'B and C use exactly the same treatment; the experiment does not isolate or attribute effects specifically to BC.','',
        'B queries diagonal_s at (executed action, executed action). C queries critic_s with the current executed action and '
        'an independently sampled action from the frozen nominal policy. The current actor generates all new paths. '
        'Each augmented actor uses 25,600 synthetic rows out of 256,000 (10%); trajectories refresh at updates 0/250/500/750. '
        'Per refresh: 32 roots at each time 0/10/25/40, rolling to the original horizon 50. Full remaining future goals use '
        'the same geometric sampler and matched indices; no padding is eligible.','',
        'Offline continuation uses the existing 5,940-episode continuation partition. The original checkpoint saw all 6,600 episodes; '
        'the excluded 660 episodes therefore are not independent validation of that checkpoint. Only obs/act feed training.','',
        '## Budget, provenance and limits','',
        'Exactly 6,000 joint updates; 64,000 training-model transitions + 179,200 independent evaluation-model transitions = '
        '243,200 model transitions; 89,600 native steps. Frozen transition/nominal hashes and the unchanged initial learner state verified. '
        'All final checkpoints were sealed before native outcomes. Native outcomes never entered learning or checkpoint selection. '
        'No additional experiment followed.','',
        'This is a modest repository integration experiment, not an exact paper reproduction. Model error, partial observability, '
        'finite training budget and different inherited diagonal heads limit interpretation. Inherited validation '
        'diagonal-energy degradation from 817e429 was +0.000181/+0.000249 for B (seeds 0/1), and '
        '+0.014969/+0.008099 for C; this comparison does not isolate the ETT response map. A pessimistic ETT optimized against I '
        'is not established worst-case for the updated actors; mixed-data training alone does not establish robust max-min optimization. '
        'Intervals crossing zero are inconclusive, and observed ties do not prove equivalence.','',
        'Artifacts: PROTOCOL.md, preregistration.json, learner_config.json, partition.npz, final full-state checkpoints, '
        'rounds/*/h*.npz, *_batch_audit.npz, *_native.npz, *_model.npz, training.json, results.json, ledgers, audit.json.','',
        'Reproduce only if separately requested, in a fresh directory: `python -m ett.pointmaze_full_f4_integration prepare --out <dir>`, '
        'then phases `train` and `evaluate`; `python -m ett.report_full_f4_integration --out <dir>` audits saved arrays without more rollouts.','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    names=list(results['native'])
    labels=['Initial','Offline 0','Obs. 0','ETT 0','Offline 1','Obs. 1','ETT 1']
    colors=['#8d949b','#626970','#9ca3aa','#296ea4','#626970','#9ca3aa','#296ea4']
    axes[0].bar(labels,[results['native'][a]['returns'] for a in names],color=colors)
    axes[0].tick_params(axis='x',rotation=35)
    axes[0].set(xlabel='Actor / training seed',ylabel='Mean discounted return (reward units)',title='Native environment: final actors')
    keys=[f'C_s{s}_minus_{b}_s{s}' for s in [0,1] for b in ['O','B']]
    rows=[results['native_contrasts'][k]['returns'] for k in keys]
    values=np.array([r['difference'] for r in rows]); bounds=np.array([r['ci95'] for r in rows])
    axes[1].errorbar(values,np.arange(4),xerr=np.stack([values-bounds[:,0],bounds[:,1]-values]),fmt='o',color='#296ea4',capsize=4)
    axes[1].axvline(0,color='#8d949b',linewidth=1)
    axes[1].set(yticks=np.arange(4),yticklabels=['ETT - offline, seed 0','ETT - obs., seed 0','ETT - offline, seed 1','ETT - obs., seed 1'],
        xlabel='Paired return difference (reward units), 95% CI',ylabel='Prespecified comparison',title='Added benefit is not consistent across seeds')
    axes[1].invert_yaxis()
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Original full-F4 Contrastive RL + BC: bounded integration',fontsize=13)
    fig.text(.5,-.04,'Source: full_f4_integration_s01_v1; 1,000 updates / actor, 256 fresh paired resets; final checkpoints only.',ha='center',fontsize=9)
    fig.savefig(out/'native_comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'native_comparison.pdf',bbox_inches='tight'); plt.close(fig)
    write(out/'manifest.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='manifest.json'})
    print(title,flush=True)
    print(json.dumps(results['native'],indent=2),flush=True)
    print(json.dumps(results['conditional_two_seed_mean'],indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__); parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
