"""Evaluation-only targets: candidate first step, frozen base thereafter."""
import json
import jax
import numpy as np

from ett import pointmaze_future_average as p
from ett import pointmaze_phase_sampling as prior
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write, sha


def boot_indices(groups):
    rng=np.random.default_rng(p.CONFIG['bootstrap_seed'])
    return np.concatenate([rng.choice(np.flatnonzero(groups==g),
        (p.CONFIG['bootstrap_repeats'], int(np.sum(groups==g)))) for g in np.unique(groups)], axis=1)


def interval(samples, indices):
    """Rows are independent root episodes; columns are paired MC repeats."""
    means=samples.mean(1); estimate=float(means.mean())
    se=float(np.sqrt(np.sum(samples.var(1, ddof=1)/samples.shape[1]))/len(samples))
    boot=means[indices].mean(1)
    return dict(mean=estimate, conditional_mc_se=se,
        conditional_mc_ci95=[estimate-1.96*se, estimate+1.96*se],
        episode_ci95=np.quantile(boot, [.025, .975]),
        repeat_half_means=[float(x.mean()) for x in np.array_split(samples, 2, axis=1)])


def resolved(row, minimum=0.):
    sign=np.sign(row['mean'])
    return bool(abs(row['mean'])>minimum and
        all(np.all(sign*np.asarray(row[key])>0) for key in ['conditional_mc_ci95', 'episode_ci95', 'repeat_half_means']))


def select_queries(records, root_count):
    rng=np.random.default_rng(p.CONFIG['time_seed'])
    times=np.empty(root_count, int); lengths=np.empty(root_count, int)
    states=np.empty((root_count, 8), np.float32); actions=np.empty((root_count, 2), np.float32)
    # Draw in root order, independent of horizon grouping or outcomes.
    by_root={int(root):(r, i) for r in records for i, root in enumerate(r['root'])}
    assert len(by_root)==root_count
    for root in range(root_count):
        r, i=by_root[root]; length=r['action'].shape[1]; t=int(rng.integers(length))
        states[root]=r['states'][i, t]; actions[root]=r['action'][i, t]
        times[root]=t; lengths[root]=length
    return dict(state=states, action=actions, t=times, length=lengths,
        h=lengths-times, weight=lengths*.95**times)


def continuation(engine, ns, next_action, remaining, key, ledger):
    """Crucially the candidate theta is not an argument to this function."""
    if remaining==0:
        return np.zeros(len(ns)), np.empty((len(ns), 0), np.float32)
    ledger.add(len(ns)*remaining, 'base_only_continuation')
    r=engine.rollout(np.zeros(48, np.float32), ns, key, remaining, next_action)
    np.testing.assert_array_equal(r['action'][:, 0], next_action)
    return .05*r['reward']@.95**np.arange(remaining), r['reward']


def surrogate_samples(engine, queries, models, context, ledger, out):
    n=len(queries['h']); repeats=p.CONFIG['repeats']; arrays={}
    for name, theta in p.candidates().items():
        mc=np.zeros((n, repeats)); pred={k:np.zeros_like(mc) for k in models}
        successors=np.zeros((n, repeats, 8), np.float32)
        actor_actions=np.zeros((n, repeats, 2), np.float32)
        rewards=np.full((n, repeats, 49), np.nan, np.float32)
        for h in sorted(np.unique(queries['h'])):
            ids=np.flatnonzero(queries['h']==h)
            s=np.repeat(queries['state'][ids], repeats, 0)
            a=np.repeat(queries['action'][ids], repeats, 0)
            seed=p.CONFIG['surrogate_seed']+int(h)*10
            xp=engine.nominal.sample(s, jax.random.PRNGKey(seed), 1, goal=np.broadcast_to(old.GOAL, s.shape))
            ledger.add(len(s), 'candidate_first_transition_'+name)
            y, detail=engine.sample(theta, s, a, xp, jax.random.PRNGKey(seed+1), 1)
            old.validate_selected_set(s, y, detail); ns=np.asarray(y[:, 0])
            next_action=np.asarray(engine.actor_jit(ns, jax.random.PRNGKey(seed+2)))
            reward=np.asarray(old.task_reward(ns, np.broadcast_to(old.GOAL, ns.shape)))
            future, future_rewards=continuation(engine, ns, next_action, int(h)-1,
                jax.random.PRNGKey(seed+3), ledger)
            mc[ids]=(.05*reward+.95*future).reshape(len(ids), repeats)
            successors[ids]=ns.reshape(len(ids), repeats, 8)
            actor_actions[ids]=next_action.reshape(len(ids), repeats, 2)
            rewards[ids, :, 0]=reward.reshape(len(ids), repeats)
            rewards[ids, :, 1:h]=future_rewards.reshape(len(ids), repeats, int(h)-1)
            for model_name, critic in models.items():
                q=critic.predict(ns, next_action, np.full(len(ns), h-1), context['mean'], context['std'])
                pred[model_name][ids]=(.05*reward+.95*q).reshape(len(ids), repeats)
        arrays[name+'_mc']=mc
        arrays.update({name+'_'+k:v for k, v in pred.items()})
        np.savez_compressed(out/(name+'_continuations.npz'), successor=successors,
            next_action=actor_actions, rewards=rewards, mc=mc, **pred)
        print(name+': candidate first-step / frozen-base MC evaluation complete.', flush=True)
    return arrays


def validity(engine, roots, ledger):
    rng=np.random.default_rng(p.CONFIG['validity_seed']); key=jax.random.PRNGKey(p.CONFIG['validity_seed'])
    n=len(roots); diagonal_action=rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    base=None; result={}
    s=np.repeat(roots, 16, 0)
    a=rng.uniform(-1, 1, (len(s), 2)).astype(np.float32)
    b=rng.uniform(-1, 1, (len(s), 2)).astype(np.float32)
    xp=rng.uniform(-1, 1, (len(s), 2)).astype(np.float32)
    for name, theta in p.candidates().items():
        ledger.add(n*4, 'diagonal_validity_'+name)
        diag, detail=engine.sample(theta, roots, diagonal_action, diagonal_action, key, 4)
        old.validate_selected_set(roots, diag, detail)
        if base is None: base=np.asarray(diag)
        drift=float(np.max(np.abs(np.asarray(diag)-base)))
        np.testing.assert_array_equal(diag, base)
        outputs=[]
        for action in [a, b]:
            ledger.add(len(s)*4, 'lipschitz_validity_'+name)
            y, detail=engine.sample(theta, s, action, xp, key, 4)
            old.validate_selected_set(s, y, detail); outputs.append(np.asarray(y))
        difference=np.linalg.norm(outputs[0]-outputs[1], axis=-1)
        distance=np.linalg.norm(a-b, axis=-1)[:, None]
        excess=float(np.max(difference-distance)); ratio=float(np.max(difference/distance))
        mats=np.asarray(old.matrices(theta[16:], 1.))
        norm=float(np.linalg.svd(mats, compute_uv=False).max())
        assert excess<=p.CONFIG['tolerance'] and norm<=1+p.CONFIG['tolerance']
        result[name]=dict(diagonal_max_drift=drift, max_action_ratio=ratio,
            max_lipschitz_excess=excess, max_component_operator_norm=norm,
            finite_geometry_history_valid=True, parameter_norm=float(np.linalg.norm(theta)))
    return result


def calibration_metrics(pred, returns, h, indices):
    target=returns.mean(1); error=pred-target
    se=returns.std(1, ddof=1)/np.sqrt(returns.shape[1])
    excess=np.maximum(pred-(1-.95**h), -pred)
    return dict(rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
        rmse_ci95=np.quantile(np.sqrt(np.mean(error[indices]**2, axis=1)), [.025, .975]),
        bias_ci95=np.quantile(error[indices].mean(1), [.025, .975]),
        range_excess=float(max(0., excess.max())), range_violation_fraction=float(np.mean(excess>1e-6)),
        max_mc_se=float(se.max()), mean_target_noise_variance=float(np.mean(se**2)))


def calibration(engine, roots, root_h, models, context, ledger, out, indices, queries, arrays):
    action=np.asarray(engine.actor_jit(roots, jax.random.PRNGKey(p.CONFIG['calibration_seed'])))
    records=early.collect(engine, np.zeros(48, np.float32), roots, 50-root_h,
        p.CONFIG['calibration_seed']+100, ledger, 'root_calibration', p.CONFIG['calibration_repeats'], action)
    returns=np.empty((len(roots), p.CONFIG['calibration_repeats']))
    for r in records:
        h=r['action'].shape[1]
        for root in np.unique(r['root']):
            returns[root]=.05*r['reward'][r['root']==root]@.95**np.arange(h)
    saved=dict(root_state=roots, root_action=action, root_h=root_h, root_returns=returns)
    metrics={}; paired={}
    for cohort, state, act, h, target in [
        ('root', roots, action, root_h, returns),
        ('visitation', queries['state'], queries['action'], queries['h'], arrays['base_mc'])]:
        predictions={}; metrics[cohort]={}; paired[cohort]={}
        for name, critic in models.items():
            pred=critic.predict(state, act, h, context['mean'], context['std'])
            predictions[name]=pred; saved[cohort+'_'+name]=pred
            metrics[cohort][name]=calibration_metrics(pred, target, h, indices)
        for seed in p.CONFIG['seeds']:
            a=predictions[f'averaged_s{seed}']-target.mean(1)
            b=predictions[f'sampled_s{seed}']-target.mean(1)
            boot=np.sqrt(np.mean(a[indices]**2, axis=1))-np.sqrt(np.mean(b[indices]**2, axis=1))
            paired[cohort][str(seed)]=dict(rmse_change=float(np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b))),
                episode_ci95=np.quantile(boot, [.025, .975]))
    np.savez_compressed(out/'calibration.npz', **saved)
    return dict(models=metrics, paired=paired)


def analyze(arrays, queries, groups):
    indices=boot_indices(groups); weight=queries['weight'][:, None]
    differences={}; errors={}; candidate_names=list(p.candidates())[1:]
    model_names=[f'{arm}_s{seed}' for seed in p.CONFIG['seeds'] for arm in p.CONFIG['arms']]
    deltas={}
    for name in candidate_names:
        target=weight*(arrays[name+'_mc']-arrays['base_mc'])
        ref=interval(target, indices); distinguishable=resolved(ref, p.CONFIG['minimum_effect'])
        row=dict(mc=ref, distinguishable=distinguishable, critics={}); deltas[name]={'mc':target}
        for model in model_names:
            values=weight*(arrays[name+'_'+model]-arrays['base_'+model])
            pred=interval(values, indices); direction_resolved=resolved(pred)
            agreement=(bool(np.sign(pred['mean'])==np.sign(ref['mean']))
                       if distinguishable and direction_resolved else None)
            row['critics'][model]=dict(**pred, direction_resolved=direction_resolved,
                direction_agreement=agreement, signed_error=pred['mean']-ref['mean'])
            deltas[name][model]=values
        differences[name]=row
    eligible=sum(r['distinguishable'] for r in differences.values())
    directions={model:dict(eligible=eligible,
        resolved=sum(r['distinguishable'] and r['critics'][model]['direction_resolved'] for r in differences.values()),
        correct=sum(r['critics'][model]['direction_agreement'] is True for r in differences.values()),
        wrong=sum(r['critics'][model]['direction_agreement'] is False for r in differences.values()))
        for model in model_names}
    truth=np.array([deltas[name]['mc'].mean() for name in candidate_names])
    truth_boot=np.stack([deltas[name]['mc'].mean(1)[indices].mean(1) for name in candidate_names], -1)
    for seed in p.CONFIG['seeds']:
        row={}; boot={}
        for arm in p.CONFIG['arms']:
            name=f'{arm}_s{seed}'
            pred=np.array([deltas[c][name].mean() for c in candidate_names])
            pred_boot=np.stack([deltas[c][name].mean(1)[indices].mean(1) for c in candidate_names], -1)
            row[arm+'_mae']=float(np.mean(np.abs(pred-truth)))
            boot[arm]=np.mean(np.abs(pred_boot-truth_boot), axis=1)
        row['averaged_minus_sampled_mae']=row['averaged_mae']-row['sampled_mae']
        row['episode_ci95']=np.quantile(boot['averaged']-boot['sampled'], [.025, .975])
        row['improvement_criterion']=bool(row['episode_ci95'][1]<0 and all(
            r['critics'][f'averaged_s{seed}']['direction_agreement'] is True
            for r in differences.values() if r['critics'][f'sampled_s{seed}']['direction_agreement'] is True))
        errors[str(seed)]=row
    return dict(differences=differences, direction_counts=directions, primary=errors,
        clear_improvement=all(r['improvement_criterion'] for r in errors.values()))


def report(out, result, cal, valid, ledger):
    fmt=lambda values:'['+', '.join(f'{v:+.5f}' for v in values)+']'
    lines=['# Future-time averaged NCE: bounded ETT usefulness comparison', '',
        ('Both paired seeds meet the predeclared improvement criterion.' if result['clear_improvement'] else
         'The comparison does not establish a clear improvement in critic usefulness for ETT updates.'), '',
        'Four critics completed 1,500 updates each on the identical 288 trajectories / 13,648 uniform-sampled rows. '
        'Both sampled baselines exactly reproduce prior checkpoints. Only the positive term changes. '
        'Actor, nominal policy and base ETT stayed frozen; zero ETT or actor updates.', '',
        '## Signed candidate-minus-base surrogate differences', '',
        'Values use the existing H*.95^t visitation weights; negative means more pessimistic. '
        'Intervals below are episode-bootstrap 95% intervals. Conditional paired-MC intervals and SEs '
        'are also retained in results.json. Direction screens require both intervals and split-half agreement.', '']
    for name, row in result['differences'].items():
        ref=row['mc']
        lines.append(f"- {name}: MC {ref['mean']:+.5f}, episode CI {fmt(ref['episode_ci95'])}, "
            f"conditional MC CI {fmt(ref['conditional_mc_ci95'])}; "+
            ('distinguishable.' if row['distinguishable'] else 'inconclusive.'))
        for model, values in row['critics'].items():
            status='inconclusive' if values['direction_agreement'] is None else ('correct direction' if values['direction_agreement'] else 'wrong direction')
            lines.append(f"  - {model}: {values['mean']:+.5f}, CI {fmt(values['episode_ci95'])}; {status}.")
    lines+=['', '## Paired error and calibration', '']
    for seed, row in result['primary'].items():
        lines.append(f"- Seed {seed}: signed-difference MAE sampled {row['sampled_mae']:.5f}, "
            f"averaged {row['averaged_mae']:.5f}; change {row['averaged_minus_sampled_mae']:+.5f}, "
            f"CI {fmt(row['episode_ci95'])}.")
    for name, row in result['direction_counts'].items():
        lines.append(f"- {name}: {row['correct']} correct, {row['wrong']} wrong, "
            f"{row['resolved']} resolved predictions among {row['eligible']} distinguishable MC differences (four probes total).")
    for cohort, metrics in cal['models'].items():
        lines+=['', cohort.capitalize()+' calibration (normalized return Q):']
        for name, row in metrics.items():
            lines.append(f"- {name}: RMSE {row['rmse']:.5f}, bias {row['bias']:+.5f}, "
                f"range excess {row['range_excess']:.5f}; maximum target MC SE {row['max_mc_se']:.5f}.")
        for seed, row in cal['paired'][cohort].items():
            lines.append(f"- Seed {seed} averaged-minus-sampled RMSE {row['rmse_change']:+.5f}, CI {fmt(row['episode_ci95'])}.")
    training=json.loads((out/'training.json').read_text())
    lines+=['', '## Validity and limits', '',
        f"All five kernels passed exact samplewise diagonal equality (maximum drift "
        f"{max(v['diagonal_max_drift'] for v in valid.values()):.1g}), geometry/history checks, and the L=1 action test "
        f"(maximum observed ratio {max(v['max_action_ratio'] for v in valid.values()):.5f}; tolerance 2e-6). "
        'All perturbations have zero diagonal offsets; common-anchor convex projection preserves the diagonal law and '
        'the global action-Lipschitz construction. These are model-side properties, not native physical validity.', '',
        f"Averaging removes conditional future-time Bernoulli label variance (row mean p(1-p) "
        f"{training['mean_label_variance']:.5f}) exactly. It does not remove finite-trajectory noise, optimization error, "
        'function approximation error, coverage limitations or generator misspecification. Two initializations share '
        'one training dataset; this experiment cannot estimate training-trajectory variability.', '',
        'The 36 episode roots are held out of critic training but were evaluated in the prior experiment. '
        'All visitation paths and MC targets here are fresh and excluded from training. Each candidate acts only '
        'on the first transition; every continuation uses the frozen base and actor. The primary target is the '
        'same model surrogate, not a native-environment return or a full candidate rollout. Episode intervals '
        'are descriptive, include visitation variation, and are not simultaneous guarantees. Four correlated '
        'small probes and unresolved effects do not establish general usefulness or equivalence.', '',
        f"Budget: {ledger.data['charged']:,}/1,000,000 new model transitions; 6,000/6,000 critic updates; "
        'zero native steps. No ETT optimization, actor training, further sweep, or budget extension occurred.', '',
        'Protocol/config/source/input hashes were sealed before training and evaluation. Raw continuations, '
        'queries, probabilities, checkpoints, predictions, metrics and validity checks are saved beside this report.', '',
        'Reproduce with a fresh directory using `python -m ett.pointmaze_future_average prepare --out <directory>`, '
        'then `python -m ett.pointmaze_future_average run --out <directory>`. Tests: '
        '`python -m unittest scripts.test_pointmaze_future_average -v`.', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def evaluate(out, models, context):
    frozen={name:old.parameter_sha(model) for name, model in models.items()}
    engine=old.Kernel(); roots_archive=dict(np.load(p.PREVIOUS/'final_contexts.npz'))
    roots=roots_archive['roots']; root_indices=roots_archive['indices']; groups=root_indices[:, 2]
    assert len(roots)==36 and len(np.unique(root_indices[:, 0]))==36
    ledger=old.Ledger(out); ledger.data['cap']=p.CONFIG['model_output_cap']
    # Determine cost from declared random times and lengths before generating any outcomes.
    lengths=50-root_indices[:, 1]; rng=np.random.default_rng(p.CONFIG['time_seed'])
    times=np.array([rng.integers(h) for h in lengths])
    planned=int(lengths.sum()+p.CONFIG['calibration_repeats']*lengths.sum()+
        5*p.CONFIG['repeats']*np.sum(lengths-times)+5*(36*4+36*16*2*4))
    write(out/'evaluation_budget.json', dict(planned=planned, cap=ledger.data['cap'],
        times=times, lengths=lengths, utc=p.timestamp(), checkpoint_seal=sha(out/'training.json')))
    if planned>ledger.data['cap']: raise RuntimeError('Fixed evaluation exceeds cap; no resampling.')
    trees=[engine.base.params, engine.nominal.params]
    before=[np.array(x) for tree in trees for x in jax.tree.leaves(tree)]
    records=early.collect(engine, np.zeros(48, np.float32), roots, root_indices[:, 1],
        p.CONFIG['query_seed'], ledger, 'fresh_visitation_paths', 1)
    prior.save_records(out/'visitation_paths.npz', records)
    queries=select_queries(records, len(roots)); queries.update(episode=root_indices[:, 0], group=groups)
    np.testing.assert_array_equal(times, queries['t'])
    np.savez_compressed(out/'queries.npz', **queries)
    valid=validity(engine, roots, ledger); write(out/'validity.json', valid)
    arrays=surrogate_samples(engine, queries, models, context, ledger, out)
    np.savez_compressed(out/'surrogate_samples.npz', **arrays)
    cal=calibration(engine, roots, lengths, models, context, ledger, out, boot_indices(groups), queries, arrays)
    write(out/'calibration.json', cal)
    result=analyze(arrays, queries, groups); write(out/'results.json', result)
    for name, model in models.items(): assert old.parameter_sha(model)==frozen[name]
    after=[np.array(x) for tree in trees for x in jax.tree.leaves(tree)]
    for a, b in zip(before, after): np.testing.assert_array_equal(a, b)
    assert ledger.data['charged']==planned
    report(out, result, cal, valid, ledger)
    write(out/'completion.json', dict(utc=p.timestamp(), model_outputs=ledger.data['charged'],
        clear_improvement=result['clear_improvement'], frozen_parameters_verified=True,
        ett_updates=0, actor_updates=0, native_steps=0))
