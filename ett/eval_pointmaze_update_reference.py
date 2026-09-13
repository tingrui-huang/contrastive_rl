"""Independent full-kernel rollout and diagonal-fit evaluation; no updates."""
import json
from collections import defaultdict

import jax
import numpy as np

from ett import pointmaze_update_reference as p
from ett import pointmaze_future_average as average
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett import pointmaze_phase_sampling as phase
from ett.finite_crl import write, sha


def bootstrap(groups):
    rng=np.random.default_rng(p.CONFIG['bootstrap_seed'])
    return np.concatenate([rng.choice(np.flatnonzero(groups==g),
        (2000, int(np.sum(groups==g)))) for g in np.unique(groups)], axis=1)


def paired_returns(difference, groups):
    means=difference.mean(1); index=bootstrap(groups)
    estimate=float(means.mean())
    se=float(np.sqrt(np.sum(difference.var(1, ddof=1)/difference.shape[1]))/len(difference))
    return dict(mean=estimate, conditional_mc_se=se,
        conditional_mc_ci95=[estimate-1.96*se, estimate+1.96*se],
        episode_ci95=np.quantile(means[index].mean(1), [.025, .975]))


def below_zero(row):
    return bool(row['episode_ci95'][1]<0 and row['conditional_mc_ci95'][1]<0)


def above_zero(row):
    return bool(row['episode_ci95'][0]>0 and row['conditional_mc_ci95'][0]>0)


def diagonal_interval(difference, episodes):
    _, inverse=np.unique(episodes, return_inverse=True)
    sums=np.bincount(inverse, weights=difference); counts=np.bincount(inverse)
    rng=np.random.default_rng(p.CONFIG['bootstrap_seed']+1)
    index=rng.integers(len(sums), size=(2000, len(sums)))
    return dict(mean=float(difference.mean()), episodes=len(sums),
        episode_ci95=np.quantile(sums[index].sum(1)/counts[index].sum(1), [.025, .975]))


def constraints(engine, theta, context, ledger):
    state=context['validation_state'][:16]; key=jax.random.PRNGKey(p.CONFIG['validation_seed']+100)
    xp=np.asarray(engine.nominal.sample(state, key, 1, goal=np.broadcast_to(old.GOAL, state.shape)))
    grid=[np.broadcast_to(np.array([x, y], np.float32), (16, 2)) for x in [-1., 0., 1.] for y in [-1., 0., 1.]]
    actions=grid+[xp, np.clip(xp+.001, -1., 1.)]; outputs=[]
    for a in actions:
        ledger.add(128, 'final_lipschitz_geometry')
        y, d=engine.sample(theta, state, a, xp, key, 8)
        old.validate_selected_set(state, y, d); outputs.append(np.asarray(y))
    # Compare against this kernel's updated diagonal head, not initialization.
    diagonal_theta=theta.copy(); diagonal_theta[16:]=0.
    ledger.add(128, 'updated_diagonal_identity')
    anchor, detail=engine.sample(diagonal_theta, state, xp, xp, key, 8)
    old.validate_selected_set(state, anchor, detail)
    drift=float(np.max(np.abs(outputs[9]-np.asarray(anchor))))
    np.testing.assert_array_equal(outputs[9], anchor)
    excess=0.
    for i in range(len(actions)):
        for k in range(i):
            distance=np.linalg.norm(outputs[i]-outputs[k], axis=-1)
            excess=max(excess, float(np.max(distance-np.linalg.norm(actions[i]-actions[k], axis=-1)[:, None])))
    mats=np.asarray(old.matrices(theta[16:]))
    norm=float(np.linalg.norm(mats, axis=(1, 2)).max())
    passed=excess<=2e-6 and norm<=1+2e-6 and drift==0.
    assert passed
    return dict(passed=passed, updated_diagonal_identity_drift=drift,
        samplewise_lipschitz_excess=excess, component_frobenius_max=norm,
        geometry_history_valid=True)


def pack_returns(records, count, repeats):
    values=np.empty((count, repeats)); rewards=np.full((count, repeats, 49), np.nan, np.float32)
    for r in records:
        h=r['action'].shape[1]
        for root in np.unique(r['root']):
            selected=r['root']==root
            assert selected.sum()==repeats
            values[root]=.05*r['reward'][selected]@.95**np.arange(h)
            rewards[root, :, :h]=r['reward'][selected]
    return values, rewards


def analyze(returns, energies, groups, episodes, guards, validity, training):
    summaries={}; contrasts={}
    for name, values in returns.items():
        degradation=diagonal_interval(energies[name]-energies['initial'], episodes)
        summaries[name]=dict(return_mean=float(values.mean()),
            group_return={g:float(values[groups==i].mean()) for i, g in enumerate(early.GROUPS)},
            diagonal_energy=float(energies[name].mean()), diagonal_degradation=degradation,
            validation_mean_exceeds_allowance=degradation['mean']>.02,
            validation_resolved_excess=bool(degradation['episode_ci95'][0]>.02),
            train_guard=guards[name], train_guard_valid=p.guard_accept(guards[name], guards['initial']),
            constraints=validity[name])
        if name!='initial':
            summaries[name]['versus_initial']=paired_returns(values-returns['initial'], groups)
    bottleneck=[]; shared_worsening=[]
    for seed in p.CONFIG['seeds']:
        diagonal=f'diagonal_s{seed}'; critic=f'critic_s{seed}'; mc=f'mc_s{seed}'
        row={label:paired_returns(returns[a]-returns[b], groups) for label, a, b in [
            ('critic_minus_diagonal', critic, diagonal), ('mc_minus_diagonal', mc, diagonal),
            ('mc_minus_critic', mc, critic)]}
        row['bottleneck_pattern']=bool(below_zero(row['mc_minus_diagonal']) and below_zero(row['mc_minus_critic'])
            and all(summaries[n]['train_guard_valid'] and summaries[n]['constraints']['passed'] and
                not summaries[n]['validation_resolved_excess'] for n in [diagonal, critic, mc]))
        row['shared_resolved_worsening']=bool(above_zero(row['critic_minus_diagonal']) and above_zero(row['mc_minus_diagonal']))
        contrasts[str(seed)]=row; bottleneck.append(row['bottleneck_pattern']); shared_worsening.append(row['shared_resolved_worsening'])
    if all(bottleneck):
        conclusion='MC improves over both controls in both seeds: evidence favors a critic-estimation bottleneck within this budget.'
    elif all(shared_worsening):
        conclusion='Both pessimistic arms worsen returns in both seeds: evidence points to a shared optimization problem within this budget.'
    else:
        conclusion='The experiment does not establish critic estimation as the blocker; the paired results require a bounded, inconclusive interpretation.'
    first={}
    for seed in p.CONFIG['seeds']:
        a=training['histories'][f'critic_s{seed}'][0]; b=training['histories'][f'mc_s{seed}'][0]
        ca=np.asarray(a['signed_terms']); cb=np.asarray(b['signed_terms'])
        da=ca[:, 0, 1]-ca[:, 1, 1]; db=cb[:, 0, 1]-cb[:, 1, 1]
        sa=np.asarray(a['clipped_step']); sb=np.asarray(b['clipped_step'])
        first[str(seed)]=dict(critic_signed_differences=da, mc_signed_differences=db,
            descriptive_sign_agreement=int(np.sum(np.sign(da)==np.sign(db))), directions=4,
            clipped_step_cosine=float(sa@sb/max(np.linalg.norm(sa)*np.linalg.norm(sb), 1e-15)),
            uncertainty_not_established=True)
    return dict(models=summaries, contrasts=contrasts, first_update=first, conclusion=conclusion,
        critic_bottleneck_supported=all(bottleneck))


def costs(ledger):
    result=defaultdict(lambda:defaultdict(int))
    for row in ledger.data['entries']:
        scope, purpose=row['purpose'].split('/', 1); result[scope][purpose]+=row['outputs']
    return {scope:dict(total=sum(values.values()), purposes=dict(values)) for scope, values in result.items()}


def report(out, results, training, cost, ledger):
    fmt=lambda values:'['+', '.join(f'{x:+.5f}' for x in values)+']'
    describe=lambda row:f"{row['mean']:+.5f}; episode CI {fmt(row['episode_ci95'])}, MC CI {fmt(row['conditional_mc_ci95'])}"
    lines=['# Controlled ETT updates: averaged-NCE versus MC continuation', '', results['conclusion'], '',
        'Three attempts per arm and seed (18 total); all 48 ETT coordinates are available to the two-loss arms. '
        'The diagonal-only loss has exactly zero response gradient. Actor/nominal remain frozen. '
        'The critic refreshes for 400 steps per update on current-model paths, then freezes with visitation. '
        'MC uses the same one-step surrogate, with continuations under its frozen pre-update kernel. '
        'Final evaluation instead applies each final kernel throughout the rollout.', '',
        '## Independent final returns', '',
        'Normalized discounted region occupancy; lower is more pessimistic. Each kernel has 64 new rollouts '
        'at each of 36 held-out episode roots. Streams are paired across kernels. All listed intervals are 95%.', '',
        f"Initialization mean return: {results['models']['initial']['return_mean']:.5f}."]
    for seed in p.CONFIG['seeds']:
        lines+=['', f'Seed {seed}:']
        for arm in p.CONFIG['arms']:
            name=f'{arm}_s{seed}'; row=results['models'][name]
            lines.append(f"- {arm}: mean {row['return_mean']:.5f}; versus initialization {describe(row['versus_initial'])}.")
        for label, row in results['contrasts'][str(seed)].items():
            if isinstance(row, dict): lines.append(f"- {label}: {describe(row)}.")
    lines+=['', '## Accepted updates and diagonal fit', '',
        'Acceptance uses the unchanged fixed train energy-score guard: initialization + .02. '
        'Independent validation uses 512 rows and 32 draws, clustered by source episode. '
        'Its intervals include finite-sample fitting noise; it does not select kernels.', '']
    for name, history in training['histories'].items():
        row=results['models'][name]; d=row['diagonal_degradation']
        accept=''.join('A' if h['accepted'] else 'R' for h in history)
        clips=[sum(h[k] for h in history) for k in ['diagonal_step_clipped', 'response_step_clipped']]
        lines.append(f"- {name}: {accept} (A accepted/R rejected); diagonal degradation {d['mean']:+.6f}, "
            f"CI {fmt(d['episode_ci95'])}; train guard {row['train_guard']:.6f}; clipped steps diagonal/response {clips}.")
    valid=all(row['constraints']['passed'] and row['train_guard_valid'] for row in results['models'].values())
    lines+=['', f"All final train guards and geometry/history/Lipschitz checks passed: {valid}. "
        f"Maximum action-Lipschitz excess {max(r['constraints']['samplewise_lipschitz_excess'] for r in results['models'].values()):.3g}; "
        'updated diagonal identity drift is exactly zero. Diagonal head offsets can change the diagonal law; '
        'identity is checked against each updated head with response zeroed. Validation allowance flags and '
        'all gradient/proposal values are retained in JSON.', '', '## Matched first update and simulation cost', '']
    for seed, row in results['first_update'].items():
        lines.append(f"- Seed {seed}: critic signed surrogate differences {fmt(row['critic_signed_differences'])}; "
            f"MC {fmt(row['mc_signed_differences'])}; raw sign agreement {row['descriptive_sign_agreement']}/4, "
            f"clipped-step cosine {row['clipped_step_cosine']:+.3f}. These signs have no independent uncertainty screen.")
    for name in training['histories']:
        lines.append(f"- {name}: {cost[name]['total']:,} training model transitions; "
            f"{cost['evaluation_'+name]['total']:,} final evaluation/audit transitions.")
    lines+=['', f"Total {ledger.data['charged']:,}/3,000,000 new transitions; 2,400/2,400 critic refresh steps; "
        'zero native simulation or actor updates. Prior averaged critic checkpoints and Adam states are reused; '
        'their historical 1,500 steps per seed are outside this new budget. Detailed costs include shared '
        'initial fitting guard and initialization evaluation.', '', '## Interpretation and limits', '',
        'MC removes the learned continuation approximation from the update, but still has rollout noise. '
        'Both pessimistic arms share finite-difference noise, one-step visitation approximation, only three '
        'update attempts, block step caps, geometry and the diagonal guard. After the first update their '
        'visitation and current kernels can differ. A failed or unresolved MC improvement cannot isolate '
        'critic error as the cause; small differences cannot establish equivalence. The report tests a '
        'model-internal pessimistic objective, not native physical validity or global worst-case recovery.', '',
        'Evaluation roots were inspected in earlier experiments; all new return samples are independent of '
        'updates and excluded from training. Episode intervals are descriptive and not simultaneous or '
        'training-seed population guarantees. No historical calibration gate was used. No actor training, '
        'additional sweep, retuning, extension or push follows this report.', '',
        'Reproduce in a fresh directory: `python -m ett.pointmaze_update_reference prepare --out <dir>`, '
        'then `python -m ett.pointmaze_update_reference run --out <dir>`. Read-only audit: '
        '`python -m ett.check_pointmaze_update_reference --out <dir>`.', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def evaluate(engine, out, final, context, ledger):
    training=json.loads((out/'training.json').read_text())
    sealed=sha(out/'final_kernels.npz')
    roots_file=dict(np.load(average.PREVIOUS/'final_contexts.npz'))
    roots=roots_file['roots']; indices=roots_file['indices']; groups=indices[:, 2]
    assert len(roots)==36 and len(np.unique(indices[:, 0]))==36
    write(out/'evaluation_started.json', dict(utc=average.timestamp(), training_sha256=sha(out/'training.json'),
        final_kernels_sha256=sealed, evaluation_seed=p.CONFIG['evaluation_seed']))
    returns={}; energies={}; guards={}; validity={}
    for name, theta in final.items():
        ledger.scope='evaluation_'+name
        records=early.collect(engine, theta, roots, indices[:, 1], p.CONFIG['evaluation_seed'],
            ledger, 'full_kernel_rollouts', 64)
        phase.save_records(out/f'{name}_final_paths.npz', records)
        values, rewards=pack_returns(records, 36, 64); returns[name]=values
        energies[name]=old.diagonal(engine, theta, context, 'validation', np.arange(512), 32,
            p.CONFIG['validation_seed'], ledger, 'validation_diagonal')
        guards[name]=float(old.diagonal(engine, theta, context, 'train', np.arange(256), 16,
            p.CONFIG['guard_seed'], ledger, 'final_train_guard').mean())
        assert p.guard_accept(guards[name], training['initial_guard'])
        validity[name]=constraints(engine, theta, context, ledger)
        np.savez_compressed(out/f'{name}_evaluation.npz', returns=values, rewards=rewards,
            diagonal_energy=energies[name], episode=indices[:, 0], group=groups, h=50-indices[:, 1])
        print(name+': independent full-kernel evaluation complete.', flush=True)
    results=analyze(returns, energies, groups, context['validation_episode'], guards, validity, training)
    cost=costs(ledger)
    write(out/'results.json', results); write(out/'costs.json', cost)
    assert sha(out/'final_kernels.npz')==sealed
    report(out, results, training, cost, ledger)
