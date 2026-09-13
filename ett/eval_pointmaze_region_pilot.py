"""Independent frozen-generator evaluation; no ETT updates or model selection."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
import torch

from ett.pointmaze_region_pilot import (
    CONFIG, H, GAMMA, GOAL, Kernel, Ledger, RegionCritic, calibration, collect,
    diagonal, fit, positives, verify, write, sha, matrices, validate_selected_set,
    task_reward)


def calibration_rows(critic, arrays, mean, std):
    rows = []
    for h in CONFIG['horizons']:
        state, action, values, outside = [arrays[f'h{h}_{k}'] for k in ['state', 'action', 'returns', 'outside']]
        prediction = critic.predict(state, action, np.full(len(state), h), mean, std)
        arrays[f'h{h}_refit_prediction'] = prediction
        delta = prediction - values.mean(1)
        rows.append(dict(h=h, rmse=float(np.sqrt(np.mean(delta**2))), bias=float(delta.mean()),
            outside_count=int(outside.sum()), outside_rmse=float(np.sqrt(np.mean(delta[outside]**2))) if outside.any() else None,
            return_mean=float(values.mean()), between_context_std=float(values.mean(1).std()),
            max_mc_standard_error=float((values.std(1, ddof=1) / np.sqrt(values.shape[1])).max()),
            value_range_excess=float(max(0., (prediction - (1-GAMMA**h)).max()))))
    passed = all(r['rmse'] <= CONFIG['calibration_rmse_gate'] and abs(r['bias']) <= CONFIG['calibration_bias_gate']
        and (r['outside_rmse'] is None or r['outside_rmse'] <= CONFIG['outside_rmse_gate']) for r in rows)
    passed &= rows[0]['between_context_std'] >= CONFIG['minimum_signal_std']
    return dict(rows=rows, passed=bool(passed))


def paired_interval(difference, clusters, *, stratified=False):
    """Episode-cluster bootstrap; repeats were averaged before this function."""
    difference, clusters = np.asarray(difference), np.asarray(clusters)
    ids, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    if stratified:
        # Root contexts: outside first 32, inside last 32, one per episode.
        assert len(ids) == len(clusters) == 64
        indices = np.concatenate([rng.integers(0, 32, (2000, 32)), rng.integers(32, 64, (2000, 32))], 1)
        boot = difference[indices].mean(1)
    else:
        indices = rng.integers(len(ids), size=(2000, len(ids)))
        boot = sums[indices].sum(1) / counts[indices].sum(1)
    return dict(mean=float(difference.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(), clusters=len(ids))


def constraints(engine, theta, data, ledger):
    s = data['validation_state'][:16]
    key = jax.random.PRNGKey(CONFIG['evaluation_seed'] + 800)
    xp = np.asarray(engine.nominal.sample(s, key, 1, goal=np.broadcast_to(GOAL, s.shape)))
    grid = [np.broadcast_to(np.array([x, y], np.float32), (16, 2)) for x in [-1., 0., 1.] for y in [-1., 0., 1.]]
    actions = grid + [xp, np.clip(xp + .001, -1., 1.)]
    outputs = []
    for a in actions:
        ledger.add(len(s) * 8, 'independent_all_action_component_checks')
        y, d = engine.sample(theta, s, a, xp, key, 8)
        validate_selected_set(s, y, d)
        outputs.append(np.asarray(y))
    excess = 0.
    for i in range(len(actions)):
        for k in range(i):
            distance = np.linalg.norm(outputs[i] - outputs[k], axis=-1)
            excess = max(excess, float(np.max(distance - np.linalg.norm(actions[i]-actions[k], axis=-1)[:, None])))
    norm = float(np.asarray(jax.numpy.linalg.norm(matrices(theta[16:]), axis=(1, 2))).max())
    return dict(samplewise_lipschitz_excess=excess, matrix_frobenius_max=norm,
        geometry_f4_step_action_checks_passed=True,
        passed=bool(excess <= CONFIG['constraint_tolerance'] and norm <= 1 + CONFIG['constraint_tolerance']))


def audit_contexts(path, data):
    """Evaluation-only hidden labels; never passed to policy, critic or selection."""
    episodes = data['validation_root_episode']
    with np.load(path) as source:
        obs = source['obs'][episodes]
        bits = source['swamp_bits'][episodes]
        died = source['entered_active_swamp'][episodes].astype(bool)
    cells = ((3, 3), (4, 3), (5, 3))
    death_row = np.full(len(episodes), -1, int)
    for e in range(len(episodes)):
        if died[e]:
            for t in range(50):
                cell = tuple(np.floor(obs[e, t+1, :2]).astype(int))
                if cell in cells and bits[e, t, cells.index(cell)]:
                    death_row[e] = t+1
                    break
            assert death_row[e] > 0, 'Native death timing could not be reconstructed.'
    dead = (death_row >= 0) & (death_row <= CONFIG['root_time'])
    roots = data['validation_roots']
    outside = np.asarray(task_reward(roots, np.broadcast_to(GOAL, roots.shape))) == 0
    stationary = np.max(np.abs(roots.reshape(-1, 4, 2) - roots[:, None, :2]), axis=(1, 2)) < 1e-6
    rewards = np.asarray(task_reward(obs[:, 41:51, :8], obs[:, 41:51, 8:]))
    # Native death freezes positions outside the goal; audit that reconstruction.
    assert np.all(rewards[dead] == 0)
    native = (1-GAMMA) * rewards @ GAMMA**np.arange(H)
    return dict(already_dead=dead, alive_outside=(~dead) & outside,
        stationary_outside=stationary & outside, inside=~outside), native, death_row


def run(out):
    verify(out)
    if (out/'evaluation_started.json').exists():
        raise ValueError('Preserve the existing independent evaluation attempt.')
    write(out/'evaluation_started.json', dict(status='started'))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    training = json.loads((out/'training.json').read_text())
    data = dict(np.load(out/'contexts.npz')); mean, std = data['mean'], data['std']
    engine, ledger = Kernel(), Ledger(out)
    ledger.data = json.loads((out/'ledger.json').read_text())
    models = ['initial'] + [r['name'] for r in training['records']]
    for record in training['records']:
        assert sha(out/'checkpoints'/(record['name']+'.npz')) == record['theta_sha256']
        assert sha(out/'checkpoints'/(record['name']+'.pt')) == record['critic_sha256']
    results, arrays_by_name = {}, {}
    for name in models:
        theta = np.load(out/'checkpoints'/(name+'.npz'))['theta']
        checkpoint = out/'checkpoints'/('initial_critic.pt' if name == 'initial' else name+'.pt')
        critic = RegionCritic(0); critic.load_state_dict(torch.load(checkpoint, weights_only=True)['model'])
        last, arrays = calibration(engine, theta, critic, data['validation_roots'], mean, std,
            CONFIG['evaluation_seed'], ledger, name+'_independent_calibration')
        # Post-training fit: fresh weights and MC labels, never used for ETT updates.
        fresh = RegionCritic(CONFIG['critic_seed']+100)
        opt = torch.optim.Adam(fresh.parameters(), lr=CONFIG['critic_lr'])
        paths = collect(engine, theta, data['train_roots'], CONFIG['evaluation_seed']+400,
            ledger, name+'_post_training_refit')
        loss = fit(fresh, opt, positives(paths, CONFIG['evaluation_seed']+401, mean, std),
            CONFIG['critic_final_steps'], CONFIG['evaluation_seed']+402)
        refit = calibration_rows(fresh, arrays, mean, std)
        torch.save(dict(model=fresh.state_dict()), out/'checkpoints'/(name+'_refit.pt'))
        es = diagonal(engine, theta, data, 'validation', np.arange(512), 32,
            CONFIG['evaluation_seed']+500, ledger, name+'_independent_diagonal')
        nll = np.asarray(engine.nll(theta, data['validation_state'], data['validation_action'], data['validation_target']))
        assert np.isfinite(es).all() and np.isfinite(nll).all()
        arrays.update(diagonal_es=es, preprojection_nll=nll)
        values = arrays['h10_returns']; paths = arrays['trajectories']
        results[name] = dict(last_training_critic=last, independent_refit=refit, refit_nce_loss=loss,
            model_value=float(values.mean()), zero_return_fraction=float(np.mean(values == 0)),
            maximum_return_fraction=float(np.mean(np.isclose(values, 1-GAMMA**H))),
            independent_diagonal_es=float(es.mean()), preprojection_nll=float(nll.mean()),
            mean_final_xy_repeat_std=float(paths[:, :, -1, :2].std(1).mean()),
            mean_step_distance=float(np.linalg.norm(np.diff(paths[..., :2], axis=2), axis=-1).mean()),
            constraints=constraints(engine, theta, data, ledger), theta=theta,
            checkpoint_sha256=sha(checkpoint), theta_sha256=sha(out/'checkpoints'/(name+'.npz')),
            refit_sha256=sha(out/'checkpoints'/(name+'_refit.pt')))
        arrays_by_name[name] = arrays
        print(name+': independent evaluation complete', flush=True)
    comparisons = {}
    for seed in CONFIG['seeds'] if training['status'] == 'complete' else []:
        joint, control = f'joint_s{seed}', f'diagonal_s{seed}'
        comparisons[joint] = {}
        for reference in [control, 'initial']:
            a, b = arrays_by_name[joint], arrays_by_name[reference]
            np.testing.assert_array_equal(a['h10_state'], b['h10_state'])
            np.testing.assert_array_equal(a['h10_action'], b['h10_action'])
            value = paired_interval((a['h10_returns']-b['h10_returns']).mean(1), data['validation_root_episode'], stratified=True)
            es = paired_interval(a['diagonal_es']-b['diagonal_es'], data['validation_episode'])
            comparisons[joint][reference] = dict(model_value_change=value, diagonal_es_change=es,
                diagonal_noninferiority_passed=es['ci95'][1] <= CONFIG['diagonal_es_tolerance'])
        comparisons[joint]['success'] = bool(
            results[joint]['last_training_critic']['passed'] and results[joint]['independent_refit']['passed']
            and results[joint]['constraints']['passed']
            and all(comparisons[joint][r]['diagonal_noninferiority_passed'] for r in [control, 'initial'])
            and comparisons[joint][control]['model_value_change']['ci95'][1] < 0)
    # Hidden information is first read after all models and critics are frozen.
    provenance = json.loads((out/'provenance.json').read_text())
    groups, native, death_row = audit_contexts(provenance['paths']['dataset'], data)
    for name, arrays in arrays_by_name.items():
        strata = {}
        for group, selected in groups.items():
            paths = arrays['trajectories'][selected]
            values = arrays['h10_returns'][selected]
            strata[group] = dict(count=int(selected.sum()), model_value=float(values.mean()) if len(values) else None,
                model_any_goal_probability=float((values > 0).mean()) if len(values) else None,
                model_any_motion_probability=float((np.max(np.abs(paths[:, :, 1:, :2]-paths[:, :, :1, :2]), axis=(2, 3)) > 1e-6).mean()) if len(values) else None,
                native_logged_behavior_value=float(native[selected].mean()) if selected.any() else None)
        results[name]['audit_only_strata'] = strata
        np.savez_compressed(out/(name+'_evaluation.npz'), **arrays)
    np.savez_compressed(out/'audit_only.npz', **groups, native_logged_behavior_return=native, death_observation_row=death_row)
    write(out/'results.json', dict(training_status=training['status'], models=results, comparisons=comparisons,
        budget=ledger.data, uncertainty='2000 paired episode-cluster bootstrap replicates; repeats averaged; fixed strata'))
    lines = ['# PointMaze region-NCE integration pilot', '',
        'This is model-continuation evidence, not native fixed-actor performance, a certified global worst case, or causal identification.', '',
        f"Training status: {training['status']}; {training['updates']} ETT updates. Final iterates only. Charged model outputs: {ledger.data['charged']:,}/{ledger.data['cap']:,}.", '',
        'The inherited rectangle generator is unchanged except for 16 trainable diagonal-head bias offsets and 32 response parameters. Actor and state-goal nominal policy are frozen. The emitted XY energy score fits the diagonal; pre-projection mixed NLL is a separate diagnostic. Samplewise Euclidean action L=1 implies a coupled Wasserstein bound, not total variation. The old geometry excludes some valid fork transitions and does not guarantee native physics.', '',
        'The adapter classifies sampled future reward-region membership with binary NCE, q=(0.5,0.5), B=32, alpha=0, and remaining horizon h. Its decoded occupancy is (1-0.95^h)*31*0.5*exp(f_1), with Q_0=0. It is separate from the production full-F4 point-goal critic. Values are normalized by (1-gamma)=0.05; unnormalized task returns are 20 times larger.', '',
        '## Independent same-generator calibration and final models', '']
    for name, r in results.items():
        lines += [f'### {name}', '',
            f"Model occupancy {r['model_value']:.6f}; zero/max return fractions {r['zero_return_fraction']:.3f}/{r['maximum_return_fraction']:.3f}. Diagonal energy score {r['independent_diagonal_es']:.6f}; pre-projection NLL {r['preprojection_nll']:.6f}. Constraints passed: {r['constraints']['passed']}; maximum numerical Lipschitz excess {r['constraints']['samplewise_lipschitz_excess']:.3g}.", '',
            f"Mean per-coordinate final XY repeat standard deviation {r['mean_final_xy_repeat_std']:.5f}; mean step distance {r['mean_step_distance']:.5f}.", '']
        for tag in ['last_training_critic', 'independent_refit']:
            lines += [f"{tag} (gate passed: {r[tag]['passed']}):", '']
            for row in r[tag]['rows']:
                lines += [f"- h={row['h']}: RMSE {row['rmse']:.5f}, bias {row['bias']:.5f}, outside-region RMSE {row['outside_rmse']}, maximum target MC SE {row['max_mc_standard_error']:.5f}, predicted range excess {row['value_range_excess']:.5f}."]
            lines += ['']
    lines += ['## Paired update results', '']
    if comparisons:
        for name, c in comparisons.items():
            lines += [f"{name}: estimator-and-update success = {c['success']}.", '']
            for ref, row in c.items():
                if ref == 'success': continue
                lines += [f"- Versus {ref}: occupancy change {row['model_value_change']}; diagonal ES change {row['diagonal_es_change']}; diagonal noninferiority {row['diagonal_noninferiority_passed']}."]
            lines += ['']
    else:
        lines += ['The predeclared pre-optimization calibration gate failed. No ETT optimization was run, so no model-value improvement or joint/control comparison is claimed. The independent evaluation and separate fixed-budget refit diagnose estimator transfer only; neither authorizes a retry.', '']
    lines += ['## Native-label audits and limitations', '',
        'Native death timing is reconstructed only after training, using logged hidden bits and episode flags. Those fields never enter context selection, critic inputs, labels for fitting, or ETT updates. A stationary atom is not model death; the generator has no persistent death state. Alive outside-goal contexts are only recovery opportunities, not certified recoverable states.', '']
    for name, r in results.items():
        lines += [f"- {name}: audit-only context results: {json.dumps(r['audit_only_strata'])}"]
    lines += ['', 'F4 histories shift exactly and both action channels remain bounded. Complete 10-step model continuations end at original task time 50. Logged subsequent behavior returns are supplementary native evidence from a different continuation policy, not hidden-context conditional ETT ground truth. Root selection is late-task and balanced inside/outside the goal region; results do not estimate reset returns. Actor training overlaps these source episodes. Confidence intervals cluster contexts/diagonal rows by episode and pair common random streams; they do not imply identical trajectories for different kernels.', '',
        'Recommendation: stop at this pilot. ' + ('The paired results above determine whether an estimator-and-update effect was established; even a passing seed does not justify actor integration or a global pessimism claim.' if comparisons else 'The failed initial estimator gate is the immediate blocker; do not optimize ETT or integrate with actor training on this evidence.'), '',
        '## Reproduction', '', 'Run from the PointMaze worktree with the local input checkpoints/dataset listed and hashed in provenance.json. Checkpoints remain local; configuration, prediction arrays, audit arrays and reports are shareable.', '', '```powershell',
        'python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback scripts.test_pointmaze_region_pilot -v',
        'python -m ett.pointmaze_region_pilot prepare --out artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1',
        'python -m ett.pointmaze_region_pilot run --out artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1',
        'python -m ett.eval_pointmaze_region_pilot --out artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1', '```', '',
        'Use a fresh output directory for a literal reproduction; existing attempts cannot be overwritten. PROTOCOL.md was saved and hashed before outcomes. results.json and *_evaluation.npz contain full metrics, trajectories, execution actions, separate auxiliary x_prime draws, and common-target last/refit predictions.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    verify(out)
    write(out/'evaluation_manifest.json', {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'evaluation_manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
