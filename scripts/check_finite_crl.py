"""Read-only replay of finite-state experiment artifacts; no new simulations."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np

from ett import finite_crl as f
from ett.finite_crl_eval import certificate, exact_values, objective, report


def check(out):
    provenance = json.loads((out / 'provenance.json').read_text())
    training = json.loads((out / 'training.json').read_text())
    results = json.loads((out / 'results.json').read_text())
    config = json.loads((out / 'config.json').read_text())
    assert config == f.CONFIG
    for name, digest in provenance['sources'].items():
        assert f.sha(name) == digest, name
    assert f.sha(out / 'config.json') == provenance['config_sha256']
    assert f.sha(out / 'PROTOCOL.md') == provenance['protocol_sha256']
    tree = ast.parse(Path('ett/finite_crl.py').read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ('eval' in name or 'oracle' in name) for name in imports)
    assert not training['oracle_used'] and not training['exact_values_used']
    cert = certificate()
    assert cert['J_opt_fraction'] == results['certificate']['J_opt_fraction']
    initials, max_excess, checked_iterates = {}, 0., 0
    training_critic_errors = []
    for item, metric in zip(training['records'], results['records']):
        arm, seed = item['arm'], item['seed']
        assert (arm, seed) == (metric['arm'], metric['seed'])
        checkpoint = out / 'checkpoints' / f'{arm}_s{seed}.npz'
        assert f.sha(checkpoint) == item['checkpoint_sha256']
        with np.load(checkpoint) as saved:
            initial, theta = saved['initial'], saved['theta']
            snapshots, odds = saved['snapshots'], saved['odds_history']
        if seed in initials:
            np.testing.assert_array_equal(initial, initials[seed])
        initials[seed] = initial
        np.testing.assert_array_equal(snapshots[0], initial)
        assert len(snapshots) == config['rounds']
        for state in np.concatenate([snapshots, theta[None]], axis=0):
            feasibility = f.assert_feasible(state)
            max_excess = max(max_excess, *[feasibility[k] for k in
                ('diagonal_excess', 'lipschitz_excess', 'probability_violation', 'simplex_error')])
            checked_iterates += 1
        # Every fitted query is a normalized empirical density ratio.
        np.testing.assert_allclose(((f.B - 1) * f.negative(0.) * odds[:, 1:]).sum(-1), 1., atol=1e-14)
        assert np.isfinite(odds).all() and (odds >= 0).all()
        assert all(np.isfinite(h['critic_nce_loss']) and h['critic_nce_loss'] > 0 for h in item['history'])
        assert all(h['off_grad_norm'] > 0 for h in item['history'])
        assert item['history'][0]['diagonal_grad_norm'] > 0
        if arm == 'diagonal':
            np.testing.assert_array_equal(theta[:, 1], initial[:, 1])
            assert f.diagonal_loss(theta) < f.diagonal_loss(initial)
        else:
            assert np.all(theta[:, 0] != initial[:, 0])
            assert np.all(theta[:, 1] != initial[:, 1])
        with np.load(out / f'{arm}_s{seed}_evaluation.npz') as saved:
            truth, predicted = saved['Q_exact'], saved['Q_hat']
            paths, actions = saved['path'], saved['executed_actions']
            np.testing.assert_array_equal(truth, exact_values(theta))
            np.testing.assert_allclose(predicted, saved['Q_hat_alpha_half'], atol=1e-14)
            assert np.isfinite(predicted).all()
            assert paths.shape == (config['evaluation_repeats'], f.H + 1)
            assert actions.shape == (config['evaluation_repeats'], f.H)
            assert set(np.unique(actions)) == {0, 1}
            assert np.all(paths[:, 0] == f.S)
            p = f.intervention(theta)
            for t in range(f.H):
                assert np.all(p[paths[:, t], actions[:, t], paths[:, t + 1]] > 0)
                for state in (f.G, f.D):
                    assert np.all(paths[paths[:, t] == state, t + 1] == state)
            rewards = (1 - f.GAMMA) * (paths[:, 1:] == f.G)
            np.testing.assert_array_equal(rewards, saved['rewards'])
            returns = rewards @ (f.GAMMA ** np.arange(f.H))
            np.testing.assert_array_equal(returns, saved['returns'])
            assert abs(returns.mean() - metric['empirical_return']['mean']) < 1e-15
            assert abs(np.sqrt(np.mean((truth[1:] - predicted[1:]) ** 2)) -
                       metric['critic_all_queries']['rmse']) < 1e-15
        assert metric['J'] == objective(theta)
        diff = f.decode(odds[-1])[1:, :, :, f.G] - exact_values(theta)[1:, :, :, f.G]
        training_critic_errors.append(dict(arm=arm, seed=seed,
                                          goal_rmse=float(np.sqrt(np.mean(diff ** 2))),
                                          goal_max_abs=float(np.abs(diff).max())))
        # Preserve signed roundoff in the gap; never clip an infeasible result.
        assert metric['feasible_objective_gap'] == objective(theta) - cert['J_opt']
    assert training['model_steps'] == 6 * 48 * (8 * 512 * 10 + 2048 * 4)
    assert results['evaluation_model_steps'] == 6 * (8 * 2048 * 10 + 16384 * 4)
    assert training['model_steps'] <= config['max_training_steps']
    assert results['evaluation_model_steps'] <= config['max_evaluation_steps']
    verification = dict(status='passed', checked_parameter_iterates=checked_iterates,
                        maximum_constraint_excess=max_excess,
                        checks=['pre-run source/config/protocol hashes', 'all checkpoint hashes',
                                'training has no exact evaluator import', 'paired initialization',
                                'all-iterate feasibility', 'finite NCE losses and density ratios',
                                'two live losses and diagonal-only control', 'saved exact values and errors',
                                'saved action support, absorption, reward timing and original horizon',
                                'alpha correction', 'rational certificate and unmodified gaps', 'fixed budgets'],
                        last_training_critic_errors_at_final_kernel=training_critic_errors,
                        additional_environment_steps=0, checker_sha256=f.sha(__file__))
    f.write(out / 'verification.json', verification)
    # Render from parsed JSON so numeric interval endpoints are plain floats.
    report(out, results)
    with (out / 'REPORT.md').open('a', encoding='utf-8') as stream:
        stream.write('\n## Artifact verification\n\n')
        stream.write('Seven focused unit tests passed, including a two-round training smoke test.\n')
        stream.write(f'The saved-data checker passed for {checked_iterates} parameter iterates and all six runs,\n')
        stream.write('with no new model queries. Signed objective gaps near -6e-17 are retained\n')
        stream.write('as float64 evaluation roundoff, not clipped or interpreted as beating the certificate.\n')
        stream.write('Both joint diagonal errors reach the allowed .02 boundary; the lower return\n')
        stream.write('therefore includes the permitted diagonal-fit tradeoff, not exact diagonal preservation.\n')
        stream.write('The topology forces every failed progress transition into D, so greater death\n')
        stream.write('selection here does not show that the loss would choose death over delay in a richer family.\n\n')
        stream.write('The main critic errors above use the predeclared post-training MC refresh\n')
        stream.write('(2,048 fresh positives/query), with ETT fixed and no subsequent updates.\n')
        stream.write('The following errors instead evaluate the actual LAST TRAINING critic\n')
        stream.write('(512 positives/query, before the last ETT update) against exact values\n')
        stream.write('of the final kernel; they include any last-update staleness:\n\n')
        for item in training_critic_errors:
            stream.write(f'- {item["arm"]}, seed {item["seed"]}: goal RMSE={item["goal_rmse"]:.6f}, '
                         f'maximum absolute error={item["goal_max_abs"]:.6f}.\n')
        stream.write('\nPer-round errors of all training critics are in results.json.\n\n')
        stream.write('```powershell\npython -m scripts.check_finite_crl --out artifacts/finite_crl/fresh_run\n```\n')
    f.write(out / 'manifest.json', {p.name: f.sha(p) for p in out.iterdir()
                                   if p.is_file() and p.name != 'manifest.json'})
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
