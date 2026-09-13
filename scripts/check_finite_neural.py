"""Audit saved neural experiment without new sampling or refitting."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from ett import finite_crl as b
from ett.finite_crl_eval import certificate, exact_values, objective
from ett.finite_neural import CONFIG, NeuralCritic, array_sha, validate_sources
from ett.finite_neural_eval import diagnostics


def check(out):
    validate_sources(out)
    torch.set_num_threads(1)
    training = json.loads((out / 'training.json').read_text())
    results = json.loads((out / 'results.json').read_text())
    assert training['complete'] and not training['exact_values_used'] and not training['oracle_used']
    tree = ast.parse(Path('ett/finite_neural.py').read_text())
    assert not any(isinstance(node, ast.ImportFrom) and node.module and 'eval' in node.module for node in ast.walk(tree))
    cert = certificate()
    assert cert['J_opt_fraction'] == results['certificate']['J_opt_fraction']
    initials, rngs, initial_neural, first_counts = {}, {}, {}, {}
    excess, parameter_states, reference_checks = 0., 0, 0
    for record, summary in zip(training['records'], results['kernels']):
        arm, seed = record['arm'], record['seed']
        name = f'{arm}_s{seed}'
        assert name == summary['name']
        path = out / 'checkpoints' / (name + '.npz')
        assert b.sha(path) == record['checkpoint_sha256']
        with np.load(path) as saved:
            theta, initial = saved['theta'], saved['initial']
            snapshots, qs, visits, counts = saved['snapshots'], saved['q_history'], saved['visits'], saved['counts']
        if seed in initials:
            np.testing.assert_array_equal(initial, initials[seed])
            np.testing.assert_array_equal(counts[0], first_counts[seed])
        initials[seed], first_counts[seed] = initial, counts[0]
        rng_hashes = [h['rng_after'] for h in record['history']]
        if seed in rngs:
            assert rng_hashes == rngs[seed]
        rngs[seed] = rng_hashes
        for index, (state, q, visitation, count, row) in enumerate(zip(snapshots, qs, visits, counts, record['history'])):
            assert array_sha(count) == row['counts_sha256']
            assert array_sha(visitation) == row['visitation_sha256']
            assert np.all(count[1:].sum(-1) == 512)
            assert np.all(q[0] == 0) and np.isfinite(q).all()
            weight = 0. if arm == 'neural_diagonal' else 4.
            grad = b.loss_gradient(state, q, visitation, weight)[-1]
            expected = b.project(state - .12 * grad)
            np.testing.assert_array_equal(expected, snapshots[index + 1] if index < 47 else theta)
        for state in np.concatenate([snapshots, theta[None]]):
            metrics = b.assert_feasible(state)
            excess = max(excess, *[metrics[k] for k in ('diagonal_excess', 'lipschitz_excess', 'probability_violation', 'simplex_error')])
            parameter_states += 1
        if arm.startswith('neural'):
            if seed in initial_neural:
                assert initial_neural[seed] == record['initial_neural_sha256']
            initial_neural[seed] = record['initial_neural_sha256']
            pt = out / 'checkpoints' / (name + '.pt')
            assert b.sha(pt) == record['neural_checkpoint_sha256']
            model = NeuralCritic(0)
            model.load_state_dict(torch.load(pt, weights_only=True)['model'])
            np.testing.assert_array_equal(model.q_values(), qs[-1])
        else:
            reference = Path(f'artifacts/finite_crl/tabular_h4_s012_v1/checkpoints/joint_s{seed}.npz')
            if reference.exists():
                with np.load(reference) as old:
                    np.testing.assert_array_equal(theta, old['theta'])
                    np.testing.assert_array_equal(snapshots, old['snapshots'])
                reference_checks += 1
        if arm == 'neural_diagonal':
            np.testing.assert_array_equal(theta[:, 1], initial[:, 1])
            assert b.diagonal_loss(theta) < b.diagonal_loss(initial)
        with np.load(out / f'{name}_evaluation.npz') as saved:
            np.testing.assert_array_equal(saved['Q_exact'], exact_values(theta))
            predictions, refit_counts = saved['Q_refits'], saved['counts']
            np.testing.assert_array_equal(predictions[0], b.decode(b.fit_nce(refit_counts)))
            records = [r for r in results['refits'] if r['kernel'] == name]
            assert len(records) == 4
            for index, row in enumerate(records):
                assert array_sha(refit_counts) == row['counts_sha256']
                if index:
                    model = NeuralCritic(0)
                    pt = out / 'checkpoints' / f'refit_{name}_s{index - 1}.pt'
                    model.load_state_dict(torch.load(pt, weights_only=True)['model'])
                    np.testing.assert_array_equal(model.q_values(), predictions[index])
                reproduced = diagnostics(theta, predictions[index])
                assert reproduced['goal'] == row['metrics']['goal']
                assert reproduced['critic_only_gradient'] == row['metrics']['critic_only_gradient']
            paths = saved['path']
            assert paths.shape == (16384, 5)
            rewards = .1 * (paths[:, 1:] == b.G)
            np.testing.assert_allclose(rewards, saved['rewards'], atol=1e-16, rtol=0)
            np.testing.assert_array_equal(saved['returns'], saved['rewards'] @ (b.GAMMA ** np.arange(4)))
        assert summary['J'] == objective(theta)
        assert summary['certified_gap'] == objective(theta) - cert['J_opt']
    assert training['transitions'] == 21233664 and training['neural_steps'] == 39168
    assert results['evaluation_transitions'] == 2064384 and results['refit_adam_steps'] == 27648
    assert len(results['last_training_crossed']) == 81
    assert len(results['refits']) == 36
    b.write(out / 'verification.json', dict(passed=True, checked_parameter_states=parameter_states,
            max_feasibility_excess=excess, old_tabular_checkpoint_matches=reference_checks,
            exact_ett_updates_replayed=432, same_kernel_refits_checked=36,
            crossed_last_training_comparisons=81, new_simulator_steps=0,
            same_initializations_and_rng_streams=True, pretrained_sources_unchanged=True))
    b.write(out / 'manifest.json', {str(p.relative_to(out)): b.sha(p) for p in out.iterdir()
                                  if p.is_file() and p.name != 'manifest.json'})
    print((out / 'verification.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
