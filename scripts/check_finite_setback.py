"""Audit saved setback results without new simulations or refits."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import torch
from ett import finite_setback as b
from ett.finite_setback_eval import certificate, diagnostics, exact, objective


def check(out):
    b.verify_sources(out)
    torch.set_num_threads(1)
    training = json.loads((out / 'training.json').read_text())
    result = json.loads((out / 'results.json').read_text())
    assert training['complete'] and not training['oracle_used'] and not training['exact_values_used']
    tree = ast.parse(Path('ett/finite_setback.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and n.module and 'eval' in n.module for n in ast.walk(tree))
    cert = certificate()
    assert result['certificate']['J_opt_fraction'] == cert['J_opt_fraction']
    initials, rngs, counts0 = {}, {}, {}
    violation, checked = 0., 0
    for record, summary in zip(training['records'], result['kernels']):
        name, seed, arm = record['name'], record['seed'], record['arm']
        path = out / 'checkpoints' / (name + '.npz')
        assert b.sha(path) == record['checkpoint_sha256']
        with np.load(path) as saved:
            initial, theta = saved['initial'], saved['theta']
            states, qs, counts, visits = saved['snapshots'], saved['q_history'], saved['counts'], saved['visits']
        if seed in initials:
            np.testing.assert_array_equal(initial, initials[seed])
            np.testing.assert_array_equal(counts[0], counts0[seed])
            assert rngs[seed] == [r['rng_state'] for r in record['history']]
        initials[seed], counts0[seed], rngs[seed] = initial, counts[0], [r['rng_state'] for r in record['history']]
        for index, (state, q, count, visit, log) in enumerate(zip(states, qs, counts, visits, record['history'])):
            assert b.array_sha(count) == log['counts_sha256']
            assert np.isfinite(q).all() and np.all(q[0] == 0)
            weight = 0. if arm == 'neural_diagonal' else 4.
            grad = b.gradient(state, q, visit, weight)[-1]
            np.testing.assert_array_equal(b.project(state - .12 * grad), states[index + 1] if index < 47 else theta)
        for state in np.concatenate([states, theta[None]]):
            violation = max(violation, b.feasibility(state)['violation']); checked += 1
        if arm == 'neural_diagonal':
            np.testing.assert_array_equal(theta[2:], initial[2:])
        if 'neural_sha256' in record:
            pt = out / 'checkpoints' / (name + '.pt')
            assert b.sha(pt) == record['neural_sha256']
            model = b.Critic(0)
            model.load_state_dict(torch.load(pt, weights_only=True)['model'])
            np.testing.assert_array_equal(model.values(), qs[-1])
        with np.load(out / f'{name}_evaluation.npz') as saved:
            np.testing.assert_array_equal(saved['Q_exact'], exact(theta))
            np.testing.assert_array_equal(saved['Q_refits'][0], b.tabular(saved['counts']))
            comparisons = [r for r in result['refits'] if r['kernel'] == name]
            for index, row in enumerate(comparisons):
                assert b.array_sha(saved['counts']) == row['counts_sha256']
                if index:
                    pt = out / 'checkpoints' / f'refit_{name}_s{index - 1}.pt'
                    assert b.sha(pt) == row['refit']['checkpoint_sha256']
                    model = b.Critic(0)
                    model.load_state_dict(torch.load(pt, weights_only=True)['model'])
                    np.testing.assert_array_equal(model.values(), saved['Q_refits'][index])
                metric = diagnostics(theta, saved['Q_refits'][index])
                assert metric['goal_error'] == row['metrics']['goal_error']
                assert metric['gradient_error'] == row['metrics']['gradient_error']
            paths = saved['path']
            assert paths.shape == (16384, 5) and np.all(paths[:, 0] == b.S)
            np.testing.assert_array_equal(saved['rewards'], (1 - b.GAMMA) * (paths[:, 1:] == b.G))
            np.testing.assert_array_equal(saved['returns'], saved['rewards'] @ b.GAMMA ** np.arange(4))
            assert np.all(paths[paths[:, 1] == b.R, 2] == b.P)
            assert np.all(paths[paths[:, 1] == b.R, 3:] == b.G)
            assert np.all(paths[paths[:, 1] == b.D, 1:] == b.D)
        assert summary['J'] == objective(theta)
        assert summary['certified_gap'] == objective(theta) - cert['J_opt']
    assert training['transitions'] == 25657344 and training['neural_updates'] == 39168
    assert result['evaluation_transitions'] == 2433024 and result['refit_steps'] == 27648
    b.write(out / 'verification.json', dict(passed=True, checked_parameter_states=checked,
          max_constraint_violation=violation, updates_replayed=432, common_kernel_refits=36,
          paired_initializations_and_streams=True, reward_recovery_and_absorption=True,
          oracle_training_separation=True, new_simulator_steps=0))
    b.write(out / 'manifest.json', {p.name: b.sha(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
    print((out / 'verification.json').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', required=True, type=Path)
    check(p.parse_args().out)
