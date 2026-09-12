"""Reproduce every saved budget evaluation and verify matched training provenance."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.synthetic_failure_bank import CONFIG, initialize, sample_data, state_sha
from scripts.synthetic_lipschitz_response import array_sha, load_checkpoint, read
from scripts.synthetic_shared_response import sha, write_json
from scripts.synthetic_diagonal_budget import PRIOR
from scripts.eval_synthetic_diagonal_budget import evaluate
from scripts.synthetic_budget_reference import fixed_class_reference, population_reference
from scripts.synthetic_failure_reference import quadrature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    config, results = read(root/'config.json'), read(root/'results.json')
    completion, histories = read(root/'completion.json'), read(root/'history.json')
    old = read(PRIOR/'results.json')
    assert completion['status']=='complete'
    assert read(root/'prior_verification.json')['status']=='passed'
    for path, digest in config['source_sha256'].items():
        assert sha(path)==digest, path
    assert sha(root/'REFERENCE.md')==config['reference_derivation_sha256']
    for path, digest in read(root/'original_hashes.json').items():
        assert sha(path)==digest, path
    for key, value in CONFIG.items():
        if key not in ['lambda_off', 'arms'] and not (config['smoke'] and key in ['steps', 'seeds', 'monitor_every']):
            assert config[key]==value, key
    assert config['lambda_off']==.004 and config['epsilon']==.01
    assert completion['new_runs']==(1 if config['smoke'] else 3)
    torch.set_num_threads(1)
    data = sample_data(config['evaluation_seed'], config['eval_diagonal'], config['eval_off'])
    np.testing.assert_array_equal(np.repeat(data[0][:, 1], 4), data[1][:, 1])
    checks = {}
    for name, saved in results.items():
        path = Path(saved['checkpoint']) if saved['reused'] else root/f'{name}.pt'
        if not saved['reused']:
            assert sha(path)==completion['checkpoints'][name]
        model, checkpoint = load_checkpoint(path, True)
        metrics, arrays = evaluate(model, config, data, saved['lambda_off'])
        for key, value in metrics.items():
            assert saved[key]==value, (name, key)
        with np.load(root/f'{name}_evaluation.npz') as archive:
            assert set(archive.files)==set(arrays)
            for key in archive.files:
                np.testing.assert_array_equal(arrays[key], archive[key])
        for row in histories[name]:
            assert np.isclose(row['training_total'], row['diagonal_mse']+saved['lambda_off']*row['failure_mse'], rtol=1e-14)
            assert all(np.isfinite(v) for v in row.values())
        assert histories[name][-1]['step']==(CONFIG['steps'] if saved['reused'] else config['steps'])
        for key in ['arbitrary_pairs', 'learned_anchor', 'grid_adjacent']:
            assert metrics['bounds'][key]['largest_positive_absolute_excess'] <= config['fp_absolute_excess_tolerance']
        assert metrics['intervals']['max_exact_interval_slope']<=1.
        if not saved['reused']:
            assert checkpoint['config']==config
            seed = saved['seed']
            assert state_sha(initialize(seed))==saved['initial_state_sha256']
            training = sample_data(config['train_seed_base']+seed, config['train_diagonal'], config['train_off'])
            np.testing.assert_array_equal(np.repeat(training[0][:, 1], 4), training[1][:, 1])
            assert [array_sha(a) for a in training]==saved['data_sha256']
            schedule = hashlib.sha256()
            rng = np.random.default_rng(config['batch_seed_base']+seed)
            for _ in range(config['steps']):
                schedule.update(rng.integers(config['train_diagonal'], size=config['diagonal_batch']).tobytes())
                schedule.update(rng.integers(config['train_off'], size=config['off_batch']).tobytes())
            assert schedule.hexdigest()==saved['batch_schedule_sha256']
            for key in saved['matched_to_prior']:
                assert saved[key]==old[f'bank_joint_s{seed}'][key]
            assert saved['all_parameters_trainable'] and not saved['bank_has_gradient']
            assert all(saved['changed_parameter_tensors'].values())
        checks[name] = dict(all_metrics_exact=True, all_arrays_exact=True, objective_arithmetic=True,
                           full_action_bounds=True, final_step=True, continuum_diagonal_reproduced=True)
    refs = {}
    for weight in [.004, .1]:
        q = quadrature(fixed_class_reference(weight), 256)
        ref = population_reference(weight)
        gap = q['diagonal_mse']+weight*q['failure_mse']-ref['total']
        assert abs(gap-ref['class_reference_population_gap'])<1e-12
        refs[str(weight)] = dict(integrated_class_reference_gap=gap, analytic=ref)
    write_json(root/'verification.json', dict(status='passed', checks=checks, references=refs,
        historical_files_unchanged=True, original_initialization_data_and_minibatch_hashes_match=True,
        source_sha256=sha(__file__)))
    print(f'Passed {len(checks)} exact evaluation replays, matched provenance and all constraint checks.')


if __name__=='__main__':
    main()
