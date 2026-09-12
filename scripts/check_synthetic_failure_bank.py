"""Independently replay bank experiment checkpoints, objectives and references."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.synthetic_failure_bank import sample_data, state_sha, initialize
from scripts.synthetic_failure_reference import population_reference, fixed_class_reference, quadrature
from scripts.eval_synthetic_failure_bank import evaluate
from scripts.synthetic_lipschitz_response import read, load_checkpoint, array_sha
from scripts.synthetic_shared_response import sha, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();root=Path(args.run_dir)
    config=read(root/'config.json');results=read(root/'results.json');history=read(root/'history.json')
    completion=read(root/'completion.json');assert completion['status']=='complete'
    assert config['source_sha256']==sha('scripts/synthetic_failure_bank.py')
    assert config['architecture_source_sha256']==sha('scripts/synthetic_lipschitz_response.py')
    assert config['reference_source_sha256']==sha('scripts/synthetic_failure_reference.py')
    assert config['reference_derivation_sha256']==sha(root/'REFERENCE.md')
    for path,digest in read(root/'original_hashes.json').items():assert sha(path)==digest,path
    torch.set_num_threads(config['threads'])
    data=sample_data(config['evaluation_seed'],config['eval_diagonal'],config['eval_off'])
    np.testing.assert_array_equal(np.repeat(data[0][:,1],4),data[1][:,1])
    checks={}
    for name,digest in completion['checkpoints'].items():
        path=root/f'{name}.pt';assert sha(path)==digest
        model,checkpoint=load_checkpoint(path,True)
        metrics,arrays=evaluate(model,config,data)
        for key,value in metrics.items():assert value==results[name][key],(name,key)
        with np.load(root/f'{name}_evaluation.npz') as saved:
            for key in saved.files:np.testing.assert_array_equal(arrays[key],saved[key])
        training=sample_data(config['train_seed_base']+checkpoint['seed'],config['train_diagonal'],config['train_off'])
        assert [array_sha(a) for a in training]==results[name]['data_sha256']
        assert state_sha(initialize(checkpoint['seed']))==results[name]['initial_state_sha256']
        rng=np.random.default_rng(config['batch_seed_base']+checkpoint['seed']);schedule=hashlib.sha256()
        for _ in range(config['steps']):
            schedule.update(rng.integers(config['train_diagonal'],size=config['diagonal_batch']).tobytes())
            schedule.update(rng.integers(config['train_off'],size=config['off_batch']).tobytes())
        assert schedule.hexdigest()==results[name]['batch_schedule_sha256']
        weight=.1 if checkpoint['arm']=='bank_joint' else 0.
        for record in history[name]:
            assert np.isclose(record['training_total'],record['diagonal_mse']+weight*record['failure_mse'],rtol=1e-14)
            assert np.isclose(record['common_lambda01_total'],record['diagonal_mse']+.1*record['failure_mse'],rtol=1e-14)
        assert history[name][-1]['step']==config['steps']
        for key in ['learned_anchor','arbitrary_pairs','grid_adjacent']:
            assert metrics['bounds'][key]['largest_positive_absolute_excess']<=config['fp_absolute_excess_tolerance']
        assert metrics['intervals']['max_exact_interval_slope']<=1.
        dec=metrics['decomposition']
        assert np.isclose(dec['shift_first_failure_reduction']+dec['action_second_failure_reduction'],9-metrics['failure_mse'],atol=1e-14)
        checks[name]=dict(all_metrics_exact=True,all_arrays_exact=True,final_checkpoint=True,
            initialization_and_batch_schedule_reproduced=True,marginal_shared=True,full_action_checks_passed=True)
    for seed in config['seeds']:
        a,b=results[f'diagonal_only_s{seed}'],results[f'bank_joint_s{seed}']
        for key in ['initial_state_sha256','batch_schedule_sha256','data_sha256']:assert a[key]==b[key]
    reference=population_reference();class_ref=quadrature(fixed_class_reference(),256)
    assert abs(class_ref['population_objective_gap']-reference['class_reference_population_gap'])<1e-12
    write_json(root/'verification.json',dict(status='passed',checks=checks,old_artifacts_unchanged=True,
        same_architecture_feasible_reference=class_ref,analytic_reference=reference,source_sha256=sha(__file__)))
    print(f'Passed exact checkpoint replay for {len(checks)} runs, matched controls, full bounds and reference calculation.')


if __name__=='__main__':main()
