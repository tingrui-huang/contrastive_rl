"""Reproduce spline results and audit frozen baselines without modifying them."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from scripts.synthetic_shared_response import make_data, sha, write_json
from scripts.synthetic_lipschitz_response import (evaluate_spline, load_checkpoint,
    read, setup_evaluation, array_sha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();root=Path(args.run_dir)
    config=read(root/'config.json');results=read(root/'results.json')
    completion=read(root/'completion.json');history=read(root/'history.json')
    assert completion['status']=='complete'
    assert config['source_sha256']==sha(Path(__file__).with_name('synthetic_lipschitz_response.py'))
    for path,digest in read(root/'original_hashes.json').items():
        assert sha(path)==digest,path
    torch.set_num_threads(config['threads'])
    data=setup_evaluation(config)
    checks={}
    for name,digest in completion['checkpoints'].items():
        assert sha(root/f'{name}.pt')==digest
        model,checkpoint=load_checkpoint(root/f'{name}.pt',True)
        metrics,arrays=evaluate_spline(model,config,data)
        for key,value in metrics.items():
            assert value==results[name][key],(name,key)
        with np.load(root/f'{name}_evaluation.npz',allow_pickle=False) as saved:
            for key in saved.files:
                np.testing.assert_array_equal(arrays[key],saved[key])
        rng=np.random.default_rng(840000+checkpoint['seed']);schedule=hashlib.sha256()
        for _ in range(config['steps']):
            schedule.update(rng.integers(config['train_diagonal'],size=config['diagonal_batch']).tobytes())
            schedule.update(rng.integers(config['train_off'],size=config['off_batch']).tobytes())
        assert schedule.hexdigest()==results[name]['batch_schedule_sha256']
        train_data=make_data(830000+checkpoint['seed'],config['train_diagonal'],config['train_off'])
        assert [array_sha(a) for a in train_data]==results[name]['data_sha256']
        mask=~np.isin(arrays['diagonal_pairs'][:,0],train_data[0][:,0])
        for record in history[name]:
            assert np.isclose(record['normalized_total'],record['normalized_diag']+record['normalized_off'],rtol=1e-14)
            assert record['max_monitor_slope']<=1.
        assert history[name][-1]['step']==config['steps']
        for key in ['anchored_learned_value','arbitrary_random_pairs','grid_adjacent_pairs']:
            assert metrics['bounds'][key]['largest_positive_absolute_excess']<=config['fp_absolute_excess_tolerance']
        assert metrics['intervals']['max_exact_interval_slope']<=1.
        with torch.no_grad():
            raw=model.conditioner(torch.from_numpy(arrays['slope_anchors']).reshape(-1,1)).numpy()[:,1:]
        desired=np.where(np.arange(model.intervals)<model.intervals//2,1.,-1.)
        feasible=arrays['interval_feasible']
        wrong_saturated=(raw*desired[None,:]<-1.) & feasible
        audit=dict(feasible_intervals=int(feasible.sum()),
            wrong_sign_saturated_count=int(wrong_saturated.sum()),
            wrong_sign_saturated_fraction=float(wrong_saturated.sum()/feasible.sum()),
            endpoint_xp1_first_interval_raw=float(raw[-1,0]),
            explanation='Wrong-sign saturation has zero direct clamp derivative; shared conditioner updates can still alter it. This does not isolate the cause of optimization error.')
        checks[name]=dict(all_saved_arrays_exact=True,all_reported_metrics_exact=True,
            batch_schedule_matches_reference_algorithm=True,training_data_matches=True,
            all_parameter_tensors_changed=all(results[name]['all_parameter_tensors_changed'].values()),
            no_output_difference_violation_above_fp_tolerance=True,
            exact_float32_diagonal_coincidences=int((~mask).sum()),
            strictly_nonoverlapping_diagonal_rmse=float(np.sqrt(np.mean(arrays['diagonal_prediction_over_L'][mask]**2))),
            slope_optimization_audit=audit)
    write_json(root/'verification.json',dict(status='passed',checks=checks,original_artifacts_unchanged=True,
        source_sha256=sha(__file__),note='Continuum guarantee follows from the construction; numerical tests audit its implementation.'))
    print(f'Passed exact output/metric reproduction for {len(checks)} final spline checkpoints; old artifacts unchanged.')


if __name__=='__main__':
    main()
