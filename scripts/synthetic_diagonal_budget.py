"""Three matched scalar runs with predetermined lambda=0.004 and epsilon=0.01."""
import argparse
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from scripts.synthetic_failure_bank import CONFIG, sample_data, train
from scripts.synthetic_lipschitz_response import read, load_checkpoint
from scripts.synthetic_shared_response import sha, write_json
from scripts.synthetic_budget_reference import DERIVATION, population_reference
from scripts.eval_synthetic_failure_bank import evaluate as evaluate_previous
from scripts.eval_synthetic_diagonal_budget import evaluate


PRIOR = Path('artifacts/synthetic_failure_bank/bank_m3_lambda01_s012')
SOURCE_FILES = ['scripts/synthetic_diagonal_budget.py', 'scripts/eval_synthetic_diagonal_budget.py',
                'scripts/synthetic_budget_reference.py', 'scripts/synthetic_failure_bank.py',
                'scripts/synthetic_lipschitz_response.py', 'scripts/eval_synthetic_failure_bank.py',
                'scripts/synthetic_failure_reference.py', 'scripts/synthetic_shared_response.py']


def verify_prior():
    """Read-only replay of the latest failure-bank results, before new updates."""
    config, results = read(PRIOR/'config.json'), read(PRIOR/'results.json')
    completion = read(PRIOR/'completion.json')
    assert completion['status']=='complete'
    assert read(PRIOR/'verification.json')['status']=='passed'
    assert config['source_sha256']==sha('scripts/synthetic_failure_bank.py')
    assert config['architecture_source_sha256']==sha('scripts/synthetic_lipschitz_response.py')
    assert config['reference_source_sha256']==sha('scripts/synthetic_failure_reference.py')
    assert config['reference_derivation_sha256']==sha(PRIOR/'REFERENCE.md')
    for key, value in CONFIG.items():
        assert config[key]==value, key
    for path, digest in read(PRIOR/'original_hashes.json').items():
        assert sha(path)==digest, path
    data = sample_data(config['evaluation_seed'], config['eval_diagonal'], config['eval_off'])
    for name, digest in completion['checkpoints'].items():
        assert sha(PRIOR/f'{name}.pt')==digest
        model, checkpoint = load_checkpoint(PRIOR/f'{name}.pt', True)
        assert checkpoint['config']==config
        metrics, arrays = evaluate_previous(model, config, data)
        for key, value in metrics.items():
            assert value==results[name][key], (name, key)
        with np.load(PRIOR/f'{name}_evaluation.npz') as saved:
            assert set(saved.files)==set(arrays)
            for key in saved.files:
                np.testing.assert_array_equal(saved[key], arrays[key])
    return dict(status='passed', checkpoint_replays=len(completion['checkpoints']),
                all_old_metrics_and_arrays_exact=True, source_config_and_checkpoint_hashes_match=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError('use a fresh output directory')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance = verify_prior()
    print('Replayed all six historical checkpoints and verified provenance.', flush=True)
    config = dict(CONFIG, lambda_off=.004, epsilon=.01, arms=['bank_joint'], smoke=args.smoke)
    if args.smoke:
        config.update(steps=5, seeds=[0], monitor_every=1)
    config.update(dtype='float64', device='cpu', python_version=platform.python_version(),
        numpy_version=np.__version__, torch_version=str(torch.__version__),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p: sha(p) for p in SOURCE_FILES}, prior_directory=PRIOR.as_posix(),
        objective='mean(G(diagonal)^2) + 0.004*mean((G(off)+3)^2); exactly two terms',
        initialization='original seed initialization through unchanged historical train(); no trained starting checkpoint',
        evaluation_policy='already-inspected evaluation inputs, fixed final step, no selection or tuning',
        nominal_marginal='same uniform xp marginal, exactly shared empirical contexts; independent minibatches',
        checkpoint_policy='local .pt files only; no automatic commit or push')
    old_hashes = {p.as_posix(): sha(p) for p in PRIOR.rglob('*') if p.is_file()}
    old_hashes.update(read(PRIOR/'original_hashes.json'))
    old_hashes.update({p: sha(p) for p in SOURCE_FILES[3:]})
    out.mkdir(parents=True)
    (out/'REFERENCE.md').write_text(DERIVATION, encoding='utf-8')
    config['reference_derivation_sha256'] = sha(out/'REFERENCE.md')
    write_json(out/'config.json', config)
    write_json(out/'original_hashes.json', old_hashes)
    write_json(out/'prior_verification.json', provenance)
    write_json(out/'population_reference.json', {str(w): population_reference(w) for w in [0., .004, .1]})
    data = sample_data(config['evaluation_seed'], config['eval_diagonal'], config['eval_off'])
    monitor = sample_data(config['monitor_seed'], 1024, 4096)
    results, history, checkpoints = {}, {}, {}
    old_results, old_history = read(PRIOR/'results.json'), read(PRIOR/'history.json')
    start = time.perf_counter()
    # Re-evaluation adds weight-specific metrics and continuum maxima, never updates old files.
    for name, old in old_results.items():
        model, checkpoint = load_checkpoint(PRIOR/f'{name}.pt', True)
        weight = .1 if checkpoint['arm']=='bank_joint' else 0.
        metrics, arrays = evaluate(model, config, data, weight)
        metrics.update(seed=checkpoint['seed'], reused=True, checkpoint=(PRIOR/f'{name}.pt').as_posix())
        key = f'prior_{name}'
        results[key] = metrics
        history[key] = [{k: v for k, v in row.items() if k!='common_lambda01_total'} for row in old_history[name]]
        np.savez_compressed(out/f'{key}_evaluation.npz', **arrays)
    for seed in config['seeds']:
        training = sample_data(config['train_seed_base']+seed, config['train_diagonal'], config['train_off'])
        model, rows, detail = train('bank_joint', seed, config, training, monitor)
        for row in rows:
            row['weighted_total'] = row.pop('common_lambda01_total')
        old = old_results[f'bank_joint_s{seed}']
        match_keys = ['initial_state_sha256', 'data_sha256']
        if not args.smoke:
            match_keys.append('batch_schedule_sha256')
        assert all(detail[k]==old[k] for k in match_keys)
        metrics, arrays = evaluate(model, config, data, config['lambda_off'])
        metrics.update(detail, seed=seed, reused=False, matched_to_prior=match_keys)
        name = f'budget_s{seed}'
        results[name], history[name] = metrics, rows
        torch.save(dict(state_dict=model.state_dict(), seed=seed, config=config), out/f'{name}.pt')
        checkpoints[name] = sha(out/f'{name}.pt')
        np.savez_compressed(out/f'{name}_evaluation.npz', **arrays)
        write_json(out/'results.json', results)
        write_json(out/'history.json', history)
        print(f'{name}: diagonal RMSE={metrics["diagonal"]["rmse"]:.7f}, continuum max='
              f'{metrics["continuum_diagonal"]["max_abs"]:.7f}, relative RMSE='
              f'{metrics["relative_response_error"]["rmse"]:.7f}', flush=True)
    assert all(sha(path)==digest for path, digest in old_hashes.items())
    write_json(out/'completion.json', dict(status='complete', new_runs=len(checkpoints),
        checkpoints=checkpoints, historical_files_unchanged=True, elapsed_seconds=time.perf_counter()-start))


if __name__=='__main__':
    main()
