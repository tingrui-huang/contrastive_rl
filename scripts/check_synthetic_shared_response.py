"""Reload local final checkpoints and independently audit saved synthetic results."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.synthetic_shared_response import SharedResponse, make_data, predict, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    read = lambda name: json.loads((root / name).read_text())
    config, results, history, completion = [read(name + '.json') for name in ('config', 'results', 'history', 'completion')]
    assert config['source_sha256'] == sha(Path(__file__).with_name('synthetic_shared_response.py'))
    torch.set_num_threads(config['threads'])
    assert completion['status'] == 'complete'
    assert len(results) == len(config['L_values']) * len(config['seeds']) * len(config['arms'])
    checks = {}
    scale_reference = {}
    train_diagonal = {seed: make_data(830000 + seed, config['train_diagonal'], config['train_off'])[0][:, 0]
                      for seed in config['seeds']}
    for name, result in results.items():
        path = root / f'{name}.pt'
        assert sha(path) == completion['checkpoint_sha256'][path.name]
        # The locally saved config contains PyTorch's string-like version type.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = SharedResponse(checkpoint['arm'] == 'signed_difference', config['hidden_width'])
        model.load_state_dict(checkpoint['state_dict'])
        with np.load(root / f'{name}_evaluation.npz', allow_pickle=False) as saved:
            off = predict(model, saved['off_pairs'])
            diagonal = predict(model, saved['diagonal_pairs'])
            np.testing.assert_array_equal(off, saved['off_prediction_over_L'])
            np.testing.assert_array_equal(diagonal, saved['diagonal_prediction_over_L'])
            pairs = saved['off_pairs'].astype(float)
            mse = np.mean((off + np.abs(pairs[:, 0] - pairs[:, 1]))**2)
            assert np.isclose(mse, result['off']['normalized_mse'], rtol=1e-12, atol=0)
            assert np.isclose(np.mean(diagonal**2), result['diagonal']['normalized_mse'], rtol=1e-12, atol=0)
            unique_mask = ~np.isin(saved['diagonal_pairs'][:, 0], train_diagonal[result['seed']])
            duplicate_count = int((~unique_mask).sum())
            unique_rmse = float(np.sqrt(np.mean(diagonal[unique_mask]**2)))
            key = (result['arm'], result['seed'])
            if key in scale_reference:
                np.testing.assert_array_equal(off, scale_reference[key])
            else:
                scale_reference[key] = off.copy()
            assert np.all(np.abs(pairs) <= 1) and np.all(pairs[:, 0] != pairs[:, 1])
        assert history[name][-1]['step'] == config['steps']
        weight = 0 if result['arm'] == 'diagonal_only' else config['lambda_off']
        for record in history[name]:
            assert all(np.isfinite(v) for v in record.values())
            assert np.isclose(record['physical_total'], record['physical_diag'] + weight * record['physical_off'], rtol=1e-6, atol=1e-12)
        checks[name] = dict(checkpoint_hash=True, exact_prediction_replay=True, final_step=True,
                           objective_two_terms=True, action_bounds=True, cross_L_normalized_prediction_equal=True,
                           independent_float32_diagonal_coincidences=duplicate_count,
                           strictly_nonoverlapping_diagonal_rmse_over_L=unique_rmse)
    write_json(root / 'verification.json', dict(status='passed', checks=checks,
        count=len(checks), verification_source_sha256=sha(__file__),
        note='Empirical ratios are diagnostics; checkpoint replay does not certify Lipschitz compliance.'))
    print(f'Passed independent verification of {len(checks)} final checkpoints.')


if __name__ == '__main__':
    main()
