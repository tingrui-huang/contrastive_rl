"""Read-only postrun audit; no fitting or new model outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ett import pointmaze_future_average as p
from ett import eval_pointmaze_future_average as e
from ett import pointmaze_phase_sampling as prior
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write, sha


def check(out):
    p.verify(out); torch.set_num_threads(1)
    train=json.loads((out/'training.json').read_text())
    records=prior.load_records(p.PREVIOUS/'shared_training_paths.npz')
    with np.load(out/'training_probabilities.npz') as z:
        np.testing.assert_array_equal(z['p'], p.future_probabilities(records))
    queries=dict(np.load(out/'queries.npz'))
    replay=e.select_queries(prior.load_records(out/'visitation_paths.npz'), 36)
    for key in replay: np.testing.assert_array_equal(queries[key], replay[key])
    planned=json.loads((out/'evaluation_budget.json').read_text())
    ledger=json.loads((out/'ledger.json').read_text())
    assert ledger['charged']==sum(row['outputs'] for row in ledger['entries'])==planned['planned']
    assert ledger['charged']<=p.CONFIG['model_output_cap']
    assert train['total_steps']==6000
    context=dict(np.load(prior.PREVIOUS/'contexts.npz'))
    models={}
    for seed in p.CONFIG['seeds']:
        a=train['models'][f'sampled_s{seed}']; b=train['models'][f'averaged_s{seed}']
        assert a['initial_sha256']==b['initial_sha256'] and a['row_stream_sha256']==b['row_stream_sha256']
        for arm in p.CONFIG['arms']:
            name=f'{arm}_s{seed}'; path=out/'checkpoints'/f'{name}.pt'
            assert sha(path)==train['checkpoint_hashes'][name]
            model=early.Critic(0); model.load_state_dict(torch.load(path, weights_only=True)['model'])
            assert old.parameter_sha(model)==train['models'][name]['final_sha256']
            models[name]=model
            if arm=='sampled':
                prev=early.Critic(0)
                prev.load_state_dict(torch.load(p.PREVIOUS/'checkpoints'/f'uniform_s{seed}.pt', weights_only=True)['model'])
                assert old.parameter_sha(prev)==old.parameter_sha(model)
    arrays=dict(np.load(out/'surrogate_samples.npz')); rep=p.CONFIG['repeats']
    for name in p.candidates():
        raw=dict(np.load(out/(name+'_continuations.npz')))
        for i, h in enumerate(queries['h']):
            assert np.isnan(raw['rewards'][i, :, h:]).all()
            assert np.isin(raw['rewards'][i, :, :h], [0, 1]).all()
            direct=.05*raw['rewards'][i, :, :h]@.95**np.arange(h)
            np.testing.assert_allclose(raw['mc'][i], direct, atol=1e-8, rtol=1e-7)
        np.testing.assert_array_equal(raw['mc'], arrays[name+'_mc'])
        state=raw['successor'].reshape(-1, 8); action=raw['next_action'].reshape(-1, 2)
        h=np.repeat(queries['h']-1, rep)
        np.testing.assert_array_equal(raw['successor'][:, :, 2:], np.broadcast_to(queries['state'][:, None, :6], (36, rep, 6)))
        for model_name, model in models.items():
            pred=(.05*raw['rewards'][:, :, 0]+.95*model.predict(state, action, h,
                context['mean'], context['std']).reshape(36, rep))
            np.testing.assert_allclose(pred, arrays[name+'_'+model_name], atol=1e-12, rtol=1e-12)
    recomputed=e.analyze(arrays, queries, queries['group'])
    # Normalize NumPy values through the same serializer, then compare exact JSON.
    write(out/'audit_recomputed_results.json', recomputed)
    assert json.loads((out/'audit_recomputed_results.json').read_text())==json.loads((out/'results.json').read_text())
    cal=dict(np.load(out/'calibration.npz')); metrics=json.loads((out/'calibration.json').read_text())
    for name, model in models.items():
        pred=model.predict(cal['root_state'], cal['root_action'], cal['root_h'], context['mean'], context['std'])
        np.testing.assert_array_equal(pred, cal['root_'+name])
        row=e.calibration_metrics(pred, cal['root_returns'], cal['root_h'], e.boot_indices(queries['group']))
        assert row['rmse']==metrics['models']['root'][name]['rmse']
    p.verify(out)
    write(out/'audit.json', dict(passed=True, utc=p.timestamp(), model_outputs_generated=0,
        probabilities_recomputed=True, baselines_reproduced=True, paired_rows_and_initializations=True,
        raw_mc_returns_recomputed=True, all_surrogate_predictions_recomputed=True,
        metrics_recomputed=True, sealed_sources_and_prior_inputs_unchanged=True,
        exact_budget=ledger['charged']))
    files=[f for f in out.rglob('*') if f.is_file() and f.name!='manifest.json']
    write(out/'manifest.json', {str(f.relative_to(out)):sha(f) for f in files})
    print('Postrun audit passed; no new model outcomes generated.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__); parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
