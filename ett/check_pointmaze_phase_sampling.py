"""Read-only post-run integrity checks; no model continuations or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ett import pointmaze_phase_sampling as p
from ett import eval_pointmaze_phase_sampling as e
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write,sha


def check(out):
    p.verify(out);torch.set_num_threads(1)
    context=dict(np.load(p.PREVIOUS/'contexts.npz'));training=json.loads((out/'training.json').read_text())
    paths=p.load_records(out/'shared_training_paths.npz');rows=dict(np.load(out/'shared_training_rows.npz'))
    recreated=p.training_rows(paths,context['train_indices'][:,2],context['mean'],context['std'])
    for key in rows:np.testing.assert_array_equal(rows[key],recreated[key])
    assert sha(out/'shared_training_paths.npz')==training['shared_paths_sha256']
    assert sha(out/'shared_training_rows.npz')==training['shared_rows_sha256']
    models={}
    for seed in [0,1]:
        initial=old.parameter_sha(early.Critic(98000000+seed))
        for arm in ['uniform','balanced']:
            name=f'{arm}_s{seed}';info=training['models'][name]
            assert info['initial_parameters_sha256']==initial and info['steps']==1500
            counts=np.zeros((3,3),int);rng=np.random.default_rng(98000000+seed)
            for _ in range(1500):
                idx=p.sample_rows(rng,rows,arm)
                counts+=np.bincount(rows['phase'][idx]*3+rows['group'][idx],minlength=9).reshape(3,3)
            np.testing.assert_array_equal(counts,info['row_exposures'])
            checkpoint=out/'checkpoints'/(name+'.pt')
            assert sha(checkpoint)==training['checkpoint_hashes'][name]
            critic=early.Critic(0);critic.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
            assert old.parameter_sha(critic)==info['final_parameters_sha256'];models[name]=critic
    old_critic=early.Critic(0);old_critic.load_state_dict(torch.load(p.PREVIOUS/'checkpoints'/'initial.pt',weights_only=True)['model'])
    assert old.parameter_sha(old_critic)==old.parameter_sha(models['uniform_s0'])
    final_selection=json.loads((out/'final_selection.json').read_text())
    assert final_selection['collection_after_training_sha256']==sha(out/'training.json')
    assert sha(out/'local_inputs'/'fresh_context_source.npz')==final_selection['native_source_sha256']
    assert sha(out/'final_contexts.npz')==final_selection['selected_contexts_sha256']
    with np.load(out/'local_inputs'/'fresh_context_source.npz') as z:obs=z['obs']
    selected,counts=early.select_roots(obs,np.arange(600),103000001)
    with np.load(out/'final_contexts.npz') as z:
        np.testing.assert_array_equal(selected,z['indices'])
        np.testing.assert_array_equal(obs[selected[:,0],selected[:,1],:8],z['roots'])
    assert len(selected)==len(np.unique(selected[:,0]))==36
    for cohort,records in [('training',paths),('final',p.load_records(out/'final_query_paths.npz'))]:
        q=e.query_records(records,36);arrays=dict(np.load(out/(cohort+'_calibration.npz')))
        report=json.loads((out/(cohort+'_calibration.json')).read_text())
        for key,value in q.items():np.testing.assert_array_equal(value,arrays[key])
        for phase in p.PHASES:
            h=arrays[phase+'_h'];returns=arrays[phase+'_returns']
            assert returns.shape==(36,64) and np.isfinite(returns).all()
            assert np.all(returns>=0) and np.all(returns<=1-.95**h[:,None]+1e-6)
            assert np.abs(arrays[phase+'_action']).max()<=1
            for name,critic in models.items():
                pred=critic.predict(q[phase+'_state'],q[phase+'_action'],h,context['mean'],context['std'])
                np.testing.assert_array_equal(pred,arrays[phase+'_'+name+'_prediction'])
                metric=e.calibration(pred,returns,h,arrays['group'])
                np.testing.assert_allclose(metric['rmse'],report['models'][name][phase]['all']['rmse'],atol=1e-14)
    ledger=json.loads((out/'ledger.json').read_text());complete=json.loads((out/'completion.json').read_text())
    assert sum(r['outputs'] for r in ledger['entries'])==ledger['charged']==complete['model_outputs']<=650000
    assert complete['native_steps']==600*50 and complete['optimizer_steps']==6000
    assert complete['ett_updates']==complete['actor_updates']==0
    write(out/'postrun_checks.json',dict(passed=True,tests_passed_before_run=33,model_outputs_added=0,
        checks=['shared path/label replay','paired initialization hashes','row exposure replay','checkpoint hashes',
            'previous uniform baseline exact replay','final collection sealed after training','observable final selection replay',
            'exact query states/actions/remaining horizons','independent target shape and bounds','all saved predictions replay',
            'RMSE replay','source/input hashes','native/model/optimizer budgets','zero ETT/actor updates'],
        source_sha256=sha('ett/check_pointmaze_phase_sampling.py'),note='Read-only audit added after the completed comparison; no gate or model selection change.'))
    print('All post-run integrity checks passed; no outcomes added.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out)
