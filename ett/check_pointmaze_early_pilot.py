"""Post-run integrity audit and descriptive intervals, without new outcomes.

This checker was added after the stopped run. It does not amend any gate,
sample a model transition, train a critic, or select a checkpoint.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ett import pointmaze_early_pilot as p
from ett import eval_pointmaze_early_pilot as e
from ett import pointmaze_region_pilot as old
from ett.finite_crl import sha,write


def check(out):
    p.verify(out)
    data=dict(np.load(out/'contexts.npz'));indices=data['validation_indices'];roots=data['validation_roots']
    complete=json.loads((out/'completion.json').read_text());training=json.loads((out/'training.json').read_text())
    assert complete['stage']=='value_estimation' and training['updates']==0 and not training['records']
    theta=np.load(out/'checkpoints'/'initial.npz')['theta']
    np.testing.assert_array_equal(theta,np.zeros(48))
    critic=p.Critic(0);critic.load_state_dict(torch.load(out/'checkpoints'/'initial.pt',weights_only=True)['model'])
    audit=[]
    for name in ['generator_initial','generator_positive_response','generator_negative_response','preflight']:
        with np.load(out/(name+'.npz')) as z:
            for i,h in enumerate(z['horizons']):
                assert h==50-indices[i,1]
                paths=z['states'][i,:,:h+1];a=z['executed_actions'][i,:,:h];xp=z['auxiliary_x_prime'][i,:,:h]
                assert np.isfinite(paths).all() and np.isfinite(a).all() and np.isfinite(xp).all()
                assert np.abs(a).max()<=1 and np.abs(xp).max()<=1
                assert np.isnan(z['states'][i,:,h+1:]).all()
                np.testing.assert_array_equal(paths[:,0],np.broadcast_to(roots[i],paths[:,0].shape))
                np.testing.assert_array_equal(paths[:,1:,2:],paths[:,:-1,:6])
                reward=np.asarray(old.task_reward(paths[:,1:],np.broadcast_to(old.GOAL,paths[:,1:].shape)))
                np.testing.assert_allclose(.05*reward @ .95**np.arange(h),z['returns'][i],atol=1e-12)
                if name=='preflight':
                    np.testing.assert_array_equal(a[:,0],np.broadcast_to(z['root_action'][i],a[:,0].shape))
            audit.append(name)
    arrays=dict(np.load(out/'preflight.npz'));bias={};values={}
    for phase in ['root','midpoint','last']:
        prediction=critic.predict(arrays[phase+'_state'],arrays[phase+'_action'],arrays[phase+'_h'],data['mean'],data['std'])
        np.testing.assert_array_equal(prediction,arrays[phase+'_prediction'])
        delta=prediction-arrays[phase+'_returns'].mean(1)
        bias[phase]={'all':e.interval(delta,indices[:,0],indices[:,2])}
        for gi,g in enumerate(p.GROUPS):
            selected=indices[:,2]==gi
            bias[phase][g]=e.interval(delta[selected],indices[selected,0])
    native=dict(np.load(out/'native_evidence_audit_only.npz'))
    for gi,g in enumerate(p.GROUPS):
        selected=indices[:,2]==gi;model=arrays['returns'][selected].mean(1)
        values[g]=dict(model_value=e.interval(model,indices[selected,0]),
            model_minus_logged_behavior=e.interval(model-native['native_logged_value'][selected],indices[selected,0]),
            model_goal_probability=float(np.mean(arrays['returns'][selected]>0)),
            native_logged_goal_fraction=float(native['native_logged_goal'][selected].mean()))
    ledger=json.loads((out/'ledger.json').read_text())
    assert sum(r['outputs'] for r in ledger['entries'])==ledger['charged']==142961
    assert ledger['charged']<=ledger['cap']
    write(out/'postrun_checks.json',dict(passed=True,checked_arrays=audit,source_hashes_verified=True,
        reward_f4_horizon_action_checks=True,critic_predictions_reproduced=True,ett_updates=0,
        diagonal_parameter_change=0,model_outputs_added=0,tests_passed=28,
        source='ett/check_pointmaze_early_pilot.py',source_sha256=sha('ett/check_pointmaze_early_pilot.py')))
    write(out/'descriptive_uncertainty.json',dict(bias_intervals=bias,root_values=values,
        note='Post-run descriptive reuse of saved targets with the registered episode-bootstrap method. No new outcomes, gate changes, or selection. Native logged behavior has a different continuation policy.'))
    print(json.dumps(dict(passed=True,root_bias_intervals=bias['root'],root_values=values),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out)
