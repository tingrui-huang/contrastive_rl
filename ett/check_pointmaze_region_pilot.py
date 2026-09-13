"""Read-only numerical audit of a completed pilot; collects no model outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from ett.pointmaze_region_pilot import CONFIG, H, GAMMA, GOAL, verify, sha, write, task_reward


def check(out):
    verify(out)
    training = json.loads((out/'training.json').read_text())
    results = json.loads((out/'results.json').read_text())
    checked = []
    for record in training['records']:
        with np.load(out/'checkpoints'/(record['name']+'.npz')) as z:
            final, snapshots = z['theta'], z['snapshots']
        assert len(snapshots) == CONFIG['updates']
        np.testing.assert_array_equal(snapshots[0], np.zeros(48))
        for it, row in enumerate(record['history']):
            rng = np.random.default_rng(CONFIG['update_seed'] + record['seed']*100000 + it*1000 + 3)
            directions = rng.normal(size=(CONFIG['directions'],48))
            sigmas = np.r_[np.full(16,.01),np.full(32,.1)]
            terms = np.asarray(row['signed_terms'])
            components = np.einsum('dt,di->ti',terms[:,0]-terms[:,1],directions)/(2*sigmas*len(directions))
            components[0,16:] = 0
            np.testing.assert_allclose(row['diagonal_gradient'],components[0],atol=1e-12)
            np.testing.assert_allclose(row['off_gradient'],components[1],atol=1e-12)
            gradient = components[0]+(components[1] if record['arm']=='joint' else 0.)
            update = gradient*np.r_[np.full(16,.01),np.full(32,.5)]
            for sl,cap in [(slice(0,16),.03),(slice(16,48),.1)]:
                update[sl] *= min(1.,cap/max(np.linalg.norm(update[sl]),1e-12))
            candidate = (snapshots[it]-update).astype(np.float32)
            accepted = row['candidate_train_es'] <= training['initial_train_es']+CONFIG['diagonal_es_tolerance']
            assert row['accepted'] == accepted
            expected = candidate if accepted else snapshots[it]
            actual = snapshots[it+1] if it+1<len(snapshots) else final
            np.testing.assert_array_equal(actual,expected)
        if record['arm']=='diagonal': np.testing.assert_array_equal(final[16:],0.)
        checked.append(record['name'])
    with np.load(out/'contexts.npz') as z:
        roots = z['validation_roots']
        assert not np.intersect1d(z['train_root_episode'],z['validation_root_episode']).size
    first = None
    for name, row in results['models'].items():
        with np.load(out/(name+'_evaluation.npz')) as z:
            paths = z['trajectories']; actions = z['executed_actions']
            np.testing.assert_array_equal(paths[:,:,0],np.repeat(roots[:,None],paths.shape[1],axis=1))
            np.testing.assert_array_equal(paths[:,:,1:,2:],paths[:,:,:-1,:6])
            np.testing.assert_array_equal(actions[:,:,0],np.repeat(z['h10_action'][:,None],paths.shape[1],axis=1))
            if first is None: first=z['h10_action']
            else: np.testing.assert_array_equal(z['h10_action'],first)
            assert np.abs(actions).max() <= 1 and np.abs(z['auxiliary_x_prime']).max() <= 1
            reward = np.asarray(task_reward(paths[:,:,1:],np.broadcast_to(GOAL,paths[:,:,1:].shape)))
            np.testing.assert_allclose((1-GAMMA)*reward @ GAMMA**np.arange(H),z['h10_returns'],atol=1e-12)
            assert abs(row['model_value']-z['h10_returns'].mean())<1e-12
            for key in z.files: assert np.isfinite(z[key]).all(),(name,key)
    ledger = json.loads((out/'ledger.json').read_text())
    assert sum(e['outputs'] for e in ledger['entries']) == ledger['charged'] <= ledger['cap']
    report = dict(passed=True, checked_update_arms=checked, checked_frozen_models=list(results['models']),
        checks=['frozen source/input hashes','paired initializations','signed Gaussian estimator reconstruction',
            'descent sign and block caps','acceptance guard and final iterate replay','zero control response',
            'episode separation','first-action alignment across repeats/models','separate bounded auxiliary x_prime',
            'complete horizon and exact F4 shift','successor reward reconstruction','finite saved arrays','budget ledger'],
        model_outputs_collected_by_this_audit=0,
        prior_tests='16 finite-state regression tests and 7 PointMaze adapter/integration tests passed; see WORK_LOG.md')
    write(out/'numerical_checks.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out)
