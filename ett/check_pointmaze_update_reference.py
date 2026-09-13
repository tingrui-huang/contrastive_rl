"""Audit stored updates and returns without emitting new model transitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ett import pointmaze_update_reference as p
from ett import eval_pointmaze_update_reference as e
from ett import pointmaze_phase_sampling as phase
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write, sha


def check(out):
    p.verify(out); torch.set_num_threads(1)
    training=json.loads((out/'training.json').read_text())
    final=dict(np.load(out/'final_kernels.npz'))
    assert sha(out/'final_kernels.npz')==training['final_kernels_sha256']
    context=dict(np.load(phase.PREVIOUS/'contexts.npz'))
    expected_cost=4096; refresh_steps=0; first={}
    for seed in p.CONFIG['seeds']:
        for arm in p.CONFIG['arms']:
            name=f'{arm}_s{seed}'; theta=np.zeros(48, np.float32)
            for iteration, history in enumerate(training['histories'][name]):
                folder=out/'updates'/f'{name}_u{iteration}'
                records=phase.load_records(folder/'paths.npz')
                queries=dict(np.load(folder/'queries.npz')); update=dict(np.load(folder/'update.npz'))
                key=p.CONFIG['update_seed']+seed*100000+iteration*1000
                replay=p.visitation(records, key+102)
                for k, v in replay.items(): np.testing.assert_array_equal(v, queries[k])
                np.testing.assert_array_equal(update['preupdate'], theta)
                proposed, info=p.propose(theta, update['directions'], update['signed'], arm!='diagonal')
                np.testing.assert_array_equal(proposed, update['proposed'])
                assert history['accepted']==p.guard_accept(history['candidate_guard'], training['initial_guard'])
                if history['accepted']: theta=proposed
                np.testing.assert_array_equal(theta, update['accepted_theta'])
                expected_cost+=sum(r['action'].shape[0]*r['action'].shape[1] for r in records)+8*128*8+256*16
                if arm=='critic':
                    critic=early.Critic(0)
                    saved=torch.load(out/'checkpoints'/f'{name}_u{iteration}.pt', weights_only=True)
                    critic.load_state_dict(saved['model'])
                    assert old.parameter_sha(critic)==history['frozen_critic_sha256']
                    assert sha(out/'checkpoints'/f'{name}_u{iteration}.pt')==history['checkpoint_sha256']
                    assert all(float(state['step'])==1500+(iteration+1)*400 for state in saved['optimizer']['state'].values())
                    refresh_steps+=history['critic_refresh']['steps']
                if arm!='diagonal':
                    for d in range(4):
                        for sign in range(2):
                            raw=dict(np.load(folder/f'd{d}_sign{sign}.npz'))
                            np.testing.assert_array_equal(raw['preupdate_theta'], update['preupdate'])
                            expected=update['preupdate']+(1 if sign==0 else -1)*np.r_[np.full(16,.01),np.full(32,.1)]*update['directions'][d]
                            np.testing.assert_array_equal(raw['candidate_theta'], expected.astype(np.float32))
                            expected_cost+=32*4
                            if arm=='mc':
                                for i, h in enumerate(queries['h']-1):
                                    rewards=raw['continuation_rewards'][i]
                                    assert np.isnan(rewards[..., h:]).all()
                                    np.testing.assert_allclose(raw['q'][i], .05*rewards[..., :h]@.95**np.arange(h), atol=1e-12)
                                expected_cost+=int(16*np.sum(queries['h']-1))
                            else:
                                pred=critic.predict(np.repeat(raw['successor'].reshape(-1,8),4,axis=0),
                                    raw['next_action'].reshape(-1,2), np.repeat(queries['h']-1,16), context['mean'],context['std'])
                                np.testing.assert_allclose(raw['q'].ravel(),pred,atol=1e-12)
                            integral=.05*raw['reward'][...,None]+.95*raw['q']
                            np.testing.assert_array_equal(integral,raw['integrand'])
                            value=np.mean(queries['weight']*integral.mean((1,2)))
                            np.testing.assert_allclose(value, update['signed'][d,sign,1],atol=1e-12)
                            if iteration==0:
                                if arm=='critic': first[seed,d,sign]=raw
                                else:
                                    for field in ['candidate_theta','preupdate_theta','successor','next_action','x_prime','reward']:
                                        np.testing.assert_array_equal(raw[field],first[seed,d,sign][field])
            np.testing.assert_array_equal(final[name],theta)
    assert refresh_steps==2400 and training['update_attempts']==18
    returns={}; energies={}; guards={}; validity={}
    results=json.loads((out/'results.json').read_text())
    for name in final:
        raw=dict(np.load(out/f'{name}_evaluation.npz'))
        records=phase.load_records(out/f'{name}_final_paths.npz')
        values,rewards=e.pack_returns(records,36,64)
        np.testing.assert_array_equal(values,raw['returns']); np.testing.assert_array_equal(rewards,raw['rewards'])
        for r in records:
            reward=np.asarray(old.task_reward(r['states'][:,1:],np.broadcast_to(old.GOAL,r['states'][:,1:].shape)))
            np.testing.assert_array_equal(reward,r['reward'])
            np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
        expected_cost+=64*int(raw['h'].sum())+512*32+256*16+12*128
        returns[name]=values; energies[name]=raw['diagonal_energy']
        guards[name]=results['models'][name]['train_guard']; validity[name]=results['models'][name]['constraints']
    recomputed=e.analyze(returns, energies, raw['group'], context['validation_episode'], guards, validity, training)
    write(out/'audit_recomputed_results.json',recomputed)
    assert json.loads((out/'audit_recomputed_results.json').read_text())==results
    ledger=json.loads((out/'ledger.json').read_text())
    assert expected_cost==ledger['charged']==sum(r['outputs'] for r in ledger['entries'])
    assert expected_cost<=p.CONFIG['model_output_cap']
    p.verify(out)
    write(out/'audit.json',dict(passed=True,model_outputs_generated=0,exact_outputs=expected_cost,
        proposals_acceptance_and_final_theta_recomputed=True,first_update_estimators_paired=True,
        current_theta_continuations_verified=True,critic_adam_steps_verified=True,
        raw_surrogate_values_and_full_rollout_returns_recomputed=True,results_recomputed=True,
        sealed_prior_inputs_unchanged=True))
    write(out/'manifest.json',{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'})
    print('Stored-update, first-step pairing, full-rollout and budget audit passed; no new simulations.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__); parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out)
