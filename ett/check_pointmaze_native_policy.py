"""Read-only routing, policy-constraint, gradient and final outcome audit."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
import torch

from ett import pointmaze_native_policy as p
from ett import eval_pointmaze_native_policy as e
from ett import pointmaze_phase_sampling as phase
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write,sha
from ett.policy_improvement import tree_sha


def check(out):
    p.verify(out);torch.set_num_threads(1)
    training=json.loads((out/'training.json').read_text());net,initial=p.actor_setup()
    bank=np.load(out/'training_contexts.npz')['policy_guard_bank']
    context=dict(np.load(phase.PREVIOUS/'contexts.npz'));expected=0;attempts=0;critic_steps=0
    actor_initial_hash=tree_sha(initial);value_error=[]
    for seed in [0,1]:
        b=training['actors'][f'B_s{seed}'];c=training['actors'][f'C_s{seed}']
        assert b['initial_actor_sha256']==c['initial_actor_sha256']==actor_initial_hash
        assert b['initial_critic_sha256']==c['initial_critic_sha256']
        for arm in ['B','C']:
            name=f'{arm}_s{seed}';info=training['actors'][name]
            for round_id,row in enumerate(info['rounds']):
                records=phase.load_records(out/'rounds'/f'{name}_r{round_id}'/'paths.npz')
                for r in records:
                    expected+=r['action'].shape[0]*r['action'].shape[1]
                    np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
                    if arm=='B':np.testing.assert_array_equal(r['action'],r['x_prime'])
                saved=torch.load(out/'checkpoints'/f'{name}_r{round_id}_critic.pt',weights_only=True)
                critic=early.Critic(0);critic.load_state_dict(saved['model'])
                assert old.parameter_sha(critic)==row['critic_sha256']
                assert all(float(s['step'])==1500+round_id*400 for s in saved['optimizer']['state'].values())
                critic_steps+=row['critic']['steps']
                r=records[0];s=r['states'][:8,0];a=r['action'][:8,0];h=np.full(len(s),r['action'].shape[1])
                torch_q=critic.predict(s,a,h,context['mean'],context['std'])
                jax_q=np.asarray(p.decoded_q(p.export_critic(critic),s,a,h,context['mean'],context['std']))
                np.testing.assert_allclose(torch_q,jax_q,rtol=1e-5,atol=1e-6)
                value_error.append(float(np.max(np.abs(torch_q-jax_q))))
                before=p.load_actor(out/'checkpoints'/f'{name}_r{round_id}_before.pkl')
                after=p.load_actor(out/'checkpoints'/f'{name}_r{round_id}_after.pkl')
                assert tree_sha(before)==row['actor_before_sha256'] and tree_sha(after)==row['actor_after_sha256']
                for step in row['steps']:
                    attempts+=1
                    assert step['accepted']==p.policy_admissible(step['bank_change'],step['current_change'],step['step_kl'])
                change=np.asarray(p.policy_change(net,initial,after,bank))
                assert p.policy_admissible(change,change,0.)
            final=p.load_actor(out/'checkpoints'/f'{name}_final.pkl')
            assert tree_sha(final)==info['final_actor_sha256']
            np.testing.assert_allclose(p.policy_change(net,initial,final,bank),info['final_bank_change'],atol=2e-6)
    assert attempts==400 and critic_steps==10800
    assert tree_sha(p.load_actor(out/'checkpoints'/'A.pkl'))==actor_initial_hash
    native_arrays={};model_arrays={}
    names=['A','B_s0','C_s0','B_s1','C_s1']
    for name in names:
        r=dict(np.load(out/f'{name}_native.npz'));r['failure']=np.load(out/f'{name}_native_audit.npz')['failure']
        np.testing.assert_array_equal(r['reset_seed'],np.arange(256)+p.CONFIG['native_reset_seed'])
        np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
        np.testing.assert_array_equal(r['reward'],np.linalg.norm(r['states'][:,1:,:2].astype(float)-p.GOAL[:2],axis=-1)<2)
        assert np.all(np.diff(r['failure'].astype(int),axis=1)>=0)
        native_arrays[name]=e.metrics(r,True)
        filename='A.pkl' if name=='A' else name+'_final.pkl'
        assert sha(out/'checkpoints'/filename)==training['final_checkpoints'][name]
    for model,(_,observational) in p.models().items():
        model_arrays[model]={}
        for name in names:
            r=dict(np.load(out/f'{model}_{name}_model.npz'));expected+=128*50
            np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
            if observational:np.testing.assert_array_equal(r['action'],r['x_prime'])
            np.testing.assert_array_equal(r['reward'],np.asarray(old.task_reward(r['states'][:,1:],np.broadcast_to(p.GOAL,r['states'][:,1:].shape))))
            model_arrays[model][name]=e.metrics(r)
    results=e.analyze(native_arrays,model_arrays)
    write(out/'audit_recomputed_results.json',results)
    assert json.loads((out/'audit_recomputed_results.json').read_text())==json.loads((out/'results.json').read_text())
    ledger=json.loads((out/'ledger.json').read_text())
    assert expected==423168==ledger['charged']==sum(r['outputs'] for r in ledger['entries'])
    assert json.loads((out/'native_ledger.json').read_text())['charged']==64000
    p.verify(out)
    write(out/'audit.json',dict(passed=True,new_simulations=0,actor_attempts=attempts,critic_steps=critic_steps,
        model_transitions=expected,native_steps=64000,observational_action_channels_verified=True,
        original_actor_unchanged=True,paired_initializations=True,policy_guards_verified=True,
        final_predictions_and_results_recomputed=True,max_jax_torch_value_error=max(value_error),
        frozen_inputs_unchanged=True))
    write(out/'manifest.json',{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'})
    print('Native-policy stored-artifact audit passed; no new simulations or updates.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out)
