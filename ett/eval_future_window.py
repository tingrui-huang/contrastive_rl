"""Paired final-policy evaluation for the shared future-window experiment."""
import argparse
from pathlib import Path
import jax
import numpy as np
from crl import checkpoint
from ett.eval_policy_improvement import native, metrics, validate_rewards
from ett.policy_improvement import make_generator, motion_diagnostics, tree_sha
from ett.rollout_return import START
from ett.run_convex_adversarial import inputs, setup
from ett.run_policy_improvement import learner_setup, model_paths
from ett.run_future_window import CONFIG, verify, frozen_signature
from ett.run_return_readout import read, write
from scripts.make_swamp_f4_failure_bank import file_sha


def paired(a, b):
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    index = rng.integers(len(a['returns']), size=(2000, len(a['returns'])))
    return {key: dict(difference=float(np.mean(a[key]-b[key])),
                     ci95=np.quantile((a[key]-b[key])[index].mean(1), [.025, .975]))
            for key in a if key in b}


def evaluate(root):
    verify(root)
    if (root/'evaluation.json').exists():
        raise ValueError('evaluation already completed')
    training = read(root/'training.json')['runs']
    _, net, _ = learner_setup()
    _, initial = checkpoint.load_checkpoint(inputs()['actor'])
    params = {'initial': initial.policy_params}
    for seed in [0, 1]:
        assert len({training[f's{seed}_{a}']['initial_state_sha256'] for a in ['A', 'B', 'C']}) == 1
        assert training[f's{seed}_B']['common_batch_sha256'] == training[f's{seed}_C']['common_batch_sha256']
        assert training[f's{seed}_B']['goal_batch_sha256'] != training[f's{seed}_C']['goal_batch_sha256']
    for name, run in training.items():
        assert run['updates'] == 2000 and run['synthetic_anchors'] == (0 if name.endswith('A') else 51200)
        path = root/'checkpoints'/f'{name}_final.pkl'
        assert file_sha(path) == run['checkpoint_sha256']
        step, state = checkpoint.load_checkpoint(path)
        assert step == 152000 and tree_sha(state) == run['final_state_sha256']
        params[name] = state.policy_params
    @jax.jit
    def actor(p, obs):
        return net.sample_eval(net.policy_network.apply(p, obs), None)
    native_result, native_each, native_checks = {}, {}, {}
    reset = CONFIG['evaluation_reset_base']+np.arange(128)
    for name, p in params.items():
        record = native(p, actor, reset)
        np.savez_compressed(root/f'{name}_native.npz', **{k: v for k, v in record.items() if k != 'failure'})
        np.savez_compressed(root/f'{name}_native_audit.npz', failure=record['failure'])
        replay = native(p, actor, reset[:4])
        for field in record:
            np.testing.assert_array_equal(record[field][:4], replay[field])
        native_checks[name] = True
        native_each[name], native_result[name] = metrics(record)
        print('native', name, 'return', native_result[name]['returns'],
              'success', native_result[name]['success'], 'failure', native_result[name]['failure'], flush=True)
    comparisons = {}
    for seed in [0, 1]:
        for arm in ['A', 'B', 'initial']:
            other = arm if arm == 'initial' else f's{seed}_{arm}'
            comparisons[f's{seed}_C_minus_{arm}'] = paired(native_each[f's{seed}_C'], native_each[other])
    model, nominal, _, _, _ = setup()
    frozen = frozen_signature(model, nominal, initial)
    generator = make_generator(model, nominal, net, 50, deterministic=True)
    model_result, model_comparison = {}, {}
    for seed in [0, 1]:
        theta = np.load(model_paths()[f's{seed}_adversarial'])['theta']
        each = {}
        for name in ['initial']+[f's{seed}_{a}' for a in ['A', 'B', 'C']]:
            value = jax.tree.map(np.asarray, generator(params[name], theta, np.tile(START, (128, 1)),
                                                       jax.random.PRNGKey(CONFIG['model_evaluation_seed'])))
            validate_rewards(value)
            each[name], summary = metrics(value)
            tag = f'{name}_model_s{seed}'
            model_result[tag] = dict(**summary, diagnostics=motion_diagnostics(value))
            np.savez_compressed(root/f'{tag}.npz', **{k: v for k, v in value.items() if k != 'aux_x_prime'})
            np.savez_compressed(root/f'{tag}_auxiliary.npz', x_prime=value['aux_x_prime'])
            print('model', tag, summary['returns'], flush=True)
        for arm in ['A', 'B', 'initial']:
            other = arm if arm == 'initial' else f's{seed}_{arm}'
            model_comparison[f's{seed}_C_minus_{arm}'] = paired(each[f's{seed}_C'], each[other])
    assert frozen_signature(model, nominal, initial) == frozen
    generation = read(root/'generation.json')['primary_model_transitions']
    assert generation+51200 <= CONFIG['total_model_cap'] and 46200 <= CONFIG['native_step_cap']
    verify(root)
    write(root/'evaluation.json', dict(native=native_result, native_paired=comparisons,
        model=model_result, model_paired=model_comparison, native_steps=46200, model_steps=51200,
        native_selected_replay_exact=native_checks, frozen_parameters_unchanged=True,
        uncertainty='2000 paired episode bootstrap resamples; conditional on two coupled model/learner seeds; unadjusted 95% percentile intervals'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    evaluate(Path(parser.parse_args().run_dir))
