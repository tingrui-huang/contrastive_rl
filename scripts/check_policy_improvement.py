"""Reproduce final predictions and audit saved policy-improvement artifacts."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import jax
import numpy as np
from crl import checkpoint
from ett.eval_policy_improvement import validate_rewards, metrics, paired
from ett.policy_improvement import make_generator, replay, tree_sha
from ett.rollout_return import GOAL, START
from ett.run_convex_adversarial import inputs, setup
from ett.run_policy_improvement import CONFIG, dataset, learner_setup, model_paths, verify
from ett.run_return_readout import read, write
from scripts.make_swamp_f4_failure_bank import file_sha


def check(root):
    verify(root)
    # Normalization lives beside the nominal weights. Verify it against the
    # committed starting-point version, not a newly chosen preprocessing file.
    nominal_config = inputs()['nominal'].with_name('config.json').as_posix()
    base_commit = read(root/'provenance.json')['commit']
    original_config = subprocess.check_output(['git', 'show', f'{base_commit}:{nominal_config}'])
    current_config = Path(nominal_config).read_bytes()
    assert current_config.replace(b'\r\n', b'\n') == original_config.replace(b'\r\n', b'\n')
    supplemental = dict(nominal_config=nominal_config, nominal_config_sha256=file_sha(nominal_config),
        committed_nominal_config_sha256=hashlib.sha256(original_config).hexdigest(),
        committed_normalization_unchanged=True,
        verification_timing='additional sidecar verification after training; no preprocessing change')
    evaluation = read(root/'evaluation.json')
    training = read(root/'training.json')['runs']
    _, net, _ = learner_setup()
    _, initial = checkpoint.load_checkpoint(inputs()['actor'])
    params = {'initial': initial.policy_params}
    for name, run in training.items():
        path = root/'checkpoints'/f'{name}_final.pkl'
        assert file_sha(path) == run['checkpoint_sha256']
        _, state = checkpoint.load_checkpoint(path)
        assert tree_sha(state) == run['final_state_sha256']
        params[name] = state.policy_params
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(state))
        assert run['frozen_verified']
        assert all(np.isfinite(row[k]) for row in run['curves'] for k in row)
    @jax.jit
    def act(p, observation):
        return net.sample_eval(net.policy_network.apply(p, observation), None)
    targets = {}
    for name, p in params.items():
        record = dict(np.load(root/f'{name}_native.npz'))
        validate_rewards(record)
        for t in range(50):
            obs = np.concatenate([record['states'][:, t], np.tile(GOAL, (128, 1))], -1)
            np.testing.assert_array_equal(record['action'][:, t], act(p, obs))
        record['failure'] = np.load(root/f'{name}_native_audit.npz')['failure']
        targets[name], summary = metrics(record)
        for key in summary:
            np.testing.assert_array_equal(summary[key], evaluation['native'][name][key])
    for seed in [0, 1]:
        for arm in ['A', 'B']:
            comparison = paired(targets[f's{seed}_C'], targets[f's{seed}_{arm}'])
            for metric, value in comparison.items():
                saved = evaluation['native_paired'][f's{seed}_C_minus_{arm}'][metric]
                np.testing.assert_array_equal(value['difference'], saved['difference'])
                np.testing.assert_array_equal(value['ci95'], saved['ci95'])
    offline_obs, _, ids, heldout = dataset()
    id_to_row = {int(e): i for i, e in enumerate(ids)}
    segments = 0
    for seed in [0, 1]:
        for arm in ['B', 'C']:
            name = f's{seed}_{arm}'
            generated = dict(np.load(root/f'{name}_generated.npz'))
            auxiliary = dict(np.load(root/f'{name}_auxiliary.npz'))
            assert set(generated) == {'states', 'action', 'episode', 'time', 'update', 'reward'}
            assert set(auxiliary) == {'x_prime', 'anchor'}
            assert generated['states'].shape == (20, 256, 4, 8)
            assert generated['action'].shape == auxiliary['x_prime'].shape == (20, 256, 3, 2)
            assert not np.isin(generated['episode'], heldout).any()
            assert np.all(generated['time'] + 3 <= 50)
            for refresh in range(20):
                ep, t = generated['episode'][refresh], generated['time'][refresh]
                row = np.array([id_to_row[int(e)] for e in ep])
                np.testing.assert_array_equal(generated['states'][refresh, :, 0], offline_obs[row, t, :8])
                states = generated['states'][refresh]
                np.testing.assert_array_equal(states[:, 1:, 2:], states[:, :-1, :6])
                buf = replay(np.concatenate([states, np.broadcast_to(GOAL, states.shape)], -1), generated['action'][refresh], 26001000)
                batch, index = buf.sample_audited(256)
                np.testing.assert_array_equal(batch.observation[:, 8:], states[index['episode'], index['future']])
                np.testing.assert_array_equal(batch.action, generated['action'][refresh, index['episode'], index['time']])
                assert np.isnan(batch.reward).all() and np.isnan(batch.discount).all()
                segments += 256
            # Verify both arms used the same starting contexts, never score-based selection.
        b = np.load(root/f's{seed}_B_generated.npz')
        c = np.load(root/f's{seed}_C_generated.npz')
        for field in ['episode', 'time', 'update']:
            np.testing.assert_array_equal(b[field], c[field])
    model, nominal, _, _, _ = setup()
    generator = make_generator(model, nominal, net, 50, deterministic=True)
    model_replays = [('initial', 0, 'control'), ('s0_C', 0, 'adversarial'),
                     ('s1_C', 1, 'adversarial'), ('s1_B', 1, 'control')]
    for actor, seed, arm in model_replays:
        theta = np.load(model_paths()[f's{seed}_{arm}'])['theta']
        record = jax.tree.map(np.asarray, generator(params[actor], theta, np.tile(START, (128, 1)),
                                      jax.random.PRNGKey(CONFIG['evaluation_model_seed'])))
        tag = f'{actor}_model_s{seed}_{arm}'
        saved = np.load(root/f'{tag}.npz')
        for field in saved.files:
            np.testing.assert_array_equal(saved[field], record[field])
        np.testing.assert_array_equal(np.load(root/f'{tag}_auxiliary.npz')['x_prime'], record['aux_x_prime'])
    verify(root)
    model_total = 61440 + 768 + evaluation['model_evaluation_steps'] + len(model_replays)*6400
    native_total = evaluation['native_steps'] + 406
    native_cap = evaluation['engineering_amendment'].get('revised_native_step_cap', CONFIG['native_step_cap'])
    assert model_total <= CONFIG['model_step_cap'] and native_total <= native_cap
    write(root/'verification.json', dict(status='passed', final_native_actions_reproduced=7*128*50,
        complete_model_replay_paths=len(model_replays)*128, verified_generated_segments=segments,
        exact_native_metric_and_paired_interval_reproduction=True,
        no_heldout_or_auxiliary_inputs=True, fixed_models_and_original_artifacts_unchanged=True,
        model_transitions_including_smoke_and_replay=model_total,
        native_transitions_including_focused_tests=native_total,
        normalization_provenance=supplemental,
        checkpoint_sha256={name: run['checkpoint_sha256'] for name, run in training.items()}))
    print('Saved independent artifact, action, reward, goal and frozen-hash verification.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    check(Path(parser.parse_args().run_dir))
