"""Independent checks and saved-data diagnostics for the future-window test."""
import argparse
from pathlib import Path
import jax
import numpy as np
from crl import checkpoint
from ett.eval_future_window import paired
from ett.eval_policy_improvement import metrics, validate_rewards
from ett.policy_improvement import make_generator, tree_sha
from ett.rollout_return import GOAL, START
from ett.run_convex_adversarial import inputs, setup
from ett.run_policy_improvement import learner_setup, model_paths, dataset
from ett.run_future_window import CONFIG, verify, frozen_signature
from ett.run_return_readout import read, write
from scripts.make_swamp_f4_failure_bank import file_sha


def goal_diagnostics(samples):
    valid = np.arange(26)[None] < samples['count'][:, None]
    i, j, h = [samples[k][valid] for k in ['time', 'future', 'horizon']]
    lag = j-i
    out = dict(count=len(j), future_index_histogram=np.bincount(j, minlength=51),
        lag_histogram=np.bincount(lag, minlength=51), anchor_histogram=np.bincount(i, minlength=3),
        available_horizon_histogram=np.bincount(h, minlength=51), mean_future_index=float(j.mean()),
        mean_lag=float(lag.mean()), fraction_beyond_three=float(np.mean(j>3)))
    for space in ['xy', 'f4']:
        distance = samples[f'goal_{space}_distance'][valid]
        out[f'{space}_distance_mean'] = float(distance.mean())
        out[f'{space}_distance_quantiles'] = np.quantile(distance, [0, .25, .5, .75, .95, 1])
        out[f'{space}_distance_below_0p5'] = float(np.mean(distance < .5))
        out[f'{space}_distance_below_2'] = float(np.mean(distance < 2))
    return out


def shared_diagnostics(shared, auxiliary):
    h = shared['horizon']
    valid = np.arange(50)[None, None] < h[..., None]
    before = shared['states'][:, :, :-1][valid]
    after = shared['states'][:, :, 1:][valid]
    np.testing.assert_array_equal(after[:, 2:], before[:, :6])
    motion = np.linalg.norm(after[:, :2]-before[:, :2], axis=-1)
    stationary = np.all(before.reshape(-1, 4, 2) == before[:, None, :2], axis=(1, 2))
    atoms = auxiliary['atom'][valid]
    moved = np.linalg.norm(after[:, :2]-auxiliary['anchor'][valid], axis=-1)>1e-5
    returns = np.nansum(shared['reward']*(.95**np.arange(50)), axis=-1)
    report = dict(valid_transitions=int(valid.sum()), paths=h.size,
        remaining_return_mean=float(returns.mean()), remaining_return_quantiles=np.quantile(returns, [0, .25, .5, .75, .95, 1]),
        remaining_return_zero_fraction=float(np.mean(returns==0)),
        stationary_history_count=int(stationary.sum()), stationary_step_fraction=float(np.mean(motion==0)),
        renewed_stationary_history_fraction=float(np.mean(motion[stationary]>1e-5)),
        sampled_atom_count=int(atoms.sum()), moved_atom_fraction=float(moved[atoms].mean()),
        projection_fraction=float(auxiliary['projection'][valid].mean()),
        boundary_fraction=float(auxiliary['boundary'][valid].mean()),
        f4_exact=True, return_by_horizon={str(k): dict(paths=int(np.sum(h==k)), mean=float(returns[h==k].mean()),
            zero_fraction=float(np.mean(returns[h==k]==0))) for k in np.unique(h)})
    return report, returns


def check(root):
    verify(root)
    training = read(root/'training.json')['runs']
    evaluation = read(root/'evaluation.json')
    generation = read(root/'generation.json')
    _, net, _ = learner_setup()
    _, initial = checkpoint.load_checkpoint(inputs()['actor'])
    params = {'initial': initial.policy_params}
    for name, run in training.items():
        path = root/'checkpoints'/f'{name}_final.pkl'
        assert file_sha(path) == run['checkpoint_sha256']
        _, state = checkpoint.load_checkpoint(path)
        assert tree_sha(state) == run['final_state_sha256']
        assert run['policy_sha256'] != tree_sha(initial.policy_params)
        assert run['critic_sha256'] != tree_sha(initial.q_params)
        params[name] = state.policy_params
    @jax.jit
    def actor(p, obs):
        return net.sample_eval(net.policy_network.apply(p, obs), None)
    native_each = {}
    behavior = {}
    reference = dict(np.load(root/'initial_native.npz'))
    for name, p in params.items():
        saved = dict(np.load(root/f'{name}_native.npz'))
        validate_rewards(saved)
        for t in range(50):
            obs = np.concatenate([saved['states'][:, t], np.tile(GOAL, (128, 1))], -1)
            np.testing.assert_array_equal(saved['action'][:, t], actor(p, obs))
        saved['failure'] = np.load(root/f'{name}_native_audit.npz')['failure']
        native_each[name], summary = metrics(saved)
        for field in summary:
            np.testing.assert_array_equal(summary[field], evaluation['native'][name][field])
        live = np.concatenate([np.ones((128, 1), bool), ~saved['failure'][:, :-1]], 1)
        action_difference = np.linalg.norm(saved['action']-reference['action'], axis=-1)
        behavior[name] = dict(mean_action_l2_vs_initial=float(action_difference.mean()),
            max_action_l2_vs_initial=float(action_difference.max()),
            action_l2_vs_initial_on_this_actor_live_steps=float(action_difference[live].mean()),
            action_live_step_denominator=int(live.sum()),
            path_xy_mean_l2_vs_initial=float(np.linalg.norm(saved['position']-reference['position'], axis=-1).mean()))
    for seed in [0, 1]:
        for arm in ['A', 'B', 'initial']:
            other = arm if arm=='initial' else f's{seed}_{arm}'
            computed = paired(native_each[f's{seed}_C'], native_each[other])
            stored = evaluation['native_paired'][f's{seed}_C_minus_{arm}']
            for metric in computed:
                for field in computed[metric]:
                    np.testing.assert_array_equal(computed[metric][field], stored[metric][field])
    obs, _, ids, heldout = dataset()
    source_row = np.full(6600, -1)
    source_row[ids] = np.arange(len(ids))
    window_results, shared_results, shared_records = {}, {}, {}
    for seed in [0, 1]:
        shared = dict(np.load(root/f's{seed}_shared.npz'))
        aux = dict(np.load(root/f's{seed}_shared_auxiliary.npz'))
        assert file_sha(root/f's{seed}_shared_auxiliary.npz') == generation['seeds'][str(seed)]['auxiliary_sha256']
        assert not np.isin(shared['episode'], heldout).any()
        np.testing.assert_array_equal(shared['states'][:, :, 0], obs[source_row[shared['episode']], shared['time'], :8])
        np.testing.assert_array_equal(shared['horizon'], 50-shared['time'])
        valid = np.arange(51)[None, None] <= shared['horizon'][..., None]
        assert np.isfinite(shared['states'][valid]).all()
        assert np.isnan(shared['states'][~valid]).all()
        shared_results[str(seed)], returns = shared_diagnostics(shared, aux)
        np.savez_compressed(root/f's{seed}_remaining_returns.npz', returns=returns, horizon=shared['horizon'])
        shared_records[seed] = (shared, aux)
        audits = {}
        for arm in ['B', 'C']:
            sample = dict(np.load(root/f's{seed}_{arm}_sample_audit.npz'))
            audits[arm] = sample
            mask = np.arange(26)[None] < sample['count'][:, None]
            block = np.broadcast_to(np.arange(2000)[:, None]//100, mask.shape)[mask]
            e, i, j, h = [sample[k][mask] for k in ['episode', 'time', 'future', 'horizon']]
            np.testing.assert_array_equal(h, shared['horizon'][block, e])
            assert np.all((i>=0)&(i<3)&(j>i)&(j<=h))
            if arm=='B':
                assert np.all(j<=3)
            goal = shared['states'][block, e, j]
            assert np.isfinite(goal).all()
            np.testing.assert_allclose(np.linalg.norm(goal[:, :2]-GOAL[:2], axis=-1),
                sample['goal_xy_distance'][mask], atol=1e-6, rtol=0)
            result = goal_diagnostics(sample)
            # Expected conditional lag distribution for these exact anchor/windows.
            n = (h if arm=='C' else 3)-i
            expected = np.zeros(51)
            for lag in range(1, 51):
                expected[lag] = np.mean(np.where(lag<=n, .95**(lag-1)*.05/(1-.95**n), 0))
            result['expected_lag_probability'] = expected
            result['max_abs_lag_frequency_error'] = float(np.max(np.abs(result['lag_histogram']/len(j)-expected)))
            window_results[f's{seed}_{arm}'] = result
        for field in ['count', 'permutation', 'source', 'episode', 'time', 'horizon']:
            np.testing.assert_array_equal(audits['B'][field], audits['C'][field])
        assert training[f's{seed}_B']['common_batch_sha256'] == training[f's{seed}_C']['common_batch_sha256']
    # Re-query only predeclared generation groups and four full model sets.
    replay_steps = 4*128*50 + sum(int(np.sum(shared_records[s][0]['horizon']==h))*h
                                for s in [0, 1] for h in [3, 26, 50])
    assert replay_steps <= 80000
    assert generation['primary_model_transitions']+evaluation['model_steps']+replay_steps <= CONFIG['total_model_cap']
    model, nominal, _, _, _ = setup()
    frozen = frozen_signature(model, nominal, initial)
    for seed in [0, 1]:
        theta = np.load(model_paths()[f's{seed}_adversarial'])['theta']
        shared, aux = shared_records[seed]
        flat_states = shared['states'].reshape(5120, 51, 8)
        flat_h = shared['horizon'].reshape(-1)
        for h in [3, 26, 50]:
            rows = np.flatnonzero(flat_h==h)
            generate = make_generator(model, nominal, net, h)
            value = jax.tree.map(np.asarray, generate(initial.policy_params, theta, flat_states[rows, 0],
                jax.random.PRNGKey(CONFIG['model_generation_base']+seed*1000+h)))
            for field, array in value.items():
                source = shared if field in shared else aux
                saved = source[field].reshape((5120,)+source[field].shape[2:])
                np.testing.assert_array_equal(array, saved[rows, :h+1 if field=='states' else h])
        generate = make_generator(model, nominal, net, 50, deterministic=True)
        for name in ['initial', f's{seed}_C']:
            value = jax.tree.map(np.asarray, generate(params[name], theta, np.tile(START, (128, 1)),
                jax.random.PRNGKey(CONFIG['model_evaluation_seed'])))
            saved = np.load(root/f'{name}_model_s{seed}.npz')
            for field in saved.files:
                np.testing.assert_array_equal(value[field], saved[field])
            np.testing.assert_array_equal(value['aux_x_prime'], np.load(root/f'{name}_model_s{seed}_auxiliary.npz')['x_prime'])
    assert frozen_signature(model, nominal, initial) == frozen
    verify(root)
    native_total = evaluation['native_steps']+200
    assert native_total <= CONFIG['native_step_cap']
    write(root/'diagnostics.json', dict(goals=window_results, shared_continuations=shared_results,
        native_action_changes=behavior,
        live_action_diagnostic='Uses this actor prior-step failure only for evaluation; compared actions may be at different visible states'))
    write(root/'verification.json', dict(status='passed', exact_native_action_replays=44800,
        all_saved_anchor_and_window_checks=True, unchanged_frozen_signature=frozen,
        paired_metrics_reproduced=True, padded_tails_never_sampled=True,
        actor_and_critic_updated=True, replay_model_steps=replay_steps,
        total_model_steps=generation['primary_model_transitions']+evaluation['model_steps']+replay_steps,
        total_native_steps=native_total))
    print('Verified shared windows, native results, selected model replays and all frozen inputs.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    check(Path(parser.parse_args().run_dir))
