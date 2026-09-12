"""Independent native deployment and frozen-model evaluation; never training."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from crl import checkpoint
from ett.fixed_actor_continuation import TimedSimulator
from ett.policy_improvement import make_generator, motion_diagnostics, tree_sha
from ett.rollout_return import GOAL, START
from ett.run_convex_adversarial import inputs, setup
from ett.run_policy_improvement import CONFIG, learner_setup, model_paths, verify
from ett.run_return_readout import read, write
from scripts.make_swamp_f4_failure_bank import file_sha


def native(params, actor, reset_seeds):
    sims = [TimedSimulator(int(seed)) for seed in reset_seeds]
    states = [np.stack([sim.observation()[:8] for sim in sims])]
    actions, rewards, positions, failures = [], [], [], []
    positions.append(np.stack([sim.env.state.copy() for sim in sims]))
    for _ in range(50):
        action = np.asarray(actor(params, np.concatenate([states[-1], np.tile(GOAL, (len(sims), 1))], -1)))
        output = [sim.step(a) for sim, a in zip(sims, action)]
        states.append(np.stack([o[:8] for o, _, _ in output]))
        actions.append(action)
        rewards.append([r for _, r, _ in output])
        positions.append(np.stack([sim.env.state.copy() for sim in sims]))
        # Privileged information enters evaluation diagnostics only.
        failures.append([sim.env.dead for sim in sims])
    assert all(sim.elapsed == 50 for sim in sims)
    for sim in sims:
        try:
            sim.step(np.zeros(2, np.float32))
        except ValueError:
            pass
        else:
            raise AssertionError('time limit not enforced')
    record = dict(states=np.stack(states, 1), action=np.stack(actions, 1),
                  reward=np.array(rewards, float).T, position=np.stack(positions, 1),
                  failure=np.array(failures, bool).T, reset_seed=np.asarray(reset_seeds))
    validate_rewards(record)
    return record


def validate_rewards(record):
    before, after = record['states'][:, :-1], record['states'][:, 1:]
    np.testing.assert_array_equal(after[..., 2:], before[..., :6])
    reward = np.linalg.norm(after[..., :2].astype(float) - GOAL[:2], axis=-1) < 2
    np.testing.assert_array_equal(record['reward'], reward)
    assert np.isfinite(record['action']).all() and np.max(np.abs(record['action'])) <= 1


def metrics(record):
    reward = record['reward']
    returns = reward @ (.95 ** np.arange(50))
    xy = record.get('position', record['states'][..., :2])
    distance = np.linalg.norm(xy - GOAL[:2], axis=-1)
    success = np.min(distance, axis=1) < .5
    delta = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    per_episode = dict(returns=returns, success=success.astype(float),
        reward_hit=(reward.max(1) > 0).astype(float),
        safe_route=np.any(xy[..., 1] < 2, axis=1).astype(float),
        mean_motion=delta.mean(1), final_distance=distance[:, -1],
        action_saturation=(np.abs(record['action']) > .99).mean((1, 2)))
    if 'failure' in record:
        per_episode['failure'] = record['failure'][:, -1].astype(float)
    summary = {k: float(v.mean()) for k, v in per_episode.items()}
    summary.update(return_quantiles=np.quantile(returns, [0, .25, .5, .75, 1]),
                   zero_fraction=float(np.mean(returns == 0)))
    return per_episode, summary


def paired(a, b):
    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
    index = rng.integers(len(a['returns']), size=(2000, len(a['returns'])))
    return {k: dict(difference=float(np.mean(a[k] - b[k])),
                   ci95=np.quantile((a[k] - b[k])[index].mean(1), [.025, .975]))
            for k in a if k in b}


def evaluate(root):
    verify(root)
    if (root/'evaluation.json').exists():
        raise ValueError('evaluation already completed')
    training = read(root/'training.json')['runs']
    assert len(training) == 6
    for seed in [0, 1]:
        assert len({training[f's{seed}_{arm}']['start_state_sha256'] for arm in ['A', 'B', 'C']}) == 1
        for arm in ['A', 'B', 'C']:
            run = training[f's{seed}_{arm}']
            assert run['updates'] == 2000 and run['total_anchors'] == 512000
            assert run['synthetic_anchors'] == (0 if arm == 'A' else 51200)
            assert run['refreshes'] == (0 if arm == 'A' else 20)
    _, net, _ = learner_setup()
    _, initial = checkpoint.load_checkpoint(inputs()['actor'])
    params = {'initial': initial.policy_params}
    for name, run in training.items():
        path = root/'checkpoints'/f'{name}_final.pkl'
        assert file_sha(path) == run['checkpoint_sha256']
        step, state = checkpoint.load_checkpoint(path)
        assert step == 152000 and tree_sha(state) == run['final_state_sha256']
        params[name] = state.policy_params
    @jax.jit
    def actor(p, obs):
        return net.sample_eval(net.policy_network.apply(p, obs), None)
    reset_seeds = CONFIG['reset_seed_base'] + np.arange(128)
    summaries, per_episode, replay_checks = {}, {}, {}
    native_steps = 0
    for name, p in params.items():
        record = native(p, actor, reset_seeds)
        native_steps += 128 * 50
        # Replay the first four predeclared trajectories, no outcome selection.
        again = native(p, actor, reset_seeds[:4])
        native_steps += 4 * 50
        for field in record:
            np.testing.assert_array_equal(record[field][:4], again[field])
        replay_checks[name] = True
        per_episode[name], summaries[name] = metrics(record)
        np.savez_compressed(root/f'{name}_native.npz', **{k: v for k, v in record.items() if k != 'failure'})
        np.savez_compressed(root/f'{name}_native_audit.npz', failure=record['failure'])
        print('native', name, summaries[name], flush=True)
    comparisons = {f's{s}_C_minus_{arm}': paired(per_episode[f's{s}_C'], per_episode[f's{s}_{arm}'])
                   for s in [0, 1] for arm in ['A', 'B']}
    for s in [0, 1]:
        comparisons[f's{s}_C_minus_initial'] = paired(per_episode[f's{s}_C'], per_episode['initial'])
    model, nominal, _, _, _ = setup()
    frozen = tree_sha((model.diagonal.params, nominal.params))
    generate = make_generator(model, nominal, net, 50, deterministic=True)
    model_summaries, model_comparisons = {}, {}
    model_steps = 0
    for seed in [0, 1]:
        for model_arm in ['control', 'adversarial']:
            theta = np.load(model_paths()[f's{seed}_{model_arm}'])['theta']
            entries = {}
            for name in ['initial'] + [f's{seed}_{a}' for a in ['A', 'B', 'C']]:
                key = jax.random.PRNGKey(CONFIG['evaluation_model_seed'])
                record = jax.tree.map(np.asarray, generate(params[name], theta, np.tile(START, (128, 1)), key))
                model_steps += 128 * 50
                validate_rewards(record)
                each, summary = metrics(record)
                tag = f'{name}_model_s{seed}_{model_arm}'
                model_summaries[tag] = dict(**summary, diagnostics=motion_diagnostics(record))
                entries[name] = each
                # Auxiliary actions never share the executed-action key.
                np.savez_compressed(root/f'{tag}.npz', **{k: v for k, v in record.items() if k != 'aux_x_prime'})
                np.savez_compressed(root/f'{tag}_auxiliary.npz', x_prime=record['aux_x_prime'])
                print('model', tag, summary['returns'], flush=True)
            for arm in ['A', 'B', 'initial']:
                other = arm if arm == 'initial' else f's{seed}_{arm}'
                model_comparisons[f's{seed}_{model_arm}_C_minus_{arm}'] = paired(entries[f's{seed}_C'], entries[other])
    assert tree_sha((model.diagonal.params, nominal.params)) == frozen
    assert native_steps <= CONFIG['native_step_cap']
    assert model_steps + 4 * 20 * 256 * 3 + 768 <= CONFIG['model_step_cap']
    verify(root)
    write(root/'evaluation.json', dict(native=summaries, native_paired=comparisons,
        model=model_summaries, model_paired=model_comparisons,
        native_steps=native_steps, model_evaluation_steps=model_steps,
        native_replay_exact=replay_checks, frozen_verified=True,
        independence='128 independent reset seeds; trajectories are bootstrap units; paired across actors',
        limitations='intervals conditional on two learner/model seeds; no multiple comparison adjustment'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    evaluate(Path(parser.parse_args().run_dir))


if __name__ == '__main__':
    main()
