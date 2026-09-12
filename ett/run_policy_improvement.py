"""Bounded A/B/C continuation of CRL with immutable ETT samplers."""
import argparse
import dataclasses
from pathlib import Path
import shutil
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from crl import checkpoint, losses, networks
from ett.policy_improvement import (make_generator, mixed_batch, motion_diagnostics,
                                    replay, tree_sha)
from ett.rollout_return import GOAL
from ett.run_convex_adversarial import inputs, setup
from ett.run_return_readout import read, write
from scripts.make_swamp_f4_failure_bank import file_sha
from scripts import run_swamp_windy_z_failneg as launcher

MODEL_ROOT = Path('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1')
SPEC = Path('notes/ett_policy_improvement_spec.md')
CONFIG = dict(seeds=[0, 1], arms=['A', 'B', 'C'], updates=2000, refresh=100,
    contexts=256, rollout_horizon=3, synthetic_fraction=.1, replay_refreshes=1,
    batch_size=256, discount=.95, evaluation_episodes=128, evaluation_horizon=50,
    reset_seed_base=21000000, evaluation_model_seed=22000000,
    train_seed_base=23000000, rollout_seed_base=24000000, bootstrap_seed=25000000,
    native_step_cap=47000, model_step_cap=220000,
    failure_negative_alpha=0., evaluation_action='deterministic tanh(loc)',
    success='minimum native XY distance < 0.5', reward='native reward; XY radius < 2',
    original_partition='all behavior sources; exclude permutation(6600, seed=0)[:660]',
    checkpoint_selection='final update only', unused_transition_fields='NaN; MC losses must not read them')


def learner_setup():
    launcher.select_version('f4')
    cfg = launcher.build_cfg('zbase', '', steps=2000, seed=0)
    cfg.obs_dim = cfg.goal_dim = 8
    cfg.action_dim = 2
    cfg.max_episode_steps = 50
    assert not cfg.use_td and not cfg.use_cpc and not cfg.use_gcbc
    assert cfg.fail_neg_alpha == 0 and not cfg.fail_bank_path
    assert cfg.batch_size == 256 and cfg.discount == .95 and cfg.bc_coef == .5
    net = networks.make_networks(8, 8, 2)
    _, update = losses.build_learner(net, cfg, lambda x: x,
        optax.adam(cfg.actor_learning_rate, eps=1e-7), optax.adam(cfg.learning_rate, eps=1e-7))
    return cfg, net, jax.jit(update)


def dataset():
    with np.load(inputs()['dataset'], allow_pickle=False) as z:
        # Training has no other array access, including evaluation or hidden labels.
        observation, action = z['obs'], z['act']
    heldout = np.random.default_rng(0).permutation(len(observation))[:660]
    ids = np.setdiff1d(np.arange(len(observation)), heldout)
    assert len(ids) == 5940 and not np.intersect1d(ids, heldout).size
    assert np.all(observation[:, :, 8:] == GOAL)
    return observation[ids], action[ids, :50], ids, heldout


def model_paths():
    return {f's{s}_{arm}': MODEL_ROOT/'checkpoints'/f's{s}_{arm}.npz'
            for s in [0, 1] for arm in ['control', 'adversarial']}


def verify(root):
    assert read(root/'config.json') == CONFIG
    p = read(root/'provenance.json')
    amendment = read(root/'evaluation_amendment.json') if (root/'evaluation_amendment.json').exists() else {}
    changes = amendment.get('source_changes', {})
    assert set(changes) <= {'ett/eval_policy_improvement.py', 'ett/run_policy_improvement.py'}
    for path, digest in p['sha256'].items():
        if path in changes:
            change = changes[path]
            assert change['original_sha256'] == digest
            assert file_sha(root/'pre_fix_sources'/Path(path).name) == digest
            digest = change['corrected_sha256']
        assert file_sha(path) == digest, path
    for filename, digest in read(root/'prepared.json')['sha256'].items():
        assert file_sha(root/filename) == digest, filename


def prepare(root):
    if root.exists():
        raise ValueError('use a fresh output directory')
    paths = inputs()
    prior = read(MODEL_ROOT/'training.json')
    for name, path in model_paths().items():
        assert file_sha(path) == prior['arms'][name]['checkpoint_sha256'], path
        with np.load(path) as data:
            assert float(data['bound']) == 1 and data['theta'].shape == (32,)
            if 'control' in name:
                assert np.all(data['theta'] == 0)
    cfg, net, _ = learner_setup()
    step, state = checkpoint.load_checkpoint(paths['actor'])
    assert step == 150000
    provenance = read(paths['actor'].with_name('arm_provenance.json'))
    assert provenance['alpha'] == 0 and provenance['seed'] == 0
    obs, act, ids, heldout = dataset()
    # Closed-over versus dynamic JIT weights can differ by float32 roundoff.
    model, nominal, old_actor, _, _ = setup()
    key = jax.random.PRNGKey(26000000)
    probe = obs[:16, 0]
    sample = jax.jit(lambda p, observation, k: net.sample(net.policy_network.apply(p, observation), k))
    old_actions = np.asarray(old_actor(probe[:, :8], probe[:, 8:], key))
    new_actions = np.asarray(sample(state.policy_params, probe, key))
    np.testing.assert_allclose(old_actions, new_actions, atol=2e-6, rtol=0)
    root.mkdir(parents=True)
    (root/'checkpoints').mkdir()
    shutil.copyfile(SPEC, root/'SPEC.md')
    write(root/'config.json', CONFIG)
    write(root/'learner_config.json', dataclasses.asdict(cfg))
    np.savez_compressed(root/'partition.npz', train=ids, heldout=heldout)
    source = ['ett/policy_improvement.py', 'ett/run_policy_improvement.py',
              'ett/eval_policy_improvement.py', 'scripts/test_policy_improvement.py',
              'crl/losses.py', 'crl/networks.py', 'crl/replay.py', 'crl/envs.py',
              'ett/fixed_actor_continuation.py', 'ett/convex_action_transition.py',
              'scripts/run_swamp_windy_z_failneg.py', 'scripts/eval_swamp_windy_z_deployment.py', str(SPEC)]
    reports = [MODEL_ROOT/'REPORT.md', MODEL_ROOT/'verification.json', MODEL_ROOT/'provenance.json',
               Path('artifacts/fixed_actor_continuation/f4_h20_g095_v1/SUMMARY.md'),
               Path('artifacts/return_readout/f4_h20_g095_v1/REPORT.md')]
    hashes = {str(p).replace('\\', '/'): file_sha(p) for p in list(paths.values()) +
              list(model_paths().values()) + list(map(Path, source)) + reports +
              [paths['actor'].with_name('arm_provenance.json')]}
    write(root/'provenance.json', dict(sha256=hashes,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        checkpoint_step=step, initial_state_tree_sha256=tree_sha(state),
        original_actor_probe_max_abs_difference=float(np.max(np.abs(old_actions-new_actions))),
        initial_policy_sha256=tree_sha(state.policy_params), initial_critic_sha256=tree_sha(state.q_params),
        optimizer='restore complete original Adam state; replace only learner RNG per seed',
        frozen_tree_sha256=tree_sha((model.diagonal.params, nominal.params)),
        training_arrays=['obs', 'act'], training_episodes=len(ids), heldout_episodes=len(heldout),
        upstream_overlap='initial actor/critic saw all 6600 original episodes; continuation excludes 660 held-out episodes',
        device=[str(x) for x in jax.devices()]))
    write(root/'prepared.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        sha256={p: file_sha(root/p) for p in ['config.json', 'SPEC.md', 'learner_config.json', 'partition.npz', 'provenance.json']}))
    print('Prepared protocol and verified all checkpoint hashes before training.', flush=True)


def train(root, smoke=False):
    verify(root)
    suffix = 'smoke' if smoke else 'training'
    if (root/f'{suffix}.json').exists():
        raise ValueError('phase already completed')
    cfg, net, update = learner_setup()
    _, initial = checkpoint.load_checkpoint(inputs()['actor'])
    model, nominal, _, _, _ = setup()
    frozen = tree_sha((model.diagonal.params, nominal.params))
    generate = make_generator(model, nominal, net, 3)
    obs, act, ids, heldout = dataset()
    updates = 5 if smoke else CONFIG['updates']
    arms = ['C'] if smoke else CONFIG['arms']
    seeds = [0] if smoke else CONFIG['seeds']
    records = {}
    for seed in seeds:
        for arm in arms:
            name = f's{seed}_{arm}'
            final_path = root/'checkpoints'/f'{name}_final.pkl'
            if not smoke and final_path.exists():
                raise ValueError(f'refusing to overwrite {final_path}')
            state = initial._replace(key=jax.random.PRNGKey(CONFIG['train_seed_base'] + seed))
            start_sha = tree_sha(state)
            offline = replay(obs, act, CONFIG['train_seed_base'] + seed + 10)
            rng = np.random.default_rng(CONFIG['rollout_seed_base'] + seed)
            permutation = np.random.default_rng(CONFIG['train_seed_base'] + seed + 20)
            synthetic = None
            theta = np.load(model_paths()[f's{seed}_'+('adversarial' if arm == 'C' else 'control')])['theta']
            theta_sha = tree_sha(theta)
            curves, diagnostics, generation, auxiliary = [], [], [], []
            n_synthetic = 0
            start = time.monotonic()
            chunk = []
            for u in range(updates):
                if arm != 'A' and u % CONFIG['refresh'] == 0:
                    e = rng.integers(len(obs), size=256)
                    t = rng.integers(0, 48, size=256)
                    initial_context = obs[e, t, :8]
                    key = jax.random.PRNGKey(CONFIG['rollout_seed_base'] + seed * 10000 + u)
                    record = jax.tree.map(np.asarray, generate(state.policy_params, theta, initial_context, key))
                    diag = motion_diagnostics(record)
                    commanded = np.broadcast_to(GOAL, record['states'].shape)
                    synthetic = replay(np.concatenate([record['states'], commanded], -1),
                                       record['action'], CONFIG['train_seed_base'] + seed + 100 + u)
                    assert len(synthetic) == 768 and synthetic._L == 4
                    diagnostics.append(dict(update=u, **diag, policy_sha256=tree_sha(state.policy_params)))
                    generation.append(dict(states=record['states'], action=record['action'],
                        episode=ids[e], time=t, update=np.array(u), reward=record['reward']))
                    auxiliary.append(dict(x_prime=record['aux_x_prime'], anchor=record['anchor']))
                batch, audit = mixed_batch(offline, synthetic, u, permutation)
                n_synthetic += audit['synthetic_count']
                state, metrics = update(state, jax.tree.map(jnp.asarray, batch))
                metrics = {k: float(v) for k, v in metrics.items()}
                assert all(np.isfinite(v) for v in metrics.values()), (name, u, metrics)
                chunk.append(metrics)
                if (u + 1) % 100 == 0 or u + 1 == updates:
                    curves.append(dict(update=u + 1, **{k: float(np.mean([m[k] for m in chunk])) for k in chunk[0]}))
                    chunk = []
                    assert all(np.isfinite(x).all() for x in jax.tree.leaves(state))
                    print(name, u + 1, 'critic', metrics['critic_loss'], 'actor', metrics['actor_loss'],
                          'seconds', round(time.monotonic() - start, 1), flush=True)
            assert tree_sha((model.diagonal.params, nominal.params)) == frozen
            assert tree_sha(theta) == theta_sha
            assert tree_sha(state.policy_params) != tree_sha(initial.policy_params)
            assert tree_sha(state.q_params) != tree_sha(initial.q_params)
            if not smoke:
                checkpoint.save_named(str(root/'checkpoints'), name+'_final', 150000 + updates, state)
                if generation:
                    np.savez_compressed(root/f'{name}_generated.npz', **{k: np.stack([r[k] for r in generation]) for k in generation[0]})
                    np.savez_compressed(root/f'{name}_auxiliary.npz', **{k: np.stack([r[k] for r in auxiliary]) for k in auxiliary[0]})
            entry = dict(updates=updates, start_state_sha256=start_sha,
                final_state_sha256=tree_sha(state), final_policy_sha256=tree_sha(state.policy_params),
                final_critic_sha256=tree_sha(state.q_params), synthetic_anchors=n_synthetic,
                total_anchors=updates*256, refreshes=len(diagnostics), curves=curves, diagnostics=diagnostics,
                seconds=time.monotonic()-start, frozen_verified=True)
            if not smoke:
                entry['checkpoint_sha256'] = file_sha(final_path)
            records[name] = entry
            write(root/f'{suffix}_progress.json', records)
    verify(root)
    write(root/f'{suffix}.json', dict(status='complete', runs=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'train'], required=True)
    args = parser.parse_args()
    root = Path(args.out_dir)
    if args.phase == 'prepare':
        prepare(root)
    else:
        train(root, smoke=args.phase == 'smoke')


if __name__ == '__main__':
    main()
