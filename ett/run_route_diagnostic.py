"""Bounded paired native/frozen-model route diagnostic. Never trains anything."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax
import numpy as np

from ett.anchored_transition import load_anchored
from ett.convex_action_transition import ConvexActionTransition
from ett.diagonal_transition import POINTMAZE_WALLS
from ett.fixed_actor_continuation import TimedSimulator
from ett.route_diagnostic import ROUTES, controller, make_rollout, validate
from ett.rollout_return import GOAL, START
from propensity.nominal_policy import load_nominal_policy

PRIOR = Path('artifacts/ett_future_window/shared_h3_full_u2000_s01_v1')
MODEL_ROOT = Path('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1')
CONFIG = dict(reset_seeds=list(range(51000000, 51000128)),
              representative_seeds=list(range(51000000, 51000004)),
              smoke_seeds=[51001000, 51001001], model_rng_base=52000000,
              bootstrap_seed=53000000, bootstrap_replicates=10000,
              horizon=50, discount=.95, native_swamp_probability=.3,
              goal=GOAL.tolist(), waypoint_tolerance=.25,
              routes={k: v.tolist() for k, v in ROUTES.items()},
              main_transitions=38400, smoke_transitions=600,
              replay_transitions=1200, planned_total=40200, strict_cap=45000,
              route_adherence='ordered x cells 2..6 in the designated y band; remain in band during crossing',
              controller='advance if distance <= .25; clip(target - current visible XY, -1, 1); retain final target',
              success='minimum XY distance < .5; native uses actual simulator XY',
              task_reward='native task reward / model successor XY radius < 2',
              native_coupling='same reset seed and default RNG; aligned while both live; draw consumption diverges after death',
              model_coupling='fold_in(PRNGKey(52000000), reset_seed); same keys across controllers and checkpoints; independent of native RNG')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, indent=2, default=default) + '\n', encoding='utf-8')


def sha(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def signature(model, nominal, coefficients):
    digest = hashlib.sha256()
    d = model.diagonal
    for leaf in jax.tree.leaves((d.params, d.context_mean, d.context_std,
                                d.delta_mean, d.delta_std, nominal.params,
                                nominal.context_mean, nominal.context_std,
                                coefficients, np.asarray(model.bound))):
        array = np.ascontiguousarray(leaf)
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_paths():
    old = read('artifacts/ett_rollout_return/residual6_s01/config.json')['paths']
    return dict(nominal=old['nominal'], diagonal=old['control'],
                nominal_config=str(Path(old['nominal']).with_name('config.json')).replace('\\', '/'),
                **{f's{s}': (MODEL_ROOT/'checkpoints'/f's{s}_adversarial.npz').as_posix() for s in [0, 1]})


def verify_hashes(root):
    for p, expected in read(root/'provenance.json')['sha256'].items():
        if not Path(p).is_file():
            raise FileNotFoundError(f'Required frozen artifact/source unavailable: {p}')
        assert sha(p) == expected, f'Changed artifact/source: {p}'
    assert read(root/'config.json') == CONFIG


def prepare(root):
    if root.exists():
        raise FileExistsError('Use a new artifact directory; no overwrite or rerun.')
    prior = read(PRIOR/'provenance.json')['sha256']
    # Verify the entire preceding manifest, including archived actor/data bytes;
    # only the nominal and diagonal checkpoints below are deserialized.
    for p, expected in prior.items():
        if not Path(p).is_file():
            raise FileNotFoundError(f'Required prior artifact unavailable: {p}')
        assert sha(p) == expected, p
    paths = frozen_paths()
    for p in paths.values():
        assert sha(p) == prior[p], p
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    assert branch == 'feature/pointmaze-causal-transition'
    # Search every available NPZ's seed-named numeric arrays, never outcomes.
    candidate = np.array(CONFIG['reset_seeds'] + CONFIG['smoke_seeds'])
    checked, arrays = 0, []
    for path in Path('artifacts').rglob('*.npz'):
        with np.load(path, allow_pickle=False) as z:
            checked += 1
            for name in z.files:
                if 'seed' not in name.lower():
                    continue
                try:
                    value = z[name]
                except ValueError:
                    continue  # object metadata is not a numeric reset-seed list
                if np.issubdtype(value.dtype, np.number):
                    assert not np.isin(candidate, value).any(), (path, name)
                    arrays.append(dict(path=path.as_posix(), field=name,
                                       count=value.size, sha256=sha(path)))
    # Verify the complete suggested polyline lies in open geometry, without steps.
    for points in ROUTES.values():
        polyline = np.concatenate([START[None, :2], points])
        for a, b in zip(polyline[:-1], polyline[1:]):
            cells = np.floor(a + np.linspace(0, 1, 1001)[:, None] * (b-a)).astype(int)
            assert (POINTMAZE_WALLS[cells[:, 0], cells[:, 1]] == 0).all()
    root.mkdir(parents=True)
    sources = ['ett/route_diagnostic.py', 'ett/run_route_diagnostic.py',
               'scripts/test_route_diagnostic.py', 'ett/anchored_transition.py',
               'ett/rollout_return.py']
    provenance = dict(starting_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        reviewed_commit='a23991f2b81708d1df36a00e71846d3fc640da4e', branch=branch,
        commits_since_review=subprocess.check_output(['git', 'log', '--oneline',
            'a23991f2b81708d1df36a00e71846d3fc640da4e..HEAD'], text=True),
        tracked_diff_at_prepare=subprocess.check_output(['git', 'diff', '--stat'], text=True),
        status_at_prepare=subprocess.check_output(['git', 'status', '--short'], text=True),
        sha256={**prior, **{p: sha(p) for p in sources},
                (PRIOR/'REPORT.md').as_posix(): sha(PRIOR/'REPORT.md'),
                (PRIOR/'provenance.json').as_posix(): sha(PRIOR/'provenance.json')},
        loaded_paths=paths, python=sys.version, executable=sys.executable,
        numpy=np.__version__, jax=jax.__version__, platform=platform.platform(),
        devices=[str(x) for x in jax.devices()], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        subsequent_changes='Only new diagnostic/controller/test/report files; existing environment, learner and model sources unchanged.')
    write(root/'config.json', CONFIG)
    write(root/'provenance.json', provenance)
    write(root/'seed_audit.json', dict(npz_files_checked=checked, seed_arrays=arrays,
        candidate_overlap=False, text_search='rg exact integer seed tokens in ett/scripts/notes/artifacts .py/.json/.md: no 51000000..51000127 or smoke-seed use',
        rejected_suggestion='41000002 and 41000003 already used by scripts/test_future_window.py'))
    write(root/'budget.json', dict(total=0, entries=[]))
    print('Prepared frozen hashes, seed lists, geometry and fixed 40200-transition plan.', flush=True)


def charge(root, label, count):
    ledger = read(root/'budget.json')
    assert ledger['total'] + count < CONFIG['strict_cap']
    assert label not in [e['label'] for e in ledger['entries']], 'No implicit retries'
    ledger['total'] += count
    ledger['entries'].append(dict(label=label, transitions_reserved=count))
    write(root/'budget.json', ledger)  # charge before work, including failed attempts


def native_episode(seed, route):
    sim = TimedSimulator(int(seed))
    assert sim.env.active_prob == .3 and sim.env._action_noise == .01
    assert sim.env.max_episode_steps == 50 and sim.env.n_frames == 4
    obs = sim.observation()
    states, goals, actions, rewards, progress = [obs[:8]], [obs[8:]], [], [], []
    # Privileged labels and full-precision position are evaluation-only output.
    failures, true_xy = [], [sim.env.state.copy()]
    index = np.int32(0)
    for _ in range(50):
        action, index = controller(obs[:2], index, ROUTES[route])
        action = np.asarray(action)
        obs, reward, _ = sim.step(action)
        states.append(obs[:8]); goals.append(obs[8:]); actions.append(action)
        rewards.append(reward); progress.append(np.asarray(index))
        failures.append(sim.env.dead); true_xy.append(sim.env.state.copy())
    assert sim.elapsed == 50
    return dict(states=np.array(states), goals=np.array(goals), action=np.array(actions),
                reward=np.array(rewards), waypoint=np.array(progress),
                failure=np.array(failures), true_xy=np.array(true_xy))


def batch(root, label, seeds, route, initial=None, rollout=None, theta=None):
    charge(root, label, len(seeds)*50)
    records = []
    for i, seed in enumerate(seeds):
        if rollout is None:
            record = native_episode(seed, route)
        else:
            key = jax.random.fold_in(jax.random.PRNGKey(CONFIG['model_rng_base']), int(seed))
            record = jax.tree.map(np.asarray, rollout(theta, initial[i], ROUTES[route], key))
        record['return'] = record['reward'].astype(float) @ (.95**np.arange(50))
        records.append(record)
    result = {k: np.stack([r[k] for r in records]) for k in records[0]}
    result['reset_seed'] = np.asarray(seeds)
    each = validate(result, route)
    if rollout is None:
        # Preserve established success definition at native simulator precision.
        true_success = np.min(np.linalg.norm(result['true_xy']-GOAL[:2], axis=-1), axis=1) < .5
        np.testing.assert_array_equal(true_success, each['success'])
        result['success'] = true_success
    else:
        assert result['box_valid'].all()
        xy = result['states'][:, 1:, :2]
        assert np.abs(np.diff(result['states'][..., :2], axis=1)).max() <= 1+2e-6
        assert (xy >= result['box_low'][:, :, 0] - 2e-6).all()
        assert (xy <= result['box_high'][:, :, 0] + 2e-6).all()
        cells = np.minimum(np.floor(xy).astype(int), [8, 4])
        assert (POINTMAZE_WALLS[cells[..., 0], cells[..., 1]] == 0).all()
        result['success'] = each['success']
    result.update({k: v for k, v in each.items() if k != 'returns'})
    return result


def save(root, tag, record):
    audit = {k: record[k] for k in ['failure', 'true_xy'] if k in record}
    if audit:
        np.savez_compressed(root/f'{tag}_native_audit.npz', **audit, reset_seed=record['reset_seed'])
    auxiliary = {'x_prime': record['x_prime']} if 'x_prime' in record else {}
    if auxiliary:
        np.savez_compressed(root/f'{tag}_auxiliary.npz', **auxiliary, reset_seed=record['reset_seed'])
    np.savez_compressed(root/f'{tag}.npz', **{k: v for k, v in record.items() if k not in audit and k not in auxiliary})


def run(root):
    verify_hashes(root)
    assert not read(root/'budget.json')['entries'], 'Run already started; no implicit retry'
    paths = frozen_paths()
    # Do not call historical setup(): it unnecessarily restores an actor/critic.
    model = ConvexActionTransition(load_anchored(paths['diagonal']).diagonal, bound=1.)
    nominal = load_nominal_policy(str(Path(paths['nominal']).parent))
    assert nominal.spec.conditioning == 'state_goal' and nominal.spec.num_components == 5
    coefficients = {s: np.load(paths[f's{s}'])['theta'] for s in [0, 1]}
    for s in [0, 1]:
        assert float(np.load(paths[f's{s}'])['bound']) == 1.
    frozen = signature(model, nominal, coefficients)
    rollout = make_rollout(model, nominal)
    # Separate smoke is fixed in advance; results never alter controller rules.
    for phase, seeds in [('smoke', CONFIG['smoke_seeds']), ('main', CONFIG['reset_seeds'])]:
        initial = None
        reference = {}
        for environment in ['native', 'model_s0', 'model_s1']:
            for route in ROUTES:
                tag = f'{environment}_{route}'
                record = batch(root, f'{phase}_{tag}', seeds, route, initial,
                               None if environment == 'native' else rollout,
                               None if environment == 'native' else coefficients[int(environment[-1])])
                if initial is None:
                    initial = record['states'][:, 0].copy()
                np.testing.assert_array_equal(record['states'][:, 0], initial)
                np.testing.assert_array_equal(initial, np.broadcast_to(START, initial.shape))
                save(root, f'{phase}_{tag}', record)
                reference[tag] = record
                print(phase, tag, 'completed', len(seeds)*50, 'transitions', flush=True)
        if phase == 'main':
            for environment in ['native', 'model_s0', 'model_s1']:
                for route in ROUTES:
                    tag = f'{environment}_{route}'
                    replay = batch(root, f'replay_{tag}', seeds[:4], route, initial[:4],
                                   None if environment == 'native' else rollout,
                                   None if environment == 'native' else coefficients[int(environment[-1])])
                    for field in replay:
                        np.testing.assert_array_equal(replay[field], reference[tag][field][:4])
    assert signature(model, nominal, coefficients) == frozen
    verify_hashes(root)
    assert read(root/'budget.json')['total'] == 40200
    write(root/'verification.json', dict(frozen_signature_before=frozen, frozen_signature_after=frozen,
        file_hashes_unchanged=True, all_actions_independently_reconstructed=True,
        waypoint_progression_checked=True, goal_and_F4_checked=True,
        direct_reward_and_return_checked=True, model_geometry_checked=True,
        matched_initial_observations=True, native_true_and_visible_success_identical=True,
        first_four_all_fields_replay_exact=True, replay_transitions=1200,
        controller_inputs=['current visible XY', 'waypoint progress index', 'fixed geometry waypoints'],
        model_inputs=['current visible F4', 'commanded goal F4', 'nominal x_prime', 'controller action', 'model PRNG key'],
        privileged_data='native failure/true precision XY saved only in *_native_audit.npz; never passed to controller or ETT',
        actor_critic_loaded=False, training_updates=0, total_transitions=40200))
    print('All rollouts and exact replays passed; frozen artifacts unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=['prepare', 'run'])
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else run)(args.out_dir)
