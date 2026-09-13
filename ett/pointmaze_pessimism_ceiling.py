"""One-shot, answer-handed PointMaze ETT pessimism headroom diagnostic.

This is a model-only diagnostic.  It performs no fitting, gradient update,
checkpoint write, native rollout, or critic/MC-guided adversary construction.
The deliberately answer-handed oracle arms are lower bounds on achievable
pessimism, not valid worst-case bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from ett.convex_action_transition import RECTANGLE_HIGH, RECTANGLE_LOW, convex_box, emit
from ett.diagonal_transition import POINTMAZE_WALLS, _project_samples, sample_displacement
from ett.finite_crl import sha, write
from ett.policy_improvement import tree_sha
from ett.rollout_return import DISCOUNT, GOAL, START, task_reward
from ett.run_convex_adversarial import inputs, setup


ARTIFACT = Path('artifacts/pointmaze_region_pilot/pessimism_ceiling_v1')
ROOTS = Path('artifacts/pointmaze_region_pilot/phase_sampling_s01_v1/final_contexts.npz')
KERNELS = Path('artifacts/pointmaze_region_pilot/update_reference_s01_v1/final_kernels.npz')
NORMALIZATION = Path('artifacts/pointmaze_region_pilot/early_contexts_s01_v1/contexts.npz')
REFERENCE_COMMIT = '817e429'
ARM_NAMES = ('a0_diagonal', 'a1_s0_learned', 'a1_s1_learned',
             'a2_lipschitz_oracle', 'a3_box_oracle', 'a4_postprocess_oracle')
ARM_CODES = (0, 1, 1, 2, 3, 4)
CONFIG = dict(
    arms=list(ARM_NAMES), horizon_setting1=10, roots_setting1=36,
    repeats_setting1=64, horizon_setting2=50, rollouts_setting2=128,
    gamma=.95, return_scale=.05, transition_seed_base=192000000,
    policy_seed_base=193000000, arm_seed_stride=100000,
    bootstrap_seed=191000000, bootstrap_repeats=2000,
    response_bound=1., numerical_tolerance=2e-6,
    a0_a4_output_cap=600000, a5_output_cap=1500000,
    identity_contexts=128, identity_draws=8,
    lipschitz_contexts=128, lipschitz_pairs=16, lipschitz_draws=1,
    planned_main_outputs=176640, planned_probe_outputs=30720,
    planned_a0_a4_outputs=207360,
    a5_protocol=dict(candidates='four selected-box corners plus anchor', depth=3,
                     continuations_per_candidate=8, decisions=29440,
                     lookahead_outputs=3532800, realized_outputs=29440,
                     probe_outputs=5120, total_including_a0_a4=3774720,
                     status='skipped_pre_outcome_cap_exceeded'),
    substantial_gap=dict(relative_reduction_percentage_points=.05,
                         require_direct_paired_ci_below_zero=True),
    tail_support_threshold=20,
    selection='all arms fixed before outcomes; no selection, sweep, or rerun',
    updates=0, native_steps=0, checkpoint_writes=0,
)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def protocol_text():
    return """# Protocol: ETT pessimism headroom ceiling diagnostic

## Status and estimand

This protocol is sealed before model rollouts. It is a pure model diagnostic:
zero learning, gradients, actor updates, native simulation, or checkpoint
writes. Answer-handed A2--A4 are constructive lower bounds on achievable
pessimism under their stated rules, not certified ceilings or valid bounds.

The repository contains no literal 36-root set collected at t=40. The fixed
t=40/H=10 artifact contains 64 roots, while the exact group-balanced 36-root
held-out set used with the learned responses at commit 817e429 has source times
1, 3, 6, and 10. To avoid inventing a 36-of-64 subset, setting 1 uses those
exact 36 held-out roots and imposes a fresh fixed H=10. Source-time metadata are
retained. Consequently its A0 need not reproduce the historical ~0.60 mean.

## Frozen arms and streams

All arms share the loaded diagonal law, nominal policy, actor, goal, geometry,
normalization, anchor draw, coordinate-step rule, boundaries, and wall endpoint
processing. Each step draws x' from the nominal policy and x from the actor.
Transition randomness starts at 192000000 and policy randomness at 193000000;
the same per-setting streams are replayed across arms. The 100000 arm stride is
reserved in configuration but deliberately not added to replay keys, because
adding it would destroy the required cross-arm pairing.

- A0: the diagonal anchor (zero response).
- A1-s0/A1-s1: only the 32 response coordinates of the frozen final critic ETT
  kernels associated with 817e429; their 16 diagonal-head offsets are ignored.
- A2: anchor + ||x-x'|| times the unit vector from goal to anchor, projected by
  the existing selected convex box. At anchor=goal, +x is the sealed tie break.
- A3: the selected convex-box corner farthest from goal. Its pre-projection
  proposal is an outward ray, so the literal active-box-clip indicator remains
  meaningful.
- A4: the farthest point from goal among all free-rectangle intersections with
  the coordinate-step square, followed only by the existing coordinate,
  boundary, and wall endpoint postprocessing.

Every arm has an exact x=x' branch returning the identical sampled anchor.

## Evaluation, intervals, and mechanisms

Setting 1 has 36 roots x 64 rollouts x H=10. Setting 2 has 128 independent
START rollouts x H=50. Returns are 0.05 sum_t .95^t r_t. Report median, p10,
p25, p75, p90, and mean. Paired differences use 2,000 bootstrap replicates at
seed 191000000. Setting 1 resamples roots within its three 12-root strata and
also reports the conditional within-root Monte Carlo interval. Setting 2
resamples paired rollouts. Paired median differences receive bootstrap
intervals and are primary if fewer than 20 nonzero same-direction rollouts
support a mean difference.

The requested signed relative change is C=(A-A0)/A0 (negative is pessimistic).
For unambiguous thresholding, reduction magnitude is R=-C=(A0-A)/A0. The same
convention is applied to p25; a zero A0 p25 makes that threshold unavailable.
Top-5% signed and absolute contribution shares, realized delta-norm summaries,
literal active box-clip fractions, reward-sequence changes versus A0, and
pre-entry/entered splits are reported. Entry at a decision means a positive
reward occurred at an earlier step in that same trajectory.

## Acceptance and charge

Identity probes: 128 contexts x 8 draws x 6 arms. Lipschitz grid: 128 contexts
x 16 paired action comparisons x 2 outputs x 6 arms. Together with both
settings, the exact charge is 207,360, below 600,000. A2 must have samplewise
excess <=2e-6; A3/A4 are expected to fail because of their diagonal branch and
the failure is recorded. Every rollout must be finite, have valid anchors,
exact F4 history shift, legal endpoints, and exact recomputed rewards.

A5 was sealed as five candidates (four selected-box corners plus anchor),
depth 3, eight continuations per candidate at every decision. It would require
3,532,800 lookahead outputs plus 29,440 realized outputs and 5,120 probes, for
3,774,720 total including A0--A4. This exceeds 1.5M, so A5 is skipped before
outcomes.

## Decision rules (ordered)

The target decision uses setting 1; setting 2 is reported prominently as a
reset-start transport check. A1 means the more pessimistic of its two seeds.

1. If A3 and A4 each achieve neither 10% mean nor 10% p25 reduction, conclude
   reward geometry/postprocessing is binding, with a rare-move caveat.
2. Else if A2 mean R>=10% and both learned A1 R<3%, conclude optimizer
   bottleneck rather than response-family limitation.
3. Else if A3 exceeds A2 mean R by >=5 percentage points and the direct paired
   95% root and conditional-MC intervals for A3-A2 are below zero, conclude
   L=1 binds.
4. A5 would analogously exceed A3 by >=5 points with paired intervals below
   zero, but is predeclared skipped.
5. Otherwise conclude measured ceiling/no separation. A material setting
   disagreement (opposite mean direction or >=5-point R gap) is a mandatory
   qualification, not a reason to rerun.
"""


class Ledger:
    def __init__(self, out):
        self.out = out
        self.data = dict(cap=CONFIG['a0_a4_output_cap'], charged=0, entries=[])

    def add(self, outputs, purpose):
        outputs = int(outputs)
        if self.data['charged'] + outputs > self.data['cap']:
            raise RuntimeError('sealed model-output cap would be exceeded')
        self.data['charged'] += outputs
        self.data['entries'].append(dict(purpose=purpose, outputs=outputs))
        write(self.out / 'ledger.json', self.data)


def _farthest_corner(low, high, goal_xy):
    return jnp.where(jnp.abs(low - goal_xy[:, None, :]) >=
                     jnp.abs(high - goal_xy[:, None, :]), low, high)


def _global_farthest(state, goal_xy):
    low = jnp.maximum(jnp.asarray(RECTANGLE_LOW)[None], state[:, None, :2] - 1.)
    high = jnp.minimum(jnp.asarray(RECTANGLE_HIGH)[None], state[:, None, :2] + 1.)
    valid = jnp.all(low <= high, axis=-1)
    corner = _farthest_corner(low, high, goal_xy)
    distance = jnp.sum((corner - goal_xy[:, None, :]) ** 2, axis=-1)
    index = jnp.argmax(jnp.where(valid, distance, -jnp.inf), axis=-1)
    selected = jnp.take_along_axis(corner, index[:, None, None], axis=1)[:, 0]
    return selected, jnp.any(valid, axis=-1)


def _pack(xy, state):
    history = jnp.broadcast_to(state[:, None, :6], xy.shape[:-1] + (6,))
    return jnp.concatenate([xy, history], axis=-1)


def apply_arm(code, theta, state, action, xp, goal, anchor):
    """Apply a sealed response rule to already sampled diagonal anchors."""
    low, high, valid, rectangle = convex_box(state, anchor)
    diagonal = jnp.all(action == xp, axis=-1)[:, None, None]
    goal_xy = goal[:, None, :2]

    def a0(_):
        xy = anchor
        return xy, anchor, jnp.zeros(anchor.shape[:-1], bool)

    def a1(_):
        output, detail = emit(theta, state, action, xp, anchor, 1.)
        proposal = anchor + detail['response'][:, None, :]
        return output[..., :2], proposal, detail['projection_corrected']

    def a2(_):
        away = anchor - goal_xy
        norm = jnp.linalg.norm(away, axis=-1, keepdims=True)
        tie = jnp.broadcast_to(jnp.array([1., 0.], anchor.dtype), away.shape)
        direction = jnp.where(norm > 0., away / jnp.maximum(norm, 1e-12), tie)
        radius = jnp.linalg.norm(action - xp, axis=-1)[:, None, None]
        proposal = anchor + radius * direction
        xy = jnp.clip(proposal, low, high)
        return xy, proposal, jnp.any(xy != proposal, axis=-1)

    def a3(_):
        corner = _farthest_corner(low, high, goal[:, :2])
        ray = goal_xy + 1024. * (corner - goal_xy)
        clipped = jnp.clip(ray, low, high)
        xy = jnp.where(diagonal, anchor, corner)
        proposal = jnp.where(diagonal, anchor, ray)
        active = jnp.where(diagonal[..., 0], False,
                           jnp.any(clipped != ray, axis=-1))
        return xy, proposal, active

    def a4(_):
        target, target_valid = _global_farthest(state, goal[:, :2])
        target = jnp.where(target_valid[:, None], target, jnp.nan)
        raw_delta = jnp.broadcast_to((target - state[:, :2])[:, None, :], anchor.shape)
        projected, _ = _project_samples(state, raw_delta, _BASE_SPEC.value)
        xy = jnp.where(diagonal, anchor, projected[..., :2])
        proposal = jnp.where(diagonal, anchor, target[:, None, :])
        return xy, proposal, jnp.zeros(anchor.shape[:-1], bool)

    xy, proposal, active = jax.lax.switch(code, (a0, a1, a2, a3, a4), operand=None)
    output = _pack(jnp.where(valid[..., None], xy, jnp.nan), state)
    return output, dict(anchor=anchor, box_low=low, box_high=high,
                        valid=valid, rectangle=rectangle, proposal=proposal,
                        box_clip_active=active,
                        delta_norm=jnp.linalg.norm(xy - anchor, axis=-1))


class _SpecHolder:
    value = None


_BASE_SPEC = _SpecHolder()


class CeilingEngine:
    def __init__(self, model, nominal, actor, horizon):
        self.model, self.nominal, self.actor = model, nominal, actor
        self.base, self.horizon = model.diagonal, int(horizon)
        _BASE_SPEC.value = self.base.spec

        def generate(code, theta, states, goals, transition_key, policy_key):
            transition_keys = jax.random.split(transition_key, self.horizon)
            policy_keys = jax.random.split(policy_key, self.horizon)

            def step(state, keys):
                tk, pk = keys
                nominal_key, actor_key = jax.random.split(pk)
                xp = nominal.sample(state, nominal_key, 1, goal=goals)
                action = actor(state, goals, actor_key)
                context = jnp.concatenate([state, xp, xp, goals], axis=-1)
                distribution = self.base._distribution(self.base.params, context)
                raw_delta, atom = sample_displacement(
                    distribution, tk, 1, self.base.delta_mean,
                    self.base.delta_std, self.base.spec)
                anchor_state, base_detail = _project_samples(state, raw_delta, self.base.spec)
                output, detail = apply_arm(code, theta, state, action, xp, goals,
                                           anchor_state[..., :2])
                following = output[:, 0]
                return following, dict(
                    next_state=following, action=action, x_prime=xp,
                    reward=task_reward(following, goals),
                    anchor=detail['anchor'][:, 0], delta_norm=detail['delta_norm'][:, 0],
                    box_clip_active=detail['box_clip_active'][:, 0],
                    box_low=detail['box_low'][:, 0], box_high=detail['box_high'][:, 0],
                    rectangle=detail['rectangle'][:, 0], valid=detail['valid'][:, 0],
                    stationary_atom=atom[:, 0],
                    base_corrected=jnp.any(base_detail['raw_position'] !=
                                           anchor_state[..., :2], axis=-1)[:, 0])

            _, record = jax.lax.scan(step, states, (transition_keys, policy_keys))
            record = jax.tree.map(lambda value: jnp.swapaxes(value, 0, 1), record)
            record['states'] = jnp.concatenate([states[:, None], record.pop('next_state')], 1)
            return record

        self._generate = jax.jit(generate)

        def probe(code, theta, state, action, xp, goal, key, draws):
            context = jnp.concatenate([state, xp, xp, goal], axis=-1)
            distribution = self.base._distribution(self.base.params, context)
            raw_delta, _ = sample_displacement(distribution, key, draws,
                                               self.base.delta_mean, self.base.delta_std,
                                               self.base.spec)
            anchor_state, _ = _project_samples(state, raw_delta, self.base.spec)
            return apply_arm(code, theta, state, action, xp, goal,
                             anchor_state[..., :2])

        self._probe = jax.jit(probe, static_argnums=(7,))

    def run(self, code, theta, states, transition_seed, policy_seed):
        goals = np.broadcast_to(GOAL, np.asarray(states).shape)
        result = self._generate(jnp.int32(code), jnp.asarray(theta), jnp.asarray(states),
                                jnp.asarray(goals), jax.random.PRNGKey(transition_seed),
                                jax.random.PRNGKey(policy_seed))
        return jax.tree.map(np.asarray, result)

    def probe(self, code, theta, states, actions, xp, seed, draws):
        goals = np.broadcast_to(GOAL, np.asarray(states).shape)
        output, detail = self._probe(jnp.int32(code), jnp.asarray(theta), jnp.asarray(states),
                                     jnp.asarray(actions), jnp.asarray(xp), jnp.asarray(goals),
                                     jax.random.PRNGKey(seed), int(draws))
        return np.asarray(output), jax.tree.map(np.asarray, detail)


def endpoint_valid(xy, tolerance=2e-6):
    xy = np.asarray(xy)
    return np.any(np.all((xy[..., None, :] >= RECTANGLE_LOW - tolerance) &
                         (xy[..., None, :] <= RECTANGLE_HIGH + tolerance), axis=-1), axis=-1)


def validate_record(record, arm):
    states = np.asarray(record['states'])
    if not all(np.isfinite(np.asarray(record[name])).all()
               for name in ('states', 'action', 'x_prime', 'anchor', 'delta_norm')):
        raise ValueError(f'{arm}: NaN or infinity')
    if not np.asarray(record['valid']).all():
        raise ValueError(f'{arm}: invalid diagonal anchor box')
    np.testing.assert_array_equal(states[:, 1:, 2:], states[:, :-1, :6])
    if np.max(np.abs(states[:, 1:, :2] - states[:, :-1, :2])) > 1 + 2e-6:
        raise AssertionError(f'{arm}: coordinate-step bound failed')
    if not endpoint_valid(states[:, 1:, :2]).all():
        raise AssertionError(f'{arm}: non-free endpoint')
    goals = np.broadcast_to(GOAL, states[:, 1:].shape)
    expected = np.asarray(task_reward(states[:, 1:], goals))
    np.testing.assert_array_equal(record['reward'], expected)
    if np.max(np.abs(record['action'])) > 1 or np.max(np.abs(record['x_prime'])) > 1:
        raise AssertionError(f'{arm}: invalid action')
    if arm == 'a0_diagonal':
        np.testing.assert_array_equal(states[:, 1:, :2], record['anchor'])
        np.testing.assert_array_equal(record['delta_norm'], np.zeros_like(record['delta_norm']))


def response_parameters():
    with np.load(KERNELS, allow_pickle=False) as data:
        s0 = np.asarray(data['critic_s0'][16:], np.float32)
        s1 = np.asarray(data['critic_s1'][16:], np.float32)
    if s0.shape != (32,) or s1.shape != (32,) or not np.isfinite([s0, s1]).all():
        raise ValueError('invalid frozen learned response parameters')
    zero = np.zeros(32, np.float32)
    return (zero, s0, s1, zero, zero, zero)


def source_hashes():
    resolved = inputs()
    paths = list(resolved.values()) + [ROOTS, KERNELS, NORMALIZATION,
        Path('ett/convex_action_transition.py'), Path('ett/diagonal_transition.py'),
        Path('ett/rollout_return.py'), Path('ett/run_convex_adversarial.py'),
        Path('ett/pointmaze_pessimism_ceiling.py'),
        Path('ett/report_pointmaze_pessimism_ceiling.py'),
        Path('scripts/test_pointmaze_pessimism_ceiling.py')]
    return {path.as_posix(): sha(path) for path in paths}


def prepare(out):
    if out.exists():
        raise ValueError('use a fresh output directory')
    if CONFIG['planned_a0_a4_outputs'] > CONFIG['a0_a4_output_cap']:
        raise AssertionError('invalid predeclared A0-A4 charge')
    if CONFIG['a5_protocol']['total_including_a0_a4'] <= CONFIG['a5_output_cap']:
        raise AssertionError('A5 skip is inconsistent with charge')
    with np.load(ROOTS, allow_pickle=False) as data:
        roots, indices = np.asarray(data['roots']), np.asarray(data['indices'])
    if roots.shape != (36, 8) or indices.shape != (36, 3):
        raise ValueError('unexpected held-out root artifact')
    groups, counts = np.unique(indices[:, 2], return_counts=True)
    if not np.array_equal(counts, np.array([12, 12, 12])):
        raise ValueError('held-out roots are not three balanced strata')
    out.mkdir(parents=True)
    (out / 'PROTOCOL.md').write_text(protocol_text(), encoding='utf-8')
    write(out / 'config.json', CONFIG)
    hashes = source_hashes()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', REFERENCE_COMMIT, 'HEAD']).returncode == 0
    paths = inputs()
    provenance = dict(prepared_utc=utc(), git_head=head,
        reference_commit=REFERENCE_COMMIT, reference_is_ancestor=ancestor,
        input_sha256=hashes, setup_paths={k: v.as_posix() for k, v in paths.items()},
        root_resolution=dict(requested='36 existing roots at t=40, H=10',
            resolved='36 held-out group-balanced roots from phase_sampling; fixed fresh H=10',
            caveat='source-time metadata are 1,3,6,10; no literal 36-root t=40 artifact exists'),
        learned_arm='critic_s0 and critic_s1 final response coordinates [16:48]; diagonal offsets discarded',
        forbidden='critics, MC values, native simulator, outcome-based tuning, gradient updates')
    write(out / 'provenance.json', provenance)
    write(out / 'preregistration.json', dict(sealed_utc=utc(),
        protocol_sha256=sha(out / 'PROTOCOL.md'), config_sha256=sha(out / 'config.json'),
        provenance_sha256=sha(out / 'provenance.json'), planned_outputs=207360,
        decision_rule_order=[1, 2, 3, 4, 5], a5_status='skipped_before_outcomes'))
    print(f'Sealed protocol and exact {CONFIG["planned_a0_a4_outputs"]:,}-output charge.', flush=True)


def verify_prepared(out):
    prereg = json.loads((out / 'preregistration.json').read_text())
    for name in ('protocol', 'config', 'provenance'):
        if sha(out / f'{name.upper()}.md' if name == 'protocol' else out / f'{name}.json') != prereg[f'{name}_sha256']:
            raise ValueError(f'{name} changed after sealing')
    provenance = json.loads((out / 'provenance.json').read_text())
    for path, digest in provenance['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'frozen input changed: {path}')


def _reshape_record(record, leading):
    return {name: np.asarray(value).reshape(tuple(leading) + value.shape[1:])
            for name, value in record.items()}


def _save_setting(path, metadata, records):
    arrays = dict(metadata)
    for arm, record in records.items():
        for name, value in record.items():
            arrays[f'{arm}__{name}'] = value
        rewards = np.asarray(record['reward'])
        arrays[f'{arm}__return'] = .05 * (rewards @ (.95 ** np.arange(rewards.shape[-1])))
    np.savez_compressed(path, **arrays)


def probes(out, engine, thetas, ledger, roots):
    n = CONFIG['identity_contexts']
    states = np.asarray(roots)[np.arange(n) % len(roots)]
    goals = np.broadcast_to(GOAL, states.shape)
    xp = np.asarray(engine.nominal.sample(states, jax.random.PRNGKey(193099998), 1, goal=goals))
    identity = {}
    for arm, code, theta in zip(ARM_NAMES, ARM_CODES, thetas):
        ledger.add(n * CONFIG['identity_draws'], f'identity/{arm}')
        output, detail = engine.probe(code, theta, states, xp, xp, 192099998,
                                      CONFIG['identity_draws'])
        np.testing.assert_array_equal(output[..., :2], detail['anchor'])
        identity[arm] = dict(max_drift=0., exact=True)

    rng = np.random.default_rng(191000001)
    pairs = []
    for k in range(CONFIG['lipschitz_pairs']):
        if k < 8:
            first = xp.copy()
            direction = np.zeros_like(xp)
            direction[:, k % 2] = (1 if (k // 2) % 2 == 0 else -1) * 1e-3
            second = np.clip(xp + direction, -1, 1)
        else:
            first = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
            second = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        pairs.append((first, second))
    lipschitz = {}
    raw = {}
    for arm, code, theta in zip(ARM_NAMES, ARM_CODES, thetas):
        excesses = []
        violations = []
        for k, (first, second) in enumerate(pairs):
            seed = 192100000 + k
            ledger.add(2 * n, f'lipschitz/{arm}/pair{k}')
            y0, _ = engine.probe(code, theta, states, first, xp, seed, 1)
            y1, _ = engine.probe(code, theta, states, second, xp, seed, 1)
            output_distance = np.linalg.norm(y1[:, 0, :2] - y0[:, 0, :2], axis=-1)
            action_distance = np.linalg.norm(second - first, axis=-1)
            excess = output_distance - action_distance
            excesses.append(excess)
            violations.append(excess > CONFIG['numerical_tolerance'])
        excesses = np.stack(excesses)
        violations = np.stack(violations)
        lipschitz[arm] = dict(max_excess=float(excesses.max()),
            violation_fraction=float(violations.mean()),
            expected_to_pass=arm not in ('a3_box_oracle', 'a4_postprocess_oracle'),
            passed=bool(excesses.max() <= CONFIG['numerical_tolerance']))
        raw[f'{arm}__excess'] = excesses
    if not lipschitz['a2_lipschitz_oracle']['passed']:
        raise AssertionError('A2 samplewise action-Lipschitz grid failed')
    if any(not lipschitz[a]['passed'] for a in ARM_NAMES[:4]):
        raise AssertionError('a nominally Lipschitz arm failed the grid')
    np.savez_compressed(out / 'probe_arrays.npz', **raw)
    result = dict(identity=identity, lipschitz=lipschitz,
                  diagonal_identity_all_exact=all(v['exact'] for v in identity.values()))
    write(out / 'acceptance_probes.json', result)
    return result


def run(out):
    verify_prepared(out)
    if (out / 'started.json').exists():
        raise ValueError('the one-shot rollout phase was already started; rerun prohibited')
    write(out / 'started.json', dict(utc=utc(), phase='one-shot model rollouts'))
    ledger = Ledger(out)
    write(out / 'ledger.json', ledger.data)
    with np.load(ROOTS, allow_pickle=False) as data:
        roots, indices = np.asarray(data['roots'], np.float32), np.asarray(data['indices'])
    groups = indices[:, 2].astype(np.int32)
    thetas = response_parameters()
    model, nominal, actor, _, actor_info = setup('rectangle')
    frozen_before = dict(diagonal_tree_sha256=tree_sha(model.diagonal.params),
                         nominal_tree_sha256=tree_sha(nominal.params),
                         actor_checkpoint_sha256=sha(inputs()['actor']),
                         normalization_sha256=sha(NORMALIZATION),
                         response_checkpoint_sha256=sha(KERNELS))
    probe_state = np.concatenate([roots[:8], np.broadcast_to(START, (8, 8))])
    probe_goal = np.broadcast_to(GOAL, probe_state.shape)
    policy_key = jax.random.PRNGKey(193099999)
    actor_before = np.asarray(actor(probe_state, probe_goal, policy_key))
    nominal_before = np.asarray(nominal.sample(probe_state, policy_key, 4, goal=probe_goal))

    engine10 = CeilingEngine(model, nominal, actor, 10)
    setting1_states = np.repeat(roots, CONFIG['repeats_setting1'], axis=0)
    setting1_records = {}
    for arm, code, theta in zip(ARM_NAMES, ARM_CODES, thetas):
        outputs = len(setting1_states) * 10
        ledger.add(outputs, f'setting1/{arm}')
        record = engine10.run(code, theta, setting1_states, 192000000, 193000000)
        validate_record(record, arm)
        setting1_records[arm] = _reshape_record(record, (36, 64))
        print(f'setting 1 {arm}: complete', flush=True)
    _save_setting(out / 'setting1_raw.npz', dict(roots=roots, source_indices=indices, groups=groups),
                  setting1_records)

    engine50 = CeilingEngine(model, nominal, actor, 50)
    setting2_states = np.broadcast_to(START, (128, 8)).copy()
    setting2_records = {}
    for arm, code, theta in zip(ARM_NAMES, ARM_CODES, thetas):
        outputs = len(setting2_states) * 50
        ledger.add(outputs, f'setting2/{arm}')
        record = engine50.run(code, theta, setting2_states, 192000001, 193000001)
        validate_record(record, arm)
        setting2_records[arm] = record
        print(f'setting 2 {arm}: complete', flush=True)
    _save_setting(out / 'setting2_raw.npz', dict(starts=setting2_states), setting2_records)

    acceptance = probes(out, engine10, thetas, ledger, roots)
    if ledger.data['charged'] != CONFIG['planned_a0_a4_outputs']:
        raise AssertionError(f'charge mismatch: {ledger.data["charged"]}')
    frozen_after = dict(diagonal_tree_sha256=tree_sha(model.diagonal.params),
                        nominal_tree_sha256=tree_sha(nominal.params),
                        actor_checkpoint_sha256=sha(inputs()['actor']),
                        normalization_sha256=sha(NORMALIZATION),
                        response_checkpoint_sha256=sha(KERNELS))
    if frozen_before != frozen_after:
        raise AssertionError('a frozen object changed')
    np.testing.assert_array_equal(actor_before, np.asarray(actor(probe_state, probe_goal, policy_key)))
    np.testing.assert_array_equal(nominal_before,
                                  np.asarray(nominal.sample(probe_state, policy_key, 4, goal=probe_goal)))
    write(out / 'frozen_hashes.json', dict(before=frozen_before, after=frozen_after,
                                           exact_match=True, actor_info=actor_info))
    write(out / 'completion.json', dict(utc=utc(), status='complete',
        charged_model_outputs=ledger.data['charged'], cap=ledger.data['cap'],
        a5_status=CONFIG['a5_protocol']['status'], acceptance=acceptance,
        updates=0, native_steps=0, checkpoint_writes=0,
        raw_sha256={name: sha(out / name) for name in
                    ('setting1_raw.npz', 'setting2_raw.npz', 'probe_arrays.npz')}))
    print(f'One-shot rollout phase complete: {ledger.data["charged"]:,} outputs.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    parser.add_argument('--out-dir', type=Path, default=ARTIFACT)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run}[args.phase](args.out_dir)


if __name__ == '__main__':
    main()
