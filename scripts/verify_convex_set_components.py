"""Saved-input and analytic emitter checks; never collect or chain transitions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import jax
import jax.numpy as jnp
import numpy as np

from ett.convex_action_transition import (
    emit_with_geometry, fork_segment, RECTANGLE_LOW, RECTANGLE_HIGH,
)
from scripts.test_convex_set_transition import historical_module

ROOT = Path('artifacts/ett_convex_set_component/fork_segment_v1')
SOURCE = Path('artifacts/ett_route_diagnostic/fixed_visible_s01_n128_v1')
MODELS = Path('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1/checkpoints')
TOL = 2e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))
    path.write_text(json.dumps(data, indent=2, default=convert)+'\n', encoding='utf-8', newline='\n')


def evaluate(mode, theta, state, action, xp, anchor):
    fn = jax.jit(lambda t, s, x, p, a: emit_with_geometry(t, s, x, p, a, geometry_mode=mode))
    return jax.tree.map(np.asarray, fn(theta, state, action, xp, anchor))


def validate_segments(state, anchor, detail):
    """Float64 endpoint/crossing inclusion checks, distinct from the proof."""
    mask = detail['segment_available'][:, 0]
    a = anchor[mask, 0].astype(float)
    b = detail['segment_end'][mask, 0].astype(float)
    lo, hi = (state[mask, :2]-1.).astype(float), (state[mask, :2]+1.).astype(float)
    assert np.all((a >= lo) & (a <= hi)) and np.all((b >= lo) & (b <= hi))
    assert np.all((b[:, 0] >= 1.) & (b[:, 0] <= RECTANGLE_HIGH[2, 0]) & (b[:, 1] < 3.))
    assert np.all((a[:, 1] >= 3.) & (a[:, 1] <= RECTANGLE_HIGH[0, 1]))
    ray = a[:, 0] > RECTANGLE_HIGH[2, 0]
    crossing = a[ray, 0]+(3.-a[ray, 1])*(b[ray, 0]-a[ray, 0])/(b[ray, 1]-a[ray, 1])
    assert np.all((crossing >= 1.) & (crossing <= 1.9375))
    return {'segments': int(mask.sum()), 'ray_segments': int(ray.sum()),
            'max_ray_crossing_x': float(crossing.max()) if len(crossing) else None,
            'literal_unit_step_excess_of_selected_anchors': float(np.maximum(0, np.abs(a-state[mask, :2].astype(float))-1).max()) if len(a) else 0.}


def check_output(state, anchor, output, detail):
    xy = output[..., :2]
    np.testing.assert_array_equal(output[..., 2:], np.broadcast_to(state[:, None, :6], output.shape[:-1]+(6,)))
    assert detail['box_valid'].all() and np.isfinite(output).all()
    selected = detail['segment_available']
    projected = anchor + detail['segment_fraction'][..., None]*(detail['segment_end']-anchor)
    np.testing.assert_allclose(xy[selected], projected[selected], atol=TOL, rtol=0)
    assert np.all((detail['segment_fraction'] >= 0) & (detail['segment_fraction'] <= 1))
    assert np.all(xy >= state[:, None, :2]-1.-TOL)
    assert np.all(xy <= state[:, None, :2]+1.+TOL)
    free = np.any(np.all((xy[..., None, :] >= RECTANGLE_LOW-TOL) &
                         (xy[..., None, :] <= RECTANGLE_HIGH+TOL), axis=-1), axis=-1)
    assert free.all()


def action_pairs(theta, state, xp, anchor):
    # Grid plus a near-pair. The proof, not this finite grid, covers all actions.
    grid = np.array([(x, y) for x in np.linspace(-1, 1, 5) for y in np.linspace(-1, 1, 5)] +
                    [(.25, -.5), (.250001, -.499999)], np.float32)
    n, k = len(state), len(grid)
    s = np.repeat(state, k, axis=0)
    a = np.repeat(anchor, k, axis=0)
    p = np.repeat(xp, k, axis=0)
    x = np.tile(grid, (n, 1))
    results = {}
    for mode in ['rectangle', 'fork_segment']:
        y, d = evaluate(mode, theta, s, x, p, a)
        if mode == 'fork_segment':
            check_output(s, a, y, d)
            for field in ['segment_available', 'segment_start', 'segment_end']:
                z = d[field].reshape((n, k)+d[field].shape[1:])
                np.testing.assert_array_equal(z, np.repeat(z[:, :1], k, axis=1))
        out = y[:, 0].reshape(n, k, 8).astype(float)
        od = np.linalg.norm(out[:, :, None]-out[:, None, :], axis=-1)
        ad = np.linalg.norm(grid.astype(float)[:, None]-grid.astype(float)[None, :], axis=-1)
        excess = float(np.maximum(0., od-ad).max())
        assert excess <= TOL
        results[mode] = {'contexts': n, 'actions_per_context': k,
                         'max_positive_bound_excess': excess,
                         'max_observed_ratio': float(np.max(od / np.where(ad > 0, ad, np.inf)))}
    return results


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((ROOT/'provenance.json').read_text())
    expected_changes = {'ett/convex_action_transition.py'}
    unchanged = {p: sha(p) == h for p, h in provenance['input_sha256'].items() if p not in expected_changes}
    assert all(unchanged.values()), [p for p, ok in unchanged.items() if not ok]
    assert sha(ROOT/'pre_change/convex_action_transition.py') == provenance['input_sha256']['ett/convex_action_transition.py']
    assert sha(ROOT/'SPEC.md') == provenance['specification_sha256']
    write(out/'config.json', {'geometry_modes': ['rectangle', 'fork_segment'], 'L': 1,
          'junction': [1.875, 3.], 'crossing_guard': 1.9375, 'squared_length_floor': 1e-12,
          'tolerance': TOL, 'model_seeds': [0, 1], 'saved_inputs': 'both lower-controller files, all 128x50 contexts each',
          'fork_subset': '1<=current X<2 and 3<=current Y<4 and active waypoint==1',
          'new_trajectories': 0, 'new_stochastic_model_draws': 0,
          'example_selection': 'first prior certified seed-0 example in reset/time order',
          'no_model_parameters_updated': True})
    historical = jax.jit(historical_module().emit)
    rows, summaries, frozen = [], {}, {}
    for seed in [0, 1]:
        saved = np.load(SOURCE/f'main_model_s{seed}_lower.npz')
        auxiliary = np.load(SOURCE/f'main_model_s{seed}_lower_auxiliary.npz')
        checkpoint = np.load(MODELS/f's{seed}_adversarial.npz')
        theta = checkpoint['theta'].astype(np.float32)
        assert float(checkpoint['bound']) == 1.
        state = saved['states'][:, :-1].reshape(-1, 8)
        anchor = saved['anchor_xy'].reshape(-1, 1, 2)
        action = saved['action'].reshape(-1, 2)
        xp = auxiliary['x_prime'].reshape(-1, 2)
        reset = np.repeat(saved['reset_seed'], 50)
        time = np.tile(np.arange(50), 128)
        waypoint = saved['waypoint'].reshape(-1)
        fork = ((state[:, 0] >= 1) & (state[:, 0] < 2) & (state[:, 1] >= 3) & (state[:, 1] < 4) & (waypoint == 1))
        old, od = evaluate('rectangle', theta, state, action, xp, anchor)
        new, nd = evaluate('fork_segment', theta, state, action, xp, anchor)
        baseline = jax.tree.map(np.asarray, historical(theta, state, action, xp, anchor))
        assert baseline[1].keys() == od.keys()
        for before, after in zip(jax.tree.leaves(baseline), jax.tree.leaves((old, od))):
            np.testing.assert_array_equal(before, after)
        historical_error = float(np.max(np.abs(old[:, 0]-saved['states'][:, 1:].reshape(-1, 8))))
        assert historical_error <= TOL
        for mode in ['rectangle', 'fork_segment']:
            diag, _ = evaluate(mode, theta, state, xp, xp, anchor)
            np.testing.assert_array_equal(diag[..., :2], anchor)
            np.testing.assert_array_equal(diag[..., 2:], state[:, None, :6])
        check_output(state, anchor, new, nd)
        available = nd['segment_available'][:, 0]
        np.testing.assert_allclose(new[~available], old[~available], atol=TOL, rtol=0)
        geometry = validate_segments(state, anchor, nd)
        delta = np.linalg.norm(new[:, 0, :2].astype(float)-old[:, 0, :2].astype(float), axis=-1)
        summary = {'historical_default_bitwise_equal': True, 'saved_output_max_error': historical_error,
                   'exact_diagonal_and_F4_both_modes': True, 'geometry': geometry,
                   'checkpoint_sha256': sha(MODELS/f's{seed}_adversarial.npz'),
                   'action_pairs_frozen_parameters': action_pairs(theta, state[fork], xp[fork], anchor[fork])}
        for name, mask in [('all_saved', np.ones(len(state), bool)), ('fork_descent', fork)]:
            summary[name] = {'contexts': int(mask.sum()), 'segment_selected': int((available & mask).sum()),
                'upper_reference_box': int(((od['rectangle'][:, 0] == 0) & mask).sum()),
                'outputs_changed_above_tolerance': int(((delta > TOL) & mask).sum()),
                'mean_xy_change': float(delta[mask].mean()), 'max_xy_change': float(delta[mask].max()),
                'rectangle_below_Y3': int(((old[:, 0, 1] < 3) & mask).sum()),
                'optional_below_Y3': int(((new[:, 0, 1] < 3) & mask).sum()),
                'rectangle_down_from_current': int(((old[:, 0, 1] < state[:, 1]) & mask).sum()),
                'optional_down_from_current': int(((new[:, 0, 1] < state[:, 1]) & mask).sum())}
        summaries[f's{seed}'] = summary
        for i in np.flatnonzero(fork):
            rows.append({'model_seed': seed, 'reset_seed': int(reset[i]), 'time': int(time[i]),
                         'segment_selected': bool(available[i]), 'reference_rectangle': int(od['rectangle'][i, 0]),
                         'current_x': state[i, 0], 'current_y': state[i, 1],
                         'old_x': old[i, 0, 0], 'old_y': old[i, 0, 1],
                         'new_x': new[i, 0, 0], 'new_y': new[i, 0, 1], 'xy_change': delta[i]})
        np.savez_compressed(out/f's{seed}_component_outputs.npz', reset_seed=reset, time=time, waypoint=waypoint,
                            fork_descent=fork, state=state, anchor=anchor, action=action, x_prime=xp,
                            rectangle_output=old, optional_output=new, reference_rectangle=od['rectangle'],
                            response=od['response'], segment_end=nd['segment_end'], segment_available=available,
                            segment_fraction=nd['segment_fraction'])
        frozen[seed] = (state, anchor, action, xp, reset, time)
    with (out/'fork_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    prior = json.loads(Path('artifacts/ett_structure_audit/fork_saved_s01_v1/convex_segment_witnesses.json').read_text())
    example = min((row for row in prior if row['model_seed'] == 0), key=lambda row: (row['reset_seed'], row['time']))
    state, anchor, action, xp, reset, time = frozen[0]
    i = np.flatnonzero((reset == example['reset_seed']) & (time == example['time']))[0]
    s, a, x, p = state[i:i+1], anchor[i:i+1], action[i:i+1], xp[i:i+1]
    # Freeze geometry BEFORE deriving a witness matrix; no runtime target lookup.
    end, ok = fork_segment(jnp.asarray(s), jnp.asarray(a))
    assert bool(ok[0, 0])
    end = np.asarray(end)
    difference = x[0].astype(float)-p[0].astype(float)
    matrix = np.outer(end[0, 0].astype(float)-a[0, 0].astype(float), difference) / (difference @ difference)
    assert np.linalg.norm(matrix, 'fro') < 1.
    theta = np.tile(matrix.astype(np.float32).ravel(), 8)
    old, od = evaluate('rectangle', theta, s, x, p, a)
    new, nd = evaluate('fork_segment', theta, s, x, p, a)
    assert od['box_low'][0, 0, 1] == 3. and old[0, 0, 1] >= 3. and new[0, 0, 1] < 3.
    np.testing.assert_allclose(new[0, 0, :2], end[0, 0], atol=TOL, rtol=0)
    check_output(s, a, new, nd)
    pairs = action_pairs(theta, s, p, a)
    witness = {'label': 'existence certificate, not trained/ground-truth/supervised',
               'model_seed_context': 0, 'reset_seed': example['reset_seed'], 'time': example['time'],
               'state': s[0], 'anchor': a[0, 0], 'execution_action': x[0], 'nominal_action': p[0],
               'geometry_only_endpoint': end[0, 0], 'matrix_all_eight_blocks': matrix,
               'matrix_frobenius': np.linalg.norm(matrix, 'fro'), 'theta_float32': theta,
               'rectangle_output': old[0, 0, :2], 'optional_output': new[0, 0, :2],
               'fixed_configuration_action_pair_checks': pairs,
               'certificate': 'rank-one M=(Q-A)d^T/||d||^2; ||M||F=||Q-A||/||d||<1; fixed convex projection for every action'}
    write(out/'existence_certificate.json', witness)
    write(out/'comparison_summary.json', summaries)
    # Preserve the prior literal-real unit-step discrepancy, rather than masking it.
    strict_counts = {}
    for seed, (s, a, _, _, _, _) in frozen.items():
        fork_rows = [r for r in rows if r['model_seed'] == seed]
        _, _, _, _, reset, time = frozen[seed]
        indices = [np.flatnonzero((reset == r['reset_seed']) & (time == r['time']))[0] for r in fork_rows]
        excess = np.max(np.abs(a[indices, 0].astype(float)-s[indices, :2].astype(float)), axis=-1)-1.
        strict_counts[f's{seed}'] = {'anchors_exceeding_literal_unit_step': int((excess > 0).sum()),
                                     'max_excess': float(np.maximum(0., excess).max())}
    source_hashes = {str(p).replace('\\', '/'): sha(p) for p in [Path('ett/convex_action_transition.py'),
                      Path('scripts/test_convex_set_transition.py'), Path('scripts/verify_convex_set_components.py')]}
    assert all(sha(p) == provenance['input_sha256'][p] for p in unchanged)
    write(out/'verification.json', {'passed': True, 'new_trajectories': 0, 'new_stochastic_draws': 0,
          'saved_contexts': 12800, 'unchanged_inputs_verified': len(unchanged), 'all_frozen_inputs_unchanged': True,
          'preserved_strict_step_discrepancy': strict_counts, 'source_sha256': source_hashes,
          'intentional_source_change': {'path': 'ett/convex_action_transition.py',
             'before': provenance['input_sha256']['ett/convex_action_transition.py'],
             'after': sha('ett/convex_action_transition.py')},
          'python': sys.version, 'numpy': np.__version__, 'jax': jax.__version__, 'platform': platform.platform(),
          'devices': [str(d) for d in jax.devices()], 'tolerance': TOL})
    print(json.dumps({'comparisons': summaries, 'witness': witness, 'strict_step': strict_counts}, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    run(parser.parse_args().out_dir)
