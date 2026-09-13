"""Saved-array ETT fork audit. No sampler, simulator, training, or rollout calls.

All capacity statements are conditional one-query results unless explicitly
labelled as a full action-domain witness. The proofs are in DERIVATIONS.md.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SOURCE = Path('artifacts/ett_route_diagnostic/fixed_visible_s01_n128_v1')
MODELS = Path('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1')
LOW = np.array([[0, 3], [1, 1], [1, 1], [7, 1]], np.float32)
HIGH = np.array([[9, 4], [8, 2], [2, 4], [8, 4]], np.float32)
HIGH = np.where(HIGH == 9, HIGH, np.nextafter(HIGH, np.float32(-np.inf)))
TOL = 2e-6  # reconstruction only; inequalities/capacity use float64, no tolerance


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, indent=2, default=default)+'\n', encoding='utf-8', newline='\n')


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def boxes(q, anchor):
    # Independent float32 reconstruction of the production selector, including
    # float upper edges, areas and first-index tie breaking.
    lo = np.maximum(LOW, q[None]-1)
    hi = np.minimum(HIGH, q[None]+1)
    contains = np.all((anchor >= lo) & (anchor <= hi), axis=1)
    area = np.prod(np.maximum(hi-lo, 0), axis=1)
    assert contains.any()
    chosen = int(np.argmax(np.where(contains, area, -1)))
    return lo, hi, contains, chosen, area


def response_matrix(s, theta):
    # Float64, independently from the float32 JAX production expressions.
    frames = s.astype(float).reshape(4, 2)
    motion = np.sqrt(np.mean(np.sum(np.diff(frames, axis=0)**2, axis=-1)))
    logits = -.5*((s[0]-np.repeat([.5, 2.5, 4.5, 7.5], 2))/2)**2
    logits -= .5*((motion-np.tile([0., .7], 4))/.35)**2
    weights = np.exp(logits-logits.max()); weights /= weights.sum()
    blocks = theta.astype(float).reshape(8, 2, 2)
    blocks = blocks/np.maximum(1., np.linalg.norm(blocks, axis=(1, 2)))[:, None, None]
    return np.einsum('j,jkl->kl', weights, blocks), weights, blocks


def endpoint_ball_min_y(anchor, radius, lo, hi):
    """Exact minimum over union of free rectangles intersected with anchor ball.

    This is only a pointwise necessary condition for a full Lipschitz map.
    """
    answers = []
    for a, b in zip(lo.astype(float), hi.astype(float)):
        if np.any(a > b):
            continue
        dx = max(a[0]-anchor[0], anchor[0]-b[0], 0.)
        if dx > radius:
            continue
        y = max(a[1], anchor[1]-np.sqrt(max(radius**2-dx**2, 0)))
        if y <= b[1] and y <= anchor[1]+np.sqrt(max(radius**2-dx**2, 0)):
            answers.append(y)
    assert answers
    return min(answers)


def full_map_min_y(q, anchor, radius):
    """Exact fixed-conditioning endpoint minimum for ANY full 1-Lipschitz map.

    For this fork subset, the local free set is a top corridor joined to the
    left vertical passage. An action segment of length r maps to a free curve
    of length <=r. Conversely any such curve gives a full-domain scalar-path
    extension. This does NOT characterize one shared fitted parameter set.
    """
    xl = max(1., float(q[0])-1)
    xh = min(float(HIGH[2, 0]), float(q[0])+1)
    bottom = max(1., float(q[1])-1)
    dx = max(xl-anchor[0], anchor[0]-xh, 0.)
    if dx == 0 or anchor[1] < 3:
        assert dx == 0
        return max(bottom, anchor[1]-radius), max(0., anchor[1]-3)
    corner_distance = np.hypot(dx, anchor[1]-3)
    if radius > corner_distance:
        return max(bottom, 3-(radius-corner_distance)), corner_distance
    return max(3., anchor[1]-radius), corner_distance


def free(points, q):
    points = np.asarray(points)
    endpoint = np.any(np.all((points[..., None, :] >= LOW) &
                             (points[..., None, :] <= HIGH), axis=-1), axis=-1)
    return endpoint & np.all(np.abs(points-q) <= 1+1e-12, axis=-1)


def path_to_descent(q, anchor, y):
    """Shortest path to the chosen lowest-Y endpoint, with x nearest anchor."""
    xl = max(1., float(q[0])-1); xh = min(float(HIGH[2, 0]), float(q[0])+1)
    tx = np.clip(anchor[0], xl, xh)
    vertices = [anchor.copy()]
    if tx != anchor[0] and y < 3:
        vertices.append(np.array([tx, 3.]))
    vertices.append(np.array([tx, y]))
    return np.array(vertices)


def curve(vertices, t):
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    remaining = np.clip(np.asarray(t), 0, starts[-1])
    result = np.broadcast_to(vertices[0], remaining.shape+(2,)).copy()
    for i, length in enumerate(lengths):
        if length > 0:
            fraction = np.clip((remaining-starts[i])/length, 0, 1)
            result += fraction[..., None]*(vertices[i+1]-vertices[i])
    return result


def witness_check(row):
    """A full-domain witness using this recorded anchor and action pair.

    Direction and path are fixed constants after selecting this conditioning;
    evaluated execution actions cannot change them. This is an existence
    certificate, not an off-diagonal training label or a rollout controller.
    """
    q, anchor, d = [np.array(row[k], float) for k in ['q', 'anchor', 'action_difference']]
    radius = np.linalg.norm(d)
    y = (3+row['all_maps_min_y'])/2  # strict descent, leaves numerical margin
    vertices = path_to_descent(q, anchor, y)
    length = np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum()
    assert length < radius and y < 3
    v = d/radius
    xp = np.array(row['nominal_action'])
    axis = np.linspace(-1, 1, 41)
    actions = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    actions = np.vstack([actions, xp, row['execution_action']])
    values = curve(vertices, (actions-xp) @ v)
    assert free(values, q).all()
    np.testing.assert_allclose(curve(vertices, 0), anchor, atol=1e-12, rtol=0)
    np.testing.assert_allclose(values[-1], vertices[-1], atol=1e-12, rtol=0)
    maximum = 0.
    for i in range(0, len(actions), 100):
        da = np.linalg.norm(actions[i:i+100, None]-actions[None], axis=-1)
        dy = np.linalg.norm(values[i:i+100, None]-values[None], axis=-1)
        maximum = max(maximum, float(np.max(dy-da)))
    assert maximum < 1e-12
    return dict(seed=row['model_seed'], reset_seed=row['reset_seed'], time=row['time'],
                current=q, anchor=anchor, xp=xp, target_action=row['execution_action'],
                fixed_direction=v, path_vertices=vertices, path_length=length,
                action_distance=radius, emitted_at_target=values[-1],
                grid_action_count=len(actions), grid_max_lipschitz_excess=maximum,
                grid_is_not_the_proof=True)


def counterexamples():
    # Near the maze's re-entrant corner: nearest-point projection jumps.
    eps = 1e-5; corner_x = float(HIGH[2, 0])
    p = np.array([corner_x+.2, 2.8-eps]); q = np.array([corner_x+.2, 2.8+eps])
    def nearest(point):
        outputs = np.clip(point, LOW.astype(float), HIGH.astype(float))
        return outputs[np.argmin(np.linalg.norm(outputs-point, axis=1))]
    fp, fq = nearest(p), nearest(q)
    ratio = np.linalg.norm(fp-fq)/np.linalg.norm(p-q)
    assert ratio > 10000
    # A pointwise anchor-ball response need not be Lipschitz across actions.
    # f(t)=(1.5, 3.5 + t*sign(sin(1/t))), f(0)=anchor, t in [-.1,.1].
    # Instead use a single explicit sign switch at t=.05 for reproducible pairs.
    a, b = .05-eps, .05+eps
    out_a = np.array([1.5, 3.5+a]); out_b = np.array([1.5, 3.5-b])
    ball_ratio = np.linalg.norm(out_a-out_b)/abs(a-b)
    assert ball_ratio > 4000
    return dict(nonconvex_nearest_projection=dict(p=p, q=q, fp=fp, fq=fq, ratio=ratio),
                anchor_ball_only_discontinuous_map=dict(anchor=[1.5, 3.5],
                    domain='x=(t,u) in [-.1,.1]^2; xp=(0,0); y offset = t if t<.05 else -t',
                    a=[a, 0], b=[b, 0], fa=out_a, fb=out_b, ratio=ball_ratio))


def analyze(root):
    if root.exists():
        raise FileExistsError('Use a new output directory; preserve existing audits.')
    initial = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source_hashes = read(SOURCE/'provenance.json')['sha256']
    for p, expected in source_hashes.items():
        assert sha(p) == expected, p
    inputs = {p.as_posix(): sha(p) for p in SOURCE.iterdir() if p.is_file()}
    for p, expected in read(SOURCE/'output_hashes.json').items():
        assert sha(SOURCE/p) == expected, p
    for p in ['ett/convex_action_transition.py', 'ett/diagonal_transition.py',
              'ett/route_diagnostic.py', 'ett/run_convex_adversarial.py',
              'notes/ett_adversarial_fixed_actor_spec.md', 'crl/envs.py', __file__,
              str(MODELS/'REPORT.md'), str(MODELS/'SPEC.md')]:
        inputs[Path(p).as_posix()] = sha(p)
    rows, summaries, witnesses, examples, capacity_errors = [], {}, {}, {}, []
    errors = dict(matrix_response=0., selected_box=0., emitted_position=0.)
    for seed in [0, 1]:
        checkpoint = MODELS/'checkpoints'/f's{seed}_adversarial.npz'
        inputs[checkpoint.as_posix()] = sha(checkpoint)
        theta = np.load(checkpoint)['theta']
        z = dict(np.load(SOURCE/f'main_model_s{seed}_lower.npz'))
        xp = np.load(SOURCE/f'main_model_s{seed}_lower_auxiliary.npz')['x_prime']
        np.testing.assert_array_equal(z['states'][:, 1:, 2:], z['states'][:, :-1, :6])
        assert np.all(z['goals'] == np.tile([8.5, 3.5], 4))
        q = z['states'][:, :-1, :2]
        subset = (q[..., 0] >= 1) & (q[..., 0] < 2) & (q[..., 1] >= 3) & (q[..., 1] < 4) & (z['waypoint'] == 1)
        # Reconstruct active waypoint from visible observations, not stored flags.
        points = np.array([[1.5, 3.5], [1.5, 1.5], [7.5, 1.5], [7.5, 3.5], [8.5, 3.5]], np.float32)
        for e, episode in enumerate(z['states']):
            index = 0
            for t, state in enumerate(episode[:-1]):
                if np.linalg.norm(points[index]-state[:2]) <= .25:
                    index = min(index+1, 4)
                assert index == z['waypoint'][e, t]
                np.testing.assert_array_equal(z['action'][e, t], np.clip(points[index]-state[:2], -1, 1))
        sr = []
        for e, t in np.argwhere(subset):
            s = z['states'][e, t]; before = s[:2]; anchor = z['anchor_xy'][e, t, 0]
            action = z['action'][e, t]; nominal = xp[e, t]
            d = action.astype(float)-nominal.astype(float); radius = np.linalg.norm(d)
            matrix, weights, blocks = response_matrix(s, theta)
            correction = matrix @ d; proposal = anchor.astype(float)+correction
            lo, hi, contains, selected, areas = boxes(before, anchor)
            np.testing.assert_array_equal(lo[selected], z['box_low'][e, t, 0])
            np.testing.assert_array_equal(hi[selected], z['box_high'][e, t, 0])
            assert selected == z['rectangle'][e, t, 0]
            errors['matrix_response'] = max(errors['matrix_response'], float(np.max(abs(correction-z['response'][e, t]))))
            final = np.clip(proposal, lo[selected], hi[selected])
            errors['emitted_position'] = max(errors['emitted_position'], float(np.max(abs(final-z['states'][e, t+1, :2]))))
            current_min = max(float(lo[selected, 1]), float(anchor[1])-radius)
            ball_min = endpoint_ball_min_y(anchor.astype(float), radius, lo, hi)
            all_min, corner_cost = full_map_min_y(before, anchor.astype(float), radius)
            # Exact per-query class bound is attained by making all eight blocks
            # identical rank-one matrices -e_y d^T/||d||. This is NOT fitting.
            candidate = np.zeros((2, 2))
            if radius:
                candidate[1] = -d/radius
            analytic_emit = np.clip(anchor.astype(float)+candidate@d, lo[selected], hi[selected])
            capacity_errors.append(abs(analytic_emit[1]-current_min))
            simple = np.clip(anchor.astype(float)+np.array([0., d[1]]), lo[selected], hi[selected])
            # Fixed B_j=diag(0,1) is a single admissible 32-coefficient choice,
            # checked on all saved contexts, not optimized and not rolled out.
            row = dict(model_seed=seed, reset_seed=int(z['reset_seed'][e]), time=int(t),
                q=before, state_f4=s, execution_action=action, nominal_action=nominal,
                action_difference=d, action_distance=radius, anchor=anchor,
                anchor_displacement=anchor.astype(float)-before, matrix=matrix, gates=weights,
                correction=correction, proposal=proposal, rectangle=selected,
                box_low=lo[selected], box_high=hi[selected], containing_rectangles=np.flatnonzero(contains),
                candidate_rectangle_areas=areas, emitted=final, emitted_displacement=final-before,
                clipping_delta=final-proposal, current_class_min_y=current_min,
                endpoint_ball_min_y=ball_min, all_maps_min_y=all_min, corner_distance=corner_cost,
                analytic_y_minimizer=candidate, fixed_vertical_matrix_emitted=simple,
                upper_rectangle=selected == 0, vertical_contains_anchor=bool(contains[2]),
                anchor_down=bool(anchor[1] < before[1]), correction_down=bool(correction[1] < 0),
                proposal_down=bool(proposal[1] < before[1]), emitted_down=bool(final[1] < before[1]),
                proposal_enters_descent=bool(proposal[1] < 3), emitted_enters_descent=bool(final[1] < 3),
                clipping_raises_y=bool(final[1] > proposal[1]+TOL),
                clipping_blocks_entry=bool(proposal[1] < 3 and final[1] >= 3),
                proposal_wrong_x=bool(proposal[0] > before[0] and action[0] < 0),
                current_class_can_move_down=bool(current_min < before[1]),
                current_class_can_enter=bool(current_min < 3),
                endpoint_ball_can_enter=bool(ball_min < 3),
                full_map_can_enter=bool(all_min < 3),
                fixed_vertical_matrix_enters=bool(simple[1] < 3),
                fixed_vertical_matrix_moves_down=bool(simple[1] < before[1]))
            sr.append(row); rows.append(row)
        count_fields = [k for k, v in sr[0].items() if isinstance(v, bool)]
        summaries[str(seed)] = dict(steps=len(sr), episodes=len(set(r['reset_seed'] for r in sr)),
            counts={k: sum(r[k] for r in sr) for k in count_fields},
            rectangle_counts=np.bincount([r['rectangle'] for r in sr], minlength=4),
            mean={k: np.mean([r[k] for r in sr], axis=0) for k in
                  ['execution_action', 'nominal_action', 'anchor_displacement', 'correction',
                   'emitted_displacement', 'clipping_delta', 'action_distance']},
            upper_box_but_full_map_can_enter=sum(r['upper_rectangle'] and r['full_map_can_enter'] for r in sr),
            upper_box_anchor_in_vertical=sum(r['upper_rectangle'] and r['vertical_contains_anchor'] for r in sr),
            all_map_entry_impossible=sum(not r['full_map_can_enter'] for r in sr),
            downward_feasible_not_learned=sum(r['current_class_can_move_down'] and not r['emitted_down'] for r in sr),
            descent_feasible_not_learned=sum(r['current_class_can_enter'] and not r['emitted_enters_descent'] for r in sr),
            shared_blocks_frobenius=np.linalg.norm(blocks, axis=(1, 2)))
        # Deterministic first occurrence in lexicographic reset/time order for
        # each mechanistic category. Empty categories explicitly remain empty.
        predicates = dict(first=lambda r: True,
            upward_before_clip=lambda r: not r['proposal_down'],
            clipping_blocks_entry=lambda r: r['clipping_blocks_entry'],
            overlap_selector_exclusion=lambda r: r['upper_rectangle'] and r['vertical_contains_anchor'] and r['full_map_can_enter'],
            anchor_outside_vertical_witness=lambda r: r['upper_rectangle'] and not r['vertical_contains_anchor'] and r['full_map_can_enter'],
            feasible_current_class_entry=lambda r: r['current_class_can_enter'],
            lipschitz_obstacle=lambda r: not r['full_map_can_enter'])
        examples[str(seed)] = {name: next((r for r in sr if pred(r)), None) for name, pred in predicates.items()}
        wr = next(r for r in sr if r['upper_rectangle'] and not r['vertical_contains_anchor'] and r['full_map_can_enter'])
        witnesses[str(seed)] = witness_check(wr)
    assert max(errors.values()) < TOL, errors
    assert max(capacity_errors) < 1e-12
    root.mkdir(parents=True)
    write(root/'config.json', dict(source=SOURCE.as_posix(), model_seeds=[0, 1], L=1,
        fork_subset='1<=current X<2, 3<=current Y<4, active waypoint index=1 (lower-left)',
        index_convention='t is zero-based pre-action transition index',
        selection='first reset/time within each explicitly named category; no trajectory selection by return',
        saved_context_evaluations=len(rows), new_transition_samples=0, new_trajectory_rollouts=0,
        parameter_optimization_steps=0, tests='deterministic arithmetic, explicit candidate and full-domain witness checks'))
    write(root/'trace_summary.json', summaries)
    write(root/'fork_traces.json', rows)
    write(root/'selected_examples.json', examples)
    write(root/'full_domain_witnesses.json', witnesses)
    write(root/'counterexamples.json', counterexamples())
    scalar = [k for k, v in rows[0].items() if np.isscalar(v)]
    with (root/'fork_capacity.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=scalar); writer.writeheader()
        writer.writerows({k: r[k] for k in scalar} for r in rows)
    for p, expected in {**source_hashes, **inputs}.items():
        assert sha(p) == expected, p
    write(root/'provenance.json', dict(starting_commit=initial,
        requested_commit='9b8bacaa3f8a83f063fa776271244a9562a17919',
        branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        relevant_changes_since_requested=subprocess.check_output(['git', 'diff', '--stat', '9b8bacaa3f8a83f063fa776271244a9562a17919'], text=True),
        input_sha256=inputs, prior_manifest_verified=True, python=sys.version, numpy=np.__version__))
    write(root/'verification.json', dict(reconstruction_max_abs_errors=errors,
        analytic_min_attainment_max_error=max(capacity_errors), controller_and_F4_checked=True,
        input_hashes_unchanged=True, deterministic_witnesses_verified=True,
        new_rollouts=0, new_sampler_calls=0, training_steps=0,
        caution='Numerical witness grids check implementation only. All-action proofs are in DERIVATIONS.md.'))
    print(json.dumps(summaries, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    analyze(parser.parse_args().out_dir)
