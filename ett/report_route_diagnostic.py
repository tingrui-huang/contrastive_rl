"""Saved-data-only statistics and figures; generates zero new transitions."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ett.diagonal_transition import POINTMAZE_WALLS
from ett.route_diagnostic import ROUTES, validate
from ett.run_route_diagnostic import CONFIG, read, write, sha, verify_hashes

ENVIRONMENTS = ['native', 'model_s0', 'model_s1']
LABELS = ['Native simulator', 'Frozen ETT seed 0', 'Frozen ETT seed 1']
COLORS = {'shortcut': '#2864A0', 'lower': '#BE651F'}


def csv_file(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def render(root):
    verify_hashes(root)
    assert read(root/'verification.json')['total_transitions'] == 40200
    bootstrap = np.random.default_rng(CONFIG['bootstrap_seed']).integers(128, size=(10000, 128))
    records, results, pairs, episodes, differences, diagnostics, audit_rows = {}, {}, {}, [], [], {}, []
    for environment in ENVIRONMENTS:
        for route in ROUTES:
            tag = f'{environment}_{route}'
            record = dict(np.load(root/f'main_{tag}.npz'))
            checked = validate(record, route)
            for key in ['success', 'lower_passage', 'shortcut_passage', 'waypoint_completed', 'route_complete']:
                np.testing.assert_array_equal(checked[key], record[key])
            records[tag] = record
            passage = record['lower_passage' if route == 'lower' else 'shortcut_passage']
            means = record['return'][bootstrap].mean(1)
            summary = dict(mean_return=record['return'].mean(), mean_return_ci95=np.quantile(means, [.025, .975]),
                success_count=record['success'].sum(), success_rate=record['success'].mean(),
                passage_adherence_count=passage.sum(), passage_adherence_rate=passage.mean(),
                lower_passage_count=record['lower_passage'].sum(), lower_passage_rate=record['lower_passage'].mean(),
                shortcut_passage_count=record['shortcut_passage'].sum(), shortcut_passage_rate=record['shortcut_passage'].mean(),
                waypoint_completion_counts=record['waypoint_completed'].sum(0),
                waypoint_completion_rates=record['waypoint_completed'].mean(0),
                full_waypoint_completion_count=record['route_complete'].sum(),
                full_waypoint_completion_rate=record['route_complete'].mean(),
                zero_return_count=(record['return'] == 0).sum(),
                final_target_counts=np.bincount(record['waypoint'][:, -1], minlength=len(ROUTES[route])))
            if environment == 'native':
                audit = dict(np.load(root/f'main_{tag}_native_audit.npz'))
                summary['native_failure_count'] = audit['failure'][:, -1].sum()
                summary['native_failure_rate'] = audit['failure'][:, -1].mean()
                # Absorption is confirmed only from native evaluation labels.
                for e, failed in enumerate(audit['failure']):
                    if failed.any():
                        first = np.flatnonzero(failed)[0]
                        np.testing.assert_array_equal(audit['true_xy'][e, first+1:],
                            np.broadcast_to(audit['true_xy'][e, first+1], audit['true_xy'][e, first+1:].shape))
                        assert not record['reward'][e, first:].any()
                    audit_rows.append(dict(reset_seed=int(record['reset_seed'][e]), controller=route,
                        failure=bool(failed[-1]), first_failure_step=int(np.flatnonzero(failed)[0]+1) if failed.any() else -1))
            else:
                xy = record['states'][..., :2]
                delta = np.diff(xy, axis=1)
                old = record['states'][:, :-1].reshape(128, 50, 4, 2)
                stationary = np.all(old == old[:, :, :1], axis=(-1, -2))
                moving = np.linalg.norm(delta, axis=-1) > 1e-5
                atoms = record['stationary_atom'][:, :, 0]
                moved_anchor = np.linalg.norm(xy[:, 1:] - record['anchor_xy'][:, :, 0], axis=-1) > 1e-5
                segment = xy[:, :-1, None] + np.linspace(0, 1, 21)[None, None, :, None]*delta[:, :, None]
                cells = np.clip(np.floor(segment).astype(int), [0, 0], [8, 4])
                crosses = (POINTMAZE_WALLS[cells[..., 0], cells[..., 1]] != 0).any(-1)
                seeking_descent = (record['waypoint'] == 1) if route == 'lower' else np.zeros((128, 50), bool)
                diagnostics[tag] = dict(outer_projection_rate=record['projection_corrected'].mean(),
                    diagonal_base_correction_rate=record['base_corrected'].mean(),
                    box_boundary_rate=record['projected_to_boundary'].mean(),
                    valid_box_rate=record['box_valid'].mean(),
                    sampled_straight_segment_wall_crossing_rate=crosses.mean(),
                    stationary_history_count=stationary.sum(),
                    stationary_history_resumed_count=(stationary & moving).sum(),
                    stationary_atom_count=atoms.sum(), moved_anchor_atom_count=(atoms & moved_anchor).sum(),
                    final_mean_distance_to_current_target=np.mean(np.linalg.norm(
                        ROUTES[route][record['waypoint'][:, -1]] - xy[:, -1], axis=-1)),
                    descent_target_steps=seeking_descent.sum(),
                    descent_target_steps_right_of_fork=(seeking_descent & (xy[:, :-1, 0] >= 2)).sum())
            results[tag] = summary
            for e, seed in enumerate(record['reset_seed']):
                episodes.append(dict(environment=environment, controller=route, reset_seed=int(seed),
                    discounted_return=float(record['return'][e]), success=bool(record['success'][e]),
                    passage_adherence=bool(passage[e]), lower_passage=bool(record['lower_passage'][e]),
                    shortcut_passage=bool(record['shortcut_passage'][e]),
                    waypoint_completion=''.join('1' if x else '0' for x in record['waypoint_completed'][e]),
                    all_waypoints_complete=bool(record['route_complete'][e]),
                    final_target_index=int(record['waypoint'][e, -1])))
        delta = records[f'{environment}_lower']['return'] - records[f'{environment}_shortcut']['return']
        pairs[environment] = dict(delta=delta.mean(), ci95=np.quantile(delta[bootstrap].mean(1), [.025, .975]),
                                  positive_count=(delta > 0).sum(), negative_count=(delta < 0).sum(), equal_count=(delta == 0).sum())
        differences.extend(dict(environment=environment, reset_seed=seed, lower_minus_shortcut=float(d))
                           for seed, d in zip(CONFIG['reset_seeds'], delta))
    gaps = {}
    for environment in ENVIRONMENTS[1:]:
        for route in ROUTES:
            tag = f'{environment}_{route}'
            diff = records[tag]['return'] - records[f'native_{route}']['return']
            gaps[tag] = dict(mean_model_minus_native=diff.mean(), ci95=np.quantile(diff[bootstrap].mean(1), [.025, .975]))
    csv_file(root/'episodes.csv', episodes)
    csv_file(root/'paired_differences.csv', differences)
    csv_file(root/'native_failure_evaluation_only.csv', audit_rows)
    write(root/'results.json', dict(controllers=results, paired=pairs, model_minus_native=gaps,
        uncertainty='10000 paired episode percentile bootstraps; unadjusted 95% intervals conditional on each checkpoint; no pooled checkpoint replicates'))
    write(root/'model_diagnostics.json', diagnostics)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for col, (environment, label) in enumerate(zip(ENVIRONMENTS, LABELS)):
        ax = axes[0, col]
        for x, route in enumerate(ROUTES):
            s = results[f'{environment}_{route}']
            ax.bar(x, s['mean_return'], color=COLORS[route], width=.55)
            lo, hi = s['mean_return_ci95']
            ax.errorbar(x, s['mean_return'], yerr=[[s['mean_return']-lo], [hi-s['mean_return']]], color='black', capsize=4)
            ax.text(x, hi+.25, f"{s['mean_return']:.3f}", ha='center')
        ax.set(title=label, xticks=[0, 1], xticklabels=['Shortcut', 'Lower controller'],
               xlabel='Fixed controller', ylabel='Mean discounted task return', ylim=(0, 15))
        p = pairs[environment]; lo, hi = p['ci95']; ax = axes[1, col]
        ax.axhline(0, color='gray', linewidth=1)
        ax.errorbar(0, p['delta'], yerr=[[p['delta']-lo], [hi-p['delta']]], fmt='o', color='black', capsize=6)
        ax.set(xticks=[0], xticklabels=['Lower minus shortcut'], xlabel='Paired controller contrast',
               ylabel='Mean return difference (task reward units)', ylim=(-4, 9))
        ax.set_title(f"Delta {p['delta']:+.3f}; 95% CI [{lo:+.3f}, {hi:+.3f}]", fontsize=10)
    fig.suptitle('Fixed route controllers: native advantage, model execution limitations', fontsize=14)
    fig.supxlabel('128 resets, 50 steps, gamma .95; 10,000 episode bootstrap resamples. Model checkpoints shown separately.', fontsize=9)
    fig.savefig(root/'return_comparison.png', dpi=170); plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(16, 8), constrained_layout=True)
    for row, (environment, label) in enumerate(zip(ENVIRONMENTS, LABELS)):
        for e, seed in enumerate(CONFIG['representative_seeds']):
            ax = axes[row, e]
            for x in range(9):
                for y in range(5):
                    if POINTMAZE_WALLS[x, y]:
                        ax.add_patch(Rectangle((x, y), 1, 1, facecolor='#D6D8DC', edgecolor='white'))
            for route in ROUTES:
                r = records[f'{environment}_{route}']; xy = r['states'][e, :, :2]
                ax.plot(xy[:, 0], xy[:, 1], '.-', markersize=2, linewidth=1.3, color=COLORS[route],
                    label=f"{route}: J={r['return'][e]:.2f}, WP={r['waypoint_completed'][e].sum()}/{len(ROUTES[route])}")
                ax.scatter(*xy[-1], marker='x', color=COLORS[route], s=30)
            ax.scatter(8.5, 3.5, marker='*', s=65, color='black')
            ax.set(title=f'{label} | reset {seed}', xlim=(0, 9), ylim=(0, 5),
                   aspect='equal', xlabel='Visible X (maze units)', ylabel='Visible Y (maze units)')
            ax.legend(fontsize=7, loc='lower center')
    fig.suptitle('First four predeclared reset seeds; lines connect visible macro-step endpoints', fontsize=13)
    fig.savefig(root/'representative_trajectories.png', dpi=170); plt.close(fig)
    write(root/'output_hashes.json', {p.name: sha(p) for p in root.iterdir() if p.is_file() and p.name != 'output_hashes.json'})
    print(json_text(pairs))
    print(json_text(gaps))


def json_text(value):
    import json
    return json.dumps(value, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    render(parser.parse_args().run_dir)
