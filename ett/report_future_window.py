"""Saved-data plots and numerical appendix for future-window exposure."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ett.run_return_readout import read

COLORS = {'initial': '#6b7280', 'A': '#2563eb', 'B': '#d97706', 'C': '#059669'}


def render(root):
    evaluation = read(root/'evaluation.json')
    diagnostic = read(root/'diagnostics.json')
    training = read(root/'training.json')['runs']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for seed in [0, 1]:
        for arm in ['B', 'C']:
            result = diagnostic['goals'][f's{seed}_{arm}']
            counts = np.asarray(result['lag_histogram'])
            axes[0, seed].plot(np.arange(1, 51), counts[1:]/counts.sum(), color=COLORS[arm],
                label=f'{arm}: '+('short' if arm=='B' else 'long'))
            samples = np.load(root/f's{seed}_{arm}_sample_audit.npz')
            valid = np.arange(26)[None] < samples['count'][:, None]
            axes[1, seed].hist(samples['goal_xy_distance'][valid], bins=np.linspace(0, 9, 46),
                              density=True, histtype='step', color=COLORS[arm], label=arm)
        axes[0, seed].set(title=f'Actual sampled future lags, seed {seed}', xlabel='Future lag j-i (steps)', ylabel='Fraction of synthetic anchors')
        axes[1, seed].set(title=f'Achieved-goal proximity, seed {seed}', xlabel='XY distance to commanded goal (maze units)', ylabel='Density (per maze unit)')
        for row in [0, 1]:
            axes[row, seed].legend()
    fig.savefig(root/'future_goal_exposure.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed, ax in enumerate(axes):
        names = ['initial']+[f's{seed}_{arm}' for arm in ['A', 'B', 'C']]
        native = [evaluation['native'][name]['returns'] for name in names]
        model = [evaluation['model'][f'{name}_model_s{seed}']['returns'] for name in names]
        ax.plot(range(4), native, 'o-', label='Native simulator')
        ax.plot(range(4), model, 's-', label='Frozen adversarial model')
        ax.set(xticks=range(4), xticklabels=['Initial', 'A offline', 'B short', 'C long'], ylabel='Mean discounted task return',
               title=f'Final deployment returns, seed {seed}')
        ax.legend()
    fig.savefig(root/'native_and_model_returns.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for seed in [0, 1]:
        for arm in ['A', 'B', 'C']:
            curves = training[f's{seed}_{arm}']['curves']
            for row, field in enumerate(['critic_loss', 'actor_loss']):
                axes[row, seed].plot([v['update'] for v in curves], [v[field] for v in curves], color=COLORS[arm], label=arm)
                axes[row, seed].set(title=f'{field.replace("_", " ")}, seed {seed}', xlabel='Additional gradient updates', ylabel='100-update mean loss')
                axes[row, seed].legend()
    fig.suptitle('Training diagnostics: B/C losses use different goal distributions', fontsize=12)
    fig.savefig(root/'learning_curves.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for seed in [0, 1]:
        for e in range(4):
            ax = axes[seed, e]
            for arm in ['initial', 'A', 'B', 'C']:
                name = arm if arm=='initial' else f's{seed}_{arm}'
                xy = np.load(root/f'{name}_native.npz')['position'][e]
                ax.plot(xy[:, 0], xy[:, 1], color=COLORS[arm], label=arm)
                ax.scatter(*xy[-1], color=COLORS[arm], s=10)
            for low, width, height in [((0, 0), 9, 1), ((0, 1), 1, 2), ((2, 2), 5, 1), ((8, 1), 1, 2)]:
                ax.add_patch(plt.Rectangle(low, width, height, color='#475569', alpha=.25))
            ax.add_patch(plt.Rectangle((3, 3), 3, 1, color='#d1d5db', alpha=.45))
            ax.add_patch(plt.Circle((8.5, 3.5), .5, fill=False, linestyle=':'))
            ax.set(xlim=(0, 9), ylim=(0, 4.1), aspect='equal', xlabel='X (maze units)', ylabel='Y (maze units)', title=f'Seed {seed}, reset {31000000+e}')
            if seed==0 and e==0:
                ax.legend(fontsize=7, loc='lower right')
    fig.savefig(root/'representative_native.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed, ax in enumerate(axes):
        values = np.load(root/f's{seed}_remaining_returns.npz')
        image = ax.hexbin(values['horizon'].ravel(), values['returns'].ravel(), gridsize=(24, 18), mincnt=1, cmap='Blues')
        ax.set(title=f'Shared complete continuations, model seed {seed}', xlabel='Remaining horizon H (steps)', ylabel='Discounted model return from context')
        fig.colorbar(image, ax=ax, label='Number of contexts')
    fig.savefig(root/'shared_continuation_returns.png', dpi=160)
    plt.close(fig)
    lines = ['# Saved numerical results', '', 'Native and model returns are separate; intervals are paired whole-episode bootstrap intervals.', '']
    for name, result in evaluation['native'].items():
        lines.append(f"- {name}: native return {result['returns']:.6f}, success {result['success']:.6f}, failure {result['failure']:.6f}, detour {result['safe_route']:.6f}, zero return {result['zero_fraction']:.6f}.")
    for source in ['native', 'model']:
        lines += ['', f'## Paired {source} differences', '']
        for name, result in evaluation[f'{source}_paired'].items():
            fields = ['returns', 'success']+(['failure'] if source=='native' else [])
            lines.append(f'- {name}: '+ '; '.join(f"{k} {result[k]['difference']:+.6f} [{result[k]['ci95'][0]:+.6f}, {result[k]['ci95'][1]:+.6f}]" for k in fields)+'.')
    lines += ['', '## Model means', '']
    for name, result in evaluation['model'].items():
        lines.append(f"- {name}: return {result['returns']:.6f}, success {result['success']:.6f}, projection {result['diagnostics']['projection_fraction']:.6f}.")
    lines += ['', '## Goal exposure', '']
    for name, result in diagnostic['goals'].items():
        lines.append(f"- {name}: mean lag {result['mean_lag']:.6f}, j>3 fraction {result['fraction_beyond_three']:.6f}, XY distance mean {result['xy_distance_mean']:.6f}, XY <0.5 fraction {result['xy_distance_below_0p5']:.6f}, XY <2 fraction {result['xy_distance_below_2']:.6f}, max lag-frequency error {result['max_abs_lag_frequency_error']:.6f}.")
    (root/'NUMERICAL_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    render(Path(parser.parse_args().run_dir))
