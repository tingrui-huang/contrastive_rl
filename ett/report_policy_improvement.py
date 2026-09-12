"""Render saved fixed-model policy results without additional interaction."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ett.run_return_readout import read


COLORS = dict(initial='#6b7280', A='#2563eb', B='#d97706', C='#059669')


def render(root):
    evaluation = read(root/'evaluation.json')
    training = read(root/'training.json')['runs']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed, ax in enumerate(axes):
        labels = ['initial']+[f's{seed}_{a}' for a in ['A', 'B', 'C']]
        x = np.arange(4)
        for offset, field, color in [(-.24, 'returns', '#2563eb'), (0, 'success', '#059669'), (.24, 'failure', '#d97706')]:
            values = [evaluation['native'][name][field] for name in labels]
            if field == 'returns':
                ax.bar(x+offset, np.array(values)/sum(.95**np.arange(50)), .23, label='Return / theoretical maximum', color=color)
            else:
                ax.bar(x+offset, values, .23, label=field.capitalize(), color=color)
        ax.set(xticks=x, xticklabels=['Initial', 'A offline', 'B control', 'C adversarial'],
               ylabel='Fraction (normalized return, success, failure)', ylim=(0, 1), title=f'Native deployment, learner seed {seed}')
        ax.legend(fontsize=8)
    fig.savefig(root/'native_outcomes.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for seed, ax in enumerate(axes):
        labels = ['initial']+[f's{seed}_{a}' for a in ['A', 'B', 'C']]
        for source, marker in [('native', 'o'), ('control', 's'), ('adversarial', '^')]:
            values = [evaluation['native'][name]['returns'] if source=='native' else
                      evaluation['model'][f'{name}_model_s{seed}_{source}']['returns'] for name in labels]
            ax.plot(np.arange(4), values, marker=marker, label=source)
        ax.set(xticks=np.arange(4), xticklabels=['Initial', 'A', 'B', 'C'], ylabel='Mean discounted task return',
               title=f'Deterministic actor: native versus model, seed {seed}')
        ax.legend()
    fig.savefig(root/'native_vs_model.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for seed in [0, 1]:
        for arm in ['A', 'B', 'C']:
            run = training[f's{seed}_{arm}']
            for j, field in enumerate(['critic_loss', 'actor_loss']):
                axes[j, seed].plot([v['update'] for v in run['curves']], [v[field] for v in run['curves']],
                                   label=arm, color=COLORS[arm])
                axes[j, seed].set(title=f'{field.replace("_", " ").title()}, seed {seed}', xlabel='Additional gradient updates', ylabel='Loss (100-update mean)')
                axes[j, seed].legend()
            if arm != 'A':
                for j, field in enumerate(['projection_fraction', 'renewed_stationary_fraction']):
                    axes[j, 2].plot([v['update'] for v in run['diagnostics']], [v[field] for v in run['diagnostics']],
                                   label=f'{arm} seed {seed}', color=COLORS[arm], linestyle='-' if seed==0 else '--')
                    axes[j, 2].set(title=field.replace('_', ' ').title(), xlabel='Update at data refresh', ylabel='Generated-step fraction')
                    axes[j, 2].legend()
    fig.savefig(root/'learning_and_generation.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for seed in [0, 1]:
        for episode in range(4):
            ax = axes[seed, episode]
            for arm in ['initial', 'A', 'B', 'C']:
                name = arm if arm == 'initial' else f's{seed}_{arm}'
                states = np.load(root/f'{name}_native.npz')['position'][episode]
                ax.plot(states[:, 0], states[:, 1], color=COLORS[arm], label=arm, linewidth=1.3)
                ax.scatter(*states[-1], color=COLORS[arm], s=12)
            ax.add_patch(plt.Rectangle((3, 3), 3, 1, facecolor='#d1d5db', alpha=.45))
            for low, width, height in [((0, 0), 9, 1), ((0, 1), 1, 2), ((2, 2), 5, 1), ((8, 1), 1, 2)]:
                ax.add_patch(plt.Rectangle(low, width, height, facecolor='#475569', alpha=.25))
            ax.add_patch(plt.Circle((8.5, 3.5), .5, fill=False, linestyle=':'))
            ax.set(xlim=(0, 9), ylim=(0, 4.1), xlabel='X (maze units)', ylabel='Y (maze units)',
                   title=f'Seed {seed}, reset {21000000+episode}', aspect='equal')
            if seed==0 and episode==0:
                ax.legend(fontsize=7, loc='lower right')
    fig.savefig(root/'representative_native.png', dpi=160)
    plt.close(fig)
    lines = ['# Numerical results', '', 'All values derive from saved outcomes. Native and model returns remain separate.', '']
    for name, result in evaluation['native'].items():
        lines.append(f"- {name}: native return {result['returns']:.6f}; success {result['success']:.6f}; failure {result['failure']:.6f}; safe-route {result['safe_route']:.6f}; zero return {result['zero_fraction']:.6f}.")
    lines += ['', '## Paired native differences', '']
    for name, result in evaluation['native_paired'].items():
        lines.append(f'- {name}: ' + '; '.join(f"{field} {result[field]['difference']:+.6f} [{result[field]['ci95'][0]:+.6f}, {result[field]['ci95'][1]:+.6f}]" for field in ['returns', 'success', 'failure'])+'.')
    lines += ['', '## Frozen-model means', '']
    for name, result in evaluation['model'].items():
        d = result['diagnostics']
        lines.append(f"- {name}: return {result['returns']:.6f}; projection {d['projection_fraction']:.6f}; boundary {d['boundary_fraction']:.6f}; sampled segment wall {d['sampled_segment_wall_fraction']:.6f}; moved atom {d['atom_moved_fraction']}; renewed stationary {d['renewed_stationary_fraction']}.")
    (root/'NUMERICAL_RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    render(Path(parser.parse_args().run_dir))
