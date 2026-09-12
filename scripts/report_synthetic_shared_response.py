"""Generate English figures and a report from fixed-budget synthetic results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.synthetic_shared_response import ARMS, sha, write_json


LABELS = dict(joint='Shared MLP: two losses', diagonal_only='Diagonal only',
              signed_difference='Shared MLP + signed difference')
COLORS = dict(joint='#1670aa', diagonal_only='#777777', signed_difference='#d36b13')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    read = lambda name: json.loads((root / name).read_text())
    config, results, history, complete = [read(n + '.json') for n in ('config', 'results', 'history', 'completion')]
    verification = read('verification.json')
    if verification['status'] != 'passed':
        raise ValueError('checkpoint verification must pass before reporting')
    if config['smoke'] or complete['status'] != 'complete':
        raise ValueError('report requires a completed primary experiment')
    seeds = config['seeds']
    selected = {arm: [results[f'{arm}_L1_s{seed}'] for seed in seeds] for arm in ARMS}
    arrays = {}
    for arm in ARMS:
        arrays[arm] = []
        for seed in seeds:
            with np.load(root / f'{arm}_L1_s{seed}_evaluation.npz') as data:
                arrays[arm].append({k: data[k] for k in data.files})
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    def finish(fig, name):
        fig.tight_layout(rect=(0, .03, 1, .94))
        fig.text(.5, .005, 'Synthetic scalar diagnostic | Final step 2500 | 3 seeds | Outputs divided by prescribed L', ha='center', fontsize=8)
        fig.savefig(root / name, dpi=160)
        plt.close(fig)

    for close, filename, title in [(False, 'responses.png', 'Learned responses at fixed held-out nominal actions'),
                                    (True, 'near_diagonal.png', 'Both sides of the diagonal: local approximation to a corner')]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for i, (ax, anchor) in enumerate(zip(axes.flat, config['slice_anchors'])):
            for arm in ARMS:
                values = np.array([a['close_over_L' if close else 'slice_over_L'][i] for a in arrays[arm]])
                x = arrays[arm][0]['close_offsets' if close else 'grid_axis']
                ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=COLORS[arm], alpha=.15)
                ax.plot(x, values.mean(axis=0), color=COLORS[arm], label=LABELS[arm])
            truth = -np.abs(x if close else x - anchor)
            ax.plot(x, truth, 'k--', linewidth=1.5, label='Analytic target')
            ax.axvline(0 if close else anchor, color='black', linewidth=.6, alpha=.3)
            ax.set(xlabel='Execution offset x - x_prime' if close else 'Execution action x',
                   ylabel='Outcome G / L', title=f'x_prime = {anchor:g}; band = seed min/max')
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(title)
        finish(fig, filename)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(axes, ['normalized_diag', 'normalized_off'], ['Diagonal monitor MSE', 'Off-diagonal monitor MSE']):
        for arm in ARMS:
            records = [history[f'{arm}_L1_s{seed}'] for seed in seeds]
            steps = [v['step'] for v in records[0]]
            values = np.array([[v[metric] for v in rec] for rec in records])
            ax.plot(steps, values.mean(axis=0), color=COLORS[arm], label=LABELS[arm])
            ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=COLORS[arm], alpha=.15)
        ax.set(xlabel='Optimizer step', ylabel='MSE / L squared', yscale='log', title=title)
    axes[0].legend(fontsize=7)
    fig.suptitle('Independent fixed monitor pairs during training; never used for selection')
    finish(fig, 'learning_curves.png')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ['<.001', '.001-.01', '.01-.1', '.1-.5', '.5-1', '1-2']
    for arm in ARMS:
        values = np.array([[b['normalized_rmse'] for b in r['strata']] for r in selected[arm]])
        ax.plot(range(6), values.mean(axis=0), 'o-', label=LABELS[arm], color=COLORS[arm])
        ax.fill_between(range(6), values.min(axis=0), values.max(axis=0), color=COLORS[arm], alpha=.15)
    ax.set(xticks=range(6), xticklabels=labels, xlabel='Absolute action difference |x - x_prime| (bins)',
           ylabel='Held-out RMSE / L', yscale='log', title='Evaluation error by distance; mean and seed min/max')
    ax.legend(fontsize=8)
    finish(fig, 'distance_errors.png')

    keys = ['anchored_learned_value', 'anchored_prescribed_c', 'arbitrary_random_pairs', 'grid_adjacent_pairs']
    labels = ['Learned\nanchor', 'Prescribed\nc=0', 'Random\naction pairs', 'Grid\nadjacent pairs']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    for index, arm in enumerate(ARMS):
        shift = (index - 1) * .23
        for ax, metric in zip(axes[:2], ['max_ratio_over_L', 'fraction_above_tolerance']):
            values = np.array([[r[k][metric] for k in keys] for r in selected[arm]])
            values *= 100 if metric == 'fraction_above_tolerance' else 1
            ax.bar(np.arange(4) + shift, values.mean(axis=0), .22, color=COLORS[arm], label=LABELS[arm])
            for seed_values in values:
                ax.scatter(np.arange(4) + shift, seed_values, s=10, color='black', zorder=3)
        bound = [r['analytical_norm_product_over_L'] for r in selected[arm]]
        axes[2].bar(index, np.mean(bound), color=COLORS[arm])
        axes[2].scatter([index] * len(bound), bound, s=12, color='black')
    for ax in axes[:2]:
        ax.set_xticks(range(4), labels, fontsize=8)
    axes[0].axhline(1.01, color='black', linestyle='--', linewidth=1)
    axes[0].set(ylabel='Maximum ratio / L', title='Empirical ratio maxima')
    axes[1].set(ylabel='Pairs above 1.01 L (%)', title='Empirical tolerance violations')
    axes[2].set(xticks=range(3), xticklabels=['Joint', 'Diag only', 'Signed diff'],
                ylabel='Norm-product bound / L', title='Analytical upper bound (loose)')
    axes[0].legend(fontsize=6)
    fig.suptitle('Anchored versus full action diagnostics | min separation 0.001 | dots = seeds')
    finish(fig, 'bound_diagnostics.png')

    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for metric in ['diagonal', 'off', 'uniform_square_off', 'grid_error']:
            values = np.array([r[metric]['normalized_rmse'] for r in selected[arm]])
            summary[arm][metric] = dict(mean=float(values.mean()), seed_std=float(values.std(ddof=1)),
                                        min=float(values.min()), max=float(values.max()), values=values.tolist())
    write_json(root / 'summary.json', summary)
    lines = ['# Shared neural response: synthetic two-loss diagnostic', '',
        '## Scope and mathematical reference', '',
        'This is an isolated scalar supervised experiment on x,x_prime in [-1,1], with c=0 and prescribed L in {0.25,1,4}. '
        'Smaller outcome is defined as worse only in this synthetic task. The known target is h(x,x_prime)=-L*abs(x-x_prime). '
        'For any response with h(x_prime,x_prime)=0 and action-Lipschitz constant at most L, h(x,x_prime)>=-L*abs(x-x_prime). '
        'The target attains this lower envelope everywhere and satisfies the full bound by the reverse triangle inequality. '
        'Its one-sided x derivatives at x=x_prime are +L and -L, so Lipschitz continuity does not require differentiability everywhere.', '',
        'The anchored comparison holds x_prime fixed and compares x with x_prime. The full condition compares any x1,x2 with the same x_prime. '
        'An anchored bound alone does not imply the full condition. The analytic target satisfies both. '
        'This construction neither identifies L from diagonal data nor identifies a real environment mechanism.', '',
        '## Existing work reviewed and isolation', '',
        'Reviewed `ett/anchored_transition.py`, `ett/rollout_return.py`, `ett/bias_ablation.py`, and the bias-ablation report at commit '
        '`eefe8d3`. Existing ETT uses an action-difference radius and an explicit equality/fallback branch; its constraints do not guarantee '
        'full action-Lipschitzness or causal validity. Its return-search experiments froze the diagonal backbone. '
        'This new diagnostic imports none of that training machinery and leaves all old results and methods unchanged. '
        'No PointMaze run, critic investigation, discriminator, failure bank, hidden signal, actor or critic training is involved.', '',
        '## Fixed design and loss scaling', '',
        'The primary network is Linear(2,32)-ReLU-Linear(32,32)-ReLU-Linear(32,1). Inputs are [x,x_prime]. '
        'There is no equality branch, diagonal label, special head, enforced anchor or action-distance output multiplier. '
        'The signed ablation adds x-x_prime as a third input, never its absolute value. '
        'It starts from the identical primary function by copying all common weights and setting only the extra input column to zero; those new weights learn normally. '
        'The primary/control have 1,185 parameters and the signed variant has 1,217.', '',
        'Write G=L*f. Physical losses are separately averaged MSE_diag=mean(G(x_prime,x_prime)^2) and '
        'MSE_off=mean((G(x,x_prime)+L*abs(x-x_prime))^2). Joint arms minimize MSE_diag+1*MSE_off. '
        'Backward uses this total divided by L squared, the same positive numerical scaling for both terms. '
        'There are exactly two terms, and every shared parameter can receive gradients from both; the diagonal-only control sets the off weight to zero. '
        'No regularization or Lipschitz loss is added. The output factor L does not force diagonal equality.', '',
        'Each of 27 runs uses 2,500 Adam steps, diagonal batch 128, off batch 256, no weight decay, and a predetermined cosine learning rate '
        'from 0.001 to 0.0001. The final step is always saved. Three seeds govern initialization, training data and minibatches. '
        'All arms within a seed share data and minibatch indices. No settings/checkpoints are selected using monitors or final grids. '
        'The three positive power-of-two L scalings produce exactly identical normalized predictions for each arm/seed, verified after reload. '
        'These are three independent seeds, not nine independent repetitions, and a steeper physical target is not a harder shape under this normalization.', '',
        'Per seed, explicitly sample 8,192 uniform diagonal examples and 32,768 off pairs. Half the off pairs have log-uniform distance '
        '[1e-4,0.1] with random sign and uniform x_prime before rejection; half are uniform independent actions conditioned on distance>=0.1. '
        'Out-of-domain proposals are rejected, never clipped. The accepted near distribution is slightly boundary-dependent. '
        'Independent evaluation uses 4,096 diagonal pairs, 16,384 balanced off pairs, an additional 16,384 uniform-square pairs, '
        '32,768 arbitrary-action triples and a 201 by 201 grid (spacing 0.01). '
        'Monitoring uses another independent fixed sample. Independent seeds, fixed anchors and all generation rules are recorded in config.json. '
        'Independent continuous draws can coincide after float32 conversion: one of 4,096 diagonal evaluation values matches a training value in seed 0, '
        'and none in seeds 1 or 2. Verification additionally reports diagonal RMSE after excluding exact coincidences; '
        'the primary seed-0 value changes from '
        f'{results["joint_L1_s0"]["diagonal"]["normalized_rmse"]:.9f} to '
        f'{verification["checks"]["joint_L1_s0"]["strictly_nonoverlapping_diagonal_rmse_over_L"]:.9f}. '
        'The training budget or model was not changed for this sensitivity check.', '',
        '## Fitting results', '',
        'All errors below are RMSE/L, mean +/- sample standard deviation over three seeds. Physical RMSE is L times these values; '
        'physical MSE is L squared times normalized MSE. Each saved result also reports native units. The known optimum has zero fitting error.', '']
    for arm in ARMS:
        values = summary[arm]
        lines.append(f'- {LABELS[arm]}: diagonal {values["diagonal"]["mean"]:.6g} +/- {values["diagonal"]["seed_std"]:.3g}; '
                     f'balanced off {values["off"]["mean"]:.6g} +/- {values["off"]["seed_std"]:.3g}; '
                     f'uniform-square off {values["uniform_square_off"]["mean"]:.6g}; dense grid {values["grid_error"]["mean"]:.6g}.')
    lines += ['', 'The unassisted shared network fits both regions approximately under this fixed budget; diagonal equality is learned, not exact. '
              'The diagonal-only control fits its observed region but leaves substantial off-diagonal error. '
              'More generally every -k*abs(x-x_prime) for arbitrary k shares the same diagonal, so diagonal data cannot determine this continuation or L.', '',
              '![Learned responses](responses.png)', '', '![Monitor learning curves](learning_curves.png)', '',
              '![Distance-stratified errors](distance_errors.png)', '', '## Feature ablation and the corner', '']
    for seed in seeds:
        a, b = results[f'joint_L1_s{seed}'], results[f'signed_difference_L1_s{seed}']
        lines.append(f'- Seed {seed}: signed minus primary diagonal RMSE/L = {b["diagonal"]["normalized_rmse"]-a["diagonal"]["normalized_rmse"]:+.6g}; '
                     f'off RMSE/L = {b["off"]["normalized_rmse"]-a["off"]["normalized_rmse"]:+.6g}.')
    lines += ['', 'The signed feature improves off-diagonal error in seeds 0 and 1 but worsens it in seed 2; '
              'its mean off RMSE is essentially unchanged, with greater seed variation. It does not give a consistent improvement here. '
              'These paired differences measure a modest feature/optimization change, including 32 additional parameters; '
              'they do not establish universal superiority of an input representation. The base network can represent the exact target: '
              '-ReLU(x-x_prime)-ReLU(x_prime-x). That construction is checked numerically but never supplied as a trained model or feature.', '']
    for arm in ['joint', 'signed_difference']:
        tiny = np.array([[r['near_diagonal']['sides'][0][side] for side in ['left_slope_over_L', 'right_slope_over_L']] for r in selected[arm]])
        far = np.array([[r['near_diagonal']['sides'][-1][side] for side in ['left_slope_over_L', 'right_slope_over_L']] for r in selected[arm]])
        lines.append(f'- {LABELS[arm]}: at offset 1e-4, mean left/right slopes divided by L = '
                     f'{tiny[:,0,:].mean():.4f}/{tiny[:,1,:].mean():.4f}; at offset 0.04 = '
                     f'{far[:,0,:].mean():.4f}/{far[:,1,:].mean():.4f}. Target slopes are +1/-1.')
    lines += ['', 'Finite ReLU networks are piecewise linear. A fitted kink need not align with the exact diagonal at every held-out x_prime. '
              'Very small offsets may fall inside one linear region on both sides; low global MSE does not ensure the exact corner or anchor. '
              'Per-anchor values, signed errors and slopes at offsets 1e-4,0.001,0.01,0.04 are saved in results.json. '
              'The 1e-4 probes are descriptive corner probes and are excluded from Lipschitz ratio compliance checks.', '',
              '![Near-diagonal close-up](near_diagonal.png)', '', '## Empirical versus analytical bound diagnostics', '',
              'All ratios use a minimum action separation of 0.001, and tolerance ratio/L <= 1.01 (absolute slope tolerance 0.01 L). '
              'The learned-anchor diagnostic is |G(x,x_prime)-G(x_prime,x_prime)|/|x-x_prime|. '
              'The prescribed-c diagnostic is |G(x,x_prime)-0|/|x-x_prime| and also exposes diagonal-anchor error. '
              'These differ when the learned diagonal is not zero. Arbitrary random pairs hold x_prime fixed and compare two execution actions. '
              'Grid adjacent ratios have the same maximum as all arbitrary pairs on that grid by telescoping; '
              'their violation fraction counts adjacent intervals, not all grid pairs. None of these samples certifies the continuum.', '']
    for arm in ARMS:
        lines.append(f'### {LABELS[arm]}')
        lines.append('')
        for key, label in zip(keys, ['Learned anchor', 'Prescribed c', 'Arbitrary random pairs', 'Grid adjacent pairs']):
            maxima = [r[key]['max_ratio_over_L'] for r in selected[arm]]
            fraction = [100 * r[key]['fraction_above_tolerance'] for r in selected[arm]]
            lines.append(f'- {label}: maxima/L by seed {[round(v,5) for v in maxima]}; above-tolerance percentages {[round(v,3) for v in fraction]}.')
        bounds = [r['analytical_norm_product_over_L'] for r in selected[arm]]
        lines.append(f'- Analytical norm-product upper bounds/L by seed: {[round(v,4) for v in bounds]}.')
        lines.append('')
    lines += ['Both joint-training arms violate the prescribed full bound even with the 1% tolerance in every seed. '
              'The primary model reaches sampled arbitrary-pair ratios of 1.18-1.45 L and grid maxima of 1.20-1.46 L. '
              'Thus the answer to empirical bound compliance is no, despite good fitting. '
              'The larger prescribed-c ratios near the diagonal also reflect nonzero anchor errors, not just excessive slopes.', '',
              'The valid analytical global bound is L*||W3||_2*||W2||_2*||W1 v||_2, where v=(1,0) for the primary model '
              'and v=(1,0,1) for the signed feature. It follows from the 1-Lipschitz property of ReLU and holds with x_prime fixed. '
              'Biases do not enter it. Reported numbers are float64 norm evaluations, not outward-rounded numerical certificates. '
              'This bound can be much larger than the actual maximum slope because it ignores mutually incompatible activation patterns. '
              'Even the exact two-ReLU reference has norm product 2L although its true constant is L. '
              'No normalization or constraint makes our learned norm products <=L by construction.', '',
              'Good target fitting is therefore distinct from empirical bound compliance and from enforcement. '
              'Inspect violations rather than interpreting a small MSE as proof of a bound; a low-slope diagonal-only model can comply while fitting the wrong response. '
              'Outcomes below the synthetic lower envelope indicate approximation/constraint violations, not discovery of a more valid pessimistic optimum. '
              'Their frequency (using outcome tolerance 0.01 L) and maximum shortfall are saved separately.', '',
              '![Bound diagnostics](bound_diagnostics.png)', '', '## Conclusion and limits before PointMaze', '',
              'This bounded experiment supports representability and approximate joint learning of diagonal observations and a deliberately supplied compatible off-diagonal target '
              'with a single ordinary shared network. It gives no reason to require a discriminator for this example. '
              'It does not solve the choice or discovery of a pessimistic target: the correct off-diagonal labels were given by construction. '
              'Prescribed L is not a causal constant inferred from observational data.', '',
              'Before applying the idea to PointMaze, one still needs a justified objective without known off-diagonal labels, '
              'a choice of distributional notion of action regularity for stochastic transitions, a treatment of multimodality, '
              'joint optimization under hard rewards and discrete sampling, and evidence against rollout drift, stopping shortcuts and broken history/geometry. '
              'This scalar fit establishes neither physical validity nor true ETT, failure-bank validity, nor worst-case Q recovery. '
              'Proceed only as a positive supervised representation diagnostic; it does not justify launching alternating PointMaze actor/transition training.', '',
              '## Reproduction and completed checks', '',
              f'The 27-run CPU experiment took {complete["elapsed_seconds"]:.1f} seconds. '
              'Six numerical tests cover the analytic target, the distinction between anchored and full bounds, bounded/explicit sampling, matched initialization without forced diagonal equality, '
              'additive shared gradients/common scaling, and the exact ReLU reference with diagnostic calibration. '
              'A separate five-step smoke was run first. Checkpoint verification reloads every final model, exactly reproduces saved diagonal/off predictions, '
              'checks separate loss arithmetic, action bounds, final-step selection and normalized equality across L. '
              'Config, metrics, evaluation arrays and English figures/report are publishable; .pt checkpoints stay local and ignored.', '',
              '```bash', 'python -m scripts.test_synthetic_shared_response',
              'python -m scripts.synthetic_shared_response --out-dir artifacts/synthetic_shared_response/smoke --smoke',
              'python -m scripts.synthetic_shared_response --out-dir artifacts/synthetic_shared_response/l025_1_4_s012',
              'python -m scripts.check_synthetic_shared_response --run-dir artifacts/synthetic_shared_response/l025_1_4_s012',
              'python -m scripts.report_synthetic_shared_response --run-dir artifacts/synthetic_shared_response/l025_1_4_s012',
              '```', '', 'Use a fresh output directory; existing runs are never overwritten. '
              'This standalone experiment needs Python, NumPy, PyTorch and Matplotlib; training versions are recorded in config.json and the plotting version in report_provenance.json '
              '(it does not require the legacy PointMaze environment). Figures show normalized L=1 results because all three L arms coincide after normalization.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(root / 'report_provenance.json', dict(source_sha256=sha(__file__), matplotlib_version=matplotlib.__version__,
               inputs={n: sha(root / n) for n in ['config.json', 'results.json', 'history.json', 'completion.json', 'verification.json']},
               uses_saved_results_only=True))
    print('Wrote five figures, summary.json and REPORT.md.')


if __name__ == '__main__':
    main()
