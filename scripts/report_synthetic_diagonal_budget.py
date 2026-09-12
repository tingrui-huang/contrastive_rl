"""English comparison report and reproducible figures for the scalar budget run."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.synthetic_lipschitz_response import read
from scripts.synthetic_shared_response import sha, write_json


GROUPS = [('prior_diagonal_only', 'Diagonal only', '#777777'),
          ('prior_bank_joint', 'Lambda = 0.1', '#c87519'),
          ('budget', 'Lambda = 0.004', '#1679aa')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    config, results = read(root/'config.json'), read(root/'results.json')
    history = read(root/'history.json')
    assert not config['smoke'] and read(root/'verification.json')['status']=='passed'
    arrays = {}
    for name in results:
        with np.load(root/f'{name}_evaluation.npz') as saved:
            arrays[name] = {k: saved[k] for k in saved.files}
    summary = {}
    for prefix, label, _ in GROUPS:
        rows = [results[f'{prefix}_s{s}'] for s in range(3)]
        summary[prefix] = {'label': label}
        for metric, values in [
            ('diagonal_rmse', [r['diagonal']['rmse'] for r in rows]),
            ('continuum_max_abs_diagonal', [r['continuum_diagonal']['max_abs'] for r in rows]),
            ('relative_response_rmse', [r['relative_response_error']['rmse'] for r in rows]),
            ('failure_mse', [r['objective']['failure_mse'] for r in rows]),
            ('diagonal_reference_rmse', [r['diagonal_reference_error']['rmse'] for r in rows])]:
            summary[prefix][metric] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), by_seed=values)
    write_json(root/'summary.json', summary)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), constrained_layout=True)
    for seed in range(3):
        for prefix, label, color in GROUPS:
            a = arrays[f'{prefix}_s{seed}']
            axes[0, seed].plot(a['diagonal_breakpoints'], a['diagonal_breakpoint_values'], label=label, color=color)
        a = arrays[f'budget_s{seed}']
        ax = axes[1, seed]
        ax.axhspan(-.01, .01, color='#d9e9d9', alpha=.5, label='Illustrative budget')
        ax.plot(a['diagonal_breakpoints'], a['diagonal_breakpoint_values'], color=GROUPS[2][2], label='Learned, exact breakpoints')
        ax.plot(a['grid_axis'], a['grid_diagonal_reference'], 'k--', label='Population optimum')
        ax.set_ylim(-.017, .003)
        for row in range(2):
            axes[row, seed].set(xlabel='Nominal action x_prime', ylabel='Diagonal outcome', title=f'Seed {seed}' + (' | budget detail' if row else ' | matched arms'))
            axes[row, seed].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.savefig(root/'diagonal_budget.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, (prefix, label, color) in enumerate(GROUPS):
        for seed in range(3):
            h = history[f'{prefix}_s{seed}']
            steps = [v['step'] for v in h]
            axes[0, col].semilogy(steps, [v['diagonal_mse'] for v in h], label=f'Seed {seed}')
            axes[1, col].plot(steps, [v['failure_mse'] for v in h], label=f'Seed {seed}')
        for row in range(2):
            axes[row, col].set(xlabel='Adam step', title=label)
            axes[row, col].grid(alpha=.2)
        axes[0, col].set_ylabel('Monitor diagonal MSE')
        axes[1, col].set_ylabel('Monitor failure MSE (unweighted)')
        axes[1, col].set_ylim(4.4, 10.3)
    axes[0, 0].legend()
    fig.savefig(root/'loss_components.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), constrained_layout=True)
    for seed in range(3):
        ax = axes[0, seed]
        for prefix, label, color in GROUPS[1:]:
            a = arrays[f'{prefix}_s{seed}']
            for anchor, style in [(.74, '-'), (.95, '--')]:
                idx = int(np.argmin(np.abs(a['grid_axis']-anchor)))
                curve = a['grid'][idx]-a['grid_diagonal'][idx]
                ax.plot(a['grid_axis'], curve, linestyle=style, color=color, label=f'{label}; xp={a["grid_axis"][idx]:.2f}')
        a = arrays[f'budget_s{seed}']
        for anchor, style in [(.74, '-'), (.95, '--')]:
            ax.plot(a['grid_axis'], -np.abs(a['grid_axis']-anchor), color='black', linestyle=style,
                    alpha=.5, label=f'Reference; xp={anchor:.2f}')
        ax.set(title=f'Seed {seed} | relative response', xlabel='Execution action x', ylabel='G(x,xp) - G(xp,xp)')
        idx = int(np.argmin(np.abs(a['grid_axis']-.95)))
        slopes = a['interval_slopes'][idx]
        raw = a['raw_slopes'][idx]
        feasible = a['interval_feasible'][idx]
        j = np.arange(16)
        axes[1, seed].plot(j[feasible], raw[feasible], 'o-', label='Raw coefficient')
        axes[1, seed].plot(j[feasible], slopes[feasible], 's-', label='Clamped slope')
        axes[1, seed].plot(j[feasible], np.where(j[feasible]<8, 1., -1.), 'k--', label='Relative optimum')
        axes[1, seed].set(title=f'Seed {seed} | xp=0.95 feasible intervals', xlabel='Signed-action interval index', ylabel='Action slope / raw coefficient')
        for row in range(2):
            axes[row, seed].grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=8)
    fig.savefig(root/'relative_response_and_slopes.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for prefix, label, color in GROUPS:
        region_values = np.array([[results[f'{prefix}_s{s}']['regions'][r]['relative_response']['rmse'] for r in ['left_boundary', 'central', 'right_boundary']] for s in range(3)])
        axes[0].errorbar(range(3), np.maximum(region_values.mean(0), 1e-16), yerr=region_values.std(0, ddof=1), color=color, marker='o', label=label)
    axes[0].set(xticks=range(3), xticklabels=['Left edge', 'Central', 'Right edge'], ylabel='Relative-response RMSE', title='Nominal-action regions')
    axes[0].legend(fontsize=8)
    for prefix, label, color in GROUPS[1:]:
        values = np.array([[v['rmse'] for v in results[f'{prefix}_s{s}']['distance_strata']] for s in range(3)])
        axes[1].plot(range(6), values.mean(0), 'o-', color=color, label=label)
    axes[1].set(xticks=range(6), xticklabels=['<.001', '<.01', '<.1', '<.5', '<1', '1-2'], xlabel='Action-distance bin', ylabel='Relative-response RMSE', title='Joint arms: distance strata')
    axes[1].legend(fontsize=8)
    for seed in range(3):
        v = results[f'budget_s{seed}']
        maximum = [v['diagonal']['max_abs'], v['grid_diagonal']['max_abs'], v['continuum_diagonal']['max_abs']]
        axes[2].plot(range(3), maximum, 'o-', label=f'Seed {seed}')
    axes[2].axhline(.01, color='black', linestyle='--', label='Epsilon')
    axes[2].set(xticks=range(3), xticklabels=['Random probes', 'Grid', 'All breakpoints'], ylabel='Maximum absolute diagonal', title='Probes versus exhaustive extrema')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root/'regions_and_maxima.png', dpi=160)
    plt.close(fig)

    lines = ['# Predetermined diagonal-error budget: lambda = 0.004', '',
        'Reducing the weight greatly improves diagonal fidelity and preserves action-dependent pessimism under the unchanged training budget. However, none of the three trained models meets epsilon=0.01 over [-1,1]. This is a partial success, not a certified-budget result. The global population optimum meets the budget only with very small slack; the learned diagonal still has residual fitting error.', '',
        '## Setup, derivation and provenance', '',
        'The existing 1,681-parameter ConditionalSpline and its structural full action-Lipschitz bound L=1 are unchanged. Inputs x and x_prime are independent uniform scalar actions on [-1,1], c=0 and the detached synthetic failure bank is {-3}. Both terms use the same nominal marginal. In each training dataset, the 8,192 diagonal contexts occur exactly four times among the 32,768 off pairs; minibatches independently sample the two terms. Only x is redrawn on an exact equality tie, so no boundary rejection changes the context population.', '',
        '`L = mean(G(x_prime,x_prime)^2) + lambda * mean((G(x,x_prime)+3)^2)`', '',
        'There are exactly two losses. No correct off-diagonal next-state labels, analytic training targets, hidden inputs, discriminator, PointMaze rollout, actor, critic or additional penalty is used. All shared parameters remain trainable. Hard-clamped coefficients may receive zero direct gradient locally. The final emitted function is evaluated directly, with no recentering or candidate-selection wrapper.', '',
        'The derivation in [REFERENCE.md](REFERENCE.md) was saved before the first update. Projection onto [-3,infinity) reduces the problem to d>=-3. At fixed d, max(-3,d-|x-x_prime|) minimizes bank distance under the full bound. The conditional objective is strictly convex in d. Its stationary solution is d*=-lambda/(1+lambda)*(5-x_prime^2)/2. For 0<=lambda<1, the minimum margin above the bank is (1-lambda)/(1+lambda)>0, so this solution lies in the assumed branch everywhere and is globally optimal. For positive lambda, G*=d*-|x-x_prime|. At lambda=0, the diagonal-only objective does not identify the relative action response.', '',
        'Epsilon=0.01 and lambda=0.004 were predetermined. The optimum has maximum |d*|=0.00996015936, RMSE=0.00931510115, failure MSE=5.62319434083 and objective=0.0225795484728. Its budget slack is just 0.00003984064. These are native scalar outcome units, not a justified PointMaze tolerance.', '',
        'Exactly three new final-step runs use seeds 0/1/2, 2,500 Adam updates, cosine learning rate 0.001 to 0.0001, diagonal/off batches 128/256, float64 and one CPU thread. The unchanged historical train() routine starts each seed from its original random initialization. Initialization, training data and every minibatch index match the lambda=0.1 runs by SHA256. Only the loss weight changes. Training does not start from an old checkpoint. Old diagonal-only and lambda=0.1 results come from the latest failure-bank experiment at cbb8e47638b74a0b8ab5aa24134743798a329bd0, not the earlier supervised experiment. All six old checkpoints, their full saved metrics/arrays, source hashes and configurations were verified before new training.', '',
        'Evaluation reuses the already-inspected 4,096 nominal contexts, 16,384 off pairs, 32,768 arbitrary-action triples, 201-by-201 grid and fixed slices. These are not newly untouched test sets. They were never used to select a checkpoint or adjust settings. The monitor stream also remains unchanged. Old files and artifacts are hash-verified unchanged.', '',
        '## Diagonal fidelity and budget', '']
    for prefix, label, _ in GROUPS:
        s = summary[prefix]
        lines.append(f'- {label}: diagonal RMSE {s["diagonal_rmse"]["mean"]:.8f} +/- {s["diagonal_rmse"]["sample_std"]:.3g}; continuum maxima by seed {[round(v, 9) for v in s["continuum_max_abs_diagonal"]["by_seed"]]}; relative-response RMSE {s["relative_response_rmse"]["mean"]:.8f} +/- {s["relative_response_rmse"]["sample_std"]:.3g}; failure MSE {s["failure_mse"]["mean"]:.7f} +/- {s["failure_mse"]["sample_std"]:.3g}.')
    lines += ['', 'Uncertainty above is sample standard deviation across three matched seeds, not a confidence interval. The new mean diagonal RMSE improves by about 95.7% relative to lambda=0.1, but remains above the diagonal-only control. RMSE below epsilon is not a maximum-error guarantee.', '']
    for seed in range(3):
        r = results[f'budget_s{seed}']; e = r['continuum_diagonal']
        lines.append(f'- Seed {seed}: empirical maximum {r["diagonal"]["max_abs"]:.9f}; exhaustive maximum {e["max_abs"]:.9f} at x_prime={e["argmax"]:.9f}; random-probe fraction exceeding epsilon {100*r["diagonal_fraction_exceeding_epsilon"]:.4f}%; uniform continuum fraction {100*e["uniform_fraction_exceeding_epsilon"]:.4f}%; diagonal RMSE to d* {r["diagonal_reference_error"]["rmse"]:.9f}.')
    lines += ['', 'The continuum calculation propagates affine pieces through every conditioner Linear/ReLU layer, splits them at the relevant coefficient clamp crossings, and evaluates every resulting diagonal breakpoint and both domain endpoints. The maximum of the absolute affine value on each piece occurs at an endpoint. Epsilon-crossing lengths also give the uniform continuum exceedance fraction. This exhaustive piecewise-linear algorithm is exact in real arithmetic, implemented in float64; it is not a formal outward-rounded numerical certificate. Endpoint reconstruction discrepancies are below 7e-16 for the new models, much smaller than their budget excesses. Finite grid or random maxima are labeled separately. Seed 0 peaks inside the domain; seeds 1/2 peak at x_prime=1.', '',
        '![Diagonal curves and budget](diagonal_budget.png)', '', '## Action-dependent response and objective components', '',
        'Relative response is h=G(x,x_prime)-G(x_prime,x_prime), evaluated against -|x-x_prime|. Its mean RMSE is essentially retained at the scale of the previous joint arm, and far below the uncontrolled diagonal-only response. This comparison is descriptive across three seeds; it is not a statistical equivalence claim. The absolute response is not merely shifted downward. Expected failure distance increases when diagonal movement is penalized more strongly, as the new optimum predicts.', '']
    for seed in range(3):
        r = results[f'budget_s{seed}']; q = r['quadrature256']; o = r['objective']
        lines.append(f'- Seed {seed}: empirical diagonal MSE {o["diagonal_mse"]:.10g}, unweighted failure MSE {o["failure_mse"]:.10g}, weighted failure {o["weighted_failure"]:.10g}; own total {o["total"]:.12g}. Integrated diagonal/failure components {q["diagonal_mse"]:.10g}/{q["failure_mse"]:.10g}; own integrated total {q["total"]:.12g}, gap to the lambda=0.004 optimum {q["population_objective_gap"]:.9g}; 128/256-node difference {r["quadrature_total_difference"]:.3g}.')
    lines += ['', 'For reference, the reused lambda=0.1 joint gaps to its own optimum 0.51696969697 are ' + ', '.join(f'{results[f"prior_bank_joint_s{s}"]["quadrature256"]["population_objective_gap"]:.9g}' for s in range(3)) + '. The diagonal-only control uses weight zero, optimum zero; its objective gap is simply its diagonal MSE. Do not rank raw total values, or infer a better task from smaller totals, across these different weights. All own-weight totals, components and reference gaps are saved in results.json.', '',
        'The integration is exact in x within each spline interval and uses numerical Gauss-Legendre integration over x_prime. The 128/256 comparison is a convergence check, not a rigorous error bound. Paired empirical gaps use the same finite data for model and reference and are reported separately from population quadrature gaps.', '',
        'The finite ReLU conditioner cannot equal a quadratic diagonal everywhere. Nevertheless, an independently constructed model of the same architecture interpolates d* at 33 knots with objective gap only 2.0264e-12 for lambda=0.004 (versus 1.1560e-9 for lambda=0.1). This brackets the architecture optimum tightly; it is evaluation only. Actual gaps of 1.42e-5 to 2.26e-5 are far larger than that approximation bound and the observed quadrature differences. The small nonzero optimal diagonal is the intended penalty tradeoff. The additional learned error and epsilon violations are not required by the population optimum; finite-sample and unresolved optimization effects remain.', '',
        '![Loss components](loss_components.png)', '', '## Boundaries, learning speed and clamp saturation', '']
    for seed in range(3):
        r = results[f'budget_s{seed}']
        region = r['regions']
        lines.append(f'- Seed {seed}: relative RMSE at left/central/right nominal regions ' + '/'.join(f'{region[k]["relative_response"]["rmse"]:.6g}' for k in ['left_boundary', 'central', 'right_boundary']) + f'; distance >=1 relative RMSE {r["distance_strata"][-1]["rmse"]:.7f}; maximum individual relative error {r["relative_response_error"]["max_abs"]:.6f}; wrong-direction saturated coefficients {r["wrong_saturation"]["count"]}/{r["wrong_saturation"]["feasible_intervals"]} feasible audited intervals; unsaturated fraction {100*r["slope_diagnosis"]["feasible_unsaturated_fraction"]:.3f}%.')
    lines += ['', 'All three new models fit the near-diagonal relative response essentially to float64 precision through action distances below 0.5. Most action error lies at large distances and the right nominal boundary, as in the old experiment. The small average error therefore does not imply uniformly accurate action responses; individual errors remain as large as 0.46. Region-specific diagonal and relative errors, raw coefficients and feasible masks are saved for review. The displayed x_prime=0.95 slice uses an existing grid context to expose this boundary behavior.', '',
        'The lower weight weakens the bank term by a factor of 25, although Adam and shared parameter gradients prevent interpreting this as a literal factor-25 learning-speed change. Monitor failure loss falls substantially within the first 500-1000 updates. Seed 2 is visibly slower early (failure MSE 5.782 at step 500, 5.685 at step 1000), but the final relative response does not collapse. The final monitor components are nearly flat, with small diagonal fluctuations; monitor sampling means need not equal evaluation or population means. No action-response checkpoints were saved during training, so these component curves do not establish a detailed temporal mechanism.', '',
        'Seeds 0/2 retain 25 wrong-direction saturated coefficients at the audited contexts, as did all three old joint runs. A raw coefficient beyond the wrong clamp endpoint has zero direct gradient through that clamp. Shared-conditioner gradients can still change it. Seed 1 has no wrongly saturated coefficient but still has wrong-sign unsaturated slopes and boundary error, so dead clamp gradients alone cannot explain all residuals. The current records do not isolate stochastic minibatch noise, conditioning or shared-network interference. No extra iterations, new optimizer, architecture change, setting or sweep was run.', '',
        '![Relative response and raw slopes](relative_response_and_slopes.png)', '',
        'For each fixed nominal action, bounded slopes in [-1,1] and continuous interval joins imply the full arbitrary-execution-action bound by integration. This holds structurally for every conditioner output, not just sampled contexts. Direct interval-slope and arbitrary-action numerical audits pass. It constrains sensitivity in x; it does not constrain diagonal displacement in x_prime or guarantee the epsilon budget.', '']
    for seed in range(3):
        r = results[f'budget_s{seed}']
        excess = max(r['bounds'][k]['largest_positive_absolute_excess'] for k in ['learned_anchor', 'arbitrary_pairs', 'grid_adjacent'])
        lines.append(f'- Seed {seed}: maximum exact audited interval slope {r["intervals"]["max_exact_interval_slope"]:.1f}; largest positive full-action excess {excess:.3g}, below the unchanged 1e-12 tolerance.')
    lines += ['', 'Comparisons with the prescribed zero anchor are retained as diagnostics; they are not tests of the full action bound when the actual learned diagonal differs from zero. No physical-validity guarantee or real-environment Lipschitz constant is inferred.', '',
        '![Region errors and continuum maxima](regions_and_maxima.png)', '', '## Decision and limitations', '',
        '1. Diagonal fidelity improved substantially at the predetermined lower weight.',
        '2. None of the trained seeds meets the illustrative maximum-error budget, either on the random probes or over the exhaustive piecewise-linear domain calculation.',
        '3. Action-dependent pessimism survived at roughly the prior joint RMSE, with unresolved localized boundary errors.',
        '4. All seeds are near their lambda-specific population objective optimum in aggregate, but their remaining gaps exceed the architecture approximation bound by many orders of magnitude.',
        '5. The population diagonal shift is an intended tradeoff. Additional diagonal and action errors reflect unresolved empirical fitting/optimization, not a proved impossibility of preserving both.',
        '6. This scalar diagnostic is understood well enough to motivate a separate read-only evaluation of the real bank\'s ranking quality. It does not clear the absolute diagonal-budget gate for model-based training. Stop the scalar experiment at its fixed budget; do not connect it to actor-critic or alternating training on this evidence.', '',
        'The single recommended next study is a bounded frozen-model audit of the real failure bank\'s ranking quality, with its actual metric and provenance fixed in advance. That evaluation is not performed here. No additional scalar tuning is recommended from the inspected evaluation set. The scalar result does not validate the historical bank metric, establish exact diagonal equality, identify causal transitions, justify a PointMaze epsilon/L, or recover worst-case Q.', '',
        '## Checks and reproduction', '',
        'Seven focused unit tests passed: weight-dependent reference and regime; conditional global-reference comparisons; shared marginals and two-loss arithmetic; matched initialization/minibatches; feasible same-architecture reference; an exhaustive maximum detecting a narrow peak missed by a grid; and piece reconstruction plus structural bounds. A five-step smoke run preceded the three full runs and passed checkpoint replay. All nine final comparison evaluations (three new and six reused) reproduce every saved metric and array exactly. Loss arithmetic, finite records, final-step selection, hashes, raw structural bounds and historical artifact preservation passed the independent verification command.', '',
        f'The three new runs and comparison evaluation took {read(root/"completion.json")["elapsed_seconds"]:.2f} seconds on CPU, excluding preflight replay and later verification/plotting. Checkpoints remain local and ignored. Code, configurations, metrics, evaluation arrays and English figures/report are available for version control; nothing is automatically committed or pushed.', '',
        '```bash',
        'python -m unittest scripts.test_synthetic_diagonal_budget',
        'python -m scripts.synthetic_diagonal_budget --out-dir artifacts/synthetic_diagonal_budget/smoke --smoke',
        'python -m scripts.check_synthetic_diagonal_budget --run-dir artifacts/synthetic_diagonal_budget/smoke',
        'python -m scripts.synthetic_diagonal_budget --out-dir artifacts/synthetic_diagonal_budget/lambda0004_s012',
        'python -m scripts.check_synthetic_diagonal_budget --run-dir artifacts/synthetic_diagonal_budget/lambda0004_s012',
        'python -m scripts.report_synthetic_diagonal_budget --run-dir artifacts/synthetic_diagonal_budget/lambda0004_s012',
        '```', '', 'Use fresh output directories when reproducing training. The six historical local checkpoint files under artifacts/synthetic_failure_bank/bank_m3_lambda01_s012 are required for preflight matched comparison. Source/config hashes and Python/NumPy/PyTorch versions are recorded. No server or PointMaze dataset is needed.', '']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(root/'report_manifest.json', dict(source_sha256=sha(__file__),
        files={p.name: sha(p) for p in [root/'REPORT.md', root/'summary.json', *root.glob('*.png')]}))
    print('Wrote English report, seed summary and four comparison figures.')


if __name__=='__main__':
    main()
