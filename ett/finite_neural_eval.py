"""Post-training crossed neural/tabular evaluation on common frozen kernels."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ett import finite_crl as b
from ett.finite_crl_eval import certificate, error, exact_values, exact_visits, mean_interval, objective
from ett.finite_neural import CONFIG, NeuralCritic, array_sha, fit_neural, validate_sources


def diagnostics(theta, prediction, visits=None):
    truth = exact_values(theta)
    exact_d = exact_visits(theta)[:-1]
    true_gradient = b.loss_gradient(theta, truth, exact_d, 0.)[3]
    critic_gradient = b.loss_gradient(theta, prediction, exact_d, 0.)[3]
    actual_gradient = b.loss_gradient(theta, prediction, exact_d if visits is None else visits, 0.)[3]
    mass = 1 - b.GAMMA ** np.arange(1, b.H + 1)
    return dict(goal=error(prediction[1:, :, :, b.G], truth[1:, :, :, b.G]),
                all_labels=error(prediction[1:], truth[1:]),
                critic_only_gradient=error(critic_gradient, true_gradient),
                actual_sampled_gradient=error(actual_gradient, true_gradient),
                exact_gradient=true_gradient, critic_gradient=critic_gradient,
                actual_gradient=actual_gradient,
                wrong_gradient_signs=int(np.sum(critic_gradient * true_gradient < 0)),
                mass_max_abs=float(np.abs(prediction[1:].sum(-1) - mass[:, None, None]).max()),
                goal_at_death_max=float(prediction[1:, b.D, :, b.G].max()),
                goal_at_goal_max_error=float(np.abs(prediction[1:, b.G, :, b.G] - mass[:, None]).max()),
                value_range_excess=float(max(0., np.max(prediction[1:] - mass[:, None, None, None]))))


def evaluate(out):
    validate_sources(out)
    if (out / 'results.json').exists() or (out / 'refit_started.json').exists():
        raise ValueError('Evaluation already started; do not select retries.')
    training = json.loads((out / 'training.json').read_text())
    assert training['complete']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    frozen = []
    for record in training['records']:
        name = f'{record["arm"]}_s{record["seed"]}'
        path = out / 'checkpoints' / (name + '.npz')
        assert b.sha(path) == record['checkpoint_sha256']
        with np.load(path) as saved:
            data = {k: saved[k] for k in saved.files}
        if record['arm'].startswith('neural'):
            assert b.sha(out / 'checkpoints' / (name + '.pt')) == record['neural_checkpoint_sha256']
        frozen.append(dict(name=name, record=record, **data))
    b.write(out / 'refit_started.json', dict(all_final_checkpoints_verified=True,
                                           training_sha256=b.sha(out / 'training.json')))
    started = time.perf_counter()
    comparisons, crossed, kernels, traces = [], [], [], []
    steps, adam_steps = 0, 0
    for target in frozen:
        name, theta = target['name'], target['theta']
        rng = np.random.default_rng(CONFIG['evaluation_seed_base'] + target['record']['seed'])
        counts, used = b.mc_positives(theta, rng, CONFIG['final_query_repeats'])
        steps += used
        tabular = b.decode(b.fit_nce(counts))
        predictions = [tabular]
        fits = []
        for seed in CONFIG['seeds']:
            model = NeuralCritic(CONFIG['refit_seed_base'] + seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['critic_lr'])
            fitted = fit_neural(model, optimizer, counts, CONFIG['refit_steps'])
            adam_steps += CONFIG['refit_steps']
            assert adam_steps <= CONFIG['max_refit_neural_steps']
            predictions.append(model.q_values())
            fits.append(fitted)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()),
                       out / 'checkpoints' / f'refit_{name}_s{seed}.pt')
        # Exact values first enter here, after the fixed-budget fits finish.
        comparisons.append(dict(kernel=name, critic='tabular_refit', refit_seed=None,
                                counts_sha256=array_sha(counts), metrics=diagnostics(theta, tabular)))
        for seed, prediction, fit in zip(CONFIG['seeds'], predictions[1:], fits):
            comparisons.append(dict(kernel=name, critic='neural_refit', refit_seed=seed,
                                    counts_sha256=array_sha(counts), fit=fit,
                                    metrics=diagnostics(theta, prediction)))
        for source in frozen:
            crossed.append(dict(kernel=name, critic=source['name'],
                                own_kernel=source['name'] == name,
                                metrics=diagnostics(theta, source['q_history'][-1])))
        paths, actions, auxiliary = b.sample_paths(theta, rng, np.full(CONFIG['evaluation_repeats'], b.S), None, b.H)
        steps += len(paths) * b.H
        assert steps <= CONFIG['max_evaluation_transitions']
        rewards = (1 - b.GAMMA) * (paths[:, 1:] == b.G)
        returns = rewards @ (b.GAMMA ** np.arange(b.H))
        visits = exact_visits(theta)
        kernels.append(dict(name=name, seed=target['record']['seed'], arm=target['record']['arm'],
                            J=objective(theta), diagonal_nll=b.diagonal_loss(theta),
                            constraints=b.assert_feasible(theta), goal_probability=float(visits[-1, b.G]),
                            death_probability=float(visits[-1, b.D]), empirical_return=mean_interval(returns),
                            last_training_critic=diagnostics(theta, target['q_history'][-1])))
        for index, (past, qhat, visitation) in enumerate(zip(target['snapshots'], target['q_history'], target['visits'])):
            traces.append(dict(name=name, round=index, J=objective(past), metrics=diagnostics(past, qhat, visitation)))
        np.savez_compressed(out / f'{name}_evaluation.npz', counts=counts, Q_exact=exact_values(theta),
                            Q_refits=np.array(predictions), path=paths, executed_actions=actions,
                            rewards=rewards, returns=returns)
        np.savez_compressed(out / f'{name}_auxiliary.npz', x_prime=auxiliary)
        print(f'{name}: common-kernel refits and crossed evaluation completed', flush=True)
    cert = certificate()
    for kernel in kernels:
        kernel['certified_gap'] = kernel['J'] - cert['J_opt']
        kernel['joint_objective_gap'] = kernel['diagonal_nll'] + 4 * kernel['J'] - cert['joint_opt']
        assert kernel['certified_gap'] >= -1e-12
    results = dict(certificate=cert, kernels=kernels, refits=comparisons,
                   last_training_crossed=crossed, training_traces=traces,
                   evaluation_transitions=steps, refit_adam_steps=adam_steps,
                   evaluation_seconds=time.perf_counter() - started,
                   success=all(k['certified_gap'] <= CONFIG['close_gap_threshold'] for k in kernels if k['arm'] == 'neural_joint'))
    b.write(out / 'results.json', results)
    render_report(out, results, training)


def render_report(out, results, training):
    lines = ['# Neural contrastive critic in the certified finite-state ETT experiment', '',
             '**' + ('Success within the declared budget.' if results['success'] else 'The declared success criterion was not met.') + '**',
             'The benchmark, actor, natural-action distribution, constraints, ETT update',
             'and certificate from 3827930 are unchanged. Only the critic is replaced.',
             'All exact values and certified gaps were computed after final kernels were sealed.', '',
             '## Protocol and calibration', '',
             'The shared critic is phi(onehot(h,s,a))^T psi(y), with a 10-32-Tanh-16',
             'MLP and a learned 4-by-16 state embedding (944 trainable parameters).',
             'It receives no exact Q or special goal/death value. CPU float64 Adam uses',
             'lr=.01, 512 initial fitting steps, then 128 per refresh with warm weights',
             'and Adam state. Each arm uses 48 ETT updates, the same original ETT',
             'initialization, 512 sampled positives per query and 2,048 visitation paths.',
             'Random streams are paired; trajectories can differ when kernels diverge.', '',
             'NCE uses sampled truncated-geometric positives, alpha=0 uniform known',
             'negatives, and one-positive/(B-1)-negative weighting with B=32. Counts',
             'compress the exact empirical full-batch objective; they are not exact-Q',
             'targets. Decode Q_h=(1-gamma^h)(B-1)q(y)exp(f); Q_0=0 by definition.',
             'Stable softplus and float64 are the only numerical stabilizations.',
             'No logits, gradients or decoded values are clipped/normalized. A guard',
             'halts nonfinite values or logits >600 instead of silently changing values.',
             'ETT uses expected Q, not mean logits. Its enumerated categorical gradient',
             'and discounted visitation are the reference implementation. Critic weights',
             'and visitation are frozen for each ETT step; no actor is trained.', '',
             '## Final ETT results', '',
             f'Certified J_opt = {results["certificate"]["J_opt"]:.11f}, exactly',
             results['certificate']['J_opt_fraction'] + '. Close means an absolute gap <=.001.', '']
    for k in results['kernels']:
        d = k['last_training_critic']
        lines.append(f'- {k["name"]}: J={k["J"]:.9f}, gap={k["certified_gap"]:.3g}; '
                     f'goal={k["goal_probability"]:.6f}, death={k["death_probability"]:.6f}; '
                     f'diagonal TV={k["constraints"]["diagonal_tv"]:.6f}, '
                     f'action TV={k["constraints"]["lipschitz_tv"]:.6f}; '
                     f'last-training goal-Q RMSE={d["goal"]["rmse"]:.6f}, '
                     f'gradient RMSE={d["critic_only_gradient"]["rmse"]:.6f}.')
    lines += ['', 'These are the actual last training critics, queried on their final kernel;',
              'errors include any final-update staleness. The full 81-pair cross-kernel',
              'matrix is in results.json, separate from the refits below. Off-own-kernel',
              'errors include distribution transfer, not just critic fitting quality.', '',
              '## Same-kernel independent refits', '',
              'For each of all nine final kernels, one new set of 2,048 MC positives/query',
              'is shared by the tabular MLE and three fresh neural initializations.',
              'Every neural refit runs exactly 1,024 Adam steps. No trained weight warm',
              'start, exact target, selection, or subsequent ETT update is used.', '']
    for k in results['kernels']:
        rows = [r for r in results['refits'] if r['kernel'] == k['name']]
        tab, neu = rows[0], rows[1:]
        lines.append(f'- {k["name"]}: tabular goal-Q/gradient RMSE '
                     f'{tab["metrics"]["goal"]["rmse"]:.6f}/{tab["metrics"]["critic_only_gradient"]["rmse"]:.6f}; '
                     'neural seeds 0/1/2 goal-Q RMSE ' + ', '.join(f'{r["metrics"]["goal"]["rmse"]:.6f}' for r in neu) +
                     '; gradient RMSE ' + ', '.join(f'{r["metrics"]["critic_only_gradient"]["rmse"]:.6f}' for r in neu) + '.')
    neural_refits = [r for r in results['refits'] if r['critic'] == 'neural_refit']
    lines += ['', 'Q errors are computed against exact DP on every (h,s,a,y); goal-only',
              'summaries are above, all-label and maximum errors are saved. Gradient',
              'comparisons above use identical exact visitation to isolate critic error.',
              'Per-round diagnostics additionally use actual saved sampled visitation.',
              'Independent root-path return estimates and 95% MC intervals are saved;',
              'exact finite-state objectives need no sampling interval. Repeated equal',
              'kernels and refit seeds are not independent environments.', '',
              f'Maximum refit excess empirical NCE over the tabular optimum: {max(r["fit"]["empirical_nce_gap"] for r in neural_refits):.6g}.',
              f'Maximum refit goal-Q at death (true zero): {max(r["metrics"]["goal_at_death_max"] for r in neural_refits):.6g}.',
              f'Maximum refit decoded mass error: {max(r["metrics"]["mass_max_abs"] for r in neural_refits):.6g}.',
              'Positive empirical NCE gaps reflect finite neural optimization and possibly',
              'function-class approximation. Finite dot-product logits cannot exactly',
              'represent zero probabilities; no absorbing-state masking hides this.',
              'The experiment does not identify a neural representation lower bound.', '',
              '## Interpretation and stopping rule', '',
              ('All three neural joint runs reach the predeclared proximity threshold while feasible.' if results['success'] else
               'At least one neural joint run leaves a gap above .001; the fixed budget is not expanded.'),
              'A small ETT gap can coexist with critic error: in this monotone, box-constrained',
              'family, sufficiently accurate gradient directions can reach the same corner.',
              'This is evidence for the loss implementation on this benchmark, not exact',
              'neural calibration, causal identification, or readiness for PointMaze.',
              'The topology sends every failed progress transition to death; it does not',
              'test choosing death over delays or alternative routes. The allowed .02',
              'diagonal error is part of the feasible family and may be fully used.', '',
              f'Training: {training["transitions"]:,} transitions and {training["neural_steps"]:,} neural steps.',
              f'Evaluation: {results["evaluation_transitions"]:,} transitions and {results["refit_adam_steps"]:,} refit steps.',
              'All budgets and source hashes were saved before collection. Final checkpoints',
              'remain local and ignored; public configuration, metrics and evaluation arrays',
              'are saved here. No actor training, loss redesign, further experiments or push.', '',
              '```powershell',
              'python -m unittest scripts.test_finite_crl scripts.test_finite_neural',
              'python -m ett.finite_neural prepare --out artifacts/finite_crl/neural_fresh',
              'python -m ett.finite_neural train --out artifacts/finite_crl/neural_fresh',
              'python -m ett.finite_neural_eval --out artifacts/finite_crl/neural_fresh',
              'python -m scripts.check_finite_neural --out artifacts/finite_crl/neural_fresh',
              '```']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    evaluate(parser.parse_args().out)
