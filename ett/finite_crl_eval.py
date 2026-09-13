"""Post-training exact evaluation and rational certificate; never training input."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path

import numpy as np

from ett.finite_crl import (B, BEHAVIOR, CONFIG, D, EPS, G, GAMMA, H, L, M,
                            OBSERVED, PI, S, assert_feasible, decode,
                            diagonal_loss, fit_nce, intervention, loss_gradient,
                            mc_positives, negative, project, sample_paths, sha, write)


def exact_values(theta):
    p = intervention(theta)
    q = np.zeros((H + 1, 4, 2, 4))
    for h in range(1, H + 1):
        v = np.einsum('a,sag->sg', PI, q[h - 1])
        q[h] = np.einsum('say,yg->sag', p, (1 - GAMMA) * np.eye(4) + GAMMA * v)
    return q


def exact_visits(theta):
    p = np.einsum('a,say->sy', PI, intervention(theta))
    visits = np.zeros((H + 1, 4))
    visits[0, S] = 1
    for t in range(H):
        visits[t + 1] = visits[t] @ p
    return visits


def objective(theta):
    return float(PI @ exact_values(theta)[H, S, :, G])


def certificate():
    """Exact global bound over the ENTIRE declared box, via monotonicity.

    This is not enumeration of sampled parameter cells or a numerical solver's
    termination flag. Every inequality used in the proof has rational inputs.
    """
    assert (EPS, L, GAMMA, H, CONFIG['off_weight']) == (.02, .15, .9, 4, 4.)
    assert np.array_equal(OBSERVED, [.7, .8])
    assert np.array_equal(PI, [.25, .75]) and np.array_equal(BEHAVIOR, [.75, .25])
    gamma, eps, ell, w, action_gap = F(9, 10), F(1, 50), F(3, 20), F(4), F(1, 2)
    means = [F(7, 10), F(4, 5)]
    lower = [m - eps for m in means]
    pmin = [d - action_gap * ell for d in lower]
    c = gamma * (1 - gamma ** (H - 1))
    opt = c * pmin[0] * pmin[1]
    derivative_bounds = []
    for s in range(2):
        endpoints = [means[s] - eps, means[s] + eps]
        denominator_lower = min(d * (1 - d) for d in endpoints)
        bound = -eps / (2 * denominator_lower) + w * c * pmin[1 - s]
        assert bound > 0
        derivative_bounds.append(str(bound))
    witness = np.column_stack(([float(d) for d in lower], [-float(ell)] * 2))
    assert_feasible(witness)
    np.testing.assert_allclose(objective(witness), float(opt), atol=1e-15, rtol=0)
    return dict(method='Exact rational monotonicity proof on the full four-dimensional box',
                family='Shared affine action-response kernel; all b cells are simultaneous',
                J_opt_fraction=str(opt), J_opt=float(opt),
                goal_probability_fraction=str(pmin[0] * pmin[1]),
                death_probability_fraction=str(1 - pmin[0] * pmin[1]),
                normalized_return_coefficient_fraction=str(c),
                witness=witness, joint_d_derivative_lower_bounds=derivative_bounds,
                joint_v_derivative_lower_bound=str(w * c * min(pmin) * action_gap),
                joint_opt=diagonal_loss(witness) + float(w * opt),
                joint_opt_note='Optimizer and global bound certified analytically; log NLL constant is float64',
                representation_gap_within_declared_family=0.)


def error(prediction, target):
    diff = prediction - target
    return dict(rmse=float(np.sqrt(np.mean(diff ** 2))),
                mae=float(np.mean(np.abs(diff))), max_abs=float(np.abs(diff).max()))


def mean_interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return dict(mean=mean, normal_95_interval=[mean - half, mean + half], n=len(values))


def evaluate(out):
    c = json.loads((out / 'config.json').read_text())
    training = json.loads((out / 'training.json').read_text())
    assert c == CONFIG and training['training_complete']
    if (out / 'results.json').exists():
        raise ValueError('Evaluation already exists; do not overwrite results.')
    # Assert ALL final checkpoints exist and match before computing any oracle.
    for record in training['records']:
        path = out / 'checkpoints' / f'{record["arm"]}_s{record["seed"]}.npz'
        assert sha(path) == record['checkpoint_sha256']
    cert = certificate()
    records, steps, paired_returns = [], 0, {}
    for item in training['records']:
        seed, arm = item['seed'], item['arm']
        with np.load(out / 'checkpoints' / f'{arm}_s{seed}.npz') as saved:
            theta, initial = saved['theta'], saved['initial']
            snapshots, old_odds = saved['snapshots'], saved['odds_history']
        rng = np.random.default_rng(c['seed_base'] + 100000 + seed)
        counts, used = mc_positives(theta, rng, c['final_query_repeats'])
        steps += used
        odds = fit_nce(counts, 0.)
        odds_half = fit_nce(counts, .5)
        q0, qhalf = decode(odds), decode(odds_half, .5)
        truth = exact_values(theta)
        visits = exact_visits(theta)
        dg = loss_gradient(theta, truth, visits[:-1], 0.)[2]
        exact_gradient = loss_gradient(theta, truth, visits[:-1], 0.)[3]
        estimated_gradient = loss_gradient(theta, q0, visits[:-1], 0.)[3]
        paths, actions, auxiliary = sample_paths(theta, rng, np.full(c['evaluation_repeats'], S), None, H)
        steps += len(paths) * H
        assert steps <= c['max_evaluation_steps']
        rewards = (1 - GAMMA) * (paths[:, 1:] == G)
        returns = rewards @ (GAMMA ** np.arange(H))
        paired_returns[(seed, arm)] = returns
        curves = []
        for t, (past_theta, past_odds) in enumerate(zip(snapshots, old_odds)):
            past_truth = exact_values(past_theta)
            q = decode(past_odds)
            v = exact_visits(past_theta)
            true_grad = loss_gradient(past_theta, past_truth, v[:-1], 0.)[3]
            critic_grad = loss_gradient(past_theta, q, v[:-1], 0.)[3]
            curves.append(dict(round=t, J=objective(past_theta),
                               critic_goal_error=error(q[1:, :, :, G], past_truth[1:, :, :, G]),
                               critic_gradient_error=error(critic_grad, true_grad)))
        j = objective(theta)
        joint_value = diagonal_loss(theta) + c['off_weight'] * j
        record = dict(seed=seed, arm=arm, initial_J=objective(initial), J=j,
                      feasible_objective_gap=j - cert['J_opt'],
                      joint_objective=joint_value, joint_objective_gap=joint_value - cert['joint_opt'],
                      diagonal_nll=diagonal_loss(theta), constraints=assert_feasible(theta),
                      diagonal_parameter_change_l2=float(np.linalg.norm(theta[:, 0] - initial[:, 0])),
                      response_parameter_change_l2=float(np.linalg.norm(theta[:, 1] - initial[:, 1])),
                      exact_goal_probability=float(visits[-1, G]),
                      exact_death_probability=float(visits[-1, D]),
                      critic_all_queries=error(q0[1:], truth[1:]),
                      critic_goal_queries=error(q0[1:, :, :, G], truth[1:, :, :, G]),
                      critic_on_policy_value_error=float(abs(PI @ (q0[H, S, :, G] - truth[H, S, :, G]))),
                      critic_gradient_error=error(estimated_gradient, exact_gradient),
                      projected_true_joint_gradient_norm=float(np.linalg.norm(
                          theta - project(
                              theta - dg - c['off_weight'] * exact_gradient))),
                      alpha_half_corrected_difference=float(np.max(np.abs(q0 - qhalf))),
                      alpha_half_wrong_q0_error=error(decode(odds_half)[1:], truth[1:]),
                      final_zero_count_fraction=float(np.mean(counts[1:] == 0)),
                      empirical_return=mean_interval(returns),
                      empirical_goal_probability=mean_interval(paths[:, -1] == G),
                      empirical_death_probability=mean_interval(paths[:, -1] == D),
                      training_curves=curves)
        assert record['feasible_objective_gap'] >= -1e-12
        assert record['joint_objective_gap'] >= -1e-12
        assert record['alpha_half_corrected_difference'] < 1e-14
        records.append(record)
        # These are evaluation predictions/targets/trajectories, not checkpoints.
        np.savez_compressed(out / f'{arm}_s{seed}_evaluation.npz', Q_exact=truth,
                            Q_hat=q0, Q_hat_alpha_half=qhalf,
                            path=paths, executed_actions=actions, rewards=rewards, returns=returns)
        np.savez_compressed(out / f'{arm}_s{seed}_auxiliary.npz', x_prime=auxiliary)
    pairs = [dict(seed=s, joint_minus_control=mean_interval(
        paired_returns[(s, 'joint')] - paired_returns[(s, 'diagonal')])) for s in c['seeds']]
    results = dict(certificate=cert, records=records, paired_comparisons=pairs,
                   evaluation_model_steps=steps, training_model_steps=training['model_steps'],
                   calibration='Same final MC positives; alpha .5 changes only known negative distribution',
                   intervals='Independent root paths; paired common random numbers within seed. Normal 95% MC intervals only.',
                   oracle_is_post_training=True)
    write(out / 'results.json', results)
    report(out, results)
    write(out / 'manifest.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})


def report(out, results):
    cert = results['certificate']
    lines = ['# Finite-state Contrastive RL pessimistic-loss validation', '',
             'This experiment jointly fits the diagonal and minimizes a calibrated',
             'Monte Carlo contrastive continuation objective inside a declared feasible',
             'four-parameter ETT family. It does not train an actor or use PointMaze data.', '',
             '## Certified benchmark and implementation', '',
             f'The certified feasible return minimum is **{cert["J_opt"]:.9f}**',
             f'({cert["J_opt_fraction"]} exactly). The analytic witness is d=(.68,.78),',
             'v=(-.15,-.15), with intervention success probabilities (.605,.705).',
             f'Its goal probability is {cert["goal_probability_fraction"]}; death is',
             f'{cert["death_probability_fraction"]}. Both stages and both natural-action',
             'cells use one simultaneously feasible shared kernel. The certificate is a',
             'rational monotonicity proof over the complete box, not sampled search.',
             'It also certifies the same optimizer for diagonal NLL + 4J; its numerical',
             f'value is {cert["joint_opt"]:.9f}. See PROTOCOL.md for the proof and limits.', '',
             'The horizon is 4; gamma=.9; rewards are .1 after each transition into G.',
             'Goal and death are absorbing, with the original horizon retained. Actor',
             'pi(1)=.75 and natural b(1)=.25 remain fixed. Every diagonal error is bounded',
             'by TV .02; every action pair at fixed state/natural action obeys TV <=',
             '.15 times action distance. This distributional bound is not samplewise.',
             'It provides no general physical or causal guarantee.', '',
             'Both arms use identical seed-specific initialization, 48 rounds and equal',
             'sampling budgets. d receives both loss gradients in the joint arm; v has',
             'zero diagonal derivative and receives the pessimistic gradient. No diagonal',
             'backbone is frozen. Projection enforces constraints at every update.',
             'Categorical successors, actor actions and natural actions are enumerated',
             'when differentiating the frozen-critic one-step surrogate. Visitation is',
             'held fixed and weighted by gamma^t; with exact Q this derivative equals',
             'the full finite-horizon return gradient. The surrogate value is not J.', '',
             '## Results (individual seeds, final iterate)', '']
    for r in results['records']:
        f = r['constraints']
        lines.append(f'- {r["arm"]}, seed {r["seed"]}: J={r["J"]:.9f}; feasible gap={r["feasible_objective_gap"]:.9f}; '
                     f'joint-objective gap={r["joint_objective_gap"]:.9f}; diagonal TV={f["diagonal_tv"]:.6f}; '
                     f'goal={r["exact_goal_probability"]:.6f}; death={r["exact_death_probability"]:.6f}; '
                     f'critic goal RMSE={r["critic_goal_queries"]["rmse"]:.6f}, max={r["critic_goal_queries"]["max_abs"]:.6f}; '
                     f'all-label RMSE={r["critic_all_queries"]["rmse"]:.6f}; '
                     f'gradient RMSE={r["critic_gradient_error"]["rmse"]:.6f}.')
    lines += ['', 'Independent MC paired joint-minus-control return estimates:']
    for p in results['paired_comparisons']:
        v = p['joint_minus_control']
        lines.append(f'- Seed {p["seed"]}: {v["mean"]:.6f}, 95% MC interval {v["normal_95_interval"]}.')
    max_viol = max(max(r['constraints'][k] for k in ('diagonal_excess', 'lipschitz_excess',
                   'probability_violation', 'simplex_error')) for r in results['records'])
    joints = [r for r in results['records'] if r['arm'] == 'joint']
    success = all(r['feasible_objective_gap'] < 1e-10 for r in joints)
    lines += ['', f'Maximum final feasibility excess: {max_viol:.3g}. All training iterates',
              'are also checked and saved in training.json. Improvements are rejected if',
              'any feasibility excess exceeds 1e-12. MC intervals describe simulator',
              'sampling error within a seed, not uncertainty across environments or',
              'training distributions. Exact finite-state results need no sampling CI.', '',
              '## Critic calibration and error attribution', '',
              'The repository binary NCE loss averages one positive and B-1 negative',
              'entries. Here negatives have a fixed known distribution rather than the',
              'other batch rows\' achieved-state marginal. Their expectation is integrated',
              'exactly. The fitted saturated table is the empirical NCE MLE, computed',
              'from sampled truncated-geometric positives. It is not fitted to exact',
              'Q or TD targets. Zero empirical cells use the extended -infinity-logit',
              'MLE, stored as finite zero odds, without arbitrary clipping.', '',
              'The stationarity equation gives Q_hat_h=(1-gamma^h)(B-1)q_alpha exp(f_h).',
              'The (B-1), known negative density and remaining-horizon mass all matter.',
              'Expected Q is averaged in probability/value space, never logit space.',
              'On the exact same final MC counts alpha=.5 correctly decoded agrees',
              f'with alpha=0 to {max(r["alpha_half_corrected_difference"] for r in results["records"]):.3g}.',
              'This is a controlled calibration identity, not evidence that failure',
              'negatives improve representation or optimization. Decoding alpha=.5',
              'with q_0 incorrectly doubles the goal prediction; errors are saved.', '',
              'Finite MC counts cause critic and gradient estimation error. Saturated',
              'NCE fitting has no iterative fitting error and no finite-support value',
              'representation error (extended logits are permitted). Some zero-count',
              'cells can reflect sampling zeros as well as truly unreachable states.',
              'The feasible-objective gap measures final transition optimization error',
              'inside the declared family. Its exact minimizer is representable, but',
              'this affine two-stage topology excludes alternative delays, routes,',
              'state-dependent policies, hidden-state mixtures and general kernels.',
              'There is no claim of zero representation error relative to such models.', '',
              '## Decision and reproducibility', '',
              ('All three joint runs reach the certified feasible optimum within numerical tolerance.' if success else
               'The bounded joint runs leave a measurable feasible optimization gap; do not enlarge the budget.'),
              'This validates the calibrated loss/gradient implementation on this small',
              'fully observed example. Stop here. It does not establish reliable neural',
              'critic extrapolation, conditional ETT identification, the true environment\'s',
              'worst case, or readiness for large PointMaze/actor-training experiments.', '',
              f'Training used {results["training_model_steps"]:,} transitions (cap 15,000,000);',
              f'independent evaluation used {results["evaluation_model_steps"]:,} (cap 1,500,000).',
              'Source/config/protocol hashes were saved before training. All six final',
              'checkpoint hashes were verified before invoking the separate oracle.',
              'Checkpoints are local and ignored; configuration, metrics, predictions,',
              'trajectories and this English report are publication-ready. No push.', '',
              '```powershell',
              'python -m unittest scripts.test_finite_crl',
              'python -m ett.finite_crl prepare --out artifacts/finite_crl/fresh_run',
              'python -m ett.finite_crl train --out artifacts/finite_crl/fresh_run',
              'python -m ett.finite_crl_eval --out artifacts/finite_crl/fresh_run',
              '```', '',
              'Prior implementation/report references and the full estimator derivation',
              'are in PROTOCOL.md and provenance.json. Background: [Eysenbach et al.',
              '(2022)](https://arxiv.org/abs/2206.07568); the finite-horizon and B-1',
              'normalization here is derived for the explicitly declared negative sampler.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    evaluate(parser.parse_args().out)
