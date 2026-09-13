"""Exact post-training evaluation of temporary setback and death semantics."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path

import numpy as np
import torch

from ett import finite_setback as b
from ett.finite_crl_eval import error, mean_interval


def exact(theta):
    q = np.zeros((b.H + 1, b.N, 2, b.N))
    p = b.intervention(theta)
    for h in range(1, b.H + 1):
        v = np.einsum('a,say->sy', b.PI, q[h - 1])
        q[h] = np.einsum('say,yg->sag', p, (1 - b.GAMMA) * np.eye(b.N) + b.GAMMA * v)
    return q


def visits(theta):
    transition = np.einsum('a,say->sy', b.PI, b.intervention(theta))
    d = np.zeros((b.H + 1, b.N))
    d[0, b.S] = 1
    for t in range(b.H):
        d[t + 1] = d[t] @ transition
    return d


def objective(theta):
    return float(b.PI @ exact(theta)[b.H, b.S, :, b.G])


def certificate():
    gamma, eps, ell = F(9, 10), F(1, 50), F(3, 20)
    cp, cr = gamma * (1 - gamma ** 3), gamma ** 2 * (1 - gamma ** 2)
    assert cp > cr > 0
    assert (b.H, b.GAMMA, b.EPS, b.L) == (4, .9, .02, .15)
    np.testing.assert_array_equal(b.OBS, [.4, .35, .25])
    np.testing.assert_array_equal(b.PI, [.25, .75])
    np.testing.assert_array_equal(b.BEHAVIOR, [.75, .25])
    baseline = cp * F(2, 5) + cr * F(7, 20)
    lower = baseline - (eps + ell / 2) * cp
    witness = np.array([.38, .35, -.15, 0.])
    death_alternative = np.array([.4, .33, 0., -.15])
    b.feasibility(witness); b.feasibility(death_alternative)
    np.testing.assert_allclose(objective(witness), float(lower), atol=1e-15, rtol=0)
    death = F(1, 4) + eps + ell / 2
    alternative_J = baseline - (eps + ell / 2) * cr
    np.testing.assert_allclose(objective(death_alternative), float(alternative_J), atol=1e-15, rtol=0)
    return dict(J_opt=float(lower), J_opt_fraction=str(lower), witness=witness,
                coefficient_fractions=[str(cp), str(cr), '0'],
                proof='For zero-sum z with TV<=r, c dot z >= -r(max c-min c); simultaneous shared-kernel witness attains both hexagon bounds.',
                root_outcomes=[.305, .35, .345], maximum_death_fraction=str(death),
                maximum_death_alternative=death_alternative,
                alternative_J=float(alternative_J), alternative_J_fraction=str(alternative_J),
                reward_gap_at_equal_maximum_death=str(alternative_J - lower),
                setback_V3_fraction=str(gamma * (1 - gamma ** 2)), death_V3_fraction='0',
                immediate_setback_and_death_reward=0, tie_horizons=[0, 1],
                uniquely_lowest_root_outcome='D at successor remaining horizon 3; maximum-death kernel is not uniquely reward-minimizing')


def diagnostics(theta, prediction, sampled_visits=None):
    truth, d = exact(theta), visits(theta)[:-1]
    true_parts = b.gradient(theta, truth, d, 4.)
    pred_parts = b.gradient(theta, prediction, d, 4.)
    actual = b.gradient(theta, prediction, d if sampled_visits is None else sampled_visits, 4.)
    true_step = b.project(theta - .12 * true_parts[-1]) - theta
    pred_step = b.project(theta - .12 * actual[-1]) - theta
    norms = np.linalg.norm(true_step) * np.linalg.norm(pred_step)
    # At a constrained optimum, zero projected steps have no defined angle.
    cosine = None if norms < 1e-20 else float(np.dot(true_step, pred_step) / norms)
    pv = np.einsum('a,hsa->hs', b.PI, prediction[..., b.G])
    tv = np.einsum('a,hsa->hs', b.PI, truth[..., b.G])
    true_contrast = true_parts[3][0] - true_parts[3][1]
    pred_contrast = pred_parts[3][0] - pred_parts[3][1]
    return dict(goal_error=error(prediction[1:, :, :, b.G], truth[1:, :, :, b.G]),
                all_label_error=error(prediction[1:], truth[1:]),
                setback_values=pv[:, b.R], death_values=pv[:, b.D],
                exact_setback_values=tv[:, b.R], exact_death_values=tv[:, b.D],
                setback_death_gap_h3=float(pv[3, b.R] - pv[3, b.D]),
                h1_predicted_tie_difference=float(pv[1, b.R] - pv[1, b.D]),
                gradient_error=error(pred_parts[3], true_parts[3]),
                sampled_gradient_error=error(actual[3], true_parts[3]),
                raw_gradient_sign_errors=int(np.sum(pred_parts[3] * true_parts[3] < 0)),
                progress_setback_contrast=float(pred_contrast),
                exact_progress_setback_contrast=float(true_contrast),
                contrast_sign_error=bool(pred_contrast * true_contrast < 0),
                projected_step_error=float(np.linalg.norm(pred_step - true_step)),
                projected_step_cosine=cosine,
                spurious_nonzero_step=bool(np.linalg.norm(true_step) < 1e-12 and np.linalg.norm(pred_step) > 1e-8))


def evaluate(out):
    b.verify_sources(out)
    if (out / 'evaluation_started.json').exists():
        raise ValueError('Evaluation already started; no outcome-selected refits.')
    training = json.loads((out / 'training.json').read_text())
    assert training['complete']
    frozen = []
    for r in training['records']:
        path = out / 'checkpoints' / (r['name'] + '.npz')
        assert b.sha(path) == r['checkpoint_sha256']
        if 'neural_sha256' in r:
            assert b.sha(out / 'checkpoints' / (r['name'] + '.pt')) == r['neural_sha256']
        with np.load(path) as saved:
            frozen.append(dict(record=r, **{k: saved[k] for k in saved.files}))
    b.write(out / 'evaluation_started.json', dict(training_sha256=b.sha(out / 'training.json'), all_checkpoints_sealed=True))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    kernels, comparisons, traces, crossed = [], [], [], []
    total_steps, adam_steps = 0, 0
    for target in frozen:
        record, theta = target['record'], target['theta']
        name = record['name']
        rng = np.random.default_rng(b.CONFIG['eval_seed_base'] + record['seed'])
        counts, used = b.mc_counts(theta, rng, 2048)
        total_steps += used
        predictions, refits = [b.tabular(counts)], []
        for seed in b.CONFIG['seeds']:
            model = b.Critic(b.CONFIG['refit_seed_base'] + seed)
            opt = torch.optim.Adam(model.parameters(), lr=.01)
            loss = b.fit(model, opt, counts, b.CONFIG['refit_steps'])
            predictions.append(model.values())
            adam_steps += b.CONFIG['refit_steps']
            assert adam_steps <= b.CONFIG['refit_neural_step_cap']
            path = out / 'checkpoints' / f'refit_{name}_s{seed}.pt'
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path)
            refits.append(dict(seed=seed, nce_loss=loss, checkpoint_sha256=b.sha(path)))
        for index, q in enumerate(predictions):
            comparisons.append(dict(kernel=name, critic='tabular' if index == 0 else 'neural',
                                    refit=None if index == 0 else refits[index - 1],
                                    counts_sha256=b.array_sha(counts), metrics=diagnostics(theta, q)))
        for source in frozen:
            crossed.append(dict(kernel=name, source=source['record']['name'], metrics=diagnostics(theta, source['q_history'][-1])))
        paths, actions, auxiliary = b.sample_paths(theta, rng, np.full(16384, b.S), None, b.H)
        total_steps += len(paths) * b.H
        assert total_steps <= b.CONFIG['evaluation_transition_cap']
        rewards = (1 - b.GAMMA) * (paths[:, 1:] == b.G)
        returns = rewards @ (b.GAMMA ** np.arange(b.H))
        root = b.PI @ b.intervention(theta)[b.S]
        kernels.append(dict(name=name, arm=record['arm'], seed=record['seed'], J=objective(theta),
                            root_outcomes=root[[b.P, b.R, b.D]], final_goal_probability=float(visits(theta)[-1, b.G]),
                            final_death_probability=float(visits(theta)[-1, b.D]), constraints=b.feasibility(theta),
                            empirical_return=mean_interval(returns), last_training=diagnostics(theta, target['q_history'][-1])))
        for i, (state, q, d) in enumerate(zip(target['snapshots'], target['q_history'], target['visits'])):
            traces.append(dict(name=name, round=i, J=objective(state), metrics=diagnostics(state, q, d)))
        np.savez_compressed(out / f'{name}_evaluation.npz', counts=counts, Q_refits=predictions,
                            Q_exact=exact(theta), path=paths, executed_actions=actions, rewards=rewards, returns=returns)
        np.savez_compressed(out / f'{name}_auxiliary.npz', x_prime=auxiliary)
        print(name + ': evaluated', flush=True)
    cert = certificate()
    for row in kernels:
        row['certified_gap'] = row['J'] - cert['J_opt']
        assert row['certified_gap'] >= -1e-12
    success = all(row['certified_gap'] <= .001 and abs(row['last_training']['setback_death_gap_h3'] - .171) <= .2 * .171
                  for row in kernels if row['arm'] == 'neural_joint')
    result = dict(certificate=cert, kernels=kernels, refits=comparisons, training_traces=traces,
                  last_training_crossed=crossed, success=success, evaluation_transitions=total_steps, refit_steps=adam_steps)
    b.write(out / 'results.json', result)
    report(out, result, training)


def report(out, result, training):
    c = result['certificate']
    lines = ['# Recoverable setbacks versus absorbing death', '',
             '**' + ('The neural method meets the preregistered criterion.' if result['success'] else 'The neural method does not meet the preregistered criterion.') + '**', '',
             'The previous benchmark remains unchanged as a regression. This five-state',
             'extension retains H=4, gamma=.9, pi(1)=.75, b(1)=.25, normalized goal',
             'reward, calibrated binary NCE, and alternating two-loss ETT optimization.',
             'S emits progress P, setback R, or death D. P->G; R->P; G and D absorb.',
             'All three emitted outcomes have immediate goal reward zero.', '',
             '## Analytic distinction and certified optimum', '',
             'At the decision-relevant remaining horizon 3, V(P)=.271, V(R)=.171,',
             'V(D)=0. Thus R retains strictly positive future goal occupancy despite',
             'the same immediate reward as D. At remaining horizon 0 or 1, R and D',
             'tie at zero. Those ties are reported, not interpreted as uniquely bad death.', '',
             'The four shared parameters define (d_P,d_R,1-d_P-d_R) plus',
             '(x-x_prime)*(u,v,-u-v). Observed root diagonal law is (.4,.35,.25).',
             'Both parameter pairs obey the declared hexagon TV constraints: diagonal',
             '.02 and full distributional action Lipschitz .15. Every action/natural-action',
             'cell uses this one kernel, marginalized over b. Projection enforces',
             'constraints, with no additional penalty. See PROTOCOL.md for all pairs.', '',
             f'Certified J_min = **{c["J_opt"]:.9f}**, exactly {c["J_opt_fraction"]}.',
             'A zero-sum TV-ball linear-functional bound proves this for the entire',
             'declared family; the simultaneous witness attains both hexagon bounds.',
             'Optimal root probabilities (P,R,D)=(.305,.35,.345). This is not a sampled',
             'minimum and no certificate or exact Q was used in learning/selection.', '',
             f'Maximum death alone is NOT equivalent: (.4,.255,.345) has the same',
             f'death probability .345 but J={c["alternative_J"]:.9f}, strictly larger.',
             'The task objective also values the timing of surviving progress. It does',
             'not require eliminating all setbacks or maximizing death as a substitute.', '',
             '## Final iterates and last training critics', '']
    for row in result['kernels']:
        m = row['last_training']
        lines.append(f'- {row["name"]}: J={row["J"]:.9f}, gap={row["certified_gap"]:.3g}; '
                     'P/R/D=' + '/'.join(f'{x:.6f}' for x in row['root_outcomes']) +
                     f'; V3(R)/V3(D)={m["setback_values"][3]:.6f}/{m["death_values"][3]:.6f}; '
                     f'Q RMSE={m["goal_error"]["rmse"]:.6f}, gradient RMSE={m["gradient_error"]["rmse"]:.6f}.')
    lines += ['', 'These are the actual final training critics, not post-training refits.',
              'Exact V3(R)/V3(D)=.171/0 on every kernel. Goal probability by episode end',
              'is P+R (both recover within four steps); D remains dead. Full horizon-wise',
              'values, maxima, probabilities and empirical return intervals are saved.', '',
              '## Common frozen-kernel refits', '',
              'Each of nine final kernels supplies one independent set of 2,048 MC',
              'positives/query to both the tabular MLE and three fresh neural refits.',
              'Neural refits run 1,024 fixed Adam steps each. Samples and kernels are',
              'identical within comparisons; no exact-Q targets or selected refits.', '']
    for row in result['kernels']:
        records = [r for r in result['refits'] if r['kernel'] == row['name']]
        lines.append(f'- {row["name"]}: tabular Q/gradient RMSE '
                     f'{records[0]["metrics"]["goal_error"]["rmse"]:.6f}/{records[0]["metrics"]["gradient_error"]["rmse"]:.6f}; '
                     'neural Q RMSE ' + ', '.join(f'{r["metrics"]["goal_error"]["rmse"]:.6f}' for r in records[1:]) +
                     '; neural V3(R)-V3(D) ' + ', '.join(f'{r["metrics"]["setback_death_gap_h3"]:.6f}' for r in records[1:]) + '.')
    lines += ['', 'All refit gradient errors and horizon-wise values are in results.json.',
              'The 81 crossed last-training comparisons are separate: they include',
              'transfer/staleness rather than an equal-data fitting comparison. Repeated',
              'equal kernels and shared seed streams are not independent environments.', '',
              '## Did critic error change update direction?', '']
    for arm in b.CONFIG['arms']:
        rows = [r['metrics'] for r in result['training_traces'] if r['name'].startswith(arm)]
        cosines = [r['projected_step_cosine'] for r in rows if r['projected_step_cosine'] is not None]
        lines.append(f'- {arm}: raw gradient sign errors={sum(r["raw_gradient_sign_errors"] for r in rows)}; '
                     f'progress-versus-setback contrast flips={sum(r["contrast_sign_error"] for r in rows)}; '
                     f'projected step maximum L2 difference={max(r["projected_step_error"] for r in rows):.6g}; '
                     f'minimum defined projected-step cosine={min(cosines) if cosines else None}; '
                     f'spurious nonzero steps at an exact stationary point={sum(r["spurious_nonzero_step"] for r in rows)}.')
    lines += ['', 'Gradient diagnostics compare the pessimistic direction with exact model',
              'values. Projected steps compare full NLL+4J updates; for the diagonal-only',
              'arm these are explicitly hypothetical joint updates, not its executed',
              'control update. Zero projected steps have no direction and are excluded',
              'from cosine minima. There is no visitation-gradient sampling error at S,',
              'because S occurs only at t=0; future reward estimation remains Monte Carlo.', '',
              '## Scope and reproducibility', '',
              'The critic is the previous dot-product neural architecture, with only',
              'input/state dimensions expanded: 11-32-Tanh-16 phi and 5x16 learned psi.',
              'Uniform known negatives q=1/5, alpha=0, B=32; one-positive/(B-1)-negative',
              'weighting and Q_h=(1-gamma^h)(B-1)q exp(f), Q_0=0. Stable softplus and',
              'float64; no clipping, hand-coded goal/death values or reward changes.',
              'Critics learn sampled truncated-geometric positives. ETT uses enumerated',
              'categorical gradients with critic and visitation frozen, then refreshes.', '',
              'Neural critics cannot express exact zero occupancy with finite logits.',
              'Report small nonzero values on h=1 ties as calibration error, not evidence',
              'that tied outcomes differ. The family has deterministic recovery and one',
              'parameterized branching state; this does not validate hidden-state models,',
              'arbitrary recovery dynamics or PointMaze integration. All conclusions are',
              'within this declared family and fixed budget. No actor or loss redesign.', '',
              f'Training: {training["transitions"]:,} transitions, {training["neural_updates"]:,} neural steps.',
              f'Evaluation: {result["evaluation_transitions"]:,} transitions, {result["refit_steps"]:,} refit steps.',
              'Final iterates only. Configuration/protocol/source hashes were saved before',
              'collection. Checkpoints stay local; English metrics/evaluation are public.', '',
              '```powershell',
              'python -m unittest scripts.test_finite_crl scripts.test_finite_neural scripts.test_finite_setback',
              'python -m ett.finite_setback prepare --out artifacts/finite_crl/setback_fresh',
              'python -m ett.finite_setback train --out artifacts/finite_crl/setback_fresh',
              'python -m ett.finite_setback_eval --out artifacts/finite_crl/setback_fresh',
              'python -m scripts.check_finite_setback --out artifacts/finite_crl/setback_fresh',
              '```', '', 'Stop after reporting; no follow-up experiment or push is performed.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--out', type=Path, required=True)
    evaluate(p.parse_args().out)
