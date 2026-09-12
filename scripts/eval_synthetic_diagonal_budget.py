"""Reuse fixed bank probes with weight-specific references and diagonal extrema."""
import numpy as np
import torch

from scripts.eval_synthetic_failure_bank import evaluate as evaluate_previous, predict
from scripts.synthetic_budget_reference import optimal_diagonal, optimal_response, population_reference, diagonal_extrema
from scripts.synthetic_shared_response import DISTANCE_EDGES


def errors(values):
    values = np.asarray(values)
    return dict(count=int(values.size), rmse=float(np.sqrt(np.mean(values**2))),
                max_abs=float(np.max(np.abs(values))), mean=float(np.mean(values)))


def evaluate(model, config, data, weight):
    previous, arrays = evaluate_previous(model, config, data)
    # Only reuse weight-independent diagnostics; the old total/reference uses 0.1.
    keys = ['bounds', 'intervals', 'wrong_saturation', 'near_diagonal', 'decomposition',
            'mean_action_change', 'nonpositive_action_change_fraction']
    result = {key: previous[key] for key in keys}
    diag, off = data
    pd, po = arrays['diagonal_prediction'], arrays['off_prediction']
    distance = np.abs(off[:, 0]-off[:, 1])
    relative_error = arrays['action_change']+distance
    reference_d = optimal_diagonal(diag[:, 1], weight)
    truth = optimal_response(off, weight)
    dloss, floss = float(np.mean(pd**2)), float(np.mean((po+3)**2))
    reference_total = float(np.mean(reference_d**2)+weight*np.mean((truth+3)**2))
    result.update(lambda_off=weight, epsilon=config['epsilon'], diagonal=errors(pd),
                  diagonal_reference_error=errors(pd-reference_d), relative_response_error=errors(relative_error),
                  response_reference_error=errors(po-truth),
                  diagonal_fraction_exceeding_epsilon=float(np.mean(np.abs(pd)>config['epsilon'])),
                  objective=dict(diagonal_mse=dloss, failure_mse=floss, weighted_failure=weight*floss,
                                 total=dloss+weight*floss, paired_reference_total=reference_total,
                                 paired_empirical_gap=dloss+weight*floss-reference_total),
                  population_reference=population_reference(weight))
    result['distance_strata'] = []
    for lo, hi in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        mask = (distance>=lo) & (distance<hi)
        result['distance_strata'].append(dict(low=float(lo), high=float(hi), **errors(relative_error[mask])))
    result['regions'] = {}
    for name, lo, hi in [('left_boundary', -1., -.8), ('central', -.8, .8), ('right_boundary', .8, 1.)]:
        dm = (diag[:, 1]>=lo) & (diag[:, 1]<hi)
        om = (off[:, 1]>=lo) & (off[:, 1]<hi)
        result['regions'][name] = dict(diagonal=errors(pd[dm]),
            diagonal_reference_error=errors((pd-reference_d)[dm]),
            fraction_exceeding_epsilon=float(np.mean(np.abs(pd[dm])>config['epsilon'])),
            relative_response=errors(relative_error[om]))
    for nodes in [128, 256]:
        old = previous[f'quadrature{nodes}']
        total = old['diagonal_mse']+weight*old['failure_mse']
        result[f'quadrature{nodes}'] = dict(nodes=nodes, diagonal_mse=old['diagonal_mse'],
            failure_mse=old['failure_mse'], weighted_failure=weight*old['failure_mse'], total=total,
            population_objective_gap=total-result['population_reference']['total'])
    result['quadrature_total_difference'] = abs(result['quadrature128']['total']-result['quadrature256']['total'])
    axis = arrays['grid_axis']
    grid_d = predict(model, np.stack((axis, axis), -1))
    result['grid_diagonal'] = dict(**errors(grid_d),
        fraction_exceeding_epsilon=float(np.mean(np.abs(grid_d)>config['epsilon'])),
        endpoints=grid_d[[0, -1]].tolist())
    result['continuum_diagonal'], extrema_arrays = diagonal_extrema(model, config['epsilon'])
    xx, xp = np.meshgrid(axis, axis)
    arrays['grid_reference'] = optimal_response(np.stack((xx, xp), -1), weight)
    arrays['optimal_prediction'] = truth
    arrays['diagonal_reference'] = reference_d
    arrays['grid_diagonal'] = grid_d
    arrays['grid_diagonal_reference'] = optimal_diagonal(axis, weight)
    arrays['relative_response_error'] = relative_error
    for prefix in ['slice', 'close']:
        anchors = arrays['slice_anchors'][:, None]
        x = axis[None, :] if prefix=='slice' else anchors+arrays['close_offsets'][None, :]
        arrays[f'{prefix}_reference'] = optimal_response(np.stack(np.broadcast_arrays(x, anchors), -1), weight)
    arrays.update(extrema_arrays)
    with torch.no_grad():
        raw = model.conditioner(torch.from_numpy(axis[:, None])).numpy()[:, 1:]
    arrays['raw_slopes'] = raw
    target = np.where(np.arange(model.intervals)<model.intervals//2, 1., -1.)
    wrong = (raw*target < -model.bound) & arrays['interval_feasible']
    arrays['wrong_saturation_mask'] = wrong
    feasible = arrays['interval_feasible']
    result['slope_diagnosis'] = dict(
        feasible_unsaturated_fraction=float(np.mean(np.abs(raw[feasible])<model.bound)),
        feasible_wrong_sign_fraction=float(np.mean((raw*target)[feasible]<0)),
        mean_absolute_slope_error=float(np.mean(np.abs(arrays['interval_slopes']-target)[feasible])))
    return result, arrays
