"""Evaluation-only reference for uniform actions, bank {-3}, lambda=0.1.

No function in this module is called by the optimizer or initialization path.
"""
import numpy as np
import torch

from scripts.synthetic_lipschitz_response import ConditionalSpline


def optimal_diagonal(xp):
    return -(5. - np.asarray(xp)**2) / 22.


def optimal_response(pairs):
    pairs = np.asarray(pairs)
    return optimal_diagonal(pairs[..., 1]) - np.abs(pairs[..., 0] - pairs[..., 1])


def population_reference():
    return dict(diagonal_mse=82/1815, diagonal_rmse=float(np.sqrt(82/1815)),
        failure_mse=8563/1815, total=853/1650,
        diagonal_range=[-5/22, -2/11], minimum_outcome=-24/11,
        minimum_margin_above_bank=9/11, optimal_shift_only_total=9/11,
        forced_zero_anchor_envelope_total=17/30,
        class_reference_max_diagonal_error=1/(16**2*88),
        class_reference_population_gap=1.1/(16**4*22**2*30))


def fixed_class_reference():
    """Independent feasible near-optimum, never a training initial checkpoint.

    The same 32-wide conditioner interpolates d*(xp) on 33 uniform knots.
    Its objective gap to the full feasible-class optimum is ~1.16e-9.
    """
    model = ConditionalSpline()
    knots = np.linspace(-1, 1, 33)
    values = optimal_diagonal(knots)
    slopes = np.diff(values) / np.diff(knots)
    coefficients = np.concatenate((slopes[:1], np.diff(slopes)))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.conditioner[0].weight[:, 0] = 1.
        model.conditioner[0].bias.copy_(torch.from_numpy(-knots[:-1]))
        model.conditioner[2].weight.copy_(torch.eye(32, dtype=torch.float64))
        model.conditioner[4].weight[0].copy_(torch.from_numpy(coefficients))
        model.conditioner[4].bias[0] = float(values[0] - 2)
        model.conditioner[4].bias[1:9] = 1.
        model.conditioner[4].bias[9:] = -1.
    return model


@torch.no_grad()
def quadrature(model, nodes=256):
    """Exact in x per spline interval, Gauss-Legendre integration over xp.

    The failure reference is -3. Simpson integration of a squared affine
    response is exact inside each execution interval. The xp integration is
    numerical, so compare 128/256-node results separately from objective gaps.
    """
    xp, weights = np.polynomial.legendre.leggauss(nodes)
    weights = weights/2
    knots = model.knots.detach().numpy()
    lo = np.maximum(xp[:, None] + knots[:-1], -1.)
    hi = np.minimum(xp[:, None] + knots[1:], 1.)
    lengths = np.maximum(0., hi-lo)
    # Values outside the valid interval have zero integration weight.
    hi = np.maximum(hi, lo)
    make_pairs = lambda x: torch.from_numpy(np.stack((x, np.broadcast_to(xp[:, None], x.shape)), -1).reshape(-1, 2))
    left = model(make_pairs(lo)).numpy().reshape(lo.shape)
    middle = model(make_pairs((lo+hi)/2)).numpy().reshape(lo.shape)
    right = model(make_pairs(hi)).numpy().reshape(lo.shape)
    diag = model(torch.from_numpy(np.stack((xp, xp), -1))).numpy()
    cond_failure = (lengths*((left+3)**2+4*(middle+3)**2+(right+3)**2)/12).sum(1)
    cond_shift = (diag+3)**2
    cond_recentered = (lengths*((left-diag[:, None]+3)**2+4*(middle-diag[:, None]+3)**2+(right-diag[:, None]+3)**2)/12).sum(1)
    diag_mse = float(weights @ (diag**2))
    failure = float(weights @ cond_failure)
    shift_only = float(weights @ cond_shift)
    recentered = float(weights @ cond_recentered)
    return dict(nodes=nodes, diagonal_mse=diag_mse, failure_mse=failure,
        common_lambda01_total=diag_mse+.1*failure,
        population_objective_gap=diag_mse+.1*failure-population_reference()['total'],
        mean_diagonal=float(weights @ diag), shift_only_failure=shift_only,
        recentered_failure=recentered, shift_first_failure_reduction=9-shift_only,
        action_second_failure_reduction=shift_only-failure)


DERIVATION = '''# Pre-training reference: fixed scalar failure bank

Fix the bank {-3}, lambda=0.1, L=1, and z=x_prime uniformly distributed on
[-1,1] in BOTH losses. Draw x conditionally as Uniform[-1,1], independently
of z; do not redraw z during any x handling. Lower outcomes are worse only
by the convention of this toy task. No pointwise next-state labels are given.

For a fixed diagonal d=G(z,z), projection of any response onto [-3,infinity)
preserves 1-Lipschitzness, weakly decreases bank distance and weakly decreases
the diagonal square if d<-3. We can restrict attention to d>=-3. For fixed d,
the best response toward -3 under the full action bound is
G_d(x,z)=max(-3,d-|x-z|). It is feasible for every x and attains the nearest
bank value allowed by the diagonal constraint, simultaneously for all x.

Let D=|x-z|. The conditional objective is
J_z(d)=d^2+lambda*E[(3+d-D)_+^2]. It is strictly convex in d.
For uniform conditional x, m(z)=E[D|z]=(1+z^2)/2. On the unsaturated branch,
d*(z)=-lambda/(1+lambda)*(3-m(z))=-(5-z^2)/22.
Checking the branch globally: D<=1+|z| and
3+d*(z)-(1+|z|)>=9/11>0. Thus no relevant optimal output reaches the bank.
This is the unique diagonal optimum; G*=d*(z)-D is the associated full-bound
response. Its minimum is -24/11, strictly above -3. The optimum diagonal ranges
from -5/22 to -2/11; a zero diagonal is NOT optimal for this two-loss objective.
The quadratic bank loss does not give the -lambda/2 shift of a linear objective.

Integrating over z yields diagonal MSE=82/1815, diagonal RMSE=0.2125536717,
failure MSE=8563/1815, and J*=853/1650=0.51696969697.
The shift-only family G=d(z) has optimum d=-3/11 and total 9/11.
The best zero-anchor response -D has total 17/30. These are analytic evaluation
references, not additional trained arms and not inputs to optimization.

The existing finite ReLU conditioner is piecewise affine in z, so its exact
diagonal cannot equal this quadratic everywhere. The global feasible-class
optimum is therefore a lower bound, not a claim of exact membership in this NN.
An independently constructed model of exactly the same architecture linearly
interpolates d* at 33 knots on [-1,1], with the correct bounded action slopes.
Its maximum diagonal approximation error is h^2/88 for h=1/16. Because the
conditional objective difference is 1.1*(d-d*)^2, its population objective gap
is exactly 1.1*h^4/(22^2*30)=1.155968868e-9. Thus the optimum of the actual
architecture lies between J* and J*+1.16e-9. This reference is used ONLY in
evaluation, never as an initializer, loss target, checkpoint or candidate selector.

For d=G(z,z) and h=G(x,z)-d, decompose improvement from the zero response in
a stated order: 9-E[(3+d)^2] from moving the diagonal first, followed by
E[(3+d)^2]-E[(3+d+h)^2] from action-dependent change. These add exactly to the
bank-distance reduction, but are order-dependent because of the cross term.
Also report recentered bank distance E[(3+h)^2]; recentering is a diagnostic,
never a replacement for the final emitted model.
'''
