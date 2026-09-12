"""Evaluation-only population references and exhaustive scalar diagonal extrema."""
import numpy as np
import torch
from torch import nn

from scripts.synthetic_lipschitz_response import ConditionalSpline


def population_reference(weight):
    if not 0 <= weight < 1:
        raise ValueError('this unsaturated reference requires 0 <= lambda < 1')
    a = weight / (1 + weight)
    diagonal = a*a*82/15
    failure = 17/3 + (a*a-2*a)*82/15
    return dict(lambda_off=weight, diagonal_mse=diagonal,
                diagonal_rmse=float(np.sqrt(diagonal)), failure_mse=failure,
                total=diagonal+weight*failure, max_abs_diagonal=2.5*a,
                minimum_margin_above_bank=1-2*a,
                class_reference_max_diagonal_error=a/(8*16**2),
                class_reference_population_gap=(1+weight)*a*a/(120*16**4),
                response_identified=bool(weight > 0))


def optimal_diagonal(xp, weight):
    population_reference(weight)
    return -weight/(1+weight)*(5-np.asarray(xp)**2)/2


def optimal_response(pairs, weight):
    pairs = np.asarray(pairs)
    return optimal_diagonal(pairs[..., 1], weight)-np.abs(pairs[..., 0]-pairs[..., 1])


def fixed_class_reference(weight):
    """A feasible approximation for evaluation, never training or initialization."""
    model = ConditionalSpline()
    knots = np.linspace(-1, 1, 33)
    values = optimal_diagonal(knots, weight)
    slopes = np.diff(values)/np.diff(knots)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.conditioner[0].weight[:, 0] = 1.
        model.conditioner[0].bias.copy_(torch.from_numpy(-knots[:-1]))
        model.conditioner[2].weight.copy_(torch.eye(32, dtype=torch.float64))
        model.conditioner[4].weight[0].copy_(torch.from_numpy(np.r_[slopes[0], np.diff(slopes)]))
        model.conditioner[4].bias[0] = float(values[0]-2)
        model.conditioner[4].bias[1:9] = 1.
        model.conditioner[4].bias[9:] = -1.
    return model


def _cuts(lo, hi, slopes, intercepts, levels):
    roots = [lo, hi]
    nonzero = slopes != 0
    for level in levels:
        candidates = (level-intercepts[nonzero])/slopes[nonzero]
        roots.extend(candidates[(candidates > lo) & (candidates < hi)].tolist())
    return np.unique(roots)


@torch.no_grad()
def diagonal_extrema(model, epsilon=.01):
    """Enumerate all ReLU and relevant clamp breakpoints on xp in [-1,1].

    Each tuple represents features A*xp+B on an interval. No grid is used to
    locate the maximum. The algorithm is exact for real piecewise affine
    arithmetic; float64 roots are not an outward-rounded formal certificate.
    """
    pieces = [(-1., 1., np.ones(1), np.zeros(1))]
    for layer in model.conditioner:
        following = []
        for lo, hi, a, b in pieces:
            if isinstance(layer, nn.Linear):
                w, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
                following.append((lo, hi, w@a, w@b+bias))
            elif isinstance(layer, nn.ReLU):
                cuts = _cuts(lo, hi, a, b, [0.])
                for left, right in zip(cuts[:-1], cuts[1:]):
                    active = a*(left+(right-left)/2)+b > 0
                    following.append((left, right, a*active, b*active))
            else:
                raise TypeError('only Linear/ReLU conditioners are supported')
        pieces = following
    lengths = np.clip(-model.knots[:-1].numpy(), 0, 4/model.intervals)
    relevant = lengths != 0
    scalar = []
    for lo, hi, a, b in pieces:
        cuts = _cuts(lo, hi, a[1:][relevant], b[1:][relevant], [-model.bound, model.bound])
        for left, right in zip(cuts[:-1], cuts[1:]):
            raw = a[1:]*(left+(right-left)/2)+b[1:]
            free = np.abs(raw) < model.bound
            slope = a[0]+np.sum(lengths*a[1:]*free)
            intercept = b[0]+np.sum(lengths*np.where(free, b[1:], np.clip(raw, -model.bound, model.bound)))
            scalar.append((left, right, slope, intercept))
    intervals = np.asarray(scalar)
    left, right, a, b = intervals.T
    endpoints = np.unique(intervals[:, :2])
    values = model(torch.from_numpy(np.stack((endpoints, endpoints), -1))).numpy()
    direct_left = model(torch.from_numpy(np.stack((left, left), -1))).numpy()
    direct_right = model(torch.from_numpy(np.stack((right, right), -1))).numpy()
    reconstruction = max(np.max(np.abs(a*left+b-direct_left)), np.max(np.abs(a*right+b-direct_right)))
    exceed_length = 0.
    for lo, hi, slope, intercept in scalar:
        cuts = _cuts(lo, hi, np.array([slope]), np.array([intercept]), [-epsilon, epsilon])
        for start, end in zip(cuts[:-1], cuts[1:]):
            if abs(slope*(start+(end-start)/2)+intercept) > epsilon:
                exceed_length += end-start
    index = int(np.argmax(np.abs(values)))
    metrics = dict(max_abs=float(abs(values[index])), argmax=float(endpoints[index]),
                   min=float(values.min()), max=float(values.max()), pieces=len(scalar),
                   uniform_fraction_exceeding_epsilon=float(exceed_length/2),
                   meets_epsilon=bool(abs(values[index]) <= epsilon),
                   max_endpoint_reconstruction_error=float(reconstruction),
                   method='exhaustive piecewise-affine breakpoint enumeration, float64; no formal rounding certificate')
    assert reconstruction < 1e-11
    return metrics, dict(diagonal_intervals=intervals, diagonal_breakpoints=endpoints, diagonal_breakpoint_values=values)


DERIVATION = '''# Pre-training derivation: a scalar diagonal-error budget

Fix x,z=x_prime independently uniform on [-1,1], c=0, bank {-3}, and the
full 1-Lipschitz bound in x at each fixed z. Both losses use the same z marginal.
Integrating separately on [-1,z] and [z,1] gives E[|x-z||z]=(1+z^2)/2.

Projection of a feasible response onto [-3,infinity) preserves its full bound,
weakly improves bank distance, and improves the diagonal square if d=G(z,z)<-3.
Thus restrict to d>=-3. At fixed d, the closest feasible response to the bank is
G_d(x,z)=max(-3,d-|x-z|), simultaneously for all x. This is feasible itself.
The minimized conditional objective is J_z(d)=d^2+lambda*E[(3+d-|x-z|)_+^2],
strictly convex in d. Write a=lambda/(1+lambda). Its unsaturated stationary
point is d*(z)=-a*(5-z^2)/2. For 0<=lambda<1 and t=|z|<=1, its minimum margin
above the bank is 2-t-a*(5-t^2)/2, decreasing in t, hence at least
1-2*a=(1-lambda)/(1+lambda)>0. The stationary point is in the valid branch
everywhere and strict convexity proves global optimality, not just stationarity.
For lambda>0 the optimal response is G*=d*(z)-|x-z|. At lambda=0 only the
diagonal is identified; -|x-z| is merely an evaluation reference for the control.

E[(3-E[D|z])^2]=82/15 and E[(3-D)^2]=17/3. Therefore
diagonal MSE=a^2*82/15, failure MSE=17/3+(a^2-2*a)*82/15, and
J*=lambda*17/3-lambda^2/(1+lambda)*82/15.
The maximum absolute optimal diagonal is 2.5*a.

Predetermine epsilon=0.01 and lambda=0.004 before any new training.
Then a=1/251, maximum |d*|=0.0099601593625498, diagonal RMSE=0.009315101151,
failure MSE=5.623194340831, J*=0.022579548472776, and minimum margin above
the bank=0.99203187251. The optimal budget slack is only 0.00003984063745.
This is an illustrative native scalar outcome budget, not a PointMaze tolerance.
A trained model can exceed it through approximation, finite-data or optimization
error. Finite probes alone do not certify a continuum bound.

The conditioner is piecewise affine in z, whereas d* is quadratic. An independent
same-architecture reference interpolates d* with 33 knots, h=1/16, and has exact
relative action slopes +1/-1. Its maximum diagonal approximation error is
a*h^2/8. Since J_z(d)-J_z(d*)=(1+lambda)*(d-d*)^2 in this regime, integration
gives gap (1+lambda)*a^2*h^4/120. This tightly brackets the finite architecture's
optimum above J*. It is evaluation only; never used to initialize or train.

The new run changes only lambda, reusing the unchanged training routine, all
three original initializations, datasets, minibatch streams, optimizer and 2500
steps. Old lambda=0.1 joint and lambda=0 diagonal-only checkpoints are replayed
read-only. Each objective gap uses its own weight; raw differently weighted
totals are not compared as performance scores. No settings or final checkpoint
are selected using the already-inspected evaluation sets.
'''
