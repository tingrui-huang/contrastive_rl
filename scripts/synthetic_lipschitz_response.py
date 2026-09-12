"""Bounded conditional spline follow-up; preserves all prior synthetic artifacts.

The conditioning network predicts a free intercept and bounded interval slopes.
There is no diagonal branch, target-shaped initializer or extra training loss.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn

from scripts.synthetic_shared_response import (CONFIG, DISTANCE_EDGES, SharedResponse,
    error_metrics, make_data, predict, ratio_metrics, sha, two_losses, write_json)
from scripts.synthetic_shared_response import evaluate as evaluate_old


PRIOR = Path('artifacts/synthetic_shared_response/l025_1_4_s012')
PRIOR_COMMIT = '752b6457debb65aeacc19979a97db833bbdb6149'
INTERVALS = 16
FP64_EXCESS_TOLERANCE = 1e-12


class ConditionalSpline(nn.Module):
    """G = b(xp) + sum_j slope_j(xp)*clamp(x-xp-k_j, 0, interval_width).

    For fixed xp, the emitted function is continuous and piecewise affine in x.
    Its slope inside interval j is the j-th bounded coefficient. Thus every
    pair of execution actions obeys the bound by integration, at every iterate.
    This statement concerns real arithmetic; float64 evaluations can round.
    """
    def __init__(self, width=32, intervals=INTERVALS, bound=1.):
        super().__init__()
        if intervals < 2 or intervals & (intervals - 1) or bound <= 0 or not np.isfinite(bound):
            raise ValueError('positive bound and power-of-two interval count >=2 required')
        self.bound = float(bound)
        self.intervals = int(intervals)
        self.register_buffer('knots', torch.linspace(-2, 2, intervals + 1, dtype=torch.float64))
        self.conditioner = nn.Sequential(nn.Linear(1, width), nn.ReLU(),
                                        nn.Linear(width, width), nn.ReLU(),
                                        nn.Linear(width, intervals + 1)).double()

    def coefficients(self, xp):
        raw = self.conditioner(xp.to(self.knots.dtype).reshape(-1, 1))
        return raw[:, 0], raw[:, 1:].clamp(-self.bound, self.bound)

    def forward(self, pairs):
        pairs = pairs.to(self.knots.dtype)
        shape = pairs.shape[:-1]
        pairs = pairs.reshape(-1, 2)
        intercept, slopes = self.coefficients(pairs[:, 1])
        signed = pairs[:, 0] - pairs[:, 1]
        lengths = (signed[:, None] - self.knots[:-1]).clamp(0., 4. / self.intervals)
        return (intercept + (slopes * lengths).sum(dim=-1)).reshape(shape)


def initialize(seed):
    torch.manual_seed(seed)
    return ConditionalSpline()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_checkpoint(path, spline=False):
    # Only the legacy local configs need this string-like version type.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ConditionalSpline() if spline else SharedResponse(checkpoint['arm'] == 'signed_difference')
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def setup_evaluation(config):
    """Exactly reproduce the previous evaluation RNG consumption and arrays."""
    diag, off = make_data(config['evaluation_seed'], config['eval_diagonal'], config['eval_off'])
    rng = np.random.default_rng(config['evaluation_seed'] + 1)
    uniform = rng.uniform(-1, 1, (config['eval_off'], 2)).astype(np.float32)
    triples = rng.uniform(-1, 1, (config['eval_pairs'] * 2, 3)).astype(np.float32)
    separation = np.abs(triples[:, 0].astype(float) - triples[:, 1])
    triples = triples[separation >= config['min_separation']][:config['eval_pairs']]
    with np.load(PRIOR / 'joint_L1_s0_evaluation.npz') as saved:
        np.testing.assert_array_equal(diag, saved['diagonal_pairs'])
        np.testing.assert_array_equal(off, saved['off_pairs'])
        np.testing.assert_array_equal(triples, saved['arbitrary_triples'])
    return diag, off, uniform, triples


def pair_diagnostics(left, right, separation, tolerance, fp_tolerance):
    gap = np.abs(np.asarray(left) - np.asarray(right))
    excess = gap - separation
    return dict(**ratio_metrics(gap / separation, tolerance),
                largest_signed_absolute_excess=float(excess.max()),
                largest_positive_absolute_excess=float(max(0., excess.max())),
                fraction_above_fp_tolerance=float(np.mean(excess > fp_tolerance)),
                fp_absolute_tolerance=fp_tolerance)


def extra_bounds(model, arrays, config, fp_tolerance):
    off = arrays['off_pairs']
    po = arrays['off_prediction_over_L']
    distance = np.abs(off[:, 0].astype(float) - off[:, 1])
    keep = distance >= config['min_separation']
    anchor = predict(model, np.stack((off[:, 1], off[:, 1]), axis=-1))
    triples = arrays['arbitrary_triples']
    left, right = predict(model, triples[:, [0, 2]]), predict(model, triples[:, [1, 2]])
    separation = np.abs(triples[:, 0].astype(float) - triples[:, 1])
    axis, grid = arrays['grid_axis'].astype(float), arrays['grid_over_L']
    grid_sep = np.broadcast_to(np.diff(axis)[None, :], grid[:, :-1].shape)
    result = dict(
        anchored_learned_value=pair_diagnostics(po[keep], anchor[keep], distance[keep], config['ratio_tolerance'], fp_tolerance),
        anchored_prescribed_c=pair_diagnostics(po[keep], np.zeros(keep.sum()), distance[keep], config['ratio_tolerance'], fp_tolerance),
        arbitrary_random_pairs=pair_diagnostics(left, right, separation, config['ratio_tolerance'], fp_tolerance),
        grid_adjacent_pairs=pair_diagnostics(grid[:, 1:], grid[:, :-1], grid_sep, config['ratio_tolerance'], fp_tolerance))
    learned_envelope_excess = anchor - distance - po
    result['envelope'] = dict(max_below_c_envelope=float(max(0., np.max(-distance - po))),
        max_below_learned_anchor_envelope=float(max(0., learned_envelope_excess.max())),
        below_c_envelope_fraction_at_001=float(np.mean(-distance - po > .01)),
        note='A nonzero learned diagonal shifts the valid envelope. Prescribed-c ratios are not full-action violations.')
    return result


def exact_interval_diagnostics(model, anchors):
    """Read exact affine coefficients on every feasible x interval at fixed xp.

    Conditional coefficients are evaluated numerically at these anchors. The
    coefficient clamp, not this finite anchor set, supplies the continuum proof.
    """
    anchors = np.asarray(anchors, np.float64)
    with torch.no_grad():
        intercept, slopes = model.coefficients(torch.from_numpy(anchors))
    intercept, slopes = intercept.numpy(), slopes.numpy()
    knots = model.knots.numpy()
    lo = np.maximum(knots[:-1][None, :] + anchors[:, None], -1.)
    hi = np.minimum(knots[1:][None, :] + anchors[:, None], 1.)
    feasible = hi > lo
    # Direct float64 endpoint evaluation avoids an artificial float32 input roundtrip.
    left = np.stack((lo[feasible], np.broadcast_to(anchors[:, None], lo.shape)[feasible]), axis=-1)
    right = np.stack((hi[feasible], left[:, 1]), axis=-1)
    with torch.no_grad():
        differences = (model(torch.from_numpy(right)) - model(torch.from_numpy(left))).numpy()
    expected = slopes[feasible] * (right[:, 0] - left[:, 0])
    return dict(max_exact_interval_slope=float(np.max(np.abs(slopes[feasible]))),
                feasible_interval_count=int(feasible.sum()),
                saturated_fraction=float(np.mean(np.abs(slopes[feasible]) == model.bound)),
                max_endpoint_identity_error=float(np.max(np.abs(differences - expected)))), dict(
                    slope_anchors=anchors, interval_slopes=slopes, interval_feasible=feasible,
                    interval_low=lo, interval_high=hi, intercepts=intercept)


def evaluate_spline(model, config, data):
    diag, off, uniform, triples = data
    pd, po = predict(model, diag), predict(model, off)
    distance = np.abs(off[:, 0].astype(float) - off[:, 1])
    error = po + distance
    result = dict(diagonal=error_metrics(pd, 1.), off=error_metrics(error, 1.), strata=[])
    for lo, hi in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        result['strata'].append(dict(low=float(lo), high=float(hi),
            **error_metrics(error[(distance >= lo) & (distance < hi)], 1.)))
    result['uniform_square_off'] = error_metrics(predict(model, uniform) + np.abs(uniform[:, 0].astype(float) - uniform[:, 1]), 1.)
    axis = np.linspace(-1, 1, config['grid_size'], dtype=np.float32)
    xx, xp = np.meshgrid(axis, axis)
    grid = predict(model, np.stack((xx.ravel(), xp.ravel()), axis=-1)).reshape(xx.shape)
    grid_target = -np.abs(xx.astype(float) - xp.astype(float))
    result['grid_error'] = error_metrics(grid - grid_target, 1.)
    anchors = np.array(config['slice_anchors'], np.float32)
    pa = predict(model, np.stack((anchors, anchors), axis=-1))
    sides = []
    for eps in np.array(config['side_offsets'], np.float32):
        lp, rp = np.stack((anchors - eps, anchors), axis=-1), np.stack((anchors + eps, anchors), axis=-1)
        pl, pr = predict(model, lp), predict(model, rp)
        dl, dr = anchors.astype(float) - lp[:, 0], rp[:, 0].astype(float) - anchors
        sides.append(dict(offset=float(eps), left_slope_over_L=((pa-pl)/dl).tolist(),
                          right_slope_over_L=((pr-pa)/dr).tolist(),
                          left_error_over_L=(pl+dl).tolist(), right_error_over_L=(pr+dr).tolist()))
    result['near_diagonal'] = dict(anchors=anchors.tolist(), diagonal_over_L=pa.tolist(), sides=sides)
    fine = np.linspace(-.04, .04, 401, dtype=np.float32)
    sp = np.stack(np.broadcast_arrays(axis[None, :], anchors[:, None]), axis=-1)
    cp = np.stack(np.broadcast_arrays(anchors[:, None] + fine, anchors[:, None]), axis=-1)
    arrays = dict(grid_axis=axis, grid_over_L=grid, grid_target_over_L=grid_target,
        off_pairs=off, off_prediction_over_L=po, diagonal_pairs=diag, diagonal_prediction_over_L=pd,
        arbitrary_triples=triples, slice_anchors=anchors,
        slice_over_L=predict(model, sp.reshape(-1, 2)).reshape(4, -1), close_offsets=fine,
        close_over_L=predict(model, cp.reshape(-1, 2)).reshape(4, -1))
    result['bounds'] = extra_bounds(model, arrays, config, FP64_EXCESS_TOLERANCE)
    result['intervals'], interval_arrays = exact_interval_diagnostics(model, np.unique(np.concatenate((axis, anchors))))
    arrays.update(interval_arrays)
    assert all(np.isfinite(v).all() for v in arrays.values())
    return result, arrays


def reuse_baselines(config):
    old_config, complete = read(PRIOR/'config.json'), read(PRIOR/'completion.json')
    assert old_config['source_sha256'] == sha('scripts/synthetic_shared_response.py')
    assert read(PRIOR/'verification.json')['status'] == 'passed'
    for name in CONFIG:
        if name not in ['L_values', 'seeds', 'steps', 'monitor_every']:
            assert config[name] == old_config[name], name
    old_results = read(PRIOR/'results.json')
    metrics, provenance = {}, {}
    data = make_data(config['evaluation_seed'], config['eval_diagonal'], config['eval_off'])
    for seed in config['seeds']:
        for arm in ('joint', 'signed_difference'):
            name = f'{arm}_L1_s{seed}'
            path = PRIOR/f'{name}.pt'
            assert sha(path) == complete['checkpoint_sha256'][path.name]
            model, checkpoint = load_checkpoint(path)
            reproduced, arrays = evaluate_old(model, 1., old_config, data)
            with np.load(PRIOR/f'{name}_evaluation.npz', allow_pickle=False) as saved:
                for key in saved.files:
                    np.testing.assert_array_equal(arrays[key], saved[key])
            for key, value in reproduced.items():
                assert value == old_results[name][key], (name, key)
            metric = dict(old_results[name])
            metric['bounds'] = extra_bounds(model, arrays, config, 1e-6)
            metric['parameter_count'] = sum(p.numel() for p in model.parameters())
            metrics[name] = metric
            provenance[name] = dict(checkpoint_sha256=sha(path), evaluation_sha256=sha(PRIOR/f'{name}_evaluation.npz'),
                all_saved_arrays_exact=True, all_original_metrics_exact=True, reran_training=False,
                original_path=str(PRIOR/f'{name}_evaluation.npz'))
    return metrics, provenance


def train(seed, config, data, monitor):
    model = initialize(seed)
    initial = {k: v.clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['steps'], eta_min=config['final_learning_rate'])
    diagonal, off = [torch.from_numpy(a).double() for a in data]
    md, mo = [torch.from_numpy(a).double() for a in monitor]
    rng = np.random.default_rng(840000 + seed)
    schedule_hash = hashlib.sha256()
    history = []
    start = time.perf_counter()
    for step in range(config['steps'] + 1):
        if step % config['monitor_every'] == 0 or step == config['steps']:
            with torch.no_grad():
                total, d, o = two_losses(model, md, mo, 1., 1.)
                _, slopes = model.coefficients(md[:, 1])
            history.append(dict(step=step, normalized_total=float(total), normalized_diag=float(d),
                                normalized_off=float(o), max_monitor_slope=float(slopes.abs().max())))
        if step == config['steps']:
            break
        di = rng.integers(len(diagonal), size=config['diagonal_batch'])
        oi = rng.integers(len(off), size=config['off_batch'])
        schedule_hash.update(di.tobytes()); schedule_hash.update(oi.tobytes())
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = two_losses(model, diagonal[di], off[oi], 1., 1.)
        total.backward()
        if not torch.isfinite(total) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('nonfinite loss/gradient')
        optimizer.step()
        scheduler.step()
    changed = {k: bool(torch.any(v != initial[k])) for k, v in model.named_parameters()}
    assert all(changed.values())
    return model, history, dict(training_seconds=time.perf_counter()-start,
        batch_schedule_sha256=schedule_hash.hexdigest(), all_parameter_tensors_changed=changed,
        parameter_count=sum(p.numel() for p in model.parameters()),
        data_sha256=[array_sha(a) for a in data])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError('use a fresh output directory')
    config = dict(CONFIG, L_values=[1.], intervals=INTERVALS, bound=1., dtype='float64',
                  smoke=args.smoke, fp_absolute_excess_tolerance=FP64_EXCESS_TOLERANCE)
    if args.smoke:
        config.update(steps=5, seeds=[0], monitor_every=1)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    config.update(source_sha256=sha(__file__), baseline_commit=PRIOR_COMMIT,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python_version=platform.python_version(), torch_version=str(torch.__version__), numpy_version=np.__version__,
        architecture='xp conditioner 1-32-32-17 ReLU; free intercept at signed coordinate -2; 16 clamped slopes; continuous integral basis',
        objective='mean(diagonal squared error) + 1 * mean(off squared error); exactly two terms',
        guarantee='real-arithmetic full action bound at every iterate; numerical float64 tolerance reported separately',
        initialization='ordinary seeded random linear layers; no target-shaped parameters; not function-matched to different baseline architecture',
        selection='fixed final step; original monitors/evaluation already inspected, never used to change settings',
        gradient_scope='all parameters trainable; both terms update shared trunk, intercept and left interval coefficients; right coefficients have zero diagonal derivative; clamp saturation can zero local gradients')
    # Read/hash only: never rewrite the baseline report, verification or checkpoints.
    old_files = [p for p in PRIOR.rglob('*') if p.is_file()]
    original_hashes = {p.as_posix():sha(p) for p in old_files}
    for p in Path('scripts').glob('*synthetic_shared_response*.py'):
        original_hashes[p.as_posix()] = sha(p)
    baselines, provenance = reuse_baselines(config)
    evaluation_data = setup_evaluation(config)
    out.mkdir(parents=True)
    write_json(out/'config.json',config)
    write_json(out/'baseline_provenance.json',provenance)
    write_json(out/'original_hashes.json',original_hashes)
    monitor = make_data(config['monitor_seed'],1024,2048)
    results, histories, checkpoints = dict(baselines), {}, {}
    start = time.perf_counter()
    for seed in config['seeds']:
        data = make_data(830000+seed,config['train_diagonal'],config['train_off'])
        model, history, detail = train(seed,config,data,monitor)
        name = f'conditional_spline_L1_s{seed}'
        result, arrays = evaluate_spline(model,config,evaluation_data)
        result.update(detail,seed=seed,arm='conditional_spline',L=1.)
        results[name], histories[name] = result,history
        torch.save(dict(state_dict=model.state_dict(),config=config,seed=seed,arm='conditional_spline'),out/f'{name}.pt')
        checkpoints[name] = sha(out/f'{name}.pt')
        np.savez_compressed(out/f'{name}_evaluation.npz',**arrays)
        write_json(out/'results.json',results)
        write_json(out/'history.json',histories)
        print(f'{name}: diag RMSE={result["diagonal"]["normalized_rmse"]:.6g}; off RMSE={result["off"]["normalized_rmse"]:.6g}; '
              f'pair excess={result["bounds"]["arbitrary_random_pairs"]["largest_positive_absolute_excess"]:.3g}',flush=True)
    assert all(sha(p)==v for p,v in original_hashes.items())
    write_json(out/'completion.json',dict(status='complete',new_runs=len(config['seeds']),
        elapsed_seconds=time.perf_counter()-start,checkpoints=checkpoints,original_artifacts_unchanged=True))


if __name__ == '__main__':
    main()
