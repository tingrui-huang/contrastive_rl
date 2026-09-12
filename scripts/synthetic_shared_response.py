"""Isolated two-loss scalar response diagnostic; no PointMaze imports or training.

G(x, xp) = L * f(x, xp). Optimize the requested physical two-term objective
divided by L**2, a common numerical scale, with separately averaged terms.
L is prescribed, not inferred. No architecture enforces the diagonal anchor.
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


ARMS = ('joint', 'diagonal_only', 'signed_difference')
DISTANCE_EDGES = np.array([0, .001, .01, .1, .5, 1., 2.000001])
CONFIG = dict(L_values=[.25, 1., 4.], seeds=[0, 1, 2], steps=2500,
              hidden_width=32, lambda_off=1., learning_rate=.001,
              final_learning_rate=.0001, diagonal_batch=128, off_batch=256,
              train_diagonal=8192, train_off=32768, eval_diagonal=4096,
              eval_off=16384, eval_pairs=32768, grid_size=201,
              monitor_every=100, min_separation=.001, ratio_tolerance=.01,
              evaluation_seed=870001, monitor_seed=860001,
              near_distance_min=.0001, near_distance_max=.1,
              slice_anchors=[-.83, -.27, .19, .74],
              side_offsets=[.0001, .001, .01, .04], threads=1)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def target(x, xp, scale=1.):
    return -scale * np.abs(x - xp)


class SharedResponse(nn.Module):
    """Ordinary shared ReLU MLP, optionally with the signed difference feature."""
    def __init__(self, signed=False, width=32):
        super().__init__()
        self.signed = signed
        self.layers = nn.Sequential(nn.Linear(3 if signed else 2, width), nn.ReLU(),
                                    nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, pairs):
        if self.signed:
            pairs = torch.cat((pairs, pairs[..., :1] - pairs[..., 1:2]), dim=-1)
        return self.layers(pairs).squeeze(-1)


def initialize(seed, signed=False, width=32):
    """Match the initial function across arms; the added feature starts at zero weight."""
    torch.manual_seed(seed)
    base = SharedResponse(False, width)
    if not signed:
        return base
    model = SharedResponse(True, width)
    state = base.state_dict()
    state['layers.0.weight'] = torch.cat((state['layers.0.weight'], torch.zeros(width, 1)), dim=1)
    model.load_state_dict(state)
    return model


def sample_off(rng, n, near_min=.0001, near_max=.1):
    """Half log-uniform near distances, half uniform independent pairs with d>=.1.

    Reject out-of-domain proposals; never clip actions. This deliberately defines
    a balanced training/evaluation distribution, not a uniform square measure.
    """
    parts = []
    for near, count in ((True, n // 2), (False, n - n // 2)):
        accepted, total = [], 0
        while total < count:
            size = max(64, 2 * (count - total))
            xp = rng.uniform(-1, 1, size)
            if near:
                distance = np.exp(rng.uniform(np.log(near_min), np.log(near_max), size))
                x = xp + rng.choice([-1, 1], size) * distance
            else:
                x = rng.uniform(-1, 1, size)
            good = (np.abs(x) <= 1) & ((np.abs(x - xp) >= near_min) if near else (np.abs(x - xp) >= near_max))
            pairs = np.stack((x[good], xp[good]), axis=-1)
            accepted.append(pairs)
            total += len(pairs)
        parts.append(np.concatenate(accepted)[:count])
    result = np.concatenate(parts).astype(np.float32)
    rng.shuffle(result)
    assert np.all(np.abs(result) <= 1) and np.all(result[:, 0] != result[:, 1])
    return result


def make_data(seed, n_diag, n_off):
    rng = np.random.default_rng(seed)
    xp = rng.uniform(-1, 1, n_diag).astype(np.float32)
    return np.stack((xp, xp), axis=-1), sample_off(rng, n_off)


def two_losses(model, diagonal, off, scale, weight):
    """Physical MSE means; all shared parameters are eligible for both gradients."""
    diag_loss = (scale * model(diagonal)).square().mean()
    off_target = -scale * (off[:, 0] - off[:, 1]).abs()
    off_loss = (scale * model(off) - off_target).square().mean()
    return diag_loss + weight * off_loss, diag_loss, off_loss


@torch.no_grad()
def predict(model, pairs):
    pairs = np.asarray(pairs, np.float32)
    return np.concatenate([model(torch.from_numpy(p)).numpy() for p in np.array_split(pairs, max(1, int(np.ceil(len(pairs) / 8192))))]).astype(np.float64)


def error_metrics(error, scale):
    error = np.asarray(error, np.float64)
    return dict(count=int(error.size), normalized_mse=float(np.mean(error**2)),
                normalized_rmse=float(np.sqrt(np.mean(error**2))),
                normalized_mae=float(np.mean(np.abs(error))),
                normalized_max_abs=float(np.max(np.abs(error))),
                physical_mse=float(np.mean(error**2) * scale**2),
                physical_rmse=float(np.sqrt(np.mean(error**2)) * scale))


def ratio_metrics(ratios, tolerance):
    ratios = np.asarray(ratios, np.float64)
    return dict(count=int(ratios.size), max_ratio_over_L=float(ratios.max()),
                p99_ratio_over_L=float(np.quantile(ratios, .99)),
                fraction_above_L=float(np.mean(ratios > 1)),
                fraction_above_tolerance=float(np.mean(ratios > 1 + tolerance)))


def action_norm_bound(model):
    """Analytical global bound for df/dx; floating-point estimate of norm product.

    Fix xp, so the feature derivative is v=(1,0) or (1,0,1). ReLU is
    1-Lipschitz. Biases do not enter the bound. No spectral constraint is imposed.
    """
    weights = [layer.weight.detach().numpy().astype(np.float64) for layer in model.layers if isinstance(layer, nn.Linear)]
    v = np.array([1., 0., 1.] if model.signed else [1., 0.])
    return float(np.linalg.norm(weights[0] @ v) * np.linalg.norm(weights[1], 2) * np.linalg.norm(weights[2], 2))


def evaluate(model, scale, config, data):
    diag, off = data
    pd, po = predict(model, diag), predict(model, off)
    distance = np.abs(off[:, 0].astype(float) - off[:, 1].astype(float))
    error = po + distance
    result = dict(diagonal=error_metrics(pd, scale), off=error_metrics(error, scale), strata=[])
    for lo, hi in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        mask = (distance >= lo) & (distance < hi)
        result['strata'].append(dict(low=float(lo), high=float(hi), **error_metrics(error[mask], scale)))
    rng = np.random.default_rng(config['evaluation_seed'] + 1)
    uniform = rng.uniform(-1, 1, (config['eval_off'], 2)).astype(np.float32)
    pu = predict(model, uniform)
    result['uniform_square_off'] = error_metrics(pu + np.abs(uniform[:, 0].astype(float) - uniform[:, 1]), scale)

    # Anchored checks distinguish the learned diagonal value from the prescribed c.
    mask = distance >= config['min_separation']
    pa = predict(model, np.stack((off[:, 1], off[:, 1]), axis=-1))
    result['anchored_learned_value'] = ratio_metrics(np.abs(po[mask] - pa[mask]) / distance[mask], config['ratio_tolerance'])
    result['anchored_prescribed_c'] = ratio_metrics(np.abs(po[mask]) / distance[mask], config['ratio_tolerance'])
    result['below_optimum_fraction'] = float(np.mean(error < -config['ratio_tolerance']))
    result['max_below_optimum_normalized'] = float(max(0., -error.min()))

    # Independently drawn arbitrary execution actions; xp is identical in each pair.
    triples = rng.uniform(-1, 1, (config['eval_pairs'] * 2, 3)).astype(np.float32)
    separation = np.abs(triples[:, 0].astype(float) - triples[:, 1])
    triples = triples[separation >= config['min_separation']][:config['eval_pairs']]
    assert len(triples) == config['eval_pairs']
    left = predict(model, triples[:, [0, 2]])
    right = predict(model, triples[:, [1, 2]])
    ratios = np.abs(left - right) / np.abs(triples[:, 0].astype(float) - triples[:, 1])
    result['arbitrary_random_pairs'] = ratio_metrics(ratios, config['ratio_tolerance'])

    axis = np.linspace(-1, 1, config['grid_size'], dtype=np.float32)
    xx, xp = np.meshgrid(axis, axis)
    grid = predict(model, np.stack((xx.ravel(), xp.ravel()), axis=-1)).reshape(xx.shape)
    grid_target = -np.abs(xx.astype(float) - xp.astype(float))
    result['grid_error'] = error_metrics(grid - grid_target, scale)
    # Telescoping shows this maximum equals the all-pairs maximum ON THIS GRID.
    grid_ratios = np.abs(np.diff(grid, axis=1)) / np.diff(axis.astype(float))[None, :]
    result['grid_adjacent_pairs'] = ratio_metrics(grid_ratios, config['ratio_tolerance'])
    bound = action_norm_bound(model)
    result['analytical_norm_product_over_L'] = bound
    result['analytical_norm_product_physical'] = scale * bound

    anchors = np.array(config['slice_anchors'], np.float32)
    offsets = np.array(config['side_offsets'], np.float32)
    anchor_prediction = predict(model, np.stack((anchors, anchors), axis=-1))
    sides = []
    for epsilon in offsets:
        left_pairs = np.stack((anchors - epsilon, anchors), axis=-1)
        right_pairs = np.stack((anchors + epsilon, anchors), axis=-1)
        left_side, right_side = predict(model, left_pairs), predict(model, right_pairs)
        dl = anchors.astype(float) - left_pairs[:, 0].astype(float)
        dr = right_pairs[:, 0].astype(float) - anchors.astype(float)
        sides.append(dict(offset=float(epsilon),
                          left_slope_over_L=((anchor_prediction - left_side) / dl).tolist(),
                          right_slope_over_L=((right_side - anchor_prediction) / dr).tolist(),
                          left_error_over_L=(left_side + dl).tolist(),
                          right_error_over_L=(right_side + dr).tolist()))
    result['near_diagonal'] = dict(anchors=anchors.tolist(), diagonal_over_L=anchor_prediction.tolist(), sides=sides)
    fine = np.linspace(-.04, .04, 401, dtype=np.float32)
    slice_pairs = np.stack(np.broadcast_arrays(axis[None, :], anchors[:, None]), axis=-1)
    close_pairs = np.stack(np.broadcast_arrays(anchors[:, None] + fine, anchors[:, None]), axis=-1)
    arrays = dict(grid_axis=axis, grid_over_L=grid, grid_target_over_L=grid_target,
                  off_pairs=off, off_prediction_over_L=po, diagonal_pairs=diag,
                  diagonal_prediction_over_L=pd, arbitrary_triples=triples, arbitrary_ratios=ratios,
                  slice_anchors=anchors, slice_over_L=predict(model, slice_pairs.reshape(-1, 2)).reshape(4, -1),
                  close_offsets=fine, close_over_L=predict(model, close_pairs.reshape(-1, 2)).reshape(4, -1))
    assert all(np.isfinite(a).all() for a in arrays.values())
    return result, arrays


def train(arm, seed, scale, config, data, monitor):
    model = initialize(seed, arm == 'signed_difference', config['hidden_width'])
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['steps'], eta_min=config['final_learning_rate'])
    diagonal, off = [torch.from_numpy(a) for a in data]
    md, mo = [torch.from_numpy(a) for a in monitor]
    rng = np.random.default_rng(840000 + seed)
    weight = 0. if arm == 'diagonal_only' else config['lambda_off']
    history = []
    for step in range(config['steps'] + 1):
        if step % config['monitor_every'] == 0 or step == config['steps']:
            with torch.no_grad():
                total, d, o = two_losses(model, md, mo, scale, weight)
            values = dict(step=step, physical_total=float(total), physical_diag=float(d), physical_off=float(o),
                          normalized_total=float(total / scale**2), normalized_diag=float(d / scale**2), normalized_off=float(o / scale**2))
            if not all(np.isfinite(v) for v in values.values()):
                raise RuntimeError('nonfinite monitor loss')
            history.append(values)
        if step == config['steps']:
            break
        bd = diagonal[rng.integers(len(diagonal), size=config['diagonal_batch'])]
        bo = off[rng.integers(len(off), size=config['off_batch'])]
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = two_losses(model, bd, bo, scale, weight)
        (total / scale**2).backward()
        if not torch.isfinite(total):
            raise RuntimeError('nonfinite training loss')
        optimizer.step()
        scheduler.step()
    changed = {k: bool(torch.any(v != initial[k])) for k, v in model.state_dict().items()}
    assert all(changed.values())
    return model, history, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError('use a fresh output directory; preserve all prior runs')
    config = dict(CONFIG)
    if args.smoke:
        config.update(steps=5, seeds=[0], L_values=[1.], monitor_every=1)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True)
    config.update(arms=list(ARMS), smoke=args.smoke, device='cpu', torch_version=torch.__version__,
                  numpy_version=np.__version__, python_version=platform.python_version(),
                  source_sha256=sha(__file__), git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  target='-L*abs(x-x_prime), c=0', objective='L_diag + lambda_off * L_off; separate physical MSE means',
                  scaling='G=L*f; backward on total/L**2; same common scaling of both losses, no third term',
                  selection='fixed final step; no early stopping, validation tuning, or final-grid selection',
                  initialization='same function across all arms; signed feature has zero initial weight',
                  checkpoint_policy='local ignored .pt files; all configurations, metrics, plots and evaluation arrays are publishable',
                  train_seed_rule='830000+seed', batch_seed_rule='840000+seed',
                  sampling='explicit uniform diagonal; 50% log-uniform distance [1e-4,.1], 50% uniform square conditioned on distance>=.1; rejection, no clipping')
    write_json(out / 'config.json', config)
    eval_data = make_data(config['evaluation_seed'], config['eval_diagonal'], config['eval_off'])
    monitor = make_data(config['monitor_seed'], 1024, 2048)
    results, histories, hashes = {}, {}, {}
    start = time.perf_counter()
    for seed in config['seeds']:
        data = make_data(830000 + seed, config['train_diagonal'], config['train_off'])
        for scale in config['L_values']:
            for arm in ARMS:
                name = f'{arm}_L{scale:g}_s{seed}'
                model, history, changed = train(arm, seed, scale, config, data, monitor)
                metrics, arrays = evaluate(model, scale, config, eval_data)
                metrics.update(arm=arm, seed=seed, L=scale, all_parameter_tensors_changed=changed)
                results[name], histories[name] = metrics, history
                torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=seed, L=scale, config=config), out / f'{name}.pt')
                hashes[f'{name}.pt'] = sha(out / f'{name}.pt')
                np.savez_compressed(out / f'{name}_evaluation.npz', **arrays)
                write_json(out / 'results.json', results)
                write_json(out / 'history.json', histories)
                print(f'{name}: diag RMSE/L={metrics["diagonal"]["normalized_rmse"]:.5g}, off RMSE/L={metrics["off"]["normalized_rmse"]:.5g}', flush=True)
    write_json(out / 'completion.json', dict(status='complete', run_count=len(results), elapsed_seconds=time.perf_counter() - start,
                                            checkpoint_sha256=hashes, checkpoints_local=True))


if __name__ == '__main__':
    main()
