"""Bounded finite-state ETT training. No exact-value or oracle dependency.

The critic is a saturated binary NCE table fitted to Monte Carlo positives.
Independent fixed-distribution negatives are integrated analytically, retaining
the repository's one-positive / (B-1)-negative weighting per row.
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


S, M, G, D = range(4)
H, GAMMA, B = 4, .9, 32
PI = np.array([.25, .75])
BEHAVIOR = np.array([.75, .25])
OBSERVED = np.array([.7, .8])
EPS, L = .02, .15
CONFIG = dict(seeds=[0, 1, 2], arms=['diagonal', 'joint'], horizon=H,
              gamma=GAMMA, batch_size=B, rounds=48, query_repeats=512,
              visitation_repeats=2048, final_query_repeats=2048,
              evaluation_repeats=16384, learning_rate=.12, off_weight=4.,
              diagonal_tv_tolerance=EPS, lipschitz=L,
              training_alpha=0., calibration_alpha=.5, seed_base=82000000,
              max_training_steps=15000000, max_evaluation_steps=1500000,
              selection='final iterate only', float_tolerance=1e-12)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(obj, indent=2, default=convert,
                                   allow_nan=False) + '\n', encoding='utf-8')


def kernel(theta):
    """T[s, executed x, auxiliary xp, successor]; all four cells share d,v."""
    theta = np.asarray(theta)
    out = np.zeros((4, 2, 2, 4))
    for s in (S, M):
        for a in range(2):
            for xp in range(2):
                p = theta[s, 0] + theta[s, 1] * (a - xp)
                out[s, a, xp, s + 1] = p
                out[s, a, xp, D] = 1 - p
    out[G, :, :, G] = 1
    out[D, :, :, D] = 1
    return out


def intervention(theta):
    return np.einsum('sxuy,u->sxy', kernel(theta), BEHAVIOR)


def feasible(theta):
    k = kernel(theta)
    diagonal_error = float(np.max(np.abs(theta[:, 0] - OBSERVED)))
    tv = .5 * np.abs(k[:, 0] - k[:, 1]).sum(-1)
    return dict(diagonal_tv=diagonal_error,
                diagonal_excess=max(0., diagonal_error - EPS),
                lipschitz_tv=float(tv.max()),
                lipschitz_excess=float(max(0., tv.max() - L)),
                probability_violation=float(max(0., -k.min(), k.max() - 1)),
                simplex_error=float(np.abs(k.sum(-1) - 1).max()))


def assert_feasible(theta):
    check = feasible(theta)
    assert max(check[k] for k in ('diagonal_excess', 'lipschitz_excess',
                                 'probability_violation', 'simplex_error')) < 1e-12
    return check


def sample_paths(theta, rng, states, first_actions, horizon):
    """Draw xp internally; only executed actions are recorded as actions.

    Goal/death absorb but the original horizon continues, with repeated goal
    rewards. No early termination, time-limit extension, or synthetic padding.
    """
    n = len(states)
    k = kernel(theta)
    states = np.asarray(states, dtype=int).copy()
    path = np.empty((n, horizon + 1), dtype=np.int8)
    actions = np.empty((n, horizon), dtype=np.int8)
    auxiliary = np.empty_like(actions)
    path[:, 0] = states
    for t in range(horizon):
        a = (rng.random(n) < PI[1]).astype(int)
        if t == 0 and first_actions is not None:
            a = np.asarray(first_actions, dtype=int)
        xp = (rng.random(n) < BEHAVIOR[1]).astype(int)
        u = rng.random(n)
        states = (u[:, None] >= np.cumsum(k[states, a, xp], axis=-1)).sum(-1)
        path[:, t + 1], actions[:, t], auxiliary[:, t] = states, a, xp
    return path, actions, auxiliary


def mc_positives(theta, rng, repeats):
    """One truncated-geometric positive per independent continuation.

    Stratify all (h,s,a), including absorbing contexts, to avoid missing a
    successor query. The first action is fixed by the query; later actions use
    the same actor. The commanded task goal remains G for every trajectory.
    """
    counts = np.zeros((H + 1, 4, 2, 4), dtype=np.int64)
    steps = 0
    ss = np.repeat(np.repeat(np.arange(4), 2), repeats)
    aa = np.repeat(np.tile(np.arange(2), 4), repeats)
    for h in range(1, H + 1):
        paths, _, _ = sample_paths(theta, rng, ss, aa, h)
        weights = GAMMA ** np.arange(h)
        offsets = rng.choice(np.arange(1, h + 1), len(ss), p=weights / weights.sum())
        positives = paths[np.arange(len(ss)), offsets]
        np.add.at(counts[h], (ss, aa, positives), 1)
        steps += len(ss) * h
    return counts, steps


def negative(alpha):
    q = np.full(4, (1 - alpha) / 4)
    q[D] += alpha
    assert np.all(q > 0)
    return q


def fit_nce(counts, alpha=0.):
    """Exact empirical NCE MLE, not an exact model-value computation.

    At each query minimize [sum p_hat softplus(-f) + (B-1) sum q softplus(f)]/B.
    The saturated table has exp(f)=p_hat/((B-1)q). Zero-count cells have
    f=-infinity at the extended MLE; we store their finite odds exactly as zero.
    There is no pseudocount, logit clip, optimizer error, or Bellman target.
    """
    denom = counts.sum(-1, keepdims=True)
    p = np.divide(counts, denom, out=np.zeros(counts.shape, float), where=denom > 0)
    return p / ((B - 1) * negative(alpha))


def decode(odds, alpha=0.):
    mass = (1 - GAMMA ** np.arange(H + 1))[:, None, None, None]
    return mass * (B - 1) * negative(alpha) * odds


def empirical_nce_loss(counts, odds, alpha=0.):
    """Finite extended-MLE loss, including exactly zero positive frequencies."""
    p = counts[1:] / counts[1:].sum(-1, keepdims=True)
    z = odds[1:]
    positive = np.zeros_like(z)
    keep = p > 0
    positive[keep] = p[keep] * np.log1p(1 / z[keep])
    negative_term = (B - 1) * negative(alpha) * np.log1p(z)
    return float((positive + negative_term).sum(-1).mean() / B)


def diagonal_loss(theta):
    d = theta[:, 0]
    return float(-np.mean(OBSERVED * np.log(d) + (1 - OBSERVED) * np.log1p(-d)))


def loss_gradient(theta, qhat, visits, off_weight):
    """Enumerated categorical score gradient with frozen Q and visitation.

    This is the policy-gradient identity for kernel parameters, not a pathwise
    gradient through a sampled state. Each categorical successor, actor action
    and xp is summed exactly. d receives both terms; v has zero diagonal
    derivative and receives only the off term. There is no third loss.
    """
    th = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    observed = torch.tensor(OBSERVED)
    diag = -(observed * th[:, 0].log() + (1 - observed) * torch.log1p(-th[:, 0])).mean()
    off = torch.zeros((), dtype=torch.float64)
    for t in range(H):
        h = H - t
        for s in (S, M):
            for a in range(2):
                # Fixed actor expectation of Q, not a mean of logits.
                good = (1 - GAMMA) * (s + 1 == G) + GAMMA * (PI @ qhat[h - 1, s + 1, :, G])
                bad = GAMMA * (PI @ qhat[h - 1, D, :, G])
                for xp in range(2):
                    p = th[s, 0] + th[s, 1] * (a - xp)
                    off = off + GAMMA ** t * visits[t, s] * PI[a] * BEHAVIOR[xp] * (p * good + (1 - p) * bad)
        # Absorbing rows have no parameters; retain their constant loss value.
        for s in (G, D):
            value = (1 - GAMMA) * (s == G) + GAMMA * (PI @ qhat[h - 1, s, :, G])
            off = off + GAMMA ** t * visits[t, s] * value
    dg = torch.autograd.grad(diag, th, retain_graph=True)[0].numpy()
    og = torch.autograd.grad(off, th)[0].numpy()
    return float(diag.detach()), float(off.detach()), dg, og, dg + off_weight * og


def project(theta):
    result = theta.copy()
    result[:, 0] = np.clip(result[:, 0], OBSERVED - EPS, OBSERVED + EPS)
    result[:, 1] = np.clip(result[:, 1], -L, L)
    return result


def prepare(out):
    if out.exists():
        raise ValueError('Use a fresh output directory; existing experiments are immutable.')
    out.mkdir(parents=True)
    sources = ['ett/finite_crl.py', 'ett/finite_crl_eval.py',
               'scripts/test_finite_crl.py', 'notes/finite_crl_spec.md',
               'contrastive/learning.py',
               'artifacts/ett_shared_design/one_step_reward_v1/REPORT.md',
               'artifacts/ett_rollout_return/residual6_s01/REPORT.md',
               'artifacts/return_readout/f4_h20_g095_v1/REPORT.md']
    write(out / 'config.json', CONFIG)
    (out / 'PROTOCOL.md').write_bytes(Path('notes/finite_crl_spec.md').read_bytes())
    write(out / 'provenance.json', dict(
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        sources={p: sha(p) for p in sources}, python=platform.python_version(),
        numpy=np.__version__, torch=str(torch.__version__), device='CPU float64',
        config_sha256=sha(out / 'config.json'), protocol_sha256=sha(out / 'PROTOCOL.md')))


def train(out):
    c = json.loads((out / 'config.json').read_text())
    assert c == CONFIG
    prov = json.loads((out / 'provenance.json').read_text())
    for p, digest in prov['sources'].items():
        assert sha(p) == digest, p
    if (out / 'training.json').exists() or (out / 'checkpoints').exists():
        raise ValueError('Training already started; do not overwrite or select retries.')
    (out / 'checkpoints').mkdir()
    torch.set_num_threads(1)
    start = time.perf_counter()
    total_steps, records = 0, []
    for seed in c['seeds']:
        init_rng = np.random.default_rng(c['seed_base'] + seed)
        initial = np.column_stack((OBSERVED + init_rng.uniform(-.015, .015, 2),
                                   init_rng.uniform(-.03, .03, 2)))
        for arm in c['arms']:
            theta, history, snapshots, odds_history = initial.copy(), [], [], []
            weight = c['off_weight'] if arm == 'joint' else 0.
            for step in range(c['rounds']):
                # Common random numbers across arms, independent refreshes.
                rng = np.random.default_rng(c['seed_base'] + 10000 + 1000 * seed + step)
                counts, steps = mc_positives(theta, rng, c['query_repeats'])
                odds = fit_nce(counts)
                qhat = decode(odds)
                paths, _, _ = sample_paths(theta, rng, np.full(c['visitation_repeats'], S), None, H)
                visits = np.stack([np.bincount(paths[:, t], minlength=4) / len(paths) for t in range(H)])
                total_steps += steps + len(paths) * H
                assert total_steps <= c['max_training_steps']
                dl, surrogate, dg, og, grad = loss_gradient(theta, qhat, visits, weight)
                assert np.isfinite([dl, surrogate, *grad.ravel()]).all()
                snapshots.append(theta.copy())
                odds_history.append(odds.copy())
                theta = project(theta - c['learning_rate'] * grad)
                history.append(dict(step=step, diagonal_nll=dl, off_surrogate=surrogate,
                                    critic_nce_loss=empirical_nce_loss(counts, odds),
                                    diagonal_grad_norm=np.linalg.norm(dg), off_grad_norm=np.linalg.norm(og),
                                    **assert_feasible(theta)))
            checkpoint = out / 'checkpoints' / f'{arm}_s{seed}.npz'
            np.savez_compressed(checkpoint, theta=theta, initial=initial,
                                snapshots=np.array(snapshots), odds_history=np.array(odds_history))
            records.append(dict(seed=seed, arm=arm, checkpoint_sha256=sha(checkpoint),
                                history=history, final_diagonal_nll=diagonal_loss(theta),
                                final_feasibility=assert_feasible(theta)))
            print(f'{arm} seed {seed}: finished {c["rounds"]} updates', flush=True)
    write(out / 'training.json', dict(records=records, model_steps=total_steps,
                                     elapsed_seconds=time.perf_counter() - start,
                                     oracle_used=False, exact_values_used=False,
                                     training_complete=True))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('phase', choices=['prepare', 'train'])
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    (prepare if args.phase == 'prepare' else train)(args.out)


if __name__ == '__main__':
    main()
