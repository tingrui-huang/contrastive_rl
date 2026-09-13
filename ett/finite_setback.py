"""Five-state setback benchmark and MC-only learning; no exact evaluator import."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ett.finite_crl import write, sha, PI, BEHAVIOR, GAMMA, H, B
from ett.finite_neural import array_sha, parameter_sha

S, P, R, G, D = range(5)
N = 5
OBS = np.array([.4, .35, .25])  # P, R, D at S on both diagonal actions.
EPS, L = .02, .15
CONFIG = dict(seeds=[0, 1, 2], arms=['neural_diagonal', 'tabular_joint', 'neural_joint'],
              rounds=48, query_repeats=512, visitation_repeats=2048, ett_lr=.12,
              off_weight=4., critic_lr=.01, first_fit_steps=512, refresh_fit_steps=128,
              refit_steps=1024, final_query_repeats=2048, evaluation_repeats=16384,
              seed_base=86000000, critic_seed_base=87000000, eval_seed_base=88000000,
              refit_seed_base=89000000, training_transition_cap=25700000,
              evaluation_transition_cap=2440000, training_neural_step_cap=39168,
              refit_neural_step_cap=27648, close_gap=.001, ordering_relative_error=.2,
              horizon=H, gamma=GAMMA, batch_size=B, alpha=0., negative_mass=.2,
              diagonal_tv_tolerance=EPS, distributional_lipschitz=L,
              network='11-32-Tanh-16 phi; learned 5x16 psi; dot product',
              final_iterate_only=True, dtype='float64 CPU', reference='1fb5622')


def project_hex(z, radius):
    """Euclidean projection onto |z0|,|z1|,|z0+z1| <= radius."""
    z = np.asarray(z, float)
    if max(abs(z[0]), abs(z[1]), abs(z.sum())) <= radius:
        return z.copy()
    vertices = radius * np.array([[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]])
    edges = np.roll(vertices, -1, axis=0) - vertices
    weight = np.clip(np.sum((z - vertices) * edges, axis=1) / np.sum(edges ** 2, axis=1), 0, 1)
    candidates = vertices + weight[:, None] * edges
    return candidates[np.argmin(np.sum((candidates - z) ** 2, axis=1))]


def project(theta):
    return np.r_[OBS[:2] + project_hex(theta[:2] - OBS[:2], EPS), project_hex(theta[2:], L)]


def kernel(theta):
    out = np.zeros((N, 2, 2, N))
    for a in range(2):
        for xp in range(2):
            probabilities = theta[:2] + (a - xp) * theta[2:]
            out[S, a, xp, [P, R, D]] = [*probabilities, 1 - probabilities.sum()]
    out[P, :, :, G] = 1
    out[R, :, :, P] = 1
    out[G, :, :, G] = 1
    out[D, :, :, D] = 1
    return out


def intervention(theta):
    return np.einsum('sxuy,u->sxy', kernel(theta), BEHAVIOR)


def feasibility(theta):
    k = kernel(theta)
    diag = np.array([theta[0], theta[1], 1 - theta[:2].sum()])
    tv = float(.5 * np.abs(diag - OBS).sum())
    action_tv = float((.5 * np.abs(k[:, 0] - k[:, 1]).sum(-1)).max())
    result = dict(diagonal_tv=tv, action_tv=action_tv,
                  violation=max(0., tv - EPS, action_tv - L, -k.min(),
                                k.max() - 1, np.abs(k.sum(-1) - 1).max()))
    assert result['violation'] < 1e-12, result
    return result


def sample_paths(theta, rng, states, first_actions, h):
    k, n = kernel(theta), len(states)
    states = np.array(states, dtype=int)
    path = np.empty((n, h + 1), np.int8)
    actions, auxiliary = np.empty((n, h), np.int8), np.empty((n, h), np.int8)
    path[:, 0] = states
    for t in range(h):
        a = (rng.random(n) < PI[1]).astype(int)
        if t == 0 and first_actions is not None:
            a = np.asarray(first_actions)
        xp = (rng.random(n) < BEHAVIOR[1]).astype(int)
        states = (rng.random(n)[:, None] >= np.cumsum(k[states, a, xp], axis=-1)).sum(-1)
        path[:, t + 1], actions[:, t], auxiliary[:, t] = states, a, xp
    return path, actions, auxiliary


def mc_counts(theta, rng, repeats):
    counts = np.zeros((H + 1, N, 2, N), np.int64)
    ss = np.repeat(np.repeat(np.arange(N), 2), repeats)
    aa = np.repeat(np.tile(np.arange(2), N), repeats)
    steps = 0
    for h in range(1, H + 1):
        paths, _, _ = sample_paths(theta, rng, ss, aa, h)
        weights = GAMMA ** np.arange(h)
        offset = rng.choice(np.arange(1, h + 1), len(ss), p=weights / weights.sum())
        np.add.at(counts[h], (ss, aa, paths[np.arange(len(ss)), offset]), 1)
        steps += len(ss) * h
    return counts, steps


def decode(odds):
    return (1 - GAMMA ** np.arange(H + 1))[:, None, None, None] * (B - 1) / N * odds


def tabular(counts):
    odds = np.zeros_like(counts, dtype=float)
    odds[1:] = counts[1:] / counts[1:].sum(-1, keepdims=True) * N / (B - 1)
    return decode(odds)


class Critic(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(H + N + 2, 32), nn.Tanh(), nn.Linear(32, 16)).double()
        self.psi = nn.Embedding(N, 16).double()
        nn.init.normal_(self.psi.weight, std=.1)
        queries = np.array(list(np.ndindex(H, N, 2)))
        self.register_buffer('features', torch.tensor(np.concatenate([
            np.eye(H)[queries[:, 0]], np.eye(N)[queries[:, 1]], np.eye(2)[queries[:, 2]]], axis=1)))

    def forward(self):
        return (self.phi(self.features) @ self.psi.weight.T).reshape(H, N, 2, N)

    @torch.no_grad()
    def values(self):
        logits = self().numpy()
        if not np.isfinite(logits).all() or logits.max() > 600:
            raise FloatingPointError('Unsafe logits: stop, do not clip or select a retry.')
        odds = np.zeros((H + 1, N, 2, N))
        odds[1:] = np.exp(logits)
        values = decode(odds)
        assert np.isfinite(values).all()
        return values


def nce_loss(logits, p):
    return (p * F.softplus(-logits) + (B - 1) / N * F.softplus(logits)).sum(-1).mean() / B


def fit(model, optimizer, counts, steps):
    p = torch.tensor(counts[1:] / counts[1:].sum(-1, keepdims=True))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nce_loss(model(), p)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters())
        optimizer.step()
    with torch.no_grad():
        final = float(nce_loss(model(), p))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return final


def gradient(theta, q, visits, weight):
    th = torch.tensor(theta, requires_grad=True)
    diagonal = torch.stack([th[0], th[1], 1 - th[0] - th[1]])
    diag = -(torch.tensor(OBS) * diagonal.log()).sum()
    off = torch.zeros((), dtype=torch.float64)
    # Keep the complete discounted visitation surrogate. Only S has parameters;
    # fixed rows still contribute their constant values, not hand-coded Q.
    fixed = intervention(theta)
    for t in range(H):
        continuation = (1 - GAMMA) * np.eye(N)[G] + GAMMA * np.einsum('a,sa->s', PI, q[H - t - 1, :, :, G])
        for s in range(N):
            for a in range(2):
                if s == S:
                    for xp in range(2):
                        p = th[:2] + (a - xp) * th[2:]
                        masses = torch.stack([p[0], p[1], 1 - p.sum()])
                        off = off + GAMMA ** t * visits[t, s] * PI[a] * BEHAVIOR[xp] * (masses * torch.tensor(continuation[[P, R, D]])).sum()
                else:
                    off = off + GAMMA ** t * visits[t, s] * PI[a] * float(fixed[s, a] @ continuation)
    dg = torch.autograd.grad(diag, th, retain_graph=True)[0].numpy()
    og = torch.autograd.grad(off, th)[0].numpy()
    return float(diag.detach()), float(off.detach()), dg, og, dg + weight * og


def prepare(out):
    if out.exists():
        raise ValueError('Use a fresh directory.')
    files = ['ett/finite_setback.py', 'ett/finite_setback_eval.py', 'scripts/test_finite_setback.py',
             'scripts/check_finite_setback.py', 'notes/finite_setback_spec.md',
             'ett/finite_crl.py', 'ett/finite_crl_eval.py', 'ett/finite_neural.py',
             'artifacts/finite_crl/neural_h4_s012_v1/REPORT.md']
    out.mkdir(parents=True)
    write(out / 'config.json', CONFIG)
    (out / 'PROTOCOL.md').write_bytes(Path('notes/finite_setback_spec.md').read_bytes())
    write(out / 'provenance.json', dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          sources={p: sha(p) for p in files}, config_sha256=sha(out / 'config.json'),
          protocol_sha256=sha(out / 'PROTOCOL.md'), numpy=np.__version__, torch=str(torch.__version__)))


def verify_sources(out):
    assert json.loads((out / 'config.json').read_text()) == CONFIG
    p = json.loads((out / 'provenance.json').read_text())
    for name, digest in p['sources'].items():
        assert sha(name) == digest, name
    assert sha(out / 'config.json') == p['config_sha256']
    assert sha(out / 'PROTOCOL.md') == p['protocol_sha256']


def train(out):
    verify_sources(out)
    ck = out / 'checkpoints'
    if ck.exists():
        raise ValueError('Training already started; no overwrite or outcome-driven retry.')
    ck.mkdir()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records, transitions, updates = [], 0, 0
    started = time.perf_counter()
    for seed in CONFIG['seeds']:
        rng = np.random.default_rng(CONFIG['seed_base'] + seed)
        initial = project(np.r_[OBS[:2] + rng.uniform(-.01, .01, 2), rng.uniform(-.03, .03, 2)])
        for arm in CONFIG['arms']:
            theta = initial.copy()
            model = None if arm == 'tabular_joint' else Critic(CONFIG['critic_seed_base'] + seed)
            opt = None if model is None else torch.optim.Adam(model.parameters(), lr=.01)
            states, qs, counts_list, vs, history = [], [], [], [], []
            for step in range(CONFIG['rounds']):
                rng = np.random.default_rng(CONFIG['seed_base'] + 10000 + 1000 * seed + step)
                counts, used = mc_counts(theta, rng, 512)
                paths, _, _ = sample_paths(theta, rng, np.full(2048, S), None, H)
                visits = np.stack([np.bincount(paths[:, t], minlength=N) / len(paths) for t in range(H)])
                transitions += used + len(paths) * H
                assert transitions <= CONFIG['training_transition_cap']
                if model is None:
                    q, loss = tabular(counts), None
                else:
                    n = CONFIG['first_fit_steps'] if step == 0 else CONFIG['refresh_fit_steps']
                    loss = fit(model, opt, counts, n)
                    updates += n
                    assert updates <= CONFIG['training_neural_step_cap']
                    q = model.values()
                states.append(theta.copy()); qs.append(q); counts_list.append(counts); vs.append(visits)
                before = None if model is None else parameter_sha(model)
                dl, off, dg, og, grad = gradient(theta, q, visits, 0. if arm == 'neural_diagonal' else 4.)
                theta = project(theta - .12 * grad)
                assert np.isfinite(grad).all()
                assert model is None or (before == parameter_sha(model) and all(not p.requires_grad for p in model.parameters()))
                history.append(dict(round=step, diagonal_nll=dl, surrogate=off, critic_nce=loss,
                                    diagonal_grad=dg, off_grad=og, counts_sha256=array_sha(counts),
                                    rng_state=rng.bit_generator.state, **feasibility(theta)))
            name = f'{arm}_s{seed}'
            path = ck / (name + '.npz')
            np.savez_compressed(path, theta=theta, initial=initial, snapshots=states, q_history=qs, counts=counts_list, visits=vs)
            record = dict(name=name, arm=arm, seed=seed, history=history, checkpoint_sha256=sha(path))
            if model is not None:
                pt = ck / (name + '.pt')
                torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), pt)
                record['neural_sha256'] = sha(pt)
            records.append(record)
            print(name + ': complete', flush=True)
    write(out / 'training.json', dict(records=records, transitions=transitions, neural_updates=updates,
          elapsed_seconds=time.perf_counter() - started, complete=True, oracle_used=False, exact_values_used=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('phase', choices=['prepare', 'train'])
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    (prepare if args.phase == 'prepare' else train)(args.out)
