"""Neural-critic-only extension of finite_crl; no oracle or exact Q imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ett import finite_crl as base


CONFIG = dict(reference_commit='3827930', seeds=[0, 1, 2],
              arms=['tabular_joint', 'neural_joint', 'neural_diagonal'],
              rounds=48, query_repeats=512, visitation_repeats=2048,
              ett_lr=.12, off_weight=4., alpha=0., batch_size=32,
              architecture='onehot(h:4,s:4,a:2) -> Linear(10,32) -> Tanh -> Linear(32,16); psi=Embedding(4,16)',
              hidden=32, embedding=16, critic_lr=.01, first_fit_steps=512,
              refresh_fit_steps=128, refit_steps=1024, optimizer='Adam defaults, no weight decay',
              precision='float64 CPU, one thread', ett_seed_base=82000000,
              critic_seed_base=83000000, refit_seed_base=84000000,
              evaluation_seed_base=85000000, final_query_repeats=2048,
              evaluation_repeats=16384, max_training_transitions=21300000,
              max_evaluation_transitions=2100000, max_training_neural_steps=39168,
              max_refit_neural_steps=27648, close_gap_threshold=.001,
              final_iterate_only=True)


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class NeuralCritic(nn.Module):
    """Shared dot-product critic; no state-specific values or output branches."""
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 16)).double()
        self.psi = nn.Embedding(4, 16).double()
        # Small learned embeddings limit initial logit scale without changing NCE.
        nn.init.normal_(self.psi.weight, std=.1)
        queries = np.array(list(np.ndindex(4, 4, 2)))
        features = np.concatenate([np.eye(4)[queries[:, 0]], np.eye(4)[queries[:, 1]],
                                   np.eye(2)[queries[:, 2]]], axis=1)
        self.register_buffer('features', torch.from_numpy(features))

    def forward(self):
        return (self.phi(self.features) @ self.psi.weight.T).reshape(4, 4, 2, 4)

    @torch.no_grad()
    def q_values(self):
        logits = self().numpy()
        if not np.isfinite(logits).all() or logits.max() > 600:
            raise FloatingPointError('Nonfinite or unsafe logits; stop, never clip or select a retry.')
        odds = np.zeros((5, 4, 2, 4))
        odds[1:] = np.exp(logits)
        values = base.decode(odds, 0.)
        assert np.isfinite(values).all()
        assert np.all(values[0] == 0)
        return values


def nce_loss(logits, probabilities):
    # Sufficient counts of sampled positives give the EXACT empirical full-batch
    # binary NCE objective; fixed uniform negative expectation is integrated.
    return ((probabilities * F.softplus(-logits) +
             (base.B - 1) * .25 * F.softplus(logits)).sum(-1).mean() / base.B)


def fit_neural(model, optimizer, counts, steps):
    probabilities = torch.from_numpy(counts[1:] / counts[1:].sum(-1, keepdims=True))
    losses = []
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nce_loss(model(), probabilities)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite NCE loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite neural gradient')
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        logits = model()
        final_loss = float(nce_loss(logits, probabilities))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return dict(first_loss=losses[0], final_loss=final_loss, steps=steps,
                min_logit=float(logits.min()), max_logit=float(logits.max()),
                tabular_empirical_optimum=base.empirical_nce_loss(counts, base.fit_nce(counts)),
                empirical_nce_gap=final_loss - base.empirical_nce_loss(counts, base.fit_nce(counts)))


def parameter_sha(model):
    return array_sha(np.concatenate([p.detach().numpy().ravel() for p in model.parameters()]))


def prepare(out):
    if out.exists():
        raise ValueError('Use a fresh output directory.')
    sources = ['ett/finite_crl.py', 'ett/finite_crl_eval.py', 'ett/finite_neural.py',
               'ett/finite_neural_eval.py', 'scripts/test_finite_neural.py',
               'scripts/check_finite_neural.py', 'notes/finite_neural_spec.md',
               'artifacts/finite_crl/tabular_h4_s012_v1/REPORT.md',
               'artifacts/finite_crl/tabular_h4_s012_v1/results.json']
    out.mkdir(parents=True)
    base.write(out / 'config.json', CONFIG)
    (out / 'PROTOCOL.md').write_bytes(Path('notes/finite_neural_spec.md').read_bytes())
    base.write(out / 'provenance.json', dict(
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={p: base.sha(p) for p in sources}, config_sha256=base.sha(out / 'config.json'),
        protocol_sha256=base.sha(out / 'PROTOCOL.md'), torch=str(torch.__version__),
        numpy=np.__version__, reference_config=base.CONFIG))


def validate_sources(out):
    assert json.loads((out / 'config.json').read_text()) == CONFIG
    provenance = json.loads((out / 'provenance.json').read_text())
    for name, digest in provenance['sources'].items():
        assert base.sha(name) == digest, name
    assert base.sha(out / 'config.json') == provenance['config_sha256']
    assert base.sha(out / 'PROTOCOL.md') == provenance['protocol_sha256']


def train(out):
    validate_sources(out)
    checkpoint_dir = out / 'checkpoints'
    if checkpoint_dir.exists():
        raise ValueError('Training already started; do not overwrite/retry.')
    checkpoint_dir.mkdir()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    transitions, neural_steps, records = 0, 0, []
    for seed in CONFIG['seeds']:
        rng = np.random.default_rng(CONFIG['ett_seed_base'] + seed)
        initial = np.column_stack((base.OBSERVED + rng.uniform(-.015, .015, 2),
                                   rng.uniform(-.03, .03, 2)))
        for arm in CONFIG['arms']:
            model = None if arm == 'tabular_joint' else NeuralCritic(CONFIG['critic_seed_base'] + seed)
            optimizer = None if model is None else torch.optim.Adam(model.parameters(), lr=CONFIG['critic_lr'])
            initial_model_sha = None if model is None else parameter_sha(model)
            theta = initial.copy()
            history, snapshots, queries, visitations, counts_all = [], [], [], [], []
            for step in range(CONFIG['rounds']):
                rng_seed = CONFIG['ett_seed_base'] + 10000 + 1000 * seed + step
                rng = np.random.default_rng(rng_seed)
                counts, used = base.mc_positives(theta, rng, CONFIG['query_repeats'])
                paths, _, _ = base.sample_paths(theta, rng, np.full(CONFIG['visitation_repeats'], base.S), None, base.H)
                visits = np.stack([np.bincount(paths[:, t], minlength=4) / len(paths) for t in range(base.H)])
                transitions += used + len(paths) * base.H
                assert transitions <= CONFIG['max_training_transitions']
                if model is None:
                    qhat = base.decode(base.fit_nce(counts))
                    fitting = dict(steps=0, final_loss=base.empirical_nce_loss(counts, base.fit_nce(counts)))
                else:
                    updates = CONFIG['first_fit_steps'] if step == 0 else CONFIG['refresh_fit_steps']
                    fitting = fit_neural(model, optimizer, counts, updates)
                    neural_steps += updates
                    assert neural_steps <= CONFIG['max_training_neural_steps']
                    qhat = model.q_values()
                snapshots.append(theta.copy())
                queries.append(qhat.copy())
                visitations.append(visits.copy())
                counts_all.append(counts.copy())
                model_before = None if model is None else parameter_sha(model)
                weight = 0. if arm == 'neural_diagonal' else CONFIG['off_weight']
                dl, surrogate, dg, og, gradient = base.loss_gradient(theta, qhat, visits, weight)
                theta = base.project(theta - CONFIG['ett_lr'] * gradient)
                assert np.isfinite([dl, surrogate, *gradient.ravel()]).all()
                if model is not None:
                    assert all(not p.requires_grad for p in model.parameters())
                    assert parameter_sha(model) == model_before
                history.append(dict(round=step, diagonal_nll=dl, surrogate=surrogate,
                                    diagonal_gradient_norm=np.linalg.norm(dg), off_gradient_norm=np.linalg.norm(og),
                                    critic=fitting, sampling_seed=rng_seed,
                                    rng_after=hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True).encode()).hexdigest(),
                                    counts_sha256=array_sha(counts), visitation_sha256=array_sha(visits),
                                    **base.assert_feasible(theta)))
            path = checkpoint_dir / f'{arm}_s{seed}.npz'
            np.savez_compressed(path, theta=theta, initial=initial, snapshots=snapshots,
                                q_history=queries, visits=visitations, counts=counts_all)
            record = dict(seed=seed, arm=arm, checkpoint_sha256=base.sha(path), history=history,
                          initial_neural_sha256=initial_model_sha)
            if model is not None:
                neural_path = checkpoint_dir / f'{arm}_s{seed}.pt'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), neural_path)
                record['neural_checkpoint_sha256'] = base.sha(neural_path)
            records.append(record)
            print(f'{arm} seed={seed}: {CONFIG["rounds"]} rounds completed', flush=True)
    base.write(out / 'training.json', dict(records=records, transitions=transitions,
               neural_steps=neural_steps, elapsed_seconds=time.perf_counter() - started,
               complete=True, exact_values_used=False, oracle_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=['prepare', 'train'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else train)(args.out)
