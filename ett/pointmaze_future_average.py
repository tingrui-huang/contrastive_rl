"""One sealed critic-only future-time Rao-Blackwellization experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

from ett import pointmaze_phase_sampling as prior
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from ett.finite_crl import write, sha

PREVIOUS = Path('artifacts/pointmaze_region_pilot/phase_sampling_s01_v1')
CONFIG = dict(base='331115090688ad3f38e1356c8c140b298a3f3282',
    arms=['sampled', 'averaged'], seeds=[0, 1], critic_seed=98000000,
    steps=1500, batch=256, lr=.003, gamma=.95, negative_weight=31,
    query_seed=117000000, time_seed=117000001, calibration_seed=118000000,
    surrogate_seed=119000000, validity_seed=120000000, bootstrap_seed=121000000,
    repeats=128, calibration_repeats=64, bootstrap_repeats=2000,
    perturbation_norm=.1, minimum_effect=.001, tolerance=2e-6,
    model_output_cap=1000000, native_steps=0, ett_updates=0, actor_updates=0)


def future_probabilities(records):
    """Exact conditional positive probability; r[:,t] is reward of s[t+1]."""
    probabilities=[]
    for record in records:
        reward=np.asarray(record['reward'], np.float64)
        assert np.isin(reward, [0, 1]).all()
        p=np.empty_like(reward); numerator=np.zeros(len(reward)); denominator=0.
        for t in range(reward.shape[1]-1, -1, -1):
            numerator=reward[:, t]+.95*numerator
            denominator=1+.95*denominator
            p[:, t]=numerator/denominator
        probabilities.append(p.ravel())
    result=np.concatenate(probabilities)
    assert np.all((result>=0)&(result<=1))
    return result


def nce(logits, label, probability, arm):
    if arm=='sampled':
        positive=F.softplus(-logits).gather(1, label[:, None]).mean()
    elif arm=='averaged':
        positive=(probability*F.softplus(-logits[:, 1])+
                  (1-probability)*F.softplus(-logits[:, 0])).mean()
    else:
        raise ValueError(arm)
    return (positive+31*F.softplus(logits).mean())/32


def fit(critic, data, probability, arm, seed, steps=1500):
    x=torch.from_numpy(data['x']); label=torch.from_numpy(data['label'])
    p=torch.from_numpy(probability); rng=np.random.default_rng(seed)
    opt=torch.optim.Adam(critic.parameters(), lr=.003)
    row_hash=hashlib.sha256(); counts=np.zeros(len(label), np.int64)
    for param in critic.parameters(): param.requires_grad_(True)
    for _ in range(steps):
        idx=prior.sample_rows(rng, data, 'uniform', 256)
        row_hash.update(idx.astype('<i8').tobytes()); np.add.at(counts, idx, 1)
        loss=nce(critic(x[idx]), label[idx], p[idx], arm)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for param in critic.parameters(): param.requires_grad_(False)
    with torch.no_grad():
        logits=critic(x); soft=F.softplus(-logits)
        variance=p*(1-p)*(soft[:, 1]-soft[:, 0])**2/32**2
    return opt, dict(final_minibatch_nce=float(loss.detach()), steps=steps,
        row_stream_sha256=row_hash.hexdigest(), exposures=counts,
        mean_conditional_sampled_loss_variance=float(variance.mean()),
        averaged_conditional_future_time_variance=0.)


def candidates():
    result={'base':np.zeros(48, np.float32)}
    for name, matrix in [('identity', np.eye(2)), ('rotation', np.array([[0., -1.], [1., 0.]]))]:
        direction=np.tile(matrix, (8, 1, 1)).ravel()
        direction=.1*direction/np.linalg.norm(direction)
        for sign, tag in [(1, 'plus'), (-1, 'minus')]:
            result[name+'_'+tag]=np.r_[np.zeros(16), sign*direction].astype(np.float32)
    return result


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def prepare(out):
    if out.exists(): raise ValueError('Fresh output directory required.')
    prior.verify(PREVIOUS)
    sources=['ett/pointmaze_future_average.py', 'ett/eval_pointmaze_future_average.py',
             'scripts/test_pointmaze_future_average.py', 'notes/pointmaze_future_average.md']
    inherited=json.loads((PREVIOUS/'preregistration.json').read_text())['sources']
    files=[PREVIOUS/f for f in ['shared_training_paths.npz', 'shared_training_rows.npz',
        'final_contexts.npz', 'training.json', 'checkpoints/uniform_s0.pt', 'checkpoints/uniform_s1.pt']]
    hashes={**inherited, **{str(f):sha(f) for f in sources+files}}
    out.mkdir(parents=True); (out/'checkpoints').mkdir()
    write(out/'config.json', CONFIG)
    (out/'PROTOCOL.md').write_bytes(Path(sources[-1]).read_bytes())
    np.savez_compressed(out/'candidates.npz', **candidates())
    write(out/'preregistration.json', dict(utc=timestamp(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources=hashes, config_sha256=sha(out/'config.json'),
        protocol_sha256=sha(out/'PROTOCOL.md'), candidates_sha256=sha(out/'candidates.npz')))
    print('Protocol, candidates, budgets and source/input hashes sealed.', flush=True)


def verify(out):
    declaration=json.loads((out/'preregistration.json').read_text())
    assert json.loads((out/'config.json').read_text())==CONFIG
    for f, key in [('config.json', 'config_sha256'), ('PROTOCOL.md', 'protocol_sha256'),
                   ('candidates.npz', 'candidates_sha256')]:
        assert sha(out/f)==declaration[key], f
    for f, digest in declaration['sources'].items(): assert sha(f)==digest, f


def run(out):
    from ett import eval_pointmaze_future_average as evaluation
    verify(out)
    if (out/'started.json').exists(): raise ValueError('Existing attempt protected.')
    write(out/'started.json', dict(utc=timestamp()))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    records=prior.load_records(PREVIOUS/'shared_training_paths.npz')
    data=dict(np.load(PREVIOUS/'shared_training_rows.npz'))
    context=dict(np.load(prior.PREVIOUS/'contexts.npz'))
    replay=prior.training_rows(records, context['train_indices'][:, 2], context['mean'], context['std'])
    for key in data: np.testing.assert_array_equal(data[key], replay[key])
    p=future_probabilities(records)
    np.savez_compressed(out/'training_probabilities.npz', p=p)
    models={}; infos={}; hashes={}
    for seed in CONFIG['seeds']:
        paired_initial=None; paired_rows=None
        for arm in CONFIG['arms']:
            name=f'{arm}_s{seed}'; model=early.Critic(CONFIG['critic_seed']+seed)
            initial=old.parameter_sha(model)
            if paired_initial is None: paired_initial=initial
            assert initial==paired_initial
            opt, info=fit(model, data, p, arm, CONFIG['critic_seed']+seed)
            if paired_rows is None: paired_rows=info['row_stream_sha256']
            assert info['row_stream_sha256']==paired_rows
            final=old.parameter_sha(model)
            if arm=='sampled':
                previous=early.Critic(0)
                previous.load_state_dict(torch.load(PREVIOUS/'checkpoints'/f'uniform_s{seed}.pt', weights_only=True)['model'])
                assert old.parameter_sha(previous)==final, 'Baseline replay mismatch'
            info.update(initial_sha256=initial, final_sha256=final)
            np.savez_compressed(out/f'{name}_exposures.npz', count=info.pop('exposures'))
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), out/'checkpoints'/f'{name}.pt')
            models[name]=model; infos[name]=info; hashes[name]=sha(out/'checkpoints'/f'{name}.pt')
            print(name+': 1500 steps complete; paired initialization and rows verified.', flush=True)
    write(out/'training.json', dict(utc=timestamp(), models=infos, checkpoint_hashes=hashes,
        total_steps=6000, paired_baselines_reproduced=True, training_rows=len(p),
        mean_label_variance=float(np.mean(p*(1-p))), ett_updates=0, actor_updates=0))
    write(out/'environment.json', dict(python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, torch=torch.__version__))
    evaluation.evaluate(out, models, context)
    verify(out)
    for name, digest in hashes.items(): assert sha(out/'checkpoints'/f'{name}.pt')==digest
    write(out/'postrun_checks.json', dict(utc=timestamp(), sealed_inputs_unchanged=True,
        checkpoints_unchanged=True, baseline_replay=True, paired_rows=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=['prepare', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args(); (prepare if args.phase=='prepare' else run)(args.out)
