"""Fixed-bank scalar experiment: no action-conditioned next-state training labels."""
import argparse
import hashlib
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from scripts.synthetic_lipschitz_response import initialize, array_sha, read
from scripts.synthetic_shared_response import sha, write_json


CONFIG = dict(bank=[-3.], lambda_off=.1, seeds=[0,1,2], arms=['diagonal_only','bank_joint'],
    steps=2500, learning_rate=.001, final_learning_rate=.0001, diagonal_batch=128, off_batch=256,
    train_diagonal=8192, train_off=32768, eval_diagonal=4096, eval_off=16384,
    train_seed_base=920000, evaluation_seed=921000, monitor_seed=922000, batch_seed_base=840000,
    monitor_every=100, grid_size=201, eval_pairs=32768, slice_anchors=[-.83,-.27,.19,.74],
    side_offsets=[.0001,.001,.01,.04], min_separation=.001, ratio_tolerance=.01,
    fp_absolute_excess_tolerance=1e-12, quadrature_nodes=[128,256], threads=1)


def sample_data(seed, n_diag, n_off):
    """Identical empirical xp marginal: every diagonal context gets n_off/n_diag x draws."""
    if n_off % n_diag:
        raise ValueError('off count must be an integer multiple of diagonal contexts')
    rng = np.random.default_rng(seed)
    xp = rng.uniform(-1,1,n_diag)
    repeated = np.repeat(xp,n_off//n_diag)
    # xp is already chosen. Only x is drawn (or redrawn in the zero-probability tie case).
    x = rng.uniform(-1,1,n_off)
    equal = x==repeated
    while equal.any():
        x[equal] = rng.uniform(-1,1,int(equal.sum()))
        equal = x==repeated
    return np.stack((xp,xp),-1), np.stack((x,repeated),-1)


def bank_losses(model, diagonal, off, bank, weight):
    """Exactly two means. Reference values are unconditional and detached."""
    bank = torch.as_tensor(bank,dtype=off.dtype,device=off.device).detach().reshape(-1)
    if not len(bank) or not torch.isfinite(bank).all():
        raise ValueError('finite nonempty fixed failure bank required')
    diag_loss = model(diagonal).square().mean()
    outputs = model(off)
    off_loss = (outputs[:,None]-bank[None,:]).square().min(dim=1).values.mean()
    return diag_loss + weight*off_loss, diag_loss, off_loss


def state_sha(model):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def train(arm,seed,config,data,monitor):
    model = initialize(seed)
    initial = {k:v.clone() for k,v in model.named_parameters()}
    initial_hash = state_sha(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['steps'],eta_min=config['final_learning_rate'])
    diagonal,off = [torch.from_numpy(a) for a in data]
    md,mo = [torch.from_numpy(a) for a in monitor]
    bank = torch.tensor(config['bank'],dtype=torch.float64)
    assert not bank.requires_grad
    weight = config['lambda_off'] if arm=='bank_joint' else 0.
    rng = np.random.default_rng(config['batch_seed_base']+seed)
    schedule = hashlib.sha256();history=[];start=time.perf_counter()
    for step in range(config['steps']+1):
        if step%config['monitor_every']==0 or step==config['steps']:
            with torch.no_grad():
                total,d,o = bank_losses(model,md,mo,bank,weight)
            history.append(dict(step=step,training_total=float(total),diagonal_mse=float(d),
                failure_mse=float(o),common_lambda01_total=float(d+config['lambda_off']*o)))
        if step==config['steps']:break
        di=rng.integers(len(diagonal),size=config['diagonal_batch'])
        oi=rng.integers(len(off),size=config['off_batch'])
        schedule.update(di.tobytes());schedule.update(oi.tobytes())
        optimizer.zero_grad(set_to_none=True)
        total,_,_=bank_losses(model,diagonal[di],off[oi],bank,weight)
        total.backward()
        if not torch.isfinite(total) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('nonfinite loss/gradient')
        optimizer.step();scheduler.step()
    changed={k:bool(torch.any(v!=initial[k])) for k,v in model.named_parameters()}
    # The diagonal-only control need not update right interval coefficients.
    return model,history,dict(training_seconds=time.perf_counter()-start,initial_state_sha256=initial_hash,
        batch_schedule_sha256=schedule.hexdigest(),data_sha256=[array_sha(a) for a in data],
        changed_parameter_tensors=changed,all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
        parameter_count=sum(p.numel() for p in model.parameters()),bank_has_gradient=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();out=Path(args.out_dir)
    if out.exists():raise ValueError('use a fresh output directory; preserve old artifacts')
    config=dict(CONFIG,smoke=args.smoke)
    if args.smoke:config.update(steps=5,seeds=[0],monitor_every=1)
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    config.update(dtype='float64',device='cpu',source_sha256=sha(__file__),
        architecture_source_sha256=sha('scripts/synthetic_lipschitz_response.py'),
        python_version=platform.python_version(),numpy_version=np.__version__,torch_version=str(torch.__version__),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initialization='existing ConditionalSpline random initialization, exactly matched within each seed; no supervised checkpoint',
        sampling='uniform xp contexts shared equally by both empirical marginals; four independent uniform x draws per context; no boundary rejection or clipping',
        objective='mean(G(diagonal)^2)+0.1*mean(min_f(G(off)-f)^2); no correct off next-state labels',
        evaluation_policy='fixed final step; no selection/tuning; independent reference used only for evaluation',
        historical_comparison='context only; sampling differs from the former balanced near/far distribution',
        output_policy='final network output only; no pointwise candidate selector or recentering wrapper')
    old_files=[]
    for folder in ['artifacts/synthetic_shared_response','artifacts/synthetic_lipschitz_response']:
        old_files.extend(p for p in Path(folder).rglob('*') if p.is_file())
    old_files.extend(Path(p) for p in ['scripts/synthetic_lipschitz_response.py','scripts/synthetic_shared_response.py',
        'ett/failure_objectives.py','scripts/make_swamp_f4_failure_bank.py','scripts/diag_offdiag_v0.py'])
    original_hashes={p.as_posix():sha(p) for p in old_files}
    out.mkdir(parents=True)
    # Save the derivation before the first gradient update. It never enters train().
    from scripts.synthetic_failure_reference import DERIVATION
    (out/'REFERENCE.md').write_text(DERIVATION,encoding='utf-8')
    config['reference_derivation_sha256']=sha(out/'REFERENCE.md')
    config['reference_source_sha256']=sha('scripts/synthetic_failure_reference.py')
    write_json(out/'config.json',config);write_json(out/'original_hashes.json',original_hashes)
    monitor=sample_data(config['monitor_seed'],1024,4096)
    evaluation_data=sample_data(config['evaluation_seed'],config['eval_diagonal'],config['eval_off'])
    results={};histories={};checkpoints={};start=time.perf_counter()
    for seed in config['seeds']:
        data=sample_data(config['train_seed_base']+seed,config['train_diagonal'],config['train_off'])
        for arm in config['arms']:
            name=f'{arm}_s{seed}'
            model,history,detail=train(arm,seed,config,data,monitor)
            # Evaluation is deliberately separate from all optimizer inputs.
            from scripts.eval_synthetic_failure_bank import evaluate
            metrics,arrays=evaluate(model,config,evaluation_data)
            metrics.update(detail,arm=arm,seed=seed)
            results[name]=metrics;histories[name]=history
            torch.save(dict(state_dict=model.state_dict(),arm=arm,seed=seed,config=config),out/f'{name}.pt')
            checkpoints[name]=sha(out/f'{name}.pt')
            np.savez_compressed(out/f'{name}_evaluation.npz',**arrays)
            write_json(out/'results.json',results);write_json(out/'history.json',histories)
            print(f'{name}: diag RMSE={metrics["diagonal"]["normalized_rmse"]:.5f}; '
                f'bank loss={metrics["failure_mse"]:.5f}; population gap={metrics["quadrature256"]["population_objective_gap"]:.6g}',flush=True)
    assert all(sha(p)==h for p,h in original_hashes.items())
    write_json(out/'completion.json',dict(status='complete',new_runs=len(checkpoints),
        elapsed_seconds=time.perf_counter()-start,checkpoints=checkpoints,old_artifacts_unchanged=True))


if __name__=='__main__':main()
