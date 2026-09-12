"""Independent saved-array and initial-diagonal checks; no optimization."""
import argparse
from pathlib import Path
import jax
import numpy as np

from ett.run_convex_adversarial import verify,setup,CONFIG
from ett.run_return_readout import read,write
from ett.eval_diagonal_transition import _open_endpoint
from scripts.make_swamp_f4_failure_bank import file_sha


def main(root):
    verify(root);model,_,_,_,_=setup();data=dict(np.load(root/'fit_contexts.npz'))
    s,a,g=[data['validation_'+k][:128] for k in ['state','action','goal']]
    key=jax.random.PRNGKey(19001000)
    old=np.asarray(model.diagonal.sample(s,a,a,key,32,goal=g))
    new=np.asarray(model.sample(s,a,a,key,32,goal=g))
    maximum=float(np.max(np.linalg.norm(old-new,axis=-1)))
    assert maximum<=2e-6
    one=model.sample(s[0],a[0],a[0],key,goal=g[0]);many=model.sample(s[0],a[0],a[0],key,3,goal=g[0])
    assert one.shape==(8,) and many.shape==(3,8)
    rng=np.random.default_rng(18500000)
    indices=np.concatenate([rng.integers(128,size=(500,128))+128*j for j in range(4)],axis=1)
    measured={};returns={};action_checks={}
    for name,arm in read(root/'training.json')['arms'].items():
        assert file_sha(root/'checkpoints'/f'{name}.npz')==arm['checkpoint_sha256']
        r=dict(np.load(root/f'{name}_rollouts.npz'));aux=dict(np.load(root/f'{name}_auxiliary.npz'))
        assert 'aux_x_prime' not in r and r['action'].shape==aux['aux_x_prime'].shape
        assert np.isfinite(r['states']).all() and _open_endpoint(r['states'][...,:2]).all()
        np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
        reconstructed=(np.linalg.norm(r['states'][:,1:,:2].astype(float)-[8.5,3.5],axis=-1)<2).astype(float)
        np.testing.assert_array_equal(reconstructed,r['reward'])
        recomputed=r['reward'].astype(float)@(.95**np.arange(50))
        np.testing.assert_allclose(recomputed,r['return'],atol=6e-6,rtol=1e-6)
        returns[name]=r['return']
        audit=dict(np.load(root/f'{name}_action_audit.npz'));out=audit['outputs'].astype(float)
        # Direct differences are stable even when float32 mean-centering an
        # exactly repeated coordinate would introduce spurious tiny variance.
        exact_invariant=np.all(out[:9]==out[0:1],axis=(0,3))
        spread=np.sqrt(np.sum(np.var(out[:9,:,:,:2],axis=0),axis=-1))
        action_checks[name]=dict(exact_grid_action_invariant_fraction=float(exact_invariant.mean()),
            float64_mean_grid_action_spread=float(spread.mean()),
            float64_threshold_action_invariant_fraction=float(np.mean(spread<1e-6)))
        measured[name]=dict(max_return_reconstruction_difference=float(np.max(np.abs(recomputed-r['return']))))
    paired=read(root/'paired_returns.json')
    for seed in [0,1]:
        delta=returns[f's{seed}_adversarial']-returns[f's{seed}_control']
        np.testing.assert_allclose(np.quantile(delta[indices].mean(1),[.025,.975]),paired[str(seed)]['interval95'],rtol=0,atol=0)
    verify(root)
    # 8192 full old/new samples plus 4 shape-check samples; formula tests use
    # fewer than the remaining 1804 samples in the declared 10000 reserve.
    write(root/'verification.json',dict(status='passed',initial_emitted_diagonal_max_difference=maximum,
        single_and_multiple_shape_checks=True,added_single_step_samples=8196,
        saved_rewards_and_history_verified=True,paired_intervals_reproduced_exactly=True,
        frozen_inputs_unchanged=True,executed_and_auxiliary_actions_separate=True,
        action_variance_numerical_check=action_checks,return_reconstruction=measured,
        note='Float64 spread/direct equality supersedes float32 variance near zero; no new trajectories or selection.',
        source_sha256=file_sha(__file__)))
    print('Initial diagonal identity, saved arrays, rewards, pairing and frozen hashes verified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);main(Path(p.parse_args().run_dir))
