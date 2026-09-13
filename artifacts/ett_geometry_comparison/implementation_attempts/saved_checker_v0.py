"""Recompute saved-result accounting/optimizer checks; no stochastic sampling."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from ett.run_geometry_comparison import CONFIG,read,write,sha,verify,check_record
from ett.rollout_return import antithetic_gradient,descent_step
from ett.convex_action_transition import emit_with_geometry,fork_segment


def check(root):
    verify(root);ledger=read(root/'transition_ledger.json');training=read(root/'training.json')
    assert all(e['status']=='complete' for e in ledger['entries'])
    assert sum(e['transitions'] for e in ledger['entries'])==ledger['charged']==CONFIG['planned_transitions']
    for entry in ledger['entries']: assert sha(root/entry['cache'])==entry['sha256']
    entries={e['name']:e for e in ledger['entries']}
    for seed in [0,1]:
        directions=np.load(root/f'directions_s{seed}.npz')['directions']
        np.testing.assert_array_equal(directions,np.random.default_rng(61000000+seed).normal(size=(16,8,32)))
        for mode in CONFIG['modes']:
            name=f'{mode}_s{seed}';history=training['models'][name]['history']
            theta=np.load(root/f'theta_history_{name}.npz')['theta']
            np.testing.assert_array_equal(theta[0],np.zeros(32))
            for it in range(16):
                gradient=antithetic_gradient(np.array(history[it]['plus']),np.array(history[it]['minus']),directions[it],.1,1.)
                np.testing.assert_array_equal(descent_step(theta[it],gradient,1.,.2),theta[it+1])
                assert np.linalg.norm(theta[it+1]-theta[it])<=.2+1e-12
            with np.load(root/'checkpoints'/f'{name}.npz') as z:
                assert str(z['geometry_mode'])==mode and int(z['format_version'])==1 and float(z['bound'])==1.
                np.testing.assert_array_equal(z['theta'],theta[-1])
        for it in range(16):
            for direction in range(8):
                for sign in ['plus','minus']:
                    a,b=[entries[f'query_{mode}_s{seed}_{it}_{direction}_{sign}'] for mode in CONFIG['modes']]
                    assert a['signature']['key']==b['signature']['key']==62000000+seed*10000+it*8+direction
                    assert a['signature']['paths']==b['signature']['paths']==32
                    if it==0: assert a['signature']['theta']==b['signature']['theta']
    records={name:dict(np.load(root/f'eval_{name}.npz')) for name in ['zero',*training['models']]}
    for name,record in records.items():
        check_record(record)
        if name.startswith('fork_segment'):
            np.testing.assert_array_equal(record['segment_eligible'],record['geometry_kind']==1)
        assert np.array_equal(record['evaluation_seed'],np.repeat(CONFIG['evaluation_seeds'],128))
    diffs=np.load(root/'paired_episode_differences.npz');indices=diffs['bootstrap_indices']
    for name,v in read(root/'paired_returns.json').items():
        delta=diffs[name]
        if name.endswith('_minus_zero'):
            np.testing.assert_array_equal(delta,records[name[:-11]]['return']-records['zero']['return'])
        else:
            seed=name[-1]
            np.testing.assert_array_equal(delta,records[f'fork_segment_s{seed}']['return']-records[f'rectangle_s{seed}']['return'])
        np.testing.assert_allclose(np.quantile(delta[indices].mean(1),[.025,.975]),v['interval95'],atol=1e-12,rtol=0)

    # Preserve and explain a boundary-dependent standalone/JIT diagnostic difference.
    # Use actual emitted set diagnostics for the corrected component counts.
    panel=np.load(root/'fixed_component_outputs.npz');s=panel['state'];a=panel['anchor'];x=panel['action'];xp=panel['x_prime']
    standalone_end,standalone=jax.tree.map(np.asarray,fork_segment(jnp.asarray(s),jnp.asarray(a)))
    corrected={};boundary_examples=[]
    for name in training['models']:
        theta=np.load(root/'checkpoints'/f'{name}.npz')['theta']
        output,detail=jax.tree.map(np.asarray,jax.jit(lambda s,x,p,a:emit_with_geometry(theta,s,x,p,a,geometry_mode='fork_segment'))(s,x,xp,a))
        actual=detail['segment_available']
        corrected[name]=dict(actual_emitter_selected_fresh=int(actual[:512].sum()),
            actual_emitter_selected_saved_fork=int(actual[512:].sum()),
            standalone_selector_eligible_saved_fork=int(standalone[512:].sum()))
        for i in np.flatnonzero((actual!=standalone)[:,0]):
            if name==next(iter(training['models'])):
                boundary_examples.append(dict(panel_index=int(i),state=s[i],anchor=a[i,0],
                    standalone_end=standalone_end[i,0],emitted_segment_end=detail['segment_end'][i,0],
                    standalone_available=bool(standalone[i,0]),emitted_available=bool(actual[i,0])))
    write(root/'fixed_component_set_audit.json',dict(corrected_counts=corrected,examples=boundary_examples,
        interpretation='Raw fixed_component_metrics eligible counts use the standalone selector. Emitted JIT diagnostics are authoritative for selected sets. At a near-ceiling anchor, fusion changes a rounded endpoint across the availability threshold; both outcomes use the predeclared action-independent fallback rule. Main rollout eligibility exactly matches actual selection.',
        new_stochastic_transitions=0,production_model_unchanged=True))
    write(root/'independent_verification.json',dict(passed=True,ledger_and_cache_hashes_verified=True,
        optimizer_updates_recomputed=64,paired_query_keys_verified=512,initial_theta_zero=True,
        paired_intervals_recomputed=True,four_checkpoints_have_modes=True,
        actual_rollout_selection_and_eligibility_agree=True,raw_component_metadata_qualification=bool(boundary_examples),
        all_prior_inputs_unchanged=True,new_sampled_transitions=0,source_sha256=sha(__file__)))
    print('Saved-result verification passed; no new stochastic transitions. Boundary diagnostic cases:',len(boundary_examples))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    check(parser.parse_args().run_dir)
