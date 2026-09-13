"""Deterministic convex-set witnesses; no model/simulator transitions.

These constructions certify existence at fixed recorded conditioning. They do
not supply supervised labels or a jointly fitted shared coefficient array.
"""
import argparse
from pathlib import Path
import numpy as np

from scripts.audit_ett_fork_structure import read, write, sha, HIGH, LOW


def segment_witness(row):
    q, a, d = [np.array(row[k], float) for k in ['q', 'anchor', 'action_difference']]
    r = np.linalg.norm(d)
    xl, xh = max(1., q[0]-1), min(float(HIGH[2, 0]), q[0]+1)
    bottom = max(1., q[1]-1)
    h = a[1]-3
    assert h > 0 and row['full_map_can_enter'] and row['upper_rectangle']
    if a[0] <= xh:
        assert a[0] >= xl
        eta = min((3-bottom)/2, (r-h)/2)
        target = np.array([a[0], 3-eta])
    else:
        dx = a[0]-xh
        corner_length = np.hypot(dx, h)
        # Extend the ray A -> guarded inner corner a little into the vertical
        # passage. Every segment point is in the top or vertical rectangle.
        eta = min((3-bottom)/2, (xh-xl)*h/(2*dx), h*(r/corner_length-1)/2)
        target = np.array([xh-dx*eta/h, 3-eta])
    assert eta > 0 and target[1] < 3
    displacement = target-a
    matrix = np.outer(displacement, d)/r**2
    bound = np.linalg.norm(matrix, 'fro')
    assert bound < 1
    # Along the line, F(u)=A+clip(d dot (u-xp)/r^2,0,1)*(target-A).
    # This is exactly projection_[A,target](A+M(u-xp)), not a switched box.
    def mapping(action):
        coefficient = np.clip((np.asarray(action)-row['nominal_action'])@d/r**2, 0, 1)
        return a+coefficient[..., None]*displacement
    np.testing.assert_allclose(mapping(row['nominal_action']), a, atol=1e-12, rtol=0)
    np.testing.assert_allclose(mapping(row['execution_action']), target, atol=1e-12, rtol=0)
    points = a+np.linspace(0, 1, 1001)[:, None]*displacement
    step_low = (q.astype(np.float32)-np.float32(1)).astype(float)
    step_high = (q.astype(np.float32)+np.float32(1)).astype(float)
    in_maze = np.any(np.all((points[:, None] >= LOW) & (points[:, None] <= HIGH), axis=-1), axis=-1)
    assert in_maze.all() and (points >= step_low).all() and (points <= step_high).all()
    strict_step_max = np.max(np.abs(points-q))
    # Check segment inclusion analytically, separately from the interpolation.
    assert xl <= target[0] <= xh and bottom <= target[1] < 3
    if a[0] > xh:
        crossing = a+(3-a[1])/(target[1]-a[1])*displacement
        assert abs(crossing[0]-xh) < 1e-12
        assert a[1] <= float(HIGH[0, 1])
    else:
        assert xl <= a[0] <= xh
    return dict(model_seed=row['model_seed'], reset_seed=row['reset_seed'], time=row['time'],
                q=q, anchor=a, xp=row['nominal_action'], execution_action=row['execution_action'],
                target=target, action_distance=r, segment_length=np.linalg.norm(displacement),
                matrix=matrix, matrix_frobenius=bound, L=1.,
                coefficient_choice='all eight existing matrix blocks set to this matrix; per-query existence only',
                segment_in_free_geometry_analytically_verified=True,
                step_envelope='production float32-rounded q +/- 1; exact constants in this mathematical witness',
                strict_unit_step_feasible=bool(strict_step_max <= 1),
                max_coordinate_step=float(strict_step_max),
                diagonal_preserved=True, full_domain_bound_proven='composition of norm<=1 matrix and fixed convex projection')


def run(root):
    source = root/'fork_traces.json'
    before = sha(source)
    assert not (root/'convex_segment_witnesses.json').exists()
    rows = read(source)
    eligible = [r for r in rows if r['upper_rectangle'] and r['full_map_can_enter']]
    witnesses = [segment_witness(r) for r in eligible]
    summary = {str(s): dict(upper_box_exclusions=sum(r['upper_rectangle'] for r in rows if r['model_seed']==s),
        certified_convex_segment_witnesses=sum(w['model_seed']==s for w in witnesses),
        strict_unit_step_witnesses=sum(w['model_seed']==s and w['strict_unit_step_feasible'] for w in witnesses),
        min_descent_depth=min(3-w['target'][1] for w in witnesses if w['model_seed']==s),
        max_matrix_frobenius=max(w['matrix_frobenius'] for w in witnesses if w['model_seed']==s)) for s in [0, 1]}
    # The Frobenius block constraint is stronger than an operator-norm bound.
    # P_C(A+(u-xp)) has derivative I around xp in an interior rectangle, yet
    # ||I||_F=sqrt(2)>1, so no current decoded matrix can equal that map there.
    operator_witness = dict(q=[1.5,3.5], anchor=[1.5,3.5], xp=[0.,0.],
        domain='[-1,1]^2', C_low=[.5,3.], C_high=[2.5,float(HIGH[0,1])],
        map='projection_C(anchor + action)', operator_norm=1., frobenius_norm=float(np.sqrt(2)),
        derivative_at_xp=[[1,0],[0,1]], excluded_by_current_matrix_family=True,
        note='This does not obstruct a single prescribed action response: rank-one matrices suffice there.')
    # A ball-feasible endpoint can fail the stronger intrinsic path condition.
    # Top corridor to vertical branch, fixed anchor (2.4,3.1), target (1.9,2.7).
    a=np.array([2.4,3.1]); target=np.array([1.9,2.7]); corner=np.array([float(HIGH[2,0]),3.])
    euclidean=np.linalg.norm(a-target)
    intrinsic=np.linalg.norm(a-corner)+np.linalg.norm(target-corner)
    assert euclidean < .68 < intrinsic
    obstacle = dict(current=[1.5,3.5], anchor=a, target=target,
                    xp=[0.,0.], execution_action=[.68,0.], action_radius=.68,
                    euclidean_distance=euclidean, shortest_free_path_length=intrinsic,
                    pointwise_ball_feasible=True, globally_1_lipschitz_extension_possible=False,
                    justification='Image of straight action segment must be a free curve of length <= .68.')
    write(root/'convex_segment_witnesses.json', witnesses)
    write(root/'convex_extension_summary.json', summary)
    write(root/'additional_counterexamples.json', dict(frobenius_vs_operator=operator_witness,
                                                       anchor_ball_vs_global_geometry=obstacle))
    # Visualize earliest anchor-outside-vertical case from each model seed.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1,2, figsize=(11,5), constrained_layout=True)
    for seed, ax in enumerate(axes):
        row = next(r for r in eligible if r['model_seed']==seed and not r['vertical_contains_anchor'])
        w = next(w for w in witnesses if (w['model_seed'],w['reset_seed'],w['time'])==(seed,row['reset_seed'],row['time']))
        ax.add_patch(Rectangle((2,1),2,2,facecolor='#CCCCCC'))
        ax.add_patch(Rectangle((0,1),1,2,facecolor='#CCCCCC'))
        lo,hi=np.array(row['box_low']),np.array(row['box_high'])
        ax.add_patch(Rectangle(lo,*(hi-lo),facecolor='#B5CCE0',alpha=.4,edgecolor='#2864A0',label='Current selected box'))
        q,a,p=[np.array(row[k]) for k in ['q','anchor','proposal']]
        target=np.array(w['target'])
        ax.plot([q[0],a[0]],[q[1],a[1]],'--',color='#666666',label='Saved diagonal displacement')
        ax.plot([a[0],p[0]],[a[1],p[1]],color='#2864A0',linewidth=3,label='Learned correction (unclipped here)')
        ax.plot([a[0],target[0]],[a[1],target[1]],color='#BC6221',linewidth=2,label='Certified alternative convex segment')
        for xy,label,dy in [(q,'Current',.08),(a,'Anchor',-.1),(p,'Emitted',.1),(target,'Witness',-.12)]:
            ax.scatter(*xy,color='black',s=20); ax.annotate(label,xy,xytext=(0,dy*100),textcoords='offset points',ha='center',fontsize=9)
        ax.axhline(3,color='gray',linewidth=.6)
        ax.set(xlim=(.8,3),ylim=(2.3,4.05),aspect='equal',xlabel='X (maze units)',ylabel='Y (maze units)',
               title=f'Seed {seed}: reset {row["reset_seed"]}, step t={row["time"]}')
        ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('Same saved anchor; a convex segment admits a turn excluded by the current box')
    fig.savefig(root/'fork_witnesses.png',dpi=170); plt.close(fig)
    assert sha(source)==before
    write(root/'convex_witness_verification.json', dict(saved_trace_hash_unchanged=True,
        source_sha256=sha(__file__), witness_count=len(witnesses),
        new_sampler_calls=0,new_rollouts=0,parameter_fitting_steps=0))
    print(summary)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    run(parser.parse_args().audit_dir)
