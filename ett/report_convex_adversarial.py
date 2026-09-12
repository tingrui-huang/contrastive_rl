"""Report the completed constrained fixed-actor prototype, without fitting."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ett.run_convex_adversarial import verify,CONFIG
from ett.run_return_readout import read,write
from ett.diagonal_transition import POINTMAZE_WALLS
from scripts.make_swamp_f4_failure_bank import file_sha


def maze(ax):
    for x,y in np.argwhere(POINTMAZE_WALLS):ax.add_patch(Rectangle((x,y),1,1,facecolor='lightgray',edgecolor='white'))
    ax.scatter(8.5,3.5,c='goldenrod',marker='*',s=50)
    ax.set(xlim=(0,9),ylim=(0,5),xlabel='X',ylabel='Y',aspect='equal')


def main(root):
    verify(root);m=read(root/'metrics.json');paired=read(root/'paired_returns.json');train=read(root/'training.json')
    before=read(root/'diagonal_before.json');completion=read(root/'completion.json');verification=read(root/'verification.json')
    assert verification['status']=='passed'
    names=['s0_control','s0_adversarial','s1_adversarial'];labels=['Control','Adversarial seed 0','Adversarial seed 1']
    records={n:dict(np.load(root/f'{n}_rollouts.npz')) for n in names}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar(range(3),[m[n]['mean_return'] for n in names]);axes[0].set(xticks=range(3),xticklabels=labels,ylabel='Discounted model return',title='Independent model rollouts')
    for i in range(2):
        p=paired[str(i)];axes[1].plot(i,p['adversarial_minus_control'],'o');axes[1].vlines(i,*p['interval95'])
    axes[1].axhline(0,c='gray',ls='--');axes[1].set(xticks=[0,1],xticklabels=['Seed 0','Seed 1'],ylabel='Adversarial minus control',title='Paired 95% model-path intervals')
    fig.savefig(root/'independent_returns.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for n,label in zip(names,labels):
        axes[0].plot(np.mean(records[n]['reward'],axis=0),label=label)
        axes[1].plot(np.mean(records[n]['boundary'],axis=0),label=label)
    axes[0].set(xlabel='Model step',ylabel='Reward probability',title='Reward timing');axes[1].set(xlabel='Model step',ylabel='On selected box boundary',title='Boundary behavior')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(root/'reward_and_boundary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(13,8),layout='constrained')
    indices=[0,1,2,3]
    for row,(n,label) in enumerate(zip(names,labels)):
        for col,index in enumerate(indices):
            ax=axes[row,col];maze(ax);xy=records[n]['states'][index,:,:2]
            ax.plot(xy[:,0],xy[:,1],lw=1);ax.scatter(*xy[0],marker='x',c='black')
            ax.set_title(f'{label}, path {index}\nreturn {records[n]["return"][index]:.2f}',fontsize=9)
    fig.savefig(root/'paired_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for row,n in enumerate(['s0_adversarial','s1_adversarial']):
        a=dict(np.load(root/f'{n}_action_audit.npz'))
        for col,c in enumerate([0,1,2,3]):
            ax=axes[row,col];anchor=a['anchor'][c,0];out=a['outputs'][:9,c,0,:2]
            for point in out:ax.plot([anchor[0],point[0]],[anchor[1],point[1]],color='gray',alpha=.5)
            colors=ax.scatter(out[:,0],out[:,1],c=np.arange(9),cmap='viridis');ax.scatter(*anchor,marker='x',c='red')
            ax.set(xlabel='New X',ylabel='New Y',title=f'{n}, context {c}');ax.set_aspect('equal',adjustable='datalim')
    colorbar=fig.colorbar(colors,ax=axes.ravel().tolist(),ticks=np.arange(9),shrink=.8)
    colorbar.ax.set_yticklabels([str((x,y)) for x in [-1,0,1] for y in [-1,0,1]])
    colorbar.set_label('Execution action (x0, x1); red cross = anchor')
    fig.savefig(root/'signed_action_responses.png',dpi=160);plt.close(fig)
    lines=['# Constrained fixed-actor ETT adversarial prototype','',
        '**The predefined within-class success criterion passes in both optimizer seeds. Diagonal outputs remain unchanged and independent model-rollout return decreases. This does not yet justify actor training.**','',
        '## Scope and change from earlier work','',
        '[RAMBO-RL](https://arxiv.org/html/2204.12581v3#S5) motivates pairing offline model fitting with adversarial model-value minimization. This prototype borrows that structure, while keeping the actor frozen and replacing its likelihood-ratio/value-estimator machinery with paired parameter perturbations and direct Monte Carlo. Its general policy and robustness guarantees are not claimed here.','',
        'The previous six-parameter search already used hard multi-step return. The change here is the admissible model class: 32 coefficients control eight context-dependent 2x2 signed-action matrices, followed by action-independent convex-box geometry. This replaces a distance-to-anchor bound and an action-dependent fallback with a continuum all-action-pairs argument. The feature family is different, not a strict superset of the old MLP. L, parameterization and gradient scaling differ from the old experiment, so its effect size is not a controlled architecture comparison.','',
        'The synthetic spline work showed why all-pairs slope control must be distinguished from a prescribed diagonal anchor; adversarial diagonal-budget work showed that a small soft weight can still miss a tight fidelity budget. Here the complete diagonal law is frozen by construction. This explicitly sacrifices shared-network diagonal learning: the first loss is constant, and only the second term updates the model.','',
        '## Specification and frozen inputs','',
        'The [pre-training specification](SPEC.md), [configuration](config.json) and [hash provenance](provenance.json) were saved before training. Actor: historical stochastic alpha0_seed0 final checkpoint; nominal: existing state-goal k=5 teacher-source MDN, fitted on episodes 1200..5999 (noisy teacher behavior including failures). The base is the preserved zero-residual guarded diagonal control used by the old return search. No alpha sweep, critic/readout fitting, new simulator rollout or actor update occurs. Reports from the independent continuation task were reviewed; its evaluation arrays are not consumed.','',
        'At every step, x_prime is sampled from the frozen nominal at current F4 and commanded goal, x from the frozen actor, and the successor from the current ETT. Both policies react to the current generated state. x_prime is never selected adversarially, reweighted or given to the actor; it is saved only in separate auxiliary files. Its nominal-marginalized transition is a learned rollout kernel, not identified interventional ground truth.','',
        'All training and evaluation paths start at four copies of (0.5,3.5), with goal (8.5,3.5), H=50 and discount .95. Reward remains the audited strict radius-2 position indicator at the new position; no reward model, shaping, hindsight goal or value estimator is used. Every number below is model-generated return. No new real-simulator return was measured, and older continuation contexts are not a matched real-return comparison.','',
        '## Exactly two terms, with a valid sampler objective','',
        '`L_ETT = L_diag + lambda * E[sum_t .95^t r_g(s_(t+1))]`','',
        'The emitted diagonal fitting score is `E||Y-y_obs|| - .5 E||Y-Y_tilde||`, evaluated at x=x_prime=a with 32 draws and an unbiased off-self-pair U-statistic. This energy score is in maze XY units and directly uses the actual projected mixed law, including atoms. F4 history is checked independently as a deterministic identity. Old unprojected zero-atom/Gaussian NLL is recorded only as a backbone diagnostic. It is not asserted to be the output density.','',
        'Projection creates boundary mass, and older F4 coordinates have a singular law. No valid emitted off-diagonal density is implemented, so `log_prob` explicitly raises. No pre-projection likelihood-ratio gradient or hard-indicator pathwise gradient is used. Antithetic N(0,I_32) perturbations estimate the gradient of the Gaussian-smoothed complete-rollout objective, including discrete draws and geometry effects. Paired signs share full random streams but each perturbed model generates its own on-model states/actions. No frozen scorer is treated as a value oracle. Value-estimation error is therefore not applicable.','',
        f'Lambda is fixed at 1/sum(.95^t) = {CONFIG["weights"][1]:.8f}, scaling J into [0,1]. With the diagonal law invariant, this controls update scale rather than a diagonal/return tradeoff. Each of optimizer seeds 0/1 compares lambda=0 to that single positive weight, from exactly zero coefficients, using matched query streams. Sixteen updates use eight directions, sigma=.10, 32 paths per signed query, learning rate 1 and update norm cap .20. The final iterate is always used. Monitors never select checkpoints; the two control parameter arrays and independent evaluation paths are identical.','',
        '## What the constraints establish','',
        'For each fixed s,x_prime,g and coupled anchor noise, normalized matrix blocks have Frobenius norm <=1; their nonnegative softmax combination has operator norm <=1. The anchor and chosen rectangle do not depend on execution action. Euclidean rectangle projection is nonexpansive, so the mathematically defined emitted F4 map satisfies `||T(x1)-T(x2)||_2 <= ||x1-x2||_2` for every action pair. Since the anchor lies inside the rectangle, x=x_prime reproduces it. All original anchor mixture/projection branches precede x and no action-dependent fallback follows it. The proof also induces a one-step distributional W2 upper bound under the same coupling, for fixed conditioning.','',
        'L=1 is prescribed, not estimated from diagonal data or certified for true structural dynamics. The argument is for the real-arithmetic emitted map including projections; floating-point checks use 2e-6 absolute tolerance, not a formal bit-level global certificate. Finite random/grid/near-pair tests support implementation correctness but do not supply the global proof.','',
        'Guaranteed geometry means free endpoints, original coordinate bounds, per-coordinate motion <=1, membership in the selected free convex rectangle and exact F4 shift. It does not establish simulator reachability, hidden absorption or causal validity. A rectangle containing the diagonal anchor need not contain the previous position. Preserving that imperfect anchor exactly retains rare corner-crossing possibilities; this limitation was declared before training, not relaxed after seeing results.','',
        '## Independent evaluation and diagonal acceptance','']
    for seed in [0,1]:
        p=paired[str(seed)];lines.append(f'- Seed {seed}: control return {m[f"s{seed}_control"]["mean_return"]:.5f}, adversarial {m[f"s{seed}_adversarial"]["mean_return"]:.5f}; paired difference {p["adversarial_minus_control"]:.5f}, 95% interval [{p["interval95"][0]:.5f}, {p["interval95"][1]:.5f}].')
    lines+=['',
        'Each arm has four independent evaluation seeds with 128 paths each. The 500 paired bootstrap replicates resample trajectories within these seed groups. Both intervals lie below zero, meeting the predeclared return criterion. Seed 0 has one small positive group difference (+0.01575); seed 1 decreases in every group. These intervals condition on trained parameters and cover model Monte Carlo variation, not optimizer-seed uncertainty or real-system correctness.','',
        '![Independent return comparison](independent_returns.png)','',
        f'Diagonal coupled output drift and maximum per-tuple energy-score increase are exactly 0 on all 2,048 training and 2,048 held-out tuples for every arm, within budgets 2e-6 and 1e-6. The held-out energy score stays {before["validation"]["energy_mean"]:.8f}. Held-out predictive-mean XY L2 errors before and after have median/p95/p99/max {before["validation"]["predictive_mean_l2_quantiles"]}; the near-1 maximum shows existing tail error despite excellent mean fit. Training energy remains {before["train"]["energy_mean"]:.8f}. The unprojected held-out diagnostic NLL remains {before["validation"]["unprojected_mixed_nll_mean"]:.6f}; it is not the emitted likelihood.','',
        'Initialization was independently checked against the original diagonal sampler; the maximum difference was '+str(verification['initial_emitted_diagonal_max_difference'])+'. Global diagonal identity follows from the construction, not from these finite rows alone.','',
        '## Action response, boundary behavior and remaining failure modes','']
    for n,label in zip(names,labels):
        a=m[n]['action_audit'];v=verification['action_variance_numerical_check'][n]
        lines.append(f'- {label}: maximum audited action ratio {a["maximum_sampled_ratio"]:.4f}; positive bound excess {a["maximum_positive_action_bound_excess"]:.2g}; exact action-invariant grid fraction {v["exact_grid_action_invariant_fraction"]:.2%}; mean action spread {v["float64_mean_grid_action_spread"]:.4f}.')
    lines+=['',
        'The baseline is exactly invariant to execution action. Direct equality and float64 variance checks in verification.json clarify small float32 centering artifacts in the original near-zero spread summary. Both adversarial models show signed X/Y responses; their maximum matrix Frobenius norms are 0.331/0.422, below prescribed L=1. Observed ratios 0.204/0.285 are empirical lower information, not estimated global constants. Common response-energy fractions on this grid are 11.4%/8.5%; this grid differs from the old audit, so those percentages are not a like-for-like mechanism comparison. A common drift would not itself violate the class.','',
        '![Signed action responses at first four held-out contexts](signed_action_responses.png)','',
        'Zero-return paths rise from 5.08% in control to 12.50%/9.38%. Mean step motion falls from .3616 to .3369/.3401. The paths do not collapse to uniformly zero or action-invariant outputs; XY trajectory spread increases from 1.018 to 1.254/1.390.','',
        'However, exact stationary steps fall from 8.79% to 0% in both adversarial arms. Off-diagonal corrections can move an anchor atom, so preserving the diagonal does not preserve absorbing semantics away from x=x_prime. The adversarial paths contain no fully stationary non-reset F4 contexts, making their conditional renewed-motion statistic undefined, not evidence of correct absorption. The control itself resumes motion on about 7.39% of its stationary-history steps.','',
        'Box clipping affects 10.17%/9.33% of adversarial rollout steps versus 0% for control; box-boundary occupancy rises from 3.80% to 10.17%/9.33%. Exact final-ten-step freezing disappears, but near-boundary motion and oscillation remain possible. Sampled 21-point straight-segment wall hits occur on 0.0273% of control steps and 0.0508%/0.0352% of adversarial steps, despite valid endpoints. Such interpolation is not the native axiswise simulator path, but it exposes the limit of endpoint feasibility.','',
        '![Reward and boundary timing](reward_and_boundary.png)','',
        '![First four common-randomness evaluation paths; lines connect macro-step endpoints](paired_trajectories.png)','',
        '## Decision and reproducibility','',
        '**The bounded prototype succeeds only in its explicitly restricted model class. Stop before actor training.** The concrete issue is transition validity beyond endpoint geometry: diagonal preservation and global action Lipschitzness do not establish off-diagonal absorption, physically reachable paths or calibrated multi-step behavior. Those deficiencies can contaminate actor learning even when pessimistic optimization works. No alpha=.5, actor update, new value fit, simulator-target training or follow-up family was launched.','',
        f'Completed {completion["primary_model_transitions"]:,} primary model transitions and {completion["additional_model_transitions"]:,} fit/audit/replay transitions, plus a 10,000-sample test reserve (including the independent 8,196-sample initialization/shape check), below the {CONFIG["total_transition_cap"]:,} cap. Search/monitoring took {train["elapsed_seconds"]:.2f} seconds after initial compilation and fit checks on local CPU. Five focused tests pass; 512 saved full evaluation paths replay exactly. Saved rewards, F4, action bounds, paired intervals and frozen hashes independently verify. All checkpoints stay in ignored checkpoints/. Reports and evaluation arrays are English and publishable.','',
        '```bash','python -m unittest scripts.test_convex_action_transition',
        'python -m ett.run_convex_adversarial --out-dir artifacts/ett_convex_adversarial/l1_matrix32_s01_v1 --phase prepare',
        'python -m ett.run_convex_adversarial --out-dir artifacts/ett_convex_adversarial/l1_matrix32_s01_v1 --phase train',
        'python -m ett.eval_convex_adversarial --run-dir artifacts/ett_convex_adversarial/l1_matrix32_s01_v1',
        'python -m scripts.check_convex_adversarial --run-dir artifacts/ett_convex_adversarial/l1_matrix32_s01_v1',
        'python -m ett.report_convex_adversarial --run-dir artifacts/ett_convex_adversarial/l1_matrix32_s01_v1','```','',
        'Use a fresh output directory from the PointMaze worktree. Missing original checkpoints fail explicitly; exact paths/hashes are in provenance.json. The sampled kernel API is ConvexActionTransition.sample(state, action, observational_action, key, num_samples, goal=goal), preserving leading batch shapes. Report/verification code was added after training for presentation and independent checks; the preregistered model, optimizer, evaluation and specification hashes remain unchanged.','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write(root/'report_manifest.json',dict(source_sha256=file_sha(__file__),
        files={p.name:file_sha(p) for p in root.iterdir() if p.is_file() and p.name!='report_manifest.json'}))
    print('Report and four plots written; no new training or rollout sampling.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True);main(Path(parser.parse_args().run_dir))
