"""Readable saved-result figures/report; no model sampling or training."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ett.run_geometry_comparison import read,write,sha
from ett.diagonal_transition import POINTMAZE_WALLS


def report(root):
    metrics=read(root/'metrics.json');paired=read(root/'paired_returns.json')
    training=read(root/'training.json');completion=read(root/'completion.json')
    names=['zero','rectangle_s0','fork_segment_s0','rectangle_s1','fork_segment_s1']
    labels=['Zero','Rectangle s0','Fork segment s0','Rectangle s1','Fork segment s1']
    colors=['#777777','#2864a5','#b65b20','#2864a5','#b65b20']
    styles=['-', '-', '-', '--','--']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12.2,4.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
    for name,label,color,style in zip(names[1:],labels[1:],colors[1:],styles[1:]):
        h=training['models'][name]['history']
        axs[0].plot([v['iteration'] for v in h],[v['monitor_return'] for v in h],label=label,color=color,linestyle=style)
    axs[0].set(xlabel='Optimizer update',ylabel='Mean discounted model return',title='Monitoring only: 32 paths / iterate')
    axs[0].legend(fontsize=8);axs[0].set_xticks([0,4,8,12,16])
    keys=['rectangle_s0_minus_zero','fork_segment_s0_minus_zero','fork_minus_rectangle_s0',
          'rectangle_s1_minus_zero','fork_segment_s1_minus_zero','fork_minus_rectangle_s1']
    ticks=['s0: rectangle − zero','s0: segment − zero','s0: segment − rectangle',
           's1: rectangle − zero','s1: segment − zero','s1: segment − rectangle']
    for i,key in enumerate(keys):
        v=paired[key];low,high=v['interval95']
        axs[1].errorbar(v['mean'],i,xerr=[[v['mean']-low],[high-v['mean']]],fmt='o',capsize=4,
                       color=['#2864a5','#b65b20','#555555'][i%3])
    axs[1].axvline(0,color='black',linewidth=.8)
    axs[1].set(yticks=range(6),yticklabels=ticks,xlabel='Paired difference in discounted model return',
               title='512 independent paired paths\n95% episode-bootstrap intervals')
    axs[1].invert_yaxis()
    fig.suptitle('Fixed actor: lower return improves only the model objective',fontsize=13)
    fig.savefig(root/'optimization_and_returns.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(12.2,4.0),layout='constrained')
    pos=np.arange(5);short=['Zero','Rect 0','Seg 0','Rect 1','Seg 1']
    for ax,field,title,ylabel in [(axs[0],'mean_motion','Emitted motion','Mean XY distance / step'),
        (axs[1],'near_grid_boundary_fraction','Near integer grid lines (<1e-4)','Fraction of emitted positions'),
        (axs[2],'zero_return_fraction','Zero-return paths','Fraction of evaluation paths')]:
        ax.bar(pos,[metrics[n][field] for n in names],color=colors)
        ax.set(xticks=pos,xticklabels=short,title=title,ylabel=ylabel)
        ax.tick_params(axis='x',labelrotation=25)
    fig.suptitle('Independent generated-state distributions; 512 paths per model',fontsize=13)
    fig.savefig(root/'transition_mechanisms.png',dpi=170);plt.close(fig)

    records={name:np.load(root/f'eval_{name}.npz') for name in names}
    fig,axs=plt.subplots(2,2,figsize=(12.2,7),layout='constrained')
    for idx,ax in enumerate(axs.flat):
        for x,y in np.argwhere(POINTMAZE_WALLS==1):
            ax.add_patch(plt.Rectangle((x,y),1,1,color='#e5e5e5',zorder=0))
        for name,label,color,style in zip(names,labels,colors,styles):
            xy=records[name]['states'][idx,:,:2]
            ax.plot(xy[:,0],xy[:,1],label=label,color=color,linestyle=style,linewidth=1.6,alpha=.85)
            ax.scatter(xy[-1,0],xy[-1,1],s=18,color=color)
        ax.scatter([8.5],[3.5],marker='*',s=80,color='black')
        ax.set(xlim=(0,9),ylim=(.8,4.2),aspect='equal',xlabel='Visible X (maze units)',ylabel='Visible Y (maze units)',
               title=f'Predeclared path {idx} · group 64000000')
    handles,legend=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside upper center',ncol=5,fontsize=9)
    fig.savefig(root/'paired_trajectories.png',dpi=170);plt.close(fig)

    rows=[]
    for seed in [0,1]:
        for mode in ['rectangle','fork_segment']:
            name=f'{mode}_s{seed}';v=paired[name+'_minus_zero'];m=metrics[name]
            rows.append(f"* **{mode}, seed {seed}:** final mean J={m['mean_return']:.6f}; final-minus-zero "
                        f"{v['mean']:+.6f}, 95% CI [{v['interval95'][0]:+.6f}, {v['interval95'][1]:+.6f}]. "
                        f"Weighted adversarial term={m['weighted_J']:.6f}.")
    diffrows=[]
    for seed in [0,1]:
        v=paired[f'fork_minus_rectangle_s{seed}']
        diffrows.append(f"* Seed {seed}: fork-segment minus rectangle **{v['mean']:+.6f}**, "
                        f"95% CI [{v['interval95'][0]:+.6f}, {v['interval95'][1]:+.6f}].")
    text='''# Does the new geometry change what the existing pessimistic objective learns under a fixed actor?

**It changes the learned coefficients and projected transition responses, but this
budget does not establish a return advantage for either geometry. Both modes
reduce independent model return for optimizer seed 0; neither establishes a
decrease for seed 1. The mode-difference intervals include zero in both seeds.**
This is a mixed optimization result. Lower J improves the adversarial objective,
not actor performance or proof of useful synthetic data. No native evaluation or
downstream CRL training was run.

## Controlled experiment and independent return

Actual starting commit is `be1b3c520ce91ea4a7e9c4d9f4a7fa0941e86345`, exactly the
requested commit, on `feature/pointmaze-causal-transition`; no intervening commit.
The [pre-execution specification](SPEC.md), [configuration](config.json),
[seed schedule](seed_schedule.json) and [input/source hashes](provenance.json)
were saved before sampling. All four models start at exactly zero 32-vectors.
Both zero modes match exactly on the sampled diagonal checks and eight matched
50-step initial paths, including actor/nominal actions, anchors, states and rewards.

The historical stochastic actor, its critic state, state-goal nominal policy,
guarded diagonal sampler and every normalization remain frozen. Each rollout
samples x_prime, then execution x, then the successor; each model's own generated
state conditions its next actions. Perturbation directions and rollout random
keys are paired across modes within each optimizer seed and across +/- queries.
Random streams are not native hidden randomness, and diverging states do not
share fixed nominal action values. Training, monitoring and evaluation keys are
disjoint. Evaluation uses four groups of 128 paths, shared across all five models.

The two-term objective stays L_diag + lambda E[J], lambda=1/sum(.95^t), H=50,
with successor-position reward 1[distance to (8.5,3.5)<2]. L_diag is structurally
constant: only the second term updates theta, not a jointly trained diagonal
network. The original antithetic parameter-perturbation estimator and capped
descent helper perform 16 updates, 8 directions, 32 paths/sign, sigma=.1, rate=1,
cap=.2. Final iterate only. No hard-reward pathwise derivative, likelihood-ratio
surrogate, readout or third loss is used. A zero-weight optimizer was not repeated.

The common initialization has mean J=10.413263, L_diag=0.014970824 and weighted
adversarial term 0.564065. L_diag is the same in every final model; it was measured
with 16 draws on each of 1,024 preserved training and 1,024 held-out tuples.
Its smaller Monte Carlo audit than the historical experiment changes neither
the population fitting term nor its zero contribution to parameter differences.
The following intervals use 2,000 paired episode-bootstrap replicates within the
four evaluation groups, not individual steps or pooled optimizer seeds.

'''+ '\n'.join(rows)+ '\n\n'+ '\n'.join(diffrows)+'''

These are conditional intervals for the five fixed checkpoints and common frozen
inputs. They do not establish broad optimizer robustness. Group-specific means,
objective terms and every paired episode difference are retained in the numerical
artifacts. Monitors use different keys at each iterate and never select checkpoints.

![Monitoring and independent paired differences](optimization_and_returns.png)

## What changed in the transitions

The segment mode retains the prior demonstrated capacity; it does not contain
every output of an individual rectangle instance. This experiment finds a change
in responses, without requiring downward motion or route completion.

On independent rollouts, supported fork-region visits are 578 for zero, then
613/598 for rectangle/segment seed 0 and 559/579 for seed 1, out of 25,600 steps
per model. Geometry-eligible counts are 569, 600/586 and 550/570 respectively.
Segment runs select their segment on all 586/570 eligible steps (2.289%/2.227%
of steps); rectangles never select one. These differing counts describe changing
generated-state distributions, not fixed-input response changes.

For the segment models, mean projection fractions are .00816/.00932. Exact anchor
mass is 65.53%/62.98%; endpoint mass is zero, and maximum fractions are .1362/.1323.
At their eligible contexts, mean anchor displacement is approximately (+.851,+.006)
and (+.851,+.005), correction is (+.0217,-.0048) and (+.0117,+.0046), and emitted
displacement is (+.8459,+.0007) and (+.8454,-.0013). Most of the sampled anchor's
rightward motion remains. Set projection can suppress a correction; the second
objective is not required to pursue the lower route.

Separately, all four learned coefficient arrays were evaluated in **both** modes
on the same 512 fresh component anchors plus 225 preserved fork/descent inputs.
No changed component output was fed forward. On the fixed 225 contexts, mean
unclipped Y correction for rectangle-trained versus segment-trained coefficients
is -.04955 versus -.02749 in seed 0, and -.03691 versus +.03603 in seed 1.
Changing only geometry leaves the matrix correction unchanged but changes emitted
XY; all eight combinations retain six below-Y=3 outputs. These descriptive
component comparisons separate parameter/projection changes from state visitation;
they are not new route rollouts or native response labels.

![Transition behavior](transition_mechanisms.png)

Zero-return path rates are 6.05% initially, 7.81%/6.64% in seed 0, and 4.69%/4.69%
in seed 1 (rectangle/segment). Mean step motion is .3591 initially, .3419/.3587 and
.3569/.3632. Occupancy within 1e-4 of an integer XY grid line rises from .0195%
to 4.77%/3.68% and 1.93%/2.95%; such a line need not be a physical wall.
Selected-set boundary occupancy is 3.95% initially, 10.36%/9.39% and
4.54%/6.36%; for a segment, this refers to relative endpoints, not a physical wall.

Every rectangle-model stationary anchor, and 99.83%/99.84% of segment-model
stationary anchors, emits motion above 1e-5. Initially stationary anchors remain
fixed, but the baseline itself resumes motion in 7.92% of 1,869 non-reset stationary
histories. The final models have **zero** such histories, so their conditional
stationary-history statistic is unavailable, not evidence that absorption is
correct. Visible stationarity is not confirmed death: these findings flag a
limitation, not a proved native violation. Rare sampled straight-line wall
intersections remain (.027%–.086% in finals); that line audit is not the native
axiswise simulator path. Valid endpoints alone do not settle kinematic validity.

![Predetermined paired trajectories](paired_trajectories.png)

## Verification, numerical qualifications and accounting

All four new checkpoints persist `format_version`, `geometry_mode`, `bound` and
theta. New evaluation requires mode metadata; historical theta/bound checkpoints
still mean rectangle. Reloaded first evaluation groups reproduce **every saved
field exactly**. Rollouts retain geometry kind, eligibility, segment endpoints and
fraction, reference box, anchor, correction, proposal, successor, goals and boundary
indicators. Validation uses the actual selected convex set, not the old box for
segment outputs. The original sampling emitter is otherwise unchanged.

Frozen component/file/normalization hashes, exact diagonal samples and F4/goal
contracts pass. Maximum normalized block Frobenius norms are .4180/.3671 in seed 0
and .4192/.3463 in seed 1. Finite fixed-conditioning grid/near-pair checks have zero
positive bound excess within 2e-6. The proof remains fixed-set nonexpansive
projection composed with ||M||op<=1, for every execution-action pair at fixed
state, goal, x_prime, anchor and parameters. L=1 is prescribed; these properties
do not identify off-diagonal causality, native kinematics, absorption or long-term
validity. The known float32 anchors exceeding literal unit steps by 1.19e-7 are
preserved; the contract is the original rounded q+/-1 box, not a repaired strict bound.
Across each of the five new evaluation sets, the maximum literal coordinate step
is 1.0000004768371582, reflecting rounding at other coordinate magnitudes as well.

The [ledger](transition_ledger.json) records exactly **2,098,720** newly sampled
model transitions against the predeclared **2,500,000** cap. All sampler calls
completed; no training restart or extra optimization occurred. Nine focused
tests pass, including metadata-loss rejection, actual segment membership and
the existing analytic optimizer sign test. Independent saved-result checks
recompute all 64 updates, paired keys, cache hashes and return intervals.

Three implementation qualifications are preserved in [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md):
a preparation-only seed-scan false positive (zero draws); an overly strict
post-hoc checker tolerance for the original float32 update cast (zero draws);
and a near-ceiling standalone-versus-JIT eligibility tally discrepancy. Actual
rollout eligibility and selection agree everywhere; the fixed-component raw
standalone tally is 217, while emitted JIT diagnostics select 218 saved fork
contexts. Both use the declared action-independent rounded construction.
The corrected selected-set counts and exact example are retained separately.
No trained model or hyperparameter was changed in response.

## Decision

**Defer downstream CRL.** This budget establishes neither a geometry return
advantage nor a repeatable decrease across both optimizer seeds, and the
stationary-anchor/boundary behavior leaves synthetic-transition validity unresolved.
The new capability is real, and its learned use changes, but lower model return
alone does not establish better training data. First resolve which of these
off-diagonal motion/absorption behaviors are admissible using explicit physical
assumptions or suitable visible action-coverage evidence; this report does not
assign a hidden failure label or blame the second loss alone.

If that concern is resolved, the later small comparison should use matched
learner seeds, update counts, replay mixture, future-goal sampling and native
evaluation for offline-only, rectangle-generated and fork-segment-generated
samples. Keep model optimizer seeds separate. That comparison was **not run**.
See [REPRODUCE.md](REPRODUCE.md) for the completed experiment commands.
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    write(root/'report_provenance.json',dict(report_source_sha256=sha(__file__),
        figures=['optimization_and_returns.png','transition_mechanisms.png','paired_trajectories.png'],
        representative_selection='group 64000000, indices 0..3 fixed before training',
        new_model_transitions=0,sampled_transitions=completion['sampled_model_transitions']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    report(parser.parse_args().run_dir)
