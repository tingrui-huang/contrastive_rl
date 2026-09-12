"""Comparison figures and English report for the bounded conditional spline."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.synthetic_shared_response import sha, write_json
from scripts.synthetic_lipschitz_response import PRIOR, read


ARMS = ['joint', 'signed_difference', 'conditional_spline']
LABELS = ['Unconstrained MLP', 'Unconstrained + signed difference', 'Bounded conditional spline']
COLORS = ['#2177ac', '#898989', '#c56514']
KEYS = ['anchored_learned_value', 'anchored_prescribed_c', 'arbitrary_random_pairs', 'grid_adjacent_pairs']


def make_figures(root, config, results, arrays, history):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})

    def save(fig,name):
        fig.tight_layout(rect=(0,.03,1,.93))
        fig.text(.5,.005,'Synthetic supervised comparison | L=1 | Three seeds, final step 2500 | No evaluation-based selection',ha='center',fontsize=8)
        fig.savefig(root/name,dpi=160);plt.close(fig)

    for close,name,title in [(False,'response_comparison.png','Final responses at the original held-out nominal actions'),
                              (True,'corner_comparison.png','Local diagonal behavior: correct slopes can coexist with a shifted anchor')]:
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for i,ax in enumerate(axes.flat):
            anchor=config['slice_anchors'][i]
            for arm,label,color in zip(ARMS,LABELS,COLORS):
                values=np.array([a['close_over_L' if close else 'slice_over_L'][i] for a in arrays[arm]])
                x=arrays[arm][0]['close_offsets' if close else 'grid_axis']
                ax.plot(x,values.mean(axis=0),label=label,color=color)
                ax.fill_between(x,values.min(axis=0),values.max(axis=0),color=color,alpha=.15)
            ax.plot(x,-np.abs(x if close else x-anchor),'k--',label='Known analytic target')
            ax.set(xlabel='Execution offset x - x_prime' if close else 'Execution action x',
                   ylabel='Outcome G (L=1)',title=f'x_prime={anchor:g}; band=seed min/max')
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(title);save(fig,name)

    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        h=[history[f'{arm}_L1_s{s}'] for s in config['seeds']]
        for ax,key,title in zip(axes[0],['normalized_diag','normalized_off'],['Diagonal monitor MSE','Off-diagonal monitor MSE']):
            values=np.array([[r[key] for r in records] for records in h]);steps=[r['step'] for r in h[0]]
            ax.plot(steps,values.mean(axis=0),color=color,label=label)
            ax.fill_between(steps,values.min(axis=0),values.max(axis=0),color=color,alpha=.15)
            ax.set(xlabel='Optimizer step',ylabel='MSE (L=1)',yscale='log',title=title)
        strata=np.array([[b['normalized_rmse'] for b in results[f'{arm}_L1_s{s}']['strata']] for s in config['seeds']])
        axes[1,0].plot(range(6),strata.mean(axis=0),'o-',color=color,label=label)
        axes[1,0].fill_between(range(6),strata.min(axis=0),strata.max(axis=0),color=color,alpha=.15)
        diagonal=np.array([np.diag(a['grid_over_L']) for a in arrays[arm]])
        axis=arrays[arm][0]['grid_axis']
        axes[1,1].plot(axis,diagonal.mean(axis=0),color=color,label=label)
        axes[1,1].fill_between(axis,diagonal.min(axis=0),diagonal.max(axis=0),color=color,alpha=.15)
    axes[0,0].legend(fontsize=7)
    axes[1,0].set(xticks=range(6),xticklabels=['<.001','.001-.01','.01-.1','.1-.5','.5-1','1-2'],
        xlabel='Action distance |x - x_prime|',ylabel='Off-diagonal RMSE',yscale='log',title='Distance-stratified evaluation')
    axes[1,1].axhline(0,color='black',linestyle='--',linewidth=1)
    axes[1,1].set(xlabel='Nominal action x_prime',ylabel='Diagonal error G(x_prime,x_prime)',title='Learned diagonal; target zero')
    fig.suptitle('Fitting cost and seed variation; original monitor/evaluation sets');save(fig,'fitting_comparison.png')

    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    labels=['Learned\nanchor','Prescribed\nc=0','Random\npairs','Grid\nintervals']
    for j,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
        for ax,metric in zip(axes[:2],['max_ratio_over_L','fraction_above_tolerance']):
            values=np.array([[results[f'{arm}_L1_s{s}']['bounds'][k][metric] for k in KEYS] for s in config['seeds']])
            if metric=='fraction_above_tolerance':values*=100
            ax.bar(np.arange(4)+(j-1)*.23,values.mean(axis=0),.22,color=color,label=label)
            for row in values:ax.scatter(np.arange(4)+(j-1)*.23,row,color='black',s=10,zorder=3)
        # Prescribed-c is intentionally excluded: it compares against a target, not a model output.
        excess=[max(results[f'{arm}_L1_s{s}']['bounds'][k]['largest_positive_absolute_excess'] for k in [KEYS[0],KEYS[2],KEYS[3]]) for s in config['seeds']]
        axes[2].scatter([j]*len(excess),np.maximum(excess,1e-18),color=color,s=30)
    for ax in axes[:2]:ax.set_xticks(range(4),labels,fontsize=8)
    axes[0].axhline(1.01,color='black',linestyle='--',linewidth=1)
    axes[0].set(ylabel='Maximum ratio / L',title='Full bound vs prescribed anchor')
    axes[1].set(ylabel='Pairs above 1.01 L (%)',title='Original 1% ratio tolerance')
    axes[2].axhline(1e-12,color='black',linestyle='--',label='Float64 audit tolerance')
    axes[2].set(xticks=range(3),xticklabels=['MLP','Signed MLP','Spline'],yscale='log',
                ylabel='Largest positive absolute excess',title='Output-difference violations')
    axes[0].legend(fontsize=6);axes[2].legend(fontsize=6)
    fig.suptitle('Bound diagnostics | Minimum separation 0.001 | Dots are individual seeds');save(fig,'constraint_comparison.png')

    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for seed,ax,a in zip(config['seeds'],axes,arrays['conditional_spline']):
        desired=np.where(np.arange(16)<8,1.,-1.)
        error=np.ma.array(np.abs(a['interval_slopes']-desired[None,:]),mask=~a['interval_feasible'])
        im=ax.imshow(error.T,origin='lower',aspect='auto',interpolation='nearest',extent=(-1,1,-2,2),vmin=0,vmax=2,cmap='magma')
        ax.set(xlabel='Nominal action x_prime',ylabel='Signed coordinate x - x_prime',title=f'Seed {seed}: interval slope error')
        fig.colorbar(im,ax=ax,label='|learned slope - target slope|')
    fig.suptitle('Exact interval coefficients at fixed contexts; blank regions lie outside x in [-1,1]')
    save(fig,'interval_slopes.png')


def summarize(root,config,results,arrays):
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for key in ['diagonal','off','grid_error','uniform_square_off']:
            v=np.array([results[f'{arm}_L1_s{s}'][key]['normalized_rmse'] for s in config['seeds']])
            summary[arm][key]=dict(mean=float(v.mean()),seed_std=float(v.std(ddof=1)),values=v.tolist())
    boundary={}
    for seed,a in zip(config['seeds'],arrays['conditional_spline']):
        axis=a['grid_axis'];error=a['grid_over_L']-a['grid_target_over_L']
        boundary[str(seed)]=dict(central_xp_grid_rmse=float(np.sqrt(np.mean(error[np.abs(axis)<=.8]**2))),
            boundary_xp_grid_rmse=float(np.sqrt(np.mean(error[np.abs(axis)>.8]**2))))
    write_json(root/'summary.json',summary)
    write_json(root/'boundary_diagnostic.json',dict(results=boundary,
        status='posthoc descriptive location audit; not a selection rule or another training run',cutoff=.8))
    return summary,boundary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();root=Path(args.run_dir)
    config,results,completion=read(root/'config.json'),read(root/'results.json'),read(root/'completion.json')
    if config['smoke'] or completion['status']!='complete' or read(root/'verification.json')['status']!='passed':
        raise ValueError('report requires the completed verified primary experiment')
    arrays={arm:[] for arm in ARMS}
    history=read(PRIOR/'history.json');history.update(read(root/'history.json'))
    for arm in ARMS:
        for seed in config['seeds']:
            base=root if arm=='conditional_spline' else PRIOR
            with np.load(base/f'{arm}_L1_s{seed}_evaluation.npz') as a:
                arrays[arm].append({k:a[k] for k in a.files})
    make_figures(root,config,results,arrays,history)
    summary,boundary=summarize(root,config,results,arrays)
    write_report(root,config,results,completion,summary,boundary)
    write_json(root/'report_provenance.json',dict(source_sha256=sha(__file__),matplotlib_version=matplotlib.__version__,
        input_hashes={name:sha(root/name) for name in ['config.json','results.json','history.json','verification.json','completion.json']},
        figures_from_saved_arrays_only=True,baseline_root=PRIOR.as_posix()))
    print('Wrote five comparison figures, summary and REPORT.md.')


def write_report(root,config,results,completion,summary,boundary):
    lines=['# Full action-Lipschitz conditional response: bounded follow-up','',
        '## Question and construction','',
        'Can a shared model jointly fit diagonal observations and a known compatible off-diagonal target while enforcing a full action bound? '
        'This isolated supervised diagnostic retains x,x_prime in [-1,1], c=0, L=1 and target -abs(x-x_prime). '
        'The two losses are mean(G(x_prime,x_prime)^2) and mean((G(x,x_prime)+abs(x-x_prime))^2), separately averaged with lambda_off=1. '
        'No third loss, discriminator, PointMaze training or actor-critic update is introduced. '
        'The original experiment at commit `752b6457debb65aeacc19979a97db833bbdb6149`, its code, report and verification were reviewed; '
        'all original files and checkpoints remain unchanged.','',
        'The new model uses signed coordinate t=x-x_prime. A single shared ReLU conditioner, 1-32-32-17, receives x_prime and emits '
        'one free intercept b(x_prime) plus 16 raw slopes. Each slope is clamped to [-1,1]. '
        'Uniform fixed knots k_j=-2+j/4 divide [-2,2] into 16 intervals. The final output is','',
        '`G(x,x_prime) = b(x_prime) + sum_j a_j(x_prime) * clamp(t-k_j, 0, 1/4)`,','',
        'where a_j=clamp(raw_j,-1,1). This is the integral form of continuous linear interpolation. '
        'The intercept is located at t=-2, not on the diagonal. All predictions use the same coefficient vector and basis. '
        'There is no absolute-difference input, target-shaped initialization, diagonal branch, residual multiplier forcing equality, '
        'separate diagonal/off heads or discriminator. The signed coordinate is a deliberate architectural input, so the original '
        'unconstrained signed-difference MLP is included as well as the original two-input MLP.','',
        '## Guarantee, expressivity and precision','',
        'Fix x_prime. The coefficients are then constants, the output is continuous at each knot, and its derivative with respect to x '
        'inside interval j is exactly a_j. Splitting any [x1,x2] at knots and summing interval changes gives '
        '|G(x1,x_prime)-G(x2,x_prime)| <= sum_j |a_j| * interval_length_j <= |x1-x2|. '
        'This is the full arbitrary-action condition for the final emitted function, not just a radius about x_prime. '
        'It holds in real arithmetic for every finite parameter setting and hence every training iterate, including initialization. '
        'No evaluation sample or gradient penalty is needed for this argument. No bound in x_prime is asserted or required here.','',
        'The target is exactly representable: a constant b=-2, eight slopes +1 followed by eight slopes -1 give -abs(t). '
        'A constant output coefficient vector can be represented by the conditioner. This assignment is used only in a numerical '
        'representation test, never as a training initializer or feature. The diagonal value remains b+sum_{j<8} a_j/4 and must be learned. '
        'The constraint therefore does not impose a positive approximation-error floor on this particular target.','',
        'There is still a strong inductive bias: execution-action slopes are constant between the prescribed signed-coordinate knots, '
        'and the target corner happens to align with the zero knot for every x_prime. Generic corners between knots or curved responses '
        'would incur interpolation error at this fixed resolution. Finer bounded-slope interpolation can approximate continuous action-Lipschitz '
        'functions on compact domains under suitable conditional approximation, but no such general approximation claim is empirically tested here. '
        'This architecture is not a generic high-dimensional transition network.','',
        'All new training and emitted outputs use float64; knots and interval width are exactly representable dyadic numbers. '
        'The analytic bound concerns real arithmetic with the stored coefficients, not a proof that a quantized machine map obeys an exact '
        'real-domain inequality bit for bit. Addition, subtraction and reduction can yield tiny positive numerical excesses. '
        'We predeclare an absolute excess audit tolerance of 1e-12 for this bounded-scale float64 experiment; this is a diagnostic tolerance, '
        'not an outward-rounded formal arithmetic certificate. Legacy float32 outputs are preserved and assigned a separate 1e-6 absolute '
        'audit tolerance. The original ratio threshold 1.01 L and minimum separation 0.001 are unchanged for all comparisons. '
        'Baseline excesses are orders of magnitude larger than either numerical tolerance.','',
        '## Fixed optimization and controls','',
        'Train only three new models, seeds 0/1/2, at L=1. Each uses the original 2,500 Adam steps, cosine learning rate 0.001 to 0.0001, '
        'zero weight decay, diagonal batch 128 and off batch 256. Reuse the exact original data generator and seeds: '
        '8,192 explicit diagonal examples and 32,768 balanced near/far off examples per seed. '
        'The RNG calls and sizes generating minibatch indices are identical to the checked original implementation; '
        'the new index-stream hashes are independently regenerated by verification. Float32 action values are promoted to float64 for the new '
        'network and target arithmetic, without changing the sampled values.','',
        'Both loss terms update the shared trunk, free intercept and left-interval coefficient rows. '
        'Right-interval output rows have zero direct diagonal gradient because their basis lengths vanish at t=0; they receive off-diagonal gradients. '
        'No parameter is frozen. Hard slope saturation can produce zero local derivatives through the clamp, so remaining optimization errors '
        'cannot be interpreted as a representational impossibility. Every parameter tensor changed during each run; this does not claim '
        'every scalar parameter received a nonzero gradient at every step.','',
        'The spline has 1,681 trainable parameters, versus 1,185 for the original MLP and 1,217 for the signed MLP. '
        'Initial functions are not matched across different architectures; the same seed convention uses ordinary random linear-layer initialization. '
        'Capacity, local basis, signed-coordinate structure, saturation and float64 arithmetic all change. This is a construction comparison, '
        'not a causal isolation of constraint enforcement alone.','',
        'Reuse six baseline checkpoints/results, covering two joint-fitting arms and three seeds at L=1. '
        'Before reuse, their saved checkpoint hashes, every saved evaluation array and every original metric reproduce exactly. '
        'No baseline is retrained and no scaled L copies are counted as new evidence. '
        'The original held-out pairs, 201-by-201 grid, four displayed nominal actions and monitor set are unchanged. '
        'They were inspected previously and are not a newly untouched test set. The final step is always selected; '
        'no tuning, checkpoint search or extra optimization follows evaluation.','',
        '## Fitting results','',
        'Values are native outcome RMSE at L=1, reported as mean +/- sample standard deviation across three seeds. '
        'The independent evaluation has 4,096 diagonal points, 16,384 balanced off pairs, 16,384 uniform-square pairs, '
        '32,768 arbitrary-action triples and the unchanged dense grid.','']
    for arm,label in zip(ARMS,LABELS):
        m=summary[arm]
        maxdiag=[results[f'{arm}_L1_s{s}']['diagonal']['normalized_max_abs'] for s in config['seeds']]
        lines.append(f'- {label}: diagonal RMSE {m["diagonal"]["mean"]:.6g} +/- {m["diagonal"]["seed_std"]:.3g}; '
            f'diagonal maximum absolute errors by seed {[round(v,6) for v in maxdiag]}; '
            f'off RMSE {m["off"]["mean"]:.6g} +/- {m["off"]["seed_std"]:.3g}; '
            f'grid RMSE {m["grid_error"]["mean"]:.6g} +/- {m["grid_error"]["seed_std"]:.3g}.')
    lines+=['','The constrained model retains approximate diagonal fitting without an enforced anchor, with worse maximum errors in two seeds. '
        'It has a fitting cost on average: off and grid RMSE increase, and variation across seeds is larger. Seed 1 improves '
        'over the plain MLP while seeds 0 and 2 worsen. The signed baseline helps distinguish the input feature from the richer spline construction, '
        'but does not remove all architectural confounding.','',
        '![Response comparison](response_comparison.png)','','![Training, distance errors and diagonal errors](fitting_comparison.png)','',
        'Distance-bin errors and per-seed physical metrics are saved in results.json. The small float32 diagonal train/evaluation coincidences '
        'from the original split remain unchanged; verification additionally reports RMSE after excluding exact coincident values.','',
        '## Constraint results and diagonal-anchor interpretation','',
        'For learned-anchor and arbitrary-action pairs, report |G1-G2|/|x1-x2|. The prescribed-c ratio uses |G(x,x_prime)-0|/|x-x_prime|. '
        'The largest positive absolute excess is max(0, max(|G1-G2|-|x1-x2|)); signed maxima are also saved. '
        'Grid-adjacent ratio maxima equal arbitrary-pair maxima on that grid by telescoping, but their violation fraction counts only adjacent intervals.','']
    for arm,label in zip(ARMS,LABELS):
        lines+=['### '+label,'']
        for key in KEYS:
            records=[results[f'{arm}_L1_s{s}']['bounds'][key] for s in config['seeds']]
            lines.append(f'- {key}: maximum ratios {[float(format(r["max_ratio_over_L"],".7g")) for r in records]}; '
                f'positive absolute excesses {[float(format(r["largest_positive_absolute_excess"],".5g")) for r in records]}; '
                f'percent above the 1% ratio tolerance {[round(100*r["fraction_above_tolerance"],3) for r in records]}.')
        lines.append('')
    lines+=['Enforcement eliminates genuine action-bound violations in all three new models: learned-anchor, random-pair and grid checks '
        'show at most 4.44e-16 positive absolute excess, below the predeclared 1e-12 tolerance. '
        'Direct interval-coefficient inspection gives maximum slope magnitude exactly 1 in every seed. '
        'The interval endpoint identity is accurate to at most 4.44e-16. These finite audits validate the implementation; '
        'the earlier integration argument supplies the continuum guarantee for all fixed x_prime, including unprobed values.','',
        'Prescribed-c ratios still reach about 14.8, 5.37 and 12.3 in the new models. They are not violations of the full output-difference bound. '
        'When d(x_prime)=G(x_prime,x_prime) is nonzero, the valid lower envelope is d(x_prime)-|x-x_prime|, '
        'not necessarily -|x-x_prime|. Predictions below the analytic c=0 envelope can follow directly from negative diagonal fitting error. '
        'The largest shortfalls below the learned-anchor envelope are only roundoff (<=4.44e-16); '
        'shortfalls below the c=0 envelope reach approximately 0.0171, 0.00636 and 0.0162. '
        'These values do not indicate discovery of a better admissible pessimistic response.','',
        '![Constraint comparison](constraint_comparison.png)','','## Corner behavior and fitting-cost diagnosis','',
        'At the four originally chosen held-out nominal actions and offsets 1e-4, 0.001, 0.01 and 0.04, '
        'the new left/right slopes closely match +1/-1. At offset 1e-4, all reported slopes are exactly +1/-1 up to evaluation precision '
        'except the seed-1 left slope at x_prime=-0.83, which is about 0.9953. '
        'Unlike the original unassisted MLP, the signed knot grid permits a corner precisely on the diagonal. '
        'It does not fix the corner height: learned vertical offsets remain visible and measurable.','',
        '![Both sides of the diagonal](corner_comparison.png)','',
        'A posthoc descriptive location audit (no new training or selection) splits the existing grid at |x_prime|=0.8.','']
    for seed in config['seeds']:
        b=boundary[str(seed)]
        value=results[f'conditional_spline_L1_s{seed}']
        lines.append(f'- Seed {seed}: central nominal-region grid RMSE {b["central_xp_grid_rmse"]:.6g}; '
            f'outer nominal-region grid RMSE {b["boundary_xp_grid_rmse"]:.6g}; '
            f'{100*value["intervals"]["saturated_fraction"]:.2f}% of feasible audited interval slopes are saturated at magnitude 1.')
    audit=read(root/'verification.json')['checks']
    counts=[audit[f'conditional_spline_L1_s{s}']['slope_optimization_audit']['wrong_sign_saturated_count'] for s in config['seeds']]
    raw=[audit[f'conditional_spline_L1_s{s}']['slope_optimization_audit']['endpoint_xp1_first_interval_raw'] for s in config['seeds']]
    lines+=['',f'Raw coefficient inspection finds {counts} wrong-sign saturated intervals out of 1,800 feasible audited intervals in seeds 0/1/2. '
        f'At x_prime=1, the first interval has raw coefficients {[round(v,5) for v in raw]}, where the target slope is +1. '
        'A wrong-sign coefficient below -1 has zero direct derivative through the slope clamp. Shared-conditioner gradients from other '
        'coefficients can still move it, so it is not a frozen parameter. This is concrete evidence of a local optimization obstacle in seed 0, '
        'not proof of its sole cause or an intrinsic approximation limit.','',
        'Larger errors are concentrated near the nominal-action boundaries. The coefficient plot identifies intervals whose learned slopes '
        'still differ from the known target; saturation is often correct because the optimum has slope magnitude 1. '
        'This fixed grid does not exclude the exact target, so the observed loss cannot be explained as a mandatory approximation floor from '
        'the Lipschitz constraint. Finite-data coverage, optimizer dynamics, conditional generalization and clamp saturation may matter, '
        'but this single bounded experiment does not isolate their contributions. No hyperparameter or architecture sweep was launched.','',
        '![Exact interval coefficients](interval_slopes.png)','','## Decision and limitations','',
        'The answer is yes for structural enforcement with retained approximate fitting, with a measurable and seed-dependent fitting cost. '
        'A shared model can learn the diagonal height while remaining action-Lipschitz throughout training. '
        'It needs neither a discriminator nor a forced diagonal anchor. The guarantee applies to action differences, not to correctness of the '
        'diagonal height, supervised target, environment dynamics or causal interpretation.','',
        'This is a useful controlled basis for a subsequent scalar experiment that removes known off-diagonal labels and defines a pessimistic objective. '
        'It separates an enforceable feasible class from objective discovery. Such a later test must still examine free-intercept drift and the tradeoff '
        'against diagonal error: the full action bound alone does not pin the absolute outcome. '
        'The current evidence does not justify moving directly to PointMaze or alternating actor-critic training. '
        'No causal identification, real-environment L estimation or worst-case Q recovery is claimed. The subsequent experiment is not run here.','',
        '## Runtime, checks and reproduction','',
        f'The three new runs and their saved evaluation took {completion["elapsed_seconds"]:.2f} seconds on CPU with one thread. '
        f'Per-seed training times were {[round(results[f"conditional_spline_L1_s{s}"]["training_seconds"],3) for s in config["seeds"]]} seconds. '
        'The old experiment recorded 116.5 seconds for all 27 scale/arm/seed runs including evaluation and writing; '
        'it did not record comparable per-arm timing, so no architecture speedup is claimed.','',
        'Five focused tests cover arbitrary-action pairs under large coefficient-network changes, exact target representability, '
        'knot continuity and one-sided slopes, additive shared gradients with a freely movable diagonal, and checkpoint roundtrip. '
        'A separate five-step smoke passed. Verification reloads all three final checkpoints and exactly reproduces every saved metric and array, '
        'checks minibatch hashes/data hashes and the two loss terms, audits interval bounds, and confirms all prior artifact hashes are unchanged. '
        'All new losses and gradients were finite. Checkpoints stay local and ignored; configurations, metrics, evaluation arrays and English figures/report '
        'are prepared for version control.','',
        '```bash',
        'python -m scripts.test_synthetic_lipschitz_response',
        'python -m scripts.synthetic_lipschitz_response --out-dir artifacts/synthetic_lipschitz_response/smoke --smoke',
        'python -m scripts.synthetic_lipschitz_response --out-dir artifacts/synthetic_lipschitz_response/l1_s012',
        'python -m scripts.check_synthetic_lipschitz_response --run-dir artifacts/synthetic_lipschitz_response/l1_s012',
        'python -m scripts.report_synthetic_lipschitz_response --run-dir artifacts/synthetic_lipschitz_response/l1_s012',
        '```','','Use a fresh output directory. Python, NumPy, PyTorch and Matplotlib versions and source hashes are recorded. '
        'Baseline reuse requires the six local L=1 checkpoints under artifacts/synthetic_shared_response/l025_1_4_s012; '
        'their expected hashes are in its completion.json. No remote server or PointMaze dataset is needed.','']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
