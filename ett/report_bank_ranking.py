"""English aggregate report and plots for the recorded PointMaze bank audit."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ett.run_bank_ranking import read, write
from scripts.make_swamp_f4_failure_bank import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args(); root = Path(args.run_dir)
    m, p = read(root/'metrics.json'), read(root/'provenance.json')
    pop, usage = read(root/'population.json'), read(root/'reference_usage.json')
    assert read(root/'replay_checks.json')['status']=='passed'
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    plotted = read(root/'plot_data.json'); edges = np.array(plotted['log10_distance_edges']); x = (edges[:-1]+edges[1:])/2
    for k in ['onset','age1','age2','established']:
        y = np.array(plotted['histograms'][k]); axes[0].plot(x, y/y.sum(), label=k)
    for k in ['onset','established','alive_waiting','alive_collision_proxy','alive_moving']:
        y = np.array(plotted['histograms'][k]); axes[1].plot(x, y/y.sum(), label=k.replace('_',' '))
    for ax in axes:
        ax.set(xlabel='log10 full-F4 squared distance (smaller is failure-like)', ylabel='Fraction per histogram bin')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0].set_title('Actual next observations by death age')
    axes[1].set_title('Alive comparison populations')
    fig.savefig(root/'distance_distributions.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    sequence = m['sequences']; quantiles = np.array([v['quantiles'] for v in sequence['complete_distance']])
    axes[0].plot(range(5), quantiles[:, 3], 'o-', label=f'Median, {sequence["complete_episodes"]} complete episodes')
    axes[0].fill_between(range(5), quantiles[:, 2], quantiles[:, 4], alpha=.2, label='Interquartile range')
    for j, v in enumerate(sequence['examples']):
        axes[1].plot(range(5), v, 'o-', label=f'Sequence {j+1}')
    for ax in axes:
        ax.set(yscale='log', xticks=range(5), xlabel='Death age (0 = first observation after fatal contact)', ylabel='Full-F4 squared distance')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_title('Same episodes: information accumulates by age 3')
    axes[1].set_title('First six complete sequences; no score selection')
    fig.savefig(root/'actual_onset_sequences.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for j, phase in enumerate(['onset','established_first']):
        r = m['matched'][phase]['ranking']; lo, hi = r['interval95']
        axes[0].errorbar(j, r['accuracy_half_ties'], yerr=[[r['accuracy_half_ties']-lo],[hi-r['accuracy_half_ties']]], fmt='o', capsize=5)
        axes[0].annotate(f'n={r["pairs"]}', (j, r['accuracy_half_ties']), xytext=(5,-18), textcoords='offset points')
    axes[0].axhline(.5, color='gray', linestyle='--')
    axes[0].set(xticks=[0,1], xticklabels=['Onset','Age 3'], ylim=(0,1.08), ylabel='Failed observation ranks closer', title='Disjoint-episode matching')
    axes[0].text(.5,.08,'All-win bootstrap at age 3 is degenerate;\n32 pairs are not a certainty guarantee.',ha='center',transform=axes[0].transAxes,fontsize=8)
    labels = ['All alive','Cell 3,3','Cell 4,3','Cell 5,3','Waiting','Collision proxy']
    keys = ['onset_vs_alive']+[f'onset_vs_alive__swamp_cell_{j}_3' for j in [3,4,5]]+['onset_vs_alive_waiting','onset_vs_alive_collision_proxy']
    axes[1].barh(labels, [m['comparisons'][k]['roc_auc'] for k in keys])
    axes[1].axvline(.5,color='gray',linestyle='--'); axes[1].set(xlim=(0,1),xlabel='Onset ROC-AUC',title='Location and behavior change the picture')
    ages = ['onset','age1','age2','established']
    newest = [m['strata'][k]['newest']['mean'] for k in ages]
    hist = [m['strata'][k]['history']['mean'] for k in ages]
    axes[2].bar(ages, newest, label='Newest XY')
    axes[2].bar(ages, hist, bottom=newest, label='Old three frames')
    axes[2].set(ylabel='Mean full-F4 squared distance',title='Contributions at the SAME reference'); axes[2].legend(fontsize=8)
    fig.savefig(root/'matching_and_confounding.png', dpi=160); plt.close(fig)

    top = sorted(usage, key=lambda v:-v['selected_rows'])[:12]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar([str(v['index']) for v in top], [v['selected_rows'] for v in top])
    axes[0].set(xlabel='Saved retained-bank index',ylabel='Selected outcome rows',title='Most frequently selected references')
    for phase in ['onset','established']:
        keys = [f'{phase}_vs_alive__time_{a}_{a+9}' for a in range(1,51,10)]
        axes[1].plot(range(5), [m['comparisons'][k]['roc_auc'] for k in keys], 'o-', label=phase)
    axes[1].set(xticks=range(5),xticklabels=['1-10','11-20','21-30','31-40','41-50'],ylim=(0,1.05),xlabel='Outcome observation row',ylabel='ROC-AUC versus alive',title='Time strata (late onset counts are small)')
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(root/'reference_usage_and_time.png', dpi=160); plt.close(fig)

    lines = ['# Current PointMaze F4: recorded-outcome failure-bank ranking', '',
        '**Decision: useful for established observed failure; no evidence of useful immediate onset ranking after context matching. Do not use this result to justify a one-step pessimistic ETT target.**', '',
        '## What this adds', '',
        'This audit evaluates the existing nearest-reference distance on actual recorded next observations with aligned evaluation-only death truth. The earlier failure-objective report studied MMD versus set distance on generated and constructed candidates, including drift and collapse. The later supervised failure-scoring report trained classifiers. Neither establishes the recorded-outcome ranking performance of this frozen distance. Their reports and the death-observability report were read before this audit. No critic diagnostic, classifier training, generated-candidate evaluation or model update is repeated.', '',
        'The historical flow-plus-nearest-failure selection experiment may have been AntMaze. Its bank, normalization, representation and success rates are not used as PointMaze evidence or reconstructed here.', '',
        '## Frozen artifacts and representation', '',
        f'- Environment: `{p["environment"]}`, per-cell active probability 0.30, fixed collector horizon 50 actions, 6,600 episodes with 51 stored observations.',
        f'- Dataset: `{p["dataset"]}`; SHA256 `{p["input_sha256"][p["dataset"]]}`.',
        f'- Original current bank: `{p["bank"]}`; SHA256 `{p["input_sha256"][p["bank"]]}`.',
        '- The original visible-entry pool had 1,438 entries; the curated bank file contains 256. The existing guarded ETT experiment retained 233 after excluding its 23 full-source validation episodes. This audit uses exactly those same 233 vectors in saved order. The unretained original 23 episodes are also excluded from evaluation. No entry is rebuilt, filtered using new outcomes, reweighted or replaced.',
        '- Bank episodes and observation rows are recorded in provenance.json for both original and retained entries and checked against the actual dataset vectors. The 233 references come from 141 uniform-coverage, 12 original teacher-source and 80 blind-demonstrator episodes, as documented by the existing run.',
        '- The bank was curated as random=0.6, deliberate=0.4 using privileged `teacher_mode` and pre-action `swamp_bits`, with death cross-checks. Visible stored vectors do not erase that selection provenance. No raw privileged fields are published by this audit.',
        '- The shorter worktree bank path and root dataset hold stale p=0.1 artifacts. They are explicitly rejected; the verified p=0.30 downloaded artifacts are used.', '',
        'The state is `[p_t,p_(t-1),p_(t-2),p_(t-3)]`, eight newest-first XY coordinates. The goal is a separate eight-coordinate tile of (8.5,3.5), constant across this dataset. The established distance scores only the state block. Goal is verified for task compatibility and used in matching, not silently added to the distance.', '',
        'The score calls `ett.failure_objectives.nearest_reference` unchanged in float32: `min_f sum_j ((y_j-f_j)/std_j)^2`. The original expert-training-next-state normalizer is loaded from the guarded experiment. Its standard deviations are approximately [1.99874,0.23877,2.21035,0.23777,2.39379,0.23676,2.55499,0.23577]; its common mean cancels. Neither mean nor scale is refitted. Smaller distance ranks as more failure-like. Exact reference ties choose the first saved index; pair ranking assigns half credit to an exact score tie. There are no operating thresholds.', '',
        '## Alignment, population and exclusions', '',
        'The collector stores observation s_t, pre-action mask U_t and action a_t before calling step. Physics moves the point, tests the landed cell against U_t, sets absorbing death, then redraws the mask; F4 pushes the new XY afterward. The first dead observation is s_(t+1), age 0. Pre-contact history remains at ages 0/1/2; age 3 has four repeated coordinates. Death rows are reconstructed from fatal landing and masks, then checked against the per-episode death audit and independently against the established ETT labels. Neither stillness, action, reward nor source membership defines death.', '',
        'Unlike the previous classifier evaluation of pre-action rows 0..49, the primary population here is actual next-state rows 1..50. The terminal next observation is valid even though its stored action is a dummy; the score never consumes that dummy action. Incoming action and displacement from row t-1 are used only for alive audit strata. Padding is excluded via episode lengths. Reset row 0 has no preceding transition and is supplementary only. Reset clears death and seeds equal frames. The environment ignores post-death actions, holds XY fixed and emits zero reward; death does not set done. The collector enforces horizon 50. Rewards and true collision flags are not stored, so no future-return or confirmed-collision analysis is fabricated.', '',
        f'The established 660-episode seed-0 source holdout loses 23 original-bank episodes, leaving {pop["evaluation_episodes"]} episodes and {pop["outcome_rows"]:,} next states. There are 117 death episodes, 4,605 dead outcomes and 27,245 alive outcomes (14.46% dead). All 637 terminal next observations are retained. No included fatal contact occurs on the final action; three death episodes are censored before age 3. Age 0/1/2/3/4 available sequence counts are 117/117/116/114/113. Censoring never creates synthetic observations.', '',
        'These data are not untouched: the original actor saw all source episodes, the expert subset was previously used in nominal/diagonal/observability evaluation, and these are part of the prior classifier test population. Of the 637 included episodes, 425 appeared in the prior objective-development context set. Metrics outside those episodes are supplied but remain upstream-reused. There are no complete observed-trajectory copies of original bank episodes or duplicate complete evaluation trajectories. Known source ranges are uniform coverage [0,1200), teacher-positive [1200,6000), and blind demonstrator [6000,6600); source membership is not an expert-quality or survival label.', '',
        'Alive strata overlap deliberately: waiting means incoming recorded action <=1e-7 in each coordinate; low motion means incoming displacement <=0.05; collision-compatible proxy means action norm >0.05 and displacement <=0.05; moving means displacement >0.05. Exact equal stacks after reset are separate. Action noise means zero action need not imply zero movement.', '']
    for name in ['onset','age1','age2','established','alive_waiting','alive_low_motion','alive_collision_proxy','alive_equal_after_reset','alive_moving']:
        v=m['strata'][name]
        lines.append(f'- {name}: {v["rows"]:,} rows / {v["episodes"]} episodes; median distance {v["distance"]["quantiles"][3]:.6g}.')
    lines += ['', 'All 637 alive reset stacks have distance 5.02040; they are not failed merely because stationary. Only one alive equal stack occurs after reset, so general recognition against persistent stationary-alive states has insufficient support. Single-class strata have null ROC-AUC/AP, not invented discrimination scores.', '',
        '## Overall recognition versus onset', '']
    for name in ['all']:
        v=m['strata'][name]; lines.append(f'All outcomes: ROC-AUC {v["roc_auc"]:.6f}, average precision {v["average_precision"]:.6f}.')
    for name in ['onset_vs_alive','age1_vs_alive','age2_vs_alive','established_vs_alive','onset_vs_alive_waiting','established_vs_alive_waiting','onset_vs_alive_collision_proxy']:
        v=m['comparisons'][name]
        lines.append(f'- {name}: ROC-AUC {v["roc_auc"]:.6f}; average precision {v["average_precision"]:.6f}; positive prevalence {100*v["prevalence"]:.3f}%.')
    lines += ['', 'These aggregate metrics weight outcome rows, so prolonged death observations dominate. They are descriptive; uncertainty for the harder primary comparisons is episode-aware below. Average precision depends on prevalence and must not be compared without the reported class proportions. Actual collision labels are unavailable and the proxy populations are not spatially balanced.', '',
        '![Distance distributions](distance_distributions.png)', '', '## Predetermined matched comparisons', '',
        'The protocol was written before any real score was calculated. For each phase take one failed observation per episode: age 0 or first established age 3. Candidate controls are alive next outcomes from another included episode, same source membership, commanded goal and newest-position cell. Require newest XY Euclidean distance <=0.25, previous XY distance <=0.5 and observation-row difference <=5. Do not match the score or the whole F4 history. Greedily process failed observations in fixed seed-93172 order and minimize the sum of the three normalized context gaps. Matching-cost ties follow original episode/row order. Each episode is used at most once, in either role, within a phase. Phases are matched separately. Controls may fail later but are alive at the scored observation.', '',
        'This is approximate context adjustment, not the same hidden or causal context. Conditioning on outcome position itself restricts the question to similarly located recorded outcomes. It cannot identify ETT or eliminate behavior-mode and history differences. Each bootstrap block contains a matched pair of disjoint episodes; 1,000 fixed-seed replicates preserve episode dependence. Intervals condition on this fixed bank and matching, and do not include rematching or upstream-selection uncertainty.', '']
    for phase in ['onset','established_first']:
        v=m['matched'][phase]; c=v['coverage']; r=v['ranking']
        lines.append(f'- {phase}: {c["failed_candidates"]} failed candidates, {c["failed_with_context_support_before_reuse"]} have context support before reuse restrictions, {r["pairs"]} matched pairs ({100*c["coverage"]:.1f}% coverage), {c["unmatched_failed"]} unmatched. Failed outcome ranks closer in {100*r["accuracy_half_ties"]:.2f}% (bootstrap interval {100*r["interval95"][0]:.2f}% to {100*r["interval95"][1]:.2f}%), with {100*r["ties"]:.1f}% ties. Mean newest/previous XY gaps {c["xy_gap"]["mean"]:.3f}/{c["previous_xy_gap"]["mean"]:.3f}; mean time gap {c["time_gap"]["mean"]:.3f} steps.')
    lines += ['', 'Onset is 40/83, consistent with chance under the fixed matching. Established failure is 32/32, but only 28.1% of eligible episodes match. The all-win bootstrap collapses to [1,1] because every resampled observed pair wins; it is not evidence of zero uncertainty or a population guarantee. This limitation is especially severe for tiny subgroup counts.', '',
        'Among matched controls, onset has only 2 waiting/low-motion pairs (one win), 81 moving pairs, and zero collision-compatible pairs. Established failure has 8 waiting/low-motion pairs, 24 moving pairs and zero collision-compatible pairs; all observed pairs win. Source-specific onset accuracy is 8/18 uniform, 22/40 teacher-source and 10/25 blind-demonstrator. Established source support is only 10/21/1 pairs. Thus the harder collision question and source-general established recognition are insufficiently supported. Unmatched examples remain in descriptive metrics; tolerances were not enlarged after inspecting coverage.', '',
        '![Matching, confounding and distance components](matching_and_confounding.png)', '', '## Location, history and reference reuse', '',
        'Onset AUC falls from 0.8406 overall to 0.3241, 0.4538 and 0.5542 within swamp cells (3,3), (4,3) and (5,3). Waiting comparisons also reverse the apparent onset ordering (AUC 0.3201): alive waiting can be closer to the bank than newly fatal outcomes. Source and time breakdowns are in metrics.json; late time bins contain only 17, 11, 7 and 4 onset examples after the first bin of 78. These results expose strong spatial and behavior-population confounding rather than establish a universal onset signal.', '',
        'Distance decomposition uses the SAME full-F4-selected reference. The old three frames contribute about 70.2%, 82.1%, 90.9% and 71.9% of total distance at ages 0,1,2 and established failure. These shares are ratios of summed contributions, not independently minimized scores and not causal attributions. A large share alone is not proof that immobility dominates: three historical frames naturally carry more coordinates. The observed temporal sequence supplies stronger evidence.', '',
        'On the same 113 complete onset episodes, median distances at ages 0..4 are approximately 1.9990, 1.3614, 0.3413, 0.01949, 0.01949. Actual examples can initially increase in distance; every complete sequence is closer by age 3 than at age 0. Once all frames repeat, score no longer changes. Recognizing established failure after three additional observations cannot be called an immediately available one-step signal.', '',
        '![Actual onset sequences](actual_onset_sequences.png)', '',
        'The chosen-reference newest-XY and historical component rankings are saved diagnostically. They are not alternative selected metrics: the nearest index already depends on the entire history. In particular, the high established newest-component AUC must not be interpreted as the performance of an independently chosen XY-only nearest bank. At matched onset their accuracies are 46.99% and 53.01%, respectively. Location compatibility and historical settling together explain the observed score behavior; this audit does not causally separate them.', '',
        f'{sum(v["selected_rows"]>0 for v in usage)} of 233 references are selected. The top three indices 189, 144 and 143 account for {100*sum(v["selected_rows"] for v in top[:3])/pop["outcome_rows"]:.2f}% of all outcomes. Most selections for these references are alive rows; being the nearest reference does not imply small absolute distance or failure. reference_usage.json records every saved index, source episode/row, selection count and episode count. The bank and its frequencies remain unchanged.', '',
        'There are zero exact original-bank vector matches in evaluation and no cross-bank complete-trajectory copies; removing exact bank overlaps therefore leaves the results unchanged. Repeated visible vectors account for 4,141 extra rows, largely prolonged deaths. A declared sensitivity keeps the first chronological occurrence of each exact visible vector, retaining all 114 distinct established outcomes: established-versus-alive AUC remains 0.999432, while AP falls from 0.995097 to 0.888425 as prevalence drops to 0.4167%. This changes population weighting, not the bank. No identical visible vector has conflicting alive/dead labels in this sample; this is not a general identifiability claim.', '',
        '![Reference reuse and time strata](reference_usage_and_time.png)', '', '## Decision and remaining support gaps', '',
        '- **Established recognition:** useful within this recorded dataset; strong overall and exact-vector sensitivity results, plus 32 successful matched pairs. Low matching coverage and sparse stationary-alive/collision support limit generalization.',
        '- **Onset ranking:** not supported by the harder matched comparison; the high broad AUC is misleading without location/source/time controls.',
        '- **Waiting, collision and location:** waiting is a concrete onset confounder; cell-conditioned results expose location effects. Collision-compatible descriptive metrics look strong but no matched collision pair exists, so collision-specific discrimination remains unestablished.',
        '- **Optimization meaning:** the bank measures similarity to curated, settled histories at stored hazard locations. Direct optimization could favor location proximity or accumulated immobility without correctly ranking immediate adverse outcomes. The prior generated-candidate drift findings remain relevant context, not new results here.', '',
        'Stop before any generator, ETT or policy optimization. If this signal is pursued, a multi-step observed-outcome score is more appropriate to investigate than an immediate one-step failure target, with new support for alive waiting and confirmed collisions. Onset results do not justify proceeding directly to a frozen-generator candidate-ranking study. No such follow-up is run. This audit does not validate generated counterfactual states, the historical AntMaze selection experiment, a Lipschitz condition, causal ETT identification or worst-case Q.', '',
        '## Checks and reproduction', '',
        'Seven focused tests passed: fatal landing/next-state alignment, horizon and stationary-alive handling, original score plus same-reference decomposition, reference tie handling, matching/reuse, metric direction/ties and episode exclusion of original versus retained bank sources. Runtime checks verify all F4 shifts, reset stacks, absorbing post-death XY, age-3 histories and agreement with existing audit labels. An independent replay reproduces primary aggregate metrics and matching exactly and checks observed scores against NumPy arithmetic. Dataset, banks, normalizers, relevant historical records and existing model files remain hash-identical. No model is trained or sampled.', '',
        'An initial preflight implementation error (bank metadata not returned to the caller) was fixed before the output directory or any score was produced. It did not change the protocol, bank or evaluation population. The completed audit took ' + f'{read(root/"verification.json")["elapsed_seconds"]:.2f}' + ' seconds, excluding later verification and reporting. Raw audit masks, teacher modes and per-row death labels are not published; only aggregate results, declared episode-set provenance and anonymous scalar sequence plots are saved.', '',
        '```bash',
        'python -m unittest scripts.test_bank_ranking',
        'python -m ett.run_bank_ranking --out-dir artifacts/bank_ranking/f4_p30_recorded_v1',
        'python -m scripts.check_bank_ranking --run-dir artifacts/bank_ranking/f4_p30_recorded_v1',
        'python -m ett.report_bank_ranking --run-dir artifacts/bank_ranking/f4_p30_recorded_v1',
        '```', '',
        'Run from the PointMaze worktree and use a fresh output directory. Required files are resolved from the existing guarded config and checked by hash; absent or mismatched current data/bank files fail explicitly. No AntMaze fallback exists. The protocol, exact normalizer, source and artifact hashes, matching coverage and aggregate metrics accompany this report. No automatic commit or push is performed.', '']
    (root/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write(root/'report_manifest.json', dict(source_sha256=file_sha(__file__),
        artifact_sha256={p.name:file_sha(p) for p in [root/'REPORT.md', *root.glob('*.png')]}))
    print('Wrote English report and four aggregate figures.')


if __name__=='__main__':
    main()
