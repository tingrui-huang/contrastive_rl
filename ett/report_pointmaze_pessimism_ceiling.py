"""Re-derive the PointMaze pessimism-ceiling report from saved arrays only."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from ett.finite_crl import sha, write
from ett.pointmaze_pessimism_ceiling import ARM_NAMES, ARTIFACT, CONFIG


QUANTILES = (.10, .25, .50, .75, .90)


def load_setting(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = {k: np.asarray(data[k]) for k in data.files if '__' not in k}
        arms = {}
        for arm in ARM_NAMES:
            prefix = arm + '__'
            arms[arm] = {k[len(prefix):]: np.asarray(data[k])
                         for k in data.files if k.startswith(prefix)}
    return metadata, arms


def distribution(values):
    values = np.asarray(values).reshape(-1)
    q = np.quantile(values, QUANTILES)
    return dict(n=int(values.size), mean=float(values.mean()), p10=float(q[0]),
                p25=float(q[1]), median=float(q[2]), p75=float(q[3]), p90=float(q[4]))


def stratified_indices(groups, rng):
    return np.concatenate([
        rng.choice(np.flatnonzero(groups == group),
                   (CONFIG['bootstrap_repeats'], int(np.sum(groups == group))), replace=True)
        for group in np.unique(groups)], axis=1)


def paired_summary(difference, setting, groups=None, seed_offset=0):
    difference = np.asarray(difference, np.float64)
    rng = np.random.default_rng(CONFIG['bootstrap_seed'] + seed_offset)
    estimate = float(difference.mean())
    median = float(np.median(difference))
    if setting == 'setting1':
        index = stratified_indices(np.asarray(groups), rng)
        sampled = difference[index]
        boot_mean = sampled.mean(axis=(1, 2))
        boot_median = np.median(sampled, axis=(1, 2))
        root_means = difference.mean(axis=1)
        conditional_se = float(np.sqrt(np.sum(difference.var(1, ddof=1) /
                                                difference.shape[1])) / len(difference))
        conditional_ci = [estimate - 1.96 * conditional_se,
                          estimate + 1.96 * conditional_se]
        root_ci = np.quantile(boot_mean, [.025, .975])
        interval = dict(root_stratified_mean_ci95=root_ci,
                        conditional_mc_se=conditional_se,
                        conditional_mc_mean_ci95=conditional_ci,
                        root_stratified_median_ci95=np.quantile(boot_median, [.025, .975]),
                        root_mean_sd=float(root_means.std(ddof=1)))
    else:
        flat = difference.reshape(-1)
        index = rng.integers(0, len(flat), (CONFIG['bootstrap_repeats'], len(flat)))
        sampled = flat[index]
        interval = dict(paired_rollout_mean_ci95=np.quantile(sampled.mean(1), [.025, .975]),
                        paired_rollout_median_ci95=np.quantile(np.median(sampled, axis=1), [.025, .975]))

    flat = difference.reshape(-1)
    number = max(1, int(math.ceil(.05 * len(flat))))
    selected = np.argsort(np.abs(flat))[-number:]
    total_signed = float(flat.sum())
    total_absolute = float(np.abs(flat).sum())
    sign = np.sign(estimate)
    same_direction = int(np.sum((np.sign(flat) == sign) & (flat != 0))) if sign else 0
    result = dict(mean_difference=estimate, paired_median_difference=median,
        top5_count=number,
        top5_signed_contribution_share=(float(flat[selected].sum() / total_signed)
                                        if total_signed != 0 else None),
        top5_absolute_contribution_share=(float(np.abs(flat[selected]).sum() / total_absolute)
                                          if total_absolute != 0 else None),
        same_direction_nonzero_support=same_direction,
        tail_determined=bool(estimate != 0 and same_direction < CONFIG['tail_support_threshold']),
        median_is_primary=bool(estimate != 0 and same_direction < CONFIG['tail_support_threshold']))
    result.update(interval)
    return result


def relative_change(value, baseline):
    if baseline == 0:
        return None
    return float((value - baseline) / baseline)


def vector_stats(values):
    values = np.asarray(values).reshape(-1)
    if len(values) == 0:
        return dict(n=0, mean=None, median=None, p90=None, max=None)
    return dict(n=int(len(values)), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), max=float(values.max()))


def fraction(values):
    values = np.asarray(values)
    return None if values.size == 0 else float(values.mean())


def mechanism(record, reference):
    reward = np.asarray(record['reward']).reshape(-1, record['reward'].shape[-1])
    ref_reward = np.asarray(reference['reward']).reshape(-1, reference['reward'].shape[-1])
    delta = np.asarray(record['delta_norm']).reshape(reward.shape)
    clipped = np.asarray(record['box_clip_active']).reshape(reward.shape)
    prefix = np.cumsum(reward, axis=1) - reward
    pre_entry = prefix == 0
    ref_prefix = np.cumsum(ref_reward, axis=1) - ref_reward
    ref_pre_entry = ref_prefix == 0
    suffix_changed = np.zeros_like(reward, dtype=bool)
    changed_now = reward != ref_reward
    suffix_changed[:, -1] = changed_now[:, -1]
    for t in range(reward.shape[1] - 2, -1, -1):
        suffix_changed[:, t] = changed_now[:, t] | suffix_changed[:, t + 1]
    phases = {}
    for name, mask in [('pre_entry', pre_entry), ('entered', ~pre_entry)]:
        phases[name] = dict(delta_norm=vector_stats(delta[mask]),
                            active_box_clip_fraction=fraction(clipped[mask]),
                            decisions=int(mask.sum()))
    changed_split = {}
    for name, mask in [('pre_entry', ref_pre_entry), ('entered', ~ref_pre_entry)]:
        changed_split[name] = dict(decisions=int(mask.sum()),
                                   suffix_reward_sequence_changed_fraction=fraction(suffix_changed[mask]))
    return dict(delta_norm=vector_stats(delta),
                active_box_clip_fraction=fraction(clipped), phases=phases,
                paired_full_reward_sequence_changed_fraction=float(np.any(changed_now, axis=1).mean()),
                paired_suffix_change_by_a0_phase=changed_split)


def arm_results(arms, setting, groups=None):
    baseline = np.asarray(arms['a0_diagonal']['return'])
    base_distribution = distribution(baseline)
    output = {}
    for number, arm in enumerate(ARM_NAMES):
        values = np.asarray(arms[arm]['return'])
        expected = .05 * (np.asarray(arms[arm]['reward']) @
                           (.95 ** np.arange(arms[arm]['reward'].shape[-1])))
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-7)
        summary = distribution(values)
        c_mean = relative_change(summary['mean'], base_distribution['mean'])
        c_p25 = relative_change(summary['p25'], base_distribution['p25'])
        summary.update(signed_relative_mean_change=c_mean,
                       mean_reduction_magnitude=(-c_mean if c_mean is not None else None),
                       signed_relative_p25_change=c_p25,
                       p25_reduction_magnitude=(-c_p25 if c_p25 is not None else None),
                       versus_a0=paired_summary(values - baseline, setting, groups, number),
                       mechanism=mechanism(arms[arm], arms['a0_diagonal']))
        output[arm] = summary
    return output


def direct_contrasts(arms, setting, groups=None):
    pairs = [('a3_minus_a2', 'a3_box_oracle', 'a2_lipschitz_oracle'),
             ('a4_minus_a3', 'a4_postprocess_oracle', 'a3_box_oracle')]
    result = {}
    for number, (name, left, right) in enumerate(pairs, 100):
        result[name] = paired_summary(np.asarray(arms[left]['return']) -
                                      np.asarray(arms[right]['return']),
                                      setting, groups, number)
    return result


def reduction_at_least(row, key, threshold):
    value = row[key]
    return value is not None and value >= threshold


def direct_below_zero(row):
    return (row['root_stratified_mean_ci95'][1] < 0 and
            row['conditional_mc_mean_ci95'][1] < 0)


def decide(setting1, direct):
    a3, a4 = setting1['a3_box_oracle'], setting1['a4_postprocess_oracle']
    a3_success = (reduction_at_least(a3, 'mean_reduction_magnitude', .10) or
                  reduction_at_least(a3, 'p25_reduction_magnitude', .10))
    a4_success = (reduction_at_least(a4, 'mean_reduction_magnitude', .10) or
                  reduction_at_least(a4, 'p25_reduction_magnitude', .10))
    if not a3_success and not a4_success:
        return dict(rule=1, code='reward_geometry_or_postprocess_binding',
            decision='A3 and A4 do not reach a 10% reduction in either the mean or p25; reward geometry/postprocessing is the measured binding factor, subject to the rare-move and lower-quantile caveat.')
    a2_r = setting1['a2_lipschitz_oracle']['mean_reduction_magnitude']
    a1_r = [setting1[a]['mean_reduction_magnitude'] for a in
            ('a1_s0_learned', 'a1_s1_learned')]
    a1_r = [x for x in a1_r if x is not None]
    if a2_r is not None and a2_r >= .10 and a1_r and max(a1_r) < .03:
        return dict(rule=2, code='optimizer_bottleneck_not_family_limit',
            decision='A2 reaches at least 10% mean reduction while both learned A1 responses remain below 3%; the bounded evidence favors an optimizer bottleneck rather than the response family as the limit.')
    a3_r = a3['mean_reduction_magnitude']
    if (a3_r is not None and a2_r is not None and a3_r - a2_r >= .05 and
            direct_below_zero(direct['a3_minus_a2'])):
        return dict(rule=3, code='lipschitz_bound_binding',
            decision='A3 exceeds A2 by at least five mean-reduction percentage points with both paired intervals below zero; the L=1 response bound is the measured binding factor.')
    return dict(rule=5, code='measured_ceiling_no_separation',
        decision='The predeclared separations are not met; this run establishes only the measured oracle headroom/no-separation result under the frozen model protocol.')


def setting_disagreement(s1, s2):
    rows = {}
    any_material = False
    for arm in ARM_NAMES[1:]:
        r1, r2 = s1[arm]['mean_reduction_magnitude'], s2[arm]['mean_reduction_magnitude']
        d1, d2 = s1[arm]['versus_a0']['mean_difference'], s2[arm]['versus_a0']['mean_difference']
        opposite = bool(d1 * d2 < 0)
        gap = None if r1 is None or r2 is None else float(abs(r1 - r2))
        material = opposite or (gap is not None and gap >= .05)
        any_material |= material
        rows[arm] = dict(setting1_reduction=r1, setting2_reduction=r2,
                         absolute_reduction_gap=gap, opposite_mean_direction=opposite,
                         material=material)
    return dict(any_material=any_material, arms=rows,
                definition='opposite paired mean direction or >=5 percentage-point mean-reduction gap')


def f(value, signed=False):
    if value is None:
        return 'unavailable'
    return f'{value:+.5f}' if signed else f'{value:.5f}'


def pct(value):
    return 'unavailable' if value is None else f'{100 * value:+.2f}%'


def ci(row, setting):
    key = 'root_stratified_mean_ci95' if setting == 'setting1' else 'paired_rollout_mean_ci95'
    v = row[key]
    return f'[{v[0]:+.5f}, {v[1]:+.5f}]'


def render_report(results, acceptance, completion):
    lines = ['# ETT pessimism headroom ceiling diagnostic', '', results['decision']['decision'], '',
        'This one-shot diagnostic uses answer-handed model oracles. Lower return is more pessimistic. '
        'Oracle results are achievable constructions under the specified frozen rules, not valid lower bounds '
        'on the true worst case.', '', '## Setting 1: 36 held-out roots, fresh H=10', '',
        'Returns are normalized discounted reward-region occupancy. The roots are the exact 36 group-balanced '
        'held-out roots associated with the learned A1 artifacts, but their stored source times are 1/3/6/10, '
        'not t=40. This resolves an inconsistency in the requested contract without selecting 36 of the separate '
        '64-root t=40 artifact.', '']
    for arm in ARM_NAMES:
        row = results['settings']['setting1']['arms'][arm]
        diff = row['versus_a0']
        lines.append(f"- {arm}: mean {row['mean']:.5f}; p10/p25/median/p75/p90 "
                     f"{row['p10']:.5f}/{row['p25']:.5f}/{row['median']:.5f}/{row['p75']:.5f}/{row['p90']:.5f}; "
                     f"paired mean {diff['mean_difference']:+.5f}, root CI {ci(diff, 'setting1')}, "
                     f"conditional-MC CI [{diff['conditional_mc_mean_ci95'][0]:+.5f}, {diff['conditional_mc_mean_ci95'][1]:+.5f}]; "
                     f"paired median {diff['paired_median_difference']:+.5f}; signed mean change {pct(row['signed_relative_mean_change'])}.")
    lines += ['', '## Setting 2: 128 independent START rollouts, H=50', '',
        'This is the mandatory reset-start transport comparison; it is not pooled with setting 1.', '']
    for arm in ARM_NAMES:
        row = results['settings']['setting2']['arms'][arm]
        diff = row['versus_a0']
        lines.append(f"- {arm}: mean {row['mean']:.5f}; p10/p25/median/p75/p90 "
                     f"{row['p10']:.5f}/{row['p25']:.5f}/{row['median']:.5f}/{row['p75']:.5f}/{row['p90']:.5f}; "
                     f"paired mean {diff['mean_difference']:+.5f}, paired-rollout CI {ci(diff, 'setting2')}; "
                     f"paired median {diff['paired_median_difference']:+.5f}; signed mean change {pct(row['signed_relative_mean_change'])}.")
    disagreement = results['setting_comparison']
    lines += ['', '## Cross-setting comparison', '',
        f"Material disagreement by the sealed definition: {disagreement['any_material']}. "
        'The definition is opposite paired-mean direction or at least a five percentage-point reduction gap.', '']
    for arm, row in disagreement['arms'].items():
        lines.append(f"- {arm}: setting-1 R {pct(row['setting1_reduction'])}; setting-2 R "
                     f"{pct(row['setting2_reduction'])}; gap {pct(row['absolute_reduction_gap'])}; "
                     f"opposite direction {row['opposite_mean_direction']}.")
    lines += ['', '## Tail sensitivity and mechanism', '',
        'Top-5% shares use paired rollout differences ranked by absolute contribution. A signed share can exceed '
        'one under cancellation. `support` counts nonzero rollouts pointing in the mean direction; below 20 makes '
        'the paired median the predeclared primary summary.', '']
    for setting in ('setting1', 'setting2'):
        lines.append(f"- {setting}:")
        for arm in ARM_NAMES[1:]:
            row = results['settings'][setting]['arms'][arm]
            d, m = row['versus_a0'], row['mechanism']
            lines.append(f"  - {arm}: top-5% signed/absolute {f(d['top5_signed_contribution_share'])}/"
                         f"{f(d['top5_absolute_contribution_share'])}; support {d['same_direction_nonzero_support']}; "
                         f"tail flag {d['tail_determined']}; delta norm mean/median/p90/max "
                         f"{f(m['delta_norm']['mean'])}/{f(m['delta_norm']['median'])}/"
                         f"{f(m['delta_norm']['p90'])}/{f(m['delta_norm']['max'])}; box clip "
                         f"{pct(m['active_box_clip_fraction'])}; full reward-sequence changed "
                         f"{pct(m['paired_full_reward_sequence_changed_fraction'])}.")
    lines += ['', 'Pre-entry versus entered mechanism details, including decision counts, delta summaries, '
        'box clips, and suffix reward-sequence changes, are retained in `results.json`.', '',
        '## Acceptance, cost, and limits', '',
        f"- Exact diagonal identity passed for every arm: {acceptance['diagonal_identity_all_exact']}.",
        f"- A2 samplewise Lipschitz grid passed with maximum excess "
        f"{acceptance['lipschitz']['a2_lipschitz_oracle']['max_excess']:.3g}.",
        f"- A3/A4 Lipschitz grid pass states (failure expected and retained): "
        f"{acceptance['lipschitz']['a3_box_oracle']['passed']}/"
        f"{acceptance['lipschitz']['a4_postprocess_oracle']['passed']}.",
        f"- Charged model outputs: {completion['charged_model_outputs']:,}/{completion['cap']:,}. "
        'A5 was skipped before outcomes because its exact 3,774,720-output design exceeds 1.5M.',
        '- Frozen diagonal, nominal, actor checkpoint, normalization, and response hashes match before/after. '
        'All saved rollouts passed finite, anchor, coordinate-step, free-endpoint, exact F4-shift, and reward checks.',
        '- There were zero updates, gradients, native steps, actor changes, or checkpoint writes. No critic or '
        'Monte Carlo value was loaded.', '',
        'The comparison is conditional on one learned checkpoint pair, one diagonal model, one actor/nominal pair, '
        'the specified goal and synthetic transition postprocessing. Small oracle effects are not certified upper '
        'bounds: a different admissible construction, deeper search, or rare lower-quantile move could do more. '
        'These model trajectories do not establish physical realizability or native policy benefit.', '',
        'Re-derive this report without rollouts: `python -m ett.report_pointmaze_pessimism_ceiling '
        '--out-dir artifacts/pointmaze_region_pilot/pessimism_ceiling_v1`.', '']
    return '\n'.join(lines)


def render_summary(results):
    s1 = results['settings']['setting1']['arms']
    s2 = results['settings']['setting2']['arms']
    strongest1 = min(ARM_NAMES[1:], key=lambda a: s1[a]['mean'])
    strongest2 = min(ARM_NAMES[1:], key=lambda a: s2[a]['mean'])
    return '\n'.join(['# Summary: ETT pessimism headroom ceiling', '',
        f"Setting 1 A0 mean was {s1['a0_diagonal']['mean']:.5f}; the lowest arm was {strongest1} at "
        f"{s1[strongest1]['mean']:.5f} ({pct(s1[strongest1]['signed_relative_mean_change'])} signed change).",
        '',
        f"Setting 2 A0 mean was {s2['a0_diagonal']['mean']:.5f}; the lowest arm was {strongest2} at "
        f"{s2[strongest2]['mean']:.5f} ({pct(s2[strongest2]['signed_relative_mean_change'])} signed change). "
        f"Material cross-setting disagreement: {results['setting_comparison']['any_material']}.", '',
        f"Decision (rule {results['decision']['rule']}): {results['decision']['decision']} This establishes "
        'neither global worst-case recovery, nor a valid bound, nor any native-policy benefit.', ''])


def report(out):
    completion = json.loads((out / 'completion.json').read_text())
    if completion['status'] != 'complete':
        raise ValueError('rollout phase is incomplete')
    for name, digest in completion['raw_sha256'].items():
        if sha(out / name) != digest:
            raise ValueError(f'raw array hash mismatch: {name}')
    meta1, arms1 = load_setting(out / 'setting1_raw.npz')
    _, arms2 = load_setting(out / 'setting2_raw.npz')
    setting1 = arm_results(arms1, 'setting1', meta1['groups'])
    setting2 = arm_results(arms2, 'setting2')
    direct1 = direct_contrasts(arms1, 'setting1', meta1['groups'])
    direct2 = direct_contrasts(arms2, 'setting2')
    results = dict(
        estimand='normalized discounted reward occupancy; lower is more pessimistic',
        settings=dict(setting1=dict(arms=setting1, direct_contrasts=direct1,
                                    roots=36, repeats=64, horizon=10,
                                    source_time_caveat='stored source times are 1/3/6/10, not 40'),
                      setting2=dict(arms=setting2, direct_contrasts=direct2,
                                    rollouts=128, horizon=50, start='fixed START')),
        setting_comparison=setting_disagreement(setting1, setting2),
        decision=decide(setting1, direct1),
        a5=dict(status='skipped_pre_outcome', exact_total_outputs=3774720,
                cap=1500000),
        interpretation_limits=dict(answer_handed_oracles_are_lower_bounds_on_achievable_pessimism=True,
            certified_ceiling=False, valid_bound=False, global_worst_case_recovery=False,
            native_policy_benefit=False))
    acceptance = json.loads((out / 'acceptance_probes.json').read_text())
    write(out / 'results.json', results)
    (out / 'REPORT.md').write_text(render_report(results, acceptance, completion), encoding='utf-8')
    (out / 'SUMMARY.md').write_text(render_summary(results), encoding='utf-8')
    note = Path('notes/pointmaze_pessimism_ceiling.md')
    note.write_text(render_summary(results) + '\n\nSee the artifact REPORT.md and results.json for the full '
                    'pre-entry/entered mechanism audit and bootstrap intervals.\n', encoding='utf-8')
    manifest = {path.name: sha(path) for path in sorted(out.iterdir())
                if path.is_file() and path.name != 'manifest.json'}
    write(out / 'manifest.json', dict(files=manifest, report_derivation='saved arrays only',
                                       rollout_calls=0))
    print(results['decision']['decision'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ARTIFACT)
    args = parser.parse_args()
    report(args.out_dir)


if __name__ == '__main__':
    main()
