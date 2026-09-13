"""从已保存数组重建 Frobenius-versus-spectral ETT 中文报告。"""
import argparse
import json
from pathlib import Path

import numpy as np

from ett import pointmaze_norm_comparison as experiment
from ett import pointmaze_phase_sampling as phase
from ett.finite_crl import sha, write
from ett.rollout_return import GOAL, task_reward


def read(path):
    return json.loads(Path(path).read_text())


def resolved_negative(row):
    return row['root_ci95'][1] < 0 and row['conditional_mc_ci95'][1] < 0


def heldout_interval(difference, groups, offset=0):
    difference = np.asarray(difference, np.float64)
    means = difference.mean(1)
    rng = np.random.default_rng(experiment.CONFIG['bootstrap_seed'] + offset)
    index = np.concatenate([rng.choice(np.flatnonzero(groups == group),
        (experiment.CONFIG['bootstrap_repeats'], int(np.sum(groups == group))))
        for group in np.unique(groups)], axis=1)
    estimate = float(difference.mean())
    se = float(np.sqrt(np.sum(difference.var(1, ddof=1) / difference.shape[1])) / len(difference))
    return dict(mean=estimate, root_ci95=np.quantile(means[index].mean(1), [.025, .975]),
                conditional_mc_ci95=[estimate - 1.96 * se, estimate + 1.96 * se])


def reset_interval(difference, offset=0):
    difference = np.asarray(difference, np.float64)
    rng = np.random.default_rng(experiment.CONFIG['bootstrap_seed'] + offset)
    index = rng.integers(len(difference), size=(experiment.CONFIG['bootstrap_repeats'], len(difference)))
    return dict(mean=float(difference.mean()), paired_ci95=np.quantile(difference[index].mean(1), [.025, .975]))


def recompute_full(path, count, repeats):
    records = phase.load_records(path)
    values = np.empty((count, repeats), np.float64)
    for record in records:
        horizon = record['reward'].shape[1]
        np.testing.assert_array_equal(record['states'][:, 1:, 2:], record['states'][:, :-1, :6])
        expected = np.asarray(task_reward(record['states'][:, 1:],
                              np.broadcast_to(GOAL, record['states'][:, 1:].shape)))
        np.testing.assert_array_equal(record['reward'], expected)
        for root in np.unique(record['root']):
            selected = record['root'] == root
            values[root] = .05 * record['reward'][selected] @ .95 ** np.arange(horizon)
    return values, records


def distribution(values):
    q = np.quantile(values, [.1, .25, .5, .75, .9])
    return dict(mean=float(np.mean(values)), p10=float(q[0]), p25=float(q[1]),
                median=float(q[2]), p75=float(q[3]), p90=float(q[4]))


def outside_mechanism(records):
    values = {name: [] for name in ('projected', 'boundary', 'base_corrected', 'response_norm')}
    count = 0
    for record in records:
        before = record['states'][:, :-1]
        outside = np.asarray(task_reward(before, np.broadcast_to(GOAL, before.shape))) == 0
        count += int(outside.sum())
        for name in values:
            values[name].append(np.asarray(record[name])[outside])
    merged = {name: np.concatenate(rows) for name, rows in values.items()}
    return dict(outside_steps=count, emitted_box_clip_fraction=float(merged['projected'].mean()),
        box_boundary_fraction=float(merged['boundary'].mean()),
        diagonal_base_correction_fraction=float(merged['base_corrected'].mean()),
        response_norm_mean=float(merged['response_norm'].mean()),
        response_norm_p90=float(np.quantile(merged['response_norm'], .9)))


def local_audits(out, training):
    results = {}
    for name, history in training['histories'].items():
        results[name] = []
        for row in history:
            if row['update'] not in experiment.CONFIG['audit_updates']:
                continue
            folder = out / 'audits' / f"{name}_u{row['update']}"
            raw = dict(np.load(folder / 'matched.npz', allow_pickle=False))
            critic_delta = (raw['proposed__critic'] - raw['before__critic']).mean(-1)
            mc_delta = (raw['proposed__mc'] - raw['before__mc']).mean(-1)
            actual = {label: experiment.uncertainty(value, raw['query__root'], index)
                      for index, (label, value) in enumerate([
                          ('critic', critic_delta), ('mc', mc_delta),
                          ('mc_minus_critic', mc_delta - critic_delta)])}
            stored = read(folder / 'matched.json')
            for label in actual:
                np.testing.assert_allclose(actual[label]['root_ci95'], stored[label]['root_ci95'], atol=1e-12)
                np.testing.assert_allclose(actual[label]['conditional_mc_ci95'],
                                           stored[label]['conditional_mc_ci95'], atol=1e-12)
            actual['update'] = row['update']
            actual['critic_resolved_negative'] = resolved_negative(actual['critic'])
            actual['mc_resolved_negative'] = resolved_negative(actual['mc'])
            actual['critic_negative_without_mc_support'] = bool(
                actual['critic_resolved_negative'] and not actual['mc_resolved_negative'])
            results[name].append(actual)
    return results


def training_constraints(training):
    result = {}
    for name, history in training['histories'].items():
        signed_active = sum(row['signed_constraint_active_count'] for row in history)
        signed_total = sum(row['signed_constraint_total'] for row in history)
        proposal_active = sum(row['proposal_constraint']['active_count'] for row in history)
        proposal_total = 8 * len(history)
        result[name] = dict(signed_active_blocks=signed_active, signed_total_blocks=signed_total,
            signed_activation_fraction=signed_active / signed_total,
            proposal_active_blocks=proposal_active, proposal_total_blocks=proposal_total,
            proposal_activation_fraction=proposal_active / proposal_total,
            max_signed_raw_block_norm=max(row['signed_constraint_max_raw_norm'] for row in history),
            max_proposal_raw_block_norm=max(max(row['proposal_constraint']['raw_block_norms']) for row in history),
            final_block_norms=experiment.block_norms(history[-1]['theta'][16:], name.split('_s')[0]))
    return result


def meaningful(row):
    return (row['mean'] <= -experiment.CONFIG['meaningful_return_effect'] and
            resolved_negative(row))


def analyze(out):
    completion = read(out / 'completion.json')
    if completion['status'] != 'complete' or completion['charged_model_outputs'] != experiment.CONFIG['exact_model_output_budget']:
        raise ValueError('incomplete or wrong-budget experiment')
    training = read(out / 'training.json')
    if sha(out / 'final_parameters.npz') != completion['final_parameters_sha256']:
        raise ValueError('final parameters changed')
    with np.load(experiment.FINAL_ROOTS, allow_pickle=False) as roots:
        groups = np.asarray(roots['indices'][:, 2])
    stored = dict(np.load(out / 'evaluation_returns.npz', allow_pickle=False))
    model_names = [f'{kind}_s{seed}' for seed in experiment.SEEDS
                   for kind in ('init', 'frobenius', 'spectral')]
    models, records = {}, {}
    for name in model_names:
        values, heldout_records = recompute_full(out / 'evaluation' / f'{name}_heldout.npz', 36, 64)
        np.testing.assert_array_equal(values, stored[name])
        reset = dict(np.load(out / 'evaluation' / f'{name}_reset.npz', allow_pickle=False))
        reset_values = .05 * reset['reward'] @ .95 ** np.arange(50)
        np.testing.assert_allclose(reset_values, stored[name + '__reset'], rtol=0, atol=1e-12)
        expected = np.asarray(task_reward(reset['states'][:, 1:],
                              np.broadcast_to(GOAL, reset['states'][:, 1:].shape)))
        np.testing.assert_array_equal(reset['reward'], expected)
        reset_record = dict(reset)
        models[name] = dict(heldout=distribution(values), reset=distribution(reset_values),
            outside_mechanism=dict(heldout=outside_mechanism(heldout_records),
                                   reset=outside_mechanism([reset_record])))
        records[name] = (values, reset_values)
    comparisons = {}
    effect_flags = {}
    for seed in experiment.SEEDS:
        init = records[f'init_s{seed}']
        comparisons[str(seed)] = {}
        for mode in experiment.MODES:
            final = records[f'{mode}_s{seed}']
            row = heldout_interval(final[0] - init[0], groups, seed * 10 + len(comparisons[str(seed)]))
            row['meaningful_improvement'] = meaningful(row)
            row['reset'] = reset_interval(final[1] - init[1], seed * 10 + 20)
            comparisons[str(seed)][mode + '_minus_init'] = row
            effect_flags[f'{mode}_s{seed}'] = row['meaningful_improvement']
        spectral = records[f'spectral_s{seed}']
        frobenius = records[f'frobenius_s{seed}']
        row = heldout_interval(spectral[0] - frobenius[0], groups, seed * 10 + 5)
        row['meaningful_spectral_advantage'] = meaningful(row)
        row['reset'] = reset_interval(spectral[1] - frobenius[1], seed * 10 + 25)
        comparisons[str(seed)]['spectral_minus_frobenius'] = row
    local = local_audits(out, training)
    mismatch = {}
    for name, rows in local.items():
        count = sum(row['critic_negative_without_mc_support'] for row in rows)
        mismatch[name] = dict(count=count, of=3,
            full_improvement=effect_flags[name], criterion_met=bool(count >= 2 and not effect_flags[name]))
    spectral_advantage = all(comparisons[str(seed)]['spectral_minus_frobenius']['meaningful_spectral_advantage']
                             for seed in experiment.SEEDS)
    critic_mismatch = any(row['criterion_met'] for row in mismatch.values())
    neither = not any(effect_flags.values()) and not spectral_advantage
    if spectral_advantage:
        decision = dict(category=2, code='constraint_change_improves_independent_return',
            text='两个种子的谱范数臂都相对 Frobenius 达到预设的 0.01 独立回报改善，并通过两类区间；约束改变改善了独立评估回报。')
    elif critic_mismatch:
        decision = dict(category=1, code='critic_prediction_not_supported_by_rollout',
            text='至少一个臂在多数预设检查点由 critic 解析地预测改善，但配对 MC 不支持，且最终独立回报未达效应阈值；证据识别出 critic-to-rollout 不一致。')
    elif neither:
        decision = dict(category=3, code='neither_arm_improves_unresolved',
            text='Frobenius 与谱范数臂都没有达到预设的独立回报改善标准；representation、projection 或 optimization 仍未解析。')
    else:
        decision = dict(category=None, code='mixed_inconclusive',
            text='结果不属于三个预设纯模式：存在局部或单种子信号，但没有一致的谱范数优势；证据仍属 mixed/inconclusive。')
    return dict(models=models, comparisons=comparisons, local_audits=local,
        critic_mismatch=mismatch, training_constraints=training_constraints(training),
        final_constraints=read(out / 'constraints.json'), evidence_flags=dict(
            spectral_advantage_both_seeds=spectral_advantage,
            critic_prediction_without_mc_support=critic_mismatch,
            neither_arm_seed_meaningfully_improves=neither), decision=decision,
        scope=dict(reward_unchanged=True, actor_frozen=True, nominal_frozen=True,
                   diagonal_frozen=True, native_steps=0, worst_case_solution=False,
                   oracle_03440b6_not_optimizer_only_evidence=True))


def fmt_interval(row):
    return (f"{row['mean']:+.5f}；root 95% CI [{row['root_ci95'][0]:+.5f}, {row['root_ci95'][1]:+.5f}]；"
            f"conditional-MC 95% CI [{row['conditional_mc_ci95'][0]:+.5f}, {row['conditional_mc_ci95'][1]:+.5f}]")


def render(results, completion):
    lines = ['# ETT Frobenius 与谱范数约束的配对优化诊断', '',
        f"**结论（类别 {results['decision']['category']}）：{results['decision']['text']}**", '',
        '本实验没有修改任务来制造改善。奖励仍是下一状态 newest XY 到固定目标 `(8.5,3.5)` 距离严格小于 2 的指示函数；'
        '折扣为 0.95，标准化回报为 `0.05*sum gamma^t r_t`。nominal expert policy、diagonal transition、fixed actor、'
        '归一化、rectangle 几何和墙/边界处理全部冻结。', '',
        '`outputs/lipschitz_audit/REPORT.md` 在当前 worktree 和主 workspace 均不存在；本报告因此以用户给出的审计结论、'
        '当前实现和已保存的 `1d6deef` 产物为依据。起点是实际训练后的 `s0_k0_u24` / `s1_k0_u24` 和对应 critic，'
        '不是只拿 `817e429` 的旧 response 做静态评估。', '',
        '## 独立 full-rollout 回报', '',
        '以下为 held-out 36 roots×64；差值越负越悲观。每个候选在轨迹每一步生效。', '']
    for seed in experiment.SEEDS:
        models = results['models']
        lines.append(f"- seed {seed} 均值：init {models[f'init_s{seed}']['heldout']['mean']:.5f}；"
                     f"Frobenius {models[f'frobenius_s{seed}']['heldout']['mean']:.5f}；"
                     f"spectral {models[f'spectral_s{seed}']['heldout']['mean']:.5f}。")
        for key in ('frobenius_minus_init', 'spectral_minus_init', 'spectral_minus_frobenius'):
            row = results['comparisons'][str(seed)][key]
            lines.append(f"  - {key}: {fmt_interval(row)}；START delta {row['reset']['mean']:+.5f}, "
                         f"95% CI [{row['reset']['paired_ci95'][0]:+.5f}, {row['reset']['paired_ci95'][1]:+.5f}]。")
    lines += ['', '## Critic 预测与配对 MC', '',
        '更新 1/3/6 的检查只改变第一步候选，之后固定为各自 pre-update ETT；critic 与 MC 共用 query、successor 和 next action。', '']
    for name, rows in results['local_audits'].items():
        lines.append(f"- {name}：critic 负向但 MC 未支持 {results['critic_mismatch'][name]['count']}/3；"
                     f"类别-1条件 {results['critic_mismatch'][name]['criterion_met']}。")
        for row in rows:
            lines.append(f"  - update {row['update']}: critic {fmt_interval(row['critic'])}；MC {fmt_interval(row['mc'])}；"
                         f"critic-only flag {row['critic_negative_without_mc_support']}。")
    lines += ['', '## 约束激活与 reward-region 外投影', '']
    for name, row in results['training_constraints'].items():
        lines.append(f"- {name}：signed block 激活 {row['signed_active_blocks']}/{row['signed_total_blocks']} "
                     f"({100*row['signed_activation_fraction']:.2f}%)；proposal 激活 {row['proposal_active_blocks']}/"
                     f"{row['proposal_total_blocks']} ({100*row['proposal_activation_fraction']:.2f}%)；"
                     f"最大 raw norm {row['max_proposal_raw_block_norm']:.4f}。")
    for seed in experiment.SEEDS:
        for mode in experiment.MODES:
            name = f'{mode}_s{seed}'
            held = results['models'][name]['outside_mechanism']['heldout']
            reset = results['models'][name]['outside_mechanism']['reset']
            lines.append(f"- {name} reward-region 外：held-out box clip/boundary/base correction "
                         f"{100*held['emitted_box_clip_fraction']:.2f}%/{100*held['box_boundary_fraction']:.2f}%/"
                         f"{100*held['diagonal_base_correction_fraction']:.2f}%；START "
                         f"{100*reset['emitted_box_clip_fraction']:.2f}%/{100*reset['box_boundary_fraction']:.2f}%/"
                         f"{100*reset['diagonal_base_correction_fraction']:.2f}%。")
    lines += ['', '所有最终 component operator norm、固定条件 action-pair grid、rectangle/F4 history 和 bitwise diagonal identity '
        '检查均通过。谱范数模式同时用于 emitter、signed finite-difference candidates 和 optimizer proposal；每个 ETT checkpoint '
        '都保存并强制校验 `constraint_mode`。', '', '## 证据边界', '',
        f"模型转移精确用量 {completion['charged_model_outputs']:,}/{completion['cap']:,}；24 个 response updates，10,000 个 critic steps；"
        '零 native steps、零 actor/nominal/diagonal updates，无 sweep、选择或重跑。', '',
        'Frobenius 排除 identity 等合法 operator-norm<=1 响应是表示事实，但只有当本次优化实际触碰差异边界且独立回报分离时，'
        '才能把不可靠改善归因于该限制。`03440b6` 的手工 oracle 使用更宽的非线性/answer-handed 响应族，其低回报不能推出“只是优化器问题”。'
        'Native 自由运动的局部常数为 1 也不能越过碰撞阈值推广为全局样本级声明。较低 critic surrogate 不是 worst-case 解，'
        '本实验不测量 native-policy benefit。', '',
        '完整机器可读结果见 `results.json`；原始局部配对数组见 `audits/*/matched.npz`；独立轨迹见 `evaluation/`。', '']
    return '\n'.join(lines)


def summary(results):
    return '\n'.join(['# 结论摘要', '', results['decision']['text'], '',
        f"spectral 双种子独立优势：{results['evidence_flags']['spectral_advantage_both_seeds']}；"
        f"critic-to-MC 不一致：{results['evidence_flags']['critic_prediction_without_mc_support']}；"
        f"两臂均未达到效应阈值：{results['evidence_flags']['neither_arm_seed_meaningfully_improves']}。", '',
        '该结果不改变奖励，不启动 actor 训练，不构成 worst-case 解，也不证明任何 native-policy benefit。', ''])


def run(out):
    experiment.verify(out)
    completion = read(out / 'completion.json')
    results = analyze(out)
    write(out / 'results.json', results)
    (out / 'REPORT.md').write_text(render(results, completion), encoding='utf-8')
    (out / 'SUMMARY.md').write_text(summary(results), encoding='utf-8')
    manifest = {str(path.relative_to(out)): sha(path) for path in out.rglob('*')
                if path.is_file() and path.name != 'manifest.json'}
    write(out / 'manifest.json', dict(files=manifest, report_model_transitions=0,
        headline_rederived_from_saved_arrays=True))
    print(results['decision']['text'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=experiment.OUT)
    run(parser.parse_args().out)
