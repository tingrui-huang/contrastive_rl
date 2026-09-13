"""Paired Frobenius-versus-spectral ETT response optimization diagnostic."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as j
import numpy as np
import torch

from ett import pointmaze_early_pilot as early
from ett import pointmaze_phase_sampling as phase
from ett import pointmaze_region_pilot as old
from ett import pointmaze_response_search as search
from ett import pointmaze_update_reference as update
from ett.convex_action_transition import convex_box, gates, validate_selected_set
from ett.diagonal_transition import _project_samples, sample_displacement
from ett.finite_crl import sha, write
from ett.policy_improvement import tree_sha
from ett.rollout_return import GOAL, START, task_reward


OUT = Path('outputs/ett_frobenius_spectral_v1')
SOURCE = Path('artifacts/pointmaze_region_pilot/response_search_s01_v1')
CONTEXT = Path('artifacts/pointmaze_region_pilot/early_contexts_s01_v1/contexts.npz')
FINAL_ROOTS = Path('artifacts/pointmaze_region_pilot/phase_sampling_s01_v1/final_contexts.npz')
MODES = ('frobenius', 'spectral')
SEEDS = (0, 1)
CONFIG = dict(
    base_commit='03440b6', source_commit='1d6deef', modes=list(MODES), seeds=list(SEEDS),
    source_parameters={str(s): f's{s}_k0_u24' for s in SEEDS},
    source_critics={str(s): f'checks/s{s}_k0_u24/critic.pt' for s in SEEDS},
    updates=6, directions=8, sigma=.1, response_lr=.05, response_step_cap=.25,
    adam_betas=[.9, .999], adam_eps=1e-8, critic_lr=.003,
    first_refresh_steps=1000, later_refresh_steps=300, total_critic_steps=10000,
    paths_per_root=4, query_count=32, successors=4, actor_draws=4,
    audit_updates=[1, 3, 6], audit_successors=8, audit_actor_draws=4,
    final_repeats=64, reset_repeats=128, bootstrap_repeats=2000,
    training_seed=202000000, matched_mc_seed=203000000,
    final_seed=204000000, reset_seed=205000000,
    constraint_seed=206000000, bootstrap_seed=201000000,
    gamma=.95, return_scale=.05, geometry='rectangle', bound=1.,
    meaningful_return_effect=.01, numerical_tolerance=2e-6,
    training_visitation_outputs=163776, optimizer_probe_outputs=55296,
    matched_audit_outputs=1185792, heldout_evaluation_outputs=650112,
    reset_evaluation_outputs=38400, constraint_outputs=8448,
    exact_model_output_budget=2101824, hard_cap=2101824,
    response_updates=24, native_steps=0, actor_updates=0, diagonal_updates=0,
    selection='fixed update 6; no sweep, early stopping, retry, or outcome selection')


def protocol_text():
    return """# Frobenius 与谱范数 ETT 配对比较：预注册协议

## 固定对象与问题

本实验只比较 2x2 响应块的约束方式。任务奖励保持
`1[||next_xy-(8.5,3.5)||<2]`，折扣保持 0.95，报告回报保持
`0.05*sum_t 0.95^t r_t`。冻结原 nominal expert policy、完整 diagonal
transition（含每个起点已有的 16 个 head offsets）、fixed actor、归一化、
rectangle 选择、边界/墙处理。不得改奖励、actor 或几何来制造改进。

两个模式都从 `1d6deef` 实际训练所得 `s0_k0_u24`、`s1_k0_u24`
参数开始，并从对应 `checks/s*_k0_u24/critic.pt` 初始化 critic。每个种子内
Frobenius 与 spectral 起点逐 bit 相同；critic 模型起点相同，Adam 都从
空状态开始。两臂使用相同的路径、actor、nominal、anchor、方向、查询和
critic minibatch 随机流。

## 一致约束

- `frobenius`: 每个原始 2x2 块除以 `max(1, ||M||_F)`。
- `spectral`: 每个原始 2x2 块除以 `max(1, ||M||_2)`。

同一模式同时用于 (a) emitted transition 解码，(b) 每个有限差分 signed
candidate，(c) optimizer 最终 proposal。Checkpoint 必须带
`constraint_mode`, `bound=1`, `format_version=1`；加载时严格校验。
两模式的 softmax convex mixture 均满足 operator norm <=1，因此在固定
`(s,g,x',noise)` 和固定 rectangle 下保留样本级 action-Lipschitz 结论。
这不是 native 碰撞动力学的全局 Lipschitz 声明。

## 优化与局部审计

每个 mode/seed 固定 6 个 response-only Adam-style ES 更新；每次 8 个
antithetic 方向、sigma=.1、lr=.05、response step cap=.25。每次从 36 个
训练 roots 各取 4 条当前模型完整剩余轨迹。Critic 第一次 refresh 1000
步，其后各 300 步；架构、归一化、batch=256、lr=.003 与现有 averaged
positive NCE 保持不变。总计 24 response updates、10,000 critic steps。

在更新 1/3/6，对 proposal 与 pre-update 做新鲜配对检查：32 queries、
8 successor draws、4 actor draws；critic 与 MC 共用 query、first successor
和 next action，MC continuation 固定使用 pre-update ETT，计算 48 个槽并
对真实剩余 horizon mask。记录 critic delta、MC delta、各自 root bootstrap
与 conditional simulation 95% 区间，以及约束激活率。

## 独立全轨迹评估

固定评估初始 checkpoint 和 update-6 checkpoint，不做选择。每个种子评估
6 个模型中的三份：shared init、Frobenius final、spectral final。使用相同
新随机流，分别运行 (1) 36 held-out roots 各 64 条实际剩余 horizon 路径；
(2) START 出发 128 条 50-step 路径。每个 ETT 在每一步生效。主要比较是
final-minus-init 与 spectral-minus-Frobenius。Held-out 区间按已有三个 root
groups 分层重采样并另报 conditional MC 区间；START 按配对路径重采样。

在 pre-state 尚未进入 reward region 的评估步骤上，报告 emitted box clip、
diagonal base correction、box boundary rate；另报 signed candidates 与最终
proposals 的 block constraint activation。

## 预算、效应与判据

精确新模型转移预算：训练 visitation 163,776；optimizer probes 55,296；
三次局部审计 1,185,792；held-out 650,112；START 38,400；constraint grid
8,448；合计 **2,101,824/2,101,824**。超过即中止。Critic steps 不计模型
转移，但固定为 10,000。零 native steps，零 actor/nominal/diagonal updates。

有意义的独立回报效应预设为 0.01。一次 final 改进需 mean delta<=-0.01，
且 held-out root 与 conditional-MC 两个 95% 区间上界都<0。谱约束优于
Frobenius 需两个种子的 spectral-minus-Frobenius 都满足该条件。

判别：

1. 若某 mode/seed 至少 2/3 个审计点显示 critic 的负向区间已解析，而 MC
   未支持负向，并且该 final 没有独立有意义改进，则识别 critic-to-rollout
   不一致。
2. 若两个种子的 spectral-minus-Frobenius 都达到上述 0.01 独立标准，则
   识别约束改变改善独立回报。
3. 若没有任何 mode/seed 达到 final-minus-init 独立标准，且规则 2 不成立，
   则两臂均未改善；representation、projection 或 optimization 仍未解析。

若证据不落入三种纯模式，报告 mixed/inconclusive，不追加 sweep。较低
surrogate 或 hand-designed oracle 均不构成 worst-case 解。
"""


def timestamp():
    from ett.pointmaze_future_average import timestamp as now
    return now()


class Ledger:
    def __init__(self, out):
        self.out = out
        self.data = dict(cap=CONFIG['hard_cap'], charged=0, entries=[])

    def add(self, number, purpose):
        number = int(number)
        if self.data['charged'] + number > self.data['cap']:
            raise RuntimeError('predeclared model-transition cap exceeded')
        self.data['charged'] += number
        self.data['entries'].append(dict(purpose=purpose, outputs=number))
        write(self.out / 'ledger.json', self.data)


def block_norms(value, mode):
    value = np.asarray(value).reshape(8, 2, 2)
    if mode == 'frobenius':
        return np.linalg.norm(value, axis=(1, 2))
    if mode == 'spectral':
        return np.linalg.svd(value, compute_uv=False)[..., 0]
    raise ValueError(mode)


def project_response(value, mode):
    raw = np.asarray(value, np.float64).reshape(8, 2, 2)
    norms = block_norms(raw, mode)
    projected = raw / np.maximum(1., norms)[:, None, None]
    info = dict(raw_block_norms=norms, active=norms > 1.,
                active_count=int(np.sum(norms > 1.)), active_fraction=float(np.mean(norms > 1.)))
    return projected.astype(np.float32).ravel(), info


def matrices_mode(theta, mode, bound=1.):
    raw = j.asarray(theta).reshape(8, 2, 2)
    if mode == 'frobenius':
        norm = j.sqrt(j.sum(raw ** 2, axis=(1, 2), keepdims=True))
    elif mode == 'spectral':
        norm = j.linalg.svd(raw, compute_uv=False)[..., :1, None]
    else:
        raise ValueError(mode)
    return bound * raw / j.maximum(1., norm)


def emit_mode(theta, state, action, xp, anchor, mode, bound=1.):
    low, high, valid, rectangle = convex_box(state, anchor)
    matrix = j.einsum('bj,jkl->bkl', gates(state), matrices_mode(theta, mode, bound))
    response = j.einsum('bij,bj->bi', matrix, action - xp)
    proposal = anchor + response[:, None, :]
    xy = j.clip(proposal, low, high)
    xy = j.where(valid[..., None], xy, j.nan)
    history = j.broadcast_to(state[:, None, :6], xy.shape[:-1] + (6,))
    output = j.concatenate([xy, history], axis=-1)
    return output, dict(anchor_xy=anchor, box_low=low, box_high=high, box_valid=valid,
        rectangle=rectangle, response=response,
        projection_corrected=j.any(xy != proposal, axis=-1),
        projected_to_boundary=j.any((xy == low) | (xy == high), axis=-1),
        emitted_change=j.linalg.norm(xy - anchor, axis=-1))


class ModeKernel(old.Kernel):
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.constraint_mode = mode
        super().__init__()

    def _sample(self, theta, state, action, xp, key, count):
        goal = j.broadcast_to(j.asarray(GOAL), state.shape)
        context = j.concatenate([state, xp, xp, goal], -1)
        distribution = self.base._distribution(self.parameters(theta), context)
        delta, atom = sample_displacement(distribution, key, count, self.base.delta_mean,
                                          self.base.delta_std, self.base.spec)
        anchor, details = _project_samples(state, delta, self.base.spec)
        output, diagnostic = emit_mode(theta[16:], state, action, xp, anchor[..., :2],
                                       self.constraint_mode, 1.)
        diagnostic['stationary_atom'] = atom
        diagnostic['base_corrected'] = j.any(details['raw_position'] != anchor[..., :2], axis=-1)
        return output, diagnostic

    def rollout(self, theta, states, key, horizon=old.H, first_actions=None):
        if horizon not in self.rollouts:
            def generate(theta, states, key, first, override):
                def step(state, data):
                    t, step_key = data
                    nominal_key, actor_key, transition_key = jax.random.split(step_key, 3)
                    goal = j.broadcast_to(j.asarray(GOAL), state.shape)
                    xp = self.nominal.sample(state, nominal_key, 1, goal=goal)
                    action = self.actor(state, goal, actor_key)
                    action = j.where((t == 0) & override, first, action)
                    output, detail = self._sample(theta, state, action, xp, transition_key, 1)
                    following = output[:, 0]
                    return following, dict(next=following, action=action, x_prime=xp,
                        reward=task_reward(following, goal), valid=detail['box_valid'][:, 0],
                        atom=detail['stationary_atom'][:, 0],
                        projected=detail['projection_corrected'][:, 0],
                        boundary=detail['projected_to_boundary'][:, 0],
                        base_corrected=detail['base_corrected'][:, 0],
                        response_norm=j.linalg.norm(detail['response'], axis=-1))
                _, data = jax.lax.scan(step, states,
                    (j.arange(horizon), jax.random.split(key, horizon)))
                data = jax.tree.map(lambda value: j.swapaxes(value, 0, 1), data)
                data['states'] = j.concatenate([states[:, None], data.pop('next')], 1)
                return data
            self.rollouts[horizon] = jax.jit(generate)
        first = np.zeros((len(states), 2), np.float32) if first_actions is None else first_actions
        result = jax.tree.map(np.asarray, self.rollouts[horizon](
            j.asarray(theta), j.asarray(states), key, j.asarray(first), first_actions is not None))
        if not result['valid'].all() or not np.isfinite(result['states']).all():
            raise ValueError('invalid rollout')
        np.testing.assert_array_equal(result['states'][:, 1:, 2:], result['states'][:, :-1, :6])
        if np.abs(result['action']).max() > 1 or np.abs(result['x_prime']).max() > 1:
            raise AssertionError('action bounds failed')
        expected = np.asarray(task_reward(result['states'][:, 1:],
                              np.broadcast_to(GOAL, result['states'][:, 1:].shape)))
        np.testing.assert_array_equal(result['reward'], expected)
        return result


def save_checkpoint(path, theta, mode, seed, update_number, source_key):
    if Path(path).exists():
        raise FileExistsError(path)
    with open(path, 'xb') as stream:
        np.savez(stream, theta=np.asarray(theta, np.float32),
                 constraint_mode=np.asarray(mode), bound=np.float32(1.),
                 format_version=np.int32(1), seed=np.int32(seed),
                 update=np.int32(update_number), source_key=np.asarray(source_key))


def load_checkpoint(path, required_mode=None):
    with np.load(path, allow_pickle=False) as data:
        required = {'theta', 'constraint_mode', 'bound', 'format_version', 'seed', 'update', 'source_key'}
        if set(data.files) != required or int(data['format_version']) != 1 or float(data['bound']) != 1.:
            raise ValueError('invalid mode checkpoint')
        mode = str(data['constraint_mode'].item())
        if mode not in MODES or (required_mode is not None and mode != required_mode):
            raise ValueError('constraint mode mismatch')
        theta = np.asarray(data['theta'], np.float32)
        if theta.shape != (48,) or not np.isfinite(theta).all():
            raise ValueError('invalid theta')
        return theta, dict(mode=mode, seed=int(data['seed']), update=int(data['update']),
                           source_key=str(data['source_key'].item()))


def response_step(theta, directions, signed, momentum, variance, step_number, mode):
    gradient = np.mean((signed[:, 0] - signed[:, 1])[:, None] * directions, axis=0) / (2 * CONFIG['sigma'])
    momentum = .9 * momentum + .1 * gradient
    variance = .999 * variance + .001 * gradient ** 2
    delta = CONFIG['response_lr'] * (momentum / (1 - .9 ** step_number)) / (
        np.sqrt(variance / (1 - .999 ** step_number)) + CONFIG['adam_eps'])
    delta *= min(1., CONFIG['response_step_cap'] / max(np.linalg.norm(delta), 1e-15))
    projected, projection = project_response(theta[16:] - delta, mode)
    candidate = np.r_[theta[:16], projected].astype(np.float32)
    return candidate, momentum, variance, dict(gradient=gradient, requested_step=delta,
        actual_step_norm=float(np.linalg.norm(candidate[16:] - theta[16:])),
        proposal_constraint=projection)


def uncertainty(samples, roots, offset=0):
    samples = np.asarray(samples, np.float64)
    means = samples.mean(1)
    root_ids, inverse = np.unique(roots, return_inverse=True)
    sums = np.bincount(inverse, weights=means)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(CONFIG['bootstrap_seed'] + offset)
    index = rng.integers(len(sums), size=(CONFIG['bootstrap_repeats'], len(sums)))
    value = float(samples.mean())
    se = float(np.sqrt(np.sum(samples.var(1, ddof=1) / samples.shape[1])) / len(samples))
    return dict(mean=value,
        root_ci95=np.quantile(sums[index].sum(1) / counts[index].sum(1), [.025, .975]),
        conditional_mc_ci95=[value - 1.96 * se, value + 1.96 * se], roots=len(root_ids))


def matched_check(engine, continuation, pre, proposed, records, critic, context,
                  seed, ledger, folder):
    queries = update.visitation(records, seed, CONFIG['query_count'])
    raw = {}
    for label, theta in [('before', pre), ('proposed', proposed)]:
        first = search.first_samples(engine, theta, queries, critic, context, seed + 10,
                                     CONFIG['audit_successors'], CONFIG['audit_actor_draws'],
                                     ledger, 'audit/first_transition')
        state = np.repeat(first['successor'].reshape(-1, 8), CONFIG['audit_actor_draws'], 0)
        action = first['next_action'].reshape(-1, 2)
        remaining = np.repeat(queries['h'] - 1,
                              CONFIG['audit_successors'] * CONFIG['audit_actor_draws'])
        ledger.add(len(state) * 48, 'audit/continuation_including_padding')
        rewards, valid = continuation.run(pre, state, action, remaining,
                                          jax.random.PRNGKey(seed + 20))
        rewards = np.asarray(rewards)
        if not np.asarray(valid).all():
            raise ValueError('invalid matched continuation')
        if not np.all(rewards[np.arange(48)[None, :] >= remaining[:, None]] == 0):
            raise AssertionError('continuation mask failed')
        mc_q = (.05 * rewards @ .95 ** np.arange(48)).reshape(
            CONFIG['query_count'], CONFIG['audit_successors'], CONFIG['audit_actor_draws'])
        first.update(mc=queries['weight'][:, None, None] *
            (.05 * first['reward'][:, :, None] + .95 * mc_q),
            continuation_rewards=rewards.reshape(CONFIG['query_count'],
                CONFIG['audit_successors'], CONFIG['audit_actor_draws'], 48),
            remaining=remaining.reshape(CONFIG['query_count'], CONFIG['audit_successors'],
                                        CONFIG['audit_actor_draws']), mc_q=mc_q)
        raw[label] = first
    np.testing.assert_array_equal(raw['before']['x_prime'], raw['proposed']['x_prime'])
    critic_delta = (raw['proposed']['critic'] - raw['before']['critic']).mean(-1)
    mc_delta = (raw['proposed']['mc'] - raw['before']['mc']).mean(-1)
    result = {name: uncertainty(value, queries['root'], number) for number, (name, value) in enumerate([
        ('critic', critic_delta), ('mc', mc_delta), ('mc_minus_critic', mc_delta - critic_delta)])}
    np.savez_compressed(folder / 'matched.npz',
        **{label + '__' + key: value for label, row in raw.items() for key, value in row.items()},
        **{'query__' + key: value for key, value in queries.items()}, preupdate=pre, proposed=proposed)
    write(folder / 'matched.json', result)
    return result


def source_inputs():
    critic_paths = [SOURCE / CONFIG['source_critics'][str(seed)] for seed in SEEDS]
    frozen_setup_inputs = list(old.inputs().values())
    return [SOURCE / 'candidate_pool.npz', SOURCE / 'manifest.json', *critic_paths,
        CONTEXT, FINAL_ROOTS, *frozen_setup_inputs,
        Path('ett/pointmaze_norm_comparison.py'), Path('ett/report_pointmaze_norm_comparison.py'),
        Path('scripts/test_pointmaze_norm_comparison.py'), Path('ett/convex_action_transition.py'),
        Path('ett/diagonal_transition.py'), Path('ett/rollout_return.py'),
        Path('ett/run_convex_adversarial.py'), Path('ett/pointmaze_region_pilot.py'),
        Path('ett/pointmaze_early_pilot.py'), Path('ett/pointmaze_response_search.py'),
        Path('ett/pointmaze_update_reference.py')]


def prepare(out):
    if out.exists():
        raise ValueError('fresh output directory required')
    if sum(CONFIG[k] for k in ['training_visitation_outputs', 'optimizer_probe_outputs',
        'matched_audit_outputs', 'heldout_evaluation_outputs', 'reset_evaluation_outputs',
        'constraint_outputs']) != CONFIG['exact_model_output_budget']:
        raise AssertionError('budget arithmetic failed')
    with np.load(CONTEXT, allow_pickle=False) as data:
        train_sum = int(np.sum(50 - data['train_indices'][:, 1]))
    with np.load(FINAL_ROOTS, allow_pickle=False) as data:
        final_sum = int(np.sum(50 - data['indices'][:, 1]))
    if train_sum != 1706 or final_sum != 1693:
        raise ValueError('root horizons changed')
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    if sha(SOURCE / 'candidate_pool.npz') != manifest['candidate_pool.npz']:
        raise ValueError('source candidate pool hash mismatch')
    for seed in SEEDS:
        key = f'checks\\s{seed}_k0_u24\\critic.pt'
        if sha(SOURCE / CONFIG['source_critics'][str(seed)]) != manifest[key]:
            raise ValueError('source critic hash mismatch')
    with np.load(SOURCE / 'candidate_pool.npz', allow_pickle=False) as pool:
        initial = {f's{seed}': np.asarray(pool[CONFIG['source_parameters'][str(seed)]], np.float32)
                   for seed in SEEDS}
    for theta in initial.values():
        for mode in MODES:
            projected, _ = project_response(theta[16:], mode)
            np.testing.assert_array_equal(projected, theta[16:])
    out.mkdir(parents=True)
    (out / 'checkpoints').mkdir()
    (out / 'audits').mkdir()
    (out / 'evaluation').mkdir()
    write(out / 'config.json', CONFIG)
    (out / 'PROTOCOL.md').write_text(protocol_text(), encoding='utf-8')
    np.savez_compressed(out / 'initial_parameters.npz', **initial)
    hashes = {path.as_posix(): sha(path) for path in source_inputs()}
    write(out / 'provenance.json', dict(utc=timestamp(), git_head=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), local_lipschitz_report_available=False,
        input_sha256=hashes, frozen_definitions=dict(reward='ett.rollout_return.task_reward',
        discount=.95, return_scale=.05, goal=GOAL, nominal_actor_diagonal='resolved by ett.run_convex_adversarial.setup',
        geometry='rectangle selected by convex_box; existing wall/boundary postprocessing'),
        source_parameters=CONFIG['source_parameters'], source_critics=CONFIG['source_critics'],
        oracle_03440b6_scope='broader hand-designed family; not optimizer-only evidence'))
    write(out / 'preregistration.json', dict(utc=timestamp(),
        protocol_sha256=sha(out / 'PROTOCOL.md'), config_sha256=sha(out / 'config.json'),
        initial_parameters_sha256=sha(out / 'initial_parameters.npz'),
        provenance_sha256=sha(out / 'provenance.json'), exact_budget=CONFIG['exact_model_output_budget']))
    print(f"已封存协议：{CONFIG['exact_model_output_budget']:,} 个模型转移。", flush=True)


def verify(out):
    prereg = json.loads((out / 'preregistration.json').read_text())
    for path, key in [(out / 'PROTOCOL.md', 'protocol_sha256'), (out / 'config.json', 'config_sha256'),
                      (out / 'initial_parameters.npz', 'initial_parameters_sha256'),
                      (out / 'provenance.json', 'provenance_sha256')]:
        if sha(path) != prereg[key]:
            raise ValueError(f'sealed file changed: {path}')
    provenance = json.loads((out / 'provenance.json').read_text())
    for path, digest in provenance['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'frozen input changed: {path}')


def frozen_hash(engine):
    return tree_sha((engine.base.params, engine.nominal.params, engine.actor_info))


def train(out, ledger):
    context = dict(np.load(CONTEXT, allow_pickle=False))
    initial = dict(np.load(out / 'initial_parameters.npz', allow_pickle=False))
    histories, finals, frozen = {}, {}, {}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for seed in SEEDS:
        source_key = CONFIG['source_parameters'][str(seed)]
        saved = torch.load(SOURCE / CONFIG['source_critics'][str(seed)], weights_only=True)
        initial_critic_sha = None
        for mode in MODES:
            name = f'{mode}_s{seed}'
            engine = ModeKernel(mode)
            frozen[name] = frozen_hash(engine)
            theta = initial[f's{seed}'].copy()
            momentum = np.zeros(32, np.float64)
            variance = np.zeros(32, np.float64)
            critic = early.Critic(0)
            critic.load_state_dict(copy.deepcopy(saved['model']))
            optimizer = torch.optim.Adam(critic.parameters(), lr=CONFIG['critic_lr'])
            current_critic_sha = old.parameter_sha(critic)
            if initial_critic_sha is None:
                initial_critic_sha = current_critic_sha
            elif current_critic_sha != initial_critic_sha:
                raise AssertionError('critic initialization mismatch')
            history = []
            continuation = search.Continuation(engine)
            for iteration in range(1, CONFIG['updates'] + 1):
                key = CONFIG['training_seed'] + seed * 1000000 + iteration * 1000
                pre = theta.copy()
                records = early.collect(engine, pre, context['train_roots'],
                    context['train_indices'][:, 1], key, ledger, f'train/{name}/visitation',
                    CONFIG['paths_per_root'])
                steps = CONFIG['first_refresh_steps'] if iteration == 1 else CONFIG['later_refresh_steps']
                fit = update.refresh(critic, optimizer, records, context, key + 100, steps)
                critic_sha = old.parameter_sha(critic)
                queries = update.visitation(records, key + 101, CONFIG['query_count'])
                directions = np.random.default_rng(key + 102).normal(size=(CONFIG['directions'], 32))
                signed = np.empty((CONFIG['directions'], 2), np.float64)
                signed_activation = []
                for direction_index, direction in enumerate(directions):
                    for sign_index, sign in enumerate((1, -1)):
                        response, activation = project_response(pre[16:] + sign * CONFIG['sigma'] * direction, mode)
                        candidate = np.r_[pre[:16], response].astype(np.float32)
                        signed[direction_index, sign_index] = search.first_samples(
                            engine, candidate, queries, critic, context, key + 200,
                            CONFIG['successors'], CONFIG['actor_draws'], ledger,
                            f'train/{name}/signed')['critic'].mean()
                        signed_activation.append(activation)
                theta, momentum, variance, step_info = response_step(
                    pre, directions, signed, momentum, variance, iteration, mode)
                local = []
                for candidate in (pre, theta):
                    local.append(float(search.first_samples(engine, candidate, queries, critic,
                        context, key + 200, CONFIG['successors'], CONFIG['actor_draws'], ledger,
                        f'train/{name}/local')['critic'].mean()))
                np.testing.assert_array_equal(theta[:16], pre[:16])
                checkpoint = out / 'checkpoints' / f'{name}_u{iteration}.npz'
                save_checkpoint(checkpoint, theta, mode, seed, iteration, source_key)
                loaded, metadata = load_checkpoint(checkpoint, mode)
                np.testing.assert_array_equal(loaded, theta)
                row = dict(update=iteration, preupdate=pre, theta=theta, directions=directions,
                    signed=signed, local_before=local[0], local_proposed=local[1], fit=fit,
                    critic_sha256=critic_sha, checkpoint_sha256=sha(checkpoint),
                    checkpoint_metadata=metadata,
                    signed_constraint_active_count=int(sum(v['active_count'] for v in signed_activation)),
                    signed_constraint_total=CONFIG['directions'] * 2 * 8,
                    signed_constraint_max_raw_norm=float(max(np.max(v['raw_block_norms']) for v in signed_activation)),
                    **step_info)
                if iteration in CONFIG['audit_updates']:
                    folder = out / 'audits' / f'{name}_u{iteration}'
                    folder.mkdir()
                    row['matched'] = matched_check(engine, continuation, pre, theta, records,
                        critic, context, CONFIG['matched_mc_seed'] + seed * 1000000 +
                        iteration * 1000, ledger, folder)
                    phase.save_records(folder / 'visitation.npz', records)
                if old.parameter_sha(critic) != critic_sha:
                    raise AssertionError('critic changed during frozen ETT probes')
                history.append(row)
                print(f'{name} 更新 {iteration}/6；critic 局部差 {local[1]-local[0]:+.6f}', flush=True)
            torch.save(dict(model=critic.state_dict(), optimizer=optimizer.state_dict(),
                            constraint_mode=mode, seed=seed),
                       out / 'checkpoints' / f'{name}_critic.pt')
            histories[name] = history
            finals[name] = theta
            if frozen_hash(engine) != frozen[name]:
                raise AssertionError('frozen model/policies changed')
    np.savez_compressed(out / 'final_parameters.npz', **finals)
    write(out / 'training.json', dict(utc=timestamp(), histories=histories,
        frozen_hashes=frozen, response_updates=24, critic_steps=10000,
        final_parameters_sha256=sha(out / 'final_parameters.npz')))
    return initial, finals


def constraint_probe(engine, theta, context, ledger, purpose):
    state = context['validation_state'][:16]
    key = jax.random.PRNGKey(CONFIG['constraint_seed'])
    xp = np.asarray(engine.nominal.sample(state, key, 1, goal=np.broadcast_to(GOAL, state.shape)))
    actions = [np.broadcast_to(np.array([x, y], np.float32), (16, 2))
               for x in (-1., 0., 1.) for y in (-1., 0., 1.)]
    actions += [xp, np.clip(xp + .001, -1, 1)]
    outputs = []
    for action in actions:
        ledger.add(16 * 8, purpose)
        output, detail = engine.sample(theta, state, action, xp, key, 8)
        validate_selected_set(state, output, detail)
        outputs.append(np.asarray(output))
        if action is xp:
            np.testing.assert_array_equal(output[..., :2], np.asarray(detail['anchor_xy']))
    excess = 0.
    for i in range(len(actions)):
        for k in range(i):
            distance = np.linalg.norm(outputs[i][..., :2] - outputs[k][..., :2], axis=-1)
            allowed = np.linalg.norm(actions[i] - actions[k], axis=-1)[:, None]
            excess = max(excess, float(np.max(distance - allowed)))
    matrices = np.asarray(matrices_mode(theta[16:], engine.constraint_mode))
    operator = np.linalg.svd(matrices, compute_uv=False)[..., 0]
    frobenius = np.linalg.norm(matrices, axis=(1, 2))
    passed = excess <= CONFIG['numerical_tolerance'] and operator.max() <= 1 + CONFIG['numerical_tolerance']
    if not passed:
        raise AssertionError('constraint grid failed')
    return dict(passed=True, samplewise_lipschitz_excess=excess,
        component_operator_max=float(operator.max()), component_frobenius_max=float(frobenius.max()),
        exact_diagonal_identity=True)


def full_returns(engine, theta, roots, indices, repeats, seed, ledger, purpose, path):
    records = early.collect(engine, theta, roots, indices[:, 1], seed, ledger, purpose, repeats)
    phase.save_records(path, records)
    values = np.empty((len(roots), repeats), np.float64)
    for record in records:
        horizon = record['reward'].shape[1]
        for root in np.unique(record['root']):
            selected = record['root'] == root
            values[root] = .05 * record['reward'][selected] @ .95 ** np.arange(horizon)
    return values


def evaluate(out, initial, finals, ledger):
    context = dict(np.load(CONTEXT, allow_pickle=False))
    with np.load(FINAL_ROOTS, allow_pickle=False) as data:
        roots, indices = np.asarray(data['roots']), np.asarray(data['indices'])
    values, resets, constraints = {}, {}, {}
    models = []
    for seed in SEEDS:
        models += [(f'init_s{seed}', 'frobenius', initial[f's{seed}']),
                   (f'frobenius_s{seed}', 'frobenius', finals[f'frobenius_s{seed}']),
                   (f'spectral_s{seed}', 'spectral', finals[f'spectral_s{seed}'])]
    for name, mode, theta in models:
        engine = ModeKernel(mode)
        values[name] = full_returns(engine, theta, roots, indices, CONFIG['final_repeats'],
            CONFIG['final_seed'], ledger, f'evaluation/heldout/{name}',
            out / 'evaluation' / f'{name}_heldout.npz')
        ledger.add(CONFIG['reset_repeats'] * 50, f'evaluation/reset/{name}')
        record = engine.rollout(theta, np.tile(START, (CONFIG['reset_repeats'], 1)),
                                jax.random.PRNGKey(CONFIG['reset_seed']), 50)
        np.savez_compressed(out / 'evaluation' / f'{name}_reset.npz', **record)
        resets[name] = .05 * record['reward'] @ .95 ** np.arange(50)
        constraints[name] = constraint_probe(engine, theta, context, ledger,
                                              f'constraint/{name}')
        print(name + '：独立全轨迹评估完成', flush=True)
    np.savez_compressed(out / 'evaluation_returns.npz', **values,
                        **{name + '__reset': value for name, value in resets.items()})
    write(out / 'constraints.json', constraints)


def run(out):
    verify(out)
    if (out / 'started.json').exists():
        raise ValueError('one-shot attempt already started')
    write(out / 'started.json', dict(utc=timestamp(), status='started'))
    ledger = Ledger(out)
    write(out / 'ledger.json', ledger.data)
    initial, finals = train(out, ledger)
    evaluate(out, initial, finals, ledger)
    if ledger.data['charged'] != CONFIG['exact_model_output_budget']:
        raise AssertionError(f"budget mismatch {ledger.data['charged']}")
    write(out / 'completion.json', dict(utc=timestamp(), status='complete',
        charged_model_outputs=ledger.data['charged'], cap=ledger.data['cap'],
        response_updates=24, critic_steps=10000, native_steps=0, actor_updates=0,
        diagonal_updates=0, reward_changed=False, selection_or_sweep=False,
        final_parameters_sha256=sha(out / 'final_parameters.npz'),
        evaluation_returns_sha256=sha(out / 'evaluation_returns.npz')))
    print(f"一次性运行完成：{ledger.data['charged']:,} 个模型转移。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else run)(args.out)


if __name__ == '__main__':
    main()
