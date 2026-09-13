# Frobenius 与谱范数 ETT 配对比较：预注册协议

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
