# PointMaze ETT：问题与实验记录

核对日期：2026-09-13。分支：`feature/pointmaze-causal-transition`。固定快照：`f85a5f44b176120333a1de0f83d6d0d02a2b290e`。

本记录依据公开仓库代码、已保存报告和结果 JSON；本次没有重跑训练或仿真。未提交到该快照的其他窗口工作不在覆盖范围。报告中的旧“下一步”属于历史建议，不能自动当成当前待办。

## 当前最大问题

**还没有建立从“ETT 的悲观训练目标降低”，到“完整模型轨迹确实变差”，再到“actor 在真实环境表现更好”的可靠证据链。**

第二个 loss 已经实现；校准后的 contrastive 价值驱动双 loss，在可求解有限状态环境里达到了指定模型族的认证最优值，也在 PointMaze 中产生过独立评估支持的模型回报下降。现在的困难是：PointMaze 中悲观效果往往小、依赖起点和训练种子；局部价值差存在估计误差；模型保留了动作/路线/吸收后果方面的局限；真实环境增益仍没有在主要对照中稳定建立。

**当前没有证据把它归结成唯一原因。** 不能分别把“critic 坏了”“优化器不够强”“Frobenius 太严格”“F4 不够”“actor 没更新”写成已确定的总解释。

## 先修正五个已经过时的判断

1. “只做好了 nominal policy，第二个 loss 没实现”——过时。见 E06、E08、E12–E19。
2. “ETT 的 diagonal 一直冻结，因此从未训练两个有效 loss”——只适用于若干受限实验。有限状态实验、48 参数 PointMaze shared-head 实验确实更新了两项；最新 E26 为隔离约束因素又冻结了 diagonal。
3. “没有 contrastive critic 与 ETT 交替更新”——过时。E15、E19 等已经做过；这不等于完整 actor–critic–ETT 三方交替训练已经被验证。
4. “旧 raw critic 局部排序失败，所以后来所有 critic 都不能用”——不成立。旧 full-F4 point-goal raw logit 和新 region-NCE 解码价值是不同评价器。
5. “oracle 能压低回报，所以只剩优化器问题”——E25 的推断过强；E26 明确指出 oracle 使用了更宽的非线性构造，不能据此排除表示限制。

## 实验索引：问题、做法、结果、边界

### E01｜专家动作与 diagonal 是否已经可用？

- 做法：teacher-source 数据训练 K5 censored Gaussian nominal density；atom + K3 moving Gaussian 拟合 diagonal；状态为 newest-first 四帧 XY，goal 另作条件。
- 结果：已有可采样 nominal 和可用的 observational diagonal 基线，后续实验持续复用。teacher 数据包含失败，并非 success-only；原生 diagonal API 拒绝 off-diagonal 查询。
- 边界：拟合 observational 数据不识别真实 off-diagonal ETT；安全绕路的数据覆盖较薄。
- 来源：[nominal 专家数据说明][nominal]；[diagonal 基线][diagonal]。

### E02｜F4、死亡分类器或 failure-bank 距离能否充当“坏后果”评价？

- 做法：比较 XY、运动历史和 F4；检查死亡发生时刻与随后几帧；训练监督失败分类器；用固定 failure-bank 距离检查真实记录的 next-state 排序。
- 结果：F4 对已经持续若干步的死亡有强信息，刚发生死亡时弱；分类器整体表现好不代表 onset 可靠；bank 距离没有建立匹配条件下的即时 onset 排序能力。
- 边界：分类监督及原始 bank 的构造含 privileged provenance。静止、没有成功、距离 bank 近，都不能直接等于死亡。没有证明必须加 death bit，也没有证明 F4 足以组成真实的长时域因果状态。
- 来源：[可观测性][observability]；[监督评分][failure-score]；[记录后果的 bank 排序][bank-ranking]。

### E03｜把 next state 拉向 failure bank，第二项是否就成立？

- 做法：条件生成样本与 failure bank 的 MMD；增加几何/anchor 检查；分解 MMD；比较最近集合距离和受控候选。
- 结果：guarded 主实验的 MMD 从约 0.781 降到 0.719，但静止历史的生成静止率从 93.15% 降到 7.45%。MMD 改善主要来自生成样本彼此更分散，和 bank 的交叉相似度反而下降。最近集合距离避免部分分散伪象，却仍可奖励位移和样本坍缩。
- 边界：证明的是这些无条件一步相似度目标有漏洞，不是所有悲观目标都无效。后续不应再把换个 bank 距离当成未经测试的方向。
- 来源：[guarded MMD][mmd]；[目标分解与集合距离][objective]。

### E04｜旧 contrastive raw logit 能否直接评价从 next state 继续行动？

- 做法：候选放在 STATE 位置，固定真实任务 goal，取冻结 actor 下的 raw critic 平均分；比较六个历史 checkpoint。
- 结果：主 checkpoint 全局死亡 ROC-AUC 0.98936，但 F4 距离不超过 0.5 的 284 对中，dead-below-alive 只有 29.93%；六个 checkpoint 均在该紧匹配子集反向。
- 边界：拒绝的是这些旧 raw-score 直接代入第二项的做法；不能外推到后来正确解码、重新训练的 region-NCE。
- 来源：[raw continuation 诊断][raw-critic]。

### E05｜冻结 contrastive 表征后，加回报读出头是否能解决？

- 做法：线性/神经读出预测日志中 H=20 回报，再用独立固定 actor continuation 检查迁移，与原始输入小 MLP 比较。
- 结果：能读出部分回报信息，整体误差有改善；紧匹配后果排序和 fixed-actor 迁移没有形成稳定优势，原始输入 MLP 也有竞争力。
- 边界：日志后续行为的回报，不等于固定 actor 的干预 continuation；表征含信息，不等于旧 scalar score 正确。
- 来源：[logged-return readout][readout]；[固定 actor continuation][fixed-continuation]。

### E06｜不靠 critic，直接最小化模型完整轨迹回报是否可行？

- 做法：冻结 diagonal/actor/nominal，只搜索 6 个 residual 参数；H=50、gamma=.95、原始 task reward；再移除两个输出 bias，只搜 4 个权重。
- 结果：6 参数搜索使独立模型回报约 10.54 → 9.33/8.61；主要表现为共同方向漂移、延迟拿奖励或边界停滞。去掉 bias 后仍有高度共同方向的残差。
- 边界：此版 diagonal loss 是常数；原始 anchor-distance bound 也不是 full all-action-pairs Lipschitz。更低回报没有验证真实动作后果或 global worst case。
- 来源：[完整 return 搜索][return-search]；[bias 消融][bias]；[修正 horizon 后的动作响应诊断][action-response]。

### E07｜共享网络的两个 loss 是否本身冲突？Lipschitz 如何约束？

- 做法：标量已知目标的共享 MLP；有界斜率 spline；移除正确 off-diagonal 标签、改用标量 bank；降低 lambda 并设 diagonal 误差预算。
- 结果：可以学习相对动作响应；普通 MLP 拟合好不自动满足 Lipschitz；有界 spline 可保证 full action bound。固定 bank 容易把 diagonal 一起推低；lambda=.004 大幅改善拟合，但三种子均未满足全域最大误差 .01 的要求。
- 边界：这是给定标量语义的合成实验，不是 PointMaze 反事实证据；RMSE 小于预算不等于最大误差满足预算。
- 来源：[共享 MLP][synthetic-shared]；[Lipschitz spline][synthetic-lipschitz]；[标量 bank][synthetic-bank]；[误差预算][synthetic-budget]。

### E08｜保证 full action-Lipschitz 的 PointMaze ETT 能否压低模型回报？

- 做法：冻结 diagonal anchor，8 个 context gates 混合 2×2 响应矩阵，共 32 个参数；convex projection；固定 actor 完整 rollout 回报最小化。
- 结果：L=1、diagonal 恒等保持；模型回报约 10.52 → 10.15/10.07，两种子独立区间支持下降。
- 边界：仍是受限 frozen-diagonal family；不能以该结果声称共享 diagonal 已训练或真实 worst-case Q 已恢复。
- 来源：[convex adversarial][convex]。

### E09｜把 frozen ETT 轨迹接回原 CRL 是否有收益？是否只是 future window 太短？

- 做法：offline-only / 10% control-model / 10% pessimistic-model；每臂两种子 2,000 更新。后续共享同一批完整轨迹，比较 3 步内 future goals 和更远 future goals。
- 结果：网络和动作改变，但各实验内 native 配对指标无改善；延长 future window 也没有建立收益。
- 边界：两次实验的 reset 样本不同，32.8% 与 42.2% 不能跨报告当作提升。synthetic BC 与 negatives 也随采样改变；这些实验不单独识别 critic positives 的因果效应。
- 来源：[早期 CRL 接入][old-policy]；[future-window 对照][future-window]。

### E10｜模型是否根本不能表达绕路或吸收后果？

- 做法：固定 shortcut/lower-detour 控制器；fork 几何审计；添加 convex fork-segment；同预算 rectangle vs segment 悲观搜索；冻结模型的死亡可表达性审计。
- 结果：native lower-detour 回报 11.07，高于 shortcut 4.22；两个模型都未完成绕路 waypoint，seed0 反而偏好 shortcut。原 rectangle 排除了一些有效 fork 转移；改 segment 后未建立稳定回报优势。指定 frozen family 在检查的 death contexts 无法实现概率 1 的可见吸收。
- 边界：这些结论限于指定 family/context。E11 已纠正“任何下一步实验都必须先实现 persistent death”的过强要求。
- 来源：[route 诊断][route]；[结构审计][structure]；[几何对照][geometry]；[死亡可表达性][death-admissibility]。

### E11｜原共享一步设计到底有没有跑？

- 做法：`031d430` 提出 2,123 参数共享生成器，以 diagonal energy score + .01 即时奖励训练；局部 reward boundary、x 均匀采样、x' 取记录动作。
- 状态：该目录只有准备和 45 个确定性组件输出，训练未运行。
- 边界：后续 E12–E19 有其他共享双 loss 训练，不能因这份 protocol 未跑就说整个项目仍未做双 loss。
- 来源：[one-step shared design][shared-design]。

### E12｜从 contrastive reward/NCE 推出的第二项，在已知最优解时对不对？

- 做法：四状态、H=4、gamma=.9 的可求解 ETT；已知 negatives 与类别权重，解码 Q；折扣 visitation、冻结 critic 的一步 surrogate；交替更新共享参数，约束 diagonal TV 与 action TV。
- 结果：三种子达到该四参数可行族的认证最小回报 0.1040294475。最优值只用于事后验证。
- 边界：认证限于这个有限状态模型族和固定 actor；不是连续 PointMaze 的证明。
- 来源：[tabular finite-state][finite-tabular]。

### E13｜换成神经 contrastive critic 后，E12 是否仍成立？

- 做法：944 参数的共享 critic，采样 NCE，正确解码 Q，仍用原 ETT 与 48 次更新；追加独立同模型 refit。
- 结果：三种子 neural joint 也达到同一认证最优值。说明这一机制并不依赖 exact-Q 直接监督或只能 tabular 工作。
- 边界：小状态空间、覆盖和 ETT 梯度条件仍远优于 PointMaze。
- 来源：[neural finite-state][finite-neural]。

### E14｜相同即时 reward 下，能否区分可恢复 setback 和 death？

- 做法：五状态扩展，progress/setback/death 的即时奖励均为零，但长时价值不同。
- 结果：剩余 horizon=3 时 V(progress)=.271、V(setback)=.171、V(death)=0；三种子达到认证最优值 .1282545。最大死亡率本身不等于最小任务回报，因为仍需考虑存活路径的进展时间。
- 边界：短到来不及恢复的 horizon 下两者可以同值；不能凭低回报唯一认定死亡。
- 来源：[setback 对照][finite-setback]。

### E15｜region-NCE 能否在 PointMaze 驱动共享双 loss？

- 做法：16 个 diagonal-head offsets + 32 个 response 参数；H=10 的 t=40 roots；reward-region membership NCE；校准后交替更新，actor/nominal 固定。
- 结果：两个 joint 种子的模型标准化 occupancy 比 control 低约 .00597/.00537；diagonal ES 退化在 .02 允许范围。两项确实参与更新。
- 关键发现：outside-goal roots 事后审计为 32/32 已死亡；模型却从这些状态重新运动，未覆盖“还活着、尚未到目标”的决策。
- 边界：校准及优化流程跑通，不代表正确找到 native death 或有用决策。
- 来源：[late-root pilot][late-pilot]。

### E16｜改为早期、仍可改变后果的 roots，问题是否消失？

- 做法：approach/transit/bypass 的早期真实 F4 roots；只按可见前缀选择；事后 native audit 72/72 alive；用固定 response probes 检查有限容量。
- 结果：覆盖问题改善，有 393 条模型 recovery-after-setback 路径；但初始 region critic 的 root RMSE .07950、bias +.04974，未过门槛。此次没有 ETT 更新。
- 边界：不能把这次称作 ETT 优化失败；当时停在 value calibration。
- 来源：[early contexts][early]。

### E17｜是不是早期样本太少？

- 做法：同轨迹/标签、同网络/训练预算，比 uniform 与 phase-balanced，把早期曝光约 10.5% 提到 33.3%；新采 600 native episodes 仅用于提供未见 context。
- 结果：训练分布早期拟合两种子均改善；fresh early RMSE seed1 明显改善，seed0 不稳定且部分组变差。没有一致的 calibration 或困难场景排序修复。
- 边界：支持采样有影响，不支持它是唯一原因；训练 ETT/actor 未发生。
- 来源：[phase sampling][phase]。

### E18｜是不是 future-time 抽样的噪声？

- 做法：同轨迹，将抽一个 future 的 NCE positive 改为对未来时间作折扣平均，保留其他设置；用固定 response probes 对照 critic 与配对 MC。
- 结果：部分 calibration 指标改善；没有建立“预测候选转移价值差更好”的一致优势，四个 probe 的 MC 方向都未解析。
- 边界：平均消除了条件 future-time 标签方差，不消除有限轨迹噪声、覆盖/拟合误差或模型偏差。该 averaged critic 后续已经用于 E19。
- 来源：[future averaging][future-average]。

### E19｜换成直接 MC continuation，ETT 更新是否更好？

- 做法：同 48 参数初始模型、每种子三次更新；diagonal-only / averaged-NCE continuation / 直接 MC continuation；候选仅第一步改变，续程固定 pre-update model；独立全轨迹评估。
- 结果：两种子 critic 与 MC 都降低完整模型回报；critic 相对 diagonal 的变化 -.02050/-.01316，MC 为 -.01686/-.01158。MC 没有优于 critic；共享步长上限和 diagonal 拒绝约束都曾生效。
- 边界：这次没有证据把 critic 当成阻止优化的瓶颈；不能据此认定任何大候选移动都估计准确。回报下降伴随允许的 diagonal 拟合退化，其贡献未隔离。
- 来源：[critic vs MC 更新][update-reference]。

### E20｜用新悲观模型直接更新 actor，native 是否变好？

- 做法：冻结 actor baseline；observational model 与 pessimistic ETT 两种训练环境；decoded-Q 受约束 actor 更新；256 个配对 native reset。
- 结果：主要回报、成功率、死亡率差的区间都含零；37/400 个 actor proposal 获接受，动作改动较小。模型仍把原 actor 评得远高于 native：约 7.80–10.31 vs 3.33。
- 边界：不能证明等价或完全没效果，也不能将瓶颈唯一归因于 actor 更新幅度。
- 来源：[decoded-Q native policy][native-policy]。

### E21｜回到原 full-F4 CRL+BC，并让 actor 明显更新，是否解决？

- 做法：unchanged I / offline O / observational B / pessimistic C；两种子、每臂 1,000 次原 MC-NCE+BC 更新；真实环境配对评估。
- 结果：所有 actor 实际改变。seed0 的 C 比 B 回报高 .4103，区间 [.0050,.8479]；seed1 未复现；两个种子都未建立 C 优于 O。两种子描述性 C−O=.0343，区间 [−.2145,.2680]。
- 边界：不能再说“所有新实验里动作和 outcome 都完全相同”，也不能说已经优于两个主要对照。
- 来源：[full-F4 integration][full-f4]。

### E22｜冻结 diagonal、扩大 response 搜索，是否只是搜得不够？

- 做法：原 response 加两个预设非零起点，两个 baseline 各三条搜索，每条 24 更新；仅 32 response 参数；选择与最终评估分开。
- 结果：一个 baseline 有独立小幅 gain，另一个没稳定 gain。原 response 出发的 baseline1 终点变化 −.01121；某 baseline0 终点反而 +.00691。18 次 critic/MC 局部均值有 9 次异号，但当时没有一次双区间确认相反方向。
- 边界：不能只凭异号点估计认定 critic 系统性反向。更大搜索有作用，但不稳定；没有隔离主导瓶颈。
- 来源：[response-only search][response]。

### E23｜是不是 MC 精度不足，掩盖 critic 的真实误差？

- 做法：对 E22 两个固定候选比较做分层、配对高精度 continuation audit；增加 pre-entry 覆盖；7,370,240 次计算转移；不训练。
- 结果：s0_k2 的 critic 差 −.02242，MC +.01710；MC−critic=.03953，97.5% 区间 [.01468,.06437]，超过 .01 容差。证明该比较存在实质价值差误差；MC 自己的区间含零，不能声称真实方向已显著反转。另一个比较未解析。
- 边界：这已经是提高精度后的结果；预设 .005 半宽仍没达到。局部 first-transition surrogate 与 full-model rollout 是不同 estimand。
- 来源：[continuation precision][precision]。

### E24｜critic 是不是普遍把好坏顺序搞反？

- 做法：只复用 E23 保存数组，检查 pre-entry 逐 query 符号、Spearman、Kendall；无新模型调用。
- 结果：两组符号一致率 56.0%/57.5%，rank association 为正；changed-sequence 子集 68.5%/72.2%。不支持这两个比较里系统性排序反转，均值相反可由差值幅度错误造成。
- 边界：不表示 critic 足够准确；也不能用小样本未定排序给整个 critic 定性。
- 来源：[contrast agreement][agreement]。

### E25｜允许规则里到底有没有大幅悲观的空间？

- 做法：现有 learned response 对比手工 Lipschitz oracle、box oracle、postprocess oracle；短 H=10 roots 与 START H=50 分开。
- 结果：START 下 learned 改变约 −2.44%/−.50%，Lipschitz oracle 约 −98.35%。说明更宽的指定构造下存在强悲观轨迹。
- 必须修正：该报告“更支持 optimizer 而非 response family”的结论过强。A2 不等于原 32 参数 gated-linear Frobenius family；A3/A4 还违反原 Lipschitz 要求。E26 明确记录此限制。
- 边界：oracle 是手工可达构造，不是所有可行模型的认证上下界，也不是 native 合法后果。
- 来源：[pessimism ceiling][ceiling]；[最新解释修正][norm-code]。

### E26｜Frobenius 约束是否比真实所需谱范数更严，从而卡住优化？

- 做法：同两个已训练 response 起点，Frobenius/spectral 每臂六次 response 更新；同 actor、nominal、diagonal、几何；两个种子、独立全轨迹和局部 MC 审计。
- 结果：没有一致的 spectral 优势，四臂均未达到预设 .01 独立效应。seed0 的两臂确有区间支持的小下降 −.00796/−.00626，不能写成“完全不下降”。最终 proposals 的约束激活率均 0；扰动查询中只有约 .65%–2.21% block 激活。
- 边界：未建立该约束是当前主因；也没排除表示、projection 或优化瓶颈。此结果不能倒推手工 oracle 属于原可训练族。
- 状态：最新提交有 `results.json`、`completion.json`、protocol 和报告生成代码，未提交该目录的 REPORT.md；本记录直接核对 JSON，未重新生成报告或运行实验。
- 来源：[norm results][norm-results]；[norm protocol][norm-protocol]；[报告解释代码][norm-code]。

## 防止重复实验的当前规则

以下问题已经有对应实验，不能再当成全新假设：

- “试试 failure bank/MMD/最近距离” → E02–E03。
- “把 next state 放 critic state slot 再检查” → E04；换用新 region-NCE 必须明确不是旧 raw score。
- “多训一个 return readout” → E05。
- “先证明双 loss/神经 critic/setback 能工作” → E07、E12–E14 已提供特定范围内的正证据。
- “早期 roots 太少、future window 太短、future sampling 噪声大” → E09、E16–E18。
- “把 critic 换成 MC 看看” → E19；“再提高 MC 精度” → E23。
- “actor 根本没更新、还没接回原 CRL” → E20–E21。
- “多起点、更大 response 搜索” → E22。
- “critic 一定把顺序反了” → E24 已作细分。
- “做 oracle 看空间，再改谱范数” → E25–E26。

重复并非永远没有价值；再次执行前，应写清：对应哪个 E 编号、与原实验相比改变的唯一假设/对象是什么、现有结果为何不能回答、怎样的结果会改变当前决策。如果这些不变，应先复用已有产物。

## 三种不能互相替代的证据

1. **critic surrogate 降低**：训练评价函数认为某次变化更差。
2. **独立模型 rollout 回报降低**：指定模型族内产生了更悲观轨迹；局部 first-step surrogate 和候选全程生效仍是不同目标。
3. **真实环境 actor 收益**：与 offline continuation / observational augmentation 配对比较，是否提高回报或降低失败。

还需区分 raw return 与 normalized occupancy：PointMaze 后者为 `.05 * sum(.95^t*r_t)`，两者差 20 倍；不同 root、horizon、actor 和确定性/随机执行不能横比。认证最优值只在 E12–E14 的有限族成立。

## 给其他窗口的当前起点

先读这份记录和最新 E19–E26 的来源。当前开放问题是：**在明确的 PointMaze 可训练 ETT 族及约束内，怎样获得稳定、独立评估支持、能迁移到 reset 的悲观效果，并让这种效果带来超出普通 offline/observational 训练的 native 收益。**

可以分解为 continuation 差值估计、可训练响应族/几何/搜索、模型后果语义、策略接入四部分。已有局部证据分别涉及这些问题；截至固定快照尚未锁定唯一主因。不应从某一篇旧报告的“下一步”自动启动其后已经做过的实验。

早期 AntMaze center/horizon 报告属于另一环境与时域，不作为当前 PointMaze H=50 的证据。本记录没有据其他窗口的描述推断其未提交代码状态。

[nominal]: https://github.com/tingrui-huang/contrastive_rl/blob/031d430500497c864709b46ad3cbd17c03c71fc0/artifacts/nominal_policy/EXPERT_ONLY_REPORT.md
[diagonal]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_diagonal/REPORT.md
[observability]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/death_observability/f4_p30_expert_s01/REPORT.md
[failure-score]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/failure_scoring/scope_b_f4_p30_s01/REPORT.md
[bank-ranking]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/bank_ranking/f4_p30_recorded_v1/REPORT.md
[mmd]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_distribution_matching/f4_p30_s01_guarded/REPORT.md
[objective]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_objective_diagnostic/f4_p30_s01/REPORT.md
[raw-critic]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/critic_continuation/f4_p30_alpha_s01/REPORT.md
[readout]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/return_readout/f4_h20_g095_v1/REPORT.md
[fixed-continuation]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/fixed_actor_continuation/f4_h20_g095_v1/SUMMARY.md
[return-search]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_rollout_return/residual6_s01/REPORT.md
[bias]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_bias_ablation/four_weights_s01/REPORT.md
[action-response]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_action_response/f4_return_s01_horizon50/REPORT.md
[synthetic-shared]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/synthetic_shared_response/l025_1_4_s012/REPORT.md
[synthetic-lipschitz]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/synthetic_lipschitz_response/l1_s012/REPORT.md
[synthetic-bank]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/synthetic_failure_bank/bank_m3_lambda01_s012/REPORT.md
[synthetic-budget]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/synthetic_diagonal_budget/lambda0004_s012/REPORT.md
[convex]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_convex_adversarial/l1_matrix32_s01_v1/REPORT.md
[old-policy]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_policy_improvement/h3_mix10_u2000_s01_v1/REPORT.md
[future-window]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_future_window/shared_h3_full_u2000_s01_v1/REPORT.md
[route]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_route_diagnostic/fixed_visible_s01_n128_v1/REPORT.md
[structure]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_structure_audit/fork_saved_s01_v1/REPORT.md
[geometry]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_geometry_comparison/zero_s01_u16_v1/REPORT.md
[death-admissibility]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_death_admissibility/frozen_f4_gate_v1/REPORT.md
[shared-design]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/ett_shared_design/one_step_reward_v1/REPORT.md
[finite-tabular]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/finite_crl/tabular_h4_s012_v1/REPORT.md
[finite-neural]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/finite_crl/neural_h4_s012_v1/REPORT.md
[finite-setback]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/finite_crl/setback_h4_s012_v1/REPORT.md
[late-pilot]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/fixed_goal_h10_s01_v1/SUMMARY.md
[early]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/early_contexts_s01_v1/SUMMARY.md
[phase]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/phase_sampling_s01_v1/SUMMARY.md
[future-average]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/future_average_s01_v1/REPORT.md
[update-reference]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/update_reference_s01_v1/SUMMARY.md
[native-policy]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/native_policy_s01_v1/SUMMARY.md
[full-f4]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/full_f4_integration_s01_v1/REPORT.md
[response]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/response_search_s01_v1/SUMMARY.md
[precision]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/continuation_precision_s01_v1/SUMMARY.md
[agreement]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/contrast_agreement_v1/SUMMARY.md
[ceiling]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/artifacts/pointmaze_region_pilot/pessimism_ceiling_v1/REPORT.md
[norm-results]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/outputs/ett_frobenius_spectral_v1/results.json
[norm-protocol]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/outputs/ett_frobenius_spectral_v1/PROTOCOL.md
[norm-code]: https://github.com/tingrui-huang/contrastive_rl/blob/f85a5f44b176120333a1de0f83d6d0d02a2b290e/ett/report_pointmaze_norm_comparison.py
