# 连续 Manski 移植:完整总结(2026-07-16,R7 更新 07-17)

## R7 主结果(windy 环境 = 用户原始设计;原始数据配方;AWR 提取;final;3 seeds)

| arm | worst-case (all_active) | natural | 安全路占比 |
|---|---|---|---|
| baseline(dπ+AWR) | 0.00 / 0.00 / 0.00 | 0.70 / 0.70 / 0.70 | 0% |
| **causal(d̲+AWR)** | **1.00 / 0.98 / 0.30** | **1.00 / 1.00 / 0.85** | 100 / 99 / 40% |

均值 worst-case:0.76±0.33 vs 0.00(离散 WindyCorridor 为 0.44→0.56)。
baseline natural 0.70 = 冲刺存活率(~0.9³),3/3 seed 判定 CONFOUNDED。
critic fork 边际:**所有 causal seed 均 +2.6**(含部分失败的 s2)——
critic 层 9/9 全对,残余方差 100% 在 AWR actor(s2 落入 60/40 混合模式)。

### 提取层的完整故事(第 7-8 条发现)
1. (Q-max+BC) 拔河:6/6 seed BC 压制 critic → 全部冲捷径;
2. critic-greedy:路线排序对但局部动作分辨率不足,OOD 过估计 → 乱走
   (离散 SUMMARY item 6 的 "greedy-critic just spins" 连续复现);
3. **AWR(β=0.5,只克隆数据动作)= 离散版早已给出的答案**,2/3 clean
   + 1/3 部分。三次移植保真教训:N(s,x) 可达集、环境重抽语义、AWR actor
   ——每次都是"没有对照离散实现核对语义"。

状态:**R5 完成——方法在原始 benchmark 上成立,worst-case 0.00→1.00。**
reachable-N seeds 1/2 复现中。

## R5 主结果(原始数据集,环境和数据零改动,final 检查点)

用户指出"后撤莫名其妙"后发现的决定性 bug:我的 N(s,x) 是动作无关的
相邻格 ball;离散 worst_case_kernel.py 的真语义是**该动作在各 u 取值下
的可达集**(枚举 5 种风况取 argmin)。修正(`manski_reachable`)后:
只有涉及 swamp 格的动作存在悲观分支(worst = 卡死吸收),其余动作可达
集是单点、零悲观、无后撤——u 够不着的地方下界自动收紧(d̲≈dπ)。

| 方法 | all_clear | all_active (worst) | natural | gap | 判定 |
|---|---|---|---|---|---|
| baseline final | 1.00 | **0.00** | 0.73 | +1.00 | CONFOUNDED_SHORTCUT_BIAS |
| **causal(reachable-N)final** | **1.00** | **1.00** | **1.00** | **0.00** | NO_CLEAR_BIAS |
| always-safe oracle | 1.00 | 1.00 | 1.00 | 0.00 | — |

causal 100% 走安全路,**与 oracle 完全重合**;连 forced fork 起点都
100% 绕行(holding 起点 0.73——已越过岔路,照走,注脚同前)。
MC 预检:原始数据集 reachable-N 下 d̲(安全)=0.151 vs d̲(捷径)=0.059
(2.6×)。**三次数据集重采全是在补偿 ball-N 的冤枉税,均不必要。**

### ⚠ 多 seed 更新(诚实降级):上表是 1/3 seeds

reachable-N s1/s2 的 final 塌回捷径;三个 seed 的 natural 曲线都在
"绕路模式(≈1.0)"和"捷径模式(≈0.7)"间震荡,s0 的满分 final 是
停在了好时刻。**关键切分诊断:critic 无辜**——所有 seed、所有阶段,
critic 在 fork 处稳定排序 f(safe) > f(short)(+1.2~+2.4 logits,
从 40k 起从未翻转,包括失败的 s1)。震荡全部来自 actor 提取:
actor loss 的 bc_coef=0.5 那一半在克隆行为混合(fork 处 ~65% 捷径
方向),与 Q 项持续拔河。对策(进行中):bc_coef 0.5→0.2,两 arm
同步,3 seeds × 2 arms 重训。
方法层结论不变(d̲ 信号正确且被 critic 稳定学到),悬而未决的是
连续 Gaussian actor 的策略提取稳定性——这是第 7 个发现:离散 argmax
不存在的"BC-critic 拔河"问题在连续 actor 上是真实的失败模式。

## R4(ball-N + hazard + safe40)降级为 ablation:seed 脆弱

s0:worst 0.91 / natural 0.98;**s1、s2 final 塌回冲捷径(worst 0.00)**
——1.79× 的翻转幅度在训练噪声下不稳定。忠实的 reachable-N 是成败线,
不是修饰。(s2 训练中段 natural 曾到 1.0,final 滑回——last-iterate
不稳定现象,值得记录。)

## 0. 一段话摘要

把 WindyCorridor(离散)上验证过的 causal contrastive RL(per-step Manski
下界 occupancy d̲ + reweighted NCE)移植到连续 2D swamp PointMaze。训练管线
唯一被改的组件是 NCE 正样本的来源(Thm 2 采样器);其余零改动。第一轮训练
失败(causal 与 baseline 学出相同的冲捷径策略),两个互相独立的根因被逐一
诊断并修复:(1) 数据集安全路缺乏连贯覆盖 + random 污染;(2) worst-case
设计比离散版软弱(退一格可恢复 vs lava 即死)。两者都修复后,训练前的
MC 预检显示 d̲ 以 1.79 倍偏好安全路而 dπ 仍偏好捷径——两 arm 注定分化,
当前正在重训验证。

## 1. 背景与目标

- 已有结果(526 slides):MiniGrid WindyCorridor 上,causal contrastive
  把 forced-U worst-case success 从 0.44 提到 0.56(oracle 0.92)。
- 目标:同样的方法在连续 state/action 上成立。最小连续测试台 =
  自建的 TwoRouteSwampMatchedEnv(连续 [x,y] + 连续 2 维动作,
  隐藏 swamp bits 为混淆 u:u→a(teacher 读 bits 决策)、
  u→s'(活跃格减速 ×0.02)、u∉obs)。
- 叙事定位:PointMaze 本来就是连续 state(此前误记为离散);
  AntMaze/pixel 是后续 scaling,不是方法验证的必要条件。

## 2. 理论对象的连续化方案

| 离散对象 | 连续化 | 备注 |
|---|---|---|
| P(x\|s) 表格 propensity | P̂(bin\|cell):8 方向扇区(旋转半宽)+停留 bin,数据集计数+Laplace | 连续密度在 Manski 分解里退化(P(X=x\|s)=0 ⇒ bound 空洞),邻域化是必需不是便利 |
| N(s,x) 相邻格 | 自身+4 邻可通行格中心 | 覆盖一切 u 配置下的一步可达集(合法性唯一要求) |
| argmin V | BFS 距 goal 的排序 + **hazard**:swamp 三格(静态几何)V̲=0 吸收 | argmin 只需排序;hazard 对应离散 lava −1e9,V̲=0 是合法下界 |
| Thm 2 采样器 | 沿存储轨迹走(经验转移)⊕ 悲观传送+re-anchor,T~Geom(1−γ) | 落进 hazard 即终止;p_override=1 ⇒ 精确退化为无悲观 walk(baseline arm) |
| Lipschitz 假设 | Lemma 2′ 存档(notes/continuous_manski_lemma2prime.md) | 两个常数(动作侧 L_a、状态侧 L_s)以显式松弛项进入 bound;不阻塞实验 |

## 3. 实现清单

- `crl/manski.py`:分箱/propensity/BFS/worst-neighbor(含 hazard)/
  ManskiSampler.walk_from(向量化)/ManskiPositiveBuffer(frozen buffer
  委托包装,只覆写 sample())/build_positive_buffer。
- `scripts/fit_propensity.py`:门 G1(胜过 uniform)、G1b(分箱无伪影)、
  G2(决策格熵高于下游走廊)、G3(BFS 图合法)、G4(传送朝后)。
- `scripts/manski_sampler_probe.py`:门 G5(p̂≡1 零传送+对齐 replay 律)、
  G6(传送集中在 holding)、G7(终点确实更悲观)、G8(锚-终 BFS 相关
  >0.15,非空洞)。
- `scripts/manski_route_diagnosis.py`:**训练前 MC 预检**(几分钟)——
  直接问采样器 fork/holding 处各动作的 P(goal),d̲ 排序不翻转就不开训。
- `crl/config.py` +4 旋钮;`crl/train.py` +13 行(audit 通过后包 buffer)。
- 训练管线其余部分(NCE loss、网络、actor、bc、负样本、offline audit)
  零改动。

## 4. 实验时间线与数字

### R1:原始数据集(safe 5%,random 20%),γ=0.95,matched 环境
- 训练结果:causal 与 baseline 行为相同(100% 冲捷径,worst-case 0.00,
  natural ≈0.71/0.73),VERDICT: CONFOUNDED_SHORTCUT_BIAS(两个都是)。
- MC 诊断:d̲(fork,捷径)=0.027 > d̲(fork,安全)=0.0092(2.9 倍);
  连 dπ 都偏捷径(0.59 vs 0.20)。critic 忠实学了 d̲——错在信号本身。

### 诊断:为什么离散 5% 覆盖能赢、连续不能(对照 causal-contrastive-rl 代码)
1. 离散 FAR expert 确定性 + **零 random 集** ⇒ 安全走廊 P(a|s)=1/步,
   5% 只付一次门票;我们 20% random 污染每步 propensity(~0.2/步)。
2. 风沿 NEAR 每步咬(6 个 lethal x 上 expert 行为依 wind 分裂);
   swamp 的 u 只在 holding 一格可见(teacher 只在 clear 时进,
   走廊数据被选择偏差"消毒")。
3. 离散 V_lower:lava = −1e9 吸收死;我版 worst-case = 退一格可恢复,
   re-anchor 还骑上成功穿越者的轨迹 ⇒ 悲观在混淆走廊内无牙。
4. 连续固有:分箱+噪声泄漏 ~0.85/步,长路径(安全路)吃亏。

### R2:数据端修复扫描(MC 预检,不训练)
| 数据集 | safe% | random% | d̲ 捷径 | d̲ 安全 | 翻转? |
|---|---|---|---|---|---|
| 原始 | 5 | 20 | 0.0270 | 0.0092 | ✗ (0.34x) |
| safe30 | 30 | 10 | 0.0302 | 0.0245 | ✗ (0.81x) |
| strong 旧 teacher | ~19 | 20 | 0.0025 | 0.0008 | ✗ |
| safe40(noise 0.1) | 40 | 5 | 0.1855 | 0.1727 | ✗ (0.93x) |
| strong-wait | 30 | 5 | 0.1435 | 0.1525 | ✓ (1.06x, 太窄) |

strong-wait 训练(150k)未产生绕路策略(natural ~0.5 震荡)——
1.06x 的翻转在 NCE 噪声+actor 提取下不存活。

### R3:worst-case 修复(hazard V̲;用户质疑触发)
| 数据集 | 无 hazard | 有 hazard | 结论 |
|---|---|---|---|
| 原始(5%) | 0.34x | 0.44x | 覆盖问题独立存在,hazard 救不了 |
| strong-wait | 1.06x | 1.69x | 翻转变结实 |
| **safe40 + matched 环境** | 0.93x | **1.79x**(0.137 vs 0.077) | **最优;无需换凶环境** |

同时 dπ 各处仍偏捷径(safe40: 0.63 vs 0.53)⇒ baseline 病理保留。

### baseline 病理的意外收获(strong-wait baseline 训练曲线)
- best@10k:绕路,natural 0.88;final@150k:100% 捷径,natural 0.52,
  worst 0.00。**有偏 dπ critic 随训练把策略从 BC 行为拽向捷径,
  natural 单调劣化**——"混淆主动伤害"的直接演示图。
- 协议教训:**对比用 final,不用 best**(best 按 natural rollout 选点
  = 给 baseline 泄漏了去混淆的模型选择)。

### R4(进行中):最终 setting
- 环境:matched(p=0.1,原 benchmark,未改)。
- 数据:safe40(force_safe 0.4,random 0.05,noise 0.1,6000 集,冻结)。
- 方法:γ=0.95,hazard V̲。
- 两 arm:swamp_safe40_manski_s0(causal)vs swamp_safe40_walkbase_s0
  (p_override=1),150k 步,bc 0.5,seed 0,唯一变量=悲观开关。
- 预注册预言:baseline final 100% 捷径(worst≈0,natural≈0.73);
  causal final 绕安全路(worst、natural → 1.0,对齐 always-safe oracle)。

## 5. 六条可写进 slides 的发现

1. **扇区旋转**:bin 边界切在行为主方向上会人为砍半 propensity——
   连续 propensity 估计的对齐问题,离散不存在。
2. **悲观随 horizon 复合**:γ=0.99(均值 100 步)对 12 步迷宫空洞;
   γ 是 bound 松紧的一阶旋钮(扫描选 0.95)。
3. **没有连贯覆盖就没有证书**:Manski 只能推荐行为策略认真走过的
   替代路;random 探索集对悲观方法是毒药(离散实验靠零 random 隐式
   满足了这一点)。
4. **worst-case 的忠实度**:连续环境的"软"陷阱(可恢复减速)需要在
   V̲ 里显式建模为吸收态(hazard),否则悲观无牙;静态几何 hazard 与
   离散 lava penalty 同级合法。
5. **混淆的几何**:入口选择型混淆(swamp)只给 Manski 一口;
   路径分布型混淆(wind)每步给一口。前者是 per-step Manski 的
   接近最坏情形——bound 强度 ∝ 行为的 u 依赖在动作分布中的可见度。
6. **offline 模型选择泄漏**:按环境 rollout 选 best checkpoint
   会替 baseline 做去混淆——严格对比必须用 final 或离线判据。

## 6. 开放问题(思考清单)

- **V̲ 自举**:BFS+hazard 是静态代理;正式版应换 critic 自举
  (spectral norm 保 Lipschitz / target network 保稳定),
  hazard 泛化为"从 V̲ 学到的低值吸收区"。
- **hazard 的合法性叙事**:静态几何(已知地图)vs 未知环境——
  离散版 docstring 的辩护可以照搬,但审稿人可能追问 learned V̲。
- **翻转幅度 vs 训练存活**:1.06x 死、1.79x 待验——中间的
  临界幅度值得一个 ablation(对 R4 成败都有解释力)。
- **多 seed**:R4 若成,3 seeds × 2 arms 出正式表。
- **propensity 条件化**:目前 cell 级;更细(子格/连续分类器)是否
  显著收紧 bound?
- **theory notes**:Lemma 2′(Lipschitz 松弛)+ hazard-V̲ 合法性
  + "混淆几何"观察,凑一节 discrete→continuous 的正式讨论。
- **scaling**:方法成立后,AntMaze(29 维、二阶动力学)与 pixel 版
  的顺序与必要性。

## 7. 工件索引

- 数据:datasets/swamp_matched_teacher_{s0,safe30_s0,safe40_s0}.npz,
  swamp_strong_waitteacher_s0.npz(全部冻结+manifest)。
- 表格/门:artifacts/manski_port*(propensity_table.npz、
  propensity_report.json、probe 报告、传送热图)。
- 诊断:scripts/manski_route_diagnosis.py(--no_critic --hazard)。
- 训练:swamp_{manski,walkbase}_s0(R1)、swamp_strongwait_*(R2/3)、
  swamp_safe40_*(R4,进行中)。
- 评测:scripts/eval_swamp_matched_deployment.py(matched)、
  eval_swamp_deployment.py(strong);对比用 final.pkl。
- 理论存档:notes/continuous_manski_lemma2prime.md。
