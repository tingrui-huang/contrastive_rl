
## E27｜死亡/持续吸收修复的因果比较：Stage 1 构造门槛未通过（2026-09-13）

- 范围：本地及远程 `feature/pointmaze-causal-transition` HEAD 均为 `f85a5f44b176120333a1de0f83d6d0d02a2b290e`，没有更新的 PointMaze 提交。主目录在 AntMaze；本次使用既有 `.worktrees/pointmaze-p30`，没有切换或修改 AntMaze 工作。
- 新问题：不是重跑 E02/E10/E15 的死亡识别或冻结模型吸收审计，而是检验能否保留原 learned kernel 作为 A，并在 B 的生成分支上定义、验证 native death onset 与持久后果。已先写 Stage 1 protocol；先前 source/ledger 阅读属于 reconnaissance，不冒充事前未知证据。
- 精确缺口：原 nominal/diagonal/response 只给可见条件下的抽样，没有连接 `x'`、当前隐藏 swamp mask、teacher memo、生成 anchor/noise 与 persistent death 的联合时序模型。teacher 的 `x'` 依赖 mask，不能独立补一个 p=.30 mask 后声称得到真实条件 ETT。此独立-mask hybrid 可以自洽定义，但只检验其自定 killing rule，尚未验证 native onset。完整 native 分支则同时替换运动/动作响应/碰撞等机制；不能把其 death label 转贴到已分歧的模型分支。
- 同架构控制：A/B 都可携带辅助 death 状态，仅 B 执行吸收。但保持 live kernel 不变时，加入 dead mixture 通常改变可见 diagonal law；补偿 live law 又引入第二个机制和未验证的时序 latent posterior。原 XY action-Lipschitz 证明不自动覆盖跳变 death bit 或完整多步过程。没有证明所有增广族不可能，也没有要求离线数据必须识别真实 ETT 才能做 simulator-assisted diagnostic。
- 执行：只做源码/历史/报告审阅、解析构造和文件校验。新 native steps、模型抽样/确定性 emitter calls、ETT/critic/actor/nominal updates 全部为 0。未实施或验证修复；按用户 gate 停在 Stage 1，没有运行 Stage 2/3，没有替代 sweep 或追加预算。
- 结论：已有特定 native-dead context 的模型运动/正回报不匹配仍成立。死亡机制是否造成重大决策相关价值误差（H1），以及修复是否改善 native policy（H2），均未解决。未测新 value/native 差值，移除 baseline discrepancy 的比例未定义，不写成零效应。
- 预设 major-effect 标准（未执行）：至少 .02 normalized dangerous-action error reduction、移除同一新评估人群至少 25% 正 baseline gap，并减少至少 .01 action-contrast error；native 阶段相对模型 A 与 ordinary offline 两个控制均需 .02 normalized return 和 5 个百分点 absorbing-failure 改善及相应区间支持。不能用当前无实验结果判定这些标准通过或失败。
- 下一步的必要条件仅为：给出 A 可恢复、generated branch 本地自洽且有 native onset 依据的联合时序构造，或独立验证明确假设的 simulator-assisted onset surrogate，并列明 diagonal/约束/模型族变化。不是另一个 death detector/critic calibration/optimizer sweep 的默认任务。

### 历史建议的当前状态（保留旧文，不覆盖历史）

- E02 的“接着做 bounded off-diagonal”已被 E06/E08/E15/E19/E22 等后续工作覆盖；**已完成/被后续实验取代**。
- E10 structure audit 的“实现 optional fork segment 再搜索”已被 geometry comparison 覆盖；**已完成**。其 death audit 的全局“任何悲观/下游实验前必须修复 absorption”要求仍按 E11 **撤销为通用前置条件**；本次只因用户指定的 death-repair 因果 gate 而停止。
- E15 的“因 late roots 停止”是该次实验的范围限制，early/phase/future-average 等 E16-E19 已继续；**不是当前待办**。
- E22 的“提高 saved pair MC 精度”已由 E23 完成，E24 已进一步分析排序；**已完成，不自动再扩预算**。E23 calibration 优先级仍是历史解释，不能替代本次的 event-construction gate。
- E25 的“oracle 更支持 optimizer 而非 family”已被 E26 **纠正**；Frobenius/spectral 比较已完成。不能把 E26 未达预设效应写成完全无下降。

本次产物：[简报](D:/Users/trhua/Research/contrastive_rl/.worktrees/pointmaze-p30/outputs/death_repair_identifiability_20260913_v1/REPORT.md)、[精确构造审计](D:/Users/trhua/Research/contrastive_rl/.worktrees/pointmaze-p30/outputs/death_repair_identifiability_20260913_v1/CONSTRUCTION.md)、[协议](D:/Users/trhua/Research/contrastive_rl/.worktrees/pointmaze-p30/outputs/death_repair_identifiability_20260913_v1/PROTOCOL.md)。原 ledger 已按字节备份在该 fresh 目录。未 commit/push。
