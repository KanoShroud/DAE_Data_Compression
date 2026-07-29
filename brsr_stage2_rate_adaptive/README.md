# BRSR Stage 2A：单动作可变率公平验证

本目录是独立于现有 BRSR Stage 1、Stage B 和主项目代码的研究脚本。
Stage 2A 不修改候选频点、波形、似然模型、RiskGreedy 评分或数值积分，
只回答一个新问题：

> 将 5-bit 反馈计入真实通信总量后，单动作贝叶斯风险策略能否在相同平均
> 总 bit 下稳定优于冻结的 Best-Static 和 SNR-only 静态方案？

## 冻结内容

- 候选池：`Current-Fisher`
- 似然：`Linear-Marginal`
- 数值积分：`5 x 64`
- 数值参考证书：`9 x 128`
- 基础载荷：2 个复频点、每个 I/Q 各 4 bit，共 16 bit
- 反馈命令：5 bit
- 新增动作：再获取 1 个复频点、I/Q 各 4 bit，共 8 bit

因此：

- Stop 总量：`16 + 5 = 21 bit`
- Acquire 总量：`16 + 5 + 8 = 29 bit`

## 数据隔离

四类随机种子完全分离：

1. prior：估计公开频谱统计量；
2. plan：冻结 16/24/32-bit 静态方案；
3. policy calibration：按 NLL 校准后验温度并冻结码率门控；
4. development：只做最终 Stage 2A 比较，不参与选频、温度或门控拟合。

Stage 2A 不生成 locked 结果。

## 运行

PyCharm 直接运行：

```text
brsr_stage2_rate_adaptive/run_stage2a.py
```

默认执行完整 development。命令行轻量检查可使用：

```powershell
D:\Software\anaconda3\envs\PyTorch\python.exe `
  brsr_stage2_rate_adaptive\run_stage2a.py --mode smoke
```

正式 development 由用户在 PyCharm 中运行。Codex 只执行 smoke。

## 主要输出

- `temperature_calibration.csv`
- `frozen_static_plans.csv`
- `frozen_rate_policies.csv`
- `stage2a_trial_results.csv`
- `stage2a_action_rows.csv`
- `rate_matched_effects.csv`
- `rate_matched_effects_by_snr.csv`
- `calibration_summary.csv`
- `threshold_sensitivity.csv`
- `gate_decision.json`
- `manifest.json`
- 4 张 SVG 诊断图

所有工程百分比门槛只作敏感性描述。Stage 2A 的核心统计门禁使用轨迹聚类
bootstrap 置信区间，不以单一 5% 阈值决定路线生死。

## Stage 2A只读失败机理证书

正式development未通过后，使用以下独立入口读取冻结CSV：

```text
brsr_stage2_rate_adaptive/diagnose_stage2a_failures.py
```

该脚本不重建信号、不拟合新策略、不修改阈值，也不覆盖源结果。它分离温度
点估计、温度动作重排、停止决策和动作排序造成的MSE变化，并检验非Oracle
评分裕量能否预测“自适应动作优于固定动作”。诊断状态只约束当前评分组件，
不等同于BRSR路线级裁决。

## Stage 2A.1：独立校准的安全动作回退

入口：

```text
brsr_stage2_rate_adaptive/run_stage2a1.py
```

Stage 2A.1不再评价停止决策。每个样本都发送基础层和一个新增复频点，并计入
5-bit反馈，总量固定为29 bit。默认发送冻结的BestStatic-B24频点；只有当：

1. 独立gate-fit集合选出的预测优势阈值具有正的保守收益；
2. 另一个独立certificate集合的单侧95%轨迹聚类bootstrap下界大于0；
3. 删除任意一条certificate轨迹后平均收益仍为正；

才允许在对应SNR使用预测动作，否则永久回退到固定动作。正式development使用
第三组全新种子，不参与阈值选择或安全认证。主比较对象是由24/32-bit静态方案
时间共享形成的同平均29-bit BestStatic和SNR-only强基线。

PyCharm直接运行时默认执行完整development；Codex只运行`--mode smoke`。

## Stage 2A.1只读协议可行性证书

入口：

```text
brsr_stage2_rate_adaptive/diagnose_stage2a1_protocol.py
```

该入口只读取冻结的Stage 2A.1正式CSV，不生成新信号、不拟合阈值，也不改变
估计器。可实施的块级反馈协议要求同一块中的全部观测共用一个动作，5-bit动作
索引只发送一次。逐样本动作结果仅作为不可部署的乐观摊销上界，不参与协议
GO/HOLD裁决。
