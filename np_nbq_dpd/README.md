# NP-NBQ-DPD Stage 0

本目录独立实现“面向未知波形被动 TDOA 的干扰参数投影与网络总预算量化直接定位”最小结构余量证书，不导入或修改 DAE、BRSR、SITQ、RAPTQ 及主评估链路。

## 科学问题

在统一量化器、24 payload bit 和直接位置后验下，联合利用多站几何与公开链路可靠性设计节点--频点--位宽分配，能否稳定优于均匀分配、单因素分配、Fisher 代理和 Jiang 式预设混合位宽。

## Stage 0 边界

- 3 个接收节点、2 个频点和离散二维 ROI；全部合法 `0--3 bit/real` 分配在 24 bit 下穷举。
- 单径 AWGN；每个节点的幅度/SNR 与 BER 由独立 pilot 或慢变链路状态公开提供。
- 固定第一频点源幅度和参考节点复增益作为尺度/相位规范；第二频点源幅度比、每频点源相位以及两个非参考节点的频带内常复增益幅度/相位采用有限先验，并在位置似然中完整边缘化。
- 分配只使用 ROI 先验、节点坐标、公开 SNR 和 BER，不读取真实位置、clean 波形或动作后误差。
- `JiangStylePresetHybrid` 只迁移 Jiang 2026 的预设混合位宽原则；它不是原论文能量观测、FIM 阈值和 GA-PDR 的严格复现。
- 当前结果不能外推到连续任意源谱/复增益、频率选择性信道、多径或 NLOS。

## 方法

- `EqualBit`：每个节点--频点的 I/Q 分量均使用 2 bit。
- `JiangStylePresetHybrid`：按外生、预共享的节点带宽能力类别预设 1/2/3 bit，并在两个频点重复；不根据本轮 sensing SNR 反推类别。
- `GeometryOnly`：实际几何、统一名义可靠性下最小化 design 后验风险。
- `ReliabilityOnly`：规范几何、实际公开可靠性下最小化 design 后验风险。
- `ProjectedFisher`：消除每频点源相位和每节点常相位后的 Schur 补相位信息代理。
- `JointRisk`：在实际几何和公开可靠性下穷举最小化 design 位置后验风险。
- `StrongestFrozenBaseline`：仅在 design 中从合法基线冻结，evaluation 不重新选择。
- `UnquantizedAll-Diagnostic`：非 bit 可比的信息下界，只作诊断。

量化区间按数值从小到大使用 `natural_binary` 标签，数字链路对标签的每一位施加独立 BSC。该映射是 Stage 0 冻结模型的一部分；当前结果不能直接外推到 Gray 码、信道编码或软译码链路。

## 运行

在 PyCharm 中直接运行 `np_nbq_dpd/run_stage0.py`。顶部 `RUN_MODE="development"` 是正式 development 配置；无需环境变量。Codex 使用 `--smoke` 仅验证工程链路，smoke 不构成性能证据。

正式输出位于 `运行结果/NP_NBQ_DPD/NP_NBQ_STAGE0_<timestamp>_development/`，包括 manifest、机械检查、全部 design 风险、冻结分配、逐样本评价、汇总、profile 分层总体效应、逐 profile 效应、Bayes 校准、design 赢家/逐 cell 位宽稳定性、Fisher 排序诊断和两张 SVG。

## 裁决

主效应为 `JointRisk` 相对 design 冻结的最强合法基线的配对位置 MSE 改善。固定 4 个 profile 分别在内部重采样，再对 profile 均值等权汇总；同时要求总体相对 `GeometryOnly` 与 `ReliabilityOnly` 的 95% CI 下界为正，且三个非平衡 profile 均不得对最强基线出现 CI 上界小于 0 的明确回归。报告效应量、分层 CI、观测对数、正态 CI 半宽和双侧 5% 显著性/80% 功效的近似 MDE；不把 `1.96 × SE` 误称为 MDE，也不设置事后百分比门槛。

matched-Bayes 校准以 `mean posterior risk / empirical MSE` 和二者差值的 bootstrap CI 报告。比值超出 `[0.5, 2]` 只触发 `WARNING_REVIEW`，不是性能门禁。design 稳定性使用现有 design 样本的 bootstrap，报告赢家频率、top-k 风险差和逐 cell 位宽选择频率；它只限制结构解释，不改变冻结分配或 evaluation 主效应。
