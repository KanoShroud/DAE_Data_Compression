# SITQ-TDOA Stage A

该目录独立验证“参考侧信息辅助的条件任务量化”机制，不导入或修改
DAE、BRSR、定位器和主评估代码。

## 当前边界

- A站选定频点完整位于融合端，B站只发送固定24 bit消息。
- A/B共享同一随机复高斯源频谱；源功率采用RRC形状。
- Stage A 的 TDOA 从`[-8, 8] sample`上的`0.25 sample`有限网格生成，
  与解码先验严格一致；连续子网格 TDOA 留到模型失配阶段验证。
- 未知复增益来自冻结离散先验，并由融合端严格边缘化。
- 6个频点、每个I/Q分量2 bit；频点和码率对所有方法完全相同。
- 没有动态掩码、索引、反馈、停止或波形重构。

比较方法：

1. `MSEQuant-24bit`：量化阈值最小化远端观测重构MSE；
2. `CondMIQuant-24bit`：量化阈值最小化TDOA后验熵；
3. `SITQ-BayesRisk-24bit`：量化阈值直接最小化平方误差损失下的
   后验贝叶斯风险（后验方差）；
4. `Unquantized-SelectedBins`：不计入同bit比较的浮点上界。

## PyCharm运行

直接运行：

`sitq_tdoa/run_stage_a.py`

脚本顶部`RUN_MODE = "development"`是正式development配置。Codex使用
`--smoke`完成的小样本运行只检查工程链路，不构成性能证据。

正式结果写入：

`运行结果/SITQ/SITQ_STAGE_A_<时间戳>_development/`

回读时优先检查：

- `validation.json`
- `config_manifest.json`
- `stage_a_summary.csv`
- `stage_a_paired_effects.csv`
- `threshold_design.csv`
- `Fig1_SITQ_StageA.svg`

Stage A通过后，才能审批联合频点/位宽设计和BPSK-RRC模型失配验证。

阈值候选在公开归一化尺度上覆盖`0.15`至`2.0`。若任一方法的最优值
落在搜索边界，脚本只给出`HOLD_THRESHOLD_SEARCH_BOUNDARY`，不得据此
作性能裁决。配对结果同时报告轨迹级95%置信区间和近似最小可检出效应。

## Stage A.1阈值稳定性证书

`diagnose_stage_a1_threshold_stability.py`只重建正式Stage A的design seeds，
不读取evaluation结果选阈值，也不修改Stage A算法和源结果。它输出：

- 每条design轨迹在全部候选阈值下的后验风险和熵；
- 轨迹分组bootstrap的阈值选择分布；
- 按完整轨迹划分的六折交叉冻结结果；
- 分seed风险方向、配对置信区间和近似MDE。

门禁评价的是交叉冻结后的风险能否稳定优于CondMI，而不是机械要求每次
选中完全相同的阈值。直接在PyCharm运行该脚本即执行完整诊断；`--smoke`
仅供工程检查。
