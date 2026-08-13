# RAPTQ-TDOA Stage 0

## RelativePhase 24-bit最终可行性周期

`run_relative_phase_rate.py`是独立的PyCharm入口，不修改历史Stage 0、SITQ或
DAE结果。它使用全新且互斥的design/evaluation trajectory，比较：

- `FullComplex`：未量化完整复观测参考；
- `RelativePhase-Continuous`：完整五维相对相位的连续信息参考；
- `CondMI-Matched24`：六个复频点各使用2-bit I和2-bit Q，总计24 bit；
- `RAPTQ-RelativePhase24`：以第一个高能量频点为相位锚点，五个相对相位按
  `5/5/5/5/4 bit`固定量化，总计24 bit。

最后一个4-bit坐标在正式运行前按冻结的
`source_power * frequency_difference**2`局部信息杠杆确定；不读取design或
evaluation性能选位宽。两种有限码率方法使用相同频点、连续公共相位模型、
TDOA先验、增益幅度先验和Bayesian后端，均无索引或反馈开销。

直接在PyCharm运行`raptq_tdoa/run_relative_phase_rate.py`。正式模式先冻结
`CondMI-Matched24`阈值，再检查连续相位周期积分和两类RelativePhase QMC的
一致性。阈值落在搜索边界或任一数值风险差超过
`delta_guard=0.0625 sample²`时，脚本在性能评价前停止。只有数值门禁通过后，
才输出全新trajectory上的连续信息headroom、同24-bit Bayes风险/MSE、严重错峰
和计算时间。`--smoke`仅验证工程链路，不构成性能证据。

本目录独立实现参考辅助复投影任务量化（RAPTQ-TDOA）的理论与连续信息
审计，不导入、不修改SITQ、BRSR、DAE、定位器或主评估代码。

## 当前研究状态

`CANDIDATE_UNDER_THEORY_AND_INFORMATION_AUDIT`

历史Stage 0只批准：

1. `Stage 0-T`：匹配Bayesian模型、连续均匀复增益相位、全局相位商
   不变性、径向加投影表示等价性，以及有限窗口非酉时延算子的机械证书；
2. `Stage 0-I`：连续统计量的TDOA Bayes风险审计。

历史Stage 0本身不实现24 bit量化。根据其后续数值证书与稀疏表示止损结果，
当前另行批准上文所述的`RelativePhase`最终有限码率可行性周期；任务码本、
联合选频、神经网络和真实波形仍未批准。

## 与SITQ的关系

- 复用SITQ Stage A冻结的6个频点、RRC功率形状、TDOA网格和增益幅度；
- 不复用SITQ的8点离散复增益相位。RAPTQ使用连续均匀相位，并通过
  Bessel函数解析边缘化，以免有限相位网格制造伪不变性；
- 每条轨迹使用一个独立复增益，并在该轨迹的SNR扫描中共享。若未来改为
  多包共享增益，必须重新定义商空间作用域。

## 连续表示

| 表示 | 含义 |
|---|---|
| `FullComplex` | 选定复频点全信息参考 |
| `Radius+Projector` | 与原观测仅差公共相位的无损等价表示 |
| `ProjectorOnly` | 丢弃径向统计量 |
| `RelativePhase` | 完整相对相位，RAPG连续上界 |
| `RelativePower` | 完整相对幅度 |
| `SparsePhase` | 两个固定相对相位，真正有损 |
| `SparseAmpPhase` | 两个相对幅度与相位，真正有损 |

连续有损表示的似然采用显式复极坐标：平方径向、功率单纯形、公共相位和
相对相位。连续增益相位已经使似然对公共相位严格不变，因此公共相位现
在解析删除，不再浪费QMC维度。体积元素中的平方径向幂通过Gamma重要性
分布抵消，其余被积坐标使用scrambled Sobol QMC。所有低能量样本保留并
单独记录，不通过阈值删除。

在当前匹配的对角循环相移模型中，`RelativePower`的分布严格不依赖TDOA，
其后验解析等于TDOA先验。该方法不再使用QMC，而是作为零TDOA信息负对照。

## Stage 0-I数值收敛证书

正式Stage 0结果`RAPTQ_STAGE0_20260805_105037_development`已经通过理论
证书，但高维稀疏表示在高SNR下未通过QMC可信度审查。独立诊断入口为：

`raptq_tdoa/diagnose_stage0_numerical.py`

该脚本固定读取上述正式manifest并重新生成两个预注册trajectory的全部
SNR context，不修改源目录。它比较：

- `2^8/2^10/2^12/2^14`主QMC与独立`2^15`参考；
- `RelativePhase`的公开模型proposal和防御性观测幅度proposal；
- `RelativePower`解析先验负对照；
- `ProjectorOnly`、`SparsePhase`和`SparseAmpPhase`的逐功率收敛；
- 每个SNR下表示排序是否随积分预算翻转。

防御性proposal使用模型预测幅度与当前完整观测幅度的等权整体混合，并用
完整混合密度进行重要性校正。它只作为离线数值参考；必须与不使用被丢弃
幅度的模型proposal收敛到同一后验，不能作为可部署方法的信息输入。

数值主容差采用一个TDOA网格步长平方，同时报告半个和两个网格步长平方的
敏感性。该门禁只认证信息排序和数值分辨率，不证明统计等价或24 bit性能。

## PyCharm运行

原始Stage 0审计入口：

`raptq_tdoa/run_stage0.py`

已经完成的历史数值收敛证书入口：

`raptq_tdoa/diagnose_stage0_numerical.py`

已完成的稀疏积分器认证入口：

`raptq_tdoa/diagnose_sparse_integrator.py`

已完成的36-context纠正性重算入口：

`raptq_tdoa/recompute_stage0_sparse_amp_phase.py`

当前已批准、待正式运行的最终可行性入口：

`raptq_tdoa/run_relative_phase_rate.py`

该入口不修改`core.py`的冻结积分路径，而是在独立模块
`raptq_tdoa/sparse_integrator.py`中使用无冗余Dirichlet stick-breaking
坐标和两种严格重要性校正proposal。运行顺序为：机械证书、7个冻结压力
context开发筛查、通过表示的原12-context协议全覆盖复核。7个压力context
来自原12-context子集，因此复核不是独立统计确认；24 bit方法和全新
trajectory仍需另行审批。

三个入口脚本顶部`RUN_MODE = "development"`均可由PyCharm直接运行，不需要
环境变量或命令行参数。当前不应重复运行原Stage 0或历史数值证书；Codex
使用`--smoke`执行的短运行只验证工程链路，不构成研究性能或数值收敛证据。

`diagnose_sparse_integrator.py`同样默认`RUN_MODE = "development"`，可直接在
PyCharm运行。正式配置固定使用`2^14/2^15/2^16`点、每档8个scramble；不会
根据运行结果自动扩大预算或放宽`delta_guard=0.0625 sample²`。历史
`HOLD_NUMERICAL_CONVERGENCE`不会因新协议而被追溯修改。

`recompute_stage0_sparse_amp_phase.py`只使用已认证的`SparseAmpPhase`积分器，
对原Stage 0的2个seed、每个seed 3条trajectory和6个SNR共36个context做
纠正性重算。它固定比较`model/defensive_mixture`两种proposal和两套独立QMC
seed family，正式预算为`2^16`点×8个scramble。四个分量按固定等权在似然层
合并，不依据结果选择有利分量；逐context的四分量后验风险跨度必须不超过
`delta_guard`。该运行仍使用历史development trajectories，只纠正信息审计，
不是独立统计确认，不评价`SparsePhase`、CondMI、24 bit量化或Stage 0-R。

输出目录：

`运行结果/RAPTQ/RAPTQ_STAGE0_<时间戳>_<mode>/`

数值证书输出目录：

`运行结果/RAPTQ/RAPTQ_STAGE0_NUMERICAL_<时间戳>_<mode>/`

稀疏积分器认证输出目录：

`运行结果/RAPTQ/RAPTQ_SPARSE_INTEGRATOR_<时间戳>_<mode>/`

36-context纠正性重算输出目录：

`运行结果/RAPTQ/RAPTQ_STAGE0_SPARSE_CORRECTION_<时间戳>_<mode>/`

优先回读：

- `config_manifest.json`
- `validation.json`
- `stage0_coordinate_registry.csv`
- `stage0_information_summary.csv`
- `stage0_paired_risk_effects.csv`
- `stage0_numerical_diagnostics.csv`
- `Fig1_RAPTQ_Stage0_InformationAudit.svg`

数值证书优先回读：

- `validation.json`
- `numerical_convergence_rows.csv`
- `numerical_convergence_summary.csv`
- `relative_phase_proposal_agreement.csv`
- `representation_ordering_stability.csv`
- `Fig1_RAPTQ_Numerical_Convergence.svg`

稀疏积分器认证优先回读：

- `validation.json`
- `mechanical_checks.csv`
- `representation_gate_summary.csv`
- `development_proposal_agreement.csv`
- `review_proposal_agreement.csv`
- `Fig1_RAPTQ_Sparse_Integrator_Certificate.svg`

36-context纠正性重算优先回读：

- `validation.json`
- `sparse_correction_numerical_summary.csv`
- `sparse_correction_context_rows.csv`
- `sparse_correction_summary_by_snr.csv`
- `sparse_correction_cluster_effects.csv`
- `Fig1_RAPTQ_SparseAmpPhase_Corrective_Audit.svg`

正式运行只有在机械证书通过且QMC误差经结果审查后，才可能裁决为
`APPROVE_RATE_AUDIT`。该状态仅允许另行审批固定24 bit粗量化，不证明
RAPTQ已经优于CondMI或构成最终创新方法。
