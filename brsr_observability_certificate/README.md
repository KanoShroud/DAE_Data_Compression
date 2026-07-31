# BRSR 可观测性证书

本目录是独立诊断入口，不修改冻结的 BRSR Stage 2A/2A.1 算法。

## 研究问题

历史 `ClairvoyantOutcomeOracle` 知道真实 TDOA 和所有动作结果，只能给出
不可部署的事后下界。该证书改为检验：只使用动作发生前可获得的信息，能否预测
各候选动作的条件风险，并在独立测试集上稳定超过同为 29 bit 的静态强基线。

## 数据隔离

- 冻结来源：`BRSR_STAGE2A_20260724_153144_development`。
- 固定模型：`Current-Fisher`、`Linear-Marginal`、`5x64` 数值积分。
- 重新生成互不重叠的 train、calibration、test seed 池。
- calibration 只选择预注册的 Ridge 或 HGB；test 不参与模型选择。
- 真值、动作后观测和动作实际误差只作为监督标签，不进入特征白名单。

## 公平比较

- 自适应方法：24 bit 载荷 + 5 bit 反馈，共 29 bit。
- 静态方法：24/32 bit 基于公开 cluster hash 时间共享，平均严格为 29 bit。
- `ClairvoyantOutcomeOracle-B29` 只画作描述性下界，不参与 GO/HOLD/NO-GO。
- 历史 IID `BlockOracle` 标为 `NO_GO_PROTOCOL_IID_BLOCK`。

## 运行

PyCharm 直接运行 `run_certificate.py` 默认执行正式 development。Codex 只运行：

```powershell
D:\Software\anaconda3\envs\PyTorch\python.exe `
  brsr_observability_certificate\run_certificate.py --mode smoke
```

正式运行结果写入：

```text
运行结果/BRSR/BRSR_OBSERVABILITY_<timestamp>_development
```

正式裁决以 `gate_decision.json`、`observable_paired_effects.csv` 和
`observable_prediction_diagnostics.csv` 为准。
