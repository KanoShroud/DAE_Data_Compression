# BRSR Transition Regime

本目录独立验证“中等 SNR 模糊度转折区 + 块内稳定频谱异质性”是否为主动频谱精化提供可重复价值。
它不修改既有 BRSR 代码，也不推翻宽 SNR 共享状态路线的
`NO_GO_ROUTE_PLANNING_NONCONVERGENCE`。

## 预注册边界

- 主 SNR：`0/5 dB`；`-10/15 dB`仅作负对照。
- 平坦 AWGN 为机制负对照，块相干公共频谱陷波为目标场景。
- 两个场景使用相同 TDOA、复增益、BPSK 符号种子、候选频点、量化器和估计器。
- `ContextStatic`获得与目标方法相同的陷波类别信息；2-bit类别索引计入通信量。
- StateOracle动作反馈为5 bit/块，类别索引与反馈均按`L=16`摊销。
- StateOracle只从planning数据选动作，不读取evaluation结果。
- 奇偶planning包交叉拟合必须先证明动作可辨识；随后独立evaluation再验证主效应。
- GO只允许另立可部署策略development，不代表方法已经成功，也不生成locked结果。

PyCharm直接运行`run_experiment.py`默认执行development。Codex仅运行
`python run_experiment.py --mode smoke`进行工程链路验证。

