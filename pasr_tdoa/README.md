# PASR-TDOA 独立验证模块

本目录实现“剖面后验驱动的主动逐层精化 TDOA 编码”（PASR-TDOA）的第一阶段最小可证伪实验。该模块不导入、不修改 `main.py`、DAE、V5、urban8 或现有定位评估链路。

## 研究问题

在相同平均 bit 和最大 bit 约束下，根据当前多峰 TDOA 后验主动请求下一频点，能否超出固定频点方案的风险与精度 Pareto 前沿？

## 文件

- `pasr_core.py`：剖面似然、连续 TDOA 后验、模态提取、有限比特量化、bit 账本和主动请求。
- `run_pasr_tdoa.py`：PyCharm 运行入口、静态强基线、Monte Carlo、统计门禁和结果导出。
- `test_pasr_core.py`：理论不变量与核心函数测试。

## 运行

直接在 PyCharm 运行 `run_pasr_tdoa.py`。文件顶部 `PYCHARM_RUN_MODE` 默认为 `research`，点击运行即执行正式独立验证。轻量检查时可临时改为 `smoke`。完整研究运行应由用户在 PyCharm 中执行。

## 预算口径

每个复频点按 I/Q 各 `qbits` 计费；每个样本只发送一次量化尺度；继续/停止标志、下一频点动作索引均计入反馈。固定掩码不需要动作反馈。所有方法共享同一量化频谱、真实时延、噪声和抖动。静态基线额外扫描 4--10 bit I/Q，以便在主动方法包含反馈开销时仍能构造相同总 bit 的静态 Pareto 包络。

反馈只在协议确实需要时计费：PASR 按实际继续/停止和动作索引计费；`Fixed-Nested` 的固定顺序预共享，因此不收动作索引；`PASR-Always4` 预定发送至4频点，因此不收停止决策位。

减性抖动的伪随机种子和静态时分调度表视为双方预共享，不按样本重复计费；该假设会写入结果 manifest。名义 SNR 定义为衰减前平均源功率与 A/B 两端相同复噪声方差之比，因此 B 端接收 SNR 还包含 `|rho|^2` 衰减。

## 公平性与统计协议

- development 与 locked 使用完全不相交的随机种子。
- 同平均 bit 的静态时分组合、同风险静态组合和高 SNR 静态参考只在 development 上选择，随后冻结；locked 结果不得参与对手选择。
- `research` 预注册 6-bit I/Q 为主检验，4-bit 和 8-bit 仅作敏感性分析，避免“任一位宽通过即成功”的多重选择偏差。
- 所有可辨识固定掩码均被枚举，但 `Exhaustive-Proxy-M2/M3/M4` 是按确定性 Chernoff/Fisher 代理选出的掩码，不是依据 Monte Carlo 风险得到的经验全局最优，因此不能表述为“穷举最优静态方法”。
- 静态对手池还包含 PFRS、Zhai-Fisher、Power 和 Uniform；所有静态方法均扫描 4--10 bit I/Q。
- 运行时会强制检查方法键唯一、数值有限、bit 分量恒等式、平均/最大预算、development/locked 隔离、跨方法同源 TDOA 和跨 SNR 配对。

## 结果边界

当前实验只验证固定确定性源谱、未知公共复增益和等噪声的非对称双站最小模型。A 端完整频谱位于融合端，B 端逐层传输有限比特频谱；结果不是 urban8、定位误差或跨信道泛化结论。`smoke` 只验证流程，不用于方法排名。

主要输出包括逐样本结果、动作轨迹、静态设计搜索、按 SNR/总体汇总、方法元数据、development 冻结的对手表、GO/NO-GO JSON、manifest 和三张 SVG。
