# BRSR共享状态StateOracle证书

本目录独立检验一个问题：

> 当多个数据包共享真实慢变物理状态时，已知该状态但不知道未来噪声的
> `StateOracle`，能否稳定优于同通信预算的强静态前沿？

该证书不训练`ObservablePolicy`，不修改既有BRSR实现，也不读取locked结果。
只有StateOracle证明确实存在“状态驱动且非事后择优”的动作价值后，才允许另行
审批可部署策略。

## 1. 冻结组件

- 波形：1024点BPSK-RRC，40 MHz，2 samples/symbol。
- 候选池：`Current-Fisher`，12个候选频点。
- 似然：`Linear-Marginal`。
- 数值积分：`5 x 64`。
- 量化：两个基础复频点各4-bit I/Q；新增一个复频点各4-bit I/Q。
- 单包前向载荷：基础16 bit + 新增8 bit = 24 bit。
- 块级动作反馈：5 bit，每个相干块只发送一次。
- TDOA范围、连续复增益先验、公共量化范围和温度校准均复用冻结Stage 2A。

## 2. 三种预注册场景

| 场景 | 块内共享状态 | 目的 |
|---|---|---|
| `S0-independent-awgn` | 无；每个数据包独立生成TDOA、复增益、符号和噪声 | 复现当前主动选择缺乏可观测状态的负面边界 |
| `S1-coherent-awgn` | TDOA和复增益固定；符号和噪声独立 | 检验相干几何/增益状态是否足以改变最优动作 |
| `S2-coherent-notch` | 与S1相同，并共享A/B共同可见的频谱陷波状态 | 检验可观测频谱异质性是否产生稳定动作价值 |

S2第一版只引入一个块内共享的公共线性相位FIR陷波，不同时加入多径、NLOS或
B端独有干扰。陷波中心固定从候选频点`137/198/251`中平衡选择，半宽12个FFT
bin，FIR长度65；滤波后恢复原平均功率，避免把总SNR变化误认为选频收益。

S1和S2对同一`state_id`复用相同TDOA、复增益、BPSK符号和标准噪声，S2只额外
施加冻结陷波，便于归因。planning与evaluation使用互相独立的符号和噪声流。

## 3. StateOracle定义

对每个共享状态`z`，只使用planning数据估计每个B24动作的条件期望损失：

\[
\widehat R(a\mid z)
=
\frac{1}{N_{\mathrm{plan}}}
\sum_{n=1}^{N_{\mathrm{plan}}}L_{a,n}.
\]

然后冻结：

\[
a_{\mathrm{state}}(z)
=
\arg\min_a \widehat R(a\mid z).
\]

evaluation数据只用于评价已冻结动作，不参与选动作。该策略知道真实状态，
因此仍不可部署；但它不知道evaluation符号、噪声和动作结果，不属于历史
`ClairvoyantOutcomeOracle`。

S0没有共享状态，`StateOracle`强制退化为SNR-only计划，不允许按独立样本真值
选择动作。

## 4. 强基线

- `BestStatic-B24/B32`：planning上跨SNR选择全局固定配置。
- `SNR-only-B24/B32`：planning上按SNR冻结配置。
- `ContextStatic-B24/B32`：S2中按公开陷波类别和SNR冻结配置；S0/S1退化为
  SNR-only。
- `StateOracle-B24`：按共享完整状态和SNR选择一个B24动作。

B32在两个基础频点之外穷举所有两频点组合，不使用贪心近似。

## 5. 通信账本

S0没有共享块，StateOracle每包总量为：

\[
B_{\mathrm{S0}}=24+5=29\ \mathrm{bit}.
\]

S1/S2在包含`L`个evaluation包的相干块内共享一个动作。正式development
预注册`L=4/8/16`，使用同一组16个evaluation包完成块长敏感性分析：

\[
B_{\mathrm{block}}
=
24+\frac{5}{L}\ \mathrm{bit/packet}.
\]

本阶段StateOracle直接知道状态，不使用pilot，因此不计pilot成本；这只用于判断
动作空间上限。未来ObservablePolicy必须单独计入任何新增pilot载荷和延迟。

静态同总bit基线对每个块长分别使用与结果无关的稳定hash，在冻结B24/B32计划
之间时间共享到相同平均通信量；允许的平均bit误差仅来自有限样本下不可避免的
整数分配量子。

## 6. 裁决

每个场景同时报告：

1. StateOracle相对B24静态基线的同前向载荷效应；
2. StateOracle相对B24/B32时间共享基线的同总bit效应；
3. 配对MSE改善、块级cluster bootstrap 95% CI、跨state seed方向和MDE；
4. planning动作裕量、动作稳定性、量化截幅率和实际平均bit。

每个块长分别裁决。场景通过要求至少一个预注册块长的StateOracle相对该场景
全部预注册强基线：

- 配对MSE改善点估计为正；
- 95% CI下界大于0；
- 每个独立state seed平均改善为正；
- 通信账本和planning/evaluation隔离全部通过。

不使用任意5%工程阈值。裁决含义：

- S1或S2通过：只批准设计新的`ObservablePolicy` development；
- 只在较长块通过：主张必须限定到相应最小相干块长；
- 点改善为正但CI未闭合：`HOLD_POWER_STATE_HEADROOM`；
- S1和S2均无正向StateOracle余量：`NO_GO_ROUTE_STATE_HEADROOM`；
- 不生成locked结果。

## 7. 运行

PyCharm直接运行：

```text
brsr_state_oracle_certificate/run_certificate.py
```

默认执行`development`。Codex验证使用：

```text
python brsr_state_oracle_certificate/run_certificate.py --mode smoke
```

输出目录：

```text
运行结果/BRSR/BRSR_STATE_ORACLE_<timestamp>_<mode>
```

正式development使用3个独立state seed、每个seed 4个状态、每个状态32个
planning包和16个evaluation包，只覆盖`-10/0/5/15 dB`四个预注册锚点。smoke
使用更小规模且只验证工程链路，不构成性能证据。
