# BRSR S1-only StateOracle精度复验

本目录执行一次预注册的确认性复验，只回答：

> 在冻结S1相干AWGN场景、算法、候选池、似然、量化、通信预算和门禁后，
> StateOracle相对同总bit静态前沿的小幅正效应能否在更多独立状态上重复？

它不修改既有BRSR或StateOracle证书，不实现`ObservablePolicy`，不使用
`OutcomeOracle`，也不读取或生成locked结果。

## 1. 历史设计依据

冻结来源：

```text
运行结果/BRSR/BRSR_STATE_ORACLE_20260731_170154_development
```

S1、`L=16`相对两类同bit基线的最小MSE点改善为
`0.3045507927 sample²`，最大state-cluster标准差为
`0.6935536535 sample²`。使用双侧`alpha=0.05`和80%功效的正态近似：

\[
N_{\min}
=
\left\lceil
\left[
\frac{(z_{0.975}+z_{0.80})\sigma_{\mathrm{cluster}}}{\Delta}
\right]^2
\right\rceil
=41.
\]

正式复验预注册48个独立状态，即6个新state seed、每个seed 8个状态。
这不是持续扩样直到显著；该样本量和种子在正式运行前固定。

## 2. planning收敛校准

原正式证书只使用32个planning包，S1分半动作一致率约14.6%。因此正式
效应复验前，先在独立的calibration states上比较：

```text
候选planning包数：32、64、128
冻结参考：256
独立check包：128
```

候选动作只由selection stream选择，再在独立check stream上计算相对
256包参考动作的MSE regret。check stream再做两折交叉拟合，检验256包参考
本身是否不劣于由独立check半流选出的动作。这样不能用“128和一个同样不可靠的
256动作恰好一致”伪造收敛。主等价界固定为历史最小效应的10%：

\[
\epsilon_{\mathrm{plan}}
=0.1\times0.3045507927
=0.0304550793\ \mathrm{sample}^2.
\]

该界表示planning近似最多消耗历史点效应的10%。同时固定报告5%、10%和20%
三档敏感性，不根据结果修改主界。按成本从低到高选择首个同时满足：

1. 平均regret不超过主等价界；
2. state-cluster bootstrap 95% CI上界不超过主等价界；
3. 每个calibration seed的平均regret不超过主等价界。
4. 256包参考动作的交叉拟合regret满足相同三项条件。

若128包仍不通过，正式development直接裁决
`NO_GO_ROUTE_PLANNING_NONCONVERGENCE`，不继续生成确认结果。

calibration selection/check状态与confirmation状态完全互斥。

## 3. 确认性复验

只有planning收敛通过后才运行：

- 场景：仅`S1-coherent-awgn`；
- SNR：`-10/0/5/15 dB`；
- 状态：6个新seed × 8个状态，共48个独立state cluster；
- 每状态evaluation包：16；
- 块长：只确认历史候选`L=16`；
- StateOracle载荷：24 bit前向载荷 + `5/16` bit/packet反馈；
- 静态基线：使用与结果无关的B24/B32时间共享严格匹配平均总bit。

主比较固定为：

```text
StateOracle-L16-B24
vs BestStatic-B24
vs SNR-only-B24
vs BestStatic-RateMatched-L16
vs SNR-only-RateMatched-L16
```

## 4. 一次性裁决

`PASS_S1_PRECISION_REPLICATION`要求StateOracle相对全部四类基线：

1. 配对MSE点改善全部为正；
2. state-cluster bootstrap 95% CI下界全部大于0；
3. 6个独立state seed的平均改善全部为正；
4. 全部机械、公平性和通信预算检查通过。

通过只允许另行预注册`ObservablePolicy` development，不代表BRSR已经可部署。
planning不收敛或确认效应未通过时，当前共享状态BRSR路线结束，不再追加样本或
修改门槛。

## 5. 运行

PyCharm直接运行：

```text
brsr_s1_precision_replication/run_replication.py
```

默认执行正式`development`。Codex仅运行：

```text
python brsr_s1_precision_replication/run_replication.py --mode smoke
```

输出目录：

```text
运行结果/BRSR/BRSR_S1_PRECISION_<timestamp>_<mode>
```
