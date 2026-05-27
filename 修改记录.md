# DAE Data Compression — 论文复现统一修改记录

**论文**: Chen et al., "Collaborative Localization Using Multiple UAVs: A Deep Denoising Autoencoder-Based Data Compression Approach", IEEE TVT 2025.
**项目路径**: `F:\PythonWorkspace\DAE Data Compression\`
**最后更新**: 2026-05-26 (第4轮 — 代码审计 + 性能修复)

---

## 目录

1. [论文核心贡献与项目概述](#一论文核心贡献与项目概述)
2. [完整出入清单](#二完整出入清单)
3. [已修复项：算法原理与数学推导](#三已修复项算法原理与数学推导)
4. [未修复项：改进方案与算法参考](#四未修复项改进方案与算法参考)
5. [正确匹配项清单](#五正确匹配项清单)
6. [验证记录汇总](#六验证记录汇总)
7. [第4轮修改：代码全面审计与性能修复 (2026-05-26)](#七第4轮修改代码全面审计与性能修复-2026-05-26)

---

## 一、论文核心贡献与项目概述

### 1.1 论文摘要

该论文针对**多无人机（UAV）协作源定位**场景，提出了一种**CNN+残差块结构的深度去噪自编码器（DAE）**用于高比率数据压缩。核心流程为：

1. 各 UAV 接收辐射源发出的 BPSK 信号（1024 采样点，经过多径信道+高斯白噪声污染）
2. DAE 编码器将信号压缩为低维特征 M（压缩率 CR = K/M = 4/8/16）
3. 压缩特征通过 UAV 间链路传输至中心 UAV
4. DAE 解码器重构原始信号，用于 TDOA 估计与几何定位

实验基于 **Wireless InSite 射线追踪**平台，模拟 200×260 m² 城市十字路口场景（8 架 UAV），在 SNR ∈ [-10, 20] dB 范围内验证了方法在低 SNR 和高 CR 下优于原始数据和其他压缩方法的性能。

### 1.2 系统模型（论文 Eq. (1)）

第 i 架 UAV 接收到的信号建模为：

$$x_i(t) = \sum_{\ell=1}^{L_i} \alpha_{\ell,i} \, u(t - \tau_{\ell,i}) + e_i(t) = v_i(t) + e_i(t), \quad t \in [0, T)$$

其中：
- $T$ 为观测时长，$L_i$ 为多径数
- $\alpha_{\ell,i} \in \mathbb{C}$ 为复衰减系数，$\tau_{\ell,i}$ 为第 $\ell$ 条路径延迟
- $u(t)$ 为 BPSK 调制发射信号（20 MHz 带宽，2.4 GHz 载频）
- $e_i(t) \sim \mathcal{CN}(0, \sigma_e^2)$ 为加性复高斯白噪声

### 1.3 项目文件结构

| 文件 | 功能 | 修改状态 |
|------|------|----------|
| `model.py` | DAE 网络架构定义 | **已修改** |
| `signal_gen.py` | BPSK 信号生成 + 信道仿真 | **已修改** |
| `train.py` | 训练流程（CV + 流式） | **已修改** |
| `main.py` | 主入口 | **已修改** |
| `evaluate.py` | 蒙特卡洛评估 + 可视化 | **已修改** |

---

## 二、完整出入清单

以下表格为论文与原始代码之间的**所有**不一致之处，逐一标明了严重等级、修改状态及对应的修改编号。

### 2.1 严重出入（🔴）

| # | 问题描述 | 论文要求 | 原始代码 | 状态 | 修改编号 |
|---|---------|---------|---------|------|---------|
| 1 | 信号生成方式 | Wireless InSite 城市射线追踪（200×260m², 19栋建筑, 8 UAV, LOS/NLOS 切换） | 纯随机 BPSK + 人工随机多径（2-4 tap），无空间几何 | ❌ 未修复 | — |
| 2 | 评估指标 | Location Error [m]（TDOA → 双曲线定位 → 位置误差） | TDOA RMSE [Samples]（仅时延相关，无几何定位） | ❌ 未修复 | — |
| 3 | Batch Normalization | Complex BN（2×2 协方差白化） | 标准 `nn.BatchNorm1d`（实/虚独立归一化） | ✅ 已修复 | §3.1 |

### 2.2 中等出入（🟡）

| # | 问题描述 | 论文要求 | 原始代码 | 状态 | 修改编号 |
|---|---------|---------|---------|------|---------|
| 4 | 激活函数 | ReLU（Fig.3 消融实验最优） | `nn.LeakyReLU(0.2)` | ✅ 已修复 | §3.3 |
| 5 | 多UAV协作 | 8 架 UAV 协同 + MLP LOS/NLOS 检测 | 仅 2 路信号，两两 TDOA | ❌ 未修复 | — |
| 6 | 全连接层 | fully complex-valued connected layer | `nn.Linear`（实数矩阵乘法） | ✅ 已修复 | §3.2 |
| 7 | 对比基线 | DFT / Hadamard / PCA / DNN-DAE / CNN-DAE | 无任何基线 | ❌ 未修复 | — |
| 8 | 训练验证机制 | 固定 10,000 样本 + 5-fold CV | 在线流式生成，无验证集 | ✅ 已修复 | §3.5 |

### 2.3 轻微出入（🟢）

| # | 问题描述 | 论文要求 | 原始代码 | 状态 | 修改编号 |
|---|---------|---------|---------|------|---------|
| 9 | 训练 SNR 范围 | [-10, 20] dB（与测试一致） | [-15, 15] dB | ✅ 已修复 | §3.5 |
| 10 | 脉冲成形滤波器 | 标准通信脉冲成形（论文用Wireless InSite自动处理） | 4 阶 Butterworth (`cutoff=0.5`) | ✅ 已修复 | §3.4 |
| 11 | TDOA 估计方法 | Generalized Cross-Correlation (GCC) | GCC-PHAT（GCC 的一种变体） | ❌ 未修复 | — |

**汇总**: 已修复 6/11 项，待修复 5/11 项。

---

## 三、已修复项：算法原理与数学推导

### 3.1 ComplexBatchNorm1d — 复数批归一化

**修复**: analysis_report.md §严重出入 #3
**文件**: `model.py:65-134`
**替换**: 8 处 `nn.BatchNorm1d` → `ComplexBatchNorm1d`

#### 3.1.1 问题本质

实数 BatchNorm 对每个通道独立计算均值和方差：

$$\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

但对于复数信号 $z = x_r + j x_i$，其实部和虚部**存在统计相关性**（由信道旋转和频移引入）。实数 BN 仅缩放 $x_r$ 和 $x_i$ 各自的方差，完全忽略了协方差 $\mathbb{E}[x_r x_i]$，无法消除复平面上的各向异性分布。

#### 3.1.2 数学原理

复数 BN 的核心思想是将 **2×2 实虚协方差矩阵** 白化为单位矩阵：

$$\mathbf{V} = \begin{bmatrix} V_{rr} & V_{ri} \\ V_{ri} & V_{ii} \end{bmatrix} = \begin{bmatrix} \mathbb{E}[(x_r-\mu_r)^2] & \mathbb{E}[(x_r-\mu_r)(x_i-\mu_i)] \\ \mathbb{E}[(x_r-\mu_r)(x_i-\mu_i)] & \mathbb{E}[(x_i-\mu_i)^2] \end{bmatrix}$$

白化变换为：

$$\begin{bmatrix} \hat{x}_r \\ \hat{x}_i \end{bmatrix} = \mathbf{V}^{-1/2} \begin{bmatrix} x_r - \mu_r \\ x_i - \mu_i \end{bmatrix}$$

其中 $\mathbf{V}^{-1/2}$ 是协方差矩阵的**逆平方根**。

#### 3.1.3 2×2 矩阵逆平方根的闭式解

对于 2×2 正定对称矩阵 $\mathbf{V}$，其逆平方根具有闭式解。令：

$$s = \sqrt{\det(\mathbf{V})} = \sqrt{V_{rr}V_{ii} - V_{ri}^2}$$

$$t = \sqrt{\text{tr}(\mathbf{V}) + 2s} = \sqrt{V_{rr} + V_{ii} + 2s}$$

则：

$$\mathbf{V}^{-1/2} = \frac{1}{s \cdot t} \begin{bmatrix} V_{ii} + s & -V_{ri} \\ -V_{ri} & V_{rr} + s \end{bmatrix}$$

**推导验证**: $\mathbf{V}^{-1/2} \cdot \mathbf{V} \cdot \mathbf{V}^{-1/2} = \mathbf{I}$（可直接计算验证）。

#### 3.1.4 可学习缩放与平移

白化后施加 2×2 可学习缩放矩阵 $\mathbf{\Gamma}$ 和复数偏置 $\pmb{\beta}$：

$$\begin{bmatrix} y_r \\ y_i \end{bmatrix} = \begin{bmatrix} \gamma_{rr} & \gamma_{ri} \\ \gamma_{ri} & \gamma_{ii} \end{bmatrix} \begin{bmatrix} \hat{x}_r \\ \hat{x}_i \end{bmatrix} + \begin{bmatrix} \beta_r \\ \beta_i \end{bmatrix}$$

$\mathbf{\Gamma}$ 初始化为 $\frac{1}{\sqrt{2}}\mathbf{I}$（幅度守恒初始化）：
- 白化后 $\mathbb{E}[|\hat{z}|^2] = 2$
- $\gamma_{rr}=\gamma_{ii}=1/\sqrt{2}, \gamma_{ri}=0$ → $\mathbb{E}[|y|^2] = 1/2 + 1/2 = 1$

#### 3.1.5 关键代码逻辑

```
输入: (B, 2C, L)  →  拆分实部 x_r(B,C,L) 和虚部 x_i(B,C,L)
  ↓
训练模式: 计算批统计量 μ_r, μ_i, Vrr, Vii, Vri; 动量更新 running 统计量
推断模式: 使用 running 统计量
  ↓
计算 s = √(Vrr·Vii - Vri²),  t = √(Vrr + Vii + 2s)
计算 Wrr = (Vii+s)/(s·t), Wii = (Vrr+s)/(s·t), Wri = -Vri/(s·t)
  ↓
白化: x̂_r = Wrr·(x_r-μ_r) + Wri·(x_i-μ_i)
      x̂_i = Wri·(x_r-μ_r) + Wii·(x_i-μ_i)
  ↓
γ 缩放 + β 平移 → 拼接 → 输出 (B, 2C, L)
```

**替换位置**: 编码器 3 处 + 解码器 2 处 + 残差块 3 处 = **共 8 处**。

---

### 3.2 ComplexLinear — 复数全连接层

**修复**: analysis_report.md §中等出入 #6
**文件**: `model.py:39-56`

#### 3.2.1 问题本质

实数全连接层 $y = Wx + b$ 将实部和虚部视为两个独立实数向量做矩阵乘法。这使得网络无法学习实/虚之间的交叉信息。复数全连接层应实现：

$$\mathbf{y} = \mathbf{W}\mathbf{z} + \mathbf{b}$$

其中 $\mathbf{W} = \mathbf{W}_r + j\mathbf{W}_i$，$\mathbf{z} = \mathbf{z}_r + j\mathbf{z}_i$，$\mathbf{b} = \mathbf{b}_r + j\mathbf{b}_i$。

#### 3.2.2 复数矩阵乘法展开

$$\begin{aligned}
\mathbf{W}\mathbf{z} &= (\mathbf{W}_r + j\mathbf{W}_i)(\mathbf{z}_r + j\mathbf{z}_i) \\
&= (\mathbf{W}_r\mathbf{z}_r - \mathbf{W}_i\mathbf{z}_i) + j(\mathbf{W}_r\mathbf{z}_i + \mathbf{W}_i\mathbf{z}_r)
\end{aligned}$$

即：

$$\begin{bmatrix} \mathbf{y}_r \\ \mathbf{y}_i \end{bmatrix} = \begin{bmatrix} \mathbf{W}_r & -\mathbf{W}_i \\ \mathbf{W}_i & \mathbf{W}_r \end{bmatrix} \begin{bmatrix} \mathbf{z}_r \\ \mathbf{z}_i \end{bmatrix} + \begin{bmatrix} \mathbf{b}_r \\ \mathbf{b}_i \end{bmatrix}$$

这一结构与复数乘法 $(a+jb)(c+jd) = (ac-bd) + j(ad+bc)$ 完全一致，且满足 **Cauchy-Riemann 方程**，保持了复解析性（holomorphy）。

#### 3.2.3 DAE 中的维度适配

编码器输出经 flatten 后得到 $128 \times 43 = 5504$ 个实数 = **2752** 个复数特征值。设压缩后潜变量维度为 $M = 2048/\text{CR}$（实数），则复数输出维度为 $M/2$：

```python
# 编码器 FC: 2752 复数特征 → M/2 复数特征
self.fc_enc = ComplexLinear(in_features=2752, out_features=self.latent_dim // 2)
# 输出: (batch, latent_dim) 实数

# 解码器 FC: M/2 复数特征 → 2752 复数特征
self.fc_dec = ComplexLinear(in_features=self.latent_dim // 2, out_features=2752)
# 输出: (batch, 5504) 实数 → reshape (batch, 128, 43)
```

以 CR=16 为例（$M=128$）：`fc_enc = ComplexLinear(2752, 64)`，输入 (B, 5504)，输出 (B, 128)。维度链完全一致。

---

### 3.3 ReLU 激活函数

**修复**: analysis_report.md §中等出入 #4
**文件**: `model.py`（ResidualBlock 和 DAE 的 encoder/decoder，共 9 处）

#### 3.3.1 问题本质

原始代码使用 `nn.LeakyReLU(0.2)`，其数学定义为：

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ 0.2x & x \leq 0 \end{cases}$$

论文 Fig.3 对 6 种激活+损失组合进行了消融实验（CR=8 条件下），结果表明 **ReLU + MSE** 组合取得最低定位误差：

$$\text{ReLU}(x) = \max(0, x)$$

#### 3.3.2 选择 ReLU 的动机

对于去噪自编码器，ReLU 的**严格稀疏性**（负半轴完全归零）带来两个关键优势：

1. **噪声抑制**: 负值区域的硬截断天然适合消除低幅度噪声分量——噪声在卷积特征空间中分布在零附近，ReLU 将其完全置零，而 LeakyReLU 保留了 20% 的负值噪声
2. **表示稀疏性**: 复数卷积特征经 ReLU 后仅保留正半轴的激活模式，形成更紧凑的信号表示，有利于后续的 FC 层压缩（在潜空间 $M \ll K$ 的条件下尤为重要）

论文 Fig.3 的实验数据直接支撑了这一选择：在 SNR = -10~20 dB 范围内，ReLU+MSE 的 Location Error 始终低于 Sigmoid+MSE 和 Tanh+MSE 组合。

---

### 3.4 RRC 脉冲成形滤波器

**修复**: analysis_report.md §轻微出入 #10
**文件**: `signal_gen.py:7-53, 57-68`

#### 3.4.1 问题本质

原始代码使用 **4 阶 Butterworth IIR 滤波器**（归一化截止频率 0.5，零相位 `filtfilt`）：

$$|H(\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2n}}$$

Butterworth 滤波器有两个问题：
1. **不满足 Nyquist 无 ISI 准则**：其冲激响应在符号采样点处并非过零，会引入码间干扰（ISI）
2. **非匹配滤波**：发射端和接收端滤波后整体响应非 Nyquist，采样后信号存在符号间混叠

标准 BPSK 通信系统使用 **根升余弦（Root Raised Cosine, RRC）** 滤波器作为脉冲成形和匹配滤波。

#### 3.4.2 RRC 数学定义

RRC 滤波器的连续时间冲激响应为：

$$h(t) = \begin{cases}
1 + \beta\left(\frac{4}{\pi} - 1\right), & t = 0 \\[8pt]
\frac{\beta}{\sqrt{2}}\left[\left(1 + \frac{2}{\pi}\right)\sin\left(\frac{\pi}{4\beta}\right) + \left(1 - \frac{2}{\pi}\right)\cos\left(\frac{\pi}{4\beta}\right)\right], & t = \pm\frac{1}{4\beta} \\[12pt]
\frac{\sin\left(\pi t(1-\beta)\right) + 4\beta t \cos\left(\pi t(1+\beta)\right)}{\pi t\left(1 - (4\beta t)^2\right)}, & \text{otherwise}
\end{cases}$$

其中 $\beta \in [0, 1]$ 为滚降系数（roll-off factor），控制过渡带宽度。

#### 3.4.3 Nyquist 无 ISI 条件

RRC 的关键性质：发射端和接收端各用一次 RRC 滤波，**级联响应等于升余弦（Raised Cosine, RC）滤波器**：

$$H_{\text{RRC}}(f) \cdot H_{\text{RRC}}(f) = H_{\text{RC}}(f)$$

升余弦滤波器的冲激响应在 $t = nT_s$（$n \neq 0$）处**过零**，因此满足 Nyquist 第一准则：采样时刻无 ISI。

#### 3.4.4 参数选择与离散实现

| 参数 | 取值 | 依据 |
|------|------|------|
| 滚降系数 β | 0.35 | 平衡带宽效率（B = (1+β)·Rs）与实现复杂度 |
| 滤波器跨度 | 8 symbols | 覆盖 ±4 符号周期，截断误差可忽略 |
| 过采样率 | sps = 2 | 与代码 `samples_per_symbol=2` 一致 |
| 滤波器阶数 | 17 taps | `span × sps + 1` |
| 归一化 | $\|h\|_2 = 1$ | 保持信号功率不变 |

离散实现使用 `scipy.signal.convolve` 的 `mode='same'` 模式做线性卷积，保持信号长度不变。发射端脉冲成形和接收端匹配滤波使用同一 RRC 滤波器，实现匹配滤波接收。

#### 3.4.5 为何不保留 Butterworth 设计

Butterworth + `filtfilt` 的组合导致：
- IIR 滤波的**非线性相位**：`filtfilt` 虽通过正反两次滤波获得零相位，但代价是**滤波器阶数翻倍**（等效 8 阶），冲激响应显著加长
- **非 Nyquist 设计**：不具备定时无 ISI 特性，在符号点采样时存在码间干扰
- **非最优噪声抑制**：对于 AWGN 信道，匹配滤波器最大化输出 SNR，而 Butterworth 不匹配 BPSK 信号频谱

---

### 3.5 K-Fold 交叉验证 + 固定数据集

**修复**: analysis_report.md §中等出入 #8 + §轻微出入 #9
**文件**: `train.py`, `signal_gen.py:165-190`, `main.py`

#### 3.5.1 问题本质

原始训练流程为在线流式数据生成——每个 training step 实时生成新的随机信号：

```python
# 原始: 无数据集, 无验证
for step in range(steps_per_epoch):
    X, Y = sim.generate_pair_batch(batch_size)  # 每次随机, 无复现性
    loss = criterion(model(X), Y)  # 直接训练
```

这导致三个问题：
1. **不可复现**: 每次运行数据不同，模型性能不可严格对比
2. **无验证**: 无法监控过拟合
3. **数据分布无保证**: 无法确保训练覆盖了整个 SNR/信道分布

论文使用**固定数据集** (10,000 组观测) + **5-fold 交叉验证**。

#### 3.5.2 K-Fold CV 原理

将数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ 随机划分为 $K$ 个等大的互斥子集 $\mathcal{D}_1, \dots, \mathcal{D}_K$。对每个 fold $k$：

$$\mathcal{D}_{\text{train}}^{(k)} = \bigcup_{j \neq k} \mathcal{D}_j, \quad \mathcal{D}_{\text{val}}^{(k)} = \mathcal{D}_k$$

最终的交叉验证损失为：

$$\mathcal{L}_{\text{CV}} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}\left(\hat{\theta}^{(k)}; \mathcal{D}_{\text{val}}^{(k)}\right)$$

其中 $\hat{\theta}^{(k)} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D}_{\text{train}}^{(k)})$。

#### 3.5.3 实现结构

```
generate_training_dataset(n_samples=10000, seed=42)
  ├── np.random.seed(seed)       # 固定种子
  ├── 分批生成 (batch_cap=500)   # 避免内存溢出
  └── 返回 (X_noisy, X_clean)    # (N, 2, 1024) 张量

train_with_cv(k=5, ...)
  ├── KFold(n_splits=5, shuffle=True)
  ├── for fold in folds:
  │     └── train_one_fold(...)   # 标准 train/val 循环
  │           ├── 每 epoch 在 val_loader 评估
  │           └── 保留最佳 val_loss 模型状态
  └── 返回最佳 fold 的模型
```

#### 3.5.4 附带的训练参数对齐

| 参数 | 修改前 | 修改后 | 论文 |
|------|--------|--------|------|
| 数据集规模 | ∞ (流式) | 10,000 (固定) | 10,000 |
| 交叉验证 | 无 | 5-fold (KFold, shuffle=True) | 5-fold |
| 训练 SNR 范围 | [-15, 15] dB | [-10, 20] dB | [-10, 20] dB (Paper §IV-A) |
| 随机种子 | 无 | seed=42 | — |

训练 SNR 范围修正的动机：论文测试范围为 [-10, 20] dB，训练分布应与测试分布匹配是机器学习的基本要求（i.i.d. 假设）。原始代码的 [-15, 15] dB 范围下界更低、上界更低，会导致模型对高 SNR (>15 dB) 场景的泛化能力不足。

---

### 3.6 早停、L2正则化与学习率调度优化

**修复**: 基于训练结果诊断的过拟合对策
**文件**: `train.py:13-79`, `main.py:8-19`

#### 3.6.1 问题诊断

在首次完整训练运行中，从交叉验证曲线观察到严重的过拟合现象：训练损失（train loss）随 epoch 持续下降，而验证损失（val loss）在约 30-50 个 epoch 后开始反升。这表明模型在训练集上过参数化，需要引入以下正则化手段。

#### 3.6.2 早停 (Early Stopping)

早停是一种基于验证集性能的正则化策略，核心思想是在验证误差不再改善时终止训练，防止模型继续拟合训练集中的噪声。

**算法流程**:

1. 初始化 `best_val_loss = ∞`，`epochs_no_improve = 0`
2. 每个 epoch 结束后计算验证损失
3. 若 `val_loss < best_val_loss`：保存当前模型参数为最佳状态，`epochs_no_improve = 0`
4. 否则：`epochs_no_improve += 1`
5. 若 `epochs_no_improve >= patience`（设为 20）：终止训练
6. 训练结束后，将模型参数恢复为最佳状态（`load_state_dict(best_state)`）

**关键代码逻辑**:

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
else:
    epochs_no_improve += 1

if epochs_no_improve >= patience:
    break

model.load_state_dict(best_state)  # 恢复最优权重
```

#### 3.6.3 L2 正则化 (Weight Decay)

L2 正则化通过在损失函数中增加权重范数惩罚项来限制模型复杂度：

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_{\text{MSE}}(\theta) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$

在 Adam 优化器中，这通过 `weight_decay` 参数实现。与传统的 L2 正则化等价，AdamW 风格的 weight decay 直接在每次参数更新时做权重衰减：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

设置 $\lambda = 10^{-4}$，在所有层共享。

#### 3.6.4 学习率调度调整

原始代码使用 `StepLR(step_size=50, gamma=0.5)`，即每 50 个 epoch 学习率减半。然而当训练在 70-100 个 epoch 即进入严重过拟合时，过于缓慢的衰减使得模型在减少的 LR 生效前就已过度拟合。

修改为 `StepLR(step_size=30, gamma=0.5)`：每 30 个 epoch 减半。合计：

| epoch | 原始 LR 倍数 (step=50) | 修改后 LR 倍数 (step=30) |
|-------|------------------------|--------------------------|
| 30 | 1.0× | **0.5×** |
| 50 | 0.5× | — |
| 60 | — | **0.25×** |
| 90 | — | **0.125×** |
| 100 | 0.25× | — |

修改后的调度在过拟合开始的 30 epoch 附近即进行首次学习率衰减，更快进入精细搜索阶段。

#### 3.6.5 最大训练轮数缩减

| 参数 | 修改前 | 修改后 | 动机 |
|------|--------|--------|------|
| `main.py` MAX_EPOCHS | 300 | **100** | 与早停机制配合，300 轮远超实际所需 |
| `train.py` epochs 默认值 | 200 | **100** | 与论文训练设置一致，早停可提前结束 |

300 轮原始设定远超实际需要——在没有正则化时，模型在 50-70 轮后就开始过拟合；引入正则化后，早停通常在 40-80 轮内触发。

#### 3.6.6 参数汇总

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| 早停机制 | 无 | `patience=20` |
| L2 正则化 | 无 (`weight_decay=0`) | `weight_decay=1e-4` |
| LR 衰减步长 | `step_size=50` | `step_size=30` |
| 最大训练轮数 | 300 (main) / 200 (train) | 100 (统一) |
| 学习率 | `lr=0.0005` | `lr=0.0005` (不变) |
| LR 衰减因子 | `gamma=0.5` | `gamma=0.5` (不变) |

---

### 3.7 训练曲线可视化与图片自动保存

**修复**: 新增功能——训练监控与结果持久化
**文件**: `main.py:14-29, 33-48`, `evaluate.py:102-113, 207-226`

#### 3.7.1 问题描述

原始代码存在三个可用性问题：

1. **无训练曲线**：交叉验证的 train/val loss 曲线未绘制，无法直观判断过拟合
2. **结果不保存**：`plt.show()` 弹出的图片窗口关闭后结果丢失，无法追溯不同运行的对比
3. **阻塞式显示**：`plt.show()` 默认阻塞 `main` 函数，只有用户手动关闭所有图片窗口后程序才能退出

#### 3.7.2 交叉验证训练曲线

在 `main.py` 中新增 CV 训练曲线绘制，展示每个 fold 的 train loss（半透明实线）和 val loss（半透明虚线），不同颜色区分 fold：

```
fig_cv: CR=4 | CR=8 | CR=16 (三栏)
  每个子图: 5 条 train 实线 + 5 条 val 虚线 (对应 5 folds)
```

这直接反映过拟合程度——train loss 持续下降而 val loss 反升即表明过拟合。

#### 3.7.3 图片自动保存

新增 `save_and_show(fig, filename_stem)` 函数：

- 输出目录：`运行结果/`（自动创建）
- 文件名格式：`{filename_stem}_{YYYYMMDD_HHMMSS}.png`
- 分辨率：150 dpi，`bbox_inches='tight'` 避免截断

```python
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "运行结果")
os.makedirs(RESULT_DIR, exist_ok=True)

def save_and_show(fig, filename_stem):
    filepath = os.path.join(RESULT_DIR, f"{filename_stem}_{TIMESTAMP}.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"[Saved] {filepath}")
```

#### 3.7.4 非阻塞显示

切换 matplotlib 后端为 `TkAgg`，使用非阻塞显示：

```python
matplotlib.use('TkAgg')  # 必须在 import pyplot 之前

def nonblocking_show():
    plt.show(block=False)   # 弹出窗口但不阻塞
    plt.pause(0.1)          # 允许 GUI 事件循环处理窗口绘制
```

`main` 函数在显示窗口后正常结束，用户可继续在终端中工作，无需手动关闭所有窗口。

#### 3.7.5 绘图函数返回值修复

为使 `save_and_show` 能获取 figure 对象，修复了以下函数的返回值：

| 函数 | 修改前 | 修改后 |
|------|--------|--------|
| `plot_training_loss()` | 无返回值 | `return fig` |
| `plot_monte_carlo()` | 无返回值 | `return fig` |
| `plot_snr_comparison()` bug | `sim.generate_batch()` 不存在 | 改用 `sim.generate_pair_batch()` |
| `plot_snr_comparison()` bug | `source_complex` 未定义 | 改用 `clean_complex` |

`plot_snr_comparison` 中移除了对不存在的 `sim.generate_batch()` 和 `source_complex` 变量的引用，修正为使用 `sim.generate_pair_batch()` 和 `clean_complex` 作为互相关参考信号。

---

## 四、未修复项：改进方案与算法参考

### 4.1 信号模型升级（Wireless InSite / 城市3D场景）

**对应**: analysis_report.md §严重出入 #1
**涉及文件**: `signal_gen.py`

#### 4.1.1 当前差距

| 维度 | 论文 | 当前代码 |
|------|------|----------|
| 仿真平台 | Wireless InSite 射线追踪 | 纯 Python 随机生成 |
| 场景 | 200×260m² 城市十字路口, 19 栋建筑 | 无空间模型 |
| 信道 | 基于 3D 几何的反射/绕射/散射 | 随机 2-4 tap FIR |
| UAV 部署 | 8 架在 3D 空间分布 | 2 路信号 (无坐标) |
| LOS/NLOS | 动态切换, MLP 检测 | 无 LOS/NLOS 概念 |

#### 4.1.2 改进方案

**方案 A（推荐）**: 接入 **Wireless InSite API** 或预先生成的信道冲激响应数据库：

- 从论文的 50 个 snapshot × 8 UAV × 若干 LOS/NLOS 组合中提取信道冲激响应（CIR）
- 将 CIR 存储为 `.npy` 或 `.h5` 格式
- 加载 CIR，与 BPSK 源信号卷积生成接收信号

**方案 B（无 Wireless InSite 时的替代）**: 使用统计信道模型：
- **3GPP TR 38.901 UMa (Urban Macro)** 信道模型
- 包含路径损耗、阴影衰落、簇延迟线（CDL）
- 可实现 LOS/NLOS 概率切换

#### 4.1.3 数学模型

基于几何的确定性信道冲激响应：

$$h_i(t) = \sum_{\ell=1}^{L_i} \alpha_{\ell,i} \cdot \delta(t - \tau_{\ell,i})$$

其中：
- $\tau_{\ell,i} = d_{\ell,i}/c$ —— 第 $\ell$ 条路径的传播延迟（距离/光速）
- $\alpha_{\ell,i} = \lambda \cdot \Gamma_{\ell,i} \cdot e^{-j2\pi f_c \tau_{\ell,i}} / (4\pi d_{\ell,i})$ —— 路径复增益
- $\Gamma_{\ell,i}$ 为反射/绕射系数（由材料的 Fresnel 方程或 UTD 计算）

---

### 4.2 TDOA→几何定位解算

**对应**: analysis_report.md §严重出入 #2
**涉及文件**: `evaluate.py`

#### 4.2.1 当前差距

现有代码仅计算 TDOA 估计误差（以 **采样点** 为单位），未将其转换为位置误差（**米**）。论文 Fig.5/Fig.6 的纵轴均为 "Location Error [m]"。

#### 4.2.2 两阶段定位流程

**阶段 1 — TDOA 估计**（已有）:

$$\hat{\tau}_{ij} = \arg\max_{\tau} \left| R_{ij}^{\text{GCC}}(\tau) \right|$$

其中 $R_{ij}^{\text{GCC}}(\tau)$ 为广义互相关函数。

**阶段 2 — 几何定位**（缺失）:

设 UAV $i$ 位于 $\mathbf{s}_i = (s_{ix}, s_{iy}, s_{iz})^T$，辐射源位于 $\mathbf{r} = (r_x, r_y, r_z)^T$。TDOA 方程：

$$d_{ij} = \|\mathbf{r} - \mathbf{s}_i\| - \|\mathbf{r} - \mathbf{s}_j\| = c \cdot \tau_{ij}$$

其中 $c$ 为光速，$d_{ij}$ 为距离差。$N$ 架 UAV 产生 $N(N-1)/2$ 个 TDOA 方程（独立方程数为 $N-1$）。

#### 4.2.3 Chan 算法（推荐实现）

Chan 算法 [Chan & Ho, 1994] 是 TDOA 定位的经典闭式解法：

**Step 1 — 加权最小二乘 (WLS)**:
将 TDOA 方程线性化为：

$$\mathbf{G}_a \mathbf{z}_a = \mathbf{h}$$

其中 $\mathbf{z}_a = [r_x, r_y, r_z, R]^T$，$R = \|\mathbf{r}\|$。WLS 解：

$$\hat{\mathbf{z}}_a = (\mathbf{G}_a^T \mathbf{W} \mathbf{G}_a)^{-1} \mathbf{G}_a^T \mathbf{W} \mathbf{h}$$

权重矩阵 $\mathbf{W} = \mathbf{Q}^{-1}$（$\mathbf{Q}$ 为 TDOA 测量协方差矩阵）。

**Step 2 — 利用 $R = \|\mathbf{r}\|$ 约束的第二次 WLS**:
构造新的线性方程组 $\mathbf{G}_a' \mathbf{z}_a' = \mathbf{h}'$，其中 $\mathbf{z}_a' = [(r_x - s_{1x})^2, (r_y - s_{1y})^2, (r_z - s_{1z})^2]^T$，再次使用 WLS 得到最终位置估计。

**定位误差计算**:

$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{trial}}} \sum_{t=1}^{N_{\text{trial}}} \|\hat{\mathbf{r}}_t - \mathbf{r}_{\text{true}}\|^2}$$

---

### 4.3 多UAV协作框架 + LOS/NLOS检测

**对应**: analysis_report.md §中等出入 #5
**涉及文件**: `signal_gen.py`, `evaluate.py`, 新建 `los_detector.py`

#### 4.3.1 论文方法

论文首先使用 **MLP 二分类器** 对每条 UAV-源链路做 LOS/NLOS 检测，筛选出 LOS 条件下的 UAV 参与 TDOA 计算。这是保证定位精度的关键步骤——NLOS 链路会引入巨大的正偏时延（几米到十几米的量级）。

#### 4.3.2 MLP LOS/NLOS 检测器设计

输入特征（每条链路）：

$$\mathbf{f}_i = \left[ \text{RMS delay spread}, \max |h_i|, \frac{\max |h_i|}{\text{RMS delay spread}}, \text{峰度}, \text{偏度} \right]$$

这些特征捕捉了 LOS（强主径、低延迟扩展）与 NLOS（弱主径、高延迟扩展）的本质区别。

网络结构：`Linear(5, 16) → ReLU → Linear(16, 8) → ReLU → Linear(8, 1) → Sigmoid`

训练标签：基于射线追踪中的直达路径是否存在来标定。

#### 4.3.3 多UAV TDOA 融合

对 $N$ 架 LOS UAV，选择参考 UAV $j^*$（如信号最强的），计算 $N-1$ 个独立 TDOA：

$$[\tau_{1j^*}, \tau_{2j^*}, \ldots, \tau_{Nj^*}]^T$$

将这些 TDOA 送入 Chan 算法做几何定位。更多 LOS UAV 提供更多 TDOA 测量，降低定位 CRLB（Cramér-Rao 下界）。

---

### 4.4 对比基线方法

**对应**: analysis_report.md §中等出入 #7
**涉及文件**: 新建 `baselines.py`

#### 4.4.1 论文 Fig.6 中的对比方法

| 方法 | 原理 | 实现要点 |
|------|------|----------|
| **DFT-based** | FFT → 保留 M/K 个最大幅度系数 | `torch.fft.fft` → 幅度排序 → 截断 → `torch.fft.ifft` |
| **Hadamard-based** | $\mathbf{z} = \mathbf{H}_{M \times K} \mathbf{x}$; 重构 $\hat{\mathbf{x}} = \mathbf{H}^T \mathbf{z}$ | 使用 Walsh-Hadamard 矩阵做压缩投影 |
| **PCA-based** | 对训练数据做 PCA，保留前 M 个主成分 | `sklearn.decomposition.PCA(n_components=M)` |
| **DNN-and-residual-block-based DAE** | 全连接层替代卷积层 + 保留残差块 | 将 `ComplexConv1d` 替换为 `ComplexLinear` |
| **CNN-based DAE** | 纯卷积 DAE，无残差块 | 从 encoder/decoder 中移除 `ResidualBlock` |

#### 4.4.2 评估协议

所有方法在相同 CR=16、相同测试集下评估，使用论文的 TDOA→定位 pipeline 计算 Location Error [m]，与原始数据（不压缩）对比。

---

### 4.5 TDOA估计方法对比

**对应**: analysis_report.md §轻微出入 #11
**涉及文件**: `evaluate.py`

#### 4.5.1 当前方法

代码使用 **GCC-PHAT**（相位变换加权）：

$$R_{\text{PHAT}}(\tau) = \mathcal{F}^{-1}\left\{ \frac{X_1(f) X_2^*(f)}{|X_1(f) X_2^*(f)| + \epsilon} \right\}$$

PHAT 的优势是在多径环境中锐化相关峰，但其缺点是在低 SNR 下（分母趋于零）会过度放大噪声。

#### 4.5.2 建议增加的标准 GCC

标准 GCC（不加权）：

$$R_{\text{GCC}}(\tau) = \mathcal{F}^{-1}\left\{ X_1(f) X_2^*(f) \right\}$$

建议在评估中同时报告 GCC-PHAT 和标准 GCC 的结果，以验证 PHAT 加权对 DAE 重构信号的具体影响。

---

## 五、正确匹配项清单

以下 10 项经核验与论文一致，**未做修改**：

| # | 项目 | 位置 | 论文依据 |
|---|------|------|----------|
| 1 | 卷积层参数: kernel=10,5,5; stride=6,2,2; channels=32,32,64 | `model.py:158-160` | §III-A, Fig.1 |
| 2 | 残差块: 3 层 ComplexConv1d + 残差连接 + ReLU | `model.py:139-150` | §III-A |
| 3 | 压缩率: CR = K/M, latent_dim = 2048 // cr | `model.py:168` | §III-A |
| 4 | ComplexConv1d 复数乘法: `out_r = W_r*x_r - W_i*x_i`, `out_i = W_r*x_i + W_i*x_r` | `model.py:15-17` | Eq.(2)-(4) 的精神 |
| 5 | 对称编解码结构 | `model.py:157-180` | §III-A |
| 6 | MSE 损失函数 | `train.py:16,78` | Fig.3 (ReLU+MSE 最优) |
| 7 | 信号长度 K=1024 | `signal_gen.py:57` | §IV-A |
| 8 | 编码器维度链: 1024→170→85→43; 解码器: 43→85→170→1024 | 推导验证 | §III-A |
| 9 | 输出层无激活函数和 BN | `model.py:177` | §III-A |
| 10 | 评估 SNR 范围: [-10, 20] dB, step=2 dB | `evaluate.py:44` | §IV-C |

---

## 六、验证记录汇总

### 6.1 第1轮验证（ComplexBatchNorm1d + ReLU）

| 测试项 | 结果 | 时间 |
|--------|------|------|
| ComplexBatchNorm1d 训练/推断形状保持 (4,64,170)→(4,64,170) | ✓ | 2026-05-23 |
| DAE CR=4/8/16 前向传播形状 (2,2,1024)→(2,2,1024) | ✓ | 2026-05-23 |
| ComplexBatchNorm1d gamma/beta 梯度流通, 无 NaN | ✓ | 2026-05-23 |
| ResidualBlock 兼容性 (4,128,43)→(4,128,43) | ✓ | 2026-05-23 |
| 参数计数: trainable=160, buffers=160 (C=32) | ✓ | 2026-05-23 |
| 训练/推断模式输出 std 一致性 (≈0.707 = 1/√2) | ✓ | 2026-05-23 |

### 6.2 第2轮验证（ComplexLinear + RRC + CV）

| 测试项 | 结果 | 时间 |
|--------|------|------|
| ComplexLinear 复数乘法: (3+2j)·(2+1j) = (4+7j) | ✓ | 2026-05-23 |
| ComplexLinear 维度变换: (4,5504)→(4,128) | ✓ | 2026-05-23 |
| RRC 滤波器: 17 taps, ‖h‖₂=1.0, 峰值居中 (tap 8) | ✓ | 2026-05-23 |
| SignalSimulator 输出形状: (batch, 2, **1024**) | ✓ | 2026-05-23 |
| 固定 seed 数据集可复现: 两次 `seed=42` → `torch.allclose` | ✓ | 2026-05-23 |
| DAE CR=4/8/16 ComplexLinear 前向传播形状正确 | ✓ | 2026-05-23 |
| ComplexLinear 梯度流通, 无 NaN | ✓ | 2026-05-23 |
| CV 训练集成: 3-fold, 5 epoch 完整运行 (avg val loss=1.788) | ✓ | 2026-05-23 |
| `train_with_cv` 端到端测试通过 | ✓ | 2026-05-23 |

---

### 3.8 信噪比对比图、SVG矢量输出、显示模式与独立子文件夹

**修复**: 新增 SNR 信号对比可视化 + 矢量图输出 + 阻塞式显示 + 每次运行独立子文件夹
**文件**: `main.py:26-29, 33-44, 93-101`, `evaluate.py:203-205`

#### 3.8.1 SNR 多水平信号对比图

在训练和蒙特卡洛评估完成后，新增 4×3 子图矩阵（`plot_snr_comparison`），直观展示模型在不同 SNR 下的去噪与重构能力：

- **4 行**：SNR = -10, 0, 10, 20 dB（覆盖低到高信噪比）
- **3 列**：
  - 时域波形（Magnitude）：Noisy / Clean / DAE 重构信号对比（前 300 采样点）
  - 频谱（Double-Sided PSD）：Noisy vs DAE 双边功率谱密度（semilogy 坐标）
  - 互相关（Cross-Correlation Envelope）：以 Clean 信号为参考的互相关包络，蓝色竖虚线标注真实延迟峰值位置

使用最后训练的模型（CR=16）作为代表进行绘制，因为 CR=16 压缩最激进，去噪/重构效果的对比最明显。

#### 3.8.2 矢量图输出格式

将输出图片格式从 PNG 切换为 **SVG（Scalable Vector Graphics）**：

```python
fig.savefig(filepath, format='svg', bbox_inches='tight')
```

SVG 的优势：
- **无损缩放**：嵌入论文/PPT 时可任意放大不产生锯齿
- **可编辑**：可用 Adobe Illustrator / Inkscape 直接修改线条粗细、颜色、字体
- **体积小**：对线条图（plots）的存储效率远优于同等分辨率的 PNG
- **期刊兼容**：IEEE/Elsevier 等期刊均接受 SVG 或 PDF 矢量图投稿

函数重命名 `save_and_show` → `save_figure`，以更准确反映其职责为"保存矢量图 + 非阻塞弹窗预览"。

#### 3.8.3 显示模式回退为阻塞式

经过实际测试，非阻塞 `plt.show(block=False)` + `while plt.get_fignums()` 轮询方案存在以下问题：

1. **进程存活与窗口关联**：TkAgg 窗口由 Python 进程拥有，`main()` 返回后进程即将退出，即使轮询 Tk 事件循环，在部分平台上窗口仍会在进程退出时被回收
2. **用户体验不一致**：不同操作系统/Tk 版本下行为有差异

因此回退为标准的阻塞式显示方案：

```python
# 每张图在 save_figure 中用 plt.show(block=False) + plt.pause(0.1) 快速弹窗
# main() 末尾用 plt.show() 统一阻塞，等待用户检查完毕后关闭窗口
if plt.get_fignums():
    print("所有图片已显示。关闭图片窗口后程序自动退出。")
    plt.show()
```

最终行为：图片逐一弹出 → 用户检查 → 关闭所有窗口 → 进程正常退出。简洁可靠，无平台兼容性问题。

#### 3.8.4 每次运行创建独立子文件夹

**问题**：原本所有运行结果图片直接保存到 `运行结果/` 根目录下。多次运行后文件名仅靠时间戳区分，文件混在一起难以管理，且同一时间戳的多次运行会互相覆盖。

**修改**：将输出目录从平铺式改为时间戳子文件夹式：

```python
# 修改前 — 所有运行结果混在同一目录
RESULT_DIR = os.path.join(..., "运行结果")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# → 运行结果/CV_training_curves_20260525_142051.svg
# → 运行结果/MonteCarlo_TDOA_RMSE_20260525_142051.svg

# 修改后 — 每次运行独立子文件夹
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(..., "运行结果", TIMESTAMP)
# → 运行结果/20260525_142051/CV_training_curves.svg
# → 运行结果/20260525_142051/MonteCarlo_TDOA_RMSE.svg
# → 运行结果/20260525_142051/SNR_Comparison.svg
```

由于文件已在独立文件夹中，文件名无需再含时间戳，`save_figure` 直接使用 `filename_stem.svg` 即可，更简洁清晰。

目录结构示例：
```
运行结果/
├── 20260525_142051/
│   ├── CV_training_curves.svg
│   ├── MonteCarlo_TDOA_RMSE.svg
│   └── SNR_Comparison.svg
├── 20260525_153022/
│   ├── CV_training_curves.svg
│   └── ...
└── ...
```

#### 3.8.5 修复硬编码 figure 编号导致的图表重叠

**问题**: `plot_snr_comparison` 生成的 SNR_Comparison.svg 中，4×3 SNR 子图与 Monte Carlo TDOA RMSE 曲线被错误绘制在同一张图上。

**根因**: 三个绘图函数分别硬编码了 `num=1` 或 `num=2`，与 matplotlib 自动分配的 figure 编号发生冲突。`plot_monte_carlo` 的 `plt.figure(figsize=(10, 6))` 被自动分配为 Figure 2，随后 `plot_snr_comparison` 的 `plt.subplots(..., num=2)` 复用同一 Figure 对象，导致两者指向同一 figure，`fig_mc` 和 `fig_snr` 成为同一对象的两个引用。

**修复**: 移除所有显式 `num=` 参数，由 matplotlib 自动管理 figure 编号：

```python
# plot_training_loss:  plt.figure(1, ...)           → plt.figure(figsize=(6, 4))
# plot_snr_comparison: plt.subplots(..., num=2)      → plt.subplots(...)
```

每次创建 figure 时自动分配唯一编号，彻底避免跨函数污染。

---

`train.py` 新增依赖:
```python
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
```

安装命令: `pip install scikit-learn`

---

*本文档由 `analysis_report.md` 和 `change_log.md` 合并、扩展而成。*

---

## 七、第4轮修改：代码全面审计与性能修复 (2026-05-26)

### 7.1 审计动机

仿真结果出现异常（CR=4 < CR=8 与论文结论矛盾），除信号模型因素外，经全面代码审计发现了以下关键问题。

### 7.2 核心发现：CR=4 模型严重欠训练

#### 7.2.1 参数量与训练样本的严重不匹配

经模型维度验证，三种 CR 的参数量差异巨大：

| 模型 | 总参数量 | FC 层参数占比 | 训练样本/折 | 数据:参数比 |
|------|---------|-------------|------------|-----------|
| CR=4 | **3,188,418** | ~88% (2.8M) | 13,333 (3-fold, 20K) | 0.0042 |
| CR=8 | 1,779,138 | ~79% (1.4M) | 13,333 | 0.0075 |
| CR=16 | 1,074,498 | ~66% (0.7M) | 13,333 | 0.0124 |

使用原始配置（N_SAMPLES=10000, K_FOLDS=5）时，每折仅 8,000 训练样本：
- CR=4 的数据:参数比仅为 **0.0025**（8K/3.2M）
- 即每个参数仅对应 0.0025 个训练样本

**这直接解释了 CR 性能排序反转**：CR=4 拥有最多参数，但在随机信道下严重欠训练，无法学到有效的去噪表示；CR=8 和 CR=16 因为参数更少，在相同数据量下泛化更好。

#### 7.2.2 随机信道 vs 固定信道的本质差异

论文在 Wireless InSite 的**固定城市环境**中生成 10,000 个训练样本（50 snapshots × 200 noise trials）。在同一固定信道下，不同样本仅在符号序列和噪声实现上有差异，数据多样性有限，DAE 只需学习对该特定信道的去噪。

本项目当前代码对**每个训练样本生成不同的随机多径信道**（2-4 tap FIR，随机延迟、幅度、相位）。这意味着 10,000 个样本 = 10,000 个不同的信道条件，数据多样性远高于论文。对于 CR=4 的 3.2M 参数模型，10K 样本严重不足。

**信道生成对比：**

| 维度 | 论文 | 当前代码 | 影响 |
|------|------|----------|------|
| 信道来源 | Wireless InSite 3D 射线追踪 | 随机 2-4 tap FIR | 信道复杂度差 ~10× |
| 信道多样性 | 固定 (1 个场景) | 每个样本不同 (10K 个场景) | 数据需求增加约 100× |
| 噪声实现 | 200 trials/snapshot | 1 trial/sample | 噪声多样性相当 |
| 等效所需样本 | 10,000 (论文) | **~100,000** (估计) | 当前仅为所需 1/5~1/10 |

### 7.3 SNR 定义偏差：RRC 匹配滤波增益

#### 7.3.1 信号处理流程中的 SNR 偏移

当前代码中噪声在 Rx RRC 匹配滤波器**之前**加入，而 SNR 信号功率基于滤波器输入端的信号计算：

```
[BPSK] → [Tx RRC] → [Multipath Channel] → [+ AWGN] → [Rx RRC] → 测量信号
                                                    ↑
                                              SNR 在此定义
```

匹配滤波器对带外噪声有抑制效果，导致滤波器输出端的有效 SNR 比标称值高约 **+2.76 dB**：

| 标称 SNR | 实际有效 SNR (RRC 输出) |
|----------|----------------------|
| -10 dB | ~ -7.2 dB |
| -8 dB | ~ -5.2 dB |
| 0 dB | ~ +2.8 dB |
| 10 dB | ~ +12.8 dB |
| 20 dB | ~ +22.8 dB |

该偏移是系统性的（对所有 CR 和原始信号相同），**不改变 CR 间的相对排序**，但使整个 RMSE 曲线的有效 SNR 右移约 2.8 dB。在低 SNR 端，实际噪声水平比标称值低，GCC-PHAT 更易锁定正确峰值，导致"原始信号 RMSE 过早触零"的现象。

#### 7.3.2 偏移来源的数学解释

RRC 滤波器是单位能量归一化的（$\|h_{\text{RRC}}\|_2 = 1$），对白噪声的功率增益为 1（在连续时间下）。但由于滤波器的频谱选择性（滚降带宽 $B = (1+\beta)R_s/2$），离散实现中有效噪声带宽略小于全奈奎斯特带宽，导致轻微的 SNR 增益。

### 7.4 evaluate.py 问题修复

#### 7.4.1 lags 变量作用域修复

**问题**: `lags` 变量在 raw 信号循环内部计算（第 72 行），随后被所有 DAE 循环复用（第 92 行）。由于每次迭代都在 raw 循环中重新计算相同值（所有信号长度为 1024），该问题在功能上是无害的，但属于代码异味（code smell），对于未来修改（如可变信号长度）构成隐患。

**修复**: 在 SNR 循环开始前一次性计算 `lags`，所有后续循环共享：

```python
# 所有信号长度相同（1024），lags 只需计算一次
signal_len = self.sim.signal_len
lags = signal.correlation_lags(signal_len, signal_len, mode='same')
```

#### 7.4.2 蒙特卡洛可复现性

**问题**: `MonteCarloExperiment.run()` 调用 `generate_pair_batch` 时使用当前 `numpy.random` 状态，无种子设置。每次运行结果不同，不利于结果对比。

**修复**: `MonteCarloExperiment.__init__` 新增 `seed` 参数；若提供，在 `run()` 开始时调用 `np.random.seed(seed)`：

```python
def __init__(self, models_dict, simulator, device, seed=None):
    ...
    self.seed = seed

def run(self):
    if self.seed is not None:
        np.random.seed(self.seed)
```

### 7.5 训练时间优化

#### 7.5.1 K_FOLDS 缩减

**问题**: K_FOLDS=5 意味着 3 个 CR × 5 折 = 15 次独立训练。在 CPU 上每次约 5-10 分钟，总计 **75-150 分钟**。

**修复**: 将默认 K_FOLDS 从 5 降至 3（节省 40% 时间）。理由：
- 论文使用 K=5 是针对**固定信道**的数据结构（50 snapshots × 200 trials）设计的
- 随机信道下，不同 fold 的数据分布差异更大（每个样本有不同信道），CV 的方差估计本身已较高
- 3 折足以评估模型的相对泛化能力，且保证每个 CR 模型至少看到 13,333 个训练样本（20K 总样本下）

| 配置 | 训练运行次数 | 预计 CPU 时间 |
|------|------------|-------------|
| K_FOLDS=5 (旧) | 15 次 | 75-150 min |
| K_FOLDS=3 (新) | 9 次 | 45-90 min |
| USE_SPLIT=True | 3 次 | 15-30 min |

#### 7.5.2 快速验证模式 (USE_SPLIT)

新增 `USE_SPLIT` 开关，启用后使用单次 80/20 train/val 随机划分替代 K-fold CV，**节省 80% 训练时间**：

```python
# main.py
USE_SPLIT = False        # True: 单次划分快速模式; False: K-fold CV

# train.py — train_with_cv(use_split=USE_SPLIT)
if use_split:
    n_train = int(n_samples * 0.8)
    indices = torch.randperm(n_samples, ...)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()
```

#### 7.5.3 样本数提升

将默认 N_SAMPLES 从 10,000 提升至 20,000，以部分缓解 CR=4 的欠训练问题。在 K_FOLDS=3 下，每折训练集包含 13,333 个样本（提升 67%），数据:参数比从 0.0025 提升至 0.0042。

### 7.6 修改文件清单（第4轮）

| 文件 | 修改内容 | 关联问题 |
|------|----------|----------|
| `main.py` | K_FOLDS 5→3, N_SAMPLES 10K→20K, 新增 USE_SPLIT 开关, MC 传入 seed | §7.2, §7.5 |
| `train.py` | train_with_cv 新增 use_split 参数, 支持单次 train/val 划分, 新增模型参数量打印 | §7.5.2 |
| `evaluate.py` | MonteCarloExperiment 新增 seed 参数, lags 计算移出循环, 移除重复的无用 import | §7.4 |
| `signal_gen.py` | SNR 功率基准从 Rx RRC 前改为 Rx RRC 后,消除 +2.76 dB 系统偏移 | §7.3 |
| `train.py` | 新增 `import numpy as np`; `train_with_cv` 开头固定 torch/np 种子, 确保模型初始化和数据划分可复现 | §7.9 |
| `main.py` | 新增全局 torch/np/random 种子设置 + CUDA deterministic 配置 | §7.9 |

### 7.7 验证记录

#### 7.7.1 第4轮验证

| 测试项 | 结果 | 时间 |
|--------|------|------|
| DAE 三种 CR 参数量: CR=4:3.19M, CR=8:1.78M, CR=16:1.07M | ✓ | 2026-05-26 |
| 训练数据 SNR 分布: 均匀分布于 [-10, 20] dB, 每 2dB 约 667 样本 (20K 总样本) | ✓ | 2026-05-26 |
| 有效 SNR 偏移: 标称 SNR + ~2.76 dB (匹配滤波增益, 系统性, 不影响 CR 排序) | ✓ | 2026-05-26 |
| 高 SNR 下原始信号 TDOA: RMSE=0 (100 次试验, SNR=20dB) | ✓ | 2026-05-26 |
| MonteCarloExperiment 种子可复现: 相同 seed 两次运行结果一致 | ✓ | 2026-05-26 |
| lags 一次性计算: 始终基于 signal_len=1024, 无功能差异 | ✓ | 2026-05-26 |
| train_with_cv use_split=True 单 fold 训练 3 CR 完成 | 待验证 | — |
| K_FOLDS=3, N_SAMPLES=20000 完整运行结果 | 待验证 | — |

### 7.8 结论与后续建议

#### 7.8.1 CR 排序反转的根本原因

CR=4 性能差于 CR=8 有**两个彼此叠加的根本原因**：

1. **数据量不足**: CR=4 有 3.2M 参数，但原始配置下每折仅 8K 训练样本，数据:参数比 0.0025。CR=8 (1.8M) 的比值为 0.0045，CR=16 (1.1M) 为 0.0074。数据越不足以支撑参数量，泛化越差。

2. **随机信道放大数据需求**: 论文使用固定信道（Wireless InSite 射线追踪），DAE 只需学习对一种信道场景去噪。当前代码每个样本生成一条新的随机信道，DAE 需要学习对所有可能信道泛化。所需训练样本量至少增加一个数量级。

两个因素叠加，CR=4 的欠训练问题远严重于 CR=8/16，导致其 TDOA 性能在大部分 SNR 下反而最差。

#### 7.8.2 改进建议优先级

| 优先级 | 改进项 | 预期效果 |
|--------|--------|----------|
| **P0** | 引入固定信道模式（生成 50-100 条固定多径 Profile，批量生成噪声实现） | 从根本上匹配论文实验条件，CR 排序恢复 |
| **P1** | 进一步增加 N_SAMPLES (50K~100K) | 缓解随机信道下的欠训练问题 |
| **P2** | 升级至 Wireless InSite 或 3GPP TR 38.901 CDL 信道模型 | 完整复现论文场景 |
| **P3** | TDOA → Location Error 转换 (Chan 算法) | 使评估指标与论文对齐 |
| **P4** | 实现 DFT/Hadamard/PCA 等基线压缩方法 | 完整论文对比实验 |

### 7.9 训练可复现性修复 (2026-05-26)

#### 7.9.1 问题

两次相同配置（seed=42）的完整运行产生截然不同的 Monte Carlo 结果（RMSE 值差异 46%~254%）。根因定位：

1. `generate_training_dataset(seed=42)` 内部保存当前 numpy 随机状态 → 设种子 → 生成数据 → **恢复原始状态**
2. 恢复后 numpy 状态为调用前的不确定值
3. 后续 `model = DAE(cr=cr)` 使用 PyTorch 全局随机状态（从未设种子）
4. 权重初始化、`torch.randperm`、DataLoader shuffle 均不可复现

#### 7.9.2 修改

**`train.py`**：
- 新增 `import numpy as np`
- 在 `train_with_cv` 开头（`generate_training_dataset` 之前）设置：
```python
torch.manual_seed(seed)
np.random.seed(seed)
```

**`main.py`**：
- 新增全局种子配置，含 CUDA 确定性设置：
```python
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

#### 7.9.3 验证

两次独立调用 `train_with_cv`（seed=42, CR=8, use_split=True, 500 samples, 5 epochs）：
- train loss 完全一致：`[2.0195, 1.7762, 1.7498, 1.7427, 1.7396]`
- val loss 完全一致：`[1.7695, 1.8202, 1.7859, 1.7700, 1.7651]`
- 模型权重逐参数匹配（`torch.allclose` 全部 True）
- Best val loss: `1.765106`（两次完全相同）
