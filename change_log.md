# 项目修改记录 (Changelog)

**项目**: DAE Data Compression — 论文复现
**论文**: Chen et al., "Collaborative Localization Using Multiple UAVs: A Deep Denoising Autoencoder-Based Data Compression Approach", IEEE TVT 2025.
**基准文档**: [analysis_report.md](analysis_report.md)

---

## 变更总览

| 修改轮次 | 涉及文件 | 修复的问题（对照 analysis_report.md） | 严重等级 |
|----------|---------|---------------------------------------|----------|
| 第1轮 | `model.py` | 严重出入 #3: BN → ComplexBN; 中等出入 #4: LeakyReLU → ReLU | 🔴🟡 |
| 第2轮 | `model.py` | 中等出入 #6: nn.Linear → ComplexLinear | 🟡 |
| 第2轮 | `signal_gen.py` | 轻微出入 #10: Butterworth → RRC; 轻微出入 #9: SNR 范围修正 | 🟢 |
| 第2轮 | `train.py` | 中等出入 #8: 流式训练 → k-fold CV + 固定数据集 | 🟡 |
| 第2轮 | `main.py` | 中等出入 #8: 改用 CV 训练流程 + 可视化 | 🟡 |

**已修复**: 4/11 项问题（3 项严重/中等等级 + 1 项轻微等级）
**待修复**: 7/11 项（详见 analysis_report.md 第五章节改进方案）

---

## 第1轮修改：网络结构对齐

**日期**: 2026-05-23
**范围**: `model.py`

### 修改 #1.1 — ComplexBatchNorm1d 实现（修复 analysis_report.md §严重出入 #3）

**问题**: 代码使用标准 `nn.BatchNorm1d` 处理复数特征，将实部/虚部视为独立实数通道。论文明确使用 Complex Batch Normalization，需对2×2实虚协方差矩阵进行白化。

**变更前** (共8处):
```python
# ResidualBlock & DAE encoder/decoder
nn.BatchNorm1d(tensor_channels)  # e.g., BatchNorm1d(64), BatchNorm1d(128)
```

**变更后**:
```python
# 新增 ComplexBatchNorm1d 类（model.py:65-134）
# 实/虚分离 → 计算 Vrr/Vii/Vri 协方差 → V^{-1/2} 白化 → γ 缩放 + β 平移
ComplexBatchNorm1d(complex_channels)  # e.g., ComplexBatchNorm1d(32), ComplexBatchNorm1d(64)
```

**核心算法**:
1. 将输入 `(B, 2C, L)` 拆分为实部 `x_r(B, C, L)` 和虚部 `x_i(B, C, L)`
2. 计算每通道协方差: `Vrr = E[(x_r-μ_r)²]`, `Vii`, `Vri = E[(x_r-μ_r)(x_i-μ_i)]`
3. 2×2 矩阵逆平方根闭式解: `W = V^{-1/2}`（s=√det(V), t=√(tr(V)+2s)）
4. 白化: `[x̂_r, x̂_i]^T = W·[x_r-μ_r, x_i-μ_i]^T`
5. 可学习 γ(2×2, 初始化为 I/√2) 缩放 + β 偏置

**API 变更**: `ComplexBatchNorm1d(complex_channels)` — 参数为**复数通道数**，与 `ComplexConv1d` 一致（如 32 而非 64）。

**替换清单（8处）**:

| 位置 | 原调用 | 新调用 |
|------|--------|--------|
| DAE 编码器 L1 | `BatchNorm1d(64)` | `ComplexBatchNorm1d(32)` |
| DAE 编码器 L2 | `BatchNorm1d(64)` | `ComplexBatchNorm1d(32)` |
| DAE 编码器 L3 | `BatchNorm1d(128)` | `ComplexBatchNorm1d(64)` |
| ResidualBlock ×3 | `BatchNorm1d(tensor_channels)` | `ComplexBatchNorm1d(complex_channels)` |
| DAE 解码器 L1 | `BatchNorm1d(64)` | `ComplexBatchNorm1d(32)` |
| DAE 解码器 L2 | `BatchNorm1d(64)` | `ComplexBatchNorm1d(32)` |

**验证结果**: 训练/推断模式输出形状保持，gamma/beta 梯度正常流通，无 NaN。

---

### 修改 #1.2 — 激活函数 ReLU 替换 LeakyReLU（修复 analysis_report.md §中等出入 #4）

**问题**: 代码使用 `nn.LeakyReLU(0.2)`，与论文 Fig.3 消融实验结论（ReLU+MSE 为最优组合）不一致。

**变更前**:
```python
nn.LeakyReLU(0.2)   # ResidualBlock 内 ×3, self.act ×1, DAE enc ×3, DAE dec ×2
```

**变更后**:
```python
nn.ReLU()           # 全部 9 处替换
```

**依据**: 论文 Fig.3 在 CR=8 条件下对比了 ReLU/Sigmoid/Tanh + MSE/MAE/Huber/Cos 共 6 种组合，ReLU+MSE 取得最低 Location Error。

**验证结果**: 前向/反向传播无异常，梯度流通正常。

---

## 第2轮修改：复数全连接层 + 脉冲成形 + 交叉验证

**日期**: 2026-05-23
**范围**: `model.py`, `signal_gen.py`, `train.py`, `main.py`

### 修改 #2.1 — ComplexLinear 复数全连接层（修复 analysis_report.md §中等出入 #6）

**问题**: 论文提及"fully complex-valued connected layer"，代码使用标准 `nn.Linear`（实矩阵乘法）退化处理。

**变更前** (`model.py:66-67`):
```python
self.fc_enc = nn.Linear(128 * 43, self.latent_dim)       # 5504 → latent_dim
self.fc_dec = nn.Linear(self.latent_dim, 128 * 43)       # latent_dim → 5504
```

**变更后** (`model.py:176-179`):
```python
# 复数全连接层：输入 2752 复数特征，输出 latent_dim/2 复数特征
self.fc_enc = ComplexLinear(128 * 43 // 2, self.latent_dim // 2)   # 2752 → latent_dim/2
self.fc_dec = ComplexLinear(self.latent_dim // 2, 128 * 43 // 2)   # latent_dim/2 → 2752
```

**ComplexLinear 实现** (`model.py:39-56`):
```python
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        self.fc_r = nn.Linear(in_features, out_features)  # 实部权重
        self.fc_i = nn.Linear(in_features, out_features)  # 虚部权重

    def forward(self, x):
        # z = x_r + j·x_i,  W = W_r + j·W_i
        # out = W·z = (W_r·x_r - W_i·x_i) + j(W_r·x_i + W_i·x_r)
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        return torch.cat([out_r, out_i], dim=1)
```

**维度兼容性**（以 CR=16 为例）:
- encoder 输出 flatten: 128×43=5504 实数 = 2752 复数特征
- `fc_enc`: ComplexLinear(2752, 64) → 输出 (batch, 128) = latent_dim ✓
- `fc_dec`: ComplexLinear(64, 2752) → 输出 (batch, 5504) → reshape (batch, 128, 43) ✓

**验证结果**: 复数乘法正确性测试 `(3+2j)×(2+1j)=(4+7j)` 通过；CR=4/8/16 三种维度前向传播形状正确。

---

### 修改 #2.2 — RRC 脉冲成形替代 Butterworth 滤波器（修复 analysis_report.md §轻微出入 #10）

**问题**: 代码使用 4 阶 Butterworth 滤波器（`signal.butter(4, 0.5)`）进行脉冲成形，不满足 Nyquist ISI 准则。标准 BPSK 通信应使用 RRC（根升余弦）滤波器。

**变更前** (`signal_gen.py:17`):
```python
self.b_lpf, self.a_lpf = signal.butter(4, 0.5, btype='low')
# 使用 filtfilt 施加 IIR 零相位滤波
signal.filtfilt(self.b_lpf, self.a_lpf, base_signal, axis=1)
```

**变更后** (`signal_gen.py:7-53, 57-68`):
```python
def rrc_filter(beta, span, sps):
    """标准 RRC 公式实现（含 t=0, t=±1/(4β) 边界处理）"""
    # h(t) = (sin(πt(1-β)) + 4βt cos(πt(1+β))) / (πt(1-(4βt)²))
    # 能量归一化: h /= √(Σ h²)
    ...

class SignalSimulator:
    def __init__(self, signal_len=1024, beta=0.35, rrc_span=8):
        self.rrc = rrc_filter(beta, span, sps)  # 17-tap FIR 滤波器

    def _apply_rrc(self, x, axis=-1):
        return signal.convolve(x, self.rrc, mode='same')  # FIR 匹配滤波
```

**滤波器参数**:
| 参数 | 取值 | 说明 |
|------|------|------|
| beta (滚降系数) | 0.35 | 典型取值，平衡带宽与 ISI |
| span (跨度) | 8 symbols | 滤波器覆盖 ±4 个符号周期 |
| sps (每符号采样) | 2 | 与 `samples_per_symbol` 一致 |
| 滤波器长度 | 17 taps | `span × sps + 1` |
| 归一化 | 能量 = 1.0 | `h /= √(Σ h²)` |

**与链路集成**: 发射端脉冲成形 + 接收端匹配滤波均使用同一 RRC 滤波器。因果 FIR 滤波（`convolve mode='same'`），无零相位 `filtfilt` 造成的非因果效应。

**同步修正**: 训练 SNR 范围从 `[-15, 15] dB` 改为 `[-10, 20] dB`，与论文测试范围一致（analysis_report.md §轻微出入 #9）。

**验证结果**: RRC 滤波器 17 taps，能量 = 1.0，冲击响应峰值居中对齐；SignalSimulator 输出形状 (batch, 2, 1024) 保持正确。

---

### 修改 #2.3 — K-Fold 交叉验证 + 固定数据集（修复 analysis_report.md §中等出入 #8）

**问题**: 原训练流程使用在线流式数据生成（每 step 实时生成新数据），无训练/测试分离，无交叉验证，结果不可复现。论文使用固定 10,000 组数据集 + 5-fold CV。

**变更前** (`train.py`):
```python
def train_model(device, epochs, cr, ...):
    sim = SignalSimulator()
    for ep in range(epochs):
        for step in range(steps_per_epoch):
            X1_noisy, ... = sim.generate_pair_batch(batch_size, snr_db=None)
            # 直接训练，无验证
```

**变更后** (新增组件):

**a) 固定数据集生成** (`signal_gen.py:165-190`):
```python
def generate_training_dataset(self, n_samples, seed=42):
    """固定种子生成可复现数据集，分批生成避免 OOM"""
    np.random.seed(seed)
    # 按 batch_cap=500 分批调用 generate_pair_batch
    # 返回 X_noisy (N,2,1024), X_clean (N,2,1024)
```

**b) 单 fold 训练函数** (`train.py:12-53`):
```python
def train_one_fold(device, model, train_loader, val_loader, epochs, lr, fold_idx):
    """标准训练/验证循环，每个 epoch 在验证集评估，保留最佳模型"""
```

**c) K-Fold CV 主函数** (`train.py:57-108`):
```python
def train_with_cv(device, cr, k=5, n_samples=10000, epochs=200, ...):
    """
    1. 生成固定数据集
    2. sklearn KFold(n_splits=k, shuffle=True) 划分
    3. 逐 fold 训练，记录 train/val loss
    4. 返回验证 loss 最低的模型
    """
```

**d) main.py 调用更新**:
```python
model, cv_results = train_with_cv(
    DEVICE, cr=cr, k=5, n_samples=10000, epochs=200, ...
)
# 自动绘制各 fold 的 train/val loss 曲线
```

**新增依赖**: `scikit-learn`（KFold 划分）

**训练参数对齐**:
| 参数 | 修改前 | 修改后 | 论文 |
|------|--------|--------|------|
| 数据集规模 | 无固定集 | 10,000 | 10,000 |
| 交叉验证 | 无 | 5-fold (KFold shuffle) | 5-fold |
| 训练 SNR | [-15, 15]dB | [-10, 20]dB | [-10, 20]dB |
| 随机可复现性 | 不可复现 | seed=42 固定 | — |

**验证结果**: 相同 seed 两次生成数据集完全一致（`torch.allclose`）；3-fold + 5 epoch 小规模 CV 完整运行通过。

---

## 不变部分（analysis_report.md §四 确认正确）

以下 10 项经核验与论文一致，**未做修改**：

1. 卷积层参数 (kernel/stride/channels: 10,5,5 / 6,2,2 / 32,32,64)
2. 残差块结构 (3层 Conv + 残差连接)
3. 压缩率定义 (CR = 2048/latent_dim)
4. ComplexConv1d / ComplexConvTranspose1d 复数乘法逻辑
5. 对称编解码结构
6. MSE 损失函数
7. 信号长度 K=1024
8. 编码器维度收缩链 1024→170→85→43
9. 输出层无激活/BN
10. 评估 SNR 范围 [-10, 20]dB, step=2dB

---

## 验证记录

### 第1轮验证（ComplexBatchNorm1d + ReLU）

| 测试项 | 结果 |
|--------|------|
| ComplexBatchNorm1d 训练/推断形状保持 (4,64,170)→(4,64,170) | ✓ |
| DAE CR=4/8/16 前向传播形状 (2,2,1024)→(2,2,1024) | ✓ |
| ComplexBatchNorm1d gamma/beta 梯度流通, 无 NaN | ✓ |
| ResidualBlock 兼容性 (4,128,43)→(4,128,43) | ✓ |
| 参数计数: trainable=160, buffers=160 | ✓ |
| 训练/推断模式输出 std 一致性 (≈0.707) | ✓ |

### 第2轮验证（ComplexLinear + RRC + CV）

| 测试项 | 结果 |
|--------|------|
| ComplexLinear 复数乘法: (3+2j)×(2+1j)=(4+7j) | ✓ |
| ComplexLinear 维度变换: (4,5504)→(4,128) | ✓ |
| RRC 滤波器: 17 taps, 能量=1.0, 峰值居中对齐 | ✓ |
| SignalSimulator 输出: (batch, 2, **1024**) | ✓ |
| 固定 seed 数据集可复现性 | ✓ |
| DAE CR=4/8/16 ComplexLinear 前向传播 | ✓ |
| ComplexLinear 梯度流通, 无 NaN | ✓ |
| CV 训练集成: 3-fold, 5 epoch 完整运行 | ✓ |
| `train_with_cv` 端到端测试 | ✓ |

---

## 待修复问题（来自 analysis_report.md 改进方案）

按优先级排序：

| 优先级 | 问题编号 | 问题描述 | 涉及文件 |
|--------|---------|---------|----------|
| 🔴 第二优先级 | 严重出入 #1 | 信号模型过于简化 — 需接入 Wireless InSite 或构建城市 3D 场景 | `signal_gen.py` |
| 🔴 第二优先级 | 严重出入 #2 | 评估指标不匹配 — 需实现 TDOA→几何定位→Location Error(m) | `evaluate.py` |
| 🔴 第二优先级 | 中等出入 #5 | 缺少多 UAV (8架) 协作框架 + LOS/NLOS 检测 | `signal_gen.py`, `evaluate.py` |
| 🟡 第二优先级 | 中等出入 #7 | 缺少 5 种对比基线方法 | 新建 `baselines.py` |
| 🟢 第三优先级 | 轻微出入 #11 | TDOA 方法 GCC-PHAT → 可对比标准 GCC | `evaluate.py` |

---

*最后更新: 2026-05-23*
