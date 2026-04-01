# main.py

import torch
import matplotlib.pyplot as plt

# 导入自定义模块
from train import train_model
from evaluate import MonteCarloExperiment, plot_training_loss, plot_snr_comparison, plot_monte_carlo

# 1. 配置参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100

# 2. 模型训练 (调用 train.py)
# 返回训练好的模型、信号模拟器(含配置参数)、Loss历史
model, sim, loss_history = train_model(device=DEVICE, epochs=EPOCHS)

# 3. 准备绘图 - Figure 1: Loss 曲线
plot_training_loss(loss_history)

# 4. 准备绘图 - Figure 2: 特定 SNR 下的波形对比
# 展示 -10, 0, 10, 20 dB 下的效果
plot_snr_comparison(model, sim, DEVICE, snr_list=[-10, 0, 10, 20])

# 5. 执行蒙特卡洛实验 (调用 evaluate.py)
# 扫描 -10dB 到 20dB
exp = MonteCarloExperiment(model, sim, DEVICE)
mc_results = exp.run()

# 6. 准备绘图 - Figure 3: 蒙特卡洛统计结果
plot_monte_carlo(mc_results)

# 7. 统一显示所有图片
print("All tasks completed. Displaying plots...")
plt.show()


