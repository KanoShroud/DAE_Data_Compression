# main.py

import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from train import train_model
from evaluate import MonteCarloExperiment, plot_monte_carlo
from signal_gen import SignalSimulator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
CR_LIST = [4, 8, 16] # 目标验证的压缩率列表

models_dict = {}

# 1. 实例化一个公共的 Simulator，供后续评估使用
sim = SignalSimulator()

# 2. 串行流式训练各个压缩率下的网络
for cr in CR_LIST:
    print(f"\n" + "="*40)
    # 直接调用流式训练，不需要传入 dataset
    model, _ = train_model(DEVICE, epochs=200, cr=cr, batch_size=64, steps_per_epoch=100)
    models_dict[cr] = model

# 3. 统一蒙特卡洛评估
print("\n" + "="*40)
exp = MonteCarloExperiment(models_dict, sim, DEVICE)
mc_results = exp.run()

# 4. 绘图展示
plot_monte_carlo(mc_results)
plt.show()

