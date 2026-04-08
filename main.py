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

# 1. 统一生成全局数据集 (保证对比的绝对公平性)
print("Generating shared dataset for all models...")
sim = SignalSimulator()
X, Y, _, _ = sim.generate_batch(5000)
dataset = TensorDataset(X, Y)

models_dict = {}

# 2. 串行训练不同 CR 的网络
for cr in CR_LIST:
    print(f"\n" + "="*40)
    model, _ = train_model(DEVICE, dataset, epochs=EPOCHS, cr=cr)
    models_dict[cr] = model

# 3. 统一蒙特卡洛评估
print("\n" + "="*40)
exp = MonteCarloExperiment(models_dict, sim, DEVICE)
mc_results = exp.run()

# 4. 绘图展示
plot_monte_carlo(mc_results)
plt.show()

