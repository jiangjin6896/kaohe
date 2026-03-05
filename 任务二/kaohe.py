# ===================== 环境配置区 =====================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP冲突
import matplotlib
matplotlib.use('Agg')  # 强制保存图像（避免弹窗问题）
import matplotlib.pyplot as plt
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

import csv
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split  # 用于划分数据集

# ===================== 1. 数据加载 & 划分训练/验证集 =====================
x_list = []
y_list = []

try:
    # 绝对路径指向task2.csv
    with open(r'C:\Users\jiangjin\Desktop\zuoye\任务二\task2.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row_idx, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"第{row_idx}行数据格式错误，需包含x和y两列，当前行：{row}")
            try:
                x = float(row[0])
                y = float(row[1])
                x_list.append(x)
                y_list.append(y)
            except ValueError:
                raise ValueError(f"第{row_idx}行数据不是数字，当前行：{row}")
except FileNotFoundError:
    print("错误：未找到task2.csv文件，请检查文件路径是否正确！")
    exit()
except Exception as e:
    print(f"数据读取失败：{e}")
    exit()

# 转换为张量并调整形状
x_tensor = torch.tensor(x_list, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y_list, dtype=torch.float32).reshape(-1, 1)
print(f"数据加载完成！共{len(x_tensor)}个样本")

# 划分训练集和验证集（8:2）
x_train, x_val, y_train, y_val = train_test_split(
    x_tensor, y_tensor, test_size=0.2, random_state=42
)
print(f"训练集样本数：{len(x_train)}，验证集样本数：{len(x_val)}")

# ===================== 2. MLP模型定义（仅新增Dropout层） =====================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 32)        # 输入层→隐藏层1（保持原参数）
        self.fc2 = nn.Linear(32, 64)       # 隐藏层1→隐藏层2（保持原参数）
        self.fc3 = nn.Linear(64, 1)        # 隐藏层2→输出层（保持原参数）
        self.relu = nn.ReLU()              # 激活函数（保持原参数）
        # 仅新增：Dropout层（核心改动，抑制过拟合）
        self.dropout = nn.Dropout(p=0.5)   # 随机丢弃50%神经元，经典参数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 隐藏层1后加Dropout
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # 隐藏层2后加Dropout
        output = self.fc3(x)
        return output

# ===================== 3. 模型训练（其余逻辑完全不变） =====================
model = MLP()
criterion = nn.MSELoss()  # 损失函数：均方误差（保持原参数）
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器（保持原学习率0.01）
epochs = 1000  # 训练轮数（保持原参数）
train_loss_history = []  # 记录训练集损失
val_loss_history = []    # 记录验证集损失
fit_results = {}         # 保存关键轮次拟合结果

print("\n开始训练模型（仅加Dropout正则化，共1000轮）...")
for epoch in range(epochs):
    model.train()  # 训练模式：Dropout生效
    # --- 训练步骤 ---
    y_pred_train = model(x_train)
    loss_train = criterion(y_pred_train, y_train)
    train_loss_history.append(loss_train.item())
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    # --- 验证步骤 ---
    model.eval()   # 验证模式：Dropout自动关闭
    with torch.no_grad():
        y_pred_val = model(x_val)
        loss_val = criterion(y_pred_val, y_val)
        val_loss_history.append(loss_val.item())
    
    # 每50轮打印进度（保持原逻辑）
    if (epoch + 1) % 50 == 0:
        print(f"训练轮数 [{epoch+1}/{epochs}]，训练损失：{loss_train.item():.6f}，验证损失：{loss_val.item():.6f}")
    
    # 关键轮次：保存拟合结果（保持原逻辑）
    if (epoch + 1) in [10, 100, 1000]:
        with torch.no_grad():
            y_fit = model(x_tensor).numpy()
        fit_results[epoch + 1] = y_fit
        model.train()

# ===================== 4. 结果可视化（保持原逻辑） =====================
model.eval()
with torch.no_grad():
    final_y_fit = model(x_tensor).numpy()  # 最终拟合结果

# 排序索引（让曲线平滑）
sorted_indices = np.argsort(x_list)
x_sorted = np.array(x_list)[sorted_indices]

# ---- 图1：最终拟合结果图 ----
plt.figure(figsize=(8, 5))
plt.scatter(x_list, y_list, color="blue", label="原始数据", alpha=0.6)
plt.plot(x_sorted, final_y_fit[sorted_indices], color="red", linewidth=2, label="MLP拟合曲线（含Dropout）")
plt.xlabel("x")
plt.ylabel("y")
plt.title("MLP神经网络函数拟合结果（最终，含Dropout正则化）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r"C:\Users\jiangjin\Desktop\zuoye\任务二\最终拟合结果.png", dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ 最终拟合结果图已保存")

# ---- 图2：训练/验证损失对比曲线（过拟合判定） ----
plt.figure(figsize=(8, 3))
plt.plot(train_loss_history, color="blue", linewidth=1.5, label="训练集损失")
plt.plot(val_loss_history, color="orange", linewidth=1.5, label="验证集损失")
plt.xlabel("训练轮数")
plt.ylabel("均方误差损失")
plt.title("训练集 vs 验证集 损失变化曲线（含Dropout正则化）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r"C:\Users\jiangjin\Desktop\zuoye\任务二\训练验证损失对比.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 训练验证损失对比图已保存（用于过拟合判定）")

# ---- 图3：不同Epoch拟合曲线对比 ----
plt.figure(figsize=(10, 6))
plt.scatter(x_list, y_list, color="gray", alpha=0.5, label="原始数据")
colors = ["orange", "red", "green"]
labels = ["Epoch=10", "Epoch=100", "Epoch=1000"]
for i, epoch in enumerate([10, 100, 1000]):
    y_fit = fit_results[epoch]
    plt.plot(x_sorted, y_fit[sorted_indices], color=colors[i], linewidth=2, label=labels[i])
plt.xlabel("x")
plt.ylabel("y")
plt.title("不同训练轮次的拟合曲线对比（含Dropout正则化）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r"C:\Users\jiangjin\Desktop\zuoye\任务二\不同Epoch拟合对比.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 不同Epoch拟合对比图已保存")

# 打印文件保存路径
print(f"\n所有图像已保存到：C:\\Users\\jiangjin\\Desktop\\zuoye\\任务二")
print("生成的文件：")
print("  1. 最终拟合结果.png → 1000轮拟合效果（含Dropout）")
print("  2. 训练验证损失对比.png → 损失对比，判定过拟合是否改善")
print("  3. 不同Epoch拟合对比.png → 不同轮次拟合曲线")