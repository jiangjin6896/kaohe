# ===================== 环境配置区 =====================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决OpenMP冲突
import matplotlib
matplotlib.use('Agg')  # 强制保存图像（避免弹窗问题，100%可用）
import matplotlib.pyplot as plt
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

import csv
import torch
import torch.nn as nn
import numpy as np

# ===================== 1. 数据加载 =====================
x_list = []
y_list = []

try:
    with open('task2.csv', 'r', encoding='utf-8') as f:
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

# 转换为张量并调整形状（适配模型输入）
x_tensor = torch.tensor(x_list, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y_list, dtype=torch.float32).reshape(-1, 1)
print(f"数据加载完成！共{len(x_tensor)}个样本")

# ===================== 2. MLP模型定义 =====================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1, 32)   # 输入层→隐藏层1
        self.fc2 = nn.Linear(32, 64)  # 隐藏层1→隐藏层2
        self.fc3 = nn.Linear(64, 1)   # 隐藏层2→输出层
        self.relu = nn.ReLU()         # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        return output

# ===================== 3. 模型训练（含关键轮次结果保存） =====================
model = MLP()
criterion = nn.MSELoss()  # 损失函数：均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
epochs = 1000  # 训练轮数调整为1000
loss_history = []  # 记录所有轮次损失
fit_results = {}   # 保存关键轮次（10/100/1000）的拟合结果

print("\n开始训练模型（共1000轮）...")
for epoch in range(epochs):
    model.train()  # 训练模式
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    loss_history.append(loss.item())  # 记录损失
    
    # 反向传播+参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每50轮打印进度
    if (epoch + 1) % 50 == 0:
        print(f"训练轮数 [{epoch+1}/{epochs}]，当前损失：{loss.item():.6f}")
    
    # 关键轮次：保存拟合结果（10/100/1000轮）
    if (epoch + 1) in [10, 100, 1000]:
        model.eval()  # 评估模式（禁用训练特有层）
        with torch.no_grad():  # 禁用梯度计算
            y_fit = model(x_tensor).numpy()  # 转换为numpy用于绘图
        fit_results[epoch + 1] = y_fit  # 保存该轮次的拟合结果
        model.train()  # 切回训练模式

# ===================== 4. 结果可视化（3张图：拟合结果+损失曲线+不同Epoch对比） =====================
model.eval()
with torch.no_grad():
    final_y_fit = model(x_tensor).numpy()  # 最终（1000轮）拟合结果

# 排序索引（让曲线平滑）
sorted_indices = np.argsort(x_list)
x_sorted = np.array(x_list)[sorted_indices]

# ---- 图1：最终（1000轮）拟合结果图 ----
plt.figure(figsize=(8, 5))
plt.scatter(x_list, y_list, color="blue", label="原始数据", alpha=0.6)
plt.plot(x_sorted, final_y_fit[sorted_indices], color="red", linewidth=2, label="MLP拟合曲线（1000轮）")
plt.xlabel("x")
plt.ylabel("y")
plt.title("MLP神经网络函数拟合结果（最终）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("最终拟合结果.png", dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ 最终拟合结果图已保存：最终拟合结果.png")

# ---- 图2：训练损失变化曲线 ----
plt.figure(figsize=(8, 3))
plt.plot(loss_history, color="green", linewidth=1.5)
plt.xlabel("训练轮数")
plt.ylabel("均方误差损失")
plt.title("训练损失变化曲线（1000轮）")
plt.grid(True, alpha=0.3)
plt.savefig("损失变化曲线.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 损失变化曲线已保存：损失变化曲线.png")

# ---- 图3：不同Epoch（10/100/1000）拟合曲线对比 ----
plt.figure(figsize=(10, 6))
# 绘制原始数据散点（灰色打底，突出曲线）
plt.scatter(x_list, y_list, color="gray", alpha=0.5, label="原始数据")
# 绘制不同轮次的拟合曲线
colors = ["orange", "red", "green"]  # 10轮=橙色，100轮=红色，1000轮=绿色
labels = ["Epoch=10", "Epoch=100", "Epoch=1000"]
for i, epoch in enumerate([10, 100, 1000]):
    y_fit = fit_results[epoch]
    plt.plot(x_sorted, y_fit[sorted_indices], color=colors[i], linewidth=2, label=labels[i])
# 图表配置
plt.xlabel("x")
plt.ylabel("y")
plt.title("不同训练轮次的拟合曲线对比（10/100/1000轮）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("不同Epoch拟合对比.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ 不同Epoch拟合对比图已保存：不同Epoch拟合对比.png")

# 打印文件保存路径，方便查找
print(f"\n所有图像已保存到：{os.getcwd()}")
print("生成的文件：")
print("  1. 最终拟合结果.png → 1000轮的拟合效果")
print("  2. 损失变化曲线.png → 1000轮的损失变化")
print("  3. 不同Epoch拟合对比.png → 10/100/1000轮拟合曲线对比")