import os
import matplotlib.pyplot as plt
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

# ===================== 核心配置：解决中文显示 + 关闭弹窗 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 关键修改：设置matplotlib后端为Agg，完全关闭弹窗
plt.switch_backend('Agg')

# ===================== 1. 数据加载 + 类别定义（无改动） =====================
classes = ('飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 注意：如果首次运行，需要把download=False改为download=True，下载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# ===================== 2. 模型定义（仅这里加了Dropout） =====================
class PassCNN(nn.Module):
    def __init__(self):
        super(PassCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        # ========== 新增：Dropout层 ==========
        self.dropout1 = nn.Dropout(0.25)  # 卷积层后用，丢弃25%神经元
        self.dropout2 = nn.Dropout(0.5)   # 全连接层后用，丢弃50%神经元

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout1(x)  # ========== 新增：卷积1后加Dropout ==========
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout1(x)  # ========== 新增：卷积2后加Dropout ==========
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)  # ========== 新增：全连接1后加Dropout ==========
        x = self.fc2(x)
        return x

# ===================== 3. 训练过程（保留训练/测试准确率输出） =====================
net = PassCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 无改动

loss_history = []
train_acc_history = []
acc_history = []
# 修改点1：打印提示改为10轮
print("开始训练（10轮）...")

# 修改点2：训练轮数从5改为10
for epoch in range(10):
    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(trainloader)
    loss_history.append(avg_loss)
    train_epoch_acc = 100 * train_correct / train_total
    train_acc_history.append(train_epoch_acc)
    
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_acc = 100 * correct / total
    acc_history.append(epoch_acc)
    
    print(f"第 {epoch+1} 轮 | 平均Loss: {avg_loss:.3f} | 训练准确率: {train_epoch_acc:.2f}% | 测试准确率: {epoch_acc:.2f}%")

# ===================== 4. 最终评估 + 可视化（无改动） =====================
net.eval()
correct = 0
total = 0
correct_imgs = []
correct_labels = []
correct_preds = []
wrong_imgs = []
wrong_labels = []
wrong_preds = []

with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        
        for img, label, pred in zip(images, labels, predicted):
            img = img.cpu() * torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
            img = torch.clamp(img, 0, 1)  
            
            if label == pred and len(correct_imgs) < 8:
                correct_imgs.append(img)
                correct_labels.append(label.item())
                correct_preds.append(pred.item())
            elif label != pred and len(wrong_imgs) < 8:
                wrong_imgs.append(img)
                wrong_labels.append(label.item())
                wrong_preds.append(pred.item())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
print(f" 最终Test Accuracy: {final_accuracy:.2f}%")
print("已达到及格线（≥50%）！" if final_accuracy >= 50 else "❌ 未达标，请检查代码~")

# ===================== 5. 绘制Loss+准确率曲线（删除plt.show()） =====================
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(range(1, 11), loss_history, marker='o', color='blue', label='平均Loss')
ax1.set_xlabel('训练轮数')
ax1.set_ylabel('Loss值', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
# 修改点3：x轴刻度从1-5改为1-10
ax1.set_xticks(range(1, 11))
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(range(1, 11), train_acc_history, marker='^', color='green', label='训练准确率')
ax2.plot(range(1, 11), acc_history, marker='s', color='red', label='测试准确率')
ax2.set_ylabel('准确率（%）', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('10轮训练Loss与准确率变化曲线')  # 可选：修改标题为10轮
plt.savefig('训练日志曲线.png', dpi=300, bbox_inches='tight')
# 关键修改：删除plt.show()，关闭弹窗

# ===================== 6. 绘制分类正确/错误图片可视化（删除plt.show()） =====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

correct_grid = make_grid(correct_imgs, nrow=4, padding=2)
ax1.imshow(np.transpose(correct_grid, (1, 2, 0)))
ax1.set_title('分类正确的图片示例（8张）')
ax1.axis('off')
correct_text = "真实标签/预测标签：\n"
for i in range(8):
    correct_text += f"第{i+1}张：{classes[correct_labels[i]]}/{classes[correct_preds[i]]}  "
    if (i+1) % 4 == 0:
        correct_text += "\n"
ax1.text(0.5, -0.1, correct_text, ha='center', va='top', transform=ax1.transAxes, fontsize=10)

wrong_grid = make_grid(wrong_imgs, nrow=4, padding=2)
ax2.imshow(np.transpose(wrong_grid, (1, 2, 0)))
ax2.set_title('分类错误的图片示例（8张）')
ax2.axis('off')
wrong_text = "真实标签/预测标签：\n"
for i in range(8):
    wrong_text += f"第{i+1}张：{classes[wrong_labels[i]]}/{classes[wrong_preds[i]]}  "
    if (i+1) % 4 == 0:
        wrong_text += "\n"
ax2.text(0.5, -0.1, wrong_text, ha='center', va='top', transform=ax2.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig('分类结果可视化.png', dpi=300, bbox_inches='tight')
# 关键修改：删除plt.show()，关闭弹窗

print("\n📝 核心文件已生成：")
print("1. 训练日志曲线.png → 训练Loss+训练/测试准确率曲线")
print("2. 分类结果可视化.png → 分类正确/错误图片示例")