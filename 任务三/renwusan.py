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

# ===================== 核心配置：解决中文显示 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据加载 + 类别定义（为可视化做准备） =====================
classes = ('飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# ===================== 2. 模型定义 =====================
class PassCNN(nn.Module):
    def __init__(self):
        super(PassCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================== 3. 训练过程 =====================
# 模型直接在CPU上实例化，无需 .to(device)
net = PassCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_history = []
acc_history = []
print("开始训练（5轮）...")

for epoch in range(5):
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        # 数据也直接在CPU上处理，无需 .to(device)
        # inputs, labels = inputs.to("cpu"), labels.to("cpu")  # 可省略
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(trainloader)
    loss_history.append(avg_loss)
    
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            # images, labels = images.to("cpu"), labels.to("cpu")  # 可省略
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_acc = 100 * correct / total
    acc_history.append(epoch_acc)
    
    print(f"第 {epoch+1} 轮 | 平均Loss: {avg_loss:.3f} | 测试准确率: {epoch_acc:.2f}%")

# ===================== 4. 最终评估 + 收集可视化样本 =====================
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

# ===================== 5. 绘制Loss+准确率曲线 =====================
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(range(1, 6), loss_history, marker='o', color='blue', label='平均Loss')
ax1.set_xlabel('训练轮数')
ax1.set_ylabel('Loss值', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(range(1, 6))
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(range(1, 6), acc_history, marker='s', color='red', label='测试准确率')
ax2.set_ylabel('测试准确率（%）', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title('5轮训练Loss与测试准确率变化曲线')
plt.savefig('训练日志曲线.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== 6. 绘制分类正确/错误图片可视化 =====================
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
plt.show()

print("\n📝 核心文件已生成：")
print("1. 训练日志曲线.png → 训练Loss+准确率曲线")
print("2. 分类结果可视化.png → 分类正确/错误图片示例")