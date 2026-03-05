import os
import matplotlib.pyplot as plt
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR, StepLR
from thop import profile, clever_format

# ===================== 核心配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')

# ===================== 1. 数据加载 =====================
classes = ('飞机', '汽车', '鸟类', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# ===================== 2. 残差块定义（3层卷积+新增连接层+残差连接） =====================
class ResidualBlock(nn.Module):
    """修改后：3层3×3卷积 + 1×1连接层 + 残差连接"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # ===================== 主路径：3层卷积（核心新增） =====================
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)  # 新增第3层卷积
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # ===================== 新增1×1连接层（跨层特征融合） =====================
        self.connect_layer = nn.Sequential(  # 新增连接层：1×1卷积+BN，融合3层卷积特征
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # ===================== 捷径路径（维度对齐） =====================
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径：3层卷积+新增连接层
        residual = x  # 残差备份
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.connect_layer(out)  # 经过新增连接层
        
        # 残差连接：主路径 + 捷径路径
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# ===================== 3. 模型定义 =====================
class ResNetLikeCNN(nn.Module):
    """带3层卷积残差块+新增连接层+残差连接的CNN"""
    def __init__(self, num_classes=10):
        super(ResNetLikeCNN, self).__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差层：3组3层卷积残差块
        self.layer1 = ResidualBlock(16, 16, stride=1)  
        self.layer2 = ResidualBlock(16, 32, stride=2)  
        self.layer3 = ResidualBlock(32, 32, stride=1)  
        
        self.overlap_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.bn_init(self.init_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.overlap_pool(x)
        x = x.view(batch_size, 32 * 8 * 8)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===================== 4. 训练策略（gamma=0.5 平缓衰减） =====================
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_warmup_scheduler(optimizer, warmup_epochs=2, total_epochs=30):
    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    return LambdaLR(optimizer, lr_lambda)

# ===================== 5. 模型复杂度计算 =====================
def calculate_complexity(model):
    dummy_input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs, params = clever_format((macs, params), "%.3f")
    return macs, params

# ===================== 6. 训练流程 =====================
net = ResNetLikeCNN()
device = torch.device("cpu")
net.to(device)
print(f"✅ 模型已加载（3层卷积残差块+新增连接层+gamma=0.5）")

flops, params = calculate_complexity(net)
print(f"📊 模型复杂度：参数量={params} | 计算量(FLOPs)={flops}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率策略
warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs=2, total_epochs=30)
step_scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

# 训练记录
loss_history = []
train_acc_history = []
test_acc_history = []
lr_history = []

total_epochs = 30
print(f"\n🚀 开始训练（共{total_epochs}轮，3层卷积+新增连接层）...")
for epoch in range(total_epochs):
    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()
        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (lam * (predicted == labels_a).sum().item() + 
                          (1 - lam) * (predicted == labels_b).sum().item())

    avg_loss = running_loss / len(trainloader)
    train_acc = 100 * train_correct / train_total
    current_lr = optimizer.param_groups[0]['lr']
    
    if epoch < 2:
        warmup_scheduler.step()
    else:
        step_scheduler.step()

    # 测试
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = 100 * test_correct / test_total

    # 记录
    loss_history.append(avg_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    lr_history.append(current_lr)

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"第 {epoch+1} 轮 | Loss: {avg_loss:.3f} | 训练准确率: {train_acc:.2f}% | 测试准确率: {test_acc:.2f}% | 学习率: {current_lr:.6f}")

# ===================== 7. 最终评估 =====================
net.eval()
final_correct = 0
final_total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        final_total += labels.size(0)
        final_correct += (predicted == labels).sum().item()

final_accuracy = 100 * final_correct / final_total
print(f"\n🏆 最终测试准确率: {final_accuracy:.2f}%（3层卷积+新增连接层+gamma=0.5）")

# ===================== 8. 训练曲线可视化 =====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(range(1, total_epochs+1), loss_history, marker='o', color='blue', label='平均Loss')
ax1_twin = ax1.twinx()
ax1_twin.plot(range(1, total_epochs+1), train_acc_history, marker='^', color='green', label='训练准确率')
ax1_twin.plot(range(1, total_epochs+1), test_acc_history, marker='s', color='red', label='测试准确率')
ax1.set_xlabel('训练轮数')
ax1.set_ylabel('Loss值', color='blue')
ax1_twin.set_ylabel('准确率（%）', color='red')
ax1.set_xticks(range(0, total_epochs+1, 5))
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, f'参数量：{params}\n计算量：{flops}\n总训练轮数：{total_epochs}\n核心：3层卷积残差块+新增连接层+gamma=0.5', 
         transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), 
         verticalalignment='top')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax1.set_title('任务四：CNN训练曲线（3层卷积残差块+新增连接层+残差连接）')

ax2.plot(range(1, total_epochs+1), lr_history, marker='D', color='purple', label='学习率')
ax2.set_xlabel('训练轮数')
ax2.set_ylabel('学习率')
ax2.set_xticks(range(0, total_epochs+1, 5))
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2.set_title('Warmup + StepLR学习率变化曲线（gamma=0.5 15轮开始衰减）')

plt.tight_layout()
plt.savefig('任务四训练日志曲线_3层卷积+连接层.png', dpi=300, bbox_inches='tight')

print("\n📁 生成文件：")
print("1. 任务四训练日志曲线_3层卷积+连接层.png → 训练过程+模型复杂度+学习率变化")