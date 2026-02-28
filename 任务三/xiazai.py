import torchvision
import torchvision.transforms as transforms

# 第一步：定义数据预处理（至少包含转张量，归一化可选）
transform = transforms.Compose([
    transforms.ToTensor(),  # 必须：将图片转为PyTorch张量
])

# 第二步：下载训练集（核心行）
trainset = torchvision.datasets.CIFAR10(
    root='./data',          # 数据保存路径（当前目录下的data文件夹）
    train=True,             # True=训练集，False=测试集
    download=True,          # 关键：自动下载（首次运行下载，后续跳过）
    transform=transform     # 数据预处理
)

# 下载测试集（同理）
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)