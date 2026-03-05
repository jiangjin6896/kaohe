from torchvision.datasets import PennFudanPed

# root='./data' 表示下载到当前目录的 data 文件夹下
# download=True 表示如果本地没有，就自动下载
dataset = PennFudanPed(root='./data', download=True)