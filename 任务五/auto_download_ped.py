import os
import zipfile
import requests
from tqdm import tqdm

# ========== 固定配置（无需修改） ==========
# 数据集保存路径：桌面\zuoye\data
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
DATA_ROOT = os.path.join(DESKTOP_PATH, "zuoye", "data")
ZIP_FILE = os.path.join(DATA_ROOT, "PennFudanPed.zip")
UNZIP_PATH = os.path.join(DATA_ROOT, "PennFudanPed")

# 官方稳定下载地址（境外源，分块下载不中断）
DOWNLOAD_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

# ========== 核心功能函数 ==========
def make_dir():
    """创建数据文件夹"""
    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)
        print(f"✅ 创建文件夹：{DATA_ROOT}")
    else:
        print(f"✅ 文件夹已存在：{DATA_ROOT}")

def download_file():
    """分块下载数据集，显示进度条"""
    if os.path.exists(ZIP_FILE):
        print(f"📦 压缩包已存在，跳过下载：{ZIP_FILE}")
        return

    print(f"🚀 开始下载数据集（约25MB），请耐心等待...")
    try:
        # 分块下载，避免中断
        response = requests.get(DOWNLOAD_URL, stream=True, timeout=300)
        total_size = int(response.headers.get("content-length", 0))
        
        # 写入文件+显示进度条
        with open(ZIP_FILE, "wb") as f, tqdm(
            desc="下载进度",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 每次下载1MB
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        print("✅ 数据集下载完成！")
    except Exception as e:
        print(f"❌ 下载失败：{str(e)}")
        # 失败后删除不完整文件
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
        exit(1)

def unzip_file():
    """解压数据集，删除压缩包"""
    if os.path.exists(UNZIP_PATH):
        print(f"📂 数据集已解压，跳过解压：{UNZIP_PATH}")
        return

    print("📦 开始解压数据集...")
    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(DATA_ROOT)
        # 删除压缩包节省空间
        os.remove(ZIP_FILE)
        print("✅ 数据集解压完成！")
    except Exception as e:
        print(f"❌ 解压失败：{str(e)}")
        exit(1)

def verify_dataset():
    """验证数据集完整性"""
    img_dir = os.path.join(UNZIP_PATH, "PNGImages")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"❌ 数据集缺失关键目录：{img_dir}")
    
    img_list = os.listdir(img_dir)
    img_count = len(img_list)
    if img_count == 170:
        print(f"✅ 数据集验证通过！图片总数：{img_count}（标准170张）")
        print(f"📍 数据集最终路径：{UNZIP_PATH}")
    else:
        print(f"⚠️  数据集数量异常：{img_count} 张（应为170张）")
        print(f"📍 请检查路径：{img_dir}")

# ========== 主程序执行 ==========
if __name__ == "__main__":
    print("="*50)
    print("开始下载 PennFudanPed 数据集")
    print("="*50)
    try:
        make_dir()
        download_file()
        unzip_file()
        verify_dataset()
        print("\n🎉 所有操作完成！数据集可直接用于后续实验。")
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        print("💡 解决方案：删除 data 文件夹后重新运行本脚本")