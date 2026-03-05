# 先设置环境变量，解决OpenMP警告问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. 初始化YOLOv8轻量版分割模型（n版最简单、得分最低）
model = YOLO('yolov8n-seg.pt')  # 直接加载预训练模型，无需训练

# 2. 配置数据集路径（绝对路径，避免任何歧义）
output_dir = r"C:\Users\jiangjin\Desktop\zuoye\data\yolo_seg_results"
img_dir = r"C:\Users\jiangjin\Desktop\zuoye\data\PennFudanPed\PNGImages"
os.makedirs(output_dir, exist_ok=True)  # 自动创建结果文件夹

# 3. 批量处理所有图片（文件直接保存到指定目录，无嵌套）
for img_name in os.listdir(img_dir):
    if img_name.endswith('.png'):  # 只处理png图片
        img_path = os.path.join(img_dir, img_name)
        
        # 关键修改：关闭YOLO自动保存，手动保存结果图（避免生成predict子文件夹）
        results = model(img_path, save=False)  # 关闭YOLO自动保存
        result = results[0]
        
        # 1. 手动保存YOLO分割结果图（直接到指定目录）
        seg_img_path = os.path.join(output_dir, f"分割结果_{img_name}")
        cv2.imwrite(seg_img_path, result.plot())  # 手动保存带框+掩码的图
        print(f"📄 已生成分割图：{seg_img_path}")
        
        # 2. 手动保存原图vs分割结果对比图（直接到指定目录）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_img = result.plot()
        
        compare_img_path = os.path.join(output_dir, f"对比图_{img_name.replace('.png', '.jpg')}")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("原图")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.title("实例分割结果（行人框+掩码）")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(compare_img_path, dpi=100)
        plt.close()
        print(f"📄 已生成对比图：{compare_img_path}")

# 4. 最后打印目录路径，方便你直接复制打开
print("\n✅ 所有文件已生成！")
print(f"📂 文件目录：{output_dir}")
print(f"💡 你可以在文件管理器中粘贴此路径直接打开：\n{output_dir}")