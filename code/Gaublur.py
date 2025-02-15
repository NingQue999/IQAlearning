import torch
from piq import psnr, ssim
import cv2
import pandas as pd
from pathlib import Path
import tabulate

# 读取图像并转换为Tensor
def load_image(path):
    img = cv2.imread(str(path))  # 使用Path对象兼容不同操作系统
    if img is None:
        raise ValueError(f"图像加载失败，请检查路径: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor

# 设置路径
base_dir = Path("D:/VScode_workplace/IQAlearning/images")
ref_path = base_dir / "I02.jpg"          # 原始图像
distorted_pattern = "I02_08_{}.jpg"      # 模糊图像命名模式

# 初始化结果存储
results = []
ref_img = load_image(ref_path)

# 遍历5个模糊级别
for level in range(1, 6):
    # 生成当前模糊图像路径
    distorted_path = base_dir / distorted_pattern.format(level)
    
    # 加载并计算指标
    distorted_img = load_image(distorted_path)
    psnr_value = psnr(distorted_img, ref_img)
    ssim_value = ssim(distorted_img, ref_img)
        
    results.append({
        "Level": level,
        "PSNR (dB)": round(float(psnr_value), 2),
        "SSIM": round(float(ssim_value), 4),
    })

# 生成并保存结果表格

df = pd.DataFrame(results)
df = df.sort_values(by="Level")
csv_path = base_dir / "quality_metrics.csv"
df.to_csv(csv_path, index=False)
print("\n质量评估结果：")
print(df[["Level", "PSNR (dB)", "SSIM"]].to_markdown(index=False))
