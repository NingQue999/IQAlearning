import torch  
from piq import psnr, ssim  
import cv2  


# 读取图像并转换为Tensor  
def load_image(path):  
    img = cv2.imread(path)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式  
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0  
    return img_tensor  

# 加载图像  
ref_img = load_image("D:\\VScode_workplace\\IQAlearning\\images\\I02.jpg")  
distorted_img = load_image("D:\\VScode_workplace\\IQAlearning\\images\\I02_01_1.jpg")  

# 计算PSNR和SSIM  
psnr_value = psnr(distorted_img, ref_img)  
ssim_value = ssim(distorted_img, ref_img)  

print(f"PSNR: {psnr_value:.2f} dB")  
print(f"SSIM: {ssim_value:.4f}")  