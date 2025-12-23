import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyiqa
from PIL import Image
from torchvision.transforms.functional import to_tensor

# ================= 配置区域 =================

METHOD_NAME = "Nano banana pro"

datasets = {
    "DIV2K": "my_images/DIV2K-Val/HR",
    "RealSR": "my_images/RealSR/RealSR/HR",
    "DRealSR": "my_images/DRealSR/DRealSR/HR"
}

sr_root_dir = "SR_result_tasks/results"

# 是使用 Y 通道计算 PSNR/SSIM 

USE_Y_CHANNEL = True 

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ================= 初始化模型 =================

print("Loading Metrics (PSNR, SSIM, LPIPS)...")

psnr_metric = pyiqa.create_metric('psnr', device=device, test_y_channel=USE_Y_CHANNEL)
ssim_metric = pyiqa.create_metric('ssim', device=device, test_y_channel=USE_Y_CHANNEL)
lpips_metric = pyiqa.create_metric('lpips', device=device, net_type='alex')

# ================= 工具函数 =================

def load_image_as_tensor(path):
    """
    读取图片并转为 (1, C, H, W) 的 Tensor，范围 [0, 1]，RGB顺序
    """
    img = Image.open(path).convert('RGB')
    return to_tensor(img).unsqueeze(0).to(device)

def match_images(hr_dir, sr_mixed_dir):
    """
    匹配 HR 和 SR 图片路径
    """
    pairs = []
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    if not os.path.exists(hr_dir):
        return []
    
    hr_files = [f for f in os.listdir(hr_dir) if f.lower().endswith(exts)]
    
    for hr_file in hr_files:
        file_stem = os.path.splitext(hr_file)[0]
        hr_full_path = os.path.join(hr_dir, hr_file)
        
        sr_candidates = [
            os.path.join(sr_mixed_dir, f"{file_stem}_SR.png"),
            os.path.join(sr_mixed_dir, f"{file_stem}.png"),
            os.path.join(sr_mixed_dir, f"{file_stem}_SR.jpg"),
            os.path.join(sr_mixed_dir, f"{file_stem}.jpg")
        ]
        
        found = False
        for sr_path in sr_candidates:
            if os.path.exists(sr_path):
                pairs.append((hr_full_path, sr_path))
                found = True
                break
    return pairs

def evaluate_pairs(pairs):
    results = {"PSNR": [], "SSIM": [], "LPIPS": []}
    
    for hr_path, sr_path in tqdm(pairs, leave=False):
        try:
            # 1. 加载图片
            hr_tensor = load_image_as_tensor(hr_path)
            sr_tensor = load_image_as_tensor(sr_path)
            
            # 2. 分辨率对齐
    
            if sr_tensor.shape != hr_tensor.shape:
                sr_tensor = F.interpolate(
                    sr_tensor, 
                    size=hr_tensor.shape[-2:],
                    mode='bicubic', 
                    align_corners=False
                )
                sr_tensor = sr_tensor.clamp(0, 1)

            # 3. 计算指标
            with torch.no_grad():
                # PSNR 
                val_psnr = psnr_metric(sr_tensor, hr_tensor).item()
                results["PSNR"].append(val_psnr)
                
                # SSIM 
                val_ssim = ssim_metric(sr_tensor, hr_tensor).item()
                results["SSIM"].append(val_ssim)
                
                # LPIPS 
                val_lpips = lpips_metric(sr_tensor, hr_tensor).item()
                results["LPIPS"].append(val_lpips)
                
        except Exception as e:
            print(f"Error: {e} | HR: {os.path.basename(hr_path)}")

    # 计算平均值
    avg_results = {}
    for k, v in results.items():
        avg_results[k] = sum(v) / len(v) if len(v) > 0 else 0.0
    return avg_results

# ================= 主程序 =================

final_data = []

print(f"\nConfiguration: Use Y-Channel = {USE_Y_CHANNEL}")
print("Starting Recalculation...")

for dataset_name, hr_path in datasets.items():
    print(f"\nProcessing {dataset_name}...")
    
    # 1. 匹配
    pairs = match_images(hr_path, sr_root_dir)
    if not pairs:
        print("  No matched images found.")
        continue
    print(f"  Matched {len(pairs)} image pairs.")
    
    # 2. 计算
    scores = evaluate_pairs(pairs)
    
    # 3. 记录
    row = {
        "Dataset": dataset_name,
        "Method": METHOD_NAME,
        "PSNR↑": round(scores["PSNR"], 4),
        "SSIM↑": round(scores["SSIM"], 4),
        "LPIPS↓": round(scores["LPIPS"], 4)
    }
    final_data.append(row)

# ================= 输出表格 =================

if final_data:
    df = pd.DataFrame(final_data)
    print("\n" + "="*50)
    print("Updated Full-Reference Metrics")
    print("="*50)
    print(df.to_markdown(index=False))
    
    df.to_csv("recalc_psnr_ssim_lpips.csv", index=False)
    print("\nSaved to recalc_psnr_ssim_lpips.csv")
else:
    print("No data processed.")