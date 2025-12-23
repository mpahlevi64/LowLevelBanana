import os
import torch
import pandas as pd
from tqdm import tqdm
import pyiqa

# ================= 配置区域 =================

METHOD_NAME = "Nano banana pro"

# HR 数据集路径
datasets = {
    "DIV2K": "my_images/DIV2K-Val/HR",
    "RealSR": "my_images/RealSR/RealSR/HR",
    "DRealSR": "my_images/DRealSR/DRealSR/HR"
}

# SR 结果的总目录
sr_root_dir = "SR_result_tasks/results"

# 设备配置
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ================= 初始化模型 =================

print("Loading Metrics (NIQE, MUSIQ, CLIPIQA)...")
niqe_metric = pyiqa.create_metric('niqe', device=device)
musiq_metric = pyiqa.create_metric('musiq', device=device, as_loss=False)
clipiqa_metric = pyiqa.create_metric('clipiqa', device=device, as_loss=False)



def find_sr_images_by_hr_ref(hr_dir, sr_mixed_dir):
    """
    根据 HR 文件夹里的文件名，去 SR 混合文件夹里找对应的图片。
    """
    valid_sr_paths = []
    
    
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    
    if not os.path.exists(hr_dir):
        print(f"Warning: HR directory not found: {hr_dir}")
        return []

    
    hr_files = [f for f in os.listdir(hr_dir) if f.lower().endswith(exts)]
    
    print(f"  > Found {len(hr_files)} reference images in HR folder.")

    found_count = 0
    missing_count = 0

    for hr_file in hr_files:
        
        file_stem = os.path.splitext(hr_file)[0]
        
        
        candidate_found = False
        
        for ext in exts:
            
            candidate_1 = os.path.join(sr_mixed_dir, f"{file_stem}_SR{ext}")
            
            candidate_2 = os.path.join(sr_mixed_dir, f"{file_stem}{ext}")
            
            if os.path.exists(candidate_1):
                valid_sr_paths.append(candidate_1)
                candidate_found = True
                break 
            elif os.path.exists(candidate_2):
                valid_sr_paths.append(candidate_2)
                candidate_found = True
                break
        
        if candidate_found:
            found_count += 1
        else:
            missing_count += 1
            # print(f"    Missing SR for: {file_stem}")

    print(f"  > Matched {found_count} SR images. (Missing: {missing_count})")
    return valid_sr_paths

def evaluate_list(img_paths):
    """
    计算图片列表的指标平均值
    """
    scores = {"NIQE": [], "MUSIQ": [], "CLIPIQA": []}
    
    for img_path in tqdm(img_paths, leave=False):
        try:
            with torch.no_grad():
                # NIQE
                scores["NIQE"].append(niqe_metric(img_path).item())
                # MUSIQ
                scores["MUSIQ"].append(musiq_metric(img_path).item())
                # CLIPIQA
                scores["CLIPIQA"].append(clipiqa_metric(img_path).item())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # 计算平均值
    avg_scores = {}
    for key, val_list in scores.items():
        avg_scores[key] = sum(val_list) / len(val_list) if len(val_list) > 0 else 0.0
            
    return avg_scores

# ================= 主程序 =================

results_list = []

print("\nStarting Evaluation...")

for dataset_name, hr_path in datasets.items():
    print(f"\nProcessing Dataset: {dataset_name}")
    print(f"  Ref Path: {hr_path}")
    
    sr_img_paths = find_sr_images_by_hr_ref(hr_path, sr_root_dir)
    
    if not sr_img_paths:
        print(f"  Skip: No matched images found for {dataset_name}")
        continue
        
    metrics = evaluate_list(sr_img_paths)
    
    row = {
        "Method": METHOD_NAME,
        "Dataset": dataset_name,
        "NIQE↓": round(metrics["NIQE"], 4),
        "MUSIQ↑": round(metrics["MUSIQ"], 4),
        "CLIPIQA↑": round(metrics["CLIPIQA"], 4)
    }
    results_list.append(row)

# ================= 生成表格 =================

if results_list:
    df = pd.DataFrame(results_list)
    df = df[["Dataset", "Method", "NIQE↓", "MUSIQ↑", "CLIPIQA↑"]]
    
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    print(df.to_markdown(index=False))
    
    df.to_csv("evaluation_results_mixed.csv", index=False)
    print(f"\nSaved to evaluation_results_mixed.csv")
else:
    print("No results generated.")