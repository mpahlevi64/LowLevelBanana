# 将 infer 图片 resize 到对应 gt 图片的尺寸后再计算所有指标
"""Evaluate metrics between infer folder and gt folder.
Calculate PSNR, SSIM, LPIPS, and DeltaE using the same logic as the project.
"""
import os
import glob
import cv2
import numpy as np
import torch
import lpips
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2lab, deltaE_ciede94

INFER_DIR_1 = "/home/dancer/zhangyw/EVL/HDR_EVL/nano/HDR+"
GT_DIR_1 = "/home/dancer/zhangyw/EVL/HDR_EVL/HDR+/expert"

INFER_DIR_2 = "/home/dancer/zhangyw/EVL/HDR_EVL/nano/FiveK"
GT_DIR_2 = "/home/dancer/zhangyw/EVL/HDR_EVL/FiveK/expert"

OUTPUT_FILE = '/home/dancer/zhangyw/EVL/HDR_EVL/nano/top10_results.txt'
GPU_ID = 0
TOP_N = 10


class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def quality_assess(X, Y, data_range=255):
    if X.ndim == 3:
        psnr = compare_psnr(Y, X, data_range=data_range)
        ssim = compare_ssim(Y, X, data_range=data_range, channel_axis=2)
        delta = np.mean(deltaE_ciede94(rgb2lab(Y.astype(np.uint8)), rgb2lab(X.astype(np.uint8))))
        return {'PSNR': psnr, 'SSIM': ssim, 'DeltaE': delta}
    else:
        raise NotImplementedError


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in 
               ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif', '.TIF'])


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = img[:, :, ::-1].copy()
    return img


def numpy2tensor(img):
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def get_image_pairs(infer_dir, gt_dir):
    infer_files = sorted([f for f in glob.glob(os.path.join(infer_dir, '*')) 
                         if is_image_file(f)])
    gt_files = sorted([f for f in glob.glob(os.path.join(gt_dir, '*')) 
                      if is_image_file(f)])
    
    gt_dict = {}
    for gt_file in gt_files:
        base_name = os.path.splitext(os.path.basename(gt_file))[0]
        gt_dict[base_name] = gt_file
    
    pairs = []
    for infer_file in infer_files:
        base_name = os.path.splitext(os.path.basename(infer_file))[0]
        if base_name in gt_dict:
            pairs.append((infer_file, gt_dict[base_name]))
        else:
            print(f"Warning: No matching GT file for {infer_file}")
    
    if len(pairs) == 0:
        raise ValueError(f"No matching image pairs found between {infer_dir} and {gt_dir}")
    
    return pairs


def evaluate_dataset(infer_dir, gt_dir, device, metric_lpips):
    if not os.path.isdir(infer_dir):
        raise ValueError(f"Infer directory does not exist: {infer_dir}")
    if not os.path.isdir(gt_dir):
        raise ValueError(f"GT directory does not exist: {gt_dir}")
    
    print(f"\n处理数据集:")
    print(f"  Infer directory: {infer_dir}")
    print(f"  GT directory: {gt_dir}")
    
    image_pairs = get_image_pairs(infer_dir, gt_dir)
    print(f"  找到 {len(image_pairs)} 对匹配的图片")
    
    per_image_metrics = []
    
    for infer_path, gt_path in tqdm(image_pairs, desc='评估中', ncols=80):
        try:
            infer_img = read_image(infer_path)
            gt_img = read_image(gt_path)
            
            gt_h, gt_w = gt_img.shape[:2]
            infer_img = cv2.resize(infer_img, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR).copy()
            
            res = quality_assess(infer_img, gt_img, data_range=255)
            psnr_val = res['PSNR']
            ssim_val = res['SSIM']
            deltae_val = res['DeltaE']
            
            infer_tensor = numpy2tensor(infer_img).to(device)
            gt_tensor = numpy2tensor(gt_img).to(device)
            
            with torch.no_grad():
                lpips_value = metric_lpips(infer_tensor, gt_tensor)
                lpips_val = torch.mean(lpips_value).item()
            
            image_name = os.path.basename(infer_path)
            per_image_metrics.append({
                'name': image_name,
                'PSNR': psnr_val,
                'SSIM': ssim_val,
                'LPIPS': lpips_val,
                'DeltaE': deltae_val
            })
        
        except Exception as e:
            print(f"\n处理图片时出错 {infer_path}: {str(e)}")
            continue
    
    return per_image_metrics


def get_composite_score(metrics):
    return metrics['PSNR'] / 50.0 + metrics['SSIM'] - metrics['LPIPS'] - metrics['DeltaE'] / 100.0


def main():
    infer_dir_1 = INFER_DIR_1
    gt_dir_1 = GT_DIR_1
    infer_dir_2 = INFER_DIR_2
    gt_dir_2 = GT_DIR_2
    output_file = OUTPUT_FILE
    gpu_id = GPU_ID
    top_n = TOP_N
    
    if not infer_dir_1 or not gt_dir_1:
        raise ValueError("请配置第一对文件夹路径 (INFER_DIR_1 和 GT_DIR_1)")
    
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    metric_lpips = lpips.LPIPS(net='alex').to(device)
    metric_lpips.eval()
    
    metrics_1 = evaluate_dataset(infer_dir_1, gt_dir_1, device, metric_lpips)
    
    metrics_2 = []
    if infer_dir_2 and gt_dir_2:
        metrics_2 = evaluate_dataset(infer_dir_2, gt_dir_2, device, metric_lpips)
    
    for m in metrics_1:
        m['score'] = get_composite_score(m)
    for m in metrics_2:
        m['score'] = get_composite_score(m)
    
    top_n_1 = sorted(metrics_1, key=lambda x: x['score'], reverse=True)[:top_n]
    top_n_2 = sorted(metrics_2, key=lambda x: x['score'], reverse=True)[:top_n] if metrics_2 else []
    
    results_text = '\n' + '='*80 + '\n'
    results_text += '两个数据集指标最高的前{}张图片\n'.format(top_n)
    results_text += '='*80 + '\n'
    
    results_text += '\n数据集 1:\n'
    results_text += f'Infer目录: {infer_dir_1}\n'
    results_text += f'GT目录: {gt_dir_1}\n'
    results_text += f'总图片数: {len(metrics_1)}\n'
    results_text += f'前{top_n}张最高指标图片:\n'
    results_text += '-'*80 + '\n'
    results_text += f'{"排名":<6} {"文件名":<35} {"PSNR":>10} {"SSIM":>10} {"LPIPS":>10} {"DeltaE":>10} {"综合得分":>10}\n'
    results_text += '-'*80 + '\n'
    for idx, m in enumerate(top_n_1, 1):
        results_text += f'{idx:<6} {m["name"]:<35} {m["PSNR"]:>10.4f} {m["SSIM"]:>10.4f} {m["LPIPS"]:>10.4f} {m["DeltaE"]:>10.4f} {m["score"]:>10.4f}\n'
    
    if top_n_2:
        results_text += '\n数据集 2:\n'
        results_text += f'Infer目录: {infer_dir_2}\n'
        results_text += f'GT目录: {gt_dir_2}\n'
        results_text += f'总图片数: {len(metrics_2)}\n'
        results_text += f'前{top_n}张最高指标图片:\n'
        results_text += '-'*80 + '\n'
        results_text += f'{"排名":<6} {"文件名":<35} {"PSNR":>10} {"SSIM":>10} {"LPIPS":>10} {"DeltaE":>10} {"综合得分":>10}\n'
        results_text += '-'*80 + '\n'
        for idx, m in enumerate(top_n_2, 1):
            results_text += f'{idx:<6} {m["name"]:<35} {m["PSNR"]:>10.4f} {m["SSIM"]:>10.4f} {m["LPIPS"]:>10.4f} {m["DeltaE"]:>10.4f} {m["score"]:>10.4f}\n'
    
    results_text += '\n' + '='*80 + '\n'
    results_text += '说明:\n'
    results_text += '  - PSNR: 峰值信噪比，越高越好 (单位: dB)\n'
    results_text += '  - SSIM: 结构相似性，越高越好 (范围: 0-1)\n'
    results_text += '  - LPIPS: 感知相似性，越低越好 (范围: 0-1)\n'
    results_text += '  - DeltaE: 色差，越低越好\n'
    results_text += '  - 综合得分: PSNR/50 + SSIM - LPIPS - DeltaE/100，越高越好\n'
    results_text += '='*80 + '\n'
    
    print(results_text)
    
    if output_file is not None:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results_text)
        print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()