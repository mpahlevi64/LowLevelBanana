import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def EN_function(img):
    histogram = torch.histc(img, bins=256, min=0, max=255)
    histogram = histogram / histogram.sum()
    mask = histogram > 0
    entropy = -torch.sum(histogram[mask] * torch.log2(histogram[mask]))
    return entropy

def SF_function(img):
    RF = img[1:, :] - img[:-1, :]
    CF = img[:, 1:] - img[:, :-1]
    RF1 = torch.sqrt(torch.mean(RF ** 2))
    CF1 = torch.sqrt(torch.mean(CF ** 2))
    SF = torch.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF

def AG_function(img):
    grad = torch.gradient(img)
    grady, gradx = grad[0], grad[1]
    s = torch.sqrt((gradx ** 2 + grady ** 2) / 2.0)
    AG = torch.mean(s)
    return AG

def SD_function(img):
    return torch.std(img, unbiased=False)

def fspecial_gaussian(size, sigma, device):
    m, n = size
    x = torch.linspace(-m//2 + 1, m//2, m, device=device)
    y = torch.linspace(-n//2 + 1, n//2, n, device=device)
    x, y = torch.meshgrid(x, y, indexing='ij')
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()

def convolve2d_torch(input, kernel):
    kH, kW = kernel.shape
    padding = (kH // 2, kW // 2)
    input_4d = input.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(input_4d, kernel_4d, padding=padding)
    return out.squeeze()

def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5, device=ref.device)

        if scale > 1:
            ref = convolve2d_torch(ref, win)
            dist = convolve2d_torch(dist, win)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d_torch(ref, win)
        mu2 = convolve2d_torch(dist, win)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = convolve2d_torch(ref * ref, win) - mu1_sq
        sigma2_sq = convolve2d_torch(dist * dist, win) - mu2_sq
        sigma12 = convolve2d_torch(ref * dist, win) - mu1_mu2
        
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0

        g[sigma2_sq < 1e-10] = 0
        sv_sq[sigma2_sq < 1e-10] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq = torch.clamp(sv_sq, min=1e-10)

        num += torch.sum(torch.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq))

    return num / (den + 1e-10)

def VIF_function(A, B, F):
    return vifp_mscale(A, F) + vifp_mscale(B, F)

def corr2_torch(a, b):
    a = a - torch.mean(a)
    b = b - torch.mean(b)
    den = torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
    if den < 1e-10:
        return torch.tensor(0.0, device=a.device)
    return torch.sum(a * b) / den

def SCD_function(A, B, F):
    return corr2_torch(F - B, A) + corr2_torch(F - A, B)

def evaluate_dataset(config):
    dataset_name = config['name']
    ir_dir, vi_dir, fus_dir = config['ir_dir'], config['vi_dir'], config['fus_dir']
    
    fusion_files = glob.glob(os.path.join(fus_dir, "*_fusion.png"))
    if not fusion_files:
        print(f"[{dataset_name}] No fusion images found.")
        return None

    print(f"\n>>> Testing {dataset_name}: {len(fusion_files)} images, using {device}")

    metrics_list = {"EN": [], "SD": [], "SF": [], "AG": [], "SCD": [], "VIF": []}
    
    for f_path in tqdm(fusion_files, desc=dataset_name):
        filename = os.path.basename(f_path)
        file_id = filename.split('_fusion.png')[0]
        source_filename = f"{file_id}{config['source_ext']}"
        
        img_ir_np = cv2.imread(os.path.join(ir_dir, source_filename), 0)
        img_vi_np = cv2.imread(os.path.join(vi_dir, source_filename), 0)
        img_f_np = cv2.imread(f_path, 0)

        if img_ir_np is None or img_vi_np is None or img_f_np is None:
            continue

        t_ir = torch.from_numpy(img_ir_np).float().to(device)
        t_vi = torch.from_numpy(img_vi_np).float().to(device)
        t_f = torch.from_numpy(img_f_np).float().to(device)

        with torch.no_grad():
            try:
                metrics_list["EN"].append(EN_function(t_f).item())
                metrics_list["SD"].append(SD_function(t_f).item())
                metrics_list["SF"].append(SF_function(t_f).item())
                metrics_list["AG"].append(AG_function(t_f).item())
                metrics_list["SCD"].append(SCD_function(t_ir, t_vi, t_f).item())
                metrics_list["VIF"].append(VIF_function(t_ir, t_vi, t_f).item())
            except Exception as e:
                print(f"Error at {filename}: {e}")
                continue

    avg_results = {k: np.mean(v) if v else 0.0 for k, v in metrics_list.items()}
    return avg_results

if __name__ == '__main__':
    datasets_config = [
        {
            "name": "M3FD",
            "ir_dir": "LowLevelEval/Infrared_Visible_Fusion/M3FD/ir",
            "vi_dir": "LowLevelEval/Infrared_Visible_Fusion/M3FD/vi",
            "fus_dir": "LowLevelEval/Infrared_Visible_Fusion/M3FD_NBPro",
            "source_ext": ".png"  
        },
        {
            "name": "RoadScene",
            "ir_dir": "LowLevelEval/Infrared_Visible_Fusion/RoadScene/ir",
            "vi_dir": "LowLevelEval/Infrared_Visible_Fusion/RoadScene/vi",
            "fus_dir": "LowLevelEval/IInfrared_Visible_Fusion/RoadScene_NBPro",
            "source_ext": ".jpg"  
        },
        {
            "name": "MSRS",
            "ir_dir": "LowLevelEval/Infrared_Visible_Fusion/MSRS/ir",
            "vi_dir": "LowLevelEval/Infrared_Visible_Fusion/MSRS/vi",
            "fus_dir": "LowLevelEval/Infrared_Visible_Fusion/MSRS_NBPro",
            "source_ext": ".png"  
        }
    ]

    all_results = {}
    for config in datasets_config:
        res = evaluate_dataset(config)
        if res:
            all_results[config['name']] = res

    print("\n" + "="*75)
    print(f"{'Dataset':<12} | {'EN':<8} | {'SD':<8} | {'SF':<8} | {'AG':<8} | {'SCD':<8} | {'VIF':<8}")
    print("-" * 75)
    for name, res in all_results.items():
        print(f"{name:<12} | {res['EN']:<8.4f} | {res['SD']:<8.4f} | {res['SF']:<8.4f} | {res['AG']:<8.4f} | {res['SCD']:<8.4f} | {res['VIF']:<8.4f}")
    print("="*75)