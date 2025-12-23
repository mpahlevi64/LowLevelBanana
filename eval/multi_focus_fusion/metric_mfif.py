import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
import math

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

def torch_hab(im1, im2, gray_level=256):
    im1 = im1.view(-1).long()
    im2 = im2.view(-1).long()
    indices = im1 * gray_level + im2
    h = torch.histc(indices.float(), bins=gray_level**2, min=0, max=gray_level**2 - 1)
    h = h.view(gray_level, gray_level)
    h = h / h.sum()
    im1_marg = torch.sum(h, dim=1)
    im2_marg = torch.sum(h, dim=0)
    mask_xy, mask_x, mask_y = h > 0, im1_marg > 0, im2_marg > 0
    h_xy = torch.sum(h[mask_xy] * torch.log2(h[mask_xy]))
    h_x = torch.sum(im1_marg[mask_x] * torch.log2(im1_marg[mask_x]))
    h_y = torch.sum(im2_marg[mask_y] * torch.log2(im2_marg[mask_y]))
    return h_xy - h_x - h_y

def NMI_function(A, B, F_img, gray_level=256):
    MIA = torch_hab(A, F_img, gray_level)
    MIB = torch_hab(B, F_img, gray_level)
    def get_entropy(im):
        hist = torch.histc(im, bins=gray_level, min=0, max=255); hist /= hist.sum()
        mask = hist > 0
        return -torch.sum(hist[mask] * torch.log2(hist[mask]))
    h_a, h_b = get_entropy(A), get_entropy(B)
    return 2 * (MIA + MIB) / (h_a + h_b + 1e-10)

def Qy_function(imgA, imgB, f):
    def gaussian_filter(window_size, sigma, dev):
        gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)], device=dev)
        gauss = gauss / gauss.sum()
        return gauss.view(1, 1, 1, -1)
    def ssim_yang(img1, img2):
        window_size, sigma = 7, 1.5
        win1d = gaussian_filter(window_size, sigma, img1.device)
        window = torch.matmul(win1d.transpose(-1, -2), win1d)
        mu1 = F.conv2d(img1, window, padding=window_size // 2)
        mu2 = F.conv2d(img2, window, padding=window_size // 2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1.pow(2), window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2.pow(2), window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim_map, sigma1_sq, sigma2_sq

    imgA = imgA.unsqueeze(0).unsqueeze(0).double() / 255.0
    imgB = imgB.unsqueeze(0).unsqueeze(0).double() / 255.0
    f = f.unsqueeze(0).unsqueeze(0).double() / 255.0
    ssim_ab, s1, s2 = ssim_yang(imgA, imgB)
    ssim_af, _, _ = ssim_yang(imgA, f)
    ssim_bf, _, _ = ssim_yang(imgB, f)
    bin_map = (ssim_ab >= 0.75).double()
    ramda = s1 / (s1 + s2 + 1e-10)
    qy_map = (ramda * ssim_af + (1 - ramda) * ssim_bf) * bin_map + torch.max(ssim_af, ssim_bf) * (1 - bin_map)
    return qy_map.mean().item()

def Qcb_function(imgA, imgB, f):
    imgA, imgB, f = imgA.double(), imgB.double(), f.double()
    imgA = (imgA - imgA.min()) / (imgA.max() - imgA.min() + 1e-10)
    imgB = (imgB - imgB.min()) / (imgB.max() - imgB.min() + 1e-10)
    f = (f - f.min()) / (f.max() - f.min() + 1e-10)
    f0, f1, a, k, h, p, q, Z = 15.3870, 1.3456, 0.7622, 1, 1, 3, 2, 0.0001
    hang, lie = imgA.shape
    u, v = torch.meshgrid(torch.fft.fftfreq(hang, device=imgA.device), torch.fft.fftfreq(lie, device=imgA.device), indexing='ij')
    r = torch.sqrt((u * (hang / 30))**2 + (v * (lie / 30))**2)
    Sd = torch.exp(-(r / f0)**2) - a * torch.exp(-(r / f1)**2)
    fim1 = torch.fft.ifft2(torch.fft.fft2(imgA) * Sd).real
    fim2 = torch.fft.ifft2(torch.fft.fft2(imgB) * Sd).real
    ffim = torch.fft.ifft2(torch.fft.fft2(f) * Sd).real
    def get_contrast(img):
        x, y = torch.meshgrid(torch.arange(-15, 16, device=img.device), torch.arange(-15, 16, device=img.device), indexing='ij')
        g2d = lambda s: (torch.exp(-(x**2 + y**2) / (2 * s**2)) / (torch.exp(-(x**2 + y**2) / (2 * s**2)).sum())).unsqueeze(0).unsqueeze(0).double()
        return F.conv2d(img.unsqueeze(0).unsqueeze(0), g2d(2), padding=15) / (F.conv2d(img.unsqueeze(0).unsqueeze(0), g2d(4), padding=15) + 1e-10) - 1
    c1, c2, cf = get_contrast(fim1), get_contrast(fim2), get_contrast(ffim)
    p_mask = lambda c: (k * (torch.abs(c)**p)) / (h * (torch.abs(c)**q) + Z)
    c1p, c2p, cfp = p_mask(c1), p_mask(c2), p_mask(cf)
    q1f = torch.where(c1p < cfp, c1p / (cfp + 1e-10), cfp / (c1p + 1e-10))
    q2f = torch.where(c2p < cfp, c2p / (cfp + 1e-10), cfp / (c2p + 1e-10))
    den = c1p**2 + c2p**2 + 1e-10
    return ((c1p**2 / den) * q1f + (c2p**2 / den) * q2f).mean().item()

def evaluate_dataset(config):
    dataset_name = config['name']
    fus_dir = config['fus_dir']
    fusion_files = glob.glob(os.path.join(fus_dir, "*-Fused-SESF.png"))
    if not fusion_files:
        return None

    metrics = {"NMI": [], "EN": [], "AG": [], "SF": [], "QY": [], "QCB": []}

    for f_path in tqdm(fusion_files, desc=dataset_name):
        base_name = os.path.basename(f_path).replace('_fusion.jpg', '')
        if dataset_name == 'Lytro':
            path_a = os.path.join("LowLevelEval/Multi_Focus_Fusion/Lytro", f"{base_name}-A.jpg")
            path_b = os.path.join("LowLevelEval/Multi_Focus_Fusion/Lytro", f"{base_name}-B.jpg")

        elif dataset_name == 'MFFW':
            path_a = os.path.join("LowLevelEval/Multi_Focus_Fusion/MFFW", f"{base_name}_A.jpg")
            path_b = os.path.join("LowLevelEval/Multi_Focus_Fusion/MFFW", f"{base_name}_B.jpg")

        elif dataset_name == 'MFI-WHU':
            path_a = os.path.join("LowLevelEval/Multi_Focus_Fusion/MFI-WHU/source_1", f"{base_name}.jpg")
            path_b = os.path.join("LowLevelEval/Multi_Focus_Fusion/MFI-WHU/source_2", f"{base_name}.jpg")

        elif dataset_name == 'SIMIF':
            path_a = os.path.join("LowLevelEval/Multi_Focus_Fusion/SIMIF", f"{base_name}left.jpg")
            path_b = os.path.join("LowLevelEval/Multi_Focus_Fusion/SIMIF", f"{base_name}right.jpg")
        else:
            continue

        img_a = cv2.imread(path_a, 0)
        img_b = cv2.imread(path_b, 0)
        img_f = cv2.imread(f_path, 0)

        if img_a is None or img_b is None or img_f is None:
            continue

        t_a = torch.from_numpy(img_a).float().to(device)
        t_b = torch.from_numpy(img_b).float().to(device)
        t_f = torch.from_numpy(img_f).float().to(device)

        with torch.no_grad():
            metrics["NMI"].append(NMI_function(t_a, t_b, t_f).item())
            metrics["EN"].append(EN_function(t_f).item())
            metrics["AG"].append(AG_function(t_f).item())
            metrics["SF"].append(SF_function(t_f).item())
            metrics["QY"].append(Qy_function(t_a, t_b, t_f))
            metrics["QCB"].append(Qcb_function(t_a, t_b, t_f))

    return {k: np.mean(v) for k, v in metrics.items()}

if __name__ == '__main__':
    RESULT_BASE = "test_results_sesf/results_2023xxxx_xxxxxx"
    datasets_config = [
        {"name": "Lytro",   "fus_dir": os.path.join(RESULT_BASE, "Lytro")},
        {"name": "MFFW",    "fus_dir": os.path.join(RESULT_BASE, "MFFW")},
        {"name": "MFI-WHU", "fus_dir": os.path.join(RESULT_BASE, "MFI-WHU")},
        {"name": "SIMIF",   "fus_dir": os.path.join(RESULT_BASE, "SIMIF")}
    ]
    all_res = {cfg['name']: evaluate_dataset(cfg) for cfg in datasets_config if evaluate_dataset(cfg)}
    print("\n" + "="*75)
    print(f"{'Dataset':<12} | {'NMI':<8} | {'EN':<8} | {'AG':<8} | {'SF':<8} | {'QY':<8} | {'QCB':<8}")
    print("-" * 75)
    for n, r in all_res.items():
        print(f"{n:<12} | {r['NMI']:<8.4f} | {r['EN']:<8.4f} | {r['AG']:<8.4f} | {r['SF']:<8.4f} | {r['QY']:<8.4f} | {r['QCB']:<8.4f}")
    print("="*75)