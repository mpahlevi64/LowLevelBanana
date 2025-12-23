import os
import cv2
import numpy as np
from skimage import transform
from scipy import ndimage
from tqdm import tqdm


def calculate_uciqe(img, crop_border=0, input_order='HWC'):
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
    if img.shape[2] == 3:
        img_bgr = img
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    coe_metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_lab[..., 0] / 255.0
    img_a = img_lab[..., 1] / 255.0
    img_b = img_lab[..., 2] / 255.0

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))
    img_sat = img_chr / (np.sqrt(np.square(img_chr) + np.square(img_lum)) + 1e-8)
    aver_sat = np.mean(img_sat)

    aver_chr = np.mean(img_chr)
    var_chr = np.sqrt(np.mean(np.abs(1 - np.square(aver_chr / (img_chr + 1e-8)))))

    dtype = img_lum.dtype
    nbins = 256 if dtype == 'uint8' else 65536
    hist, bins = np.histogram(img_lum, nbins)
    cdf = np.cumsum(hist) / np.sum(hist)
    ilow = np.where(cdf > 0.01)[0]
    ihigh = np.where(cdf >= 0.99)[0]
    con_lum = 0.0 if (len(ilow) == 0 or len(ihigh) == 0) else (ihigh[0] - ilow[0]) / (nbins - 1)

    return coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[2] * aver_sat


def _uicm(img):
    img = np.array(img, dtype=np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    RG = R - G
    YB = (R + G) / 2 - B
    K = R.shape[0] * R.shape[1]
    
    RG1 = np.sort(RG.reshape(1, K))
    alphaL, alphaR = 0.1, 0.1
    start, end = int(alphaL * K + 1), int(K * (1 - alphaR))
    RG1 = RG1[0, start:end] if start < end else RG1[0, :]
    N = max(1, K * (1 - alphaR - alphaL))
    meanRG = np.sum(RG1) / N
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) **2) / N)
    
    YB1 = np.sort(YB.reshape(1, K))
    YB1 = YB1[0, start:end] if start < end else YB1[0, :]
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB)** 2) / N)
    
    return -0.0268 * np.sqrt(meanRG **2 + meanYB** 2) + 0.1586 * np.sqrt(deltaYB **2 + deltaRG** 2)


def _uiconm(img):
    img = np.array(img, dtype=np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    patchez = 5
    m, n = R.shape[0], R.shape[1]
    
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez) if m % patchez != 0 else m
        y = int(n - n % patchez + patchez) if n % patchez != 0 else n
        R = transform.resize(R, (x, y), anti_aliasing=True)
        G = transform.resize(G, (x, y), anti_aliasing=True)
        B = transform.resize(B, (x, y), anti_aliasing=True)
    
    m, n = R.shape[0], R.shape[1]
    k1, k2 = m // patchez, n // patchez
    
    def cal_amee(channel):
        amee = 0.0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                patch = channel[i:i+patchez, j:j+patchez]
                Max, Min = np.max(patch), np.min(patch)
                if (Max != 0 or Min != 0) and Max != Min:
                    amee += np.log((Max - Min)/(Max + Min)) * ((Max - Min)/(Max + Min))
        return np.abs(amee) / (k1 * k2) if k1 * k2 != 0 else 0.0
    
    return cal_amee(R) + cal_amee(G) + cal_amee(B)


def _uism(img):
    img = np.array(img, dtype=np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest') + ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest') + ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest') + ndimage.convolve(B, hy, mode='nearest'))
    
    patchez = 5
    m, n = SobelR.shape[0], SobelR.shape[1]
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez) if m % patchez != 0 else m
        y = int(n - n % patchez + patchez) if n % patchez != 0 else n
        SobelR = transform.resize(SobelR, (x, y), anti_aliasing=True)
        SobelG = transform.resize(SobelG, (x, y), anti_aliasing=True)
        SobelB = transform.resize(SobelB, (x, y), anti_aliasing=True)
    
    m, n = SobelR.shape[0], SobelR.shape[1]
    k1, k2 = m // patchez, n // patchez
    
    def cal_eme(channel):
        eme = 0.0
        for i in range(0, m, patchez):
            for j in range(0, n, patchez):
                patch = channel[i:i+patchez, j:j+patchez]
                Max, Min = np.max(patch), np.min(patch)
                if Max > 0 and Min > 0:
                    eme += np.log(Max / Min)
        return 2 * np.abs(eme) / (k1 * k2) if k1 * k2 != 0 else 0.0
    
    lambdaR, lambdaG, lambdaB = 0.299, 0.587, 0.114
    return lambdaR * cal_eme(SobelR) + lambdaG * cal_eme(SobelG) + lambdaB * cal_eme(SobelB)


def calculate_uiqm(img, crop_border=0, input_order='HWC', return_submetrics=False):
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    if img.shape[2] != 3:
        img = np.stack([img[..., 0]] * 3, axis=-1)
    
    img = img.astype(np.float32)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    
    uicm_val = _uicm(img)
    uism_val = _uism(img)
    uiconm_val = _uiconm(img)
    
    uiqm_total = c1 * uicm_val + c2 * uism_val + c3 * uiconm_val
    
    if return_submetrics:
        return uiqm_total, uicm_val, uism_val, uiconm_val
    else:
        return uiqm_total


def get_image_path_by_basename(folder, basename):
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for filename in os.listdir(folder):
        file_basename = os.path.splitext(filename)[0]
        if file_basename == basename:
            return os.path.join(folder, filename)
    return None


def calculate_average_metrics(infer_dir, gt_dir, crop_border=0, input_order='HWC'):
    if not os.path.isdir(infer_dir):
        raise ValueError(f"推理文件夹不存在：{infer_dir}")
    if not os.path.isdir(gt_dir):
        raise ValueError(f"GT文件夹不存在：{gt_dir}")
    
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    gt_filenames = [f for f in os.listdir(gt_dir) if f.lower().endswith(img_extensions)]
    if not gt_filenames:
        raise ValueError(f"GT文件夹中未找到有效图片：{gt_dir}")
    
    total_uciqe = 0.0
    total_uiqm = 0.0
    total_uicm = 0.0
    total_uism = 0.0
    total_uiconm = 0.0
    valid_count = 0
    missing_count = 0
    resize_count = 0

    print(f"开始计算（共{len(gt_filenames)}张GT图）...")
    for gt_filename in tqdm(gt_filenames):
        gt_basename = os.path.splitext(gt_filename)[0]
        gt_path = os.path.join(gt_dir, gt_filename)
        
        infer_path = get_image_path_by_basename(infer_dir, gt_basename)
        if not infer_path:
            missing_count += 1
            print(f"警告：未找到匹配的推理图，已跳过 -> GT图：{gt_filename}（basename：{gt_basename}）")
            continue
        
        gt_img = cv2.imread(gt_path)
        infer_img = cv2.imread(infer_path)
        if gt_img is None:
            print(f"警告：GT图读取失败，已跳过 -> {gt_filename}")
            continue
        if infer_img is None:
            print(f"警告：推理图读取失败，已跳过 -> {infer_path}")
            continue
        
        gt_height, gt_width = gt_img.shape[:2]
        infer_height, infer_width = infer_img.shape[:2]
        if (infer_height, infer_width) != (gt_height, gt_width):
            infer_img = cv2.resize(
                infer_img, 
                (gt_width, gt_height),
                interpolation=cv2.INTER_CUBIC
            )
            resize_count += 1
        
        try:
            uciqe = calculate_uciqe(infer_img, crop_border, input_order)
            uiqm_total, uicm_val, uism_val, uiconm_val = calculate_uiqm(
                infer_img, crop_border, input_order, return_submetrics=True
            )
        except Exception as e:
            print(f"警告：计算指标失败，已跳过 -> {gt_basename}，错误：{str(e)}")
            continue
        
        total_uciqe += uciqe
        total_uiqm += uiqm_total
        total_uicm += uicm_val
        total_uism += uism_val
        total_uiconm += uiconm_val
        valid_count += 1
    
    if valid_count == 0:
        raise RuntimeError("没有有效图片参与计算，请检查图片匹配和格式")
    
    return {
        "总GT图片数": len(gt_filenames),
        "未匹配推理图数": missing_count,
        "需resize推理图数": resize_count,
        "有效计算图片数": valid_count,
        "平均UCIQE": round(total_uciqe / valid_count, 4),
        "平均UIQM": round(total_uiqm / valid_count, 4),
        "平均UICM": round(total_uicm / valid_count, 4),
        "平均UISM": round(total_uism / valid_count, 4),
        "平均UICONM": round(total_uiconm / valid_count, 4)
    }


def main():
    infer_dir = "/home/dancer/zhangyw/uiecontrast/U45/UWCNN/epoch_1000"
    gt_dir = "/home/dancer/zhangyw/uiecontrast/U45/UIEC/epoch_1000"
    result_save_path = "/home/dancer/zhangyw/uiecontrast/U45/UWCNN/u45_view_nogt.txt"
    
    crop_border = 0
    input_order = 'HWC'
    
    try:
        metrics = calculate_average_metrics(
            infer_dir=infer_dir,
            gt_dir=gt_dir,
            crop_border=crop_border,
            input_order=input_order
        )
        
        with open(result_save_path, 'w', encoding='utf-8') as f:
            f.write("推理图质量评估指标（resize至GT尺寸后）\n")
            f.write("====================================\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}：{value}\n")
            f.write("\n指标说明：\n")
            f.write("1. UCIQE：水下彩色图像质量评估指标（越高越好）\n")
            f.write("2. UIQM：水下图像质量评估指标（越高越好）\n")
            f.write("3. UICM：色彩均衡度指标（值越接近0越好）\n")
            f.write("4. UISM：图像清晰度指标（越高越好）\n")
            f.write("5. UICONM：对比度指标（越高越好）\n")
        
        print(f"计算完成！结果已保存至：{result_save_path}")
    except Exception as e:
        print(f"计算失败：{str(e)}")


if __name__ == "__main__":
    main()