import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ==================== 可自定义参数 ====================
PSNR_THRESHOLD = 10.0  # PSNR低于该值判定为“过低”
LOW_PSNR_OUTPUT = r"polyu_low_psnr_images.txt"  # 低PSNR图片名保存路径
# ======================================================

def get_image_stem(filename):
    """提取文件名主干（去掉后缀和_dehazed标识）"""
    stem = os.path.splitext(filename)[0]
    if stem.endswith("_dehazed"):
        stem = stem[:-len("_dehazed")]
    elif stem.endswith("_mean"):
        stem = stem[:-len("_mean")]
    return stem

def load_image(image_path):
    """用Pillow读取图片并强制转为3维RGB数组（H, W, 3），增强维度校验"""
    try:
        with Image.open(image_path) as img:
            # 处理多帧图片（如GIF），仅取第一帧
            #if img.is_animated:
                #img.seek(0)  # 定位到第一帧
            # 统一转为RGB（灰度/RGBA→RGB）
            img_rgb = img.convert('RGB')
            # 转为numpy数组
            img_np = np.array(img_rgb, dtype=np.float32)
            
            # 强制校验维度：必须是3维 (H, W, 3)
            if len(img_np.shape) != 3 or img_np.shape[-1] != 3:
                raise ValueError(f"图片维度异常，期望 (H, W, 3)，实际 {img_np.shape}")
            
            return img_np
    except Exception as e:
        raise ValueError(f"读取/处理图片失败：{e} | 路径：{image_path}")

def calculate_psnr(img1, img2):
    """手动实现PSNR计算（峰值信噪比）"""
    # 双重校验维度（避免计算异常）
    assert len(img1.shape) == 3 and img1.shape[-1] == 3, f"img1维度错误：{img1.shape}"
    assert len(img2.shape) == 3 and img2.shape[-1] == 3, f"img2维度错误：{img2.shape}"
    
    # 计算均方误差（MSE）
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # 两张图完全一致，PSNR无穷大
        return float('inf')
    # PSNR公式：10 * log10((255^2) / MSE)
    psnr_value = 10 * np.log10((255.0 ** 2) / mse)
    return psnr_value

def calculate_ssim(img1, img2):
    """手动实现RGB图片的SSIM计算（增强维度校验）"""
    # 前置校验
    assert len(img1.shape) == 3 and img1.shape[-1] == 3, f"img1维度错误：{img1.shape}"
    assert len(img2.shape) == 3 and img2.shape[-1] == 3, f"img2维度错误：{img2.shape}"
    
    C1 = (0.01 * 255) ** 2  # 稳定项1
    C2 = (0.03 * 255) ** 2  # 稳定项2

    # 高斯核（用于计算局部均值/方差/协方差）
    def gaussian_kernel(size=11, sigma=1.5):
        x = np.linspace(-(size//2), size//2, size)
        g = np.exp(-0.5 * (x/sigma) ** 2)
        kernel = np.outer(g, g)
        return kernel / np.sum(kernel)

    kernel = gaussian_kernel()
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # 对每个通道计算SSIM，最后取平均
    ssim_channels = []
    for c in range(3):  # RGB 3个通道
        img1_c = img1[..., c]  # 取出单通道 (H, W)
        img2_c = img2[..., c]  # 取出单通道 (H, W)

        # 校验单通道维度（必须是2D）
        assert len(img1_c.shape) == 2, f"img1通道{c}维度错误：{img1_c.shape}"
        assert len(img2_c.shape) == 2, f"img2通道{c}维度错误：{img2_c.shape}"

        # 填充边界（避免边缘像素计算偏差）
        img1_pad = np.pad(img1_c, pad, mode='reflect')
        img2_pad = np.pad(img2_c, pad, mode='reflect')

        # 卷积计算局部均值（仅对2D数组卷积）
        mu1 = np.convolve(np.convolve(img1_pad, kernel, mode='valid'), kernel.T, mode='valid')
        mu2 = np.convolve(np.convolve(img2_pad, kernel, mode='valid'), kernel.T, mode='valid')

        # 局部方差
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = np.convolve(np.convolve(img1_pad**2, kernel, mode='valid'), kernel.T, mode='valid') - mu1_sq
        sigma2_sq = np.convolve(np.convolve(img2_pad**2, kernel, mode='valid'), kernel.T, mode='valid') - mu2_sq
        sigma12 = np.convolve(np.convolve(img1_pad*img2_pad, kernel, mode='valid'), kernel.T, mode='valid') - mu1_mu2

        # 计算通道SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_channels.append(np.mean(ssim_map))

    # 3个通道取平均，得到整体SSIM
    return np.mean(ssim_channels)

def calculate_psnr_ssim(img1, img2):
    """计算单对图片的PSNR和SSIM（Pillow读取的numpy数组）"""
    # 检查尺寸是否一致，不一致则缩放（用Pillow缩放）
    if img1.shape != img2.shape:
        print(f"警告：图片尺寸不一致，自动缩放匹配 → {img1.shape} → {img2.shape}")
        # 转换为Pillow对象缩放，再转回numpy
        img2_pil = Image.fromarray(img2.astype(np.uint8))
        img2_resized = img2_pil.resize((img1.shape[1], img1.shape[0]), Image.Resampling.LANCZOS)
        img2 = np.array(img2_resized, dtype=np.float32)
    
    # 计算PSNR（数据范围0-255）
    psnr_value = psnr(img1, img2, data_range=255.0)
    
    # 计算SSIM（channel_axis=-1适配RGB数组）
    ssim_value = ssim(img1, img2, data_range=255.0, channel_axis=-1)
    
    return psnr_value, ssim_value

def batch_calculate_psnr_ssim(folder1, folder2, output_txt=None):
    """批量计算配对图片的PSNR/SSIM，同时保存低PSNR图片名（增强异常调试）"""
    # 构建文件名映射
    folder1_files = {}
    for filename in os.listdir(folder1):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            stem = get_image_stem(filename)
            folder1_files[stem] = os.path.join(folder1, filename)

    folder2_files = {}
    for filename in os.listdir(folder2):
        #print(filename.lower())
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            #print(1)
            stem = get_image_stem(filename)
            folder2_files[stem] = os.path.join(folder2, filename)

    # 找共同主干
    #print(folder2_files.keys())
    common_stems = set(folder1_files.keys()) & set(folder2_files.keys())
    if not common_stems:
        print("错误：无配对图片！")
        return

    # 初始化存储变量
    results = []
    low_psnr_stems = []  # 存储PSNR过低的图片主干名
    total_psnr = 0.0
    total_ssim = 0.0

    print(f"找到 {len(common_stems)} 对可对比图片（PSNR阈值：{PSNR_THRESHOLD}）：")
    print("-" * 80)
    print(f"{'文件名主干':<10} {'PSNR':<10} {'SSIM':<10} {'状态'}")
    print("-" * 80)

    for stem in sorted(common_stems):
        img1_path = folder1_files[stem]
        img2_path = folder2_files[stem]
        try:
            # 读取图片并打印维度（调试关键！）
            img1 = load_image(img1_path)
            img2 = load_image(img2_path)
            print(f"调试：{stem} → 原始图维度：{img1.shape}，处理图维度：{img2.shape}")
            
            # 计算PSNR/SSIM
            psnr_val, ssim_val = calculate_psnr_ssim(img1, img2)

            # 判断是否为低PSNR图片
            status = "正常"
            if np.isfinite(psnr_val) and psnr_val < PSNR_THRESHOLD:
                status = "PSNR过低"
                low_psnr_stems.append(stem)  # 记录低PSNR图片主干名

            # 累加统计值（排除无穷大的情况）
            total_psnr += psnr_val if np.isfinite(psnr_val) else 0
            total_ssim += ssim_val
            results.append((stem, psnr_val, ssim_val))
            
            # 打印每行结果（标注状态）
            print(f"{stem:<10} {psnr_val:<10.4f} {ssim_val:<10.4f} {status}")

        except Exception as e:
            # 详细打印异常信息，定位问题图片
            print(f"跳过 {stem}：{e}")
            # 可选：将异常图片名保存到文件
            with open("error_images.txt", 'a', encoding='utf-8') as f:
                f.write(f"{stem} → 原始图：{img1_path} | 处理图：{img2_path} | 错误：{e}\n")
            continue

    # 计算平均值
    valid_psnr = [p for _, p, _ in results if np.isfinite(p)]
    avg_psnr = np.mean(valid_psnr) if valid_psnr else 0
    avg_ssim = np.mean([s for _, _, s in results]) if results else 0

    print("-" * 80)
    print(f"{'平均值':<10} {avg_psnr:<10.4f} {avg_ssim:<10.4f}")
    print(f"\nPSNR过低的图片数量：{len(low_psnr_stems)}")

    # 1. 保存低PSNR图片名到文件
    if low_psnr_stems:
        with open(LOW_PSNR_OUTPUT, 'w', encoding='utf-8') as f:
            f.write(f"PSNR阈值：{PSNR_THRESHOLD}\n")
            f.write("低PSNR图片主干名（对应原始图：{stem}.png，处理图：{stem}_dehazed.png）\n")
            f.write("-" * 50 + "\n")
            for stem in low_psnr_stems:
                f.write(f"{stem}\n")
        print(f"低PSNR图片名已保存到：{LOW_PSNR_OUTPUT}")
    else:
        print("无PSNR过低的图片")

    # 2. 保存完整的PSNR/SSIM结果（可选）
    if output_txt and results:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"PSNR阈值：{PSNR_THRESHOLD}\n")
            f.write("文件名主干,PSNR,SSIM\n")
            for stem, p, s in results:
                f.write(f"{stem},{p:.4f},{s:.4f}\n")
            f.write(f"平均值,,{avg_psnr:.4f},{avg_ssim:.4f}\n")
        print(f"完整结果已保存到：{output_txt}")

# ==================== 示例使用 ====================
if __name__ == "__main__":
    # 替换为你的实际路径
    FOLDER_RAW = r"Denoising_raw\SIDD_GT"          # 原始图：1.png, 2.png...
    FOLDER_DEHAZED = r"nano_tasks_denoise_v1\SIDD_Noisy"  # 去雾图：1_dehazed.png...
    OUTPUT_TXT = r"sidd_results_v1.txt"

    batch_calculate_psnr_ssim(FOLDER_RAW, FOLDER_DEHAZED, OUTPUT_TXT)