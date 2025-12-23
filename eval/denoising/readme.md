# low light enhancement and denoising evaluation
we compute ssim and psnr in this document. 
please change the below"_dehazed" and "_mean" to the corresponding prefix.
```
def get_image_stem(filename):
    stem = os.path.splitext(filename)[0]
    if stem.endswith("_dehazed"):
        stem = stem[:-len("_dehazed")]
    elif stem.endswith("_mean"):
        stem = stem[:-len("_mean")]
    return stem
```

and change the correct folder name.(FOLDER_RAW is the gt, and the FOLDER_DEHAZED is the generated images.)
```
if __name__ == "__main__":
    # 替换为你的实际路径
    FOLDER_RAW = r"Denoising_raw\SIDD_GT"          # 原始图：1.png, 2.png...
    FOLDER_DEHAZED = r"nano_tasks_denoise_v1\SIDD_Noisy"  # 去雾图：1_dehazed.png...
    OUTPUT_TXT = r"sidd_results_v1.txt"

    batch_calculate_psnr_ssim(FOLDER_RAW, FOLDER_DEHAZED, OUTPUT_TXT)
```

this script can also record images with low psnr, you can change the "LOW_PSNR_OUTPUT" to record low psnr images in different datasets.
```
PSNR_THRESHOLD = 10.0  # PSNR低于该值判定为“过低”
LOW_PSNR_OUTPUT = r"polyu_low_psnr_images.txt"  # 低PSNR图片名保存路径
```