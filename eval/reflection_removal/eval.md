# Image Quality Assessment Tool

This tool calculates standard image restoration metrics (**PSNR**, **SSIM**, **LPIPS**, and **MS-SSIM**) between Ground Truth (GT) images and model outputs. It is designed to handle datasets with varying image sizes and different naming conventions through automatic matching and resizing.

## Features
- **Automatic Matching**: Pairs files based on core IDs. For example, `001.png` (GT) will automatically match `001_dehazed.png` (Output), even if suffixes like `_output` or `_result` are present in the output filename.
- **Auto-Resize**: If output dimensions differ from GT, the script resizes the output image using `Lanczos` interpolation before calculation to ensure metric validity.
- **GPU Acceleration**: Uses CUDA for LPIPS and MS-SSIM calculations if a compatible GPU is detected.
- **CSV Export**: Generates a comprehensive CSV report containing metrics for every individual image plus the final dataset average.

## Requirements

### Environment
- Python 3.7+
- CUDA (optional, recommended for faster LPIPS/MS-SSIM calculation)

### Dependencies
Install the required packages via pip:
```bash
pip install numpy torch torchvision pillow scikit-image lpips piq
```

## Usage

### 1. Direct Python Execution
You can run the evaluation script directly from the terminal by passing the required directory paths.

```bash
python eval.py \
    --gt /path/to/gt_folder \
    --output /path/to/output_folder \
    --save ./results.csv \
    --resize-dir ./resized_cache
```

**Arguments:**
- `-g, --gt`: (Required) Path to the folder containing Ground Truth images.
- `-o, --output`: (Required) Path to the folder containing model inference results.
- `-s, --save`: (Required) File path where the results CSV will be saved.
- `-r, --resize-dir`: (Optional) Directory to save images that were resized to match GT dimensions.

### 2. Using the Bash Script
For convenience, a shell script (`run_eval.sh`) is provided to manage parameters in one place.

1.  Open `run_eval.sh` in a text editor.
2.  Edit the configuration variables:
    ```bash
    PYTHON_SCRIPT="eval.py"
    GT_DIR="path/to/gt"
    OUTPUT_DIR="path/to/output"
    SAVE_CSV="path/to/save/result.csv"
    ```
3.  Make the script executable and run it:
    ```bash
    ./run_eval.sh
    ```

## Metrics Overview

| Metric | Full Name | Range | Direction |
| :--- | :--- | :--- | :--- |
| **PSNR** | Peak Signal-to-Noise Ratio | 0 to ∞ | Higher is better |
| **SSIM** | Structural Similarity Index | 0 to 1 | Higher is better |
| **LPIPS** | Learned Perceptual Similarity (AlexNet) | 0 to 1 | **Lower** is better |
| **MS-SSIM** | Multi-Scale Structural Similarity | 0 to 1 | Higher is better |

## Filename Matching Logic
The script extracts a "core ID" from filenames. Suffixes such as `_dehazed`, `_output`, and `_result` are ignored during the matching process. 
- **GT Image**: `001.png`
- **Output Image**: `001_dehazed.png` (or `001_output.png`, `001_result.png`)
- **Result**: Matched successfully.