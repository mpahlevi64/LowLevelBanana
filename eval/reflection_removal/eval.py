import os
import csv
import argparse
import numpy as np
import torch
import lpips
import piq
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def get_core_filename(filename):
    """Extract the core ID from a filename by removing extensions and common suffixes."""
    base = os.path.splitext(filename)[0]
    for suffix in ["_dehazed", "_output", "_result"]:
        if suffix in base:
            base = base.replace(suffix, "")
    return base


def load_image(path):
    """Load an image and convert to RGB numpy array (uint8)."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {str(e)}")


def resize_image(img, target_shape, interpolation=Image.LANCZOS):
    """Resize image to target shape (H, W, C) using Lanczos interpolation."""
    img_pil = Image.fromarray(img)
    target_height, target_width = target_shape[:2]
    resized_pil = img_pil.resize((target_width, target_height), resample=interpolation)
    return np.array(resized_pil, dtype=np.uint8)


def save_resized_image(resized_img, original_output_path, save_dir):
    """Save the resized image to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    original_filename = os.path.basename(original_output_path)
    save_path = os.path.join(save_dir, original_filename)
    Image.fromarray(resized_img).save(save_path)
    return save_path


def calculate_psnr_single(gt_img, pred_img):
    """Calculate average PSNR across RGB channels."""
    gt = gt_img.astype(np.float32)
    pred = pred_img.astype(np.float32)
    psnr_vals = [psnr(gt[..., c], pred[..., c], data_range=255) for c in range(3)]
    return np.mean(psnr_vals)


def calculate_ssim_single(gt_img, pred_img):
    """Calculate average SSIM across RGB channels."""
    gt = gt_img.astype(np.float32)
    pred = pred_img.astype(np.float32)
    ssim_vals = [ssim(gt[..., c], pred[..., c], data_range=255) for c in range(3)]
    return np.mean(ssim_vals)


def calculate_lpips_single(gt_img, pred_img, loss_fn):
    """Calculate LPIPS score using the provided model (expects [-1, 1] range)."""
    # Normalize to [-1, 1]
    gt = gt_img.astype(np.float32) / 255.0 * 2 - 1
    pred = pred_img.astype(np.float32) / 255.0 * 2 - 1
    
    # Transpose to NCHW
    gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0)
    
    device = next(loss_fn.parameters()).device
    gt_tensor = gt_tensor.to(device, dtype=torch.float32)
    pred_tensor = pred_tensor.to(device, dtype=torch.float32)
    
    with torch.no_grad():
        lpips_score = loss_fn(gt_tensor, pred_tensor).item()
    return lpips_score


def calculate_ms_ssim_single(gt_img, pred_img):
    """Calculate MS-SSIM score using piq library (expects [0, 1] range)."""
    # Normalize to [0, 1]
    gt = gt_img.astype(np.float32) / 255.0
    pred = pred_img.astype(np.float32) / 255.0
    
    # Transpose to NCHW
    gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0)
    pred_tensor = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_tensor = gt_tensor.to(device, dtype=torch.float32)
    pred_tensor = pred_tensor.to(device, dtype=torch.float32)
    
    with torch.no_grad():
        ms_ssim_score = piq.multi_scale_ssim(
            pred_tensor, gt_tensor, data_range=1.0
        ).item()
    return ms_ssim_score


def calculate_all_metrics(gt_img, pred_img, loss_fn):
    """Aggregate all 4 metrics into a dictionary."""
    return {
        "psnr": calculate_psnr_single(gt_img, pred_img),
        "ssim": calculate_ssim_single(gt_img, pred_img),
        "lpips": calculate_lpips_single(gt_img, pred_img, loss_fn),
        "ms_ssim": calculate_ms_ssim_single(gt_img, pred_img),
    }


def find_matching_output(gt_filename, output_dir):
    """Match GT filename with output file based on core ID."""
    gt_core = get_core_filename(gt_filename)
    if not os.path.exists(output_dir):
        return None
    for output_filename in os.listdir(output_dir):
        output_core = get_core_filename(output_filename)
        if output_core == gt_core:
            return os.path.join(output_dir, output_filename)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Image Quality Assessment (PSNR, SSIM, LPIPS, MS-SSIM)"
    )
    parser.add_argument("--gt", "-g", required=True, help="Path to Ground Truth folder")
    parser.add_argument("--output", "-o", required=True, help="Path to Model Output folder")
    parser.add_argument("--save", "-s", required=True, help="Path to save results CSV")
    parser.add_argument(
        "--resize-dir", "-r", default=None, help="Directory to save resized output images (optional)"
    )
    args = parser.parse_args()

    print("Initializing LPIPS model...")
    loss_fn = lpips.LPIPS(net="alex", version="0.1")
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
        print("Using GPU for LPIPS/MS-SSIM.")
    else:
        print("Using CPU for LPIPS/MS-SSIM.")

    if not os.path.exists(args.gt):
        print(f"Error: GT directory '{args.gt}' not found.")
        return

    gt_files = [
        f for f in os.listdir(args.gt) if os.path.isfile(os.path.join(args.gt, f))
    ]

    with open(args.save, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Image_ID",
            "Output_Path",
            "Original_Size",
            "Resized_Size",
            "PSNR",
            "SSIM",
            "LPIPS",
            "MS-SSIM",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_psnr, total_ssim, total_lpips, total_ms_ssim = [], [], [], []

        for gt_filename in gt_files:
            core_id = get_core_filename(gt_filename)
            gt_path = os.path.join(args.gt, gt_filename)
            output_path = find_matching_output(gt_filename, args.output)

            if not output_path:
                print(f"⚠️ Warning: No matching output for GT: {gt_filename}")
                continue

            try:
                gt_img = load_image(gt_path)
                output_img = load_image(output_path)
                original_shape = output_img.shape

                img_to_eval = output_img
                resized_flag = "None"

                # Automatic resize if dimensions don't match
                if output_img.shape != gt_img.shape:
                    img_to_eval = resize_image(output_img, gt_img.shape)
                    resized_flag = f"{gt_img.shape[0]}x{gt_img.shape[1]}"
                    if args.resize_dir:
                        save_resized_image(img_to_eval, output_path, args.resize_dir)

                metrics = calculate_all_metrics(gt_img, img_to_eval, loss_fn)

                total_psnr.append(metrics["psnr"])
                total_ssim.append(metrics["ssim"])
                total_lpips.append(metrics["lpips"])
                total_ms_ssim.append(metrics["ms_ssim"])

                writer.writerow(
                    {
                        "Image_ID": core_id,
                        "Output_Path": output_path,
                        "Original_Size": f"{original_shape[0]}x{original_shape[1]}",
                        "Resized_Size": resized_flag,
                        "PSNR": round(metrics["psnr"], 4),
                        "SSIM": round(metrics["ssim"], 4),
                        "LPIPS": round(metrics["lpips"], 6),
                        "MS-SSIM": round(metrics["ms_ssim"], 6),
                    }
                )
                print(f"✅ Processed: {core_id}")
            except Exception as e:
                print(f"❌ Error processing {core_id}: {e}")

        # Calculate and write averages
        if total_psnr:
            avg_metrics = {
                "Image_ID": "AVERAGE",
                "PSNR": round(np.mean(total_psnr), 4),
                "SSIM": round(np.mean(total_ssim), 4),
                "LPIPS": round(np.mean(total_lpips), 6),
                "MS-SSIM": round(np.mean(total_ms_ssim), 6),
            }
            writer.writerow(avg_metrics)
            print("-" * 30)
            print(f"Final Results:")
            print(f"PSNR: {avg_metrics['PSNR']} | SSIM: {avg_metrics['SSIM']} | LPIPS: {avg_metrics['LPIPS']} | MS-SSIM: {avg_metrics['MS-SSIM']}")
            print("-" * 30)


if __name__ == "__main__":
    main()