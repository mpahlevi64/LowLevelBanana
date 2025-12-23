import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import os
from PIL import Image
import argparse
from metrics_single import SSIM, PSNR

NAME = 'RealDOF'

def load_image_as_tensor(image_path):
	"""Load image and convert to PyTorch tensor with shape (1, C, H, W)."""
	img = Image.open(image_path).convert('RGB')
	img_array = np.array(img).astype(np.float32)
	# Convert to (C, H, W) and add batch dim -> (1, C, H, W)
	img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
	return img_tensor

def load_image_as_numpy(image_path):
	"""Load image and return as a NumPy array."""
	img = Image.open(image_path).convert('RGB')
	img_array = np.array(img).astype(np.float32)
	return img_array

def find_matching_pairs(results_dir, target_dir):
	"""Find matching image pairs between results and target folders."""
	results_files = set(os.listdir(results_dir))
	target_files = set(os.listdir(target_dir))

	# Exact filename matches present in both folders
	common_files = results_files.intersection(target_files)

	# If no exact matches, try basename matching (without extension)
	if len(common_files) == 0:
		results_basenames = {os.path.splitext(f)[0]: f for f in results_files}
		target_basenames = {os.path.splitext(f)[0]: f for f in target_files}
		common_basenames = set(results_basenames.keys()).intersection(set(target_basenames.keys()))
		pairs = [(results_basenames[bn], target_basenames[bn]) for bn in common_basenames]
	else:
		pairs = [(f, f) for f in common_files]

	return pairs

def compute_metrics_for_pairs(results_dir, target_dir, device='cpu'):
	"""Compute SSIM and PSNR for matched image pairs from results and target folders."""
	pairs = find_matching_pairs(results_dir, target_dir)

	if len(pairs) == 0:
		print(f"Warning: No matching image pairs found in {results_dir} and {target_dir}")
		return []

	results = []

	print(f"Found {len(pairs)} matching pairs")

	for result_file, target_file in pairs:
		result_path = os.path.join(results_dir, result_file)
		target_path = os.path.join(target_dir, target_file)

		try:
			# Read target image to get target size
			target_img = Image.open(target_path).convert('RGB')
			target_size = target_img.size  # (width, height)

			# Read result image and resize to target size
			result_img = Image.open(result_path).convert('RGB')
			result_img_resized = result_img.resize(target_size, Image.BILINEAR)

			# Convert to NumPy arrays (H, W, C) in range [0, 255]
			result_array = np.array(result_img_resized).astype(np.float32)
			target_array = np.array(target_img).astype(np.float32)

			# Normalize to [0, 1] for SSIM
			result_array_norm = result_array / 255.0
			target_array_norm = target_array / 255.0

			# Convert to tensors (1, C, H, W) for SSIM
			result_tensor = torch.from_numpy(result_array_norm).permute(2, 0, 1).unsqueeze(0).to(device)
			target_tensor = torch.from_numpy(target_array_norm).permute(2, 0, 1).unsqueeze(0).to(device)

			# Keep original [0, 255] range for PSNR
			result_numpy = result_array
			target_numpy = target_array

			# Compute SSIM ([0,1] input expected)
			ssim_value = SSIM(result_tensor, target_tensor).item()

			# Compute PSNR
			psnr_value = PSNR(result_numpy, target_numpy)

			results.append({
				'result_file': result_file,
				'target_file': target_file,
				'ssim': ssim_value,
				'psnr': psnr_value
			})

			print(f"{result_file}: SSIM={ssim_value:.4f}, PSNR={psnr_value:.4f}")

		except Exception as e:
			print(f"Error processing {result_file} and {target_file}: {str(e)}")
			continue

	return results

def save_results_to_csv(results, output_path):
	"""Save results to a CSV file."""
	import csv

	with open(output_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=['result_file', 'target_file', 'ssim', 'psnr'])
		writer.writeheader()
		writer.writerows(results)

	print(f"\nResults saved to: {output_path}")

def print_summary(results):
	"""Print summary statistics."""
	if len(results) == 0:
		return

	ssim_values = [r['ssim'] for r in results]
	psnr_values = [r['psnr'] for r in results]

	print("\n" + "="*50)
	print("Summary:")
	print("="*50)
	print(f"Total pairs: {len(results)}")
	print(f"\nSSIM:")
	print(f"  Mean: {np.mean(ssim_values):.4f}")
	print(f"  Std: {np.std(ssim_values):.4f}")
	print(f"  Min: {np.min(ssim_values):.4f}")
	print(f"  Max: {np.max(ssim_values):.4f}")
	print(f"\nPSNR:")
	print(f"  Mean: {np.mean(psnr_values):.4f}")
	print(f"  Std: {np.std(psnr_values):.4f}")
	print(f"  Min: {np.min(psnr_values):.4f}")
	print(f"  Max: {np.max(psnr_values):.4f}")
	print("="*50)

def main():
	parser = argparse.ArgumentParser(description='Compute SSIM and PSNR for matched image pairs in results and target folders')
	parser.add_argument('--results_dir', type=str, default=f'{NAME}/results', 
						help='Path to results folder (default: NAME/results)')
	parser.add_argument('--target_dir', type=str, default=f'{NAME}/target',
						help='Path to target folder (default: NAME/target)')
	parser.add_argument('--output', type=str, default=f'{NAME}/metrics.csv',
						help='Output CSV path (default: NAME/metrics.csv)')
	parser.add_argument('--device', type=str, default='cuda:0',
						help='Compute device (e.g., cpu, cuda, default: cuda:0)')

	args = parser.parse_args()

	# Validate folders
	if not os.path.exists(args.results_dir):
		print(f"Error: results folder does not exist: {args.results_dir}")
		return

	if not os.path.exists(args.target_dir):
		print(f"Error: target folder does not exist: {args.target_dir}")
		return

	# Compute metrics
	results = compute_metrics_for_pairs(args.results_dir, args.target_dir, device=args.device)

	# Print summary
	print_summary(results)

	# Save results
	if args.output:
		save_results_to_csv(results, args.output)

if __name__ == '__main__':
	main()

