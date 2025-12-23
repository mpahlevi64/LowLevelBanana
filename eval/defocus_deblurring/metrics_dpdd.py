import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import os
import re
from PIL import Image
import argparse
from metrics_single import SSIM, PSNR

NAME = 'DPDD'

def extract_number_from_filename(filename):
	"""
	Extract the numeric part from a filename.
	Example: "1P0A0917.png" -> 917
	"""
	# 提取文件名（不含扩展名）
	basename = os.path.splitext(filename)[0]
	# Find all digits
	numbers = re.findall(r'\d+', basename)
	if numbers:
		# Return the last digit sequence (usually the index)
		return int(numbers[-1])
	return None

def find_matching_pairs_dpdd(results_dir, target_dir):
	"""
	Find matched pairs in DPDD-style results/target folders.
	Pairing rule: sort by numeric index and pair by position.
	Example: smallest index in results pairs with smallest index in target.
	"""
	pairs = []
	
	if not os.path.exists(results_dir) or not os.path.exists(target_dir):
		return pairs
	
	# List all files
	results_files = [f for f in os.listdir(results_dir) 
					 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	target_files = [f for f in os.listdir(target_dir)
					if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	
	# Extract indexes from results and sort
	results_with_number = []
	for result_file in results_files:
		number = extract_number_from_filename(result_file)
		if number is not None:
			results_with_number.append((number, result_file))
	
	# Extract indexes from target and sort
	target_with_number = []
	for target_file in target_files:
		number = extract_number_from_filename(target_file)
		if number is not None:
			target_with_number.append((number, target_file))
	
	# Sort by index
	results_with_number.sort(key=lambda x: x[0])
	target_with_number.sort(key=lambda x: x[0])
	
	# Pair by position: i-th smallest results with i-th smallest target
	min_len = min(len(results_with_number), len(target_with_number))
	for i in range(min_len):
		result_file = results_with_number[i][1]
		target_file = target_with_number[i][1]
		pairs.append((result_file, target_file))
	
	return pairs

def compute_metrics_for_pairs(results_dir, target_dir, device='cpu'):
	"""Compute SSIM and PSNR for paired images in DPDD-style folders."""
	pairs = find_matching_pairs_dpdd(results_dir, target_dir)
	
	if len(pairs) == 0:
		print(f"Warning: no matched pairs found in {results_dir} and {target_dir}")
		return []
	
	results = []
	
	print(f"Found {len(pairs)} matched image pairs")
	
	for result_file, target_file in pairs:
		result_path = os.path.join(results_dir, result_file)
		target_path = os.path.join(target_dir, target_file)
		
		try:
			# Load target to get size
			target_img = Image.open(target_path).convert('RGB')
			target_size = target_img.size  # (width, height)
			
			# Load result and resize to target size
			result_img = Image.open(result_path).convert('RGB')
			result_img_resized = result_img.resize(target_size, Image.BILINEAR)
			
			# Convert to numpy arrays (H, W, C) in [0, 255]
			result_array = np.array(result_img_resized).astype(np.float32)
			target_array = np.array(target_img).astype(np.float32)
			
			# Normalize to [0, 1] for SSIM
			result_array_norm = result_array / 255.0
			target_array_norm = target_array / 255.0
			
			# Convert to torch tensors (1, C, H, W) for SSIM
			result_tensor = torch.from_numpy(result_array_norm).permute(2, 0, 1).unsqueeze(0).to(device)
			target_tensor = torch.from_numpy(target_array_norm).permute(2, 0, 1).unsqueeze(0).to(device)
			
			# Keep original [0, 255] for PSNR
			result_numpy = result_array
			target_numpy = target_array
			
			# SSIM (expects [0, 1])
			ssim_value = SSIM(result_tensor, target_tensor).item()
			
			# PSNR
			psnr_value = PSNR(result_numpy, target_numpy)
			
			results.append({
				'result_file': result_file,
				'target_file': target_file,
				'ssim': ssim_value,
				'psnr': psnr_value
			})
			
			print(f"{result_file} <-> {target_file}: SSIM={ssim_value:.4f}, PSNR={psnr_value:.4f}")
			
		except Exception as e:
			print(f"Error processing {result_file} and {target_file}: {str(e)}")
			continue
	
	return results

def save_results_to_csv(results, output_path):
	"""Save results to CSV file"""
	import csv
	
	# Ensure output directory exists
	os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
	
	with open(output_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=['result_file', 'target_file', 'ssim', 'psnr'])
		writer.writeheader()
		writer.writerows(results)
	
	print(f"\nResults saved to: {output_path}")

def print_summary(results):
	"""Print summary statistics"""
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
	parser = argparse.ArgumentParser(description='Compute SSIM and PSNR for paired images in DPDD-style folders (adjacent index pairing supported)')
	parser.add_argument('--results_dir', type=str, default=f'{NAME}/results', 
					help='Path to results folder (default: DPDD/results)')
	parser.add_argument('--target_dir', type=str, default=f'{NAME}/target',
					help='Path to target folder (default: DPDD/target)')
	parser.add_argument('--output', type=str, default=f'{NAME}/metrics.csv',
					help='Output CSV file path (default: DPDD/metrics.csv)')
	parser.add_argument('--device', type=str, default='cuda:0',
					help='Compute device (default: cuda:0, options: cpu)')
	
	args = parser.parse_args()
	
	# Check directories
	if not os.path.exists(args.results_dir):
		print(f"Error: results directory not found: {args.results_dir}")
		return
	
	if not os.path.exists(args.target_dir):
		print(f"Error: target directory not found: {args.target_dir}")
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

