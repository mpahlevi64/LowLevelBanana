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

NAME = 'GoPro'

def find_matching_pairs_gopro(results_dir, target_dir):
	"""
	Find matched image pairs in GoPro-style results/target folders.
	Structure: results/{scene_name}/blur/{filename}.png
	          target/{scene_name}/sharp/{filename}.png
	"""
	pairs = []
	
	# Get all scene folders
	if not os.path.exists(results_dir):
		return pairs
	
	# Iterate all scene folders under results
	for scene_name in os.listdir(results_dir):
		results_scene_dir = os.path.join(results_dir, scene_name)
		target_scene_dir = os.path.join(target_dir, scene_name)
		
		# Skip non-directory entries
		if not os.path.isdir(results_scene_dir):
			continue
		
		# Check corresponding target scene exists
		if not os.path.exists(target_scene_dir):
			continue
		
		# Get blur and sharp subfolders
		results_blur_dir = os.path.join(results_scene_dir, 'blur')
		target_sharp_dir = os.path.join(target_scene_dir, 'sharp')
		
		if not os.path.exists(results_blur_dir) or not os.path.exists(target_sharp_dir):
			continue
		
		# List image files in blur folder
		if not os.path.isdir(results_blur_dir):
			continue
		
		results_files = set(f for f in os.listdir(results_blur_dir) 
							if f.lower().endswith(('.png', '.jpg', '.jpeg')))
		
		# List image files in sharp folder
		if not os.path.isdir(target_sharp_dir):
			continue
		
		target_files = set(f for f in os.listdir(target_sharp_dir)
						  if f.lower().endswith(('.png', '.jpg', '.jpeg')))
		
		# Intersect common filenames
		common_files = results_files.intersection(target_files)
		
		# Build path pairs for each match
		for filename in common_files:
			result_path = os.path.join(results_blur_dir, filename)
			target_path = os.path.join(target_sharp_dir, filename)
			
			# Use relative paths as identifiers
			result_rel_path = os.path.join(scene_name, 'blur', filename)
			target_rel_path = os.path.join(scene_name, 'sharp', filename)
			
			pairs.append((result_path, target_path, result_rel_path, target_rel_path))
	
	return pairs

def compute_metrics_for_pairs(results_dir, target_dir, device='cpu'):
	"""Compute SSIM and PSNR for paired images in GoPro-style folders."""
	pairs = find_matching_pairs_gopro(results_dir, target_dir)
	
	if len(pairs) == 0:
		print(f"Warning: no matched pairs found in {results_dir} and {target_dir}")
		return []
	
	results = []
	
	print(f"Found {len(pairs)} matched image pairs")
	
	for result_path, target_path, result_rel_path, target_rel_path in pairs:
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
				'result_file': result_rel_path,
				'target_file': target_rel_path,
				'ssim': ssim_value,
				'psnr': psnr_value
			})
			
			print(f"{result_rel_path}: SSIM={ssim_value:.4f}, PSNR={psnr_value:.4f}")
			
		except Exception as e:
			print(f"Error processing {result_rel_path} and {target_rel_path}: {str(e)}")
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
	parser = argparse.ArgumentParser(description='Compute SSIM and PSNR for paired images in GoPro-style folders')
	parser.add_argument('--results_dir', type=str, default=f'{NAME}/results', 
					help='Path to results folder (default: GoPro/results)')
	parser.add_argument('--target_dir', type=str, default=f'{NAME}/target',
					help='Path to target folder (default: GoPro/target)')
	parser.add_argument('--output', type=str, default=f'{NAME}/metrics.csv',
					help='Output CSV file path (default: GoPro/metrics.csv)')
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

