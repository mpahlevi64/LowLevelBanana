hdr_evl.py

This code is used to evaluate the image quality between the inferred image folder and the ground truth (GT) image folder. It supports the simultaneous calculation of four metrics: PSNR, SSIM, LPIPS, and DeltaE. The evaluation results are saved to a specified file. It also supports evaluating the image quality of two datasets at the same time.

Configuration parameters: Set INFER_DIR_1, GT_DIR_1 (the first pair of folders), INFER_DIR_2, GT_DIR_2 (the second pair of folders, optional), OUTPUT_FILE (result output path), and GPU_ID (GPU number) in the Configuration section of the code.
The evaluation results will be printed to the console and saved to the path specified by OUTPUT_FILE.