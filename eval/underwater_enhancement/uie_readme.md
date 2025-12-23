uie_evl_withoutgt

This code is used to evaluate the quality of underwater images. It adjusts the inferred images to the same size as the ground truth (GT) images, then calculates two no-reference image quality assessment metrics, namely UCIQE and UIQM, as well as their sub-metrics.
It supports batch processing of images and automatically matches inferred images with GT images.
The evaluation results are saved to a specified file.

Configuration parameters:
Set infer_dir (inferred image folder), gt_dir (GT image folder), result_save_path (result saving path), crop_border (cropping edge size), and input_order (image channel order) in the main function.
The evaluation results will be printed to the console and saved to the path specified by result_save_path.