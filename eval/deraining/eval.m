clc;close all;clear all;addpath(genpath('./'));

% Derained outputs and ground truth paths
file_path = './output/Rain200H/';           % your outputs
gt_path = './data/Rain200H/test/target/';             % ground truths
% file_path is the folder that contains derained images to be evaluated
% gt_path is the folder that contains ground truth images

% Optional: evaluate only a subset
max_images = 1000;                 % set [] to evaluate all outputs found
selected_list_file = '';          % e.g., 'G:/path/to/selected_100.txt' (one filename per line, with extension)
selected_names = {};              % e.g., {'0001.png','0002.png'}; takes precedence over selected_list_file

path_list = [dir(fullfile(file_path,'*.jpg')); dir(fullfile(file_path,'*.png'))];
[~, order] = sort({path_list.name});
path_list = path_list(order);

if isempty(selected_names) && ~isempty(selected_list_file) && exist(selected_list_file, 'file')
    raw = fileread(selected_list_file);
    selected_names = regexp(raw, '\r\n|\n|\r', 'split');
    selected_names = selected_names(~cellfun('isempty', selected_names));
end
if ~isempty(selected_names)
    keep = ismember({path_list.name}, selected_names);
    path_list = path_list(keep);
end
if ~isempty(max_images) && numel(path_list) > max_images
    path_list = path_list(1:max_images);
end

img_num = numel(path_list);

total_psnr = 0;
total_ssim = 0;
valid_num = 0;
missing_gt = 0;
if img_num == 0
    fprintf('No output images found under: %s\n', file_path);
else
    for j = 1:img_num
       image_name = path_list(j).name;
       input_file = fullfile(file_path, image_name);
       gt_file = find_gt_file(gt_path, image_name);
       if isempty(gt_file)
           missing_gt = missing_gt + 1;
           continue;
       end
       input = imread(input_file);
       gt = imread(gt_file);
       ssim_val = compute_ssim(input, gt);
       psnr_val = compute_psnr(input, gt);
       total_ssim = total_ssim + ssim_val;
       total_psnr = total_psnr + psnr_val;
       valid_num = valid_num + 1;
   end
end
if valid_num == 0
    qm_psnr = NaN;
    qm_ssim = NaN;
else
    qm_psnr = total_psnr / valid_num;
    qm_ssim = total_ssim / valid_num;
end

fprintf('PSNR: %f SSIM: %f (valid=%d, missing_gt=%d)\n', qm_psnr, qm_ssim, valid_num, missing_gt);

function gt_file = find_gt_file(gt_path, output_name)
    gt_file = fullfile(gt_path, output_name);
    if exist(gt_file, 'file')
        return;
    end
    [~, base, ~] = fileparts(output_name);
    exts = {'.png', '.jpg', '.jpeg'};
    for i = 1:numel(exts)
        cand = fullfile(gt_path, [base exts{i}]);
        if exist(cand, 'file')
            gt_file = cand;
            return;
        end
    end
    gt_file = '';
end

function ssim_mean=compute_ssim(img1,img2)
    if size(img1, 3) == 3
        img1 = rgb2ycbcr(img1);
        img1 = img1(:, :, 1);
    end

    if size(img2, 3) == 3
        img2 = rgb2ycbcr(img2);
        img2 = img2(:, :, 1);
    end
    ssim_mean = SSIM_index(img1, img2);
end

function psnr=compute_psnr(img1,img2)
    if size(img1, 3) == 3
        img1 = rgb2ycbcr(img1);
        img1 = img1(:, :, 1);
    end

    if size(img2, 3) == 3
        img2 = rgb2ycbcr(img2);
        img2 = img2(:, :, 1);
    end

    imdff = double(img1) - double(img2);
    imdff = imdff(:);
    rmse = sqrt(mean(imdff.^2));
    psnr = 20*log10(255/rmse);
    
end

function [mssim, ssim_map] = SSIM_index(img1, img2, K, window, L)

%========================================================================
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author is with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error measurement to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, no. 1, Jan. 2004.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%
%========================================================================


if (nargin < 2 || nargin > 5)
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) || (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);	%
   K(1) = 0.01;								      % default settings
   K(2) = 0.03;								      %
   L = 255;                                  %
end

if (nargin == 3)
   if ((M < 11) || (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 || (H > M) || (W > N))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 || K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);

mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;

if (C1 > 0 & C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
	denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);

end
