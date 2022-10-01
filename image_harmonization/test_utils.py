from pathlib import Path
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from image_harmonization.config import *
from skimage import data, img_as_float
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms.functional as tf
import skimage
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_id(list_path):
    img_ids = [i_id.strip() for i_id in open(list_path)]
    return img_ids


def get_test_data(img_path, mask_path, target_path, name):
    name_prepare = name.split("_", 2)
    mask_name = name_prepare[0] + '_' + name_prepare[1]
    img_name = name
    target_name = name_prepare[0]

    mask_file = os.path.join(mask_path, "%s.png" % mask_name)
    img_file = os.path.join(img_path, "%s" % img_name)
    target_file = os.path.join(target_path, "%s.jpg" % target_name)

    image = Image.open(img_file).convert('RGB')
    mask = Image.open(mask_file).convert('1')
    target = Image.open(target_file).convert('RGB')

    image = tf.resize(image, [256, 256])
    mask = tf.resize(mask, [256, 256])
    target = tf.resize(target, [256, 256])

    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)
    target = transforms.ToTensor()(target)

    return image.to(device).unsqueeze(0), mask.to(device).unsqueeze(
        0), target.to(device).unsqueeze(0)


def calc_metrics(pred, gt, mask):
    n, c, h, w = pred.shape
    assert n == 1
    total_pixels = h * w
    fg_pixels = int(torch.sum(mask, dim=(2, 3))[0][0].cpu().numpy())

    pred = torch.clamp(pred * 255, 0, 255)
    gt = torch.clamp(gt * 255, 0, 255)

    pred = pred[0].permute(1, 2, 0).cpu().numpy()
    gt = gt[0].permute(1, 2, 0).cpu().numpy()
    mask = mask[0].permute(1, 2, 0).cpu().numpy()

    mse = skimage.metrics.mean_squared_error(pred, gt)
    fmse = skimage.metrics.mean_squared_error(pred * mask,
                                              gt * mask) * total_pixels / \
           fg_pixels
    psnr = skimage.metrics.peak_signal_noise_ratio(pred, gt,
                                                   data_range=pred.max() -
                                                              pred.min())
    ssim = skimage.metrics.structural_similarity(pred, gt, multichannel=True)

    return mse, fmse, psnr, ssim


def iterate_over_dataset_and_get_metrics(model, test_list_path, img_path,
                                         mask_path, target_path):
    mses = []
    psnrs = []
    fmses = []
    ssims = []
    img_ids = get_test_id(test_list_path)
    for img_id in img_ids:
        image, mask, target = get_test_data(img_path, mask_path, target_path,
                                            img_id)
        with torch.no_grad():
            output = model(image, mask)
            # output2 = model.calc_reconstruction_debug(image, mask, target)
            # import ipdb;
            # ipdb.set_trace()
            # if img_id == 'd90000014-10_1_2.jpg':
            #     output2 = model.calc_reconstruction_debug(image, mask, target)
                # import ipdb; ipdb.set_trace()
            mse, fmse, psnr, ssim = calc_metrics(output, target, mask)
            mses.append(mse)
            psnrs.append(psnr)
            fmses.append(fmse)
            ssims.append(ssim)

    return mses, psnrs, fmses, ssims, img_ids



def iterate_over_dataset_and_get_mask_metrics(model, test_list_path, img_path,
                                              mask_path, target_path):
    mses = []
    psnrs = []
    fmses = []
    ssims = []
    img_ids = get_test_id(test_list_path)
    for img_id in img_ids:
        image, mask, target = get_test_data(img_path, mask_path, target_path,
                                            img_id)
        with torch.no_grad():
            output = model(image)
            mse, fmse, psnr, ssim = calc_metrics(output, target, mask)
            mses.append(mse)
            psnrs.append(psnr)
            fmses.append(fmse)
            ssims.append(ssim)

    return mses, psnrs, fmses, ssims, img_ids


def plot_one_result_in_debug_mode(image, target, mask, output):
    plt.title('Composite Image')
    plt.imshow(image.squeeze(0).cpu().detach().permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
    plt.title('Real Image')
    plt.imshow(target.squeeze(0).cpu().detach().permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
    plt.title('Foreground Mask')
    plt.imshow(mask.squeeze(0).cpu().detach().permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
    plt.title('Output Image')
    plt.imshow(output.squeeze(0).cpu().detach().permute(1, 2, 0))
    plt.tight_layout()
    plt.show()
    plt.title('Diff Image')
    plt.imshow(
        torch.abs(output - target).squeeze(0).cpu().detach().sum(axis=[0]))
    plt.colorbar()
    plt.tight_layout()
    plt.show()

