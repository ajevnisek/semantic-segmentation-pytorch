import os
import torch
import torch.nn as nn

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from types import SimpleNamespace

from mit_semseg.lib.utils import as_numpy
from mit_semseg.dataset import TestDataset
from mit_semseg.utils import find_recursive, colorEncode
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to


from mit_semseg.models import ModelBuilder, SegmentationModule


IMAGE_SHAPE = (256, 256)
MODEL_CONFIGURATION = SimpleNamespace(**{
    'arch_encoder': 'resnet50dilated',
    'arch_decoder': 'ppm_deepsup',
    'weights_encoder': 'ckpt/ade20k-resnet50dilated-ppm_deepsup/'
                       'encoder_epoch_20.pth',
    'weights_decoder': 'ckpt/ade20k-resnet50dilated-ppm_deepsup/'
                       'decoder_epoch_20.pth',
    'fc_dim': 2048,
    'num_class': 150})
DATASET_CONFIGURATION = SimpleNamespace(**{
    'root_dataset': './data/',
    'list_train': './data/training.odgt',
    'list_val': './data/validation.odgt',
    'num_class': 150,
    'imgSizes': (300, 375, 450, 525, 600),
    'imgMaxSize': 1000,
    'padding_constant': 8,
    'segm_downsampling_rate': 8,
    'random_flip': True
})


def get_segmentation_module(model_configuration=MODEL_CONFIGURATION):
    net_encoder = ModelBuilder.build_encoder(
            arch=model_configuration.arch_encoder,
            fc_dim=model_configuration.fc_dim,
            weights=model_configuration.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
            arch=model_configuration.arch_decoder,
            fc_dim=model_configuration.fc_dim,
            num_class=model_configuration.num_class,
            weights=model_configuration.weights_decoder,
            use_softmax=True)
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    return segmentation_module


def get_data_loader(image_dir_path, image_list_path, batch_size=4, shuffle=False):
    imgs = find_recursive(image_dir_path)
    with open(image_list_path, 'r') as f:
        image_list = f.read()
    image_list = image_list.split('\n')[:-1]

    trimmed_images = [img for img in imgs if img.split('/')[-1] in image_list]
    list_test = [{'fpath_img': x} for x in trimmed_images]
    dataset = TestDataset(
        list_test,
        DATASET_CONFIGURATION)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)
    return dataloader


def get_segmentation_map(segmentation_module, batch, gpu=0,
                         num_class=MODEL_CONFIGURATION.num_class,
                         image_shape=IMAGE_SHAPE):
    scores_list = []
    for batch_data in batch:
        # process data
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, num_class,
                                 segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(DATASET_CONFIGURATION.imgSizes)

            # scores = scores.cpu()
            scores_list.append(scores)
    return torch.cat([transforms.Resize(image_shape)(x) for x in scores_list])


def visualize_result(data, preds, results_directory):
    from scipy.io import loadmat
    import csv
    colors = loadmat('data/color150.mat')['colors']
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

    (imgs, infos) = data

    # print predictions in descending order
    for img, info, pred in zip(imgs, infos, preds):
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        # print("Predictions in [{}]:".format(info))
        for idx in np.argsort(counts)[::-1]:
            name = names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            # if ratio > 0.1:
            #     print("  {}: {:.2f}%".format(name, ratio))

        # colorize prediction
        pred_color = colorEncode(pred, colors).astype(np.uint8)

        # aggregate images and save
        im_vis = np.concatenate((img.permute(1, 2, 0), pred_color), axis=1)

        img_name = info.split('/')[-1]
        Image.fromarray(im_vis).save(
            os.path.join(results_directory, img_name.replace('.jpg', '.png')))


def test_segmentation_map_on_batch():
    test_dataloader = get_data_loader(
        '../data/Image_Harmonization_Dataset/Hday2night/composite_images/',
        '../data/Image_Harmonization_Dataset/Hday2night/Hday2night_test.txt',
    batch_size=32)
    segmentation_module = get_segmentation_module().cuda()
    segmentation_module.eval()
    batch = next(iter(test_dataloader))
    import ipdb; ipdb.set_trace()
    scores = get_segmentation_map(segmentation_module, batch).cpu()
    _, pred = torch.max(scores, dim=1)
    # visualization
    print(f"pred.shape: {pred.shape}")
    data = ([transforms.Resize(IMAGE_SHAPE)(
        torch.from_numpy(b['img_ori']).permute(2, 0, 1)) for b in batch],
            [b['info'] for b in batch])
    results_directory = os.path.join('examples', 'Hday2night', 'composite_images')
    os.makedirs(results_directory, exist_ok=True)
    visualize_result(data, pred, results_directory)
    print(f"scores.shape: {scores.shape}")


if __name__ == '__main__':
    test_segmentation_map_on_batch()
