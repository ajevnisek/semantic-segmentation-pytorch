import os
import random

import torch
import torchvision
import torch.nn as nn
from torch.nn import init

import numpy as np
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm
from tensorboardX import SummaryWriter

import image_harmonization.utils as utils
import image_harmonization.model as model
from image_harmonization.config import train_parsers_mask_generator
from image_harmonization.dataset import HarmDataSetWithSegmentation
from image_harmonization.test_utils import iterate_over_dataset_and_get_metrics
from segmentation_map_for_image_harmonization import get_segmentation_map, \
    get_segmentation_module


args = train_parsers_mask_generator()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize path
model_save_dir = os.path.join('image_harmonization_results',
                              'checkpoints',
                              'generating_mask',
                              f"{args.model_save_dir}_{args.dataset_name}")
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

tensorboard_log_path = os.path.join('image_harmonization_results',
                                    'tensorboard_logs',
                                    'generating_mask',
                                    args.dataset_name)
os.makedirs(tensorboard_log_path, exist_ok=True)
writer = SummaryWriter(tensorboard_log_path)


reconstruction_save_dir = os.path.join('image_harmonization_results',
                                       'reconstructions',
                                       'generating_mask',
                                       f'reconstructions_{args.dataset_name}')
if not os.path.exists(reconstruction_save_dir):
    os.makedirs(reconstruction_save_dir)

if args.debug_mode:
    network = model.MaskGeneratingNetwork(input_dim=3 + 1).to(device)
else:
    network = model.MaskGeneratingNetwork(input_dim=3 + 150).to(device)
criterion = {'MSE': nn.MSELoss(),
             'L1': nn.L1Loss(),
             'BCE': nn.CrossEntropyLoss(),
             'NLL': nn.NLLLoss(ignore_index=-1),
             }[args.loss]

adadelta_optimizer = torch.optim.Adadelta(network.parameters())
optimizer = {'Adadelta': adadelta_optimizer,
             'Adam': torch.optim.Adam(network.parameters(), lr=3e-4)}[args.optimizer]
# restart the training process
if args.resume is True:
    network, optimizer = utils.load_checkpoint(model_save_dir, network, optimizer=optimizer)

dst = HarmDataSetWithSegmentation(args.img_path, args.list_path, args.mask_path, args.target_path)

def user_scattered_collate(batch):
    return batch


trainloader = data.DataLoader(dst, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_workers, pin_memory=True,
                              drop_last=True, collate_fn=user_scattered_collate)
test_dst = HarmDataSetWithSegmentation(args.img_path, args.test_list_path,
                                  args.mask_path, args.target_path,
                                  is_test=True)
test_dst.files = test_dst.files[::20]
test_dst.support_trimmed_files()
testloader = data.DataLoader(test_dst, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.n_workers,
                             pin_memory=True, collate_fn=user_scattered_collate)
dst_train_to_show = HarmDataSetWithSegmentation(args.img_path, args.list_path,
                                                args.mask_path, args.target_path,
                                                is_test=True)

dst_train_to_show.files = dst_train_to_show.files[::50]
dst_train_to_show.support_trimmed_files()
to_show_loader = data.DataLoader(dst_train_to_show, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.n_workers,
                                 pin_memory=True,
                                 collate_fn=user_scattered_collate)


batch_size = args.batch_size

segmentation_module = get_segmentation_module().cuda()
segmentation_module.eval()


for epoch in tqdm(range(args.epoch_size)):
    # to track the training loss as the model trains
    train_losses = \
        utils.AverageMeter()
    network.train()
    for batch_index, batch in enumerate(trainloader):
        image = torch.cat([b['composite_image'].unsqueeze(0)
                          for b in batch]).to(device)
        mask = torch.cat([b['mask'].unsqueeze(0)
                          for b in batch]).to(device)
        target = torch.cat([b['real_image'].unsqueeze(0)
                            for b in batch]).to(device)
        trimmed_batch = [
            {'img_ori': item['img_ori'],
             'img_data': item['img_data'],
             'info': item['info'],
             }
            for item in batch
        ]
        with torch.no_grad():
            if args.debug_mode:
                segmentation_map = mask
            else:
                segmentation_map = get_segmentation_map(segmentation_module,
                                                        trimmed_batch)
                # segmentation_map = torch.cat([sm.unsqueeze(0)
                #                               for sm in segmentation_map])
        input_tensor = torch.cat([image, segmentation_map], axis=1)
        output = network(input_tensor)
        loss = criterion(mask, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), output.size(0))
    train_loss = train_losses.avg
    writer.add_scalar('train/loss', train_loss, epoch)

    test_losses = utils.AverageMeter()
    test_l1_meter = utils.AverageMeter()
    test_mse_meter = utils.AverageMeter()
    test_bce_meter = utils.AverageMeter()

    network.eval()
    for batch_index, batch in enumerate(testloader):
        image = torch.cat([b['composite_image'].unsqueeze(0)
                          for b in batch]).to(device)
        mask = torch.cat([b['mask'].unsqueeze(0)
                          for b in batch]).to(device)
        target = torch.cat([b['real_image'].unsqueeze(0)
                            for b in batch]).to(device)
        trimmed_batch = [
            {'img_ori': item['img_ori'],
             'img_data': item['img_data'],
             'info': item['info'],
             }
            for item in batch
        ]
        with torch.no_grad():
            if args.debug_mode:
                segmentation_map = mask
            else:
                segmentation_map = get_segmentation_map(segmentation_module,
                                                        trimmed_batch)
            input_tensor = torch.cat([image, segmentation_map], axis=1)
            output = network(input_tensor)

        loss = criterion(mask, output)

        train_losses.update(loss.item(), output.size(0))
        mse = nn.MSELoss()(output, mask).item()
        l1 = nn.L1Loss()(output, mask).item()
        bce = nn.CrossEntropyLoss()(mask, output).item()
        test_l1_meter.update(l1, output.size(0))
        test_mse_meter.update(mse, output.size(0))
        test_bce_meter.update(bce, output.size(0))

    writer.add_scalar('test/MSE', test_mse_meter.avg, epoch)
    writer.add_scalar('test/L1', test_l1_meter.avg, epoch)
    writer.add_scalar('test/CrossEntropy', test_bce_meter.avg, epoch)

    batch = next(iter(testloader))
    image = torch.cat([b['composite_image'].unsqueeze(0)
                       for b in batch]).to(device)
    mask = torch.cat([b['mask'].unsqueeze(0)
                      for b in batch]).to(device)
    target = torch.cat([b['real_image'].unsqueeze(0)
                        for b in batch]).to(device)
    trimmed_batch = [
        {'img_ori': item['img_ori'],
         'img_data': item['img_data'],
         'info': item['info'],
         }
        for item in batch
    ]
    with torch.no_grad():
        if args.debug_mode:
            segmentation_map = mask
        else:
            segmentation_map = get_segmentation_map(segmentation_module,
                                                    trimmed_batch)
        input_tensor = torch.cat([image, segmentation_map], axis=1)
        reconstructed_image = network(input_tensor)
    for i in range(image.shape[0]):
        writer.add_images(
            f'test/Composite_Mask_Target_Reconstruction_{i}',
            torch.cat([
                image[i].unsqueeze(0),
                torch.tile(mask[i], (3, 1, 1)).unsqueeze(0),
                target[i].unsqueeze(0),
                torch.tile(reconstructed_image[i], (3, 1, 1)).unsqueeze(0)
            ]),
            epoch)
    batch = [dst.get_by_name('d5381-20110611-034022_1_1.jpg')]
    image = torch.cat([b['composite_image'].unsqueeze(0)
                       for b in batch]).to(device)
    mask = torch.cat([b['mask'].unsqueeze(0)
                      for b in batch]).to(device)
    target = torch.cat([b['real_image'].unsqueeze(0)
                        for b in batch]).to(device)
    trimmed_batch = [
        {'img_ori': item['img_ori'],
         'img_data': item['img_data'],
         'info': item['info'],
         }
        for item in batch
    ]
    with torch.no_grad():
        if args.debug_mode:
            segmentation_map = mask
        else:
            segmentation_map = get_segmentation_map(segmentation_module,
                                                    trimmed_batch)
        input_tensor = torch.cat([image, segmentation_map], axis=1)
        reconstructed_image = network(input_tensor)
    writer.add_images(
        f'train/Composite_Mask_Target_Reconstruction_{0}',
        torch.cat([
            image[0].unsqueeze(0),
            torch.tile(mask[0], (3, 1, 1)).unsqueeze(0),
            target[0].unsqueeze(0),
            torch.tile(reconstructed_image[0], (3, 1, 1)).unsqueeze(0)
        ]),
        epoch)

    batch = next(iter(to_show_loader))
    image = torch.cat([b['composite_image'].unsqueeze(0)
                       for b in batch]).to(device)
    mask = torch.cat([b['mask'].unsqueeze(0)
                      for b in batch]).to(device)
    target = torch.cat([b['real_image'].unsqueeze(0)
                        for b in batch]).to(device)
    trimmed_batch = [
        {'img_ori': item['img_ori'],
         'img_data': item['img_data'],
         'info': item['info'],
         }
        for item in batch
    ]
    with torch.no_grad():
        if args.debug_mode:
            segmentation_map = mask
        else:
            segmentation_map = get_segmentation_map(segmentation_module,
                                                    trimmed_batch)
        input_tensor = torch.cat([image, segmentation_map], axis=1)
        reconstructed_image = network(input_tensor)
    utils.save_reconstructions(image, mask, target,
                               torch.clamp(reconstructed_image, 0, 1),
                               [b['name'] for b in batch],
                               epoch, reconstruction_save_dir)
    utils.save_checkpoint(network, epoch, model_save_dir, optimizer)
    # scheduler.step()
