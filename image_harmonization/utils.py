import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, epoch, filename, optimizer=None, save_arch=False, params=None):
    attributes = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    network_filename = os.path.join(filename, 'network.pth')

    if optimizer is not None:
        attributes['optimizer'] = optimizer.state_dict()

    if save_arch:
        attributes['arch'] = model

    if params is not None:
        attributes['params'] = params

    try:
        torch.save(attributes, network_filename)
        if epoch % 10 == 0:
            filename = os.path.join(filename, f'network_{epoch:03d}.pth')
            torch.save(attributes, filename)
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, network_filename)


def save_reconstructions(image, mask, target, reconstructed_image, name,
                         epoch, reconstruction_save_dir):
    root = os.path.join(reconstruction_save_dir, f'{epoch:03d}')
    os.makedirs(root, exist_ok=True)
    for _image, _mask, _target, _reconstructed_image, _name in zip(image, mask, target, reconstructed_image, name):
        plt.clf()
        plt.subplot(1, 5, 1)
        plt.title('mask')
        plt.imshow(_mask.cpu().detach().permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 5, 2)
        plt.title('composite')
        plt.imshow(_image.cpu().detach().permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 5, 3)
        plt.title('real image')
        plt.imshow(_target.cpu().detach().permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 5, 4)
        plt.title('reconstruction')
        plt.imshow(_reconstructed_image.cpu().detach().permute(1,2,0))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 5, 5)
        plt.title('L1 diff image')
        plt.imshow(torch.abs(
            _mask.cpu().detach() -
            _reconstructed_image.cpu().detach()
        ).squeeze(0), cmap='jet') ; plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.gcf().set_size_inches(15, 4)
        plt.tight_layout()
        figure_path = os.path.join(root, _name)
        plt.savefig(figure_path)






def load_checkpoint(path, model=None, optimizer=None):
    resume = torch.load(path)

    if model is not None:
        if ('module' in list(resume['state_dict'].keys())[0]) \
                and not (isinstance(model, torch.nn.DataParallel)):
            new_state_dict = OrderedDict()
            for k, v in resume['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v  # remove DataParallel wrapping

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(resume['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])

    return model, optimizer

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp