import argparse

def test_parsers():
    parser = argparse.ArgumentParser("test code for image harmonization")
    # Basic options
    parser.add_argument('--img_path', type=str, default='Path_to_images',
                        help='Directory path to a batch of images')
    parser.add_argument('--mask_path', type=str, default='Path_to_masks',
                        help='Directory path to a batch of mask')
    parser.add_argument('--test_list_path', type=str, default='Path_to_test_list',
                        help='Directory path to test list')
    parser.add_argument('--target_path', type=str, default='Path_to_target',
                        help='Directory to a batch of target')
    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model', type=str, default='./checkpoint/network.pth')
    parser.add_argument('--alpha_adain', type=float, default=1.0,
                        help='adain parameter')
    parser.add_argument('--metrics_dir', type=str,
                        default='samples/HCOCO_sample')
    args = parser.parse_args()
    return args

def train_parsers():
    parser = argparse.ArgumentParser("Pytorch training code for image harmonization")
    # Basic options
    parser.add_argument('--img_path', type=str, default='Path_to_images',
                        help='Directory path to images')
    parser.add_argument('--dataset_name', type=str, default='',
                        help='dataset name')
    parser.add_argument('--mask_path', type=str, default='Path_to_masks',
                        help='Directory path to mask')
    parser.add_argument('--list_path', type=str, default='Path_to_txt',
                        help='path to train images list')
    parser.add_argument('--test_list_path', type=str, default='Path_to_txt',
                        help='path to test images list')
    parser.add_argument('--target_path', type=str, default='Path_to_target',
                        help='Directory to target')
    # training options
    parser.add_argument('--alpha_adain', type=float, default=1.0,
                        help='out = adain_alpha * transfered_content + ('
                             '1 - adain_alpha) * old_content')
    parser.add_argument('--loss', type=str, default='MSE',
                        choices=['MSE', 'L1', 'MaskWeightedMSE',
                                 'MaskWeightedL1'],
                        help='which loss to use')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        choices=['Adadelta', 'Adam'],
                        help='which optimizer to use')

    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model_save_dir', default='./checkpoint',
                        help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epoch_size', type=int, default=230,
                        help="training epoch size")
    parser.add_argument('--n_workers', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for training")
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training or not')
    # model parameters

    # cosine_similarity=False, softmax=False, receptive_field_factor
    parser.add_argument('--receptive_field_factor', type=int,
                        default=4,
                        choices=[2, 4, 8, 16],
                        help='Each feature corresponds to RFFxRFF pixels')
    parser.add_argument('--feature_similarity', type=str,
                        default='cosine',
                        choices=['cosine', 'dot-product'],
                        help='If set, use cosine similarity, otherwise: use '
                             'dot product')
    parser.add_argument('--dic_normalization', type=str,
                        default='softmax',
                        choices=['softmax', 'max', 'mean'],
                        help='Scale attention map with softmax or scale by '
                             'max.')
    args = parser.parse_args()
    return args


def train_parsers_mask_generator():
    parser = argparse.ArgumentParser("Pytorch training code for image harmonization")
    parser.add_argument('--debug_mode', action='store_true', default=False,
                        help='Is in debug mode.')
    # Basic options
    parser.add_argument('--img_path', type=str, default='Path_to_images',
                        help='Directory path to images')
    parser.add_argument('--dataset_name', type=str, default='',
                        help='dataset name')
    parser.add_argument('--mask_path', type=str, default='Path_to_masks',
                        help='Directory path to mask')
    parser.add_argument('--list_path', type=str, default='Path_to_txt',
                        help='path to train images list')
    parser.add_argument('--test_list_path', type=str, default='Path_to_txt',
                        help='path to test images list')
    parser.add_argument('--target_path', type=str, default='Path_to_target',
                        help='Directory to target')
    # training options
    parser.add_argument('--loss', type=str, default='MSE',
                        choices=['MSE', 'L1', 'BCE', 'NLL'],
                        help='which loss to use')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        choices=['Adadelta', 'Adam'],
                        help='which optimizer to use')

    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model_save_dir', default='./checkpoint',
                        help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epoch_size', type=int, default=230,
                        help="training epoch size")
    parser.add_argument('--n_workers', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for training")
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training or not')
    # model parameters
    args = parser.parse_args()
    return args