import os
from PIL import Image


root = '../data/Image_Harmonization_Dataset/'
dataset_name = 'Hday2night'
dataset_name = 'HFlickr'
dataset_name = 'HAdobe5k'
dataset_name = 'HCOCO'
# mode = 'train'
for mode in ['train', 'test']:
    path = os.path.join('../data/Image_Harmonization_Dataset/', dataset_name,
                        f'{dataset_name}_{mode}.txt')

    keys = ["fpath_img", "fpath_segm", "width", "height"]
    with open(path, 'r') as f:
        composite_images = f.read().split('\n')[:-1]

    odgt_lines = []
    for composite_image in composite_images:
        mask_name = "_".join(composite_image.split('_')[:-1]) + '.png'
        fpath_img = os.path.join(dataset_name, 'composite_images', composite_image)
        fpath_segm = os.path.join(dataset_name, 'masks', mask_name)
        width, height = Image.open(os.path.join(root, fpath_img)).size
        line = "{" + '"fpath_img": "' + fpath_img + '", "fpath_segm": "' + fpath_segm + '", "width": '+ str(width) + ', "height": ' + str(height) + '}'
        odgt_lines.append(line)

    file_name = f'{dataset_name}-training.odgt' if mode == 'train' else f'{dataset_name}-validation.odgt'
    with open(os.path.join('data', file_name), 'w') as f:
        f.write('\n'.join(odgt_lines))

