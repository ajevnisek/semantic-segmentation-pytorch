import os


datasets = ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']


def get_file_name(trained_on, tested_on):
    return os.path.join('train_on_one_dataset_test_on_another',
                        f"trained-on-{trained_on}-tested-on-{tested_on}.log")


def extract_data(path):
    """Data with structure:
class [0], IoU: 0.9330
class [1], IoU: 0.1165
[Eval Summary]:
Mean IoU: 0.5247, Accuracy: 93.36%
"""
    with open(path, 'r') as f:
        raw_data = f.read()
    lines = raw_data.split('\n')
    class_0_iou = float([l for l in lines if l.startswith('class [0]')][
                            0].split('IoU: ')[-1])
    class_1_iou = float([l for l in lines if l.startswith('class [1]')][
                            0].split('IoU: ')[-1])
    mean_iou = float([l for l in lines if l.startswith('Mean IoU: ')][
                            0].split('IoU: ')[-1].split(',')[0])
    accuracy = float([l for l in lines if l.startswith('Mean IoU: ')][
                         0].split('Accuracy: ')[-1].split('%')[0])
    return {'class_0_iou': class_0_iou, 'class_1_iou': class_1_iou,
            'mean_iou': mean_iou, 'accuracy': accuracy}


accuracy = {}
class_1_iou = {}
for trained_on in datasets:
    accuracy[trained_on] = {}
    class_1_iou[trained_on] = {}
    for tested_on in datasets:
        path = get_file_name(trained_on, tested_on)
        data_dict = extract_data(path)
        accuracy[trained_on][tested_on] = data_dict['accuracy']
        class_1_iou[trained_on][tested_on] = data_dict['class_1_iou']


mask_from_test_image = lambda x: '_'.join(x.split('_')[:-1]) + '.png'
for dataset in datasets:
    path = os.path.join('../data/Image_Harmonization_Dataset/', dataset, f'{dataset}_test.txt')
    with open(path, 'r') as f:
        test_images = f.read()
    test_images = test_images.split('\n')
    test_images = test_images[:-1]
    masks_paths = [os.path.join('../data/Image_Harmonization_Dataset/',dataset,'masks', mask_from_test_image(x)) for x in test_images]
    white_pixels = 0
    black_pixels = 0
    for mask_path in masks_paths:
        image = Image.open(mask_path).convert('L')
        white_pixels += (np.array(image) == 0).sum()
        black_pixels += (np.array(image) != 0).sum()
    print(f"{dataset}: {white_pixels/(white_pixels + black_pixels) * 100.0:.2f}% white pixels")
