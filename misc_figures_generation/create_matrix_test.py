import os
import shutil


datasets = ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']
dataset_to_path = lambda dataset: f'config/{dataset}-resnet50dilated-ppm_deepsup.yaml'
dataset_to_dummy_path = lambda trained_on_dataset, tested_on_dataset: f'dummy_config/trained-on-{trained_on_dataset}-tested-on-{tested_on_dataset}-resnet50dilated-ppm_deepsup.yaml'


for trained_on in datasets:
    for tested_on in datasets:
        shutil.copy(src=dataset_to_path(trained_on), dst=dataset_to_dummy_path(trained_on, tested_on))

for trained_on in datasets:
    for tested_on in datasets:
        path = f'dummy_config/trained-on-{trained_on}-tested-on-{tested_on}-resnet50dilated-ppm_deepsup.yaml'
        with open(path, 'r') as f:
            data = f.read()
        lines = data.splitlines()
        with open(path, 'w') as f:
            f.write('\n'.join([l if 'list_val' not in l else l.replace(trained_on, tested_on) for l in lines]))


def dataset_to_run(trained_on, tested_on):
        line_to_run = f"python3 eval_multipro.py --gpus 0 --cfg " \
                      f"dummy_config/" \
                      f"trained-on-{trained_on}-tested-on-{tested_on}-" \
                      f"resnet50dilated-ppm_deepsup.yaml > " \
                      f"train_on_one_dataset_test_on_another/trained-on-" \
                      f"{trained_on}-tested-on-{tested_on}.log"
        return line_to_run

to_run = []
datasets.reverse()
for trained_on in datasets:
    for tested_on in datasets:
        to_run.append(dataset_to_run(trained_on, tested_on))

print(' ; '.join(to_run))

