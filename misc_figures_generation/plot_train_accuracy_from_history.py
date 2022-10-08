import os
import torch
import matplotlib.pyplot as plt


model = 'Encoder: Resnet50dilated, \nDecoder: PSPNet+Deep Supervision Trick'

for data_term in ['acc', 'loss']:
    for dataset in ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']:
        plt.clf()
        history_path = os.path.join('ckpt',
                                    f'{dataset}-resnet50dilated-ppm_deepsup',
                                    'history_epoch_20.pth')
        d = torch.load(history_path)
        epochs = range(20)
        plt.plot(d['train']['epoch'], [x * 100.0 for x in d['train'][data_term]])
        title = 'Train per-pixel-accuracy' if data_term == 'acc' else \
            'Train loss'
        plt.title(f'{title}\n Dataset: {dataset}\n Model: {model}')
        plt.grid(True)
        plt.xlabel('epoch')
        ylabel = 'accuracy [%]' if data_term == 'acc' else 'loss'
        plt.ylabel(ylabel)
        plt.plot([e + 1 for e in epochs], [
            sum([x for x, y in zip(d['train'][data_term], d['train']['epoch']) if
                 y < epoch + 1 and y >= epoch]) / 5000 * 20 * 100.0
            for epoch in epochs], '--x')
        plt.legend(['raw', 'smoothed'])
        plt.tight_layout()
        name = f'{dataset}-train-accuracy.png' if data_term == 'acc' else\
            f'{dataset}-train-loss.png'
        plt.savefig(os.path.join('misc_figures', name))


accuracy_per_dataset = {'train': [], 'test': []}
iou_per_dataset = {'class 0 mean-iou': [],
                   'class 1 mean-iou': [],
                   'mean-iou': [],}
class_0_iou = {'test': []}
class_1_iou = {'test': []}
datasets = ['HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']
for dataset in datasets:
    epoch_to_accuracy = [
                sum([x
                     for x, y in zip(d['train']['acc'], d['train']['epoch'])
                     if y < epoch + 1 and y >= epoch
                     ]) / 5000 * 20 * 100.0
                for epoch in epochs]
    final_train_accuracy = epoch_to_accuracy[-1]
    accuracy_per_dataset['train'].append(final_train_accuracy)
    path = f'ckpt/{dataset}-resnet50dilated-ppm_deepsup/eval_log.txt'
    with open(path, 'r') as f:
        data = f.read()
    interesting_line = [line for line in data.split('\n')
                        if line.startswith('Mean IoU:')][0]
    mean_iou = float(interesting_line.split(',')[0].split(' ')[-1])
    iou_per_dataset['mean-iou'].append(mean_iou)
    test_accuracy = float(
        interesting_line.split(',')[-1].split(' ')[-1].split('%')[0])
    accuracy_per_dataset['test'].append(test_accuracy)
    # class [0], IoU: 0.9099
    interesting_line = [line for line in data.split('\n')
                        if line.startswith('class [0], IoU: ')][0]
    current_class_0_iou = float(interesting_line.split('class [0], IoU: ')[-1])
    iou_per_dataset['class 0 mean-iou'].append(current_class_0_iou)
    interesting_line = [line for line in data.split('\n')
                        if line.startswith('class [1], IoU: ')][0]
    current_class_1_iou = float(interesting_line.split('class [1], IoU: ')[-1])
    iou_per_dataset['class 1 mean-iou'].append(current_class_1_iou)

df = pd.DataFrame(accuracy_per_dataset, index=datasets)
ax = df.plot.bar(rot=0)
plt.ylabel('accuracy [%]')
plt.title('train and test accuracies [%]')
plt.grid(True)
fig = plt.gcf()
ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
                bbox_transform=fig.transFigure, ncol=3)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.ylim([90, 100.5])
plt.yticks(range(90, 100+1))
plt.savefig(os.path.join('misc_figures', 'train-and-test-accuracies.png'))

df = pd.DataFrame(iou_per_dataset, index=datasets)
ax = df.plot.bar(rot=0)
plt.ylabel('IoU')
plt.title('IoU for class 0, class 1 (object to harmonize) and Mean-IoU')
plt.grid(True)
fig = plt.gcf()
ax.legend(bbox_to_anchor=(1, 0), loc="lower right",
          bbox_transform=fig.transFigure, ncol=3)
plt.yticks([x / 10.0 for x in range(0, 10 + 1)])

plt.tight_layout()
plt.savefig(os.path.join('misc_figures', 'test-mean-iou.png'))
